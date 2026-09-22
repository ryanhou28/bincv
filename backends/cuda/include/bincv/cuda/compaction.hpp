#pragma once

/// @file compaction.hpp
/// @brief The capacity contract for device-side compaction: what a detection
/// kernel does when it finds more than the caller's buffer holds, and how the
/// caller finds out.
///
/// ---------------------------------------------------------------------------
/// WHY THIS IS SHARED RATHER THAN PER-FAMILY
///
/// A detector does not know its own output size. On the host that is a counted
/// loop against a bound -- `detectFast(img, out, capacity, &truncated)` fills,
/// stops, and sets a flag. On the device it is thousands of threads appending
/// through one atomic, and the count passes capacity mid-launch on whichever
/// thread happens to be there. Every family that emits a variable number of
/// things -- FAST, good-features, the descriptor gate, stereo -- needs the same
/// three pieces: a bounds-checked append, an unclamped counter, and a read-back
/// that says whether the answer is whole. Written twelve times it is written
/// twelve ways, and the way that gets it wrong still returns plausible corners.
///
/// ---------------------------------------------------------------------------
/// THE CONTRACT
///
/// **The kernel truncates, the counter keeps counting, and the caller cannot
/// read a number out of the result without saying which number they meant.**
///
/// * **TRUNCATE, DO NOT FAIL.** A caller who caps their buffer on purpose --
///   the pipeline that wants at most 500 corners and sized for it -- is doing
///   the normal thing, and an error return would make the normal thing an
///   exceptional path. The elements written are real detections, not garbage,
///   so throwing the launch away throws away work that is already correct. The
///   host makes the same choice in `detectFast` and in `selectGoodFeatures`.
///
/// * **THE COUNTER IS NOT CLAMPED.** Every candidate increments it, the ones
///   that did not fit included, so `found()` is the TRUE count -- and it is
///   exactly the capacity a complete re-run needs. Clamping at capacity would
///   report `capacity` for every overflow and leave the caller unable to size
///   the retry, which is the one thing an overflowed caller needs to know.
///
/// * **THE COUNT CANNOT BE READ NEUTRALLY.** `DeviceAppendResult` has no
///   `count()`, no `size()` and no implicit conversion. A caller either asks
///   for the whole answer -- `completeCount`, which REFUSES when the run
///   overflowed and is `[[nodiscard]]` so ignoring the refusal does not
///   compile under this project's warnings -- or accepts a partial one through
///   `acceptTruncated`, whose name is then sitting at the call site for a
///   reader to see. The host's `bool* truncated = nullptr` defaults to "do not
///   tell me"; that default is the shape this type exists to not have.
///
/// * **DOWNLOAD GOES THROUGH THE VERDICT.** `downloadAppended` takes the
///   `DeviceAppendResult`, so there is no path from a device append buffer to a
///   host array that skips the object holding the truncation answer.
///
/// ---------------------------------------------------------------------------
/// WHICH ELEMENTS SURVIVE A TRUNCATED RUN IS NOT SPECIFIED
///
/// The atomic decides, so two runs of the same kernel over the same frame may
/// keep different candidates. The host truncates in raster order. A truncated
/// device result is therefore **not a prefix of the host's** and must not be
/// compared against one element by element -- compare `found()`, or give both
/// runs a capacity that holds every candidate. Bit-exactness against the host
/// is a claim about COMPLETE runs, and a suite that forgets this fails
/// intermittently, which is worse than failing.
///
/// A family that genuinely needs the host's truncation ORDER cannot get it from
/// an atomic and must compact by prefix sum instead; `reserve` and `store`
/// below are the pieces that build on, and the contract above is unchanged.
///
/// ---------------------------------------------------------------------------
/// DOMAIN
///
/// Counts and capacities are `uint32_t`. A 32-bit atomic is the device's fast
/// one, and a candidate set this cannot count does not arise from an image
/// this backend can address. The factories assert the bound; a launcher that
/// takes a host `size_t` count must reject it outside the bound rather than
/// narrowing an index into silence.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core.hpp"
#include "deviceBinMat.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief What a compaction produced, and whether that is the whole of it.
/// @note Deliberately not an aggregate of two public integers: the whole point
/// is that the two numbers cannot be confused for each other, and a bare
/// `.count` field is how they get confused.
class DeviceAppendResult {
public:
    DeviceAppendResult() = default;

    /// @param found The counter's final value -- the TRUE candidate count,
    /// which may exceed `capacity`.
    /// @param capacity The buffer the run was given.
    BINCV_CUDA_HD DeviceAppendResult(uint32_t found, uint32_t capacity)
        : found_(found), capacity_(capacity) {}

    /// @brief The true number of candidates the kernel produced -- **not** the
    /// number stored. Size a re-run to this.
    BINCV_CUDA_HD uint32_t found() const { return found_; }

    /// @brief The capacity the run was given.
    BINCV_CUDA_HD uint32_t capacity() const { return capacity_; }

    /// @brief True when candidates were counted but not stored.
    BINCV_CUDA_HD bool truncated() const { return found_ > capacity_; }

    /// @brief The count, **only when it is the whole count**.
    /// @param count Written with the candidate count on success; left untouched
    /// on a truncated run, so a caller who ignores the return value does
    /// not silently get a plausible number.
    /// @return false when the run overflowed the buffer.
    /// @note `[[nodiscard]]` on purpose, and it is the only one in the library:
    /// a discarded return here is precisely the mistake this file exists to
    /// prevent, and this project compiles with `-Werror` in its gates.
    [[nodiscard]] BINCV_CUDA_HD bool completeCount(uint32_t& count) const {
        if (truncated()) return false;
        count = found_;
        return true;
    }

    /// @brief `min(found, capacity)` -- the elements actually in the buffer.
    /// @note Named for what taking it MEANS. Reading this is accepting a
    /// possibly partial answer, which is a legitimate thing to do and an
    /// illegitimate thing to do by accident; the name puts the decision in
    /// the caller's source rather than in this file's documentation.
    BINCV_CUDA_HD uint32_t acceptTruncated() const {
        return found_ < capacity_ ? found_ : capacity_;
    }

private:
    uint32_t found_ = 0;
    uint32_t capacity_ = 0;
};

/// @brief Non-owning view of an append target: the caller's buffer, its
/// capacity, and the device counter the kernel advances.
/// @tparam T The element the family emits -- a POD from features.hpp.
/// @note A view, so it owns nothing (CLAUDE.md): the buffer is the caller's and
/// nothing here allocates. The three fields travel together because
/// separating them is how a kernel ends up with a pointer and no bound.
template <typename T>
struct DeviceAppendBufferView {
    T* out = nullptr;           ///< first element, in device memory
    uint32_t* counter = nullptr;///< ONE uint32 in device memory, zeroed before the launch
    uint32_t capacity = 0;      ///< elements `out` holds

    DeviceAppendBufferView() = default;
    BINCV_CUDA_HD DeviceAppendBufferView(T* out_, uint32_t* counter_, uint32_t capacity_)
        : out(out_), counter(counter_), capacity(capacity_) {}

    BINCV_CUDA_HD bool empty() const { return out == nullptr || capacity == 0; }

#ifdef __CUDACC__
    /// @brief Appends one element. **Device code only.**
    /// @return true if it was stored; false if the buffer was already full -- in
    /// which case it was still COUNTED, which is what makes the true total
    /// recoverable. Most kernels can ignore the return value; the counter
    /// carries the outcome either way.
    __device__ bool append(const T& value) const {
        const uint32_t slot = atomicAdd(counter, 1u);
        if (slot >= capacity) return false;
        out[slot] = value;
        return true;
    }

    /// @brief Reserves `n` consecutive slots for the calling thread.
    /// @return The first reserved slot. It may be at or past `capacity` (the
    /// whole run rejected) or may straddle it -- `store` is what decides
    /// per element, so a straddling reservation stores its head and drops
    /// its tail without a second bound test in the caller.
    /// @note Here so warp-aggregated appending -- one atomic per warp instead
    /// of one per element, which is the standard fix when a detector's
    /// hit rate is high -- is written once rather than by each family.
    /// The counter advances by the full `n`, rejected elements included,
    /// so `found()` stays the true count.
    __device__ uint32_t reserve(uint32_t n) const { return atomicAdd(counter, n); }

    /// @brief Stores at an already-reserved slot, if it fits. **Device only.**
    __device__ bool store(uint32_t slot, const T& value) const {
        if (slot >= capacity) return false;
        out[slot] = value;
        return true;
    }
#endif
};

/// @brief The single device `uint32_t` an append buffer counts through.
/// @note An owning container, not a view -- allocation lives in containers here
/// exactly as it does on the host. Four bytes is not worth a caller's own
/// `cudaMalloc` in twelve places, and `reset` being a member is what makes
/// "zero it before every launch" a thing a caller can see they did.
class DeviceAppendCounter {
public:
    DeviceAppendCounter() : word_(1) {}

    uint32_t* devicePtr() { return word_.data(); }
    const uint32_t* devicePtr() const { return word_.data(); }

    /// @brief Zeroes the counter. **Every launch starts here**: a counter still
    /// holding the previous frame's total appends past the buffer's start,
    /// and reports a count that is the sum of two frames.
    cudaError_t reset(cudaStream_t stream = nullptr) {
        return cudaMemsetAsync(word_.data(), 0, sizeof(uint32_t), stream);
    }

private:
    DeviceArray<uint32_t> word_;
};

/// @brief Names an append target over an owned array and a counter.
/// @note Asserts the `uint32_t` count domain rather than narrowing quietly.
template <typename T>
inline DeviceAppendBufferView<T> appendBuffer(DeviceArray<T>& buffer,
                                              DeviceAppendCounter& counter) {
    BINCV_ASSERT(buffer.size() <= 0xFFFFFFFFu,
                 "appendBuffer: capacity outside the device's uint32 count domain");
    return DeviceAppendBufferView<T>{buffer.data(), counter.devicePtr(),
                                     static_cast<uint32_t>(buffer.size())};
}

/// @brief Reads the counter back and pairs it with the capacity it ran against.
/// @return The copy's error code; `result` is written only on success.
/// @note Synchronizes `stream`: the count is a host-side decision (how many
/// elements to download, whether to re-run larger), so it cannot be read
/// before the launch that produces it has finished.
template <typename T>
inline cudaError_t readAppendResult(const DeviceAppendBufferView<T>& buffer,
                                    DeviceAppendResult& result,
                                    cudaStream_t stream = nullptr) {
    BINCV_ASSERT(buffer.counter != nullptr, "readAppendResult: null counter");
    uint32_t found = 0;
    const cudaError_t copyErr = cudaMemcpyAsync(&found, buffer.counter, sizeof(uint32_t),
                                                cudaMemcpyDeviceToHost, stream);
    if (copyErr != cudaSuccess) return copyErr;
    const cudaError_t syncErr = cudaStreamSynchronize(stream);
    if (syncErr != cudaSuccess) return syncErr;
    result = DeviceAppendResult(found, buffer.capacity);
    return cudaSuccess;
}

/// @brief Copies the appended elements to host memory.
/// @param result The verdict from `readAppendResult` for THIS buffer. Taking it
/// is the enforcement: there is no path from an append buffer to a host
/// array that does not pass through the object that knows whether the
/// answer is complete.
/// @param host At least `result.acceptTruncated()` elements.
/// @note Copies the stored elements only. Whether they are the whole answer is
/// `result.truncated()`, and this function deliberately does not decide
/// that for the caller.
template <typename T>
inline cudaError_t downloadAppended(const DeviceAppendBufferView<T>& buffer,
                                    const DeviceAppendResult& result, T* host,
                                    cudaStream_t stream = nullptr) {
    BINCV_ASSERT(result.capacity() == buffer.capacity,
                 "downloadAppended: the result was read from a different buffer");
    const uint32_t n = result.acceptTruncated();
    if (n == 0) return cudaSuccess;
    BINCV_ASSERT(host != nullptr && buffer.out != nullptr,
                 "downloadAppended: a non-empty transfer needs non-null pointers");
    return cudaMemcpyAsync(host, buffer.out, static_cast<size_t>(n) * sizeof(T),
                           cudaMemcpyDeviceToHost, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
