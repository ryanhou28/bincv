#pragma once

/// @file pyramid.hpp
/// @brief The device arm of ops/pyramid.hpp's BOX route, and the resident
/// ladder a device tracker holds across frames.
///
/// pyrDownBox one level: 2x2 box mean, BORDER_REPLICATE, NIn bits in,
/// NOut bits out. Bit-exact against the HOST pyrDownBox.
/// DevicePyramid a ladder of N-bit levels in ONE device allocation.
/// buildPyramidBox fills levels 1..L-1 from level 0, on device, no transfers.
///
/// ---------------------------------------------------------------------------
/// THE RESIDENCY CONTRACT IS THE DESIGN
///
/// A tracker's pyramid is not an output. It is state that lives on the device
/// for the whole session, and the only thing that crosses the bus per frame is
/// LEVEL 0. Two `DevicePyramid` objects are constructed ONCE, outside the frame
/// loop, and ping-ponged with `swap()`, which is an O(1) pointer exchange and
/// copies nothing. Per frame the caller writes level 0 and nothing else --
/// either `upload()` of a host-packed base, or `uploadImage()` of the wide
/// frame followed by the device `packBits`/`packQuant` in cuda/pack.hpp, which
/// is the resident-pipeline path. `buildPyramidBox` then fills 1..L-1 with
/// L-1 launches on one stream: nothing allocated, nothing downloaded, no
/// synchronize. Levels 1..L-1 are never uploaded and never downloaded.
///
/// At 752x480 with the ladder {1, 3, 4, 5} that is 45.0 KB up per frame on the
/// host-packed path (or 352.5 KB on the wide path, which `packBits` then turns
/// into the same 45.0 KB on device), against a ladder that occupies 93.5 KB
/// resident -- and the SAME-SHAPED CV_8U ladder occupies 468.2 KB at a tight
/// pitch. The numbers are computed by `sizeInBytes()`, not estimated; see
/// the benchmark for the comparison and the meter it is read on.
///
/// **THE PING-PONG IS A CALLER CONTRACT, NOT A TYPE INVARIANT.** Nothing here
/// prevents a caller from overwriting level 0 of the pyramid the tracker has
/// not finished reading; the result is silently wrong flow with no diagnostic.
/// Whether a `DevicePyramidPair` that owns both and exposes current()/previous()
/// should make it structural is an open scope question, not an oversight.
///
/// ---------------------------------------------------------------------------
/// WHAT IS AND IS NOT CLAIMED HERE
///
/// At the reference frame size the WHOLE ladder is 93.5 KB, far inside this
/// device's 4 MB L2, and the build is three launches of a few microseconds
/// each. Neither binCV nor a byte-per-pixel arm pays the DRAM traffic the
/// format ratio describes, so **at 752x480 the traffic ratio is a FOOTPRINT
/// claim, not a speed claim.** It becomes a speed argument only where the
/// ladder exceeds L2 -- around 4K, or with many concurrent streams. The
/// benchmark measures both sizes for exactly that reason, and the report says
/// which of the two a number is.
///
/// ---------------------------------------------------------------------------
/// WHAT THIS FILE DOES NOT PROVIDE, AND WHY
///
/// There is no device `pyrDownFiltered` and no device Gaussian5x5. At
/// NIn == NOut == 8 a Gaussian device arm has NO footprint advantage by
/// construction -- eight planes of one bit is one byte, exactly as the host
/// header says -- and it would reconstruct each pixel from eight plane loads
/// against `cv::cuda::pyrDown`'s one byte load. It is predicted to lose on both
/// axes before a line of it is written, which makes it a scope decision rather
/// than a benchmark row. The box route, whose input is narrow, is where the
/// thesis pays.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core.hpp"
#include "deviceBinMat.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

// ---------------------------------------------------------------------------
// Geometry -- the host's four constexpr functions, device-callable
//
// These are TWINS of ops/pyramid.hpp's pyrDownWidth / pyrDownHeight /
// pyrLevelToBase / pyrBaseToLevel, and they exist for the reason core.hpp's
// rowWords twin does: the host originals are plain host-inline code, and a
// kernel cannot call them. The copy is not trusted -- test_cuda_pyramid sweeps
// each one against the host original rather than asserting agreement in prose.
// ---------------------------------------------------------------------------

/// @brief Destination width of one pyramid level: ceil(srcWidth / 2).
/// **API TIER 3**, with the host twin. An odd width keeps its last column;
/// that column's 2x2 block replicates the edge pixel.
BINCV_CUDA_HD constexpr size_t pyrDownWidth(size_t srcWidth) { return (srcWidth + 1) / 2; }

/// @brief Destination height of one pyramid level: ceil(srcHeight / 2).
/// **API TIER 3.**
BINCV_CUDA_HD constexpr size_t pyrDownHeight(size_t srcHeight) { return (srcHeight + 1) / 2; }

/// @brief Where a level-`level` pixel CENTER sits in level-0 coordinates.
/// **API TIER 3.**
/// @note Needed from DEVICE code: a keypoint detected at level L is described
/// and reported at level 0. Every factor is a power of two, so the map and
/// its inverse are exact in float and round-trip without drift -- which the
/// suite asserts rather than assumes.
BINCV_CUDA_HD constexpr float pyrLevelToBase(float c, size_t level) {
    return (c + 0.5f) * static_cast<float>(1u << level) - 0.5f;
}

/// @brief The inverse: a level-0 coordinate in level-`level` pixels.
/// **API TIER 3.**
BINCV_CUDA_HD constexpr float pyrBaseToLevel(float c, size_t level) {
    return (c + 0.5f) / static_cast<float>(1u << level) - 0.5f;
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

/// @brief One pyramid level on the device: 2x2 box mean of `src`, subsampled,
/// requantized to `dst.planes` bits. **API TIER 2** -- `cv::cuda::resize` at
/// INTER_AREA and scale 0.5 computes the same filter, with deliberately
/// different numerics; bit-exact against the HOST `pyrDownBox`, which is
/// where the OpenCV comparison lives (this suite builds without OpenCV).
///
/// @param src An N-bit level, 1 to 8 planes.
/// @param dst An N-bit level, 1 to 8 planes, `pyrDownWidth(src.width)` by
/// `pyrDownHeight(src.height)`. Must share no word with `src`.
/// @param stream The stream to enqueue on.
/// @return `cudaErrorInvalidValue` when the shape contract is violated -- the
/// domain below is NAMED, asserted in debug, and reported in every build.
/// Otherwise the launch's own status.
///
/// @note **THE DOMAIN, NAMED.** `src.planes` and `dst.planes` in [1, 8];
/// `dst` sized exactly `pyrDownWidth` x `pyrDownHeight` of `src`; both
/// views non-null when non-empty; no word shared. An empty destination is a
/// no-op, not an error, as on the host.
/// @note `dst(y, x) = floor((S * (2^NOut - 1) + 2 * (2^NIn - 1)) / (4 * (2^NIn - 1)))`
/// where S is the 2x2 sum at source rows 2y, 2y+1 and columns 2x, 2x+1 --
/// the host expression, digit for digit, which is what makes bit-exactness
/// a statement about the code rather than about the values it happened to
/// be run on.
/// @note Odd widths and heights REPLICATE the edge pixel into the missing half
/// of the block, before the arithmetic, so the divisor stays 4 everywhere
/// and a dirty source padding bit can never reach a live destination pixel.
/// @note Padding bits past `dst.width` are zero on return in every plane. In
/// the reference arm that is by construction -- lanes past the width
/// contribute 0 to every ballot -- and in the fast arm by the trailing
/// mask, as the host stores it.
/// @note Allocates nothing and takes no scratch: the four 2x2 phases are
/// registers on both arms, which is the host's fusion argument unchanged.
cudaError_t pyrDownBox(DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                       cudaStream_t stream = nullptr);

/// @brief Whether the word-parallel fast arm is INSTANTIATED for this
/// (source bits, destination bits) pair.
///
/// @note Public, and deliberately so: this is the device analogue of
/// `simdStatus()`. The fast arm is a templated bit-sliced kernel, so it
/// exists only for a bounded set of (NIn, NOut) pairs -- the pipeline
/// ladder, the identities, and the 8->8 case -- and 64 kernels for the rest
/// is code size for nothing. A caller outside the set silently takes the
/// reference arm, and "silently" is the part this function removes. It is
/// also the benchmark's mandated gate-excluded case: an uncovered pair must
/// read ~1.00x with the switch on and off, or the fast arm is not running
/// where the table says it is.
bool pyrFastArmCovers(size_t nIn, size_t nOut);

namespace impl {

/// @brief Runtime switch for `pyrDownBox`'s word-parallel fast arm.
/// **INTERNAL.** Same contract as `denseFastArmEnabled` / `censusTiledEnabled`:
/// a fast arm that cannot be switched off cannot be shown to be on, and the
/// suite holds both arms to the same output in one binary.
bool& pyrBitSlicedEnabled();

} // namespace impl

// ---------------------------------------------------------------------------
// The ladder
// ---------------------------------------------------------------------------

/// @brief A resident pyramid: one N-bit level per entry of `LevelBits`, level 0
/// first, **all of them in ONE device allocation**.
/// @tparam LevelBits Bits per pixel at each level. `DevicePyramid<1, 3, 4, 5>`
/// is the VIO frontend's ladder.
///
/// @note **API TIER 2** for the container's role (`cv::buildPyramid`). There is
/// no `cv::cuda::buildPyramid` at all: `cv::cuda::SparsePyrLKOpticalFlow`
/// builds one internally and does not expose it, so the honest GPU bar is
/// the per-level `cv::cuda::pyrDown` loop a caller actually writes.
/// @note **ONE ALLOCATION FOR THE WHOLE LADDER**, not one per level. That is a
/// decision, not an accident: it makes the footprint exactly the closed
/// formula with no per-allocation rounding to explain away, makes `swap()`
/// one pointer exchange, and hands the driver one granule instead of L.
/// Every level is a slice of it, at its own offset and its own stride.
/// @note The default row stride is TIGHT, as `DeviceBinMat`'s is. Rounding this
/// ladder's rows to 128 bytes costs 2.65x the memory (253,440 B against
/// 95,760 B at 752x480), and CLAUDE.md says memory wins an unforced
/// conflict.
/// @note This is a CONTAINER. It owns and allocates; the kernels take views and
/// allocate nothing.
template <size_t... LevelBits>
class DevicePyramid {
    static_assert(sizeof...(LevelBits) >= 1, "a device pyramid needs at least one level");

public:
    /// Number of levels, level 0 included.
    static constexpr size_t Levels = sizeof...(LevelBits);

    /// @brief Bits per pixel at level `i`, at compile time.
    template <size_t I>
    static constexpr size_t levelBits() {
        static_assert(I < Levels, "pyramid level index out of range");
        return kBits[I];
    }

    /// @brief An empty pyramid that owns nothing.
    DevicePyramid() = default;

    /// @brief Allocates every level from level 0's extent, zero-filled.
    /// @param width Level 0's width in pixels.
    /// @param height Level 0's height in pixels.
    /// @param rowAlignmentBytes Power of two, multiple of 4. 4 is tight.
    /// @note Zero-filled for the reason `DeviceBinMat` is: the padding bits
    /// must START clear or word-wise reductions over-count.
    DevicePyramid(int width, int height, size_t rowAlignmentBytes = 4) {
        if (width < 0 || height < 0)
            BINCV_THROW(std::invalid_argument, "DevicePyramid: negative dimensions");
        if (rowAlignmentBytes < 4 || (rowAlignmentBytes & (rowAlignmentBytes - 1)) != 0)
            BINCV_THROW(std::invalid_argument,
                        "DevicePyramid: rowAlignmentBytes must be a power of two >= 4");
        layout(static_cast<size_t>(width), static_cast<size_t>(height),
               rowAlignmentBytes / 4);
        allocate();
    }

    DevicePyramid(const DevicePyramid& other) : words_(other.words_) {
        for (size_t i = 0; i < Levels; ++i) levels_[i] = other.levels_[i];
        allocate();
        if (ptr_ != nullptr) {
            BINCV_CUDA_CHECK(cudaMemcpy(ptr_, other.ptr_, words_ * sizeof(uint32_t),
                                        cudaMemcpyDeviceToDevice));
        }
    }

    DevicePyramid& operator=(const DevicePyramid& other) {
        if (this == &other) return *this;
        DevicePyramid fresh(other);
        swap(fresh);
        return *this;
    }

    DevicePyramid(DevicePyramid&& other) noexcept { swap(other); }
    DevicePyramid& operator=(DevicePyramid&& other) noexcept {
        if (this != &other) {
            releaseAndClear();
            swap(other);
        }
        return *this;
    }

    ~DevicePyramid() { releaseAndClear(); }

    /// @brief Level `I` as an N-bit view, mutable. Level 0 is the one a caller
    /// fills per frame; 1..L-1 are the device's.
    template <size_t I>
    DevicePlaneBlockView level() {
        static_assert(I < Levels, "pyramid level index out of range");
        return levelAt(I);
    }
    template <size_t I>
    DevicePlaneBlockConstView level() const {
        static_assert(I < Levels, "pyramid level index out of range");
        return levelAt(I);
    }

    /// @brief Level `i` by RUNTIME index -- what a ladder walk needs.
    /// @note Out of range returns an empty view rather than reading past the
    /// table: a level index is data in a loop, not a template argument.
    DevicePlaneBlockView levelAt(size_t i) {
        if (i >= Levels || ptr_ == nullptr) return DevicePlaneBlockView{};
        const Extent& e = levels_[i];
        return DevicePlaneBlockView{ptr_ + e.offset, e.width, e.height, e.stride, e.planes};
    }
    DevicePlaneBlockConstView levelAt(size_t i) const {
        if (i >= Levels || ptr_ == nullptr) return DevicePlaneBlockConstView{};
        const Extent& e = levels_[i];
        return DevicePlaneBlockConstView{ptr_ + e.offset, e.width, e.height, e.stride,
                                         e.planes};
    }

    size_t levelWidth(size_t i) const { return i < Levels ? levels_[i].width : 0; }
    size_t levelHeight(size_t i) const { return i < Levels ? levels_[i].height : 0; }
    size_t levelPlanes(size_t i) const { return i < Levels ? levels_[i].planes : 0; }
    size_t levelStride(size_t i) const { return i < Levels ? levels_[i].stride : 0; }
    /// @brief Words in level `i` alone: planes * height * stride.
    size_t levelWords(size_t i) const {
        return i < Levels ? levels_[i].planes * levels_[i].height * levels_[i].stride : 0;
    }

    bool empty() const { return ptr_ == nullptr; }

    /// @brief Total words across every level -- the whole resident footprint.
    /// @note A PEAK, not a per-buffer ratio: the levels coexist, because a
    /// tracker reads all of them.
    size_t sizeInWords() const { return words_; }
    /// @brief Total bytes. This is the number the memory rule is an equality
    /// against, and it is a closed formula rather than a measurement.
    size_t sizeInBytes() const { return words_ * sizeof(uint32_t); }

    /// @brief O(1) pointer exchange. **The frame loop's ping-pong.**
    /// @note Copies nothing and allocates nothing. The contract it serves is
    /// the CALLER's: swap, then write level 0 of the new current pyramid,
    /// then build. Writing level 0 before the consumer has finished with
    /// the previous frame is silently wrong and nothing here can diagnose
    /// it -- see the file header.
    void swap(DevicePyramid& other) noexcept {
        uint32_t* p = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = p;
        const size_t w = words_;
        words_ = other.words_;
        other.words_ = w;
        for (size_t i = 0; i < Levels; ++i) {
            const Extent e = levels_[i];
            levels_[i] = other.levels_[i];
            other.levels_[i] = e;
        }
    }

    /// @brief The ladder's byte count WITHOUT allocating it.
    /// @note What a caller sizing a device budget needs, and what the memory
    /// rule compares the allocation against. Same arithmetic the constructor
    /// runs, in one place so the two cannot disagree.
    static size_t bytesFor(int width, int height, size_t rowAlignmentBytes = 4) {
        if (width < 0 || height < 0) return 0;
        const size_t alignWords = rowAlignmentBytes / 4;
        size_t w = static_cast<size_t>(width);
        size_t h = static_cast<size_t>(height);
        size_t total = 0;
        for (size_t i = 0; i < Levels; ++i) {
            const size_t stride =
                alignWords == 0 ? rowWords(w)
                                : ((rowWords(w) + alignWords - 1) / alignWords) * alignWords;
            total += kBits[i] * h * stride;
            w = pyrDownWidth(w);
            h = pyrDownHeight(h);
        }
        return total * sizeof(uint32_t);
    }

private:
    struct Extent {
        size_t width = 0;
        size_t height = 0;
        size_t stride = 0;  // words
        size_t planes = 0;
        size_t offset = 0;  // words from the base of the one allocation
    };

    /// @brief The bit depths as a table, so a RUNTIME ladder walk can read what
    /// the template pack declares. Implicitly inline in C++17, so there is no
    /// out-of-class definition to keep in step with it.
    static constexpr size_t kBits[Levels] = {LevelBits...};

    void layout(size_t width, size_t height, size_t alignWords) {
        size_t w = width;
        size_t h = height;
        size_t offset = 0;
        for (size_t i = 0; i < Levels; ++i) {
            const size_t minWords = rowWords(w);
            const size_t stride =
                alignWords <= 1 ? minWords
                                : ((minWords + alignWords - 1) / alignWords) * alignWords;
            levels_[i].width = w;
            levels_[i].height = h;
            levels_[i].stride = stride;
            levels_[i].planes = kBits[i];
            levels_[i].offset = offset;
            offset += kBits[i] * h * stride;
            w = pyrDownWidth(w);
            h = pyrDownHeight(h);
        }
        words_ = offset;
    }

    void allocate() {
        if (words_ == 0) return;
        void* p = nullptr;
        BINCV_CUDA_CHECK(cudaMalloc(&p, words_ * sizeof(uint32_t)));
        ptr_ = static_cast<uint32_t*>(p);
        BINCV_CUDA_CHECK(cudaMemset(ptr_, 0, words_ * sizeof(uint32_t)));
    }

    void releaseAndClear() {
        if (ptr_ != nullptr) cudaFree(ptr_);  // destructor path: no throw
        ptr_ = nullptr;
        words_ = 0;
        for (size_t i = 0; i < Levels; ++i) levels_[i] = Extent{};
    }

    uint32_t* ptr_ = nullptr;
    size_t words_ = 0;
    Extent levels_[Levels]{};
};

/// @brief Fills levels 1..L-1 of `p` from level 0, entirely on the device.
/// **API TIER 2** for the role; no `cv::cuda` equivalent exists.
/// @param p The ladder. Level 0 is the caller's input and is not touched.
/// @param stream The stream every level's launch is enqueued on, so the ladder
/// is ordered by the stream rather than by a synchronize.
/// @return The first failing level's status, or `cudaSuccess`.
/// @note L-1 launches. Nothing is allocated, nothing is downloaded, and there
/// is no synchronize: this is the per-frame device work of a resident
/// tracker, and a synchronize in it would be the whole cost.
/// @note Every level is `pyrDownBox` of the one above it, so the odd-extent and
/// padding rules are that kernel's, at every rung.
template <size_t... LevelBits>
inline cudaError_t buildPyramidBox(DevicePyramid<LevelBits...>& p,
                                   cudaStream_t stream = nullptr) {
    for (size_t i = 1; i < DevicePyramid<LevelBits...>::Levels; ++i) {
        const cudaError_t err = pyrDownBox(p.levelAt(i - 1), p.levelAt(i), stream);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
