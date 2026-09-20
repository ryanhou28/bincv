#pragma once

/// @file corner.hpp
/// @brief The device arm of ops/corner.hpp: the minimum-eigenvalue response map
/// and the good-features selection it feeds. **API TIER 2.**
///
/// ---------------------------------------------------------------------------
/// TIER 2, AND THE HOST HEADER'S REASONS ARE UNCHANGED HERE
///
/// Same ROLE as `cv::cuda::createGoodFeaturesToTrackDetector` over
/// `cv::cuda::createMinEigenValCorner`, with deliberately different numerics:
/// binCV's derivatives are binarized `[-1, 0, 1]` taps over a ONE-BIT image,
/// OpenCV's are a Sobel over bytes. Those are different numbers before any
/// window is summed. Correctness here is settled against the HOST LIBRARY, byte
/// for byte, and never against OpenCV.
///
/// ---------------------------------------------------------------------------
/// THE RESPONSE IS BIT-EXACT BY CONSTRUCTION, AND THERE IS NO LOOKUP TABLE
///
/// `bincv::impl::minEigenValue` already carries BINCV_HOST_DEVICE, and
/// `test_cuda_shared_helpers.cu` already sweeps nvcc's device compilation of it
/// against the host's over the reachable `(xx, yy, xy)` domain. So the device
/// kernels CALL IT. There is no device transcription of the response, no
/// host-built table uploaded to stand in for it, and nothing for the two to
/// drift over.
///
/// **AND NO `-fmad=false` ON THIS FILE, WHICH IS A CORRECTION TO THE FAMILY'S
/// OWN DESIGN.** The hazard the design guarded against does not exist here:
/// `disc = d*d + 4.0*c*c` has BOTH products exact integers (corner.hpp's
/// PRECISION section bounds `D <= 5*blockSize^4`, far under 2^53), so their sum
/// is exact and a fused multiply-add and a separate mul-then-add give bit-
/// identical results at every blockSize this library can express.
/// `0.5*(s - sqrt(disc))` offers no multiply-add to contract at all. The
/// contraction hazard is real in ops/subpix.hpp and is handled there; naming it
/// here too would have been superstition dressed as care.
///
/// ---------------------------------------------------------------------------
/// THE STRUCTURAL CLAIM: THE FRAME-SIZED INTERMEDIATES DO NOT EXIST
///
/// `cv::cuda`'s path to the same corners runs a Sobel into two `CV_32F` planes,
/// forms a `CV_32FC3` covariance buffer, box-filters it, writes a `CV_32F` eig
/// map, and the gftt detector then dilates that into a second `CV_32F` map
/// before sorting. binCV reads four ONE-BIT planes -- 0.5 B/px -- and the FUSED
/// arm materialises **zero frame-sized bytes**: the 3x3 box sums are full-adder
/// trees over whole words (one `xor`/`and`/`or` advances 32 pixels of a 0..9 box
/// sum) and the 3x3 suppression window lives in shared memory.
///
/// **AND THE HALF OF IT THAT HAS NO OpenCV COUNTERPART AT ALL.**
/// `cv::cuda::GoodFeaturesToTrackDetector::detect` DOWNLOADS its candidate list
/// and runs the minimum-distance spacing filter ON THE HOST whenever
/// `minDistance >= 1` (cudaimgproc/src/gftt.cpp). The reference configuration's
/// `minDistance` is 33.33, so that is the path any like-for-like comparison
/// measures. binCV's selection stays device-resident end to end. That is a
/// residency claim in its own right and the benchmark states it at the number,
/// because a kernel-event clock on OpenCV's side would silently exclude its
/// host pass and a wall clock includes a round trip binCV does not pay.
///
/// ---------------------------------------------------------------------------
/// THE SELECTION'S ORDER IS THE SPECIFICATION, AND IT SURVIVES THE DEVICE
///
/// Threshold, then 3x3 NMS, then rank, then greedy spacing -- gftt.cpp's order,
/// which the host header shows changes the answer if swapped. Three device
/// facts make the device's answer the host's:
///
/// 1. **The global maximum covers EVERY pixel of the frame**, border row and
/// column included, exactly as `cv::minMaxLoc` does in the reference. The
/// tiles partition the frame, each block reduces its own OWNED pixels, and
/// nothing outside the frame contributes. A maximum taken over the NMS
/// interior instead would move the threshold and change the whole surviving
/// set, and corner.hpp's own BorderRing cases establish that border
/// responses are real.
/// 2. **The append order cannot reach the output.** Everything after it sorts
/// under `impl::CornerStronger`, which corner.hpp establishes is a TOTAL
/// order over distinct positions -- so the sorted sequence is unique and an
/// unordered atomic append is as good as an ordered one.
/// 3. **The zero prune is exact and needs no running state.** A candidate whose
/// response is exactly `0.0f` can never survive `val > float(double(maxVal)*q)`,
/// because that threshold is never negative and the comparison is strictly
/// greater. The host's streaming form prunes with a running threshold for the
/// same purpose -- the flat background of an edge map -- and its top-K
/// argument tolerates any prune that only removes non-members of the
/// surviving set.
///
/// **NO RANK TABLE, AND THE REASON IS A MEASURED ONE.** The family's design
/// proposed ranking by a 1900-entry positional rank over the `(xx, yy, xy)`
/// code. Evaluated, those 1900 codes produce only 380 DISTINCT float responses,
/// and 379 of the 380 are shared by two or more codes -- the response is
/// symmetric in `xx <-> yy` and even in `xy`, so `(xx, yy, xy)` and
/// `(yy, xx, -xy)` always collide. A positional rank would therefore ORDER
/// corners the host treats as exactly tied and `CornerStronger`'s tie rule would
/// never fire across codes. Sorting on the `float` response itself, with the
/// host's own comparator, cannot have that bug; the counting sort it replaces was
/// an optimisation, and one that would have been wrong.
///
/// ---------------------------------------------------------------------------
/// THE ORDER COSTS LESS THAN THE CORNER: THE SELECTION IS NOT ONE BLOCK
///
/// The selection used to be one block for the whole tail -- a bitonic network
/// over 16-byte corner records, then a greedy spacing filter that rescanned
/// every surviving candidate once per acceptance. On the reference frame, with
/// 25,115 candidates and 200 corners kept, that was 5.6 ms of sort and 23.1 ms
/// of spacing against 0.13 ms of candidate generation: 98% of the operation, on
/// one of 48 SMs, at 1.29% of the part's throughput.
///
/// Two facts about the REPRESENTATION replace it, and both are the same fact.
///
/// 1. **`CornerStronger` is a 64-bit unsigned comparison, and the key is the
/// corner.** A response is never negative and never NaN (the PRECISION
/// section below), so its IEEE bit pattern orders exactly as the float does;
/// `x` and `y` are below 65536. So
/// `key = (~responseBits) << 32 | (0xFFFF - y) << 16 | (0xFFFF - x)`
/// makes ASCENDING key order identical to `CornerStronger`, and the pack is
/// LOSSLESS -- the corner comes back out with three shifts. The sort moves
/// 8 bytes per element instead of 16, no payload array exists, and the
/// spacing filter reads a position out of 4 bytes. Keys are unique because
/// positions are, so the sorted sequence is unique and the sort's own
/// order-dependence cannot reach the answer.
/// 2. **The spacing filter TESTS, it does not kill.** Killing is one pass over
/// every survivor per acceptance. Testing -- the host's own loop: for each
/// candidate in rank order, accept it when no accepted corner is within
/// `minDistance` -- is one pass over the candidates, and the accepted set is
/// at most `maxCorners` points in shared memory. A block tests 1024
/// candidates at once and then resolves the acceptances inside that chunk
/// serially, which is at most one round per corner kept.
///
/// The sort is a standard bitonic network made device-wide: a block sorts 4096
/// keys in shared memory and the stages wider than a chunk are their own
/// launches, because a block barrier is not a device barrier.
///
/// ---------------------------------------------------------------------------
/// THE DEVICE DOMAIN, NAMED because it is narrower than the host's (ruling R4)
///
/// * Word type `uint32_t`. The host compiles at 8, 16, 32 and 64.
/// * **Ternary planes only** -- four `DeviceBinMatConstView`s, which is what the
/// host's promise 4 restricts its own container spelling to. An N-bit level
/// cannot be spelled here at all.
/// * **No `mask` overload.** The host's masked selection gates candidates and the
/// threshold but NOT the suppression, and the device arm does not implement it.
/// A masking caller uses the host arm. Stated as a subset rather than left to
/// be discovered.
/// * Candidate counts, capacities and corner counts are `uint32_t`
/// (compaction.hpp's domain), asserted and refused rather than narrowed.
/// * **At most 65536 pixels on a side.** The selection's ordering key packs
/// `CornerStronger`'s tie rule -- y descending, then x descending -- into 16
/// bits per axis, which is what makes the key lossless and the sort a plain
/// unsigned comparison. A larger frame is refused, not wrapped.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/corner.hpp"
#include "compaction.hpp"
#include "core.hpp"
#include "features.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief What the device selection reports back -- the host's `CornerResult`,
/// plus the one failure only a device buffer can have.
/// @note `candidateOverflow` is not decoration and it is not `candidatesTruncated`.
/// The host's candidate buffer IS its output array, so a buffer too small for
/// every NMS survivor is a documented restriction with an honest flag. The
/// device's candidate buffer is a separate device array the caller sized, and
/// when the raw 3x3 maxima overflow it the selection cannot compute
/// `candidatesRanked` or `candidatesTruncated` at all -- the numbers it would
/// return are not a restriction of the host's answer, they are unrelated to
/// it. So it returns nothing and says so, and the caller re-runs against the
/// candidate counter's `found()`, which is exact.
struct DeviceCornerResult {
    uint32_t count = 0;                ///< corners written to the caller's array
    uint32_t candidatesRanked = 0;     ///< NMS survivors above the threshold that were ranked
    uint32_t candidatesTruncated = 0;  ///< 1 when `capacity` could not hold every survivor
    uint32_t candidateOverflow = 0;    ///< 1 when the DEVICE candidate buffer overflowed

    /// @brief The host library's triple. Host-side; there is no device use for it.
    CornerResult toHost() const {
        CornerResult r;
        r.count = count;
        r.candidatesRanked = candidatesRanked;
        r.candidatesTruncated = candidatesTruncated != 0;
        return r;
    }
};

/// @brief The device memory `goodFeaturesToTrackAsync` works in, all of it the
/// caller's. **No kernel here allocates.**
/// @note A view bundle, not a container: four independent pieces that must
/// travel together because separating them is how a kernel ends up with a
/// buffer and no counter.
struct DeviceGoodFeaturesWorkspace {
    /// @brief Where raw 3x3 maxima are appended, with its counter
    /// (compaction.hpp). **The counter must be zeroed before every launch.**
    /// Size it for the raw maxima of the frame, NOT for `maxCorners`: on
    /// binarized content the zero prune removes the flat background, but a
    /// textured frame can still produce thousands.
    DeviceCornerBuffer candidates;
    /// @brief ONE `uint32_t` of device memory holding the frame maximum's IEEE
    /// bit pattern. **Must be zeroed before every launch** -- a response is
    /// never negative, so zero is the correct identity for the maximum.
    uint32_t* maxBits = nullptr;
    /// @brief At least `goodFeaturesScratchBytes(candidates.capacity)` bytes,
    /// 16-byte aligned.
    void* scratch = nullptr;
    size_t scratchBytes = 0;
    /// @brief The frame-sized response map. **Required by the REFERENCE arm and
    /// nothing else**: with `impl::cornerFusedEnabled()` true it is unused and
    /// may be empty, which is the arm's whole claim -- zero frame-sized bytes.
    DeviceImageView<float> frameMap;
};

/// @brief Scratch bytes `goodFeaturesToTrackAsync` needs for a candidate capacity.
/// @note **TOTAL, over both arms and every internal path.** It does not depend on
/// which switch is set, on `blockSize`, or on any shared-memory budget: a
/// sizing function whose answer changes when an implementation detail moves
/// turns a correct caller into a device-side out-of-bounds write.
size_t goodFeaturesScratchBytes(size_t candidateCapacity);

/// @brief The minimum-eigenvalue corner response at every pixel, on the device.
/// **API TIER 2.** Bit-exact against `bincv::cornerMinEigenVal` over
/// `BinMatConstView<uint32_t>` planes, pixel for pixel.
/// @param magX,magY Magnitude planes of the x- and y-derivatives.
/// @param signX,signY Sign planes; a SET bit is NEGATIVE.
/// @param blockSize Side of the square covariance window, >= 1. The pipeline
/// runs 3, which is the size the bit-sliced arm covers.
/// @param dst Caller-owned `float` map with the planes' dimensions. Every pixel
/// is written; nothing is read from it.
/// @return `cudaErrorInvalidValue` on a domain violation, else the launch's code.
/// @note **Windows CLIP at the frame edge**, as on the host: a row outside the
/// frame is a zero plane and a bit shifted in from a word that does not exist
/// is 0. The three counts are the same exact integers the host produces.
/// @note **This is the form for a caller who wants the map.** A caller who wants
/// only the corners should call `goodFeaturesToTrackAsync` with the fused arm,
/// which never materialises one.
cudaError_t cornerMinEigenValAsync(DeviceBinMatConstView magX, DeviceBinMatConstView magY,
                                   DeviceBinMatConstView signX, DeviceBinMatConstView signY,
                                   int blockSize, DeviceImageView<float> dst,
                                   cudaStream_t stream = nullptr);

/// @brief `goodFeaturesToTrack` over device ternary derivative planes: response,
/// threshold, 3x3 NMS, rank, greedy spacing. **API TIER 2**, and **bit-for-bit
/// the same corners, order and result triple** as the host's
/// `goodFeaturesToTrackStreaming` whenever `work.candidates` held every raw
/// maximum.
/// @param magX,magY,signX,signY The four derivative planes, in device memory.
/// @param params The host's four values; defaults are the pipeline's.
/// @param work Caller-owned candidate buffer, frame-maximum word, scratch and
/// (reference arm only) response map. See DeviceGoodFeaturesWorkspace.
/// @param corners Caller-owned device output array.
/// @param capacity Entries in `corners`. **This is the HOST's `capacity`
/// contract, not `maxCorners`**: it bounds the survivors that can be RANKED,
/// and `candidatesTruncated` reports when it could not hold them all.
/// @param result One `DeviceCornerResult` in device memory.
/// @return `cudaErrorInvalidValue` on a domain violation, else the launches' code.
/// @note Asynchronous. Read `result` back with a copy on the same stream.
/// @note **Never allocates.** Every byte it writes is in `work`, `corners` or
/// `result`.
cudaError_t goodFeaturesToTrackAsync(DeviceBinMatConstView magX, DeviceBinMatConstView magY,
                                     DeviceBinMatConstView signX, DeviceBinMatConstView signY,
                                     const GoodFeaturesParams& params,
                                     const DeviceGoodFeaturesWorkspace& work,
                                     DeviceCorner* corners, uint32_t capacity,
                                     DeviceCornerResult* result,
                                     cudaStream_t stream = nullptr);

namespace impl {

/// @brief Forces the per-pixel window arm of the response, for the benchmark and
/// the tests. **INTERNAL.**
/// @note A vector arm must be switchable off and the benchmark must show it is
/// on. Both arms are held to the same map in one binary by the suite, and the
/// benchmark prints the on/off ratio beside a case the fast arm's own gate
/// excludes, which must read ~1.00x.
bool& cornerSlicedEnabled();

/// @brief Whether the bit-sliced arm would run at this `blockSize`. **INTERNAL** --
/// the benchmark and the suite need to name the gate-excluded case without
/// restating the gate.
/// @note The gate is `blockSize == 3`: the full-adder tree that sums three
/// 2-bit numbers into four planes is a 3x3 identity, and 3 is the size the
/// host's own fast path covers and every pipeline here runs. `blockSize = 7`
/// is therefore the control case that must read ~1.00x between switch
/// positions.
bool cornerSlicedApplies(int blockSize);

/// @brief Selects the FUSED arm (default) or the frame-map REFERENCE arm.
/// **INTERNAL.** The reference arm needs `work.frameMap`; the fused arm
/// materialises no frame-sized bytes at all. Both are held to the same corner
/// array and the same result triple in one binary.
bool& cornerFusedEnabled();

/// @brief Selects the DEVICE-WIDE sort (default) or the one-block network.
/// **INTERNAL.** Both sort the same 64-bit ordering keys with the same bitonic
/// network and are held to the same corner array in one binary; the difference
/// is whether the stages wider than a shared-memory chunk are their own
/// launches or another pass of one block.
/// @note The one-block arm is what the operation shipped with, and on the
/// reference frame it was 5.6 ms of a 9.8 ms detection.
bool& cornerSortParallelEnabled();

/// @brief Whether the device-wide sort would differ from the one-block sort for
/// this candidate capacity. **INTERNAL** -- the benchmark and the suite need to
/// name the gate-excluded case without restating the gate.
/// @note The gate is `nextPow2(candidateCapacity) > 2048`: at or below one
/// shared-memory chunk the "device-wide" ladder IS one block sorting one chunk,
/// so both switch positions run the same network and the benchmark must report
/// ~1.00x.
bool cornerSortParallelApplies(size_t candidateCapacity);

/// @brief Selects the chunked spacing filter (default) or the reference arm that
/// kills forward once per acceptance. **INTERNAL.**
/// @note The two are the same answer and the chunked one is the HOST's own loop:
/// for each candidate in rank order, accept it when no accepted corner is within
/// `minDistance`. The reference arm reaches that answer by marking the losers
/// instead, which costs one pass over every surviving candidate per acceptance
/// -- 200 passes over 25,115 candidates on the reference frame, and 23.1 ms of a
/// 28.9 ms detection when it was measured alone.
bool& cornerSpacingChunkedEnabled();

/// @brief Whether the spacing filter runs at all for these parameters.
/// **INTERNAL.** The gate is `minDistance >= 1.0`, which is `cv::gftt`'s own:
/// below it there is no spacing pass, both switch positions take the same
/// branch, and the benchmark must report ~1.00x.
bool cornerSpacingChunkedApplies(const GoodFeaturesParams& params);

} // namespace impl

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
