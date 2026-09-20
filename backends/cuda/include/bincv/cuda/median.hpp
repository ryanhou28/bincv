#pragma once

/// @file median.hpp
/// @brief The median family on the device: the reference pipeline's three-pixel
/// median over PACKED BITS (`denoiseMedian3`) and over a WIDE 8- or 16-bit
/// image with a caller-chosen neighbourhood (`medianWide`).
///
/// ---------------------------------------------------------------------------
/// ONE HEADER, TWO HOST HEADERS
///
/// Every other header in this backend maps 1:1 onto a host header. This one
/// covers two -- `ops/denoise.hpp` and `ops/medianWide.hpp` -- because they are
/// one operation at two pixel widths, and splitting them would duplicate the
/// paragraph below about the neighbourhood and the zero border in two places
/// that could then disagree. The semantics are the host headers'; read those
/// for the derivation. What this file adds is what is device-specific: the
/// arms, their gates, and the domain the device narrows to.
///
/// ---------------------------------------------------------------------------
/// THE NEIGHBOURHOOD AND THE BORDER (both from the host, unchanged)
///
/// `denoiseMedian3` is an asymmetric three-pixel **L** -- above, self, right --
/// with the out-of-image neighbours reading 0. `medianWide` takes the pattern
/// as an argument; `kMedianReferenceL` is that same L and
/// `kMedianReferencePlus` is the five-pixel plus. Out-of-range samples read
/// **zero**, not replicate and not reflect. Neither is `cv::medianBlur` and
/// neither borrows its name: **API TIER 3** for every entry point here.
///
/// ---------------------------------------------------------------------------
/// WHERE THE FORMAT PAYS, AND WHERE IT DOES NOT -- SAID PLAINLY
///
/// `denoiseMedian3` is the family's structural result. On {0,1} pixels `min` is
/// AND and `max` is OR, so the reference's sorting network collapses to
/// `maj3(above, self, right)` -- ONE LOP3.LUT on this architecture, for 32
/// pixels. The operation moves 0.25 B/px where a byte-per-pixel GPU median
/// moves 2 B/px, which is the format's own 1-bit-against-1-byte ratio and no
/// cleverness at all.
///
/// `medianWide` has **NO structural advantage and none is claimed**. One byte
/// per pixel in, one byte per pixel out: binCV moves exactly what a `cv::cuda`
/// byte kernel moves, and the packed representation appears nowhere in it.
/// What it has instead is (a) a cheaper OPERATION -- 3 samples, or 5, against a
/// 3x3 square's 9 -- which is the algorithm's win and not the representation's,
/// and (b) an on-ramp: a caller whose wide frame is already resident cannot
/// reach the host arm without paying a download and an upload. Reporting
/// `medianWide`'s speed as evidence for 1-bit packing would be false.
///
/// ---------------------------------------------------------------------------
/// THE DEVICE DOMAIN, NARROWER THAN THE HOST'S, AND NAMED
///
/// The host `medianWide<K, SrcT>` accepts any odd `K` and any integer `SrcT`.
/// This one accepts:
///
/// * **K in {1, 3, 5, 7, 9}** -- a compile-time error otherwise, because the
/// selection network is unrolled at compile time and a runtime `K` would
/// need a dynamically indexed local array, which spills.
/// * **`SrcT` in {uint8_t, uint16_t}** -- the input contract's two integer
/// widths; there is no overload for anything else.
/// * **offsets within +/-127 in each axis** -- asserted, and a launcher handed
/// a wider pattern returns `cudaErrorInvalidValue` rather than launching.
///
/// Those are the only narrowings. Border behaviour, sample values and the
/// answer itself are the host's, bit for bit, and the test suite proves it
/// against the host library rather than against this file.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/medianWide.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace impl {

/// @brief Force `medianWide`'s per-pixel arm, for the benchmark and the tests.
/// **INTERNAL.**
bool& medianWideFastArmEnabled();

/// @brief The pattern as plain kernel-argument data. **INTERNAL.**
/// @note By VALUE into the launch: a host reference cannot reach a kernel, and
/// at nine offsets this is 76 bytes against a 4 KB parameter limit.
struct MedianOffsetsPod {
    int dx[9];
    int dy[9];
    int samples;
};

/// @brief The pattern as a POD, and the one place the host type is unpacked.
template <size_t K>
inline MedianOffsetsPod toPod(const MedianPattern<K>& pattern) {
    static_assert(K >= 1 && K <= 9, "cuda medianWide: 1 to 9 samples");
    MedianOffsetsPod pod{};
    pod.samples = static_cast<int>(K);
    for (size_t k = 0; k < K; ++k) {
        pod.dx[k] = pattern.offset[k].dx;
        pod.dy[k] = pattern.offset[k].dy;
    }
    return pod;
}

cudaError_t medianWideImpl(DeviceImageConstView<uint8_t> src,
                           DeviceImageView<uint8_t> dst, const MedianOffsetsPod& pattern,
                           cudaStream_t stream);
cudaError_t medianWideImpl(DeviceImageConstView<uint16_t> src,
                           DeviceImageView<uint16_t> dst,
                           const MedianOffsetsPod& pattern, cudaStream_t stream);

} // namespace impl

// ---------------------------------------------------------------------------
// The binary median
// ---------------------------------------------------------------------------

/// @brief `dst[y][x] = median(src[y-1][x], src[y][x], src[y][x+1])` over packed
/// bits, with the out-of-image neighbours reading 0. **API TIER 3.**
///
/// Device twin of `bincv::denoiseMedian3<uint32_t>`, bit-identical to it by
/// test. For binary pixels the median of three IS their majority, so each
/// destination word costs one `maj3` -- `ops/bitslice.hpp`'s own expression,
/// shared rather than re-derived here -- over 32 pixels.
///
/// @param src Source view.
/// @param dst Destination view; must have `src`'s dimensions.
/// @return The launch's error code. `cudaErrorInvalidValue` if `dst` overlaps
/// `src` (see the aliasing note) or a view's stride is short of a row.
///
/// @note **IN PLACE IS NOT SUPPORTED**, for the host kernel's reason: row `y`
/// reads row `y - 1`, so an in-place call would feed on its own output
/// from the second row onwards.
/// @note **THE ALIASING CHECK IS STRICTER THAN THE HOST'S.** The host uses
/// `impl::viewsShareNoWord`, a per-row predicate that correctly accepts
/// two interleaved views over one buffer. A launcher cannot afford to walk
/// rows, so this one rejects any BOUNDING-BOX overlap. Views the host
/// would accept are therefore refused here; that is a narrowing of the
/// domain, not a difference in what the kernel computes.
/// @note The destination's padding bits are zero on return, and the source's
/// padding bits are never read as pixels. One mask carries both, exactly
/// as it does on the host: the trailing source word is masked BEFORE the
/// right shift, because the right neighbour of pixel `width - 1` is the
/// bit the shift would otherwise pull in, and the zero border says that
/// neighbour must read 0. Moving that mask to the store passes every
/// interior test and fails on widths that are not multiples of 32.
/// @note Empty views (width or height 0) are a no-op, not an error.
/// @note No scratch, no allocation, one launch. The above-neighbour is a row
/// index and the right-neighbour is a register.
/// @note **ONE IMPLEMENTATION, NO OFF-SWITCH, AND THAT IS A MEASURED RESULT.** A
/// uint4 arm -- 128 pixels per thread through one 128-bit load -- was
/// written, proven bit-exact and timed on a ladder to 4096x2160. It never
/// separated from this one, because at 4K the whole operation moves 2.21 MB
/// (about 3.6 us of traffic on the reference GPU) against a 7-10 us launch:
/// the kernel is cheaper than the launch that carries it at every frame
/// size, so no kernel shape can move the number. See median.cu for the
/// figures.
cudaError_t denoiseMedian3(DeviceBinMatConstView src, DeviceBinMatView dst,
                           cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// The wide median
// ---------------------------------------------------------------------------

/// @brief Median filter over a caller-chosen neighbourhood on a wide image.
/// **API TIER 3.** Device twin of `bincv::medianWide<K, uint8_t>`, byte-
/// identical to it by test.
///
/// @param src,dst Wide device views of equal extent. **They must not alias.**
/// @param pattern `K` sample offsets relative to the pixel being written;
/// `kMedianReferenceL` and `kMedianReferencePlus` are the shipped two.
/// @return The launch's error code; `cudaErrorInvalidValue` for an out-of-domain
/// pattern or aliasing views.
///
/// @note Out-of-range samples read **zero** -- the host's border rule, written
/// once in a per-pixel helper that both arms call.
/// @note **K IS COMPILE-TIME AND MUST BE 1, 3, 5, 7 OR 9.** See the domain note
/// at the top of this file.
/// @note No scratch, no shared memory, one launch. Shared staging was not
/// attempted: at K <= 9 each source byte is read at most nine times by
/// threads of the same block, which are L1 hits, and the recorded
/// precedent for staging an already-resident window here is 1.28x SLOWER.
/// @note There is **no structural memory advantage** over a byte-per-pixel GPU
/// median and none is reported; see the top of this file.
template <size_t K>
inline cudaError_t medianWide(DeviceImageConstView<uint8_t> src,
                              DeviceImageView<uint8_t> dst,
                              const MedianPattern<K>& pattern,
                              cudaStream_t stream = nullptr) {
    static_assert(K == 1 || K == 3 || K == 5 || K == 7 || K == 9,
                  "cuda medianWide: K must be 1, 3, 5, 7 or 9");
    return impl::medianWideImpl(src, dst, impl::toPod(pattern), stream);
}

/// @brief The 16-bit spelling. **API TIER 3.**
/// @note The input contract keeps sources wider than 8 bits because
/// downconverting first destroys the small gradients a threshold is looking
/// for, so this overload is not an afterthought -- it is the reason the
/// contract says "integer-typed", not "8-bit".
/// @note **NO `cv::cuda` COUNTERPART EXISTS FOR THIS INPUT TYPE AT ANY API
/// LEVEL**: `cv::cuda::createMedianFilter` accepts `CV_8UC1` only. Its
/// speed verdict against a GPU alternative is therefore OUTSTANDING, not
/// won; the benchmark says so on the line rather than quoting a CPU number
/// in a GPU comparison.
template <size_t K>
inline cudaError_t medianWide(DeviceImageConstView<uint16_t> src,
                              DeviceImageView<uint16_t> dst,
                              const MedianPattern<K>& pattern,
                              cudaStream_t stream = nullptr) {
    static_assert(K == 1 || K == 3 || K == 5 || K == 7 || K == 9,
                  "cuda medianWide: K must be 1, 3, 5, 7 or 9");
    return impl::medianWideImpl(src, dst, impl::toPod(pattern), stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
