// subpix_scene.hpp -- seeds for cornerSubPix, built the same way on a host and on a
// microcontroller: the sparse stereo scene's left frame, binarized, its derivatives,
// and the corners the reference detector finds in them.
//
// bincv_m7_subpix refines them on the board and m7_reference prints the host's digests
// for the board's to be held to. The caller supplies every buffer, so the board holds
// them in its own memory; nothing here allocates.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bincv/ops/corner.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/subpix.hpp"
#include "stereo_scene.hpp"

namespace subpixscene {

using stereoscene::kH;
using stereoscene::kW;
inline constexpr size_t kStride = (kW + 31) / 32;
inline constexpr size_t kPlaneWords = kStride * kH;
inline constexpr size_t kPool = 8192;
inline constexpr size_t kMaxCorners = 200;

/// The frame's bits, their derivatives, and the reference detector's corners as
/// seeds; returns how many. `img` is the 8-bit frame's buffer and `ring` three rows.
/// `dx` and `dy` hold two planes each (magnitude, sign); `ringFloats` three response
/// rows.
inline size_t build(uint8_t* img, uint8_t* ring, uint32_t* bits, uint32_t* dx, uint32_t* dy,
                    float* ringFloats, bincv::CornerCandidate* pool, bincv::Corner* corners,
                    bincv::Point2f* seeds) {
    stereoscene::buildLeft(img, ring);
    const bincv::BinMatView<uint32_t> b(bits, kW, kH, kStride);
    stereoscene::pack(img, b);
    bincv::SignedQuantMat<1, uint32_t> sx = bincv::SignedQuantMat<1, uint32_t>::wrap(
        dx, 2 * kPlaneWords, static_cast<int>(kW), static_cast<int>(kH), kStride);
    bincv::SignedQuantMat<1, uint32_t> sy = bincv::SignedQuantMat<1, uint32_t>::wrap(
        dy, 2 * kPlaneWords, static_cast<int>(kW), static_cast<int>(kH), kStride);
    const bincv::BinMatConstView<uint32_t> src[1] = {b};
    bincv::BinMatView<uint32_t> mx[1] = {sx.magnitude(0)}, my[1] = {sy.magnitude(0)};
    bincv::derivativeX<1, uint32_t>(src, mx, sx.sign());
    bincv::derivativeY<1, uint32_t>(src, my, sy.sign());
    const bincv::GoodFeaturesParams gftt;
    const bincv::ResponseMap rr{ringFloats, kW, bincv::kResponseRingRows, kW};
    const bincv::CornerResult cr = bincv::goodFeaturesToTrackStreaming<uint32_t>(
        sx.constMagnitude(0), sy.constMagnitude(0), sx.constSign(), sy.constSign(), gftt, rr,
        pool, kPool, corners, kMaxCorners);
    for (size_t i = 0; i < cr.count; ++i) {
        seeds[i] = bincv::Point2f{static_cast<float>(corners[i].x),
                                  static_cast<float>(corners[i].y)};
    }
    return cr.count;
}

/// The derivatives `build` made, as the containers cornerSubPix takes, over the same
/// buffers.
inline bincv::SignedQuantMat<1, uint32_t> view(uint32_t* d) {
    return bincv::SignedQuantMat<1, uint32_t>::wrap(d, 2 * kPlaneWords, static_cast<int>(kW),
                                                    static_cast<int>(kH), kStride);
}

} // namespace subpixscene
