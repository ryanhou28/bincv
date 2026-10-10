// stereo_scene.hpp -- the sparse stereo benchmark's synthetic pair, built so that a
// microcontroller can build it too, and a digest of everything the matcher produces.
//
// stereo_benchmark.cpp prices the matcher's stages on a host; bincv_m7_stereo prices
// them on the board. Both build the pair from this source, so the board's digests can
// be held to the host's. The pair: a smoothed random 752x480 frame on the left and the
// same frame shifted 21 pixels on the right, so every keypoint has a true disparity.
//
// THE FRAME IS MADE A ROW AT A TIME. The two smoothing passes need three rows of the
// first pass, not a second frame, so the whole scene fits one 8-bit frame -- the left
// frame, shifted in place into the right one once the left's descriptors and bits are
// taken -- beside the two binary frames.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/stereo.hpp"

namespace stereoscene {

inline constexpr size_t kW = 752, kH = 480;  // the reference frame size
inline constexpr int kDisp = 21;              // the right frame's shift
inline constexpr size_t kKp = 500;            // keypoints per frame
inline constexpr size_t kBits = 256;          // descriptor bits
inline constexpr size_t kWords = kBits / 32;  // uint32_t words per descriptor

/// The smoothed random frame into `img` (kW x kH bytes): the raster-order LCG, then a
/// [1 2 1] / 4 filter across and down over the interior, the border rows and columns
/// left as drawn. `ring` is three rows of scratch.
inline void buildLeft(uint8_t* img, uint8_t* ring) {
    uint64_t st = 0xFEEDFACEULL;
    for (size_t i = 0; i < kW * kH; ++i) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        img[i] = static_cast<uint8_t>(st >> 40);
    }
    // The across pass of row r, into ring slot r % 3; rows 0 and kH - 1 have none and
    // read as zero, as the host's untouched first-pass rows do.
    const auto across = [&](size_t r) {
        uint8_t* t = ring + (r % 3) * kW;
        for (size_t x = 0; x < kW; ++x) t[x] = 0;
        if (r == 0 || r + 1 >= kH) return;
        const uint8_t* s = img + r * kW;
        for (size_t x = 1; x + 1 < kW; ++x)
            t[x] = static_cast<uint8_t>((s[x - 1] + 2u * s[x] + s[x + 1]) / 4u);
    };
    across(0);
    across(1);
    for (size_t y = 1; y + 1 < kH; ++y) {
        across(y + 1);  // before row y is overwritten: it reads row y + 1's draws
        const uint8_t* a = ring + ((y - 1) % 3) * kW;
        const uint8_t* b = ring + (y % 3) * kW;
        const uint8_t* c = ring + ((y + 1) % 3) * kW;
        uint8_t* d = img + y * kW;
        for (size_t x = 1; x + 1 < kW; ++x)
            d[x] = static_cast<uint8_t>((a[x] + 2u * b[x] + c[x]) / 4u);
    }
}

/// The left frame into the right one, in place: every row shifted kDisp to the left,
/// the vacated columns zero.
inline void shiftToRight(uint8_t* img) {
    for (size_t y = 0; y < kH; ++y) {
        uint8_t* row = img + y * kW;
        for (size_t x = 0; x + kDisp < kW; ++x) row[x] = row[x + kDisp];
        for (size_t x = kW - kDisp; x < kW; ++x) row[x] = 0;
    }
}

/// The keypoints, left and right, (x, y) interleaved: the same draws on both sides,
/// the right one kDisp further left.
inline void buildKeypoints(float* left, float* right) {
    uint64_t st = 0x5EA15EEDULL;
    for (size_t i = 0; i < kKp; ++i) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        const int x = 80 + static_cast<int>((st >> 33) % (kW - 160));
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        const int y = 40 + static_cast<int>((st >> 33) % (kH - 80));
        left[2 * i] = static_cast<float>(x);
        left[2 * i + 1] = static_cast<float>(y);
        right[2 * i] = static_cast<float>(x - kDisp);
        right[2 * i + 1] = static_cast<float>(y);
    }
}

/// The frame's bits, as the benchmark binarizes it.
inline void pack(const uint8_t* img, bincv::BinMatView<uint32_t> bits) {
    bincv::packBits<bincv::PackRule::GreaterThan>(img, kW, kH, kW, bits, uint8_t{127});
}

inline uint32_t fnv1a(uint32_t h, const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}
inline constexpr uint32_t kFnvSeed = 2166136261u;

/// A digest of a frame's bits, row by row, so padding never enters it.
inline uint32_t digestBits(bincv::BinMatConstView<uint32_t> bits) {
    uint32_t h = kFnvSeed;
    for (size_t y = 0; y < bits.height; ++y) h = fnv1a(h, bits.row(y), bits.stride * 4);
    return h;
}

/// A digest of the matcher's records: disparity bits, distance, right index, validity.
inline uint32_t digestMatches(const bincv::StereoMatch* m, size_t n) {
    uint32_t h = kFnvSeed;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t idx = static_cast<uint32_t>(m[i].rightIndex);
        h = fnv1a(h, &m[i].disparity, sizeof(float));
        h = fnv1a(h, &m[i].distance, sizeof(unsigned));
        h = fnv1a(h, &idx, sizeof idx);
        h = fnv1a(h, &m[i].valid, 1);
    }
    return h;
}

} // namespace stereoscene
