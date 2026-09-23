// Descriptor matching, sparse rectified stereo and Hamming block matching:
// device against host, byte for byte, both arms.
//
// A CUDA translation unit, and it has to be one. Two of the cases below call
// __device__ code AS A UNIT rather than through a kernel that uses it -- the
// displaced-row builder and the RowSpan extraction -- and those are the two
// pieces of arithmetic in this family whose failure mode is a plausible
// disparity rather than a crash. They are pinned against the host struct
// BEFORE any kernel that calls them is checked, so a sweep failure names the
// primitive instead of the operation.
//
// EVERY CASE RUNS TWICE, fast arm on and fast arm off, and compares the host,
// the fast arm and the reference arm to each other. An arm that is only ever
// compared against the host cannot tell you the two device arms agree, and the
// switch that selects between them is the thing a mis-attached #define once
// broke here.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/sparseMatch.cuh"
#include "bincv/cuda/sparseMatch.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/blockMatch.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/stereo.hpp"
#include "test_util.hpp"

namespace {

using bincv::cuda::DeviceArray;
using bincv::cuda::DeviceBinMat;

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// A plane of random bits with the padding CLEAN -- the library's own
/// precondition, so the host reference is computed on a legal input.
bincv::BinMat<uint32_t> randomPlane(size_t w, size_t h, uint64_t seed) {
    bincv::BinMat<uint32_t> m(static_cast<int>(w), static_cast<int>(h));
    if (w == 0 || h == 0) return m;
    const size_t words = bincv::impl::minRowWords<uint32_t>(w);
    const uint32_t tail = bincv::impl::rowTailMask<uint32_t>(w);
    bincv::BinMatView<uint32_t> v = m.plane(0);
    uint64_t s = seed;
    for (size_t y = 0; y < h; ++y) {
        for (size_t i = 0; i < words; ++i) {
            const uint32_t word = static_cast<uint32_t>(splitmix(s));
            v.row(y)[i] = (i + 1 == words) ? (word & tail) : word;
        }
    }
    return m;
}

/// Sets every padding bit past `width` in a DEVICE plane.
/// @note The adversarial half of the padding-bit invariant. No op in this
/// family writes a plane, so what has to hold is that a DIRTY plane cannot
/// change an answer -- and a clean plane makes a missing mask invisible.
/// Done on the device rather than on the host copy because `upload` moves
/// whole PIXEL bytes, so a host-side dirty word past the last pixel byte
/// never reaches the device at all.
__global__ void dirtyPaddingKernel(bincv::cuda::DeviceBinMatView v, uint32_t notTail,
                                   size_t lastWord) {
    const size_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= v.height) return;
    v.row(y)[lastWord] |= notTail;
}

void dirtyDevicePadding(DeviceBinMat& m) {
    const size_t w = m.getWidth();
    if (w == 0 || m.getHeight() == 0) return;
    const uint32_t tail = bincv::impl::rowTailMask<uint32_t>(w);
    if (tail == 0xFFFFFFFFu) return;   // the row ends on a word boundary
    const size_t words = bincv::impl::minRowWords<uint32_t>(w);
    const unsigned block = 128;
    const unsigned grid =
        static_cast<unsigned>((m.getHeight() + block - 1) / block);
    dirtyPaddingKernel<<<grid, block>>>(m.view(), ~tail, words - 1);
    cudaDeviceSynchronize();
}

DeviceBinMat uploadPlane(const bincv::BinMat<uint32_t>& m) {
    DeviceBinMat d(static_cast<int>(m.getWidth()), static_cast<int>(m.getHeight()));
    if (m.getWidth() != 0 && m.getHeight() != 0)
        BINCV_CHECK_EQ(bincv::cuda::upload<uint32_t>(m.plane(0), d.view()), cudaSuccess);
    return d;
}

template <typename T>
DeviceArray<T> uploadVec(const std::vector<T>& v) {
    DeviceArray<T> d(v.size());
    if (!v.empty())
        BINCV_CHECK_EQ(cudaMemcpy(d.data(), v.data(), v.size() * sizeof(T),
                                  cudaMemcpyHostToDevice),
                       cudaSuccess);
    return d;
}

/// The same upload, but never a null pointer for an empty set.
/// @note An empty OCTAVE array is not the same thing as no octave array:
/// `hasOctave()` is a null test, so an empty `DeviceArray`'s null `data()`
/// would turn "octaves supplied, zero keypoints" into "no octaves" and the
/// gated matcher's null-or-both check would then refuse a legal call. The
/// host has the same distinction and the same rule.
template <typename T>
DeviceArray<T> uploadVecNonNull(const std::vector<T>& v) {
    DeviceArray<T> d(v.empty() ? 1 : v.size());
    if (!v.empty())
        BINCV_CHECK_EQ(cudaMemcpy(d.data(), v.data(), v.size() * sizeof(T),
                                  cudaMemcpyHostToDevice),
                       cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> downloadVec(const DeviceArray<T>& d, size_t count) {
    std::vector<T> v(count);
    if (count != 0)
        BINCV_CHECK_EQ(cudaMemcpy(v.data(), d.data(), count * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       cudaSuccess);
    return v;
}

std::vector<uint32_t> randomDescriptors(size_t count, size_t words, uint64_t seed) {
    std::vector<uint32_t> d(count * words);
    uint64_t s = seed;
    for (auto& w : d) w = static_cast<uint32_t>(splitmix(s));
    return d;
}

std::vector<float> randomPositions(size_t count, float w, float h, uint64_t seed) {
    std::vector<float> xy(2 * count);
    uint64_t s = seed;
    for (size_t i = 0; i < count; ++i) {
        xy[2 * i] = static_cast<float>(splitmix(s) % 100000u) * (w / 100000.0f);
        xy[2 * i + 1] = static_cast<float>(splitmix(s) % 100000u) * (h / 100000.0f);
    }
    return xy;
}

/// One element for an empty set, so `.data()` is never null.
/// @note For the same reason `uploadVecNonNull` exists on the device side:
/// the HOST gated matcher's null-or-both octave check reads a null
/// `std::vector::data()` as "no octaves supplied", so a zero-length train
/// set would make a legal call assert. Nothing reads the padding element.
template <typename T>
std::vector<T> atLeastOne(std::vector<T> v) {
    if (v.empty()) v.resize(1);
    return v;
}

std::vector<int32_t> randomOctaves(size_t count, int levels, uint64_t seed) {
    std::vector<int32_t> o(count);
    uint64_t s = seed;
    for (auto& v : o) v = static_cast<int32_t>(splitmix(s) % static_cast<uint64_t>(levels));
    return o;
}

/// Field-for-field, because the device record narrows `trainIndex` to 32 bits
/// under its documented domain and therefore is NOT the host struct byte for
/// byte. Every other field is compared at the host's own type.
size_t matchesDiffering(const std::vector<bincv::DescriptorMatch>& want,
                        const std::vector<bincv::cuda::DeviceDescriptorMatch>& got) {
    size_t bad = 0;
    if (want.size() != got.size()) return want.size() + got.size();
    for (size_t i = 0; i < want.size(); ++i) {
        if (static_cast<size_t>(got[i].trainIndex) != want[i].trainIndex) ++bad;
        if (got[i].distance != want[i].distance) ++bad;
        if (got[i].secondDistance != want[i].secondDistance) ++bad;
        if ((got[i].valid != 0u) != want[i].valid) ++bad;
    }
    return bad;
}

/// `disparity` compared BIT FOR BIT as a float, not with an epsilon: the whole
/// point of compiling this family without FMA contraction is that the two
/// targets produce the same bits, and a tolerance fitted to an observed
/// difference cannot fail.
size_t stereoDiffering(const std::vector<bincv::StereoMatch>& want,
                       const std::vector<bincv::cuda::DeviceStereoMatch>& got) {
    size_t bad = 0;
    if (want.size() != got.size()) return want.size() + got.size();
    for (size_t i = 0; i < want.size(); ++i) {
        if (std::memcmp(&want[i].disparity, &got[i].disparity, sizeof(float)) != 0) ++bad;
        if (got[i].distance != want[i].distance) ++bad;
        if (static_cast<size_t>(got[i].rightIndex) != want[i].rightIndex) ++bad;
        if ((got[i].valid != 0u) != (want[i].valid != 0u)) ++bad;
    }
    return bad;
}

size_t tracksDiffering(const std::vector<bincv::Point2f>& wantPts,
                       const std::vector<uint8_t>& wantSt,
                       const std::vector<bincv::Point2f>& gotPts,
                       const std::vector<uint8_t>& gotSt) {
    size_t bad = 0;
    if (wantPts.size() != gotPts.size()) return wantPts.size() + gotPts.size();
    for (size_t i = 0; i < wantPts.size(); ++i) {
        if (wantSt[i] != gotSt[i]) ++bad;
        if (std::memcmp(&wantPts[i].x, &gotPts[i].x, sizeof(float)) != 0) ++bad;
        if (std::memcmp(&wantPts[i].y, &gotPts[i].y, sizeof(float)) != 0) ++bad;
    }
    return bad;
}

const bool kArms[2] = {true, false};

} // namespace

// ===========================================================================
// 1. The displaced-row builder, against the host struct, before any kernel
//    that calls it.
// ===========================================================================
namespace {

__global__ void shiftedRowProbe(bincv::cuda::DeviceBinMatConstView v, long long offLo,
                                unsigned offCount, unsigned wordCount, uint32_t* out) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned total =
        static_cast<unsigned>(v.height) * offCount * wordCount;
    if (idx >= total) return;
    const unsigned i = idx % wordCount;
    const unsigned t = idx / wordCount;
    const unsigned o = t % offCount;
    const unsigned y = t / offCount;
    out[idx] = bincv::cuda::impl::deviceDisplacedRow(v, static_cast<long long>(y),
                                                     offLo + static_cast<long long>(o))
                   .word(i);
}

__global__ void rowSpanProbe(bincv::cuda::DeviceBinMatConstView v, long long baseLo,
                             unsigned baseCount, unsigned pCount, uint32_t* out) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned total = static_cast<unsigned>(v.height) * baseCount * pCount;
    if (idx >= total) return;
    const unsigned p = idx % pCount;
    const unsigned t = idx / pCount;
    const unsigned b = t % baseCount;
    const unsigned y = t / baseCount;
    out[idx] = bincv::cuda::impl::loadRowSpan(v, static_cast<long long>(y),
                                              baseLo + static_cast<long long>(b))
                   .run(p);
}

} // namespace

BINCV_TEST(CudaSparseMatch, DisplacedRowIsTheHostStructOnEveryOffset) {
    // Widths covering "tail mask is all ones" (32, 64), "tail holds 16 pixels"
    // (752, the reference size), a single-word row, and one that is not a
    // multiple of 32 anywhere near a boundary.
    const size_t widths[] = {1, 31, 32, 33, 63, 64, 97, 752};
    size_t bad = 0;
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); ++wi) {
        const size_t w = widths[wi];
        const size_t h = 5;
        const bincv::BinMat<uint32_t> host = randomPlane(w, h, 0x51A0u + w);
        DeviceBinMat dev = uploadPlane(host);
        // Dirty padding on the DEVICE side: the builder masks the trailing
        // partial word through the host's own `sourceWord`, so a dirty plane
        // must read exactly as a clean one.
        dirtyDevicePadding(dev);

        // Offsets sweeping the window wholly off both ends, so every one of
        // the four edge cases fires: left-outside, right-outside, straddling,
        // and the whole window off the plane.
        const long long offLo = -static_cast<long long>(w) - 64;
        const unsigned offCount = static_cast<unsigned>(2 * w + 129);
        const unsigned wordCount = static_cast<unsigned>(
            bincv::impl::minRowWords<uint32_t>(w) + 2);

        const size_t total = h * offCount * wordCount;
        DeviceArray<uint32_t> dOut(total);
        const unsigned block = 256;
        const unsigned grid = static_cast<unsigned>((total + block - 1) / block);
        shiftedRowProbe<<<grid, block>>>(dev.constView(), offLo, offCount, wordCount,
                                         dOut.data());
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<uint32_t> got = downloadVec(dOut, total);

        size_t k = 0;
        for (size_t y = 0; y < h; ++y) {
            for (unsigned o = 0; o < offCount; ++o) {
                const bincv::impl::ReplicatedShiftedRow<uint32_t> row =
                    bincv::impl::displacedRow<uint32_t>(host.plane(0),
                                                        static_cast<long long>(y),
                                                        offLo + static_cast<long long>(o));
                for (unsigned i = 0; i < wordCount; ++i, ++k)
                    if (row.word(i) != got[k]) ++bad;
            }
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSparseMatch, RowSpanIsThreeWordsOfTheSameFunctionOfTheColumn) {
    // The claim RowSpan rests on: `word(i)` is a pure function of the SOURCE
    // COLUMN, so three consecutive calls hold 96 consecutive columns and any
    // funnel-shifted run out of them equals the row built at that shift.
    const size_t widths[] = {1, 31, 32, 33, 64, 97, 752};
    size_t bad = 0;
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); ++wi) {
        const size_t w = widths[wi];
        const size_t h = 4;
        const bincv::BinMat<uint32_t> host = randomPlane(w, h, 0x5A900u + w * 3u);
        DeviceBinMat dev = uploadPlane(host);
        dirtyDevicePadding(dev);

        const long long baseLo = -static_cast<long long>(w) - 40;
        const unsigned baseCount = static_cast<unsigned>(2 * w + 81);
        const unsigned pCount = 64;

        const size_t total = h * baseCount * pCount;
        DeviceArray<uint32_t> dOut(total);
        const unsigned block = 256;
        const unsigned grid = static_cast<unsigned>((total + block - 1) / block);
        rowSpanProbe<<<grid, block>>>(dev.constView(), baseLo, baseCount, pCount,
                                      dOut.data());
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<uint32_t> got = downloadVec(dOut, total);

        size_t k = 0;
        for (size_t y = 0; y < h; ++y) {
            for (unsigned b = 0; b < baseCount; ++b) {
                for (unsigned p = 0; p < pCount; ++p, ++k) {
                    const bincv::impl::ReplicatedShiftedRow<uint32_t> row =
                        bincv::impl::displacedRow<uint32_t>(
                            host.plane(0), static_cast<long long>(y),
                            baseLo + static_cast<long long>(b) + static_cast<long long>(p));
                    if (row.word(0) != got[k]) ++bad;
                }
            }
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
}

// ===========================================================================
// 2. The brute-force matcher
// ===========================================================================
namespace {

struct MatchFixture {
    size_t queryCount;
    size_t trainCount;
    size_t words;
    std::vector<uint32_t> qd, td;
    std::vector<float> qxy, txy;
    std::vector<int32_t> qo, to;
    DeviceArray<uint32_t> dQd, dTd;
    DeviceArray<float> dQxy, dTxy;
    DeviceArray<int32_t> dQo, dTo;
    DeviceArray<bincv::cuda::DeviceDescriptorMatch> dOut;

    MatchFixture(size_t q, size_t t, size_t w, uint64_t seed, int levels = 4)
        : queryCount(q), trainCount(t), words(w),
          qd(atLeastOne(randomDescriptors(q, w, seed))),
          td(atLeastOne(randomDescriptors(t, w, seed ^ 0xBEEFu))),
          qxy(atLeastOne(randomPositions(q, 640.0f, 480.0f, seed + 11u))),
          txy(atLeastOne(randomPositions(t, 640.0f, 480.0f, seed + 29u))),
          qo(atLeastOne(randomOctaves(q, levels, seed + 41u))),
          to(atLeastOne(randomOctaves(t, levels, seed + 53u))),
          dQd(uploadVecNonNull(qd)), dTd(uploadVecNonNull(td)),
          dQxy(uploadVecNonNull(qxy)), dTxy(uploadVecNonNull(txy)),
          dQo(uploadVecNonNull(qo)), dTo(uploadVecNonNull(to)), dOut(q == 0 ? 1 : q) {}

    bincv::cuda::DeviceDescriptorSetConstView query() const {
        return bincv::cuda::descriptorSet(dQd.data(), queryCount, words);
    }
    bincv::cuda::DeviceDescriptorSetConstView train() const {
        return bincv::cuda::descriptorSet(dTd.data(), trainCount, words);
    }
    bincv::cuda::DeviceKeypointSetConstView queryPts(bool withOctave) const {
        return bincv::cuda::keypointSet(dQxy.data(), queryCount,
                                        withOctave ? dQo.data() : nullptr);
    }
    bincv::cuda::DeviceKeypointSetConstView trainPts(bool withOctave) const {
        return bincv::cuda::keypointSet(dTxy.data(), trainCount,
                                        withOctave ? dTo.data() : nullptr);
    }
};

/// Runs a matcher case under both arms and every instantiated tile width, and
/// returns the number of differing fields across all of them.
size_t runMatchCase(MatchFixture& f, bool gated, float maxDx, float maxDy, bool withOctave,
                    int maxOctaveDelta, unsigned maxRatio) {
    std::vector<bincv::DescriptorMatch> want(f.queryCount);
    if (gated) {
        bincv::matchDescriptorsGated<uint32_t>(
            f.qd.data(), f.qxy.data(), f.queryCount, f.td.data(), f.txy.data(), f.trainCount,
            f.words, maxDx, maxDy, want.data(), maxRatio,
            withOctave ? f.qo.data() : nullptr, withOctave ? f.to.data() : nullptr,
            maxOctaveDelta);
    } else {
        bincv::matchDescriptors<uint32_t>(f.qd.data(), f.queryCount, f.td.data(),
                                          f.trainCount, f.words, want.data(), maxRatio);
    }

    size_t bad = 0;
    const unsigned tiles[3] = {1, 4, 8};
    for (size_t a = 0; a < 2; ++a) {
        for (size_t ti = 0; ti < 3; ++ti) {
            bincv::cuda::impl::matchTiledArmEnabled() = kArms[a];
            bincv::cuda::impl::matchQueryTile() = tiles[ti];
            if (f.queryCount != 0)
                cudaMemset(f.dOut.data(), 0xA5,
                           f.queryCount * sizeof(bincv::cuda::DeviceDescriptorMatch));
            const cudaError_t e =
                gated ? bincv::cuda::matchDescriptorsGated(
                            f.query(), f.queryPts(withOctave), f.train(),
                            f.trainPts(withOctave), maxDx, maxDy, f.dOut.data(), maxRatio,
                            maxOctaveDelta)
                      : bincv::cuda::matchDescriptors(f.query(), f.train(), f.dOut.data(),
                                                      maxRatio);
            if (e != cudaSuccess) return 1000;
            if (cudaDeviceSynchronize() != cudaSuccess) return 1000;
            bad += matchesDiffering(want, downloadVec(f.dOut, f.queryCount));
            // The reference arm is instantiated once per tile width only
            // because the switch is read at launch; one pass with it off is
            // all the information there is, and the loop keeps the shapes
            // symmetric rather than special-casing.
        }
    }
    bincv::cuda::impl::matchTiledArmEnabled() = true;
    bincv::cuda::impl::matchQueryTile() = bincv::cuda::impl::kMatchDefaultTile;
    return bad;
}

} // namespace

BINCV_TEST(CudaSparseMatch, BruteForceMatcherAcrossShapesRatiosAndWidths) {
    const size_t shapes[][2] = {{1, 0}, {1, 1}, {2, 2}, {7, 13}, {33, 1000}, {470, 470}};
    const size_t wordSweep[] = {4, 8, 16};
    const unsigned ratios[] = {0, 75, 80, 100, 255};
    size_t bad = 0;
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); ++si) {
        for (size_t wi = 0; wi < 3; ++wi) {
            MatchFixture f(shapes[si][0], shapes[si][1], wordSweep[wi],
                           0x1000u + si * 97u + wi * 13u);
            for (size_t ri = 0; ri < 5; ++ri)
                bad += runMatchCase(f, false, 0.0f, 0.0f, false, 0, ratios[ri]);
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSparseMatch, TiesKeepTheFirstTrainIndexAndEqualDistancesTurnOnTheRatio) {
    // Several train entries SHARE the minimum, which is what the packed-min
    // trick claims to resolve the host's way; and a set where every distance
    // is equal, where `second == best` and `valid` rests on the ratio
    // arithmetic alone.
    const size_t words = 8;
    const size_t q = 4, t = 16;
    MatchFixture f(q, t, words, 0x7135u);
    // Make train entries 3, 7 and 11 identical to query 0, and every other
    // train entry identical to train entry 0.
    for (size_t k = 0; k < words; ++k) {
        f.td[3 * words + k] = f.qd[k];
        f.td[7 * words + k] = f.qd[k];
        f.td[11 * words + k] = f.qd[k];
    }
    BINCV_CHECK_EQ(cudaMemcpy(f.dTd.data(), f.td.data(), f.td.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   cudaSuccess);
    size_t bad = 0;
    const unsigned ratios[] = {0, 80, 100, 255};
    for (size_t ri = 0; ri < 4; ++ri)
        bad += runMatchCase(f, false, 0.0f, 0.0f, false, 0, ratios[ri]);

    // All distances equal: every train descriptor the same.
    MatchFixture g(q, t, words, 0x7136u);
    for (size_t j = 1; j < t; ++j)
        for (size_t k = 0; k < words; ++k) g.td[j * words + k] = g.td[k];
    BINCV_CHECK_EQ(cudaMemcpy(g.dTd.data(), g.td.data(), g.td.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   cudaSuccess);
    for (size_t ri = 0; ri < 4; ++ri)
        bad += runMatchCase(g, false, 0.0f, 0.0f, false, 0, ratios[ri]);
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSparseMatch, AnEmptyTrainSetWritesTheHostsDefaultRecordNotTheSentinelIndex) {
    // The write-back trap: with nothing admitted the host leaves
    // `trainIndex == 0`, while a packed sentinel minimum carries
    // 0xFFFFFFFF in its low half. A caller who ignores `valid` must not find
    // a match against train descriptor 4,294,967,295.
    MatchFixture f(5, 0, 8, 0x4444u);
    BINCV_CHECK_EQ(runMatchCase(f, false, 0.0f, 0.0f, false, 0, 80), 0u);
    const std::vector<bincv::cuda::DeviceDescriptorMatch> got = downloadVec(f.dOut, 5);
    size_t bad = 0;
    for (size_t i = 0; i < 5; ++i) {
        if (got[i].trainIndex != 0u) ++bad;
        if (got[i].valid != 0u) ++bad;
    }
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSparseMatch, GatedMatcherAcrossWindowsOctavesAndBands) {
    const size_t shapes[][2] = {{1, 0}, {1, 1}, {2, 2}, {7, 13}, {33, 1000}, {470, 470}};
    const float windows[] = {0.0f, 1.0f, 48.0f, 1.0e9f};
    const int deltas[] = {0, 1, 3};
    size_t bad = 0;
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); ++si) {
        MatchFixture f(shapes[si][0], shapes[si][1], 8, 0x2000u + si * 31u);
        for (size_t wi = 0; wi < 4; ++wi) {
            for (size_t oi = 0; oi < 2; ++oi) {
                const bool withOctave = oi == 1;
                for (size_t di = 0; di < 3; ++di)
                    bad += runMatchCase(f, true, windows[wi], windows[wi], withOctave,
                                        deltas[di], 75);
            }
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSparseMatch, AnUnboundedGateReproducesBruteForceExactly) {
    // The structural check that the gate is the ONLY difference between the
    // two kernels: with a window nothing can fall outside and no octave
    // arrays, the gated arm must equal the ungated one field for field.
    MatchFixture f(97, 233, 8, 0x3131u);
    std::vector<bincv::DescriptorMatch> plain(f.queryCount);
    bincv::matchDescriptors<uint32_t>(f.qd.data(), f.queryCount, f.td.data(), f.trainCount,
                                      f.words, plain.data(), 80);
    size_t bad = 0;
    for (size_t a = 0; a < 2; ++a) {
        bincv::cuda::impl::matchTiledArmEnabled() = kArms[a];
        BINCV_CHECK_EQ(bincv::cuda::matchDescriptorsGated(f.query(), f.queryPts(false),
                                                          f.train(), f.trainPts(false),
                                                          1.0e30f, 1.0e30f, f.dOut.data(), 80),
                       cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        bad += matchesDiffering(plain, downloadVec(f.dOut, f.queryCount));
    }
    bincv::cuda::impl::matchTiledArmEnabled() = true;
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSparseMatch, ADescriptorWiderThanTheTileFallsToTheReferenceArm) {
    // The fast path's OWN gate: a 2048-bit descriptor (64 words) is wider than
    // the shared-memory staging accepts, so both switch positions run the same
    // kernel and must give the same answer. This is the input the benchmark
    // times for its ~1.00x control.
    const size_t words = bincv::cuda::impl::kMatchTileMaxWords + 1;
    MatchFixture f(40, 90, words, 0x9999u);
    BINCV_CHECK(!(words <= bincv::cuda::impl::kMatchTileMaxWords));
    BINCV_CHECK_EQ(runMatchCase(f, false, 0.0f, 0.0f, false, 0, 80), 0u);
    BINCV_CHECK_EQ(runMatchCase(f, true, 48.0f, 48.0f, true, 1, 75), 0u);
}

// ===========================================================================
// 3. Sparse stereo -- the coarse descriptor stage
// ===========================================================================
namespace {

size_t runStereoCoarse(size_t leftCount, size_t rightCount, size_t words, uint64_t seed,
                       const bincv::StereoMatchParams& params) {
    const std::vector<uint32_t> ld = randomDescriptors(leftCount, words, seed);
    const std::vector<uint32_t> rd = randomDescriptors(rightCount, words, seed ^ 0x1234u);
    const std::vector<float> lxy = randomPositions(leftCount, 752.0f, 480.0f, seed + 7u);
    // Right positions deliberately CLOSE to the left ones, so the row band and
    // the disparity range actually admit candidates rather than rejecting
    // everything and testing nothing.
    std::vector<float> rxy(2 * rightCount);
    uint64_t s = seed + 19u;
    for (size_t j = 0; j < rightCount; ++j) {
        const size_t src = leftCount == 0 ? 0 : j % leftCount;
        const float bx = leftCount == 0 ? 300.0f : lxy[2 * src];
        const float by = leftCount == 0 ? 200.0f : lxy[2 * src + 1];
        rxy[2 * j] = bx - static_cast<float>(splitmix(s) % 70u);
        rxy[2 * j + 1] = by + static_cast<float>(splitmix(s) % 7u) - 3.0f;
    }

    std::vector<bincv::StereoMatch> want(leftCount);
    bincv::stereoDescriptorMatch<uint32_t>(lxy.data(), leftCount, ld.data(), rxy.data(),
                                           rightCount, rd.data(), words, want.data(), params);

    DeviceArray<uint32_t> dLd(uploadVec(ld)), dRd(uploadVec(rd));
    DeviceArray<float> dLxy(uploadVec(lxy)), dRxy(uploadVec(rxy));
    DeviceArray<bincv::cuda::DeviceStereoMatch> dOut(leftCount == 0 ? 1 : leftCount);

    size_t bad = 0;
    for (size_t a = 0; a < 2; ++a) {
        bincv::cuda::impl::sparseStereoFastArmEnabled() = kArms[a];
        if (leftCount != 0)
            cudaMemset(dOut.data(), 0x5A,
                       leftCount * sizeof(bincv::cuda::DeviceStereoMatch));
        const cudaError_t e = bincv::cuda::stereoDescriptorMatch(
            bincv::cuda::keypointSet(dLxy.data(), leftCount),
            bincv::cuda::descriptorSet(dLd.data(), leftCount, words),
            bincv::cuda::keypointSet(dRxy.data(), rightCount),
            bincv::cuda::descriptorSet(dRd.data(), rightCount, words), dOut.data(), params);
        if (e != cudaSuccess) return 1000;
        if (cudaDeviceSynchronize() != cudaSuccess) return 1000;
        bad += stereoDiffering(want, downloadVec(dOut, leftCount));
    }
    bincv::cuda::impl::sparseStereoFastArmEnabled() = true;
    return bad;
}

} // namespace

BINCV_TEST(CudaSparseMatch, StereoCoarseStageAcrossCountsTolerancesAndThresholds) {
    const size_t counts[][2] = {{0, 0}, {1, 0}, {1, 1}, {5, 1}, {40, 90}, {500, 500}};
    const int tolerances[] = {0, 2, 5};
    const unsigned hammings[] = {0, 100, 256};
    size_t bad = 0;
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ++ci) {
        for (size_t ti = 0; ti < 3; ++ti) {
            for (size_t hi = 0; hi < 3; ++hi) {
                bincv::StereoMatchParams p;
                p.rowTolerance = tolerances[ti];
                p.maxHamming = hammings[hi];
                bad += runStereoCoarse(counts[ci][0], counts[ci][1], 8,
                                       0x5000u + ci * 17u + ti * 5u + hi, p);
            }
        }
    }
    // A range that admits nothing at all.
    bincv::StereoMatchParams narrow;
    narrow.minDisparity = 900;
    narrow.maxDisparity = 1000;
    bad += runStereoCoarse(60, 60, 8, 0x5100u, narrow);
    // A descriptor wider than the tile: the fast arm's own gate excludes it.
    bincv::StereoMatchParams wide;
    bad += runStereoCoarse(60, 60, bincv::cuda::impl::kMatchTileMaxWords + 1, 0x5200u, wide);
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSparseMatch, RaisingMaxHammingToAcceptEverythingStillReportsNoCandidate) {
    // A caller who sets maxHamming to its ceiling must still get "no
    // candidate" where the gate admitted none -- never a match fabricated from
    // right keypoint 0. The range below admits nothing, so every record must
    // come back invalid at the most permissive threshold there is.
    bincv::StereoMatchParams p;
    p.minDisparity = 900;
    p.maxDisparity = 1000;
    p.maxHamming = 0xFFFFFFFEu;
    BINCV_CHECK_EQ(runStereoCoarse(64, 64, 8, 0x5300u, p), 0u);
}

// ===========================================================================
// 4. Sparse stereo -- the refinement stage
// ===========================================================================
namespace {

/// One refinement case: the keypoints and the seed disparities are given, so
/// borders and out-of-range initial values can be placed exactly.
size_t runStereoRefine(size_t w, size_t h, const std::vector<float>& xy,
                       const std::vector<bincv::StereoMatch>& seed,
                       const bincv::StereoMatchParams& params, uint64_t planeSeed,
                       bool dirtyPadding) {
    const size_t count = seed.size();
    const bincv::BinMat<uint32_t> left = randomPlane(w, h, planeSeed);
    const bincv::BinMat<uint32_t> right = randomPlane(w, h, planeSeed ^ 0xABCDu);

    std::vector<bincv::StereoMatch> want = seed;
    bincv::stereoRefineDisparity<uint32_t>(left.plane(0), right.plane(0), xy.data(), count,
                                           want.data(), params);

    DeviceBinMat dLeft = uploadPlane(left);
    DeviceBinMat dRight = uploadPlane(right);
    if (dirtyPadding) {
        dirtyDevicePadding(dLeft);
        dirtyDevicePadding(dRight);
    }
    DeviceArray<float> dXy(uploadVec(xy));

    std::vector<bincv::cuda::DeviceStereoMatch> seedDev(count);
    for (size_t i = 0; i < count; ++i) {
        seedDev[i].disparity = seed[i].disparity;
        seedDev[i].distance = seed[i].distance;
        seedDev[i].rightIndex = static_cast<uint32_t>(seed[i].rightIndex);
        seedDev[i].valid = seed[i].valid;
    }

    size_t bad = 0;
    for (size_t a = 0; a < 2; ++a) {
        bincv::cuda::impl::sparseStereoFastArmEnabled() = kArms[a];
        DeviceArray<bincv::cuda::DeviceStereoMatch> dIo(uploadVec(seedDev));
        const cudaError_t e = bincv::cuda::stereoRefineDisparity(
            dLeft.constView(), dRight.constView(),
            bincv::cuda::keypointSet(dXy.data(), count), dIo.data(), params);
        if (e != cudaSuccess) return 1000;
        if (cudaDeviceSynchronize() != cudaSuccess) return 1000;
        bad += stereoDiffering(want, downloadVec(dIo, count));
    }
    bincv::cuda::impl::sparseStereoFastArmEnabled() = true;
    return bad;
}

/// Keypoints placed to hit every border case, plus a scatter of interior ones.
void borderKeypoints(size_t w, size_t h, std::vector<float>& xy,
                     std::vector<bincv::StereoMatch>& seed, uint64_t s) {
    const float fw = static_cast<float>(w), fh = static_cast<float>(h);
    const float xs[] = {0.0f, 1.5f, fw * 0.5f, fw - 2.0f, fw + 40.0f, -40.0f};
    const float ys[] = {0.0f, 2.5f, fh * 0.5f, fh - 1.0f, fh + 25.0f, -25.0f};
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            xy.push_back(xs[i]);
            xy.push_back(ys[j]);
            bincv::StereoMatch m;
            m.disparity = static_cast<float>(splitmix(s) % 70u);
            m.distance = 40;
            m.rightIndex = 3;
            m.valid = 1;
            seed.push_back(m);
        }
    }
    // A record that arrives INVALID: the refinement must leave it untouched.
    xy.push_back(fw * 0.25f);
    xy.push_back(fh * 0.25f);
    bincv::StereoMatch dead;
    dead.disparity = 12.5f;
    dead.valid = 0;
    seed.push_back(dead);
}

} // namespace

BINCV_TEST(CudaSparseMatch, StereoRefinementAcrossBordersRadiiAndSubPixel) {
    const size_t widths[] = {64, 97, 752};
    const int radii[] = {1, 4, 8};
    size_t bad = 0;
    for (size_t wi = 0; wi < 3; ++wi) {
        const size_t w = widths[wi];
        const size_t h = 61;
        std::vector<float> xy;
        std::vector<bincv::StereoMatch> seed;
        borderKeypoints(w, h, xy, seed, 0x6000u + wi);
        for (size_t ri = 0; ri < 3; ++ri) {
            for (size_t sp = 0; sp < 2; ++sp) {
                bincv::StereoMatchParams p;
                p.refineRadius = radii[ri];
                p.subPixel = sp == 0;
                bad += runStereoRefine(w, h, xy, seed, p, 0x6100u + wi * 7u + ri, false);
                bad += runStereoRefine(w, h, xy, seed, p, 0x6100u + wi * 7u + ri, true);
            }
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSparseMatch, StereoRefinementHandlesAnEmptyCandidateRangeAndANegativeMinimum) {
    const size_t w = 97, h = 53;
    std::vector<float> xy;
    std::vector<bincv::StereoMatch> seed;
    borderKeypoints(w, h, xy, seed, 0x6200u);
    size_t bad = 0;

    // d0 pinned at both ends of the range, so `lo > hi` is reachable.
    bincv::StereoMatchParams high;
    high.minDisparity = 60;
    high.maxDisparity = 64;
    bad += runStereoRefine(w, h, xy, seed, high, 0x6300u, false);

    // A NEGATIVE minimum -- legal here, and the case the packed argmin would
    // get wrong if it packed `d` instead of `d - lo`.
    bincv::StereoMatchParams neg;
    neg.minDisparity = -40;
    neg.maxDisparity = 20;
    bad += runStereoRefine(w, h, xy, seed, neg, 0x6400u, false);
    bad += runStereoRefine(w, h, xy, seed, neg, 0x6400u, true);

    // A window the funnel-shift arm's own span bound rejects: both switch
    // positions then run the reference kernel.
    bincv::StereoMatchParams wide;
    wide.winWidth = 41;
    wide.winHeight = 41;
    wide.refineRadius = 8;
    BINCV_CHECK(!bincv::cuda::impl::windowSpanFits(wide.winWidth, wide.winHeight,
                                                   2 * wide.refineRadius + 2));
    bad += runStereoRefine(w, h, xy, seed, wide, 0x6500u, false);
    BINCV_CHECK_EQ(bad, 0u);
}

// ===========================================================================
// 5. Block matching
// ===========================================================================
namespace {

struct Ladder {
    std::vector<bincv::BinMat<uint32_t>> prev, next;
    std::vector<DeviceBinMat> dPrev, dNext;
    std::vector<bincv::BlockMatchLevel<uint32_t>> hostLevels;
    std::vector<bincv::cuda::DeviceBlockMatchLevel> devLevels;

    Ladder(size_t w, size_t h, size_t levels, uint64_t seed, bool dirty) {
        size_t lw = w, lh = h;
        for (size_t i = 0; i < levels; ++i) {
            prev.push_back(randomPlane(lw, lh, seed + i * 31u));
            next.push_back(randomPlane(lw, lh, seed + i * 31u + 7u));
            lw = lw > 1 ? (lw + 1) / 2 : 1;
            lh = lh > 1 ? (lh + 1) / 2 : 1;
        }
        for (size_t i = 0; i < levels; ++i) {
            dPrev.push_back(uploadPlane(prev[i]));
            dNext.push_back(uploadPlane(next[i]));
            if (dirty) {
                dirtyDevicePadding(dPrev.back());
                dirtyDevicePadding(dNext.back());
            }
        }
        for (size_t i = 0; i < levels; ++i) {
            bincv::BlockMatchLevel<uint32_t> hl;
            hl.prev = prev[i].plane(0);
            hl.next = next[i].plane(0);
            hostLevels.push_back(hl);
            bincv::cuda::DeviceBlockMatchLevel dl;
            dl.prev = dPrev[i].constView();
            dl.next = dNext[i].constView();
            devLevels.push_back(dl);
        }
    }
};

size_t runBlockMatch(size_t w, size_t h, size_t levels, const std::vector<bincv::Point2f>& pts,
                     const bincv::BlockMatchParams& params, uint64_t seed, bool dirty) {
    Ladder L(w, h, levels, seed, dirty);
    const size_t n = pts.size();

    std::vector<bincv::Point2f> wantPts(n == 0 ? 1 : n);
    std::vector<uint8_t> wantSt(n == 0 ? 1 : n);
    bincv::calcOpticalFlowBlockMatch<uint32_t>(L.hostLevels.data(), levels, pts.data(),
                                               wantPts.data(), wantSt.data(), n, params);
    wantPts.resize(n);
    wantSt.resize(n);

    DeviceArray<bincv::Point2f> dPrev(uploadVec(pts));
    DeviceArray<bincv::Point2f> dNext(n == 0 ? 1 : n);
    DeviceArray<uint8_t> dStatus(n == 0 ? 1 : n);
    DeviceArray<uint8_t> dScratch(bincv::cuda::blockMatchScratchBytes(n == 0 ? 1 : n));

    size_t bad = 0;
    for (size_t a = 0; a < 2; ++a) {
        bincv::cuda::impl::blockMatchFastArmEnabled() = kArms[a];
        if (n != 0) {
            cudaMemset(dNext.data(), 0x3C, n * sizeof(bincv::Point2f));
            cudaMemset(dStatus.data(), 0x3C, n);
        }
        const cudaError_t e = bincv::cuda::calcOpticalFlowBlockMatch(
            L.devLevels.data(), levels, dPrev.data(), dNext.data(), dStatus.data(), n,
            dScratch.data(), dScratch.size(), params);
        if (e != cudaSuccess) return 1000;
        if (cudaDeviceSynchronize() != cudaSuccess) return 1000;
        bad += tracksDiffering(wantPts, wantSt, downloadVec(dNext, n), downloadVec(dStatus, n));
    }
    bincv::cuda::impl::blockMatchFastArmEnabled() = true;
    return bad;
}

std::vector<bincv::Point2f> trackPoints(size_t w, size_t h, size_t n, uint64_t s) {
    std::vector<bincv::Point2f> pts;
    const float fw = static_cast<float>(w), fh = static_cast<float>(h);
    // The four borders and two positions outside the frame, first, so a small
    // point count still exercises the clipping and the loss rule.
    const float bx[] = {0.5f, fw - 1.5f, fw * 0.5f, fw * 0.5f, -30.0f, fw + 30.0f};
    const float by[] = {fh * 0.5f, fh * 0.5f, 0.5f, fh - 1.5f, fh * 0.5f, fh * 0.5f};
    for (size_t i = 0; i < 6 && pts.size() < n; ++i) pts.push_back({bx[i], by[i]});
    while (pts.size() < n) {
        bincv::Point2f p;
        p.x = static_cast<float>(splitmix(s) % 100000u) * (fw / 100000.0f);
        p.y = static_cast<float>(splitmix(s) % 100000u) * (fh / 100000.0f);
        pts.push_back(p);
    }
    return pts;
}

} // namespace

BINCV_TEST(CudaSparseMatch, BlockMatchingAcrossCountsLevelsRadiiAndSubPixel) {
    const size_t widths[] = {64, 97, 752};
    const size_t counts[] = {0, 1, 500};
    const size_t levelSweep[] = {0, 1, 4};
    const int radii[] = {1, 2, 4};
    size_t bad = 0;
    for (size_t wi = 0; wi < 3; ++wi) {
        const size_t w = widths[wi];
        const size_t h = 97;
        for (size_t ci = 0; ci < 3; ++ci) {
            const std::vector<bincv::Point2f> pts =
                trackPoints(w, h, counts[ci], 0x7000u + wi * 13u + ci);
            for (size_t li = 0; li < 3; ++li) {
                for (size_t ri = 0; ri < 3; ++ri) {
                    bincv::BlockMatchParams p;
                    p.searchRadius = radii[ri];
                    p.subPixel = (ri % 2) == 0;
                    // A 31x31 window over a 64-wide frame caps the ladder at
                    // level 0, which is the `usableLevels` rule; the 752-wide
                    // sweep is where four levels are really used.
                    bad += runBlockMatch(w, h, levelSweep[li], pts, p,
                                         0x7100u + wi * 7u + ci * 3u + li, false);
                }
            }
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSparseMatch, BlockMatchingOnADirtyPlaneAndThroughItsOwnGate) {
    const size_t w = 97, h = 97;
    const std::vector<bincv::Point2f> pts = trackPoints(w, h, 64, 0x7200u);
    size_t bad = 0;
    bincv::BlockMatchParams p;
    bad += runBlockMatch(w, h, 3, pts, p, 0x7300u, true);

    // A window the funnel-shift arm's span bound rejects: both switch
    // positions run the reference kernel and must agree with the host.
    bincv::BlockMatchParams wide;
    wide.winWidth = 48;
    wide.winHeight = 48;
    wide.searchRadius = 2;
    BINCV_CHECK(!bincv::cuda::impl::windowSpanFits(wide.winWidth, wide.winHeight,
                                                   2 * wide.searchRadius));
    bad += runBlockMatch(w, h, 2, pts, wide, 0x7400u, false);

    // And a window the arm DOES accept, so the check above is not vacuous.
    BINCV_CHECK(bincv::cuda::impl::windowSpanFits(31, 31, 4));
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSparseMatch, BlockMatchingRefusesOutsideItsDocumentedDomain) {
    // A device op may accept a narrower domain than its host twin if it NAMES
    // it, ASSERTS it and RETURNS AN ERROR outside it. Both refusals below are
    // asserted first -- deliberate domain violations; see
    // BINCV_CHECK_EQ_UNLESS_CHECKED.
    const size_t w = 64, h = 64;
    const std::vector<bincv::Point2f> pts = trackPoints(w, h, 8, 0x7500u);
    Ladder L(w, h, 2, 0x7600u, false);
    DeviceArray<bincv::Point2f> dPrev(uploadVec(pts));
    DeviceArray<bincv::Point2f> dNext(pts.size());
    DeviceArray<uint8_t> dStatus(pts.size());
    DeviceArray<uint8_t> dScratch(bincv::cuda::blockMatchScratchBytes(pts.size()));

    bincv::BlockMatchParams tooWide;
    tooWide.searchRadius = bincv::cuda::kMaxBlockMatchRadius + 1;
    BINCV_CHECK_EQ_UNLESS_CHECKED(bincv::cuda::calcOpticalFlowBlockMatch(
                                      L.devLevels.data(), 2, dPrev.data(), dNext.data(),
                                      dStatus.data(), pts.size(), dScratch.data(),
                                      dScratch.size(), tooWide),
                                  cudaErrorInvalidValue);

    // Too few scratch bytes for the arm that needs them.
    bincv::BlockMatchParams ok;
    BINCV_CHECK_EQ_UNLESS_CHECKED(bincv::cuda::calcOpticalFlowBlockMatch(
                                      L.devLevels.data(), 2, dPrev.data(), dNext.data(),
                                      dStatus.data(), pts.size(), dScratch.data(), 8, ok),
                                  cudaErrorInvalidValue);
}

BINCV_TEST(CudaSparseMatch, TheScratchFormulaIsTheStructsOwnSize) {
    BINCV_CHECK_EQ(bincv::cuda::blockMatchScratchBytes(500),
                   500 * sizeof(bincv::cuda::DeviceBlockMatchState));
    BINCV_CHECK_EQ(sizeof(bincv::cuda::DeviceBlockMatchState), 16u);
}

// ===========================================================================
// 6. The composition
// ===========================================================================

BINCV_TEST(CudaSparseMatch, StereoMatchRectifiedIsTheTwoStagesAndNothingElse) {
    const size_t w = 752, h = 97, n = 200, words = 8;
    const bincv::BinMat<uint32_t> left = randomPlane(w, h, 0x8100u);
    const bincv::BinMat<uint32_t> right = randomPlane(w, h, 0x8200u);
    const std::vector<uint32_t> ld = randomDescriptors(n, words, 0x8300u);
    const std::vector<uint32_t> rd = randomDescriptors(n, words, 0x8400u);
    const std::vector<float> lxy = randomPositions(n, static_cast<float>(w),
                                                   static_cast<float>(h), 0x8500u);
    std::vector<float> rxy(2 * n);
    uint64_t s = 0x8600u;
    for (size_t j = 0; j < n; ++j) {
        rxy[2 * j] = lxy[2 * j] - static_cast<float>(splitmix(s) % 60u);
        rxy[2 * j + 1] = lxy[2 * j + 1] + static_cast<float>(splitmix(s) % 5u) - 2.0f;
    }

    bincv::StereoMatchParams params;
    std::vector<bincv::StereoMatch> want(n);
    bincv::stereoMatchRectified<uint32_t>(left.plane(0), right.plane(0), lxy.data(), n,
                                          ld.data(), rxy.data(), n, rd.data(), words,
                                          want.data(), params);

    DeviceBinMat dLeft = uploadPlane(left);
    DeviceBinMat dRight = uploadPlane(right);
    DeviceArray<uint32_t> dLd(uploadVec(ld)), dRd(uploadVec(rd));
    DeviceArray<float> dLxy(uploadVec(lxy)), dRxy(uploadVec(rxy));
    DeviceArray<bincv::cuda::DeviceStereoMatch> dOut(n);

    size_t bad = 0;
    for (size_t a = 0; a < 2; ++a) {
        bincv::cuda::impl::sparseStereoFastArmEnabled() = kArms[a];
        BINCV_CHECK_EQ(bincv::cuda::stereoMatchRectified(
                           dLeft.constView(), dRight.constView(),
                           bincv::cuda::keypointSet(dLxy.data(), n),
                           bincv::cuda::descriptorSet(dLd.data(), n, words),
                           bincv::cuda::keypointSet(dRxy.data(), n),
                           bincv::cuda::descriptorSet(dRd.data(), n, words), dOut.data(),
                           params),
                       cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        bad += stereoDiffering(want, downloadVec(dOut, n));
    }
    bincv::cuda::impl::sparseStereoFastArmEnabled() = true;
    BINCV_CHECK_EQ(bad, 0u);
}

// ===========================================================================

namespace {
bool cudaDevicePresent() {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return false;
    }
    return true;
}
} // namespace

#if BINCV_TEST_WITH_GTEST
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    const int summaryRc = ::bincv::test::summarize("CUDA sparse match/stereo tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA sparse match/stereo tests", argc, argv);
}
#endif
