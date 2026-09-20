// The device Lucas-Kanade tracker against its HOST twin, bit for bit.
//
// WHY EQUALITY RATHER THAN A TOLERANCE, BECAUSE IT IS NOT A STYLE CHOICE
//
// LK's float layer is O(iterations) per keypoint, never O(pixels), so exactness
// costs almost nothing to buy -- and it is worth a great deal, because the
// epsilon test and the oscillation test are FLOAT BRANCHES. A one-ulp
// disagreement changes a point's iteration count and can move its endpoint by
// PIXELS. A tolerance on `nextPts` would therefore have no stable meaning: it
// would have to be wide enough to admit a different number of Gauss-Newton
// steps, at which point it admits a wrong answer too. So `nextPts` is compared
// as raw 32-bit WORDS, `status` as bytes, `err` as words.
//
// FOUR LAYERS, DELIBERATELY, BECAUSE A WHOLE-TRACKER COMPARISON IS A POOR ORACLE
// ON ITS OWN
//
//  1. THE EXACT-INTEGER LAYER -- lkResidualSumsProbeAsync against the host's
//     impl::residualSums, and lkCovarianceProbeAsync against BOTH the host's
//     gradientCovariance AND this backend's gradientCovarianceBatchAsync. These
//     ten (and three) numbers are INTEGERS, so equality is a real check and no
//     float difference can hide inside it.
//  2. THE WHOLE TRACKER, over a parameter sweep and a keypoint set built to hit
//     every exit the algorithm has.
//  3. THE ARMS -- every combination of the two runtime switches, on the same
//     inputs, in ONE binary, required to produce byte-identical output. And
//     because byte-identical output between two arms CANNOT distinguish "the arm
//     ran" from "the arm was compiled out", cuda::lkPathName() is asserted too:
//     it reports the arm a launch actually takes, BY TAKING IT.
//  4. THE DOMAIN REFUSALS, so a later relaxation cannot quietly become a wrong
//     answer, and the FMA GUARD, so the build flags the equality claim rests on
//     cannot be lost in silence.
//
// A .cpp rather than a .cu, for test_cuda_frontend_corner's reason: the oracle
// here is the HOST tracker, ops/opticalFlow.hpp gates its AVX2 keypoint batch off
// under __CUDACC__, and a host arm compiled by nvcc is not the host arm a caller
// runs. This translation unit is also compiled -ffp-contract=off, which is half
// of what case 4 pins.
//
// Exits 77 when no CUDA device is present, like the other suites here.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/core/types.hpp"
#include "bincv/cuda/covariance.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/opticalFlow.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/covariance.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/opticalFlow.hpp"
#include "bincv/quantMat.hpp"
#include "test_util.hpp"

namespace {

namespace bc = bincv::cuda;
using bincv::BinMatConstView;
using bincv::GradientCovariance;
using bincv::LKEntryLevel;
using bincv::LKParams;
using bincv::Point2f;
using bincv::QuantMat;
using bincv::Rect;
using bincv::SignedQuantMat;

uint64_t splitmix(uint64_t s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

template <size_t N>
BinMatConstView<uint32_t> stackOf(const QuantMat<N, uint32_t>& m) {
    return BinMatConstView<uint32_t>(m.data(), m.getWidth(), N * m.getHeight(),
                                     m.getAlignedWidth());
}
template <size_t N>
BinMatConstView<uint32_t> signedStackOf(const SignedQuantMat<N, uint32_t>& m) {
    return BinMatConstView<uint32_t>(m.data(), m.getWidth(), (N + 1) * m.getHeight(),
                                     m.getAlignedWidth());
}

/// A level with STRUCTURE in it. Pure noise makes every pixel an edge and a flat
/// frame makes none; both make a gradient covariance degenerate, and a tracker
/// measured on either is measured on the one case it can do nothing with.
/// `flatBox` is a deliberately featureless rectangle, which is how loss rule 2
/// gets exercised at all.
template <size_t N>
QuantMat<N, uint32_t> makeLevel(size_t w, size_t h, uint64_t seed, bool flatBox) {
    QuantMat<N, uint32_t> m(static_cast<int>(w), static_cast<int>(h));
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            const uint64_t bx = static_cast<uint64_t>(x / 5);
            const uint64_t by = static_cast<uint64_t>(y / 4);
            unsigned v = static_cast<unsigned>(
                splitmix(seed ^ (bx * 0x9E3779B1ULL) ^ (by * 0x85EBCA77ULL)) %
                (1ull << N));
            if (flatBox && x >= w / 2 && x + 40 < w && x >= w / 2 && x < w / 2 + 40 &&
                y >= h / 2 && y < h / 2 + 40) {
                v = 0u;
            }
            m.set(static_cast<int>(y), static_cast<int>(x), v);
        }
    }
    return m;
}

/// The same field displaced by (dx, dy) with replicate at the border -- a frame
/// pair a tracker can actually converge on, which is what makes the iteration
/// count, the epsilon test and the oscillation test run at all rather than
/// thrashing for twenty iterations on noise.
template <size_t N>
QuantMat<N, uint32_t> shiftLevel(const QuantMat<N, uint32_t>& src, int dx, int dy) {
    const size_t w = src.getWidth(), h = src.getHeight();
    QuantMat<N, uint32_t> out(static_cast<int>(w), static_cast<int>(h));
    for (size_t y = 0; y < h; ++y) {
        long long sy = static_cast<long long>(y) - dy;
        if (sy < 0) sy = 0;
        if (sy >= static_cast<long long>(h)) sy = static_cast<long long>(h) - 1;
        for (size_t x = 0; x < w; ++x) {
            long long sx = static_cast<long long>(x) - dx;
            if (sx < 0) sx = 0;
            if (sx >= static_cast<long long>(w)) sx = static_cast<long long>(w) - 1;
            out.set(static_cast<int>(y), static_cast<int>(x),
                    src.at(static_cast<int>(sy), static_cast<int>(sx)));
        }
    }
    return out;
}

/// Sets every bit at or past `width` in each row's trailing word. The padding-bit
/// invariant says no reduction may count one; this is how the test asks from the
/// CONSUMER's side rather than trusting the producer.
void dirtyPadding(uint32_t* data, size_t width, size_t rows, size_t stride) {
    const size_t words = (width + 31) / 32;
    if (width % 32 == 0) return;  // no padding bits exist to dirty
    const uint32_t dirt = ~((uint32_t{1} << (width % 32)) - 1u);
    for (size_t y = 0; y < rows; ++y) data[y * stride + words - 1] |= dirt;
}

/// One level's host containers, its device copies, and the two views of it the
/// two backends take. Owns its device memory for the life of the case.
template <size_t N>
struct Level {
    QuantMat<N, uint32_t> prev, next;
    SignedQuantMat<N, uint32_t> dx, dy;
    bc::DeviceBinMat dPrev, dNext, dDx, dDy;

    Level(size_t w, size_t h, uint64_t seed, int shiftX, int shiftY, bool flatBox)
        : prev(makeLevel<N>(w, h, seed, flatBox)),
          next(shiftLevel<N>(prev, shiftX, shiftY)),
          dx(static_cast<int>(w), static_cast<int>(h)),
          dy(static_cast<int>(w), static_cast<int>(h)),
          dPrev(static_cast<int>(w), static_cast<int>(N * h)),
          dNext(static_cast<int>(w), static_cast<int>(N * h)),
          dDx(static_cast<int>(w), static_cast<int>((N + 1) * h)),
          dDy(static_cast<int>(w), static_cast<int>((N + 1) * h)) {
        bincv::derivativeX<N, uint32_t>(prev, dx);
        bincv::derivativeY<N, uint32_t>(prev, dy);
        upload();
    }

    /// Dirties the padding of every plane on BOTH sides, so the two backends see
    /// the same dirt and both must still give the clean plane's answer.
    void dirty() {
        dirtyPadding(prev.data(), prev.getWidth(), N * prev.getHeight(),
                     prev.getAlignedWidth());
        dirtyPadding(next.data(), next.getWidth(), N * next.getHeight(),
                     next.getAlignedWidth());
        dirtyPadding(dx.data(), dx.getWidth(), (N + 1) * dx.getHeight(),
                     dx.getAlignedWidth());
        dirtyPadding(dy.data(), dy.getWidth(), (N + 1) * dy.getHeight(),
                     dy.getAlignedWidth());
        upload();
    }

    void upload() {
        BINCV_CHECK_EQ(bc::upload(stackOf(prev), dPrev.view()), cudaSuccess);
        BINCV_CHECK_EQ(bc::upload(stackOf(next), dNext.view()), cudaSuccess);
        BINCV_CHECK_EQ(bc::upload(signedStackOf(dx), dDx.view()), cudaSuccess);
        BINCV_CHECK_EQ(bc::upload(signedStackOf(dy), dDy.view()), cudaSuccess);
    }

    bincv::LKLevelN<N, uint32_t> host() const {
        return bincv::lkLevel<N, uint32_t>(prev, next, dx, dy);
    }

    bc::DeviceLKLevel device() const {
        return bc::deviceLkLevel(bc::planeBlock(dPrev.constView(), N),
                                 bc::planeBlock(dNext.constView(), N),
                                 bc::planeBlock(dDx.constView(), N + 1),
                                 bc::planeBlock(dDy.constView(), N + 1));
    }
};

// ---------------------------------------------------------------------------
// LAYER 1 -- the exact-integer probes
// ---------------------------------------------------------------------------

struct IntegerTally {
    size_t cases = 0;
    size_t residualMismatches = 0;
    size_t covHostMismatches = 0;
    size_t covBatchMismatches = 0;
    size_t launchErrors = 0;
};

void checkIntegerTally(const char* what, const IntegerTally& t) {
    std::printf("  [probe] %-42s %5zu windows\n", what, t.cases);
    BINCV_CHECK(t.cases > 0);
    BINCV_CHECK_EQ(t.residualMismatches, size_t{0});
    BINCV_CHECK_EQ(t.covHostMismatches, size_t{0});
    BINCV_CHECK_EQ(t.covBatchMismatches, size_t{0});
    BINCV_CHECK_EQ(t.launchErrors, size_t{0});
}

template <size_t N>
void probeOneWindow(const Level<N>& lv, Rect window, long long tapX, long long tapY,
                    IntegerTally& t) {
    ++t.cases;
    const bincv::LKLevelN<N, uint32_t> h = lv.host();
    const bincv::impl::RegionWords<uint32_t> r =
        bincv::impl::clipRegion<uint32_t>(h.width(), h.height(), window);

    bincv::impl::TapSums wantX, wantY;
    GradientCovariance wantCov{};
    if (!r.isEmpty) {
        bincv::impl::residualSums<N, uint32_t>(h, r, tapX, tapY, wantX, wantY);
        wantCov = bincv::gradientCovariance<N, uint32_t>(h.dxMag, h.dyMag, h.dxSign,
                                                         h.dySign, window);
    }

    const bc::DeviceLKLevel dlv = lv.device();
    bc::DeviceArray<bc::impl::DeviceTapSums> dSums(2);
    bc::DeviceArray<bc::DeviceGradientCovariance> dCov(1);
    bc::DeviceArray<bc::DeviceGradientCovariance> dBatch(1);
    bc::DeviceArray<Rect> dWin(1);

    if (bc::impl::lkResidualSumsProbeAsync(dlv, window, tapX, tapY, dSums.data()) !=
            cudaSuccess ||
        bc::impl::lkCovarianceProbeAsync(dlv, window, dCov.data()) != cudaSuccess) {
        ++t.launchErrors;
        return;
    }
    if (cudaMemcpy(dWin.data(), &window, sizeof(Rect), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        ++t.launchErrors;
        return;
    }
    // The batch kernel traverses the window a completely different way. Two arms
    // that share a traversal agree for reasons that have nothing to do with being
    // right, which is why this cross-check is worth its launch.
    if (bc::gradientCovarianceBatchAsync(bc::planeBlock(lv.dDx.constView(), N + 1),
                                         bc::planeBlock(lv.dDy.constView(), N + 1),
                                         dWin.data(), 1, dBatch.data()) != cudaSuccess) {
        ++t.launchErrors;
        return;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ++t.launchErrors;
        return;
    }

    bc::impl::DeviceTapSums got[2];
    bc::DeviceGradientCovariance gotCov{}, gotBatch{};
    BINCV_CUDA_CHECK(cudaMemcpy(got, dSums.data(), sizeof(got), cudaMemcpyDeviceToHost));
    BINCV_CUDA_CHECK(
        cudaMemcpy(&gotCov, dCov.data(), sizeof(gotCov), cudaMemcpyDeviceToHost));
    BINCV_CUDA_CHECK(
        cudaMemcpy(&gotBatch, dBatch.data(), sizeof(gotBatch), cudaMemcpyDeviceToHost));

    const auto same = [](const bc::impl::DeviceTapSums& g, const bincv::impl::TapSums& w) {
        return g.t00 == w.t00 && g.t01 == w.t01 && g.t10 == w.t10 && g.t11 == w.t11 &&
               g.self == w.self;
    };
    if (!same(got[0], wantX) || !same(got[1], wantY)) ++t.residualMismatches;
    if (gotCov.sumXX != wantCov.sumXX || gotCov.sumYY != wantCov.sumYY ||
        gotCov.sumXY != wantCov.sumXY) {
        ++t.covHostMismatches;
    }
    if (gotCov.sumXX != gotBatch.sumXX || gotCov.sumYY != gotBatch.sumYY ||
        gotCov.sumXY != gotBatch.sumXY) {
        ++t.covBatchMismatches;
    }
}

/// Every window geometry and displacement the domain admits, over one level.
template <size_t N>
void sweepOneLevel(const Level<N>& lv, IntegerTally& t) {
    const int w = static_cast<int>(lv.prev.getWidth());
    const int h = static_cast<int>(lv.prev.getHeight());
    // The stated domain's two corners, the shipped 31x31, and OpenCV's 21x21.
    const int wins[][2] = {{3, 3}, {21, 21}, {31, 31}, {32, 64}, {32, 3}, {3, 64}};
    // Anchors clipping each of the four EDGES and each of the four CORNERS, one
    // straddling a word boundary, one wholly outside.
    const int anchors[][2] = {{10, 10},     {-5, 20},      {w - 8, 20},  {20, -5},
                              {20, h - 8},  {-5, -5},      {w - 8, -5},  {-5, h - 8},
                              {w - 8, h - 8}, {30, 17},    {31, 17},     {-400, -400}};
    // Past the plane on both sides, so the REPLICATE fill and its clamps are
    // exercised including the case where every tap bit comes from the fill.
    const long long taps[] = {-40, -33, -32, -1, 0, 1, 31, 32, 40};

    for (const auto& win : wins) {
        for (const auto& an : anchors) {
            for (long long tx : taps) {
                probeOneWindow<N>(lv, Rect(an[0], an[1], win[0], win[1]), tx, -tx, t);
            }
            probeOneWindow<N>(lv, Rect(an[0], an[1], win[0], win[1]), 0, 0, t);
        }
    }
}

// ---------------------------------------------------------------------------
// LAYER 2 -- the whole tracker
// ---------------------------------------------------------------------------

/// A keypoint set built to reach every exit: interior points, points whose
/// window clips each edge and each corner, points OUTSIDE the frame (loss rule
/// 1), points inside the flat box (loss rule 2, det == 0), the exact frame
/// corners, and fractional positions.
std::vector<Point2f> keypointSet(size_t w, size_t h) {
    std::vector<Point2f> pts;
    for (int i = 0; i < 40; ++i) {
        const uint64_t r = splitmix(0xC0FFEEull + static_cast<uint64_t>(static_cast<unsigned>(i)));
        pts.push_back(Point2f{static_cast<float>(r % w) + 0.37f,
                              static_cast<float>((r >> 20) % h) + 0.61f});
    }
    pts.push_back(Point2f{0.0f, 0.0f});
    pts.push_back(Point2f{static_cast<float>(w - 1), static_cast<float>(h - 1)});
    pts.push_back(Point2f{0.5f, static_cast<float>(h) - 0.5f});
    pts.push_back(Point2f{static_cast<float>(w) - 0.5f, 0.5f});
    pts.push_back(Point2f{-500.0f, 20.0f});                 // loss rule 1
    pts.push_back(Point2f{20.0f, -500.0f});                 // loss rule 1
    pts.push_back(Point2f{static_cast<float>(w) + 500.0f, 20.0f});
    pts.push_back(Point2f{static_cast<float>(w / 2 + 20), static_cast<float>(h / 2 + 20)});
    pts.push_back(Point2f{static_cast<float>(w / 2 + 18), static_cast<float>(h / 2 + 22)});
    return pts;
}

/// THE ORACLE IS THE HOST'S **REFERENCE** ARM, AND THAT IS NOT A CONVENIENCE.
///
/// ops/opticalFlow.hpp ships an AVX2 eight-keypoint batch whose own docstring
/// says `impl::lkBatchEnabled()` is "how the batched path is held to
/// BIT-EXACTNESS" -- the scalar `trackOnePoint` is the reference and the batch
/// is what is on trial against it. So the device arm is held to the scalar one.
///
/// It is held to the SHIPPED one as well, and separately, because that is the
/// arm a caller on an AVX2 x86 host actually runs. Running this suite found the
/// two host arms disagreeing at exactly one parameter value -- see
/// `hostArmDivergentZeroIter` below -- so the two comparisons are counted apart
/// rather than merged into one number that would hide which arm moved.
struct TrackTally {
    size_t configs = 0;
    size_t ptsMismatch = 0;
    size_t statusMismatch = 0;
    size_t errMismatch = 0;
    size_t launchErrors = 0;
    size_t trackedPoints = 0;
    size_t lostPoints = 0;
    /// Points where the host's SHIPPED arm differs from its own reference arm at
    /// `maxIterations >= 1`. Must be zero: the host's two arms are required to
    /// agree, and a new disagreement anywhere in the useful range is a
    /// regression this suite is entitled to fail on.
    size_t hostArmDivergentRunning = 0;
    /// The same, at `maxIterations == 0` only. NOT asserted, and printed loudly:
    /// the host's batch arm performs ONE iteration where the caller asked for
    /// none, because it tests its cap at the BOTTOM of a do-while-shaped loop
    /// (`if (s.it >= c.maxIterations) finishLane(L)`) where `trackOnePoint`
    /// tests it at the top of a `for`. Asserting this to zero would demand a fix
    /// this landing does not make; asserting it to NON-zero would bake a defect
    /// in as required behaviour. So it is reported, which is what CLAUDE.md asks
    /// for when a measurement contradicts a documented claim.
    size_t hostArmDivergentZeroIter = 0;
};

void checkTrackTally(const char* what, const TrackTally& t) {
    std::printf("  [tracker] %-34s %4zu configs, %6zu tracked, %5zu lost\n", what,
                t.configs, t.trackedPoints, t.lostPoints);
    BINCV_CHECK(t.configs > 0);
    BINCV_CHECK_EQ(t.ptsMismatch, size_t{0});
    BINCV_CHECK_EQ(t.statusMismatch, size_t{0});
    BINCV_CHECK_EQ(t.errMismatch, size_t{0});
    BINCV_CHECK_EQ(t.launchErrors, size_t{0});
    BINCV_CHECK_EQ(t.hostArmDivergentRunning, size_t{0});
    if (t.hostArmDivergentZeroIter != 0) {
        std::printf("  [NOTICE] the HOST's vector arm differs from its own scalar arm on\n"
                    "           %zu point(s), at maxIterations == 0 and nowhere else. The\n"
                    "           device arm matches the scalar arm. Reported, not worked\n"
                    "           around -- see TrackTally::hostArmDivergentZeroIter.\n",
                    t.hostArmDivergentZeroIter);
    }
    // A comparison over points that were all lost before the iteration loop would
    // pass without exercising anything.
    BINCV_CHECK(t.trackedPoints > 0);
    BINCV_CHECK(t.lostPoints > 0);
}

/// Runs one configuration on THREE arms and compares: the host's scalar
/// reference arm (the oracle), the host's shipped arm, and the device.
void runConfig(const std::vector<Point2f>& pts, const bc::DeviceLKLevel* devLevels,
               size_t levelCount, const LKParams& params, bool wantErr,
               const std::function<void(const Point2f*, Point2f*, uint8_t*, float*, size_t,
                                        const LKParams&)>& hostRun,
               TrackTally& t) {
    ++t.configs;
    const size_t n = pts.size();
    std::vector<Point2f> refNext(n), shipNext(n), devNext(n);
    std::vector<uint8_t> refSt(n, 7), shipSt(n, 8), devSt(n, 9);
    std::vector<float> refErr(n, -1.0f), shipErr(n, -3.0f), devErr(n, -2.0f);

    // Under useInitialFlow every arm must start from the SAME guess, so the seed
    // is written into all three output arrays before any of them runs.
    for (size_t i = 0; i < n; ++i) {
        const uint64_t r = splitmix(0xBEEFull + i);
        const Point2f seed{pts[i].x + static_cast<float>(static_cast<int>(r % 7)) - 3.0f,
                           pts[i].y + static_cast<float>(static_cast<int>((r >> 8) % 7)) -
                               3.0f};
        refNext[i] = seed;
        shipNext[i] = seed;
        devNext[i] = seed;
    }

    bincv::impl::lkBatchEnabled() = false;
    hostRun(pts.data(), refNext.data(), refSt.data(), wantErr ? refErr.data() : nullptr, n,
            params);
    bincv::impl::lkBatchEnabled() = true;
    hostRun(pts.data(), shipNext.data(), shipSt.data(), wantErr ? shipErr.data() : nullptr,
            n, params);

    bc::DeviceArray<float> dPrev(2 * n), dNext(2 * n), dErr(n);
    bc::DeviceArray<uint8_t> dStatus(n);
    BINCV_CUDA_CHECK(cudaMemcpy(dPrev.data(), pts.data(), n * sizeof(Point2f),
                                cudaMemcpyHostToDevice));
    BINCV_CUDA_CHECK(cudaMemcpy(dNext.data(), devNext.data(), n * sizeof(Point2f),
                                cudaMemcpyHostToDevice));

    bc::DeviceLKTracks tracks;
    tracks.dPrevXY = dPrev.data();
    tracks.dNextXY = dNext.data();
    tracks.dStatus = dStatus.data();
    tracks.dErr = wantErr ? dErr.data() : nullptr;
    tracks.count = static_cast<uint32_t>(n);

    if (bc::calcOpticalFlowPyrLKAsync(devLevels, levelCount, tracks, params) !=
            cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        ++t.launchErrors;
        return;
    }
    BINCV_CUDA_CHECK(cudaMemcpy(devNext.data(), dNext.data(), n * sizeof(Point2f),
                                cudaMemcpyDeviceToHost));
    BINCV_CUDA_CHECK(cudaMemcpy(devSt.data(), dStatus.data(), n, cudaMemcpyDeviceToHost));
    if (wantErr) {
        BINCV_CUDA_CHECK(cudaMemcpy(devErr.data(), dErr.data(), n * sizeof(float),
                                    cudaMemcpyDeviceToHost));
    }

    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&refNext[i], &devNext[i], sizeof(Point2f)) != 0) ++t.ptsMismatch;
        if (refSt[i] != devSt[i]) ++t.statusMismatch;
        if (wantErr && std::memcmp(&refErr[i], &devErr[i], sizeof(float)) != 0) {
            ++t.errMismatch;
        }
        const bool hostArmsDiffer =
            std::memcmp(&refNext[i], &shipNext[i], sizeof(Point2f)) != 0 ||
            refSt[i] != shipSt[i] ||
            (wantErr && std::memcmp(&refErr[i], &shipErr[i], sizeof(float)) != 0);
        if (hostArmsDiffer) {
            if (params.maxIterations == 0) {
                ++t.hostArmDivergentZeroIter;
            } else {
                ++t.hostArmDivergentRunning;
            }
        }
        if (refSt[i] != 0) {
            ++t.trackedPoints;
        } else {
            ++t.lostPoints;
        }
    }
}

/// The parameter sweep. Not a full cross product -- that would be thousands of
/// launches for no extra coverage -- but every value of every field appears, and
/// the fields that INTERACT (entry level with useInitialFlow, maxResidual with
/// err, maxIterations with the termination rules) are crossed.
std::vector<std::pair<LKParams, bool>> parameterSweep(int winW, int winH) {
    std::vector<std::pair<LKParams, bool>> out;
    const int iters[] = {0, 1, 3, 20};
    const float epss[] = {0.0f, 0.03f, 10.0f};
    const float minEigs[] = {0.0f, 0.001f, 1e9f};
    const float residuals[] = {0.0f, 0.05f};
    for (int it : iters) {
        for (float eps : epss) {
            LKParams p;
            p.winWidth = winW;
            p.winHeight = winH;
            p.maxIterations = it;
            p.epsilon = eps;
            out.emplace_back(p, false);
            out.emplace_back(p, true);
        }
    }
    for (float me : minEigs) {
        for (float mr : residuals) {
            for (bool initial : {false, true}) {
                for (LKEntryLevel el :
                     {LKEntryLevel::Coarsest, LKEntryLevel::DeepestFitting}) {
                    LKParams p;
                    p.winWidth = winW;
                    p.winHeight = winH;
                    p.minEigThreshold = me;
                    p.maxResidual = mr;
                    p.useInitialFlow = initial;
                    p.entryLevel = el;
                    // maxResidual makes the residual computed whether or not `err`
                    // was asked for -- the host's own note, and the reason both
                    // spellings are run.
                    out.emplace_back(p, mr > 0.0f ? false : true);
                    out.emplace_back(p, true);
                }
            }
        }
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------

BINCV_TEST(CudaOpticalFlow, ResidualAndCovarianceProbesMatchTheHostExactly) {
    // Widths chosen at the word boundary and either side of it, plus the
    // reference ladder's four. 64 is the boundary and 65 the word-crossing.
    const size_t widths[] = {33, 63, 64, 65, 94, 188};
    for (size_t w : widths) {
        IntegerTally t1, t2;
        {
            Level<1> lv(w, 70, 0x1234ull + w, 3, -2, true);
            sweepOneLevel<1>(lv, t1);
        }
        {
            Level<2> lv(w, 70, 0x5678ull + w, -3, 2, true);
            sweepOneLevel<2>(lv, t2);
        }
        char label[64];
        std::snprintf(label, sizeof(label), "width %zu, N = 1", w);
        checkIntegerTally(label, t1);
        std::snprintf(label, sizeof(label), "width %zu, N = 2", w);
        checkIntegerTally(label, t2);
    }
}

BINCV_TEST(CudaOpticalFlow, DirtyPaddingCannotChangeAnyProbeAnswer) {
    // The padding-bit invariant, asked from the CONSUMER's side: every bit at or
    // past `width` in each trailing word is set on BOTH backends, and both must
    // still give the clean plane's answer.
    for (size_t w : {size_t{33}, size_t{63}, size_t{65}, size_t{94}}) {
        Level<1> clean(w, 70, 0xABCDull + w, 2, 1, false);
        IntegerTally before;
        sweepOneLevel<1>(clean, before);

        Level<1> dirt(w, 70, 0xABCDull + w, 2, 1, false);
        dirt.dirty();
        IntegerTally after;
        sweepOneLevel<1>(dirt, after);

        char label[64];
        std::snprintf(label, sizeof(label), "dirty padding, width %zu", w);
        checkIntegerTally(label, after);
        BINCV_CHECK_EQ(after.cases, before.cases);
    }
    // At N = 2 as well, where the sign plane is one of the dirtied ones.
    Level<2> dirt2(94, 70, 0x2222ull, -2, 3, false);
    dirt2.dirty();
    IntegerTally t;
    sweepOneLevel<2>(dirt2, t);
    checkIntegerTally("dirty padding, width 94, N = 2", t);
}

BINCV_TEST(CudaOpticalFlow, WholeTrackerMatchesTheHostBitForBit) {
    // 512x384 halves to 64x48 at level 3, so all four levels clear a 31x31 window
    // and `usableLevelCount`'s prefix rule keeps all of them.
    const size_t W = 512, H = 384;
    Level<1> l0(W, H, 0x11ull, 3, -2, true);
    Level<1> l1(W / 2, H / 2, 0x12ull, 2, -1, false);
    Level<1> l2(W / 4, H / 4, 0x13ull, 1, -1, false);
    Level<1> l3(W / 8, H / 8, 0x14ull, 1, 0, false);

    bincv::LKLevelN<1, uint32_t> hostLevels[4] = {l0.host(), l1.host(), l2.host(),
                                                  l3.host()};
    bc::DeviceLKLevel devLevels[4] = {l0.device(), l1.device(), l2.device(), l3.device()};

    const std::vector<Point2f> pts = keypointSet(W, H);
    TrackTally t;
    for (const auto& pc : parameterSweep(31, 31)) {
        runConfig(pts, devLevels, 4, pc.first, pc.second,
                  [&](const Point2f* p, Point2f* n, uint8_t* s, float* e, size_t c,
                      const LKParams& pr) {
                      bincv::calcOpticalFlowPyrLK<1, uint32_t>(hostLevels, 4, p, n, s, e, c,
                                                               pr);
                  },
                  t);
    }
    checkTrackTally("1/1/1/1 ladder, 31x31", t);

    // The degenerate ladders: zero levels (every point lost, nextPts a copy of
    // prevPts unless useInitialFlow) and one level.
    TrackTally td;
    for (size_t levelCount : {size_t{0}, size_t{1}}) {
        for (bool initial : {false, true}) {
            LKParams p;
            p.useInitialFlow = initial;
            runConfig(pts, devLevels, levelCount, p, true,
                      [&](const Point2f* pp, Point2f* n, uint8_t* s, float* e, size_t c,
                          const LKParams& pr) {
                          bincv::calcOpticalFlowPyrLK<1, uint32_t>(hostLevels, levelCount,
                                                                   pp, n, s, e, c, pr);
                      },
                      td);
        }
    }
    std::printf("  [tracker] %-34s %4zu configs\n", "levelCount 0 and 1", td.configs);
    BINCV_CHECK_EQ(td.ptsMismatch, size_t{0});
    BINCV_CHECK_EQ(td.statusMismatch, size_t{0});
    BINCV_CHECK_EQ(td.errMismatch, size_t{0});
    BINCV_CHECK_EQ(td.launchErrors, size_t{0});

    // A smaller window, which changes both the lane utilisation and the number of
    // levels `usableLevelCount` keeps.
    TrackTally ts;
    for (const auto& pc : parameterSweep(7, 7)) {
        runConfig(pts, devLevels, 4, pc.first, pc.second,
                  [&](const Point2f* p, Point2f* n, uint8_t* s, float* e, size_t c,
                      const LKParams& pr) {
                      bincv::calcOpticalFlowPyrLK<1, uint32_t>(hostLevels, 4, p, n, s, e, c,
                                                               pr);
                  },
                  ts);
    }
    checkTrackTally("1/1/1/1 ladder, 7x7", ts);
}

BINCV_TEST(CudaOpticalFlow, WholeTrackerMatchesTheHostOnAWidthThatIsNotAMultipleOf32) {
    // 505 % 32 == 25, and every level below it lands on a different residue --
    // 252, 126, 63 -- so the head/tail masks and the aligned extraction are
    // exercised at four different offsets in one ladder.
    const size_t W = 505, H = 387;
    Level<1> l0(W, H, 0x21ull, 2, -3, true);
    Level<1> l1(W / 2, H / 2, 0x22ull, 1, -2, false);
    Level<1> l2(W / 4, H / 4, 0x23ull, 1, -1, false);
    Level<1> l3(W / 8, H / 8, 0x24ull, 0, -1, false);

    bincv::LKLevelN<1, uint32_t> hostLevels[4] = {l0.host(), l1.host(), l2.host(),
                                                  l3.host()};
    bc::DeviceLKLevel devLevels[4] = {l0.device(), l1.device(), l2.device(), l3.device()};
    const std::vector<Point2f> pts = keypointSet(W, H);

    TrackTally t;
    for (const auto& pc : parameterSweep(31, 31)) {
        runConfig(pts, devLevels, 4, pc.first, pc.second,
                  [&](const Point2f* p, Point2f* n, uint8_t* s, float* e, size_t c,
                      const LKParams& pr) {
                      bincv::calcOpticalFlowPyrLK<1, uint32_t>(hostLevels, 4, p, n, s, e, c,
                                                               pr);
                  },
                  t);
    }
    checkTrackTally("505x387, 31x31", t);
}

BINCV_TEST(CudaOpticalFlow, WholeTrackerMatchesTheHostOnTheShipped1222Ladder) {
    // The mixed ladder the host ships. Level 0 is ONE bit -- which is also what
    // keeps `err` inside this backend's domain -- and levels 1..3 are two.
    const size_t W = 512, H = 384;
    Level<1> l0(W, H, 0x31ull, 3, -2, true);
    Level<2> l1(W / 2, H / 2, 0x32ull, 2, -1, false);
    Level<2> l2(W / 4, H / 4, 0x33ull, 1, -1, false);
    Level<2> l3(W / 8, H / 8, 0x34ull, 1, 0, false);

    bincv::LKLevels<uint32_t, 1, 2, 2, 2> ladder;
    ladder.get<0>() = l0.host();
    ladder.get<1>() = l1.host();
    ladder.get<2>() = l2.host();
    ladder.get<3>() = l3.host();

    bc::DeviceLKLevel devLevels[4] = {l0.device(), l1.device(), l2.device(), l3.device()};
    const std::vector<Point2f> pts = keypointSet(W, H);

    TrackTally t;
    for (const auto& pc : parameterSweep(31, 31)) {
        runConfig(pts, devLevels, 4, pc.first, pc.second,
                  [&](const Point2f* p, Point2f* n, uint8_t* s, float* e, size_t c,
                      const LKParams& pr) {
                      bincv::calcOpticalFlowPyrLK<uint32_t, 1, 2, 2, 2>(ladder, p, n, s, e,
                                                                        c, pr);
                  },
                  t);
    }
    checkTrackTally("1/2/2/2 ladder, 31x31", t);
}

BINCV_TEST(CudaOpticalFlow, EveryArmCombinationGivesOneAnswerAndSaysWhichItTook) {
    const size_t W = 512, H = 384;
    Level<1> l0(W, H, 0x41ull, 3, -2, true);
    Level<2> l1(W / 2, H / 2, 0x42ull, 2, -1, false);
    Level<2> l2(W / 4, H / 4, 0x43ull, 1, -1, false);
    Level<2> l3(W / 8, H / 8, 0x44ull, 1, 0, false);
    bc::DeviceLKLevel devLevels[4] = {l0.device(), l1.device(), l2.device(), l3.device()};
    const std::vector<Point2f> pts = keypointSet(W, H);
    const size_t n = pts.size();

    LKParams params;
    params.maxResidual = 0.05f;

    bc::DeviceArray<float> dPrev(2 * n), dNext(2 * n), dErr(n);
    bc::DeviceArray<uint8_t> dStatus(n);
    BINCV_CUDA_CHECK(cudaMemcpy(dPrev.data(), pts.data(), n * sizeof(Point2f),
                                cudaMemcpyHostToDevice));
    bc::DeviceLKTracks tracks;
    tracks.dPrevXY = dPrev.data();
    tracks.dNextXY = dNext.data();
    tracks.dStatus = dStatus.data();
    tracks.dErr = dErr.data();
    tracks.count = static_cast<uint32_t>(n);

    std::vector<Point2f> refPts;
    std::vector<uint8_t> refSt;
    std::vector<float> refErr;
    size_t combos = 0, mismatches = 0, pathMissed = 0;

    for (bool intrinsic : {true, false}) {
        for (bool cache : {true, false}) {
            bc::impl::lkWarpReduceIntrinsicEnabled() = intrinsic;
            bc::impl::lkTapCacheEnabled() = cache;
            ++combos;

            BINCV_CUDA_CHECK(cudaMemset(dNext.data(), 0, 2 * n * sizeof(float)));
            BINCV_CHECK_EQ(
                bc::calcOpticalFlowPyrLKAsync(devLevels, 4, tracks, params), cudaSuccess);
            BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

            std::vector<Point2f> gotPts(n);
            std::vector<uint8_t> gotSt(n);
            std::vector<float> gotErr(n);
            BINCV_CUDA_CHECK(cudaMemcpy(gotPts.data(), dNext.data(), n * sizeof(Point2f),
                                        cudaMemcpyDeviceToHost));
            BINCV_CUDA_CHECK(
                cudaMemcpy(gotSt.data(), dStatus.data(), n, cudaMemcpyDeviceToHost));
            BINCV_CUDA_CHECK(cudaMemcpy(gotErr.data(), dErr.data(), n * sizeof(float),
                                        cudaMemcpyDeviceToHost));

            if (refPts.empty()) {
                refPts = gotPts;
                refSt = gotSt;
                refErr = gotErr;
            } else {
                for (size_t i = 0; i < n; ++i) {
                    if (std::memcmp(&refPts[i], &gotPts[i], sizeof(Point2f)) != 0 ||
                        refSt[i] != gotSt[i] ||
                        std::memcmp(&refErr[i], &gotErr[i], sizeof(float)) != 0) {
                        ++mismatches;
                    }
                }
            }
        }
    }
    bc::impl::lkWarpReduceIntrinsicEnabled() = true;
    bc::impl::lkTapCacheEnabled() = true;

    BINCV_CHECK_EQ(combos, size_t{4});
    BINCV_CHECK_EQ(mismatches, size_t{0});

    // BYTE-IDENTICAL OUTPUT CANNOT SHOW THAT AN ARM RAN. lkPathName reports the
    // arm a launch actually takes, by taking it -- so an arm that was compiled
    // out, or a switch that stopped switching, fails HERE rather than quietly
    // costing time forever.
    const char* name = bc::lkPathName();
    std::printf("  [arms] %s\n", name);
    if (std::strstr(name, "warp-per-keypoint") == nullptr) ++pathMissed;
    // sm_86 is this backend's reference GPU, and the intrinsic arm exists there.
    // A build for sm_70 would legitimately report the shuffle tree, so the
    // assertion is that the string names ONE of the two, not that it names the
    // fast one.
    if (std::strstr(name, "reduce_add_sync") == nullptr &&
        std::strstr(name, "shuffle-tree") == nullptr) {
        ++pathMissed;
    }
    if (std::strstr(name, "tap-cache") == nullptr) ++pathMissed;
    BINCV_CHECK_EQ(pathMissed, size_t{0});
}

BINCV_TEST(CudaOpticalFlow, DomainRefusalsReturnInvalidValueAndWriteNothing) {
    const size_t W = 128, H = 96;
    Level<1> l0(W, H, 0x51ull, 1, 1, false);
    bc::DeviceLKLevel lv = l0.device();
    bc::DeviceLKLevel many[17];
    for (auto& m : many) m = lv;

    const size_t n = 4;
    bc::DeviceArray<float> dPrev(2 * n), dNext(2 * n), dErr(n);
    bc::DeviceArray<uint8_t> dStatus(n);
    const float seed[2 * n] = {10.f, 10.f, 20.f, 20.f, 30.f, 30.f, 40.f, 40.f};
    BINCV_CUDA_CHECK(
        cudaMemcpy(dPrev.data(), seed, sizeof(seed), cudaMemcpyHostToDevice));
    const uint8_t sentinel = 0xAB;
    BINCV_CUDA_CHECK(cudaMemset(dStatus.data(), sentinel, n));

    bc::DeviceLKTracks tracks;
    tracks.dPrevXY = dPrev.data();
    tracks.dNextXY = dNext.data();
    tracks.dStatus = dStatus.data();
    tracks.dErr = nullptr;
    tracks.count = static_cast<uint32_t>(n);

    const auto refused = [&](const LKParams& p, size_t levelCount,
                             const bc::DeviceLKLevel* levels, bool wantErr) {
        bc::DeviceLKTracks tr = tracks;
        if (wantErr) tr.dErr = dErr.data();
        return bc::calcOpticalFlowPyrLKAsync(levels, levelCount, tr, p);
    };

    LKParams wide;
    wide.winWidth = 33;
    BINCV_CHECK_EQ(refused(wide, 1, &lv, false), cudaErrorInvalidValue);
    LKParams tall;
    tall.winHeight = 65;
    BINCV_CHECK_EQ(refused(tall, 1, &lv, false), cudaErrorInvalidValue);
    LKParams tiny;
    tiny.winWidth = 2;
    BINCV_CHECK_EQ(refused(tiny, 1, &lv, false), cudaErrorInvalidValue);
    BINCV_CHECK_EQ(refused(LKParams(), 17, many, false), cudaErrorInvalidValue);

    // bits > 2: a three-plane level, which this backend does not instantiate.
    bc::DeviceLKLevel deep = lv;
    deep.bits = 3;
    BINCV_CHECK_EQ(refused(LKParams(), 1, &deep, false), cudaErrorInvalidValue);

    // `err` needs a ONE-BIT level 0, because the popcount identity for
    // |Jinterp - I| is exact only when I is a bit.
    Level<2> two(W, H, 0x52ull, 1, 1, false);
    bc::DeviceLKLevel lv2 = two.device();
    BINCV_CHECK_EQ(refused(LKParams(), 1, &lv2, true), cudaErrorInvalidValue);
    LKParams withResidual;
    withResidual.maxResidual = 0.05f;
    BINCV_CHECK_EQ(refused(withResidual, 1, &lv2, false), cudaErrorInvalidValue);
    // ...and is accepted at one bit, so the refusal is about the depth and not
    // about `err` itself.
    BINCV_CHECK_EQ(refused(LKParams(), 1, &lv, true), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // A refused call writes NOTHING -- the sentinel survives. (The accepted call
    // above ran last, so this re-checks a fresh refusal against a fresh
    // sentinel.)
    BINCV_CUDA_CHECK(cudaMemset(dStatus.data(), sentinel, n));
    BINCV_CHECK_EQ(refused(wide, 1, &lv, false), cudaErrorInvalidValue);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint8_t> got(n, 0);
    BINCV_CUDA_CHECK(cudaMemcpy(got.data(), dStatus.data(), n, cudaMemcpyDeviceToHost));
    size_t touched = 0;
    for (uint8_t v : got) {
        if (v != sentinel) ++touched;
    }
    BINCV_CHECK_EQ(touched, size_t{0});

    // A null output array is a refusal, not a fault.
    bc::DeviceLKTracks nullStatus = tracks;
    nullStatus.dStatus = nullptr;
    BINCV_CHECK_EQ(bc::calcOpticalFlowPyrLKAsync(&lv, 1, nullStatus, LKParams()),
                   cudaErrorInvalidValue);
    // Zero points is legal and does nothing.
    bc::DeviceLKTracks empty = tracks;
    empty.count = 0;
    BINCV_CHECK_EQ(bc::calcOpticalFlowPyrLKAsync(&lv, 1, empty, LKParams()), cudaSuccess);
}

BINCV_TEST(CudaOpticalFlow, FmaContractionIsPinnedOnBothTargets) {
    // THE GATE NOBODY HAS WATCHED FAIL IS NOT KNOWN TO WORK. These operands are
    // chosen so that `a*b - c*d` differs between a contracted and an uncontracted
    // evaluation: the two products are equal to within one ulp, so the exact
    // difference is entirely in the rounding that an FMA would not perform.
    const double a = 1.0 + 1.0 / 9007199254740992.0;  // 1 + 2^-53, not representable
    const double cases[][4] = {
        {a, a, 1.0, 1.0},
        {33554432.0 + 1.0, 33554432.0 - 1.0, 33554432.0, 33554432.0},
        {1.0000000000000002, 1.0000000000000002, 1.0, 1.0},
        {3.0, 1.0 / 3.0, 1.0, 1.0},
    };
    bc::DeviceArray<double> dOut(1);
    size_t mismatches = 0;
    for (const auto& c : cases) {
        const double want = c[0] * c[1] - c[2] * c[3];
        BINCV_CHECK_EQ(
            bc::impl::lkFmaGuardProbeAsync(c[0], c[1], c[2], c[3], dOut.data()),
            cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        double got = 0.0;
        BINCV_CUDA_CHECK(
            cudaMemcpy(&got, dOut.data(), sizeof(double), cudaMemcpyDeviceToHost));
        if (std::memcmp(&got, &want, sizeof(double)) != 0) ++mismatches;
    }
    BINCV_CHECK_EQ(mismatches, size_t{0});
    // The second case is the one that actually discriminates: 2^25+1 times 2^25-1
    // is 2^50 - 1, exactly representable, while a fused form keeps the full
    // product. If a future flag change makes the device contract, this fails.
    BINCV_CHECK_EQ((33554432.0 + 1.0) * (33554432.0 - 1.0) - 33554432.0 * 33554432.0,
                   -1.0);
}

BINCV_TEST(CudaOpticalFlow, TheSynchronousConvenienceAgreesWithTheAsyncForm) {
    const size_t W = 256, H = 192;
    Level<1> l0(W, H, 0x61ull, 2, -1, true);
    Level<1> l1(W / 2, H / 2, 0x62ull, 1, -1, false);
    bincv::LKLevelN<1, uint32_t> hostLevels[2] = {l0.host(), l1.host()};
    bc::DeviceLKLevel devLevels[2] = {l0.device(), l1.device()};

    const std::vector<Point2f> pts = keypointSet(W, H);
    const size_t n = pts.size();
    std::vector<Point2f> hostNext(n), devNext(n);
    std::vector<uint8_t> hostSt(n), devSt(n);
    std::vector<float> hostErr(n), devErr(n);

    LKParams p;
    bincv::calcOpticalFlowPyrLK<1, uint32_t>(hostLevels, 2, pts.data(), hostNext.data(),
                                             hostSt.data(), hostErr.data(), n, p);
    bc::calcOpticalFlowPyrLK(devLevels, 2, pts.data(), devNext.data(), devSt.data(),
                             devErr.data(), n, p);

    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&hostNext[i], &devNext[i], sizeof(Point2f)) != 0) ++mismatches;
        if (hostSt[i] != devSt[i]) ++mismatches;
        if (std::memcmp(&hostErr[i], &devErr[i], sizeof(float)) != 0) ++mismatches;
    }
    BINCV_CHECK_EQ(mismatches, size_t{0});
}

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
    const int summaryRc = ::bincv::test::summarize("CUDA optical-flow tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA optical-flow tests", argc, argv);
}
#endif
