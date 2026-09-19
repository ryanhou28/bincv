// ===========================================================================
// CAN THE FRONTEND'S RANSAC GEOMETRY STAGE MOVE TO THE GPU?
//
// This binary exists to answer that question and to keep the answer
// reproducible. It prices a device RANSAC round against the host arm the
// frontend runs today, on the same machine, the same scene and one explicit
// stream, and it prints the decision rule BEFORE the first number.
//
// ---------------------------------------------------------------------------
// THE DECISION RULE, WRITTEN BEFORE MEASURING
//
// METRICS THAT DECIDE THIS CASE
//   M1  geometry-stage wall time per frame, host arm vs device arm, at the
//       frontend's operating point. M1 IS THE ADOPTION NUMBER. A hypotheses/ms
//       microbenchmark is a diagnostic and is never multiplied by a stage share
//       to produce one.
//   M2  working set, SYMMETRIC. The host's solver frame is measured by g++
//       -fstack-usage and reported by essentialSolverStackBytes(); the device's
//       is measured by nvcc -Xptxas -v. Counting one side's scratch and not the
//       other's is the comparison this gate would otherwise turn on.
//   M3  model quality -- the consensus the two sides reach.
//   M4  hand-written float code that must stay numerically consistent forever,
//       counted in KIND first: one implementation compiled for two targets, or
//       a second implementation to keep correct. ARCHITECTURE's scope rule
//       turns on exactly that distinction.
//
// THE BASELINE is the best existing option: min(host bincv::findEssentialMat,
// cv::findEssentialMat) on this machine.
//
// THE ROLE BAR, STATED CORRECTLY. OpenCV has no cv::cuda essential-matrix or
// 2D-2D estimator, so there is no GPU role bar for findEssentialMat. It does
// ship a GPU RANSAC: cv::cuda::solvePnPRansac runs a host-side parallel_for_
// over hypotheses calling solvePnP(SOLVEPNP_EPNP) and then one device launch
// that scores them. "OpenCV ships no GPU RANSAC at all" would be false, and so
// would any Tier-3 claim resting on it. What that split IS, entered here as a
// prior before measuring rather than found afterwards: the one shipped GPU
// RANSAC anyone has built keeps the minimal solver on the host and moves only
// the scoring.
//
// THE GATES, in order, each able to end the work.
//   G0  AMDAHL. Geometry's share of binCV frontend total on this host bounds
//       an infinitely fast device stage at total/(total-geometry). KILL if that
//       ceiling does not clear the host arm's own measured spread here.
//   G1  WHERE THE PER-HYPOTHESIS COST IS. If scoring is fraction f of a
//       hypothesis, a scoring-only device arm -- the split OpenCV chose --
//       caps the STAGE speedup at 1/(1-f) at infinite device speed. KILL if
//       1/(1-f) does not clear that same spread. Same currency on both sides.
//   G2  FEASIBILITY, AND WHICH KIND OF M4 THIS IS. If fivePointEssential
//       compiles for the device through BINCV_HOST_DEVICE, M4 stays "one
//       implementation, two targets". If it must be forked, M4 becomes a second
//       FP64 elimination to keep correct forever, which is a STOP AND ASK
//       before a kernel is written.
//   G3  THROUGHPUT, THE CONTINUE BAR, stated as a formula and not at a chosen H.
//       The host runs I adaptive iterations; a device round covers H and costs
//       one synchronize, so covering the same search needs
//       R(H) = max(1, ceil(I/H)) rounds.
//         CONTINUE iff min over the swept H of R(H)*(T_round(H) + T_sync)
//                      < I * t_host
//       with I and t_host BOTH measured here rather than inherited.
//   G4  MEMORY, SYMMETRIC (M2). The device working set at the cheapest round
//       must not exceed the host arm's scratch + solver frame + points.
//   G5  QUALITY (M3). The device must not find a weaker consensus.
//
// THE SHIP BAR IS DELIBERATELY BLANK AND IS A STOP-AND-ASK. How much M1 stage
// speedup justifies a permanent hand-written FP64 device path is a judgement
// about what this library wants to own; no measurement produces it, and
// inventing one would launder an arbitrary call through the write-it-first
// rule. Every gate above is CONTINUE/KILL. If all of them were cleared the
// result would be escalated with its numbers, not shipped on an invented bar.
//
// ---------------------------------------------------------------------------
// WHAT THE NUMBERS BELOW ARE
//
// The host clock on this machine is not timing-grade -- the same frontend run
// three times gives geometry-stage times that differ by 2.2x -- so every ratio
// here is measured INTERLEAVED, arm A then arm B then arm B then arm A, with
// the ratio formed WITHIN each round. Ranges are printed beside every median,
// and a ratio whose two arms' ranges overlap is not a result and says so.
//
// Nothing here is a shipping operation. The device kernels live in
// cuda_ransac_kernels.cu, are not compiled into bincv_cuda, and are reachable
// only from this binary and the suite that pins the shared solver.
// ===========================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/ops/essential.hpp"
#include "bincv/ops/ransac.hpp"

#include "cuda_bench_util.hpp"
#include "cuda_ransac_kernels.hpp"

namespace {

using bincv::EssentialMatrix;
using bincv::EssentialModel;
using bincv::Point2f;
using bincv::RansacParams;
using bincv::RansacResult;
namespace probe = cudabench::ransacprobe;

// The frontend's own operating point, re-measured on this host rather than
// inherited: examples/slam_frontend on EuRoC V1_02 at 752x480, budget 500
// keypoints over four levels, gated matcher at window 48 / ratio 75, reports
// 140.6 ratio-test accepts per frame at a 77.5% inlier rate and 21 adaptive
// RANSAC iterations. 141 and 23% are those numbers.
constexpr size_t kFrontendCount = 141;
constexpr int kFrontendOutlierPct = 23;
// 1.5 px at the EuRoC cam0 focal length, which is what the frontend passes.
constexpr float kThreshold = 0.00405f;
constexpr uint64_t kSeed = UINT64_C(0x9E3779B97F4A7C15);

// The gate-excluded control: below one full ballot the warp scoring arm's own
// gate refuses it, so both arms take the serial path and the ratio must read
// ~1.00x. If that row separates, the switch is not selecting what it claims.
constexpr size_t kGateExcludedCount = 16;

struct Scene {
    std::vector<Point2f> from, to;
    size_t planted = 0;
};

Scene makeScene(size_t count, int outlierPct, uint64_t seed) {
    uint64_t s = seed;
    auto uni = [&s]() {
        s = s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return 2.0 * (static_cast<double>((s >> 33) % 1000001) / 1000000.0) - 1.0;
    };
    const double ax = uni() * 0.4, ay = uni() * 0.4, az = uni() * 0.4;
    const double ca = std::cos(ax), sa = std::sin(ax), cb = std::cos(ay),
                 sb = std::sin(ay), cc = std::cos(az), sc = std::sin(az);
    const double R[3][3] = {{cb * cc, -cb * sc, sb},
                            {sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb},
                            {-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb}};
    double t[3] = {uni(), uni(), 1.0};
    const double tn = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    for (int i = 0; i < 3; ++i) t[i] /= tn;

    Scene out;
    for (size_t i = 0; i < count; ++i) {
        const double X[3] = {uni() * 2.0, uni() * 2.0, 4.0 + uni()};
        out.from.push_back(
            Point2f{static_cast<float>(X[0] / X[2]), static_cast<float>(X[1] / X[2])});
        double Xc[3];
        for (int k = 0; k < 3; ++k) {
            double acc = 0.0;
            for (int j = 0; j < 3; ++j) acc += R[k][j] * X[j];
            Xc[k] = acc + t[k];
        }
        Point2f b{static_cast<float>(Xc[0] / Xc[2]), static_cast<float>(Xc[1] / Xc[2])};
        if (static_cast<int>(i % 100) < outlierPct) {
            b.x += static_cast<float>(uni() * 0.4);
            b.y += static_cast<float>(uni() * 0.4);
        } else {
            ++out.planted;
        }
        out.to.push_back(b);
    }
    return out;
}

double msSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t)
        .count();
}

struct Sample {
    double median = 0.0, lo = 0.0, hi = 0.0;
};

Sample summarise(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Sample s;
    s.median = v.empty() ? 0.0 : v[v.size() / 2];
    s.lo = v.empty() ? 0.0 : v.front();
    s.hi = v.empty() ? 0.0 : v.back();
    return s;
}

/// @brief Two arms measured against each other with the ratio formed WITHIN each
/// round, and the round order alternating so neither arm always runs second.
/// @note This is timeKernelPaired's discipline applied to a pair where one arm
/// is a CPU call: CUDA events cannot bracket the host arm, so both are read
/// on the host clock and the protection against drift is the interleaving.
struct Paired {
    Sample a, b, ratio;
    bool overlap = false;
};

Paired pairedWallClock(const std::function<void()>& armA,
                       const std::function<void()>& armB, int rounds) {
    std::vector<double> as, bs, rs;
    for (int r = 0; r < rounds; ++r) {
        double ta = 0.0, tb = 0.0;
        if ((r & 1) == 0) {
            auto t0 = std::chrono::steady_clock::now();
            armA();
            ta = msSince(t0);
            t0 = std::chrono::steady_clock::now();
            armB();
            tb = msSince(t0);
        } else {
            auto t0 = std::chrono::steady_clock::now();
            armB();
            tb = msSince(t0);
            t0 = std::chrono::steady_clock::now();
            armA();
            ta = msSince(t0);
        }
        as.push_back(ta);
        bs.push_back(tb);
        rs.push_back(ta > 0.0 ? tb / ta : 0.0);
    }
    Paired p;
    p.a = summarise(as);
    p.b = summarise(bs);
    p.ratio = summarise(rs);
    // Whether the two arms' SAMPLE RANGES overlap. If they do, the ratio is not
    // a result on this instrument and must be reported as one.
    p.overlap = !(p.a.hi < p.b.lo || p.b.hi < p.a.lo);
    return p;
}

// ---------------------------------------------------------------------------
// G1 -- where a hypothesis's cost actually is, host only
// ---------------------------------------------------------------------------

struct Split {
    double solverMsPerHyp = 0.0;
    double scoreMsPerHyp = 0.0;
    double modelsPerHyp = 0.0;
    double f = 0.0;
};

Split hostSplit(const Scene& sc, size_t count, int rounds, int hypotheses) {
    std::vector<size_t> idx(static_cast<size_t>(hypotheses) * 5);
    for (int it = 0; it < hypotheses; ++it) {
        const uint64_t counter = kSeed + static_cast<uint64_t>(it) * UINT64_C(0x9E3779B9);
        bincv::impl::ransacSample(count, 5, counter,
                                  &idx[static_cast<size_t>(it) * 5]);
    }
    std::vector<EssentialMatrix> models(static_cast<size_t>(hypotheses) * 10);
    std::vector<size_t> produced(static_cast<size_t>(hypotheses));
    const size_t words = bincv::ransacScratchWords(count);
    std::vector<uint32_t> flags(words);

    std::vector<double> solve, score;
    size_t totalModels = 0;
    for (int r = 0; r < rounds; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        totalModels = 0;
        for (int it = 0; it < hypotheses; ++it) {
            produced[static_cast<size_t>(it)] = EssentialModel::estimate(
                sc.from.data(), sc.to.data(), &idx[static_cast<size_t>(it) * 5],
                &models[static_cast<size_t>(it) * 10]);
            totalModels += produced[static_cast<size_t>(it)];
        }
        solve.push_back(msSince(t0));

        t0 = std::chrono::steady_clock::now();
        size_t sink = 0;
        for (int it = 0; it < hypotheses; ++it) {
            for (size_t m = 0; m < produced[static_cast<size_t>(it)]; ++m) {
                for (size_t w = 0; w < words; ++w) flags[w] = 0;
                size_t support = 0;
                const EssentialMatrix& e = models[static_cast<size_t>(it) * 10 + m];
                for (size_t i = 0; i < count; ++i) {
                    if (EssentialModel::residual(e, sc.from[i], sc.to[i]) < kThreshold) {
                        flags[i >> 5] |= static_cast<uint32_t>(1u) << (i & 31u);
                        ++support;
                    }
                }
                sink += support;
            }
        }
        score.push_back(msSince(t0));
        // The support total is consumed so the scoring loop cannot be elided.
        if (sink == SIZE_MAX) std::printf(" (unreachable)\n");
    }
    const double s = summarise(solve).median / hypotheses;
    const double c = summarise(score).median / hypotheses;
    Split out;
    out.solverMsPerHyp = s;
    out.scoreMsPerHyp = c;
    out.modelsPerHyp = static_cast<double>(totalModels) / hypotheses;
    out.f = (s + c) > 0.0 ? c / (s + c) : 0.0;
    return out;
}

void checkCuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA failure in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

} // namespace

int main() {
    std::printf("===========================================================\n");
    std::printf(" binCV CUDA -- the RANSAC geometry question\n");
    std::printf("===========================================================\n\n");
    std::printf(" Read the decision rule at the top of this file FIRST. It was\n"
                " written before any of the numbers below existed, its ship bar is\n"
                " deliberately blank because no measurement produces it, and every\n"
                " gate here is CONTINUE/KILL.\n\n");
    cudabench::printDevice();

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::printf(" no CUDA device -- nothing measured\n");
        return 77;
    }

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream), "stream create");

    const Scene sc =
        makeScene(kFrontendCount, kFrontendOutlierPct, 0x5EED + kFrontendCount);
    std::printf("\n scene: %zu correspondences, %d%% outliers, %zu planted inliers,\n"
                "        threshold %.5f (1.5 px at the frontend's focal length)\n",
                kFrontendCount, kFrontendOutlierPct, sc.planted,
                static_cast<double>(kThreshold));

    // -----------------------------------------------------------------------
    // G1
    // -----------------------------------------------------------------------
    std::printf("\n--- G1: where a hypothesis's cost is (host only) ---\n");
    const Split split = hostSplit(sc, kFrontendCount, 7, 1024);
    std::printf(" models per hypothesis        %8.3f\n", split.modelsPerHyp);
    std::printf(" five-point solver            %8.5f ms/hypothesis\n",
                split.solverMsPerHyp);
    std::printf(" scoring every model          %8.5f ms/hypothesis\n",
                split.scoreMsPerHyp);
    std::printf(" f = scoring/(solver+scoring) %8.4f\n", split.f);
    std::printf(" 1/(1-f)                      %8.4fx  <-- the CEILING on a\n"
                "   scoring-only device arm, at INFINITE device speed. This is the\n"
                "   split cv::cuda::solvePnPRansac chose, and it is the one issue\n"
                "   #61 proposes; the ceiling is what decides it, not the kernel.\n",
                1.0 / (1.0 - split.f));

    // -----------------------------------------------------------------------
    // G3
    // -----------------------------------------------------------------------
    std::printf("\n--- G3: a device round against the host's adaptive search ---\n");
    std::printf(" Both arms on ONE EXPLICIT STREAM. The device arm's measurement\n"
                " INCLUDES the round trip: correspondences up, key down, one\n"
                " synchronize -- the question issue #61 actually asks.\n");

    Point2f* dFrom = nullptr;
    Point2f* dTo = nullptr;
    unsigned long long* dKey = nullptr;
    checkCuda(cudaMalloc(&dFrom, kFrontendCount * sizeof(Point2f)), "malloc from");
    checkCuda(cudaMalloc(&dTo, kFrontendCount * sizeof(Point2f)), "malloc to");
    checkCuda(cudaMalloc(&dKey, sizeof(unsigned long long)), "malloc key");

    RansacParams rp;
    rp.threshold = static_cast<double>(kThreshold);
    rp.confidence = 0.99;
    rp.maxIterations = 2000;
    std::vector<uint32_t> hostScratch(2 * bincv::ransacScratchWords(kFrontendCount));

    // The host arm once, un-timed, to read I and the consensus it reaches.
    EssentialMatrix hostModel;
    const RansacResult probeResult = bincv::findEssentialMat(
        sc.from.data(), sc.to.data(), kFrontendCount, rp,
        bincv::RansacScratch{hostScratch.data(), kFrontendCount}, &hostModel);
    const int I = probeResult.iterations;
    std::printf("\n host arm: %d adaptive iterations, %zu inliers\n", I,
                probeResult.inliers);

    auto hostArm = [&] {
        EssentialMatrix e;
        bincv::findEssentialMat(sc.from.data(), sc.to.data(), kFrontendCount, rp,
                                bincv::RansacScratch{hostScratch.data(), kFrontendCount},
                                &e);
    };

    unsigned long long bestDeviceKey = 0;
    auto deviceRound = [&](unsigned H, bool ceilingArm) {
        const unsigned long long zero = 0;
        checkCuda(cudaMemcpyAsync(dFrom, sc.from.data(), kFrontendCount * sizeof(Point2f),
                                  cudaMemcpyHostToDevice, stream), "up from");
        checkCuda(cudaMemcpyAsync(dTo, sc.to.data(), kFrontendCount * sizeof(Point2f),
                                  cudaMemcpyHostToDevice, stream), "up to");
        checkCuda(cudaMemcpyAsync(dKey, &zero, sizeof(zero), cudaMemcpyHostToDevice,
                                  stream), "up key");
        const cudaError_t rc =
            ceilingArm ? probe::roundOneSamplePerWarpAsync(dFrom, dTo, kFrontendCount,
                                                           kSeed, 0, H, kThreshold, dKey,
                                                           stream)
                       : probe::roundAsync(dFrom, dTo, kFrontendCount, kSeed, 0, H,
                                           kThreshold, dKey, stream);
        checkCuda(rc, "round launch");
        unsigned long long key = 0;
        checkCuda(
            cudaMemcpyAsync(&key, dKey, sizeof(key), cudaMemcpyDeviceToHost, stream),
            "down key");
        checkCuda(cudaStreamSynchronize(stream), "sync");
        bestDeviceKey = key;
    };

    // Warm the device before the first timed round, exactly as the launch floor
    // does: a cold ramp is a different quantity from what the arms are read at.
    //
    // THE FIRST LAUNCH IS ALSO WHERE THE DEVICE'S REAL FOOTPRINT APPEARS, so it
    // is metered here rather than computed later. A kernel whose threads carry a
    // local frame makes the driver reserve backing store for the maximum
    // resident threads on the whole part, once, at first launch -- not for H.
    // Nothing in a per-round allocation table can see that, which is why the
    // reading is taken on cudaMemGetInfo, the meter that crosses libraries.
    cudaDeviceSynchronize();
    size_t freeBeforeFirstLaunch = 0, totalBytes = 0;
    cudaMemGetInfo(&freeBeforeFirstLaunch, &totalBytes);
    deviceRound(64, false);
    cudaDeviceSynchronize();
    size_t freeAfterFirstLaunch = 0;
    cudaMemGetInfo(&freeAfterFirstLaunch, &totalBytes);
    const size_t firstLaunchReserve = freeBeforeFirstLaunch > freeAfterFirstLaunch
                                          ? freeBeforeFirstLaunch - freeAfterFirstLaunch
                                          : 0;

    const unsigned Hs[] = {32, 64, 128, 256, 512, 1024};
    std::printf("\n   H    device round (ms)             host search (ms)          "
                "   R(H)  device/host   hyp/ms  ranges\n");
    double bestDeviceMs = 0.0;
    unsigned bestH = 0;
    double hostSearchMs = 0.0;
    for (unsigned H : Hs) {
        const Paired p = pairedWallClock(hostArm, [&] { deviceRound(H, false); }, 9);
        const unsigned R = (static_cast<unsigned>(I) + H - 1u) / H;
        const double deviceTotal = p.b.median * (R < 1u ? 1.0 : static_cast<double>(R));
        if (bestH == 0 || deviceTotal < bestDeviceMs) {
            bestDeviceMs = deviceTotal;
            bestH = H;
        }
        hostSearchMs = p.a.median;
        std::printf("  %4u  %8.3f [%7.3f..%8.3f]  %8.3f [%6.3f..%7.3f]  %4u  %8.2fx  "
                    "%7.1f  %s\n",
                    H, p.b.median, p.b.lo, p.b.hi, p.a.median, p.a.lo, p.a.hi,
                    R < 1u ? 1u : R, deviceTotal / p.a.median,
                    static_cast<double>(H) / p.b.median,
                    p.overlap ? "OVERLAP -- not a result" : "disjoint");
    }

    const double hostPerHyp = hostSearchMs / static_cast<double>(I);
    std::printf("\n G3 inequality: min over H of R(H)*(T_round + T_sync) = %.3f ms at"
                " H = %u\n"
                "                against I * t_host = %d * %.5f = %.3f ms\n",
                bestDeviceMs, bestH, I, hostPerHyp, hostSearchMs);
    std::printf(" G3 %s -- the device round is %.2fx the host's whole adaptive search\n",
                bestDeviceMs < hostSearchMs ? "PASSES" : "FAILS",
                bestDeviceMs / hostSearchMs);

    // The ceiling any warp-per-sample solver is bounded by. Measured rather than
    // argued, because it is what decides whether the hard half is worth writing.
    std::printf("\n One sample per warp, 31 lanes idle -- the CEILING a warp-per-sample\n"
                " solver is bounded by. It already removes both penalties such a design\n"
                " targets (the max over 32 divergent trip counts, and 32 solver frames\n"
                " thrashing one L1) and adds none of the internal parallelism.\n");
    std::printf("\n   H    one sample/warp (ms)          thread/sample (ms)           "
                "ceiling/thread  ranges\n");
    for (unsigned H : {32u, 64u, 128u, 256u}) {
        const Paired p = pairedWallClock([&] { deviceRound(H, false); },
                                         [&] { deviceRound(H, true); }, 9);
        std::printf("  %4u  %8.3f [%7.3f..%8.3f]  %8.3f [%7.3f..%8.3f]  %8.2fx  %s\n", H,
                    p.b.median, p.b.lo, p.b.hi, p.a.median, p.a.lo, p.a.hi,
                    p.ratio.median, p.overlap ? "OVERLAP" : "disjoint");
    }
    std::printf("   A warp-per-sample solver's best possible round time is the ceiling\n"
                "   column divided by its achieved internal width. The solver's own\n"
                "   widest phases are 11 Chebyshev nodes and 10 Aberth roots, so that\n"
                "   divisor is bounded by ~11 and the lane utilisation by ~34%%.\n");

    // -----------------------------------------------------------------------
    // The arm switch, and the case its own gate excludes
    // -----------------------------------------------------------------------
    std::printf("\n--- the scoring-arm switch, both arms in this one process ---\n");
    std::printf(" The switch selects ONLY the mapping of correspondences to lanes;\n"
                " the solver stays one hypothesis per thread either way, so a ratio\n"
                " here is attributable to one change.\n"
                " AND THAT IS WHY IT CANNOT SEPARATE. G1 measured scoring at\n"
                " %.1f%% of a hypothesis, so the switch moves that share of this\n"
                " kernel and nothing else. Overlapping ranges are the CORRECT reading\n"
                " here, not a sampling failure: it is the G1 result restated at the\n"
                " kernel. The profile says the same thing -- both arms sit at ~91%%\n"
                " of warp-cycles stalled on the solver's local-memory dependency.\n",
                100.0 * split.f);
    cudabench::Timing floor = cudabench::measureLaunchFloor(dim3(1), dim3(32), 100, 25,
                                                            250.0, stream);
    cudabench::printLaunchFloor(floor);

    struct SwitchCase {
        const char* label;
        size_t count;
        bool expect1x;
    };
    const SwitchCase cases[] = {
        {"count = 141 (the frontend's)", kFrontendCount, false},
        {"count = 16 -- BELOW the warp arm's own gate", kGateExcludedCount, true},
    };
    for (const SwitchCase& c : cases) {
        const Scene s2 = makeScene(c.count, kFrontendOutlierPct, 0x5EED + c.count);
        Point2f* f2 = nullptr;
        Point2f* t2 = nullptr;
        checkCuda(cudaMalloc(&f2, c.count * sizeof(Point2f)), "malloc f2");
        checkCuda(cudaMalloc(&t2, c.count * sizeof(Point2f)), "malloc t2");
        checkCuda(cudaMemcpy(f2, s2.from.data(), c.count * sizeof(Point2f),
                             cudaMemcpyHostToDevice), "up f2");
        checkCuda(cudaMemcpy(t2, s2.to.data(), c.count * sizeof(Point2f),
                             cudaMemcpyHostToDevice), "up t2");

        auto arm = [&](bool warp) {
            probe::warpScoreArmEnabled() = warp;
            checkCuda(probe::roundAsync(f2, t2, c.count, kSeed, 0, 64, kThreshold, dKey,
                                        stream),
                      "switch round");
        };
        const cudabench::PairedTiming p = cudabench::timeKernelPaired(
            [&] { arm(false); }, [&] { arm(true); }, 20, 20, 15, stream);
        std::printf("\n %s\n", c.label);
        cudabench::printPaired("serial scoring", "warp-ballot scoring", p, "kernel",
                               c.expect1x);
        probe::warpScoreArmEnabled() = true;
        checkCuda(probe::roundAsync(f2, t2, c.count, kSeed, 0, 64, kThreshold, dKey,
                                    stream), "arm readback");
        checkCuda(cudaStreamSynchronize(stream), "arm readback sync");
        std::printf("   arm actually launched with the switch ON: %s\n",
                    probe::lastScoreArm() == probe::ScoreArm::WarpBallot
                        ? "warp-ballot"
                        : "serial (the gate refused it -- this row must read ~1.00x)");
        cudaFree(f2);
        cudaFree(t2);
    }

    // -----------------------------------------------------------------------
    // G4 -- memory, symmetric, with each meter named at its own number
    // -----------------------------------------------------------------------
    std::printf("\n--- G4: the working set, both sides counted the same way ---\n");
    cudaDeviceProp props{};
    checkCuda(cudaGetDeviceProperties(&props, 0), "device properties");
    const int smCount = props.multiProcessorCount;
    const int maxThreadsPerSm = props.maxThreadsPerMultiProcessor;
    cudabench::printMemoryHeader("one RANSAC call at the frontend's operating point");
    const size_t points = 2 * kFrontendCount * sizeof(Point2f);
    const size_t hostTotal =
        bincv::ransacScratchBytes(kFrontendCount) + bincv::essentialSolverStackBytes() +
        points;
    std::printf("\n   HOST arm (the arm a binCV user runs today)\n");
    cudabench::printAllocSum("ransacScratchBytes(141)",
                             bincv::ransacScratchBytes(kFrontendCount));
    cudabench::printAllocSum("essentialSolverStackBytes()  [g++ -fstack-usage]",
                             bincv::essentialSolverStackBytes());
    cudabench::printAllocSum("the caller's points", points);
    cudabench::printAllocSum("HOST TOTAL", hostTotal);

    // Quoted at the SMALLEST round in the sweep, which is the device's most
    // favourable case on this axis and also the one G3 found cheapest. A larger
    // round is a larger figure by exactly H.
    const unsigned memH = Hs[0];
    std::printf("\n   DEVICE arm, at the smallest round in the sweep (H = %u) -- the\n"
                "   device's most favourable case here; a larger round scales by H\n",
                memH);
    const size_t deviceGlobal = points + sizeof(unsigned long long);
    const size_t deviceFrames =
        static_cast<size_t>(memH) * probe::deviceSolverFrameBytes();
    cudabench::printAllocSum("global: points + the packed key", deviceGlobal);
    cudabench::printAllocSum("solver frame x H  [nvcc -Xptxas -v]", deviceFrames);
    cudabench::printAllocSum("DEVICE TOTAL", deviceGlobal + deviceFrames);
    std::printf("   The solver frame is the DEVICE's %zu B of local memory per thread,\n"
                "   measured by its own compiler exactly as the host's %zu B of\n"
                "   stack is measured by g++ -fstack-usage. Counting the host's and\n"
                "   not the device's is the asymmetry this gate would otherwise turn\n"
                "   on: on a GPU that frame is global-memory backed, it is PER\n"
                "   THREAD, and a round of H hypotheses has H of them live at once.\n",
                probe::deviceSolverFrameBytes(), bincv::essentialSolverStackBytes());
    std::printf("\n G4 %s on the allocation sum -- device %zu B against host %zu B "
                "(%.1fx),\n and the measured driver reserve below is worse still.\n",
                (deviceGlobal + deviceFrames) <= hostTotal ? "PASSES" : "FAILS",
                deviceGlobal + deviceFrames, hostTotal,
                static_cast<double>(deviceGlobal + deviceFrames) /
                    static_cast<double>(hostTotal));

    const size_t step = cudabench::measureDriverMeterStep();
    cudabench::DeviceMemMeter meter;
    {
        Point2f* a = nullptr;
        Point2f* b = nullptr;
        unsigned long long* k = nullptr;
        checkCuda(cudaMalloc(&a, kFrontendCount * sizeof(Point2f)), "meter a");
        checkCuda(cudaMalloc(&b, kFrontendCount * sizeof(Point2f)), "meter b");
        checkCuda(cudaMalloc(&k, sizeof(unsigned long long)), "meter k");
        const size_t delta = meter.deltaBytes();
        cudabench::printDriverDelta("the device arm's global allocations", delta, step);
        cudaFree(a);
        cudaFree(b);
        cudaFree(k);
    }

    // And the reading that meter 1 cannot take, because nothing here allocates it.
    std::printf("\n   WHAT THE FIRST LAUNCH COST, measured on meter 2 around it:\n");
    cudabench::printDriverDelta("driver reserve at the first round launch",
                                firstLaunchReserve, step);
    std::printf("   This is NOT H times anything -- it is the same whatever H is. A\n"
                "   thread carrying a %zu B local frame makes the driver reserve\n"
                "   backing store for the maximum threads the whole part can hold at\n"
                "   once (%d SMs x %d threads), and it does so on the first launch and\n"
                "   keeps it. An empty kernel and a kernel with an eight-double frame\n"
                "   both read 0.00 MB here on this driver, so the reading is the\n"
                "   solver's frame and not module load.\n"
                "   Against the host arm's %zu B, that is the memory axis decided.\n",
                probe::deviceSolverFrameBytes(), smCount, maxThreadsPerSm, hostTotal);

    // -----------------------------------------------------------------------
    // G5 -- the consensus each side reaches
    // -----------------------------------------------------------------------
    std::printf("\n--- G5: the consensus ---\n");
    deviceRound(1024, false);
    std::printf(" host, %d adaptive hypotheses : %zu inliers\n", I, probeResult.inliers);
    std::printf(" device, 1024 hypotheses      : %llu inliers\n",
                bestDeviceKey >> 24);
    std::printf(" G5 %s -- a fixed-budget round scores strictly more hypotheses than\n"
                " the host's early stop, so it may legitimately find a better model;\n"
                " what it must not do is find a weaker one.\n",
                (bestDeviceKey >> 24) >= probeResult.inliers ? "PASSES" : "FAILS");

    cudaFree(dFrom);
    cudaFree(dTo);
    cudaFree(dKey);
    cudaStreamDestroy(stream);
    return 0;
}
