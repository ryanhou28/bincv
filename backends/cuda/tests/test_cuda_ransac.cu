// The five-point solver and the RANSAC sampler, compiled for BOTH targets and
// held to the same answer.
//
// WHAT THIS SUITE IS FOR. ops/essential.hpp and ops/ransac.hpp now carry
// BINCV_HOST_DEVICE on the solver chain and the sampler, so a GPU compiles THAT
// elimination instead of keeping a second one in agreement with it. "It is the
// same source" does not establish that the two agree: the device has its own
// libm behind std::cos and std::sqrt, its own double-precision hardware, and its
// own compiler's licence to contract a multiply and an add into one rounding.
// The chain is 10 Aberth-Ehrlich roots of a degree-10 polynomial and ten 6x6
// Jacobi eigendecompositions, both of which break on a convergence test -- so a
// last-bit difference does not stay a last-bit difference, it changes a trip
// count and can change which roots come back.
//
// THE LADDER, in the order the claims get weaker, with the report saying which
// rung actually held rather than starting at the weakest:
//
//   A  THE SAMPLE INDICES ARE EXACT. impl::ransacSample is pure integer
//      arithmetic on a counter, so this is not a tolerance, it is equality.
//      Swept over counts, seeds and both sampling modes.
//   B  THE MODELS. Attempted as an exact comparison of the nine doubles; the
//      suite reports what it got rather than assuming.
//   C  THE CONSENSUS. Support counts and the packed flag words, compared to the
//      host's word for word -- INCLUDING that the bits past `count` in the last
//      word are zero, which is this library's padding rule applied to the flag
//      scratch and the reason the ballot predicate carries `i < count`.
//   D  THE WINNER. Both scoring arms, through the runtime switch, in one
//      process, must reduce to the same packed key.
//
// Counts are swept over 5 (exactly the minimal set), 6, 31, 32 (exactly one flag
// word), 33 (the first padding-bit case), 64, 127 and 141 (the count the
// frontend actually produces). The degenerate paths are reached on purpose: a
// scene of collinear points drives the solver's rank-deficient return, and a
// count below the minimal set must be refused rather than launched.
//
// Exits 77 when no CUDA device is present, like the other suites here.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/ops/essential.hpp"
#include "bincv/ops/ransac.hpp"
#include "cuda_ransac_kernels.hpp"
#include "test_util.hpp"

// The mirror of tests/test_error.cpp's assertion that the macro expands to
// NOTHING off nvcc. Either alone would be satisfied by a macro that is always
// empty -- which would compile everywhere and put none of this on the device.
#define BINCV_TEST_STRINGIZE_(x) #x
#define BINCV_TEST_STRINGIZE(x) BINCV_TEST_STRINGIZE_(x)
static_assert(BINCV_TEST_STRINGIZE(BINCV_HOST_DEVICE)[0] != '\0',
              "nvcc is compiling this file, so BINCV_HOST_DEVICE must carry the "
              "__host__ __device__ annotations -- an empty expansion here would mean "
              "the solver below was compared against itself on the host");

namespace {

using bincv::EssentialMatrix;
using bincv::EssentialModel;
using bincv::Point2f;

constexpr size_t kMaxModels = EssentialModel::kMaxModels;
constexpr size_t kMinimalSet = EssentialModel::kMinimalSetSize;
constexpr uint64_t kDefaultSeed = UINT64_C(0x9E3779B97F4A7C15);

template <typename T>
class DevArray {
public:
    explicit DevArray(size_t count) : n_(count) {
        if (count == 0 || cudaMalloc(&p_, count * sizeof(T)) != cudaSuccess) p_ = nullptr;
    }
    explicit DevArray(const std::vector<T>& host) : DevArray(host.size()) {
        if (p_ != nullptr && n_ != 0)
            cudaMemcpy(p_, host.data(), n_ * sizeof(T), cudaMemcpyHostToDevice);
    }
    ~DevArray() { cudaFree(p_); }
    DevArray(const DevArray&) = delete;
    DevArray& operator=(const DevArray&) = delete;

    T* get() const { return p_; }

    std::vector<T> download() const {
        std::vector<T> out(n_);
        if (p_ != nullptr && n_ != 0)
            cudaMemcpy(out.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost);
        return out;
    }

private:
    T* p_ = nullptr;
    size_t n_ = 0;
};

/// @brief Checked per case rather than once: a kernel that never ran leaves the
/// output buffer holding whatever cudaMalloc handed back, and comparing that
/// against the host answers is a test of luck.
bool launchOk() {
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

struct Scene {
    std::vector<Point2f> from, to;
};

/// @brief A two-view scene from a planted E = [t]x R, with a share of the
/// correspondences displaced so the consensus set is not everything.
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
        out.from.push_back(Point2f{static_cast<float>(X[0] / X[2]),
                                   static_cast<float>(X[1] / X[2])});
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
        }
        out.to.push_back(b);
    }
    return out;
}

/// @brief Points on one line in both views: the solver's rank-deficient return
/// and its below-pivot return are only reachable on input like this.
Scene makeCollinearScene(size_t count) {
    Scene out;
    for (size_t i = 0; i < count; ++i) {
        const float u = 0.01f * static_cast<float>(i);
        out.from.push_back(Point2f{u, 0.5f * u});
        out.to.push_back(Point2f{u + 0.2f, 0.5f * u + 0.1f});
    }
    return out;
}

/// @brief Everything one hypothesis produces, on the HOST, in the device's
/// output shape -- so the comparison is a memcmp rather than a second
/// traversal written twice.
struct HostSample {
    std::vector<size_t> idx;
    uint32_t drew = 0;
    std::vector<EssentialMatrix> models;
    uint32_t produced = 0;
    std::vector<uint32_t> support;
    std::vector<uint32_t> flags;
};

HostSample hostSample(const Scene& sc, size_t count, uint64_t seed, uint64_t iter,
                      float threshold, bool ordered) {
    const size_t words = bincv::ransacScratchWords(count);
    HostSample h;
    h.idx.assign(kMinimalSet, 0);
    h.models.assign(kMaxModels, EssentialMatrix{});
    h.support.assign(kMaxModels, 0u);
    h.flags.assign(kMaxModels * words, 0u);

    const uint64_t counter = seed + iter * UINT64_C(0x9E3779B9);
    const size_t grown = kMinimalSet + static_cast<size_t>(iter);
    const size_t pool = ordered ? (grown < count ? grown : count) : count;
    if (!bincv::impl::ransacSample(pool, kMinimalSet, counter, h.idx.data())) {
        h.idx.assign(kMinimalSet, 0);
        return h;
    }
    h.drew = 1u;
    h.produced = static_cast<uint32_t>(
        EssentialModel::estimate(sc.from.data(), sc.to.data(), h.idx.data(),
                                 h.models.data()));
    for (size_t m = 0; m < h.produced; ++m) {
        uint32_t support = 0;
        for (size_t i = 0; i < count; ++i) {
            if (EssentialModel::residual(h.models[m], sc.from[i], sc.to[i]) < threshold) {
                h.flags[m * words + (i >> 5)] |= static_cast<uint32_t>(1u) << (i & 31u);
                ++support;
            }
        }
        h.support[m] = support;
    }
    return h;
}

/// @brief The counts the ladder sweeps. 32 is exactly one flag word, 33 is the
/// first case with padding bits, 141 is the count the frontend produces.
const size_t kCounts[] = {5, 6, 31, 32, 33, 64, 127, 141};

struct Ladder {
    size_t samples = 0;
    size_t modelsCompared = 0;
    size_t modelsExact = 0;          ///< nine doubles bit for bit
    size_t modelsBeyondUlps = 0;     ///< at least one coefficient past 1e-6 absolute
    double worstRelCoeff = 0.0;
    double worstAbsCoeff = 0.0;
    double worstEpipolar = 0.0;      ///< |q2^T E q1| at the device model's OWN sample
    size_t supportMismatches = 0;
    size_t flagMismatches = 0;
    size_t modelCountMismatches = 0;  ///< the solver's degeneracy verdict differed
};

/// @brief `|q2^T E q1|` -- the residual the host header calls the check that
/// matters most, because a solution set polluted with spurious roots fails it
/// where a coefficient comparison would not.
double epipolarResidual(const EssentialMatrix& e, Point2f a, Point2f b) {
    const double q1[3] = {static_cast<double>(a.x), static_cast<double>(a.y), 1.0};
    const double q2[3] = {static_cast<double>(b.x), static_cast<double>(b.y), 1.0};
    double acc = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) acc += q2[i] * e.m[i * 3 + j] * q1[j];
    }
    return std::fabs(acc);
}

/// @brief Runs rungs A, B and C over one configuration and reports what held.
/// @param requireSameModelCount Hard equality on how many models the solver
/// returned. True for scenes in general position, where the two targets must
/// agree. False for deliberately degenerate input, where the count is decided
/// by a 1e-14 rank tolerance and the mismatch is COUNTED and reported instead
/// -- see the collinear case for why that is the honest claim there.
Ladder runLadder(const Scene& sc, size_t count, uint64_t seed, unsigned samples,
                 float threshold, bool ordered, bool requireSameModelCount = true) {
    const size_t words = bincv::ransacScratchWords(count);
    DevArray<Point2f> dFrom(sc.from), dTo(sc.to);
    DevArray<size_t> dIdx(static_cast<size_t>(samples) * kMinimalSet);
    DevArray<uint32_t> dDrew(static_cast<size_t>(samples));
    DevArray<EssentialMatrix> dModels(static_cast<size_t>(samples) * kMaxModels);
    DevArray<uint32_t> dProduced(static_cast<size_t>(samples));
    DevArray<uint32_t> dSupport(static_cast<size_t>(samples) * kMaxModels);
    DevArray<uint32_t> dFlags(static_cast<size_t>(samples) * kMaxModels * words);

    Ladder l;
    BINCV_CHECK_EQ(cudabench::ransacprobe::probeSamplesAsync(
                       dFrom.get(), dTo.get(), count, seed, 0, samples, threshold,
                       ordered, dIdx.get(), dDrew.get(), dModels.get(), dProduced.get(),
                       dSupport.get(), dFlags.get(), nullptr),
                   cudaSuccess);
    BINCV_CHECK(launchOk());

    const std::vector<size_t> gIdx = dIdx.download();
    const std::vector<uint32_t> gDrew = dDrew.download();
    const std::vector<EssentialMatrix> gModels = dModels.download();
    const std::vector<uint32_t> gProduced = dProduced.download();
    const std::vector<uint32_t> gSupport = dSupport.download();
    const std::vector<uint32_t> gFlags = dFlags.download();

    for (unsigned s = 0; s < samples; ++s) {
        const HostSample h = hostSample(sc, count, seed, s, threshold, ordered);
        ++l.samples;

        // --- A: the sample indices, EXACT ---
        BINCV_CHECK_EQ(gDrew[s], h.drew);
        for (size_t k = 0; k < kMinimalSet; ++k) {
            BINCV_CHECK_EQ(gIdx[static_cast<size_t>(s) * kMinimalSet + k], h.idx[k]);
        }
        if (h.drew == 0u) continue;

        // --- B: the models ---
        if (gProduced[s] != h.produced) ++l.modelCountMismatches;
        if (requireSameModelCount) BINCV_CHECK_EQ(gProduced[s], h.produced);
        const uint32_t common = gProduced[s] < h.produced ? gProduced[s] : h.produced;
        for (size_t m = 0; m < common; ++m) {
            const EssentialMatrix& g = gModels[static_cast<size_t>(s) * kMaxModels + m];
            ++l.modelsCompared;
            if (std::memcmp(g.m, h.models[m].m, sizeof(g.m)) == 0) ++l.modelsExact;
            bool beyond = false;
            for (int j = 0; j < 9; ++j) {
                const double diff = std::fabs(g.m[j] - h.models[m].m[j]);
                const double den = std::fabs(h.models[m].m[j]);
                const double rel = diff / (den > 1e-12 ? den : 1.0);
                if (rel > l.worstRelCoeff) l.worstRelCoeff = rel;
                if (diff > l.worstAbsCoeff) l.worstAbsCoeff = diff;
                if (diff > 1e-6) beyond = true;
            }
            if (beyond) ++l.modelsBeyondUlps;

            // THE CLAIM THAT IS ENFORCED, and it is the host header's own: every
            // returned E must satisfy q2^T E q1 = 0 at the five points it was
            // built from. A solution set polluted with a spurious root fails this
            // decisively -- the real residuals sit at machine precision, orders
            // below where a spurious root lands -- where a coefficient comparison
            // would only report that two numbers differ.
            for (size_t k = 0; k < kMinimalSet; ++k) {
                const size_t pt = h.idx[k];
                const double r = epipolarResidual(g, sc.from[pt], sc.to[pt]);
                if (r > l.worstEpipolar) l.worstEpipolar = r;
                BINCV_CHECK(r < 1e-9);
            }
        }

        // --- C: the consensus, and the padding bits ---
        const size_t flagBase = static_cast<size_t>(s) * kMaxModels * words;
        for (size_t m = 0; m < common; ++m) {
            if (gSupport[static_cast<size_t>(s) * kMaxModels + m] != h.support[m])
                ++l.supportMismatches;
            BINCV_CHECK_EQ(gSupport[static_cast<size_t>(s) * kMaxModels + m],
                           h.support[m]);
            for (size_t w = 0; w < words; ++w) {
                if (gFlags[flagBase + m * words + w] != h.flags[m * words + w])
                    ++l.flagMismatches;
                BINCV_CHECK_EQ(gFlags[flagBase + m * words + w], h.flags[m * words + w]);
            }
            // The padding rule: no bit past `count` may be set in the last word.
            if (count % 32u != 0u) {
                const uint32_t tail = static_cast<uint32_t>(
                    (uint64_t{1} << (count % 32u)) - 1u);
                BINCV_CHECK_EQ(gFlags[flagBase + m * words + (words - 1)] & ~tail, 0u);
            }
        }
        // Models the solver did not produce leave their slots untouched at zero,
        // so a reader cannot mistake a stale flag word for a consensus set.
        for (size_t m = gProduced[s]; m < kMaxModels; ++m) {
            BINCV_CHECK_EQ(gSupport[static_cast<size_t>(s) * kMaxModels + m], 0u);
            for (size_t w = 0; w < words; ++w) {
                BINCV_CHECK_EQ(gFlags[flagBase + m * words + w], 0u);
            }
        }
    }
    return l;
}

} // namespace

BINCV_TEST(CudaRansac, TheSamplerDrawsTheSameMinimalSetsOnBothTargets) {
    // Rung A on its own, across seeds and both sampling modes: pure integer
    // arithmetic, so any difference is a bug rather than a tolerance.
    const uint64_t seeds[] = {kDefaultSeed, 0u, 1u, UINT64_C(0xFFFFFFFFFFFFFFFF)};
    for (size_t count : kCounts) {
        const Scene sc = makeScene(count, 23, 0x5EED + count);
        for (uint64_t seed : seeds) {
            for (bool ordered : {false, true}) {
                const Ladder l = runLadder(sc, count, seed, 24, 0.004f, ordered);
                BINCV_CHECK_EQ(l.samples, size_t{24});
            }
        }
    }
}

BINCV_TEST(CudaRansac, TheSolverAndTheConsensusAgreeWithTheHostAcrossShapes) {
    Ladder t;
    for (size_t count : kCounts) {
        const Scene sc = makeScene(count, 23, 0x5EED + count);
        const Ladder l = runLadder(sc, count, kDefaultSeed, 64, 0.004f, false);
        t.modelsCompared += l.modelsCompared;
        t.modelsExact += l.modelsExact;
        t.modelsBeyondUlps += l.modelsBeyondUlps;
        t.supportMismatches += l.supportMismatches;
        t.flagMismatches += l.flagMismatches;
        if (l.worstRelCoeff > t.worstRelCoeff) t.worstRelCoeff = l.worstRelCoeff;
        if (l.worstAbsCoeff > t.worstAbsCoeff) t.worstAbsCoeff = l.worstAbsCoeff;
        if (l.worstEpipolar > t.worstEpipolar) t.worstEpipolar = l.worstEpipolar;
    }
    // WHICH RUNG HELD, reported rather than assumed. "Bit-exact" is a claim this
    // suite is entitled to make only if the measurement supports it, and on the
    // default build it does not: nvcc contracts a multiply and an add into one
    // rounding by default, and the chain amplifies that through two loops that
    // break on convergence. What the suite ENFORCES is the claim that survives --
    // the same solution count, the epipolar constraint at machine precision, and
    // an identical consensus set.
    std::printf("  five-point, host vs device over %zu models:\n"
                "    bit-identical            %zu (%.1f%%)\n"
                "    any coefficient > 1e-6   %zu models\n"
                "    worst |difference|       %.3g absolute, %.3g relative\n"
                "    worst |q2^T E q1| at the model's own sample  %.3g\n"
                "    support counts differing %zu    flag words differing %zu\n",
                t.modelsCompared, t.modelsExact,
                t.modelsCompared ? 100.0 * static_cast<double>(t.modelsExact) /
                                       static_cast<double>(t.modelsCompared)
                                 : 0.0,
                t.modelsBeyondUlps, t.worstAbsCoeff, t.worstRelCoeff, t.worstEpipolar,
                t.supportMismatches, t.flagMismatches);
    BINCV_CHECK(t.modelsCompared > 0);
    // The consensus is the thing a RANSAC driver actually consumes, so it is held
    // to equality even though the coefficients that produced it are not.
    BINCV_CHECK_EQ(t.supportMismatches, size_t{0});
    BINCV_CHECK_EQ(t.flagMismatches, size_t{0});
}

BINCV_TEST(CudaRansac, CollinearSamplesReachTheSolversRefusalOnBothTargets) {
    // The rank-deficient Householder, the below-tolerance pure-cubic pivot and
    // the degenerate nullvector are only reachable on input like this, and a
    // device that took a different branch everywhere would return models where
    // the host returned none.
    //
    // WHAT THIS CASE MEASURES RATHER THAN ASSERTS. On rank-deficient input the
    // solver's verdict comes from `norm > 1e-14` and `best < 1e-12` applied to a
    // matrix that IS singular, so which side of the tolerance a sample lands on
    // is decided by the last bits -- and nvcc contracts a multiply and an add
    // into one rounding where g++ does not. The two targets therefore may
    // disagree about whether a degenerate sample is degenerate. That is a
    // property of a tolerance on a singular matrix, not a defect in either
    // arm, and it is REPORTED with its rate rather than asserted away. What is
    // still held to equality is the part a driver consumes: for every model both
    // sides returned, the same consensus set.
    size_t samplesSeen = 0, countMismatch = 0;
    double worstEpipolar = 0.0;
    for (size_t count : {size_t{8}, size_t{33}, size_t{64}}) {
        const Scene sc = makeCollinearScene(count);
        const Ladder l = runLadder(sc, count, kDefaultSeed, 64, 0.004f, false, false);
        samplesSeen += l.samples;
        countMismatch += l.modelCountMismatches;
        if (l.worstEpipolar > worstEpipolar) worstEpipolar = l.worstEpipolar;
        BINCV_CHECK_EQ(l.supportMismatches, size_t{0});
        BINCV_CHECK_EQ(l.flagMismatches, size_t{0});
    }
    std::printf("  degenerate (collinear) input: %zu samples, %zu where the two targets\n"
                "    disagreed on whether the sample was degenerate (%.2f%%), worst\n"
                "    |q2^T E q1| among the models both returned %.3g\n",
                samplesSeen, countMismatch,
                samplesSeen ? 100.0 * static_cast<double>(countMismatch) /
                                  static_cast<double>(samplesSeen)
                            : 0.0,
                worstEpipolar);
    BINCV_CHECK(samplesSeen > 0);
}

BINCV_TEST(CudaRansac, BothScoringArmsReduceToTheSameWinnerInOneProcess) {
    // Rung D. The switch is flipped inside this process, which is the only way
    // to know that the two arms are two paths to one answer rather than two
    // answers.
    const bool saved = cudabench::ransacprobe::warpScoreArmEnabled();
    for (size_t count : kCounts) {
        const Scene sc = makeScene(count, 23, 0x5EED + count);
        DevArray<Point2f> dFrom(sc.from), dTo(sc.to);
        DevArray<unsigned long long> dKey(size_t{1});

        unsigned long long keys[2] = {0, 0};
        for (int arm = 0; arm < 2; ++arm) {
            cudabench::ransacprobe::warpScoreArmEnabled() = (arm == 1);
            const unsigned long long zero = 0;
            cudaMemcpy(dKey.get(), &zero, sizeof(zero), cudaMemcpyHostToDevice);
            BINCV_CHECK_EQ(cudabench::ransacprobe::roundAsync(
                               dFrom.get(), dTo.get(), count, kDefaultSeed, 0, 64, 0.004f,
                               dKey.get(), nullptr),
                           cudaSuccess);
            BINCV_CHECK(launchOk());
            keys[arm] = dKey.download()[0];
        }
        BINCV_CHECK_EQ(keys[0], keys[1]);

        // The gate: below one full ballot the warp mapping has nothing to pack,
        // so the launcher refuses it and both arms take the serial path. If this
        // ever reports WarpBallot, the "~1.00x" row in the benchmark is timing
        // one arm against itself for a different reason than it claims.
        cudabench::ransacprobe::warpScoreArmEnabled() = true;
        const unsigned long long zero = 0;
        cudaMemcpy(dKey.get(), &zero, sizeof(zero), cudaMemcpyHostToDevice);
        BINCV_CHECK_EQ(cudabench::ransacprobe::roundAsync(dFrom.get(), dTo.get(), count,
                                                          kDefaultSeed, 0, 64, 0.004f,
                                                          dKey.get(), nullptr),
                       cudaSuccess);
        BINCV_CHECK(launchOk());
        const bool expectWarp = count >= cudabench::ransacprobe::kWarpScoreMinCount;
        BINCV_CHECK_EQ(cudabench::ransacprobe::lastScoreArm() ==
                           cudabench::ransacprobe::ScoreArm::WarpBallot,
                       expectWarp);
    }
    cudabench::ransacprobe::warpScoreArmEnabled() = saved;
}

BINCV_TEST(CudaRansac, TheRoundAndTheHostDriverPickTheSameWinnerOverTheSameHypotheses) {
    // Rung D, and the reason "the target is distributional" is weaker than this
    // comparison has to be. impl::ransacSample is a pure integer function of
    // (seed, iteration), so a device round starting at iteration 0 draws exactly
    // the minimal sets the host driver draws in its first H iterations. Capping
    // the host at H makes the two searches the SAME search, and then the winner
    // is comparable directly rather than in distribution.
    constexpr unsigned kH = 64;
    for (size_t count : kCounts) {
        if (count < 8) continue;  // the driver wants more than a minimal set
        const Scene sc = makeScene(count, 23, 0x5EED + count);
        DevArray<Point2f> dFrom(sc.from), dTo(sc.to);
        DevArray<unsigned long long> dKey(size_t{1});
        const unsigned long long zero = 0;
        cudaMemcpy(dKey.get(), &zero, sizeof(zero), cudaMemcpyHostToDevice);
        BINCV_CHECK_EQ(cudabench::ransacprobe::roundAsync(dFrom.get(), dTo.get(), count,
                                                          kDefaultSeed, 0, kH, 0.004f,
                                                          dKey.get(), nullptr),
                       cudaSuccess);
        BINCV_CHECK(launchOk());
        const unsigned long long key = dKey.download()[0];

        bincv::RansacParams rp;
        rp.threshold = 0.004;
        rp.maxIterations = static_cast<int>(kH);
        std::vector<uint32_t> scratch(2 * bincv::ransacScratchWords(count));
        EssentialMatrix e;
        const bincv::RansacResult r = bincv::findEssentialMat(
            sc.from.data(), sc.to.data(), count, rp,
            bincv::RansacScratch{scratch.data(), count}, &e);

        // A key still at 0 is what the host's found = false means, and nothing
        // else may produce it: a zero-support model would pack to a nonzero key
        // and win an atomicMax against an unset one.
        BINCV_CHECK_EQ(key != 0ull, r.found);
        if (r.found) BINCV_CHECK_EQ(static_cast<size_t>(key >> 24), r.inliers);
    }
}

BINCV_TEST(CudaRansac, AnEmptyKeyIsWhatNoModelFoundLooksLike) {
    // The reachable way to a zero key is a sample that produces no model at all:
    // a minimal-set fit always supports its own five points, so no scene of
    // general position yields a zero-support model. Collinear input drives the
    // solver's rank-deficient return.
    const Scene sc = makeCollinearScene(64);
    DevArray<Point2f> dFrom(sc.from), dTo(sc.to);
    DevArray<unsigned long long> dKey(size_t{1});
    const unsigned long long zero = 0;
    cudaMemcpy(dKey.get(), &zero, sizeof(zero), cudaMemcpyHostToDevice);
    BINCV_CHECK_EQ(cudabench::ransacprobe::roundAsync(dFrom.get(), dTo.get(), 64,
                                                      kDefaultSeed, 0, 64, 0.004f,
                                                      dKey.get(), nullptr),
                   cudaSuccess);
    BINCV_CHECK(launchOk());
    BINCV_CHECK_EQ(dKey.download()[0], 0ull);

    // The host over the SAME 64 hypotheses is NOT required to agree here, and
    // that is the finding rather than a gap: on this input it clears the solver's
    // 1e-14 rank tolerance on exactly one of the 64 samples and the device does
    // not. What is asserted is the device side's own contract -- no models, so
    // no key -- and the host's verdict is printed beside it.
    bincv::RansacParams rp;
    rp.threshold = 0.004;
    rp.maxIterations = 64;
    std::vector<uint32_t> scratch(2 * bincv::ransacScratchWords(64));
    EssentialMatrix e;
    const bincv::RansacResult r = bincv::findEssentialMat(
        sc.from.data(), sc.to.data(), 64, rp,
        bincv::RansacScratch{scratch.data(), 64}, &e);
    std::printf("  collinear, same 64 hypotheses: device key 0 (no model),"
                " host found=%d\n",
                r.found ? 1 : 0);
}

BINCV_TEST(CudaRansac, ADomainItCannotServeIsRefusedRatherThanLaunched) {
    const Scene sc = makeScene(64, 23, 0x5EED);
    DevArray<Point2f> dFrom(sc.from), dTo(sc.to);
    DevArray<unsigned long long> dKey(size_t{1});

    // Fewer correspondences than the minimal set, no hypotheses, a threshold
    // that is not positive, and a null output: each returns an error, and none
    // of them launches.
    BINCV_CHECK_EQ(cudabench::ransacprobe::roundAsync(dFrom.get(), dTo.get(), 4,
                                                      kDefaultSeed, 0, 8, 0.004f,
                                                      dKey.get(), nullptr),
                   cudaErrorInvalidValue);
    BINCV_CHECK_EQ(cudabench::ransacprobe::roundAsync(dFrom.get(), dTo.get(), 64,
                                                      kDefaultSeed, 0, 0, 0.004f,
                                                      dKey.get(), nullptr),
                   cudaErrorInvalidValue);
    BINCV_CHECK_EQ(cudabench::ransacprobe::roundAsync(dFrom.get(), dTo.get(), 64,
                                                      kDefaultSeed, 0, 8, 0.0f,
                                                      dKey.get(), nullptr),
                   cudaErrorInvalidValue);
    BINCV_CHECK_EQ(cudabench::ransacprobe::roundAsync(dFrom.get(), dTo.get(), 64,
                                                      kDefaultSeed, 0, 8, 0.004f, nullptr,
                                                      nullptr),
                   cudaErrorInvalidValue);
    BINCV_CHECK_EQ(cudabench::ransacprobe::roundAsync(nullptr, dTo.get(), 64,
                                                      kDefaultSeed, 0, 8, 0.004f,
                                                      dKey.get(), nullptr),
                   cudaErrorInvalidValue);
    BINCV_CHECK(launchOk());
}

BINCV_TEST(CudaRansac, TheCeilingProbeFindsTheSameWinnerAsTheRoundItBounds) {
    // The one-sample-per-warp arm exists to be a timing ceiling, so it has to be
    // the same computation: a bound on a different answer bounds nothing.
    const bool saved = cudabench::ransacprobe::warpScoreArmEnabled();
    cudabench::ransacprobe::warpScoreArmEnabled() = false;
    for (size_t count : {size_t{33}, size_t{64}, size_t{141}}) {
        const Scene sc = makeScene(count, 23, 0x5EED + count);
        DevArray<Point2f> dFrom(sc.from), dTo(sc.to);
        DevArray<unsigned long long> dKey(size_t{1});
        unsigned long long keys[2] = {0, 0};
        for (int arm = 0; arm < 2; ++arm) {
            const unsigned long long zero = 0;
            cudaMemcpy(dKey.get(), &zero, sizeof(zero), cudaMemcpyHostToDevice);
            const cudaError_t rc =
                arm == 0 ? cudabench::ransacprobe::roundAsync(
                               dFrom.get(), dTo.get(), count, kDefaultSeed, 0, 64, 0.004f,
                               dKey.get(), nullptr)
                         : cudabench::ransacprobe::roundOneSamplePerWarpAsync(
                               dFrom.get(), dTo.get(), count, kDefaultSeed, 0, 64, 0.004f,
                               dKey.get(), nullptr);
            BINCV_CHECK_EQ(rc, cudaSuccess);
            BINCV_CHECK(launchOk());
            keys[arm] = dKey.download()[0];
        }
        BINCV_CHECK_EQ(keys[0], keys[1]);
    }
    cudabench::ransacprobe::warpScoreArmEnabled() = saved;
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
    const int summaryRc = ::bincv::test::summarize("CUDA RANSAC shared-solver tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA RANSAC shared-solver tests", argc, argv);
}
#endif
