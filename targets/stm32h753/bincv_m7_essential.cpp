/// @file bincv_m7_essential.cpp
/// @brief The five-point essential matrix under RANSAC on the M7, over the scenes of
/// benchmark/essential_scene.hpp.
///
///   ESSENTIAL       per scene: what the estimator found -- inliers, iterations, a digest
///                   of the model's bits and one of the inlier mask -- for the host's
///                   (m7_reference essential) to be held to; then cycles, median of 3,
///                   with the published scoring and with the squared test (digests of
///                   both, which must agree), and with the real-root solver; then the
///                   five-point solve alone, both solvers, over 200 samples
///   ESSENTIALDUMP   per scene: the model's nine doubles and the inlier mask, in full,
///                   from both solvers (E/M published, R/Q the real-root solver),
///                   for a difference from the host's to be measured rather than only
///                   detected (m7_reference essential --dump prints the same lines)
///
/// Double precision throughout, on the M7's double-precision FPU. Nothing is allocated:
/// the inlier flags are the caller's scratch, the solver's working set is stack.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "board.h"
#include "essential_scene.hpp"
#include "m7_harness.hpp"

namespace {

using namespace essentialscene;

constexpr size_t kRepeats = 3;

bincv::Point2f g_from[kMaxCount];
bincv::Point2f g_to[kMaxCount];
uint8_t g_mask[kMaxCount];
uint32_t g_flags[2 * bincv::ransacScratchWords(kMaxCount)];

void cmdEssential(const char*) {
    boardPuts("-- five-point RANSAC, normalised coordinates, threshold 0.002, cap 500 --\n");
    for (const Case& c : kCases) {
        const size_t planted = build(g_from, g_to, c.count, c.outlierPct, c.seed);
        const bincv::RansacScratch scratch{g_flags, c.count};
        const bincv::RansacParams p = params();
        bincv::EssentialMatrix model;
        bincv::RansacResult r;
        uint32_t cycles[kRepeats], residualCycles[kRepeats];
        uint32_t residualModel = 0, residualMask = 0;
        size_t residualInliers = 0;
        for (size_t k = 0; k < kRepeats; ++k) {
            // The published scoring, `residual < threshold` at every point, then the
            // squared test, which this image runs by default.
            bincv::impl::ransacSquaredTestEnabled() = false;
            uint32_t t0 = boardCycles();
            r = bincv::findEssentialMat(g_from, g_to, c.count, p, scratch, &model, g_mask);
            residualCycles[k] = boardCycles() - t0;
            residualInliers = r.inliers;
            residualModel = fnv1a(kFnvSeed, model.m, sizeof model.m);
            residualMask = fnv1a(kFnvSeed, g_mask, c.count);
            bincv::impl::ransacSquaredTestEnabled() = true;
            t0 = boardCycles();
            r = bincv::findEssentialMat(g_from, g_to, c.count, p, scratch, &model, g_mask);
            cycles[k] = boardCycles() - t0;
        }
        boardPuts("  ");
        boardPutU32(static_cast<uint32_t>(c.count));
        boardPuts(" correspondences, ");
        boardPutU32(static_cast<uint32_t>(c.outlierPct));
        boardPuts("% outliers: found ");
        boardPutU32(r.found ? 1u : 0u);
        boardPuts(" inliers ");
        boardPutU32(static_cast<uint32_t>(r.inliers));
        boardPuts(" of ");
        boardPutU32(static_cast<uint32_t>(planted));
        boardPuts(" planted, iterations ");
        boardPutU32(static_cast<uint32_t>(r.iterations));
        boardPuts(", model ");
        boardPutHex32(fnv1a(kFnvSeed, model.m, sizeof model.m));
        boardPuts(", mask ");
        boardPutHex32(fnv1a(kFnvSeed, g_mask, c.count));
        boardPuts("\n    residual scoring: inliers ");
        boardPutU32(static_cast<uint32_t>(residualInliers));
        boardPuts(", model ");
        boardPutHex32(residualModel);
        boardPuts(", mask ");
        boardPutHex32(residualMask);
        boardPuts("\n    cycles, residual scoring  ");
        boardPutU32(m7::medianOf(residualCycles, kRepeats));
        boardPuts(" cyc\n    cycles, squared test  ");
        boardPutU32(m7::medianOf(cycles, kRepeats));
        boardPuts(" cyc\n");

        // The same scene through the solver built for a small core.
        bincv::EssentialMatrix rrModel;
        bincv::RansacResult rr;
        uint32_t rrCycles[kRepeats];
        for (size_t k = 0; k < kRepeats; ++k) {
            const uint32_t t0 = boardCycles();
            rr = bincv::findEssentialMatRealRoots(g_from, g_to, c.count, p, scratch, &rrModel,
                                                  g_mask);
            rrCycles[k] = boardCycles() - t0;
        }
        boardPuts("    real-root solver: found ");
        boardPutU32(rr.found ? 1u : 0u);
        boardPuts(" inliers ");
        boardPutU32(static_cast<uint32_t>(rr.inliers));
        boardPuts(", iterations ");
        boardPutU32(static_cast<uint32_t>(rr.iterations));
        boardPuts(", model ");
        boardPutHex32(fnv1a(kFnvSeed, rrModel.m, sizeof rrModel.m));
        boardPuts(", mask ");
        boardPutHex32(fnv1a(kFnvSeed, g_mask, c.count));
        boardPuts("\n    cycles, real-root solver  ");
        boardPutU32(m7::medianOf(rrCycles, kRepeats));
        boardPuts(" cyc\n");
    }

    // The minimal solve alone: the last scene's correspondences five at a time, outliers
    // included, as RANSAC draws them.
    constexpr size_t kSamples = 200;
    bincv::EssentialMatrix sols[10];
    uint32_t pubSolutions = 0, rrSolutions = 0;
    uint32_t t0 = boardCycles();
    for (size_t k = 0; k < kSamples; ++k)
        pubSolutions += static_cast<uint32_t>(
            bincv::fivePointEssential(g_from + 5 * k, g_to + 5 * k, sols));
    const uint32_t pubCycles = boardCycles() - t0;
    t0 = boardCycles();
    for (size_t k = 0; k < kSamples; ++k)
        rrSolutions += static_cast<uint32_t>(
            bincv::fivePointEssentialRealRoots(g_from + 5 * k, g_to + 5 * k, sols));
    const uint32_t rrSolveCycles = boardCycles() - t0;
    boardPuts("-- the five-point solve alone, 200 samples of the last scene --\n");
    boardPuts("  published: ");
    boardPutU32(pubSolutions);
    boardPuts(" solutions, ");
    boardPutU32(pubCycles / kSamples);
    boardPuts(" cyc a solve\n  real-root solver: ");
    boardPutU32(rrSolutions);
    boardPuts(" solutions, ");
    boardPutU32(rrSolveCycles / kSamples);
    boardPuts(" cyc a solve\n");

    // Scoring alone: every correspondence of the last scene against one model, the way
    // the driver tests each hypothesis -- by the residual, and by the squared test.
    bincv::EssentialMatrix e;
    bincv::findEssentialMat(g_from, g_to, kMaxCount, params(), {g_flags, kMaxCount}, &e);
    const float threshold = static_cast<float>(kThreshold);
    uint32_t inResidual = 0, inSquared = 0;
    t0 = boardCycles();
    for (size_t i = 0; i < kMaxCount; ++i)
        inResidual += bincv::EssentialModel::residual(e, g_from[i], g_to[i]) < threshold ? 1u : 0u;
    const uint32_t residualScore = boardCycles() - t0;
    t0 = boardCycles();
    for (size_t i = 0; i < kMaxCount; ++i)
        inSquared += bincv::EssentialModel::isInlier(e, g_from[i], g_to[i], threshold) ? 1u : 0u;
    const uint32_t squaredScore = boardCycles() - t0;
    boardPuts("-- scoring alone, 1000 correspondences against one model --\n  residual: ");
    boardPutU32(inResidual);
    boardPuts(" inliers, ");
    boardPutU32(residualScore / static_cast<uint32_t>(kMaxCount));
    boardPuts(" cyc a point\n  squared test: ");
    boardPutU32(inSquared);
    boardPuts(" inliers, ");
    boardPutU32(squaredScore / static_cast<uint32_t>(kMaxCount));
    boardPuts(" cyc a point\n");
    boardPuts("OK\n");
}

void cmdEssentialDump(const char*) {
    for (const Case& c : kCases) {
        build(g_from, g_to, c.count, c.outlierPct, c.seed);
        const bincv::RansacScratch scratch{g_flags, c.count};
        bincv::EssentialMatrix model;
        bincv::findEssentialMat(g_from, g_to, c.count, params(), scratch, &model, g_mask);
        boardPuts("E ");
        boardPutU32(static_cast<uint32_t>(c.count));
        for (const double v : model.m) {
            uint64_t b;
            std::memcpy(&b, &v, sizeof b);
            boardPuts(" ");
            boardPutHex32(static_cast<uint32_t>(b >> 32));
            boardPutHex32(static_cast<uint32_t>(b));
        }
        boardPuts("\nM ");
        boardPutU32(static_cast<uint32_t>(c.count));
        boardPuts(" ");
        for (size_t i = 0; i < c.count; ++i) boardPutc(g_mask[i] ? '1' : '0');
        boardPuts("\n");
        bincv::findEssentialMatRealRoots(g_from, g_to, c.count, params(), scratch, &model,
                                         g_mask);
        boardPuts("R ");
        boardPutU32(static_cast<uint32_t>(c.count));
        for (const double v : model.m) {
            uint64_t b;
            std::memcpy(&b, &v, sizeof b);
            boardPuts(" ");
            boardPutHex32(static_cast<uint32_t>(b >> 32));
            boardPutHex32(static_cast<uint32_t>(b));
        }
        boardPuts("\nQ ");
        boardPutU32(static_cast<uint32_t>(c.count));
        boardPuts(" ");
        for (size_t i = 0; i < c.count; ++i) boardPutc(g_mask[i] ? '1' : '0');
        boardPuts("\n");
    }
    boardPuts("OK\n");
}

const m7::Command kCommands[] = {{"ESSENTIAL", &cmdEssential},
                                 {"ESSENTIALDUMP", &cmdEssentialDump}};

}  // namespace

int main() { m7::serve("essential", kCommands, sizeof(kCommands) / sizeof(kCommands[0])); }
