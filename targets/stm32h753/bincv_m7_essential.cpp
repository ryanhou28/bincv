/// @file bincv_m7_essential.cpp
/// @brief The five-point essential matrix under RANSAC on the M7, over the scenes of
/// benchmark/essential_scene.hpp.
///
///   ESSENTIAL       per scene: what the estimator found -- inliers, iterations, a digest
///                   of the model's bits and one of the inlier mask -- for the host's
///                   (m7_reference essential) to be held to; then cycles, median of 3,
///                   with the published scoring and with the squared test (digests of
///                   both, which must agree)
///   ESSENTIALDUMP   per scene: the model's nine doubles and the inlier mask, in full,
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
    }
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
    }
    boardPuts("OK\n");
}

const m7::Command kCommands[] = {{"ESSENTIAL", &cmdEssential},
                                 {"ESSENTIALDUMP", &cmdEssentialDump}};

}  // namespace

int main() { m7::serve("essential", kCommands, sizeof(kCommands) / sizeof(kCommands[0])); }
