/// @file bincv_m7_stereo.cpp
/// @brief Sparse rectified stereo on the M7: the host benchmark's synthetic pair, built
/// here from the same source (benchmark/stereo_scene.hpp), each stage timed.
///
///   STEREO          build the pair; digests of its bits, its descriptors and the
///                   matches, for the host's to be held to; then cycles per stage,
///                   median of 5 interleaved repeats: the descriptor stage with the float
///                   and the integer row test, refinement with the staged arm off and
///                   on (digests of each pair, which must agree)
///
/// Memory: one 8-bit frame (the left, shifted in place into the right once its
/// descriptors and bits are taken) and both frames' bits in the AXI SRAM; keypoints,
/// descriptors and match records in the D2 SRAM. Nothing is allocated.

#include <cstddef>
#include <cstdint>

#include "board.h"
#include "m7_harness.hpp"
#include "stereo_scene.hpp"

namespace {

using namespace stereoscene;

constexpr size_t kStride = (kW + 31) / 32;
constexpr size_t kRepeats = 5;

__attribute__((section(".frames"), aligned(32))) uint8_t g_img[kW * kH];
__attribute__((section(".frames"), aligned(32))) uint32_t g_left[kStride * kH];
__attribute__((section(".frames"), aligned(32))) uint32_t g_right[kStride * kH];
__attribute__((section(".sram2"), aligned(32))) uint8_t g_ring[3 * kW];
__attribute__((section(".sram2"), aligned(32))) float g_kpL[2 * kKp];
__attribute__((section(".sram2"), aligned(32))) float g_kpR[2 * kKp];
__attribute__((section(".sram2"), aligned(32))) uint32_t g_dl[kKp * kWords];
__attribute__((section(".sram2"), aligned(32))) uint32_t g_dr[kKp * kWords];
__attribute__((section(".sram2"), aligned(32))) bincv::StereoMatch g_m[kKp];
bincv::BriefPattern<kBits> g_pattern;

void putDigest(const char* name, uint32_t h) {
    boardPuts("  ");
    boardPuts(name);
    boardPuts("  ");
    boardPutHex32(h);
    boardPuts("\n");
}

void putCycles(const char* name, uint32_t cycles) {
    boardPuts("  ");
    boardPuts(name);
    boardPuts("  ");
    boardPutU32(cycles);
    boardPuts(" cyc\n");
}

void cmdStereo(const char*) {
    const bincv::BinMatView<uint32_t> left(g_left, kW, kH, kStride);
    const bincv::BinMatView<uint32_t> right(g_right, kW, kH, kStride);
    bincv::makeBriefPattern<kBits>(g_pattern);
    buildKeypoints(g_kpL, g_kpR);

    // The pair, and the descriptors timed as they are taken: the left frame's, then the
    // right's from the same buffer shifted. Repeats rebuild the frame, untimed.
    uint32_t briefL[kRepeats], briefR[kRepeats];
    for (size_t r = 0; r < kRepeats; ++r) {
        buildLeft(g_img, g_ring);
        pack(g_img, left);
        uint32_t t0 = boardCycles();
        bincv::computeBrief<kBits, uint8_t, uint32_t>(g_img, kW, kH, kW, g_kpL, kKp, g_pattern,
                                                      g_dl);
        briefL[r] = boardCycles() - t0;
        shiftToRight(g_img);
        pack(g_img, right);
        t0 = boardCycles();
        bincv::computeBrief<kBits, uint8_t, uint32_t>(g_img, kW, kH, kW, g_kpR, kKp, g_pattern,
                                                      g_dr);
        briefR[r] = boardCycles() - t0;
    }

    bincv::StereoMatchParams params;
    params.maxDisparity = 64;
    // Refinement alone, from the true disparity, with the staged arm off and on; and a
    // radius no strip can hold, where the arm declines every window and the two must
    // time the same -- the check that the arm is running where it is said to be.
    bincv::StereoMatchParams wide = params;
    wide.refineRadius = 12;
    const auto refine = [&](const bincv::StereoMatchParams& p, bool staged) {
        for (size_t i = 0; i < kKp; ++i) {
            g_m[i].valid = 1;
            g_m[i].disparity = static_cast<float>(kDisp);
        }
        bincv::impl::stereoWindowCountEnabled() = staged;
        const uint32_t t0 = boardCycles();
        bincv::stereoRefineDisparity<uint32_t>(left, right, g_kpL, kKp, g_m, p);
        const uint32_t t = boardCycles() - t0;
        bincv::impl::stereoWindowCountEnabled() = true;
        return t;
    };
    // The descriptor stage with the published float row test and with the integer one.
    const auto coarse = [&](bool integerGate) {
        bincv::impl::stereoIntegerRowGateEnabled() = integerGate;
        const uint32_t t0 = boardCycles();
        bincv::stereoDescriptorMatch<uint32_t>(g_kpL, kKp, g_dl, g_kpR, kKp, g_dr, kWords, g_m,
                                               params);
        const uint32_t t = boardCycles() - t0;
        bincv::impl::stereoIntegerRowGateEnabled() = true;
        return t;
    };
    uint32_t coarse0[kRepeats], coarse1[kRepeats], fine0[kRepeats], fine1[kRepeats],
        wide0[kRepeats], wide1[kRepeats], both[kRepeats];
    uint32_t hCoarse0 = 0, hCoarse1 = 0, hFine0 = 0, hFine1 = 0, hWide0 = 0, hWide1 = 0;
    for (size_t r = 0; r < kRepeats; ++r) {
        coarse0[r] = coarse(false);
        hCoarse0 = digestMatches(g_m, kKp);
        coarse1[r] = coarse(true);
        hCoarse1 = digestMatches(g_m, kKp);
        uint32_t t0;
        fine0[r] = refine(params, false);
        hFine0 = digestMatches(g_m, kKp);
        fine1[r] = refine(params, true);
        hFine1 = digestMatches(g_m, kKp);
        wide0[r] = refine(wide, false);
        hWide0 = digestMatches(g_m, kKp);
        wide1[r] = refine(wide, true);
        hWide1 = digestMatches(g_m, kKp);
        t0 = boardCycles();
        bincv::stereoMatchRectified<uint32_t>(left, right, g_kpL, kKp, g_dl, g_kpR, kKp, g_dr,
                                              kWords, g_m, params);
        both[r] = boardCycles() - t0;
    }

    size_t valid = 0;
    for (size_t i = 0; i < kKp; ++i) valid += g_m[i].valid;
    boardPuts("-- digests, for the host's --\n");
    putDigest("left bits      ", digestBits(left));
    putDigest("right bits     ", digestBits(right));
    putDigest("left brief     ", fnv1a(kFnvSeed, g_dl, sizeof g_dl));
    putDigest("right brief    ", fnv1a(kFnvSeed, g_dr, sizeof g_dr));
    putDigest("matches        ", digestMatches(g_m, kKp));
    putDigest("coarse, float rows", hCoarse0);
    putDigest("coarse, integer rows", hCoarse1);
    putDigest("refined, per row", hFine0);
    putDigest("refined, staged ", hFine1);
    putDigest("radius 12, per row", hWide0);
    putDigest("radius 12, staged ", hWide1);
    boardPuts("  matched  ");
    boardPutU32(static_cast<uint32_t>(valid));
    boardPuts(" of ");
    boardPutU32(static_cast<uint32_t>(kKp));
    boardPuts("\n-- sparse stereo, 752x480, 500 keypoints, cycles, median of 5 --\n");
    putCycles("brief, left frame", m7::medianOf(briefL, kRepeats));
    putCycles("brief, right frame", m7::medianOf(briefR, kRepeats));
    putCycles("descriptor stage, float rows", m7::medianOf(coarse0, kRepeats));
    putCycles("descriptor stage, integer rows", m7::medianOf(coarse1, kRepeats));
    putCycles("refinement stage, per row", m7::medianOf(fine0, kRepeats));
    putCycles("refinement stage, staged", m7::medianOf(fine1, kRepeats));
    putCycles("radius 12, per row", m7::medianOf(wide0, kRepeats));
    putCycles("radius 12, arm declines", m7::medianOf(wide1, kRepeats));
    putCycles("both stages", m7::medianOf(both, kRepeats));
    boardPuts("OK\n");
}

const m7::Command kCommands[] = {{"STEREO", &cmdStereo}};

}  // namespace

int main() { m7::serve("stereo", kCommands, sizeof(kCommands) / sizeof(kCommands[0])); }
