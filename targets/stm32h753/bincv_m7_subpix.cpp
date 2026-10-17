/// @file bincv_m7_subpix.cpp
/// @brief `cornerSubPix` on the M7, refining the corners the reference detector finds in
/// the scene of benchmark/subpix_scene.hpp.
///
///   SUBPIX          build the scene and its seeds; refine them; a digest of the refined
///                   points' bits, for the host's (m7_reference subpix) to be held to;
///                   then cycles, median of 5 (each repeat refines the same seeds)
///   SUBPIXDUMP      the seeds and the refined points in full, for a difference from the
///                   host's to be measured rather than only detected (m7_reference subpix
///                   --dump prints the same lines)
///
/// Memory: the 8-bit frame and its bits in the AXI SRAM; both derivatives, the detector's
/// pool and ring and the points in the D2 SRAM. Nothing is allocated: the derivative
/// containers wrap those buffers.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "board.h"
#include "m7_harness.hpp"
#include "subpix_scene.hpp"

namespace {

using namespace subpixscene;

constexpr size_t kRepeats = 5;

__attribute__((section(".frames"), aligned(32))) uint8_t g_img[kW * kH];
__attribute__((section(".sram2"), aligned(32))) uint8_t g_ring[3 * kW];
__attribute__((section(".frames"), aligned(32))) uint32_t g_bits[kPlaneWords];
__attribute__((section(".sram2"), aligned(32))) uint32_t g_dx[2 * kPlaneWords];
__attribute__((section(".sram2"), aligned(32))) uint32_t g_dy[2 * kPlaneWords];
__attribute__((section(".sram2"), aligned(32))) float g_response[3 * kW];
__attribute__((section(".sram2"), aligned(32))) bincv::CornerCandidate g_pool[kPool];
__attribute__((section(".sram2"), aligned(32))) bincv::Corner g_corners[kMaxCorners];
bincv::Point2f g_seeds[kMaxCorners];
bincv::Point2f g_pts[kMaxCorners];

void cmdSubpix(const char*) {
    const size_t n = build(g_img, g_ring, g_bits, g_dx, g_dy, g_response, g_pool, g_corners,
                           g_seeds);
    const bincv::SignedQuantMat<1, uint32_t> dx = view(g_dx), dy = view(g_dy);
    const bincv::SubPixParams p;
    bincv::SubPixResult r;
    uint32_t cycles[kRepeats];
    for (size_t k = 0; k < kRepeats; ++k) {
        std::memcpy(g_pts, g_seeds, n * sizeof(bincv::Point2f));
        const uint32_t t0 = boardCycles();
        r = bincv::cornerSubPix<1, uint32_t>(dx, dy, g_pts, n, p);
        cycles[k] = boardCycles() - t0;
    }
    boardPuts("-- cornerSubPix, 752x480, winHalf 5, 40 iterations, eps 0.001 --\n  ");
    boardPutU32(static_cast<uint32_t>(n));
    boardPuts(" seeds, ");
    boardPutU32(static_cast<uint32_t>(r.refined));
    boardPuts(" refined, points ");
    boardPutHex32(stereoscene::fnv1a(stereoscene::kFnvSeed, g_pts, n * sizeof(bincv::Point2f)));
    boardPuts("\n  cycles  ");
    boardPutU32(m7::medianOf(cycles, kRepeats));
    boardPuts(" cyc\nOK\n");
}

void putPoints(const char* tag, const bincv::Point2f* p, size_t n) {
    boardPuts(tag);
    for (size_t i = 0; i < n; ++i) {
        uint32_t x, y;
        std::memcpy(&x, &p[i].x, sizeof x);
        std::memcpy(&y, &p[i].y, sizeof y);
        boardPuts(" ");
        boardPutHex32(x);
        boardPutHex32(y);
    }
    boardPuts("\n");
}

void cmdSubpixDump(const char*) {
    const size_t n = build(g_img, g_ring, g_bits, g_dx, g_dy, g_response, g_pool, g_corners,
                           g_seeds);
    std::memcpy(g_pts, g_seeds, n * sizeof(bincv::Point2f));
    bincv::cornerSubPix<1, uint32_t>(view(g_dx), view(g_dy), g_pts, n, bincv::SubPixParams());
    putPoints("S", g_seeds, n);
    putPoints("P", g_pts, n);
    boardPuts("OK\n");
}

const m7::Command kCommands[] = {{"SUBPIX", &cmdSubpix}, {"SUBPIXDUMP", &cmdSubpixDump}};

}  // namespace

// Where abort() ends. Wrapping the derivative buffers reaches a binCV validation path,
// which prints through _write and aborts -- the only image that links one -- and a
// board has nothing to return to, so it stops with that message the last thing on the
// UART.
extern "C" void _exit(int) {
    for (;;) {
    }
}

int main() { m7::serve("subpix", kCommands, sizeof(kCommands) / sizeof(kCommands[0])); }
