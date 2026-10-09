/// @file bincv_m7_sensor.cpp
/// @brief The sensor stage on the M7, by part: the median's row body, the edge
/// threshold's row body, and the stream that drives both, each over 480 rows of 752.
///
///   SENSOR          every row of each part with its DSP arm on and off, compared;
///                   then cycles for each part and arm, median of 5 interleaved repeats
///
/// The tracker image times the stage as one number; this says which half of it to
/// work on. The rows are a fixed pseudo-random image of low contrast, so the edge
/// test lands on both sides of its threshold everywhere.

#include <cstddef>
#include <cstdint>

#include "board.h"
#include "m7_harness.hpp"

#include "bincv/ops/edge.hpp"
#include "bincv/ops/medianEdgeStream.hpp"
#include "bincv/ops/medianWide.hpp"

namespace {

constexpr size_t kW = 752, kH = 480;
constexpr size_t kRows = 8;  // a small set of source rows, cycled
constexpr size_t kRepeats = 5;

__attribute__((section(".frames"), aligned(32))) uint8_t g_rows[kRows * kW];
__attribute__((section(".frames"), aligned(32))) uint8_t g_out[kW];
__attribute__((section(".frames"), aligned(32))) uint32_t g_bits[kH * ((kW + 31) / 32)];
__attribute__((section(".frames"), aligned(32))) uint8_t g_scratch[5 * kW];
__attribute__((section(".frames"), aligned(32))) uint8_t g_outB[kW];
__attribute__((section(".frames"), aligned(32))) uint32_t g_bitsB[(kW + 31) / 32];

void fill() {
    uint32_t s = 0x9E3779B9u;
    for (size_t i = 0; i < kRows * kW; ++i) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        g_rows[i] = static_cast<uint8_t>(100u + (s >> 27));  // 100..131: gradients near 17
    }
}

const uint8_t* row(size_t i) { return g_rows + (i % kRows) * kW; }

uint32_t timeMedian() {
    const uint32_t t0 = boardCycles();
    for (size_t y = 0; y < kH; ++y) {
        const uint8_t* rows[3] = {y == 0 ? nullptr : row(y - 1), row(y), row(y)};
        bincv::impl::medianRow<3, uint8_t>(rows, kW, g_out, bincv::kMedianReferenceL);
    }
    return boardCycles() - t0;
}

uint32_t timeEdge() {
    constexpr size_t stride = (kW + 31) / 32;
    const uint32_t t0 = boardCycles();
    for (size_t y = 0; y < kH; ++y) {
        bincv::impl::edgeRow<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                             bincv::EdgeSpatial::Wide, uint8_t, uint32_t>(
            row(y + kRows - 1), row(y), row(y + 1), kW, g_bits + y * stride, uint8_t{17});
    }
    return boardCycles() - t0;
}

uint32_t timeStream() {
    constexpr size_t stride = (kW + 31) / 32;
    using Stream = bincv::MedianEdgeStream<3, uint8_t>;
    Stream st(g_scratch, sizeof g_scratch, bincv::kMedianReferenceL,
              bincv::BinMatView<uint32_t>(g_bits, kW, kH, stride), uint8_t{17});
    st.reset();
    const uint32_t t0 = boardCycles();
    for (size_t y = 0; y < kH; ++y) st.push(row(y), 1, kW);
    return boardCycles() - t0;
}

void setArms(bool on) {
#if defined(BINCV_EDGE_DSP)
    bincv::impl::edgeSimdEnabled() = on;
#endif
#if defined(BINCV_MEDIAN_DSP)
    bincv::impl::medianSimdEnabled() = on;
#endif
    (void)on;
}

/// Each row's median and edge output with the arms on against the same with them
/// off. A faster arm that computes something else is not a result.
size_t armsDisagree() {
    constexpr size_t stride = (kW + 31) / 32;
    size_t bad = 0;
    for (size_t y = 0; y < kH; ++y) {
        const uint8_t* rows[3] = {y == 0 ? nullptr : row(y - 1), row(y), row(y)};
        setArms(true);
        bincv::impl::medianRow<3, uint8_t>(rows, kW, g_out, bincv::kMedianReferenceL);
        bincv::impl::edgeRow<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                             bincv::EdgeSpatial::Wide, uint8_t, uint32_t>(
            row(y + kRows - 1), row(y), row(y + 1), kW, g_bits, uint8_t{17});
        setArms(false);
        bincv::impl::medianRow<3, uint8_t>(rows, kW, g_outB, bincv::kMedianReferenceL);
        bincv::impl::edgeRow<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                             bincv::EdgeSpatial::Wide, uint8_t, uint32_t>(
            row(y + kRows - 1), row(y), row(y + 1), kW, g_bitsB, uint8_t{17});
        for (size_t x = 0; x < kW; ++x) bad += g_out[x] != g_outB[x];
        for (size_t w = 0; w < stride; ++w) bad += g_bits[w] != g_bitsB[w];
    }
    setArms(true);
    return bad;
}

void cmdSensor(const char*) {
    fill();
    const size_t bad = armsDisagree();
    boardPuts("-- arms on against arms off, every row --\n  ");
    boardPutU32(static_cast<uint32_t>(bad));
    boardPuts(bad == 0 ? " values differ\n" : " values DIFFER\n");
    if (bad != 0) {
        boardPuts("ERR the DSP arms do not compute what the portable ones do\n");
        return;
    }
    uint32_t med[2][kRepeats], edge[2][kRepeats], stream[2][kRepeats];
    for (size_t r = 0; r < kRepeats; ++r) {
        for (int a = 1; a >= 0; --a) {
            setArms(a != 0);
            med[a][r] = timeMedian();
            edge[a][r] = timeEdge();
            stream[a][r] = timeStream();
        }
    }
    setArms(true);
    boardPuts("-- sensor stage parts, 752x480, cycles, median of 5 --\n");
    const char* names[3] = {"median rows", "edge rows", "stream, whole stage"};
    uint32_t (*sets[3])[kRepeats] = {med, edge, stream};
    for (size_t p = 0; p < 3; ++p) {
        const uint32_t on = m7::medianOf(sets[p][1], kRepeats);
        const uint32_t off = m7::medianOf(sets[p][0], kRepeats);
        boardPuts("  ");
        boardPuts(names[p]);
        boardPuts(", DSP arm  ");
        boardPutU32(on);
        boardPuts(" cyc\n  ");
        boardPuts(names[p]);
        boardPuts(", portable arm  ");
        boardPutU32(off);
        boardPuts(" cyc\n    arm buys ");
        m7::putRatio(off, on);
        boardPuts("x\n");
    }
    boardPuts("OK\n");
}

const m7::Command kCommands[] = {{"SENSOR", &cmdSensor}};

}  // namespace

int main() { m7::serve("sensor", kCommands, sizeof(kCommands) / sizeof(kCommands[0])); }
