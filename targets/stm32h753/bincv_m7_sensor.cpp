/// @file bincv_m7_sensor.cpp
/// @brief The sensor stage on the M7, by part: the median's row body, the edge
/// threshold's row body, and the stream that drives both, each over 480 rows of 752.
///
///   SENSOR          every row of each part with its DSP arms on and off, compared,
///                   and the aligned edge arm against the portable body over every
///                   predicate at thresholds on both sides of each lane boundary;
///                   then cycles for each part and arm, median of 5 interleaved repeats
///
/// The tracker image times the stage as one number; this says which half of it to
/// work on. The rows are a fixed pseudo-random image of low contrast, so the edge
/// test lands on both sides of its threshold everywhere. The stream is timed twice:
/// one row a push, from rows that stay in the cache, and sixteen rows a push from a
/// whole frame in the AXI SRAM, each chunk read once before it is pushed -- the shape
/// of a caller whose rows arrive in blocks and are checked on arrival.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "board.h"
#include "m7_harness.hpp"

#include "bincv/ops/edge.hpp"
#include "bincv/ops/medianEdgeStream.hpp"
#include "bincv/ops/medianWide.hpp"

namespace {

using bincv::EdgeCombine;
using bincv::EdgeRelation;
using bincv::EdgeSpatial;

constexpr size_t kW = 752, kH = 480;
constexpr size_t kStride = (kW + 31) / 32;
constexpr size_t kRows = 8;  // a small set of source rows, cycled
constexpr size_t kChunk = 16;
constexpr size_t kRepeats = 5;

__attribute__((section(".frames"), aligned(32))) uint8_t g_rows[kRows * kW + 4];
__attribute__((section(".frames"), aligned(32))) uint8_t g_out[kW];
__attribute__((section(".frames"), aligned(32))) uint32_t g_bits[kH * kStride];
__attribute__((section(".frames"), aligned(32))) uint8_t g_scratch[4 * kW];
__attribute__((section(".frames"), aligned(32))) uint8_t g_outB[kW];
__attribute__((section(".frames"), aligned(32))) uint32_t g_bitsB[kH * kStride];
__attribute__((section(".frames"), aligned(32))) uint8_t g_frame[kW * kH];
__attribute__((section(".frames"), aligned(32))) uint8_t g_chunk[kChunk * kW];

static_assert(sizeof g_scratch >=
                  bincv::MedianEdgeStream<3, uint8_t>::scratchElements(kW,
                                                                       bincv::kMedianReferenceL),
              "the stream's scratch");

uint32_t xorshift(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

void fill() {
    uint32_t s = 0x9E3779B9u;
    for (size_t i = 0; i < sizeof g_rows; ++i) {
        g_rows[i] = static_cast<uint8_t>(100u + (xorshift(s) >> 27));  // 100..131: gradients near 17
    }
    s = 0x12345679u;
    for (size_t i = 0; i < kW * kH; ++i) g_frame[i] = static_cast<uint8_t>(100u + (xorshift(s) >> 27));
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

/// `shift` moves all three rows off their 4-byte alignment, which the aligned arm's
/// gate excludes: with it set, that arm's switch must make no difference.
uint32_t timeEdge(size_t shift) {
    const uint32_t t0 = boardCycles();
    for (size_t y = 0; y < kH; ++y) {
        bincv::impl::edgeRow<EdgeCombine::Or, EdgeRelation::Ge, EdgeSpatial::Wide, uint8_t,
                             uint32_t>(row(y + kRows - 1) + shift, row(y) + shift,
                                       row(y + 1) + shift, kW, g_bits + y * kStride,
                                       uint8_t{17});
    }
    return boardCycles() - t0;
}

using Stream = bincv::MedianEdgeStream<3, uint8_t>;

uint32_t timeStream(uint32_t* bits) {
    Stream st(g_scratch, sizeof g_scratch, bincv::kMedianReferenceL,
              bincv::BinMatView<uint32_t>(bits, kW, kH, kStride), uint8_t{17});
    st.reset();
    const uint32_t t0 = boardCycles();
    for (size_t y = 0; y < kH; ++y) st.push(row(y), 1, kW);
    return boardCycles() - t0;
}

/// Sixteen rows a push from the frame. Each chunk is copied into the receive buffer
/// and read once, untimed, so it is in the cache as a checked arrival leaves it.
uint32_t timeChunked(uint32_t* bits) {
    Stream st(g_scratch, sizeof g_scratch, bincv::kMedianReferenceL,
              bincv::BinMatView<uint32_t>(bits, kW, kH, kStride), uint8_t{17});
    st.reset();
    uint32_t total = 0;
    for (size_t y = 0; y < kH; y += kChunk) {
        std::memcpy(g_chunk, g_frame + y * kW, kChunk * kW);
        uint32_t sum = 0;
        for (size_t i = 0; i < kChunk * kW; i += 4) sum += g_chunk[i];
        __asm__ volatile("" ::"r"(sum));
        const uint32_t t0 = boardCycles();
        st.push(g_chunk, kChunk, kW);
        total += boardCycles() - t0;
    }
    return total;
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

void setAligned(bool on) {
#if defined(BINCV_EDGE_DSP)
    bincv::impl::edgeAlignedRowEnabled() = on;
#endif
    (void)on;
}

size_t wordsDiffer(const uint32_t* a, const uint32_t* b, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) bad += a[i] != b[i];
    return bad;
}

/// Each row's median and edge output with the arms on against the same with them
/// off. A faster arm that computes something else is not a result.
size_t armsDisagree() {
    size_t bad = 0;
    for (size_t y = 0; y < kH; ++y) {
        const uint8_t* rows[3] = {y == 0 ? nullptr : row(y - 1), row(y), row(y)};
        setArms(true);
        bincv::impl::medianRow<3, uint8_t>(rows, kW, g_out, bincv::kMedianReferenceL);
        bincv::impl::edgeRow<EdgeCombine::Or, EdgeRelation::Ge, EdgeSpatial::Wide, uint8_t,
                             uint32_t>(row(y + kRows - 1), row(y), row(y + 1), kW, g_bits,
                                       uint8_t{17});
        setArms(false);
        bincv::impl::medianRow<3, uint8_t>(rows, kW, g_outB, bincv::kMedianReferenceL);
        bincv::impl::edgeRow<EdgeCombine::Or, EdgeRelation::Ge, EdgeSpatial::Wide, uint8_t,
                             uint32_t>(row(y + kRows - 1), row(y), row(y + 1), kW, g_bitsB,
                                       uint8_t{17});
        for (size_t x = 0; x < kW; ++x) bad += g_out[x] != g_outB[x];
        bad += wordsDiffer(g_bits, g_bitsB, kStride);
    }
    setArms(true);
    return bad;
}

/// One predicate's rows through the aligned arm and through the portable body.
template <EdgeCombine C, EdgeRelation R>
size_t predicateDisagrees(uint8_t t) {
    size_t bad = 0;
    for (size_t y = 0; y < kRows; ++y) {
        const uint8_t* u = row(y + kRows - 1);
        const uint8_t* m = row(y);
        const uint8_t* d = row(y + 1);
        setArms(true);
        bincv::impl::edgeRow<C, R, EdgeSpatial::Wide, uint8_t, uint32_t>(u, m, d, kW, g_bits, t);
        setArms(false);
        bincv::impl::edgeRow<C, R, EdgeSpatial::Wide, uint8_t, uint32_t>(u, m, d, kW, g_bitsB, t);
        bad += wordsDiffer(g_bits, g_bitsB, kStride);
    }
    setArms(true);
    return bad;
}

/// The aligned arm against the portable body over all four predicates of the wide
/// difference, on full-range rows and on rows whose differences sit at each
/// threshold, at thresholds either side of 0, 128 and 255.
size_t alignedDisagrees() {
    static const uint8_t kT[] = {0, 1, 2, 16, 17, 18, 64, 127, 128, 129, 200, 254, 255};
    size_t bad = 0;
    uint32_t s = 0xC0FFEE11u;
    for (int pass = 0; pass < 2; ++pass) {
        for (const uint8_t t : kT) {
            // pass 0: every byte value; pass 1: a base level plus offsets in [0, t + 1],
            // so differences land on t - 1, t and t + 1 everywhere.
            for (size_t i = 0; i < kRows * kW; ++i) {
                const uint32_t r = xorshift(s);
                if (pass == 0) {
                    g_rows[i] = static_cast<uint8_t>(r >> 24);
                } else {
                    const uint32_t span = t >= 254 ? 256u : static_cast<uint32_t>(t) + 2u;
                    const uint32_t base = (r >> 24) % (257u - span);  // base + span - 1 <= 255
                    g_rows[i] = static_cast<uint8_t>(base + ((r >> 8) & 0xFFFFu) % span);
                }
            }
            bad += predicateDisagrees<EdgeCombine::Or, EdgeRelation::Ge>(t);
            bad += predicateDisagrees<EdgeCombine::And, EdgeRelation::Ge>(t);
            bad += predicateDisagrees<EdgeCombine::Or, EdgeRelation::Gt>(t);
            bad += predicateDisagrees<EdgeCombine::And, EdgeRelation::Gt>(t);
        }
    }
    fill();
    return bad;
}

void putPart(const char* name, const char* arm, uint32_t cyc) {
    boardPuts("  ");
    boardPuts(name);
    boardPuts(", ");
    boardPuts(arm);
    boardPuts("  ");
    boardPutU32(cyc);
    boardPuts(" cyc\n");
}

void putRatio(const char* what, uint32_t slow, uint32_t fast) {
    boardPuts("    ");
    boardPuts(what);
    boardPuts(" ");
    m7::putRatio(slow, fast);
    boardPuts("x\n");
}

void cmdSensor(const char*) {
    fill();
    size_t bad = armsDisagree();
    boardPuts("-- arms on against arms off, every row --\n  ");
    boardPutU32(static_cast<uint32_t>(bad));
    boardPuts(bad == 0 ? " values differ\n" : " values DIFFER\n");
    const size_t badPred = alignedDisagrees();
    boardPuts("-- aligned edge arm against the portable body, 4 predicates x 13 thresholds --\n  ");
    boardPutU32(static_cast<uint32_t>(badPred));
    boardPuts(badPred == 0 ? " words differ\n" : " words DIFFER\n");
    setArms(false);
    timeChunked(g_bitsB);
    setArms(true);
    timeChunked(g_bits);
    const size_t badChunk = wordsDiffer(g_bits, g_bitsB, kH * kStride);
    boardPuts("-- sixteen rows a push, arms on against off, the whole frame --\n  ");
    boardPutU32(static_cast<uint32_t>(badChunk));
    boardPuts(badChunk == 0 ? " words differ\n" : " words DIFFER\n");
    if (bad != 0 || badPred != 0 || badChunk != 0) {
        boardPuts("ERR the DSP arms do not compute what the portable ones do\n");
        return;
    }

    enum { kMedOn, kMedOff, kEdgeAl, kEdgeUn, kEdgeOff, kMisOn, kMisOff, kStrOn, kStrOff,
           kChOn, kChOff, kParts };
    uint32_t c[kParts][kRepeats];
    for (size_t r = 0; r < kRepeats; ++r) {
        setArms(true);
        c[kMedOn][r] = timeMedian();
        c[kEdgeAl][r] = timeEdge(0);
        c[kMisOn][r] = timeEdge(1);
        c[kStrOn][r] = timeStream(g_bits);
        c[kChOn][r] = timeChunked(g_bits);
        setAligned(false);
        c[kEdgeUn][r] = timeEdge(0);
        c[kMisOff][r] = timeEdge(1);
        setAligned(true);
        setArms(false);
        c[kMedOff][r] = timeMedian();
        c[kEdgeOff][r] = timeEdge(0);
        c[kStrOff][r] = timeStream(g_bits);
        c[kChOff][r] = timeChunked(g_bits);
    }
    setArms(true);
    uint32_t v[kParts];
    for (size_t p = 0; p < kParts; ++p) v[p] = m7::medianOf(c[p], kRepeats);

    boardPuts("-- sensor stage parts, 752x480, cycles, median of 5 --\n");
    putPart("median rows", "DSP arm", v[kMedOn]);
    putPart("median rows", "portable arm", v[kMedOff]);
    putRatio("DSP arm buys", v[kMedOff], v[kMedOn]);
    putPart("edge rows", "aligned DSP arm", v[kEdgeAl]);
    putPart("edge rows", "unaligned DSP arm", v[kEdgeUn]);
    putPart("edge rows", "portable arm", v[kEdgeOff]);
    putRatio("aligned arm buys over the unaligned", v[kEdgeUn], v[kEdgeAl]);
    putRatio("DSP arm buys", v[kEdgeOff], v[kEdgeAl]);
    putPart("edge rows, misaligned", "aligned arm switched on", v[kMisOn]);
    putPart("edge rows, misaligned", "aligned arm switched off", v[kMisOff]);
    putRatio("switch makes (its gate excludes these rows)", v[kMisOff], v[kMisOn]);
    putPart("stream, one row a push", "DSP arms", v[kStrOn]);
    putPart("stream, one row a push", "portable arms", v[kStrOff]);
    putRatio("arms buy", v[kStrOff], v[kStrOn]);
    putPart("stream, 16 rows a push from the frame", "DSP arms", v[kChOn]);
    putPart("stream, 16 rows a push from the frame", "portable arms", v[kChOff]);
    putRatio("arms buy", v[kChOff], v[kChOn]);
    boardPuts("OK\n");
}

const m7::Command kCommands[] = {{"SENSOR", &cmdSensor}};

}  // namespace

int main() { m7::serve("sensor", kCommands, sizeof(kCommands) / sizeof(kCommands[0])); }
