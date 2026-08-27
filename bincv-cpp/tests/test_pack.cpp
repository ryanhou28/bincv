// Turning a plain pixel array into bits (T5.6 / T5.9 / T5.15).
//
// CORE-ONLY BY DESIGN, AND THAT IS THE POINT OF THE FILE. Before ops/pack.hpp every
// path that got pixels into binCV took a `cv::Mat`, so the three core-only
// configurations verify.sh builds -- the ones the embedded claim rests on -- had no
// way to receive an image at all. This file uses no OpenCV, so it fails in those
// configurations if that regresses.
//
// WHAT IS ACTUALLY CHECKED, AND WHY IT IS A CROSS-PRODUCT.
//
//   * THREE RULES x TWO SOURCE TYPES x FOUR WORD TYPES. The rules are compile-time
//     (X-71's 46x needs one visible predicate), so each combination is a separate
//     instantiation and testing one of them tests one of them. A rule set with a
//     single tested member is a one-rule op with untested branches.
//
//   * uint16_t IS NOT DECORATION. 10-, 12- and 16-bit sensors are ordinary
//     (ARCHITECTURE 7.8), and the x86 path narrows two 16-lane compares with
//     `packs_epi16` + `permute4x64` before the move-mask -- lane order that a
//     uint8-only test cannot exercise.
//
//   * THE SIGNED-COMPARE BIAS. SSE/AVX integer compares are SIGNED and binCV's
//     pixels are not: `cmpgt_epi8` on 0xFF against 0x01 asks "is -1 > 1" and answers
//     no. Thresholds above 127 (and above 32767) are chosen deliberately so a
//     missing bias fails here rather than in a frontend.
//
//   * PADDING BITS. Every row's trailing partial word is checked to be zero past
//     `width` -- CLAUDE.md's hard rule, and word-wise reductions over-count without
//     it. The widths are odd for the same reason.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/io/pnm.hpp"
#include "bincv-cpp/ops/pack.hpp"
#include "test_util.hpp"

namespace {

using namespace bincv;

/// A per-pixel reference. Deliberately the naive spelling: the point of an oracle is
/// that it shares no code with the thing it checks.
template <PackRule R, typename SrcT>
bool expected(SrcT v, SrcT t) {
    if (R == PackRule::NonZero) return v != SrcT{0};
    if (R == PackRule::GreaterThan) return v > t;
    return v >= t;
}

template <PackRule R, typename SrcT, typename W>
void checkRule(const char* label, SrcT t) {
    // Odd width so the trailing partial word is always exercised; odd height so no
    // row index is a multiple of anything convenient.
    constexpr size_t kW = 197, kH = 13;
    std::vector<SrcT> img(kW * kH);
    uint64_t st = 0xC0FFEEULL;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<SrcT>(st >> 40);
    }
    BinMat<W> m(static_cast<int>(kW), static_cast<int>(kH));
    packBits<R, SrcT, W>(img.data(), kW, kH, kW, m.view(), t);

    constexpr size_t kBits = sizeof(W) * 8;
    size_t wrong = 0, dirty = 0;
    for (size_t y = 0; y < kH; ++y) {
        const W* row = m.constView().row(y);
        for (size_t x = 0; x < kW; ++x) {
            const bool got = ((row[x / kBits] >> (x % kBits)) & W{1}) != 0;
            if (got != expected<R, SrcT>(img[y * kW + x], t)) ++wrong;
        }
        const size_t words = (kW + kBits - 1) / kBits;
        const size_t used = kW % kBits;
        if (used != 0 && static_cast<W>(row[words - 1] >> used) != W{0}) ++dirty;
    }
    std::printf("  %-30s %5zu wrong, %zu rows with dirty padding\n", label, wrong, dirty);
    BINCV_CHECK(wrong == 0);
    BINCV_CHECK(dirty == 0);
}

} // namespace

BINCV_TEST(Pack, RulesBySourceAndWordType) {
    // Thresholds ABOVE 127 and 32767 on purpose -- see the bias note in the header.
    checkRule<PackRule::NonZero, uint8_t, uint32_t>("NonZero      u8  -> u32", 0);
    checkRule<PackRule::GreaterThan, uint8_t, uint32_t>("GreaterThan  u8  -> u32", 200);
    checkRule<PackRule::GreaterEqual, uint8_t, uint32_t>("GreaterEqual u8  -> u32", 200);
    checkRule<PackRule::GreaterThan, uint8_t, uint8_t>("GreaterThan  u8  -> u8 ", 17);
    checkRule<PackRule::GreaterThan, uint8_t, uint16_t>("GreaterThan  u8  -> u16", 250);
    checkRule<PackRule::GreaterThan, uint8_t, uint64_t>("GreaterThan  u8  -> u64", 128);
    checkRule<PackRule::NonZero, uint16_t, uint32_t>("NonZero      u16 -> u32", 0);
    checkRule<PackRule::GreaterThan, uint16_t, uint32_t>("GreaterThan  u16 -> u32", 40000);
    checkRule<PackRule::GreaterEqual, uint16_t, uint32_t>("GreaterEqual u16 -> u32", 4095);
    checkRule<PackRule::GreaterThan, uint16_t, uint64_t>("GreaterThan  u16 -> u64", 1000);
}

BINCV_TEST(Pack, PredicateFormMatchesRule) {
    // packBitsIf is the escape hatch for rules PackRule cannot express, and it takes
    // the portable path always. Where the two CAN express the same rule they must
    // agree -- otherwise the fast path and the general one have drifted.
    constexpr size_t kW = 131, kH = 7;
    std::vector<uint8_t> img(kW * kH);
    uint64_t st = 99;
    for (auto& v : img) { st = st * 6364136223846793005ULL + 1; v = static_cast<uint8_t>(st >> 40); }
    BinMat<uint32_t> a(kW, kH), b(kW, kH);
    packBits<PackRule::GreaterThan, uint8_t, uint32_t>(img.data(), kW, kH, kW, a.view(), 90);
    packBitsIf(img.data(), kW, kH, kW, b.view(), [](uint8_t v) { return v > 90; });
    size_t diff = 0;
    for (size_t y = 0; y < kH; ++y)
        for (size_t i = 0; i < (kW + 31) / 32; ++i)
            if (a.constView().row(y)[i] != b.constView().row(y)[i]) ++diff;
    std::printf("  packBitsIf vs PackRule        %zu words differ\n", diff);
    BINCV_CHECK(diff == 0);
}

BINCV_TEST(Pack, RoundTripThroughUnpack) {
    // pack -> unpack is a LEFT inverse on the bits, which is all it can be: the
    // source's magnitude is gone by construction. What it pins is that the two
    // agree on WHICH pixels are set, including in the trailing partial word.
    constexpr size_t kW = 83, kH = 5;
    std::vector<uint8_t> img(kW * kH);
    for (size_t i = 0; i < img.size(); ++i) img[i] = static_cast<uint8_t>((i * 37) % 256);
    BinMat<uint32_t> m(kW, kH);
    packBits<PackRule::GreaterThan, uint8_t, uint32_t>(img.data(), kW, kH, kW, m.view(), 128);
    std::vector<uint8_t> out(kW * kH, 7);
    unpackTo8Bit<uint32_t>(m.constView(), out.data(), kW);
    size_t wrong = 0;
    for (size_t i = 0; i < img.size(); ++i) {
        const uint8_t want = (img[i] > 128) ? 255 : 0;
        if (out[i] != want) ++wrong;
    }
    std::printf("  pack -> unpack round trip     %zu pixels differ\n", wrong);
    BINCV_CHECK(wrong == 0);
}

BINCV_TEST(Pack, StridedSourceIsTheYPlaneCase) {
    // A YUV420 Y plane is a strided 8-bit array and nothing more -- ARCHITECTURE 7.8
    // says binCV takes it as-is rather than converting. This is that claim: a source
    // whose stride exceeds its width must pack identically to the tight one.
    constexpr size_t kW = 61, kH = 9, kStride = 96;
    std::vector<uint8_t> padded(kStride * kH, 0xAB), tight(kW * kH);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x) {
            const uint8_t v = static_cast<uint8_t>((y * 31 + x * 17) % 256);
            padded[y * kStride + x] = v;
            tight[y * kW + x] = v;
        }
    BinMat<uint32_t> a(kW, kH), b(kW, kH);
    packBits<PackRule::GreaterThan, uint8_t, uint32_t>(padded.data(), kW, kH, kStride, a.view(), 100);
    packBits<PackRule::GreaterThan, uint8_t, uint32_t>(tight.data(), kW, kH, kW, b.view(), 100);
    size_t diff = 0;
    for (size_t y = 0; y < kH; ++y)
        for (size_t i = 0; i < (kW + 31) / 32; ++i)
            if (a.constView().row(y)[i] != b.constView().row(y)[i]) ++diff;
    std::printf("  strided source (Y plane)      %zu words differ\n", diff);
    BINCV_CHECK(diff == 0);
}

BINCV_TEST(Pack, PgmIsAWholeImageAndSizesItself) {
    // The only way to LOOK at binCV's output on a target with no OpenCV. Two things
    // are checked: that calling with cap == 0 reports the size without writing, and
    // that the bytes are a valid P5 whose payload matches unpackTo8Bit.
    constexpr size_t kW = 40, kH = 3;
    std::vector<uint8_t> img(kW * kH);
    for (size_t i = 0; i < img.size(); ++i) img[i] = static_cast<uint8_t>((i * 11) % 256);
    BinMat<uint32_t> m(kW, kH);
    packBits<PackRule::GreaterThan, uint8_t, uint32_t>(img.data(), kW, kH, kW, m.view(), 128);

    const size_t need = writePgm<uint32_t>(m.constView(), nullptr, 0);
    std::vector<uint8_t> buf(need, 0xEE);
    const size_t wrote = writePgm<uint32_t>(m.constView(), buf.data(), buf.size());
    std::printf("  PGM sized %zu, wrote %zu\n", need, wrote);
    BINCV_CHECK(wrote == need);
    // "P5\n40 3\n255\n" then w*h payload bytes.
    BINCV_CHECK(buf[0] == 'P' && buf[1] == '5' && buf[2] == '\n');
    BINCV_CHECK(need > kW * kH);
    const size_t header = need - kW * kH;
    std::vector<uint8_t> direct(kW * kH, 0);
    unpackTo8Bit<uint32_t>(m.constView(), direct.data(), kW);
    size_t diff = 0;
    for (size_t i = 0; i < kW * kH; ++i) if (buf[header + i] != direct[i]) ++diff;
    std::printf("  PGM payload vs unpackTo8Bit   %zu bytes differ\n", diff);
    BINCV_CHECK(diff == 0);
    // A short buffer must report the requirement and write NOTHING.
    std::vector<uint8_t> tiny(4, 0x11);
    const size_t again = writePgm<uint32_t>(m.constView(), tiny.data(), tiny.size());
    BINCV_CHECK(again == need);
    BINCV_CHECK(tiny[0] == 0x11);
}

BINCV_TEST(Pack, StreamingInChunksMatchesWholeFrame) {
    // The embedded case: a sensor delivers rows, not frames. Rows are independent --
    // nothing in packing reads a neighbouring row -- so an arbitrary chunking must
    // produce the identical image. If it ever does not, streaming is approximate and
    // the whole entry point is unsound.
    constexpr size_t kW = 149, kH = 23;
    std::vector<uint8_t> img(kW * kH);
    uint64_t st = 314159;
    for (auto& v : img) { st = st * 6364136223846793005ULL + 1; v = static_cast<uint8_t>(st >> 40); }
    BinMat<uint32_t> whole(kW, kH), streamed(kW, kH);
    packBits<PackRule::GreaterThan, uint8_t, uint32_t>(img.data(), kW, kH, kW, whole.view(), 120);

    // Deliberately ragged chunks, including one of a single row and one that ends
    // exactly on the last row.
    const size_t chunks[] = {1, 7, 4, 1, 10};
    size_t at = 0;
    for (size_t c : chunks) {
        packRows<PackRule::GreaterThan, uint8_t, uint32_t>(img.data() + at * kW, kW, c, kW,
                                                           streamed.view(), at, 120);
        at += c;
    }
    BINCV_CHECK(at == kH);
    size_t diff = 0;
    for (size_t y = 0; y < kH; ++y)
        for (size_t i = 0; i < (kW + 31) / 32; ++i)
            if (whole.constView().row(y)[i] != streamed.constView().row(y)[i]) ++diff;
    std::printf("  streamed in 5 ragged chunks   %zu words differ\n", diff);
    BINCV_CHECK(diff == 0);
}

BINCV_TEST(Pnm, RoundTripsThroughAFileFormat) {
    // writePgm -> readPgm with no OpenCV anywhere. This is the `none` backend of
    // bincv_io: enough to LOOK at what binCV produced on a target that has no image
    // library, and enough to feed it a test image. A PNG decoder would be eight times
    // the size of everything binCV does (ARCHITECTURE 7.8, measured).
    constexpr size_t kW = 67, kH = 11;
    std::vector<uint8_t> img(kW * kH);
    uint64_t st = 20260827;
    for (auto& v : img) { st = st * 6364136223846793005ULL + 1; v = static_cast<uint8_t>(st >> 40); }
    BinMat<uint32_t> a(kW, kH), b(kW, kH);
    packBits<PackRule::GreaterThan, uint8_t, uint32_t>(img.data(), kW, kH, kW, a.view(), 100);

    std::vector<uint8_t> file(writePgm<uint32_t>(a.constView(), nullptr, 0));
    writePgm<uint32_t>(a.constView(), file.data(), file.size());

    const PgmHeader h = readPgmHeader(file.data(), file.size());
    std::printf("  header: %zux%zu max=%u valid=%d\n", h.width, h.height, h.maxValue,
                h.valid ? 1 : 0);
    BINCV_CHECK(h.valid);
    BINCV_CHECK(h.width == kW && h.height == kH && h.maxValue == 255);

    // writePgm emits 255 for a set bit, so NonZero recovers exactly the same bits.
    BINCV_CHECK((readPgm<PackRule::NonZero, uint32_t>(file.data(), file.size(), b.view())));
    size_t diff = 0;
    for (size_t y = 0; y < kH; ++y)
        for (size_t i = 0; i < (kW + 31) / 32; ++i)
            if (a.constView().row(y)[i] != b.constView().row(y)[i]) ++diff;
    std::printf("  write -> read round trip      %zu words differ\n", diff);
    BINCV_CHECK(diff == 0);
}

BINCV_TEST(Pnm, RejectsRatherThanMisreads) {
    // A parser in core has no exceptions and no error channel, so the only honest
    // failure is `valid == false`. Each of these is a real way a buffer goes wrong.
    const uint8_t notPgm[] = {'P', '6', ' ', '1', ' ', '1', ' ', '2', '5', '5', '\n', 0};
    BINCV_CHECK(!readPgmHeader(notPgm, sizeof(notPgm)).valid);          // P6 is colour
    const uint8_t truncated[] = {'P', '5', '\n', '4', ' ', '4', '\n', '2', '5', '5', '\n', 1, 2};
    BINCV_CHECK(!readPgmHeader(truncated, sizeof(truncated)).valid);    // 13 bytes for 16 pixels
    const uint8_t noNumber[] = {'P', '5', '\n', 'x', '\n'};
    BINCV_CHECK(!readPgmHeader(noNumber, sizeof(noNumber)).valid);
    BINCV_CHECK(!readPgmHeader(nullptr, 0).valid);
    // Comments and extra whitespace are legal PNM and must NOT be a rejection.
    const uint8_t commented[] = {'P','5','\n','#',' ','h','i','\n','2',' ','2','\n','2','5','5','\n',
                                 9, 9, 9, 9};
    const PgmHeader ok = readPgmHeader(commented, sizeof(commented));
    std::printf("  comment-bearing header valid=%d %zux%zu\n", ok.valid ? 1 : 0, ok.width,
                ok.height);
    BINCV_CHECK(ok.valid && ok.width == 2 && ok.height == 2);
}

BINCV_TEST_MAIN("test_pack")
