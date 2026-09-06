// The frame-sequence blob ("BSQ1") -- io/sequence.hpp against its writer.
//
// CORE-ONLY BY DESIGN, like test_pack.cpp and for the same reason: the blob
// exists so a target with no filesystem and no OpenCV can run real dataset
// frames, so its reader has to be verified in the configurations that have
// neither. Nothing here needs an image library; the committed fixtures were
// written ONCE by scripts/make_sequence_blob.py and are compared against the
// sources at test time, so the tool-to-reader agreement is pinned forever
// without python in the loop.
//
// WHAT IS ACTUALLY CHECKED:
//
// * REJECTS RATHER THAN MISREADS. A parser in core has no exceptions and no
//   error channel, so the only honest failure is `valid == false` -- and each
//   rejection case here is a real way a blob goes wrong: bad magic, unknown
//   mode, zero dimensions, non-zero reserved bytes, a header cut short, a
//   frame index past the count, a blob truncated mid-frame.
//
// * THE BODY IS EXACTLY A PNM BODY. Round trips build the blob BY HAND out of
//   writePbm/writePgm output with the ASCII header stripped, so "a frame body
//   is a P4/P5 body" is asserted against the writer that owns those layouts
//   rather than against this file's opinion of them.
//
// * FOUR WORD TYPES, ODD WIDTH. 67 is not a multiple of 8, so the last file
//   byte is partial; nor of 8/16/32/64, so the last word is partial at every
//   word type -- two different paddings that must both end zero (CLAUDE.md's
//   hard rule; a set padding bit makes every word-wise reduction over-count).
//
// * THE FIXTURES CLOSE THE TOOL-TO-READER LOOP. The 8-bit blob's bodies must
//   equal the source PGMs' bodies byte for byte, and the packed blob's frames
//   must equal binCV's OWN sensor stage -- medianWide with the reference L,
//   then edgeThreshold -- run on those sources. That last comparison is the
//   strong one: the python tool's median/filter2D/threshold pipeline and the
//   C++ kernels are two independent implementations of the reference's
//   preprocessing, required here to agree bit for bit.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bincv/io/pnm.hpp"
#include "bincv/io/sequence.hpp"
#include "bincv/ops/edge.hpp"
#include "bincv/ops/medianWide.hpp"
#include "bincv/ops/pack.hpp"
#include "test_util.hpp"

namespace {

using namespace bincv;

// ---- helpers ---------------------------------------------------------------

/// A 32-byte BSQ1 header, fields little-endian, reserved bytes zero.
std::vector<uint8_t> blobHeader(uint32_t mode, uint32_t w, uint32_t h, uint32_t n) {
    std::vector<uint8_t> b(kSequenceHeaderBytes, 0);
    b[0] = 'B'; b[1] = 'S'; b[2] = 'Q'; b[3] = '1';
    const auto put = [&](size_t at, uint32_t v) {
        b[at + 0] = static_cast<uint8_t>(v & 0xFFu);
        b[at + 1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
        b[at + 2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
        b[at + 3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
    };
    put(4, mode); put(8, w); put(12, h); put(16, n);
    return b;
}

/// The body of a PNM file the write* functions produced: everything past the
/// ASCII header. Found by the PARSERS, not by assuming a header length, so this
/// helper cannot agree with a broken writer by construction.
std::vector<uint8_t> pbmBody(const std::vector<uint8_t>& file) {
    const PbmHeader h = readPbmHeader(file.data(), file.size());
    if (!h.valid) return {};
    return std::vector<uint8_t>(file.begin() + static_cast<std::ptrdiff_t>(h.pixelOffset), file.end());
}
std::vector<uint8_t> pgmBody(const std::vector<uint8_t>& file) {
    const PgmHeader h = readPgmHeader(file.data(), file.size());
    if (!h.valid) return {};
    return std::vector<uint8_t>(file.begin() + static_cast<std::ptrdiff_t>(h.pixelOffset), file.end());
}

/// A deterministically pseudorandom binary matrix, the suite's content source.
template <typename W>
BinMat<W> randomBits(size_t w, size_t h, uint64_t seed) {
    std::vector<uint8_t> img(w * h);
    uint64_t st = seed;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<uint8_t>(st >> 40);
    }
    BinMat<W> m(static_cast<int>(w), static_cast<int>(h));
    packBits<PackRule::GreaterThan, uint8_t, W>(img.data(), w, h, w, m.view(), 100);
    return m;
}

/// Words that differ between two equal-sized matrices, whole words, so the
/// padding bits are compared too.
template <typename W>
size_t wordsDiffering(const BinMat<W>& a, const BinMat<W>& b) {
    const size_t words = (a.getWidth() + BinMat<W>::WordBits - 1) / BinMat<W>::WordBits;
    size_t diff = 0;
    for (size_t y = 0; y < a.getHeight(); ++y)
        for (size_t i = 0; i < words; ++i)
            if (a.constView().row(y)[i] != b.constView().row(y)[i]) ++diff;
    return diff;
}

/// Set bits past `width` -- CLAUDE.md's padding invariant, on the reading side.
template <typename W>
size_t paddingBitsSet(const BinMat<W>& m) {
    const size_t words = (m.getWidth() + BinMat<W>::WordBits - 1) / BinMat<W>::WordBits;
    size_t set = 0;
    for (size_t y = 0; y < m.getHeight(); ++y)
        for (size_t x = m.getWidth(); x < words * BinMat<W>::WordBits; ++x)
            if ((m.constView().row(y)[x / BinMat<W>::WordBits] >>
                 (x % BinMat<W>::WordBits)) & W{1}) ++set;
    return set;
}

/// The whole file, or empty. Paths are derived from __FILE__ so the suite runs
/// from any working directory (the pattern tests/test_opencv_interop.cpp uses).
std::vector<uint8_t> readFile(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return {};
    std::vector<uint8_t> bytes;
    uint8_t chunk[4096];
    size_t got = 0;
    while ((got = std::fread(chunk, 1, sizeof(chunk), f)) > 0)
        bytes.insert(bytes.end(), chunk, chunk + got);
    std::fclose(f);
    return bytes;
}

std::string imageDir() {
    const std::string self = __FILE__;
    return self.substr(0, self.rfind('/')) + "/images";
}

} // namespace

BINCV_TEST(Sequence, HeaderParsesAndRejects) {
    // The two modes, and the frame size each implies. 32 pixels is 4 P4 bytes;
    // 33 forces the row up to 5, which is the byte-boundary padding rule.
    const std::vector<uint8_t> h8 = blobHeader(0, 32, 24, 3);
    const SequenceHeader a = readSequenceHeader(h8.data(), h8.size());
    std::printf(" 8-bit header: valid=%d mode=%u %zux%zu x%zu, %zu B/frame\n",
                a.valid ? 1 : 0, static_cast<unsigned>(a.mode), a.width, a.height,
                a.frameCount, a.frameBytes);
    BINCV_CHECK(a.valid);
    BINCV_CHECK(a.mode == kSequenceMode8Bit);
    BINCV_CHECK(a.width == 32 && a.height == 24 && a.frameCount == 3);
    BINCV_CHECK_EQ(a.frameBytes, size_t{32 * 24});

    const std::vector<uint8_t> hp = blobHeader(1, 33, 24, 2);
    const SequenceHeader b = readSequenceHeader(hp.data(), hp.size());
    BINCV_CHECK(b.valid && b.mode == kSequenceModePacked);
    BINCV_CHECK_EQ(b.frameBytes, size_t{5 * 24});

    // A header is parseable from EXACTLY its own 32 bytes, with no body behind
    // it -- the streaming property, and the "reject before allocating" one: a
    // target reads these 32 bytes off a UART and decides from frameBytes and
    // frameCount whether the rest can fit at all.
    BINCV_CHECK(readSequenceHeader(h8.data(), kSequenceHeaderBytes).valid);

    // Each rejection is a real corruption, answered with valid == false.
    std::vector<uint8_t> bad = blobHeader(0, 32, 24, 3);
    bad[3] = '2';                                                   // wrong magic
    BINCV_CHECK(!readSequenceHeader(bad.data(), bad.size()).valid);
    bad = blobHeader(2, 32, 24, 3);                                 // unknown mode
    BINCV_CHECK(!readSequenceHeader(bad.data(), bad.size()).valid);
    bad = blobHeader(0, 0, 24, 3);                                  // zero width
    BINCV_CHECK(!readSequenceHeader(bad.data(), bad.size()).valid);
    bad = blobHeader(0, 32, 0, 3);                                  // zero height
    BINCV_CHECK(!readSequenceHeader(bad.data(), bad.size()).valid);
    bad = blobHeader(0, 32, 24, 3);
    bad[25] = 1;   // a reserved byte in use means a revision this reader is not
    BINCV_CHECK(!readSequenceHeader(bad.data(), bad.size()).valid);
    const std::vector<uint8_t> good = blobHeader(0, 32, 24, 3);
    BINCV_CHECK(!readSequenceHeader(good.data(), 31).valid);        // header cut short
    BINCV_CHECK(!readSequenceHeader(nullptr, 0).valid);

    // The extreme the u32 fields can express: 0xFFFFFFFF x 0xFFFFFFFF bytes.
    // (2^32-1)^2 FITS in a 64-bit size_t, so on every host that RUNS this suite
    // the header is arithmetically valid and its frameBytes exact -- the
    // header's own overflow guard is live only at 32-bit pointer width, where
    // verify_cortex_m.sh compiles it. What a 64-bit host must still refuse is
    // any frame of that size from a real buffer: nothing addressable holds one,
    // and the range check believes the buffer.
    bad = blobHeader(0, 0xFFFFFFFFu, 0xFFFFFFFFu, 3);
    const SequenceHeader huge = readSequenceHeader(bad.data(), bad.size());
    std::printf(" 4G x 4G header: valid=%d, frameBytes=%zu\n", huge.valid ? 1 : 0,
                huge.frameBytes);
    BINCV_CHECK(huge.valid);
    BINCV_CHECK_EQ(huge.frameBytes, size_t{0xFFFFFFFFu} * size_t{0xFFFFFFFFu});
    BINCV_CHECK(!sequenceFrame(huge, bad.data(), bad.size(), 0).valid);
    BINCV_CHECK(!sequenceFrame(huge, bad.data(), bad.size(), 2).valid);
}

BINCV_TEST(Sequence, FrameRangeIsBoundsChecked) {
    // Mode 1, 8x2: one P4 byte per row, two per frame, two frames -- small
    // enough that every offset below is checkable by eye.
    std::vector<uint8_t> blob = blobHeader(1, 8, 2, 2);
    const uint8_t bodies[] = {0xA5, 0x5A, 0xFF, 0x00};
    for (const uint8_t v : bodies) blob.push_back(v);
    const SequenceHeader h = readSequenceHeader(blob.data(), blob.size());
    BINCV_CHECK(h.valid);
    BINCV_CHECK_EQ(h.frameBytes, size_t{2});

    const SequenceFrameRange f0 = sequenceFrame(h, blob.data(), blob.size(), 0);
    const SequenceFrameRange f1 = sequenceFrame(h, blob.data(), blob.size(), 1);
    std::printf(" frame 0 at +%td, frame 1 at +%td, both %zu bytes\n",
                f0.data - blob.data(), f1.data - blob.data(), f0.size);
    BINCV_CHECK(f0.valid && f0.size == 2 && f0.data == blob.data() + 32);
    BINCV_CHECK(f1.valid && f1.size == 2 && f1.data == blob.data() + 34);
    BINCV_CHECK(f0.data[0] == 0xA5 && f1.data[1] == 0x00);

    // Out of range, and truncated: the header PROMISES two frames, but the
    // range check believes the buffer, not the promise.
    BINCV_CHECK(!sequenceFrame(h, blob.data(), blob.size(), 2).valid);
    BINCV_CHECK(!sequenceFrame(h, blob.data(), blob.size() - 1, 1).valid);
    BINCV_CHECK(sequenceFrame(h, blob.data(), blob.size() - 1, 0).valid);

    // The unpackers inherit the same verdicts.
    BinMat<uint32_t> m(8, 2);
    BINCV_CHECK(!(readSequenceFrame<uint32_t>(h, blob.data(), blob.size(), 2, m.view())));
    BINCV_CHECK(!(readSequenceFrame<uint32_t>(h, blob.data(), blob.size() - 1, 1, m.view())));
    BINCV_CHECK((readSequenceFrame<uint32_t>(h, blob.data(), blob.size(), 1, m.view())));
    // A wrong-sized destination is a rejection, not a partial read.
    BinMat<uint32_t> wrong(9, 2);
    BINCV_CHECK(!(readSequenceFrame<uint32_t>(h, blob.data(), blob.size(), 0, wrong.view())));
}

namespace {

/// One packed-mode round trip at a word type: three frames of content ->
/// writePbm bodies -> hand-built blob -> readSequenceFrame -> compare, whole
/// words. Returns words differing; padding bits set come back separately.
template <typename W>
size_t packedRoundTrip(size_t w, size_t h, size_t& padBitsSet) {
    constexpr size_t kFrames = 3;
    std::vector<BinMat<W>> frames;
    std::vector<uint8_t> blob = blobHeader(kSequenceModePacked, static_cast<uint32_t>(w),
                                           static_cast<uint32_t>(h), kFrames);
    for (size_t f = 0; f < kFrames; ++f) {
        frames.push_back(randomBits<W>(w, h, 20260906 + f));
        std::vector<uint8_t> file(writePbm<W>(frames.back().constView(), nullptr, 0));
        writePbm<W>(frames.back().constView(), file.data(), file.size());
        const std::vector<uint8_t> body = pbmBody(file);
        blob.insert(blob.end(), body.begin(), body.end());
    }

    const SequenceHeader sh = readSequenceHeader(blob.data(), blob.size());
    if (!sh.valid || sh.frameBytes != impl::pbmRowBytes(w) * h) return ~size_t{0};

    size_t diff = 0;
    padBitsSet = 0;
    for (size_t f = 0; f < kFrames; ++f) {
        BinMat<W> got(static_cast<int>(w), static_cast<int>(h));
        if (!readSequenceFrame<W>(sh, blob.data(), blob.size(), f, got.view()))
            return ~size_t{0};
        diff += wordsDiffering(frames[f], got);
        padBitsSet += paddingBitsSet(got);
    }
    return diff;
}

} // namespace

BINCV_TEST(Sequence, PackedRoundTripsAtEveryWordType) {
    // 67: the last file byte is partial AND the last word is partial at every
    // word type -- see the file header.
    constexpr size_t kW = 67, kH = 11;
    size_t pad8 = 0, pad16 = 0, pad32 = 0, pad64 = 0;
    const size_t d8 = packedRoundTrip<uint8_t>(kW, kH, pad8);
    const size_t d16 = packedRoundTrip<uint16_t>(kW, kH, pad16);
    const size_t d32 = packedRoundTrip<uint32_t>(kW, kH, pad32);
    const size_t d64 = packedRoundTrip<uint64_t>(kW, kH, pad64);
    std::printf(" packed round trip %zux%zu x3: words differing u8=%zu u16=%zu u32=%zu"
                " u64=%zu\n", kW, kH, d8, d16, d32, d64);
    BINCV_CHECK_EQ(d8, size_t{0});
    BINCV_CHECK_EQ(d16, size_t{0});
    BINCV_CHECK_EQ(d32, size_t{0});
    BINCV_CHECK_EQ(d64, size_t{0});
    std::printf(" padding bits set after the read: u8=%zu u16=%zu u32=%zu u64=%zu\n",
                pad8, pad16, pad32, pad64);
    BINCV_CHECK_EQ(pad8, size_t{0});
    BINCV_CHECK_EQ(pad16, size_t{0});
    BINCV_CHECK_EQ(pad32, size_t{0});
    BINCV_CHECK_EQ(pad64, size_t{0});
}

BINCV_TEST(Sequence, EightBitRoundTripsThroughPgmBodies) {
    // Mode 0's body is EXACTLY a P5 body. Assembled from writePgm output with
    // the header stripped; writePgm emits 255 for a set bit, so NonZero
    // recovers the same bits -- the identical relationship test_pack.cpp pins
    // for the standalone file format.
    constexpr size_t kW = 67, kH = 11, kFrames = 3;
    std::vector<BinMat<uint32_t>> frames;
    std::vector<std::vector<uint8_t>> bodies;
    std::vector<uint8_t> blob = blobHeader(kSequenceMode8Bit, kW, kH, kFrames);
    for (size_t f = 0; f < kFrames; ++f) {
        frames.push_back(randomBits<uint32_t>(kW, kH, 424243 + f));
        std::vector<uint8_t> file(writePgm<uint32_t>(frames.back().constView(), nullptr, 0));
        writePgm<uint32_t>(frames.back().constView(), file.data(), file.size());
        bodies.push_back(pgmBody(file));
        blob.insert(blob.end(), bodies.back().begin(), bodies.back().end());
    }

    const SequenceHeader sh = readSequenceHeader(blob.data(), blob.size());
    BINCV_CHECK(sh.valid && sh.mode == kSequenceMode8Bit);
    BINCV_CHECK_EQ(sh.frameBytes, kW * kH);

    size_t bytesDiffer = 0, wordsDiffer = 0, padBits = 0;
    for (size_t f = 0; f < kFrames; ++f) {
        // The raw range IS the input contract: its bytes are the P5 body's.
        const SequenceFrameRange r = sequenceFrame(sh, blob.data(), blob.size(), f);
        BINCV_CHECK(r.valid && r.size == bodies[f].size());
        for (size_t i = 0; i < r.size; ++i)
            if (r.data[i] != bodies[f][i]) ++bytesDiffer;

        BinMat<uint32_t> got(kW, kH);
        BINCV_CHECK((readSequenceFrame<PackRule::NonZero, uint32_t>(
            sh, blob.data(), blob.size(), f, got.view())));
        wordsDiffer += wordsDiffering(frames[f], got);
        padBits += paddingBitsSet(got);
    }
    std::printf(" 8-bit round trip %zux%zu x%zu: %zu body bytes differ, %zu words differ,"
                " %zu padding bits\n", kW, kH, kFrames, bytesDiffer, wordsDiffer, padBits);
    BINCV_CHECK_EQ(bytesDiffer, size_t{0});
    BINCV_CHECK_EQ(wordsDiffer, size_t{0});
    BINCV_CHECK_EQ(padBits, size_t{0});

    // The mode-0 unpacker refuses a packed header and vice versa: the mode
    // field exists so a reader CANNOT confuse the two body layouts.
    BinMat<uint32_t> m(kW, kH);
    const std::vector<uint8_t> ph = blobHeader(kSequenceModePacked, kW, kH, 1);
    const SequenceHeader wrongMode = readSequenceHeader(ph.data(), ph.size());
    BINCV_CHECK(!(readSequenceFrame<PackRule::NonZero, uint32_t>(
        wrongMode, blob.data(), blob.size(), 0, m.view())));
    BINCV_CHECK(!(readSequenceFrame<uint32_t>(sh, blob.data(), blob.size(), 0, m.view())));
}

BINCV_TEST(Sequence, BodyFormMatchesTheWholeBlobForm) {
    // The streaming caller holds ONE frame's bytes, not the blob. Feeding a
    // frame's body through readSequenceFrameBody must give exactly what the
    // whole-blob read gives -- otherwise streaming is approximate, and the
    // reused-buffer fread loop the worked example runs would be unsound.
    constexpr size_t kW = 67, kH = 11;
    const BinMat<uint32_t> src = randomBits<uint32_t>(kW, kH, 555);
    std::vector<uint8_t> file(writePbm<uint32_t>(src.constView(), nullptr, 0));
    writePbm<uint32_t>(src.constView(), file.data(), file.size());
    const std::vector<uint8_t> body = pbmBody(file);
    std::vector<uint8_t> blob = blobHeader(kSequenceModePacked, kW, kH, 1);
    blob.insert(blob.end(), body.begin(), body.end());

    const SequenceHeader sh = readSequenceHeader(blob.data(), blob.size());
    BINCV_CHECK(sh.valid);
    BinMat<uint32_t> whole(kW, kH), streamed(kW, kH);
    BINCV_CHECK((readSequenceFrame<uint32_t>(sh, blob.data(), blob.size(), 0, whole.view())));
    BINCV_CHECK((readSequenceFrameBody<uint32_t>(sh, body.data(), body.size(),
                                                 streamed.view())));
    const size_t diff = wordsDiffering(whole, streamed);
    std::printf(" body form vs whole-blob form: %zu words differ\n", diff);
    BINCV_CHECK_EQ(diff, size_t{0});

    // And it rejects what the whole-blob form would: a short body, a wrong
    // mode, a wrong-sized destination.
    BINCV_CHECK(!(readSequenceFrameBody<uint32_t>(sh, body.data(), body.size() - 1,
                                                  streamed.view())));
    BinMat<uint32_t> wrong(kW, kH + 1);
    BINCV_CHECK(!(readSequenceFrameBody<uint32_t>(sh, body.data(), body.size(),
                                                  wrong.view())));
    const std::vector<uint8_t> h8 = blobHeader(kSequenceMode8Bit, kW, kH, 1);
    const SequenceHeader mode0 = readSequenceHeader(h8.data(), h8.size());
    BINCV_CHECK(!(readSequenceFrameBody<uint32_t>(mode0, body.data(), body.size(),
                                                  streamed.view())));
}

// ---------------------------------------------------------------------------
// THE COMMITTED FIXTURES: the real tool's output, pinned without python.
//
// tests/images/seq_32x24_{8bit,packed}.bsq were generated ONCE by
// scripts/make_sequence_blob.py from the three seq_32x24_f*.pgm sources
// beside them (threshold 17, the reference's). If the tool's format or its
// sensor stage ever drifts from the reader or from binCV's kernels,
// regenerating the fixtures turns one of these red.
// ---------------------------------------------------------------------------

BINCV_TEST(Sequence, FixtureEightBitBlobIsTheSourcePgmsBackToBack) {
    const std::string dir = imageDir();
    const std::vector<uint8_t> blob = readFile(dir + "/seq_32x24_8bit.bsq");
    std::printf(" %s/seq_32x24_8bit.bsq: %zu bytes\n", dir.c_str(), blob.size());
    BINCV_CHECK(!blob.empty());
    const SequenceHeader sh = readSequenceHeader(blob.data(), blob.size());
    BINCV_CHECK(sh.valid && sh.mode == kSequenceMode8Bit);
    BINCV_CHECK(sh.width == 32 && sh.height == 24 && sh.frameCount == 3);
    BINCV_CHECK_EQ(blob.size(), kSequenceHeaderBytes + sh.frameBytes * sh.frameCount);

    size_t bytesDiffer = 0;
    for (size_t f = 0; f < 3; ++f) {
        const std::vector<uint8_t> pgm =
            readFile(dir + "/seq_32x24_f" + std::to_string(f) + ".pgm");
        BINCV_CHECK(!pgm.empty());
        const std::vector<uint8_t> body = pgmBody(pgm);
        const SequenceFrameRange r = sequenceFrame(sh, blob.data(), blob.size(), f);
        BINCV_CHECK(r.valid && r.size == body.size());
        for (size_t i = 0; i < r.size && i < body.size(); ++i)
            if (r.data[i] != body[i]) ++bytesDiffer;
    }
    std::printf(" 3 frames against their source PGMs: %zu bytes differ\n", bytesDiffer);
    BINCV_CHECK_EQ(bytesDiffer, size_t{0});
}

namespace {

/// The packed fixture against binCV's own sensor stage at one word type.
/// Returns words differing across all three frames, or ~0 on a setup failure.
template <typename W>
size_t fixtureAgainstSensorStage(const std::string& dir, size_t& padBitsSet) {
    const std::vector<uint8_t> blob = readFile(dir + "/seq_32x24_packed.bsq");
    if (blob.empty()) return ~size_t{0};
    const SequenceHeader sh = readSequenceHeader(blob.data(), blob.size());
    if (!sh.valid || sh.mode != kSequenceModePacked || sh.frameCount != 3)
        return ~size_t{0};

    size_t diff = 0;
    padBitsSet = 0;
    for (size_t f = 0; f < 3; ++f) {
        const std::vector<uint8_t> pgm =
            readFile(dir + "/seq_32x24_f" + std::to_string(f) + ".pgm");
        const PgmHeader ph = readPgmHeader(pgm.data(), pgm.size());
        if (!ph.valid || ph.width != sh.width || ph.height != sh.height)
            return ~size_t{0};

        // The exact sensor-stage spelling benchmark/frontend_sequence.cpp runs:
        // the reference L median, then the wide edge threshold at 17 -- the
        // same two stages the python tool ran to make this fixture.
        std::vector<uint8_t> med(ph.width * ph.height);
        medianWide<3, uint8_t>(pgm.data() + ph.pixelOffset, ph.width, ph.height,
                               ph.width, med.data(), ph.width, kMedianReferenceL);
        BinMat<W> expect(static_cast<int>(ph.width), static_cast<int>(ph.height));
        edgeThreshold<EdgeCombine::Or, EdgeRelation::Ge, EdgeSpatial::Wide, uint8_t, W>(
            med.data(), ph.width, ph.height, ph.width, expect.view(), uint8_t{17});

        BinMat<W> got(static_cast<int>(sh.width), static_cast<int>(sh.height));
        if (!readSequenceFrame<W>(sh, blob.data(), blob.size(), f, got.view()))
            return ~size_t{0};
        diff += wordsDiffering(expect, got);
        padBitsSet += paddingBitsSet(got);
    }
    return diff;
}

} // namespace

BINCV_TEST(Sequence, FixturePackedBlobIsBincvsOwnSensorStage) {
    // The tool's python preprocessing (cv2 or numpy) and binCV's
    // medianWide + edgeThreshold are independent implementations of the
    // reference's two-stage sensor pipeline. Zero tolerance: they must agree on
    // every bit, or one side's border rule, relation or median has drifted.
    const std::string dir = imageDir();
    size_t pad32 = 0, pad64 = 0;
    const size_t d32 = fixtureAgainstSensorStage<uint32_t>(dir, pad32);
    const size_t d64 = fixtureAgainstSensorStage<uint64_t>(dir, pad64);
    std::printf(" packed fixture vs medianWide+edgeThreshold: words differing u32=%zu"
                " u64=%zu, padding bits %zu/%zu\n", d32, d64, pad32, pad64);
    BINCV_CHECK_EQ(d32, size_t{0});
    BINCV_CHECK_EQ(d64, size_t{0});
    BINCV_CHECK_EQ(pad32, size_t{0});
    BINCV_CHECK_EQ(pad64, size_t{0});
}

BINCV_TEST_MAIN("test_sequence")
