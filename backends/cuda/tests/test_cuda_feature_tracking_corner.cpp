// Device-versus-host bit-exactness for the pipeline's entry point: FAST, the
// minimum-eigenvalue response, good-features selection and sub-pixel refinement.
//
// The host library is the truth and the format is shared, so every case here is
// a raw comparison: run both, download, compare bytes. A device kernel that is
// faster and different is not an optimization.
//
// HOST C++ RATHER THAN A CUDA TRANSLATION UNIT, and that choice is load-bearing
// for two of these families. `ops/fast.hpp` gates its AVX2 bit-plane kernels off
// under __CUDACC__ on purpose, and so does `ops/medianWide.hpp`; compiled by
// nvcc the host arm would silently become the portable one and this suite would
// be comparing the device against a kernel nobody runs. Everything the device
// side needs here is a host-callable launcher, so there is no kernel to write.
//
// WHAT IS COMPARED, AND WHERE THE COMPARISON STOPS.
//   * FAST: memcmp of the whole FastCorner array -- positions AND the long long
//     score -- plus the count. The comparison is on the HOST type, after
//     `cuda::toHost`: the device record is 12 bytes and widens its score on the
//     way out, so what a caller actually reads back is what gets memcmp'd against
//     the host detector's own array. ORDER IS PART OF THE COMPARISON; the sort
//     inside detectFastAsync is what makes that possible. A TRUNCATED run is
//     compared only on `found()`, because compaction.hpp refuses to promise which
//     elements an atomic kept and a suite that forgets this fails intermittently.
//   * Response map: memcmp over every pixel, bit for bit, no tolerance.
//   * Corners: memcmp of the Corner array plus the whole result triple against
//     goodFeaturesToTrackStreaming.
//   * SubPix: exact float compare of the positions, no tolerance -- a tolerance
//     fitted to the observed value is the failure ops/subpix.hpp's own docstring
//     records -- plus all four counters.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/corner.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/fast.hpp"
#include "bincv/cuda/subpix.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/corner.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/fast.hpp"
#include "bincv/ops/subpix.hpp"
#include "bincv/quantMat.hpp"
#include "test_util.hpp"

namespace bc = bincv::cuda;
using bincv::BinMat;
using bincv::Corner;
using bincv::CornerResult;
using bincv::FastCorner;
using bincv::GoodFeaturesParams;
using bincv::Point2f;
using bincv::ResponseMap;
using bincv::SubPixParams;
using bincv::SubPixResult;
using bincv::TernaryMat;
using Word = uint32_t;

namespace {

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// A binary frame with structure rather than pure noise: pure noise makes almost
/// every pixel a FAST corner and almost no pixel a response maximum, so it
/// exercises neither compaction path at a realistic rate.
BinMat<Word> structuredBits(size_t w, size_t h, uint64_t seed, int period = 7) {
    BinMat<Word> m(static_cast<int>(w), static_cast<int>(h));
    for (size_t y = 0; y < h; ++y) {
        Word* row = m.view().row(y);
        for (size_t x = 0; x < w; ++x) {
            const bool blob = ((x / static_cast<size_t>(period)) +
                               (y / static_cast<size_t>(period))) % 2 == 0;
            const bool noise = (splitmix(seed) & 7u) == 0u;
            if (blob != noise) row[x / 32] |= (Word{1} << (x % 32));
        }
    }
    return m;
}

BinMat<Word> constantBits(size_t w, size_t h, bool value) {
    BinMat<Word> m(static_cast<int>(w), static_cast<int>(h));
    if (!value) return m;
    for (size_t y = 0; y < h; ++y) {
        Word* row = m.view().row(y);
        for (size_t x = 0; x < w; ++x) row[x / 32] |= (Word{1} << (x % 32));
    }
    return m;
}

// ---------------------------------------------------------------------------
// FAST
// ---------------------------------------------------------------------------

struct DeviceFastRun {
    std::vector<FastCorner> corners;
    uint32_t found = 0;
    bool truncated = false;
    cudaError_t err = cudaSuccess;
};

DeviceFastRun runDeviceFast(const BinMat<Word>& img, size_t capacity, int arcLength,
                            bool dirtyPadding = false) {
    DeviceFastRun out;
    bc::DeviceBinMat dimg(static_cast<int>(img.getWidth()), static_cast<int>(img.getHeight()));
    if (!dirtyPadding) {
        bc::upload(img.constView(), dimg.view());
    } else {
        // A raw pitched copy of whole WORDS, with every bit past `width` in the
        // trailing word set. `upload` moves pixel bytes only, by contract, so it
        // cannot build this -- and this is exactly what the padding invariant is
        // a claim about.
        const size_t words = bc::rowWords(img.getWidth());
        const uint32_t tail = bc::rowTailMask(img.getWidth());
        std::vector<Word> raw(words * img.getHeight());
        for (size_t y = 0; y < img.getHeight(); ++y) {
            const Word* src = img.constView().row(y);
            for (size_t w = 0; w < words; ++w) raw[y * words + w] = src[w];
            raw[y * words + words - 1] |= ~tail;
        }
        cudaMemcpy2D(dimg.view().ptr, dimg.view().stride * sizeof(Word), raw.data(),
                     words * sizeof(Word), words * sizeof(Word), img.getHeight(),
                     cudaMemcpyHostToDevice);
    }

    bc::DeviceArray<bc::DeviceFastCorner> dout(capacity);
    bc::DeviceAppendCounter counter;
    counter.reset();
    const bc::DeviceFastCornerBuffer buf =
        bc::DeviceFastCornerBuffer(dout.data(), counter.devicePtr(),
                                   static_cast<uint32_t>(capacity));
    // REFERENCE, not the default: this harness is called from `checkFastAllArms`,
    // which flips `fastOrderedEnabled()` under one buffer. A caller that runs both
    // arms sizes for the larger, and the ordered arm reads only the front of it.
    const size_t scratchBytes = bc::fastScratchBytes(img.getWidth(), img.getHeight(),
                                                     capacity, bc::FastArm::Reference);
    bc::DeviceArray<uint8_t> scratch(scratchBytes);

    out.err = bc::detectFastAsync(dimg.constView(), buf, scratch.data(), scratchBytes, arcLength);
    cudaDeviceSynchronize();
    if (out.err != cudaSuccess) return out;

    bc::DeviceAppendResult res;
    bc::readAppendResult(buf, res);
    out.found = res.found();
    out.truncated = res.truncated();
    std::vector<bc::DeviceFastCorner> raw(res.acceptTruncated());
    if (!raw.empty()) {
        bc::downloadAppended(buf, res, raw.data());
        cudaStreamSynchronize(nullptr);
    }
    out.corners.resize(raw.size());
    bc::toHost(raw.data(), raw.size(), out.corners.data());
    return out;
}

/// One complete-run comparison: the device array must be the host array, byte
/// for byte and in the host's raster order.
void checkFastMatches(const BinMat<Word>& img, int arcLength, const char* what) {
    const size_t generous = static_cast<size_t>(img.getWidth()) * img.getHeight() + 16;
    std::vector<FastCorner> host(generous);
    bool hostTruncated = false;
    const size_t hostCount =
        bincv::detectFast<Word>(img.constView(), host.data(), generous, &hostTruncated, arcLength);
    BINCV_CHECK(!hostTruncated);

    const DeviceFastRun dev = runDeviceFast(img, generous, arcLength);
    BINCV_CHECK_EQ(static_cast<int>(dev.err), static_cast<int>(cudaSuccess));
    BINCV_CHECK(!dev.truncated);
    BINCV_CHECK_EQ(dev.corners.size(), hostCount);
    if (dev.corners.size() != hostCount) {
        std::printf("  [%s] arc %d: device %zu corners, host %zu\n", what, arcLength,
                    dev.corners.size(), hostCount);
        return;
    }
    const bool same = hostCount == 0 || std::memcmp(dev.corners.data(), host.data(),
                                                    hostCount * sizeof(FastCorner)) == 0;
    BINCV_CHECK(same);
    if (!same) {
        for (size_t i = 0; i < hostCount; ++i) {
            if (dev.corners[i].x != host[i].x || dev.corners[i].y != host[i].y ||
                dev.corners[i].score != host[i].score) {
                std::printf("  [%s] arc %d: first mismatch at %zu: device (%d,%d,%lld) "
                            "host (%d,%d,%lld)\n",
                            what, arcLength, i, dev.corners[i].x, dev.corners[i].y,
                            dev.corners[i].score, host[i].x, host[i].y, host[i].score);
                break;
            }
        }
    }
}

/// Every arm combination, held to one answer -- which is what makes the runtime
/// switches a claim rather than a comment.
void checkFastAllArms(const BinMat<Word>& img, int arcLength, const char* what) {
    for (int ordered = 1; ordered >= 0; --ordered) {
        for (int tiled = 1; tiled >= 0; --tiled) {
            for (int maskScore = 1; maskScore >= 0; --maskScore) {
                bc::impl::fastOrderedEnabled() = ordered != 0;
                bc::impl::fastTiledEnabled() = tiled != 0;
                bc::impl::fastMaskScoreEnabled() = maskScore != 0;
                checkFastMatches(img, arcLength, what);
            }
        }
    }
    bc::impl::fastOrderedEnabled() = true;
    bc::impl::fastTiledEnabled() = true;
    bc::impl::fastMaskScoreEnabled() = true;
}

} // namespace

BINCV_TEST(CudaFast, MatchesHostAcrossShapes) {
    const size_t widths[] = {752, 640, 97, 65, 33, 31, 7};
    const size_t heights[] = {37, 13, 7};
    uint64_t seed = 0x1234;
    for (size_t w : widths) {
        for (size_t h : heights) {
            const BinMat<Word> img = structuredBits(w, h, seed++);
            checkFastAllArms(img, 9, "shape");
        }
    }
}

BINCV_TEST(CudaFast, MatchesHostAcrossArcLengths) {
    // 9 and 12 take the tiled arm's compile-time schedule; 10, 16, 3 and 1 fall
    // to the reference arm, which is the gate-excluded case the benchmark
    // reports at ~1.00x.
    const int arcs[] = {9, 12, 10, 16, 3, 1};
    const BinMat<Word> img = structuredBits(97, 29, 0xBEEF);
    for (int a : arcs) checkFastAllArms(img, a, "arc");
}

BINCV_TEST(CudaFast, HostScoringPathsBothAgree) {
    // impl::fastScoreMaskThreshold picks between the host's two scoring arms per
    // chunk. Forcing it to each end re-proves the host's own claim that they
    // agree, and holds the device to BOTH results rather than to whichever one
    // the default happened to take.
    const BinMat<Word> img = structuredBits(752, 19, 0xC0FFEE);
    bincv::impl::fastScoreMaskThreshold() = 0;
    checkFastAllArms(img, 9, "host masks always");
    bincv::impl::fastScoreMaskThreshold() = 1 << 20;
    checkFastAllArms(img, 9, "host masks never");
    bincv::impl::fastScoreMaskThreshold() = 3;
}

BINCV_TEST(CudaFast, PaddingBitsPastWidthAreNotCorners) {
    // 752 is 23 full words plus a 16-pixel tail, so the trailing word carries 16
    // padding bits. A dirty plane must give byte-identical output: this is the
    // device statement of the padding rule and it is exactly what the cross-word
    // read at the last word of a row can get wrong.
    const size_t widths[] = {752, 97, 65, 33, 31};
    for (size_t w : widths) {
        const BinMat<Word> img = structuredBits(w, 23, w * 31u + 7u);
        const size_t generous = w * 23 + 16;
        std::vector<FastCorner> host(generous);
        bool ht = false;
        const size_t hostCount =
            bincv::detectFast<Word>(img.constView(), host.data(), generous, &ht, 9);
        const DeviceFastRun dev = runDeviceFast(img, generous, 9, /*dirtyPadding=*/true);
        BINCV_CHECK_EQ(dev.corners.size(), hostCount);
        if (dev.corners.size() == hostCount) {
            BINCV_CHECK(hostCount == 0 ||
                        std::memcmp(dev.corners.data(), host.data(),
                                    hostCount * sizeof(FastCorner)) == 0);
        }
    }
}

BINCV_TEST(CudaFast, UniformFramesHaveNoCorners) {
    for (int value = 0; value <= 1; ++value) {
        const BinMat<Word> img = constantBits(97, 29, value != 0);
        const DeviceFastRun dev = runDeviceFast(img, 4096, 9);
        BINCV_CHECK_EQ(dev.found, 0u);
        BINCV_CHECK(!dev.truncated);
    }
}

BINCV_TEST(CudaFast, BorderColumnsAreNeverCandidates) {
    // A single isolated pixel at each of x = 3, x = width - 4 and the word
    // boundaries 31, 32, 63 -- and at x = 2 and x = width - 3, where the host
    // refuses. Compared against the host, which is the one that decides.
    const size_t w = 97, h = 23;
    const int xs[] = {2, 3, 31, 32, 63, static_cast<int>(w) - 4, static_cast<int>(w) - 3};
    for (int x : xs) {
        BinMat<Word> img(static_cast<int>(w), static_cast<int>(h));
        for (int dy = -1; dy <= 1; ++dy) {
            Word* row = img.view().row(static_cast<size_t>(11 + dy));
            row[static_cast<size_t>(x) / 32] |= (Word{1} << (static_cast<size_t>(x) % 32));
        }
        checkFastAllArms(img, 9, "border column");
    }
}

BINCV_TEST(CudaFast, TruncationCountsTheTruthAndSaysSo) {
    const BinMat<Word> img = structuredBits(97, 29, 0x5150);
    const size_t generous = 97 * 29 + 16;
    std::vector<FastCorner> host(generous);
    bool ht = false;
    const size_t total = bincv::detectFast<Word>(img.constView(), host.data(), generous, &ht, 9);
    BINCV_CHECK(total > 4);

    // A capacity that cannot hold the answer. The COUNTER is still the true
    // total -- which is exactly the capacity a re-run needs -- and the stored
    // elements are NOT compared against the host's prefix, because the atomic
    // decided which ones they are.
    const size_t small = total / 2;
    const DeviceFastRun dev = runDeviceFast(img, small, 9);
    BINCV_CHECK(dev.truncated);
    BINCV_CHECK_EQ(static_cast<size_t>(dev.found), total);
    BINCV_CHECK_EQ(dev.corners.size(), small);

    // Capacity zero: the host returns no corners and reports no truncation, and
    // so does this.
    const DeviceFastRun none = runDeviceFast(img, 0, 9);
    BINCV_CHECK_EQ(none.found, 0u);
    BINCV_CHECK(!none.truncated);
}

BINCV_TEST(CudaFast, TheOrderedArmTruncatesTheHostsWay) {
    // The two arms differ on a TRUNCATED run and the difference is a property of
    // the arms, not of the operation: an atomic decides which corners the sort
    // arm stored, while a prefix sum writes every corner at its raster RANK, so
    // an index below `capacity` is the host's prefix by construction. Both are
    // held to the same `found()`; only the ordered one is held to the prefix.
    const BinMat<Word> img = structuredBits(97, 29, 0x5150);
    const size_t generous = 97 * 29 + 16;
    std::vector<FastCorner> host(generous);
    bool ht = false;
    const size_t total = bincv::detectFast<Word>(img.constView(), host.data(), generous, &ht, 9);
    BINCV_CHECK(total > 8);

    const size_t small = total / 2;
    BINCV_CHECK(bc::impl::fastOrderedApplies(97, 29, small));
    const DeviceFastRun dev = runDeviceFast(img, small, 9);
    BINCV_CHECK(dev.truncated);
    BINCV_CHECK_EQ(static_cast<size_t>(dev.found), total);
    BINCV_CHECK_EQ(dev.corners.size(), small);
    const bool prefix =
        std::memcmp(dev.corners.data(), host.data(), small * sizeof(FastCorner)) == 0;
    BINCV_CHECK(prefix);

    // Capacity 1 is the sharpest form of it: the single corner kept must be the
    // FIRST in raster order, not whichever thread arrived first.
    if (bc::impl::fastOrderedApplies(97, 29, 1)) {
        const DeviceFastRun one = runDeviceFast(img, 1, 9);
        BINCV_CHECK_EQ(one.corners.size(), size_t{1});
        BINCV_CHECK(std::memcmp(one.corners.data(), host.data(), sizeof(FastCorner)) == 0);
    }
}

BINCV_TEST(CudaFast, TheOrderedArmsGateIsTheCallersScratch) {
    // The gate-excluded case, named through the predicate rather than restated:
    // a capacity too small to hold one `uint32_t` per block cannot run the
    // ordered arm, and there both switch positions are the same code. The
    // benchmark reports ~1.00x for exactly this case.
    BINCV_CHECK(bc::impl::fastOrderedApplies(752, 480, 32768));
    BINCV_CHECK(bc::impl::fastOrderedApplies(752, 480, 512));
    BINCV_CHECK(!bc::impl::fastOrderedApplies(752, 480, 8));
    BINCV_CHECK(!bc::impl::fastOrderedApplies(752, 480, 0));
    // And it still answers correctly there -- the fallback is an arm, not a gap.
    const BinMat<Word> img = structuredBits(752, 37, 0x0A11);
    const size_t generous = 752 * 37 + 16;
    std::vector<FastCorner> host(generous);
    bool ht = false;
    const size_t total = bincv::detectFast<Word>(img.constView(), host.data(), generous, &ht, 9);
    BINCV_CHECK(total > 8);
    const DeviceFastRun gated = runDeviceFast(img, 8, 9);
    BINCV_CHECK_EQ(static_cast<size_t>(gated.found), total);
}

BINCV_TEST(CudaFast, RefusesAnArcLengthOutsideTheRing) {
    // R4: a device op may accept a narrower domain than its host twin, but it
    // must NAME it, assert it and RETURN AN ERROR outside it. Release only --
    // the assertion aborts in a checked build, which is what it is for.
#if !BINCV_DEBUG_CHECKS
    const BinMat<Word> img = structuredBits(64, 16, 7);
    bc::DeviceBinMat dimg(64, 16);
    bc::upload(img.constView(), dimg.view());
    bc::DeviceArray<bc::DeviceFastCorner> dout(16);
    bc::DeviceAppendCounter counter;
    counter.reset();
    const bc::DeviceFastCornerBuffer buf(dout.data(), counter.devicePtr(), 16u);
    const size_t want = bc::fastScratchBytes(64, 16, 16);
    bc::DeviceArray<uint8_t> scratch(want);
    BINCV_CHECK_EQ(static_cast<int>(bc::detectFastAsync(dimg.constView(), buf, scratch.data(),
                                                        want, 17)),
                   static_cast<int>(cudaErrorInvalidValue));
    BINCV_CHECK_EQ(static_cast<int>(bc::detectFastAsync(dimg.constView(), buf, scratch.data(),
                                                        want, 0)),
                   static_cast<int>(cudaErrorInvalidValue));
    // A scratch buffer shorter than the sizing function says is a refusal, not a
    // device-side out-of-bounds write.
    BINCV_CHECK_EQ(static_cast<int>(bc::detectFastAsync(dimg.constView(), buf, scratch.data(),
                                                        want - 1, 9)),
                   static_cast<int>(cudaErrorInvalidValue));
    BINCV_CHECK_EQ(static_cast<int>(cudaGetLastError()), static_cast<int>(cudaSuccess));
#endif
}

BINCV_TEST(CudaFast, AnOrderedSizedBufferMeetingTheReferenceArmIsRefused) {
    // THE PROPERTY THE ARM PARAMETER EXISTS FOR, exercised rather than documented.
    // `fastScratchBytes` is allowed to answer differently per arm only because the
    // op checks the arm it is ABOUT TO RUN against the bytes it was handed. So:
    // size for the ordered arm, flip the switch, and the call must come back
    // cudaErrorInvalidValue -- not truncate, not write past the buffer, not run.
    //
    // No build gate on this one: the short-buffer path is a plain comparison and a
    // return, not an assertion, so it is the same code in a checked build.
    const BinMat<Word> img = structuredBits(752, 37, 0x51ED);
    bc::DeviceBinMat dimg(752, 37);
    bc::upload(img.constView(), dimg.view());
    const uint32_t capacity = 4096;
    bc::DeviceArray<bc::DeviceFastCorner> dout(capacity);
    bc::DeviceAppendCounter counter;
    counter.reset();
    const bc::DeviceFastCornerBuffer buf(dout.data(), counter.devicePtr(), capacity);

    const size_t ordered = bc::fastScratchBytes(752, 37, capacity, bc::FastArm::Ordered);
    const size_t reference = bc::fastScratchBytes(752, 37, capacity, bc::FastArm::Reference);
    // The whole reason this round happened: the two numbers are far apart, and the
    // shipped arm wants the small one.
    BINCV_CHECK(ordered < reference);
    BINCV_CHECK(bc::impl::fastOrderedApplies(752, 37, capacity));

    bc::DeviceArray<uint8_t> small(ordered);
    bc::impl::fastOrderedEnabled() = true;
    BINCV_CHECK_EQ(static_cast<int>(bc::detectFastAsync(dimg.constView(), buf, small.data(),
                                                        ordered, 9)),
                   static_cast<int>(cudaSuccess));
    BINCV_CHECK_EQ(static_cast<int>(cudaDeviceSynchronize()), static_cast<int>(cudaSuccess));

    // Same buffer, other arm. This is the case that used to be impossible to reach
    // because every caller was handed the larger number whether it wanted it or not.
    bc::impl::fastOrderedEnabled() = false;
    counter.reset();
    BINCV_CHECK_EQ(static_cast<int>(bc::detectFastAsync(dimg.constView(), buf, small.data(),
                                                        ordered, 9)),
                   static_cast<int>(cudaErrorInvalidValue));
    bc::impl::fastOrderedEnabled() = true;
    BINCV_CHECK_EQ(static_cast<int>(cudaGetLastError()), static_cast<int>(cudaSuccess));

    // And the reference arm against a buffer sized for it still runs, so the
    // refusal above is the size and not the switch.
    bc::DeviceArray<uint8_t> big(reference);
    bc::impl::fastOrderedEnabled() = false;
    counter.reset();
    BINCV_CHECK_EQ(static_cast<int>(bc::detectFastAsync(dimg.constView(), buf, big.data(),
                                                        reference, 9)),
                   static_cast<int>(cudaSuccess));
    BINCV_CHECK_EQ(static_cast<int>(cudaDeviceSynchronize()), static_cast<int>(cudaSuccess));
    bc::impl::fastOrderedEnabled() = true;
}

BINCV_TEST(CudaFast, TheScoreNarrowingIsLosslessOverEveryValueItCanTake) {
    // THE DEVICE RECORD IS 12 BYTES AND THE HOST TYPE IS 16, so the score crosses a
    // narrowing on its way into device memory. This is the proof that the crossing
    // is lossless, and it is EXHAUSTIVE rather than a sample: the bit-plane score
    // is `impl::fastLongestRun(ring, arcLength)`, one 16-bit ring in and one int
    // out, so its attainable set is every value that function can return -- all
    // 65,536 rings at all sixteen arc lengths, 1,048,576 cases with nothing left
    // over. Each is stored in the device record's field and read back through the
    // public conversion, and must be the same number.
    BINCV_CHECK_EQ(sizeof(bc::DeviceFastCorner), size_t{12});
    size_t bad = 0;
    int lo = 1 << 30, hi = -(1 << 30);
    for (int arcLength = 1; arcLength <= 16; ++arcLength) {
        for (unsigned ring = 0; ring < 65536u; ++ring) {
            const int s = bincv::impl::fastLongestRun(ring, arcLength);
            if (s < lo) lo = s;
            if (s > hi) hi = s;
            bc::DeviceFastCorner d;
            d.x = 0;
            d.y = 0;
            d.score = s;
            if (d.toHost().score != static_cast<long long>(s)) ++bad;
        }
    }
    BINCV_CHECK_EQ(bad, size_t{0});
    // The range itself, printed as a claim rather than left implicit: a score is an
    // arc length around a 16-pixel ring, so it cannot leave [1, 16] and the 32-bit
    // field has 31 bits of headroom over the worst case.
    BINCV_CHECK_EQ(lo, 1);
    BINCV_CHECK_EQ(hi, 16);
}

// ---------------------------------------------------------------------------
// The response map and the selection
// ---------------------------------------------------------------------------

namespace {

/// The four ternary derivative planes of a binary frame, host and device.
struct Derivatives {
    TernaryMat<Word> dx;
    TernaryMat<Word> dy;
    bc::DeviceBinMat magX, magY, signX, signY;

    explicit Derivatives(const BinMat<Word>& frame)
        : dx(static_cast<int>(frame.getWidth()), static_cast<int>(frame.getHeight())),
          dy(static_cast<int>(frame.getWidth()), static_cast<int>(frame.getHeight())),
          magX(static_cast<int>(frame.getWidth()), static_cast<int>(frame.getHeight())),
          magY(static_cast<int>(frame.getWidth()), static_cast<int>(frame.getHeight())),
          signX(static_cast<int>(frame.getWidth()), static_cast<int>(frame.getHeight())),
          signY(static_cast<int>(frame.getWidth()), static_cast<int>(frame.getHeight())) {
        bincv::derivativeX<1, Word>(frame, dx);
        bincv::derivativeY<1, Word>(frame, dy);
        bc::upload(dx.constMagnitude(0), magX.view());
        bc::upload(dy.constMagnitude(0), magY.view());
        bc::upload(dx.constSign(), signX.view());
        bc::upload(dy.constSign(), signY.view());
    }
};

void checkResponseMapMatches(const BinMat<Word>& frame, int blockSize, const char* what) {
    const size_t w = frame.getWidth(), h = frame.getHeight();
    Derivatives d(frame);

    std::vector<float> host(w * h, 0.0f);
    bincv::cornerMinEigenVal<Word>(d.dx.constMagnitude(0), d.dy.constMagnitude(0),
                                   d.dx.constSign(), d.dy.constSign(), blockSize,
                                   ResponseMap(host.data(), w, h, w));

    for (int sliced = 1; sliced >= 0; --sliced) {
        bc::impl::cornerSlicedEnabled() = sliced != 0;
        bc::DeviceImage<float> dmap(static_cast<int>(w), static_cast<int>(h));
        const cudaError_t err = bc::cornerMinEigenValAsync(
            d.magX.constView(), d.magY.constView(), d.signX.constView(), d.signY.constView(),
            blockSize, dmap.view());
        BINCV_CHECK_EQ(static_cast<int>(err), static_cast<int>(cudaSuccess));
        cudaDeviceSynchronize();
        std::vector<float> dev(w * h, -1.0f);
        bc::downloadImage<float>(dmap.constView(), dev.data(), w);
        cudaStreamSynchronize(nullptr);
        const bool same = std::memcmp(dev.data(), host.data(), w * h * sizeof(float)) == 0;
        BINCV_CHECK(same);
        if (!same) {
            for (size_t i = 0; i < w * h; ++i) {
                if (dev[i] != host[i]) {
                    std::printf("  [%s] bs %d sliced %d: first mismatch at (%zu,%zu): "
                                "device %.9g host %.9g\n",
                                what, blockSize, sliced, i % w, i / w,
                                static_cast<double>(dev[i]), static_cast<double>(host[i]));
                    break;
                }
            }
        }
    }
    bc::impl::cornerSlicedEnabled() = true;
}

struct DeviceCornerRun {
    std::vector<Corner> corners;
    bc::DeviceCornerResult result;
    float frameMax = -1.0f;  ///< what the kernels reduced, read back for its own check
    cudaError_t err = cudaSuccess;
};

DeviceCornerRun runDeviceCorners(const Derivatives& d, size_t w, size_t h,
                                 const GoodFeaturesParams& params, size_t capacity,
                                 size_t candidateCapacity, bool fused) {
    DeviceCornerRun out;
    bc::impl::cornerFusedEnabled() = fused;

    bc::DeviceArray<bc::DeviceCorner> dcand(candidateCapacity);
    bc::DeviceAppendCounter counter;
    counter.reset();
    bc::DeviceArray<uint32_t> dmax(1);
    cudaMemset(dmax.data(), 0, sizeof(uint32_t));
    const size_t scratchBytes = bc::goodFeaturesScratchBytes(candidateCapacity);
    bc::DeviceArray<uint8_t> scratch(scratchBytes);
    bc::DeviceArray<bc::DeviceCorner> dout(capacity);
    bc::DeviceArray<bc::DeviceCornerResult> dres(1);
    // The frame map exists only for the reference arm; the fused arm is handed
    // an empty view, which is the arm's whole claim.
    bc::DeviceImage<float> dmap(fused ? 0 : static_cast<int>(w),
                                fused ? 0 : static_cast<int>(h));

    bc::DeviceGoodFeaturesWorkspace work;
    work.candidates = bc::DeviceCornerBuffer(dcand.data(), counter.devicePtr(),
                                             static_cast<uint32_t>(candidateCapacity));
    work.maxBits = dmax.data();
    work.scratch = scratch.data();
    work.scratchBytes = scratchBytes;
    work.frameMap = dmap.view();

    out.err = bc::goodFeaturesToTrackAsync(
        d.magX.constView(), d.magY.constView(), d.signX.constView(), d.signY.constView(), params,
        work, dout.data(), static_cast<uint32_t>(capacity), dres.data());
    cudaDeviceSynchronize();
    if (out.err != cudaSuccess) return out;
    cudaMemcpy(&out.result, dres.data(), sizeof(bc::DeviceCornerResult), cudaMemcpyDeviceToHost);
    uint32_t maxBits = 0;
    cudaMemcpy(&maxBits, dmax.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::memcpy(&out.frameMax, &maxBits, sizeof(float));

    std::vector<bc::DeviceCorner> raw(out.result.count);
    if (!raw.empty()) {
        cudaMemcpy(raw.data(), dout.data(), raw.size() * sizeof(bc::DeviceCorner),
                   cudaMemcpyDeviceToHost);
    }
    out.corners.resize(raw.size());
    bc::toHost(raw.data(), raw.size(), out.corners.data());
    return out;
}

void checkCornersMatch(const BinMat<Word>& frame, const GoodFeaturesParams& params,
                       size_t capacity, const char* what) {
    const size_t w = frame.getWidth(), h = frame.getHeight();
    Derivatives d(frame);

    std::vector<float> ring(bincv::kResponseRingRows * w, 0.0f);
    std::vector<Corner> host(capacity == 0 ? 1 : capacity);
    const CornerResult hr = bincv::goodFeaturesToTrackStreaming<Word>(
        d.dx.constMagnitude(0), d.dy.constMagnitude(0), d.dx.constSign(), d.dy.constSign(),
        params, ResponseMap(ring.data(), w, bincv::kResponseRingRows, w),
        capacity == 0 ? nullptr : host.data(), capacity);

    // Big enough for every raw 3x3 maximum the frame can have, so the device is
    // answering the same question the host is.
    const size_t candidateCapacity = w > 2 && h > 2 ? (w - 2) * (h - 2) : 1;
    // Three switches, every position, all held to the ONE host answer: the
    // candidate arm, the sort's parallelism and the spacing filter's shape. The
    // spacing pair is also where the integer distance test is proven -- the
    // reference arm keeps the host's double comparison, so agreement here is the
    // proof that `s < ceil(minDistanceSq)` is the same predicate.
    for (int sortPar = 1; sortPar >= 0; --sortPar) {
        for (int chunked = 1; chunked >= 0; --chunked) {
            bc::impl::cornerSortParallelEnabled() = sortPar != 0;
            bc::impl::cornerSpacingChunkedEnabled() = chunked != 0;
            for (int fused = 1; fused >= 0; --fused) {
                const DeviceCornerRun dev =
                    runDeviceCorners(d, w, h, params, capacity, candidateCapacity, fused != 0);
                BINCV_CHECK_EQ(static_cast<int>(dev.err), static_cast<int>(cudaSuccess));
                BINCV_CHECK_EQ(dev.result.candidateOverflow, 0u);
                BINCV_CHECK_EQ(static_cast<size_t>(dev.result.count), hr.count);
                BINCV_CHECK_EQ(static_cast<size_t>(dev.result.candidatesRanked),
                               hr.candidatesRanked);
                BINCV_CHECK_EQ(dev.result.candidatesTruncated != 0, hr.candidatesTruncated);
                if (dev.result.count != hr.count) {
                    std::printf("  [%s] fused %d: device %u corners, host %zu\n", what, fused,
                                dev.result.count, hr.count);
                    continue;
                }
                const bool same = hr.count == 0 || std::memcmp(dev.corners.data(), host.data(),
                                                               hr.count * sizeof(Corner)) == 0;
                BINCV_CHECK(same);
                if (!same) {
                    for (size_t i = 0; i < hr.count; ++i) {
                        if (dev.corners[i].x != host[i].x || dev.corners[i].y != host[i].y ||
                            dev.corners[i].response != host[i].response) {
                            std::printf("  [%s] fused %d: first mismatch at %zu: device "
                                        "(%d,%d,%.9g) host (%d,%d,%.9g)\n",
                                        what, fused, i, dev.corners[i].x, dev.corners[i].y,
                                        static_cast<double>(dev.corners[i].response), host[i].x,
                                        host[i].y, static_cast<double>(host[i].response));
                            break;
                        }
                    }
                }
            }
        }
    }
    bc::impl::cornerSortParallelEnabled() = true;
    bc::impl::cornerSpacingChunkedEnabled() = true;
    bc::impl::cornerFusedEnabled() = true;
}

} // namespace

BINCV_TEST(CudaCorner, ResponseMapMatchesHostAcrossShapes) {
    const size_t widths[] = {752, 640, 97, 65, 33, 31, 6, 1};
    const size_t heights[] = {23, 7, 1};
    uint64_t seed = 0x9001;
    for (size_t w : widths) {
        for (size_t h : heights) {
            checkResponseMapMatches(structuredBits(w, h, seed++), 3, "shape");
        }
    }
}

BINCV_TEST(CudaCorner, ResponseMapMatchesHostAcrossBlockSizes) {
    // 3 is the sliced arm; 1, 5, 7 and 15 are the window arm on both switch
    // positions, which is the gate-excluded case.
    const int blocks[] = {1, 3, 5, 7, 15};
    const BinMat<Word> frame = structuredBits(97, 29, 0x4242);
    for (int bs : blocks) checkResponseMapMatches(frame, bs, "blockSize");
}

BINCV_TEST(CudaCorner, UniformFramesHaveNoResponseAndNoCorners) {
    // corner.hpp's BorderRing case: with reflect-101 the derivative of a uniform
    // frame is exactly zero everywhere, so maxVal is 0, the threshold admits
    // nothing, and the answer is no corners at every block size.
    for (int value = 0; value <= 1; ++value) {
        const BinMat<Word> frame = constantBits(97, 29, value != 0);
        for (int bs : {1, 3, 7}) {
            checkResponseMapMatches(frame, bs, "uniform");
            GoodFeaturesParams p;
            p.blockSize = bs;
            checkCornersMatch(frame, p, 256, "uniform");
        }
    }
}

BINCV_TEST(CudaCorner, CornersMatchHostAcrossShapes) {
    const size_t widths[] = {752, 97, 65, 33, 31};
    const size_t heights[] = {37, 13};
    uint64_t seed = 0x7777;
    for (size_t w : widths) {
        for (size_t h : heights) {
            GoodFeaturesParams p;
            p.minDistance = 3.5;
            checkCornersMatch(structuredBits(w, h, seed++), p, 512, "shape");
        }
    }
}

BINCV_TEST(CudaCorner, CornersMatchHostAcrossParameters) {
    const BinMat<Word> frame = structuredBits(97, 41, 0xABCD);
    const double distances[] = {0.0, 0.5, 3.5, 33.33333333333};
    const int caps[] = {0, 1, 200, 100000};
    const double qualities[] = {0.01, 1.0, 1e-6};
    for (double dist : distances) {
        for (int cap : caps) {
            for (double q : qualities) {
                GoodFeaturesParams p;
                p.minDistance = dist;
                p.maxCorners = cap;
                p.qualityLevel = q;
                checkCornersMatch(frame, p, 512, "parameters");
            }
        }
    }
}

BINCV_TEST(CudaCorner, CornersMatchHostAtBlockSizeSeven) {
    // The window arm end to end -- response, suppression and selection -- which
    // is also the arm that has no bit-sliced fast path to be compared against.
    const BinMat<Word> frame = structuredBits(65, 29, 0x1111);
    for (int bs : {1, 3, 7}) {
        GoodFeaturesParams p;
        p.blockSize = bs;
        p.minDistance = 3.5;
        checkCornersMatch(frame, p, 256, "blockSize");
    }
}

BINCV_TEST(CudaCorner, CapacityContractHoldsWhenSurvivorsDoNotFit) {
    // The host's capacity contract: `capacity` bounds the survivors that can be
    // RANKED, and candidatesTruncated is the only way a caller learns the answer
    // is a restriction. Sweeping right through the boundary is what catches an
    // off-by-one in "the strongest capacity of S".
    const BinMat<Word> frame = structuredBits(97, 41, 0x2468);
    GoodFeaturesParams p;
    p.minDistance = 3.5;
    p.maxCorners = 0;
    const size_t caps[] = {0, 1, 2, 7, 31, 64, 1024};
    for (size_t c : caps) checkCornersMatch(frame, p, c, "capacity");
}

BINCV_TEST(CudaCorner, TiesAreBrokenTheReferencesWay) {
    // CornerStronger's "later position first" rule has to decide real output, or
    // a suite cannot tell a correct tie-break from a reversed one. A ONE-PIXEL
    // checkerboard will not do it and that is worth recording: the `[-1, 0, 1]`
    // tap reads `src(x+1) - src(x-1)`, which is identically zero there, so the
    // frame has no response at all and the case passes whatever the comparator
    // says -- measured, with a deliberately reversed tie rule.
    //
    // TWO-PIXEL BLOCKS do it: 2,640 of the ranked candidates share one response
    // (8.0), so the tie rule alone chooses every corner the spacing filter keeps.
    BinMat<Word> frame(64, 48);
    for (size_t y = 0; y < 48; ++y) {
        Word* row = frame.view().row(y);
        for (size_t x = 0; x < 64; ++x) {
            if ((((x / 2) + (y / 2)) & 1u) == 0u) row[x / 32] |= (Word{1} << (x % 32));
        }
    }
    for (double dist : {0.0, 3.0, 9.0}) {
        GoodFeaturesParams p;
        p.minDistance = dist;
        p.maxCorners = 25;
        checkCornersMatch(frame, p, 4096, "block ties");
    }
}

BINCV_TEST(CudaCorner, FrameMaximumIsTheWholeMapsMaximum) {
    // The host's maxVal is `cv::minMaxLoc` over the WHOLE map, border row and
    // column included -- NOT over the suppression interior. This pins the number
    // the device reduced against the host map's own maximum, directly.
    //
    // AND A FINDING, because the obvious sharper test does not exist. The design
    // review asked for a frame whose maximum lies on the border and whose
    // interior maximum is smaller, so the two candidate specifications would give
    // different thresholds and different corners. Searched: three million random
    // frames, widths 8-16, heights 5-10, four densities, blockSize 1, 3, 5, 7 and
    // 9 -- and NOT ONE has a whole-map maximum above its interior maximum. The
    // reason is structural rather than lucky. BORDER_REFLECT_101 makes the
    // derivative exactly zero ON the border line (`src(x+1) - src(x-1)` with
    // `src(-1) == src(1)`), and a clipped window is a subset of the neighbouring
    // interior pixel's, so both magnitude sums at a border pixel are bounded by
    // an interior neighbour's. Border responses are real -- corner.hpp's
    // BorderRing cases are right about that -- but on this derivative path they
    // appear never to be MAXIMAL, so the region the maximum is taken over is not
    // observable through the corner set. The implementation follows the host's
    // specification anyway; this check is what holds it there.
    uint64_t seed = 0xB0DE;
    for (size_t w : {97u, 64u, 33u}) {
        for (size_t h : {29u, 8u}) {
            const BinMat<Word> frame = structuredBits(w, h, seed++);
            Derivatives d(frame);
            std::vector<float> map(w * h, 0.0f);
            bincv::cornerMinEigenVal<Word>(d.dx.constMagnitude(0), d.dy.constMagnitude(0),
                                           d.dx.constSign(), d.dy.constSign(), 3,
                                           ResponseMap(map.data(), w, h, w));
            float whole = 0.0f;
            for (float v : map) whole = v > whole ? v : whole;

            GoodFeaturesParams p;
            p.minDistance = 3.5;
            const size_t candCap = (w - 2) * (h - 2);
            for (int fused = 1; fused >= 0; --fused) {
                const DeviceCornerRun dev =
                    runDeviceCorners(d, w, h, p, 512, candCap, fused != 0);
                BINCV_CHECK_EQ(static_cast<int>(dev.err), static_cast<int>(cudaSuccess));
                BINCV_CHECK(dev.frameMax == whole);
            }
            bc::impl::cornerFusedEnabled() = true;
        }
    }
}

BINCV_TEST(CudaCorner, CandidateOverflowRefusesRatherThanGuesses) {
    // The device candidate buffer is NOT the host's capacity contract. When the
    // raw 3x3 maxima overflow it, the numbers the selection would return are not
    // a restriction of the host's answer -- they are unrelated to it -- so it
    // returns nothing and says so, and the counter carries the size a re-run
    // needs.
    const BinMat<Word> frame = structuredBits(97, 41, 0x3131);
    Derivatives d(frame);
    GoodFeaturesParams p;
    p.minDistance = 3.5;
    const DeviceCornerRun dev = runDeviceCorners(d, 97, 41, p, 256, 4, /*fused=*/true);
    BINCV_CHECK_EQ(static_cast<int>(dev.err), static_cast<int>(cudaSuccess));
    BINCV_CHECK_EQ(dev.result.candidateOverflow, 1u);
    BINCV_CHECK_EQ(dev.result.count, 0u);
    BINCV_CHECK_EQ(dev.result.candidatesRanked, 0u);
    bc::impl::cornerFusedEnabled() = true;
}

BINCV_TEST(CudaCorner, RefusesOutsideItsDocumentedDomain) {
#if !BINCV_DEBUG_CHECKS
    const BinMat<Word> frame = structuredBits(64, 16, 5);
    Derivatives d(frame);
    bc::DeviceImage<float> dmap(64, 16);
    BINCV_CHECK_EQ(static_cast<int>(bc::cornerMinEigenValAsync(
                       d.magX.constView(), d.magY.constView(), d.signX.constView(),
                       d.signY.constView(), 0, dmap.view())),
                   static_cast<int>(cudaErrorInvalidValue));
    bc::DeviceImage<float> wrong(32, 16);
    BINCV_CHECK_EQ(static_cast<int>(bc::cornerMinEigenValAsync(
                       d.magX.constView(), d.magY.constView(), d.signX.constView(),
                       d.signY.constView(), 3, wrong.view())),
                   static_cast<int>(cudaErrorInvalidValue));

    // The reference arm needs the frame map the fused arm exists not to have,
    // and says so rather than writing through a null pointer.
    bc::impl::cornerFusedEnabled() = false;
    bc::DeviceArray<bc::DeviceCorner> dcand(64);
    bc::DeviceAppendCounter counter;
    counter.reset();
    bc::DeviceArray<uint32_t> dmax(1);
    bc::DeviceArray<uint8_t> scratch(bc::goodFeaturesScratchBytes(64));
    bc::DeviceArray<bc::DeviceCorner> dout(16);
    bc::DeviceArray<bc::DeviceCornerResult> dres(1);
    bc::DeviceGoodFeaturesWorkspace work;
    work.candidates = bc::DeviceCornerBuffer(dcand.data(), counter.devicePtr(), 64u);
    work.maxBits = dmax.data();
    work.scratch = scratch.data();
    work.scratchBytes = bc::goodFeaturesScratchBytes(64);
    BINCV_CHECK_EQ(static_cast<int>(bc::goodFeaturesToTrackAsync(
                       d.magX.constView(), d.magY.constView(), d.signX.constView(),
                       d.signY.constView(), GoodFeaturesParams(), work, dout.data(), 16u,
                       dres.data())),
                   static_cast<int>(cudaErrorInvalidValue));
    // And a scratch buffer shorter than the sizing function says.
    bc::impl::cornerFusedEnabled() = true;
    work.scratchBytes = bc::goodFeaturesScratchBytes(64) - 1;
    BINCV_CHECK_EQ(static_cast<int>(bc::goodFeaturesToTrackAsync(
                       d.magX.constView(), d.magY.constView(), d.signX.constView(),
                       d.signY.constView(), GoodFeaturesParams(), work, dout.data(), 16u,
                       dres.data())),
                   static_cast<int>(cudaErrorInvalidValue));
    BINCV_CHECK_EQ(static_cast<int>(cudaGetLastError()), static_cast<int>(cudaSuccess));

    // A frame wider than the ordering key's 16 bits per axis. Refused, not
    // wrapped -- the key packs `CornerStronger`'s tie rule into those bits, and
    // a wrapped coordinate would order corners plausibly and wrongly.
    bc::DeviceBinMat wideFrame(65537, 1);
    work.scratchBytes = bc::goodFeaturesScratchBytes(64);
    BINCV_CHECK_EQ(static_cast<int>(bc::goodFeaturesToTrackAsync(
                       wideFrame.constView(), wideFrame.constView(), wideFrame.constView(),
                       wideFrame.constView(), GoodFeaturesParams(), work, dout.data(), 16u,
                       dres.data())),
                   static_cast<int>(cudaErrorInvalidValue));
    BINCV_CHECK_EQ(static_cast<int>(cudaGetLastError()), static_cast<int>(cudaSuccess));
#endif
}

// ---------------------------------------------------------------------------
// cornerSubPix
// ---------------------------------------------------------------------------

namespace {

void checkSubPixMatches(const BinMat<Word>& frame, const SubPixParams& params,
                        const std::vector<Point2f>& seeds, const char* what) {
    Derivatives d(frame);

    std::vector<Point2f> host = seeds;
    const SubPixResult hr =
        bincv::cornerSubPix<1, Word>(d.dx, d.dy, host.data(), host.size(), params);

    bc::DeviceSubPixMask mask(params);
    for (int skip = 1; skip >= 0; --skip) {
        bc::impl::subPixSpreadEnabled() = skip != 0;
        bc::impl::subPixSkipEnabled() = skip != 0;
        bc::DeviceArray<float> dxy(seeds.size() * 2);
        cudaMemcpy(dxy.data(), seeds.data(), seeds.size() * sizeof(Point2f),
                   cudaMemcpyHostToDevice);
        bc::DeviceArray<bc::DeviceSubPixResult> dres(1);
        cudaMemset(dres.data(), 0, sizeof(bc::DeviceSubPixResult));

        const cudaError_t err = bc::cornerSubPixAsync(
            d.magX.constView(), d.magY.constView(), d.signX.constView(), d.signY.constView(),
            dxy.data(), static_cast<uint32_t>(seeds.size()), params, mask.devicePtr(),
            dres.data());
        BINCV_CHECK_EQ(static_cast<int>(err), static_cast<int>(cudaSuccess));
        cudaDeviceSynchronize();

        std::vector<Point2f> dev(seeds.size());
        cudaMemcpy(dev.data(), dxy.data(), dev.size() * sizeof(Point2f), cudaMemcpyDeviceToHost);
        bc::DeviceSubPixResult dr;
        cudaMemcpy(&dr, dres.data(), sizeof(dr), cudaMemcpyDeviceToHost);

        // Exact, not near: a tolerance fitted to the observed value cannot fail.
        const bool same = seeds.empty() ||
                          std::memcmp(dev.data(), host.data(), dev.size() * sizeof(Point2f)) == 0;
        BINCV_CHECK(same);
        if (!same) {
            for (size_t i = 0; i < dev.size(); ++i) {
                if (dev[i].x != host[i].x || dev[i].y != host[i].y) {
                    std::printf("  [%s] skip %d: first mismatch at %zu: device (%.9g,%.9g) "
                                "host (%.9g,%.9g)\n",
                                what, skip, i, static_cast<double>(dev[i].x),
                                static_cast<double>(dev[i].y), static_cast<double>(host[i].x),
                                static_cast<double>(host[i].y));
                    break;
                }
            }
        }
        BINCV_CHECK_EQ(static_cast<size_t>(dr.refined), hr.refined);
        BINCV_CHECK_EQ(static_cast<size_t>(dr.singular), hr.singular);
        BINCV_CHECK_EQ(static_cast<size_t>(dr.clamped), hr.clamped);
        BINCV_CHECK_EQ(static_cast<size_t>(dr.diverged), hr.diverged);
    }
    bc::impl::subPixSkipEnabled() = true;
    bc::impl::subPixSpreadEnabled() = true;
}

std::vector<Point2f> gridSeeds(size_t w, size_t h, int step) {
    std::vector<Point2f> seeds;
    for (size_t y = 0; y < h; y += static_cast<size_t>(step)) {
        for (size_t x = 0; x < w; x += static_cast<size_t>(step)) {
            Point2f p;
            p.x = static_cast<float>(x);
            p.y = static_cast<float>(y);
            seeds.push_back(p);
        }
    }
    return seeds;
}

} // namespace

BINCV_TEST(CudaSubPix, MatchesHostAcrossWindowSizes) {
    const BinMat<Word> frame = structuredBits(97, 61, 0xFACE);
    const std::vector<Point2f> seeds = gridSeeds(97, 61, 9);
    for (int winHalf : {1, 2, 5, 15}) {
        SubPixParams p;
        p.winHalf = winHalf;
        checkSubPixMatches(frame, p, seeds, "winHalf");
    }
}

BINCV_TEST(CudaSubPix, MatchesHostAcrossTerminationAndZeroZone) {
    const BinMat<Word> frame = structuredBits(97, 61, 0x0BAD);
    const std::vector<Point2f> seeds = gridSeeds(97, 61, 11);
    const int zeroHalves[] = {-1, 0, 2};
    const int iters[] = {1, 3, 40};
    const double epsilons[] = {0.001, 0.0, 1.0};
    for (int z : zeroHalves) {
        for (int it : iters) {
            for (double e : epsilons) {
                SubPixParams p;
                p.zeroHalf = z;
                p.maxIterations = it;
                p.epsilon = e;
                checkSubPixMatches(frame, p, seeds, "termination");
            }
        }
    }
}

BINCV_TEST(CudaSubPix, MatchesHostOnAllOnesWindowsWhereTheSkipSavesNothing) {
    // The gate-excluded control for the skip: with every magnitude bit set the
    // skip can eliminate nothing, so the two arms run the same work. Here that is
    // a correctness case; the benchmark is where it must read ~1.00x.
    const BinMat<Word> frame = constantBits(97, 61, true);
    const std::vector<Point2f> seeds = gridSeeds(97, 61, 7);
    SubPixParams p;
    checkSubPixMatches(frame, p, seeds, "dense windows");
}

BINCV_TEST(CudaSubPix, MatchesHostOnBordersAndEmptySets) {
    const BinMat<Word> frame = structuredBits(65, 33, 0x5A5A);
    // Seeds whose window leaves the image: the host counts them `clamped` and
    // leaves them where they were, and so must this.
    std::vector<Point2f> seeds;
    const float xs[] = {0.0f, 1.0f, 4.0f, 32.0f, 60.0f, 64.0f};
    const float ys[] = {0.0f, 2.0f, 16.0f, 30.0f, 32.0f};
    for (float x : xs) {
        for (float y : ys) {
            Point2f p;
            p.x = x;
            p.y = y;
            seeds.push_back(p);
        }
    }
    SubPixParams p;
    checkSubPixMatches(frame, p, seeds, "borders");
    checkSubPixMatches(frame, p, std::vector<Point2f>(), "empty set");
}

BINCV_TEST(CudaSubPix, RefusesOutsideItsDocumentedDomain) {
#if !BINCV_DEBUG_CHECKS
    const BinMat<Word> frame = structuredBits(64, 32, 9);
    Derivatives d(frame);
    SubPixParams good;
    bc::DeviceSubPixMask mask(good);
    bc::DeviceArray<float> dxy(8);
    bc::DeviceArray<bc::DeviceSubPixResult> dres(1);

    SubPixParams tooBig;
    tooBig.winHalf = bincv::impl::kMaxWinHalf + 1;
    BINCV_CHECK_EQ(static_cast<int>(bc::cornerSubPixAsync(
                       d.magX.constView(), d.magY.constView(), d.signX.constView(),
                       d.signY.constView(), dxy.data(), 4u, tooBig, mask.devicePtr(),
                       dres.data())),
                   static_cast<int>(cudaErrorInvalidValue));
    SubPixParams tooSmall;
    tooSmall.winHalf = 0;
    BINCV_CHECK_EQ(static_cast<int>(bc::cornerSubPixAsync(
                       d.magX.constView(), d.magY.constView(), d.signX.constView(),
                       d.signY.constView(), dxy.data(), 4u, tooSmall, mask.devicePtr(),
                       dres.data())),
                   static_cast<int>(cudaErrorInvalidValue));
    BINCV_CHECK_EQ(static_cast<int>(bc::cornerSubPixAsync(
                       d.magX.constView(), d.magY.constView(), d.signX.constView(),
                       d.signY.constView(), nullptr, 4u, good, mask.devicePtr(), dres.data())),
                   static_cast<int>(cudaErrorInvalidValue));
    BINCV_CHECK_EQ(static_cast<int>(cudaGetLastError()), static_cast<int>(cudaSuccess));
#endif
}

BINCV_TEST(CudaSubPix, TheMaskIsTheHostsOwn) {
    // CUDA's exp(double) is documented at 1-2 ulp where glibc's is about 0.5, so
    // the weights must come from the host's builder rather than be recomputed.
    // This is the standing check that the container did not quietly stop doing
    // that: the uploaded doubles are compared bit for bit against
    // impl::subPixMask's output.
    for (int winHalf : {1, 5, 15}) {
        for (int zeroHalf : {-1, 2}) {
            SubPixParams p;
            p.winHalf = winHalf;
            p.zeroHalf = zeroHalf;
            bc::DeviceSubPixMask mask(p);
            const size_t side = static_cast<size_t>(2 * winHalf + 1);
            BINCV_CHECK_EQ(mask.size(), side * side);
            std::vector<double> host(side * side, 0.0);
            bincv::impl::subPixMask(winHalf, zeroHalf, host.data());
            std::vector<double> dev(side * side, -1.0);
            cudaMemcpy(dev.data(), mask.devicePtr(), dev.size() * sizeof(double),
                       cudaMemcpyDeviceToHost);
            BINCV_CHECK(std::memcmp(dev.data(), host.data(), dev.size() * sizeof(double)) == 0);
        }
    }
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
    const int summaryRc = ::bincv::test::summarize("CUDA pipeline corner tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA pipeline corner tests", argc, argv);
}
#endif
