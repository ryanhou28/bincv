// denoise -- the reference pipeline's three-pixel median -- against OpenCV,
// and against binCV's own composed spelling.
//
// THE DENOMINATOR (the design notes, CLAUDE.md): OpenCV performing the SAME
// SEMANTIC OPERATION on the SAME binary content stored as CV_8U. For this
// operation that denominator is not a judgement call -- it is
// the reference frontend's denoiser, `three_pix_median_filter`, ported
// call for call, because that IS what the pipeline runs today without binCV.
// The two `cv::Mat::zeros` neighbour matrices and the two range-limited copyTo
// calls are part of the work, not setup: they are how that implementation
// obtains its neighbours, and a version that skipped them would be a different
// (and cheaper) operation being passed off as the baseline.
//
// THREE IMPLEMENTATIONS, and the third is the point of the file:
//
// OpenCV CV_8U the reference above, one byte per pixel.
// binCV fused ops/denoise.hpp -- one pass, NO scratch.
// binCV composed shiftDown + shiftLeft + majority3 -- three passes and TWO
// FRAME-SIZED SCRATCH BUFFERS the caller must own.
//
// The fused and composed rows compute the same image (checked before anything is
// timed) and differ only in traversals and working set, so they are the evidence
// behind ops/denoise.hpp's claim that fusing was free rather than a trade. binCV
// has no rule that says "always fuse"; it has a rule that says memory wins when
// memory and speed conflict, and this pair of rows is how one finds out whether
// they conflict here.
//
// FOOTPRINT IS REPORTED ALONGSIDE EVERY TIMING, per CLAUDE.md ("Report memory
// and speed together -- they trade off, so one alone cannot be weighed against
// goals that conflict"), as the WORKING SET OF ONE CALL rather than as a
// per-buffer ratio.
//
// ---------------------------------------------------------------------------
// MEASUREMENT VALIDITY -- the four hazards the other benchmarks in this
// directory enumerate, answered here in the same way:
//
// 1. DEAD CODE. Every timed body writes to memory the next iteration reads, and
// feeds one word of its destination to a volatile sink so the store cannot be
// sunk out of the loop. THAT WORD IS NOT THE VALIDITY ARGUMENT -- a kernel
// that computed only the first word would satisfy it. What does: after the
// timing, every implementation is run once more on image 0 and its
// destination is folded PIXEL BY PIXEL into a checksum printed in the table.
// All rows must print the same number, because they compute the same image
// in four different containers, and a row that stopped computing most of its
// output prints a different one. (An earlier version of this file printed
// only the volatile sink and claimed it was a checksum of each destination.
// It was one word.)
// 2. CONSTANT FOLDING. Four distinct random images are rotated through, so no
// iteration can reuse the previous one's answer.
// 3. CALIBRATED BATCHES. A fixed iteration count measures clock resolution at
// 94x60; every case runs enough iterations to fill a target millisecond
// budget, and the reported figure is the minimum over several batches.
// 4. THE SIDES MUST AGREE. All five implementations are compared pixel for
// pixel BEFORE anything is timed, and a disagreement skips the size and
// exits non-zero rather than printing a table under a warning.
// 5. THE BASELINE'S FIXED PER-CALL COST IS MEASURED, NOT ASSUMED. The reference
// implementation makes EIGHT cv:: calls per frame (two setTo, two copyTo,
// four min/max), each of which pays a size-independent dispatch cost. At
// 640x480 that is noise; at 94x60 it is most of the frame time, so the
// ns/pixel ladder is NOT a pure statement about the operation and the small
// sizes' ratios are not comparable with the large ones. The floor is
// measured directly (the same eight calls on a 2x2 frame) and printed beside
// every size, so a reader can subtract it instead of taking the ladder's
// shape as a cache result. binCV's own floor is measured the same way, for
// the same reason and to keep the comparison symmetric.
//
// ---------------------------------------------------------------------------
// WHERE THIS IS AUTHORITATIVE
//
// On x86_64 it is INDICATIVE ONLY (EXPERIMENTS.md, "Measurement platforms"). The
// numbers that belong in a claim come from the reference device:
//
// BINCV_PI_OPENCV=1./scripts/run_on_pi.sh <target>
// './benchmark/denoise_benchmark > denoise_benchmark.log'
//
// BINCV_PI_OPENCV=1 is required: the denominator is an OpenCV call, and the
// device's default build is core-only.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "bench_util.hpp"
#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/bitslice.hpp"
#include "bincv-cpp/ops/denoise.hpp"
#include "bincv-cpp/ops/shift.hpp"

namespace {

constexpr int kInputs = 4;      // distinct images, rotated through
constexpr int kRepeats = 5;     // batches per case; the minimum is reported
constexpr double kTargetMs = 40.0;

volatile uint64_t g_sink = 0;

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/// @brief One image in every representation under test, from ONE draw per pixel.
/// @note The packed matrices and the CV_8U mask are not merely statistically
/// similar -- they are the same picture, which is what makes the comparison
/// like for like (the design notes: the same binary content).
struct Image {
    bincv::BinMat<uint32_t> packed32;
    bincv::BinMat<uint64_t> packed64;
    cv::Mat mask;
};

void makeImage(Image& out, int width, int height, uint64_t seed) {
    out.packed32 = bincv::BinMat<uint32_t>(width, height);
    out.packed64 = bincv::BinMat<uint64_t>(width, height);
    out.mask = cv::Mat::zeros(height, width, CV_8U);

    uint64_t state = seed;
    for (int y = 0; y < height; ++y) {
        uint8_t* row = out.mask.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            if ((nextRandom(state) >> 40) < (UINT64_C(1) << 23)) {  // ~50% fill
                out.packed32.set(y, x, true);
                out.packed64.set(y, x, true);
                row[x] = 255;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The denominator: the reference implementation, ported call for call
// ---------------------------------------------------------------------------

/// @brief The reference frontend's three-pixel median, in two spellings: the denominator, and
/// the reference exactly as written.
/// @note WHAT IS HOISTED, PRECISELY. The reference constructs SEVEN cv::Mats per
/// call with `cv::Mat::zeros`, i.e. seven allocations and six zero-fills
/// (the seventh is the input). Both variants below hoist all seven
/// ALLOCATIONS, as a caller in a frame loop would -- timing malloc would
/// flatter binCV, which allocates nothing at all here. They differ in the
/// zero-fills:
///
/// openCvMedian3 re-zeroes the two NEIGHBOUR matrices only.
/// The other four (minAB, maxAB, minMax, out)
/// are fully overwritten by the cv::min/cv::max
/// that follows, so their zero-fill is dead
/// work no caller would keep. THIS IS THE
/// DENOMINATOR.
/// openCvMedian3AsWritten re-zeroes all six, which is what the
/// reference actually costs today.
///
/// Both are timed and both appear in the table. The ratio quoted anywhere
/// else is against the FASTER of the two, so it is conservative by exactly
/// the gap the table prints -- rather than resting on a memset the
/// reference pays and a competent port would not. The two neighbour
/// matrices' zero-fill is NOT optional in either: `right`'s last column and
/// `above`'s first row are never written by the copyTo calls, and those
/// zeros are the border (ops/denoise.hpp).
struct OpenCvScratch {
    cv::Mat right, above, minAB, maxAB, minMax, out;

    void resize(int width, int height) {
        right = cv::Mat::zeros(height, width, CV_8U);
        above = cv::Mat::zeros(height, width, CV_8U);
        minAB = cv::Mat::zeros(height, width, CV_8U);
        maxAB = cv::Mat::zeros(height, width, CV_8U);
        minMax = cv::Mat::zeros(height, width, CV_8U);
        out = cv::Mat::zeros(height, width, CV_8U);
    }
};

void openCvMedian3(const cv::Mat& img, OpenCvScratch& s) {
    s.right.setTo(cv::Scalar(0));
    s.above.setTo(cv::Scalar(0));
    if (img.cols > 1) img.colRange(1, img.cols).copyTo(s.right.colRange(0, img.cols - 1));
    if (img.rows > 1) img.rowRange(0, img.rows - 1).copyTo(s.above.rowRange(1, img.rows));

    // | | p1 | |
    // | | p2 | p3 |
    // Median = max(min(p1, p2), min(max(p1, p2), p3))
    cv::min(s.above, img, s.minAB);
    cv::max(s.above, img, s.maxAB);
    cv::min(s.maxAB, s.right, s.minMax);
    cv::max(s.minAB, s.minMax, s.out);
}

/// @brief The same filter with all six `cv::Mat::zeros` fills the reference pays,
/// not just the two the border needs. See the note above OpenCvScratch.
void openCvMedian3AsWritten(const cv::Mat& img, OpenCvScratch& s) {
    s.right.setTo(cv::Scalar(0));
    s.above.setTo(cv::Scalar(0));
    s.minAB.setTo(cv::Scalar(0));
    s.maxAB.setTo(cv::Scalar(0));
    s.minMax.setTo(cv::Scalar(0));
    s.out.setTo(cv::Scalar(0));
    if (img.cols > 1) img.colRange(1, img.cols).copyTo(s.right.colRange(0, img.cols - 1));
    if (img.rows > 1) img.rowRange(0, img.rows - 1).copyTo(s.above.rowRange(1, img.rows));

    cv::min(s.above, img, s.minAB);
    cv::max(s.above, img, s.maxAB);
    cv::min(s.maxAB, s.right, s.minMax);
    cv::max(s.minAB, s.minMax, s.out);
}

// ---------------------------------------------------------------------------
// Timing (duplicated from reduce_benchmark.cpp -- see the note there on why the
// published measurement code is not refactored underneath a recorded result)
// ---------------------------------------------------------------------------

template <typename Body>
int calibrate(Body body, double targetMs) {
    int iterations = 8;
    for (int attempt = 0; attempt < 24; ++attempt) {
        const auto start = bench::Clock::now();
        for (int i = 0; i < iterations; ++i) body(i);
        const double ms =
            std::chrono::duration<double, std::milli>(bench::Clock::now() - start).count();
        if (ms >= targetMs || iterations >= (1 << 22)) return iterations;
        const double scale = (ms > 0.0) ? (targetMs / ms) : 8.0;
        const double next = static_cast<double>(iterations) * std::min(scale * 1.3, 16.0);
        iterations = static_cast<int>(next) + 1;
    }
    return iterations;
}

template <typename Body>
double measureNs(Body body, int repeats, double targetMs) {
    const int iterations = calibrate(body, targetMs);
    double bestNs = -1.0;
    for (int r = 0; r < repeats; ++r) {
        const auto start = bench::Clock::now();
        for (int i = 0; i < iterations; ++i) body(i);
        const double ns =
            std::chrono::duration<double, std::nano>(bench::Clock::now() - start).count() /
            static_cast<double>(iterations);
        if (bestNs < 0.0 || ns < bestNs) bestNs = ns;
    }
    return bestNs;
}

struct Row {
    std::string impl;
    double nsPerPixel = 0.0;
    double workingSetBytes = 0.0;   // one call's live buffers
    int traversals = 0;             // passes over the frame
    uint64_t checksum = 0;          // hazard 1: a fold over EVERY destination pixel
    bool isDenominator = false;
};

std::vector<Row> g_rows;

// ---------------------------------------------------------------------------
// Hazard 1: a fold over every destination PIXEL, not over one word
// ---------------------------------------------------------------------------
//
// Representation-independent on purpose: it reads pixels, so the packed and the
// CV_8U destinations of the same image produce the SAME number and the table's
// checksum column is an agreement check as well as a dead-code check. It runs
// once per case, outside every timed region.

uint64_t foldPixel(uint64_t h, bool bit) {
    h ^= static_cast<uint64_t>(bit ? 1u : 0u);
    return h * UINT64_C(0x100000001B3);
}

template <typename WordType>
uint64_t pixelChecksum(const bincv::BinMat<WordType>& m) {
    uint64_t h = UINT64_C(0xCBF29CE484222325);
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) h = foldPixel(h, m.at(y, x));
    }
    return h;
}

uint64_t pixelChecksum(const cv::Mat& m) {
    uint64_t h = UINT64_C(0xCBF29CE484222325);
    for (int y = 0; y < m.rows; ++y) {
        const uint8_t* row = m.ptr<uint8_t>(y);
        for (int x = 0; x < m.cols; ++x) h = foldPixel(h, row[x] != 0);
    }
    return h;
}

// ---------------------------------------------------------------------------
// Agreement, before anything is timed
// ---------------------------------------------------------------------------

template <typename WordType>
int comparePixels(const bincv::BinMat<WordType>& packed, const cv::Mat& mask) {
    int differing = 0;
    for (int y = 0; y < packed.rows(); ++y) {
        const uint8_t* row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < packed.cols(); ++x) {
            if (packed.at(y, x) != (row[x] != 0)) ++differing;
        }
    }
    return differing;
}

// ---------------------------------------------------------------------------
// Hazard 5: the size-independent cost of one call, measured
// ---------------------------------------------------------------------------
//
// WHY THIS EXISTS. The reference implementation makes EIGHT cv:: calls per frame.
// Each pays a dispatch cost that does not depend on the frame -- argument
// checking, type dispatch, the parallel_for decision -- and a ns/PIXEL figure
// divides that fixed cost by the pixel count, so it grows without limit as the
// frame shrinks. That is visible in the numbers rather than theoretical: down the
// pyramid ladder the baseline's ns/pixel RISES while its working set FALLS, which
// no cache explanation predicts.
//
// So it is measured directly: the same eight calls on a 2x2 frame, where the work
// proportional to the image is four bytes and everything left is the floor.
// binCV's kernel is measured the same way, and its floor is a function call.
//
// A 2x2 frame is deliberately not 1x1: the reference's two copyTo calls are
// guarded on cols > 1 / rows > 1 (an empty cv::Range is a cv::Mat assertion, not
// a no-op), and a 1x1 frame would skip two of the eight calls.

double g_openCvCallFloorUs = 0.0;
double g_binCvCallFloorUs = 0.0;

void measureCallFloors() {
    Image tiny;
    makeImage(tiny, 2, 2, UINT64_C(0xF100));
    OpenCvScratch s;
    s.resize(2, 2);
    bincv::BinMat<uint32_t> tinyDst(2, 2);

    g_openCvCallFloorUs = measureNs(
                              [&](int) {
                                  openCvMedian3(tiny.mask, s);
                                  g_sink += s.out.ptr<uint8_t>(0)[0];
                              },
                              kRepeats, kTargetMs) /
                          1000.0;
    g_binCvCallFloorUs = measureNs(
                             [&](int) {
                                 bincv::denoiseMedian3(tiny.packed32.constView(), tinyDst.view());
                                 g_sink += tinyDst.ptr(0)[0];
                             },
                             kRepeats, kTargetMs) /
                         1000.0;

    std::printf("\nFIXED PER-CALL COST (hazard 5), measured on a 2x2 frame -- the part of every\n"
                "row below that is not the operation:\n");
    std::printf(" OpenCV reference (8 cv:: calls): %8.3f us per call\n", g_openCvCallFloorUs);
    std::printf(" binCV denoiseMedian3: %8.3f us per call\n", g_binCvCallFloorUs);
    std::printf(" Divided by a frame's pixel count this is what a small frame's ns/pixel is\n"
                " mostly made of. It is NOT subtracted from the tables; it is printed beside\n"
                " them so the ladder is read with it.\n");
}

// ---------------------------------------------------------------------------
// One size
// ---------------------------------------------------------------------------

bool runSize(int width, int height) {
    const double pixels = static_cast<double>(width) * static_cast<double>(height);
    std::printf("\n================ %d x %d (%.0f pixels) ================\n", width, height,
                pixels);

    std::vector<Image> images(static_cast<size_t>(kInputs));
    for (int i = 0; i < kInputs; ++i) {
        makeImage(images[static_cast<size_t>(i)], width, height,
                  UINT64_C(0xDE0125E) + static_cast<uint64_t>(i) * UINT64_C(7919));
    }

    bincv::BinMat<uint32_t> dst32(width, height);
    bincv::BinMat<uint64_t> dst64(width, height);
    bincv::BinMat<uint32_t> composed32(width, height);
    bincv::BinMat<uint32_t> scratchAbove(width, height);
    bincv::BinMat<uint32_t> scratchRight(width, height);
    OpenCvScratch cvScratch;
    cvScratch.resize(width, height);

    const double packed32Bytes = static_cast<double>(dst32.sizeInWords() * sizeof(uint32_t));
    const double packed64Bytes = static_cast<double>(dst64.sizeInWords() * sizeof(uint64_t));

    // --- footprint, reported before the timings rather than after -----------
    //
    // The OpenCV column counts what the reference implementation actually holds
    // live: the input, two neighbour matrices and four result matrices. The
    // 4-buffer figure beside it is what a caller who reused temporaries could get
    // away with, so the comparison is not resting on the reference being
    // wasteful -- binCV's 2 buffers beat even that.
    std::printf(" one frame: binCV uint32 %.0f B, binCV uint64 %.0f B, OpenCV CV_8U %.0f B "
                "(%.1fx)\n",
                packed32Bytes, packed64Bytes, pixels, pixels / packed32Bytes);
    std::printf(" WORKING SET of one call (the number CLAUDE.md asks for):\n");
    std::printf(" binCV fused %10.0f B (src + dst, no scratch)\n",
                2.0 * packed32Bytes);
    std::printf(" binCV composed %10.0f B (src + dst + 2 scratch frames)\n",
                4.0 * packed32Bytes);
    std::printf(" OpenCV reference %10.0f B (7 CV_8U frames, as written)\n", 7.0 * pixels);
    std::printf(" OpenCV minimal %10.0f B (4 CV_8U frames, temporaries reused)\n",
                4.0 * pixels);

    // --- hazard 4: the sides must agree before anything is timed -------------
    {
        const Image& img = images[0];
        // The as-written spelling first, so `cvScratch.out` below is the
        // denominator's own answer rather than its predecessor's.
        openCvMedian3AsWritten(img.mask, cvScratch);
        const uint64_t asWrittenSum = pixelChecksum(cvScratch.out);
        openCvMedian3(img.mask, cvScratch);
        if (pixelChecksum(cvScratch.out) != asWrittenSum) {
            std::printf(" THE TWO OpenCV SPELLINGS DISAGREE -- SKIPPING THIS SIZE.\n"
                        " Dropping the four dead zero-fills changed the image, which means one\n"
                        " of the four temporaries was NOT fully overwritten after all.\n");
            return false;
        }
        bincv::denoiseMedian3(img.packed32.constView(), dst32.view());
        bincv::denoiseMedian3(img.packed64.constView(), dst64.view());
        bincv::shiftDown(img.packed32.constView(), scratchAbove.view(), 1);
        bincv::shiftLeft(img.packed32.constView(), scratchRight.view(), 1);
        bincv::majority3(scratchAbove.constView(), img.packed32.constView(),
                         scratchRight.constView(), composed32.view());

        const int d32 = comparePixels(dst32, cvScratch.out);
        const int d64 = comparePixels(dst64, cvScratch.out);
        const int dComposed = comparePixels(composed32, cvScratch.out);
        if (d32 != 0 || d64 != 0 || dComposed != 0) {
            std::printf(" RESULTS DISAGREE (%d / %d / %d pixels) -- SKIPPING THIS SIZE.\n"
                        " The implementations do not compute the same image, so no timing of\n"
                        " them is a comparison.\n",
                        d32, d64, dComposed);
            return false;
        }
    }

    const size_t firstRow = g_rows.size();

    // --- OpenCV: the reference, and the denominator -------------------------
    //
    // THE DENOMINATOR IS THE LEANER OF THE TWO OpenCV ROWS (the one that does not
    // re-zero the four fully-overwritten temporaries), so every ratio below is
    // conservative by the gap between them. See the note above OpenCvScratch.
    {
        const double ns = measureNs(
            [&](int i) {
                openCvMedian3(images[static_cast<size_t>(i % kInputs)].mask, cvScratch);
                g_sink += cvScratch.out.ptr<uint8_t>(0)[0];
            },
            kRepeats, kTargetMs);
        openCvMedian3(images[0].mask, cvScratch);
        Row row{"OpenCV CV_8U", ns / pixels, 7.0 * pixels, 7, pixelChecksum(cvScratch.out), true};
        g_rows.push_back(row);
    }
    {
        const double ns = measureNs(
            [&](int i) {
                openCvMedian3AsWritten(images[static_cast<size_t>(i % kInputs)].mask, cvScratch);
                g_sink += cvScratch.out.ptr<uint8_t>(0)[0];
            },
            kRepeats, kTargetMs);
        openCvMedian3AsWritten(images[0].mask, cvScratch);
        g_rows.push_back(
            Row{"OpenCV as-written", ns / pixels, 7.0 * pixels, 7, pixelChecksum(cvScratch.out), false});
    }

    // --- binCV, fused, at both word widths -----------------------------------
    {
        const double ns = measureNs(
            [&](int i) {
                bincv::denoiseMedian3(images[static_cast<size_t>(i % kInputs)].packed32.constView(),
                                      dst32.view());
                g_sink += dst32.ptr(0)[0];
            },
            kRepeats, kTargetMs);
        bincv::denoiseMedian3(images[0].packed32.constView(), dst32.view());
        g_rows.push_back(
            Row{"binCV fused u32", ns / pixels, 2.0 * packed32Bytes, 1, pixelChecksum(dst32), false});
    }
    {
        const double ns = measureNs(
            [&](int i) {
                bincv::denoiseMedian3(images[static_cast<size_t>(i % kInputs)].packed64.constView(),
                                      dst64.view());
                g_sink += dst64.ptr(0)[0];
            },
            kRepeats, kTargetMs);
        bincv::denoiseMedian3(images[0].packed64.constView(), dst64.view());
        g_rows.push_back(
            Row{"binCV fused u64", ns / pixels, 2.0 * packed64Bytes, 1, pixelChecksum(dst64), false});
    }

    // --- binCV, composed: the spelling the fused kernel replaced -------------
    {
        const double ns = measureNs(
            [&](int i) {
                const bincv::BinMat<uint32_t>& src = images[static_cast<size_t>(i % kInputs)].packed32;
                bincv::shiftDown(src.constView(), scratchAbove.view(), 1);
                bincv::shiftLeft(src.constView(), scratchRight.view(), 1);
                bincv::majority3(scratchAbove.constView(), src.constView(),
                                 scratchRight.constView(), composed32.view());
                g_sink += composed32.ptr(0)[0];
            },
            kRepeats, kTargetMs);
        bincv::shiftDown(images[0].packed32.constView(), scratchAbove.view(), 1);
        bincv::shiftLeft(images[0].packed32.constView(), scratchRight.view(), 1);
        bincv::majority3(scratchAbove.constView(), images[0].packed32.constView(),
                         scratchRight.constView(), composed32.view());
        g_rows.push_back(Row{"binCV composed u32", ns / pixels, 4.0 * packed32Bytes, 3,
                             pixelChecksum(composed32), false});
    }

    // --- the table -----------------------------------------------------------
    double reference = 0.0;
    for (size_t i = firstRow; i < g_rows.size(); ++i) {
        if (g_rows[i].isDenominator) reference = g_rows[i].nsPerPixel;
    }

    std::printf("\n %-20s %12s %10s %14s %8s %20s\n", "IMPLEMENTATION", "ns/pixel", "vs OpenCV",
                "working set", "passes", "checksum");
    std::printf(" ------------------------------------------------------------------------"
                "----------------\n");
    for (size_t i = firstRow; i < g_rows.size(); ++i) {
        const Row& row = g_rows[i];
        std::printf(" %-20s %12.5f %9.2fx %12.0f B %8d %20llu\n", row.impl.c_str(),
                    row.nsPerPixel, (row.nsPerPixel > 0.0) ? reference / row.nsPerPixel : 0.0,
                    row.workingSetBytes, row.traversals,
                    static_cast<unsigned long long>(row.checksum));
    }
    std::printf(" ratio > 1 means binCV is faster. All checksums must be EQUAL (hazard 1).\n");

    // --- hazard 5: how much of the denominator is not the operation ----------
    const double baselineUs = reference * pixels / 1000.0;
    const double binCvUs = g_rows[firstRow + 2].nsPerPixel * pixels / 1000.0;
    std::printf(" PER-FRAME: OpenCV %.2f us, of which %.2f us is the measured fixed per-call\n"
                " cost (%.0f%%). binCV fused u32 %.2f us, fixed cost %.2f us (%.0f%%).\n"
                " SUBTRACT BEFORE COMPARING SIZES: the ratio column at the small\n"
                " sizes is mostly this, not the operation.\n",
                baselineUs, g_openCvCallFloorUs, 100.0 * g_openCvCallFloorUs / baselineUs, binCvUs,
                g_binCvCallFloorUs, 100.0 * g_binCvCallFloorUs / binCvUs);
    std::printf(" sink=%llu\n", static_cast<unsigned long long>(g_sink));
    return true;
}

}  // namespace

int main() {
    // binCV is single-threaded here, so OpenCV is held to one thread too. Left at its
    // default a multi-core box turns the ratio into a measurement of parallelism.
    cv::setNumThreads(1);

    std::printf(" denoise -- three-pixel median vs the reference implementation\n");
    std::printf("================================================================================\n\n");
    std::printf("OpenCV %s, cv::getNumThreads() = %d; binCV is single-threaded\n\n",
                CV_VERSION, cv::getNumThreads());
    std::printf("DENOMINATOR (ARCHITECTURE 10.3): the reference frontend's three-pixel median on the SAME\n");
    std::printf("binary content stored as CV_8U -- cv::min/cv::max over two zero-filled\n");
    std::printf("neighbour matrices, ported call for call. That is what the pipeline runs\n");
    std::printf("today without binCV.\n\n");
    std::printf("binCV rows: the fused one-pass kernel (ops/denoise.hpp) at uint32 and uint64,\n");
    std::printf("and the composed shiftDown + shiftLeft + majority3 spelling it replaced --\n");
    std::printf("same image, three passes, two frame-sized scratch buffers.\n\n");
    std::printf("TWO OpenCV ROWS. 'OpenCV CV_8U' re-zeroes only the two neighbour matrices,\n");
    std::printf("whose zeros ARE the border; 'OpenCV as-written' also re-zeroes the four\n");
    std::printf("temporaries the reference builds with cv::Mat::zeros and then completely\n");
    std::printf("overwrites. The DENOMINATOR is the first, so every ratio is conservative by\n");
    std::printf("the gap between them.\n\n");
    std::printf("The checksum column folds EVERY destination pixel and is representation-\n");
    std::printf("independent, so all rows of a size must print the same value.\n\n");
    std::printf("Working set is one call's live buffers, not a per-buffer ratio (CLAUDE.md,\n");
    std::printf("ARCHITECTURE 10.4).\n");

    measureCallFloors();

    // The frame and the pyramid ladder below it: the sizes a VIO frontend runs
    // this filter at, and the ones where the whole working set moves in and out
    // of a Cortex-A cache level.
    const int sizes[][2] = {{640, 480}, {320, 240}, {160, 120}, {94, 60}};

    bool ok = true;
    for (const auto& size : sizes) {
        if (!runSize(size[0], size[1])) ok = false;
    }

    std::printf("\n");
    if (!ok) {
        std::printf("AT LEAST ONE SIZE WAS NOT MEASURED CLEANLY -- see above.\n");
        return 1;
    }
    return 0;
}
