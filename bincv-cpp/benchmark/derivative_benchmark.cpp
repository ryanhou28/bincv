// binarized spatial derivative -- against cv::filter2D with the same kernel.
//
// THE DENOMINATOR (the design notes, CLAUDE.md): OpenCV performing the SAME
// SEMANTIC OPERATION on the SAME binary content stored as CV_8U. For this
// operation that is not a judgement call either -- it is
// SEAL/src/keypoint_tracking/gradients.cpp's calcBinarizedDeriv, which is two
// cv::filter2D calls with [-1, 0, 1] as a 1x3 and a 3x1. That IS what the
// pipeline runs today without binCV.
//
// TWO OpenCV ROWS, and the DENOMINATOR IS THE LEANER ONE:
//
// OpenCV filter2D x2 the two cv::filter2D calls into pre-allocated CV_16S
// destinations. This is the derivative proper and nothing
// else, and every ratio below is taken against it.
// OpenCV as-written the same two calls plus the reference's `*= 16` on each
// result and its cv::merge into one CV_16SC2 image. Those
// three extra passes are what the reference actually costs
// today; binCV reproduces none of them, so charging them
// to the baseline would flatter binCV. They are timed and
// printed, and not used as the denominator.
//
// binCV drops the scale factor deliberately (ops/derivative.hpp: it is
// representational, not semantic) and keeps the two axes as two images rather
// than merging them, because binCV's reductions read plane views.
//
// binCV ROWS:
//
// binCV u32 / u64 derivativeX + derivativeY into two TernaryMats. One pass
// per axis, NO scratch.
// binCV composed u32 the same image through ops/shift.hpp and ops/logic.hpp:
// shiftLeft + shiftRight + xor + not + and per axis, which
// is four passes and TWO FRAME-SIZED SCRATCH BUFFERS the
// caller must own. It is the spelling ops/derivative.hpp's
// "no scratch" claim is a claim ABOUT, so it is measured
// rather than asserted -- binCV has no rule that says
// "always fuse", it has a rule that says memory wins when
// memory and speed conflict, and this pair of rows is how
// one finds out whether they conflict here.
//
// AND AN N-BIT LADDER. ops/derivative.hpp claims the N-bit path is LINEAR in N
// (derivativeAdderStages(N) == 2N) rather than exponential
// (derivativeReplicatedInputs(N) == 2*(2^N - 1)). N = 1, 2, 3, 4 and 5 are timed
// at one size so the shape of that curve is a measurement and not a comment. The
// denominator there is cv::filter2D on a CV_8U image holding the pixel VALUES,
// which needs no scale factor at all -- and it is TIMED, in its own measureNs,
// beside every row. It used to be run once outside the timed region as a
// correctness oracle only, which left the N >= 2 path -- the one every pyramid
// level above 0 uses -- with a denominator named in three places and measured in
// none. The ladder also carries a WORKING-SET column, because N is the one axis
// in this benchmark along which binCV's footprint moves.
//
// ---------------------------------------------------------------------------
// MEASUREMENT VALIDITY -- the five hazards, answered as the other benchmarks in
// this directory answer them:
//
// 1. DEAD CODE. Every timed body writes memory and feeds one word to a volatile
// sink. THAT WORD IS NOT THE VALIDITY ARGUMENT. What is: after the timing,
// every implementation is run once more on image 0 and its destination is
// folded PIXEL BY PIXEL into a checksum printed in the table. The fold reads
// the derivative VALUE, so it is representation-independent and all rows of
// a size must print the same number.
// 2. CONSTANT FOLDING. Four distinct random images are rotated through.
// 3. CALIBRATED BATCHES. Every case runs enough iterations to fill a target
// millisecond budget; the reported figure is the minimum over several
// batches.
// 4. THE SIDES MUST AGREE. Every implementation is compared pixel for pixel
// BEFORE anything is timed, and a disagreement skips the size and exits
// non-zero rather than printing a table under a warning.
// 5. THE BASELINE'S FIXED PER-CALL COST IS MEASURED, NOT ASSUMED. cv::filter2D
// pays a size-independent dispatch cost -- argument checking, kernel
// analysis (it separates and specializes [-1,0,1] on every call), the
// parallel_for decision. At 640x480 that is small; at 94x60 it is most of
// the frame time, so the ns/pixel ladder is NOT a pure statement about the
// operation. The floor is measured directly on a 2x2 frame and printed
// beside every size.
//
// AND THE ONE THIS OPERATION SHARES WITH earlier measurements: A LARGE RATIO AT
// 640x480 IS PARTLY A CACHE-RESIDENCY RESULT, NOT PURELY AN ARITHMETIC ONE. The
// working-set column is what makes that readable -- binCV's whole working set for
// both axes is 5 bit-planes where OpenCV's is 5 BYTE-planes, i.e. 8x smaller, and
// on a 1 MiB-L2 Cortex-A that is the difference between fitting and not. The
// ladder exists so the reader can watch the ratio move as both sides drop below
// the cache: if the ratio collapses at 160x120 and below, the 640x480 figure was
// residency; if it holds, it is the arithmetic.
//
// ---------------------------------------------------------------------------
// WHERE THIS IS AUTHORITATIVE
//
// On x86_64 it is INDICATIVE ONLY (EXPERIMENTS.md, "Measurement platforms"). The
// numbers that belong in a claim come from the reference device:
//
// BINCV_PI_OPENCV=1./scripts/run_on_pi.sh <target>
// './benchmark/derivative_benchmark > derivative_benchmark.log'
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
#include <opencv2/imgproc.hpp>

#include "bench_util.hpp"
#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/ops/shift.hpp"
#include "bincv-cpp/quantMat.hpp"

namespace {

constexpr int kInputs = 4;
constexpr int kRepeats = 5;
constexpr double kTargetMs = 40.0;

volatile uint64_t g_sink = 0;

uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// Content: one draw per pixel, into every representation under test
// ---------------------------------------------------------------------------

struct Image {
    bincv::BinMat<uint32_t> packed32;
    bincv::BinMat<uint64_t> packed64;
    cv::Mat mask;  // CV_8U, {0, 255}: the reference's own representation
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
// The denominator: calcBinarizedDeriv, ported
// ---------------------------------------------------------------------------

/// @brief The kernels the reference builds, hoisted out of the timed region.
/// @note WHAT IS HOISTED, PRECISELY: the two 1x3/3x1 cv::Mat kernels and the two
/// CV_16S destinations, as a caller in a frame loop would. Timing their
/// allocation would flatter binCV, which allocates nothing at all here.
/// Nothing else is hoisted -- in particular cv::filter2D's per-call kernel
/// analysis is NOT, because it is not something a caller can hoist.
struct OpenCvScratch {
    cv::Mat kernelX = (cv::Mat_<int>(1, 3) << -1, 0, 1);
    cv::Mat kernelY = (cv::Mat_<int>(3, 1) << -1, 0, 1);
    cv::Mat dx, dy, merged;

    void resize(int width, int height) {
        dx = cv::Mat::zeros(height, width, CV_16S);
        dy = cv::Mat::zeros(height, width, CV_16S);
        merged = cv::Mat::zeros(height, width, CV_16SC2);
    }
};

/// @brief THE DENOMINATOR: the derivative and nothing else.
void openCvDeriv(const cv::Mat& src, OpenCvScratch& s) {
    cv::filter2D(src, s.dx, CV_16S, s.kernelX);
    cv::filter2D(src, s.dy, CV_16S, s.kernelY);
}

/// @brief calcBinarizedDeriv as the reference writes it: scale, then merge.
void openCvDerivAsWritten(const cv::Mat& src, OpenCvScratch& s) {
    cv::filter2D(src, s.dx, CV_16S, s.kernelX);
    cv::filter2D(src, s.dy, CV_16S, s.kernelY);
    const int scaleFactor = 16;
    s.dx *= scaleFactor;
    s.dy *= scaleFactor;
    std::vector<cv::Mat> channels = {s.dx, s.dy};
    cv::merge(channels, s.merged);
}

// ---------------------------------------------------------------------------
// Timing (duplicated from denoise_benchmark.cpp -- see the note there on why the
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
    double workingSetBytes = 0.0;
    int traversals = 0;
    uint64_t checksum = 0;
    bool isDenominator = false;
};

std::vector<Row> g_rows;

// ---------------------------------------------------------------------------
// Hazard 1: a fold over every destination PIXEL VALUE
// ---------------------------------------------------------------------------
//
// Reads the derivative VALUE in [-max, +max], so a packed sign-magnitude
// destination and a CV_16S one holding the same picture produce the SAME number.
// The checksum column is therefore an agreement check as well as a dead-code
// check. It runs once per case, outside every timed region.

uint64_t foldValue(uint64_t h, int value) {
    h ^= static_cast<uint64_t>(static_cast<int64_t>(value) + 4096);
    return h * UINT64_C(0x100000001B3);
}

template <size_t N, typename WordType>
uint64_t valueChecksum(const bincv::SignedQuantMat<N, WordType>& dx,
                       const bincv::SignedQuantMat<N, WordType>& dy) {
    uint64_t h = UINT64_C(0xCBF29CE484222325);
    for (int y = 0; y < dx.rows(); ++y) {
        for (int x = 0; x < dx.cols(); ++x) {
            h = foldValue(h, dx.at(y, x));
            h = foldValue(h, dy.at(y, x));
        }
    }
    return h;
}

/// @param divisor 4080 for the reference's {0,255} content scaled by 16, 255 for
/// the unscaled filter2D rows, 1 for an N-bit source holding values.
uint64_t valueChecksum(const cv::Mat& dx, const cv::Mat& dy, int divisor) {
    uint64_t h = UINT64_C(0xCBF29CE484222325);
    for (int y = 0; y < dx.rows; ++y) {
        for (int x = 0; x < dx.cols; ++x) {
            h = foldValue(h, dx.at<short>(y, x) / divisor);
            h = foldValue(h, dy.at<short>(y, x) / divisor);
        }
    }
    return h;
}

// ---------------------------------------------------------------------------
// The composed spelling: what "no scratch" is a claim about
// ---------------------------------------------------------------------------
//
// Per axis: two shifts into scratch, then `mag = a ^ b` and `sign = b & ~a`.
// The `~a` is written back over `a` (ops/logic.hpp supports the exact-alias
// in-place case,) and `a` is dead by then, so the two scratch frames are
// enough for both axes -- this is the LEANEST composed spelling, not a strawman.
// Four passes per axis against the fused kernel's one.

template <typename WordType>
void composedDeriv(const bincv::BinMat<WordType>& src, bincv::TernaryMat<WordType>& dx,
                   bincv::TernaryMat<WordType>& dy, bincv::BinMat<WordType>& a,
                   bincv::BinMat<WordType>& b) {
    bincv::shiftLeft(src.constView(), a.view(), 1, bincv::BORDER_REFLECT_101);
    bincv::shiftRight(src.constView(), b.view(), 1, bincv::BORDER_REFLECT_101);
    bincv::bitwiseXor(a.constView(), b.constView(), dx.magnitude(0));
    bincv::bitwiseNot(a.constView(), a.view());
    bincv::bitwiseAnd(b.constView(), a.constView(), dx.sign());

    bincv::shiftUp(src.constView(), a.view(), 1, bincv::BORDER_REFLECT_101);
    bincv::shiftDown(src.constView(), b.view(), 1, bincv::BORDER_REFLECT_101);
    bincv::bitwiseXor(a.constView(), b.constView(), dy.magnitude(0));
    bincv::bitwiseNot(a.constView(), a.view());
    bincv::bitwiseAnd(b.constView(), a.constView(), dy.sign());
}

// ---------------------------------------------------------------------------
// Hazard 5: the size-independent cost of one call, measured
// ---------------------------------------------------------------------------

double g_openCvCallFloorUs = 0.0;
double g_binCvCallFloorUs = 0.0;

void measureCallFloors() {
    Image tiny;
    makeImage(tiny, 2, 2, UINT64_C(0xF100));
    OpenCvScratch s;
    s.resize(2, 2);
    bincv::TernaryMat<uint32_t> dx(2, 2);
    bincv::TernaryMat<uint32_t> dy(2, 2);

    // THE SINK ON A 2x2 FRAME IS CONSTANT ON BOTH SIDES, AND UNAVOIDABLY SO.
    // Under reflect-101 a 2x2 image maps column 2 back to column 0, so BOTH taps
    // of every pixel read the same source pixel and the whole derivative is
    // identically zero for any content -- measured: dx(0,0) and dx(1,1) are 0 on
    // every draw. That is true of binCV's floor row below in exactly the same
    // way, so the two floors stay symmetric; and neither call can be elided
    // regardless, cv::filter2D being an out-of-line library call and derivativeX
    // writing through a pointer the compiler cannot prove dead. The sizes that
    // carry the actual comparison read an INTERIOR pixel -- see runSize.
    g_openCvCallFloorUs = measureNs(
                              [&](int) {
                                  openCvDeriv(tiny.mask, s);
                                  g_sink += static_cast<uint64_t>(s.dx.at<short>(1, 1) + 4096);
                              },
                              kRepeats, kTargetMs) /
                          1000.0;
    g_binCvCallFloorUs = measureNs(
                             [&](int) {
                                 bincv::derivativeX(tiny.packed32, dx);
                                 bincv::derivativeY(tiny.packed32, dy);
                                 g_sink += dx.data()[0];
                             },
                             kRepeats, kTargetMs) /
                         1000.0;

    std::printf("\nFIXED PER-CALL COST (hazard 5), measured on a 2x2 frame -- the part of every\n"
                "row below that is not the operation:\n");
    std::printf(" OpenCV, 2 x cv::filter2D: %8.3f us per call\n", g_openCvCallFloorUs);
    std::printf(" binCV, derivativeX + derivativeY: %8.3f us per call\n", g_binCvCallFloorUs);
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
                  UINT64_C(0xDE21A71) + static_cast<uint64_t>(i) * UINT64_C(7919));
    }

    bincv::TernaryMat<uint32_t> dx32(width, height);
    bincv::TernaryMat<uint32_t> dy32(width, height);
    bincv::TernaryMat<uint64_t> dx64(width, height);
    bincv::TernaryMat<uint64_t> dy64(width, height);
    bincv::TernaryMat<uint32_t> cdx32(width, height);
    bincv::TernaryMat<uint32_t> cdy32(width, height);
    bincv::BinMat<uint32_t> scratchA(width, height);
    bincv::BinMat<uint32_t> scratchB(width, height);
    OpenCvScratch cvScratch;
    cvScratch.resize(width, height);

    const double planeBytes32 =
        static_cast<double>(images[0].packed32.sizeInWords() * sizeof(uint32_t));
    const double planeBytes64 =
        static_cast<double>(images[0].packed64.sizeInWords() * sizeof(uint64_t));

    // --- footprint, reported before the timings rather than after -----------
    //
    // The working set is ONE CALL's live buffers, per CLAUDE.md, and both axes are
    // one call's worth of work here because that is what a VIO frontend needs
    // before it can form the covariance.
    const double binCvSet32 = 5.0 * planeBytes32;         // src + dx(2) + dy(2)
    const double binCvSet64 = 5.0 * planeBytes64;
    const double composedSet32 = 7.0 * planeBytes32;      // + 2 scratch frames
    const double openCvSet = 5.0 * pixels;                // CV_8U src + 2 x CV_16S
    const double openCvSetAsWritten = 9.0 * pixels;       // + the CV_16SC2 merge

    std::printf(" one plane: binCV uint32 %.0f B, binCV uint64 %.0f B, OpenCV CV_8U %.0f B\n",
                planeBytes32, planeBytes64, pixels);
    std::printf(" WORKING SET of one call, BOTH AXES (the number CLAUDE.md asks for):\n");
    std::printf(" binCV u32 %10.0f B src + 2 ternary results, no scratch\n",
                binCvSet32);
    std::printf(" binCV u64 %10.0f B the same at 64-bit words\n", binCvSet64);
    std::printf(" binCV composed u32 %10.0f B + 2 frame-sized scratch buffers (%.2fx)\n",
                composedSet32, composedSet32 / binCvSet32);
    std::printf(" OpenCV filter2D %10.0f B CV_8U src + 2 x CV_16S (%.1fx binCV u32)\n",
                openCvSet, openCvSet / binCvSet32);
    std::printf(" OpenCV as-written %10.0f B + the CV_16SC2 merge (%.1fx binCV u32)\n",
                openCvSetAsWritten, openCvSetAsWritten / binCvSet32);

    // --- hazard 4: the sides must agree before anything is timed -------------
    {
        const Image& img = images[0];
        openCvDeriv(img.mask, cvScratch);
        const uint64_t reference = valueChecksum(cvScratch.dx, cvScratch.dy, 255);
        openCvDerivAsWritten(img.mask, cvScratch);
        const uint64_t asWritten = valueChecksum(cvScratch.dx, cvScratch.dy, 4080);
        if (reference != asWritten) {
            std::printf(" THE TWO OpenCV SPELLINGS DISAGREE -- SKIPPING THIS SIZE.\n");
            return false;
        }

        bincv::derivativeX(img.packed32, dx32);
        bincv::derivativeY(img.packed32, dy32);
        bincv::derivativeX(img.packed64, dx64);
        bincv::derivativeY(img.packed64, dy64);
        composedDeriv(img.packed32, cdx32, cdy32, scratchA, scratchB);

        const uint64_t a = valueChecksum(dx32, dy32);
        const uint64_t b = valueChecksum(dx64, dy64);
        const uint64_t c = valueChecksum(cdx32, cdy32);
        if (a != reference || b != reference || c != reference) {
            std::printf(" RESULTS DISAGREE -- SKIPPING THIS SIZE.\n"
                        " reference %llu, binCV u32 %llu, u64 %llu, composed %llu.\n"
                        " The implementations do not compute the same image, so no timing of\n"
                        " them is a comparison.\n",
                        static_cast<unsigned long long>(reference),
                        static_cast<unsigned long long>(a), static_cast<unsigned long long>(b),
                        static_cast<unsigned long long>(c));
            return false;
        }
    }

    const size_t firstRow = g_rows.size();

    {
        const double ns = measureNs(
            [&](int i) {
                openCvDeriv(images[static_cast<size_t>(i % kInputs)].mask, cvScratch);
                // (1, 1), NOT (0, 0). Reflect-101 pins dx(0, y) to exactly 0 for
                // every input -- both taps read column 1 -- so a sink reading the
                // corner would add a CONSTANT on the OpenCV side while binCV's
                // sink (dx32.data[0], which spans columns 0..31) is genuinely
                // content-dependent. Measured on six random 640x480 draws:
                // dx(0,0) = 0 every time, dx(1,1) alternates 0 and 4080.
                g_sink += static_cast<uint64_t>(cvScratch.dx.at<short>(1, 1) + 4096);
            },
            kRepeats, kTargetMs);
        openCvDeriv(images[0].mask, cvScratch);
        g_rows.push_back(Row{"OpenCV filter2D x2", ns / pixels, openCvSet, 2,
                             valueChecksum(cvScratch.dx, cvScratch.dy, 255), true});
    }
    {
        const double ns = measureNs(
            [&](int i) {
                openCvDerivAsWritten(images[static_cast<size_t>(i % kInputs)].mask, cvScratch);
                // (1, 1) for the reason the row above gives; on the CV_16SC2
                // merge that lands on an interleaved component of an interior
                // pixel, which varies with the content.
                g_sink += static_cast<uint64_t>(cvScratch.merged.at<short>(1, 1) + 4096);
            },
            kRepeats, kTargetMs);
        openCvDerivAsWritten(images[0].mask, cvScratch);
        g_rows.push_back(Row{"OpenCV as-written", ns / pixels, openCvSetAsWritten, 5,
                             valueChecksum(cvScratch.dx, cvScratch.dy, 4080), false});
    }
    {
        const double ns = measureNs(
            [&](int i) {
                const bincv::BinMat<uint32_t>& src =
                    images[static_cast<size_t>(i % kInputs)].packed32;
                bincv::derivativeX(src, dx32);
                bincv::derivativeY(src, dy32);
                g_sink += dx32.data()[0];
            },
            kRepeats, kTargetMs);
        bincv::derivativeX(images[0].packed32, dx32);
        bincv::derivativeY(images[0].packed32, dy32);
        g_rows.push_back(
            Row{"binCV u32", ns / pixels, binCvSet32, 2, valueChecksum(dx32, dy32), false});
    }
    {
        const double ns = measureNs(
            [&](int i) {
                const bincv::BinMat<uint64_t>& src =
                    images[static_cast<size_t>(i % kInputs)].packed64;
                bincv::derivativeX(src, dx64);
                bincv::derivativeY(src, dy64);
                g_sink += dx64.data()[0];
            },
            kRepeats, kTargetMs);
        bincv::derivativeX(images[0].packed64, dx64);
        bincv::derivativeY(images[0].packed64, dy64);
        g_rows.push_back(
            Row{"binCV u64", ns / pixels, binCvSet64, 2, valueChecksum(dx64, dy64), false});
    }
    {
        const double ns = measureNs(
            [&](int i) {
                composedDeriv(images[static_cast<size_t>(i % kInputs)].packed32, cdx32, cdy32,
                              scratchA, scratchB);
                g_sink += cdx32.data()[0];
            },
            kRepeats, kTargetMs);
        composedDeriv(images[0].packed32, cdx32, cdy32, scratchA, scratchB);
        g_rows.push_back(Row{"binCV composed u32", ns / pixels, composedSet32, 8,
                             valueChecksum(cdx32, cdy32), false});
    }

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

    const double baselineUs = reference * pixels / 1000.0;
    const double binCvUs = g_rows[firstRow + 2].nsPerPixel * pixels / 1000.0;
    std::printf(" PER-FRAME: OpenCV %.2f us, of which %.2f us is the measured fixed per-call\n"
                " cost (%.0f%%). binCV u32 %.2f us, fixed cost %.2f us (%.0f%%).\n"
                " SUBTRACT BEFORE COMPARING SIZES.\n",
                baselineUs, g_openCvCallFloorUs, 100.0 * g_openCvCallFloorUs / baselineUs, binCvUs,
                g_binCvCallFloorUs, 100.0 * g_binCvCallFloorUs / binCvUs);
    std::printf(" sink=%llu\n", static_cast<unsigned long long>(g_sink));
    return true;
}

// ---------------------------------------------------------------------------
// The N-bit ladder: is the cost linear in N?
// ---------------------------------------------------------------------------
//
// ops/derivative.hpp claims derivativeAdderStages(N) == 2N stages against the
// rejected replication route's 2*(2^N - 1) single-bit inputs. A ns/pixel column
// that roughly doubles from N = 1 to N = 2 and then grows by a roughly constant
// increment per further plane is the linear claim; anything that doubles per
// plane is the exponential one. The denominator is cv::filter2D on a CV_8U image
// holding the pixel VALUES, which is the same operation with no scale factor, and
// it is TIMED here rather than merely consulted for agreement.

/// @brief One row of the N-bit ladder: binCV, its OpenCV denominator, and the
/// working set each of them costs.
/// @note THE OpenCV FIGURE HERE IS TIMED, NOT AN ORACLE. It used to be neither:
/// the two cv::filter2D calls ran once, outside the timed region, purely to
/// check agreement, while this file's header and both described them as
/// "the denominator" -- a denominator that was never measured, so the N >= 2
/// path (which is what every pyramid level above 0 runs) had no OpenCV
/// comparison anywhere. They are now inside their own measureNs, and the
/// ladder prints the ratio.
struct NBitRow {
    double binCvNs = -1.0;
    double openCvNs = -1.0;
    double binCvBytes = 0.0;
    double openCvBytes = 0.0;
};

template <size_t N>
NBitRow timeNBit(int width, int height, uint64_t& checksumOut) {
    std::vector<bincv::QuantMat<N, uint32_t>> srcs;
    srcs.reserve(static_cast<size_t>(kInputs));
    for (int i = 0; i < kInputs; ++i) {
        bincv::QuantMat<N, uint32_t> m(width, height);
        uint64_t state = UINT64_C(0xA5A5) + static_cast<uint64_t>(i) * UINT64_C(7919) + N;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                m.set(y, x, static_cast<unsigned>(nextRandom(state) % (1u << N)));
            }
        }
        srcs.push_back(std::move(m));
    }
    bincv::SignedQuantMat<N, uint32_t> dx(width, height);
    bincv::SignedQuantMat<N, uint32_t> dy(width, height);

    // The same four images as CV_8U VALUE planes -- the representation a caller
    // without binCV has, and what the denominator runs on.
    std::vector<cv::Mat> src8(static_cast<size_t>(kInputs));
    for (int i = 0; i < kInputs; ++i) {
        cv::Mat m(height, width, CV_8U);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                m.at<uchar>(y, x) =
                    static_cast<uchar>(srcs[static_cast<size_t>(i)].at(y, x));
            }
        }
        src8[static_cast<size_t>(i)] = m;
    }
    cv::Mat kernelX = (cv::Mat_<int>(1, 3) << -1, 0, 1);
    cv::Mat kernelY = (cv::Mat_<int>(3, 1) << -1, 0, 1);
    cv::Mat refX = cv::Mat::zeros(height, width, CV_16S);
    cv::Mat refY = cv::Mat::zeros(height, width, CV_16S);

    const double ns = measureNs(
        [&](int i) {
            const bincv::QuantMat<N, uint32_t>& src = srcs[static_cast<size_t>(i % kInputs)];
            bincv::derivativeX(src, dx);
            bincv::derivativeY(src, dy);
            g_sink += dx.data()[0];
        },
        kRepeats, kTargetMs);

    // The denominator, timed the same way and against the same four images. The
    // destinations are hoisted exactly as the main table hoists them.
    const double openCvNs = measureNs(
        [&](int i) {
            const cv::Mat& m = src8[static_cast<size_t>(i % kInputs)];
            cv::filter2D(m, refX, CV_16S, kernelX);
            cv::filter2D(m, refY, CV_16S, kernelY);
            g_sink += static_cast<uint64_t>(refX.at<short>(1, 1) + 4096);
        },
        kRepeats, kTargetMs);

    bincv::derivativeX(srcs[0], dx);
    bincv::derivativeY(srcs[0], dy);

    // Hazard 4 applies here too: agreement before any number is reported.
    cv::filter2D(src8[0], refX, CV_16S, kernelX);
    cv::filter2D(src8[0], refY, CV_16S, kernelY);
    if (valueChecksum(refX, refY, 1) != valueChecksum(dx, dy)) {
        std::printf(" N = %zu DISAGREES WITH cv::filter2D -- the timing below is not a "
                    "comparison.\n",
                    N);
        checksumOut = 0;
        return NBitRow{};
    }
    checksumOut = valueChecksum(dx, dy);

    const double pixels = static_cast<double>(width) * static_cast<double>(height);
    NBitRow row;
    row.binCvNs = ns / pixels;
    row.openCvNs = openCvNs / pixels;
    // WORKING SET, because CLAUDE.md asks for memory and speed together and this
    // is the one table where the footprint MOVES with the variable on the x axis.
    // binCV: N source planes + 2 axes x (N magnitude planes + 1 sign plane).
    // OpenCV: one CV_8U value plane + 2 x CV_16S.
    const double planeBytes =
        static_cast<double>(srcs[0].plane(0).stride * static_cast<size_t>(height) *
                            sizeof(uint32_t));
    row.binCvBytes = static_cast<double>(N + 2 * (N + 1)) * planeBytes;
    row.openCvBytes = 5.0 * pixels;
    return row;
}

bool runNBitLadder(int width, int height) {
    std::printf("\n================ N-BIT LADDER, %d x %d ================\n", width, height);
    std::printf(" Is the N-bit path LINEAR in N (2N stages) or exponential (2*(2^N - 1))?\n");
    std::printf(" Every row is checked against cv::filter2D on the same values first, and\n");
    std::printf(" cv::filter2D is then TIMED on those same values -- so N >= 2, which is what\n");
    std::printf(" every pyramid level above 0 runs, has a real denominator and not only an\n");
    std::printf(" oracle. The denominator needs no scale factor here: the CV_8U image holds\n");
    std::printf(" the pixel VALUES, not {0, 255}.\n\n");
    std::printf(" %-4s %11s %11s %10s %8s %12s %13s %9s %11s\n", "N", "binCV ns/px",
                "OpenCV ns/px", "vs OpenCV", "vs N=1", "binCV bytes", "OpenCV bytes", "2N stg",
                "replicated");
    std::printf(" --------------------------------------------------------------------------"
                "-------------------------\n");

    uint64_t sum = 0;
    const NBitRow rows[] = {timeNBit<1>(width, height, sum), timeNBit<2>(width, height, sum),
                            timeNBit<3>(width, height, sum), timeNBit<4>(width, height, sum),
                            timeNBit<5>(width, height, sum)};
    const double n1 = rows[0].binCvNs;
    bool ok = true;
    for (size_t k = 0; k < 5; ++k) {
        const size_t n = k + 1;
        const NBitRow& r = rows[k];
        if (r.binCvNs < 0.0 || n1 <= 0.0) {
            ok = false;
            continue;
        }
        std::printf(" %-4zu %11.5f %11.5f %9.2fx %7.2fx %10.0f B %11.0f B %9zu %11zu\n", n,
                    r.binCvNs, r.openCvNs, r.openCvNs / r.binCvNs, r.binCvNs / n1, r.binCvBytes,
                    r.openCvBytes, bincv::derivativeAdderStages(n),
                    bincv::derivativeReplicatedInputs(n));
    }
    std::printf("\n 'vs N=1' against the STAGES column is the whole reading: a cost that tracks\n"
                " 2N is the linear formulation, one that tracks the replicated column is the\n"
                " exponential one this was written to avoid. The two axes together write\n"
                " 2*(N+1) destination planes at row N, so part of any growth is stores.\n");
    std::printf(" READ THE TWO BYTE COLUMNS WITH THE RATIO, per CLAUDE.md. binCV's footprint\n"
                " is the column that MOVES: 3(N+1) - 1 bit-planes against OpenCV's flat one\n"
                " byte-plane plus two 16-bit planes. binCV stays smaller at every N in this\n"
                " ladder, but the margin narrows as N rises, and the crossing point is what\n"
                " (bits per pyramid level) is for -- it is not assumed here.\n");
    std::printf(" Note the fixed per-call cost printed above applies to the OpenCV column at\n"
                " every row of this ladder too.\n");
    return ok;
}

}  // namespace

int main() {
    std::printf(" binarized spatial derivative -- vs cv::filter2D with the same kernel\n");
    std::printf("================================================================================\n\n");
    std::printf("DENOMINATOR (ARCHITECTURE 10.3): SEAL's calcBinarizedDeriv on the SAME binary\n");
    std::printf("content stored as CV_8U -- two cv::filter2D calls with [-1, 0, 1] as a 1x3 and\n");
    std::printf("a 3x1, ported. That is what the pipeline runs today without binCV.\n\n");
    std::printf("The DENOMINATOR row is 'OpenCV filter2D x2' -- the derivative and nothing else.\n");
    std::printf("'OpenCV as-written' adds the reference's own `*= 16` and cv::merge, which binCV\n");
    std::printf("reproduces neither of (ops/derivative.hpp: the scale is representational, and\n");
    std::printf("interleaved channels are unreachable to a word-parallel popcount). Charging\n");
    std::printf("those to the baseline would flatter binCV, so they are printed and not used.\n\n");
    std::printf("Every row computes BOTH AXES, because that is what a VIO frontend needs before\n");
    std::printf("it can form the covariance.\n\n");
    std::printf("The checksum column folds every destination pixel's VALUE and is\n");
    std::printf("representation-independent, so all rows of a size must print the same number.\n\n");
    std::printf("READ THE WORKING-SET COLUMN WITH THE RATIO. binCV's whole working set is five\n");
    std::printf("BIT-planes where OpenCV's is one byte-plane and two 16-bit planes -- 8x smaller\n");
    std::printf("-- so on a 1 MiB-L2 Cortex-A part of any large ratio at 640x480 is residency\n");
    std::printf("rather than arithmetic, exactly as in. The size ladder is\n");
    std::printf("there to separate the two: if the ratio collapses once BOTH sides fit in cache,\n");
    std::printf("the headline figure was residency; if it holds, it is the operation.\n");

    measureCallFloors();

    const int sizes[][2] = {{640, 480}, {320, 240}, {160, 120}, {94, 60}};

    bool ok = true;
    for (const auto& size : sizes) {
        if (!runSize(size[0], size[1])) ok = false;
    }
    if (!runNBitLadder(640, 480)) ok = false;

    std::printf("\n");
    if (!ok) {
        std::printf("AT LEAST ONE CASE WAS NOT MEASURED CLEANLY -- see above.\n");
        return 1;
    }
    return 0;
}
