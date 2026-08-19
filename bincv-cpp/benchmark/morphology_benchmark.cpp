// T3.3 morphology -- erode / dilate / morphologyEx against OpenCV.
//
// THE DENOMINATOR (ARCHITECTURE 10.3, CLAUDE.md): OpenCV performing the SAME
// SEMANTIC OPERATION on the SAME binary content stored as CV_8U -- cv::erode,
// cv::dilate and cv::morphologyEx with the SAME structuring element, the same
// anchor and the same border. That is what a user does today without binCV, and
// the two sides are checked pixel for pixel before anything is timed, so every
// row of every table computes one image.
//
// SIX CASES PER SIZE, chosen because they are the things a reader wants to know
// rather than to make a long table:
//
//   erode  3x3 rect      the common case, and the one T3.3 asks to be special-
//                        cased. Both sides' fastest path.
//   dilate 3x3 rect      the same shape with the opposite fold and the opposite
//                        border fill (D-12), so a fill bug shows up as an
//                        asymmetry between two rows that should match.
//   erode  5x5 ellipse   the GENERAL path: 17 set cells, no special case, and
//                        the case where OpenCV stops being separable too.
//   OPEN   3x3 rect      the compound path, which is where binCV's caller-
//                        provided scratch frame appears in the working set.
//   erode 3x3 REPLICATE  } THE BORDER AXIS. See below -- these are not padding.
//   erode 3x3 REFLECT_101}
//
// THE BORDER TYPE IS A CASE AXIS, NOT A CONSTANT. binCV handles BORDER_CONSTANT
// entirely in the word path and the other four in a per-pixel fixup over the
// 2 * reach edge columns, so the two have genuinely different cost structures and
// a ratio measured at one says nothing about the other. Measured during T3.3's
// review, at a point when that fixup walked the whole row instead of its two
// bands: erode 3x3 at 640x480 cost 19.5 us under BORDER_CONSTANT and 241-260 us
// under the other four -- a published 1.11x that was really 0.21x for four of the
// five values a caller may pass. Every row below names its border type, and two
// cases exist solely so the non-constant path is measured rather than assumed.
// BORDER_WRAP is absent because it has no denominator: cv::morphologyEx refuses
// it by assertion (`columnBorderType != BORDER_WRAP`).
//
// WHAT IS DELIBERATELY *NOT* IN THIS FILE: the 3x3 special case priced against
// the general row kernel. That comparison is binCV against binCV, so it needs no
// OpenCV denominator, and it lives in morphology_path_benchmark.cpp -- because
// MEASURED, adding its one extra call site to THIS translation unit moved the
// headline erode 3x3 row by ~10% (0.143-0.159 against 0.126-0.129 ns/pixel at
// 640x480 on x86, same header, same everything else). A benchmark that changes
// the code it measures is not measuring it, and the number this file exists to
// publish is the one against OpenCV.
//
// FOOTPRINT IS REPORTED ALONGSIDE EVERY TIMING, per CLAUDE.md, as the WORKING SET
// OF ONE CALL rather than as a per-buffer ratio. For this operation the footprint
// column is the more interesting one and it is not close: a packed frame is 1/8
// of a CV_8U frame, and erode/dilate need no scratch at all.
//
// OPENCV'S OWN WORKING SET IS MEASURED, NOT ASSUMED FROM THE ALGEBRA. It is
// tempting to write "OPEN = erode then dilate, so three frames"; cv::morphologyEx
// does `erode(src,dst)` then `dilate(dst,dst)` and allocates NOTHING, so it is
// two. Probed with VmHWM around a single 4096x4096 call, one op per process:
// OPEN, CLOSE, TOPHAT and BLACKHAT each moved the high-water mark by 0 kB and
// only MORPH_GRADIENT by ~one frame (17188 kB of a 16384 kB frame). The
// openCvBuffers field below carries that measurement, and an earlier value of 3
// for OPEN overstated binCV's footprint advantage on the compound case by 1.5x.
//
// ---------------------------------------------------------------------------
// MEASUREMENT VALIDITY -- the five hazards this directory enumerates:
//
//   1. DEAD CODE. Every timed body writes memory the next iteration reads and
//      feeds one destination word to a volatile sink. That word is NOT the
//      argument: after the timing every implementation is run once more on image
//      0 and its destination folded PIXEL BY PIXEL into a checksum printed in the
//      table. All rows of a case must print the same number.
//   2. CONSTANT FOLDING. Four distinct random images are rotated through. That
//      makes the loop's RESIDENT set larger than the working set of ONE call,
//      which the tables report and which is the number CLAUDE.md asks for: at
//      640x480 the timed loop keeps 4 sources + 1 destination live, 1500 KiB on
//      the OpenCV side against 188 KiB at uint32. Only the OpenCV side straddles
//      a 1 MiB L2, so the asymmetry can only flatter binCV -- it is printed per
//      size below so a reader can bound it rather than argue about it.
//   3. CALIBRATED BATCHES. Every case runs enough iterations to fill a target
//      millisecond budget; the reported figure is the minimum over five batches.
//   4. THE SIDES MUST AGREE, checked before anything is timed. A disagreement
//      skips the size and exits non-zero rather than printing a table under a
//      warning.
//   5. THE FIXED PER-CALL COST IS MEASURED, NOT ASSUMED. A cv:: call pays a
//      size-independent dispatch cost, and a ns/PIXEL figure divides it by the
//      pixel count -- so it grows without limit as the frame shrinks. It is
//      measured on a 2x2 frame and printed beside every table, for both sides, so
//      the ladder is read with it rather than mistaken for a cache result.
//
// ---------------------------------------------------------------------------
// WHERE THIS IS AUTHORITATIVE
//
// On x86_64 it is INDICATIVE ONLY (EXPERIMENTS.md, "Measurement platforms"). The
// numbers that belong in a claim come from the reference device:
//
//   BINCV_PI_OPENCV=1 ./scripts/run_on_pi.sh <target>
//       './benchmark/morphology_benchmark > morphology_benchmark.log'
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
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/ops/morphology.hpp"
#include "bincv-cpp/ops/shift.hpp"

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

/// @brief One image in every representation under test, from ONE draw per pixel.
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

// ---------------------------------------------------------------------------
// Hazard 1: a fold over every destination PIXEL, not over one word
// ---------------------------------------------------------------------------

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
// The four cases
// ---------------------------------------------------------------------------

struct Case {
    const char* name;
    bincv::MorphOp op;
    bincv::StructuringElement se;
    int cvShape;
    int cvCols;
    int cvRows;
    int binCvBuffers;   ///< frames live during one binCV call, src and dst included
    int openCvBuffers;  ///< the same for OpenCV -- MEASURED, see the header note
    bincv::BorderType border;  ///< binCV's border type for this case
    int cvBorder;              ///< the same value spelled for OpenCV
};

/// @note `openCvBuffers` is 2 everywhere here, including OPEN, because that is
///       what cv::morphologyEx was measured to allocate (header note). It is 3
///       only for MORPH_GRADIENT, which this table does not run.
std::vector<Case> cases() {
    return {
        {"erode 3x3 rect", bincv::MORPH_ERODE, bincv::StructuringElement::rect(3, 3),
         cv::MORPH_RECT, 3, 3, 2, 2, bincv::BORDER_CONSTANT, cv::BORDER_CONSTANT},
        {"dilate 3x3 rect", bincv::MORPH_DILATE, bincv::StructuringElement::rect(3, 3),
         cv::MORPH_RECT, 3, 3, 2, 2, bincv::BORDER_CONSTANT, cv::BORDER_CONSTANT},
        {"erode 5x5 ellipse", bincv::MORPH_ERODE, bincv::StructuringElement::ellipse(5, 5),
         cv::MORPH_ELLIPSE, 5, 5, 2, 2, bincv::BORDER_CONSTANT, cv::BORDER_CONSTANT},
        {"open 3x3 rect", bincv::MORPH_OPEN, bincv::StructuringElement::rect(3, 3),
         cv::MORPH_RECT, 3, 3, 3, 2, bincv::BORDER_CONSTANT, cv::BORDER_CONSTANT},
        {"erode 3x3 rect", bincv::MORPH_ERODE,
         bincv::StructuringElement::rect(3, 3), cv::MORPH_RECT, 3, 3, 2, 2,
         bincv::BORDER_REPLICATE, cv::BORDER_REPLICATE},
        {"erode 3x3 rect", bincv::MORPH_ERODE,
         bincv::StructuringElement::rect(3, 3), cv::MORPH_RECT, 3, 3, 2, 2,
         bincv::BORDER_REFLECT_101, cv::BORDER_REFLECT_101},
    };
}

const char* borderName(bincv::BorderType b) {
    switch (b) {
        case bincv::BORDER_CONSTANT: return "BORDER_CONSTANT";
        case bincv::BORDER_REPLICATE: return "BORDER_REPLICATE";
        case bincv::BORDER_REFLECT: return "BORDER_REFLECT";
        case bincv::BORDER_REFLECT_101: return "BORDER_REFLECT_101";
        default: return "BORDER_WRAP";
    }
}

cv::Mat cvKernel(const Case& c) {
    return cv::getStructuringElement(c.cvShape, cv::Size(c.cvCols, c.cvRows));
}

/// @brief OpenCV's side of one case, with the destination pre-allocated.
/// @note cv::erode / cv::dilate / cv::morphologyEx reuse a destination that
///       already has the right size and type, so hoisting it out of the loop
///       measures the operation rather than malloc -- which is the same courtesy
///       the binCV rows get, since they allocate nothing at all.
void runOpenCv(const Case& c, const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel) {
    // morphologyDefaultBorderValue() is what cv::erode/cv::dilate use by default
    // and is ignored for the non-constant types; passing it explicitly is what
    // lets the border argument be varied without changing anything else (D-12).
    const cv::Scalar bv = cv::morphologyDefaultBorderValue();
    switch (c.op) {
        case bincv::MORPH_ERODE:
            cv::erode(src, dst, kernel, cv::Point(-1, -1), 1, c.cvBorder, bv);
            return;
        case bincv::MORPH_DILATE:
            cv::dilate(src, dst, kernel, cv::Point(-1, -1), 1, c.cvBorder, bv);
            return;
        default:
            cv::morphologyEx(src, dst, static_cast<int>(c.op), kernel, cv::Point(-1, -1), 1,
                             c.cvBorder, bv);
            return;
    }
}

template <typename WordType>
void runBinCv(const Case& c, const bincv::BinMat<WordType>& src, bincv::BinMat<WordType>& dst,
              bincv::BinMat<WordType>& scratch) {
    switch (c.op) {
        case bincv::MORPH_ERODE:
            bincv::erode(src.constView(), dst.view(), c.se, c.border);
            return;
        case bincv::MORPH_DILATE:
            bincv::dilate(src.constView(), dst.view(), c.se, c.border);
            return;
        default:
            bincv::morphologyEx(src.constView(), dst.view(), c.op, c.se, scratch.view(),
                                c.border);
            return;
    }
}

// ---------------------------------------------------------------------------
// The COMPOSED spelling -- the one ops/morphology.hpp does not use
// ---------------------------------------------------------------------------
//
// dilate is an OR of shifted copies and erode is an AND of shifted copies
// (D-12), and this is that sentence written literally with ops/shift.hpp and
// ops/logic.hpp -- the two kernels T2.3/T2.4 and T2.2 already ship. It is what a
// caller would write today without ops/morphology.hpp, and the shipped kernel is
// its FUSED form.
//
// It is here because the choice between them is a footprint choice and CLAUDE.md
// says those are settled by measurement. The composed form needs a FRAME-SIZED
// TEMPORARY between the shift and the combine -- a kernel may not allocate one,
// so it would have to be a caller-provided scratch on erode and dilate as well,
// which is a third frame on the hottest call in the MVP. The fused form needs
// none. This row is what says whether that costs anything in time.
//
// Its traversal count is 2k - 1 for a k-cell element against the fused form's
// one, so the expectation is that it is also slower; an expectation is not a
// measurement, which is the point.
template <bool IsErode, typename WordType>
void runComposed(const bincv::StructuringElement& se, const bincv::BinMat<WordType>& src,
                 bincv::BinMat<WordType>& dst, bincv::BinMat<WordType>& tmp) {
    const int ax = se.anchorCol();
    const int ay = se.anchorRow();
    bool first = true;
    for (int ey = 0; ey < se.rows; ++ey) {
        for (int ex = 0; ex < se.cols; ++ex) {
            if (!se.activeAt(ex, ey)) continue;
            const ptrdiff_t dx = static_cast<ptrdiff_t>(ex) - static_cast<ptrdiff_t>(ax);
            const ptrdiff_t dy = static_cast<ptrdiff_t>(ey) - static_cast<ptrdiff_t>(ay);
            if (first) {
                bincv::shift(src.constView(), dst.view(), dx, dy, bincv::BORDER_CONSTANT,
                             IsErode);
                first = false;
                continue;
            }
            bincv::shift(src.constView(), tmp.view(), dx, dy, bincv::BORDER_CONSTANT, IsErode);
            if (IsErode) {
                bincv::bitwiseAnd(dst.constView(), tmp.constView(), dst.view());
            } else {
                bincv::bitwiseOr(dst.constView(), tmp.constView(), dst.view());
            }
        }
    }
}

/// @brief How many frame traversals the composed form makes: 2k - 1 for k cells.
int composedTraversals(const bincv::StructuringElement& se) {
    int cells = 0;
    for (int ey = 0; ey < se.rows; ++ey) {
        for (int ex = 0; ex < se.cols; ++ex) {
            if (se.activeAt(ex, ey)) ++cells;
        }
    }
    return 2 * cells - 1;
}

// ---------------------------------------------------------------------------
// Hazard 5: the size-independent cost of one call, measured
// ---------------------------------------------------------------------------

/// @brief The size-independent cost of ONE call of `c`, for both sides.
/// @note PER CASE, not once for the table. The floor is a property of the
///       operation and its element: measured here, cv::erode 3x3 and
///       cv::morphologyEx OPEN 3x3 differ by more than 2x, because OPEN issues
///       two filter calls. Printing one case's floor beside another case's row
///       -- which this benchmark did until T3.3's review -- understates it most
///       for exactly the compound row where the ladder argument matters, and the
///       floor is what X-13's "not cache residency" conclusion rests on.
void measureCallFloors(const Case& c, double& openCvUs, double& binCvUs) {
    Image tiny;
    makeImage(tiny, 2, 2, UINT64_C(0xF100));
    const cv::Mat kernel = cvKernel(c);
    cv::Mat cvDst(2, 2, CV_8U);
    bincv::BinMat<uint32_t> dst(2, 2);
    bincv::BinMat<uint32_t> scratch(2, 2);

    openCvUs = measureNs(
                   [&](int) {
                       runOpenCv(c, tiny.mask, cvDst, kernel);
                       g_sink += cvDst.ptr<uint8_t>(0)[0];
                   },
                   kRepeats, kTargetMs) /
               1000.0;
    binCvUs = measureNs(
                  [&](int) {
                      runBinCv(c, tiny.packed32, dst, scratch);
                      g_sink += dst.ptr(0)[0];
                  },
                  kRepeats, kTargetMs) /
              1000.0;
}

void printCallFloorPreamble() {
    std::printf("\nFIXED PER-CALL COST (hazard 5) is measured on a 2x2 frame FOR EACH CASE and\n"
                "printed on that case's PER-FRAME line -- it is the part of the row that is not\n"
                "the operation. Divided by a frame's pixel count it is what a small frame's\n"
                "ns/pixel is mostly made of. It is NOT subtracted from the tables.\n");
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
                  UINT64_C(0x3033DEC) + static_cast<uint64_t>(i) * UINT64_C(7919));
    }

    bincv::BinMat<uint32_t> dst32(width, height);
    bincv::BinMat<uint64_t> dst64(width, height);
    bincv::BinMat<uint32_t> scratch32(width, height);
    bincv::BinMat<uint64_t> scratch64(width, height);
    bincv::BinMat<uint32_t> composed32(width, height);
    bincv::BinMat<uint32_t> tmp32(width, height);
    cv::Mat cvDst(height, width, CV_8U);

    const double packed32Bytes = static_cast<double>(dst32.sizeInWords() * sizeof(uint32_t));
    const double packed64Bytes = static_cast<double>(dst64.sizeInWords() * sizeof(uint64_t));

    std::printf("  one frame:   binCV uint32 %.0f B, binCV uint64 %.0f B, OpenCV CV_8U %.0f B "
                "(%.1fx smaller)\n",
                packed32Bytes, packed64Bytes, pixels, pixels / packed32Bytes);
    // Hazard 2's cost, stated rather than argued: the timed loop rotates kInputs
    // sources past one destination, so what is RESIDENT is larger than the
    // working set of one call that the tables report.
    std::printf("  timed loop resident (hazard 2, %d rotated inputs + 1 dst, NOT the working "
                "set): binCV uint32 %.0f B, OpenCV CV_8U %.0f B\n",
                kInputs, static_cast<double>(kInputs + 1) * packed32Bytes,
                static_cast<double>(kInputs + 1) * pixels);

    bool ok = true;
    for (const Case& c : cases()) {
        const cv::Mat kernel = cvKernel(c);
        double openCvCallFloorUs = 0.0;
        double binCvCallFloorUs = 0.0;
        measureCallFloors(c, openCvCallFloorUs, binCvCallFloorUs);
        std::printf("\n  --- %s, %s ---\n", c.name, borderName(c.border));
        std::printf("  WORKING SET of one call (the number CLAUDE.md asks for):\n");
        std::printf("    binCV uint32   %10.0f B   (%d frames%s)\n",
                    static_cast<double>(c.binCvBuffers) * packed32Bytes, c.binCvBuffers,
                    c.binCvBuffers == 2 ? ", src + dst, NO scratch"
                                        : ", src + dst + caller scratch");
        std::printf("    binCV uint64   %10.0f B   (%d frames)\n",
                    static_cast<double>(c.binCvBuffers) * packed64Bytes, c.binCvBuffers);
        std::printf("    OpenCV CV_8U   %10.0f B   (%d frames)\n",
                    static_cast<double>(c.openCvBuffers) * pixels, c.openCvBuffers);

        // --- hazard 4: the sides must agree before anything is timed ---------
        runOpenCv(c, images[0].mask, cvDst, kernel);
        runBinCv(c, images[0].packed32, dst32, scratch32);
        runBinCv(c, images[0].packed64, dst64, scratch64);
        const int d32 = comparePixels(dst32, cvDst);
        const int d64 = comparePixels(dst64, cvDst);
        if (d32 != 0 || d64 != 0) {
            std::printf("  RESULTS DISAGREE (%d / %d pixels) -- SKIPPING THIS CASE.\n"
                        "  The implementations do not compute the same image, so no timing of\n"
                        "  them is a comparison.\n",
                        d32, d64);
            ok = false;
            continue;
        }

        struct Row {
            const char* impl;
            double nsPerPixel;
            double workingSet;
            uint64_t checksum;
        };
        std::vector<Row> rows;

        {
            const double ns = measureNs(
                [&](int i) {
                    runOpenCv(c, images[static_cast<size_t>(i % kInputs)].mask, cvDst, kernel);
                    g_sink += cvDst.ptr<uint8_t>(0)[0];
                },
                kRepeats, kTargetMs);
            runOpenCv(c, images[0].mask, cvDst, kernel);
            rows.push_back(Row{"OpenCV CV_8U", ns / pixels,
                               static_cast<double>(c.openCvBuffers) * pixels,
                               pixelChecksum(cvDst)});
        }
        {
            const double ns = measureNs(
                [&](int i) {
                    runBinCv(c, images[static_cast<size_t>(i % kInputs)].packed32, dst32,
                             scratch32);
                    g_sink += dst32.ptr(0)[0];
                },
                kRepeats, kTargetMs);
            runBinCv(c, images[0].packed32, dst32, scratch32);
            rows.push_back(Row{"binCV uint32", ns / pixels,
                               static_cast<double>(c.binCvBuffers) * packed32Bytes,
                               pixelChecksum(dst32)});
        }
        {
            const double ns = measureNs(
                [&](int i) {
                    runBinCv(c, images[static_cast<size_t>(i % kInputs)].packed64, dst64,
                             scratch64);
                    g_sink += dst64.ptr(0)[0];
                },
                kRepeats, kTargetMs);
            runBinCv(c, images[0].packed64, dst64, scratch64);
            rows.push_back(Row{"binCV uint64", ns / pixels,
                               static_cast<double>(c.binCvBuffers) * packed64Bytes,
                               pixelChecksum(dst64)});
        }

        // The composed spelling below shifts with BORDER_CONSTANT, so it is a
        // comparison only for the BORDER_CONSTANT cases. The border-axis cases
        // skip it rather than time two different operations against each other.
        if ((c.op == bincv::MORPH_ERODE || c.op == bincv::MORPH_DILATE) &&
            c.border == bincv::BORDER_CONSTANT) {
            const bool isErode = (c.op == bincv::MORPH_ERODE);
            // Hazard 4 again: the composed spelling must compute the same image
            // before it is timed, or the row is a comparison of two operations.
            if (isErode) {
                runComposed<true>(c.se, images[0].packed32, composed32, tmp32);
            } else {
                runComposed<false>(c.se, images[0].packed32, composed32, tmp32);
            }
            if (comparePixels(composed32, cvDst) != 0) {
                std::printf("  THE COMPOSED SPELLING DISAGREES -- not timed.\n");
                ok = false;
            } else {
                const double ns = measureNs(
                    [&](int i) {
                        const bincv::BinMat<uint32_t>& in =
                            images[static_cast<size_t>(i % kInputs)].packed32;
                        if (isErode) {
                            runComposed<true>(c.se, in, composed32, tmp32);
                        } else {
                            runComposed<false>(c.se, in, composed32, tmp32);
                        }
                        g_sink += composed32.ptr(0)[0];
                    },
                    kRepeats, kTargetMs);
                if (isErode) {
                    runComposed<true>(c.se, images[0].packed32, composed32, tmp32);
                } else {
                    runComposed<false>(c.se, images[0].packed32, composed32, tmp32);
                }
                rows.push_back(Row{"binCV composed u32", ns / pixels, 3.0 * packed32Bytes,
                                   pixelChecksum(composed32)});
                std::printf("  composed spelling: shift + bitwise per cell, %d traversals "
                            "against the fused kernel's 1, and a THIRD frame.\n",
                            composedTraversals(c.se));
            }
        }

        const double reference = rows[0].nsPerPixel;
        std::printf("\n  %-16s %12s %11s %14s %14s %20s\n", "IMPLEMENTATION", "ns/pixel",
                    "vs OpenCV", "working set", "vs OpenCV", "checksum");
        std::printf("  --------------------------------------------------------------------"
                    "--------------------------------\n");
        for (const Row& row : rows) {
            std::printf("  %-16s %12.5f %10.2fx %12.0f B %13.2fx %20llu\n", row.impl,
                        row.nsPerPixel,
                        (row.nsPerPixel > 0.0) ? reference / row.nsPerPixel : 0.0, row.workingSet,
                        (row.workingSet > 0.0) ? rows[0].workingSet / row.workingSet : 0.0,
                        static_cast<unsigned long long>(row.checksum));
        }
        std::printf("  ratio > 1 means binCV is better. All checksums must be EQUAL (hazard 1).\n");

        const double baselineUs = reference * pixels / 1000.0;
        const double binCvUs = rows[1].nsPerPixel * pixels / 1000.0;
        std::printf("  PER-FRAME: OpenCV %.2f us (fixed per-call cost %.2f us, %.0f%%), "
                    "binCV u32 %.2f us (%.2f us, %.0f%%). Both floors are THIS case's.\n",
                    baselineUs, openCvCallFloorUs, 100.0 * openCvCallFloorUs / baselineUs,
                    binCvUs, binCvCallFloorUs, 100.0 * binCvCallFloorUs / binCvUs);
    }

    std::printf("  sink=%llu\n", static_cast<unsigned long long>(g_sink));
    return ok;
}

}  // namespace

int main() {
    std::printf("T3.3 morphology -- erode / dilate / morphologyEx vs OpenCV\n");
    std::printf("================================================================================\n\n");
    std::printf("DENOMINATOR (ARCHITECTURE 10.3): cv::erode / cv::dilate / cv::morphologyEx on\n");
    std::printf("the SAME binary content stored as CV_8U, with the same structuring element,\n");
    std::printf("anchor and border. That is what a user does today without binCV.\n\n");
    std::printf("binCV rows: ops/morphology.hpp at uint32 (the default word type, D-14) and\n");
    std::printf("uint64. erode and dilate use NO scratch; morphologyEx(OPEN) uses exactly one\n");
    std::printf("caller-provided frame.\n\n");
    std::printf("Working set is one call's live buffers, not a per-buffer ratio (CLAUDE.md,\n");
    std::printf("ARCHITECTURE 10.4). Both columns are ratios against the OpenCV row.\n");

    printCallFloorPreamble();

    // The frame and the pyramid ladder below it: the sizes a VIO frontend runs
    // morphology at, and the ones where the whole working set moves in and out of
    // a Cortex-A cache level.
    const int sizes[][2] = {{640, 480}, {320, 240}, {160, 120}, {94, 60}};

    bool ok = true;
    for (const auto& size : sizes) {
        if (!runSize(size[0], size[1])) ok = false;
    }

    std::printf("\n");
    if (!ok) {
        std::printf("AT LEAST ONE CASE WAS NOT MEASURED CLEANLY -- see above.\n");
        return 1;
    }
    return 0;
}
