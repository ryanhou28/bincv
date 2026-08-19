// Morphology (T3.3): erode / dilate / morphologyEx, and the StructuringElement
// they take.
//
// TWO HALVES, the same split as tests/test_shift.cpp and tests/test_reduce.cpp.
//
//   1. The CORE half needs no OpenCV, so it runs in all four verification
//      configurations -- including Debug, the only place the kernels'
//      BINCV_ASSERT preconditions are live, and -fno-exceptions, which is the
//      embedded claim. It checks the kernel against a PER-PIXEL reference written
//      in terms of coordinates and an INDEPENDENTLY WRITTEN border mapping (the
//      do-while shape OpenCV documents, not the library's closed form), against
//      ops/shift.hpp for the single-cell case, and the 3x3 special case against
//      the general path it replaced.
//
//   2. The OPENCV half is the Tier 1 promise: bit-exact against cv::erode,
//      cv::dilate and cv::morphologyEx -- and, separately, StructuringElement
//      cell-for-cell against cv::getStructuringElement, because an element that
//      is not OpenCV's element makes every image comparison a comparison of two
//      different operations.
//
// ---------------------------------------------------------------------------
// WHY THERE ARE ASYMMETRIC ELEMENTS HERE, MEASURED
//
// All three parametric shapes are point-symmetric about their centre, so at a
// centred anchor the offset set E satisfies E == -E and NEGATING EVERY OFFSET
// CHANGES NOTHING. A suite built only from rect / cross / ellipse at the default
// anchor therefore cannot see an inverted shift sign. That is not a worry, it is
// a measurement: with `dx = ex - anchorX` inverted to `anchorX - ex` in both the
// general and the 3x3 path, a 5040-case sweep of the three shapes at centred
// anchors passed 5040/5040, while the same sweep with off-centre anchors and the
// three custom masks below failed 3803 of 5040. Both routes to asymmetry are
// swept here for that reason.
//
// ---------------------------------------------------------------------------
// WHERE THE BORDER IS CHECKED
//
// Everywhere, and deliberately not only through whole images. cv::erode and
// cv::dilate default to BORDER_CONSTANT with morphologyDefaultBorderValue(),
// which is NOT the same constant for the two (D-12) -- ones outside for an
// erosion, zeros for a dilation. The sweeps below run every size in T2.1's
// matrix including width 1 and height 1, where EVERY pixel is a border pixel, and
// the non-constant types are swept separately because they are the only path
// through the per-pixel fixup.
//
// BORDER_WRAP is core-only on purpose: cv::morphologyEx REFUSES it
// ("columnBorderType != BORDER_WRAP" is an OpenCV assertion), so there is no
// Tier 1 denominator to compare against. binCV supports it, and it is checked
// against this file's per-pixel reference and against ops/shift.hpp instead.
//
// CHECK ACCOUNTING: the core half reports one check per (case, element, op) with
// its DISAGREEMENT COUNT, so CHECKS tracks cases rather than pixels and a failure
// still says how badly it failed. The OpenCV half goes through
// BINCV_EXPECT_BIT_EXACT, which is two checks (pixels, then padding) per
// comparison whatever the image size.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/morphology.hpp"
#include "bincv-cpp/ops/shift.hpp"
#include "test_util.hpp"

// OpenCV-only, and self-guarding: everything in it sits behind BINCV_WITH_OPENCV.
#include "equivalence.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

using bincv::BinMat;
using bincv::BorderType;
using bincv::MorphOp;
using bincv::MorphShape;
using bincv::StructuringElement;

/// @def MORPH_EXPECT
/// @brief One check, with a detail string built only when it fails.
#define MORPH_EXPECT(ok, what, detailExpr) \
    ::bincv::test::reportCheck((ok), (what), __FILE__, __LINE__, \
                               (ok) ? std::string() : (detailExpr))

// ---------------------------------------------------------------------------
// Content: the same generator as tests/equivalence.hpp, minus OpenCV
// ---------------------------------------------------------------------------
//
// Duplicated rather than shared, for T2.1's reason: a harness that shared a
// generator with the suite judging it could cancel a fault through both sides.

uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

constexpr uint32_t FILL_SCALE = 1u << 24;

uint32_t fillThreshold(float fillRatio) {
    if (!(fillRatio > 0.0f)) return 0;
    if (fillRatio >= 1.0f) return FILL_SCALE;
    const double rounded = static_cast<double>(fillRatio) * static_cast<double>(FILL_SCALE) + 0.5;
    if (rounded >= static_cast<double>(FILL_SCALE)) return FILL_SCALE;
    return static_cast<uint32_t>(rounded);
}

/// @brief Fills through set(), so the padding bits stay clear on entry.
template <typename WordType>
void fillRandom(BinMat<WordType>& m, float fillRatio, uint64_t seed) {
    uint64_t state = seed;
    const uint32_t threshold = fillThreshold(fillRatio);
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            if (static_cast<uint32_t>(nextRandom(state) >> 40) < threshold) m.set(y, x, true);
        }
    }
}

uint64_t caseSeed(int width, int height, size_t index) {
    return UINT64_C(0x3033000000001) + static_cast<uint64_t>(width) * UINT64_C(1000003) +
           static_cast<uint64_t>(height) * UINT64_C(10007) + static_cast<uint64_t>(index);
}

// ---------------------------------------------------------------------------
// The border mapping, written a SECOND time
// ---------------------------------------------------------------------------

/// @brief cv::borderInterpolate as OpenCV writes it: a loop, not a closed form.
/// @note The library uses a signed modulo into the reflect pattern's period
///       (impl::borderIndex). Two different algorithms for one function is the
///       point: the core half of this suite must be able to fail when that closed
///       form is wrong, and it cannot be if it judges itself.
int borderIndexReference(int p, int len, BorderType type) {
    if (p >= 0 && p < len) return p;
    if (type == bincv::BORDER_REPLICATE) return p < 0 ? 0 : len - 1;
    if (type == bincv::BORDER_REFLECT || type == bincv::BORDER_REFLECT_101) {
        if (len == 1) return 0;
        const int delta = (type == bincv::BORDER_REFLECT_101) ? 1 : 0;
        do {
            if (p < 0) {
                p = -p - 1 + delta;
            } else {
                p = len - 1 - (p - len) - delta;
            }
        } while (p < 0 || p >= len);
        return p;
    }
    if (type == bincv::BORDER_WRAP) {
        while (p < 0) p += len;
        return p % len;
    }
    return -1;  // BORDER_CONSTANT
}

// ---------------------------------------------------------------------------
// The element catalogue
// ---------------------------------------------------------------------------
//
// The three custom masks are ASYMMETRIC -- for each, negating every offset gives
// a different set. See the measurement in this file's header comment.

const uint8_t kMaskL[9] = {1, 0, 0,
                           1, 0, 0,
                           1, 1, 1};

// A wedge, NOT a plain diagonal. The first version of this mask was the 5x5 main
// diagonal, and Morphology.ElementStructure rejected it: a diagonal is invariant
// under a 180-degree rotation, so its offset set satisfies E == -E at a centred
// anchor and it belongs in the SYMMETRIC catalogue. That check exists precisely
// so a mask that looks asymmetric cannot be filed as one.
const uint8_t kMaskWedge[25] = {0, 0, 0, 0, 1,
                                0, 0, 0, 1, 0,
                                0, 0, 1, 0, 0,
                                0, 1, 1, 0, 0,
                                1, 1, 1, 0, 0};

const uint8_t kMaskHook[15] = {0, 1, 1, 1, 0,
                               0, 0, 0, 1, 0,
                               0, 0, 0, 1, 0};

struct NamedElement {
    const char* name;
    StructuringElement se;
};

/// @brief The symmetric catalogue: the three shapes at the default anchor.
const std::vector<NamedElement>& symmetricElements() {
    static const std::vector<NamedElement> v{
        {"rect3x3", StructuringElement::rect(3, 3)},
        {"cross3x3", StructuringElement::cross(3, 3)},
        {"ellipse3x3", StructuringElement::ellipse(3, 3)},
        {"rect5x3", StructuringElement::rect(5, 3)},
        {"cross5x5", StructuringElement::cross(5, 5)},
        {"ellipse7x5", StructuringElement::ellipse(7, 5)},
        {"rect1x1", StructuringElement::rect(1, 1)},
    };
    return v;
}

/// @brief The asymmetric catalogue: off-centre anchors and hand-written masks.
const std::vector<NamedElement>& asymmetricElements() {
    static const std::vector<NamedElement> v{
        {"rect3x3@(0,0)", StructuringElement::rect(3, 3, 0, 0)},
        {"cross5x5@(4,1)", StructuringElement::cross(5, 5, 4, 1)},
        {"ellipse5x5@(0,4)", StructuringElement::ellipse(5, 5, 0, 4)},
        {"maskL3x3", StructuringElement::custom(kMaskL, 3, 3)},
        {"maskL3x3@(0,2)", StructuringElement::custom(kMaskL, 3, 3, 0, 2)},
        {"maskWedge5x5", StructuringElement::custom(kMaskWedge, 5, 5)},
        {"maskHook5x3@(3,2)", StructuringElement::custom(kMaskHook, 5, 3, 3, 2)},
    };
    return v;
}

const std::vector<MorphOp>& allOps() {
    static const std::vector<MorphOp> v{bincv::MORPH_ERODE,    bincv::MORPH_DILATE,
                                        bincv::MORPH_OPEN,     bincv::MORPH_CLOSE,
                                        bincv::MORPH_GRADIENT, bincv::MORPH_TOPHAT,
                                        bincv::MORPH_BLACKHAT};
    return v;
}

const char* opName(MorphOp op) {
    switch (op) {
        case bincv::MORPH_ERODE: return "ERODE";
        case bincv::MORPH_DILATE: return "DILATE";
        case bincv::MORPH_OPEN: return "OPEN";
        case bincv::MORPH_CLOSE: return "CLOSE";
        case bincv::MORPH_GRADIENT: return "GRADIENT";
        case bincv::MORPH_TOPHAT: return "TOPHAT";
        default: return "BLACKHAT";
    }
}

const char* borderName(BorderType t) {
    switch (t) {
        case bincv::BORDER_CONSTANT: return "CONSTANT";
        case bincv::BORDER_REPLICATE: return "REPLICATE";
        case bincv::BORDER_REFLECT: return "REFLECT";
        case bincv::BORDER_WRAP: return "WRAP";
        default: return "REFLECT_101";
    }
}

#ifdef BINCV_WITH_OPENCV
/// @brief Border types with an OpenCV morphology denominator -- WRAP is absent.
/// @note Guarded because it names OpenCV's restriction and nothing else uses it;
///       an unused function is a -Werror failure in the core configurations.
const std::vector<BorderType>& openCvBorderTypes() {
    static const std::vector<BorderType> v{bincv::BORDER_CONSTANT, bincv::BORDER_REPLICATE,
                                           bincv::BORDER_REFLECT, bincv::BORDER_REFLECT_101};
    return v;
}
#endif  // BINCV_WITH_OPENCV

const std::vector<BorderType>& allBorderTypes() {
    static const std::vector<BorderType> v{bincv::BORDER_CONSTANT, bincv::BORDER_REPLICATE,
                                           bincv::BORDER_REFLECT, bincv::BORDER_REFLECT_101,
                                           bincv::BORDER_WRAP};
    return v;
}

// The reduced sweep. The full T2.1 matrix is used where one element and two ops
// are swept over it; the combinatorial cases (7 ops x 7 elements x 5 borders)
// use this instead, which keeps every packing-sensitive width and drops only the
// two dimensions that cost seconds and cannot expose a word-boundary bug.
const std::vector<int>& reducedWidths() {
    static const std::vector<int> v{1, 7, 31, 33, 40, 63, 65, 70};
    return v;
}
const std::vector<int>& reducedHeights() {
    static const std::vector<int> v{1, 2, 3, 17};
    return v;
}
const std::vector<float>& reducedFills() {
    static const std::vector<float> v{0.0f, 0.5f, 1.0f};
    return v;
}

std::string label(const char* wordTypeName, int width, int height, const char* extra) {
    return std::string(wordTypeName) + " " + std::to_string(width) + "x" +
           std::to_string(height) + " " + extra;
}

// ---------------------------------------------------------------------------
// The per-pixel reference
// ---------------------------------------------------------------------------

/// @brief erode / dilate at ONE pixel, written in coordinates.
/// @note Shares nothing with the kernel but the element itself: the fold is a
///       bool, the neighbourhood is a double loop over cells, and the border goes
///       through borderIndexReference() above rather than through impl::.
template <typename WordType>
bool referencePixel(const BinMat<WordType>& src, int x, int y, const StructuringElement& se,
                    BorderType borderType, bool borderValue, bool isErode) {
    const int ax = se.anchorCol();
    const int ay = se.anchorRow();
    bool acc = isErode;
    for (int ey = 0; ey < se.rows; ++ey) {
        for (int ex = 0; ex < se.cols; ++ex) {
            if (!se.activeAt(ex, ey)) continue;
            const int sy = borderIndexReference(y + (ey - ay), src.rows(), borderType);
            const int sx = borderIndexReference(x + (ex - ax), src.cols(), borderType);
            const bool v = (sy < 0 || sx < 0) ? borderValue : src.at(sy, sx);
            acc = isErode ? (acc && v) : (acc || v);
        }
    }
    return acc;
}

/// @brief The whole reference image, into a BinMat.
template <typename WordType>
void referenceImage(const BinMat<WordType>& src, BinMat<WordType>& dst,
                    const StructuringElement& se, BorderType borderType, bool borderValue,
                    bool isErode) {
    for (int y = 0; y < src.rows(); ++y) {
        for (int x = 0; x < src.cols(); ++x) {
            dst.set(y, x, referencePixel(src, x, y, se, borderType, borderValue, isErode));
        }
    }
}

template <typename WordType>
int countDisagreements(const BinMat<WordType>& a, const BinMat<WordType>& b) {
    int differing = 0;
    for (int y = 0; y < a.rows(); ++y) {
        for (int x = 0; x < a.cols(); ++x) {
            if (a.at(y, x) != b.at(y, x)) ++differing;
        }
    }
    return differing;
}

/// @brief Set bits across the whole STRIDE, padding included.
/// @note binCV exposes no per-word popcount (D-6), so the test writes its own.
template <typename WordType>
int paddingBitsSet(const BinMat<WordType>& m) {
    constexpr size_t wordBits = BinMat<WordType>::WordBits;
    if (m.empty()) return 0;
    const size_t minWords = (m.getWidth() + wordBits - 1) / wordBits;
    const size_t tailBits = m.getWidth() % wordBits;
    int set = 0;
    for (int y = 0; y < m.rows(); ++y) {
        const WordType* row = m.ptr(y);
        if (tailBits != 0) {
            for (size_t b = tailBits; b < wordBits; ++b) {
                if ((row[minWords - 1] >> b) & 1u) ++set;
            }
        }
        for (size_t w = minWords; w < m.getAlignedWidth(); ++w) {
            for (size_t b = 0; b < wordBits; ++b) {
                if ((row[w] >> b) & 1u) ++set;
            }
        }
    }
    return set;
}

/// @brief Sets every padding bit of every row, through raw row pointers.
template <typename WordType>
void dirtyThePadding(BinMat<WordType>& m) {
    constexpr size_t wordBits = BinMat<WordType>::WordBits;
    if (m.empty()) return;
    const size_t minWords = (m.getWidth() + wordBits - 1) / wordBits;
    const size_t tailBits = m.getWidth() % wordBits;
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));
    for (int y = 0; y < m.rows(); ++y) {
        WordType* row = m.ptr(y);
        if (tailBits != 0) {
            row[minWords - 1] =
                static_cast<WordType>(row[minWords - 1] | static_cast<WordType>(allOnes << tailBits));
        }
        for (size_t w = minWords; w < m.getAlignedWidth(); ++w) row[w] = allOnes;
    }
}

// ---------------------------------------------------------------------------
// CORE 1: the structuring element's own properties
// ---------------------------------------------------------------------------
//
// The ELLIPSE's exact cell set is OpenCV's arithmetic and is checked against the
// real cv::getStructuringElement in the OpenCV half -- there is no independent
// formulation of it, because "OpenCV's rounding" is the specification. What is
// checkable without OpenCV is the structure every shape must have, and it is what
// the kernels' correctness rests on: that spanOfRow() and activeAt() agree, that
// the parametric shapes are point-symmetric at a centred anchor (the property
// that makes the asymmetric catalogue necessary), and that rect and cross match
// their closed forms.

void testElementStructure() {
    const int sizes[] = {1, 3, 5, 7, 9, 15, 31};
    const MorphShape shapes[] = {bincv::MORPH_RECT, bincv::MORPH_CROSS, bincv::MORPH_ELLIPSE};

    for (int c : sizes) {
        for (int r : sizes) {
            for (MorphShape shape : shapes) {
                const StructuringElement se{shape, c, r, -1, -1, nullptr};
                const std::string what = std::string("shape=") + std::to_string(shape) + " " +
                                         std::to_string(c) + "x" + std::to_string(r);

                // spanOfRow is what the kernels iterate; activeAt is what the
                // reference asks. A cell outside the span must be clear, or the
                // kernel skips a cell the reference counts.
                int outsideSpanSet = 0;
                int cells = 0;
                for (int row = 0; row < r; ++row) {
                    int first = 0;
                    int last = 0;
                    se.spanOfRow(row, first, last);
                    for (int col = 0; col < c; ++col) {
                        const bool active = se.activeAt(col, row);
                        if (active) ++cells;
                        if (active && (col < first || col >= last)) ++outsideSpanSet;
                    }
                }
                MORPH_EXPECT(outsideSpanSet == 0, "every set cell lies inside spanOfRow()",
                             what + ": " + std::to_string(outsideSpanSet) + " outside");

                // The CONVERSE, which is what the kernels rely on: a parametric
                // shape's span is SOLID, so the word loop can iterate it without
                // a per-cell test (StructuringElement::spanIsDense()). Without
                // this check that optimisation would be an assumption.
                int holesInSpan = 0;
                for (int row = 0; row < r; ++row) {
                    int first = 0;
                    int last = 0;
                    se.spanOfRow(row, first, last);
                    for (int col = first; col < last; ++col) {
                        if (!se.activeAt(col, row)) ++holesInSpan;
                    }
                }
                MORPH_EXPECT(se.spanIsDense() && holesInSpan == 0,
                             "a parametric shape's spanOfRow() range is solid",
                             what + ": " + std::to_string(holesInSpan) + " holes");
                MORPH_EXPECT(cells > 0 && se.valid(), "the element has at least one set cell",
                             what);

                // Point symmetry at a centred anchor, for odd extents. This is the
                // property that makes a rect/cross/ellipse-only suite unable to
                // see an inverted offset sign.
                if (c % 2 == 1 && r % 2 == 1) {
                    int asymmetric = 0;
                    for (int row = 0; row < r; ++row) {
                        for (int col = 0; col < c; ++col) {
                            if (se.activeAt(col, row) != se.activeAt(c - 1 - col, r - 1 - row)) {
                                ++asymmetric;
                            }
                        }
                    }
                    MORPH_EXPECT(asymmetric == 0, "parametric shapes are point-symmetric",
                                 what + ": " + std::to_string(asymmetric) + " cells");
                }

                // The two closed forms that do not need OpenCV.
                if (shape == bincv::MORPH_RECT) {
                    MORPH_EXPECT(cells == c * r, "MORPH_RECT sets every cell", what);
                }
                if (shape == bincv::MORPH_CROSS && !(c == 1 && r == 1)) {
                    int wrong = 0;
                    for (int row = 0; row < r; ++row) {
                        for (int col = 0; col < c; ++col) {
                            const bool expected = (row == se.anchorRow()) || (col == se.anchorCol());
                            if (se.activeAt(col, row) != expected) ++wrong;
                        }
                    }
                    MORPH_EXPECT(wrong == 0, "MORPH_CROSS is the anchor's row and column", what);
                }
                if (shape == bincv::MORPH_ELLIPSE) {
                    // Bounded by the rect and containing the centre cross's stem.
                    MORPH_EXPECT(cells <= c * r && se.activeAt(c / 2, r / 2),
                                 "MORPH_ELLIPSE is inside its box and contains the centre", what);
                }
            }
        }
    }

    // A 1x1 element is a filled 1x1 whatever the shape says -- OpenCV's rule.
    for (MorphShape shape : shapes) {
        const StructuringElement se{shape, 1, 1, -1, -1, nullptr};
        MORPH_EXPECT(se.activeAt(0, 0), "a 1x1 element is set whatever its shape",
                     std::string("shape=") + std::to_string(shape));
    }

    // -1 resolves to the centre; an explicit anchor is taken as given.
    const StructuringElement centred = StructuringElement::rect(7, 5);
    MORPH_EXPECT(centred.anchorCol() == 3 && centred.anchorRow() == 2,
                 "an anchor of -1 resolves to size / 2", "rect(7,5)");
    const StructuringElement explicitAnchor = StructuringElement::rect(7, 5, 0, 4);
    MORPH_EXPECT(explicitAnchor.anchorCol() == 0 && explicitAnchor.anchorRow() == 4,
                 "an explicit anchor is taken as given", "rect(7,5,0,4)");

    // A cross follows its anchor, which is cv::getStructuringElement's behaviour
    // and the reason the anchor is a field of the element rather than a separate
    // argument to the kernel.
    const StructuringElement offCross = StructuringElement::cross(5, 5, 0, 4);
    MORPH_EXPECT(offCross.activeAt(0, 0) && offCross.activeAt(4, 4) && !offCross.activeAt(4, 0),
                 "MORPH_CROSS is centred on the ANCHOR, not on the element", "cross(5,5,0,4)");

    // The mask overrides the shape and is read row-major.
    const StructuringElement masked = StructuringElement::custom(kMaskL, 3, 3);
    int wrong = 0;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            if (masked.activeAt(col, row) != (kMaskL[row * 3 + col] != 0)) ++wrong;
        }
    }
    MORPH_EXPECT(wrong == 0, "a custom mask is read row-major and overrides the shape", "maskL");

    // The asymmetric catalogue must actually BE asymmetric, or the mutation
    // argument in this file's header comment is about nothing.
    for (const NamedElement& e : asymmetricElements()) {
        const int ax = e.se.anchorCol();
        const int ay = e.se.anchorRow();
        bool differs = false;
        for (int row = 0; row < e.se.rows && !differs; ++row) {
            for (int col = 0; col < e.se.cols && !differs; ++col) {
                if (!e.se.activeAt(col, row)) continue;
                // Is the negated offset also in the element?
                const int mx = 2 * ax - col;
                const int my = 2 * ay - row;
                const bool inside = mx >= 0 && mx < e.se.cols && my >= 0 && my < e.se.rows;
                if (!inside || !e.se.activeAt(mx, my)) differs = true;
            }
        }
        MORPH_EXPECT(differs, "the asymmetric catalogue is asymmetric about its anchor", e.name);
    }
}

// ---------------------------------------------------------------------------
// CORE 2: the kernel against the per-pixel reference
// ---------------------------------------------------------------------------

template <typename WordType>
void testAgainstReference(const char* wordTypeName) {
    size_t index = 0;
    for (int width : reducedWidths()) {
        for (int height : reducedHeights()) {
            for (float fill : reducedFills()) {
                BinMat<WordType> src(width, height);
                fillRandom(src, fill, caseSeed(width, height, index++));

                BinMat<WordType> dst(width, height);
                BinMat<WordType> expected(width, height);

                for (const NamedElement& elements : symmetricElements()) {
                    for (BorderType borderType : allBorderTypes()) {
                        // BOTH fills, but only where a fill is read. D-12 makes
                        // the fill the caller's, so both values are part of the
                        // contract -- and the four non-constant types ignore it
                        // entirely, so sweeping it there doubles the cost of this
                        // case for no case.
                        const int fills = (borderType == bincv::BORDER_CONSTANT) ? 2 : 1;
                        for (int isErode = 0; isErode < 2; ++isErode) {
                            for (int fillBit = 0; fillBit < fills; ++fillBit) {
                                const bool borderValue = fillBit != 0;
                                if (isErode) {
                                    bincv::erode(src.constView(), dst.view(), elements.se,
                                                 borderType, borderValue);
                                } else {
                                    bincv::dilate(src.constView(), dst.view(), elements.se,
                                                  borderType, borderValue);
                                }
                                referenceImage(src, expected, elements.se, borderType, borderValue,
                                               isErode != 0);
                                const int differing = countDisagreements(dst, expected);
                                const int padding = paddingBitsSet(dst);
                                MORPH_EXPECT(differing == 0 && padding == 0,
                                             "erode/dilate matches the per-pixel reference",
                                             label(wordTypeName, width, height,
                                                   (std::string(elements.name) + " " +
                                                    (isErode ? "erode" : "dilate") + " " +
                                                    borderName(borderType) + " fill=" +
                                                    (borderValue ? "1" : "0") + ": " +
                                                    std::to_string(differing) + " pixels, " +
                                                    std::to_string(padding) + " padding bits")
                                                       .c_str()));
                            }
                        }
                    }
                }

                // The asymmetric catalogue, at the morphological default fill.
                for (const NamedElement& elements : asymmetricElements()) {
                    for (BorderType borderType : allBorderTypes()) {
                        for (int isErode = 0; isErode < 2; ++isErode) {
                            const bool borderValue = isErode != 0;
                            if (isErode) {
                                bincv::erode(src.constView(), dst.view(), elements.se, borderType,
                                             borderValue);
                            } else {
                                bincv::dilate(src.constView(), dst.view(), elements.se, borderType,
                                              borderValue);
                            }
                            referenceImage(src, expected, elements.se, borderType, borderValue,
                                           isErode != 0);
                            const int differing = countDisagreements(dst, expected);
                            MORPH_EXPECT(differing == 0,
                                         "an asymmetric element matches the reference",
                                         label(wordTypeName, width, height,
                                               (std::string(elements.name) + " " +
                                                (isErode ? "erode" : "dilate") + " " +
                                                borderName(borderType) + ": " +
                                                std::to_string(differing) + " pixels")
                                                   .c_str()));
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CORE 3: a one-cell element IS a shift
// ---------------------------------------------------------------------------
//
// This is what stands behind the fused kernel's word recurrence. ops/shift.hpp's
// horizontal recurrence and impl::morphShiftedWord are separately written; an
// element with exactly one set cell reduces erode and dilate to
// shift(src, dst, dx, dy, borderType, fill), so requiring the two to agree pixel
// for pixel -- at every offset the element can express, at all five border types,
// and at both fills -- is a direct comparison of the two.

template <typename WordType>
void testSingleOffsetEqualsShift(const char* wordTypeName) {
    // A 7x5 mask with one cell set reaches dx in [-3, 3] and dy in [-2, 2],
    // which crosses a word boundary at every supported width.
    uint8_t mask[35];
    size_t index = 0;

    for (int width : reducedWidths()) {
        for (int height : reducedHeights()) {
            BinMat<WordType> src(width, height);
            fillRandom(src, 0.5f, caseSeed(width, height, 5000 + index++));

            BinMat<WordType> viaMorphology(width, height);
            BinMat<WordType> viaShift(width, height);

            for (int cell = 0; cell < 35; ++cell) {
                for (int i = 0; i < 35; ++i) mask[i] = 0;
                mask[cell] = 1;
                const StructuringElement se = StructuringElement::custom(mask, 7, 5);
                const ptrdiff_t dx = static_cast<ptrdiff_t>(cell % 7) - 3;
                const ptrdiff_t dy = static_cast<ptrdiff_t>(cell / 7) - 2;

                for (BorderType borderType : allBorderTypes()) {
                    for (int isErode = 0; isErode < 2; ++isErode) {
                        const bool borderValue = isErode != 0;
                        if (isErode) {
                            bincv::erode(src.constView(), viaMorphology.view(), se, borderType,
                                         borderValue);
                        } else {
                            bincv::dilate(src.constView(), viaMorphology.view(), se, borderType,
                                          borderValue);
                        }
                        bincv::shift(src.constView(), viaShift.view(), dx, dy, borderType,
                                     borderValue);
                        const int differing = countDisagreements(viaMorphology, viaShift);
                        MORPH_EXPECT(differing == 0,
                                     "a one-cell element is exactly ops/shift.hpp's shift",
                                     label(wordTypeName, width, height,
                                           ("dx=" + std::to_string(dx) + " dy=" +
                                            std::to_string(dy) + " " + borderName(borderType) +
                                            " " + (isErode ? "erode" : "dilate") + ": " +
                                            std::to_string(differing) + " pixels")
                                               .c_str()));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CORE 4: the 3x3 special case against the general path it replaced
// ---------------------------------------------------------------------------

template <typename WordType>
void testFastPathEqualsGeneric(const char* wordTypeName) {
    // Every 3x3 element the special case can be handed, including masks whose
    // cells make it skip work -- and a centred anchor, which is its precondition.
    uint8_t mask[9];
    std::vector<StructuringElement> elements3x3{
        StructuringElement::rect(3, 3), StructuringElement::cross(3, 3),
        StructuringElement::ellipse(3, 3), StructuringElement::custom(kMaskL, 3, 3)};

    size_t index = 0;
    for (int width : reducedWidths()) {
        for (int height : reducedHeights()) {
            BinMat<WordType> src(width, height);
            fillRandom(src, 0.5f, caseSeed(width, height, 9000 + index++));
            BinMat<WordType> fast(width, height);
            BinMat<WordType> generic(width, height);

            std::vector<StructuringElement> cases = elements3x3;
            // Every one-cell 3x3 mask as well: those are the configurations where
            // the fast path's `if (cell[...])` guards each take a different branch.
            for (int cell = 0; cell < 9; ++cell) {
                for (int i = 0; i < 9; ++i) mask[i] = 0;
                mask[cell] = 1;
                cases.push_back(StructuringElement::custom(mask, 3, 3));

                for (BorderType borderType : allBorderTypes()) {
                    for (int isErode = 0; isErode < 2; ++isErode) {
                        const bool borderValue = isErode != 0;
                        const StructuringElement& se = cases.back();
                        if (isErode) {
                            bincv::impl::morphApply<true, bincv::impl::MorphPath::Auto>(
                                src.constView(), fast.view(), se, borderType, borderValue);
                            bincv::impl::morphApply<true, bincv::impl::MorphPath::Generic>(
                                src.constView(), generic.view(), se, borderType, borderValue);
                        } else {
                            bincv::impl::morphApply<false, bincv::impl::MorphPath::Auto>(
                                src.constView(), fast.view(), se, borderType, borderValue);
                            bincv::impl::morphApply<false, bincv::impl::MorphPath::Generic>(
                                src.constView(), generic.view(), se, borderType, borderValue);
                        }
                        const int differing = countDisagreements(fast, generic);
                        MORPH_EXPECT(differing == 0,
                                     "the 3x3 fast path agrees with the general path",
                                     label(wordTypeName, width, height,
                                           ("one-cell mask " + std::to_string(cell) + " " +
                                            borderName(borderType) + ": " +
                                            std::to_string(differing))
                                               .c_str()));
                    }
                }
                cases.pop_back();
            }

            for (const StructuringElement& se : elements3x3) {
                for (BorderType borderType : allBorderTypes()) {
                    for (int isErode = 0; isErode < 2; ++isErode) {
                        const bool borderValue = isErode != 0;
                        if (isErode) {
                            bincv::impl::morphApply<true, bincv::impl::MorphPath::Auto>(
                                src.constView(), fast.view(), se, borderType, borderValue);
                            bincv::impl::morphApply<true, bincv::impl::MorphPath::Generic>(
                                src.constView(), generic.view(), se, borderType, borderValue);
                        } else {
                            bincv::impl::morphApply<false, bincv::impl::MorphPath::Auto>(
                                src.constView(), fast.view(), se, borderType, borderValue);
                            bincv::impl::morphApply<false, bincv::impl::MorphPath::Generic>(
                                src.constView(), generic.view(), se, borderType, borderValue);
                        }
                        const int differing = countDisagreements(fast, generic);
                        MORPH_EXPECT(differing == 0,
                                     "the 3x3 fast path agrees with the general path",
                                     label(wordTypeName, width, height,
                                           (std::string(borderName(borderType)) + ": " +
                                            std::to_string(differing))
                                               .c_str()));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CORE 5: morphologyEx against the compositions it is defined as
// ---------------------------------------------------------------------------

template <typename WordType>
void testCompound(const char* wordTypeName) {
    size_t index = 0;
    for (int width : reducedWidths()) {
        for (int height : reducedHeights()) {
            for (float fill : reducedFills()) {
                BinMat<WordType> src(width, height);
                fillRandom(src, fill, caseSeed(width, height, 13000 + index++));

                BinMat<WordType> dst(width, height);
                BinMat<WordType> scratch(width, height);
                BinMat<WordType> a(width, height);
                BinMat<WordType> b(width, height);
                BinMat<WordType> expected(width, height);

                // A SUBSET, deliberately. Every op is swept over every element
                // and every border type SOMEWHERE -- erode and dilate by
                // Morphology.Reference_*, and all seven ops over all fourteen
                // elements by Morphology.OpenCvOps_* against the real
                // cv::morphologyEx. What this case is the only cover for is the
                // COMPOSITION: the scratch discipline, the argument order of the
                // three subtractions, and BORDER_WRAP (which cv::morphologyEx
                // refuses). Three border types and six elements exercise all of
                // that; the full 14 x 5 cross product ran for ten of this suite's
                // thirteen seconds and added no distinct failure mode.
                const NamedElement compoundCatalogue[] = {
                    {"rect3x3", StructuringElement::rect(3, 3)},
                    {"cross5x5", StructuringElement::cross(5, 5)},
                    {"ellipse7x5", StructuringElement::ellipse(7, 5)},
                    {"rect3x3@(0,0)", StructuringElement::rect(3, 3, 0, 0)},
                    {"maskL3x3", StructuringElement::custom(kMaskL, 3, 3)},
                    {"maskHook5x3@(3,2)", StructuringElement::custom(kMaskHook, 5, 3, 3, 2)},
                };
                const BorderType compoundBorders[] = {bincv::BORDER_CONSTANT,
                                                      bincv::BORDER_REFLECT_101,
                                                      bincv::BORDER_WRAP};

                for (const NamedElement& elements : compoundCatalogue) {
                    for (BorderType borderType : compoundBorders) {
                        for (MorphOp op : allOps()) {
                            bincv::morphologyEx(src.constView(), dst.view(), op, elements.se,
                                                scratch.view(), borderType);

                            // The definition, spelled out with the per-pixel
                            // reference rather than with the kernel: `a` and `b`
                            // are built by referenceImage(), so an erode and a
                            // dilate that were wrong the same way could not agree
                            // their way past this.
                            switch (op) {
                                case bincv::MORPH_ERODE:
                                    referenceImage(src, expected, elements.se, borderType, true,
                                                   true);
                                    break;
                                case bincv::MORPH_DILATE:
                                    referenceImage(src, expected, elements.se, borderType, false,
                                                   false);
                                    break;
                                case bincv::MORPH_OPEN:
                                    referenceImage(src, a, elements.se, borderType, true, true);
                                    referenceImage(a, expected, elements.se, borderType, false,
                                                   false);
                                    break;
                                case bincv::MORPH_CLOSE:
                                    referenceImage(src, a, elements.se, borderType, false, false);
                                    referenceImage(a, expected, elements.se, borderType, true,
                                                   true);
                                    break;
                                case bincv::MORPH_GRADIENT:
                                    referenceImage(src, a, elements.se, borderType, false, false);
                                    referenceImage(src, b, elements.se, borderType, true, true);
                                    for (int y = 0; y < height; ++y) {
                                        for (int x = 0; x < width; ++x) {
                                            expected.set(y, x, a.at(y, x) && !b.at(y, x));
                                        }
                                    }
                                    break;
                                case bincv::MORPH_TOPHAT:
                                    referenceImage(src, a, elements.se, borderType, true, true);
                                    referenceImage(a, b, elements.se, borderType, false, false);
                                    for (int y = 0; y < height; ++y) {
                                        for (int x = 0; x < width; ++x) {
                                            expected.set(y, x, src.at(y, x) && !b.at(y, x));
                                        }
                                    }
                                    break;
                                default:  // MORPH_BLACKHAT
                                    referenceImage(src, a, elements.se, borderType, false, false);
                                    referenceImage(a, b, elements.se, borderType, true, true);
                                    for (int y = 0; y < height; ++y) {
                                        for (int x = 0; x < width; ++x) {
                                            expected.set(y, x, b.at(y, x) && !src.at(y, x));
                                        }
                                    }
                                    break;
                            }

                            const int differing = countDisagreements(dst, expected);
                            const int padding = paddingBitsSet(dst);
                            MORPH_EXPECT(differing == 0 && padding == 0,
                                         "morphologyEx matches its definition",
                                         label(wordTypeName, width, height,
                                               (std::string(elements.name) + " " + opName(op) +
                                                " " + borderName(borderType) + ": " +
                                                std::to_string(differing) + " pixels, " +
                                                std::to_string(padding) + " padding bits")
                                                   .c_str()));
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CORE 6: padding, dirty sources, degenerate shapes
// ---------------------------------------------------------------------------

template <typename WordType>
void testPaddingAndDegenerate(const char* wordTypeName) {
    // A source whose padding bits are all ones is a SUPPORTED construction
    // (BinMat's wrap constructor, D-13's neighbours). It must not change a pixel.
    size_t index = 0;
    for (int width : reducedWidths()) {
        for (int height : reducedHeights()) {
            BinMat<WordType> clean(width, height);
            fillRandom(clean, 0.5f, caseSeed(width, height, 21000 + index++));
            BinMat<WordType> dirty = clean;
            dirtyThePadding(dirty);

            BinMat<WordType> fromClean(width, height);
            BinMat<WordType> fromDirty(width, height);

            for (const NamedElement& elements : symmetricElements()) {
                for (int isErode = 0; isErode < 2; ++isErode) {
                    const bool borderValue = isErode != 0;
                    if (isErode) {
                        bincv::erode(clean.constView(), fromClean.view(), elements.se,
                                     bincv::BORDER_CONSTANT, borderValue);
                        bincv::erode(dirty.constView(), fromDirty.view(), elements.se,
                                     bincv::BORDER_CONSTANT, borderValue);
                    } else {
                        bincv::dilate(clean.constView(), fromClean.view(), elements.se,
                                      bincv::BORDER_CONSTANT, borderValue);
                        bincv::dilate(dirty.constView(), fromDirty.view(), elements.se,
                                      bincv::BORDER_CONSTANT, borderValue);
                    }
                    const int differing = countDisagreements(fromClean, fromDirty);
                    const int padding = paddingBitsSet(fromDirty);
                    MORPH_EXPECT(differing == 0 && padding == 0,
                                 "a source's dirty padding changes no pixel, and dst stays clean",
                                 label(wordTypeName, width, height,
                                       (std::string(elements.name) + ": " +
                                        std::to_string(differing) + " pixels, " +
                                        std::to_string(padding) + " padding bits")
                                           .c_str()));
                }
            }
        }
    }

    // WIDE elements, for two reasons at once.
    //
    //   * Every offset leaves the frame, which is what a kernel that clamps an
    //     offset to the image gets wrong.
    //   * An element whose horizontal reach is a WHOLE WORD or more is the only
    //     thing that reaches morphRowGeneric's fallback recurrence -- the
    //     sliding three-word window cannot express a shift of a word or more. The
    //     sizes below cross that threshold at every supported word width: reach 4
    //     at uint8_t, 16 at uint16_t, 32 at uint32_t and 64 at uint64_t. Without
    //     129 here the fallback would never run at uint64_t in any configuration.
    //
    // THE ANCHOR SWEEP IS NOT DECORATION, and it is a measurement. A wide rect at
    // the DEFAULT anchor is point-symmetric -- its offset set is {-size/2 .. +size/2},
    // which satisfies E == -E -- and so was the sparse wide mask below, whose cells
    // sat at {0, half, cols-1} around an anchor at half. Those were the only two
    // families reaching the fallback recurrence, so `morphShiftedWord`'s offset
    // sign was the one offset site in the file with NO discriminating case:
    // measured, inserting `dx = -dx;` as its first statement left the whole suite
    // at 298541/298541 while the same mutation at any of the other three sites
    // went red. Anchoring these rects at column 0 and at column `size - 1` makes
    // the offset set {0..size-1} and {-(size-1)..0}, neither of which is its own
    // negation, and the mutation fails. That is the same reasoning the asymmetric
    // catalogue rests on, applied to the one path it had not reached.
    const int wideSizes[] = {9, 33, 65, 129};
    const int wideDims[][2] = {{3, 2}, {70, 3}};
    for (const auto& dims : wideDims) {
        for (int size : wideSizes) {
            BinMat<WordType> src(dims[0], dims[1]);
            fillRandom(src, 0.5f, caseSeed(dims[0], dims[1], 31000 + static_cast<size_t>(size)));
            BinMat<WordType> dst(dims[0], dims[1]);
            BinMat<WordType> expected(dims[0], dims[1]);
            // -1 is OpenCV's "centre"; 0 and size - 1 are the asymmetric ones.
            const int wideAnchors[] = {-1, 0, size - 1};
            for (int anchorX : wideAnchors) {
                const StructuringElement se = StructuringElement::rect(size, 3, anchorX, 1);
                for (BorderType borderType : allBorderTypes()) {
                    for (int isErode = 0; isErode < 2; ++isErode) {
                        const bool borderValue = isErode != 0;
                        if (isErode) {
                            bincv::erode(src.constView(), dst.view(), se, borderType, borderValue);
                        } else {
                            bincv::dilate(src.constView(), dst.view(), se, borderType,
                                          borderValue);
                        }
                        referenceImage(src, expected, se, borderType, borderValue, isErode != 0);
                        MORPH_EXPECT(
                            countDisagreements(dst, expected) == 0 && paddingBitsSet(dst) == 0,
                            "a wide element -- the fallback recurrence -- is still exact",
                            label(wordTypeName, dims[0], dims[1],
                                  (std::to_string(size) + "x3 anchorX=" +
                                   std::to_string(anchorX) + " " + borderName(borderType) + " " +
                                   (isErode ? "erode" : "dilate"))
                                      .c_str()));
                    }
                }
            }
        }
    }

    // A SPARSE wide element, which is the only thing that discriminates the
    // fallback recurrence.
    //
    // Measured, and the reason this case exists as well as the solid rects above:
    // with morphShiftedWord's offset perturbed by +1, the whole suite still passed
    // 298381/298381. A solid 129-wide rect over a 70-wide frame cannot see that --
    // sliding a contiguous offset range by one changes only which two columns fall
    // OUTSIDE the image, and those read the border either way. Two cells at the
    // extremes of a wide span do see it. The half-widths below straddle the
    // window's threshold at every word width: 5 at uint8_t, 17 at uint16_t, 33 at
    // uint32_t, 65 at uint64_t.
    {
        const int halfWidths[] = {5, 17, 33, 65};
        BinMat<WordType> src(140, 3);
        fillRandom(src, 0.5f, caseSeed(140, 3, 41000));
        BinMat<WordType> dst(140, 3);
        BinMat<WordType> expected(140, 3);
        for (int half : halfWidths) {
            const int cols = 2 * half + 1;
            std::vector<uint8_t> mask(static_cast<size_t>(cols), 0);
            mask[0] = 1;
            mask[static_cast<size_t>(cols - 1)] = 1;
            mask[static_cast<size_t>(half)] = 1;
            const StructuringElement se = StructuringElement::custom(mask.data(), cols, 1);
            for (BorderType borderType : allBorderTypes()) {
                for (int isErode = 0; isErode < 2; ++isErode) {
                    const bool borderValue = isErode != 0;
                    if (isErode) {
                        bincv::erode(src.constView(), dst.view(), se, borderType, borderValue);
                    } else {
                        bincv::dilate(src.constView(), dst.view(), se, borderType, borderValue);
                    }
                    referenceImage(src, expected, se, borderType, borderValue, isErode != 0);
                    MORPH_EXPECT(countDisagreements(dst, expected) == 0 && paddingBitsSet(dst) == 0,
                                 "a sparse wide element -- the fallback recurrence -- is exact",
                                 label(wordTypeName, 140, 3,
                                       ("half=" + std::to_string(half) + " " +
                                        borderName(borderType) + " " +
                                        (isErode ? "erode" : "dilate"))
                                           .c_str()));
                }
            }
        }
    }

    // Empty views are a no-op, not a crash.
    {
        bincv::BinMatConstView<WordType> emptySrc{};
        bincv::BinMatView<WordType> emptyDst{};
        bincv::erode(emptySrc, emptyDst, StructuringElement::rect(3, 3));
        bincv::dilate(emptySrc, emptyDst, StructuringElement::rect(3, 3));
        bincv::morphologyEx(emptySrc, emptyDst, bincv::MORPH_OPEN, StructuringElement::rect(3, 3),
                            emptyDst);
        MORPH_EXPECT(true, "an empty view is a no-op", "");
    }

    // morphologyExNeedsScratch is the contract callers size a buffer from.
    MORPH_EXPECT(!bincv::morphologyExNeedsScratch(bincv::MORPH_ERODE) &&
                     !bincv::morphologyExNeedsScratch(bincv::MORPH_DILATE) &&
                     bincv::morphologyExNeedsScratch(bincv::MORPH_OPEN) &&
                     bincv::morphologyExNeedsScratch(bincv::MORPH_CLOSE) &&
                     bincv::morphologyExNeedsScratch(bincv::MORPH_GRADIENT) &&
                     bincv::morphologyExNeedsScratch(bincv::MORPH_TOPHAT) &&
                     bincv::morphologyExNeedsScratch(bincv::MORPH_BLACKHAT),
                 "morphologyExNeedsScratch names exactly the five compound ops", "");

    // ERODE and DILATE ignore the scratch view entirely -- an empty one is legal,
    // which is what makes the predicate above usable.
    {
        BinMat<WordType> src(37, 5);
        fillRandom(src, 0.5f, 0xA11CEu);
        BinMat<WordType> viaEx(37, 5);
        BinMat<WordType> direct(37, 5);
        bincv::BinMatView<WordType> noScratch{};
        bincv::morphologyEx(src.constView(), viaEx.view(), bincv::MORPH_ERODE,
                            StructuringElement::rect(3, 3), noScratch);
        bincv::erode(src.constView(), direct.view(), StructuringElement::rect(3, 3));
        MORPH_EXPECT(countDisagreements(viaEx, direct) == 0,
                     "morphologyEx(MORPH_ERODE) needs no scratch and equals erode()",
                     wordTypeName);
    }
}

// ---------------------------------------------------------------------------
// The OpenCV half -- the Tier 1 promise
// ---------------------------------------------------------------------------

#ifdef BINCV_WITH_OPENCV

/// @brief The element as OpenCV would build it, so the two sides run one element.
cv::Mat toCvKernel(const StructuringElement& se) {
    if (se.mask != nullptr) {
        cv::Mat kernel(se.rows, se.cols, CV_8U);
        for (int row = 0; row < se.rows; ++row) {
            for (int col = 0; col < se.cols; ++col) {
                kernel.at<uint8_t>(row, col) = se.activeAt(col, row) ? 1 : 0;
            }
        }
        return kernel;
    }
    return cv::getStructuringElement(static_cast<int>(se.shape), cv::Size(se.cols, se.rows),
                                     cv::Point(se.anchorX, se.anchorY));
}

cv::Point toCvAnchor(const StructuringElement& se) {
    return cv::Point(se.anchorX, se.anchorY);
}

/// @brief cv::getStructuringElement, cell for cell.
/// @note Without this, every image comparison below could be two implementations
///       agreeing about the wrong element. The parametric shapes' exact cell sets
///       -- MORPH_ELLIPSE's rounding above all -- have no specification other than
///       what OpenCV computes, so this is where they are pinned.
void testElementMatchesOpenCv() {
    const int sizes[] = {1, 2, 3, 4, 5, 7, 9, 11, 15, 21, 31};
    const MorphShape shapes[] = {bincv::MORPH_RECT, bincv::MORPH_CROSS, bincv::MORPH_ELLIPSE};

    for (int c : sizes) {
        for (int r : sizes) {
            for (MorphShape shape : shapes) {
                // The default anchor and two off-centre ones; MORPH_CROSS's cell
                // set depends on the anchor, so this is not a redundant sweep.
                const cv::Point anchors[] = {cv::Point(-1, -1), cv::Point(0, 0),
                                             cv::Point(c - 1, r - 1), cv::Point(c / 2, 0)};
                for (const cv::Point& anchor : anchors) {
                    const StructuringElement se{shape, c, r, anchor.x, anchor.y, nullptr};
                    const cv::Mat kernel =
                        cv::getStructuringElement(static_cast<int>(shape), cv::Size(c, r), anchor);
                    int wrong = 0;
                    for (int row = 0; row < r; ++row) {
                        for (int col = 0; col < c; ++col) {
                            if (se.activeAt(col, row) != (kernel.at<uint8_t>(row, col) != 0)) {
                                ++wrong;
                            }
                        }
                    }
                    MORPH_EXPECT(wrong == 0, "StructuringElement matches cv::getStructuringElement",
                                 std::string("shape=") + std::to_string(shape) + " " +
                                     std::to_string(c) + "x" + std::to_string(r) + " anchor=(" +
                                     std::to_string(anchor.x) + "," + std::to_string(anchor.y) +
                                     "): " + std::to_string(wrong) + " cells");
                }
            }
        }
    }
}

/// @brief One binCV result against one cv::morphologyEx result, bit for bit.
template <typename WordType>
void expectMatchesOpenCv(const BinMat<WordType>& src, const cv::Mat& cvSrc,
                         const StructuringElement& se, MorphOp op, BorderType borderType,
                         const std::string& context) {
    BinMat<WordType> dst(src.cols(), src.rows());
    BinMat<WordType> scratch(src.cols(), src.rows());
    bincv::morphologyEx(src.constView(), dst.view(), op, se, scratch.view(), borderType);

    cv::Mat expected;
    cv::morphologyEx(cvSrc, expected, static_cast<int>(op), toCvKernel(se), toCvAnchor(se), 1,
                     static_cast<int>(borderType), cv::morphologyDefaultBorderValue());

    BINCV_EXPECT_BIT_EXACT(dst.constView(), expected, context);
}

/// @brief The full T2.1 matrix: the three 3x3 shapes, erode and dilate.
/// @note Through cv::erode / cv::dilate DIRECTLY rather than through
///       cv::morphologyEx, because those are the two functions ARCHITECTURE 5.1
///       names as Tier 1 and their default borderValue is the premise D-12 rests
///       on. Content is generated ONCE per (size, fill) and reused across the six
///       (shape, op) combinations -- the generator and the comparison are both
///       per-pixel loops, and at 640x480 they, not OpenCV, are the cost.
template <typename WordType>
void testOpenCvErodeDilate(const char* wordTypeName) {
    uint64_t seed = 0x3033ull;
    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : bincv::test::equivalenceHeights()) {
            for (float fill : bincv::test::equivalenceFillRatios()) {
                const BinMat<WordType> src =
                    bincv::test::randomBinary<WordType>(width, height, fill, ++seed);
                const cv::Mat cvSrc = bincv::test::toCvMask(src.constView());
                BinMat<WordType> dst(width, height);

                for (const NamedElement& elements : symmetricElements()) {
                    if (elements.se.cols != 3 || elements.se.rows != 3) continue;
                    const cv::Mat kernel = toCvKernel(elements.se);

                    bincv::erode(src.constView(), dst.view(), elements.se);
                    cv::Mat expectedErode;
                    cv::erode(cvSrc, expectedErode, kernel);
                    BINCV_EXPECT_BIT_EXACT(
                        dst.constView(), expectedErode,
                        bincv::test::caseLabel(wordTypeName, width, height, fill) + " erode " +
                            elements.name);

                    bincv::dilate(src.constView(), dst.view(), elements.se);
                    cv::Mat expectedDilate;
                    cv::dilate(cvSrc, expectedDilate, kernel);
                    BINCV_EXPECT_BIT_EXACT(
                        dst.constView(), expectedDilate,
                        bincv::test::caseLabel(wordTypeName, width, height, fill) + " dilate " +
                            elements.name);
                }
            }
        }
    }
}

/// @brief Every MorphOp and every element, over the reduced matrix.
template <typename WordType>
void testOpenCvOps(const char* wordTypeName) {
    uint64_t seed = 0x30330000ull;
    std::vector<NamedElement> catalogue = symmetricElements();
    for (const NamedElement& e : asymmetricElements()) catalogue.push_back(e);

    for (int width : reducedWidths()) {
        for (int height : reducedHeights()) {
            for (float fill : reducedFills()) {
                const BinMat<WordType> src =
                    bincv::test::randomBinary<WordType>(width, height, fill, ++seed);
                const cv::Mat cvSrc = bincv::test::toCvMask(src.constView());

                for (const NamedElement& elements : catalogue) {
                    for (MorphOp op : allOps()) {
                        expectMatchesOpenCv(src, cvSrc, elements.se, op, bincv::BORDER_CONSTANT,
                                            bincv::test::caseLabel(wordTypeName, width, height,
                                                                   fill) +
                                                " " + elements.name + " " + opName(op));
                    }
                }
            }
        }
    }
}

/// @brief The four border types cv::morphologyEx accepts, over the reduced matrix.
/// @note This is the path through the per-pixel fixup, and it is swept at width 1
///       and height 1 as well -- the sizes where EVERY pixel is a border pixel and
///       an interior-only implementation still passes an interior-only test.
template <typename WordType>
void testOpenCvBorders(const char* wordTypeName) {
    uint64_t seed = 0x303300000000ull;
    const NamedElement borderCatalogue[] = {
        {"rect3x3", StructuringElement::rect(3, 3)},
        {"ellipse7x5", StructuringElement::ellipse(7, 5)},
        {"maskL3x3@(0,2)", StructuringElement::custom(kMaskL, 3, 3, 0, 2)},
        {"maskWedge5x5", StructuringElement::custom(kMaskWedge, 5, 5)},
    };

    for (int width : reducedWidths()) {
        for (int height : reducedHeights()) {
            for (float fill : reducedFills()) {
                const BinMat<WordType> src =
                    bincv::test::randomBinary<WordType>(width, height, fill, ++seed);
                const cv::Mat cvSrc = bincv::test::toCvMask(src.constView());

                for (const NamedElement& elements : borderCatalogue) {
                    for (BorderType borderType : openCvBorderTypes()) {
                        for (MorphOp op : {bincv::MORPH_ERODE, bincv::MORPH_DILATE,
                                           bincv::MORPH_GRADIENT}) {
                            expectMatchesOpenCv(
                                src, cvSrc, elements.se, op, borderType,
                                bincv::test::caseLabel(wordTypeName, width, height, fill) + " " +
                                    elements.name + " " + opName(op) + " " +
                                    borderName(borderType));
                        }
                    }
                }
            }
        }
    }
}

/// @brief D-12 from binCV's side: the fill is the caller's, and the default is
///        the one OpenCV picks for the operation.
/// @note tests/test_shift.cpp pins the premise about cv::erode. This pins that
///       binCV's DEFAULTS follow it, and that passing the other value reproduces
///       OpenCV's other answer -- so the parameter is not decorative.
void testBorderFillIsTheCallers() {
    const int n = 8;
    cv::Mat white(n, n, CV_8U, cv::Scalar(255));
    BinMat<uint32_t> src(n, n);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) src.set(y, x, true);
    }
    BinMat<uint32_t> dst(n, n);
    const StructuringElement se = StructuringElement::rect(3, 3);

    bincv::erode(src.constView(), dst.view(), se);
    cv::Mat expectedDefault;
    cv::erode(white, expectedDefault, cv::Mat());
    BINCV_EXPECT_BIT_EXACT(dst.constView(), expectedDefault,
                           "erode default border on an all-white frame");
    MORPH_EXPECT(dst.countNonZero() == n * n,
                 "binCV's default erode border leaves an all-white frame intact",
                 std::to_string(dst.countNonZero()) + " of " + std::to_string(n * n));

    bincv::erode(src.constView(), dst.view(), se, bincv::BORDER_CONSTANT, false);
    cv::Mat expectedZero;
    cv::erode(white, expectedZero, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT,
              cv::Scalar(0));
    BINCV_EXPECT_BIT_EXACT(dst.constView(), expectedZero, "erode with an explicit zero border");
    MORPH_EXPECT(dst.countNonZero() == (n - 2) * (n - 2),
                 "a zero border does erode the frame's edge away -- so the fill is load-bearing",
                 std::to_string(dst.countNonZero()) + " of " + std::to_string((n - 2) * (n - 2)));

    cv::Mat black = cv::Mat::zeros(n, n, CV_8U);
    BinMat<uint32_t> empty(n, n);
    bincv::dilate(empty.constView(), dst.view(), se);
    cv::Mat expectedDilate;
    cv::dilate(black, expectedDilate, cv::Mat());
    BINCV_EXPECT_BIT_EXACT(dst.constView(), expectedDilate,
                           "dilate default border on an all-black frame");
    MORPH_EXPECT(dst.countNonZero() == 0,
                 "binCV's default dilate border grows nothing from an empty frame",
                 std::to_string(dst.countNonZero()));
}

#endif  // BINCV_WITH_OPENCV

}  // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

BINCV_TEST(Morphology, ElementStructure) { testElementStructure(); }

BINCV_TEST(Morphology, Reference_uint8_t)  { testAgainstReference<uint8_t>("uint8_t"); }
BINCV_TEST(Morphology, Reference_uint16_t) { testAgainstReference<uint16_t>("uint16_t"); }
BINCV_TEST(Morphology, Reference_uint32_t) { testAgainstReference<uint32_t>("uint32_t"); }
BINCV_TEST(Morphology, Reference_uint64_t) { testAgainstReference<uint64_t>("uint64_t"); }

BINCV_TEST(Morphology, SingleOffsetEqualsShift_uint8_t)  { testSingleOffsetEqualsShift<uint8_t>("uint8_t"); }
BINCV_TEST(Morphology, SingleOffsetEqualsShift_uint16_t) { testSingleOffsetEqualsShift<uint16_t>("uint16_t"); }
BINCV_TEST(Morphology, SingleOffsetEqualsShift_uint32_t) { testSingleOffsetEqualsShift<uint32_t>("uint32_t"); }
BINCV_TEST(Morphology, SingleOffsetEqualsShift_uint64_t) { testSingleOffsetEqualsShift<uint64_t>("uint64_t"); }

BINCV_TEST(Morphology, FastPathEqualsGeneric_uint8_t)  { testFastPathEqualsGeneric<uint8_t>("uint8_t"); }
BINCV_TEST(Morphology, FastPathEqualsGeneric_uint16_t) { testFastPathEqualsGeneric<uint16_t>("uint16_t"); }
BINCV_TEST(Morphology, FastPathEqualsGeneric_uint32_t) { testFastPathEqualsGeneric<uint32_t>("uint32_t"); }
BINCV_TEST(Morphology, FastPathEqualsGeneric_uint64_t) { testFastPathEqualsGeneric<uint64_t>("uint64_t"); }

BINCV_TEST(Morphology, Compound_uint8_t)  { testCompound<uint8_t>("uint8_t"); }
BINCV_TEST(Morphology, Compound_uint16_t) { testCompound<uint16_t>("uint16_t"); }
BINCV_TEST(Morphology, Compound_uint32_t) { testCompound<uint32_t>("uint32_t"); }
BINCV_TEST(Morphology, Compound_uint64_t) { testCompound<uint64_t>("uint64_t"); }

BINCV_TEST(Morphology, Padding_uint8_t)  { testPaddingAndDegenerate<uint8_t>("uint8_t"); }
BINCV_TEST(Morphology, Padding_uint16_t) { testPaddingAndDegenerate<uint16_t>("uint16_t"); }
BINCV_TEST(Morphology, Padding_uint32_t) { testPaddingAndDegenerate<uint32_t>("uint32_t"); }
BINCV_TEST(Morphology, Padding_uint64_t) { testPaddingAndDegenerate<uint64_t>("uint64_t"); }

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Morphology, ElementMatchesOpenCv) { testElementMatchesOpenCv(); }

BINCV_TEST(Morphology, OpenCv_uint8_t)  { testOpenCvErodeDilate<uint8_t>("uint8_t"); }
BINCV_TEST(Morphology, OpenCv_uint16_t) { testOpenCvErodeDilate<uint16_t>("uint16_t"); }
BINCV_TEST(Morphology, OpenCv_uint32_t) { testOpenCvErodeDilate<uint32_t>("uint32_t"); }
BINCV_TEST(Morphology, OpenCv_uint64_t) { testOpenCvErodeDilate<uint64_t>("uint64_t"); }

BINCV_TEST(Morphology, OpenCvOps_uint8_t)  { testOpenCvOps<uint8_t>("uint8_t"); }
BINCV_TEST(Morphology, OpenCvOps_uint16_t) { testOpenCvOps<uint16_t>("uint16_t"); }
BINCV_TEST(Morphology, OpenCvOps_uint32_t) { testOpenCvOps<uint32_t>("uint32_t"); }
BINCV_TEST(Morphology, OpenCvOps_uint64_t) { testOpenCvOps<uint64_t>("uint64_t"); }

BINCV_TEST(Morphology, OpenCvBorders_uint8_t)  { testOpenCvBorders<uint8_t>("uint8_t"); }
BINCV_TEST(Morphology, OpenCvBorders_uint16_t) { testOpenCvBorders<uint16_t>("uint16_t"); }
BINCV_TEST(Morphology, OpenCvBorders_uint32_t) { testOpenCvBorders<uint32_t>("uint32_t"); }
BINCV_TEST(Morphology, OpenCvBorders_uint64_t) { testOpenCvBorders<uint64_t>("uint64_t"); }

BINCV_TEST(Morphology, BorderFillIsTheCallers) { testBorderFillIsTheCallers(); }
#endif

BINCV_TEST_MAIN("test_morphology")
