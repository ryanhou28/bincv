// Logic kernels: bitwiseAnd / bitwiseOr / bitwiseXor / bitwiseNot.
//
// TWO HALVES, and they answer different questions.
//
// 1. The CORE half (this whole file up to the OpenCV guard) needs no OpenCV, so
// it runs in all four verification configurations -- including the Debug one,
// which is the only place BINCV_ASSERT is live, and the -fno-exceptions one,
// which is the embedded claim. It checks the kernels against a per-pixel
// reference, over differing strides, in place, at the degenerate sizes, and
// against the padding-bit invariant that no pixel comparison can see.
//
// 2. The OPENCV half asserts what Tier 1 actually promises: bit-exactness
// against cv::bitwise_and/or/xor/not on the same binary content stored as
// CV_8U (the design notes, 10.3), through that work’s harness, across its full
// size and fill matrix, at all four word widths.
//
// The two are not redundant. A per-pixel reference written next to the kernel can
// share a misunderstanding with it; OpenCV cannot. And the OpenCV half cannot run
// in three of the four configurations, so a kernel verified only there would be
// unverified everywhere binCV claims to be usable.
//
// WHY THE CHECK COUNT IS NOT ONE PER PIXEL: a 640x480 case would contribute
// 307200 checks and drown the summary. Each swept case reports its DISAGREEMENT
// COUNT as a single check instead, so CHECKS tracks cases and a failure still
// says how badly it failed.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "test_util.hpp"

// OpenCV-only, and self-guarding: everything in it sits behind BINCV_WITH_OPENCV,
// so this include is a no-op in the three configurations that have no OpenCV.
#include "equivalence.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#endif

namespace {

using bincv::bitwiseAnd;
using bincv::bitwiseNot;
using bincv::bitwiseOr;
using bincv::bitwiseXor;

// ---------------------------------------------------------------------------
// The operations under test, as data
// ---------------------------------------------------------------------------

enum class Op { And, Or, Xor, Not };

const Op ALL_OPS[] = {Op::And, Op::Or, Op::Xor, Op::Not};

const char* opName(Op op) {
    switch (op) {
        case Op::And: return "and";
        case Op::Or:  return "or";
        case Op::Xor: return "xor";
        case Op::Not: return "not";
    }
    return "?";
}

/// @brief The per-pixel meaning of each operation. The reference the core half
/// compares against, written independently of the word arithmetic.
bool reference(Op op, bool a, bool b) {
    switch (op) {
        case Op::And: return a && b;
        case Op::Or:  return a || b;
        case Op::Xor: return a != b;
        case Op::Not: return !a;
    }
    return false;
}

/// @brief Runs one operation through the VIEW kernels, never the container.
/// @note bitwiseNot ignores `b`, which is why it is passed anyway: it keeps the
/// sweeps below one loop over ALL_OPS rather than two.
template <typename WordType>
void runOp(Op op, const bincv::BinMat<WordType>& a, const bincv::BinMat<WordType>& b,
           bincv::BinMat<WordType>& dst) {
    switch (op) {
        case Op::And: bitwiseAnd(a.constView(), b.constView(), dst.view()); break;
        case Op::Or:  bitwiseOr(a.constView(), b.constView(), dst.view()); break;
        case Op::Xor: bitwiseXor(a.constView(), b.constView(), dst.view()); break;
        case Op::Not: bitwiseNot(a.constView(), dst.view()); break;
    }
}

// ---------------------------------------------------------------------------
// Content: the same generator as tests/equivalence.hpp, minus OpenCV
// ---------------------------------------------------------------------------
//
// tests/equivalence.hpp's randomBinary is behind BINCV_WITH_OPENCV, and three of
// the four configurations this suite runs in have no OpenCV. This is the same
// SplitMix64 draw and the same threshold mapping, so a case that fails here
// reproduces there; it is duplicated rather than shared because moving it out of
// equivalence.hpp would mean the harness and the suite it judges shared a
// generator, and that work’s whole argument is that shared machinery cancels faults.

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
    const double rounded =
        static_cast<double>(fillRatio) * static_cast<double>(FILL_SCALE) + 0.5;
    if (rounded >= static_cast<double>(FILL_SCALE)) return FILL_SCALE;
    return static_cast<uint32_t>(rounded);
}

/// @brief Fills a matrix through set, so the padding bits stay clear on entry.
template <typename WordType>
void fillRandom(bincv::BinMat<WordType>& m, float fillRatio, uint64_t seed) {
    uint64_t state = seed;
    const uint32_t threshold = fillThreshold(fillRatio);
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            if (static_cast<uint32_t>(nextRandom(state) >> 40) < threshold) m.set(y, x, true);
        }
    }
}

uint64_t caseSeed(int width, int height, size_t index) {
    return UINT64_C(0x109C0DE00000001) + static_cast<uint64_t>(width) * UINT64_C(1000003) +
           static_cast<uint64_t>(height) * UINT64_C(10007) + static_cast<uint64_t>(index);
}

// ---------------------------------------------------------------------------
// Observers
// ---------------------------------------------------------------------------

/// @brief Set bits across the whole STRIDE, padding included.
/// @note Deliberately not a library operation -- binCV exposes no per-word
/// popcount. Comparing it against countNonZero's per-pixel loop is
/// how a padding-bit violation becomes visible: they agree only when every
/// bit past `width` is zero. This is the check that a word-wise kernel
/// cannot pass by accident, and the one bitwiseNot fails without its mask.
template <typename WordType>
int bitsAcrossStride(const bincv::BinMat<WordType>& m) {
    int bits = 0;
    for (int y = 0; y < m.rows(); ++y) {
        const WordType* row = m.ptr(y);
        for (size_t w = 0; w < m.getAlignedWidth(); ++w) {
            WordType v = row[w];
            while (v != 0) {
                bits += static_cast<int>(v & static_cast<WordType>(1));
                v = static_cast<WordType>(v >> 1);
            }
        }
    }
    return bits;
}

/// @brief Pixels on which the kernel's result differs from the reference.
template <typename WordType>
int disagreements(Op op, const bincv::BinMat<WordType>& a, const bincv::BinMat<WordType>& b,
                  const bincv::BinMat<WordType>& dst) {
    int differing = 0;
    for (int y = 0; y < dst.rows(); ++y) {
        for (int x = 0; x < dst.cols(); ++x) {
            if (dst.at(y, x) != reference(op, a.at(y, x), b.at(y, x))) ++differing;
        }
    }
    return differing;
}

std::string label(const char* wordTypeName, Op op, int width, int height, const char* extra) {
    return std::string(wordTypeName) + " " + opName(op) + " " + std::to_string(width) + "x" +
           std::to_string(height) + " " + extra;
}

// ---------------------------------------------------------------------------
// The sweep matrix
//
// The widths, plus 128 -- an exact multiple of every supported word width,
// which is the only shape that reaches the single-contiguous-run path in
// logic.hpp. Without it that path would be exercised only at 640, and only when
// every argument also happened to be tightly strided.
// ---------------------------------------------------------------------------

const int WIDTHS[] = {1, 7, 31, 33, 40, 63, 65, 70, 128, 640};
const int HEIGHTS[] = {1, 2, 3, 17};
const float FILLS[] = {0.0f, 0.01f, 0.5f, 0.99f, 1.0f};

// An over-aligned row stride (the design rule makes alignment a per-object choice): 32 bytes
// is a whole number of 1-, 2-, 4- and 8-byte words, so every word type gets a
// stride strictly larger than the ceil(width / WordBits) words its rows need.
constexpr size_t PADDED_ALIGNMENT = 32;

// ===========================================================================
// 1. Against a per-pixel reference, over the sweep
// ===========================================================================

template <typename WordType>
void testAgainstReference(const char* wordTypeName) {
    std::cout << "\n--- logic vs per-pixel reference: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : HEIGHTS) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                const uint64_t seed = caseSeed(width, height, f);
                bincv::BinMat<WordType> a(width, height);
                bincv::BinMat<WordType> b(width, height);
                fillRandom(a, FILLS[f], seed);
                fillRandom(b, FILLS[f], seed ^ UINT64_C(0xDEADBEEF));

                for (Op op : ALL_OPS) {
                    bincv::BinMat<WordType> dst(width, height);
                    runOp(op, a, b, dst);

                    ::bincv::test::reportCheck(
                        disagreements(op, a, b, dst) == 0, "matches the per-pixel reference",
                        __FILE__, __LINE__, label(wordTypeName, op, width, height, "reference"));

                    // The invariant a pixel comparison is blind to: bits past
                    // `width` must be zero, or the first word-wise reduction built
                    // on this result over-counts. bitwiseNot sets every one of
                    // them before the mask removes them.
                    ::bincv::test::reportCheck(
                        bitsAcrossStride(dst) == dst.countNonZero(),
                        "padding bits are zero past width", __FILE__, __LINE__,
                        label(wordTypeName, op, width, height, "padding"));
                }
            }
        }
    }
}

// ===========================================================================
// 2. Differing strides between a, b and dst
// ===========================================================================
//
// The failure this exists to catch: a kernel that walks its arguments as one
// dense run is correct whenever every argument was built the same way, and wrong
// the moment one is over-aligned or wraps a caller's buffer. Measured during earlier work
// on the default-alignment sweep -- stride was the minimum in 48 of 48 cases, so
// nothing there could have noticed.
//
// Three stride flavours per argument:
// tight stride == ceil(width / WordBits) the default
// padded stride from a 32-byte row alignment the opt-in
// odd stride == tight + 3, a wrapped buffer a stride no allocator would pick

enum class Stride { Tight, Padded, Odd };

const char* strideName(Stride s) {
    switch (s) {
        case Stride::Tight:  return "tight";
        case Stride::Padded: return "padded";
        case Stride::Odd:    return "odd";
    }
    return "?";
}

/// @brief A matrix at the requested stride flavour, plus the buffer behind it.
/// @note The `odd` flavour wraps a caller-provided buffer, so the buffer has to
/// outlive the matrix -- which is why it is returned alongside rather than
/// being a local of a factory function.
template <typename WordType>
struct StridedMat {
    std::vector<WordType> buffer;   // used by Stride::Odd only
    bincv::BinMat<WordType> mat;
};

template <typename WordType>
void makeStrided(StridedMat<WordType>& out, Stride flavour, int width, int height) {
    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;
    const size_t minWords = (static_cast<size_t>(width) + wordBits - 1) / wordBits;

    switch (flavour) {
        case Stride::Tight:
            out.mat = bincv::BinMat<WordType>(width, height);
            return;
        case Stride::Padded:
            out.mat = bincv::BinMat<WordType>(width, height, PADDED_ALIGNMENT);
            return;
        case Stride::Odd: {
            const size_t stride = minWords + 3;
            // Zero-initialized, which is what an owning BinMat would have given:
            // a wrapped buffer's padding bits are the caller's responsibility.
            out.buffer.assign(stride * static_cast<size_t>(height), static_cast<WordType>(0));
            out.mat = bincv::BinMat<WordType>(out.buffer.data(), width, height, stride);
            return;
        }
    }
}

template <typename WordType>
void testDifferingStrides(const char* wordTypeName) {
    std::cout << "\n--- logic across differing strides: " << wordTypeName << " ---\n";

    // Six combinations rather than all 27: each argument takes each flavour at
    // least twice, and every pair of arguments differs in at least two of them.
    const Stride combos[][3] = {
        {Stride::Tight,  Stride::Tight,  Stride::Padded},
        {Stride::Tight,  Stride::Padded, Stride::Tight},
        {Stride::Padded, Stride::Tight,  Stride::Odd},
        {Stride::Odd,    Stride::Padded, Stride::Tight},
        {Stride::Padded, Stride::Odd,    Stride::Padded},
        {Stride::Odd,    Stride::Odd,    Stride::Odd},
    };

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                const uint64_t seed = caseSeed(width, height, f + 100);
                for (const auto& combo : combos) {
                    StridedMat<WordType> a, b, dst;
                    makeStrided(a, combo[0], width, height);
                    makeStrided(b, combo[1], width, height);
                    fillRandom(a.mat, FILLS[f], seed);
                    fillRandom(b.mat, FILLS[f], seed ^ UINT64_C(0xDEADBEEF));

                    for (Op op : ALL_OPS) {
                        makeStrided(dst, combo[2], width, height);
                        runOp(op, a.mat, b.mat, dst.mat);

                        const std::string extra =
                            std::string(strideName(combo[0])) + "/" + strideName(combo[1]) +
                            "/" + strideName(combo[2]);
                        ::bincv::test::reportCheck(
                            disagreements(op, a.mat, b.mat, dst.mat) == 0,
                            "matches the per-pixel reference", __FILE__, __LINE__,
                            label(wordTypeName, op, width, height, extra.c_str()));
                        ::bincv::test::reportCheck(
                            bitsAcrossStride(dst.mat) == dst.mat.countNonZero(),
                            "padding bits are zero past width", __FILE__, __LINE__,
                            label(wordTypeName, op, width, height, extra.c_str()));
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 3. Aliasing: dst IS a, or dst IS b
// ===========================================================================
//
// The documented contract (see logic.hpp): a destination that is EXACTLY one of
// the sources -- same first word, same stride -- is supported, because these
// operations are pointwise in the word index. Anything that overlaps at a
// different offset is undefined and asserted against in debug builds; that half
// is covered as a death test (tests/test_assert_abort.cpp, case logic-alias),
// since a failed assert takes the process down.

template <typename WordType>
void testInPlace(const char* wordTypeName) {
    std::cout << "\n--- logic in place: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                const uint64_t seed = caseSeed(width, height, f + 200);
                bincv::BinMat<WordType> a(width, height);
                bincv::BinMat<WordType> b(width, height);
                fillRandom(a, FILLS[f], seed);
                fillRandom(b, FILLS[f], seed ^ UINT64_C(0xDEADBEEF));

                for (Op op : ALL_OPS) {
                    // Out of place first, as the expectation.
                    bincv::BinMat<WordType> expected(width, height);
                    runOp(op, a, b, expected);

                    // dst IS a
                    bincv::BinMat<WordType> intoA(a);       // deep copy
                    switch (op) {
                        case Op::And: bitwiseAnd(intoA.constView(), b.constView(), intoA.view()); break;
                        case Op::Or:  bitwiseOr(intoA.constView(), b.constView(), intoA.view()); break;
                        case Op::Xor: bitwiseXor(intoA.constView(), b.constView(), intoA.view()); break;
                        case Op::Not: bitwiseNot(intoA.constView(), intoA.view()); break;
                    }
                    ::bincv::test::reportCheck(
                        disagreements(op, a, b, intoA) == 0, "in place over a matches out of place",
                        __FILE__, __LINE__, label(wordTypeName, op, width, height, "dst == a"));
                    ::bincv::test::reportCheck(
                        bitsAcrossStride(intoA) == intoA.countNonZero(),
                        "padding bits are zero past width", __FILE__, __LINE__,
                        label(wordTypeName, op, width, height, "dst == a"));

                    if (op == Op::Not) continue;   // one source; dst == b has no meaning

                    // dst IS b
                    bincv::BinMat<WordType> intoB(b);
                    switch (op) {
                        case Op::And: bitwiseAnd(a.constView(), intoB.constView(), intoB.view()); break;
                        case Op::Or:  bitwiseOr(a.constView(), intoB.constView(), intoB.view()); break;
                        case Op::Xor: bitwiseXor(a.constView(), intoB.constView(), intoB.view()); break;
                        case Op::Not: break;
                    }
                    ::bincv::test::reportCheck(
                        disagreements(op, a, b, intoB) == 0, "in place over b matches out of place",
                        __FILE__, __LINE__, label(wordTypeName, op, width, height, "dst == b"));
                }
            }
        }
    }
}

// ===========================================================================
// 4. Degenerate shapes: empty, and 1x1
// ===========================================================================

template <typename WordType>
void testDegenerate(const char* wordTypeName) {
    std::cout << "\n--- logic at degenerate sizes: " << wordTypeName << " ---\n";

    // Empty is a no-op, not an error. All three shapes of empty, since a kernel
    // could plausibly guard one and not the others.
    const int emptyShapes[][2] = {{0, 0}, {0, 5}, {5, 0}};
    for (const auto& shape : emptyShapes) {
        bincv::BinMat<WordType> a(shape[0], shape[1]);
        bincv::BinMat<WordType> b(shape[0], shape[1]);
        bincv::BinMat<WordType> dst(shape[0], shape[1]);
        for (Op op : ALL_OPS) {
            runOp(op, a, b, dst);
            BINCV_CHECK(dst.empty());
            BINCV_CHECK_EQ(dst.countNonZero(), 0);
        }
    }

    // An empty DESTINATION must not be written through even when its buffer
    // exists: a 5x0 matrix wrapping a live buffer is the shape a kernel that
    // trusted `width` alone would scribble on.
    {
        std::vector<WordType> buffer(4, static_cast<WordType>(0x2Du));
        bincv::BinMat<WordType> dst(buffer.data(), 5, 0, 1);
        bincv::BinMat<WordType> src(buffer.data(), 5, 0, 1);
        for (Op op : ALL_OPS) {
            runOp(op, src, src, dst);
        }
        bool untouched = true;
        for (WordType w : buffer) untouched = untouched && (w == static_cast<WordType>(0x2Du));
        BINCV_CHECK(untouched);
    }

    // 1x1: the whole truth table, one pixel at a time. The trailing-word mask is
    // at its most aggressive here -- one live bit, WordBits - 1 padding bits.
    for (int av = 0; av < 2; ++av) {
        for (int bv = 0; bv < 2; ++bv) {
            bincv::BinMat<WordType> a(1, 1);
            bincv::BinMat<WordType> b(1, 1);
            a.set(0, 0, av != 0);
            b.set(0, 0, bv != 0);
            for (Op op : ALL_OPS) {
                bincv::BinMat<WordType> dst(1, 1);
                runOp(op, a, b, dst);
                BINCV_CHECK_EQ(dst.at(0, 0) ? 1 : 0,
                               reference(op, av != 0, bv != 0) ? 1 : 0);
                BINCV_CHECK_EQ(bitsAcrossStride(dst), dst.countNonZero());
            }
        }
    }

    std::cout << " " << wordTypeName << ": empty and 1x1 shapes handled\n";
}

// ===========================================================================
// 5. The kernel writes nothing outside the destination it was given
// ===========================================================================

template <typename WordType>
void testGuardWords(const char* wordTypeName) {
    std::cout << "\n--- logic writes only its destination: " << wordTypeName << " ---\n";

    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;
    const WordType sentinel = static_cast<WordType>(0xA5A5A5A5A5A5A5A5ull);

    for (int width : WIDTHS) {
        const int height = 3;
        const size_t minWords = (static_cast<size_t>(width) + wordBits - 1) / wordBits;
        const size_t stride = minWords + 2;          // two padding words per row
        const size_t lead = 2;                       // words before the matrix
        const size_t trail = 2;                      // words after it

        for (Op op : ALL_OPS) {
            std::vector<WordType> buffer(lead + stride * static_cast<size_t>(height) + trail,
                                         sentinel);
            bincv::BinMat<WordType> dst(buffer.data() + lead, width, height, stride);
            // The wrapped destination starts dirty, sentinel bits and all, which
            // is also what makes this a test of the trailing-word mask: the
            // kernel has to CLEAR bits it never sets.
            bincv::BinMat<WordType> a(width, height);
            bincv::BinMat<WordType> b(width, height);
            fillRandom(a, 0.5f, caseSeed(width, height, 300));
            fillRandom(b, 0.5f, caseSeed(width, height, 301));

            runOp(op, a, b, dst);

            bool guardsIntact = true;
            for (size_t i = 0; i < lead; ++i) guardsIntact = guardsIntact && buffer[i] == sentinel;
            for (size_t i = buffer.size() - trail; i < buffer.size(); ++i) {
                guardsIntact = guardsIntact && buffer[i] == sentinel;
            }
            // The words INSIDE the destination's own stride, past the words its
            // pixels need. logic.hpp says the kernel does not write them and they
            // belong to the caller, and until this loop existed nothing checked
            // it: a kernel that zeroed them stayed green in every configuration,
            // because the lead/trail guards are outside the rows and the padding
            // scan below only reads the pixel words. For a destination that
            // windows a taller image those words are a neighbour's pixels.
            for (int y = 0; y < height; ++y) {
                for (size_t w = minWords; w < stride; ++w) {
                    guardsIntact =
                        guardsIntact &&
                        buffer[lead + static_cast<size_t>(y) * stride + w] == sentinel;
                }
            }
            ::bincv::test::reportCheck(guardsIntact, "guard words outside dst are untouched",
                                       __FILE__, __LINE__,
                                       label(wordTypeName, op, width, height, "guards"));
            ::bincv::test::reportCheck(disagreements(op, a, b, dst) == 0,
                                       "matches the per-pixel reference", __FILE__, __LINE__,
                                       label(wordTypeName, op, width, height, "wrapped dst"));

            // The trailing partial word must be clean even though the buffer was
            // handed over dirty; the whole padding words past it are outside what
            // the kernel writes, and are documented as the caller's.
            int pixelWordBits = 0;
            for (int y = 0; y < height; ++y) {
                const WordType* row = dst.ptr(y);
                for (size_t w = 0; w < minWords; ++w) {
                    WordType v = row[w];
                    while (v != 0) {
                        pixelWordBits += static_cast<int>(v & static_cast<WordType>(1));
                        v = static_cast<WordType>(v >> 1);
                    }
                }
            }
            ::bincv::test::reportCheck(pixelWordBits == dst.countNonZero(),
                                       "padding bits are zero past width", __FILE__, __LINE__,
                                       label(wordTypeName, op, width, height, "wrapped dst"));
        }
    }
}

// ===========================================================================
// 6. SOURCES WHOSE PADDING BITS ARE ALREADY DIRTY
// ===========================================================================
//
// The case that makes the trailing-word mask load-bearing for AND / OR / XOR
// rather than only for NOT, and the case the suite did not have.
//
// MEASURED: with `& tailMask` deleted from impl::applyBinary and applyUnary left
// alone, this file passed 56044 of 56044 checks under OpenCV and 43948 of 43948
// core -- fully green in both. Every source in every other sweep is built by an
// owning BinMat or through set, so every source's padding is already zero and
// `Op(0, 0)` is 0 for all three binary operations. The mask has nothing to do.
//
// A wrapped buffer is where it does. BinMat's wrap constructor documents that a
// caller's padding bits are the caller's -- they are not zeroed on wrap -- so a
// source with dirty padding is a supported construction, and logic.hpp property 3
// claims a destination comes out clean anyway. Without the mask, at width 5 over
// an all-ones buffer: bits across the stride 16 against countNonZero 10 at
// uint8_t, 32 vs 10 at uint16_t, 64 vs 10 at uint32_t, 128 vs 10 at uint64_t --
// a CLAUDE.md hard-rule violation with no test able to see it.

/// @brief A matrix wrapping a buffer whose PADDING bits are all ones.
/// @note Written through set after the wrap, so the pixel bits are the drawn
/// content and every bit past `width` keeps the 1 it was born with. Stride
/// is the tight minimum so that bitsAcrossStride covers exactly the words
/// the kernel writes -- the words past that are the caller's by contract and
/// are the subject of testGuardWords instead.
template <typename WordType>
void makeDirtyPadded(StridedMat<WordType>& out, int width, int height, float fillRatio,
                     uint64_t seed) {
    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;
    const size_t minWords = (static_cast<size_t>(width) + wordBits - 1) / wordBits;
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));

    out.buffer.assign(minWords * static_cast<size_t>(height), allOnes);
    out.mat = bincv::BinMat<WordType>(out.buffer.data(), width, height, minWords);

    uint64_t state = seed;
    const uint32_t threshold = fillThreshold(fillRatio);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            out.mat.set(y, x, static_cast<uint32_t>(nextRandom(state) >> 40) < threshold);
        }
    }
}

template <typename WordType>
void testDirtySources(const char* wordTypeName) {
    std::cout << "\n--- logic with dirty source padding: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                const uint64_t seed = caseSeed(width, height, f + 800);
                StridedMat<WordType> a, b;
                makeDirtyPadded(a, width, height, FILLS[f], seed);
                makeDirtyPadded(b, width, height, FILLS[f], seed ^ UINT64_C(0xDEADBEEF));

                for (Op op : ALL_OPS) {
                    bincv::BinMat<WordType> dst(width, height);
                    runOp(op, a.mat, b.mat, dst);

                    ::bincv::test::reportCheck(
                        disagreements(op, a.mat, b.mat, dst) == 0,
                        "matches the per-pixel reference", __FILE__, __LINE__,
                        label(wordTypeName, op, width, height, "dirty sources"));
                    // The verdict the mask exists for: the destination is clean
                    // even though both inputs' padding was all ones.
                    ::bincv::test::reportCheck(
                        bitsAcrossStride(dst) == dst.countNonZero(),
                        "padding bits are zero past width", __FILE__, __LINE__,
                        label(wordTypeName, op, width, height, "dirty sources"));

                    // In place into the dirty matrix itself, so the kernel has to
                    // CLEAR bits rather than merely not set them.
                    StridedMat<WordType> inPlace;
                    makeDirtyPadded(inPlace, width, height, FILLS[f], seed);
                    switch (op) {
                        case Op::And:
                            bitwiseAnd(inPlace.mat.constView(), b.mat.constView(),
                                       inPlace.mat.view());
                            break;
                        case Op::Or:
                            bitwiseOr(inPlace.mat.constView(), b.mat.constView(),
                                      inPlace.mat.view());
                            break;
                        case Op::Xor:
                            bitwiseXor(inPlace.mat.constView(), b.mat.constView(),
                                       inPlace.mat.view());
                            break;
                        case Op::Not:
                            bitwiseNot(inPlace.mat.constView(), inPlace.mat.view());
                            break;
                    }
                    ::bincv::test::reportCheck(
                        disagreements(op, a.mat, b.mat, inPlace.mat) == 0,
                        "in place over a dirty source matches out of place", __FILE__, __LINE__,
                        label(wordTypeName, op, width, height, "dirty in place"));
                    ::bincv::test::reportCheck(
                        bitsAcrossStride(inPlace.mat) == inPlace.mat.countNonZero(),
                        "padding bits are zero past width", __FILE__, __LINE__,
                        label(wordTypeName, op, width, height, "dirty in place"));
                }
            }
        }
    }
}

// ===========================================================================
// 7. Aliasing, the ACCEPTING half: views over one buffer that share no word
// ===========================================================================
//
// logic.hpp promises a destination may be exactly a source, or share no memory
// with it. The second half of that had no test, and the predicate enforcing it
// compared the two views' BOUNDING SPANS -- so every view laid over one buffer
// was rejected, whether or not it shared a byte.
//
// MEASURED, before the fix: each of the three cases below aborted a Debug build
// with "dst must alias an input exactly or not overlap it", and each was fully
// correct in a Release build -- 0 wrong words and 0 source words modified. Both
// shapes are ordinary: alternate row bands are what a pyramid downsample takes
// (the design notes), column tiles are how one frame is split across a loop, and
// the design rule says a kernel takes any {ptr, width, height, stride}.
//
// This case runs in every configuration but only MEANS anything in the Debug one,
// where BINCV_ASSERT is live. That is the configuration verify.sh added for
// exactly this class of fault: an assertion whose own condition is wrong.

template <typename WordType>
void testAliasAcceptsDisjointViews(const char* wordTypeName) {
    std::cout << "\n--- logic accepts disjoint views over one buffer: " << wordTypeName
              << " ---\n";

    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;
    const size_t rowWords = 2;                        // both cases use 2-word rows
    const int width = static_cast<int>(2 * wordBits); // exactly two words, no tail

    // (a) Interleaved ROW BANDS: dst is physical rows 0, 2, 4 and src is rows
    // 1, 3, 5 of one 6-row image. Same stride, half a stride apart.
    {
        const int height = 3;
        std::vector<WordType> buffer(rowWords * 6, static_cast<WordType>(0));
        for (size_t i = 0; i < buffer.size(); ++i) {
            buffer[i] = static_cast<WordType>(0x5Au + i);
        }
        const std::vector<WordType> before = buffer;

        const bincv::BinMatConstView<WordType> src{buffer.data() + rowWords, static_cast<size_t>(width),
                                                   static_cast<size_t>(height), 2 * rowWords};
        const bincv::BinMatView<WordType> dst{buffer.data(), static_cast<size_t>(width),
                                              static_cast<size_t>(height), 2 * rowWords};
        bitwiseNot(src, dst);

        bool correct = true;
        for (size_t y = 0; y < static_cast<size_t>(height); ++y) {
            for (size_t w = 0; w < rowWords; ++w) {
                const size_t srcIndex = (2 * y + 1) * rowWords + w;
                const size_t dstIndex = (2 * y) * rowWords + w;
                correct = correct && buffer[dstIndex] == static_cast<WordType>(~before[srcIndex]);
                correct = correct && buffer[srcIndex] == before[srcIndex];   // src untouched
            }
        }
        ::bincv::test::reportCheck(correct, "interleaved row bands are accepted and correct",
                                   __FILE__, __LINE__,
                                   label(wordTypeName, Op::Not, width, height, "row bands"));
    }

    // (b) Left and right COLUMN TILES of one image: same stride, one row apart.
    {
        const int height = 4;
        const size_t stride = 2 * rowWords;
        std::vector<WordType> buffer(stride * static_cast<size_t>(height),
                                     static_cast<WordType>(0));
        for (size_t i = 0; i < buffer.size(); ++i) {
            buffer[i] = static_cast<WordType>(0x33u + i);
        }
        const std::vector<WordType> before = buffer;

        const bincv::BinMatConstView<WordType> src{buffer.data(), static_cast<size_t>(width),
                                                   static_cast<size_t>(height), stride};
        const bincv::BinMatView<WordType> dst{buffer.data() + rowWords, static_cast<size_t>(width),
                                              static_cast<size_t>(height), stride};
        bitwiseNot(src, dst);

        bool correct = true;
        for (size_t y = 0; y < static_cast<size_t>(height); ++y) {
            for (size_t w = 0; w < rowWords; ++w) {
                const size_t srcIndex = y * stride + w;
                const size_t dstIndex = y * stride + rowWords + w;
                correct = correct && buffer[dstIndex] == static_cast<WordType>(~before[srcIndex]);
                correct = correct && buffer[srcIndex] == before[srcIndex];
            }
        }
        ::bincv::test::reportCheck(correct, "column tiles are accepted and correct", __FILE__,
                                   __LINE__,
                                   label(wordTypeName, Op::Not, width, height, "column tiles"));
    }

    // (c) One row, in place, described with two different strides. A single-row
    // view never reads its stride (row(0) == ptr), and BinMatView::row already
    // exempts height <= 1 from its own non-zero-stride precondition; the alias
    // predicate used to demand the two agree and aborted on a correct call.
    {
        std::vector<WordType> buffer(rowWords, static_cast<WordType>(0x0Fu));
        const std::vector<WordType> before = buffer;
        const bincv::BinMatConstView<WordType> src{buffer.data(), static_cast<size_t>(width), 1,
                                                   rowWords};
        const bincv::BinMatView<WordType> dst{buffer.data(), static_cast<size_t>(width), 1,
                                              rowWords + 5};
        bitwiseNot(src, dst);

        bool correct = true;
        for (size_t w = 0; w < rowWords; ++w) {
            correct = correct && buffer[w] == static_cast<WordType>(~before[w]);
        }
        ::bincv::test::reportCheck(correct, "a single row in place ignores the stride mismatch",
                                   __FILE__, __LINE__,
                                   label(wordTypeName, Op::Not, width, 1, "one row, two strides"));
    }
}

// ===========================================================================
// 8. What a destination's trailing word does to the bits past `width`
// ===========================================================================
//
// The precondition stated at the top of logic.hpp, pinned at BIT granularity.
//
// Every kernel here stores the trailing partial word MASKED, which writes zeros
// into bits [width, minRowWords * WordBits) of each destination row. That is the
// padding-bit invariant when those bits are padding, and it is 1..WordBits-1
// destroyed pixels when the destination is a sub-width window onto a wider image.
// MEASURED: a 70-pixel-wide destination windowed onto a 640-wide 2-row image
// cleared all 52 live pixels in columns 70..95.
//
// testGuardWords cannot see this -- it checks whole WORDS, and the bits in
// question live in a word the kernel is entitled to write. This case exists so
// that the behavior is a recorded contract rather than a blind spot: if the tail
// handling ever changes to read-modify-write, this is the test that says so.

template <typename WordType>
void testTrailingWordContract(const char* wordTypeName) {
    std::cout << "\n--- logic clears the destination's trailing bits: " << wordTypeName
              << " ---\n";

    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));

    // A width with a partial trailing word at every supported word size.
    for (int width : {1, 7, 31, 33, 63, 65, 70}) {
        const size_t w = static_cast<size_t>(width);
        const size_t tail = w % wordBits;
        if (tail == 0) continue;
        const size_t minWords = (w + wordBits - 1) / wordBits;
        const int height = 3;

        for (Op op : ALL_OPS) {
            // The destination wraps a buffer that is all ones, i.e. every bit past
            // `width` in its trailing word starts SET.
            std::vector<WordType> buffer(minWords * static_cast<size_t>(height), allOnes);
            bincv::BinMat<WordType> dst(buffer.data(), width, height, minWords);
            bincv::BinMat<WordType> a(width, height);
            bincv::BinMat<WordType> b(width, height);
            fillRandom(a, 0.5f, caseSeed(width, height, 900));
            fillRandom(b, 0.5f, caseSeed(width, height, 901));

            runOp(op, a, b, dst);

            const WordType padMask = static_cast<WordType>(allOnes << tail);
            bool cleared = true;
            for (int y = 0; y < height; ++y) {
                cleared = cleared && (dst.ptr(y)[minWords - 1] & padMask) == 0;
            }
            ::bincv::test::reportCheck(
                cleared, "bits past width in the trailing word are written as zero", __FILE__,
                __LINE__, label(wordTypeName, op, width, height, "trailing word"));
        }
    }
}

// ===========================================================================
// 9. The QuantMat overloads: the same operations, per plane
// ===========================================================================

template <size_t N, typename WordType>
int quantDisagreements(Op op, const bincv::QuantMat<N, WordType>& a,
                       const bincv::QuantMat<N, WordType>& b,
                       const bincv::QuantMat<N, WordType>& dst) {
    const unsigned mask = bincv::QuantMat<N, WordType>::MaxValue;
    int differing = 0;
    for (int y = 0; y < dst.rows(); ++y) {
        for (int x = 0; x < dst.cols(); ++x) {
            unsigned expected = 0;
            switch (op) {
                case Op::And: expected = a.at(y, x) & b.at(y, x); break;
                case Op::Or:  expected = a.at(y, x) | b.at(y, x); break;
                case Op::Xor: expected = a.at(y, x) ^ b.at(y, x); break;
                case Op::Not: expected = (~a.at(y, x)) & mask; break;
            }
            if (dst.at(y, x) != expected) ++differing;
        }
    }
    return differing;
}

template <size_t N, typename WordType>
void runQuantOp(Op op, const bincv::QuantMat<N, WordType>& a,
                const bincv::QuantMat<N, WordType>& b, bincv::QuantMat<N, WordType>& dst) {
    switch (op) {
        case Op::And: bitwiseAnd(a, b, dst); break;
        case Op::Or:  bitwiseOr(a, b, dst); break;
        case Op::Xor: bitwiseXor(a, b, dst); break;
        case Op::Not: bitwiseNot(a, dst); break;
    }
}

template <size_t N, typename WordType>
void fillRandomQuant(bincv::QuantMat<N, WordType>& m, uint64_t seed) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            m.set(y, x, static_cast<unsigned>(nextRandom(state) >> 40) &
                            bincv::QuantMat<N, WordType>::MaxValue);
        }
    }
}

template <typename WordType>
void testQuantMatOverloads(const char* wordTypeName) {
    std::cout << "\n--- logic over QuantMat planes: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            bincv::QuantMat<3, WordType> a(width, height);
            bincv::QuantMat<3, WordType> b(width, height);
            fillRandomQuant(a, caseSeed(width, height, 400));
            fillRandomQuant(b, caseSeed(width, height, 401));

            for (Op op : ALL_OPS) {
                bincv::QuantMat<3, WordType> dst(width, height);
                runQuantOp(op, a, b, dst);
                ::bincv::test::reportCheck(
                    quantDisagreements(op, a, b, dst) == 0,
                    "QuantMat<3> overload matches the per-pixel reference", __FILE__, __LINE__,
                    label(wordTypeName, op, width, height, "quantmat<3>"));

                // Every plane's padding, not just plane 0's: the overload is a
                // loop, and a loop that stopped early would still leave plane 0
                // correct.
                bool paddingClean = true;
                for (size_t p = 0; p < 3; ++p) {
                    const bincv::BinMatConstView<WordType> plane = dst.constPlane(p);
                    constexpr size_t wordBits = bincv::BinMatConstView<WordType>::WordBits;
                    const size_t minWords =
                        (plane.width + wordBits - 1) / wordBits;
                    const size_t tail = plane.width % wordBits;
                    if (tail == 0) continue;
                    const WordType padMask = static_cast<WordType>(
                        static_cast<WordType>(~static_cast<WordType>(0)) << tail);
                    for (size_t y = 0; y < plane.height; ++y) {
                        if ((plane.row(y)[minWords - 1] & padMask) != 0) paddingClean = false;
                    }
                }
                ::bincv::test::reportCheck(paddingClean, "every plane's padding bits are zero",
                                           __FILE__, __LINE__,
                                           label(wordTypeName, op, width, height, "quantmat<3>"));
            }

            // N == 1 goes through the same overload, BinMat being QuantMat<1>.
            bincv::BinMat<WordType> ba(width, height);
            bincv::BinMat<WordType> bb(width, height);
            fillRandom(ba, 0.5f, caseSeed(width, height, 402));
            fillRandom(bb, 0.5f, caseSeed(width, height, 403));
            for (Op op : ALL_OPS) {
                bincv::BinMat<WordType> dst(width, height);
                runQuantOp(op, ba, bb, dst);
                ::bincv::test::reportCheck(disagreements(op, ba, bb, dst) == 0,
                                           "QuantMat<1> overload matches the per-pixel reference",
                                           __FILE__, __LINE__,
                                           label(wordTypeName, op, width, height, "quantmat<1>"));
            }
        }
    }
}

// ===========================================================================
// 10. Tier 1: bit-exact against OpenCV, across the the matrix
// ===========================================================================
//
// WHERE OPENCV'S INPUTS COME FROM, AND WHY IT IS NOT toCvMask.
//
// The obvious spelling builds both sides through the same conversion: pack the
// content into a BinMat, unpack it with toCvMask to feed cv::bitwise_*, run the
// binCV kernel, and compare the two with expectBitExact -- which unpacks again.
// equivalence.hpp property 2 says exactly what that costs, and MEASURED here it
// costs everything: compiled with BINCV_EQUIVALENCE_INJECT_FAULT=1 (a one-column
// rotation in the unpacking path) this file passed 56044 of 56044 checks and
// exited 0, and with =3 (a transposing conversion) likewise. The fault cancels
// through a pointwise operation, on both sides, in every case.
//
// So OpenCV's inputs are built by randomCvMask, the harness's SECOND generator,
// which writes CV_8U bytes directly and never touches a BinMat or the unpacking
// path. The content is identical by construction -- same SplitMix64, same seed,
// same threshold, same row-major order -- so this is not a different test, it is
// the same test with an independent left-hand side. tests/CMakeLists.txt now
// builds this file under faults 1, 2 and 3 as WILL_FAIL cases, so the property is
// a ctest result rather than a paragraph.

#ifdef BINCV_WITH_OPENCV

/// @brief The content of a QuantMat<3>, as three independent CV_8U plane masks.
/// @note Written from the SAME draw that sets the QuantMat pixel, byte by byte,
/// so the two representations agree without either being derived from the
/// other. The plane overloads' oracle cannot come from constPlane ->
/// unpackTo8U for the same reason the binary sweep's cannot.
template <typename WordType>
void fillRandomQuantWithMasks(bincv::QuantMat<3, WordType>& m, cv::Mat (&planes)[3],
                              uint64_t seed) {
    for (size_t p = 0; p < 3; ++p) {
        planes[p] = cv::Mat::zeros(m.rows(), m.cols(), CV_8U);
    }
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            const unsigned value = static_cast<unsigned>(nextRandom(state) >> 40) &
                                   bincv::QuantMat<3, WordType>::MaxValue;
            m.set(y, x, value);
            for (size_t p = 0; p < 3; ++p) {
                planes[p].ptr<uint8_t>(y)[x] =
                    ((value >> p) & 1u) ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
            }
        }
    }
}

/// @brief What OpenCV produces for one operation on the same content as CV_8U.
/// @note the design notes's denominator, as an oracle: the same binary content a
/// user has today without binCV, through the function they call today.
cv::Mat openCvResult(Op op, const cv::Mat& a, const cv::Mat& b) {
    cv::Mat out;
    switch (op) {
        case Op::And: cv::bitwise_and(a, b, out); break;
        case Op::Or:  cv::bitwise_or(a, b, out); break;
        case Op::Xor: cv::bitwise_xor(a, b, out); break;
        case Op::Not: cv::bitwise_not(a, out); break;
    }
    return out;
}

template <typename WordType>
void testOpenCvEquivalence(const char* wordTypeName) {
    std::cout << "\n--- logic vs cv::bitwise_*: " << wordTypeName << " ---\n";

    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : bincv::test::equivalenceHeights()) {
            for (size_t f = 0; f < bincv::test::equivalenceFillRatios().size(); ++f) {
                const float fill = bincv::test::equivalenceFillRatios()[f];
                const uint64_t seed = caseSeed(width, height, f + 500);

                bincv::BinMat<WordType> a =
                    bincv::test::randomBinary<WordType>(width, height, fill, seed);
                bincv::BinMat<WordType> b = bincv::test::randomBinary<WordType>(
                    width, height, fill, seed ^ UINT64_C(0xDEADBEEF));

                // NOT toCvMask(a): the same seed through the harness's second
                // generator, which never touches the packing or the unpacking
                // path. See the note at the head of this section.
                const cv::Mat cvA = bincv::test::randomCvMask(width, height, fill, seed);
                const cv::Mat cvB = bincv::test::randomCvMask(width, height, fill,
                                                              seed ^ UINT64_C(0xDEADBEEF));

                for (Op op : ALL_OPS) {
                    bincv::BinMat<WordType> dst(width, height);
                    runOp(op, a, b, dst);
                    BINCV_EXPECT_BIT_EXACT(
                        dst.constView(), openCvResult(op, cvA, cvB),
                        bincv::test::caseLabel(wordTypeName, width, height, fill) + " [" +
                            opName(op) + "]");
                }
            }
        }
    }
}

/// @brief The same Tier 1 claim, with the three views deliberately mis-matched in
/// stride -- the case a real caller hits by mixing an over-aligned frame
/// with a tightly-packed one.
template <typename WordType>
void testOpenCvEquivalenceMixedStrides(const char* wordTypeName) {
    std::cout << "\n--- logic vs cv::bitwise_* across strides: " << wordTypeName << " ---\n";

    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : {1, 3, 17}) {
            const float fill = 0.5f;
            const uint64_t seed = caseSeed(width, height, 600);

            // a tight, b over-aligned, dst over-aligned -- three views of the same
            // shape whose strides do not agree.
            bincv::BinMat<WordType> a =
                bincv::test::randomBinary<WordType>(width, height, fill, seed);
            bincv::BinMat<WordType> b = bincv::test::randomBinary<WordType>(
                width, height, fill, seed ^ UINT64_C(0xDEADBEEF), PADDED_ALIGNMENT);

            const cv::Mat cvA = bincv::test::randomCvMask(width, height, fill, seed);
            const cv::Mat cvB =
                bincv::test::randomCvMask(width, height, fill, seed ^ UINT64_C(0xDEADBEEF));

            for (Op op : ALL_OPS) {
                bincv::BinMat<WordType> dst(width, height, PADDED_ALIGNMENT);
                runOp(op, a, b, dst);
                BINCV_EXPECT_BIT_EXACT(dst.constView(), openCvResult(op, cvA, cvB),
                                       bincv::test::caseLabel(wordTypeName, width, height, fill) +
                                           " [" + opName(op) + ", mixed strides]");
            }
        }
    }
}

/// @brief The QuantMat overloads, per plane, against OpenCV.
/// @note Each plane is a binary image in its own right, so Tier 1 applies to it
/// unchanged -- which is the point of the bit-plane representation.
template <typename WordType>
void testOpenCvEquivalencePlanes(const char* wordTypeName) {
    std::cout << "\n--- QuantMat logic vs cv::bitwise_*: " << wordTypeName << " ---\n";

    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : {1, 3, 17}) {
            bincv::QuantMat<3, WordType> a(width, height);
            bincv::QuantMat<3, WordType> b(width, height);

            // Both representations from the same draw, neither derived from the
            // other -- constPlane -> toCvMask on the OpenCV side would put
            // the plane view AND the unpacking path on both sides of the
            // comparison, which is the cancellation this section exists to avoid.
            cv::Mat cvA[3];
            cv::Mat cvB[3];
            fillRandomQuantWithMasks(a, cvA, caseSeed(width, height, 700));
            fillRandomQuantWithMasks(b, cvB, caseSeed(width, height, 701));

            for (Op op : ALL_OPS) {
                bincv::QuantMat<3, WordType> dst(width, height);
                runQuantOp(op, a, b, dst);
                for (size_t p = 0; p < 3; ++p) {
                    BINCV_EXPECT_BIT_EXACT(
                        dst.constPlane(p), openCvResult(op, cvA[p], cvB[p]),
                        bincv::test::caseLabel(wordTypeName, width, height, 0.5f) + " [" +
                            opName(op) + ", plane " + std::to_string(p) + "]");
                }
            }
        }
    }
}

#endif // BINCV_WITH_OPENCV

} // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

BINCV_TEST(Logic, Reference_uint8_t)  { testAgainstReference<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, Reference_uint16_t) { testAgainstReference<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, Reference_uint32_t) { testAgainstReference<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, Reference_uint64_t) { testAgainstReference<uint64_t>("uint64_t"); }

BINCV_TEST(Logic, Strides_uint8_t)  { testDifferingStrides<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, Strides_uint16_t) { testDifferingStrides<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, Strides_uint32_t) { testDifferingStrides<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, Strides_uint64_t) { testDifferingStrides<uint64_t>("uint64_t"); }

BINCV_TEST(Logic, InPlace_uint8_t)  { testInPlace<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, InPlace_uint16_t) { testInPlace<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, InPlace_uint32_t) { testInPlace<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, InPlace_uint64_t) { testInPlace<uint64_t>("uint64_t"); }

BINCV_TEST(Logic, Degenerate_uint8_t)  { testDegenerate<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, Degenerate_uint16_t) { testDegenerate<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, Degenerate_uint32_t) { testDegenerate<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, Degenerate_uint64_t) { testDegenerate<uint64_t>("uint64_t"); }

BINCV_TEST(Logic, GuardWords_uint8_t)  { testGuardWords<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, GuardWords_uint16_t) { testGuardWords<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, GuardWords_uint32_t) { testGuardWords<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, GuardWords_uint64_t) { testGuardWords<uint64_t>("uint64_t"); }

BINCV_TEST(Logic, DirtySources_uint8_t)  { testDirtySources<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, DirtySources_uint16_t) { testDirtySources<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, DirtySources_uint32_t) { testDirtySources<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, DirtySources_uint64_t) { testDirtySources<uint64_t>("uint64_t"); }

BINCV_TEST(Logic, AliasAccepts_uint8_t)  { testAliasAcceptsDisjointViews<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, AliasAccepts_uint16_t) { testAliasAcceptsDisjointViews<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, AliasAccepts_uint32_t) { testAliasAcceptsDisjointViews<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, AliasAccepts_uint64_t) { testAliasAcceptsDisjointViews<uint64_t>("uint64_t"); }

BINCV_TEST(Logic, TrailingWord_uint8_t)  { testTrailingWordContract<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, TrailingWord_uint16_t) { testTrailingWordContract<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, TrailingWord_uint32_t) { testTrailingWordContract<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, TrailingWord_uint64_t) { testTrailingWordContract<uint64_t>("uint64_t"); }

BINCV_TEST(Logic, QuantMat_uint8_t)  { testQuantMatOverloads<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, QuantMat_uint16_t) { testQuantMatOverloads<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, QuantMat_uint32_t) { testQuantMatOverloads<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, QuantMat_uint64_t) { testQuantMatOverloads<uint64_t>("uint64_t"); }

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Logic, OpenCv_uint8_t)  { testOpenCvEquivalence<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, OpenCv_uint16_t) { testOpenCvEquivalence<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, OpenCv_uint32_t) { testOpenCvEquivalence<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, OpenCv_uint64_t) { testOpenCvEquivalence<uint64_t>("uint64_t"); }

BINCV_TEST(Logic, OpenCvMixedStrides_uint8_t)  { testOpenCvEquivalenceMixedStrides<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, OpenCvMixedStrides_uint16_t) { testOpenCvEquivalenceMixedStrides<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, OpenCvMixedStrides_uint32_t) { testOpenCvEquivalenceMixedStrides<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, OpenCvMixedStrides_uint64_t) { testOpenCvEquivalenceMixedStrides<uint64_t>("uint64_t"); }

BINCV_TEST(Logic, OpenCvPlanes_uint8_t)  { testOpenCvEquivalencePlanes<uint8_t>("uint8_t"); }
BINCV_TEST(Logic, OpenCvPlanes_uint16_t) { testOpenCvEquivalencePlanes<uint16_t>("uint16_t"); }
BINCV_TEST(Logic, OpenCvPlanes_uint32_t) { testOpenCvEquivalencePlanes<uint32_t>("uint32_t"); }
BINCV_TEST(Logic, OpenCvPlanes_uint64_t) { testOpenCvEquivalencePlanes<uint64_t>("uint64_t"); }
#endif

BINCV_TEST_MAIN("BinMat logic kernel tests")
