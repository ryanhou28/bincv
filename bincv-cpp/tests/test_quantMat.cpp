// Core tests for the N-bit container (T1.5) and its sign-magnitude reading
// (T1.6). Deliberately free of OpenCV, so this suite also runs in the core-only
// and no-exceptions configurations -- QuantMat is what makes the embedded claim
// concrete, so it has to be verified in the configurations that claim serves.
//
// What is being defended here, in one line: a binary frame does not stay binary
// through the pyramid (ARCHITECTURE 7.2 measured 1 -> 3 -> 4 -> 5 bits), so the
// N-plane container is required, and the thing that makes it affordable is that
// N planes cost ONE allocation of exactly N times the binary footprint -- not N
// allocations, and not a byte more than the arithmetic says.

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <type_traits>
#include <utility>

#include "bincv-cpp/quantMat.hpp"
#include "test_util.hpp"

// ---------------------------------------------------------------------------
// Allocation counters
//
// "One contiguous allocation" and "a wrapped buffer allocates nothing" are
// claims about the allocator, so the allocator is what answers them. Same idiom
// as test_storage.cpp, with the byte count added: the footprint figures below
// (3 x 38400 for a 640x480 QuantMat<3>) are the point of the container, and
// sizeInWords() alone would only report what the container believes.
// ---------------------------------------------------------------------------

namespace {

size_t g_newCount = 0;
size_t g_deleteCount = 0;
size_t g_newBytes = 0;

// Keeps the optimizer from eliding an allocation whose result it can prove is
// unused -- C++14 permits new/delete elision, which would make the counters lie.
const void* volatile g_sink = nullptr;
inline void escape(const void* p) { g_sink = p; }

} // namespace

// Never inlined. At -O2 GCC otherwise inlines these replacement operators into
// the container's constructors, loses track of which new/delete pair it is
// looking at, and reports a correctly paired new[]/delete[] as mismatched
// (-Wmismatched-new-delete, pointing at Storage's `new WordType[words]()` and at
// the std::free below -- which are each other's counterpart). Not inlining them
// also keeps the counters exact, since there is nothing left for the optimizer
// to fold away. A `#pragma GCC diagnostic ignored` does NOT work here: the
// warning is raised after inlining, where the pragma state of this file no
// longer applies.
#if defined(__GNUC__)
#define BINCV_TEST_NOINLINE __attribute__((noinline))
#else
#define BINCV_TEST_NOINLINE
#endif

BINCV_TEST_NOINLINE void* operator new(std::size_t bytes) {
    ++g_newCount;
    g_newBytes += bytes;
    // Reject unsatisfiable requests up front; array-new passes SIZE_MAX as its
    // overflow sentinel and expects the allocator to fail. See test_storage.cpp.
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    // Cannot throw std::bad_alloc: this file also builds with -fno-exceptions.
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}

BINCV_TEST_NOINLINE void* operator new[](std::size_t bytes) { return ::operator new(bytes); }

BINCV_TEST_NOINLINE void operator delete(void* p) noexcept {
    if (p != nullptr) ++g_deleteCount;
    std::free(p);
}

BINCV_TEST_NOINLINE void operator delete[](void* p) noexcept { ::operator delete(p); }
BINCV_TEST_NOINLINE void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
BINCV_TEST_NOINLINE void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

namespace {

using bincv::BinMat;
using bincv::QuantMat;
using bincv::SignedQuantMat;
using bincv::TernaryMat;

/// @brief Bits per word, spelled out rather than taken from the container, so a
///        wrong WordBits cannot agree with itself.
template <typename W>
constexpr size_t wordBits() { return sizeof(W) * 8; }

/// @brief ceil(width / WordBits) -- the stride at the default alignment (D-4).
template <typename W>
size_t expectedStride(size_t width) {
    return (width + wordBits<W>() - 1) / wordBits<W>();
}

/// @brief Reads bit `x` of a row directly out of the packed words.
/// @note Independent of the container's own indexing on purpose: a round trip
///       through set()/at() alone would agree with itself even if the packing
///       were wrong. Column x lives at bit x % WordBits of word x / WordBits.
template <typename W>
bool rawBit(const W* row, size_t x) {
    const W mask = static_cast<W>(static_cast<W>(1) << (x % wordBits<W>()));
    return (row[x / wordBits<W>()] & mask) != 0;
}

template <typename W>
void setRawBit(W* row, size_t x, bool value) {
    const W mask = static_cast<W>(static_cast<W>(1) << (x % wordBits<W>()));
    if (value) {
        row[x / wordBits<W>()] |= mask;
    } else {
        row[x / wordBits<W>()] &= static_cast<W>(~mask);
    }
}

/// @brief True if every bit at or past `width` in this row is zero.
template <typename W>
bool paddingBitsClear(const W* row, size_t width, size_t strideWords) {
    for (size_t x = width; x < strideWords * wordBits<W>(); ++x) {
        if (rawBit(row, x)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// T1.5 -- layout: one allocation, plane p at offset p * planeWords
// ---------------------------------------------------------------------------

template <size_t N, typename W>
void testPlaneLayout(const char* label) {
    std::cout << "\n--- QuantMat<" << N << ", " << label << "> layout ---\n";

    // A width that is deliberately not a multiple of any word size: packing bugs
    // live in the trailing partial word.
    const int width = 70;
    const int height = 5;
    QuantMat<N, W> m(width, height);

    const size_t stride = expectedStride<W>(static_cast<size_t>(width));
    BINCV_CHECK_EQ(m.getWidth(), size_t(width));
    BINCV_CHECK_EQ(m.getHeight(), size_t(height));       // per plane, not the stack
    BINCV_CHECK_EQ(m.getAlignedWidth(), stride);
    // Parenthesized: the preprocessor would otherwise split the template
    // argument list at its comma and hand the macro three arguments.
    BINCV_CHECK_EQ((QuantMat<N, W>::Planes), N);
    BINCV_CHECK_EQ(m.planeWords(), size_t(height) * stride);
    BINCV_CHECK_EQ(m.sizeInWords(), N * size_t(height) * stride);
    BINCV_CHECK(m.ownsMemory());
    BINCV_CHECK(!m.empty());

    // Every plane is a view of the same shape, and plane p begins exactly
    // p * planeWords() into the single block -- contiguity, stated as addresses.
    for (size_t p = 0; p < N; ++p) {
        bincv::BinMatView<W> v = m.plane(p);
        BINCV_CHECK(v.ptr == m.data() + p * m.planeWords());
        BINCV_CHECK_EQ(v.width, size_t(width));
        BINCV_CHECK_EQ(v.height, size_t(height));
        BINCV_CHECK_EQ(v.stride, stride);
    }

    // The const overload describes the same memory.
    const QuantMat<N, W>& cm = m;
    for (size_t p = 0; p < N; ++p) {
        bincv::BinMatConstView<W> cv = cm.plane(p);
        BINCV_CHECK(cv.ptr == cm.data() + p * cm.planeWords());
        BINCV_CHECK_EQ(cv.stride, stride);
    }

    // The last plane ends exactly at the end of the allocation: no gap, no
    // overrun, which is what "one contiguous allocation" means in addresses.
    BINCV_CHECK(m.plane(N - 1).ptr + m.planeWords() == m.data() + m.sizeInWords());
}

// The footprint claim from T1.5, in bytes, measured at the allocator.
void testFootprint() {
    std::cout << "\n--- Footprint: QuantMat<3, uint32_t> at 640x480 ---\n";

    const size_t binaryBytes = 38400;  // 640x480 at 1 bit/px, word granularity (T1.3)

    size_t newsBefore = g_newCount;
    size_t bytesBefore = g_newBytes;
    size_t news = 0;
    size_t bytes = 0;
    {
        QuantMat<3, uint32_t> frame(640, 480);
        escape(frame.data());
        news = g_newCount - newsBefore;
        bytes = g_newBytes - bytesBefore;

        BINCV_CHECK_EQ(frame.getAlignedWidth(), size_t(20));       // 640 / 32, no padding
        BINCV_CHECK_EQ(frame.planeWords(), size_t(9600));          // 480 * 20
        BINCV_CHECK_EQ(frame.sizeInWords() * sizeof(uint32_t), 3 * binaryBytes);
        BINCV_CHECK_EQ(frame.sizeInWords() * sizeof(uint32_t), size_t(115200));
    }
    // ONE allocation, of exactly 3 x 38400 bytes -- not three allocations, and
    // not a byte of slack. This is the whole memory argument (ARCHITECTURE 4.6).
    BINCV_CHECK_EQ(news, size_t(1));
    BINCV_CHECK_EQ(bytes, 3 * binaryBytes);

    // And the per-plane cost really is the binary frame's cost: the N-bit
    // container is N binary frames, with nothing added for being N of them.
    newsBefore = g_newCount;
    bytesBefore = g_newBytes;
    {
        BinMat<uint32_t> binary(640, 480);
        escape(binary.data());
        news = g_newCount - newsBefore;
        bytes = g_newBytes - bytesBefore;
    }
    BINCV_CHECK_EQ(news, size_t(1));
    BINCV_CHECK_EQ(bytes, binaryBytes);

    // Every supported N, checked against the same arithmetic rather than against
    // a table: N planes cost N times one plane, and one allocation each.
    newsBefore = g_newCount;
    bytesBefore = g_newBytes;
    {
        QuantMat<8, uint32_t> eight(640, 480);
        escape(eight.data());
        news = g_newCount - newsBefore;
        bytes = g_newBytes - bytesBefore;
    }
    BINCV_CHECK_EQ(news, size_t(1));
    BINCV_CHECK_EQ(bytes, 8 * binaryBytes);
}

// ---------------------------------------------------------------------------
// T1.5 -- N = 1 is BinMat itself, not a generic instantiation
// ---------------------------------------------------------------------------

void testSinglePlaneIsBinMat() {
    std::cout << "\n--- N = 1 is the hand-written path ---\n";

    // The strongest form this claim can take: the two names are ONE type. There
    // is no QuantMat<1> wrapper object, no forwarding layer and no plane loop to
    // measure, because the generic template is never instantiated at N = 1 --
    // the hand-written specialization in binMat.hpp is what the compiler picks.
    static_assert(std::is_same<BinMat<uint32_t>, QuantMat<1, uint32_t>>::value,
                  "BinMat must BE QuantMat<1>, not merely wrap it");
    static_assert(std::is_same<BinMat<>, QuantMat<1, uint32_t>>::value,
                  "the default word type must survive the alias");
    static_assert(std::is_same<bincv::BinMat8, QuantMat<1, uint8_t>>::value, "");
    static_assert(std::is_same<bincv::BinMat64, QuantMat<1, uint64_t>>::value, "");
    static_assert(BinMat<uint32_t>::Planes == 1, "");
    BINCV_CHECK(true);  // the four static_asserts above, counted once

    // Consequently the whole BinMat API is available on a QuantMat<1>, and it is
    // the specialized single-plane implementation that runs: fill() writes words,
    // countNonZero() reads them, neither iterates planes.
    QuantMat<1, uint32_t> single(70, 3);
    single.fill(true);
    BINCV_CHECK_EQ(single.countNonZero(), 210);
    single.set(1, 5, false);
    BINCV_CHECK(!single.at(1, 5));
    BINCV_CHECK_EQ(single.countNonZero(), 209);

    // plane(0) is view() -- same pointer, same stride, no offset arithmetic.
    BINCV_CHECK(single.plane(0).ptr == single.view().ptr);
    BINCV_CHECK_EQ(single.plane(0).stride, single.view().stride);
    BINCV_CHECK_EQ(single.planeWords(), single.sizeInWords());

    // A QuantMat<1> hands its plane to the same kernel-facing view type as any
    // other plane, which is what lets a kernel be written once for QuantMat<N>.
    BinMat<uint32_t>& asBinMat = single;
    BINCV_CHECK(asBinMat.plane(0).ptr == single.data());

    // Size is not paid for the plane machinery either: a QuantMat<1> is exactly
    // the BinMat it is, and nothing is stored per plane.
    static_assert(sizeof(QuantMat<1, uint32_t>) == sizeof(BinMat<uint32_t>), "");
    BINCV_CHECK_EQ(sizeof(QuantMat<3, uint32_t>), sizeof(BinMat<uint32_t>));
}

// ---------------------------------------------------------------------------
// T1.5 -- planes are independent, and plane(0) is the LSB
// ---------------------------------------------------------------------------

template <size_t N, typename W>
void testPlanesIndependent(const char* label) {
    std::cout << "\n--- QuantMat<" << N << ", " << label << "> plane independence ---\n";

    const int width = 37;
    const int height = 4;
    QuantMat<N, W> m(width, height);

    // Writing one plane through its view must not disturb any other plane. Each
    // plane gets a different column so a mix-up shows up as a wrong column too.
    for (size_t p = 0; p < N; ++p) {
        bincv::BinMatView<W> v = m.plane(p);
        setRawBit(v.row(1), p, true);
    }
    bool independent = true;
    for (size_t p = 0; p < N; ++p) {
        bincv::BinMatConstView<W> v = m.plane(p);
        for (size_t x = 0; x < static_cast<size_t>(width); ++x) {
            const bool expected = (x == p);
            if (rawBit(v.row(1), x) != expected) independent = false;
        }
        // and no other row was touched
        for (size_t y = 0; y < static_cast<size_t>(height); ++y) {
            if (y == 1) continue;
            for (size_t x = 0; x < static_cast<size_t>(width); ++x) {
                if (rawBit(v.row(y), x)) independent = false;
            }
        }
    }
    BINCV_CHECK(independent);

    // plane(0) is the LEAST significant bit: setting the pixel to 1 must light
    // plane 0 and nothing else, and setting it to 2^p must light plane p alone.
    QuantMat<N, W> lsb(width, height);
    bool lsbCorrect = true;
    for (size_t p = 0; p < N; ++p) {
        lsb.set(2, 3, static_cast<unsigned>(1u << p));
        for (size_t q = 0; q < N; ++q) {
            const bool bitSet = rawBit(lsb.plane(q).row(2), 3);
            if (bitSet != (q == p)) lsbCorrect = false;
        }
    }
    BINCV_CHECK(lsbCorrect);

    // Read back through the container: a value written bit-by-bit into the
    // planes is assembled with plane p as bit p.
    QuantMat<N, W> assembled(width, height);
    const unsigned pattern = static_cast<unsigned>(0xB5u) & QuantMat<N, W>::MaxValue;
    for (size_t p = 0; p < N; ++p) {
        if ((pattern >> p) & 1u) setRawBit(assembled.plane(p).row(0), 9, true);
    }
    BINCV_CHECK_EQ(static_cast<unsigned>(assembled.at(0, 9)), pattern);
}

// ---------------------------------------------------------------------------
// T1.5 -- round trip, every N in 1..8
// ---------------------------------------------------------------------------

template <size_t N, typename W>
void testRoundTrip(const char* label) {
    std::cout << "\n--- QuantMat<" << N << ", " << label << "> round trip ---\n";

    // Widths on both sides of a word boundary and one that is not a multiple of
    // any supported word size.
    const int widths[] = {1, 33, 70};
    const int height = 6;

    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); ++wi) {
        const int width = widths[wi];
        QuantMat<N, W> m(width, height);
        const unsigned maxValue = QuantMat<N, W>::MaxValue;

        // A known pattern per pixel, chosen so neighbouring pixels differ in the
        // low bit (which is where a plane mix-up shows) and the pattern is not
        // constant down a column.
        auto expectedAt = [&](int y, int x) -> unsigned {
            return static_cast<unsigned>((y * 7 + x * 3 + 1)) & maxValue;
        };

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                m.set(y, x, expectedAt(y, x));
            }
        }

        bool roundTripped = true;
        bool planesAgree = true;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const unsigned expected = expectedAt(y, x);
                // Path 1: the container's own accessor.
                if (static_cast<unsigned>(m.at(y, x)) != expected) roundTripped = false;
                // Path 2: reassembled from the raw plane words, which is the
                // independent check -- at() agreeing with set() proves nothing
                // about where the bits actually went.
                unsigned fromPlanes = 0;
                for (size_t p = 0; p < N; ++p) {
                    if (rawBit(m.plane(p).row(static_cast<size_t>(y)),
                               static_cast<size_t>(x))) {
                        fromPlanes |= (1u << p);
                    }
                }
                if (fromPlanes != expected) planesAgree = false;
            }
        }
        BINCV_CHECK(roundTripped);
        BINCV_CHECK(planesAgree);

        // Overwriting must replace, not accumulate: set() writes every plane,
        // including the ones whose bit is zero.
        m.set(0, 0, maxValue);
        m.set(0, 0, 0u);
        BINCV_CHECK_EQ(static_cast<unsigned>(m.at(0, 0)), 0u);

        // Padding bits stay zero through all of that, in EVERY plane. Word-wise
        // reductions read the trailing partial word whole, so a stray bit past
        // `width` would be counted as a pixel that does not exist.
        bool paddingClear = true;
        for (size_t p = 0; p < N; ++p) {
            bincv::BinMatConstView<W> v = m.plane(p);
            for (size_t y = 0; y < static_cast<size_t>(height); ++y) {
                if (!paddingBitsClear(v.row(y), static_cast<size_t>(width), v.stride)) {
                    paddingClear = false;
                }
            }
        }
        BINCV_CHECK(paddingClear);
    }
}

// ---------------------------------------------------------------------------
// T1.5 -- non-owning wrap: the Tier 2 path, allocating nothing
// ---------------------------------------------------------------------------

void testWrapAllocatesNothing() {
    std::cout << "\n--- Non-owning wrap ---\n";

    // The caller's one buffer holds all three planes, plane 0 first. Sized by
    // the same arithmetic the container uses, and it is the caller's to lay out.
    constexpr size_t width = 70;
    constexpr size_t height = 3;
    constexpr size_t stride = (width + 31) / 32;      // 3 words at 32 bits
    constexpr size_t planes = 3;
    static uint32_t buffer[planes * height * stride] = {};

    // The measured region contains no checks at all: the check macros allocate
    // (std::to_string) and free, so a counter read across a region that checks
    // as it goes measures the checking. Observations are recorded into plain
    // locals here and asserted after the counters have been read.
    bool wrapOwnsNothing = false;
    bool wrapDataIsBuffer = false;
    size_t wrapWords = 0;
    size_t wrapPlaneWords = 0;
    bool planeOffsetsCorrect = false;
    bool wroteThroughToBuffer = false;
    unsigned readBackThroughBuffer = 0;

    const size_t newsBefore = g_newCount;
    const size_t deletesBefore = g_deleteCount;
    {
        QuantMat<3, uint32_t> wrapped(buffer, static_cast<int>(width),
                                      static_cast<int>(height), stride);
        escape(wrapped.data());

        wrapOwnsNothing = !wrapped.ownsMemory();
        wrapDataIsBuffer = wrapped.data() == buffer;
        wrapWords = wrapped.sizeInWords();
        wrapPlaneWords = wrapped.planeWords();

        // Plane p really is the caller's memory at offset p * planeWords.
        planeOffsetsCorrect = wrapped.plane(1).ptr == buffer + height * stride &&
                              wrapped.plane(2).ptr == buffer + 2 * height * stride;

        // Writes land in the caller's buffer, at the word the packing says.
        wrapped.set(2, 33, 5u);                       // 0b101: planes 0 and 2
        wroteThroughToBuffer =
            rawBit(buffer + 2 * stride, size_t(33)) &&                          // plane 0
            !rawBit(buffer + height * stride + 2 * stride, size_t(33)) &&       // plane 1
            rawBit(buffer + 2 * height * stride + 2 * stride, size_t(33));      // plane 2

        // And the container reads back what is in the caller's buffer, including
        // bits written behind its back -- the DMA / sensor ingest case.
        setRawBit(buffer + height * stride + 2 * stride, size_t(33), true);
        readBackThroughBuffer = static_cast<unsigned>(wrapped.at(2, 33));
    }
    const size_t news = g_newCount - newsBefore;
    const size_t deletes = g_deleteCount - deletesBefore;

    BINCV_CHECK_EQ(news, size_t(0));      // the Tier 2 claim, measured
    BINCV_CHECK_EQ(deletes, size_t(0));   // and nothing freed the caller's buffer
    BINCV_CHECK(wrapOwnsNothing);
    BINCV_CHECK(wrapDataIsBuffer);
    BINCV_CHECK_EQ(wrapWords, planes * height * stride);
    BINCV_CHECK_EQ(wrapPlaneWords, height * stride);
    BINCV_CHECK(planeOffsetsCorrect);
    BINCV_CHECK(wroteThroughToBuffer);
    BINCV_CHECK_EQ(readBackThroughBuffer, 7u);

    // An empty container touches the allocator too -- which is to say, it does not.
    bool emptyIsEmpty = false;
    size_t emptyHeight = 1;
    size_t emptyWords = 1;
    size_t emptyPlaneWords = 1;
    bool emptyOwnsNothing = false;
    const size_t emptyNewsBefore = g_newCount;
    {
        QuantMat<4, uint32_t> emptyMat;
        escape(emptyMat.data());
        emptyIsEmpty = emptyMat.empty();
        emptyHeight = emptyMat.getHeight();
        emptyWords = emptyMat.sizeInWords();
        emptyPlaneWords = emptyMat.planeWords();
        emptyOwnsNothing = !emptyMat.ownsMemory();
    }
    const size_t emptyNews = g_newCount - emptyNewsBefore;
    BINCV_CHECK_EQ(emptyNews, size_t(0));
    BINCV_CHECK(emptyIsEmpty);
    BINCV_CHECK_EQ(emptyHeight, size_t(0));
    BINCV_CHECK_EQ(emptyWords, size_t(0));
    BINCV_CHECK_EQ(emptyPlaneWords, size_t(0));
    BINCV_CHECK(emptyOwnsNothing);
}

// ---------------------------------------------------------------------------
// T1.5/T1.6 -- the plane index is validated, in every build
//
// REGRESSION. plane() used to be debug-checked, and all three verified
// configurations are Release, so the check was compiled out of every one of
// them. Measured in that state, one identical caller mistake had three unrelated
// outcomes across one type family:
//   QuantMat<3>::plane(3)             -> a writable view of the NEXT allocation
//                                        (a.plane(3).ptr == b.data() exactly; a
//                                        plane-wide write is a 37.5 KiB heap
//                                        overflow at 640x480)
//   BinMat::plane(7)                  -> quietly plane 0
//   SignedQuantMat<2>::magnitude(2)   -> the SIGN plane, in bounds and therefore
//                                        invisible to every sanitizer
// It is one behaviour now, and it is a throw. That is what makes index handling
// testable at one N and carried to the others.
// ---------------------------------------------------------------------------

void testPlaneIndexIsChecked() {
    std::cout << "\n--- Plane index: checked in every build ---\n";

    QuantMat<3, uint32_t> m(70, 5);
    const QuantMat<3, uint32_t>& cm = m;
    BINCV_CHECK_THROWS(m.plane(3), std::out_of_range);
    BINCV_CHECK_THROWS(cm.plane(3), std::out_of_range);
    BINCV_CHECK_THROWS(m.constPlane(3), std::out_of_range);
    BINCV_CHECK_THROWS(m.plane(size_t(-1)), std::out_of_range);

    // N = 1 answers the same way. Before, it discarded the index and returned
    // plane 0 -- the one spelling where a wrong index looked like success.
    BinMat<uint32_t> b(70, 5);
    const BinMat<uint32_t>& cb = b;
    BINCV_CHECK_THROWS(b.plane(1), std::out_of_range);
    BINCV_CHECK_THROWS(cb.plane(7), std::out_of_range);
    BINCV_CHECK_THROWS(b.constPlane(1), std::out_of_range);

    // And the sign plane cannot be reached through the magnitude accessor. i == N
    // is in bounds on the underlying QuantMat<N+1>, so this is the index where an
    // unchecked off-by-one produced wrong numbers with no memory error at all.
    SignedQuantMat<2, uint32_t> s(64, 4);
    const SignedQuantMat<2, uint32_t>& cs = s;
    BINCV_CHECK_THROWS(s.magnitude(2), std::out_of_range);
    BINCV_CHECK_THROWS(cs.magnitude(2), std::out_of_range);
    BINCV_CHECK_THROWS(s.constMagnitude(2), std::out_of_range);
    BINCV_CHECK_THROWS(s.magnitude(3), std::out_of_range);

    // The valid indices are untouched by the check, and still address the planes
    // the layout says they do -- including the sign plane through sign(), which is
    // the accessor that is allowed to reach plane N. Driven through the const
    // references too: each const overload carries its own copy of the check, so
    // each has to be exercised on both sides of the boundary. (It also keeps them
    // used in the no-exceptions build, where BINCV_CHECK_THROWS does not evaluate
    // its expression.)
    bool validIndicesUnchanged = true;
    for (size_t p = 0; p < 3; ++p) {
        if (m.plane(p).ptr != m.data() + p * m.planeWords()) validIndicesUnchanged = false;
        if (cm.plane(p).ptr != cm.data() + p * cm.planeWords()) validIndicesUnchanged = false;
    }
    for (size_t p = 0; p < 2; ++p) {
        if (s.magnitude(p).ptr != s.data() + p * s.planeWords()) validIndicesUnchanged = false;
        if (cs.magnitude(p).ptr != cs.data() + p * cs.planeWords()) validIndicesUnchanged = false;
    }
    if (s.sign().ptr != s.data() + 2 * s.planeWords()) validIndicesUnchanged = false;
    if (cs.sign().ptr != cs.data() + 2 * cs.planeWords()) validIndicesUnchanged = false;
    if (b.plane(0).ptr != b.data()) validIndicesUnchanged = false;
    if (cb.plane(0).ptr != cb.data()) validIndicesUnchanged = false;
    BINCV_CHECK(validIndicesUnchanged);

    // The sign plane and the last magnitude plane are different memory, which is
    // the fact the check protects.
    BINCV_CHECK(s.sign().ptr != s.magnitude(1).ptr);
}

// ---------------------------------------------------------------------------
// T1.5/T1.6 -- const views of a plane, from a NON-const matrix
//
// REGRESSION. A plane could not be handed to a kernel declared
// `f(BinMatConstView<W>)`: on a non-const matrix the mutable overload wins, and
// template argument deduction does not consider the BinMatView ->
// BinMatConstView conversion (view.hpp:77-85). That is the exact call shape T3.6
// specifies -- `countNonZero(dx.magnitude(0), window)` -- and the reason T1.3
// gave BinMat a constView() callable on a non-const object. The named const
// accessors are that, per plane.
// ---------------------------------------------------------------------------

/// @brief Stands in for a T2/T3 kernel: deduces W from a read-only view.
template <typename W>
size_t constViewKernel(bincv::BinMatConstView<W> v) { return v.width * v.height; }

void testConstPlaneAccessors() {
    std::cout << "\n--- constPlane / constMagnitude / constSign ---\n";

    QuantMat<3, uint32_t> q(70, 5);
    TernaryMat<> dx(64, 4);
    BinMat<uint32_t> b(70, 5);

    // Compiling at all is the check: each of these would be
    // "couldn't deduce template parameter W" through the mutable accessor.
    BINCV_CHECK_EQ(constViewKernel(q.constPlane(0)), size_t(350));
    BINCV_CHECK_EQ(constViewKernel(dx.constMagnitude(0)), size_t(256));
    BINCV_CHECK_EQ(constViewKernel(dx.constSign()), size_t(256));
    BINCV_CHECK_EQ(constViewKernel(b.constPlane(0)), size_t(350));

    static_assert(std::is_same<decltype(q.constPlane(0)),
                               bincv::BinMatConstView<uint32_t>>::value,
                  "constPlane must return the read-only view type");
    static_assert(std::is_same<decltype(dx.constMagnitude(0)),
                               bincv::BinMatConstView<uint32_t>>::value, "");
    static_assert(std::is_same<decltype(dx.constSign()),
                               bincv::BinMatConstView<uint32_t>>::value, "");
    static_assert(std::is_same<decltype(b.constPlane(0)),
                               bincv::BinMatConstView<uint32_t>>::value, "");
    BINCV_CHECK(true);  // the static_asserts above, counted once

    // A view of the same memory, not a copy of it -- these are renamings, and
    // they describe the plane the mutable accessor describes.
    BINCV_CHECK(q.constPlane(2).ptr == q.plane(2).ptr);
    BINCV_CHECK_EQ(q.constPlane(2).stride, q.plane(2).stride);
    BINCV_CHECK(dx.constMagnitude(0).ptr == dx.magnitude(0).ptr);
    BINCV_CHECK(dx.constSign().ptr == dx.sign().ptr);
    BINCV_CHECK(b.constPlane(0).ptr == b.constView().ptr);

    // Writes through the mutable accessor are visible through the const one.
    dx.set(1, 7, -1);
    BINCV_CHECK(rawBit(dx.constMagnitude(0).row(1), size_t(7)));
    BINCV_CHECK(rawBit(dx.constSign().row(1), size_t(7)));
}

// ---------------------------------------------------------------------------
// T1.5/T1.6 -- wrap() checks the buffer length the constructor cannot
//
// REGRESSION. The wrapping constructor is spelled exactly like BinMat's -- same
// four arguments -- but needs N times as many words, and is handed no length, so
// an undersized buffer was accepted with no diagnostic even though the container
// knows exactly how many words it needs. wrap() is the spelling that is told.
// ---------------------------------------------------------------------------

void testWrapChecksBufferLength() {
    std::cout << "\n--- wrap(): the buffer length is checked ---\n";

    constexpr size_t width = 64;
    constexpr size_t height = 8;
    constexpr size_t stride = 2;                      // 64 / 32
    constexpr size_t exact = 3 * height * stride;     // 48 words for QuantMat<3>
    static uint32_t buffer[exact] = {};

    // Aliased because BINCV_CHECK_THROWS takes exactly two macro arguments, and a
    // template argument list's comma is not hidden from the preprocessor by the
    // angle brackets the way a function call's is.
    using Q3 = QuantMat<3, uint32_t>;
    using S2 = SignedQuantMat<2, uint32_t>;

    // Exactly enough is enough: same object the constructor builds, same lack of
    // ownership, and still no allocation.
    bool wrappedOk = false;
    bool ownsNothing = false;
    size_t words = 0;
    const size_t newsBefore = g_newCount;
    {
        Q3 m = Q3::wrap(
            buffer, exact, static_cast<int>(width), static_cast<int>(height), stride);
        escape(m.data());
        wrappedOk = m.data() == buffer &&
                    m.plane(2).ptr == buffer + 2 * height * stride &&
                    m.plane(2).ptr + m.planeWords() == buffer + exact;
        ownsNothing = !m.ownsMemory();
        words = m.sizeInWords();
    }
    const size_t news = g_newCount - newsBefore;
    BINCV_CHECK_EQ(news, size_t(0));
    BINCV_CHECK(wrappedOk);
    BINCV_CHECK(ownsNothing);
    BINCV_CHECK_EQ(words, exact);

    // One word short is rejected -- and the boundary is where the arithmetic says,
    // not somewhere near it.
    BINCV_CHECK_THROWS(Q3::wrap(buffer, exact - 1, 64, 8, 2), std::invalid_argument);
    // What the unchecked constructor accepts silently: a buffer sized for ONE
    // plane, handed to a three-plane container.
    BINCV_CHECK_THROWS(Q3::wrap(buffer, height * stride, 64, 8, 2), std::invalid_argument);

    // The signed container needs N+1 planes' worth, one more than its type
    // parameter reads as -- which is the easiest length to get wrong.
    BINCV_CHECK_THROWS(S2::wrap(buffer, 2 * height * stride, 64, 8, 2), std::invalid_argument);
    bool signedWrapOk = false;
    {
        S2 s = S2::wrap(buffer, 3 * height * stride, 64, 8, 2);
        signedWrapOk = !s.ownsMemory() && s.data() == buffer &&
                       s.sign().ptr == buffer + 2 * height * stride;
    }
    BINCV_CHECK(signedWrapOk);

    // The other checks still report first and in their own words: wrap() adds a
    // length check, it does not take over the constructor's diagnostics.
    BINCV_CHECK_THROWS(Q3::wrap(buffer, exact, 64, 8, 1),
                       std::invalid_argument);   // stride shorter than a row
    BINCV_CHECK_THROWS(Q3::wrap(buffer, exact, -1, 8, 2),
                       std::invalid_argument);   // negative width

    // An empty matrix requires nothing, and wrap() must not invent a requirement.
    bool emptyWrapOk = false;
    {
        Q3 emptyWrap = Q3::wrap(nullptr, 0, 0, 0, 0);
        emptyWrapOk = emptyWrap.empty() && emptyWrap.sizeInWords() == 0;
    }
    BINCV_CHECK(emptyWrapOk);
}

// ---------------------------------------------------------------------------
// T1.6 -- the magnitude is taken in unsigned arithmetic
//
// REGRESSION. set() computed it as `-value`, which at INT_MIN is signed-overflow
// UB -- reported by UBSan at both -O0 and -O2, and licensing the optimizer to
// assume the `value < 0` branch it guards is never taken. The range assert above
// it is compiled out under NDEBUG, i.e. in every configuration this project
// verifies, so nothing kept INT_MIN away from it.
//
// The static_assert is the regression test, and it is one that cannot be
// skipped: a constant expression may not contain signed overflow, so rewriting
// the helper as `-value` makes this file fail to COMPILE, in all three
// configurations at once.
// ---------------------------------------------------------------------------

void testSignedMagnitudeIsUnsigned() {
    std::cout << "\n--- Magnitude is computed in unsigned arithmetic ---\n";

    static_assert(bincv::impl::signedMagnitude(INT_MIN) == 2147483648u,
                  "the most negative int must have a defined magnitude; `-value` is UB here");
    static_assert(bincv::impl::signedMagnitude(-1) == 1u, "");
    static_assert(bincv::impl::signedMagnitude(0) == 0u, "");
    static_assert(bincv::impl::signedMagnitude(7) == 7u, "");
    static_assert(bincv::impl::signedMagnitude(INT_MAX) == 2147483647u, "");
    BINCV_CHECK(true);  // the static_asserts above, counted once

    // And the values a caller may legally write still land where they did: the
    // extreme of the valid range for every magnitude width, both signs.
    SignedQuantMat<7, uint32_t> wide(40, 2);
    const int maxMagnitude = static_cast<int>(SignedQuantMat<7, uint32_t>::MaxMagnitude);
    wide.set(0, 3, -maxMagnitude);
    wide.set(0, 4, maxMagnitude);
    BINCV_CHECK_EQ(wide.at(0, 3), -maxMagnitude);
    BINCV_CHECK_EQ(wide.at(0, 4), maxMagnitude);
    BINCV_CHECK_EQ(wide.magnitudeAt(0, 3), static_cast<unsigned>(maxMagnitude));
}

// ---------------------------------------------------------------------------
// T1.5 -- value semantics (D-8), inherited from the storage layer
// ---------------------------------------------------------------------------

void testValueSemantics() {
    std::cout << "\n--- Value semantics ---\n";

    QuantMat<3, uint32_t> original(40, 3);
    original.set(1, 7, 5u);

    // Copy means deep copy: a second allocation, and the two do not alias.
    size_t newsBefore = g_newCount;
    QuantMat<3, uint32_t> copy(original);
    const size_t news = g_newCount - newsBefore;
    BINCV_CHECK_EQ(news, size_t(1));
    BINCV_CHECK(copy.data() != original.data());
    BINCV_CHECK_EQ(static_cast<unsigned>(copy.at(1, 7)), 5u);
    copy.set(1, 7, 2u);
    BINCV_CHECK_EQ(static_cast<unsigned>(original.at(1, 7)), 5u);

    // Move transfers the block rather than copying it, and leaves a valid empty
    // container behind.
    const uint32_t* originalData = original.data();
    newsBefore = g_newCount;
    QuantMat<3, uint32_t> moved(std::move(original));
    const size_t moveNews = g_newCount - newsBefore;  // read before any check allocates
    BINCV_CHECK_EQ(moveNews, size_t(0));
    BINCV_CHECK(moved.data() == originalData);
    BINCV_CHECK_EQ(static_cast<unsigned>(moved.at(1, 7)), 5u);
    BINCV_CHECK(original.empty());          // NOLINT: use-after-move is the point
    BINCV_CHECK_EQ(original.getHeight(), size_t(0));

    // Copying a WRAPPED container still deep-copies into owned storage (D-8):
    // the rule is a property of the container, not of how the source was built.
    static uint32_t buffer[3 * 2 * 2] = {};
    QuantMat<3, uint32_t> wrapped(buffer, 40, 2, 2);
    QuantMat<3, uint32_t> ownedCopy(wrapped);
    BINCV_CHECK(!wrapped.ownsMemory());
    BINCV_CHECK(ownedCopy.ownsMemory());
    BINCV_CHECK(ownedCopy.data() != buffer);
}

// ---------------------------------------------------------------------------
// T1.6 -- sign-magnitude, and the canonical-zero rule
// ---------------------------------------------------------------------------

void testTernaryRoundTrip() {
    std::cout << "\n--- TernaryMat round trip ---\n";

    static_assert(std::is_same<TernaryMat<uint32_t>, SignedQuantMat<1, uint32_t>>::value,
                  "ternary is the N=1 case of the general signed form");
    static_assert(TernaryMat<>::Planes == 2, "one magnitude plane plus one sign plane");
    static_assert(TernaryMat<>::MagnitudePlanes == 1, "");
    static_assert(TernaryMat<>::SignPlaneIndex == 1, "the sign plane is plane N");
    static_assert(TernaryMat<>::MaxMagnitude == 1u, "");
    BINCV_CHECK(true);  // the static_asserts above, counted once

    const int width = 37;
    const int height = 5;
    TernaryMat<> t(width, height);

    // All three values, per pixel, cycling so that every value meets every
    // column phase within a word.
    auto expectedAt = [](int y, int x) -> int { return ((y + x) % 3) - 1; };  // -1, 0, +1

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            t.set(y, x, expectedAt(y, x));
        }
    }

    bool roundTripped = true;
    bool planesAgree = true;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int expected = expectedAt(y, x);
            if (t.at(y, x) != expected) roundTripped = false;

            // Independently, from the two planes: magnitude says whether the
            // pixel is non-zero, the sign plane says which way.
            const bool magnitudeBit =
                rawBit(t.magnitude(0).row(static_cast<size_t>(y)), static_cast<size_t>(x));
            const bool signBit =
                rawBit(t.sign().row(static_cast<size_t>(y)), static_cast<size_t>(x));
            const int fromPlanes = magnitudeBit ? (signBit ? -1 : 1) : 0;
            if (fromPlanes != expected) planesAgree = false;
        }
    }
    BINCV_CHECK(roundTripped);
    BINCV_CHECK(planesAgree);

    // Two bits per pixel, in one allocation: 37 columns is 2 words at 32 bits,
    // 5 rows, 2 planes.
    BINCV_CHECK_EQ(t.getAlignedWidth(), size_t(2));
    BINCV_CHECK_EQ(t.planeWords(), size_t(10));
    BINCV_CHECK_EQ(t.sizeInWords(), size_t(20));

    // Padding bits stay clear in both planes.
    bool paddingClear = true;
    for (size_t y = 0; y < static_cast<size_t>(height); ++y) {
        if (!paddingBitsClear(t.magnitude(0).row(y), static_cast<size_t>(width),
                              t.getAlignedWidth())) paddingClear = false;
        if (!paddingBitsClear(t.sign().row(y), static_cast<size_t>(width),
                              t.getAlignedWidth())) paddingClear = false;
    }
    BINCV_CHECK(paddingClear);
}

void testCanonicalZero() {
    std::cout << "\n--- Canonical zero: magnitude 0 ignores the sign bit ---\n";

    TernaryMat<> t(64, 2);

    // Half of the rule: a sign bit set over a ZERO magnitude does not create a
    // "-0". It reads as 0, exactly as a clear sign bit does. This is what lets a
    // kernel write the two planes independently and skip a fix-up pass.
    setRawBit(t.sign().row(0), size_t(5), true);
    BINCV_CHECK(rawBit(t.sign().row(0), size_t(5)));       // the bit really is set
    BINCV_CHECK(!rawBit(t.magnitude(0).row(0), size_t(5)));
    BINCV_CHECK_EQ(t.at(0, 5), 0);                         // and the pixel is still zero
    BINCV_CHECK_EQ(t.magnitudeAt(0, 5), 0u);

    // Both spellings of zero are the same value; there is no -0 to distinguish.
    // Stated as the pair of facts that gives the rule its content: the two
    // encodings DIFFER in storage -- the uninterpreted container can see that
    // one has its top plane set -- and AGREE in value, because the sign-magnitude
    // reading is what erases the distinction. Without both halves, "the sign bit
    // is ignored" would be untestable: an int has no -0 for at() to return.
    setRawBit(t.sign().row(0), size_t(6), false);
    BINCV_CHECK(t.planes().at(0, 5) != t.planes().at(0, 6));  // storage differs
    BINCV_CHECK_EQ(t.planes().at(0, 5), 2u);                  // sign plane set, magnitude clear
    BINCV_CHECK_EQ(t.planes().at(0, 6), 0u);
    BINCV_CHECK_EQ(t.at(0, 6), t.at(0, 5));                   // value does not

    // The other half: writing 0 CANONICALIZES, clearing the sign bit as well as
    // the magnitude, so a value written through set() never leaves a stray sign
    // bit for a later reader to puzzle over.
    t.set(1, 9, -1);
    BINCV_CHECK(rawBit(t.sign().row(1), size_t(9)));
    BINCV_CHECK(rawBit(t.magnitude(0).row(1), size_t(9)));
    t.set(1, 9, 0);
    BINCV_CHECK(!rawBit(t.sign().row(1), size_t(9)));      // sign cleared, not left set
    BINCV_CHECK(!rawBit(t.magnitude(0).row(1), size_t(9)));
    BINCV_CHECK_EQ(t.at(1, 9), 0);

    // +1 and -1 differ in the sign plane only.
    t.set(1, 10, 1);
    t.set(1, 11, -1);
    BINCV_CHECK(rawBit(t.magnitude(0).row(1), size_t(10)));
    BINCV_CHECK(rawBit(t.magnitude(0).row(1), size_t(11)));
    BINCV_CHECK(!rawBit(t.sign().row(1), size_t(10)));
    BINCV_CHECK(rawBit(t.sign().row(1), size_t(11)));
    BINCV_CHECK_EQ(t.at(1, 10), 1);
    BINCV_CHECK_EQ(t.at(1, 11), -1);

    // The rule is not ternary-specific: it holds for every magnitude width.
    SignedQuantMat<3, uint32_t> wide(64, 2);
    setRawBit(wide.sign().row(0), size_t(4), true);
    BINCV_CHECK_EQ(wide.at(0, 4), 0);
    wide.set(0, 4, -7);
    BINCV_CHECK_EQ(wide.at(0, 4), -7);
    BINCV_CHECK_EQ(wide.magnitudeAt(0, 4), 7u);
    wide.set(0, 4, 0);
    BINCV_CHECK(!rawBit(wide.sign().row(0), size_t(4)));
    BINCV_CHECK_EQ(wide.at(0, 4), 0);
}

template <size_t N>
void testSignedRoundTrip(const char* label) {
    std::cout << "\n--- SignedQuantMat<" << label << "> round trip ---\n";

    const int width = 33;
    const int height = 4;
    SignedQuantMat<N, uint32_t> m(width, height);
    const int maxMagnitude = static_cast<int>(SignedQuantMat<N, uint32_t>::MaxMagnitude);
    const int span = 2 * maxMagnitude + 1;  // -max .. +max inclusive

    auto expectedAt = [&](int y, int x) -> int {
        return ((y * 5 + x) % span) - maxMagnitude;
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            m.set(y, x, expectedAt(y, x));
        }
    }

    bool roundTripped = true;
    bool planesAgree = true;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int expected = expectedAt(y, x);
            if (m.at(y, x) != expected) roundTripped = false;

            unsigned magnitude = 0;
            for (size_t p = 0; p < N; ++p) {
                if (rawBit(m.magnitude(p).row(static_cast<size_t>(y)),
                           static_cast<size_t>(x))) {
                    magnitude |= (1u << p);
                }
            }
            const bool negative =
                rawBit(m.sign().row(static_cast<size_t>(y)), static_cast<size_t>(x));
            const int fromPlanes =
                magnitude == 0 ? 0 : (negative ? -static_cast<int>(magnitude)
                                               : static_cast<int>(magnitude));
            if (fromPlanes != expected) planesAgree = false;
        }
    }
    BINCV_CHECK(roundTripped);
    BINCV_CHECK(planesAgree);
}

// The wrapper adds a reading of the planes and not a byte of storage.
void testNoStorageDuplication() {
    std::cout << "\n--- SignedQuantMat<N> is QuantMat<N+1>, byte for byte ---\n";

    static_assert(sizeof(TernaryMat<>) == sizeof(QuantMat<2, uint32_t>),
                  "a signed matrix must not be larger than the container it wraps");
    static_assert(sizeof(SignedQuantMat<3, uint32_t>) == sizeof(QuantMat<4, uint32_t>), "");
    static_assert(sizeof(SignedQuantMat<7, uint64_t>) == sizeof(QuantMat<8, uint64_t>), "");
    static_assert(std::is_same<TernaryMat<>::Container, QuantMat<2, uint32_t>>::value, "");
    BINCV_CHECK(true);  // the static_asserts above, counted once

    const int width = 640;
    const int height = 480;

    size_t signedNews = 0;
    size_t signedBytes = 0;
    size_t plainNews = 0;
    size_t plainBytes = 0;

    size_t newsBefore = g_newCount;
    size_t bytesBefore = g_newBytes;
    {
        SignedQuantMat<3, uint32_t> signedMat(width, height);
        escape(signedMat.data());
        signedNews = g_newCount - newsBefore;
        signedBytes = g_newBytes - bytesBefore;

        newsBefore = g_newCount;
        bytesBefore = g_newBytes;
        {
            QuantMat<4, uint32_t> plain(width, height);
            escape(plain.data());
            plainNews = g_newCount - newsBefore;
            plainBytes = g_newBytes - bytesBefore;

            BINCV_CHECK_EQ(signedMat.sizeInWords(), plain.sizeInWords());
            BINCV_CHECK_EQ(signedMat.planeWords(), plain.planeWords());
            BINCV_CHECK_EQ(signedMat.getAlignedWidth(), plain.getAlignedWidth());
        }
    }
    // One allocation each, of the same size: the wrapper has no storage path of
    // its own to allocate through.
    BINCV_CHECK_EQ(signedNews, size_t(1));
    BINCV_CHECK_EQ(plainNews, size_t(1));
    BINCV_CHECK_EQ(signedBytes, plainBytes);
    BINCV_CHECK_EQ(signedBytes, 4 * size_t(38400));

    // The accessors are renamings of the underlying planes, addressing the same
    // words -- not copies of them.
    SignedQuantMat<3, uint32_t> m(70, 3);
    BINCV_CHECK(m.data() == m.planes().data());
    BINCV_CHECK(m.magnitude(0).ptr == m.planes().plane(0).ptr);
    BINCV_CHECK(m.magnitude(2).ptr == m.planes().plane(2).ptr);
    BINCV_CHECK(m.sign().ptr == m.planes().plane(3).ptr);
    BINCV_CHECK(m.sign().ptr == m.data() + 3 * m.planeWords());

    // Writing through the interpreted accessor is visible through the
    // uninterpreted one: same memory, two readings of it.
    m.set(1, 5, -3);
    BINCV_CHECK_EQ(m.planes().at(1, 5), 3u | (1u << 3));  // magnitude 3, sign bit set
    TernaryMat<> constructedEmpty;
    BINCV_CHECK(constructedEmpty.empty());
    BINCV_CHECK_EQ(constructedEmpty.sizeInWords(), size_t(0));

    // A signed matrix wraps a caller's buffer through the same path, allocating
    // nothing: N+1 planes laid out plane 0 first, sign plane last.
    static uint32_t buffer[2 * 3 * 3] = {};
    bool signedWrapOwnsNothing = false;
    bool signedWrapPlanesPlaced = false;
    bool signedWrapWroteThrough = false;
    const size_t newsAtWrap = g_newCount;
    {
        TernaryMat<> wrapped(buffer, 70, 3, 3);
        escape(wrapped.data());
        signedWrapOwnsNothing = !wrapped.ownsMemory();
        signedWrapPlanesPlaced = wrapped.magnitude(0).ptr == buffer &&
                                 wrapped.sign().ptr == buffer + 9;
        wrapped.set(0, 40, -1);
        signedWrapWroteThrough = rawBit(buffer, size_t(40)) &&      // magnitude, row 0
                                 rawBit(buffer + 9, size_t(40));    // sign, row 0
    }
    const size_t wrapNews = g_newCount - newsAtWrap;
    BINCV_CHECK_EQ(wrapNews, size_t(0));
    BINCV_CHECK(signedWrapOwnsNothing);
    BINCV_CHECK(signedWrapPlanesPlaced);
    BINCV_CHECK(signedWrapWroteThrough);
}

} // namespace

int main() {
    std::cout << "=== QuantMat / SignedQuantMat core tests (no OpenCV) ===\n";

    // Layout across every supported word width and a spread of N.
    testPlaneLayout<2, uint8_t>("uint8_t");
    testPlaneLayout<3, uint16_t>("uint16_t");
    testPlaneLayout<3, uint32_t>("uint32_t");
    testPlaneLayout<5, uint64_t>("uint64_t");
    testPlaneLayout<8, uint32_t>("uint32_t");

    testFootprint();
    testSinglePlaneIsBinMat();

    testPlanesIndependent<2, uint8_t>("uint8_t");
    testPlanesIndependent<3, uint32_t>("uint32_t");
    testPlanesIndependent<8, uint64_t>("uint64_t");

    // Round trip for every N in 1..8, N = 1 going through BinMat's own paths.
    testRoundTrip<1, uint32_t>("uint32_t");
    testRoundTrip<2, uint32_t>("uint32_t");
    testRoundTrip<3, uint32_t>("uint32_t");
    testRoundTrip<4, uint32_t>("uint32_t");
    testRoundTrip<5, uint32_t>("uint32_t");
    testRoundTrip<6, uint32_t>("uint32_t");
    testRoundTrip<7, uint32_t>("uint32_t");
    testRoundTrip<8, uint32_t>("uint32_t");
    testRoundTrip<3, uint8_t>("uint8_t");
    testRoundTrip<3, uint64_t>("uint64_t");

    testWrapAllocatesNothing();
    testPlaneIndexIsChecked();
    testConstPlaneAccessors();
    testWrapChecksBufferLength();
    testValueSemantics();

    testTernaryRoundTrip();
    testCanonicalZero();
    testSignedRoundTrip<1>("1");
    testSignedRoundTrip<2>("2");
    testSignedRoundTrip<7>("7");
    testSignedMagnitudeIsUnsigned();
    testNoStorageDuplication();

    return bincv::test::summarize("QuantMat core tests");
}
