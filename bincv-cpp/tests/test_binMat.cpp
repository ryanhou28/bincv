// Core BinMat tests. Deliberately free of OpenCV so this suite also builds and
// runs in embedded (core-only) configurations -- see tests/test_opencv_interop.cpp
// for the interop coverage.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <vector>
#include <utility>

#include "bincv-cpp/binMat.hpp"
#include "test_util.hpp"

// ---------------------------------------------------------------------------
// Allocation counters
//
// "Wrapping a caller-provided buffer allocates nothing and frees nothing" is the
// Tier 2 / DMA claim, and it is only a claim if something counts. Replacing the
// global operators is the cheapest way to observe every allocation the process
// makes. Same shape as tests/test_storage.cpp.
// ---------------------------------------------------------------------------

namespace {

size_t g_newCount = 0;
size_t g_deleteCount = 0;

// Keeps the optimizer from eliding an allocation whose result it can prove is
// unused -- C++14 permits new/delete elision, which would make the counters lie.
const void* volatile g_sink = nullptr;
inline void escape(const void* p) { g_sink = p; }

} // namespace

void* operator new(std::size_t bytes) {
    ++g_newCount;
    // Reject unsatisfiable requests up front. array-new passes SIZE_MAX as its
    // overflow sentinel, expecting the allocator to fail; forwarding it to
    // malloc also makes GCC warn (-Walloc-size-larger-than=) once this operator
    // inlines into Storage's array-new at -O2.
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    // Cannot throw std::bad_alloc: this file also has to build with -fno-exceptions.
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}

void* operator new[](std::size_t bytes) { return ::operator new(bytes); }

void operator delete(void* p) noexcept {
    if (p != nullptr) ++g_deleteCount;
    std::free(p);
}

void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

namespace {

/// @brief Reads pixel (y, x) through a view's own row pointer and word arithmetic.
/// @note Deliberately does not go through BinMat, so that comparing this against
///       BinMat::at() actually tests that the view describes the same layout.
template <typename WordType>
bool viewPixel(const bincv::BinMatConstView<WordType>& v, size_t y, size_t x) {
    constexpr size_t bits = bincv::BinMatConstView<WordType>::WordBits;
    return (v.row(y)[x / bits] & (static_cast<WordType>(1) << (x % bits))) != 0;
}

/// @brief Counts every set bit in the backing words, row padding included.
/// @note Deliberately NOT a library operation -- binCV exposes no per-word
///       popcount (D-6). It lives here because it is the only way to see the bits
///       countNonZero()'s per-pixel loop cannot: whenever this disagrees with
///       countNonZero(), the padding-bit invariant is broken and every word-wise
///       reduction built on it would over-count.
template <typename WordType>
int countBitsAcrossStride(const bincv::BinMat<WordType>& m) {
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

// Runs the shared behavioural contract against one word type, so the packing
// logic is exercised at 8/16/32/64 bits rather than only the default.
template <typename WordType>
void testWordType(const char* label) {
    std::cout << "\n--- BinMat<" << label << "> (" << bincv::BinMat<WordType>::WordBits
              << " bits/word) ---\n";

    // Construction
    bincv::BinMat<WordType> mat(8, 4);
    BINCV_CHECK_EQ(mat.getWidth(), size_t(8));
    BINCV_CHECK_EQ(mat.getHeight(), size_t(4));
    BINCV_CHECK_EQ(mat.cols(), 8);
    BINCV_CHECK_EQ(mat.rows(), 4);
    BINCV_CHECK(!mat.empty());
    BINCV_CHECK_EQ(mat.countNonZero(), 0);

    // An owning matrix allocates through Storage, so it reports itself as owning.
    BINCV_CHECK(mat.ownsMemory());
    BINCV_CHECK(mat.data() != nullptr);

    // set / at
    mat.set(0, 0, true);
    mat.set(3, 7, true);
    BINCV_CHECK(mat.at(0, 0));
    BINCV_CHECK(mat.at(3, 7));
    BINCV_CHECK(!mat.at(1, 1));
    BINCV_CHECK_EQ(mat.countNonZero(), 2);

    // Clearing a bit
    mat.set(0, 0, false);
    BINCV_CHECK(!mat.at(0, 0));
    BINCV_CHECK_EQ(mat.countNonZero(), 1);

    // Row pointer access writes through to the same storage
    bincv::BinMat<WordType> ptrMat(8, 4);
    WordType* row1 = ptrMat.ptr(1);
    row1[0] |= static_cast<WordType>(0xF0);  // sets columns 4..7 of row 1
    BINCV_CHECK(ptrMat.at(1, 4));
    BINCV_CHECK(ptrMat.at(1, 7));
    BINCV_CHECK(!ptrMat.at(1, 3));

    // fill / countNonZero / sparsity. Width 70 is not a multiple of any supported
    // word size, so this also pins down the row-padding-bit handling.
    bincv::BinMat<WordType> odd(70, 5);
    odd.fill(true);
    BINCV_CHECK_EQ(odd.countNonZero(), 70 * 5);
    BINCV_CHECK(odd.sparsity() == 0.0f);

    // The trailing partial word is the ONLY padding that exists at the default
    // (tight) stride, and countNonZero()'s per-pixel loop never reads it -- so the
    // mask clearTrailingBits() applies has to be asserted directly. 70 pixels
    // leaves 6 valid bits in the last word at every supported word width.
    constexpr size_t bitsPerWord = bincv::BinMat<WordType>::WordBits;
    constexpr size_t lastWord = (70 - 1) / bitsPerWord;
    constexpr size_t validBits = 70 - lastWord * bitsPerWord;
    const WordType tailMask =
        static_cast<WordType>((static_cast<WordType>(1) << validBits) - 1);
    BINCV_CHECK_EQ(odd.getAlignedWidth(), lastWord + 1);
    for (int y = 0; y < odd.rows(); ++y) {
        BINCV_CHECK(odd.ptr(y)[lastWord] == tailMask);
    }
    // Same claim stated over the whole buffer: no bit is set that is not a pixel.
    BINCV_CHECK_EQ(countBitsAcrossStride(odd), 70 * 5);

    odd.fill(false);
    BINCV_CHECK_EQ(odd.countNonZero(), 0);
    BINCV_CHECK_EQ(countBitsAcrossStride(odd), 0);
    BINCV_CHECK(odd.sparsity() == 1.0f);

    // resize preserves the overlapping region
    bincv::BinMat<WordType> rs(16, 4);
    rs.set(1, 2, true);
    rs.set(3, 15, true);
    rs.resize(8, 4);                 // narrows: column 15 falls outside
    BINCV_CHECK_EQ(rs.getWidth(), size_t(8));
    BINCV_CHECK(rs.at(1, 2));
    BINCV_CHECK_EQ(rs.countNonZero(), 1);

    // resize to empty
    bincv::BinMat<WordType> toEmpty(8, 8);
    toEmpty.fill(true);
    toEmpty.resize(0, 0);
    BINCV_CHECK(toEmpty.empty());
    BINCV_CHECK_EQ(toEmpty.countNonZero(), 0);

    // pad shifts content by (top, left) and grows the matrix
    bincv::BinMat<WordType> pd(4, 2);
    pd.set(0, 0, true);
    pd.set(1, 3, true);
    pd.pad(1, 2, 2, 1);              // top, bottom, left, right
    BINCV_CHECK_EQ(pd.getWidth(), size_t(4 + 2 + 1));
    BINCV_CHECK_EQ(pd.getHeight(), size_t(2 + 1 + 2));
    BINCV_CHECK(pd.at(0 + 1, 0 + 2));
    BINCV_CHECK(pd.at(1 + 1, 3 + 2));
    BINCV_CHECK_EQ(pd.countNonZero(), 2);

    // pad with value = true fills the border with ones and must not leak
    // phantom bits into the row padding
    bincv::BinMat<WordType> pdOnes(4, 2);
    pdOnes.pad(1, 1, 1, 1, true);
    BINCV_CHECK_EQ(pdOnes.getWidth(), size_t(6));
    BINCV_CHECK_EQ(pdOnes.getHeight(), size_t(4));
    // border all ones, interior still zero => total = area - original area
    BINCV_CHECK_EQ(pdOnes.countNonZero(), 6 * 4 - 4 * 2);
    // ...and the ones stopped at the row's last pixel, not at the word boundary.
    BINCV_CHECK_EQ(countBitsAcrossStride(pdOnes), 6 * 4 - 4 * 2);

    // transposed / transpose
    bincv::BinMat<WordType> tr(5, 2);
    tr.set(0, 4, true);
    tr.set(1, 0, true);
    auto trd = tr.transposed();
    BINCV_CHECK_EQ(trd.getWidth(), size_t(2));
    BINCV_CHECK_EQ(trd.getHeight(), size_t(5));
    BINCV_CHECK(trd.at(4, 0));
    BINCV_CHECK(trd.at(0, 1));
    BINCV_CHECK_EQ(trd.countNonZero(), 2);
    BINCV_CHECK_EQ(tr.getWidth(), size_t(5));   // original untouched

    // Round trip returns the original
    auto roundTrip = trd.transposed();
    BINCV_CHECK_EQ(roundTrip.getWidth(), tr.getWidth());
    BINCV_CHECK_EQ(roundTrip.getHeight(), tr.getHeight());
    BINCV_CHECK(roundTrip.at(0, 4));
    BINCV_CHECK(roundTrip.at(1, 0));

    // In-place transpose on a non-square matrix
    bincv::BinMat<WordType> inPlace(2, 5);
    inPlace.set(0, 0, true);
    inPlace.set(4, 1, true);
    inPlace.transpose();
    BINCV_CHECK_EQ(inPlace.getWidth(), size_t(5));
    BINCV_CHECK_EQ(inPlace.getHeight(), size_t(2));
    BINCV_CHECK(inPlace.at(0, 0));
    BINCV_CHECK(inPlace.at(1, 4));

    // forEachNonZero visits exactly the set pixels
    bincv::BinMat<WordType> fe(10, 3);
    fe.set(0, 1, true);
    fe.set(2, 9, true);
    std::vector<std::pair<int, int>> visited;
    fe.forEachNonZero([&](int r, int c) { visited.emplace_back(r, c); });
    BINCV_CHECK_EQ(visited.size(), size_t(2));
    if (visited.size() == 2) {
        BINCV_CHECK(visited[0] == std::make_pair(0, 1));
        BINCV_CHECK(visited[1] == std::make_pair(2, 9));
    }

    // Empty matrix behaviour
    bincv::BinMat<WordType> empty;
    BINCV_CHECK(empty.empty());
    BINCV_CHECK_EQ(empty.countNonZero(), 0);
    BINCV_CHECK(empty.transposed().empty());
    BINCV_CHECK(!empty.ownsMemory());          // nothing allocated, nothing owned
    BINCV_CHECK(empty.data() == nullptr);
    empty.printMatrix();  // must be a no-op, not a crash
    BINCV_CHECK_THROWS(empty.sparsity(), std::runtime_error);
    BINCV_CHECK_THROWS(empty.forEachNonZero([](int, int) {}), std::runtime_error);

    // A matrix with one zero dimension still has a shape to transpose: 40x0 goes
    // to 0x40 rather than collapsing to 0x0, and the row alignment survives with
    // it. Zero pixels means nothing is allocated either way.
    bincv::BinMat<WordType> flat(40, 0, 32);
    bincv::BinMat<WordType> flatT = flat.transposed();
    BINCV_CHECK(flatT.empty());
    BINCV_CHECK_EQ(flatT.getWidth(), size_t(0));
    BINCV_CHECK_EQ(flatT.getHeight(), size_t(40));
    BINCV_CHECK_EQ(flatT.getRowAlignment(), size_t(32));
    BINCV_CHECK(!flatT.ownsMemory());
    BINCV_CHECK(flatT.data() == nullptr);

    // In-place, on the other axis.
    bincv::BinMat<WordType> thin(0, 24);
    thin.transpose();
    BINCV_CHECK_EQ(thin.getWidth(), size_t(24));
    BINCV_CHECK_EQ(thin.getHeight(), size_t(0));
    BINCV_CHECK(thin.empty());

    // Argument validation. These are setup-time checks, so they still report
    // through BINCV_THROW (T1.4). The at()/set() bounds cases that used to sit
    // here cannot: those accessors are debug-checked and unchecked in release
    // now, so an out-of-range index aborts instead of throwing and no
    // in-process check can observe it. They moved to tests/test_assert_abort.cpp
    // (cases at-row, at-col, at-negative, set-row, set-col), which forces the
    // checked configuration and runs them as death tests in every build.
    //
    // Note also that in a build without exceptions BINCV_CHECK_THROWS reports a
    // SKIP rather than a pass; the same sites are covered there by
    // tests/test_error_abort.cpp.
    BINCV_CHECK_THROWS(bincv::BinMat<WordType>(-1, 10), std::invalid_argument);
    BINCV_CHECK_THROWS(bincv::BinMat<WordType>(10, 10, 0), std::invalid_argument);
    BINCV_CHECK_THROWS(bincv::BinMat<WordType>(10, 10, 3), std::invalid_argument);
    BINCV_CHECK_THROWS(mat.resize(-1, 5), std::invalid_argument);
    BINCV_CHECK_THROWS(mat.pad(-1, 0, 0, 0), std::invalid_argument);
}

// Copy is deep and move empties the source (D-8), for every word width.
template <typename WordType>
void testValueSemantics(const char* label) {
    std::cout << "\n--- Value semantics, BinMat<" << label << "> ---\n";

    bincv::BinMat<WordType> src(70, 3);
    src.set(1, 69, true);

    // Copy construction duplicates the buffer rather than sharing it.
    bincv::BinMat<WordType> copied(src);
    BINCV_CHECK(copied.ownsMemory());
    BINCV_CHECK(copied.data() != src.data());
    BINCV_CHECK(copied.at(1, 69));
    BINCV_CHECK_EQ(copied.getWidth(), src.getWidth());
    BINCV_CHECK_EQ(copied.getHeight(), src.getHeight());
    BINCV_CHECK_EQ(copied.getAlignedWidth(), src.getAlignedWidth());
    BINCV_CHECK_EQ(copied.getRowAlignment(), src.getRowAlignment());

    // Deep, so the two are independent in both directions.
    src.set(1, 69, false);
    BINCV_CHECK(copied.at(1, 69));
    BINCV_CHECK_EQ(src.countNonZero(), 0);
    copied.set(0, 0, true);
    BINCV_CHECK(!src.at(0, 0));

    // Opt-in alignment is part of the value and survives the copy.
    bincv::BinMat<WordType> alignedSrc(70, 3, 32);
    bincv::BinMat<WordType> alignedCopy(alignedSrc);
    BINCV_CHECK_EQ(alignedCopy.getRowAlignment(), size_t(32));
    BINCV_CHECK_EQ(alignedCopy.getAlignedWidth(), alignedSrc.getAlignedWidth());

    // Copy assignment is deep too, and replaces whatever the target held.
    bincv::BinMat<WordType> assigned(2, 2);
    assigned = copied;
    BINCV_CHECK(assigned.ownsMemory());
    BINCV_CHECK(assigned.data() != copied.data());
    BINCV_CHECK_EQ(assigned.getWidth(), size_t(70));
    BINCV_CHECK(assigned.at(1, 69));
    BINCV_CHECK_EQ(assigned.countNonZero(), copied.countNonZero());

    // Self-assignment must not free the buffer it is about to read. Routed
    // through a reference so this is a real runtime self-assignment rather than
    // something the compiler diagnoses and folds away.
    bincv::BinMat<WordType>& selfRef = assigned;
    assigned = selfRef;
    BINCV_CHECK_EQ(assigned.getWidth(), size_t(70));
    BINCV_CHECK(assigned.at(1, 69));
    BINCV_CHECK_EQ(assigned.countNonZero(), copied.countNonZero());

    // Move hands the buffer over and leaves the source a valid EMPTY matrix --
    // dimensions included, so empty() cannot report false while data() is null.
    bincv::BinMat<WordType> movedTo(std::move(assigned));
    BINCV_CHECK(movedTo.at(1, 69));
    BINCV_CHECK(movedTo.ownsMemory());
    BINCV_CHECK(assigned.empty());
    BINCV_CHECK(assigned.data() == nullptr);
    BINCV_CHECK_EQ(assigned.sizeInWords(), size_t(0));
    BINCV_CHECK_EQ(assigned.getAlignedWidth(), size_t(0));
    BINCV_CHECK_EQ(assigned.countNonZero(), 0);

    bincv::BinMat<WordType> moveAssigned;
    moveAssigned = std::move(movedTo);
    BINCV_CHECK(moveAssigned.at(1, 69));
    BINCV_CHECK(movedTo.empty());
    BINCV_CHECK(movedTo.data() == nullptr);

    // Move-assigning from a matrix that WRAPS this object's own storage is the one
    // transfer that cannot be honoured -- the block would have to be freed and then
    // re-adopted through an interior pointer -- so the target is left UNCHANGED.
    // The failure mode being pinned here is a half-applied move: taking the
    // source's dimensions over a buffer that never moved makes every row pointer
    // address the wrong row and breaks sizeInWords() == height * alignedWidth.
    bincv::BinMat<WordType> owner(64, 4);
    for (int y = 0; y < 4; ++y) owner.set(y, y, true);
    const WordType* const ownerData = owner.data();
    bincv::BinMat<WordType> subRows(owner.data() + 2 * owner.getAlignedWidth(), 64, 2,
                                    owner.getAlignedWidth());
    BINCV_CHECK(subRows.at(0, 2));            // subRows row 0 IS owner row 2
    owner = std::move(subRows);
    BINCV_CHECK_EQ(owner.getWidth(), size_t(64));
    BINCV_CHECK_EQ(owner.getHeight(), size_t(4));
    BINCV_CHECK_EQ(owner.sizeInWords(), owner.getHeight() * owner.getAlignedWidth());
    BINCV_CHECK(owner.data() == ownerData);
    BINCV_CHECK(owner.ownsMemory());
    BINCV_CHECK(owner.at(0, 0));
    BINCV_CHECK(owner.at(3, 3));
    BINCV_CHECK_EQ(owner.countNonZero(), 4);
    BINCV_CHECK(subRows.empty());             // the source is emptied either way
    BINCV_CHECK(subRows.data() == nullptr);

    // The copy-assignment form of the same aliasing case IS honoured, because the
    // deep copy is taken while this object still holds its buffer: owner2 ends up
    // owning a real copy of the two rows the wrapper described. (subRows2 wraps
    // freed memory afterwards and must not be touched again.)
    bincv::BinMat<WordType> owner2(64, 4);
    for (int y = 0; y < 4; ++y) owner2.set(y, y, true);
    bincv::BinMat<WordType> subRows2(owner2.data() + 2 * owner2.getAlignedWidth(), 64, 2,
                                     owner2.getAlignedWidth());
    owner2 = subRows2;
    BINCV_CHECK(owner2.ownsMemory());
    BINCV_CHECK_EQ(owner2.getWidth(), size_t(64));
    BINCV_CHECK_EQ(owner2.getHeight(), size_t(2));
    BINCV_CHECK_EQ(owner2.sizeInWords(), owner2.getHeight() * owner2.getAlignedWidth());
    BINCV_CHECK(owner2.at(0, 2));             // was owner2 row 2
    BINCV_CHECK(owner2.at(1, 3));             // was owner2 row 3
    BINCV_CHECK(!owner2.at(0, 0));
    BINCV_CHECK_EQ(owner2.countNonZero(), 2);

    // Moving from a wrapper of UNRELATED memory is an ordinary transfer, though:
    // the target adopts the wrap and stays non-owning. Refusing that too would
    // make the aliasing rule swallow every move out of a Tier 2 matrix.
    constexpr size_t extStride = 64 / bincv::BinMat<WordType>::WordBits;
    WordType external[extStride * 2] = {};
    bincv::BinMat<WordType> wrapExternal(external, 64, 2, extStride);
    wrapExternal.set(1, 63, true);
    bincv::BinMat<WordType> target(8, 8);
    target = std::move(wrapExternal);
    BINCV_CHECK(!target.ownsMemory());
    BINCV_CHECK(target.data() == external);
    BINCV_CHECK_EQ(target.getWidth(), size_t(64));
    BINCV_CHECK_EQ(target.getHeight(), size_t(2));
    BINCV_CHECK_EQ(target.sizeInWords(), extStride * 2);
    BINCV_CHECK(target.at(1, 63));
    BINCV_CHECK(wrapExternal.empty());
}

// The non-owning constructor: wrap caller memory, never free it, and deep-copy
// out of it like any other BinMat.
template <typename WordType>
void testExternalBuffer(const char* label) {
    std::cout << "\n--- External buffer, BinMat<" << label << "> ---\n";

    constexpr size_t bits = bincv::BinMat<WordType>::WordBits;
    constexpr size_t width = 70;
    constexpr size_t height = 4;
    constexpr size_t rowWords = (width + bits - 1) / bits;
    // Deliberately looser than the row needs: a wrapped stride is whatever the
    // caller's buffer already has, not something BinMat gets to choose.
    constexpr size_t stride = rowWords + 1;

    WordType buffer[stride * height] = {};

    bincv::BinMat<WordType> wrapped(buffer, static_cast<int>(width),
                                    static_cast<int>(height), stride);
    BINCV_CHECK(!wrapped.ownsMemory());
    BINCV_CHECK(wrapped.data() == buffer);
    BINCV_CHECK_EQ(wrapped.getWidth(), width);
    BINCV_CHECK_EQ(wrapped.getHeight(), height);
    BINCV_CHECK_EQ(wrapped.getAlignedWidth(), stride);
    BINCV_CHECK_EQ(wrapped.sizeInWords(), stride * height);

    // Writes through the container land in the caller's buffer...
    wrapped.set(1, 0, true);
    BINCV_CHECK(buffer[stride] == static_cast<WordType>(1));
    // ...and writes the caller makes are visible through the container.
    buffer[2 * stride] |= static_cast<WordType>(1);
    BINCV_CHECK(wrapped.at(2, 0));
    BINCV_CHECK_EQ(wrapped.countNonZero(), 2);

    // A view of a wrapped matrix addresses the caller's buffer directly, so a
    // kernel cannot tell wrapped from owning memory (D-5).
    BINCV_CHECK(wrapped.view().ptr == buffer);
    BINCV_CHECK_EQ(wrapped.view().stride, stride);

    // Copying a NON-OWNING BinMat still deep-copies and still yields an owning
    // matrix: the rule is a property of BinMat, not of how the source was built.
    bincv::BinMat<WordType> copied(wrapped);
    BINCV_CHECK(copied.ownsMemory());
    BINCV_CHECK(copied.data() != wrapped.data());
    BINCV_CHECK(copied.at(1, 0));
    BINCV_CHECK(copied.at(2, 0));
    BINCV_CHECK_EQ(copied.countNonZero(), wrapped.countNonZero());

    // Genuinely detached: neither side sees the other's writes afterwards.
    copied.set(3, 69, true);
    BINCV_CHECK(!wrapped.at(3, 69));
    wrapped.set(0, 1, true);
    BINCV_CHECK(!copied.at(0, 1));

    // Copy assignment from a non-owning source is deep for the same reason.
    bincv::BinMat<WordType> assigned(4, 4);
    assigned = wrapped;
    BINCV_CHECK(assigned.ownsMemory());
    BINCV_CHECK(assigned.data() != wrapped.data());
    BINCV_CHECK_EQ(assigned.getWidth(), width);
    BINCV_CHECK_EQ(assigned.countNonZero(), wrapped.countNonZero());

    // Reallocating operations detach from the caller's buffer rather than
    // writing outside it. The buffer is left exactly as it was.
    const WordType before = buffer[0];
    bincv::BinMat<WordType> resized(buffer, static_cast<int>(width),
                                    static_cast<int>(height), stride);
    resized.resize(static_cast<int>(width), static_cast<int>(height) + 1);
    BINCV_CHECK(resized.ownsMemory());
    BINCV_CHECK(resized.data() != buffer);
    BINCV_CHECK(buffer[0] == before);

    // transpose() reallocates as resize() and pad() do, so it detaches from the
    // caller's buffer and leaves it untouched -- including for a square matrix,
    // where an in-place bit transpose would be the natural reading.
    constexpr size_t sqSide = 8;
    constexpr size_t sqStride = (sqSide + bits - 1) / bits;
    WordType square[sqStride * sqSide] = {};
    bincv::BinMat<WordType> sqWrap(square, static_cast<int>(sqSide),
                                   static_cast<int>(sqSide), sqStride);
    sqWrap.set(0, 5, true);
    const WordType squareBefore = square[0];
    sqWrap.transpose();
    BINCV_CHECK(sqWrap.ownsMemory());
    BINCV_CHECK(sqWrap.data() != square);
    BINCV_CHECK(square[0] == squareBefore);   // the caller's buffer is left alone
    BINCV_CHECK(sqWrap.at(5, 0));
    BINCV_CHECK_EQ(sqWrap.countNonZero(), 1);

    // A wrapped buffer may carry set bits past `width`: the wrap constructor hands
    // that invariant to the caller. Copying takes ownership, and an owning BinMat
    // owes the invariant to every word-wise reduction, so the copy re-establishes
    // it. Without this the copy reads correctly pixel by pixel while carrying
    // phantom bits that only a whole-word count can see.
    WordType dirty[stride * height];
    std::memset(dirty, 0xFF, sizeof dirty);
    bincv::BinMat<WordType> wrappedDirty(dirty, static_cast<int>(width),
                                         static_cast<int>(height), stride);
    bincv::BinMat<WordType> cleaned(wrappedDirty);
    BINCV_CHECK(cleaned.ownsMemory());
    BINCV_CHECK_EQ(cleaned.countNonZero(), static_cast<int>(width * height));
    BINCV_CHECK_EQ(countBitsAcrossStride(cleaned), static_cast<int>(width * height));
    // The wrapped source keeps every bit the caller put there, phantom ones too.
    BINCV_CHECK_EQ(countBitsAcrossStride(wrappedDirty),
                   static_cast<int>(stride * height * bits));

    // Wrapping zero pixels is legal and stays empty rather than pretending to
    // address the pointer it was given.
    bincv::BinMat<WordType> emptyWrap(nullptr, 0, 0, 0);
    BINCV_CHECK(emptyWrap.empty());
    BINCV_CHECK(!emptyWrap.ownsMemory());
    BINCV_CHECK(emptyWrap.data() == nullptr);

    // Validation: a stride below ceil(width / WordBits) would make consecutive
    // rows overlap, and a null pointer cannot back a non-empty matrix.
    BINCV_CHECK_THROWS(bincv::BinMat<WordType>(buffer, static_cast<int>(width), 2, rowWords - 1),
                       std::invalid_argument);
    BINCV_CHECK_THROWS(bincv::BinMat<WordType>(buffer, -1, 2, stride), std::invalid_argument);
    BINCV_CHECK_THROWS(bincv::BinMat<WordType>(buffer, 2, -1, stride), std::invalid_argument);
    BINCV_CHECK_THROWS(bincv::BinMat<WordType>(nullptr, 8, 2, stride), std::invalid_argument);
}

// view() / constView() describe exactly the pixels the container holds.
template <typename WordType>
void testViews(const char* label) {
    std::cout << "\n--- Views over BinMat<" << label << "> ---\n";

    // Width 70 is not a multiple of any supported word size, so the view has to
    // agree with the container about the trailing partial word too.
    bincv::BinMat<WordType> m(70, 3);
    m.set(0, 0, true);
    m.set(1, 33, true);
    m.set(2, 69, true);

    // Not named `cv`: with OpenCV interop enabled that identifier is a namespace.
    bincv::BinMatConstView<WordType> constV = m.constView();
    BINCV_CHECK(constV.ptr == m.data());
    BINCV_CHECK_EQ(constV.width, m.getWidth());
    BINCV_CHECK_EQ(constV.height, m.getHeight());
    BINCV_CHECK_EQ(constV.stride, m.getAlignedWidth());
    BINCV_CHECK(!constV.empty());

    // Round-trip: every pixel read through the view's own row/word arithmetic
    // matches BinMat::at(), and the set pixels are where they were put.
    bool matches = true;
    size_t setPixels = 0;
    for (size_t y = 0; y < m.getHeight(); ++y) {
        for (size_t x = 0; x < m.getWidth(); ++x) {
            const bool viaView = viewPixel<WordType>(constV, y, x);
            if (viaView != m.at(static_cast<int>(y), static_cast<int>(x))) matches = false;
            if (viaView) ++setPixels;
        }
    }
    BINCV_CHECK(matches);
    BINCV_CHECK_EQ(setPixels, size_t(3));

    // A mutable view writes through to the container...
    bincv::BinMatView<WordType> v = m.view();
    BINCV_CHECK(v.ptr == m.data());
    BINCV_CHECK_EQ(v.stride, m.getAlignedWidth());
    v.row(1)[0] |= static_cast<WordType>(1);
    BINCV_CHECK(m.at(1, 0));
    BINCV_CHECK_EQ(m.countNonZero(), 4);

    // ...and a change made through the container is visible through a fresh view.
    m.set(1, 0, false);
    BINCV_CHECK(!viewPixel<WordType>(m.constView(), 1, 0));

    // Implicit BinMatView -> BinMatConstView conversion (D-9).
    bincv::BinMatConstView<WordType> converted = m.view();
    BINCV_CHECK(converted.ptr == m.data());
    BINCV_CHECK_EQ(converted.stride, m.getAlignedWidth());

    // constView() is const-callable, which is how a const BinMat is handed to a
    // read-only kernel.
    const bincv::BinMat<WordType>& constRef = m;
    BINCV_CHECK(constRef.constView().ptr == m.data());
    BINCV_CHECK_EQ(constRef.constView().height, size_t(3));

    // An empty matrix yields empty views rather than a view over nothing.
    bincv::BinMat<WordType> none;
    BINCV_CHECK(none.view().empty());
    BINCV_CHECK(none.constView().empty());
    BINCV_CHECK(none.view().ptr == nullptr);
}

// D-4: the default row stride is ceil(width / WordBits) words with no padding
// beyond it. The 640x480 figure is the number the decision was argued on, so it
// is pinned here rather than left to be re-derived.
void testDefaultFootprint() {
    std::cout << "\n--- Allocation footprint (D-4) ---\n";

    // 640 px / 32 bits = 20 words per row; 20 words * 4 B * 480 rows = 38400 B.
    bincv::BinMat<uint32_t> frame(640, 480);
    BINCV_CHECK_EQ(frame.getRowAlignment(), sizeof(uint32_t));
    BINCV_CHECK_EQ(frame.getAlignedWidth(), size_t(20));
    BINCV_CHECK_EQ(frame.sizeInWords(), size_t(9600));
    BINCV_CHECK_EQ(frame.sizeInWords() * sizeof(uint32_t), size_t(38400));
    BINCV_CHECK(frame.ownsMemory());

    // 38400 B is 640 * 480 / 8 exactly: one bit per pixel and nothing else.
    BINCV_CHECK_EQ(frame.sizeInWords() * sizeof(uint32_t), size_t(640 * 480 / 8));

    // The previous 32-byte default cost 46080 B for the same frame (+20%).
    // Larger alignment is still available -- it is opt-in now, not the default.
    bincv::BinMat<uint32_t> aligned32(640, 480, 32);
    BINCV_CHECK_EQ(aligned32.getRowAlignment(), size_t(32));
    BINCV_CHECK_EQ(aligned32.getAlignedWidth(), size_t(24));
    BINCV_CHECK_EQ(aligned32.sizeInWords() * sizeof(uint32_t), size_t(46080));

    // Where D-4 actually bites: pyramid level 3, which LK touches every frame.
    // 94 px -> 3 words (720 B) at word granularity, 8 words (1920 B) at 32 B.
    bincv::BinMat<uint32_t> level3(94, 60);
    BINCV_CHECK_EQ(level3.getAlignedWidth(), size_t(3));
    BINCV_CHECK_EQ(level3.sizeInWords() * sizeof(uint32_t), size_t(720));
    bincv::BinMat<uint32_t> level3Aligned(94, 60, 32);
    BINCV_CHECK_EQ(level3Aligned.sizeInWords() * sizeof(uint32_t), size_t(1920));

    // The word type does not change the footprint when the width divides evenly:
    // 640x480 is 38400 B at every supported width.
    bincv::BinMat<uint8_t> frame8(640, 480);
    bincv::BinMat<uint16_t> frame16(640, 480);
    bincv::BinMat<uint64_t> frame64(640, 480);
    BINCV_CHECK_EQ(frame8.sizeInWords() * sizeof(uint8_t), size_t(38400));
    BINCV_CHECK_EQ(frame16.sizeInWords() * sizeof(uint16_t), size_t(38400));
    BINCV_CHECK_EQ(frame64.sizeInWords() * sizeof(uint64_t), size_t(38400));
    BINCV_CHECK_EQ(frame8.getAlignedWidth(), size_t(80));
    BINCV_CHECK_EQ(frame16.getAlignedWidth(), size_t(40));
    BINCV_CHECK_EQ(frame64.getAlignedWidth(), size_t(10));

    // A width that does not fill its last word still pays only that one word.
    bincv::BinMat<uint32_t> ragged(70, 2);
    BINCV_CHECK_EQ(ragged.getAlignedWidth(), size_t(3));
    BINCV_CHECK_EQ(ragged.sizeInWords(), size_t(6));
}

// Wrapping a caller buffer must not touch the allocator, in either direction.
void testWrapAllocatesNothing() {
    std::cout << "\n--- Wrapping allocates nothing, frees nothing ---\n";

    constexpr size_t stride = 3;    // 70 px at 32 bits/word
    constexpr size_t height = 4;
    uint32_t buffer[stride * height] = {};

    // No I/O and no BINCV_CHECK inside a measured region: both allocate (the
    // check macro builds its diagnostic string eagerly), which would make the
    // counters describe the harness rather than the container. Snapshot first,
    // assert afterwards.
    const size_t newsBefore = g_newCount;
    const size_t deletesBefore = g_deleteCount;
    {
        bincv::BinMat<uint32_t> wrapped(buffer, 70, static_cast<int>(height), stride);
        wrapped.set(0, 0, true);
        wrapped.set(3, 69, true);
        escape(wrapped.data());
        escape(wrapped.view().ptr);
        escape(wrapped.constView().ptr);
    }   // destructor runs here; it must not free the caller's array
    const size_t wrapNews = g_newCount - newsBefore;
    const size_t wrapDeletes = g_deleteCount - deletesBefore;

    // For contrast: an owning matrix of the same shape does allocate exactly one
    // block and release it. That is the "BinMat allocates through Storage" half.
    const size_t ownNewsBefore = g_newCount;
    const size_t ownDeletesBefore = g_deleteCount;
    {
        bincv::BinMat<uint32_t> owned(70, static_cast<int>(height));
        owned.set(0, 0, true);
        escape(owned.data());
    }
    const size_t ownNews = g_newCount - ownNewsBefore;
    const size_t ownDeletes = g_deleteCount - ownDeletesBefore;

    BINCV_CHECK_EQ(wrapNews, size_t(0));
    BINCV_CHECK_EQ(wrapDeletes, size_t(0));
    BINCV_CHECK_EQ(ownNews, size_t(1));
    BINCV_CHECK_EQ(ownDeletes, size_t(1));

    // The buffer still holds what the destroyed wrapper wrote into it, which it
    // could not if the wrapper had freed it.
    BINCV_CHECK(buffer[0] == 1u);
    BINCV_CHECK(buffer[3 * stride + 2] == (1u << (69 % 32)));
}

// Row strides must honour the requested byte alignment.
void testRowAlignment() {
    std::cout << "\n--- Row alignment ---\n";

    // The default is word granularity: 70 pixels at 32 bits/word is 3 words, and
    // that is the whole stride (D-4).
    bincv::BinMat<uint32_t> tight(70, 2);
    BINCV_CHECK_EQ(tight.getRowAlignment(), sizeof(uint32_t));
    BINCV_CHECK_EQ(tight.getAlignedWidth(), size_t(3));
    BINCV_CHECK_EQ(tight.sizeInWords(), size_t(6));

    // 70 pixels at 32 bits/word needs 3 words = 12 bytes; aligned to 32 bytes
    // that becomes 32 bytes = 8 words.
    bincv::BinMat<uint32_t> a(70, 2, 32);
    BINCV_CHECK_EQ(a.getAlignedWidth(), size_t(8));
    BINCV_CHECK_EQ(a.sizeInWords(), size_t(16));

    // With 4-byte alignment the stride is just the 3 words actually needed --
    // which for uint32_t is the same thing the default asks for.
    bincv::BinMat<uint32_t> b(70, 2, 4);
    BINCV_CHECK_EQ(b.getAlignedWidth(), size_t(3));
    BINCV_CHECK_EQ(b.getRowAlignment(), size_t(4));

    // Alignment does not change observable contents
    a.set(1, 69, true);
    b.set(1, 69, true);
    tight.set(1, 69, true);
    BINCV_CHECK(a.at(1, 69));
    BINCV_CHECK(b.at(1, 69));
    BINCV_CHECK(tight.at(1, 69));
    BINCV_CHECK_EQ(a.countNonZero(), b.countNonZero());
    BINCV_CHECK_EQ(a.countNonZero(), tight.countNonZero());

    // Padding bits stay zero at the wider alignment: fill(true) writes whole
    // words across the stride, and countNonZero must not see them.
    a.fill(true);
    BINCV_CHECK_EQ(a.countNonZero(), 70 * 2);
    for (int y = 0; y < a.rows(); ++y) {
        BINCV_CHECK(a.ptr(y)[3] == 0u);   // first pure-padding word of the row
        BINCV_CHECK(a.ptr(y)[7] == 0u);   // last one
    }

    // At the DEFAULT stride there are no pure-padding words at all -- the only
    // padding left is the tail of the last word, which is exactly the part every
    // whole-word reduction reads. 70 pixels at 32 bits/word leaves 6 valid bits.
    tight.fill(true);
    BINCV_CHECK_EQ(tight.countNonZero(), 70 * 2);
    for (int y = 0; y < tight.rows(); ++y) {
        BINCV_CHECK(tight.ptr(y)[2] == 0x3Fu);
    }
    BINCV_CHECK_EQ(countBitsAcrossStride(tight), 70 * 2);

    // Alignment carries through the operations that reallocate.
    a.resize(40, 2);
    BINCV_CHECK_EQ(a.getRowAlignment(), size_t(32));
    BINCV_CHECK_EQ(a.getAlignedWidth(), size_t(8));
    b.pad(0, 0, 0, 2);
    BINCV_CHECK_EQ(b.getRowAlignment(), size_t(4));
    BINCV_CHECK_EQ(b.getAlignedWidth(), size_t(3));
}

// Word-type-independent guarantees.
void testTypeAliases() {
    std::cout << "\n--- Type aliases ---\n";
    static_assert(bincv::BinMat<>::WordBits == 32, "default word type should be 32-bit");
    static_assert(bincv::BinMat8::WordBits == 8, "BinMat8 should be 8-bit");
    static_assert(bincv::BinMat16::WordBits == 16, "BinMat16 should be 16-bit");
    static_assert(bincv::BinMat32::WordBits == 32, "BinMat32 should be 32-bit");
    static_assert(bincv::BinMat64::WordBits == 64, "BinMat64 should be 64-bit");

    // The default alignment is word granularity for every word type (D-4).
    static_assert(bincv::BinMat8::DefaultRowAlignment == 1, "");
    static_assert(bincv::BinMat16::DefaultRowAlignment == 2, "");
    static_assert(bincv::BinMat32::DefaultRowAlignment == 4, "");
    static_assert(bincv::BinMat64::DefaultRowAlignment == 8, "");

    bincv::BinMat32 m(8, 8);
    m.fill(true);
    BINCV_CHECK_EQ(m.countNonZero(), 64);

    // bincv::Size from core/types.hpp
    bincv::Size s = m.getSize();
    BINCV_CHECK_EQ(s.width, 8);
    BINCV_CHECK_EQ(s.height, 8);
    BINCV_CHECK_EQ(s.area(), 64);
    BINCV_CHECK(!s.empty());
    BINCV_CHECK(s == bincv::Size(8, 8));
}

} // namespace

int main() {
    std::cout << "=== BinMat core tests (no OpenCV) ===\n";

    testWordType<uint8_t>("uint8_t");
    testWordType<uint16_t>("uint16_t");
    testWordType<uint32_t>("uint32_t");
    testWordType<uint64_t>("uint64_t");

    testValueSemantics<uint8_t>("uint8_t");
    testValueSemantics<uint16_t>("uint16_t");
    testValueSemantics<uint32_t>("uint32_t");
    testValueSemantics<uint64_t>("uint64_t");

    testExternalBuffer<uint8_t>("uint8_t");
    testExternalBuffer<uint16_t>("uint16_t");
    testExternalBuffer<uint32_t>("uint32_t");
    testExternalBuffer<uint64_t>("uint64_t");

    testViews<uint8_t>("uint8_t");
    testViews<uint16_t>("uint16_t");
    testViews<uint32_t>("uint32_t");
    testViews<uint64_t>("uint64_t");

    testDefaultFootprint();
    testWrapAllocatesNothing();
    testRowAlignment();
    testTypeAliases();

    return bincv::test::summarize("BinMat core tests");
}
