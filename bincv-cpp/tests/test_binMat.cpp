// Core BinMat tests. Deliberately free of OpenCV so this suite also builds and
// runs in embedded (core-only) configurations -- see tests/test_opencv_interop.cpp
// for the interop coverage.

#include <iostream>
#include <vector>
#include <utility>

#include "bincv-cpp/binMat.hpp"
#include "test_util.hpp"

namespace {

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
    odd.fill(false);
    BINCV_CHECK_EQ(odd.countNonZero(), 0);
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
    empty.printMatrix();  // must be a no-op, not a crash
    BINCV_CHECK_THROWS(empty.sparsity(), std::runtime_error);
    BINCV_CHECK_THROWS(empty.forEachNonZero([](int, int) {}), std::runtime_error);

    // Argument validation
    BINCV_CHECK_THROWS(bincv::BinMat<WordType>(-1, 10), std::invalid_argument);
    BINCV_CHECK_THROWS(bincv::BinMat<WordType>(10, 10, 0), std::invalid_argument);
    BINCV_CHECK_THROWS(bincv::BinMat<WordType>(10, 10, 3), std::invalid_argument);
    BINCV_CHECK_THROWS(mat.at(999, 0), std::out_of_range);
    BINCV_CHECK_THROWS(mat.at(0, 999), std::out_of_range);
    BINCV_CHECK_THROWS(mat.set(-1, 5, true), std::out_of_range);
    BINCV_CHECK_THROWS(mat.resize(-1, 5), std::invalid_argument);
    BINCV_CHECK_THROWS(mat.pad(-1, 0, 0, 0), std::invalid_argument);
}

// Row strides must honour the requested byte alignment.
void testRowAlignment() {
    std::cout << "\n--- Row alignment ---\n";

    // 70 pixels at 32 bits/word needs 3 words = 12 bytes; aligned to 32 bytes
    // that becomes 32 bytes = 8 words.
    bincv::BinMat<uint32_t> a(70, 2, 32);
    BINCV_CHECK_EQ(a.getAlignedWidth(), size_t(8));
    BINCV_CHECK_EQ(a.sizeInWords(), size_t(16));

    // With 4-byte alignment the stride is just the 3 words actually needed.
    bincv::BinMat<uint32_t> b(70, 2, 4);
    BINCV_CHECK_EQ(b.getAlignedWidth(), size_t(3));
    BINCV_CHECK_EQ(b.getRowAlignment(), size_t(4));

    // Alignment does not change observable contents
    a.set(1, 69, true);
    b.set(1, 69, true);
    BINCV_CHECK(a.at(1, 69));
    BINCV_CHECK(b.at(1, 69));
    BINCV_CHECK_EQ(a.countNonZero(), b.countNonZero());
}

// Word-type-independent guarantees.
void testTypeAliases() {
    std::cout << "\n--- Type aliases ---\n";
    static_assert(bincv::BinMat<>::WordBits == 32, "default word type should be 32-bit");
    static_assert(bincv::BinMat8::WordBits == 8, "BinMat8 should be 8-bit");
    static_assert(bincv::BinMat16::WordBits == 16, "BinMat16 should be 16-bit");
    static_assert(bincv::BinMat32::WordBits == 32, "BinMat32 should be 32-bit");
    static_assert(bincv::BinMat64::WordBits == 64, "BinMat64 should be 64-bit");

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
    testRowAlignment();
    testTypeAliases();

    return bincv::test::summarize("BinMat core tests");
}
