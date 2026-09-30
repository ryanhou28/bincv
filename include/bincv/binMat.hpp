#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
// No <iostream> here, and none reachable from here: including it registers the
// iostream static initializer (an ios_base::Init construction plus an
// __cxa_atexit entry) in EVERY translation unit that includes this container,
// whether or not it ever prints. That is the cost core/error.hpp avoids by
// reporting through <cstdio>, and a header-only container has no business
// reimposing it on an embedded target where code size is often the binding
// constraint. The printing helpers use std::fprintf; only
// operator<<, which cannot be written without a stream type, pulls <ostream> --
// and <ostream> alone carries no static initializer.
// <stdexcept> likewise: core/error.hpp includes it when, and only when, the
// throw path is the one being compiled.
#include "core/error.hpp"
#include "core/storage.hpp"
#include "core/types.hpp"
#include "core/view.hpp"

// OpenCV integration is optional and provided behind a compile-time switch.
// The core library never requires OpenCV; see CMakeLists.txt.
#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#endif

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief A binary matrix storing one bit per pixel, packed into words.
/// **API TIER 3 as a type** -- OpenCV has no packed 1-bit image; `fromCVMat` and
/// `toCVMat` are the bridge to `cv::Mat`.
/// @tparam WordType The unsigned integral type used to pack pixels.
/// Supported: uint8_t, uint16_t, uint32_t (default), uint64_t.
/// @note Spelled QuantMat<1, WordType> here and BinMat<WordType> everywhere else:
/// this IS the N=1 specialization of the N-bit container, and BinMat is an
/// alias for it (core/types.hpp). The two names are one
/// type. The specialization exists so the 1-bit case keeps the hand-written
/// single-plane paths below -- at, set, fill, countNonZero and the
/// rest address the one plane directly, with no loop over planes and no
/// delegation -- while still being what a QuantMat<N> parameter binds to.
/// @note Only the N=1 case is written out by hand. The general N>1 container
/// (quantMat.hpp) is a shape over this one: N planes stacked in a single
/// allocation, each plane a view of the same layout.
/// @note Parameterizing on the storage word *type* (rather than a bit count)
/// follows boost::dynamic_bitset<Block> and cv::Mat_<T>. The bit width is
/// derived as WordBits, so the type never has to be recovered from a number.
/// @note Backed by Storage (core/storage.hpp), which is what lets the same
/// container hold either an owning heap allocation or a caller-provided
/// buffer -- the no-heap / DMA path. Storage is also why
/// nothing here uses std::vector: owning allocation has to work with
/// exceptions disabled.
/// @note Kernels never take this type. They take the views returned by view and
/// constView, so a kernel compiles once per WordType regardless of
/// how its arguments were allocated or how their rows are aligned.
/// @note Error policy (core/error.hpp): every constructor and
/// every argument check below reports through BINCV_THROW, which throws by
/// default and prints-and-aborts where exceptions are disabled. The two
/// per-pixel accessors, at and set, are the exception -- they are
/// debug-checked and unchecked in release. So each `@throws` clause here
/// reads "throws, or aborts with that message under BINCV_NO_EXCEPTIONS",
/// and it is not repeated per function.
template <typename WordType_>
class QuantMat<1, WordType_> {
    static_assert(std::is_integral<WordType_>::value && std::is_unsigned<WordType_>::value,
                  "WordType must be an unsigned integral type");
    static_assert(sizeof(WordType_) == 1 || sizeof(WordType_) == 2 ||
                  sizeof(WordType_) == 4 || sizeof(WordType_) == 8,
                  "WordType must be 8, 16, 32, or 64 bits wide");

public:
    /// The storage word type, exposed for pointer-level access.
    using WordType = WordType_;

    /// Number of pixels packed into a single word.
    static constexpr size_t WordBits = sizeof(WordType) * 8;

    /// @brief Number of bit-planes: one, which is what makes this a BinMat.
    /// @note Present so that code generic over QuantMat<N> reads the same count
    /// here as it does for N > 1.
    static constexpr size_t Planes = 1;

    /// @brief Largest value a pixel can hold: 1, this being a binary matrix.
    /// @note Present for the same reason Planes is -- code generic over
    /// QuantMat<N> asks the container for its range rather than computing
    /// 2^N - 1 itself. The per-pixel accessors here are still bool-valued:
    /// a one-bit unsigned would be a worse type for the binary case, and
    /// the binary case is the one this specialization exists to serve.
    static constexpr unsigned MaxValue = 1u;

    /// @brief Default row-stride alignment: one word, i.e. no padding beyond the
    /// ceil(width / WordBits) words a row inherently needs.
    /// @note Word granularity already gives kernels the only property they rely
    /// on -- that the trailing partial word can be read and written whole --
    /// whereas fixed 16/32/64-byte row alignment costs up to 172% memory on
    /// upper pyramid levels, which LK touches every frame. Memory and speed are
    /// co-equal goals and memory wins a tie, so larger alignment is opt-in per
    /// object via the constructor's rowAlignment argument, not the default.
    /// @note Measured on the reference device (a Raspberry Pi 4, aarch64): across
    /// four alignments, two kernels and two frame sizes the best over-aligned
    /// case was 1.02× faster, inside its own batch spread, and countNonZero
    /// -- which walks rows unconditionally and so isolates alignment alone --
    /// was flat to within 0.5%. Over-aligning is also 3.3-4.8× SLOWER on
    /// bitwiseAnd at 640x480, because a stride wider than the row's own words
    /// disables ops/logic.hpp's contiguous fast path. So rowAlignment is for
    /// a caller who has measured a reason, not a default.
    static constexpr size_t DefaultRowAlignment = sizeof(WordType);

    // Constructors

    /// @brief Constructs an empty matrix with no allocated storage.
    QuantMat();

    /// @brief Constructs a zero-filled matrix that owns its storage.
    /// @param width Width of the matrix in pixels
    /// @param height Height of the matrix in pixels
    /// @param rowAlignment Number of bytes to align each row's stride to. Must be
    /// a positive power of two and a multiple of the word size; default is
    /// DefaultRowAlignment, i.e. word granularity with no padding.
    /// @throws std::invalid_argument if dimensions are negative or alignment is invalid.
    QuantMat(int width, int height, size_t rowAlignment = DefaultRowAlignment);

    /// @brief Wraps a caller-provided buffer without taking ownership of it.
    /// @param data First word of the caller's buffer. Must outlive this object and
    /// hold at least `height * strideWords` words.
    /// @param width Width of the matrix in pixels
    /// @param height Height of the matrix in pixels
    /// @param strideWords Distance between consecutive rows, in WORDS. Must be at
    /// least the ceil(width / WordBits) words a row needs.
    /// @throws std::invalid_argument if dimensions are negative, if `strideWords`
    /// cannot hold a row, or if `data` is null for a non-empty matrix.
    /// @note Allocates nothing and frees nothing: the buffer belongs to the caller
    /// for this object's whole lifetime. This is the no-heap path, and the
    /// way sensor / DMA memory is ingested without a copy.
    /// @note The buffer is used as-is; it is neither zeroed nor trailing-bit
    /// cleared, because the caller may be wrapping data it has already
    /// filled in, or a sub-region of a larger image whose surrounding
    /// columns must not be disturbed. The padding-bit invariant (bits from
    /// `width` to the end of the last used word are zero) is therefore the
    /// caller's to establish -- word-wise reductions over-count otherwise.
    /// Copying such a matrix re-establishes the invariant, since the copy
    /// owns its storage; see the copy constructor.
    /// @note Operations that reallocate (resize, pad, transpose, transposed, and
    /// fromCVMat) replace the wrapped buffer with an owning allocation rather
    /// than writing outside it, so the caller's buffer is left alone but no
    /// longer referenced.
    /// @note `strideWords` describes the caller's buffer, not an allocation policy:
    /// a wrapped matrix reports getRowAlignment == DefaultRowAlignment, so
    /// if it later reallocates it does so at word granularity regardless of
    /// how the wrapped rows were aligned. Construct at the desired alignment
    /// and copy in if a reallocating matrix must keep a wider stride.
    /// @note Operations that write whole words (fill, pad with `true`) write
    /// across the full stride, so wrapping a sub-region of a larger image
    /// and then calling them disturbs the surrounding columns.
    QuantMat(WordType* data, int width, int height, size_t strideWords);

    // Special members

    /// @brief Deep-copies `other`, always. The copy owns its storage.
    /// @note Value semantics: copy means deep copy, with no reference counting; sharing is
    /// expressed by taking a view instead. This holds even when `other`
    /// wraps an external buffer -- copying a non-owning BinMat allocates and
    /// copies rather than producing a second wrapper. A user-facing rule
    /// that silently changed with how the source happened to be constructed
    /// would reintroduce exactly the aliasing surprises value semantics exist to avoid.
    /// (Storage's own copy does alias a non-owning source; that stays an
    /// internal detail of the storage layer.)
    /// @note Because the copy owns its storage, it also re-establishes the
    /// padding-bit invariant when the source was a wrapped buffer whose bits
    /// past `width` were dirty. Otherwise the copy would inherit phantom bits
    /// that no per-pixel read can see but every word-wise reduction counts.
    QuantMat(const QuantMat& other);

    /// @brief Deep-copies `other`, always. See the copy constructor.
    /// @note The copy is built before this object releases anything, so assigning
    /// from a BinMat that wraps this object's own buffer copies live data.
    QuantMat& operator=(const QuantMat& other);

    /// @brief Takes over `other`'s storage, leaving it empty.
    /// @note A moved-from BinMat is a valid empty matrix -- dimensions included,
    /// so that empty cannot report false while data is null.
    QuantMat(QuantMat&& other) noexcept;

    /// @brief Takes over `other`'s storage, leaving it empty. See the move
    /// constructor.
    /// @note Moving from a matrix that WRAPS this object's own storage leaves this
    /// object unchanged (`other` is still emptied). Storage cannot honor
    /// such a transfer -- it would have to free the block and then adopt a
    /// pointer into it -- and this is the only answer that keeps the move
    /// allocation-free and noexcept, which the no-heap path depends on.
    QuantMat& operator=(QuantMat&& other) noexcept;

    ~QuantMat() = default;

    // Accessors
    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }

    /// @brief Byte alignment this matrix rounds its row stride up to when it
    /// allocates. Describes the allocation policy, not the current stride:
    /// a wrapped buffer keeps whatever stride the caller supplied.
    size_t getRowAlignment() const { return rowAlignment; }
    Size getSize() const { return Size(static_cast<int>(width), static_cast<int>(height)); }

    /// @brief Row stride in words: the distance from one row to the next.
    /// @note At the default alignment this is exactly ceil(width / WordBits).
    size_t getAlignedWidth() const { return alignedWidth; }

    /// @brief True if the matrix has no pixels.
    bool empty() const { return width == 0 || height == 0; }

    /// @brief True if this matrix will free its storage; false when it wraps a
    /// caller-provided buffer (or is empty).
    bool ownsMemory() const { return storage.ownsMemory(); }

    /// @brief The dimensions under the names cv::Mat gives its `rows` and `cols`
    /// data members, as accessors.
    int rows() const { return static_cast<int>(height); }
    int cols() const { return static_cast<int>(width); }

    /// @brief Raw access to the packed storage, for bulk/SIMD operations.
    const WordType* data() const { return storage.data(); }
    WordType* data() { return storage.data(); }

    /// @brief Total number of words in the backing store (height * alignedWidth).
    size_t sizeInWords() const { return storage.size(); }

    // Views -- the kernel interface

    /// @brief Non-owning mutable view over this matrix's pixels.
    /// @note The view borrows; it does not extend this matrix's lifetime, and it
    /// is invalidated by anything that reallocates (resize, pad, transpose,
    /// fromCVMat, assignment) exactly as a raw pointer would be.
    BinMatView<WordType> view() {
        return BinMatView<WordType>{storage.data(), width, height, alignedWidth};
    }

    /// @brief Non-owning read-only view over this matrix's pixels.
    /// @note Available on a non-const BinMat too, which is how a caller passes a
    /// mutable matrix to a kernel that only reads. See the note on
    /// BinMatView's conversion operator for why the explicit call is
    /// usually needed at a template kernel's call site.
    BinMatConstView<WordType> constView() const {
        return BinMatConstView<WordType>{storage.data(), width, height, alignedWidth};
    }

    /// @brief Bit-plane `i` as a view. There is exactly one, and it is this matrix.
    /// @param i Plane index; the only valid value is 0.
    /// @note The QuantMat<N> plane interface, at N = 1. Identical to view --
    /// same pointer, same stride, no offset arithmetic and no loop -- which
    /// is the whole point of the specialization: generic code can address
    /// the binary case by plane without the binary case paying for it.
    /// @throws std::out_of_range if `i != 0`, in EVERY build -- NOT debug-only.
    /// This is a view factory, not element access, and
    /// it is checked here for the same reason it is checked on QuantMat<N>:
    /// so that one wrong plane index has one defined behavior across the
    /// family. Discarding the index in release would mean generic code could
    /// not be tested for index handling at N = 1 and have the result carry
    /// to N > 1 -- which is precisely the portability the
    /// QuantMat<1> spelling exists to provide. The compare is against a
    /// constant and folds away at the constant call sites.
    BinMatView<WordType> plane(size_t i) {
        if (i != 0) BINCV_THROW(std::out_of_range, "QuantMat<1>::plane: index out of range");
        return view();
    }
    BinMatConstView<WordType> plane(size_t i) const {
        if (i != 0) BINCV_THROW(std::out_of_range, "QuantMat<1>::plane: index out of range");
        return constView();
    }

    /// @brief Plane `i` as a read-only view, from a NON-const matrix.
    /// @note The QuantMat<N> spelling of constView, present so that generic code
    /// can hand a plane to a read-only kernel at N = 1 as well. Same reason
    /// constView exists: template argument deduction ignores the
    /// BinMatView -> BinMatConstView conversion.
    BinMatConstView<WordType> constPlane(size_t i) const { return plane(i); }

    /// @brief Words occupied by one plane -- here, by the whole matrix.
    /// @note The QuantMat<N> layout accessor at N = 1: plane i begins at word
    /// offset i * planeWords, and with one plane that offset is zero.
    size_t planeWords() const { return height * alignedWidth; }

#ifdef BINCV_WITH_OPENCV
    // OpenCV interoperability (only available when BINCV_WITH_OPENCV is defined)

    /// @brief Replaces this matrix with a binarized copy of a `CV_8UC1` cv::Mat:
    /// any nonzero byte becomes 1. **API TIER 3** -- OpenCV has no packed 1-bit
    /// image, so this is the bridge rather than an equivalent.
    /// @param mat The input, which must be `CV_8UC1`; other depths are the
    /// caller's to convert first, or to pack directly through ops/pack.hpp.
    /// @throws std::invalid_argument if `mat` is empty or not `CV_8UC1`.
    /// @note Allocates owning storage at this matrix's row alignment, so calling
    /// this on a matrix that wraps a caller buffer detaches it from that buffer.
    void fromCVMat(const cv::Mat& mat);

    /// @brief Writes this matrix as a `CV_8UC1` cv::Mat holding 0 or 1 per pixel.
    /// **API TIER 3** -- see fromCVMat.
    /// @param mat The output; always `CV_8UC1`.
    void toCVMat(cv::Mat& mat) const;

    /// @brief Writes this matrix as a `CV_8UC1` cv::Mat holding 0 or 255 per pixel,
    /// which is what an image viewer or an OpenCV operation expects of a binary
    /// image. **API TIER 3** -- see fromCVMat.
    /// @param mat The output; always `CV_8UC1`.
    void toCVMatNormalized(cv::Mat& mat) const;
#endif // BINCV_WITH_OPENCV

    /// @brief Gets the value of a single element at (row, col). Not a reference.
    /// **API TIER 1** -- `cv::Mat::at`'s contract on a binary image.
    /// @note DEBUG-CHECKED, UNCHECKED IN RELEASE, as cv::Mat::at is. An index
    /// outside [0, height) x [0, width) trips a BINCV_ASSERT in a debug
    /// build and is undefined behavior in a release one -- it does not
    /// throw. This is what
    /// keeps the bounds test out of every per-pixel loop and lets a release
    /// build inline the access down to a row offset, a shift and a mask.
    /// @note Reading a column in [width, alignedWidth * WordBits) therefore
    /// silently returns a padding bit rather than reporting the mistake.
    bool at(int row, int col) const;

    /// @brief Sets a single element at (row, col) to value. **API TIER 1** -- the
    /// write half of `cv::Mat::at`'s contract.
    /// @note Debug-checked, unchecked in release; see at.
    /// @note Writing past `width` breaks the padding-bit invariant that every
    /// word-wise reduction depends on, and release builds will not stop you.
    void set(int row, int col, bool value);

    /// @brief Sets a single element from an integral value; only 0 and 1 fit.
    /// @note Exists so the QuantMat<N> value-range precondition survives at N = 1.
    /// The parameter above is `bool`, which is the right type for the binary
    /// case but silently accepts anything nonzero: code written generically
    /// as `m.set(y, x, value)` got "value does not fit in N bits" for N >= 2
    /// and a quiet narrowing to 1 for N == 1, in the very build whose job is
    /// to catch it. This overload restores the check without changing what
    /// `set(y, x, true)` means -- a bool argument is an exact match for the
    /// bool overload and never reaches here.
    /// @note Debug-checked like the rest of the per-pixel path. Negative values
    /// fail it too: the cast makes them large, which is what they are as a
    /// pixel value.
    template <typename T, typename = typename std::enable_if<
                              std::is_integral<T>::value &&
                              !std::is_same<T, bool>::value>::type>
    void set(int row, int col, T value) {
        BINCV_ASSERT(static_cast<unsigned long long>(value) <= MaxValue,
                     "QuantMat<1>::set: value does not fit in N bits");
        set(row, col, value != 0);
    }

    /// @brief First word of row `row`, read-only. **API TIER 3** -- `cv::Mat::ptr`'s
    /// shape, but the row it points at is bit-packed words, which OpenCV has no
    /// equivalent of.
    /// @note Unchecked: an out-of-range row is undefined behavior. The caller
    /// addresses pixels within the row through impl::wordIndex / impl::bitMask,
    /// or hands the whole matrix to a kernel as a view, which is the usual way.
    const WordType* ptr(int row) const;

    /// @brief First word of row `row`, writable. **API TIER 3** -- see the const form.
    /// @note Writing past `width` breaks the padding-bit invariant; see set.
    WordType* ptr(int row);

    /// @brief Reshapes the matrix to `newWidth` x `newHeight`, keeping the pixels
    /// that still fit and zero-filling the rest. **API TIER 3** -- a crop or
    /// extension at the origin: NOT `cv::resize` (no interpolation) and not
    /// `cv::Mat::resize` (which changes rows only).
    /// @note Pixels at indices past the new extent are dropped; new rows and
    /// columns are zero. To add rows or columns on a chosen side, use pad.
    /// @note Allocates owning storage, so this detaches a wrapped matrix from the
    /// caller's buffer instead of writing outside it.
    /// @throws std::invalid_argument if either dimension is negative.
    void resize(int newWidth, int newHeight);

    /// @brief Adds `top`, `bottom`, `left` and `right` pixels of border, filled with
    /// `value`. **API TIER 3** -- `cv::copyMakeBorder`'s role with a constant
    /// border only: no replicate or reflect modes.
    /// @note Zero fill unless `value` is true.
    /// @note Allocates owning storage, as resize does.
    /// @throws std::invalid_argument if any border is negative.
    void pad(int top, int bottom, int left, int right, bool value = false);

    /// @brief A transposed copy of this matrix. **API TIER 3** -- `cv::transpose`'s
    /// role on a bit-packed matrix, as a container method; no equivalence
    /// against OpenCV is claimed or tested.
    /// @note The original matrix is unchanged.
    /// @note The result owns its storage and is built at this matrix's row
    /// alignment, whether or not this matrix wraps a caller buffer.
    /// @note An empty matrix transposes to its transposed shape, not to 0x0:
    /// 640x0 gives 0x640.
    QuantMat transposed() const;

    /// @brief Transposes this matrix. **API TIER 3** -- see transposed.
    /// @note "In-place" describes the variable, not the buffer: this builds the
    /// transpose in a fresh owning allocation and adopts it, so it detaches a
    /// wrapped matrix from the caller's buffer as resize and pad do. The
    /// caller's buffer is left unmodified -- including for a square matrix,
    /// where an in-place bit transpose would otherwise be the natural reading.
    void transpose();

    /// @brief Iterates over all non-zero pixels, invoking callback(row, col).
    /// **API TIER 3** -- `cv::findNonZero`'s role as a callback rather than a
    /// point list, since a point list would allocate.
    /// @throws std::runtime_error on an empty matrix: iterating something with no
    /// pixels is a caller mistake, not a degenerate case with an obvious answer.
    template <typename Func>
    void forEachNonZero(Func callback) const;

    /// @brief Prints the pixels as 0/1 characters, one row per line, to stdout.
    /// **API TIER 3** -- a debugging aid.
    /// @note Writes through <cstdio>, not std::cout, so that including this
    /// container never drags the iostream static initializers into an embedded
    /// build. See the include block at the top of this file.
    void printMatrix() const;

    /// @brief Prints the packed words of the backing store, one row per line, to
    /// stdout. **API TIER 3** -- a debugging aid.
    /// @param hex If true, prints each word in hex; otherwise decimal.
    void printInternalData(bool hex = false) const;

    /// @brief Sets every pixel to `value`. **API TIER 3** -- the role of
    /// `cv::Mat::setTo` on a binary image.
    /// @note Padding bits stay zero: a `true` fill writes whole words and then
    /// clears the bits past `width`.
    void fill(bool value);

    /// @brief The number of set pixels. **API TIER 1** -- the count `cv::countNonZero`
    /// returns for the same image, checked in tests/test_opencv_interop.cpp.
    /// @note A per-pixel loop. The bulk, word-wise form is
    /// `bincv::countNonZero(m.constView())` in ops/reduce.hpp, which is the one
    /// to call on a hot path; see the definition for why this member does not
    /// simply forward to it.
    int countNonZero() const;

    /// @brief The fraction of pixels that are zero, in [0.0, 1.0]. **API TIER 3.**
    /// @throws std::runtime_error on an empty matrix: sparsity is undefined for
    /// 0/0, and no value is invented.
    float sparsity() const;

private:
    // @brief Zeroes the padding bits beyond `width` in every row.
    // @note Bulk word-wise operations (fill here, and the kernels in ops/logic.hpp
    // and ops/reduce.hpp) write or read whole words, so bits past the end of a
    // row must stay zero or a word-wise reduction over-counts.
    // @note This matters more, not less, now that the default stride is tight: at
    // word granularity the only padding left is the tail of the last word,
    // and that tail is read by every whole-word reduction.
    void clearTrailingBits();

    // Dimensions are in number of pixels
    size_t width;
    size_t height;
    size_t rowAlignment;  // bytes each row's stride is rounded up to when allocating
    size_t alignedWidth;  // row stride in words

    // @note width stores the number of pixels in each row, while alignedWidth stores
    // the actual number of words between one row and the next. At the default
    // alignment they differ only by the ceil to a whole word.
    // @note rowAlignment aligns only the row STRIDE. The base pointer is whatever
    // `new` returns, so no kernel may assume an aligned base; the vector arms
    // use unaligned loads.

    // Internal storage: row-wise packed 1-bit pixels, height * alignedWidth words.
    // Storage (not std::vector) so the same container can wrap caller-provided
    // memory and so owning allocation works without exceptions.
    Storage<WordType> storage;
};

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv

#include "impl/binMat_impl.hpp"
