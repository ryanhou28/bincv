#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

// BinMat -- which is QuantMat<1>, the hand-written single-plane container this
// file builds the N-plane case out of (core/types.hpp declares the alias, and the
// primary template both it and this file specialize/define). It carries
// core/error.hpp, core/storage.hpp, core/types.hpp and core/view.hpp with it, as
// well as the impl:: word helpers the per-pixel accessors below use.
#include "binMat.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

namespace impl {

/// @brief |value| as an unsigned, computed WITHOUT negating a signed int.
/// @note Not a stylistic choice. `-value` at value == INT_MIN is signed overflow,
///       which is undefined behaviour -- and undefined in a way that matters more
///       than the wrong answer: it licenses the optimizer to assume the argument
///       is never INT_MIN and to drop the `value < 0` test that guards it. The
///       range assert in SignedQuantMat::set is compiled out under NDEBUG, which
///       is every configuration this project verifies, so the guard cannot be
///       relied on to keep INT_MIN away from here.
/// @note constexpr on purpose, so the property is testable: a constant expression
///       may not contain signed overflow, so `signedMagnitude(INT_MIN)` inside a
///       static_assert is ill-formed the moment anyone rewrites this as `-value`.
///       tests/test_quantMat.cpp holds that static_assert.
constexpr unsigned signedMagnitude(int value) {
    return value < 0 ? 0u - static_cast<unsigned>(value) : static_cast<unsigned>(value);
}

#ifdef BINCV_WITH_OPENCV
/// @brief 8x8 bit-matrix transpose: bit (8r + c) moves to bit (8c + r).
/// @note The pivot of QuantMat's cv::Mat conversions (X-47). One call moves 8
///       pixels x 8 planes between the two layouts -- byte r holding plane r's
///       bits in, byte c holding pixel c's value out -- so the conversion runs at
///       ~3 ops per pixel for ALL planes instead of one bit test per (pixel,
///       plane). Three delta-swaps (Hacker's Delight 7-3); an involution, so both
///       conversion directions call the same function.
/// @note Planes p >= N hold zero bytes on the way in, so value bits >= N are zero
///       on the way out -- the N < 8 case needs no masking.
inline uint64_t transpose8x8(uint64_t x) {
    uint64_t t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
    x ^= t ^ (t << 28);
    return x;
}
#endif // BINCV_WITH_OPENCV

} // namespace impl

/// @brief An N-bit image, stored as N bit-planes in ONE contiguous allocation.
/// @tparam N Number of bit-planes, i.e. bits per pixel. 1 to 8.
/// @tparam WordType_ The unsigned integral type pixels are packed into.
///
/// @note WHY THIS EXISTS, and it is measured rather than speculative: a binary
///       frame does not stay binary through the reference pyramid. Distinct
///       values grow 2 -> 5 -> 15 -> 26 across levels 0..3, i.e. 1 -> 3 -> 4 -> 5
///       bits (ARCHITECTURE 7.2). A binary-only library cannot represent pyramid
///       level 1 at all, so the VIO frontend needs this container.
/// @note Layout (ARCHITECTURE 4.1, 4.3): plane p holds bit p of every pixel,
///       row-packed exactly as a BinMat is, and plane p begins at word offset
///       p * planeWords(). ONE allocation holds all N planes -- not N of them --
///       which is what keeps the footprint arithmetic honest on a constrained
///       target, keeps planes local to each other, and lets a caller wrap one
///       external buffer rather than N.
/// @note plane(0) is the LEAST significant bit.
/// @note UNINTERPRETED. This container holds N planes and assigns them no
///       meaning: it does not know whether they encode an unsigned value, a
///       signed one, or N unrelated masks. Sign-magnitude interpretation is
///       SignedQuantMat, below, which adds no storage of its own.
/// @note Implemented as a shape over BinMat rather than as a second container:
///       the member below is a single BinMat of `N * height` rows, so the whole
///       plane stack is one BinMat-shaped allocation and plane p is rows
///       [p*height, (p+1)*height) of it. Everything subtle about the storage --
///       deep-copy value semantics (D-8), the aliasing rules on copy- and
///       move-assignment, owning versus wrapped buffers, the zero-fill that keeps
///       padding bits clear -- is therefore the code T1.1/T1.3 already have tests
///       for, not a second implementation of it that could drift.
/// @note Kernels never take this type. They take the per-plane views returned by
///       plane() (D-5), so a kernel compiles once per WordType and does not care
///       how many planes its caller happens to have.
/// @note Error policy: the constructors validate through BinMat's checks, so a
///       bad argument reports with BinMat's diagnostic -- the two containers are
///       one type family and share the check. Per-pixel access is debug-checked
///       and unchecked in release, as BinMat::at() and BinMat::set() are.
///       plane() is NOT per-pixel and is checked in every build; see there.
/// @note N == 1 does NOT come here. QuantMat<1, WordType> is a separate,
///       hand-written specialization (binMat.hpp, spelled BinMat) with no plane
///       loop and no delegation anywhere on its per-pixel path. It differs from
///       this template in THREE observable ways. Two are consequences of the
///       binary case deserving `bool`:
///         - at() returns bool rather than a one-bit unsigned;
///         - set() takes bool, so its value-range precondition is checked by a
///           separate integral overload rather than falling out of the parameter
///           type. Passing 2 to QuantMat<1>::set is diagnosed there, exactly as
///           passing 2^N to this one is here.
///       The third is a deliberate semantic difference, not a spelling one:
///         - fromCVMat at N == 1 reads any NONZERO byte as 1; this template
///           quantizes to NEAREST, so the two disagree for input bytes 1..127
///           (X-47, and the block comment on the conversions below).
///       Everything else -- Planes, MaxValue, planeWords(), plane(), constPlane()
///       and the plane index contract -- reads the same at N == 1 as it does here,
///       so code generic over QuantMat<N> is testable at N == 1 EXCEPT on the
///       fromCVMat path, where the third difference above makes N == 1 a genuinely
///       different function rather than the same one at a smaller N.
template <size_t N, typename WordType_>
class QuantMat {
    static_assert(N >= 1 && N <= 8, "N outside supported range");

public:
    /// The storage word type, exposed for pointer-level access.
    using WordType = WordType_;

    /// Number of pixels packed into a single word.
    static constexpr size_t WordBits = sizeof(WordType) * 8;

    /// Number of bit-planes, i.e. bits per pixel.
    static constexpr size_t Planes = N;

    /// Largest value a pixel can hold: 2^N - 1.
    static constexpr unsigned MaxValue = (1u << N) - 1u;

    /// @brief Default row-stride alignment: word granularity (D-4). See BinMat.
    static constexpr size_t DefaultRowAlignment = sizeof(WordType);

    // Constructors

    /// @brief Constructs an empty matrix with no allocated storage.
    QuantMat() = default;

    /// @brief Constructs a zero-filled matrix that owns its storage.
    /// @param width Width of the matrix in pixels
    /// @param height Height of the matrix in pixels
    /// @param rowAlignment Bytes to align each row's stride to; default is word
    ///        granularity with no padding (D-4). Opt-in larger alignment applies
    ///        to every plane, since the planes share one stride.
    /// @throws std::invalid_argument if dimensions are negative, if the plane
    ///         stack does not fit an int, or if the alignment is invalid.
    /// @note Exactly ONE allocation, of N * height * stride words, zero-filled --
    ///       which is what starts the padding bits clear in every plane.
    QuantMat(int width, int height, size_t rowAlignment = DefaultRowAlignment)
        : stack_(width, checkedStackHeight(height), rowAlignment) {}

    /// @brief Wraps a caller-provided buffer holding all N planes, without
    ///        taking ownership of it.
    /// @param data First word of the caller's buffer. Must outlive this object
    ///        and hold at least `N * height * strideWords` words.
    /// @param width Width of the matrix in pixels
    /// @param height Height of the matrix in pixels
    /// @param strideWords Distance between consecutive rows, in WORDS. Must be at
    ///        least ceil(width / WordBits).
    /// @throws std::invalid_argument as the sized constructor does, and if `data`
    ///         is null for a non-empty matrix.
    /// @note Allocates nothing: the Tier 2 / DMA path, and the reason the planes
    ///       are one contiguous block rather than N -- a caller has one buffer to
    ///       hand over, laid out plane 0 first.
    /// @note The buffer is used as-is -- neither zeroed nor trailing-bit cleared.
    ///       The padding-bit invariant is the caller's to establish, exactly as it
    ///       is for a wrapped BinMat.
    /// @note THE SIZE REQUIREMENT IS N-FOLD AND THIS SIGNATURE CANNOT CHECK IT.
    ///       It is spelled exactly like BinMat's wrapping constructor but needs N
    ///       times as many words for the same arguments, and it is handed no
    ///       length, so an undersized buffer is accepted silently and every access
    ///       above plane 0 lands in the caller's neighbouring objects. Prefer
    ///       wrap(), below, which takes the buffer length and checks it -- this
    ///       constructor is the one the T1.5 spec names, kept for callers whose
    ///       length is not a value they have.
    QuantMat(WordType* data, int width, int height, size_t strideWords)
        : stack_(data, width, checkedStackHeight(height), strideWords) {}

    /// @brief Wraps a caller-provided buffer, CHECKING that it is long enough.
    /// @param data First word of the caller's buffer, plane 0 first.
    /// @param bufferWords Words the caller's buffer actually holds.
    /// @param width Width of the matrix in pixels
    /// @param height Height of the matrix in pixels
    /// @param strideWords Distance between consecutive rows, in WORDS.
    /// @throws std::invalid_argument if the buffer cannot hold N * height *
    ///         strideWords words, plus everything the wrapping constructor throws.
    /// @note Same object as the wrapping constructor produces, same lack of
    ///       ownership and same zero allocations -- the one difference is that the
    ///       container is told how much memory it was given, so the one thing it
    ///       cannot otherwise verify becomes a setup-time check. Free on the Tier 2
    ///       path: it runs once per wrap, not per access.
    /// @note Named rather than a fifth constructor parameter so the buffer and its
    ///       length sit together and neither can be defaulted away.
    static QuantMat wrap(WordType* data, size_t bufferWords, int width, int height,
                         size_t strideWords) {
        // The N-fold height overflow first, so that a caller who tripped it reads
        // that diagnostic rather than a length complaint derived from it.
        const int stackRows = checkedStackHeight(height);
        // Compared by division rather than by forming N * height * strideWords: a
        // product that overflows size_t would wrap into a small requirement that
        // the undersized buffer then satisfies, which is the opposite of a check.
        // A non-positive height or a zero stride is left to the constructor below,
        // whose diagnostics name those two mistakes properly.
        if (stackRows > 0 && strideWords > 0 &&
            bufferWords / strideWords < static_cast<size_t>(stackRows)) {
            BINCV_THROW(std::invalid_argument,
                        "QuantMat::wrap: buffer is shorter than N x height x strideWords words");
        }
        return QuantMat(data, width, height, strideWords);
    }

    // Special members are the compiler's: the only member is a BinMat, whose own
    // copy/move already implement D-8 (copy means deep copy, wrapped or not) and
    // the two assignment aliasing rules. Writing them out here would duplicate
    // that reasoning, and any divergence would be silent.

    // Accessors

    /// @brief Width in pixels.
    size_t getWidth() const { return stack_.getWidth(); }

    /// @brief Height in pixels, of ONE plane -- not of the plane stack.
    size_t getHeight() const { return stack_.getHeight() / N; }

    Size getSize() const {
        return Size(static_cast<int>(getWidth()), static_cast<int>(getHeight()));
    }

    /// @brief Row stride in words, shared by every plane.
    size_t getAlignedWidth() const { return stack_.getAlignedWidth(); }

    /// @brief Byte alignment rows are rounded up to when this matrix allocates.
    size_t getRowAlignment() const { return stack_.getRowAlignment(); }

    /// @brief True if the matrix has no pixels.
    bool empty() const { return stack_.empty(); }

    /// @brief True if this matrix will free its storage.
    bool ownsMemory() const { return stack_.ownsMemory(); }

    // OpenCV-compatible aliases
    int rows() const { return static_cast<int>(getHeight()); }
    int cols() const { return static_cast<int>(getWidth()); }

    /// @brief First word of plane 0. The other planes follow it contiguously.
    const WordType* data() const { return stack_.data(); }
    WordType* data() { return stack_.data(); }

    /// @brief Total words backing ALL N planes: N * planeWords().
    size_t sizeInWords() const { return stack_.sizeInWords(); }

    /// @brief Words in one plane: height * stride.
    /// @note This is the plane pitch. Plane p begins at data() + p * planeWords().
    size_t planeWords() const { return getHeight() * getAlignedWidth(); }

    // Planes -- the kernel interface (D-5)

    /// @brief Plane `i` as a mutable view, plane 0 being the LEAST significant bit.
    /// @param i Plane index, in [0, N).
    /// @throws std::out_of_range if `i >= N`, in EVERY build. (Or, without
    ///         exceptions, prints that message and aborts -- see core/error.hpp.)
    /// @note Non-owning, like every view: it borrows this matrix's storage and is
    ///       invalidated by anything that reallocates.
    /// @note VALIDATION, not element access -- which is why this is the one
    ///       accessor on the container that is checked in release too, where at()
    ///       and set() are not. ARCHITECTURE 5.3 puts the unchecked-in-release
    ///       rule on per-pixel access; plane() is a view factory called at most N
    ///       (<= 8) times per image, so the compare against a compile-time
    ///       constant costs nothing measurable and folds away entirely at the
    ///       constant call sites this interface is written for. The blast radius
    ///       also differs in kind: an out-of-range at() reads one bit, whereas an
    ///       out-of-range plane() hands back a fully-formed, writable view of
    ///       whatever allocation follows this one -- 37.5 KiB of someone else's
    ///       memory at 640x480, with no diagnostic.
    /// @note The same check, with the same spelling, is on QuantMat<1>::plane()
    ///       and on SignedQuantMat::magnitude(). One wrong plane index has one
    ///       defined behaviour across the whole type family, so index handling
    ///       tested at one N carries to the others.
    BinMatView<WordType> plane(size_t i) {
        if (i >= N) BINCV_THROW(std::out_of_range, "QuantMat::plane: index out of range");
        return BinMatView<WordType>{data() + i * planeWords(), getWidth(), getHeight(),
                                    getAlignedWidth()};
    }

    /// @brief Plane `i` as a read-only view. See the mutable overload.
    BinMatConstView<WordType> plane(size_t i) const {
        if (i >= N) BINCV_THROW(std::out_of_range, "QuantMat::plane: index out of range");
        return BinMatConstView<WordType>{data() + i * planeWords(), getWidth(), getHeight(),
                                         getAlignedWidth()};
    }

    /// @brief Plane `i` as a read-only view, from a NON-const matrix.
    /// @note Mirrors BinMat::constView(), and exists for the same reason: template
    ///       argument deduction does not consider user-defined conversions, so a
    ///       kernel declared `f(BinMatConstView<W>)` will not deduce W from the
    ///       BinMatView the mutable overload above returns -- and on a non-const
    ///       matrix that overload is the one that wins. `f(m.constPlane(0))` is
    ///       how a mutable matrix hands a plane to a read-only kernel (D-5, D-9).
    BinMatConstView<WordType> constPlane(size_t i) const { return plane(i); }

    // Per-pixel access

    /// @brief Reads the N-bit value at (row, col), plane 0 contributing bit 0.
    /// @return The value in [0, MaxValue].
    /// @note DEBUG-CHECKED, UNCHECKED IN RELEASE, as BinMat::at() is. An index
    ///       outside the matrix trips a BINCV_ASSERT in a debug build and is
    ///       undefined behaviour in a release one.
    /// @note This is the per-pixel path, provided for setup, tests and printing.
    ///       It is not how images are processed: bulk work goes plane-wise
    ///       through plane(), which is the whole point of the representation.
    /// @note Assembling bit p of the pixel from plane p is the container's one
    ///       fixed reading of its planes -- it is what "plane(0) is the least
    ///       significant bit" means. It says nothing about signedness; a
    ///       SignedQuantMat's sign plane would read as this value's top bit,
    ///       which is why that type has its own accessor.
    unsigned at(int row, int col) const {
        BINCV_ASSERT(row >= 0 && row < static_cast<int>(getHeight()) &&
                         col >= 0 && col < static_cast<int>(getWidth()),
                     "QuantMat::at: index out of range");
        const size_t x = static_cast<size_t>(col);
        const WordType* word = data() + static_cast<size_t>(row) * getAlignedWidth() +
                               impl::wordIndex<WordType>(x);
        const WordType mask = impl::bitMask<WordType>(x);
        const size_t pitch = planeWords();

        unsigned value = 0;
        for (size_t p = 0; p < N; ++p) {
            if (word[p * pitch] & mask) value |= (1u << p);
        }
        return value;
    }

    /// @brief Writes the N-bit value at (row, col), plane 0 taking bit 0.
    /// @param value Must be in [0, MaxValue]; bits above N are a caller error.
    /// @note Debug-checked, unchecked in release; see at().
    /// @note Writes every plane, including the ones whose bit is 0, so the pixel
    ///       ends up holding exactly `value` rather than the OR of what was there.
    void set(int row, int col, unsigned value) {
        BINCV_ASSERT(row >= 0 && row < static_cast<int>(getHeight()) &&
                         col >= 0 && col < static_cast<int>(getWidth()),
                     "QuantMat::set: index out of range");
        BINCV_ASSERT(value <= MaxValue, "QuantMat::set: value does not fit in N bits");
        const size_t x = static_cast<size_t>(col);
        WordType* word = data() + static_cast<size_t>(row) * getAlignedWidth() +
                         impl::wordIndex<WordType>(x);
        const WordType mask = impl::bitMask<WordType>(x);
        const size_t pitch = planeWords();

        for (size_t p = 0; p < N; ++p) {
            if ((value >> p) & 1u) {
                word[p * pitch] |= mask;
            } else {
                word[p * pitch] &= static_cast<WordType>(~mask);
            }
        }
    }

#ifdef BINCV_WITH_OPENCV
    // OpenCV interoperability (only when BINCV_WITH_OPENCV is defined).
    //
    // THE VALUE MAPPING IS FIXED AND DELIBERATELY ASYMMETRIC (X-47):
    //
    //   toCVMat            raw values 0..MaxValue. Exact -- and ONE-WAY at N < 8:
    //                      run back through fromCVMat, raw values would be
    //                      re-quantized toward zero. For masks and inspection.
    //   toCVMatNormalized  round(v * 255 / MaxValue) -- the OpenCV bridge. What
    //                      to call before handing a wide intermediate to an
    //                      OpenCV operation (X-46: above the crossover, OpenCV is
    //                      the faster implementation and this is the way there).
    //   fromCVMat          round(v * MaxValue / 255) -- toCVMatNormalized's EXACT
    //                      inverse: fromCVMat(toCVMatNormalized(m)) == m at every
    //                      pixel and every N. No rounding ties exist: 255 and
    //                      MaxValue are both odd, so v*255/MaxValue and
    //                      v*MaxValue/255 are never half-integers.
    //
    // The QuantMat<1> SPECIALIZATION keeps its established nonzero-threshold
    // fromCVMat (any set byte reads 1); this general form quantizes to nearest,
    // so the two disagree for bytes 1..127 at N == 1. Recorded difference
    // (X-47), not retroactively unified -- N == 1 callers depend on the
    // threshold reading. THIS IS THE THIRD DIVERGENCE the class docstring
    // counts, and the one that makes generic-over-N code NOT testable at N == 1
    // on the conversion path specifically.

    /// @brief Replaces this matrix with a quantized copy of an 8-bit cv::Mat:
    ///        each byte v becomes round(v * MaxValue / 255). **API TIER 3** --
    ///        no cv:: equivalent, since OpenCV has no packed sub-byte image type.
    /// @throws std::invalid_argument if the input is empty or not CV_8UC1.
    /// @note ALLOCATES, and therefore DETACHES a wrapped buffer: a matrix built on
    ///       the wrapping constructor stops referring to the caller's memory here
    ///       and owns its storage afterwards, exactly as the N == 1 specialization
    ///       does. The row alignment IS preserved.
    /// @note Container-level, not a kernel -- CLAUDE.md's no-heap rule binds
    ///       kernels, and this has the same shape as the N == 1 fromCVMat.
    /// @note Commit-last, as the N == 1 specialization and resize(): the new
    ///       storage is built completely before this object's dimensions change,
    ///       so a failed allocation leaves the matrix as it was.
    /// @note Padding bits are zero on return: the fresh storage starts zeroed and
    ///       only real pixels are written (D-13).
    void fromCVMat(const cv::Mat& input) {
        if (input.empty()) {
            BINCV_THROW(std::invalid_argument, "Input cv::Mat is empty");
        }
        if (input.type() != CV_8UC1) {
            BINCV_THROW(std::invalid_argument, "Input cv::Mat must be of type CV_8UC1");
        }
        // Quantize once into a table, not once per pixel: 256 bytes, and the
        // per-pixel work drops to one load.
        uint8_t lut[256];
        for (unsigned v = 0; v < 256u; ++v) {
            lut[v] = static_cast<uint8_t>((v * MaxValue + 127u) / 255u);
        }
        // Alignment is carried through UNCONDITIONALLY, as the N == 1
        // specialization and resize() do. An `empty() ? DefaultRowAlignment : ...`
        // guard here was a bug: every constructor establishes a valid alignment, so
        // it protected against nothing and silently downgraded an opt-in Tier 2 /
        // DMA stride. It bit on buffer REUSE rather than the degenerate case -- a
        // moved-from matrix is empty but keeps its alignment, so
        // `dst = std::move(src); src.fromCVMat(frame);` rebuilt src at word
        // granularity.
        QuantMat fresh(input.cols, input.rows, getRowAlignment());
        const size_t w = static_cast<size_t>(input.cols);
        const size_t pitch = fresh.planeWords();
        for (size_t y = 0; y < static_cast<size_t>(input.rows); ++y) {
            const uint8_t* in = input.ptr<uint8_t>(static_cast<int>(y));
            WordType* row0 = fresh.data() + y * fresh.getAlignedWidth();
            for (size_t x0 = 0; x0 < w; x0 += 8) {
                const size_t n = (w - x0 < 8) ? (w - x0) : size_t{8};
                uint64_t m = 0;
                for (size_t i = 0; i < n; ++i) {
                    m |= static_cast<uint64_t>(lut[in[x0 + i]]) << (8 * i);
                }
                const uint64_t t = impl::transpose8x8(m);
                // A group of 8 never straddles words: WordBits is a multiple of 8.
                const size_t wi = impl::wordIndex<WordType>(x0);
                const size_t shift = x0 % WordBits;
                for (size_t p = 0; p < N; ++p) {
                    const uint64_t bits = (t >> (8 * p)) & 0xFFu;
                    row0[p * pitch + wi] |= static_cast<WordType>(bits << shift);
                }
            }
        }
        *this = std::move(fresh);
    }

    /// @brief Writes this matrix as CV_8U holding the RAW values 0..MaxValue.
    ///        **API TIER 3** -- see fromCVMat.
    /// @note One-way at N < 8 -- see the block comment above.
    void toCVMat(cv::Mat& output) const {
        uint8_t lut[MaxValue + 1u];
        for (unsigned v = 0; v <= MaxValue; ++v) lut[v] = static_cast<uint8_t>(v);
        toCVMatWith(output, lut);
    }

    /// @brief Writes this matrix as CV_8U scaled to the full byte range:
    ///        round(v * 255 / MaxValue). **API TIER 3** -- see fromCVMat.
    /// @note `fromCVMat(toCVMatNormalized(m)) == m` exactly, at every pixel and
    ///       every N -- a LEFT inverse. THE OTHER COMPOSITION IS NOT THE IDENTITY
    ///       and no claim is made about it: fromCVMat is not injective for N < 8
    ///       (at N = 3 the bytes 0..18 all quantize to 0), so it has no inverse in
    ///       that direction.
    void toCVMatNormalized(cv::Mat& output) const {
        uint8_t lut[MaxValue + 1u];
        for (unsigned v = 0; v <= MaxValue; ++v) {
            lut[v] = static_cast<uint8_t>((v * 255u + (MaxValue >> 1)) / MaxValue);
        }
        toCVMatWith(output, lut);
    }
#endif // BINCV_WITH_OPENCV

private:
#ifdef BINCV_WITH_OPENCV
    /// @brief The shared export loop: 8 pixels x N planes per transpose, then a
    ///        table lookup per pixel. `lut` maps a raw value to its output byte.
    void toCVMatWith(cv::Mat& output, const uint8_t (&lut)[MaxValue + 1u]) const {
        if (empty()) {
            output = cv::Mat();
            return;
        }
        output.create(rows(), cols(), CV_8U);
        const size_t w = getWidth();
        const size_t pitch = planeWords();
        for (size_t y = 0; y < getHeight(); ++y) {
            const WordType* row0 = data() + y * getAlignedWidth();
            uint8_t* out = output.ptr<uint8_t>(static_cast<int>(y));
            for (size_t x0 = 0; x0 < w; x0 += 8) {
                const size_t wi = impl::wordIndex<WordType>(x0);
                const size_t shift = x0 % WordBits;
                uint64_t m = 0;
                for (size_t p = 0; p < N; ++p) {
                    m |= ((static_cast<uint64_t>(row0[p * pitch + wi]) >> shift) & 0xFFu)
                         << (8 * p);
                }
                const uint64_t t = impl::transpose8x8(m);
                // The tail group reads padding bits, which are zero by invariant,
                // and writes only the real pixels.
                const size_t n = (w - x0 < 8) ? (w - x0) : size_t{8};
                for (size_t i = 0; i < n; ++i) {
                    out[x0 + i] = lut[static_cast<size_t>((t >> (8 * i)) & 0xFFu)];
                }
            }
        }
    }
#endif // BINCV_WITH_OPENCV

private:
    /// @brief Rows the plane stack needs: N per image row.
    /// @note Checked rather than trusted, because it is the one arithmetic this
    ///       container adds on top of BinMat's: BinMat takes int dimensions, and
    ///       an N-fold height that no longer fits one would otherwise be
    ///       truncated into a plausible smaller matrix. Negative input is left to
    ///       BinMat, so that its diagnostic stays the one a caller sees.
    static int checkedStackHeight(int height) {
        constexpr size_t maxHeight =
            static_cast<size_t>(std::numeric_limits<int>::max()) / N;
        if (height > 0 && static_cast<size_t>(height) > maxHeight) {
            BINCV_THROW(std::invalid_argument,
                        "QuantMat height x N planes does not fit in an int");
        }
        return height > 0 ? height * static_cast<int>(N) : height;
    }

    // The N planes as one BinMat of N * height rows: plane p is rows
    // [p*height, (p+1)*height), so plane p starts at word offset p * planeWords()
    // and every plane shares one stride and one allocation.
    BinMat<WordType> stack_;
};

/// @brief A signed N-bit image: N magnitude planes plus one sign plane.
/// @tparam N Number of MAGNITUDE planes, 1 to 7. The container holds N+1.
/// @tparam WordType_ The unsigned integral type pixels are packed into.
///
/// @note Sign-magnitude rather than two's complement (D-3, ARCHITECTURE 4.2),
///       because it makes the LK gradient covariance fall out as population
///       counts over masks, and because ternary is then the N=1 instance of the
///       general form rather than a special case.
/// @note CONVENTION -- both halves matter:
///       - A SET sign bit means NEGATIVE.
///       - MAGNITUDE 0 MEANS ZERO, AND THE SIGN BIT IS IGNORED. "-0" is not a
///         distinct value: a pixel whose magnitude planes are all clear reads as
///         0 whatever its sign bit says. at() implements that reading, and set()
///         canonicalizes -- writing 0 clears the sign bit -- so a value written
///         through this class never leaves a stray sign bit behind. A sign bit
///         set directly through sign() over a zero magnitude is legal and harmless
///         precisely because it cannot change what the pixel means; kernels that
///         write the two planes independently therefore do not need a fix-up pass.
/// @note THIN WRAPPER, no storage of its own: the only member is a
///       QuantMat<N+1, WordType>, and every accessor here is a renaming of one of
///       its planes. There is no second allocation path, no second layout, and
///       nothing to keep in sync -- sizeInWords() and the plane pitch are the
///       underlying container's, byte for byte. planes() exposes it directly.
/// @note Which plane is which: magnitude(i) is plane i, and sign() is plane N,
///       the last one. Magnitude bit 0 is the least significant, as in QuantMat.
/// @note Error policy and per-pixel checking follow QuantMat exactly.
template <size_t N, typename WordType_ = uint32_t>
class SignedQuantMat {
    static_assert(N >= 1 && N <= 7,
                  "N outside supported range: a signed value needs N+1 planes");

public:
    /// The storage word type, exposed for pointer-level access.
    using WordType = WordType_;

    /// @brief The uninterpreted container underneath. N magnitude planes + sign.
    using Container = QuantMat<N + 1, WordType>;

    /// Number of pixels packed into a single word.
    static constexpr size_t WordBits = sizeof(WordType) * 8;

    /// Number of magnitude planes.
    static constexpr size_t MagnitudePlanes = N;

    /// Total planes, magnitude and sign together.
    static constexpr size_t Planes = N + 1;

    /// Index of the sign plane within the underlying container.
    static constexpr size_t SignPlaneIndex = N;

    /// Largest representable magnitude: 2^N - 1. Values run [-MaxMagnitude, +MaxMagnitude].
    static constexpr unsigned MaxMagnitude = (1u << N) - 1u;

    /// @brief Default row-stride alignment: word granularity (D-4). See BinMat.
    static constexpr size_t DefaultRowAlignment = sizeof(WordType);

    // Constructors -- the same three shapes as QuantMat, forwarded to it.

    /// @brief Constructs an empty matrix with no allocated storage.
    SignedQuantMat() = default;

    /// @brief Constructs a zero-filled matrix that owns its storage.
    /// @note Zero-filled means every pixel starts at canonical zero: magnitude 0,
    ///       sign clear.
    SignedQuantMat(int width, int height, size_t rowAlignment = DefaultRowAlignment)
        : planes_(width, height, rowAlignment) {}

    /// @brief Wraps a caller-provided buffer holding all N+1 planes. Allocates
    ///        nothing; see QuantMat's wrapping constructor, including the note on
    ///        the size requirement this signature cannot check -- here it is
    ///        (N+1)-fold, one plane MORE than the type parameter reads as.
    SignedQuantMat(WordType* data, int width, int height, size_t strideWords)
        : planes_(data, width, height, strideWords) {}

    /// @brief Wraps a caller-provided buffer, CHECKING that it holds all N+1
    ///        planes. See QuantMat::wrap().
    /// @param bufferWords Words the caller's buffer actually holds; must be at
    ///        least (N + 1) * height * strideWords.
    /// @note Delegates to the container's factory rather than repeating the
    ///       arithmetic, so the plane count the check uses is the one the storage
    ///       is actually laid out for.
    static SignedQuantMat wrap(WordType* data, size_t bufferWords, int width, int height,
                               size_t strideWords) {
        return SignedQuantMat(Container::wrap(data, bufferWords, width, height, strideWords));
    }

    // Accessors -- the container's, unchanged.

    size_t getWidth() const { return planes_.getWidth(); }
    size_t getHeight() const { return planes_.getHeight(); }
    Size getSize() const { return planes_.getSize(); }
    size_t getAlignedWidth() const { return planes_.getAlignedWidth(); }
    size_t getRowAlignment() const { return planes_.getRowAlignment(); }
    bool empty() const { return planes_.empty(); }
    bool ownsMemory() const { return planes_.ownsMemory(); }
    int rows() const { return planes_.rows(); }
    int cols() const { return planes_.cols(); }
    const WordType* data() const { return planes_.data(); }
    WordType* data() { return planes_.data(); }
    size_t sizeInWords() const { return planes_.sizeInWords(); }
    size_t planeWords() const { return planes_.planeWords(); }

    /// @brief The underlying uninterpreted container -- this object's only member.
    /// @note Handed out so that code which does not care about the sign
    ///       convention (a copy, a footprint measurement, a plane-wise logic op)
    ///       can address the planes without going through the interpretation.
    Container& planes() { return planes_; }
    const Container& planes() const { return planes_; }

    // Sign-magnitude planes

    /// @brief Magnitude plane `i`, plane 0 being the least significant bit.
    /// @param i Magnitude plane index, in [0, N).
    /// @throws std::out_of_range if `i >= N`, in EVERY build. (Or prints and
    ///         aborts where exceptions are disabled.)
    /// @note Checked in release as well as debug, and this is the index in the
    ///       whole family where that matters most. i == N is IN BOUNDS on the
    ///       underlying QuantMat<N+1>, so an unchecked off-by-one would not read
    ///       past anything -- it would quietly hand back the SIGN plane as a
    ///       magnitude plane. No sanitizer can see that, because the address is
    ///       valid storage; the reassembled magnitude is simply wrong, while at()
    ///       and magnitudeAt() go on reporting the right value. The one distinction
    ///       this class exists to make is the one that would be erased.
    BinMatView<WordType> magnitude(size_t i) {
        if (i >= N) {
            BINCV_THROW(std::out_of_range, "SignedQuantMat::magnitude: index out of range");
        }
        return planes_.plane(i);
    }
    BinMatConstView<WordType> magnitude(size_t i) const {
        if (i >= N) {
            BINCV_THROW(std::out_of_range, "SignedQuantMat::magnitude: index out of range");
        }
        return planes_.plane(i);
    }

    /// @brief Magnitude plane `i` as a read-only view, from a NON-const matrix.
    /// @note See QuantMat::constPlane() for why the explicitly-named const
    ///       accessor is needed at a template kernel's call site.
    BinMatConstView<WordType> constMagnitude(size_t i) const { return magnitude(i); }

    /// @brief The sign plane: a set bit means NEGATIVE.
    /// @note Where the magnitude is zero this bit carries no information -- see
    ///       the canonical-zero rule on the class.
    BinMatView<WordType> sign() { return planes_.plane(SignPlaneIndex); }
    BinMatConstView<WordType> sign() const { return planes_.plane(SignPlaneIndex); }

    /// @brief The sign plane as a read-only view, from a NON-const matrix.
    /// @note See QuantMat::constPlane().
    BinMatConstView<WordType> constSign() const { return sign(); }

    // Per-pixel access

    /// @brief Reads the signed value at (row, col).
    /// @return A value in [-MaxMagnitude, +MaxMagnitude].
    /// @note Implements the canonical-zero rule: magnitude 0 returns 0 whatever
    ///       the sign bit holds, so no caller can observe a "-0".
    /// @note Debug-checked, unchecked in release; see QuantMat::at().
    int at(int row, int col) const {
        // One pass over the planes: the container assembles all N+1 of them, and
        // the sign is this value's top bit rather than a second read.
        const unsigned raw = planes_.at(row, col);
        const unsigned magnitudeValue = raw & MaxMagnitude;
        if (magnitudeValue == 0) return 0;  // canonical zero: sign is ignored
        const bool negative = ((raw >> SignPlaneIndex) & 1u) != 0;
        return negative ? -static_cast<int>(magnitudeValue)
                        : static_cast<int>(magnitudeValue);
    }

    /// @brief Reads the magnitude at (row, col), ignoring the sign plane.
    unsigned magnitudeAt(int row, int col) const {
        // The container's own reading assembles all N+1 planes, sign included;
        // masking it to the magnitude planes is cheaper than a second loop.
        return planes_.at(row, col) & MaxMagnitude;
    }

    /// @brief Writes a signed value at (row, col).
    /// @param value Must be in [-MaxMagnitude, +MaxMagnitude].
    /// @note Canonicalizes: writing 0 clears the sign bit as well as the
    ///       magnitude planes, so a zero written here is the canonical zero.
    /// @note Debug-checked, unchecked in release; see QuantMat::at().
    /// @note The magnitude is taken in UNSIGNED arithmetic (impl::signedMagnitude).
    ///       Writing it as `-value` would be undefined behaviour at INT_MIN, and
    ///       the assert above is compiled out in every configuration the project
    ///       verifies, so nothing would stop INT_MIN from reaching it.
    void set(int row, int col, int value) {
        BINCV_ASSERT(value >= -static_cast<int>(MaxMagnitude) &&
                         value <= static_cast<int>(MaxMagnitude),
                     "SignedQuantMat::set: value does not fit in N magnitude bits");
        const unsigned magnitudeValue = impl::signedMagnitude(value);
        const bool negative = value < 0;  // value < 0 implies magnitude != 0
        planes_.set(row, col,
                    magnitudeValue | (negative ? (1u << SignPlaneIndex) : 0u));
    }

private:
    /// @brief Adopts an already-validated container. Used by wrap().
    explicit SignedQuantMat(Container&& container) : planes_(std::move(container)) {}

    // The N+1 planes. This is the whole object: SignedQuantMat adds a reading of
    // the planes, not a byte of storage.
    Container planes_;
};

/// @brief A ternary image: values {-1, 0, +1}, one magnitude plane and one sign
///        plane, two bits per pixel.
/// @note The N=1 instance of the general signed form, not a special case
///       (ARCHITECTURE 4.2). It is what the binarized spatial derivative at
///       pyramid level 0 produces (7.4) and what the LK gradient covariance
///       consumes as masked population counts (7.5).
/// @note Class template argument deduction does not see through an alias template
///       before C++20, so spell the argument list: TernaryMat<> or
///       TernaryMat<uint32_t>.
template <typename WordType = uint32_t>
using TernaryMat = SignedQuantMat<1, WordType>;

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
