#pragma once

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <stdexcept>
#include <vector>
#include "core/types.hpp"

// OpenCV integration is optional and provided behind a compile-time switch.
// The core library never requires OpenCV; see CMakeLists.txt.
#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#endif

namespace bincv {

/// @brief A binary matrix storing one bit per pixel, packed into words.
/// @tparam WordType The unsigned integral type used to pack pixels.
///         Supported: uint8_t, uint16_t, uint32_t (default), uint64_t.
/// @note Parameterizing on the storage word *type* (rather than a bit count)
///       follows boost::dynamic_bitset<Block> and cv::Mat_<T>. The bit width is
///       derived as WordBits, so the type never has to be recovered from a number.
/// @note The underlying storage is std::vector for embedded compatibility.
///       32-bit words typically provide optimal performance on most platforms.
template <typename WordType_>
class BinMat {
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

    // Constructors

    /// @brief Constructs an empty matrix with no allocated storage.
    BinMat();

    /// @brief Constructs a zero-filled matrix.
    /// @param width Width of the matrix in pixels
    /// @param height Height of the matrix in pixels
    /// @param rowAlignment Number of bytes to align each row's stride to.
    ///        Must be a positive power of two; default is 32 bytes (AVX2 / cache friendly).
    /// @throws std::invalid_argument if dimensions are negative or alignment is invalid.
    BinMat(int width, int height, size_t rowAlignment = 32);

    // Accessors
    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }
    size_t getRowAlignment() const { return rowAlignment; }
    Size getSize() const { return Size(static_cast<int>(width), static_cast<int>(height)); }

    /// @brief Number of words allocated per row, including alignment padding.
    size_t getAlignedWidth() const { return alignedWidth; }

    /// @brief True if the matrix has no pixels.
    bool empty() const { return width == 0 || height == 0; }

    // OpenCV-compatible aliases
    int rows() const { return static_cast<int>(height); }
    int cols() const { return static_cast<int>(width); }

    /// @brief Raw access to the packed storage, for bulk/SIMD operations.
    const WordType* data() const { return storage.data(); }
    WordType* data() { return storage.data(); }

    /// @brief Total number of words in the backing store (height * alignedWidth).
    size_t sizeInWords() const { return storage.size(); }

#ifdef BINCV_WITH_OPENCV
    // OpenCV interoperability (only available when BINCV_WITH_OPENCV is defined)

    // @brief Converts a cv::Mat to a BinMat.
    // @param mat The input cv::Mat, must be of type CV_8UC1 (for now)
    // @note Any nonzero pixel in the input cv::Mat will be set to 1 in the BinMat.
    // @todo: Support other types like CV_32FC1, etc.
    void fromCVMat(const cv::Mat& mat);

    // @brief Converts the BinMat to a cv::Mat where each pixel is either 0 or 1.
    // @param mat The output cv::Mat, will be of type CV_8UC1
    // @note Each pixel will maintain its original value
    // @todo Support other types like CV_32FC1, etc.
    void toCVMat(cv::Mat& mat) const;

    // @brief Converts the BinMat to a cv::Mat with normalized values.
    // @param mat The output cv::Mat, will be of type CV_8UC1
    // @note Each pixel will be set to 255 if it is 1 in the BinMat, otherwise 0.
    // @todo Support other types like CV_32FC1, etc.
    void toCVMatNormalized(cv::Mat& mat) const;
#endif // BINCV_WITH_OPENCV

    // @todo add multi-element / slicing access that indexes by BinMat range

    // @brief Gets the value of a single element at (row, col). Not a reference.
    bool at(int row, int col) const;

    // @brief Sets a single element at (row, col) to value.
    void set(int row, int col, bool value);

    // Fast row-level access to packed words via pointers
    // @brief Needed for efficient access to values when being more performant
    // @note Users need to be careful as they expose the internal storage directly
    //   and need to deal with pixel alignment and word packing.

    // @brief Gets a const pointer to the start of the specified row.
    // @note Out of bounds access is not protected and may lead to undefined behavior.
    //    This is intended for performance-sensitive code where bounds are checked externally.
    const WordType* ptr(int row) const;

    // @brief Gets a pointer to the start of the specified row.
    // @note Out of bounds access is not protected and may lead to undefined behavior.
    //    This is intended for performance-sensitive code where bounds are checked externally.
    WordType* ptr(int row);

    // @brief Resizes the BinMat to given dimensions.
    // @note For smaller sizes, it will clear the existing data.
    // @note For larger sizes, it will zero-fill the new area, appending rows
    //    and columns as needed at larger indices.
    // @note To extend size at specific dimensions, use pad() instead.
    void resize(int newWidth, int newHeight);

    // @brief Pads the BinMat with given value at sides with non-zero padding.
    // @note Zero-padding unless value is specified as true.
    // @todo Add support for "replicate" padding modes
    void pad(int top, int bottom, int left, int right, bool value = false);

    // @brief Returns a transposed version of the BinMat.
    // @note The original BinMat remains unchanged.
    BinMat transposed() const;

    // @brief Transposes the BinMat in-place.
    // @note The BinMat dimensions and data are updated.
    void transpose();

    // @brief Iterates over all non-zero pixels, invoking callback(row, col).
    template <typename Func>
    void forEachNonZero(Func callback) const;

    // @brief Prints the binary values as a human-readable matrix.
    // @note Output uses 0/1 per pixel, with rows and columns corresponding to image layout.
    // @note Prints in row-major order.
    void printMatrix() const;

    // @brief Prints the packed words of internal storage row by row.
    // @param hex If true, prints each word in hex. Otherwise, prints decimal.
    // @note Prints in row-major order.
    void printInternalData(bool hex = false) const;

    // @brief Fills the entire BinMat with the given binary value.
    void fill(bool value);

    // @brief Counts the number of non-zero (set) pixels in the matrix.
    // @return The total number of 1s in the matrix.
    int countNonZero() const;

    // @brief Returns the sparsity ratio (fraction of zero pixels).
    // @return A float in [0.0, 1.0] representing how sparse the matrix is.
    // @note Empty matrices have undefined sparsity and will throw an exception.
    float sparsity() const;

    // @todo: add functions or representations for sparse formats e.g. CSR/CSC
    // @todo: Consider adding "channels" support for multi-channel binary images

private:
    // @brief Zeroes the padding bits beyond `width` in every row.
    // @note Bulk word-wise operations (fill, and future bitwise/popcount ops) write
    //       whole words, which can set bits past the end of a row. Those bits must stay
    //       zero or countNonZero and friends would over-count once they go word-wise.
    void clearTrailingBits();

    // Dimensions are in number of pixels
    size_t width;
    size_t height;
    size_t rowAlignment;  // bytes each row's stride is aligned to, for performance
    size_t alignedWidth;  // words per row in internal storage, aligned to rowAlignment bytes

    // @note Each row is padded such that its stride aligns with the chosen memory alignment.
    //       width stores the number of pixels in each row, while alignedWidth stores the
    //       actual number of words used for each row in the internal storage.
    // @todo rowAlignment currently aligns only the row *stride*. Aligning the base pointer
    //       too (for aligned SIMD loads) requires a custom aligned allocator.

    // Internal storage: row-wise packed 1-bit pixels, height * alignedWidth words.
    // Using std::vector for embedded compatibility (no OpenCV dependency in core).
    std::vector<WordType> storage;
};

} // namespace bincv

#include "impl/binMat_impl.hpp"
