#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <stdexcept>
#include <functional>

namespace bincv {

/// @brief A binary matrix class that stores binary data in a packed format using OpenCV's cv::Mat.
/// @tparam WordSize The size of the word (in bits) used for packing binary pixels..
///         Currently supported values are 8, 16, 32, and 64.
/// @note Currently the underlying storage is OpenCV's cv::Mat, and it seems that 32-bit words are optimized more.
template <size_t WordSize = 32>
class BinMat {
    // Ensure WordSize is one of the supported sizes
    static_assert(
        WordSize == 8 || WordSize == 16 || WordSize == 32 || WordSize == 64,
        "WordSize must be one of: 8, 16, 32, or 64 bits");

    // Deduce the correct word type based on WordSize
    using WordType = typename std::conditional<
        WordSize == 8, uint8_t,
        typename std::conditional<
            WordSize == 16, uint16_t,
            typename std::conditional<
                WordSize == 32, uint32_t,
                typename std::conditional<WordSize == 64, uint64_t, void>::type
            >::type
        >::type
    >::type;

public:
    // Constructors
    BinMat();

    // @param width Width of the matrix in pixels
    // @param height Height of the matrix in pixels
    // @param memAlignment Number of bytes to align the row stride to.
    //        Must be a positive power of 2, default is 32 bytes.
    BinMat(size_t width, size_t height, size_t rowAlignment = 32);

    // Ensure rowAlignment is a positive power of 2
    static_assert(
        (WordSize & (WordSize - 1)) == 0 && WordSize > 0,
        "WordSize must be a positive power of 2");

    // Accessors
    size_t width() const { return width; }
    size_t height() const { return height; }
    size_t rowAlignment() const { return rowAlignment; }
    const cv::Mat& getCVMat() const { return mat; }

    // @brief Converts a cv::Mat to a BinMat.
    // @param mat The input cv::Mat, must be of type CV_8UC (for now)
    // @note Any nonzero pixel in the input cv::Mat will be set to 1 in the BinMat.
    // @todo: Support other types like CV_32FC1, etc.
    void fromCVMat(const cv::Mat& mat);

    // @brief Converts the BinMat to a cv::Mat where each pixel is either 0 or 1.
    // @param mat The output cv::Mat, will be of type CV_8UC
    // @note Each pixel will maintain its original value
    // @todo Support other types like CV_32FC1, etc.
    void toCVMat(cv::Mat& mat) const;

    // @brief Converts the BinMat to a cv::Mat with normalized values.
    // @param mat The output cv::Mat, will be of type CV_8UC
    // @note Each pixel will be set to 255 if it is 1 in the BinMat, otherwise it will be set to 0.
    // @todo Support other types like CV_32FC1, etc.
    void toCVMatNormalized(cv::Mat& mat) const;

    // @todo add multi-element / slicing access that indexes by BinMat range

    // @brief Gets the value of a single element at (row, col). Not a reference.
    // @todo should the arguments be of type int?
    inline bool at(int row, int col) const;

    // @brief Sets a single element at (row, col) to value.
    inline void set(int row, int col, bool value);

    // Fast row-level access to packed bytes via pointers
    // @brief Needed for efficient access to values when being more performant
    // @note Users need to be careful as they expose the internal storage directly
    //   can need to deal with pixel alignment and byte packing.

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
    void resize(int new_width, int new_height);

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

    // @brief Iterates over all non-zero pixels in the BinMat and applies the callback function.
    template <typename Func>
    void forEachNonZero(Func callback) const;

    // @brief Prints the binary values as a human-readable matrix.
    // @note Output uses 0/1 per pixel, with rows and columns corresponding to image layout.
    // @note Prints in row-major order.
    void printMatrix() const;

    // @brief Overload stream operator to print BinMat
    // @note Prints in row-major order.
    friend std::ostream& operator<<(std::ostream& os, const BinMat& binmat);

    // @brief Prints the packed bytes of internal storage row by row.
    // @param hex If true, prints each byte in hex. Otherwise, prints decimal.
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

    // Dimensions are in number of pixels
    size_t width;
    size_t height;
    size_t rowAlignment;  // number of bytes to align the size of each row to for performance
    size_t alignedWidth; // number of words in a row in internal storage, aligned to rowAlignment bytes

    // @todo Consider defining cvType used in ::impl functions as a static constant since it is known at compile time

    // @note Each row is padded such that its size aligns with the chosen memory alignment.
    //       Thus, width stores the number of pixels in each row, while rowBytes stores the actual
    //       number of bytes used for each row in the internal storage, which is aligned to rowAlignment.

    // Internal storage: row-wise packed 1-bit pixels, where each row is a multiple of WordType size.
    // Dimension of the matrix is height by the row-aligned width
    cv::Mat mat;
};

} // namespace bincv

#include "impl/binMat_impl.hpp"