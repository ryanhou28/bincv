#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <stdexcept>
#include <functional>

namespace bincv {

namespace detail {

// Helper function to calculate word index
// Computes which word in the row contains the pixel at column x.
template <typename WordType>
static inline int word_index(int x) {
    static_assert(std::is_unsigned<WordType>::value, "WordType must be unsigned");
    return x / (8 * sizeof(WordType));
}

// Helper function to create a bit mask with a 1 at the bit corresponding to column x.
// Creates a bitmask where the bit corresponding to column x is set to 1, and all others are 0.
// @note Indexing starts with LSB->MSB, so bit 0 is the least significant bit.
template <typename WordType>
static inline WordType bit_mask(int x) {
    return WordType(1) << (x % (8 * sizeof(WordType)));
}

} // namespace detail

template <typename WordType = uint64_t>
class BinMat {
    static_assert(std::is_unsigned<WordType>::value, "WordType must be an unsigned integral type");

public:
    // Constructors
    BinMat();
    BinMat(int width, int height);

    // Accessors
    int width() const { return width_; }
    int height() const { return height_; }
    int strideBytes() const { return stride_bytes_; }
    const cv::Mat& getCVMat() const { return mat_; }

    // @brief Converts a cv::Mat to a BinMat.
    // @param mat The input cv::Mat, must be of type CV_8UC (for now)
    // @note Any nonzero pixel in the input cv::Mat will be set to 1 in the BinMat.
    // @todo: Support other types like CV_32FC1, etc.
    void fromCVMat(const cv::Mat& mat);

    // @brief Converts the BinMat to a cv::Mat.
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

    // @brief Gets a single element at (row, col). Not a reference.
    inline bool at(int row, int col) const {
        if (row < 0 || row >= height_ || col < 0 || col >= width_) {
            throw std::out_of_range("BinMat::at: index out of range");
        }
        const WordType* row_ptr = mat_.ptr<WordType>(row);
        return (row_ptr[detail::word_index<WordType>(col)] & detail::bit_mask<WordType>(col)) != 0;
    }

    // @brief Sets a single element at (row, col) to value.
    inline void set(int row, int col, bool value) {
        if (row < 0 || row >= height_ || col < 0 || col >= width_) {
            throw std::out_of_range("BinMat::set: index out of range");
        }
        WordType* row_ptr = mat_.ptr<WordType>(row);
        WordType& word = row_ptr[detail::word_index<WordType>(col)];
        WordType mask = detail::bit_mask<WordType>(col);

        if (value) {
            word |= mask;   // set bit
        } else {
            word &= ~mask;  // clear bit
        }
    }

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
    void pad(int top, int bottom, int left, int right, bool value = false);

    // @brief Returns a transposed version of the BinMat.
    // @note The original BinMat remains unchanged.
    BinMat transposed() const;

    // @brief Transposes the BinMat in-place.
    // @note The BinMat dimensions and data are updated.
    void transpose();

    // @brief Iterates over all non-zero pixels in the BinMat and applies the callback function.
    template <typename Func>
    void forEachNonZero(Func callback) const {
        if (width_ == 0 || height_ == 0) {
            throw std::runtime_error("BinMat is empty, cannot iterate over non-zero pixels");
        }
        for (int y = 0; y < height_; ++y) {
            const WordType* row = mat_.ptr<WordType>(y);
            for (int x = 0; x < width_; ++x) {
                if (row[detail::word_index<WordType>(x)] & detail::bit_mask<WordType>(x)) {
                    callback(y, x);
                }
            }
        }
    }

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

    // Dimensions are in pixels
    int width_;
    int height_;
    int stride_bytes_;  // number of bytes per row (aligned for SIMD/CUDA)

    // Internal storage: row-wise packed 1-bit pixels, where each row is a multiple of WordType size.
    // size: height_ × stride_bytes_
    cv::Mat mat_;
};

} // namespace bincv