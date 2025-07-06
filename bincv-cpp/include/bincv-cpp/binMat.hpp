#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <iostream>

namespace bincv {

namespace detail {

// Helper function to calculate byte index
// Computes which byte in the row contains the pixel at column x.
static inline int byte_index(int x) {
    return x >> 3; // x / 8
}

// Helper function to calculate bit mask
// Creates a bitmask where the bit corresponding to column x is set to 1, and all others are 0.
// @note Indexing starts with LSB->MSB, so bit 0 is the least significant bit.
static inline uint8_t bit_mask(int x) {
    return 1 << (x & 7);
}

} // namespace detail

class BinMat {
public:
    // Constructors
    BinMat();
    BinMat(int width, int height);

    // Accessors
    int width() const { return width_; }
    int height() const { return height_; }
    int strideBytes() const { return stride_bytes_; }
    const cv::Mat& getCVMat() const { return mat_; }

    // fromCVMat
    // 
    // @brief Converts a cv::Mat to a BinMat.
    // @param mat The input cv::Mat, must be of type CV_8UC (for now)
    // @note Any nonzero pixel in the input cv::Mat will be set to 1 in the BinMat.
    void fromCVMat(const cv::Mat& mat);

    // toCVMat
    // 
    // @brief Converts the BinMat to a cv::Mat.
    // @param mat The output cv::Mat, will be of type CV_8UC
    // @note Each pixel will maintain its original value
    void toCVMat(cv::Mat& mat) const;

    // toCVMatNormalized
    //
    // @brief Converts the BinMat to a cv::Mat with normalized values.
    // @param mat The output cv::Mat, will be of type CV_8UC
    // @note Each pixel will be set to 255 if it is 1 in the BinMat, otherwise it will be set to 0.
    void toCVMatNormalized(cv::Mat& mat) const;

    // @todo add multi-element / slicing access that indexes by BinMat range

    // at
    //
    // @brief Gets a single element at (row, col). Not a reference.
    inline bool at(int row, int col) const {
        if (row < 0 || row >= height_ || col < 0 || col >= width_) {
            throw std::out_of_range("BinMat::at: index out of range");
        }
        const uint8_t* row_ptr = mat_.ptr<uint8_t>(row);
        return (row_ptr[detail::byte_index(col)] & detail::bit_mask(col)) != 0;
    }

    // set
    //
    // @brief Sets a single element at (row, col) to value.
    inline void set(int row, int col, bool value) {
        if (row < 0 || row >= height_ || col < 0 || col >= width_) {
            throw std::out_of_range("BinMat::set: index out of range");
        }
        uint8_t* row_ptr = mat_.ptr<uint8_t>(row);
        uint8_t& byte = row_ptr[detail::byte_index(col)];
        uint8_t mask = detail::bit_mask(col);

        if (value) {
            byte |= mask;   // set bit
        } else {
            byte &= ~mask;  // clear bit
        }
    }

    // Fast row-level access to packed bytes via pointers
    // @brief Needed for efficient access to values when being more performant
    // @note Users need to be careful as they expose the internal storage directly
    //   can need to deal with pixel alignment and byte packing.

    // ptr
    //
    // @brief Gets a const pointer to the start of the specified row.
    // @note Out of bounds access is not protected and may lead to undefined behavior.
    //    This is intended for performance-sensitive code where bounds are checked externally.
    const uint8_t* ptr(int row) const;

    // ptr
    //
    // @brief Gets a pointer to the start of the specified row.
    // @note Out of bounds access is not protected and may lead to undefined behavior.
    //    This is intended for performance-sensitive code where bounds are checked externally.
    uint8_t* ptr(int row);

    // resize
    //
    // @brief Resizes the BinMat to given dimensions.
    // @note For smaller sizes, it will clear the existing data.
    // @note For larger sizes, it will zero-fill the new area, appending rows
    //    and columns as needed at larger indices.
    // @note To extend size at specific dimensions, use pad() instead.
    void resize(int new_width, int new_height);

    // pad
    //
    // @brief Pads the BinMat with given value at sides with non-zero padding.
    // @note Zero-padding unless value is specified as true.
    void pad(int top, int bottom, int left, int right, bool value = false);

    // transposed
    //
    // @brief Returns a transposed version of the BinMat.
    // @note The original BinMat remains unchanged.
    BinMat transposed() const;

    // transpose
    //
    // @brief Transposes the BinMat in-place.
    // @note The BinMat dimensions and data are updated.
    void transpose();

    // forEachNonZero
    //
    // @brief Iterates over all non-zero pixels in the BinMat and applies the callback function.
    template <typename Func>
    void forEachNonZero(Func callback) const {
        if (width_ == 0 || height_ == 0) {
            throw std::runtime_error("BinMat is empty, cannot iterate over non-zero pixels");
        }
        for (int y = 0; y < height_; ++y) {
            const uint8_t* row = mat_.ptr<uint8_t>(y);
            for (int x = 0; x < width_; ++x) {
                if (row[detail::byte_index(x)] & detail::bit_mask(x)) {
                    callback(y, x);
                }
            }
        }
    }

    // printMatrix
    //
    // @brief Prints the binary values as a human-readable matrix.
    // @note Output uses 0/1 per pixel, with rows and columns corresponding to image layout.
    // @note Prints in row-major order.
    void printMatrix() const;

    // operator<<
    // 
    // @brief Overload stream operator to print BinMat
    // @note Prints in row-major order.
    friend std::ostream& operator<<(std::ostream& os, const BinMat& binmat);

    // printInternalData
    //
    // @brief Prints the packed bytes of internal storage row by row.
    // @param hex If true, prints each byte in hex. Otherwise, prints decimal.
    // @note Prints in row-major order.
    void printInternalData(bool hex = false) const;

    // @todo: add a function to determine the number of non-zero pixels
    // @todo: add a function to determine the sparsity of the matrix
    // @todo: add a function fill() to fill with a specific value
    // @todo: add functions or representations for sparse formats e.g. CSR/CSC

private:

    // Dimensions are in pixels
    int width_;
    int height_;
    int stride_bytes_;  // number of bytes per row (aligned for SIMD/CUDA)

    // Internal storage: row-wise packed 1-bit pixels, 8 per byte
    // size: height_ × stride_bytes_, type = CV_8UC1
    cv::Mat mat_;
};

} // namespace bincv