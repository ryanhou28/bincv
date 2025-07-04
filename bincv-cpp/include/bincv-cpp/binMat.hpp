#pragma once

#include <opencv2/core.hpp>
#include <cstdint>

namespace bincv {

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
    // @brief Converts the BinMat to a cv::Mat with normalized values.
    // @param mat The output cv::Mat, will be of type CV_8UC
    // @note Each pixel will be set to 255 if it is 1 in the BinMat, otherwise it will be set to 0.
    void toCVMatNormalized(cv::Mat& mat) const;

    // @todo: add overload operators to index into the BinMat at given coordinates

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