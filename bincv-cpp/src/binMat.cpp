#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include "bincv-cpp/binMat.hpp"

namespace bincv {

// Helper function to calculate byte index
inline int byte_index(int x) {
    return x >> 3; // x / 8
}

// Helper function to calculate bit mask
inline uint8_t bit_mask(int x) {
    return 1 << (7 - (x & 7)); // 7 - (x % 8)
}

BinMat::BinMat()
    : width_(0), height_(0), mat_() {}

BinMat::BinMat(int width, int height)
    : width_(width), height_(height) {
    stride_bytes_ = ((width + 7) / 8 + 31) & ~31; // align to 32 bytes
    mat_ = cv::Mat::zeros(height_, stride_bytes_, CV_8UC1);
}

void BinMat::fromCVMat(const cv::Mat& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input cv::Mat is empty");
    }
    if (input.type() != CV_8UC1) {
        throw std::invalid_argument("Input cv::Mat must be of type CV_8UC1");
    }

    width_ = input.cols;
    height_ = input.rows;
    mat_ = cv::Mat::zeros(height_, stride_bytes_, CV_8UC1);

    for (int y = 0; y < height_; ++y) {
        const uint8_t* row_in = input.ptr<uint8_t>(y);
        uint8_t* row_out = mat_.ptr<uint8_t>(y);
        for (int x = 0; x < width_; ++x) {
            if (row_in[x]) {
                row_out[byte_index(x)] |= bit_mask(x);
            }
        }
    }
}

// @todo: could an approach using OpenCV's resize and scaling functions be more efficient?
//              need to think more about how to do this efficiently
void BinMat::toCVMat(cv::Mat& output) const {
    output = cv::Mat::zeros(height_, width_, CV_8UC1);
    for (int y = 0; y < height_; ++y) {
        const uint8_t* row_in = mat_.ptr<uint8_t>(y);
        uint8_t* row_out = output.ptr<uint8_t>(y);
        for (int x = 0; x < width_; ++x) {
            row_out[x] = (row_in[byte_index(x)] & bit_mask(x)) ? 1 : 0;
        }
    }
}

void BinMat::toCVMatNormalized(cv::Mat& output) const {
    output = cv::Mat::zeros(height_, width_, CV_8UC1);
    for (int y = 0; y < height_; ++y) {
        const uint8_t* row_in = mat_.ptr<uint8_t>(y);
        uint8_t* row_out = output.ptr<uint8_t>(y);
        for (int x = 0; x < width_; ++x) {
            row_out[x] = (row_in[byte_index(x)] & bit_mask(x)) ? 255 : 0;
        }
    }
}

} // namespace bincv
