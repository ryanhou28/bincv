#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include "bincv-cpp/binMat.hpp"

namespace bincv {

// @todo: make the stride alignment to 32 bytes an inline function

inline int internal_stride_bytes(int width) {
    // Align to 32 bytes, don't calculate for 0 width
    return (width == 0) ? 0 : ((width + 7) / 8 + 31) & ~31;
}

BinMat::BinMat()
    : width_(0), height_(0), mat_() {}

BinMat::BinMat(int width, int height)
    : width_(width), height_(height) {
    if (width < 0 || height < 0)
        throw std::invalid_argument("BinMat dimensions must be non-negative");

    stride_bytes_ = internal_stride_bytes(width);
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
                row_out[bincv::detail::byte_index(x)] |= bincv::detail::bit_mask(x);
            }
        }
    }
}

// @todo: could an approach using OpenCV's resize and scaling functions be more efficient?
//              need to think more about how to do this efficiently
void BinMat::toCVMat(cv::Mat& output) const {
    if (width_ == 0 || height_ == 0) {
        // Separate case for empty matrix needed to avoid accessing uninitialized memory with ptr()
        output = cv::Mat();
        return;
    }
    output = cv::Mat::zeros(height_, width_, CV_8UC1);
    for (int y = 0; y < height_; ++y) {
        const uint8_t* row_in = mat_.ptr<uint8_t>(y);
        uint8_t* row_out = output.ptr<uint8_t>(y);
        for (int x = 0; x < width_; ++x) {
            row_out[x] = (row_in[bincv::detail::byte_index(x)] & bincv::detail::bit_mask(x)) ? 1 : 0;
        }
    }
}

void BinMat::toCVMatNormalized(cv::Mat& output) const {
    if (width_ == 0 || height_ == 0) {
        // Separate case for empty matrix needed to avoid accessing uninitialized memory with ptr()
        output = cv::Mat();
        return;
    }
    output = cv::Mat::zeros(height_, width_, CV_8UC1);
    for (int y = 0; y < height_; ++y) {
        const uint8_t* row_in = mat_.ptr<uint8_t>(y);
        uint8_t* row_out = output.ptr<uint8_t>(y);
        for (int x = 0; x < width_; ++x) {
            row_out[x] = (row_in[bincv::detail::byte_index(x)] & bincv::detail::bit_mask(x)) ? 255 : 0;
        }
    }
}

const uint8_t* BinMat::ptr(int row) const {
    return mat_.ptr<uint8_t>(row);
}

uint8_t* BinMat::ptr(int row) {
    return mat_.ptr<uint8_t>(row);
}

// @todo: This could potentially be optimized
void BinMat::resize(int new_width, int new_height) {
    if (width_ < 0 || height_ < 0)
        throw std::invalid_argument("BinMat dimensions must be non-negative");

    int new_stride_bytes = internal_stride_bytes(new_width);

    // Create new zero-initialized matrix
    cv::Mat new_mat = cv::Mat::zeros(new_height, new_stride_bytes, CV_8UC1);

    // Determine region to copy from old matrix
    int min_width = std::min(width_, new_width);
    int min_height = std::min(height_, new_height);

    if (min_width == 0 || min_height == 0) {
        // If either dimension is zero, we can skip copying
        width_ = new_width;
        height_ = new_height;
        stride_bytes_ = new_stride_bytes;
        mat_ = std::move(new_mat);
        return;
    }

    for (int y = 0; y < min_height; ++y) {
        const uint8_t* old_row = mat_.ptr<uint8_t>(y);
        uint8_t* new_row = new_mat.ptr<uint8_t>(y);

        for (int x = 0; x < min_width; ++x) {
            if (old_row[bincv::detail::byte_index(x)] & bincv::detail::bit_mask(x)) {
                new_row[bincv::detail::byte_index(x)] |= bincv::detail::bit_mask(x);
            }
        }
    }

    // Update internal state
    width_ = new_width;
    height_ = new_height;
    stride_bytes_ = new_stride_bytes;
    mat_ = std::move(new_mat);
}

// @todo: This could potentially be optimized further
void BinMat::pad(int top, int bottom, int left, int right, bool value) {
    int new_width = width_ + left + right;
    int new_height = height_ + top + bottom;
    int new_stride_bytes = internal_stride_bytes(new_width);

    cv::Mat new_mat = cv::Mat::zeros(new_height, new_stride_bytes, CV_8UC1);
    
    uint8_t fill_mask = value ? 0xFF : 0x00;

    // Fill left/right columns of new rows (if value == 1)
    if (value && (left > 0 || right > 0)) {
        for (int y = 0; y < new_height; ++y) {
            uint8_t* row = new_mat.ptr<uint8_t>(y);
            if (left > 0) {
                std::fill(row, row + bincv::detail::byte_index(left), fill_mask);
            }
            if (right > 0) {
                int start = bincv::detail::byte_index(new_width - right);
                std::fill(row + start, row + new_stride_bytes, fill_mask);
            }
        }
    }

    // If existing data is empty, we can skip copying
    if (width_ == 0 || height_ == 0) {
        width_ = new_width;
        height_ = new_height;
        stride_bytes_ = new_stride_bytes;
        mat_ = std::move(new_mat);
        return;
    }

    // Copy existing data into centered position
    for (int y = 0; y < height_; ++y) {
        const uint8_t* src_row = mat_.ptr<uint8_t>(y);
        uint8_t* dst_row = new_mat.ptr<uint8_t>(y + top);
        int dst_bit_offset = left;
        int dst_byte_offset = bincv::detail::byte_index(dst_bit_offset);

        for (int x = 0; x < width_; ++x) {
            if ((src_row[bincv::detail::byte_index(x)] & bincv::detail::bit_mask(x)) != 0) {
                int dst_x = x + left;
                dst_row[bincv::detail::byte_index(dst_x)] |= bincv::detail::bit_mask(dst_x);
            }
        }
    }

    width_ = new_width;
    height_ = new_height;
    stride_bytes_ = new_stride_bytes;
    mat_ = std::move(new_mat);
}

BinMat BinMat::transposed() const {
    // Skip if empty matrix
    if (width_ == 0 || height_ == 0) {
        return BinMat();
    }

    BinMat result(height_, width_);
    for (int y = 0; y < height_; ++y) {
        const uint8_t* row = mat_.ptr<uint8_t>(y);
        for (int x = 0; x < width_; ++x) {
            if (row[bincv::detail::byte_index(x)] & bincv::detail::bit_mask(x)) {
                result.set(x, y, true);
            }
        }
    }
    return result;
}

// @todo: this could potentially be optimized further to avoid copying data in the transposed() implementation
void BinMat::transpose() {
    *this = this->transposed();
}


void BinMat::printMatrix() const {
    if (width_ == 0 || height_ == 0) {
        return;
    }

    for (int y = 0; y < height_; ++y) {
        const uint8_t* row = mat_.ptr<uint8_t>(y);
        for (int x = 0; x < width_; ++x) {
            std::cout << ((row[bincv::detail::byte_index(x)] & bincv::detail::bit_mask(x)) ? '1' : '0');
        }
        std::cout << '\n';
    }
}

// @todo: consider std::ostringstream or wrapping std::ostream& for reusable logging.
std::ostream& operator<<(std::ostream& os, const BinMat& binmat) {
    if (binmat.width() == 0 || binmat.height() == 0) {
        return os; // Nothing to print for empty matrix
    }

    for (int y = 0; y < binmat.height(); ++y) {
        const uint8_t* row = binmat.ptr(y);
        for (int x = 0; x < binmat.width(); ++x) {
            os << ((row[bincv::detail::byte_index(x)] & bincv::detail::bit_mask(x)) ? '1' : '0');
        }
        os << '\n';
    }
    return os;
}

void BinMat::printInternalData(bool hex) const {
    if (width_ == 0 || height_ == 0) {
        return; // Nothing to print for empty matrix
    }

    for (int y = 0; y < height_; ++y) {
        const uint8_t* row = mat_.ptr<uint8_t>(y);
        for (int b = 0; b < stride_bytes_; ++b) {
            if (hex)
                std::cout << std::hex << static_cast<int>(row[b]) << " ";
            else
                std::cout << std::dec << static_cast<int>(row[b]) << " ";
        }
        std::cout << '\n';
    }
    std::cout << std::dec;  // Reset output stream to decimal
}

} // namespace bincv
