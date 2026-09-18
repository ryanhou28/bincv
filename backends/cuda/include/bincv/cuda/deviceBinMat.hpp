#pragma once

/// @file deviceBinMat.hpp
/// @brief The owning device containers: a bit-packed matrix and a wide image,
/// both in GPU memory, both value-semantic.
///
/// The container is where allocation lives, exactly as on the host: kernels
/// take the views, and no kernel in this backend allocates. Copy means deep
/// copy (device-to-device); sharing is a view. A DeviceBinMat is zero-filled on
/// allocation, which is what establishes the padding-bit invariant the format
/// shares with the host library -- every kernel here preserves it, and upload
/// copies only pixel bytes, so it never goes stale.
///
/// The default row stride is TIGHT (ceil(width / 32) words), the same choice
/// the host container records: memory wins ties. Wider alignment is opt-in per
/// object through rowAlignmentBytes, and whether the 128-byte transaction
/// boundary buys anything on this backend's kernels is a measured question in
/// the benchmark, not an assumption baked in here.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief A binary matrix in device memory: one bit per pixel, uint32 words,
/// the host format's bytes on the other side of the bus.
class DeviceBinMat {
public:
    /// @brief Allocates a zero-filled width x height matrix on the device.
    /// @param rowAlignmentBytes Power of two, multiple of 4. The stride each
    /// row is rounded up to; 4 (tight, the default) unless measured otherwise.
    DeviceBinMat(int width, int height, size_t rowAlignmentBytes = 4) {
        if (width < 0 || height < 0)
            BINCV_THROW(std::invalid_argument, "DeviceBinMat: negative dimensions");
        if (rowAlignmentBytes < 4 || (rowAlignmentBytes & (rowAlignmentBytes - 1)) != 0)
            BINCV_THROW(std::invalid_argument,
                        "DeviceBinMat: rowAlignmentBytes must be a power of two >= 4");
        width_ = static_cast<size_t>(width);
        height_ = static_cast<size_t>(height);
        const size_t alignWords = rowAlignmentBytes / 4;
        stride_ = ((rowWords(width_) + alignWords - 1) / alignWords) * alignWords;
        allocate();
    }

    DeviceBinMat() = default;

    /// @brief Deep-copies, always: a device-to-device copy of the whole plane.
    DeviceBinMat(const DeviceBinMat& other)
        : width_(other.width_), height_(other.height_), stride_(other.stride_) {
        allocate();
        if (ptr_ != nullptr) {
            BINCV_CUDA_CHECK(cudaMemcpy(ptr_, other.ptr_,
                                        height_ * stride_ * sizeof(uint32_t),
                                        cudaMemcpyDeviceToDevice));
        }
    }

    DeviceBinMat& operator=(const DeviceBinMat& other) {
        if (this == &other) return *this;
        DeviceBinMat fresh(other);
        swap(fresh);
        return *this;
    }

    DeviceBinMat(DeviceBinMat&& other) noexcept { swap(other); }
    DeviceBinMat& operator=(DeviceBinMat&& other) noexcept {
        if (this != &other) {
            releaseAndClear();
            swap(other);
        }
        return *this;
    }

    ~DeviceBinMat() { releaseAndClear(); }

    size_t getWidth() const { return width_; }
    size_t getHeight() const { return height_; }
    /// @brief Row stride in words, as the host container reports it.
    size_t getAlignedWidth() const { return stride_; }
    bool empty() const { return width_ == 0 || height_ == 0; }

    DeviceBinMatView view() { return DeviceBinMatView{ptr_, width_, height_, stride_}; }
    DeviceBinMatConstView constView() const {
        return DeviceBinMatConstView{ptr_, width_, height_, stride_};
    }

private:
    void allocate() {
        const size_t words = height_ * stride_;
        if (words == 0) return;
        void* p = nullptr;
        BINCV_CUDA_CHECK(cudaMalloc(&p, words * sizeof(uint32_t)));
        ptr_ = static_cast<uint32_t*>(p);
        // Zero-filled like the host container, and for the same reason: the
        // padding bits must START clear or word-wise reductions over-count.
        BINCV_CUDA_CHECK(cudaMemset(ptr_, 0, words * sizeof(uint32_t)));
    }

    void releaseAndClear() {
        if (ptr_ != nullptr) cudaFree(ptr_);  // destructor path: no throw
        ptr_ = nullptr;
        width_ = height_ = stride_ = 0;
    }

    void swap(DeviceBinMat& other) noexcept {
        uint32_t* p = ptr_; ptr_ = other.ptr_; other.ptr_ = p;
        size_t t;
        t = width_;  width_ = other.width_;  other.width_ = t;
        t = height_; height_ = other.height_; other.height_ = t;
        t = stride_; stride_ = other.stride_; other.stride_ = t;
    }

    uint32_t* ptr_ = nullptr;
    size_t width_ = 0;
    size_t height_ = 0;
    size_t stride_ = 0;  // words
};

/// @brief A wide image in device memory: one T per pixel, tight stride.
/// @tparam T `uint8_t` or `uint16_t` for an input (the input contract's integer
/// types) or a disparity map; `uint32_t` for a packed census descriptor,
/// where the "pixel" is one pixel's whole K-bit comparison word
/// (census.hpp, censusTransformPacked).
template <typename T>
class DeviceImage {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
                  "DeviceImage: one-, two- or four-byte pixels only");

public:
    DeviceImage(int width, int height) {
        if (width < 0 || height < 0)
            BINCV_THROW(std::invalid_argument, "DeviceImage: negative dimensions");
        width_ = static_cast<size_t>(width);
        height_ = static_cast<size_t>(height);
        stride_ = width_;
        allocate();
    }

    DeviceImage() = default;

    DeviceImage(const DeviceImage& other)
        : width_(other.width_), height_(other.height_), stride_(other.stride_) {
        allocate();
        if (ptr_ != nullptr) {
            BINCV_CUDA_CHECK(cudaMemcpy(ptr_, other.ptr_, height_ * stride_ * sizeof(T),
                                        cudaMemcpyDeviceToDevice));
        }
    }

    DeviceImage& operator=(const DeviceImage& other) {
        if (this == &other) return *this;
        DeviceImage fresh(other);
        swap(fresh);
        return *this;
    }

    DeviceImage(DeviceImage&& other) noexcept { swap(other); }
    DeviceImage& operator=(DeviceImage&& other) noexcept {
        if (this != &other) {
            releaseAndClear();
            swap(other);
        }
        return *this;
    }

    ~DeviceImage() { releaseAndClear(); }

    size_t getWidth() const { return width_; }
    size_t getHeight() const { return height_; }
    size_t getStride() const { return stride_; }
    bool empty() const { return width_ == 0 || height_ == 0; }

    DeviceImageView<T> view() { return DeviceImageView<T>{ptr_, width_, height_, stride_}; }
    DeviceImageConstView<T> constView() const {
        return DeviceImageConstView<T>{ptr_, width_, height_, stride_};
    }

private:
    void allocate() {
        const size_t n = height_ * stride_;
        if (n == 0) return;
        void* p = nullptr;
        BINCV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
        ptr_ = static_cast<T*>(p);
        BINCV_CUDA_CHECK(cudaMemset(ptr_, 0, n * sizeof(T)));
    }

    void releaseAndClear() {
        if (ptr_ != nullptr) cudaFree(ptr_);
        ptr_ = nullptr;
        width_ = height_ = stride_ = 0;
    }

    void swap(DeviceImage& other) noexcept {
        T* p = ptr_; ptr_ = other.ptr_; other.ptr_ = p;
        size_t t;
        t = width_;  width_ = other.width_;  other.width_ = t;
        t = height_; height_ = other.height_; other.height_ = t;
        t = stride_; stride_ = other.stride_; other.stride_ = t;
    }

    T* ptr_ = nullptr;
    size_t width_ = 0;
    size_t height_ = 0;
    size_t stride_ = 0;  // elements
};

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
