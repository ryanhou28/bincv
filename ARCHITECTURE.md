# binCV Architecture & Implementation Plan

## Executive Summary

binCV is an accelerated computer vision library optimized for binary (1-bit) image processing. It targets applications like single-photon cameras, event camera frame representations, and other novel binary sensors where traditional libraries like OpenCV are inefficient.

**Core Value Proposition:** Achieve 10-100x performance improvements over traditional libraries by:
1. Exploiting bit-packed storage (8-64x memory reduction)
2. Leveraging bitwise operations native to binary data
3. SIMD vectorization of packed operations
4. Optimized CPU and GPU implementations

This document provides the technical architecture and concrete implementation plan for building binCV.

---

## 1. Motivation & Technical Opportunity

### Target Applications

1. **Single-Photon Avalanche Diode (SPAD) Cameras**
   - Generate thousands of binary frames per second
   - Each pixel = photon detected (1) or not (0)
   - Example: 1000 fps @ 1024x1024 = 1 GB/s with uint8, only 128 MB/s with 1-bit
   - Often deployed on resource-constrained embedded platforms

2. **Event Camera Frame Representations**
   - Asynchronous event streams collapsed into synchronous frames
   - Binary masks indicating which pixels had events
   - Edge/embedded deployment for robotics and autonomous systems

3. **Depth Sensing & Structured Light**
   - Binary patterns for structured light projection
   - Binary masks for depth discontinuities
   - Embedded depth sensors on mobile and edge devices

4. **Medical Imaging**
   - Binary segmentation masks
   - High-throughput morphological processing
   - Portable/embedded medical devices

5. **Document Processing**
   - Binarized text images
   - Morphological operations on text
   - Mobile document scanning

### Target Platforms

binCV is designed to run efficiently across the full spectrum of computing platforms:

1. **High-End Embedded** (Raspberry Pi, Jetson Nano, i.MX8)
   - ARM Cortex-A processors with NEON SIMD
   - 512MB - 4GB RAM
   - Optional GPU acceleration
   - Full feature set available

2. **Mid-Range Embedded** (BeagleBone, i.MX6)
   - ARM Cortex-A processors
   - 256MB - 1GB RAM
   - Core operations, optional SIMD

3. **Desktop/Server** (x86_64, ARM workstations)
   - Full AVX2/AVX-512 SIMD support
   - CUDA GPU acceleration
   - All features including Python bindings

4. **Low-End Embedded** (ARM Cortex-M7) - Future Support
   - Microcontrollers with limited RAM
   - Scalar operations only
   - Static allocation

### Why Existing Libraries Fail

**OpenCV's Limitations:**
- Stores binary images as CV_8U (8 bits per pixel) - 8x memory waste
- Operations process 8-bit values even for binary data
- No bitwise operation optimizations
- Cannot leverage population count (popcount), parallel bit extract (PEXT), etc.
- Cache inefficient for binary data

**Performance Gap Example (256x256 image):**
```
Operation        OpenCV      binCV Goal      Target Speedup
Memory           64 KB       8 KB            8x
Transpose        0.003 ms    0.003 ms        1x (match)
Convolution      0.5 ms      0.05 ms         10x
Morphology       1.0 ms      0.05 ms         20x
```

---

## 2. Technical Opportunities

### 2.1 Bit-Level Operations

**Boolean Logic as Image Processing:**
- **AND:** Intersection/masking
- **OR:** Union/accumulation
- **XOR:** Difference detection
- **NOT:** Inversion
- **POPCOUNT:** Area calculation, histogram

### 2.2 SIMD Vectorization

Modern CPUs can process 256-512 bits per instruction:
- AVX2: 256 bits = 256 pixels per operation
- AVX-512: 512 bits = 512 pixels per operation
- ARM NEON: 128 bits = 128 pixels per operation

**Opportunity:** Process entire rows/blocks in single instructions.

### 2.3 GPU Parallelism

GPUs excel at bit-packed operations:
- **Warp-level primitives:** `__popc()`, `__ballot_sync()`
- **Shared memory:** Efficient for bit-packed convolutions
- **Coalesced access:** Natural with bit-packed storage

### 2.4 Cache Efficiency

Binary images fit in cache hierarchy:
- 1024x1024 binary: 128 KB (fits in L2)
- 1024x1024 uint8: 1 MB (overflows L2 on many CPUs)
- 4096x4096 binary: 2 MB (fits in L3)

**Opportunity:** Orders of magnitude fewer cache misses.

### 2.5 Specialized Algorithms

**Binary Morphology:**
- Traditional: Per-pixel comparisons with structuring element
- Binary-optimized: Parallel OR/AND with shifted masks

**Binary Convolution:**
- Traditional: Multiply-accumulate operations
- Binary-optimized: POPCOUNT of masked regions

---

## 3. Current State Analysis

### 3.1 Strengths

✅ **Solid Foundation:**
- Templated `BinMat` class with flexible word sizes (8, 16, 32, 64 bits)
- OpenCV integration for compatibility
- Comprehensive benchmarking infrastructure
- Basic matrix operations working

✅ **Performance Wins:**
- Fill operations: 1.2-4.3x faster than OpenCV
- Memory: 8x reduction demonstrated

### 3.2 Critical Weaknesses

❌ **Naive Implementations:**
- Transpose: 50-175x SLOWER than OpenCV (pixel-by-pixel)
- No SIMD utilization
- No cache-aware algorithms

❌ **Missing Core Operations:**
- No convolution/filtering (except basic CUDA edge filter)
- No morphological operations (erode, dilate, open, close)
- No connected components
- No distance transforms

❌ **Implementation Bugs:**
- `fromCVMat()` has undefined variable references
- Template specializations missing scope qualifiers
- `forEachNonZero()` references wrong namespace

❌ **Testing Gap:**
- No formal test framework
- Limited test coverage
- No correctness validation against OpenCV

---

## 4. Architecture Design

### 4.1 Core Design Principles

1. **OpenCV-Compatible API:** Users should feel at home
2. **Zero-Copy Interop:** Seamless conversion to/from OpenCV (when available)
3. **Performance by Default:** Optimized paths without user intervention
4. **Correctness First:** Match OpenCV semantics exactly
5. **Modular Design:** CPU, GPU, and future accelerators as plugins
6. **Compile-Time Optimization:** Template-based specialization
7. **Progressive Enhancement:** Core works everywhere, features layer on top
8. **Zero Dependencies:** Core library has no external dependencies (not even OpenCV)
9. **Platform Flexibility:** From embedded to desktop with same codebase

### 4.2 Layered Architecture

**Progressive Enhancement Model:**

```
┌─────────────────────────────────────────────────────────┐
│         Optional: Python Bindings (Desktop)             │  ← Desktop/Server
├─────────────────────────────────────────────────────────┤
│      Optional: OpenCV Integration (opencv.hpp)          │  ← Desktop/Embedded with OpenCV
├─────────────────────────────────────────────────────────┤
│                   Algorithm Layer                       │  ← All Platforms
│  (Vision kernels: filter, morph, transform, etc.)      │
├─────────────────────────────────────────────────────────┤
│            Dispatch Layer (Optional)                    │  ← Runtime or Compile-time
│  (Runtime CPU/GPU selection, SIMD detection)           │
├─────────────────────────────────────────────────────────┤
│              Implementation Backends                    │  ← Mix and Match
│  ┌────────┬────────┬───────┬────────┬────────┐        │
│  │ Scalar │  NEON  │ AVX2  │ AVX512 │  CUDA  │        │  ← Auto-detected
│  └────────┴────────┴───────┴────────┴────────┘        │
├─────────────────────────────────────────────────────────┤
│            Core Layer (Zero Dependencies)               │  ← All Platforms
│  BinMat, Size, bit utilities, memory management        │
│           (C++11, std::vector only)                    │
└─────────────────────────────────────────────────────────┘
```

**Platform Configurations:**

| Platform | Core | SIMD | OpenCV | CUDA | Python |
|----------|------|------|--------|------|--------|
| **Desktop x86** | ✓ | AVX2/512 | ✓ | ✓ | ✓ |
| **Desktop ARM** | ✓ | NEON | ✓ | - | ✓ |
| **Raspberry Pi** | ✓ | NEON | Optional | - | Optional |
| **BeagleBone** | ✓ | Optional | Optional | - | - |
| **Jetson Nano** | ✓ | NEON | ✓ | ✓ | ✓ |
| **Cortex-M7** | ✓ | - | - | - | - |

### 4.3 Core Data Structure: BinMat

**Refined Design (Zero Dependencies):**

```cpp
template <typename WordType = uint32_t>
class BinMat {
public:
    // Type traits
    using word_type = WordType;
    static constexpr size_t word_size = sizeof(WordType) * 8;

    // NOTE: Currently using fixed 32-byte alignment (good for AVX2, cache lines, ARM NEON)
    // FUTURE OPTIMIZATION: Could make this compile-time configurable based on platform
    // (e.g., 64 bytes for AVX-512, 16 bytes for constrained embedded)
    static constexpr size_t alignment = 32; // bytes

    // Construction - owning memory
    BinMat();
    BinMat(int rows, int cols);
    BinMat(Size size);

    // Construction - non-owning (wrap external buffer)
    BinMat(int rows, int cols, WordType* data, size_t step_words);

    // Properties
    int rows() const;
    int cols() const;
    Size size() const;
    bool empty() const;
    size_t total() const;
    bool ownsMemory() const;

    // Element access (use sparingly - slow)
    bool at(int row, int col) const;
    void set(int row, int col, bool value);
    bool operator()(int row, int col) const;

    // Row access (efficient)
    const WordType* ptr(int row) const;
    WordType* ptr(int row);

    // Memory layout
    size_t step() const; // Row stride in bytes
    WordType* data();
    const WordType* data() const;

    // Metadata
    bool isContinuous() const;
    BinMat clone() const;

    // Factory methods
    static BinMat zeros(int rows, int cols);
    static BinMat ones(int rows, int cols);

private:
    WordType* data_;                    // Pointer to data
    std::vector<WordType> storage_;     // Owned storage (only used if owns_memory_)
    int rows_, cols_;
    size_t step_words_;                 // Row stride in words
    bool owns_memory_;                  // True if storage_ is used
};
```

**Key Design Decisions:**

1. **No OpenCV Dependency:** Uses `std::vector` for owned memory, not `cv::Mat`
2. **External Buffer Support:** Can wrap user-provided buffers (for embedded, DMA, sensor buffers)
3. **Simple Ownership:** Owned memory uses `std::vector`, wrapped buffers never own
4. **OpenCV-Compatible API:** Same method names and conventions
5. **Alignment Note:** Documents future optimization opportunity

### 4.4 OpenCV Integration (Optional)

**Separate Header:** `bincv/opencv.hpp`

```cpp
#ifdef BINCV_HAVE_OPENCV

#include <opencv2/core.hpp>
#include <bincv/bincv.hpp>

namespace bincv {
namespace opencv {

// Conversion from OpenCV
BinMat fromMat(const cv::Mat& mat);
void threshold(const cv::Mat& src, BinMat& dst, double thresh, int type);

// Conversion to OpenCV
cv::Mat toMat(const BinMat& bin);          // Returns CV_8U with values 0 or 255
cv::Mat toDisplay(const BinMat& bin);      // Alias for toMat (for clarity)

} // namespace opencv
} // namespace bincv

#endif // BINCV_HAVE_OPENCV
```

**Usage Patterns:**

```cpp
// Desktop with OpenCV
#include <bincv/bincv.hpp>
#include <bincv/opencv.hpp>

cv::Mat cv_img = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
bincv::BinMat bin = bincv::opencv::fromMat(cv_img);
bincv::erode(bin, result, kernel);
cv::imshow("Result", bincv::opencv::toDisplay(result));

// Embedded without OpenCV
#include <bincv/bincv.hpp>

uint32_t sensor_buffer[256];
bincv::BinMat bin(32, 32, sensor_buffer, 8);
bincv::erode(bin, result, kernel);
```

### 4.5 Core Types

**Size Struct:** Replaces `cv::Size` in core

```cpp
namespace bincv {
    struct Size {
        int width;
        int height;

        Size() : width(0), height(0) {}
        Size(int w, int h) : width(w), height(h) {}

        int area() const { return width * height; }
    };
}
```

**Type Aliases:**

```cpp
namespace bincv {
    template<typename WordType> class BinMat;

    using BinMat8  = BinMat<uint8_t>;   // 8 pixels per word
    using BinMat16 = BinMat<uint16_t>;  // 16 pixels per word
    using BinMat32 = BinMat<uint32_t>;  // 32 pixels per word (default)
    using BinMat64 = BinMat<uint64_t>;  // 64 pixels per word
}
```

### 4.6 Memory Layout

**Row-Major with Word Packing:**

```
Image: 10 cols × 3 rows (WordType = uint32_t)
Pixels per word: 32

Row 0: [p0-p9  | padding zeros] = 1 word (32 bits)
Row 1: [p10-p19 | padding zeros] = 1 word (32 bits)
Row 2: [p20-p29 | padding zeros] = 1 word (32 bits)

Memory alignment: 32 bytes (cache line aligned)
```

**Design Rationale:**
- Each row independently aligned → efficient row access
- Padding within words → no cross-word pixel access
- Cache line aligned → maximize throughput
- Works with external buffers (DMA, sensors) on embedded

### 4.7 Algorithm Categories

**Priority 1: Foundational Operations**
1. **Point Operations:** threshold, bitwise (AND, OR, XOR, NOT)
2. **Morphology:** erode, dilate, open, close, morphologyEx
3. **Filtering:** blur, medianBlur (binary variants)
4. **Geometric:** resize, warpAffine, flip, rotate, transpose

**Priority 2: Analysis Operations**
5. **Statistics:** countNonZero, mean, moments
6. **Components:** connectedComponents, connectedComponentsWithStats
7. **Contours:** findContours, drawContours
8. **Distance:** distanceTransform

**Priority 3: Advanced Operations**
9. **Feature Detection:** corners, edges, keypoints
10. **Template Matching:** matchTemplate
11. **Optical Flow:** calcOpticalFlow (for binary sequences)

### 4.8 Dispatch Mechanism

**Runtime CPU Feature Detection:**

```cpp
namespace bincv {
namespace dispatch {

enum class Backend {
    Scalar,     // Fallback
    SSE2,       // x86 baseline
    AVX2,       // 256-bit SIMD
    AVX512,     // 512-bit SIMD
    NEON,       // ARM SIMD
    CUDA,       // NVIDIA GPU
};

Backend detectBestBackend();

// Function dispatch
template <typename Func>
auto dispatch(Func scalar_impl) {
    static Backend backend = detectBestBackend();

    switch (backend) {
        case Backend::AVX512:
            if constexpr (has_avx512_impl<Func>())
                return avx512_impl<Func>();
        case Backend::AVX2:
            if constexpr (has_avx2_impl<Func>())
                return avx2_impl<Func>();
        default:
            return scalar_impl;
    }
}

} // namespace dispatch
} // namespace bincv
```

---

## 5. API Design

### 5.1 Core API (C++)

**Desktop with OpenCV:**

```cpp
#include <bincv/bincv.hpp>
#include <bincv/opencv.hpp>  // Optional OpenCV interop

// Construction
bincv::BinMat src(480, 640);
bincv::BinMat dst;

// From OpenCV (requires opencv.hpp)
cv::Mat cv_gray = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
bincv::BinMat bin = bincv::opencv::fromMat(cv_gray);

// Bitwise operations - flat namespace
bincv::BinMat mask = bincv::BinMat::ones(480, 640);
bincv::bitwise_and(src, mask, dst);
bincv::bitwise_or(src, mask, dst);
bincv::bitwise_xor(src, mask, dst);
bincv::bitwise_not(src, dst);

// Morphology - flat namespace
bincv::BinMat kernel = bincv::getStructuringElement(bincv::MORPH_RECT, {3, 3});
bincv::erode(src, dst, kernel);
bincv::dilate(src, dst, kernel);
bincv::morphologyEx(src, dst, bincv::MORPH_OPEN, kernel);

// Analysis
int count = bincv::countNonZero(src);
float density = count / (float)src.total();

// Display (requires opencv.hpp)
cv::imshow("Result", bincv::opencv::toDisplay(dst));
```

**Embedded without OpenCV:**

```cpp
#include <bincv/bincv.hpp>  // Core only, no dependencies

// Wrap external buffer (sensor, DMA, etc.)
uint32_t sensor_buffer[256];
bincv::BinMat src(32, 32, sensor_buffer, 8);

// Or allocate
bincv::BinMat dst(32, 32);

// All operations work the same
bincv::BinMat kernel = bincv::getStructuringElement(bincv::MORPH_RECT, {3, 3});
bincv::erode(src, dst, kernel);

int count = bincv::countNonZero(dst);
```

**GPU Acceleration (nested namespace):**

```cpp
#include <bincv/bincv.hpp>
#include <bincv/cuda.hpp>

// CPU version (flat namespace)
bincv::erode(src, dst, kernel);

// GPU version (nested namespace - explicit)
bincv::cuda::erode(src, dst, kernel);

// Async GPU execution
bincv::cuda::Stream stream;
stream.erode(src, dst, kernel);
stream.synchronize();
```

### 5.2 Python API

**NumPy/OpenCV-Compatible:**

```python
import bincv as bcv
import cv2
import numpy as np

# From NumPy/OpenCV
cv_gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
_, cv_binary = cv2.threshold(cv_gray, 128, 255, cv2.THRESH_BINARY)
bin_img = bcv.from_numpy(cv_binary)

# Bitwise operations
mask = bcv.ones((480, 640), dtype=bcv.binary)
result = bcv.bitwise_and(bin_img, mask)

# Morphology
kernel = bcv.getStructuringElement(bcv.MORPH_RECT, (3, 3))
eroded = bcv.erode(bin_img, kernel)

# Analysis
count = bcv.countNonZero(bin_img)
n_components, labels = bcv.connectedComponents(bin_img)
```

### 5.3 Naming Conventions

**Follow OpenCV Conventions:**
- Functions: `camelCase` (e.g., `morphologyEx`, `connectedComponents`)
- Classes: `PascalCase` (e.g., `BinMat`, `Size`)
- Constants: `UPPER_CASE` (e.g., `MORPH_RECT`, `INTER_NEAREST`)
- Namespaces: `lowercase` (e.g., `bincv`, `bincv::cuda`, `bincv::opencv`)

**Namespace Organization:**
- **Flat namespace for CPU operations:** `bincv::erode()`, `bincv::bitwise_and()` (OpenCV-style)
- **Nested for backends:** `bincv::cuda::erode()` (explicit GPU)
- **Nested for optional features:** `bincv::opencv::fromMat()` (requires OpenCV)

---

## 6. Optimization Techniques

### 6.1 SIMD Bitwise Operations

**Example: Bitwise AND (AVX2)**

```cpp
void bitwise_and_avx2(const BinMat& src1, const BinMat& src2, BinMat& dst) {
    assert(src1.size() == src2.size());
    dst = BinMat(src1.rows(), src1.cols());

    const size_t n_words = src1.rows() * (src1.step() / sizeof(uint32_t));
    const uint32_t* p1 = reinterpret_cast<const uint32_t*>(src1.data());
    const uint32_t* p2 = reinterpret_cast<const uint32_t*>(src2.data());
    uint32_t* pd = reinterpret_cast<uint32_t*>(dst.data());

    size_t i = 0;
    // Process 8 words (256 pixels) at a time
    for (; i + 8 <= n_words; i += 8) {
        __m256i v1 = _mm256_loadu_si256((__m256i*)(p1 + i));
        __m256i v2 = _mm256_loadu_si256((__m256i*)(p2 + i));
        __m256i vd = _mm256_and_si256(v1, v2);
        _mm256_storeu_si256((__m256i*)(pd + i), vd);
    }

    // Scalar remainder
    for (; i < n_words; ++i) {
        pd[i] = p1[i] & p2[i];
    }
}
```

### 6.2 Binary Morphology Optimization

**Traditional Approach (Slow):**
```cpp
// For each pixel, check neighborhood - O(kernel_size * image_size)
for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
        bool result = false;
        for (int kr = 0; kr < kernel_rows; ++kr) {
            for (int kc = 0; kc < kernel_cols; ++kc) {
                if (kernel(kr, kc) && src(r+kr, c+kc)) {
                    result = true;
                    break;
                }
            }
        }
        dst(r, c) = result;
    }
}
```

**Binary-Optimized Approach (Fast):**
```cpp
// Dilate = OR of shifted versions
void dilate_3x3(const BinMat& src, BinMat& dst) {
    dst = src.clone();

    // OR with 8 shifted neighbors
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;

            // Shift and OR (vectorized)
            BinMat shifted = shift(src, dr, dc);
            bitwise_or(dst, shifted, dst);
        }
    }
}
```

### 6.3 Population Count (POPCOUNT)

**Example: countNonZero with SIMD**

```cpp
int countNonZero_avx2(const BinMat& src) {
    const uint64_t* data = reinterpret_cast<const uint64_t*>(src.data());
    const size_t n_words = src.rows() * src.step() / sizeof(uint64_t);

    int total = 0;
    size_t i = 0;

    // AVX2: Process 4 × uint64 at a time
    __m256i sum = _mm256_setzero_si256();
    for (; i + 4 <= n_words; i += 4) {
        __m256i v = _mm256_loadu_si256((__m256i*)(data + i));

        // Use lookup table method for POPCNT
        __m256i lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
        );
        __m256i low = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
        __m256i high = _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0x0F));
        __m256i cnt_low = _mm256_shuffle_epi8(lookup, low);
        __m256i cnt_high = _mm256_shuffle_epi8(lookup, high);
        __m256i cnt = _mm256_add_epi8(cnt_low, cnt_high);

        sum = _mm256_add_epi64(sum, _mm256_sad_epu8(cnt, _mm256_setzero_si256()));
    }

    // Reduce vector to scalar
    uint64_t counts[4];
    _mm256_storeu_si256((__m256i*)counts, sum);
    total = counts[0] + counts[1] + counts[2] + counts[3];

    // Scalar remainder
    for (; i < n_words; ++i) {
        total += __builtin_popcountll(data[i]);
    }

    return total;
}
```

### 6.4 Cache-Blocked Transpose

```cpp
template <typename WordType>
void transpose_blocked(const BinMat<WordType>& src, BinMat<WordType>& dst) {
    constexpr int BLOCK_SIZE = 32; // Tune for L1 cache

    dst = BinMat<WordType>(src.cols(), src.rows());

    for (int i = 0; i < src.rows(); i += BLOCK_SIZE) {
        for (int j = 0; j < src.cols(); j += BLOCK_SIZE) {
            int max_i = std::min(i + BLOCK_SIZE, src.rows());
            int max_j = std::min(j + BLOCK_SIZE, src.cols());

            // Transpose block
            for (int ii = i; ii < max_i; ++ii) {
                for (int jj = j; jj < max_j; ++jj) {
                    dst.set(jj, ii, src.at(ii, jj));
                }
            }
        }
    }
}
```

### 6.5 GPU Morphology (CUDA)

```cuda
__global__ void dilate_3x3_kernel(
    const uint32_t* src, uint32_t* dst,
    int rows, int cols, int step_words
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) return;

    // Each thread processes 32 pixels (1 word)
    uint32_t result = 0;

    // OR with neighborhood
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            int nr = row + dr;
            int nc = col + dc;
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                uint32_t neighbor = src[nr * step_words + nc];

                // Handle bit shifts for pixel alignment
                if (dc < 0) {
                    neighbor >>= 1;
                    if (nc + 1 < cols) {
                        neighbor |= (src[nr * step_words + nc + 1] & 1) << 31;
                    }
                } else if (dc > 0) {
                    neighbor <<= 1;
                    if (nc > 0) {
                        neighbor |= (src[nr * step_words + nc - 1] >> 31) & 1;
                    }
                }

                result |= neighbor;
            }
        }
    }

    dst[row * step_words + col] = result;
}
```

---

## 7. Performance Targets

**Benchmark Suite (1024×1024 image):**

| Operation | OpenCV (uint8) | binCV Target | Speedup Goal |
|-----------|----------------|--------------|--------------|
| Bitwise AND | 0.05 ms | 0.005 ms | 10x |
| Erode 3×3 | 0.8 ms | 0.05 ms | 16x |
| Dilate 3×3 | 0.8 ms | 0.05 ms | 16x |
| Morphology Open | 1.6 ms | 0.1 ms | 16x |
| countNonZero | 0.1 ms | 0.01 ms | 10x |
| Resize 2×2 | 0.3 ms | 0.05 ms | 6x |
| Connected Components | 5.0 ms | 0.5 ms | 10x |
| Distance Transform | 2.0 ms | 0.2 ms | 10x |

**Memory Targets:**
- 1024×1024 binary: 128 KB (vs 1 MB for uint8)
- 8× memory reduction

---

## 8. Error Handling Strategy

**Two-Phase Approach:**

**Phase 0-2: Exceptions (Simpler Development)**

```cpp
void erode(const BinMat& src, BinMat& dst, const BinMat& kernel) {
    if (src.empty()) {
        throw std::invalid_argument("erode: source image is empty");
    }
    if (kernel.empty()) {
        throw std::invalid_argument("erode: kernel is empty");
    }
    // ... implementation
}
```

- Clean, idiomatic C++
- Good error messages
- Works well for desktop/Python bindings

**Phase 3+: Optional No-Exceptions Mode (Embedded Support)**

```cpp
// error.hpp
#ifdef BINCV_NO_EXCEPTIONS
    #define BINCV_CHECK(cond, msg) assert((cond) && msg)
#else
    #define BINCV_CHECK(cond, msg) if(!(cond)) throw std::invalid_argument(msg)
#endif

// Refactored implementation
void erode(const BinMat& src, BinMat& dst, const BinMat& kernel) {
    BINCV_CHECK(!src.empty(), "erode: source image is empty");
    BINCV_CHECK(!kernel.empty(), "erode: kernel is empty");
    // ... implementation
}
```

- Compile flag: `-DBINCV_NO_EXCEPTIONS`
- Embedded toolchains with `-fno-exceptions` supported
- Desktop users unaffected

---

## 9. Build System Strategy

**CMake Auto-Configuration:**

```cmake
# Automatic feature detection
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    target_compile_definitions(bincv INTERFACE BINCV_HAVE_OPENCV)
    message(STATUS "OpenCV found - enabling interop")
else()
    message(STATUS "OpenCV not found - core-only mode (embedded)")
endif()

# SIMD detection
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" HAVE_AVX2)
check_cxx_compiler_flag("-mavx512f" HAVE_AVX512)
check_cxx_compiler_flag("-mfpu=neon" HAVE_NEON)

# CUDA detection
find_package(CUDA QUIET)
if(CUDA_FOUND)
    target_compile_definitions(bincv INTERFACE BINCV_HAVE_CUDA)
endif()

# Platform-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    message(STATUS "ARM target - optimizing for size")
    target_compile_options(bincv INTERFACE -Os)
endif()
```

**Build Modes:**

| Mode | Command | Features |
|------|---------|----------|
| **Full** | `cmake ..` | Auto-detect all |
| **No OpenCV** | `cmake -DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON ..` | Core only |
| **No CUDA** | `cmake -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON ..` | CPU only |
| **Embedded** | `cmake -DBINCV_EMBEDDED=ON ..` | Core + scalar, size-optimized |

---

## 10. Implementation Plan

### Phase 0: Embedded Foundation & Core Refactoring

**Goal:** Remove OpenCV dependency from core, enable embedded support, maintain desktop usability

**Priority:** CRITICAL - Must be done before Phase 1

**Tasks:**

1. **Create Core Types** (`bincv/core/types.hpp`)
   - Add `struct Size { int width, height; }`
   - Add type aliases: `BinMat8`, `BinMat16`, `BinMat32`, `BinMat64`
   - Add enums: `MorphShape`, `MorphOp`, border types, etc.

2. **Refactor `BinMat` Storage** (`bincv/core/binmat.hpp`)
   - Replace `cv::Mat data_` with `std::vector<WordType> storage_`
   - Add `WordType* data_` pointer member
   - Add `bool owns_memory_` flag
   - Add external buffer constructor: `BinMat(int rows, int cols, WordType* data, size_t step_words)`
   - Update all internal methods to use new storage
   - Add alignment constant with comment about future platform-specific optimization

3. **Create OpenCV Interop Header** (`bincv/opencv.hpp`)
   - Move `fromMat()`, `toMat()`, `toDisplay()` to `bincv::opencv` namespace
   - Guard with `#ifdef BINCV_HAVE_OPENCV`
   - Keep same functionality, just separate from core

4. **Update Existing Operations**
   - Replace `cv::Size` with `bincv::Size` throughout
   - Replace `cv::Rect` with `bincv::Rect` (if used)
   - Update `getStructuringElement()` signature
   - Ensure no OpenCV usage in core implementations

5. **CMake Configuration**
   - Make OpenCV optional (auto-detect)
   - Auto-detect SIMD capabilities
   - Add configuration summary
   - Support cross-compilation for ARM

6. **Testing**
   - Ensure all existing tests pass
   - Add test for external buffer wrapping
   - Cross-compile for ARM (Raspberry Pi)
   - Measure binary size (core only, no OpenCV)
   - Create golden test files (pre-computed results)

7. **Documentation**
   - Update this ARCHITECTURE.md with decisions
   - Update ROADMAP.md with Phase 0 details
   - Add embedded build guide
   - Document build without OpenCV

**Completion Criteria:**
- ✅ Core builds without OpenCV (`-DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON`)
- ✅ Desktop workflow unchanged (just add `#include <bincv/opencv.hpp>` where needed)
- ✅ All existing tests pass
- ✅ Cross-compiles for ARM
- ✅ Binary size measured and documented
- ✅ External buffer wrapping works and tested

**Estimated Effort:** 2-3 days

---

### Phase 1: Foundation & Correctness

**Goals:**
- Fix existing bugs
- Establish testing infrastructure
- Implement core optimized operations
- Match OpenCV semantics exactly

**Concrete Tasks:**

1. **Fix Critical Bugs**
   - Fix `fromCVMat()` undefined variable references ([binMat_impl.hpp:42](bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp))
   - Fix template specialization scope issues ([binMat_impl.hpp:156+](bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp))
   - Fix `forEachNonZero()` namespace errors ([binMat_impl.hpp:345](bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp))
   - Validate all existing operations compile and work

2. **Testing Infrastructure**
   - Integrate Google Test framework
   - Create test utilities for comparing with OpenCV:
     ```cpp
     void assertMatEqual(const BinMat& actual, const cv::Mat& expected);
     BinMat createRandomBinary(int rows, int cols, float sparsity);
     ```
   - Implement property-based testing
   - Add correctness tests for all operations (100+ test cases)

3. **Optimize Transpose**
   - Implement cache-blocked algorithm
     - Target: Match or beat OpenCV (0.003 ms for 256×256)
     - Use 32×32 or 64×64 tiles for cache locality
   - Add AVX2 SIMD variant for hot paths
   - Handle edge cases (non-square, small matrices)

4. **Core Bitwise Operations**
   - Implement: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`
   - Multiple backends:
     - Scalar (fallback)
     - AVX2 (256-bit SIMD)
     - AVX-512 (512-bit SIMD)
     - NEON (ARM)
   - Runtime dispatch based on CPU features
   - Target: 10× faster than OpenCV

5. **Documentation Setup**
   - Set up Doxygen for API documentation
   - Document all public APIs with usage examples
   - Create benchmarking guide

**Phase 1 Completion Criteria:**
- ✅ All existing bugs fixed and code compiles
- ✅ All existing operations validated against OpenCV
- ✅ Transpose performance matches OpenCV
- ✅ Core bitwise operations 10× faster
- ✅ 100+ passing unit tests
- ✅ Google Test integrated

---

### Phase 2: Core Vision Operations

**Goals:**
- Implement morphological operations
- Implement filtering operations
- Establish CPU SIMD dispatch
- Comprehensive benchmarking

**Concrete Tasks:**

1. **Morphological Operations**
   - Implement separable morphology:
     ```cpp
     void erode(const BinMat& src, BinMat& dst, const BinMat& kernel);
     void dilate(const BinMat& src, BinMat& dst, const BinMat& kernel);
     void morphologyEx(const BinMat& src, BinMat& dst, int op, const BinMat& kernel);
     ```
   - Operations: `MORPH_ERODE`, `MORPH_DILATE`, `MORPH_OPEN`, `MORPH_CLOSE`, `MORPH_GRADIENT`, `MORPH_TOPHAT`, `MORPH_BLACKHAT`
   - Optimizations:
     - Shift-and-OR/AND for 3×3, 5×5 kernels (most common)
     - SIMD horizontal OR/AND reductions
     - Separable decomposition for rectangular kernels
   - Target: 15-20× faster than OpenCV

2. **Filtering Operations**
   - Binary blur (majority filter):
     ```cpp
     void blur(const BinMat& src, BinMat& dst, cv::Size ksize);
     ```
   - Binary median filter:
     ```cpp
     void medianBlur(const BinMat& src, BinMat& dst, int ksize);
     ```
   - Use SIMD POPCOUNT for counting
   - Target: 10× faster than OpenCV

3. **Runtime Dispatch System**
   - CPU feature detection at runtime
   - Function pointer tables for operations
   - Compile-time backend selection option

4. **Padding Operations**
   - Implement all OpenCV border types:
     ```cpp
     void copyMakeBorder(const BinMat& src, BinMat& dst,
                         int top, int bottom, int left, int right,
                         int borderType, bool value = false);
     ```
   - Types: `BORDER_CONSTANT`, `BORDER_REPLICATE`, `BORDER_REFLECT`, `BORDER_WRAP`

5. **Comprehensive Benchmarking**
   - Extend benchmark suite:
     - All morphological operations
     - All filter operations
     - Multiple image sizes: 256², 512², 1024², 2048², 4096²
     - Multiple sparsity levels: 1%, 10%, 50%, 90%, 99%
   - Generate performance reports

**Phase 2 Completion Criteria:**
- ✅ Morphology: erode, dilate, open, close, etc. (15-20× faster)
- ✅ Filters: blur, median (10× faster)
- ✅ Runtime SIMD dispatch working
- ✅ Padding with all border types
- ✅ Comprehensive benchmark results

---

### Phase 3: Analysis & GPU Acceleration

**Goals:**
- Connected components analysis
- Distance transforms
- Statistical operations
- CUDA GPU implementations

**Concrete Tasks:**

1. **Connected Components**
   - Union-find on binary images:
     ```cpp
     int connectedComponents(const BinMat& image, BinMat& labels, int connectivity = 8);
     int connectedComponentsWithStats(const BinMat& image, BinMat& labels,
                                      cv::Mat& stats, cv::Mat& centroids, int connectivity = 8);
     ```
   - Optimizations:
     - Parallel union-find
     - Block-based processing for cache efficiency
     - SIMD for label propagation
   - Target: 10× faster than OpenCV

2. **Distance Transform**
   - Euclidean and L1 distance:
     ```cpp
     void distanceTransform(const BinMat& src, cv::Mat& dst, int distanceType, int maskSize);
     ```
   - Use fast sequential algorithm (Felzenszwalb)
   - Optimize with SIMD for distance comparisons
   - Target: 5-10× faster than OpenCV

3. **Statistical Operations**
   - Optimized counting:
     ```cpp
     int countNonZero(const BinMat& src);
     double mean(const BinMat& src);
     cv::Moments moments(const BinMat& src);
     ```
   - Use SIMD POPCOUNT implementations
   - Target: 10× faster for countNonZero

4. **CUDA Implementations**
   - Port optimized operations to CUDA:
     - Bitwise operations
     - Morphology: erode, dilate, open, close
     - Connected components (GPU parallel union-find)
     - Distance transform (parallel algorithm)
   - Optimizations:
     - Shared memory for convolution-like ops
     - Warp-level primitives
     - Coalesced memory access
   - Target: 100× faster on GPU for large images (2048² and above)

5. **Async GPU Execution**
   - Non-blocking API:
     ```cpp
     namespace bincv::cuda {
         class Stream {
         public:
             void erode(const BinMat& src, BinMat& dst, const BinMat& kernel);
             void synchronize();
         };
     }
     ```
   - Overlapped CPU/GPU execution

**Phase 3 Completion Criteria:**
- ✅ Connected components (10× faster on CPU)
- ✅ Distance transform (5-10× faster on CPU)
- ✅ CUDA implementations of core operations
- ✅ GPU performance: 100× faster for 2048² images
- ✅ Async CUDA API functional

---

### Phase 4: Advanced Operations & Language Bindings

**Goals:**
- Python bindings
- Advanced vision operations
- Comprehensive documentation
- Example applications

**Concrete Tasks:**

1. **Python Bindings (binCV-py)**
   - Use pybind11 for C++ bindings
   - NumPy interop:
     ```python
     def from_numpy(arr: np.ndarray) -> BinMat
     def to_numpy(mat: BinMat) -> np.ndarray
     ```
   - Match OpenCV-Python API style
   - Support Python type hints
   - Zero-copy where possible
   - Release GIL in all operations

2. **Contour Operations**
   - Find contours:
     ```cpp
     void findContours(const BinMat& image,
                       std::vector<std::vector<cv::Point>>& contours,
                       int mode, int method);
     ```
   - Draw contours:
     ```cpp
     void drawContours(BinMat& image,
                       const std::vector<std::vector<cv::Point>>& contours,
                       int contourIdx, bool color, int thickness = 1);
     ```

3. **Template Matching**
   - Binary correlation:
     ```cpp
     void matchTemplate(const BinMat& image, const BinMat& templ,
                        cv::Mat& result, int method);
     ```
   - Methods: `TM_SQDIFF`, `TM_CCORR`, `TM_CCOEFF`
   - Use FFT for large templates
   - Use SIMD POPCOUNT for small templates

4. **Feature Detection**
   - Binary corner detection:
     ```cpp
     void goodFeaturesToTrack(const BinMat& image,
                              std::vector<cv::Point2f>& corners,
                              int maxCorners, double qualityLevel, double minDistance);
     ```

5. **Example Applications**
   - SPAD camera processing pipeline
   - Event camera frame processing
   - Document image morphology
   - Depth discontinuity detection
   - Binary pattern matching

6. **Documentation**
   - Tutorials:
     - Getting started
     - Migration from OpenCV
     - Performance optimization
     - CUDA programming with binCV
   - API reference (Doxygen)
   - Architecture overview
   - Benchmarking guide

**Phase 4 Completion Criteria:**
- ✅ Python bindings working with pip package structure
- ✅ Contour operations matching OpenCV
- ✅ Template matching (10× faster)
- ✅ 5+ example applications
- ✅ Complete documentation
- ✅ Pip package installable locally

---

### Phase 5: Platform Expansion & Advanced Backends

**Goals:**
- Additional platform support
- Framework integration
- Advanced GPU backends
- Profiling infrastructure

**Concrete Tasks:**

1. **ARM Platform Support**
   - Implement ARM NEON SIMD backend
   - Runtime detection for ARM CPUs
   - Optimize for mobile and edge devices

2. **Framework Integration**
   - PyTorch custom operators
   - TensorFlow custom ops
   - JAX integration

3. **Additional GPU Backends**
   - Apple Metal GPU backend
   - Vulkan compute backend
   - OpenCL backend

4. **Advanced GPU Features**
   - Multi-GPU support
   - GPU streams and events
   - Unified memory

5. **Performance Profiling Tools**
   - Built-in profiler:
     ```cpp
     bincv::Profiler profiler;
     profiler.start("erode");
     bincv::erode(src, dst, kernel);
     profiler.stop("erode");
     profiler.report();
     ```

**Phase 5 Completion Criteria:**
- ✅ ARM NEON backend functional
- ✅ PyTorch/TensorFlow custom operators working
- ✅ Metal or Vulkan GPU backend implemented
- ✅ Multi-GPU support for CUDA
- ✅ Built-in profiling infrastructure

---

## 9. Technical Challenges & Solutions

### Challenge 1: Bit Shifting Across Word Boundaries

**Problem:** Morphology operations need to access neighboring pixels, which may cross word boundaries.

**Solution:**
- Precompute shifted copies for small shifts (±1, ±2)
- Use SIMD shuffle for intra-register shifts
- Cache word pairs in hot loops

### Challenge 2: Non-Multiple-of-Word-Size Dimensions

**Problem:** Image width may not be divisible by word size.

**Solution:**
- Always pad width to next multiple of word size
- Mark padding bits as 0
- Handle remainder in scalar code for correctness

### Challenge 3: Cache Coherency on GPU

**Problem:** Bit-packed data reduces memory bandwidth but increases computation for unpacking.

**Solution:**
- Use shared memory aggressively
- Unpack once per block, process many pixels
- Coalesce memory accesses

### Challenge 4: Python GIL for NumPy Interop

**Problem:** Python's Global Interpreter Lock limits parallelism.

**Solution:**
- Release GIL in all binCV operations:
  ```cpp
  py::gil_scoped_release release;
  bincv::erode(src, dst, kernel);
  ```
- Zero-copy NumPy arrays where possible

### Challenge 5: Maintaining OpenCV Compatibility

**Problem:** OpenCV has quirks (coordinate conventions, border handling).

**Solution:**
- Extensive testing against OpenCV reference
- Document any unavoidable differences
- Provide conversion utilities for edge cases

---

## 10. Quality & Performance Metrics

### Performance Targets

**Quantitative Goals:**
1. **Speedup:** 10-100× faster than OpenCV for core operations
2. **Memory:** 8× reduction (1-bit vs 8-bit storage)
3. **Throughput:** Process 1000 fps @ 1024×1024 on consumer GPU
4. **Latency:** <1ms for common operations on CPU

**Benchmark Coverage:**
- 50+ operations benchmarked
- 5 image sizes: 256², 512², 1024², 2048², 4096²
- 5 sparsity levels: 1%, 10%, 50%, 90%, 99%
- 3 backends: CPU scalar, CPU SIMD, GPU CUDA

### Correctness Requirements

**Testing:**
- 100% test coverage of public API
- 500+ unit tests passing
- Validation against OpenCV on 1000+ random images
- Property-based testing for morphological operations
- Fuzz testing for edge cases

**Code Quality:**
- All public APIs documented with Doxygen
- Static analysis passing (clang-tidy, cppcheck)
- No memory leaks (valgrind clean)
- Thread-safe where applicable

---

## 11. Comparison with Related Work

| Library | Binary Support | Performance | Usability | GPU Support |
|---------|----------------|-------------|-----------|-------------|
| **OpenCV** | No (uses uint8) | Baseline | Excellent | CUDA module |
| **scikit-image** | No (uses uint8) | Slow (Python) | Good | No |
| **IPP** (Intel) | No | Fast (SIMD) | Good | No |
| **NPP** (NVIDIA) | No | Fast (GPU) | Good | Yes (CUDA only) |
| **Halide** | Custom DSL | Fast (JIT) | Complex | Yes |
| **binCV** | **Yes (native)** | **10-100× faster** | **OpenCV-like** | **Yes (CUDA, Vulkan)** |

**Unique Value Proposition:**
- Only library with native binary (1-bit) support
- 10-100× speedup for binary operations
- OpenCV-compatible API (drop-in replacement)
- Cross-platform CPU and GPU acceleration

---

## 11. Implementation Priorities

### Critical Path

The following sequence must be followed as each builds on the previous:

1. **Phase 0: Core Refactoring** → Remove OpenCV dependency, enable embedded
2. **Fix Compilation Bugs** → Required for any development
3. **Establish Testing** → Required for validating correctness
4. **Optimize Transpose** → High-impact, foundational operation
5. **Core Bitwise Ops** → Building blocks for advanced operations
6. **Morphology** → Most important vision operations for binary images
7. **SIMD Dispatch** → Performance multiplier for all operations
8. **GPU Backend** → Massive speedup for large images
9. **Advanced Operations** → Build on stable foundation
10. **Language Bindings** → Enable broader usage

### Dependency Graph

```
Phase 0: Core Refactoring (Remove OpenCV dependency)
      ↓
Testing Framework
      ↓
Bug Fixes → Transpose Opt → Bitwise Ops → Morphology → Filtering
                                  ↓            ↓          ↓
                            SIMD Dispatch ─────┴──────────┘
                                  ↓
                            GPU Backend
                                  ↓
                      Connected Components, Distance Transform
                                  ↓
                         Advanced Operations
                                  ↓
                          Python Bindings
```

### Platform Support Timeline

| Phase | Desktop | High-End Embedded | Mid-Range Embedded | Low-End Embedded |
|-------|---------|-------------------|-------------------|------------------|
| **Phase 0** | Core working | Core working | Core working | Core working |
| **Phase 1** | + Optimized ops | + Optimized ops | + Scalar ops | + Scalar ops |
| **Phase 2** | + SIMD (AVX2) | + SIMD (NEON) | Optional SIMD | - |
| **Phase 3** | + CUDA | + CUDA (Jetson) | - | - |
| **Phase 4** | + Python | Optional Python | - | - |
| **Phase 5** | Full features | Full features | Core features | Core features |

---

## 12. Conclusion

binCV fills a critical gap in computer vision for binary image processing. By leveraging bit-level operations, SIMD vectorization, and GPU parallelism, we can achieve 10-100× performance improvements over existing libraries while supporting platforms from embedded microcontrollers to high-performance servers.

**Key Success Factors:**
1. **Correctness First:** Match OpenCV semantics exactly
2. **Performance by Default:** Optimize common cases automatically
3. **Usability:** OpenCV-compatible API
4. **Platform Flexibility:** From embedded to desktop with same codebase
5. **Zero Dependencies:** Core library works everywhere
6. **Progressive Enhancement:** Features layer on top as available
7. **Comprehensive Testing:** Validate everything against reference
8. **Systematic Implementation:** Follow dependency-ordered plan

**Unique Value Proposition:**
- **Only library with native 1-bit binary support**
- **10-100× speedup across all platforms**
- **Works on embedded devices** (Raspberry Pi, ARM Cortex-A)
- **No mandatory dependencies** (OpenCV optional)
- **Same codebase, automatic adaptation** to platform capabilities

With systematic execution of this implementation plan, binCV can enable new applications in emerging vision technologies like SPAD cameras, event cameras, and high-speed binary pattern processing - on any platform from embedded edge devices to high-performance computing clusters.
