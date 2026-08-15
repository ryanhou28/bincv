# binCV Implementation Roadmap

This document provides a concrete, actionable roadmap for implementing binCV. For complete architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Vision

Build the world's fastest binary image processing library by exploiting bit-packed storage, bitwise operations, SIMD vectorization, and GPU parallelism to achieve **10-100× speedup** over OpenCV.

---

## Core Principles

1. **OpenCV-Compatible API** - Familiar interface
2. **Correctness First** - Match OpenCV semantics exactly
3. **Performance by Default** - Automatic optimization
4. **Zero-Copy Interop** - Seamless OpenCV integration (when available)
5. **Modular Design** - CPU, GPU, future accelerators as plugins
6. **Platform Flexibility** - From embedded to desktop with same codebase
7. **Zero Dependencies** - Core works everywhere, features are optional
8. **Progressive Enhancement** - Capabilities layer on as available

---

## Current Status

### Working
- ✅ Templated `BinMat` class with 8/16/32/64-bit word sizes
- ✅ Basic operations: fill, resize, padding, transpose (slow)
- ✅ OpenCV conversion (with bugs)
- ✅ Benchmarking infrastructure
- ✅ Fill operation 1.2-4.3× faster than OpenCV

### Broken
- ❌ **3 critical compilation bugs** (must fix first)
- ❌ Transpose 50-175× SLOWER than OpenCV (naive implementation)
- ❌ No SIMD optimization
- ❌ No formal testing framework

### Missing
- ❌ Core bitwise operations (AND, OR, XOR, NOT)
- ❌ Morphological operations (erode, dilate, open, close)
- ❌ Connected components
- ❌ Distance transform
- ❌ Python bindings

---

## Implementation Phases

### Phase 0: Embedded Foundation & Core Refactoring

**Goal:** Remove OpenCV dependency from core, enable embedded support, maintain desktop usability

**Priority:** CRITICAL - Must be done first (before Phase 1)

**Rationale:** Binary sensors (SPAD, event cameras) are often deployed on embedded/edge devices. The core library must work without OpenCV to support these platforms.

#### 0.1 Create Core Types

**File:** `bincv-cpp/include/bincv-cpp/core/types.hpp` (NEW)

```cpp
namespace bincv {
    // Size struct (replaces cv::Size in core)
    struct Size {
        int width, height;
        Size() : width(0), height(0) {}
        Size(int w, int h) : width(w), height(h) {}
        int area() const { return width * height; }
    };

    // Type aliases for convenience
    template<typename WordType> class BinMat;
    using BinMat8  = BinMat<uint8_t>;
    using BinMat16 = BinMat<uint16_t>;
    using BinMat32 = BinMat<uint32_t>;  // Default
    using BinMat64 = BinMat<uint64_t>;

    // Morphology enums
    enum MorphShape {
        MORPH_RECT = 0,
        MORPH_CROSS = 1,
        MORPH_ELLIPSE = 2
    };

    enum MorphOp {
        MORPH_ERODE = 0,
        MORPH_DILATE = 1,
        MORPH_OPEN = 2,
        MORPH_CLOSE = 3,
        MORPH_GRADIENT = 4,
        MORPH_TOPHAT = 5,
        MORPH_BLACKHAT = 6
    };
}
```

#### 0.2 Refactor BinMat Storage

**Goal:** Remove `cv::Mat` dependency from core

**Changes to `binMat.hpp` and `binMat_impl.hpp`:**

1. **Replace internal storage:**
   ```cpp
   // OLD
   cv::Mat data_;  // Underlying storage

   // NEW
   WordType* data_;                    // Pointer to data
   std::vector<WordType> storage_;     // Owned storage (only if owns_memory_)
   bool owns_memory_;                  // True if we own the memory
   ```

2. **Add external buffer constructor:**
   ```cpp
   // Non-owning constructor for embedded (wrap sensor buffers, DMA, etc.)
   BinMat(int rows, int cols, WordType* data, size_t step_words);
   ```

3. **Add alignment comment:**
   ```cpp
   // NOTE: Currently using fixed 32-byte alignment (good for AVX2, cache lines, ARM NEON)
   // FUTURE OPTIMIZATION: Could make this compile-time configurable based on platform
   // (e.g., 64 bytes for AVX-512, 16 bytes for constrained embedded)
   static constexpr size_t alignment = 32;
   ```

4. **Update factory methods:**
   ```cpp
   static BinMat zeros(int rows, int cols);
   static BinMat ones(int rows, int cols);
   ```

#### 0.3 Create OpenCV Interop Header

**File:** `bincv-cpp/include/bincv-cpp/opencv.hpp` (NEW)

```cpp
#ifndef BINCV_OPENCV_HPP
#define BINCV_OPENCV_HPP

#ifdef BINCV_HAVE_OPENCV

#include <opencv2/core.hpp>
#include <bincv-cpp/bincv.hpp>

namespace bincv {
namespace opencv {

// Convert from OpenCV Mat to BinMat
BinMat fromMat(const cv::Mat& mat);

// Threshold OpenCV Mat directly to BinMat
void threshold(const cv::Mat& src, BinMat& dst, double thresh, int type);

// Convert BinMat to OpenCV Mat (returns CV_8U with values 0 or 255)
cv::Mat toMat(const BinMat& bin);

// Alias for clarity
inline cv::Mat toDisplay(const BinMat& bin) { return toMat(bin); }

} // namespace opencv
} // namespace bincv

#endif // BINCV_HAVE_OPENCV

#endif // BINCV_OPENCV_HPP
```

**Implementation:** Move existing `fromCVMat()`, `toCVMat()` logic here.

#### 0.4 Update Existing Operations

1. **Replace cv::Size with bincv::Size:**
   - Update `getStructuringElement()` signature
   - Update all function signatures using `cv::Size`
   - Update implementation code

2. **Remove OpenCV dependencies from core:**
   - Audit all core operations
   - Ensure no `cv::` usage except in `opencv.hpp`

#### 0.5 CMake Configuration

**Update `CMakeLists.txt`:**

```cmake
cmake_minimum_required(VERSION 3.15)
project(bincv)

# Core library (header-only, no dependencies)
add_library(bincv_core INTERFACE)
target_include_directories(bincv_core INTERFACE include/)
target_compile_features(bincv_core INTERFACE cxx_std_11)

# Optional: OpenCV integration
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    target_compile_definitions(bincv_core INTERFACE BINCV_HAVE_OPENCV)
    target_link_libraries(bincv_core INTERFACE ${OpenCV_LIBS})
    message(STATUS "✓ OpenCV found - enabling interop (bincv/opencv.hpp available)")
else()
    message(STATUS "✗ OpenCV not found - core-only mode (embedded)")
endif()

# SIMD detection
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" HAVE_AVX2)
check_cxx_compiler_flag("-mavx512f" HAVE_AVX512)

if(HAVE_AVX512)
    target_compile_definitions(bincv_core INTERFACE BINCV_HAVE_AVX512)
    message(STATUS "✓ AVX-512 support enabled")
elseif(HAVE_AVX2)
    target_compile_definitions(bincv_core INTERFACE BINCV_HAVE_AVX2)
    message(STATUS "✓ AVX2 support enabled")
endif()

# ARM NEON detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    check_cxx_compiler_flag("-mfpu=neon" HAVE_NEON)
    if(HAVE_NEON)
        target_compile_definitions(bincv_core INTERFACE BINCV_HAVE_NEON)
        message(STATUS "✓ ARM NEON support enabled")
    endif()
    # Optimize for size on embedded ARM
    target_compile_options(bincv_core INTERFACE -Os)
    message(STATUS "✓ ARM target detected - optimizing for size (-Os)")
endif()

# Configuration summary
message(STATUS "")
message(STATUS "========== binCV Configuration ==========")
message(STATUS "Platform: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(OpenCV_FOUND)
    message(STATUS "OpenCV: YES (${OpenCV_VERSION})")
else()
    message(STATUS "OpenCV: NO (embedded mode)")
endif()
message(STATUS "==========================================")
message(STATUS "")
```

#### 0.6 Update Tests

1. **Split OpenCV-dependent tests:**
   ```cpp
   #ifdef BINCV_HAVE_OPENCV
   TEST(BinMat, ConvertFromOpenCV) {
       cv::Mat cv_mat = cv::Mat::zeros(10, 10, CV_8U);
       bincv::BinMat bin = bincv::opencv::fromMat(cv_mat);
       EXPECT_EQ(bin.rows(), 10);
       EXPECT_EQ(bin.cols(), 10);
   }
   #endif

   // Core tests (no OpenCV)
   TEST(BinMat, Construction) {
       bincv::BinMat mat(10, 10);
       EXPECT_EQ(mat.rows(), 10);
       EXPECT_EQ(mat.cols(), 10);
       EXPECT_TRUE(mat.ownsMemory());
   }

   TEST(BinMat, ExternalBuffer) {
       uint32_t buffer[32];
       bincv::BinMat mat(8, 8, buffer, 1);
       EXPECT_FALSE(mat.ownsMemory());
       EXPECT_EQ(mat.data(), buffer);
   }
   ```

2. **Update test utilities:**
   - Make OpenCV comparison utilities optional
   - Add golden file-based testing

3. **Add embedded-specific tests:**
   - External buffer wrapping
   - Non-owning semantics
   - Memory measurement

#### 0.7 Cross-Compilation Test

**For Raspberry Pi (example):**

```bash
# Install cross-compiler
sudo apt-get install g++-arm-linux-gnueabihf

# Configure for ARM
mkdir build-arm && cd build-arm
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/arm-linux-gnueabihf.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON

# Build
make -j$(nproc)

# Measure binary size
size test_binMat
ls -lh test_binMat
```

**Document binary size:** Track size without OpenCV for embedded use.

#### 0.8 Documentation Updates

1. **This file (ROADMAP.md):** ✓ Adding Phase 0
2. **ARCHITECTURE.md:** ✓ Updated with embedded support
3. **Create `docs/EMBEDDED.md`:** Build guide for embedded platforms
4. **Update README.md:** Add embedded use case examples

**Phase 0 Completion Criteria:**
- ✅ Core builds without OpenCV (`-DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON`)
- ✅ Desktop workflow unchanged (just add `#include <bincv/opencv.hpp>`)
- ✅ All existing tests pass
- ✅ External buffer wrapping works
- ✅ Cross-compiles for ARM
- ✅ Binary size measured (<100KB for core + scalar ops)
- ✅ CMake auto-configuration working

**Estimated Effort:** 2-3 days

---

### Phase 1: Foundation & Correctness

**Goal:** Fix bugs, establish testing, optimize critical operations

**Note:** Phase 0 must be completed first.

#### 1.1 Fix Critical Bugs

**Priority: CRITICAL - Must be done first**

1. **Fix `fromCVMat()` undefined variables**
   - Location: [binMat_impl.hpp:42](bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp)
   - Problem: References undefined `mat_` instead of `mat`
   - Fix: Change all `mat_` to `mat` in function body

2. **Fix template specialization scope issues**
   - Location: [binMat_impl.hpp:156+](bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp)
   - Problem: Missing `BinMat::` scope qualifier in template specializations
   - Fix: Add `BinMat<WordSize>::` prefix to all specialized methods

3. **Fix `forEachNonZero()` namespace error**
   - Location: [binMat_impl.hpp:345](bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp)
   - Problem: References `detail::bitMask` instead of `impl::bitMask`
   - Fix: Change namespace from `detail` to `impl`

**Verification:**
```bash
cd bincv-cpp/build
make clean
make -j$(nproc)  # Should compile without errors
./test_binMat    # Should pass
```

#### 1.2 Testing Infrastructure

**Setup Google Test:**
```bash
cd bincv-cpp
git submodule add https://github.com/google/googletest.git external/googletest
```

**Update CMakeLists.txt:**
```cmake
# Add Google Test
add_subdirectory(external/googletest)
include_directories(external/googletest/googletest/include)

# Link tests with gtest
target_link_libraries(test_binMat gtest gtest_main ${OpenCV_LIBS})
```

**Create test utilities** ([bincv-cpp/tests/test_util.hpp](bincv-cpp/tests/test_util.hpp)):
```cpp
namespace bincv::test {
    // Compare binCV result with OpenCV reference
    void assertMatEqual(const BinMat& actual, const cv::Mat& expected);

    // Generate random binary matrix
    BinMat createRandomBinary(int rows, int cols, float sparsity);

    // Create test patterns
    BinMat createCheckerboard(int rows, int cols, int tileSize);
    BinMat createHorizontalStripes(int rows, int cols, int stripeWidth);
}
```

**Port existing tests to Google Test:**
- Convert manual assertions to `EXPECT_EQ`, `EXPECT_TRUE`, etc.
- Add parameterized tests for multiple sizes/sparsity levels
- Target: 100+ test cases

#### 1.3 Optimize Transpose

**Goal:** Match OpenCV performance (0.003 ms for 256×256)

**Implement cache-blocked transpose:**
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

**Optimization steps:**
1. Implement scalar cache-blocked version
2. Profile to find hotspots
3. Add bit-level optimization (process multiple bits per iteration)
4. Consider SIMD for bit manipulation
5. Benchmark against OpenCV

**Target:** Match or beat OpenCV (0.003 ms for 256×256)

#### 1.4 Core Bitwise Operations

**Implement basic operations:**
```cpp
namespace bincv {
    void bitwise_and(const BinMat& src1, const BinMat& src2, BinMat& dst);
    void bitwise_or(const BinMat& src1, const BinMat& src2, BinMat& dst);
    void bitwise_xor(const BinMat& src1, const BinMat& src2, BinMat& dst);
    void bitwise_not(const BinMat& src, BinMat& dst);
}
```

**Implementation strategy:**
1. **Scalar version** (fallback):
   ```cpp
   void bitwise_and_scalar(const BinMat& src1, const BinMat& src2, BinMat& dst) {
       const uint32_t* p1 = src1.data();
       const uint32_t* p2 = src2.data();
       uint32_t* pd = dst.data();
       size_t n_words = src1.rows() * (src1.step() / sizeof(uint32_t));

       for (size_t i = 0; i < n_words; ++i) {
           pd[i] = p1[i] & p2[i];
       }
   }
   ```

2. **AVX2 version** (256-bit SIMD):
   ```cpp
   void bitwise_and_avx2(const BinMat& src1, const BinMat& src2, BinMat& dst) {
       // Process 8 words (256 bits) at a time
       // See ARCHITECTURE.md section 6.1 for full implementation
   }
   ```

3. **AVX-512 version** (512-bit SIMD):
   ```cpp
   void bitwise_and_avx512(const BinMat& src1, const BinMat& src2, BinMat& dst) {
       // Process 16 words (512 bits) at a time
   }
   ```

4. **Runtime dispatch:**
   ```cpp
   void bitwise_and(const BinMat& src1, const BinMat& src2, BinMat& dst) {
       static Backend backend = detectBestBackend();

       switch (backend) {
           case Backend::AVX512:
               return bitwise_and_avx512(src1, src2, dst);
           case Backend::AVX2:
               return bitwise_and_avx2(src1, src2, dst);
           default:
               return bitwise_and_scalar(src1, src2, dst);
       }
   }
   ```

**Testing:**
- Test against OpenCV for correctness
- Test edge cases (empty, 1×1, non-aligned sizes)
- Benchmark on multiple sizes: 256², 512², 1024², 2048²
- Target: 10× faster than OpenCV

#### 1.5 Documentation Setup

**Configure Doxygen:**
```bash
cd bincv-cpp
doxygen -g Doxyfile
# Edit Doxyfile: set PROJECT_NAME, INPUT, OUTPUT_DIRECTORY
doxygen Doxyfile
```

**Document all public APIs:**
```cpp
/**
 * @brief Performs bitwise AND operation on two binary matrices.
 *
 * Computes element-wise logical AND of src1 and src2.
 * Equivalent to OpenCV's cv::bitwise_and for binary images.
 *
 * @param src1 First input binary matrix
 * @param src2 Second input binary matrix
 * @param dst Output binary matrix (resized to match inputs)
 *
 * @throws std::invalid_argument if src1 and src2 have different sizes
 *
 * @note Performance: ~10× faster than OpenCV for binary images
 *
 * @par Example
 * @code
 * BinMat mask = BinMat::ones(480, 640);
 * BinMat result;
 * bincv::bitwise_and(input, mask, result);
 * @endcode
 */
void bitwise_and(const BinMat& src1, const BinMat& src2, BinMat& dst);
```

**Phase 1 Complete When:**
- ✅ All bugs fixed, code compiles
- ✅ Google Test integrated with 100+ tests
- ✅ Transpose matches OpenCV performance
- ✅ Bitwise operations 10× faster than OpenCV
- ✅ Doxygen documentation configured

---

### Phase 2: Core Vision Operations

**Goal:** Morphology, filtering, SIMD dispatch, comprehensive benchmarks

#### 2.1 Morphological Operations

**Implement core operations:**
```cpp
namespace bincv {
    void erode(const BinMat& src, BinMat& dst, const BinMat& kernel);
    void dilate(const BinMat& src, BinMat& dst, const BinMat& kernel);
    void morphologyEx(const BinMat& src, BinMat& dst, int op, const BinMat& kernel);

    // Convenience function
    BinMat getStructuringElement(int shape, cv::Size ksize);
}
```

**Optimization for 3×3 kernels (most common):**
```cpp
void dilate_3x3(const BinMat& src, BinMat& dst) {
    // Dilate = OR of 9 shifted versions (including original)
    dst = src.clone();

    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;

            BinMat shifted = shift(src, dr, dc); // Implement shift operation
            bitwise_or(dst, shifted, dst);       // Reuse bitwise OR
        }
    }
}
```

**Implement shift operation:**
```cpp
// Shift image by (dr, dc) pixels
BinMat shift(const BinMat& src, int dr, int dc) {
    BinMat dst = BinMat::zeros(src.size());

    // Copy with offset, handling boundaries
    for (int r = 0; r < src.rows(); ++r) {
        for (int c = 0; c < src.cols(); ++c) {
            int sr = r - dr;
            int sc = c - dc;
            if (sr >= 0 && sr < src.rows() && sc >= 0 && sc < src.cols()) {
                dst.set(r, c, src.at(sr, sc));
            }
        }
    }

    return dst;
}
```

**Optimize shift with SIMD:**
- For horizontal shifts: use bit shifts on words
- For vertical shifts: copy rows with offset
- Handle boundary conditions correctly

**Support all morphology operations:**
- `MORPH_ERODE`: Minimum filter
- `MORPH_DILATE`: Maximum filter
- `MORPH_OPEN`: Erode then dilate
- `MORPH_CLOSE`: Dilate then erode
- `MORPH_GRADIENT`: Dilate - erode
- `MORPH_TOPHAT`: Original - open
- `MORPH_BLACKHAT`: Close - original

**Target:** 15-20× faster than OpenCV

#### 2.2 Filtering Operations

**Binary blur (majority filter):**
```cpp
void blur(const BinMat& src, BinMat& dst, cv::Size ksize) {
    // For each pixel, count 1s in window
    // Set output to 1 if count > threshold (e.g., kernel_size/2)

    int threshold = (ksize.width * ksize.height) / 2;

    for (int r = 0; r < src.rows(); ++r) {
        for (int c = 0; c < src.cols(); ++c) {
            int count = 0;

            // Count 1s in window
            for (int kr = 0; kr < ksize.height; ++kr) {
                for (int kc = 0; kc < ksize.width; ++kc) {
                    int rr = r + kr - ksize.height/2;
                    int cc = c + kc - ksize.width/2;
                    if (rr >= 0 && rr < src.rows() && cc >= 0 && cc < src.cols()) {
                        if (src.at(rr, cc)) count++;
                    }
                }
            }

            dst.set(r, c, count > threshold);
        }
    }
}
```

**Optimize with POPCOUNT:**
- Extract window as bit vector
- Use `__builtin_popcountll()` or SIMD POPCOUNT
- Dramatically faster than counting pixels one by one

**Binary median filter:**
```cpp
void medianBlur(const BinMat& src, BinMat& dst, int ksize) {
    // Similar to blur, but use median instead of mean
    int threshold = (ksize * ksize) / 2;
    // Implementation similar to blur
}
```

**Target:** 10× faster than OpenCV

#### 2.3 Runtime Dispatch System

**CPU feature detection:**
```cpp
namespace bincv::simd {
    enum class ISA {
        Scalar,
        SSE2,
        AVX2,
        AVX512,
        NEON
    };

    ISA detect() {
    #ifdef __AVX512F__
        if (__builtin_cpu_supports("avx512f")) {
            return ISA::AVX512;
        }
    #endif
    #ifdef __AVX2__
        if (__builtin_cpu_supports("avx2")) {
            return ISA::AVX2;
        }
    #endif
    #ifdef __SSE2__
        if (__builtin_cpu_supports("sse2")) {
            return ISA::SSE2;
        }
    #endif
    #ifdef __ARM_NEON
        return ISA::NEON;
    #endif
        return ISA::Scalar;
    }
}
```

**Function pointer tables:**
```cpp
namespace bincv::dispatch {
    struct BitwiseOps {
        void (*bitwise_and)(const BinMat&, const BinMat&, BinMat&);
        void (*bitwise_or)(const BinMat&, const BinMat&, BinMat&);
        void (*bitwise_xor)(const BinMat&, const BinMat&, BinMat&);
        void (*bitwise_not)(const BinMat&, BinMat&);
    };

    extern BitwiseOps ops;

    void initDispatch() {
        ISA isa = simd::detect();

        switch (isa) {
            case ISA::AVX512:
                ops = {bitwise_and_avx512, bitwise_or_avx512, ...};
                break;
            case ISA::AVX2:
                ops = {bitwise_and_avx2, bitwise_or_avx2, ...};
                break;
            default:
                ops = {bitwise_and_scalar, bitwise_or_scalar, ...};
        }
    }
}
```

#### 2.4 Padding Operations

**Implement all border types:**
```cpp
void copyMakeBorder(const BinMat& src, BinMat& dst,
                    int top, int bottom, int left, int right,
                    int borderType, bool value = false) {
    dst = BinMat(src.rows() + top + bottom, src.cols() + left + right);

    switch (borderType) {
        case BORDER_CONSTANT:
            // Fill borders with constant value
            break;
        case BORDER_REPLICATE:
            // Replicate edge pixels
            break;
        case BORDER_REFLECT:
            // Reflect pixels at boundary
            break;
        case BORDER_WRAP:
            // Wrap around (periodic)
            break;
    }
}
```

#### 2.5 Comprehensive Benchmarking

**Extend benchmark suite:**
```cpp
// benchmark/morphology_benchmark.cpp
int main() {
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    std::vector<float> sparsities = {0.01, 0.1, 0.5, 0.9, 0.99};

    for (int size : sizes) {
        for (float sparsity : sparsities) {
            BinMat src = createRandomBinary(size, size, sparsity);
            BinMat dst;
            BinMat kernel = getStructuringElement(MORPH_RECT, {3, 3});

            // Benchmark binCV
            auto start = high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                erode(src, dst, kernel);
            }
            auto end = high_resolution_clock::now();
            double bincv_time = duration_cast<microseconds>(end - start).count() / 1000.0 / iterations;

            // Benchmark OpenCV
            cv::Mat cv_src = src.toCVMat();
            cv::Mat cv_dst;
            cv::Mat cv_kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
            start = high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                cv::erode(cv_src, cv_dst, cv_kernel);
            }
            end = high_resolution_clock::now();
            double opencv_time = duration_cast<microseconds>(end - start).count() / 1000.0 / iterations;

            // Report
            std::cout << "Size: " << size << "×" << size
                      << ", Sparsity: " << sparsity
                      << ", binCV: " << bincv_time << " ms"
                      << ", OpenCV: " << opencv_time << " ms"
                      << ", Speedup: " << (opencv_time / bincv_time) << "×\n";
        }
    }
}
```

**Generate performance reports:**
- CSV output for plotting
- Graphs showing speedup vs image size
- Graphs showing speedup vs sparsity
- Memory usage comparison

**Phase 2 Complete When:**
- ✅ Morphology operations 15-20× faster
- ✅ Filtering operations 10× faster
- ✅ Runtime SIMD dispatch working
- ✅ All border types implemented
- ✅ Comprehensive benchmark results

---

### Phase 3: Analysis & GPU Acceleration

**Goal:** Connected components, distance transform, CUDA implementations

#### 3.1 Connected Components

**Union-find algorithm:**
```cpp
int connectedComponents(const BinMat& image, BinMat& labels, int connectivity = 8) {
    // Two-pass algorithm
    // Pass 1: Assign provisional labels
    // Pass 2: Merge equivalent labels

    // Implementation details in ARCHITECTURE.md
}
```

**Optimizations:**
- Parallel union-find with path compression
- Block-based processing for cache efficiency
- SIMD for label propagation

**Target:** 10× faster than OpenCV

#### 3.2 Distance Transform

**Implement Felzenszwalb algorithm:**
```cpp
void distanceTransform(const BinMat& src, cv::Mat& dst, int distanceType, int maskSize) {
    // Fast sequential algorithm for distance transform
    // Two passes: forward and backward
}
```

**Target:** 5-10× faster than OpenCV

#### 3.3 Statistical Operations

**Optimize countNonZero:**
```cpp
int countNonZero(const BinMat& src) {
    // Use SIMD POPCOUNT
    // See ARCHITECTURE.md section 6.3 for full implementation
}
```

**Implement moments:**
```cpp
cv::Moments moments(const BinMat& src) {
    // Compute image moments efficiently
    // m00, m10, m01, m20, m11, m02, etc.
}
```

**Target:** 10× faster for countNonZero

#### 3.4 CUDA Implementations

**Port operations to CUDA:**
```cuda
// bitwise_and.cu
__global__ void bitwise_and_kernel(
    const uint32_t* src1, const uint32_t* src2, uint32_t* dst,
    int rows, int step_words
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = rows * step_words;

    if (idx < total_words) {
        dst[idx] = src1[idx] & src2[idx];
    }
}

void bitwise_and_cuda(const BinMat& src1, const BinMat& src2, BinMat& dst) {
    // Allocate device memory
    // Launch kernel
    // Copy result back
}
```

**Morphology on GPU:**
```cuda
__global__ void dilate_3x3_kernel(
    const uint32_t* src, uint32_t* dst,
    int rows, int cols, int step_words
) {
    // See ARCHITECTURE.md section 6.5 for full implementation
}
```

**Async API:**
```cpp
namespace bincv::cuda {
    class Stream {
    public:
        Stream() { cudaStreamCreate(&stream_); }
        ~Stream() { cudaStreamDestroy(stream_); }

        void erode(const BinMat& src, BinMat& dst, const BinMat& kernel);
        void synchronize() { cudaStreamSynchronize(stream_); }

    private:
        cudaStream_t stream_;
    };
}
```

**Target:** 100× faster on GPU for 2048² and larger

**Phase 3 Complete When:**
- ✅ Connected components 10× faster on CPU
- ✅ Distance transform 5-10× faster on CPU
- ✅ CUDA implementations of core operations
- ✅ GPU performance 100× faster for large images
- ✅ Async CUDA API functional

---

### Phase 4: Advanced Operations & Language Bindings

**Goal:** Python bindings, advanced operations, documentation, examples

#### 4.1 Python Bindings

**Use pybind11:**
```cpp
// python/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(bincv, m) {
    m.doc() = "binCV: Accelerated binary image processing";

    // BinMat class
    py::class_<BinMat>(m, "BinMat")
        .def(py::init<int, int>())
        .def("rows", &BinMat::rows)
        .def("cols", &BinMat::cols)
        .def("at", &BinMat::at)
        .def("set", &BinMat::set);

    // NumPy conversion
    m.def("from_numpy", [](py::array_t<uint8_t> arr) {
        // Convert NumPy array to BinMat
    });

    m.def("to_numpy", [](const BinMat& mat) {
        // Convert BinMat to NumPy array
    });

    // Operations
    m.def("bitwise_and", &bitwise_and);
    m.def("erode", &erode);
    m.def("dilate", &dilate);
}
```

**Setup pip package:**
```python
# setup.py
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "bincv",
        ["python/bindings.cpp"],
        include_dirs=["bincv-cpp/include"],
        libraries=["opencv_core"],
    ),
]

setup(
    name="bincv",
    version="0.1.0",
    ext_modules=ext_modules,
    install_requires=["numpy"],
)
```

#### 4.2 Contour Operations

**Implement findContours:**
```cpp
void findContours(const BinMat& image,
                  std::vector<std::vector<cv::Point>>& contours,
                  int mode, int method) {
    // Trace contours in binary image
    // Use OpenCV-compatible algorithm
}
```

#### 4.3 Template Matching

**Implement matchTemplate:**
```cpp
void matchTemplate(const BinMat& image, const BinMat& templ,
                   cv::Mat& result, int method) {
    // For TM_CCORR: Use POPCOUNT of AND operation
    // result(r,c) = popcount(image_window(r,c) AND template)
}
```

**Optimize with FFT for large templates:**
- Use FFT-based correlation for templates > 16×16
- Use direct POPCOUNT for smaller templates

**Target:** 10× faster than OpenCV

#### 4.4 Example Applications

**Create examples directory:**
```
examples/
├── spad_camera.cpp        # Process SPAD camera stream
├── event_camera.cpp       # Event camera frame processing
├── document_morph.cpp     # Document morphology
├── depth_discontinuity.cpp # Depth sensing
└── template_match.cpp     # Binary pattern matching
```

**SPAD camera example:**
```cpp
// examples/spad_camera.cpp
#include <bincv/bincv.hpp>

int main() {
    // Load stream of binary frames
    std::vector<BinMat> frames = loadSPADStream("data/spad_stream.bin");

    // Accumulate frames
    BinMat accumulated = BinMat::zeros(frames[0].size());
    for (const auto& frame : frames) {
        bincv::bitwise_or(accumulated, frame, accumulated);
    }

    // Denoise with morphology
    BinMat kernel = bincv::getStructuringElement(bincv::MORPH_RECT, {3, 3});
    bincv::morphologyEx(accumulated, accumulated, bincv::MORPH_OPEN, kernel);

    // Find objects
    BinMat labels;
    int n = bincv::connectedComponents(accumulated, labels);
    std::cout << "Found " << n << " objects\n";

    // Display
    cv::imshow("Result", accumulated.toDisplay());
    cv::waitKey(0);
}
```

#### 4.5 Documentation

**Write tutorials:**
- Getting started guide
- Migration from OpenCV
- Performance optimization
- CUDA programming with binCV

**Generate API reference:**
- Configure Doxygen
- Generate HTML documentation
- Host on GitHub Pages

**Phase 4 Complete When:**
- ✅ Python bindings working
- ✅ Pip package installable
- ✅ Contour operations implemented
- ✅ Template matching 10× faster
- ✅ 5+ example applications
- ✅ Complete documentation

---

### Phase 5: Platform Expansion & Advanced Backends

**Goal:** ARM support, framework integration, additional GPU backends

#### 5.1 ARM NEON Backend

**Implement NEON SIMD:**
```cpp
#ifdef __ARM_NEON
void bitwise_and_neon(const BinMat& src1, const BinMat& src2, BinMat& dst) {
    const uint32_t* p1 = src1.data();
    const uint32_t* p2 = src2.data();
    uint32_t* pd = dst.data();
    size_t n_words = src1.rows() * (src1.step() / sizeof(uint32_t));

    size_t i = 0;
    // Process 4 words (128 bits) at a time
    for (; i + 4 <= n_words; i += 4) {
        uint32x4_t v1 = vld1q_u32(p1 + i);
        uint32x4_t v2 = vld1q_u32(p2 + i);
        uint32x4_t vd = vandq_u32(v1, v2);
        vst1q_u32(pd + i, vd);
    }

    // Scalar remainder
    for (; i < n_words; ++i) {
        pd[i] = p1[i] & p2[i];
    }
}
#endif
```

#### 5.2 Framework Integration

**PyTorch custom operator:**
```cpp
// pytorch/bincv_ops.cpp
#include <torch/extension.h>

torch::Tensor erode_torch(torch::Tensor input, torch::Tensor kernel) {
    // Convert torch::Tensor to BinMat
    // Call binCV erode
    // Convert result back to torch::Tensor
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("erode", &erode_torch);
}
```

**Usage:**
```python
import torch
import bincv.torch as bcv_torch

x = torch.randint(0, 2, (1, 1, 1024, 1024), dtype=torch.uint8)
kernel = bcv_torch.get_structuring_element('rect', (3, 3))
y = bcv_torch.erode(x, kernel)
```

#### 5.3 Additional GPU Backends

**Vulkan compute shader:**
```glsl
// vulkan/bitwise_and.comp
#version 450

layout(binding = 0) readonly buffer Input1 { uint data1[]; };
layout(binding = 1) readonly buffer Input2 { uint data2[]; };
layout(binding = 2) writeonly buffer Output { uint data_out[]; };

layout(local_size_x = 256) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    data_out[idx] = data1[idx] & data2[idx];
}
```

#### 5.4 Profiling Infrastructure

**Built-in profiler:**
```cpp
namespace bincv {
    class Profiler {
    public:
        void start(const std::string& name);
        void stop(const std::string& name);
        void report() const;

    private:
        std::unordered_map<std::string, std::vector<double>> timings_;
    };
}
```

**Usage:**
```cpp
bincv::Profiler profiler;

profiler.start("erode");
bincv::erode(src, dst, kernel);
profiler.stop("erode");

profiler.start("dilate");
bincv::dilate(src, dst, kernel);
profiler.stop("dilate");

profiler.report();
// Output:
// erode: 0.05 ms (avg over 1 calls)
// dilate: 0.05 ms (avg over 1 calls)
```

**Phase 5 Complete When:**
- ✅ ARM NEON backend functional
- ✅ PyTorch/TensorFlow operators working
- ✅ Vulkan or Metal backend implemented
- ✅ Multi-GPU support for CUDA
- ✅ Built-in profiling infrastructure

---

## Performance Targets

| Operation | OpenCV (uint8) | binCV Target | Status |
|-----------|----------------|--------------|--------|
| Fill | 0.003 ms | 0.002 ms | ✅ Achieved |
| Transpose | 0.003 ms | 0.003 ms | ❌ 50× slower (needs fix) |
| Bitwise AND | 0.05 ms | 0.005 ms | ⚪ Planned (10×) |
| Erode 3×3 | 0.8 ms | 0.05 ms | ⚪ Planned (16×) |
| Dilate 3×3 | 0.8 ms | 0.05 ms | ⚪ Planned (16×) |
| countNonZero | 0.1 ms | 0.01 ms | ⚪ Planned (10×) |
| Connected Components | 5.0 ms | 0.5 ms | ⚪ Planned (10×) |
| Distance Transform | 2.0 ms | 0.2 ms | ⚪ Planned (10×) |

---

## Implementation Priorities

### Critical Path (Must Follow Order)

1. **Phase 0: Core Refactoring** → Remove OpenCV dependency, enable embedded
2. **Fix Bugs** → Required for compilation
3. **Testing** → Required for validation
4. **Transpose** → Foundational, high-impact
5. **Bitwise Ops** → Building blocks
6. **Morphology** → Most important operations
7. **SIMD Dispatch** → Performance multiplier
8. **GPU** → Massive speedup
9. **Advanced Ops** → Build on foundation
10. **Python** → Enable broader usage

### Dependency Graph

```
Phase 0: Core Refactoring (Remove OpenCV dependency)
      ↓
Testing Framework
      ↓
Bug Fixes → Transpose → Bitwise Ops → Morphology → Filtering
                             ↓            ↓          ↓
                       SIMD Dispatch ─────┴──────────┘
                             ↓
                       GPU Backend
                             ↓
                 Connected Components, Distance
                             ↓
                    Advanced Operations
                             ↓
                     Python Bindings
```

### Platform Support Timeline

| Phase | Desktop | High-End Embedded | Mid-Range Embedded |
|-------|---------|-------------------|-------------------|
| **Phase 0** | Core working | Core working | Core working |
| **Phase 1** | + Optimized ops | + Optimized ops | + Scalar ops |
| **Phase 2** | + SIMD (AVX2) | + SIMD (NEON) | Optional SIMD |
| **Phase 3** | + CUDA | + CUDA (Jetson) | - |
| **Phase 4** | + Python | Optional Python | - |
| **Phase 5** | Full features | Full features | Core features |

---

## Quick Start Commands

### Build (Desktop with OpenCV)
```bash
cd bincv-cpp/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Build (Embedded without OpenCV)
```bash
cd bincv-cpp/build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON ..
make -j$(nproc)
```

### Build (Cross-compile for ARM)
```bash
mkdir build-arm && cd build-arm
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/arm-linux-gnueabihf.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON
make -j$(nproc)
```

### Test
```bash
./test_binMat
```

### Benchmark
```bash
cd ../scripts
./run_all_benchmarks.sh
```

### Profile
```bash
perf record -g ./transpose_benchmark
perf report
```

---

## Next Actions

### Immediate (Phase 0 - Core Refactoring)
1. Create `bincv/core/types.hpp` with Size struct and enums
2. Refactor BinMat to use `std::vector` instead of `cv::Mat`
3. Add external buffer constructor for embedded
4. Create `bincv/opencv.hpp` with conversion functions
5. Update CMake for optional OpenCV
6. Update all existing code to use `bincv::Size`
7. Split tests into core and OpenCV-dependent
8. Cross-compile for ARM and measure binary size

### Next Steps (Phase 1)
1. Fix 3 critical compilation bugs
2. Set up Google Test framework
3. Create golden test files
4. Implement cache-blocked transpose
5. Implement scalar bitwise operations

### Future Work
1. SIMD implementations (AVX2, NEON)
2. Morphology operations
3. GPU CUDA backend
4. Python bindings
5. Advanced operations
