# Getting Started with binCV Development

Quick start guide for developing binCV, an accelerated binary image processing library.

---

## What is binCV?

binCV is a computer vision library optimized for binary (1-bit) images, targeting:
- **SPAD cameras**: 1000s of fps binary frames
- **Event cameras**: Binary event frame representations
- **Document processing**: Binarized text images
- **Depth sensing**: Binary masks and patterns

**Goal:** Achieve 10-100× speedup over OpenCV by exploiting bit-packed storage, bitwise operations, SIMD, and GPU parallelism.

---

## Essential Reading

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete technical architecture and design
2. **[ROADMAP.md](ROADMAP.md)** - Concrete implementation tasks
3. This document - Quick start guide

---

## Development Environment Setup

### Prerequisites

**Required:**
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.12+
- OpenCV 4.0+

**Optional:**
- CUDA Toolkit 11.0+ (for GPU acceleration)
- Google Test (will be added as submodule)

### Linux/WSL Setup

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev git

# Optional: CUDA toolkit
# Follow: https://developer.nvidia.com/cuda-downloads

# Navigate to repository
cd /path/to/bincv
```

### macOS Setup

```bash
# Install dependencies via Homebrew
brew install cmake opencv git

cd /path/to/bincv
```

### Windows (WSL Recommended)

Use Windows Subsystem for Linux and follow Linux setup above.

---

## Building binCV

### C++ Library (CPU)

```bash
cd bincv-cpp
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run tests
./test_binMat
```

### C++ Library with CUDA (GPU)

```bash
cd bincv-cuda
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run tests
./test_edge_filter
```

### Build Options

```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build
cmake -DCMAKE_BUILD_TYPE=Release ..

# Specific compiler
cmake -DCMAKE_CXX_COMPILER=clang++ ..

# Verbose
make VERBOSE=1
```

---

## Running Tests & Benchmarks

### Current Tests

```bash
cd bincv-cpp/build
./test_binMat
```

### Run All Benchmarks

```bash
cd bincv-cpp/scripts
./run_all_benchmarks.sh

# Results saved to bincv-cpp/results/
cat ../results/fill_benchmark_*.log
cat ../results/transpose_benchmark_*.log
```

### Run Single Benchmark

```bash
cd bincv-cpp/build
./fill_benchmark
./transpose_benchmark
```

---

## Current Performance

Measured in **Release** mode, sparsity 0.5, 100 iterations. Earlier numbers in
`results/*.log` were produced without a pinned build type and are not comparable.

**256x256**

| Operation | OpenCV | binCV | Ratio |
|-----------|--------|-------|-------|
| fill | 0.0024 ms | 0.00019 ms | 12.6x faster |
| set 1000 px | 0.0033 ms | 0.0010 ms | 3.2x faster |
| resize | 0.069 ms | 0.200 ms | 2.9x slower |
| transpose | 0.0029 ms | 0.208 ms | **71x slower** |

**1024x1024**

| Operation | OpenCV | binCV | Ratio |
|-----------|--------|-------|-------|
| fill | 0.0225 ms | 0.0021 ms | 10.5x faster |
| set 1000 px | 0.0036 ms | 0.0029 ms | 1.2x faster |
| resize | 0.540 ms | 3.62 ms | 6.7x slower |
| transpose | 0.110 ms | 3.97 ms | **36x slower** |

**Memory:** 7.83x reduction on the sample image (46 KB vs 361 KB), close to the
theoretical 8x -- the gap is row-stride alignment padding.

### Reading these numbers

`fill` is the only operation that currently exploits bit packing: it writes whole
words over contiguous storage, so it gets roughly the memory-traffic win the
architecture predicts.

Everything else still loops **pixel by pixel**, paying bit-extraction cost per
pixel while OpenCV moves whole bytes. `transpose` and `resize` are slower than
OpenCV for exactly this reason, and no amount of tuning fixes that shape -- they
need bit-parallel algorithms (Phase 1.3), not micro-optimization.

So the 10-100x thesis is **not yet demonstrated**. The operations that would
demonstrate it -- bitwise AND/OR/XOR, popcount-based `countNonZero`, morphology
-- are not implemented. `fill` is the existence proof that the approach works
when an operation is written word-wise.

---

## Build Configurations

### Desktop (OpenCV auto-detected)
```bash
cmake -S bincv-cpp -B bincv-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build bincv-cpp/build -j$(nproc)
cd bincv-cpp/build && ctest --output-on-failure
```

### Core-only / embedded (no OpenCV)
```bash
cmake -S bincv-cpp -B bincv-cpp/build-noocv \
      -DCMAKE_BUILD_TYPE=Release -DBINCV_USE_OPENCV=OFF
cmake --build bincv-cpp/build-noocv -j$(nproc)
cd bincv-cpp/build-noocv && ctest --output-on-failure
```

CMake prints a configuration summary showing the detected platform, build type,
SIMD capability, and whether OpenCV interop is enabled.

**Options:** `BINCV_USE_OPENCV` (default ON), `BINCV_BUILD_TESTS` (ON),
`BINCV_BUILD_BENCHMARKS` (ON, skipped automatically without OpenCV).

Benchmarks are only built with OpenCV, since they measure against it.

---

## Testing

Tests use a minimal in-repo harness (`tests/test_util.hpp`) that reports
failures with file/line and returns a non-zero exit code, so `ctest` catches
regressions. Google Test is planned for Phase 1.2.

- `tests/test_binMat.cpp` — core suite, no OpenCV. Runs the full behavioural
  contract against all four word widths.
- `tests/test_opencv_interop.cpp` — cv::Mat round-trips, built only with OpenCV.

Add core tests to the first and OpenCV-dependent ones to the second; keeping
them separate is what makes the embedded build verifiable.

---

## Code Tour

### Key Files

**Core Data Structure:**
- [bincv-cpp/include/bincv-cpp/binMat.hpp](bincv-cpp/include/bincv-cpp/binMat.hpp) - Main class declaration
- [bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp](bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp) - Template implementation

**Utilities:**
- [bincv-cpp/include/bincv-cpp/util.hpp](bincv-cpp/include/bincv-cpp/util.hpp) - Helper functions

**Tests:**
- [bincv-cpp/tests/test_binMat.cpp](bincv-cpp/tests/test_binMat.cpp) - Test suite

**Benchmarks:**
- [bincv-cpp/benchmark/](bincv-cpp/benchmark/) - Performance benchmarks
- [bincv-cpp/benchmark/bench_util.hpp](bincv-cpp/benchmark/bench_util.hpp) - Benchmark utilities

**CUDA:**
- [bincv-cuda/src/edge_filter.cu](bincv-cuda/src/edge_filter.cu) - GPU kernels

### Code Architecture

```
BinMat<WordType>          // WordType = uint8_t/uint16_t/uint32_t (default)/uint64_t
├── Storage: std::vector<WordType> (no OpenCV dependency)
├── Layout: Row-major, bit-packed into words
├── Alignment: 32 bytes (cache-line aligned)
└── Operations:
    ├── Construction & conversion
    ├── Element access (slow - bit unpacking)
    ├── Row access (fast - direct pointer)
    └── Matrix operations (resize, transpose, pad)
```

---

## Development Workflow

### Phase 1 Priorities

**1. Fix Critical Bugs**
- Fix 3 compilation bugs listed above
- Verify tests pass

**2. Setup Testing**
- Add Google Test framework
- Create test utilities
- Add 20+ correctness tests

**3. Optimize Transpose**
- Implement cache-blocked algorithm
- Target: Match OpenCV (0.003 ms for 256×256)

**4. Implement Bitwise Operations**
- AND, OR, XOR, NOT
- Scalar + SIMD variants (AVX2, AVX-512)
- Target: 10× faster than OpenCV

### Code Style

**Follow OpenCV Conventions:**
- Functions: `camelCase`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Namespaces: `lowercase`

**Documentation Example:**
```cpp
/**
 * @brief Performs bitwise AND operation on two binary matrices.
 *
 * @param src1 First input binary matrix
 * @param src2 Second input binary matrix
 * @param dst Output binary matrix
 *
 * @throws std::invalid_argument if src1 and src2 have different sizes
 *
 * @note Performance: ~10× faster than OpenCV
 *
 * @code
 * BinMat mask = BinMat::ones(480, 640);
 * BinMat result;
 * bincv::bitwise_and(input, mask, result);
 * @endcode
 */
void bitwise_and(const BinMat& src1, const BinMat& src2, BinMat& dst);
```

---

## Debugging Tips

### Common Issues

**Compilation errors:**
```bash
# Make sure you're including impl headers
#include "bincv-cpp/binMat.hpp"  // Includes impl/binMat_impl.hpp
```

**Slow performance:**
```bash
# Build in Release mode
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Test failures:**
```bash
# Validate against OpenCV
BinMat result = bincv::transpose(input);
cv::Mat cv_result;
cv::transpose(input.toCVMat(), cv_result);
assertMatEqual(result, cv_result);
```

### Profiling Performance

**Linux with perf:**
```bash
perf record -g ./transpose_benchmark
perf report
perf annotate
```

**Intel VTune:**
```bash
vtune -collect hotspots -result-dir vtune_results -- ./transpose_benchmark
vtune -report hotspots -result-dir vtune_results
```

**gprof:**
```bash
cmake -DCMAKE_CXX_FLAGS="-pg" ..
make -j$(nproc)
./transpose_benchmark
gprof ./transpose_benchmark gmon.out > analysis.txt
```

---

## Testing Guidelines

### Write Tests First (TDD)

```cpp
#include <gtest/gtest.h>
#include "bincv-cpp/binMat.hpp"

TEST(BitwiseTest, AndOperation) {
    BinMat src1 = createTestMatrix({{1, 0, 1}, {0, 1, 0}});
    BinMat src2 = createTestMatrix({{1, 1, 0}, {0, 1, 1}});
    BinMat expected = createTestMatrix({{1, 0, 0}, {0, 1, 0}});

    BinMat dst;
    bincv::bitwise_and(src1, src2, dst);

    EXPECT_EQ(dst, expected);
}

TEST(BitwiseTest, AndAgainstOpenCV) {
    BinMat src1 = createRandomBinary(256, 256, 0.5);
    BinMat src2 = createRandomBinary(256, 256, 0.5);

    BinMat bincv_result;
    bincv::bitwise_and(src1, src2, bincv_result);

    cv::Mat cv_result;
    cv::bitwise_and(src1.toCVMat(), src2.toCVMat(), cv_result);

    EXPECT_MAT_EQUAL(bincv_result, cv_result);
}
```

### Benchmark Template

```cpp
#include "bench_util.hpp"

int main() {
    BinMat src1 = createRandomBinary(1024, 1024, 0.5);
    BinMat src2 = createRandomBinary(1024, 1024, 0.5);
    BinMat dst;

    // Benchmark binCV
    auto start = high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        bincv::bitwise_and(src1, src2, dst);
    }
    auto end = high_resolution_clock::now();
    double bincv_time = duration_cast<microseconds>(end - start).count() / 1000.0 / 1000;

    // Benchmark OpenCV
    cv::Mat cv_src1 = src1.toCVMat();
    cv::Mat cv_src2 = src2.toCVMat();
    cv::Mat cv_dst;
    start = high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        cv::bitwise_and(cv_src1, cv_src2, cv_dst);
    }
    end = high_resolution_clock::now();
    double opencv_time = duration_cast<microseconds>(end - start).count() / 1000.0 / 1000;

    std::cout << "binCV:  " << bincv_time << " ms\n";
    std::cout << "OpenCV: " << opencv_time << " ms\n";
    std::cout << "Speedup: " << (opencv_time / bincv_time) << "×\n";
}
```

---

## Learning Resources

### Binary Image Processing
- Digital Image Processing (Gonzalez & Woods)
- [Mathematical morphology (Wikipedia)](https://en.wikipedia.org/wiki/Mathematical_morphology)

### SIMD Programming
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Agner Fog's optimization manuals](https://www.agner.org/optimize/)

### CUDA Programming
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Optimization](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Performance Engineering
- Computer Systems: A Programmer's Perspective
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)

---

## Next Steps

### Immediate Actions
Phase 0 is done — the core compiles, tests pass, and it builds without OpenCV.
Next up (Phase 1):
1. Implement scalar bitwise operations (AND, OR, XOR, NOT) — the first real test
   of the performance thesis
2. Replace `countNonZero`'s per-pixel loop with word-wise popcount
3. Implement cache-blocked / bit-parallel transpose
4. Swap the in-repo test harness for Google Test

### Short Term
1. Implement cache-blocked transpose
2. Benchmark and optimize transpose
3. Implement scalar bitwise operations
4. Validate against OpenCV
5. Add AVX2 SIMD variants

### Medium Term
1. Implement morphology operations
2. Add CUDA GPU backend
3. Create Python bindings
4. Implement advanced operations
5. Platform expansion (ARM, Metal, Vulkan)

---

## Quick Reference

### Build Commands
```bash
# Build C++
cd bincv-cpp/build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)

# Build CUDA
cd bincv-cuda/build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
```

### Test Commands
```bash
cd bincv-cpp/build
./test_binMat
```

### Benchmark Commands
```bash
cd bincv-cpp/scripts
./run_all_benchmarks.sh
```

### Profile Commands
```bash
cd bincv-cpp/build
perf record -g ./transpose_benchmark
perf report
```

---

For the complete architecture and implementation plan, see [ARCHITECTURE.md](ARCHITECTURE.md) and [ROADMAP.md](ROADMAP.md).
