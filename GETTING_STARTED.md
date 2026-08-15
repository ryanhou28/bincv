# Getting Started with binCV Development

Practical guide for building, testing, and contributing to binCV.

**Read first:** [ARCHITECTURE.md](ARCHITECTURE.md) for the design and its
rationale, [ROADMAP.md](ROADMAP.md) for what to work on next.

---

## The one thing to keep in mind

binCV exists to answer whether bit-parallel software can make low-bit-width image
processing efficient enough for embedded and mobile deployment. Two goals are
co-equal: **performance and memory footprint**. When they conflict and no
explicit choice has been made, **memory wins** — a user who wants raw throughput
and has memory to spare already has OpenCV.

If a change stores more bits per pixel than the data contains, or adds an
operation no VIO frontend calls, it is probably drift.

---

## Environment Setup

### Required
- C++17 compiler (GCC 7+, Clang 5+)
- CMake 3.15+

That is all the core needs — it has no dependencies.

### Optional
- **OpenCV 4.0+** — enables interop, the equivalence harness, and comparison
  benchmarks. Strongly recommended for development.
- **aarch64 cross-compiler** — for the primary target platform.

### Linux / WSL

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev git

# For the primary target platform
sudo apt-get install -y g++-aarch64-linux-gnu
```

### macOS

```bash
brew install cmake opencv
```

---

## Build Configurations

### Desktop (OpenCV auto-detected)

```bash
cmake -S bincv-cpp -B bincv-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build bincv-cpp/build -j$(nproc)
cd bincv-cpp/build && ctest --output-on-failure
```

### Core-only (no OpenCV)

```bash
cmake -S bincv-cpp -B bincv-cpp/build-core \
      -DCMAKE_BUILD_TYPE=Release -DBINCV_USE_OPENCV=OFF
cmake --build bincv-cpp/build-core -j$(nproc)
cd bincv-cpp/build-core && ctest --output-on-failure
```

Run this before committing. It is what keeps the core dependency-free, and it
regresses silently otherwise.

### No exceptions / no heap (Tier 2 correctness)

```bash
cmake -S bincv-cpp -B bincv-cpp/build-noexcept \
      -DBINCV_USE_OPENCV=OFF -DCMAKE_CXX_FLAGS="-fno-exceptions"
```

binCV commits to compiling and running correctly in this configuration. See
[ARCHITECTURE §2](ARCHITECTURE.md#tier-2--cortex-m-class-correctness-only).

CMake prints a configuration summary showing platform, build type, SIMD
capability, and whether OpenCV interop is enabled.

**Options:** `BINCV_USE_OPENCV` (default ON), `BINCV_BUILD_TESTS` (ON),
`BINCV_BUILD_BENCHMARKS` (ON, skipped automatically without OpenCV).

---

## Testing

Tests are registered with `ctest`. The suites are split so the embedded
configuration is actually exercised rather than assumed:

- `tests/test_binMat.cpp` — core suite, **no OpenCV**. Runs the behavioural
  contract against all supported word widths.
- `tests/test_storage.cpp` — the storage layer on its own.
- `tests/test_error.cpp` — the error policy as compiled in *this* build.
- `tests/test_error_checked.cpp` — the error policy with checks forced **on**.
- `tests/test_opencv_interop.cpp` — `cv::Mat` round-trips, built only with OpenCV.

Add core tests to the first and OpenCV-dependent tests to the last.

The interim harness (`tests/test_util.hpp`) reports failures with file and line
and returns a non-zero exit code. Google Test replaces it in Phase 1.6.

### Tests for checks that kill the process

A failed binCV check terminates the process — it throws where exceptions exist
and calls `std::abort()` where they do not ([§5.3](ARCHITECTURE.md#53-error-policy))
— so no assertion inside that process can observe it. Those cases are **death
tests**:

- `tests/test_error_abort.cpp` — one case per `BINCV_THROW` site.
- `tests/test_assert_abort.cpp` — one case per `BINCV_ASSERT` site.
- `tests/expect_fatal.cmake` runs a case as its own process and passes only if it
  terminated *abnormally* **and** printed the expected diagnostic. A clean
  non-zero return is a failure, not a pass.

Adding a validation check means adding a case to `test_error_abort.cpp` and
registering it in `tests/CMakeLists.txt` with the message it must print.
`BINCV_CHECK_THROWS` is not a substitute: without exceptions it cannot evaluate
its expression, so it reports a SKIP and covers nothing.

Two of these suites `#undef NDEBUG` before their includes, which forces
`BINCV_DEBUG_CHECKS` on. That is deliberate: every configuration the project
verifies is Release, so the debug-checked half of the policy would otherwise
never be compiled at all.

### Correctness standards

What "correct" means depends on the API tier
([ARCHITECTURE §5.1](ARCHITECTURE.md#51-three-tiers)):

| Tier | Standard |
|---|---|
| 1 — identical semantics | **Bit-exact** against the equivalent OpenCV expression |
| 2 — specialized numerics | Downstream task accuracy (VIO trajectory) preserved |
| 3 — no OpenCV equivalent | Against hand-derived reference implementations |

Every Tier 1 operation ships with an equivalence test. That harness is Phase 2.1
and is built *before* the kernels it validates.

---

## Benchmarking

```bash
cd bincv-cpp/build
./benchmark/fill_benchmark --width 640 --height 480 --iterations 100 \
                           --dtype binary --sparsity 0.5
```

Or run the full sweep:

```bash
cd bincv-cpp/scripts && ./run_all_benchmarks.sh
```

### Benchmarking rules

**Always build Release.** An unoptimized build makes the numbers meaningless.
CMake defaults to Release for this reason.

**Use the right denominator.** Compare against OpenCV performing the *same
semantic operation on the same binary content stored as `CV_8U`* — that is
exactly what a user does today without binCV. Not OpenCV on grayscale (different
information content), and not a strawman implementation.

**Report peak working set, not per-buffer ratios.** A target either fits the
pipeline in its memory budget or it does not
([ARCHITECTURE §10.4](ARCHITECTURE.md#104-the-metric-that-matters)).

**Commit the measurement.** Every performance claim in this repository must be
reproducible from a committed benchmark.

---

## Code Tour

### Core
- [bincv-cpp/include/bincv-cpp/core/types.hpp](bincv-cpp/include/bincv-cpp/core/types.hpp) — `Size`, morphology and border enums, type aliases
- [bincv-cpp/include/bincv-cpp/core/storage.hpp](bincv-cpp/include/bincv-cpp/core/storage.hpp) — owning or caller-provided backing memory
- [bincv-cpp/include/bincv-cpp/core/view.hpp](bincv-cpp/include/bincv-cpp/core/view.hpp) — `BinMatView` / `BinMatConstView`, the kernel interface
- [bincv-cpp/include/bincv-cpp/core/error.hpp](bincv-cpp/include/bincv-cpp/core/error.hpp) — `BINCV_THROW` / `BINCV_ASSERT`, the error policy
- [bincv-cpp/include/bincv-cpp/binMat.hpp](bincv-cpp/include/bincv-cpp/binMat.hpp) — the 1-bit container
- [bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp](bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp) — template implementation

### Support
- [bincv-cpp/include/bincv-cpp/util.hpp](bincv-cpp/include/bincv-cpp/util.hpp) — image I/O for tests (OpenCV-only)
- [bincv-cpp/tests/](bincv-cpp/tests/) — test suites
- [bincv-cpp/benchmark/](bincv-cpp/benchmark/) — comparison benchmarks

### Current shape

```
storage {ptr, stride, owns}          <- Phase 1.1
  |
  +-- BinMatView / QuantView<N>      <- Phase 1.2, the kernel interface
  |
  +-- QuantMat<N, WordType>          <- Phase 1.3, compile-time N
        |
        +-- BinMat<WordType>         <- exists today; becomes the N=1 specialization
```

Only `BinMat` exists today, and it still embeds a `std::vector` rather than the
storage model. See [ROADMAP Phase 1](ROADMAP.md#phase-1--container-foundation).

---

## Conventions

### Naming
Follow OpenCV: `camelCase` functions, `PascalCase` types, `UPPER_CASE` constants,
lowercase namespaces, destination as out-parameter — `op(src, dst, ...)`.

Tier 3 operations (no OpenCV equivalent) must **not** borrow OpenCV names, so
that Tier 1's drop-in promise stays credible.

### Errors
Validation throws; `BINCV_NO_EXCEPTIONS` converts to assert/abort. `at()` is
bounds-checked in debug and unchecked in release, matching `cv::Mat::at`. Kernels
never throw.

### Interfaces
**Kernels take views, never owning containers.** A kernel compiles once per
`(WordType, N)` and works regardless of its arguments' alignment or ownership.

**Never expose a per-word popcount.** Reductions are bulk-only — region, masked,
or windowed. On aarch64 a per-word popcount pays two register-domain crossings
per 64 pixels ([ARCHITECTURE §6.2](ARCHITECTURE.md#62-reductions-are-bulk-only)).

### Documentation

```cpp
/**
 * @brief Bitwise AND of two binary matrices.
 *
 * @param src1 First input
 * @param src2 Second input
 * @param dst  Output, resized to match
 *
 * @note API tier 1 - bit-exact against cv::bitwise_and on equivalent content.
 */
void bitwiseAnd(const BinMatView& src1, const BinMatView& src2, BinMatView dst);
```

State the API tier in the docstring. It tells a reader whether OpenCV
equivalence is a guarantee or explicitly not one.

---

## Profiling

```bash
perf record -g ./benchmark/transpose_benchmark
perf report
```

For memory, peak working set is the number that matters:

```bash
/usr/bin/time -v ./your_benchmark 2>&1 | grep "Maximum resident"
valgrind --tool=massif ./your_benchmark
```

---

## Adding an Operation

1. **Check it is in scope.** Is it called by a binary-frame VIO frontend? If not,
   it likely belongs in [ROADMAP Phase 6](ROADMAP.md#phase-6--deferred).
2. **Determine its API tier** and name it accordingly.
3. **Write the equivalence or reference test first.**
4. **Express it in the primitive vocabulary** — logic, shift, majority,
   thresholded count, bulk reduction. If it does not decompose into those, that
   is worth understanding before writing it.
5. **Take views, not containers.**
6. **Benchmark against the right denominator**, and commit the benchmark.

---

## Learning Resources

### Binary and morphological image processing
- Digital Image Processing (Gonzalez & Woods)
- [Mathematical morphology](https://en.wikipedia.org/wiki/Mathematical_morphology)

### Bit manipulation
- Hacker's Delight (Warren) — the reference for bit-parallel algorithms
- [Faster Population Counts Using AVX2](https://arxiv.org/abs/1611.07612) (Muła, Kurz, Lemire)

### SIMD
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Agner Fog's optimization manuals](https://www.agner.org/optimize/)

### Performance engineering
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)
