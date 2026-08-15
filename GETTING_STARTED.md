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

**Before committing, run the gate rather than these by hand:**

```bash
./scripts/verify.sh          # all four configurations, warnings fatal, ~40 s
./scripts/verify_arm.sh      # aarch64 correctness under emulation
```

Run `verify.sh` first: `verify_arm.sh` compares its emulated check counts against
the reference `verify.sh` writes to `bincv-cpp/build-logs/checks-*.txt`, and says
`NOT PERFORMED` rather than `PASS` when that reference is missing or describes a
different tree. It exits **77** when Docker or arm64 emulation is unavailable —
not a failure, and not a pass either.

The individual commands below are for driving one configuration during
development. They do **not** enable `-Werror`; `verify.sh` does.

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

### Debug (the checked configuration)

```bash
cmake -S bincv-cpp -B bincv-cpp/build-debug \
      -DCMAKE_BUILD_TYPE=Debug -DBINCV_USE_OPENCV=OFF
cmake --build bincv-cpp/build-debug -j$(nproc)
cd bincv-cpp/build-debug && ctest --output-on-failure
```

The only configuration where `BINCV_DEBUG_CHECKS` is 1, so it is the only one
that compiles `BINCV_ASSERT` — the bounds checks in `at()` and `set()`, and every
kernel precondition. Everything else defines `NDEBUG` and deletes them.

`verify.sh` does not take that on trust. It reads `BINCV_DEBUG_CHECKS` and
`BINCV_EXCEPTIONS_ENABLED` back out of the `test_error` binary each configuration
built, and fails on a mismatch — an exported `CXXFLAGS=-DNDEBUG` otherwise turns
this configuration into a second copy of core-only, with an identical check
count and every assertion gone.

CMake prints a configuration summary showing platform, build type, SIMD
capability, the warning flags in force, the test backend, and whether OpenCV
interop is enabled.

**Options:** `BINCV_USE_OPENCV` (default ON), `BINCV_BUILD_TESTS` (ON),
`BINCV_BUILD_BENCHMARKS` (ON, skipped automatically without OpenCV),
`BINCV_WERROR` (OFF; `verify.sh` sets it), `BINCV_USE_GTEST`
(`AUTO`/`ON`/`OFF`).

### Warnings

`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion` are on for
every first-party target, from `bincv-cpp/cmake/BincvWarnings.cmake`. They are
deliberately **not** attached to `bincv_core`'s interface — a consumer's warning
policy is theirs to pick.

`-Wconversion` earns its keep here: the container is templated on the word
*type*, so each mask and shift compiles at 8, 16, 32 and 64 bits, and integer
promotion means the narrow instantiations are the ones that silently narrow back.
Cast deliberate narrowing explicitly.

`-Werror` is off by default, so a build mid-edit shows you every diagnostic
instead of stopping at the first. `verify.sh` turns it on, and nothing should be
committed until that is green.

**Linking `bincv_warnings` is not optional.** `bincv_assert_warning_policy()`
runs after every `add_subdirectory()` and fails the configure step, naming the
target, if a first-party target that compiles anything does not link it. A
build-log scan cannot substitute: a target with no warning flags emits no
diagnostics, so there is nothing in the log to find, and the run stays green.
Use `bincv_add_test_target()` and the wiring comes with it.

`verify.sh` proves both halves still bite before it builds anything, via two
throwaway builds that must fail (`-DBINCV_SELFTEST_WARNING=ON` and
`-DBINCV_SELFTEST_UNWIRED=ON`, over `tests/selftest_warning.cpp`). If you ever
need to see what the gate rejects, run those by hand.

---

## Testing

Tests are registered with `ctest`. The suites are split so the embedded
configuration is actually exercised rather than assumed:

- `tests/test_binMat.cpp` — core suite, **no OpenCV**. Runs the behavioural
  contract against all supported word widths.
- `tests/test_storage.cpp` — the storage layer on its own.
- `tests/test_error.cpp` — the error policy as compiled in *this* build.
- `tests/test_error_checked.cpp` — the error policy with checks forced **on**.
- `tests/test_quantMat.cpp` — `QuantMat<N>` / `SignedQuantMat<N>` bit planes.
- `tests/test_harness.cpp` — the harness's own regression check, built twice:
  once ordinary, once with a case that fails on purpose and is registered
  `WILL_FAIL`, so "a failing check exits non-zero" is verified rather than
  assumed.
- `tests/test_opencv_interop.cpp` — `cv::Mat` round-trips, built only with OpenCV.

Add core tests to the first and OpenCV-dependent tests to the last.

### Writing a test

```cpp
#include "test_util.hpp"

BINCV_TEST(SuiteName, CaseName) {
    BINCV_CHECK(condition);
    BINCV_CHECK_EQ(actual, expected);
    BINCV_CHECK_THROWS(expr, std::invalid_argument);
}

BINCV_TEST_MAIN("Suite label")     // once per test binary, last line
```

`tests/test_util.hpp` runs these over **two backends**, and the suite source does
not know which:

- **Google Test** in the OpenCV, core-only and Debug configurations. `BINCV_TEST`
  is `TEST`, failures carry the original file and line, `--gtest_filter` works.
- **The built-in harness** in the dependency-free configuration (core-only,
  `-fno-exceptions`), where the cases are registered at static init and run by a
  shared `main`. A bare substring argument filters, the same way.

Both count checks and print the same `N/M checks passed` summary, which is what
lets `verify.sh` compare configurations and catch a suite quietly losing
assertions.

`verify.sh` discovers the suites from `ctest --show-only=json-v1` rather than
from a list, so a suite appended to `BINCV_CORE_TESTS` is picked up with no other
edit. It then compares each suite's count against
**`tests/expected-checks.txt`** — a committed floor, per configuration, per
suite — and fails on a drop, on a listed suite that did not run, and on a suite
that ran with no row of its own. Adding a suite therefore needs one line there
too; `./scripts/verify.sh --update-checks-baseline` writes it, and the diff is
meant to be reviewed rather than rubber-stamped.

Google Test is obtained by `FetchContent` with `FIND_PACKAGE_ARGS`, so an
installed copy satisfies it without a download. The dependency-free
configuration declines it on purpose — not because it cannot compile there (it
can; `-DBINCV_USE_GTEST=ON` works) but because that configuration exists to show
binCV needs nothing but a C++17 compiler.

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
- [bincv-cpp/include/bincv-cpp/binMat.hpp](bincv-cpp/include/bincv-cpp/binMat.hpp) — the 1-bit container, which is `QuantMat<1>`
- [bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp](bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp) — template implementation
- [bincv-cpp/include/bincv-cpp/quantMat.hpp](bincv-cpp/include/bincv-cpp/quantMat.hpp) — the N-bit container, and its signed / ternary reading

### Support
- [bincv-cpp/include/bincv-cpp/util.hpp](bincv-cpp/include/bincv-cpp/util.hpp) — image I/O for tests (OpenCV-only)
- [bincv-cpp/tests/](bincv-cpp/tests/) — test suites
- [bincv-cpp/benchmark/](bincv-cpp/benchmark/) — comparison benchmarks

### Current shape

```
Storage {ptr, words, owns}           <- T1.1
  |
  +-- BinMatView / BinMatConstView   <- T1.2, the kernel interface
  |
  +-- QuantMat<N, WordType>          <- T1.5, compile-time N, one allocation
        |
        +-- BinMat<WordType>         <- T1.3, the hand-written N=1 specialization
        |
        +-- SignedQuantMat<N, W>     <- T1.6, N magnitude planes + 1 sign plane
              |
              +-- TernaryMat<W>      <- the N=1 case: {-1, 0, +1}
```

`BinMat<WordType>` is an **alias** for `QuantMat<1, WordType>`, not a separate
type: the 1-bit case keeps its hand-written single-plane paths while still being
what a `QuantMat<N>` parameter binds to (ARCHITECTURE
[4.4](ARCHITECTURE.md#44-container-hierarchy)). One consequence under C++17:
class template argument deduction does not see through an alias template, so a
default-word-type container is spelled `BinMat<> m(w, h)`, not `BinMat m(w, h)`.

`SignedQuantMat` adds no storage — it is a reading of a `QuantMat<N+1>`.

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
