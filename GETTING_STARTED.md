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
`BINCV_BUILD_BENCHMARKS` (ON; the OpenCV-comparison benchmarks are skipped
automatically without OpenCV, the binCV-versus-binCV experiment benchmarks still
build),
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

#### Sanitizers, for the kernels where undefined behaviour is the likely bug

`verify.sh` does not run them — four clean builds plus two sanitizer builds is a
gate people stop running — so they are a per-kernel step, taken where the kernel's
own hazard calls for it and recorded in the task's notes when it is.

A bit-packed shift is the clearest case: `x << WordBits` is undefined, and on x86
the natural encoding **masks the shift count** and returns `x` where the algebra
wants 0, so the bug is invisible at every shift distance except the exact multiples
of the word width. Reading the code is not a proof that the guard branch is there;
watching the sanitizer stay quiet, and then watching it fire when the branch is
removed, is.

```bash
cd bincv-cpp
# the shipping configuration: optimized, assertions compiled out
g++ -std=c++17 -O2 -DNDEBUG -fsanitize=undefined -fno-sanitize-recover=all \
    -Iinclude -Itests tests/test_shift.cpp -o /tmp/test_shift_ubsan && /tmp/test_shift_ubsan

# and with BINCV_ASSERT live, plus ASan for the wrapped-buffer cases
g++ -std=c++17 -O1 -g -fsanitize=undefined,address -fno-sanitize-recover=all \
    -Iinclude -Itests tests/test_shift.cpp -o /tmp/test_shift_asan && /tmp/test_shift_asan
```

Both compile without CMake and without OpenCV: the suites' Tier 1 halves are
behind `BINCV_WITH_OPENCV` and the built-in harness is the default backend, so a
sanitizer run needs a compiler and nothing else.

---

## Choosing an operating point

**binCV is fast when its INPUT is narrow.** A bit-sliced kernel's cost scales with the
precision it *reads*, not the precision it writes — so the advantage is a property of
the data, not of the operation. `pyrDown` against `cv::pyrDown`, 640×480, reference
device ([X-46](EXPERIMENTS.md)):

| bits in → out | filter | vs `cv::pyrDown` |
|---|---|---|
| 1 → 3 | box 2×2 | **4.4× faster** |
| 1 → 3 | Gaussian 5×5 | 0.86× — rough parity |
| 8 → 8 | Gaussian 5×5 | **13.7× SLOWER** |

The crossover is **filter-dependent** — a box stops paying above ~4 bits, a 5×5
Gaussian above ~1. There is no single number; the table is the shape, not the rule.

**ON x86, BUILD WITH `-DBINCV_X86_POPCNT=ON`.** Baseline x86-64 predates SSE4.2, so
`POPCNT` is not in the default ISA and `__builtin_popcountll` compiles to a **software
fallback** — measured: zero `popcnt` instructions in the portable binary. binCV counts
bits for a living, and the flag is worth **3.75× on the whole frontend**
([X-57](EXPERIMENTS.md)): 12.9 → 3.4 ms, and from 3.8× slower than OpenCV to **0.91×**,
near parity at 6.23× less memory. It is OFF by default only because it raises the
minimum CPU to 2007–08 hardware, which is the caller's call to make.

**Two things follow, and they are easy to conflate:**

- **The footprint advantage is universal** — 6.23× over an OpenCV frontend
  ([X-49](EXPERIMENTS.md)), identical on every platform, because it is a property of
  the representation.
- **The speed advantage is measured on aarch64**, and on x86 it depends entirely on the
  flag above: **0.27× of OpenCV without it, 0.91× with it** ([X-57](EXPERIMENTS.md)).
  **Benchmark on your target, with your flags** — the default portable build is not
  the configuration to judge binCV by.

**Above the crossover, hand the data to OpenCV.** `QuantMat<N>::toCVMatNormalized` and
`fromCVMat` are the bridge, and the round trip is **3.7× faster than binCV's own 8-bit
path** ([D-42](ARCHITECTURE.md)). Send an operation to OpenCV when
`native_binCV − native_OpenCV` exceeds the conversion tax — which a chain of wide
operations pays only once at each end. Matching OpenCV's *output precision* is what
costs; matching its *filter* is nearly free.

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
reproducible from a committed benchmark — and, for anything closed on the
reference device, from a committed raw log too.

**Two kinds of benchmark live in `benchmark/`.** The comparisons against OpenCV
need OpenCV and are the ones the denominator rule above is about. The experiment
benchmarks that settle an E-question (`alignment_`, `wordwidth_`, `window_`)
compare binCV against binCV, need no OpenCV, and **build in the core-only
configuration** — which is what `scripts/run_on_pi.sh` configures by default, so
reproducing X-9, X-10 or X-11 on the reference device takes no extra flags.

---

## Code Tour

### Core
- [bincv-cpp/include/bincv-cpp/core/types.hpp](bincv-cpp/include/bincv-cpp/core/types.hpp) — `Size`, `Rect`, morphology and border enums, type aliases
- [bincv-cpp/include/bincv-cpp/core/storage.hpp](bincv-cpp/include/bincv-cpp/core/storage.hpp) — owning or caller-provided backing memory
- [bincv-cpp/include/bincv-cpp/core/view.hpp](bincv-cpp/include/bincv-cpp/core/view.hpp) — `BinMatView` / `BinMatConstView`, the kernel interface
- [bincv-cpp/include/bincv-cpp/core/error.hpp](bincv-cpp/include/bincv-cpp/core/error.hpp) — `BINCV_THROW` / `BINCV_ASSERT`, the error policy
- [bincv-cpp/include/bincv-cpp/binMat.hpp](bincv-cpp/include/bincv-cpp/binMat.hpp) — the 1-bit container, which is `QuantMat<1>`
- [bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp](bincv-cpp/include/bincv-cpp/impl/binMat_impl.hpp) — template implementation
- [bincv-cpp/include/bincv-cpp/quantMat.hpp](bincv-cpp/include/bincv-cpp/quantMat.hpp) — the N-bit container, and its signed / ternary reading

### Kernels
- [bincv-cpp/include/bincv-cpp/ops/logic.hpp](bincv-cpp/include/bincv-cpp/ops/logic.hpp) — `bitwiseAnd` / `Or` / `Xor` / `Not` (T2.2), over views and per `QuantMat` plane
- [bincv-cpp/include/bincv-cpp/ops/shift.hpp](bincv-cpp/include/bincv-cpp/ops/shift.hpp) — `shiftLeft` / `Right` / `Up` / `Down` and the 2-D `shift` (T2.3, T2.4), with OpenCV `BorderType` semantics
- [bincv-cpp/include/bincv-cpp/ops/reduce.hpp](bincv-cpp/include/bincv-cpp/ops/reduce.hpp) — `countNonZero`, `countAnd`, `countAndSplit`, `countCovariance`, `SlidingWindowCount` (T2.5, T2.6, T2.11). Bulk only, per [D-6](ARCHITECTURE.md#d-6-bulk-only-reductions): there is no per-word popcount in the public surface, and [D-13](ARCHITECTURE.md#d-13-a-reduction-counts-pixels-never-padding) says a reduction never counts a bit past `width`. Several of these compute what another one also computes, faster, by traversing less — the file's "which shape to reach for" section is the access-pattern argument, and it is measured ([X-11](EXPERIMENTS.md), [D-15](ARCHITECTURE.md#d-15-window-reductions-get-incremental-state-and-a-fused-covariance))
- [bincv-cpp/include/bincv-cpp/ops/bitslice.hpp](bincv-cpp/include/bincv-cpp/ops/bitslice.hpp) — `maj3`, `bitSlicedSum`, `thresholdGE` and the view-level `majority3` (T2.7). Small-count arithmetic over bit planes, **API tier 3**; its whole per-lane input space is enumerated by `tests/test_bitslice.cpp` rather than sampled
- [bincv-cpp/include/bincv-cpp/ops/denoise.hpp](bincv-cpp/include/bincv-cpp/ops/denoise.hpp) — `denoiseMedian3` (T3.1), the reference pipeline's three-pixel median. **API tier 3**: the neighbourhood is an asymmetric above/self/right L with a ZERO-FILL border, which is the reference's behaviour and not `cv::medianBlur`'s. One pass, no scratch buffer ([X-12](EXPERIMENTS.md))
- [bincv-cpp/include/bincv-cpp/ops/threshold.hpp](bincv-cpp/include/bincv-cpp/ops/threshold.hpp) — `threshold` from a `CV_8U` source (**API tier 1**, bit-exact against `cv::threshold` with `THRESH_BINARY`) and `binarize` from a `QuantMat<N>` (**API tier 3**, no OpenCV equivalent) (T3.2). Both compare **strictly greater than**, and the suite enumerates that boundary rather than sampling it
- [bincv-cpp/include/bincv-cpp/ops/pyramid.hpp](bincv-cpp/include/bincv-cpp/ops/pyramid.hpp) — **three entry points, and picking the wrong one is the easiest mistake in the library.** `pyrDown` is **exactly `cv::pyrDown`** — 5×5 `[1,4,6,4,1]` Gaussian, `BORDER_REFLECT_101`, **API tier 1 at `NIn == NOut == 8`** and proven bit-exact against OpenCV. `pyrDownBox` is the 2×2 box with `BORDER_REPLICATE` — **binCV's own operating point**, and what every performance number here is measured on. `pyrDownFiltered<F, …, Bo>` is the full space: five filters × three borders, dispatching to a per-filter specialisation where one exists. `Pyramid<W, N0, N1, …>::build<F, Bo>()` defaults to the OpenCV pair, so **a pipeline must ask for the box explicitly** (T3.4, [D-39](ARCHITECTURE.md), [X-48](EXPERIMENTS.md))
- [bincv-cpp/include/bincv-cpp/ops/derivative.hpp](bincv-cpp/include/bincv-cpp/ops/derivative.hpp) — the binarized `[-1, 0, 1]` spatial derivative into a `SignedQuantMat<N>` (T3.5). Sign-magnitude, not two's complement ([D-3](ARCHITECTURE.md)), which is what makes the LK covariance fall out as popcounts
- [bincv-cpp/include/bincv-cpp/ops/covariance.hpp](bincv-cpp/include/bincv-cpp/ops/covariance.hpp) — the LK gradient covariance over a window (T3.6, T3.10), N-bit
- [bincv-cpp/include/bincv-cpp/ops/corner.hpp](bincv-cpp/include/bincv-cpp/ops/corner.hpp) — `cornerMinEigenVal` and the `goodFeaturesToTrack` port (T3.7, T3.11), including the streaming three-row response ring that keeps detection off the heap
- [bincv-cpp/include/bincv-cpp/ops/opticalFlow.hpp](bincv-cpp/include/bincv-cpp/ops/opticalFlow.hpp) — `calcOpticalFlowPyrLK` over an `LKLevels` ladder (T3.8). **The hot kernel**: 68% of the frontend, and where D-30…D-33, X-35 and X-40's NEON work lives. Read [D-37](ARCHITECTURE.md) and [D-40](ARCHITECTURE.md) before optimising it — counting, addressing, cache and layout have each been priced and each declined or exhausted
- [bincv-cpp/include/bincv-cpp/ops/morphology.hpp](bincv-cpp/include/bincv-cpp/ops/morphology.hpp) — `erode` / `dilate` (T3.3), **API tier 1** against OpenCV for the rectangular structuring elements
- [bincv-cpp/include/bincv-cpp/ops/resample.hpp](bincv-cpp/include/bincv-cpp/ops/resample.hpp) — nearest-neighbour resize over packed bits
- [bincv-cpp/include/bincv-cpp/ops/blockMatch.hpp](bincv-cpp/include/bincv-cpp/ops/blockMatch.hpp) — Hamming block matching, E-6's route (a). Kept as the measured alternative to LK, not as the shipped tracker ([D-24](ARCHITECTURE.md))
- [bincv-cpp/include/bincv-cpp/impl/kernel_util.hpp](bincv-cpp/include/bincv-cpp/impl/kernel_util.hpp) — the row-tail mask, the stride check and the [D-11](ARCHITECTURE.md#d-11-kernels-alias-exactly-or-not-at-all) overlap predicates, shared by every kernel under `ops/`

### Support
- [bincv-cpp/include/bincv-cpp/util.hpp](bincv-cpp/include/bincv-cpp/util.hpp) — image I/O for tests (OpenCV-only)
- [bincv-cpp/tests/](bincv-cpp/tests/) — test suites
- [bincv-cpp/benchmark/](bincv-cpp/benchmark/) — OpenCV comparisons (need OpenCV)
  and the binCV-versus-binCV experiment benchmarks (core-only, no OpenCV)

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
