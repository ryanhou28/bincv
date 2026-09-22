# Getting started

binCV is header-only and needs a C++17 compiler. **The library itself never uses
OpenCV** — every `#include <opencv2/...>` in `include/` sits behind
`BINCV_WITH_OPENCV`. OpenCV buys you `cv::Mat` interop, the tests that check Tier 1
operations against it, and the benchmarks that compare against it.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the tests:

```bash
cd build && ctest --output-on-failure
```

Building without it — the core-only configuration an embedded target uses:

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release -DBINCV_USE_OPENCV=OFF
```

### Options

| Option | Default | Effect |
|---|---|---|
| `BINCV_USE_OPENCV` | ON | Interop, the interop tests, and the OpenCV-comparison benchmarks |
| `BINCV_BUILD_TESTS` | ON | Build the suites and register them with ctest |
| `BINCV_BUILD_BENCHMARKS` | ON | Benchmarks. The binCV-versus-binCV ones build without OpenCV; the comparisons are skipped without it |
| `BINCV_BUILD_EMBEDDED` | OFF | The bare-metal targets under `targets/`. Needs an arm-none-eabi cross build — see `cmake/toolchain-cortex-m7.cmake` |
| `BINCV_X86_POPCNT` | ON | Build with `-mpopcnt`. Off is supported and slower; see the comment in `CMakeLists.txt` |

## Benchmark

```bash
./build/benchmark/fill_benchmark --width 640 --height 480 \
    --iterations 100 --dtype binary --sparsity 0.5
```

Or the full sweep, `./scripts/run_all_benchmarks.sh`.

**Always benchmark a Release build**, and read the rules below on the comparison
denominator before quoting a ratio.

## Use it in your project

With CMake — **do this rather than adding the include path by hand:**

```cmake
add_subdirectory(path/to/bincv)
target_link_libraries(your_target PRIVATE bincv_core)
```

The ISA flags that select the fast paths ride on the `bincv_core` target. Adding
the `include/` directory to your include path alone gives you a correct library that is
several times slower, with no warning — the vector kernels produce identical results, so
nothing looks wrong.

**Log this once at start-up and you will never wonder:**

```cpp
#include "bincv/core/simd.hpp"
std::printf("%s\n", bincv::simdStatusString());
// binCV SIMD: NEON=yes AVX2=n/a popcount=hardware  (fast paths active)
```

## First program

Threshold a grayscale frame straight into bit-planes — no OpenCV, no 8-bit intermediate:

```cpp
#include <cstdio>
#include <vector>
#include "bincv/ops/edge.hpp"
#include "bincv/ops/reduce.hpp"

int main() {
    const size_t w = 640, h = 480;
    std::vector<uint8_t> gray(w * h);            // your camera's bytes

    bincv::BinMat<uint32_t> edges(w, h);         // 38 400 B, not 307 200
    bincv::edgeThreshold(gray.data(), w, h, /*stride=*/w, edges.view(), uint8_t{17});

    std::printf("%zu edge pixels\n", bincv::countNonZero(edges.constView()));
}
```

`BinMat<uint32_t>` is a binary image at one bit per pixel. `edges.view()` hands a kernel
a non-owning `{pointer, width, height, stride}` — kernels take views, never containers.

## Choosing a word type

`uint32_t` unless you have measured a reason otherwise. It is the default because it
balances work per operation against the memory a wider stride wastes on small images.

If you already hold 64-bit words, keep them and narrow at the call:

```cpp
bincv::edgeThreshold(src, w, h, stride, bincv::narrowPlaneMutable(dst64.view()), t);
```

That is a reinterpretation, not a copy — a 64-bit bit-plane already is a 32-bit one with
twice the stride — and it runs at native 32-bit speed.

## A feature tracking pipeline

`examples/vio_frontend.cpp` is a complete feature tracking pipeline: sensor
stage, pyramid, derivatives, corner detection, Lucas–Kanade, and re-detection when tracks
run out. (A *VIO frontend* is what visual-inertial odometry calls exactly that stack — the
image-processing half that feeds the optimizer.) It is the best starting point for anything
larger than one operation.

```bash
./build/examples/vio_frontend <directory-of-png-frames>   # OpenCV builds
./build/examples/vio_frontend frames.bsq                  # any build, core-only included
```

With OpenCV present it reads PNG through `cv::imread` and runs the sensor stage in
OpenCV's vocabulary — deliberately, to show the caller's half of the input boundary. A
core-only build reads a **sequence blob** instead (`scripts/make_sequence_blob.py` turns
a frame directory into one on any host) and runs the sensor stage in binCV's own
spelling (`medianWide` + `edgeThreshold`), so the embedded configuration runs real
dataset frames end to end with no OpenCV anywhere. binCV itself links no codec on any
target — see [ARCHITECTURE.md](docs/ARCHITECTURE.md) — and the library's own file I/O
is PNM (`readPbm`/`writePbm`, `readPgm`/`writePgm`) plus the blob reader
(`io/sequence.hpp`), which need nothing.

`examples/slam_frontend.cpp` is the descriptor-association counterpart — the
SLAM frontend loop: FAST per pyramid level, intensity-centroid orientation, steered
BRIEF, Hamming matching against the previous frame, and the five-point essential
matrix under RANSAC. It prints a per-stage, per-level profile and its headline is
the RANSAC inlier rate.

```bash
./build/examples/slam_frontend <directory-of-png-frames>
```


## Embedded targets

Two things to set before you build for a small part:

```cpp
// The tracker stages windows on the stack. Declare what you have and the build
// fails if it would not fit, instead of overflowing silently at run time.
#define BINCV_STAGING_BUDGET_BYTES 8192
```

`bincv::stagingStackBytes<N, WordType>()` gives the exact figure for a configuration.

binCV never allocates inside a kernel and never throws; scratch buffers, where an
operation needs one, are parameters you provide.

## Conventions

Function names, argument order and semantics follow OpenCV where an equivalent exists.
Every public entry point declares an **API tier** in its docstring:

- **Tier 1** — bit-exact with the OpenCV function it names.
- **Tier 2** — same role and call shape, different numerics; the docstring says how.
- **Tier 3** — no OpenCV equivalent, and deliberately not an OpenCV name.

Destinations are out-parameters, as in OpenCV.

## Where to look next

| | |
|---|---|
| [docs/API.md](docs/API.md) | every public entry point, its brief and its tier |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the library is put together, and why |
| the headers | densely commented; the reasoning for a kernel is next to it |
