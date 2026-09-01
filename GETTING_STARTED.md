# Getting started

binCV is header-only and needs a C++17 compiler. OpenCV is optional.

## Build

```bash
cmake -S bincv-cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the tests:

```bash
cd build && ctest --output-on-failure
```

Without OpenCV:

```bash
cmake -S bincv-cpp -B build -DCMAKE_BUILD_TYPE=Release -DBINCV_USE_OPENCV=OFF
```

## Use it in your project

With CMake — **do this rather than adding the include path by hand:**

```cmake
add_subdirectory(path/to/bincv-cpp)
target_link_libraries(your_target PRIVATE bincv_core)
```

The ISA flags that select the fast paths ride on the `bincv_core` target. Adding
`bincv-cpp/include` to your include path alone gives you a correct library that is
several times slower, with no warning — the vector kernels produce identical results, so
nothing looks wrong.

**Log this once at start-up and you will never wonder:**

```cpp
#include "bincv-cpp/core/simd.hpp"
std::printf("%s\n", bincv::simdStatusString());
// binCV SIMD: NEON=yes AVX2=n/a popcount=hardware  (fast paths active)
```

## First program

Threshold a grayscale frame straight into bit-planes — no OpenCV, no 8-bit intermediate:

```cpp
#include <cstdio>
#include <vector>
#include "bincv-cpp/ops/edge.hpp"

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

## A tracking frontend

`bincv-cpp/examples/vio_frontend.cpp` is a complete keypoint-tracking frontend: sensor
stage, pyramid, derivatives, corner detection, Lucas–Kanade, and re-detection when tracks
run out. It is the best starting point for anything larger than one operation.

```bash
./build/examples/vio_frontend <directory-of-pgm-or-png-frames>
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
