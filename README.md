# binCV

**Image processing for low-bit-width frames — binary, ternary, few-bit — at their true
bit width.** One bit per pixel, not eight, with OpenCV's API shape.

```cpp
#include "bincv-cpp/ops/edge.hpp"

// A camera hands you bytes. No OpenCV needed, and no 8-bit intermediate.
bincv::BinMat<uint32_t> edges(width, height);
bincv::edgeThreshold(pixels, width, height, stride, edges.view(), uint8_t{17});
```

## Why

A binarised 752×480 frame is 45 kB at one bit per pixel and 361 kB at one byte. Storing
it as `CV_8U` — which is what you do today, because that is what the libraries take —
spends 8× the memory to hold values that are always 0 or 1, and then does 8-bit
arithmetic on them.

binCV stores the bits and operates on them 32 or 64 pixels at a time, in ordinary
integer registers. It is header-only and needs nothing but a C++17 compiler.

## Where it stands

A keypoint-tracking frontend — median, edge threshold, pyramid, derivatives, corner
detection and Lucas–Kanade — against OpenCV performing the same operations on the same
content. Both sides start from the same grayscale frame and each builds its own binary
one, so this is end to end.

| **one thread each side** | binCV | OpenCV | |
|---|---|---|---|
| **peak working set** | **436 704 B** | 2 719 832 B | **6.23× smaller** |
| Cortex-A72 | **4.88 ms/frame** | 29.63 | **6.07× faster** |
| x86-64 | **1.07 ms/frame** | 4.46 | **4.16× faster** |

Tracking agrees with OpenCV within 1 px on **100%** of flow vectors, median difference
0.032 px.

**Both sides get the same thread count**, which is the only comparison that isolates the
implementation from the parallelism. binCV threads through a caller-installed backend
and is serial by default; at four threads each on x86 the ratio narrows to roughly 3×,
because OpenCV has more byte-work to spread across cores.

## What is in it

- **Getting to bits** — packing from 8- and 16-bit sources, a wide median, and a
  gradient-threshold edge filter that writes bit-planes directly.
- **Primitives** — logic, shifts, bulk and windowed reductions, morphology, resampling,
  bit-sliced arithmetic, thresholding.
- **Features and tracking** — pyramid, derivatives, gradient covariance, corner
  response, `goodFeaturesToTrack`, pyramidal Lucas–Kanade, FAST, BRIEF descriptors and
  Hamming matching, sub-pixel refinement.
- **Interop** — `cv::Mat` in and out when OpenCV is present; PGM and raw buffers when it
  is not.

Every public entry point states an **API tier**: bit-exact with the OpenCV function it
names, the same role with different numerics, or no OpenCV equivalent. See
[docs/API.md](docs/API.md).

## Platforms

| | |
|---|---|
| **x86-64** — desktop | measured. `POPCNT` required; AVX2 selected at run time |
| **aarch64** — mobile and embedded Cortex-A | measured. NEON |
| **32-bit ARM, Cortex-M** | supported target; not yet built or measured |
| **RISC-V** | supported target; not yet built or measured |

Log `bincv::simdStatusString()` once at start-up — it names every vector path and says
whether it is active.

## Building

```bash
cmake -S bincv-cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

OpenCV is optional: `-DBINCV_USE_OPENCV=OFF` builds the library and its tests without
it. binCV is header-only, so you can also just add `bincv-cpp/include` to your include
path — but **link the `bincv_core` target if you use CMake**, because the ISA flags
ride on it.

See [GETTING_STARTED.md](GETTING_STARTED.md).

## Scope

binCV is the **image processing**. Geometry and estimation — RANSAC, PnP, IMU fusion,
bundle adjustment — belong to the application above it.

The input boundary is a rule rather than a list: **binCV accepts a single-channel,
integer-typed, strided pixel array and turns it into an N-bit matrix.** Getting to that
array — decoding, demosaicing, colour conversion — is the caller's.

## Status

**Pre-release, and the API is not stable.** Expect names and signatures to move.

## Licence

Not yet chosen. **Until a licence file exists this code is "all rights reserved" by
default and cannot be used or redistributed.**
