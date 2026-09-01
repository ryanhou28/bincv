# binCV

**Image processing for low-bit-width images — binary, ternary, few-bit — at their true
bit width.** One bit per pixel, not eight, with OpenCV's API shape.

```cpp
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/ops/morphology.hpp"
#include "bincv-cpp/ops/reduce.hpp"

// Binary images, one bit per pixel. A 640x480 mask is 38 KB, not 307 KB.
bincv::BinMat<uint32_t> mask(640, 480), roi(640, 480);
bincv::BinMat<uint32_t> cleaned(640, 480), scratch(640, 480);

// Remove speckle, then keep only what falls inside a region of interest.
bincv::morphologyEx(mask.constView(), cleaned.view(), bincv::MORPH_OPEN,
                    bincv::StructuringElement{}, scratch.view());
bincv::bitwiseAnd(cleaned.constView(), roi.constView(), cleaned.view());

const size_t pixels = bincv::countNonZero(cleaned.constView());
```

Each of those operations touches 32 pixels per instruction, because 32 pixels fit in a
`uint32_t`. The same code compiles at 8, 16, 32 or 64 bits per word.

## Why

Binary images are everywhere in vision — masks, thresholded edges, morphology,
occupancy grids, structured-light patterns — and every mainstream library stores them one
**byte** per pixel. Eight bits to hold a value that is 0 or 1, and then eight-bit
arithmetic to combine them.

That is 8× the memory and 8× the work, and on a small device the memory is the part that
hurts: a buffer either fits or it does not.

binCV stores one bit per pixel and operates on whole machine words. An AND over two
images becomes an AND over their words. Counting set pixels becomes a population count.
Dilating becomes a shift and an OR. The pixel loop mostly disappears, and what is left is
ordinary integer code that needs no special hardware.

It also handles **few-bit** images the same way — 2, 3, 4 bits per pixel — by storing
each bit as its own plane, so arithmetic stays word-wide instead of degrading into
per-pixel work.

## Performance

A keypoint-tracking frontend — median, edge threshold, pyramid, derivatives, corner
detection and Lucas–Kanade — against OpenCV performing the same operations on the same
content. Both sides start from the same grayscale frame and each builds its own binary
one, so this is end to end.

| one thread each side | binCV | OpenCV | |
|---|---|---|---|
| peak working set | **436 704 B** | 2 719 832 B | **6.23× smaller** |
| Cortex-A72 | **4.88 ms/frame** | 29.63 | **6.07× faster** |
| x86-64 | **1.07 ms/frame** | 4.46 | **4.16× faster** |

Tracking agrees with OpenCV within 1 px on **100%** of flow vectors, median difference
0.032 px.

**Both sides get the same thread count**, which is the only comparison that isolates the
implementation from the parallelism. binCV threads through a caller-installed backend and
is serial by default; at four threads each on x86 the ratio narrows to roughly 3×,
because OpenCV has more byte-work to spread across cores.

## What is in it

- **Getting to bits** — packing from 8- and 16-bit sources, a wide median, and a
  gradient-threshold edge filter that writes bit-planes directly.
- **Primitives** — logic, shifts, bulk and windowed reductions, morphology, resampling,
  bit-sliced arithmetic, thresholding.
- **Features and tracking** — pyramid, derivatives, gradient covariance, corner response,
  `goodFeaturesToTrack`, pyramidal Lucas–Kanade, FAST, BRIEF descriptors and Hamming
  matching, sub-pixel refinement.
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

OpenCV is optional: `-DBINCV_USE_OPENCV=OFF` builds the library and its tests without it.
binCV is header-only, so you can also just add `bincv-cpp/include` to your include path —
but **link the `bincv_core` target if you use CMake**, because the ISA flags ride on it.

See [GETTING_STARTED.md](GETTING_STARTED.md).

## Status

**Pre-release, and the API is not stable.** Expect names and signatures to move.

## Licence

TBD.
