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

The point of storing a bit per pixel is that an operation over a row becomes a handful of
word operations, and a buffer that used to be a megabyte becomes an eighth of one. Both
show up in practice, and the memory one is usually the one that decides whether something
fits on a small device.

How much you gain depends on the operation, the image size, the word type, the compiler
and the machine — so rather than quote a number here, the benchmarks are in the
repository and report on your hardware:

```bash
cmake -S bincv-cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/benchmark/logic_benchmark          # a primitive, against OpenCV
./build/benchmark/frontend_sequence <dir>  # a whole tracking frontend, against OpenCV
```

Each one reports binCV and OpenCV side by side on the same content, with peak memory
alongside the timings, and prints which vector paths were active. **Compare at equal
thread counts** — binCV is serial unless you install a threading backend, and comparing
one thread against many measures the parallelism rather than the implementation.

## What is in it

- **Getting to bits** — packing from 8- and 16-bit sources, a wide median, and a
  gradient-threshold edge filter that writes bit-planes directly.
- **Primitives** — logic, shifts, bulk and windowed reductions, morphology, resampling,
  bit-sliced arithmetic, thresholding.
- **Features and tracking** — pyramid, derivatives, gradient covariance, corner response,
  `goodFeaturesToTrack`, pyramidal Lucas–Kanade, FAST, BRIEF descriptors and Hamming
  matching, sub-pixel refinement.
- **Interop** — `cv::Mat` in and out when OpenCV is present; raw buffers and PNM (`P4`,
  `P5`) when it is not. binCV links no codec on any target: a camera's Y plane, a V4L2
  buffer and a sensor's DMA rows are already the input contract, so decoding sits on no
  path binCV is on.

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

## License

TBD.
