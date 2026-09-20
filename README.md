# binCV

**Image processing for low-bit-width images — binary, ternary, few-bit — at their true
bit width.** One bit per pixel, not eight, with OpenCV's API shape.

```cpp
#include "bincv/ops/logic.hpp"
#include "bincv/ops/morphology.hpp"
#include "bincv/ops/reduce.hpp"

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
and the machine, so no single number is the answer. A few measured ones, both sides of
each comparison, with the machine and the report they come from. Each row names its own
unit — milliseconds, microseconds, bytes — and on every one of them the smaller number is
the better side, so a row where binCV's cell is the larger one is a row binCV lost:

<!-- figure-check values="OpenCV|binCV" source="source" -->
| what | measured against | machine | OpenCV | binCV | source |
|---|---|---|---|---|---|
| whole tracking frontend, ms/frame | the OpenCV frontend | aarch64 | 23.249–23.451 | 4.906–4.949 | [frontend.md](docs/reports/frontend.md) |
| whole tracking frontend, peak, bytes | the OpenCV frontend | x86-64 and aarch64 | 2,719,832 | 436,704 | [footprint.md](docs/reports/footprint.md) |
| optical flow, 140 points, ms/call | `cv::calcOpticalFlowPyrLK` | aarch64 | 23.476 | 2.843 | [features.md](docs/reports/features.md) |
| dense disparity, ms/frame | `cv::StereoBM` | aarch64 | 79.8 | 60.4 | [stereo.md](docs/reports/stereo.md) |
| dense disparity, working set | `cv::StereoBM` | x86-64 and aarch64 | ≥ 722 KB, output alone | 32.4 KB scratch, 1 B/px out | [stereo.md](docs/reports/stereo.md) |
| dense disparity, ms/frame | `cv::cuda::StereoBM(64, 9)` | RTX 3070 Ti | 0.7152 | 0.0648 | [cuda.md](docs/reports/cuda.md) |
| FAST, wide image, ms/call | `cv::FAST` | aarch64 | 2.906 | 3.024 | [features.md](docs/reports/features.md) |
| `pyrDown`, 8 bits in, µs/call | `cv::pyrDown` on `CV_8U` | x86-64 | 48.3 | 2034.4 | [limits.md](docs/reports/limits.md) |

aarch64 there is a Raspberry Pi 4 at a pinned clock, x86-64 a Ryzen 5 5600X desktop, and
the GPU row an RTX 3070 Ti against OpenCV's own CUDA module. **The last two rows are
losses**, and they are in the table for the same reason as the rest: binCV ties `cv::FAST`
on a byte image rather than beating it, and at eight bits per pixel there is nothing left
for bit-slicing to skip. [docs/reports/](docs/reports/README.md) has the whole set — every
operation measured, on both architectures and the GPU, wins and losses in the same tables,
with what each one was measured against and how.

Those are one build of binCV against one build of OpenCV on one machine each. Your
operation, image size, word type, compiler and machine will move them, so the benchmarks
are in the repository and report on your hardware:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
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
  `goodFeaturesToTrack`, pyramidal Lucas–Kanade, FAST (wide and bit-plane), sub-pixel
  refinement.
- **Descriptors and stereo** — intensity-centroid orientation, BRIEF and steered
  (rotation-compensated) BRIEF with caller-supplied patterns (cv::ORB's own table
  included, byte-exact), Hamming matching plain and prior-gated, sparse rectified
  stereo matching with sub-pixel disparity, and dense disparity via the census
  transform — streamed, so the cost volume never exists. The SLAM half of the
  feature path; `examples/slam_frontend.cpp` runs it end to end.
- **Geometry** — RANSAC over caller-owned scratch: 2D affine, and the five-point
  essential matrix.
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
| **Cortex-M** — microcontrollers | built and run on an STM32H753ZI (Cortex-M7): correct, and a 752×480 frame is 46 KB. Scalar only — no NEON, no popcount instruction. Only the reductions are timed so far |
| **32-bit ARM Cortex-A** | supported target; not yet built or measured |
| **RISC-V** | supported target; not yet built or measured |
| **CUDA** — NVIDIA GPUs | a separate backend in [backends/cuda/](backends/cuda/), sharing the format and forking the kernels. Device-typed, never a drop-in dispatch target |

Log `bincv::simdStatusString()` once at start-up — it names every vector path and says
whether it is active.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

OpenCV is optional: `-DBINCV_USE_OPENCV=OFF` builds the library and its tests without it.
binCV is header-only, so you can also just add the `include/` directory to your include path —
but **link the `bincv_core` target if you use CMake**, because the ISA flags ride on it.

See [GETTING_STARTED.md](GETTING_STARTED.md).

## Status

**Pre-release, and the API is not stable.** Expect names and signatures to move.

## License

TBD.
