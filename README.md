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

Masks, thresholded edges, morphology, occupancy grids and structured-light patterns are
binary, and every mainstream library stores them one **byte** per pixel — eight bits to hold
a 0 or a 1, then eight-bit arithmetic to combine them. That is 8× the memory and 8× the
work, and on a small device the memory is the part that hurts.

binCV stores one bit per pixel and operates on whole machine words: an AND becomes an AND
over words, counting set pixels becomes a population count, dilating becomes a shift and an
OR — ordinary integer code needing no special hardware. **Few-bit** images (2, 3, 4 bits)
work the same way, one plane per bit, so arithmetic stays word-wide.

## Performance

binCV replaces individual OpenCV calls, so these are single operations against the single
call each replaces. No one number is the answer, and binCV loses some of these. Each row
names its own unit, and on all of them the smaller number is the faster side:

<!-- figure-check values="OpenCV, x86-64|binCV, x86-64|x86-64, OpenCV ÷ binCV (>1× = binCV faster)|OpenCV, aarch64|binCV, aarch64|aarch64, OpenCV ÷ binCV (>1× = binCV faster)" source="source" -->
| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64, OpenCV ÷ binCV (>1× = binCV faster) | OpenCV, aarch64 | binCV, aarch64 | aarch64, OpenCV ÷ binCV (>1× = binCV faster) | source |
|---|---|---|---|---|---|---|---|---|
| `bitwiseAnd`, ns/pixel | `cv::bitwise_and` | 0.02734 | 0.00273 | 10.01× | 0.64783 | 0.02266 | 28.59× | [primitives.md](docs/reports/primitives.md) |
| optical flow, 140 points, ms/call | `cv::calcOpticalFlowPyrLK` | 3.871 | 0.543 | 7.13× | 23.476 | 2.843 | 8.26× | [features.md](docs/reports/features.md) |
| Hamming matching, kNN=2, ms | `cv::BFMatcher` | 9.184 | 1.947 | 4.72× | 38.269 | 19.391 | 1.97× | [features.md](docs/reports/features.md) |
| dense disparity, ms/frame | `cv::StereoBM` | ~14.7 | ~12.0 | ~1.2× | 79.8 | 60.4 | 1.32× | [stereo.md](docs/reports/stereo.md) |
| `goodFeaturesToTrack`, ns/pixel | `cv::goodFeaturesToTrack`, binarized | 13.63–14.24 | 14.46–15.01 | 0.92× | 75.02–75.82 | 51.25–51.31 | 1.45× | [features.md](docs/reports/features.md) |
| FAST, wide image, ms/call | `cv::FAST` | 0.363 | 0.344 | 1.05× | 2.906 | 3.024 | 0.96× | [features.md](docs/reports/features.md) |
| `erode` 5×5 ellipse, ns/pixel | `cv::erode` | 0.22759 | 0.70415 | 0.32× | 1.81575 | 3.58631 | 0.51× | [primitives.md](docs/reports/primitives.md) |
| `pyrDown`, 8 bits in, µs/call | `cv::pyrDown` on `CV_8U` | 48.3 | 2034.4 | 0.02× | 521.4 | 7358.6 | 0.07× | [limits.md](docs/reports/limits.md) |

The bottom two rows are where the idea stops paying, and both are structural: a
non-separable element costs one shifted-OR per set element, and at eight bits per pixel both
sides store a byte. [limits.md](docs/reports/limits.md) is the page about that.

Memory is the other half, and usually the half that decides whether something fits. Peak
working set of one call, from buffer geometry, identical on both architectures:

<!-- figure-check values="OpenCV|binCV|OpenCV ÷ binCV (>1× = binCV smaller)" source="source" -->
| operation | measured against | OpenCV | binCV | OpenCV ÷ binCV (>1× = binCV smaller) | source |
|---|---|---|---|---|---|
| denoise, 3-pixel median, bytes | composed `cv::min` / `cv::max` | 2,150,400 | 76,800 | 28.0× | [footprint.md](docs/reports/footprint.md) |
| FAST input plane, bytes | `cv::FAST` on `CV_8U` | 360,960 | 46,080 | 7.83× | [footprint.md](docs/reports/footprint.md) |
| `goodFeaturesToTrack`, bytes | `cv::goodFeaturesToTrack`, binarized | 9,014,976 | 1,580,064 | 5.71× | [footprint.md](docs/reports/footprint.md) |
| dense disparity, working set | `cv::StereoBM` | ≥ 722 KB, output alone | 32.4 KB scratch, 1 B/px out | ~22× | [stereo.md](docs/reports/stereo.md) |

There is a CUDA backend too, measured against `cv::cuda` on the same GPU:

<!-- figure-check values="cv::cuda, ms|binCV, ms|speed, cv::cuda ÷ binCV (>1× = binCV faster)|cv::cuda, KB|binCV, KB|memory, cv::cuda ÷ binCV (>1× = binCV smaller)" source="source" -->
| on an RTX 3070 Ti | measured against | cv::cuda, ms | binCV, ms | speed, cv::cuda ÷ binCV (>1× = binCV faster) | cv::cuda, KB | binCV, KB | memory, cv::cuda ÷ binCV (>1× = binCV smaller) | source |
|---|---|---|---|---|---|---|---|---|
| dense disparity, binary entry, per frame | `cv::cuda::StereoBM(64, 9)` | 0.7152 | 0.0648 | 11.0× | 3072.0 | 448.0 | 6.857× | [cuda.md](docs/reports/cuda.md) |
| BRIEF descriptors, per call at N=1000 | `cv::cuda::ORB::computeAsync` | 0.1070 | 0.0107 | 9.3× | 2048.0 | 48.0 | 42.67× | [cuda.md](docs/reports/cuda.md) |

x86-64 is a Ryzen 5 5600X desktop and aarch64 a Raspberry Pi 4 at a pinned clock — different
measurements against different OpenCV builds, never averaged.
**[docs/reports/](docs/reports/README.md) has the whole set**, every operation on both
architectures and the GPU, wins and losses in the same tables.

Each of these is one build of binCV against one build of OpenCV on one machine, so reproduce
it — [build](#building), then `./build/benchmark/logic_benchmark`. **Compare at equal thread
counts** — binCV is serial unless you install a threading backend, and comparing one thread
against many measures the parallelism rather than the implementation.

## What is in it

Packing from 8- and 16-bit sources, logic and shifts, bulk and windowed reductions,
morphology, resampling, thresholding, bit-sliced arithmetic; pyramids, derivatives, corner
detection, FAST, BRIEF, Hamming matching and pyramidal Lucas–Kanade; sparse and dense stereo;
RANSAC over caller-owned scratch. `cv::Mat` in and out when OpenCV is present, raw buffers
and PNM when it is not. Every entry point states an **API tier** — bit-exact with the OpenCV
function it names, the same role with different numerics, or no OpenCV equivalent.

Inventory: [docs/API.md](docs/API.md). What each area covers and what the tiers mean:
[docs/README.md](docs/README.md#what-is-in-it).

## Platforms

Header-only C++ with zero dependencies: desktop, mobile, embedded Linux and bare-metal
microcontrollers, plus `backends/cuda/` for NVIDIA GPUs.
**[docs/README.md](docs/README.md#where-bincv-runs) says which targets have been built, which
have been measured, and what each one gets.** Log `bincv::simdStatusString()` at start-up; it
names every vector path and says whether it is active.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

OpenCV is optional (`-DBINCV_USE_OPENCV=OFF`). Adding `include/` to your include path works
too — but **link the `bincv_core` target if you use CMake**, because the ISA flags ride on
it. See [GETTING_STARTED.md](GETTING_STARTED.md).

## Status

**Pre-release, and the API is not stable.** Expect names and signatures to move.

## License

TBD.
