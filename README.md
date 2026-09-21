# binCV

**Image processing at the bit width the image actually has**, with OpenCV's API shape. A
binary image gets one bit per pixel, a ternary image two, a few-bit image three or four.

Binary images are everywhere in a vision pipeline — masks, thresholded edges. Mainstream
libraries store them one **byte** per pixel and do eight-bit arithmetic on them: 8× the
memory and 8× the work. On a small device the memory is the part that hurts.

binCV packs them and operates on whole machine words: an AND becomes an AND over words,
counting set pixels becomes a population count, dilating becomes a shift and an OR — plain
integer code that any CPU runs. **Few-bit** images work the same way, one plane per bit, so
arithmetic stays word-wide.

binCV provides alternatives to individual OpenCV calls. Keep the pipeline you have and call
binCV for the operations where a packed representation pays. It takes a single-channel,
integer-typed, strided pixel array; decoding, demosaicing and color conversion stay on your
side of that line. Header-only C++, zero dependencies.

## Performance

<!-- figure-check values="OpenCV, x86-64|binCV, x86-64|speedup, x86-64|OpenCV, aarch64|binCV, aarch64|speedup, aarch64" source="source" -->
| operation | OpenCV equivalent | OpenCV, x86-64 | binCV, x86-64 | speedup, x86-64 | OpenCV, aarch64 | binCV, aarch64 | speedup, aarch64 | source |
|---|---|---|---|---|---|---|---|---|
| `bitwiseAnd`, ns/pixel | `cv::bitwise_and` | 0.02734 | 0.00273 | 10.01× | 0.64783 | 0.02266 | 28.59× | [primitives.md](docs/reports/primitives.md) |
| optical flow, 140 points, ms/call | `cv::calcOpticalFlowPyrLK` | 3.871 | 0.543 | 7.13× | 23.476 | 2.843 | 8.26× | [features.md](docs/reports/features.md) |
| `pyrDown`, 1 bit in → 3 bits out, µs/call | `cv::pyrDown` on `CV_8U` | 48.3 | 31.0 | 1.56× | 521.4 | 93.8 | 5.56× | [primitives.md](docs/reports/primitives.md) |
| Hamming matching, kNN=2 over 1000×1000, ms | `cv::BFMatcher` | 9.184 | 1.947 | 4.72× | 38.269 | 19.391 | 1.97× | [features.md](docs/reports/features.md) |
| `countNonZero`, ns/pixel | `cv::countNonZero` | 0.01548 | 0.00956 | 1.62× | 0.17116 | 0.06366 | 2.69× | [primitives.md](docs/reports/primitives.md) |
| `goodFeaturesToTrack`, ns/pixel | `cv::goodFeaturesToTrack`, binarized | 13.67 | 12.06 | 1.13× | 74.99 | 43.49 | 1.72× | [features.md](docs/reports/features.md) |
| dense disparity, ms/frame | `cv::StereoBM` | ~14.7 | ~12.0 | ~1.2× | 79.8 | 60.4 | 1.32× | [stereo.md](docs/reports/stereo.md) |
| `erode`, 5×5 ellipse, ns/pixel | `cv::erode` | 0.22759 | 0.70415 | 0.32× | 1.81575 | 3.58631 | 0.51× | [primitives.md](docs/reports/primitives.md) |

x86-64 is a desktop Ryzen 5 5600X, aarch64 a Raspberry Pi 4 at a pinned clock. Both columns
are one thread — compare at equal thread counts, or a ratio means nothing.
[Where it does not pay](#where-it-does-not-pay) covers the rows where a packed representation
costs more than it saves.

Memory is the other half, and usually the half that decides whether something fits. Each row
is one call's peak working set, from buffer geometry, identical on both architectures:

<!-- figure-check values="OpenCV|binCV|× smaller" source="source" -->
| operation | OpenCV equivalent | OpenCV | binCV | × smaller | source |
|---|---|---|---|---|---|
| `bitwiseAnd` / `Or` / `Xor` / `Not`, bytes | `cv::bitwise_*` | 921,600 | 115,200 | 8.0× | [primitives.md](docs/reports/primitives.md) |
| FAST input plane, bytes | `cv::FAST` on `CV_8U` | 360,960 | 46,080 | 7.83× | [footprint.md](docs/reports/footprint.md) |
| `goodFeaturesToTrack`, bytes | `cv::goodFeaturesToTrack`, binarized | 9,014,976 | 1,580,064 | 5.71× | [footprint.md](docs/reports/footprint.md) |
| `morphologyEx(MORPH_OPEN)`, bytes | `cv::morphologyEx` | 614,400 | 115,200 | 5.33× | [footprint.md](docs/reports/footprint.md) |
| dense disparity, working set | `cv::StereoBM` | ≥ 722 KB, output alone | 32.4 KB scratch + 1 B/px out | ~22×, a lower bound | [stereo.md](docs/reports/stereo.md) |

There is a CUDA backend too, against `cv::cuda` on the same GPU:

<!-- figure-check values="cv::cuda, ms|binCV, ms|speedup|cv::cuda, KB|binCV, KB|× smaller" source="source" -->
| on an RTX 3070 Ti | cv::cuda equivalent | cv::cuda, ms | binCV, ms | speedup | cv::cuda, KB | binCV, KB | × smaller | source |
|---|---|---|---|---|---|---|---|---|
| dense disparity, binary entry, per frame | `cv::cuda::StereoBM(64, 9)` | 0.7152 | 0.0648 | 11.0× | 3,072.0 | 448.0 | 6.857× | [cuda.md](docs/reports/cuda.md) |
| descriptor matching, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 1.9491 | 0.2189 | 9.1× | 8,277.3 | 400.0 | 20.7× | [cuda.md](docs/reports/cuda.md) |

**[docs/reports/](docs/reports/README.md) has the whole set** — every operation on both
architectures and the GPU, wins and losses in the same tables, with the machines, the method
and the command that reproduces each row.

## Using it

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
`uint32_t`. The same code compiles at 8, 16, 32 or 64 bits per word. Kernels take their
scratch from the caller, so what a call will cost is known before it runs.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

OpenCV is optional (`-DBINCV_USE_OPENCV=OFF`). Adding `include/` to your include path works
too — but **link the `bincv_core` target if you use CMake**, because the ISA flags ride on
it. See [GETTING_STARTED.md](GETTING_STARTED.md).

## What is in it

Packing from 8- and 16-bit sources, logic and shifts, bulk and windowed reductions,
morphology, resampling, thresholding, bit-sliced arithmetic; pyramids, derivatives, corner
detection, FAST, BRIEF, Hamming matching and pyramidal Lucas–Kanade; sparse and dense stereo;
RANSAC over caller-owned scratch. `cv::Mat` in and out when OpenCV is present, raw buffers
and PNM otherwise. Every entry point states an **API tier** — bit-exact with the OpenCV
function it names, the same role with different numerics, or no OpenCV equivalent.

Inventory: [docs/API.md](docs/API.md). What each area covers and what the tiers mean:
[docs/README.md](docs/README.md#what-is-in-it).

## Platforms

Header-only C++ with zero dependencies: desktop, mobile, embedded Linux and bare-metal
microcontrollers, plus `backends/cuda/` for NVIDIA GPUs.
**[docs/README.md](docs/README.md#where-bincv-runs) says which targets have been built, which
have been measured, and what each one gets.**

## Where it does not pay

**A non-separable structuring element.** A 5×5 ellipse costs one shifted-OR per set element,
and binCV runs it at 0.32× of `cv::erode` on x86-64. The fused kernel shipped at that price
because it holds 76,800 bytes against `cv::erode`'s 614,400 — when speed and footprint
conflict here, footprint wins. A 3×3 element is 1.04× on x86-64 and 1.00× on the Pi, and
holds the same 76,800 bytes. ([primitives.md](docs/reports/primitives.md))

**Wide inputs into a bit-sliced filter.** binCV's filters carry one plane per bit, so an
accumulator has to hold the weighted sum and the work grows with input depth. At one bit
there is nothing to accumulate; at eight, a byte kernel's vector unit wins outright —
`pyrDown` fed eight bits runs at 0.02× on x86-64 and 0.07× on aarch64. That is the library
outside its premise: it ships as 1 bit in, 3 bits out, and the crossover sits at a different
depth on each machine. ([limits.md](docs/reports/limits.md))

**On the GPU, one path is faster and bigger.** The census dense entry beats
`cv::cuda::StereoBM` at 1.47× but holds 4,512.0 KB against its 3,072.0, because the census
transform expands eight bits per pixel into a 32-bit descriptor word. It is the one row in
that report where binCV is larger. ([cuda.md](docs/reports/cuda.md))

## Status

**Pre-release. Names and signatures will move.**

## License

TBD.
