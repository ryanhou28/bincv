# binCV

**binCV provides computer vision operations for binary, ternary and other low-bit-width
images at their true native bit widths.** A binary image gets one bit per pixel, a ternary
image two, a few-bit image three or four. It is a header-only C++ library with zero
dependencies.

Binary images are everywhere in a vision pipeline — masks, thresholded edges. Conventional
libraries store them one **byte** per pixel and do eight-bit arithmetic on them: 8× the
memory and 8× the work.

binCV packs them and operates on whole machine words: an AND becomes an AND over words,
counting set pixels becomes a population count, dilating becomes a shift and an OR — plain
integer code that any CPU runs. **Few-bit** images work the same way, one plane per bit, so
arithmetic stays word-wide. [Performance and memory](#performance-and-memory) has what that
buys, measured on both counts.

For the operations OpenCV also has, binCV provides the same functionality in the same API
shape — keep the pipeline you have and call binCV where a packed representation pays. Some
operations match less closely, and some have no OpenCV counterpart at all; every entry point
states which ([docs/API.md](docs/API.md)). binCV takes a single-channel, integer-typed,
strided pixel array; decoding, demosaicing and color conversion stay on your side of that
line.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

OpenCV is optional (`-DBINCV_USE_OPENCV=OFF`). Adding `include/` to your include path works
too — but **link the `bincv_core` target if you use CMake**, because the ISA flags ride on
it. See [GETTING_STARTED.md](GETTING_STARTED.md).

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

## Platforms

Zero dependencies and header-only C++17: desktop, mobile, embedded Linux and bare-metal
microcontrollers, plus `backends/cuda/` for NVIDIA GPUs.
**[docs/README.md](docs/README.md#where-bincv-runs) says which targets have been built, which
have been measured, and what each one gets.**

## What is in it

Packing from 8- and 16-bit sources, logic and shifts, bulk and windowed reductions,
morphology, resampling, thresholding, bit-sliced arithmetic; pyramids, derivatives, corner
detection, FAST, BRIEF, Hamming matching and pyramidal Lucas–Kanade; sparse and dense stereo;
RANSAC over caller-owned scratch. `cv::Mat` in and out when OpenCV is present, raw buffers
and PNM otherwise. Every entry point states an **API tier** — bit-exact with the OpenCV
function it names, the same role with different numerics, or no OpenCV equivalent.

Inventory: [docs/API.md](docs/API.md). What each area covers and what the tiers mean:
[docs/README.md](docs/README.md#what-is-in-it).

## Performance and memory

<!-- figure-check values="OpenCV, x86-64|binCV, x86-64|ratio, x86-64|OpenCV, aarch64|binCV, aarch64|ratio, aarch64" source="source" -->
| operation | OpenCV equivalent | OpenCV, x86-64 | binCV, x86-64 | ratio, x86-64 | OpenCV, aarch64 | binCV, aarch64 | ratio, aarch64 | source |
|---|---|---|---|---|---|---|---|---|
| `bitwiseAnd`, ns/pixel | `cv::bitwise_and` | 0.02823 | 0.002810 | 9.97× [9.82, 10.3] | 0.6266 | 0.02369 | 26.7× [26.1, 27.4] | [primitives.md](docs/reports/primitives.md) |
| optical flow, 140 points, ms/call | `cv::calcOpticalFlowPyrLK` | 3.978 | 0.5585 | 7.19× [6.89, 7.40] | 23.47 | 2.837 | 8.27× [8.20, 8.36] | [features.md](docs/reports/features.md) |
| `pyrDown`, 1 bit in → 3 bits out, µs/call | `cv::pyrDown` on `CV_8U` | 47.70 | 30.70 | 1.56× [1.54, 1.60] | 516.5 | 93.8 | 5.51× [5.48, 5.55] | [primitives.md](docs/reports/primitives.md) |
| Hamming matching, kNN=2 over 1000×1000, ms | `cv::BFMatcher` | 9.071 | 1.916 | 4.70× [4.65, 4.79] | 38.19 | 19.52 | 1.95× [1.94, 1.97] | [features.md](docs/reports/features.md) |
| `countNonZero`, ns/pixel | `cv::countNonZero` | 0.01501 | 0.009270 | 1.62× [1.61, 1.63] | 0.1692 | 0.06365 | 2.66× [2.62, 2.67] | [primitives.md](docs/reports/primitives.md) |
| `goodFeaturesToTrack`, ns/pixel | `cv::goodFeaturesToTrack` | 8.807 | 6.368 | 1.38× [1.35, 1.43] | 58.34 | 24.10 | 2.42× [2.41, 2.42] | [features.md](docs/reports/features.md) |
| dense disparity, ms/frame | `cv::StereoBM` | 12.68 | 10.41 | 1.22× [1.20, 1.24], unpaired | 79.90 | 60.57 | 1.32× [1.32, 1.32], unpaired | [stereo.md](docs/reports/stereo.md) |
| `erode`, 5×5 ellipse, ns/pixel | `cv::erode` | 0.2238 | 0.6985 | 0.319× [0.318, 0.323] | 1.852 | 3.596 | 0.514× [0.510, 0.522] | [primitives.md](docs/reports/primitives.md) |

x86-64 is a desktop Ryzen 5 5600X, aarch64 a Raspberry Pi 4 at a pinned clock. Both columns
are one thread — compare at equal thread counts, or a ratio means nothing. **Every ratio is
OpenCV ÷ binCV**, so above 1× binCV is ahead, and every one is the median of a sweep of whole
process launches with the bootstrap 95% interval those launches put around it: **thirty on
x86-64, ten on the device** (seven for dense disparity), which needs fewer because its clock
is pinned. The dense-disparity row is *unpaired* — its two arms are separate binaries, so its
interval comes from resampling the two sweeps independently.
[Where it does not pay](#where-it-does-not-pay), below, covers the rows where a packed
representation costs more than it saves.

Memory is the other half, and usually the half that decides whether something fits. Each row
is one call's peak working set, from buffer geometry, identical on both architectures. The
dense-disparity row counts output plus scratch on both sides; `cv::StereoBM`'s internal
buffers were not measured, so that row is a lower bound — what the streaming design refuses
to allocate is the 23 MB cost volume, and its whole scratch is 32,352 B:

<!-- figure-check values="OpenCV|binCV|× smaller" source="source" -->
| operation | OpenCV equivalent | OpenCV | binCV | × smaller | source |
|---|---|---|---|---|---|
| `bitwiseAnd` / `Or` / `Xor` / `Not`, bytes | `cv::bitwise_*` | 921,600 | 115,200 | 8.00× | [primitives.md](docs/reports/primitives.md) |
| FAST input plane, bytes | `cv::FAST` on `CV_8U` | 360,960 | 46,080 | 7.83× | [footprint.md](docs/reports/footprint.md) |
| `goodFeaturesToTrack`, bytes | `cv::goodFeaturesToTrack`, binarized | 9,014,976 | 1,580,064 | 5.71× | [footprint.md](docs/reports/footprint.md) |
| `morphologyEx(MORPH_OPEN)`, bytes | `cv::morphologyEx` | 614,400 | 115,200 | 5.33× | [footprint.md](docs/reports/footprint.md) |
| dense disparity, output + scratch, bytes | `cv::StereoBM` | ≥ 721,920 | 393,312 | ≥ 1.84×, a lower bound | [stereo.md](docs/reports/stereo.md) |

There is a CUDA backend too, against `cv::cuda` on the same GPU:

<!-- figure-check values="cv::cuda, ms|binCV, ms|ratio|cv::cuda, KiB|binCV, KiB|× smaller" source="source" -->
| on an RTX 3070 Ti | cv::cuda equivalent | cv::cuda, ms | binCV, ms | ratio | cv::cuda, KiB | binCV, KiB | × smaller | source |
|---|---|---|---|---|---|---|---|---|
| dense disparity, binary entry, per frame | `cv::cuda::StereoBM(64, 9)` | 0.7134 | 0.06400 | 11.2× | 3072.0 | 448.0 | 6.86× | [cuda.md](docs/reports/cuda.md) |
| descriptor matching, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 2.009 | 0.2105 | 9.51× | 8192.0 | 368.0 | 22.3× | [cuda.md](docs/reports/cuda.md) |

**[docs/reports/](docs/reports/README.md) has the whole set** — every operation on both
architectures and the GPU, wins and losses in the same tables, with the machines, the method
and the command that reproduces each row.

### Where it does not pay

**A non-separable structuring element.** A 5×5 ellipse costs one shifted-OR per set element,
and binCV runs it at 0.319× of `cv::erode` on x86-64. The fused kernel shipped at that price
because it holds 76,800 bytes against `cv::erode`'s 614,400 — when speed and footprint
conflict here, footprint wins. A 3×3 element is 1.05× on x86-64 and 1.02× [0.991, 1.04] on
the Pi, a dead heat, at the same 76,800 bytes. ([primitives.md](docs/reports/primitives.md))

**Wide inputs into a bit-sliced filter.** One plane per bit means the work grows with input
depth, and at eight bits a byte kernel's vector unit wins outright — `pyrDown` fed eight bits
runs at 0.0235× on x86-64 and 0.0701× on aarch64. That is the library outside its premise: it
ships as 1 bit in, 3 bits out. ([limits.md](docs/reports/limits.md))

**On the GPU, one path is faster and bigger.** The census dense entry beats
`cv::cuda::StereoBM` at 1.50× on time but holds 4512.0 KiB against its 3072.0 — 1.47×
larger — because the census transform expands eight bits per pixel into a 32-bit descriptor
word. It is the one row in that report where binCV is larger.
([cuda.md](docs/reports/cuda.md))

## Status

**Pre-release. Names and signatures will move.**

## License

Not yet chosen. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the third-party
material the repository contains and the terms each carries.
