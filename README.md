# binCV

**Image processing for low-bit-width frames — binary, ternary, few-bit — at their true
bit width.** One bit per pixel, not eight, with OpenCV's API shape.

Built for the vision frontend of a visual-inertial odometry (VIO) system on embedded
and mobile CPUs, where memory footprint and energy bind before FLOPs do.

```cpp
#include "bincv-cpp/ops/pack.hpp"
#include "bincv-cpp/ops/edge.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"

// A camera hands you bytes. No OpenCV needed, and no 8-bit intermediate.
bincv::BinMat<uint32_t> edges(width, height);
bincv::edgeThreshold(pixels, width, height, stride, edges.view(), uint8_t{17});

// ... build a pyramid, take derivatives, track. See GETTING_STARTED.md.
```

## Why

A binarised 752×480 frame is 45 kB at one bit per pixel and 361 kB at one byte. Storing
it as `CV_8U` — which is what you do today, because that is what the libraries take —
spends 8× the memory to hold values that are always 0 or 1, and then does 8-bit
arithmetic on them.

binCV stores the bits and operates on them 32 or 64 pixels at a time, in ordinary
integer registers.

## Where it stands

Measured on **EuRoC V1_02**, against OpenCV performing the same semantic operations on
the same content. **Both frontends start from the same grayscale frame and each builds
its own binary one** — the sensor stage (median, then gradient-threshold edge filter) is
inside both timings, so this is end to end rather than from a binary frame someone else
prepared.

| **one thread each side** | binCV | OpenCV | |
|---|---|---|---|
| **peak working set** | **436 704 B** | 2 719 832 B | **6.23× smaller** |
| frontend, reference device (Cortex-A72) | **4.88 ms/frame** | 29.63 | **6.07× faster** |
| frontend, x86_64 | **1.07 ms/frame** | 4.46 | **4.16× faster** |

**Every figure names its conditions, and that is not decoration.** The sequence, the
thread count and whether the sensor stage is inside the timing each move these numbers
by more than most of the optimisations in this repository are worth. Three times during
development a headline was wrong because one of them went unstated — including one that
stood in this README ([D-58](docs/ARCHITECTURE.md#8-design-decisions)).

**Both sides get the same thread count**, which is the only comparison that isolates the
implementation from the parallelism. binCV threads through a caller-installed backend
and is serial by default; at four threads each on x86 the ratio narrows to roughly 3×,
because OpenCV has more byte-work to spread across cores. The reference device is
measured pinned to a single core — deliberately, so OpenCV's threads cannot escape it
either — so no thread-scaling figure is quoted for it.

The x86 rows were taken on an otherwise idle machine; under load the ratio *rises*,
because OpenCV degrades faster than binCV does. The quiet reading is the conservative
one and it is the one quoted.

## What is in it

- **Sensor stage** — `pack` (8- and 16-bit sources, three 1-bit rules and an N-bit
  quantisation policy, streaming, no OpenCV), `medianWide`, `edgeThreshold`. The last
  two are bit-exact against the reference pipeline's filters.
- **Frontend** — pyramid, derivatives, gradient covariance, corner response,
  `goodFeaturesToTrack`, and pyramidal Lucas–Kanade tracking on bit-planes.
- **Features** — FAST (including a **bit-plane** form on binCV's own type, bit-exact
  with `cv::FAST` corner for corner), BRIEF descriptors, Hamming matching.
- **Primitives** — logic, shifts, bulk reductions, morphology, resampling, bit-slicing,
  block matching, thresholding and binarisation.
- **Interop** — `cv::Mat` in and out when OpenCV is present; PGM and raw buffers when it
  is not.

## Supported platforms

| | status |
|---|---|
| **x86_64** | measured. `POPCNT` on by default; AVX2 selected at run time, so the baseline ISA is unchanged |
| **aarch64** | measured. NEON. The reference measurement device |
| 32-bit ARM (incl. Cortex-M) | **not built or tested.** CMake has a code path; nobody has compiled it |
| RISC-V | not attempted |

## Building

```bash
cmake -S bincv-cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./scripts/verify.sh          # four configurations, warnings fatal
```

OpenCV is optional: `-DBINCV_USE_OPENCV=OFF` builds the library and its tests without
it. Three of the four gated configurations are core-only, because that is the
configuration the memory argument is about.

## Documentation

| | |
|---|---|
| [GETTING_STARTED.md](GETTING_STARTED.md) | build, test, benchmark, and a tour of the operation set |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | design, the input contract, and every design decision with its evidence |
| [ROADMAP.md](docs/ROADMAP.md) | phases and success criteria |
| [TASKS.md](docs/TASKS.md) | the backlog |
| [docs/API.md](docs/API.md) | **the API reference** — every public entry point, its brief and its API tier |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | the measurement log — every number above traces to an entry here and a committed benchmark |
| [docs/README.md](docs/README.md) | which document is for whom, and how the D/E/X record system works |

## How performance claims are made here

Measure the alternatives, weigh the result, record all three — and **write the decision
rule before measuring**. Every performance claim in this repository has a committed
benchmark behind it and an entry in [EXPERIMENTS.md](docs/EXPERIMENTS.md) giving the
platform, the workload and the decision rule that was fixed in advance.

That discipline is not ceremony. It has caught several ceilings that overstated, an
optimisation that was 1.75× in the kernel and 3.3× *slower* on the workload, a vector
kernel that a mis-attached `#define` had compiled out of three consecutive "improvements",
and headline figures that were measuring something other than what they claimed —
including one that stood in this file.

## Scope

binCV is the **image processing**. Geometry and estimation — RANSAC, PnP, IMU fusion,
bundle adjustment — belong to the VIO application and are deliberately out of scope, as
are GPU backends.

The input boundary is a rule rather than a list
([ARCHITECTURE §7.8](docs/ARCHITECTURE.md#78-the-input-contract--where-the-operation-set-begins)):
**binCV accepts a single-channel, integer-typed, strided pixel array and turns it into
an N-bit matrix.** Getting to that array — decoding, demosaicing, colour conversion —
is the caller's.

## Status

**Pre-release, and the API is not stable.** It is a research library that has grown a
validated frontend; expect names and signatures to move.

## Licence

Not yet chosen — see [TASKS.md](docs/TASKS.md) T6.1. **Until a licence file exists this code
is "all rights reserved" by default and cannot be used or redistributed.**
