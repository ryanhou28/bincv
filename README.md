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

Measured on **EuRoC V1_02**, the full 1710-frame sequence, against OpenCV performing
the same semantic operations on the same binary content stored as `CV_8U`.

| | binCV | OpenCV | |
|---|---|---|---|
| **peak working set** | **436 704 B** | 2 719 832 B | **6.23× smaller** |
| frontend, reference device, **1 thread each** | 6.77 ms/frame | 16.72 | **2.47× faster** |
| frontend, reference device, **4 threads each** | 3.63 ms/frame | 7.07 | **1.94× faster** |
| frontend, x86_64, **1 thread each** | 1.96 ms/frame | 3.06 | **1.57× faster** |
| frontend, x86_64, **4 threads each** | 1.08 ms/frame | 1.46 | **1.35× faster** |

**Every figure names its conditions, and that is not decoration.** The sequence and the
thread count each move these numbers by more than most of the optimisations in this
repository are worth — twice during development a headline was wrong because one of
them went unstated ([D-58](ARCHITECTURE.md#8-design-decisions)).

Both sides are given the same number of threads, which is why the ratio barely moves
between the rows: **the advantage is the implementation, not the parallelism.** Against
OpenCV left at *its* default of one thread, binCV at four reads 4.72× on the device —
true, and not quoted as the headline.

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
| [ARCHITECTURE.md](ARCHITECTURE.md) | design, the input contract, and every design decision with its evidence |
| [ROADMAP.md](ROADMAP.md) | phases and success criteria |
| [TASKS.md](TASKS.md) | the backlog |
| [EXPERIMENTS.md](EXPERIMENTS.md) | the measurement log — every number above traces to an entry here and a committed benchmark |

## How performance claims are made here

Measure the alternatives, weigh the result, record all three — and **write the decision
rule before measuring**. Every performance claim in this repository has a committed
benchmark behind it and an entry in [EXPERIMENTS.md](EXPERIMENTS.md) giving the
platform, the workload and the decision rule that was fixed in advance.

That discipline is not ceremony. It has caught five ceilings that overstated, an
optimisation that was 1.75× in the kernel and 3.3× *slower* on the workload, and three
headline figures that were measuring something other than what they claimed.

## Scope

binCV is the **image processing**. Geometry and estimation — RANSAC, PnP, IMU fusion,
bundle adjustment — belong to the VIO application and are deliberately out of scope, as
are GPU backends.

The input boundary is a rule rather than a list
([ARCHITECTURE §7.8](ARCHITECTURE.md#78-the-input-contract--where-the-operation-set-begins)):
**binCV accepts a single-channel, integer-typed, strided pixel array and turns it into
an N-bit matrix.** Getting to that array — decoding, demosaicing, colour conversion —
is the caller's.

## Status

**Pre-release, and the API is not stable.** It is a research library that has grown a
validated frontend; expect names and signatures to move.

## Licence

Not yet chosen — see [TASKS.md](TASKS.md) T6.1. **Until a licence file exists this code
is "all rights reserved" by default and cannot be used or redistributed.**
