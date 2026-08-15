# binCV

**A computer vision library for low-bit-width image frames.**

binCV processes binary, ternary, and few-bit quantized images at their true bit
width — 1 bit per pixel instead of 8 — while keeping OpenCV's API shape. It
targets embedded and mobile CPUs, where memory footprint and energy are the
binding constraints.

```cpp
#include <bincv-cpp/binMat.hpp>

bincv::BinMat<> frame(640, 480);   // 37.5 KiB, not 300 KiB
```

---

## Why

Libraries like OpenCV store a binary image as `CV_8U` — **one byte per pixel to
carry one bit of information**. Every operation then moves and computes on eight
times more data than the image contains. On a memory-constrained device that is
not merely slow, it is disqualifying: the buffers do not fit.

At 640×480, a conventional optical-flow frontend needs roughly 4 MiB for two
frames and their derivatives. The same pipeline over bit-packed planes needs
about 0.5 MiB.

## What it's for

Sensors and pipelines whose frames are genuinely low-bit:

- **Single-photon (SPAD) cameras** — thousands of binary frames per second
- **Event cameras** — binary event-frame representations
- **Binary edge frames for visual odometry** — the driving use case
- **Segmentation masks, structured light, binarized documents**

The immediate goal is a **binary-frame VIO frontend** that runs on embedded and
mobile hardware with a fraction of the memory footprint of a byte-per-pixel
pipeline.

## What it isn't

- Not a general-purpose OpenCV replacement — OpenCV is excellent at 8-bit and float
- Not a quantized neural network runtime
- Not a geometry or estimation library — binCV's boundary is *pixels in,
  features and flow out*
- Not GPU-first — the CPU path is the product

---

## Status

**Early. Under active architectural development.** The container model is being
built out; the operation set is not yet implemented.

See [ROADMAP.md](ROADMAP.md) for what exists and what is planned.

## Build

```bash
cmake -S bincv-cpp -B bincv-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build bincv-cpp/build -j$(nproc)
cd bincv-cpp/build && ctest --output-on-failure
```

The core has **no dependencies**. OpenCV is detected automatically and enables
interop plus the comparison benchmarks; to build without it:

```bash
cmake -S bincv-cpp -B build-core -DBINCV_USE_OPENCV=OFF
```

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Design, rationale, and recorded decisions |
| [ROADMAP.md](ROADMAP.md) | Phases, MVP operation set, success criteria |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Build, test, benchmark, and contribute |

## Background

The direction is motivated by SEAL (ISCA 2025), which showed that binary edge
frames are sufficient input for visual-inertial odometry — but obtained its
efficiency from dedicated in-sensor hardware. binCV asks whether bit-parallel
software can recover that win on commodity embedded and mobile hardware.
