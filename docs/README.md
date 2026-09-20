# binCV documentation

| | for | what it is |
|---|---|---|
| [../README.md](../README.md) | anyone | what binCV is, headline comparisons, how to build |
| [../GETTING_STARTED.md](../GETTING_STARTED.md) | a user | build it, use it, conventions |
| [API.md](API.md) | a user | **the API reference** — every public entry point, its brief and its tier |
| [ARCHITECTURE.md](ARCHITECTURE.md) | a contributor | how the library is put together, and why |
| [reports/](reports/README.md) | anyone weighing it up | **what it costs and what it saves** — every operation measured against the OpenCV call it replaces, on x86-64, aarch64 and a GPU, wins and losses in the same tables |

Deciding whether to adopt binCV: start at [reports/README.md](reports/README.md) — its *At a
glance* is one row per operation, both sides' measured figures with the ratio beside them.
About to write a call: [API.md](API.md) names the entry point and
[GETTING_STARTED.md](../GETTING_STARTED.md) shows the shape of a program that uses it. About
to change the library: [ARCHITECTURE.md](ARCHITECTURE.md) says why the thing you are changing
is the way it is.

Three pages inside the reports are prerequisites rather than results.
[reports/methodology-memory.md](reports/methodology-memory.md) names the four different
quantities that get called "memory" and the four measurement errors this project published;
[reports/methodology-timing.md](reports/methodology-timing.md) is its twin on the other axis;
and [reports/README.md](reports/README.md)'s *How the numbers are taken* states the
denominator rule, the one-thread rule and the correctness check that runs before any timing.

The GPU backend is documented in three places that do not repeat each other:
[ARCHITECTURE.md](ARCHITECTURE.md) §8.5 is the design — why the representation is shared and
the kernels are forked; [../backends/cuda/README.md](../backends/cuda/README.md) is how to
build and use it; [reports/cuda.md](reports/cuda.md) is what it measures against `cv::cuda`.

## What is in it

- **Getting to bits** — packing from 8- and 16-bit sources, a wide median, and a
  gradient-threshold edge filter that writes bit-planes directly.
- **Primitives** — logic, shifts, bulk and windowed reductions, morphology, resampling,
  bit-sliced arithmetic, thresholding.
- **Features and tracking** — pyramid, derivatives, gradient covariance, corner response,
  `goodFeaturesToTrack`, pyramidal Lucas–Kanade, FAST (wide and bit-plane), sub-pixel
  refinement.
- **Descriptors and stereo** — intensity-centroid orientation, BRIEF and steered
  (rotation-compensated) BRIEF with caller-supplied patterns (`cv::ORB`'s own table included,
  byte-exact), Hamming matching plain and prior-gated, sparse rectified stereo matching with
  sub-pixel disparity, and dense disparity via the census transform — streamed, so the cost
  volume never exists.
- **Geometry** — RANSAC over caller-owned scratch: 2D affine, and the five-point essential
  matrix.
- **Interop** — `cv::Mat` in and out when OpenCV is present; raw buffers and PNM (`P4`, `P5`)
  when it is not. binCV links no codec on any target: a camera's Y plane, a V4L2 buffer and a
  sensor's DMA rows are already the input contract.

`examples/slam_frontend.cpp` runs the feature half end to end. It is a program this project
wrote to exercise the operations, not an operation binCV offers — see
[reports/README.md](reports/README.md#assembled-pipelines).

## Where binCV runs

The status column is what has actually been done, not what is supported in principle.

| target | vector path | status |
|---|---|---|
| **x86-64** — desktops, servers | AVX2, selected at run time; `POPCNT` required | measured — Ryzen 5 5600X |
| **aarch64** (64-bit Arm Cortex-A) — phones, SBCs, embedded Linux | NEON | measured — Raspberry Pi 4 at a pinned clock, the reference device |
| **Arm Cortex-M**, bare metal — microcontrollers, no OS | none: no NEON, no popcount instruction | built and run on an STM32H753ZI (Cortex-M7); bit-exact against the host. No OpenCV on that part, so no comparison — see below |
| **armv7-a** (32-bit Arm Cortex-A) — older phones, SBCs | 32-bit NEON | **not built.** No toolchain here and no measurement. A different target from Cortex-M above |
| **RISC-V** | none | **not built.** No measurement |
| **CUDA** — NVIDIA GPUs | [a separate backend](../backends/cuda/), device-typed, never a drop-in dispatch target | measured — RTX 3070 Ti, against `cv::cuda` ([reports/cuda.md](reports/cuda.md)) |

On the Cortex-M7: a 752×480 frame is 46,080 bytes, and a 320×240 dense-disparity map runs
in 6.5 KB of scratch — 825 ms at the 64 MHz reset clock, with no figure at its full clock.

A 64-bit OS is a requirement for the measured Cortex-A results rather than a preference: on
32-bit Arm every `uint64_t` operation is synthesised from 32-bit pairs, which would measure
the compiler rather than the machine.

Log `bincv::simdStatusString()` once at start-up — it names every vector path and says
whether it is active. **Compare at equal thread counts** when you benchmark: binCV is serial
unless you install a threading backend.

## API tiers

Every public entry point states one:

- **Tier 1** — bit-exact with the OpenCV function it names, proven by a test.
- **Tier 2** — OpenCV's role and call shape, different numerics.
- **Tier 3** — no OpenCV equivalent. **These deliberately do not borrow OpenCV names**,
  because a familiar name on different semantics is worse than an unfamiliar one.

## The API reference

[API.md](API.md) is **generated from the headers**:

```bash
python3 scripts/gen_api_index.py     # writes docs/API.md
```

It is committed so a reader in a browser has one without running anything. Every line is the
`@brief` from the declaration itself, so it cannot drift from the code without the code
changing — regenerate it in the same commit as any signature change.

For full signatures, parameters and the rationale paragraphs — often the useful part — read
the header, or generate the HTML:

```bash
doxygen docs/Doxyfile     # writes docs/api/html; needs doxygen installed
```
