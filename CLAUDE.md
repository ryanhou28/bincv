# binCV — Working Notes

## What this project is

binCV processes **low-bit-width image frames** — binary, ternary, few-bit
quantized — at their true bit width (1 bit per pixel, not 8), while keeping
OpenCV's API shape. Targets embedded and mobile CPUs, where memory footprint and
energy bind.

**Performance and memory footprint are co-equal goals. When they conflict and no
explicit choice has been made, memory wins.**

The near-term goal is a binary-frame VIO frontend that runs on embedded/mobile
hardware with a fraction of the memory of a byte-per-pixel pipeline.

## Where to look

| | |
|---|---|
| [TASKS.md](TASKS.md) | **Start here.** Executable backlog; pick the lowest-numbered task whose deps are `DONE` |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Design and recorded decisions (D-1…D-9), open experiments (E-1…E-7) |
| [ROADMAP.md](ROADMAP.md) | Phase structure and success criteria |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Build, test, benchmark, conventions |

## Verify before committing

```bash
./scripts/verify.sh          # once T1.8 exists
```

Until then, all three configurations must build warning-free and pass:

```bash
cmake -S bincv-cpp -B bincv-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build bincv-cpp/build -j$(nproc) && (cd bincv-cpp/build && ctest --output-on-failure)

cmake -S bincv-cpp -B bincv-cpp/build-core -DCMAKE_BUILD_TYPE=Release -DBINCV_USE_OPENCV=OFF
cmake --build bincv-cpp/build-core -j$(nproc) && (cd bincv-cpp/build-core && ctest --output-on-failure)

cmake -S bincv-cpp -B bincv-cpp/build-noexcept -DBINCV_USE_OPENCV=OFF -DCMAKE_CXX_FLAGS="-fno-exceptions"
cmake --build bincv-cpp/build-noexcept -j$(nproc) && (cd bincv-cpp/build-noexcept && ctest --output-on-failure)
```

The core-only and no-exceptions builds regress silently if not run. They are the
whole embedded claim.

## Hard rules

These are settled decisions. Do not relitigate them mid-task; if one seems wrong,
say so rather than working around it.

- **Kernels take views, never owning containers.** A kernel compiles once per
  `(WordType, N)` and must not care about its arguments' alignment or ownership.
- **Never expose a per-word popcount.** Reductions are bulk only — region, masked,
  or windowed. On aarch64 a per-word popcount pays two register-domain crossings
  per 64 pixels. Internal helpers stay in `impl::`.
- **No heap allocation inside kernels.** Scratch buffers are caller-provided.
- **Value semantics** — copy means deep copy. No reference counting. Sharing is a
  view.
- **Padding bits stay zero.** Any operation that writes whole words past `width`
  must clear them, or word-wise reductions over-count.
- **Tier 1 operations must be bit-exact against OpenCV**, proven by a test.
  State the API tier in every public docstring.
- **Existing code is a prototype.** Replace it where it conflicts with the
  architecture; breaking current behavior and tests is expected.

## Scope discipline

The MVP is defined by what a binary-frame VIO frontend calls
([ARCHITECTURE §7](ARCHITECTURE.md#7-the-mvp-operation-set)) — not by OpenCV's
table of contents. An operation no such pipeline calls is deferred, however
prominent it is in OpenCV.

Out of scope, deliberately: quantized-NN/MAC-heavy workloads, geometry and
estimation (RANSAC, PnP, IMU fusion), GPU backends, connected components,
distance transform, contours, template matching.

**Do not mention specific vendor hardware or toolchains in this repo.** Platform
language stays generic: Cortex-A / Cortex-M, "memory-constrained embedded
targets".

## Reference implementation

Ground-truth semantics for the VIO frontend operations live at
`~/seal/SEAL/SEAL_HybVIO/HybVIO/SEAL/`. When a task says to match reference
behavior, **read that code** rather than inferring it — notably
`src/temporal_processing/denoise.cpp`, `src/keypoint_tracking/gradients.cpp`,
`src/keypoint_tracking/pyramids.cpp`, and `SEAL/seal_params.yaml` for the
configuration the paper actually used.

## Benchmarking

- **Always Release.** CMake defaults to it; do not benchmark other build types.
- **Denominator:** OpenCV doing the *same semantic operation on the same binary
  content stored as `CV_8U`* — that is what a user does today without binCV.
- **Report peak working set**, not per-buffer ratios. A target either fits or it
  does not.
- **Commit the benchmark.** Every performance claim must be reproducible.

## Stop and ask

Surface the question rather than deciding, if:

- A task spec is ambiguous or contradicts ARCHITECTURE.md
- A decision is needed that isn't recorded in
  [ARCHITECTURE §8](ARCHITECTURE.md#8-design-decisions)
- Something in scope turns out to be impossible as specified
- **A measurement contradicts a documented claim** — this is valuable; report it
  rather than adjusting the code to fit the doc
- The work would add an operation outside the MVP set

## Style

- OpenCV conventions: `camelCase` functions, `PascalCase` types, `UPPER_CASE`
  constants, lowercase namespaces, destination as out-parameter.
- Tier 3 operations (no OpenCV equivalent) must **not** borrow OpenCV names.
- Match the comment density and idiom of surrounding code.
- Commit messages: `[area] Summary`, then what changed and why.
