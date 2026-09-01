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
| [GitHub Issues](https://github.com/ryanhou28/bincv/issues) | **START HERE.** All open work, labelled. `experiment` items carry a decision rule |
| [docs/TASKS.md](docs/TASKS.md) | The COMPLETED record — 61 finished tasks and why. Not a backlog any more |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design and every recorded decision (D-records); open questions (E-records) link to their issue |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Measurement log — every result that informed a decision. **Stays in-tree: it is evidence, not discussion** |
| [docs/MEASUREMENT_HARDWARE.md](docs/MEASUREMENT_HARDWARE.md) | Reference-device setup, and what works without it |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phase structure and success criteria |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Build, test, benchmark, conventions |

## How performance and footprint decisions get made

**Measure the alternatives, weigh the result against the project's goals, then
decide — and record all three.** Argument alone does not settle a performance or
footprint question; neither does a benchmark without a stated decision rule.

- **Write the decision rule before measuring.** What result favors which choice,
  written down first. Deciding afterward invites fitting the conclusion to the
  numbers.
- **Compare alternatives**, not one option, on representative workloads.
- **Report memory and speed together** — they trade off, so one alone cannot be
  weighed against goals that conflict.
- **Log it in [EXPERIMENTS.md](docs/EXPERIMENTS.md)** and commit the measurement code.
- **Promote the conclusion** to a D-record in ARCHITECTURE §8.

Experiments run **in the phase whose code they gate**, never at the end. A
decision made without this loop is provisional and must say so. **No decision on
the list is provisional right now** — D-4 was the last one, and X-9 closed it on
the reference device.

If a task needs a performance or footprint choice that no experiment has settled,
**stop and ask** rather than picking one.

## Verify before committing

```bash
./scripts/verify.sh            # ~35 s, four configurations, warnings fatal
./scripts/verify_arm.sh        # aarch64 correctness under emulation; skips without Docker
./scripts/check_arm_syntax.sh  # ~2.5 s, aarch64 SYNTAX only, on the device
```

**A third of `ops/opticalFlow.hpp` is invisible to every x86 build.** D-33's tap
batching and X-40's accumulator live inside `#if BINCV_HAVE_NEON && __aarch64__`, so an
edit there can be structurally broken and still pass all four `verify.sh`
configurations. `verify_arm.sh` covers it but **emulates** aarch64 and needs Docker;
when the daemon is down it skips and that region goes unchecked.

`check_arm_syntax.sh` is the inner loop for that case: it uses the **reference device
as a compiler** — which it is — and compiles one TU with the gate's full warning set in
about two seconds. It checks that the NEON region COMPILES, not that it computes the
right answer, so it does not replace `verify_arm.sh` or a device test run. X-72
abandoned a working refactor after reporting there was "no way to compile for
aarch64"; there was, and it takes two seconds.

`verify.sh` builds and tests four configurations — Release+OpenCV, Release
core-only, `-fno-exceptions` core-only, and **Debug** core-only — with
`-DBINCV_WERROR=ON`, and exits non-zero if anything fails. It starts with a
**gate self-check**: two throwaway builds that are *supposed* to fail, one on a
warning and one on a target that omits `bincv_warnings`. A gate nobody has
watched fail is not known to work.

Read the two numbers in its summary table:

- **CTEST** — cases run.
- **CHECKS** — assertions executed. `+Ns` means checks that configuration cannot
  express in-process (without exceptions `BINCV_CHECK_THROWS` cannot evaluate its
  argument) and covers as death tests instead. A drop in this column is a
  regression even when every ctest case still passes — and it is **checked**, not
  left to a reader: per-suite floors live in `bincv-cpp/tests/expected-checks.txt`
  and a count below one of them fails the run. Raising a floor is a reviewed edit
  (`./scripts/verify.sh --update-checks-baseline`, then commit the diff with the
  change that earned it).

Each configuration also has to *be* the configuration it claims to be:
`verify.sh` reads `BINCV_DEBUG_CHECKS` and `BINCV_EXCEPTIONS_ENABLED` back out of
the built `test_error` binary and fails on a mismatch. `CXXFLAGS=-DNDEBUG` used
to turn the Debug build into a second copy of core-only, silently.

The core-only, no-exceptions and Debug builds regress silently if not run. The
first two are the whole embedded claim; the third is the only thing that
compiles `BINCV_ASSERT`.

**Warnings are project policy, not the script's.** They live in
`bincv-cpp/cmake/BincvWarnings.cmake` and are on in every build:
`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion`. `-Werror` is
off by default so a mid-edit build still finishes; the gate turns it on. Warnings
apply to first-party targets only — never to `bincv_core`'s interface, because a
consumer's warning policy is theirs.

The wiring is not optional: `bincv_assert_warning_policy()` runs at configure
time and **fails the configuration** if any first-party target does not link
`bincv_warnings`. That is a structural check, not a log scan, because a target
compiled with no warning flags emits nothing for a log scan to find — measured,
such a target produced `WARN 0 … PASS` and `ALL CONFIGURATIONS GREEN`. Build
`bincv_add_test_target()` targets and you get the wiring for free.

`-Wconversion` is the load-bearing one: the library is templated on the word
*type* (D-1), so every mask and shift is compiled at 8, 16, 32 and 64 bits, and
an expression that is exact at `uint64_t` can truncate at `uint8_t`. Deliberate
narrowing needs a `static_cast`. That is the point — a cast is where a reader is
told the truncation is intended.

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
([ARCHITECTURE §7](docs/ARCHITECTURE.md#7-the-mvp-operation-set)) — not by OpenCV's
table of contents. An operation no such pipeline calls is deferred, however
prominent it is in OpenCV.

**Out of scope** — a different problem, not a later one: quantized-NN/MAC-heavy
workloads, geometry and estimation (RANSAC, PnP, IMU fusion), connected components,
distance transform, contours, template matching.

**DESCHEDULED is not the same thing, and GPU backends are DESCHEDULED.** A CUDA
prototype is in `bincv-cuda/` and the view/storage model was chosen to keep a device
backend possible ([ROADMAP](docs/ROADMAP.md)). It is not being worked on and no task depends
on it — that is all "descheduled" means. Do not write it up as out of scope, and do not
delete it as though the project had rejected it.

**The input boundary is a rule, not a list**
([ARCHITECTURE §7.8](docs/ARCHITECTURE.md#78-the-input-contract--where-the-operation-set-begins)):
binCV accepts a **single-channel, integer-typed, strided pixel array** and turns it into
an N-bit `QuantMat`. Getting to that array is the caller's. **An operation NOT ON THE PATH
FROM PIXELS TO BITS is somebody else's** — decoding, demosaicing, colour conversion,
each of which leaves the caller exactly as far from bits as before. (Narrower than "any
wide output": §7.1's median is wide-in, wide-out and is an MVP operation.)
Everything from such an array down to bits is binCV's, **including sources wider than 8
bits**, because downconverting first destroys small gradients before the threshold can
see them (§7.8.1).

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
- **A new operation gets a benchmark arm when it is written, even with no caller.** The
  rules above trigger on a performance *claim*, and a kernel nobody calls makes none —
  so it ships correct, untimed, and unoptimised, and nothing notices until something
  calls it. **X-89 measured exactly that**: `medianWide` and `edgeThreshold` were written
  bit-exact against the reference under T5.10/T5.11, benchmarked by nobody, and were
  **78% of the whole frontend** the day T5.8 gave them a caller. A kernel with no
  benchmark is unoptimised by default.
- **A vector arm must be switchable off, and the benchmark must show it is on.** X-89
  also shipped a vector block that a mis-attached `#define` had compiled out, and
  measured three "improvements" against it. Two things catch this and both are cheap: a
  runtime switch so the benchmark can time both arms, and a case where the fast path's
  own gate excludes it — if that case does not report ~1.00×, the fast path is not
  running where you think it is.

## Stop and ask

Surface the question rather than deciding, if:

- A task spec is ambiguous or contradicts ARCHITECTURE.md
- A decision is needed that isn't recorded in
  [ARCHITECTURE §8](docs/ARCHITECTURE.md#8-design-decisions)
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
