# binCV Roadmap

Phase structure and success criteria. For the executable task breakdown, see
**[TASKS.md](TASKS.md)** — that is where work is picked up. For the design and its
rationale, see [ARCHITECTURE.md](ARCHITECTURE.md).

**The goal every phase serves:** run a binary-frame VIO frontend on embedded and
mobile CPUs with a fraction of the memory footprint and better performance than a
byte-per-pixel pipeline.

---

## Sequencing Principle

**Depth-first on the VIO frontend, not breadth-first across OpenCV's API.**

The MVP is defined by what a real binary-frame VIO pipeline calls
([ARCHITECTURE §7](ARCHITECTURE.md#7-the-mvp-operation-set)). An operation that
no such pipeline calls is not in the MVP, regardless of how prominent it is in
OpenCV. This is the primary defense against scope drift.

---

## Current Status

### Exists and works
- `BinMat<WordType>` templated on the storage word type, 8/16/32/64-bit
- Core builds and tests without OpenCV; interop behind `BINCV_WITH_OPENCV`
- CMake auto-configuration (OpenCV detection, SIMD detection, build type)
- Operations: fill, resize, pad, transpose, countNonZero, sparsity
- Test suite: 282 core checks across all word widths + 21 interop checks, in `ctest`
- Benchmark harness comparing against OpenCV
- Measured 7.83× memory reduction versus `CV_8U`

### Foundation work still required
The existing `BinMat` predates the architecture in its current form. Per
[D-7](ARCHITECTURE.md#d-7-existing-code-is-not-a-constraint), it is a prototype
to be reshaped, not preserved:

- Storage is a hard-coded `std::vector`; needs the `{ptr, stride, owns}` model
- No views — kernels have nowhere to bind
- No bit-plane container; only the N=1 case exists
- Row alignment defaults to 32 bytes; should be word granularity
- `at()` throws unconditionally; should be debug-checked only
- No `BINCV_NO_EXCEPTIONS` path
- Transpose and resize are naive per-pixel loops
- No bitwise operations, no reductions beyond a per-pixel `countNonZero`

### Not started
Everything in the MVP operation set. Bit-plane containers, bulk reductions, the
SEAL-derived frontend kernels, NEON, VIO integration.

---

## Phase 1 — Container Foundation

**Goal:** the data model from [ARCHITECTURE §4](ARCHITECTURE.md#4-data-model),
correct and complete, before any kernel is written on top of it.

Kernels written against the wrong container have to be rewritten. This phase
comes first for that reason alone.

### 1.1 Storage model
- `{pointer, stride, ownership}` storage replacing the embedded `std::vector`
- Owning heap backing as one option
- Caller-provided buffer backing (no heap) — serves MCU, DMA, and future GPU
- Single contiguous allocation for all planes, at fixed offsets

### 1.2 Views
- `BinMatView` / `QuantView<N>`: non-owning `{ptr, width, height, stride}`
- Implicit conversion from owning containers
- **Kernels take views exclusively**, so they compile once per `(WordType, N)`

### 1.3 Bit-plane container
- `QuantMat<N, WordType>` with compile-time N
- `BinMat<WordType>` as the N=1 specialization, with hand-written kernels that
  pay no plane-loop overhead
- Sign-magnitude convention for signed images; ternary as one magnitude plane
  plus one sign plane

### 1.4 Alignment default
- Row stride defaults to word granularity
- Larger alignment as an opt-in per-object argument

### 1.5 Error policy
- Validation throws; `BINCV_NO_EXCEPTIONS` converts to assert/abort
- `at()` bounds-checked in debug, unchecked in release
- Kernels never throw
- Verify the library compiles clean with `-fno-exceptions`

### 1.6 Test restructuring
- Port the existing suite onto the new containers
- Add coverage for views, non-owning storage, and plane indexing
- Google Test, replacing the interim in-repo harness

**Done when:** `QuantMat<N>` and views exist and are tested; `BinMat` is the N=1
specialization; the library builds and passes with `-fno-exceptions` and no heap
allocation; alignment defaults to word granularity.

---

## Phase 2 — Bit-Parallel Primitives

**Goal:** the closed kernel vocabulary from
[ARCHITECTURE §6.1](ARCHITECTURE.md#61-bit-parallel-primitives), plus the
correctness machinery that every later phase depends on.

### 2.1 Equivalence harness
Build this **first**, before the kernels it validates. Every Tier 1 operation
asserts bit-exactness against the equivalent OpenCV expression on the same
content. Cheap now; it is what makes "same accuracy" a claim rather than an
assertion.

### 2.2 Logic operations
`bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`. Tier 1 semantics.
Word-wise over contiguous storage. The first real test of the performance thesis.

### 2.3 Shifts
Horizontal shifts with cross-word carry; vertical shifts as row offsets. Boundary
handling per `BorderType`. Morphology and the derivative kernels are both built
from these.

### 2.4 Bulk reductions
Per [D-6](ARCHITECTURE.md#d-6-bulk-only-reductions), **no per-word popcount is
exposed.** The API provides:
- whole-image and region population counts
- **masked** population counts
- **windowed** counts sized for the LK covariance

The windowed and masked forms are required by
[ARCHITECTURE §7.5](ARCHITECTURE.md#75-lk-gradient-covariance) and belong in the
MVP, so the reduction interface is designed for them now rather than growing a
second interface later. Experiment [E-3](ARCHITECTURE.md#9-open-questions-and-planned-experiments)
determines whether incremental accumulation state is exposed.

### 2.5 Majority and thresholded counts
`maj3` and the bit-sliced adder network for thresholding small counts. Feeds
denoise and pyramid downsample.

**Done when:** the primitive vocabulary is complete, Tier 1 operations are
bit-exact against OpenCV, and reductions are bulk-only with masked and windowed
forms.

---

## Phase 3 — VIO Frontend Operations

**Goal:** the operation set a binary-frame VIO frontend actually calls.

### 3.1 Denoise — median of 3
Majority of three; one expression per word.

### 3.2 Pyramid downsample — box 2×2
2×2 popcount and threshold. Pyramid construction over bit-planes.

### 3.3 Edge filter / threshold
Produces 1-bit frames from higher-precision input, for pipelines that binarize on
the host.

### 3.4 Binarized spatial derivative
Shift-and-mask producing sign-magnitude ternary. The first operation whose output
is a multi-plane `QuantMat`, and therefore the real test of the container design.

### 3.5 LK gradient covariance
`ΣIx²`, `ΣIy²`, `ΣIxIy` as masked population counts. The load-bearing operation.

### 3.6 Corner response
Built on the same covariance machinery.

### 3.7 Morphology
`erode`, `dilate`, `morphologyEx` from shifts and logic. Tier 1 semantics.

### 3.8 Optical flow — hybrid
Per [ARCHITECTURE §7.9](ARCHITECTURE.md#79-the-known-hard-problem-subpixel-interpolation),
route (b): bit-parallel window extraction and covariance accumulation,
floating-point solve. Preserves the accuracy result that motivates the project.

**Done when:** a complete binary frontend — denoise, pyramid, derivative, corner
detection, tracking — runs end to end on real image sequences.

---

## Phase 4 — Validation

**Goal:** produce the result the project exists to produce.

### 4.1 Accuracy
Trajectory accuracy of a binCV-backed frontend versus the byte-per-pixel
equivalent, on real sequences. The claim is *equivalent* accuracy; anything less
must be understood before proceeding.

### 4.2 Peak footprint
Peak working-set measurement of the full frontend, end to end
([ARCHITECTURE §10.4](ARCHITECTURE.md#104-the-metric-that-matters)). This is the
headline number, not per-buffer ratios.

### 4.3 Performance
Against OpenCV performing the same semantic operations on the same binary content
stored as `CV_8U` ([ARCHITECTURE §10.3](ARCHITECTURE.md#103-benchmark-denominator)).

### 4.4 Run the planned experiments
Settle [E-1 through E-4](ARCHITECTURE.md#9-open-questions-and-planned-experiments)
with committed benchmarks. E-1 in particular decides whether the alignment default
stands and whether a profile system is worth building.

**Done when:** accuracy, footprint, and performance are measured and published in
the repository, and the provisional decisions have data behind them.

---

## Phase 5 — Platform Hardening

**Goal:** make Tier 1 fast and Tier 2 correct.

### 5.1 NEON kernels
NEON as the reference SIMD implementation
([ARCHITECTURE §6.3](ARCHITECTURE.md#63-simd-strategy)). Bulk reductions keeping
data in vector registers, avoiding the GPR↔NEON round trip.

### 5.2 Cross-compilation and validation on Cortex-A
Build for aarch64; validate on real ARM hardware. Track binary size.

### 5.3 x86 portability path
SSE4.2 `popcnt`; AVX2 via `pshufb` nibble-LUT where it wins. Development and
comparison platform, not a deployment target.

### 5.4 Cortex-M correctness
Confirm the commitment from
[ARCHITECTURE §2](ARCHITECTURE.md#tier-2--cortex-m-class-correctness-only):
compiles and runs correctly with `-fno-exceptions`, no heap, scalar kernels.
Track code size. **No Cortex-M-specific optimization** — deliberately unscoped.

**Done when:** NEON kernels are the reference path and measured on ARM hardware;
Tier 2 correctness is verified in CI.

---

## Phase 6 — Deferred

Not scoped. Listed so the ordering is deliberate rather than accidental.

- **GPU backends.** Jetson runs the CPU path today. The view/storage model keeps
  zero-copy viable later without an API break.
- **Python bindings.**
- **Route (a) fully bit-parallel tracking** — census/Hamming block matching
  ([E-6](ARCHITECTURE.md#9-open-questions-and-planned-experiments)). The research
  upside, explored only after the hybrid frontend is validated.
- **Connected components, distance transform, contours, template matching.** Not
  called by the VIO frontend. Add only when an application demands them.

---

## Dependency Order

```
Phase 1  Container foundation
              |
              v
Phase 2  Primitives + equivalence harness
              |
              v
Phase 3  VIO frontend operations
              |
              v
Phase 4  Validation  <-- the project's result
              |
              v
Phase 5  Platform hardening
              |
              v
Phase 6  Deferred (GPU, bindings, research routes)
```

Phase 1 gates everything: kernels written against the wrong container get
rewritten. Phase 2's equivalence harness precedes the kernels it validates.
Phase 4 is where the thesis is confirmed or refuted, so nothing beyond it should
start until it produces numbers.

---

## Success Criteria

The project has succeeded when, on Tier 1 hardware, a binary-frame VIO frontend
built on binCV demonstrates:

1. **Equivalent trajectory accuracy** to the byte-per-pixel pipeline
2. **Several-fold smaller peak memory footprint**, measured end to end
3. **Faster execution** on the bit-parallel operation set, against the
   byte-per-pixel denominator

That is the answer to the question in
[ARCHITECTURE §1](ARCHITECTURE.md#the-motivating-result): whether bit-parallel
software can recover on commodity hardware what dedicated silicon achieved.
