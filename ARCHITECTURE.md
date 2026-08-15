# binCV Architecture

## Thesis

> **binCV is a computer vision library for low-bit-width image frames — binary,
> ternary, and few-bit quantized — that keeps OpenCV's API shape while storing
> pixels at their true bit width and computing on them bit-parallel. It targets
> embedded and mobile CPUs first, where memory footprint and energy are the
> binding constraints.**

Every design decision in this repository should be traceable to that sentence.
When a decision is not obviously implied by it, record the reasoning in
[Design Decisions](#8-design-decisions) below.

---

## 1. Scope

### What binCV is

A CV library whose data model is **N bits per pixel**, where N is small
(typically 1–4), and whose kernels operate on many pixels per machine word using
bitwise logic and population counts.

### What binCV is not

| Not this | Why |
|---|---|
| A general-purpose OpenCV replacement | OpenCV is excellent at 8-bit and float. binCV only wins where pixels are genuinely low-bit. |
| A quantized neural network runtime | MAC-heavy workloads favor SWAR packing over bit-planes (see [D-2](#d-2-bit-planes-over-swar-packing)). Explicit non-goal. |
| A geometry or estimation library | RANSAC, PnP, essential-matrix solvers, IMU fusion belong to Eigen and the application. binCV's boundary is **pixels in, features and flow out**. |
| A GPU-first library | GPUs are a later target. The CPU path is the product. |

### The problem

Libraries like OpenCV store a binary image as `CV_8U` — **one byte per pixel to
carry one bit of information**. Every operation then moves and computes on eight
times more data than the image contains. On a memory-constrained device that is
not merely slow, it is disqualifying: the buffers do not fit.

binCV stores that image at 1 bit per pixel and computes on 32 or 64 pixels per
instruction.

### The motivating result

The SEAL work (ISCA 2025) showed that **binary edge frames are sufficient input
for visual-inertial odometry** — feed them to an existing VIO framework and
tracking accuracy holds up. But SEAL obtained its energy and latency wins from
dedicated in-sensor hardware. Its software artifact is a functional simulator:
the "binary" images are binary-*valued* `CV_8U`, and every downstream stage
(corner detection, optical flow, pyramids) is ordinary byte-per-pixel OpenCV code.

That leaves a specific, unanswered question, and it is the one binCV exists to
answer:

> **SEAL showed binary frames are sufficient. SEAL needed custom silicon to make
> them efficient. Can bit-parallel software recover that win on commodity
> embedded and mobile hardware?**

Answering it requires a library that is bit-parallel end to end, which is what
this repository is.

---

## 2. Target Platforms

Two tiers, with genuinely different constraints. They are not a continuum.

### Tier 1 — Cortex-A class (primary)

Raspberry Pi, Jetson (CPU), Android/iOS phones, ARM SBCs.

- `aarch64` with NEON, uniformly. **One SIMD target covers this entire tier.**
- Full OS, heap, exceptions available.
- This is where the VIO story lives. Optimize here.

Desktop x86_64 is supported as a development and comparison platform, not as a
deployment target.

### Tier 2 — Cortex-M class (correctness only)

Microcontrollers.

**Commitment:** binCV compiles and runs correctly with `-fno-exceptions` and
without a heap. Scalar kernels, static or caller-provided buffers.

**Not committed:** Cortex-M-specific optimization (DSP extensions, hand-tuned
SWAR). Deliberately unscoped — revisit only if a concrete application demands it.

Constraints that shape the API for this tier:

1. **No exceptions.** See [error policy](#53-error-policy).
2. **No heap.** See [storage model](#43-storage-model-and-views).
3. **Code size is often the binding constraint**, before RAM. Keep the template
   instantiation surface small.
4. **No popcount instruction** — the compiler emits a ~15-instruction SWAR
   sequence. Still roughly 0.25 instructions per pixel versus several for a
   per-pixel loop, so the approach holds; the margin is just smaller.

---

## 3. Design Principles

1. **Bit width is the point.** If a design stores more bits per pixel than the
   data contains, it is wrong regardless of how fast it benchmarks.
2. **Memory footprint and performance are co-equal goals.** When they conflict
   and no explicit choice has been made, **favor memory.** A user who wants raw
   throughput and has memory to spare already has OpenCV.
3. **OpenCV's shape, not OpenCV's internals.** Match names, call conventions and
   semantics. Do not match runtime type erasure or dynamic dispatch.
4. **Compile-time over runtime.** Bit width, word type, and plane count are
   template parameters. No dispatch tables, no dynamic allocation in kernels.
5. **Measure before optimizing, and record the measurement.** Every performance
   claim in this repository must be reproducible from a committed benchmark.
6. **The MVP is a VIO frontend, not a feature checklist.** Depth-first on the
   operations a real pipeline calls, not breadth-first across OpenCV's API.

---

## 4. Data Model

### 4.1 Bit-plane representation

An N-bit image is stored as **N bit-planes**. Plane *i* holds bit *i* of every
pixel, packed into machine words. A 1-bit image is one plane; a ternary
derivative is two.

```
QuantMat<3>  (3-bit pixels, 8 pixels shown)

pixel:      5   2   7   0   3   1   6   4
           ---------------------------------
plane 2:    1   0   1   0   0   0   1   1     <- MSB
plane 1:    0   1   1   0   1   0   1   0
plane 0:    1   0   1   0   1   1   0   0     <- LSB
```

Memory is exactly N bits per pixel. Logic operations apply per plane and are
free. Reductions are population counts. Arbitrary N is expressible.

Rejected alternative: SWAR sub-byte packing. See [D-2](#d-2-bit-planes-over-swar-packing).

### 4.2 Signed values: sign-magnitude

Signed low-bit images (gradients, differences) use **magnitude planes plus one
sign plane**, not two's complement.

Ternary — the case that matters most — is then one magnitude plane and one sign
plane:

```
value   mag  sign
  0      0    -
 +1      1    0
 -1      1    1
```

This makes the Lucas-Kanade gradient covariance fall out directly as population
counts (see [§7.5](#75-lk-gradient-covariance)), and it degrades gracefully:
ternary is simply the one-magnitude-plane case of the general signed form.
Two's complement would require a comparator network for the same result and
would make ternary a special case rather than an instance.

### 4.3 Storage model and views

Storage is **`{pointer, stride, ownership}`**, not a `std::vector` baked into the
container. An owning heap allocation is one backing option among several.

This single mechanism serves four independent needs, which is why it is core
rather than an add-on:

| Need | Served by |
|---|---|
| MCU static allocation (no heap) | caller-provided buffer |
| DMA / sensor buffer ingest | non-owning wrap, zero copy |
| GPU zero-copy and unified memory (later) | non-owning wrap |
| Kernels that must not care about alignment | view with runtime stride |

**Views are the kernel interface.** A view is a non-owning
`{ptr, width, height, stride}`. Kernels take views, so a kernel compiles once per
`(WordType, N)` regardless of the alignment or ownership of its arguments, and
matrices of differing alignment interoperate freely.

**All planes live in one contiguous allocation** at fixed offsets, not N separate
allocations. One allocation matters on constrained targets, improves locality,
and keeps external-buffer wrapping tractable.

### 4.4 Container hierarchy

```
storage {ptr, stride, owns}
  |
  +-- BinMatView / QuantView<N>      non-owning, runtime stride, kernel interface
  |
  +-- QuantMat<N, WordType>          owning, compile-time N
        |
        +-- BinMat<WordType>         alias for the N=1 specialization
```

`BinMat` remains a distinct name because the 1-bit path deserves hand-written
kernels with no plane-loop overhead, and because it carries the project's
identity.

### 4.5 Row alignment

**Default: word granularity.** A row's stride is `ceil(width / WordBits)` words,
with no padding beyond that. Larger alignment is an opt-in per-object argument.

Word-granularity padding is inherent and free, and it already provides the only
property kernels need: that the trailing partial word can be read and written
whole, without a bounds check.

Aggressive row alignment was measured and rejected as a default. See
[D-4](#d-4-word-granularity-alignment-by-default) — this is flagged for
re-measurement in [§9](#9-open-questions-and-planned-experiments).

### 4.6 Memory arithmetic

Why the data model is the whole argument, at 640×480:

| Buffer | Byte-per-pixel | binCV | Ratio |
|---|---|---|---|
| One frame (level 0, 1-bit) | 300 KiB | 37.5 KiB | 8× |
| 4-level pyramid | ~400 KiB | ~78 KiB | ~5× |
| LK spatial derivative, level 0 (2ch `CV_16S`) | **1.2 MiB** | ~150 KiB (4 planes) | 8× |
| Two frames + derivatives | **~4 MiB** | **~0.6 MiB** | ~6× |

The pyramid ratio is lower than 8× because **pyramid levels are not binary** — see
[§7.2](#72-pyramid-downsample--box-22). Level 0 is 1-bit; upper levels need 3–5
bits. binCV chooses that quantization, which is why the container is N-bit rather
than binary-only.

On a device with a few megabytes of memory, the conventional path spends its
entire budget before the odometry backend receives a byte. The derivative buffer
alone exceeds what many targets have. **This is the argument for the library, and
it is a memory argument, not a speed argument.**

---

## 5. API Design

### 5.1 Three tiers

Adding quantization creates a category with no OpenCV counterpart, so the
compatibility promise has to be stated per tier.

**Tier 1 — identical semantics.** `bitwise_and/or/xor/not`, `erode`, `dilate`,
`morphologyEx`, `countNonZero`, `copyMakeBorder`. Drop-in for OpenCV users;
results are bit-exact against OpenCV on equivalent content. Verified by the
equivalence harness ([§10.2](#102-equivalence-harness)).

**Tier 2 — same name, specialized numerics.** `calcOpticalFlowPyrLK`,
`goodFeaturesToTrack`, `pyrDown`. Same call shape and role, deliberately
different math. **Not** bit-exact against OpenCV; validated against downstream
task accuracy instead.

**Tier 3 — no OpenCV equivalent.** Plane packing and unpacking, bit-sliced
arithmetic, census/Hamming matching, masked window reductions. These must **not**
borrow OpenCV names, precisely so that Tier 1's drop-in promise stays credible.

### 5.2 Naming conventions

Follow OpenCV: `camelCase` functions, `PascalCase` types, `UPPER_CASE`
constants, lowercase namespaces, destination-as-out-parameter
(`op(src, dst, ...)`).

### 5.3 Error policy

Split validation from the hot path, matching OpenCV's own convention.

**Validation** — construction, `resize`, argument checking. Throws by default.
Called at setup, not per pixel. Compiling with `BINCV_NO_EXCEPTIONS` converts
these to assert/abort, which is what makes Tier 2 platforms viable.

**Element access** — `at()` is bounds-checked in debug builds and unchecked in
release, exactly as `cv::Mat::at` behaves. This removes throws from hot paths
entirely and lets release builds inline access to a shift and a mask.

Kernels never throw. A kernel that receives inconsistent views is a programming
error, caught by assertion in debug.

---

## 6. Compute Strategy

### 6.1 Bit-parallel primitives

The kernel vocabulary is small and closed:

| Primitive | Form |
|---|---|
| logic | `&`, `\|`, `^`, `~` per plane |
| shift | word shifts with cross-word carry (horizontal), row offset (vertical) |
| majority / median | `(a&b) \| (b&c) \| (a&c)` for 3 inputs |
| threshold on a count | bit-sliced adder network, then compare |
| reduction | population count over a region or mask |

Nearly every operation in the MVP set is a composition of these.

### 6.2 Reductions are bulk-only

**binCV must not expose a per-word popcount primitive.** This is a hard interface
rule derived from measurement.

On `aarch64` — the primary target — there is no scalar popcount instruction.
`__builtin_popcountll` compiles to:

```asm
fmov   d0, x0          ; GPR -> NEON  (domain crossing)
cnt    v0.8b, v0.8b    ; the actual popcount
uaddlv h0, v0.8b       ; horizontal add
fmov   w0, s0          ; NEON -> GPR  (domain crossing)
```

The cost is dominated by the two register-domain crossings, not by `cnt`. A
caller that popcounts word by word in scalar code pays both crossings per 64
pixels.

Therefore reductions are exposed only in bulk form — over a region, a row range,
or a mask — so the implementation keeps data in vector registers and accumulates
with `cnt` + `uaddlv` without crossing back. The same interface lowers to
`popcntq` in a loop on x86 and to the SWAR sequence on Cortex-M, so one API stays
optimal on all three.

### 6.3 SIMD strategy

**NEON is the reference implementation. x86 is the portability path.** This is
the inverse of the usual habit and it follows directly from the platform tiers:
if the popcount and shift abstractions are designed against AVX-512, they will
not port to the hardware that actually matters here.

Relevant asymmetries, verified:

| | scalar popcount | vector popcount |
|---|---|---|
| aarch64 / NEON | none (via NEON, 2 domain crossings) | `cnt` + `uaddlv` |
| x86 SSE4.2+ | `popcnt`, fast | — |
| x86 AVX2 | `popcnt` | **none** — requires `pshufb` nibble-LUT |
| x86 AVX-512 | `popcnt` | `vpopcntdq` (server-class; not a target) |
| Cortex-M | none — SWAR sequence | none |

Dispatch is compile-time where possible. Runtime dispatch is added only if a
deployment target requires a single binary across ISA levels.

---

## 7. The MVP Operation Set

Derived from what a real binary-frame VIO frontend calls, not from OpenCV's
table of contents. Each entry lists its bit-parallel form, which is why this set
is tractable.

### 7.1 Denoise — median of 3

For binary input, median equals majority:

```
maj3(a, b, c) = (a & b) | (b & c) | (a & c)
```

One expression, 64 pixels per word, no branches.

### 7.2 Pyramid downsample — box 2×2

**This is where binary stops being enough, and it is measured, not assumed.**

The reference pipeline applies a 2×2 box blur and subsamples, with no
re-binarization. Starting from a binary level 0, the distinct-value count grows:

| Level | Distinct values | Bits |
|---|---|---|
| 0 | 2 — `{0, 255}` | 1 |
| 1 | 5 — `{0, 64, 128, 192, 255}` | 3 |
| 2 | 15 | 4 |
| 3 | 26 | 5 |

Two consequences:

1. **The N-bit container is required, not speculative.** A binary-only library
   cannot represent pyramid level 1. This is the concrete justification for
   `QuantMat<N>` ([§4.1](#41-bit-plane-representation)).
2. **binCV chooses the quantization, and that is a lever.** The reference lets
   precision grow into a full byte; binCV can cap levels at N bits and control
   footprint directly. Whether a capped N preserves tracking accuracy is
   [E-7](#9-open-questions-and-planned-experiments).

The 2×2 sum itself stays bit-parallel: a 4-input bit-sliced adder over the source
planes, then requantization to N bits.

### 7.3 Edge filter / threshold

Produces the 1-bit frame from a higher-precision source. In a deployed system
this may happen in-sensor; binCV provides it for pipelines that binarize on the
host.

### 7.4 Spatial derivative — binarized `[-1, 0, 1]`

**On a 1-bit input** (pyramid level 0) the derivative is **ternary**, computed by
shifts and masks rather than convolution:

```
pos = (src >> 1) & ~(src << 1)      // rising edge
neg = (src << 1) & ~(src >> 1)      // falling edge
```

Output is a sign-magnitude ternary image: `mag = pos | neg`, `sign = neg`.

**On an N-bit input** (pyramid levels ≥ 1, per [§7.2](#72-pyramid-downsample--box-22))
the derivative is a signed (N+1)-bit value, computed as a bit-sliced subtraction
of the shifted planes. Ternary is the N=1 instance of the same operation, not a
separate code path — which is what the sign-magnitude convention buys.

### 7.5 LK gradient covariance

The load-bearing operation, and the strongest evidence that a software approach
can work. Lucas-Kanade needs the 2×2 matrix `[ΣIx², ΣIxIy; ΣIxIy, ΣIy²]` over a
window. With sign-magnitude ternary derivatives, every entry is a population
count over a mask:

```
ΣIx²  = popcount(mag_x)
ΣIy²  = popcount(mag_y)
ΣIxIy = popcount(mag_x & mag_y & ~(sign_x ^ sign_y))   // agreeing signs: +1
      - popcount(mag_x & mag_y &  (sign_x ^ sign_y))   // opposing signs: -1
```

The window is large (31×31 in practice), so the reduction API must support
**masked, windowed, and preferably incremental** accumulation from the start.
This requirement is in the MVP and shapes the reduction interface — it is not a
later addition.

The identity above is exact for ternary derivatives, i.e. pyramid level 0. For
N-bit levels the same structure holds with **bit-sliced weighted sums**: each
plane pair contributes at its binary weight, so the covariance is a weighted
combination of the same masked popcounts rather than a single one. The reduction
interface is therefore specified over plane pairs, not over a single mask.

### 7.6 Corner response

Built from the same covariance machinery as §7.5.

### 7.7 Morphology

`erode`, `dilate`, `morphologyEx`. Shifted ANDs and ORs. Tier 1 semantics — must
match OpenCV bit-exactly on binary input.

### 7.8 Explicitly out of the MVP

Subpixel refinement, RANSAC, essential-matrix estimation, IMU fusion, bundle
adjustment. Not image operations. They belong to the VIO application.

### 7.9 The known hard problem: subpixel interpolation

Lucas-Kanade warps its window to subpixel positions and bilinearly interpolates.
That is inherently continuous and does not bit-parallelize. Two routes:

- **(a)** Integer-pixel tracking with binary block matching (census / Hamming +
  popcount). Fully bit-parallel, but a different algorithm whose accuracy must
  be re-validated.
- **(b)** Hybrid: bit-parallel window extraction and covariance accumulation,
  floating-point solve.

**Decision: start with (b).** It preserves the accuracy result that motivates the
project and still captures the memory win, which is the dominant claim. Route (a)
is the research upside, explored only after (b) is validated end to end.

---

## 8. Design Decisions

Recorded so that future work can tell what was chosen deliberately from what was
merely inherited.

### D-1: Template on the word *type*, not a bit count

`BinMat<uint32_t>`, not `BinMat<32>`.

Follows `boost::dynamic_bitset<Block>` (the closest analogue: runtime-sized,
bit-packed, user-chosen storage word) and `cv::Mat_<T>`. `std::bitset<N>` is not
a counterexample — its `N` is the container's extent, and the storage word is
derived internally.

Practical consequence: the bit width derives from the type, so helpers never need
to recover the type from a number.

### D-2: Bit-planes over SWAR packing

| | bit-planes | SWAR packed |
|---|---|---|
| logic | free, per plane | fine |
| **popcount reductions** | **native** | needs field extraction |
| arbitrary N | natural | awkward outside N ∈ {2,4,8} |
| 1-bit case | the base case | a special case |
| add | ~5 ops/plane, ripple | cheap |
| multiply / MAC | expensive (N² adders) | still hard |

Every MVP operation is logic and popcount; none is a MAC. Bit-planes win on
exactly the operations needed, and they make 1-bit the natural base case rather
than an oddity.

**This is why MAC-heavy quantized-NN workloads are an explicit non-goal:** that
is the workload where SWAR would win, and declaring it out of scope prevents a
future argument for changing the representation on those grounds.

### D-3: Sign-magnitude over two's complement

See [§4.2](#42-signed-values-sign-magnitude). Chosen because the LK covariance
reduces to masked popcounts directly, and because ternary becomes an instance of
the general form rather than a special case.

### D-4: Word-granularity alignment by default

Measured on the current implementation:

```
640x480 frame,    align=32 ->  46080 B vs  38400 ideal   (+20%)
94x60 pyr level3, align=32 ->   1920 B vs    705 ideal  (+172%)
94x60 pyr level3, align=4  ->    720 B                    (+2%)
```

Upper pyramid levels — which LK uses on every frame — pay up to **172% memory
overhead** for fixed 32-byte row alignment. Meanwhile the benefit of aggressive
alignment is weak on the relevant hardware: unaligned loads on ARMv8 and modern
x86 are close to free, and the property kernels actually rely on (safe whole-word
access to the trailing partial word) is already provided by word granularity.

Given principle 2 — memory wins ties — word granularity is the default and
larger alignment is opt-in per object.

**This decision is provisional and flagged for experimental validation**; see
[E-1](#9-open-questions-and-planned-experiments). No profile system is built
until data justifies one.

### D-5: Views are core, not an add-on

Four independent needs converged on one mechanism
([§4.3](#43-storage-model-and-views)). A design element that four requirements
independently demand belongs in the foundation.

### D-6: Bulk-only reductions

See [§6.2](#62-reductions-are-bulk-only). Derived from measured `aarch64`
codegen, not from preference.

### D-8: Value semantics, not reference counting

`cv::Mat` copies are shallow and reference-counted. binCV containers instead have
**value semantics: copy means deep copy.** Sharing is expressed by taking a view.

Reference counting would require atomics, which cost size and cycles on Tier 2
targets and add a thread-safety surface the library does not otherwise need. It is
also a well-known source of aliasing surprises in OpenCV code. Since views already
provide the sharing mechanism ([D-5](#d-5-views-are-core-not-an-add-on)),
refcounting would be a second way to do the same thing.

This is a deliberate divergence from OpenCV's shape, and one of the few. It is
worth it because the alternative silently changes aliasing behavior — a class of
bug that is expensive to find.

### D-9: Two view types, not a const-templated one

`BinMatView` (mutable) and `BinMatConstView` (read-only) are separate types rather
than `BinMatView<const WordType>`.

Templating on constness interacts badly with the unsigned-integral constraint on
`WordType` and produces error messages that are hard to read. Two plain types are
more verbose to declare and considerably easier to work with.

### D-10: Versioned inline namespace for configuration-dependent bodies

*(Added during T1.4, not pre-planned — see the note below.)*

Every binCV header opens `inline namespace BINCV_ABI_NAMESPACE`, whose name
encodes the exceptions and debug-check configuration the translation unit was
compiled with. Users never spell it.

**Why it is necessary.** [§5.3](#53-error-policy) makes `NDEBUG` and
`BINCV_NO_EXCEPTIONS` change the *body* of inline and template functions —
`at()`'s bounds check, every `BINCV_THROW` site's throw-versus-abort. binCV is
header-only, so those bodies are emitted into whichever objects use them. Two
objects compiled with different settings then define the same symbol differently:
an ODR violation where the linker keeps one arbitrarily. The symptom is bounds
checks that appear to vanish — silent, and very hard to attribute.

The versioned namespace makes that mismatch a link error instead. Same technique
as libstdc++'s `__cxx11`.

**Status: keep, but confirm.** This was not in T1.4's spec. It was kept rather
than reverted because T1.4 is what *created* the hazard, and reverting would leave
a real ODR trap with a silent failure mode. Recorded here so it is a deliberate
decision rather than an accident.

### D-11: Kernels alias exactly or not at all

*(Added during T2.2, not pre-planned — the first task with kernels in it was the
first task that needed the rule.)*

Every kernel in `ops/` accepts a destination that is **exactly** one of its
sources — same first word, same stride, so word *i* of the destination is word *i*
of the source — or one that shares **no word** with any source. Any other overlap
is undefined.

**Why those two and nothing between.** The T2.2 kernels are pointwise in the word
index, so an exact alias reads each word immediately before overwriting it; that is
the in-place idiom (`m &= other`) and it is free. A destination overlapping a
source at a *different* offset makes word *i* of the destination word *j* of the
source, and the row loop then reads words it has already written. Supporting it
would mean either a temporary — forbidden, [no heap in kernels](#53-error-policy)
— or a direction-aware loop whose correctness depends on the sign of the offset.

**Why it is asserted, not thrown.** It is a programming error at the call site, not
a runtime condition, and it is on the per-pixel path
([§5.3](#53-error-policy)). It also cannot be diagnosed any other way: every
address involved is valid memory, no sanitizer sees anything, and only some of the
pixels are wrong.

**"Shares no word" means per row, not per bounding box.** Two views laid over one
buffer can interleave without sharing a byte — alternate row bands (what a pyramid
downsample takes, [§7.2](#7-the-mvp-operation-set)) and left/right column tiles
both do. [D-5](#d-5-views-are-core-not-an-add-on) says a kernel takes any
`{ptr, width, height, stride}`, so a check that rejects those is wrong, and it was:
the first version compared spans and aborted every Debug build on a call that was
correct in release.

**Binds:** every kernel added under `ops/` — T2.3 shift, T2.4 morphology, and after.

### D-7: Existing code is not a constraint

The pre-existing `BinMat` implementation is a prototype. Where it conflicts with
this architecture it is replaced, not preserved. Behavior changes to tests and
call sites are expected and acceptable.

---

## 9. Open Questions and Planned Experiments

### How performance and footprint decisions get made

binCV has two co-equal goals that routinely conflict. Which one a given design
serves is usually not obvious from reasoning, so the standing method is:

> **Measure the alternatives, weigh the result against the project's goals, then
> decide — and record all three.**

This is a process requirement, not a suggestion. Concretely:

1. **Register the question** as an E-entry below, stating what decision it would
   change. A design choice with no registered question behind it should be
   uncontroversial or explicitly provisional.
2. **State the decision rule before measuring.** Write down what result favors
   which choice *first*. Deciding afterward invites fitting the conclusion to
   whatever the numbers happened to show.
3. **Measure alternatives**, on representative workloads, reporting **memory and
   speed together** — a result that reports one alone cannot be weighed against
   goals that trade off against each other.
4. **Log it** in [EXPERIMENTS.md](EXPERIMENTS.md), including the method and the
   code, so the result is reproducible and the reasoning is auditable.
5. **Promote the conclusion** to a D-record in [§8](#8-design-decisions).

**Run each experiment in the phase whose code it gates, not at the end.** An
experiment that runs after the code it was meant to decide is not a decision
procedure — it is a rationalization, and if it contradicts the code it is
expensive rather than useful.

A decision made without this loop is **provisional by definition and must say
so**. [D-4](#d-4-word-granularity-alignment-by-default) is currently the only
such decision.

### Register

Deliberately unresolved, to be settled with data rather than argument. Each
becomes a committed benchmark and an [EXPERIMENTS.md](EXPERIMENTS.md) entry.

| ID | Question | Why it matters | Decision it would change | Gates | Runs |
|---|---|---|---|---|---|
| **E-1** | Does row alignment beyond word granularity measurably help any kernel on NEON? | [D-4](#d-4-word-granularity-alignment-by-default) was decided on a memory measurement plus an untested weak-benefit assumption. | Whether a profile system is worth building at all; whether the default flips. | T1.3 and every kernel | **Phase 2** (T2.8) |
| **E-2** | Word width: is `uint64_t` the best default on aarch64, or does `uint32_t` win on cache and register pressure? | Default word type affects every kernel. | `BinMat`'s default template argument. | all kernels | **Phase 2** (T2.9) |
| **E-3** | At what window size does incremental/sliding popcount beat recomputation for the LK covariance? | The 31×31 window is recomputed per keypoint; windows overlap heavily. | Reduction API shape — whether incremental state is exposed. | T2.6, T3.6 | **Phase 2** (T2.10) |
| **E-4** | Does bit-sliced generic-N ever regress the specialized N=1 and ternary paths? | The promise is arbitrary N at no cost to the common cases. | Whether N is capped rather than arbitrary. | T1.5 specialization strategy | **Phase 3** (T3.9) |
| **E-7** | How many bits does each pyramid level actually need to preserve tracking accuracy? | Measured growth is 1/3/4/5 bits ([§7.2](#72-pyramid-downsample--box-22)), but the reference never chose that — it fell out of using `CV_8U`. Capping N is a direct footprint lever. | Pyramid level bit depths; a large share of total frontend footprint. | T3.4 (parameterized, so deferrable) | **Phase 4** (T4.1) |
| **E-6** | Route (b) hybrid LK versus route (a) binary block matching: accuracy and cost. | [§7.9](#79-the-known-hard-problem-subpixel-interpolation). | Whether the frontend stays hybrid or goes fully bit-parallel. | frontend architecture | **Phase 4** (T4.2) |
| **E-5** | Real speedup and peak-footprint numbers for a binary VIO frontend versus the byte-per-pixel equivalent. | This is the project's headline claim. | Nothing — it is the result the project exists to produce. | — | **Phase 4** (T4.3) |

The **Gates** column is why the **Runs** column is not simply "Phase 4". E-1, E-2
and E-3 constrain code written in Phases 1–2; running them afterward would mean
either rewriting that code or quietly keeping a decision the data does not
support. E-7 is deferrable only because [T3.4](TASKS.md) takes the bit depth as a
parameter rather than baking it in — **parameterizing a contested choice is what
buys the right to defer measuring it.**

---

## 10. Quality Strategy

### 10.1 What "correct" means

Tier 1 operations are correct when bit-exact against OpenCV on equivalent
content. Tier 2 operations are correct when the downstream task — VIO trajectory
accuracy — is preserved. Tier 3 operations are correct against hand-derived
reference implementations.

### 10.2 Equivalence harness

Every Tier 1 operation ships with a test asserting bit-exactness against the
equivalent OpenCV expression on the same content. Built early: it is cheap now
and it is what makes "same accuracy" a claim rather than an assertion.

### 10.3 Benchmark denominator

Performance is measured against **OpenCV performing the same semantic operation
on the same binary content stored as `CV_8U`** — because that is exactly what a
user does today without binCV. Not against OpenCV on grayscale (different
information content), and not against a strawman implementation.

### 10.4 The metric that matters

**Peak working-set footprint of the full frontend, measured end to end** — not
per-buffer ratios. A target either fits the pipeline in its memory budget or it
does not. Per-buffer ratios are supporting evidence for that headline number.

### 10.5 Defensible claims

The claim this architecture supports:

> Equivalent VIO accuracy, several-fold smaller peak memory footprint, and
> faster execution on the bit-parallel operation set.

Not "10–100× faster than OpenCV." OpenCV is well optimized; on operations that
are not bit-parallel it will win, and chasing a throughput crown would pull
development toward benchmarking operations no real pipeline calls.
