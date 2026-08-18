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

Aggressive row alignment was measured and rejected as a default — on both sides.
X-1 priced the memory cost, and X-9 (T2.8) then measured the benefit on the
reference device and found it to be **zero**: no alignment beat word granularity
on any kernel at any size, and over-aligning is 3.3–4.8× *slower* on `bitwiseAnd`
because it disables `ops/logic.hpp`'s contiguous fast path. See
[D-4](#d-4-word-granularity-alignment-by-default). Nothing is pending behind it —
[E-1 is closed](#9-open-questions-and-planned-experiments).

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

Nearly every operation in the MVP set is a composition of these — with **two
known gaps, recorded here rather than left for a later task to discover**:

- **There is no resample row, and the table needs one.** [§7.2](#72-pyramid-downsample--box-22)'s
  pyramid step is "box 2×2 sum *then subsample*", and nothing in `ops/` decimates
  horizontally. Vertically it is free — a view with twice the stride and half the
  height. Horizontally it wants output bit *j* to come from input bit 2*j*, which
  no pointwise (`ops/logic.hpp`), uniform-shift (`ops/shift.hpp`) or per-lane
  (`ops/bitslice.hpp`) kernel can express. The two known routes trade speed
  against footprint, so the choice is registered as [E-8](#9-open-questions-and-planned-experiments)
  and gates T3.4 rather than being made in passing.
- **The bit-sliced adder is single-bit and equal-weight.** `bitSlicedSum` counts
  *k* inputs each worth one, which is the 2×2 box over a **1-bit** source and
  nothing else. §7.2 measures levels 1–3 as 3, 4 and 5 bits, and a 2×2 box over an
  N-bit source adds four values worth up to 2^N − 1 each. It can be expressed by
  replicating plane *p* of each pixel 2^p times — correct, and exponential:
  *k* = 4·(2^N − 1), so 4 inputs at N = 1 but 124 at N = 5. A bit-sliced add over
  multi-bit operands is linear in N instead, and T3.4 is where its shape gets
  fixed, by the caller that needs it.

The middle two live in `ops/bitslice.hpp` (T2.7) — `maj3`, `bitSlicedSum` and
`thresholdGE`, plus the view-level `majority3` that T3.1's denoise is written in.
**A bit-sliced sum is the bit-parallel alternative to the popcount §6.2 forbids
exposing, not an exception to it.** A popcount collapses a word's 64 independent
pixels into one scalar; a bit-sliced sum of k inputs answers the same question per
*lane* and returns ceil(log2(k+1)) *planes*, so the result is still 64 pixels wide
and the next operation is still word-parallel. Nothing in it crosses to the vector
register file, and nothing in it reduces across lanes. A reduction counts pixels
across a region and returns a number; this counts inputs per pixel and returns
planes.

### 6.2 Reductions are bulk-only

**binCV must not expose a per-word popcount primitive.** This is a hard interface
rule derived from measurement.

On `aarch64` — the primary target — there is no scalar popcount instruction.
`__builtin_popcountll` on a minimal standalone function compiles to:

```asm
fmov   d0, x0          ; GPR -> NEON  (domain crossing)
cnt    v0.8b, v0.8b    ; the actual popcount
uaddlv h0, v0.8b       ; horizontal add
fmov   w0, s0          ; NEON -> GPR  (domain crossing)
```

The cost is dominated by the register-domain crossings, not by `cnt`. A caller
that popcounts word by word in scalar code pays them per 64 pixels, and cannot
amortize them, because the crossings are in the caller's data flow rather than in
the popcount helper's.

**Therefore reductions are exposed only in bulk form** — over a region, a row
range, or a mask — *so that* the implementation **can** keep data in vector
registers and accumulate there, crossing back once per row or per region instead
of once per word. The same interface lowers to `popcntq` in a loop on x86 (where
the build enables it — [X-7](EXPERIMENTS.md)) and to the SWAR sequence on
Cortex-M, so one API stays right on all three.

**The interface decision is settled; the implementation is not there yet, and the
two must not be read as one sentence.** What `ops/reduce.hpp` ships today is the
scalar per-word form T2.5 specifies, and on the reference device its interior loop
crosses back on *every* word — so it measures at **0.99×** of the per-word loop
this section forbids exposing, i.e. the bulk API currently buys nothing. A vector
accumulator at the identical load width is **1.8×** faster and is Phase 5's to
land. Measured, three runs, in [X-7b](EXPERIMENTS.md) and
`bincv-cpp/results/reduce_target_benchmark.log`. The crossing counts also differ
per entry point rather than being "two per word" everywhere — `countNonZero` 1,
`countAnd` 2, `countAndSplit` 4 — and `ops/reduce.hpp`'s own header carries the
sequences its kernels actually emit.

That gap is an argument *for* the rule, not against it: because the API is bulk,
the fix is a change to one file and no caller is revisited.

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

Through the shipped T2.5/T2.6 primitives, over one window, that is exactly:

```cpp
BinMat<W> signXor(width, height);                        // once per level
bitwiseXor(dx.constSign(), dy.constSign(), signXor.view());

const size_t    sumXX = countNonZero(dx.constMagnitude(0), window);
const size_t    sumYY = countNonZero(dy.constMagnitude(0), window);
const SplitCount s    = countAndSplit(dx.constMagnitude(0), dy.constMagnitude(0),
                                      signXor.constView(), window);
const long long sumXY = s.crossTerm();     // whenClear - whenSet, SIGNED
```

Three details of that snippet are load-bearing and each has cost someone an hour:
`constMagnitude` / `constSign`, not `magnitude` / `sign` — the kernels take
`BinMatConstView` and deduction does not consider the conversion
([D-9](#d-9-two-view-types-not-a-const-templated-one)); `crossTerm()`, not
`whenClear - whenSet` — the fields are `size_t` and the difference is signed;
and `signXor` is a **frame-sized** plane, because all three views are indexed by
the same window in the image's coordinate frame.

That snippet is the **current** shape, and [T2.11](TASKS.md) replaces it: the
three calls become one fused covariance call, and the four-argument
`countAndSplit` removes the `signXor` plane entirely — so the third detail above
stops being load-bearing once T2.11 lands.

**Masked and windowed accumulation is in the MVP and shapes the reduction
interface — it is not a later addition.** That much T2.6 built.

**Incremental/sliding accumulation was not, and now is — measured, not assumed.**
The window is large (31×31) and consecutive windows overlap heavily, which is the
regime where a sliding accumulator could win. Whether it did was
[E-3](#9-open-questions-and-planned-experiments) (T2.10), and the answer on the
reference device is that overlap pays: at 31×31 the adopted vertically-sliding
form is **7.3×** on an 8×8 search sweep and **20×** on a dense scan
([X-11](EXPERIMENTS.md)). Its 1.32× on *isolated* keypoints is **not** an
incremental result — the sliding path never executes there, and that number is the
separate accumulator-split finding recorded at the end of
[D-15](#d-15-window-reductions-get-incremental-state-and-a-fused-covariance).

`ops/reduce.hpp` therefore gains a vertically-sliding accumulator, and T3.6 is
written against it rather than against the recompute-only shape — see
[D-15](#d-15-window-reductions-get-incremental-state-and-a-fused-covariance)
and [T2.11](TASKS.md). Until T2.11 lands, `ops/reduce.hpp` still recomputes per
window; the interface was deliberately not changed in the same commit as the
measurement. See [D-13](#d-13-a-reduction-counts-pixels-never-padding) for the
neighbouring reduction decision T2.6 did settle.

The second interface question composing the snippet above exposed is settled the
same way: those three calls make **three traversals** of the same window, issuing
the same popcounts a single fused pass would. X-8 measured **1.30×** for that, and
X-11 reproduced it at **1.27×** (`uint32_t`) and **1.29×** (`uint64_t`) across
three window sizes — past T2.10's 15% threshold every time
(`bincv-cpp/results/window_benchmark.log`). **The covariance gets its own entry
point** (D-15, T2.11), returning all four numbers from one pass, and T3.6 is
written against it.

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

**The benefit side is now measured too, and it is zero** (X-9 / T2.8, reference
device, two sets of three runs — the second on a harness whose validity controls
were corrected first). Across four alignments × two kernels × both sizes, the best
result anywhere was 1.015×, inside both its 8.6% batch spread and its 1.6%
run-to-run scatter; and `countNonZero`, which walks rows unconditionally and so
isolates alignment on its own, was flat to within 0.5% at 640×480 across all four.
Two alignments were much *worse*: over-aligning disables `ops/logic.hpp`'s
contiguous fast path, so `bitwiseAnd` at 640×480 runs **3.3× slower at align 32
and 4.8× slower at align 64**, for 20% and 60% more memory.

**This decision is therefore confirmed and no longer provisional**, and
[E-1](#9-open-questions-and-planned-experiments) is closed. **No profile system is
built.** Larger alignment stays opt-in per object, documented as costing memory to
buy nothing measurable. One limit on what was tested: `BinMat` allocates with
`new[]`, so `rowAlignment` aligns the row *stride* and not the base pointer — the
result says that making rows mutually congruent buys nothing, not that absolute
64-byte row addresses would. No binCV API can request those and nothing in the MVP
wants them.

### D-5: Views are core, not an add-on

Four independent needs converged on one mechanism
([§4.3](#43-storage-model-and-views)). A design element that four requirements
independently demand belongs in the foundation.

### D-6: Bulk-only reductions

See [§6.2](#62-reductions-are-bulk-only). Derived from measured `aarch64`
codegen, not from preference.

**This binds the interface, not the current implementation, and the difference is
measured rather than assumed.** No per-word popcount is public; the shipped
reduction is nevertheless still one `__builtin_popcountll` per word and performs
accordingly on the primary target (0.99× of the loop the rule forbids exposing,
against 1.8× available from a vector accumulator — [X-7b](EXPERIMENTS.md)). D-6 is
what makes closing that gap a change to `ops/reduce.hpp` alone.

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

**Narrowed by T2.3: the in-place half applies only to kernels that are pointwise
in the word index.** `ops/shift.hpp` accepts the "shares no word" case and
**refuses** the exact-alias one. The reason is the mechanism this record already
names: an exact alias is safe *because* word *i* of the destination is read from
word *i* of the source and from nothing else, and a shift reads words
*i* ± `wordShift` instead. A direction-aware loop would rescue the purely
horizontal case (ascending for `shiftLeft`, descending for `shiftRight` — the
`memmove` argument), and it does **not** rescue the vertical one: with a row
offset and a non-constant border, the source row for destination row *y* is not
monotonic in *y*. A 10-row image shifted up by 5 under `BORDER_REFLECT_101` has
destination row 9 reading source row 4 — a row an ascending loop overwrote four
iterations earlier, and one a descending loop has not written yet but will need
again. No row order works for every `(dy, BorderType)`, and the temporary that
would make one unnecessary is forbidden.

So the rule is per kernel family and stated in each kernel's docstring, not
inferred from this record. `impl::kernel_util.hpp` carries the two predicates
separately (`viewsShareNoWord`, `destinationAliasIsSafe`) so that a kernel picks
the one that matches its access pattern rather than the one that happens to be
there.

### D-12: A shift carries a border, and the fill is the caller's

*(Added during T2.3/T2.4, not pre-planned — the task that built the shift was the
first that had to choose.)*

Every entry point in `ops/shift.hpp` takes `(BorderType borderType, bool
borderValue)`, defaulting to `BORDER_CONSTANT` / `false`. The alternative — a
shift that always zero-fills, leaving borders to whatever calls it — was rejected
because the two callers the MVP has want **opposite** fills:

| | form | a pixel outside the image must… | so it must read |
|---|---|---|---|
| `dilate` ([§7.7](#77-morphology)) | OR of shifted copies | contribute nothing to an OR | **0** |
| `erode` | AND of shifted copies | contribute nothing to an AND | **1** |

With one fixed fill one of the two is wrong at every edge: a zero fill makes
`erode` eat a *k*-wide band off a full frame, a one fill makes `dilate` grow one.
Since the choice cannot be made once for both, it is not the kernel's to make.

**OpenCV reaches the same conclusion and encodes it the same way**, which is what
makes this the Tier 1 answer rather than merely a workable one: `cv::erode` and
`cv::dilate` default `borderValue` to `morphologyDefaultBorderValue()`, which the
implementation resolves to the depth's **maximum** for erosion and its **minimum**
for dilation. Measured, not inferred — `cv::erode` on an all-white 8×8 frame with
the default border leaves 64 of 64 pixels set, and against an explicit zero border
leaves 36. `tests/test_shift.cpp` (`Shift.MorphologyFillPremise`) pins that
premise against the real `cv::erode`/`cv::dilate`, because the paragraph above is
a claim about OpenCV and T3.3's defaults will follow it.

The four non-constant types are not optional either. T3.3 is Tier 1, so its border
behaviour must be OpenCV's for every `BorderType` a caller passes through, not
only for the morphological default. They are implemented as a per-pixel fixup over
the at most `min(|dx|, width)` affected columns at one edge, leaving the interior
word-parallel.

**Binds:** T3.3 morphology and T3.5 derivative, which are the two callers this
record exists to serve.

### D-13: A reduction counts pixels, never padding

*(Added during T2.5/T2.6, not pre-planned — the first task whose kernels read a
row's trailing partial word without writing it.)*

Every reduction in `ops/reduce.hpp` counts **only bits inside the requested region
intersected with the image**. A bit at or past `width` is never counted, whatever
it holds. `countNonZero(v)` therefore equals `cv::countNonZero` of the same
content as `CV_8U` even when `v` wraps a buffer whose padding bits are all ones.

**This is a deliberate departure from T2.5's wording**, which says "whole-word
accumulation; correctness depends on padding bits being zero". Read literally that
means accumulating each row's trailing word unmasked and trusting the invariant.
Three reasons not to:

- **A source with dirty padding is a supported construction, not a bug.**
  `BinMat`'s wrap constructor states that a wrapped buffer's padding belongs to
  its caller — sensor DMA, a sub-region of a larger frame — and
  `tests/test_logic.cpp` already sweeps sources built that way. A reduction that
  over-counts on such a view returns a wrong answer from a legal input.
- **`ops/shift.hpp` already decided the same question the same way**, and pays
  considerably more for it: every source word goes through
  `impl::extendedRowWord`. Two kernel families disagreeing about whether a
  source's padding is trustworthy would be worse than either rule alone.
- **The cost is one AND per row**, not per word. The trailing word is masked
  outside the interior loop, exactly as in `ops/logic.hpp`, so the loop Phase 5
  vectorizes carries no mask.

The invariant remains load-bearing everywhere else: a region's *interior* words
are accumulated unmasked, which is correct only because every bit of a word
strictly inside a row is a pixel.

**The same rule answers a question padding alone does not.** A view that windows a
wider image has its neighbours' live pixels sitting past its `width` — an LK
window is exactly that ([§7.5](#75-lk-gradient-covariance)) — and a reduction over
a window must be over the window. One sentence covers both cases.

**Measured, because the alternative is invisible otherwise:** with the whole-image
count changed to trust the invariant (i.e. T2.5's literal reading), 545919 of
`test_reduce`'s 546468 core checks still pass and the only case family that goes
red is `Reduce.DirtyPadding_*`. Every value sweep, every Tier 1 comparison against
`cv::countNonZero`, and the covariance identity stay green — so this decision has
exactly one test standing behind it, and that is why that test exists.

**Binds:** every reduction added under `ops/`, and T3.6's covariance, which reduces
over windows of a frame.

### D-7: Existing code is not a constraint

The pre-existing `BinMat` implementation is a prototype. Where it conflicts with
this architecture it is replaced, not preserved. Behavior changes to tests and
call sites are expected and acceptable.

### D-14: `uint32_t` is the default word type

Measured on the reference device (X-10 / T2.9, three runs), against a rule written
first: change the default only if `uint64_t` wins by >10% on bulk kernels **and**
does not increase footprint at representative widths.

```
speed   countNonZero  uint64 vs uint32:  1.94x @ 640x480,  1.56x @ 94x60
        bitwiseAnd    uint64 vs uint32:  0.96-1.07x (null; memory-bound)
footprint (bytes/plane, word granularity)
        640x480   uint32 38400   uint64 38400     0.0%
        160x120   uint32  2400   uint64  2880   +20.0%
        94x60     uint32   720   uint64   960   +33.3%
```

**The two clauses point opposite ways and the rule is a conjunction, so footprint
decides: `uint32_t` stays.** A measured 1.94× on the reduction is declined on
footprint grounds — principle 2, memory wins ties, doing exactly the work it exists
to do. The penalty appears only at the upper pyramid levels LK touches every frame,
which is why T2.9 required measuring at 94×60: measured only at 640×480, every
footprint row reads 0.0% and the decision inverts.

Narrow words are worse on both counts: `uint8_t` reduces at 0.25× of `uint32_t`,
which is the per-word popcount lowering (D-6, X-7) paid eight times as often.

**Not decided here:** the word type is a per-object template parameter (D-1), so a
pyramid could use `uint64_t` at the levels where it costs no bytes and `uint32_t`
above them. X-10 priced both sides; choosing is a new question, registered as E-9.

### D-15: window reductions get incremental state and a fused covariance

Measured on the reference device (X-11 / T2.10, three runs), against three rules
written first. All three axes moved off the simpler shape:

```
axis 1  incremental vs recompute @ 31x31:  7.3x search, 20x dense (INC-ROW, the
        form adopted); INC-COL reaches 36x on dense but is rejected below. The
        1.32x "sparse" column is NOT an incremental result -- see the end of this
        record.
axis 2  fused vs composed covariance @ 31x31:  1.27x (uint32), 1.29x (uint64)
axis 3  selector plane vs four-argument:  plane 16-18% faster per frame, and a
                                          fifth plane at every level (+25% of the
                                          derivative working set; 38400 B at
                                          640x480); four-arg 0 B, 0 B/level
```

So `ops/reduce.hpp` gains, before T3.6 is written against the current shape:

1. **Incremental state**, in the INC-ROW form — slide vertically with one scalar
   accumulator, gaining the incoming row's windowed popcount and losing the
   outgoing row's. Measured **7.3× on an 8×8 search sweep and 20× on a dense
   scan** at 31×31; it wins or ties in every access pattern and needs no caller
   scratch, so it does not drag an allocation argument into the interface. The
   popcount-free per-column accumulator is faster still on a dense sweep (36×) but
   loses 12× on isolated keypoints and needs a scratch array, so it is a second
   shape rather than the one to expose first.
2. **A covariance-shaped entry point** returning `xx`, `yy`, `whenClear`,
   `whenSet` from one `visitRowWords` pass. The composition's three calls load each
   region word three times; the popcount count is identical, so the 1.27–1.29× is
   pure redundant traversal.
3. **A four-argument `countAndSplit`** taking the two sign planes instead of a
   precomputed selector plane. Here speed and memory disagree — the plane is
   16–18% faster per frame *including* its formation cost, and costs a fifth plane
   at every pyramid level on top of the four the covariance already reads: +25% of
   the derivative working set, 38400 B at 640×480 and scaling down with the level.
   Principle 2 decides: memory wins, so the plane stops being mandatory.

Scheduled as [T2.11](TASKS.md); T3.6 is written against the extended interface.

**A separate finding from the same experiment, needing no interface change:**
`impl::countViewRegion` carries one accumulator across the whole region, so a
window traversal is a single dependency chain through the popcount latency.
Splitting it into per-row partial sums measured **1.15–1.32×** at LK window sizes
on identical popcounts — the isolated-keypoint column of axis 1 is exactly that
comparison, since the sliding path never executes there.

**Two things that follow from that, and must not be lost between here and T2.11.**
The 1.32× belongs to this finding alone; quoting it for item 1 as well counts one
measurement twice. And every ratio in item 1 was measured against the *pre-split*
recompute baseline, so landing this finding first makes that baseline up to 1.32×
faster and shrinks item 1's 7.3×/20× to roughly 5.6×/15×. Both stay far past the
15% line, so nothing about the decision changes — only the numbers to quote for
it.

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
so**. [D-4](#d-4-word-granularity-alignment-by-default) was the only such decision
in the project; X-9 closed it on the reference device and it is now confirmed.
**There is currently no provisional decision on this list.**

### Register

Deliberately unresolved, to be settled with data rather than argument. Each
becomes a committed benchmark and an [EXPERIMENTS.md](EXPERIMENTS.md) entry.

| ID | Question | Why it matters | Decision it would change | Gates | Runs |
|---|---|---|---|---|---|
| ~~**E-1**~~ **RESOLVED** | Does row alignment beyond word granularity measurably help any kernel on NEON? | [D-4](#d-4-word-granularity-alignment-by-default) was decided on a memory measurement plus an untested weak-benefit assumption. | **Answered: no — best of twelve combinations was 1.015×, inside its spread; over-aligning costs 3.3–4.8× on `bitwiseAnd`. D-4 confirmed, no profile system, D-4 no longer provisional.** [X-9](EXPERIMENTS.md) | T1.3 and every kernel | Phase 2 (T2.8) ✔ |
| ~~**E-2**~~ **RESOLVED** | Word width: is `uint64_t` the best default on aarch64, or does `uint32_t` win on cache and register pressure? | Default word type affects every kernel. | **Answered: `uint32_t` stays. `uint64_t` reduces 1.94× faster but costs +33% at 94×60, and the rule is a conjunction — memory wins.** [D-14](#d-14-uint32_t-is-the-default-word-type), [X-10](EXPERIMENTS.md) | all kernels | Phase 2 (T2.9) ✔ |
| ~~**E-3**~~ **RESOLVED** | Three questions about the same interface: (a) at what window size does incremental/sliding popcount beat recomputation for the LK covariance? (b) does a fused covariance entry point beat composing it from three T2.6 calls? (c) frame-sized selector plane versus a four-argument `countAndSplit` — memory against speed. | The 31×31 window is recomputed per keypoint; windows overlap heavily; and §7.5's covariance needs four numbers that today cost three traversals. | **Answered, and all three moved off the simpler shape: (a) 7.3×–20× for the adopted sliding form → expose incremental state; (b) 1.27–1.29× → add a covariance entry point; (c) plane 16–18% faster but a fifth plane at every level → four-argument form, memory wins.** [D-15](#d-15-window-reductions-get-incremental-state-and-a-fused-covariance), [X-11](EXPERIMENTS.md) | T2.6, T3.6 | Phase 2 (T2.10) ✔ |
| **E-4** | Does bit-sliced generic-N ever regress the specialized N=1 and ternary paths? | The promise is arbitrary N at no cost to the common cases. | Whether N is capped rather than arbitrary. | T1.5 specialization strategy | **Phase 3** (T3.9) |
| **E-8** | Horizontal decimation for `pyrDown` ([§6.1](#61-bit-parallel-primitives)): a per-pixel gather loop, or a log2(width) word-parallel unshuffle that needs frame-sized constant masks? | The pyramid's subsample half has no primitive, and the two routes sit on opposite sides of the project's speed/footprint tiebreak — masks measured in frames against a loop measured in ns/px. | Whether `ops/` gains a resample primitive, and whether it is word-local (word literals only) or frame-masked. | T3.4 | **Phase 3** (T3.4) |
| **E-7** | How many bits does each pyramid level actually need to preserve tracking accuracy? | Measured growth is 1/3/4/5 bits ([§7.2](#72-pyramid-downsample--box-22)), but the reference never chose that — it fell out of using `CV_8U`. Capping N is a direct footprint lever. | Pyramid level bit depths; a large share of total frontend footprint. | T3.4 (parameterized, so deferrable) | **Phase 4** (T4.1) |
| **E-6** | Route (b) hybrid LK versus route (a) binary block matching: accuracy and cost. | [§7.9](#79-the-known-hard-problem-subpixel-interpolation). | Whether the frontend stays hybrid or goes fully bit-parallel. | frontend architecture | **Phase 4** (T4.2) |
| **E-5** | Real speedup and peak-footprint numbers for a binary VIO frontend versus the byte-per-pixel equivalent. | This is the project's headline claim. | Nothing — it is the result the project exists to produce. | — | **Phase 4** (T4.3) |
| **E-9** | Should the word type vary down the pyramid — `uint64_t` where it costs no bytes (L0, L1), `uint32_t` above? | [X-10](EXPERIMENTS.md) measured both sides: `uint64_t` reduces **1.94×** faster and costs **+33%** at 94×60 but **0%** at 640×480, so the right answer may not be one type. The width is already a per-object template parameter (D-1), so this costs no new machinery — only a decision. | Whether the pyramid picks a word type per level, and whether kernels that walk several levels pay for two instantiations. | T3.4's pyramid, [D-14](#d-14-uint32_t-is-the-default-word-type) | unscheduled |

The **Gates** column is why the **Runs** column is not simply "Phase 4". E-1, E-2
and E-3 constrained code written in Phases 1–2; running them afterward would have
meant either rewriting that code or quietly keeping a decision the data does not
support. All three closed in Phase 2, on the reference device, before T3.6 — and
E-3 is the case in point: it selected the branch that **rejects** the interface
T2.6 currently ships, which is precisely the rewrite that running it late would
have made expensive. E-7 is deferrable only because [T3.4](TASKS.md) takes the bit depth as a
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
