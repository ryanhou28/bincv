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
| **A VIO system** | binCV supplies the kernels a binary-frame VIO frontend calls. **Building the VIO framework is a separate repository's job.** binCV may be swapped into an existing frontend *to test its own kernels*; that is an instrument, not a deliverable. |

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

**THAT TABLE COUNTS IMAGE PLANES AND NOTHING ELSE, AND THE FRONTEND'S LARGEST
BUFFER TURNED OUT NOT TO BE ONE.** [X-20](EXPERIMENTS.md) measured the whole
frontend — denoise, pyramid, derivative, corner, track — at **1 721 568 B**, of
which **1 228 800 B (71.4%) was `ops/corner.hpp`'s `float` response scratch**, a
buffer no row above accounts for and 2.7× the "~0.6 MiB" this table projects.
[X-23](EXPERIMENTS.md) removed it: the streaming corner response
([D-22](#d-22-the-corner-response-streams-over-a-three-row-ring-and-that-is-the-recommended-path))
keeps three rows instead of a frame, and the measured frontend is **500 464 B** —
**3.44×** smaller, and inside this table's own projection rather than over it, with
identical corners and 1.29× faster at the reference pipeline's block size. **The
lesson is in the row that was missing, not in the ones that were there**: a
byte-per-pixel scratch inside a bit-per-pixel pipeline is invisible to a table of
image planes, and it was the dominant term.

---

## 5. API Design

### 5.1 Three tiers

Adding quantization creates a category with no OpenCV counterpart, so the
compatibility promise has to be stated per tier.

**Tier 1 — identical semantics.** `bitwise_and/or/xor/not`, `erode`, `dilate`,
`morphologyEx`, `countNonZero`, `copyMakeBorder`, `threshold` (`THRESH_BINARY`,
from a `CV_8U` source). Drop-in for OpenCV users;
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
| resample | horizontal decimation by two, word-local (`ops/resample.hpp`); vertical is a stride-doubled view |
| multi-bit add | ripple-carry over plane *arrays* — the 2×2 box over an N-bit source (`ops/pyramid.hpp`, [D-18](#d-18-the-n-bit-box-is-a-multi-bit-adder-and-the-requantization-is-a-documented-rescale)) |
| sign-magnitude difference | ripple-borrow subtract, then a conditional two's-complement negate — the derivative over an N-bit source (`ops/derivative.hpp`, [D-19](#d-19-the-derivatives-border-is-reflect-101-and-its-sign-is-the-borrow)). The borrow out *is* the sign, which is why the canonical-zero rule needs no fix-up pass |

Nearly every operation in the MVP set is a composition of these — with **two
known gaps, recorded here rather than left for a later task to discover. Both are
now closed**:

- ~~**There is no resample row, and the table needs one.**~~ **CLOSED by
  [D-17](#d-17-horizontal-decimation-is-word-local) / [X-14](EXPERIMENTS.md).**
  [§7.2](#72-pyramid-downsample--box-22)'s pyramid step is "box 2×2 sum *then
  subsample*", and nothing in `ops/` decimated horizontally. Vertically it is free
  — a view with twice the stride and half the height. Horizontally it wants output
  bit *j* to come from input bit 2*j*, which no pointwise (`ops/logic.hpp`),
  uniform-shift (`ops/shift.hpp`) or per-lane (`ops/bitslice.hpp`) kernel can
  express. E-8 registered the choice as speed against footprint; measured, **there
  was no trade** — the word-local unshuffle is word-parallel and costs zero
  auxiliary bytes, and it beat both alternatives by 8.3×–26.4× on the reference
  device. The table has its resample row and it is word-local.
- ~~**The bit-sliced adder is single-bit and equal-weight.**~~ **CLOSED by
  [D-18](#d-18-the-n-bit-box-is-a-multi-bit-adder-and-the-requantization-is-a-documented-rescale)
  / T3.4.** `bitSlicedSum` counts *k* inputs each worth one, which is the 2×2 box
  over a **1-bit** source and nothing else; over an N-bit source the only
  composition available was replicating plane *p* of each pixel 2^p times —
  correct, and exponential (*k* = 4·(2^N − 1), so 4 inputs at N = 1 but 124 at
  N = 5). `ops/pyramid.hpp` adds the multi-bit form: a tree of three ripple-carry
  additions, **3·N + 1 full-adder stages**, linear in N and equal to the
  single-bit route at N = 1. The rejected route stays under test and under
  measurement as `impl::boxSum4Replicated`, which is why the "exponential" in that
  sentence is a measured ratio ([X-15](EXPERIMENTS.md)) rather than an estimate.
  The primitives themselves — a multi-bit add, a constant multiply, a constant
  restoring division — stay in `impl::` inside the file whose caller fixed their
  shape, exactly as `ops/bitslice.hpp` said they should.

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

**Shipped as `denoiseMedian3` in `ops/denoise.hpp` (T3.1), API tier 3.** Two
details are the reference implementation's rather than this expression's, and both
are invisible on interior pixels:

- **The neighbourhood is an asymmetric L** — the pixel above, the pixel itself,
  and the pixel to its RIGHT. No left neighbour, no below neighbour.
- **The border is zero fill**, not replicate and not reflect: the first row's
  above-neighbour and the last column's right-neighbour read 0. That is not stated
  in the reference's comment; it follows from its two neighbour matrices being
  `cv::Mat::zeros` with only their interiors copied into.

The kernel is **one pass with no scratch buffer**: the above-neighbour is a row
index (a vertical shift moves no bits, T2.4) and the right-neighbour is computed
into a register, so the composed `shift`+`shift`+`majority3` spelling's two
frame-sized temporaries never exist. [X-12](EXPERIMENTS.md) priced that on the
reference device — 3.1–3.5× faster and half the memory, so the two goals agreed
and no decision was needed.

### 7.2 Pyramid downsample — box 2×2

**This is where binary stops being enough, and it is measured, not assumed.**

The reference pipeline applies a 2×2 box blur and subsamples, with no
re-binarization. Starting from a binary level 0, the value count grows — and
**there are two different numbers here, which [X-2](EXPERIMENTS.md) reported as
one**. [X-15](EXPERIMENTS.md) separated them against the reference's actual
`PyrDownInvoker` path, which is what X-2's own caveat asked for:

| Level | Values the arithmetic can REACH | Bits needed | Values a 256² frame CONTAINED (X-2) |
|---|---|---|---|
| 0 | 2 — `{0, 255}` | 1 | 2 |
| 1 | 5 — `{0, 64, 128, 192, 255}` | 3 | 5 |
| 2 | 17 | 5 | 15 |
| 3 | 65 | 7 | 26 |

**The right-hand column is a frame statistic and falls with the frame size** —
level 3 of a 256² pyramid is 32×32, i.e. 1024 pixels drawn from an alphabet of 65,
and a 640×480 frame shows 34 of them. The left-hand column does not move: an
uncapped 2×2 mean adds exactly two bits per level, because a four-input sum of
N-bit values needs N + 2. "1/3/4/5" was the sample, not the requirement.

Two details of the reference's arithmetic, both measured in
[X-15](EXPERIMENTS.md) rather than inferred, and both reproduced as checks in
`tests/test_pyramid.cpp`:

- **`cv::blur` on `CV_8U` rounds the mean UP, not to nearest.** Its 2×2 box is
  exactly `ceil((a+b+c+d)/4)`. That is where `192` in the level-1 set comes from;
  the exact mean is 191.25 and rounding to nearest gives 191.
- **Its window sits half a pixel up and to the left of the aligned block.**
  `cv::blur(src, dst, cv::Size(2, 2))` takes OpenCV's default anchor, which for an
  even kernel size is (1, 1), so the window for output (y, x) is source rows
  2y−1…2y and columns 2x−1…2x.

binCV matches neither, deliberately — see
[D-18](#d-18-the-n-bit-box-is-a-multi-bit-adder-and-the-requantization-is-a-documented-rescale).

Two consequences:

1. **The N-bit container is required, not speculative.** A binary-only library
   cannot represent pyramid level 1. This is the concrete justification for
   `QuantMat<N>` ([§4.1](#41-bit-plane-representation)).
2. **binCV chooses the quantization, and that is a lever — a smaller one than it
   looked.** The reference lets precision grow into a full byte; binCV caps levels
   at N bits and controls footprint directly. Measured at 640×480 over four levels
   ([X-15](EXPERIMENTS.md)), the whole range from "keep every bit the box
   produces" (1-3-5-7, 84 240 B) to "re-binarize every level" (1-1-1-1, 51 120 B)
   is **1.65×**, against the **4.84×–7.98×** the pyramid already wins over the
   `CV_8U` equivalent — because level 0 is 38 400 of those bytes and no cap
   touches it. Whether a capped N preserves tracking accuracy is still
   [E-7](#9-open-questions-and-planned-experiments); it is now a 1.65× question
   rather than an order-of-magnitude one, and should be run knowing that.

**Shipped as `pyrDown` in `ops/pyramid.hpp` (T3.4), API tier 2.** The 2×2 sum
stays bit-parallel and is a **multi-bit** bit-sliced adder over the source planes,
not the single-bit one — three ripple-carry additions, 3·N + 1 full-adder stages,
linear in N where the replication route is exponential
([D-18](#d-18-the-n-bit-box-is-a-multi-bit-adder-and-the-requantization-is-a-documented-rescale)).
The subsample half is a row index vertically and `impl::gatherEvenBits`
([D-17](#d-17-horizontal-decimation-is-word-local)) horizontally, fused into the
same pass, so the kernel takes **no scratch at all**.

### 7.3 Edge filter / threshold

Produces the 1-bit frame from a higher-precision source. In a deployed system
this may happen in-sensor; binCV provides it for pipelines that binarize on the
host.

**Shipped in `ops/threshold.hpp` (T3.2), and it is two operations at two tiers.**
`threshold(const cv::Mat&, dst, thresh)` is **tier 1** — bit-exact against
`cv::threshold` with `THRESH_BINARY` for every `thresh` with `|thresh| < 2^31` —
and takes OpenCV's name for that reason; what differs is the output container, not
the answer. *The domain is stated because it is real, not as a hedge:* beyond
`int`'s range `cv::threshold` reduces its `double` through `cvFloor`, whose
conversion is undefined there, and it answers the opposite of the comparison in
both directions (measured: every pixel set at `+1e300`, every pixel cleared at
`-1e300`). binCV answers the arithmetic instead and pins it with a test rather
than chasing that. `binarize(planes, dst,
thresh)` reads an N-bit `QuantMat` and is **tier 3**, since OpenCV has no N-bit
image type, so it deliberately does not borrow the name ([§5.1](#51-three-tiers)).

Both compare **strictly greater than**, which is `cv::threshold`'s semantics and
the one place an off-by-one moves a whole value class of pixels rather than a few.
`binarize` is `thresholdGE` ([§6.1](#61-bit-parallel-primitives), T2.7) at
`thresh + 1`, so the N-bit path costs one bit-sliced comparison per 8–64 pixels
and adds no arithmetic of its own.


**binCV OWNS THIS STAGE** (T5.8). `benchmark/frontend_sequence.cpp` and
`examples/vio_frontend.cpp` currently run the reference pipeline's `medianBlur` +
`|d/dx| + |d/dy|` in **OpenCV**, and the benchmark's own comment calls that stage
"deliberately NOT binCV's claim" — which contradicts this section. The contradiction is
resolved in favour of this section: **binCV ships it, and the OpenCV spelling becomes
the control it is measured against.**

**Read out of the reference rather than inferred** —
`SEAL/src/temporal_processing/edge_filter.cpp`, `rl_fast_edge_filter_wide`:

```
kernel_x = [-1  0  1]      diff_x = |filter2D(img, kernel_x)|
kernel_y = [-1  0  1]ᵀ     diff_y = |filter2D(img, kernel_y)|
mask     = (diff_x >= t) OR (diff_y >= t)
```

Three details that are easy to get wrong, and all three are the **defaults**: the
combination is **OR, not AND**; the relation is **`>=`, not `>`**; and *"wide"* is the
`[-1, 0, 1]` **central** difference — left neighbour against right neighbour, spanning
two pixels — not an adjacent `[-1, 1]`.

**All twelve combinations ship** — combine `{Or, And}` × relation `{Ge, Gt}` × spatial
`{Wide, Forward, Backward}` — as compile-time parameters, and are **tested as a
cross-product**. A twelve-way option set with one tested combination is a
one-combination op with eleven untested branches.

**Tier 3, and it must not borrow an OpenCV name.** There is no `cv::` equivalent:
`Sobel` + `threshold` is a different computation with different border handling.

**It takes `SrcT`, not just `uint8_t`** — see
[§7.8.1](#781-why-the-wide-source-path-is-bincvs-and-not-the-callers), which exists
almost entirely because of this operation. It is also **8-bit-in, 1-bit-out, and should
fuse with ingestion**: the comparison already yields a boolean per pixel, which is one
move-mask from being a bit-plane, so done properly this is the fastest way *into* binCV
rather than a step before it.

### 7.4 Spatial derivative — binarized `[-1, 0, 1]`

**On a 1-bit input** (pyramid level 0) the derivative is **ternary**, computed by
shifts and masks rather than convolution:

```
pos = (src >> 1) & ~(src << 1)      // rising edge
neg = (src << 1) & ~(src >> 1)      // falling edge
```

Output is a sign-magnitude ternary image: `mag = pos | neg`, `sign = neg`.

**On an N-bit input** (pyramid levels ≥ 1, per [§7.2](#72-pyramid-downsample--box-22))
the derivative is a signed (N+1)-bit value — N magnitude planes plus a sign,
which is exactly `SignedQuantMat<N>` — computed as a bit-sliced subtraction of
the shifted planes. Ternary is the N=1 instance of the same operation, not a
separate code path — which is what the sign-magnitude convention buys.

**Shipped as `ops/derivative.hpp` (T3.5), and two things about it are not what
the four lines above imply.**

*The taps are the RIGHT and LEFT neighbours in that order, because `cv::filter2D`
CORRELATES.* The reference is `SEAL/src/keypoint_tracking/gradients.cpp`'s
`calcBinarizedDeriv`, which is two `cv::filter2D` calls with `[-1, 0, 1]`; with
the anchor at the centre that computes `dst(x) = src(x+1) − src(x−1)`, not its
negation. Verified by experiment rather than read off the documentation, and
pinned by `Derivative.OpenCvFilter2D_Direction` — **which is the only thing that
pins it.** The tempting claim is that [§7.5](#75-lk-gradient-covariance)'s
covariance would catch an inversion, since `ΣIx²` and `ΣIy²` are popcounts of the
MAGNITUDE and `ΣIxIy` is the one entry that reads the sign planes. It would not:
reversing the taps negates *both* derivatives, and `(−Ix)(−Iy) = IxIy`, so the
whole 2×2 matrix — cross term included — is **invariant** under the mistake.
Measured on a diagonal edge through the real derivative: a 31×31 window gives
`ΣIxIy = −61` before negating both planes and `−61` after; the whole frame gives
`−124` and `−124`. `tests/test_covariance.cpp` pins that invariance, so the
sentence cannot quietly revert. What *does* negate the cross term is negating
**one** derivative and not the other, which no tap-order convention produces.

*The border is `BORDER_REFLECT_101`, not
[D-12](#d-12-a-shift-carries-a-border-and-the-fill-is-the-callers)'s
`BORDER_CONSTANT` default.* That is `cv::filter2D`'s default and therefore the
reference's, and it is also the right answer independently: reflect-101 makes both
taps read the same pixel on the outer column and row, so the derivative there is
exactly zero, where a zero fill manufactures a full-strength edge around the whole
frame for [§7.6](#76-corner-response) to detect.
[D-19](#d-19-the-derivatives-border-is-reflect-101-and-its-sign-is-the-borrow)
records both, and the scale factor binCV does not reproduce.

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

Through the shipped primitives, over one window, that is exactly **one call and
no scratch**:

```cpp
const CovarianceCount c = countCovariance(dx.constMagnitude(0), dy.constMagnitude(0),
                                          dx.constSign(), dy.constSign(), window);

const size_t    sumXX = c.xx;
const size_t    sumYY = c.yy;
const long long sumXY = c.crossTerm();     // whenClear - whenSet, SIGNED
```

Two details of that snippet are load-bearing and each has cost someone an hour:
`constMagnitude` / `constSign`, not `magnitude` / `sign` — the kernels take
`BinMatConstView` and deduction does not consider the conversion
([D-9](#d-9-two-view-types-not-a-const-templated-one)); and `crossTerm()`, not
`whenClear - whenSet` — the fields are `size_t` and the difference is signed.

**There is no third detail any more.** The snippet used to be three calls plus a
frame-sized `signXor = sign_x ^ sign_y` plane the caller had to form once per
pyramid level, and "that plane is frame-sized, not window-sized" was the third
thing to get wrong. [T2.11](TASKS.md) landed both halves of
[D-15](#d-15-window-reductions-get-incremental-state-and-a-fused-covariance) that
remove it: `countCovariance` returns all four numbers from one traversal, and its
four-argument form XORs the two sign planes inside the word loop. A caller that
already holds such a plane for other reasons still passes it —
`countCovariance(magX, magY, signXor, window)` — and is 11–14% faster for having
it ([X-11b](EXPERIMENTS.md); X-11 measured 16–18% against the pre-split code);
nothing *obliges* one to exist.

Sweeping a **column** of window positions — the corner response of §7.6, or a
search region — calls `SlidingWindowCount` for `ΣIx²` and `ΣIy²` rather than
recomputing them per position. **Only those two slide.** `SlidingWindowCount`
slides one plane's popcount, so the cross term — which needs `mag_x & mag_y` split
by `sign_x ^ sign_y` — has no incremental form here and is recomputed per
position. Making it slide would mean materializing two frame-sized planes per
pyramid level, which is *more* memory than the single selector plane
[D-15](#d-15-window-reductions-get-incremental-state-and-a-fused-covariance)'s
third item already declined; it is not a free win and is not registered as one.
The 15.9×/5.96× below are `countNonZero` sweeps, not covariance sweeps.

**Masked and windowed accumulation is in the MVP and shapes the reduction
interface — it is not a later addition.** That much T2.6 built.

**Incremental/sliding accumulation was not, and now is — measured, not assumed.**
The window is large (31×31) and consecutive windows overlap heavily, which is the
regime where a sliding accumulator could win. Whether it did was
[E-3](#9-open-questions-and-planned-experiments) (T2.10), and the answer on the
reference device is that overlap pays: at 31×31 the adopted vertically-sliding
form is **7.3×** on an 8×8 search sweep and **20×** on a dense scan
([X-11](EXPERIMENTS.md)) — **5.96× and 15.9× against the code that shipped**
([X-11b](EXPERIMENTS.md)), item 4 having landed first and made the denominator
faster, exactly as X-11 required and predicted. Its 1.32× on *isolated* keypoints
is **not** an incremental result: the sliding path never executes there. X-11 read
that column as the accumulator-split finding; X-11b then showed it is not wholly
that either, and the amendment at the end of
[D-15](#d-15-window-reductions-get-incremental-state-and-a-fused-covariance) is
what to read for what the number does and does not mean.

`ops/reduce.hpp` therefore gained a vertically-sliding accumulator —
`SlidingWindowCount`, landed by [T2.11](TASKS.md) — and T3.6 is written against it
rather than against the recompute-only shape; see
[D-15](#d-15-window-reductions-get-incremental-state-and-a-fused-covariance). The
interface was deliberately not changed in the same commit as the measurement,
which is why the two are different tasks. See
[D-13](#d-13-a-reduction-counts-pixels-never-padding) for the neighbouring
reduction decision T2.6 did settle.

The second interface question composing the snippet above exposed is settled the
same way: those three calls make **three traversals** of the same window, issuing
the same popcounts a single fused pass would. X-8 measured **1.30×** for that, and
X-11 reproduced it at **1.27×** (`uint32_t`) and **1.29×** (`uint64_t`) across
three window sizes — past T2.10's 15% threshold every time
(`bincv-cpp/results/window_benchmark.log`). **The covariance has its own entry
point** (`countCovariance`; D-15, T2.11), returning all four numbers from one
pass, and T3.6 is written against it.

**[T3.6](TASKS.md) ships that snippet as `ops/covariance.hpp`** —
`gradientCovariance(dx, dy, window)` returning `{sumXX, sumYY, sumXY}` as signed
64-bit values, one call, **0 B of scratch**, tier 3. **The identity above is now a
checked property rather than a claim**: `tests/test_covariance.cpp` compares it
against a per-pixel **float** covariance — the multiply-and-accumulate formulation
these popcounts are asserted to replace, written before the kernel and sharing no
code with it — and requires **exact** agreement, integer and float, at **383 200
window positions** across four word types, three window sizes and four
independently built frames, with origins swept from a full window outside every
edge to a full window past it. The sweep frame is **taller than the largest
window**, which is a correctness property of the suite and not a sizing choice: at
its former 11 rows every 15×15 and 31×31 position was clipped, no check anywhere
reduced a window taller than 11 image rows, and a mutant returning junk for taller
windows passed the suite unchanged. Every entry of the matrix is an integer, so a
tolerance would be a place for a real disagreement to hide rather than a rounding
allowance. The measurement side of T3.6 is not closed: its benchmark is committed
and its device number is outstanding ([X-17](EXPERIMENTS.md), `PARTIAL`), which
changes no shipped code because D-15 already decided the shape.

The identity above is exact for ternary derivatives, i.e. pyramid level 0. For
N-bit levels the same structure holds with **bit-sliced weighted sums**: each
plane pair contributes at its binary weight, so the covariance is a weighted
combination of the same masked popcounts rather than a single one. The reduction
interface is therefore specified over plane pairs, not over a single mask.

**[T3.10](TASKS.md) ships that**, and it is no longer a generalisation waiting for
a caller. [X-20](EXPERIMENTS.md) found the tracker's accuracy failure **is** the
1-bit pyramid — on windows that never clip, four 1-bit levels are still ~600×
worse than one — so an N-bit level became a precondition rather than an
optimisation, and until T3.10 binCV could not form this matrix above one bit at
all. For magnitude planes `m[0..N-1]` and sign plane `s`:

```
sumXX = Σ_i Σ_j 2^(i+j) · popcount(m_x[i] & m_x[j])          // sign squares away
sumYY = Σ_i Σ_j 2^(i+j) · popcount(m_y[i] & m_y[j])
sumXY = Σ_i Σ_j 2^(i+j) · [ popcount(m_x[i] & m_y[j] & ~(s_x^s_y))
                          - popcount(m_x[i] & m_y[j] &  (s_x^s_y)) ]
```

**It is quadratic in N where §7.4's derivative is linear**, and that is inherent to
a product of two N-bit values rather than a formulation to be optimised away —
[D-21](#d-21-generic-n-is-not-capped-and-the-n--1-specialization-is-kept-as-a-test-oracle)
closed E-4 for N = 1 only and flagged exactly this. Exploiting the symmetry of the
two diagonal entries, the cost is `3N² + N` popcounts per word — 4 at N = 1, which
is precisely `countCovariance`'s four, so **ternary is the N = 1 instance and is
required to be bit-identical to it**, checked at 61 232 window positions per word
type rather than argued. One traversal whatever N is, no heap, and no caller
scratch: the N² per-pair counters are automatic storage and the sign planes are
still XORed inside the word loop, so D-15 axis 3's no-plane property survives.
[X-22](EXPERIMENTS.md) priced it on the reference device — **3.5× / 6.5× / 12.2×
at N = 2, 3, 4 against a 1-bit level, for 1.5× / 2× / 2.5× the derivative
footprint** — and that table is what [T4.1](TASKS.md) weighs a per-level bit depth
against. X-22 takes no bit-depth decision, and it carries two caveats worth reading
before quoting it: the same kernel's absolute cost moved by up to 1.46× between
two binaries built from unchanged source, and **`3N² + N` is a model with a window
size attached** — it is inside X-22's ±25% band at W = 15 and W = 31 but
under-predicts N = 2 by ~36% at W = 7, where the per-window and per-row fixed costs
are largest relative to the word work.

### 7.6 Corner response

Built from the same covariance machinery as §7.5.

**[T3.7](TASKS.md) ships it as `ops/corner.hpp`** — `cornerMinEigenVal` (the
response map) and `goodFeaturesToTrack` (the selection), **API tier 2**: the role
and call shape of the OpenCV operations of those names, with the reference
pipeline's **binarized** derivatives rather than a Sobel, and therefore **not**
bit-exact against OpenCV. `cv::cornerMinEigenVal` works in float over a float box
filter of float Sobel outputs, so an exact comparison against it is not available
even in principle; what stands behind the operation is a per-pixel reference, two
exact *integer* properties of the eigenvalue, and a literal port of the
reference's `gftt.cpp` selection.

**The response is `(S − √D)/2` with `S = ΣIx² + ΣIy²` and
`D = (ΣIx² − ΣIy²)² + 4·ΣIxIy²`** — the smaller eigenvalue of §7.5's matrix,
spelled so that **both operands of the square root are exact integers** and the
only rounding in the operation is the root itself (taken in `double`, where IEEE
correct rounding makes it reproducible across platforms and word types; stored as
`float`). Two consequences are checked rather than assumed: the response is
**exactly 0 iff the window's matrix is singular** — which is every straight edge,
so a 45° step edge with an enormous gradient yields no corner at all — and any
non-zero response is at least `1/(2·blockSize²)`, so the selection's `> threshold`
test needs no tolerance.

**The selection's ORDER is its specification**, read out of `gftt.cpp`: the
maximum over the whole map, a `THRESH_TOZERO` cut at `qualityLevel × max`, a 3×3
non-maximum suppression over `[1, h−1) × [1, w−1)`, then a descending sort and a
greedy minimum-distance filter. NMS before the spacing filter is not
interchangeable with the reverse — NMS deletes a point beside a *higher* one
whether or not that higher one is ever accepted — and `tests/test_corner.cpp`
pins it with a case whose two orders give different survivors.

**The border decision of [D-19](#d-19-the-derivatives-border-is-reflect-101-and-its-sign-is-the-borrow)
is verified here rather than restated.** Reflect-101 was chosen partly because a
zero fill would manufacture an edge around the whole frame for this operation to
select; the suite requires **zero** corners on blank, uniform and striped frames
under reflect-101 and requires the ring to **appear** under `BORDER_CONSTANT`
(measured at 41×37: 4 spurious corners on a uniform frame, 14 on a striped one,
all in the outermost columns).

**THE OPERATION HAS TWO SHAPES AND THEY RETURN THE SAME CORNERS.**
`goodFeaturesToTrack` takes a frame-sized `float` map; **`goodFeaturesToTrackStreaming`
([T3.11](TASKS.md#t311--rolling-response-map-e-10--done)) takes a THREE-ROW RING** — 1 228 800 B against 7 680 B at
640×480 — and returns identical corners: same count, same coordinates, same order,
same `CornerResult`, proven by full-array comparison over 1.66 M corner records
including frames whose entire interior ties. It is the **recommended path**
([D-22](#d-22-the-corner-response-streams-over-a-three-row-ring-and-that-is-the-recommended-path)),
and it is **faster** — 0.77× at `blockSize` 3 on the reference device, because a ring
forces the row-major sweep X-18 already measured as the quicker traversal there.
The frame map stays for callers who need the map itself (the documented mask route,
or selecting twice over one map) and is the faster shape at large `blockSize` — above
15 at `uint32_t`, and from 15 at `uint64_t`.

**The selection's global properties survive the ring EXACTLY rather than
approximately**, which is the whole difficulty: the threshold is relative to the
frame's maximum and the spacing filter needs a frame-wide ordering. One pass
suffices because the threshold is a pure post-filter over the raw 3×3 maxima, and
because `CornerStronger` orders on response first — so the survivors are an
*up-set*, a top-K over raw maxima intersected with the threshold is the frame-map
form's ranked set, and `candidatesTruncated` reduces to one carried `float`. The
whole extra carry is **16 B**. [X-23](EXPERIMENTS.md) has the argument and the
measurement; the frontend's peak falls 1 721 568 B → 500 464 B.

**The dense sweep uses T2.11's incremental form, and [X-18](EXPERIMENTS.md)
measured that this is not a win at every window size.** Two of the three numbers
slide (`sumXX`, `sumYY`); `sumXY` has no incremental form and is recomputed. On
the reference device at 640×480 the shipped sliding sweep is **1.21× faster at
`blockSize` 31 and 1.19× SLOWER at `blockSize` 3** — the size
`SEAL/seal_params.yaml` actually runs — because the incremental state itself is a
loss below ~15 (0.94× at 3) and the column-major traversal it forces costs a
further 12% there. That contradicts the unqualified guidance in §8's D-15 and in
`ops/reduce.hpp`; the qualification now lives in both. Whether the operation
should select on `blockSize` is registered as open in X-18 and deliberately not
decided from one device.

### 7.7 Morphology

`erode`, `dilate`, `morphologyEx`. Shifted ANDs and ORs. Tier 1 semantics — must
match OpenCV bit-exactly on binary input.

Landed by [T3.3](TASKS.md) as `ops/morphology.hpp`. The shift/fold composition is
**fused into one pass over the destination**, so `erode` and `dilate` allocate
nothing and take no scratch; the five compound operations take exactly one
caller-provided frame — [D-16](#d-16-morphology-fuses-the-shift-and-the-fold-and-only-the-compound-ops-take-scratch),
priced by [X-13](EXPERIMENTS.md#x-13--t33-morphology-against-cverode--cvdilate--done).
The border fills are opposite for the two operations and that is
[D-12](#d-12-a-shift-carries-a-border-and-the-fill-is-the-callers).

| | working set of one call | scratch |
|---|---|---|
| `erode`, `dilate` | src + dst | none |
| `morphologyEx` ERODE/DILATE | src + dst | none |
| `morphologyEx` OPEN/CLOSE/GRADIENT/TOPHAT/BLACKHAT | src + dst + 1 frame | one, the caller's |

### 7.8 The input contract — where the operation set begins

**binCV accepts a single-channel, integer-typed, strided pixel array and turns it into
an N-bit `QuantMat`. Getting to that array is the caller's job.**

That one sentence is the boundary, and it settles cases in both directions without
case-by-case judgement:

| | | why |
|---|---|---|
| 8-bit grayscale | **binCV** | it *is* such an array |
| 10/12/14/16-bit in `uint16_t` | **binCV** | it *is* such an array |
| the Y plane of YUV420 (NV12/NV21) | **binCV** | it *is* such an array — the `stride` parameter already covers it |
| Bayer / demosaic | caller | produces another **wide** image |
| RGB → grey | caller | same |
| packed 10-bit MIPI (5 bytes / 4 px) | caller | not a plain array; the driver unpacks it |
| float sources | caller | not integer, and no binariser wants one |
| encoded files (PNG, JPEG) | caller, or `bincv_io` | a decoder is **8× the size of everything binCV does** — measured |

**The general form of the exclusion: an operation that is NOT ON THE PATH FROM PIXELS
TO BITS is somebody else's** — decoding, demosaicing, colour conversion. Each of those
turns one wide image into another and leaves the caller exactly as far from bits as
before.

**That is narrower than "any operation with a wide output", and deliberately so.**
[§7.1](#71-denoise--median-of-3)'s median filter runs on the **grayscale** image
immediately before binarisation and is wide-in, wide-out — it is an MVP operation
([§7.1](#71-denoise--median-of-3)) and a phrasing that excluded it would be wrong. The
test is *does this step exist to reach bits*, not *what type does it return*.

#### 7.8.1 Why the wide-source path is binCV's, and not the caller's

**Because the obvious caller-side workaround is not merely slower — it changes the
answer, in the direction that hurts most.**

"Downconvert 12→8 yourself, then call binCV" is `v >> 4`, and it **discards the four
low bits before the threshold decides**. For a plain threshold that is a rounding
difference at the boundary. **For the gradient-magnitude edge extractor
([§7.3](#73-edge-filter--threshold)) it is a total loss:** the operation is
`|I(x+1) − I(x−1)| ≥ t`, so a genuine 12-bit gradient of **15 counts becomes exactly
zero** once the operands are truncated. The edge is gone before binCV sees the pixel.

**And low contrast is exactly where a VIO frontend needs every edge it can get** —
indoors, at night, on untextured walls. The workaround fails hardest in the conditions
that matter most.

Two lesser reasons, both real: the downconversion needs a **full-frame 8-bit
intermediate**, which is the buffer binCV exists to avoid and which a
memory-constrained target may not have; and since these kernels already fold their
comparison to a single predicate, the marginal cost of a source-type template parameter
is close to nothing.

**So every op that takes 8-bit input takes `SrcT`** — the packers, the quantisation
policies, the median and the edge extractor alike.

### 7.9 Explicitly out of the MVP

Subpixel refinement, RANSAC, essential-matrix estimation, IMU fusion, bundle
adjustment. Not image operations. They belong to the VIO application.

### 7.10 The known hard problem: subpixel interpolation

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

**[T3.8](TASKS.md) shipped (b) as `ops/opticalFlow.hpp`, and the boundary landed
further over than this section predicted.** "Floating-point solve" turned out to
be the whole of it: the RESIDUAL is bit-parallel too, because bilinear
interpolation is LINEAR in its four taps, the taps of a binary frame are bits, and
the four weights are constant over the window since every pixel is displaced by
the same vector. So

    b1 = w00·S(T00) + w01·S(T01) + w10·S(T10) + w11·S(T11) − S(I),
    S(M) = popcount(magX & M) − 2·popcount(magX & M & signX)

is the residual exactly, from ten integer counts per window per iteration, with
nothing rounded before the weights are applied — where the reference rounds every
interpolated sample to 14-bit fixed point. The same collapse gives the error term,
because `|Jinterp − I| = I + (1−2I)·Jinterp` when `I` is a bit. **What is left on
the float side is O(iterations), not O(iterations × window area):** the subpixel
position, the four weights, the 2×2 solve, the update. Those are the irreducibly
continuous part, and this section was right that they exist.

Two things were traded away to get there, both deliberate and both Tier 2's
reason for existing here. A bit-plane derivative cannot be interpolated — §7.5's
identity is exact only for {−1, 0, +1} — so the PREVIOUS window is anchored at
`floor(prevPt − halfWin)` and the whole subpixel displacement rides on the next
frame; the aperture moves by up to half a pixel, the estimated flow does not. And
the window is CLIPPED rather than padded, declining the reference's
`winSize`-wide reflected border on every level, which at 640×480 with a 31×31
window is 1.24× each level's own footprint.

**And route (b) has a precondition this section did not name.** The popcount
covariance is exact only for a ternary derivative, so binCV can build only 1-bit
pyramid levels today — and [X-20](EXPERIMENTS.md) measured accuracy on the
reference pipeline's own edge-map content **degrading monotonically as 1-bit
levels are added**, from 0.0017 px RMS at one level to 3.25 px at four for a 1 px
translation, and still 0.0024 → 1.47 px on the windows that never clip at any
level. A level whose pixels are bits cannot localise a sub-pixel motion better than
its own quantisation, and that error is multiplied by 2^level on the way down.
[E-7](#register) is therefore not a footprint optimisation; it is what makes the
pyramid usable, and it needs the bit-sliced weighted-sum covariance §7.5 describes
and nothing implements.

**Two other things fail on that content, and they are not E-7's.** X-20's control
measurements separate them, and this section records them here because both are
consequences of the trades named just above. **The clipped window costs accuracy,
not only footprint**: all 141 eligible points give 3.25 px at four levels where the
58 whose window never clips give 1.47 — about half the error is the border binCV
declined, an accuracy cost that was not measured when the trade was made. And **the
tracker has a level-0 stationary point on one-pixel-wide edge maps**: on a diagonal
1 px displacement, ~28% of points return exactly zero flow with `b1 = b2 = 0`, a
degenerate configuration the `minEigThreshold` does not reject (the weakest such
window is 33× above it). Neither is fixed by a deeper alphabet, so neither may be
folded into E-7's scope.

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

**The "1-bit case | the base case" cell is now measured, not argued.** It was the
one row of the table above that made a claim about generated code rather than about
an operation count, and it stood on reasoning for two phases.
[X-21](EXPERIMENTS.md) measured it on the reference device, at the strongest
resolution available: the N-generic derivative route and the hand-written `N == 1`
specialization compile to a function of **the same size to the byte — 2264 B, `nm`
on both objects — and the same instruction count, 567**, and time to within 0.1%.
The two are not literally the same instruction stream; GCC allocates different
registers through the row loop. But nothing distinguishes them in size, in
instruction count or in time, which is what "base case rather than special case"
asserts. The corollary is uncomfortable and is recorded with it:
`impl::signedDifference`'s `if constexpr (N == 1)` branch is **measured redundant
for speed** — `-O3` already reduces the N-generic ripple to it — and is kept for a
different reason, which [D-21](#d-21-generic-n-is-not-capped-and-the-n--1-specialization-is-kept-as-a-test-oracle)
states. **What this does not say:** it is an N = 1 measurement and does not price
N = 3 or N = 5, and it compares binCV against binCV — against code with no
genericity at all the same kernel pays 8–43% in time, for reasons that are mostly
not about N ([E-12](#9-open-questions-and-planned-experiments)).

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
        1.32x "sparse" column is NOT an incremental result -- and is not wholly
        the accumulator-split finding either; read the AMENDMENT at the end of
        this record before quoting it for anything.
        [X-11b, against the SHIPPED code: 5.96x search, 15.9x dense, 1.10x sparse]
axis 2  fused vs composed covariance @ 31x31:  1.27x (uint32), 1.29x (uint64)
        [X-11b, against the SHIPPED code: 1.20x (uint32), 1.27x (uint64);
         1.20x-1.65x across the three window sizes]
axis 3  selector plane vs four-argument:  plane 16-18% faster per frame [X-11b,
                                          against the SHIPPED code: 11-14%], and a
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
   16–18% faster per frame *including* its formation cost (11–14% against the code
   that shipped, [X-11b](EXPERIMENTS.md)), and costs a fifth plane
   at every pyramid level on top of the four the covariance already reads: +25% of
   the derivative working set, 38400 B at 640×480 and scaling down with the level.
   Principle 2 decides: memory wins, so the plane stops being mandatory.

**Landed by [T2.11](TASKS.md)** as `SlidingWindowCount`, `countCovariance`, and
four-argument overloads of both `countAndSplit` and `countCovariance`; T3.6 is
written against that interface.

**A separate finding from the same experiment, needing no interface change:**
`impl::countViewRegion` carries one accumulator across the whole region, so a
window traversal is a single dependency chain through the popcount latency.
Splitting it into per-row partial sums was recorded here as **1.15–1.32×** at LK
window sizes on identical popcounts, read off the isolated-keypoint column of
axis 1 on the argument that the sliding path never executes there.

> **AMENDED after T2.11 landed it and measured it directly
> ([X-11b](EXPERIMENTS.md)): the split is worth 1.03–1.09× at the LK window
> sizes, not 1.15–1.32×, and it is a 5–6% loss on the overlapping patterns at
> W=7.** The original figure was inferred, never measured: the isolated-keypoint
> column differs from a per-window `countNonZero` in *two* ways, not one — where
> the sum lands, and that the sliding form clips its column band once at
> construction rather than per position. Timed interleaved against itself with
> only the accumulator changed, item 4 is 1.08× at SPARSE W=31 and the remaining
> 1.10× belongs to the call structure. The decision does not change — the split
> costs no memory and no interface and is a gain at both window sizes LK uses —
> but 1.03–1.09× is the number to quote.

> **AMENDED BY [X-29](EXPERIMENTS.md) (E-13): ITEM 4 IS AN `N = 1` RESULT.** Every
> number above was measured at one bit per pixel, where `BitSlicedPairCounts` is
> four counters. At `N = 4` it is **sixty-four**, and the per-row zero-and-add
> costs `~3N² + N` adds plus `4N²` words of zeroing per row against 1–2 `uint64_t`
> words of real work. On the reference device the window-wide shape is
> **1.114× / 1.348× / 1.248×** faster at `N = 2 / 3 / 4` while per-row still wins
> at `N = 1` (0.917×), so `gradientCovariance<N>` selects on `N` at compile time
> ([D-26](#d-26-the-covariance-accumulator-shape-is-chosen-on-n-and-the-noise-floor-is-measured)).
> **The item-4 decision stands for the operations it was measured on**, which are
> the `N = 1` reductions in ops/reduce.hpp; what changed is that it does not
> generalize to a structure sixteen times larger.

**Two things that follow from that, and were not lost between here and T2.11.**
The 1.32× belongs to this finding alone; quoting it for item 1 as well counts one
measurement twice. (X-11b then showed the 1.32× does not belong entirely to this
finding either — see the amendment above. It belongs to neither item on its own,
and the honest reading is that nothing should be quoted off that column.) And
every ratio in item 1 was measured against the *pre-split* recompute baseline, so
landing this finding first makes that baseline **1.03–1.09× faster**
([X-11b](EXPERIMENTS.md)) and shrinks item 1's 7.3×/20× to **5.96×/15.9×**. T2.11
landed it first for exactly that reason and re-measured afterwards;
[X-11b](EXPERIMENTS.md) carries the post-split ratios beside the pre-split ones
rather than in place of them. The branch the rule selected does not change — only
the magnitudes quoted for it.

> **AMENDED AGAIN at T3.7, the first real caller of item 1's shape
> ([X-18](EXPERIMENTS.md)): "a dense sweep wants the incremental form" is TRUE
> ONLY ABOVE A WINDOW SIZE, and below it the advice is backwards.** The corner
> response sweeps every pixel of a frame — item 1's DENSE pattern exactly — and on
> the reference device at 640×480 the incremental sweep is **1.22× faster at 31×31
> and 1.20× SLOWER at 3×3** (four-run medians; run-to-run scatter on the ratio is
> 0.18–0.34% below `blockSize` 31 and 3.3% at 31, and the ranking holds in every
> run, so the boundary is measured rather than read off one run). Two independent
> reasons, separated by a column-major recompute control: the incremental state
> alone is worth 0.93× at 3, **1.04× at 7** and 1.24× at 31 — so THAT effect crosses
> over between 3 and 7, while the NET crosses between 7 and 15 — and
> `SlidingWindowCount` slides only DOWNWARD, so a caller
> that wants it must sweep column-major, which costs a further 12% at 3, 7% at 7
> and 1% at 31 on a 32 KiB L1. The 15.9× of item 1 stands for what it measured — one plane,
> one number — and does not transfer to a caller that can slide only two of its
> three numbers at a 3-row window. The decision does not change (the incremental
> form still exists and still pays where the window is large), but the guidance in
> this record and in `ops/reduce.hpp`'s table now carries the qualification, and
> whether `ops/corner.hpp` should select on `blockSize` is registered as open in
> X-18 rather than decided from one device.

### D-16: morphology fuses the shift and the fold, and only the compound ops take scratch

*(Added during T3.3, not pre-planned — the task that built morphology was the
first whose kernel had a choice between allocating nothing and being written as
the composition it is defined as.)*

`erode` and `dilate` ([§7.7](#77-morphology)) are, by definition, an AND and an OR
over shifted copies of the source ([D-12](#d-12-a-shift-carries-a-border-and-the-fill-is-the-callers)).
Written that way, with `ops/shift.hpp` and `ops/logic.hpp`, each element cell
needs a **frame-sized temporary** between the shift and the combine — and a kernel
may not allocate one, so that temporary would have to be a caller-provided scratch
argument on `erode` and `dilate` themselves. `ops/morphology.hpp` instead
accumulates **in the destination row**, folding each cell's shifted source word
into it, so the operation is one pass and needs **no scratch at all**.

Measured on the reference device against a decision rule written first
([X-13](EXPERIMENTS.md#x-13--t33-morphology-against-cverode--cvdilate--done),
640×480, ns/pixel; `uint32_t`):

```
erode 3x3 rect      fused 0.720   composed 1.487   fused 2.07x faster, 2 frames vs 3
dilate 3x3 rect     fused 0.482   composed 1.374   fused 2.85x faster, 2 frames vs 3
erode 5x5 ellipse   fused 3.596   composed 2.881   COMPOSED 1.25x faster (1.35x at 320x240)
```

**The rule's live branch fired, and the answer is recorded rather than
reversed.** On a non-separable element the composed spelling is *faster*, by up to
1.35×. The fused kernel still ships because it is strictly smaller and the
tiebreak is that memory wins; the cost is part of this record. Its cause is not
mysterious — `ops/shift.hpp` hoists its shift amount out of the word loop while
the fused kernel's inner loop runs over element cells at a data-dependent shift
count — and closing it is Phase 5 vectorisation, not a third frame on every 3×3
erosion.

**The five compound operations do take one caller-provided frame, and one is
enough for all of them.** OPEN and CLOSE are two kernels through it; GRADIENT,
TOPHAT and BLACKHAT would each look like they need a second, and do not, because
`ops/logic.hpp` supports the exact-alias in-place case
([D-11](#d-11-kernels-alias-exactly-or-not-at-all)) and the scratch is dead by the
time the subtraction runs. `morphologyExNeedsScratch(op)` is that contract as a
predicate, so a caller sizing a buffer from a runtime `op` asks rather than
hard-codes the list.

**The structuring element is a 32-byte value, not a container.** `MORPH_RECT`,
`MORPH_CROSS` and `MORPH_ELLIPSE` are evaluated on demand from
`{shape, cols, rows, anchorX, anchorY}` exactly as `cv::getStructuringElement`
computes them, so any odd size works with no capacity limit and no allocation; a
non-owning `mask` pointer is the escape hatch for an arbitrary cell set, and it is
a view like every other argument (D-5). One consequence is load-bearing and is
asserted rather than assumed: **each parametric shape's row is a SOLID run**, so a
kernel iterating that run needs no per-cell test — and the per-cell test is the
shape query, which for the ellipse is a `sqrt`. Measured: evaluating it once per
(word, cell) rather than once per element row made a 5×5 ellipse erosion of a
640×480 frame **4.23 ns/pixel, 17× slower than `cv::erode`**. A shape query is not
the operation and must not be inside the word loop.

**The 3×3 special case is a second row kernel, and what it buys is measured.**
`morphRow3x3` runs when the element is 3×3 and centred; `morphRowGeneric` handles
everything else. A duplicated kernel is a maintenance cost forever, so
`benchmark/morphology_path_benchmark.cpp` prices it by running the same call with
the special case refused (`impl::MorphPath::Generic`, the same entry point the
correctness suite uses to require the two to agree image for image). On the
reference device the general path costs **2.1×–3.7×** across the whole pyramid
ladder, at both word widths and for rect and cross alike. `MorphPath` is a
**template** parameter rather than an argument, which is also a measurement: as an
argument it was constant-folded only while every call site in a translation unit
agreed, and adding the benchmark's one `Generic` call site made the branch live in
the shipped path and moved the headline row ~10%.

**The border is a boundary and its cost must scale with the boundary.**
`BORDER_CONSTANT` is exact in the word path; the other four map each out-of-range
column to a *different* source column, so binCV recomputes the `2 × reach` edge
columns of each row per pixel. That fixup must visit only those two bands. Written
as one loop over the row with the interior skipped by test, it cost `width`
iterations to rewrite `2 × reach` pixels and made erode **6–10× slower than
`cv::erode`** on four of the five border types — 241–260 µs against 19.5 µs for
the same call under `BORDER_CONSTANT`, at 640×480. Any kernel that repairs an edge
inherits this: index the bands, do not filter the row.

**Binds:** T3.5's derivative and anything else built from a neighbourhood over
`ops/shift.hpp`, which faces the same choice — and the same border-fixup shape.

### D-17: horizontal decimation is word-local

Measured on the reference device ([X-14](EXPERIMENTS.md) / T3.4, two runs), against
a rule written and committed first: adopt the frame-masked route only if it is
**≥ 1.5× faster** than the best word-local route with non-overlapping spreads at
both word types; decide between the two zero-byte routes on speed alone, taking
the simpler one if the difference sits inside the spread.

```
640x480 -> 320x240, median ns per destination pixel, spreads <= 1.0%
        gather loop        4.0188 (u32)   4.8823 (u64)      0 B aux
        word-local         0.2750         0.1847            0 B aux
        frame-masked       3.0938         1.5416         1408 B aux
        ratios vs word-local:  14.61x / 26.43x  and  11.25x / 8.35x
```

**`ops/resample.hpp` ships `decimateColumnsBy2(src, dst)`, the word-local
unshuffle.** Destination word *i* is the even bits of source words 2*i* and
2*i*+1, gathered by log2(WordBits) mask/shift steps in registers — word literals
only, no mask table, no scratch, no prepared plan. Vertical decimation stays what
it always was: `rowsDecimatedBy2()`, a view with twice the stride.

**What this decision actually settles is that [E-8](#9-open-questions-and-planned-experiments)
asked a leading question.** It framed the choice as buying word-parallel speed
with frame-sized constant masks, and offered the per-pixel loop as the zero-byte
alternative. The route that won is neither: word-parallel *and* zero-byte. A
register entry is a hypothesis about the shape of the answer, and this one was
wrong — which is why CLAUDE.md requires the measurement rather than the argument.

**Consequences.** T3.4's `pyrDown` carries no scratch for its subsample half and
`ops/` gains no plan-shaped API. The frame-masked route is not a tuning target:
it does the same log-depth gather with a memory pass per step instead of a
register step, and pads each row to a power-of-two bit count, so at 640 columns
and `uint64_t` it runs 10 passes over 16 words where the word-local route makes
one pass over 5 destination words (at `uint32_t`, 10 passes over 32 against one
over 10). Counted at one word type, as [X-14](EXPERIMENTS.md) now is.

**Not decided here:** X-14 measured the winner **1.49× faster at `uint64_t`** and
the gather loop **1.21× slower** there. That is [E-9](#9-open-questions-and-planned-experiments)'s
question, not this one; [D-14](#d-14-uint32_t-is-the-default-word-type) stands.

**Binds:** T3.4's pyramid, and any later operation that resamples. Both losing
arms remain in `impl::` and under test, so the experiment can be re-run against
the shipped code rather than against a description of it.

---

### D-18: the N-bit box is a multi-bit adder, and the requantization is a documented rescale

T3.4 opened with two blocking gaps. [D-17](#d-17-horizontal-decimation-is-word-local)
closed the first. This closes the second, and settles three choices `pyrDown` had
to make that no experiment could decide because none of them is a
speed-against-footprint trade.

**1. The 2×2 sum is a bit-sliced multi-bit ADD, not a bigger `bitSlicedSum`.**
Three ripple-carry additions in a tree — `(a+b) + (c+d)` — each stage a full adder
whose carry is `ops/bitslice.hpp`'s `maj3`:

```
                          NIn = 1   2    3    4    5    8
3*NIn + 1   shipped             4   7   10   13   16    25
4*(2^NIn-1) replication route   4  12   28   60  124  1020
```

Equal at NIn = 1 — which is why T2.7 could ship a single-bit adder and call it the
box — and 40× apart at NIn = 8, where the replication route also wants 1020 words
of stack per destination word. Measured on the shipped code on the reference
device at 640×480 → 320×240, the two routes are **2.08× apart at NIn = 1 and
3.46× at NIn = 4**, widening with every bit, spreads ≤ 1.1%
([X-15](EXPERIMENTS.md)). The rejected route stays in `impl::` under test and
under measurement, so the word "exponential" here is a ratio and not an estimate.

*Linear in NIn is a statement about the operation count and is exact. X-15 also
records that the shipped route's measured time grows faster than its stage count
predicts (8.0× against 3.75× from NIn = 1 to 4), most likely register pressure —
a tuning note, not a correction to this decision.*

The requantization that follows is a constant multiply (`(S << NOut) − S`, one
borrow chain, because an all-ones constant is one less than a power of two), a
constant add, and a **restoring division by a constant** — NOut steps of
`thresholdGE` plus a masked subtract. Quadratic in NOut, linear in NIn,
exponential in neither. Deliberately not a reciprocal multiply: a reciprocal
accurate enough to round identically for every input needs a constant wider than
the value, and a bit-sliced multiply costs one addition per set bit of it.

**2. The output is the mean re-expressed on the OUTPUT's full scale.**

```
dst(y,x) = round( (S / 4) * (2^NOut - 1) / (2^NIn - 1) )      [half up]
```

A `QuantMat<N>` value *v* means the intensity *v* / (2^N − 1) — 1 is white at
N = 1 exactly as 255 is white in `CV_8U`. Storing the sum on its own scale instead
would make white read as 4/7 of full scale at NOut = 3, and the error compounds
down the ladder. At NIn == NOut the multiply and the divide cancel and this is
`round(S / 4)`, the reference's case; nothing special-cases it.

**3. An odd extent REPLICATES its edge pixel** into the missing half of the block,
so the destination stays ceil(w/2) × ceil(h/2) — `cv::pyrDown`'s `dsize` and the
reference's — and the divisor stays 4 everywhere, which is what lets the
requantization be one rule with no per-column special case. Zero fill would darken
the last column and row of every level, which on a frontend whose keypoints live
near edges is a systematic bias; dropping the odd column would lose a column of
image per level.

**Three deviations from the reference pipeline, each measured and each pinned by a
test** (`tests/test_pyramid.cpp`, [X-15](EXPERIMENTS.md)):

| | reference (`SEAL`, `BOX_2x2`) | binCV | why |
|---|---|---|---|
| precision | grows into `CV_8U`, never capped | capped at `NOut` | the footprint lever [E-7](#9-open-questions-and-planned-experiments) exists to price |
| rounding | `ceil(sum/4)` — the mean rounded UP | the mean rounded to nearest, half up | rounding up brightens every level systematically |
| window phase | rows 2y−1…2y, cols 2x−1…2x (`cv::blur`'s default anchor on an even kernel) | the aligned block, rows 2y…2y+1 | the aligned block maps the destination grid onto the source by a factor of two with no offset |

Tier 2 is what buys the right to differ ([§5.1](#51-three-tiers)): `pyrDown` has
`cv::pyrDown`'s name and role and is validated against downstream task accuracy,
not against bit-exactness. The test checks OpenCV's rule as well as binCV's, so a
change in either fails there rather than being quietly absorbed.

**What this does NOT decide.** How many bits each level should keep is
[E-7](#9-open-questions-and-planned-experiments), still open and still in Phase 4.
`NOut` is a template parameter precisely so that it can be measured rather than
argued — and X-15 has now measured its footprint axis, so E-7 knows the band it is
trading against accuracy is 1.65×.

**Binds:** T3.5's derivative, which reads these levels; T3.6's tracker, which reads
all of them at once; and any later operation that requantizes.

---

### D-19: the derivative's border is reflect-101, and its sign is the borrow

*(Added during T3.5, not pre-planned — the task that built the derivative was the
first whose reference operation carried an OpenCV default binCV does not share,
and the first to produce a multi-plane SIGNED output.)*

**1. `ops/derivative.hpp` defaults to `BORDER_REFLECT_101`, breaking with
[D-12](#d-12-a-shift-carries-a-border-and-the-fill-is-the-callers)'s
`BORDER_CONSTANT`/`false`.** The border stays a parameter — D-12's argument that
the fill is the caller's is untouched — but the *default* is the reference's,
for two reasons that point the same way:

- `cv::filter2D`'s default border is `cv::BORDER_DEFAULT`, which **is**
  `BORDER_REFLECT_101` (both are 4, in OpenCV and in `core/types.hpp`). Measured
  on a 1×8 row with one set pixel at column 1: the default and an explicit
  reflect-101 both give `dx(0) = 0`, an explicit `BORDER_CONSTANT` gives `+255`.
  So `calcBinarizedDeriv` reflects, and T3.6 is written against derivatives that
  agree with it at every pixel including the edges.
- Reflect-101 makes **both taps read the same source pixel** on the first and last
  column (row), so the derivative there is exactly 0 whatever the frame holds. A
  zero fill instead reads the second column against nothing and manufactures a
  full-strength edge all the way around the frame — which
  [§7.6](#76-corner-response)'s min-eigenvalue response would then select as a
  ring of keypoints along the image border. Reflect-101 is the answer that is both
  reference-exact and correct; a compatibility tax that also happened to be wrong
  would have been worth arguing about, and this is not one.

The degenerate extent is OpenCV's too: at `width == 1` both reflect flavours map
every out-of-range coordinate to 0, so the derivative is 0 — which is what
`cv::filter2D` returns on a 1×1 image.

**2. The scale factor is NOT reproduced, and that is representational.** The
reference multiplies by 16 into `CV_16S` over `{0, 255}` content, so its values are
`{−4080, 0, +4080}`; binCV's pixels are `{0, 1}` so its derivative is
`{−1, 0, +1}`. Sign and magnitude structure are identical — a common positive
factor multiplies every entry of [§7.5](#75-lk-gradient-covariance)'s 2×2 matrix
alike, leaving eigenvector directions and the min-eigenvalue *ordering*
[§7.6](#76-corner-response) selects on unchanged. Reproducing 4080 costs 13 more
magnitude planes per pixel to carry no information. `tests/test_derivative.cpp`
divides the ported reference's output by 4080 and **requires the division to be
exact**, so "no other value ever appears" is checked rather than assumed. binCV
also keeps the two axes as two images where the reference `cv::merge`s them:
interleaved channels put every second word out of reach of a word-parallel
popcount, and nothing in binCV consumes a two-channel image.

**3. The N-bit path is linear in N, and the sign costs nothing.** `a − b` over
N-bit bit-sliced operands is one ripple-borrow subtraction (N full-subtractor
stages, the borrow being `maj3` with the minuend inverted) followed by one
conditional two's-complement negate (N half-adder stages): `2·N` adder-class
stages, against `2·(2^N − 1)` single-bit inputs for the replication route
[D-18](#d-18-the-n-bit-box-is-a-multi-bit-adder-and-the-requantization-is-a-documented-rescale)
rejected for the box sum. Measured on the reference device at 640×480, N = 1 → 5
costs **6.93×** against 5× from the stage count and 3× from the destination plane
count — and against **31×** for the replication route ([X-16](EXPERIMENTS.md)).

**Against the denominator, and the residency question answered rather than
assumed.** Both axes cost **0.201 ns/pixel at 640×480 in 192 000 B**, against
**5.007 ns/pixel in 1 536 000 B** for the two `cv::filter2D` calls
`calcBinarizedDeriv` makes: 24.9× faster in 8.0× less memory. With each side's
measured fixed per-call cost subtracted the ratio is 24.8× at 640×480 and 20.0× at
160×120, where both working sets fit in cache — so **~20× is the arithmetic and at
most a quarter is residency**, which is a different balance from
[X-6, X-12 and X-13](EXPERIMENTS.md) and is stated because those three had to say
the opposite. The **fused kernel is 2.94× faster than the composed spelling as
well as 1.40× smaller**, so unlike D-16's non-separable element there is no speed
cost accepted for footprint here. (**1.40× counts both axes** — 5 planes against
7, the two scratch frames being shared between them, which is what a caller
forming a covariance needs. One axis in isolation is 1.67×, 3 planes against 5.
`ops/derivative.hpp` states both and says which is which; a footprint claim about
this operation should use 1.40×.)

**What the N-bit path costs against the denominator is still open**, and is
registered rather than glossed: the N-bit ladder's `cv::filter2D` comparison was
described as a denominator in three places and never timed, so N ≥ 2 — every
pyramid level above 0 — has no measured ratio on the device yet. The benchmark now
times it; see the [X-16 amendment](EXPERIMENTS.md). Nothing in this decision
depends on that number: the border rule, the sign-is-the-borrow construction and
the linear-in-N claim are each settled without it.

**THE CANONICAL-ZERO RULE HOLDS BY CONSTRUCTION AND NEEDS NO FIX-UP PASS**, which
is the answer to the question T3.5 was set to ask about the container: the sign
plane **is** the subtraction's borrow-out, and a borrow out of `a − b` means
`a < b`, which forces a non-zero magnitude. No input can produce a set sign over a
zero magnitude, in either direction. `SignedQuantMat`'s docstring already permits
kernels to write the two planes independently for exactly this reason; T3.5 is the
first operation to rely on it, and it turns out not to need the permission.

**What the container did cost, recorded because it is D-3's bill arriving.** A
two's-complement destination would be the subtraction alone — N+1 stages over
sign-extended operands — so sign-magnitude costs roughly N−1 extra adder-class
stages per destination word here, approaching 2× at large N. That is the price of
[§7.5](#75-lk-gradient-covariance)'s covariance being three population counts over
masks rather than a bit-sliced multiply, and §7.5 runs 31×31 times per keypoint
where this runs once per pixel. Recorded, not measured: no experiment has priced
the two's-complement alternative end to end, because nothing in the MVP would
consume it.

**The kernel takes no scratch**, following
[D-16](#d-16-morphology-fuses-the-shift-and-the-fold-and-only-the-compound-ops-take-scratch),
which named this operation as its second caller. Written as the composition it is
defined as — two `ops/shift.hpp` calls per axis and then the mask — each axis needs
two frame-sized temporaries, and a kernel may not allocate. The horizontal taps are
computed into registers from the source row's words; the vertical ones are a row
index (T2.4). D-16's border lesson applies too and is why the edge fixup is one
bit per row rather than a test per word: the left tap is carried as a synthetic
word before word 0, and the right tap is folded into the trailing word's branch,
which the tail mask requires anyway.

**Binds:** T3.6's covariance and T3.7's corner response, both of which read these
two planes; and any later operation that produces a `SignedQuantMat`.

---

### D-20: the tracker's per-pixel work is all popcounts; only the solve is float

[§7.9](#79-the-known-hard-problem-subpixel-interpolation) chose route (b) and
described it as "bit-parallel window extraction and covariance accumulation,
floating-point solve". [T3.8](TASKS.md) shipped it with the line drawn further
over, and the placement is a decision rather than an implementation detail because
it is what the whole hybrid claim is measured against.

**On the bit-parallel side, exact integers, no per-pixel float at all:** the
window (a `Rect` through `impl::clipRegion` and `impl::visitRowWords` — no patch
is ever copied out, where the reference copies `winSize.area()*3` shorts per
invoker); the 2×2 matrix (one `gradientCovariance` call, D-15); the residual and
the error term, by the identities in §7.9; and the four bilinear tap planes, which
are word-aligned reads of two next-frame rows with one cross-word bit shift.

**On the float side, once per window per iteration and never per pixel:** the
subpixel position and its split into an integer tap offset and a fraction; the
four bilinear weights; the 2×2 solve, its determinant and the minimum eigenvalue;
and the iteration itself.

Three consequences follow and each is load-bearing.

1. **The singularity test needs no epsilon.** `det = xx·yy − xy²` is a difference
   of products of exact popcounts: it is 0 or at least 1. The reference guards
   `D < FLT_EPSILON` because its `D` is a float product of float box-filtered float
   Sobel outputs; here the test is `det <= 0`, for the same reason
   [§7.6](#76-corner-response)'s selection needs no epsilon.
2. **There is no scratch buffer of any kind** — not one byte — because nothing is
   copied out of the window. That is what makes the tracker 0.2% of the frontend's
   footprint ([X-20](EXPERIMENTS.md)).
3. **The result is word-type-invariant, and cross-ISA bit-identity is not
   PROMISED — though it is what is currently measured.** Every count is exact, so
   all four word types give identical flow. The solve is float, so a fused
   multiply-add may move the last digits, and unlike `cornerMinEigenVal` — whose
   only rounding is a correctly-rounded `sqrt` — this operation declines to promise
   otherwise. Measured, x86-64 and the aarch64 reference device print **identical**
   values on all thirteen synthetic accuracy cases; the only quantity that moves is
   the residual identity's worst rounding gap, 6.306e-14 against 6.573e-14.
   [X-20](EXPERIMENTS.md) previously reported an accuracy difference between the
   two platforms and has withdrawn it: the two figures came from two different
   builds.

**The units are part of the decision.** binCV's `Ix` is the raw `[-1, 0, 1]` tap
over {0, 1} pixels, which is twice the central difference; the reference reaches
the same gradient by a different route (pixels in {0, 255}, derivative ×16,
intensity descaled by `W_BITS1-5`). Substituting `g = Ix/2` scales `A` by 1/4 and
`b` by 1/2, so the step from raw taps is multiplied by exactly 2. Dropping that
factor does not diverge — it halves every step and stops early on the epsilon
test, which looks like "slightly worse accuracy" rather than like a bug.

---

### D-21: generic-`N` is not capped, and the `N == 1` specialization is kept as a test oracle

`QuantMat<N>` keeps arbitrary `N` in [1, 8]. **No cap.**

The promise D-2 rests on is that bit-planes make the 1-bit case the *base* case
rather than a special case — so the interesting risk was never that N-bit costs
something, but that carrying `N` costs the 1-bit and ternary paths the frontend
actually runs. [E-4](#9-open-questions-and-planned-experiments) registered that
question and [X-21](EXPERIMENTS.md) measured it on the reference device against
three arms, the third of which — a hand-written binary-only control that includes
no binCV header — is what makes the comparison mean anything.

**Measured:** at `N = 1` the generic route and the specialization produce a
derivative of **the same size to the byte (2264 B) and the same instruction count
(567)**, time to within 0.1%, and generic-N's whole object is **90 B smaller**. They
are *not* the same instruction stream — GCC allocates different registers — and the
equality is the derivative's; the covariance and count functions differ by 40 B and
24 B and time inside the batch spread. Nothing distinguishes the two routes in
size, instruction count or time. **Carrying `N` is free at `N = 1`, so capping `N`
would buy nothing.**

**`impl::signedDifference`'s `if constexpr (N == 1)` branch is kept, and not for
speed.** It is measured redundant — `-O3` already reduces the N-generic ripple to
the ternary spelling — and deleting it would save 90 B. What it buys is
`Derivative.RoutesAgree_*`, which compares two independently written formulations
of one operation image for image; delete the branch and that test compares a route
against itself. **For a bit-parallel kernel whose failure mode is a silently wrong
bit, an independent second formulation is a cheap oracle**, and the same device is
used elsewhere in the project (`ops/pyramid.hpp`'s rejected box sum). The reason is
recorded here so a later reader does not remove it as dead weight — it is dead
weight, for the purpose it was written for, and live for another.

**What this decision does NOT cover, stated because the register row is what gets
cited:**

- **It is an `N = 1` result.** Every arm was `QuantMat<1>` / `TernaryMat`. It does
  not price `N = 3` or `N = 5`, and the shapes differ in kind: the derivative's
  ripple-borrow work is linear in `N`, while [§7.5](#75-lk-gradient-covariance)'s
  bit-sliced covariance contributes plane **pairs** and is quadratic. "Generic-N is
  free" means "free against the specialization at `N = 1`".
- **It compares binCV against binCV.** Against the hand-written control both binCV
  routes pay **8–43% in time** and **2.63× in code size** (`-fno-exceptions`, the
  configuration the Tier 2 claim rests on; 2.84× with exceptions on). X-21's
  decomposition puts most of that in genericity that is **not** in `N` — runtime
  `BorderType`, the word type, the argument contract — which is
  [E-12](#9-open-questions-and-planned-experiments) and is **not settled by this
  record**. Scalarizing `N`'s per-plane arrays was measured and recovers about a
  fifth of the per-row cost: worth having, and not the fix.

**Supersedes nothing.** It confirms the specialization strategy chosen at T1.5 and
amends [D-2](#d-2-bit-planes-over-swar-packing), whose "1-bit case | the base case"
cell was an argument until X-21.

---

### D-22: the corner response streams over a three-row ring, and that is the recommended path

`ops/corner.hpp` exposes **two shapes of one operation**, returning **identical**
corners: `goodFeaturesToTrack` over a caller-provided frame-sized `float` map, and
**`goodFeaturesToTrackStreaming` over a `kResponseRingRows`-row ring**. The
streaming form is the **recommended path**; the frame-map form stays because T3.7
made the map caller-provided and some callers want the map.

**The problem it solves is that the largest buffer in a bit-per-pixel frontend was
a byte-per-pixel scratch.** [X-20](EXPERIMENTS.md) measured 1 228 800 B of `float`
response map inside a 1 721 568 B frontend — **71.4%, more than everything else
combined**, where every image plane is one or two BITS and the tracker itself is
0.2%. [E-10](#9-open-questions-and-planned-experiments) registered the question and
[X-23](EXPERIMENTS.md) settled it under a rule committed before the streaming form
existed.

**Measured on the reference device, 640×480, `uint32_t`, `blockSize` 3 — the
reference pipeline's own value — medians of 11 interleaved batches, within-run
spreads 0.15–0.27%, arms in separate translation units, order swapped and re-run,
and the whole benchmark run twice:**

| | whole detector | response stage | corner stage peak | frontend peak |
|---|---|---|---|---|
| frame map | 132.8 ns/px (40.79 ms/frame) | 107.1 ns/px | 1 333 848 B | 1 721 568 B |
| **streaming** | **102.8 ns/px (31.59 ms/frame)** | **77.1 ns/px** | **112 744 B** | **500 464 B** |
| | **0.774×** | 0.720× | **11.83×** | **3.44×** |

Both footprint totals are **read off a live-byte high-water mark** in
`Flow.FrontendFootprint_640x480`, each shape in its own scope with the other's
buffers destroyed, and the per-stage rows are required to account for the reading to
the byte — so a heap buffer that no row names, anywhere in the frontend, fails the
case. (Corrected at triage: the first version summed the buffers its author had
enumerated, and printed the streaming total while the 1 228 800 B frame map was still
live.)

The other device run reads `T` = **0.764×**: **run-to-run scatter is ~1.3% on the
ratio and up to ~3.4% on an individual ns/pixel column, an order of magnitude above
the within-run spread**, so the ratio is the quotable quantity and **both logs are
committed**. The verdict is the same in both runs at every block size, both word
types and both frame sizes.

**IT IS FASTER, WHICH IS NOT WHAT THIS DECISION WAS SCHEDULED TO WEIGH.** T3.11,
this document's own E-10 row, X-20's decision 3 and TASKS.md T3.8's X-20 write-up
(the fourth site, which the rule's list of three did not name and triage found) all
estimated "roughly 2× the response compute"; all four are corrected by X-23 by name. A ring FORCES a
row-major sweep, and X-18 had already measured the shipped column-major sliding
sweep 1.19× *slower* than row-major recomputation at `blockSize` 3 — so the
streaming form collects a traversal discount before paying for anything, and it
never pays for a second pass.

**Equality is a contract, not a resemblance, and it is what makes this a decision
about bytes rather than about behaviour.** Same count, coordinates, order and
`CornerResult` — 1 664 932 corner records compared across four word types, six block
sizes, five frame sizes, capacities that truncate, and frames whose entire interior
ties. Three deliberate mutants of the streaming form were caught by those cases.

**How the GLOBAL properties survive three rows, since neither is local:**

1. the threshold is a **pure post-filter** over the raw 3×3 maxima — the shipped
   selection already fuses threshold into NMS on exactly that argument;
2. `CornerStronger` orders on **response first**, so the survivors are an **up-set**
   of the raw maxima, and `topK(rawMaxima) ∩ survivors` **is** the frame-map form's
   ranked set. The top-K buffer is the caller's own candidate array — **zero extra
   bytes for the global sort**;
3. `candidatesTruncated` means `|survivors| > capacity`, which is equivalent to
   "the strongest DISCARDED candidate is above the threshold" — **one carried
   `float`**;
4. flat plateaus are pruned against the **running** maximum, which is monotone and
   therefore never exceeds the final threshold, so the prune removes only
   non-survivors and cannot move the answer.

Total carry: **16 B**. The two-pass shape the estimate described was built and
priced as X-23's arm S2 — 1.33× for **the same peak** (the same ring, the same
candidate array, a scalar or two apart) — and is **not shipped**: with the peaks
indistinguishable, X-23's arm-tie rule falls to its second clause and S2 is 1.71×
S1.

**Where it costs, and it is not the same place in the two word types.** At
`uint32_t`, `T` = 0.77× at `blockSize` 3, 0.92× at 7, 1.00× at 15, **1.08× at 31** —
crossover between 15 and 31. At `uint64_t`, 0.77× / 0.93× / **1.03×** / 1.13× —
crossover between **7 and 15**, which is exactly where X-18 put it, so this decision
moves X-18's boundary at the narrower word and reproduces it at the wider one. Both
device runs agree on both. In every case the crossover is above the MVP's own block
size (3). A caller running a large window, or wanting the map, keeps the frame-map
form — and the header carries both peaks and both times so that choice is made with
numbers.

**What this does NOT decide.** The corner stage's dominant term is now the
**candidate array** (105 048 B at 640×480, a per-frame reading whose structural
maximum is 3 659 568 B). That is a contract question — what a caller provisions and
what truncation costs them — and X-23 deliberately leaves it open.
[E-11](#9-open-questions-and-planned-experiments) is likewise untouched.

**Supersedes nothing; amends
[§4.6](#46-memory-arithmetic)**, whose "~0.6 MiB" projection the frame-map frontend
exceeded 2.7× and the streaming one meets.

---


### D-23: the tracker clips its window and does not pad its levels — measured, not argued

`ops/opticalFlow.hpp` intersects each keypoint's window with the level (deviation
(ii)) where `buildOpticalFlowPyramid` allocates every level with a `winSize`-wide
reflected border. **That was an argued footprint decision and is now a measured
one, and the measurement favours it on ACCURACY as well as bytes.**

[X-25](EXPERIMENTS.md) built the padded arm and compared it against clipping on the
frontend's actual product — **yield**, the fraction of eligible keypoints tracked
within X-20's 1.0 px. **Padding is worse than or equal to clipping in five of seven
cases** and better by at most 1.4 points in the other two, for **1.38× the bytes**
under the reference's own per-level scheme (589 968 B against 427 680 B at 752×480,
ladder `1/2/2/2`).

**The reason the border looked necessary was a statistic, not a phenomenon.** Every
prior reading of this operation — X-20's, X-24's, and this document's own E-14 row —
used RMS over all points. On `(1, 0)` the shipped tracker has `rms(all)` **0.8356 px**
and yield **98.6% at `rms(usable)` 0.0009 px**: 139 of 141 keypoints tracked to a
thousandth of a pixel, two catastrophically wrong. *(Re-measured on the corrected
two-stage preprocessing 2026-08-21: 98.0% at 0.0013 px over 102 eligible keypoints —
the same statement, and the padded arm now ties or loses in every cell rather than
winning two. See [X-26](EXPERIMENTS.md)'s correction.)* RMS over everything reports a
small catastrophic tail as though it were the body of the distribution. Restricting
to never-clipping points moved the RMS because it excluded the outliers, not because
clipping caused them — **clipping costs about two keypoints out of 141, not the 59%
X-24 attributed to it.**

**`LKEntryLevel` ships with two policies and `Coarsest` is the default.**
`DeepestFitting` — a keypoint enters at the coarsest level whose window fully
contains it, costing no memory and discarding no keypoints — is the best arm X-25
measured on small motion (**100% yield at 0.0010 px on `(1, 0)`, the only perfect
cell in the table**) and the worst on large motion (77.9% at `(6, 4)` against 99.3%).
That is not a defect: a point denied its coarse levels cannot capture a large
displacement, which is what a pyramid is for. It is offered to callers whose motion
is known to be small, and it is not the default because the frontend's is not.

**What this does NOT settle** is the 0.25–0.32 px `rms(usable)` floor that every
arm shares, padded included; X-20's own single-level figure with no pyramid at all
is 0.2860 px. That is [E-16](#register) and it is a question about the
representation, not about the pyramid.


### D-24: both tracking routes ship; route (b) is the default and route (a) is the memory-constrained one

[§7.9](#79-known-hard-problems) named two routes for tracking on binary frames and
expected one of them to win. [X-26](EXPERIMENTS.md) built route (a) and measured
both, and **the split resolves to both routes existing**, which is not what either
arm anticipated.

| | route (b) hybrid LK | route (a) Hamming block matching |
|---|---|---|
| yield (real frame) | **75.9–99.3%** | 56.7–75.0% |
| `rms(usable)`, sub-pixel cases | 0.2502–0.3208 px | **0.2347–0.2594 px** |
| pyramid-stage bytes | 306 720 | **102 240 (3.00× smaller)** |
| build | 285.0 µs | **196.9 µs (1.45× cheaper)** |
| track | 14 024.5 µs | **13 006.3 µs (0.93×)** |
| usable points per KB | 0.32 | **1.04 (3.2×)** |
| usable points per ms | **8.2** | 8.0 |

**Route (a) needs no derivative at all**, which is where the 3.00× comes from: route
(b) carries two frames *and* two `SignedQuantMat` ladders, route (a) carries two
frames. That is a footprint result, not an implementation detail.

**Route (b) is the default because yield is what a frontend produces.** Route (a)
loses **2–12 points** of it on the *same* ladder, so it is an algorithm difference and
not a representation one.

> **CORRECTED 2026-08-21.** The table above and an earlier "15–25 points" figure
> were measured with the reference **denoise stage missing** from the harness —
> `SEALProcessor::temporal_process` runs `median_filter` then
> `rl_fast_edge_filter_wide` and only the second was implemented. Re-measured on
> the correct content, **route (a)'s yield rises from 56.7–75.0% to 68.6–88.1%**
> and it wins one cell outright. The correction is **not symmetric**: a median
> filter removes isolated noise, and isolated noise is what manufactures false
> minima on a Hamming surface, which route (a) takes at face value where route
> (b) averages it over a 31×31 window. The decision is unchanged — route (b)
> leads on yield, route (a) keeps its 3.00× footprint advantage — but the trade
> is materially more favourable to route (a) than first recorded. See
> [X-26](EXPERIMENTS.md).

**Route (a) is not closed, and shipping it is not hedging.** It is **3.2× more
keypoint-efficient per byte at equal keypoint-efficiency per millisecond**, and on
the corrected content it is within 2–12 yield points of route (b) rather than 15–25.
[CLAUDE.md](../CLAUDE.md)'s "memory wins" tiebreak covers *speed against footprint*;
this is *accuracy against footprint*, and §1 puts that on the integrating pipeline's
side of the boundary — a VIO frontend that RANSACs its correspondences may
rationally prefer more, cheaper, noisier points. This repo supplies both and states
the trade rather than making the caller's choice for them.

**`searchRadius` is route (a)'s whole cost story, and enlarging it fails twice
over:** R = 4 costs 2.90× LK against R = 2's 0.93× *and* is less accurate in
aggregate, because a wider search finds more false minima. There is no radius at
which route (a) wins by searching harder.

**This refines [D-20](#d-20-the-trackers-per-pixel-work-is-all-popcounts-only-the-solve-is-float)
rather than contradicting it.** A parabolic fit to a Hamming surface — integer
arithmetic, four extra window scores — is **more precise than LK's Gauss-Newton
solve on the points both find** (0.2347–0.2594 against 0.2502–0.3208 px). So the
continuous formulation does not buy precision; it buys **robustness**. §7.9's
"LK's accuracy comes from its continuous formulation" was too coarse, and the
correction is that continuity is load-bearing for *which* points match, not for how
well the matched ones are located.


### D-25: the 1-bit representation is not the accuracy limit — the tracker is

[T3.8](TASKS.md)'s 0.25 px RMS criterion has been a documented MISS since
[X-20](EXPERIMENTS.md), and three experiments eliminated three pyramid parameters
without explaining it: [X-24](EXPERIMENTS.md) (level bit depth),
[X-25](EXPERIMENTS.md) (the border and the entry policy), and X-25 again showing
`rms(usable)` stuck at 0.25–0.29 px **in every arm including the padded one**. The
natural next suspicion was the representation itself.

**[X-27](EXPERIMENTS.md) measured it, and the representation is nowhere near the
limit.** A 31×31 window of a 1-bit reference edge map resolves **29.3 distinct
binary states per pixel of displacement**, giving a floor of **0.025 px** noise-free
and **0.10 px** at one gray level of sensor noise. Even at σ = 4 — a poor sensor —
the floor is 0.174 px, still inside the tolerance.

**So the 0.25 px criterion was always achievable, and it stands unchanged.** No
tolerance in this project has been widened at any point, and this is the entry that
could most easily have widened one.

**X-20's derivation was wrong in its suspect half and conservative by ~7×.** It
assumed "an effective count of four independent crossings" in a 31×31 window; the
real count of distinguishable states is 29.3. Being wrong in the safe direction is
why nothing downstream of it broke.

**A bigger window buys almost no localisation, and that is a design consequence.**
From 11×11 to 41×41 the set pixels grow 7.3× while distinct states grow only 1.8×:
the set pixels lie on **connected contours**, and an edge constrains motion only
perpendicular to itself, so they are not independent samples and averaging does not
go as `1/√area`. Window sizing cannot be justified by "more pixels average better".

**What remains is [E-17](#register): a factor of 2.5–3 that belongs to the
tracker.** The prime suspect is named rather than left open — **deviation (i)**,
the previous window anchored on the integer grid, which `ops/opticalFlow.hpp`
already calls "the concrete thing route (b) trades away" and which displaces the
aperture by up to half a pixel. That is the right order of magnitude for the gap,
and it is a hypothesis for E-17 to test rather than a conclusion.


### D-26: the covariance accumulator shape is chosen on N, and the noise floor is measured

[D-15](#8-design-decisions) item 4 gives window reductions a **per-row partial
accumulator**, to break the serialized dependency chain through popcount latency.
That was measured — [X-11b](EXPERIMENTS.md), 1.08× at W = 31 — **at `N = 1`, where
`BitSlicedPairCounts` is four counters.** At `N = 4` it is **sixty-four**, and the
per-row zero-and-add costs `~3N² + N` adds plus `4N²` words of zeroing **per row**
against 1–2 `uint64_t` words of real work per row.

**[X-29](EXPERIMENTS.md) measured both shapes on the reference device:**

| N | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| window-wide vs per-row | **0.917×** | **1.114×** | **1.348×** | **1.248×** |

So the per-row shape **pays at `N = 1` and costs above it**, and the crossover sits
between 1 and 2 rather than somewhere in the middle. `gradientCovariance<N>` now
selects with `if constexpr`, which is free because `N` is already a template
parameter, and **D-15 item 4 is amended to be an `N = 1` statement**.

Both forms add the same integers in a different order, and `size_t` addition is
associative, so **results are bit-identical by construction** — this changes timing
and nothing else.

**It lands where the frontend spends its time.** [D-23](#8-design-decisions)
adopted the `1/2/2/2` ladder, so three of four levels run at `N = 2` and take the
1.114×, while level 0 keeps the per-row shape that suits it.

**THE NOISE FLOOR WAS MEASURED, NOT ASSUMED, AND THAT IS THE PART WORTH COPYING.**
X-29 compiled the *same* arm into two translation units and timed both, so their
spread is pure code layout. On the Cortex-A72 it is **0.0–0.3%**; on an x86_64
development machine it reaches **10.6% at N = 2 — larger than the entire effect at
that N**, which reads `IN NOISE` there and `W wins` on the device.
[X-22](EXPERIMENTS.md) declined to close this question on a single-binary A/B and
was right to.

**A corollary worth recording:** every prior code-layout caution in this repository
— X-22's 1.46× between binaries, `morphology_path_benchmark`'s ~10% within one
object — was measured **on x86**. The device's noise floor is an order of magnitude
smaller. That does **not** retire the discipline of splitting arms across
translation units; these device numbers are trustworthy *because* they were split.
It does add a reason to prefer the reference device for A/B work, alongside the ISA
argument already in [EXPERIMENTS.md](EXPERIMENTS.md)'s "Measurement platforms".


### D-27: Phase 5.1 vectorizes two functions, and they are the same kernel shape

> **SUPERSEDED IN ITS ORDERING BY [D-28](#d-28-the-corner-response-uses-bit-sliced-33-box-sums-and-d-27s-ordering-was-wrong).**
> The profile below timed **one detection per frame**; a real frontend re-detects on
> a **3.0% duty cycle**, so corner detection is **under 2%** of frontend time rather
> than 52.7%, and **`residualSums` is ~97%**. The finding that both kernels are
> **addressing-bound rather than arithmetic-bound** — and therefore that SIMD is not
> the first move — is unaffected and is what this record is still good for.


[ROADMAP Phase 5](../ROADMAP.md#phase-5--platform-hardening) says NEON kernels but
deliberately left the target list to be "detailed once Phase 4 produces numbers".
[X-30](EXPERIMENTS.md) produced them, on the reference device at the frontend's real
operating point:

| stage | ms/frame | share |
|---|---|---|
| **`cornerMinEigenVal`'s response sweep** | **30.367** | **52.7%** |
| **`residualSums` × iterations** | **25.182** | **43.7%** |
| `gradientCovariance` + LK setup | 0.833 | 1.4% |
| corner selection (NMS, ranking, spacing) | 0.773 | 1.3% |
| `pyrDown` ×2 + both derivative ladders | 0.424 | 0.7% |

**Two functions are 96.5% of the frontend. Everything else, summed, is 3.5%.**

**They are the same kernel shape — a windowed popcount reduction — which is exactly
what [D-6](#d-6-bulk-only-reductions) reserved the NEON domain for.** On aarch64
there is no scalar popcount: `CNT` lives in the vector registers, so every word
currently pays `fmov` in and `fmov` out. Keeping a whole window in vector registers
is one piece of work that covers both call sites. **Phase 5.1 is not a catalogue of
kernels to vectorize; it is one shape, applied twice, in the order above.**

**What this rules OUT is as useful as what it rules in.** The per-pixel bit-parallel
primitives — `pyrDown`, `derivativeX/Y`, threshold — are the intuitive NEON targets
and they are **0.7% of the frontend**. Vectorizing all of them perfectly caps the
frontend gain at **1.007×**. [E-12](#register) was registered against the
derivative's +93% per-row genericity cost ([X-21](EXPERIMENTS.md)); that cost is
real and is **not worth removing**, and E-12 closes on that basis.

**Two earlier results are re-weighted by this profile, and both stay correct.**
[X-29](EXPERIMENTS.md)'s accumulator win is 1.114× on a stage worth 1.4%, i.e.
**~0.17% end to end** — the right answer to E-13's question and not a big lever.
[D-22](#8-design-decisions)'s streaming corner ring is confirmed as a **footprint**
decision: selection is 2.5% of detection, the response sweep 97.5%.

**The context is [X-28](EXPERIMENTS.md)'s unmet criterion 4** — binCV is **14×
slower** than a SIMD, 12-threaded OpenCV on the same binary content. That criterion
was not restated, and this record is what makes it actionable rather than a
standing complaint.


### D-28: the corner response uses bit-sliced 3×3 box sums, and D-27's ordering was wrong

`cornerMinEigenValRow` dispatches at `blockSize == 3` — `seal_params.yaml`'s value
and the frontend's — to a form that computes the 3×3 covariance as **box sums of
bit-planes**, word-at-a-time, with full adders. Other block sizes keep the
per-pixel form, the same shape [D-22](#8-design-decisions) uses. The frame-map
`cornerMinEigenVal` has its own column-major sliding implementation and is
deliberately untouched: it is not on the frontend's path, and changing it would add
an equality surface for no measured gain.

**Measured on the reference device, 752×480 real reference content, bit-exact:**

| arm | ms | vs shipped |
|---|---|---|
| per-pixel (was shipped) | 37.934 | 1.00× |
| bit-sliced box sums | 7.886 | **4.81×** |
| + word-level sparsity skip | **5.433** | **6.98×** |

**Why it was there to be had.** The per-pixel form issued one `clipRegion` and ~12
popcounts **per pixel** over a window **three bits wide in a 32-bit word**.
Sweeping `blockSize` 3/5/7/9 and fitting `T = A + B·bs²` gave **A ≈ 12.1 ms against
B·9 ≈ 2.3 ms**: the kernel was **84% addressing**. This is
[D-2](#8-design-decisions)'s own technique — the one `pyrDown`'s `boxSum4` already
uses — applied to the kernel it had never been applied to.

**Exact, not approximate.** Box sums of bits are exact integers and
`minEigenValue` takes the same integers, so the response is bit-identical and
therefore so are the corners, their order and their count. Verified on synthetic
texture and four real frames from 8.69% to 26.67% set; `test_corner` passes 3 655
checks unchanged.

---

**AND [D-27](#d-27-phase-51-vectorizes-two-functions-and-they-are-the-same-kernel-shape)'s
TARGET ORDERING WAS WRONG. This record corrects it.**

D-27 put the corner response at **52.7%** of the frontend and `residualSums` at
43.7%, from [X-30](EXPERIMENTS.md)'s profile. **X-30 timed one detection and one
track per frame.** A real frontend re-detects only when tracks run down —
**measured: 12 re-detections in 399 frames, a 3.0% duty cycle**. So detection is
**under 2%** of real frontend time, not 52.7%, and this 6.98× kernel win moves the
sequence-level frontend by **1.04%** (22.82 → 22.01 ms/frame).

Corrected weighting:

| stage | share of the real frontend |
|---|---|
| **LK tracking (`residualSums`)** | **~97%** |
| corner detection, amortized | ~2% |
| build | ~2% |

**D-27's method was sound and its arithmetic was sound; its workload was not.** The
profile measured something adjacent to the frontend rather than the frontend. That
is the same failure as X-25's RMS over a tailed distribution and X-24's clipping
attribution — **a number measured on the wrong thing, with nothing in the number
itself to say so**.

**What survives from D-27:** both hot kernels are **addressing-bound, not
arithmetic-bound**, so SIMD is still not the first move. `residualSums` was measured
at ~9.4 cycles per popcount — **and [D-29](#8-design-decisions) shows that figure was
over-interpreted: tap extraction is 13.7%, not 90%. That kernel is arithmetic-bound
and SIMD IS its lever.** **The next target is deriving `t01` from `t00` and `t11` from
`t10` by one shift instead of four independent extractions**, and it is now
~97% of the frontend rather than 43.7%.

**What survives from this entry:** the kernel is free at runtime, bit-exact, and a
real win for any caller that detects often — `goodFeaturesToTrack` used directly,
or a frontend that re-seeds every frame. It is kept on those grounds, not on a
frontend number it does not deliver.


### D-29: `residualSums` is arithmetic-bound, so SIMD is its lever — unlike the corner response

[X-32](EXPERIMENTS.md) tried the obvious reformulation and it **lost**. `t01` is
`t00` shifted one pixel and `t11` is `t10` shifted one pixel, so two of four
`ReplicatedShiftedRow::word()` calls per word can be replaced by a shift and an or.
The identity holds exactly — **0 of 130 windows differ** — and the result is
**0.974×**, i.e. slower.

**Because tap extraction is 13.7% of the kernel, not ~90%:**

| variant | share of full |
|---|---|
| full `residualSums` | 100% |
| taps only, no popcounts | **13.7%** |
| popcounts only, no taps | 18.9% (a floor, not a measurement — see X-32) |

**THE ~9.4-CYCLES-PER-POPCOUNT FIGURE DOES NOT MEAN WHAT IT WAS USED TO MEAN.** It
was measured correctly and then over-interpreted: a popcount is 1 cycle throughput,
so 9.4 was read as "90% is addressing". But the loop issues `20N²` popcounts *and* a
comparable number of masks, ANDs and accumulates — at `N = 2`, ~240 operations per
word of which popcounts are ~33%. **9.4 cycles per popcount is simply what a loop
with ~5 other operations per popcount and a long dependency chain looks like.** The
ratio was real; the localisation was invented. It is cited in
[D-27](#8-design-decisions) and in X-31's rationale, and both are annotated.

**THIS SPLITS THE SIMD RECOMMENDATION BY KERNEL, WHICH IS THE USEFUL RESULT.**

* **The corner response WAS addressing-bound** — 84% per-pixel overhead producing
  nothing — so reformulating beat vectorizing, and did:
  [D-28](#8-design-decisions), 6.98× bit-exact.
* **`residualSums` is NOT.** It has no comparable dead weight; it is doing a large
  amount of distributed work. Masks, popcounts and accumulates all vectorize, and
  NEON would additionally relieve the register pressure of ten live tap words at
  `N = 2`. **SIMD is the right lever here, and it was the wrong one there** — 84%
  removable overhead against 13.7%.

**S3 is rejected and not shipped.** `benchmark/residual_benchmark.cpp` and its arms
are committed regardless: a rejected optimisation with a measurement attached is
what stops it being tried again.


### D-30: the sliced signed sum has a NEON path, and it is D-6's reservation being spent

`impl::slicedSignedSum` gains a vector path at `N == 2, uint32_t` behind
`BINCV_HAVE_NEON && __aarch64__` — the depth three of four levels of the adopted
`1/2/2/2` ladder ([D-23](#8-design-decisions)) run at. Every other `N`, word type
and platform keeps the scalar path, which remains **the reference and the equality
oracle**.

**Measured on the reference device:** **1.24×** on `residualSums`, **1.21×** on the
whole LK stage (25.540 → 21.088 ms), and LK is 94.7% of the real frontend, so
**~1.20× end to end**. Bit-exact: 0 of 130 windows differ, and on-device `ctest`
passes `test_opticalflow` with the vector path live, including the per-pixel oracle
at `N = 1..5`.

**This is [D-6](#d-6-bulk-only-reductions) being cashed in for the first time.** D-6
forbade exposing a per-word popcount, on the argument that aarch64's `CNT` lives in
the vector registers and a per-word scalar popcount pays `fmov` in and `fmov` out.
The payoff is exactly here: `slicedSignedSum` issues `2N²` popcounts — eight at
`N = 2` — and batching them into lanes crosses the register domain **once instead of
eight times**. **None of this would have been available if callers held
`popcountWord`**; the eight counts had to be inside one function for the crossing to
be collapsible.

**The ceiling was measured before the kernel was written, and it bounded the result
correctly.** Batched NEON popcount against scalar, everything else stripped, is
**3.42×**; the real kernel gets **1.24×** because the popcounts are diluted by tap
extraction (13.7%), masks and accumulator updates. **The 3.42× is not the result and
is not quoted as one** — that would repeat the error [D-28](#8-design-decisions)
records, where a 6.98× kernel win moved the frontend 1%.

**Most of the ceiling remains, and where it is is known.** The horizontal add still
runs once per call — ~620 domain crossings per window — where the ceiling amortized
its extraction across the buffer. Carrying **vector accumulators across the window**
is the remaining 2–3×, and it is a larger change: `TapSums` becomes vector state.
That is [E-18](#register), not this record.

**`UseNeon` is a template parameter, not a tuning knob.** It exists so both
spellings can be compiled and compared on the same machine, which is how the
bit-exactness claim stays checkable rather than asserted.


### D-31: the residual extracts its window into one aligned word

`impl::residualSums` dispatches to an aligned path when the clipped region fits a
single word — which at `seal_params.yaml`'s 31×31 is every window at every word
type binCV supports — and keeps the per-word path for wider windows.

**A 31-pixel window at an arbitrary offset spans 1.94 `uint32_t` words on average.**
It fits in one only when `x0 % 32 ≤ 1`, two cases in thirty-two, so the general path
issued **twice the popcounts it needed**, each covering 15.5 useful pixels instead
of 31. The aligned path extracts the region into bits `[0, width)` of one word, and
**the taps cost nothing extra** — `ReplicatedShiftedRow` already shifts, so
`word(0)` with `off = x0 + tapX` returns exactly the source bits at the window's
left edge.

**Measured on the reference device, bit-exact (0 of 130 windows differ):**

| | |
|---|---|
| kernel | **2.13×** |
| LK stage | 21.088 → **11.638 ms**, 1.81× |
| frontend | 22.01 → **13.55 ms/frame**, 1.62× |

**It beat its own ceiling**, which was 1.463×: the ceiling measured only the word
count, and the aligned path also deletes the per-word loop and its head/tail mask
construction. A bound on one mechanism does not bound a change that removes two.

**`RegionWords` gained `x0`/`x1`** — `regionFromExtent` is handed them and was
throwing them away.

**WHERE THIS LEAVES THE OPENCV COMPARISON.** LK against LK, same points, same bits,
OpenCV pinned to one thread: binCV is **3.08× slower on the `1/2/2/2` ladder and
1.34× slower on `1/1/1/1`** — against 4.11× and 2.00× before. **At `1/1/1/1` binCV
is within 1.34× of SIMD OpenCV while using 8× less memory**, which is a materially
different claim from where this started (14×).

**And it makes the LADDER the dominant speed factor.**
[D-23](#8-design-decisions) adopted `1/2/2/2` on accuracy with its speed cost
estimated at 1.35×; isolated it is **2.30×**, and it was chosen when corner
detection was believed to be 52.7% of the frontend rather than 2%. That decision is
**not reversed here** — it bought real accuracy and this record measures speed — but
it is now the largest single speed lever left, and [E-19](#register) exists to
re-decide it against the corrected profile rather than leave it standing on the old
one.


### D-32: the tap machinery, and bit-parallel tracking reaching parity with SIMD OpenCV

After [D-31](#8-design-decisions) aligned the window, `residualSums`' **arithmetic
was already ahead of OpenCV's** — 0.65 popcounts per pixel at `N = 1` against
~1.2 SIMD ops — and binCV was still slower. The remaining gap was entirely
machinery: per window row, ~5 ops/pixel of addressing around 0.65 ops/pixel of work.

Two changes, both bit-exact ([X-35](EXPERIMENTS.md)):

* **The `+1` tap is a shift.** Aligned, `t01`'s bits lie inside the word `t00`
  already holds, so `t01 = t00 >> 1` and **two of four displaced-row constructions
  disappear**. [X-32](EXPERIMENTS.md) tried this identity in the per-word path and
  it LOST at 0.974×, because there `t01` needed a bit from the next word.
  **D-31 is what made it true** — a rejected optimisation became correct because an
  unrelated change moved the ground under it.
* **An interior fast path.** `displacedRow` built the replicate border
  unconditionally, two `edgeFill` loads per tap, for windows that are mostly
  interior.

**Cumulative on the LK stage, reference device: 25.540 → 7.421 ms, 3.44×**
(D-30 1.21×, D-31 2.19×, this 3.44×).

**AND IT REACHES THE RESULT THE PROJECT WAS FOR.** LK against LK, same points, same
bits, OpenCV pinned to one thread, median of seven repeats on an idle machine:

| | median ms |
|---|---|
| binCV `1/2/2/2` (shipped) | 9.819 |
| **binCV `1/1/1/1`** | **4.216** |
| **OpenCV `CV_8U`, 1 thread** | **4.134** |

**At `N = 1`, bit-parallel tracking is level with vectorized byte-per-pixel tracking
— 1.02× — on an eighth of the memory.** The arithmetic was always ahead; what stood
in the way was addressing, not the idea. That is the honest form of the claim
[§1](#the-motivating-result) opens with.

**What remains is the ladder, and only the ladder.** `1/2/2/2` costs **2.33×**, and
that is now the entire difference between parity and 2.38× slower. It was adopted on
accuracy under a profile that has since been corrected twice, and
[E-19](#register) is no longer one lever among several — it is the only remaining
one of this size.

**A methodological note that cost a wrong number.** An earlier reading of this same
comparison, taken while `verify.sh` was building in the background, reported 1.00×
— and OpenCV's own time swung **4.425 → 3.803 → 5.480 ms on identical code**, a
1.44× spread from machine load alone, larger than most effects this project
measures. **A single timing run on a busy development machine is not a
measurement.**


### D-33: the residual batches across TAPS at N = 1, and the footprint buys no speed

Two results, and **the second matters more than the first**.

**LK IS COMPUTE-BOUND. THE 8× FOOTPRINT ADVANTAGE DOES NOT CONVERT INTO TRACKING
SPEED.** [X-36](EXPERIMENTS.md) measured it on the reference device: **33× more
points and 36× more data move the per-point cost by under 13%.** A 31×31 window is
**120 bytes at one bit** — two to four cache lines either way — and the compute per
window dwarfs the load.

This project has carried an implicit assumption that the footprint result and the
speed result reinforce each other. **They are independent.** The footprint decides
what fits on a device — [D-31](#8-design-decisions)'s 6.23× end to end is real and
is the stronger of binCV's two claims — but further speed has to come from doing
less work, not from touching less data. Anyone reasoning about this library's
performance should start from that.

**The optimisation.** `impl::slicedSignedSum`'s NEON path batches the `N²` plane
pairs, so at `N = 1` there is exactly one pair and it does nothing — meaning
**level 0, the largest level of every ladder, ran fully scalar even on aarch64**.
The structure that exists at *every* depth is the five taps, and four fit one
128-bit register. And because [D-31](#8-design-decisions) aligned the window, each
row is one word, so the lane accumulators **run the whole window and cross the
register domain once per window** instead of once per row.

**1.736× on the kernel at `N = 1`, bit-exact.** `N == 2` keeps the plane-pair
batching, which is the better shape at that depth — the two batchings compete for
the same registers and do not compose.

**BUT THE LADDER GATES IT.** The LK stage moved only **1.04×**, because `1/2/2/2`
has one level at `N = 1` and three at `N = 2`, and every level costs the same in LK.
At `1/1/1/1` all four levels would take the 1.736×. So [E-19](#register) is no
longer only about the `N²` arithmetic: **the ladder also decides how much of the
vectorized path is reachable at all.**

**Cumulative, reference device, LK track:**

| ladder | before D-30 | now | |
|---|---|---|---|
| `1/1/1/1` | 20 485.6 µs | **5 479.8** | **3.74×** |
| `1/2/2/2` | 27 571.5 µs | **9 639.6** | **2.86×** |

**And binCV still has no x86 vector path at all**, so [D-32](#8-design-decisions)'s
parity result was binCV **scalar** against OpenCV **SSE**.


### D-34: on the deployment target binCV's tracker is 1.4×–11× faster than SIMD OpenCV

**Every previous reading of criterion 4 was taken on x86, where binCV has no vector
path at all.** [D-32](#8-design-decisions)'s parity was binCV **scalar** against
OpenCV **SSE**. On the reference device binCV has NEON ([D-30](#8-design-decisions),
[D-33](#8-design-decisions)) and so does OpenCV — its 4.10 build reports
`Baseline: NEON FP16`. [X-37](EXPERIMENTS.md) is that comparison, SIMD against SIMD,
on the platform the library targets, with iteration count forced so both trackers do
identical work.

| iterations | binCV `1/1/1/1` | binCV `1/2/2/2` |
|---|---|---|
| 1 | **11.1× faster** | 3.4× |
| 4 | **8.4×** | 2.7× |
| 20 | **5.0×** | 1.7× |
| 50 | **4.0×** | 1.4× |

**Fitted `T = setup + iterations × slope`:**

| arm | setup ms | ms/iteration |
|---|---|---|
| binCV `1/1/1/1` | **1.077** | **0.2264** |
| OpenCV `CV_8U` | **13.810** | **0.7065** |

**OPENCV'S SETUP IS 12.8× binCV'S, AND THAT IS WHERE THE ADVANTAGE LIVES.** OpenCV
copies the warped patch into `IWinBuf`/`derivIWinBuf` — **961 pixels × 3 shorts per
point per level** — before it iterates. **binCV copies nothing**: it reads the frame
in place through the region walk, which is what "kernels take views, never owning
containers" and "no scratch buffer" have been buying all along.

**[X-36](EXPERIMENTS.md) is what makes this legible rather than lucky.** That entry
showed the kernel is **compute-bound**, so the 8× smaller data does not make the
arithmetic faster. What it does is **remove an entire stage** — there is no patch to
copy when the window is read where it lies. The footprint advantage converts into
speed as a *structural* saving, not a bandwidth one, and the two entries only make
sense together.

The per-iteration advantage is real but smaller at **3.1×**, so the shape is "no
setup, and cheaper steady state", with the first term dominating at the iteration
counts a frontend actually runs.

**What this does NOT claim.** It is **LK against LK**, not the whole frontend.
[X-28](EXPERIMENTS.md)'s end-to-end comparison ran on x86 and included detection,
build and preprocessing; **the frontend comparison has never been run on the
device**, because the EuRoC sequence is not there. Criterion 4 is answered for the
**tracker on the deployment target** and not yet for the frontend — [E-20](#register).


### D-35: all four ROADMAP success criteria are met, on the deployment target

[X-38](EXPERIMENTS.md) ran the **whole frontend** — detection, pyramid build,
preprocessing and tracking — against OpenCV on the reference device, the **full
1710-frame** EuRoC V1_02_medium sequence, both frontends on bit-identical input,
OpenCV pinned to one thread.

| criterion | binCV | OpenCV | |
|---|---|---|---|
| 1 · tier 1 bit-exact | enforced per operation | — | ✔ |
| 2 · median track lifetime | **11 frames** | 12 | one frame short |
| 2 · per-frame survival | **96.4%** | 96.6% | 0.2 points short |
| 2 · flow difference | **median 0.0434 px** | — | 95.4% within 1 px |
| 3 · peak footprint | **436 704 B** | 2 719 832 B | **6.23× smaller** |
| **4 · speed** | **10.644 ms/frame** | 16.289 ms/frame | **1.53× FASTER** |

**1.53× faster and 6.23× smaller simultaneously.** That is the result
[§1](#the-motivating-result) opens by asking for.

*(Speed re-measured by [X-49](EXPERIMENTS.md) after [D-37](#8-design-decisions)'s
window-carried accumulators landed: 11.198 → 10.644 ms, with **every criterion-2
figure bit-identical**. The accuracy rows below are X-38's and are unchanged.)*

**THE EARLIER "EQUAL ON CRITERION 2" WAS AN ARTIFACT OF AN EASY PREFIX.** This
record previously read 13 vs 13 frames and 97.1% vs 97.1% — *equal* — from the first
**692** frames, which were all that had transferred before the dataset drive dropped.
Re-running that prefix on the current commit reproduces those accuracy figures
**exactly** (13/13, 97.1%/97.1%, median 0.0386 px, 97.4% within 1 px), so nothing
regressed: the remaining 1018 frames are simply harder, and **both** frontends
degrade on them. binCV degrades slightly more. Backing the prefix out of the totals
puts the tail at roughly **95.9% survival for binCV against 96.3% for OpenCV** — a
0.34-point gap where the prefix had none. Criterion 2 asks for *agreement frame by
frame*, and one lifetime frame out of twelve still meets it; **parity does not, and
is no longer claimed.**

Where the gap opens is informative rather than mysterious: it opens on **large
motion**, which is where coarse pyramid levels carry the estimate, and coarse levels
are exactly what the downsampling filter builds. [X-39](EXPERIMENTS.md) measured the
filter arms spreading furthest apart at its largest shifts — `DIRECT_SUBSAMPLE` falls
to 59.4% yield at `shift (6, 4)` against 94.1% for the shipped box — so
[D-36](#d-36-box_2x2-stays-the-default-the-filter-set-ships-as-options)'s
`BOX_2x2`/`BOX_3x3` trade and this 0.34-point tail plausibly meet. **Plausibly** is
the honest word: nothing has yet measured the filter arms *through the frontend* on a
sequence, and the connection is a hypothesis this record is registering, not a
finding.

**EVERY PREVIOUS READING OF CRITERION 4 WAS A FACT ABOUT x86, NOT ABOUT THE
PRODUCT.** 14× slower → 6.3× → 3.8× → parity → **1.46× faster**. The first four were
measured on a platform where binCV has **no vector path at all** — ROADMAP 5.3 is
still unwritten — against an SSE-vectorized OpenCV. **The measurements were correct
and the platform was wrong**, and it took [X-37](EXPERIMENTS.md) to notice after
four entries had reported the gap as a property of the library.
`benchmark/frontend_sequence.cpp` now prints which case it is in, because the fixed
disclaimer it carried had quietly gone false the moment D-30 landed.

**THE PROFILE HAS MOVED AND SO HAS THE NEXT TARGET.** At the real duty cycle:

| stage | ms/frame | share |
|---|---|---|
| track (LK) | 7.815 | 69.8% |
| **build (`pyrDown` + derivatives)** | **2.811** | **25.1%** |
| detect | 0.571 | 5.1% |

**`pyrDown` is now a quarter of the frontend**, up from 4.5%, because LK got 3.44×
faster ([D-32](#8-design-decisions)) and the build did not. That is precisely where
the downsampling-filter design space ([E-21](#register)) lands, so the next piece of
work and the next hot stage are the same thing.

**On the speed figures.** binCV lands at **11.169 / 11.195 / 11.198 ms** across the
three runs of this experiment — 0.26% spread. OpenCV moves 16.324–17.060 ms (±2.3%)
on identical input, so the *ratio* carries OpenCV's variance, not binCV's: 1.46× and
1.52× are the same measurement. The conservative figure is the one quoted.


### D-36: `BOX_2x2` stays the default; the filter set ships as options

> **ACCURACY FIGURES RE-MEASURED ON THE SHIPPED PIPELINE ([X-53](EXPERIMENTS.md)).**
> The harness behind these yield numbers used to build its levels in floating point;
> it now runs binCV's own `pyrDownFiltered` cascade. **The filter rankings moved by at
> most 0.16 points and stand as first-hand** — the hedge that relative comparisons at a
> fixed ladder would survive the idealisation was correct. What did NOT survive is the
> harness's pricing of *ladder depth*, which [D-44](#d-44-the-accuracy-harness-measures-a-different-question-from-the-frontend)
> shows disagrees with the frontend by 4.2 points for reasons still unknown. The speed
> figures were always measured on the real kernels.
>
> **PARTLY SUPERSEDED BY [D-39](#d-39-the-filter-frameworks-3-tax-was-genericity-and-d-36-is-restated).**
> The decision below stands — `BOX_2x2` is still the default and the set still ships
> as options — but **two of its stated reasons do not.** [X-42](EXPERIMENTS.md) removed
> the framework tax this record names as a caveat, and with it removed the
> `GAUSSIAN_5x5` anchor is **affordable** (1.32× faster than OpenCV, not 0.97× slower)
> and `BOX_3x3` no longer costs less than `GAUSSIAN_3x3`. Read the speed column here
> as history; D-39 carries the current prices.

[X-39](EXPERIMENTS.md) mapped the pyramid's two-dimensional design space —
downsampling filter × bit depth — and `impl::pyrDownFilteredRoute` now implements
five of the reference's six `LKPyrDownFilterType` variants as bit-sliced kernels,
**verified exact against a per-pixel integer reference**.

| filter | µs (640×480→320×240) | vs shipped | yield vs anchor | est. frontend |
|---|---|---|---|---|
| **`BOX_2x2` (default)** | **93.7** | 1.00× | −1.26 | **11.169 ms, 1.48× faster** |
| `DIRECT_SUBSAMPLE` | 20.9 | 0.22× | −12.65 | 10.978 ms |
| `BOX_3x3` | 398.0 | 4.25× | **−0.10** | 11.968 ms, 1.38× faster |
| `GAUSSIAN_3x3` | 497.7 | 5.31× | −0.37 | *dominated by `BOX_3x3`* |
| **`GAUSSIAN_5x5`** (anchor) | **2 352.9** | **25.10×** | 0.00 | **17.099 ms — SLOWER than OpenCV** |

*(Yields are N=3 over the **full 1710-frame** sequence, 1.18 M eligible keypoint-cases
per cell — X-39's sequence arm. The single-frame table this record first carried
overstated every gap by 1.8× to 8× while getting the **ordering exactly right**; the
speed column is unchanged.)*

**THE TWO AXES ARE NOT INDEPENDENT, WHICH IS THE FINDING BENEATH THE TABLE.** A 2×2
box sum of four values has five possible outcomes, so it **saturates immediately**:
over the sequence it gains **+0.02** yield points across the whole N=2→7 axis, a flat
line. A 5×5 Gaussian keeps paying and gains **+0.73**. **The filter decides how much
depth is useful**, so [E-19](#register)'s ladder question was never separable from the
filter it was asked about — and every bit-depth result in this project was measured at
the filter that benefits least from depth. The corollary is a load-bearing one for the
shipped configuration: **binCV's 2-bit levels give up essentially nothing under
`BOX_2x2`**, so `1/2/2/2` is not a compromise, it is the right depth for that filter.

**Standard-LK accuracy is reachable and costs more than it is worth here.**
`GAUSSIAN_5x5` adds ~5.9 ms and puts binCV **behind** OpenCV, forfeiting
[D-35](#8-design-decisions)'s criterion-4 result. SEAL §4.2.2 reached the same
conclusion by a different route (SRAM), so the two agree on the choice while
disagreeing on the binding constraint.

**`BOX_3x3` IS THE GAUSSIAN ANCHOR, FOR PRACTICAL PURPOSES**: **−0.10 points at
1.18 M samples**, for +0.8 ms, still 1.38× faster than OpenCV — and it **dominates
`GAUSSIAN_3x3`**, being both cheaper and more accurate at every depth from N=3 up.
The single frame read this as closing 65% of the gap to standard LK; the sequence
reads it as **92%**, at a **sixth** of the anchor's cost. **Standard-LK accuracy is
available in a bit-sliced kernel** — that is a stronger claim than this record
originally made, and it survives 1 900× more data than it was first measured on.

**Three quarters of every filtered number is framework, not filter.** The generic
route runs `BOX_2x2` at **2.96×** the hand-written one **computing the same
function**, so `BOX_3x3`'s 4.25× is roughly 1.4× of filter and 3× of genericity.
The frontier is measured on a framework that has had no optimisation at all
([E-22](#register)).

**`MEDIAN_3x3` is deliberately not implemented.** X-39 measured it **5.07 points
below the box** and *exactly* flat in `N`: a median of a mostly-zero neighbourhood returns
zero, so it **erodes** a sparse edge map rather than blurring it. It is the right
tool in the temporal denoiser, which is where SEAL uses it.

**The `BOX_2x2`/`BOX_3x3` trade — 1.16 yield points for ~0.8 ms — is the caller's**,
as [D-24](#8-design-decisions) put route (a) and [D-32](#8-design-decisions) put
`maxIterations`. binCV ships the default and the options, not a decision.

**The per-frame spread is six times the gap, which is why this needed a sequence.**
At N=3, per-frame yield runs p10 ≈ 89% / median ≈ 94.4% / p90 ≈ 98.4% for `BOX_2x2`
and p10 ≈ 92% / median ≈ 95.5% / p90 ≈ 98.8% for `BOX_3x3`. The bands separate
cleanly when whole samples are compared and **overlap almost entirely frame to
frame** — so the ranking is a claim about the mean, and **no single frame could have
established it.** Ten independent stride-10 shards reproduce every percentile of
every arm to within 0.7 points.

### D-37: `residualSums` is extraction-bound, not count-bound

[X-40](EXPERIMENTS.md) gave N = 2 the window-carried lane accumulators
[D-33](#8-design-decisions) gave N = 1 — `impl::alignedResidualSumsNeon2`, which puts
the four taps in lanes and folds the four plane pairs inside the row with
`vmlaq_n_s32`, reducing **once per window** instead of ~310 times. It is exact, and
the gate now proves it: `Flow.ResidualNeonMatchesScalar_{N1,N2,N3}` compares the
vector and scalar spellings over 728 windows per depth, **0 differ on aarch64**.

**It delivered 1.069×, against a 1.461× ceiling.** That is worth about **1.52×
against OpenCV** on the frontend, from 1.46× — **under four percent**, and it is
quoted that way.

**THE REAL RESULT IS THE FLOOR ARM.** Running the whole per-row tap machinery with
the counting **removed** costs **275.6 µs against the kernel's 607.6 — 45.4%**. So:

| | |
|---|---|
| if counting were **free** | the kernel gains **2.205× and no more** |
| [E-18](#register) was chartered on | a remaining **2–3×** |
| this reshaping collected | **1.069×** of that 2.205× |

**E-18 is resolved NEGATIVELY on its own terms: the 2–3× is not in the counting.**
[D-29](#8-design-decisions) put tap extraction at **13.7%**; it is now **45.4%**, not
because it got slower but because [D-30](#8-design-decisions),
[D-31](#8-design-decisions), [D-33](#8-design-decisions) and X-35 made the counting
about three times faster and never touched it. **The same thing that happened to
`pyrDown` in [D-35](#d-35-all-four-roadmap-success-criteria-are-met-on-the-deployment-target)
has happened inside `residualSums`.** The next target is addressing, not arithmetic —
registered as [E-23](#register).

**TWO CEILINGS IN A ROW HAVE OVERSTATED THE DELIVERED RESULT.** X-33: 3.42× ceiling,
1.24× delivered. X-40: 1.461× ceiling, 1.069× delivered — and X-40's ceiling was
deliberately built close to the real shape. Even isolating the counting, the part the
reshaping touches gained only **1.133×**, because *in situ* the accumulators compete
for registers with the tap machinery and no ceiling that omits the tap machinery can
see that. **A ceiling bounds the shape, not the kernel.** The
ceiling-before-arm discipline stands — it is cheap and it has cancelled work before —
but its output is an upper bound on one transformation, not a forecast.

**A cost declared rather than hidden:** `alignedResidualSumsNeon2` duplicates the
per-row extraction block from `alignedResidualSums`, as the N = 1 path already did.
**Extraction now exists in three copies and extraction is what E-23 must change**, so
E-23 starts by paying that down.

### D-38: `residualSums`' extraction is instruction-bound — not addressing, not layout

[X-41](EXPERIMENTS.md) took the 45.4% of `residualSums` that
[D-37](#d-37-residualsums-is-extraction-bound-not-count-bound) identified as
extraction and asked what it is made of. **Two hypotheses, both pre-registered, both
wrong.**

| what was removed | result |
|---|---|
| every loop-invariant: both `(w0, s)` descriptors, their `s == 0` case, their bounds test, the `.row(y)` multiplies, the `interior` branch (y-loop split) | **1.023×** |
| the memory system: same code on a level small enough that all ten planes fit in **L1D** together | **1.129×** |

**The 8× overfetch is real and is not the constraint.** A 31×31 window touches 31 rows
of ten separate planes — **310 distinct cache lines, ~19.8 KB fetched for ~2.5 KB of
useful bits** — and eliminating it entirely buys 13%.

**What binds is the instruction stream.** ~3 660 cycles per window, ~118 per row, for
roughly a hundred instructions of shifts, ors, masks and border machinery. The lever
that remains is to *issue fewer*: the twelve `alignedWord` extractions in a row share
exactly **two** `(w0, s)` descriptors, so twelve scalar load-shift-or sequences could
be about three vector ones. Registered as [E-24](#register).

**A PREDICTION WITHDRAWN BEFORE IT WAS ACTED ON.** X-41's Band C pre-committed a
successor — *"the extraction is loads, not addressing, and E-23's successor is a
memory-layout question"*. The second measurement tested that prediction instead of
adopting it, and **it is false**. Writing the rule down first is what made the
difference visible; a rule that only fires bands and never has its reasoning checked
would have sent the next experiment at the layout.

**This is the third profile relocation in a row, and the pattern is now the finding.**
[D-28](#8-design-decisions) moved the target from detection to tracking;
[D-35](#d-35-all-four-roadmap-success-criteria-are-met-on-the-deployment-target) moved
it from tracking to `pyrDown`; D-37 moved it from counting to extraction; D-38 finds
extraction is not addressable by the two obvious means. **Every optimisation this
project lands relocates the bottleneck rather than removing it**, which is what a
kernel with no remaining stalls looks like from the inside.

**The budget is closed and stated.** X-40 capped `residualSums` at **2.205×** if
counting were free; X-41 adds that extraction is not cheaply removable either. From
this kernel alone the frontend cannot pass **~1.9× against OpenCV**.

### D-39: the filter framework's 3× tax was genericity, and D-36 is restated

[X-42](EXPERIMENTS.md) changed **three signatures** in `impl` — `addShifted`'s extents
and shift, `weightedAxis`' tap count and weights and output width, and
`requantizeWeighted`'s divisor via a new `divideByConstantT` — from runtime arguments
to template parameters. `F` was **already** a template parameter, so every one of
those values was a compile-time constant that the helpers were discarding at their own
signatures. **No algorithm changed**, and `tests/test_pyramid.cpp` passes with the
identical 262 322 checks, still exact against the per-pixel reference.

| arm | [X-39](EXPERIMENTS.md) | **X-42** | speedup |
|---|---|---|---|
| hand-written `pyrDown` N=3 *(control)* | 93.7 | 93.8 | 1.00× |
| **generic `BOX_2x2` N=3** | 277.8 | **111.9** | **2.48×** |
| `BOX_3x3` N=3 | 398.0 | **228.0** | 1.75× |
| `GAUSSIAN_3x3` N=3 | 497.7 | **225.7** | 2.21× |
| **`GAUSSIAN_5x5` N=3** (anchor) | 2 352.9 | **549.8** | **4.28×** |

**The generic route ran `BOX_2x2` at 2.96× the hand-written one; it now runs it at
1.19×.** There was never a genericity/speed trade here to make — only a signature that
threw the constants away.

*(Yield figures here come from the same idealised-pyramid harness
[X-51](EXPERIMENTS.md) found wanting — see the note on D-36. The speed figures, which
are what this record is mainly about, were measured on the real kernels and stand.)*

**THE STANDARD-LK ANCHOR IS AFFORDABLE, WHICH REVERSES
[D-36](#d-36-box_2x2-stays-the-default-the-filter-set-ships-as-options)'s CENTRAL
CLAIM.** Scaled by the level geometry against X-38's 11.198 ms and OpenCV's 16.324:

| filter | D-36 recorded | **now** |
|---|---|---|
| `BOX_2x2` shipped | 11.169 ms, 1.48× | 11.198 ms, **1.46×** |
| `BOX_3x3` | 11.968 ms, 1.38× | **11.550 ms, 1.41×** — **+0.35 ms, was +0.80** |
| **`GAUSSIAN_5x5`** | **17.099 ms, 0.97× — SLOWER than OpenCV** | **12.395 ms, 1.32× FASTER** |

D-36 said standard-LK accuracy *"costs more than it is worth here"* and would forfeit
[D-35](#d-35-all-four-roadmap-success-criteria-are-met-on-the-deployment-target)'s
criterion-4 result. **It would not.** The anchor costs **+1.20 ms** and leaves binCV
**1.32× faster than OpenCV** — so binCV can have standard-LK pyramid accuracy *and*
criterion 4, trading 0.14× of speed for 1.25 yield points. **`BOX_2x2` stays the
default** on the footprint and speed grounds that have not changed, but the reason the
anchor is not the default is now *"the default should be cheap"*, not *"the anchor is
unaffordable"*.

**`BOX_3x3` no longer dominates `GAUSSIAN_3x3` on cost.** D-36 recorded it as both
cheaper and more accurate; it is now **228.0 against 225.7 µs, one percent the other
way**. Still the one to prefer, on accuracy (−0.10 against −0.37) — but at **equal
cost**, not at a 1.25× discount.

**THE CAVEAT WAS LARGER THAN THE EFFECT BEING DECIDED.** X-39 mapped a design space on
an **unoptimised framework**, named the tax as a caveat and registered E-22 — and two
of its four conclusions do not survive the caveat being removed. **The registration is
what saved it**: the number was flagged provisional at the time, so this is a
correction and not a hidden error. The rule this project should keep is *measure the
framework before mapping a design space on it, or say loudly that the map is
provisional.*

**Scope:** the shipped default calls the hand-written route, so **the effect on binCV
as shipped is exactly zero.** What changed is the price of the options. The
hand-written `pyrDown` is now a **deletion candidate** at 1.19% overhead over the
generic route — a separate decision, registered as [E-25](#register), not taken here,
because it is the route every prior result in this project was measured on.

### D-40: the extraction's obstacle is the plane layout, and `residualSums` is done

[X-43](EXPERIMENTS.md) took [D-38](#8-design-decisions)'s one remaining lever —
issue fewer instructions — and measured it three ways.

| arm | µs | vs scalar extraction |
|---|---|---|
| **A — scalar extraction** (shipped) | 254.8 | 1.000× |
| **B — vector, real gather** | 288.0 | **0.885× — SLOWER** |
| **C — vector, gather removed** | 155.6 | **1.638×** |

**The shifts are cheap; the gather is not.** Twelve scalar load-shift-ors really do
become three vector ones, and that is worth **1.638×** — but `QuantMat` stacks its
planes at word offset `p × height × stride` ([§4.1](#41-storage-layout)), so the eight
words a vector wants sit in eight unrelated cache lines, and **aarch64 has no gather
instruction**. Eight scalar loads plus eight lane inserts cost *more* than the eight
shift-or sequences they replace. **Arm B is not written.**

**THE MECHANISM WAS PREDICTED BEFORE THE MEASUREMENT, which is worth recording because
[X-41](EXPERIMENTS.md)'s prediction was wrong.** X-43's rule named the stacked-plane
layout and the missing gather as the likely obstacle, and named the consequence — that
the finding would be *"the layout forbids it"* rather than *"vectorisation does not
work"*. Both held.

**THIS IS AN INSTRUCTION-COUNT ARGUMENT FOR RELAYOUT, NOT A CACHE ONE.** D-38 refuted
the *cache* case at **1.129×** — fitting the whole working set in L1 buys 13%. This is
a different case, for a different reason, worth **1.638×** on the extraction. The two
must not be conflated, and the rule pre-registered the distinction so the successor
could not inherit the refutation by association.

**WHY THE PRIZE IS SMALLER THAN ARM C.** At arm C the kernel would run 552.1 → 452.9 µs
(**1.219×**), about **1.18× on LK**, putting the frontend near **9.6 ms, ~1.70× against
OpenCV**. But arm C assumes the eight words are contiguous, and they belong to **five
separate containers** — `prev`, `dxMag`, `dyMag` as `QuantMat`s and `dxSign`, `dySign`
as `BinMat`s. Interleaving *within* a `QuantMat` yields 2-wide contiguity at best.
**The full 1.638× needs an `LKLevelN`'s five containers merged into one interleaved
allocation**, which would make every single-plane bulk operation stride instead of
stream. **Arm C is the ceiling for a design that does not exist**, and D-38's lesson
applies: a ceiling bounds the shape, not the kernel. Registered as [E-26](#register)
with its cost side attached.

**`residualSums` IS FINISHED under the current layout.** The account closes:

| | |
|---|---|
| [D-37](#d-37-residualsums-is-extraction-bound-not-count-bound): cap if counting were free | **2.205×** |
| D-37: collected by reshaping the counts | **1.069×** |
| D-38: available from addressing / from cache | **1.023× / 1.129×** |
| D-40: available from vectorising the extraction **as laid out today** | **0.885×** |

Four experiments, one small win, and the remainder is behind a container redesign. The
frontend stands at **1.52× against OpenCV**; this kernel cannot move it further without
E-26.

### D-41: interleaving will not be binCV's general layout; the rest is escalated

[X-44](EXPERIMENTS.md) priced interleaved storage on both sides, with a real buffer
rather than [X-43](EXPERIMENTS.md)'s fabricated one.

| | measured |
|---|---|
| extraction, planar → real interleaved (in-kernel, bit-exact) | 255.7 → **177.0 µs, 1.445×** |
| `residualSums` overall | 550.2 → **471.5 µs, 1.167×** |
| conversion, per level per frame | **23.7 µs** |
| **streaming one plane: planar → interleaved** | 0.605 → **3.129 µs, 5.17× COST** |
| interleaved buffer, largest N = 2 level | **92 160 B** |

**WHAT IS DECIDED: interleaving does not become binCV's general storage layout.**
Arm 4's **5.17×** settles it — striding by four words uses one word per cache line and
discards the rest. Any adoption is a **conversion confined to the operation that
benefits**, never a change to how binCV stores images.

**THE CROSSOVER — ONE DATA POINT, EXPLICITLY NOT A CRITERION.** Conversion costs
23.7 µs and each 31-row window saves 0.605 µs, so on **this** level, at **this**
geometry, on **this** device, it paid after **≈ 40 windows**. The frontend does ~600
windows per level per frame, amortising ~15×.

**That number must not be reused as a threshold.** It was measured at one level size
(376×240), one plane count (8), one word type and one cache hierarchy, and every one
of those moves it: the conversion scales with level *area* while the saving scales
with *window count*, so the crossover is a ratio between two quantities that vary
independently. **A future operation asking this question gets its own measurement**,
and this figure is useful only as evidence that the amortisation argument is sound in
principle — not as a number to check against.

**WHAT IS NOT DECIDED, AND WHY IT IS NOT MINE TO DECIDE.** Net frontend effect is
**~1.65× against OpenCV** from 1.52× — about **+8% speed**. Converting one level at a
time costs **+92 160 B on a 436 704 B peak: +21%**, taking criterion 3 from **6.23× to
5.15×**.

[CLAUDE.md](../CLAUDE.md) makes performance and footprint **co-equal**, and says memory
wins when they conflict and no explicit choice has been made. **X-44's bands were
written on speed alone — a defect in the rule, reported rather than patched after the
fact** — so the experiment cannot settle its own question. Escalated under "stop and
ask": *a decision is needed that isn't recorded in §8*. [E-26](#register) stays open,
reduced to a single yes/no with both numbers attached.

**A PREMISE CONFIRMED AND A CEILING THAT BEHAVED.** The conversion **does** amortise,
~15×, against an earlier note that priced it per-window. And X-43's fabricated buffer
overstated by only **1.14×** — far better than [D-37](#8-design-decisions)'s 1.37× or
[D-40](#8-design-decisions)'s, because it differed from the real thing only in where
the memory lived. **A ceiling's accuracy tracks how few things it abstracts away**,
which is a sharper rule than "ceilings overstate".

### D-42: above the bit-width crossover, the answer is interop — not specialisation

[X-46](EXPERIMENTS.md) measured binCV 2.5–14× slower than OpenCV above the
filter-dependent crossover (box: 4–5 bits; Gaussian 5×5: 1–2). The proposal on the
table was an internal byte-representation specialisation for wide `N`.
[X-47](EXPERIMENTS.md) built the alternative first and priced both:
`QuantMat<N>` ↔ `cv::Mat` conversions — transpose-based, 8 pixels × 8 planes per
step — now first-class for every `N`, exactness-tested (round-trip law exact at every
`N`, padding invariant held, verified numerically before the C++ existed).

| | time | peak bytes |
|---|---|---|
| round trip `toCVMatNormalized` → `cv::pyrDown` → `fromCVMat`, 8→8 | **1 906.2 µs** | **844 800** |
| native bit-sliced `GAUSSIAN_5x5` 8→8 | 7 092.8 µs | 384 000 |
| `cv::pyrDown` alone — a specialisation's ceiling | 516.1 µs | — |

All spreads under 1%. **The honest ordering is `specialisation (516) < interop
(1 906) < native bit-sliced (7 093)`: a specialisation would be FASTER, and the reason
not to build one is the cost model — a second storage layout plus a second
implementation of every kernel — not the clock.** That model was pre-registered.
Interop wins on time-per-unit-of-machinery, and the margin cannot fund the
alternative. **Closed, not deferred.**

**Footprint is 2.20× against the native path** — the interop route materialises a full
byte-per-pixel frame, exactly what binCV exists to avoid. It does not reverse the
decision, because at 8 bits binCV has **no footprint advantage to protect**
([X-45](EXPERIMENTS.md): 8 bpp on both sides by construction) and the byte buffers are
transient rather than pipeline-resident — **but that is an argument, and X-47's rule
failed to make it in advance.** The rule was written on speed alone, which is the
defect [X-44](EXPERIMENTS.md) reported in its own rule one experiment earlier;
recorded there as a rule defect rather than repaired silently.

**The two paths differ on the border and nowhere else:** 1 114 of 76 800 destination
pixels, **zero of them interior**, max |Δ| 73/255 — `cv::pyrDown`'s
`BORDER_REFLECT_101` against `pyrDownFiltered`'s zero-fill. For a caller sending a
wide intermediate out, that is an improvement; it is recorded because "3.7× faster"
without it is an incomplete claim.

**The decision rule for callers is a formula:** send an operation to OpenCV when
`native_binCV − native_OpenCV > T(in) + T(out)`, each term at the size that side
processes — 952.6 µs out and 1 616.6 µs back at 640×480, 389.3 µs back at 320×240. For
the 8→8 Gaussian that is 6 577 against 1 342, **4.9× over**; a chain pays the tax once
at each end. With [X-46](EXPERIMENTS.md)'s table this answers every wide-`N` question
without another sweep.

**Scope note:** this is also the first time `QuantMat<N>` is *reachable* from
`cv::Mat` at `N > 1` — the interop existed only for `QuantMat<1>` before. The
`QuantMat<1>` specialisation keeps its established nonzero-threshold `fromCVMat`;
the general form quantises to nearest, and the two disagree for bytes 1..127 at
N = 1 — a recorded difference, not a bug to unify.

### ~~D-43~~ WITHDRAWN: the operating point stays `1/2/2/2` + `BOX_2x2`

> **WITHDRAWN BY [X-51](EXPERIMENTS.md) BEFORE IT WAS ENACTED.** The frontend
> measurement this record itself said was required refuted it. `1/2/2/1` + `BOX_3x3`
> measures **90.6% within 1 px against the shipped point's 95.4%**, lifetime **9 vs
> 11**, and is **slower** (10.787 vs 10.644 ms). Both of X-50's accuracy claims fail
> in the same direction, because its harness builds the pyramid **in floating point**
> and quantizes each level from the float chain — modelling a pyramid with **no
> cascaded quantization error**, where the shipped one has three rounds of it. The
> harness therefore systematically **understates the cost of removing bits**: it
> priced level 3's bit at −0.69 points; the real loss is **−4.6**.
>
> [D-23](#8-design-decisions) stands, now on a frontend measurement rather than a
> proxy. The table below is left as measured — its **speed and footprint columns are
> sound** and reproduce at the frontend; only the yield column is a property of the
> idealised chain. Successor: [E-27](#register).

[X-50](EXPERIMENTS.md) swept **ladder × filter** on three axes — yield over the full
1710-frame sequence, build+track on the reference device, exact bytes — and the shipped
point is **the only one of seven that is not on the Pareto frontier**.

| ladder | filter | build+track | yield | bytes |
|---|---|---|---|---|
| **`1/2/2/1`** | **`BOX_3x3`** | **5 642 µs (−2.4%)** | **94.97% (+0.48)** | **354 720 (−0.8%)** |
| `1/2/2/2` | `BOX_2x2` *(shipped)* | 5 778 | 94.49% | 357 600 |
| `1/2/2/1` | `BOX_2x2` | 4 849 (−16.1%) | 93.80% (−0.69) | 354 720 |
| `1/1/1/1` | `BOX_2x2` | 3 311 (−42.7%) | 90.69% (−3.80) | 306 720 |

**Faster, more accurate and smaller. No trade.**

**[D-23](#8-design-decisions) WAS RIGHT ON THE PRICES IT HAD.** It fixed the filter at
`BOX_2x2` because `BOX_3x3` cost **+0.8 ms**; [X-42](EXPERIMENTS.md) re-priced it to
**+0.35 ms** by removing a genericity tax nobody had looked for. The swap — spend
level 3's bit, buy the wider filter — only became free when that tax went.

**FILTER AND DEPTH ARE SUBSTITUTES OVER PART OF THE RANGE**, which is why pricing them
on separate axes produced a dominated point. `BOX_3x3` is worth **+1.32** yield points
at `1/2/1/1`, **+1.17** at `1/2/2/1`, **+0.78** at `1/2/2/2` — and **−0.02** at
`1/1/1/1`, because a 1-bit level cannot represent the smoother result at all.

**EVERY COARSE LEVEL'S SECOND BIT STILL EARNS ITS PLACE** — E-19's open sub-question.
At a fixed filter, `1/2/1/1` loses 2.10 points and `1/2/2/1` loses 0.69. `1/2/2/2`'s
*shape* was right; what pays for dropping level 3's bit is the better filter, not the
bit being redundant.

**The frontier ships as documented operating points**, not just the default:
`1/2/2/1` + `BOX_2x2` is **−16.1% time for −0.69 points**, which a power- or
footprint-bound caller may want, and `1/1/1/1` remains the floor at −42.7% time and
−14.2% bytes.

**NOT YET ENACTED.** Changing the shipped ladder re-bases every performance number
here, exactly as the `pyrDown` swap did, so it needs [X-49](EXPERIMENTS.md)'s treatment
first — a frontend re-measure confirming accuracy is unchanged and re-stating criterion
4. **This record establishes the operating point; it does not claim the frontend has
moved.**

### D-44: the accuracy harness measures a different question from the frontend

[E-27](#register) held that the accuracy harness misled X-50 because it built its
pyramid in floating point. [X-53](EXPERIMENTS.md) fixed exactly that — the harness now
runs **binCV's own `pyrDownFiltered` cascade**, quantizing each level from the quantized
level above — and **the fix did not close the gap.**

| level 3's second bit, `1/2/2/1` vs `1/2/2/2` at `BOX_3x3` | |
|---|---|
| old float-cascade harness | −0.30 points |
| **corrected harness** | **−0.42** |
| **frontend** ([X-51](EXPERIMENTS.md)) | **−4.60** |

**Removing the cascade moved the number by 0.12 where the gap is 4.2**, and the
corrections ran in **both directions** — `1/1/1/1`, the ladder with the *most* cascaded
quantization, moved **up** 1.89 points where the mechanism predicted the largest fall.
**[X-51](EXPERIMENTS.md)'s mechanism is withdrawn.**

**What the fix DID settle:** the filter axis moved by **at most 0.16 points**, exactly
as X-51 hedged it would. [D-36](#8-design-decisions) and
[D-39](#d-39-the-filter-frameworks-3-tax-was-genericity-and-d-36-is-restated)'s filter
rankings stand and are now first-hand rather than proxied. Their warnings narrow from
*"the accuracy figures rest on an idealised pyramid"* to **"the ladder figures did; the
filter figures did not."**

**The likely cause is structural and is NOT measured, so it is not claimed.** The
harness warps a single frame and asks whether LK recovers a known warp, so `prev` and
`next` are binarizations of *the same image* with near-identical edge maps. The
frontend tracks **real consecutive frames**, whose binarizations differ wherever a pixel
sits near the threshold, over a sequence where error compounds and tracks are
re-detected. **Those are different questions.** And the tension is fundamental rather
than a bug: the harness uses synthetic warps **because it needs ground truth**, which is
precisely what makes it unrepresentative.

**THE RULE, TIGHTENED RATHER THAN RELAXED:**

> **No accuracy conclusion from the synthetic-warp harness may be promoted to a shipped
> default** — with or without a corrected cascade. It answers a *sensitivity* question,
> not a *tracking* one. **Frontend accuracy is measured at the frontend.**

That rule, in its weaker form, is what caught [D-43](#d-43-withdrawn-the-operating-point-stays-1222--box_2x2)
before it shipped. Successor: [E-28](#register).

### D-45: one word type, because the two halves of the frontend want different ones

[X-54](EXPERIMENTS.md) priced `uint64_t` against `uint32_t` on the shipped ladder, and
the answer is a split rather than a verdict:

| `1/2/2/2` | build | track | bytes |
|---|---|---|---|
| `uint32_t` | 424.5 µs | 4 838.9 | 357 600 |
| **`uint64_t`** | **255.5 — 1.66× FASTER** | **6 368.2 — 1.32× SLOWER** | +2.0% |

**The same library is 1.66× faster and 1.32× slower at the same time.** Build is
word-parallel — `pyrDown`, the derivatives — and a wider word does strictly less work
per pixel. Track is `residualSums`, whose three NEON paths are guarded on
**`sizeof(WordType) == 4`**, so `uint64_t` runs it fully scalar.

**[D-1](#8-design-decisions)'s genericity in the word type is real in the API and NOT
real in the tracker's fast path.** That was previously implicit in three `if constexpr`
guards; it is now a measured number.

**binCV's frontend is track-dominated (68.3% against build's 26.3%), so the stage that
loses is the stage that matters** — weighting by those shares puts `uint64_t` at
roughly **+11% frontend time**. One word type, `uint32_t`, and E-9's own named cost —
kernels walking several levels needing two instantiations — buys nothing.

**A build-dominated pipeline would want the opposite**, and that is now a documented
operating point rather than an unexamined assumption.

**The footprint objection is dead either way:** +2.0% on the shipped ladder. X-10's
"+33%" was the 94×60 level in isolation and does not survive being weighed against the
levels above it.

**What would change the answer** is a `uint64_t` NEON path for `residualSums` — the
guards are a specialisation gap, not a property of the ISA, since aarch64 counts bits
in a 128-bit register regardless of how the caller sliced them. [E-29](#register), and
it is the same shape as the x86 gap: one kernel, one missing specialisation.

### D-46: the kernel set is sufficient for a VIO frontend — and detection is 39–57% of it

[X-56](EXPERIMENTS.md) built the loop [T4.3b](TASKS.md) split off and never ran:
`examples/vio_frontend.cpp`, modelled on HybVIO's frontend, with a **persistent track
set**, culling on `FAILED_FLOW` / `FLOW_OUT_OF_RANGE`, and **topping up by detection
with `applyMinDistance` against the survivors**. 1710 frames, no gap requiring an
operation binCV lacks. **T4.3b's sufficiency question is answered YES.**

**AND IT CORRECTS THE SCOPE OF [D-28](#8-design-decisions).** D-28 measured detection at
a **4.8% duty cycle** and concluded it was uninteresting — a figure that comes from
`frontend_sequence` re-detecting every N frames. A frontend that maintains a target
feature count detects far more often:

| | `frontend_sequence` | **top up below target** | **60% hysteresis** |
|---|---|---|---|
| detections | 4.8% of frames | **91.0%** | 45.0% |
| **detect** | 0.570 ms | **18.070** | 7.831 |
| track (LK) | 7.270 | 12.062 | 10.658 |
| **binCV total** | 10.644 | **31.641** | **19.996** |

**Detection is 39–57% of this frontend, against D-28's 4.8%.** D-28 is not wrong — it
measured what it measured — but **its conclusion does not transfer**, and every
optimisation priority derived from that profile rests on a detection policy nobody had
written down.

**THE POLICY IS WORTH 1.58× AND IT IS NOT binCV's.** Moving the low-water mark from
100% to 60% of target takes binCV from **31.6 to 20.0 ms**, for 13% fewer live features
and a *slightly better* lifetime. **A caller tuning one number outside the library moves
the frontend more than any optimisation this project has landed.** Documentation, not a
kernel.

**A sizing rule the API's own flag caught twice.** The NMS pool peaks at **70 831
survivors on a 752×480 frame — 19.6% of all pixels** — because a binarized
min-eigenvalue map takes few distinct values and enormous numbers of pixels tie. Size it
from `candidatesRanked`, never from `maxCorners`: truncation happens *before* the
spacing filter, so what survives is the first found rather than the strongest.

Successor: [E-30](#register).

### D-47: binCV's x86 deficit was a missing instruction, not a missing vector path

[X-52](EXPERIMENTS.md) found LK to be the whole x86 deficit and proposed porting
D-33/X-40's NEON tap batching to AVX2. [X-57](EXPERIMENTS.md) cancels **that specific
port** — but **not** the broader case for x86 vector work, which an earlier draft of
this record wrongly folded into the same verdict.

**The NEON batching exists because aarch64 has no scalar popcount** — `CNT` is a vector
instruction, so every scalar count pays `fmov` in and out ([D-6](#8-design-decisions)).
**x86-64 has `POPCNT` as a scalar instruction**, so that trick answers a problem x86
does not have — *if the instruction is emitted*. **It was not:** baseline x86-64
predates SSE4.2, so `__builtin_popcountll` compiled to a software fallback, and the
shipped binary contained **zero `popcnt` instructions**.

| build | binCV | OpenCV | ratio |
|---|---|---|---|
| default (portable) | 12.92 ms | 3.43 | **0.27×** |
| **`-mpopcnt`** | **3.45** | 3.15 | **0.91×** |

**3.75× from one flag**, and from 3.8× slower than OpenCV to near parity — at 6.23×
less memory. The stage profile snaps to the aarch64 shape (track 67.3 / build 26.7 /
detect 6.0 against 68.3 / 26.3 / 5.4). **The library was never mis-shaped on x86; it
was mis-compiled.**

**`BINCV_X86_POPCNT` is ON by default:** binCV's x86 baseline is a POPCNT-capable CPU
(Nehalem 2008, Barcelona 2007). **Shipping a bit-counting library that counts bits in
software is a worse default than a 2008 minimum.** It can be turned off for pre-SSE4.2
targets at 3.75×, and the configure summary prints which side it is on.

**A lesson worth more than the flag.** X-52 predicted LK near 2.7 ms and the frontend
near 3.9 ms at parity, and flagged it as an extrapolation after three failures.
Measured: **2.307 ms and 3.43 ms.** The number was right and **the mechanism was
wrong** — a correct prediction is not evidence of a correct model, and had the AVX2 port
been written it would have "confirmed" the hypothesis while the real cause went
unfound.

**WHAT REMAINS TRUE AFTER THE FLAG: binCV HAS NO x86 VECTOR CODE.** At 3.429 ms
against OpenCV's 3.150 it is 9% short of parity, competing word-parallel scalar against
hand-tuned AVX2. **binCV processes `uint32_t` words on a machine with 256-bit
registers** — 32 bits per operation where AVX2 offers 256 — and the build stage is
0.915 ms of long contiguous plane loops, which is what a vector unit is for. NEON paths
exist; x86 paths do not. [E-32](#register).

Successors: [E-31](#register), [E-32](#register).

### D-48: the bit-plane layout is what stops the compiler vectorising binCV

[X-58](EXPERIMENTS.md) asked how much of x86's available vector win the compiler
already takes. **`-mavx2` buys 1–2%, inside the 6.8% run-to-run spread of the same
binary** — and the audit says why.

| kernel | `%ymm` instructions |
|---|---|
| `derivativeX/Y` | **63 — vectorised** |
| `pyrDownRoute`, `boxSum4`, `cornerMinEigenValRow` | **0** |
| `residualSums` | 2 |

GCC's own reason (`-fopt-info-vec-missed`): *"not vectorized: **multiple nested
loops**"*, *"**control flow in loop**"*.

**THAT IS THE REPRESENTATION.** A bit-sliced kernel's outer loop walks **words** — the
dimension a vector unit wants — and its **body is a nest over planes**. GCC will not
vectorise an outer loop whose body is a loop nest. The one kernel that is a plain word
loop, `derivative`, vectorises without being asked.

> **The KERNELS' loop order stops the compiler. The LAYOUT does not** — and an earlier
> draft of this record conflated the two, which would have sent the next experiment at
> the representation.

**THE LAYOUT IS IDEAL FOR SIMD AND IS NOT WHAT NEEDS CHANGING.** A plane row is
`ptr + y * stride` with **consecutive words**, so eight consecutive words are **32
contiguous bytes — one `_mm256_loadu_si256`**. Nothing about the storage resists
vectorisation. What resists it is that the *kernel* puts the word loop outside and a
plane nest inside, which is the one shape an auto-vectoriser refuses. **Restructuring
is an implementation change; the bytes on the heap do not move.**

**This is not a flag and not a drop-in intrinsic.** AVX2 requires **restructuring each
kernel to process 8 destination words at once**, with plane arrays held as vectors
rather than `WordType[N]` locals — a redesign per kernel, and **zero change to the
storage layout**. `-mavx2` is **not adopted**: it would raise the minimum CPU to Haswell
2013 and buy nothing measurable.

**AND THE SAME RESTRUCTURING IS THE CUDA MAPPING.** A GPU port wants the **word index as
the parallel dimension** — thread `t` handles word `t`, so consecutive threads read
consecutive addresses and the load **coalesces perfectly** — with the plane loop kept
in-thread and unrolled into registers. That is *exactly* the loop order AVX2 needs.
**Bit-planes are better for GPUs than the alternatives, not worse:** an interleaved
layout would make thread `t` read at `t * N`, which is strided and defeats coalescing.
So [E-33](#register) is groundwork for [ROADMAP](../ROADMAP.md) Phase 6's GPU backend
rather than a detour from it, and **the case for keeping D-2 is now three-sided**:
scalar, SIMD and GPU all want the same bytes in the same order.

**The upside once restructured is large and specific:** bit-sliced arithmetic is pure
`AND`/`XOR`/`OR`, which AVX2 does **256 bits at a time** against the current 32.
**Nothing about the maths resists vectorisation; only the loop order does.**
[E-33](#register).

**Where binCV stands on x86:** **3.70 ms against OpenCV's 3.70 — parity**, from POPCNT
alone, at 6.23× less memory. Beating OpenCV needs E-33.

### D-49: the mismatch is GRANULARITY, not layout — five ceilings say so

[X-60](EXPERIMENTS.md) built the AVX2 path [X-59](EXPERIMENTS.md)'s 7.9× ceiling
authorised, proved it **bit-exact** (728 windows, 0 differ), measured it at
**0.53× — 1.88× slower** — and reverted it.

**Two failures, and the first is a trap worth publishing.**
`__attribute__((target("avx2")))` **blocks inlining**: `slicedSignedSum` became **310
real calls per window** (confirmed by `objdump`: 20 call sites, a standalone symbol).
**That is exactly the mechanism [E-31](#register) proposes for runtime dispatch, now
measured as unusable for an inline hot function.** Rebuilt with compile-time `__AVX2__`
it inlines — and is **still 1.88× slower.**

> **The eight words are COMPUTED IN REGISTERS, not loaded from memory.** X-59's ceiling
> used a contiguous-array load. Here the vector must be **assembled** from eight scalars
> and **disassembled** through a store and eight reloads. **The pack and unpack cost
> more than the eight `POPCNT`s they replace.**

**FIVE CEILINGS HAVE NOW OVERSTATED, AND THAT IS THE RESULT:**

| | ceiling | delivered |
|---|---|---|
| X-33 | 3.42× | 1.24× |
| [D-37](#8-design-decisions) | 1.461× | 1.069× |
| [D-40](#8-design-decisions) | 1.638× | **0.885×** |
| [D-41](#8-design-decisions) | 1.638× | 1.445× |
| **X-60** | **7.9×** | **0.53×** |

**Every one was measured on bulk contiguous data and applied to a kernel that works on a
handful of register-resident words.** Not five unlucky estimates — **one structural
mismatch measured five times.** binCV's hot kernels are not array operations
(`residualSums` touches **one word per row** of a 31-row window) and a vector unit wants
arrays.

**This supersedes the part of [D-48](#d-48-the-bit-plane-layout-is-what-stops-the-compiler-vectorising-bincv)
that implicated the representation.** The layout is contiguous and ideal; the
**granularity** is what resists SIMD. Those are different problems and only the second
one is real.

**E-33 is narrowed, not closed.** `residualSums` — 67% of the x86 frontend — is refuted
at its current granularity. **`build` is not**: `pyrDown` and the derivatives are
genuine bulk passes over contiguous rows, which is the shape X-59's **adder** ceiling
(4.7×) was actually measured on, and `derivative` already auto-vectorises. **The
untested half is the half the ceiling applies to.**

### D-50: `residualSums` resists SIMD because of its ACCESS PATTERN

Three attempts, three measured refutations, three different reasons — and the third one
names the property:

| attempt | granularity | why it failed |
|---|---|---|
| [X-59](EXPERIMENTS.md) ceiling | bulk contiguous array | 7.9×, but **not the kernel's shape** |
| [X-60](EXPERIMENTS.md) | within one window row | values **register-resident**; pack/unpack > 8 `POPCNT`s |
| [X-61](EXPERIMENTS.md) | **across 8 keypoints** | words **scattered**; gather > the loads it replaces |

**`residualSums` reads scattered single words; SIMD needs contiguous runs.** Not the
bit-plane layout ([D-48](#d-48-the-bit-plane-layout-is-what-stops-the-compiler-vectorising-bincv)
corrected that), not granularity alone ([D-49](#d-49-the-mismatch-is-granularity-not-layout--five-ceilings-say-so)
refined it) — the **access pattern**. Making those words contiguous is what
[E-26](#register) priced at **+21% footprint** and declined.

**The arithmetic behind it.** binCV holds 31 pixels in a word where OpenCV's AVX2 holds
32 per lane-row — no advantage at that width — and then spends **~120 operations per
window row against OpenCV's ~18**: N² = 4 plane pairs × 5 taps ([D-20](#8-design-decisions))
× 2 components × 2 for sign-magnitude ([D-3](#8-design-decisions)). **8× packing ÷ 6.7×
op-count ≈ 1.2×**, which is where binCV sits on x86.

**WHAT THIS IS NOT.** It is **not** "binCV cannot beat OpenCV". binCV **is 1.53× faster
on aarch64** — the deployment target — where the same scattered access is a *win*,
because `CNT` is cheap there and OpenCV's NEON coverage is thinner than its AVX2. The
x86 result is a statement about one kernel on one non-target platform.

**And `build` is untested with a favourable prior:** `pyrDown` and the derivatives are
bulk contiguous passes — the shape X-59's 4.7× adder ceiling was actually measured on —
and `derivative` already auto-vectorises unaided. 27% of the x86 frontend, and the only
untried avenue.

### D-51: binCV wins at BITWISE work and loses at ARITHMETIC — LK is arithmetic

Three x86 vectorisation attempts failed ([D-50](#d-50-residualsums-resists-simd-because-of-its-access-pattern))
and the question they were chasing — *why is a 1-bit library not beating an 8-bit one?*
— has a simpler answer than any of them. **It is not the layout, not the data format
and not parallelism. It is the operation count.**

**THE BITWIDTHS BEING COMPARED**, because the question is meaningless without them:
OpenCV's `LKTrackerInvoker` holds its patch and derivatives in **`CV_16S` — 16-bit** —
so AVX2 gives it **16 pixels per operation**. binCV works at **N = 2 bits** in
`uint32_t` words: **32 pixels per operation**. **binCV's packing advantage in the
shipped build is 2×, not the 8× an AVX2-vs-AVX2 comparison would suggest** — because
binCV is not using AVX2 at all.

**PER 31×31 WINDOW PER LK ITERATION:**

| | ops per window |
|---|---|
| binCV: 120/row × 31 rows | **3 720** |
| OpenCV: interpolate patch (360) + residual (300) | **660** |
| | **5.6×** |

*(An earlier draft said 12×, having omitted OpenCV's bilinear interpolation pass.
Corrected: 5.6× on operations, and **2.8× on cost** once binCV's 2× packing advantage
is applied — which is the order of the gap actually measured.)*

Three multipliers, each a recorded decision rather than an accident:

| | why |
|---|---|
| **×5** taps | **Bits cannot be interpolated.** OpenCV interpolates the displaced patch ONCE and takes one product; binCV must correlate the four displaced patches separately and combine them afterwards ([D-20](#8-design-decisions)). |
| **×4** plane pairs | A bit-sliced N-bit × N-bit product needs **N² popcount pairs**. At N = 2 that is 4, where OpenCV does **one multiply**. |
| **×2** | Sign-magnitude: `total − 2·opposing` is two popcounts ([D-3](#8-design-decisions)). |

**LOW BITWIDTH DOES HELP THE ARITHMETIC — ENORMOUSLY — AND STILL DOES NOT WIN.** A
bit-sliced N×N product costs **N² popcount pairs**: 1 at N=1, **4 at N=2**, 64 at N=8,
**256 at N=16**. binCV at 2 bits is **64× cheaper than a bit-sliced 16-bit would be**,
which is the whole reason the representation is viable. **But a hardware multiplier
costs ONE operation at ANY width** — the silicon is fixed-width, so narrowing the data
buys OpenCV **nothing** and buys binCV N². binCV narrows its way back to *4 pairs
against 1 multiply-add*, then pays ×5 for the taps on top.

**That is the whole account:** 2× packing against 5.6× operations is **2.8×**, and
2.8× is the gap.

**THE GENERAL SHAPE OF THE LIBRARY, STATED PLAINLY:**

> **binCV is enormously faster at operations that are natively BITWISE, and slower at
> operations that are natively ARITHMETIC.** `pyrDownBox` is **4.5–6× faster than
> `cv::pyrDown`** ([X-46](EXPERIMENTS.md)); denoise, morphology and thresholding are the
> same shape. **LK's residual is a multiply-accumulate**, and bit-slicing multiplies
> badly — N² popcounts where silicon has a multiplier.

The frontend nets to **1.53× faster on aarch64** because the bitwise stages and the
6.23× footprint carry the arithmetic one, and because OpenCV's NEON coverage is thinner
than its AVX2. On x86, where OpenCV's arithmetic is at its strongest, the two cancel.

**THE ALTERNATIVE TRACKER WAS ALREADY MEASURED AND IS NOT A WAY OUT.** Hamming block
matching is the natively-bitwise tracker — one popcount per candidate row, no N², no
taps — and [D-24](#8-design-decisions) measured it at **0.93× on track time** while
losing **2–12 yield points**. Being algorithmically aligned with the representation did
not make it faster, which is worth knowing before anyone proposes it again.

**AND THE ×5 IS THE ONE THAT CAN GO.** Collapsing the four tap correlations into one
takes binCV toward **~1 800 ops** once the weighted patch is formed — the collapse is
**not free**, it costs bit-sliced adds and a requantise — which against OpenCV's 660
with a 2× packing advantage is roughly **1.4× cost instead of 2.8×**. **A ~2× cut on
the kernel that is 67% of the frontend**, and the only one of the three multipliers
that is a design choice rather than a property of bit-sliced arithmetic.

**WHAT HAS NEVER BEEN QUESTIONED IS THE ×5.** It is the largest multiplier, it sits in
the kernel that is **67% of the frontend**, and D-20 records it as a consequence of the
design rather than as a measured choice. [E-34](#register).

### D-52: the 31-pixel window caps the packing advantage at 1.94×

[X-63](EXPERIMENTS.md) tested the packing argument cleanly — `uint64_t` on x86, where
no NEON guard confounds it — and **track moved 2% for a doubled word.**

> **A 31-pixel window occupies ONE `uint32` word.** Widening the word does not let
> `residualSums` do more per operation; it only wastes more bits.

| word / register | bits | utilisation at a 31-px window |
|---|---|---|
| **`uint32` (shipped)** | 32 | **97%** |
| `uint64` | 64 | 48% |
| **AVX2** | 256 | **12%** |

**THIS IS THE COMMON CAUSE BEHIND TWO REFUTATIONS —** [X-58](EXPERIMENTS.md) (compiler
AVX2) and [X-60](EXPERIMENTS.md) (hand-written within a row). Both tried to fill a
256-bit register **from a 31-pixel window**, and both had a proximate reason (nested
loops, pack/unpack); this is the one underneath them.

**IT DOES NOT COVER [X-61](EXPERIMENTS.md), AND AN EARLIER VERSION OF THIS RECORD
CLAIMED IT DID.** X-61 batched **eight keypoints** into the lanes, so the window's
width is irrelevant to it — the register is full by construction. Its blocker was the
**gather**, and its own entry records that **the vector arithmetic won** (≈48 vector
ops against ≈100 scalar) and the gathers gave it back. That is a **data-movement**
failure, not a packing one, and lumping it in here made a fixable engineering problem
look like an information-theoretic wall. See [D-55](#d-55-the-vector-dimension-is-keypoints-and-the-blocker-is-staging-not-packing).

**binCV's real rate in LK is 31 px/op** — not 32, 64 or 256 — against OpenCV's 16.
**A 1.94× packing advantage, capped by the ALGORITHM rather than by the word type or
the ISA**, against 5.6× the operations: **2.9× cost, which is the gap, fully accounted.**

**What this rules out:** widening *within a window* — wider words and wider registers
fed from one 31-pixel row. **It does NOT rule out batching across windows**, where the
register is full by construction; [X-61](EXPERIMENTS.md) measured **one implementation**
of that and was blocked by its gather, not by packing. The levers that remain are the
**operation count** ([E-34](#register)'s ×5, closed negative by
[X-62](EXPERIMENTS.md)) and the **access pattern** ([E-36](#register)).

**One thing it does not settle:** `uint64` made `build` **worse on x86 (1.2×)** where
[X-54](EXPERIMENTS.md) measured it **1.66× better on aarch64**. Build is a bulk pass, so
the window cap does not apply — that it reverses by platform is unexplained, and it
weakens the prior on [E-33](#register)'s "restructure `build`" avenue.

### D-53: a ceiling prices the operation, not the algorithm

[X-62](EXPERIMENTS.md) measured the tap collapse at **1.75× in the kernel** on the
reference device — the op-count model of [D-51](#d-51-bincv-wins-at-bitwise-work-and-loses-at-arithmetic--lk-is-arithmetic) was
correct, and the runtime-weight dispatch was priced rather than assumed. On the
**frontend** the same change was **3.3× slower**.

| `maxIterations` | 1 | 2 | 3 | 5 | 10 | **20 (shipped)** |
|---|---|---|---|---|---|---|
| B vs A, reference device | 1.06× | 1.06× | 0.96× | 0.77× | 0.51× | **0.31×** |

**The mechanism is in the shape of the curves.** A's cost flattens as the cap rises,
because its points converge and stop. B's climbs linearly, because its points never
converge — **rounding the interpolated patch back into the pixels' own N-bit alphabet
destroys exactly the signal LK converges on.** The per-iteration saving is real and is
spent several times over on iterations that would not otherwise have happened.

> **A CEILING BOUNDS THE COST OF ONE OPERATION. IT SAYS NOTHING ABOUT HOW MANY TIMES
> THE ALGORITHM WILL PERFORM IT.** When a change alters accuracy, it can alter the
> iteration count, and then the per-operation ratio is not the ratio that matters.

This is a **different and worse failure than the five in [D-49](#d-49-ceilings-in-this-project-have-overstated)**.
Those mispriced the operation, and a more careful ceiling catches them. This one
priced the operation correctly; no amount of care *inside* the ceiling would have
caught it.

**The rule this adds, for every experiment that changes what a kernel COMPUTES rather
than only how fast it computes it:** the pre-registered bands must include a
**whole-workload** arm, not only a kernel ratio and an accuracy figure. X-62's four
bands were each a statement about a per-iteration ratio, and **none of them could
express the result** — Band D is phrased "B is SLOWER", which B was not.

**E-34 is closed negative.** The arm is reverted (recoverable at commit `4351bd6`);
`benchmark/tapcollapse_ceiling.cpp` stays and reproduces the 1.75×. No `LKParams`
option ships: at `maxIterations ≤ 2` the gain is inside run-to-run scatter and at the
shipped 20 it is a large loss, so there is no regime to offer — unlike
[D-24](#8-design-decisions)'s route (a) or [D-36](#8-design-decisions)'s filter set,
which ship precisely because each has one.

### D-54: the x86 deficit was threads; at equal cores binCV leads

[X-64](EXPERIMENTS.md) controlled a variable the x86 runs had left free.
`benchmark/frontend_sequence` set `cv::setNumThreads` only when
`BINCV_OPENCV_THREADS` was set, and it was not — so those runs compared
**single-threaded binCV against twelve-threaded OpenCV**, and the resulting ratio was
read as a **SIMD** deficit.

| OpenCV threads | ratio, three runs |
|---|---|
| **1** | **1.19× / 1.14× / 1.19× — binCV faster** |
| 12 | 0.65× / 0.65× / 0.64× |

**~~binCV leads OpenCV on x86 at equal core count~~ — WITHDRAWN AS CLAIMED, THEN EARNED.**
Those runs used `MH_01_easy`, off a transient mount. Re-measured on **V1_02**, the
sequence [X-38](EXPERIMENTS.md), X-49 and X-52 all used, binCV was **0.89× at one
thread** — *behind* single-threaded OpenCV. It is now **1.04×**, from
[X-69](EXPERIMENTS.md) and [X-70](EXPERIMENTS.md), which is a different thing entirely:
**a lead measured on the harder sequence and produced by two changes, not by a choice of
input.** The withdrawal stands as written; what replaced it was earned. V1_02 is harder (`track` 2.28 ms
against MH_01's 1.66), and **the ratio moves further than most experiments here
measure.**

**What survives is the reason this record exists:** OpenCV gains **1.68–1.77×** from
threads on both sequences, so an uncontrolled comparison measures parallelism and reads
as implementation. The one-thread default, the disclaimer and the corrected NOTE all
stand. **What does not survive is the lead**, which was a property of one sequence
stated as a property of the library.

**The reference device was never affected**, which is why this survived so long:
`run_on_pi.sh` runs under `taskset -c 3` and OpenCV's threads cannot escape one pinned
core. Every recorded entry that states a thread count states one. **The x86 runs
drifted; the device could not.** `frontend_sequence` now defaults to one thread and
prints a loud disclaimer at any other count.

**This does not reopen [E-32](#register) or overturn
[D-52](#d-52-the-31-pixel-window-caps-the-packing-advantage-at-194).** X-58/X-60/X-61
measured vector width; none of that was threads and none of it changes. **What changes
is which question is open**: not vector width, but that **binCV is single-threaded and
OpenCV is not** — an axis this project has never examined, and worth more than the 12%
register utilisation D-52 ruled out. Registered as [E-35](#register).

**Not decided here:** whether multi-core OpenCV is the fairer denominator for a
multi-core target. A frontend on a phone has more than one core, and binCV using one is
a real limitation rather than a measurement artefact — a question about the product's
shape, left open rather than settled by a correction.

### D-55: the vector dimension is KEYPOINTS, and the blocker is staging, not packing

Three x86 vector attempts failed and the verdict was recorded as "`residualSums` is
closed for x86 vectorisation" ([D-50](#8-design-decisions)). **That verdict is too
broad, and [D-52](#d-52-the-31-pixel-window-caps-the-packing-advantage-at-194)'s
original wording made it worse** by attributing all three to the 31-pixel window.
Re-reading the measurements:

| attempt | granularity | what it actually shows |
|---|---|---|
| [X-58](EXPERIMENTS.md) | compiler, within a row | the loop nest is not vectorisable — real, and about **shape** |
| [X-60](EXPERIMENTS.md) | hand-written, within a row | 31 px cannot fill 256 bits — real, and about **packing** |
| **[X-61](EXPERIMENTS.md)** | **8 keypoints in 8 lanes** | **the arithmetic WON, ≈48 vector ops against ≈100 scalar.** The five `_mm256_i32gather_epi32` per row (~75 cycles) gave it back |

**X-61 is a data-movement result, not a packing result.** Eight keypoints fill the
register by construction; nothing about a 31-pixel row constrains it. So the question
is not *can the arithmetic be vectorised* — it was, and it won — but **why the operands
were gathered at all.**

**THEY WERE GATHERED BECAUSE X-61 GATHERED IN THE INNER LOOP, ONCE PER ROW PER
ITERATION. Most of what it gathered does not change across iterations.** Of the twelve
words `alignedResidualSums` reads per row, **eight are iteration-invariant** — `self`,
`magX`, `magY`, `signX`, `signY` all belong to the **previous** frame and its
derivative, which LK linearises about and never re-reads at a new offset. Only the tap
words move, and their *integer* part changes only when the estimate crosses a pixel
boundary. **LK runs up to twenty iterations per keypoint per level**, so a gather paid
once and reused twenty times is a twentieth of the cost X-61 measured.

**THIS IS NOT [E-26](#register) AGAIN.** E-26 priced converting **a whole level** to
interleaved layout: +92 160 B on a 436 704 B peak, **+21%**, declined. Staging **eight
windows' invariant words** is 8 × 31 × 8 × 4 B ≈ **8 KB, +1.8%** — and it is a
transient tile, not a second copy of the pyramid. **The two differ by an order of
magnitude and the earlier decision does not settle the smaller one.**

**AND THE POPCOUNT ITSELF SHOULD PROBABLY NOT BE A POPCOUNT.** AVX2 has no vector
popcount (this class of core has no `AVX512-VPOPCNTDQ`), so a vectorised reduction must
emulate it per word — which is why X-60's pack/unpack lost. But the kernel does not
need each row's count; it needs **the sum of 31 rows' counts**. That is exactly what a
**carry-save adder tree** computes: compress 31 one-bit-per-lane values into a 5-plane
bit-sliced sum with `AND`/`XOR` only, then popcount **five** words with weights
1,2,4,8,16 instead of thirty-one. **binCV already has the primitives** — `maj3` and
`addShifted` in [ops/bitslice.hpp](bincv-cpp/include/bincv-cpp/ops/bitslice.hpp) are a
full adder — and CSA is pure boolean work, which is precisely what AVX2 does well and
what binCV's representation is made of.

**Decision: `residualSums` is REOPENED for x86, at the keypoint granularity only, and
the next attempt must STAGE rather than gather.** [D-50](#8-design-decisions) stands
for the two within-row granularities and is narrowed to them. Registered as
[E-36](#register). **The ceiling must be a whole-frontend arm**
([D-53](#d-53-a-ceiling-prices-the-operation-not-the-algorithm)), and
[D-49](#8-design-decisions) says to expect it to overstate.

### D-56: binCV is single-threaded by default, not by decision

There is **no record anywhere in this repository of choosing a single-threaded design.**
It is an unexamined default, and [X-64](EXPERIMENTS.md) shows what it costs: OpenCV
gains **1.68×** from threads on the same workload, which was misread as a SIMD deficit
for a working session.

**The measurement protocol is part of why it went unexamined.** `run_on_pi.sh` pins to
one core with `taskset -c 3` — correct for isolating a kernel, and it makes a
parallelism question *invisible* rather than answered. A protocol that hides an axis is
not neutral about it.

**The obstacles are smaller than they look, and the memory tiebreak mostly does not
bite.** LK over keypoints has **no shared state**: pyramids and derivative ladders are
**read-only** and shared, kernels take views ([D-5](#8-design-decisions)) and allocate
nothing ([CLAUDE.md](CLAUDE.md)), and `residualSums`' scratch is a handful of
stack words per window. **The 436 704 B working set does not multiply with thread
count** — only the per-thread stack does. `build` is a dependency chain *down* the
ladder but its rows within a level are independent, so it parallelises over row bands
with a barrier per level.

**Decision: registered as [E-35](#register), and the tiebreak is stated in advance** —
if a threading arm does multiply the shared working set, [CLAUDE.md](CLAUDE.md)'s rule
applies and memory wins.

**THE API SHAPE IS DECIDED PROVISIONALLY AND AHEAD OF THE NUMBERS, WHICH CLAUDE.md
REQUIRES BE SAID OUT LOUD.** An earlier draft of this record proposed a **serial**
default on the grounds that a library at this level should not own the caller's thread
policy — with the reference implementation as evidence, since HybVIO runs
single-worker pools per stage and takes its parallelism at the pipeline level. That is
right *for that integrator* and wrong as a default, for a reason this project already
has in writing: **OpenCV ships parallel by default, so a serial default means every
casual comparison is single-threaded binCV against multi-threaded OpenCV** — the exact
trap [X-64](EXPERIMENTS.md) documents, in this repository's own benchmark, for most of
a working session. **A default that makes the library lose its own benchmark is not
neutral.**

**The default follows the BUILD PROFILE, which is the only place it can be both honest
and fast:** hosted builds parallel and sized to hardware concurrency; **core-only,
`-fno-exceptions` and freestanding stay serial with no pool and no allocation.** That
is not a compromise but the only shape available — `bincv_core` is allocation-free and
builds without exceptions, where `std::thread` is not usable, so the pool must live
outside core whatever the policy is. Surface: `setNumThreads(n)` with `1` serialising,
a swappable backend, and a `parallelFor` hook for an integrator's existing pool —
OpenCV's surface, so integrators find what they expect, and HybVIO's model is one
`setNumThreads(1)` call away.

**Determinism is not a casualty:** keypoints are independent and `build`'s row bands
write disjoint memory, so a threaded arm is **bit-exact** against serial, and X-65
makes that a precondition rather than a band.

**Provisional until [X-65](EXPERIMENTS.md) measures it.** If its Band B or D fires, the
shape ships serial by default however well the argument reads.

### D-57: threading is the largest lever measured, and it moves the bottleneck to `build`

[X-65](EXPERIMENTS.md) split the point array across threads — **no library change**,
because `calcOpticalFlowPyrLK` already takes an array and the pyramids are read-only —
and checked bit-exactness against serial on every frame before timing anything.

| threads | `track` speedup | `build` share | binCV vs **1-thread** OpenCV |
|---|---|---|---|
| 1 | 1.00× | 26.4% | 0.90× |
| **4** | **2.60×** | 44.4% | **1.50×** |
| 12 | 3.71× | **52.2%** | 1.75× |

**Peak RSS is flat — 29 848 / 29 828 / 29 844 KB at T = 1 / 4 / 12, a 0.07% spread.**
[D-56](#d-56-bincv-is-single-threaded-by-default-not-by-decision)'s claim that only
per-thread stack scales is confirmed against Band A's 5% with three orders of magnitude
of margin, so **the memory tiebreak never engages** and the provisional API shape
survives its gate: **hosted builds default parallel.**

**TWO THINGS THIS DOES NOT SAY.** Arm C — threading `build` — was **never built**, so
Band A's numeric thresholds are met by arm B alone and the band's own wording is not
fully satisfied. And **at equal thread counts the result is PARITY, not a lead**: binCV
and OpenCV are both 1.674 ms at twelve threads. The 1.50–1.75× figures are against
**one-thread** OpenCV and must be labelled that way every time.

> **`build` DOES NOT SCALE AND IS NOW THE BOTTLENECK** — 26.4% of the frontend at one
> thread, **52.2% at twelve**, and flat in absolute terms (0.89 → 0.86 ms). Threading
> `track` moved the constraint rather than removing it.

**~~which makes [E-33](#register) the highest-value target left~~ — WRONG, and
[X-67](EXPERIMENTS.md) corrects it.** That inference treated `build` as one thing.
Decomposed, **`pyrDown` is 3.6% of the frontend and `fromCVMat` is 20.4%**; E-33 is
close to the *lowest*-value target on x86, not the highest. See
[D-59](#d-59-eighty-percent-of-build-is-the-input-conversion-not-a-kernel).

### D-58: state the sequence, or the headline is not about the library

**Two headlines in two days rested on an uncontrolled variable.**
[X-64](EXPERIMENTS.md) compared single-threaded binCV against twelve-threaded OpenCV
and read the gap as SIMD. Its *correction* then claimed a lead that existed only on
`MH_01_easy` and vanished on **V1_02** — 1.14–1.19× became **0.89×**.

**The two failures have the same shape**: a variable that changes the answer by more
than the effect being measured, left unstated because it had always been the same.
Threads were constant because the reference device is pinned; the sequence was constant
because it lived at one path.

**The rule, in the same form as the one X-64 already produced for threads:** *every
whole-frontend ratio in [EXPERIMENTS.md](EXPERIMENTS.md) names its sequence, and the
canonical one is V1_02* (`~/bincv-data/euroc-v1_02-cam0`) — what X-38, X-49 and X-52
used. A number from any other sequence is reported **as a comparison**, never as the
headline. `MH_01_easy` is measurably the easier sequence — `track` 1.66 ms against 2.28 —
and is not a substitute.

### D-59: eighty percent of `build` is the input conversion, not a kernel

[X-67](EXPERIMENTS.md) decomposed the stage [D-57](#d-57-threading-is-the-largest-lever-measured-and-it-moves-the-bottleneck-to-build)
had just called the bottleneck:

| | T=1 | T=4 |
|---|---|---|
| `fromCVMat` — the `CV_8U` → bit-plane conversion | **20.4%** | **34.4%** |
| `pyrDown` | **3.6%** | **6.1%** |
| derivatives | 1.2% | 2.1% |

**An infinite speedup on `pyrDown` is worth 1.037× at one thread.**
[E-33](#register) is demoted — not closed, since the reference device's profile
differs and [X-30](EXPERIMENTS.md) put build at 25.8% there, but on x86 it is not a
lever.

**The conversion is a property of the HARNESS's input, not of binCV's pipeline.** It
exists because the reference preprocessing emits an OpenCV `Mat`; a binary-frame
frontend receives bits from its sensor stage and never performs it. OpenCV pays nothing
equivalent. **So the frontend ratio is reported BOTH ways and labelled** — 0.94×
as-measured at T=1, 1.18× with the conversion excluded — because reporting only the
second is self-serving and only the first understates by a fifth.
[D-58](#d-58-state-the-sequence-or-the-headline-is-not-about-the-library) is the same
rule about a different variable, and this is its second application in a day.

**A CAUTION THIS RECORD IS ITSELF EVIDENCE FOR.** D-57's wrong inference took six hours
to catch and was made *from a correct measurement* — `build` really is 52% at twelve
threads. **A stage total is not a target.** Decompose before choosing what to optimise;
[D-53](#d-53-a-ceiling-prices-the-operation-not-the-algorithm) said a ceiling can price
the wrong thing, and this says a *profile* can too.

### D-60: the previous frame's words are staged once per point per level

[X-66](EXPERIMENTS.md) found **contiguity, not the popcount**, was the whole of its
2.09×. If the win is addressing it should be available **with no vector code**, and
[X-69](EXPERIMENTS.md) is that arm: **1.27× on `track`, 1.09% on the frontend,
bit-exact, 2 048 B of stack and zero heap.**

Eight of the twelve words read per row belong to the **previous** frame, which LK
linearises about and never re-reads at a new offset, and `region` is fixed per point per
level — so they are extracted **once** and all
[X-68](EXPERIMENTS.md)'s **4.29 mean iterations** read from the buffer.
[D-37](#8-design-decisions) put extraction at 45.4% of the kernel; the model predicted
**1.30×** and measured **1.27×**.

`stageWindow` **declines** — wider than a word, or taller than 64 rows — rather than
overrunning a fixed stack buffer, so the unstaged path stays live and correct.
[CLAUDE.md](CLAUDE.md) forbids a kernel allocating and this operation has no caller
scratch, which is why the bound exists at all.

**Bit-exactness is pinned by `Flow.StagedMatchesUnstaged_{N1,N2,N3}`, watched to fail**
on an injected one-bit staging fault (374/487/522 of 624 windows differ).

**This works on aarch64 too**, where [E-36](#register)'s AVX2 batch never would.

**~~It also shifts that experiment's baseline~~ — IT DOES NOT, and
[X-75](EXPERIMENTS.md) checked.** X-66's arm A reads a **pre-extracted, per-keypoint
contiguous buffer** through the same `slicedSignedSum` this ships, so the baseline was
already current: **2.06–2.14× re-measured on the shipped tree.**

### D-61: the taps are cached on the integer displacement, and binCV crosses 1.0

[X-69](EXPERIMENTS.md) staged the eight iteration-invariant words and left the four
taps, because they move. [X-70](EXPERIMENTS.md) observes that **they move as
`floor(offX)` while the iteration is shrinking `off`** — so the integer part settles
long before `maxIterations`, and the same words are re-extracted for nothing. Caching
them on `(tapX, tapY)` is sound by construction: the taps are a pure function of
`lv.next`, `region` and that key.

| arm | `track` | frontend | vs 1-thread OpenCV |
|---|---|---|---|
| unstaged (before) | 2.168 | 3.151 | 0.905× |
| staged ([D-60](#d-60-the-previous-frames-words-are-staged-once-per-point-per-level)) | 1.746 | 2.923 | 0.965× |
| **+ tap cache** | **1.486** | **2.651** | **1.06×** |

**Cumulative: `track` 1.46×, the frontend 1.19×, and the ratio crosses 1.0.**
[X-64](EXPERIMENTS.md) claimed a lead and had to withdraw it because the claim came from
an easier sequence; **this is V1_02, and it was earned by two changes rather than by a
choice of input.**

**The cost is stack, and it is not free at high N.** `TapCache` is 4 × N × 64 words on
top of `StagedWindow`'s 8 × N × 64: **4 KB at the shipped `N = 2`, ≈15 KB at the `N = 8`
ceiling.** That is a lot for a Cortex-M and is stated rather than hidden — the shipped
ladder is `1/2/2/2`, so the real figure is 4 KB. Both decline above 64 rows rather than
overrunning, and neither touches the heap ([CLAUDE.md](CLAUDE.md)). A **byte**-bounded
cap rather than a row-bounded one is [E-38](#register), not a guess made here.

### D-62: where x86 stands, consolidated

Six experiments in one session moved the x86 picture, and the pieces are scattered
across [X-64](EXPERIMENTS.md) … [X-70](EXPERIMENTS.md). **This is the whole of it**, on
**V1_02**, full 1710 frames, **OpenCV pinned to one thread throughout**
([D-58](#d-58-state-the-sequence-or-the-headline-is-not-about-the-library)):

| threads | `track` | `fromCVMat` | binCV | vs OpenCV | *(was, this morning)* |
|---|---|---|---|---|---|
| **1** | 1.340 | 0.830 (33%) | 2.499 | **1.04×** | 0.90× |
| 2 | 0.856 | 0.820 | 2.000 | 1.33× | 1.20× |
| **4** | 0.542 | 0.822 (49%) | 1.689 | **1.56×** | 1.50× |
| 12 | 0.392 | 0.818 (**53%**) | 1.539 | 1.71× | 1.75× |

**`track` went 2.19 → 0.39 ms — 5.5×** — 1.46× from staging and the tap cache
([D-60](#d-60-the-previous-frames-words-are-staged-once-per-point-per-level),
[D-61](#d-61-the-taps-are-cached-on-the-integer-displacement-and-bincv-crosses-10))
and 3.7× from threading ([D-57](#d-57-threading-is-the-largest-lever-measured-and-it-moves-the-bottleneck-to-build)).
**All of it bit-exact**, all of it at unchanged peak working set.

**THE BOTTLENECK HAS MOVED TWICE IN ONE NIGHT AND IS NOW THE INPUT CONVERSION.**
`fromCVMat` is **53% of the frontend at twelve threads** and does not scale — it was 20%
at the start. It is also the one item on the list that **a deployed binary-frame
pipeline would not run at all**
([D-59](#d-59-eighty-percent-of-build-is-the-input-conversion-not-a-kernel)): excluding
it, T=4 is **0.867 ms against OpenCV's 2.638 — 3.04×.**

**Both numbers, always.** 1.56× is what this harness measures; 3.04× is what a pipeline
fed by its sensor would see. Neither is the honest number alone.

**What this does NOT show.** T=12 got slightly *worse* (1.75× → 1.71×) because `track`
is now small enough that threading has less to bite on — Amdahl arriving on schedule.

**AND NONE OF IT REACHES aarch64 YET, WHICH IS A DELIBERATE HOLD.** `residualSums`
dispatches `N` ∈ {1,2} at `uint32_t` — **the shipped ladder's whole depth range** — to
[D-33](#8-design-decisions)'s and [X-40](EXPERIMENTS.md)'s NEON accumulators, and the
staged path has neither. Taking it there would swap a **measured** optimisation for an
**unmeasured** one, so `stageWindow` **declines on aarch64 at those depths**: zero
regression risk on the deployment target, and zero gain there either. The staged NEON
variants are [E-39](#register), and they wait for a device window rather than being
written blind.

### D-63: the input conversion is a move-mask, and the portable arm alone is 10×

[X-71](EXPERIMENTS.md) replaced a per-pixel branch with a bit-plane extraction.

| arm | vs shipped | needs |
|---|---|---|
| **portable branchless** | **10.3×** | nothing |
| x86 `movemask` | **46.1×** | AVX2, at run time |
| aarch64 NEON bitmask | *unmeasured* | NEON (baseline on aarch64) |

**`fromCVMat` is 15.5× on the real workload and falls from 32% of the frontend to 3%;
the frontend gains 1.43×.** At four threads **binCV is 3.06× single-threaded OpenCV at
6.23× less memory** — and [X-67](EXPERIMENTS.md)'s "conversion excluded" projection of
3.04× now matches the measurement to two decimals.

**The portable arm is the one that matters most:** ten times faster with **no intrinsics
at all**, so every target gets it — including one with no vector unit, where it is the
whole path.

**AVX2 ships without an `-mavx2` build.** `movemask32` is `target("avx2")` behind a
cached `__builtin_cpu_supports`, leaving the baseline at the SSE4.2-era floor
[D-47](#8-design-decisions) set. This is [X-66](EXPERIMENTS.md)'s rule from X-60's
failure — **mark one coarse entry point, never leaf helpers** — and here the marked
function *is* the unit of work, so nothing that mattered was blocked from inlining.

**Why the orders line up:** `bitMask(x)` is `1 << (x % WordBits)` and
`_mm256_movemask_epi8` returns byte `i` in bit `i`, both LSB-first. **The two
conventions are the same one**, so no shuffle is needed anywhere.

**A bug `uint64_t` caught and nothing narrower could have.** The vector loop first
stopped at a multiple of 32; at 64-bit words that is *half* a word, so the portable
tail wrote its bits at the wrong offsets. Fixed by consuming whole **words**, with the
precondition now a `BINCV_ASSERT`. [CLAUDE.md](CLAUDE.md)'s four-word-type sweep is
load-bearing and this is what it is for.

**aarch64, measured on the reference device:** portable **5.18×**, NEON **14.00×**,
both bit-exact. On the frontend `fromCVMat` is **7.9×** and falls from **21% to 3.2%**,
the frontend gains **1.22×**, and **the ratio against OpenCV goes 1.53× → 1.85× on the
deployment target** — above [X-38](EXPERIMENTS.md)'s recorded 1.46×. **Band A on both
architectures.**

The standalone arms read 14.0× (aarch64) and 46× (x86) where the frontend reads 7.9×
and 15.5×: the gap is the allocation and the `cv::Mat` row-pointer work *around* the
packing, and **it is the same ratio on both machines**, which makes it an explanation
rather than an excuse.

### D-64: threading ships, and the headline is the equal-thread number

[X-73](EXPERIMENTS.md) measured the shipped API on both architectures.

| | binCV | OpenCV | ratio |
|---|---|---|---|
| **reference device, 1 thread each** | 9.772 | 19.023 | **1.95×** |
| **reference device, 4 threads each** | 3.633 | 7.065 | **1.94×** |
| **x86_64, 4 threads each** | 1.082 | 1.462 | **1.35×** |

**The ratio is the same at one thread and at four**, because both sides scale. Threading
moves the absolute numbers and leaves the advantage where it was — which means the
advantage is the *implementation*.

**So the shipped figures are the equal-thread ones.** binCV at four threads against
OpenCV at one reads 4.72× on the device, and that number is true and is what a caller
sees who leaves OpenCV at its default — but quoting it bare would mix a parallelism
difference into what reads as an implementation one, which is exactly the error
[D-58](#d-58-state-the-sequence-or-the-headline-is-not-about-the-library) was written
about.

**`track` scales 3.69× on the device's four cores against 2.50× on x86** — a smaller
cache and a simpler core, on a workload that was already memory-light.

**The API shape's gate passes.** [D-56](#d-56-bincv-is-single-threaded-by-default-not-by-decision)
recorded it as provisional pending X-65's bands; neither Band B (footprint) nor Band D
(<1.5×) fired, so **hosted builds default parallel and the pool stays outside
`bincv_core`** — which was never a preference: core is allocation-free and builds
`-fno-exceptions`, and `std::thread` is usable under neither.

### D-65: one row reader, and the deployment target reaches 2.48×

[X-74](EXPERIMENTS.md) finished what [X-72](EXPERIMENTS.md) reverted.
[X-41](EXPERIMENTS.md) counted **three copies** of the window-row extraction block and
recommended collapsing them for maintenance; staging and tap-caching made it worth
doing for speed, because they had to reach the NEON kernels too and a separate copy
each would have made five. **There is now one `RowReader`**, serving scalar and NEON,
staged and unstaged, with X-34's `+1`-tap-is-a-shift and X-35's interior fast path in
it once.

| reference device, pinned | `track` | frontend | vs OpenCV |
|---|---|---|---|
| staging held off | 7.61 ms | 9.00 | 1.85× |
| **staged NEON** | **5.37** | **6.77** | **2.47×** |

**x86 pays nothing** — 1.50 ms against 1.54, inside the spread.

**Two shapes are load-bearing and both were measured, not guessed**
([X-72](EXPERIMENTS.md)): `Staged` must be a **template** parameter, because a runtime
test per row costs 17% of `track` on x86; and the operands must **alias** rather than
copy, because copying gives back what staging bought. Neither alone reached parity.

**[D-60](#d-60-the-previous-frames-words-are-staged-once-per-point-per-level)'s hold is
lifted.** It declined staging on aarch64 across the shipped ladder's whole depth range
because the staged path lacked [D-33](#8-design-decisions)'s tap batching and
[X-40](EXPERIMENTS.md)'s accumulator. It has both now.

**And the tooling lesson stands on its own:** X-72 abandoned working code because
compiling for aarch64 looked expensive. It costs **2.5 seconds**
(`scripts/check_arm_syntax.sh`). A third of `ops/opticalFlow.hpp` is invisible to every
x86 build, and a project that cannot cheaply compile that third will keep making this
mistake.

### D-66: eight keypoints per AVX2 register, and the refill is the design

[X-79](EXPERIMENTS.md) shipped [E-36](#9-open-questions-and-planned-experiments)'s
keypoint batch. `track` **1.37×**, the frontend **1.29×**, and the headline against a
one-thread OpenCV **2.20× → 2.81×** — at four threads each side, **1.35× → 1.67×**.
Bit-exact: 2 208 points, **0 positions differ**, watched to fail on two injected faults.

**THE LAYOUT IS THE RESULT.** `[row][plane][lane]` puts eight keypoints' words at the
same row and plane in eight adjacent `uint32_t`, so a vector load fetches one word from
each of eight keypoints. [X-61](EXPERIMENTS.md) lost this exact fight with vector
arithmetic that WON on operation count and **gathers** that gave it back, and
[D-52](#d-52-the-31-pixel-window-caps-the-packing-advantage-at-194) filed that as a
packing wall it was not. The fix was never a better gather; it was arranging not to need
one.

**AND THE REFILL IS PART OF THE DESIGN, NOT AN OPTIMISATION ON TOP.**
[X-78](EXPERIMENTS.md) counted the iteration distribution *before* the batch was
written: **72.6% of point-levels finish in two iterations or fewer and a 3.6% tail runs
the cap**, so eight lanes in lockstep would have wasted **39.9% of every lane slot** and
turned a 3.1× kernel into 1.31×. A converged lane takes the next untracked point
instead. **Pricing the waste first cost one afternoon's instrumentation and changed what
got built.**

**Two corrections this leaves on the record**, both reported rather than absorbed:
[X-68](EXPERIMENTS.md)'s **4.29 mean iterations is high — the direct count is 3.235** —
and its **91.5% for iterated `residualSums` counts the tap extraction**, which the batch
does not replace. Inverting Amdahl on the measured frontend puts the vectorised
arithmetic at **~44% of `track`**. Both errors point the same way, because both come
from a timing decomposition attributing fixed per-point cost to the loop.

**What is deliberately NOT claimed:** this is x86-only, runtime-dispatched on
`__builtin_cpu_supports("avx2")`, and aarch64 is untouched. A window wider than a word
or taller than 32 rows is **tracked by `trackOnePoint`, not refused** — the 32-row cap
is a footprint choice (~20 KB of `[row][plane][lane]` arrays at `N = 2` against ~41 KB
at 64 rows, and the working set has to stay in L1).

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
| ~~**E-3**~~ **RESOLVED** | Three questions about the same interface: (a) at what window size does incremental/sliding popcount beat recomputation for the LK covariance? (b) does a fused covariance entry point beat composing it from three T2.6 calls? (c) frame-sized selector plane versus a four-argument `countAndSplit` — memory against speed. | The 31×31 window is recomputed per keypoint; windows overlap heavily; and §7.5's covariance needs four numbers that today cost three traversals. | **Answered, and all three moved off the simpler shape: (a) 7.3×–20× for the adopted sliding form → expose incremental state; (b) 1.27–1.29× → add a covariance entry point; (c) plane 16–18% faster but a fifth plane at every level → four-argument form, memory wins.** Re-measured against the shipped code by [X-11b](EXPERIMENTS.md) — 5.96×/15.9×, 1.20–1.27×, 11–14% — with no branch changing. [D-15](#d-15-window-reductions-get-incremental-state-and-a-fused-covariance), [X-11](EXPERIMENTS.md) | T2.6, T3.6 | Phase 2 (T2.10) ✔ |
| ~~**E-4**~~ **RESOLVED** | Does bit-sliced generic-N ever regress the specialized N=1 and ternary paths? | The promise is arbitrary N at no cost to the common cases. | **Answered: no. At N = 1 the generic route and the specialization produce a derivative of the SAME SIZE to the byte (2264 B) and the SAME instruction count (567), time to within 0.1%, and generic-N's whole object is 90 B smaller — they are not literally the same instruction stream (GCC allocates different registers), and the equality holds for the derivative only; the covariance and count differ by 40 B and 24 B and time inside the batch spread. `N` STAYS ARBITRARY; no cap.** [D-21](#d-21-generic-n-is-not-capped-and-the-n--1-specialization-is-kept-as-a-test-oracle), [X-21](EXPERIMENTS.md) · **This is an N = 1 result and closes only the N = 1 question** — every arm was `QuantMat<1>`/`TernaryMat`, and it says nothing about N = 3 or N = 5, where §7.5's covariance contributes plane PAIRS and is quadratic in N where the derivative is linear. **The larger cost X-21 found while answering this is [E-12](#9-open-questions-and-planned-experiments), not this row** — both binCV routes sit 8–43% in time and 2.63× in code size (`-fno-exceptions`; 2.84× with exceptions on) above a hand-written binary-only control, and the decomposition puts most of it in genericity that is NOT in N. | Whether N is capped rather than arbitrary. | T1.5 specialization strategy | Phase 3 (T3.9) ✔ |
| ~~**E-8**~~ **RESOLVED** | Horizontal decimation for `pyrDown` ([§6.1](#61-bit-parallel-primitives)): a per-pixel gather loop, or a log2(width) word-parallel unshuffle that needs frame-sized constant masks? | The pyramid's subsample half has no primitive, and the two routes sit on opposite sides of the project's speed/footprint tiebreak — masks measured in frames against a loop measured in ns/px. | **Answered, and the question was leading: there is no tiebreak. A third route the register did not list — the WORD-LOCAL unshuffle — is word-parallel and costs zero bytes, and beat the gather loop by 14.6×/26.4× and the frame-masked route by 11.3×/8.3×. `ops/` gains a word-local resample primitive taking `(src, dst)`; no plan, no scratch.** [D-17](#d-17-horizontal-decimation-is-word-local), [X-14](EXPERIMENTS.md) | T3.4 | Phase 3 (T3.4) ✔ |
| ~~**E-12**~~ **RESOLVED — NOT WORTH REMOVING** | How much of the `ops/` kernel's **per-row** cost is genericity that is **not** in N? | [X-21](EXPERIMENTS.md) measured T3.5's derivative at **+15% per word and +93% per row** against a hand-written binary-only control, and only +19.8 of the 93 points were separated. | **Answered by measuring what it is worth, which is almost nothing.** [X-30](EXPERIMENTS.md) profiled the frontend on the reference device: the derivative sits inside a build stage worth **0.7%**, so **eliminating the entire stage caps the frontend gain at 1.0062×**. X-21's +93% stands as measured and is **not worth removing**. The half of this row's registration that did bite — "every `ops/` kernel with a per-row prologue" — is answered instead: **`cornerMinEigenVal`'s response sweep (52.7%) and `residualSums` (43.7%) are 96.5% of the frontend**, they are the same kernel shape, and they are Phase 5.1's whole target list ([D-27](#d-27-phase-51-vectorizes-two-functions-and-they-are-the-same-kernel-shape)). | Whether `ops/` kernels get a compile-time border form or a monomorphic word-width path. **Neither — the stage is 0.7%.** | — | **Phase 4** (X-30) ✔ |
| ~~**E-13**~~ **RESOLVED — IT IS AN N = 1 RESULT** | Does the **per-row partial accumulator** still pay above N = 1? | D-15 item 4 was measured at N = 1, where `BitSlicedPairCounts` is four counters; at N = 4 it is sixty-four. | **Answered: it pays at N = 1 and costs above it.** [X-29](EXPERIMENTS.md) on the reference device: window-wide vs per-row is **0.917× / 1.114× / 1.348× / 1.248×** at N = 1/2/3/4, so the crossover is between 1 and 2. `gradientCovariance<N>` selects with `if constexpr` (free — N is already a template parameter) and results are **bit-identical** by construction. It lands on the adopted `1/2/2/2` ladder, where three of four levels run at N = 2. **The noise floor was MEASURED**: the same arm in two translation units spreads **0.0–0.3% on the Cortex-A72** and **up to 10.6% on x86_64** — larger than the whole effect at N = 2, which reads IN NOISE there and W wins on the device. X-22 declined to close on a single-binary A/B and was right to. [D-26](#d-26-the-covariance-accumulator-shape-is-chosen-on-n-and-the-noise-floor-is-measured). | Whether the N-bit covariance keeps per-row partials. **Only at N = 1.** | — | **Phase 4** (X-29) ✔ |
| ~~**E-14**~~ **RESOLVED — NO** | Does the tracker need a border on its coarse pyramid levels — the reference's `winSize`-wide reflected pad, a cheaper replicate pad, or a keypoint policy that never places a window near a coarse-level edge? | X-24 read the clipped coarse window as the dominant term. | **Answered: no, and the question rested on a statistic that does not fit the data.** [X-25](EXPERIMENTS.md) measured yield — eligible keypoints tracked within 1.0 px — instead of RMS, and **a padded pyramid is worse than or equal to clipping in five of seven cases for 1.38× the bytes**. **Deviation (ii) is vindicated and is now MEASURED rather than argued.** The finding underneath it corrects X-24: arm A on `(1,0)` has `rms(all)` 0.8356 px but **98.6% yield at 0.0009 px** — 139 of 141 keypoints tracked to a thousandth of a pixel and two catastrophically wrong — so **clipping costs about two keypoints out of 141, not the 59% X-24 attributed to it**; the never-clipping subset simply excluded the outliers. What remains is the **level-0 1-bit floor** (`rms(usable)` 0.25–0.32 px in EVERY arm including the padded one, against X-20's own no-pyramid single-level 0.2860 px), which is [E-16](#register) and is not a pyramid parameter at all. | Whether pyramid levels gain a border. **They do not.** | — | **Phase 4** (X-25) ✔ |
| **E-15** | Why does tracking accuracy PEAK AT 2 BITS and degrade with more? | [X-24](EXPERIMENTS.md) measured it and did not explain it: on `(1,0)` unclipped, 1/2/3/5 bits give 1.4742 / **0.0010** / 0.5334 / 0.8567 px, and on `(2,−3)` the ladder is exact at 2 and 3 bits and fails from 4. Two explanations remain open and were NOT separated. **(i)** The bit-sliced covariance and residual weight plane pairs by `2^(i+j)`, so a few high-magnitude pixels dominate a window whose sub-pixel accuracy comes from averaging many edge crossings — this predicts degradation at SMALL motion and none at large, which is the observed pattern (`(12,−8)` is exact at every depth, `(2,−3)` fails from 4 bits, `(1,0)` from 3). **(ii)** `1/2/2/2`'s upper levels collapse to two distinct values anyway, so its advantage may be density preservation rather than precision. `requantizeBoxSum` is already excluded as the cause: it is a faithful rescaled average at every depth. | Whether the weighting in [§7.5](#75-lk-gradient-covariance)'s bit-sliced form is right for TRACKING as opposed to for corner response, and whether a depth cap belongs in the pyramid API. | E-7's final ladder | **Phase 4** |
| ~~**E-34**~~ **RESOLVED — NEGATIVE, and the ceiling was the wrong measurement** | The five-tap decomposition is a **×5 multiplier on 67% of the frontend**. Can the four bilinear correlations become one? | [D-51](#d-51-bincv-wins-at-bitwise-work-and-loses-at-arithmetic--lk-is-arithmetic) accounts for binCV's LK issuing **5.6×** OpenCV's operations, and **×5 is the largest of the three factors**. [D-20](#8-design-decisions) records it as a consequence of bits not being interpolable, not as a choice weighed against options. | **Answered: YES it can, at 1.75× in the kernel — and it is 3.3× SLOWER on the frontend.** [X-62](EXPERIMENTS.md) built the ceiling (runtime weights priced, not assumed), measured the accuracy cost at **+0.112 px rms with yield UNCHANGED**, and then measured the frontend, where the two curves separate: A's cost flattens as `maxIterations` rises because its points **converge and stop**; B's climbs linearly because **rounding the interpolated patch into the pixels' own N-bit alphabet destroys exactly the signal LK converges on**. The per-iteration saving is spent several times over on iterations that would not otherwise happen. **The arm is reverted** (recoverable at `4351bd6`); no `LKParams` option ships, because there is no regime where it wins. [D-53](#d-53-a-ceiling-prices-the-operation-not-the-algorithm). | Whether binCV's tracker is fast because of the representation or in spite of it. **Because of it — the interpolation is where the sub-pixel information lives.** | D-51 | **Phase 5** (X-62) ✔ |
| **E-31** | Should binCV get runtime POPCNT dispatch on x86, or simply require the instruction? | [X-57](EXPERIMENTS.md) measured the software fallback costing **3.75×**; `BINCV_X86_POPCNT` now defaults ON, so the question is whether the minimum can be removed. | **[X-60](EXPERIMENTS.md) measured the obvious mechanism and it does not work.** `__attribute__((target(...)))` **blocks inlining**, turning an inline hot function into 310 calls per window and costing 1.9×. So dispatch for `popcountWord` cannot be per-call; it would have to be per-**kernel**, chosen once at a coarse boundary — or the 2008-era minimum simply kept, which is what several vision libraries do. | Whether binCV's portable build is portable at a 3.75× price. | D-47 | **Phase 5** |
| ~~**E-32**~~ **RESOLVED — the obstruction is the loop order** | binCV has NEON paths and no x86 vector code. Where would AVX2 pay? | [X-57](EXPERIMENTS.md) left binCV at 0.91× of OpenCV with no x86 vector code at all. | **Answered: not from the compiler.** [X-58](EXPERIMENTS.md): `-mavx2` buys **1–2%**, inside the same binary's run-to-run spread. `derivative` vectorises (63 `%ymm`); `pyrDownRoute`, `boxSum4` and `cornerMinEigenValRow` get **zero**, because a bit-sliced kernel's outer loop walks words while its **body is a nest over planes** — GCC: *"multiple nested loops"*. [D-48](#d-48-the-bit-plane-layout-is-what-stops-the-compiler-vectorising-bincv). | Whether x86 needs hand-written kernels. **It does**, and the reason is structural. | D-47 | (X-58) ✔ |
| **E-33** *(narrowed to `build`)* | Restructure the **bulk** kernels — `pyrDown`, the derivatives — to process 8 words at once so AVX2 can reach them. | [X-59](EXPERIMENTS.md) priced the adder at **≈4.7×** on contiguous data. [X-60](EXPERIMENTS.md) then **refuted the `residualSums` half**: bit-exact and **1.88× slower**, because its eight words are register-resident and the pack/unpack costs more than the `POPCNT`s. | **`build` is the half the ceiling actually applies to** — bulk passes over contiguous plane rows, the shape X-59 measured — and `derivative` already auto-vectorises, so the target is `pyrDown`. It is **27% of the x86 frontend, not 67%**, so the prize is correspondingly smaller. [D-49](#d-49-the-mismatch-is-granularity-not-layout--five-ceilings-say-so) says to expect a ceiling to overstate. | Whether binCV is ahead of OpenCV on x86, and whether the GPU port inherits the restructuring. | D-49 | **Phase 5** |
| **E-35** | binCV is **single-threaded**; OpenCV is not. Should the frontend parallelise — across keypoints in LK, across tiles in `build` — and what does that cost in footprint? | [X-64](EXPERIMENTS.md) found the x86 "deficit" was **threads, not vector width**: at one thread binCV **leads 1.14–1.19×**, at twelve it reads 0.65×. OpenCV gains **1.68×** from parallelism. [D-52](#d-52-the-31-pixel-window-caps-the-packing-advantage-at-194) closed every *widening* avenue, so this is the axis that is left — and it is worth more than the 12% register utilisation D-52 ruled out. | **Unmeasured, and not obviously free.** LK over keypoints is embarrassingly parallel and needs no shared state; `build` is a dependency chain down the ladder. Against that, **CLAUDE.md's tiebreak is memory** — per-thread scratch multiplies the footprint that is binCV's strongest claim (6.23×), and a Cortex-A little core is not an x86 core. **Rule and ceiling first**, and the ceiling must be a WHOLE-FRONTEND arm, not a kernel ratio ([D-53](#d-53-a-ceiling-prices-the-operation-not-the-algorithm)). | Whether binCV's single-threaded shape is a deliberate constraint or an unexamined default — and whether multi-core OpenCV is the fairer denominator, which X-64 deliberately left open. | D-54 | **Phase 5** |
| ~~**E-35**~~ **RESOLVED — BAND A, and the bottleneck moved** | binCV is single-threaded; OpenCV is not. Should the frontend parallelise, and what does it cost in footprint? | [X-64](EXPERIMENTS.md) found the x86 "deficit" was threads. | **Answered: yes, and it is the largest lever this project has measured.** [X-65](EXPERIMENTS.md) split the point array across threads with **no library change** — `calcOpticalFlowPyrLK` already takes an array and the pyramids are read-only — **bit-exact on every one of 300 frames**. `track` scales **2.60× at T=4**, the frontend goes **0.90× → 1.50×** against one-thread OpenCV, and **peak RSS is FLAT (0.07% across T=1..12)**, so the memory tiebreak never engages. **Two limits: arm C (threading `build`) was never built, and at EQUAL thread counts the result is PARITY (1.674 ms each), not a lead.** `build` does not scale and is now **52.2%** of the frontend at T=12, which hands the next move to [E-33](#register). [D-57](#d-57-threading-is-the-largest-lever-measured-and-it-moves-the-bottleneck-to-build). | Whether binCV's single-threaded shape was a constraint or an unexamined default. **Unexamined — and hosted builds now default parallel.** | D-54, D-56 | **Phase 5** (X-65) ✔ |
| ~~**E-41**~~ **RESOLVED** | Vectorise `detectFast` and `computeBrief` on both architectures. | [X-76](EXPERIMENTS.md) first measured them **3.9× and 1.4× SLOWER** than OpenCV -- implausible for operations that are pure comparison. | **Answered: both were bugs.** `computeBrief` 7.3× faster with **no intrinsics** (two multiplies per pair inside a 256-iteration loop, hoisted); `detectFast` vectorised on AVX2 **and** NEON, with a compass reject both vector paths were missing. Final: FAST **1.06× on x86, 0.96× on the device**; describe **5.2× / 10.7×**; matching **4.7× / 1.95×**. Correctness held through four rewrites -- 1818/1818 against `cv::FAST` throughout. | Whether the descriptor family is competitive or merely correct. **Competitive.** | X-76 | **Phase 5** (X-76) ✔ |
| **E-38** | `StagedWindow` and `TapCache` cap at **64 rows**, so their stack cost scales with `N`: 4 KB at `N = 2`, **≈15 KB at `N = 8`**. Should the cap be on BYTES instead? | [X-70](EXPERIMENTS.md) shipped both and stated the figure rather than hiding it. A row cap is the wrong axis — it fixes the shape, not the footprint, and footprint is the constraint [CLAUDE.md](CLAUDE.md) says wins ties. | **A byte cap would decline the shipped 31-row window at `N = 8`** (≈28 rows fit in 4 KB), sending it to the unstaged path — a correctness-preserving but silent slowdown at high `N`. The shipped ladder is `1/2/2/2` so nothing today is affected, which is exactly why this is registered rather than guessed: **there is no measurement of what high-`N` callers exist.** | Whether the staging path is usable on a Cortex-M at `N > 2`. | D-61 | **Phase 5** |
| ~~**E-39**~~ **RESOLVED — SHIPPED** | Write the staged NEON variants so aarch64 gets [X-69](EXPERIMENTS.md)/[X-70](EXPERIMENTS.md)'s staging. | `residualSums` dispatches N ∈ {1,2} at `uint32_t` to [D-33](#8-design-decisions)'s and [X-40](EXPERIMENTS.md)'s NEON accumulators, and the staged path has neither, so [D-60](#d-60-the-previous-frames-words-are-staged-once-per-point-per-level) holds staging OFF there. | **[X-72](EXPERIMENTS.md) measured it: 2.13× on the reference device, up from 1.85×** — `track` 1.18×, frontend 1.14×, 298/298 checks including the staged oracle. **The code is NOT committed**: it is a structural edit to functions that compile only on aarch64, and this environment has no cross-compiler, no working Docker for `verify_arm.sh`, and a 5–10 minute device round-trip. Three repair attempts did not clear brace-level damage x86 cannot see. **The next attempt needs, in order:** a working `verify_arm.sh`; the reader written once and whole rather than spliced; and — measured, not guessed — **`Staged` as a template parameter AND operands as aliasing pointers**, since a runtime flag costs 17% of x86 `track` and copying the operands gives back what staging bought. Neither alone sufficed. | Whether the deployment target gets the 1.15× that is now known to be there. | D-60, D-61 | **Phase 5** |
| **E-40** | `fromCVMat` is **33% of the x86 frontend at one thread and 53% at twelve**, and does not scale. Optimise it on **both** architectures. | [X-67](EXPERIMENTS.md) found it at 20.4%; [X-69](EXPERIMENTS.md)/[X-70](EXPERIMENTS.md) shrank `track` around it until it became the largest single item. The 1-bit path is a **per-pixel branch and read-modify-write**. | **Bit-plane extraction from bytes is what a move-mask instruction does.** `_mm256_movemask_epi8` returns one plane of 32 pixels in one instruction; aarch64 has no movemask but AND-with-bit-weights plus pairwise-add gives 16 in about six. A portable **branchless** arm is measured too, because if it gets most of the win the intrinsics are not worth their maintenance. | Whether binCV's interop boundary is a real cost or an artefact of an unoptimised loop — and, via the allocation arm, whether the packing was ever the problem. | D-59, D-62 | **Phase 5** |
| **E-42** | `seal_params.yaml` caps LK at **20 iterations**, and [X-78](EXPERIMENTS.md) measured that **3.6% of point-levels run the whole cap and converge to nothing — burning 22% of every iteration the tracker executes.** Should the cap come down? | X-78 counted iterations directly for the first time: **72.6% of point-levels finish in two or fewer**, and the distribution is bimodal with a tail pegged at the cap. The tracker spends a fifth of its time on points it is about to drop. | **It is an ACCURACY change, not a free one**, which is why it is registered rather than taken. A point at iteration 15 has not converged but its estimate is not therefore worthless, and `status` does not currently distinguish "converged" from "ran out". The arms are the cap itself (20 / 10 / 5) measured against **track lifetime and endpoint error**, not against time alone — [D-53](#8-design-decisions) applies, and a cap that speeds up `track` by shortening tracks has not helped anyone. | Whether a fifth of `track` is being spent on points that were already lost. | D-58 | **Phase 5** |
| **E-37** | `loadLevel0` converts **both** frames every frame and `build()` builds **both** pyramids — but this frame's `next` is next frame's `prev`. Should the frontend ping-pong? | [X-67](EXPERIMENTS.md) measured `fromCVMat` at **20.4% of the frontend at T=1 and 34.4% at T=4**, and roughly half of it is recomputation. | **Not fixed where it was found, on purpose.** Changing the harness changes every recorded frontend number in [EXPERIMENTS.md](EXPERIMENTS.md) — that is a decision about the denominator, not a cleanup, and it needs its own rule and a restated baseline. Note OpenCV's `calcOpticalFlowPyrLK` also rebuilds both pyramids per call, so the redundancy may be **symmetric** and removing it only on binCV's side would flatter binCV. **Measure both before changing either.** | Whether ~10% of the x86 frontend is recomputation, and whether the denominator moves with it. | D-59 | **Phase 5** |
| ~~**E-36**~~ **RESOLVED — SHIPPED** | Re-attempt x86 vectorisation of `residualSums` at the **keypoint** granularity, **staging instead of gathering**, and replacing the per-row popcount with a **carry-save adder tree**. | [X-61](EXPERIMENTS.md)'s vector arithmetic WON (≈48 ops against ≈100) and its **gathers** gave it back — a data-movement result that [D-52](#d-52-the-31-pixel-window-caps-the-packing-advantage-at-194) wrongly filed as a packing wall. | **Two changes, neither of which is a port.** (1) **Stage once, reuse across iterations**: eight of the twelve words read per row belong to the PREVIOUS frame and never move, and LK runs up to 20 iterations — so the gather is paid once, not 20 times. ~8 KB, **+1.8%**, an order of magnitude below [E-26](#register)'s declined +21%. (2) **CSA instead of popcount**: AVX2 has no vector popcount, but the kernel needs the SUM of 31 rows' counts, and a carry-save tree compresses 31 words into 5 planes with AND/XOR alone — `maj3` and `addShifted` already exist. | **Answered: it does not.** [X-79](EXPERIMENTS.md) shipped it — `track` **1.37×**, frontend **1.29×**, headline **2.20× → 2.81×** at one thread and **1.35× → 1.67×** at four, bit-exact over 2 208 points and watched to fail twice. The CSA half was NOT what did it (measured a wash earlier); the **layout** was. And [X-78](EXPERIMENTS.md) priced the lockstep waste at **39.9% of lane slots** before a line was written, which is why the shipped form refills a converged lane instead of idling it. | Whether [D-50](#8-design-decisions)'s "closed for x86" holds at the granularity that was never properly tried. **It does not.** | D-55, D-52, D-66 | **Phase 5** (X-79) ✔ |
| ~~**E-18**~~ **RESOLVED — NEGATIVE** | Can `residualSums` carry **vector accumulators across the window**, reducing once per window instead of once per call? | [X-33](EXPERIMENTS.md) measured a **3.42× ceiling** for batched NEON popcounts and delivered **1.24×**. The gap is the horizontal add: it runs once per `slicedSignedSum` call — **~310 register-domain crossings per window**. | **Answered: YES it can, and NO it is not worth 2–3×.** [X-40](EXPERIMENTS.md) built it (`impl::alignedResidualSumsNeon2`, bit-exact, gate-enforced) and it delivers **1.069×** against a 1.461× ceiling — about **1.52× against OpenCV** on the frontend, from 1.46×. **The floor arm is the finding**: the per-row tap machinery with the counting REMOVED is **45.4%** of the kernel, so **if counting were free the cap would be 2.205×**. The 2–3× this question was chartered on is not in the counting. [D-37](#d-37-residualsums-is-extraction-bound-not-count-bound); successor is [E-23](#register). | Whether `TapSums` becomes vector state. **It did, and the profile moved instead.** | X-28's unmet criterion 4 | **Phase 5** (X-40) ✔ |
| ~~**E-23**~~ **RESOLVED — NEGATIVE** | `residualSums` is extraction-bound: 45.4% of the kernel is addressing with zero counting. How much of it is addressable? | [X-40](EXPERIMENTS.md) measured it with a floor arm; it was 13.7% at [D-29](#8-design-decisions) and grew because D-30, D-31, D-33 and X-35 made the counting ~3× faster and never touched it. | **Answered: almost none of it, by either obvious route.** [X-41](EXPERIMENTS.md) hoisted every loop-invariant — both `(w0, s)` descriptors, their branches, the `.row(y)` multiplies, the `interior` test — for **1.023×**; and fitting all ten planes in L1D together for **1.129×**. The 8× cache-line overfetch is real and is **not** the constraint. **The instruction stream is**: ~118 cycles per row for ~100 instructions. [D-38](#d-38-residualsums-extraction-is-instruction-bound--not-addressing-not-layout). | Whether the three copies of the extraction block collapse. **They should, but for maintenance — not for speed.** | D-37 | **Phase 5** (X-41) ✔ |
| ~~**E-24**~~ **RESOLVED — NEGATIVE** | The twelve `alignedWord` extractions in a row share two `(w0, s)` descriptors. Can twelve scalar load-shift-ors become three vector ones? | [X-41](EXPERIMENTS.md) ruled out addressing (1.023×) and cache (1.129×), leaving instruction count as the only lever. | **Answered: the shifts YES, the loads NO.** [X-43](EXPERIMENTS.md): removing the gather makes the extraction **1.638×** faster, but paying for it makes it **0.885× — slower than scalar**. `QuantMat` stacks planes, so the eight words are in eight unrelated lines and **aarch64 has no gather**; eight loads plus eight lane inserts cost more than the shift-ors they replace. **The obstacle is the layout, and the rule predicted that before measuring.** [D-40](#d-40-the-extractions-obstacle-is-the-plane-layout-and-residualsums-is-done). | Whether the aligned path vectorises its loads. **It cannot, as laid out.** | D-38 | **Phase 5** (X-43) ✔ |
| ~~**E-26**~~ **RESOLVED — NO** | Should the tracker convert a level to interleaved layout per frame — **+8% frontend speed for +21% peak footprint**? | [X-44](EXPERIMENTS.md) measured both sides: extraction **1.445×**, `residualSums` **1.167×**, net frontend **~1.65×** from 1.52×; cost **+92 160 B on a 436 704 B peak**, criterion 3 **6.23× → 5.15×**. | **Answered: NO — the trade is declined.** binCV does not spend 21% of its footprint advantage on 8% of speed. The measurements stand as the record of what was on offer. Interleaving as a general layout was already ruled out separately by the 5.17× streaming cost ([D-41](#d-41-interleaving-will-not-be-bincvs-general-layout-the-rest-is-escalated)). | **Settled by the project's goals, not by a measurement** — which is why X-44 escalated it rather than picking a band. `residualSums` is now closed at every level: counting, addressing, cache, vectorisation and layout have each been priced and each declined or exhausted. | D-41 | **CLOSED** |
| ~~**E-19**~~ **RESOLVED — the shipped point STANDS** | Is the `1/2/2/2` ladder still the right operating point? | [X-50](EXPERIMENTS.md) swept ladder × filter on three axes and concluded `1/2/2/1` + `BOX_3x3` dominated. [X-51](EXPERIMENTS.md) ran the frontend confirmation X-50 required and **refuted it**: 90.6% within 1 px against 95.4%, lifetime 9 vs 11, and slower. | **Answered: YES, `1/2/2/2` + `BOX_2x2` stands** — and every coarse level's second bit earns its place by MORE than the proxy could see. X-50's accuracy harness builds the pyramid in float and so understates the cost of removing bits by ~6.7×. Speed and footprint from X-50 are sound; the accuracy proxy is not. [D-43 withdrawn](#d-43-withdrawn-the-operating-point-stays-1222--box_2x2). | Whether the shipped ladder changes. **It does not.** | D-23 | **Phase 5** (X-50, X-51) ✔ |
| ~~**E-27**~~ **RESOLVED — fix shipped, purpose NOT achieved** | Should the accuracy harness build its levels with binCV's own `pyrDownFiltered` cascade instead of in floating point? | [X-51](EXPERIMENTS.md) blamed the float cascade for the harness mispricing level 3's bit by 6.7×. | **Answered: YES it should, and it was — but that was not the cause.** [X-53](EXPERIMENTS.md): the corrected harness prices the bit at **−0.42** points where the frontend measures **−4.60**, and its corrections run in BOTH directions. X-51's mechanism is withdrawn. The filter axis moved ≤0.16 points, vindicating X-51's hedge and letting D-36/D-39's filter figures stand as first-hand. [D-44](#d-44-the-accuracy-harness-measures-a-different-question-from-the-frontend). | Whether the harness can be trusted for shipped defaults. **It cannot, and the rule tightened.** | X-51 | **Phase 5** (X-53) ✔ |
| **E-28** | The synthetic-warp harness and the frontend disagree by **4.2 yield points** on the same configuration, and it is **not** the float cascade. What is it? | [X-53](EXPERIMENTS.md) eliminated the leading hypothesis. The remaining structural difference: the harness warps ONE frame, so `prev` and `next` are binarizations of the same image with near-identical edge maps; the frontend tracks REAL consecutive frames, whose binarizations differ near the threshold, over a sequence where error compounds and tracks are re-detected. | **The tension may be irreducible**: the harness uses synthetic warps BECAUSE it needs ground truth, which is what makes it unrepresentative. A candidate resolution is a harness on consecutive real frames with OpenCV's flow as the reference rather than a known warp — trading exact ground truth for representativeness, which is a different bargain rather than a strictly better one. | Whether binCV can price an accuracy trade without a full frontend run. | D-44 | **Phase 5** |
| ~~**E-22**~~ **RESOLVED** | How much of `pyrDownFilteredRoute`'s cost is genericity rather than filter? | [X-39](EXPERIMENTS.md) measured the generic route running `BOX_2x2` at **2.96×** the hand-written one **computing the same function**, so that tax rides on every filter in the set. | **Answered: nearly all of it, and it was never necessary.** [X-42](EXPERIMENTS.md) made three helper signatures take their already-`constexpr` values as template parameters instead of runtime arguments — **no algorithm change** — and the generic route went **2.96× → 1.19×**, `GAUSSIAN_5x5` **4.28× faster**. **This reverses D-36:** the standard-LK anchor now costs +1.20 ms and leaves binCV **1.32× FASTER** than OpenCV, where D-36 recorded 0.97× — slower. [D-39](#d-39-the-filter-frameworks-3-tax-was-genericity-and-d-36-is-restated). | Whether D-36's filter prices are real. **They were not.** | D-36 | **Phase 5** (X-42) ✔ |
| **E-25** | The hand-written `pyrDown` is now only **1.19×** faster than the generic route computing the same function. Should it be deleted, leaving one implementation for all six filters? | [X-42](EXPERIMENTS.md) closed the gap from 2.96×. Two implementations of `BOX_2x2` is a standing correctness liability that `tests/test_pyramid.cpp` currently pays for by holding them to agreement. | **Not a free call.** The hand-written route is what **every prior result in this project was measured on**, including D-35's criterion-4 numbers, so deleting it re-bases the whole speed record by 1.19% — small, but it must be re-measured rather than assumed. Against that: one implementation, one place for the next optimisation, and the three structural costs (serial accumulation, materialised intermediate, worst-case widths) become worth attacking because they would then be on the shipped path. | Whether binCV ships one pyramid kernel or two. | D-39 | **Phase 5** |
| ~~**E-21**~~ **RESOLVED** | What does the downsampling-filter axis look like? | binCV implemented one of six variants, so every accuracy result sat at one point of a two-dimensional space. | **Answered, and the axes are NOT independent.** `BOX_2x2` saturates at 3 bits (+0.82 yield points N=2→7) where `GAUSSIAN_5x5` gains +3.93 — **the filter decides how much depth is useful**. Standard-LK accuracy is reachable and **costs criterion 4**: `GAUSSIAN_5x5` is 25.10× the shipped route and would put binCV behind OpenCV. `BOX_3x3` recovers **65% of the gap for +0.8 ms** and dominates `GAUSSIAN_3x3`. `DIRECT_SUBSAMPLE` is −19.68 points, confirming the paper's ">2.5 cm worse". [D-36](#d-36-box_2x2-stays-the-default-the-filter-set-ships-as-options). | The pyramid's default filter. **`BOX_2x2` stays; the set ships as options.** | E-19 | **Phase 5** (X-39) ✔ |
| ~~**E-20**~~ **RESOLVED — CRITERION 4 MET** | What is the WHOLE FRONTEND's speed against OpenCV on the reference device? | Every end-to-end reading had been taken on x86 where binCV runs scalar. | **Answered: binCV is 1.46× FASTER end to end** — 11.198 against 16.324 ms/frame over the FULL 1710-frame EuRoC sequence, OpenCV pinned to one thread — **while using 6.23× less memory**, with track lifetime one frame short of OpenCV's (11 vs 12) and per-frame survival 0.2 points short (96.4% vs 96.6%); the earlier EQUAL reading came from the easy 692-frame prefix, which still reproduces exactly. [X-38](EXPERIMENTS.md), [D-35](#d-35-all-four-roadmap-success-criteria-are-met-on-the-deployment-target). The profile also moved: build is now **25.8%** of the frontend, up from 4.5%, which is where [E-21](#register) lands. | Whether ROADMAP criterion 4 can be closed. **It is.** | — | **Phase 5** (X-38) ✔ |
| **E-17** | Where does the tracker lose the factor of **2.5–3** between the representation's 0.10 px floor and its own 0.25–0.29 px? | [X-27](EXPERIMENTS.md) closed off the representation, and X-24/X-25 closed off three pyramid parameters, so this is what is left of T3.8's MISS — and it is now a located problem rather than a diffuse one. **The prime suspect is deviation (i)**: the previous window is anchored on the integer grid because a bit-plane derivative cannot be interpolated, which displaces the aperture by up to half a pixel. `ops/opticalFlow.hpp` already calls that "the concrete thing route (b) trades away", and half a pixel of aperture displacement is the right order of magnitude for a 2.5× gap on a sub-pixel measurement. Two other candidates are untested: the single linearization per iteration on binary content, and the asymmetry that `I` is read at integer positions while `J` is interpolated. | Whether deviation (i) is a trade worth reversing, and at what cost — reversing it needs an interpolatable previous-frame gradient, which is a representation change. | T3.8's standing accuracy MISS | **Phase 4** |
| ~~**E-16**~~ **RESOLVED — THE REPRESENTATION IS NOT THE LIMIT** | Is X-20's 0.25 px RMS tolerance reachable at all with a 1-bit level 0 on real edge maps — and if not, what IS the representation's floor? | Three pyramid parameters had been eliminated without explaining T3.8's MISS, leaving the representation as the natural suspect. | **Answered: reachable, by a wide margin.** [X-27](EXPERIMENTS.md) measured a 31×31 window resolving **29.3 distinct binary states per pixel of displacement** — floor **0.025 px** noise-free, **0.10 px** at σ = 1 gray level, **0.174 px** even at σ = 4. **The 0.25 px criterion stands unchanged and no tolerance was widened.** X-20's "four independent crossings" was wrong but **conservative by ~7×**. Band D fired: from 11×11 to 41×41 set pixels grow 7.3× while distinct states grow only 1.8×, so the crossings lie on connected contours, are not independent, and **a bigger window buys almost no localisation** — window sizing cannot be justified by averaging. The remaining factor of 2.5–3 is the TRACKER's and is [E-17](#register). | Whether T3.8's tolerance is achievable as stated (**yes**), and whether level 0's depth is a variable (**it need not be**). | — | **Phase 4** (X-27) ✔ |
| ~~**E-7**~~ **ANSWERED — BUT THE QUESTION PRESUPPOSED THE WRONG CAUSE** | How many bits does each pyramid level actually need to preserve tracking accuracy? | The reference never chose its depths — they fell out of using `CV_8U`. [X-20](EXPERIMENTS.md) promoted this from an optimisation to a precondition on the reading that a 1-bit level cannot localise sub-pixel motion. | **[X-24](EXPERIMENTS.md) measured it and the premise does not survive. BAND C: no ladder up to `1/3/5/7` brings the four-level tracker inside X-20's tolerance on the full 141-point set — a deeper alphabet is NOT what fixes T3.8's miss.** Three things were learned instead, and two of them contradict this row's old text. **(1) Accuracy is PEAKED AT 2 BITS, not monotone in depth** — on `(1,0)` over the points that never clip, 1 bit gives 1.4742 px, **2 bits gives 0.0010 px (1474×)**, and 5 bits gives back 0.8567 px. Band D's own check excludes the artifact: every ladder tracked every point (141/141, 58/58), so the rows share identical point sets and `minEigThreshold` rejection is not involved. **(2) The dominant residual is CLIPPING, not quantisation** — the same ladder goes 0.8356 → **0.0010** px when restricted to the **58 of 141 points (41%)** whose 31×31 window is inside every level, so deviation (ii) is essentially all of that ladder's error rather than X-20's estimated half. That is now **[E-14](#register)**. **(3) 1-bit coarse levels ARE broken, so X-20 was half right — but not for its stated reason**: down a 1-bit ladder the edge map is THINNED AWAY (level 3 keeps 154 set pixels of 5 640 against `1/2/2/2`'s 1 028), so the second bit buys **content survival**, not precision. Why more than two bits then HURTS is **[E-15](#register)** and is not settled here. `1/2/2/2` is the leader on all three axes and is **cheap**: measured on the reference device it costs **1.35× the shipped ladder's tracking time, 1.51× its build and 1.17× its bytes** — against a pre-written cost model that predicted 3.25×, so the `20N²` popcounts per word do not dominate a tracked frame. The pre-registered "more than 2× is a headline" condition does not fire. **[X-25](EXPERIMENTS.md) then unblocked the adoption by answering E-14 NO**: measured on yield rather than RMS, `1/2/2/2` delivers **88.7–99.3% usable keypoints against `1/1/1/1`'s 75.9–88.7%**, so X-24's leader is confirmed on the metric that matters and is the recommended ladder ([D-23](#d-23-the-tracker-clips-its-window-and-does-not-pad-its-levels--measured-not-argued)). Footprint axis unchanged from [X-15](EXPERIMENTS.md). | Pyramid level bit depths. | T3.4, **T3.8**, T3.10 (the N-bit tracker, T4.1) | **Phase 4** (T4.1) ✔ |
| ~~**E-6**~~ **RESOLVED — BOTH** | Does fully bit-parallel tracking (census/Hamming) match hybrid LK's accuracy, and what does it cost? | Only route (b) of [§7.9](#79-known-hard-problems)'s two-route split had ever been built. | **Answered: it does not match, it costs 3.00× LESS, and the split resolves to BOTH routes shipping.** [X-26](EXPERIMENTS.md) built route (a) and measured both: yield **56.7–75.0%** against route (b)'s 75.9–99.3% — an algorithm difference, since route (a) loses on the SAME ladder — but **102 240 B against 306 720 B** because it forms no derivative, a **1.45× cheaper build**, **0.93×** the tracking time at R=2, and **3.2× more usable keypoints per KB at equal keypoints per millisecond**. Route (b) stays the default; route (a) ships as the memory-constrained alternative, because CLAUDE.md's tiebreak covers speed against footprint and this is accuracy against footprint, which §1 puts on the integrating pipeline's side. The derived floor was confirmed to two figures (0.3873 px measured against 0.408 px derived; exactly 0.0000 px on integer motion). **It also refines [D-20](#d-20-the-trackers-per-pixel-work-is-all-popcounts-only-the-solve-is-float)**: a parabolic fit to a Hamming surface is MORE precise than the Gauss-Newton solve on the points both find, so LK's continuity buys robustness, not precision. | Whether route (a) replaces route (b). **Neither: both ship.** | — | **Phase 4** (T4.2) ✔ |
| ~~**E-5**~~ **RESOLVED** | Real speedup and peak-footprint numbers for a binary VIO frontend versus the byte-per-pixel equivalent. | T4.3 split it: 4.3a is the kernel comparison, 4.3b the sufficiency check. | **Both halves are now done.** 4.3a: [X-49](EXPERIMENTS.md), **1.53× faster and 6.23× smaller** on the reference device. 4.3b: [X-56](EXPERIMENTS.md) ran a real frontend loop over 1710 frames with **no gap requiring an operation binCV lacks** — and found detection is **39–57%** of that frontend against D-28's 4.8%. [D-46](#d-46-the-kernel-set-is-sufficient-for-a-vio-frontend--and-detection-is-3957-of-it). | Whether the kernel set is sufficient, and what it costs. | T4.3 | (X-49, X-56) ✔ |
| **E-30** | Detection is **39–57%** of a real VIO frontend and has never been optimised, because [D-28](#8-design-decisions) measured it at 4.8% and moved on. What is in those 18 ms? | [X-56](EXPERIMENTS.md) measured the duty cycle a frontend maintaining a feature count actually runs at. The NMS pool is **70 831 survivors on 752×480** — 19.6% of pixels — because a binarized response map takes few distinct values and ties are everywhere. **Ranking 70 k candidates every other frame is the obvious suspect and has never been profiled.** | **The largest unexamined term in the frontend.** Also worth separating: how much of it is the detection POLICY (worth 1.58× and outside binCV) versus the detector. | Where the next optimisation goes, now that `residualSums` is closed (D-40). | D-46 | **Phase 5** |
| ~~**E-10**~~ **RESOLVED** | Does the corner response need a frame-sized float map, or a rolling ring — and what does the ring's carry cost once the selection's global properties are preserved exactly? | **Answered: it does not need the map, and the ring is not a trade. The frontend goes 1 721 568 B → 500 464 B (3.44×) and the corner stage 1 333 848 B → 112 744 B (11.83×), with corners IDENTICAL to the byte and the whole detector 0.774× the time** — 40.79 → 31.59 ms/frame at 640×480. [D-22](#d-22-the-corner-response-streams-over-a-three-row-ring-and-that-is-the-recommended-path), [X-23](EXPERIMENTS.md). **~~for roughly 2× the response compute~~ — THIS ROW SAID THAT AND IT WAS WRONG, and X-23's rule pre-declared the correction rather than allowing it to be absorbed.** A ring forces a row-major sweep, which X-18 had already measured as the *faster* traversal at `blockSize` 3; the measured figure is **0.774×** (0.764× in the other device run), and even the two-pass shape the estimate described is 1.33×. The whole carry for the two GLOBAL properties — a frame-wide maximum and a frame-wide ordering — is **16 B**, because the threshold is a pure post-filter and the survivors are an up-set of the raw 3×3 maxima, so a top-K over the caller's existing candidate array is exactly the frame-map form's ranked set. It costs at large blocks — above `blockSize` 15 at `uint32_t` (1.08× at 31) and from `blockSize` 15 at `uint64_t` (1.03× at 15, 1.13× at 31) — and the frame-map form stays for that and for callers who want the map. (The corner stage's dominant term is now the CANDIDATE ARRAY — 8 754 survivors at 640×480, 9 774 on the real frame, structural maximum 3 659 568 B — which is a contract question and is deliberately left open. Still measured against no `CV_8U` denominator; that comparison is E-5's.) | Whether `cornerMinEigenVal` keeps a caller-provided frame map or gains a streaming form. | T3.7 (made caller-provided rather than decided) | Phase 3 (T3.11) ✔ |
| ~~**E-11**~~ **RESOLVED — already implemented** | Should `cornerMinEigenVal` select its window strategy on `blockSize`? | [X-18](EXPERIMENTS.md) measured the incremental form losing below blockSize 15 — 0.84× at 3, which is what `seal_params.yaml` configures. | **Answered: YES, and it already does on the path that matters.** [X-55](EXPERIMENTS.md) reproduced X-18 to two decimal places (0.85× at 3, 1.10× at 15, crossover between 7 and 15) — then found the premise stale: since [X-31](EXPERIMENTS.md), `cornerMinEigenValRow` **dispatches on `blockSize == 3`** to a bit-sliced path, and the frontend runs that, not the frame map X-18 measured. | Whether the sliding form is unconditional. **On the frontend's path it never was.** The frame-map API keeps its column-major slide rather than growing a branch no measured path exercises. | T3.7 | (X-55) ✔ |
| ~~**E-9**~~ **RESOLVED — NO** | Should the word type vary down the pyramid — `uint64_t` where it costs no bytes, `uint32_t` above? | [X-10](EXPERIMENTS.md) measured `uint64_t` reducing 1.94× faster and costing +33% at 94×60 but 0% at 640×480. | **Answered: NO, and the reason is a split.** [X-54](EXPERIMENTS.md): `uint64_t` is **1.66× faster on build** and **1.32× slower on track**, because build is word-parallel and track's NEON paths are guarded on `sizeof(WordType) == 4`. binCV is track-dominated, so it loses by ≈11% of frontend time. Footprint is +2.0%, not +33% — that figure was one level in isolation. [D-45](#d-45-one-word-type-because-the-two-halves-of-the-frontend-want-different-ones). | Whether the pyramid picks a word type per level. **It does not.** | D-14 | (X-54) ✔ |
| **E-29** | Should `residualSums` get a `uint64_t` NEON path, so the word type is genuinely free? | [X-54](EXPERIMENTS.md) measured the cost of not having one: `uint64_t` loses 1.32× on track while winning 1.66× on build. The guards are a **specialisation gap, not an ISA property** — aarch64's `CNT` counts a 128-bit register regardless of how the caller sliced it. | Same shape as the x86 gap ([X-52](EXPERIMENTS.md)): one kernel, one missing specialisation. Worth it only if a build-dominated pipeline appears, or if the `uint64_t` build win (1.66×) is wanted without the track loss. | Whether D-1's word-type genericity is real in the hot path or only in the API. | D-45 | unscheduled |
