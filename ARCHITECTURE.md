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
| **E-18** | Can `residualSums` carry **vector accumulators across the window**, reducing once per window instead of once per call? | [X-33](EXPERIMENTS.md) measured a **3.42× ceiling** for batched NEON popcounts and delivered **1.24×**. The gap is the horizontal add: it runs once per `slicedSignedSum` call — **~620 register-domain crossings per window** — where the ceiling amortized its extraction across the whole buffer. Collapsing that to one crossing per window is the remaining **2–3×** on a kernel that is **94.7% of the real frontend**, which makes it the largest single item left in Phase 5.1. | Whether `TapSums` becomes vector state and `residualSums` restructures around it — a real interface change, and the scalar path has to keep working. | X-28's unmet criterion 4 | **Phase 5** |
| **E-19** | Is the `1/2/2/2` ladder still the right operating point now that LK is 94.7% of the frontend and the ladder costs **2.30×**? | [D-23](#8-design-decisions) adopted it on ACCURACY — yield 88.7–99.3% against `1/1/1/1`'s 75.9–88.7% ([X-25](EXPERIMENTS.md)) — with its speed cost **estimated at 1.35×** from a confounded measurement, and chosen when corner detection was believed to be **52.7%** of the frontend rather than 2%. Isolated after [X-34](EXPERIMENTS.md) it is **2.30×**, and at `1/1/1/1` binCV is **1.34× slower than single-threaded SIMD OpenCV** against 3.08× at `1/2/2/2`. **This is the largest single speed lever left, larger than E-18.** The intermediate ladders were never measured for speed at all: `1/2/1/1` and `1/2/2/1` may buy most of the accuracy for a fraction of the cost, since the coarse levels track the same points through the same window and each N=2 level costs the same 4× regardless of how few pixels it has. | The shipped ladder, and whether the accuracy/speed trade belongs to the caller as `LKLevels` already lets it be. | D-23 | **Phase 5** |
| **E-17** | Where does the tracker lose the factor of **2.5–3** between the representation's 0.10 px floor and its own 0.25–0.29 px? | [X-27](EXPERIMENTS.md) closed off the representation, and X-24/X-25 closed off three pyramid parameters, so this is what is left of T3.8's MISS — and it is now a located problem rather than a diffuse one. **The prime suspect is deviation (i)**: the previous window is anchored on the integer grid because a bit-plane derivative cannot be interpolated, which displaces the aperture by up to half a pixel. `ops/opticalFlow.hpp` already calls that "the concrete thing route (b) trades away", and half a pixel of aperture displacement is the right order of magnitude for a 2.5× gap on a sub-pixel measurement. Two other candidates are untested: the single linearization per iteration on binary content, and the asymmetry that `I` is read at integer positions while `J` is interpolated. | Whether deviation (i) is a trade worth reversing, and at what cost — reversing it needs an interpolatable previous-frame gradient, which is a representation change. | T3.8's standing accuracy MISS | **Phase 4** |
| ~~**E-16**~~ **RESOLVED — THE REPRESENTATION IS NOT THE LIMIT** | Is X-20's 0.25 px RMS tolerance reachable at all with a 1-bit level 0 on real edge maps — and if not, what IS the representation's floor? | Three pyramid parameters had been eliminated without explaining T3.8's MISS, leaving the representation as the natural suspect. | **Answered: reachable, by a wide margin.** [X-27](EXPERIMENTS.md) measured a 31×31 window resolving **29.3 distinct binary states per pixel of displacement** — floor **0.025 px** noise-free, **0.10 px** at σ = 1 gray level, **0.174 px** even at σ = 4. **The 0.25 px criterion stands unchanged and no tolerance was widened.** X-20's "four independent crossings" was wrong but **conservative by ~7×**. Band D fired: from 11×11 to 41×41 set pixels grow 7.3× while distinct states grow only 1.8×, so the crossings lie on connected contours, are not independent, and **a bigger window buys almost no localisation** — window sizing cannot be justified by averaging. The remaining factor of 2.5–3 is the TRACKER's and is [E-17](#register). | Whether T3.8's tolerance is achievable as stated (**yes**), and whether level 0's depth is a variable (**it need not be**). | — | **Phase 4** (X-27) ✔ |
| ~~**E-7**~~ **ANSWERED — BUT THE QUESTION PRESUPPOSED THE WRONG CAUSE** | How many bits does each pyramid level actually need to preserve tracking accuracy? | The reference never chose its depths — they fell out of using `CV_8U`. [X-20](EXPERIMENTS.md) promoted this from an optimisation to a precondition on the reading that a 1-bit level cannot localise sub-pixel motion. | **[X-24](EXPERIMENTS.md) measured it and the premise does not survive. BAND C: no ladder up to `1/3/5/7` brings the four-level tracker inside X-20's tolerance on the full 141-point set — a deeper alphabet is NOT what fixes T3.8's miss.** Three things were learned instead, and two of them contradict this row's old text. **(1) Accuracy is PEAKED AT 2 BITS, not monotone in depth** — on `(1,0)` over the points that never clip, 1 bit gives 1.4742 px, **2 bits gives 0.0010 px (1474×)**, and 5 bits gives back 0.8567 px. Band D's own check excludes the artifact: every ladder tracked every point (141/141, 58/58), so the rows share identical point sets and `minEigThreshold` rejection is not involved. **(2) The dominant residual is CLIPPING, not quantisation** — the same ladder goes 0.8356 → **0.0010** px when restricted to the **58 of 141 points (41%)** whose 31×31 window is inside every level, so deviation (ii) is essentially all of that ladder's error rather than X-20's estimated half. That is now **[E-14](#register)**. **(3) 1-bit coarse levels ARE broken, so X-20 was half right — but not for its stated reason**: down a 1-bit ladder the edge map is THINNED AWAY (level 3 keeps 154 set pixels of 5 640 against `1/2/2/2`'s 1 028), so the second bit buys **content survival**, not precision. Why more than two bits then HURTS is **[E-15](#register)** and is not settled here. `1/2/2/2` is the leader on all three axes and is **cheap**: measured on the reference device it costs **1.35× the shipped ladder's tracking time, 1.51× its build and 1.17× its bytes** — against a pre-written cost model that predicted 3.25×, so the `20N²` popcounts per word do not dominate a tracked frame. The pre-registered "more than 2× is a headline" condition does not fire. **[X-25](EXPERIMENTS.md) then unblocked the adoption by answering E-14 NO**: measured on yield rather than RMS, `1/2/2/2` delivers **88.7–99.3% usable keypoints against `1/1/1/1`'s 75.9–88.7%**, so X-24's leader is confirmed on the metric that matters and is the recommended ladder ([D-23](#d-23-the-tracker-clips-its-window-and-does-not-pad-its-levels--measured-not-argued)). Footprint axis unchanged from [X-15](EXPERIMENTS.md). | Pyramid level bit depths. | T3.4, **T3.8**, T3.10 (the N-bit tracker, T4.1) | **Phase 4** (T4.1) ✔ |
| ~~**E-6**~~ **RESOLVED — BOTH** | Does fully bit-parallel tracking (census/Hamming) match hybrid LK's accuracy, and what does it cost? | Only route (b) of [§7.9](#79-known-hard-problems)'s two-route split had ever been built. | **Answered: it does not match, it costs 3.00× LESS, and the split resolves to BOTH routes shipping.** [X-26](EXPERIMENTS.md) built route (a) and measured both: yield **56.7–75.0%** against route (b)'s 75.9–99.3% — an algorithm difference, since route (a) loses on the SAME ladder — but **102 240 B against 306 720 B** because it forms no derivative, a **1.45× cheaper build**, **0.93×** the tracking time at R=2, and **3.2× more usable keypoints per KB at equal keypoints per millisecond**. Route (b) stays the default; route (a) ships as the memory-constrained alternative, because CLAUDE.md's tiebreak covers speed against footprint and this is accuracy against footprint, which §1 puts on the integrating pipeline's side. The derived floor was confirmed to two figures (0.3873 px measured against 0.408 px derived; exactly 0.0000 px on integer motion). **It also refines [D-20](#d-20-the-trackers-per-pixel-work-is-all-popcounts-only-the-solve-is-float)**: a parabolic fit to a Hamming surface is MORE precise than the Gauss-Newton solve on the points both find, so LK's continuity buys robustness, not precision. | Whether route (a) replaces route (b). **Neither: both ship.** | — | **Phase 4** (T4.2) ✔ |
| **E-5** | Real speedup and peak-footprint numbers for a binary VIO frontend versus the byte-per-pixel equivalent. | This is the project's headline claim. | Nothing — it is the result the project exists to produce. | — | **Phase 4** (T4.3) |
| ~~**E-10**~~ **RESOLVED** | Does the corner response need a frame-sized float map, or a rolling ring — and what does the ring's carry cost once the selection's global properties are preserved exactly? | **Answered: it does not need the map, and the ring is not a trade. The frontend goes 1 721 568 B → 500 464 B (3.44×) and the corner stage 1 333 848 B → 112 744 B (11.83×), with corners IDENTICAL to the byte and the whole detector 0.774× the time** — 40.79 → 31.59 ms/frame at 640×480. [D-22](#d-22-the-corner-response-streams-over-a-three-row-ring-and-that-is-the-recommended-path), [X-23](EXPERIMENTS.md). **~~for roughly 2× the response compute~~ — THIS ROW SAID THAT AND IT WAS WRONG, and X-23's rule pre-declared the correction rather than allowing it to be absorbed.** A ring forces a row-major sweep, which X-18 had already measured as the *faster* traversal at `blockSize` 3; the measured figure is **0.774×** (0.764× in the other device run), and even the two-pass shape the estimate described is 1.33×. The whole carry for the two GLOBAL properties — a frame-wide maximum and a frame-wide ordering — is **16 B**, because the threshold is a pure post-filter and the survivors are an up-set of the raw 3×3 maxima, so a top-K over the caller's existing candidate array is exactly the frame-map form's ranked set. It costs at large blocks — above `blockSize` 15 at `uint32_t` (1.08× at 31) and from `blockSize` 15 at `uint64_t` (1.03× at 15, 1.13× at 31) — and the frame-map form stays for that and for callers who want the map. (The corner stage's dominant term is now the CANDIDATE ARRAY — 8 754 survivors at 640×480, 9 774 on the real frame, structural maximum 3 659 568 B — which is a contract question and is deliberately left open. Still measured against no `CV_8U` denominator; that comparison is E-5's.) | Whether `cornerMinEigenVal` keeps a caller-provided frame map or gains a streaming form. | T3.7 (made caller-provided rather than decided) | Phase 3 (T3.11) ✔ |
| **E-11** | Should `cornerMinEigenVal` select its window strategy on `blockSize`? | [X-18](EXPERIMENTS.md) measured the incremental form **losing** below `blockSize` 15 — 0.84× at 3, which is what `seal_params.yaml` actually configures. But one device at one frame size is thin, and x86 showed the *opposite sign* there. | Whether the sliding form is unconditional or `blockSize`-gated. | T3.7 (left unconditional, qualified in the docs) | unscheduled |
| **E-9** | Should the word type vary down the pyramid — `uint64_t` where it costs no bytes (L0, L1), `uint32_t` above? | [X-10](EXPERIMENTS.md) measured both sides: `uint64_t` reduces **1.94×** faster and costs **+33%** at 94×60 but **0%** at 640×480, so the right answer may not be one type. The width is already a per-object template parameter (D-1), so this costs no new machinery — only a decision. | Whether the pyramid picks a word type per level, and whether kernels that walk several levels pay for two instantiations. | T3.4's pyramid, [D-14](#d-14-uint32_t-is-the-default-word-type) | unscheduled |

E-8 is the Phase 3 instance of the same discipline, and of the same lesson: it
gated the *primitive* T3.4 is built from, so it ran before that primitive was
written rather than after — and the answer was that its own framing of the trade
was wrong. Running it late would have meant shipping a pyramid built on a
prepared-plan API that nothing needed.

**E-4 closed in Phase 3 too, and E-12 is what it left behind.** E-4 asked whether
generic-`N` regresses the specialized paths, and the answer is no — but
[X-21](EXPERIMENTS.md) could only reach that answer by measuring both binCV routes
against a hand-written control with no genericity at all, and that control exposed
a cost E-4 had never named: **+93% per row**, of which only about a fifth is
genericity in `N`. **A register entry that closes cleanly can still hand back a
question, and the honest move is to register the new one rather than either widen
the old entry to swallow it or leave it as prose in a log.** E-12 is that entry. It
is gated on T4.1 for the reason the **Gates** column exists: the cost is worst on
the upper pyramid levels, which is exactly where T4.1's N-bit paths will run, so
measuring it afterward would mean either rewriting them or keeping a shape the data
does not support.

**E-13 arrived the same way, out of T3.10.** [X-22](EXPERIMENTS.md) priced the
N-bit covariance so that T4.1 could weigh a bit depth, and found on the way that a
decision D-15 took on measurement **at N = 1** — one accumulator per row — is being
carried into a kernel where its cost grows as N² while the work it is amortized
over grows as N² per *word*. The measurement that would settle it is confounded, in
that same entry, by a code-layout effect large enough (1.46× on unchanged source)
to swamp the difference. **Neither half of that belongs in a log**: the question is
real, the evidence is not conclusive, and the experiment that closes it has to be
designed to escape the confound rather than repeat it. It is gated on T4.1 for
E-12's reason — T4.1 is what runs at N > 1, so a shape change afterward is a
rewrite.

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

The claims this architecture supports are **kernel-level**, because kernels are
what binCV ships:

> Tier 1 operations bit-exact against OpenCV; tier 2 operations agreeing with the
> reference frontend frame by frame; several-fold smaller peak footprint over the
> frontend operation set; and faster execution on the bit-parallel operations,
> against the byte-per-pixel denominator.

**Not "equivalent VIO accuracy."** Trajectory error is a property of the whole
integration — frontend, estimator, IMU fusion and tuning — and this repository
supplies only the first. binCV can be flawless and a trajectory still poor, or
sloppy and a trajectory still fine, because the estimator absorbs a great deal.
Claiming it would also contradict [§1](#what-bincv-is-not), which puts estimation
out of scope.

Trajectory accuracy is still worth measuring, as **evidence** that these kernels
are sufficient for the job they were designed for. It is recorded that way — a
sufficiency check attributed to the integration — never as binCV's own result.

Not "10–100× faster than OpenCV." OpenCV is well optimized; on operations that
are not bit-parallel it will win, and chasing a throughput crown would pull
development toward benchmarking operations no real pipeline calls.
