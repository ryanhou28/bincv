# binCV — design notes

How the library is put together, and why the load-bearing choices are what they are.
For what each function does, see [API.md](API.md); for how to build and use it, see
[GETTING_STARTED.md](../GETTING_STARTED.md).

---

## 1. The representation

A binary image is stored one bit per pixel: pixel `x` of a row lives at bit `x % W` of
word `x / W`, where `W` is the word width. A whole row is a handful of machine words, and
a pointwise operation over 640 pixels is 20 `uint32_t` operations rather than 640 byte
operations.

Everything else follows from wanting that to stay true as the pixel type widens.

### N-bit values are bit-planes, not packed fields

An N-bit image is **N separate 1-bit planes**, not pixels packed N bits wide. Plane `i`
holds bit `i` of every pixel.

The reason is arithmetic. With packed fields, adding two images means masking each field,
adding, and handling carries between neighbours — work proportional to the field width
and awkward at every boundary. With bit-planes, addition is a **bit-sliced adder**: a
chain of full-adders built from AND, XOR and OR over whole words, where each gate
processes `W` pixels at once. Comparison, thresholding and weighted sums decompose the
same way.

It also means the 1-bit case is not a special case. A binary image is an N-bit image with
N = 1, and every kernel is written once.

### Signed values are sign-magnitude

A derivative is signed, and a two's-complement bit-plane representation would make the
sign bit participate in every carry. Instead a signed value carries **N magnitude planes
plus one sign plane**.

This pays off directly in the tracker. A ternary derivative — the N = 1 case, values in
{−1, 0, +1} — makes the gradient covariance a set of population counts over masks, with
no multiplies at all:

```
Σ Ix²  = popcount(magX)
Σ IxIy = popcount(magX & magY & ~(signX ^ signY))     // agreeing signs: +1
       − popcount(magX & magY &  (signX ^ signY))     // opposing signs: −1
```

### Padding bits are always zero

A row whose width is not a multiple of `W` has bits past the last pixel in its final
word. **Every operation that writes whole words clears them.** This is not tidiness: a
word-wise reduction counts set bits, so a stray padding bit is a phantom pixel. Measured
during development, a word-wise NOT without the mask was bit-exact against OpenCV on
every test and left 826 200 phantom bits behind for the next reduction to count.

---

## 2. Containers, views and ownership

**Containers own memory and have value semantics.** Copying a matrix copies its pixels.
There is no reference counting and no shared mutable state, so a function that takes a
matrix by value cannot surprise its caller.

**Kernels take views, never containers.** A view is `{pointer, width, height, stride}`.
This is the single most consequential interface decision in the library:

- A kernel compiles **once** per (word type, depth) rather than once per container type,
  and works on caller memory, a sub-region, a sensor DMA buffer, or a container, without
  knowing which.
- Strides are read per row, so a kernel is correct on over-aligned rows and on a caller's
  buffer with its own pitch.
- Sharing is explicit. A view is how you say "look at this without owning it".

There are two view types, mutable and const, rather than one templated on constness —
because template deduction does not consider the mutable-to-const conversion, and a
single type would have made every call site spell out its arguments.

### Kernels do not allocate

No kernel allocates, and none throws. Scratch buffers, where an operation needs one, are
caller-provided parameters. On a target with no heap this is the difference between a
library that runs and one that does not, and it also means the memory an operation costs
is visible in its signature.

---

## 3. Word type

Every container and kernel is templated on the word type — `uint8_t` through `uint64_t` —
and **`uint32_t` is the default**.

Wider words do less work per pixel, and on bulk operations 64-bit is measurably faster.
But binCV's memory footprint is the claim it exists to make, and a wider word rounds each
row's stride up more coarsely. Measured across a pyramid, 64-bit words cost **+33% at the
upper levels** and nothing at the base — and the upper levels are where a small target is
tightest. Where speed and footprint conflict and nothing else settles it, footprint wins.

**A 64-bit caller does not lose the vector kernels.** On little-endian a 64-bit bit-plane
already *is* a 32-bit bit-plane with twice the stride, so `narrowPlane` and `narrowLevel`
reinterpret it — no copy, no allocation — and the 32-bit vector paths apply. Measured on
the reference device, a narrowed buffer runs at 1.00× of native 32-bit, bit-identical.
That is why there is no second set of 64-bit kernels: they could at best match it.

---

## 4. Reductions are bulk-only

**There is no public per-word popcount, and that is deliberate.** On aarch64 the population
count instruction operates on a vector register, so counting a single general-purpose word
costs two register-domain crossings — roughly the cost of the count itself. A library that
exposes `popcount(word)` invites callers to write loops that pay that per word.

So reductions are offered over **regions, masks and sliding windows**, and the crossings
are amortized across the whole traversal. Internal helpers stay internal.

---

## 5. API tiers

Every public entry point declares one:

- **Tier 1** — bit-exact with the OpenCV function it names, proven by a test. It takes
  OpenCV's name because it gives OpenCV's answer.
- **Tier 2** — the same role and call shape, different numerics. It may take the name;
  the docstring says where it differs and by how much.
- **Tier 3** — no OpenCV equivalent. **These deliberately do not borrow OpenCV names**,
  because a familiar name on unfamiliar semantics is worse than an unfamiliar one.

---

## 6. Errors

A programming error — mismatched dimensions, a stride too short, overlapping buffers that
must not overlap — is an assertion, active in debug builds and absent in release. A
condition a caller cannot check in advance is a return value.

The library builds with exceptions disabled, so nothing in a kernel throws.

---

## 7. Where the operation set begins and ends

**binCV accepts a single-channel, integer-typed, strided pixel array and turns it into an
N-bit matrix.** Getting to that array is the caller's.

That one sentence settles cases in both directions. An 8-bit grayscale frame, a 12-bit
sensor buffer in `uint16_t`, the Y plane of an NV12 frame — all of them *are* such an
array, and the stride parameter already covers them. Decoding a PNG, demosaicing a Bayer
frame or converting color are not: each turns one wide image into another and leaves the
caller exactly as far from bits as before.

**Everything from such an array down to bits is binCV's, including sources wider than
8 bits.** Downconverting first is not merely slower, it changes the answer: a 12-bit
gradient of 15 counts becomes exactly zero once the operands are truncated to 8 bits, and
low-contrast scenes are where a frontend needs every edge it can get.

### What binCV computes, as opposed to what it accepts

The paragraphs above are the **input** boundary and they are a rule. What binCV *computes*
is not a rule, and it is worth saying so plainly rather than implying a taxonomy that does
not exist.

binCV provides memory- and performance-optimized versions of operations a vision pipeline
already runs. It takes no position on which algorithm a caller should use; the point is to
make the one they chose cost less. So the operation set grows with the use cases that turn
up. An operation belongs here when a caller needs it on their path *and* binCV can make it
smaller or faster — and does not when binCV would contribute nothing but a second
implementation to keep correct.

That covers image processing, features and tracking, and the geometry the frontend consumes
downstream of them. IMU fusion and bundle adjustment are absent because nothing has needed
them yet, which is a fact about the use cases rather than a line drawn on principle.

### binCV links no codec, on any target

This follows from the input boundary rather than from a size budget. **Every tier's real
frame source already is the input contract.** A capture SDK's buffer, a camera's YUV420 Y
plane, a V4L2 buffer and a sensor's DMA rows are all single-channel, integer-typed, strided
pixel arrays — which is why `packBits` and `packRows` take a stride, so they consume one
with no conversion at all. Nothing on a caller's path decodes anything.

Encoded files turn up in exactly one place, identically on every tier: reading a **dataset**
to test or benchmark against. That is tooling, and tooling runs on a host — including on
desktop, where the host already has OpenCV.

So there is no optional decoder target and no vendored codec. The measured size argument —
`libpng` + `libz` at 336 KB against the frontend's 436,704-byte peak working set — is real
but secondary; it argues about linkage. The decisive point is that a decoder would sit on a
path nobody walks. It is also worth noting where a vendored decoder fits worst: the target
with no package manager is the one that can hold neither the decoder nor the wide frame it
would produce, and it is the target that argument was aimed at.

binCV therefore reads and writes **PNM only** — `P4` and `P5` — because that is a header and
a copy rather than a codec. Two properties keep this honest, and both are load-bearing:

- **Output costs the representation's footprint, not the source image's.** `P4` stores one
  bit per pixel, which is binCV's own layout, so a 752×480 frame writes as 45,131 bytes.
  `P5` stores a byte per pixel and writes the same frame as 360,975 — an 8× buffer, on the
  target where buffers are scarcest, for the one use that justifies carrying a format at
  all. `writePbm` is the default; `writePgm` remains for grey levels. `writePbm` is also
  roughly twice as slow, which is the trade: both run once per file, off every per-frame
  path, so the buffer is what counts and memory wins.
- **Input streams; a resident wide frame is never assumed.** A `P5` body is a byte per
  pixel, so reading one whole costs exactly the frame binCV exists not to hold. That is
  free out of memory-mapped flash and not free off a UART or an SD card, so
  `readPgmHeaderFromPrefix` parses the header from the first bytes to arrive and `packRows`
  takes the pixels a chunk of rows at a time. Rows are independent, so the streamed result
  is bit-identical to the whole-buffer one. `P4` needs no such path: its file already is
  the matrix.

---

## 8. Platforms

binCV targets desktop, mobile and embedded CPUs. None of these is a lesser target; they
differ in what they make expensive.

| | what shapes the code |
|---|---|
| **x86-64** — desktop, laptops | `POPCNT` is required. AVX2 is selected at run time, so the baseline ISA is unchanged and one binary runs everywhere |
| **aarch64 Cortex-A** — mobile, single-board | NEON throughout. Population count is a vector instruction, which is why reductions are bulk-only |
| **Cortex-M** — microcontrollers | No population count and no NEON, so the software fallback is the only path. Stack is the binding constraint, not throughput |
| **RISC-V** | Population count is in an optional extension, so it is the Cortex-M question on a target where it may go either way |

The first two are measured. The other two are supported targets that have not yet been
built and measured, and until they are, the claims here are about the first two.

**The embedded constraint that actually bites is the stack.** The tracker stages each
window into stack buffers, and their size grows with the bit depth: about 4 KB at the
shipped depth, 15 KB at the deepest supported. On a desktop that is nothing; on a part
with a 16 KB stack it is everything, and overflowing one is silent corruption rather than
a crash. `BINCV_STAGING_BUDGET_BYTES` declares the budget and a static assertion checks
it, so a build that would not fit fails to compile instead. `stagingStackBytes<N, W>()`
gives the exact figure.

---

## 9. Threading

binCV is **serial by default and threads through a caller-installed backend**. It does not
create threads, and on a core-only build the parallel path compiles to the serial one.

Tracking splits over keypoints, which is safe by construction: each keypoint writes only
its own outputs and reads only shared const state. Measured, this scales about 2.6× at
four threads with peak memory flat, because the only per-thread cost is stack.

An integrator with an existing pool installs theirs and binCV never spawns anything.

---

## 10. Vector paths

Vector arms exist for both measured architectures, and three rules govern them.

**Every vector arm is switchable off**, so a benchmark can time both and a test can hold
them to producing identical results. A vector kernel that is faster and different is not
an optimization.

**A feature gate is derived from the compiler's own macros wherever the compiler can know
it.** NEON is mandatory in ARMv8, so `__ARM_NEON` is defined with no flags — routing that
through a build-system define once made every NEON kernel vanish for a consumer who added
the include path without linking the target, silently, at 1.78× the cost. Build-system
defines are reserved for what the compiler genuinely cannot know, such as whether
`-mpopcnt` was passed, and those are reported by `simdStatus()` instead.

**Runtime dispatch is per-kernel, not per-call.** Marking a small hot function with a
target attribute blocks inlining; measured, that cost 1.9× — more than the dispatch saved.
