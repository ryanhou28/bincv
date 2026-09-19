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
make the one they chose cost less. An operation belongs here when it sits on a path
**users** run *and* binCV can make it smaller or faster — and does not when binCV would
contribute nothing but a second implementation to keep correct. A library's users include
people outside this repository, so an operation does not wait for an in-repo caller to
exist (owner's decision, 2026-09-11). What an in-repo caller *is* for is pricing: every
operation still gets a benchmark arm the day it is written, and a representative pipeline
is what turns kernel numbers into shares.

That covers image processing, features and tracking, stereo, and the geometry the frontend
consumes downstream of them. The SLAM use case brought the descriptor path — orientation,
steered BRIEF, Hamming matching — and sparse rectified stereo, for the same reason tracking
brought LK: users' pipelines run them, and bits make them cheaper. Dense disparity is
scheduled on the same test. IMU fusion and bundle adjustment are absent on the second
prong, not the first: float linear algebra offers the representation nothing to exploit.

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

A frame **sequence** travels the same way. `io/sequence.hpp`'s blob is a fixed 32-byte
header followed by `P4`- or `P5`-shaped bodies back to back — the same two layouts,
concatenated, no third pixel format and still no codec — so a filesystem-less target can
be fed a whole dataset from one byte range: a file to `fread` or mmap, an app asset, an
`xxd -i` array in flash, a stream over USB/UART. `scripts/make_sequence_blob.py` writes
one on the host, which is where the decoders live; its packed mode also runs the sensor
stage there, trading coverage for 8× more frames in the same flash, and the header
records which trade a blob made so a reader cannot confuse them.

---

## 8. Platforms

binCV targets desktop, mobile and embedded CPUs. None of these is a lesser target; they
differ in what they make expensive.

| | what shapes the code |
|---|---|
| **x86-64** — desktop, laptops | `POPCNT` is required. AVX2 is selected at run time, so the baseline ISA is unchanged and one binary runs everywhere |
| **aarch64 Cortex-A** — mobile, single-board | NEON throughout. Population count is a vector instruction, which is why reductions are bulk-only |
| **Cortex-M** — microcontrollers | No population count and no NEON, so the software path is the only path. Built and run on a Cortex-M7 |
| **RISC-V** | Population count is in an optional extension, so it is the Cortex-M question on a target where it may go either way |

x86-64 and aarch64 are measured throughout these notes. **Cortex-M is built, run and
partly measured** — see below for exactly how far that goes. 32-bit Cortex-A and RISC-V
have not been built, and until they are, nothing here is a claim about them.

### What runs freestanding, and what does not

Measured on an STM32H753ZI (Cortex-M7) with arm-none-eabi GCC 14.2, `-fno-exceptions
-fno-rtti`, newlib, and no vendor SDK:

**Runs.** `bincv_core` in full — containers, views and the `ops/` kernels. 34 of the 35
test suites cross-compile clean under the whole warning set at 32-bit `size_t`, a pointer
width the four-word-type sweep had never been compiled at before; the one 64-bit
assumption it exposed was in a test, not the library.

**Does not, by design.** `threads/pool.hpp` needs `<thread>`, `<mutex>` and
`<condition_variable>`, which newlib does not supply. That is section 9 behaving as
intended rather than a gap: binCV is serial by default and threads through a
caller-installed backend, so a target with no threads never installs one.

**Needs care, neither on a kernel path.** `std::aligned_alloc` is absent from newlib's
`std` even though `::aligned_alloc` exists, and `simdStatusString` uses `snprintf`, which
drags in enough of newlib's stdio to want `_sbrk` at link time. No kernel allocates,
throws or does I/O, so neither reaches one.

**What is NOT measured there:** any OpenCV comparison, any frontend or tracker timing, and
anything at the part's full clock — the reductions were timed at the 64 MHz reset default.

**The stack was expected to be the binding constraint. On the first real part it was
not.** The tracker stages each window into stack buffers whose size grows with the bit
depth: about 4 KB at the shipped depth, 15 KB at the deepest supported. Measured on the
STM32H753ZI, that is **4,120 bytes at N = 2 against a 16 KB stack** — a quarter of it, and
comfortable. The prediction still holds at the deepest depths, where 15 KB of a 16 KB
stack leaves nothing, and it would hold on a smaller part; it simply did not bite here.

The mechanism is what makes that checkable rather than hopeful:
`BINCV_STAGING_BUDGET_BYTES` declares the budget and a static assertion fails the build
rather than overflowing at run time, which matters because overflowing a stack is silent
corruption rather than a crash. `stagingStackBytes<N, W>()` gives the exact figure, and
the bare-metal target sets the budget from the same number its linker script reserves so
the two cannot drift.

---

## 8.5 Backends

A backend runs binCV's operations on a different compute device. There is one
today, `backends/cuda/`, and the shape it takes is a settled decision rather
than an open question (issue #34).

**The format and the contract are shared; containers, kernels and the execution
model are forked entirely.** A device bit-plane is byte-identical to a host one
— pixel `x` at bit `x % 32` of word `x / 32`, rows a stride apart, padding bits
zero — so the invariants of section 1 have exactly one definition, upload and
download are raw pitched copies with no conversion, and "the GPU gives the CPU's
answer" is a byte comparison a test runs, not an aspiration. What is *not*
shared is every line that touches the device: allocation, streams, tiling, the
warp. A device traversal has nothing in common with a row loop, and pretending
otherwise would cost the performance the backend exists for.

**A backend is never a drop-in dispatch target.** Data location is visible in
the type. `bincv::cuda::DeviceBinMatView` is a distinct type from
`BinMatView<uint32_t>` — layout-identical, asserted so — precisely so that
handing a host pointer to a device kernel is a compile error rather than a
late, memory-corrupting runtime one. Operations live in `bincv::cuda::`, take
device views and an explicit stream, and allocate nothing (section 2's rule,
unchanged); upload and download are the only operations that name host memory.
A kernel launch is microseconds against operations measured in microseconds, so
a per-op host/device switch would be latency-bound regardless — the GPU wins
only when a whole pipeline stays resident, which is a different API shape, not a
hidden copy behind a familiar call.

**The device word type is `uint32_t`, and only that.** A CUDA core is a 32-bit
integer machine: there is no 64-bit integer datapath, so a `uint64_t` operation
is two 32-bit operations and a register pair — the Cortex-M result (section 3's
inversion: `uint64_t` slower at 32-bit width), not the aarch64 one. The 32-bit
granule is also the width the hardware's own primitives speak: `__popc` counts a
32-bit register, and `__ballot_sync` returns one bit per lane of a 32-lane warp
— a packed 32-pixel word produced in a single instruction, which is how the
device sensor stage and census transform pack. Wider *memory* access still
happens, as 128-bit `uint4` loads inside a kernel; load width and format word
width are independent, exactly as a NEON load moves 128 bits of 32-bit-word
plane on the host. No caller is restricted by the choice: on little-endian a
plane at any host word width is byte-identical to a `uint32_t` plane (section 3,
`narrowPlane`), so upload accepts all four host word types as a byte copy.

**The views are forked; a handful of scalar helpers are shared, and the line
between those two is the whole of the rule.** `core/error.hpp` defines
`BINCV_HOST_DEVICE`, which expands to `__host__ __device__` under nvcc and to
*nothing* under every other compiler. It is not a general annotation of the core
types, and it is not a door to a shared kernel. A function may carry it only
when it is **scalar and traversal-free** — no loop over pixels, rows or words,
no allocation, and no walk over an image. That covers the closed-form rules the
two sides would otherwise each have to derive: `impl::borderIndex`,
`impl::reflect101Edge`, `impl::clipRegion` (with `regionFromExtent` and
`clipColumns` under it), `impl::minEigenValue`, `impl::quantScale`,
`impl::thresholdCutoff`, `impl::extendedRowWord`, `maj3` and `thresholdGE`.
Anything that walks an image stays forked, because that is exactly where the
host's row-major, popcount, cache-line shape and the device's warp shape
genuinely disagree.

**The line is traversal, not pointer-freedom**, and two entries in that list are
where the distinction is visible: `thresholdGE` takes a plane pointer and reads
one word from each of `nPlanes`, and `impl::extendedRowWord` takes a row pointer
and reads one word at a caller-computed index. Neither decides *which* pixels to
visit or in what order — the caller's traversal does that, and the traversal is
what stays forked. What they encode is a closed-form rule about the bits once
read, which is exactly the thing that must not exist twice. `extendedRowWord` is
the clearest case: it is the row-edge blend that makes padding bits past `width`
read as the fill, and a copy of it that differs by one bit is invisible in the
middle of a frame and makes every word-wise reduction over-count. It was briefly
restated twice inside the CUDA backend — once in `cuda/shift.hpp` and once,
independently, in `morphology.cu` — and the two copies had already diverged in
spelling before either shipped, which is the failure this rule exists to
prevent arriving exactly on schedule. Folding them back onto the host's own
function is **instruction-neutral** (the morphology translation unit compiles
identically either way); the reason to do it is that there is now one
definition of the rule rather than three.

The reason the line sits there is that a second derivation of one rule is this
project's recurring failure mode: the copy that drifts does not crash, it
answers a plausible question nobody asked. `backends/cuda/src/reduce.cu` carried
a hand-written restatement of the clip geometry for its batch kernel while its
own header comment claimed nothing was copied, and `pack.cu` carried a
`quantScaleDevice` restating a rounding form whose divergence from OpenCV is
deliberate — a drifted copy there would have read as that divergence finally
being fixed. Both now call the host's own definition.

Three properties are **proven rather than asserted**. The library still compiles
under a plain C++17 compiler with no CUDA installed — `tests/test_error.cpp`
inspects the macro's expansion and fails the build if it is not empty, in every
configuration including the Cortex-M gate, where the compiler is
`arm-none-eabi-g++`. Its mirror in the CUDA suite fails if the expansion is
empty *there*, since either assertion alone would be satisfied by a macro that
is always empty. And `backends/cuda/tests/test_cuda_shared_helpers.cu` sweeps
every shared helper on both targets and compares the answers exactly —
`minEigenValue` by float bit pattern, because "close enough" is not the claim.
`BINCV_ASSERT` works inside a shared helper (`detail::assertFailed` has a
`printf`/`__trap()` device branch), so preconditions do not vanish on one of the
two targets; `BINCV_THROW` deliberately does not, since it reports a host-side
setup failure.

The shared layer is therefore the byte layout, the contracts, those scalar
rules, and the equality tests. The two device copies of the row-geometry helpers
(`rowWords`, `rowTailMask`) remain copies — they are `uint32_t`-specific device
forms of templated host originals — and are asserted equal to those originals
across widths rather than trusted to stay in step.

**What runs, and what is measured.** The backend provides bitwise logic, bulk
population-count reductions, the sensor stage (`packBits`), the census
transform, and dense disparity by both entries — an already-binary pair, and a
wide 8-bit pair transformed to census on device. Correctness is bit-exactness
against the host library, proven by `backends/cuda/tests` and gated by
`scripts/verify_cuda.sh` (which exits 77 without a GPU, the "not performed" code
the other cross-target gates use). Speed is measured on an **RTX 3070 Ti (SM
8.6) under WSL2**, both kernel-resident and end-to-end, next to the host
library's own arm on the same machine and against `cv::cuda::StereoBM` as the
best existing GPU option; the numbers and what each covers are in
`docs/reports/cuda.md`. As with every other platform in section 8, a number is a
claim only about the hardware it was measured on: **no Jetson or other device
has been run**, so nothing here is a claim about one — though the design is
built to accommodate one without a rewrite (the ops take views, so a unified- or
managed-memory pointer wraps into a device view with no copy, and the target SM
is a build setting, not a source change).

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
