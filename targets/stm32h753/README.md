# binCV on Cortex-M7 — STM32H753ZI

A bare-metal target for the NUCLEO-H753ZI. It exists to answer the question epic #21
asks on two targets and #13 asks here: **what does binCV cost where there is no
hardware population count?**

Everything in this directory is the harness. binCV itself is `bincv_core`, unchanged.

## The decision rule, written before the measurement

CLAUDE.md requires the rule be recorded before the numbers exist, so that the
conclusion cannot be fitted to them afterwards. This section was committed before the
board ran anything.

### What is being compared

Four ways to count the bits of one 32-bit word, all producing identical results, timed
over the bulk reduction that actually calls them (`countNonZero` over a 752x480
`BinMat<uint32_t>`) rather than in isolation:

| Arm | What it is |
|---|---|
| **A — shipped** | `impl::popcountWord`, i.e. `__builtin_popcountll`. On this target GCC emits a call to libgcc's `__popcountsi2`. |
| **B — portable** | `impl::popcountWordPortable`, the documented "sequence a Cortex-M build compiles to". |
| **C — SWAR32** | The same SWAR shape, inlined, never widening past 32 bits. |
| **D — SWAR32+DSP** | C, with the final byte-sum done by `USAD8` (ARMv7E-M DSP extension). |

**A is the baseline**, because A is what a caller gets today. B is measured beside it
rather than assumed to be the fallback: reduce.hpp describes B as the Cortex-M
sequence, and whether the compiler actually chooses it is a fact, not a given.

### Why C and D are candidates at all

This section originally argued that A and B both waste half their work widening a
`uint32_t` to `uint64_t`. Disassembling the image before running it showed that is
only true of B, and the corrected version is the reason to run the experiment at all:

- **A does not widen.** GCC proves the upper 32 bits are zero and calls
  `__popcountsi2`, not `__popcountdi2`. And `__popcountsi2` in libgcc *is* arm C's
  algorithm — the identical SWAR, differing only in that it sums the final bytes
  with two shift-adds where C uses a multiply.
- **So A and C are the same arithmetic**, and the only difference between them is
  that A is **out of line**: a `bl` and a `bx lr` for every word of every frame,
  plus the register pressure a call imposes on the loop around it.
- **B does widen**, genuinely: it is inlined 64-bit SWAR, and at 180 bytes it is by
  far the largest of the four.

That makes the question sharper than "is SWAR faster than the fallback". The
arithmetic is already SWAR either way. What is being priced is **the call**, and then
whether `USAD8` buys anything on top of removing it. If C beats A, the cost was never
the algorithm — which would also mean the fix generalises to every target without a
population count instruction, not just this one.

### The rule

**This section first set a 15% adopt threshold. That number was invented** — no prior
benchmark, no project policy, nothing behind it — and writing it down before measuring
made a guess read as though it were derived. binCV sets adopt/reject bars **per case**,
naming the metrics and the magnitude each one requires (CLAUDE.md), and has no
project-wide percentage. The threshold is deleted rather than rewritten, because the
decision it was standing in for belongs to a person, not to this file.

What the comparison is for, unchanged: find which of the four is fastest, prove all
four agree bit-for-bit, and report speed beside code size so the two are weighed
together. The adopt decision is recorded under **Result**, with who made it and why.

**If B beats A**, that is reportable on its own regardless of C and D: it would mean
the shipped path is slower than the portable fallback it exists to improve on, on the
platform binCV names first.

### What is held fixed

All four arms run in one firmware image, back to back, same frame, same buffers, same
clock, interleaved and repeated to spread any drift across arms rather than
concentrating it in one. Reported together, per CLAUDE.md:

- **time**, as DWT cycle counts, median of repeats;
- **footprint**, as `.text` bytes per arm from `arm-none-eabi-size`.

Conditions are printed by the firmware itself rather than assumed by the reader:
clock, cache state, and which arm is which. A run that cannot state its clock is not
a measurement.

### The switch, and the check that it is real

CLAUDE.md requires a vector-or-fast arm to be switchable off, and requires the
benchmark to show it is on. `BINCV_M7_POPCOUNT_ARM` selects the arm at build time and
every arm is compiled in every image, so the firmware times all four in one run and a
mis-attached `#define` cannot silently select one. The self-check is that **all four
arms must agree bit-for-bit on the same input**; the firmware reports the counts
beside the timings, and disagreement is a failure rather than a footnote.

## Result

Measured on the board: HSI 64 MHz, I+D cache on, 752x480 frame, 11 520 words per
pass, median of 7 interleaved repeats, three consecutive power-on reports agreeing to
better than 0.5%. All four arms and `bincv::countNonZero` returned 180 378.

| Arm | cycles | vs A | `.text` |
|---|---|---|---|
| **A — shipped** (`__popcountsi2`) | 201 600 | 1.00x | 48 B |
| **B — portable** (64-bit SWAR) | 374 270 | **1.85x** | 180 B |
| **C — SWAR32** | 213 300 | 1.05x | 78 B |
| **D — SWAR32 + USAD8** | 178 700 | **0.88x** | 78 B |

**Decision: not adopted. Arm A stays.** D is the fastest and the library keeps the
slower arm deliberately. Recorded rather than inferred, the grounds were: 12% does not
buy a chip-specific code path, and a DSP intrinsic that helps one core is
hyper-optimisation this library does not want yet.

Two things make that easy to agree with. The 12% is a **microbenchmark** — the timed
loop counts bits and does nothing else — so its share of a real frame is smaller by a
factor nobody has measured. And the cost is not the five lines; it is that a
hand-written arm must stay bit-exact with the portable one on every future change,
forever, on a core nobody here runs CI on.

Three things this settled, two of them against what was written here first:

- **The call was not the cost.** C removes A's per-word `bl`/`bx lr` and is *slower*,
  by 5%. The M7 predicts that call perfectly and dual-issues around it, while the
  inlined SWAR adds register pressure to the loop that the call did not. The
  reasoning recorded above for why C was a candidate is simply wrong, and the
  measurement is what says so.
- **`USAD8` is the only thing that helped**, and only by removing two shift-adds
  from the tail. That it does not clear the bar is the useful part: it means there
  is no cheap win here, so #13's "a SWAR popcount is probably the answer" is
  answered no.
- **B is 1.85x slower than what ships**, which contradicts reduce.hpp's description
  of `popcountWordPortable` as "the sequence a Cortex-M build compiles to". It is not
  what this target compiles to — GCC calls libgcc's `__popcountsi2` — and it is
  nearly twice the cost of what it does compile to. That claim is corrected in
  reduce.hpp.

For scale rather than for the rule: 201 600 cycles is **3.15 ms** to count a 752x480
frame at 64 MHz. Whether that scales to ~0.42 ms at the part's 480 MHz is not
something this measurement can say, because at 7.5x the core clock the AXI SRAM
becomes the limit rather than the counter. That is the next measurement, not an
extrapolation from this one.

## The loop, which is where the win actually was

The arms above change the per-word counter. `reduce.hpp` says the thing to replace is
the **loop**, and it is right. Four loop shapes, in `benchmark/reduce_loop_arms.hpp`
so the board and the host time the same source, none using an intrinsic or a target
`#if`:

- **L0** — the shipped shape: one accumulator, one call per word.
- **L1** — four independent accumulators, same per-word call.
- **L2** — keep the SWAR's per-byte lanes in an accumulator and pay its horizontal
  collapse once per 16 words instead of once per word.
- **L3** — L2, plus merging word pairs before the nibble step.

| | M7 (no popcount) | x86, no POPCNT | x86, POPCNT |
|---|---|---|---|
| L0 | 1.00x | 1.00x | 1.00x |
| L1 | **1.18x (worse)** | 0.93x | **0.60x** |
| **L2** | **0.39x** | **0.12x** | 1.06x |
| L3 | 0.42x | 0.11x | 0.96x |

M7 figures are cycle counts, three consecutive reports agreeing within 0.5%. Host
figures are medians whose run-to-run spread was 49-107%, so only the order-of-magnitude
differences there carry — which is all that is claimed from them.

**L2 is a 2.6x speed-up on the M7 and roughly 8x on x86 without POPCNT**, and it is
plain C++ that any target compiles. Three things it settled:

- **The split is by family, not by part.** L2 wins wherever the count is a software
  SWAR and is neutral-to-slightly-worse where the hardware has an instruction.
  Gating it on "no hardware population count" leaves x86-with-POPCNT and aarch64 on
  exactly the code they have today, so adopting it cannot regress the two platforms
  binCV has actually measured.
- **L3 does not beat L2**, so what remained was never in the body of the count. That
  is what L3 was carried to find out.
- **L1 is target-dependent and not a candidate**: 1.67x faster on x86 with POPCNT,
  and 18% *slower* on the M7, where four 64-bit accumulators cost more registers than
  the broken dependency chain saves. reduce.hpp's per-row accumulator split is the
  same idea at row granularity and was measured at 1.03-1.09x; this says it does not
  extend inward on an M-profile core.

Worth noting for the x86 POPCNT argument in the top-level `CMakeLists`: L2 without any
population count instruction runs within 4% of L0 *with* one (0.373 vs 0.357 ns/word).
That does not overturn the 3.75x tracking-pipeline figure, which covers far more than
reductions, but it does mean the reductions' share of it would shrink.

**Not measured: aarch64.** The reference device was unavailable. It has `cnt`, so it
is in the family where L2 is predicted to be neutral or slightly worse — and the gate
above means it keeps its current path either way. That prediction is unverified and
should be checked before anyone relies on it.

### And what it is worth on the feature tracking pipeline: nothing

`benchmark/feature_tracking_profile.cpp`, built with `BINCV_X86_POPCNT` ON and OFF, prices how
much of the pipeline is population count at all (640x480, 140 keypoints, 31x31 window;
host spreads 7-47%, so read the large ratios only):

| stage | POPCNT on | POPCNT off | ratio |
|---|---|---|---|
| LK covariance + setup | 0.483 ms | 1.111 ms | **2.30x** |
| corner response sweep | 3.101 ms | 3.503 ms | 1.13x |
| corner selection | 2.413 ms | 2.214 ms | ~1 |
| build | 1.032 ms | 0.992 ms | ~1 |
| whole pipeline | 7.31 ms | 8.12 ms | **1.11x** |

**L2 does not apply to any of it.** The one popcount-bound stage is the LK covariance,
and `bitSlicedPairRowRegion` issues `3N^2 + N` scalar per-word counts over a **31x31
window** — one to two words per row. L2 amortizes a collapse across sixteen
consecutive words, and a window row does not contain sixteen. Nothing on the hot path
calls a whole-frame `countNonZero` either.

So L2's 2.6x is real and is confined to **long contiguous reductions** — a full-frame
or large-region count, which the public API offers and this pipeline never performs.
Adopting it for the pipeline's sake would be optimizing a loop shape the pipeline does
not execute, which is the failure CLAUDE.md names: a kernel nobody calls makes no
performance claim.

Two things this turned up that are worth more than L2 was:

- **The pipeline is only 1.11x popcount-sensitive here**, against the 3.75x recorded
  in the top-level `CMakeLists`. That is not a correction — this is a
  different workload on a different machine (640x480 synthetic at 7.31 ms, against
  that run's 12.9 -> 3.4 ms) and the conditions to reproduce it are not recorded here.
  It is a discrepancy large enough to be worth resolving before the 3.75x is quoted
  again.
- **The biggest stage is not popcount-bound at all.** The corner response sweep moves
  **1.13x** where the LK covariance moves 2.30x, so whatever governs it is not the
  population count. It is **56-61% of detect**, and 42-43% of this profile's whole
  pipeline — but that second figure assumes detection runs every frame, which #7
  records as the misleading assumption: detection's share is a function of the duty
  cycle, not a property of the operation. The 1.13x is the part that does not depend
  on the duty cycle, and it is the part worth carrying to #7.

If the covariance is worth attacking, the shape that fits it is L2's *idea* applied
across a window's rows and its `3N^2 + N` sums -- accumulate lanes, collapse once per
window -- not L2's code. Even done perfectly that is bounded by the covariance's
13.7% share of the software-popcount pipeline.

## Dense disparity, and the word-type claim (rule written before the board ran it)

`denseDisparity.hpp` directs callers to instantiate the binary dense kernel at
`uint64` "unless they have measured a reason otherwise" — a guidance measured on a
64-bit A72, where it is worth 1.63x. On this core every `uint64` operation is
synthesized from register pairs, so the guidance is a documented claim this target
has never tested. This is also the first time the kernel **executes** on M-profile
at all: the compile gate proves it builds at 32-bit `size_t`, and building is not
running.

### What is compared

`denseDisparityBinary` at 320x240, D=32, 9x9 — a frame the part's RAM holds with
room to spare (the 752x480 output map alone would be 361 KB of its 512 KB AXI
SRAM) — at `uint32` against `uint64`, over the same bits: the `uint64` buffers are
byte copies of the `uint32` ones, and a 320-pixel row has zero padding at both
widths, so the two arms read identical frames.

### Metrics, and the gate on reporting them

DWT cycles per frame, median of interleaved repeats, with milliseconds at the
stated clock and scratch bytes for both types beside it. Correctness gates the
timing, as everywhere in this harness: the pair is a known constant shift, the
supported region must answer with EXACTLY that constant, and the two word types'
maps must agree byte for byte — a faster arm that computes something else is not a
result.

### No adopt threshold

There is no decision to gate: both word types ship today and the caller chooses.
Whichever wins, the header's word-type note gains the measured fact for 32-bit
cores; if the 64-bit guidance is contradicted here, that is reported as a
measurement against a documented claim, not silently edited around. Absolute
milliseconds at 64 MHz HSI are a floor, not the part's number, like every other
figure in this file.

### Result

Measured on the board: HSI 64 MHz, I+D cache on, 320x240, D=32, 9x9, median of 3
interleaved repeats. Both word types exact on the supported region, and the two
maps byte-identical -- the kernel's first execution on M-profile is bit-exact.

| word type | cycles | ms @ 64 MHz | scratch |
|---|---|---|---|
| `uint32` | 52 823 363 | **825** | 6 480 B |
| `uint64` | 68 807 055 | 1 075 | 6 480 B |

**u64 is 1.30x the u32 time: the header's 64-bit guidance inverts here**, as a
32-bit core synthesizing every 64-bit ripple from register pairs would suggest --
and now as a measurement rather than a suggestion. The header's word-type note
carries the corrected, width-qualified guidance. The scratch tie is geometry, not
a rule: at this width both types happen to pack the same bytes.

The milliseconds are the 64 MHz HSI floor, not the part's number, per Status
below. For scale only: a QVGA depth map on a microcontroller in under a second at
an eighth of the part's clock, in 6.5 KB of scratch.

## Status

The clock is the reset default — **HSI at 64 MHz**, no PLL. That is deliberate for
the first bring-up: PLL1 plus VOS0 plus flash wait states is the step most likely to
fail silently into a wrong frequency, and a wrong frequency is a wrong measurement
that still looks plausible. Arm-to-arm ratios are unaffected by it. Absolute
per-frame figures at 64 MHz are a floor, not the part's number, and are labelled as
such until the 480 MHz path is brought up and verified against a known interval.
