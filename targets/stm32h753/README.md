# binCV on Cortex-M7 — STM32H753ZI

A bare-metal target for the NUCLEO-H753ZI. It exists to answer one question: **what does
binCV cost where there is no hardware population count and no vector unit?** Everything in
this directory is the harness; binCV itself is `bincv_core`, unchanged.

## Build, flash, read

Needs an `arm-none-eabi` GCC on `PATH` (or `BINCV_ARM_TOOLCHAIN_DIR` pointing at one).

```bash
cmake -S . -B build-m7 \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-cortex-m7.cmake \
      -DBINCV_BUILD_EMBEDDED=ON -DBINCV_BUILD_TESTS=OFF \
      -DBINCV_USE_OPENCV=OFF -DBINCV_BUILD_BENCHMARKS=OFF
cmake --build build-m7 --target bincv_m7      # writes build-m7/targets/stm32h753/bincv_m7.bin
```

Flash by copying `bincv_m7.bin` onto the mass-storage drive the board's ST-LINK presents
(or with `st-flash write bincv_m7.bin 0x8000000`). The firmware reports over USART3, which
the ST-LINK exposes as a virtual COM port: **115200 8N1**. It prints the clock, the cache
state and which arm is which before any number, so a report that cannot state its clock is
not a measurement.

The tests do not build for this target as a whole (`threads/pool.hpp` needs `<thread>`,
which newlib does not supply), which is why `BINCV_BUILD_TESTS=OFF` above.
`scripts/verify_cortex_m.sh` is the compile gate: it builds the 37 suites that can be built
at 32-bit `size_t` and links this image; it does not run them.

## What was measured

Two things, each with its rule written down before the board ran:

1. **Population count without an instruction** — four ways to count the bits of a 32-bit
   word, timed inside the bulk reduction that calls them.
2. **Dense disparity at 32-bit pointer width** — the binary dense kernel's first execution
   on an M-profile core, at `uint32` against `uint64`.

All figures are at the reset clock, **HSI 64 MHz, no PLL**, I+D cache on. That is
deliberate for a first bring-up: the PLL, voltage-scaling and flash-wait-state path is the
step most likely to fail silently into a wrong frequency, and a wrong frequency is a wrong
measurement that still looks plausible. Ratios between arms are unaffected; absolute
milliseconds are a floor, not the part's number, until the 480 MHz path is brought up and
checked against a known interval.

## 1. Population count

### What is compared

Four ways to count the bits of one 32-bit word, all producing identical results, timed
over `countNonZero` on a 752×480 `BinMat<uint32_t>` (11,520 words per pass) rather than in
isolation:

| arm | what it is |
|---|---|
| **A — shipped** | `impl::popcountWord`, i.e. `__builtin_popcountll`. On this target GCC emits a call to libgcc's `__popcountsi2`. |
| B — portable | `impl::popcountWordPortable`, a 64-bit SWAR sequence inlined. |
| C — SWAR32 | The same SWAR arithmetic as libgcc's, inlined, never widening past 32 bits. |
| D — SWAR32 + DSP | C, with the final byte sum done by `USAD8` (ARMv7E-M DSP extension). |

A is the baseline because A is what a caller gets today. Disassembly showed that A and C
are the same arithmetic — libgcc's `__popcountsi2` is the SWAR sequence — so the only
difference between them is that A is out of line: a `bl` and a `bx lr` per word. B does
genuinely widen to 64 bits and is the largest of the four at 180 bytes.

The rule: find the fastest of the four, prove all four agree bit-for-bit, and report speed
beside code size so the two are weighed together. There is no adopt threshold here; a
threshold is a judgement made per case, and this file records the decision and the grounds
under Result. All four arms are compiled into every image and selected by
`BINCV_M7_POPCOUNT_ARM`, so one firmware times all four in one run and a mis-attached
`#define` cannot silently select one; the four counts must agree, and disagreement is a
failure rather than a footnote.

### Result

Median of 7 interleaved repeats, three consecutive power-on reports agreeing to better
than 0.5%. All four arms and `bincv::countNonZero` returned 180,378.

| arm | cycles | vs A | `.text` |
|---|---|---|---|
| **A — shipped** (`__popcountsi2`) | 201,600 | 1.00× | 48 B |
| B — portable (64-bit SWAR) | 374,270 | 1.85× slower | 180 B |
| C — SWAR32 | 213,300 | 1.05× slower | 78 B |
| D — SWAR32 + `USAD8` | 178,700 | 1.13× faster | 78 B |

**Decision: not adopted; arm A stays.** D is the fastest and the library keeps the slower
arm deliberately: a 1.13× gain on a microbenchmark that counts bits and does nothing else
does not buy a chip-specific code path that must stay bit-exact with the portable one on
every future change, on a core with no CI here.

Three things it settled:

- **The call was not the cost.** C removes A's per-word call and is slower by 5%: the M7
  predicts the call perfectly and dual-issues around it, while the inlined SWAR adds
  register pressure the call did not.
- **`USAD8` is the only thing that helped**, and only by removing two shift-adds from the
  tail. There is no cheap win in the per-word counter on this core.
- **B is 1.85× slower than what ships.** `popcountWordPortable` is not what this target
  compiles to (GCC calls libgcc's `__popcountsi2`), and `reduce.hpp` says so.

For scale: 201,600 cycles is **3.15 ms** to count a 752×480 frame at 64 MHz. Whether that
scales to ~0.42 ms at 480 MHz is not something this measurement can say, because at 7.5×
the core clock the AXI SRAM becomes the limit rather than the counter.

### The loop, which is where the win was

The arms above change the per-word counter. The thing to change is the **loop**. Four loop
shapes, in `benchmark/reduce_loop_arms.hpp` so the board and the host time the same source,
none using an intrinsic or a target `#if`:

- **L0** — the shipped shape: one accumulator, one call per word.
- **L1** — four independent accumulators, same per-word call.
- **L2** — keep the SWAR's per-byte lanes in an accumulator and pay its horizontal collapse
  once per 16 words instead of once per word.
- **L3** — L2, plus merging word pairs before the nibble step.

| | M7 (no popcount) | x86, no POPCNT | x86, POPCNT |
|---|---|---|---|
| L0 | 1.00× | 1.00× | 1.00× |
| L1 | 1.18× slower | 0.93× | 0.60× |
| **L2** | **0.39×** | **0.12×** | 1.06× |
| L3 | 0.42× | 0.11× | 0.96× |

M7 figures are cycle counts, three consecutive reports agreeing within 0.5%. Host figures
are medians whose run-to-run spread was 49–107%, so only the order-of-magnitude differences
there carry, which is all that is claimed from them.

**L2 is 2.6× faster on the M7 and roughly 8× on x86 without POPCNT**, in plain C++ any
target compiles. The split is by family, not by part: L2 wins wherever the count is a
software SWAR and is neutral-to-slightly-worse where the hardware has an instruction, so
gating it on "no hardware population count" leaves x86-with-POPCNT and aarch64 on exactly
the code they have today. L3 does not beat L2, so what remained was never in the body of
the count. L1 is target-dependent (1.67× faster on x86 with POPCNT, 18% slower on the M7)
and not a candidate. aarch64 was not measured; it has `cnt`, so L2 is predicted neutral or
slightly worse there, and the gate keeps it on its current path either way.

### What it is worth on the feature tracking pipeline: nothing

`benchmark/feature_tracking_profile.cpp`, built with `BINCV_X86_POPCNT` ON and OFF, prices
how much of the pipeline is population count at all (640×480, 140 keypoints, 31×31 window;
host spreads 7–47%, so read the large ratios only):

| stage | POPCNT on | POPCNT off | ratio |
|---|---|---|---|
| LK covariance + setup | 0.483 ms | 1.111 ms | 2.30× |
| corner response sweep | 3.101 ms | 3.503 ms | 1.13× |
| corner selection | 2.413 ms | 2.214 ms | ~1 |
| build | 1.032 ms | 0.992 ms | ~1 |
| whole pipeline | 7.31 ms | 8.12 ms | 1.11× |

L2 applies to none of it. The one popcount-bound stage is the LK covariance, whose
per-window counts are one to two words per row; L2 amortizes a collapse across sixteen
consecutive words, and a window row does not contain sixteen. Nothing on the hot path calls
a whole-frame `countNonZero`. So L2's 2.6× is real and confined to **long contiguous
reductions**, which the public API offers and this pipeline never performs; adopting it for
the pipeline's sake would optimize a loop shape the pipeline does not execute.

Two things this turned up are worth more than L2 was. The pipeline is only 1.11×
popcount-sensitive here, against the 3.75× recorded in the top-level `CMakeLists.txt` for a
different workload on a different machine; that discrepancy is worth resolving before the
3.75× is quoted again. And the biggest stage, the corner response sweep, is not
popcount-bound at all (1.13× where the covariance moves 2.30×); whatever governs it is
elsewhere. Its share of the whole depends on how often detection runs, which is a property
of the caller's duty cycle, not of the operation.

## 2. Dense disparity, and the word-type claim

`denseDisparity.hpp` recommends `uint64` for the binary dense kernel on 64-bit cores, where
it is worth 1.63× on the reference device. On this core every `uint64` operation is
synthesized from register pairs, so that guidance was a claim this target had never tested,
and the compile gate proves the kernel builds at 32-bit `size_t` without proving it runs.

### What is compared

`denseDisparityBinary` at 320×240, 32 disparities, 9×9 — a frame the part's RAM holds with
room to spare (a 752×480 output map alone would be 361 KB of its 512 KB AXI SRAM) — at
`uint32` against `uint64` over the same bits: the `uint64` buffers are byte copies of the
`uint32` ones, and a 320-pixel row has zero padding at both widths. DWT cycles per frame,
median of interleaved repeats, scratch bytes beside them. Correctness gates the timing: the
pair is a known constant shift, the supported region must answer with exactly that
constant, and the two word types' maps must agree byte for byte. There is no adopt
decision: both word types ship and the caller chooses; the header carries the result.

### Result

Median of 3 interleaved repeats. Both word types exact on the supported region, and the
two maps byte-identical: the kernel's first execution on M-profile is bit-exact.

| word type | cycles | ms at 64 MHz | scratch |
|---|---|---|---|
| **`uint32`** | 52,823,363 | **825** | 6,480 B |
| `uint64` | 68,807,055 | 1,075 | 6,480 B |

**`uint64` is 1.30× slower: the 64-bit guidance inverts at 32-bit pointer width**, as a
core synthesizing every 64-bit ripple from register pairs would suggest, and now as a
measurement. `denseDisparity.hpp` carries the width-qualified guidance. The scratch tie is
geometry, not a rule: at this width both types pack the same bytes.

For scale only: a QVGA depth map on a microcontroller in under a second at an eighth of the
part's clock, in 6.5 KB of scratch.
