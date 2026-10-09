# binCV on Cortex-M7 — STM32H753ZI

A bare-metal target for the NUCLEO-H753ZI. It exists to answer one question: **what does
binCV cost where there is no hardware population count and no vector unit?** Everything in
this directory is the harness; binCV itself is `bincv_core`, unchanged.

## Build, flash, run

Needs an `arm-none-eabi` GCC on `PATH` (or `BINCV_ARM_TOOLCHAIN_DIR` pointing at one).

```bash
cmake -S . -B build-m7 \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-cortex-m7.cmake \
      -DBINCV_BUILD_EMBEDDED=ON -DBINCV_BUILD_TESTS=OFF \
      -DBINCV_USE_OPENCV=OFF -DBINCV_BUILD_BENCHMARKS=OFF
cmake --build build-m7          # one image per benchmark: bincv_m7_reduce, bincv_m7_dense
```

**One image per benchmark**, each named after its own translation unit
(`bincv_m7_reduce.cpp` is `bincv_m7_reduce`), for the reason there is one host binary per
benchmark: `scripts/check_figure_staleness.py` ages a figure when a file its image compiles
changes, and one image compiling every kernel would age every figure on every edit.
`m7_harness.cpp` is what they share — bring-up and the command loop.

The images are driven from the host by **`scripts/m7_host.py`**:

```bash
scripts/m7_host.py flash bincv_m7_reduce            # through the ST-LINK's drive
scripts/m7_host.py run PING CLOCK REDUCE            # one session, replies printed
scripts/m7_host.py launches -n 10 --image bincv_m7_reduce \
    -o docs/reports/logs/m7-reduce-cortex-m7-launches.log REDUCE STACK
```

`launches` is `run_launches.sh` for the board: it builds the image from the current tree,
flashes it, and takes N launches, each one a system reset so caches, stack and statics start
cold. Its log carries the same header the host runners write, `# sources:` included, so a
figure taken on the board goes stale when the code under it moves, like any other.

Every exchange is a request and a reply: the host sends a command line, the board answers
with lines ending in `OK` or `ERR`, and binary data only travels when the board has asked for
exactly that many bytes, so nothing the board times includes the link and its receive FIFO
cannot overrun. The link runs at 4 Mbaud over the ST-LINK's virtual COM port and moved 3.7 MB
without a CRC failure. Under WSL2 the COM port stays on the Windows side and
`scripts/m7_bridge.ps1` pipes it to a socket, which needs no administrator; the ST-LINK's
drive is flashed by ejecting it first, because Windows otherwise reports the copy as
corrupted.

The tests do not build for this target as a whole (`threads/pool.hpp` needs `<thread>`,
which newlib does not supply), which is why `BINCV_BUILD_TESTS=OFF` above.
`scripts/verify_cortex_m.sh` is the compile gate: it builds the 37 suites that can be built
at 32-bit `size_t` and links every image; it does not run them.

## The clock, and how it is checked

**480 MHz**: PLL1 from the ST-LINK's 8 MHz clock in HSE bypass, `DIVM1 = 2`, `DIVN1 = 240`,
`DIVP1 = 2`, voltage scale 0, the AXI and AHB buses at 240 MHz and every APB at 120 MHz,
flash at 4 wait states — the settings Zephyr uses for this board. I+D caches on throughout.

A misconfigured PLL still runs everything, and every number it produces is then plausible and
wrong, so the configured frequency is not trusted: the `CLOCK` command counts core cycles
across one second of the board's 32.768 kHz LSE crystal, through LPTIM1, and every launch
logs it. At 480 MHz the core read **480.22–480.64 MHz** across the twenty launches below:
about 0.1% fast, wandering by up to 870 ppm. That wander is the ST-LINK's 8 MHz reference
rather than the method — on the internal 64 MHz oscillator the same check read 64.236–64.253
MHz over fourteen checks, a 270 ppm spread that includes the RC oscillator's own drift
([log](../../docs/reports/logs/m7-reduce-cortex-m7-resetclock-launches.log)). **Figures here
are cycle counts, which none of this touches**; milliseconds are converted at 480 MHz and
carry that 0.1%.

The first figures on this board were taken on the 64 MHz reset clock, before this path was
brought up and checked. `-DBINCV_M7_RESET_CLOCK=ON` still builds that configuration, and its
logs sit beside the 480 MHz ones. **Its cycle counts do not transfer to 480 MHz**: the
dense-disparity kernel takes 52,593,359 cycles there and 46,931,454 at 480 MHz
([log](../../docs/reports/logs/m7-dense-cortex-m7-resetclock-launches.log)), while the
population count goes the other way, 201,771 against 204,853. The two configurations differ
in flash wait states (7 at reset), bus ratio and voltage scale, and which of those moves
which count was not isolated.

## What was measured

Two things, each with its rule written down before the board ran:

1. **Population count without an instruction** — four ways to count the bits of a 32-bit
   word, timed inside the bulk reduction that calls them.
2. **Dense disparity at 32-bit pointer width** — the binary dense kernel on an M-profile
   core, at `uint32` against `uint64`.

Both are re-taken at 480 MHz, ten launches each, from the logs named under each table. Every
launch returned the same cycle count to within 0.002%: this core has no frequency scaling,
no other process and no OS, so the run-to-run scatter the host logs exist to measure is
not there to measure.

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

Each launch is the median of 7 interleaved repeats; the cycles are the median over ten
launches, which agreed exactly ([log](../../docs/reports/logs/m7-reduce-cortex-m7-launches.log)).
All four arms and `bincv::countNonZero` returned 180,378.

| arm | cycles | vs A | `.text` |
|---|---|---|---|
| **A — shipped** (`__popcountsi2`) | 204,853 | 1.00× | 48 B |
| B — portable (64-bit SWAR) | 377,148 | 1.84× slower | 180 B |
| C — SWAR32 | 216,146 | 1.06× slower | 78 B |
| D — SWAR32 + `USAD8` | 181,578 | 1.13× faster | 78 B |

**Decision at the time: not adopted; arm A stays.** D is the fastest, and the library kept
the slower arm on two grounds: a 1.13× gain on a microbenchmark that counts bits and does
nothing else, and a chip-specific path that must stay bit-exact with the portable one on
every future change with nothing in CI executing it. **The second ground no longer
stands** — an M-profile arm is now judged on its measurement, with its correctness proven
on the board, rather than declined because CI cannot run it — so the decision is open
again. It belongs with the Lucas–Kanade work below, where the counts that remain are
single words, rather than with this loop.

Three things it settled:

- **The call was not the cost.** C removes A's per-word call and is slower by 5%: the M7
  predicts the call perfectly and dual-issues around it, while the inlined SWAR adds
  register pressure the call did not.
- **`USAD8` is the only thing that helped**, and only by removing two shift-adds from the
  tail. There is no cheap win in the per-word counter on this core.
- **B is 1.85× slower than what ships.** `popcountWordPortable` is not what this target
  compiles to (GCC calls libgcc's `__popcountsi2`), and `reduce.hpp` says so.

For scale: 204,853 cycles is **0.43 ms** to count a 752×480 frame at 480 MHz. The first
bring-up expected the AXI SRAM, at half the core clock, to become the limit at 480 MHz; it
did not — the count takes 1.5% more cycles than the 201,771 it takes at the reset clock,
because 46 KB in 0.2 million cycles is far below what the bus delivers.

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
| **L2** | **0.42×** | **0.12×** | 1.06× |
| L3 | 0.44× | 0.11× | 0.96× |

M7 figures are cycle ratios at 480 MHz, ten launches that agreed exactly
([log](../../docs/reports/logs/m7-reduce-cortex-m7-launches.log); L0 is 205,591 cycles). Host
figures are medians whose run-to-run spread was 49–107%, so only the order-of-magnitude
differences there carry, which is all that is claimed from them.

**L2 is 2.4× faster on the M7 and roughly 8× on x86 without POPCNT**, in plain C++ any
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

Each launch is the median of 3 interleaved repeats; the cycles are the median over ten
launches, which agreed to 0.002% ([log](../../docs/reports/logs/m7-dense-cortex-m7-launches.log)).
Both word types exact on the supported region, and the two maps byte-identical.

| word type | cycles | ms at 480 MHz | scratch |
|---|---|---|---|
| **`uint32`** | 46,931,454 | **97.8** | 6,480 B |
| `uint64` | 62,589,097 | 130.4 | 6,480 B |

**`uint64` is 1.33× slower: the 64-bit guidance inverts at 32-bit pointer width**, as a
core synthesizing every 64-bit ripple from register pairs would suggest, and now as a
measurement — 1.30× at the reset clock, 1.33× at 480 MHz. `denseDisparity.hpp` carries the
width-qualified guidance. The scratch tie is geometry, not a rule: at this width both types
pack the same bytes.

For scale only: a QVGA depth map on a microcontroller in 98 ms, in 6.5 KB of scratch. There
is no comparison against anything else on this part here; this is the kernel's own cost.
