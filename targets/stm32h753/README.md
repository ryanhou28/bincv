# binCV on Cortex-M7 — STM32H753ZI

A bare-metal target for the NUCLEO-H753ZI: binCV where there is no hardware population
count and no vector unit, and 1 MB of RAM in five banks. Everything in this directory is the
harness; binCV itself is `bincv_core`, the same headers every other target compiles.

**The assembled feature tracker runs here at the reference sequence's full 752×480,
bit-identical to the x86-64 host on every frame, in 684,664 bytes** — section 3.

## Build, flash, run

Needs an `arm-none-eabi` GCC on `PATH` (or `BINCV_ARM_TOOLCHAIN_DIR` pointing at one).

```bash
cmake -S . -B build-m7 \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-cortex-m7.cmake \
      -DBINCV_BUILD_EMBEDDED=ON -DBINCV_BUILD_TESTS=OFF \
      -DBINCV_USE_OPENCV=OFF -DBINCV_BUILD_BENCHMARKS=OFF
cmake --build build-m7          # one image per benchmark, below
```

**One image per benchmark**, each named after its own translation unit
(`bincv_m7_reduce.cpp` is `bincv_m7_reduce`), for the reason there is one host binary per
benchmark: `scripts/check_figure_staleness.py` ages a figure when a file its image compiles
changes, and one image compiling every kernel would age every figure on every edit.
`m7_harness.cpp` is what they share — bring-up and the command loop.

| image | command | what it measures |
|---|---|---|
| `bincv_m7_reduce` | `REDUCE` | population count without an instruction (section 1) |
| `bincv_m7_dense` | `DENSE` | dense disparity at both word types (section 2) |
| `bincv_m7_track` | `TRACK`, `SET lkwc` | the assembled feature tracker over a sequence (section 3) |
| `bincv_m7_sensor` | `SENSOR` | the sensor stage by part, DSP arms on and off (section 4) |
| `bincv_m7_lkstages` | `TRACK` | diagnostic only, never a published figure: `track` by stage |

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
logs it. At 480 MHz the core read **480.28–480.64 MHz** across the launches below: about
0.1% fast, wandering by up to 750 ppm. That wander is the ST-LINK's 8 MHz reference
rather than the method — on the internal 64 MHz oscillator the same check read 64.246–64.269
MHz over fourteen checks, a 350 ppm spread that includes the RC oscillator's own drift
([log](../../docs/reports/logs/m7-reduce-cortex-m7-resetclock-launches.log)). **Figures here
are cycle counts, which none of this touches**; milliseconds are converted at 480 MHz and
carry that 0.1%.

The first figures on this board were taken on the 64 MHz reset clock, before this path was
brought up and checked. `-DBINCV_M7_RESET_CLOCK=ON` still builds that configuration, and its
logs sit beside the 480 MHz ones. **Its cycle counts do not transfer to 480 MHz**: the
dense-disparity kernel takes 52,595,561 cycles there and 46,934,832 at 480 MHz
([log](../../docs/reports/logs/m7-dense-cortex-m7-resetclock-launches.log)), while the
population count goes the other way, 201,411 against 204,317. The two configurations differ
in flash wait states (7 at reset), bus ratio and voltage scale, and which of those moves
which count was not isolated.

## What was measured

1. **Population count without an instruction** — four ways to count the bits of a 32-bit
   word, timed inside the bulk reduction that calls them.
2. **Dense disparity at 32-bit pointer width** — the binary dense kernel on an M-profile
   core, at `uint32` against `uint64`.
3. **The assembled feature tracker** at 752×480, against the host frame by frame.
4. **The sensor stage**, by part, with and without the DSP extension.

All at 480 MHz, from the logs named under each table. Every launch returned the same cycle
count to within 0.02%: this core has no frequency scaling, no other process and no OS, so
the run-to-run scatter the host logs exist to measure is not there to measure.

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
| **A — shipped** (`__popcountsi2`) | 204,317 | 1.00× | 48 B |
| B — portable (64-bit SWAR) | 377,405 | 1.85× slower | 180 B |
| C — SWAR32 | 216,525 | 1.06× slower | 78 B |
| D — SWAR32 + `USAD8` | 181,546 | 1.13× faster | 78 B |

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

For scale: 204,317 cycles is **0.43 ms** to count a 752×480 frame at 480 MHz. The first
bring-up expected the AXI SRAM, at half the core clock, to become the limit at 480 MHz; it
did not — the count takes 1.4% more cycles than the 201,411 it takes at the reset clock,
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
([log](../../docs/reports/logs/m7-reduce-cortex-m7-launches.log); L0 is 206,169 cycles). Host
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
| **`uint32`** | 46,934,832 | **97.8** | 6,480 B |
| `uint64` | 62,584,544 | 130.4 | 6,480 B |

**`uint64` is 1.33× slower: the 64-bit guidance inverts at 32-bit pointer width**, as a
core synthesizing every 64-bit ripple from register pairs would suggest, and now as a
measurement — 1.30× at the reset clock, 1.33× at 480 MHz. `denseDisparity.hpp` carries the
width-qualified guidance. The scratch tie is geometry, not a rule: at this width both types
pack the same bytes.

For scale only: a QVGA depth map on a microcontroller in 98 ms, in 6.5 KB of scratch. There
is no comparison against anything else on this part here; this is the kernel's own cost.

## 3. The feature tracker

### What runs

The assembled tracker of [feature-tracking.md](../../docs/reports/feature-tracking.md) —
sensor stage, a 1/2/2/2 pyramid, derivatives, `goodFeaturesToTrack` whenever fewer than 60
tracks survive, pyramidal Lucas–Kanade, the cull — over EuRoC `V1_02_medium` `cam0`, 1,709
frame pairs at **752×480**: the reference sequence at its full resolution, with the reference
pipeline's parameters.

It is **one source on both sides**. `benchmark/frontend_pipeline.hpp` is the benchmark's binCV
half written against views over memory the caller provides, since containers allocate and
this part has a 4 KB heap; the host's `frontend_sequence` runs it and reproduces the
published tracking statistics exactly (82 detections, 10,279 tracks, median lifetime 11
frames, 96.4% survival), and `bincv_m7_track` runs the same source here.

One thing differs in shape, and it is what makes the tracker fit: the sensor stage takes the
frame **row by row** ([ops/medianEdgeStream.hpp](../../include/bincv/ops/medianEdgeStream.hpp)),
as a camera delivers it, instead of as a whole 8-bit frame with an 8-bit median buffer beside
it — 721,920 bytes at 752×480 that this part could not hold alongside the rest. The stream is
bit-identical to the two whole-frame calls (`tests/test_median_edge_stream.cpp`). The board
asks the host for rows and the host answers with them and their CRC-32; no transfer is
inside a timed region.

### Exact against the host, every frame

The board digests every frame's integer outputs — the binary frame, the pyramid, the
derivatives, the corners — and sends Lucas–Kanade's points back as exact float bits; the
host compares both with the same source run on x86-64
([log](../../docs/reports/logs/m7-track-cortex-m7.log)).

| | result |
|---|---|
| integer digests equal | 1,709 of 1,709 frames |
| points bit-identical | 1,709 of 1,709 frames: 0 of 158,839 points differ |
| detections, tracks observed, median lifetime | 82, 10,279, 11 frames — the published figures |

The integer digests are the guarantee: every integer stage is exact by construction. The
float agreement is an observation about what the two compilers emitted for this source, not
something either was configured to promise.

### Memory

Buffer geometry, so exact; the stack is measured by painting.

| | bytes | bank |
|---|---|---|
| pyramids, two | 152,640 | AXI SRAM |
| derivative ladders | 275,040 | AXI SRAM |
| response ring | 9,024 | AXI SRAM |
| sensor stream rows | 3,760 | D2 SRAM |
| corner candidates (20,000) | 240,000 | D2 SRAM |
| tracks | 4,200 | D2 SRAM |
| **everything the pipeline holds** | **684,664** | |
| stack, measured | 8,728 | DTCM |
| flash, the whole image | 104,252 | |

The first three rows are the 436,704 bytes the published footprint counts. The corner
candidates are the largest thing left that it does not, and they exist only because the
spacing filter runs after ranking.

### Speed

Mean cycles per frame over the sequence, at the real duty cycle (82 detections in 1,709
frames):

| stage | cycles per frame | ms at 480 MHz | share |
|---|---|---|---|
| sensor stage | 4,778,077 | 9.95 | 16.8% |
| pyramid | 480,126 | 1.00 | 1.7% |
| derivatives | 451,985 | 0.94 | 1.6% |
| detect | 1,410,239 | 2.94 | 5.0% |
| track | 21,267,343 | 44.31 | 74.9% |
| **the frame** | **28,387,772** | **59.14** | |

For scale only — there is nothing on this part to compare it with yet: a full-resolution
frame through the whole tracker in 59 ms.

### What moved it

**One image, two arms.** Lucas–Kanade's window-count arm (`impl::lkWindowCountEnabled()`, on by
default where a population count is software) against the per-row count every other target
runs — the published tracking code — over the same 60 frames, both bit-identical to the host
([off](../../docs/reports/logs/m7-track-lkwc0-cortex-m7.log),
[on](../../docs/reports/logs/m7-track-lkwc1-cortex-m7.log)):

| | per-row count | window count | |
|---|---|---|---|
| track, cycles per frame | 87,968,397 | 33,407,993 | 2.63× |
| whole frame | 94,786,325 | 40,225,885 | 2.36× |

The arm keeps per-byte partial counts down the window and collapses them once, where the
per-row count calls libgcc's SWAR sequence for every single-word count — eighty per window
row at two bits per pixel — and folds each into a `long long`. When the window is staged it
also runs the rows inside each term, so the partial counts stay in registers; and an
iteration whose displacement floors to the previous one's reuses its sums, since they depend
on nothing else, with the previous frame's two carried across every iteration. Its code is
9,822 bytes of the 104,252-byte image. Whether that is worth it is the owner's call
([#104](https://github.com/ryanhou28/bincv/issues/104)); it ships on by default meanwhile.

**The steps, in order**, each measured on the same 60 frames as it was made. The images
they ran on no longer exist, so these are a record rather than figures to re-take; the A/B
above is the committed measurement.

| step | belongs to | track, cycles per frame |
|---|---|---|
| the published code | — | 89.75M |
| the residual counted down the window | the arm | 47.82M |
| the staged covariance counted down the window | the arm | 45.24M |
| register-blocked when the window is staged | the arm | 41.48M |
| an iteration that floors to the previous displacement reuses its sums | the arm | 34.71M |
| the previous frame's two sums carried across iterations | the arm | 33.16M |

The sums reuse would apply on any target — Lucas–Kanade's ten integers are a function of
the window and the integer displacement only, and a converging iteration usually moves by
less than a pixel — but it runs only under this arm: a change that would move the x86-64 or
Pi figures is measured there in its own right, not carried in with microcontroller work.

**Where `track` still goes** — the diagnostic image over the same 60 frames, never a
figure: about 81% in the iterations (almost all of it the residual's counting), 11% the
covariance, 7% staging the window. The sums reuse cut the residual calls that count from
90,274 to 67,024.

### What this does not claim

**No comparison.** Nothing here is measured against OpenCV or any other library on this part
([#108](https://github.com/ryanhou28/bincv/issues/108)).

**One sequence**, at one resolution. At 376×240 the reference parameters break down —
spacing 33 pixels apart leaves room for fewer tracks than the 60 the lifecycle demands, so
the tracker re-detects on almost every frame — and a smaller frame needs its own parameters,
not these.

## 4. The sensor stage, with the DSP extension

On a core with no vector unit the widest arithmetic is the ARMv7E-M DSP extension's four
byte lanes in a general register. The median and the edge threshold each have an arm on it
(`__ARM_FEATURE_SIMD32`, the compiler's own macro; switchable at run time like every vector
arm): a byte min or max is `usub8` then `sel`, an absolute difference two `uqsub8`, and
`usub8` against the threshold leaves each lane's answer in its GE flag, which one `sel` and
one multiply gather into four bits.

`bincv_m7_sensor` runs every row of both parts with the arms on and off and compares them
before it times anything; 0 values differ
([log](../../docs/reports/logs/m7-sensor-cortex-m7-launches.log)). Cycles for 480 rows of 752,
median of 5:

| | portable | DSP arm | |
|---|---|---|---|
| median rows | 4,334,864 | 1,211,331 | 3.58× |
| edge rows | 8,592,403 | 2,268,840 | 3.79× |
| the stream, both together | 13,349,559 | 4,084,439 | 3.27× |

The tracker runs with them on, and its digests hold to the host's.
