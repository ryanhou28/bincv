# binCV on Cortex-M7 — STM32H753ZI

A bare-metal target for the NUCLEO-H753ZI: binCV where there is no hardware population
count and no vector unit, and 1 MB of RAM in five banks. Everything in this directory is the
harness; binCV itself is `bincv_core`, the same headers every other target compiles.

**The assembled feature tracker runs here at the reference sequence's full 752×480,
bit-identical to the x86-64 host on every frame, in 512,600 bytes — or 238,712 with its
derivatives made where they are read, 175,936 detecting on a grid** — section 3. Sparse stereo, the five-point RANSAC
and sub-pixel corner refinement run here too, each against the host's output (sections 5
to 7).

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
| `bincv_m7_track` | `TRACK`, `SET lkwc`, `SET rt` | the assembled feature tracker over a sequence (section 3) |
| `bincv_m7_track_ondemand` | `TRACK`, `SET lkwc`, `SET rt` | the same tracker, its derivatives made where they are read (section 3) |
| `bincv_m7_track_grid` | `TRACK`, `SET lkwc`, `SET rt` | the on-demand tracker detecting on a grid (section 3) |
| `bincv_m7_sensor` | `SENSOR` | the sensor stage by part, DSP arms on and off (section 4) |
| `bincv_m7_stereo` | `STEREO` | sparse stereo by stage, its arms off and on (section 5) |
| `bincv_m7_essential` | `ESSENTIAL`, `ESSENTIALDUMP` | the five-point RANSAC, scoring both ways; the models and masks in full (section 6) |
| `bincv_m7_subpix` | `SUBPIX`, `SUBPIXDUMP` | `cornerSubPix`, with and without reuse; the points in full (section 7) |
| `bincv_m7_lkstages` | `TRACK` | diagnostic only, never a published figure: `track` by stage |

`m7_reference` (a host benchmark) prints what the `*DUMP` commands print, from the same
scene sources, and `scripts/m7_dump_compare.py` holds one to the other value by value.

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

**The images replace newlib-nano's `memcpy`, `memmove` and `memset`** with word-wide ones
([string.c](string.c)). Nano builds all three as byte loops, and binCV's kernels call them: the
sensor stream copies each row it is given, and GCC compiles a kernel's own zero-filling loop
into a `memset` call — thirty-two of them in the tracker image. Against nano's, the sensor
stream is 13% faster (section 4) and dense disparity 14% (section 2): the logs before and
after are in the repository's history. A build on nano pays that, whatever it runs.

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
logs sit beside the 480 MHz ones. The two configurations differ in flash wait states (7
at reset), bus ratio and voltage scale, and their cycle counts are close but not equal: the
dense-disparity kernel takes 40,160,476 cycles there and 40,177,240 at 480 MHz
([log](../../docs/reports/logs/m7-dense-cortex-m7-resetclock-launches.log)), and the
population count 201,321 against 204,590. Which setting moves which count was not
isolated. (With newlib-nano's byte-loop `memset`, before `string.c`, the dense counts were
52,595,561 and 46,934,832 — 12% apart — so a figure that leans on the C library inherits
its sensitivity to the bus.)

## What was measured

1. **Population count without an instruction** — four ways to count the bits of a 32-bit
   word, timed inside the bulk reduction that calls them.
2. **Dense disparity at 32-bit pointer width** — the binary dense kernel on an M-profile
   core, at `uint32` against `uint64`.
3. **The assembled feature tracker** at 752×480, against the host frame by frame, with its
   derivatives stored and made where they are read, and detecting on a grid.
4. **The sensor stage**, by part, with and without the DSP extension.
5. **Sparse stereo**: BRIEF, the descriptor stage and the window refinement.
6. **The five-point essential matrix under RANSAC.**
7. **Sub-pixel corner refinement.**

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
| **A — shipped** (`__popcountsi2`) | 204,590 | 1.00× | 48 B |
| B — portable (64-bit SWAR) | 377,909 | 1.85× slower | 180 B |
| C — SWAR32 | 216,032 | 1.06× slower | 78 B |
| D — SWAR32 + `USAD8` | 182,058 | 1.12× faster | 78 B |

**Decision at the time: not adopted; arm A stays.** D is the fastest, and the library kept
the slower arm on two grounds: a 1.12× gain on a microbenchmark that counts bits and does
nothing else, and a chip-specific path that must stay bit-exact with the portable one on
every future change with nothing in CI executing it. **The second ground no longer
stands** — an M-profile arm is now judged on its measurement, with its correctness proven
on the board, rather than declined because CI cannot run it — so the decision is open
again. It belongs with the Lucas–Kanade work below, where the counts that remain are
single words, rather than with this loop.

Three things it settled:

- **The call was not the cost.** C removes A's per-word call and is slower by 6%: the M7
  predicts the call perfectly and dual-issues around it, while the inlined SWAR adds
  register pressure the call did not.
- **`USAD8` is the only thing that helped**, and only by removing two shift-adds from the
  tail. There is no cheap win in the per-word counter on this core.
- **B is 1.85× slower than what ships.** `popcountWordPortable` is not what this target
  compiles to (GCC calls libgcc's `__popcountsi2`), and `reduce.hpp` says so.

For scale: 204,590 cycles is **0.43 ms** to count a 752×480 frame at 480 MHz. The first
bring-up expected the AXI SRAM, at half the core clock, to become the limit at 480 MHz; it
did not — the count takes 1.6% more cycles than the 201,321 it takes at the reset clock,
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
| L1 | 1.17× slower | 0.93× | 0.60× |
| **L2** | **0.42×** | **0.12×** | 1.06× |
| L3 | 0.43× | 0.11× | 0.96× |

M7 figures are cycle ratios at 480 MHz, ten launches that agreed exactly
([log](../../docs/reports/logs/m7-reduce-cortex-m7-launches.log); L0 is 205,904 cycles). Host
figures are medians whose run-to-run spread was 49–107%, so only the order-of-magnitude
differences there carry, which is all that is claimed from them.

**L2 is 2.4× faster on the M7 and roughly 8× on x86 without POPCNT**, in plain C++ any
target compiles. The split is by family, not by part: L2 wins wherever the count is a
software SWAR and is neutral-to-slightly-worse where the hardware has an instruction, so
gating it on "no hardware population count" leaves x86-with-POPCNT and aarch64 on exactly
the code they have today. L3 does not beat L2, so what remained was never in the body of
the count. L1 is target-dependent (1.67× faster on x86 with POPCNT, 17% slower on the M7)
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
launches, which agreed to within two cycles ([log](../../docs/reports/logs/m7-dense-cortex-m7-launches.log)).
Both word types exact on the supported region, and the two maps byte-identical.

| word type | cycles | ms at 480 MHz | scratch |
|---|---|---|---|
| **`uint32`** | 40,177,240 | **83.7** | 6,480 B |
| `uint64` | 56,819,546 | 118.4 | 6,480 B |

**`uint64` is 1.41× slower: the 64-bit guidance inverts at 32-bit pointer width**, as a
core synthesizing every 64-bit ripple from register pairs would suggest, and now as a
measurement — 1.41× at the reset clock too. `denseDisparity.hpp` carries the
width-qualified guidance. The scratch tie is geometry, not a rule: at this width both types
pack the same bytes.

For scale only: a QVGA depth map on a microcontroller in 84 ms, in 6.5 KB of scratch. There
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
this part has a 4 KB heap; the host's `frontend_sequence` runs it, and `bincv_m7_track` runs
the same source here.

Two things differ in shape from the published benchmark, and they are what make the tracker
fit:

- **The sensor stage takes the frame row by row**
  ([ops/medianEdgeStream.hpp](../../include/bincv/ops/medianEdgeStream.hpp)), as a camera
  delivers it, instead of as a whole 8-bit frame with an 8-bit median buffer beside it —
  721,920 bytes at 752×480 that this part could not hold alongside the rest. The stream is
  bit-identical to the two whole-frame calls (`tests/test_median_edge_stream.cpp`).
- **Detection collects candidates into a pool of 8,192** (64 KB) where the benchmark gives
  it an array of 20,000 (240 KB). The bounded-pool `goodFeaturesToTrackStreaming` returns
  exactly the corners an unbounded pool would, sweeping the frame again when the spacing walk
  outruns the pool. The benchmark's array does truncate one of its 82 detections, so where
  it reports 10,279 tracks and 96.4% survival this reports the untruncated answer:
  10,286 tracks, median lifetime 11 frames, 96.5% survival.

The board asks the host for rows and the host answers with them and their CRC-32; no
transfer is inside a timed region.

### Exact against the host, every frame

The board digests every frame's integer outputs — the binary frame, the pyramid, the
derivatives, the corners — and sends Lucas–Kanade's points back as exact float bits; the
host compares both with the same source run on x86-64
([log](../../docs/reports/logs/m7-track-cortex-m7.log)).

| | result |
|---|---|
| integer digests equal | 1,709 of 1,709 frames |
| points bit-identical | 1,709 of 1,709 frames: 0 of 159,510 points differ |
| detections, tracks observed, median lifetime | 82, 10,286, 11 frames |

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
| corner pool (8,192) and the kept corners (200) | 67,936 | D2 SRAM |
| tracks | 4,200 | D2 SRAM |
| **everything the pipeline holds** | **512,600** | |
| stack, measured | 8,428 | DTCM |
| flash, the whole image | 102,380 | |

The first three rows are the 436,704 bytes the published footprint counts. The pool's
size is a cost in sweeps, never in the answer: 34 of the 82 detections outran it and
swept again, at most three times.

### Speed

Mean cycles per frame over the sequence, at the real duty cycle (82 detections in 1,709
frames):

| stage | cycles per frame | ms at 480 MHz | share |
|---|---|---|---|
| sensor stage | 4,265,948 | 8.89 | 23.2% |
| pyramid | 480,215 | 1.00 | 2.6% |
| derivatives | 453,340 | 0.94 | 2.5% |
| detect | 1,189,124 | 2.48 | 6.5% |
| track | 12,023,667 | 25.05 | 65.3% |
| **the frame** | **18,412,296** | **38.36** | |

For scale only — there is nothing on this part to compare it with yet: a full-resolution
frame through the whole tracker in 38 ms.

### What moved it

**One image, two arms.** Lucas–Kanade's window-count arm (`impl::lkWindowCountEnabled()`, on by
default where a population count is software) against the per-row count every other target
runs — the published tracking code — over the same 60 frames, both bit-identical to the host
([off](../../docs/reports/logs/m7-track-lkwc0-cortex-m7.log),
[on](../../docs/reports/logs/m7-track-lkwc1-cortex-m7.log)):

| | per-row count | window count | |
|---|---|---|---|
| track, cycles per frame | 89,264,119 | 19,536,428 | 4.57× |
| whole frame | 96,098,291 | 26,369,597 | 3.64× |

Its price is flash: **13,556 bytes**, measured by building the same image without it —
0.7% of this part's 2 MB, and no RAM: the arm adds no buffer. It is on by default on every
core whose population count is software
([#104](https://github.com/ryanhou28/bincv/issues/104)).

What the arm does, against the per-row count's eighty libgcc calls per window row at two bits
per pixel:

- **It counts down the window.** Per-byte partial counts are carried in a register and
  collapsed once per fold, instead of each single-word count being collapsed and widened.
- **It writes the signed gradient in two's complement**, N + 1 planes from the magnitude
  and the sign, so a term costs N(N + 1) counts where `pos − neg` costs 2N² — six a row
  at N = 2, not eight. Counts of one weight are added as nibbles before they are widened,
  so a term carries N + 1 lanes and they stay in registers.
- **It holds the count's masks in registers**, so each `(x >> s) & m` is one Thumb-2
  instruction.
- **It counts less.** An iteration whose displacement floors to the previous one's reuses
  its sums; the previous frame's two sums are carried across every iteration; and when the
  displacement moves by a pixel the taps the two positions share are carried too — every
  move between counted iterations on this sequence was at most a pixel per axis.
- **It reads the next frame straight down a column**, one row pointer and bit offset per
  plane for the whole window.
- The staged covariance gets the same treatment: the selector-signed gradient in two's
  complement, the squares merged by weight.

**The steps, in order**, each measured on the same 60 frames as it was made. The images
they ran on no longer exist, so these are a record rather than figures to re-take; the A/B
above is the committed measurement. On this part a figure moves by about 1% with unrelated
code — where the linker puts things — so a step smaller than that is not resolved.

| step | track, cycles per frame |
|---|---|
| the published code | 89.75M |
| the residual counted down the window | 47.82M |
| the staged covariance counted down the window | 45.24M |
| register-blocked when the window is staged | 41.48M |
| an iteration that floors to the previous displacement reuses its sums | 34.71M |
| the previous frame's two sums carried across iterations | 33.16M |
| the fold loops unrolled, the term chosen once outside the rows | 33.74M |
| the gradient in two's complement, counts merged by weight before widening | 26.76M |
| the count's masks held in registers | 24.84M |
| tap sums carried across a one-pixel move | 23.18M |
| the term count out of line, staged windows only (−14 KB of flash) | 23.55M |
| the staged covariance in two's complement | 22.24M |
| taps read down a column | 19.94M |
| word-wide `memcpy`/`memset` in the image (above) | 19.50M |

The reuse would apply on any target — Lucas–Kanade's ten integers are a function of the
window and the integer displacement only — but it runs only under this arm, and the arm is
compiled only where it is the default: a change that would move the x86-64 or Pi figures is
measured there in its own right, not carried in with microcontroller work.

**Detection** runs three changes of its own. The 3×3 response's square root comes from a
table on M-profile (`BINCV_CORNER_ROOT_TABLE`, switchable with `SET rt`): a 3×3 window's
counts are 0 to 9, so the discriminant has a hundred values, and the table holds their
correctly rounded roots inside `minEigenValue`'s own double expression — the same float.
It costs 2,984 bytes of flash, measured the same way, and over the 60 frames above, whose
one detection sweeps three times, it takes `detect` from 2,401,976 cycles a frame to
1,634,107 — 1.47×, the frame 2.9%
([off](../../docs/reports/logs/m7-track-rt0-cortex-m7.log),
[on](../../docs/reports/logs/m7-track-rt1-cortex-m7.log)). The scalar suppression compares the responses' bit patterns
rather than the floats, which order the same way because a response is never negative, and
which a Cortex-M branches on without moving flags out of the FPU. And the pool orders its
candidates by one 64-bit compare. Over the sequence `detect` went from 1,410,239 cycles a
frame to the 1,189,124 above, while its pool shrank from 240 KB to 64 KB and 34 of the 82
detections sweep more than once.

### Derivatives where they are read

The derivative ladders are 275,040 of the 512,600 bytes, and nothing reads them but the
tracker's window staging and the detector's sweep. Both can make what they read instead:
`LKLevelOnDemand` ([ops/opticalFlowOnDemand.hpp](../../include/bincv/ops/opticalFlowOnDemand.hpp))
stages each window's derivative words from the previous frame as it stages the window, and
`goodFeaturesToTrackStreaming` over a `GradientSource` derives three gradient rows as its
sweep reaches them. The stored form stays the default; a caller picks by what it constructs.
`bincv_m7_track_ondemand` is the tracker built that way, over the same 1,709 frames
([log](../../docs/reports/logs/m7-track-ondemand-cortex-m7.log)):

| | stored derivatives | made where read | |
|---|---|---|---|
| everything the pipeline holds, bytes | 512,600 | 238,712 | −53% |
| derivatives, cycles per frame | 453,340 | 0 | |
| detect | 1,189,124 | 1,214,612 | |
| track | 12,023,667 | 12,090,307 | |
| **the frame** | **18,412,296** | **18,078,375** | |
| integer digests, points | equal, bit-identical | equal, bit-identical | |

Making the derivatives where they are read costs tracking 0.6% and detection 2.1% — 92,000
cycles a frame between them — against the 453,340 the derivative stage no longer takes, so
the frame is no slower for holding less than half: 1,152 bytes of gradient rows replace
275,040 of ladders, and the measured stack is smaller too, 7,716 bytes against 8,428. The
window must stage — a word wide and `kStagedMaxRows` tall — or every point is reported lost.

### Detection on a grid

`bincv_m7_track_grid` is the on-demand tracker detecting with `gridCorners`
([ops/gridCorners.hpp](../../include/bincv/ops/gridCorners.hpp)) instead of
`goodFeaturesToTrackStreaming`: the reference's detection gridded — one corner in each cell
the reference spacing wide (34 pixels), the same quality level, window and cap of 200
(`frontend::gridParams`) — over the same 1,709 frames, with the same lifecycle
([log](../../docs/reports/logs/m7-track-grid-cortex-m7.log)). It is a different detector, so
it is a different tracker; the table is what a caller choosing between them weighs:

| | `goodFeaturesToTrackStreaming` | `gridCorners` |
|---|---|---|
| everything the pipeline holds, bytes | 238,712 | **175,936** |
| detections | 82 | 51 |
| cycles a detection | 25.3 M, 1 to 3 sweeps | **15.5 M, one sweep** |
| detect, cycles per frame | 1,214,612 | 462,814 |
| points tracked over the sequence | 159,510 | 199,970 |
| track, cycles per frame | 12,090,307 | 15,039,139 |
| **the frame** | **18,078,375** (37.7 ms) | **20,257,549** (42.2 ms) |
| tracks observed, median lifetime, survival | 10,286, 11 frames, 96.5% | 9,716, 14 frames, 96.6% |
| integer digests, points | equal, bit-identical | equal, bit-identical |

The grid drops the 64 KB pool for 2,760 bytes of slots, never sweeps twice, and its tracks
live longer, so it detects less often. Its frame costs more because each detection fills the
cap — 200 corners where the spacing walk often finds fewer — so a quarter more points are
tracked; per point tracked the cost is the same, 129 thousand cycles either way.

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
| median rows | 4,334,880 | 1,211,778 | 3.58× |
| edge rows | 8,953,198 | 2,268,987 | 3.95× |
| the stream, both together | 12,803,171 | 3,555,291 | 3.60× |

The tracker runs with them on, and its digests hold to the host's.

## 5. Sparse stereo

### What runs

The sparse rectified matcher of [ops/stereo.hpp](../../include/bincv/ops/stereo.hpp) — BRIEF
descriptors, the descriptor stage along the epipolar row, the Hamming-window refinement with
its parabola — over the host benchmark's synthetic pair: a smoothed random 752×480 frame
and the same frame shifted 21 pixels, 500 keypoints a side, 256-bit descriptors, disparities
0 to 64, 11×11 windows. The board builds the pair from the host's own source
([benchmark/stereo_scene.hpp](../../benchmark/stereo_scene.hpp)) a row at a time, so one 8-bit
frame serves both sides: described and binarized as the left, then shifted in place into
the right.

### Exact against the host

Both frames' bits, both descriptor sets and the 500 matches digest equal to the x86-64
host's `stereo_benchmark` ([log](../../docs/reports/logs/m7-stereo-cortex-m7-launches.log)).
The matcher is integers until the parabola, whose double comes from three integer scores
with no multiply-add to fuse.

### Speed

Cycles, median of 5:

| | published | with the M-profile arms | |
|---|---|---|---|
| BRIEF, one frame | 1,710,851 | | |
| descriptor stage | 4,740,054 | 1,642,506 | 2.89× |
| refinement stage | 8,601,493 | 1,369,919 | 6.28× |

Matching a frame pair — both stages, one call — takes 3,014,812 cycles, 6.3 ms, where the
published stages take 13.3 million between them.

**Refinement reads its windows once** (`BINCV_STEREO_WINDOW_COUNT`, where a population count
is software; `impl::stereoWindowCountEnabled`). The published stage scores eleven
disparities a keypoint by re-extracting both frames' rows for each and counting every word
whole — with libgcc's `__popcountsi2`, on this core. The arm stages the left window, and a
right strip holding the window at every disparity it can score, one word a row each; a score
is then a shift, an XOR and a count whose byte lanes are added down the window and collapsed
once. The window and the strip must lie inside their frames and the strip fit 32 bits —
`winW + 2 × refineRadius + 2` — and a keypoint that does not takes the published path. At
radius 12 no strip fits, and the two time the same, 20,134,416 cycles against 20,140,382:
the check that the arm runs where it is said to. 1,137 bytes of flash.

**The descriptor stage tests rows on bit patterns** (`BINCV_STEREO_INTEGER_ROW_GATE`,
M-profile; `impl::stereoIntegerRowGateEnabled`). The published stage tests all 250,000
left–right pairs here with a float subtraction and two comparisons, and on this core a float
comparison reaches a branch only through the FPU's flags, while the two-sided test's branch
goes either way at random: 19 cycles a pair. With `yL` fixed, the right rows the test passes
are one interval of floats, because `y − yL` rounds monotonically in `y`, and so one interval
of their bit patterns read as ordered integers. The arm finds that interval once per left
keypoint — from where float arithmetic puts its ends, `yL ∓ rowTol`, galloping outward and
evaluating the published comparison itself — and a pair becomes one integer range check on
the bits as they lie in memory, which on one side of zero are already in order. A NaN passes
the published test and lies in no interval, so a call holding one, or a left keypoint that
is not finite, takes the published path. 6.6 cycles a pair, and 2,209 bytes of flash.

BRIEF is 12 instructions a tested pair of pixels and takes 13 cycles a pair: its samples are
byte loads scattered over a 31×31 patch of the 8-bit frame, which sits in AXI SRAM because
that is where the caller put it. Nothing here was changed in it.

Both stages hold nothing but their inputs and outputs; the refinement arm stages 248 bytes
on the stack.

## 6. The five-point essential matrix under RANSAC

### What runs

`findEssentialMat` ([ops/essential.hpp](../../include/bincv/ops/essential.hpp)) — and
`findEssentialMatRealRoots`, the same estimation over a minimal solver built for a part like
this one — over three scenes of
[benchmark/essential_scene.hpp](../../benchmark/essential_scene.hpp): points seen by two
cameras a known rotation and translation apart, in normalised coordinates, with a planted
share of outliers — 200 and 500 correspondences at 30%, 1,000 at 50% — threshold 0.002 and a
cap of 500 iterations. The scene is built without trigonometry, so the board and the host
start from the same bits. Double precision throughout, on the M7's double-precision FPU; the
solver's working set is stack, the inlier flags two bits per correspondence of caller
scratch.

### Against the host

| | 200 at 30% | 500 at 30% | 1,000 at 50% |
|---|---|---|---|
| iterations, inliers — both solvers | 26, 140 | 25, 352 | 134, 508 |
| inlier mask — both solvers | identical | identical | identical |
| the model, largest difference from the host's | 5.6e-15 | 4.5e-12 | 4.8e-11 |
| the same, real-root solver | 6.2e-15 | 4.5e-12 | 4.8e-11 |

**The models differ in their last bits, and the reason is the compiler, not the kernel.**
GCC's default `-ffp-contract=fast` fuses the solvers' multiply-adds into the M7's
`vfma.f64`, and the x86-64 baseline has no FMA to fuse them into. Built with
`-ffp-contract=off`, both solvers' models are the host's to the bit in all three scenes, as
is everything else — newlib's `cos`, `sin` and `log` gave glibc's answers here
([default](../../docs/reports/logs/m7-essential-cortex-m7-launches.log),
[contraction off](../../docs/reports/logs/m7-essential-nofpcontract-cortex-m7-launches.log);
`scripts/m7_dump_compare.py` against `m7_reference essential --dump`).

### Speed

Cycles for the whole estimation, median of 3:

| | 200 at 30% | 500 at 30% | 1,000 at 50% |
|---|---|---|---|
| published, scoring by the residual | 78,473,363 | 139,759,973 | 374,341,838 |
| published, the squared test | 78,139,173 | 138,909,783 | 366,574,063 |
| published, the float scorer — this core's default | 76,968,176 | 135,939,432 | 336,601,925 |
| **real-root solver**, the float scorer | **14,422,282** | **19,643,874** | **147,028,761** |
| ms at 480 MHz, the defaults: published → real-root | 160.4 → **30.0** | 283.2 → **40.9** | 701.3 → **306.3** |

And their parts, from the same log: a five-point solve, over 200 samples of the last scene
drawn five at a time, costs 3,385,140 cycles published and **419,682** real-root — **8.07×**;
scoring costs 238 cycles a correspondence by the residual, 191 by the squared test and 143 by
the float scorer.

**The published solver's cost is its root finder.** The solve's cost moves with the sample,
because its root finder iterates until it converges, 92 to 208 passes a solve on these
scenes. On the host's instruction counts over the three scenes the solver is 73% of the
estimation and scoring 23%, and inside the solver the Aberth–Ehrlich iteration that finds
the degree-10 polynomial's roots is 80% — all ten roots, complex, to a step of 1e-14, with
two double divides per pair of roots per pass — and the Jacobi eigendecomposition that
recovers each root's `(x, y)` is 15%.

### A solver built for the part

`fivePointEssentialRealRoots` shares the published solver's system — the nullspace, the ten
cubics, `det M(z)` from the same eleven nodes, one source for both — and replaces the two
steps that are nearly all of its cost:

- **Only the real roots, isolated between the derivative's.** Between two consecutive real
  roots of `p'`, `p` is monotone, so it holds one root there exactly when it changes sign.
  The highest derivative is linear; each lower one's roots are bracketed by the one above,
  and polished by Newton's method held inside the bracket. About 10 n² operations for
  degree n, where the simultaneous iteration costs that per pass. A tangent root, which
  changes no sign, is looked for at a root of `p'` and accepted by the published finder's
  own near-real test, translated.
- **Each root's `(x, y)` from one elimination** with complete pivoting on `M(z)` itself,
  instead of an eigendecomposition of `M^T M`, which squares the condition number.
- Its Chebyshev nodes are tabulated, so it links no trigonometry.

**Its accuracy is its own, and it is held to every bound the published solver is held to.**
Every accuracy case in `tests/test_essential.cpp` runs for both solvers at the same bounds,
and they return the same figures: the planted E recovered on 196 of 200 samples, 4.70
solutions a call, every solution epipolar; 25 of 25 scenes cleanly separated; inlier counts
within 5% of `cv::findEssentialMat` on 12 of 12 scenes and on three degenerate geometries;
epipolar error under noise 0.000409 against OpenCV's 0.000392. Solution by solution, the two
agree wherever the problem is well conditioned — every published solution within 1e-10 of
satisfying the essential constraints has a real-root counterpart within 4.1e-9 on x86-64
and 1.5e-7 on aarch64 — and
differ, both inaccurate, where it is not: 2.3% of solutions on planted poses
(`Essential.RealRootSolverAgreesWhereWellConditioned`). On the board's three scenes the two
find the same inliers in the same iterations.

Its stack is the published solver's, 5,512 bytes on this core — the peak is the cubic
construction the two share — and its code 20,896 bytes against 21,316, with no `cos` or
`sin` behind it. The published solver's own code is untouched: it compiles to the same
instructions as before on x86-64 and aarch64.

**With the solve cheap, scoring is what is left** at large point counts: at 1,000
correspondences about 81 of the real-root estimation's 147 million cycles are scoring — 134
iterations, about 4.2 hypotheses each (848 solutions in the 200 samples above), 1,000
correspondences, 143 cycles a test — and 56 million the solves. At 200 the solves are three
quarters of it.

**The squared test** (`BINCV_RANSAC_SQUARED_TEST`, M-profile, `impl::ransacSquaredTestEnabled`)
decides each point's `residual < threshold` on squares wherever the squares decide it —
outside a band of 2⁻²⁰ either side of the threshold, eight times the float rounding the
residual takes — and takes the square root and the divide, two of the slowest instructions
this core has, only inside it. Every decision is the published one: the masks, the
iterations and the models are identical with it on and off, and
`Essential.SquaredTestDecidesAsTheResidual` holds it to the residual at thresholds placed on
each residual and one float either side. Inside the band the residual is the published one
to the bit only if the compiler fuses the two copies of it alike, and on this board it does:
both are nine fused multiply-adds, seven multiplies and six adds.

**The float scorer** stands in front of it (same switch, and
`impl::ransacFloatScorerEnabled`): per hypothesis E goes to single precision once, and per
correspondence the residual's numerator and denominator are computed in float beside bounds
on their error — every float rounding, fused or not, and E's own: `12 u m L1 L2` on the
numerator and `36 u m² (L1² + L2²)` on the denominator, with `m` the largest entry of E,
`L = |x| + |y| + 1` and `u = 2⁻²⁴`. A correspondence whose whole interval lies inside the
threshold, with 2⁻¹⁸ to spare, is an inlier however the double arithmetic would round; one
wholly outside is an outlier; anything else — and anything outside the range where the
products stay finite and normal — goes to the squared test. So every decision is still the
residual's: the tests hold it there at thresholds on each residual and one float either side,
at coordinates 300 times larger, and on random matrices at the points where only the
denominator vanishes, which is the case its bound exists for and which an essential matrix
never presents. 143 cycles a correspondence against 191.

Together the two are worth what scoring is worth — 1.02× at 200 correspondences and 1.11×
at 1,000, on the published solver — for 1,914 bytes of flash.

## 7. Sub-pixel corner refinement

### What runs

`cornerSubPix` ([ops/subpix.hpp](../../include/bincv/ops/subpix.hpp)) over the 200 corners
`goodFeaturesToTrackStreaming` finds in the stereo scene's left frame
([benchmark/subpix_scene.hpp](../../benchmark/subpix_scene.hpp)), from the binarized
frame's 1-bit signed derivatives: 11×11 window, 40 iterations, ε 0.001. The scene is random
texture, so the derivatives are dense — 84 of a window's 121 pixels carry a gradient — and
that is the operation's expensive case: its word-wise skip of flat pixels has little to skip.

### Against the host

All 200 refined points are the host's to the bit, with contraction on and off
([log](../../docs/reports/logs/m7-subpix-cortex-m7-launches.log)).

### Speed

Cycles for the 200 corners, median of 5:

| | cycles | per corner | |
|---|---|---|---|
| every iteration's sums computed (published) | 7,588,206 | 37,941 | |
| an anchor's sums reused | 5,588,056 | 27,940 | 1.36× |

**An iteration that anchors where the previous one did reuses its sums**
(`BINCV_SUBPIX_REUSE`, M-profile, `impl::subPixReuseEnabled`). The window, its weights and
its offsets all hang off the rounded position, so the five sums are a function of the anchor
alone, and the step that proves convergence is nearly always one that moved less than half
a pixel: here 192 of the 688 window sums repeat their predecessor's anchor, one for each
corner that converged. With no zero zone every weight is at least e⁻², so the per-pixel test
for a zero weight is not made either. Both are exact on any build — nothing is recomputed
differently, only not recomputed — and `SubPix.AnchorReuseIsBitExact` holds them to the
published loop over 180 parameter sets. 205 bytes of flash.

What remains is the double-precision arithmetic of each pixel with a gradient: about 50
instructions, seven multiplies and two fused multiply-adds among them. Rewriting it could
change what GCC fuses, and so the answer's last bits, and is not done.

The Gaussian mask is a stack buffer sized for the largest window, 7,688 bytes whatever the
window is — half of this harness's 16 KB stack.

### What sections 5 to 7 do not claim

**One synthetic scene each**, chosen so the board can build it from the host's own source,
which is what lets its output be held to the host's. They price kernels, not a pipeline: a frontend that calls them sets its own
keypoint counts, windows and thresholds, and the descriptor stage's cost in particular grows
with the product of the two sides' keypoint counts. **No comparison** against another library
on this part ([#108](https://github.com/ryanhou28/bincv/issues/108)).
