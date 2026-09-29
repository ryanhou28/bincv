# Limits

Where binCV ties, loses, or stops paying at all. Every figure comes from the same committed
benchmarks as the rest of the reports. The useful question about a library like this one is
not "how fast can it be" but "when does the idea stop working", and it has four answers.

Every table reads the same way: **both sides' measured times in the unit the column names,
with the ratio beside them.** A row where binCV's number is the larger one — and the ratio
under 1.00× — is a row binCV lost, which is most of this page. **x86-64 and aarch64 are
separate columns** and are never averaged.

**Every x86-64 figure here is the median of thirty pinned launches**, with the bootstrap 95%
interval those thirty put around each ratio; aarch64 is the median of ten, with the same
interval. Times are quoted to four significant figures, ratios and intervals to three.

## 1. At eight bits per pixel, the idea is gone

binCV wins by not paying for bits it does not use. At eight bits per pixel there are none to
skip, and both sides store a byte.

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

`pyrDown`, 640×480 → 320×240, against `cv::pyrDown` on `CV_8U` at one thread. Bold marks
the shipped configuration:

| arm | x86-64 (µs) | x86-64 ratio | aarch64 (µs) | aarch64 ratio |
|---|---|---|---|---|
| `cv::pyrDown`, `CV_8U` (the denominator) | 47.70 | — | 516.5 | — |
| **binCV `BOX_2x2`, 1 bit in → 3 bits out (shipped)** | 30.70 | 1.56× [1.54, 1.60] | 93.8 | 5.51× [5.48, 5.55] |
| binCV `GAUSSIAN_5x5`, 1 → 3 | 185.0 | 0.259× [0.255, 0.266] | 599.1 | 0.862× [0.858, 0.868] |
| binCV `BOX_2x2`, 8 → 8 | 651.9 | 0.0746× [0.0733, 0.0760] | 2572 | 0.201× [0.200, 0.202] |
| binCV `GAUSSIAN_5x5`, 8 → 8 (`cv::pyrDown`'s shape) | 2040 | 0.0235× [0.0233, 0.0242] | 7360 | 0.0701× [0.0698, 0.0706] |

An `8 → 8` call is **correct, not fast**, and it is documented that way rather than hidden.
The structural reason is accumulator width: a bit-sliced filter needs enough accumulator
planes to hold the weighted sum of its inputs, so the plane count — and with it the work per
output pixel — grows with the input bit depth. At one bit there is almost nothing to
accumulate; at eight, the bit-sliced form is doing by hand what a byte kernel's vector unit
does in one instruction. Past a certain depth that is simply the better machine.

## 2. The crossover is real, and it moves with the architecture

The same geometry across input and output bit widths, through the generic filtered path
(`pyrDownFiltered<Box2x2, NOut, NIn>`), one process per arm — timing every width in one
process leaves all of their planes resident and inflates the cheap arms. Each machine has its
own `cv::pyrDown` denominator in the first row:

| arm | x86-64 (µs) | x86-64 ratio | aarch64 (µs) | aarch64 ratio |
|---|---|---|---|---|
| `cv::pyrDown`, `CV_8U` (the denominator) | 47.85 | — | 517.8 | — |
| box filter, 1 → 3 (generic filtered path — the shipped kernel is the row in §1) | 32.20 | 1.47× [1.46, 1.50] | 275.9 | 1.88× [1.87, 1.92] |
| box filter, 1 → 1 | 79.05 | 0.605× [0.597, 0.617] | 319.7 | 1.63× [1.61, 1.70] |
| box filter, 2 → 2 | 62.35 | 0.772× [0.758, 0.779] | 205.0 | 2.53× [2.52, 2.59] |
| box filter, 3 → 3 | 94.30 | 0.508× [0.495, 0.516] | 306.5 | 1.69× [1.69, 1.73] |
| box filter, 4 → 4 | 132.7 | 0.358× [0.352, 0.364] | 444.2 | 1.17× [1.16, 1.20] |
| box filter, 5 → 5 | 176.4 | 0.272× [0.268, 0.277] | 647.3 | 0.801× [0.799, 0.820] |
| box filter, 8 → 8 | 645.3 | 0.0736× [0.0723, 0.0753] | 2575 | 0.201× [0.201, 0.205] |

**The aarch64 column of this table is provisional.** On aarch64 this benchmark reads the
generic `1 → 3` call at 275.9 µs where `pyrfilter_benchmark` reads the identical call at
116.1 µs, on the same commit (`80ff0a8`, on `main` as `086428c`;
[log](logs/pyrfilter-aarch64-launches.log)). On x86-64 the two benchmarks agree (32.20
against 32.40 µs), and the `8 → 8` arm agrees on the device too (2575 against 2572 µs). The
inflation is on the cheap arms, which is exactly what moves the crossover point, so the
device column stands only until it is re-taken, and the crossover it implies may understate
binCV.

**The crossover is not a property of the algorithm — it moves by several bits between the two
machines.** As this table reads, the bit-sliced box filter on the reference device stays
ahead of `cv::pyrDown` through four bits per pixel and crosses between four and five. On
x86-64 the only shape that beats it is the shipped one, and even `1 → 1` is behind at 0.605×.

The reason is the denominator, not binCV: OpenCV's x86-64 build dispatches at run time over
SSE4.1 through AVX-512 code paths (its build line at the top of every x86-64 log reads
`Dispatched code generation: SSE4_1 SSE4_2 FP16 AVX AVX2 AVX512_SKX`) and its pyramid is
very good; its aarch64 pyramid is relatively weaker against the same machine. binCV's own
times scale about as expected between the two platforms; OpenCV's do not. This is the single
most important caveat in these reports, and it generalises — **a ratio measured on a desktop
is not a ratio on a deployment part, in either direction.**

## 3. A vectorised byte kernel is a real competitor

Bit packing gives a `uint32_t` thirty-two pixels; an AVX2 register of bytes holds exactly
thirty-two pixels too. Packing alone therefore buys nothing until the boolean algebra also
moves into a vector register — so where OpenCV has already done that work binCV ties or loses,
and where it has done less the same binCV code wins.

640×480 except the `cv::FAST` row (752×480, the wide-image entry point's own frame):

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio | why |
|---|---|---|---|---|---|---|---|---|
| FAST, wide-image entry point | `cv::FAST` | 0.359 ms | 0.345 ms | 1.04× [1.03, 1.05] | 2.910 ms | 3.025 ms | 0.962× [0.961, 0.963] | parity with a mature vectorised kernel |
| `erode`, 5×5 ellipse | `cv::erode` | 0.2238 ns/px | 0.6985 ns/px | 0.319× [0.318, 0.323] | 1.852 ns/px | 3.596 ns/px | 0.514× [0.510, 0.522] | a non-separable element costs one shifted-OR per set element |
| `erode`, `BORDER_REPLICATE` | `cv::erode` | 0.09870 ns/px | 0.1489 ns/px | 0.666× [0.659, 0.672] | 0.6985 ns/px | 0.9295 ns/px | 0.752× [0.741, 0.769] | a rim pass `BORDER_CONSTANT` does not need |
| `erode`, `BORDER_REFLECT_101` | `cv::erode` | 0.09931 ns/px | 0.1553 ns/px | 0.635× [0.628, 0.645] | 0.7002 ns/px | 0.9437 ns/px | 0.742× [0.727, 0.759] | the same |
| `erode`, 3×3 rect | `cv::erode` | 0.1013 ns/px | 0.09595 ns/px | 1.05× [1.04, 1.07] | 0.7360 ns/px | 0.7219 ns/px | 1.02× [0.991, 1.04] | a dead heat |
| `countNonZero` | `cv::countNonZero` | 0.01501 ns/px | 0.009270 ns/px | 1.62× [1.61, 1.63] | 0.1692 ns/px | 0.06365 ns/px | 2.66× [2.62, 2.67] | OpenCV is bandwidth-bound; binCV moves an eighth of the data at a fifth of the rate |

Parity on FAST ships as parity. A caller who is holding bytes should not be told to pack them
first, and for that caller the honest answer is that binCV costs nothing to adopt and gains
nothing either. The [bit-plane overload](features.md#fast) is where the thesis actually
applies, and it is 1.65× on x86-64 and 2.37× on the device.

`goodFeaturesToTrack` is not on this list. Against stock `cv::goodFeaturesToTrack` it reads
1.38× on x86-64 and 2.42× on the device, and the row is in
[features.md](features.md#corner-detection). Its margin, like FAST's, is wider against the
denominator doing less vector work.

## 4. A footprint win is not a speed win

Eight times less data does not make a compute-bound kernel faster, and Lucas–Kanade is
compute-bound. Two sweeps at one level with a 31×31 window, binCV against itself — no OpenCV
arm, so no ratio.

**The frame-size sweep is the one that isolates the question.** The point count is fixed at
140, so the compute is identical and only the data grows:

| frame | input, KiB at 1 bit | time, x86-64 (µs/point) | time, aarch64 (µs/point) |
|---|---|---|---|
| 320×240 | 9.4 | 4.658 | 25.64 |
| 640×480 | 37.5 | 4.141 | 23.13 |
| 1280×960 | 150.0 | 4.792 | 26.99 |
| 1920×1440 | 337.5 | 4.640 | 27.23 |

Thirty-six times more data moves the per-point cost by **0.4%** on x86-64 — 4.658 µs/point at
320×240 against 4.640 at 1920×1440, which is no change at all — and **6%** on the device,
which has a 1 MiB shared L2 where a residency effect would show most clearly if there were
one. A 31×31 window is 120 bytes at one bit per pixel, two to four cache lines, and it would
be two to four cache lines as bytes too.

**The point-count sweep varies the compute as well as the data**, so it is not evidence
either way. It is here because a per-point cost that stayed flat across a 33-fold change in
point count is worth seeing:

| points | time, x86-64 (µs/point) | time, aarch64 (µs/point) |
|---|---|---|
| 35 | 4.363 | 24.14 |
| 80 | 3.979 | 22.24 |
| 140 | 4.097 | 23.13 |
| 300 | 4.345 | 24.27 |
| 560 | 4.391 | 24.43 |
| 1160 | 4.489 | 25.39 |

**The memory result and the speed result are independent here.** The footprint decides what
fits on a device; it does not make this kernel fast, and further speed has to come from doing
less work rather than from touching less data.

## The algorithm caps the packing advantage

Instruction density rather than a timing: these figures come from the word and lane widths
the two sides use, not from a benchmark.

binCV's real rate inside Lucas–Kanade is 31 pixels per operation, because a 31-pixel window
occupies one `uint32_t` word and the thirty-second bit is wasted — 97% utilisation. Against
OpenCV's 16 pixels per operation (`CV_16S` in AVX2 lanes) that is a 1.94× packing advantage,
and **it is capped there by the window size, not by the implementation**. Widening the word
does not lift the cap, it lowers the utilisation: a 64-bit word carrying a 31-pixel window is
48% used. The gain comes from matching the word to the window, and there is no more of it to
have at this window size.

## The vector arms, and proving they are on

Every vector arm is switchable off, which is the only way to know a measurement is of the
path it claims. On x86-64 the eight-keypoint AVX2 batch in the tracker, toggled at run time
in the same binary, **thirty pinned launches per arm over the whole 1709-frame sequence**
([off](logs/lk_batch_off-x86_64-launches.log), [on](logs/lk_batch_on-x86_64-launches.log)).
The two arms are two settings of the same measurement on the same machine, not two
architectures; bold marks the shipped setting:

| arm | binCV tracking, ms/frame | binCV pipeline, ms/frame | OpenCV pipeline, ms/frame | ratio |
|---|---|---|---|---|
| `BINCV_LK_BATCH=0` | 1.304 | 1.561 | 3.773 | 2.42× [2.41, 2.42] |
| **`BINCV_LK_BATCH=1`** | 0.7070 | 0.9620 | 3.745 | 3.90× [3.88, 3.92] |

**The batch is worth 1.84× [1.82, 1.87] on tracking** and takes the whole pipeline from
2.42× to 3.90×. It is bit-exact with the scalar path.

The batch-on sweep is also an independent repeat of the pipeline figure in
[feature-tracking.md](feature-tracking.md), through a different command line: **3.90×
[3.88, 3.92] here against that page's 3.97× [3.94, 4.00]**, 1.8% apart. The two are not
at the same commit — this sweep was taken at `5c6a47d` (on `main` as `0d6e302`) and that
page's at `880704b` (on `main` as `8729e05`) — so they are not expected to coincide, and
1.8% across an intervening change to the detect stage is the check that the protocol
reproduces rather than just the row.

This machinery exists because it has caught real errors. A vector block was once compiled out
entirely by a mis-attached `#define`, and three consecutive "improvements" were measured
against it. A build that reaches binCV's headers without linking the `bincv_core` CMake target
loses its ISA flags silently — the kernels are still correct, still pass every test, and run
substantially slower with nothing to indicate why. That is why `simdStatusString()` exists:
`feature_tracking_sequence` prints it, and the feature tracking logs in [logs/](logs/)
open with it, showing `NEON=yes` on the device and `AVX2=yes popcount=hardware` on x86-64.
Read that line before trusting any number you take from these benchmarks on your own machine.

## What is not measured at all

**32-bit ARM Cortex-A and RISC-V are supported targets that have not been built or timed.**
Nothing in these reports says anything about them.

**Cortex-M has been built and partly measured, and none of it is in these reports.** binCV
runs on an STM32H753ZI (Cortex-M7): the reductions are bit-exact against the library's own
entry point, a 752×480 frame occupies 46,080 bytes where a `CV_8U` one would occupy 360,960,
and the tracker's staging buffers measure 4,120 bytes at the shipped pyramid depth of 2 bits
per pixel against that board's 16 KiB stack — so the constraint this section expected to
bite did not. What does **not** exist for that part is any OpenCV comparison, any pipeline or
tracker timing, and any figure at the part's full clock; the one operation timed there ran at
the reset default of 64 MHz. `stagingStackBytes<N, W>()` gives the exact stack figure for a
configuration, and the build-time budget fails compilation rather than overflowing at run
time.

**No trajectory-accuracy claim is made anywhere in these reports.** binCV produces features
and flow; what a pose estimator does with them is a property of the whole integration.

The operation set is also smaller than a vision pipeline needs, and grows with the use cases
that turn up rather than from a fixed taxonomy — so an operation's absence says nothing about
whether it belongs.

## Reproduce

```bash
./build/benchmark/pyrfilter_benchmark            # the 8-bit boundary
for i in $(seq 0 15); do ./build/benchmark/bitwidth_crossover $i; done
./build/benchmark/morphology_benchmark           # the element and border losses
./build/benchmark/corner_opencv_benchmark
./build/benchmark/feature_benchmark
./build/benchmark/lk_memorybound                 # compute-bound, not memory-bound
BINCV_LK_BATCH=0 ./build/benchmark/feature_tracking_sequence <dir> 400
BINCV_LK_BATCH=1 ./build/benchmark/feature_tracking_sequence <dir> 400
```

Taken as a launch sweep — `scripts/run_launches.sh -n 30 ./build/benchmark/<bench>` on
x86-64, `scripts/run_launches.sh -n 10 -g ./build/benchmark/<bench>` on the device — and
read back with `scripts/aggregate_launches.py`.

## Logs

The sweep behind each column, and the single launch each replaced (thirty launches on
x86-64, ten on the device):
[pyrDown](logs/pyrfilter-x86_64-launches.log), [single](logs/pyrfilter-x86_64.log), [aarch64](logs/pyrfilter-aarch64-launches.log), [single](logs/pyrfilter-aarch64.log) ·
[crossover](logs/bitwidth_crossover-x86_64-launches.log), [single](logs/bitwidth_crossover-x86_64.log), [aarch64](logs/bitwidth_crossover-aarch64-launches.log), [single](logs/bitwidth_crossover-aarch64.log) ·
[morphology](logs/morphology-x86_64-launches.log), [single](logs/morphology-x86_64.log), [aarch64](logs/morphology-aarch64-launches.log), [single](logs/morphology-aarch64.log) ·
[goodFeaturesToTrack](logs/goodfeatures-x86_64-launches.log), [first thirty](logs/goodfeatures-x86_64.log), [aarch64](logs/goodfeatures-aarch64-launches.log), [single](logs/goodfeatures-aarch64.log) ·
[features](logs/features-x86_64-launches.log), [single](logs/features-x86_64.log), [aarch64](logs/feature-aarch64-launches.log), [single](logs/features-aarch64.log) ·
[LK memory bound](logs/lk_memorybound-x86_64-launches.log), [single](logs/lk_memorybound-x86_64.log), [aarch64](logs/lk_memorybound-aarch64-launches.log), [single](logs/lk_memorybound-aarch64.log) ·
LK batch arm [off](logs/lk_batch_off-x86_64-launches.log), [on](logs/lk_batch_on-x86_64-launches.log), [the two-run reading](logs/lk_batch_arm-x86_64.log)

Logs marked stale in [expected-stale.txt](logs/expected-stale.txt) were taken before a
bit-identical change to the code beneath them; the file records which files moved and why
the figure stands.

## Changes to published figures

| date | row | previous | current | why |
|---|---|---|---|---|
| 2026-09-21 | every x86-64 cell in §1–§3 | one launch | median of thirty launches, with interval | a single launch is one draw from a distribution no log had characterised |
| 2026-09-21 | §4 frame-size sweep, x86-64 change in per-point cost | 12% | 0.4% | all of the 12% was one slow launch at the largest frame |
| 2026-09-21 | §4 frame-size sweep, aarch64 change in per-point cost | 5% (one launch) | 6% (ten launches) | re-taken |
| 2026-09-22 | `goodFeaturesToTrack`, formerly in §3 | 0.530× on both machines; then 0.920× x86-64 / 1.45× aarch64, both against a hand-written OpenCV pipeline | 1.38× / 2.42× against stock `cv::goodFeaturesToTrack`, in features.md | the first figure timed the frame-map spelling on an older response kernel; the second was taken before the response sweep's tail was rewritten; the denominator was changed to the call a caller makes |
| 2026-09-24 | LK batch, gain on tracking | 1.66×–1.88× (two runs an arm over 400 frames) | 1.84× [1.82, 1.87] (thirty launches an arm, 1709 frames) | re-taken |
| 2026-09-24 | LK batch, pipeline ratio with the batch on | 3.63× | 3.90× | the detect stage became 1.93× faster (0.1390 → 0.0720 ms/frame) with the selection-stage optimisation in `ops/corner.hpp`; the earlier log was taken from a modified tree, so the staleness gate could not name the change |
