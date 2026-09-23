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
interval. The three
threading rows near the end are the exception and say so.

## 1. At eight bits per pixel, the idea is gone

binCV wins by not paying for bits it does not use. At eight bits per pixel there are none to
skip, and both sides store a byte.

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

`pyrDown`, 640×480 → 320×240, against `cv::pyrDown` on `CV_8U` at one thread:

| arm | x86-64 (µs) | x86-64 ratio | aarch64 (µs) | aarch64 ratio |
|---|---|---|---|---|
| `cv::pyrDown`, `CV_8U` (the denominator) | 47.70 | — | 516.5 | — |
| **binCV `BOX_2x2`, 1 bit in → 3 bits out (shipped)** | **30.70** | 1.556× [1.536, 1.597] | **93.8** | **5.509× [5.480, 5.549]** |
| binCV `GAUSSIAN_5x5`, 1 → 3 | 184.95 | 0.259× [0.255, 0.266] | 599.1 | 0.862× [0.858, 0.868] |
| binCV `BOX_2x2`, 8 → 8 | 651.9 | 0.0746× [0.0733, 0.0760] | 2571.9 | 0.2007× [0.2001, 0.2019] |
| binCV `GAUSSIAN_5x5`, 8 → 8 (`cv::pyrDown`'s shape) | 2040.0 | **0.0235× [0.0233, 0.0242]** | 7359.9 | **0.0701× [0.0698, 0.0706]** |

An `8 → 8` call is **correct, not fast**, and it is documented that way rather than hidden.
The structural reason is accumulator width: a bit-sliced filter needs enough accumulator
planes to hold the weighted sum of its inputs, so the plane count — and with it the work per
output pixel — grows with the input bit depth. At one bit there is almost nothing to
accumulate; at eight, the bit-sliced form is doing by hand what a byte kernel's vector unit
does in one instruction. Past a certain depth that is simply the better machine.

## 2. The crossover is real, and it moves with the architecture

The same geometry across input and output bit widths, one process per arm because the sweep
is cache-invalid in a single one. Each machine has its own `cv::pyrDown` denominator in the
first row:

| arm | x86-64 (µs) | x86-64 ratio | aarch64 (µs) | aarch64 ratio |
|---|---|---|---|---|
| `cv::pyrDown`, `CV_8U` (the denominator) | 47.85 | — | 517.75 | — |
| **box filter, 1 → 3 (shipped shape)** | **32.20** | **1.474× [1.455, 1.504]** | **275.85** | **1.878× [1.870, 1.923]** |
| box filter, 1 → 1 | 79.05 | 0.605× [0.597, 0.617] | 319.70 | **1.633× [1.613, 1.696]** |
| box filter, 2 → 2 | 62.35 | 0.772× [0.758, 0.779] | 205.00 | **2.526× [2.518, 2.588]** |
| box filter, 3 → 3 | 94.30 | 0.508× [0.495, 0.516] | 306.50 | **1.689× [1.685, 1.730]** |
| box filter, 4 → 4 | 132.65 | 0.358× [0.352, 0.364] | 444.15 | **1.166× [1.162, 1.195]** |
| box filter, 5 → 5 | 176.40 | 0.272× [0.268, 0.277] | 647.30 | 0.801× [0.799, 0.820] |
| box filter, 8 → 8 | 645.30 | 0.0736× [0.0723, 0.0753] | 2574.75 | 0.201× [0.201, 0.205] |

**The crossover is not a property of the algorithm — it moves by several bits between the two
machines.** On the reference device the bit-sliced box filter stays ahead of `cv::pyrDown`
through four bits per pixel and crosses between four and five. On x86-64 the only shape that
beats it is the shipped one, and even `1 → 1` is behind at 0.605×.

The reason is the denominator, not binCV: OpenCV's x86 pyramid is AVX2-dispatched and very
good, and its aarch64 pyramid is relatively weaker against the same machine. binCV's own
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
| FAST, wide-image entry point | `cv::FAST` | 0.359 ms | **0.345 ms** | 1.039× [1.033, 1.048] | 2.910 ms | 3.025 ms | 0.962× [0.961, 0.963] | parity with a mature vectorised kernel |
| `erode`, 5×5 ellipse | `cv::erode` | 0.2238 ns/px | 0.6985 ns/px | 0.319× [0.318, 0.323] | 1.85196 ns/px | 3.59587 ns/px | 0.514× [0.510, 0.522] | a non-separable element costs one shifted-OR per set element |
| `erode`, `BORDER_REPLICATE` | `cv::erode` | 0.09870 ns/px | 0.1489 ns/px | 0.666× [0.659, 0.672] | 0.69852 ns/px | 0.92952 ns/px | 0.752× [0.741, 0.769] | a rim pass `BORDER_CONSTANT` does not need |
| `erode`, `BORDER_REFLECT_101` | `cv::erode` | 0.09931 ns/px | 0.1553 ns/px | 0.635× [0.628, 0.645] | 0.70016 ns/px | 0.94370 ns/px | 0.742× [0.727, 0.759] | the same |
| `erode`, 3×3 rect | `cv::erode` | 0.1013 ns/px | **0.09595 ns/px** | 1.053× [1.035, 1.066] | 0.73595 ns/px | **0.72189 ns/px** | 1.021× [0.991, 1.040] | a dead heat |
| `countNonZero` | `cv::countNonZero` | 0.01501 ns/px | **0.009270 ns/px** | 1.62× [1.61, 1.63] | 0.16921 ns/px | **0.06365 ns/px** | 2.658× [2.618, 2.673] | both sides bandwidth-bound; binCV moves less data |

Parity on FAST ships as parity. A caller who is holding bytes should not be told to pack them
first, and for that caller the honest answer is that binCV costs nothing to adopt and gains
nothing either. The [bit-plane overload](features.md#fast) is where the thesis actually
applies, and it is 1.47× on x86 and 2.365× on the device.

**`goodFeaturesToTrack` has left this list, and the way it left is worth keeping.** It was
published here twice and was wrong both times. The first version read 0.53× on *both*
architectures and concluded that this was "a property of the operation rather than of one
machine's dispatch"; in fact it had timed the frame-map spelling while that spelling was
still on an older response kernel than the streaming form every pipeline here calls. The
second version read 0.92× on x86 and 1.45× on the device and called that a genuine split.
It was not: those numbers were taken before the response sweep's tail was rewritten.

**A third thing was wrong with both, and it was the denominator.** All of those figures were
against a hand-written OpenCV pipeline reproducing binCV's semantics, not against the call a
caller makes. Against stock `cv::goodFeaturesToTrack` this operation *was* on this page until
recently — 0.737× on x86-64 — and the page never said so, because the headline column used
the other baseline. It leaves the list now on the strength of a measurement against the right
one: **1.383× on x86 and 2.421× on the device**, ahead on both, so the row belongs in
[features.md](features.md#corner-detection). What survives of the original point is the
second half of this section's thesis rather than the first: the margin is wider against the
denominator doing less vector work.

## 4. A footprint win is not a speed win

Eight times less data does not make a compute-bound kernel faster, and Lucas–Kanade is
compute-bound. Two sweeps at one level with a 31×31 window, binCV against itself — no OpenCV
arm, so no ratio.

**The frame-size sweep is the one that isolates the question.** The point count is fixed at
140, so the compute is identical and only the data grows:

| frame | input, KB at 1 bit | time, x86-64 (µs/point) | time, aarch64 (µs/point) |
|---|---|---|---|
| 320×240 | 9.4 | 4.658 | 25.64 |
| 640×480 | 37.5 | 4.141 | 23.13 |
| 1280×960 | 150.0 | 4.792 | 26.99 |
| 1920×1440 | 337.5 | 4.640 | 27.23 |

Thirty-six times more data moves the per-point cost by **0.4%** on x86 — 4.658 µs/point at
320×240 against 4.640 at 1920×1440, which is no change at all — and **6%** on the device,
which has a 1 MiB shared L2 where a residency effect would show most clearly if there were
one. The x86 column read 12% before it was taken at thirty launches, and all of that 12% was
one slow launch at the largest frame; the device column read 5% at one launch and 6% at ten,
which is the same answer. A 31×31 window is 120 bytes at one bit per pixel, two to four
cache lines, and it would be two to four cache lines as bytes too.

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

Both columns now have all six. The two aarch64 entries that read `—` were point counts the
old single device launch did not run; ten launches of the same binary give them at no extra
cost ([x86-64](logs/lk_memorybound-x86_64-launches.log),
[aarch64](logs/lk_memorybound-aarch64-launches.log)).

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
architectures:

| arm | binCV tracking, ms/frame | binCV pipeline, ms/frame | OpenCV pipeline, ms/frame | ratio |
|---|---|---|---|---|
| `BINCV_LK_BATCH=0` | 1.3140 | 1.6340 | 3.8185 | 2.319× [2.305, 2.326] |
| **`BINCV_LK_BATCH=1`** | **0.7050** | **1.0285** | 3.7175 | **3.632× [3.614, 3.648]** |

**The batch is worth 1.864× [1.843, 1.903] on tracking** and takes the whole pipeline from
2.32× to 3.63×. It is bit-exact with the scalar path. An earlier reading of the same pair —
two runs an arm over 400 frames — put the tracking figure between 1.66× and 1.88×; thirty
launches an arm narrow that to the interval above, which is what the extra launches bought.

The batch-on sweep is also an independent repeat of the pipeline figure in
[feature-tracking.md](feature-tracking.md) as that page then carried it — 3.632×
[3.614, 3.648] against that sweep's 3.658× [3.633, 3.681], 0.7% apart with overlapping
intervals. (Both are that commit's; the page now reads 3.969× on a faster detect stage.) Two sweeps of the same quantity
through different command lines agreeing to within their intervals is the check that the
protocol reproduces, not just the row.

This machinery exists because it has caught real errors. A vector block was once compiled out
entirely by a mis-attached `#define`, and three consecutive "improvements" were measured
against it. A build that reaches binCV's headers without linking the `bincv_core` CMake target
loses its ISA flags silently — the kernels are still correct, still pass every test, and run
substantially slower with nothing to indicate why. That is why `simdStatusString()` exists:
`feature_tracking_sequence` prints it, and the feature tracking logs in [logs/](logs/)
open with it, showing `NEON=yes` on the device and `AVX2=yes popcount=hardware` on x86. Read that line before
trusting any number you take from these benchmarks on your own machine.

## What is not measured at all

**32-bit ARM Cortex-A and RISC-V are supported targets that have not been built or timed.**
Nothing in these reports says anything about them.

**Cortex-M has been built and partly measured, and none of it is in these reports.** binCV
runs on an STM32H753ZI (Cortex-M7): the reductions are bit-exact against the library's own
entry point, a 752×480 frame occupies 46,080 bytes where a `CV_8U` one would occupy 360,960,
and the tracker's staging buffers measure 4,120 bytes at N = 2 against that board's 16 KB
stack — so the constraint this section expected to bite did not. What does **not** exist for
that part is any OpenCV comparison, any pipeline or tracker timing, and any figure at the
part's full clock; the one operation timed there ran at the reset default of 64 MHz.
`stagingStackBytes<N, W>()` gives the exact stack figure for a configuration, and the
build-time budget fails compilation rather than overflowing at run time.

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

Logs — the sweep behind each column, and the single launch each replaced (thirty launches on
x86-64, ten on the device):
[pyrDown](logs/pyrfilter-x86_64-launches.log), [single](logs/pyrfilter-x86_64.log), [aarch64](logs/pyrfilter-aarch64-launches.log), [single](logs/pyrfilter-aarch64.log) ·
[crossover](logs/bitwidth_crossover-x86_64-launches.log), [single](logs/bitwidth_crossover-x86_64.log), [aarch64](logs/bitwidth_crossover-aarch64-launches.log), [single](logs/bitwidth_crossover-aarch64.log) ·
[morphology](logs/morphology-x86_64-launches.log), [single](logs/morphology-x86_64.log), [aarch64](logs/morphology-aarch64-launches.log), [single](logs/morphology-aarch64.log) ·
[goodFeaturesToTrack](logs/goodfeatures-x86_64-launches.log), [first thirty](logs/goodfeatures-x86_64.log), [aarch64](logs/goodfeatures-aarch64-launches.log), [single](logs/goodfeatures-aarch64.log) ·
[features](logs/features-x86_64-launches.log), [single](logs/features-x86_64.log), [aarch64](logs/feature-aarch64-launches.log), [single](logs/features-aarch64.log) ·
[LK memory bound](logs/lk_memorybound-x86_64-launches.log), [single](logs/lk_memorybound-x86_64.log), [aarch64](logs/lk_memorybound-aarch64-launches.log), [single](logs/lk_memorybound-aarch64.log) ·
LK batch arm [off](logs/lk_batch_off-x86_64-launches.log), [on](logs/lk_batch_on-x86_64-launches.log), [the two-run reading](logs/lk_batch_arm-x86_64.log)
