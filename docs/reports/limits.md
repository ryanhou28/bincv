# Limits

Where binCV ties, loses, or stops paying at all. Every figure comes from the same committed
benchmarks as the rest of the reports. The useful question about a library like this one is
not "how fast can it be" but "when does the idea stop working", and it has four answers.

Every table reads the same way: **both sides' measured times in the unit the column names,
with the ratio beside them.** A row where binCV's number is the larger one — and the ratio
under 1.00× — is a row binCV lost, which is most of this page. **x86-64 and aarch64 are
separate columns** and are never averaged.

## 1. At eight bits per pixel, the idea is gone

binCV wins by not paying for bits it does not use. At eight bits per pixel there are none to
skip, and both sides store a byte.

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

`pyrDown`, 640×480 → 320×240, against `cv::pyrDown` on `CV_8U` at one thread:

| arm | x86-64 (µs) | x86-64 ratio | aarch64 (µs) | aarch64 ratio |
|---|---|---|---|---|
| `cv::pyrDown`, `CV_8U` (the denominator) | 48.3 | — | 521.4 | — |
| **binCV `BOX_2x2`, 1 bit in → 3 bits out (shipped)** | **31.0** | 1.56× | **93.8** | **5.56×** |
| binCV `GAUSSIAN_5x5`, 1 → 3 | 195.4 | 0.25× | 599.0 | 0.87× |
| binCV `BOX_2x2`, 8 → 8 | 707.6 | 0.07× | 2574.2 | 0.20× |
| binCV `GAUSSIAN_5x5`, 8 → 8 (`cv::pyrDown`'s shape) | 2034.4 | **0.02×** | 7358.6 | **0.07×** |

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
| `cv::pyrDown`, `CV_8U` (the denominator) | 48.5 | — | 514.7 | — |
| **box filter, 1 → 3 (shipped shape)** | **32.6** | **1.49×** | **275.6** | **1.87×** |
| box filter, 1 → 1 | 86.8 | 0.56× | 319.8 | **1.61×** |
| box filter, 2 → 2 | 63.7 | 0.76× | 205.1 | **2.51×** |
| box filter, 3 → 3 | 97.2 | 0.50× | 306.5 | **1.68×** |
| box filter, 4 → 4 | 136.0 | 0.36× | 444.2 | **1.16×** |
| box filter, 5 → 5 | 176.1 | 0.28× | 648.1 | 0.79× |
| box filter, 8 → 8 | 701.8 | 0.07× | 2604.4 | 0.20× |

**The crossover is not a property of the algorithm — it moves by several bits between the two
machines.** On the reference device the bit-sliced box filter stays ahead of `cv::pyrDown`
through four bits per pixel and crosses between four and five. On x86-64 the only shape that
beats it is the shipped one, and even `1 → 1` is behind at 0.56×.

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
| FAST, wide-image entry point | `cv::FAST` | 0.363 ms | **0.344 ms** | 1.05× | 2.906 ms | 3.024 ms | 0.96× | parity with a mature vectorised kernel |
| `erode`, 5×5 ellipse | `cv::erode` | 0.22759 ns/px | 0.70415 ns/px | 0.32× | 1.81575 ns/px | 3.58631 ns/px | 0.51× | a non-separable element costs one shifted-OR per set element |
| `erode`, `BORDER_REPLICATE` | `cv::erode` | 0.09669 ns/px | 0.15217 ns/px | 0.64× | 0.67176 ns/px | 0.93004 ns/px | 0.72× | a rim pass `BORDER_CONSTANT` does not need |
| `erode`, `BORDER_REFLECT_101` | `cv::erode` | 0.09789 ns/px | 0.15761 ns/px | 0.62× | 0.67158 ns/px | 0.94380 ns/px | 0.71× | the same |
| `erode`, 3×3 rect | `cv::erode` | 0.10013 ns/px | **0.09605 ns/px** | 1.04× | 0.71993 ns/px | 0.72012 ns/px | 1.00× | a dead heat |
| `goodFeaturesToTrack` | `cv::goodFeaturesToTrack`, binarized | 13.63–14.24 ns/px | 14.46–15.01 ns/px | 0.92× | 75.02–75.82 ns/px | **51.25–51.31 ns/px** | **1.45×** | seven float planes of locality binCV declines to buy |
| `countNonZero` | `cv::countNonZero` | 0.01548 ns/px | **0.00956 ns/px** | 1.62× | 0.17116 ns/px | **0.06366 ns/px** | 2.69× | both sides bandwidth-bound; binCV moves less data |

Parity on FAST ships as parity. A caller who is holding bytes should not be told to pack them
first, and for that caller the honest answer is that binCV costs nothing to adopt and gains
nothing either. The [bit-plane overload](features.md#fast) is where the thesis actually
applies, and it is 1.50× on x86 and 2.37× on the device.

**`goodFeaturesToTrack` is on this list for x86 only, and it is the sharpest illustration of
the point above it.** An earlier version of these reports published 0.53× on *both*
architectures and concluded that this was "a property of the operation rather than of one
machine's dispatch". Both halves were wrong: the figure measured the frame-map spelling while
it was still on an older response kernel than the streaming spelling every frontend here
calls. Once the two share one kernel the operation is 0.92× on x86 and **1.45× on the
reference device** — a loss on the desktop and a win on the deployment part, from identical
code over identical buffers returning identical corners.

## 4. A footprint win is not a speed win

Eight times less data does not make a compute-bound kernel faster, and Lucas–Kanade is
compute-bound. Two sweeps at one level with a 31×31 window, binCV against itself — no OpenCV
arm, so no ratio.

**The frame-size sweep is the one that isolates the question.** The point count is fixed at
140, so the compute is identical and only the data grows:

| frame | input, KB at 1 bit | time, x86-64 (µs/point) | time, aarch64 (µs/point) |
|---|---|---|---|
| 320×240 | 9.4 | 4.82 | 25.67 |
| 640×480 | 37.5 | 4.41 | 23.31 |
| 1280×960 | 150.0 | 4.76 | 27.24 |
| 1920×1440 | 337.5 | 5.42 | 26.99 |

Thirty-six times more data moves the per-point cost by 12% on x86 and 5% on the device — and
the device has a 1 MiB shared L2, where a residency effect would show most clearly if there
were one. A 31×31 window is 120 bytes at one bit per pixel, two to four cache lines, and it
would be two to four cache lines as bytes too.

**The point-count sweep varies the compute as well as the data**, so it is not evidence
either way. It is here because a per-point cost that stayed flat across a 33-fold change in
point count is worth seeing:

| points | time, x86-64 (µs/point) | time, aarch64 (µs/point) |
|---|---|---|
| 35 | 5.75 | 24.22 |
| 140 | 4.18 | 23.30 |
| 560 | 4.61 | 24.62 |
| 1160 | 4.85 | 25.55 |

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
in the same binary over 400 frames. **The two runs are two repeats of the same measurement on
the same machine, not two architectures**:

| arm | run | binCV tracking, ms/frame | binCV frontend, ms/frame | OpenCV frontend, ms/frame | ratio |
|---|---|---|---|---|---|
| `BINCV_LK_BATCH=0` | 1 | 1.363 | 1.782 | 4.090 | 2.30× |
| `BINCV_LK_BATCH=0` | 2 | 1.407 | 1.804 | 4.222 | 2.34× |
| **`BINCV_LK_BATCH=1`** | 1 | **0.820** | **1.201** | 4.221 | **3.51×** |
| **`BINCV_LK_BATCH=1`** | 2 | **0.747** | **1.120** | 3.991 | **3.56×** |

The batch is worth 1.66–1.88× on tracking and takes the whole frontend from about 2.3× to
about 3.5×. It is bit-exact with the scalar path.

This machinery exists because it has caught real errors. A vector block was once compiled out
entirely by a mis-attached `#define`, and three consecutive "improvements" were measured
against it. A build that reaches binCV's headers without linking the `bincv_core` CMake target
loses its ISA flags silently — the kernels are still correct, still pass every test, and run
substantially slower with nothing to indicate why. That is why `simdStatusString()` exists:
`frontend_sequence` prints it, and the frontend logs in [logs/](logs/) open with it, showing
`NEON=yes` on the device and `AVX2=yes popcount=hardware` on x86. Read that line before
trusting any number you take from these benchmarks on your own machine.

## What is not measured at all

**32-bit ARM Cortex-A and RISC-V are supported targets that have not been built or timed.**
Nothing in these reports says anything about them.

**Cortex-M has been built and partly measured, and none of it is in these reports.** binCV
runs on an STM32H753ZI (Cortex-M7): the reductions are bit-exact against the library's own
entry point, a 752×480 frame occupies 46,080 bytes where a `CV_8U` one would occupy 360,960,
and the tracker's staging buffers measure 4,120 bytes at N = 2 against that board's 16 KB
stack — so the constraint this section expected to bite did not. What does **not** exist for
that part is any OpenCV comparison, any frontend or tracker timing, and any figure at the
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
BINCV_LK_BATCH=0 ./build/benchmark/frontend_sequence <dir> 400
BINCV_LK_BATCH=1 ./build/benchmark/frontend_sequence <dir> 400
```

Logs: [pyrDown](logs/pyrfilter-x86_64.log), [aarch64](logs/pyrfilter-aarch64.log) ·
[crossover](logs/bitwidth_crossover-x86_64.log), [aarch64](logs/bitwidth_crossover-aarch64.log) ·
[morphology](logs/morphology-x86_64.log), [aarch64](logs/morphology-aarch64.log) ·
[goodFeaturesToTrack](logs/goodfeatures-x86_64.log), [aarch64](logs/goodfeatures-aarch64.log) ·
[features](logs/features-x86_64.log), [aarch64](logs/features-aarch64.log) ·
[LK memory bound](logs/lk_memorybound-x86_64.log), [aarch64](logs/lk_memorybound-aarch64.log) ·
[LK batch arm](logs/lk_batch_arm-x86_64.log)
