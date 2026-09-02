# Limits

Where binCV ties, loses, or stops paying at all. Every figure here comes from the same
committed benchmarks as the rest of the reports; nothing in this page is an argument that
was not measured.

This exists because the useful question about a library like this one is not "how fast can
it be" but "when does the idea stop working". It has four answers.

## 1. At eight bits per pixel, the idea is gone

binCV wins by not paying for bits it does not use. At eight bits per pixel there are none to
skip, and both sides store a byte.

`pyrDown`, 640×480 → 320×240, against `cv::pyrDown` on `CV_8U` at one thread:

| arm | x86 µs | vs `cv::pyrDown` | aarch64 µs | vs `cv::pyrDown` |
|---|---|---|---|---|
| binCV `BOX_2x2`, 1 bit in → 3 bits out (shipped) | 31.0 | **1.56×** | 93.8 | **5.56×** |
| `cv::pyrDown`, `CV_8U` (the denominator) | 48.3 | 1.00× | 521.4 | 1.00× |
| binCV `GAUSSIAN_5x5`, 1 → 3 | 195.4 | 0.25× | 599.0 | 0.87× |
| binCV `BOX_2x2`, 8 → 8 | 707.6 | 0.07× | 2574.2 | 0.20× |
| binCV `GAUSSIAN_5x5`, 8 → 8 (`cv::pyrDown`'s shape) | 2034.4 | **0.02×** | 7358.6 | **0.07×** |

An `8 → 8` call is **correct, not fast**, and it is documented that way rather than hidden.
The structural reason is the accumulator width: a bit-sliced filter needs enough accumulator
planes to hold the weighted sum of its inputs, so the plane count — and with it the work per
output pixel — grows with the input bit depth. At one bit there is almost nothing to
accumulate; at eight, the bit-sliced form is doing by hand what a byte kernel's vector unit
does in one instruction. Past a certain depth that is simply the better machine.

## 2. The crossover is real, and it moves with the architecture

The same geometry across input and output bit widths, one process per arm because the sweep
is cache-invalid in a single one. Denominator: `cv::pyrDown` at 48.5 µs.

| bits in → out | box filter, x86-64 | vs `cv::pyrDown` |
|---|---|---|
| 1 → 3 (shipped shape) | 32.6 µs | **1.49×** |
| 1 → 1 | 86.8 | 0.56× |
| 2 → 2 | 63.7 | 0.76× |
| 3 → 3 | 97.2 | 0.50× |
| 4 → 4 | 136.0 | 0.36× |
| 5 → 5 | 176.1 | 0.28× |
| 8 → 8 | 701.8 | 0.07× |

The same sweep on the reference device, against `cv::pyrDown` at 514.7 µs:

| bits in → out | box filter, aarch64 | vs `cv::pyrDown` |
|---|---|---|
| 1 → 3 (shipped shape) | 275.6 µs | **1.87×** |
| 1 → 1 | 319.8 | **1.61×** |
| 2 → 2 | 205.1 | **2.51×** |
| 3 → 3 | 306.5 | **1.68×** |
| 4 → 4 | 444.2 | **1.16×** |
| 5 → 5 | 648.1 | 0.79× |
| 8 → 8 | 2604.4 | 0.20× |

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

Bit packing gives a `uint32_t` thirty-two pixels. An AVX2 register of bytes holds exactly
thirty-two pixels too. Against a mature vectorised byte kernel, packing alone buys nothing
until the boolean algebra also moves into a vector register — and where OpenCV has already
done that work, binCV ties or loses. Where OpenCV has done less of it, the same binCV code
wins, which is why the two columns below disagree as often as they agree.

| operation | x86-64 | aarch64 | what happened |
|---|---|---|---|
| `cv::FAST`, wide-image entry point | 1.05× | 0.96× | parity; `cv::FAST` is a mature vectorised kernel |
| `erode`, 5×5 ellipse | **0.32×** | **0.51×** | non-separable element costs one shifted-OR per set element |
| `erode`, `BORDER_REPLICATE` | 0.64× | 0.72× | a rim pass `BORDER_CONSTANT` does not need |
| `erode`, `BORDER_REFLECT_101` | 0.62× | 0.71× | the same |
| `erode`, 3×3 rect | 1.04× | 1.00× | dead heat with a vectorised byte kernel |
| `goodFeaturesToTrack` | 0.92× | *1.45× — ahead* | seven float planes of locality binCV declines to buy |
| `countNonZero` | 1.62× | 2.69× | both sides are bandwidth-bound; binCV moves less data, that is all |

Parity on FAST ships as parity. A caller who is holding bytes should not be told to pack
them first, and for that caller the honest answer is that binCV costs nothing to adopt and
gains nothing either. The [bit-plane overload](features.md#fast) is where the thesis actually
applies, and it is 1.50× on x86 and 2.37× on the device.

**`goodFeaturesToTrack` is on this list for x86 only, and it is the sharpest illustration of
the point above it.** An earlier version of these reports published 0.53× on *both*
architectures and concluded that this was "a property of the operation rather than of one
machine's dispatch". Both halves of that were wrong. The figure measured the frame-map
spelling while it was still on an older response kernel than the streaming spelling every
frontend here calls, and once the two share one kernel the operation is 0.92× on x86 and
**1.45× on the reference device** — a loss on the desktop and a win on the deployment part,
from identical code over identical buffers returning identical corners.

## 4. A footprint win is not a speed win

Eight times less data does not make a compute-bound kernel faster, and Lucas–Kanade is
compute-bound. At one level with a 31×31 window:

| points | x86 µs/point | aarch64 µs/point | | frame | KB at 1 bit | x86 | aarch64 |
|---|---|---|---|---|---|---|---|
| 35 | 5.75 | 24.22 | | 320×240 | 9.4 | 4.82 | 25.67 |
| 140 | 4.18 | 23.30 | | 640×480 | 37.5 | 4.41 | 23.31 |
| 560 | 4.61 | 24.62 | | 1280×960 | 150.0 | 4.76 | 27.24 |
| 1160 | 4.85 | 25.55 | | 1920×1440 | 337.5 | 5.42 | 26.99 |

The right column is the one that isolates the question: the point count is fixed at 140, so
the compute is identical and only the frame grows. Thirty-six times more data moves the
per-point cost by 12% on x86 and 5% on the device — and the device has a 1 MiB shared L2,
which is where a residency effect would show most clearly if there were one. A 31×31 window
is 120 bytes at one bit per pixel, two to four cache lines, and it would be two to four cache
lines as bytes too.

The left column varies the compute as well as the data, so it is not evidence either way; it
is here because a per-point cost that stayed flat across a 33-fold change in point count is
worth seeing.

**The memory result and the speed result are independent here.** The footprint decides what
fits on a device; it does not make this kernel fast, and further speed has to come from doing
less work rather than from touching less data.

## The algorithm caps the packing advantage

binCV's real rate inside Lucas–Kanade is 31 pixels per operation, because a 31-pixel window
occupies one `uint32_t` word and the thirty-second bit is wasted — 97% utilisation, which is
excellent. Against OpenCV's 16 pixels per operation (`CV_16S` in AVX2 lanes) that is a 1.94×
packing advantage, and **it is capped there by the window size, not by the implementation**.

Widening the word does not lift the cap, it lowers the utilisation: a 64-bit word carrying a
31-pixel window is 48% used. That is the shape of the whole result — the gain comes from
matching the word to the window, and there is no more of it to have at this window size.

## The vector arms, and proving they are on

Every vector arm is switchable off, which is the only way to know a measurement is of the
path it claims. On x86 the eight-keypoint AVX2 batch in the tracker, toggled at run time in
the same binary over 400 frames:

| | tracking, ms/frame | frontend, ms/frame | vs OpenCV |
|---|---|---|---|
| `BINCV_LK_BATCH=0` | 1.363 / 1.407 | 1.782 / 1.804 | 2.30× / 2.34× |
| `BINCV_LK_BATCH=1` | 0.820 / 0.747 | 1.201 / 1.120 | **3.51× / 3.56×** |

The batch is worth 1.66–1.88× on tracking and takes the whole frontend from about 2.3× to
about 3.5×. It is bit-exact with the scalar path.

This machinery exists because it has caught real errors. A vector block was once compiled out
entirely by a mis-attached `#define`, and three consecutive "improvements" were measured
against it. A build that reaches binCV's headers without linking the `bincv_core` CMake
target loses its ISA flags silently — the kernels are still correct, still pass every test,
and run substantially slower with nothing to indicate why. That is why
`simdStatusString()` exists: `frontend_sequence` prints it, and the frontend logs in
[logs/](logs/) open with it, showing `NEON=yes` on the device and
`AVX2=yes popcount=hardware` on x86. Read that line before trusting any number you take from
these benchmarks on your own machine.

## What is not measured at all

**32-bit ARM, Cortex-M and RISC-V are supported targets that have not been built or timed.**
Nothing in these reports says anything about them. The constraint expected to bite there is
stack rather than throughput: the tracker stages each window into stack buffers whose size
grows with the bit depth, which is nothing on a desktop and can be everything on a part with
a 16 KB stack. The library exposes `stagingStackBytes<N, W>()` for the exact figure and a
build-time budget that fails compilation rather than overflowing at run time — but no timing
on those parts exists, here or elsewhere.

**No trajectory-accuracy claim is made anywhere in these reports.** binCV produces features
and flow; what a pose estimator does with them is a property of the whole integration, and
these reports measure kernels. The agreement figures in [frontend.md](frontend.md) are
evidence that the kernels are sufficient, not a claim about pose error.

The operation set is also smaller than a vision pipeline needs, and grows with the use cases
that turn up rather than from a fixed taxonomy — so an operation's absence here says nothing
about whether it belongs.

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
