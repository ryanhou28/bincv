# Primitives

Per-operation results against the OpenCV call each one replaces, on the same binary content
stored as `CV_8U`. 640×480, `uint32_t` words, one thread on both sides unless a row says
otherwise. Setup and denominator rule: [README.md](README.md).

Every benchmark runs a ladder of sizes and the logs carry all of it: the filter benchmarks
downward from 640×480 to 94×60 (a frame to the top level of a four-level pyramid), so a ratio
that collapses once both sides fit in cache can be told from one that holds; the
bandwidth-bound logic and reduction benchmarks upward to 8192×4096, where the question is what
happens when neither side fits.

**x86-64 and aarch64 get separate columns everywhere on this page.** They are different
measurements against different OpenCV builds on different machines, and are never averaged.

## Summary

### Speed

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Every measurement cell is **nanoseconds per pixel**, so the smaller number of each pair is
the faster implementation. Two rows have no aarch64 measurements to show and say `ratio
only` instead.

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `bitwiseAnd` | `cv::bitwise_and` | 0.02734 | **0.00273** | **10.01×** | 0.64783 | **0.02266** | **28.59×** |
| `bitwiseNot` | `cv::bitwise_not` | 0.08591 | **0.00343** | **25.04×** | 0.31658 | **0.01943** | **16.30×** |
| `countNonZero` | `cv::countNonZero` | 0.01548 | **0.00956** | 1.62× | 0.17116 | **0.06366** | 2.69× |
| `countAnd` | `cv::bitwise_and` + `countNonZero` | 0.04164 | **0.01199** | **3.47×** | 0.57602 | **0.08792** | **6.55×** |
| denoise, 3-pixel median | composed `cv::min` / `cv::max` | 0.18609 | **0.01059** | **17.58×** | ratio only | ratio only | **57.66×** |
| spatial derivative | `cv::filter2D` ×2 | 0.54843 | **0.04793** | **11.44×** | ratio only | ratio only | **24.28×** |
| `erode` 3×3 | `cv::erode` | 0.10013 | 0.09605 | 1.04× | 0.71993 | 0.72012 | 1.00× |
| `dilate` 3×3 | `cv::dilate` | 0.10407 | 0.13037 | 0.80× | 0.72416 | **0.48424** | **1.50×** |
| `morphologyEx(OPEN)` | `cv::morphologyEx` | 0.22507 | **0.19595** | 1.15× | 1.34128 | **1.20437** | 1.11× |

`pyrDown` is timed per call rather than per pixel — 640×480 → 320×240, **microseconds per
call**:

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `pyrDown`, 1 bit in | `cv::pyrDown` on `CV_8U` | 48.3 | **31.0** | 1.56× | 521.4 | **93.8** | **5.56×** |

**The denoise and derivative rows are ratio-only on aarch64.** The device times behind those
ratios were not carried into this report, and the committed denoise and derivative logs do
not reproduce the x86-64 values printed beside them either, so neither side can be restated
without changing a published figure. The ratios stand as published and those rows are owed a
re-measurement.

### Memory

Peak working set of one call — the live buffers, not a per-buffer ratio. Computed from
buffer geometry, so it is exact and **identical on both architectures**.

| operation | measured against | OpenCV | binCV | ratio |
|---|---|---|---|---|
| `bitwiseAnd` / `Or` / `Xor` / `Not` | `cv::bitwise_*` | 921,600 B | **115,200 B** | 8.0× |
| `countNonZero`, per input plane | `cv::countNonZero` | 307,200 B | **38,400 B** | 8.0× |
| `countAnd`, per input plane | `cv::bitwise_and` + `countNonZero` | 307,200 B | **38,400 B** | 8.0×, and no temporary |
| denoise, 3-pixel median | composed `cv::min` / `cv::max` | 2,150,400 B | **76,800 B** | **28.0×** |
| spatial derivative, both axes | `cv::filter2D` ×2 | 1,536,000 B | **192,000 B** | 8.00× |
| `erode` / `dilate` 3×3 | `cv::erode` / `cv::dilate` | 614,400 B | **76,800 B** | 8.00× |
| `morphologyEx(OPEN)` | `cv::morphologyEx` | 614,400 B | **115,200 B** | 5.33× |
| four-level `pyrDown` ladder | a `CV_8U` pyramid | 408,000 B | **63,840 B** | 6.39× — [footprint.md](footprint.md#the-pyramid) |

The composed `countAnd` baseline has to materialise the `cv::bitwise_and` result before it
can count it. binCV allocates nothing, so that temporary never appears in its column.

## Logic

An AND over two images becomes an AND over their words, 32 pixels per instruction. 640×480,
**nanoseconds per pixel**:

| operation | OpenCV, x86-64 | binCV `u32`, x86-64 | binCV `u64`, x86-64 | x86-64 ratio, u32 / u64 | OpenCV, aarch64 | binCV `u32`, aarch64 | binCV `u64`, aarch64 | aarch64 ratio, u32 / u64 |
|---|---|---|---|---|---|---|---|---|
| `bitwiseAnd` | 0.02734 | **0.00273** | 0.00399 | 10.01× / 6.85× | 0.64783 | **0.02266** | 0.02327 | 28.59× / 27.85× |
| `bitwiseOr` | 0.02820 | 0.00365 | **0.00274** | 7.74× / 10.28× | 0.64992 | 0.02393 | **0.02212** | 27.16× / 29.38× |
| `bitwiseXor` | 0.02719 | 0.00369 | 0.00360 | 7.37× / 7.56× | 0.64933 | 0.02521 | 0.02363 | 25.76× / 27.48× |
| `bitwiseNot` | 0.08591 | **0.00343** | 0.00373 | 25.04× / 23.06× | 0.31658 | **0.01943** | 0.01663 | 16.30× / 19.04× |

binCV is ahead in all sixteen. The word type does not order these on x86 — `bitwiseAnd`
reads faster at 32 bits and `bitwiseOr` faster at 64, on the same data in the same run —
which is what a bandwidth-bound operation looks like when the arithmetic is free. Do not
read a word-width preference into those two columns.

The *reason* for the speedup is easily misattributed. Both sides run close to the machine's
copy bandwidth — on x86-64, binCV at 72–137 GB/s against OpenCV's 23–110 GB/s across the four
— so neither is inefficient: binCV is faster because it moves an eighth as much data at a
comparable rate. The benchmark prints a measured physical bound beside every row and flags any
result that exceeds it. `bitwiseNot` leads the others because OpenCV's is the slowest of its
four here, not because binCV's is special.

Every operation is checked for identical output before it is timed; the set-pixel counts
appear in the log and the benchmark exits non-zero if they disagree.

## Reductions

640×480, **nanoseconds per pixel**:

| operation | measured against | OpenCV, x86-64 | binCV `u32`, x86-64 | binCV `u64`, x86-64 | x86-64 ratio, u32 / u64 | OpenCV, aarch64 | binCV `u32`, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|---|
| `countNonZero` | `cv::countNonZero` | 0.01548 | **0.00956** | **0.00598** | 1.62× / 2.59× | 0.17116 | **0.06366** | 2.69× |
| `countAnd` | `cv::bitwise_and` then `cv::countNonZero` | 0.04164 | **0.01199** | — | 3.47× / — | 0.57602 | **0.08792** | 6.55× |

`countNonZero` is the most modest ratio in this report: `cv::countNonZero` is already
bandwidth-bound at 64.6 GB/s on x86-64 and binCV reads an eighth as many bytes at 13.1 GB/s, so
the whole difference is how much data has to move. `countAnd` is the clearer win, and not
because of packing: OpenCV has no fused form, so the baseline must materialise a temporary.

Reductions are offered over regions, masks and sliding windows, and **never per word**. On
aarch64 the population count instruction operates on a vector register, so counting a single
general-purpose word pays two register-domain crossings — about the cost of the count itself.
The API therefore has no `popcount(word)`, and the crossings are amortized over the traversal.

The `vs the per-pixel loop` column in the log — 0.44753 ns/pixel against binCV's 0.00956 at
640×480 on x86-64, 46.8× — is what the bit-parallel form is worth against the naive
alternative. It is not a claim against OpenCV and is not quoted as one.

## Denoise

A three-pixel median, against a byte-per-pixel implementation of the same filter ported call
for call from the tracking pipeline binCV was written to replace.

| implementation | x86-64 (ns/px) | x86-64 ratio | aarch64 (ns/px) | aarch64 ratio | working set (bytes) |
|---|---|---|---|---|---|
| OpenCV `CV_8U`, composed (the denominator) | 0.18609 | — | ratio only | — | 2,150,400 |
| **binCV fused, `uint32_t`** | **0.01059** | **17.58×** | ratio only | **57.66×** | **76,800** |
| binCV fused, `uint64_t` | 0.01102 | 16.88× | ratio only | **73.96×** | 76,800 |
| binCV composed, `uint32_t` | 0.05010 | 3.71× | ratio only | 16.56× | 153,600 |

The aarch64 column carries no times because none were recorded here. The committed logs
measure this operation differently enough on both architectures that filling the column from
them would move the published figures, so it is left as a ratio and owes a re-measurement.

Read the working-set column with the ratio: on a part with 1 MiB of shared L2, a large part of
any headline number here is residency rather than arithmetic, and the size ladder in the log
separates them — if the ratio collapses once both sides fit in cache, the headline was
residency.

The composed spelling — `shiftDown`, `shiftLeft`, `majority3` as three passes over two scratch
frames — is what the fused kernel replaced. Fusing was worth 4.73× on x86 and 3.48× on the
device *and* halved the memory, so nothing was traded for it.

## Spatial derivative

Both axes, which is what a tracker needs before it can form a gradient covariance.

| implementation | x86-64 (ns/px) | x86-64 ratio | aarch64 (ns/px) | aarch64 ratio | working set (bytes) | passes |
|---|---|---|---|---|---|---|
| `cv::filter2D` ×2 (the denominator) | 0.54843 | — | ratio only | — | 1,536,000 | 2 |
| **binCV, `uint32_t`** | **0.04793** | **11.44×** | ratio only | **24.28×** | **192,000** | 2 |
| binCV, `uint64_t` | 0.02579 | **21.27×** | ratio only | **46.48×** | 192,000 | 2 |
| binCV composed, `uint32_t` | 0.10089 | 5.44× | ratio only | 8.55× | 268,800 | 8 |

The aarch64 column is ratio-only for the same reason as denoise, and owes the same
re-measurement.

The denominator is `cv::filter2D` twice with `[-1, 0, 1]` as a 1×3 and a 3×1 — the derivative
and nothing else. The reference implementation also multiplies by 16 and merges the two axes
into an interleaved two-channel image; binCV reproduces neither, so charging those to the
baseline would flatter binCV. That row is printed in the log and not used.

Some of this ratio is fixed per-call cost, measured separately on a 2×2 frame: at 640×480
OpenCV pays 1.91 µs of its 168.48 µs per frame against binCV's 0.01 µs of 14.73 µs — about 1%
of each side. At 94×60 it is most of what the per-pixel figure is made of, which is why the log
prints it per size.

## Morphology

The most mixed result in this report. 640×480, **nanoseconds per pixel**, same element,
anchor and border on both sides. **Bold marks a binCV cell that beats the OpenCV cell for
its architecture**:

| case | OpenCV, x86-64 | binCV `u32`, x86-64 | binCV `u64`, x86-64 | x86-64 ratio, u32 / u64 | OpenCV, aarch64 | binCV `u32`, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `erode` 3×3 rect, `BORDER_CONSTANT` | 0.10013 | **0.09605** | **0.06127** | 1.04× / 1.63× | 0.71993 | 0.72012 | 1.00× |
| `dilate` 3×3 rect, `BORDER_CONSTANT` | 0.10407 | 0.13037 | **0.06185** | 0.80× / 1.68× | 0.72416 | **0.48424** | 1.50× |
| `morphologyEx(OPEN)` 3×3 | 0.22507 | **0.19595** | **0.11979** | 1.15× / 1.88× | 1.34128 | **1.20437** | 1.11× |
| `erode` 5×5 ellipse | 0.22759 | 0.70415 | 0.35309 | 0.32× / 0.64× | 1.81575 | 3.58631 | 0.51× |
| `erode` 3×3, `BORDER_REPLICATE` | 0.09669 | 0.15217 | 0.11852 | 0.64× / 0.82× | 0.67176 | 0.93004 | 0.72× |
| `erode` 3×3, `BORDER_REFLECT_101` | 0.09789 | 0.15761 | 0.12159 | 0.62× / 0.81× | 0.67158 | 0.94380 | 0.71× |

At the default `uint32_t` word binCV leads on two of the six cases on x86-64 and two on
aarch64; the wider word takes x86-64 to three. Ten of the eighteen ratio figures are under
1.00×, and every one of those is `cv::erode` or `cv::dilate` ahead of binCV.

Every `erode` and `dilate` case above holds **76,800 B** live against `cv::erode`'s
**614,400 B**; `morphologyEx(OPEN)` holds **115,200 B** against the same 614,400. `erode`
and `dilate` need no scratch at all — a dilation is a shift and an OR — so the footprint
advantage is the full 8×. `morphologyEx(OPEN)` needs one caller-provided frame where OpenCV
needs none, which is why its footprint advantage is 5.33× rather than 8×.

**binCV loses on the 5×5 ellipse**, and by a wide margin. A non-separable structuring
element costs the bit-parallel form one shifted-OR per set element, where OpenCV's
vectorised byte kernel amortises the same work across a SIMD register. The fused kernel was
kept anyway, because it is 8× smaller and the alternative spelling is slower still — a
deliberate speed-for-footprint trade, priced in [footprint.md](footprint.md).

**binCV also loses on non-constant borders.** `BORDER_REPLICATE` and `BORDER_REFLECT_101`
each cost binCV a rim pass that `BORDER_CONSTANT` does not need; the interior kernel is
unchanged.

## Pyramid downsample

640×480 → 320×240, **microseconds per call**, against `cv::pyrDown` on `CV_8U` at one
thread:

| arm | x86-64 (µs) | x86-64 ratio | aarch64 (µs) | aarch64 ratio |
|---|---|---|---|---|
| `cv::pyrDown`, `CV_8U` (the denominator) | 48.3 | — | 521.4 | — |
| **binCV `BOX_2x2`, 1 → 3 (shipped)** | **31.0** | 1.56× | **93.8** | **5.56×** |
| binCV `GAUSSIAN_5x5`, 1 → 3 | 195.4 | 0.25× | 599.0 | 0.87× |

The shipped route takes a 1-bit level and produces a 2- or 3-bit one with a 2×2 box filter.
That is not `cv::pyrDown`'s filter, so the comparison is Tier 2: the same role, a different
answer. Where binCV matches OpenCV's filter exactly — a Gaussian 5×5 with `BORDER_REFLECT_101`,
bit-exact against `cv::pyrDown` at five size parities — it is substantially slower on x86 and
roughly at parity on the device, at three-eighths of the stored bits.

The four-level ladder this feeds holds 63,840 bytes against a `CV_8U` pyramid's 408,000 at
the same geometry; the ladder table is in [footprint.md](footprint.md#the-pyramid).

**The eight-bit case is the boundary of the whole approach and is covered in
[limits.md](limits.md).** Bit-slicing wins by not paying for bits it does not use; at eight
bits per pixel there are none to skip, and an `8 → 8` call is correct, not fast.

## Reproduce

```bash
./build/benchmark/logic_benchmark
./build/benchmark/reduce_benchmark
./build/benchmark/denoise_benchmark
./build/benchmark/derivative_benchmark
./build/benchmark/morphology_benchmark
./build/benchmark/pyrfilter_benchmark
```

Each is self-contained and needs no dataset. Logs:
[logic](logs/logic-x86_64.log), [aarch64](logs/logic-aarch64.log) ·
[reduce](logs/reduce-x86_64.log), [aarch64](logs/reduce-aarch64.log) ·
[denoise](logs/denoise-x86_64.log), [aarch64](logs/denoise-aarch64.log) ·
[derivative](logs/derivative-x86_64.log), [aarch64](logs/derivative-aarch64.log) ·
[morphology](logs/morphology-x86_64.log), [aarch64](logs/morphology-aarch64.log) ·
[pyrDown](logs/pyrfilter-x86_64.log), [aarch64](logs/pyrfilter-aarch64.log)
