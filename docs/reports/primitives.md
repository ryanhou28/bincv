# Primitives

Per-operation results against each one's OpenCV equivalent, on the same binary content
stored as `CV_8U`. 640×480, `uint32_t` words, one thread on both sides unless a row says
otherwise. Setup and denominator rule: [README.md](README.md).

Every benchmark runs a ladder of sizes and the logs carry all of it: the filter benchmarks
downward from 640×480 to 94×60 (a frame to the top level of a four-level pyramid), so a ratio
that collapses once both sides fit in cache can be told from one that holds; the
bandwidth-bound logic and reduction benchmarks upward to 8192×4096, where the question is what
happens when neither side fits.

**x86-64 and aarch64 get separate columns everywhere on this page.** They are different
measurements against different OpenCV builds on different machines, and are never averaged.

**Every x86-64 figure on this page is the median of thirty pinned launches**, and every
x86-64 ratio carries the bootstrap 95% interval the thirty put around it. The ratio is
formed inside each launch before the median is taken, so it is not the quotient of the two
cells beside it. aarch64 is a single pinned launch, which is enough on that host —
[methodology-timing.md](methodology-timing.md#the-protocol-each-host-needs) says why, and
[the index](README.md#on-the-x86-64-host) says what changed when the x86 figures were
re-taken.

## Summary

### Speed

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Every measurement cell is **nanoseconds per pixel**, so the smaller number of each pair is
the faster implementation. Two rows have no aarch64 measurements to show and say `ratio
only` instead.

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `bitwiseAnd` | `cv::bitwise_and` | 0.02823 | **0.002810** | **9.97× [9.82, 10.28]** | 0.64783 | **0.02266** | **28.59×** |
| `bitwiseNot` | `cv::bitwise_not` | 0.06336 | **0.003510** | **17.99× [17.82, 18.14]** | 0.31658 | **0.01943** | **16.30×** |
| `countNonZero` | `cv::countNonZero` | 0.01501 | **0.009270** | 1.62× [1.61, 1.63] | 0.17116 | **0.06366** | 2.69× |
| `countAnd` | `cv::bitwise_and` + `countNonZero` | 0.04172 | **0.01196** | **3.49× [3.45, 3.52]** | 0.57602 | **0.08792** | **6.55×** |
| denoise, 3-pixel median | composed `cv::min` / `cv::max` | 0.1887 | **0.009865** | **19.09× [18.94, 19.38]** | ratio only | ratio only | **57.66×** |
| spatial derivative | `cv::filter2D` ×2 | 0.5156 | **0.04645** | **11.12× [11.07, 11.28]** | ratio only | ratio only | **24.28×** |
| `erode` 3×3 | `cv::erode` | 0.1013 | **0.09595** | 1.053× [1.035, 1.066] | 0.71993 | 0.72012 | 1.00× |
| `dilate` 3×3 | `cv::dilate` | 0.09972 | **0.09446** | 1.057× [1.046, 1.073] | 0.72416 | **0.48424** | **1.50×** |
| `morphologyEx(OPEN)` | `cv::morphologyEx` | 0.1956 | **0.1904** | 1.022× [1.016, 1.043] | 1.34128 | **1.20437** | 1.11× |

`pyrDown` is timed per call rather than per pixel — 640×480 → 320×240, **microseconds per
call**:

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `pyrDown`, 1 bit in | `cv::pyrDown` on `CV_8U` | 47.70 | **30.70** | 1.556× [1.536, 1.597] | 521.4 | **93.8** | **5.56×** |

**`dilate` 3×3 was published as a 0.80× loss on x86-64, and it is not one.** Thirty launches
put it at 1.057×, ahead in 30 of 30, on a kernel whose source has not changed since the single
launch that produced 0.80× — that launch timed binCV's arm at 0.13037 ns/pixel where thirty
span 0.09260 to 0.09680. It is the one row in these reports whose *sign* the single-launch
protocol got wrong. Two rows moved the other way and are recorded as such:
`bitwiseNot` reads 17.99× rather than 25.04× because `cv::bitwise_not` was slow in the old
launch, and `morphologyEx(OPEN)` reads 1.022× rather than 1.15×, four of its thirty launches
falling below 1.00. [The index](README.md#on-the-x86-64-host) has the whole comparison.

**The denoise and derivative rows are ratio-only on aarch64.** The device times behind those
ratios were not carried into this report, so that column cannot be restated. Their x86-64
columns now have a committed source for the first time: the previously published x86 values
appear nowhere in this repository's logs, and the cells above are the thirty-launch medians
of [denoise](logs/denoise-x86_64-launches.log) and
[derivative](logs/derivative-x86_64-launches.log). The aarch64 ratios stand as published and
still owe a re-measurement.

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
| `bitwiseAnd` | 0.02823 | **0.002810** | 0.003755 | **9.97× [9.82, 10.28]** / 7.58× [7.35, 7.75] | 0.64783 | **0.02266** | 0.02327 | 28.59× / 27.85× |
| `bitwiseOr` | 0.02794 | 0.003665 | **0.002825** | 7.69× [7.41, 7.78] / **9.99× [9.88, 10.10]** | 0.64992 | 0.02393 | **0.02212** | 27.16× / 29.38× |
| `bitwiseXor` | 0.02779 | 0.003605 | 0.003610 | 7.70× [7.62, 7.78] / 7.72× [7.52, 7.80] | 0.64933 | 0.02521 | 0.02363 | 25.76× / 27.48× |
| `bitwiseNot` | 0.06336 | **0.003510** | 0.003540 | **17.99× [17.82, 18.14]** / 17.91× [17.76, 18.15] | 0.31658 | **0.01943** | 0.01663 | 16.30× / 19.04× |

binCV is ahead in all sixteen, and on x86-64 in every one of the thirty launches behind each
cell. The word type does not order these on x86 — `bitwiseAnd` reads faster at 32 bits and
`bitwiseOr` faster at 64, on the same data in the same run — which is what a bandwidth-bound
operation looks like when the arithmetic is free. Do not read a word-width preference into
those two columns.

The *reason* for the speedup is easily misattributed. Both sides run close to the machine's
copy bandwidth — on x86-64, binCV at 71–133 GB/s against OpenCV's 32–108 GB/s across the four
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
| `countNonZero` | `cv::countNonZero` | 0.01501 | **0.009270** | **0.005895** | 1.62× [1.61, 1.63] / 2.55× [2.53, 2.57] | 0.17116 | **0.06366** | 2.69× |
| `countAnd` | `cv::bitwise_and` then `cv::countNonZero` | 0.04172 | **0.01196** | — | **3.49× [3.45, 3.52]** / — | 0.57602 | **0.08792** | 6.55× |

`countNonZero` is the most modest ratio in this report: `cv::countNonZero` is already
bandwidth-bound at 66.6 GB/s on x86-64 and binCV reads an eighth as many bytes at 13.5 GB/s, so
the whole difference is how much data has to move. `countAnd` is the clearer win, and not
because of packing: OpenCV has no fused form, so the baseline must materialise a temporary.

Reductions are offered over regions, masks and sliding windows, and **never per word**. On
aarch64 the population count instruction operates on a vector register, so counting a single
general-purpose word pays two register-domain crossings — about the cost of the count itself.
The API therefore has no `popcount(word)`, and the crossings are amortized over the traversal.

The `vs the per-pixel loop` column in the log — 0.4505 ns/pixel against binCV's 0.009270 at
640×480 on x86-64, 48.5× [48.0, 48.7] — is what the bit-parallel form is worth against the
naive alternative. It is not a claim against OpenCV and is not quoted as one.

## Denoise

A three-pixel median, against a byte-per-pixel implementation of the same filter ported call
for call from the tracking pipeline binCV was written to replace.

| implementation | x86-64 (ns/px) | x86-64 ratio | aarch64 (ns/px) | aarch64 ratio | working set (bytes) |
|---|---|---|---|---|---|
| OpenCV `CV_8U`, composed (the denominator) | 0.1887 | — | ratio only | — | 2,150,400 |
| **binCV fused, `uint32_t`** | **0.009865** | **19.09× [18.94, 19.38]** | ratio only | **57.66×** | **76,800** |
| binCV fused, `uint64_t` | 0.007385 | **25.39× [25.16, 25.80]** | ratio only | **73.96×** | 76,800 |
| binCV composed, `uint32_t` | 0.04820 | 3.91× [3.88, 3.95] | ratio only | 16.56× | 153,600 |

The aarch64 column carries no times because none were recorded here, and it owes a
re-measurement. The x86-64 column is thirty pinned launches
([log](logs/denoise-x86_64-launches.log)); the values it replaces reproduced from no committed
log at all, which is how a figure with no source stays in a table.

Read the working-set column with the ratio: on a part with 1 MiB of shared L2, a large part of
any headline number here is residency rather than arithmetic, and the size ladder in the log
separates them — if the ratio collapses once both sides fit in cache, the headline was
residency.

The composed spelling — `shiftDown`, `shiftLeft`, `majority3` as three passes over two scratch
frames — is what the fused kernel replaced. Fusing was worth 4.91× on x86 and 3.48× on the
device *and* halved the memory, so nothing was traded for it.

## Spatial derivative

Both axes, which is what a tracker needs before it can form a gradient covariance.

| implementation | x86-64 (ns/px) | x86-64 ratio | aarch64 (ns/px) | aarch64 ratio | working set (bytes) | passes |
|---|---|---|---|---|---|---|
| `cv::filter2D` ×2 (the denominator) | 0.5156 | — | ratio only | — | 1,536,000 | 2 |
| **binCV, `uint32_t`** | **0.04645** | **11.12× [11.07, 11.28]** | ratio only | **24.28×** | **192,000** | 2 |
| binCV, `uint64_t` | 0.02527 | **20.49× [20.27, 20.67]** | ratio only | **46.48×** | 192,000 | 2 |
| binCV composed, `uint32_t` | 0.09700 | 5.33× [5.30, 5.38] | ratio only | 8.55× | 268,800 | 8 |

The aarch64 column is ratio-only for the same reason as denoise, and owes the same
re-measurement. The x86-64 column is thirty pinned launches
([log](logs/derivative-x86_64-launches.log)), and like denoise it is the first committed
source these cells have had.

The denominator is `cv::filter2D` twice with `[-1, 0, 1]` as a 1×3 and a 3×1 — the derivative
and nothing else. The reference implementation also multiplies by 16 and merges the two axes
into an interleaved two-channel image; binCV reproduces neither, so charging those to the
baseline would flatter binCV. That row is printed in the log and not used.

Some of this ratio is fixed per-call cost, measured separately on a 2×2 frame: at 640×480
OpenCV pays 1.85 µs of its 158.4 µs per frame against binCV's 0.012 µs of 14.27 µs — about 1%
of each side. At 94×60 it is most of what the per-pixel figure is made of, which is why the log
prints it per size.

## Morphology

The most mixed result in this report. 640×480, **nanoseconds per pixel**, same element,
anchor and border on both sides. **Bold marks a binCV cell that beats the OpenCV cell for
its architecture**:

| case | OpenCV, x86-64 | binCV `u32`, x86-64 | binCV `u64`, x86-64 | x86-64 ratio, u32 / u64 | OpenCV, aarch64 | binCV `u32`, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `erode` 3×3 rect, `BORDER_CONSTANT` | 0.1013 | **0.09595** | **0.06221** | 1.053× [1.035, 1.066] / **1.629× [1.577, 1.641]** | 0.71993 | 0.72012 | 1.00× |
| `dilate` 3×3 rect, `BORDER_CONSTANT` | 0.09972 | **0.09446** | **0.05378** | 1.057× [1.046, 1.073] / **1.857× [1.828, 1.872]** | 0.72416 | **0.48424** | 1.50× |
| `morphologyEx(OPEN)` 3×3 | 0.1956 | **0.1904** | **0.1156** | 1.022× [1.016, 1.043] / **1.688× [1.669, 1.716]** | 1.34128 | **1.20437** | 1.11× |
| `erode` 5×5 ellipse | 0.2238 | 0.6985 | 0.3453 | 0.319× [0.318, 0.323] / 0.648× [0.643, 0.656] | 1.81575 | 3.58631 | 0.51× |
| `erode` 3×3, `BORDER_REPLICATE` | 0.09870 | 0.1489 | 0.1151 | 0.666× [0.659, 0.672] / 0.853× [0.848, 0.870] | 0.67176 | 0.93004 | 0.72× |
| `erode` 3×3, `BORDER_REFLECT_101` | 0.09931 | 0.1553 | 0.1214 | 0.635× [0.628, 0.645] / 0.812× [0.805, 0.827] | 0.67158 | 0.94380 | 0.71× |

At the default `uint32_t` word binCV leads on three of the six cases on x86-64 and two on
aarch64; the wider word leads on the same three. Eight of the eighteen ratio figures are under
1.00×, and every one of those is `cv::erode` or `cv::dilate` ahead of binCV. The three x86-64
wins at `uint32_t` are narrow — 1.022× to 1.057× — and two of them have four of their thirty
launches on the other side of 1.00×; the `uint64_t` column is where the margin is.

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
| `cv::pyrDown`, `CV_8U` (the denominator) | 47.70 | — | 521.4 | — |
| **binCV `BOX_2x2`, 1 → 3 (shipped)** | **30.70** | 1.556× [1.536, 1.597] | **93.8** | **5.56×** |
| binCV `GAUSSIAN_5x5`, 1 → 3 | 184.95 | 0.259× [0.255, 0.266] | 599.0 | 0.87× |

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

Each is self-contained and needs no dataset. Taken as a launch sweep, which is what the
x86-64 cells above are:

```bash
./scripts/run_launches.sh -n 30 ./build/benchmark/logic_benchmark
./scripts/aggregate_launches.py logic_benchmark-x86_64-launches.log \
    --ratio 't1:bitwiseAnd OpenCV CV_8U/t1:bitwiseAnd binCV uint32'
```

Logs — the thirty-launch sweep behind each x86-64 column, the single launch it replaced, and
the device:
[logic](logs/logic-x86_64-launches.log), [single](logs/logic-x86_64.log), [aarch64](logs/logic-aarch64.log) ·
[reduce](logs/reduce-x86_64-launches.log), [single](logs/reduce-x86_64.log), [aarch64](logs/reduce-aarch64.log) ·
[denoise](logs/denoise-x86_64-launches.log), [single](logs/denoise-x86_64.log), [aarch64](logs/denoise-aarch64.log) ·
[derivative](logs/derivative-x86_64-launches.log), [single](logs/derivative-x86_64.log), [aarch64](logs/derivative-aarch64.log) ·
[morphology](logs/morphology-x86_64-launches.log), [single](logs/morphology-x86_64.log), [aarch64](logs/morphology-aarch64.log) ·
[pyrDown](logs/pyrfilter-x86_64-launches.log), [single](logs/pyrfilter-x86_64.log), [aarch64](logs/pyrfilter-aarch64.log)
