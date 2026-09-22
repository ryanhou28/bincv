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
cells beside it. **Every aarch64 figure is the median of ten pinned launches** with the
same bootstrap interval on its ratio —
[methodology-timing.md](methodology-timing.md#the-protocol-each-host-needs) says why ten
there and thirty here, and [the index](README.md#on-the-x86-64-host) says what changed when
each column was re-taken.

## Summary

### Speed

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Every measurement cell is **nanoseconds per pixel**, so the smaller number of each pair is
the faster implementation. Every row now has both machines' times: the two that read `ratio
only` were the denoise and derivative rows, and the device sweep that re-took this column
recorded them.

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `bitwiseAnd` | `cv::bitwise_and` | 0.02823 | **0.002810** | **9.97× [9.82, 10.28]** | 0.62656 | **0.02369** | **26.68× [26.09, 27.37]** |
| `bitwiseNot` | `cv::bitwise_not` | 0.06336 | **0.003510** | **17.99× [17.82, 18.14]** | 0.31522 | **0.01965** | **16.06× [15.67, 16.72]** |
| `countNonZero` | `cv::countNonZero` | 0.01501 | **0.009270** | 1.62× [1.61, 1.63] | 0.16921 | **0.06365** | 2.658× [2.618, 2.673] |
| `countAnd` | `cv::bitwise_and` + `countNonZero` | 0.04172 | **0.01196** | **3.49× [3.45, 3.52]** | 0.54791 | **0.08775** | **6.242× [6.155, 6.313]** |
| denoise, 3-pixel median | composed `cv::min` / `cv::max` | 0.1887 | **0.009865** | **19.09× [18.94, 19.38]** | 3.4379 | **0.05941** | **57.71× [56.91, 58.03]** |
| spatial derivative | `cv::filter2D` ×2 | 0.5156 | **0.04645** | **11.12× [11.07, 11.28]** | 5.0430 | **0.20753** | **24.28× [24.13, 24.51]** |
| `erode` 3×3 | `cv::erode` | 0.1013 | **0.09595** | 1.053× [1.035, 1.066] | 0.73595 | **0.72189** | 1.021× [0.991, 1.040] |
| `dilate` 3×3 | `cv::dilate` | 0.09972 | **0.09446** | 1.057× [1.046, 1.073] | 0.73419 | **0.48573** | **1.507× [1.476, 1.533]** |
| `morphologyEx(OPEN)` | `cv::morphologyEx` | 0.1956 | **0.1904** | 1.022× [1.016, 1.043] | 1.37899 | **1.20347** | 1.146× [1.122, 1.166] |

`pyrDown` is timed per call rather than per pixel — 640×480 → 320×240, **microseconds per
call**:

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `pyrDown`, 1 bit in | `cv::pyrDown` on `CV_8U` | 47.70 | **30.70** | 1.556× [1.536, 1.597] | 516.5 | **93.8** | **5.509× [5.480, 5.549]** |

**`dilate` 3×3 was published as a 0.80× loss on x86-64, and it is not one.** Thirty launches
put it at 1.057×, ahead in 30 of 30, on a kernel whose source has not changed since the single
launch that produced 0.80× — that launch timed binCV's arm at 0.13037 ns/pixel where thirty
span 0.09260 to 0.09680. It is the one row in these reports whose *sign* the single-launch
protocol got wrong, on either machine. Two rows moved the other way and are recorded as such:
`bitwiseNot` reads 17.99× rather than 25.04× because `cv::bitwise_not` was slow in the old
launch, and `morphologyEx(OPEN)` reads 1.022× rather than 1.15×, four of its thirty launches
falling below 1.00. [The index](README.md#on-the-x86-64-host) has the whole comparison.

**The aarch64 column has nothing of that kind in it.** Ten launches of every benchmark on
this page reproduce the published device ratios, and the cells that moved moved because the
`cv::` denominator did: `countAnd` reads 6.242× rather than 6.55× on a binCV arm that shifted
0.19%, and `morphologyEx(OPEN)` reads 1.146× rather than 1.11× on one that shifted 0.07%.
The one figure whose *size* is worth a second look is `bitwiseAnd`'s 28.59×, now 26.68× — and
that is the protocol rather than the kernel, because the packed 38 KB plane's residency
varies per launch and the row scatters 17.8% across ten of them. 28.59× sits inside the
observed per-launch range of 25.58× to 30.32×. [The index](README.md#on-the-aarch64-device)
has the whole comparison.

**The denoise and derivative rows are no longer ratio-only.** Their x86-64 columns got a
committed source when the x86 sweep was taken — the previously published x86 values appear
nowhere in this repository's logs — and their aarch64 columns get one here.

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
| `bitwiseAnd` | 0.02823 | **0.002810** | 0.003755 | **9.97× [9.82, 10.28]** / 7.58× [7.35, 7.75] | 0.62656 | **0.02369** | 0.02319 | 26.68× [26.09, 27.37] / 27.13× [26.49, 28.52] |
| `bitwiseOr` | 0.02794 | 0.003665 | **0.002825** | 7.69× [7.41, 7.78] / **9.99× [9.88, 10.10]** | 0.64293 | 0.02383 | **0.02272** | 27.04× [26.05, 27.63] / 28.41× [27.31, 28.75] |
| `bitwiseXor` | 0.02779 | 0.003605 | 0.003610 | 7.70× [7.62, 7.78] / 7.72× [7.52, 7.80] | 0.64685 | 0.02347 | 0.02286 | 27.62× [26.61, 28.76] / 28.25× [27.59, 28.74] |
| `bitwiseNot` | 0.06336 | **0.003510** | 0.003540 | **17.99× [17.82, 18.14]** / 17.91× [17.76, 18.15] | 0.31522 | **0.01965** | 0.01924 | 16.06× [15.67, 16.72] / 16.57× [15.56, 16.88] |

binCV is ahead in all sixteen, in every one of the launches behind each cell on both
machines. The word type does not order these on x86 — `bitwiseAnd` reads faster at 32 bits and
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
| `countNonZero` | `cv::countNonZero` | 0.01501 | **0.009270** | **0.005895** | 1.62× [1.61, 1.63] / 2.55× [2.53, 2.57] | 0.16921 | **0.06365** | 2.658× [2.618, 2.673] |
| `countAnd` | `cv::bitwise_and` then `cv::countNonZero` | 0.04172 | **0.01196** | — | **3.49× [3.45, 3.52]** / — | 0.54791 | **0.08775** | 6.242× [6.155, 6.313] |

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
| OpenCV `CV_8U`, composed (the denominator) | 0.1887 | — | 3.4379 | — | 2,150,400 |
| **binCV fused, `uint32_t`** | **0.009865** | **19.09× [18.94, 19.38]** | **0.05941** | **57.71× [56.91, 58.03]** | **76,800** |
| binCV fused, `uint64_t` | 0.007385 | **25.39× [25.16, 25.80]** | 0.04731 | **72.67× [72.24, 73.03]** | 76,800 |
| binCV composed, `uint32_t` | 0.04820 | 3.91× [3.88, 3.95] | 0.21097 | 16.28× [16.16, 16.40] | 153,600 |

The aarch64 column read `ratio only` for three rounds because the device times behind those
ratios were never carried into this report. They are here now, ten pinned launches
([log](logs/denoise-aarch64-launches.log)), and the ratios they produce are the published
ones: 57.66× re-reads 57.71×. The x86-64 column is thirty pinned launches
([log](logs/denoise-x86_64-launches.log)); the values it replaces reproduced from no committed
log at all, which is how a figure with no source stays in a table.

Read the working-set column with the ratio: on a part with 1 MiB of shared L2, a large part of
any headline number here is residency rather than arithmetic, and the size ladder in the log
separates them — if the ratio collapses once both sides fit in cache, the headline was
residency.

The composed spelling — `shiftDown`, `shiftLeft`, `majority3` as three passes over two scratch
frames — is what the fused kernel replaced. Fusing was worth 4.91× on x86 and 3.55× on the
device *and* halved the memory, so nothing was traded for it.

## Spatial derivative

Both axes, which is what a tracker needs before it can form a gradient covariance.

| implementation | x86-64 (ns/px) | x86-64 ratio | aarch64 (ns/px) | aarch64 ratio | working set (bytes) | passes |
|---|---|---|---|---|---|---|
| `cv::filter2D` ×2 (the denominator) | 0.5156 | — | 5.0430 | — | 1,536,000 | 2 |
| **binCV, `uint32_t`** | **0.04645** | **11.12× [11.07, 11.28]** | **0.20753** | **24.28× [24.13, 24.51]** | **192,000** | 2 |
| binCV, `uint64_t` | 0.02527 | **20.49× [20.27, 20.67]** | 0.10870 | **46.37× [46.09, 46.81]** | 192,000 | 2 |
| binCV composed, `uint32_t` | 0.09700 | 5.33× [5.30, 5.38] | 0.58679 | 8.626× [8.561, 8.665] | 268,800 | 8 |

The aarch64 column is measured for the same reason as denoise's, ten pinned launches
([log](logs/derivative-aarch64-launches.log)), and it reproduces the published 24.28×
to four figures. The x86-64 column is thirty pinned launches
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
| `erode` 3×3 rect, `BORDER_CONSTANT` | 0.1013 | **0.09595** | **0.06221** | 1.053× [1.035, 1.066] / **1.629× [1.577, 1.641]** | 0.73595 | **0.72189** | 1.021× [0.991, 1.040] |
| `dilate` 3×3 rect, `BORDER_CONSTANT` | 0.09972 | **0.09446** | **0.05378** | 1.057× [1.046, 1.073] / **1.857× [1.828, 1.872]** | 0.73419 | **0.48573** | **1.507× [1.476, 1.533]** |
| `morphologyEx(OPEN)` 3×3 | 0.1956 | **0.1904** | **0.1156** | 1.022× [1.016, 1.043] / **1.688× [1.669, 1.716]** | 1.37899 | **1.20347** | **1.146× [1.122, 1.166]** |
| `erode` 5×5 ellipse | 0.2238 | 0.6985 | 0.3453 | 0.319× [0.318, 0.323] / 0.648× [0.643, 0.656] | 1.85196 | 3.59587 | 0.514× [0.510, 0.522] |
| `erode` 3×3, `BORDER_REPLICATE` | 0.09870 | 0.1489 | 0.1151 | 0.666× [0.659, 0.672] / 0.853× [0.848, 0.870] | 0.69852 | 0.92952 | 0.752× [0.741, 0.769] |
| `erode` 3×3, `BORDER_REFLECT_101` | 0.09931 | 0.1553 | 0.1214 | 0.635× [0.628, 0.645] / 0.812× [0.805, 0.827] | 0.70016 | 0.94370 | 0.742× [0.727, 0.759] |

At the default `uint32_t` word binCV leads on three of the six cases on each machine, and the
wider word leads on the same three. Nine of the eighteen ratio figures are under 1.00×, and
every one of those is `cv::erode` or `cv::dilate` ahead of binCV. The narrow wins are narrow
on both hosts and the launches say so: x86-64's `erode` 3×3 and `morphologyEx(OPEN)` each put
four of thirty launches on the other side of 1.00×, and aarch64's `erode` 3×3 reads 1.021×
with six of ten above 1.00 and an interval that straddles it — **a dead heat measured ten
times is still a dead heat**. The `uint64_t` column is where the margin is.

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
| `cv::pyrDown`, `CV_8U` (the denominator) | 47.70 | — | 516.5 | — |
| **binCV `BOX_2x2`, 1 → 3 (shipped)** | **30.70** | 1.556× [1.536, 1.597] | **93.8** | **5.509× [5.480, 5.549]** |
| binCV `GAUSSIAN_5x5`, 1 → 3 | 184.95 | 0.259× [0.255, 0.266] | 599.1 | 0.862× [0.858, 0.868] |

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

Each is self-contained and needs no dataset. Taken as a launch sweep, which is what every
cell above is — thirty launches on x86-64, ten on the device with the governor locked:

```bash
./scripts/run_launches.sh -n 30 ./build/benchmark/logic_benchmark
./scripts/aggregate_launches.py logic_benchmark-x86_64-launches.log \
    --ratio 't1:bitwiseAnd OpenCV CV_8U/t1:bitwiseAnd binCV uint32'
```

Logs — the sweep behind each column, and the single launch each replaced:
[logic](logs/logic-x86_64-launches.log), [single](logs/logic-x86_64.log), [aarch64](logs/logic-aarch64-launches.log), [single](logs/logic-aarch64.log) ·
[reduce](logs/reduce-x86_64-launches.log), [single](logs/reduce-x86_64.log), [aarch64](logs/reduce-aarch64-launches.log), [single](logs/reduce-aarch64.log) ·
[denoise](logs/denoise-x86_64-launches.log), [single](logs/denoise-x86_64.log), [aarch64](logs/denoise-aarch64-launches.log), [single](logs/denoise-aarch64.log) ·
[derivative](logs/derivative-x86_64-launches.log), [single](logs/derivative-x86_64.log), [aarch64](logs/derivative-aarch64-launches.log), [single](logs/derivative-aarch64.log) ·
[morphology](logs/morphology-x86_64-launches.log), [single](logs/morphology-x86_64.log), [aarch64](logs/morphology-aarch64-launches.log), [single](logs/morphology-aarch64.log) ·
[pyrDown](logs/pyrfilter-x86_64-launches.log), [single](logs/pyrfilter-x86_64.log), [aarch64](logs/pyrfilter-aarch64-launches.log), [single](logs/pyrfilter-aarch64.log)
