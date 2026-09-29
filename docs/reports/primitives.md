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
there and thirty here. Times are quoted to four significant figures, ratios and intervals to
three. Bold marks the row name of the shipped configuration where a table has more than one.

## Summary

### Speed

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Every measurement cell is **nanoseconds per pixel**, so the smaller number of each pair is
the faster implementation.

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `bitwiseAnd` | `cv::bitwise_and` | 0.02823 | 0.002810 | 9.97× [9.82, 10.3] | 0.6266 | 0.02369 | 26.7× [26.1, 27.4] |
| `bitwiseNot` | `cv::bitwise_not` | 0.06336 | 0.003510 | 18.0× [17.8, 18.1] | 0.3152 | 0.01965 | 16.1× [15.7, 16.7] |
| `countNonZero` | `cv::countNonZero` | 0.01501 | 0.009270 | 1.62× [1.61, 1.63] | 0.1692 | 0.06365 | 2.66× [2.62, 2.67] |
| `countAnd` | `cv::bitwise_and` + `countNonZero` | 0.04172 | 0.01196 | 3.49× [3.45, 3.52] | 0.5479 | 0.08775 | 6.24× [6.16, 6.31] |
| denoise, 3-pixel median | composed `cv::min` / `cv::max` | 0.1926 | 0.009930 | 19.2× [19.0, 19.5] | 3.438 | 0.05941 | 57.7× [56.9, 58.0] |
| spatial derivative | `cv::filter2D` ×2 | 0.5156 | 0.04645 | 11.1× [11.1, 11.3] | 5.043 | 0.2075 | 24.3× [24.1, 24.5] |
| `erode` 3×3 | `cv::erode` | 0.1013 | 0.09595 | 1.05× [1.04, 1.07] | 0.7360 | 0.7219 | 1.02× [0.991, 1.04] |
| `dilate` 3×3 | `cv::dilate` | 0.09972 | 0.09446 | 1.06× [1.05, 1.07] | 0.7342 | 0.4857 | 1.51× [1.48, 1.53] |
| `morphologyEx(OPEN)` | `cv::morphologyEx` | 0.1956 | 0.1904 | 1.02× [1.02, 1.04] | 1.379 | 1.203 | 1.15× [1.12, 1.17] |

`pyrDown` is timed per call rather than per pixel — 640×480 → 320×240, **microseconds per
call**:

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `pyrDown`, 1 bit in | `cv::pyrDown` on `CV_8U` | 47.70 | 30.70 | 1.56× [1.54, 1.60] | 516.5 | 93.8 | 5.51× [5.48, 5.55] |

Three rows are closer than their ratio makes them look. `erode` 3×3 and `morphologyEx(OPEN)`
on x86-64 each put four of their thirty launches on the other side of 1.00×, and `erode` 3×3
on the device reads 1.02× with an interval that straddles it: those are dead heats, measured
thirty and ten times. `dilate` 3×3 on x86-64 is ahead in 30 of 30 launches.

### Memory

Peak working set of one call — the live buffers, not a per-buffer ratio. Computed from
buffer geometry, so it is exact and **identical on both architectures**.

| operation | measured against | OpenCV | binCV | ratio |
|---|---|---|---|---|
| `bitwiseAnd` / `Or` / `Xor` / `Not` | `cv::bitwise_*` | 921,600 B | 115,200 B | 8.00× |
| `countNonZero`, per input plane | `cv::countNonZero` | 307,200 B | 38,400 B | 8.00× |
| `countAnd`, per input plane | `cv::bitwise_and` + `countNonZero` | 307,200 B | 38,400 B | 8.00×, and no temporary |
| denoise, 3-pixel median | composed `cv::min` / `cv::max` | 2,150,400 B | 76,800 B | 28.0× |
| spatial derivative, both axes | `cv::filter2D` ×2 | 1,536,000 B | 192,000 B | 8.00× |
| `erode` / `dilate` 3×3 | `cv::erode` / `cv::dilate` | 614,400 B | 76,800 B | 8.00× |
| `morphologyEx(OPEN)` | `cv::morphologyEx` | 614,400 B | 115,200 B | 5.33× |
| four-level `pyrDown` ladder | a `CV_8U` pyramid | 408,000 B | 63,840 B | 6.39× — [footprint.md](footprint.md#the-pyramid) |

The composed `countAnd` baseline has to materialise the `cv::bitwise_and` result before it
can count it. binCV allocates nothing, so that temporary never appears in its column.

## Logic

An AND over two images becomes an AND over their words, 32 pixels per instruction. 640×480,
**nanoseconds per pixel**, one row per word type:

| operation | word | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `bitwiseAnd` | `u32` | 0.02823 | 0.002810 | 9.97× [9.82, 10.3] | 0.6266 | 0.02369 | 26.7× [26.1, 27.4] |
| `bitwiseAnd` | `u64` | 0.02823 | 0.003755 | 7.58× [7.35, 7.75] | 0.6266 | 0.02319 | 27.1× [26.5, 28.5] |
| `bitwiseOr` | `u32` | 0.02794 | 0.003665 | 7.69× [7.41, 7.78] | 0.6429 | 0.02383 | 27.0× [26.1, 27.6] |
| `bitwiseOr` | `u64` | 0.02794 | 0.002825 | 9.99× [9.88, 10.1] | 0.6429 | 0.02272 | 28.4× [27.3, 28.8] |
| `bitwiseXor` | `u32` | 0.02779 | 0.003605 | 7.70× [7.62, 7.78] | 0.6469 | 0.02347 | 27.6× [26.6, 28.8] |
| `bitwiseXor` | `u64` | 0.02779 | 0.003610 | 7.72× [7.52, 7.80] | 0.6469 | 0.02286 | 28.3× [27.6, 28.7] |
| `bitwiseNot` | `u32` | 0.06336 | 0.003510 | 18.0× [17.8, 18.1] | 0.3152 | 0.01965 | 16.1× [15.7, 16.7] |
| `bitwiseNot` | `u64` | 0.06336 | 0.003540 | 17.9× [17.8, 18.2] | 0.3152 | 0.01924 | 16.6× [15.6, 16.9] |

binCV is ahead in all sixteen, in every one of the launches behind each cell on both
machines. The word type does not order these on x86-64 — `bitwiseAnd` reads faster at 32 bits
and `bitwiseOr` faster at 64, on the same data in the same run — which is what a
bandwidth-bound operation looks like when the arithmetic is free. Do not read a word-width
preference into the two rows of an operation.

On the device the 640×480 ratios include L2 residency: binCV's 115 KB working set (three
packed planes; KB is 1000 bytes here) fits the 1 MiB shared L2 and OpenCV's 921 KB does not. Further up the size
ladder in the same log, `bitwiseAnd` reads 13.6× [13.3, 14.3] at 1024×1024 and 8.07×
[7.98, 8.17] at 8192×4096, where neither side fits — the data ratio alone.

The *reason* for the speedup is easily misattributed. Both sides run close to the machine's
copy bandwidth — on x86-64, binCV at 71–133 GB/s against OpenCV's 32–108 GB/s across the four
— so neither is inefficient: binCV is faster because it moves an eighth as much data at a
comparable rate. The benchmark prints a measured physical bound beside every row and flags any
result that exceeds it. `bitwiseNot` leads the others because OpenCV's is the slowest of its
four here, not because binCV's is special.

Every operation is checked for identical output before it is timed; the set-pixel counts
appear in the log and the benchmark exits non-zero if they disagree.

## Reductions

640×480, **nanoseconds per pixel**, one row per word type; the device column is `uint32_t`
only:

| operation | word | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|---|
| `countNonZero` | `u32` | `cv::countNonZero` | 0.01501 | 0.009270 | 1.62× [1.61, 1.63] | 0.1692 | 0.06365 | 2.66× [2.62, 2.67] |
| `countNonZero` | `u64` | `cv::countNonZero` | 0.01501 | 0.005895 | 2.55× [2.53, 2.57] | — | — | — |
| `countAnd` | `u32` | `cv::bitwise_and` then `cv::countNonZero` | 0.04172 | 0.01196 | 3.49× [3.45, 3.52] | 0.5479 | 0.08775 | 6.24× [6.16, 6.31] |

`countNonZero` is the most modest ratio in this report. `cv::countNonZero` runs at
66.6 GB/s on x86-64 — memory bandwidth. binCV reads an eighth as many bytes but at
13.5 GB/s, so its popcount loop, not memory, is the limit, and the 1.62× is what an eighth
of the bytes buys at a fifth of the rate; the `uint64_t` word lifts it to 2.55×. `countAnd`
is the clearer win, and not because of packing: OpenCV has no fused form, so the baseline
must materialise a temporary.

Reductions are offered over regions, masks and sliding windows, and **never per word**. On
aarch64 the population count instruction operates on a vector register, so counting a single
general-purpose word pays two register-domain crossings — about the cost of the count itself.
The API therefore has no `popcount(word)`, and the crossings are amortized over the traversal.

The `vs the per-pixel loop` column in the log — 0.4505 ns/pixel against binCV's 0.009270 at
640×480 on x86-64, 48.5× [48.0, 48.7] — is what the bit-parallel form is worth against the
naive alternative. It is not a claim against OpenCV and is not quoted as one.

## Denoise

A three-pixel median, against a byte-per-pixel implementation of the same filter ported call
for call from the reference pipeline (the visual-inertial odometry system, not in this
repository, that binCV was built to serve stage by stage; see
[docs/ARCHITECTURE.md](../ARCHITECTURE.md)).

| implementation | x86-64 (ns/px) | x86-64 ratio | aarch64 (ns/px) | aarch64 ratio | working set (bytes) |
|---|---|---|---|---|---|
| OpenCV `CV_8U`, composed (the denominator) | 0.1926 | — | 3.438 | — | 2,150,400 |
| **binCV fused, `uint32_t` (shipped)** | 0.009930 | 19.2× [19.0, 19.5] | 0.05941 | 57.7× [56.9, 58.0] | 76,800 |
| binCV fused, `uint64_t` | 0.007385 | 25.4× [25.2, 25.8] | 0.04731 | 72.7× [72.2, 73.0] | 76,800 |
| binCV composed, `uint32_t` | 0.04838 | 3.92× [3.88, 3.99] | 0.2110 | 16.3× [16.2, 16.4] | 153,600 |

Read the working-set column with the ratio: on a part with 1 MiB of shared L2, a large part of
any headline number here is residency rather than arithmetic, and the size ladder in the log
separates them — if the ratio collapses once both sides fit in cache, the headline was
residency.

The composed spelling — `shiftDown`, `shiftLeft`, `majority3` as three passes over two scratch
frames — is what the fused kernel replaced. Fusing was worth 4.86× [4.83, 4.90] on x86-64
and 3.55× [3.48, 3.56] on the device (paired over the same launches) *and* halved the
memory, so nothing was traded for it.

## Spatial derivative

Both axes, which is what a tracker needs before it can form a gradient covariance.

| implementation | x86-64 (ns/px) | x86-64 ratio | aarch64 (ns/px) | aarch64 ratio | working set (bytes) | passes |
|---|---|---|---|---|---|---|
| `cv::filter2D` ×2 (the denominator) | 0.5156 | — | 5.043 | — | 1,536,000 | 2 |
| **binCV, `uint32_t` (shipped)** | 0.04645 | 11.1× [11.1, 11.3] | 0.2075 | 24.3× [24.1, 24.5] | 192,000 | 2 |
| binCV, `uint64_t` | 0.02527 | 20.5× [20.3, 20.7] | 0.1087 | 46.4× [46.1, 46.8] | 192,000 | 2 |
| binCV composed, `uint32_t` | 0.09700 | 5.33× [5.30, 5.38] | 0.5868 | 8.63× [8.56, 8.67] | 268,800 | 8 |

The denominator is `cv::filter2D` twice with `[-1, 0, 1]` as a 1×3 and a 3×1 — the derivative
and nothing else. The reference pipeline also multiplies by 16 and merges the two axes
into an interleaved two-channel image; binCV reproduces neither, so charging those to the
baseline would flatter binCV. That row is printed in the log and not used.

Some of this ratio is fixed per-call cost, measured separately on a 2×2 frame: at 640×480
OpenCV pays 1.85 µs of its 158.4 µs per frame against binCV's 0.012 µs of 14.27 µs — about 1%
of each side. At 94×60 it is most of what the per-pixel figure is made of, which is why the log
prints it per size.

## Morphology

The most mixed result in this report. 640×480, **nanoseconds per pixel**, same element,
anchor and border on both sides, one row per word type; the device column is `uint32_t`
only:

| case | word | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `erode` 3×3 rect, `BORDER_CONSTANT` | `u32` | 0.1013 | 0.09595 | 1.05× [1.04, 1.07] | 0.7360 | 0.7219 | 1.02× [0.991, 1.04] |
| ″ | `u64` | 0.1013 | 0.06221 | 1.63× [1.58, 1.64] | — | — | — |
| `dilate` 3×3 rect, `BORDER_CONSTANT` | `u32` | 0.09972 | 0.09446 | 1.06× [1.05, 1.07] | 0.7342 | 0.4857 | 1.51× [1.48, 1.53] |
| ″ | `u64` | 0.09972 | 0.05378 | 1.86× [1.83, 1.87] | — | — | — |
| `morphologyEx(OPEN)` 3×3 | `u32` | 0.1956 | 0.1904 | 1.02× [1.02, 1.04] | 1.379 | 1.203 | 1.15× [1.12, 1.17] |
| ″ | `u64` | 0.1956 | 0.1156 | 1.69× [1.67, 1.72] | — | — | — |
| `erode` 5×5 ellipse | `u32` | 0.2238 | 0.6985 | 0.319× [0.318, 0.323] | 1.852 | 3.596 | 0.514× [0.510, 0.522] |
| ″ | `u64` | 0.2238 | 0.3453 | 0.648× [0.643, 0.656] | — | — | — |
| `erode` 3×3, `BORDER_REPLICATE` | `u32` | 0.09870 | 0.1489 | 0.666× [0.659, 0.672] | 0.6985 | 0.9295 | 0.752× [0.741, 0.769] |
| ″ | `u64` | 0.09870 | 0.1151 | 0.853× [0.848, 0.870] | — | — | — |
| `erode` 3×3, `BORDER_REFLECT_101` | `u32` | 0.09931 | 0.1553 | 0.635× [0.628, 0.645] | 0.7002 | 0.9437 | 0.742× [0.727, 0.759] |
| ″ | `u64` | 0.09931 | 0.1214 | 0.812× [0.805, 0.827] | — | — | — |

At the default `uint32_t` word binCV leads on three of the six cases on each machine, and the
wider word leads on the same three. Nine of the eighteen ratios are under 1.00×, and every
one of those is `cv::erode` or `cv::dilate` ahead of binCV. The narrow wins are narrow on
both hosts and the launches say so: x86-64's `erode` 3×3 and `morphologyEx(OPEN)` each put
four of thirty launches on the other side of 1.00×, and aarch64's `erode` 3×3 reads 1.02×
with six of ten above 1.00 and an interval that straddles it — **a dead heat measured ten
times is still a dead heat**. The `uint64_t` rows are where the margin is.

Every `erode` and `dilate` case above holds **76,800 B** live against `cv::erode`'s
**614,400 B**; `morphologyEx(OPEN)` holds **115,200 B** against the same 614,400. `erode`
and `dilate` need no scratch at all — a dilation is a shift and an OR — so the footprint
advantage is the full 8.00×. `morphologyEx(OPEN)` needs one caller-provided frame where
OpenCV needs none, which is why its footprint advantage is 5.33× rather than 8.00×.

**binCV loses on the 5×5 ellipse**, and by a wide margin. A non-separable structuring
element costs the bit-parallel form one shifted-OR per set element, where OpenCV's
vectorised byte kernel amortises the same work across a SIMD register. The fused kernel was
kept anyway, because it is 8.00× smaller and the alternative spelling is slower still — a
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
| **binCV `BOX_2x2`, 1 → 3 (shipped)** | 30.70 | 1.56× [1.54, 1.60] | 93.8 | 5.51× [5.48, 5.55] |
| binCV `GAUSSIAN_5x5`, 1 → 3 | 185.0 | 0.259× [0.255, 0.266] | 599.1 | 0.862× [0.858, 0.868] |

The shipped route takes a 1-bit level and produces a 2- or 3-bit one with a 2×2 box filter.
That is not `cv::pyrDown`'s filter, so the comparison is Tier 2: the same role, a different
answer. Where binCV matches OpenCV's filter exactly — a Gaussian 5×5 with `BORDER_REFLECT_101`,
bit-exact against `cv::pyrDown` for even and odd widths and heights — it is substantially
slower on x86-64 and roughly at parity on the device, at three-eighths of the stored bits.

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
cell above is — thirty launches on x86-64, ten on the device with the governor locked
(`-g`):

```bash
./scripts/run_launches.sh -n 30 ./build/benchmark/logic_benchmark
./scripts/aggregate_launches.py logic_benchmark-x86_64-launches.log \
    --ratio 't1:bitwiseAnd OpenCV CV_8U/t1:bitwiseAnd binCV uint32'
```

The x86-64 sweeps carry their aggregate appended as a comment block; the aarch64 sweeps do
not, and their intervals come from running the aggregator on the log as above.

## Logs

The sweep behind each column, and the single launch each replaced:
[logic](logs/logic-x86_64-launches.log), [single](logs/logic-x86_64.log), [aarch64](logs/logic-aarch64-launches.log), [single](logs/logic-aarch64.log) ·
[reduce](logs/reduce-x86_64-launches.log), [single](logs/reduce-x86_64.log), [aarch64](logs/reduce-aarch64-launches.log), [single](logs/reduce-aarch64.log) ·
[denoise](logs/denoise-x86_64-launches.log), [single](logs/denoise-x86_64.log), [aarch64](logs/denoise-aarch64-launches.log), [single](logs/denoise-aarch64.log) ·
[derivative](logs/derivative-x86_64-launches.log), [single](logs/derivative-x86_64.log), [aarch64](logs/derivative-aarch64-launches.log), [single](logs/derivative-aarch64.log) ·
[morphology](logs/morphology-x86_64-launches.log), [single](logs/morphology-x86_64.log), [aarch64](logs/morphology-aarch64-launches.log), [single](logs/morphology-aarch64.log) ·
[pyrDown](logs/pyrfilter-x86_64-launches.log), [single](logs/pyrfilter-x86_64.log), [aarch64](logs/pyrfilter-aarch64-launches.log), [single](logs/pyrfilter-aarch64.log)

Logs marked stale in [expected-stale.txt](logs/expected-stale.txt) were taken before a
bit-identical change to the code beneath them; the file records which files moved and why
the figure stands.

## Changes to published figures

| date | row | previous | current | why |
|---|---|---|---|---|
| 2026-09-21 | `dilate` 3×3, x86-64 | 0.800× (one launch) | 1.06× [1.05, 1.07] | the single launch timed binCV's arm at 0.13037 ns/px where thirty launches span 0.09260–0.09680; ahead in 30 of 30 |
| 2026-09-21 | `bitwiseNot`, x86-64 | 25.0× (one launch) | 18.0× [17.8, 18.1] | `cv::bitwise_not` read 0.08591 ns/px in the single launch where thirty span 0.06156–0.07519 |
| 2026-09-21 | `morphologyEx(OPEN)`, x86-64 | 1.15× (one launch) | 1.02× [1.02, 1.04] | four of thirty launches fall below 1.00× |
| 2026-09-21 | denoise and spatial derivative, x86-64 times | figures with no committed log | thirty launches | the earlier figures appear in no log in this repository |
| 2026-09-21 | `countAnd`, aarch64 | 6.55× (one launch) | 6.24× [6.16, 6.31] | the `cv::` denominator moved; binCV's arm moved 0.19% |
| 2026-09-21 | `morphologyEx(OPEN)`, aarch64 | 1.11× (one launch) | 1.15× [1.12, 1.17] | the `cv::` denominator moved; binCV's arm moved 0.07% |
| 2026-09-21 | `bitwiseAnd`, aarch64 | 28.6× (one launch) | 26.7× [26.1, 27.4] | the packed plane's cache residency varies per launch; the ten launches span 25.6×–30.3× and the ratio scatters 17.8% (binCV's arm 16.9%, OpenCV's 4.0%) |
| 2026-09-21 | denoise and spatial derivative, aarch64 times | ratio only (57.66×, 24.28×) | both times, ten launches | the device times behind the ratios had never been carried into this page |
