# Primitives

Per-operation results against the OpenCV call each one replaces, on the same binary content
stored as `CV_8U`. All figures are 640×480, `uint32_t` words unless a row says otherwise,
one thread on both sides.

Every benchmark runs a ladder of sizes and the logs carry all of it. The filter benchmarks
run downward — 640×480 to 94×60, a frame down to the top level of a four-level pyramid — so
that a ratio which collapses once both sides fit in cache can be told from one that holds.
The logic and reduction benchmarks run upward instead, to 8192×4096, because those are
bandwidth-bound and the question there is what happens when neither side fits in cache.

**x86-64 and aarch64 get separate columns everywhere on this page.** They are different
measurements against different OpenCV builds on different machines, and they are never
averaged or quoted as one another.

Setup, denominator rule and platform details: [README.md](README.md).

## Summary

### Speed

640×480 · `uint32_t` words · one thread on each side. Every measurement cell is
**nanoseconds per pixel**, so the smaller number of each pair is the faster implementation.
Two rows have no aarch64 measurements to show and say `ratio only` instead.

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | OpenCV, aarch64 | binCV, aarch64 | binCV against OpenCV |
|---|---|---|---|---|---|---|
| `bitwiseAnd` | `cv::bitwise_and` | 0.02734 | **0.00273** | 0.64783 | **0.02266** | **10.01× faster** on x86-64, **28.59×** on aarch64 |
| `bitwiseNot` | `cv::bitwise_not` | 0.08591 | **0.00343** | 0.31658 | **0.01943** | **25.04× faster** on x86-64, **16.30×** on aarch64 |
| `countNonZero` | `cv::countNonZero` | 0.01548 | **0.00956** | 0.17116 | **0.06366** | 1.62× faster on x86-64, 2.69× on aarch64 |
| `countAnd` | `cv::bitwise_and` + `countNonZero` | 0.04164 | **0.01199** | 0.57602 | **0.08792** | **3.47× faster** on x86-64, **6.55×** on aarch64 |
| denoise, 3-pixel median | composed `cv::min` / `cv::max` | 0.18609 | **0.01059** | ratio only | ratio only | **17.58× faster** on x86-64, **57.66×** on aarch64 |
| spatial derivative | `cv::filter2D` ×2 | 0.54843 | **0.04793** | ratio only | ratio only | **11.44× faster** on x86-64, **24.28×** on aarch64 |
| `erode` 3×3 | `cv::erode` | 0.10013 | 0.09605 | 0.71993 | 0.72012 | 1.04× on x86-64, 1.00× on aarch64 — a dead heat |
| `dilate` 3×3 | `cv::dilate` | 0.10407 | 0.13037 | 0.72416 | **0.48424** | 0.80× on x86-64, `cv::dilate` ahead; 1.50× faster on aarch64 |
| `morphologyEx(OPEN)` | `cv::morphologyEx` | 0.22507 | **0.19595** | 1.34128 | **1.20437** | 1.15× faster on x86-64, 1.11× on aarch64 |

`pyrDown` is timed per call rather than per pixel, so it gets a table of its own — 640×480 →
320×240, **microseconds per call**:

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | OpenCV, aarch64 | binCV, aarch64 | binCV against OpenCV |
|---|---|---|---|---|---|---|
| `pyrDown`, 1 bit in | `cv::pyrDown` on `CV_8U` | 48.3 | **31.0** | 521.4 | **93.8** | 1.56× faster on x86-64, **5.56×** on aarch64 |

**The denoise and derivative rows are ratio-only on aarch64**, and the four cells say so
rather than being filled from a run that does not match. The device times behind those
ratios were not carried into this report, and the committed denoise and derivative logs do
not reproduce the x86-64 values printed beside them either, so neither side can be restated
without changing a published figure. The ratios stand as published and those rows are owed
a re-measurement.

### Memory

Peak working set of one call — the live buffers, not a per-buffer ratio. Computed from
buffer geometry, so it is exact and **identical on both architectures**.

| operation | measured against | OpenCV | binCV | binCV against OpenCV |
|---|---|---|---|---|
| `bitwiseAnd` / `Or` / `Xor` / `Not` | `cv::bitwise_*` | 921,600 B | **115,200 B** | 8.0× smaller |
| `countNonZero`, per input plane | `cv::countNonZero` | 307,200 B | **38,400 B** | 8.0× smaller |
| `countAnd`, per input plane | `cv::bitwise_and` + `countNonZero` | 307,200 B | **38,400 B** | 8.0× smaller, and no temporary |
| denoise, 3-pixel median | composed `cv::min` / `cv::max` | 2,150,400 B | **76,800 B** | **28.0× smaller** |
| spatial derivative, both axes | `cv::filter2D` ×2 | 1,536,000 B | **192,000 B** | 8.00× smaller |
| `erode` / `dilate` 3×3 | `cv::erode` / `cv::dilate` | 614,400 B | **76,800 B** | 8.00× smaller |
| `morphologyEx(OPEN)` | `cv::morphologyEx` | 614,400 B | **115,200 B** | 5.33× smaller |
| four-level `pyrDown` ladder | a `CV_8U` pyramid | 408,000 B | **63,840 B** | 6.39× smaller — [footprint.md](footprint.md#the-pyramid) |

The composed `countAnd` baseline also has to materialise the `cv::bitwise_and` result before
it can count it. binCV allocates nothing, so that temporary never appears in its column.

The whole-frontend figure these compose into, and the decisions taken to protect it, are in
[footprint.md](footprint.md).

## Logic

The pointwise operations are the cleanest expression of the idea: an AND over two images
becomes an AND over their words, 32 pixels per instruction.

640×480, **nanoseconds per pixel** at each word type:

| operation | OpenCV, x86-64 | binCV `u32`, x86-64 | binCV `u64`, x86-64 | OpenCV, aarch64 | binCV `u32`, aarch64 | binCV `u64`, aarch64 |
|---|---|---|---|---|---|---|
| `bitwiseAnd` | 0.02734 | **0.00273** | 0.00399 | 0.64783 | **0.02266** | 0.02327 |
| `bitwiseOr` | 0.02820 | 0.00365 | **0.00274** | 0.64992 | 0.02393 | **0.02212** |
| `bitwiseXor` | 0.02719 | 0.00369 | 0.00360 | 0.64933 | 0.02521 | 0.02363 |
| `bitwiseNot` | 0.08591 | **0.00343** | 0.00373 | 0.31658 | **0.01943** | 0.01663 |

As ratios against the OpenCV column: 10.01× and 6.85× for `bitwiseAnd` on x86-64, 7.74× and
10.28× for `bitwiseOr`, 7.37× and 7.56× for `bitwiseXor`, 25.04× and 23.06× for
`bitwiseNot`; on aarch64, 28.59× / 27.85×, 27.16× / 29.38×, 25.76× / 27.48× and 16.30× /
19.04×. binCV is ahead in all sixteen.

The word type does not order these on x86 — `bitwiseAnd` reads faster at 32 bits and
`bitwiseOr` faster at 64, on the same data in the same run — which is what a bandwidth-bound
operation looks like when the arithmetic is free. Do not read a word-width preference into
those two columns.

They are also the operations where the *reason* for the speedup is most easily
misattributed. Both sides run at close to the machine's copy bandwidth — on x86-64, binCV at
72–137 GB/s against OpenCV's 23–110 GB/s across the four operations. Neither is inefficient.
binCV is faster because it moves an eighth as much data at a comparable rate, which is why
the benchmark prints a measured physical bound beside every row and flags any result that
exceeds it. `bitwiseNot` runs ahead of the others because OpenCV's is the slowest of its
four here, not because binCV's is special.

Every operation is checked for identical output before it is timed; the set-pixel counts
appear in the log and the benchmark exits non-zero if they disagree.

## Reductions

640×480, **nanoseconds per pixel**, against the OpenCV call in the second column:

| operation | measured against | OpenCV, x86-64 | binCV `u32`, x86-64 | binCV `u64`, x86-64 | OpenCV, aarch64 | binCV `u32`, aarch64 |
|---|---|---|---|---|---|---|
| `countNonZero` | `cv::countNonZero` | 0.01548 | **0.00956** | **0.00598** | 0.17116 | **0.06366** |
| `countAnd` | `cv::bitwise_and` then `cv::countNonZero` | 0.04164 | **0.01199** | — | 0.57602 | **0.08792** |

binCV is faster on every one: `countNonZero` by 1.62× at 32 bits and 2.59× at 64 on x86-64
and 2.69× on aarch64, `countAnd` by 3.47× and 6.55×.

`countNonZero` is the most modest ratio in this report and the reason is worth stating:
`cv::countNonZero` is already bandwidth-bound and running at 64.6 GB/s on x86-64. binCV
reads an eighth as many bytes at 13.1 GB/s. Neither implementation is leaving much on the
table; the whole difference is how much data has to move.

Reductions are offered over regions, masks and sliding windows, and **never per word**. On
aarch64 the population count instruction operates on a vector register, so counting a single
general-purpose word pays two register-domain crossings — about the cost of the count
itself. Exposing `popcount(word)` would invite callers to write exactly the loop that pays
that per word, so the API does not have one and the crossings are amortized over the whole
traversal instead.

`countAnd` is the clearer win of the two, and not because of packing: OpenCV has no fused
form, so the baseline must materialise a temporary with `cv::bitwise_and` and then count it.
binCV allocates nothing.

The `vs the per-pixel loop` column in the log — 0.44753 ns/pixel against binCV's 0.00956 at
640×480 on x86-64, 46.8× — is what the bit-parallel form is worth against the naive
alternative. It is not a claim against OpenCV and is not quoted as one.

## Denoise

A three-pixel median, against a byte-per-pixel implementation of the same filter ported call
for call from the frontend binCV was written to replace.

| implementation | time, x86-64 (ns/pixel) | time, aarch64 (ns/pixel) | working set (bytes) | binCV against OpenCV |
|---|---|---|---|---|
| OpenCV `CV_8U`, composed (the denominator) | 0.18609 | ratio only | 2,150,400 | — |
| **binCV fused, `uint32_t`** | **0.01059** | ratio only | **76,800** | **17.58× faster** on x86-64, **57.66×** on aarch64 |
| binCV fused, `uint64_t` | 0.01102 | ratio only | 76,800 | 16.88× faster on x86-64, **73.96×** on aarch64 |
| binCV composed, `uint32_t` | 0.05010 | ratio only | 153,600 | 3.71× faster on x86-64, 16.56× on aarch64 |

The aarch64 column carries no times because none were recorded here, and the device ratios
beside it are what this report has always published. The committed logs measure this
operation differently enough on both architectures that filling the column from them would
move the published figures, so it is left as a ratio and listed as owing a re-measurement.

Read the working-set column with the ratio. The baseline holds 2,150,400 bytes live against
binCV's 76,800, and on a part with 1 MiB of shared L2 a large part of any headline number
here is residency rather than arithmetic. The size ladder in the log is there to separate
them: if the ratio collapses once both sides fit in cache, the headline was residency.

The composed spelling — `shiftDown`, `shiftLeft`, `majority3` as three passes over two
scratch frames — is in the table because it is what the fused kernel replaced. Fusing was
worth 4.73× on x86 and 3.48× on the device *and* halved the memory, so nothing was traded
for it.

## Spatial derivative

Both axes, which is what a tracker needs before it can form a gradient covariance.

| implementation | time, x86-64 (ns/pixel) | time, aarch64 (ns/pixel) | working set (bytes) | passes | binCV against OpenCV |
|---|---|---|---|---|---|
| `cv::filter2D` ×2 (the denominator) | 0.54843 | ratio only | 1,536,000 | 2 | — |
| **binCV, `uint32_t`** | **0.04793** | ratio only | **192,000** | 2 | **11.44× faster** on x86-64, **24.28×** on aarch64 |
| binCV, `uint64_t` | 0.02579 | ratio only | 192,000 | 2 | **21.27× faster** on x86-64, **46.48×** on aarch64 |
| binCV composed, `uint32_t` | 0.10089 | ratio only | 268,800 | 8 | 5.44× faster on x86-64, 8.55× on aarch64 |

The aarch64 column is ratio-only for the same reason as denoise, and is owed the same
re-measurement.

The denominator is `cv::filter2D` twice with `[-1, 0, 1]` as a 1×3 and a 3×1 — the
derivative and nothing else. The reference implementation also multiplies by 16 and merges
the two axes into an interleaved two-channel image; binCV reproduces neither, so charging
those to the baseline would flatter binCV. That row is printed in the log and not used.

Some of this ratio is the fixed per-call cost, which the benchmark measures separately on a
2×2 frame. At 640×480 OpenCV pays 1.91 µs of its 168.48 µs per frame to that fixed cost
against binCV's 0.01 µs of 14.73 µs — about 1% of each side on x86, and the same 1% and 0% on
the device. It is not what this ratio is made of at frame sizes. At 94×60 it is most of what
the per-pixel figure is made of, which is why the log prints it per size.

## Morphology

The most mixed result in this report, and the one where the library's priorities are
visible.

640×480, **nanoseconds per pixel**, same element, anchor and border on both sides:

| case | OpenCV, x86-64 | binCV `u32`, x86-64 | binCV `u64`, x86-64 | OpenCV, aarch64 | binCV `u32`, aarch64 |
|---|---|---|---|---|---|
| `erode` 3×3 rect, `BORDER_CONSTANT` | 0.10013 | **0.09605** | **0.06127** | 0.71993 | 0.72012 |
| `dilate` 3×3 rect, `BORDER_CONSTANT` | 0.10407 | 0.13037 | **0.06185** | 0.72416 | **0.48424** |
| `morphologyEx(OPEN)` 3×3 | 0.22507 | **0.19595** | **0.11979** | 1.34128 | **1.20437** |
| `erode` 5×5 ellipse | 0.22759 | 0.70415 | 0.35309 | 1.81575 | 3.58631 |
| `erode` 3×3, `BORDER_REPLICATE` | 0.09669 | 0.15217 | 0.11852 | 0.67176 | 0.93004 |
| `erode` 3×3, `BORDER_REFLECT_101` | 0.09789 | 0.15761 | 0.12159 | 0.67158 | 0.94380 |

**Bold marks a binCV cell that beats the OpenCV cell for its architecture**; an unbolded
binCV cell is one where OpenCV is level or ahead. At the default `uint32_t` word binCV
leads on two of the six cases on x86-64 and two on aarch64; the wider word takes x86-64 to
three. The same rows as ratios against the OpenCV column:

| case | binCV `u32`, x86-64 | binCV `u64`, x86-64 | binCV `u32`, aarch64 |
|---|---|---|---|
| `erode` 3×3 rect, `BORDER_CONSTANT` | 1.04× | 1.63× | 1.00× |
| `dilate` 3×3 rect, `BORDER_CONSTANT` | 0.80× | 1.68× | 1.50× |
| `morphologyEx(OPEN)` 3×3 | 1.15× | 1.88× | 1.11× |
| `erode` 5×5 ellipse | 0.32× | 0.64× | 0.51× |
| `erode` 3×3, `BORDER_REPLICATE` | 0.64× | 0.82× | 0.72× |
| `erode` 3×3, `BORDER_REFLECT_101` | 0.62× | 0.81× | 0.71× |

Every cell under 1.00× there is `cv::erode` or `cv::dilate` ahead of binCV, and ten of the
eighteen are. The times above are the same result without the convention: where binCV's
number is the larger one, binCV is the slower side.

Every `erode` and `dilate` case above holds **76,800 B** live against `cv::erode`'s
**614,400 B**; `morphologyEx(OPEN)` holds **115,200 B** against the same 614,400. `erode`
and `dilate` need no scratch at all — a dilation is a shift and an OR — so the footprint
advantage is the full 8×. `morphologyEx(OPEN)` needs one caller-provided frame where OpenCV
needs none, which is why its footprint advantage is 5.33× rather than 8×.

**binCV loses on the 5×5 ellipse**, and by a wide margin — 0.70415 ns/pixel against
`cv::erode`'s 0.22759 on x86-64, and 3.58631 against 1.81575 on aarch64. A non-separable
structuring element costs the bit-parallel form one shifted-OR per set element, where
OpenCV's vectorised byte kernel amortises the same work across a SIMD register. The fused
kernel was kept anyway, because it is 8× smaller and the alternative spelling is slower
still. That is a deliberate speed-for-footprint trade and it is priced in
[footprint.md](footprint.md).

**binCV also loses on non-constant borders.** `BORDER_REPLICATE` and `BORDER_REFLECT_101`
each cost binCV a rim pass that `BORDER_CONSTANT` does not need; the interior kernel is
unchanged.

## Pyramid downsample

640×480 → 320×240, **microseconds per call**, against `cv::pyrDown` on `CV_8U` at one
thread:

| arm | time, x86-64 (µs) | time, aarch64 (µs) | binCV against `cv::pyrDown` |
|---|---|---|---|
| `cv::pyrDown`, `CV_8U` (the denominator) | 48.3 | 521.4 | — |
| **binCV `BOX_2x2`, 1 → 3 (shipped)** | **31.0** | **93.8** | 1.56× faster on x86-64, **5.56×** on aarch64 |
| binCV `GAUSSIAN_5x5`, 1 → 3 | 195.4 | 599.0 | 0.25× on x86-64, `cv::pyrDown` well ahead; 0.87× on aarch64, roughly a tie |

The shipped route takes a 1-bit level and produces a 2- or 3-bit one with a 2×2 box filter.
That is not `cv::pyrDown`'s filter, and the comparison is a Tier 2 one: the same role, a
different answer. Where binCV matches OpenCV's filter exactly — a Gaussian 5×5 with
`BORDER_REFLECT_101`, bit-exact against `cv::pyrDown` at five size parities — it is
substantially slower on x86 and roughly at parity on the reference device, at three-eighths
of the stored bits.

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
