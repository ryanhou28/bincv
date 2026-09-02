# Primitives

Per-operation results against the OpenCV call each one replaces, on the same binary content
stored as `CV_8U`. All figures are 640×480, `uint32_t` words unless a row says otherwise,
one thread on both sides.

Every benchmark runs a ladder of sizes and the logs carry all of it. The filter benchmarks
run downward — 640×480 to 94×60, a frame down to the top level of a four-level pyramid — so
that a ratio which collapses once both sides fit in cache can be told from one that holds.
The logic and reduction benchmarks run upward instead, to 8192×4096, because those are
bandwidth-bound and the question there is what happens when neither side fits in cache.

Setup, denominator rule and platform details: [README.md](README.md).

## Summary

Speedup against OpenCV, at 640×480. A ratio below 1.00 means binCV is slower, and several
are.

| operation | OpenCV call | x86-64 | aarch64 | smaller by |
|---|---|---|---|---|
| `bitwiseAnd` | `cv::bitwise_and` | **10.01×** | **28.59×** | 8.0× |
| `bitwiseNot` | `cv::bitwise_not` | **25.04×** | **16.30×** | 8.0× |
| `countNonZero` | `cv::countNonZero` | 1.62× | 2.69× | 8.0× |
| `countAnd` | `cv::bitwise_and` + `countNonZero` | **3.47×** | **6.55×** | 8.0× |
| denoise, 3-pixel median | composed `cv::min` / `cv::max` | **17.58×** | **57.66×** | 28.0× |
| spatial derivative | `cv::filter2D` ×2 | **11.44×** | **24.28×** | 8.0× |
| `erode` 3×3 | `cv::erode` | 1.04× | 1.00× | 8.0× |
| `dilate` 3×3 | `cv::dilate` | 0.80× | 1.50× | 8.0× |
| `morphologyEx(OPEN)` | `cv::morphologyEx` | 1.15× | 1.11× | 5.33× |
| `pyrDown`, 1 bit in | `cv::pyrDown` | 1.56× | **5.56×** | see below |

## Logic

The pointwise operations are the cleanest expression of the idea: an AND over two images
becomes an AND over their words, 32 pixels per instruction.

Speedup against OpenCV at 640×480, per word type:

| operation | x86 `uint32_t` | x86 `uint64_t` | aarch64 `uint32_t` | aarch64 `uint64_t` |
|---|---|---|---|---|
| `bitwiseAnd` | 10.01× | 6.85× | **28.59×** | 27.85× |
| `bitwiseOr` | 7.74× | 10.28× | **27.16×** | 29.38× |
| `bitwiseXor` | 7.37× | 7.56× | 25.76× | 27.48× |
| `bitwiseNot` | 25.04× | 23.06× | **16.30×** | 19.04× |

The word type does not order these on x86 — `bitwiseAnd` reads faster at 32 bits and
`bitwiseOr` faster at 64, on the same data in the same run — which is what a bandwidth-bound
operation looks like when the arithmetic is free. Do not read a word-width preference into
that column.

They are also the operations where the *reason* for the speedup is most easily
misattributed. Both sides run at close to the machine's copy bandwidth — binCV at 72–137
GB/s, OpenCV at 23–110 GB/s across the four operations. Neither is inefficient. binCV is
faster because it moves an eighth as much data at a comparable rate, which is why the
benchmark prints a measured physical bound beside every row and flags any result that
exceeds it. `bitwiseNot` runs ahead of the others because OpenCV's is the slowest of its
four here, not because binCV's is special.

Every operation is checked for identical output before it is timed; the set-pixel counts
appear in the log and the benchmark exits non-zero if they disagree.

## Reductions

640×480, against the OpenCV call in the second column:

| binCV | denominator | x86 `uint32_t` | x86 `uint64_t` | aarch64 `uint32_t` |
|---|---|---|---|---|
| `countNonZero` | `cv::countNonZero` | 1.62× | 2.59× | 2.69× |
| `countAnd` | `cv::bitwise_and` then `cv::countNonZero` | **3.47×** | — | **6.55×** |

`countNonZero` is the most modest ratio in this report and the reason is worth stating:
`cv::countNonZero` is already bandwidth-bound and running at 64.6 GB/s. binCV reads an
eighth as many bytes at 13.1 GB/s. Neither implementation is leaving much on the table; the
whole difference is how much data has to move.

Reductions are offered over regions, masks and sliding windows, and **never per word**. On
aarch64 the population count instruction operates on a vector register, so counting a single
general-purpose word pays two register-domain crossings — about the cost of the count
itself. Exposing `popcount(word)` would invite callers to write exactly the loop that pays
that per word, so the API does not have one and the crossings are amortized over the whole
traversal instead.

`countAnd` is the clearer win of the two, and not because of packing: OpenCV has no fused
form, so the baseline must materialise a temporary with `cv::bitwise_and` and then count it.
binCV allocates nothing.

The `vs the per-pixel loop` column in the log — 46.8× at 640×480 — is what the bit-parallel
form is worth against the naive alternative. It is not a claim against OpenCV and is not
quoted as one.

## Denoise

A three-pixel median, against a byte-per-pixel implementation of the same filter ported call
for call from the frontend binCV was written to replace.

| implementation | x86 ns/pixel | vs OpenCV | aarch64 | working set |
|---|---|---|---|---|
| OpenCV `CV_8U` (the denominator) | 0.18609 | 1.00× | 1.00× | 2,150,400 B |
| binCV fused, `uint32_t` | 0.01059 | **17.58×** | **57.66×** | 76,800 B |
| binCV fused, `uint64_t` | 0.01102 | 16.88× | **73.96×** | 76,800 B |
| binCV composed, `uint32_t` | 0.05010 | 3.71× | 16.56× | 153,600 B |

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

| implementation | x86 ns/pixel | vs OpenCV | aarch64 | working set | passes |
|---|---|---|---|---|---|
| `cv::filter2D` ×2 (the denominator) | 0.54843 | 1.00× | 1.00× | 1,536,000 B | 2 |
| binCV, `uint32_t` | 0.04793 | **11.44×** | **24.28×** | 192,000 B | 2 |
| binCV, `uint64_t` | 0.02579 | **21.27×** | **46.48×** | 192,000 B | 2 |
| binCV composed, `uint32_t` | 0.10089 | 5.44× | 8.55× | 268,800 B | 8 |

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

Speedup against the matching OpenCV call, same element, anchor and border:

| case | x86 `uint32_t` | x86 `uint64_t` | aarch64 `uint32_t` | footprint |
|---|---|---|---|---|
| `erode` 3×3 rect, `BORDER_CONSTANT` | 1.04× | 1.63× | 1.00× | 8.00× |
| `dilate` 3×3 rect, `BORDER_CONSTANT` | 0.80× | 1.68× | 1.50× | 8.00× |
| `morphologyEx(OPEN)` 3×3 | 1.15× | 1.88× | 1.11× | 5.33× |
| `erode` 5×5 **ellipse** | **0.32×** | 0.64× | **0.51×** | 8.00× |
| `erode` 3×3, `BORDER_REPLICATE` | 0.64× | 0.82× | 0.72× | 8.00× |
| `erode` 3×3, `BORDER_REFLECT_101` | 0.62× | 0.81× | 0.71× | 8.00× |

`erode` and `dilate` need no scratch at all — a dilation is a shift and an OR — so the
footprint advantage is the full 8×. `morphologyEx(OPEN)` needs one caller-provided frame
where OpenCV needs none, which is why its footprint row is 5.33×.

**binCV loses on the 5×5 ellipse**, and by a wide margin. A non-separable structuring
element costs the bit-parallel form one shifted-OR per set element, where OpenCV's
vectorised byte kernel amortises the same work across a SIMD register. The fused kernel was
kept anyway, because it is 8× smaller and the alternative spelling is slower still. That is
a deliberate speed-for-footprint trade and it is priced in [footprint.md](footprint.md).

**binCV also loses on non-constant borders.** `BORDER_REPLICATE` and `BORDER_REFLECT_101`
each cost binCV a rim pass that `BORDER_CONSTANT` does not need; the interior kernel is
unchanged.

## Pyramid downsample

640×480 → 320×240, against `cv::pyrDown` on `CV_8U` at one thread:

| arm | x86 µs | vs `cv::pyrDown` | aarch64 µs | vs `cv::pyrDown` |
|---|---|---|---|---|
| binCV `BOX_2x2`, 1 → 3 (shipped) | 31.0 | **1.56×** | 93.8 | **5.56×** |
| `cv::pyrDown`, `CV_8U` (the denominator) | 48.3 | 1.00× | 521.4 | 1.00× |
| binCV `GAUSSIAN_5x5`, 1 → 3 | 195.4 | 0.25× | 599.0 | 0.87× |

The shipped route takes a 1-bit level and produces a 2- or 3-bit one with a 2×2 box filter.
That is not `cv::pyrDown`'s filter, and the comparison is a Tier 2 one: the same role, a
different answer. Where binCV matches OpenCV's filter exactly — a Gaussian 5×5 with
`BORDER_REFLECT_101`, bit-exact against `cv::pyrDown` at five size parities — it is
substantially slower on x86 and roughly at parity on the reference device, at three-eighths
of the stored bits.

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
