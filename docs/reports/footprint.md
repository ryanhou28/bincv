# Memory footprint

**A binCV tracking frontend holds 436,704 bytes live where the OpenCV equivalent holds
2,719,832 — 6.23× smaller, on both architectures, for the same work.**

Footprint is the claim binCV exists to make. Speed varies by operation, by architecture and
by what the compiler did that day; the byte count does not. It is computed from buffer
geometry, it is exact, and it is the same number on x86-64 and aarch64.

For how the counting works and why it is arithmetic rather than a sampled RSS, see
[README.md](README.md#how-the-numbers-are-taken).

## The frontend, end to end

Peak working set over the whole frontend operation set — two pyramids, the derivative
ladders, and whatever the corner stage holds:

| | bytes | what it is |
|---|---|---|
| binCV | **436,704** | `1/2/2/2` pyramid ×2, derivative ladders, 3-row response ring |
| OpenCV | **2,719,832** | `CV_8U` pyramid ×2 with a 31-pixel border per level, `CV_32F` eigen map |
| | **6.23× smaller** | |

Most of that is structural rather than won by packing. binCV carries **no `winSize` border**
on any pyramid level, and **no frame-sized float response map**. Those two decisions are
worth more than the eight-to-one storage ratio is:

- The reference tracker pads every pyramid level by the window width so a window near the
  edge can be read without clipping. binCV clips instead. Measured, the border is worse or
  equal on keypoint yield in five of seven cases and better by at most 1.4 points in the
  other two, for 1.38× the bytes.
- A corner detector that materialises a `float` response map for the whole frame spends
  1,228,800 bytes at 640×480 — on its own more than everything else in the frontend
  combined. binCV sweeps a **three-row ring** instead: 7,680 bytes, and it is *also* faster,
  because the frame-sized map was never buying locality worth its size. That single change
  took the corner stage from 1,333,848 bytes to 112,744, and the frontend from 1,721,568 to
  500,464 at 640×480.

## Per operation

Working set of one call — the live buffers, not a per-buffer ratio — at 640×480, `uint32_t`
words, against the same binary content stored as `CV_8U`:

| operation | binCV | OpenCV | smaller by |
|---|---|---|---|
| `erode` / `dilate`, 3×3 | 76,800 B | 614,400 B | **8.00×** |
| `morphologyEx(MORPH_OPEN)` | 115,200 B | 614,400 B | **5.33×** |
| denoise, three-pixel median | 76,800 B | 2,150,400 B | **28.0×** |
| spatial derivative, both axes | 192,000 B | 1,536,000 B | **8.00×** |
| `goodFeaturesToTrack` | 1,580,064 B | 9,014,976 B | **5.71×** |
| FAST input plane | 46,080 B | 360,960 B | **7.83×** |

Three of those deserve the qualification they carry.

**`morphologyEx` is 5.33×, not 8×**, because binCV's fused kernel needs a caller-provided
scratch frame — three frames live against OpenCV's two — where `erode` and `dilate` need
none. `cv::morphologyEx` allocates nothing of its own for `OPEN`, which was probed rather
than assumed: the process high-water mark does not move around the call. Reporting 8× here
would have been wrong by a factor of 1.5 on the number most likely to be quoted.

**`goodFeaturesToTrack` is 5.71× at the measured survivor count and 2.23× when both sides
are provisioned for their worst case.** Both are in [the log](logs/goodfeatures-x86_64.log);
the pessimistic one is the safer number to design against. The gap is the candidate array,
which is a per-frame reading rather than a bound: a binarized min-eigenvalue map takes few
distinct values, so large numbers of pixels tie and survive non-maximum suppression, and how
many survive depends on the frame. Size that pool from the ranked count rather than from
`maxCorners`, and watch the truncation flag the API returns — the structural worst case is
every interior pixel.

**The denoise 28× is not a packing result.** The reference implementation composes the
filter out of `cv::min` and `cv::max` over zero-filled neighbour matrices and holds seven
buffers live; binCV's fused kernel holds two and makes one pass. Most of that ratio is the
composition, not the bit width.

## The pyramid

A four-level pyramid at 640×480, with each level capped at the bit depth its arithmetic can
actually reach. The `vs 8U` column is a **computed** byte-per-pixel-per-level denominator,
not a timed OpenCV run:

| ladder | bits per level | bytes | vs `CV_8U` |
|---|---|---|---|
| uncapped | 1/3/5/7 | 84,240 | 4.84× |
| reference-shaped | 1/3/4/5 | 80,400 | 5.07× |
| | 1/3/3/3 | 76,560 | 5.33× |
| **shipped** | **1/2/2/2** | **63,840** | **6.39×** |
| re-binarized | 1/1/1/1 | 51,120 | 7.98× |

The useful finding here is how *little* room there is. Level 0 is 38,400 of those bytes and
no cap touches it, so the entire range from uncapped to re-binarized spans 1.65× — against
the 4.84× to 7.98× already won over a byte-per-pixel pyramid. Choosing a ladder is a
tracking-accuracy decision with a small footprint side effect, not a footprint lever.

## What footprint costs, and what it does not buy

Two results are worth stating plainly because they cut against the obvious reading.

**Eight times less data does not make the tracker eight times faster.** With one level and a
31×31 window, growing the frame 36-fold from 320×240 to 1920×1440 — at a fixed 140 keypoints,
so the compute is identical and only the data grows — moves the per-point cost by 12% on x86
and 5% on the device. A 31×31 window is 120 bytes at one bit per pixel, two to four cache
lines, and it would be two to four cache lines as bytes too. Lucas–Kanade is compute-bound,
so the footprint advantage decides what fits on a device and not how fast it runs. Further
speed there has to come from doing less work, not from touching less data.

**Threading is free in memory.** Peak resident set size across one, four and twelve tracking
threads is 29,180 / 29,164 / 29,156 KB — a 0.08% spread, and it moves *downward*, so the
whole range is noise. Tracking splits over keypoints, each thread writes only its own
outputs, and the only per-thread cost is stack.

This row is a sampled RSS rather than a computed working set, because thread stacks are
exactly the thing buffer arithmetic does not see. It is whole-process — it includes the
OpenCV side of the benchmark — so it is a bound on the effect, not a measurement of binCV's
own peak. As a bound it is enough: nothing grows.

## Where speed was declined to protect it

The project's rule is that when speed and footprint conflict and nothing else settles it,
footprint wins. That rule has fired, and these are the bills:

| decision | speed offered | footprint cost | outcome |
|---|---|---|---|
| `uint64_t` as the default word type | **1.95×** on `countNonZero` | +20% at 160×120, +33% at 94×60 | **declined** |
| an occupancy mask for spacing detections | *slower* — 23.7× at the operating point | 38,400 B | **declined twice over** |
| fused morphology kernel | costs up to 3.1× on a 5×5 ellipse | 8× smaller | **accepted, and it costs** |
| interleaved bit-plane layout † | +8% on the frontend | +21% of the frontend peak | **declined** |

The word-type row is the canonical one. `uint64_t` is genuinely 1.95× faster on
`countNonZero` at 640×480 on the reference device, and it was turned down, because a wider
word rounds each row's stride up more coarsely and the upper pyramid levels are exactly where
a small target is tightest. The wider word costs +20% at 160×120 and +33% at 94×60, measured
by exact stride arithmetic.

**A 64-bit caller loses nothing for that choice.** On little-endian a 64-bit bit-plane already
*is* a 32-bit bit-plane at twice the stride, so it is reinterpreted rather than converted — no
copy, no allocation. Measured on `edgeThreshold` at 640×480, a narrowed view runs at 1.00× of
a native 32-bit buffer on the reference device and 0.97× on x86, with 0 of 307,200 pixels
differing. The same buffer taken down the scalar fallback instead runs at 0.12× and 0.02×,
which is what the narrowing exists to avoid.

The occupancy-mask row is the easiest of them, because the mask lost on both axes at once.
Spacing new detections against live tracks by marking a 1-bit occupancy frame costs 38,400
bytes, and at the benchmark's stated operating point — 120 live tracks, 300 candidates, 80
free slots — testing each candidate against the live set directly is **23.7× faster**. The
mask only catches up past about 5,000 candidates, an order of magnitude more than a detection
top-up produces. The rule required the mask to be faster to justify its bytes; it was not
close.

**† The interleaved-layout row is the one figure on this page that cannot be re-run from a
committed benchmark.** It was measured with a one-off probe that is not part of the
repository, so it is reported here as development history rather than as a reproducible
claim. The decision it records is the point: the layout was 1.445× on the extraction it was
built for and would have taken the frontend from about 1.52× to 1.65× against OpenCV — for
92,160 additional bytes on a 436,704-byte peak, taking the footprint result from 6.23× to
5.15×. Twenty-one percent of the footprint advantage for eight percent of the speed is not a
trade this library makes. The 1.52× baseline it was weighed against is an older frontend
figure, superseded by the 3.36× and 4.73× in [frontend.md](frontend.md); the proportions are
what the decision turned on.

Two other figures on this page come from that same development record rather than from a
benchmark in this repository: the pyramid border's keypoint-yield comparison, and the
corner-stage restructuring that took the frontend from 1,721,568 to 500,464 bytes at 640×480.
The buffer sizes in both are exact arithmetic and reproduce anywhere; the yield and speed
comparisons around them do not have a committed benchmark here.

## Reproduce

```bash
./build/benchmark/frontend_sequence <euroc-cam0-dir>   # frontend peak, both sides
./build/benchmark/morphology_benchmark                 # working set per call
./build/benchmark/derivative_benchmark
./build/benchmark/denoise_benchmark
./build/benchmark/corner_opencv_benchmark              # itemized, both sides
./build/benchmark/pyramid_benchmark                    # ladder bytes; computed denominator
./build/benchmark/wordwidth_benchmark                  # word type against footprint
BINCV_LK_THREADS=4 /usr/bin/time -v ./build/benchmark/frontend_sequence <dir> 600
./build/benchmark/spacing_benchmark                    # the occupancy mask that lost
./build/benchmark/wordtype_narrow                      # 64-bit callers: narrow, do not convert
```

Logs: [frontend](logs/frontend-x86_64.log) ·
[morphology](logs/morphology-x86_64.log), [aarch64](logs/morphology-aarch64.log) ·
[derivative](logs/derivative-x86_64.log), [aarch64](logs/derivative-aarch64.log) ·
[denoise](logs/denoise-x86_64.log), [aarch64](logs/denoise-aarch64.log) ·
[goodFeaturesToTrack](logs/goodfeatures-x86_64.log), [aarch64](logs/goodfeatures-aarch64.log) ·
[pyramid](logs/pyramid-x86_64.log), [aarch64](logs/pyramid-aarch64.log) ·
[word width](logs/wordwidth-x86_64.log), [aarch64](logs/wordwidth-aarch64.log) ·
[peak RSS against threads](logs/frontend-rss-x86_64.log) ·
[spacing](logs/spacing-x86_64.log) ·
[64-bit narrowing](logs/wordtype_narrow-x86_64.log), [aarch64](logs/wordtype_narrow-aarch64.log)
