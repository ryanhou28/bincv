# Memory footprint

Footprint is the claim binCV exists to make. Speed varies by operation, by architecture and
by what the compiler did that day; the byte count does not. Every figure here is computed
from buffer geometry, so it is exact and identical on x86-64 and aarch64. How the counting
works: [README.md](README.md#how-the-numbers-are-taken). KB on this page is 1000 bytes and
KiB is 1024.

## Per operation

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Working set of one call — the live buffers, not a per-buffer ratio — at 640×480, `uint32_t`
words, against the same binary content stored as `CV_8U`:

| operation | measured against | OpenCV, bytes | binCV, bytes | ratio |
|---|---|---|---|---|
| `erode` / `dilate`, 3×3 | `cv::erode` / `cv::dilate` | 614,400 | 76,800 | 8.00× |
| `morphologyEx(MORPH_OPEN)` | `cv::morphologyEx` | 614,400 | 115,200 | 5.33× |
| denoise, three-pixel median | composed `cv::min` / `cv::max` | 2,150,400 | 76,800 | 28.0× |
| spatial derivative, both axes | `cv::filter2D` ×2 | 1,536,000 | 192,000 | 8.00× |
| `goodFeaturesToTrack`, at the measured survivor count | `cv::goodFeaturesToTrack`, binarized | 9,014,976 | 1,580,064 | 5.71× |
| `goodFeaturesToTrack`, worst-case provisioning | `cv::goodFeaturesToTrack`, binarized | 11,348,512 | 5,080,368 | 2.23× |
| FAST input plane | `cv::FAST` on `CV_8U` | 360,960 | 46,080 | 7.83× |

Three of those rows carry a qualification that changes how the figure reads.

**`morphologyEx` is 5.33×, not 8.00×**, because binCV's fused kernel needs a caller-provided
scratch frame — three frames live against OpenCV's two — where `erode` and `dilate` need
none. Reporting 8.00× here would have been wrong by a factor of 1.5 on the number most likely
to be quoted. `cv::morphologyEx` does not materialise an intermediate frame for `OPEN`, but it
is not allocation-free either: measured at the allocator with the destination preallocated
it takes **5,784 B** of row buffers at 640×480, and that figure tracks width rather than
area (3,224 B at 320 wide, 10,904 B at 1,280 wide, unchanged when only the height doubles).
Counting it puts the row at 620,184 B and 5.38×, so the headline is unchanged.

**`goodFeaturesToTrack` has two footprints, because it has two spellings**, and two rows
above, because its largest buffer is sized by the content. At worst-case provisioning the
frame-map form holds 16.54 bytes per pixel and the streaming form 12.56 — a 1,228,800-byte
response map against a 7,680-byte three-row ring. Both return the same corners; the streaming
form is what every pipeline here calls.

Both rows in the table are the frame-map spelling, sized symmetrically with the denominator:
binCV's corner array and OpenCV's candidate pointer vector are provisioned for every interior
pixel in the worst-case row (the structural worst case, since every interior pixel of an
all-equal response map is a 3×3 maximum) and for the 13,272 survivors measured over four
frames in the other. The worst-case row is the one to design against: a binarized
min-eigenvalue map takes few distinct values, so large numbers of pixels tie and survive
non-maximum suppression, and the candidate pool should be sized from the ranked count rather
than from `maxCorners`, with the truncation flag watched. The streaming spelling's worst case
is 3,859,248 B — 12.56 bytes per pixel against the 29.00 that stock `cv::goodFeaturesToTrack`,
the denominator [features.md](features.md#corner-detection) uses for speed, materialises
inside the call, 2.31× smaller. The table keeps the binarized denominator because stock's
buffers are *accounted* and not read: `cornerMinEigenVal` materializes `Dx`, `Dy` and a
`CV_32FC3` covariance inside the call and `gftt` adds `eig` and a dilate destination, none of
which a caller can measure from outside, and every figure in this table is read off the
objects.

**The denoise 28.0× is not a packing result.** The reference pipeline (the visual-inertial
odometry system, not in this repository, that binCV was built to serve stage by stage; see
[docs/ARCHITECTURE.md](../ARCHITECTURE.md)) composes the filter out of `cv::min` and
`cv::max` over zero-filled neighbour matrices and holds seven buffers live; binCV's fused
kernel holds two and makes one pass. Most of that ratio is the composition, not the bit width.

## The pyramid

A four-level pyramid at 640×480, each level capped at the bit depth its arithmetic can
reach. The `CV_8U` column is a **computed** byte-per-pixel-per-level denominator, not a timed
OpenCV run; bold marks the shipped ladder:

| ladder | bits per level | `CV_8U` pyramid, bytes | binCV, bytes | `CV_8U` ÷ binCV |
|---|---|---|---|---|
| uncapped | 1/3/5/7 | 408,000 | 84,240 | 4.84× |
| reference-shaped | 1/3/4/5 | 408,000 | 80,400 | 5.07× |
| | 1/3/3/3 | 408,000 | 76,560 | 5.33× |
| **shipped** | 1/2/2/2 | 408,000 | 63,840 | 6.39× |
| re-binarized | 1/1/1/1 | 408,000 | 51,120 | 7.98× |

The useful finding is how *little* room there is. Level 0 is 38,400 of those bytes and no cap
touches it, so the entire range from uncapped to re-binarized spans 1.65× — against the 4.84×
to 7.98× already won over a byte-per-pixel pyramid. Choosing a ladder is a tracking-accuracy
decision with a small footprint side effect, not a footprint lever.

## Assembled: the whole feature tracking pipeline

The figure below is not a per-operation result — it is the pipeline
[feature-tracking.md](feature-tracking.md) describes, held up as evidence that the
per-call savings compose:

|  | OpenCV | binCV | ratio |
|---|---|---|---|
| peak working set at 752×480, bytes | 2,719,832 | 436,704 | 6.23× |
| what it holds | `CV_8U` pyramid ×2 with a 31-pixel border per level, `CV_32F` eigen map | `1/2/2/2` pyramid ×2, derivative ladders, 3-row response ring | — |

Most of that is structural rather than won by packing. binCV carries **no `winSize` border**
on any pyramid level and **no frame-sized float response map**, and those two decisions are
worth more than the eight-to-one storage ratio is:

- The reference tracker pads every pyramid level by the window width so a window near the
  edge can be read without clipping. binCV clips instead. Measured, the border is worse or
  equal on keypoint yield in five of seven cases and better by at most 1.4 percentage points
  of yield in the other two, for 1.38× the bytes.
- A corner detector that materialises a `float` response map for the whole frame spends
  1,228,800 bytes at 640×480 — on its own more than everything else in the pipeline
  combined. binCV sweeps a **three-row ring** instead: 7,680 bytes, and it is *also* faster.
  That single change took the corner stage from 1,333,848 bytes to 112,744, and the pipeline
  from 1,721,568 to 500,464 — figures for an earlier pipeline shape at 640×480, from the
  uncommitted probe named below, and not comparable with the 752×480 peak in the table.

## What footprint costs, and what it does not buy

**Eight times less data does not make the tracker eight times faster.** Growing the frame
36-fold at a fixed keypoint count moves Lucas–Kanade's per-point cost by 0.4% on x86-64 and
6% on the device: it is compute-bound, so the footprint advantage decides what fits on a
device and not how fast it runs. The sweep is in [limits.md](limits.md).

**Threading is free in memory.** Peak resident set size across one, four and twelve tracking
threads is 29,180 KiB, 29,164 KiB and 29,156 KiB — a 0.08% spread, and it moves *downward*,
so the whole range is noise. This row is a sampled RSS rather than a computed working set,
because thread stacks are exactly what buffer arithmetic does not see, and it is
whole-process — it includes the OpenCV side of the benchmark — so it is a bound on the
effect rather than a measurement of binCV's own peak. As a bound it is enough: nothing grows.

## Where speed was declined to protect it

When speed and footprint conflict and nothing else settles it, footprint wins. That rule has
fired, and these are the bills. Speed and memory are separate columns, each carrying what was
measured; a cell holding one number is a figure whose other side did not survive, and it says
so.

| decision | speed | memory | outcome |
|---|---|---|---|
| `uint64_t` as the default word type | 1.95× [1.95, 1.95] faster on `countNonZero` at 640×480, aarch64 | 2,880 B against `uint32_t`'s 2,400 at 160×120 and 960 against 720 at 94×60 — 1.20× and 1.33× more | declined |
| an occupancy mask for spacing detections | 76,077 ns against the direct test's 3,281 on x86-64 — the mask is 23.2× [23.1, 23.3] slower | 38,400 B against the direct test's 0 | declined twice over |
| fused morphology kernel | 0.6985 ns/pixel against `cv::erode`'s 0.2238 on a 5×5 ellipse, x86-64 — binCV at 0.319× | 76,800 B against `cv::erode`'s 614,400 — 8.00× smaller | accepted, and it costs |
| interleaved bit-plane layout † | 1.08× on the pipeline — not reproducible | +92,160 B on a 436,704-byte peak, 1.21× — not reproducible | declined |

The word-type row is the canonical one. `uint64_t` is 1.95× faster on `countNonZero` at
640×480 on the reference device, and it was turned down, because a wider word rounds each
row's stride up more coarsely and the upper pyramid levels are exactly where a small target is
tightest.

Two committed benchmarks time that comparison, and they disagree by two per cent for a reason
worth knowing. The reduce benchmark reads 1.95× [1.95, 1.95] over ten device launches
([log](logs/reduce-aarch64-launches.log)); the word-width sweep reads 1.91× over ten
launches, tighter still ([log](logs/wordwidth-aarch64-launches.log)). Both are right: the
word-width sweep interleaves all four widths in one batch — 1,350 KiB resident, past the
device's 1 MiB L2 — where the reduce benchmark times one width at a time inside a
38,400-byte plane. The gap is what residency is worth here. The row quotes the reduce
figure, which is the one that matches the operation as a caller runs it.

**A 64-bit caller loses nothing for that choice.** On little-endian a 64-bit bit-plane already
*is* a 32-bit bit-plane at twice the stride, so it is reinterpreted rather than converted — no
copy, no allocation. Measured on `edgeThreshold` at 640×480 against a native 32-bit buffer as
the baseline, with 0 of 307,200 pixels differing:

| arm | x86-64 (ns) | x86-64, native ÷ this arm | aarch64 (ns) | aarch64, native ÷ this arm |
|---|---|---|---|---|
| native `uint32_t` buffer | 17,650 | — | 257,350 | — |
| `uint64_t` buffer, narrowed view | 18,380 | 0.961× [0.960, 0.961] | 257,489 | 1.00× [0.999, 1.00] |
| `uint64_t` buffer, scalar fallback | 678,902 | 0.0260× [0.0260, 0.0260] | 2,163,581 | 0.119× [0.119, 0.119] |

The narrowed view runs at 0.961× of the native buffer on x86-64 — a real cost rather than a
measurement artefact, and the tightest interval in these reports: thirty launches of the row
scatter 1.2% and resolve a difference of 1.001× — and at 1.00× on the device. The same
buffer taken down the scalar fallback instead is what the narrowing exists to avoid.

The occupancy-mask row lost on both axes at once. At the benchmark's stated operating point —
120 live tracks, 300 candidates, 80 free slots, on x86-64:

|  | direct test against the live set | a 1-bit occupancy frame | mask ÷ direct |
|---|---|---|---|
| time (ns) | 3,281 | 76,077 | 23.2× [23.1, 23.3] |
| memory (bytes) | 0 | 38,400 | — |

The mask only catches up past about 5,000 candidates, an order of magnitude more than a
detection top-up produces. The rule required the mask to be faster to justify its bytes; it
was not close.

**† The interleaved-layout row cannot be re-run from a committed benchmark.** It was measured
with a one-off probe that is not in the repository, so it is development history rather than a
reproducible claim. The decision is the point: the layout was 1.45× on the extraction it was
built for and would have taken the pipeline from about 1.52× to 1.65× against OpenCV — for
92,160 additional bytes on a 436,704-byte peak, taking the footprint result from 6.23× to
5.15×. A 1.21× larger footprint for a 1.08× faster pipeline is not a trade this library
makes. (The 1.52× baseline is an older pipeline figure; the current ones are 3.97× and 5.32×
in [feature-tracking.md](feature-tracking.md), and the proportions are what the decision
turned on.)

Two other figures here come from that same record rather than a committed benchmark: the
pyramid border's keypoint-yield comparison, and the corner-stage restructuring from 1,721,568
to 500,464 bytes. The buffer sizes in both are exact arithmetic; the yield and speed
comparisons around them are not reproducible here.

The morphology row's speed cost has also been quoted as "up to 3.13× on a 5×5 ellipse" — the
same measurement written the other way up, published as binCV at 0.319× in
[primitives.md](primitives.md#morphology).

## Reproduce

```bash
./build/benchmark/feature_tracking_sequence <euroc-cam0-dir>   # pipeline peak, both sides
./build/benchmark/morphology_benchmark                 # working set per call
./build/benchmark/derivative_benchmark
./build/benchmark/denoise_benchmark
./build/benchmark/corner_opencv_benchmark              # itemized, both sides
./build/benchmark/pyramid_benchmark                    # ladder bytes; computed denominator
./build/benchmark/wordwidth_benchmark                  # word type against footprint
BINCV_LK_THREADS=4 /usr/bin/time -v ./build/benchmark/feature_tracking_sequence <dir> 600
./build/benchmark/spacing_benchmark                    # the occupancy mask that lost
./build/benchmark/wordtype_narrow                      # 64-bit callers: narrow, do not convert
```

## Logs

The bytes on this page are computed from buffer geometry and do not vary between launches;
the **times** beside them do, so each time here is the median of a launch sweep — thirty on
x86-64, ten on the device:
[feature tracking](logs/feature-tracking-x86_64-launches.log), [single](logs/feature-tracking-x86_64.log), [aarch64](logs/feature-tracking-aarch64-launches.log) ·
[morphology](logs/morphology-x86_64-launches.log), [single](logs/morphology-x86_64.log), [aarch64](logs/morphology-aarch64-launches.log), [single](logs/morphology-aarch64.log) ·
[derivative](logs/derivative-x86_64-launches.log), [single](logs/derivative-x86_64.log), [aarch64](logs/derivative-aarch64-launches.log), [single](logs/derivative-aarch64.log) ·
[denoise](logs/denoise-x86_64-launches.log), [single](logs/denoise-x86_64.log), [aarch64](logs/denoise-aarch64-launches.log), [single](logs/denoise-aarch64.log) ·
[reduce](logs/reduce-x86_64-launches.log), [aarch64](logs/reduce-aarch64-launches.log) — the word-type row's source ·
[goodFeaturesToTrack](logs/goodfeatures-x86_64-launches.log), [first thirty](logs/goodfeatures-x86_64.log), [aarch64](logs/goodfeatures-aarch64-launches.log), [single](logs/goodfeatures-aarch64.log) ·
[pyramid](logs/pyramid-x86_64.log), [aarch64](logs/pyramid-aarch64-launches.log), [single](logs/pyramid-aarch64.log) ·
[word width](logs/wordwidth-x86_64.log), [aarch64](logs/wordwidth-aarch64-launches.log), [single](logs/wordwidth-aarch64.log) ·
[peak RSS against threads](logs/feature-tracking-rss-x86_64.log) ·
[spacing](logs/spacing-x86_64-launches.log), [single](logs/spacing-x86_64.log) ·
[64-bit narrowing](logs/wordtype_narrow-x86_64-launches.log), [single](logs/wordtype_narrow-x86_64.log), [aarch64](logs/wordtype_narrow-aarch64-launches.log), [single](logs/wordtype_narrow-aarch64.log)

[pyramid](logs/pyramid-x86_64.log) and [word width](logs/wordwidth-x86_64.log) are single
launches on x86-64; both publish computed byte counts, exact and identical on both
architectures, and nothing on this page reads an x86-64 time out of either. Logs marked stale
in [expected-stale.txt](logs/expected-stale.txt) were taken before a bit-identical change to
the code beneath them; the file records which files moved and why the figure stands.

## Changes to published figures

| date | row | previous | current | why |
|---|---|---|---|---|
| 2026-09-03 | `cv::morphologyEx` internal allocation | "allocates nothing" (read from a process high-water mark) | 5,784 B of row buffers at 640×480 (allocator probe) | a process RSS cannot resolve 5.8 KB; the row's ratio stays 5.33× (5.38× if counted) |
| 2026-09-21 | 64-bit narrowing, x86-64 times, all three arms | one launch, about 1.2× high | thirty launches | launch noise on the host is one-sided ([methodology-timing.md](methodology-timing.md)); the ratios between the arms did not move |
| 2026-09-21 | Lucas–Kanade per-point cost across a 36-fold frame-size change, device | 5% | 6% | ten launches replaced one ([limits.md](limits.md)) |
