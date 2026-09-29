# The assembled feature tracking pipeline

**This page is not a headline result.** binCV is an operation library; what a caller adopts
is a call at a time, and those comparisons are in [primitives.md](primitives.md),
[features.md](features.md) and [stereo.md](stereo.md). This page exists to show that the
per-operation wins survive being **wired together** — measured on a pipeline this project
assembled for that purpose, which is not a standard benchmark and is not anyone else's
pipeline.

**What the pipeline is.** Six stages over 8-bit grayscale frames:

```
sensor stage  →  pyramid  →  derivatives  →  detect  →  track  →  lifecycle
median +          4 levels    both axes      good-      pyramidal  cull, re-detect
edge threshold                               Features   Lucas–     on the same
                                             ToTrack    Kanade     schedule
```

That is an ordinary sparse feature tracker — the front half of a visual odometry system. It
is built twice: once entirely in binCV, once entirely in OpenCV (`cv::filter2D`,
`cv::buildOpticalFlowPyramid`, `cv::goodFeaturesToTrack`, `cv::calcOpticalFlowPyrLK`).
Neither side is handed the other's intermediate results, and each detects and re-detects on
its own schedule.

**Each side builds its own binary frame, and the two are bit-identical** — 0 pixels differ
over 1709 frames, checked every frame. That control is what makes this a comparison of the
two pipelines rather than of two different inputs.

## Setup

EuRoC `V1_02_medium` `cam0`, 1709 consecutive frame pairs · 752×480 · four pyramid levels on
a `1/2/2/2` bit ladder · 31×31 tracking window · 20 iterations maximum · `uint32_t` words ·
**one thread on each side** · Release. Vector paths are live on both sides on both
architectures — AVX2 against AVX2, NEON against NEON — and the run prints both.

**x86-64 and aarch64 are separate columns everywhere on this page.** They are different
measurements against different OpenCV builds on different machines, and are never averaged.

**Both columns are of the same code**, taken at `880704b` (on `main` as `8729e05`): thirty
pinned launches on x86-64 and ten on aarch64 with the governor locked. Both include the
one-pyramid-per-frame asymmetry described [below](#one-pyramid-build-per-frame).

## Speed

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Milliseconds per frame over the whole sequence, so the smaller number is the faster
pipeline.

|  | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|
| the assembled pipeline, ms/frame | 4.007 | 1.010 | 3.97× [3.94, 4.00] | 23.61 | 4.435 | 5.32× [5.31, 5.34] |

Each ratio is the ratio of the two medians, with a percentile bootstrap over the same
launches; the per-launch ratio is above 1.00 in 30 of 30 launches on x86-64 and 10 of 10 on
the device. Almost all of the x86-64 scatter is OpenCV's — its arm's launches run 3.924 to
4.446 ms while binCV's run 0.9750 to 1.084. The device's arms scatter 1.4% across ten
launches against the desktop's 11–13% across thirty, which is what a pinned, governor-locked
machine buys and why ten launches are quoted there and thirty here. The reference device is
where binCV does better, and it is the deployment-class part.

## Memory

Peak working set in bytes, computed from buffer geometry, so it is exact and **identical on
both architectures**.

|  | OpenCV | binCV | ratio |
|---|---|---|---|
| peak working set, bytes | 2,719,832 | 436,704 | 6.23× |

Itemized in [footprint.md](footprint.md).

## Where the time goes

binCV against itself at the duty cycle the benchmark runs (82 re-detections in 1709 frames,
4.8%), so there is no OpenCV column and no ratio — the shares are of each machine's own
total:

| stage | time, x86-64 (ms/frame) | share of the x86-64 pipeline | time, aarch64 (ms/frame) | share of the aarch64 pipeline |
|---|---|---|---|---|
| track (Lucas–Kanade) | 0.7365 | 73.0% | 3.279 | 73.9% |
| build (pyramid + derivatives) | 0.1995 | 19.8% | 0.8525 | 19.2% |
| — sensor stage | 0.098 | 9.7% | 0.505 | 11.4% |
| — `pyrDown` | 0.059 | 5.8% | 0.190 | 4.3% |
| — derivatives | 0.042 | 4.2% | 0.1535 | 3.5% |
| detect | 0.073 | 7.2% | 0.298 | 6.7% |

Tracking dominates on both, so the operations that move this number are the ones inside the
Lucas–Kanade loop rather than the ones with the largest per-operation ratios: `pyrDown` is
5.8% of the x86-64 pipeline and 4.3% of the device's, so an infinite speedup on it would be
worth 1.06× on the desktop and 1.04× on the device. Detection is 7.2% and 6.7%, so another
halving of `goodFeaturesToTrack` would be worth about 1.04× end to end. On every stage the
two architectures spend their time within five points of each other, so nothing here is
bottlenecked on anything architecture-specific.

**The Lucas–Kanade row varies between sweeps more than within one.** Across independent
device sweeps of code that did not change, the track stage has read 3.307 ms/frame (one
launch, `25065d7`), 3.791 (five launches at `80ff0a8`, interval 0.4%,
[log](logs/feature-tracking-spotcheck-aarch64-launches.log)) and 3.279 here (ten launches,
interval 0.2%) — a 1.16× swing between sweeps whose own intervals are under 1%, on an
identical workload (1710 frames, 1709 pairs, 82 re-detections, the same track lifetimes, the
same 436,704 B peak). A difference in that row smaller than that swing should not be read
from two sweeps.

## Accuracy

The claim is that the tracking is *equivalent*, not identical — the numerics differ, so this
is a Tier 2 comparison. How far binCV's flow vectors sit from OpenCV's on the same frames
(one distribution, not two sides, so the smaller number is the closer agreement):

| | x86-64 | aarch64 |
|---|---|---|
| flow difference, median (px) | 0.0437 | 0.0434 |
| flow difference, p90 (px) | 0.1614 | 0.1614 |
| flow difference, p99 (px) | 22.49 | 22.49 |
| flow difference, max (px) | 213.8 | 213.8 |
| flow vectors agreeing within 1 px | 95.6% | 95.4% |

What each tracker did with those vectors:

| | binCV, x86-64 | OpenCV, x86-64 | binCV, aarch64 | OpenCV, aarch64 |
|---|---|---|---|---|
| median track lifetime, frames | 11 | 12 | 11 | 12 |
| per-frame survival | 96.4% | 96.6% | 96.4% | 96.6% |
| tracks observed | 10,279 | 10,108 | 10,279 | 10,129 |

**Parity is not claimed.** The median track lives one frame less than OpenCV's and per-frame
survival is 0.2 points behind, on both architectures. binCV's track count is the same on
both architectures; OpenCV's differs between its two builds.

The p99 of 22.49 px is a real tail: ninety-five to ninety-six percent of flow vectors agree
to within a pixel and a small number diverge completely, which is what track divergence looks
like as a percentile. The RMS over all comparisons is 7.031 px on x86-64 and 7.089 on
aarch64, and is in the logs; on a distribution with this shape the percentiles are the honest
summary.

## What this does not claim

**It is not a trajectory-accuracy result.** Geometry and estimation are outside this
library. The agreement figures are evidence that the kernels are sufficient, not a claim
about pose error.

**The detection duty cycle belongs to the benchmark.** This harness re-detects only when it
runs out of tracks — 82 times in 1709 frames, 4.8% — so detection is 6.7–7.2% of the total.
A tracker that tops up whenever its track count falls below a target detects far more often,
and the detect stage then dominates in a way none of these numbers show. That also cuts the
other way: halving `goodFeaturesToTrack` is worth about 1.04× on *this* pipeline and would
be worth much more on that one.

**It is one thread on each side, and that is binCV's best case.** binCV is serial unless a
caller installs a threading backend; OpenCV is not. Both scale, OpenCV scales better, so the
lead narrows. The rows below are x86-64, unpinned — a threading arm cannot be measured under
`taskset` — and are **single runs without an interval**, taken on 2026-09-01 at `25065d7`
(on `main` as `8780bd1`), before the one-pyramid-per-frame change. Read the trend down the
rows; the 1-thread row is not the headline above, which is a later sweep of later code.

| threads, each side | OpenCV, ms/frame | binCV, ms/frame | ratio |
|---|---|---|---|
| 1 | 3.944 | 1.172 | 3.36× |
| 2 | 2.832 | 0.942 | 3.01× |
| 4 | 2.407 | 0.940 | 2.56× |

binCV barely improves past two threads because only tracking splits over keypoints; the
sensor stage, pyramid build and derivatives stay serial. The split that does exist costs no
additional memory — the only per-thread state is stack. Quoting a threaded binCV against a
single-threaded OpenCV would roughly double these ratios and would be measuring the thread
count.

**It is one sequence.** Nothing on this page was measured on another, and a pipeline figure
quoted without its sequence is not a figure.

## One pyramid build per frame

The pipeline builds one pyramid per frame: the previous frame's is swapped in rather than
rebuilt, and the swap is proven bit-identical to a rebuild (`BINCV_PYR_CHECK=1`: 0 of
12,920,040 words differ over the full sequence on both architectures). OpenCV's
`calcOpticalFlowPyrLK` rebuilds both of its pyramids per call. The asymmetry is deliberate,
the benchmark prints it beside the ratio, and both columns above include it.

## Reproduce

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/benchmark/feature_tracking_sequence <euroc-V1_02-cam0-dir>

# equal thread counts on both sides
BINCV_LK_THREADS=4 BINCV_OPENCV_THREADS=4 ./build/benchmark/feature_tracking_sequence <dir>

# the columns above: thirty launches on x86-64, ten on the device
./scripts/run_launches.sh -n 10 ./build/benchmark/feature_tracking_sequence <dir>
./scripts/aggregate_launches.py feature_tracking_sequence-aarch64-launches.log --ratio "OpenCV :/binCV :"
```

The frame directory is any set of `.png` files in name order. The benchmark prints a warning
block if OpenCV is left at a thread count other than one, because that is the single easiest
way to produce a wrong ratio here.

Logs: [x86-64, thirty launches](logs/feature-tracking-x86_64-launches.log) ·
[aarch64, ten launches](logs/feature-tracking-aarch64-launches.log) ·
[aarch64, five launches at `80ff0a8`](logs/feature-tracking-spotcheck-aarch64-launches.log) ·
[threading](logs/feature-tracking-threads-x86_64.log) ·
single launches at `25065d7` and `83087b0`: [x86-64](logs/feature-tracking-x86_64.log),
[x86-64 repeats](logs/feature-tracking-repeats-x86_64.log),
[aarch64](logs/feature-tracking-aarch64.log),
[aarch64 repeats](logs/feature-tracking-repeats-aarch64.log).
Logs marked stale in [expected-stale.txt](logs/expected-stale.txt) were taken before a
bit-identical change to the code beneath them; the file records which files moved and why
the figure stands.

## Changes to published figures

| date | row | previous | current | why |
|---|---|---|---|---|
| 2026-09-06 | binCV pipeline, both columns | two pyramid builds per frame | one build per frame | the redundant rebuild was removed and the swap proven bit-identical; `pyrDown` fell from 0.137 to 0.057 ms/frame on x86-64 |
| 2026-09-21 | x86-64 column | one launch, 3.30× | thirty pinned launches, 3.658× [3.633, 3.681] at `80ff0a8` | protocol change, and the pyramid change above |
| 2026-09-21 | aarch64 column | one launch, 4.73× | five pinned launches, 4.62× at `80ff0a8` | protocol change; the track stage read 3.791 ms/frame against 3.307 in the single launch, on unchanged code |
| 2026-09-21 | 120-frame device check | 4.76× | superseded by the full-sequence sweeps | a 120-frame check with its own warm-up cannot resolve a 1.02× move |
| 2026-09-22 | both columns | 3.658× x86-64, 4.62× aarch64 | 3.97× [3.94, 4.00], 5.32× [5.31, 5.34] at `880704b` | `goodFeaturesToTrack`'s selection optimization (device detect 0.457 → 0.298 ms/frame); the track stage read 3.279 ms/frame, within 1% of the first sweep |
