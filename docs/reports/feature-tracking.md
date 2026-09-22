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

## Speed

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Milliseconds per frame over the whole sequence, so the smaller number is the faster
pipeline.

|  | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|
| the assembled pipeline, ms/frame | 3.841–4.485 | **1.134–1.283** | **3.30×** | 23.249–23.451 | **4.906–4.949** | **4.73×** |

Each ratio is the conservative one of its repeats: five x86 runs span 3.30× to 3.50×, three
device runs 4.73× to 4.74×. Almost all of the x86 spread is OpenCV's — binCV's own time
moves 1.134 to 1.283 ms while OpenCV's moves 3.841 to 4.485. The device's 0.2% spread
against the desktop's 4% is what a pinned, governor-locked machine buys.

**The reference device is where binCV does better**, and it is the deployment-class part.

Re-verified on aarch64 after the corner-sweep and census optimizations landed: binCV 6.17
ms/frame against OpenCV's 29.3 at 120 frames — **4.76×**, unchanged within spread. (Absolute
ms/frame differ from the table because frame count and warm-up differ; the ratio is the
claim.) The detect stage itself runs 12% faster (0.336 → 0.295 ms/frame), but at this
sequence's 1.7% re-detection duty cycle that amortizes to under 1% of the whole pipeline — the
duty-cycle dependence issue #7 records.

## Memory

Peak working set in bytes, computed from buffer geometry, so it is exact and **identical on
both architectures**.

|  | OpenCV | binCV | ratio |
|---|---|---|---|
| peak working set, bytes | 2,719,832 | **436,704** | **6.23× smaller** |

Itemized in [footprint.md](footprint.md).

## Where the time goes

binCV against itself at the duty cycle the benchmark runs (82 re-detections in 1709 frames,
4.8%), so there is no OpenCV column and no ratio — the shares are of each machine's own
total:

| stage | time, x86-64 (ms/frame) | share of the x86-64 pipeline | time, aarch64 (ms/frame) | share of the aarch64 pipeline |
|---|---|---|---|---|
| track (Lucas–Kanade) | 0.799 | 62.3% | 3.307 | 66.8% |
| build (pyramid + derivatives) | 0.297 | 23.2% | 1.072 | 21.7% |
| — sensor stage | 0.118 | 9.2% | 0.543 | 11.0% |
| — `pyrDown` | 0.137 | 10.6% | 0.377 | 7.6% |
| — derivatives | 0.043 | 3.3% | 0.149 | 3.0% |
| detect | 0.186 | 14.5% | 0.570 | 11.5% |

Tracking dominates on both, so the operations that move this number are the ones inside the
Lucas–Kanade loop rather than the ones with the largest per-operation ratios: `pyrDown` is
10.6% of the x86 pipeline, and an infinite speedup on it would be worth about 1.12×. The two
architectures spend their time within five points of each other on every stage, so nothing
here is bottlenecked on anything architecture-specific.

## Accuracy

The claim is that the tracking is *equivalent*, not identical — the numerics differ, so this
is a Tier 2 comparison. How far binCV's flow vectors sit from OpenCV's on the same frames
(one distribution, not two sides, so the smaller number is the closer agreement):

| | x86-64 | aarch64 |
|---|---|---|
| flow difference, median (px) | **0.0437** | **0.0434** |
| flow difference, p90 (px) | 0.1614 | 0.1614 |
| flow difference, p99 (px) | 22.49 | 22.49 |
| flow difference, max (px) | 213.8 | 213.8 |
| flow vectors agreeing within 1 px | **95.6%** | **95.4%** |

What each tracker did with those vectors:

| | binCV, x86-64 | OpenCV, x86-64 | binCV, aarch64 | OpenCV, aarch64 |
|---|---|---|---|---|
| median track lifetime, frames | 11 | 12 | 11 | 12 |
| per-frame survival | 96.4% | 96.6% | 96.4% | 96.6% |
| tracks observed | 10,279 | 10,108 | 10,279 | 10,129 |

**Parity is not claimed.** The median track lives one frame less than OpenCV's and per-frame
survival is 0.2 points behind, on both architectures.

The p99 of 22.5 px is a real tail: ninety-six percent of flow vectors agree to within a
pixel and a small
number diverge completely, which is what track divergence looks like as a percentile. The
RMS over all comparisons is 7.03 px and is in the log; on a distribution with this shape the
percentiles are the honest summary.

## What this does not claim

**It is not a trajectory-accuracy result.** Geometry and estimation are outside this
library. The agreement figures are evidence that the kernels are sufficient, not a claim
about pose error.

**The detection duty cycle belongs to the benchmark.** This harness re-detects only when it
runs out of tracks — 4.8% of frames here, so detection is 11.5–14.5% of the total. A
tracker that tops up whenever its track count falls below a target detects far more often,
and the detect stage then dominates in a way none of these numbers show.

**It is one thread on each side, and that is binCV's best case.** binCV is serial unless a
caller installs a threading backend; OpenCV is not. Both scale, OpenCV scales better, so the
lead narrows (x86-64, unpinned — a threading arm cannot be measured under `taskset`):

| threads, each side | OpenCV, ms/frame | binCV, ms/frame | ratio |
|---|---|---|---|
| 1 | 3.944 | **1.172** | **3.36×** |
| 2 | 2.832 | **0.942** | **3.01×** |
| 4 | 2.407 | **0.940** | **2.56×** |

binCV barely improves past two threads because only tracking splits over keypoints; the
sensor stage, pyramid build and derivatives stay serial. The split that does exist costs no
additional memory — the only per-thread state is stack. Quoting a threaded binCV against a
single-threaded OpenCV would roughly double these ratios and would be measuring the thread
count.

**It is one sequence.** `V1_02_medium` is the harder of the two EuRoC sequences measured
here. On the easier `MH_01_easy` the whole-pipeline ratio comes out on the other side of the
comparison — a difference larger than most of the effects these reports measure. A
pipeline figure quoted without its sequence is not a figure.

## Addendum, 2026-09-06: one pyramid build per frame

The redundant pyramid rebuild was removed after this report's numbers were taken: the
pipeline now swaps the previous frame's pyramid in and builds only the incoming one, **proven
bit-identical to a rebuild** (`BINCV_PYR_CHECK=1`; 0 of 12,920,040 words differ over the full
sequence on both architectures). The removed `hold` buffer also drops a full binary frame the
footprint table never counted. OpenCV's `calcOpticalFlowPyrLK` still rebuilds both of its
pyramids per call; removing the redundancy on binCV's side only is a recorded owner decision,
and the benchmark's output says so beside the ratio.

One governor-locked device run after the change: build 1.072 → 0.836 ms/frame (the predicted
pyrDown halving), detect 0.570 (identical to the table), track 3.787 against the table's
3.307 — 14% above the recorded runs **for reasons not established** (the device's
soft-temperature-limit flag had tripped at some point in the session). The structural change
and the build-stage saving are what this addendum records; the table stands until a proper
re-measurement replaces it whole rather than row by row.

## Reproduce

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/benchmark/feature_tracking_sequence <euroc-V1_02-cam0-dir>

# equal thread counts on both sides
BINCV_LK_THREADS=4 BINCV_OPENCV_THREADS=4 ./build/benchmark/feature_tracking_sequence <dir>
```

The frame directory is any set of `.png` files in name order. The benchmark prints a warning
block if OpenCV is left at a thread count other than one, because that is the single easiest
way to produce a wrong ratio here.

Logs: [x86-64](logs/feature-tracking-x86_64.log) ·
[x86 repeats](logs/feature-tracking-repeats-x86_64.log) ·
[aarch64](logs/feature-tracking-aarch64.log) ·
[device repeats](logs/feature-tracking-repeats-aarch64.log) ·
[threading](logs/feature-tracking-threads-x86_64.log)
