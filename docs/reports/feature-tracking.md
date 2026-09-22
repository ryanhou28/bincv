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

**Both columns are now of the same code.** The x86-64 figures are thirty pinned launches and
the aarch64 figures five, both taken at commit `80ff0a8`, after the [2026-09-06 change to one
pyramid build per frame](#addendum-2026-09-06-one-pyramid-build-per-frame). The aarch64
column used to predate that change and to be a single launch; re-taking it is what found the
regression the next section describes.

## Speed

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Milliseconds per frame over the whole sequence, so the smaller number is the faster
pipeline.

|  | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|
| the assembled pipeline, ms/frame | 3.7275 | **1.0215** | **3.658× [3.633, 3.681]** | 23.588 | **5.097** | **4.620× [4.596, 4.628]** |

Both ratios are formed inside each launch and are medians of those per-launch ratios, above
1.00 in 30 of 30 on x86-64 and 5 of 5 on the device; each interval is a percentile bootstrap
over its own launches. Almost all of the x86 scatter is OpenCV's — its arm's launches run
3.622 to 4.919 ms while binCV's run 0.991 to 1.217. The device's 0.6% ratio scatter against
the desktop's 13% is what a pinned, governor-locked machine buys, and it is why five launches
are quoted there and thirty here.

**The aarch64 row was 4.73× and it is 4.620×, and that is a regression rather than noise.**
It is the one published figure the device re-measurement moved beyond its own band, and the
stage table below says where: **every stage but one got faster, `pyrDown` by half, and the
pipeline still got slower** — because Lucas–Kanade is three quarters of it and grew 14.5%.
An independent five-launch sweep taken separately reads 4.616× [4.611, 4.637], so the two
sweeps agree and the published 4.73× is outside both.

**An independent sweep agrees.** The `BINCV_LK_BATCH=1` arm in
[limits.md](limits.md#the-vector-arms-and-proving-they-are-on) is the same pipeline in a
separate thirty-launch sweep and reads 3.632× [3.614, 3.648] — 0.7% away, intervals
overlapping.

**The reference device is where binCV does better**, and it is the deployment-class part.

It was re-verified on aarch64 once before, after the corner-sweep and census optimizations
landed, at 120 frames rather than 1709: binCV 6.17 ms/frame against OpenCV's 29.3 —
**4.76×**, read as unchanged. **The full-sequence re-take above supersedes that**, and the
disagreement is the useful part: a 120-frame check with its own warm-up was not a fine enough
instrument to see a 2.3% move in the ratio, and it was quoted as if it were. The detect-stage
figure from that check stands on its own terms — 12% faster, 0.336 to 0.295 ms/frame — and at
this sequence's 1.7% re-detection duty cycle it amortizes to under 1% of the whole pipeline,
the duty-cycle dependence issue #7 records.

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
| track (Lucas–Kanade) | 0.704 | 68.9% | 3.787 | 74.3% |
| build (pyramid + derivatives) | 0.181 | 17.7% | 0.860 | 16.9% |
| — sensor stage | 0.086 | 8.4% | 0.519 | 10.2% |
| — `pyrDown` | 0.057 | 5.6% | 0.189 | 3.7% |
| — derivatives | 0.038 | 3.7% | 0.150 | 2.9% |
| detect | 0.139 | 13.6% | 0.457 | 9.0% |

Tracking dominates on both, so the operations that move this number are the ones inside the
Lucas–Kanade loop rather than the ones with the largest per-operation ratios: `pyrDown` is
5.6% of the x86 pipeline and 3.7% of the device's, and an infinite speedup on it would now be
worth about 1.06× on either. Both columns now carry the one-build-per-frame change, so the
two `pyrDown` shares finally describe the same code; on every other stage the two
architectures spend their time within five points of each other, so nothing here is
bottlenecked on anything architecture-specific.

**What the device column moved, stage by stage**, against the same table at commit `25065d7`:

| stage | published | re-taken at `80ff0a8` | |
|---|---|---|---|
| — `pyrDown` | 0.377 | **0.189** | −49.9% |
| build | 1.072 | 0.860 | −19.8% |
| detect | 0.570 | **0.457** | −19.8% |
| — sensor stage | 0.543 | 0.519 | −4.4% |
| **track (Lucas–Kanade)** | **3.307** | **3.787** | **+14.5%** |
| **the pipeline** | **4.906–4.949** | **5.097** | **+3.0 to +3.9%** |

**Lucas–Kanade's own kernel is not what regressed.** `lk_headtohead` on the same commit is
flat — 2.843 ms to 2.838, and OpenCV's arm 23.476 to 23.400 — so the 14.5% is in what the
pipeline hands the tracker rather than in the tracker. `pyrDown` halving over the same
interval is the obvious place to look and the pyramid it produces is what LK reads, but
nothing here measures that link, and naming a cause this page has not measured is how a
figure like 4.73× survived three weeks in the first place. The workload is identical on both
sides of the comparison: 1710 frames, 1709 pairs, 82 re-detections, the same track lifetimes,
and a peak of 436,704 B exactly as published.

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
runs out of tracks — 4.8% of frames here, so detection is 11.5–13.6% of the total. A
tracker that tops up whenever its track count falls below a target detects far more often,
and the detect stage then dominates in a way none of these numbers show.

**It is one thread on each side, and that is binCV's best case.** binCV is serial unless a
caller installs a threading backend; OpenCV is not. Both scale, OpenCV scales better, so the
lead narrows (x86-64, unpinned — a threading arm cannot be measured under `taskset`, so these
three rows are **single runs without an interval**, which every other x86-64 figure on this
page now has):

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
soft-temperature-limit flag had tripped at some point in the session). That run is not enough
to replace the aarch64 column, which is why that column is still the pre-change one.

**The x86-64 half of the re-measurement now exists, and the tables above carry it.** Thirty
pinned launches
([logs/feature-tracking-x86_64-launches.log](logs/feature-tracking-x86_64-launches.log)):
OpenCV 3.7275 ms/frame, binCV 1.0215, **3.658×** with a bootstrap 95% interval of
[3.633, 3.681] and above 1.00 in 30 of 30, against the 3.30× the pre-change table carried.
The stage shares moved the way the change predicts — `pyrDown` 0.137 → 0.057 ms/frame, build
0.297 → 0.181, sensor 0.118 → 0.086, track 0.799 → 0.704, detect 0.186 → 0.139. **This is the
row-by-row edit the paragraph above declined, and it is being made deliberately**: leaving a
figure that the code no longer produces, in order to keep two columns of the same vintage,
trades a true number for a tidy table. The columns say which commit each belongs to instead.
The device half is still owed.

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

Logs: [x86-64, thirty launches](logs/feature-tracking-x86_64-launches.log) ·
[x86-64, the single run it replaced](logs/feature-tracking-x86_64.log) ·
[x86 repeats](logs/feature-tracking-repeats-x86_64.log) ·
[aarch64, five launches](logs/feature-tracking-aarch64-launches.log) ·
[aarch64, the independent sweep that confirms them](logs/feature-tracking-spotcheck-aarch64-launches.log) ·
[aarch64, the single run they replaced](logs/feature-tracking-aarch64.log) ·
[device repeats](logs/feature-tracking-repeats-aarch64.log) ·
[threading](logs/feature-tracking-threads-x86_64.log)
