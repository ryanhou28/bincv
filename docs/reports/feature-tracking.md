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
| the assembled pipeline, ms/frame | 4.0070 | **1.0095** | **3.969× [3.935, 4.004]** | 23.614 | **4.435** | **5.324× [5.307, 5.335]** |

Both ratios are formed inside each launch and are medians of those per-launch ratios, above
1.00 in 30 of 30 on x86-64 and 10 of 10 on the device; each interval is a percentile bootstrap
over its own launches. Re-taken at commit `880704b`. Almost all of the x86 scatter is
OpenCV's — its arm's launches run 3.924 to 4.446 ms while binCV's run 0.975 to 1.084. The
device's 0.5% ratio scatter against the desktop's 13% is what a pinned, governor-locked
machine buys, and it is why ten launches are quoted there and thirty here.

**The device row moved 4.620× → 5.324×, and only part of that is this round's work.** The
stage table below splits it, and the split matters more than the headline:

* **`detect` went 0.457 → 0.298 ms/frame**, which is `goodFeaturesToTrack`'s selection
  optimization ([features.md](features.md#corner-detection)) arriving at the pipeline's
  4.8% re-detection duty cycle. On its own that stage change takes the row to about
  **4.78×**.
* **`track` went 3.787 → 3.279 ms/frame, and nothing here changed Lucas–Kanade.** That is
  the rest of the move, and this page does not have a cause for it.

**The previously published Lucas–Kanade regression does not reproduce.** This page recorded
3.307 → 3.787 ms/frame as "a regression rather than noise" and said the pipeline got slower
because of it. At `880704b` that stage reads **3.279** — back where it started, across ten
launches whose own scatter is 1.8%. Three measurements of a stage nobody edited reading
3.307, 3.787 and 3.279 is not a regression that was fixed; it is a stage whose
run-to-run behaviour across *sweeps* is wider than any one sweep's interval suggests, and the
+14.5% should not have been called a regression on one re-take. Recorded rather than
explained: naming a cause this page has not measured is the failure it already has a name for
two sections down.

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
| track (Lucas–Kanade) | 0.7365 | 73.0% | 3.279 | 73.9% |
| build (pyramid + derivatives) | 0.1995 | 19.8% | 0.8525 | 19.2% |
| — sensor stage | 0.098 | 9.7% | 0.505 | 11.4% |
| — `pyrDown` | 0.059 | 5.8% | 0.190 | 4.3% |
| — derivatives | 0.042 | 4.2% | 0.1535 | 3.5% |
| **detect** | **0.073** | **7.2%** | **0.298** | **6.7%** |

Tracking dominates on both, so the operations that move this number are the ones inside the
Lucas–Kanade loop rather than the ones with the largest per-operation ratios: `pyrDown` is
5.6% of the x86 pipeline and 3.7% of the device's, and an infinite speedup on it would now be
worth about 1.06× on either. Both columns now carry the one-build-per-frame change, so the
two `pyrDown` shares finally describe the same code; on every other stage the two
architectures spend their time within five points of each other, so nothing here is
bottlenecked on anything architecture-specific.

**What the device column has done over three sweeps**, the same table at three commits:

| stage | `25065d7` | `80ff0a8` | `880704b` | |
|---|---|---|---|---|
| — `pyrDown` | 0.377 | **0.189** | 0.190 | halved at `80ff0a8`, flat since |
| build | 1.072 | 0.860 | 0.8525 | |
| **detect** | 0.570 | 0.457 | **0.298** | **−34.8% this round** |
| — sensor stage | 0.543 | 0.519 | 0.505 | |
| **track (Lucas–Kanade)** | **3.307** | **3.787** | **3.279** | **unedited, and it has moved ±14%** |
| **the pipeline** | **4.906–4.949** | **5.097** | **4.435** | |

**`detect` is the one this round moved, and it is the only one this round touched.**
`goodFeaturesToTrack`'s selection is 1.8× faster ([features.md](features.md#corner-detection))
and this is that arriving at a 4.8% re-detection duty cycle. Detection is now 6.7% of the
device pipeline and 7.2% of the desktop's, so there is little left in it: another halving
would be worth about 1.04× end to end.

**Lucas–Kanade is the row to be careful about.** Nothing has edited it across these three
sweeps and it reads 3.307, 3.787, 3.279. `lk_headtohead` was flat across the first two —
2.843 ms to 2.838, and OpenCV's arm 23.476 to 23.400 — which was read at the time as
evidence that the +14.5% lived in "what the pipeline hands the tracker". The third sweep
undoes it without anything being fixed, so the simpler reading is that this stage's
sweep-to-sweep spread is wider than any one sweep's 1.8% interval, and that it should not
have been called a regression. The workload is identical across all three: 1710 frames, 1709
pairs, 82 re-detections, the same track lifetimes, and a peak of 436,704 B exactly as
published.

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
runs out of tracks — 4.8% of frames here, so detection is 6.7–7.2% of the total. A
tracker that tops up whenever its track count falls below a target detects far more often,
and the detect stage then dominates in a way none of these numbers show. That also cuts the
other way: `goodFeaturesToTrack` getting 1.8× faster is worth 4% of *this* pipeline and would
be worth much more of that one.

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
([logs/feature-tracking-x86_64-launches.log](logs/feature-tracking-x86_64-launches.log))
read OpenCV 3.7275 ms/frame, binCV 1.0215, **3.658×** [3.633, 3.681], above 1.00 in 30 of 30,
against the 3.30× the pre-change table carried. (Those are that round's figures; the tables
above are re-taken at `880704b`, where the same sweep reads 3.969×.)
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
