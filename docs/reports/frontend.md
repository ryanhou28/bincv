# The frontend, end to end

**Over 1709 consecutive frame pairs of EuRoC `V1_02_medium`, a binCV tracking frontend runs
3.30× faster than the OpenCV equivalent on x86-64 and 4.73× faster on the reference device,
holding 6.23× less memory on both, with flow that agrees with OpenCV's to 0.0437 px at the
median.**

This is the result the library exists to produce. Everything in
[primitives.md](primitives.md) is a component of it.

## What is compared

Two complete, independent frontends over the same 8-bit grayscale frames:

```
sensor stage  →  pyramid  →  derivatives  →  detect  →  track  →  lifecycle
median +          4 levels    both axes      good-      pyramidal  cull, re-detect
edge threshold                               Features   Lucas–     on the same
                                             ToTrack    Kanade     schedule
```

The binCV side runs all six stages in binCV. The OpenCV side runs all six in OpenCV —
`cv::filter2D` for the sensor stage, `cv::buildOpticalFlowPyramid`,
`cv::goodFeaturesToTrack`, `cv::calcOpticalFlowPyrLK`. Neither side is handed the other's
intermediate results.

**Each side builds its own binary frame, and the two are bit-identical** — 0 pixels differ
over 1709 frames, checked every frame. That control matters more than it looks: it is what
makes this a comparison of the frontends rather than of two different inputs.

Each side also detects and re-detects on its own schedule, so a divergence in tracking shows
up as a divergence in detection load rather than being hidden by a shared trigger.

## Setup

752×480 · four pyramid levels on a `1/2/2/2` bit ladder · 31×31 tracking window · 20
iterations maximum · `uint32_t` words · **one thread on each side** · Release.

binCV's vector paths are live on both architectures and OpenCV's are too — AVX2 against
AVX2 on x86, NEON against NEON on the device. The run prints both.

## Speed and memory

| | binCV | OpenCV | ratio |
|---|---|---|---|
| **x86-64**, ms/frame | 1.134–1.283 | 3.841–4.485 | **3.30×** (conservative of five runs) |
| **aarch64**, ms/frame | 4.906–4.949 | 23.249–23.451 | **4.73×** (conservative of three runs) |
| **peak working set**, bytes | 436,704 | 2,719,832 | **6.23× smaller** |

The footprint figure is computed from buffer geometry and is identical on both
architectures. The speed figures are not comparable across the two rows: they are different
OpenCV builds on different machines, and the reference device is the one that carries a
deployment claim.

Five x86 runs span 3.30× to 3.50×, and almost all of that spread is OpenCV's: binCV's own
time moves 1.134 to 1.283 ms while OpenCV's moves 3.841 to 4.485. Three device runs span
4.73× to 4.74× — a 0.2% spread, against the x86 box's 4% — which is what a pinned,
governor-locked, single-purpose machine buys over a desktop under a hypervisor. The
conservative figure is quoted on both.

**The reference device is where binCV does better, and that is the point.** It is the
deployment-class part, and the gap is larger there than on the desktop.

## Where the time goes

binCV's own stages, at the duty cycle the benchmark actually runs (82 re-detections in 1709
frames, 4.8%):

| stage | x86 ms/frame | share | aarch64 ms/frame | share |
|---|---|---|---|---|
| track (Lucas–Kanade) | 0.799 | 62.3% | 3.307 | 66.8% |
| build (pyramid + derivatives) | 0.297 | 23.2% | 1.072 | 21.7% |
| — sensor stage | 0.118 | 9.2% | 0.543 | 11.0% |
| — `pyrDown` | 0.137 | 10.6% | 0.377 | 7.6% |
| — derivatives | 0.043 | 3.3% | 0.149 | 3.0% |
| detect | 0.186 | 14.5% | 0.570 | 11.5% |

Tracking dominates on both, which is why the operations that matter most to this number are
the ones inside the Lucas–Kanade loop rather than the ones with the largest per-operation
ratios. `pyrDown` is 10.6% of the x86 frontend: an infinite speedup on it would be worth
about 1.12×.

The two architectures spend their time in nearly the same proportions — within five points on
every stage, and within three on all but tracking — even though the device is roughly four
times slower in absolute terms. The frontend is not bottlenecked on anything
architecture-specific.

## Accuracy

The claim is that the tracking is *equivalent*, not that it is identical — the numerics
differ, so this is a Tier 2 comparison.

| | x86-64 | aarch64 |
|---|---|---|
| flow difference, median | **0.0437 px** | **0.0434 px** |
| p90 / p99 / max | 0.1614 / 22.49 / 213.8 px | 0.1614 / 22.49 / 213.8 px |
| agreeing within 1 px | **95.6%** | **95.4%** |
| median track lifetime, binCV vs OpenCV | 11 vs 12 frames | 11 vs 12 frames |
| per-frame survival, binCV vs OpenCV | 96.4% vs 96.6% | 96.4% vs 96.6% |
| tracks observed, binCV vs OpenCV | 10,279 vs 10,108 | 10,279 vs 10,129 |

The two architectures agree to within 0.2 points on every accuracy figure, which is the
expected result — the kernels are bit-exact across them, and the small residual differences
come from the number of flow vectors each run had to compare.

**Parity is not claimed.** The median track lives one frame less than OpenCV's and per-frame
survival is 0.2 points behind, on both architectures.

The p99 of 22.5 px is a real tail, not a measurement artifact. Ninety-six percent of flow
vectors agree to within a pixel and a small number diverge completely, which is what track
divergence looks like when it is summarised as a percentile. The RMS over all comparisons is
7.03 px and is reported in the log for completeness; on a distribution with this shape the
percentiles are the honest summary and the mean is not.

## What this does not claim

**It is not a trajectory-accuracy result.** Geometry and estimation are outside this
library, so what a full VIO system does with these features is a property of the
integration. The agreement figures above are evidence that the kernels are sufficient, not a
claim about pose error.

**The detection duty cycle belongs to the benchmark, not to binCV.** This harness re-detects
only when it runs out of tracks, which on this sequence is 4.8% of frames, so detection is
11.5–14.5% of the total. A frontend that tops up its track set whenever it falls below a
target detects far more often, and the detect stage then dominates in a way none of these
numbers show. The re-detection policy is the application's choice and it moves the total
more than most of the kernel differences in these reports do.

**It is one thread on each side, and that is binCV's best case.** binCV is serial unless a
caller installs a threading backend; OpenCV is not. Both scale, and OpenCV scales better, so
the lead narrows as threads are added (x86-64, unpinned — a threading arm cannot be measured
under `taskset`):

| threads, each side | binCV ms/frame | OpenCV ms/frame | ratio |
|---|---|---|---|
| 1 | 1.172 | 3.944 | **3.36×** |
| 2 | 0.942 | 2.832 | **3.01×** |
| 4 | 0.940 | 2.407 | **2.56×** |

binCV barely improves past two threads because only tracking splits over keypoints; the
sensor stage, pyramid build and derivatives stay serial and are an increasing share of what
is left. Quoting a threaded binCV against a single-threaded OpenCV would roughly double
these ratios and would be measuring the thread count.

The split that does exist is safe by construction — each keypoint writes only its own
outputs — and costs no additional memory, since the only per-thread state is stack.

**It is one sequence.** `V1_02_medium` is the harder of the two EuRoC sequences this project
has measured. On the easier `MH_01_easy` the tracker has materially less work to do per
frame, and the whole-frontend ratio comes out on the other side of the comparison — a
difference larger than most of the effects these reports measure. Only `V1_02` numbers appear
here, and a frontend figure quoted without its sequence is not a figure.

## Reproduce

```bash
cmake -S bincv-cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/benchmark/frontend_sequence <euroc-V1_02-cam0-dir>

# equal thread counts on both sides
BINCV_LK_THREADS=4 BINCV_OPENCV_THREADS=4 ./build/benchmark/frontend_sequence <dir>
```

The frame directory is any set of `.png` files in name order; the numbers above are the full
`V1_02_medium` `cam0` stream. The benchmark prints a warning block if OpenCV is left at a
thread count other than one, because that is the single easiest way to produce a wrong ratio
here.

Logs: [x86-64](logs/frontend-x86_64.log), [x86 repeats](logs/frontend-repeats-x86_64.log) ·
[aarch64](logs/frontend-aarch64.log), [device repeats](logs/frontend-repeats-aarch64.log) ·
[threading](logs/frontend-threads-x86_64.log)
