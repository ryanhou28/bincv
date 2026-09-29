# Features and tracking

Corner detection, FAST, descriptors, matching and optical flow, against each one's OpenCV
equivalent. One thread on both sides. Setup and denominator rule: [README.md](README.md).

The results here are the least uniform in these reports: optical flow is seven to eight times
faster, FAST on 8-bit input is at parity, and corner detection wins by far more on the
deployment target than on the desktop. **x86-64 and aarch64 are separate columns** — different
OpenCV builds on different machines, never averaged.

**Every x86-64 figure on this page is the median of thirty pinned launches**, with the
bootstrap 95% interval those thirty put around each ratio; **every aarch64 figure is the
median of ten** with the same interval —
[methodology-timing.md](methodology-timing.md#the-protocol-each-host-needs) says why ten
there and thirty here. Except where a table says otherwise, the ratio is formed inside each
launch and is the median of those per-launch ratios, so it is not the quotient of the two
cells beside it. Each table names the commit its logs stamp and the commit on `main` that
carries that code.

## Summary

### Speed

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Each pair is one measurement against the other, in the unit the row names, so the smaller
number is the faster side. Taken at `05ab53c` on x86-64 and `80ff0a8` on aarch64 (both on
`main` as `086428c`), except the bit-plane FAST row, taken at `550d45a` (on `main` as
`cb995e7`), and the `goodFeaturesToTrack` row, taken at `6d74d57` (on `main` as `8729e05`).

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| **Lucas–Kanade, `1/2/2/2` (shipped)** | `cv::calcOpticalFlowPyrLK` | 3.978 ms | 0.5585 ms | 7.19× [6.89, 7.40] | 23.40 ms | 2.838 ms | 8.23× [8.19, 8.28] |
| Lucas–Kanade, `1/1/1/1` | `cv::calcOpticalFlowPyrLK` | 3.978 ms | 0.1380 ms | 29.2× [26.8, 29.9] | 23.40 ms | 0.6030 ms | 38.8× [38.5, 39.1] |
| BRIEF descriptors | `cv::ORB::compute` † | 0.6388 ms | 0.1231 ms | 5.18× [5.15, 5.22] | 7.167 ms | 0.6579 ms | 10.8× [10.6, 11.2] |
| Hamming matching, kNN=2 | `cv::BFMatcher` | 9.071 ms | 1.916 ms | 4.70× [4.65, 4.79] | 38.19 ms | 19.52 ms | 1.95× [1.94, 1.97] |
| FAST, 8-bit, synthetic 752×480 (4144 corners) | `cv::FAST` | 0.3591 ms | 0.3446 ms | 1.04× [1.03, 1.05] | 2.910 ms | 3.025 ms | 0.962× [0.961, 0.963] |
| FAST, bit-plane, EuRoC frame (6724 corners) | `cv::FAST` | 267.7 µs | 161.7 µs | 1.65× [1.64, 1.66] | 2051 µs | 865.2 µs | 2.37× [2.37, 2.37] |
| `goodFeaturesToTrack` ‡ | `cv::goodFeaturesToTrack` (stock) | 8.807 ns/px | 6.368 ns/px | 1.38× [1.35, 1.43] | 58.34 ns/px | 24.10 ns/px | 2.42× [2.41, 2.42] |
| `cornerSubPix` | `cv::cornerSubPix` | — | — | — (no sweep) | 8.595 ms | 0.6250 ms | 13.8× [13.7, 13.8] |

† `cv::ORB::compute` also computes orientation and rotates its pattern per keypoint. It is
not a like-for-like comparison and is printed for scale rather than claimed.

‡ The `goodFeaturesToTrack` ratio and interval are the ratio of the two medians, bootstrapped
over the same launches; [Corner detection](#corner-detection) prints the table they come from.

`1/1/1/1` resolves only 1.09× on the x86-64 host, so its interval is wide and that row
should not be read past three digits. `cornerSubPix` has no x86-64 launch sweep, so its
x86-64 cells are empty rather than estimated.

### Memory

Peak working set, computed from buffer geometry, so it is identical on both architectures.
The itemization is in [footprint.md](footprint.md).

<!-- figure-check values="OpenCV|binCV|ratio" source="@docs/reports/footprint.md" -->
| operation | measured against | OpenCV | binCV | ratio |
|---|---|---|---|---|
| FAST input plane | `cv::FAST` on `CV_8U` | 360,960 B | 46,080 B | 7.83× |
| `goodFeaturesToTrack`, at the measured survivor count | `cv::goodFeaturesToTrack`, binarized (read off the objects) | 9,014,976 B | 1,580,064 B | 5.71× |
| `goodFeaturesToTrack`, worst-case provisioned | stock `cv::goodFeaturesToTrack` (accounted, not read) | 29.00 B/px | 12.56 B/px | 2.31× |
| `cornerSubPix` | `cv::cornerSubPix` | — | refines in place, no allocation | — |

**Lucas–Kanade has no footprint row because the tracking benchmark times only.** The 6.23×
memory result is a whole-pipeline figure and belongs to
[feature-tracking.md](feature-tracking.md); it is not a per-call property of
`calcOpticalFlowPyrLK`.

The `goodFeaturesToTrack` speed row is measured against **stock `cv::goodFeaturesToTrack`**;
[Corner detection](#corner-detection) prints the hand-written binarized pipeline beside it
and says why stock is the denominator. The footprint's byte-count row leads with the
binarized pipeline because that side's buffers are read off the objects, where stock's can
only be accounted; the second row is the accounted comparison. `cornerSubPix` refines the
same seeds from its already-computed ternary derivatives against `cv::cornerSubPix` on the
8-bit image, each side's own natural input.

## Optical flow

The single largest component of a feature tracking pipeline, and binCV's strongest result.
640×480 frames · 140 points · 31×31 window · four levels · 20 iterations maximum · synthetic
content · one thread on each side. Taken at `05ab53c` on x86-64 and `80ff0a8` on aarch64
(both on `main` as `086428c`).

| arm | x86-64 (ms) | x86-64 ratio | aarch64 (ms) | aarch64 ratio |
|---|---|---|---|---|
| `cv::calcOpticalFlowPyrLK` on the same bits as `CV_8U` | 3.978 | — | 23.40 | — |
| **binCV, `1/2/2/2` ladder (shipped)** | 0.5585 | 7.19× [6.89, 7.40] | 2.838 | 8.23× [8.19, 8.28] |
| binCV, `1/1/1/1` ladder | 0.1380 | 29.2× [26.8, 29.9] | 0.6030 | 38.8× [38.5, 39.1] |

**This is the widest x86-64 interval in these reports.** Thirty launches resolve 1.04× on
the shipped ladder and only 1.09× on `1/1/1/1`: this host cannot tell 29× from 31×, so that
cell is quoted to three significant figures and no more. Both arms swing together — the
OpenCV arm's own launches run 3.724 to 8.215 ms — which is why the ratio survives what the
individual times do not.

Both trackers stop early on their own convergence rules here, so they do not run the same
number of iterations — the realistic comparison, but it leaves iteration count as a confound.
Forcing both to twenty iterations (`BINCV_FORCE_ITERS=1`) gives 0.881 ms against OpenCV's
8.403 for the shipped ladder and 0.194 against the same 8.403 for `1/1/1/1` — 9.54× and
43.3×. Those three figures are one launch
([log](logs/lk_headtohead-x86_64.log), taken at `25065d7`) and carry the uncertainty a
single launch on this host has; the free-running numbers are the conservative ones and are
what is quoted.

Most of the advantage is in setup rather than in the iteration: OpenCV copies a 961-pixel
window times three shorts, per point, per level, into its own buffers before it iterates.
binCV reads the bit-planes in place.

**The ladder is the dominant cost on binCV's side**: `1/2/2/2` costs 4.10× [3.89, 4.20] on
x86-64 and 4.71× [4.67, 4.76] on the device over `1/1/1/1` (binCV ÷ binCV, paired per
launch), because the tracker pays roughly `20N²` population counts per window row at every
level regardless of how small that level is. `1/1/1/1` is faster and less accurate; the
shipped ladder is the operating point that keeps keypoint yield up.

**This is Lucas–Kanade against Lucas–Kanade.** Wired into a whole pipeline the end-to-end
figure is 3.97× on x86-64 and 5.32× on aarch64
([feature-tracking.md](feature-tracking.md#speed)); the stages around tracking do not have
this ratio.

## Descriptors and matching

752×480 · 256-bit descriptors · 1000 keypoints · OpenCV pinned to one thread. Taken at
`05ab53c` on x86-64 and `80ff0a8` on aarch64 (both on `main` as `086428c`).

| arm | OpenCV, x86-64 (ms) | binCV, x86-64 (ms) | x86-64 ratio | OpenCV, aarch64 (ms) | binCV, aarch64 (ms) | aarch64 ratio |
|---|---|---|---|---|---|---|
| describe, against `cv::ORB` † | 0.6388 | 0.1231 | 5.18× [5.15, 5.22] | 7.167 | 0.6579 | 10.8× [10.6, 11.2] |
| match, kNN=2 over 1000×1000, against `cv::BFMatcher` | 9.071 | 1.916 | 4.70× [4.65, 4.79] | 38.19 | 19.52 | 1.95× [1.94, 1.97] |

† `cv::ORB::compute` also computes orientation and rotates its pattern per keypoint; the
row is printed for scale rather than claimed.

Matching suits binCV's thesis most directly — a Hamming distance over 256-bit descriptors is
four population counts — and it is **the result that transfers worst to the deployment
target**. x86 has a scalar `POPCNT` instruction; aarch64's `CNT` is a vector instruction whose
result must then be reduced across lanes. The same property that makes binCV's reductions
bulk-only halves this ratio on the machine binCV is aimed at, and it is worth knowing before
designing around the desktop number.

## FAST

Two entry points, and they give different answers. All three rows are 752×480: the first on
synthetic content, the other two on the tracking pipeline's own EuRoC frame. The synthetic
row is taken at `05ab53c` on x86-64 and `80ff0a8` on aarch64 (both on `main` as `086428c`);
the two EuRoC rows at `550d45a` (on `main` as `cb995e7`).

| input | corners | `cv::FAST`, x86-64 | binCV, x86-64 | x86-64 ratio | `cv::FAST`, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| 8-bit, synthetic (`CV_8U` in) | 4144 | 0.3591 ms | 0.3446 ms | 1.04× [1.03, 1.05] | 2.910 ms | 3.025 ms | 0.962× [0.961, 0.963] |
| 8-bit, EuRoC frame (`CV_8U` in) | 6724 | 267.7 µs | 262.0 µs | 1.02× [1.01, 1.02] | 2051 µs | 2052 µs | 0.998× [0.997, 1.00] |
| **bit-plane, same EuRoC frame** | 6724 | 267.7 µs | 161.7 µs | 1.65× [1.64, 1.66] | 2051 µs | 865.2 µs | 2.37× [2.37, 2.37] |

**Parity on the 8-bit entry point is the outcome, and it ships that way.** `cv::FAST` is a
mature vectorised kernel, and a caller who is holding bytes should not be told to pack them
first — for that caller the answer is that binCV matches OpenCV and costs nothing to adopt.

The bit-plane overload is the interesting one. A caller who already has a binary image gets
1.65× on x86-64 and 2.37× on the device on an input of 46,080 bytes against `cv::FAST`'s
360,960, bit-exact corner-for-corner with `cv::FAST` in scan order. It is one of the few
results *better* on the deployment target, and the reason is register pressure: the arc test
needs sixteen live vectors, and aarch64 has thirty-two vector registers where x86 has sixteen,
so the AVX2 form spends part of its win on spill traffic.

The vector arm is switchable off at runtime, and the same sweep shows it on: with the switch
off the call reads 605.4 µs on x86-64, 3.73× [3.71, 3.77] the vector arm's time (binCV ÷
binCV, paired per launch), and 2921 µs on the device.

Scoring is a substantial part of the cost. The bit-plane path chooses per chunk between a
per-corner transpose and arc masks; sweeping that threshold on x86-64 moves the whole operation
between 160.8 µs and 202.9 µs against `cv::FAST`'s 267.7 — 1.66× at the fast end, 1.32×
with the masks switched off entirely, and 161.7 µs or 1.65× at the shipped adaptive
setting. On aarch64 the sweep is flat at about 863.5 µs, because the scored arm measured as
a loss on NEON and is compiled out there. binCV's score is a different quantity from
OpenCV's — the longest qualifying arc rather than the largest surviving threshold — which is
why this is Tier 2.

## Corner detection

**1.38× on the desktop and 2.42× on the reference device against stock
`cv::goodFeaturesToTrack`, at 12.56 bytes per pixel against its 29.00.**

Both spellings, at 640×480, returning the same corners and timed in the same interleaved run.
Time is nanoseconds per pixel and working set is bytes per pixel, so the smaller number is
the better one. Taken at `6d74d57` (on `main` as `8729e05`); thirty launches on x86-64, ten
on the device. **In this table each ratio is the ratio of the two medians**, with a
bootstrap interval over the same launches.

| variant | x86-64 (ns/px) | x86-64 vs stock | aarch64 (ns/px) | aarch64 vs stock | working set (B/px) |
|---|---|---|---|---|---|
| `cv::goodFeaturesToTrack` (stock, the denominator) | 8.807 | — | 58.34 | — | 29.00 † |
| OpenCV, binarized (the correctness reference) | 14.45 | 0.609× [0.598, 0.632] | 75.53 | 0.772× [0.770, 0.776] | 36.94 |
| binCV, frame map | 7.490 | 1.18× [1.16, 1.21] | 25.36 | 2.30× [2.29, 2.30] | 16.54 |
| **binCV, streaming ring (shipped)** | 6.368 | 1.38× [1.35, 1.43] | 24.10 | 2.42× [2.41, 2.42] | 12.56 |
| binCV, streaming, vector arms off | 7.596 | 1.16× [1.14, 1.20] | 24.08 | 2.42× [2.41, 2.43] | 12.56 |

† Stock's working set is **accounted, not read** — `cornerMinEigenVal` materializes `Dx`,
`Dy` and a `CV_32FC3` covariance inside the call and `gftt` adds `eig` and a dilate
destination, none of which a caller can measure from outside. Every other figure in that
column is read off the objects.

### The denominator

Stock `cv::goodFeaturesToTrack` is the call a caller makes, so it is the denominator: the
baseline for a new implementation is the best existing option, not a worse one. The
hand-written binarized pipeline (`openCvBinarized()` in
`benchmark/corner_opencv_benchmark.cpp` — about ten OpenCV calls reproducing binCV's exact
semantics) is printed beside it because it is the **correctness reference**, and the only arm
that can be one: binCV and it agree on 723 corners of 723 with a worst displacement of
0.00 px over the benchmark's four frames, and their response maps are bit-identical over
360,960 pixels on the real frame. Stock cannot prove that, because it runs a 3×3 Sobel over
the byte image and finds different corners by construction — which is what Tier 2 means
here.

A speed ratio against an arm whose yield nobody recorded is speed at unknown accuracy, so the
benchmark records stock's: over the same four frames binCV returns **723 corners to stock's
686**, and 540 of binCV's sit within 3 px of a stock one. The two arms do comparable amounts
of work under identical parameters. They are not finding the same corners, and this page does
not claim they are.

### Where the time goes

The response sweep — the stage the packed representation accelerates — is the *minority* of
`goodFeaturesToTrack`'s runtime; selection is the rest. An instruction-count profile of one
frame (deterministic, where this host's timing is not) puts the spacing filter at 18.5% of a
detection, the out-of-range guard in the box-word lookup at 9.3%, the running maximum at
6.8%, and the rank at 7% of instructions but 27% of time — what a comparison sort over a pool
of ties costs in branch mispredicts. Two of those shape the code:

* **The running maximum does not vectorize as written over floats.** `max` over floats is
  not reassociable — NaN and −0.0 make the result depend on the order the lanes combine in —
  so GCC and Clang both leave it scalar whatever the flags. A response is never negative and
  never NaN, so reducing the IEEE **bit patterns** is the same answer and does vectorize.
* **The rank need not compare two corners at all.** The suppression sweep appends candidates
  in raster order and the threshold pass only compacts, so *reversed*, the pool is already in
  `CornerStronger`'s tie order — y descending, then x descending. The rank is therefore a
  stable sort on the response alone: counting passes over the byte lanes of its bit pattern,
  no comparator, nothing to mispredict — 4.9× on that stage. It is the host form of what
  `backends/cuda/src/corner.cu` does on the device, where count, prefix-sum and emit replaced
  a comparison sort at 4×.

`benchmark/detect_stage_profile` is built on the library's own functions rather than on a
copy of the kernel's prefix, so each arm it times is a prefix of the shipped call.

### Output and footprint

Peak working set is 12.56 B/px for the shipped form: the spacing filter reorders the accepted
set it already had, and the rank's counting passes scatter into the caller's own array past
the candidates — slack the capacity contract already sizes for the worst case. Where that
slack is absent, or the top-K heap has reordered the pool, `std::sort` runs instead, and
`Corner.RankArmsAgree` holds the two arms to the same answer. Output is what makes this a
speed comparison: 723 corners of 723 against the binarized pipeline at 0.00 px, frame map and
streaming identical corner for corner, asserted before anything is timed.

### Reading the two hosts

**The device numbers are the trustworthy ones, and on x86-64 a single launch settles
nothing.** On the device the within-run spread is 0.16–1.09% and ten launches put the ratio
between 2.41× and 2.42×. The x86-64 host scatters 12–30% across thirty launches of the *same
binary*; what survives there is the median and its interval, 1.38× [1.35, 1.43], with 30 of
30 launches above 1.00×. A second, independent thirty-launch sweep on that host, an hour
apart, read 1.37×, inside that interval. Quote the interval or nothing.

**The margin is wider on the device because the desktop's denominator is the faster one.**
binCV's code is identical in both columns. Stock `cv::goodFeaturesToTrack` runs at 1.64× the
binarized pipeline on x86-64 and 1.29× on the device, so OpenCV's x86-64 build is the one
getting more out of its machine — a property of the two OpenCV builds, not of binCV.

**The vector arms are provably running.** The `vector arms off` row is the same call with
`impl::cornerSimdEnabled()` false: on x86-64 it is 1.19× [1.17, 1.21] slower (binCV ÷
binCV), and on aarch64 it is 0.999× [0.999, 1.00] — the arm is x86-only by measurement, so
the architecture its own gate excludes reports 1.00×, which is the check that the fast path
is the one being timed. A ratio near 1.00× on x86-64 would mean it was not.

## Reproduce

```bash
./build/benchmark/lk_headtohead                  # optical flow, LK against LK
BINCV_FORCE_ITERS=1 ./build/benchmark/lk_headtohead
./build/benchmark/feature_benchmark              # FAST, BRIEF, matching
./build/benchmark/fast_bitplane_benchmark        # FAST on a bit-plane
./build/benchmark/corner_opencv_benchmark        # goodFeaturesToTrack
./build/benchmark/corner_subpix_benchmark        # cornerSubPix
```

All five binaries are self-contained. Each column above is
`./scripts/run_launches.sh -n <N> ./build/benchmark/<bench>` read back through
`./scripts/aggregate_launches.py` — thirty launches on x86-64, ten on the device. Logs — the
sweep behind each column, and the single launches also committed:
[optical flow](logs/lk_headtohead-x86_64-launches.log), [single](logs/lk_headtohead-x86_64.log), [aarch64](logs/lk_headtohead-aarch64-launches.log), [single](logs/lk_headtohead-aarch64.log) ·
[features](logs/features-x86_64-launches.log), [single](logs/features-x86_64.log), [aarch64](logs/feature-aarch64-launches.log), [single](logs/features-aarch64.log) ·
[bit-plane FAST](logs/fast_bitplane-x86_64-launches.log), [single](logs/fast_bitplane-x86_64.log), [aarch64](logs/fast_bitplane-aarch64-launches.log), [single](logs/fast_bitplane-aarch64.log) ·
[goodFeaturesToTrack](logs/goodfeatures-x86_64-launches.log), [earlier sweep](logs/goodfeatures-x86_64.log), [aarch64](logs/goodfeatures-aarch64-launches.log), [earlier sweep](logs/goodfeatures-aarch64.log) ·
[cornerSubPix](logs/corner_subpix-aarch64-launches.log) — aarch64 only.
Logs marked stale in [expected-stale.txt](logs/expected-stale.txt) were taken before a
bit-identical change to the code beneath them; the file records which files moved and why
the figure stands.

## Changes to published figures

| date | row | previous | current | why |
|---|---|---|---|---|
| before 2026-09-21 | `goodFeaturesToTrack` against the binarized pipeline (denominator since replaced) | 0.53× on both architectures | 0.92× x86-64, 1.45× aarch64; later 1.73× aarch64 | the frame-map spelling was moved onto the streaming kernel; then a re-take after the response sweep's tail was rewritten to run eight pixels at a time |
| 2026-09-21 | every x86-64 cell | one launch | median of thirty pinned launches, with interval | protocol change; the ratios reproduced, the times moved |
| 2026-09-21 | BRIEF, x86-64 | 4.69× | 5.18× | the single launch was a slow draw; a refactor suspected as the cause was A/B'd at thirty launches each and is not one (122,596 ns against 123,834, intervals overlapping) |
| 2026-09-21 | `cornerSubPix`, aarch64 | 13.70×, ratio only | 13.8× [13.7, 13.8], 8.595 against 0.6250 ms | ten device launches committed; the x86-64 half has no sweep |
| 2026-09-22 | `goodFeaturesToTrack`, denominator | the hand-written binarized pipeline (headline 1.132× on x86-64) | stock `cv::goodFeaturesToTrack` | the baseline is the call a caller makes; against it the kernel of the time read 0.737× [0.721, 0.754] on x86-64 and 1.368× [1.364, 1.382] on the device |
| 2026-09-22 | `goodFeaturesToTrack`, both columns | 0.737× x86-64, 1.368× aarch64 | 1.38× [1.35, 1.43], 2.42× [2.41, 2.42] | selection stage optimized (bit-pattern running maximum, counting-sort rank, spacing filter): binCV 11.310 → 6.368 ns/px on x86-64 and 43.337 → 24.10 on the device, denominators unchanged within their intervals |
| 2026-09-23 | FAST bit-plane, x86-64 | 1.472× (180.85 µs) | 1.65× (161.7 µs) | the vector arm's runtime switch was read once per image row, keeping the scalar body live; read once per call |
