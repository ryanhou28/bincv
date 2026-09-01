# Features and tracking

Corner detection, FAST, descriptors, matching and optical flow, against the OpenCV call each
replaces. One thread on both sides. Setup and denominator rule: [README.md](README.md).

These are the operations closest to what a tracking frontend actually spends its time on,
and they are also where the results are least uniform. Optical flow is seven to eight times
faster, FAST on a wide image is at parity, and corner detection is about twice as slow while
holding a fifth of the memory.

## Summary

| operation | OpenCV call | x86-64 | aarch64 | footprint |
|---|---|---|---|---|
| Lucas–Kanade, `1/2/2/2` | `cv::calcOpticalFlowPyrLK` | **7.13×** | **8.26×** | — ‡ |
| Lucas–Kanade, `1/1/1/1` | `cv::calcOpticalFlowPyrLK` | **28.53×** | **38.54×** | — ‡ |
| BRIEF descriptors | `cv::ORB::compute` † | 4.69× | 10.51× | — |
| Hamming matching, kNN=2 | `cv::BFMatcher` | **4.72×** | 1.97× | — |
| FAST, wide image | `cv::FAST` | 1.05× | 0.96× | — |
| FAST, bit-plane | `cv::FAST` | **1.50×** | **2.37×** | 7.8× smaller input |
| `goodFeaturesToTrack` | `cv::goodFeaturesToTrack` | **0.53×** | **0.53×** | 5.71× smaller |

† `cv::ORB::compute` also computes orientation and rotates its pattern per keypoint. It is
not a like-for-like comparison and is printed for scale rather than claimed.

‡ The tracking benchmark measures time, not footprint. The 6.23× memory result is a
whole-frontend figure and belongs to [frontend.md](frontend.md); it is not a per-call
property of `calcOpticalFlowPyrLK`.

## Optical flow

The single largest component of the frontend, and binCV's strongest result.

| arm | x86-64 | vs OpenCV | aarch64 | vs OpenCV |
|---|---|---|---|---|
| binCV, `1/2/2/2` ladder (shipped) | 0.543 ms | **7.13×** | 2.843 ms | **8.26×** |
| binCV, `1/1/1/1` ladder | 0.136 ms | **28.53×** | 0.609 ms | **38.54×** |
| `cv::calcOpticalFlowPyrLK` on the same bits as `CV_8U` | 3.871 ms | 1.00× | 23.476 ms | 1.00× |

140 points, 31×31 window, four levels, 20 iterations maximum, synthetic content.

Both trackers stop early on their own convergence rules here, so they do not run the same
number of iterations — which is the realistic comparison but leaves iteration count as a
confound. Forcing both to run exactly twenty iterations (`BINCV_FORCE_ITERS=1`) moves the
figures to 9.54× and 43.28×; the free-running numbers are the conservative ones and are what
is quoted.

Most of the advantage is in setup rather than in the iteration. OpenCV copies a 961-pixel
window times three shorts, per point, per level, into its own buffers before it iterates.
binCV reads the bit-planes in place.

**The ladder is the dominant cost on binCV's side**: the `1/2/2/2` bit ladder costs 4.00× on
x86 and 4.67× on the device over `1/1/1/1`, because the tracker pays roughly `20N²`
population counts per window row at every level regardless of how small that level is.
`1/1/1/1` is faster and less accurate; the shipped ladder is the operating point that keeps
keypoint yield up.

**This is Lucas–Kanade against Lucas–Kanade, not the whole frontend.** The end-to-end figure
is 3.36×, and the gap between the two is the honest part of the result — the stages around
tracking do not have this ratio. See [frontend.md](frontend.md).

## Descriptors and matching

752×480, 256-bit descriptors, 1000 keypoints, OpenCV pinned to one thread.

| arm | binCV | OpenCV | x86 ratio | aarch64 ratio |
|---|---|---|---|---|
| describe | 0.141 / 0.648 ms | 0.660 / 6.816 ms (`cv::ORB`) | 4.69× † | 10.51× † |
| match, kNN=2 over 1000×1000 | 1.947 / 19.391 ms | 9.184 / 38.269 ms (`cv::BFMatcher`) | **4.72×** | **1.97×** |

Matching is the operation that suits binCV's thesis most directly — a Hamming distance over
256-bit descriptors is four population counts — and it is **the result that transfers worst
to the deployment target**: 4.72× on x86 and 1.97× on the device. x86 has a scalar `POPCNT`
instruction; aarch64's `CNT` is a vector instruction whose result must then be reduced across
lanes. The same property that makes binCV's reductions bulk-only halves this ratio on the
machine binCV is aimed at, and it is worth knowing before designing around the desktop
number.

## FAST

Two entry points, and they give different answers.

| platform | input | corners | `cv::FAST` | binCV | ratio |
|---|---|---|---|---|---|
| x86-64 | `CV_8U`, wide image | 4144 | 0.363 ms | 0.344 ms | 1.05× |
| aarch64 | `CV_8U`, wide image | 4144 | 2.906 ms | 3.024 ms | 0.96× |
| x86-64 | `CV_8U`, frontend's own frame | 6724 | 266.2 µs | 262.5 µs | 1.01× |
| x86-64 | **bit-plane**, same frame | 6724 | 266.2 µs | **177.0 µs** | **1.50×** |
| aarch64 | `CV_8U`, frontend's own frame | 6724 | 2054.5 µs | 2051.0 µs | 1.00× |
| aarch64 | **bit-plane**, same frame | 6724 | 2054.5 µs | **865.3 µs** | **2.37×** |

**Parity on the wide-image entry point is the honest outcome and it ships that way.**
`cv::FAST` is a mature vectorised kernel, and a caller who is holding bytes should not be
told to pack them first — for that caller the answer is that binCV matches OpenCV and costs
nothing to adopt.

The bit-plane overload is the interesting one. A caller who already has a binary image —
which, in a frontend built on binCV, is everyone — gets 1.50× on x86 and **2.37× on the
device** on an input that is 7.8× smaller, bit-exact corner-for-corner with `cv::FAST` in
scan order. This is one of the few results that is *better* on the deployment target, and
the reason is register pressure: the arc test needs sixteen live vectors, and aarch64 has
thirty-two vector registers where x86 has sixteen, so the AVX2 form spends part of its win
on spill traffic.

The lesson is worth repeating: *an operation that consumes bytes is not thereby a byte
operation*. The earlier conclusion that FAST could not benefit from packing was true only of
the signature it had been given, not of the operation.

Scoring is a substantial part of the operation, and how it is computed matters: the bit-plane
path chooses per chunk between a per-corner transpose and arc masks, and sweeping that
threshold moves the whole operation between 1.40× and 1.55× on x86. The shipped adaptive
setting lands at 1.50×. binCV's score is also a genuinely different quantity from OpenCV's —
the longest qualifying arc rather than the largest surviving threshold — which is why this is
a Tier 2 operation.

## Corner detection

**binCV is about twice as slow here, and roughly five times smaller.**

| variant | x86 ns/pixel | spread | aarch64 ns/pixel | spread | vs OpenCV | bytes/pixel |
|---|---|---|---|---|---|---|
| binCV | 26.060 | 35.7% | 140.457 | 0.08% | **0.53×** both | 16.54 |
| OpenCV, binarized (the denominator) | 13.873 | 16.6% | 74.383 | 0.29% | 1.00× | 36.94 |
| `cv::goodFeaturesToTrack` (stock, different numerics) | 12.977 | 47.6% | 57.799 | 0.48% | 1.07× / 1.29× | 29.00 |

Agreement is exact on both: 723 corners against 723, every position matching, worst
displacement 0.00 px.

The two architectures land on the same 0.53×, which is a stronger statement than either
reading alone — this is a property of the operation rather than of one machine's dispatch.

The trade is deliberate and it is the clearest example of the library's tie-break. OpenCV's
detector materialises seven `float` planes — 28 bytes per pixel — to buy locality that binCV
declines to buy, sweeping a three-row response ring instead. binCV holds 5.14 bytes per
pixel at the measured survivor count against 29.35. The speed loss was accepted for that.

Two qualifications. **The x86 spreads are wide** — 36% on binCV's own row, 48% on OpenCV's —
so on that platform treat this as "roughly half speed" rather than as 0.53. The device rows
have spreads of 0.08% to 0.48%, which is the difference between a quiet pinned single-purpose
machine and a desktop under a hypervisor, and is a good reason to read the aarch64 column as
the precise one throughout these reports. And detection is 11.5–14.5% of the frontend at this
benchmark's duty cycle, so the end-to-end cost of the loss is small.

## Reproduce

```bash
./build/benchmark/lk_headtohead                  # optical flow, LK against LK
BINCV_FORCE_ITERS=1 ./build/benchmark/lk_headtohead
./build/benchmark/feature_benchmark              # FAST, BRIEF, matching
./build/benchmark/fast_bitplane_benchmark        # FAST on a bit-plane
./build/benchmark/corner_opencv_benchmark        # goodFeaturesToTrack
```

All four are self-contained. Logs:
[optical flow](logs/lk_headtohead-x86_64.log), [aarch64](logs/lk_headtohead-aarch64.log) ·
[features](logs/features-x86_64.log), [aarch64](logs/features-aarch64.log) ·
[bit-plane FAST](logs/fast_bitplane-x86_64.log), [aarch64](logs/fast_bitplane-aarch64.log) ·
[goodFeaturesToTrack](logs/goodfeatures-x86_64.log), [aarch64](logs/goodfeatures-aarch64.log)
