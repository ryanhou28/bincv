# Features and tracking

Corner detection, FAST, descriptors, matching and optical flow, against the OpenCV call each
replaces. One thread on both sides. Setup and denominator rule: [README.md](README.md).

These are the operations closest to what a tracking frontend actually spends its time on,
and they are also where the results are least uniform. Optical flow is seven to eight times
faster, FAST on a wide image is at parity, and corner detection is slightly behind on the
desktop and ahead on the deployment target — while holding a fifth of the memory on both.

## Summary

| operation | OpenCV call | x86-64 | aarch64 | footprint |
|---|---|---|---|---|
| Lucas–Kanade, `1/2/2/2` | `cv::calcOpticalFlowPyrLK` | **7.13×** | **8.26×** | — ‡ |
| Lucas–Kanade, `1/1/1/1` | `cv::calcOpticalFlowPyrLK` | **28.53×** | **38.54×** | — ‡ |
| BRIEF descriptors | `cv::ORB::compute` † | 4.69× | 10.51× | — |
| Hamming matching, kNN=2 | `cv::BFMatcher` | **4.72×** | 1.97× | — |
| FAST, wide image | `cv::FAST` | 1.05× | 0.96× | — |
| FAST, bit-plane | `cv::FAST` | **1.50×** | **2.37×** | 7.8× smaller input |
| `goodFeaturesToTrack` | `cv::goodFeaturesToTrack` | 0.92× | **1.45×** | 5.71× smaller |

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
is 3.30×, and the gap between the two is the honest part of the result — the stages around
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

**Slightly behind on x86 at 0.92×, and 1.45× ahead on the reference device — while holding a
fifth of the memory on both.**

Both spellings, at 640×480. The two return the same corners and are timed in the same
interleaved run:

| variant | x86 ns/pixel | vs OpenCV | aarch64 ns/pixel | vs OpenCV | bytes/pixel |
|---|---|---|---|---|---|
| binCV, frame map | 14.72–15.96 | 0.85–0.97× | 51.64–51.67 | **1.45–1.47×** | 16.54 |
| binCV, streaming ring | 14.46–15.01 | 0.92–0.98× | 51.25–51.31 | **1.46–1.48×** | **12.56** |
| OpenCV, binarized (the denominator) | 13.63–14.24 | 1.00× | 75.02–75.82 | 1.00× | 36.94 |
| `cv::goodFeaturesToTrack` (stock, different numerics) | 8.53 | 1.60× | 59.34–59.44 | 1.26–1.28× | 29.00 |

Agreement is exact against OpenCV: 723 corners against 723, every position matching, worst
displacement 0.00 px. And the two binCV spellings agree with each other corner for corner,
which the benchmark now asserts before it times anything.

**The device numbers are the trustworthy ones here.** Their spreads are 0.08% to 1.62%
against 5% to 63% on the shared x86 box, which is the difference between a pinned,
governor-locked, single-purpose machine and a desktop under a hypervisor carrying other
work.

**An earlier version of this report published 0.53× here, and that was wrong.** The number
was real but it measured a path the library does not ship. `goodFeaturesToTrack` and
`goodFeaturesToTrackStreaming` returned identical corners by contract but did not share a
response kernel: the streaming form used bit-sliced 3×3 box sums, and the frame-map form had
been left on the older per-position sweep when that kernel was written. Every frontend here
calls the streaming form; the benchmark called the other one. They now share one kernel, and
the frame-map form went from 26.06 to about 14.9 ns/pixel — the whole of the difference
between "twice as slow" and "a little behind".

**The correction also reversed the conclusion, not just the number.** This operation used to
be the report's clearest example of "memory wins, speed loses" — the trade where binCV
declines the seven `float` planes of locality OpenCV's detector materialises, 28 bytes per
pixel against binCV's 5.14 at the measured survivor count, and pays for it in time. The
footprint half is unchanged and still the point. The speed half turned out not to be a trade
at all on the machine binCV is aimed at: same buffers, same corners, 1.45× faster.

What remains true is the caveat this whole report keeps running into. On x86 the same call is
0.92×, because OpenCV's x86 detector is AVX2-dispatched and its aarch64 one is relatively
weaker against the same machine. Detection is 11.5–14.6% of the frontend at this benchmark's
duty cycle, so the end-to-end cost either way is small.

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
