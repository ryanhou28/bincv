# Features and tracking

Corner detection, FAST, descriptors, matching and optical flow, against each one's OpenCV
equivalent. One thread on both sides. Setup and denominator rule: [README.md](README.md).

The results here are the least uniform in these reports: optical flow is seven to eight times
faster, FAST on a wide image is at parity, and corner detection wins by far more on the
deployment target than on the desktop. **x86-64 and aarch64 are separate columns** — different
OpenCV builds on different machines, never averaged.

## Summary

### Speed

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Each pair is one measurement against the other, in the unit the row names, so the smaller
number is the faster side.

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| Lucas–Kanade, `1/2/2/2` | `cv::calcOpticalFlowPyrLK` | 3.871 ms | **0.543 ms** | **7.13×** | 23.476 ms | **2.843 ms** | **8.26×** |
| Lucas–Kanade, `1/1/1/1` | `cv::calcOpticalFlowPyrLK` | 3.871 ms | **0.136 ms** | **28.53×** | 23.476 ms | **0.609 ms** | **38.54×** |
| BRIEF descriptors | `cv::ORB::compute` † | 0.660 ms | **0.141 ms** | 4.69× | 6.816 ms | **0.648 ms** | 10.51× |
| Hamming matching, kNN=2 | `cv::BFMatcher` | 9.184 ms | **1.947 ms** | **4.72×** | 38.269 ms | **19.391 ms** | 1.97× |
| FAST, wide image | `cv::FAST` | 0.363 ms | 0.344 ms | 1.05× | 2.906 ms | 3.024 ms | 0.96× |
| FAST, bit-plane | `cv::FAST` | 266.2 µs | **177.0 µs** | **1.50×** | 2054.5 µs | **865.3 µs** | **2.37×** |
| `goodFeaturesToTrack` | `cv::goodFeaturesToTrack` | 13.67 ns/px | **12.06 ns/px** | 1.13× | 74.99 ns/px | **43.49 ns/px** | **1.72×** |
| `cornerSubPix` | `cv::cornerSubPix` | not published | not published | ~13× | not published | not published | 13.70× |

**`cornerSubPix` is the one row here whose measurements did not survive into the
repository.** The two times were taken and the ratio recorded; the values behind it were
not, so the cells say so rather than being filled in. `corner_subpix_benchmark` exists and is
named below, so the remedy is one run — and until that run happens this row is a claim on
trust where every other row on this page can be checked.

### Memory

Peak working set, computed from buffer geometry, so it is identical on both architectures.
The itemization is in [footprint.md](footprint.md).

<!-- figure-check values="OpenCV|binCV|ratio" source="@docs/reports/footprint.md" -->
| operation | measured against | OpenCV | binCV | ratio |
|---|---|---|---|---|
| FAST input plane | `cv::FAST` on `CV_8U` | 360,960 B | **46,080 B** | 7.83× |
| `goodFeaturesToTrack` | `cv::goodFeaturesToTrack`, binarized | 9,014,976 B | **1,580,064 B** | 5.71× |
| `cornerSubPix` | `cv::cornerSubPix` | — | refines in place, no allocation | — |

**Lucas–Kanade has no footprint row because the tracking benchmark times only.** The 6.23×
memory result is a whole-pipeline figure and belongs to
[feature-tracking.md](feature-tracking.md); it is not a per-call property of
`calcOpticalFlowPyrLK`.

† `cv::ORB::compute` also computes orientation and rotates its pattern per keypoint. It is
not a like-for-like comparison and is printed for scale rather than claimed.

The `goodFeaturesToTrack` rows were recorded before the response sweep was optimized and
carried a note saying so. They have been re-measured on the shipped kernel and the note is
gone; [what the old rows were measuring](#corner-detection) says what moved.
`cornerSubPix` refines the same seeds
from its already-computed ternary derivatives against `cv::cornerSubPix` on the 8-bit image,
each side's own natural input.

## Optical flow

The single largest component of a feature tracking pipeline, and binCV's strongest result. 140
points · 31×31 window · four levels · 20 iterations maximum · synthetic content · one thread
on each side.

| arm | x86-64 (ms) | x86-64 ratio | aarch64 (ms) | aarch64 ratio |
|---|---|---|---|---|
| `cv::calcOpticalFlowPyrLK` on the same bits as `CV_8U` | 3.871 | — | 23.476 | — |
| **binCV, `1/2/2/2` ladder (shipped)** | **0.543** | **7.13×** | **2.843** | **8.26×** |
| binCV, `1/1/1/1` ladder | 0.136 | **28.53×** | 0.609 | **38.54×** |

Both trackers stop early on their own convergence rules here, so they do not run the same
number of iterations — the realistic comparison, but it leaves iteration count as a confound.
Forcing both to twenty iterations (`BINCV_FORCE_ITERS=1`) gives 0.881 ms against OpenCV's
8.403 for the shipped ladder and 0.194 against the same 8.403 for `1/1/1/1` — 9.54× and
43.28×. The free-running numbers are the conservative ones and are what is quoted.

Most of the advantage is in setup rather than in the iteration: OpenCV copies a 961-pixel
window times three shorts, per point, per level, into its own buffers before it iterates.
binCV reads the bit-planes in place.

**The ladder is the dominant cost on binCV's side**: `1/2/2/2` costs 4.00× on x86 and 4.67×
on the device over `1/1/1/1`, because the tracker pays roughly `20N²` population counts per
window row at every level regardless of how small that level is. `1/1/1/1` is faster and
less accurate; the shipped ladder is the operating point that keeps keypoint yield up.

**This is Lucas–Kanade against Lucas–Kanade.** Wired into a whole pipeline the end-to-end
figure is 3.30×, and the gap between the two is the honest part of the result — the stages
around tracking do not have this ratio. See [feature-tracking.md](feature-tracking.md).

## Descriptors and matching

752×480 · 256-bit descriptors · 1000 keypoints · OpenCV pinned to one thread.

| arm | OpenCV, x86-64 (ms) | binCV, x86-64 (ms) | x86-64 ratio | OpenCV, aarch64 (ms) | binCV, aarch64 (ms) | aarch64 ratio |
|---|---|---|---|---|---|---|
| describe, against `cv::ORB` † | 0.660 | **0.141** | 4.69× | 6.816 | **0.648** | 10.51× |
| match, kNN=2 over 1000×1000, against `cv::BFMatcher` | 9.184 | **1.947** | **4.72×** | 38.269 | **19.391** | 1.97× |

Matching suits binCV's thesis most directly — a Hamming distance over 256-bit descriptors is
four population counts — and it is **the result that transfers worst to the deployment
target**. x86 has a scalar `POPCNT` instruction; aarch64's `CNT` is a vector instruction whose
result must then be reduced across lanes. The same property that makes binCV's reductions
bulk-only halves this ratio on the machine binCV is aimed at, and it is worth knowing before
designing around the desktop number.

## FAST

Two entry points, and they give different answers. 752×480 for the wide-image row, the
tracking pipeline's own frame for the others.

| input | corners | `cv::FAST`, x86-64 | binCV, x86-64 | x86-64 ratio | `cv::FAST`, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| `CV_8U`, wide image | 4144 | 0.363 ms | 0.344 ms | 1.05× | 2.906 ms | 3.024 ms | 0.96× |
| `CV_8U`, the pipeline's own frame | 6724 | 266.2 µs | 262.5 µs | 1.01× | 2054.5 µs | 2051.0 µs | 1.00× |
| **bit-plane**, same frame | 6724 | 266.2 µs | **177.0 µs** | **1.50×** | 2054.5 µs | **865.3 µs** | **2.37×** |

**Parity on the wide-image entry point is the honest outcome and it ships that way.**
`cv::FAST` is a mature vectorised kernel, and a caller who is holding bytes should not be
told to pack them first — for that caller the answer is that binCV matches OpenCV and costs
nothing to adopt.

The bit-plane overload is the interesting one. A caller who already has a binary image gets
1.50× on x86 and **2.37× on the device** on an input of 46,080 bytes against `cv::FAST`'s
360,960, bit-exact corner-for-corner with `cv::FAST` in scan order. It is one of the few
results *better* on the deployment target, and the reason is register pressure: the arc test
needs sixteen live vectors, and aarch64 has thirty-two vector registers where x86 has sixteen,
so the AVX2 form spends part of its win on spill traffic. The earlier conclusion that FAST
could not benefit from packing was true of the signature it had been given, not of the
operation.

Scoring is a substantial part of the cost. The bit-plane path chooses per chunk between a
per-corner transpose and arc masks; sweeping that threshold on x86-64 moves the whole operation
between 171.9 µs and 189.9 µs against `cv::FAST`'s 266.2 — 1.55× at the fast end, 1.40× at the
slow one, and 177.0 µs or 1.50× at the shipped adaptive setting. binCV's score is a different
quantity from OpenCV's — the longest qualifying arc rather than the largest surviving threshold
— which is why this is Tier 2.

## Corner detection

**1.72× on the reference device and 1.13× on the desktop, while holding a fifth of the
memory on both.**

Both spellings, at 640×480, returning the same corners and timed in the same interleaved run.
Time is nanoseconds per pixel and working set is bytes per pixel, so the smaller number is
the better one. Both ratio columns are against the binarized denominator in the first row —
including on the last row, where both sides are OpenCV:

| variant | x86-64 (ns/px) | x86-64, denominator ÷ this arm | aarch64 (ns/px) | aarch64, denominator ÷ this arm | working set (B/px) |
|---|---|---|---|---|---|
| OpenCV, binarized (the denominator) | 13.67 | — | 74.99 | — | 36.94 |
| binCV, frame map | 12.03 | **1.14×** | 43.70 | **1.72×** | 16.54 |
| **binCV, streaming ring (shipped)** | **12.06** | **1.13×** | **43.49** | **1.72×** | **12.56** |
| `cv::goodFeaturesToTrack` (stock, different numerics) | 8.14 | **1.68×** | 58.31 | **1.29×** | 29.00 |

Every cell is a median of whole process launches — seven on the device, thirty on x86-64 —
rather than one run; why the two counts differ is two paragraphs down.

Agreement is exact against OpenCV: 723 corners against 723, every position matching, worst
displacement 0.00 px. The two binCV spellings agree corner for corner, which the benchmark
asserts before it times anything.

**The device numbers are the trustworthy ones, and on x86 a single launch settles nothing.**
On the device the within-run spread is 0.05–1.11% and seven launches put the ratio between
1.714× and 1.728×. On the shared x86 box the same thirty launches scatter the *ratio* from
0.95× to 1.60× — wider than either arm alone, so interleaving the two arms does not cancel
it. What survives there is the median: 1.13×, with a bootstrap 95% confidence interval of
1.11–1.15× over the thirty. Quote that interval or nothing; a lone run on this host can
return almost anything.

**The margin is smaller on the desktop because the desktop's denominator is the faster one.**
binCV's code is identical in both columns. Stock `cv::goodFeaturesToTrack` runs at 1.68× the
binarized pipeline on x86-64 and 1.29× on the device, so OpenCV's x86 build is the one
getting more out of its machine — a property of the two OpenCV builds, not of binCV.

### This row has been corrected twice

**The first version published 0.53× on both architectures, and that was wrong**: it timed the
frame-map spelling while that spelling was still on an older response kernel than the
streaming form every pipeline calls. Sharing one kernel took the frame map from 26.06 to
about 14.9 ns/pixel, and the row became 0.92× on x86 and 1.45× on the device.

**Those figures were stale when they were published, and this is the second correction.** They
were recorded at commit `05ce58f`, before the response sweep's tail was rewritten to run
eight pixels at a time; the page carried a note calling them conservative and the note stood.
Rebuilt and run today in the same harness, that commit's kernel measures 51.93 ns/pixel on
the device and puts the ratio back at 1.45× — the published number, reproduced — against the
shipped kernel's 43.49 and 1.72×. The denominator moved 0.8%, so the whole of the difference
is binCV's arm. **The split the previous version of this page called genuine was not one.**

On x86-64 the same rebuild separates two causes the old row had fused. That commit's kernel,
run here today against the shipped one in twelve alternating pairs, measures **1.04×** — not
the 0.92× that was published. The distance from 0.92× to 1.04× is three launches on a host
whose ratio scatters by half; the distance from 1.04× to 1.13× is the kernel, which came out
1.21× faster in eleven of those twelve pairs. On the device there is nothing to separate: the
old kernel reproduces its published 1.45×, and every bit of the move is the kernel.

The two kernels return the same answer, which is what makes that a speed comparison at all:
the response map is bit-identical and the corner list matches in position, order and value
over the benchmark's four frames, on both architectures. The footprint was never in question
— 28 bytes per pixel against binCV's 5.14 at the measured survivor count. Detection is
11.5–14.6% of the assembled pipeline at [that benchmark's](feature-tracking.md) duty cycle,
so the end-to-end cost either way is small.

## Reproduce

```bash
./build/benchmark/lk_headtohead                  # optical flow, LK against LK
BINCV_FORCE_ITERS=1 ./build/benchmark/lk_headtohead
./build/benchmark/feature_benchmark              # FAST, BRIEF, matching
./build/benchmark/fast_bitplane_benchmark        # FAST on a bit-plane
./build/benchmark/corner_opencv_benchmark        # goodFeaturesToTrack
./build/benchmark/corner_subpix_benchmark        # cornerSubPix; the ratio-only row above
```

All five are self-contained. Logs:
[optical flow](logs/lk_headtohead-x86_64.log), [aarch64](logs/lk_headtohead-aarch64.log) ·
[features](logs/features-x86_64.log), [aarch64](logs/features-aarch64.log) ·
[bit-plane FAST](logs/fast_bitplane-x86_64.log), [aarch64](logs/fast_bitplane-aarch64.log) ·
[goodFeaturesToTrack](logs/goodfeatures-x86_64.log), [aarch64](logs/goodfeatures-aarch64.log)
