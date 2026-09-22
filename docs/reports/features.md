# Features and tracking

Corner detection, FAST, descriptors, matching and optical flow, against each one's OpenCV
equivalent. One thread on both sides. Setup and denominator rule: [README.md](README.md).

The results here are the least uniform in these reports: optical flow is seven to eight times
faster, FAST on a wide image is at parity, and corner detection wins by far more on the
deployment target than on the desktop. **x86-64 and aarch64 are separate columns** — different
OpenCV builds on different machines, never averaged.

**Every x86-64 figure on this page is the median of thirty pinned launches**, with the
bootstrap 95% interval those thirty put around each ratio. The ratio is formed inside each
launch, so it is not the quotient of the two cells beside it. **Every aarch64 figure is the
median of ten** with the same interval —
[methodology-timing.md](methodology-timing.md#the-protocol-each-host-needs) says why ten
there and thirty here.

## Summary

### Speed

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

Each pair is one measurement against the other, in the unit the row names, so the smaller
number is the faster side.

| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio |
|---|---|---|---|---|---|---|---|
| Lucas–Kanade, `1/2/2/2` | `cv::calcOpticalFlowPyrLK` | 3.978 ms | **0.5585 ms** | **7.19× [6.89, 7.40]** | 23.400 ms | **2.838 ms** | **8.227× [8.189, 8.284]** |
| Lucas–Kanade, `1/1/1/1` | `cv::calcOpticalFlowPyrLK` | 3.978 ms | **0.1380 ms** | **29.2× [26.8, 29.9]** | 23.400 ms | **0.603 ms** | **38.81× [38.50, 39.13]** |
| BRIEF descriptors | `cv::ORB::compute` † | 0.639 ms | **0.123 ms** | 5.18× [5.15, 5.22] | 7.167 ms | **0.658 ms** | 10.81× [10.59, 11.17] |
| Hamming matching, kNN=2 | `cv::BFMatcher` | 9.071 ms | **1.916 ms** | **4.70× [4.65, 4.79]** | 38.187 ms | **19.520 ms** | 1.953× [1.944, 1.972] |
| FAST, wide image | `cv::FAST` | 0.359 ms | **0.345 ms** | 1.039× [1.033, 1.048] | 2.910 ms | 3.025 ms | 0.962× [0.961, 0.963] |
| FAST, bit-plane | `cv::FAST` | 265.2 µs | **180.2 µs** | **1.472× [1.470, 1.474]** | 2048.2 µs | **865.8 µs** | **2.365× [2.363, 2.370]** |
| `goodFeaturesToTrack` | `cv::goodFeaturesToTrack` | 13.65 ns/px | **12.09 ns/px** | 1.132× [1.118, 1.145] | 75.29 ns/px | **43.49 ns/px** | **1.731× [1.723, 1.735]** |
| `cornerSubPix` | `cv::cornerSubPix` | not published | not published | ~13× | 8.595 ms | **0.625 ms** | 13.76× [13.70, 13.80] |

**`FAST, bit-plane`'s move from 1.50× to 1.472× is the one on this page that is not noise.**
The runtime switch that makes the vector arm provably off-switchable is read once per image
row; reverting only that read measures 12.8% faster with the two intervals disjoint, and
hoisting it out of the row loop keeps the switch and recovers all of it —
[issue #73](https://github.com/ryanhou28/bincv/issues/73), which would take the row to about
1.66×. Of the other moves here, `BRIEF` gaining 4.69× → 5.18× is the largest, and a
refactor that looked like its cause was A/B'd at thirty launches each and is not
(122,596 ns against 123,834, intervals overlapping) — the old cell was a slow draw.
**`1/1/1/1` resolves only 1.090× on this host**, so its interval is wide and that row should
not be read to three digits. [The index](README.md#on-the-x86-64-host) has the whole comparison.

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
| `cv::calcOpticalFlowPyrLK` on the same bits as `CV_8U` | 3.978 | — | 23.400 | — |
| **binCV, `1/2/2/2` ladder (shipped)** | **0.5585** | **7.19× [6.89, 7.40]** | **2.838** | **8.227× [8.189, 8.284]** |
| binCV, `1/1/1/1` ladder | 0.1380 | **29.2× [26.8, 29.9]** | 0.603 | **38.81× [38.50, 39.13]** |

**This is the widest x86-64 interval in these reports, and the row says so.** Thirty launches
resolve 1.044× on the shipped ladder and only 1.090× on `1/1/1/1`: this host cannot tell
29× from 31×, so that cell is quoted to three significant figures and no more. Both arms swing
together — the OpenCV arm's own launches run 3.724 to 8.215 ms — which is why the ratio
survives what the individual times do not.

Both trackers stop early on their own convergence rules here, so they do not run the same
number of iterations — the realistic comparison, but it leaves iteration count as a confound.
Forcing both to twenty iterations (`BINCV_FORCE_ITERS=1`) gives 0.881 ms against OpenCV's
8.403 for the shipped ladder and 0.194 against the same 8.403 for `1/1/1/1` — 9.54× and
43.28×. **Those three figures are single-launch and were not re-taken**, so they carry the
uncertainty this page's other x86 numbers no longer do. The free-running numbers are the
conservative ones and are what is quoted.

Most of the advantage is in setup rather than in the iteration: OpenCV copies a 961-pixel
window times three shorts, per point, per level, into its own buffers before it iterates.
binCV reads the bit-planes in place.

**The ladder is the dominant cost on binCV's side**: `1/2/2/2` costs 4.10× [3.89, 4.20] on
x86 and 4.67× on the device over `1/1/1/1`, because the tracker pays roughly `20N²`
population counts per window row at every level regardless of how small that level is.
`1/1/1/1` is faster and less accurate; the shipped ladder is the operating point that keeps
keypoint yield up.

**This is Lucas–Kanade against Lucas–Kanade.** Wired into a whole pipeline the end-to-end
figure is 3.66×, and the gap between the two is the honest part of the result — the stages
around tracking do not have this ratio. See [feature-tracking.md](feature-tracking.md).

## Descriptors and matching

752×480 · 256-bit descriptors · 1000 keypoints · OpenCV pinned to one thread.

| arm | OpenCV, x86-64 (ms) | binCV, x86-64 (ms) | x86-64 ratio | OpenCV, aarch64 (ms) | binCV, aarch64 (ms) | aarch64 ratio |
|---|---|---|---|---|---|---|
| describe, against `cv::ORB` † | 0.639 | **0.123** | 5.18× [5.15, 5.22] | 7.167 | **0.658** | 10.81× [10.59, 11.17] |
| match, kNN=2 over 1000×1000, against `cv::BFMatcher` | 9.071 | **1.916** | **4.70× [4.65, 4.79]** | 38.187 | **19.520** | 1.953× [1.944, 1.972] |

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
| `CV_8U`, wide image | 4144 | 0.359 ms | **0.345 ms** | 1.039× [1.033, 1.048] | 2.910 ms | 3.025 ms | 0.962× [0.961, 0.963] |
| `CV_8U`, the pipeline's own frame | 6724 | 265.2 µs | **258.8 µs** | 1.024× [1.022, 1.026] | 2048.2 µs | 2052.6 µs | 0.998× [0.997, 1.000] |
| **bit-plane**, same frame | 6724 | 265.2 µs | **180.2 µs** | **1.472× [1.470, 1.474]** | 2048.2 µs | **865.8 µs** | **2.365× [2.363, 2.370]** |

**Parity on the wide-image entry point is the honest outcome and it ships that way.**
`cv::FAST` is a mature vectorised kernel, and a caller who is holding bytes should not be
told to pack them first — for that caller the answer is that binCV matches OpenCV and costs
nothing to adopt.

The bit-plane overload is the interesting one. A caller who already has a binary image gets
1.47× on x86 and **2.365× on the device** on an input of 46,080 bytes against `cv::FAST`'s
360,960, bit-exact corner-for-corner with `cv::FAST` in scan order. It is one of the few
results *better* on the deployment target, and the reason is register pressure: the arc test
needs sixteen live vectors, and aarch64 has thirty-two vector registers where x86 has sixteen,
so the AVX2 form spends part of its win on spill traffic. The earlier conclusion that FAST
could not benefit from packing was true of the signature it had been given, not of the
operation.

Scoring is a substantial part of the cost. The bit-plane path chooses per chunk between a
per-corner transpose and arc masks; sweeping that threshold on x86-64 moves the whole operation
between 178.4 µs and 202.4 µs against `cv::FAST`'s 265.2 — 1.49× at the fast end, 1.31× with
the masks switched off entirely, and 180.2 µs or 1.472× at the shipped adaptive setting. The
whole sweep sits about 4% above where the single-launch log had it, for the reason
[issue #73](https://github.com/ryanhou28/bincv/issues/73) names. binCV's score is a different
quantity from OpenCV's — the longest qualifying arc rather than the largest surviving threshold
— which is why this is Tier 2.

## Corner detection

**1.73× on the reference device and 1.13× on the desktop, while holding a fifth of the
memory on both.**

Both spellings, at 640×480, returning the same corners and timed in the same interleaved run.
Time is nanoseconds per pixel and working set is bytes per pixel, so the smaller number is
the better one. Both ratio columns are against the binarized denominator in the first row —
including on the last row, where both sides are OpenCV:

| variant | x86-64 (ns/px) | x86-64, denominator ÷ this arm | aarch64 (ns/px) | aarch64, denominator ÷ this arm | working set (B/px) |
|---|---|---|---|---|---|
| OpenCV, binarized (the denominator) | 13.65 | — | 75.29 | — | 36.94 |
| binCV, frame map | 12.05 | **1.148× [1.124, 1.166]** | 43.69 | **1.723× [1.716, 1.727]** | 16.54 |
| **binCV, streaming ring (shipped)** | **12.09** | **1.132× [1.118, 1.145]** | **43.49** | **1.731× [1.723, 1.735]** | **12.56** |
| `cv::goodFeaturesToTrack` (stock, different numerics) | 8.139 | **1.675× [1.651, 1.705]** | 58.41 | **1.290× [1.278, 1.294]** | 29.00 |

Every cell is a median of whole process launches — ten on the device, **sixty** on x86-64,
two independent thirties pooled — rather than one run; why the two counts differ is two
paragraphs down.

Agreement is exact against OpenCV: 723 corners against 723, every position matching, worst
displacement 0.00 px. The two binCV spellings agree corner for corner, which the benchmark
asserts before it times anything.

**The device numbers are the trustworthy ones, and on x86 a single launch settles nothing.**
On the device the within-run spread is 0.05–1.11% and ten launches put the ratio between
1.723× and 1.735×. On the shared x86 box sixty launches scatter the *ratio* from 0.86× to
1.61× — wider than either arm alone, so interleaving the two arms does not cancel it. What
survives there is the median: 1.132×, with a bootstrap 95% confidence interval of
[1.118, 1.145] over the sixty, and 56 of them above 1.00×. Quote that interval or nothing; a
lone run on this host can return almost anything. The two sweeps behind those sixty were
taken a fortnight apart and landed 0.4% apart, each interval containing the other's median —
which is the evidence that the protocol, and not just the row, reproduces.

**The margin is smaller on the desktop because the desktop's denominator is the faster one.**
binCV's code is identical in both columns. Stock `cv::goodFeaturesToTrack` runs at 1.68× the
binarized pipeline on x86-64 and 1.290× on the device, so OpenCV's x86 build is the one
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
shipped kernel's 43.49 and 1.731×. The denominator moved 0.8%, so the whole of the difference
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
11.5–13.6% of the assembled pipeline at [that benchmark's](feature-tracking.md) duty cycle,
so the end-to-end cost either way is small.

## Reproduce

```bash
./build/benchmark/lk_headtohead                  # optical flow, LK against LK
BINCV_FORCE_ITERS=1 ./build/benchmark/lk_headtohead
./build/benchmark/feature_benchmark              # FAST, BRIEF, matching
./build/benchmark/fast_bitplane_benchmark        # FAST on a bit-plane
./build/benchmark/corner_opencv_benchmark        # goodFeaturesToTrack
./build/benchmark/corner_subpix_benchmark        # cornerSubPix
```

All five are self-contained. Each column above is `./scripts/run_launches.sh -n <N> <binary>`
read back through `./scripts/aggregate_launches.py` — thirty launches on x86-64, ten on the
device. Logs — the sweep behind each column, and the single launch each replaced:
[optical flow](logs/lk_headtohead-x86_64-launches.log), [single](logs/lk_headtohead-x86_64.log), [aarch64](logs/lk_headtohead-aarch64-launches.log), [single](logs/lk_headtohead-aarch64.log) ·
[features](logs/features-x86_64-launches.log), [single](logs/features-x86_64.log), [aarch64](logs/feature-aarch64-launches.log), [single](logs/features-aarch64.log) ·
[bit-plane FAST](logs/fast_bitplane-x86_64-launches.log), [single](logs/fast_bitplane-x86_64.log), [aarch64](logs/fast_bitplane-aarch64-launches.log), [single](logs/fast_bitplane-aarch64.log) ·
[goodFeaturesToTrack](logs/goodfeatures-x86_64-launches.log), [first thirty](logs/goodfeatures-x86_64.log), [aarch64](logs/goodfeatures-aarch64-launches.log), [single](logs/goodfeatures-aarch64.log) ·
[cornerSubPix](logs/corner_subpix-aarch64-launches.log) — aarch64 only
