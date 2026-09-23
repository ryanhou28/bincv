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
| FAST, bit-plane | `cv::FAST` | 267.7 µs | **161.7 µs** | **1.651× [1.643, 1.658]** | 2051.1 µs | **865.2 µs** | **2.371× [2.368, 2.374]** |
| `goodFeaturesToTrack` | `cv::goodFeaturesToTrack` (stock) | 8.807 ns/px | **6.368 ns/px** | **1.383× [1.350, 1.426]** | 58.338 ns/px | **24.099 ns/px** | **2.421× [2.412, 2.424]** |
| `cornerSubPix` | `cv::cornerSubPix` | not published | not published | ~13× | 8.595 ms | **0.625 ms** | 13.76× [13.70, 13.80] |

**`FAST, bit-plane` is 1.651× on x86-64, up from a published 1.472×, and the gain is a gate
that was being read in the wrong place.** The runtime switch that makes the vector arm
provably off-switchable was read once per image row, which kept the scalar body live in
every row; read once per call instead, the whole operation is 180.85 µs → 161.70 µs, with
the intervals disjoint and the old code re-measured twice in the same session on either
side of the new one (180.85 and 181.30). The switch still works: the scalar arm reads
605.35 µs, 3.73× the vector line. Of the other moves here, `BRIEF` gaining 4.69× → 5.18× is the largest, and a
refactor that looked like its cause was A/B'd at thirty launches each and is not
(122,596 ns against 123,834, intervals overlapping) — the old cell was a slow draw.
**`1/1/1/1` resolves only 1.090× on this host**, so its interval is wide and that row should
not be read to three digits. [The index](README.md#on-the-x86-64-host) has the whole comparison.

**`cornerSubPix`'s measurements did not survive into the repository, and the device half of
them now has.** The two times were taken and the ratio recorded; the values behind it were
not, so the cells said so rather than being filled in. Ten device launches supply them —
8.595 ms against 0.625, a ratio of 13.76× where 13.70× was published — and the x86-64 half
is still a claim on trust, because that host has no sweep of this benchmark.

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

The `goodFeaturesToTrack` speed row is now measured against **stock
`cv::goodFeaturesToTrack`** rather than the hand-written binarized pipeline it used to lead
with; [Corner detection](#corner-detection) says why the denominator changed and prints both.
The footprint row still leads with the binarized pipeline, because that side's buffers are
READ off the objects and stock's can only be accounted.
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
| `CV_8U`, the pipeline's own frame | 6724 | 267.7 µs | **262.0 µs** | 1.021× [1.009, 1.024] | 2051.1 µs | 2052.1 µs | 0.998× [0.997, 1.001] |
| **bit-plane**, same frame | 6724 | 267.7 µs | **161.7 µs** | **1.651× [1.643, 1.658]** | 2051.1 µs | **865.2 µs** | **2.371× [2.368, 2.374]** |

**Parity on the wide-image entry point is the honest outcome and it ships that way.**
`cv::FAST` is a mature vectorised kernel, and a caller who is holding bytes should not be
told to pack them first — for that caller the answer is that binCV matches OpenCV and costs
nothing to adopt.

The bit-plane overload is the interesting one. A caller who already has a binary image gets
1.65× on x86 and **2.371× on the device** on an input of 46,080 bytes against `cv::FAST`'s
360,960, bit-exact corner-for-corner with `cv::FAST` in scan order. It is one of the few
results *better* on the deployment target, and the reason is register pressure: the arc test
needs sixteen live vectors, and aarch64 has thirty-two vector registers where x86 has sixteen,
so the AVX2 form spends part of its win on spill traffic. The earlier conclusion that FAST
could not benefit from packing was true of the signature it had been given, not of the
operation.

Scoring is a substantial part of the cost. The bit-plane path chooses per chunk between a
per-corner transpose and arc masks; sweeping that threshold on x86-64 moves the whole operation
between 160.8 µs and 202.9 µs against `cv::FAST`'s 267.7 — 1.665× at the fast end, 1.318×
with the masks switched off entirely, and 161.7 µs or 1.651× at the shipped adaptive
setting. On aarch64 the sweep is flat at ~863.5 µs, because that path's own measurement
made the scored arm a loss on NEON and it is compiled out. binCV's score is a different
quantity from OpenCV's — the longest qualifying arc rather than the largest surviving threshold
— which is why this is Tier 2.

## Corner detection

**1.383× on the desktop and 2.421× on the reference device against stock
`cv::goodFeaturesToTrack`, at 12.56 bytes per pixel against its 29.00.**

Both spellings, at 640×480, returning the same corners and timed in the same interleaved run.
Time is nanoseconds per pixel and working set is bytes per pixel, so the smaller number is
the better one. Measured at commit `6d74d57`; thirty launches on x86-64, ten on the device.

| variant | x86-64 (ns/px) | x86-64 vs stock | aarch64 (ns/px) | aarch64 vs stock | working set (B/px) |
|---|---|---|---|---|---|
| `cv::goodFeaturesToTrack` (stock, the denominator) | 8.807 | — | 58.338 | — | 29.00 † |
| OpenCV, binarized (the correctness reference) | 14.452 | 0.609× [0.598, 0.632] | 75.532 | 0.772× [0.770, 0.776] | 36.94 |
| binCV, frame map | 7.490 | **1.176× [1.159, 1.213]** | 25.362 | **2.300× [2.291, 2.304]** | 16.54 |
| **binCV, streaming ring (shipped)** | **6.368** | **1.383× [1.350, 1.426]** | **24.099** | **2.421× [2.412, 2.424]** | **12.56** |
| binCV, streaming, vector arms off | 7.596 | 1.159× | 24.079 | 2.423× | 12.56 |

† Stock's working set is **accounted, not read** — `cornerMinEigenVal` materializes `Dx`,
`Dy` and a `CV_32FC3` covariance inside the call and `gftt` adds `eig` and a dilate
destination, none of which a caller can measure from outside. Every other figure in that
column is read off the objects.

### The denominator changed, and that is the point of this row

**Until now this row led with the binarized pipeline, and that was the wrong baseline.** The
denominator was `openCvBinarized()` in `benchmark/corner_opencv_benchmark.cpp` — about ten
OpenCV calls hand-written to reproduce binCV's exact semantics. CLAUDE.md: *"The bar for a
new implementation is the best existing option, not the worst."* Nobody writes that pipeline;
they call `cv::goodFeaturesToTrack`. Against the real call the published x86-64 figure
changed **sign**, and the reports printed the flattering one in the headline column.

So stock leads now, on both architectures, and the binarized pipeline stays exactly where it
earns its place: it is the **correctness reference**, and the only arm that can be one. binCV
and it agree on 723 corners of 723 with a worst displacement of 0.00 px over the benchmark's
four frames, and their response maps are bit-identical over 360,960 pixels on the real frame.
Stock cannot prove that, because it runs a 3×3 Sobel over the byte image and finds different
corners by construction — which is what Tier 2 means here.

**A speed ratio against an arm whose yield nobody recorded is speed at unknown accuracy**, so
the benchmark now records stock's: over the same four frames binCV returns **723 corners to
stock's 686**, and 540 of binCV's sit within 3 px of a stock one. The two arms are doing
comparable amounts of work under identical parameters. They are not finding the same corners
and this page does not claim they are.

### What the 1.383× replaced, and where it came from

At the previous commit (`cc06082`) this operation was **0.737× [0.721, 0.754] against stock
on x86-64 — a loss, in 0 of 30 launches above 1.00** — and 1.368× [1.364, 1.382] on the
device. Both sweeps were taken on these same two machines in the same session as the ones
above, and the denominators moved less than their own intervals between them, so the whole of
the difference is binCV's arm: **11.310 → 6.368 ns/px on x86-64 (1.78×) and 43.337 → 24.099
on the device (1.80×)**.

**The stage that paid for it had never been optimized, and the x86 host could not see it.**
`goodFeaturesToTrack`'s response sweep — the stage the packed representation accelerates — is
the *minority* of its own runtime; selection is the rest. `benchmark/detect_stage_profile`
was supposed to split that and on this host it put the spacing filter at **minus 8%** of a
detection, because it differenced prefix arms on a machine with 50% run-to-run scatter. An
instruction-count profile of the same frame is deterministic and named four stages nothing
had looked at: the spacing filter at 18.5%, `boxWordAt`'s out-of-range guard at 9.3%, the
running maximum at 6.8%, and the rank at 7% by instruction count but **27% by time**, which
is what a comparison sort over a pool of ties costs in branch mispredicts.

Two of those are worth stating as findings rather than as changes:

* **The running maximum never vectorized**, though the code carried a comment saying the
  compiler was free to turn it into a vector max-reduce. `max` over floats is not
  reassociable — NaN and −0.0 make the result depend on the order the lanes combine in — so
  GCC and Clang both leave it scalar whatever the flags. A response is never negative and
  never NaN, so reducing the IEEE **bit patterns** is the same answer and does vectorize.
* **The rank need not compare two corners at all.** The suppression sweep appends candidates
  in raster order and the threshold pass only compacts, so *reversed*, the pool is already in
  `CornerStronger`'s tie order — y descending, then x descending. The rank is therefore a
  stable sort on the response alone: counting passes over the byte lanes of its bit pattern,
  no comparator, nothing to mispredict. **4.9× on that stage.** This is the host form of what
  `backends/cuda/src/corner.cu` found on the device, where replacing a comparison sort with
  count, prefix-sum and emit was worth 4×.

`detect_stage_profile` is rebuilt on the library's own functions rather than a verbatim copy
of the kernel's prefix. The copy had gone stale — it kept the scalar running maximum and the
per-candidate heap after the kernel dropped both — so the arm that was supposed to be a
*prefix* of the shipped call ran slower than the whole of it. That is where the negative
stage came from, and a profile that can print one is not measuring the shipped kernel.

### Nothing was traded for it

**Peak working set is unchanged at 12.56 B/pixel**, which the pre-registered rule made a
ceiling rather than a budget. None of the five changes takes a byte: the spacing filter
reorders the accepted set it already had, and the rank's counting passes scatter into the
caller's own array past the candidates — slack the capacity contract already sizes for the
worst case. Where that slack is absent, or the top-K heap has reordered the pool,
`std::sort` runs instead and `Corner.RankArmsAgree` holds the two to the same answer.

**Output is unchanged**, which is what makes any of this a speed comparison: 723 corners of
723 against the binarized pipeline at 0.00 px, frame map and streaming identical corner for
corner, asserted before anything is timed, and 40 tests / 3717 assertions green on x86-64
and on the device.

### Reading the two hosts

**The device numbers are the trustworthy ones, and on x86 a single launch settles nothing.**
On the device the within-run spread is 0.16–1.09% and ten launches put the ratio between
2.412× and 2.424×. The x86 host scatters 12–30% across thirty launches of the *same binary*;
what survives there is the median and its interval, 1.383× [1.350, 1.426], with 30 of 30
launches above 1.00×. Quote that interval or nothing.

Two independent thirty-launch sweeps of this kernel were taken an hour apart on that host and
landed at 1.374× and 1.383× — within each other's intervals, which is the control this round
had. The published row is the one taken at the commit.

**The margin is wider on the device because the desktop's denominator is the faster one.**
binCV's code is identical in both columns. Stock `cv::goodFeaturesToTrack` runs at 1.64× the
binarized pipeline on x86-64 and 1.29× on the device, so OpenCV's x86 build is the one
getting more out of its machine — a property of the two OpenCV builds, not of binCV.

**The vector arms are provably running.** The `vector arms off` row is the same call with
`impl::cornerSimdEnabled()` false: on x86-64 it is 1.193× [1.167, 1.212] slower, and on
aarch64 it is 0.999× [0.9986, 1.0003] — the arm is x86-only by measurement, so the
architecture its own gate excludes reports 1.00× exactly as CLAUDE.md requires. A ratio near
1.00× on x86-64 would mean the fast path was not the one being timed.

### This row has been corrected three times

**The first version published 0.53× on both architectures, and that was wrong**: it timed the
frame-map spelling while that spelling was still on an older response kernel than the
streaming form every pipeline calls. Sharing one kernel took the frame map from 26.06 to
about 14.9 ns/pixel, and the row became 0.92× on x86 and 1.45× on the device.

**Those figures were stale when they were published, and that was the second correction.**
They were recorded at commit `05ce58f`, before the response sweep's tail was rewritten to run
eight pixels at a time; the page carried a note calling them conservative and the note stood.
Rebuilt and run in the same harness, that commit's kernel measured 51.93 ns/pixel on the
device and put the ratio back at 1.45× — the published number, reproduced — against the then
shipped kernel's 43.49 and 1.731×. **The split the previous version of this page called
genuine was not one.**

**This is the third, and it is a correction of the BASELINE rather than of the number.** Both
earlier corrections were measured against the hand-written binarized pipeline, so both were
true of a denominator nobody runs. Against the call people actually make, the kernel those
corrections were defending was losing on x86-64 the whole time, and this page printed
1.132× in its headline column while the row against stock read 0.673×. The figure was in the
report; it was not in the headline. That is what changed.

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
