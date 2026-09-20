# CUDA backend

The GPU backend, measured operation by operation against the `cv::cuda` call a caller
would otherwise make. Design: [ARCHITECTURE §8.5](../ARCHITECTURE.md). API and build:
[backends/cuda/README.md](../../backends/cuda/README.md).

Every number here sits on a bit-exactness result: `scripts/verify_cuda.sh` proves each
device kernel gives the host library's answer byte for byte, and every optimized arm is
held to its own reference arm's map in the same binary ([Coverage](#coverage)).

**Both sides' measured values are published, with the ratio beside them** — the
measurements are what make the ratio checkable, the ratio is there so nobody has to
divide in their head. How a speed difference is judged real is
[methodology-timing.md](methodology-timing.md); how a memory figure is measured is
[methodology-memory.md](methodology-memory.md). Read the first before quoting a ratio.

## Conditions

**Device.** NVIDIA GeForce RTX 3070 Ti (SM 8.6, 48 SMs, ~608 GB/s) under WSL2. CUDA 11.1
nvcc, g++-9 host compiler. Every figure is the median of 7 independent process runs
unless the row says otherwise.

**Two clocks.** *Kernel-resident* is CUDA events around the enqueued work — the per-frame
cost once data lives on device, and what the tables below use. *End-to-end* is the host
clock around upload + kernels + download + synchronize. WSL2 inflates launch overhead
specifically, so kernel-resident numbers travel better.

**One explicit stream on both sides.** OpenCV synchronizes the whole device on the default
stream — `if (stream == 0) cudaSafeCall( cudaDeviceSynchronize() );` sits in cudev's grid
transform, `cudafilters`, `cudawarping` and three times in `cudastereo`. binCV carries no
such guard, so it is the control:

| call | default stream (ms) | explicit stream (ms) | default ÷ explicit |
|---|---|---|---|
| `cv::cuda::threshold` | 0.0704 | 0.0102 | 6.91× |
| `cv::cuda::resize` INTER_AREA | 0.0559 | 0.0091 | 6.17× |
| `cv::cuda::pyrDown` | 0.0706 | 0.0098 | 7.18× |
| `cv::cuda` erode 3×3 (0.15 ms kernel) | 0.2125 | 0.1465 | 1.45× |
| `cv::cuda` median 3×3 (6 ms kernel) | 6.4260 | 6.2180 | 1.03× |
| *control* — binCV `threshold` | 0.0146 | 0.0142 | 1.03× |
| *control* — binCV `erode` 3×3 | 0.0095 | 0.0089 | 1.07× |

Both controls read ~1.00× and the surcharge scales inversely with kernel length, which is
what a fixed per-call sync must do. **The bar is the explicit-stream number**, because a
resident pipeline uses streams and OpenCV supports them on every call here. Three figures
were inflated by it and are corrected below — `threshold`, `edgeThreshold` and the pyramid
ladder. Morphology, the medians and the StereoBM rows were unaffected. The default-stream
pair in `cuda_sensor_benchmark` that printed **"7.35× FASTER"** for `threshold` was deleted
rather than repaired; `cuda_role_benchmark` owns that comparison, and it reads a null.

**One memory meter per comparison.** An earlier **"1.7× smaller"** for the census working
set — 6.0 MB against 10.0 MB — was not a both-sides reading on one region and **should not
be quoted again**. A `cudaMemGetInfo` delta is the only meter readable
on both sides, so it is the meter for every cross-library figure, and it is never mixed
with binCV's own allocation sums. This driver reserves in 2 MB units, so each side is
replicated until its own total clears eight units; at 256 replicas one unit is 8.0
KB/frame. `GpuMat` pads its pitch and may be backed by a `BufferPool`, so OpenCV's side is
an **upper** reading: each binCV lead below is a lower bound, and so is the one loss.

**The launch floor** on this host was measured at **8.66–9.58 µs** in every run of the
sensor-stage benchmark, and at **11–13 µs** for the shape the device-occupancy probe
measured. Neither is derived from the other.
Two sensor-stage ops and all of binCV's binary morphology sit on it at the frame sizes a
vision pipeline runs, so a figure within a small multiple of either is a launch-cost
figure rather than a kernel one.

## Speed, operation by operation

**The binary dense-disparity path leads `cv::cuda::StereoBM` on both axes at once** —
faster *and* lighter. That is the operating point a binCV pipeline runs: it already holds
one bit per pixel, and on bits the dense cost is one XOR per 32-pixel word.

**Every `ratio` column below is `cv::cuda` ÷ binCV: above 1× means binCV is ahead, below 1× means `cv::cuda` is.** Where a table divides something else, its header says so.

Kernel-resident clock, both arms on one explicit stream, 752×480 unless the row names a
geometry, medians of 7 independent process runs, from `cuda_role_benchmark`. Time in
milliseconds, so the smaller cell is the faster side and the faster side is bold.

| operation | `cv::cuda` arm | cv::cuda (ms) | binCV (ms) | ratio | what the rounds say |
|---|---|---|---|---|---|
| `denseDisparityBinary` — the binary entry | `cv::cuda::StereoBM(64, 9)` | 0.7152 | **0.0648** | **11.0×** | 7/7 disjoint; a result (10.95× against a 1.78× bar) |
| census entry (2 transforms + match) | ″ | 0.6996 | **0.5076** | **1.47×** | 7/7; a result (1.47× vs 1.25×) — but it **loses on memory** |
| census matcher alone | ″ | 0.6864 | **0.3692** | **1.87×** | 7/7; a result (1.94× vs 1.37×) |
| `calcOpticalFlowBlockMatch` | `SparsePyrLKOpticalFlow` | 0.2320 | **0.0540** | not published | 7/7; speed met, **accuracy floor unset** |
| `detectFastAsync` | `FastFeatureDetector` | 0.1459 | **0.0247** | **6.01×** | 7/7; a result (6.01× vs 3.39×), from **14.84× behind** |
| `computeBrief`, N=1000 | `cv::cuda::ORB::computeAsync` | 0.1070 | **0.0107** | **9.3×** | 7/7; a result (9.25× vs 2.10×) |
| `matchDescriptors`, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 1.9491 | **0.2189** | **9.1×** | 7/7; a result (9.11× vs 1.29×) |
| `goodFeaturesToTrackAsync`, wall clock | `createGoodFeaturesToTrackDetector` | 3.7282 | **0.8405** | **5.3×** | 105 of 105 rounds, 3.53× to 15.52×; a result (4.42× vs 3.16×) |
| `cornerMinEigenValAsync` | `createMinEigenValCorner` | **0.0515** | 0.0590 | null result | 23 of 105 rounds binCV's way; 1.15× against a 1.76× bar |
| `calcOpticalFlowPyrLKAsync`, 204 pts | `SparsePyrLKOpticalFlow` | 0.1475 | **0.0792** | **1.35× to 4.36×** | 105 of 105 rounds; direction established, magnitude a null |
| `calcOpticalFlowPyrLKAsync`, 2048 pts | ″ | **0.3287** | 0.3558 | null result | 98–7, seven rounds crossed — the crossover |
| `threshold` → bits, 752×480 | `cv::cuda::threshold` | 0.0091 | **0.0084** | null result | 44–60 with one round tied |
| `threshold` → bits, 1920×1080 | ″ | 0.0116 | **0.0100** | null result | 17–88 |
| `threshold` → bits, 3840×2160 | ″ | 0.0362 | **0.0268** | null result | 8–97; the 1.35× median is unquotable on this host |

**A null result is not a loss.** It means neither the direction nor the size cleared this
host's noise, and it is itself a result. `threshold`'s own written rule was a *fail*
condition — slower than `cv::cuda::threshold` by more than both spreads — and a null is
not slower; it was 2.22× slower at 4K one round ago and is now never above 1.00× at any
geometry.

**Block matching is the one row these reports never published as a factor.** The ratio is
published only as 0.230 (binCV ÷ cv::cuda), so the cell names the side and not a number
rather than inventing one.

**Two rows carry a denominator that moves.** `goodFeaturesToTrack`'s `cv::cuda` arm swings
**3.24–13.61 ms** across runs because its min-distance spacing filter runs on the CPU;
binCV crosses parity against every version of it, including round 2's quieter 3.55–4.28 ms.
And **FAST's direction is certain while its size is not**: over 21 independent processes
the per-run median ratio moves between **3.44× and 6.26×** because binCV's arm is bimodal
(≈0.0205 or ≈0.0310 ms by process) against a steady `cv::cuda` arm, and every one of those
21 runs is 15–0 binCV's way.

**The three dense rows come from one of two sweeps of the same machine, and both are
defensible.** The other reads the binary entry at **0.0679 ms against 0.7438 — 10.8×**,
the census entry at **0.5418 against 0.7584 — 1.43×** and the census matcher at **0.4223
— 1.88×**. Which to quote is **not settled here**; the census row's run-to-run scatter is
1.14×, so its 1.38×, 1.43× and 1.47× readings are one number rather than three.

## Memory, operation by operation

`cudaMemGetInfo` delta taken identically on both sides, 752×480, peak working set per
frame. Kilobytes, so the smaller cell is the lighter side.

| operation | `cv::cuda` arm | cv::cuda (KB) | binCV (KB) | ratio |
|---|---|---|---|---|
| `denseDisparityBinary` (2 bit planes + map) | `cv::cuda::StereoBM(64, 9)` | 3,072.0 | **448.0** | **6.857×** |
| census entry (2 wide + 2 descriptors + map) | ″ | **3,072.0** | 4,512.0 | `cv::cuda` smaller, by **1.47×** |
| `computeBrief`, N=1000 | `cv::cuda::ORB::computeAsync` | 2,048.0 | **48.0** | **42.67×** |
| `matchDescriptors`, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 8,277.3 | **400.0** | **20.7×** |
| `detectFastAsync` @ capacity 32,768 | `FastFeatureDetector` | 680.0 | **432.0** | **1.574×** |
| `goodFeaturesToTrackAsync` | `createGoodFeaturesToTrackDetector` | 10,240.0 | **1,920.0** | **5.33×** |
| `cornerMinEigenValAsync` response map | `createMinEigenValCorner` | 10,240.0 | **2,048.0** | **5.00×** |
| LK tracker resident state | `SparsePyrLKOpticalFlow` | 1,408.0 | **448.0** | **3.14×** |
| `threshold` | `cv::cuda::threshold` | 1,024.0 | **416.0** | **2.46×** |

**The binary entry's 448.0 KB is the format's own arithmetic.** Two bit planes at 24 words
of 32 bits per row plus a 752×480 byte map is 442.5 KB against 448.0 metered, the
difference being the driver's rounding across three allocations. It reads 448.0 KB/frame
at both 64 and 256 replicas — 1.0000× — and identically to the byte in every one of seven
independent processes. StereoBM's footprint is **not content-dependent**, which had to be
checked rather than assumed: metered on a synthetic pair and on a real EuRoC pair it reads
3,072.0 KB both times.

**The census entry is the one row where binCV is larger, and that loss is the algorithm's
rather than this implementation's.** Census expands 8 bits a pixel into a 32-bit
descriptor word, so two transformed images are 2 × 1,410.0 KB = **2,820 KB before a
disparity map exists**, where StereoBM matches the 8-bit frames directly. No
implementation of census in this layout can undercut its own descriptors. This report
publishes that row both ways — **0.68×** as cv::cuda ÷ binCV, **1.47×** as binCV ÷
cv::cuda — and the table quotes the second because it names the winner. Re-read at 256
replicas the row is 4,504.0 KB against the same 3,072.0 — 1.466×, the 8 KB being one meter
unit. A caller who wants the memory back can have it: the plane-layout intermediate is
3,217.5 KB against the packed one's 3,877.5 KB, at 21.7× the matcher time.

**So the census entry is a split verdict — ahead on speed, behind on memory — on the axis
this project breaks ties with.** Whether an operation may ship on those terms is a
judgement nobody has made, and it is recorded as one rather than rounded into a headline.
It is also the path where binCV has **no structural advantage**: the layout that made it
fast is the conventional one-word-per-pixel descriptor, not bit-planes. The claim lives in
the binary entry, whose caller already holds bits.

## The families measured by their own benchmarks

The sensor, window and pyramid ops are compared against the same `cv::cuda` calls, on the
family benchmarks rather than on `cuda_role_benchmark`: nine independent process runs, each
an interleaved median of fifteen rounds, both arms on one explicit stream. Memory is taken
at 752×480 only, because the ratio is not constant in frame size.

| operation | denominator | geometry | denominator (ms) | binCV (ms) | ratio | memory ratio |
|---|---|---|---|---|---|---|
| `edgeThreshold` | composed `createDerivFilter` + abs + threshold | 752×480 | 0.0821 | **0.0096** | **8.1×** | **24.6×** |
| `edgeThreshold` | ″ | 1920×1080 | 0.1886 | **0.0164** | **11.1×** | not measured |
| `erode` rect 3×3 | `createMorphologyFilter` | 752×480 | 0.1429 | **0.0118** | **11.6–11.9×** | **16.0×** |
| `erode` rect 3×3 | ″ | 1920×1080 | 0.2011 | **0.0148** | **13.0–13.5×** | not measured |
| `erode` ellipse 5×5 | ″ | 752×480 | 0.2196 | **0.0154** | **12.8–13.9×** | (as rect 3×3) |
| `erode` ellipse 5×5 | ″ | 1920×1080 | 0.4048 | **0.0176** | **22.9×** | not measured |
| `morphologyEx` OPEN 3×3 | ″ | 1920×1080 | 0.4209 | **0.0238** | **15.1–16.1×** | not measured |
| `medianWide` K=9 | `createMedianFilter` | 752×480 | 6.1888 | **0.0249** | **235–249×** | **123×** |
| `medianWide` K=9 | ″ | 1920×1080 | 34.1522 | **0.0538** | **634–740×** | not measured |
| `denoiseMedian3` | binCV's own byte `medianWide`, fast arm on | 4096×2160 | 0.0388 | **0.0087** | **4.6–5.0×** | **8.58×** |
| `buildPyramidBox` ×3 | `cv::cuda::resize` INTER_AREA ×3 | 752×480 | **0.0233** | 0.0240 | a tie — published as 1.02×, binCV ÷ denominator | **7.17×** |
| `buildPyramidBox` ×3 | `cv::cuda::pyrDown` ×3 | 752×480 | 0.0256 | **0.0239** | a tie — published as 0.95×, binCV ÷ denominator | (same ladder) |
| `shift` | `cudaMemcpy2DAsync` (a pitched DMA) | 752×480 | 0.0092 | **0.0091** | a wash — published as 0.948–0.985×, binCV ÷ denominator | **7.8333×** |

Four caveats, each of which changes how a row reads.

- **`edgeThreshold`'s bar is the composed `cv::cuda` spelling** of the same computation
  (`createDerivFilter(CV_8UC1, CV_16SC1, ksize=1, normalize=false)`, whose kernel at ksize
  1 is exactly `[-1,0,1]` and whose default border already matches), which is separable —
  nine launches against binCV's one. The rule written first asked for ≥5× on memory and
  ≥3× on speed; measured 24.6× and 8.1–11.1×, ranges disjoint in 9 of 9 runs at 1080p. A
  previously circulated **37.8–49.9× is withdrawn** as the default-stream artifact it was.
- **Morphology's margin is mostly OpenCV's per-call cost.** A single-call probe shows
  OpenCV's kernel alone is 69–85% of its batched time while binCV's morphology sits at
  **1.36× the launch floor**, so roughly 10–17× of the ratio is kernel-to-kernel and the
  rest is OpenCV's host overhead. Earlier family figures of 17.4×/18.6×/16.3× became
  11.6×/13.3×/12.8× under the corrected protocol — same verdict, smaller magnitude.
- **The 123× on the wide median is the competitor's design, not binCV's representation.**
  OpenCV's `filtering.cpp` sizes its histograms at roughly **98 MB of device scratch for
  one 752×480 frame**; binCV allocates zero, because no kernel here heap-allocates. The
  implementer's own written expectation for that row was parity. **The K=9 row is the fair
  one**, since it compares equal sample counts.
- **The pyramid row was written down as a pass before it was measured.** Its disposition
  table's middle row said *ranges overlap ⇒ tie, which passes*; measured 1.015× and 1.039×
  across two sweeps with 0 of 9 runs disjoint. A **5.61× figure is withdrawn** — `resize`
  was paying 6.17× and `pyrDown` 7.18× on the default stream. The ladder itself is 93.5 KB
  in one `cudaMalloc`, equal to the closed formula to the byte.

## Where the leads come from

One mechanism per published number, and nothing here is a second claim.

- **The binary entry: one thread owns one 32-pixel *word*.** The raw cost is one XOR, the
  nine-wide horizontal sum is a carry-save tree into four bit-planes, and the
  winner-take-all is the host library's own `planesLess`/`planesSelect` ported rather than
  reinvented — **6.0× over the arm it replaced and 25× over the reference**, at an unchanged
  442 KB and zero shared memory. Every earlier arm handed `__popc` a 9-bit run in a 32-bit
  register ([#63](https://github.com/ryanhou28/bincv/issues/63)).
- **The census entry: a warp-cooperative separable box matcher.** One lane owns one
  descriptor column and slides its vertical sum in a register; the horizontal aggregation is
  a compile-time decomposition over `__shfl_down_sync`, 4 shuffles at winW = 9. **2.49×
  [2.40–2.50]** over the packed matcher, 7 of 7 disjoint, at **0 bytes** of added scratch and
  0 shared memory, with its gate-excluded control at winW = 19 reading 1.00× [0.99–1.03].
  The packed layout under it was worth 8.55× over the plane block
  ([#62](https://github.com/ryanhou28/bincv/issues/62)).
- **FAST: the ordering, then the record.** A single-block bitonic raster sort became
  count → prefix-sum → emit, off-switch ratio **35.19×**, and no two corners are ever
  compared. The memory half was two accidents: `fastScratchBytes` handed every caller the
  *reference* arm's scratch — 512 KB to use 380 bytes of — and the corner record carried the
  host's 64-bit score for a value that is an arc length and cannot leave [1, 16]. The record
  is 12 bytes, and the narrowing is proved by sweeping all 65,536 ring patterns at all
  sixteen arc lengths rather than argued.
- **`goodFeaturesToTrack`: the selection was 98% of the operation** and ran in one block.
  The ordering key is now the corner itself — `key = (~responseBits)<<32 | (0xFFFF−y)<<16 |
  (0xFFFF−x)`, whose ascending `uint64_t` order *is* the host's `CornerStronger` — so eight
  bytes replace sixteen with no payload array, and `goodFeaturesScratchBytes(65536)` fell
  **1,088 KB → 576 KB, 1.89×**.
- **`threshold`: two software divides.** It is header-only over the host's own cutoff
  composed with `cuda::packBits`, so the kernel under the number is `packKernel`, and
  `cuobjdump -sass` showed 184 instructions around one LDG and one STG — including
  `wordIdx / words` and a 64-bit divide. A row-grid shape deletes both and a byte-lane shape
  reads four pixels per lane: **184 instructions became 104**, and the profile moves from
  67.2% SM / 19.3% DRAM to 49.4% / 63.8%, where a packer belongs.
- **Descriptor matching: kernel shape, not popcount width.** `cuda/descriptor.hpp` claimed
  word emission gave a matcher "a real ~4× instruction advantage" over `cv::cuda`'s `uchar`
  popcounts. The instruction count is right and **it buys 1.02×**, measured on OpenCV's own
  side over identical bytes, because the kernel is `__syncthreads()`-bound. The prediction is
  recorded as made and falsified; the header has been corrected. The lead is two
  `__syncthreads()` per launch against one per descriptor chunk per train block.
- **Lucas-Kanade: the launch shape, and the kernel is a loss.** Profiled with both sides in
  one run at 61 keypoints, binCV is 65.4 µs in one launch against `cv::cuda`'s
  9.3 + 12.2 + 10.2 + 10.2 = **41.9 µs** — **a 1.56× kernel loss inside a wall-clock win**,
  which is also why the lead stops above 512–1024 keypoints. The counter-fact is in the same
  profile: `cv::cuda`'s `pyrlk::sparseKernel` does 146 MiB of local-memory loads per launch
  and binCV's does zero.
- **The batched covariance is a signature, not a kernel.** 200 windows of 31×31 in one
  launch is **467× faster** than 200 launches of the single-region form, which pays ~5–10 µs
  of launch overhead against nanoseconds of work. Both forms compute identical counts, so
  the 467× is purely what the signature costs.

## What is not delivered

- **`cornerSubPixAsync` misses its own round-trip rule, and the median has changed sides.**
  Measured as ONE paired comparison rather than three separately-timed medians added
  together, against the whole-plane download — the tighter of the two baselines by 50× —
  the device arm reads **0.5656 ms** against the round trip's **0.3911 ms**, with **72 of
  77** rounds favouring the round trip: 1.44× apart against a 1.39× bar, where fourteen
  earlier runs read a null at parity. Either reading is a miss, not the 1.07× in the device
  arm's favour published before. The limiter is a ceiling of the signature: at 200 corners
  the kernel is 6.25 warps of work on a part that holds 2,304 (2.1% achieved occupancy),
  and the one decomposition that would add parallelism — a warp per corner — is what
  `subpix.hpp`'s bit-exactness argument forbids, since double addition is not associative.
  About 450 corners are needed before a second warp per SM exists, so **below roughly 450
  corners a caller should refine on the host**. **This one is a stop-and-ask.**
- **`cornerMinEigenValAsync` did not resolve, for the third round running.** 1.15× apart
  against a 1.76× bar and 23 of its 105 rounds fall binCV's way, so both halves of the
  verdict decline it. The limiter is named — neither arm is at 80% of either roof, so the
  stall histogram decided and it says **occupancy**: OpenCV's kernel runs the same shape at
  **87.5% achieved occupancy** against binCV's 16.0%. What is missing is resolution, not an
  explanation.
- **The census entry is LARGER than `cv::cuda::StereoBM`**, and faster on the same run. A
  split verdict, above.
- **Block matching's accuracy floor and sparse stereo's residual floor are unset.** Both
  clear their speed and memory bars; neither has a stated accuracy magnitude, and inventing
  one is forbidden. Block matching is **ship-blocked** on that. Sparse stereo ships on its
  stated bar — 0.019 ms against a 0.39 ms bar at 135.6 KB against 442 KB, no scratch — with
  its accuracy in plain sight: against a pair with exact ground-truth disparity 21, **236 of
  500 within half a pixel and 261 pinned at the scan's low edge**, mean 2.11 px. That
  behaviour is bimodal on a globally thresholded frame and it is the *host* algorithm's; the
  fix is a richer packing, not this kernel.
- **Lucas-Kanade leads only up to about 512–1024 keypoints**, and the win is the launch
  shape rather than the kernel. Whether it ships with the density named in the header, or
  the traversal is redesigned first, is the ship-rule escape and it is unmade. The *second*
  question that rode with it — whether a lead this host cannot size is a lead at all — is
  **ruled and closed**: the direction is established by the rounds and the magnitude is
  published as a range rather than as one number
  ([methodology-timing.md](methodology-timing.md#the-case-that-forced-the-ruling-and-where-it-landed)).
- **The gated matcher's device speed rationale is not established.** Gated against brute
  force reads 1.11× at an admitted fraction of 4.73%, which this host's noise cannot
  resolve. It ships for the caller who already has the gate, not on that number.
- **Device occupancy was dropped on a measurement.** `markOccupiedBatch` / `occupiedBatch`
  / `clearOccupancy` have no equivalent anywhere in OpenCV, and the best existing option
  for the job is the host library's own `spaceCandidates` at **3,333 ns and zero bytes** —
  *below* this host's 11–13 µs launch floor, so no device shape can clear it: one launch
  costs more than the whole host arm.
- **The geometry stays on the host.** Five-point RANSAC was measured on device and **lost
  by at least 15.9×** — median 18.58× over seven runs, range 15.88–19.54, ranges disjoint
  — so nothing shipped. The binding constraint is not FP64 rate but the solver's
  6,800-byte per-thread local frame, which is why a faster-FP64 part would not obviously
  change the answer.

## Operations with no `cv::cuda` counterpart

These operations have no OpenCV counterpart at any API level, on CPU or GPU, so no speed
bar exists and none was invented. They ship on correctness, memory and the host comparison,
with the speed verdict **OUTSTANDING**, and **no CPU number is quoted in place of a missing
GPU one anywhere on this list**:

- **the gradient covariance** (`gradientCovarianceAsync`, `gradientCovarianceBatchAsync`) —
  `cornerHarris` and `createMinEigenValCorner` compute a dense float response *through* a
  covariance; neither exposes one. Scratch is **0 B**, verified as a `cudaMemGetInfo` delta
  of exactly 0 across 200 batch launches, at ≤24 B/window.
- **orientation** — no `cv::cuda` entry point orients provided keypoints.
- **`keypointsFromCorners`** — the detector-to-keypoint-set link, which exists because a
  resident pipeline needs no mid-frame synchronize and a host pipeline does not need the op
  at all. It is the one op here that met both halves of its rule: 1.00 synchronize per
  frame against the round-trip arm's 2.00, and **0.0043 ms against 0.1547 ms — 36×**, with
  frame totals disjoint in all 7 runs.
- **`shift`**, **`binarize`** and the 16-bit **`medianWide`**, each measured above or
  against binCV's own arm rather than against an invented bar.
- **`stereoDescriptorMatch`, `stereoRefineDisparity`, `stereoMatchRectified`** — OpenCV's
  sparse stereo is `StereoBM`'s dense map plus a host lookup, not an operation.
- **`matchDescriptorsGated`** — no library exposes a gated matcher.

`calcOpticalFlowBlockMatch` is measured against `cv::cuda`'s LK above because that is the
best existing option for the job, not because it is the same operation
(`cv::cuda::FastOpticalFlowBM` is a dense field, not a sparse tracker).

**Descriptor matching left this list**, because it now has a device arm and therefore a
real bar; it is timed above.

## The assembled pipelines

These are **not operations and they have no `cv::cuda` counterpart.** They are binCV's own
example pipelines, built out of the operations above and measured against binCV's own host
library on the same machine. They exist to show that the ops compose bit-exactly and that
residency is where launch cost gets amortized — not to make a claim against OpenCV. Issues [#58](https://github.com/ryanhou28/bincv/issues/58) and
[#59](https://github.com/ryanhou28/bincv/issues/59) were judged on them. A reader choosing an operation should use the tables
above.

**The resident VIO frontend** (`backends/cuda/examples/cuda_vio_frontend.cpp`): sensor
stage → pyramid → derivatives → `goodFeaturesToTrack` → `keypointsFromCorners` →
orientation → BRIEF, over **400 real EuRoC V1_02 cam0 frames**, 7 process runs, one
explicit stream, CUDA events on the device side and the host library's own clock on the
host side.

- **0.945 ms/frame [0.918–1.065] against the host arm's 5.313 ms [5.198–5.422] — 5.62×**,
  ranges disjoint in all 7 runs *and globally* (device maximum below host minimum).
- It was **1.22× slower** one round ago, at 6.462 ms. The host arm reads 5.313 identically
  in both rounds, which is what makes the move like-for-like: the device side went
  6.462 → 0.945 and nothing else changed. Detection fell from **97.0% to 75.2%** of the
  device frame while everything else moved 0.194 → 0.209 ms, which is the Amdahl check.
- Residency itself: exactly **1.00 synchronize per frame** in all 7 runs, and 360,960 B up
  against 11,536 B down — a 31.3× bus asymmetry with nothing frame-sized returning.
- Correctness over 400 frames × 7 runs: **0** corner-count, **0** position, **0** keep-byte,
  **0** rotation-bin and **0** descriptor-word differences. Peak device memory 2,106,180 B
  (allocation sum).

**The tracking sequence**: the same 400 frames through the sensor stage, the shipped
1/2/2/2 ladder, the previous frame's derivatives and LK, at a fixed re-detection cadence.
Wall clock on both arms, with the H2D upload and one stream synchronize per frame **inside**
the device arm's clock. Agreement is checked before any timing in every run — level-0
frame 11,520 words compared, 0 differ; the ternary derivative planes 46,080 words, 0
differ; 204 = 204 corners with 0 position differences.

| cadence | host binCV (ms/frame) | device binCV (ms/frame) | host ÷ device |
|---|---|---|---|
| re-detect every 10 frames | 1.932 [1.779–1.992] | **0.424** [0.383–0.471] | **4.56×** |
| re-detect every frame | 6.455 [6.388–6.652] | **1.056** [0.974–1.159] | **6.11×** |

Ranges disjoint 7/7 and globally on both rows. Peak device memory, identical in all 7 runs
because every allocation is made at construction: **2,212.7 KB** for the whole resident
state and **451.7 KB** for the tracker alone. The host x86 arm is not timing-grade here, so
these rows locate a magnitude rather than a decimal.

## Recorded negatives

Shapes that were measured, lost, and should not be retried without new information. Each is
recorded in full on the issue it belongs to.

- **The census matcher** — shared-memory staging of raw per-pixel costs (1.28× slower;
  the redundant loads were already L1 hits), the disparity-tile sweep (4 → 1.97 ms,
  16 → 1.11, 8 → 0.91), and `__launch_bounds__` forcing the register budget instead of
  shrinking the state (0.98× / 1.70× / 1.59× against 2.53× unbound) — [#62](https://github.com/ryanhou28/bincv/issues/62).
- **The binary matcher** — six attempts that together located the limit: a `planes == 1`
  specialization (1.23× slower), hoisting the disparity tile's right-image loads (null),
  strip length 8 → 0.421 ms and 32 → 0.649 against 16's 0.395, packing cost and disparity
  into one register (84 registers, no movement) and `__launch_bounds__` at 4/6/8 blocks per
  SM (worse at every setting). All six attacked *memory* pressure on a kernel that was never
  memory-bound; reading them as a set is what pointed at arithmetic density —
  [#63](https://github.com/ryanhou28/bincv/issues/63).
- **The frontend's dropped arms** — a `__dp4a` wide-orientation arm (0 of 6 runs disjoint at
  the operating point, 1.46× only at N=100,000), a funnel-shift covariance arm (1.13×
  slower where the comparison is decidable), a one-thread-per-window covariance and a
  dense-rank counting sort for FAST scores, plus the packer's row-grid arm left flagged
  rather than settled — [#58](https://github.com/ryanhou28/bincv/issues/58).
- **The tracker's unbuilt arms** — the lane-0 broadcast and the thread-per-keypoint shape,
  both predicted to lose from one profile rather than six experiments, and `pyrDownBox`'s
  bit-sliced arm left as an open disposition — [#59](https://github.com/ryanhou28/bincv/issues/59).
- **The window family** — a fused single-kernel OPEN/CLOSE, a shared-memory tiled
  morphology arm refused on the 1.28× precedent, and the log-depth fold that is the best
  remaining unexploited win in the family but buys nothing while the ops are launch-bound —
  [#60](https://github.com/ryanhou28/bincv/issues/60).
- **Everything the profiler found and nothing implemented** — the ranked opportunity list
  R1–R10, the profiler-enablement recipe, and the readings that **contradicted** four
  documented limiters while leaving every decision standing, are on
  [#64](https://github.com/ryanhou28/bincv/issues/64). Only the profile readings that explain a published number
  above are repeated here. `nsys` and `compute-sanitizer` do not work on this machine, so
  no device out-of-bounds read in this backend is observable by any tool here; two guards
  are pinned by swept arithmetic invariants instead.

- **Five negatives have no issue home and are recorded here so they are not lost.** A
  `uint4` arm for `denoiseMedian3` shipped bit-exact, timed on the full ladder and was
  **dropped with its off-switch**: it never separated (0.97–1.04× at every rung), because
  at 4096×2160 the whole operation moves 2.21 MB ≈ 3.6 µs of traffic against a 7–10 µs
  launch. The host library's 4.81× for a bit-sliced blockSize-3 corner response **does not
  port** — on device it is 0.74×, slower with ranges disjoint, because the host's win was
  removing per-pixel addressing where the device form hands one thread 32 pixels of `sqrt`.
  `cornerSubPixAsync`'s set-bit-skip control, in which both arms are the same kernel doing
  the same work, reads **skip-ON 4% slower in 77 of 77 rounds and 154 of 154 across a
  second sweep** — inside the ±5% band it asserts, so nothing fails, but a one-sided sign
  count on identical code is a finding about that control. Narrowing the FAST record moved
  `fastOrderedApplies`' arm crossover, so over widths 32..2016 and heights 7..1199 **about
  4.5% of (frame, capacity) points now take the reference arm**, never above capacity 128
  — no answer changes, and the point is that a crossover that moves when a record size
  changes is a heuristic rather than a derived number. That narrowing also costs store
  traffic: `fastEmitKernel<9,1>` reads **53,949 global store sectors against the old
  record's 37,931**, invisible at 15% occupancy today and not invisible if
  [#64](https://github.com/ryanhou28/bincv/issues/64) succeeds.

**Timing and profiling never mix.** `ncu` serializes and replays kernels and locks clocks
to base, so no timing number in this report comes from a profiled run and no profile
reading is quoted as a duration.

## Coverage

`scripts/verify_cuda.sh` proves each device kernel gives the host library's answer byte for
byte — **seventeen suites, 194,975 checks in the Release configuration and 194,922 in the
Debug one**. The counts differ by design: a suite exercising a narrowed domain can only
test the half of that contract its configuration has — the assertion is live in Debug, the
error return reachable in Release — and each such suite prints which half it ran.

**One test-method finding qualifies that evidence.** Every dense-disparity case built its
right image as an *exact shift* of the left, where the correct disparity's window cost is 0
and every rival's is hundreds — so a deliberately broken halo passed 508 checks
byte-identically. Those cases now run on two contents each and all pass, and a mutation
battery against the box matcher fires 58–140 failures on each of seven mutations, naming
the three that are benign with the structural reason rather than papering over them
([#62](https://github.com/ryanhou28/bincv/issues/62)).

**One previously published count was too high.** The eight pre-frontend suites were
reported at 38,901 Release checks; **2,112 of those never existed as distinct assertions**,
because `BINCV_CHECK_EQ` evaluated its first argument twice and `test_cuda_median` passes
it a helper containing four checks — 528 invocations × 4 = 2,112 exactly, and
38,901 − 2,112 + 3 = 36,792 (they read 36,844 today). Coverage is unchanged; the second
execution tested nothing the first had not. The macro now binds the value once, which also
stopped `BINCV_CHECK_EQ(cudaFree(p), cudaSuccess)` being a double free.

**Twenty-two of the twenty-seven host operation headers have device arms** — the
reductions and dense stereo end to end (`logic`, `reduce`, `pack`, `census`,
`denseDisparity`); the sensor and window stages (`threshold`, `edge`, `morphology`,
`denoise`, `medianWide`, `pyramid`, `shift`); the frontend (`derivative`, `covariance`,
`corner`, `fast`, `orientation`, `descriptor`, `subpix`); and `opticalFlow`, `blockMatch`
and the sparse-stereo half of `stereo`. Two device operations have no host header at all:
`keypoints.hpp` and `sparseMatch.hpp`'s descriptor matcher, both of which a resident
pipeline needs and a host pipeline does not. The geometry has no device arm, and that is
the round's answer rather than its backlog.

Several device ops accept a **narrower domain than their host twin**, and each names it in
its docstring, asserts it, and returns `cudaErrorInvalidValue` outside it rather than
computing a wrong answer: morphology takes elements up to 32 rows by 512 columns (32
masked), `medianWide` takes K ∈ {1,3,5,7,9} at compile time, `pyrDownBox` takes 1–8 planes
a side, `binarize` takes 1–32, the derivative takes N ∈ [1,4], and steered BRIEF names its
angle domain [-2π, 2π]. A Tier 1 claim here is a claim over *that* domain, said so where
the claim is made. The Debug configuration is what proves those assertions reach nvcc's
device pass, which is why it is a gate rather than a convenience.

## Reproduce

```bash
cmake -S . -B build -DBINCV_CUDA=ON -DBINCV_BUILD_BENCHMARKS=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.1/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=g++-9
cmake --build build -j
```

| table | binary |
|---|---|
| [speed](#speed-operation-by-operation) and [memory](#memory-operation-by-operation), operation by operation | `cuda_role_benchmark` — one process, one launch floor, every pair interleaved on one explicit stream, `cudaMemGetInfo` on *both* sides of every memory figure |
| the dense memory rows | `cuda_role_benchmark stereo`, section 7b |
| dense disparity, the optimization ladder | `cuda_dense_benchmark` |
| the foundation ops | `cuda_foundation_benchmark` |
| the sensor stage | `cuda_sensor_benchmark` |
| morphology and the medians | `cuda_window_benchmark`, `cuda_median_benchmark` |
| the pyramid ladder and `shift` | `cuda_pyramid_benchmark` |
| the StereoBM comparison, with a cudastereo-enabled OpenCV | `cuda_stereobm_benchmark` |

The GPU-against-GPU rows need an OpenCV built with the relevant `cuda*` modules, which no
packaged OpenCV ships; point `-DBINCV_CUDA_OPENCV_DIR=<prefix>` at such a build. Every
target that wants one is always built and reports its role bar as BLOCKED or OUTSTANDING
when it is absent.

**No raw output for these tables is committed under [logs/](logs/)**, which holds host
benchmark output only. Each table above names the binary that produces it instead.
