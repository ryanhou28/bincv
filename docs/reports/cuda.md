# CUDA backend

The GPU backend, measured operation by operation against the `cv::cuda` call a caller
would otherwise make, on one machine: an NVIDIA GeForce RTX 3070 Ti (SM 8.6, 48 SMs,
~608 GB/s) under WSL2, OpenCV 4.5.4 built with its `cuda*` modules, binCV built with CUDA
11.1's nvcc and g++-9 as host compiler. Every device kernel timed here is proven bit-exact
against the host library by `scripts/verify_cuda.sh`. **Both sides' measured values are
published, with the ratio beside them.** Design: [ARCHITECTURE §8.5](../ARCHITECTURE.md);
API, build and coverage: [backends/cuda/README.md](../../backends/cuda/README.md); how a
difference is judged real: [methodology-timing.md](methodology-timing.md); how memory is
metered: [methodology-memory.md](methodology-memory.md).

## Conditions

**Protocol.** Every figure is the median of 7 independent process runs. Within a run each
row is timed in 15 paired rounds — a *round* is one interleaved timing of both arms, order
alternating — so a row rests on 105 rounds. The clock is kernel-resident (CUDA events around
the enqueued work) unless the row says wall clock.

**One explicit stream on both sides.** OpenCV calls `cudaDeviceSynchronize()` on the default
stream inside cudev's grid transform, `cudafilters`, `cudawarping` and `cudastereo`; on its
calls the default stream costs 1.41×–6.21× at the median over 7 runs (erode 3×3 1.41×,
`pyrDown` 5.25×, `resize` 6.15×, `threshold` 6.21×), on binCV's controls 0.98×–1.00×. A
resident pipeline uses streams, so the explicit-stream figure is the one compared against.

**One memory meter.** A `cudaMemGetInfo` delta on both sides, each side replicated until its
total clears the driver's 2 MB reservation unit, in KiB (1024 bytes). `GpuMat` pads its pitch
and may be pooled, so OpenCV's side is an upper reading and every binCV lead a lower bound.

**The launch floor.** An empty kernel under the same protocol costs 6.12 µs at the median
(range over 7 runs: 5.77–6.60); an arm within a small multiple of it is measuring launch cost.

## Speed, operation by operation

**The binary dense-disparity path leads `cv::cuda::StereoBM` on both axes at once** — faster
and lighter. That is the operating point a binCV pipeline runs: it already holds one bit per
pixel, and on bits the dense cost is one XOR per 32-pixel device word (64 on the host).

**Every `ratio` column in this report is `cv::cuda` ÷ binCV: above 1× means binCV is ahead,
below 1× means `cv::cuda` is.** Times in milliseconds, 752×480 unless the row names a
geometry, from `cuda_role_benchmark` (LK rows from the same binary on a real frame sequence;
block matching from `cuda_sparse_benchmark`).

| operation | `cv::cuda` arm | cv::cuda (ms) | binCV (ms) | ratio | range over 7 runs | rounds won by binCV; result |
|---|---|---|---|---|---|---|
| **`denseDisparityBinary`** — the binary entry | `cv::cuda::StereoBM(64, 9)` | 0.7134 | 0.06400 | 11.2× | 10.6–12.2 | 105 of 105; result (11.2× against a 2.28× bar) |
| census entry (2 transforms + match) | ″ | 0.7101 | 0.4789 | 1.50× | 1.38–1.54 | 105 of 105; result (1.50× against 1.25×); larger on memory, below |
| census matcher alone | ″ | 0.7190 | 0.3900 | 1.89× | 1.87–2.00 | 105 of 105; result (1.89× against 1.46×) |
| `detectFastAsync` | `FastFeatureDetector` | 0.1221 | 0.02078 | 6.02× | 5.77–6.43 | 105 of 105; result (6.02× against 3.08×) |
| `computeBriefSteered`, N=1000 | `cv::cuda::ORB::computeAsync` | 0.08960 | 0.009353 | 8.82× | 8.19–9.16 | 105 of 105; result (8.82× against 2.66×) |
| `matchDescriptors`, 5000² | `BFMatcher::knnMatchAsync(k=2)`, `CV_32S` | 2.009 | 0.2105 | 9.51× | 9.33–9.59 | 105 of 105; result (9.51× against 1.66×) |
| `goodFeaturesToTrackAsync`, wall clock | `createGoodFeaturesToTrackDetector` | 3.403 | 0.4240 | 8.76× | 7.17–9.23 | 105 of 105; result (8.76× against 3.72×) |
| `cornerMinEigenValAsync` | `createMinEigenValCorner` | 0.05192 | 0.02007 | 2.60× | 2.59–2.62 | 105 of 105; result (2.60× against 1.61×) |
| `calcOpticalFlowPyrLKAsync`, 204 pts | `SparsePyrLKOpticalFlow` | 0.1748 | 0.08931 | 1.41×–5.86× per round, median 2.01× | 1.77–2.17 | 105 of 105; direction established; magnitude a null (2.01× against 2.53×) |
| `calcOpticalFlowPyrLKAsync`, 2048 pts | ″ | 0.3769 | 0.3976 | 0.952× | 0.927–0.963 | 14 of 105; null (0.952× against 1.16×) — the crossover |
| `threshold` → bits, 752×480 | `cv::cuda::threshold` | 0.008326 | 0.007902 | 1.02× | 0.983–1.05 | 69 of 105; null (1.02× against 3.27×); both arms on the launch floor |
| `threshold` → bits, 1920×1080 | ″ | 0.01133 | 0.01149 | 0.990× | 0.953–1.01 | 45 of 105; null (0.990× against 3.56×) |
| `threshold` → bits, 3840×2160 | ″ | 0.03642 | 0.02550 | 1.43× | 1.39–1.44 | 98 of 105; null (1.43× against 2.47×) |
| `calcOpticalFlowBlockMatch` | none — OpenCV has no sparse block matcher | — | 0.05701 | no comparison possible | 0.05416–0.07977 ms | binCV's own figure over 7 processes, not compared |

*Rounds won* counts the 105 paired rounds in which binCV's arm was faster. *Result*: the
median per-round ratio exceeds the noise bar — the larger of the within-run swing of the
per-round ratio and the run-to-run scatter of the 7 per-run medians, as
`scripts/aggregate_cuda_runs.py` computes both from the log. *Direction established*: no round
crossed 1×, so the sign is settled and the per-round range bounds the size. *Null*: neither —
and a null is not a loss. *Range over 7 runs* is the smallest and largest per-run median ratio.

**`threshold` is a null at every geometry**, and its own rule was a fail condition — slower
than `cv::cuda::threshold` by more than both spreads; at 4K the 1.43× median sits under a 2.47×
within-run swing and is not quotable as a lead.

**`goodFeaturesToTrack`'s denominator is wall clock, and it moves.** `cv::cuda`'s detector
downloads its candidate list, runs the min-distance spacing filter on the host and uploads the
survivors, so a CUDA-event clock would exclude work a caller pays; its per-run median swings
3.363–4.186 ms (1.24×) over these 7 runs, and the row publishes the median per-round ratio
like every other because the median is the one statistic that is stable across runs.

**The three dense rows quote the committed sweep** (log below). An earlier, uncommitted sweep
read 10.8×, 1.43× and 1.88× for the same rows; the difference is inside the census row's 1.12×
run-to-run scatter.

**Lucas–Kanade is decided by corner density.** binCV is one launch with one warp per keypoint
and the level loop inside the kernel; `cv::cuda` is one launch per level with 256 threads per
keypoint, and those shapes cross. At 204 keypoints — the spacing of the reference pipeline
(the visual-inertial odometry system, not in this repository, that binCV was built to serve
stage by stage; see [ARCHITECTURE](../ARCHITECTURE.md)) — binCV is ahead in every round; at
512 the lead is unanimous but 1.62×; at 2048 it is gone. Both rows: a real EuRoC frame pair,
31×31 window, 4 levels, both pyramids resident on entry.

**Provenance.** Every row comes from one of three committed sweeps of 7 processes each:
[`cuda_role-x86_64-cuda-launches.log`](logs/cuda_role-x86_64-cuda-launches.log) taken at
`7c8055f`, [`cuda_role_lk-`](logs/cuda_role_lk-x86_64-cuda-launches.log) at `06246c2` and
[`cuda_sparse-`](logs/cuda_sparse-x86_64-cuda-launches.log) at `a608102` (all three on `main`
as `0d6e302`), read by `scripts/check_figure_staleness.py`, so a change to a first-party file
beneath a row ages that row.

## Memory, operation by operation

`cudaMemGetInfo` delta on both sides, 752×480, peak working set per frame, KiB. Every reading
below was identical in all 7 runs, so no range is given.

| operation | `cv::cuda` arm | cv::cuda (KiB) | binCV (KiB) | ratio |
|---|---|---|---|---|
| **`denseDisparityBinary`** (2 bit planes + map) | `cv::cuda::StereoBM(64, 9)` | 3072.0 | 448.0 | 6.86× |
| census entry (2 wide frames + 2 descriptor images + map) | ″ | 3072.0 | 4512.0 | 0.681× |
| `computeBriefSteered`, N=1000 | `cv::cuda::ORB::computeAsync` | 2048.0 | 44.0 | 46.5× |
| `matchDescriptors`, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 8192.0 | 368.0 | 22.3× |
| `detectFastAsync` at capacity 32,768 | `FastFeatureDetector` | 768.0 | 432.0 | 1.78× |
| `goodFeaturesToTrackAsync` | `createGoodFeaturesToTrackDetector` | 10240.0 | 1920.0 | 5.33× |
| `cornerMinEigenValAsync` response map | `createMinEigenValCorner` | 10240.0 | 2048.0 | 5.00× |
| LK tracker resident state | `SparsePyrLKOpticalFlow` | 1408.0 | 448.0 | 3.14× |
| `threshold` (src + dst) | `cv::cuda::threshold` | 1024.0 | 416.0 | 2.46× |
| `edgeThreshold` (every intermediate the chain needs) | composed `createDerivFilter` chain | 10240.0 | 416.0 | 24.6× |
| `erode` 3×3 (src + dst + filter buffer) | `createMorphologyFilter` | 1536.0 | 96.0 | 16.0× |
| `medianWide` (src + dst + filter buffer) | `createMedianFilter` | 101376.0 | 824.0 | 123× |
| `buildPyramidBox`, 4-level ladder resident | `resize` / `pyrDown` ladder | 688.0 | 96.0 | 7.17× |
| `denoiseMedian3` on bits | binCV's own byte `medianWide` — not `cv::cuda` | 824.0 | 96.0 | 8.58× |

**The binary entry's 448.0 KiB is the format's own arithmetic**: two bit planes at 24 words of
32 bits per row plus a 752×480 byte map is 442.5 KiB by allocation sum, the rest being the
driver's rounding across three allocations. StereoBM's 3072.0 is not content-dependent: a
synthetic pair and a real EuRoC pair read the same.

**The census entry is the one row where binCV is larger, by 1.47×, and the loss is the
algorithm's rather than this implementation's.** Census expands 8 bits a pixel into a 32-bit
descriptor word, so two transformed images are 2 × 1410.0 = 2820.0 KiB before a disparity map
exists, where StereoBM matches the 8-bit frames directly (at 256 replicas the row reads 4504.0
KiB, 0.682×, one meter unit from the 64-replica reading above). Ahead on speed and behind on
memory, the row is published as both numbers with no recommendation between them: binCV's own
tie-break puts memory first, and whether that trade is acceptable for a wide-input caller has
not been decided. It is also the path with no structural advantage — the layout that made it
fast is the conventional one-word-per-pixel descriptor, not bit planes. A caller who wants
memory back can take the plane-layout intermediate (3217.5 KiB against the packed 3877.5 KiB)
at 21.7× the matcher time, measured in `cuda_dense_benchmark`.

**The wide median's 123× is the competitor's design**: OpenCV sizes its histograms at roughly
98 MiB of device scratch for one 752×480 frame, and binCV allocates nothing.

## The sensor, window and pyramid families

The same `cuda_role_benchmark` sweep, same protocol, same columns as above. These operations sit
on or near the launch floor at 752×480, which is why the 1920×1080 rows are carried.

| operation | `cv::cuda` arm | geometry | cv::cuda (ms) | binCV (ms) | ratio | range over 7 runs | rounds won by binCV; result |
|---|---|---|---|---|---|---|---|
| `edgeThreshold` | composed `createDerivFilter` + abs + threshold | 752×480 | 0.06011 | 0.008038 | 7.53× | 7.19–8.34 | 105 of 105; result (7.53× against 3.17×) |
| `edgeThreshold` | ″ | 1920×1080 | 0.1799 | 0.01633 | 11.1× | 11.0–11.2 | 105 of 105; result (11.1× against 1.75×) |
| `erode` rect 3×3 | `createMorphologyFilter` | 752×480 | 0.08784 | 0.008000 | 10.8× | 9.39–11.7 | 105 of 105; result (10.8× against 2.32×) |
| `erode` rect 3×3 | ″ | 1920×1080 | 0.1251 | 0.01100 | 12.4× | 11.7–13.1 | 105 of 105; result (12.4× against 2.61×) |
| `erode` ellipse 5×5 | ″ | 752×480 | 0.1375 | 0.01128 | 12.5× | 10.9–14.8 | 105 of 105; result (12.5× against 3.29×) |
| `erode` ellipse 5×5 | ″ | 1920×1080 | 0.3120 | 0.01309 | 23.5× | 22.7–23.9 | 105 of 105; result (23.5× against 2.02×) |
| `morphologyEx` OPEN 3×3 | ″ | 752×480 | 0.2014 | 0.01588 | 12.2× | 11.1–13.2 | 105 of 105; result (12.2× against 3.42×) |
| `morphologyEx` OPEN 3×3 | ″ | 1920×1080 | 0.2643 | 0.01839 | 13.3× | 12.9–14.7 | 105 of 105; result (13.3× against 2.83×) |
| `medianWide` K=9 | `createMedianFilter` 3×3 | 752×480 | 5.692 | 0.02232 | 256× | 233–368 | 105 of 105; result (256× against 3.51×) |
| `medianWide` K=9 | ″ | 1920×1080 | 31.23 | 0.05263 | 593× | 587–669 | 105 of 105; result (593× against 1.36×) |
| `denoiseMedian3` on bits | binCV's own byte `medianWide` K=3 — not `cv::cuda` | 752×480 | 0.006950 | 0.006242 | 1.18× | 1.10–1.32 | 79 of 105, two ties; null (1.18× against 2.30×) |
| `denoiseMedian3` on bits | ″ | 1920×1080 | 0.01177 | 0.006400 | 1.89× | 1.77–2.14 | 100 of 105; null (1.89× against 3.41×) |
| `denoiseMedian3` on bits | ″ | 4096×2160 | 0.03881 | 0.007469 | 5.30× | 2.83–5.50 | 105 of 105; result (5.30× against 4.31×) |
| `buildPyramidBox` ×3 | `cv::cuda::resize` INTER_AREA ×3 — the same 2×2 box | 752×480 | 0.01946 | 0.01997 | 0.992× | 0.959–1.03 | 49 of 105, one tie; null (0.992× against 2.86×) — a tie |
| `buildPyramidBox` ×3 | `cv::cuda::pyrDown` ×3 — a Gaussian 5×5, a different filter | 752×480 | 0.02017 | 0.01861 | 1.09× | 1.01–1.16 | 71 of 105, one tie; null (1.09× against 4.25×) — a tie |
| `shift` | `cudaMemcpy2DAsync` — a pitched DMA, a reference rather than a counterpart | 752×480 | 0.009882 | 0.01027 | 1.01× | 0.893–1.20 | 55 of 105, two ties; null (1.01× against 5.16×) — a wash |

- **`edgeThreshold`'s counterpart is the composed `cv::cuda` spelling** of the same computation
  (`createDerivFilter(CV_8UC1, CV_16SC1, ksize=1, normalize=false)`, whose kernel at ksize 1 is
  exactly `[-1, 0, 1]` and whose default border matches), separable — several launches against
  binCV's one.
- **Morphology's margin is mostly OpenCV's per-call cost**: binCV's arm sits on the launch
  floor at 752×480. OpenCV's filter is `BORDER_CONSTANT` only, so binCV runs that border here.
- **The median rows compare different filters.** `createMedianFilter` is a 3×3 square with a
  replicated border; `medianWide` K=9 is the closest sample-count match and the fair row (the
  log's 3- and 5-sample rows, 599× and 430×, are cheaper operations, not the representation's
  win).
- **The pyramid rows are ties.** binCV's ladder is N-bit per rung (1, 3, 4, 5 bits), an output
  OpenCV has no type for, so this is a role comparison.

## Where the leads come from

- **The binary entry: one thread owns one 32-pixel word.** The raw cost is one XOR, the
  nine-wide horizontal sum is a carry-save tree into four bit planes, and the winner-take-all is
  the host library's own `planesLess` / `planesSelect` ported rather than reinvented, with no
  shared memory and no scratch.
- **The census entry: a warp-cooperative separable box matcher.** One lane owns one descriptor
  column and slides its vertical sum in a register; the horizontal aggregation is a compile-time
  `__shfl_down_sync` decomposition. Its off-switch control in the committed sweep reads 2.58×
  (range over 7 runs: 2.42–2.65) over the packed matcher it replaced.
- **FAST: the ordering, then the record.** A single-block sort became count → prefix-sum → emit,
  so no two corners are ever compared; the corner record is 12 bytes because an arc length
  cannot leave [1, 16], proved by sweeping all 65,536 ring patterns at all sixteen arc lengths.
- **`goodFeaturesToTrack`: the selection never leaves the device.** The ordering key is the
  corner itself — `key = (~responseBits)<<32 | (0xFFFF−y)<<16 | (0xFFFF−x)`, whose ascending
  `uint64_t` order is the host's `CornerStronger` — so eight bytes replace sixteen with no
  payload array.
- **`threshold` is header-only over the host's own cutoff composed with `cuda::packBits`**,
  whose row-grid launch shape removed two software divides and whose warp stores eight
  consecutive words as one sector; both arms sit on the launch floor at 752×480, hence the null.
- **Descriptor matching: kernel shape, not popcount width.** Word emission was once predicted to
  give a ~4× instruction advantage over `cv::cuda`'s byte popcounts; measured on OpenCV's own
  kernel over identical bytes it buys parity, because that kernel is `__syncthreads()`-bound.
  The lead is two barriers per launch against one per descriptor chunk per train block.
- **Lucas–Kanade: the launch shape, while the kernel itself is a loss.** Profiled with both sides
  in one run at 61 keypoints, binCV's one launch is 65.4 µs against `cv::cuda`'s four launches
  summing to 41.9 µs — a kernel loss inside a wall-clock win, which is why the lead ends between
  512 and 1024 keypoints.
- **The batched covariance is a signature, not a kernel.** `countCovarianceBatchAsync` takes N
  windows in one launch where the single-region form pays a launch per window against
  nanoseconds of work; both compute identical counts (`cuda_derivcov_benchmark`).

## What is not delivered

- **`cornerSubPixAsync` below roughly 450 corners: refine on the host.** Measured as one paired
  comparison against the whole-plane download-and-refine round trip in
  `cuda_feature_tracking_benchmark` (no committed log), the device arm reads 0.5656 ms against
  0.3911 ms, 72 of 77 rounds favouring the round trip. At 200 corners the kernel is 6.25 warps
  of work on a part that holds 2,304, and a warp per corner is what `subpix.hpp`'s bit-exactness
  argument forbids, since double addition is not associative.
- **The census entry is larger than `cv::cuda::StereoBM`** (0.681×) and faster (1.50×) on the
  same run: published as both numbers, above.
- **Block matching is not shipped.** It has no `cv::cuda` counterpart, no accuracy bar has been
  stated for it, and it does not ship until one is; binCV's own figure is 0.05701 ms over 7
  processes, not compared.
- **Sparse stereo ships for callers with a richer packing than a global threshold.** Coarse
  search plus one-bit window refinement reads 0.01981 ms (range over 7 runs: 0.01869–0.02432)
  against 0.06400 ms for the dense binary map it replaces — unpaired: two binaries — at 136.9
  KiB of allocations against the dense entry's 442.5, no scratch. On the committed synthetic
  pair of known disparity 21, 500 of 500 matches land within half a pixel (mean 0.0145 px). On a
  frame packed by one global threshold, a keypoint whose one-bit window is uniform scores every
  disparity equally and the tie rule pins it to the scan's low edge — a run on a real
  thresholded frame put about half of 500 keypoints there (log not committed). That is the host
  algorithm's behaviour; the answer is the N-bit or census packing, not this kernel.
- **Lucas–Kanade leads only up to about 512 keypoints**, and the win is the launch shape rather
  than the kernel. It ships for the density its header names; its direction is established by
  the rounds and its magnitude is published as a range ([methodology-timing.md](methodology-timing.md)).
- **The gated matcher is not faster than brute force**: 0.814× on synthetic frames (range over 7
  runs: 0.704–0.894), a null against a 6.24× bar. It ships for the caller who already has the
  gate, not on that number.
- **Device occupancy was dropped on a measurement.** `markOccupiedBatch` / `occupiedBatch` /
  `clearOccupancy` have no equivalent in OpenCV, and the best existing option is the host
  library's own `spaceCandidates` at 3,333 ns and zero bytes — below the launch floor.
- **The geometry stays on the host.** Five-point RANSAC on device (`cuda_ransac_benchmark`, no
  committed log) lost by at least 15.9× — median 18.58× over seven runs, range 15.88–19.54,
  ranges disjoint. The binding constraint is the solver's 6,800-byte per-thread local frame, not
  FP64 rate, so a faster-FP64 part would not obviously change the answer.

## Operations with no `cv::cuda` counterpart

These have no OpenCV counterpart at any API level, so no speed comparison is possible and none
was invented; they ship on correctness, memory and the host comparison, and no CPU number is
quoted in place of a missing GPU one:

- **the gradient covariance** (`gradientCovarianceAsync`, `gradientCovarianceBatchAsync`) —
  `cornerHarris` and `createMinEigenValCorner` compute a dense float response *through* a
  covariance; neither exposes one. Scratch is 0 B across 200 batch launches.
- **orientation** — no `cv::cuda` entry point orients provided keypoints.
- **`keypointsFromCorners`** — the detector-to-keypoint-set link a resident pipeline needs and a
  host pipeline does not: 1.00 synchronize per frame against the round-trip arm's 2.00, and
  0.0043 ms against 0.1547 ms, frame totals disjoint in all 7 runs
  (`cuda_feature_tracking_benchmark`, no committed log).
- **`shift`**, **`binarize`** and the 16-bit **`medianWide`**, each measured above or against
  binCV's own arm.
- **`stereoDescriptorMatch`, `stereoRefineDisparity`, `stereoMatchRectified`** — OpenCV's sparse
  stereo is `StereoBM`'s dense map plus a host lookup; priced against binCV's own dense path.
- **`matchDescriptorsGated`** — no library exposes a gated matcher.
- **`calcOpticalFlowBlockMatch`** — pyramidal tracking by integer Hamming block matching. OpenCV
  ships no sparse block matcher: `cv::cuda::FastOpticalFlowBM` is a dense field, not a tracker,
  and `SparsePyrLKOpticalFlow` solves a different equation and is already the counterpart of
  `calcOpticalFlowPyrLKAsync`, which solves the same one. Where there is no matching algorithm
  there is no comparison, so the table gives binCV's figure alone.

## The assembled pipelines

Not operations, and with no `cv::cuda` counterpart: binCV's own example pipelines, built from
the operations above and measured against binCV's own host library on the same machine, to
show that the ops compose bit-exactly and that residency is where launch cost gets amortized.
Neither has a committed log.

**The resident VIO frontend** (`backends/cuda/examples/cuda_vio_frontend.cpp`): sensor stage →
pyramid → derivatives → `goodFeaturesToTrack` → `keypointsFromCorners` → orientation → BRIEF,
over 400 real EuRoC V1_02 cam0 frames, 7 process runs, one explicit stream, CUDA events on the
device side and the host library's own clock on the host side: 0.945 ms/frame (range over 7
runs: 0.918–1.065) against the host arm's 5.313 ms (range: 5.198–5.422) — 5.62×, ranges
disjoint in all 7 runs and globally. Detection is 75.2% of the device frame; exactly 1.00
synchronize per frame in all 7 runs, 360,960 B up against 11,536 B down; peak device memory
2,106,180 B (allocation sum); over 400 frames × 7 runs, 0 corner-count, position, keep-byte,
rotation-bin or descriptor-word differences against the host.

**The tracking sequence** (`cuda_feature_tracking_benchmark`): the same 400 frames through the
sensor stage, the shipped 1/2/2/2 ladder, the previous frame's derivatives and LK, at a fixed
re-detection cadence. Wall clock on both arms, with the H2D upload and one stream synchronize
per frame inside the device arm's clock; agreement (level-0 frame, derivative planes, 204
corner positions) is checked before any timing in every run.

| cadence | host binCV (ms/frame) | device binCV (ms/frame) | host ÷ device |
|---|---|---|---|
| re-detect every 10 frames | 1.932 (range over 7 runs: 1.779–1.992) | 0.424 (range: 0.383–0.471) | 4.56× |
| re-detect every frame | 6.455 (range: 6.388–6.652) | 1.056 (range: 0.974–1.159) | 6.11× |

Ranges disjoint in all 7 runs and globally on both rows. Peak device memory, identical in all
7 runs because every allocation is made at construction: 2212.7 KiB for the whole resident
state and 451.7 KiB for the tracker alone. The host x86-64 arm is not timing-grade, so these
rows locate a magnitude rather than a decimal.

## Reproduce

```bash
cmake -S . -B build -DBINCV_CUDA=ON -DBINCV_BUILD_BENCHMARKS=ON \
      -DBINCV_CUDA_OPENCV_DIR=<prefix of an OpenCV built with its cuda* modules>
cmake --build build -j
scripts/run_cuda_launches.sh -n 7 ./build/backends/cuda/benchmark/cuda_role_benchmark
python3 scripts/aggregate_cuda_runs.py cuda_role_benchmark-cuda-launches.log
```

The reference figures were built with CUDA 11.1's nvcc and g++-9 as host compiler; set
`CMAKE_CUDA_COMPILER` and `CMAKE_CUDA_HOST_COMPILER` if yours differ, and
`CMAKE_CUDA_ARCHITECTURES` (default 86, the reference GPU) for another part. The LK rows need a
real frame sequence: `BINCV_CUDA_ROLE_FRAMES` pointed at an 8-bit BSQ1 blob
(`scripts/make_sequence_blob.py`). No packaged OpenCV ships the `cuda*` modules; a benchmark
that wants one is always built and, without it, prints each comparison as `BLOCKED` (OpenCV
built without the module) or `OUTSTANDING` (no counterpart exists) — harness tokens, not grades.

| table | binary |
|---|---|
| speed, memory and the families | `cuda_role_benchmark` — one process, one launch floor, every pair interleaved on one explicit stream, `cudaMemGetInfo` on both sides of every memory figure |
| block matching, sparse stereo, the gated matcher | `cuda_sparse_benchmark` |
| the dense optimization ladder; the plane-layout census intermediate | `cuda_dense_benchmark` |
| the families' own arms and off-switches | `cuda_sensor_benchmark`, `cuda_window_benchmark`, `cuda_median_benchmark`, `cuda_pyramid_benchmark` |
| the assembled pipelines, `cornerSubPixAsync`, `keypointsFromCorners`; the geometry | `cuda_feature_tracking_benchmark`, `examples/cuda_vio_frontend`; `cuda_ransac_benchmark` |

**Timing and profiling never mix.** `ncu` serializes and replays kernels and locks clocks to
base, so no timing number here comes from a profiled run and no profile reading is quoted as a
duration.

## Logs

Three sweeps are committed under [logs/](logs/): `cuda_role-x86_64-cuda-launches.log` (taken
at `7c8055f`), `cuda_role_lk-x86_64-cuda-launches.log` (`06246c2`) and
`cuda_sparse-x86_64-cuda-launches.log` (`a608102`), all on `main` as `0d6e302`, each written by
`scripts/run_cuda_launches.sh` and read by `scripts/check_figure_staleness.py`. Every speed and
memory table above is read from them. The assembled-pipeline, `cornerSubPixAsync`,
`keypointsFromCorners` and RANSAC figures have no committed log and are not gated; each names
its binary. Logs marked stale in [expected-stale.txt](logs/expected-stale.txt) were taken before
a bit-identical change to the code beneath them; the file records which files moved and why the
figure stands.

## Changes to published figures

| date | row | previous | current | why |
|---|---|---|---|---|
| 2026-09-28 | every ratio | four significant figures (11.16×, 1.504×, 6.015×, 2.597×, 6.857×) | three (11.2×, 1.50×, 6.02×, 2.60×, 6.86×), with the range over 7 runs | precision rule; no result moved |
| 2026-09-28 | noise bar, every speed row | the widest within-run swing among the 7 runs (3.46× on the binary row) | the median within-run swing, as `scripts/aggregate_cuda_runs.py` computes it (2.28×) | reproducible from the committed log; no result moved |
| 2026-09-28 | memory: `computeBriefSteered`, `matchDescriptors`, `detectFastAsync` | 2048.0 / 48.0 / 42.67×; 8277.3 / 400.0 / 20.7×; 680.0 / 432.0 / 1.574× | 2048.0 / 44.0 / 46.5×; 8192.0 / 368.0 / 22.3×; 768.0 / 432.0 / 1.78× | re-sourced to the committed sweep; the earlier sweep's log was not committed |
| 2026-09-28 | memory: census entry | "`cv::cuda` smaller, by 1.47×" | 0.681× | every ratio column is `cv::cuda` ÷ binCV |
| 2026-09-28 | the families table | family-benchmark sweeps, 9 runs, no committed log (edge 8.1× / 11.1×; erode 11.6–11.9× / 13.0–13.5× / 12.8–13.9× / 22.9×; open 15.1–16.1×; median 235–249× / 634–740×; denoise 4.6–5.0×; pyramid 1.02× / 0.95× and shift 0.948–0.985× as binCV ÷ `cv::cuda`) | the committed role sweep (7.53× / 11.1×; 10.8× / 12.4× / 12.5× / 23.5×; 12.2× / 13.3×; 256× / 593×; 5.30×; 0.992× / 1.09×; 1.01×) | re-sourced; no result moved |
| 2026-09-28 | default-stream surcharge; launch floor | 6.91× / 6.17× / 7.18× / 1.45× / 1.03× from one uncommitted run; 8.66–9.58 µs and 11–13 µs | 6.21× / 6.15× / 5.25× / 1.41× / 1.01× as medians over 7 runs; 6.12 µs (5.77–6.60) | the committed sweeps' own readings |
| 2026-09-28 | sparse stereo | 0.019 ms against 0.39 ms at 135.6 KiB; 236 of 500 within half a pixel, 261 at the scan edge, mean 2.11 px | 0.01981 ms against 0.06400 ms at 136.9 KiB; 500 of 500 within half a pixel, mean 0.0145 px | 135.6 was the block matcher's footprint; the dense figure was re-taken; the committed sweep ran a synthetic pair |
| 2026-09-28 | the gated matcher | 1.11× at an admitted fraction of 4.73% | 0.814× (0.704–0.894), null | re-sourced to the committed sweep |
| 2026-09-23 | `cornerMinEigenValAsync` | a null at 0.0590 ms | 2.60× | the response's FP64 square root memoized over its 100-cell integer domain |
| 2026-09-23 | `detectFastAsync`, `matchDescriptors`, census matcher, `denseDisparityBinary`, `computeBriefSteered`; the `threshold` rows and the census pair | 6.01×, 9.1×, 1.87×, 11.0×, 9.3×; earlier sweeps | 6.02×, 9.51×, 1.89×, 11.2×, 8.82×; re-taken | re-taken in the committed sweep — the first five with no code change, inside run-to-run scatter; the rest because their kernels moved |
| before 2026-09-23 | `threshold` "7.35× faster"; census working set "1.7× smaller"; `edgeThreshold` 37.8–49.9×; pyramid 5.61×; morphology 17.4× / 18.6× / 16.3×; `cornerSubPixAsync` 1.07× in the device arm's favour | not republished | the rows above | default-stream artefacts, a memory reading that was not both sides on one meter, or three separately timed medians in place of one paired comparison |
