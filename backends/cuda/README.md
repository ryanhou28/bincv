# binCV CUDA backend

A GPU backend that **shares binCV's representation and forks its kernels**. See
[ARCHITECTURE §8.5](../../docs/ARCHITECTURE.md) for the design decision, and for the
measurements: [docs/reports/cuda.md](../../docs/reports/cuda.md) for the results,
[cuda-evidence.md](../../docs/reports/cuda-evidence.md) for how each one was earned, and
[methodology-timing.md](../../docs/reports/methodology-timing.md) for how a speed
difference is decided.

The one-line version: a device bit-plane is byte-identical to a host one, so
upload/download is a raw copy and every device kernel is proven bit-exact
against the host library. Memory location is visible in the type
(`bincv::cuda::DeviceBinMatView` is not `BinMatView`), so no call hides where
its data lives. The device word type is `uint32_t` only — a CUDA core is a
32-bit machine, and `__ballot_sync` packs the format's own 32-pixel word in one
instruction.

## What it provides

All in namespace `bincv::cuda`, all taking device views plus an optional stream,
none allocating inside a kernel:

| header | operations |
|---|---|
| `core.hpp` | the views: `DeviceBinMatView`, `DeviceImageView<T>`, and `DevicePlaneBlockView` — N bit-planes in one allocation, plane `p` at rows `[p*H, (p+1)*H)`, which is `QuantMat<N>`'s own layout |
| `deviceBinMat.hpp` | `DeviceBinMat`, `DeviceImage<T>`, `DeviceArray<T>` — owning device containers, value semantics |
| `features.hpp` | `DeviceKeypointSetConstView`, `DeviceDescriptorSetView` / `ConstView`, and the result PODs `DeviceCorner`, `DeviceFastCorner`, `DeviceDescriptorMatch`, `DeviceStereoMatch`, each with `toHost` |
| `compaction.hpp` | `DeviceAppendBufferView<T>`, `DeviceAppendCounter`, `DeviceAppendResult` — the capacity contract for kernels that emit a variable number of things |
| `transfer.hpp` | `upload` / `download` (any host word width), `uploadImage` / `downloadImage` |
| `logic.hpp` | `bitwiseAnd` / `Or` / `Xor` / `Not` |
| `reduce.hpp` | `countNonZero`, `countAnd`, `countAndSplit`, `countCovariance` (both selector forms), and **`countCovarianceBatchAsync`** — N windows in one launch |
| `pack.hpp` | `packBits`, `packRows`, `packQuant` (N-bit), `unpackTo8Bit` |
| `packCustom.cuh` | `packBitsIf`, `packQuantWith` — arbitrary device predicates; **requires an nvcc-compiled caller** |
| `census.hpp` | `censusTransform` (K-plane block, the host's layout) and `censusTransformPacked` (one descriptor word per pixel) |
| `denseDisparity.hpp` | `denseDisparityBinary`, `denseDisparityCensusPacked` (the fast wide-input path), `denseDisparityCensus` (plane block) |
| `threshold.hpp` | `threshold` (Tier 1, header-only over the host's own cutoff plus `packBits` — no kernel of its own) and `binarize` (N planes → bits, one launch, plane count 1…32) |
| `edge.hpp` | `edgeThreshold` — gradient magnitude straight into bits, uint8 and uint16, with a byte-lane arm behind `impl::edgeVectorEnabled()` |
| `morphology.hpp` | `erode`, `dilate`, `morphologyEx` (all seven `MorphOp`), `toDeviceElement`, and three arms behind `morphFastArmEnabled()` / `morphWordBorderEnabled()` / `morphAndNotFusedEnabled()` |
| `median.hpp` | `denoiseMedian3` over packed bits, and `medianWide<K, T>` over a wide image for K ∈ {1,3,5,7,9} at uint8 and uint16, with a fast arm behind `impl::medianWideFastArmEnabled()` |
| `pyramid.hpp` | `pyrDownBox`, `DevicePyramid<LevelBits...>` (the whole ladder in **one** allocation), `buildPyramidBox`, `pyrFastArmCovers`, and a bit-sliced arm behind `impl::pyrBitSlicedEnabled()` |
| `shift.hpp` | `shift` plus `shiftLeft`/`Right`/`Up`/`Down`, with a `__funnelshift` arm behind `impl::shiftFunnelEnabled()` |
| `derivative.hpp` | `derivativeX`, `derivativeY`, and a fused `derivativeXY` behind its own switch — ternary derivatives over N ∈ [1,4] plane blocks |
| `covariance.hpp` | `gradientCovarianceAsync` and `gradientCovarianceBatchAsync` (one block per window, N windows in one launch), `DeviceGradientCovariance` with `toHost` |
| `corner.hpp` | `cornerMinEigenValAsync` (window and bit-sliced arms), `goodFeaturesToTrackAsync` (fused-tile and frame-map arms), `DeviceCornerResult`, `goodFeaturesScratchBytes` |
| `fast.hpp` | `detectFastAsync` (reference and tiled arms, two scoring arms, warp-aggregated append), `fastScratchBytes` |
| `orientation.hpp` | intensity-centroid orientation over a wide image (three arms) and over a bit plane (two arms), `DiscPod` |
| `descriptor.hpp` | `computeBrief` / `computeBriefSteered` (uint8 and uint16), `DeviceBriefPattern`, `uploadBriefPattern`, with a `__ballot_sync` arm |
| `subpix.hpp` | `cornerSubPixAsync`, `DeviceSubPixMask`, `DeviceSubPixResult` — skip and dense arms |
| `opticalFlow.hpp` | `calcOpticalFlowPyrLKAsync` — the whole pyramid ladder in one launch, one warp per keypoint, the level loop inside the kernel |
| `sparseMatch.hpp` | `matchDescriptors`, `matchDescriptorsGated`, `stereoDescriptorMatch`, `stereoRefineDisparity`, `stereoMatchRectified`, `calcOpticalFlowBlockMatch`. **No host twin for the matcher**: a resident pipeline needs it and a host pipeline does not |
| `sparseMatch.cuh` | the device-side matcher internals, for an nvcc-compiled caller |
| `denseCensusBox.hpp` | `denseDisparityCensusBoxPacked` — the warp-cooperative separable box matcher, a second packed matcher beside the first |
| `keypoints.hpp` | `keypointsFromCorners` — the detector-to-keypoint-set link. **No host twin**: it exists so a resident pipeline needs no mid-frame synchronize |

Twenty-two of binCV's twenty-seven host operation headers have device arms, plus
`keypoints.hpp` and `sparseMatch.hpp`'s descriptor matcher, which have no host twin.
**The geometry does not, and that is an answer rather than a backlog**: five-point RANSAC
was measured on device and lost by at least 15.9× — the median of its seven runs is
18.58× — so nothing shipped. The remaining work stays filed as issues.

Matching used to be listed here as the notable absence, on the prediction that it was
where the format's word utilisation would pay: these descriptors come out as `uint32_t`
words and a matcher issues 8 `__popc` per 256-bit descriptor where `cv::cuda`'s `uchar`
path issues 32. **It now has a device arm, and that prediction was measured and is
wrong** — the instruction ratio is real and buys 1.02×, because the kernel is
`__syncthreads()`-bound rather than math-bound. The lead is kernel shape, and it is in
the table below.

**Several device ops accept a narrower domain than their host twin**, by the
rule that a device op may do so provided the docstring names the domain, the op
asserts it, and it returns an error outside it. Morphology takes elements up to
32 rows × 512 columns (32 columns masked); `medianWide` takes K ∈ {1,3,5,7,9}
as a compile-time parameter; `pyrDownBox` takes 1–8 planes a side; `binarize`
takes 1–32; the derivative takes N ∈ [1,4]; steered BRIEF names its angle domain
[-2π, 2π]. Outside its domain each returns `cudaErrorInvalidValue` **without
launching**, and a Tier 1 claim on one of them is a claim over that domain,
which the docstring says where it makes the claim. The Debug gate configuration
compiles those assertions, so they are built as well as written.

## Where it stands against `cv::cuda`

Both sides' measured values, taken from
[the role bars](../../docs/reports/cuda.md#the-role-bars) and, for the dense binary row's
memory, [the headline](../../docs/reports/cuda.md#the-headline). Speed and memory are
separate tables because they are separate questions.

### Speed

Kernel-resident clock (CUDA events), both arms on **one explicit stream**, medians of 7
independent process runs, 752×480 unless the row names a geometry. Time in milliseconds,
so the smaller cell is the faster side, and the faster side is bold.

<!-- figure-check values="cv::cuda (ms)|binCV (ms)" source="@docs/reports/cuda.md" -->
| operation | `cv::cuda` arm | cv::cuda (ms) | binCV (ms) | disjoint |
|---|---|---|---|---|
| `threshold` → bits, 3840×2160 | `cv::cuda::threshold` | 0.0362 | **0.0268** | 1/7 |
| `detectFastAsync` | `FastFeatureDetector` | 0.1459 | **0.0247** | 7/7 |
| `goodFeaturesToTrackAsync`, wall clock | `createGoodFeaturesToTrackDetector` | 3.7282 | **0.8405** | 7/7 |
| `computeBrief`, N=1000 | `cv::cuda::ORB::computeAsync` | 0.1070 | **0.0107** | 7/7 |
| `matchDescriptors`, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 1.9491 | **0.2189** | 7/7 |
| `calcOpticalFlowPyrLKAsync`, 204 pts | `SparsePyrLKOpticalFlow` | 0.1475 | **0.0792** | 7/7 |
| `calcOpticalFlowPyrLKAsync`, 2048 pts | ″ | **0.3287** | 0.3558 | 0/7 |
| `cornerMinEigenValAsync` | `createMinEigenValCorner` | **0.0515** | 0.0590 | 0/7 |
| `denseDisparityBinary` | `cv::cuda::StereoBM(64, 9)` | 0.7152 | **0.0648** | 7/7 |
| census entry (transform ×2 + match) | ″ | 0.6996 | **0.5076** | 7/7 |
| `calcOpticalFlowBlockMatch` | `SparsePyrLKOpticalFlow` | 0.2320 | **0.0540** | 7/7 |

Two of those rows are **null results rather than losses**, and the difference matters:
Lucas-Kanade at 2048 points and `cornerMinEigenValAsync` each have 0 of 7 runs disjoint,
so neither direction is established — binCV's cell being the larger one is not a finding.
LK at 2048 points is the crossover the tracker's header names; the lead holds to about
512 keypoints and stops there.

### Memory

Peak working set at 752×480, `cudaMemGetInfo` delta taken identically on both sides — the
only meter readable across libraries. Kilobytes, so the smaller cell is the lighter side.

<!-- figure-check values="cv::cuda (KB)|binCV (KB)|which is smaller" source="@docs/reports/cuda.md" -->
| operation | `cv::cuda` arm | cv::cuda (KB) | binCV (KB) | which is smaller |
|---|---|---|---|---|
| `threshold` | `cv::cuda::threshold` | 1024.0 | **416.0** | binCV, by **2.46×** |
| `detectFastAsync` @ capacity 32,768 | `FastFeatureDetector` | 680.0 | **432.0** | binCV, by **1.574×** |
| `goodFeaturesToTrackAsync` | `createGoodFeaturesToTrackDetector` | 10240.0 | **1920.0** | binCV, by 5.33× |
| `computeBrief`, N=1000 | `cv::cuda::ORB::computeAsync` | 2048.0 | **48.0** | binCV, by **42.67×** |
| `matchDescriptors`, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 8277.3 | **400.0** | binCV, by **20.7×** |
| LK tracker resident state | `SparsePyrLKOpticalFlow` | 1408.0 | **448.0** | binCV, by **3.14×** |
| `cornerMinEigenValAsync` response map | `createMinEigenValCorner` | 10240.0 | **2048.0** | binCV, by 5.00× |
| `denseDisparityBinary`, per frame | `cv::cuda::StereoBM(64, 9)` | 3072.0 | **448.0** | binCV, by **6.857×** |
| census entry working set, per frame | ″ | **3072.0** | 4512.0 | **`cv::cuda`**, by **1.47×** |

**The census row is the one where binCV is the larger side, and it needs no marker to say
so.** That loss is the algorithm's rather than this implementation's: census expands 8
bits a pixel into a 32-bit descriptor word, so two transformed images are 2,820 KB before
a disparity map exists, where `cv::cuda::StereoBM` matches the 8-bit frames directly. It
is ahead on speed and behind on memory — a split verdict, recorded as one.

## Why those rows read as they do

**`cuda::threshold`'s earlier miss** was 2.22× slower, and it was two software divides in
`packKernel`'s grid-stride addressing; a row-grid shape removes them and a byte-lane shape
reads four pixels per lane through `__vsetgeu4`. It is now never above 1.00× at any
geometry, with the memory result byte-identical.

**`detectFastAsync` was 1.545× *larger*** one round ago, for two reasons that were both
accidents rather than costs: the sizing
function handed every caller the reference arm's scratch, so the shipped arm was
given 512 KB to use 380 bytes of, and the corner record carried the host's 64-bit
score where the value is an arc length that cannot leave [1, 16]. The arm is now
an argument to `fastScratchBytes`, the op refuses rather than writes when the arm
it is about to run needs more than it was handed, and the record is 12 bytes with
`toHost` widening the score back — so what a caller reads back is unchanged, and
the whole attainable score range is swept exhaustively by the suite.

**`cornerSubPixAsync` is the one operation with no row above, because it has no
`cv::cuda` counterpart and therefore no role bar — and it misses the rule it does have.**
That rule asks the device arm to be strictly cheaper than
the round trip it replaces; re-measured as **one paired comparison** on the wall
clock — rather than as three separately-timed medians added together, which
carries all three passes' drift — the whole-plane round trip reads **0.3911 ms** against
the device arm's **0.5656 ms** in the latest seven-process sweep, 72 of 77 rounds the
round trip's way. It is not a miss by a stated margin and it is certainly not a pass.
The profiler says why, and it is **not** the FP64 rate this file used to give as the
reason: `math_pipe_throttle` does not appear in the kernel's stall histogram at all. At
200 corners the kernel is 6.25 warps of work on a part that holds 2,304, and the one
decomposition that would add parallelism — a warp per corner — is the one `subpix.hpp`'s
bit-exactness argument forbids, because double addition is not associative. Below roughly
450 corners a caller should refine on the host.

All of these are documented as they read in
[docs/reports/cuda.md](../../docs/reports/cuda.md) rather than presented as
results.

**A compaction truncates, counts the truth, and cannot pass as complete.** A
detection kernel appends through `DeviceAppendBufferView<T>` and the counter is
never clamped, so `DeviceAppendResult::found()` is the TRUE candidate count —
the capacity a complete re-run needs — even when the buffer overflowed. There is
no neutral count accessor: a caller either asks for the whole answer
(`completeCount`, which refuses and leaves its output untouched on an overflow)
or accepts a partial one (`acceptTruncated`, whose name is then at the call
site), and `downloadAppended` takes the result object so no path to host memory
skips the verdict. **Which** candidates a truncated run keeps is the atomic's
business and is not specified — the host truncates in raster order, so a
truncated device result is not a prefix of the host's and bit-exactness is a
claim about complete runs.

**Reductions are batched, not per-call.** `countCovarianceBatchAsync` takes the
whole window set and issues one launch; measured, that is **467× faster** than
looping the single-region form over 200 keypoints, because a per-window launch
is latency against nanoseconds of work. Anything keypoint-shaped should use it.

**Wide-input stereo should use the packed census path.** A dense matcher reads
a descriptor one pixel at a time across all K comparisons, and in the plane
block those K bits sit in K different arrays — K loads and K popcounts per
pixel pair. `censusTransformPacked` puts a pixel's whole descriptor in one
word, so `denseDisparityCensusPacked` pays one load, one XOR and one `__popc`:
measured **7.75 ms against 0.91 ms, 8.55×**, for an intermediate that goes from
**3,217.5 KB to 3,877.5 KB**. That was the layout step; since the warp-cooperative box
matcher landed on top of the packed layout, the plane form costs **21.7× the matcher
time** rather than 8.55×. Both layouts are bit-exact against the host; the plane form
stays for callers who want the smaller intermediate and can pay the time.

## Requirements

- An NVIDIA GPU and the CUDA toolkit. Developed against CUDA 11.1 (nvcc) with
  g++-9 as the host compiler, targeting SM 8.6.
- The host library (`include/`) — the backend shares its format and contracts.

## Build

```bash
cmake -S ../.. -B build -DBINCV_CUDA=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.1/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=g++-9
cmake --build build --target bincv_cuda -j
```

Target a different GPU with `-DCMAKE_CUDA_ARCHITECTURES=87` (Jetson Orin, for
instance) — no source change. The `BINCV_CUDA` option is off by default, so a
host build is exactly what it was.

## Verify

```bash
./scripts/verify_cuda.sh    # builds -Werror on both halves, runs device-vs-host
```

Exits 77 (not a pass) without a toolkit or a device. It runs **two**
configurations — Release, which also compiles the benchmarks, and Debug, the
only one where `BINCV_ASSERT` reaches nvcc's device pass — and derives its suite
list from `tests/CMakeLists.txt` rather than a hard-coded one, cross-checking
`bincv_add_test_target()` against `add_test()` so neither half can quietly lose
a suite. **Seventeen suites, 194,975 checks in Release and 194,922 in Debug**: every
device kernel compared against the host library byte for byte, every optimized arm held
to the same map as its reference arm in one binary, and every padding invariant asserted
separately so a failure names the right thing. The two counts differ by design — a suite
exercising a narrowed domain can only test the half of that contract its configuration
has, and prints which half it ran. An earlier figure of 38,901 for the eight pre-frontend
suites was **too high by 2,112**, because `BINCV_CHECK_EQ` evaluated its first argument
twice; those suites read 36,844 today and
[cuda.md](../../docs/reports/cuda.md#coverage) has the arithmetic.

There is **no per-suite check-count floor for the CUDA suites**, because
`tests/expected-checks.txt` is read by `verify.sh`, which builds no `.cu`. An
edit that quietly drops half a sweep therefore still passes this gate green.
That is a known gap, not an oversight.

## Benchmark

```bash
cmake -S ../.. -B build -DBINCV_CUDA=ON -DBINCV_BUILD_BENCHMARKS=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.1/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=g++-9
cmake --build build --target cuda_dense_benchmark cuda_foundation_benchmark -j
./build/backends/cuda/benchmark/cuda_dense_benchmark
```

Seven benchmark binaries, each always built: `cuda_dense_benchmark`,
`cuda_foundation_benchmark`, `cuda_sensor_benchmark`, `cuda_window_benchmark`,
`cuda_median_benchmark`, `cuda_pyramid_benchmark` and `cuda_role_benchmark`.
Every arm prints its kernel-resident and/or end-to-end time next to the host
library's CPU arm on the same frame, with the measured launch floor beside it
and the vector-arm-on-off ratio the project's rule requires — including a
gate-excluded case that must read ~1.00×, run at a geometry where the
measurement can actually resolve one.

**`cuda_role_benchmark` is where the cross-library claim is made**, once: one
process, one launch floor, every pair interleaved **on one explicit stream**,
and `cudaMemGetInfo` on *both* sides of every memory figure. Running the
comparison on the default stream instead costs OpenCV a full device
synchronization per call and inflates binCV's lead by up to 7× on short
kernels, so the family benchmarks' own OpenCV arms are the development view and
this binary is the one that ships numbers.

The GPU-vs-GPU role comparisons need an OpenCV built with the relevant `cuda*`
modules, which no packaged OpenCV ships; point `-DBINCV_CUDA_OPENCV_DIR=<prefix>`
at such a build. Every target that wants one is **always built** and reports its
role bar as BLOCKED or OUTSTANDING when it is absent, rather than appearing only
when an OpenCV is present — a target that exists conditionally is one the gate
tries to build everywhere. `cuda_stereobm_benchmark` is the single exception and
is hand-excluded by name in `scripts/verify_cuda.sh`.
