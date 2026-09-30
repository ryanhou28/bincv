# binCV CUDA backend

A GPU backend that **shares binCV's representation and forks its kernels**. Design
decision: [ARCHITECTURE §8.5](../../docs/ARCHITECTURE.md). Measurements:
[docs/reports/cuda.md](../../docs/reports/cuda.md).

A device bit-plane is byte-identical to a host one, so upload/download is a raw copy and
every device kernel is proven bit-exact against the host library. Memory location is
visible in the type (`bincv::cuda::DeviceBinMatView` is not `BinMatView`), so no call
hides where its data lives. The device word type is `uint32_t` only — a CUDA core is a
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

### Coverage of the host library

Twenty-two of binCV's twenty-seven host operation headers have a device arm, plus two device
operations with no host twin:

| area | host headers with a device arm |
|---|---|
| reductions and dense stereo | `logic`, `reduce`, `pack`, `census`, `denseDisparity` |
| sensor and window stages | `threshold`, `edge`, `morphology`, `denoise`, `medianWide`, `pyramid`, `shift` |
| feature tracking | `derivative`, `covariance`, `corner`, `fast`, `orientation`, `descriptor`, `subpix` |
| tracking and sparse stereo | `opticalFlow`, `blockMatch`, the sparse half of `stereo` |
| device only, no host header | `keypoints`, `sparseMatch`'s descriptor matcher |

The geometry has no device arm, and that is an answer rather than a backlog: five-point RANSAC
was measured on device (`cuda_ransac_benchmark`) and lost by at least 15.9× — the median of its
seven runs is 18.58× — so nothing shipped.

**Several device ops accept a narrower domain than their host twin**, each named in its
docstring; a Tier 1 claim on one of them is a claim over that domain. Morphology takes elements
up to 32 rows × 512 columns (32 columns masked); `pyrDownBox` takes 1–8 planes a side;
`binarize` takes 1–32; the derivative takes N ∈ [1,4] — outside these, each refuses, returning
`cudaErrorInvalidValue` without launching. So do `packQuant`'s plane count (1–8), keypoint
orientation's radius (1–31), `denseDisparityBinary`'s window area (≤ 255 pixels, the
bit-sliced accumulator's range), `unpackTo8Bit`'s height (≤ 65,535) and the packed census
transform's staged tile (≤ 48 KiB), each with a test that calls it out of domain and checks
the error. `medianWide` takes K ∈ {1,3,5,7,9} as a compile-time parameter. The one bound a
launcher cannot check is steered BRIEF's angle domain ([-2π, 2π]), because the angles are
device-resident: it is asserted in Debug builds, where the kernel traps, and unchecked in
Release. The Debug gate configuration compiles that assertion, so it is built as well as
written.

## Where it stands against `cv::cuda`

Both sides' measured values with the ratio beside them, taken from
[docs/reports/cuda.md](../../docs/reports/cuda.md), which carries the conditions, the
per-row ranges and the caveats. Speed and memory are separate tables because they are
separate questions.

### Speed

**Every `ratio` column below is `cv::cuda` ÷ binCV: above 1× means binCV is ahead, below 1×
means `cv::cuda` is.**

Kernel-resident clock (CUDA events), both arms on **one explicit stream**, medians of 7
independent process runs, 752×480 unless the row names a geometry. Time in milliseconds,
so the smaller cell is the faster side.

<!-- figure-check values="cv::cuda (ms)|binCV (ms)|ratio" source="@docs/reports/cuda.md" -->
| operation | `cv::cuda` arm | cv::cuda (ms) | binCV (ms) | ratio |
|---|---|---|---|---|
| **`denseDisparityBinary`** | `cv::cuda::StereoBM(64, 9)` | 0.7134 | 0.06400 | 11.2× |
| census entry (transform ×2 + match) | ″ | 0.7101 | 0.4789 | 1.50× |
| `detectFastAsync` | `FastFeatureDetector` | 0.1221 | 0.02078 | 6.02× |
| `computeBriefSteered`, N=1000 | `cv::cuda::ORB::computeAsync` | 0.08960 | 0.009353 | 8.82× |
| `matchDescriptors`, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 2.009 | 0.2105 | 9.51× |
| `goodFeaturesToTrackAsync`, wall clock | `createGoodFeaturesToTrackDetector` | 3.403 | 0.4240 | 8.76× |
| `calcOpticalFlowPyrLKAsync`, 204 pts | `SparsePyrLKOpticalFlow` | 0.1748 | 0.08931 | 1.41×–5.86× per round |
| `calcOpticalFlowPyrLKAsync`, 2048 pts | ″ | 0.3769 | 0.3976 | 0.952× — null result |
| `cornerMinEigenValAsync` | `createMinEigenValCorner` | 0.05192 | 0.02007 | 2.60× |
| `threshold` → bits, 3840×2160 | `cv::cuda::threshold` | 0.03642 | 0.02550 | 1.43× — null result |
| `calcOpticalFlowBlockMatch` | none — OpenCV has no sparse block matcher | — | 0.05701 | no comparison possible |

**A null result is not a loss.** Lucas-Kanade at 2048 points and `threshold` each sit inside
this host's noise, so neither direction is established and binCV's cell being the larger one
on the LK row is not a finding. LK at 2048 points is the crossover the tracker's header names:
the lead holds to about 512 keypoints and stops there. Block matching has no `cv::cuda`
counterpart and no accuracy bar stated; the report gives binCV's own figure over 7 processes
and compares it with nothing.

### Memory

Peak working set at 752×480, `cudaMemGetInfo` delta taken identically on both sides — the
only meter readable across libraries. KiB (1024 bytes), so the smaller cell is the lighter side.

<!-- figure-check values="cv::cuda (KiB)|binCV (KiB)|ratio" source="@docs/reports/cuda.md" -->
| operation | `cv::cuda` arm | cv::cuda (KiB) | binCV (KiB) | ratio |
|---|---|---|---|---|
| **`denseDisparityBinary`**, per frame | `cv::cuda::StereoBM(64, 9)` | 3072.0 | 448.0 | 6.86× |
| census entry working set, per frame | ″ | 3072.0 | 4512.0 | 0.681× |
| `computeBriefSteered`, N=1000 | `cv::cuda::ORB::computeAsync` | 2048.0 | 44.0 | 46.5× |
| `matchDescriptors`, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 8192.0 | 368.0 | 22.3× |
| `goodFeaturesToTrackAsync` | `createGoodFeaturesToTrackDetector` | 10240.0 | 1920.0 | 5.33× |
| `cornerMinEigenValAsync` response map | `createMinEigenValCorner` | 10240.0 | 2048.0 | 5.00× |
| LK tracker resident state | `SparsePyrLKOpticalFlow` | 1408.0 | 448.0 | 3.14× |
| `detectFastAsync` at capacity 32,768 | `FastFeatureDetector` | 768.0 | 432.0 | 1.78× |
| `threshold` | `cv::cuda::threshold` | 1024.0 | 416.0 | 2.46× |

**The census row is the one where binCV is the larger side, by 1.47×, and the loss is the
algorithm's rather than this implementation's**: census expands 8 bits a pixel into a
32-bit descriptor word, so two transformed images are 2820.0 KiB before a disparity map
exists, where `cv::cuda::StereoBM` matches the 8-bit frames directly. Ahead on speed and
behind on memory, and the report publishes both numbers.

**`cornerSubPixAsync` has no row above**, because it has no `cv::cuda` counterpart. Its own
test is whether the device arm is cheaper than the whole-plane download-and-refine round trip
it replaces, and below roughly 450 corners it is not: refine on the host there.
[cuda.md](../../docs/reports/cuda.md#what-is-not-delivered) has the measurement.

## Three contracts a caller has to know

**A compaction truncates, counts the truth, and cannot pass as complete.** A detection
kernel appends through `DeviceAppendBufferView<T>` and the counter is never clamped, so
`DeviceAppendResult::found()` is the TRUE candidate count — the capacity a complete re-run
needs — even when the buffer overflowed. There is no neutral count accessor: a caller
either asks for the whole answer (`completeCount`, which refuses and leaves its output
untouched on an overflow) or accepts a partial one (`acceptTruncated`, whose name is then
at the call site), and `downloadAppended` takes the result object so no path to host
memory skips the check. **Which** candidates a truncated run keeps is the atomic's
business and is not specified — the host truncates in raster order, so a truncated device
result is not a prefix of the host's, and bit-exactness is a claim about complete runs.

**Reductions are batched, not per-call.** `countCovarianceBatchAsync` takes the whole
window set and issues one launch; looping the single-region form pays a launch per window
against nanoseconds of work, which `cuda_derivcov_benchmark` prices. Anything keypoint-shaped
should use it.

**Wide-input stereo should use the packed census path.** A dense matcher reads a
descriptor one pixel at a time across all K comparisons, and in the plane block those K
bits sit in K different arrays. `censusTransformPacked` puts a pixel's whole descriptor in
one word, so `denseDisparityCensusPacked` pays one load, one XOR and one `__popc` per
comparison: the plane form costs 21.7× the matcher time (`cuda_dense_benchmark`) for an
intermediate of 3217.5 KiB against the packed layout's 3877.5 KiB. Both layouts are
bit-exact; the plane form stays for callers who want the smaller intermediate.

## Requirements

- An NVIDIA GPU and the CUDA toolkit. The reference build — the one every published figure
  was taken with — used CUDA 11.1's nvcc with g++-9 as the host compiler, on an RTX 3070 Ti
  (SM 8.6). Other toolkit and compiler versions have not been tested here.
- CMake 3.18 or newer (`CMAKE_CUDA_ARCHITECTURES`, `find_package(CUDAToolkit)`).
- The host library (`include/`) — the backend shares its format and contracts.

## Build

```bash
cmake -S ../.. -B build -DBINCV_CUDA=ON \
      -DCMAKE_CUDA_COMPILER=<path to nvcc> \
      -DCMAKE_CUDA_HOST_COMPILER=<host compiler nvcc should drive>
cmake --build build --target bincv_cuda -j
```

`CMAKE_CUDA_ARCHITECTURES` defaults to 86, the reference GPU, and the build emits no PTX for
older parts: on any other GPU set it (`-DCMAKE_CUDA_ARCHITECTURES=87` for a Jetson Orin, 75
for a T4), or every launch fails with `cudaErrorNoKernelImageForDevice`; the configure step
prints which architecture it is building. The `BINCV_CUDA` option is off by default, so a
host build is exactly what it was.

## Verify

```bash
./scripts/verify_cuda.sh    # builds -Werror on both halves, runs device-vs-host
```

Exits 77 (not a pass) without a toolkit or a device. It runs **two** configurations —
Release, which also compiles the benchmarks, and Debug, the only one where `BINCV_ASSERT`
reaches nvcc's device pass — and derives its suite list from `tests/CMakeLists.txt` rather
than a hard-coded one, cross-checking `bincv_add_test_target()` against `add_test()` so
neither half can quietly lose a suite. On another GPU pass the architecture through
`BINCV_CUDA_ARCH=<compute capability>` (`native` on CMake 3.24 or newer). **Eighteen suites,
191,804 checks in Release and 191,744 in Debug**: every device kernel compared against the
host library byte for byte,
every optimized arm held to the same map as its reference arm in one binary. The two counts
differ by design — a deliberate domain violation that trips an assertion can only have its
error return checked where the assertion is compiled out, so those call sites print
`[not run in a checked build]` in Debug (`BINCV_CHECK_EQ_UNLESS_CHECKED`).

The dense-disparity cases run each input on two contents, because a right image that is an
exact shift of the left gives the correct disparity a window cost of zero and lets a broken
halo pass; a mutation battery against the box matcher fires 58–140 failures on each of seven
mutations, naming the three benign ones with the structural reason.

**Every suite's check count is held to a floor, per configuration**, in
[`tests/expected-checks.txt`](tests/expected-checks.txt) — the same contract `verify.sh`
enforces for the host suites, in a separate file because `verify.sh` builds no `.cu`. A
suite that runs fewer checks than its row fails the gate even though every check that did
run passed; raising a floor is `./scripts/verify_cuda.sh --update-checks-baseline` and a
reviewed diff.

## Benchmark

```bash
cmake -S ../.. -B build -DBINCV_CUDA=ON -DBINCV_BUILD_BENCHMARKS=ON \
      -DCMAKE_CUDA_COMPILER=<path to nvcc> \
      -DCMAKE_CUDA_HOST_COMPILER=<host compiler nvcc should drive>
cmake --build build --target cuda_dense_benchmark cuda_foundation_benchmark -j
./build/backends/cuda/benchmark/cuda_dense_benchmark
```

Sixteen benchmark binaries, each always built: `cuda_role_benchmark` (every cross-library
comparison), `cuda_sparse_benchmark`, `cuda_dense_benchmark`, `cuda_census_box_benchmark`,
`cuda_stereobm_benchmark`, `cuda_foundation_benchmark`, `cuda_sensor_benchmark`,
`cuda_window_benchmark`, `cuda_median_benchmark`, `cuda_pyramid_benchmark`,
`cuda_packfast_benchmark`, `cuda_derivcov_benchmark`, `cuda_orientation_descriptor_benchmark`,
`cuda_opticalflow_benchmark`, `cuda_feature_tracking_benchmark` and `cuda_ransac_benchmark`.
Every arm prints its kernel-resident and/or end-to-end time next to the host library's CPU
arm on the same frame, with the measured launch floor beside it and the ratio of each vector
arm switched on against off — including a gate-excluded case that must read ~1.00×, run at a
geometry where the measurement can resolve one, so that a fast path which has silently
stopped running is caught.

**`cuda_role_benchmark` is where the cross-library claim is made**, once: one process, one
launch floor, every pair interleaved **on one explicit stream**, and `cudaMemGetInfo` on
*both* sides of every memory figure. On the default stream OpenCV pays a full device
synchronization per call, which inflates binCV's lead by up to 6.21× on short kernels, so the
family benchmarks' own OpenCV arms are the development view and this binary ships the
numbers. `scripts/run_cuda_launches.sh -n 7 <binary>` runs it as seven separate processes and
`scripts/aggregate_cuda_runs.py` applies the decision rule across them.

The GPU-vs-GPU comparisons need an OpenCV built with the relevant `cuda*` modules, which no
packaged OpenCV ships; point `-DBINCV_CUDA_OPENCV_DIR=<prefix>` at such a build. Every
target that wants one is **always built** and, without it, prints each comparison as
`BLOCKED` (OpenCV was built without the module) or `OUTSTANDING` (no `cv::cuda` counterpart
exists, so no comparison is possible) — harness tokens, not grades — rather than existing
conditionally, because a target that exists conditionally is one the gate tries to build
everywhere. `cuda_stereobm_benchmark` is the single exception; `scripts/verify_cuda.sh` asks
CMake which declared benchmarks the configuration created and prints the ones a guard left
out by name.
