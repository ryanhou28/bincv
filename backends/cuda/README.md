# binCV CUDA backend

A GPU backend that **shares binCV's representation and forks its kernels**. See
[ARCHITECTURE §8.5](../../docs/ARCHITECTURE.md) for the design decision and
[docs/reports/cuda.md](../../docs/reports/cuda.md) for the measured results.

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
| `keypoints.hpp` | `keypointsFromCorners` — the detector-to-keypoint-set link. **No host twin**: it exists so a resident pipeline needs no mid-frame synchronize |

Nineteen of binCV's twenty-seven host operation headers have device arms, plus
`keypoints.hpp`, which has no host twin. The rest do not yet — **descriptor
matching**, tracking, sparse stereo and the geometry — and the remaining work
stays filed as issues. Matching is the notable absence: it is where the format's
word utilisation would pay, since these descriptors come out as `uint32_t` words
and a matcher issues 8 `__popc` per 256-bit descriptor where `cv::cuda`'s
`uchar` path issues 32.

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

**`cuda::threshold` now leads on both axes** — 0.744× the time of
`cv::cuda::threshold` at 3840×2160 and never above 1.00× at any geometry, at
2.46× less device memory. The earlier miss (2.22× slower) was two software
divides in `packKernel`'s grid-stride addressing; a row-grid shape removes them
and a byte-lane shape reads four pixels per lane through `__vsetgeu4`.

**Two operations do NOT lead on both axes, and they are the ones to read the
report about before using.** `detectFastAsync` misses its role bar by 16.8× on
real frontend content and is 1.55× larger on the memory meter — its corner set is
provably the host's, but 98.6% of its time is a single-block raster sort.
`cornerSubPixAsync` misses its own round-trip rule: downloading the derivative
planes and refining on the host is 0.53 ms against 1.91 ms resident, because
bit-exactness forces `double` on a part that runs FP64 at 1/64 rate. And the
**resident frontend as a whole is 1.22× slower than binCV's own CPU frontend**,
for one reason: `selectKernel` runs in one block at 1.29% of the SMs. All three
are documented as misses in
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
measured **8.55× faster** (7.75 → 0.91 ms) for 21% more intermediate memory.
Both layouts are bit-exact against the host; the plane form stays for callers
who want the smaller intermediate.

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
a suite. **Eight suites, 38,901 checks**: every device kernel compared against
the host library byte for byte, every optimized arm held to the same map as its
reference arm in one binary, and every padding invariant asserted separately so
a failure names the right thing.

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
