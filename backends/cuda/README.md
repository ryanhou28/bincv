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

These five host operation headers have **complete** device arms. The rest of
binCV's operation set does not yet — the remaining work is filed as issues
#58 (frontend), #59 (tracking), #60 (per-pixel families) and #61 (sparse stereo
and geometry).

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

Exits 77 (not a pass) without a toolkit or a device. The suite itself
(`tests/test_cuda_backend.cpp`) compares every device kernel against the host
library byte for byte, and holds every optimized arm to the same map as its
reference arm in one binary.

## Benchmark

```bash
cmake -S ../.. -B build -DBINCV_CUDA=ON -DBINCV_BUILD_BENCHMARKS=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.1/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=g++-9
cmake --build build --target cuda_dense_benchmark cuda_foundation_benchmark -j
./build/backends/cuda/benchmark/cuda_dense_benchmark
```

Every arm prints its kernel-resident and/or end-to-end time next to the host
library's CPU arm on the same frame, with the vector-arm-on-off ratio the
project's rule requires. The GPU-vs-GPU role comparison against
`cv::cuda::StereoBM` needs an OpenCV built with the `cudastereo` module; point
`-DBINCV_CUDA_OPENCV_DIR=<prefix>` at it to build `cuda_stereobm_benchmark`.
