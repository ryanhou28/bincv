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
| `deviceBinMat.hpp` | `DeviceBinMat`, `DeviceImage<T>` — owning device containers, value semantics |
| `transfer.hpp` | `upload` / `download` (any host word width), `uploadImage` / `downloadImage` |
| `logic.hpp` | `bitwiseAnd` / `Or` / `Xor` / `Not` |
| `reduce.hpp` | `countNonZero` (whole view and clipped region), async and sync forms |
| `pack.hpp` | `packBits` — the sensor stage, `uint8_t` and `uint16_t` sources |
| `census.hpp` | `censusTransform` — wide image to a K-plane block |
| `denseDisparity.hpp` | `denseDisparityBinary`, `denseDisparityCensus` |

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
