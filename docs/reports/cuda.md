# CUDA backend

The GPU backend, measured against the host library on the same machine and
against `cv::cuda::StereoBM` as the best existing GPU option. Design:
[ARCHITECTURE §8.5](../ARCHITECTURE.md). What the backend is and how to build it:
[backends/cuda/README.md](../../backends/cuda/README.md).

The backend shares binCV's format and forks its kernels, so every number here
sits on top of a bit-exactness result: `scripts/verify_cuda.sh` proves each
device kernel gives the host library's answer byte for byte (284 checks), and
every optimized arm is held to its own reference arm's map in the same binary.
Speed is what follows once correctness is settled.

## The headline

**The binary dense-disparity path leads `cv::cuda::StereoBM` on both axes at
once** — faster *and* lighter, GPU against GPU, the same result the host binary
path earned against CPU StereoBM. This is the operating point a binCV pipeline
runs: it already holds packed bits, and on bits the dense cost is one XOR per
32-pixel word.

752×480, 64 disparities, 9×9 support, resident on the device, kernel-resident
time (CUDA events, medians):

| | time | device memory | role |
|---|---|---|---|
| **binCV binary entry** | **0.58 ms** | **2.0 MB** | Hamming on packed bits |
| `cv::cuda::StereoBM(64, 9)` | 0.77 ms | 10.0 MB | SAD on prefiltered bytes |
| binCV vs StereoBM | **1.33× faster** | **5× smaller** | role only — different maps |

Role only: the two match different costs and produce different maps;
correctness is settled against the host library, not against StereoBM. The
memory figures are `cudaMemGetInfo` deltas around each side's working-set
allocation, measured identically on both sides (GpuMat may pool, so StereoBM's
is an upper reading).

## Setup

- **Device:** NVIDIA GeForce RTX 3070 Ti (SM 8.6, 48 SMs, ~608 GB/s), under
  WSL2. CUDA 11.1 nvcc, g++-9 host compiler.
- **Two clocks, each labelled at its number.** *Kernel-resident* is CUDA events
  around the enqueued work — the per-frame cost once data lives on device, the
  number a resident pipeline pays. *End-to-end* is the host clock around
  upload + kernels + download + synchronize — the cost when the GPU does only
  this for the frame. WSL2 inflates launch overhead specifically, so
  kernel-resident numbers travel better; both are honest here and the spread is
  printed beside each.
- **CPU arm** is the host library's own best on this same machine, measured by
  the project's interleaved protocol. WSL2 CPU timing carries large spread
  (recorded in the memory notes); it is a same-machine reference, not a
  cross-device claim, and the Pi remains the timing-grade CPU number
  ([stereo.md](stereo.md)).
- Reproduce: `cuda_dense_benchmark`, `cuda_foundation_benchmark`, and (with a
  cudastereo-enabled OpenCV) `cuda_stereobm_benchmark`.

## Dense disparity, both entries

752×480, D=64, 9×9. Each GPU arm is shown with its reference arm from the same
binary through the runtime switch — the ratio is the project's "is the fast arm
actually running" check.

| arm | clock | time | vs its reference |
|---|---|---|---|
| binary, tiled | kernel | **0.58 ms** | 2.9× over reference |
| binary, reference | kernel | 1.70 ms | — |
| binary, upload + kernel + download | e2e | **0.74 ms** | — |
| binary, host CPU arm (same machine) | cpu | ~15 ms (48% spread) | — |
| census transform (both frames) | kernel | 0.14 ms | 17.4× over reference |
| census matcher (K=24), tiled | kernel | **11.7 ms** | 3.3× over reference |
| census, wide frames up to map down | e2e | 12.0 ms | — |
| census, host CPU path (same machine) | cpu | ~267 ms (24% spread) | — |

The **binary end-to-end round trip — packed pair up, tiled matcher, map down —
is 0.74 ms, under StereoBM's 0.77 ms kernel-resident time.** The device working
set is 442 KB against the 23 MB cost volume the design refuses.

The **census entry** is the wide-input story: upload two 8-bit frames, census
on device, match, download — 12.0 ms end to end, 22× the host census path. Its
matcher is behind StereoBM on time (census matches 24 planes × 9 rows ×
64 disparities of popcount per pixel, ~24× the binary path's arithmetic and
near that floor) and is stated here rather than averaged into the headline. The
binary path is the operating point that wins; the census path is for callers
who arrive with wide frames and no bits yet.

## Foundation and sensor ops

Microbenchmarks — one kernel in a loop, next to the host library's CPU arm on
the same frame. Shares of a real pipeline come from the dense benchmark, not
from these.

| op | GPU kernel | CPU arm | note |
|---|---|---|---|
| upload packed frame (45 KB) | 0.009 ms | — | contiguous fast path |
| upload wide frame (361 KB) | 0.048 ms | — | |
| download wide frame (361 KB) | 0.080 ms | — | |
| `bitwiseAnd` | 0.007 ms | 0.002 ms | memcpy-bound both sides |
| `countNonZero` | 0.014 ms | 0.004 ms | |
| `packBits` (sensor stage) | 0.009 ms | 0.027 ms | `__ballot_sync` packer |
| `censusTransform` (24 planes) | 0.071 ms | 3.03 ms | shared-memory tile, 17.4× over its own reference |

The foundation ops are individually so cheap on both sides that their
microbenchmark ratios are dominated by launch and loop overhead — they are
correctness-and-price-at-birth arms, and their real value is composing into the
resident dense pipeline above, where the transfers are the tax and the kernels
are the work. The sensor-stage measured question — upload wide then pack on
device, vs pack on CPU then upload bits — is close (path A 0.10 ms vs path B
0.06 ms at this size); device-side pack wins whenever the wide frame is already
resident, which is the point of a resident pipeline.

## How the numbers were earned

Every arm shipped correct-first, then optimized with its reference arm kept
reachable and both held to the same map:

| stage | binary matcher | census matcher | census transform |
|---|---|---|---|
| reference kernel (one thread per pixel) | 1.70 ms | 39.2 ms | 1.23 ms |
| shared-memory tiling | **0.58 ms** | **11.7 ms** | — |
| shared-memory tile + all-K ballots | — | — | **0.071 ms** |

Recorded negatives (measured, reverted, not to be retried on the same shape):
widening the disparity tile from 8 to 16 was 2.1× slower on the binary matcher
and 2.4× on census (register pressure); tile width 4 and block width 256 were
null against spread.

## What is not measured

- **No Jetson or other device.** These numbers are a claim about the RTX 3070 Ti
  and nothing else, exactly as ARCHITECTURE §8 treats every platform. The design
  accommodates a unified-memory device without a rewrite (ops take views, the
  target SM is a build setting), but that is a design property, not a result.
- **`ncu` counters are unavailable under this WSL2 setup** (the driver returns
  "Unknown Error" on counter access), so these are event-and-wall-clock timings,
  not occupancy or memory-throughput profiles. The stage-by-stage method stood
  in for a profiler: each optimization was measured against the arm it replaced.
- **The census matcher is near its arithmetic floor, not its roofline.** A
  sliding-window reformulation (the host's approach) could cut the redundant
  popcounts, but it does not map cleanly onto the per-pixel-thread model and is
  recorded as open, not attempted.
