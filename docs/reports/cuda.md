# CUDA backend

The GPU backend, measured against the host library on the same machine and
against `cv::cuda::StereoBM` as the best existing GPU option. Design:
[ARCHITECTURE §8.5](../ARCHITECTURE.md). What the backend is and how to build it:
[backends/cuda/README.md](../../backends/cuda/README.md).

The backend shares binCV's format and forks its kernels, so every number here
sits on top of a bit-exactness result: `scripts/verify_cuda.sh` proves each
device kernel gives the host library's answer byte for byte (527 checks across
two suites), and every optimized arm is held to its own reference arm's map in
the same binary. Speed is what follows once correctness is settled.

**Coverage, stated plainly.** Five host operation headers have complete device
arms — `logic`, `reduce`, `pack`, `census`, `denseDisparity` — which is the
sensor stage, the reductions, and dense stereo end to end. The rest of the
operation set does not, and the remaining work is filed as issues #58–#61
rather than implied here.

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
| **binCV binary entry** | **0.39 ms** | **2.0 MB** | Hamming on packed bits |
| `cv::cuda::StereoBM(64, 9)` | 0.78 ms | 10.0 MB | SAD on prefiltered bytes |
| binCV vs StereoBM | **2.0× faster** | **5× smaller** | role only — different maps |

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
| binary, sliding | kernel | **0.39 ms** | 4.2× over reference |
| binary, reference | kernel | 1.66 ms | — |
| binary, upload + kernel + download | e2e | **0.62 ms** | — |
| binary, host CPU arm (same machine) | cpu | ~19 ms (17% spread) | — |
| census transform (both frames) | kernel | 0.14 ms | 17.4× over reference |
| census matcher (K=24), sliding | kernel | **7.8 ms** | 5.1× over reference |
| census, wide frames up to map down | e2e | 8.4 ms | — |
| census, host CPU path (same machine) | cpu | ~387 ms (27% spread) | — |

The **binary end-to-end round trip — packed pair up, matcher, map down — is
0.62 ms, comfortably under StereoBM's 0.78 ms kernel-resident time.** The
device working set is 442 KB against the 23 MB cost volume the design refuses.

### The census entry is still behind, and by how much

The **census entry** is the wide-input story: upload two 8-bit frames, census
on device, match, download — 8.4 ms end to end, 46× the host census path but
**still about 10× behind `cv::cuda::StereoBM`**, which serves the same caller.
That gap is stated here rather than averaged into the headline, and it is the
one place this backend does not lead its role bar.

The sliding arm below narrowed it from ~15× to ~10×; it did not close it. Two
reasons, and only the first is fundamental:

- **Census compares 24 planes where SAD compares one byte.** A census cost is
  24 bit-comparisons per pixel pair against SAD's single subtract-and-add.
  That is the descriptor's price for illumination invariance, and no kernel
  shape removes it.
- **One pixel per thread wastes the popcount.** The kernel counts a 9-bit
  window with a 32-bit `__popc`, so roughly three quarters of every count is
  idle. The host solves exactly this with bit-sliced arithmetic — 32 pixels per
  word operation — and that shape has not been tried here. It is a real
  avenue, not a floor, and it is recorded as open below rather than attempted.

The binary path is the operating point a binCV pipeline runs, and it leads on
both axes; the census path is for callers who arrive with wide frames and no
bits yet, and today it buys them device residency and memory, not a win on
time.

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
| `packBits` (sensor stage) | 0.010 ms | 0.025 ms | `__ballot_sync` packer |
| `packQuant` N=2 (N-bit ingestion) | 0.015 ms | 0.064 ms | N ballots per 32 pixels |
| `censusTransform` (24 planes) | 0.073 ms | 2.02 ms | shared-memory tile, 17.1× over its own reference |

### The covariance, and the measurement that chose its signature

The gradient covariance is the operation ARCHITECTURE §1's identity turns into
population counts, and the one issue #34 predicted would pay best here —
`__popc` is a single instruction on a GPU where it costs two register-domain
crossings on aarch64. The interesting result is not the arithmetic, though; it
is the **signature**.

200 keypoints, 31×31 windows — the tracker's shape:

| arm | time | |
|---|---|---|
| `countCovarianceBatchAsync` — one launch | **0.008 ms** | |
| per-window loop — 200 launches of the single-region form | 3.667 ms | **the batch is 467× faster** |
| host `countCovariance` ×200 | 0.015 ms | batch is 1.9× the CPU |

Both device forms compute identical counts — integer addition, and the tests
pin the batch against both the single-region form and the host — so the 467×
is *purely* what the signature costs. A per-window launch pays ~5–10 µs of
launch overhead against a window whose work is nanoseconds; batching pays it
once. This is the "ceiling versus signature" question from binCV's own notes,
answered: the cap was the signature's, not the operation's.

Against the CPU the honest figure is **1.9×**, not a headline. 200 windows of
31×31 is 0.015 ms of host work — too little to beat by much once a launch is in
the path at all. The batched form earns its place by being the shape a resident
tracker can use at all, not by winning this microbenchmark; what it will be
judged on is the frontend it is built for (issue #58), where the planes are
already on device and the launch is amortized over the whole frame.

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
| reference kernel (one thread per pixel) | 1.66 ms | 39.3 ms | 1.23 ms |
| shared-memory tiling, 8-wide disparity tiles | 0.58 ms | 11.7 ms | — |
| sliding vertical window, 16-row strips | **0.39 ms** | **7.8 ms** | — |
| shared-memory tile + all-K ballots | — | — | **0.071 ms** |

The last matcher step is the one binCV had already taken on the CPU and not
yet here. Both earlier arms re-evaluated the whole window for every output row,
so the window's height multiplied the cost; the sliding arm pays a row twice
instead of `winHeight` times — the same change the host's own optimization
curve records at ≈4.5× ([stereo.md](stereo.md)). Each thread owns an output
column and a 16-row strip, and the disparity tile is kept, so the two savings
compose. The leaving row is recomputed rather than cached: a cached ring needs
a dynamically indexed register array, which spills to local memory and costs
more than the recomputation saves. Strip length was swept — 8 gave 1.35×, 16
gave 1.53×, and **32 regressed to 0.6× as the register file spilled**.

Recorded negatives (measured, reverted, not to be retried on the same shape):
widening the disparity tile from 8 to 16 was 2.1× slower on the binary matcher
and 2.4× on census (register pressure); tile width 4 and block width 256 were
null against spread.

The reductions took a different lesson. Their kernels were never the problem —
the 467× came from changing what a caller may *ask for*, not from changing how
a window is counted. Two traversals ship for that reason: grid-stride with one
atomic per warp for a single region that may be a whole frame, and one block
per region with no atomics for a batch of windows. The host's
`SlidingWindowCount` is deliberately **not** ported: it exists because
consecutive CPU windows re-read the same words *serially*, and on the device
every window is already its own block, so a sliding traversal would serialize
what is currently parallel.

## What is not measured

- **No Jetson or other device.** These numbers are a claim about the RTX 3070 Ti
  and nothing else, exactly as ARCHITECTURE §8 treats every platform. The design
  accommodates a unified-memory device without a rewrite (ops take views, the
  target SM is a build setting), but that is a design property, not a result.
- **`ncu` counters are unavailable under this WSL2 setup** (the driver returns
  "Unknown Error" on counter access), so these are event-and-wall-clock timings,
  not occupancy or memory-throughput profiles. The stage-by-stage method stood
  in for a profiler: each optimization was measured against the arm it replaced.
- **The census matcher's remaining ~10× has an untried avenue.** The sliding
  reformulation is done (5.1× over the reference arm), and the kernel still
  counts a 9-bit window with a 32-bit `__popc` — three quarters of every count
  idle. Processing 32 pixels per thread word-parallel, the host's bit-sliced
  shape, would use the whole instruction, but it needs bit-sliced adders for
  the window sums and is a different kernel rather than a tuning change. Not
  attempted, and not assumed to win: recorded so the next attempt starts from
  the measurement rather than the guess.
