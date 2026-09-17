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
| **binCV binary entry** (pair already packed) | **0.39 ms** | **2.0 MB** | Hamming on packed bits |
| binCV census entry (wide frames in) | 1.04 ms | 6.0 MB | Hamming on census descriptors |
| `cv::cuda::StereoBM(64, 9)` | 0.83 ms | 10.0 MB | SAD on prefiltered bytes |
| binary vs StereoBM | **2.1× faster** | **5× smaller** | role only — different maps |
| census vs StereoBM | 1.25× slower | 1.7× smaller | the wide-input comparison |

The **binary entry leads its role bar on both axes**. The **census entry** —
the like-for-like comparison for a caller holding wide 8-bit frames, which is
what StereoBM takes — sits just behind on time at less memory, having started
this work 15× behind.

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
| census transform, packed (both frames) | kernel | 0.12 ms | — |
| census matcher (K=24), packed | kernel | **0.91 ms** | 43× over reference |
| census matcher (K=24), plane layout | kernel | 7.75 ms | 5.1× over reference |
| census, wide frames up to map down | e2e | **1.42 ms** | — |
| census, host CPU path (same machine) | cpu | ~295 ms (38% spread) | — |

The **binary end-to-end round trip — packed pair up, matcher, map down — is
0.62 ms, comfortably under StereoBM's 0.78 ms kernel-resident time.** The
device working set is 442 KB against the 23 MB cost volume the design refuses.

### The census entry, and the layout that closed its gap

The **census entry** is the wide-input story: upload two 8-bit frames, census
on device, match, download — **1.42 ms end to end**, 208× the host census path
and within **1.25×** of `cv::cuda::StereoBM` resident-to-resident, at less
device memory.

It started 15× behind. The last and largest step was not a kernel trick but a
**layout** one, and it is worth stating plainly because the first two guesses
were wrong about where the time went.

The matcher reads a census descriptor **one pixel at a time, across all K
comparisons at once**. In binCV's plane block those K bits live in K *different
arrays*, so one pixel pair cost K loads and K popcounts — and each `__popc`
counted a single useful bit out of 32. At K = 24 and 64 disparities that is
about 4,200 word loads per output pixel: the kernel was load-bound, not
arithmetic-bound.

Packed — a pixel's whole 24-bit descriptor in one `uint32` — the same pixel
pair is one load each, one XOR, one `__popc` with 24 of 32 bits doing useful
work. Measured: **7.75 ms → 0.91 ms, 8.55×**, well past the ~2.7× predicted
from popcount utilization alone, because it fixed the load traffic too.

This is issue #34's anticipated case and its prescribed answer: *add a second
documented layout to the shared core rather than fork, because two independent
definitions mean neither can be checked against the other.* Both layouts come
from the same pattern and the same comparison rule, and the packed matcher's
output map is held byte-equal to the host's wide path by test — Hamming
distance is invariant under a permutation of a descriptor's bits, so the bit
order carries no meaning beyond "both images use one".

**The trade, stated:** packed costs 32 bits per pixel against the plane block's
24, so the census working set goes from ~3.3 MB to ~4.0 MB — a 21% increase for
8.55× on the matcher. Both paths ship and both are bit-exact;
`denseDisparityCensus` (plane block) remains for a caller who wants the smaller
intermediate and can pay the time.

What remains fundamental is unchanged: census compares 24 bits per pixel pair
where SAD compares one byte. That is the descriptor's price for illumination
invariance, and it is why the census entry sits just behind StereoBM rather
than ahead of it, while the binary entry — the operating point a binCV pipeline
runs — leads on both axes.

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
| sliding vertical window, 16-row strips | **0.39 ms** | 7.75 ms | — |
| packed descriptor layout | — | **0.91 ms** | — |
| shared-memory tile + all-K ballots | — | — | **0.071 ms** |

The census matcher is **43× its reference kernel** across those steps, and the
two largest factors came from different places: tiling and sliding are kernel
shape, the last one is data layout. The order is worth noting — two rounds of
kernel tuning bought 5.1× before anyone asked what the memory access pattern
actually was, and the answer (K separate arrays per pixel) was worth another
8.55× on its own.

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
null against spread. The packed matcher's own tile was swept separately — 4 gave
1.97 ms and 16 gave 1.11 ms against 8's 0.91 — so eight is measured there too
rather than inherited.

**The one that looked obvious and lost.** After the packed layout won, the
next step appeared to be sharing the horizontal window: neighbouring threads'
9-pixel windows overlap by 8, so a block loads nearly every word nine times.
Staging each pixel pair's raw cost in shared memory once, for all overlapping
windows to read, measured **1.16 ms against 0.91 — 1.28× slower**, and was
reverted. The redundant loads were already L1 hits, so the staging bought
nothing while costing 624 `__syncthreads()` per block and byte-wide
shared-memory bank conflicts. The lesson generalizes past this kernel: the
packed-layout win came from touching *less distinct memory* (24 arrays down to
one), not from issuing fewer load instructions, and those are not the same
quantity.

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
- **The census entry's remaining 1.25× was chased and did not fall.** The
  shared-memory attempt above lost, and the packed matcher's tile width is at
  its measured optimum, so this kernel is at a local optimum for its shape.
  What remains are genuine rewrites rather than tuning, each with an uncertain
  payoff: marching a thread along x so the horizontal window slides in
  registers (no barriers, unlike the attempt that failed); a two-pass separable
  box filter over a per-disparity cost buffer (~360 KB, textbook, O(1) per
  pixel-disparity, but 64 launches or a fused loop to schedule); and vectorized
  `uint4` loads for the consecutive right-image words. None attempted; the
  measurement, not the guess, is what the next attempt should start from.
- **The plane-layout matcher is kept but is 8.55× slower** than the packed one.
  It ships because it consumes the host's own layout and costs less memory, not
  because it is the fast path; a caller with wide frames should use
  `censusTransformPacked` + `denseDisparityCensusPacked`.
