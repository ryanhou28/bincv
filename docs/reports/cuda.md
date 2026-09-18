# CUDA backend

The GPU backend, measured against the host library on the same machine and
against `cv::cuda::StereoBM` as the best existing GPU option. Design:
[ARCHITECTURE §8.5](../ARCHITECTURE.md). What the backend is and how to build it:
[backends/cuda/README.md](../../backends/cuda/README.md).

The backend shares binCV's format and forks its kernels, so every number here
sits on top of a bit-exactness result: `scripts/verify_cuda.sh` proves each
device kernel gives the host library's answer byte for byte (605 checks across
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
| **binCV binary entry** (pair already packed) | **0.069 ms** | **2.0 MB** | Hamming on packed bits |
| binCV census entry (wide frames in) | 1.08 ms | 6.0 MB | Hamming on census descriptors |
| `cv::cuda::StereoBM(64, 9)` | 0.84 ms | 10.0 MB | SAD on prefiltered bytes |
| binary vs StereoBM | **12.2× faster** | **5× smaller** | role only — different maps |
| census vs StereoBM | 1.29× slower | 1.7× smaller | the wide-input comparison |

The **binary entry leads its role bar on both axes**. The **census entry** —
the like-for-like comparison for a caller holding wide 8-bit frames, which is
what StereoBM takes — sits just behind on time at less memory, having started
this work 15× behind.

**Which of these is binCV's claim, and which is the on-ramp.** Only the binary
entry rests on the representation: its caller already holds one bit per pixel,
its cost is an XOR and a population count, and its device memory is 2.0 MB
where StereoBM's is 10 MB on the same meter. The census entry is a **standard
stereo technique implemented in the standard way** — census *expands* data
rather than compressing it (8 bits per pixel in, 24 out), and the layout that
finally made it fast is the conventional one-word-per-pixel descriptor, not
binCV's bit-planes. It exists so a caller arriving with ordinary camera frames
has a way in, and it is competitive; it is not where the thesis pays, and
nothing here should be read as claiming otherwise.

One number makes the distinction concrete. Binary does a twenty-fourth of
census's work, and the host captures that: **17× on x86-64, 7.6× on aarch64**
([stereo.md](stereo.md)). This device kernel captures **13.7×**, inside that
band — but only since the matcher started treating a word as a word. The
per-pixel arm it replaced captured 2.3×, because it spent a 32-bit `__popc` on
a 9-bit window and never used the fact that 32 pixels share a register. The
word-parallel arm below is what collecting the rest looks like.

Role only: the two match different costs and produce different maps;
correctness is settled against the host library, not against StereoBM. The
memory figures are `cudaMemGetInfo` deltas around each side's working-set
allocation, measured identically on both sides (GpuMat may pool, so StereoBM's
is an upper reading).

**One meter per comparison, named at the number.** Two memory meters appear in
this report and they do not mix. A `cudaMemGetInfo` delta measures what the
driver reserves; it is the meter for every figure that crosses libraries,
because it is the only one readable on both sides. An allocation sum — what the
arrays themselves ask for, the figure `cuda_dense_benchmark` prints — measures
binCV against binCV and against the cost volume the design refuses. Crossing
them inflates: binCV's 442 KB of arrays set beside StereoBM's 10 MB reading
would look like 23×, and that ratio answers no question. On this driver
`cudaMemGetInfo` moves in 2 MB steps — a one-byte allocation reads 2.00 MB — so
the binary entry's 2.0 MB is the meter's floor rather than its footprint, which
makes the 5× above a lower bound on the memory lead and not a measurement of
it.

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
| binary, word-parallel | kernel | **0.069 ms** | 25× over reference |
| binary, per-pixel sliding | kernel | 0.408 ms | 4.2× over reference |
| binary, reference | kernel | 1.71 ms | — |
| binary, upload + kernel + download | e2e | **0.28 ms** | — |
| binary, host CPU arm (same machine) | cpu | ~19 ms (17% spread) | — |
| census transform, packed (both frames) | kernel | 0.14 ms | — |
| census matcher (K=24), packed | kernel | **0.94 ms** | 44× over reference |
| census matcher (K=24), plane layout | kernel | 7.78 ms | 5.3× over reference |
| census, wide frames up to map down | e2e | **1.51 ms** | — |
| census, host CPU path (same machine) | cpu | ~309 ms (5% spread) | — |

Every row above comes from one re-measurement session — seven independent runs
of `cuda_dense_benchmark`, medians of medians — so the ratios between rows are
ratios between numbers taken the same afternoon. The census arms did not change
in it; they read a few percent off their previously published values, which is
what this host's run-to-run spread looks like.

The **binary end-to-end round trip — packed pair up, matcher, map down — is
0.28 ms, a third of StereoBM's 0.84 ms kernel-resident time.** Transfers now
dominate that round trip three to one, which is the shape a resident pipeline
exists to remove. The device working set is 442 KB of arrays against the 23 MB
cost volume the design refuses — an allocation sum on both sides, not the
`cudaMemGetInfo` reading the role table uses.

### The census entry, and the layout that closed its gap

The **census entry** is the wide-input story: upload two 8-bit frames, census
on device, match, download — **1.51 ms end to end**, 205× the host census path
and within **1.29×** of `cv::cuda::StereoBM` resident-to-resident, at less
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
| reference kernel (one thread per pixel) | 1.71 ms | 39.3 ms | 1.23 ms |
| shared-memory tiling, 8-wide disparity tiles | 0.58 ms | 11.7 ms | — |
| sliding vertical window, 16-row strips | 0.408 ms | 7.75 ms | — |
| packed descriptor layout | — | **0.94 ms** | — |
| word-parallel bit-slicing, one thread per word | **0.069 ms** | — | — |
| shared-memory tile + all-K ballots | — | — | **0.071 ms** |

The binary matcher's last step is the one that collects the representation's
advantage rather than tuning around it. Every earlier arm mapped one thread to
one output pixel, so each candidate cost a 32-bit `__popc` on a 9-bit window —
a wide instruction doing narrow work, and no use at all of the 32 pixels
sharing the register. The word-parallel arm gives a thread one *word* of
output: the raw cost for 32 pixels is one XOR, the nine-wide horizontal sum is
a carry-save tree into four bit-planes, and the winner-take-all is a borrow
chain and a masked select — the host library's own `planesLess`/`planesSelect`,
ported rather than reinvented. That is **6.0× over the arm it replaced** and 25×
over the reference, at an unchanged 442 KB and zero shared memory.

Three shapes had to be settled by measurement rather than argument. Summing
horizontally *before* vertically reverses the host's order, because on the
device a lane shift crosses into the neighbouring thread's registers and would
cost a warp shuffle per plane per stage; addition commutes, so the map is
identical. Eight disparity chunks per word fold through a warp shuffle, because
word-parallel work is dense enough that the reference frame is otherwise 1,416
threads — under one warp per SM. And the fold compares `(cost, disparity)`
lexicographically in one borrow chain: a cost-only "strictly less wins" is
correct for a linear scan but not for a tree, whose second step already holds
winners from non-adjacent chunks. That last one was a real bug, caught by the
suite on 22 pixels out of 12,000.

At 69 µs the arm is close enough to this host's launch floor that its own
spread runs 24–60% where the arm it replaced runs 7–18%. The figures here are
medians of seven independent runs, each itself a median of nine batches; the
two arms' sample ranges do not overlap in any pairing.

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
- **No hardware profiler was available, and that shaped the method.** Nsight
  Compute (`ncu` 2024.2.1) fails with `Unknown Error on device 0` — reproduced
  on a three-line kernel with a single basic metric, so it is device-level
  counter access being refused rather than anything about these kernels. The
  usual cause on a GeForce part is NVIDIA's default of restricting GPU
  performance counters to administrators; the fix is Windows-side and needs a
  reboot (NVIDIA Control Panel → Desktop → Enable Developer Settings →
  Developer → *Allow access to the GPU performance counters to all users*, or
  `RmProfilingAdminOnly = 0` under
  `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak`). Nsight
  Systems is no fallback here either: the `nsys` bundled with CUDA 11.1 crashes
  on this glibc (`__libc_dlsym` left the private ABI in 2.35).

  So everything here is CUDA-event timing, `nvcc -Xptxas -v` for the static
  picture (registers, spills, occupancy arithmetic), and controlled A/B against
  the arm each change replaced. That was enough to locate the dense matchers'
  limiter, but it took six experiments where a stall-reason profile would have
  taken one run. **Anyone picking up #62 or #63 should get `ncu` working
  first** — `smsp__pcsamp_warps_issue_stalled_*` answers directly what those six
  experiments had to triangulate.
- **The binary matcher was chased too — six attempts, and together they locate
  the limit.** `ptxas -v` reports 116 registers and no spills for the binary
  kernel, 255 (the ceiling) for the census one. *Not memory-traffic bound:*
  shared staging (1.28× slower), hoisting the disparity tile's right-image
  loads (null), and a `planes == 1` specialization (1.23× slower) all failed —
  a warp's lanes read consecutive anchors, so its loads already coalesce into
  one or two lines. *Not occupancy bound:* packing cost and disparity into one
  register (`(cost << 8) | d` orders exactly as the tie rule needs) cut the
  binary kernel to **84 registers**, enough for another block per SM, and the
  runtime did not move. *Not register-starved:* `__launch_bounds__` at 4, 6 and
  8 blocks/SM made both kernels **worse**, so the compiler's occupancy-for-ILP
  trade is already the right one. What all six located was instructions per
  useful bit — roughly 19 instructions to produce 9 bit-comparisons, because
  `__popc` was handed a 9-bit run in a 32-bit register. That is what the
  word-parallel arm above attacks, and it is the one thing on this kernel that
  worked: the six failures were all attempts to relieve *memory* pressure on a
  kernel that was never memory-bound. Reading them as a set is what pointed at
  arithmetic density; reading any one alone would not have.
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
