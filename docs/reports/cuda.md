# CUDA backend

The GPU backend, measured against the host library on the same machine and
against `cv::cuda::StereoBM` as the best existing GPU option. Design:
[ARCHITECTURE §8.5](../ARCHITECTURE.md). What the backend is and how to build it:
[backends/cuda/README.md](../../backends/cuda/README.md).

The backend shares binCV's format and forks its kernels, so every number here
sits on top of a bit-exactness result: `scripts/verify_cuda.sh` proves each
device kernel gives the host library's answer byte for byte (**thirteen suites —
57,630 checks in the Release configuration, 57,585 in the Debug one**), and
every optimized arm is held to its own reference arm's map in the same binary.
Speed is what follows once correctness is settled. The two counts differ by
design rather than by accident: a suite exercising a narrowed domain can only
test the half of that contract its configuration has — the assertion is live in
Debug, the error return is reachable in Release — and each such suite prints
which half it ran instead of silently shrinking.

**One previously published count here was too high, and the correction is worth
stating rather than quietly applying.** The eight pre-frontend suites were
reported at 38,901 Release checks. They now report **36,792**, and the whole
difference is accounted for: **2,112 of those checks never existed as distinct
assertions.** `BINCV_CHECK_EQ` evaluated its first argument twice — once for the
comparison and once inside `std::to_string` for the failure message — and
`test_cuda_median` passes it a helper that itself contains four checks. Each of
that helper's 528 invocations therefore ran its four assertions on identical data
twice and counted eight. 528 × 4 = 2,112 exactly, and adding the three checks of
the new shared-helper sweep closes the arithmetic to the unit:
38,901 − 2,112 + 3 = 36,792. Coverage is unchanged; the second execution tested
nothing the first had not. The macro now binds the value once, which also stops
`BINCV_CHECK_EQ(cudaFree(p), cudaSuccess)` being a double free — which is how a
CUDA suite found it.

**Coverage, stated plainly.** Nineteen of the twenty-seven host operation headers
have device arms: `logic`, `reduce`, `pack`, `census` and `denseDisparity` —
the reductions and dense stereo end to end — then `threshold`, `edge`,
`morphology`, `denoise`, `medianWide`, `pyramid` and `shift`, which is the
sensor stage and the window stage a frontend runs per frame, and from the
frontend round `derivative`, `covariance`, `corner`, `fast`, `orientation`,
`descriptor` and `subpix`. One device operation has no host header at all:
`keypoints.hpp`, the detector-to-keypoint-set link, which exists because a
resident pipeline needs it and a host pipeline does not. **Descriptor matching,
tracking, sparse stereo and the geometry still have none** — and matching is the
notable absence, because it is where the format's word utilisation would pay.

Several accept a **narrower domain than their host twin**, and each names it in
its docstring, asserts it, and returns `cudaErrorInvalidValue` outside it rather
than computing a wrong answer: morphology takes elements up to 32 rows by 512
columns (32 masked), `medianWide` takes K ∈ {1,3,5,7,9} at compile time,
`pyrDownBox` takes 1–8 planes a side, `binarize` takes 1–32, the derivative
takes N ∈ [1,4], and steered BRIEF names its angle domain [-2π, 2π]. A Tier 1
claim here is a claim over *that* domain, said so where the claim is made. The
Debug configuration is what proves those assertions reach nvcc's device pass,
which is why it is a gate rather than a convenience. The rest of the operation
set has no device arm, and the remaining work stays filed as issues rather than
implied here.

**What is not delivered, stated before anything that is.** The **resident
frontend is slower than binCV's own CPU frontend** — 0.82×, ranges disjoint in
all 7 runs — because one kernel is 97% of the frame and runs in a single block.
**FAST misses its role bar by 16.8× on real content and is 1.55× larger** on the
memory meter, so it clears neither axis. `cornerSubPixAsync` misses its own
round-trip rule. Device occupancy was dropped on a measurement rather than
written. Each is stated where it belongs rather than left for a reader to notice
by absence.

`cuda::threshold`'s round-1 failure, reported here previously as a miss, **is
cleared**: it was 2.22× slower at 4K with 7 of 9 runs disjoint and is now 0.744×,
never above 1.00× at any geometry, with its 2.46× memory result unchanged to the
byte. The section below records what the fix was and where the previously stated
limiter was wrong.

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
`cudaMemGetInfo` reserves in 2 MB units, so its reading moves in 2 MB steps and
nowhere in between. A one-byte allocation therefore reads 2.00 MB when it starts
a fresh unit and 0.00 MB when it fits the unit the previous allocation was
already using — the step is the stable quantity, not any single probe, which is
why `cuda_bench_util.hpp` measures the step (allocate one byte at a time until
the reading moves) rather than probing once. The binary entry's 2.0 MB is
therefore the meter's resolution around a 442 KB working set rather than its
footprint, which makes the 5× above a lower bound on the memory lead and not a
measurement of it.

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
- **Every cross-library comparison runs both sides on one explicit stream, and
  finding that out moved three numbers in this document.** OpenCV synchronizes
  the whole device on the default stream. The guard
  `if (stream == 0) cudaSafeCall( cudaDeviceSynchronize() );` is not one
  function: it is in cudev's grid transform (which backs `cudaarithm`'s
  `threshold`), in `cudafilters`' morphology, linear and median filters, in
  `cudawarping`'s `resize` and `pyrDown`, and three times in `cudastereo`'s
  StereoBM. On the default stream OpenCV therefore cannot pipeline across a
  batch while binCV can, so an event bracket around a batch times N serialized
  round trips on one side against N pipelined launches on the other. The
  surcharge, measured with binCV as the control because it carries no such
  guard anywhere:

  | call | default stream | explicit stream | surcharge |
  |---|---|---|---|
  | `cv::cuda::threshold` | 0.0704 ms | 0.0102 ms | 6.91× |
  | `cv::cuda::resize` INTER_AREA | 0.0559 ms | 0.0091 ms | 6.17× |
  | `cv::cuda::pyrDown` | 0.0706 ms | 0.0098 ms | 7.18× |
  | `cv::cuda` erode 3×3 (0.15 ms kernel) | 0.2125 ms | 0.1465 ms | 1.45× |
  | `cv::cuda` median 3×3 (6 ms kernel) | 6.4260 ms | 6.2180 ms | 1.03× |
  | *control* — binCV `threshold` | 0.0146 ms | 0.0142 ms | 1.03× |
  | *control* — binCV `erode` 3×3 | 0.0095 ms | 0.0089 ms | 1.07× |

  Both controls read ~1.00×, and the surcharge scales inversely with kernel
  length, which is exactly what a fixed per-call sync must do. **The bar is the
  explicit-stream number.** The project's rule is that the bar is the best
  existing option; a resident pipeline uses streams and OpenCV supports them on
  every call here, so quoting the default-stream figure would be measuring
  against a fallback nobody would use. Three figures published below were
  inflated by it and are corrected in place — `threshold`, `edgeThreshold` and
  the pyramid ladder. **Morphology, the medians and the StereoBM headline were
  not affected**: their kernels are long enough that one sync is noise, and the
  StereoBM row was re-taken under this protocol and reproduced.
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

## Foundation ops

Microbenchmarks — one kernel in a loop, next to the host library's CPU arm on
the same frame. Shares of a real pipeline come from the dense benchmark, not
from these. These are **CPU arms, not GPU role bars**: they price an operation
against the host library on the same machine, which is a different question
from how it compares to `cv::cuda`. The GPU-against-GPU comparisons are the
three sections below.

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

## The sensor stage: threshold, binarize, edgeThreshold

The three ops that turn a wide frame into bits. Kernel-resident, both sides on
one explicit stream, nine independent process runs per figure, each an
interleaved median of fifteen rounds. The **launch floor measured in every run
is 8.66–9.58 µs**, and it is quoted beside every figure here because two of
these three ops sit on it at the frame size a pipeline runs. Memory is
`cudaMemGetInfo` on **both** sides, each replicated until its own delta clears
eight of the driver's measured 2.00 MB units and then divided by its own
replica count — one meter, named, never crossed with an allocation sum.

| op | geometry | `cv::cuda` | binCV | speed | device memory |
|---|---|---|---|---|---|
| `threshold` → bits | 752×480 | 0.0113 ms | 0.0130 ms | **0.86× — behind** | **2.46×** smaller |
| `threshold` → bits | 3840×2160 | 0.0378 ms | 0.0845 ms | **0.45× — behind** | not measured |
| `edgeThreshold` | 752×480 | 0.0821 ms | 0.0096 ms | **8.1×** | **24.6×** smaller |
| `edgeThreshold` | 1920×1080 | 0.1886 ms | 0.0164 ms | **11.1×** | not measured |

Every memory figure in this document's op-expansion tables is taken at 752×480,
and the rows at other geometries say "not measured" rather than carrying it
forward — the ratio is not constant in frame size, because `GpuMat`'s pitch
padding is a per-width quantity and the driver's reservation step is not.

**The speed column is an aggregate of per-round ratios, not the two medians
divided.** Each round times both arms interleaved and forms that round's ratio;
the column reports the range those ratios span across runs. Dividing the median
column by hand gives a nearby but different number, and the per-round form is
the one that cancels drift — which is the whole reason the arms are
interleaved rather than run to completion one after the other.

**`edgeThreshold` leads on both axes, and it is this family's result.** Its role
bar is the composed `cv::cuda` spelling of the same computation —
`createDerivFilter(CV_8UC1, CV_16SC1, ksize=1, normalize=false)`, whose kernel
at ksize 1 is exactly `[-1,0,1]` and whose default border is already
`BORDER_REFLECT_101`. That filter is separable, so the bar is nine launches
against binCV's one. The rule written before measuring asked for ≥5× on memory
and ≥3× on speed; measured 24.6× and 8.1–11.1×, with sample ranges disjoint in
9 of 9 runs at 1080p. A previously circulated 37.8–49.9× is **withdrawn** — it
was the default-stream artifact above — and the stated bar is still cleared with
about 3× of margin.

The speed comes from a byte-lane arm that does four pixels per lane. `__byte_perm`
builds the shifted neighbour quads, `__vabsdiffu4` does four |a−b| in one
instruction, `__dp4a` folds four byte flags into a nibble, and a three-step
`__shfl_xor_sync` butterfly assembles eight lanes' nibbles into one output
word; the quad holding the last pixel falls back to the very same `edgePixel`
function the reference arm runs, so the arithmetic has one spelling rather than
two. It also cuts the warp's load instructions fourfold. Against its own
reference arm it measures **0.36× at 3840×2160** with disjoint ranges in 4 of 7
runs, and **0.75× at 752×480 with 0 of 7 disjoint — no result there, because
neither arm is distinguishable from the launch floor at that size**. The arm was
kept on the 4K evidence; the two gate-excluded controls (uint16, and the forward
difference, both outside the arm's own gate) read ~1.00× where the measurement
can resolve them. One measured correction to the design that proposed it:
`__vsetgeu4` is **six** instructions on sm_86, not one, and `__vminu4`/`__vmaxu4`
are six each, so the four byte comparisons are about 12 of the ~18 instructions
a lane spends on its quad — the arm wins by amortising loads and addressing over
four pixels, not because the byte-lane arithmetic is cheap.

**`threshold` misses its own speed bar, and is not presented as a win.** The rule
its author wrote before measuring named the fail condition in as many words:
*slower than OpenCV by more than both printed spreads*. At 752×480 and 1080p
the ranges overlap and there is no result in either direction. At 3840×2160
binCV is **2.22× slower with 7 of 9 runs disjoint** — the fail condition, met at
the one size where this measurement can decide anything. The memory bar (≥1.77×
on the working set) passes at 2.46×. By the project's ship rule that combination
does not merge on the memory argument: it gets optimized first, or the gap is
explicitly accepted with the price stated. **Neither has happened yet**, and the
op is documented here as a miss.

The mechanism was located without a profiler. `cuda::threshold` is header-only —
the host's own `impl::thresholdCutoff` reduction composed with `cuda::packBits`,
which is what makes its Tier 1 claim provable rather than restated — so the
kernel under the number is `packKernel<uint8_t, GreaterEqual>`. `cuobjdump -sass`
shows **184 instructions around one LDG and one STG**: 62 IMAD, 26 IADD3, 18
ISETP, and **two software divides** (`I2F.U32.RP → MUFU.RCP → F2I`, and a 64-bit
one) which are the grid-stride loop's `wordIdx / words` and `wordIdx - y*words`.
binCV moves **1.90× less traffic and takes 1.89× longer**: at 1080p it runs at
6.8× its own bandwidth floor where OpenCV runs at 1.9× of its. Two fixes are
named and both are already precedent in this backend — a 2-D grid with
`blockIdx.y` as the row deletes both divides outright, and four-pixels-per-lane
vector loads are exactly what `edgeThreshold`'s byte-lane arm above does to win
8–11× in this same family. Neither was attempted this round. This also settles,
in the opposite direction to the one expected, an earlier suspicion that
`cv::cuda::threshold` was anomalously slow: it was not: binCV was slow and the
default stream was hiding it.

`binarize` — N bit-planes in, one bit-plane out, one launch, templated on plane
count 1…32 for register residency and with **zero spills across all 32
instantiations** — has **no `cv::cuda` counterpart at any API level**. Its speed
verdict is therefore recorded **OUTSTANDING** against the resident pipeline that
will later price it, and no substitute bar is invented for it. For shape only,
and labelled as such rather than as a role comparison: against `cuda::packBits`
on 3.000× less traffic it measures 0.20× at 4K (5 of 7 runs disjoint) and
0.87–0.93× at 752×480 with 0 of 7 disjoint, where it is on the launch floor. It
is still on the launch floor at 4K — 0.008 ms for a 3.1 MB working set — and
would need roughly an 8000×4500 frame to become visible at all.

## The window family: morphology and the medians

Morphology's role bar is `cv::cuda::createMorphologyFilter(op, CV_8UC1,
kernel)->apply()`, which is NPP-backed; the medians' is
`cv::cuda::createMedianFilter(CV_8UC1, 3)`. Same protocol as above, both sides
on one explicit stream. These are the rows the stream correction did **not**
move — their kernels are long enough that a per-call sync is noise, which the
surcharge table's 1.45× and 1.03× rows show directly.

| case | geometry | `cv::cuda` | binCV | speed | device memory |
|---|---|---|---|---|---|
| `erode` rect 3×3 | 752×480 | 0.1429 ms | 0.0118 ms | **11.6–11.9×** | **16.0×** smaller |
| `erode` rect 3×3 | 1920×1080 | 0.2011 ms | 0.0148 ms | **13.0–13.5×** | not measured |
| `morphologyEx` OPEN 3×3 | 1920×1080 | 0.4209 ms | 0.0238 ms | **15.1–16.1×** | not measured |
| `erode` ellipse 5×5 | 752×480 | 0.2196 ms | 0.0154 ms | **12.8–13.9×** | (as rect 3×3) |
| `erode` ellipse 5×5 | 1920×1080 | 0.4048 ms | 0.0176 ms | **22.9×** | not measured |
| `medianWide` K=9 | 752×480 | 6.1888 ms | 0.0249 ms | **235–249×** | **123×** smaller |
| `medianWide` K=9 | 1920×1080 | 34.1522 ms | 0.0538 ms | **634–740×** | not measured |
| `denoiseMedian3` vs binCV's own byte arm | 4096×2160 | 0.0388 ms | 0.0087 ms | **4.6–5.0×** | **8.58×** at 752×480 |

All three required morphology cases clear on both axes with 8–9 of 9 runs
disjoint, which is the leads-on-both-axes disposition their pre-written rule
named. **The case worth pointing at is `erode` with a 5×5 ellipse**, because it
is the one the host arm *loses* — 0.32× against `cv::erode` on x86, where an
AVX2 lane holds 32 bytes and a packed word holds 32 pixels, so the byte side
gets its width for free. On the device that reverses, and the reason is
measurable rather than rhetorical: `__vminu4` and `__vmaxu4` are **six
instructions each on sm_86** (LOP3×3, SHF, IADD3, PRMT; the PTX `vmin4` is
worse at 19), so a byte competitor's lane narrows to four pixels at six
instructions while a packed word stays 32 pixels per instruction. The operation
the host representation loses is the one the device representation wins by
13–23×.

**The honest caveat, printed at the number.** A single-call probe shows
OpenCV's *kernel alone* is 69–85% of its batched time, and binCV's morphology
sits at **1.36× the launch floor**. So this is better read as "binCV is
essentially free and OpenCV is 16× above the floor" than as a kernel-versus-
kernel ratio; roughly 10–17× of it is kernel-to-kernel and the remainder is
OpenCV's per-call host cost. Earlier family figures of 17.4×/18.6×/16.3× become
11.6×/13.3×/12.8× under the corrected protocol — same verdict, smaller
magnitude.

**The 123× on the wide median is not binCV's representation, and saying so is
the point.** OpenCV's `filtering.cpp` sizes its histograms at
`cols*256*partitions + cols*8*partitions` CV_32S, which is roughly **98 MB of
device scratch for one 752×480 frame**; binCV allocates **zero** scratch,
because no kernel in this project heap-allocates. The ratio is an artifact of
the competitor's design, not evidence for bit-planes, and the implementer's own
written expectation for this row was *parity*. The speed figure needs the same
honesty: OpenCV's CUDA median runs 128 blocks of 32 threads — 4,096 threads on
48 SMs, about 5,200× its own bandwidth floor — and is simply a poor
implementation. **The K=9 row is the fair one**, since it compares equal sample
counts, and at 235×/634× it is still the largest role margin in this backend.
The bar is stream-independent (1.03×), so it is robust to the correction above.
The 16-bit `medianWide` has no counterpart at any API level and its speed
verdict is **OUTSTANDING**.

`denoiseMedian3` — the 3-sample median over packed bits, two instructions per 32
pixels via the host's own `maj3` — is measured against **binCV's own byte
`medianWide` with its fast arm on**, deliberately, rather than against the
composed `cv::cuda` spelling (7 buffers, 8 launches) that would have flattered
it. Identical operation, identical border, one launch each, so the only variable
is the representation. Its memory gate is the format's own formula
`width / (rowWords(width)*4)` and it agrees exactly at 7.8333×, zero scratch
both sides. Its **speed gate decides only at 4K**: 1.01× at 752×480 with 0 of 9
runs disjoint, 0.63× at 1080p with 1 of 9, and 0.22× at 4096×2160 with 5–6 of 9.
At every frame size a vision pipeline actually runs, this op is under the launch
floor and **no standalone speed measurement can decide it** — which is a
property of the op's cheapness, not a defect, and is why its share of a real
resident path (8.8% of `bits → denoiseMedian3 → denseDisparityBinary`) is the
number that matters more than its ratio.

**Internal arms, and three honest nulls.** `medianWide`'s fast arm — four
pixels per lane, one aligned 32-bit load per sample offset — is **0.41× at
4096×2160 with 6 of 7 runs disjoint** (K=5: 0.45×, 6 of 7), and its
gate-excluded control at 4095×2160, where a tight stride of 4095 is not a
multiple of 4 and the alignment gate refuses the arm, reads **1.00× in 7 of 7
runs**. That control is run at the top of the ladder on purpose: at a
launch-bound size a control cannot detect a mis-attached switch, and the
benchmark prints why. Morphology's three arms did *not* separate: the 3×3
specialization reads 0.98×/0.92× with 0 of 7 disjoint at both sizes, the
word-parallel `__brev` border 0.74×/0.59× with 0 and 1 of 7, and the `andNot`
fusion 0.73×/0.76× with 0 of 7. Their compulsory traffic is 92 KB at 752×480 —
**0.0002 ms against a ~0.011 ms launch floor** — so binCV's binary morphology is
launch-bound at every frame size a vision pipeline uses and these comparisons
have no resolution rather than a negative result. All three remain the default;
the word-parallel border is additionally defensible on correctness surface,
since it deletes the per-pixel border path, its divergence and its lost-update
race together. **Two dispositions here are owner calls, not measurements**, and
are recorded as open rather than settled.

The `uint4` arm for `denoiseMedian3` was written, proven bit-exact, timed on the
full ladder, and **dropped with its off-switch** — it never separated (0.97–1.04×
at every rung). The reason is arithmetic rather than contention, so no re-run
changes it: at 4096×2160 the whole operation moves 2.21 MB ≈ 3.6 µs of traffic
against a 7–10 µs launch. The kernel is cheaper than the launch that carries it
at every frame size, and it would take roughly 8× more pixels than 4K to change
that. One implementation ships with no switch, on the `censusTransformPacked`
precedent.

## The pyramid, the resident ladder and shift

| comparison | geometry | `cv::cuda` | binCV | speed | device memory |
|---|---|---|---|---|---|
| `buildPyramidBox` vs `resize` INTER_AREA ×3 | 752×480 | 0.0233 ms | 0.0240 ms | **1.02× — a tie** | **7.17×** smaller |
| `buildPyramidBox` vs `pyrDown` ×3 | 752×480 | 0.0256 ms | 0.0239 ms | 0.95× — a tie | (same ladder) |
| `shift` vs `cudaMemcpy2DAsync` | 752×480 | 0.0092 ms | 0.0091 ms | **a wash** | **7.8333×** (formula) |

**The pyramid ladder passes as a tie, and that was written down as a pass before
it was measured.** The disposition table its author wrote first had three rows,
and the middle one said: ranges overlap ⇒ tie, which passes, and reads "a wash on
time, N× on memory". Measured 1.015× and 1.039× across two independent sweeps
with **0 of 9 runs disjoint in both** — squarely that row. Only the 752×480
comparison is like-for-like: `cv::cuda::resize` gives `dsize = 376` at width 753
where `pyrDownWidth(753) = 377`, so at an odd width the two sides are not doing
the same operation and the benchmark says so at the number. **A 5.61× figure
from the family's own pass is withdrawn** — `resize` was paying 6.17× and
`pyrDown` 7.18× on the default stream.

The memory side is where this op is actually interesting, and it is an
**equality rather than a threshold**: the four-level ladder
(752×480 at 1 plane, then 3, 4, 5 planes) is **93.5 KB in one `cudaMalloc`**,
equal to the closed formula to the byte, and the suite asserts the levels are
consecutive slices of that one allocation rather than only printing the total.
`cudaMemGetInfo` on both sides reads 7.17×, which corroborates the 7.378×
read back from `GpuMat::step` — the byte ladder's real pitches are
1024/512/512/512 B per row, not the 512 B the design assumed for level 0.

`shift` has **no OpenCV counterpart at any API level**, so its speed verdict is
**OUTSTANDING**. What it is measured against instead is the thing a byte
pipeline would actually use for an integer translation — `cudaMemcpy2DAsync`, a
pitched DMA — and the result written down in advance was that a wash or a loss
would be the expected and acceptable outcome. It is a wash: 0.948–0.985×, 0 of 9
disjoint. **The "structural twice over" claim this op was designed under is
wrong on its instruction half and is withdrawn here rather than quietly
dropped.** A DMA spends *zero* ALU instructions per pixel, so binCV's one
`__funnelshift` is compared against none, not against thirty-two; and at 752×480
both 46 KB and 361 KB sit inside this part's 4 MB L2, so the traffic half is a
footprint claim too. The op is **7.8333× smaller at width 752** — the format's
formula `height*rowWords(width)*4` against `height*width*1`, which reaches
8.0000× only where the width is a multiple of 32, and 752 is not — and a wash
on time against a DMA engine. That is the whole of it. The `__funnelshift` arm still ships as the
default against the two-shift-or arm it ties with (1.01×, 0 of 7), on the
correctness-surface argument that it is defined at a shift count of zero and
removes the undefined-behaviour branch entirely — a correctness argument, stated
as one rather than smuggled in as speed.

**The ladder's fast arm is an open disposition, not a result.** `pyrDownBox`'s
bit-sliced arm B beats the ballot-based arm A by 0.555× at 3840×2160, but with
sample ranges disjoint in only **6 of 14 pairings** — a minority, where the rule
written before measuring required a reproduced disjoint win. At 752×480 the
per-level ratios are 0.93/0.97/0.97 with 0 of 7 disjoint, because every level
there is on the launch floor, which that same rule predicted in writing and so
is not a measured negative. By the letter of the rule arm A ships; the medians
consistently favour arm B and the runtime switch makes it a one-line change
either way. **It is left as arm B and flagged, rather than resolved by relaxing
the bar that was written to decide it.**

## The frontend on device

The feature path a visual-odometry frontend runs per frame — derivatives, the
gradient covariance, corner response and selection, FAST, orientation, BRIEF —
forked onto the device, each kernel bit-exact against its host twin. The
end-to-end claim this round was judged on is **negative, and it is one kernel**:

> **The resident device frontend is 1.22× SLOWER than binCV's own CPU
> frontend** — 6.462 ms [6.419–6.509] against 5.313 ms [5.152–5.421], **ranges
> disjoint in all 7 runs**. Detection is **97.0%** of the device frame, and the
> kernel inside it that dominates runs in **one block at 1.29% of the SMs**.

That is not a ceiling and not a format result. It is a parallelization defect
with a measured size, and the rest of this section is mostly the evidence that
everything *around* it already works.

### The frame, stage by stage

200 real EuRoC V1_02 cam0 frames through `backends/cuda/examples/cuda_vio_frontend.cpp`,
7 process runs, one explicit stream, CUDA events on the device side and the host
library's own clock on the host side — **two different clocks, labelled as such
at every row**. The host column is context for locating the defect; it is not a
role bar, and no `cv::cuda` whole-frontend counterpart exists to be one.

| stage | device ms | share | host ms | device/host |
|---|---|---|---|---|
| upload (H2D) | 0.0505 | 0.8% | — | — |
| sensor: `medianWide<3>` + `edgeThreshold` | 0.1333 | 2.1% | 0.0910 | 1.46× slower |
| pyramid ×3 | 0.0196 | 0.3% | 0.0602 | 3.07× faster |
| `derivativeXY` | 0.0133 | 0.2% | 0.0220 | 1.65× faster |
| **detect: `goodFeaturesToTrack`** | **6.1073** | **97.0%** | 4.9639 | **1.23× slower** |
| `keypointsFromCorners` | 0.0043 | 0.1% | — | no host twin |
| orientation r=15 | 0.0078 | 0.1% | 0.1036 | 13.3× faster |
| describe BRIEF-256 | 0.0078 | 0.1% | 0.0585 | 7.5× faster |
| download (D2H) | 0.0383 | 0.6% | — | — |

**The arithmetic that settles it.** If every stage other than detection went to
*zero*, the frame would still be 6.107 + 0.089 = **6.20 ms**, against the host's
5.31 ms. No amount of work on the other seven stages makes this pipeline beat
the CPU. Conversely, a `selectKernel` using even a quarter of the machine puts
the frame near 0.9 ms — roughly **5.9× faster than the host**. The whole value
of the resident frontend rests on one kernel.

**What residency itself delivers, and it is not nothing.** Exactly **1.00
synchronize per frame** in all 7 runs; 360,960 B up against 11,536 B down, a
**31.3×** bus asymmetry with nothing frame-sized returning. Correctness across
the chain is a gate rather than a metric and it passes: 0 corner-count, 0
position, 0 `keep`, 0 rotation-bin and **0 of 222,432 descriptor words** differ
from the host, with max |Δangle| 2.384e-07 rad, identical in all 7 runs.

### The one kernel, located with a profiler rather than argued

`ncu` works on this machine now — as `/opt/nvidia/nsight-compute/2024.2.1/ncu`
by full path, which supersedes the "no hardware profiler" note below for
everything measured after it. It settles in one run what the previous round had
to triangulate:

| kernel | duration | SM % | DRAM % | grid | top stall | limiter |
|---|---|---|---|---|---|---|
| `selectKernel` (gftt) | 13.6 ms | **1.29%** | **0.45%** | **1 block** | tex_throttle 37% | parallelism-starved |
| `fastSortKernel` | — | **1.1%** | **0.7%** | **1 block** | long_sb 43% | parallelism-starved |
| `fusedCandidateKernel` | 144 µs | 65.4% | 0.6% | 180 blocks | short_sb 64% | MIO/compute |
| `responseKernelWindow` | 64.4 µs | 78.1% | 4.1% | 9 | short_sb 46% | compute |
| `briefBallotKernel` | 8.5 µs | 8.9% | 20.6% | 6 | long_sb 77% | memory latency |
| `orientWideWarpKernel` | 7.9 µs | 21.8% | 7.3% | 6 | long_sb 65% | latency |
| `derivativeXYKernel` | 2.9 µs | 9.4% | 8.4% | 6 | imc_miss 49% | launch-bound |
| `covBatchKernel` | 4.0 µs | 10.6% | 8.5% | 16 | long_sb 34% | launch-bound |

`selectKernel` and `fastSortKernel` are **the same defect twice**: both order a
variable-length result in a single block, both use about 1% of the SMs and under
1% of DRAM. On a 48-SM part the GPU is ~99% idle for the stage that owns the
frame. Neither is a format problem and neither is a kernel-arithmetic problem —
`compaction.hpp` already names the alternative (prefix-sum compaction), and it
was not taken here.

### FAST: the bar is missed, and the cost is the ordering

| edge | corners | nextPow2 | `cv::cuda` FAST | binCV | ratio | disjoint | set gate |
|---|---|---|---|---|---|---|---|
| 10 | 23,274 | 32,768 | 0.1286 ms | 2.2726 ms | **17.9× slower** | 7/7 | agree 7/7 |
| 17 | 19,898 | 32,768 | 0.1338 ms | 2.2563 ms | **16.8× slower** | 7/7 | agree 7/7 |
| 30 | 12,379 | 16,384 | 0.1222 ms | 1.0643 ms | 8.8× slower | 7/7 | agree 7/7 |
| 50 | 886 | 1,024 | 0.1063 ms | 0.0532 ms | **2.1× faster** | 6/7 | agree 7/7 |
| 80 | 161 | 256 | 0.1004 ms | 0.0353 ms | **3.8× faster** | 7/7 | agree 7/7 |

The corner **set** is identical to the host's in every row, so this is
like-for-like. The cost tracks **`nextPow2(corners found)`** and nothing else:
23,274 and 19,898 corners cost the same 2.26 ms because they share a
power-of-two bucket, halving the bucket halves the time (2.12× measured against
bitonic's predicted 2.30×), and the arm is flat at 2.22 ms from capacity 32,768
through 262,144 — so it is not capacity either. The crossover is around
1,000–4,000 corners; the reference frontend's own edge map sits far above it.

**The locator, and why the ordering is the whole gap:** the same detector with
its store capped at 512 runs at **0.0303 ms against OpenCV's 0.1721 ms — 5.2×
faster**, every pixel still tested. The ring algebra clears parity roughly 5×
over. The raster sort on top of it is **98.6% of the operation**.

FAST also **loses on memory** at the capacity this content needs: 1088.0 KB
against OpenCV's 704.0 KB, `cudaMemGetInfo` on both sides — **binCV is 1.55×
larger**. Under the both-axes rule it does not ship as-is on either axis, and it
is reported here as a miss rather than as a result with a price attached.

### The role bars

Both arms on **one explicit stream**, medians of 7 process runs. The stream is
not a detail: OpenCV synchronizes the whole device on the default stream, a
surcharge measured up to 7.18× here, and a default-stream pair in
`cuda_sensor_benchmark` that printed "7.35× FASTER" for `threshold` has been
deleted rather than repaired — `cuda_role_benchmark` owns that comparison.

| operation | `cv::cuda` arm | OpenCV | binCV | ratio | disjoint | verdict |
|---|---|---|---|---|---|---|
| `threshold` 752×480 | `cv::cuda::threshold` | 0.0100 ms | 0.0094 ms | 1.000 | 0/7 | **PASS** |
| `threshold` 1920×1080 | ″ | 0.0124 ms | 0.0114 ms | 0.908 | 0/7 | **PASS** |
| `threshold` 3840×2160 | ″ | 0.0367 ms | 0.0272 ms | **0.744** | 1/7 | **PASS** |
| describe, N=1000 | `cv::cuda::ORB::computeAsync` | 0.1070 ms | 0.0107 ms | **0.108** | **7/7** | **MET, 9.3×** |
| FAST | `FastFeatureDetector` | 0.1505 ms | 2.2343 ms | 14.84 | 7/7 | **MISSED** |
| `goodFeaturesToTrack` (wall) | `createGoodFeaturesToTrackDetector` | 3.7514 ms | 9.8579 ms | 2.659 | 2/7 | slower |
| min-eigenvalue response | `createMinEigenValCorner` | 0.0552 ms | 0.0671 ms | 1.292 | **0/7** | **not a result** |

Memory, `cudaMemGetInfo` delta taken identically **on both sides** — the only
meter readable across libraries, replicated until each total clears eight of
this driver's 2 MB units, and never mixed with binCV's own allocation sums:

| operation | binCV | OpenCV | ratio |
|---|---|---|---|
| `threshold` | 416.0 KB | 1024.0 KB | **2.46× smaller** |
| describe, N=1000 | 48.0 KB | 2048.0 KB | **42.67× smaller** |
| `goodFeaturesToTrack` | 2048.0 KB | 10240.0 KB | 5.00× smaller |
| corner response | 2048.0 KB | 10240.0 KB | 5.00× smaller |
| FAST @ capacity 32,768 | 1088.0 KB | 704.0 KB | **0.65× — binCV is 1.55× LARGER** |

**`cuda::threshold` clears the bar its own round-1 rule failed.** That rule's
fail condition — slower than `cv::cuda::threshold` by more than both printed
spreads — was met at 2.22× slower at 4K with 7 of 9 runs disjoint. It is now
0.744× at 4K and never above 1.00× at any geometry, with the memory result
byte-identical at 2.46×. The mechanism was two software divides from a
`wordIdx / words` in the flat index: a row-grid shape (`blockIdx.y` **is** the
row) removes them, and a byte-lane shape on top of it reads four pixels per lane
through one 32-bit load and `__vsetgeu4`. 184 SASS instructions became 104, with
zero software divides and zero spills.

**The profiler corrects this op's stated limiter, and the correction is the
point rather than the conclusion.** The optimizing pass wrote that the residual
gap to its derived 0.0215 ms target was "launch overhead on this host, not the
kernel". It is not. `ncu` reads the kernel itself at 28.26 µs against a 27.2 µs
batched median — the launch is pipelined away — at **63.8% of peak DRAM with 53%
long-scoreboard**. The residual is **bandwidth realization, about 330 GB/s of
this part's 608 GB/s peak**. The same profile confirms both the gain and the
claimed mechanism independently: arm 0 sat at 67.2% SM / 19.3% DRAM
(compute-bound on its divides) and arm 2 sits at 49.4% SM / 63.8% DRAM
(memory-bound, where a packer belongs), 93.3 µs → 28.3 µs = 3.30× against a
timed 3.22×. The decision does not move; the stated reason was wrong and is
corrected here rather than left standing.

### Where binCV has no structural advantage, said plainly

Three of these ops have none, and the reason in each case is that the work is
not in bits.

**The corner response.** `cornerMinEigenValAsync` reads 1.292× against
`createMinEigenValCorner` with **0 of 7 runs disjoint** — that is not a result
in either direction, and it is reported as one that did not resolve rather than
rounded to parity. The response is a `sqrt` and two products per *pixel* in
`float`; the covariance feeding it is bit-work, but the response on top of it is
float arithmetic of exactly the shape a byte pipeline already does well. The
host library's own 4.81× for a bit-sliced blockSize-3 response **does not port**:
on device it measures 0.74× — slower, ranges disjoint — because the host's win
was removing per-pixel *addressing*, while the device form hands one thread 32
pixels of `sqrt`, i.e. 32× less parallelism on the float half. That is a
negative result and is printed as one; the default is unchanged.

**Sub-pixel refinement.** `cornerSubPixAsync` **misses its own round-trip rule**:
0.53 ms to download the derivative planes and refine on the host, against 1.91 ms
resident. Bit-exactness forces `double` and one thread per corner, and this
GeForce part runs FP64 at 1/64 rate. It has no `cv::cuda` counterpart at any API
level, so it carries no speed bar — but the honest reading of its own rule is
that a caller should refine on the host today. It ships as a resident
convenience with that number stated, not as a win.

**The derivative.** It is **launch-bound at every size tested**, including
3840×2160 (0.0070 ms against a 0.0070–0.0083 ms floor; `ncu`: 2.9 µs, 9.4% SM,
16.3% occupancy). It measures 0.210 against `cv::cuda::createDerivFilter` with 5
of 7 disjoint, and its own rule's validity clause fires: that ratio is a **lower
bound on the gap and says nothing about binCV's kernel**. The honest statement is
that binCV's derivative costs a launch at this size and OpenCV's costs a launch
plus its filter work. Its memory result is real and separately gated — 229,376 ..
262,144 B against a predicted 230,400 B, inside the interval, against OpenCV's
6,291,456 .. 6,324,224 B, which is **24.0×–27.6× smaller** and reported rather
than gated, since no memory floor for it has been set.

### Verdicts recorded OUTSTANDING

Four operations have **no OpenCV counterpart at any API level**, on CPU or GPU,
so no speed bar exists and none was invented. They ship on correctness, memory
and the host comparison, with the speed verdict **OUTSTANDING**:

- **the gradient covariance** (`gradientCovarianceAsync`, `gradientCovarianceBatchAsync`)
  — `cornerHarris` and `createMinEigenValCorner` compute a dense float response
  *through* a covariance; neither exposes one. Scratch is **0 B**, verified as a
  `cudaMemGetInfo` delta of exactly 0 across 200 batch launches, at ≤24 B/window.
- **orientation** — no `cv::cuda` entry point orients provided keypoints. No CPU
  number was quoted in place of the missing GPU one.
- **`keypointsFromCorners`** — the detector-to-keypoint-set link, which has no
  counterpart because no other library needs it. It is the one op here that
  **met both halves of its rule**: 1.00 synchronize per frame against the
  round-trip arm's 2.00, and 0.0043 ms against 0.1547 ms — 36×, stage ranges
  fully disjoint, and frame totals disjoint in all 7 runs.
- **`shift`**, as recorded above.

**Descriptor matching is not on this list, and must not be filed as OUTSTANDING.**
The OUTSTANDING verdict covers a binCV operation with no OpenCV counterpart.
Matching is the reverse: OpenCV has `BFMatcher(NORM_HAMMING)` and **binCV has no
device arm at all**. There is nothing to time. The format's advantage there is
real and unspent — descriptors come out as `uint32_t` words, so a matcher issues
8 `__popc` per 256-bit descriptor where `cv::cuda`'s `HammingDist::reduceIter` at
`uchar` issues 32 — and it sits on the side of the pipeline that would make the
residency argument close.

### Arms measured and left off

Two optimized arms cleared a static case and lost the timed one, and both ship
reachable but **off by default** with the numbers recorded rather than deleted:

- **the `__dp4a` wide-orientation arm.** 3.4× fewer loads and 1.55× fewer
  instructions in the SASS, and at the operating point it buys nothing: 0.945 at
  N=470 and 0.812 at N=1000, **0 of 6 runs disjoint at both**, because the kernel
  is launch-bound there. At N=100,000 it is 1.46× faster, 6 of 6 disjoint. Its
  rule named N=470 and N=1000 as the deciding points, so it stays off.
- **the funnel-shift covariance arm.** Re-priced where the comparison is
  decidable (4,000 windows of 31×240): **1.13× slower, 5 of 7 disjoint**. The
  funnel saves popcounts, not loads — a run straddling two words must still read
  both — so the word-visit count fell and the byte count did not.

Two arms were removed on measurement: a one-thread-per-window covariance (200
windows is 200 threads on a 48-SM part; lost 14 readings of 14) and a dense-rank
counting sort for FAST scores (1,900 codes collapsed to 380 distinct responses —
it was wrong, not merely slow).

**The packer's row-grid arm is the one disposition this round did not settle.**
Its static case is unambiguous (184 → 48 instructions, 2 software divides → 0)
and `ncu` reads the effect as real (75.5 µs against 93.3 µs, 1.24×), but its
ranges do not separate — 1 of 7 at 1920×1080, **0 of 7 at 3840×2160** — and the
rule written for it required separation at one of those two geometries. By that
rule it goes. It is kept and flagged instead, because it is the live arm for
uint16, `packQuant`, odd strides and cutoff 256, which the byte-lane arm's gate
excludes; deleting it sends those cases back to the grid-stride arm that `ncu`
shows is compute-bound at 67–74% SM. The byte-lane arm does not depend on the
outcome: it clears its bar at **2.66× over the row grid at 4K, 7 of 7 disjoint**,
and `packQuant` shows no regression at any geometry.

### A harness limit that now decides more than it should

Three real effects in this round have medians far from 1.00 with every run
one-sided and only a minority of runs range-disjoint: the row-grid packer arm,
the fused `derivativeXY` (2.00× at 752×480, decaying to 1.32× at 4K exactly as
its author predicted in writing for a signature result, but 3/7, 3/7, 1/7
disjoint) and the bit-plane orientation arm. `PairedTiming::separated()` is a
min/max range test, and a single outlier destroys it where the per-round ratio
distribution from the same rounds stays tight and one-sided. On an idle GPU this
is still the wall. It is recorded here as an open question about the harness —
whether `separated()` or a one-sided per-round test is the standard — and not
worked around by relaxing any individual bar.

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
- **No hardware profiler was available for the stereo, sensor and window
  numbers, and that shaped the method. It is available now, and the frontend
  numbers use it.** Nsight Compute failed with `Unknown Error on device 0` —
  reproduced on a three-line kernel with a single basic metric, so it was
  device-level counter access being refused rather than anything about these
  kernels. The cause was NVIDIA's default of restricting GPU performance
  counters to administrators, and the fix is Windows-side: set
  `RmProfilingAdminOnly = 0` under
  `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak` and
  **reboot Windows** — a `wsl --shutdown` does not reload the driver.

  **`ncu` now works, as `/opt/nvidia/nsight-compute/2024.2.1/ncu` by full path.**
  Nothing is on `PATH`, and the `ncu` shipped in `/usr/local/cuda-11.1/bin` is
  version 2020.2.0, too old for this driver: build with the 11.1 nvcc as always,
  profile with the 2024 tool. `compute-sanitizer` works from CUDA 11.7 or 12.1
  (the 11.1 one does not), verified against a deliberate out-of-bounds write
  rather than against a clean run. **`nsys` remains unusable**: it writes a
  report containing zero CUDA kernel events at any trace setting, which is the
  WSL2 CUDA-tracing limitation in 2022.1.3, the newest available here.

  This matters for reading the rest of this document. Every limiter claim above
  the frontend section rests on static SASS evidence and controlled A/B, because
  that was all there was; the frontend section's claims are hardware counters.
  Where the two have been compared, the profiler has already corrected one
  stated limiter — see `cuda::threshold` — while leaving its decision intact.

  So the stereo, sensor and window numbers are CUDA-event timing, `nvcc -Xptxas
  -v` for the static picture (registers, spills, occupancy arithmetic), and
  controlled A/B against the arm each change replaced. That was enough to locate
  the dense matchers' limiter, but it took six experiments where a stall-reason
  profile takes one run — `smsp__pcsamp_warps_issue_stalled_*` answers directly
  what those six had to triangulate, and it is available now. The frontend round
  used it to settle in one run what a whole round of argument could not: that two
  kernels launch a single block each.

  The op-expansion round extended this note in two directions. `cuobjdump -sass`
  turned out to answer more than expected: it located `cuda::threshold`'s two
  software divides and priced every byte-lane intrinsic these families were
  designed around — several of which the designs had budgeted at one instruction
  and which are six. A host-enqueue probe and a single-call probe together
  separated OpenCV's kernel time from its per-call host cost without a profiler
  at all. But **`cuda-memcheck` is also broken here, not just `ncu` and `nsys`**:
  given a deliberate 1020-element overread of a 4-element allocation it printed
  `ERROR SUMMARY: 0 errors`. No device out-of-bounds read in this backend is
  observable by any tool available on this machine, which is why the pyramid's
  source-word guard is pinned as a swept arithmetic invariant instead — see
  below.
- **One guard in a shipped default cannot be proven by any value test, and is
  documented as such.** `pyrDownBox` guards a source-word read that, on
  analysis, can only ever feed destination columns past `width`: source word
  `2i+1` supplies columns `[32i+16, 32i+32)` and is missing exactly when
  `srcWidth ≤ 64i+32`, which forces `dstWidth ≤ 32i+16`. Removing the guard
  changes **not one output bit**, and that was watched: with it removed the
  whole 1,287-check suite still passes. It is a memory-safety guard, not a
  correctness one; `cuda-memcheck` cannot see the difference either, so what
  pins it is a swept invariant over 4,096 widths asserting the missing word can
  only feed padding. The header says plainly that it is there for safety rather
  than for the answer.
- **Device occupancy was dropped on a measurement, not left unwritten.**
  `markOccupiedBatch` / `occupiedBatch` / `clearOccupancy` have no `cv::cuda`
  equivalent and no CPU OpenCV equivalent, so no role bar exists for them. The
  best existing option for the job they do is the host library's own
  `spaceCandidates`, at **3,333 ns and zero bytes** — which is *below this
  host's measured 11–13 µs launch floor*, so no device shape can clear it: one
  launch costs more than the entire host arm. The only other bar on offer was
  the host *mask* arm (88,767 ns on x86-64, 380,629 ns on aarch64), which
  `spaceCandidates` already beats by 26.6×, and passing against that would be
  measuring against a fallback nobody would use. Shipping a mask producer and a
  mask reader with no device consumer between them would also add two kernels to
  the bit-exactness budget forever. Not built; the numbers are recorded here so
  the decision is not re-taken from scratch.
- **Named, measured, and not built.** A fused single-kernel OPEN/CLOSE (one
  launch saved, `BORDER_CONSTANT` only, a second hand-written morphology kernel
  to keep bit-exact forever, and its apron arithmetic was wrong at every block
  seam). A separable RECT/CROSS decomposition and the log-depth fold over a flat
  span — the latter is the **best remaining unexploited win in the window
  family**, taking a 15×15 rect's horizontal pass from ~56 ops per 32 pixels to
  8, but it buys nothing at 3×3 and nothing measurable at any size where the op
  is launch-bound, which on this device is every size. A vertical bit-transpose,
  which is the only idea in reach that changes the family's asymptotics in the
  tall-element direction, and is entirely unprobed. A shared-memory tiled
  morphology arm, refused on the recorded 1.28× precedent above plus the fact
  that an erosion's fold is not invertible. `pyrDownFiltered` and a Gaussian 5×5
  device arm, which at `NIn == NOut == 8` has no footprint advantage by
  construction and whose CPU analogue the host already records at 13.7× slower
  than `cv::pyrDown`. A fused two-level ladder, cut before measuring because
  CUDA graphs are the real answer to its one-launch-of-three argument.
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
