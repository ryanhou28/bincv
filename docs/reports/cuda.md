# CUDA backend

The GPU backend, measured against the host library on the same machine and
against `cv::cuda::StereoBM` as the best existing GPU option. Design:
[ARCHITECTURE §8.5](../ARCHITECTURE.md). What the backend is and how to build it:
[backends/cuda/README.md](../../backends/cuda/README.md).

The backend shares binCV's format and forks its kernels, so every number here
sits on top of a bit-exactness result: `scripts/verify_cuda.sh` proves each
device kernel gives the host library's answer byte for byte (**eight suites —
38,901 checks in the Release configuration, 38,881 in the Debug one**), and
every optimized arm is held to its own reference arm's map in the same binary.
Speed is what follows once correctness is settled. The two counts differ by
design rather than by accident: a suite exercising a narrowed domain can only
test the half of that contract its configuration has — the assertion is live in
Debug, the error return is reachable in Release — and each such suite prints
which half it ran instead of silently shrinking.

**Coverage, stated plainly.** Twelve of the twenty-seven host operation headers
have device arms: `logic`, `reduce`, `pack`, `census` and `denseDisparity` —
the reductions and dense stereo end to end — and, from the op-expansion round,
`threshold`, `edge`, `morphology`, `denoise`, `medianWide`, `pyramid` and
`shift`, which is the sensor stage and the window stage a frontend runs per
frame. Tracking, the frontend's feature path, sparse stereo and the geometry
have none.

Four of the six new ones accept a **narrower domain than their host twin**, and
each names it in its docstring, asserts it, and returns `cudaErrorInvalidValue`
outside it rather than computing a wrong answer: morphology takes elements up
to 32 rows by 512 columns (32 masked), `medianWide` takes K ∈ {1,3,5,7,9} at
compile time, `pyrDownBox` takes 1–8 planes a side, and `binarize` takes 1–32.
A Tier 1 claim here is a claim over *that* domain, said so where the claim is
made. The rest of the operation set has no device arm, and the remaining work
stays filed as issues rather than implied here.

**What this round does not deliver.** `cuda::threshold` is correct, is 2.46×
lighter than `cv::cuda::threshold`, and is **slower** — it meets the fail
condition its own written rule named, so it is reported below as a miss with
its mechanism located, not as a result. Device occupancy was dropped on a
measurement rather than written. Both are stated where they belong rather than
left for a reader to notice by absence.

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
