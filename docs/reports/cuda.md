# CUDA backend

The GPU backend, measured against the host library on the same machine and
against `cv::cuda::StereoBM` as the best existing GPU option. Design:
[ARCHITECTURE §8.5](../ARCHITECTURE.md). What the backend is and how to build it:
[backends/cuda/README.md](../../backends/cuda/README.md).

The backend shares binCV's format and forks its kernels, so every number here
sits on top of a bit-exactness result: `scripts/verify_cuda.sh` proves each
device kernel gives the host library's answer byte for byte (**seventeen suites —
194,975 checks in the Release configuration, 194,922 in the Debug one**), and
every optimized arm is held to its own reference arm's map in the same binary.
Speed is what follows once correctness is settled. The two counts differ by
design rather than by accident: a suite exercising a narrowed domain can only
test the half of that contract its configuration has — the assertion is live in
Debug, the error return is reachable in Release — and each such suite prints
which half it ran instead of silently shrinking.

**One previously published count here was too high, and the correction is worth
stating rather than quietly applying.** The eight pre-frontend suites were
reported at 38,901 Release checks. The round that found it brought them to
**36,792** (they read **36,844** today, the 52 being the dense cases added
since), and the whole difference was accounted for: **2,112 of those checks never existed as distinct
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

**Coverage, stated plainly.** Twenty-two of the twenty-seven host operation
headers have device arms: `logic`, `reduce`, `pack`, `census` and
`denseDisparity` — the reductions and dense stereo end to end — then
`threshold`, `edge`, `morphology`, `denoise`, `medianWide`, `pyramid` and
`shift`, which is the sensor stage and the window stage a frontend runs per
frame; from the frontend round `derivative`, `covariance`, `corner`, `fast`,
`orientation`, `descriptor` and `subpix`; and from the round below
`opticalFlow` (Lucas-Kanade), `blockMatch` and the sparse-stereo half of
`stereo`. Two device operations have no host header at all: `keypoints.hpp`,
the detector-to-keypoint-set link, and `sparseMatch.hpp`'s descriptor matcher —
both exist because a resident pipeline needs them and a host pipeline does not.

**What still has no device arm, and one of them is a decision rather than a
gap.** The geometry does not, and that is the round's answer rather than its
backlog: five-point RANSAC was measured on device and **lost by 15.9×**, so
nothing shipped (see *The geometry, and the measurement that says it stays on
the host*). `denseCensusBox` adds a second packed matcher beside the first. The
rest of the operation set has no device arm, and the remaining work stays filed
as issues rather than implied here.

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

**What is not delivered, stated before anything that is.**

- **`cornerSubPixAsync` misses its own round-trip rule, and the median has
  changed sides.** Measured as ONE paired comparison rather than three
  separately-timed medians added together, and against the whole-plane download —
  the tighter of the two baselines on this machine, by 50× — the device arm reads
  **0.5656 ms** against the round trip's **0.3911 ms**, with **72 of 77** paired
  rounds favouring the round trip. That is 1.44× apart against a 1.39× bar, so a
  result *against* the device arm in this sweep, where fourteen earlier runs read
  a null at parity; either reading is a miss, not the 1.07× in the device arm's
  favour that was published before. The earlier reading added three medians that
  drift independently, and the host term alone swings 0.280–0.437 ms across seven
  runs. The profiler says why the device arm cannot pull ahead and the answer
  is a ceiling, not a defect: at 200 corners the kernel is 6.25 warps of work
  on a part that holds 2,304, and the one decomposition that would add
  parallelism is the one its bit-exactness argument forbids. **This one is a
  stop-and-ask**, not a number to fill in.
- **`cornerMinEigenValAsync` did not resolve, for the third round running.**
  Re-taken over seven fresh processes it reads **0.0515 ms for `cv::cuda`
  against 0.0590 ms for binCV** — 1.15× apart against a 1.76× bar, so a **null
  result** on the magnitude; and 23 of its 105 paired rounds fall binCV's way, so
  the direction is not established either. Both halves of the verdict decline it,
  which is what noise looks like. The limiter is named
  (occupancy, not either roof); what is missing is resolution, not an
  explanation.
- **The census entry is LARGER than `cv::cuda::StereoBM`** on a meter run on both
  sides, which reverses a figure previously published here. It is faster on the
  same run. **That loss is inherent to census, not a defect in this
  implementation:** the transform expands 8 bits a pixel into a 32-bit descriptor
  word, so the two transformed images alone are 2 × 1,410.0 KB = **2,820 KB before
  anything else is allocated**, against StereoBM working on the 8-bit frames
  directly. No implementation of census in the standard layout can be smaller than
  its own descriptors. A caller who wants the memory back can have it — the
  **plane-layout** intermediate is 3,217.5 KB against the packed one's 3,877.5 KB
  — at **21.7× the matcher time**. That is a documented choice, not a defect and
  not a single path.
- **Block matching's accuracy floor and sparse stereo's residual floor are
  unset.** Both clear their speed and memory bars; neither has a stated accuracy
  magnitude, and inventing one is forbidden. Block matching is **ship-blocked**
  on that; sparse stereo ships on its stated bar with its accuracy published.
- **Lucas-Kanade leads only up to about 512–1024 keypoints**, and the win is the
  launch shape rather than the kernel — profiled, the kernel work is a 1.56×
  loss. Whether that ships with the density named in the header, or the
  traversal is redesigned first, is the ship-rule escape and it is unmade. The
  *second* question that rode with it — whether a lead this host cannot size is a
  lead at all — **is ruled and closed**: at the frontend's own spacing binCV is
  faster in **105 of 105** paired rounds, by **1.35× to 4.36×** (0.0792 ms against
  `cv::cuda`'s 0.1475 ms). The direction is established by the rounds; the
  magnitude is not, and is published as that range rather than as one number. See
  [the judgement the ruling settled](#the-judgement-the-ruling-settled).
- **The gated matcher's device speed rationale is not established** — gated
  against brute force reads 1.11×, and a difference that small is not one this
  host's noise can resolve. It ships for the caller who already has the gate,
  not on that number. (That row was originally called "not a result" because the
  two arms' *ranges overlapped*; that test has since been corrected to the
  project's own rule, and this row has **not** been re-taken under it. 1.11× is
  well inside the noise either way, so the disposition stands and the evidence
  for it is weaker than the corrected rule could make it.)
- **Device occupancy was dropped on a measurement** rather than written.

Three entries left this list. The **resident frontend** was 1.22× slower than
binCV's own CPU frontend and is now **5.62× faster**, ranges globally disjoint.
**FAST** missed its role bar by 16.8×, then cleared it on speed while still
trailing on memory, and now **leads both axes**: 6.01× faster with 7 of 7 runs
disjoint, and **1.574× smaller** on `cudaMemGetInfo` where it was 1.545× larger.
**`goodFeaturesToTrackAsync` ships**: it is 5.3× faster and 5.33× smaller than
its `cv::cuda` counterpart with 7 of 7 runs disjoint, and although its bar was
escalated twice and never set — and no number was invented in its place — the
owner has now ruled that this clears. The caveat rides with it rather than being
dropped: **OpenCV's arm swings 3.24–13.61 ms across runs** because its
min-distance spacing filter runs on the CPU, so the denominator is not one
number, and binCV crosses parity against **every version of it**, including
round 2's own quieter 3.55–4.28 ms.

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
time (CUDA events), **both sides on one explicit stream**, medians of 7
independent process runs with the range beside each. The ratio column is the
median **per-round paired** ratio, not a ratio of the two medians, and the
verdict column is the rule in [How a difference is decided](#how-a-difference-is-decided):

| | time | vs StereoBM | verdict | disjoint |
|---|---|---|---|---|
| **binCV binary entry** (pair already packed) | **0.0679 ms** [0.0678–0.0703] | **10.8× faster** | RESULT | 7/7 |
| binCV census entry (wide frames in, transform + match) | 0.5418 ms [0.527–0.547] | **1.43× faster** | RESULT | 7/7 |
| binCV census matcher alone | 0.4223 ms [0.398–0.436] | **1.88× faster** | RESULT | 7/7 |
| `cv::cuda::StereoBM(64, 9)` | 0.7438 ms [0.718–0.842] | — | — | — |

The two census rows are taken in their own section against their own StereoBM
arm, which read 0.7584 ms [0.730–0.816] there; their ratio column is that
section's paired ratio and not this table's StereoBM median divided into
theirs. Every ratio here cleared the rule on both halves: within-run 1.80× /
1.33× / 1.43× against run-to-run 1.18× / 1.11× / 1.11×, with all 105 paired
rounds falling the same way in each.

**The census entry has crossed.** It was 1.29× behind StereoBM in the previous
round and is now 1.43× ahead, and the whole move is one kernel — a
warp-cooperative box matcher that is **2.49× [2.40–2.50]** over the packed
matcher it replaces, disjoint in all 7 runs, at **0 bytes** of added scratch and
0 bytes of shared memory. It is described below.

**And the census entry now loses on memory, which reverses a figure this
document previously published.** Round 3 is the first run that metered *both*
sides of the census path in one region, and it reads **4,512.0 KB for binCV
against 3,072.0 KB for StereoBM — OpenCV smaller by 1.47×** (`cudaMemGetInfo`,
64 replicas a side, 141 against 96 of the meter's 2 MB units). Re-read at 256
replicas in the one-region block below it is 4,504.0 KB against the same
3,072.0 KB — 1.466×, the 8 KB difference being one meter unit of rounding at
the lower count. The figure it
replaces — 6.0 MB against 10.0 MB, "1.7× smaller" — was not a both-sides
reading on one region, and it should not be quoted again.

**The binary entry's memory lead has now been retaken, and it is 6.857×.**
It was previously stated as *unretaken* rather than quoted at 5×, because the
StereoBM figure under it came from a different region at a different replica
count than the one the census path was read on — 10.0 MB in one round against
3.0 MB in another, the same library on the same meter, 3.3× apart. All three
working sets are now metered **in one process, on one region, at one replica
count**, by `cuda_role_benchmark stereo` (section 7b):

| working set, 752×480, 64 disparities, 9×9 | `cudaMemGetInfo`, 256 replicas | vs StereoBM |
|---|---|---|
| **binCV binary entry** (2 bit planes + map) | **448.0 KB/frame** | **6.857× smaller** |
| binCV census entry (2 wide + 2 descriptors + map) | 4,504.0 KB/frame | 1.466× larger |
| `cv::cuda::StereoBM(64, 9)` | 3,072.0 KB/frame | — |

The discipline, because the previous disagreement was entirely a discipline
problem. One meter unit is 2.00 MB, so at 256 replicas **one unit is 8.0
KB/frame**: 1.79% of the binary entry, 0.18% of the census entry, 0.26% of
StereoBM. That the rounding is small is a bound and not a proof of convergence,
so the smallest of the three is also read at 64 replicas, where a unit is 32.0
KB/frame — it reads **448.0 KB/frame at both counts, 1.0000×**. Every figure
above is identical **to the byte** in every run of it — seven independent
processes in the verification pass read 117,440,512 / 1,180,696,576 /
805,306,368 / 805,306,368 raw bytes, seven times out of seven.
The format's own arithmetic agrees from the other direction — two bit planes at
24 words of 32 bits per row plus a 752×480 byte map is 442.5 KB, against 448.0
KB metered, the 5.5 KB being the driver's rounding across three allocations.

**StereoBM's footprint is not content-dependent**, which had to be checked
rather than assumed: OpenCV's FAST sizes its output by corners *found*, so its
working set moves with the picture. StereoBM was therefore metered twice, on
the synthetic pair and on the real EuRoC pair, and reads **3,072.0 KB/frame
both times, 1.000×** — it sizes by geometry and disparity count, not by what it
finds.

binCV's two readings are `cudaMalloc` with nothing between the op and the
driver. StereoBM's is an **upper reading**: `GpuMat` pads its pitch, it may be
backed by a `BufferPool`, and anything `compute()` holds past the call is
inside the delta. So 6.857× is a **lower bound** on the binary entry's lead.

So the **binary entry leads on speed by 10.8× and on memory by 6.857×** — both
axes, one region, one meter; the **census entry leads on speed by 1.43× and
trails on memory by 1.466×**, which is a split verdict on the axis
this project breaks ties with. That is an unmade judgement and it is recorded as
one, not rounded into a headline — but the memory side of it is **not a defect
to be fixed**. Two packed descriptor images are 2,820 KB before a disparity map
exists; census expands and StereoBM does not, and no implementation of this
algorithm in this layout can undercut its own descriptors. The plane layout is
the smaller intermediate (3,217.5 KB against 3,877.5 KB) and ships, at 21.7× the
matcher time. What is unmade is whether an op may ship on those terms, not what
the number is.

**Which of these is binCV's claim, and which is the on-ramp.** Only the binary
entry rests on the representation: its caller already holds one bit per pixel,
its cost is an XOR and a population count, and its device memory is 448.0
KB/frame where StereoBM's is 3,072.0 KB/frame on the same meter, in the same
region, at the same replica count. The census entry is a **standard
stereo technique implemented in the standard way** — census *expands* data
rather than compressing it (8 bits per pixel in, 24 out), and the layout that
finally made it fast is the conventional one-word-per-pixel descriptor, not
binCV's bit-planes. It exists so a caller arriving with ordinary camera frames
has a way in, and it is competitive; it is not where the thesis pays, and
nothing here should be read as claiming otherwise.

One number makes the distinction concrete, and it has to be quoted carefully
because it is a ratio between two things that both moved. Binary does a
twenty-fourth of census's work, and the host captures that: **17× on x86-64,
7.6× on aarch64** ([stereo.md](stereo.md)). The device's binary matcher captured
**2.3×** when it spent a 32-bit `__popc` on a 9-bit window, and **13.7×** once it
started treating a word as a word — inside the host's band, and the result that
closed the issue asking for it.

**It now reads 5.4×, and nothing about the binary matcher got worse.** The
census matcher got 2.49× faster underneath it (0.94 → 0.369 ms) while the binary
matcher stayed at 0.069. Both numbers are improvements and the ratio between
them is not a measure of either. What the ratio does still say is that binCV's
structural advantage on this device is real and partly uncollected: the binary
kernel does a twenty-fourth of the work for a fifth of the time, and the
profiling section's R7 names where the rest of it is (13.7% achieved occupancy,
registers binding, `wait` the top stall at 36% with DRAM at 1.2%).

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
- **How a difference is decided, and there are three answers rather than two.**
  Every ratio here is a per-round **paired** ratio and the quoted value is its
  **median**. Two separate questions are then asked of the same rounds, and both
  answers are printed:
  - **Is the direction settled?** If no round crossed 1.00× — every paired round
    fell to the same arm — the sign of the difference is established by the
    observations, and the row is reported as a **range with its sign count**
    rather than collapsed to one number.
  - **Does the size clear the noise?** `benchmark/measure_util.hpp`'s own rule:
    the difference must exceed **the larger of** the within-run spread and the
    run-to-run scatter. Unchanged in every particular.

  A row can satisfy both, either, or neither, and "neither" is a **null result**,
  which that header is explicit is itself a result. Earlier rounds of this
  document used `PairedTiming::separated()` — arm A's *worst* sample beating arm
  B's *best* — which appears nowhere in `measure_util.hpp` or `CLAUDE.md`, is
  strictly stronger, and is vetoed by a single round slow in **both** arms, which
  is the drift the paired design exists to cancel. Separation is still computed
  and printed at every row, as a fact beside the verdict rather than as the
  verdict. The full account — the owner's reasoning for the third value, why the
  deciding quantities are factors and not percentages, and what a tie does — is
  in [How a difference is decided](#how-a-difference-is-decided).
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
| census matcher (K=24), **warp-cooperative box** | kernel | **0.369 ms** | 2.49× over the packed arm |
| census matcher (K=24), packed | kernel | 0.94 ms | 44× over reference |
| census matcher (K=24), plane layout | kernel | 7.78 ms | 5.3× over reference |
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
on device, match, download. Resident-to-resident it is **1.47× ahead** of
`cv::cuda::StereoBM` — re-taken over seven processes, 1.47× apart against a
1.25× bar, all 105 paired rounds one way, ranges disjoint in all 7 runs — having
started this work 15× behind and having been 1.29× behind as recently as the
previous round. (The 1.38× and 1.43× this document has carried at different
points are the same row on earlier sweeps; its run-to-run scatter is 1.14×, so
the three readings are one number, not three.) It is **larger** than StereoBM on the memory meter; that is
in the headline above and is not softened here.

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

**The trade, stated, and it is the caller's to make rather than ours.** Packed
costs 32 bits per pixel against the plane block's 24, so the census working set
goes from **3,217.5 KB to 3,877.5 KB** on the allocation sum — 20.5% — for a
matcher that is **21.7× faster** (plane 7.82–7.94 ms against packed 0.361–0.369
ms, three runs). The 8.55× this document previously quoted for that gap was the
plane-to-packed step *before* the warp-cooperative box arm landed; that arm made
the packed matcher 2.49× faster again, so the layout choice now costs more than
it did. Both paths ship and both are bit-exact; `denseDisparityCensus` (plane
block) remains for a caller who wants the smaller intermediate and can pay the
time, and it is documented here as a choice rather than left as a path nobody
mentions.

What remains fundamental is unchanged: census compares 24 bits per pixel pair
where SAD compares one byte. That is the descriptor's price for illumination
invariance, and it is the reason this path took three rounds to reach parity
where the binary entry — the operating point a binCV pipeline runs — was ahead
from the start.

### The warp-cooperative box matcher, and the model it refuted

The last 1.25× did fall, and it fell to a **separable box** rather than to more
tuning of the packed matcher. `denseKernelPackedWarpBox` gives one warp a run of
`33 - winW` anchor columns and a strip of output rows: one lane owns one
descriptor column and slides that column's vertical window sum in a register,
and the horizontal `winW`-wide aggregation is a compile-time binary
decomposition over `__shfl_down_sync` — **4 shuffles at winW = 9**. Zero shared
memory, zero barriers, zero scratch, 80 registers and **0 bytes spilled** in all
eight `winW` instantiations.

**Decision rule, written first:** ratio ≥ 1.50× against the shipped packed
matcher, both arms in one binary on one explicit stream, ranges disjoint,
0 B added global scratch and 0 B shared, bit-exactness non-negotiable. The
1.50× is the owner's margin over a break-even that is stated separately and not
used to derive it: at the document's own transform (0.14–0.17 ms) and matcher
figures, the *entry* breaks even at a matcher around 0.67–0.70 ms, i.e. ≈1.35×.

**Measured 2.49× [2.40–2.50], disjoint in 7 of 7 serial runs**, and 2.34–2.57×
across 27 indicative runs on three different GPU contention states — the
absolute times move by 2.5× with contention and the interleaved ratio does not.
The gate-excluded control (`winW = 19`, outside the arm's gate) reads
**1.00× [0.99–1.03]**; the in-gate control at `winW = 17` reads **3.17×**, and
is *further* from 1.00 than at 9×9 because the shipped matcher's cost is
`O(winW)` where the box arm's is not.

**The profile refuted the model this change was designed against, and that is
the more useful half of the result.** The design modelled the shipped matcher
as popcount-bound at ~42% pipe efficiency and predicted 3.6–5.7×. It is
**neither**: `math_pipe_throttle` is 3.5% of warp stall samples and DRAM is
0.72% of peak. The kernel is **latency-bound at 15.2% achieved occupancy**, and
its leading throughput throttle is `mio_throttle` at 12.4% — the load/store
*instruction* queue, the number of loads rather than the bytes.

That reading also decided what not to try. `long_scoreboard` at 2.4% said
`uint4` loads attack nothing; DRAM at 0.72% said an O(1) per-pixel-disparity
cost volume attacks nothing (and it fails the memory cap besides). What was
left was warp-cooperative aggregation.

**And the first implementation of exactly that shape measured 1.10×.** It
removed precisely the throttles it aimed at — `mio_throttle` 12.4% → 0.16%,
`math_pipe_throttle` 3.5% → 0.83% — and was barely faster, because at 254
registers the limiter had merely *moved* onto the shuffle chain
(`short_scoreboard` 22.2%) and `long_scoreboard` (19.3%) with `wait` unchanged
at 15% occupancy. The fix was the **register budget**, and the sweep found it by
*shortening* the strip from 16 to 4 — which does strictly more work per output
row, and takes occupancy from 13.7% to **40.6%**. Instruction totals confirm the
mechanism independently: **77.4 M warp-instructions against 158.5 M**, and
**3.77 M global-load instructions against 20.2 M** (2.05× and 5.36× fewer).

**Recorded negative — `__launch_bounds__`.** Forcing the register budget instead
of shrinking the state: (128,4) → 128 registers, 396 B spilled, **0.98× — slower
than the kernel it replaces**; (128,6) → 80 registers, 148 B spilled, 1.70×;
(128,8) → 64 registers, 244 B spilled, 1.59×. Against 2.53× for the same shape
with no bound.

**A test-method finding that reaches past this kernel.** The first suite for
this arm — 508 checks over the designed shape list — **passed a deliberately
broken halo**: `33 - winW` output lanes widened to `34 - winW`, which makes the
top lane of every warp aggregate one column twice instead of reaching a column
outside its warp. Byte-identical map, every shape. The cause is the test *pair*:
where the right image is an exact shift of the left, the correct disparity's
window cost is 0 and every other candidate's is hundreds, so the argmin is
insensitive to window-sum errors of a few hundred. **Every dense-disparity case
in `backends/cuda/tests/test_cuda_backend.cpp` built its pair that way** —
binary and census, plane and packed.

**Those cases now run on two contents each, and the follow-up finding is that
the binary matcher was not as exposed as the census one.** Every dense case in
that file now also runs against a right image that is a shift *plus a
deterministic perturbation*, and all of them pass — the binary word-parallel
matcher and both census matchers are bit-exact against the host on content that
is not an exact shift too. A two-mutation battery against the bit-sliced kernel
then asked whether the old content had actually been blind: a vertical window
one row short and a horizontal window one column too wide are **both caught by
the exact-shift cases as well**, so the blind spot does not transfer to this
kernel at the severity it had on the census box matcher. The reason is the cost
dynamic range — a 9×9 binary window's cost is at most 81, so a one-column error
is a large relative change, where a census window's costs run to hundreds and a
few hundred of error disappears into them. The perturbed content is
nevertheless strictly the more sensitive of the two (3,900 differing pixels
against 597 on the row-short mutation at D=64), so it stays. The box suite now
runs every shape on two contents and adds content that forces exact ties; a
mutation battery against it fires 58–140 failures on each of seven mutations
(halo width, vertical slide, entering row, decomposition offset, tie order,
packing major order, constant output), and names the three that are benign and
*why* structurally rather than papering over them. Widening the other cases the
same way is cheap and is filed.

**Scope caveat, and it stands unchanged by the result.** This is the path where
binCV has **no structural advantage**. Census *expands* data — 8 bits per pixel
in, 24–32 out — and the layout that made it fast is the conventional
one-word-per-pixel descriptor, not binCV's bit-planes, which appear nowhere in
this kernel. **That is also why this path loses on memory, and the loss is
inherent rather than a defect:** two packed descriptor images are 2 × 1,410.0 KB
= 2,820 KB before a disparity map exists, against `cv::cuda::StereoBM` matching
the 8-bit frames directly. An implementation cannot be smaller than the
descriptors the algorithm requires. The plane layout is the smaller intermediate
and is kept for exactly that caller, at 21.7× the matcher time.

What landed is a standard separable-box technique implemented in the standard
way, optimized for **completeness** on the entry a wide-input caller meets
first. A recorded negative would have been an acceptable outcome
here; it simply is not the outcome. Nothing about the library's thesis changes
either way — the claim lives in the binary entry.

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
| `threshold` → bits | 752×480 | 0.0100 ms | 0.0094 ms | **1.00× — parity** | **2.46×** smaller |
| `threshold` → bits | 3840×2160 | 0.0367 ms | 0.0272 ms | **1.34×** | not measured |
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

**`threshold` missed its own speed bar, and was optimized rather than excused.**
The rule its author wrote before measuring named the fail condition in as many
words: *slower than OpenCV by more than both printed spreads*. It was met — at
3840×2160 binCV measured **2.22× slower with 7 of 9 runs disjoint**, while the
memory bar passed at 2.46×. The project's ship rule says that combination does
not merge on the memory argument, so the op was optimized in the following
round. It now reads **1.00× / 0.91× / 0.74× across the ladder**, never above
1.00× at any geometry, with memory byte-identical at 2.46×. The row above is the
post-fix reading; the *The role bars* table carries the same numbers from the
same protocol.

The mechanism was located without a profiler, and later confirmed with one.
`cuda::threshold` is header-only — the host's own `impl::thresholdCutoff`
reduction composed with `cuda::packBits`, which is what makes its Tier 1 claim
provable rather than restated — so the kernel under the number is
`packKernel<uint8_t, GreaterEqual>`. `cuobjdump -sass` showed **184 instructions
around one LDG and one STG**: 62 IMAD, 26 IADD3, 18 ISETP, and **two software
divides** (`I2F.U32.RP → MUFU.RCP → F2I`, and a 64-bit one) which are the
grid-stride loop's `wordIdx / words` and `wordIdx - y*words`. A 2-D grid with
`blockIdx.y` as the row deletes both outright; a byte-lane arm on top reads four
pixels per lane through one 32-bit load and `__vsetgeu4`, which is what
`edgeThreshold`'s arm above already does to win 8–11× in this same family. 184
instructions became 104, with zero software divides and zero spills, and the
profiler reads the kernel moving from 67.2% SM / 19.3% DRAM — compute-bound on
its divides — to 49.4% SM / 63.8% DRAM, which is where a packer belongs.

This also settled, in the opposite direction to the one expected, a suspicion
that `cv::cuda::threshold` was anomalously slow. It was not: binCV was slow, and
the default stream was hiding it.

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
benchmark prints why. **Morphology's three arms, re-judged under the project's
own rule over seven processes each** (medians quoted 752×480 / 1920×1080, and
the bar each must clear is the larger of the within-run swing and the
run-to-run scatter):

| arm | 752×480 | 1920×1080 |
|---|---|---|
| 3×3 specialization | 0.995×, bar 2.31× — **null** | 0.894×, bar 2.74× — **null** |
| word-parallel `__brev` border | 0.735×, bar 1.95× — **null** | **0.539× (1.86× apart), bar 1.66× — a result**, 77 of 77 rounds, 6 of 7 runs disjoint |
| `andNot` fusion (GRADIENT) | 0.787×, bar 2.49× — **null** | 0.770×, bar 2.54× — **null** |

The compulsory traffic behind these is 92 KB at 752×480 — **0.0002 ms against a
~0.011 ms launch floor** — so binCV's binary morphology is launch-bound at every
frame size a vision pipeline uses, and the four nulls are a lack of resolution
rather than a negative result. What changed with the rule is one row: the
word-parallel border's 1.86× at 1080p now clears its noise, so it has a measured
speed argument on top of the correctness-surface one that already justified it
(it deletes the per-pixel border path, its divergence and its lost-update race
together). The other two remain defaults on grounds other than a measured win,
which is **an owner call and is recorded as open rather than settled**.

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
forked onto the device, each kernel bit-exact against its host twin.

**The negative this section used to lead with has been closed, and the fix was
exactly the size the defect was.** The previous round's finding was:

> The resident device frontend is 1.22× SLOWER than binCV's own CPU frontend —
> 6.462 ms against 5.313 ms, ranges disjoint in all 7 runs. Detection is 97.0%
> of the device frame, and the kernel inside it that dominates runs in one
> block at 1.29% of the SMs.

It now reads, on the same 400 real frames and the same host arm:

> **The resident device frontend is 5.62× FASTER than binCV's own CPU
> frontend** — **0.945 ms [0.918–1.065]** against **5.313 ms [5.198–5.422]**,
> ranges disjoint in all 7 runs *and globally* (device maximum 1.065 below host
> minimum 5.198). Detection is **75.2%** of the device frame, down from 97.0%.

The host arm reads **5.313 identically** to the previous round, which is what
makes this a like-for-like: the device side went 6.462 → 0.945 and nothing else
moved. Correctness over 400 frames × 7 runs: **0** corner-count, **0** position,
**0** keep-byte, **0** rotation-bin and **0** descriptor-word differences. Peak
device memory, meter 1 (allocation sum): **2,106,180 B**; meter 2 reads 4.00 MB,
which is meter 1 rounded up to the driver's granule.

**What changed, and it is two orderings.** Both of the single-block kernels this
document named are gone from the default path:

- **FAST's ordering.** The single-block bitonic raster sort is replaced by
  count → prefix-sum → emit: per-unit `__popc`, a block-local 128-wide exclusive
  scan, and a base taken from the block sums below. Blocks own a *contiguous*
  run of the raster unit index, which is the correctness requirement rather than
  a tuning choice. **No two corners are ever compared.** Off-switch ratio
  **35.19× [24.5–40.45]**.
- **The corner ordering key is the corner.**
  `key = (~responseBits)<<32 | (0xFFFF−y)<<16 | (0xFFFF−x)` — ascending
  `uint64_t` order *is* the host's `CornerStronger`, and the pack is lossless,
  so the corner comes back with three shifts. Eight bytes instead of sixteen and
  no payload array at all, which is also where the selection's memory went:
  `goodFeaturesScratchBytes(65536)` fell **1,088 KB → 576 KB, 1.89×**. The sort
  is a device-wide bitonic ladder (off-switch **2.60×**) and the spacing filter
  runs the host's own loop in chunks of 1,024 (off-switch **4.04×**).

**The critique this round answered, and the answer is the round.** The previous
round's own weak spot was written down as *"the one-block selection's cost is
inherited on trust — measure it before tuning the fused kernel."* Measured, the
selection was **98% of the operation** and the fused candidate tile was ~1%. The
selection was fixed and the tile was left alone, with its profile recorded as a
named follow-up (65.4% SM, `short_scoreboard` 79%, and a 32-way bank conflict on
its shared-tile write — see R3 in the profiling section).

### The frame, stage by stage

400 real EuRoC V1_02 cam0 frames through `backends/cuda/examples/cuda_vio_frontend.cpp`,
7 process runs, one explicit stream, CUDA events on the device side and the host
library's own clock on the host side — **two different clocks, labelled as such
at every row**. The host column is context for locating the defect; it is not a
role bar, and no `cv::cuda` whole-frontend counterpart exists to be one.

**What the current pass measured, and what it did not.** The serial pass
re-measured the frame total, the detection share and the correctness gate; it
did **not** re-take the nine-way stage split, so that table is kept as the
**previous round's** measurement and is not restated as current.

| | previous round | current |
|---|---|---|
| device frame | 6.462 ms | **0.945 ms** [0.918–1.065] |
| host frame (same arm both rounds) | 5.313 ms | **5.313 ms** [5.198–5.422] |
| detection's share of the device frame | 97.0% | **75.2%** [74.5–75.7] |
| everything except detection | 0.194 ms | **0.209 ms** |
| corner / position / keep / rotation / descriptor differences | 0 | **0** (400 frames × 7 runs) |

The non-detection total barely moved — 0.194 → 0.209 ms — which is the Amdahl
check the rule asked for: the frame fell by 6.8× because detection fell, and
nothing else was touched.

The previous round's stage split, kept for the shares it establishes:

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

**The arithmetic that settled it, and the prediction it made.** If every stage
other than detection had gone to *zero*, the frame would still have been
6.107 + 0.089 = **6.20 ms** against the host's 5.31 — no amount of work on the
other seven stages makes that pipeline beat the CPU. The same arithmetic
predicted the other direction in writing: *"a `selectKernel` using even a
quarter of the machine puts the frame near 0.9 ms — roughly 5.9× faster than the
host."*

**Measured: 0.945 ms and 5.62×.** The prediction was made before the work and is
quoted here against the outcome rather than after it, which is the only form in
which a projection like that is worth anything.

**What residency itself delivers, and it is not nothing.** Exactly **1.00
synchronize per frame** in all 7 runs; 360,960 B up against 11,536 B down, a
**31.3×** bus asymmetry with nothing frame-sized returning. Correctness across
the chain is a gate rather than a metric and it passes: 0 corner-count, 0
position, 0 `keep`, 0 rotation-bin and **0 of 222,432 descriptor words** differ
from the host, with max |Δangle| 2.384e-07 rad, identical in all 7 runs.

### The one kernel, located with a profiler rather than argued

`ncu` works on this machine now — see *The profiler pass* for how it was
enabled and what it read on every kernel in the backend. It settles in one run
what the round before it had to triangulate:

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
`compaction.hpp` already names the alternative (prefix-sum compaction).

**Both are now off the default path**, which is why the frontend row above
changed direction. `fastSortKernel` is a fallback arm only; `selectKernel` has
been split into `keyKernel`, the bitonic ladder (multi-block by default) and
`spacingKernel`. **One single-block kernel survives on a shipped path** —
`spacingKernel`, at 0.9–1.0% of the SMs, which the profiling section prices at
**37% of `goodFeaturesToTrack`** and carries as R2.

### FAST: the bar was missed, and the ordering is why — now closed on both axes

The table below is the **previous round's** measurement, kept because it is what
the fix was aimed at and because the mechanism it isolates is the whole story.
The current reading, under the serial role protocol and re-taken after the
memory round below, is **binCV 0.0247 ms against
`cv::cuda::FastFeatureDetector`'s 0.1459 — 0.166×, 6.01× faster**, disjoint in
**all 7** runs (per-process ratio medians 0.1476–0.1784), corner-set gate
19,898 = 19,898 in all 7. The memory round did not touch the ring algebra and
this is not read as a speed improvement over the 0.171× it replaces; it is read
as no regression.

Two arithmetic gates written with it also hold: the replacement runs
detection twice and performs no comparison, so its predicted cost was ≤ 0.10 ms
(measured 0.024), and the **complexity gate** — cost must stop tracking
`nextPow2(found)` — read **1.01×** between capacity 512 and the full pool over
identical detection work when it was taken as a paired in-process measurement.
It used to be 2.12×. The memory round did not touch that mechanism, and the
seven role runs above agree from the other side: binCV is **0.0247 ms at
capacity 32,768 against 0.0254 ms at 512**, so the full pool is if anything the
cheaper of the two.

**The memory half has now moved, and it flips the row.** It used to read 1,088.0
KB against OpenCV's 704.0 KB — 1.545× larger — and the reason was two allocations
that the shipped arm never touched. `fastScratchBytes` returned the *reference*
arm's number to every caller, so a frame that used 380 B of block sums was handed
`nextPow2(capacity)` corner records: **512 KB of scratch to use 380 bytes of it**.
And `DeviceFastCorner` carried the host's `long long` score, which cost the
record 8 bytes for the field plus **4 bytes of padding** to align it after `y` —
12 of the 16 bytes spent on a value that is an arc length around a 16-pixel ring
and cannot leave [1, 16].

Both are fixed. The sizing function takes the **arm** as an argument, and
`detectFastAsync` validates against the arm it is *about to run* rather than the
one the caller sized for, so a buffer sized for one arm and met with the other is
`cudaErrorInvalidValue` and not a device-side write past its end. The record is
12 bytes: positions stay `int`, the score narrows to `int32_t`, and
`DeviceFastCorner::toHost` widens it, so what a caller reads back is unchanged.

**A narrowing is a correctness claim, so it is proved rather than argued.** The
score's attainable set is swept **exhaustively** in the suite — all 65,536 ring
patterns at all sixteen arc lengths, round-tripped through `toHost` — and the
observed range is pinned at exactly [1, 16]. Verification added a second check
the suite does not make, on **real frames**: two EuRoC frames through the sensor
stage at two thresholds, seven arc lengths and all eight arm combinations, 224
complete runs, **13,469,376 corner records `memcmp`'d against the host
detector's own array** on the host type — positions, raster order and the widened
score — with zero differences, and with the score's *whole* range [1, 16]
actually occurring in the data rather than assumed to. The sizing refusal was
checked the same way, since `compute-sanitizer` does not run on this host: a
buffer sized for the ordered arm, met with the reference arm, inside a poisoned
1 MB guard region either side. The call returns `cudaErrorInvalidValue`,
**disturbs zero poisoned bytes** in either guard or in the scratch window, leaves
the output buffer bit-identical to its own poison and the counter at 0 — and the
ordered arm then runs against that same 380-byte buffer, touches neither guard,
and finds the same 19,898 corners.

| per working set, capacity 32,768 | before | after |
|---|---|---|
| bit plane, 24 words × 480 rows × 4 B | 46,080 B | 46,080 B |
| corner array | 32,768 × 16 = 524,288 B | 32,768 × **12** = 393,216 B |
| scratch | nextPow2(32,768) × 16 = 524,288 B | 95 blocks × 4 = **380 B** |
| allocation sum (meter 1) | 1,069.0 KB | **429.4 KB** |
| **`cudaMemGetInfo`, 256 sets (meter 2)** | **1,072.0 KB** | **440.0 KB** |

**Both cells of that meter row are one harness at one replica count**, which is
the only way a before and an after are comparable. HEAD's own benchmark reads its
side over 32 sets and this branch's over 256, so neither could supply both; the
row is a verification probe that allocates each shape 256 times, and it
reproduces HEAD's benchmark exactly (1,088.0 KB) when told to use 32. The two
meters are kept apart and never divided into each other: they agree to 3.0 KB on
the before and 10.6 KB on the after, inside one and two of the driver meter's own
units at this count.

**Against OpenCV, on that same meter, both sides taken identically** — this is
the cross-library figure and it comes from the committed `cuda_role_benchmark`:
**binCV 432.0 KB against `cv::cuda::FastFeatureDetector`'s 680.0 KB — 1.574×
smaller**, identical in five consecutive processes (54 and 85 of the meter's
2 MB units).

**The two harnesses do not read the same absolute numbers, and that is worth
knowing.** The same probe run against *both* sides in a fresh process reads
**440.0 KB against 704.0 KB — 1.600×**, identical in seven processes: one unit
high on binCV's side and three on OpenCV's compared with the benchmark. The
reason is that a delta taken late in a long benchmark reads low — `cudaFree`
returns memory to the driver in whole units, so allocations that land inside a
unit the process already holds cost this meter nothing. **The quoted
cross-library figure is the benchmark's**, because it is the committed,
reproducible one and also the more conservative of the two. The probe is a
verification cross-check rather than a committed benchmark, and its numbers are
recorded here as corroboration, not as the claim. Both readings give the same
sign, the same magnitude and a ratio within 1.7% of each other.

**The replica count is part of that reading, and saying so is not a footnote.**
This row was previously taken over 32 working sets and then 64. Both clear the
eight-unit rule this document quotes — but clearing it is not the same as
resolving. At 64 sets binCV's side reads 13 units, so one unit of rounding is
32 KB a frame on a 430 KB reading, and two harnesses that both cleared the rule
quoted **1.615× and 1.714× for the same pair**. At 256 sets the unit is 8 KB a
frame and the reading stops moving in either harness. The benchmark now takes it
there.

**One behaviour moved that nobody asked to move.** `fastOrderedApplies` compares
the two arms' working areas, and its left side carries `sizeof(DeviceFastCorner)`
— so narrowing the record shifted that comparison by a quarter. At 752×480 the
crossing is capacity 17 either way and nothing moved at the reference geometry;
over a sweep of widths 32..2016 and heights 7..1199, **about 4.5% of (frame,
capacity) points now take the reference arm where they took the ordered one**,
always in that direction and never above a capacity of 128. No answer changes —
both arms are bit-exact and both are cheap at those counts — but the crossover is
a **heuristic rather than a derived number**, and the proof of that is that it
moved when a record size did. Reading a byte count as a work unit is what is
wrong here; a measured crossover is filed rather than invented.

**What the narrowing did NOT buy, stated because the profile says so.** A 12-byte
record at 4-byte alignment is three scattered 32-bit stores where a 16-byte record
at 8-byte alignment packed into wider ones, and `ncu` reads `fastEmitKernel<9,1>`
at **53,949 global store sectors against the old record's 37,931 — 42% more L1
store traffic for 25% fewer bytes actually reaching DRAM** (1.73 MB through L1,
DRAM throughput 6.8% against 8.4%). It costs nothing today: the kernel's duration
is unchanged (7.74–7.84 µs on the profiler's clock, which is not a timing number),
SM throughput 19.6–19.9% either way, achieved occupancy **15.50% against 15.51%**
with registers and shared memory both capping the launch at 10 blocks an SM, and
the stall histogram moves by less than its own sampling noise at ~429 samples
(`wait` 18.9% → 23.5%, `imc_miss` 16.1% → 14.7%, `barrier` 12.9% → 16.1%). **The
limiter did not move: it is occupancy in both, which is what issue #64 is about.**
The store scatter is invisible at 15% occupancy and would stop being invisible if
#64 succeeded, so it is recorded here rather than discovered then.

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

**The locator, and why the ordering was the whole gap:** the same detector with
its store capped at 512 ran at **0.0303 ms against OpenCV's 0.1721 ms — 5.2×
faster**, every pixel still tested. The ring algebra cleared parity roughly 5×
over. The raster sort on top of it was **98.6% of the operation** — and removing
it recovered exactly that. It is the strongest form this kind of prediction can
take: the locator named a share before the fix, and the fix delivered it.

### The role bars

Both arms on **one explicit stream**, medians of 7 process runs. The stream is
not a detail: OpenCV synchronizes the whole device on the default stream, a
surcharge measured up to 7.18× here, and a default-stream pair in
`cuda_sensor_benchmark` that printed "7.35× FASTER" for `threshold` has been
deleted rather than repaired — `cuda_role_benchmark` owns that comparison.

The **verdict** column below is the *published* one, set when these rows were
taken under the range test. The rightmost column is what the project's rule says
on a seven-run re-take of every one of them, in the three values the owner's
2026-09-19 ruling defines — *direction established* (no round crossed 1.00×),
*a result* (the size clears the larger noise), or neither
(see [the judgement the ruling settled](#the-judgement-the-ruling-settled)).

| operation | `cv::cuda` arm | OpenCV | binCV | ratio | disjoint | verdict | re-taken: apart vs bar |
|---|---|---|---|---|---|---|---|
| `threshold` 752×480 | `cv::cuda::threshold` | 0.0091 ms | 0.0084 ms | 0.992 | 0/7 | **PASS** | 44–60 with one round tied — null |
| `threshold` 1920×1080 | ″ | 0.0116 ms | 0.0100 ms | 0.889 | 0/7 | **PASS** | 17–88 — null |
| `threshold` 3840×2160 | ″ | 0.0362 ms | **0.0268 ms** | **0.743** | 1/7 | **PASS** | 8–97 — null; the 1.35× stays unquotable |
| describe, N=1000 | `cv::cuda::ORB::computeAsync` | 0.1070 ms | 0.0107 ms | **0.108** | **7/7** | **MET, 9.3×** | 9.25× vs 2.10× — **a result** |
| FAST | `FastFeatureDetector` | 0.1459 ms | **0.0247 ms** | **0.166** | **7/7** | **MET, 6.01× — was 14.84** | 6.01× vs 3.39× — **a result** |
| `goodFeaturesToTrack` (wall) | `createGoodFeaturesToTrackDetector` | 3.7282 ms | **0.8405 ms** | **0.226** | **7/7** | **SHIPS, 5.3×** | **105 of 105, by 3.53× to 15.52×** — direction established **and** a result (4.42× vs 3.16×) |
| min-eigenvalue response | `createMinEigenValCorner` | 0.0515 ms | 0.0590 ms | 1.146 | **0/7** | **still not a result** | 82–23, 23 rounds crossed — null on both halves |
| Lucas-Kanade, 204 pts | `SparsePyrLKOpticalFlow` | 0.1475 ms | **0.0792 ms** | **0.537** | **7/7** | **MET at frontend density** | **faster in 105 of 105 rounds, by 1.35× to 4.36×** — direction established; magnitude a null (1.84× vs 2.49×) |
| Lucas-Kanade, 2048 pts | ″ | 0.3287 ms | 0.3558 ms | 1.080 | **0/7** | **not a result — the crossover** | 98–7, seven rounds crossed — null on both halves |
| descriptor matching, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 1.9491 ms | **0.2189 ms** | **0.109** | **7/7** | **MET, 9.1×** | 9.11× vs 1.29× — **a result** |
| census matcher | `cv::cuda::StereoBM(64,9)` | 0.6864 ms | **0.3692 ms** | **0.536** | **7/7** | **MET, 1.87×** | 1.94× vs 1.37× — **a result** |
| census entry (2 transforms + match) | ″ | 0.6996 ms | **0.5076 ms** | **0.723** | **7/7** | **MET on speed, LOSES on memory** | 1.47× vs 1.25× — **a result** |
| binary entry | ″ | 0.7152 ms | **0.0648 ms** | **0.091** | **7/7** | **MET, 11.0×** | 10.95× vs 1.78× — **a result** |
| block matching | `SparsePyrLKOpticalFlow` | 0.2320 ms | **0.0540 ms** | **0.230** | **7/7** | speed MET; **accuracy floor unset** |

**Three rows changed direction since the previous round, and one row is new
information rather than a better number.** FAST went from 14.84× behind to 6.01×
ahead and `goodFeaturesToTrack` from 2.659× behind to 5.3× ahead, both on the
ordering rewrites described above. The LK row is the first tracking measurement
in this backend. And the census entry's row is a **split verdict** — ahead on
speed, behind on memory — which is stated in the headline and not resolved here.
FAST was the other split verdict and is no longer one: the memory round below
takes it to 1.574× smaller, so it leads both axes.

**Two rows measured again for the re-judging pass came back differently enough to
record, and both are reported rather than folded into the numbers above.**

- **`goodFeaturesToTrack`'s wall-clock row was never being judged at all.** Its
  paired summary was assembled by hand rather than through `summarizePaired`, so
  its sign counters stayed at zero and the row printed a 0–0 split with p = 1
  over rounds that are in fact unanimous. Judged properly over seven fresh
  processes it is **3.7282 ms against 0.8405 ms, 105 of 105 rounds, 3.53× to
  15.52×** — direction established and a result. The bug is fixed; the row above
  keeps its published verdict and this is the first honest sign count for it.
- **FAST's 6.01× is not a stable figure on this host, and the direction is.**
  Over 21 independent processes (seven here, fourteen in the re-judging sweep)
  the per-run median ratio moves between **3.44× and 6.26×**, because binCV's own
  arm is bimodal: it lands at either ≈0.0205 ms or ≈0.0310 ms depending on the
  process, with `cv::cuda`'s arm steady at 0.10–0.14 ms. **Every one of those 21
  runs is 15–0 in binCV's favour**, so the lead is not in question and its size
  is quoted more precisely above than this machine supports. Flagged rather than
  silently re-numbered: fixing it is a re-take round of its own.

**`goodFeaturesToTrack`'s OpenCV arm remains noisy** (3.24–13.61 ms across runs)
because its spacing filter runs on the CPU; that is stated rather than hidden,
and it is why the row is reported against round 2's own quieter 3.55–4.28 ms
denominator as well. binCV crosses parity against either.

Memory, `cudaMemGetInfo` delta taken identically **on both sides** — the only
meter readable across libraries, replicated until each total clears eight of
this driver's 2 MB units, and never mixed with binCV's own allocation sums:

| operation | binCV | OpenCV | ratio |
|---|---|---|---|
| `threshold` | 416.0 KB | 1024.0 KB | **2.46× smaller** |
| describe, N=1000 | 48.0 KB | 2048.0 KB | **42.67× smaller** |
| descriptor matching, 5000² | **400.0 KB** | 8277.3 KB | **20.7× smaller** |
| LK tracker resident state | **448.0 KB** | 1408.0 KB | **3.14× smaller** |
| `goodFeaturesToTrack` | 1920.0 KB | 10240.0 KB | 5.33× smaller |
| corner response | 2048.0 KB | 10240.0 KB | 5.00× smaller |
| FAST @ capacity 32,768 | **432.0 KB** | 680.0 KB | **1.574× smaller** — was 1.545× LARGER |
| census entry working set | 4512.0 KB | 3072.0 KB | **0.68× — binCV is 1.47× LARGER** |

**One of these rows goes the other way, and it is printed as it reads.** The
helper that prints them used to say "binCV smaller by that factor" whatever the
ratio was — which is a claim rather than a reading on any row where OpenCV is
smaller. It now follows the number. FAST was the second such row and no longer
is; the census entry remains one, and its loss is **inherent to census rather
than a defect in this implementation** — see below. The census-entry row is the
first time both sides of that path were metered in one region.

**FAST's row is now read over 256 working sets a side — this file's own default
`kReplicas` — where it was the outlier at 32.** That is not a different meter,
only a finer one, and the change is stated because the row moved by 6% between
two harnesses that both satisfied the eight-unit rule at 64.

### The judgement the ruling settled

Replacing the range test with `measure_util.hpp`'s own rule was expected to
*unblock* effects the range test had vetoed, and it did — one, the word-parallel
morphology border. Re-taking **every** published role bar under the same rule
turned up the other direction too, and one row turned up something the rule as
then written could not express at all.

**The row was Lucas-Kanade at the frontend's own keypoint spacing.** Every one of
its paired rounds favours binCV, all seven runs are range-disjoint, and the two
arms were **never once seen closer than 1.35× apart**. Yet the per-round ratio
swings 2.49× across those rounds — because the rounds where binCV wins by 4.4×
sit further from 1.00× than the rounds where it wins by 1.35× — and a difference
of 1.84× does not exceed 2.49×. So the rule, read literally, called *"the two
arms are the same speed as far as this run can tell"* on a pair this machine
never once saw level.

**The owner ruled on 2026-09-19: the spread bounds the magnitude, not the
direction.** When no observed round crosses 1.00×, the sign of the difference is
established by the observations themselves, and the spread tells you only that
the *size* of the win varies. The row is then reported as a range with its sign
count, rather than forced into a single verdict:

> **binCV is faster in 105 of 105 paired rounds, by 1.35× to 4.36×**
> (median 1.84×; `cv::cuda` 0.1475 ms against binCV 0.0792 ms).

Two things the ruling deliberately does **not** do. It does not relax the
noise test — `differenceClearsNoise` is unchanged, and re-judging 146 published
rows moved none of them across the RESULT/NULL line in either direction. And it
does not introduce a minimum round count: two unanimous rounds and a hundred
unanimous rounds both satisfy "no round crossed", so what separates them is the
exact two-sided sign-test p printed beside every verdict — 4.9×10⁻³² for the row
above. A round-count floor would have been a project-wide "X is enough" bar
invented on the spot, which `CLAUDE.md` forbids.

**Where the ruling landed, over a seven-process re-take of every role bar:**

| row | `cv::cuda` vs binCV, per-run medians | rounds won by binCV | now |
|---|---|---|---|
| **LK at the frontend's spacing (204 pts)** | 0.1475 ms vs **0.0792 ms** | **105 of 105** | **direction established, 1.35×–4.36×**; magnitude a null |
| LK @256 | 0.1467 ms vs **0.0728 ms** | **105 of 105** | direction established, 1.43×–5.65×, **and** a result at 2.00× |
| LK @512 | 0.1634 ms vs **0.1056 ms** | **105 of 105** | direction established, 1.13×–3.80×; magnitude a null |
| `threshold` 3840×2160 | 0.0362 ms vs **0.0268 ms** | 97 of 105 | still a null on both halves |
| `threshold` 1920×1080 / 752×480 | 0.0116 / 0.0091 ms vs 0.0100 / 0.0084 ms | 88 / 60 of 105 (one tied at 752) | still a null on both halves |

The `threshold` rows cost nothing to restate: that op's written rule was a
*fail* condition — slower than `cv::cuda::threshold` by more than both spreads —
and a null result is not slower. **`PASS` there means "no longer fails", and it
still does.** What should stop being quoted is the *magnitude*: 1.35× at 4K is a
median this host's noise does not resolve, and the ruling does not rescue it
because eight of its 105 rounds fell the other way.

**What the ruling settles is how a row is reported, and a narrower question is
left where it was.** `CLAUDE.md`'s ship rule asks whether an operation "holds up
on both axes", which is worded as a pass/fail. A direction-established row is
neither a pass nor a null — it is a range — so whether the label **MET** attaches
to LK's role bar is an owner's call and not a measurement's. What is *not* in
doubt is that the row does not fail: binCV is ahead in every round measured, in
two independent sweeps, and it is **3.14× smaller** on resident state. The row is
therefore published as the comparison, and the label is left as it stands.

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
not in bits. The **census entry** is a fourth and is covered in its own section
above: it is ahead on speed and behind on memory, and the memory side of that is
the algorithm's, not the implementation's — census expands 8 bits a pixel into a
32-bit descriptor word, so its two transformed images are 2,820 KB before
anything else, where `cv::cuda::StereoBM` matches the 8-bit frames directly.
**FAST is not on this list and used to be**: it leads both axes as of the memory
round recorded above.

**The corner response.** `cornerMinEigenValAsync` reads **1.131×
[0.554–1.613]** against `createMinEigenValCorner` with **0 of 7 runs disjoint** —
not a result in either direction, for the second round running, and reported as
one that did not resolve rather than rounded to parity. The median moved from
1.292× to 1.131× and the *verdict* did not.

Its rule for this round said: read the profile first; at ≥80% of either roof,
record a negative and do not optimize; below both, the stall histogram names the
limiter and **one** change addresses it. Neither arm is at 80% of either roof, so
the histogram decided, and it names **occupancy** rather than arithmetic:
`responseKernelWindow` 78.6% SM / 3.6% DRAM with `short_scoreboard` 40% + `tex`
25%, `responseKernelSliced` 69.3% / 3.4% with `short_scoreboard` 69% at **16.0%**
achieved occupancy. The control settles it — OpenCV's `cornerMinEigenVal_kernel`
runs the same shape at **87.5% achieved occupancy**. For the sliced arm
`launch__occupancy_limit_shared_mem` binds at 5 blocks/SM against the registers'
7, so shared-memory capacity is the binding resource and the grid is only 90
blocks on 48 SMs. The one change taken was `responseKernelSliced` staging
through shared memory on a 33-float pitch, which took its global store sectors
from 360,960 for a 1.44 MB map (8× write amplification) to **45,120 — exact —
and its local loads to zero**. It did not move the verdict. The response is a `sqrt` and two products per *pixel* in
`float`; the covariance feeding it is bit-work, but the response on top of it is
float arithmetic of exactly the shape a byte pipeline already does well. The
host library's own 4.81× for a bit-sliced blockSize-3 response **does not port**:
on device it measures 0.74× — slower, ranges disjoint — because the host's win
was removing per-pixel *addressing*, while the device form hands one thread 32
pixels of `sqrt`, i.e. 32× less parallelism on the float half. That is a
negative result and is printed as one; the default is unchanged.

**Sub-pixel refinement.** `cornerSubPixAsync` **still has no verdict against its
own round-trip rule**, and that is the honest outcome rather than a placeholder.
The rule asks whether refining resident beats downloading the derivative planes
and refining on the host, against the tighter of two baselines.

It is now measured as **one paired comparison** instead of three separately
timed medians added together, because adding three medians carries all three
passes' drift and offers nothing to decide with — which is why it has been
reported as "met in 6 of 7 runs", where the seventh is not a different machine
but the same three passes landing differently. The host term alone swings
**0.280–0.437 ms** across seven runs. Both arms are now on the wall clock, since
a CUDA-event bracket around a round trip times the copies and silently drops
the host refinement, which is its largest term; both are spelled exactly as the
inequality is written, down to the 1.6 KB transfer that sits on each side.

**The tighter baseline is the whole-plane download, and that is itself a
finding**: 200 pitched 11-row copies cost 10.7 ms against 0.21 ms for one copy
of four planes, so the narrow form of the round trip is 50× the wide one on
this machine and pairing against it would be measuring against a spelling
nobody would keep.

Against the right baseline, laid out as measured over seven processes: the
**device arm reads 0.5656 ms [0.5547–0.6051]** and the **whole-plane round trip
0.3911 ms [0.3790–0.5728]**, with **72 of 77 paired rounds favouring the round
trip**. That is 1.44× apart against a bar of 1.39× (within-run swing 1.37×,
run-to-run scatter 1.39×), so in this sweep it is **a result against the device
arm**; over a separate fourteen-process sweep the same pair read a null at
parity, five rounds the other way. The two readings differ on *how far* the
device arm is behind, not on which side it is on. The op's written ship
condition is *strictly cheaper*; it is not met. It was 3.6× adrift two rounds ago and missed by 1.08× one round
ago. **This contradicts the previous round's "met by medians at 1.07×" and is
reported rather than adjusted**, per the stop-and-ask rule.

**The profiler names the limiter and it is a ceiling of the signature.** The
kernel runs 7 blocks × 32 threads at **2.1% achieved occupancy** against a 33.3%
theoretical ceiling; no local memory, no spills, and `math_pipe_throttle` does
not appear at all. 200 corners at one thread per corner is 6.25 warps on a part
that holds 2,304 — more blocks cannot help, because there are only 7 warps of
work. The one decomposition that would add parallelism, a warp per corner, is
the one `subpix.hpp`'s bit-exactness argument forbids: double addition is not
associative. **About 450 corners are needed before a second warp per SM exists
at all.** The spread arm that does ship (one warp per block instead of 256
threads) is 3.20× over the packed arm, and the profile explains the gap between
that and the predicted ÷B exactly: `sm__inst_executed_pipe_fp64` reads 45.65% of
peak, and 7 × 0.457 = 3.20 against a measured 3.18.

It has no `cv::cuda` counterpart at any API level, so it carries no speed bar.
The honest reading is that **below roughly 450 corners a caller should refine on
the host**, and that is a documentation statement somebody has to make rather
than a measurement — it is filed as one.

**Its set-bit-skip control is reading as a red flag and has been for two sweeps,
so it is recorded here rather than left in a log.** The control sets every
magnitude bit, so the skip removes no work and the two arms are meant to be the
same kernel doing the same thing; the harness expects ~1.00× and prints a warning
if identical code does not scatter both ways. It does not: **skip-ON is 4%
slower in 77 of 77 rounds here and in 154 of 154 across a second sweep**, ranges
disjoint in every run. 4% is small and the ratio is inside the ±5% band the
control asserts, so nothing fails — but a one-sided sign count on code that is
supposed to be identical is a finding about the skip's own test, not about noise.
It is the only control in this backend that does this; every other gate-excluded
pair splits its rounds roughly evenly. **Reported, not chased**: it does not move
a published number, and running it down is its own round.

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

Five more joined the list, each with the reason it has no `cv::cuda` bar:
`stereoDescriptorMatch`, `stereoRefineDisparity` and `stereoMatchRectified`
(OpenCV's sparse stereo is `StereoBM`'s dense map plus a host lookup, not an
operation), `calcOpticalFlowBlockMatch` (`cv::cuda::FastOpticalFlowBM` is a
dense field, not a sparse tracker) and `matchDescriptorsGated` (no library
exposes a gated matcher). **No CPU number is quoted in place of a missing GPU
one anywhere on this list.** Two of them carry a further condition that is not a
speed verdict and is not treated as one: their accuracy floors are unset, so
they are reported with their accuracy in plain sight rather than claimed.

**Descriptor matching has left this list, because it now has a device arm and
therefore a real bar.** It was recorded here as the reverse of an OUTSTANDING —
OpenCV had `BFMatcher(NORM_HAMMING)` and binCV had nothing to time. It is timed
above at **0.109× against `knnMatchAsync(k=2)` on `CV_32S`, 20.7× smaller**. The
note that used to sit here predicted the win would come from word emission —
8 `__popc` per 256-bit descriptor against `cv::cuda`'s 32 at `uchar`. **That
prediction was measured and is wrong**: the instruction ratio is real and worth
1.02×. The lead is kernel shape. The prediction is recorded here as made and
falsified rather than quietly replaced by the result.

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

### The harness limit that decided more than it should — resolved, and what it did not change

Earlier rounds of this document recorded an open question: three effects had
medians far from 1.00 with every run one-sided and only a minority of runs
range-disjoint — the row-grid packer arm, the fused `derivativeXY` and the
bit-plane orientation arm — and `PairedTiming::separated()`, a min/max range
test, called all three "not a result".

**The question is answered, and not in the direction that would have been
convenient.** The range test was never this project's rule: `measure_util.hpp`
asks for a difference-against-spread test on medians, against the larger of the
within-run spread and the run-to-run scatter, and `max(A) < min(B)` appears
nowhere in it or in `CLAUDE.md`. It is strictly stronger, and it is vetoed by a
single round that is slow in **both** arms — which is drift, the exact thing the
paired design cancels. The harness now applies the project's rule (see
[Setup](#setup)), and separation is printed beside every verdict as a fact.

**Adopting the correct rule did not turn those effects into wins, and neither
did the 2026-09-19 ruling that added the direction verdict on top of it.**
Re-measured over seven independent processes each, with all three values
available:

| effect | geometry | median | apart | within-run swing | run-to-run | sign | verdict |
|---|---|---|---|---|---|---|---|
| packer row grid | 1920×1080 | 0.704 | 1.421× | 1.966× | 1.051× | 6–99 | **null result** |
| packer row grid | 752×480 | 0.911 | 1.098× | 2.077× | 1.045× | 17–86 | null result |
| packer row grid | 3840×2160 | 0.826 | 1.211× | 1.742× | 1.010× | 9–96 | null result |
| `packQuant` row grid | 752×480 / 1080p / 4K | 0.842 / 0.748 / 0.841 | 1.188 / 1.338 / 1.189× | 1.934 / 1.914 / 1.639× | ≤1.13× | 13–92 / 7–98 / 6–99 | null result |
| morphology 3×3 specialization | 752×480 | 0.946 | 1.057× | 2.719× | 1.138× | 20–56 | null result |
| morphology 3×3 specialization | 1920×1080 | 0.847 | 1.181× | 2.041× | 1.141× | 8–69 | null result |
| **morphology word-parallel border** | **1920×1080** | **0.476** | **2.099×** | **1.951×** | **1.150×** | **1–76** | ***a result*, but the direction is NOT established — one round in 77 crossed** |
| morphology word-parallel border | 752×480 | 0.717 | 1.394× | 1.542× | 1.075× | 5–72 | null result |

**One verdict changed and the rest did not**, which is the useful thing to be
able to say about a rule change: it did not act as a way to manufacture wins.
The ruling did not manufacture any either — **not one of these rows is
direction-established**, because every one of them has rounds falling both ways.
The 1920×1080 border arm is the closest, and one crossing round in seventy-seven
is enough to withhold it.

**Why the internal arms stay null is measurable rather than arguable.** The
row-grid arm's seven per-run medians at 1920×1080 are 0.7005, 0.7344, 0.7084,
0.7036, 0.7039, 0.7022, 0.7366 — a 1.05× scatter across processes — and 99 of
105 paired rounds fall the same way. Yet the *within-run* swing of the per-round
ratio is 1.97×, and the rule takes the larger of the two noises. That larger one
is within-run: it is this WSL2 host's scheduling on batches of a few hundred
microseconds, not the kernels. Shrinking it means more enqueues per round, which
would move every number in this document — a change of its own, not a correction
to this one.

So the arms remain as they were: the row-grid packer and the 3×3 morphology
specialization are defaults on grounds other than a measured speed win, and the
word-parallel border now has one at 1080p on top of the correctness-surface
argument that already justified it.

## Tracking on device

Lucas-Kanade, the whole pyramid ladder and the tracker's derivative planes,
resident. One launch per frame: the level loop runs **inside** the kernel, one
warp per keypoint, lane *l* owning window rows *l* and *l + 32*. Level
descriptors ride in the 4 KB parameter space. `-Xptxas -v`: **124 registers at
N = 2, 103 at N = 1, 0 bytes spilled, 0 shared memory**, 2,624 B of constant
bank.

Every rule below was written before the first measurement and is printed by
`cuda_opticalflow_benchmark` ahead of the first number.

### The sequence-level number, which is what the issue asked for

A per-frame kernel ratio is not a tracking result, so this row is a whole
sequence: **400 real EuRoC V1_02 cam0 frames**, sensor stage → the shipped
1/2/2/2 ladder → the previous frame's derivatives → LK track, at a fixed
re-detection cadence. Wall clock on both arms, **one stream synchronize per
frame inside the device arm's clock**, and the H2D upload of every frame inside
it too. 3 passes per process × 7 processes.

Agreement is checked before any timing, in every run: level-0 binary frame
**11,520 words compared, 0 differ**; the `dx0`/`dy0` ternary planes **46,080
words compared, 0 differ**; the detector **204 = 204** corners with **0**
position differences; tracked counts equal on every frame.

| cadence | HOST binCV | DEVICE binCV | ratio | ranges |
|---|---|---|---|---|
| re-detect every 10 frames | **1.932 ms/frame** [1.779–1.992] | **0.424 ms/frame** [0.383–0.471] | **4.56× faster** | disjoint 7/7 **and globally** |
| re-detect every frame | **6.455 ms/frame** [6.388–6.652] | **1.056 ms/frame** [0.974–1.159] | **6.11× faster** | disjoint 7/7 **and globally** |

"Globally" means the device arm's slowest sample is below the host arm's fastest
across all runs, not merely that each run separated.

**Two different clocks.** The device column is wall clock around a stream
synchronize; the host column is the host library's own clock on the same
machine. The host x86 arm is not timing-grade here, so this row locates a
magnitude, not a decimal.

**Peak device memory, reported with the speed and not after it** — identical in
all 7 runs, and peak equals resident because every allocation is made once at
construction and the per-frame loop calls no `cudaMalloc`:

- meter 1, allocation sum, whole resident state including the detector pool and
  the sensor stage's wide frames: **2,212.7 KB**
- meter 1, tracker alone (two ladders, previous derivatives, keypoint arrays):
  **451.7 KB**
- meter 2, `cudaMemGetInfo` over 32 independent tracker states: **2,240.0
  KB/tracker** (35 of the meter's units — it resolves)
- the host arm: **779.0 KB of HOST bytes — meter 3, a different meter**, never
  divided into the device figure.

### The role bar, and the density at which it stops holding

`cv::cuda::SparsePyrLKOpticalFlow`, **both sides' pyramids resident** — `calc()`
takes `std::vector<GpuMat>` and skips `buildImagePyramid`, so this is a
like-for-like row and not a subtraction estimate — one explicit stream, 752×480,
4 levels, **the same 31×31 window on both sides**, iteration cap 20, `err` off
both sides, both free-running. The ship rule as written asked for **strictly
faster AND sample ranges disjoint**; separation is no longer the verdict
anywhere in this document, so the rows below are judged by the project's rule
and the disjointness is reported as the fact it is.

Both sides are laid out as measured, over seven independent processes (15 paired
rounds each, 105 in all per row). The per-round comparison is beside them, not
in place of them.

| keypoints | `cv::cuda` per-run medians | binCV per-run medians | rounds binCV won | how much it won by | median |
|---|---|---|---|---|---|
| 64 | 0.1396 ms [0.1249–0.1610] | **0.0523 ms** [0.0522–0.0535] | **105 of 105** | 1.77× to 10.95× | 2.64× |
| 128 | 0.1412 ms [0.1283–0.1496] | **0.0548 ms** [0.0542–0.0586] | **105 of 105** | 1.78× to 4.89× | 2.46× |
| **204** (the reference spacing) | 0.1475 ms [0.1314–0.1700] | **0.0792 ms** [0.0787–0.0794] | **105 of 105** | **1.35× to 4.36×** | 1.84× |
| 256 | 0.1467 ms [0.1370–0.1782] | **0.0728 ms** [0.0718–0.0764] | **105 of 105** | 1.43× to 5.65× | 2.00× |
| 512 | 0.1634 ms [0.1347–0.1844] | **0.1056 ms** [0.1046–0.1059] | **105 of 105** | 1.13× to 3.80× | 1.55× |
| 1024 | **0.1883 ms** [0.1860–0.2005] | 0.2020 ms [0.1993–0.2031] | 31 of 105 | — rounds fell both ways | 1.07× *against* binCV |
| 2048 | **0.3287 ms** [0.3264–0.3316] | 0.3558 ms [0.3533–0.3586] | 7 of 105 | — seven rounds crossed | 1.08× *against* binCV |

**Every row down to 512 keypoints is direction-established**: not one of the 105
paired rounds at those densities fell to `cv::cuda`, two-sided sign-test
p = 4.9×10⁻³². At 204 and 512 the *magnitude* is still a null — the win varies by
more than it averages, which is what the range column is for — and the ruling
that separates those two statements is recorded
[above](#the-judgement-the-ruling-settled).

**The sweep finds where it stops.** Above roughly 512–1024 keypoints binCV is no
longer ahead, rounds fall both ways, and neither direction is a result there. The
header names the density, because the mechanism says it must: one launch with one
warp per keypoint beats six launches with 256 threads per keypoint only while the
*launch shape* dominates, and the crossover is where it stops dominating.

**Almost all the noise on these rows is `cv::cuda`'s, not binCV's.** Across the
seven processes binCV's own median moves by 1.007× at 204 points and
`cv::cuda`'s by 1.29×, so the swing the magnitude verdict is charged for is
uncertainty about how far *ahead* binCV was, not about whether it was.

**Memory: 448.0 KB against 1,408.0 KB — 3.14× smaller.** `cudaMemGetInfo`
delta, the region opening **before any frame data is on device** and closing
after the first tracker call, 96 replicas against 32 (21 against 22 of the
meter's units). Read independently in a second binary, which prints the same
448.0/1,408.0. The honest weakness is printed at the number: binCV *stores* the
ternary derivative planes where `cv::cuda` recomputes them, and that is most of
the ratio.

### What the profiler said, including where it contradicted the design

`ncu --kernel-name regex:trackKernel --launch-count 1`:

- **`math_pipe_throttle` is 0 of 2,753 stall samples.** The design's
  pre-registered "~85% FP64-bound" is **refuted outright**. This was the single
  most load-bearing assumption in the tracker's design document.
- `short_scoreboard` **50.9%** (warp shuffle / MIO — the cross-lane reduction),
  `wait` 21.8%, `long_scoreboard` 11.4%, `gpu__dram_throughput` **0.93%**.
  Reduction-latency-bound at low occupancy; not FP64, not bandwidth.
- Achieved occupancy **7.7%** against `cv::cuda`'s **21.3%** — the parallelism
  asymmetry the header stated *before* measuring, materialised.
- **Both sides in one profiled run, 61 keypoints:** binCV 65.4 µs (one launch,
  four levels) against `cv::cuda` 9.3 + 12.2 + 10.2 + 10.2 = **41.9 µs**.

That last line is the one to read twice. **The kernel work is a 1.56× LOSS.**
The wall-clock win above is the launch shape — one launch against six on a host
where a launch is expensive — and it is stated that way rather than as a kernel
result. The counter-fact is in the same profile: `cv::cuda`'s `pyrlk::sparseKernel`
does **146 MiB of local-memory loads per launch** and binCV's does **zero**,
which is why the loss is 1.56× and not more.

Two arms were **not built** because the profile predicted they would lose. The
lane-0 broadcast exists to cut FP64 *issue* cost, which the histogram says is
zero, and it would add shuffles to a kernel already half-stalled on them. The
thread-per-keypoint arm would give ~5 warps on 48 SMs at 148 keypoints. One
profile in place of six experiments.

### Shared, not forked

Under the project's one-implementation rule the tracker calls the host's own
`ReplicatedShiftedRow::word`/`sourceWord`, `floorDiv`, `edgeFill`,
`alignedWord`, `TapSums::combine`, `referenceMinEigScale`, `clipRegion`,
`lowBitsMask`, `minEigenValue`, `combineBitSlicedPairs` and `usableLevelCount`.
`combineBitSlicedPairs` is the notable one: the design said it would have to be
forked, and it did not — it already carried the annotation.

Three structural wins were taken, each an identity on integer sums rather than
an approximation: the self terms hoist out of the iteration (10 reductions
become 8), the sign plane is pre-split to positive/negative at staging (one AND
per term removed), and `tapIsShift` keys on the **clipped** window span, which
is the host's actual rule.

### The device domain, and three findings reported rather than smoothed

`N ≥ 3`, windows wider than 32 or taller than 64, more than 16 levels, and `err`
at `N > 1` all **refuse with `cudaErrorInvalidValue`** — named in the docstring,
asserted, and tested on both sides of the boundary. Whether "the device backend
supports a strict subset of the host's domain" is an acceptable API contract is
a contract statement rather than a measurement, and it is filed as such.

1. **A host bug, found by the device suite and not fixed here.**
   `bincv::impl::lkBatchEnabled()` on against off **changes the answer at
   `maxIterations == 0`**: the AVX2 eight-keypoint batch performs one iteration
   when the caller asked for none, because it tests its cap at the bottom of a
   do-while-shaped loop where `trackOnePoint` tests it at the top of a `for`.
   264–276 points diverge per suite, at that parameter value and nowhere else.
   The device arm matches the host's **scalar** arm exactly. The suite asserts
   zero divergence at `maxIterations >= 1` and *reports* the zero-iteration
   count without asserting it. It is a host x86 vector arm and fixing it needs
   the host gate and the host LK benchmark re-taken.
2. **Two prior documents were wrong about `cv::cuda`'s termination rule.** Read
   out of `pyrlk.cu`, it is `if (fabs(delta.x) < 0.01f && fabs(delta.y) < 0.01f)
   break;` — a plain convergence break, neither the design's "fixed iteration
   count, no epsilon" nor the critique's "oscillation rule with half-step
   back-off". The forced-equal-iterations arm was dropped on that reading and
   both sides run free.
3. **`reduce.hpp` and `covariance.hpp` call the batched covariance "THE ENTRY
   POINT A TRACKER USES"; this tracker does not call it.** One launch of
   `gradientCovarianceBatchAsync` over the same windows measures **15–26% of the
   whole four-level tracker**, for one level and before its second traversal.
   The suite holds a probe equal to that kernel so the two spellings cannot
   drift. Which document changes is not this section's call.

## Sparse stereo, matching, and the geometry

The sparse path — the one a SLAM frontend actually runs — and the geometry
downstream of it. Six operations landed; the geometry did not, and that is the
measured answer rather than the backlog.

All six are **API Tier 3**: `matchDescriptors`, `matchDescriptorsGated`,
`stereoDescriptorMatch`, `stereoRefineDisparity`, `stereoMatchRectified` and
`calcOpticalFlowBlockMatch`. Each is bit-exact against its host twin and each
keeps its reference arm reachable in the same binary.

### Descriptor matching, which is where the format was supposed to pay

**The role bar is `knnMatchAsync(k = 2)` on `CV_8U` — what `cv::cuda::ORB`
emits — *and* on `CV_32S` over identical bytes**, because `matchHamming_gpu<int>`
is instantiated and is therefore the better existing option. Both arms on one
explicit stream. `knnMatchConvert` is a host download and sits **outside** the
event window on every arm; the OpenCV side is timed doing the ratio test on
device in a committed kernel, or not at all — so the ratios below are an **upper
bound on OpenCV's standing**, not a favourable framing of binCV's.

The case was pre-registered to decide at **5000×5000**; 470×470 is reported and
explicitly does not decide, because both arms sit near the launch floor there.

| descriptors | `cv::cuda` CV_8U | `cv::cuda` CV_32S | binCV | ratio | disjoint |
|---|---|---|---|---|---|
| 5000×5000 *(deciding)* | 1.9702 ms | 1.9491 ms | **0.2195 / 0.2189 ms** | **0.110 / 0.109** | 7/7 |
| 470×470 *(does not decide)* | 0.0455 ms | 0.0445 ms | 0.0105 ms | 0.223 / 0.231 | 7/7 |

**Memory: 400.0 KB against 8,277.3 KB — 20.7× smaller**, `cudaMemGetInfo` on
both sides, 128 replicas against 48 (25 against 194 of the meter's units).

**And here a documented claim is contradicted by measurement, which is the more
valuable half of this row.** `cuda/descriptor.hpp` stated that emitting
`uint32_t` words gives a matcher "a real ~4× instruction advantage" over
`cv::cuda`'s `uchar` popcounts. The *source reading is correct* — 32 `__popc`
per 256-bit descriptor against 8. **It buys 1.02×.** Measured on OpenCV's own
side, CV_8U against CV_32S over identical bytes at map scale: 5.233 ms against
5.143 ms. The profile says why: `math_pipe_throttle` is 871 of ~108,000 warp
stall samples — under 1% — while `barrier` is 35,540. The kernel is
`__syncthreads()`-bound, and a kernel that is not math-bound does not care how
wide its popcounts are.

**binCV's 9–13× lead is kernel shape, not representation**: two
`__syncthreads()` per launch against one per descriptor chunk per train block,
and 0.25% of peak DRAM against 12.5%. The header has been corrected to say so.
Word emission is still the right output — it is what lets a matcher hold a
descriptor in eight registers, and it costs nothing to keep — but it is not
where the lead comes from.

**A second predicted wash turned out not to be one.** The design computed both
sides unpitched and predicted memory parity. Measured on one meter it is
0.03 MB against 0.50 MB per set, because a 256-bit descriptor is a **32-byte
row** and `GpuMat` pitches to a 512-byte multiple — roughly 16× of overhead
OpenCV carries and a packed array does not. That is a `GpuMat` property, not a
census of anyone's algorithm, and it is stated at the number.

**The structural lever the design missed, and it is the one that mattered.**
Every candidate reads the *same rows* at consecutive shifts, so each row's 96
covering columns are assembled **once** (`RowSpan`) and every candidate becomes
one `__funnelshift_r`. `2R + 3` candidates cost one row assembly instead of
`2R + 3`. It also removes the `s == 0` shift-undefined-behaviour hazard outright,
because `__funnelshift_r` is defined at shift 0. Its correctness argument —
`word(i)` is a pure function of the source column — is pinned by a sweep rather
than asserted.

Tile width was measured rather than inherited: at 5000², tile 1 = 2.542 ms,
tile 4 = 0.978, **tile 8 = 0.638**. At 470² all three sit at the floor. Default 8.

### The gated matcher: the speed rationale does not survive on device

Pre-registered two-sided, with the prediction written down: the gated arm is
worth having on device only if it beats device brute force by more than both
ranges at frontend scale, and memory counts *against* the gate on device by
arithmetic stated up front.

Admitted fraction **4.73%**. Device gated/brute **1.11×, ranges OVERLAP — not a
result**. On the host the same comparison reads 0.81× (host clock, indicative).
The op ships because a caller who has the gate already will not pay to discard
it, and the suite pins that an unbounded window reproduces brute force exactly —
but **the device speed rationale for gating is not established**, and no number
was massaged to establish it.

### Sparse stereo, against binCV's own dense arm

Bar written first: coarse + refine must come in **under 0.39 ms** and under
**442 KB** — binCV's own device `denseDisparityBinary` plus an index, which is
the best existing option for the job, not a fallback.

**Met on both, decisively:** 0.019 ms (20× under) at **135.6 KB** of working set
and **no scratch at all**.

**The accuracy is published and is not a pass.** Against a pair with exact
ground-truth disparity 21: **236 of 500 within half a pixel, 261 pinned at the
scan's low edge**, mean 2.11 px. The behaviour is **bimodal on a globally
thresholded frame** and it is the *host* algorithm's behaviour — the suite holds
the device bit-exact to it. Where the local binary window is uniform every
disparity scores 0 and the tie rule takes the smallest; a FAST corner in the
wide image is not a corner in the thresholded one. The fix is a richer packing
(census or N-bit), not this kernel. **What residual makes the sparse arm not
worth running is unset**, so the op ships on its stated bar with its accuracy in
plain sight rather than on an accuracy claim.

### Block matching

Role row only: **0.054 ms against `cv::cuda`'s LK at 0.232 ms — 0.23×**, ranges
disjoint. The footprint condition was written as *derived and falsifiable* — the
measured meter-2 ratio must reproduce the computed one within the meter's step —
and it holds: **computed 6.99×** (135.6 KB against 948.0 KB) against **measured
7.33×** (0.19 against 1.38 MB per replica). Agreement within the step is what
lets the footprint claim rest on the format rather than on one reading. Both
caveats are printed at the number: `GpuMat` pitch makes the computed LK figure a
lower bound, and pooling makes the measured one an upper bound.

**It is ship-blocked on an accuracy floor nobody has set.** How much tracking
yield this route may give up against `cv::cuda` LK is a judgement; both sides
report 500/500 status bytes, which is not an accuracy measurement.

### The geometry, and the measurement that says it stays on the host

**The answer is NO. The geometry stage should not move to the GPU.** Nothing
shipped: there is no `bincv::cuda` geometry operation, nothing in
`include/bincv/cuda/`, nothing in `bincv_cuda`. What exists is the experiment
that settled it, kept as a benchmark and a suite so the negative is
reproducible.

The rule was five CONTINUE/KILL gates with every magnitude derived or measured,
and the **ship bar deliberately left blank** — the design's "~10.7×" was an
invented multiplier wearing a derivation's clothes and was deleted rather than
filled in.

**The operating point was wrong on three counts and all three were
re-measured.** On EuRoC V1_02 over 400 frames: **N = 140.6** accepts per frame
(the design assumed 200), **I = 21.0** adaptive iterations (assumed ~12), and
geometry is **41.1 / 43.9 / 45.2%** of binCV's total (assumed 15–30%). The
Amdahl ceiling is 1.70–1.82× and does **not** kill — the 43% share makes the
adoption case *stronger* than the design assumed. That matters, because
everything below is a negative found despite a favourable premise.

**G1 kills the scoring-only design, which is literally what the issue proposed.**
At N = 141: solver 0.08646 ms/hypothesis, scoring 0.00571 ms/hypothesis, 4.334
models per hypothesis → **f = 0.062**, so a scoring-only arm caps the *stage* at
`1/(1−f)` = **1.066×** at infinite device speed (7-run median **1.0740×**, range
1.039–1.131). That is below the host arm's own measured spread. Moving all
scoring to an infinitely fast device buys 1.07× on the stage and ≤1.03× on
binCV's total.

**G2 passes and is the reason the rest is cheap.** `fivePointEssential` compiles
for device **unmodified** under `BINCV_HOST_DEVICE` — 255 registers, a
**6,800-byte stack frame**, 1,296 B of spill stores, clean with assertions live.
No fork, so the Gauss-Seidel-versus-Jacobi question, the compaction ordering the
tie rule depends on and the pivot tie rule are all identical by construction
rather than by review. That is the shared-implementation rule earning its keep
on a path that did not ship.

**G3 fails by 15.9×, and this is the result.** 7 independent process runs, each
an interleaved paired median:

| | median | range |
|---|---|---|
| device round, best swept H (= 32), round trip and sync included | **56.18 ms** | 56.02–56.45 |
| host adaptive search | **~3.1 ms** | — |
| ratio | **18.58×** | 15.88–19.54 |

The two arms' sample ranges are **disjoint**, so it is a result rather than a
spread. A single device lane takes ~1.8 ms for one hypothesis where a CPU core
takes 0.13 ms.

**Where the device does win, and why it does not matter.** Extending the sweep,
round time is flat at 95–101 ms from H = 128 to H = 4096 and linear after.
Saturation throughput is **68 hypotheses/ms against the host's 7.8** — the
device is 8.8× better *per hypothesis*, but only at H ≈ 98,304, which is
**3,600× the hypotheses the frontend runs**.

**G4 fails on the axis the design called the one it was confident about, and it
contradicts a documented figure by 113×.** Host side:
`ransacScratchBytes(141)` 40 B + `essentialSolverStackBytes()` 5,376 B + points
2,256 B = **7,672 B**. Device allocation sum at its most favourable round
(H = 32): 2,264 B global + 32 × 6,800 B of per-thread frame = **219,864 B, 28.7×**.
And on meter 2 the first launch of the round kernel **reserves 406.00 MB**,
**independent of H** (48 SMs × 1,536 threads × 6,800 B ≈ 478 MB) — a null kernel
and an eight-double-frame kernel both read 0.00 MB in the same process, so the
reading is the solver's frame and not module load. The design's memory plan said
7,436 B at H = 512, "under both", because its table never counted the per-thread
local frame that `ptxas` reports.

**G5 passes** — device 100 inliers over 1024 hypotheses against host 98 over 27.
It is the only gate that does.

**The profiler changed the decision's reason, which is worth more than the
decision.** The design named FP64 at 1/64 rate as "THE HEADLINE RISK".

| | `kRoundSerial` | `kRoundWarpBallot` |
|---|---|---|
| `short_scoreboard` | 64,723 = **90.5%** | 820,493 = **90.9%** |
| `math_pipe_throttle` | 56 = **0.08%** | 370 = **0.04%** |
| `sm__throughput` | 0.71% | 10.36% |
| `gpu__dram_throughput` | 0.43% | 0.40% |
| `sm__warps_active` | 2.08% | 3.61% |

**FP64 is not the binding constraint.** The math pipe is throttled 0.04–0.08% of
the time. What binds is the solver's 6,800-byte local frame: 91% of warp-cycles
wait on a local-memory dependency at 2% occupancy with nothing to hide it
behind. That reading is also why register pressure is not the lever, verified
rather than assumed: `-maxrregcount` at 255 / 128 / 64 gives 51.8 / 51.4 /
50.9 ms — identical within noise **despite spill stores rising 1,296 → 10,512
bytes**. Latency-bound on a serial chain, not bandwidth-bound. And DRAM at
0.40% closes the narrowing-`Point2f` question with a number instead of a
sentence.

**The design's own mitigation was killed by measuring its ceiling before writing
it.** One sample per *warp* with the solver serial in lane 0 and 31 lanes idle
already removes both penalties that design targets — divergence over 32 trip
counts, and 32 solver frames thrashing one L1 — and adds none of the internal
parallelism. It measures **1.01–1.02× of thread-per-sample**: not faster. The
arm's best possible round time is therefore 56.2 ms ÷ W, with W bounded by the
solver's widest phase (~11) at ~34% lane utilisation. **Even a perfect 11× leaves
5.1 ms against the host's 3.79.** A design whose central mitigation fails its own
ceiling did not need to be written.

**What is exact and what is not, rung by rung.** Sample indices: **exact** —
`impl::ransacSample` is a pure integer function of (seed, iteration), so the two
targets draw *identical* minimal sets and "distributional" was the wrong target.
Models: **not bit-exact, and the suite says so** — 0 of 1,942 bit-identical at
the default build, and **1,398 of 1,942 (72.0%) with `--fmad=false`**, worst
relative coefficient difference 1.5e-06, so FMA contraction is the dominant
source and the residue is libm (≈31 `cos`/`sin` per hypothesis) amplified by two
break-on-convergence loops. What is *enforced* is the same solution count and
**|q2ᵀEq1| ≤ 4.58e-16** at every device model's own five points. Consensus:
**exact**, 0 support mismatches, 0 flag-word mismatches, padding past `count`
zero. Winner: **exact**. On collinear input the two targets disagree on whether
a sample is degenerate for **1 of 192 (0.52%)** — a 1e-14 rank tolerance on a
singular matrix, reported with its rate rather than asserted away.

**The recommendation this measurement supports.** The binding constraint is not
FP64 rate but a 6,800-byte per-thread local frame, and register-file and L1
sizes are the same on a datacenter part — so a 1/2-rate FP64 GPU would not
obviously change the answer. This is a property of the algorithm's state, not of
this GPU's price bracket.

The `BINCV_HOST_DEVICE` annotations on the solver chain, the sampler and
`ransacScratchWords` were kept. They cost nothing on the host — the host check
counts are unchanged at 304 / 143 / 31, matching `tests/expected-checks.txt`
exactly — they are gate-covered, and they are what makes this negative
reproducible and a future revisit cheap.

## The profiler pass

Everything above the frontend section was optimized **without a hardware
profiler**, and that shaped both the method and where it stopped. A profiler
runs here now. This section records how, what it read on every device kernel,
which earlier conclusions it **confirmed**, which it **contradicted**, and what
it says to do next.

Nothing in this section was optimized and no source file was changed for it.
It is diagnosis.

### How the profiler was enabled

`ncu` failed with `Unknown Error on device 0`, reproduced on a three-line kernel
asking for a single basic metric — so it was device-level counter access being
refused rather than anything about these kernels. The cause is NVIDIA's default
of restricting GPU performance counters to administrators, and the fix is
Windows-side, not WSL-side:

- set `RmProfilingAdminOnly = 0` (DWORD) under
  `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak`, from an
  elevated shell;
- **reboot Windows.** A `wsl --shutdown` does not reload the display driver and
  is not sufficient.

Then use **`/opt/nvidia/nsight-compute/2024.2.1/ncu` by full path**. Nothing is
on `PATH`, and the `ncu` shipped in `/usr/local/cuda-11.1/bin` is 2020.2.0 —
too old for this driver. Build with the 11.1 `nvcc` as always; profile with the
2024 tool.

```
/opt/nvidia/nsight-compute/2024.2.1/ncu --kernel-name regex:<name> \
    --launch-count 1 --metrics <list> <binary>
```

**`nsys` remains unusable** — the CUDA 11.1 bundle crashes on this glibc, and a
newer one writes a report containing zero CUDA kernel events at any trace
setting, which is the WSL2 CUDA-tracing limitation. **`compute-sanitizer` does
not work here either**; given a deliberate 1,020-element overread of a
4-element allocation it printed `ERROR SUMMARY: 0 errors`. No device
out-of-bounds read in this backend is observable by any tool on this machine,
which is why two guards in this backend are pinned by swept arithmetic
invariants instead.

### Three caveats, stated before any number

1. **No duration here is a timing number.** `ncu` serializes, replays and locks
   clocks to base. Durations below are for *structure and ratios within one
   profile*. Every speed number in this document comes from a free-running
   benchmark, and **timing and profiling never mixed**: no timing run in this
   report was taken from a profiled process.
2. **`imc_miss` dominating a sub-3 µs kernel is an artifact, not a limiter.**
   Each profiled launch runs with flushed caches, so the constant bank is cold;
   at 45 blocks — 0.2 waves on a 48-SM part — there is nothing to amortize it
   against. Read those rows as *"too short to have a limiter."*
3. **`dram__bytes_*` is only trustworthy when the working set exceeds the 4 MB
   L2.** At 752×480 an output can live and die in L2 and never appear as a DRAM
   write. The honest traffic figure at small geometries is the SM's own sector
   counts, which is what the traffic table uses; where the working set genuinely
   exceeds L2 (the scale sweep at 4K and 8K) the DRAM counters reproduce the
   arithmetic exactly and are used there.

**Coverage: 69 of the 71 distinct `__global__` functions in the backend, 87
instantiations**, across four passes — the 18 `smsp__pcsamp_warps_issue_stalled_*`
counters plus roofline and occupancy; a purpose-built scale probe at 752×480,
1920×1080, 3840×2160 and 7680×4320; the L1/L2 sector and shared-bank-conflict
counters; and the 17 test binaries for the kernels no benchmark launches first.
The two not reachable from a benchmark are the tracker's `<<<1,32>>>`
correctness probes, and they are covered from the test pass.

### The kernels that carry a limiter

`us` is an ncu-clock duration and **is not a timing number**. `SM%`/`DR%` are
`sm__throughput` and `gpu__dram_throughput` as percentages of peak sustained
elapsed; `occA`/`occT` are achieved against theoretical occupancy; `gLD`/`gST`
are global load/store sectors × 32 B at 752×480. Rows marked **[ref]** are
reference oracle arms that nothing ships.

| kernel | us | SM% | DR% | occA/occT | grid×blk | regs | top stalls |
|---|---|---|---|---|---|---|---|
| `blockMatchRefKernel` **[ref]** | 3287.1 | 2.7 | 0.0 | 8.2/41.7 | 4×128 | 86 | wait 49, not_sel 27, branch 15 |
| `denseKernel` **[ref]** | 2029.1 | 83.2 | 0.0 | 87.3/100 | 2880×128 | 40 | not_sel 36, mathpipe 32 |
| `censusKernel<uint8>` **[ref]** | 1188.9 | 40.5 | 83.1 | 83.6/100 | 34560×256 | 34 | long_sb 34, lg_thr 27, mio 23 |
| `fastSortKernel` **[ref]** | 1155.7 | **1.2** | 0.1 | 66.6/66.7 | **1**×1024 | 19 | long_sb 41, lg_thr 27, barrier 15 |
| `denseKernelPacked` | 1113.7 | 53.0 | 0.7 | **15.2**/16.7 | 180×128 | **255** | wait 44, not_sel 28, **mio 13** |
| `bitonicOneBlockKernel` **[ref]** | 1010.2 | **1.0** | 0.1 | 66.7/66.7 | **1**×1024 | 20 | long_sb 53, wait 20, barrier 16 |
| `subPixKernel<1>` | 581.6 | 5.5 | 0.1 | **2.1**/33.3 | 7×32 | 61 | short_sb 51, wait 25 |
| `denseKernelSliding` | 461.8 | 53.2 | 0.2 | 28.5/33.3 | 180×128 | 116 | wait 40, not_sel 17 |
| **`denseKernelPackedWarpBox<9,16,4>`** | **454.0** | 52.0 | 1.7 | **40.6**/50.0 | 944×128 | **80** | wait 35, short_sb 23, long_sb 13 |
| `spacingKernel<1>` | 194.1 | **1.0** | 0.0 | 66.6/66.7 | **1**×1024 | 22 | barrier 35, wait 21 |
| `fusedCandidateKernel<1>` | 145.7 | 68.8 | 1.0 | 30.7/58.3 | 180×128 | 64 | **short_sb 67**, tex 11 |
| `censusKernelTiled<uint8>` | 81.6 | 70.1 | 59.2 | 85.5/100 | 1440×256 | 40 | long_sb 22, wait 20 |
| `denseKernelBitSliced<4>` | 77.5 | 65.4 | 1.2 | **13.7**/16.7 | 354×32 | **239** | **wait 36**, not_sel 32 |
| `responseKernelSliced` | 74.2 | 69.3 | 3.4 | **16.0**/41.7 | 90×128 | 72 | **short_sb 69** |
| `trackKernel<2>` | 69.5 | 34.1 | 0.9 | **8.0**/33.3 | 37×128 | 124 | **short_sb 47**, wait 20 |
| `censusPackedKernel<uint8>` | 62.8 | 80.4 | 72.0 | 83.8/100 | 1440×256 | 40 | not_sel 23, mathpipe 19 |
| `responseKernelWindow` | 64.2 | 78.6 | 3.6 | 69.0/75.0 | 2820×128 | 56 | short_sb 40, tex 25 |
| `medianWideFastKernel<9>` | 17.4 | 61.1 | 7.0 | 53.7/66.7 | 360×256 | 64 | not_sel 32, mathpipe 29 |
| `blockMatchLevelKernel` | 14.6 | 39.2 | 0.7 | **21.0**/66.7 | 125×128 | 64 | wait 32, short_sb 17 |
| `matchTiledKernel<0,1>` | 11.4 | 35.3 | 0.9 | 68.5/100 | 470×128 | 40 | mio 23, long_sb 18, barrier 16 |
| `stereoRefineFastKernel` | 10.2 | 24.8 | 2.1 | **20.6**/100 | 125×128 | 38 | short_sb 38, wait 20 |
| `packQuantKernelRowGrid<uint8>` | 8.0 | 47.1 | 9.8 | 82.3/100 | 1440×256 | 20 | long_sb 29, wait 23 |
| `briefBallotKernel<uint8>` | 9.6 | 18.6 | 34.1 | 41.2/100 | 125×256 | 30 | long_sb 66 |
| `fastEmitKernel<9,1>` | 7.9 | 19.5 | 6.8 | 15.5/83.3 | 95×128 | 48 | wait 24, barrier 16, imc_miss 15 |
| `edgeKernelVec<0>` | 7.4 | 39.4 | 10.1 | 62.5/100 | 360×256 | 32 | wait 22, long_sb 19 |
| `fastCountKernel<9>` | 4.4 | 13.5 | 2.5 | 15.8/100 | 95×128 | 40 | imc_miss 26, wait 18 |
| `packKernelByteLane` | 3.8 | 18.9 | 18.4 | 64.1/100 | 480×256 | 16 | long_sb 32, wait 20 |

The remaining ~45 instantiations sit at or below 3 µs on 0.2 waves, with
`imc_miss` above 30% — the cold-constant-bank artifact of caveat 2. They are
covered, and there is nothing in them to read.

**Two readings belong with the table.** The **reference oracle arms are the
slowest kernels in the backend and nothing ships them** — they exist to be the
byte-for-byte oracle, and a reader scanning for the largest numbers will land on
them first. And three kernels are **genuinely healthy, arithmetic-bound and have
nothing to take**: `medianWideFastKernel<9>` and the dense reference arm both
sit at 61–83% of peak SM with `not_selected` + `math_pipe_throttle` as their top
pair, which is what a kernel doing its work looks like.

### The scale sweep, which decides both asserted assumptions

One launch per geometry, output arithmetic checked against the counters:

| kernel | geometry | SM% | DR% | occA | GB/s | top stalls |
|---|---|---|---|---|---|---|
| `binaryKernel<OpAnd>` | 752×480 | 4.8 | **6.9** | 16.1 | 41 | imc_miss 29, wait 21 |
| | 1920×1080 | 19.6 | 26.7 | 70.2 | 158 | long_sb 59 |
| | 3840×2160 | 31.4 | 65.1 | 75.2 | 383 | long_sb 59 |
| | 7680×4320 | 44.9 | **87.9** | 83.0 | **519** | long_sb 71 |
| `regionKernel<Count,u64>` | 752×480 | 6.0 | 3.7 | 16.4 | 21 | imc_miss 61 |
| | 3840×2160 | 38.0 | 23.0 | 73.5 | 136 | long_sb 28 |
| | 7680×4320 | 47.2 | **36.8** | 81.2 | 217 | long_sb 51 |
| `packKernelByteLane` | 752×480 | 19.9 | 19.1 | 63.8 | 113 | long_sb 43 |
| | 3840×2160 | 49.3 | 63.9 | 75.8 | 378 | long_sb 48 |
| | 7680×4320 | 55.0 | **65.4** | 77.2 | 387 | long_sb 46, wait 20 |
| `packQuantKernelRowGrid` | 752×480 | 46.4 | 9.9 | 82.1 | 58 | long_sb 28, wait 24 |
| | 3840×2160 | 66.1 | 15.8 | 83.5 | 94 | long_sb 34, wait 26 |
| | 7680×4320 | **66.9** | **15.0** | 83.6 | 89 | long_sb 34, wait 26 |

**Assumption one — "memcpy-bound" for the pointwise ops: REPLACED, and it is
two different answers.**

`bitwiseAnd` is **true asymptotically and false at the number it annotates.** At
7680×4320 it runs at **87.9% of peak DRAM** with `long_scoreboard` at 71% —
memcpy-bound, and the claim about the *operation* is right. But this document
attached it to the **752×480** row, and there the kernel reads **6.9% of peak
DRAM and 4.9% of peak SM** on 45 blocks — **0.2 waves** on a 48-SM part — with
`imc_miss` 41% and `wait` 25%. At that geometry it is bound by launch and by
having less than one wave of work, and the 0.007 ms is a launch-floor number
wearing a bandwidth explanation. Traffic is exactly optimal at every size, so
nothing is wrong with the kernel; the **label** was wrong.

`countNonZero` is **false at every size measured** — 3.7% → 23.0% → **36.8%** of
peak DRAM, stopping well short of the roof while SM reaches 47.2%. `bitwiseAnd`
runs the *identical* grid-stride traversal with three times the bytes per loop
index and reaches 87.9%. Traffic is 1.0× optimal on both, so the difference is
per-index arithmetic: `regionKernel` pays `idx / rowSpan` — a 64-bit software
divide — for every 4 bytes it loads, where `bitwiseAnd` pays it for every 12.
That is the same divide the round-1 pass found in `packKernel` and that the
row-grid shape deleted there.

**Assumption two — "near-optimal" for the ballot packer: NOT CONFIRMED, and the
two packers fail it differently.**

`packBits`' shipped byte-lane arm **plateaus between the roofs**: DRAM 19.1 →
49.6 → 63.9 → **65.4%** and SM 19.9 → 37.9 → 49.3 → 55.0%, with
`long_scoreboard` at 43–48% throughout. Neither roof is reached at any size and
the DRAM figure *stops improving* between 4K and 8K. It is memory-**latency**
bound at one outstanding load per thread, roughly 1.5× off the bandwidth roof,
and it does not converge to it. It also writes **2.0× its output in store
sectors**, because a warp stores only four contiguous words.

`packQuant` — which has no byte-lane twin — is **false by a wide margin**: DRAM
9.9 → 15.8 → **15.0%** while SM climbs to **66.9%**. Compute/issue-bound at
every size, never approaching any memory roof. At 7680×4320 it takes **477.6 µs
against `packBits`' 100.1 µs on 1.16× the traffic — 4.8×** — and carries **8.0×
store-sector amplification**. The decision not to give it a byte-lane arm was
taken without this number; the profile prices that decision at 4.8×.

### Traffic, as the honest memory number

Allocation sums and `cudaMemGetInfo` deltas say what is *reserved*; these are
what the hardware *moves*, per launch at 752×480, against the arithmetic
minimum:

| kernel | gLD KiB | gST KiB | locLD KiB | locST KiB | LD× | ST× |
|---|---|---|---|---|---|---|
| `binaryKernel<OpAnd>` | 90 | 45 | 0 | 0 | 1.0 | 1.0 |
| `regionKernel<Count,u64>` | 45 | 0 | 0 | 0 | 1.0 | — |
| `packKernelByteLane` | 360 | 90 | 0 | 0 | 1.0 | **2.0** |
| `packKernel<uint8,GT>` / RowGrid | 525 | 360 | 0 | 0 | 1.5 | **8.0** |
| `packQuantKernel` / RowGrid | 525 | 720 | 0 | 0 | 1.5 | **8.0** |
| `unpackKernel` | 360 | 532 | 0 | 0 | **8.0** | 1.5 |
| **`censusKernelTiled<uint8>`** | 1787 | 8640 | **69120** | **24480** | 5.1 | **8.0** |
| `censusPackedKernel<uint8>` | 1788 | 1410 | **69090** | **24480** | 5.1 | 1.0 |
| `denseKernelBitSliced<4>` | 14575 | 1711 | 0 | 0 | *161.9* | 4.9 |
| `medianWideFastKernel<3>` | 1782 | 460 | 0 | 0 | 5.1 | 1.3 |

Two of these change what this document should say about memory.

**`denseKernelBitSliced<4>`'s 161.9× load figure is not a defect** and must not
be quoted as traffic: it is L1 sector *requests* for a sliding disparity sweep,
and L1 absorbs 93% of it (L2 read is 944 KiB, DRAM 1.2% of peak). The same is
true of the matchers' large `gLD` columns, and the useful comparison is between
them — the round-3 warp-box arm issues **555,012 KiB against `denseKernelPacked`'s
2,640,960 KiB, 4.76× fewer**, which is half of why it is faster.

**The census transform moves 91.4 MiB of per-thread local memory for a 0.35 MiB
input and a 1.06 MiB output.** That is the single largest gap in this backend
between what the allocation figures say and what the hardware moves, and it is
R1 below.

### What the profiler confirmed, and what it contradicted

**Contradicted — the more valuable half, so it goes first.**

- **The census matcher's model was wrong.** The design modelled
  `denseKernelPacked` as popcount-bound at ~42% pipe efficiency and predicted
  3.6–5.7× from a warp-cooperative shape. `math_pipe_throttle` is **3.5%** of
  stall samples and DRAM is **0.72%**: it is latency-bound at 15.2% occupancy,
  and its leading throughput throttle is `mio_throttle` — the load/store
  instruction *queue*. The replacement landed at 2.49×, and the register budget
  rather than the shape was what finally paid.
- **The tracker's "~85% FP64-bound" is refuted.** `math_pipe_throttle` is **0 of
  2,753 stall samples** in `trackKernel`. This was the most load-bearing
  assumption in that design, and it was wrong.
- **The five-point solver's "headline risk" is not FP64 either.**
  `math_pipe_throttle` 0.04–0.08%, `short_scoreboard` 90.5–90.9%. The binding
  constraint is a 6,800-byte per-thread local frame, which changes what a future
  revisit of that decision should test.
- **The `~4×` popcount advantage in `cuda/descriptor.hpp` buys 1.02×.** Real as
  an instruction count, worth nothing on a `__syncthreads()`-bound matcher.
  Header corrected.
- **"memcpy-bound" and "near-optimal"**, the two assumptions this pass existed
  to settle: one replaced with a size-dependent answer, one refuted.
- **`cuda::threshold`'s residual was mis-explained a second time.** The round-2
  correction (bandwidth realization approaching a roof with size) is itself
  incomplete: the DRAM figure **plateaus at 65.4%** at 8K rather than
  converging. It is memory-latency bound plus 2.0× store amplification.

**Confirmed.**

- **`cuda::threshold`'s byte-lane profile reproduces to 0.1 point.** This pass
  reads 49.3% SM / 63.9% DRAM at 3840×2160 where round 2's own `ncu` run
  recorded 49.4% / 63.8% — which is the cross-check that this pass is measuring
  the same thing the earlier one did.
- **The RANSAC "NO" was not premature.** All three experiment kernels sit at 255
  registers, 3.6–3.7% achieved occupancy against a 16.7% ceiling, and 73–74%
  `short_scoreboard`. Nothing in the profile suggests a tuning answer.
- **`cornerSubPixAsync` is at a ceiling of its signature, exactly as its rule
  predicted.** Its author wrote "spreading over B blocks divides the time by ~B
  until another limiter appears; if it does not, the profiler must name what
  does." The profiler's answer is that **there is no other limiter — there is
  not enough work**: 7 blocks × 32 threads, **2.1% achieved occupancy** against
  a 33.3% ceiling, no local memory, no spills, and `math_pipe_throttle` does not
  appear at all. 200 corners at one thread per corner is 6.25 warps on a part
  that holds 2,304. More blocks cannot help. The one decomposition that would
  add parallelism — a warp per corner — is the one the bit-exactness argument
  forbids, because double addition is not associative. And where a single number
  was predicted it matched: `sm__inst_executed_pipe_fp64` at 45.65% of peak
  explains the spread arm's measured 3.18× against a predicted 7 × 0.457 = 3.20.
- **The two single-block kernels the frontend round found are gone, and their
  replacements read as claimed.** `fastSortKernel` at 1155.7 µs in one block is
  now `fastCountKernel` 4.4 µs + `fastEmitKernel` 8.9 µs at 95 blocks each.
- **`cornerMinEigenValAsync`'s own rule branched correctly.** Neither arm is
  ≥80% on either roof, so the stall histogram decides, and it names
  **occupancy**: `responseKernelWindow` 78.6% SM / 3.6% DRAM, `responseKernelSliced`
  69.3% / 3.4% at **16.0%** achieved occupancy. The control is decisive —
  OpenCV's `cornerMinEigenVal_kernel` runs the same shape at **87.5% achieved
  occupancy**.

### Ranked opportunities

Ranked by expected end-to-end value, each with the metric that identifies it and
a falsifiable prediction. **None implemented.** A microbenchmark ratio here is
not an end-to-end result, and each entry says what share it can move.

**R1 — `censusTransform`'s offset table lives in per-thread local memory.**
`l1tex__t_sectors_pipe_lsu_mem_local_op_ld` = 2,211,840 sectors = **67.5 MiB**
and `..._op_st` = 783,360 = **23.9 MiB** per 752×480 / 24-plane launch, against
1.75 MiB of global loads and a 1.06 MiB output. 783,360 × 32 B ÷ 360,960 threads
= **69.4 B per thread** — exactly `sizeof(CensusOffsetsPod)` (68 B) rounded to 8.
Confirmed three independent ways: `cuobjdump -res-usage` reports **STACK:72** on
all six census instantiations, the SASS shows 18 `STL` and 30 `LDL`, and the
counters agree. It survived because `-Xptxas -v` reports this as `STACK`, not as
the "spill stores/loads" string everyone greps for. The cause is
`CensusOffsetsPod::dx[32]/dy[32]` indexed by the runtime plane loop in a
by-value kernel parameter. **A sweep of the whole backend for the same defect**
found only four files with any non-zero stack frame: `census.cu` (STACK:72, all
six), `descriptor.cu` (STACK:48), `fast.cu` (STACK:96 — but only on the
generic-arcLength instantiations; **the shipped arcLength-9 ones are clean**) and
`orientation.cu`'s two reference arms. Everything else is zero. **The project
already owns the fix and applied it once**: `orientation.cu`'s `DiscPod`
bit-packs its per-row table into two `uint32` "because a stack round trip inside
the disc loop is the cost the quad arm exists to avoid." That reasoning was
never carried to `census.cu`. *Prediction:* staging `dx`/`dy` into `__shared__`
at block start — the tiled kernels already `__syncthreads()` there — takes local
sectors to 0 and DRAM writes from ~25.9 MB to ~1.1 MB. **Share:** the census
transform is 0.14–0.17 ms of a ~0.51 ms census entry, so a win here moves the
entry by its share and not by its own ratio.

**R2 — `spacingKernel` is the last one-block kernel on a shipped path.** grid
**1** × 1024 threads, `sm__throughput` **0.9–1.0%**, DRAM **0.0%**, 32 KiB read /
19 KiB written, stalls `barrier` 35% / `wait` 21%. The benchmark's own
free-running split prices it: `goodFeaturesToTrack` is 0.450 ms with spacing
against 0.285 ms without — **37% of the operation on 1/48 of the machine**.
FAST's identical defect was removed this round at 35× on the off-switch ratio;
this one was not. *Honest caveat:* greedy spacing is sequential by construction,
so what the profile licenses is "this is 1/48 of the machine on a shipped path",
not "here is the parallel algorithm". The named candidate is a spatial index
over the accepted points — accepted corners are pairwise at least `minDistance`
apart, so a cell of that side holds at most 4, taking `ranked × kept` to
`ranked × O(1)`.

**R3 — `fusedCandidateKernel`: 509,440 shared-store bank conflicts, and the fix
is not padding.** 51,440 shared store instructions against **509,440**
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` = **9.9 extra wavefronts
per store**; zero load conflicts; `short_scoreboard` 65–67%. The mechanism is in
the source: `kFuseCols = 320`, and thread *t*'s base offset is
`(t/10)*320 + (t%10)*32` — both terms are multiples of 32, so every thread in a
warp starts on bank 0 and they collide on all 32 elements. **The control that
proves it is in the same file**: `responseKernelSliced` pads its stage array to
33 floats and records **zero** conflicts on 23,040 shared stores, with a comment
saying the pad exists "because at 32 the whole warp would hit one bank on every
write." The fused kernel is that sentence, unfixed. *Fix shape, and why not the
obvious one:* `launch__occupancy_limit_shared_mem` already binds this kernel to
7 blocks/SM against the registers' 8, so padding costs occupancy — have a
**warp** cooperatively write one unit's 32 floats instead, which is
conflict-free at zero extra shared memory. Also flagged, smaller:
`bitonicChunkKernel` at 0.36 conflicts per shared-load instruction,
`fastEmitKernel<9,1>` at 0.38, `fastCountKernel<9>` at 0.46.

**R4 — every ballot packer stores one word per warp: 8.0× store-sector
amplification.** Identical at every geometry (table above). `packKernelByteLane`,
which stores four words per warp, is 2.0×; `censusPackedKernel`, fully
coalesced, is 1.0×. Round 2 attributed the byte-lane arm's 2.66× at 4K entirely
to the two deleted software divides; the store-sector count is a **second,
independent mechanism** that write-up did not have. It binds only where the
packer is store-limited — `packQuant` is compute-bound, so fixing its store
shape alone will not move it (see R8). *Prediction:* a warp producing eight
output words (256 pixels) fills a 32 B sector and takes `gST` to 1.0×.

**R5 — `packBits`' byte-lane arm is latency-bound between the roofs and does not
converge.** DRAM 19.1 / 49.6 / 63.9 / **65.4%**, `long_scoreboard` 43–48%
throughout. One 32-bit load per lane per warp-group is **one outstanding load
per thread**; a `uint4` load (16 pixels per lane, 512 B per warp) quadruples
memory-level parallelism against exactly the stall the histogram names. This is
`cuda::threshold`'s remaining gap and it is the shipped sensor-stage packer.

**R6 — the grid-stride index divide survives in `logic`, `reduce` and the
fallback packers.** `countNonZero` tops out at 36.8% of peak DRAM where
`bitwiseAnd` on the identical traversal with 3× the bytes per index reaches
87.9%, with traffic 1.0× optimal on both. Still live in `binaryKernel`,
`notKernel`, `regionKernel`, `batchKernel`, `packKernel` and `packQuantKernel`.
The row-grid shape that deleted it in `pack.cu` transfers unchanged.

**R7 — `denseKernelBitSliced<4>`: 13.7% achieved occupancy, registers binding.**
`launch__occupancy_limit_registers` = 8 blocks/SM (the block is a single warp) →
16.7% theoretical, 13.7% achieved; 239 registers; SM 65.4%, DRAM 1.2%; stalls
`wait` **36%** + `not_selected` 32%. *Precedent, and what does not transfer:*
`denseKernelPacked` had the same signature and the warp-box restructuring took
it to 6 blocks/SM and 40.6% occupancy, 2.45× faster on the profiler's clock —
but that change **also** cut L1 load sectors 4.76×, and the binary kernel's
loads are already only 14,575 KiB, so only the occupancy half of the precedent
applies. *Why it is still worth trying given that PR #57 found register cuts
that changed nothing:* `wait` is a fixed-latency ALU dependency, hideable only
by more resident warps, and it is the top stall at 36% while DRAM sits at 1.2%.
There is nothing else for this kernel to be bound on. The register pressure is
`bestC[kBsStrip = 8][…]` + `bestD[8][8]`, and `kBsStrip` is the same lever
`kBoxStrip` 16 → 4 was. **This is the backend's headline kernel**, so a miss here
costs more than a miss anywhere else.

**R8 — `packQuant` has no fast arm and the profile prices that decision at
4.8×.** SM 46.4 → 66.9% and DRAM 9.9 → 15.0% across the sweep; at 8K, 477.6 µs
against `packBits`' 100.1 µs on 1.16× the traffic. Compute/issue-bound at every
size. The N-bit ingestion path is the slowest thing per byte in the sensor stage,
and it is the path a caller uses for anything above one bit.

**R9 — `trackKernel`: 142 register-indexed constant loads in the inner loop.**
`short_scoreboard` 47% of stalls with **zero** shared memory, **zero** local
memory and no spills. SASS names the source: **142 `LDC … c[0x0][R#+…]`**
(register-indexed constant, which goes through MIO) against 22 uniform `ULDC`,
in a 15,534-instruction kernel. The cause is
`const DeviceLKLevel& lv = a.levels[li];` — a *reference* into the by-value
parameter block, so every `lv.width` / `lv.bits` / plane pointer inside the
iteration loop is a separate indexed constant load. *Fix shape:* copy by value
so the descriptor is read once per level into registers. Registers already bind
occupancy at 4 blocks/SM, so this must be **measured** for register pressure
rather than assumed. *Context, so this is not read as a repair:* `trackKernel`
already beats `cv::cuda::SparsePyrLKOpticalFlow` at frontend density, and the
profile shows the mechanism in one number — OpenCV's `pyrlk::sparseKernel` does
**146 MiB of local loads per launch** and binCV's does **zero**.

**R10 — `responseKernelSliced` is shared-memory-capacity bound and
under-gridded.** `launch__occupancy_limit_shared_mem` = 5 blocks/SM against the
registers' 7 → 41.7% theoretical, **16.0%** achieved, because the grid is only
90 blocks on 48 SMs. `short_scoreboard` 69–71%, and **zero** bank conflicts (the
33-float pad works). This is the `cornerMinEigenVal` row that did not resolve.


## How a difference is decided

Every ratio in this report is a **per-round paired** ratio: each round times
both arms back to back, alternating their order, so a drift that moved both
arms divides out instead of landing on whichever ran second. What is quoted is
the **median** of those per-round ratios, and what decides whether it is real
is `benchmark/measure_util.hpp`'s own rule, carried across to the paired design
in `backends/cuda/benchmark/paired_stats.hpp`:

> A difference smaller than the spread is a null result, and a null result is a
> result. The spread bounds WITHIN-run noise only; run-to-run scatter is a
> separate and sometimes larger number, and an entry that calls a difference
> real should clear the larger of the two.

So a difference must beat **the larger of two noises**, and both are measured
as factors:

- **within-run** — how many times the per-round ratio itself swung inside one
  process, `max / min`. The *ratio's* swing, not the arms': charging it for the
  arms' spreads would charge it for the drift the pairing already removed.
- **run-to-run** — `max / min` of the per-run medians across seven independent
  processes. One process cannot see this, so
  `scripts/aggregate_cuda_runs.py` computes it from a directory of runs.

That sentence decides one thing — whether the *size* of a difference clears the
noise. It does not decide whether the *sign* of the difference is settled, and
on one row in this document the two came apart.

### The spread bounds the magnitude, not the direction

**That test answers one question, and reading it as though it answered every
question produced a verdict the owner ruled wrong (2026-09-19).** The case that
forced it is in this document. Device Lucas-Kanade at the frontend's own
keypoint spacing: binCV faster in **every** paired round, never by less than
1.35×, and the per-round ratio nonetheless swings 2.49× across the rounds —
entirely because the rounds where binCV wins by 4.4× sit so much further from
1.00× than the rounds where it wins by 1.35×. The difference then fails to
exceed that swing, so the rule read literally called 105–0 a null.

**A spread lying wholly on one side of 1.00× is uncertainty about how big the
difference is, not about whether there is one.** Charging it against the
difference treats *"we do not know whether this is 1.35× or 4.4×"* as if it were
*"we do not know whether these arms differ at all"*. Those are not the same
doubt. So there are **three verdicts**:

| verdict | what it says | how the row is quoted |
|---|---|---|
| **DIRECTION ESTABLISHED** | no round crossed 1.00× | a **range**, smallest to largest per-round factor, beside the median and the sign count |
| **A RESULT** | the difference exceeds the larger noise | the median, as before |
| **NULL RESULT** | neither | "the same speed as far as this run can tell" — still a result |

**A row can be both of the first two, and the strong ones are.** They are
different statements — *which arm is ahead is not in doubt* and *the distance
between them exceeds the noise* — so both get printed, because the stronger does
not contain the weaker. Measured on the seven runs behind this section: a
unanimous row whose win varies more than three-fold in size is
direction-established and **not** a result (LK at 204 points, 1.35×–4.36×, 105
of 105); and a row that clears the noise on a 2–103 split is a result whose
direction is **not** established (`threshold`'s byte-lane arm at 1920×1080,
2.39× apart, two rounds the other way). Collapsing either into the other loses
the half that was true.

**No round-count threshold, which is why the sign test's exact p is printed.**
Two unanimous rounds and a hundred unanimous rounds both satisfy "no round
crossed", and they are not equally strong evidence. The tempting fix is a
minimum round count — a project-wide "X is enough" bar invented on the spot,
which is the one thing `CLAUDE.md` says not to do. What is printed instead is
the exact two-sided sign-test p: 1.0 at one round, 0.5 at two, 6.1×10⁻⁵ at
fifteen, 4.9×10⁻³² at the 105 behind these rows. The reader judges strength from
a number on the page, and the verdict gates on nothing.

**A tie breaks the direction verdict, and that is the inconvenient choice.** A
round whose two arms time identically favours neither. Three reasons it has to
count against a verdict that claims *every* round fell one way:

- The sentence the verdict licenses is "faster in N of N rounds". Drop a tie
  from N and that printed sentence is false.
- Ties get **more** common as the clock gets coarser, so excluding them would
  make the verdict easier to earn the *worse* the instrument is. A criterion
  that rewards a worse instrument is not a criterion.
- The tie bucket also holds rounds that were **not measurements at all** — a
  clock that read zero. A tie-blind verdict would let rounds that never happened
  be the ones it ignored. Those are counted separately, so a row says which kind
  it tripped on.

It costs rows, and it is meant to. The arithmetic is pinned by
`PairedStats.ATieBreaksTheDirectionVerdict`: nine rounds at 1.50× and one exact
tie. Every *usable* round fell the same way, so a tie-blind predicate would call
it unanimous — and the tie's own ratio is exactly 1.0, which is the range's
**minimum**, so that predicate would then print "won by **1.00×** to 1.50×",
quoting as a win a round in which nobody won. Remove the tie and the same nine
rounds establish a direction at 1.50×–1.50×. That is the whole content of the
decision. Ties did occur in the sweeps behind this document, all of them genuine
measurements rather than clocks reading zero, and they fell on rows that were
mixed-sign anyway.

The sign *test* still excludes ties from n, which is what a sign test does with
them. That is a statement about a probability model with no third outcome, not
about which observations a verdict may look at.

**What did not change.** The medians, the geometric mean, the factors, the
swap-invariance and the noise predicate itself are exactly as they were.
DIRECTION ESTABLISHED is an **additional** statement about the same rounds, not
a second way to pass the old one — pinned by
`PairedStats.TheOldPredicateIsUntouchedByTheRuling`, and confirmed by outcome:
re-judging 146 published rows **on the same rounds** moved zero of them across
the RESULT/NULL line in either direction. (Re-*measuring* them does move a few:
five of 137 rows common to two sweeps flip, all of them sitting within a few
percent of their own bar. That is the host, not the rule — see the caveat at
the end of the next section.)

**Two things this is not.** It is not a test for disjoint sample ranges. A
range test is vetoed by a single round that is slow in *both* arms — which is
drift, the exact thing pairing exists to cancel — and it passes cleanly
separated arms whose ratio scatters from 1.01× to 1.60×. Separation is still
computed and still printed, as a fact beside the verdict. It is also not a
p-value threshold: the sign test over the paired rounds is reported at every
row and gated on at none, because a project-wide significance bar is exactly
the kind of number `CLAUDE.md` says not to invent.

**Why the deciding quantities are factors and not percentages.** The obvious
spelling — `|median − 1|` against `(max − min) / median`, both as percentages —
is not a comparison at all, because it depends on which arm is the denominator.
`|median − 1|` cannot exceed 100% for the arm that is *faster*, however far
ahead it is, while the spread has no ceiling. Measured on a row re-taken for
this pass — binCV's `erode` on a 5×5 ellipse against `cv::cuda`'s, fifteen
paired rounds at 752×480, one process:

| orientation | difference | spread | verdict |
|---|---|---|---|
| binCV / OpenCV | 93.5% | 267.2% | NULL |
| OpenCV / binCV | 1427.1% | 106.2% | RESULT |

Same rounds, same arms, opposite answers, decided by argument order — one arm
is **fifteen times** the other and one spelling reports "the same speed as far
as this run can tell". In factors both quantities survive the inversion
(**15.27× apart against a 4.62× swing**, either way round), which is the property
the geometric mean is already used for here.
`PairedStats.TheVerdictDoesNotDependOnWhichArmIsTheDenominator` pins it.

**It is not a rare coincidence of one run.** Across the seven processes behind
this section the same pair flips verdict with argument order in **four of
seven**; in the other three the swing happened to be small enough that both
spellings agreed. The factor spelling reads RESULT in all seven. Whether the
wrong spelling changes an answer depends on how noisy the run happened to be,
which is exactly why it cannot be left to chance.

**The median of an even number of ratios is their multiplicative midpoint**, for
the same reason. An arithmetic midpoint does not commute with inverting the
ratio — `(x + y)/2` is not `1/((1/x + 1/y)/2)` — so at even round counts the
verdict depended on argument order even in factors: two rounds whose ratios are
1.0 and 1.5127 read **1.2563× apart one way and 1.2040× the other**. Over a
sweep of random pairs every even-count pair disagreed with its own mirror image,
four of them all the way to opposite verdicts; odd counts were already exact,
because inverting reverses the sorted order and leaves the middle sample where
it was. `backends/cuda/tests/test_cuda_bench_stats` pins the whole of this
against hand-computed values, in 280 checks, and needs no GPU.

### What changed when the rule was applied

Two things happened here, a round apart, and they are separated on purpose.
First the range test was replaced by `measure_util.hpp`'s own
difference-against-spread rule — which moved rows in **both** directions. Then
the owner's 2026-09-19 ruling **added** the direction verdict without touching
that rule, and the table below is a fresh seven-process sweep (105 paired rounds
a row) judged under all three values.

**Re-judging the same rounds moved nothing across the RESULT/NULL line.** Over
146 rows, zero went RESULT→NULL or NULL→RESULT — which is what it means for the
ruling to *add* a verdict rather than relax one. What it added is a verdict for
rows that were unanimous and had nowhere to say so.

| effect | both arms, per-run medians | old verdict | now |
|---|---|---|---|
| **LK at the frontend's own spacing, 204 pts** | `cv::cuda` **0.1475 ms** [0.1314–0.1700] · binCV **0.0792 ms** [0.0787–0.0794] | null (1.84× apart against a 2.49× swing) | **DIRECTION ESTABLISHED** — faster in **105 of 105** rounds, by **1.35× to 4.36×**, p = 4.9×10⁻³² |
| LK @256 pts | `cv::cuda` **0.1467 ms** [0.1370–0.1782] · binCV **0.0728 ms** [0.0718–0.0764] | null | **DIRECTION + RESULT** — 105 of 105, **1.43× to 5.65×**, median 2.00× |
| LK @512 pts | `cv::cuda` **0.1634 ms** [0.1347–0.1844] · binCV **0.1056 ms** [0.1046–0.1059] | null | **DIRECTION ESTABLISHED** — 105 of 105, **1.13× to 3.80×** |
| LK @64 / @128 pts | `cv::cuda` 0.1396 / 0.1412 ms · binCV **0.0523 / 0.0548 ms** | RESULT | DIRECTION + RESULT — 105 of 105, 1.77×–10.95× / 1.78×–4.89× |
| LK @1024 pts | `cv::cuda` **0.1883 ms** · binCV 0.2020 ms | null | **still null** — 74–31, ratio 0.60–1.19, both sides crossed |
| LK @2048 pts (the whole set) | `cv::cuda` **0.3287 ms** · binCV 0.3558 ms | null | **still null** — 98–7, seven rounds crossed |
| `goodFeaturesToTrack`, wall clock | `cv::cuda` **3.7282 ms** [3.3029–4.1832] · binCV **0.8405 ms** [0.8313–0.8587] | *unjudgeable* — the harness reported a 0–0 sign split (see below) | **DIRECTION + RESULT** — 105 of 105, **3.53× to 15.52×**, median 4.42× |
| `threshold` 752×480 | `cv::cuda` 0.0091 ms · binCV 0.0084 ms | null on magnitude | **still null** — 44–60 with 1 round tied |
| `threshold` 1920×1080 | `cv::cuda` 0.0116 ms · binCV **0.0100 ms** | null on magnitude | **still null** — 17–88 |
| `threshold` 3840×2160 | `cv::cuda` 0.0362 ms · binCV **0.0268 ms** | null on magnitude | **still null** — 8–97, and the quoted 1.35× stays unquotable |
| min-eigenvalue response | `cv::cuda` **0.0515 ms** · binCV 0.0590 ms | null | **still null** — 82–23 |
| `cornerMinEigenVal`, bit-sliced arm | — | RESULT | **RESULT, not direction-established** — 3–74, three rounds crossed |
| packer row grid vs grid-stride, 752 / 1920 / 3840 | — | null ×3 | **still null ×3** — 17–86 / 6–99 / 9–96 |
| `packQuant` row grid, all three geometries | — | null ×3 | **still null ×3** — 13–92 / 7–98 / 6–99 |
| packer **byte lane** vs grid-stride @3840×2160 | — | RESULT | **DIRECTION + RESULT** — 105 of 105, **2.01× to 4.56×** |
| `threshold`'s byte-lane arm vs grid-stride @1920×1080 | — | RESULT | **RESULT, not direction-established** — 2–103: two rounds in 105 crossed |
| morphology 3×3 specialization, 752 / 1920 | — | null | **still null** — 20–56 / 8–69 |
| morphology word-parallel border @1920×1080 | — | RESULT | **RESULT, not direction-established** — 1–76: one round in 77 crossed |
| morphology word-parallel border @752×480 | — | null | **still null** — 5–72 |
| `andNot` GRADIENT fusion, 752 / 1920 | — | null | **still null** — 11–66 / 7–70 |
| `cornerSubPixAsync` round-trip rule | device arm vs whole-plane round trip | null at parity | **RESULT in this sweep, against the device arm** — 5–72, 1.44× apart against a 1.39× bar. Marginal, and it read null over the previous fourteen runs |
| `edgeThreshold` vs `cv::cuda` deriv+abs+threshold+or @3840×2160 | — | RESULT | DIRECTION + RESULT — 63 of 63, **15.0× to 61.5×** |

**Seven rows in this sweep are a RESULT without a direction, and that is a
distinction the two-valued table could not draw at all.** Each has one to five
rounds falling the other way out of a hundred: the size clears the noise, the
sign is not unanimous, and both facts are now on the page instead of one of
them.

**Two harness defects surfaced while re-judging, and both were producing
published numbers.**

- `gftt_wall` and the frontend's wall-clock pair assembled their paired summary
  **by hand** instead of through `summarizePaired`, so `roundsFavouringA/B` were
  left at zero. Two published role bars therefore printed a **0–0 sign split and
  p = 1 over rounds that are in fact 105–0**, and took the ratio's median from
  the *arithmetic* even-count midpoint that the factor spelling exists to avoid.
  The `goodFeaturesToTrack` wall row above is the first honest judging of that
  comparison.
- `cuda_sensor_benchmark` never set its per-geometry scope, so its two
  geometries pooled under one key and the aggregation read **the difference
  between the geometries** as run-to-run scatter. Measured on this sweep: pooled,
  the byte-lane pair reads a 2.82× "scatter" and nulls; scoped, 3840×2160 is
  **0.1396 ms against 0.0498 ms — 63 of 63 rounds, 1.65× to 3.47×** — and 752×480
  is a separate null (0.0101 ms against 0.0078 ms, 9–54). That geometry carries **no bar** — it is labelled
  kernel-visibility-only — so no published claim moved; a diagnostic that was
  reading its own sweep as noise stopped doing so.

Four of these deserve a sentence rather than a row.

**The headline was the case that exposed the original defect.** Under the
percentage spelling the binary dense result read NULL, because its faster arm
sits a few multiples above the launch floor and swings while StereoBM's does
not. It is a result under the rule as written; it was a null under a spelling of
it that could not survive inverting the ratio.

**The LK row is what the ruling was for, and it now reads as a comparison rather
than a label.** At the frontend's own spacing, `cv::cuda::SparsePyrLKOpticalFlow`
takes **0.1475 ms** (per-run medians 0.1314–0.1700) and binCV takes **0.0792 ms**
(0.0787–0.0794). binCV is faster in **105 of 105** paired rounds and the two arms
were never seen closer than **1.35×** apart, nor further than 4.36×. That range
*is* the result. The median, 1.84× apart, is quoted beside it and is not a
substitute for it — and it does not clear the 2.49× swing, so the magnitude half
stays a null. **Almost all of the swing lives in the OpenCV arm**: across these
seven runs binCV's own medians move by 1.007× and `cv::cuda`'s by 1.29×.

**`cornerSubPixAsync`'s round-trip rule is still not met, and this sweep says so
more strongly than the last.** Its written ship condition is that the device arm
be *strictly cheaper* than the round trip it replaces, priced against the tighter
of two baselines. Measured as **one paired thing** rather than three
separately-timed medians added together, and against the whole-plane download
(the tighter baseline here by 50×): the device arm reads **0.5656 ms** and the
round trip **0.3911 ms**, with 72 of 77 rounds favouring the round trip — 1.44×
apart against a 1.39× bar, so a RESULT *against* the device arm in this sweep,
where fourteen earlier runs read a null at parity. Either way it is not a pass.
The previous "met in 6/7 runs" came from comparing three independently-drifting
medians. **This is a stop-and-ask**: a measurement contradicting a documented
claim, reported rather than fixed.

**The launch floor, not the rule, is what blocks the internal packer and
morphology arms — and the direction verdict does not rescue them either.** Their
per-round ratio swings 2–3× inside a single process while their per-run medians
agree to within a few percent: at 1920×1080 the row-grid arm's seven medians are
0.7005, 0.7344, 0.7084, 0.7036, 0.7039, 0.7022, 0.7366 — a 1.05× scatter — and 99
of 105 paired rounds favour the arm. But **six rounds crossed**, so the direction
is not established either, and the rule takes the *larger* of the two noises,
which is the within-run one. That is the rule behaving correctly on a host whose
individual 0.4 ms batches are at the mercy of WSL2 scheduling, and it is the
measurable reason these arms stay unquotable here — not an argument about the
kernels. Shrinking it means more enqueues per round, which would move every
number in this report and is therefore a change of its own.

**One caveat that belongs with every number above: on this host the magnitude
half of the verdict is not stable between sweeps, and the direction half is.**
Measured directly — 137 rows are common to two independent sweeps of this branch,
and **five of them flip RESULT↔NULL**, every one sitting within a few percent of
its own bar: LK @256 (2.00× against 1.99× here, 2.25× against 2.80× there), the
byte-lane packer at 1920×1080, the frontend's spacing pair, and the
`cornerSubPix` round trip. **No row's direction verdict flipped.** LK at 204
points is direction-established in both; every row unanimous in one sweep is
unanimous in the other. That asymmetry is itself the argument for the ruling: on
this machine the *sign* of these differences reproduces and their *exact size*
does not, so a row quoted as a range says something that survives a re-run and a
row quoted as a single factor sometimes does not.

## How the numbers were earned

Every arm shipped correct-first, then optimized with its reference arm kept
reachable and both held to the same map:

| stage | binary matcher | census matcher | census transform |
|---|---|---|---|
| reference kernel (one thread per pixel) | 1.71 ms | 39.3 ms | 1.23 ms |
| shared-memory tiling, 8-wide disparity tiles | 0.58 ms | 11.7 ms | — |
| sliding vertical window, 16-row strips | 0.408 ms | 7.75 ms | — |
| packed descriptor layout | — | 0.94 ms | — |
| word-parallel bit-slicing, one thread per word | **0.069 ms** | — | — |
| warp-cooperative separable box | — | **0.369 ms** | — |
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
- **A hardware profiler is available now, and it did not exist for most of what
  is above.** How it was enabled, what it read on every kernel and which earlier
  conclusions it overturned are in *The profiler pass* above; this entry records
  only what that means for reading the rest of the document.

  **Every limiter claim above the frontend section rests on static SASS evidence
  and controlled A/B, because that was all there was.** The frontend, tracking,
  sparse and census sections' limiter claims are hardware counters. Where the
  two have been compared the profiler has corrected four stated limiters —
  `cuda::threshold`'s twice, the census matcher's, the tracker's and the
  five-point solver's — while leaving every one of those decisions intact. That
  is the pattern worth carrying: the profiler has so far changed *reasons*, not
  *verdicts*, and a reason that was wrong is a revisit waiting to be taken from
  the wrong starting point.

  So the stereo, sensor and window numbers are CUDA-event timing, `nvcc -Xptxas
  -v` for the static picture, and controlled A/B against the arm each change
  replaced. That was enough to locate the dense matchers' limiter, but it took
  six experiments where a stall-reason profile takes one run. Two later rounds
  spent one profile each in place of an experiment sweep — the tracker's lane-0
  broadcast arm and the sparse matcher's lane-remapping arm were both
  **predicted to lose and not built**, on one histogram apiece.

  **`cuobjdump -sass` answered more than expected and still does.** It located
  `cuda::threshold`'s two software divides, priced every byte-lane intrinsic the
  sensor families were designed around — several budgeted at one instruction
  and costing six — and it is what named `trackKernel`'s 142 register-indexed
  constant loads (R9) and the census transform's `STACK:72` (R1). A host-enqueue
  probe and a single-call probe together separated OpenCV's kernel time from its
  per-call host cost without a profiler at all.

  **`nsys` and `compute-sanitizer` are both still unusable here** — see the
  profiling section for what each does instead of working. Given a deliberate
  1,020-element overread of a 4-element allocation, `cuda-memcheck` printed
  `ERROR SUMMARY: 0 errors`. **No device out-of-bounds read in this backend is
  observable by any tool available on this machine**, which is why two guards
  are pinned as swept arithmetic invariants instead — the pyramid's source-word
  guard below, and the census box matcher's out-of-row lane guard, whose test
  header states the structural argument rather than papering over the gap.
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
- **The census entry's remaining 1.25× has fallen, and the previous entry here
  predicted the wrong shape.** It read: *"this kernel is at a local optimum for
  its shape… what remains are genuine rewrites"*, and named three — marching a
  thread along x with the window sliding in registers, a two-pass separable box
  filter over a per-disparity cost buffer, and `uint4` loads. The one that
  landed is **the first and second combined and neither as described**: a
  warp-cooperative separable box that slides *vertically* in a register per lane
  and decomposes horizontally over shuffles, with **no cost buffer at all** —
  the ~360 KB the separable idea was priced at is 0 B. It measures **2.49×**
  (re-taken over seven processes it reads 2.51× against a 1.24× bar, 105–0 —
  the same number inside this row's 1.08× run-to-run scatter), and the census
  entry is now **1.47× ahead** of `cv::cuda::StereoBM` rather than 1.25× behind. `uint4` remains unspent and the profile now says why it would
  not pay here (`long_scoreboard` 13.9%, `mio_throttle` 0.05% — the load path is
  no longer the throttle).

  **What is still unspent on this path, recorded and not taken:** SWAR-packing
  two disparities per accumulator word, which aims squarely at the remaining
  23.8% `short_scoreboard`; `__shfl_up_sync` sharing of the right-image words;
  and exploiting K ≤ 16 to settle two pixels per `__popc`, which is a change to
  `censusTransformPacked`'s public layout rather than a matcher change.
- **The plane-layout matcher is kept but is 21.7× slower** than the packed one.
  It ships because it consumes the host's own layout and costs less memory, not
  because it is the fast path; a caller with wide frames should use
  `censusTransformPacked` + `denseDisparityCensusPacked`.
