# CUDA backend

The GPU backend, measured against `cv::cuda` as the best existing GPU option and against
the host library on the same machine. Design:
[ARCHITECTURE §8.5](../ARCHITECTURE.md). What the backend is and how to build it:
[backends/cuda/README.md](../../backends/cuda/README.md).

The backend shares binCV's format and forks its kernels, so every number here sits on top
of a bit-exactness result: `scripts/verify_cuda.sh` proves each device kernel gives the
host library's answer byte for byte, and every optimized arm is held to its own reference
arm's map in the same binary. Speed is what follows once correctness is settled.
[Coverage](#coverage) has the counts.

**This report is three files, and none of them is optional.**

| file | what it holds | when you want it |
|---|---|---|
| **cuda.md** — this one | the headline, what is not delivered, the role bars on both axes, the operations that have no bar at all | you want to know what the backend costs and what it saves |
| [cuda-evidence.md](cuda-evidence.md) | how every figure here was earned: the per-operation narratives, the profiler pass entire, every recorded negative, every withdrawn claim, every "not a result" row | you are about to quote one of these numbers, or you want to know why it should be believed |
| [methodology-timing.md](methodology-timing.md) | how a speed difference is decided here — the paired design, the three verdicts, the worked examples | you are about to read a ratio, a verdict or a sign count |

Every role-bar row below links into the section of [cuda-evidence.md](cuda-evidence.md)
that earned it. A number quoted out of this file without the row behind it is a number
quoted out of its conditions.

## How to read a table here

Set once, and applied in all three files.

- **Both sides' measured values appear, with the unit in the column header.** `0.1459`
  beside `0.0247` in a column headed milliseconds needs no legend: the smaller one is
  faster. A loss needs no marker either, because you can see it.
- **The baseline comes first** — the row or the column a reader needs as the denominator
  before the numerator means anything.
- **Speed and memory never share a column**, and where they share a table the column
  group names its axis and its unit.
- **A ratio is secondary and adjacent, never a replacement.** Where one appears, either
  the column header names the division (`binCV ÷ cv::cuda`) or the cell names the winner
  in words. A bare `Nx` on its own appears nowhere.
- **A condition strip sits above every table** — geometry, clock, stream, run count —
  rather than being buried in the prose under it.

## Conditions you cannot read a number without

### The device, the two clocks, and the CPU arm

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

### One explicit stream on both sides

- **Every cross-library comparison runs both sides on one explicit stream, and
  finding that out moved three numbers in these reports.** OpenCV synchronizes
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

  | call | default stream (ms) | explicit stream (ms) | surcharge — default ÷ explicit |
  |---|---|---|---|
  | `cv::cuda::threshold` | 0.0704 | 0.0102 | 6.91× |
  | `cv::cuda::resize` INTER_AREA | 0.0559 | 0.0091 | 6.17× |
  | `cv::cuda::pyrDown` | 0.0706 | 0.0098 | 7.18× |
  | `cv::cuda` erode 3×3 (0.15 ms kernel) | 0.2125 | 0.1465 | 1.45× |
  | `cv::cuda` median 3×3 (6 ms kernel) | 6.4260 | 6.2180 | 1.03× |
  | *control* — binCV `threshold` | 0.0146 | 0.0142 | 1.03× |
  | *control* — binCV `erode` 3×3 | 0.0095 | 0.0089 | 1.07× |

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

### Two memory meters, and one meter per comparison

**One meter per comparison, named at the number.** Two memory meters appear in
these reports and they do not mix. A `cudaMemGetInfo` delta measures what the
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
footprint, which makes the 6.857× in [the headline](#the-headline) a lower bound on
the memory lead and not a measurement of it.

### The launch floor

Two of the sensor-stage ops and all of binCV's binary morphology sit on this host's launch
floor at the frame sizes a vision pipeline runs, so the floor is quoted beside them rather
than left implicit. It was measured in every run of the sensor-stage benchmark at
**8.66–9.58 µs**, and the occupancy note records it as **11–13 µs** for the shape it was
measuring there. Those are two measurements of the same host and neither is derived from
the other; a figure within a small multiple of either is a launch-cost figure, not a
kernel one.

### How a difference is decided, in four sentences

Every ratio here is a **per-round paired** ratio and the quoted value is its **median**.
Two questions are then asked of the same rounds and both answers printed: **is the
direction settled** — did no paired round cross 1.00× — and **does the size clear the
noise**, which means exceeding the larger of the within-run spread and the run-to-run
scatter. A row can satisfy both, either or neither, and "neither" is a **null result**,
which is itself a result rather than a missing one. The full account — the owner's
2026-09-19 ruling that gave the direction its own verdict, why a tie breaks it, why the
deciding quantities are factors rather than percentages, and the inversion demonstration
that shows what the percentage spelling does — is in
[methodology-timing.md](methodology-timing.md).

## The headline

**The binary dense-disparity path leads `cv::cuda::StereoBM` on both axes at
once** — faster *and* lighter, GPU against GPU, the same result the host binary
path earned against CPU StereoBM. This is the operating point a binCV pipeline
runs: it already holds packed bits, and on bits the dense cost is one XOR per
32-pixel word.

752×480 · 64 disparities · 9×9 support · resident on the device · **kernel-resident**
clock (CUDA events) · **both sides on one explicit stream** · medians of 7 independent
process runs, range in brackets. The ratio is the median **per-round paired** ratio, not a
ratio of the two medians; the verdict is the rule in
[methodology-timing.md](methodology-timing.md).

| arm | time (ms) | binCV against StereoBM | verdict | disjoint |
|---|---|---|---|---|
| `cv::cuda::StereoBM(64, 9)` — the baseline | 0.7438 [0.718–0.842] | — | — | — |
| **binCV binary entry** (pair already packed) | **0.0679** [0.0678–0.0703] | binCV **10.8× faster** | RESULT | 7/7 |
| binCV census entry (wide frames in, transform + match) | 0.5418 [0.527–0.547] | binCV **1.43× faster** | RESULT | 7/7 |
| binCV census matcher alone | 0.4223 [0.398–0.436] | binCV **1.88× faster** | RESULT | 7/7 |

**These are not the role-bars table's figures for the same three rows, and the difference
is reported rather than reconciled.** [The role bars](#the-role-bars) read the binary entry
at **0.0648 ms against StereoBM's 0.7152 — 11.0×**, the census entry at **0.5076 against
0.6996 — 1.47×**, and the census matcher at **0.3692 against 0.6864 — 1.87×**; the census
section in [cuda-evidence.md](cuda-evidence.md#the-census-entry-and-the-layout-that-closed-its-gap)
carries 1.47× too and states that the 1.38× and 1.43× carried at different points are the
same row on earlier sweeps, its run-to-run scatter being 1.14×. Two sweeps on one machine,
both defensible; **which of the two a reader should quote is not settled here.**

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
0 bytes of shared memory. It is described in
[cuda-evidence.md](cuda-evidence.md#the-warp-cooperative-box-matcher-and-the-model-it-refuted).

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

| arm | peak working set (KB/frame) | binCV against StereoBM |
|---|---|---|
| `cv::cuda::StereoBM(64, 9)` — the baseline | 3,072.0 | — |
| **binCV binary entry** (2 bit planes + map) | **448.0** | binCV **6.857× smaller** |
| binCV census entry (2 wide + 2 descriptors + map) | 4,504.0 | binCV **1.466× larger** |

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
profiling section's [R7](cuda-evidence.md#ranked-opportunities) names where the rest of it
is (13.7% achieved occupancy,
registers binding, `wait` the top stall at 36% with DRAM at 1.2%).

Role only: the two match different costs and produce different maps;
correctness is settled against the host library, not against StereoBM. The
memory figures are `cudaMemGetInfo` deltas around each side's working-set
allocation, measured identically on both sides (GpuMat may pool, so StereoBM's
is an upper reading).

## What is not delivered

Stated before anything that is.

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
  [methodology-timing.md](methodology-timing.md#the-case-that-forced-the-ruling-and-where-it-landed).
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
byte. [cuda-evidence.md](cuda-evidence.md#the-sensor-stage-threshold-binarize-edgethreshold)
records what the fix was and where the previously stated limiter was wrong.

## The role bars

Both arms on **one explicit stream**, medians of 7 process runs. The stream is
not a detail: OpenCV synchronizes the whole device on the default stream, a
surcharge measured up to 7.18× here, and a default-stream pair in
`cuda_sensor_benchmark` that printed "7.35× FASTER" for `threshold` has been
deleted rather than repaired — `cuda_role_benchmark` owns that comparison.

### Speed

RTX 3070 Ti · 752×480 unless the row names a geometry · both arms on **one explicit
stream** · kernel-resident clock (CUDA events) · medians of 7 independent process runs ·
from `cuda_role_benchmark`.

**Reading the table.** Both sides' medians are in milliseconds, so the smaller cell is the
faster side and the faster side is bold — a row where binCV's cell is the larger one is a
row binCV lost, and it needs no marker to say so. The ratio column is stated as a division
in its header, `binCV ÷ cv::cuda`, so a value below 1.000 means binCV is faster; the
verdict column states the same comparison the other way up, in words, and those two are
the same fact rather than two. The **verdict** column is the *published* one, set when
these rows were taken under the range test; the rightmost column is what the project's
rule says on a seven-run re-take of every one of them, in the three values the owner's
2026-09-19 ruling defines — *direction established* (no round crossed 1.00×), *a result*
(the size clears the larger noise), or neither (see
[methodology-timing.md](methodology-timing.md#the-case-that-forced-the-ruling-and-where-it-landed)).

| operation | `cv::cuda` arm | cv::cuda (ms) | binCV (ms) | binCV ÷ cv::cuda — below 1.000 is binCV faster | disjoint | verdict | re-taken: apart vs bar |
|---|---|---|---|---|---|---|---|
| [`threshold` 752×480](cuda-evidence.md#the-sensor-stage-threshold-binarize-edgethreshold) | `cv::cuda::threshold` | 0.0091 | **0.0084** | 0.992 | 0/7 | **PASS** — "no longer fails"; magnitude a null | 44–60 with one round tied — null |
| [`threshold` 1920×1080](cuda-evidence.md#the-sensor-stage-threshold-binarize-edgethreshold) | ″ | 0.0116 | **0.0100** | 0.889 | 0/7 | **PASS** — magnitude a null | 17–88 — null |
| [`threshold` 3840×2160](cuda-evidence.md#the-sensor-stage-threshold-binarize-edgethreshold) | ″ | 0.0362 | **0.0268** | **0.743** | 1/7 | **PASS** — magnitude a null | 8–97 — null; the 1.35× stays unquotable |
| [describe, N=1000](cuda-evidence.md#descriptor-matching-which-is-where-the-format-was-supposed-to-pay) | `cv::cuda::ORB::computeAsync` | 0.1070 | **0.0107** | **0.108** | **7/7** | **MET** — binCV **9.3×** faster | 9.25× vs 2.10× — **a result** |
| [FAST](cuda-evidence.md#fast-the-bar-was-missed-and-the-ordering-is-why--now-closed-on-both-axes) | `FastFeatureDetector` | 0.1459 | **0.0247** | **0.166** | **7/7** | **MET** — binCV **6.01×** faster, where it was **14.84× behind** | 6.01× vs 3.39× — **a result** |
| [`goodFeaturesToTrack` (wall)](cuda-evidence.md#the-frontend-on-device) | `createGoodFeaturesToTrackDetector` | 3.7282 | **0.8405** | **0.226** | **7/7** | **SHIPS** — binCV **5.3×** faster | **105 of 105, by 3.53× to 15.52×** — direction established **and** a result (4.42× vs 3.16×) |
| [min-eigenvalue response](cuda-evidence.md#where-bincv-has-no-structural-advantage-said-plainly) | `createMinEigenValCorner` | **0.0515** | 0.0590 | 1.146 | **0/7** | **still not a result** — `cv::cuda` ahead on the median, neither direction established | 82–23, 23 rounds crossed — null on both halves |
| [Lucas-Kanade, 204 pts](cuda-evidence.md#the-role-bar-and-the-density-at-which-it-stops-holding) | `SparsePyrLKOpticalFlow` | 0.1475 | **0.0792** | **0.537** | **7/7** | **MET at frontend density** — binCV ahead in 105 of 105 rounds | **faster in 105 of 105 rounds, by 1.35× to 4.36×** — direction established; magnitude a null (1.84× vs 2.49×) |
| [Lucas-Kanade, 2048 pts](cuda-evidence.md#the-role-bar-and-the-density-at-which-it-stops-holding) | ″ | **0.3287** | 0.3558 | 1.080 | **0/7** | **not a result — the crossover** | 98–7, seven rounds crossed — null on both halves |
| [descriptor matching, 5000²](cuda-evidence.md#descriptor-matching-which-is-where-the-format-was-supposed-to-pay) | `BFMatcher::knnMatchAsync(k=2)` | 1.9491 | **0.2189** | **0.109** | **7/7** | **MET** — binCV **9.1×** faster | 9.11× vs 1.29× — **a result** |
| [census matcher](cuda-evidence.md#the-census-entry-and-the-layout-that-closed-its-gap) | `cv::cuda::StereoBM(64,9)` | 0.6864 | **0.3692** | **0.536** | **7/7** | **MET** — binCV **1.87×** faster | 1.94× vs 1.37× — **a result** |
| [census entry (2 transforms + match)](cuda-evidence.md#the-census-entry-and-the-layout-that-closed-its-gap) | ″ | 0.6996 | **0.5076** | **0.723** | **7/7** | **MET on speed, LOSES on memory** — a split verdict, stated as one | 1.47× vs 1.25× — **a result** |
| [binary entry](cuda-evidence.md#dense-disparity-both-entries) | ″ | 0.7152 | **0.0648** | **0.091** | **7/7** | **MET** — binCV **11.0×** faster | 10.95× vs 1.78× — **a result** |
| [block matching](cuda-evidence.md#block-matching) | `SparsePyrLKOpticalFlow` | 0.2320 | **0.0540** | **0.230** | **7/7** | speed MET; **accuracy floor unset** — ship-blocked | not re-taken this round |

**Three rows changed direction since the previous round, and one row is new
information rather than a better number.** FAST went from 14.84× behind to 6.01×
ahead and `goodFeaturesToTrack` from 2.659× behind to 5.3× ahead, both on the
ordering rewrites in
[cuda-evidence.md](cuda-evidence.md#the-frontend-on-device). The LK row is the first tracking measurement
in this backend. And the census entry's row is a **split verdict** — ahead on
speed, behind on memory — which is stated in the headline and not resolved here.
FAST was the other split verdict and is no longer one: [the memory table](#memory)
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

### Memory

RTX 3070 Ti · 752×480 · `cudaMemGetInfo` delta taken identically **on both sides** — the
only meter readable across libraries — each side replicated until its own total clears eight of this
driver's measured 2 MB units, and never mixed with binCV's own allocation sums. The
lighter side is bold.

| operation | `cv::cuda` arm | cv::cuda (KB) | binCV (KB) | which is smaller |
|---|---|---|---|---|
| `threshold` | `cv::cuda::threshold` | 1024.0 | **416.0** | binCV, by **2.46×** |
| describe, N=1000 | `cv::cuda::ORB::computeAsync` | 2048.0 | **48.0** | binCV, by **42.67×** |
| descriptor matching, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 8277.3 | **400.0** | binCV, by **20.7×** |
| LK tracker resident state | `SparsePyrLKOpticalFlow` | 1408.0 | **448.0** | binCV, by **3.14×** |
| `goodFeaturesToTrack` | `createGoodFeaturesToTrackDetector` | 10240.0 | **1920.0** | binCV, by 5.33× |
| corner response | `createMinEigenValCorner` | 10240.0 | **2048.0** | binCV, by 5.00× |
| FAST @ capacity 32,768 | `FastFeatureDetector` | 680.0 | **432.0** | binCV, by **1.574×** — it was 1.545× **larger** one round ago |
| census entry working set | `cv::cuda::StereoBM(64,9)` | **3072.0** | 4512.0 | **`cv::cuda`**, by **1.47×** — the published ratio for this row is 0.68× |

**One of these rows goes the other way, and it is printed as it reads.** The
helper that prints them used to say "binCV smaller by that factor" whatever the
ratio was — which is a claim rather than a reading on any row where OpenCV is
smaller. It now follows the number. FAST was the second such row and no longer
is; the census entry remains one, and its loss is **inherent to census rather
than a defect in this implementation** — see [the headline](#the-headline). The census-entry row is the
first time both sides of that path were metered in one region.

**FAST's row is now read over 256 working sets a side — this file's own default
`kReplicas` — where it was the outlier at 32.** That is not a different meter,
only a finer one, and the change is stated because the row moved by 6% between
two harnesses that both satisfied the eight-unit rule at 64.

## Verdicts recorded OUTSTANDING

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
- **`shift`**, as recorded in
  [cuda-evidence.md](cuda-evidence.md#the-pyramid-the-resident-ladder-and-shift).

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

## Coverage

`scripts/verify_cuda.sh` proves each device kernel gives the host library's answer byte
for byte — **seventeen suites, 194,975 checks in the Release configuration and 194,922 in
the Debug one** — and every optimized arm is held to its own reference arm's map in the
same binary. The two counts differ by design rather than by accident: a suite exercising
a narrowed domain can only test the half of that contract its configuration has — the
assertion is live in Debug, the error return is reachable in Release — and each such
suite prints which half it ran instead of silently shrinking.

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

**Which operations have device arms.** Twenty-two of the twenty-seven host operation
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
backlog: five-point RANSAC was measured on device and **lost by at least 15.9×** — the
median of its seven runs is 18.58×, range 15.88–19.54 — so
nothing shipped (see
[the geometry section](cuda-evidence.md#the-geometry-and-the-measurement-that-says-it-stays-on-the-host)). `denseCensusBox` adds a second packed matcher beside the first. The
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
which is why it is a gate rather than a convenience.

## Reproduce

From the repository root:

```bash
cmake -S . -B build -DBINCV_CUDA=ON -DBINCV_BUILD_BENCHMARKS=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.1/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=g++-9
cmake --build build -j
```

| table | binary |
|---|---|
| [the role bars](#the-role-bars), both axes | `cuda_role_benchmark` — one process, one launch floor, every pair interleaved on one explicit stream, `cudaMemGetInfo` on *both* sides of every memory figure |
| the headline's memory table | `cuda_role_benchmark stereo`, section 7b |
| dense disparity, the optimization ladder | `cuda_dense_benchmark` |
| the foundation ops | `cuda_foundation_benchmark` |
| the sensor stage | `cuda_sensor_benchmark` |
| morphology and the medians | `cuda_window_benchmark`, `cuda_median_benchmark` |
| the pyramid ladder and `shift` | `cuda_pyramid_benchmark` |
| the StereoBM comparison, with a cudastereo-enabled OpenCV | `cuda_stereobm_benchmark` |

The GPU-against-GPU rows need an OpenCV built with the relevant `cuda*` modules, which no
packaged OpenCV ships; point `-DBINCV_CUDA_OPENCV_DIR=<prefix>` at such a build. Every
target that wants one is always built and reports its role bar as BLOCKED or OUTSTANDING
when it is absent.

**No raw output for these tables is committed under [logs/](logs/)**, which holds host
benchmark output only. Each table above names the binary that produces it instead.
