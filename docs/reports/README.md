# binCV measurement reports

What binCV's **operations** cost and what they save, each measured against its OpenCV
equivalent, on two CPUs and one GPU.

Every report names its denominator, publishes **both sides' measured values with the ratio
beside them**, gives speed and memory together, and names the command that reproduces it.
Losses sit in the same tables as wins: every unit here is one where smaller is better, so a
row whose binCV cell is the larger number is a row binCV lost, and it needs no marker.

| report | what it covers |
|---|---|
| [primitives.md](primitives.md) | logic, reductions, denoise, morphology, derivative, pyramid downsample |
| [features.md](features.md) | corner detection, FAST, descriptors, matching, optical flow |
| [stereo.md](stereo.md) | dense disparity against `cv::StereoBM` |
| [cuda.md](cuda.md) | the CUDA backend, GPU against GPU |
| [footprint.md](footprint.md) | the memory result itemized, and the speed declined to protect it |
| [limits.md](limits.md) | where binCV ties, loses, or stops paying at all |
| [feature-tracking.md](feature-tracking.md) | the assembled feature tracking pipeline — not an operation; see [Assembled pipelines](#assembled-pipelines) |
| [methodology-memory.md](methodology-memory.md) | how memory is measured, and the errors that shaped it — read before quoting a memory number |
| [methodology-timing.md](methodology-timing.md) | how a difference between two timings is judged real — read before quoting a speed ratio |

Raw output for the host tables is in [logs/](logs/). The stereo and CUDA tables have no
committed logs; those two reports name the binary that produces each number instead.

## At a glance

**One row per operation, and the two host machines are columns.** x86-64 and aarch64 are
different measurements against different OpenCV builds on different hardware; they are never
averaged, and neither stands in for the other. The assembled feature tracking pipeline
is not in these tables — it is not an operation — and is
[further down](#assembled-pipelines).

### Speed, CPU

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

640×480 and `uint32_t` words unless the row names otherwise, one thread on both sides. Each
row names its own unit, and on all of them the smaller number is the faster side.

**Each ratio carries the bootstrap 95% interval of its launches** — thirty on x86-64, ten on
the device, seven for dense disparity — and each time cell is those launches' median. The ratio is formed inside each
launch, so it is not the quotient of the two cells beside it.
[methodology-timing.md](methodology-timing.md#the-protocol-each-host-needs) says why the two
counts differ.

<!-- figure-check values="OpenCV, x86-64|binCV, x86-64|x86-64 ratio|OpenCV, aarch64|binCV, aarch64|aarch64 ratio" source="source" -->
| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio | source |
|---|---|---|---|---|---|---|---|---|
| `bitwiseAnd`, ns/pixel | `cv::bitwise_and` | 0.02823 | 0.002810 | 9.97× [9.82, 10.28] | 0.62656 | 0.02369 | 26.68× [26.09, 27.37] | [primitives.md](primitives.md) |
| `countNonZero`, ns/pixel | `cv::countNonZero` | 0.01501 | 0.009270 | 1.62× [1.61, 1.63] | 0.16921 | 0.06365 | 2.658× [2.618, 2.673] | [primitives.md](primitives.md) |
| denoise, 3-pixel median, ns/pixel | composed `cv::min` / `cv::max` | 0.1887 | 0.009865 | 19.09× [18.94, 19.38] | 3.4379 | 0.05941 | 57.71× [56.91, 58.03] | [primitives.md](primitives.md) |
| spatial derivative, both axes, ns/pixel | `cv::filter2D` ×2 | 0.5156 | 0.04645 | 11.12× [11.07, 11.28] | 5.0430 | 0.20753 | 24.28× [24.13, 24.51] | [primitives.md](primitives.md) |
| `erode` 3×3 rect, ns/pixel | `cv::erode` | 0.1013 | 0.09595 | 1.053× [1.035, 1.066] | 0.73595 | 0.72189 | 1.021× [0.991, 1.040] | [primitives.md](primitives.md) |
| `erode` 5×5 ellipse, ns/pixel | `cv::erode` | 0.2238 | 0.6985 | 0.319× [0.318, 0.323] | 1.85196 | 3.59587 | 0.514× [0.510, 0.522] | [primitives.md](primitives.md) |
| `pyrDown`, 1 bit in, µs/call | `cv::pyrDown` on `CV_8U` | 47.70 | 30.70 | 1.556× [1.536, 1.597] | 516.5 | 93.8 | 5.509× [5.480, 5.549] | [primitives.md](primitives.md) |
| `pyrDown`, 8 bits in, µs/call | `cv::pyrDown` on `CV_8U` | 47.70 | 2040.0 | 0.0235× [0.0233, 0.0242] | 516.5 | 7359.9 | 0.0701× [0.0698, 0.0706] | [limits.md](limits.md) |
| optical flow, 140 points, ms/call | `cv::calcOpticalFlowPyrLK` | 3.978 | 0.5585 | 7.19× [6.89, 7.40] | 23.400 | 2.838 | 8.227× [8.189, 8.284] | [features.md](features.md) |
| BRIEF descriptors, 1000 kpts, ms | `cv::ORB::compute` | 0.639 | 0.123 | 5.18× [5.15, 5.22] | 7.167 | 0.658 | 10.81× [10.59, 11.17] | [features.md](features.md) |
| Hamming matching, kNN=2 over 1000×1000, ms | `cv::BFMatcher` | 9.071 | 1.916 | 4.70× [4.65, 4.79] | 38.187 | 19.520 | 1.953× [1.944, 1.972] | [features.md](features.md) |
| `goodFeaturesToTrack`, ns/pixel | `cv::goodFeaturesToTrack` | 8.807 | 6.368 | 1.383× [1.350, 1.426] | 58.338 | 24.099 | 2.421× [2.412, 2.424] | [features.md](features.md) |
| FAST, wide image, ms/call | `cv::FAST` | 0.359 | 0.345 | 1.039× [1.033, 1.048] | 2.910 | 3.025 | 0.962× [0.961, 0.963] | [features.md](features.md) |
| FAST, bit-plane, µs/call | `cv::FAST` | 265.2 | 180.2 | 1.472× [1.470, 1.474] | 2048.2 | 865.8 | 2.365× [2.363, 2.370] | [features.md](features.md) |
| dense disparity, ms/frame | `cv::StereoBM` | 12.675 | 10.405 | 1.218× [1.199, 1.240] | 79.90 | 60.57 | 1.319× | [stereo.md](stereo.md) |

Four rows carry a qualification their report states and a table cell cannot:

- **`pyrDown` at 8 bits in is the boundary of the whole idea, not a regression.** Both sides
  store a byte there, so there is nothing for bit-slicing to skip.
- **`erode` on a 5×5 ellipse is a deliberate trade**, not an unfinished kernel: a
  non-separable element costs one shifted-OR per set element, and the fused kernel was kept
  because it holds 8× less. [footprint.md](footprint.md) prices it.
- **The denoise and derivative rows had no aarch64 measurements for three rounds**, only the
  published ratio. The device sweep recorded them, and the ratios they produce are the
  published ones; see [Published as a ratio only](#published-as-a-ratio-only).
- **The dense-disparity interval is the one that cannot be paired.** Its two arms are
  separate binaries, so the launches cannot be matched up and the interval comes from
  resampling two sweeps independently — wider, and the weakest interval in the table.

### Speed, GPU

RTX 3070 Ti · 752×480 unless the row names a geometry · both arms on **one explicit stream**
· kernel-resident clock · medians of 7 independent process runs. Milliseconds, so the
smaller number is the faster side.

<!-- figure-check values="cv::cuda, ms|binCV, ms|ratio" source="source" -->
| operation | `cv::cuda` arm | cv::cuda, ms | binCV, ms | ratio | source |
|---|---|---|---|---|---|
| dense disparity, binary entry | `cv::cuda::StereoBM(64, 9)` | 0.7152 | 0.0648 | 11.0× | [cuda.md](cuda.md) |
| dense disparity, census entry | ″ | 0.6996 | 0.5076 | 1.47× | [cuda.md](cuda.md) |
| FAST | `cv::cuda::FastFeatureDetector` | 0.1459 | 0.0247 | 6.01× | [cuda.md](cuda.md) |
| descriptor matching, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 1.9491 | 0.2189 | 9.1× | [cuda.md](cuda.md) |
| describe, N=1000 | `cv::cuda::ORB::computeAsync` | 0.1070 | 0.0107 | 9.3× | [cuda.md](cuda.md) |
| `goodFeaturesToTrack`, wall clock | `createGoodFeaturesToTrackDetector` | 3.7282 | 0.8405 | 5.3× | [cuda.md](cuda.md) |
| optical flow, 204 points | `SparsePyrLKOpticalFlow` | 0.1475 | 0.0792 | 1.35×–4.36×, per round | [cuda.md](cuda.md) |
| optical flow, 2048 points | ″ | 0.3287 | 0.3558 | null result | [cuda.md](cuda.md) |
| min-eigenvalue response | `createMinEigenValCorner` | 0.0515 | 0.0590 | null result, 1.15× apart | [cuda.md](cuda.md) |

**The two rows where binCV's cell is larger are published as null results, not as losses.**
Both are inside this host's noise — 0 of 7 runs disjoint on each — so [cuda.md](cuda.md)
records that neither direction is established rather than claiming OpenCV won. Two more rows
are qualified there: optical flow's direction is settled by 105 of 105 paired rounds but its
magnitude is not, which is why its cell is a range; and `goodFeaturesToTrack`'s denominator
swings 3.24–13.61 ms across runs because `cv::cuda`'s spacing filter runs on the CPU. GPU
FAST's direction is certain in all 21 runs taken and its size is not — the per-run ratio
moves between 3.44× and 6.26×.

The two dense-disparity rows come from one of cuda.md's
[two sweeps](cuda.md#speed-operation-by-operation) of the same machine. Which sweep to quote
is not settled there; both rows here are read off the same one rather than picked across two,
and cuda.md publishes the other sweep's readings beside them.

### Memory, CPU

Peak working set of one call, computed from buffer geometry, so it is exact and **identical
on both architectures** — one column pair, not two. Read
[methodology-memory.md](methodology-memory.md) before quoting any of it.

<!-- figure-check values="OpenCV|binCV|ratio" source="source" -->
| operation | measured against | OpenCV | binCV | ratio | source |
|---|---|---|---|---|---|
| `bitwiseAnd` / `Or` / `Xor` / `Not`, bytes | `cv::bitwise_*` | 921,600 | 115,200 | 8.0× | [primitives.md](primitives.md) |
| `countNonZero`, per input plane, bytes | `cv::countNonZero` | 307,200 | 38,400 | 8.0× | [primitives.md](primitives.md) |
| denoise, 3-pixel median, bytes | composed `cv::min` / `cv::max` | 2,150,400 | 76,800 | 28.0× | [footprint.md](footprint.md) |
| spatial derivative, both axes, bytes | `cv::filter2D` ×2 | 1,536,000 | 192,000 | 8.00× | [footprint.md](footprint.md) |
| `erode` / `dilate` 3×3, bytes | `cv::erode` / `cv::dilate` | 614,400 | 76,800 | 8.00× | [footprint.md](footprint.md) |
| `morphologyEx(MORPH_OPEN)`, bytes | `cv::morphologyEx` | 614,400 | 115,200 | 5.33× | [footprint.md](footprint.md) |
| `goodFeaturesToTrack`, bytes | `cv::goodFeaturesToTrack`, binarized | 9,014,976 | 1,580,064 | 5.71× | [footprint.md](footprint.md) |
| FAST input plane, bytes | `cv::FAST` on `CV_8U` | 360,960 | 46,080 | 7.83× | [footprint.md](footprint.md) |
| dense disparity, working set | `cv::StereoBM` | ≥ 722 KB, output alone | 32.4 KB scratch, 1 B/px out | ~22× | [stereo.md](stereo.md) |

`cv::StereoBM`'s figure is its output buffer alone, before its internal buffers, which the
tooling in [methodology-memory.md](methodology-memory.md) cannot observe from outside — so
binCV's lead on that row is a lower bound. `morphologyEx` is 5.33× rather than 8× because
binCV's fused kernel needs a caller-provided scratch frame where `erode` and `dilate` need
none.

### Memory, GPU

`cudaMemGetInfo` delta taken identically on both sides — the only meter readable across
libraries — never mixed with binCV's own allocation sums.

<!-- figure-check values="cv::cuda, KB|binCV, KB|ratio" source="source" -->
| operation | `cv::cuda` arm | cv::cuda, KB | binCV, KB | ratio | source |
|---|---|---|---|---|---|
| describe, N=1000 | `cv::cuda::ORB::computeAsync` | 2048.0 | 48.0 | 42.67× | [cuda.md](cuda.md) |
| descriptor matching, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 8277.3 | 400.0 | 20.7× | [cuda.md](cuda.md) |
| dense disparity, binary entry, per frame | `cv::cuda::StereoBM(64, 9)` | 3072.0 | 448.0 | 6.857× | [cuda.md](cuda.md) |
| `goodFeaturesToTrack` | `createGoodFeaturesToTrackDetector` | 10240.0 | 1920.0 | 5.33× | [cuda.md](cuda.md) |
| corner response | `createMinEigenValCorner` | 10240.0 | 2048.0 | 5.00× | [cuda.md](cuda.md) |
| LK tracker resident state | `SparsePyrLKOpticalFlow` | 1408.0 | 448.0 | 3.14× | [cuda.md](cuda.md) |
| `threshold` | `cv::cuda::threshold` | 1024.0 | 416.0 | 2.46× | [cuda.md](cuda.md) |
| FAST at capacity 32,768 | `cv::cuda::FastFeatureDetector` | 680.0 | 432.0 | 1.574× | [cuda.md](cuda.md) |
| dense disparity, census entry, per frame | `cv::cuda::StereoBM(64, 9)` | 3072.0 | 4512.0 | `cv::cuda` smaller, by 1.47× | [cuda.md](cuda.md) |

**The census entry is the one memory row binCV loses, and it is not a defect to be fixed.**
The census transform expands 8 bits per pixel into a 32-bit descriptor word, so the two
transformed images are 2,820 KB before a disparity map exists, where StereoBM works on the
8-bit frames directly. It is faster on the same run; [cuda.md](cuda.md) records that split
verdict as unmade. `cv::cuda`'s figures are upper readings — `GpuMat` may pool and pads its
pitch — so binCV's lead on the other rows is a lower bound.

### Published as a ratio only

**This list is down to one row.** The rest had both their measurements taken by the device
sweep, and each ratio the sweep produced is the one that was published beside them —
57.66× reads 57.71×, 24.28× reads 24.28×, 13.70× reads 13.76×, and `countNonZero` at
`uint64_t` reads 1.953× against a published 1.95×. What was missing was never the ratio; it
was the two numbers underneath it.

<!-- figure-check values="published ratio, x86-64|published ratio, aarch64" source="source" -->
| operation | measured against | published ratio, x86-64 | published ratio, aarch64 | source |
|---|---|---|---|---|
| `cornerSubPix` | `cv::cornerSubPix` | ~13×, binCV faster | measured above | [features.md](features.md) |

The `cornerSubPix` x86-64 cell stays here because no x86 sweep of that benchmark exists; its
aarch64 pair moved into the table above.

## Assembled pipelines

**These are programs this project wrote, not operations binCV offers**, and they are down
here for that reason. The *feature tracking pipeline* is sensor stage, pyramid, derivatives,
corner detection, tracking and keypoint lifecycle — what a visual-odometry system runs ahead
of its optimizer, and what that field calls a VIO frontend. Two of them are
built for this measurement — one calling only binCV, one calling only OpenCV
(`cv::filter2D`, `cv::buildOpticalFlowPyramid`, `cv::goodFeaturesToTrack`,
`cv::calcOpticalFlowPyrLK`) — and run over 1709 consecutive frame pairs of EuRoC
`V1_02_medium`. Each side builds its own binary frame and the two are bit-identical, so this
compares the two pipelines rather than two different inputs.

What it is evidence for is that the operations **compose**: the per-call results above do
not cancel out when a real pipeline runs them. It is not a claim about anyone else's
pipeline, and it is not the number to compare against a library call.

<!-- figure-check values="OpenCV, x86-64|binCV, x86-64|x86-64 ratio|OpenCV, aarch64|binCV, aarch64|aarch64 ratio" source="source" -->
|  | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio | source |
|---|---|---|---|---|---|---|---|
| time, ms/frame | 3.7275 | 1.0215 | 3.658× [3.633, 3.681] | 23.588 | 5.097 | 4.620× [4.596, 4.628] | [feature-tracking.md](feature-tracking.md) |

Peak working set is computed from buffer geometry and is identical on both architectures, so
it is one pair rather than two:

<!-- figure-check values="OpenCV|binCV|ratio" source="source" -->
|  | OpenCV | binCV | ratio | source |
|---|---|---|---|---|
| peak working set, bytes | 2,719,832 | 436,704 | 6.23× | [footprint.md](footprint.md) |

Flow agrees with OpenCV's to 0.0437 px at the median;
[feature-tracking.md](feature-tracking.md) carries the stage breakdown, the agreement
figures and the spreads. The **CUDA resident pipeline** is the
same idea on the device and is down here for the same reason — it is measured against binCV's
own CPU pipeline rather than against OpenCV, at 5.62× faster ([cuda.md](cuda.md)).

## What these are not

Not a survey of binCV against every alternative, and not tuned comparisons. Every number is
one build of binCV against one build of OpenCV on one machine, on a single commit.

Nothing here is a claim about a target that has not been measured. 32-bit ARM Cortex-A and
RISC-V are supported and appear nowhere in these reports because neither has been built and
timed. Cortex-M has been built and partly measured on an STM32H753ZI — [limits.md](limits.md)
says what that covers — but it produced no OpenCV comparison, so it is not here either.

## The machines

Neither CPU stands in for the other: results move a long way between them, and always
because of what OpenCV's two builds do rather than what binCV's code does. Hamming matching
is 4.70× on x86 and 1.953× on the device; bit-plane FAST goes the other way, 1.47× against
2.365×; and the bit width at which a bit-sliced pyramid stops beating `cv::pyrDown` differs by
several bits between them.

| | **x86-64** — development host | **aarch64** — reference device |
|---|---|---|
| CPU | AMD Ryzen 5 5600X, 6 cores / 12 threads | Broadcom BCM2711, Cortex-A72, 4 cores |
| cache | 32 KiB L1d per core, 512 KiB L2 per core, 32 MiB shared L3 | 32 KiB L1d per core, 1 MiB shared L2 |
| OS | Ubuntu 22.04 under WSL2 | Raspberry Pi OS Lite, 64-bit |
| compiler | g++ 11.4.0, Release `-O3 -DNDEBUG` | g++ 14.2.0, Release |
| OpenCV | 4.8.0-dev, baseline SSE3, dispatching through AVX-512 | 4.10.0, NEON baseline |
| binCV vector paths | `POPCNT`, AVX2 selected at run time | NEON |

The GPU rows are a third machine, the CUDA backend's alone: an NVIDIA GeForce RTX 3070 Ti
(SM 8.6, 48 SMs, ~608 GB/s) under WSL2, CUDA 11.1 nvcc and g++-9, measured against the
`cv::cuda` counterpart of each operation.

A 64-bit OS is a requirement rather than a preference: on 32-bit ARM every `uint64_t`
operation is synthesised from 32-bit pairs, which would measure the compiler rather than the
machine.

## How the numbers are taken

**The denominator is OpenCV on the same content stored as `CV_8U`** — one bit per pixel for
binCV, a byte holding `{0, 1}` for OpenCV, same image, same parameters, same border. Where
the comparison is against a composed sequence of OpenCV calls rather than a stock one, the
report says so and charges the baseline only for the work binCV also does.

**Both sides get one thread.** binCV is serial unless a caller installs a threading backend
and OpenCV is not; left at its default, a comparison on a multi-core box measures parallelism
and reads as implementation. Every benchmark pins `cv::setNumThreads(1)` and prints the count
it got.

**Both sides are SIMD**, and each build's configuration is printed into the logs rather than
assumed: the x86 OpenCV dispatches through AVX-512, the device build has NEON as its
baseline, and binCV's live paths come from `simdStatusString()` — `AVX2=yes
popcount=hardware` on x86, `NEON=yes` on the device.

**Correctness is checked before speed.** Every comparison first asserts that the two sides
computed the same image — bit-exact for Tier 1, a stated agreement bound for Tier 2. A
benchmark whose arms disagree fails rather than reporting a ratio.

**Speed is the median of many interleaved batches**, minimum, maximum and spread reported
beside it. Arms run round-robin so drift moves all of them together; results are consumed
through a `volatile` sink and inputs are varied, because a loop whose result is unused is
deleted by the optimizer.

**That spread is within one process, and a process cannot see past itself.** Whatever a
launch pays once — where the allocator landed, which neighbour the scheduler put on the
sibling core, what the clock was doing when the batch size was calibrated — is constant
inside the process, so it moves none of the batches it times. It moves between them.
`scripts/run_launches.sh` runs a benchmark as N separate pinned processes and
`scripts/aggregate_launches.py` reports both halves side by side: the within-run spread the
harness printed, the run-to-run scatter it could not, a bootstrap interval on the median
across launches, and **the smallest difference that many launches can resolve on that row**.
A row seen in one launch gets no interval at all — blank because a single process carries no
run-to-run information, not because it has none to carry.

**Memory is the peak working set of a call, computed from buffer geometry** — which is why it
is exact and identical on both architectures, and it works because no binCV kernel allocates.
**Where the OpenCV side allocates internally, buffer arithmetic cannot see it**:
`cv::morphologyEx` was measured at the allocator rather than assumed, which moved binCV's
advantage there from 8.0× to 5.33×. A per-buffer ratio is not the metric — a bit-plane is
eight times smaller than a byte plane by construction and saying so measures nothing.
[methodology-memory.md](methodology-memory.md) has the instruments and the four measurement
errors this project published, including one that made OpenCV look 17× smaller than it is.

**A synthetic scene must be noisy enough to separate a good estimator from a lucky one.**
The RANSAC estimators taught this: their scenes generated every inlier exactly from the
transform, which made a missing least-squares refit a no-op — invisible to every test while
costing 13× in accuracy on data with noise in it.

### On the reference device

A Pi 4 will produce stable-looking numbers that are wrong, so four conditions are enforced by
the runner rather than remembered: the architecture is asserted to be `aarch64`; the governor
is pinned to `performance` for the run and restored after, because on `ondemand` a short
benchmark measures the ramp between 600 MHz and 1.5 GHz; the process is pinned to one core
with `taskset`; and throttle state is read before and after, a change during a run
invalidating it. Two runs in this project's history were discarded that way. The environment
block each run prints is at the top of every aarch64 log in [logs/](logs/).

**The throttle flag alone is not enough on a Pi that has ever throttled.** `get_throttled`
returns sticky history, so a board reading `0x80000` — soft temperature limit *has occurred*,
at some point since boot — reads the same before and after a run that throttled and one that
did not. The device sweep behind the current figures sampled the core's actual clock every
two seconds instead: 1,183 samples across the whole run, every one at 1,800,000 kHz, peak
65.2 °C against a 80 °C limit. That is the evidence the numbers were not taken on a ramp; the
flag can only say nothing new happened, which on a sticky flag it cannot.

### On the x86-64 host

**Every x86-64 figure in these reports is now the median of thirty pinned launches with a
bootstrap 95% interval**, and until this round none of them was. All 24 committed x86 logs
were single process launches, and `benchmark/measure_util.hpp` reports *within-run* spread by
construction — a process cannot see what it paid once. So every x86 figure was one draw from a
distribution nobody had characterised. Not wrong; unexamined, which is not a state the
"commit the benchmark" rule leaves room for.

The desktop under WSL2 is not timing-grade and the launch sweep is how much it is not.
goodFeaturesToTrack there prints a ~21–39% within-run spread and the arms it times scatter
12–30% **across thirty launches of the same binary** — so one launch has returned anything
from 0.86× to 1.61× on rows whose interval is a few per cent wide. Thirty launches put a 95%
interval of [1.350, 1.426] around a median of 1.383×, which is ±2.8%: enough to settle a 38%
question and not a 3% one. Two independent thirties of that kernel taken an hour apart landed
at 1.374× and 1.383×, each inside the other's interval — the protocol reproduces, not just
the row.

**Nineteen sweeps carry the x86-64 tables**, one per benchmark, in the
`*-x86_64-launches.log` files beside the single launches they replaced. The device was re-taken
separately and the next section is its half.

**The ratios came back; the individual times largely did not.** Ten published x86 ratios sit
inside the new interval unchanged — `bitwiseAnd` 10.01× against 9.97×, `countNonZero` 1.62×
against 1.62×, `countAnd` 3.47× against 3.49×, optical flow 7.13× against 7.19× and its
`1/1/1/1` ladder 28.53× against 29.2×, Hamming matching 4.72× against 4.70×, `pyrDown` 1.56×
against 1.556×, the crossover's shipped shape 1.49× against 1.474×, `erode` 3×3 1.04× against
1.053× and on a 5×5 ellipse 0.32× against 0.319×. The cells behind them moved much more.

**Launch noise here is one-sided, and that is the finding underneath all of this.** Of the 85
x86-64 time cells this round re-took whose kernel has not changed, **61 read slower in their
single launch than in the median of thirty**, and the asymmetry is in the size as well as the
count: the worst overstatement is **49%** and the worst understatement **3.3%**. A launch can
go badly wrong and cannot go much right. The median cell reads 1.6% slow.

So the single-launch protocol biased the published *times* slow while the *ratios* largely
survived, because a launch that lands slow lands slow on both arms at once.
`wordtype_narrow` is the clean demonstration: all three of its arms read about 20% high in
one launch and the ratios between them did not move. It also means the sweep cannot rescue a
figure that was quoted as a time — those had to be replaced, which is what these tables now
carry.

**Four rows moved enough to change what a reader would conclude**, and they are different in
kind:

- **`bitwiseNot` is 17.99×, not 25.04× — the largest loss in this round.** The move is
  entirely in the denominator: `cv::bitwise_not` read 0.08591 ns/pixel in the single launch,
  where thirty launches span 0.06156 to 0.07519. The old figure flattered binCV by timing a
  slow OpenCV.
- **`morphologyEx(OPEN)` is 1.022×, not 1.15×**, and four of its thirty launches fall below
  1.00×. It is a near-parity row published as a clear win.
- **`FAST, bit-plane` is 1.472×, not 1.50×, and that one is real.** The runtime switch that
  makes the vector arm provably off-switchable is read once per image row, and reverting only
  that read measures 12.8% faster with the intervals disjoint. Hoisting it out of the row loop
  keeps the switch and recovers all of it —
  [issue #73](https://github.com/ryanhou28/bincv/issues/73), which would take the row to about
  1.66×.
- **`dilate` 3×3 was published as a 0.80× loss and is not one.** Thirty launches put it at
  1.057×, ahead in 30 of 30, on a kernel whose source has not changed. That launch timed
  binCV's arm at 0.13037 ns/pixel where thirty span 0.09260 to 0.09680 — one slow draw, on the
  row where it changed the answer's sign.

**Two rows moved because the code did.** The assembled pipeline is 3.658× rather than 3.30×
because the 2026-09-06 change to one pyramid build per frame landed after that table was
taken — `pyrDown` falls from 0.137 to 0.057 ms/frame, the halving that change predicted. And
`BRIEF` gains 4.69× → 5.18×; a refactor that looked like the cause was A/B'd at thirty
launches an arm and is not one (122,596 ns against 123,834, intervals overlapping), so that
row is a slow draw rather than a code change.

**Two x86 rows got a committed source for the first time**, and four device figures did. The
published x86 denoise and derivative figures appeared in no log in this repository — the
committed single launches read different numbers — so those cells were unsourced rather than
merely uncertain, and are now thirty launches each. On the device, `cornerSubPix`, the two
stereo arms and the whole aarch64 stereo column had no committed log at all; they have one
now, which is how the census row's arm mismatch surfaced.

**What this host can resolve at thirty launches is a property of the row**, between 1.002×
and 1.090×. `FAST, bit-plane` resolves 1.002×; Lucas–Kanade's `1/1/1/1` ladder resolves
1.090×, so that row cannot tell 29× from 31× and its figure is not quoted to three digits.
`countAndSplit` at a 31×31 window on the largest geometry resolves only 1.32×, and carries
no published claim. Each
sweep's log prints its own number in the `minres` column.

**Seven x86 logs were not re-taken, and are named here rather than left to look current.**
`essential-x86_64.log` and `ransac-x86_64.log` back no published timing — checked
structurally, by which report links to which log, not by matching numbers. `pyramid` and
`wordwidth` publish computed byte counts, and `essential_stack`,
`feature-tracking-rss` and `feature-tracking-threads` publish stack bytes, resident set and
thread counts. [logs/README.md](logs/README.md) lists them with their reasons. The three
threading rows in [feature-tracking.md](feature-tracking.md#what-this-does-not-claim) are the
one place an un-swept x86 *timing* is still published, because a threading arm cannot be
pinned; that table says so.

### On the aarch64 device

**The device column has been re-taken too, and the headline is that it held.** Nineteen
benchmarks, ten launches each — seven on the two stereo binaries and five on the
1709-frame pipeline, which costs a minute a launch — governor locked to `performance` and
restored afterwards, every one at commit `80ff0a8`. Of the published device
figures, **two moved beyond their own band for a reason that is not the measurement**:

| figure | published | re-taken | what moved |
|---|---|---|---|
| the assembled pipeline | 4.73× | **4.620×** | binCV's arm, +3% — a regression, [feature-tracking.md](feature-tracking.md#speed) |
| `denseDisparity` census, aarch64 | 462 ms | **730.6 ms** | the row's two columns were timing different word-type arms, [stereo.md](stereo.md) |

Everything else reproduced. The ratios that shifted shifted because the `cv::` denominator
did — `countAnd` 6.55× to 6.242× on a binCV arm that moved 0.19%, `morphologyEx(OPEN)` 1.11×
to 1.146× on one that moved 0.07% — and `goodFeaturesToTrack`'s corrected figure came back
from an independent sweep within its interval, which is the control this round had. (That
round's `goodFeaturesToTrack` numbers were against the binarized denominator; the row now
leads with stock, and [features.md](features.md#corner-detection) says why.)

**The device is reliable, but not uniformly, and the old protocol note overstated it.**
`goodFeaturesToTrack` scatters 1.2% across ten launches and is what "0.1–0.8% run-to-run" was
read from. The small-frame logic and reduction rows scatter far more: **binCV's 640×480
`bitwiseAnd` arm scatters 16.9% across ten launches** where OpenCV's scatters 4.0%, and the
mechanism is binCV's own advantage — a 38 KB packed plane's cache residency is decided per
launch by where the allocator put it, while OpenCV's 307 KB arm never fits and so never
varies. The scatter falls to 2.0–2.3% at 8192×4096, where neither side fits. That is why
`bitwiseAnd` reads 26.68× rather than 28.59× and why neither number is worth defending: the
published figure sits inside the ten launches' own range of 25.58× to 30.32×.

**What this round could not do is date the old figures.** A figure that reproduces was either
never stale or was stale in a kernel that has since come back to where it was, and nothing in
the repository distinguishes those. What the pipeline row shows is that the second case is
not hypothetical: every stage of it but one got faster, `pyrDown` by half, and its total got
slower — so a stable headline can sit on top of a column that has moved underneath it.

## The workload

Sequence-level results use **EuRoC MAV `V1_02_medium`, camera `cam0`** — 1710 frames of
752×480 8-bit grayscale, 1709 consecutive pairs, used whole. Which sequence is not a detail:
`V1_02` gives the tracker materially more work per frame than the easier `MH_01_easy`, and a
whole-pipeline ratio measured on the two comes out differently enough to change the
conclusion.

Operation-level results need no dataset. They run on synthetic content across a ladder of
sizes — the filter benchmarks from 640×480 down to 94×60, the bandwidth-bound ones up to
8192×4096 — so that a ratio which collapses once both sides fit in cache can be told from one
that holds.

## Reproducing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/benchmark/logic_benchmark                     # one operation, against OpenCV
./build/benchmark/feature_tracking_sequence <euroc-cam0-dir>  # the assembled pipeline
```

Any of those can be taken as a launch sweep instead of a single run, which is what an x86
figure needs:

```bash
./scripts/run_launches.sh -n 30 ./build/benchmark/corner_opencv_benchmark   # -g on the Pi
./scripts/aggregate_launches.py corner_opencv_benchmark-x86_64-launches.log \
    --column ns/pixel --ratio 't1:OpenCV binarized/t1:binCV streaming' --ladder
```

On x86-64 that pair **is** how every figure in these reports was taken, not an option beside
it.

`--ladder` answers how many launches a row needs by resampling the ones already taken;
`--resolve 5%` asks whether a difference of a size **you** state is resolvable here, because
how much is worth having is a per-case judgement and not something either script decides.

Each report's **Reproduce** section names the exact binary for its tables. The sequence
benchmarks need a directory of `.png` frames; everything else is self-contained.

Two things will move your numbers more than anything in the code. **Link the `bincv_core`
CMake target** rather than only adding the include path — the ISA flags ride on the target,
and a consumer who added the include path alone measured binCV 2.25× slower on this device
with nothing to indicate anything was wrong. And **check what OpenCV you are measuring
against**: the two builds used here differ in version and in dispatched instruction sets, and
the benchmarks print both.
