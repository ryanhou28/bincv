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
| [methodology-memory.md](methodology-memory.md) | how memory is measured — read before quoting a memory number |
| [methodology-timing.md](methodology-timing.md) | how a difference between two timings is judged real — read before quoting a speed ratio |

Raw output for every table is in [logs/](logs/): one file per benchmark per machine, each a
sweep of separate process launches with the aggregate appended, and each recording, in its
header, a content hash of every first-party file it measured — so the staleness gate can
check it on any clone without the commit it also names ([logs/README.md](logs/README.md)).
Logs listed in [logs/expected-stale.txt](logs/expected-stale.txt) were taken before a
change to the code beneath them; that file records which files moved and why each figure
still stands, and a figure is re-taken rather than argued when it does not.

## At a glance

**One row per operation, and the two host machines are columns.** x86-64 and aarch64 are
different measurements against different OpenCV builds on different hardware; they are never
averaged, and neither stands in for the other. The assembled feature tracking pipeline
is not in these tables — it is not an operation — and is
[further down](#assembled-pipelines).

### Speed, CPU

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means
OpenCV is.** Where a table divides something else, its header says so.

640×480 and `uint32_t` words unless the row names otherwise, one thread on both sides. Each
row names its own unit, and on all of them the smaller number is the faster side.

**Each ratio carries the bootstrap 95% interval of its launches** — thirty on x86-64, ten on
the device, seven for dense disparity — and each time cell is those launches' median. The
ratio is formed inside each launch where both arms share a binary, so it is not the quotient
of the two cells beside it; a row marked *unpaired* has its arms in separate binaries and
its interval from resampling the two sweeps independently.
[methodology-timing.md](methodology-timing.md#the-protocol-each-host-needs) says why the two
launch counts differ. Times are quoted to four significant figures and ratios to three.

<!-- figure-check values="OpenCV, x86-64|binCV, x86-64|x86-64 ratio|OpenCV, aarch64|binCV, aarch64|aarch64 ratio" source="source" -->
| operation | measured against | OpenCV, x86-64 | binCV, x86-64 | x86-64 ratio | OpenCV, aarch64 | binCV, aarch64 | aarch64 ratio | source |
|---|---|---|---|---|---|---|---|---|
| `bitwiseAnd`, ns/pixel | `cv::bitwise_and` | 0.02823 | 0.002810 | 9.97× [9.82, 10.3] | 0.6266 | 0.02369 | 26.7× [26.1, 27.4] | [primitives.md](primitives.md) |
| `countNonZero`, ns/pixel | `cv::countNonZero` | 0.01501 | 0.009270 | 1.62× [1.61, 1.63] | 0.1692 | 0.06365 | 2.66× [2.62, 2.67] | [primitives.md](primitives.md) |
| denoise, 3-pixel median, ns/pixel | composed `cv::min` / `cv::max` | 0.1926 | 0.009930 | 19.2× [19.0, 19.5] | 3.438 | 0.05941 | 57.7× [56.9, 58.0] | [primitives.md](primitives.md) |
| spatial derivative, both axes, ns/pixel | `cv::filter2D` ×2 | 0.5156 | 0.04645 | 11.1× [11.1, 11.3] | 5.043 | 0.2075 | 24.3× [24.1, 24.5] | [primitives.md](primitives.md) |
| `erode` 3×3 rect, ns/pixel | `cv::erode` | 0.1013 | 0.09595 | 1.05× [1.04, 1.07] | 0.7360 | 0.7219 | 1.02× [0.991, 1.04] | [primitives.md](primitives.md) |
| `erode` 5×5 ellipse, ns/pixel | `cv::erode` | 0.2238 | 0.6985 | 0.319× [0.318, 0.323] | 1.852 | 3.596 | 0.514× [0.510, 0.522] | [primitives.md](primitives.md) |
| `pyrDown`, 1 bit in, µs/call | `cv::pyrDown` on `CV_8U` | 47.70 | 30.70 | 1.56× [1.54, 1.60] | 516.5 | 93.8 | 5.51× [5.48, 5.55] | [primitives.md](primitives.md) |
| `pyrDown`, 8 bits in, µs/call | `cv::pyrDown` on `CV_8U` | 47.70 | 2040 | 0.0235× [0.0233, 0.0242] | 516.5 | 7360 | 0.0701× [0.0698, 0.0706] | [limits.md](limits.md) |
| optical flow, 140 points, ms/call | `cv::calcOpticalFlowPyrLK` | 3.978 | 0.5585 | 7.19× [6.89, 7.40] | 23.47 | 2.837 | 8.27× [8.20, 8.36] | [features.md](features.md) |
| BRIEF descriptors, 1000 kpts, ms | `cv::ORB::compute` | 0.6388 | 0.1231 | 5.18× [5.15, 5.22] | 7.167 | 0.6579 | 10.8× [10.6, 11.2] | [features.md](features.md) |
| Hamming matching, kNN=2 over 1000×1000, ms | `cv::BFMatcher` | 9.071 | 1.916 | 4.70× [4.65, 4.79] | 38.19 | 19.52 | 1.95× [1.94, 1.97] | [features.md](features.md) |
| `goodFeaturesToTrack`, ns/pixel | `cv::goodFeaturesToTrack` | 8.807 | 6.368 | 1.38× [1.35, 1.43] | 58.34 | 24.10 | 2.42× [2.41, 2.42] | [features.md](features.md) |
| FAST, 8-bit, synthetic 752×480, ms/call | `cv::FAST` | 0.3591 | 0.3446 | 1.04× [1.03, 1.05] | 2.910 | 3.025 | 0.962× [0.961, 0.963] | [features.md](features.md) |
| FAST, bit-plane, µs/call | `cv::FAST` | 267.7 | 161.7 | 1.65× [1.64, 1.66] | 2051 | 865.2 | 2.37× [2.37, 2.37] | [features.md](features.md) |
| dense disparity, ms/frame | `cv::StereoBM` | 12.68 | 10.41 | 1.22× [1.20, 1.24], unpaired | 79.90 | 60.57 | 1.32× [1.32, 1.32], unpaired | [stereo.md](stereo.md) |

Three rows carry a qualification a table cell cannot:

- **`pyrDown` at 8 bits in is the boundary of the whole idea, not a regression.** Both sides
  store a byte there, so there is nothing for bit-slicing to skip.
- **`erode` on a 5×5 ellipse is a deliberate trade**, not an unfinished kernel: a
  non-separable element costs one shifted-OR per set element, and the fused kernel was kept
  because it holds 8× less. [footprint.md](footprint.md) prices it.
- **Dense disparity is the row that cannot be paired.** Its two arms are separate binaries,
  so the interval comes from resampling the two sweeps independently — the weakest interval
  in the table on x86-64, and a very tight one on the device, whose two arms scatter 0.4%
  and 0.6% across launches.

`cornerSubPix` is measured on the device only: 13.8× [13.7, 13.8] against `cv::cornerSubPix`
([features.md](features.md)); no x86-64 sweep of it exists, so it has no row here.

### Speed, GPU

RTX 3070 Ti · 752×480 unless the row names a geometry · both arms on **one explicit stream**
· kernel-resident clock · medians of 7 independent process runs. Milliseconds, so the
smaller number is the faster side. The table quotes the committed sweep;
[cuda.md](cuda.md) gives an earlier sweep's readings beside each row, all inside the
run-to-run scatter.

<!-- figure-check values="cv::cuda, ms|binCV, ms|ratio" source="source" -->
| operation | `cv::cuda` arm | cv::cuda, ms | binCV, ms | ratio | source |
|---|---|---|---|---|---|
| dense disparity, binary entry | `cv::cuda::StereoBM(64, 9)` | 0.7134 | 0.06400 | 11.2× | [cuda.md](cuda.md) |
| dense disparity, census entry | ″ | 0.7101 | 0.4789 | 1.50× | [cuda.md](cuda.md) |
| FAST | `cv::cuda::FastFeatureDetector` | 0.1221 | 0.02078 | 6.02× | [cuda.md](cuda.md) |
| descriptor matching, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 2.009 | 0.2105 | 9.51× | [cuda.md](cuda.md) |
| describe, N=1000 | `cv::cuda::ORB::computeAsync` | 0.08960 | 0.009353 | 8.82× | [cuda.md](cuda.md) |
| `goodFeaturesToTrack`, wall clock | `createGoodFeaturesToTrackDetector` | 3.403 | 0.4240 | 8.76× | [cuda.md](cuda.md) |
| optical flow, 204 points | `SparsePyrLKOpticalFlow` | 0.1748 | 0.08931 | 1.41×–5.86× per round, median 2.01× | [cuda.md](cuda.md) |
| optical flow, 2048 points | ″ | 0.3769 | 0.3976 | 0.952×, null | [cuda.md](cuda.md) |
| min-eigenvalue response | `createMinEigenValCorner` | 0.05192 | 0.02007 | 2.60× | [cuda.md](cuda.md) |

**The one row where binCV's cell is larger is published as a null result, not a loss.**
binCV wins 14 of its 105 paired rounds there and the difference is inside this host's noise
bar, so [cuda.md](cuda.md) records that no direction is established rather than claiming
OpenCV won. Two more rows are qualified there: optical flow at 204 points is faster in every
one of 105 paired rounds but its size moves between 1.41× and 5.86× across them, which is
why its cell is a range around the median; and `goodFeaturesToTrack`'s denominator swings
3.363–4.186 ms across runs because `cv::cuda`'s spacing filter runs on the CPU. Every row's
spread across the 7 runs is a column of cuda.md's own table.

### Memory, CPU

Peak working set of one call, computed from buffer geometry, so it is exact and **identical
on both architectures** — one column pair, not two. Read
[methodology-memory.md](methodology-memory.md) before quoting any of it.

<!-- figure-check values="OpenCV|binCV|ratio" source="source" -->
| operation | measured against | OpenCV | binCV | ratio | source |
|---|---|---|---|---|---|
| `bitwiseAnd` / `Or` / `Xor` / `Not`, bytes | `cv::bitwise_*` | 921,600 | 115,200 | 8.00× | [primitives.md](primitives.md) |
| `countNonZero`, per input plane, bytes | `cv::countNonZero` | 307,200 | 38,400 | 8.00× | [primitives.md](primitives.md) |
| denoise, 3-pixel median, bytes | composed `cv::min` / `cv::max` | 2,150,400 | 76,800 | 28.0× | [footprint.md](footprint.md) |
| spatial derivative, both axes, bytes | `cv::filter2D` ×2 | 1,536,000 | 192,000 | 8.00× | [footprint.md](footprint.md) |
| `erode` / `dilate` 3×3, bytes | `cv::erode` / `cv::dilate` | 614,400 | 76,800 | 8.00× | [footprint.md](footprint.md) |
| `morphologyEx(MORPH_OPEN)`, bytes | `cv::morphologyEx` | 614,400 | 115,200 | 5.33× | [footprint.md](footprint.md) |
| `goodFeaturesToTrack`, bytes | `cv::goodFeaturesToTrack`, binarized | 9,014,976 | 1,580,064 | 5.71× | [footprint.md](footprint.md) |
| FAST input plane, bytes | `cv::FAST` on `CV_8U` | 360,960 | 46,080 | 7.83× | [footprint.md](footprint.md) |
| dense disparity, output + scratch, bytes | `cv::StereoBM` | ≥ 721,920 | 393,312 | ≥ 1.84× | [stereo.md](stereo.md) |

`morphologyEx` is 5.33× rather than 8× because binCV's fused kernel needs a caller-provided
scratch frame where `erode` and `dilate` need none; `cv::morphologyEx` measured at the
allocator holds a further 5,784 B of row buffers, which would move that row to 5.38×, and
the table keeps the geometry-only figure. The dense-disparity row compares outputs plus
scratch on both sides: `cv::StereoBM` writes a 2-byte disparity map and its internal buffers
were not measured, so binCV's ≥ 1.84× is a lower bound. What the streaming design refuses
to allocate is the 23 MB cost volume; its whole scratch is 32,352 B ([stereo.md](stereo.md)).

### Memory, GPU

`cudaMemGetInfo` delta taken identically on both sides — the only meter readable across
libraries — never mixed with binCV's own allocation sums.

<!-- figure-check values="cv::cuda, KiB|binCV, KiB|ratio" source="source" -->
| operation | `cv::cuda` arm | cv::cuda, KiB | binCV, KiB | ratio | source |
|---|---|---|---|---|---|
| describe, N=1000 | `cv::cuda::ORB::computeAsync` | 2048.0 | 44.0 | 46.5× | [cuda.md](cuda.md) |
| descriptor matching, 5000² | `BFMatcher::knnMatchAsync(k=2)` | 8192.0 | 368.0 | 22.3× | [cuda.md](cuda.md) |
| dense disparity, binary entry, per frame | `cv::cuda::StereoBM(64, 9)` | 3072.0 | 448.0 | 6.86× | [cuda.md](cuda.md) |
| `goodFeaturesToTrack` | `createGoodFeaturesToTrackDetector` | 10240.0 | 1920.0 | 5.33× | [cuda.md](cuda.md) |
| corner response | `createMinEigenValCorner` | 10240.0 | 2048.0 | 5.00× | [cuda.md](cuda.md) |
| LK tracker resident state | `SparsePyrLKOpticalFlow` | 1408.0 | 448.0 | 3.14× | [cuda.md](cuda.md) |
| `threshold` | `cv::cuda::threshold` | 1024.0 | 416.0 | 2.46× | [cuda.md](cuda.md) |
| FAST at capacity 32,768 | `cv::cuda::FastFeatureDetector` | 768.0 | 432.0 | 1.78× | [cuda.md](cuda.md) |
| dense disparity, census entry, per frame | `cv::cuda::StereoBM(64, 9)` | 3072.0 | 4512.0 | 0.681× (`cv::cuda` smaller) | [cuda.md](cuda.md) |

**The census entry is the one memory row binCV loses, and it is not a defect to be fixed.**
The census transform expands 8 bits per pixel into a 32-bit descriptor word, so the two
transformed images are 2,820 KiB before a disparity map exists, where StereoBM works on the
8-bit frames directly. It is faster on the same run; a caller arriving with 8-bit frames
weighs the two. `cv::cuda`'s figures are upper readings — `GpuMat` may pool and pads its
pitch — so binCV's lead on the other rows is a lower bound.

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
| time, ms/frame | 4.007 | 1.010 | 3.97× [3.94, 4.00] | 23.65 | 4.434 | 5.34× [5.32, 5.35] | [feature-tracking.md](feature-tracking.md) |

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
timed. Cortex-M has been built and partly measured on an STM32H753ZI —
[targets/stm32h753/README.md](../../targets/stm32h753/README.md) says what that covers — but
it produced no OpenCV comparison, so it is not here either.

## The machines

Neither CPU stands in for the other: results move a long way between them, and mostly
because of what OpenCV's two builds do rather than what binCV's code does. Hamming matching
is 4.70× on x86 and 1.95× on the device; bit-plane FAST goes the other way, 1.65× against
2.37×; and the bit width at which a bit-sliced pyramid stops beating `cv::pyrDown` differs by
several bits between them.

| | **x86-64** — development host | **aarch64** — reference device |
|---|---|---|
| CPU | AMD Ryzen 5 5600X, 6 cores / 12 threads | Broadcom BCM2711, Cortex-A72, 4 cores, 1.8 GHz |
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
**Where the OpenCV side allocates internally, buffer arithmetic cannot see it**, so those
figures are measured at the allocator where a report says so and stated as lower bounds
where it does not. A per-buffer ratio is not the metric — a bit-plane is eight times smaller
than a byte plane by construction and saying so measures nothing.
[methodology-memory.md](methodology-memory.md) has the instruments and the ways a memory
figure goes wrong, including one that made OpenCV look 17× smaller than it is.

**A synthetic scene must be noisy enough to separate a good estimator from a lucky one.** The
RANSAC benchmarks generate their inliers with noise on them, because a scene whose inliers
fit the transform exactly makes a missing least-squares refit invisible — it was, once, at a
13× cost in accuracy on data with noise in it.

### On the reference device

A Pi 4 will produce stable-looking numbers that are wrong, so four conditions are enforced by
the runner rather than remembered: the architecture is asserted to be `aarch64`; the governor
is pinned to `performance` for the run and restored after, because on `ondemand` a short
benchmark measures the ramp between 600 MHz and 1.8 GHz; the process is pinned to one core
with `taskset`; and throttle state is read before and after, a change during a run
invalidating it. The environment block each run prints is at the top of every aarch64 log
in [logs/](logs/).

**The throttle flag alone is not enough on a Pi that has ever throttled.** `get_throttled`
returns sticky history, so a board reading `0x80000` — soft temperature limit *has occurred*,
at some point since boot — reads the same before and after a run that throttled and one that
did not. The device sweep behind the current figures sampled the core's actual clock every
two seconds instead: 1,183 samples across the whole run, every one at 1,800,000 kHz, peak
65.2 °C against an 80 °C limit. That is the evidence the numbers were not taken on a ramp.

**The device column is ten launches per benchmark** (seven on the two stereo binaries),
governor locked, taken at `80ff0a8` (on `main` as `086428c`); `goodFeaturesToTrack` was
re-taken at `880704b` (on `main` as `8729e05`) after the selection-stage optimization that
moved it, and the three tracker sweeps (optical flow, the assembled pipeline, the
memory-bound study) at `4c4b3bf` after the NEON covariance arm gained its off-switch, where
every cell landed inside its previous interval. Small-plane rows scatter more than large ones
on this device — binCV's 115 KB `bitwiseAnd` arm scatters 16.9% across ten launches where
OpenCV's 921 KB arm scatters 4.0%, because a plane that fits the 1 MiB L2 has its residency
decided per launch by where the allocator put it; at 8192×4096 both fall to about 2%.
[methodology-timing.md](methodology-timing.md) has the mechanism.

### On the x86-64 host

**Every x86-64 figure in these reports is the median of thirty pinned launches with a
bootstrap 95% interval.** The desktop under WSL2 is not timing-grade, and the launch sweep
is how much it is not: `goodFeaturesToTrack` prints a 21–39% within-run spread, and the arms
it times scatter 12–30% across thirty launches of the same binary, so one launch has
returned anything from 0.86× to 1.61× on a row whose thirty-launch interval is
[1.35, 1.43] around 1.38× — ±2.8%, enough to settle a 38% question and not a 3% one. Two
independent thirties of that kernel taken an hour apart landed at 1.374× and 1.383×, each
inside the other's interval. Launch noise on this host is one-sided: a launch can land slow
and cannot land much fast, so a single launch overstates times while ratios largely survive;
[methodology-timing.md](methodology-timing.md) has the finding.

**What this host can resolve at thirty launches is a property of the row**, between 1.002×
and 1.09×. `FAST, bit-plane` resolves 1.002×; Lucas–Kanade's `1/1/1/1` ladder resolves
1.09×, so that row cannot tell 29× from 31× and its figure is not quoted to three digits.
`countAndSplit` at a 31×31 window on the largest geometry resolves only 1.32×, and carries
no published claim. Each sweep's log prints its own number in the `minres` column.

**Seven x86 logs have no sweep, and are named rather than left to look current.**
`essential-x86_64.log` and `ransac-x86_64.log` back no published timing. `pyramid` and
`wordwidth` publish computed byte counts, and `essential_stack`, `feature-tracking-rss` and
`feature-tracking-threads` publish stack bytes, resident set and thread counts.
[logs/README.md](logs/README.md) lists them with their reasons. The three threading rows in
[feature-tracking.md](feature-tracking.md#what-this-does-not-claim) are the one place an
un-swept x86 *timing* is still published, because a threading arm cannot be pinned; that
table says so.

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

## Changes to published figures

Figures that moved after publication, with the cause. Each report carries its own table of
the same shape; this one holds the rows an index reader would have seen.

| date | row | previous | current | why |
|---|---|---|---|---|
| 2026-09-21 | every x86-64 time and ratio | one process launch each | median of thirty pinned launches with a bootstrap interval | the single-launch protocol was one draw from an uncharacterised distribution; ten ratios came back inside the new interval, the times moved more |
| 2026-09-21 | `bitwiseNot`, x86-64 | 25.04× | 17.99× | the single launch had timed a slow `cv::bitwise_not` (0.08591 ns/px against a thirty-launch span of 0.06156–0.07519) |
| 2026-09-21 | `morphologyEx(OPEN)`, x86-64 | 1.15× | 1.02× | a near-parity row had been published as a clear win; four of thirty launches fall below 1.00× |
| 2026-09-21 | `dilate` 3×3, x86-64 | 0.80× (a loss) | 1.06× | one slow draw of binCV's arm (0.13037 ns/px against a span of 0.09260–0.09680) had changed the sign |
| 2026-09-21 | `bitwiseAnd`, aarch64 | 28.59× | 26.7× | inside the ten launches' own range of 25.58×–30.32×; small-plane cache residency varies per launch |
| 2026-09-21 | `denseDisparity` census, aarch64 | 462 ms | 730.6 ms | the row's two columns had timed different word-type arms; both are now the `uint32` arm |
| 2026-09-23 | `FAST`, bit-plane, x86-64 | 1.50× then 1.47× | 1.65× | the vector arm's runtime switch was read once per row, keeping the scalar body live; read once per call |
| 2026-09-06 / 09-22 | assembled pipeline, x86-64 | 3.30× | 3.66× then 3.97× | one pyramid build per frame; then the `goodFeaturesToTrack` selection-stage optimization |
| 2026-09-21 / 09-22 | assembled pipeline, aarch64 | 4.73× | 4.62× then 5.32× | the device sweep read the LK stage 3% slower and an independent spot-check agreed; the selection-stage optimization then moved the row |
| 2026-09-22 | `goodFeaturesToTrack`, both | against a binarized `cv::goodFeaturesToTrack` | against stock, 1.38× / 2.42× | the selection-stage optimization made the stock denominator the right one to lead with ([features.md](features.md#corner-detection)) |
