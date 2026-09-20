# binCV measurement reports

What binCV costs and what it saves, measured against OpenCV on two CPU architectures and
one GPU.

Each report states a claim, names the OpenCV call it is measured against, gives speed and
memory together, and names the command that reproduces it.

**Both sides' measurements are published here, not only the ratio between them.** A cell
reading `0.543` beside one reading `3.871`, in a column headed milliseconds, needs no
convention and no legend: the smaller number is the faster one, and a row where binCV's
cell is the larger one is a row where binCV lost. Ratios appear beside the measurements
where the comparison is the whole point of the row — never instead of them, and never as
the only thing a row carries unless the measurements did not survive, which is a table of
its own below.

<!-- figure-check values="what it turns on" source="report" -->
| report | what it covers | what it turns on |
|---|---|---|
| [frontend.md](frontend.md) | a whole tracking frontend, end to end, over a real sequence | 1.134–1.283 ms/frame against the OpenCV frontend's 3.841–4.485 on x86-64; 436,704 bytes live against 2,719,832 |
| [primitives.md](primitives.md) | logic, reductions, denoise, morphology, derivative, pyramid downsample | on x86-64, denoise at 0.01059 ns/pixel against 0.18609, and `erode` on a 5×5 ellipse behind at 0.32× |
| [features.md](features.md) | corner detection, FAST, descriptors, matching, optical flow | Lucas–Kanade at 0.543 ms against `cv::calcOpticalFlowPyrLK`'s 3.871 on x86-64 |
| [stereo.md](stereo.md) | dense disparity against `cv::StereoBM` | 60.4 ms/frame against 79.8 on aarch64, in 32.4 KB of scratch against ≥ 722 KB |
| [cuda.md](cuda.md) | the CUDA backend, GPU against GPU | dense disparity at 0.0648 ms/frame against `cv::cuda::StereoBM`'s 0.7152, in 448.0 KB/frame against 3072.0 |
| [cuda-evidence.md](cuda-evidence.md) | the trail behind every CUDA figure — the layouts tried, the arms dropped, the claims withdrawn, the profiler pass | why the row above is worth quoting rather than believing |
| [footprint.md](footprint.md) | the memory result on its own, and the speed declined to protect it | 436,704 bytes against 2,719,832 over the frontend |
| [limits.md](limits.md) | where binCV ties, loses, or stops paying at all | four ways it stops working |
| [methodology-memory.md](methodology-memory.md) | how memory is measured, and the errors that shaped it | read before quoting a memory number |
| [methodology-timing.md](methodology-timing.md) | how a difference between two timings is judged real | read before quoting a speed ratio |

The raw output the host tables were cut from is in [logs/](logs/). The stereo and CUDA
tables have no committed logs; those two reports name the binary that produces each number
instead.

Inside the reports, paired figures are x86-64 then aarch64. The two are different
measurements against different OpenCV builds and are never averaged. The tables below give
each machine a row of its own rather than a pair in one cell.

## At a glance

One table for speed and one for memory, because they are two questions and a column that
holds both cannot be read. Every cell is a measured value with its unit named in the row;
the machine is named in the row too, because an x86-64 figure and an aarch64 figure are
different measurements against different OpenCV builds. The source column names the report
the pair was cut from, which is where the conditions, the spread and the verdict live.

Losses are ordinary rows here. Nothing is marked, because nothing needs to be: a binCV cell
larger than the OpenCV cell beside it is the whole of what "binCV is slower" looks like.

### Speed

Time per call or per frame, so the smaller number is the faster side.

<!-- figure-check values="OpenCV|binCV" source="source" -->
| what | measured against | machine | OpenCV | binCV | source |
|---|---|---|---|---|---|
| whole tracking frontend, ms/frame | the OpenCV frontend | x86-64 | 3.841–4.485 | 1.134–1.283 | [frontend.md](frontend.md) |
| whole tracking frontend, ms/frame | the OpenCV frontend | aarch64 | 23.249–23.451 | 4.906–4.949 | [frontend.md](frontend.md) |
| optical flow, 140 points, ms/call | `cv::calcOpticalFlowPyrLK` | x86-64 | 3.871 | 0.543 | [features.md](features.md) |
| optical flow, 140 points, ms/call | `cv::calcOpticalFlowPyrLK` | aarch64 | 23.476 | 2.843 | [features.md](features.md) |
| dense disparity, ms/frame | `cv::StereoBM` | x86-64 | ~14.7 | ~12.0 | [stereo.md](stereo.md) |
| dense disparity, ms/frame | `cv::StereoBM` | aarch64 | 79.8 | 60.4 | [stereo.md](stereo.md) |
| denoise, 3-pixel median, ns/pixel | composed `cv::min` / `cv::max` | x86-64 | 0.18609 | 0.01059 | [primitives.md](primitives.md) |
| spatial derivative, both axes, ns/pixel | `cv::filter2D` ×2 | x86-64 | 0.54843 | 0.04793 | [primitives.md](primitives.md) |
| `bitwiseAnd`, 640×480, ns/pixel | `cv::bitwise_and` | x86-64 | 0.02734 | 0.00273 | [primitives.md](primitives.md) |
| `bitwiseAnd`, 640×480, ns/pixel | `cv::bitwise_and` | aarch64 | 0.64783 | 0.02266 | [primitives.md](primitives.md) |
| Hamming matching, kNN=2 over 1000×1000, ms | `cv::BFMatcher` | x86-64 | 9.184 | 1.947 | [features.md](features.md) |
| Hamming matching, kNN=2 over 1000×1000, ms | `cv::BFMatcher` | aarch64 | 38.269 | 19.391 | [features.md](features.md) |
| `pyrDown`, 1 bit in, µs/call | `cv::pyrDown` on `CV_8U` | x86-64 | 48.3 | 31.0 | [primitives.md](primitives.md) |
| `pyrDown`, 1 bit in, µs/call | `cv::pyrDown` on `CV_8U` | aarch64 | 521.4 | 93.8 | [primitives.md](primitives.md) |
| `pyrDown`, 8 bits in, µs/call | `cv::pyrDown` on `CV_8U` | x86-64 | 48.3 | 2034.4 | [limits.md](limits.md) |
| `pyrDown`, 8 bits in, µs/call | `cv::pyrDown` on `CV_8U` | aarch64 | 521.4 | 7358.6 | [limits.md](limits.md) |
| `erode` 3×3 rect, ns/pixel | `cv::erode` | x86-64 | 0.10013 | 0.09605 | [primitives.md](primitives.md) |
| `erode` 3×3 rect, ns/pixel | `cv::erode` | aarch64 | 0.71993 | 0.72012 | [primitives.md](primitives.md) |
| `erode` 5×5 ellipse, ns/pixel | `cv::erode` | x86-64 | 0.22759 | 0.70415 | [primitives.md](primitives.md) |
| `erode` 5×5 ellipse, ns/pixel | `cv::erode` | aarch64 | 1.81575 | 3.58631 | [primitives.md](primitives.md) |
| `goodFeaturesToTrack`, ns/pixel | `cv::goodFeaturesToTrack`, binarized | x86-64 | 13.63–14.24 | 14.46–15.01 | [features.md](features.md) |
| `goodFeaturesToTrack`, ns/pixel | `cv::goodFeaturesToTrack`, binarized | aarch64 | 75.02–75.82 | 51.25–51.31 | [features.md](features.md) |
| FAST, wide image, ms/call | `cv::FAST` | x86-64 | 0.363 | 0.344 | [features.md](features.md) |
| FAST, wide image, ms/call | `cv::FAST` | aarch64 | 2.906 | 3.024 | [features.md](features.md) |
| dense disparity, binary entry, ms/frame | `cv::cuda::StereoBM(64, 9)` | RTX 3070 Ti | 0.7152 | 0.0648 | [cuda.md](cuda.md) |
| dense disparity, census entry, ms/frame | `cv::cuda::StereoBM(64, 9)` | RTX 3070 Ti | 0.6996 | 0.5076 | [cuda.md](cuda.md) |
| optical flow, 204 points, ms/call | `SparsePyrLKOpticalFlow` | RTX 3070 Ti | 0.1475 | 0.0792 | [cuda.md](cuda.md) |
| optical flow, 2048 points, ms/call | `SparsePyrLKOpticalFlow` | RTX 3070 Ti | 0.3287 | 0.3558 | [cuda.md](cuda.md) |
| FAST, ms/call | `cv::cuda::FastFeatureDetector` | RTX 3070 Ti | 0.1459 | 0.0247 | [cuda.md](cuda.md) |
| `goodFeaturesToTrack`, wall clock, ms/call | `createGoodFeaturesToTrackDetector` | RTX 3070 Ti | 3.7282 | 0.8405 | [cuda.md](cuda.md) |
| min-eigenvalue response, ms/call | `createMinEigenValCorner` | RTX 3070 Ti | 0.0515 | 0.0590 | [cuda.md](cuda.md) |
| descriptor matching, 5000², ms | `BFMatcher::knnMatchAsync(k=2)` | RTX 3070 Ti | 1.9491 | 0.2189 | [cuda.md](cuda.md) |

Eight of those rows carry a qualification their report states and a table cell cannot:

- **The two GPU rows where binCV's cell is larger are published as *null results*, not as
  losses.** The min-eigenvalue response and Lucas–Kanade at 2048 points are both inside
  this host's noise — 0 of 7 runs disjoint on each — so [cuda.md](cuda.md) records that
  neither direction is established rather than claiming OpenCV won.
- **GPU FAST's direction is certain and its size is not.** Across 21 independent processes
  binCV's arm is bimodal and the per-run ratio moves between 3.44× and 6.26×, while every
  one of those runs falls binCV's way 15–0. The pair above is one sweep's medians.
- **`goodFeaturesToTrack`'s GPU denominator is noisy** — `cv::cuda`'s arm swings
  3.24–13.61 ms across runs because its spacing filter runs on the CPU.
- **`pyrDown` at 8 bits in is the boundary of the whole idea, not a regression.** Both
  sides store a byte there, so there is nothing for bit-slicing to skip; it is
  [limits.md](limits.md)'s first answer and the row is here rather than filed away in it.
- **The two GPU dense-disparity rows are quoted from one of two sweeps that
  [cuda.md](cuda.md) publishes side by side.** Its [role bars](cuda.md#the-role-bars) read
  the binary entry at 0.0648 ms against 0.7152 and the census entry at 0.5076 against
  0.6996 — the pairs above; its [headline](cuda.md#the-headline) reads 0.0679 against
  0.7438 and 0.5418 against 0.7584 on an earlier sweep of the same machine. cuda.md states
  that which of the two to quote is not settled, so this table takes both rows from the
  same table rather than picking across them.

### Memory

Peak working set, so the smaller number is the lighter side. The host figures are computed
from buffer geometry and are identical on both architectures; the GPU figures are a
`cudaMemGetInfo` delta taken the same way on both sides. Read
[methodology-memory.md](methodology-memory.md) before quoting any of them.

<!-- figure-check values="OpenCV|binCV" source="source" -->
| what | measured against | machine | OpenCV | binCV | source |
|---|---|---|---|---|---|
| whole tracking frontend, peak, bytes | the OpenCV frontend | x86-64 and aarch64 | 2,719,832 | 436,704 | [footprint.md](footprint.md) |
| denoise, one call at 640×480, bytes | composed `cv::min` / `cv::max` | x86-64 and aarch64 | 2,150,400 | 76,800 | [footprint.md](footprint.md) |
| spatial derivative, one call, bytes | `cv::filter2D` ×2 | x86-64 and aarch64 | 1,536,000 | 192,000 | [footprint.md](footprint.md) |
| `erode` / `dilate` 3×3, one call, bytes | `cv::erode` / `cv::dilate` | x86-64 and aarch64 | 614,400 | 76,800 | [footprint.md](footprint.md) |
| `morphologyEx(MORPH_OPEN)`, one call, bytes | `cv::morphologyEx` | x86-64 and aarch64 | 614,400 | 115,200 | [footprint.md](footprint.md) |
| `goodFeaturesToTrack`, one call, bytes | `cv::goodFeaturesToTrack` | x86-64 and aarch64 | 9,014,976 | 1,580,064 | [footprint.md](footprint.md) |
| FAST input plane, bytes | `cv::FAST` on `CV_8U` | x86-64 and aarch64 | 360,960 | 46,080 | [footprint.md](footprint.md) |
| dense disparity, working set | `cv::StereoBM` | x86-64 and aarch64 | ≥ 722 KB, output alone | 32.4 KB scratch, 1 B/px out | [stereo.md](stereo.md) |
| dense disparity, binary entry, KB/frame | `cv::cuda::StereoBM(64, 9)` | RTX 3070 Ti | 3072.0 | 448.0 | [cuda.md](cuda.md) |
| dense disparity, census entry, KB/frame | `cv::cuda::StereoBM(64, 9)` | RTX 3070 Ti | 3072.0 | 4512.0 | [cuda.md](cuda.md) |
| LK tracker resident state, KB | `SparsePyrLKOpticalFlow` | RTX 3070 Ti | 1408.0 | 448.0 | [cuda.md](cuda.md) |
| FAST at capacity 32,768, KB | `cv::cuda::FastFeatureDetector` | RTX 3070 Ti | 680.0 | 432.0 | [cuda.md](cuda.md) |
| `threshold`, KB | `cv::cuda::threshold` | RTX 3070 Ti | 1024.0 | 416.0 | [cuda.md](cuda.md) |
| descriptor matching, 5000², KB | `BFMatcher::knnMatchAsync(k=2)` | RTX 3070 Ti | 8277.3 | 400.0 | [cuda.md](cuda.md) |

**The census entry on the GPU is the one memory row binCV loses**, and it is not a defect to
be fixed: the census transform expands 8 bits per pixel into a 32-bit descriptor word, so
the two transformed images are 2,820 KB before a disparity map exists, where StereoBM works
on the 8-bit frames directly. It is faster on the same run. [cuda.md](cuda.md) records that
split verdict as unmade rather than rounding it into a headline.

`cv::StereoBM`'s host figure is its output buffer alone, before its internal buffers, which
the tooling in [methodology-memory.md](methodology-memory.md) cannot observe from outside;
`cv::cuda`'s GPU figures are upper readings for the same reason. Both are stated that way in
their reports, and both mean binCV's lead on those rows is a lower bound.

### Published as a ratio only

For these rows the two measurements are not in the reports — only the ratio between them
survived. They are kept out of the tables above rather than dressed to look as checkable as
the rows around them, and each one is owed a re-measurement.

<!-- figure-check values="published ratio" source="source" -->
| what | measured against | machine | published ratio | source |
|---|---|---|---|---|
| denoise, 3-pixel median | composed `cv::min` / `cv::max` | aarch64 | 57.66×, binCV faster | [primitives.md](primitives.md) |
| spatial derivative, both axes | `cv::filter2D` ×2 | aarch64 | 24.28×, binCV faster | [primitives.md](primitives.md) |
| `cornerSubPix` | `cv::cornerSubPix` | x86-64 | ~13×, binCV faster | [features.md](features.md) |
| `cornerSubPix` | `cv::cornerSubPix` | aarch64 | 13.70×, binCV faster | [features.md](features.md) |
| `uint64_t` words on `countNonZero` | binCV at `uint32_t` | aarch64 | 1.95×, the wider word faster | [footprint.md](footprint.md) |

## What these are not

They are not a survey of binCV against every alternative, and they are not tuned
comparisons. Every number is one build of binCV against one build of OpenCV on one
machine, taken on a single commit, with the losses reported alongside the wins.

Nothing here is a claim about a target that has not been measured. binCV also supports
32-bit ARM Cortex-A and RISC-V, and neither appears in these reports because neither has
been built and timed. Cortex-M has since been built and partly measured on an STM32H753ZI
— see [limits.md](limits.md) for what that covers — but it produced no OpenCV comparison,
so nothing about it belongs here either.

## Platforms

Both CPUs are measured. Neither is a stand-in for the other: an x86 figure and an aarch64
figure are different measurements against different OpenCV builds, and the reports never
average them or quote one as the other.

| | **x86-64** — development host | **aarch64** — reference device |
|---|---|---|
| CPU | AMD Ryzen 5 5600X, 6 cores / 12 threads | Broadcom BCM2711, Cortex-A72, 4 cores |
| cache | 32 KiB L1d per core, 512 KiB L2 per core, 32 MiB shared L3 | 32 KiB L1d per core, 1 MiB shared L2 |
| OS | Ubuntu 22.04 under WSL2 | Raspberry Pi OS Lite, 64-bit |
| compiler | g++ 11.4.0, Release `-O3 -DNDEBUG` | g++ 14.2.0, Release |
| OpenCV | 4.8.0-dev, baseline SSE3, dispatching through AVX-512 | 4.10.0, NEON baseline |
| binCV vector paths | `POPCNT`, AVX2 selected at run time | NEON |

**The GPU rows are a third machine**, and it is the CUDA backend's alone: an NVIDIA GeForce
RTX 3070 Ti (SM 8.6, 48 SMs, ~608 GB/s) under WSL2, built with CUDA 11.1 nvcc and g++-9,
measured against the `cv::cuda` counterpart of each operation. [cuda.md](cuda.md) names the
two clocks and the two memory meters it reads, and no GPU figure here mixes them.

The reference device is the one that closes a question. It is a deployment-class part with a
small cache, and results move — sometimes a long way — between the two. Hamming matching is
4.72× on x86 and 1.97× on the device, because x86 has a scalar population count and aarch64's
is a vector instruction whose result must be reduced. The bit-plane FAST goes the other way,
1.50× against 2.37×, because aarch64 has twice the vector registers. And the bit-width at
which a bit-sliced pyramid stops beating `cv::pyrDown` differs by several bits between them.
A desktop measurement does not predict any of that.

The 64-bit OS is a requirement rather than a preference. On 32-bit ARM every `uint64_t`
operation is synthesised from 32-bit pairs, which would measure the compiler rather than
the machine.

## How the numbers are taken

**The denominator is OpenCV on the same content stored as `CV_8U`.** One bit per pixel for
binCV, a byte holding `{0, 1}` for OpenCV, same image, same parameters, same border. That
is what a user runs today without binCV. Where the comparison is against a composed
sequence of OpenCV calls rather than a single stock one, the report says so and charges the
baseline only for the work binCV also does.

**Both sides get one thread.** binCV is serial unless a caller installs a threading
backend, and OpenCV is not; left at its default, a comparison on a multi-core box measures
parallelism and reads as implementation. Every benchmark here pins `cv::setNumThreads(1)`
and prints the count it actually got. Where a threaded binCV figure is given it is stated
against a threaded OpenCV at the same count.

**Both sides are SIMD.** OpenCV's dispatched vector paths are on, and its build configuration
is printed into the logs rather than assumed — the x86 build dispatches through AVX-512, the
device build has NEON as its baseline. binCV's live paths come from `simdStatusString()`,
which the frontend benchmark prints: `AVX2=yes popcount=hardware` on x86, `NEON=yes` on the
device. A comparison of scalar binCV against vectorised OpenCV, or the reverse, is not a
comparison of the implementations.

**Correctness is checked before speed.** Every comparison first asserts that the two sides
computed the same image — bit-exact for Tier 1 operations, and a stated agreement bound for
Tier 2. A benchmark whose arms disagree fails rather than reporting a ratio.

**A synthetic scene must be noisy enough to separate a good estimator from a lucky one.**
Where a benchmark plants a ground-truth model and asks whether binCV recovers it, the
planted data carries noise, because exact data hides whole classes of error. The RANSAC
estimators are the case that taught this: their scenes generated every inlier *exactly*
from the transform, which makes a fit through any three of them exact, which made a missing
least-squares refit a no-op — invisible to every test while costing 13× in accuracy on data
with noise in it. Agreement on the inlier *set* did not catch it, because the inlier set was
right; only the model was wrong.

**Speed is the median of many interleaved batches**, with the minimum, maximum and spread
reported beside it. Arms are run round-robin so drift moves all of them together. Results
are consumed through a `volatile` sink and inputs are varied, because a loop whose result is
unused is deleted by the optimizer and the resulting number looks excellent.

**Memory is the peak working set, and [methodology-memory.md](methodology-memory.md) says
how it is measured.** Read that page before quoting a memory number from these reports. It
names the four different quantities that get called "memory", gives the instruments, and
lists the four measurement errors this project actually published — including one that made
OpenCV look 17× smaller than it is, and one that came from measuring binCV's stack against
OpenCV's heap.

For the image pipeline the figure is computed from buffer geometry — the live buffers of
one call or one frame, counted in bytes. That is arithmetic over container sizes rather
than a sampled RSS, which is why it is exact, reproducible and identical on both
architectures, and it works because no binCV kernel allocates, so scratch appears in the
caller's signature and therefore in the count. **Where the OpenCV side allocates
internally, buffer arithmetic cannot see it** and an allocator-level probe is required;
`cv::morphologyEx` was measured that way rather than assumed, which moved binCV's advantage
there from 8.0× to 5.33×.

One figure is a sampled RSS rather than a computed working set, and says so where it appears:
the effect of thread count on peak memory, since thread stacks are the one thing buffer
arithmetic cannot see.

**Peak working set is the metric, not a per-buffer ratio.** A bit-plane is eight times
smaller than a byte plane by construction and saying so measures nothing. What matters is
the total a stage holds live, which is where an operation that needs three buffers against
OpenCV's two gives some of that back.

### On the reference device

A Pi 4 will produce stable-looking numbers that are wrong, so four conditions are enforced
by the runner rather than remembered:

- **Architecture** is asserted to be `aarch64`; the runner refuses to measure otherwise.
- **The governor** is pinned to `performance` for the run and restored afterwards. Left on
  `ondemand` a short benchmark measures the governor's ramp between 600 MHz and 1.5 GHz.
- **The process is pinned to one core** with `taskset`, on an image with no desktop session.
- **Throttle state is read before and after.** The flags distinguish *currently throttling*
  from *has throttled since boot*; a run is invalidated by a change during it. Two runs in
  this project's history were discarded for that reason and re-taken after cooling.

The environment block each run prints — device, CPU, kernel, compiler, governor, throttle
state before and after, and the commit — is at the top of every aarch64 log in
[logs/](logs/).

## The workload

The sequence-level results use **EuRoC MAV `V1_02_medium`, camera `cam0`** — 1710 frames of
752×480 8-bit grayscale, giving 1709 consecutive frame pairs. It is used whole; no prefix,
no subsample.

Which sequence is not a detail. `V1_02` gives the tracker materially more work per frame than
the easier `MH_01_easy`, and a whole-frontend ratio measured on the two comes out differently
enough to change the conclusion. Every sequence-level number here names its sequence for that
reason, and all of them are `V1_02`.

Operation-level results do not need a dataset. They run on synthetic content across a ladder
of sizes — the filter benchmarks from 640×480 down to 94×60, a frame and the top level of a
four-level pyramid, and the bandwidth-bound ones upward to 8192×4096 — so
that a ratio which collapses once both sides fit in cache can be told apart from one that
holds.

## Reproducing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/benchmark/logic_benchmark                     # a primitive, against OpenCV
./build/benchmark/frontend_sequence <euroc-cam0-dir>  # the whole frontend, against OpenCV
```

Each report's **Reproduce** section names the exact binary for its tables. The sequence
benchmarks need a directory of `.png` frames; everything else is self-contained.

Two things will move your numbers more than anything in the code. **Link the `bincv_core`
CMake target** rather than only adding the include path — the ISA flags ride on the target,
and a consumer who added the include path alone measured binCV 2.25× slower on this device
without any indication that anything was wrong. And **check what OpenCV you are measuring
against**: the two builds used here differ in version and in dispatched instruction sets,
and the benchmarks print both so a number can be traced back to what produced it.
