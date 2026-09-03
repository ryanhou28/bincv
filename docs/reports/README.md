# binCV measurement reports

What binCV costs and what it saves, measured against OpenCV on two architectures.

Each report states a claim, names the OpenCV call it is measured against, gives speed and
memory together, and ends with the command that reproduces it. The raw output every table
was cut from is in [logs/](logs/).

| report | what it covers | headline |
|---|---|---|
| [frontend.md](frontend.md) | a whole tracking frontend, end to end, over a real sequence | **3.30× / 4.73× faster, 6.23× smaller** |
| [primitives.md](primitives.md) | logic, reductions, denoise, morphology, derivative, pyramid downsample | 1.0× to 58×, by operation |
| [features.md](features.md) | corner detection, FAST, descriptors, matching, optical flow | **7.13× / 8.26×** on optical flow |
| [footprint.md](footprint.md) | the memory result on its own, and the speed declined to protect it | **6.23×** over the frontend |
| [limits.md](limits.md) | where binCV ties, loses, or stops paying at all | four ways it stops working |
| [methodology-memory.md](methodology-memory.md) | how memory is measured, and the errors that shaped it | read before quoting a memory number |

Paired figures are x86-64 then aarch64. The two are different measurements against different
OpenCV builds and are never averaged.

## At a glance

The whole result in one table — same commit, same sequence, one thread on each side:

| | x86-64 | aarch64 |
|---|---|---|
| whole frontend, speed | **3.30×** | **4.73×** |
| whole frontend, peak memory | **6.23× smaller** | **6.23× smaller** |
| optical flow, LK against LK | 7.13× | 8.26× |
| spatial derivative | 11.44× | 24.28× |
| denoise | 17.58× | 57.66× |
| `bitwiseAnd` | 10.01× | 28.59× |
| `pyrDown`, 1 bit in | 1.56× | 5.56× |
| `erode` 3×3 | 1.04× | 1.00× |
| FAST, wide image | 1.05× | 0.96× |
| `goodFeaturesToTrack` | 0.92× | **1.45×** |
| `erode`, 5×5 ellipse | 0.32× | 0.51× |

## What these are not

They are not a survey of binCV against every alternative, and they are not tuned
comparisons. Every number is one build of binCV against one build of OpenCV on one
machine, taken on a single commit, with the losses reported alongside the wins.

Nothing here is a claim about a target that has not been measured. binCV supports 32-bit
ARM, Cortex-M and RISC-V, and none of them appear in these reports because none of them
have been built and timed.

## Platforms

Both are measured. Neither is a stand-in for the other: an x86 ratio and an aarch64 ratio
are different measurements against different OpenCV builds, and the reports never average
them or quote one as the other.

| | **x86-64** — development host | **aarch64** — reference device |
|---|---|---|
| CPU | AMD Ryzen 5 5600X, 6 cores / 12 threads | Broadcom BCM2711, Cortex-A72, 4 cores |
| cache | 32 KiB L1d per core, 512 KiB L2 per core, 32 MiB shared L3 | 32 KiB L1d per core, 1 MiB shared L2 |
| OS | Ubuntu 22.04 under WSL2 | Raspberry Pi OS Lite, 64-bit |
| compiler | g++ 11.4.0, Release `-O3 -DNDEBUG` | g++ 14.2.0, Release |
| OpenCV | 4.8.0-dev, baseline SSE3, dispatching through AVX-512 | 4.10.0, NEON baseline |
| binCV vector paths | `POPCNT`, AVX2 selected at run time | NEON |

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
cmake -S bincv-cpp -B build -DCMAKE_BUILD_TYPE=Release
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
