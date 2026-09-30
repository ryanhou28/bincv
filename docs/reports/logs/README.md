# Raw benchmark output

Every figure in the reports above comes from a file here. Two shapes live in this directory
and they are not interchangeable.

| name | what it is |
|---|---|
| `<bench>-x86_64-launches.log` | **thirty separate pinned process launches** of one benchmark, with the aggregate appended as a comment block. This is what an x86-64 figure in the reports is. |
| `<bench>-x86_64.log` | the **single launch** that figure used to be. Kept, not deleted — a number that moved should be checkable against the reading it replaced. |
| `<bench>-aarch64-launches.log` | **ten separate pinned process launches** on the reference device, governor locked to `performance` and restored after, with the aggregate appended. This is what an aarch64 figure in the reports is. The two stereo binaries have seven. `wordwidth-aarch64-launches.log` has no aggregate because it prints byte counts, not timings. |
| `<bench>-aarch64.log` | the **single launch** that figure used to be. Kept for the same reason as the x86 singles. |
| `<bench>-spotcheck-aarch64-launches.log` | an **independent** device sweep of a benchmark whose figure moved, taken separately to check it. Two exist, for `feature-tracking` and `pyrfilter`. |
| `<bench>-x86_64-launches-repeat.log` | a **second, independent** thirty-launch sweep of the same benchmark, taken on a deliberately busier machine before the figures were published. Two exist, for `logic` and `morphology`. |
| `<bench>-cuda-launches.log` | **separate processes of one CUDA benchmark on the reference GPU**, written by `scripts/run_cuda_launches.sh`. Three are committed — `cuda_role-`, `cuda_role_lk-` and `cuda_sparse-x86_64-cuda-launches.log` — and the fourteen marked rows of [cuda.md](../cuda.md)'s speed table come from them. There is more than one because a benchmark measures what it measures: the Lucas-Kanade rows need a real frame sequence and refuse to synthesize one, and block matching is timed in the sparse family's own binary. The header contract is described below. |

## Which commit a log names

`run_launches.sh` stamps each log with the commit it was taken at. Those are commits on the
branch that did the measuring, and the branch was squash-merged, so the stamp itself is not
reachable from `main`. This table maps each stamp to the commit on `main` that carries the
same code and the log:

| log stamp | on `main` as | what the merge was |
|---|---|---|
| `80ff0a8`, `ac33cf1`, `05ab53c`, `0b9b73c`, `a152536` | `086428c` | the launch-sweep protocol and the re-taken x86-64 and aarch64 columns |
| `25065d7` (the single-launch logs), `83087b0`, `592bce4` | `8780bd1` / `c3e1f78` | the first measurement reports |
| `05ce58f` | `c6192bb` | footprint report corrections |
| `880704b`, `6d74d57` | `8729e05` | the `goodFeaturesToTrack` selection-stage optimization and its re-take |
| `7c8055f`, `06246c2`, `5c6a47d`, `aa8d4bb`, `a608102` | `0d6e302` | the CUDA figures with provenance |
| `550d45a` | `cb995e7` | the bit-plane FAST gate read once per call |
| `d13ea10`, `a8214de`, `9ab7325` | `77c46a0` | the RANSAC estimators |

Logs taken on the pre-release branch (`b36dc73`, `211acaa`, `4c4b3bf`) name commits of that branch,
reachable from `main` once it is merged.

`scripts/check_figure_staleness.py` compares the code a log measured, at its stamp, against
the current tree; `expected-stale.txt` lists the logs whose code has moved, with the argument
for why each figure still holds.

## The CUDA sweeps

Every CUDA figure is the median of 7 independent process runs, and the runs are committed
because a figure with no run behind it cannot be checked: before these sweeps were kept, a
`goodFeaturesToTrack` row re-run at its own published commit read 8.40× on the machine
against 5.3× on the page, with nothing left behind to say which session was wrong.

`scripts/run_cuda_launches.sh` is the CUDA counterpart of `run_launches.sh`. It writes one
file per sweep with `### run N` between launches — the same shape, so
`scripts/aggregate_cuda_runs.py` reads a committed sweep and a directory of per-process
outputs identically — and a header that records what a CUDA figure is a claim about:
the **device** and its compute capability, the **driver** and the CUDA version it exposes,
the **nvcc** and **architecture** the kernels were built for (read from the build tree, not
from `PATH` — this backend builds under 11.1 while the driver here exposes 12.6), the
**OpenCV** supplying the role comparison's denominators, the **GPU clock, throttle reasons
and temperature** before and after, and that **no profiler was running**. That last one is a
refusal, not a note: `ncu` replays every kernel to collect its counters, so a timing taken
beside one is a timing of the replay, and the run will not start.

The clock is recorded rather than held. No host this backend has run on permits locking it
— WSL2 reports application clocks as `N/A` — so the header says so, and an unlocked clock
shows up where it should, in the scatter across launches.

The sweep is gated like the host ones: `scripts/check_figure_staleness.py` maps a CUDA log
through its benchmark's includes **and one link step**, because the backend is not
header-only. A benchmark includes `bincv/cuda/threshold.hpp`, which declares the entry and
contains no kernel; the kernel is `backends/cuda/src/threshold.cu`, a translation unit no
`#include` names. The gate pairs the two by name, checks that pairing is complete before it
will run, and reads the rest of a benchmark's translation units — `cuda_bench_null.cu`
carries the launch floor — out of the `add_executable` that names them. `--explain <log>`
answers for one log, and the runner calls it on what it has just written: a sweep the gate
cannot map is a sweep whose figures nothing can ever call stale.

## The device spot-checks

When the device sweep moved a published figure beyond its band, two rows were re-taken as an
independent sweep before anything was adopted — one that moved and one that did not. The rule
was written first, in the same shape as the x86 repeats below: the spot-check's interval
must overlap the sweep's, and a disagreement is investigated rather than averaged.

| row | device sweep | independent spot-check | |
|---|---|---|---|
| the assembled pipeline, ms/frame | 5.097 [5.084, 5.145] | 5.101 [5.093, 5.124] | agrees; both outside the published 4.906–4.949 |
| the assembled pipeline, ratio | 4.6199 [4.5957, 4.6278] | 4.6160 [4.6112, 4.6371] | agrees |
| — of which `track (LK)` | 3.787 | 3.791 | agrees — the stage the regression is in |
| — of which `pyrDown` | 0.189 | 0.190 | agrees — halved from a published 0.377 |
| `pyrDown` 1-bit, ratio | 5.5093 [5.4797, 5.5493] | 5.4920 [5.4717, 5.6411] | agrees; binCV's arm 93.8 µs against 93.7 |

The clock was sampled every two seconds on the pinned core throughout both, 169 samples all
at 1,800,000 kHz, peak 62.8 °C — because `vcgencmd get_throttled` on this board reads
`0x80000` before and after and cannot report a *new* event.

## The x86-64 repeats

Two x86-64 sweeps were re-taken independently on a deliberately busier machine before any
figure was adopted: one row that had changed sign between protocols (`dilate` 3×3) and one
that had not (`bitwiseAnd`), plus five others in the same binaries. The rule was written
first: the repeat's bootstrap interval must overlap the first sweep's, and a disagreement is
investigated rather than averaged.

| row | first sweep | repeat | |
|---|---|---|---|
| `dilate` 3×3, `uint32` | 1.0570 [1.0463, 1.0725] | 1.0589 [1.0527, 1.0776] | agrees |
| `erode` 3×3, `uint32` | 1.0532 [1.0351, 1.0656] | 1.0369 [1.0297, 1.0611] | agrees |
| `morphologyEx(OPEN)`, `uint32` | 1.0222 [1.0158, 1.0430] | 1.0251 [1.0169, 1.0406] | agrees |
| `erode` 5×5 ellipse, `uint32` | 0.3189 [0.3177, 0.3227] | 0.3186 [0.3152, 0.3213] | agrees |
| `bitwiseAnd`, `uint32` | 9.9748 [9.8200, 10.2812] | 10.2036 [9.9359, 10.4291] | agrees |
| `bitwiseOr`, `uint64` | 9.9857 [9.8834, 10.1002] | 9.9405 [9.7337, 10.1349] | agrees |
| `bitwiseNot`, `uint32` | 17.9859 [17.8190, 18.1402] | 18.2741 [18.0552, 19.0970] | intervals overlap; **the repeat's median is 0.7% above the first sweep's interval** |

**Six of the seven repeats land inside the first sweep's interval and one lands just
outside it.** That is the caveat to carry: a bootstrap interval is over the thirty
launches that were taken, and is not a bound on what a different thirty will say.

## Why sweeps rather than single launches

`benchmark/measure_util.hpp` reports the spread *within* a process by construction, and says
so; a process cannot see what it paid once, at start-up, for the whole of its own run. On the
x86-64 host that gap is eleven-fold — `goodFeaturesToTrack` prints a ~5% spread and the same
ratio scatters 66% across sixty launches — which is why the single-launch `<bench>-x86_64.log`
files are kept only as the readings the sweeps replaced.

The sweeps were taken with `scripts/run_launches.sh` and read back with
`scripts/aggregate_launches.py`, whose output is the `# ---- aggregate` block at the foot of
each `-launches.log`. Re-running the aggregator on one of those files reproduces it, except
that the appended block itself parses as extra tables — read the ones numbered before it.

[The index](../README.md#changes-to-published-figures) lists the published figures that moved
when the sweeps replaced the single launches, and why.

## The x86-64 logs with no sweep

Seven are here without a `-launches` twin, deliberately:

- `essential-x86_64.log`, `ransac-x86_64.log` — no report links to them and none publishes a
  timing from them.
- `pyramid-x86_64.log`, `wordwidth-x86_64.log` — their published figures are computed byte
  counts, exact and architecture-independent, plus one aarch64 ratio.
- `essential_stack-x86_64.log`, `feature-tracking-rss-x86_64.log`,
  `feature-tracking-threads-x86_64.log` — stack bytes, resident set size and thread counts,
  not timings. The threading log is the one place an un-swept x86 *timing* is still published
  ([feature-tracking.md](../feature-tracking.md#what-this-does-not-claim)), because a
  threading arm cannot be pinned to one core; that table says so.

`lk_batch_arm-x86_64.log` is superseded by `lk_batch_off-` and `lk_batch_on-` and is kept as
the earlier two-run reading.
