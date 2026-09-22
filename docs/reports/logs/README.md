# Raw benchmark output

Every figure in the reports above comes from a file here. Two shapes live in this directory
and they are not interchangeable.

| name | what it is |
|---|---|
| `<bench>-x86_64-launches.log` | **thirty separate pinned process launches** of one benchmark, with the aggregate appended as a comment block. This is what an x86-64 figure in the reports is. |
| `<bench>-x86_64.log` | the **single launch** that figure used to be. Kept, not deleted — a number that moved should be checkable against the reading it replaced. |
| `<bench>-aarch64-launches.log` | **ten separate pinned process launches** on the reference device, governor locked to `performance` and restored after. This is what an aarch64 figure in the reports is. The two stereo binaries have seven. |
| `<bench>-aarch64.log` | the **single launch** that figure used to be. Kept for the same reason as the x86 singles. |
| `<bench>-spotcheck-aarch64-launches.log` | an **independent** device sweep of a benchmark whose figure moved, taken separately to check it. Two exist, for `feature-tracking` and `pyrfilter`. |
| `<bench>-x86_64-launches-repeat.log` | a **second, independent** thirty-launch sweep of the same benchmark, taken on a deliberately busier machine before the figures were published. Two exist, for `logic` and `morphology`. |

## The device spot-checks, and what they showed

The device sweep moved one published figure beyond its band and left the rest where they
were, so two rows were re-taken as an independent sweep before anything was adopted — one
that moved and one that did not. The rule was written first, in the same shape as the x86
repeats below: the spot-check's interval must overlap the sweep's, and a disagreement is
investigated rather than averaged.

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

## The repeats, and what they showed

Two sweeps were re-taken independently before any figure was adopted: one row that
moved and changed sign (`dilate` 3×3) and one that did not (`bitwiseAnd`), plus five
others that came free with them. The rule was written first: the repeat's bootstrap
interval must overlap the first sweep's, and a disagreement is investigated rather than
averaged.

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

## What changed, and when

Until 2026-09-21 every x86-64 log in this directory was a single process launch, and every
x86-64 figure in the reports was one draw from a distribution nobody had characterised.
`benchmark/measure_util.hpp` reports the spread *within* a process by construction, and says
so; a process cannot see what it paid once, at start-up, for the whole of its own run. On
this host that gap is eleven-fold — `goodFeaturesToTrack` prints a ~5% spread and the same
ratio scatters 66% across sixty launches.

The sweeps were taken with `scripts/run_launches.sh` and read back with
`scripts/aggregate_launches.py`, whose output is the `# ---- aggregate` block at the foot of
each `-launches.log`. Re-running the aggregator on one of those files reproduces it, except
that the appended block itself parses as extra tables — read the ones numbered before it.

[The index](../README.md#on-the-x86-64-host) says which published figures moved and why.

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
