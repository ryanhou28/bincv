# Raw benchmark output

Every figure in the reports above comes from a file here. Two shapes live in this directory
and they are not interchangeable.

| name | what it is |
|---|---|
| `<bench>-x86_64-launches.log` | **thirty separate pinned process launches** of one benchmark, with the aggregate appended as a comment block. This is what an x86-64 figure in the reports is. |
| `<bench>-x86_64.log` | the **single launch** that figure used to be. Kept, not deleted — a number that moved should be checkable against the reading it replaced. |
| `<bench>-aarch64.log` | the reference device, one pinned launch with the governor locked. That host's run-to-run scatter is 0.1–0.8%, so one launch is quoted there. |

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
