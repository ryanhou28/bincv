# How a speed difference is decided

**Read this before quoting a speed ratio from any report here.** It is the sibling of
[methodology-memory.md](methodology-memory.md): that page says how a memory figure is
measured, this one says how a difference between two timings is judged real.

The rule is `benchmark/measure_util.hpp`'s, carried across to the paired design in
`backends/cuda/benchmark/paired_stats.hpp`, so it stands behind the speed figures in every
report in this directory. The worked examples are drawn from [cuda.md](cuda.md), because that
is where it was written down and tested.

## How a difference is decided

Every ratio in [cuda.md](cuda.md) is a **per-round paired** ratio: each round times both arms
back to back, alternating their order, so a drift that moved both arms divides out instead of
landing on whichever ran second. What is quoted is the **median** of those per-round ratios,
and what decides whether it is real is `measure_util.hpp`'s own rule:

> A difference smaller than the spread is a null result, and a null result is a result. The
> spread bounds WITHIN-run noise only; run-to-run scatter is a separate and sometimes larger
> number, and an entry that calls a difference real should clear the larger of the two.

So a difference must beat **the larger of two noises**, both measured as factors:

- **within-run** — how many times the per-round ratio itself swung inside one process,
  `max / min`. The *ratio's* swing, not the arms': charging it for the arms' spreads would
  charge it for the drift the pairing already removed.
- **run-to-run** — `max / min` of the per-run medians across seven independent processes. One
  process cannot see this, so `scripts/aggregate_cuda_runs.py` computes it from a directory of
  runs.

**Seven is a convention, not a sufficiency, and the shared x86-64 box has already broken it.**
Re-measuring `goodFeaturesToTrack` there took thirty pinned launches of one interleaved
benchmark: the per-launch ratio ran from 0.95× to 1.60×, a scatter of 57% of its own median,
and it scattered *wider than either arm did on its own* — so the pairing that is supposed to
divide out drift did not, because the drift is not common to the two arms on a machine with
other work on it. Seven launches there can return almost any answer; thirty put a bootstrap
95% interval of 1.11–1.15× around the median. The rule that follows is not a bigger number
for every row, it is that **a host's resolution has to be measured before the row is quoted,
and quoted with it** — see [features.md](features.md#corner-detection), where that is done.

## The protocol each host needs

**A single-launch x86-64 figure is not publishable here.** Every x86-64 number in these
reports is the **median of thirty pinned launches**, with a percentile bootstrap 95% interval
on that median (10,000 resamples, seed 12345) and each ratio formed *inside* a launch before
the median is taken — so a ratio is not the quotient of the two times printed beside it.
`scripts/run_launches.sh` takes the launches, `scripts/aggregate_launches.py` reads them, and
every sweep is committed as `logs/*-x86_64-launches.log` with its aggregate appended. A
ratio is quoted to the digits its interval separates and no further.

**Thirty is this host's price, not a project-wide bar.** `--ladder` resamples the launches
already taken to say what a shorter sweep would have said: on the corner row one launch
resolves nothing at all (±32%), ten resolve 1.053×, thirty 1.026×. What thirty launches
resolve is a **per-row** figure the aggregate prints as `minres`, and across the rows here it
ranges from 1.002× to 1.090×. A row whose `minres` exceeds the difference being claimed does
not support that claim, however many digits its median has.

**An interval is over the launches that were taken.** It is not a bound on what a different
thirty will say, and that was tested rather than assumed: seven rows were re-swept from
scratch on a busier machine before any figure here was adopted
([logs/README.md](logs/README.md#the-repeats-and-what-they-showed)). Six landed inside the
first sweep's interval; `bitwiseNot`'s repeat landed 0.7% above the top of it, with the two
intervals still overlapping. Read a quoted interval as the resolution of one sweep, and a
difference near its edge as unsettled until a second sweep agrees.

**aarch64 is quoted from one pinned launch, and that is enough there.** Seven launches of
the corner benchmark on the governor-locked Pi 4 hold 0.05–1.18% within a run and scatter
0.1–0.8% across the seven ([goodfeatures-aarch64.log](logs/goodfeatures-aarch64.log)) — the
run-to-run half is smaller than a single x86 launch's own printed spread. The protocol
follows the host, not the architecture.

The difference-against-spread test decides whether the *size* of a difference clears the
noise. It does not decide whether the *sign* is settled, and on one row the two came apart.

## The spread bounds the magnitude, not the direction

**A spread lying wholly on one side of 1.00× is uncertainty about how big the difference is,
not about whether there is one.** Charging it against the difference treats *"we do not know
whether this is 1.35× or 4.4×"* as if it were *"we do not know whether these arms differ at
all"*. Reading the noise test as though it answered both questions produced a verdict the
owner ruled wrong on 2026-09-19; the case is [below](#the-case-that-forced-the-ruling-and-where-it-landed).
So there are **three verdicts**:

| verdict | what it says | how the row is quoted |
|---|---|---|
| **DIRECTION ESTABLISHED** | no round crossed 1.00× | a **range**, smallest to largest per-round factor, beside the median and the sign count |
| **A RESULT** | the difference exceeds the larger noise | the median, as before |
| **NULL RESULT** | neither | "the same speed as far as this run can tell" — still a result |

**A row can be both of the first two, and the strong ones are.** They are different statements
— *which arm is ahead is not in doubt* and *the distance between them exceeds the noise* — so
both get printed. Measured on the seven runs behind this section: a unanimous row whose win
varies more than three-fold is direction-established and **not** a result (LK at 204 points,
1.35×–4.36×, 105 of 105); a row that clears the noise on a 2–103 split is a result whose
direction is **not** established (`threshold`'s byte-lane arm at 1920×1080, 2.39× apart, two
rounds the other way). Collapsing either into the other loses the half that was true.

**No round-count threshold, which is why the sign test's exact p is printed.** Two unanimous
rounds and a hundred unanimous rounds both satisfy "no round crossed", and they are not equally
strong evidence. A minimum round count would be a project-wide "X is enough" bar invented on
the spot, which is the one thing `CLAUDE.md` says not to do. What is printed instead is the
exact two-sided sign-test p: 1.0 at one round, 0.5 at two, 6.1×10⁻⁵ at fifteen, 4.9×10⁻³² at
the 105 behind these rows. The **sign count** is printed beside it, because p answers *how many
rounds* and nothing else — see the knife edge at the end of this page.

**A tie breaks the direction verdict, and that is the inconvenient choice.** A round whose two
arms time identically favours neither. Three reasons it counts against a verdict that claims
*every* round fell one way: the sentence the verdict licenses is "faster in N of N rounds", and
dropping a tie from N makes that sentence false; ties get **more** common as the clock gets
coarser, so excluding them would make the verdict easier to earn the *worse* the instrument is;
and the tie bucket also holds rounds that were **not measurements at all** — a clock that read
zero — which are counted separately so a row says which kind it tripped on.

`PairedStats.ATieBreaksTheDirectionVerdict` pins the arithmetic: nine rounds at 1.50× and one
exact tie. A tie-blind predicate would call it unanimous and then print "won by **1.00×** to
1.50×", quoting as a win a round in which nobody won. Remove the tie and the same nine rounds
establish a direction at 1.50×–1.50×. Ties did occur in the sweeps behind these reports, all
genuine measurements, and they fell on rows that were mixed-sign anyway. The sign *test* still
excludes ties from n, which is what a sign test does with them.

**What did not change.** The medians, the geometric mean, the factors, the swap-invariance and
the noise predicate are exactly as they were. DIRECTION ESTABLISHED is an **additional**
statement about the same rounds, not a second way to pass the old one — pinned by
`PairedStats.TheOldPredicateIsUntouchedByTheRuling` and confirmed by outcome: re-judging 146
published rows on the same rounds moved zero across the RESULT/NULL line.

**Two things this is not.** It is not a test for disjoint sample ranges: a range test is vetoed
by a single round that is slow in *both* arms — which is drift, the thing pairing exists to
cancel — and it passes cleanly separated arms whose ratio scatters from 1.01× to 1.60×.
Separation is still computed and printed as a fact beside the verdict. It is also not a p-value
threshold: the sign test is reported at every row and gated on at none.

**Why the deciding quantities are factors and not percentages.** The obvious spelling —
`|median − 1|` against `(max − min) / median`, both as percentages — is not a comparison at
all, because it depends on which arm is the denominator. `|median − 1|` cannot exceed 100% for
the arm that is *faster*, however far ahead it is, while the spread has no ceiling. Measured on
binCV's `erode` on a 5×5 ellipse against `cv::cuda`'s, fifteen paired rounds at 752×480, one
process:

| orientation | difference | spread | verdict |
|---|---|---|---|
| binCV / OpenCV | 93.5% | 267.2% | NULL |
| OpenCV / binCV | 1427.1% | 106.2% | RESULT |

Same rounds, same arms, opposite answers, decided by argument order — one arm is **fifteen
times** the other and one spelling reports "the same speed as far as this run can tell". In
factors both quantities survive the inversion (**15.27× apart against a 4.62× swing**, either
way round). `PairedStats.TheVerdictDoesNotDependOnWhichArmIsTheDenominator` pins it. It is not
a coincidence of one run: across the seven processes behind this section the same pair flips
verdict with argument order in **four of seven**, and the factor spelling reads RESULT in all
seven.

**The median of an even number of ratios is their multiplicative midpoint**, for the same
reason. An arithmetic midpoint does not commute with inverting the ratio, so at even round
counts the verdict depended on argument order even in factors: two rounds whose ratios are 1.0
and 1.5127 read **1.2563× apart one way and 1.2040× the other**. Over a sweep of random pairs
every even-count pair disagreed with its own mirror image, four of them all the way to opposite
verdicts; odd counts were already exact. `backends/cuda/tests/test_cuda_bench_stats` pins the
whole of this against hand-computed values, in 280 checks, and needs no GPU.

## The case that forced the ruling, and where it landed

**The row was Lucas-Kanade at the tracking pipeline's own keypoint spacing.** Every one
of its paired rounds favours binCV, all seven runs are range-disjoint, and the two arms were **never once
seen closer than 1.35× apart**. Yet the per-round ratio swings 2.49× across those rounds —
because the rounds where binCV wins by 4.4× sit further from 1.00× than the rounds where it
wins by 1.35× — and a difference of 1.84× does not exceed 2.49×. So the rule, read literally,
called *"the two arms are the same speed as far as this run can tell"* on a pair this machine
never once saw level.

**The owner ruled on 2026-09-19: the spread bounds the magnitude, not the direction.** The row
is reported as a range with its sign count:

> **binCV is faster in 105 of 105 paired rounds, by 1.35× to 4.36×** (median 1.84×; `cv::cuda`
> 0.1475 ms against binCV 0.0792 ms, p = 4.9×10⁻³²).

**Almost all of the swing lives in the OpenCV arm**: across these seven runs binCV's own
medians move by 1.007× and `cv::cuda`'s by 1.29×.

**Where the ruling landed, over a seven-process re-take of every role bar:**

| row | `cv::cuda` vs binCV, per-run medians | rounds won by binCV | now |
|---|---|---|---|
| **LK at the pipeline's spacing (204 pts)** | 0.1475 ms vs **0.0792 ms** | **105 of 105** | **direction established, 1.35×–4.36×**; magnitude a null |
| LK @256 | 0.1467 ms vs **0.0728 ms** | **105 of 105** | direction established, 1.43×–5.65×, **and** a result at 2.00× |
| LK @512 | 0.1634 ms vs **0.1056 ms** | **105 of 105** | direction established, 1.13×–3.80×; magnitude a null |
| `threshold` 3840×2160 | 0.0362 ms vs **0.0268 ms** | 97 of 105 | still a null on both halves |
| `threshold` 1920×1080 | 0.0116 ms vs **0.0100 ms** | 88 of 105 | still a null on both halves |
| `threshold` 752×480 | 0.0091 ms vs **0.0084 ms** | 60 of 105, one round tied | still a null on both halves |

The `threshold` rows cost nothing to restate: that op's written rule was a *fail* condition —
slower than `cv::cuda::threshold` by more than both spreads — and a null result is not slower.
**`PASS` there means "no longer fails", and it still does.** What should stop being quoted is
the *magnitude*: 1.35× at 4K is a median this host's noise does not resolve, and eight of its
105 rounds fell the other way.

**A follow-on question was raised here and answered by reading the rule it was raised
against.** The worry was that the ship rule asks whether an operation "holds up on both axes",
worded as a pass/fail, while a direction-established row is a range. But that rule is about a
kernel that *"loses its role comparison badly"*, and its remedy is that such a kernel "gets
optimized first, or the owner explicitly accepts the gap". It has nothing to say about a kernel
that wins: LK is ahead in every round of two independent sweeps and **3.14× smaller** on
resident state. LK ships, and [its role bar](cuda.md#speed-operation-by-operation) reads as the comparison it
is.

The rule still bites, and on this backend it currently bites one operation.
[`cornerSubPixAsync`](cuda.md#what-is-not-delivered) is
measured losing to the whole-plane round trip it exists to avoid — a loss in the sense the rule
means — so it is optimized or the gap is accepted explicitly, not excused by a range.

## What changed when the rule was applied

Two things happened here, a round apart. First the range test was replaced by
`measure_util.hpp`'s own difference-against-spread rule, which moved rows in **both**
directions. Then the 2026-09-19 ruling **added** the direction verdict without touching that
rule. The table below is a fresh seven-process sweep, 105 paired rounds a row, judged under all
three values.

**Re-judging the same rounds moved nothing across the RESULT/NULL line.** Over 146 rows, zero
went RESULT→NULL or NULL→RESULT. What the ruling added is a verdict for rows that were
unanimous and had nowhere to say so. (Re-*measuring* does move a few: five of 137 rows common
to two sweeps flip, all within a few percent of their own bar. That is the host, not the rule —
see the caveat at the end.)

| effect | both arms, per-run medians | old verdict | now |
|---|---|---|---|
| **LK at the pipeline's own spacing, 204 pts** | `cv::cuda` **0.1475 ms** [0.1314–0.1700] · binCV **0.0792 ms** [0.0787–0.0794] | null (1.84× apart against a 2.49× swing) | **DIRECTION ESTABLISHED** — **105 of 105**, by **1.35× to 4.36×**, p = 4.9×10⁻³² |
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
| `threshold`'s byte-lane arm vs grid-stride @1920×1080 | — | RESULT | **RESULT, not direction-established** — 2–103 |
| morphology 3×3 specialization, 752 / 1920 | — | null | **still null** — 20–56 / 8–69 |
| morphology word-parallel border @1920×1080 | — | RESULT | **RESULT, not direction-established** — 1–76 |
| morphology word-parallel border @752×480 | — | null | **still null** — 5–72 |
| `andNot` GRADIENT fusion, 752 / 1920 | — | null | **still null** — 11–66 / 7–70 |
| `cornerSubPixAsync` round-trip rule | device arm vs whole-plane round trip | null at parity | **RESULT in this sweep, against the device arm** — 5–72, 1.44× apart against a 1.39× bar. Marginal, and it read null over the previous fourteen runs |
| `edgeThreshold` vs `cv::cuda` deriv+abs+threshold+or @3840×2160 | — | RESULT | DIRECTION + RESULT — 63 of 63, **15.0× to 61.5×** |

**Seven rows are a RESULT without a direction**, each with one to five rounds falling the other
way out of a hundred — a distinction the two-valued table could not draw at all.

**Two harness defects surfaced while re-judging, and both were producing published numbers.**

- `gftt_wall` and the pipeline's wall-clock pair assembled their paired summary **by hand**
  instead of through `summarizePaired`, so `roundsFavouringA/B` were left at zero. Two published
  role bars printed a **0–0 sign split and p = 1 over rounds that are in fact 105–0**, and took
  the ratio's median from the *arithmetic* even-count midpoint the factor spelling exists to
  avoid. The `goodFeaturesToTrack` wall row above is the first honest judging of that
  comparison.
- `cuda_sensor_benchmark` never set its per-geometry scope, so its two geometries pooled under
  one key and the aggregation read **the difference between the geometries** as run-to-run
  scatter. Pooled, the byte-lane pair reads a 2.82× "scatter" and nulls; scoped, 3840×2160 is
  **0.1396 ms against 0.0498 ms — 63 of 63 rounds, 1.65× to 3.47×** — and 752×480 is a separate
  null (0.0101 ms against 0.0078 ms, 9–54). That geometry carries **no bar**, so no published
  claim moved.

**The headline was the case that exposed the original defect.** Under the percentage spelling
the binary dense result read NULL, because its faster arm sits a few multiples above the launch
floor and swings while StereoBM's does not. It is a result under the rule as written.

**`cornerSubPixAsync`'s round-trip rule is still not met, and this sweep says so more strongly
than the last.** Its written ship condition is that the device arm be *strictly cheaper* than
the round trip it replaces, priced against the tighter of two baselines. Measured as **one
paired thing** rather than three separately-timed medians added together, and against the
whole-plane download (the tighter baseline here by 50×): the device arm reads **0.5656 ms** and
the round trip **0.3911 ms**, with 72 of 77 rounds favouring the round trip — 1.44× apart
against a 1.39× bar. The previous "met in 6/7 runs" came from comparing three
independently-drifting medians. **This is a stop-and-ask**: a measurement contradicting a
documented claim, reported rather than fixed.

**The launch floor, not the rule, is what blocks the internal packer and morphology arms.**
Their per-round ratio swings 2–3× inside a single process while their per-run medians agree to
within a few percent: at 1920×1080 the row-grid arm's seven medians are 0.7005, 0.7344, 0.7084,
0.7036, 0.7039, 0.7022, 0.7366 — a 1.05× scatter — and 99 of 105 paired rounds favour the arm.
But six rounds crossed, so the direction is not established either, and the rule takes the
*larger* of the two noises. That is the rule behaving correctly on a host whose individual
0.4 ms batches are at the mercy of WSL2 scheduling. Shrinking it means more enqueues per round,
which would move every number in these reports.

**One caveat belongs with every number above: on this host NEITHER half of the verdict is
stable at the margin.** Two independent sweeps of this branch — the seven processes behind this
section and a separate fourteen — share 146 rows. Between them **3 rows flip RESULT↔NULL and 7
flip direction**:

| row | 14-run sweep | 7-run sweep |
|---|---|---|
| `goodFeaturesToTrack`, device-wide sort ladder | 151–3 | **77–0** |
| packer byte lane vs grid-stride @1920×1080 *(two keys, one comparison)* | 2–208 | **0–105** |
| `matcher_u8` @470×470 (does not decide) | 2–208 | **0–105** |
| `packBits` vs `binarize` n=2 @3840×2160 (no bar) | 2–124 | **0–63** |
| `denoise3` vs byte bar @4096×2160 | **0–210** | 1–104 |
| `matcher_s32` @470×470 (does not decide) | **0–210** | 1–104 |

**Every one sits within three rounds of the boundary**, which is what an all-or-nothing
criterion must do there: a row at 1–104 and a row at 0–105 are the same measurement on either
side of a knife edge. And the printed p does not warn you: 0–105 is p = 4.9×10⁻³² and 1–104 is
p = 5.2×10⁻³⁰ — two orders apart on a scale where both mean "overwhelmingly one-sided". A
reader wanting to know whether a direction row is near its edge has to read the **sign count**,
which is why it is printed beside every verdict and never summarized away.

**What is stable is the rows that are not near the edge.** Fifty rows are direction-established
in both sweeps, and not one has a single minority round in either. All five LK densities the
ruling moved are 0–210 in one sweep and 0–105 in the other: **no round has crossed in 315
paired rounds across 21 processes.** That is the distinction to make when quoting a direction
row.
