# How a speed difference is decided

**Read this before quoting a speed ratio from any report here.** It is the sibling of
[methodology-memory.md](methodology-memory.md): that page says how a memory figure is
measured, this one says how a difference between two timings is judged real.

The rule is `benchmark/measure_util.hpp`'s, carried across to the paired design in
`backends/cuda/benchmark/paired_stats.hpp`, so it stands behind the speed figures in every
report in this directory. The worked examples are snapshots of [cuda.md](cuda.md)'s tables
at the time of writing; cuda.md is authoritative for those figures, and the host-CPU examples
name the report that owns them.

## How a difference is decided

Every ratio in [cuda.md](cuda.md) is a **per-round paired** ratio. A round is one interleaved
timing of both arms — each arm timed once, back to back, in alternating order — so a drift
that moved both arms divides out instead of landing on whichever ran second; a process runs a
fixed number of rounds (fifteen for most rows) and a sweep is seven processes, 105 rounds.
What is quoted is the **median** of those per-round ratios, and what decides whether it is
real is `measure_util.hpp`'s own rule:

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

**Seven is a convention, not a sufficiency, and the shared x86-64 host breaks it.**
Re-measuring `goodFeaturesToTrack` there took thirty pinned launches of one interleaved
benchmark: the per-launch ratio ran from 1.26× to 1.62×, a scatter of 25% of its median,
against 30% and 12% for its two arms on their own — so the pairing removed little of the
drift, because on a machine with other work on it the drift is not common to the two arms.
Seven launches there can return almost any answer; thirty put a bootstrap 95% interval of
[1.350, 1.426] around a median of 1.383× ([features.md](features.md#corner-detection)). The
rule that follows is not a bigger number for every row, it is that **a host's resolution has
to be measured before the row is quoted, and quoted with it**.

## The protocol each host needs

**A single-launch x86-64 figure is not publishable here.** Every x86-64 number in these
reports is the **median of thirty pinned launches**, with a percentile bootstrap 95% interval
on that median (10,000 resamples, seed 12345) and each ratio formed *inside* a launch before
the median is taken — so a ratio is not the quotient of the two times printed beside it.
`scripts/run_launches.sh` takes the launches and `scripts/aggregate_launches.py` reads them.
Every x86-64 sweep is committed as `logs/*-x86_64-launches.log` with its aggregate appended;
the aarch64 sweeps are committed without the aggregate block, and their intervals come from
running the aggregator on the log. Times are quoted to four significant figures, ratios and
intervals to three.

**Thirty is this host's price, not a project-wide bar.** `--ladder` resamples the launches
already taken to say what a shorter sweep would have said: on the corner row one launch
resolves nothing at all (±32%), ten resolve 1.053×, thirty 1.026×. What thirty launches
resolve is a **per-row** figure the aggregate prints as `minres`, and across the rows here it
ranges from 1.002× to 1.090×. A row whose `minres` exceeds the difference being claimed does
not support that claim, however many digits its median has.

**Launch noise on this host is one-sided.** Of 85 x86-64 time cells re-taken at thirty
launches on unchanged kernels, 61 read slower in a single launch than in the median of thirty;
the worst overstatement is 49% and the worst understatement 3.3%, and the median cell reads
1.6% slow. A launch can go badly wrong and cannot go much right, so a single launch biases
times slow while ratios largely survive, because a launch that lands slow lands slow on both
arms at once — `wordtype_narrow`'s three arms all read about 20% high in one launch and the
ratios between them did not move. A figure quoted as a time therefore cannot be rescued by a
later sweep of ratios; it has to be replaced.

**An interval is over the launches that were taken.** It is not a bound on what a different
thirty will say, and that was tested rather than assumed: seven rows were re-swept from
scratch on a busier machine before any figure here was adopted
([logs/README.md](logs/README.md#the-x86-64-repeats)). Six fell inside the
first sweep's interval; `bitwiseNot`'s repeat fell 0.7% above the top of it, with the two
intervals still overlapping. Read a quoted interval as the resolution of one sweep, and a
difference near its edge as unsettled until a second sweep agrees.

**aarch64 is quoted from ten pinned launches, governor locked to `performance` and restored
after.** Ten rather than thirty because the device resolves far more per launch: `--ladder` on
the corner row there resolves **1.008× at one launch and 1.005× at ten**, where the same row
on x86-64 resolves nothing at one (±32%) and 1.026× at thirty. Ten was chosen to cover every
published row rather than a few rows deeply, because the question the device sweep answers is
whether a figure is still *true*, which shows up in the first launch, and not how finely it
can be split. `minres` is still per-row and still printed: the loosest row behind a published
device figure resolves 1.065×, not 1.005×.

**On the device, small-plane rows scatter more than large ones, and the mechanism is binCV's
own advantage.** `goodFeaturesToTrack` scatters 1.2% across ten launches. binCV's 640×480
`bitwiseAnd` arm scatters 16.9%, because a 115 KB working set of packed planes (KB is 1000
bytes here) has its cache residency decided per launch by where the allocator put it, while
OpenCV's 921 KB arm never fits the 1 MiB L2 and scatters 4.0%. At 8192×4096, where neither side fits, the two arms
scatter 2.3% and 1.9%. A host's resolution is a property of the row as much as the machine.

The difference-against-spread test decides whether the *size* of a difference clears the
noise. It does not decide whether the *sign* is settled, and on one row the two came apart.

## The spread bounds the magnitude, not the direction

**A spread lying wholly on one side of 1.00× is uncertainty about how big the difference is,
not about whether there is one.** Charging it against the difference treats *"we do not know
whether this is 1.4× or 5.9×"* as if it were *"we do not know whether these arms differ at
all"*. Reading the noise test as though it answered both questions is what the rule as first
written did; the case that changed it is [below](#the-case-that-changed-the-rule). So there
are **three outcomes**:

| outcome | what it says | how the row is quoted |
|---|---|---|
| **DIRECTION ESTABLISHED** | no round crossed 1.00× | a **range**, smallest to largest per-round factor, beside the median and the sign count |
| **A RESULT** | the difference exceeds the larger noise | the median, as before |
| **NULL RESULT** | neither | "the same speed as far as this run can tell" — still a result |

**A row can be both of the first two, and the strong ones are.** They are different statements
— *which arm is ahead is not in doubt* and *the distance between them exceeds the noise* — so
both get printed. A unanimous row whose win varies four-fold is direction-established and
**not** a result (Lucas–Kanade at 204 points, 1.41×–5.86×, 105 of 105); a row that clears
the noise on a lopsided but not unanimous split is a result whose direction is **not**
established. Collapsing either into the other loses the half that was true.

**No round-count threshold, which is why the sign test's exact p is printed.** Two unanimous
rounds and a hundred unanimous rounds both satisfy "no round crossed", and they are not equally
strong evidence. A minimum round count would be a project-wide "X is enough" bar invented on
the spot. What is printed instead is the exact two-sided sign-test p: 1.0 at one round, 0.5 at
two, 6.1×10⁻⁵ at fifteen, 4.9×10⁻³² at the 105 behind these rows. The **sign count** is
printed beside it, because p answers *how many rounds* and nothing else — see
[the sign count](#the-sign-count-not-p) at the end of this page.

**A tie breaks the direction outcome, and that is the inconvenient choice.** A round whose two
arms time identically favours neither. Three reasons it counts against an outcome that claims
*every* round fell one way: the sentence the outcome licenses is "faster in N of N rounds", and
dropping a tie from N makes that sentence false; ties get **more** common as the clock gets
coarser, so excluding them would make the outcome easier to earn the *worse* the instrument is;
and the tie bucket also holds rounds that were **not measurements at all** — a clock that read
zero — which are counted separately so a row says which kind it tripped on.

The arithmetic is pinned by a unit test: nine rounds at 1.50× and one exact tie. A tie-blind
predicate would call it unanimous and then print "won by **1.00×** to 1.50×", quoting as a win
a round in which nobody won. Remove the tie and the same nine rounds establish a direction at
1.50×–1.50×. Ties did occur in the sweeps behind these reports, all genuine measurements, and
they fell on rows that were mixed-sign anyway. The sign *test* still excludes ties from n,
which is what a sign test does with them.

**The direction outcome is an additional statement about the same rounds.** The medians, the
geometric mean, the factors, the swap-invariance and the noise predicate are unchanged by it;
it does not move a row across the RESULT/NULL line in either direction.

**Two things this is not.** It is not a test for disjoint sample ranges: a range test is vetoed
by a single round that is slow in *both* arms — which is drift, the thing pairing exists to
cancel — and it passes cleanly separated arms whose ratio scatters from 1.01× to 1.60×.
Separation is still computed and printed as a fact beside the outcome. It is also not a p-value
threshold: the sign test is reported at every row and gated on at none.

**Why the deciding quantities are factors and not percentages.** The obvious spelling —
`|median − 1|` against `(max − min) / median`, both as percentages — is not a comparison at
all, because it depends on which arm is the denominator. `|median − 1|` cannot exceed 100% for
the arm that is *faster*, however far ahead it is, while the spread has no ceiling. Measured on
binCV's `erode` on a 5×5 ellipse against `cv::cuda`'s, fifteen paired rounds at 752×480, one
process:

| orientation | difference | spread | outcome |
|---|---|---|---|
| binCV / OpenCV | 93.5% | 267.2% | NULL |
| OpenCV / binCV | 1427.1% | 106.2% | RESULT |

Same rounds, same arms, opposite answers, decided by argument order — one arm is **fifteen
times** the other and one spelling reports "the same speed as far as this run can tell". In
factors both quantities survive the inversion (**15.3× apart against a 4.62× swing**, either
way round). It is not a coincidence of one run: across the seven processes behind this
section the same pair flips outcome with argument order in **four of seven**, and the factor
spelling reads RESULT in all seven.

**The median of an even number of ratios is their multiplicative midpoint**, for the same
reason. An arithmetic midpoint does not commute with inverting the ratio, so at even round
counts the outcome depended on argument order even in factors: two rounds whose ratios are 1.0
and 1.5127 read **1.26× apart one way and 1.20× the other**. Over a sweep of random pairs
every even-count pair disagreed with its own mirror image, four of them all the way to opposite
outcomes; odd counts were already exact. `backends/cuda/tests/test_cuda_bench_stats` pins the
whole of this section — the tie, the swap-invariance and the midpoint — against hand-computed
values, in 280 checks, and needs no GPU.

## The case that changed the rule

**The row was Lucas–Kanade at the tracking pipeline's own keypoint spacing, 204 points.**
Every one of its 105 paired rounds favours binCV, and the two arms were **never once seen
closer than 1.41× apart**. Yet the per-round ratio swings from 1.41× to 5.86× across those
rounds — because the rounds where binCV wins by 5.9× sit further from 1.00× than the rounds
where it wins by 1.41× — and its median of 2.01× does not exceed the 3.80× the larger noise
reads. So the rule, read literally, called *"the two arms are the same speed as far as this
run can tell"* on a pair this machine never once saw level.

**The rule was changed on 2026-09-19: the spread bounds the magnitude, not the direction.**
The row is reported as a range with its sign count:

> **binCV is faster in 105 of 105 paired rounds, by 1.41× to 5.86×** (median 2.01×;
> `cv::cuda` 0.1748 ms against binCV 0.0893 ms, p = 4.9×10⁻³²).

A direction-established row is a range rather than a pass/fail, and the condition an
operation ships under — that it not lose its role comparison badly — is about a kernel that
loses; it says nothing about one that wins in every round. Lucas–Kanade is ahead in every
round and **3.14× smaller** on resident state, so it ships, and
[its row in cuda.md](cuda.md#speed-operation-by-operation) reads as the comparison it is. The
one operation on that backend the condition does bite is
[`cornerSubPixAsync`](cuda.md#what-is-not-delivered), measured losing to the whole-plane round
trip it exists to avoid — a loss in the sense the condition means — so it is optimized or the
gap is accepted explicitly, not excused by a range.

## Worked examples

The rows below are snapshots of [cuda.md](cuda.md)'s tables at the time of writing; cuda.md is
authoritative. 752×480, seven processes, 105 paired rounds a row, kernel-resident clock;
milliseconds, so the smaller cell is the faster side:

| row | `cv::cuda`, ms | binCV, ms | rounds won by binCV | outcome |
|---|---|---|---|---|
| Lucas–Kanade, 204 points (the pipeline's own spacing) | 0.1748 | 0.0893 | 105 of 105 | DIRECTION ESTABLISHED, 1.41×–5.86×; magnitude a null (median 2.01× against a 3.80× noise) |
| `goodFeaturesToTrack`, wall clock | 3.4029 | 0.4240 | 105 of 105 | DIRECTION ESTABLISHED and A RESULT, median 8.76× |
| Lucas–Kanade, 2048 points | 0.3769 | 0.3976 | 91 of 105 | NULL RESULT — fourteen rounds crossed |
| `threshold`, 752×480 | 0.0083 | 0.0079 | 69 of 105 | NULL RESULT — on the launch floor |

The `threshold` row is a null on both halves, and that is not a loss: the operation's written
condition was a *fail* condition — slower than `cv::cuda::threshold` by more than both
spreads — and a null result is not slower. What is not quoted for such a row is a magnitude.

**The launch floor, not the rule, is what turns some internal comparisons on this host into
nulls.** An arm that runs a few multiples above the launch floor (about 0.4 ms a batch here)
swings 2–3× per round inside one process while its per-run medians agree to within a few per
cent, and a few rounds cross; the rule takes the *larger* of the two noises, so the row reads
null and its direction is not established either. That is the rule behaving correctly on a
host whose individual batches are at the mercy of WSL2 scheduling; a batch large enough to
hide the floor would move every number in the report.

## The sign count, not p

**On this host neither half of an outcome is stable at the margin.** Two independent sweeps
of the same rows — one of fourteen processes and one of seven — shared 146 rows, and between
them three rows flipped between RESULT and NULL and seven flipped direction. **Every one sits
within three rounds of the unanimous boundary**, which is what an all-or-nothing criterion
must do there: a row at 1–104 and a row at 0–105 are the same measurement on either side of a
knife edge. The printed p does not warn you: 0–105 is p = 4.9×10⁻³² and 1–104 is
p = 5.2×10⁻³⁰, two orders apart on a scale where both mean "overwhelmingly one-sided". A
reader wanting to know whether a direction row is near its edge has to read the **sign
count**, which is why it is printed beside every outcome and never summarized away.

**What is stable is the rows that are not near the edge.** Fifty rows were direction-established
in both sweeps, and not one had a single minority round in either; the Lucas–Kanade rows
among them had no round cross in 315 paired rounds across 21 processes. That is the
distinction to make when quoting a direction row.

Logs marked stale in [expected-stale.txt](logs/expected-stale.txt) were taken before a
bit-identical change to the code beneath them; the file records which files moved and why
the figure stands.
