# binCV Experiment Log

Running record of every performance and footprint measurement that informed a
design decision.

**Why this file exists:** binCV has two co-equal goals that routinely conflict —
performance and memory footprint. Which one a given design serves is usually not
obvious from reasoning. So the standing method is:

> **Measure the alternatives, weigh the result against the project's goals, then
> decide — and record all three.**

Argument alone does not settle a performance or footprint question. Neither does
a benchmark without a stated decision rule.

---

## The loop

```
ARCHITECTURE.md §9          this file                ARCHITECTURE.md §8
  E-register        ->     experiment      ->      D-record (decision)
  open question            method + result          rationale + what it
  what it would change     + decision               supersedes
```

1. An open question is registered as **E-n** in
   [ARCHITECTURE §9](ARCHITECTURE.md#9-open-questions-and-planned-experiments),
   stating what decision it would change.
2. A task in [TASKS.md](TASKS.md) runs it, logging here as **X-n**.
3. The conclusion is promoted to a **D-record** in
   [ARCHITECTURE §8](ARCHITECTURE.md#8-design-decisions), and the E-entry is
   marked resolved.

A decision made *without* this loop is provisional by definition and must say so.

---

## Rules

**State the decision rule before measuring.** Write down what result would favor
which choice *first*. Deciding afterward invites fitting the conclusion to
whatever the numbers happened to show. If a measurement is surprising, that is a
finding — report it rather than reinterpreting the rule.

**Measure alternatives, not one option.** "X is fast" is not a result. "X versus
Y, on this workload, by this metric" is.

**Use representative workloads.** VIO frontend resolutions and access patterns.
A microbenchmark that no pipeline resembles can favor the wrong design.

**Report memory and speed together.** They trade against each other; a result
that reports one alone cannot be weighed against the project's goals.

**Always Release builds.** See [CLAUDE.md](CLAUDE.md).

**Commit the measurement code.** Every entry here must be reproducible from a
committed benchmark or probe.

**Null and negative results are results.** "No measurable difference" resolves a
question and is worth the same as a large effect — it usually means take the
cheaper or simpler option.

**Verify the benchmark measures something.** A loop whose result is unused gets
deleted by the optimizer, and the resulting number looks spectacular. Consume the
result through a `volatile` sink, vary the input data so it cannot be constant
folded, and sanity-check against a physical bound — if a memory-bound kernel
reports more than DRAM bandwidth, the measurement is wrong, not the kernel. This
rule exists because the first attempt at the platform comparison below was
dead-code eliminated on both platforms and produced 524288 GB/s.

---

## Measurement platforms

Not every platform can answer every question. **State the platform in every entry**,
and do not close an experiment on a platform that cannot authoritatively answer it.

| Platform | Correctness | Algorithmic wins (large effects) | Micro-decisions (E-1, E-2, E-3) |
|---|---|---|---|
| **QEMU / Docker `--platform linux/arm64`** | ✅ authoritative | ❌ | ❌ |
| **Native x86_64** | ✅ | ✅ | ⚠️ indicative only — wrong ISA |
| **Apple Silicon** | ✅ | ✅ real aarch64 + NEON | ⚠️ risky — see below |
| **Raspberry Pi 4 (Cortex-A72)** | ✅ | ✅ | ✅ **authoritative — the reference device** |
| Other Cortex-A (Pi 5, Jetson) | ✅ | ✅ | ✅ authoritative |

**The Pi 4 is the reference measurement device.** Its Cortex-A72 is squarely the
deployment class this library targets — 32 KiB L1D and 1 MiB shared L2, against
roughly 128 KiB L1D and 12 MiB L2 on an M-series core. Cache pressure is visible
there and hidden on a laptop, which is precisely what E-1 and E-2 are asking about.
It also reproduces the popcount situation behind
[D-6](ARCHITECTURE.md#d-6-bulk-only-reductions): ARMv8-A has no scalar popcount, so
`CNT` runs in the NEON domain with the crossings that motivated bulk-only reductions.

**Emulation is for correctness only.** QEMU user-mode does dynamic binary
translation without modelling cache hierarchy, instruction latency, or memory
bandwidth. Measured on this machine, the same popcount loop takes ~15.7 ms native
x86 versus ~54 ms emulated aarch64 — but the slowdown is not uniform across
instruction mixes, so A/B rankings between design variants can invert. Since
E-1 and E-2 are *cache* questions and D-6 rests on *instruction latency*, emulation
cannot answer any of them. It answers correctness perfectly: the core suite passes
261/261 under emulation with identical results.

**Apple Silicon is real aarch64 but not a representative Cortex-A.** M-series
cores have roughly 4× the L1D and far larger L2 than a Cortex-A72, plus much wider
out-of-order execution and more aggressive prefetching. Those differences sit
directly on the variables E-1 and E-2 measure — a padding or word-width penalty
that an M-series core absorbs may be plainly visible on a deployment-class core.
Real ARM numbers, so strictly better than x86 for ARM questions; not a substitute
for the target class.

**Rule:** an experiment gating a shipped default is closed only on a
deployment-class Cortex-A device. Earlier platforms narrow the search and catch
large effects; they do not close the question. Record the platform and mark the
entry `PARTIAL` if it was not the authoritative one.

### Measuring on the Pi 4

Setup instructions: [docs/MEASUREMENT_HARDWARE.md](docs/MEASUREMENT_HARDWARE.md).

A Pi 4 will happily produce stable-looking numbers that are wrong. Four hazards,
all of which the runner script ([T1.10](TASKS.md)) must handle rather than leaving
to whoever is at the keyboard:

**1. The OS must be 64-bit.** `uname -m` must report `aarch64`, not `armv7l`.
Raspberry Pi OS shipped 32-bit by default for years, and on 32-bit ARM every
`uint64_t` operation is synthesised from 32-bit pairs — which would make
[E-2](ARCHITECTURE.md#9-open-questions-and-planned-experiments) measure the
compiler's 64-bit emulation rather than the hardware. The deployment target is
aarch64 (Jetson, modern phones), so a 32-bit result is not merely noisy, it answers
a different question. **The runner script must refuse to run on `armv7l`.**

**2. Thermal throttling.** The BCM2711 throttles around 80 °C, and an uncooled Pi
4 will reach that during a sustained benchmark. Check `vcgencmd get_throttled`
**before and after** every run: a non-zero result means the numbers are invalid,
not merely slower. Discard and re-run with cooling, do not record.

**3. CPU frequency governor.** The default is `ondemand`, scaling 600 MHz–1.5 GHz.
Left alone, a short benchmark measures the governor's ramp behavior. Pin to
`performance` for the run and record which governor was active.

**4. Core migration and background load.** Pin with `taskset` and prefer a Lite
image without a desktop session.

Record all four in the log entry: architecture, throttle state before/after,
governor, and core pinning. An entry without them is not reproducible.

---

## Entry template

```markdown
### X-n · <title> · <TODO | PARTIAL | DONE>

**Gates:** D-n / E-n — what decision this determines
**Question:** one sentence
**Hypothesis:** what is expected, and why
**Decision rule:** *(written before measuring)* if <result> then <choice>
**Variants:** what is being compared
**Workload:** sizes, data, iterations
**Metric:** what is measured, memory and speed
**Method:** how, and where the code lives

**Result:**
<data, as a table>

**Conclusion:** what it means
**Decision:** what changed, and which D-record it produced
```

---

# Completed

### X-1 · Row alignment memory cost · `DONE`

**Gates:** [D-4](ARCHITECTURE.md#d-4-word-granularity-alignment-by-default) ·
[E-1](ARCHITECTURE.md#9-open-questions-and-planned-experiments)
**Question:** What does fixed 32-byte row alignment cost in memory at VIO
resolutions?
**Decision rule:** if overhead exceeds ~10% at any resolution the pipeline
actually uses, alignment must not be the default unless it buys a measured
speedup of comparable size.
**Variants:** `rowAlignment` = 32 bytes versus word granularity (4 bytes for
`uint32_t`).
**Workload:** 640×480 and 752×480 frames; 94×60, i.e. pyramid level 3.
**Metric:** allocated bytes versus the information-theoretic minimum.
**Method:** direct measurement against the existing `BinMat`.

**Result**

| Case | align=32 | align=word | ideal | overhead @32 |
|---|---|---|---|---|
| 640×480 | 46080 B | 38400 B | 38400 B | **+20.0%** |
| 752×480 | 46080 B | 46080 B | 45120 B | +2.1% |
| 94×60 (pyr L3) | 1920 B | 720 B | 705 B | **+172.3%** |

**Conclusion:** Cost is severe and worst exactly where it hurts most — upper
pyramid levels, which LK touches every frame. Overhead is highly
resolution-dependent, so a single fixed alignment is a poor default.

**Decision:** Word granularity becomes the default; larger alignment is opt-in
per object. Recorded as [D-4](ARCHITECTURE.md#d-4-word-granularity-alignment-by-default).

**Why this was `PARTIAL` until T2.8:** only the *cost* side was measured here. The
*benefit* side — whether aligned rows measurably speed up any bulk kernel — was
untested, and there were no bulk kernels to test it on, so D-4 stood
**provisional**. That is exactly the situation this log exists to make visible
rather than let pass as settled, and it is now closed:
[X-9](#x-9--does-row-alignment-earn-its-memory--done) measured the benefit on the
reference device and found none — the best of twelve alignment/kernel/size
combinations was 1.015×, inside its own batch spread. D-4 is confirmed and no
longer provisional.

---

### X-2 · Pyramid bit growth · `DONE`

**Gates:** [§7.2](ARCHITECTURE.md#72-pyramid-downsample--box-22) ·
spawned [E-7](ARCHITECTURE.md#9-open-questions-and-planned-experiments)
**Question:** Does a binary frame stay binary through the reference pyramid?
**Hypothesis:** it does not — a box blur produces intermediate values — but the
rate of growth was unknown.
**Decision rule:** if levels ≥ 1 require more than 1 bit, a binary-only container
is insufficient and `QuantMat<N>` is mandatory rather than speculative.
**Workload:** 256×256 random binary frame, 4 pyramid levels.
**Metric:** distinct pixel values per level, and implied bit depth.
**Method:** `cv::blur(2×2)` then subsample, mirroring the reference
`BOX_2x2` path; distinct values counted per level.

**Result**

| Level | Size | Distinct values | Bits |
|---|---|---|---|
| 0 | 256² | 2 — `{0, 255}` | 1 |
| 1 | 128² | 5 — `{0, 64, 128, 192, 255}` | 3 |
| 2 | 64² | 15 | 4 |
| 3 | 32² | 26 | 5 |

**Conclusion:** Binary survives exactly one level. The reference performs no
re-binarization, so precision grows into the byte it was already paying for.
Three documented claims were wrong and were corrected: the pyramid is ~5× smaller
rather than 8×, the derivative is ternary at level 0 only, and the N-bit container
is required rather than a generalization.

**Decision:** `QuantMat<N>` confirmed mandatory.
[§7.2](ARCHITECTURE.md#72-pyramid-downsample--box-22),
[§7.4](ARCHITECTURE.md#74-spatial-derivative--binarized--1-0-1) and the
[§4.6](ARCHITECTURE.md#46-memory-arithmetic) memory table corrected. Registered
E-7, since binCV *chooses* this quantization and capping it is a direct footprint
lever the reference never had reason to consider.

**Caveat:** subsampling used `cv::resize(INTER_NEAREST)` as a stand-in for the
reference's `PyrDownInvoker` with Gaussian disabled. The growth conclusion is
robust — blur inherently creates intermediate values — but exact value sets may
differ slightly. Worth re-running against the reference path during
[T3.4](TASKS.md).

---

### X-3 · Popcount codegen across targets · `DONE`

**Gates:** [D-6](ARCHITECTURE.md#d-6-bulk-only-reductions)
**Question:** How does `__builtin_popcountll` lower on each target tier?
**Hypothesis:** x86 has a scalar instruction; ARM's situation was unclear.
**Decision rule:** if any primary target lacks a cheap scalar popcount, the public
API must expose only bulk reductions, so the implementation can keep data in
whatever register domain is efficient.
**Variants:** `aarch64`, `armv7m` (Cortex-M4), `x86_64`.
**Metric:** emitted instruction sequence.
**Method:** `clang -O2 --target=<t> -S` on a minimal popcount function.

**Result**

| Target | Emitted |
|---|---|
| aarch64 | `fmov d0,x0` · `cnt v0.8b` · `uaddlv h0,v0.8b` · `fmov w0,s0` |
| Cortex-M4 | ~15-instruction SWAR sequence — no instruction exists |
| x86_64 | `popcntq %rdi,%rax` |

**Conclusion:** aarch64 — the primary target — has **no scalar popcount**. The
cost is dominated by the two GPR↔NEON domain crossings, not by `cnt` itself. A
caller popcounting word-by-word in scalar code pays both crossings per 64 pixels.
Bulk operation keeps data in vector registers and amortizes them away.

**Decision:** No per-word popcount in the public API; reductions are bulk only.
Recorded as [D-6](ARCHITECTURE.md#d-6-bulk-only-reductions).

---

### X-4 · Set-pixel benchmark discontinuity at T1.4 · `DONE`

**Gates:** nothing. This is a **comparability record**, not a decision
experiment — the choice it follows from was already made by
[D-7](ARCHITECTURE.md#d-7-existing-code-is-not-a-constraint) /
[§5.3](ARCHITECTURE.md#53-error-policy), so there is no decision rule written in
advance. It is logged because a published performance ratio changed for a reason
that has nothing to do with performance work, and the log exists to make exactly
that visible.

**Question:** How much of the `BinMat set pixels` number is the bounds check that
T1.4 removed?
**Variants:** `set()` as it is now (`BINCV_ASSERT`, compiled away under NDEBUG)
versus `set()` as it was before T1.4 (an `if` that threw `std::out_of_range`,
live in release).
**Workload:** `set_pixels_benchmark --iterations 20000`, default 640×480, 1000
random coordinates per run; 5 runs of each variant, same machine, back to back.
**Metric:** ms per run, both variants and the OpenCV denominator.
**Method:** the shipped benchmark, built Release against each variant of
`impl/binMat_impl.hpp`.

**Result**

| Variant | ms per run (5 runs) | median |
|---|---|---|
| before T1.4 — checked, throwing | 0.00114 · 0.00134 · 0.00143 · 0.00166 · 0.00167 | 0.00143 |
| after T1.4 — unchecked in release | 0.00086 · 0.00087 · 0.00093 · 0.00117 · 0.00120 | 0.00093 |
| OpenCV `cv::Mat::at` (denominator) | unchanged: 0.00324, 0.00335 | — |

**Conclusion:** Roughly **1.3–1.6× faster**, from deleting a per-pixel branch —
not from any change to the packing or the write. The denominator did not move,
because `cv::Mat::at` was already unchecked under NDEBUG, so the binCV-versus-
OpenCV ratio for this benchmark shifts by that factor on the T1.4 commit alone.

**Consequence for the log:** set-pixel numbers recorded before T1.4 —
`bincv-cpp/results/set_pixels_benchmark.log` — are **not comparable** with
numbers recorded after it. Numbers taken after it are a like-for-like comparison
for the first time: both sides are now unchecked in release.

**Decision:** none. No D-record; D-7 already covers the behaviour change.

---

### X-6 · Is the T2.2 logic speedup real? · `DONE`

**Gates:** the T2.2 performance claim, flagged unconfirmed at commit `22cbe5c`.
**Question:** the committed benchmark reported 8-10x on x86; an independent probe
measured 2.1-2.8x. Which is right, and was the near-constant ns/px across a 64x
size range evidence of a broken measurement?
**Decision rule** *(written before measuring on the Pi)*: if ns/px degrades once
the working set exceeds L2 on the Cortex-A72, the x86 flatness is explained by
that machine's large L3 and is not evidence of a broken benchmark. If it stays
flat on the Pi too, the benchmark is measuring something other than it claims.
**Platform:** Pi 4, Cortex-A72, 32 KiB L1D / 1 MiB shared L2, governor pinned to
performance, taskset -c 3, throttled=0x0 before and after.

**Result** — binCV `bitwiseAnd`, uint32, alone (no OpenCV comparison yet):

| size | working set | ns/px | GB/s |
|---|---|---|---|
| 256x256 | 8 KiB x3 | 0.05064 | 7.41 |
| 512x512 | 32 KiB x3 | 0.02997 | 12.51 |
| 1024x1024 | 128 KiB x3 | 0.03116 | 12.04 |
| 2048x2048 | 512 KiB x3 | **0.06475** | **5.79** |

**Conclusion on the size-invariance concern: RESOLVED, and my suspicion was wrong.**
ns/px degrades 2x at 2048x2048, where the 1.5 MiB working set exceeds the Pi's
1 MiB L2 -- exactly what a bandwidth-bound kernel must do. The x86 flatness is
explained by that machine's 32 MiB L3, which swallows the same working set whole.
Suspecting the benchmark was reasonable; the Pi's smaller cache is what settled it,
which is the entire argument for having a deployment-class reference device.

**Second decision rule** *(written before the ratio was measured, 2026-08-16)*:
the honest headline is whatever the Cortex-A72 reports, whatever it is.
- ratio >= 6x  -> the x86 8-10x figure is broadly corroborated; publish the Pi number
- 2x <= ratio < 6x -> the x86 figure was optimistic; publish the Pi number and
  correct the record. binCV still wins, by less than x86 suggested.
- ratio < 2x  -> report plainly that logic ops are near parity on the target, and
  that the memory ratio rather than throughput is what carries the thesis.
No outcome is a failure. Overclaiming would be.

**Result — Pi 4, Cortex-A72, governor `performance`, `taskset -c 3`,
`throttled=0x0` before and after, gcc 14.2, 640×480, sparsity 0.5:**

| op | binCV uint32 | OpenCV `CV_8U` | ratio |
|---|---|---|---|
| bitwiseAnd | 0.02240 ns/px · 16.74 GB/s | 0.65272 ns/px · 4.60 GB/s | **29.1×** |
| bitwiseOr | 0.02231 ns/px · 16.81 GB/s | 0.66016 ns/px · 4.54 GB/s | **29.6×** |
| bitwiseXor | 0.02369 ns/px · 15.83 GB/s | 0.65907 ns/px · 4.55 GB/s | **27.8×** |
| bitwiseNot | 0.01872 ns/px · 13.35 GB/s | 0.32362 ns/px · 6.18 GB/s | **17.3×** |

Set-pixel counts agree exactly between implementations on every op, so the two are
computing the same thing.

**Conclusion: the ratio on the target device is ~28-30×, well above the x86 8-10×
the decision rule was written to adjudicate — and the mechanism matters more than
the number.**

Working set per call: **binCV 115 KB, OpenCV 922 KB**, against the Pi's **1 MiB
shared L2**. binCV fits; OpenCV does not. OpenCV runs at 4.5-4.6 GB/s, which is
essentially Pi 4 DRAM bandwidth — it is memory-bound. binCV runs at 16.7 GB/s,
above DRAM, because it is being served from cache.

So the speedup is **not** a cleverer inner loop, and should never be described as
one. It is the 8× smaller representation crossing a cache-residency threshold on a
device whose cache is small. That predicts the shape of the whole project:

- **the win grows as the device gets more memory-constrained** — 8-10× on a
  desktop with 32 MiB of L3 that swallows both working sets, ~28-30× on a
  Cortex-A72 where only binCV's fits
- it will shrink again for operations that are not word-parallel, and for images
  small enough that both sides fit in cache anyway

That is the thesis in [ARCHITECTURE §1](ARCHITECTURE.md#the-problem) behaving as
designed, measured on the target class rather than argued.

**Supersedes** the unconfirmed x86 figure flagged at commit `22cbe5c`. The honest
headline is the Pi number, with the mechanism stated alongside it.

**Added later — the reference-device raw log, and a 1.5× disagreement resolved.**
This entry originally committed no Pi log at all (`results/logic_benchmark.log`
holds only the x86_64 run), and its `bitwiseAnd` figure of 0.02240 ns/px sits
about 1.5× away from the 0.0327–0.0330 ns/px that
[X-9](#x-9--does-row-alignment-earn-its-memory--done) and
[X-10](#x-10--default-word-width--done) later measured for the same kernel on the
same device. Re-running this benchmark on the device settles it: **the difference
is the fixture, not the kernel.** `logic_benchmark` holds one image triple
(115.2 KiB resident); the two experiment benchmarks hold nine images per variant
(337.5 KiB) and interleave four variants. The sweep in this very entry predicts
that: 0.02997 ns/px at a 96 KiB working set, 0.03116 at 384 KiB. The corroboration
run reproduces the binCV side to within 5% (0.02350 ns/px) and is committed as
`bincv-cpp/results/logic_benchmark_pi4.log`. It is **not** a re-issue of the
ratios above — the device now carries OpenCV 4.10.0, so the denominator moved
(`bitwiseAnd` 27.95× there against 29.1× here) — but the ~28–30× headline stands.

---

### X-5 · Bandwidth-ceiling probes for the T2.2 logic benchmark · `DONE`

**Gates:** nothing about binCV's design. This is a **method record**: it is the
measurement that decides what the T2.2 benchmark's physical-bound check is allowed
to claim, and it is logged because the first two answers to that question were both
wrong in the published log rather than in the code.

**Question:** What is a usable upper bound on memory throughput at the footprints
`logic_benchmark` measures, so that a row reporting more than the machine can move
is flagged rather than published?

**Decision rule:** *(written before re-measuring)* the bound is whichever candidate
is (a) reproducible within its own run-to-run spread and (b) tight enough that a
kernel measured 2× too fast trips it. A candidate that fails either is not a bound
and must not be described as one.

**Variants:** `std::memcpy` at the footprint; a hand-written one-read-one-write
`uint64_t` copy loop at the footprint.

**Workload:** 38400 B, 131072 B, 307200 B, 1048576 B, 4.19 MB, 33.55 MB — the
buffer sizes the three benchmark sizes produce on both sides. Nine batches per
measurement in the original, twenty-seven after. `taskset -c 3`, three whole runs.

**Metric:** GB/s, best batch *and* worst batch. The spread is the point.

**Method:** the probes shipped inside `bincv-cpp/benchmark/logic_benchmark.cpp`;
raw output in `bincv-cpp/results/logic_benchmark.log`.

**Result**

| footprint | copy loop, best | copy loop, worst | `std::memcpy` |
|---|---|---|---|
| 38400 B | 142.06–143.40 | 55.05–58.41 | 118.5–133.6 |
| 131072 B | 130.29–135.54 | 39.61–82.08 | 8.4–8.7 |
| 307200 B | 108.99–116.89 | 48.32–73.38 | 8.2–8.3 |
| 1048576 B | 105.82–112.55 | 49.83–63.79 | 6.3–7.9 |
| 4.19 MB | 96.48–108.21 | 49.31–60.86 | 7.0–8.6 |
| 33.55 MB | 14.98–18.02 | 6.46–7.66 | 21.0–28.2 |

**Conclusion — two published claims were wrong, in opposite directions.**

1. The copy loop was described in the benchmark header and in the results log as
   "a steady 48–68 GB/s". It is not steady and it is not 48–68: it is 15–143 GB/s
   depending on footprint, with a **1.65× spread at a single footprint across
   three runs** (122.61, 102.20, 74.08 GB/s at 307200 B in the shipped log). A
   4× threshold built on a 100–140 GB/s probe needed 400–560 GB/s to fire, so it
   could only ever have caught total dead-code elimination — a kernel measured
   three times too fast passed it silently. It now uses 1.5×, which the port-count
   argument supports (2 loads + 1 store against 1 load + 1 store), takes three
   times as many batches, and prints its own worst batch so the ceiling's
   instability is visible rather than asserted away.

2. `std::memcpy` was excluded as the bound for a stated reason — "above ~128 KB
   glibc switches to non-temporal stores, so it reports DRAM bandwidth" — and the
   probe's own output refutes it: memcpy comes back **up** to 21–28 GB/s at
   33.55 MB, above its 6–8 GB/s at 128 KB–4 MB. Whatever produces that shape, it
   is not monotonic in size and the non-temporal-store story does not fit. The
   exclusion stands (a primitive that swings 16× across the sizes under test
   cannot bound them), the *explanation* is withdrawn, and the header no longer
   offers one it cannot support.

**Decision:** no D-record — this changes the benchmark's method, not binCV's
design. `logic_benchmark.cpp` now flags at 1.5× the best batch, reports the probe's
spread, and describes memcpy as unexplained rather than explained. The affected
numbers in `results/logic_benchmark.log` and TASKS.md T2.2 were corrected rather
than annotated, because a wrong bound licenses every row beneath it.

---

### X-7 · What `__builtin_popcountll` actually compiles to in binCV's own build · `DONE`

**Gates:** nothing new — this is a **finding against a recorded result**, X-3, and
a measurement-validity record for `results/reduce_benchmark.log`.
**Scope:** x86_64. The `aarch64` half — the tier D-6 is actually derived from — is
[X-7b](#x-7b--the-same-question-on-aarch64-where-d-6-comes-from--done) below, which
was `PARTIAL`'s missing piece; read the two together before quoting either.
**Question:** T2.5 specifies scalar `__builtin_popcountll` "for now". In the
configuration binCV ships, what is that?
**Hypothesis:** X-3 recorded `popcntq %rdi, %rax` for x86_64, so the scalar form
was expected to cost one instruction per word there and to be slow only on the
aarch64 and Cortex-M tiers.
**Decision rule** *(written before measuring)*: if the shipping build's lowering
differs from X-3's, report it and change nothing — enabling an ISA baseline is a
dispatch decision (ROADMAP 2.3) that no experiment has settled, and quietly adding
`-mpopcnt` to make a benchmark look better is exactly the shape of change this
log exists to prevent.
**Variants:** GCC 13 and clang, x86_64, with and without `-mpopcnt`.
**Workload:** `int f(unsigned long long x) { return __builtin_popcountll(x); }`,
`-O2 -S`; then `benchmark/reduce_benchmark.cpp` at four sizes.
**Metric:** emitted instruction sequence, then ns/pixel.

**Result — the codegen**

| build | emitted per word |
|---|---|
| `g++ -O2` (**binCV's default**) | `call __popcountdi2@PLT` — a libgcc call |
| `clang -O2` (binCV's default) | ~15-instruction inline SWAR sequence |
| either, `-mpopcnt` | `popcntq` |
| aarch64 (X-3, unchanged) | `fmov` · `cnt` · `uaddlv` · `fmov` |

**Result — what it costs** (640×480, x86_64, `taskset -c 2`, full table in
`bincv-cpp/results/reduce_benchmark.log`)

| | as shipped | `-mpopcnt` |
|---|---|---|
| `countNonZero` binCV `uint64` | 0.04544 ns/px | 0.00639 ns/px |
| `cv::countNonZero` on `CV_8U` | 0.01098 ns/px | 0.01304 ns/px |
| ratio | **0.24× — binCV 4.2× slower** | **2.04× — binCV 2.0× faster** |
| versus `BinMat::countNonZero()`'s per-pixel loop | 6.1× faster | 35.3× faster |

**Conclusion.** X-3's x86_64 row is right about the ISA and wrong about binCV:
`popcntq` exists, and the library does not compile with it, because
`bincv-cpp/CMakeLists.txt` deliberately detects AVX2/AVX-512 without applying any
`-march` flag until runtime dispatch lands. So on the shipping x86 build the
"scalar popcount" T2.5 asks for is **a function call per word**, and T2.5's own
done-when clause — "expected to be a large win over the per-pixel loop" — holds
(6.1×) while the Tier 1 comparison against OpenCV does not (0.24×).

This does not weaken [D-6](ARCHITECTURE.md#d-6-bulk-only-reductions); it is a
second instance of it. D-6's argument is that the per-word cost is where the time
goes on the targets that matter, and here a *third* target tier turns out to have
a per-word cost that dwarfs `cnt` — an entire PLT call. Because the public API is
bulk (`ops/reduce.hpp` exposes no per-word popcount at all), the fix lands in one
file; had the API exported `popcount(WordType)`, the same call would now be inlined
into every caller's loop and every caller would have to be revisited.

**Decision:** no code change, and specifically **no `-mpopcnt`**. Recorded so the
Phase 5 vectorization task starts from a measured baseline rather than from X-3's
row, and so no one reads `results/reduce_benchmark.log` as an algorithmic result.

---

### X-7b · The same question on `aarch64`, where D-6 comes from · `DONE`

**Gates:** nothing new. This is a **finding against a documented claim** —
ARCHITECTURE §6.2's present tense — and the measured baseline Phase 5 starts from.
Added when the T2.5/T2.6 review pointed out that §6.2 describes an implementation
`ops/reduce.hpp` does not have.

**Question:** X-7 measured the shipping *x86_64* lowering. On the primary target,
does the bulk reduction API — the thing D-6 exists to make possible — actually beat
the per-word popcount loop D-6 forbids exposing?

**Hypothesis:** it should not, yet. T2.5 specifies scalar `__builtin_popcountll`
per word and Phase 5 owns the vectorized form, so the two loops were expected to be
close. §6.2, which is written in the present tense, says otherwise.

**Decision rule** *(written before measuring,*
`bincv-cpp/benchmark/reduce_target_benchmark.cpp`*)*:
- bulk ≥ 1.15× the per-word loop → §6.2's present tense is defensible; correct only
  the instruction sequence it quotes.
- bulk within ±15% of the per-word loop → the **interface** decision (D-6) stands
  and the **implementation** claim does not. Separate the two in §6.2 and in
  `ops/reduce.hpp`, record the numbers, and **change no kernel** — vectorization is
  Phase 5 and this is a documentation defect, not a kernel defect.
- Either way, no `-march` flag and no intrinsics enter the library — the same
  standing decision X-7's x86_64 half already recorded.

**Variants:** the shipped `countNonZero`; a caller-written per-word
`__builtin_popcountll` loop at the same load width; and — **as a headroom probe,
not a candidate** — identical 64-bit loads with the running total kept in a NEON
register (`vcnt_u8` + `vpadal_u8`, one crossing per *row*).
**Workload:** 4096×8 `uint64` (4 KiB, L1-resident on a 32 KiB L1D), so the loop is
measured rather than the memory system. Counts compared before timing.
**Metric:** ns/pixel. **Platform:** the reference device, three runs.

**Result — the codegen** (`g++ 14.2 -O2 -DNDEBUG -S` on the device, reading the
interior loop of each entry point rather than a minimal function)

| kernel | GPR↔NEON crossings per word | interior loop |
|---|---|---|
| `countNonZero` | **1** | `ldr d31,[x2],8` · `cnt` · `addv b31` · `fmov x1,d31` · `add x0,x0,x1` |
| `countAnd` | **2** | the AND is done in GPRs, so the word moves in *and* out |
| `countAndSplit` | **4** | two `fmov` in, two out — and this is the §7.5 covariance path |
| X-3's minimal function | 2 | `fmov d0,x0` · `cnt` · `uaddlv h0` · `fmov w0,s0` |

X-3's row is an accurate ISA illustration and is **not** what these kernels emit:
`countNonZero`'s inbound `fmov` is elided because the word is loaded straight into
a NEON register, and gcc emits `addv b31` where clang emitted `uaddlv h0`. The
accumulator stays in a general-purpose register in all three, which is the point —
the compiler crosses back on every word.

**Result — what it costs**

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| `countNonZero`, shipped bulk API | 0.04109 | 0.03800 | 0.03992 ns/px |
| caller-written per-word loop | 0.04094 | 0.03773 | 0.03970 ns/px |
| **bulk ÷ per-word** | **1.00×** | **0.99×** | **0.99×** |
| vector accumulator (headroom probe) | 0.02333 | 0.02045 | 0.02209 ns/px |
| **headroom for Phase 5** | **1.76×** | **1.86×** | **1.81×** |

**Conclusion.** On the target D-6 was derived from, the bulk entry point is
currently worth **nothing** against the loop shape it exists to prevent — the two
*are* the same loop, because the emitted interior crosses the register-domain
boundary once per word and accumulates in a GPR. So §6.2's sentence "the
implementation keeps data in vector registers and accumulates with `cnt` + `uaddlv`
without crossing back" was false of the shipped code. That sentence has been split
into the interface decision (settled) and the implementation status (scalar today,
1.8× available). `ops/reduce.hpp`'s own D-6 preamble now quotes the sequences its
kernels emit, with the per-entry-point crossing counts, instead of X-3's.

This does not weaken [D-6](ARCHITECTURE.md#d-6-bulk-only-reductions) — it is the
clearest instance of it yet. The 1.8× is reachable by editing one file precisely
because no caller was ever made to write the per-word loop; had `popcount(WordType)`
been public, the 1.8× would be spread across every call site instead.

**Decision:** documents corrected, **no kernel changed**, no intrinsics in the
library. The headroom probe lives in the benchmark, where measurement code belongs.
The width axis was deliberately left alone: everything above is at one 64-bit load
width, so the ratio isolates the crossing rather than mixing in a wider load, and
choosing a load width is Phase 5's job with its own decision rule.

---

### X-8 · What composing the LK covariance out of T2.6 costs · `DONE`

**Gates:** [E-3](ARCHITECTURE.md#9-open-questions-and-planned-experiments) (T2.10),
whose brief this widens, and therefore T3.6.

**Question:** `countAndSplit` is single-pass, as T2.6 requires. But ARCHITECTURE
§7.5's 2×2 covariance needs *three* calls — `countNonZero(mag_x)`,
`countNonZero(mag_y)`, `countAndSplit(...)` — and therefore three traversals of the
same window, issuing the popcounts a single fused traversal would issue once. What
does that composition cost?

**Hypothesis:** the popcount is the expensive operation (D-6), so re-traversal was
expected to be cheap and the composition close to free.

**Decision rule** *(written before measuring)*:
- composition within 15% of a fused traversal → free enough; record and close.
- composition > 15% worse → **widen T2.10's brief** to measure a covariance-shaped
  entry point against the composition *before* T3.6 is written against either, and
  register that in TASKS.md and ARCHITECTURE §9. **Do not add the entry point on
  the strength of this measurement** — choosing T2.6's interface without the
  experiment is exactly what T2.6 forbids for incremental state.
- 15% is T2.10's own existing threshold, adopted rather than invented, so two
  questions about one interface are not judged on two scales.

**Variants:** the composition as shipped, versus one `impl::visitRowWords` pass
producing all four numbers. **Workload:** 640×480 `uint64`, 200 keypoints (the
reference `gftt_max_corners`), 31×31 windows, edge-clipped windows included.
**Metric:** ns/keypoint. **Platform:** the reference device, three runs. Both sides
compared on every window before timing.

**Result**

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| composed, as shipped | 844.6 | 841.0 | 841.1 ns/kp |
| fused, one traversal | 650.9 | 644.8 | 646.1 ns/kp |
| **composed ÷ fused** | **1.30×** | **1.30×** | **1.30×** |

**Conclusion.** 1.30×, reproducibly, with the *same* number of popcounts on both
sides — so the delta is redundant traversal and redundant loads, not extra work per
word. Past the 15% line.

**Decision:** **no interface change now.** T2.10's brief is widened to carry this
second axis (composed-versus-fused) alongside incremental-versus-recompute, and
T3.6 continues to depend on T2.10 settling first. Recorded here rather than acted
on because the alternative — adding a covariance entry point because one benchmark
liked it — is the shape of change this log exists to prevent.

---

# Phase 2 experiments — rules recorded first, then measured

The three entries below were written **before any measurement existed** and
committed on their own (4245210), so the history shows the rules predate the data
(3383996 committed the benchmarks, still before any run). Each **Decision rule** is
copied **verbatim** from its [TASKS.md](TASKS.md) task entry and was not re-scaled,
re-scoped or softened once numbers arrived. All three have since been measured on
the reference device and carry their **Result**, **Conclusion** and **Decision**.

**All three were then RE-MEASURED**, at 0072c1a (X-9, X-10) and 3f32493 (X-11),
after a review found defects in the shared harness rather than in any kernel: a
physical-bound sanity check computed from the wrong footprint, an input rotation
that degenerated when a batch was one call long, and two reporting gaps in X-11's
axes 2 and 3. **Each fix was committed before the run that used it**, the same
discipline the original benchmarks were held to. No ratio moved and no decision
changed; the tables now carry the second set of runs, and each entry names what
was wrong with the first.

Two of them landed where the rule's wording had to do real work, which is the
argument for writing rules first in one sentence: X-10 declined a measured **1.94×**
because the rule is a conjunction and the footprint clause failed, and X-11's data
selected the branch that **rejects the simpler API** it would have been comfortable
to keep.

**Numbering.** X-8 is already taken by the composed-versus-fused finding above and
is cross-referenced from [ARCHITECTURE §9](ARCHITECTURE.md#9-open-questions-and-planned-experiments)
and TASKS.md, so these take **the next free X-numbers**, as T2.10 instructs:
X-9 (E-1/T2.8), X-10 (E-2/T2.9), X-11 (E-3/T2.10). T2.9's older "logged as X-4" is
likewise superseded — X-4 is taken.

**One confound applies to all three, from X-7/X-7b.** binCV builds with no `-march`
flags, so on x86_64 `__builtin_popcountll` lowers to `call __popcountdi2@PLT` — a
library call per word — while on aarch64 it lowers to `fmov`/`cnt`/`uaddlv`/`fmov`.
Any measurement involving a reduction is therefore measuring that lowering as much
as the design variant under test, and an x86 pre-run cannot be extrapolated to the
target. **No `-march` flag is to be added to settle any of these three.** Choosing
a dispatch/baseline-ISA policy is its own decision (ROADMAP 2.3) that no experiment
has settled, and changing it mid-experiment would confound exactly the comparisons
below. Every entry states its platform, and only the reference device closes any of
them.

---

### X-9 · Does row alignment earn its memory? · `DONE`

**Gates:** [D-4](ARCHITECTURE.md#d-4-word-granularity-alignment-by-default) —
**was provisional until this entry closed it**, and was the only such decision in
the project ·
[E-1](ARCHITECTURE.md#9-open-questions-and-planned-experiments) · task
[T2.8](TASKS.md) · completes
[X-1](#x-1--row-alignment-memory-cost--done), which measured the cost side and
not the benefit side.

**Question:** Does row alignment beyond word granularity measurably speed up any
bulk kernel, enough to justify up to 172% memory overhead?

**Hypothesis:** little to no speedup. The bulk kernels stream whole rows word by
word, aarch64 handles unaligned scalar and `LD1` loads without a fault or a
documented penalty on this core, and X-1 already showed the cost is worst at the
upper pyramid levels the frontend touches every frame. If that holds, the
interesting outcome is the null result — which under the rule below closes E-1 and
means no profile system gets built.

**Decision rule** *(written before measuring; verbatim from [T2.8](TASKS.md))*:
- Speedup < 5% on all kernels → D-4 confirmed, close E-1, **do not build a
  profile system**
- 5–20% → D-4 stands as default; larger alignment stays opt-in and is documented
  as worth it for specific kernels
- \> 20% on a kernel the frontend calls per frame → **reopen D-4**, report before
  changing anything

**Variants:** `rowAlignment` ∈ {word granularity, 16, 32, 64} bytes.
**Workload:** `bitwiseAnd` (T2.2) and `countNonZero` (T2.5) at 640×480 and 94×60 —
the two extremes from X-1. Enough iterations for stable timing.
**Metric:** ns/pixel **and** allocated bytes. Both, per the protocol.
**Platform:** closes on the **Pi 4** via `scripts/run_on_pi.sh` (T1.10) — this is a
cache question, which is exactly what a laptop hides. x86 first is fine as a cheap
signal but cannot close E-1; never under emulation. Architecture, throttle state
before **and** after, governor and core pinning get recorded with the result.

**Method:** `bincv-cpp/benchmark/alignment_benchmark.cpp`, committed at 3383996
**before** it was run; raw output and the full reading in
`bincv-cpp/results/alignment_benchmark.log`. Four alignments are built, checked to
agree with the word-granularity baseline on every pixel of every image, and then
timed **interleaved** — one batch of each per round, round-robin — because a 5%
question is the same order as the drift a sequential run would charge to whichever
variant went last. Ratios are on medians of 9 batches; min/median/max is printed
per variant. Reduction timings carry the X-7 popcount-lowering caveat above.

**RE-MEASURED at 0072c1a; the table below is the second set of three runs.** A
review found that the entry's own validity control was computed from the wrong
footprint: the physical-bound check derived the cache tier from the *traffic* of
one `bitwiseAnd` call (three images) and printed it as the working set, reporting
"112.5 KiB — L2-resident" for a batch that actually holds 337.5 KiB inside a
1.58 MiB interleaved round. The shared harness also restarted its input rotation
each batch. Both were fixed and **committed before the re-run**, exactly as the
original benchmark was. Neither touched a ratio; the numbers below reproduce the
first set's conclusion on a harness whose controls now hold.

**Environment** (identical for all three runs; `scripts/run_on_pi.sh pi4`):

```
device:           pi4
cpu:              Raspberry Pi 4 Model B Rev 1.5 (Cortex-A72)
arch / kernel:    aarch64 / 6.18.34+rpt-rpi-v8
compiler:         g++ (Debian 14.2.0-19) 14.2.0, Release (-O2 -DNDEBUG)
governor:         performance (restored to ondemand on exit)
core pinning:     taskset -c 3
throttled before / after: throttled=0x0 / throttled=0x0   (every run)
benchmark commit: 0072c1a
```

**Result** — speedup against word granularity, three runs. **>1.00× means the
wider alignment is faster**, which is the direction the rule reads.

| Size | Alignment | bytes/image | bitwiseAnd | countNonZero |
|---|---|---|---|---|
| 640×480 | 4 (word) | 38400 | 1.000× | 1.000× |
| 640×480 | 16 | 38400 (+0%) | 1.003 / 0.999 / 1.015× | 0.999 / 1.000 / 1.000× |
| 640×480 | 32 | 46080 (+20%) | **0.305 / 0.302 / 0.309×** | 1.000 / 0.998 / 0.997× |
| 640×480 | 64 | 61440 (+60%) | **0.205 / 0.207 / 0.209×** | 0.996 / 0.996 / 0.995× |
| 94×60 | 4 (word) | 720 | 1.000× | 1.000× |
| 94×60 | 16 | 960 (+33%) | 0.996 / 0.985 / 1.000× | 0.990 / 0.981 / 1.000× |
| 94×60 | 32 | 1920 (+167%) | 0.978 / 0.969 / 0.969× | 0.967 / 0.886 / 1.000× |
| 94×60 | 64 | 3840 (+433%) | 0.915 / 0.878 / 0.849× | 0.944 / 0.899 / 0.897× |

**Two noise figures, and a difference has to clear the larger.** The *batch
spread* the benchmark prints — (max−min)/median inside one run — is 5–13% on the
640×480 `bitwiseAnd` rows and ≤1.4% everywhere else. The *run-to-run scatter*
across the three runs above is a different number and is sometimes far larger:
94×60 / align 32 / `countNonZero` has a batch spread of 0.1–0.3% and a scatter of
**11.4%** (0.967 / 0.886 / 1.000×). The first version of this entry quoted only
the batch spread, which understated what a difference must exceed. No conclusion
here turns on it — every cell is a slowdown or a null on either measure — but the
bound is now reported honestly, and both figures are in the log.

**Conclusion:** **No alignment beats word granularity anywhere, on either kernel,
in any run.** The largest number anywhere in the table is 1.015×, against an 8.6%
batch spread on that row and a 1.6% run-to-run scatter — inside the noise on
either measure and inside the rule's first band on both. A null result, not a
small win. The hypothesis held.

Two of the slowdowns are large enough to be worth naming, and both are
consequences of choosing the alignment rather than confounds:

1. `ops/logic.hpp`'s contiguous fast path requires every stride to equal the words
   a row needs, so **over-aligning disables it**: `bitwiseAnd` at 640×480 runs
   3.3× slower at alignment 32 and 4.8× slower at alignment 64, while using 20%
   and 60% more memory. At 94×60 the width is not a whole number of words, no
   variant takes that path, and the cliff does not appear.
2. Padding words are allocated and never read, so a wider stride spends cache
   lines on nothing — the residual 0.85–1.00× at 94×60.

`countNonZero` has no fast path and walks rows unconditionally, so its column is
the clean isolation of alignment alone: **flat to within 0.5% at 640×480 across
all four alignments.** On this core the alignment effect by itself is zero.

**The absolute ns/pixel here is not comparable with [X-6](#x-6--is-the-t22-logic-speedup-real--done)'s,
and the reason is the fixture.** This benchmark holds nine images per variant
(four input pairs plus a destination, 337.5 KiB at 640×480) and interleaves four
variants; `logic_benchmark` holds one triple (115.2 KiB). So `bitwiseAnd` uint32
at 640×480 reads 0.0327–0.0330 ns/px here and 0.0224–0.0235 ns/px there — a 1.5×
gap on the same kernel, same device, same governor. X-6's own working-set sweep
predicts it: 0.02997 ns/px at a 96 KiB working set, 0.03116 at 384 KiB. That was
checked by re-running `logic_benchmark` on the device, which also supplied the
reference-device raw log X-6 never committed —
`bincv-cpp/results/logic_benchmark_pi4.log`. Ratios are unaffected either way,
since every variant here shares one fixture.

One thing this does *not* test, stated so nobody credits it with more than it
did: `BinMat` allocates with `new[]`, whose guarantee here is 16 bytes, so
`rowAlignment` aligns the row **stride** and not the base pointer — the measured
base alignment varies with the allocator, not with the request. So the result is
that making rows mutually congruent buys nothing. Absolutely-64-byte-aligned rows
are not something any binCV API can currently ask for, and nothing in the MVP
asks for them.

**Decision:** The rule's **first band** applies: speedup < 5% on all kernels →
**D-4 confirmed, E-1 closed, and no profile system is built.** D-4 loses the
"provisional" qualifier it has carried since X-1; the project now has no
provisional decisions. Larger alignment stays available per object and is
documented as costing 20–433% of memory to buy nothing measurable — and, for a
dense-strided kernel at a word-multiple width, to cost 3–5× the time as well.
[X-1](#x-1--row-alignment-memory-cost--done) is completed by this entry and is
no longer `PARTIAL`.

---

### X-10 · Default word width · `DONE`

**Gates:** `BinMat`'s default template argument — affects every kernel ·
[E-2](ARCHITECTURE.md#9-open-questions-and-planned-experiments) · task
[T2.9](TASKS.md)

**Question:** Is `uint32_t` the right default, or does `uint64_t` win on bulk
throughput?

**Hypothesis:** `uint64_t` is expected to win something on bulk throughput — half
the loop iterations and, on aarch64, one `cnt`/`uaddlv` round trip per 64 pixels
instead of per 32 — but a row stride rounds up to a whole word, so wider words are
expected to *cost* bytes at the upper pyramid levels while costing nothing at 640
px wide. The two halves of the rule are therefore expected to point in opposite
directions, which is why the tiebreak is written down first.

**Decision rule** *(written before measuring; verbatim from [T2.9](TASKS.md))*:
- `uint64_t` wins by > 10% on bulk kernels **and** does not increase footprint at
  representative widths → change the default
- Within 10%, or footprint increases at small pyramid levels → keep `uint32_t`
  (memory wins ties)

Note the interaction: wider words round row strides up more coarsely, so the
footprint effect is worst exactly at upper pyramid levels. **Measure footprint at
94×60, not only at 640×480**, or this experiment will reach the wrong conclusion.

**Variants:** `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`.
**Workload:** same kernels and sizes as X-9 / T2.8.
**Metric:** ns/pixel and allocated bytes at both resolutions.
**Platform:** the footprint half is architecture-independent and closes anywhere;
**the speed half closes on the Pi 4** (T1.10). This is the most 32-bit-sensitive of
the three — on `armv7l` every `uint64_t` operation is synthesised from 32-bit
pairs, so the result would describe the compiler rather than the hardware.
`aarch64` is confirmed before anything is recorded.

**RE-MEASURED at 0072c1a; the speed table below is the second set of three runs.**
This benchmark carried the same wrong-footprint sanity check as X-9 — it reported
"112.5 KiB, inside its 1 MiB L2" for a batch holding 337.5 KiB inside a 1.32 MiB
interleaved round — and the same batch-restarted input rotation. Both were fixed
and committed before the re-run. No ratio moved. The footprint half is exact
integer arithmetic and was never in question.

**Method:** `bincv-cpp/benchmark/wordwidth_benchmark.cpp`, committed at 3383996
**before** it was run; raw output in `bincv-cpp/results/wordwidth_benchmark.log`.
All four widths hold the same four images, are checked to agree on every count and
every AND pixel, and are timed interleaved round-robin; ratios are on medians of 9
batches. The footprint half is exact integer arithmetic over a pyramid ladder and
needs no device. The width axis interacts with the X-7 lowering directly: on x86
every width pays the same per-word `__popcountdi2` call, which flatters the narrow
words and would invert the ranking. x86 results, if taken, are signal only.

**Environment** (identical for all three runs; `scripts/run_on_pi.sh pi4`):

```
device:           pi4
cpu:              Raspberry Pi 4 Model B Rev 1.5 (Cortex-A72)
arch / kernel:    aarch64 / 6.18.34+rpt-rpi-v8
compiler:         g++ (Debian 14.2.0-19) 14.2.0, Release (-O2 -DNDEBUG)
governor:         performance (restored to ondemand on exit)
core pinning:     taskset -c 3
throttled before / after: throttled=0x0 / throttled=0x0   (every run)
benchmark commit: 0072c1a
```

**Result — speed**, three runs, against `uint32_t`. >1.00× is faster than the
current default. Batch spread is within-run; run-to-run scatter across the three
runs is in the last column, and a difference has to clear the larger of the two.

| Size | Word | bitwiseAnd | countNonZero | batch spread | scatter |
|---|---|---|---|---|---|
| 640×480 | `uint8_t` | 0.968 / 0.990 / 0.996× | 0.247 / 0.247 / 0.247× | 8–16% / 0.1% | 2.8% / 0.0% |
| 640×480 | `uint16_t` | 0.969 / 0.988 / 0.979× | 0.463 / 0.463 / 0.463× | 9–12% / 0.1% | 1.9% / 0.0% |
| 640×480 | `uint64_t` | 0.945 / 0.970 / 0.970× | **1.942 / 1.942 / 1.939×** | 7–12% / 0.4% | 2.5% / 0.3% |
| 94×60 | `uint8_t` | 0.604 / 0.608 / 0.606× | 0.316 / 0.316 / 0.317× | 0.3% / 0.1% | 0.4% / 0.1% |
| 94×60 | `uint16_t` | 0.578 / 0.579 / 0.578× | 0.577 / 0.580 / 0.580× | 0.1% | 0.1% / 0.3% |
| 94×60 | `uint64_t` | **1.285 / 1.284 / 1.271×** | **1.563 / 1.563 / 1.564×** | 0.2% / 0.1% | 1.4% / 0.1% |

The absolute ns/pixel here is not comparable with
[X-6](#x-6--is-the-t22-logic-speedup-real--done)'s for the same reason it is not
in X-9: this fixture holds nine images per width and interleaves four widths, so a
batch streams from L2 rather than L1. The ratios are unaffected — all four widths
share the fixture.

**Result — footprint**, exact, architecture-independent, bytes for one plane at
word granularity:

| Size | `uint8` | `uint16` | `uint32` | `uint64` | u64 vs u32 |
|---|---|---|---|---|---|
| 640×480 (L0) | 38400 | 38400 | 38400 | 38400 | 0.0% |
| 320×240 (L1) | 9600 | 9600 | 9600 | 9600 | 0.0% |
| 160×120 (L2) | 2400 | 2400 | 2400 | 2880 | **+20.0%** |
| 752×480 (L0) | 45120 | 45120 | 46080 | 46080 | 0.0% |
| 188×120 (L2) | 2880 | 2880 | 2880 | 2880 | 0.0% |
| 94×60 (L3) | 720 | 720 | 720 | 960 | **+33.3%** |
| 47×30 (L4) | 180 | 180 | 240 | 240 | 0.0% |

**Conclusion:** **The two halves of the rule point in opposite directions, exactly
as the hypothesis above predicted — and the rule is a conjunction, so the
footprint clause decides.**

`uint64_t` is a large, reproducible win on the reduction: **1.94× at 640×480 and
1.56× at 94×60**, stable to 0.3% across three runs. That clears ">10% on bulk
kernels" outright. `bitwiseAnd` is a null result at 640×480 (0.945–0.970× against
a 7–12% batch spread) because it is memory-bound and all four widths move
identical bytes there; at 94×60, where per-row overhead dominates, `uint64_t`
wins 1.27–1.29×.
Narrow words are much worse on the reduction — `uint8_t` at 0.25× is the per-word
popcount lowering paid eight times as often, which is X-7's finding seen from the
other side.

But `uint64_t` costs **+33.3% at 94×60 and +20.0% at 160×120**, and nothing at
either full frame. The penalty lands exactly where T2.9 warned it would — the
upper pyramid levels LK touches every frame. **Had this been measured only at
640×480, every footprint row would have read 0.0% and the experiment would have
concluded the opposite.** The trap was real, not hypothetical.

**Decision:** **Keep `uint32_t` as the default.** The rule's second clause fires
— footprint increases at small pyramid levels — and memory wins ties. This is not
a tie being broken casually: a measured 1.94× on the reduction is being **declined
on footprint grounds**, which is the project's stated tiebreak doing exactly the
work it exists to do, and it is recorded here so that nobody has to rediscover
that the speed was on the table. Promoted to
[D-14](ARCHITECTURE.md#d-14-uint32_t-is-the-default-word-type).

**What this does not settle, and deliberately does not decide:** the word type is
a per-object template parameter (D-1), so nothing stops a pyramid from using
`uint64_t` at the levels where it costs no bytes (L0, L1, and 188×120) and
`uint32_t` above them. This experiment has now priced both sides of that, but
per-level word width is a *new* decision that no E-entry registers, and picking it
here would be the same mistake this log exists to prevent. Registered as
[E-9](ARCHITECTURE.md#9-open-questions-and-planned-experiments).

---

### X-11 · Incremental versus recomputed window reductions · `DONE`

**Gates:** [T2.6](TASKS.md)'s interface and [T3.6](TASKS.md)'s implementation ·
[E-3](ARCHITECTURE.md#9-open-questions-and-planned-experiments) · task
[T2.10](TASKS.md). T3.6 cannot start until this closes.

**Question:** At what window size does incremental/sliding accumulation beat
recomputation for overlapping windows?

**Hypothesis:** the incremental advantage should grow with window size — at 7×7
recompute is expected to win outright, since the accumulator's bookkeeping costs
more than the few words it saves — and the heavy-overlap 31×31 case is where
incremental has its best chance. Against it: the accumulator is extra resident
state on a 32 KiB L1D, and D-6 exists because the per-word crossing, not the
traversal, is what costs.

**Decision rule** *(written before measuring; verbatim from [T2.10](TASKS.md))*:
- Recompute within 15% of incremental at 31×31 → **keep the simpler recompute
  API**, close E-3, and record that incremental state was rejected on data
- Incremental wins by > 15% at 31×31 → extend T2.6 with incremental state
  *before* T3.6 is written against the simpler form

**Variants:** recompute-per-window versus a sliding accumulator.

**SECOND AXIS — composed versus fused.** Added by the T2.5/T2.6 review and already
measured at **1.30×** on the reference device
([X-8](#x-8--what-composing-the-lk-covariance-out-of-t26-costs--done)), past this
task's own 15% line. Its rule, verbatim from T2.10:
- **Decision rule, same threshold:** a covariance-shaped entry point (returning
  `xx`, `yy`, `whenClear`, `whenSet` from one `visitRowWords` pass) beats the
  composition by > 15% at 31×31 → add it to T2.6 *before* T3.6 is written; within
  15% → keep the composition and record that the fused form was rejected on data.

**THIRD AXIS — selector plane versus four-argument `countAndSplit`.** `countAndSplit`'s
selector `c` must be a frame-sized plane (`sign_x ^ sign_y`, one bit per pixel,
38400 B at 640×480, formed once per pyramid level). A four-argument form taking
`c0` and `c1` and XOR-ing them in the word loop would need no plane at all. T2.10
states no numeric threshold for this axis; it requires that **both memory and
speed** be reported, per CLAUDE.md, since this is precisely a case where the two
goals may disagree. No threshold is invented here — the weighing is against the
project's stated tiebreak (memory wins when the goals conflict and no explicit
choice has been made), and it is recorded as such when the numbers exist.

**Workload:** window sizes 7, 15, 31 at realistic keypoint densities (~200
keypoints, per the reference `gftt_max_corners`); include the heavy-overlap case,
since that is what favors incremental.
**Metric:** ns per window, plus any additional memory the accumulator needs.
**Platform:** closes on the **Pi 4** (T1.10). The tradeoff turns on whether the
accumulator stays resident in a 32 KiB L1D — a laptop with four times the L1 would
favour incremental more than the deployment target does.

**RE-MEASURED at 3f32493; every table below is the second set of three runs.**
Three harness defects were found by review and fixed **before** the re-run, each
committed first. The one that mattered here: `measureInterleaved` restarted its
input rotation every batch, and `calibrate()` returns a batch of **one** when a
single call already exceeds the budget — which DENSE recompute at W=31 does, at
~73 ms per call. So that variant timed image 0 forever while the variant it is
divided by rotated through all four, and the two sides of the 20×/36× ratios were
not on identical inputs. The other two were reporting defects, both stated under
their axes below. No ratio moved on the corrected harness.

**Method:** `bincv-cpp/benchmark/window_benchmark.cpp`, committed at 3383996
**before** it was run; raw output in `bincv-cpp/results/window_benchmark.log`.
Every variant is compared against the shipped kernel on every window of every
image before anything is timed, as X-8 did, and the variants under comparison are
timed interleaved round-robin. Every variant here is popcount-bound except INC-COL,
which issues none — so the X-7 caveat is at its strongest: the ratio between
recompute and incremental *is* a function of what a popcount costs relative to a
load, and that is exactly what the missing `-march` changes. x86 cannot rank these.

**Environment** (identical for all three runs; `scripts/run_on_pi.sh pi4`):

```
device:           pi4
cpu:              Raspberry Pi 4 Model B Rev 1.5 (Cortex-A72)
arch / kernel:    aarch64 / 6.18.34+rpt-rpi-v8
compiler:         g++ (Debian 14.2.0-19) 14.2.0, Release (-O2 -DNDEBUG)
governor:         performance (restored to ondemand on exit)
core pinning:     taskset -c 3
throttled before / after: throttled=0x0 / throttled=0x0   (every run)
benchmark commit: 3f32493
```

**Two accumulators, not one**, because "a sliding accumulator" over bit-packed
rows is not a single design and the two differ in exactly the resource T2.10 asks
about:

- **INC-COL** — the separable box accumulator: per-column running sums slid in x
  and y. Issues **no popcount at all**. Needs a caller-provided array of
  `sweepWidth + W − 1` counters.
- **INC-ROW** — slides vertically with **one scalar**: the window sum gains the
  incoming row's windowed popcount and loses the outgoing row's. Stays
  word-parallel, needs no scratch.

**Three access patterns, all of them in the MVP**, because the rule names a window
size but not a pattern, and T2.10's two requirements ("~200 keypoints" and
"include the heavy-overlap case") are not the same workload:

- **SPARSE** — 200 isolated keypoints (LK covariance, ARCHITECTURE §7.5)
- **SEARCH** — 200 keypoints × an 8×8 sweep = 12800 windows (heavy local overlap)
- **DENSE** — every window position in the frame (corner response, §7.6, which
  §7.6 builds from the same covariance machinery)

**Result — axis 1**, ns/window, uint32_t, 640×480, run 3 (runs 1 and 2 agree to
within 3% on every cell; per-run batch spread was 0.0–3.1%):

| Pattern | W | recompute | INC-COL | INC-ROW | INC-COL × | INC-ROW × | scratch |
|---|---|---|---|---|---|---|---|
| SPARSE | 7 | 67.1 | 248.3 | 68.7 | 0.27× | 0.98× | 28 B |
| SEARCH | 7 | 47.2 | 18.0 | 13.1 | 2.62× | 3.59× | 56 B |
| DENSE | 7 | 44.9 | 6.9 | 8.8 | 6.48× | 5.09× | 2560 B |
| SPARSE | 15 | 135.2 | 975.5 | 117.2 | 0.14× | 1.15× | 60 B |
| SEARCH | 15 | 110.7 | 39.6 | 20.3 | 2.79× | 5.46× | 88 B |
| DENSE | 15 | 106.7 | 7.1 | 10.2 | 15.02× | 10.43× | 2560 B |
| SPARSE | **31** | 306.8 | 3874.8 | 232.5 | 0.08× | **1.32×** | 124 B |
| SEARCH | **31** | 269.7 | 103.8 | 36.8 | 2.60× | **7.33×** | 152 B |
| DENSE | **31** | 267.3 | 7.4 | 13.2 | 36.34× | **20.25×** | 2560 B |

Across runs at W=31: SPARSE INC-ROW 1.32/1.31/1.32×, SEARCH INC-ROW
7.39/7.31/7.33×, DENSE INC-ROW 20.51/20.22/20.25×, DENSE INC-COL
36.35/36.36/36.34×.

**INC-ROW is the form this axis adopts, so INC-ROW's column is the one to quote
for it: 1.32× / 7.3× / 20×.** The 36× belongs to INC-COL, which the decision
below explicitly declines to expose.

**Result — axis 2**, ns/keypoint, composed ÷ fused (ns from run 3):

| Word | W | composed | fused | run 1 | run 2 | run 3 |
|---|---|---|---|---|---|---|
| `uint32_t` | 7 | 197.8 | 156.4 | 1.27× | 1.26× | 1.26× |
| `uint32_t` | 15 | 438.8 | 351.4 | 1.26× | 1.25× | 1.25× |
| `uint32_t` | **31** | 1019.2 | 803.2 | **1.28×** | **1.27×** | **1.27×** |
| `uint64_t` | 7 | 208.9 | 153.3 | 1.38× | 1.38× | 1.36× |
| `uint64_t` | 15 | 399.6 | 309.8 | 1.29× | 1.30× | 1.29× |
| `uint64_t` | **31** | 832.9 | 644.5 | **1.29×** | **1.29×** | **1.29×** |

**Extra memory: 0 B for both forms** — neither needs scratch, and the fused pass
returns its four counters by value. The first version of this entry reported axis
2's speed with no memory figure at all, which the protocol does not allow even
when the answer is zero.

**Result — axis 3**, ns/keypoint, uint32_t, and the memory each form needs (ns
from run 3):

| W | plane (shipped) | four-arg XOR | plane ÷ 4arg |
|---|---|---|---|
| 7 | 110.9 | 149.5 | 0.77 / 0.76 / 0.74× |
| 15 | 252.5 | 327.5 | 0.79 / 0.78 / 0.77× |
| 31 | 584.7 | 733.0 | 0.80 / 0.80 / 0.80× |

Per frame at W=31 and 200 keypoints, **with the plane's formation cost included**:
plane 116.9 µs + 7.2 µs = **124.1 µs**; four-arg **146.6 µs**. The plane is
**16–18%** faster per level across the three runs, even after paying to build it.

Memory: the plane is **38400 B at 640×480, and it scales with the level** — 9600 B
at L1, 2400 B at L2, ~51 kB summed over a 4-level pyramid — held for the frame's
lifetime, against **0 B at every level** for the four-argument form. The first
version of this entry wrote "38400 B per pyramid level", which read literally
overstates the pyramid-wide cost about 3× *in favour of the memory side that won
this axis*. The level-invariant statement is the relative one, and it is correct
at every level: the ternary `dx`/`dy` the covariance reads are 4 planes (153600 B
at 640×480), so the selector is a fifth plane — **+25% on the derivative working
set of every level**.

**Conclusion:**

*Axis 1.* At 31×31 an accumulator beats recompute in **every** pattern — 1.32×
where windows barely overlap, 7.3× and 20× where they do for the form being
adopted (36× for the one that is not). Past the 15% line in all three. Which accumulator wins depends on the pattern and they are not
interchangeable: DENSE favours INC-COL (36×, and its cost per window is constant
in W — 6.9, 7.1 and 7.4 ns at W = 7, 15 and 31 against recompute's 44.9, 106.7
and 267.3, because it issues no popcount), while SPARSE
punishes INC-COL brutally (0.08×: with nothing to slide across it builds W×W bit
reads of state and discards it). INC-ROW never loses (0.98× at its worst), needs
no scratch, and is therefore the safer shape for an API that cannot know the
caller's access pattern.

***The SPARSE row is not an incremental win at all, and this is the surprise of
the experiment.*** At SPARSE the sweep is 1×1, so INC-ROW's sliding path never
executes: it issues **exactly the same popcounts over exactly the same masked
words** as the shipped `countNonZero`, and is still 1.32× faster at W=31, 1.16× at
15, 0.98× at 7. The only difference is where the sum lands.
`impl::countViewRegion` carries **one** `size_t` accumulator across every row and
word of the region, making the whole traversal one serialized dependency chain
through the popcount latency; INC-ROW accumulates per row and adds the rows,
giving the core W independent chains. The gain growing with W is what a
dependency-chain explanation predicts. **That is a finding about the shipped
reduction's codegen, not about incremental state**, and it is available to every
region reduction in `ops/reduce.hpp` with no interface change whatsoever. It is
also a warning about reading axis 1's SPARSE column as evidence for incremental
state: it is not.

**Two consequences follow, and both have to travel with these numbers.** First,
crediting 1.32× to the sliding accumulator *and* crediting the same 1.32× again to
the accumulator split counts one measurement twice — item 1 and item 4 of the
decision below are not independently supported by the SPARSE column, only item 4
is. Second, **every ratio in the axis-1 table is measured against the pre-split
recompute baseline.** Land item 4 first and that baseline gets up to 1.32× faster,
so SEARCH's 7.3× and DENSE's 20× shrink to roughly 5.6× and 15×. Both stay far
past the 15% line, so the branch this rule selects does not change; the magnitudes
quoted for it do.

*Axis 2.* 1.27× (`uint32_t`) and 1.29× (`uint64_t`) at 31×31, in all three runs,
past the 15% line — reproducing X-8's 1.30× in a separate session and extending it
to two word widths and three window sizes, where it holds from 1.25× to 1.39×. The
popcount count is identical on both sides, so the delta is redundant traversal:
three calls visit and load each region word three times.

*Axis 3.* The two goals disagree, which is why T2.10 set no threshold: the plane is
16–18% faster per frame and costs a fifth plane at every pyramid level (+25% of
the derivative working set; 38400 B at 640×480, 9600 B at L1, ~51 kB over four
levels); the four-argument form is that much slower and costs nothing anywhere.

**Decision:**

1. **Axis 1 → the rule's second branch: extend T2.6 with incremental state before
   T3.6 is written.** INC-ROW is the form to expose, at **1.32× / 7.3× / 20×** —
   it wins or ties everywhere,
   is word-parallel, and needs no caller scratch, so it does not drag the
   no-heap-in-kernels rule into the interface. INC-COL's 36× on DENSE is real and
   large, but it is a *second* shape with a scratch-buffer argument and a pattern
   in which it loses 12×, so exposing both is a bigger interface question than
   this experiment's rule answers.
2. **Axis 2 → add a covariance-shaped entry point to T2.6 before T3.6 is
   written**, returning `xx`, `yy`, `whenClear`, `whenSet` from one
   `visitRowWords` pass.
3. **Axis 3 → the four-argument form**, by CLAUDE.md's stated tiebreak: memory
   wins when the goals conflict and no explicit choice has been made. It buys
   38400 B per level — a quarter of the derivative working set — for 16% of one
   operation's time. Recorded rather than enacted, since it lands with 1 and 2 as
   T2.6 API work.
4. **A fourth item that no axis asked for**, from the SPARSE finding: break
   `impl::countViewRegion`'s single accumulator into per-row partial sums. Free,
   no interface change, 1.16–1.32× on the region reduction at LK window sizes.

None of the four is implemented in this entry: T2.10 is the experiment, and
writing the interface it gates in the same commit as the measurement is the exact
inversion this log exists to prevent. They are scheduled as
[T2.11](TASKS.md#t211--t26-api-extensions-mandated-by-e-3--done).

E-3 is **resolved** — with the answer that the simpler API does *not* survive,
which is the opposite of what the first branch would have given and is why the
rule was written down first.

---

#### X-11b · The same three axes re-measured after T2.11 landed them · `DONE`

**Why this section exists.** Everything above was measured against *copies* of the
winning shapes, living in the benchmark, because writing them into
`ops/reduce.hpp` in the same commit as the measurement that gated them is the
inversion this log exists to prevent. [T2.11](TASKS.md) then landed them, and its
done-when requires the axis-1 numbers to be re-measured **after item 4** and the
post-split ratios recorded **next to** the pre-split ones rather than in place of
them. This is that record. Neither set supersedes the other: the tables above
answer *what is the accumulator worth*, these answer *what does it buy in the
shipped library*.

**Method:** the same `bincv-cpp/benchmark/window_benchmark.cpp`, now timing
`bincv::SlidingWindowCount`, `bincv::countCovariance` and the four-argument
`bincv::countAndSplit` instead of its own copies, plus one new variant —
`recompute-1acc`, the recompute path with item 4 undone — timed interleaved with
the rest so that item 4's own effect is measured the same way everything else here
is. Raw output in `bincv-cpp/results/window_benchmark_t211.log`. The pre-split
`window_benchmark.log` keeps its measured data unchanged — not one number in it
was touched; the only edit is a header note at the top pointing at this
re-measurement and saying that neither file supersedes the other.

**Environment** (identical for all three runs; `scripts/run_on_pi.sh pi4`):

```
device:           pi4
cpu:              Raspberry Pi 4 Model B Rev 1.5 (Cortex-A72)
arch / kernel:    aarch64 / 6.18.34+rpt-rpi-v8
compiler:         g++ (Debian 14.2.0-19) 14.2.0, Release (-O2 -DNDEBUG)
governor:         performance (restored to ondemand on exit)
core pinning:     taskset -c 3
throttled before / after: throttled=0x0 / throttled=0x0   (every run)
```

**Axis 1, INC-ROW — the adopted form — pre-split beside post-split**, ×
against recompute, at each window size (post-split figures are run 3; the three
runs agree to within 2%):

| Pattern | W | X-11 pre-split | X-11b post-split | predicted |
|---|---|---|---|---|
| SPARSE | 7 | 0.98× | 1.05× | — |
| SEARCH | 7 | 3.59× | 3.74× | — |
| DENSE | 7 | 5.09× | 4.94× | — |
| SPARSE | 15 | 1.15× | 1.09× | — |
| SEARCH | 15 | 5.46× | 4.60× | — |
| DENSE | 15 | 10.43× | 8.51× | — |
| SPARSE | **31** | **1.32×** | **1.10×** | ~1.0× — **not met**, see the amendment below |
| SEARCH | **31** | **7.33×** | **5.96×** | **~5.6× — met** |
| DENSE | **31** | **20.25×** | **15.92×** | **~15× — met** |

Across runs at W=31: SEARCH 5.96/5.95/5.96×, DENSE 15.81/15.74/15.92×, SPARSE
1.10/1.10/1.10×. Scratch is still **0 B**. The two ratios the decision rests on
land where this entry predicted they would when it wrote down that landing item 4
first would shrink them.

**Axis 2**, composed ÷ fused, both sides carrying the accumulator split:

| Word | W | X-11 | X-11b |
|---|---|---|---|
| `uint32_t` | 7 | 1.26× | 1.37× |
| `uint32_t` | 15 | 1.25× | 1.25× |
| `uint32_t` | **31** | **1.27×** | **1.20×** |
| `uint64_t` | 7 | 1.36× | 1.65× |
| `uint64_t` | 15 | 1.29× | 1.39× |
| `uint64_t` | **31** | **1.29×** | **1.27×** |

Still past the 15% line at every point. `uint32_t` at 31×31 moved 1.27× → 1.20×
because item 4 helps the denominator more than the numerator: the composition
makes two `countNonZero` calls and the fused pass makes none. Extra memory **0 B**
for both, unchanged.

**Axis 3**, plane ÷ four-argument, `uint32_t`:

| W | X-11 | X-11b |
|---|---|---|
| 7 | 0.74× | 0.82× |
| 15 | 0.77× | 0.85× |
| **31** | **0.80×** | **0.85×** |

Per frame at W=31 with formation included: plane 118.2 µs + 7.5 µs = **125.7 µs**,
four-argument **139.1 µs** — the plane is **11–14%** faster across the three runs,
against X-11's 16–18%. Memory is unchanged and is what decides the axis: 38400 B
at 640×480 scaling with the level, ~51 kB over four levels, a fifth plane on top
of the derivative's four (+25% of that working set), against 0 B at every level.
The direction did not move, so CLAUDE.md's tiebreak reaches the same place.

**A MEASUREMENT THAT CONTRADICTS A CLAIM IN THIS ENTRY, reported rather than
quietly corrected.** X-11's decision 4 above, and
[D-15](ARCHITECTURE.md#d-15-window-reductions-get-incremental-state-and-a-fused-covariance),
record the accumulator split as worth **1.15–1.32× at LK window sizes**. Measured
directly and interleaved — `recompute-1acc` ÷ `recompute`, identical popcounts
over identical words, the only difference being where the sum lands — it is worth:

| Pattern | W=7 | W=15 | W=31 |
|---|---|---|---|
| SPARSE | 1.09× | 1.04× | **1.08×** |
| SEARCH | 0.95× | 0.99× | **1.03×** |
| DENSE | 0.94× | 0.98× | **1.04×** |

**1.03–1.09× at the LK window sizes, and a 5–6% loss on the overlapping patterns
at W=7.** The original figure was never measured directly: X-11 *inferred* it from
axis 1's SPARSE column, on the argument that at a 1×1 sweep INC-ROW's sliding path
never executes and so the only thing left to explain 1.32× is the accumulator.
That argument has a gap — INC-ROW differs from a per-window `countNonZero` in
**two** ways, not one: where the sum lands, *and* that it clips its column band
once at construction instead of running the full region clip per position. This
run separates them: item 4 alone is 1.08× at SPARSE W=31, and `SlidingWindowCount`
against the already-split recompute is still 1.10× there, on a path that issues no
incremental update at all. 1.08 × 1.10 = 1.19×, not 1.32×; the remainder is not
accounted for here and is not claimed — `recompute-1acc` is the pre-split
*accumulator* on today's clipping code rather than a byte-exact restoration of
commit 3f32493's kernel, and the two figures come from different sessions.

**Nothing about any decision changes.** Item 4 costs no memory and no interface
and is a gain at both window sizes LK uses; items 1–3 keep the branches their rules
selected, by wide margins. What changes is the number to quote for item 4:
**1.03–1.09×, not 1.15–1.32×**, and the reason the older figure was wrong is that
it attributed a two-variable difference to one variable.

---

### X-12 · T3.1 denoise against the reference implementation · `DONE`

**Gates:** nothing that was open. T3.1's done-when requires a committed benchmark,
and CLAUDE.md requires a decision rule written before the numbers exist — so this
entry records **a measurement that could not have changed the code**, and says so
plainly rather than dressing a confirmation up as an experiment.

**Question:** what does the bit-parallel three-pixel median cost against the
reference implementation on the target device, and does fusing the two neighbour
shifts into the kernel trade memory for speed or win both?

**Decision rule** *(written into `benchmark/denoise_benchmark.cpp`'s header before
the device run, and stated here at full strength):*

- If the **composed** spelling (`shiftDown` + `shiftLeft` + `majority3`, three
  passes and two frame-sized scratch buffers) is **faster** than the fused kernel,
  the fused kernel still ships: it is strictly smaller, and CLAUDE.md's tiebreak
  is that memory wins when the goals conflict and no explicit choice has been
  made. The measurement would then be recorded as **a known speed cost accepted
  for footprint**, with the number attached.
- If the fused kernel is **faster or equal**, nothing is traded and there is no
  decision to record — only a claim to substantiate.

**That first branch is why this is not a free confirmation.** It has a real
consequence (a documented speed cost, quotable against the project), and it was
written before the device had run. What it does *not* have is a branch that
changes the code, and an entry that pretended otherwise would be worse than one
that admits it.

**Variants:** the reference implementation on `CV_8U` in two spellings — *as
written* (all six `cv::Mat::zeros` fills) and the **denominator**, which drops the
four fills the reference immediately overwrites; `bincv::denoiseMedian3` at
`uint32_t` and `uint64_t`; the composed binCV spelling at `uint32_t`.

**Workload:** 640×480 and the pyramid ladder below it (320×240, 160×120, 94×60),
~50% fill, four distinct images rotated through, batches calibrated to a 40 ms
budget with the minimum of five batches reported.

**Metric:** ns/pixel **and** the working set of one call, together (CLAUDE.md),
plus the **measured fixed per-call cost** of each side — see the conclusion for
why the ladder cannot be read without it.

**Method:** `bincv-cpp/benchmark/denoise_benchmark.cpp`. The denominator is
ARCHITECTURE 10.3's, and for this operation it is not a judgement call: it is
`SEAL/src/temporal_processing/denoise.cpp`'s `three_pix_median_filter` **ported
call for call** — two `cv::Mat::zeros` neighbour matrices, the two range-limited
`copyTo` calls, then `cv::min`/`cv::max` in its order — on the same binary content
stored as `CV_8U`. The neighbour construction is timed because that is how the
reference obtains its neighbours.

*What is hoisted, precisely, because an earlier version of this entry said "only
the allocations" and that was not true.* All seven `cv::Mat` **allocations** are
hoisted, as a caller in a frame loop would hoist them — timing `malloc` would
flatter binCV, which allocates nothing here. Of the six **zero-fills**, the
denominator row re-pays only the two the border depends on (`right`'s last column
and `above`'s first row are never written by the `copyTo` calls, so those zeros
*are* the border); the other four buffers are completely overwritten by the
`cv::min`/`cv::max` that follows, so their fill is dead work no caller would keep.
A fifth row, **`OpenCV as-written`**, pays all six and is 1.5×–2.1× slower down
the ladder. Every ratio in this entry is against the *faster* of the two, so it is
conservative by exactly that gap.

All five implementations are compared pixel for pixel before anything is timed,
and a disagreement skips the size and exits non-zero. After the timing, each one
is run once more on the same image and its destination folded **pixel by pixel**
into a representation-independent checksum printed in the table; all five must
match. (The previous version fed one word per iteration to a volatile sink and
called it a checksum of the destination — a kernel that computed only its first
word would have satisfied it.)

**Platform:** Pi 4, Cortex-A72, 32 KiB L1D / 1 MiB shared L2, aarch64, kernel
6.18.34+rpt-rpi-v8, g++ 14.2.0, Release with `-DBINCV_USE_OPENCV=ON`, governor
`performance`, `taskset -c 3`, `throttled=0x0` before **and** after. Raw log:
`bincv-cpp/results/denoise_benchmark_pi4.log`, which records the four source
hashes as well as the commit — `71ef245` is HEAD and contains neither
`ops/denoise.hpp` nor the benchmark, so the commit alone does not pin what ran.

**Result — reference device:**

| size | impl | ns/px | vs denominator | working set |
|---|---|---|---|---|
| 640×480 | reference `CV_8U` *(denominator)* | 3.44739 | 1.00× | 2 150 400 B |
| 640×480 | reference `CV_8U` as written | 5.15740 | 0.67× | 2 150 400 B |
| 640×480 | binCV fused `uint32` | 0.06074 | **56.8×** | 76 800 B |
| 640×480 | binCV fused `uint64` | 0.04739 | **72.8×** | 76 800 B |
| 640×480 | binCV composed `uint32` | 0.21166 | 16.3× | 153 600 B |
| 320×240 | reference `CV_8U` *(denominator)* | 1.46580 | 1.00× | 537 600 B |
| 320×240 | reference `CV_8U` as written | 2.19517 | 0.67× | 537 600 B |
| 320×240 | binCV fused `uint32` | 0.06443 | 22.8× | 19 200 B |
| 320×240 | binCV fused `uint64` | 0.05192 | 28.2× | 19 200 B |
| 320×240 | binCV composed `uint32` | 0.21055 | 7.0× | 38 400 B |
| 160×120 | reference `CV_8U` *(denominator)* | 2.26354 | 1.00× | 134 400 B |
| 160×120 | reference `CV_8U` as written | 3.58778 | 0.63× | 134 400 B |
| 160×120 | binCV fused `uint32` | 0.07389 | 30.6× | 4 800 B |
| 160×120 | binCV fused `uint64` | 0.07070 | 32.0× | 5 760 B |
| 160×120 | binCV composed `uint32` | 0.22940 | 9.9× | 9 600 B |
| 94×60 | reference `CV_8U` *(denominator)* | 3.64058 | 1.00× | 39 480 B |
| 94×60 | reference `CV_8U` as written | 7.19723 | 0.51× | 39 480 B |
| 94×60 | binCV fused `uint32` | 0.12633 | 28.8× | 1 440 B |
| 94×60 | binCV fused `uint64` | 0.08549 | 42.6× | 1 920 B |
| 94×60 | binCV composed `uint32` | 0.38962 | 9.3× | 2 880 B |

**Fixed per-call cost, measured on a 2×2 frame** (the same eight `cv::` calls, on
an image whose pixel work is four bytes): **reference 4.147 µs, binCV 0.014 µs.**
That is 20% of the 94×60 baseline frame, 10% of 160×120, 4% of 320×240 and 0.4%
of 640×480.

**Run-to-run spread on the device**, recorded so a later reader does not over-read
a small difference: an earlier run of the same binary gave 56.4× / 23.1× / 33.5× /
26.5× for fused `uint32` against 56.8× / 22.8× / 30.6× / 28.8× here. The two
largest sizes repeat to within 2%; the two smallest move by up to 10%, which is
what a frame whose denominator is a fifth per-call overhead looks like.

*x86_64 is no longer quoted here.* The previous version of this entry carried a
one-line contrast row with no committed log behind it, and it does not reproduce:
three consecutive runs on an idle desktop give 19.25× / 16.05× / 19.34× at
640×480, a 20% spread on one binary and one input, and an earlier set spread 45%.
The log is now committed anyway (`bincv-cpp/results/denoise_benchmark.log`) so the
claim and its evidence live together; nothing in this entry rests on it.

**Conclusion.**

*On the fuse-versus-compose question, the rule's first branch did not fire.* The
fused kernel is **3.1–3.5× faster** than the composed one at every size (3.49× /
3.27× / 3.11× / 3.08× down the ladder) and uses **half** its memory, so nothing
was traded and no decision needed recording. The gap is what three traversals
versus one predicts, plus two frame-sized writes the fused form never issues. Note
that this settles the question **for this neighbourhood only** — T3.3's morphology
shifts by an arbitrary structuring element, where composing is the only general
spelling, and nothing here says what that costs.

*On the headline ratio, read the working-set column with it.* At 640×480 the
reference implementation holds 2 150 400 B live against the Pi's **1 MiB shared
L2** and binCV's fused call holds 76 800 B, so 57×/73× is substantially a
**cache-residency** result and not a cleverer inner loop — the same mechanism
[X-6](#x-6--is-the-t22-logic-speedup-real--done) identified for the logic kernels,
and it should be quoted the same way. The direct evidence is inside the baseline
rather than across the two sides: the *same OpenCV code* costs 3.45 ns/pixel at
640×480 and 1.47 ns/pixel at 320×240, i.e. it gets 2.4× slower per pixel on the
larger frame, which is the signature of the working set crossing L2. Some of the
8× is also the representation doing less work per pixel rather than moving fewer
bytes: the reference spends four passes on `cv::min`/`cv::max` plus two neighbour
copies where binCV spends one `maj3` per 32 or 64 pixels.

*THE PYRAMID-LEVEL RATIOS ARE NOT A CACHE RESULT, AND THE PREVIOUS VERSION OF
THIS ENTRY SAID THEY WERE.* Below 320×240 the baseline's per-pixel cost **rises**
(1.47 → 2.26 → 3.64 ns/px) while its working set **falls** (537 600 → 134 400 →
39 480 B), all of it comfortably inside L2. No cache explanation predicts that,
and the earlier claim that "the ratio is largest exactly where the reference stops
fitting and smallest at 320×240 where both sides fit" was fitted to a shape it did
not explain. What does explain it, measured rather than argued: the reference
makes **eight `cv::` calls per frame**, whose fixed cost this run measures directly
at **4.147 µs** — 20% of the 94×60 frame and 0.4% of the 640×480 one. Subtracting
even that leaves a per-pixel cost still rising down the ladder (1.41 → 2.05 → 2.91
ns/px), so the remainder is per-frame and per-row overhead in those eight calls
rather than the operation. **Consequence: ratios at different sizes are not
comparable with each other, and the 31×/29× pyramid rows are not a stronger claim
than the 640×480 one — they are a weaker one with more overhead in the
denominator.**

*`uint64_t` beats `uint32_t` at every size in this run* (1.28×, 1.24×, 1.05×,
1.48×) — but the 160×120 margin **changes sign** between two runs of the same
binary, so nothing may be concluded from it. That is consistent with, and does not
re-open,
[X-10](#x-10--default-word-width--done): the default is `uint32_t` on footprint
grounds at upper pyramid levels, and this row is a per-kernel speed observation,
not a word-width experiment. E-9 already holds the per-level version of it.

**Decision:** none. No D-record; nothing was traded, and the one thing that was
measured against a real branch came out the way the code already was. The number
that may now be quoted for T3.1 is **57× at 640×480 on the reference device, with
28× less memory**, never the ratio alone and never a pyramid-level ratio in its
place.

### X-13 · T3.3 morphology against `cv::erode` / `cv::dilate` · `DONE`

**Gates:** nothing that was open, and one thing that was not written down. T3.3's
done-when requires a committed benchmark; CLAUDE.md requires the decision rule to
exist before the numbers do. **The decision rule below was committed before the
reference device ran**, and its live branch is the first one, which fired.

> **Re-measured during T3.3's review, and four of its numbers changed.** The
> border axis, the per-case call floor and OpenCV's real `OPEN` footprint were
> all missing; the erode 3×3 `uint32_t` ratio does not reproduce at the precision
> it was quoted to; and a border-fixup defect the review found made four of the
> five border types 6–10× slower than `cv::erode` while this entry published a
> number for the fifth. Everything below is the post-review measurement. The
> superseded figures are named where they were wrong rather than deleted.

**Question:** what does bit-parallel morphology cost against `cv::erode` /
`cv::dilate` on the same binary content as `CV_8U`, at what footprint — and does
the **fused** kernel that `ops/morphology.hpp` ships (accumulating in the
destination row, no scratch) trade speed for that footprint against the
**composed** spelling the operation is *defined* as (a `shift` per element cell
into a temporary, combined with `ops/logic.hpp`)?

**Hypothesis:** the packed frame is 8× smaller and that is not in doubt. Time is:
OpenCV's 3×3 morphology is separable and vectorised, and binCV's is scalar until
Phase 5, so parity rather than a large win is expected at `uint32_t` and a modest
win at `uint64_t`. The general path (a 5×5 ellipse, non-separable for both sides)
is expected to be **slower** than OpenCV, because 17 shifted folds per word
against a NEON kernel is not a fair fight without vectorisation.

**Decision rule** *(written before the device run):*

1. **Fused versus composed — the branch that can change the code.** If the
   composed spelling is **faster** than the fused kernel, the fused kernel still
   ships (it is strictly smaller: two frames against three, and no caller-owned
   scratch on the MVP's hottest morphological call), and the gap is recorded here
   as **a known speed cost accepted for footprint**, quotable against the
   project. If the fused kernel is **faster or equal**, nothing is traded and
   there is no trade-off to record — only a claim to substantiate.
2. **Against OpenCV.** No branch: T3.3 is Tier 1 and ships whatever the ratio is,
   because bit-exact drop-in compatibility is the deliverable and the 8× footprint
   is the project's goal. What the rule fixes is **what may be said**: a ratio is
   only quotable **with the working set beside it**, at 640×480, and never as a
   pyramid-level number in place of the frame number. If binCV is **slower** on
   any case, that number is recorded here in the same sentence as the footprint —
   not omitted, and not softened.
3. **Cache residency.** If any ratio exceeds ~5×, it is checked against the
   ladder for the shape [X-6](#x-6--is-the-t22-logic-speedup-real--done) and
   [X-12](#x-12--t31-denoise-against-the-reference-implementation--done) found —
   a ratio that grows as the frame shrinks is a fixed per-call cost, and one that
   grows as the frame *grows* is cache residency. Both are stated as such rather
   than reported as the operation's speed.

**Variants:** `cv::erode` / `cv::dilate` / `cv::morphologyEx` on `CV_8U` (the
ARCHITECTURE 10.3 denominator); `ops/morphology.hpp` at `uint32_t` (D-14's
default) and `uint64_t`; the composed `shift` + `bitwise` spelling at `uint32_t`;
and — added during T3.3's review — the general row kernel with the 3×3 special
case refused (`impl::MorphPath::Generic`), which is binCV against binCV and lives
in its own binary.

**Cases:** erode 3×3 rect (the special-cased common path), dilate 3×3 rect (the
same shape with the opposite fold and the opposite border fill, D-12), erode 5×5
ellipse (the general path, non-separable for both sides, **17** set cells), and
`morphologyEx` OPEN 3×3 rect (the compound path, where binCV's one
caller-provided scratch frame appears in the working set). **Plus a BORDER TYPE
AXIS**: erode 3×3 at `BORDER_REPLICATE` and `BORDER_REFLECT_101`, because binCV
handles `BORDER_CONSTANT` in the word path and the other four in a per-pixel edge
fixup, so a ratio measured at one says nothing about the other. `BORDER_WRAP` is
absent: `cv::morphologyEx` refuses it by assertion, so it has no denominator.

**Workload:** 640×480 and the pyramid ladder below it (320×240, 160×120, 94×60),
~50% fill, four distinct images rotated through, batches calibrated to a 40 ms
budget with the minimum of five batches reported.

**Metric:** ns/pixel **and** the working set of one call, together (CLAUDE.md),
plus the fixed per-call cost of both sides **measured per case** — the compound
op's floor is 2.1× the 3×3 erosion's, so one global figure printed beside every
row understates it exactly where the ladder argument needs it.

**Method:** `bincv-cpp/benchmark/morphology_benchmark.cpp` and
`bincv-cpp/benchmark/morphology_path_benchmark.cpp`, on the reference device via
`./scripts/run_on_pi.sh pi4` (the first with `BINCV_PI_OPENCV=1`). Both sides are
compared pixel for pixel before anything is timed, and every row folds its whole
destination into a checksum that must match.

**Environment** (reference device, `./scripts/run_on_pi.sh`):

```
device:           pi4
cpu:              Raspberry Pi 4 Model B Rev 1.5 (Cortex-A72)
arch / kernel:    aarch64 / 6.18.34+rpt-rpi-v8
compiler:         g++ (Debian 14.2.0-19) 14.2.0
governor:         performance (restored to ondemand on exit)
core pinning:     taskset -c 3
throttled before: throttled=0x0
throttled after:  throttled=0x0
```

Full logs: [`bincv-cpp/results/morphology_benchmark_pi4.log`](bincv-cpp/results/morphology_benchmark_pi4.log)
and [`bincv-cpp/results/morphology_path_benchmark_pi4.log`](bincv-cpp/results/morphology_path_benchmark_pi4.log).
Both carry the platform block above including both throttle readings, and the
sha256 of every source that produced them — T3.3's files are untracked at the
time of the run, so the commit hash alone does not pin what was measured (the
convention [X-12](#x-12--t31-denoise-against-the-reference-implementation--done)
established).

**Result — 640×480, `BORDER_CONSTANT`, ns/pixel and the working set of one call,
together:**

| case | impl | ns/pixel | vs OpenCV | working set | vs OpenCV |
|---|---|---:|---:|---:|---:|
| erode 3×3 rect | OpenCV `CV_8U` | 0.71590 | 1.00× | 614400 B | 1.00× |
| | binCV `uint32_t` | 0.72014 | 0.99× | 76800 B | **8.00×** |
| | binCV `uint64_t` | 0.37922 | **1.89×** | 76800 B | **8.00×** |
| | binCV composed `uint32_t` | 1.48731 | 0.48× | 115200 B | 5.33× |
| dilate 3×3 rect | OpenCV `CV_8U` | 0.71457 | 1.00× | 614400 B | 1.00× |
| | binCV `uint32_t` | 0.48186 | **1.48×** | 76800 B | **8.00×** |
| | binCV `uint64_t` | 0.26196 | **2.73×** | 76800 B | **8.00×** |
| | binCV composed `uint32_t` | 1.37363 | 0.52× | 115200 B | 5.33× |
| erode 5×5 ellipse | OpenCV `CV_8U` | 1.85886 | 1.00× | 614400 B | 1.00× |
| | binCV `uint32_t` | 3.59575 | **0.52×** | 76800 B | **8.00×** |
| | binCV `uint64_t` | 2.54880 | **0.73×** | 76800 B | **8.00×** |
| | binCV composed `uint32_t` | 2.88145 | 0.65× | 115200 B | 5.33× |
| `morphologyEx` OPEN 3×3 | OpenCV `CV_8U` | 1.36395 | 1.00× | 614400 B | 1.00× |
| | binCV `uint32_t` | 1.20153 | **1.14×** | 115200 B | **5.33×** |
| | binCV `uint64_t` | 0.64088 | **2.13×** | 115200 B | **5.33×** |

**OPENCV'S WORKING SET FOR `OPEN` IS TWO FRAMES, NOT THREE, AND IT IS MEASURED.**
It is tempting to write "OPEN = erode then dilate, so OpenCV needs a temporary";
`cv::morphologyEx` does `erode(src,dst)` then `dilate(dst,dst)` and allocates
nothing. Probed with `VmHWM` around a single 4096×4096 call, **one op per
process**: OPEN, CLOSE, TOPHAT and BLACKHAT each moved the high-water mark by
0 kB, and only `MORPH_GRADIENT` by ~one frame (17188 kB of a 16384 kB frame). The
compound row's footprint advantage is therefore **5.33×, not the 8.00× this entry
previously recorded** — binCV holds three frames there (src, dst, caller scratch)
against OpenCV's two. Memory is a co-equal goal here, so this is the number most
likely to be quoted and it was wrong by 1.5×.

**The BORDER AXIS, 640×480, erode 3×3 — the four non-constant types are a
different kernel and a different answer:**

| border | OpenCV | binCV `uint32_t` | binCV `uint64_t` |
|---|---:|---:|---:|
| `BORDER_CONSTANT` | 0.71590 | 0.72014 (0.99×) | 0.37922 (1.89×) |
| `BORDER_REPLICATE` | 0.70245 | 0.92953 (**0.76×**) | 0.61351 (1.14×) |
| `BORDER_REFLECT_101` | 0.70882 | 0.94370 (**0.75×**) | 0.63452 (1.12×) |

Under a non-constant border binCV recomputes the `2 × reach` edge columns of each
row one pixel at a time — 2 of 640 for a 3×3 element — and that costs about 30% at
`uint32_t`, which is stated here rather than left to the reader of a
`BORDER_CONSTANT` number. **It cost 12× before T3.3's review.** The fixup walked
every column of every row and `continue`d over the interior, so it paid `width`
iterations to rewrite `2 × reach` pixels: measured at 640×480 `uint64_t` on x86,
19.5 µs under `BORDER_CONSTANT` against 241–260 µs under the other four, which
made binCV **6–10× slower than `cv::erode`** on every border type but the default
while this entry published 1.11× for the default alone. It now visits the two
bands by index and the same four cost 40–45 µs. The lesson is recorded in the
kernel: **the border is a boundary and its cost must scale with the boundary.**

**The ladder, `vs OpenCV` at each size** (`uint32_t`, `BORDER_CONSTANT` except
where named):

| size | erode 3×3 | dilate 3×3 | erode 5×5 ellipse | OPEN 3×3 | erode 3×3 REPLICATE |
|---|---:|---:|---:|---:|---:|
| 640×480 | 0.99× | 1.48× | 0.52× | 1.14× | 0.76× |
| 320×240 | 1.13× | 1.68× | 0.46× | 1.33× | 0.69× |
| 160×120 | 1.67× | 2.28× | 0.52× | 1.94× | 0.76× |
| 94×60 | 2.92× | 3.62× | 0.84× | 3.24× | 1.06× |

Fixed per-call cost, **measured per case** on a 2×2 frame: `cv::erode` 3×3
**2.77 µs**, `cv::erode` 5×5 ellipse **3.36 µs**, `cv::morphologyEx` OPEN 3×3
**5.93 µs**; binCV 0.22 / 1.08 / 0.43 µs. One global figure would have been wrong
by 2.1× on the compound row.

**A NUMBER THIS ENTRY PREVIOUSLY QUOTED DOES NOT REPRODUCE, AND THAT IS THE
FINDING.** The erode 3×3 `uint32_t` row was recorded as **1.11×** from a single
run. Four runs of the current benchmark give 0.72195, 0.73935, 0.72016 and
0.72014 ns/pixel — a tie with `cv::erode`, reproducible to 2% *within one build*.
But the **same library call** timed by `morphology_path_benchmark.cpp` instead
reads **0.66153 ns/pixel at a 0.1% batch spread**, which is the old figure. The
9% sits between two translation units, not between two kernels: adding a single
`MorphPath::Generic` call site to the comparison benchmark moved its erode row by
~10% on x86 as well, which is why that comparison was split into its own binary.
**So the honest statement for this case is that binCV's `uint32_t` 3×3 erosion is
within code-layout noise of `cv::erode` — 0.99×–1.11× depending on the object
file — and 1.11× was quoted to a precision the instrument does not have.** The
`uint64_t` rows, the dilation and the compound op are all well outside that band
and stand.

**What the 3×3 special case is worth** (`morphology_path_benchmark`, binCV against
binCV, the same kernel with `MorphPath::Generic`; batch spreads under 4%):

| size | rect3×3 erode | rect3×3 dilate | cross3×3 erode | cross3×3 dilate |
|---|---:|---:|---:|---:|
| 640×480 `uint32_t` | 2.12× | 3.18× | 2.79× | 3.37× |
| 640×480 `uint64_t` | 2.48× | 3.70× | 3.27× | 3.68× |
| 94×60 `uint32_t` | 2.29× | 2.86× | 2.54× | 2.67× |

T3.3 asks for the special case; this is what it buys, across the whole ladder:
**2.1×–3.7×**. Its docstring used to justify it by load count — "one
`extendedRowWord` per word per element row where the general path pays two per set
cell" — which is false, since `morphRowGeneric`'s window branch hoists the same
call per word for any element reaching less than a word sideways, i.e. every 3×3.
What it actually removes is the inner loop over element cells, the data-dependent
shift count and the per-row span queries. Now measured rather than asserted.

**Conclusion, taking the decision rule's three clauses in order.**

**1. Fused versus composed — the live branch FIRED, on one case out of four.**
On the two 3×3 cases the fused kernel wins on both axes and there is nothing to
trade: 0.720 against 1.487 ns/pixel for erode (**2.07×**) and 0.482 against 1.374
for dilate (**2.85×**), at two frames rather than three. **On the 5×5 ellipse the
composed spelling is FASTER than the shipped kernel** — 2.881 against 3.596
ns/pixel at 640×480 (**1.25×**), and 1.34× at 320×240, 1.35× at 160×120, 1.04× at
94×60. Under the rule written before the run the fused kernel still ships, because
it is strictly smaller (two frames and no caller-owned scratch against three), and
**the gap is recorded here as a known speed cost accepted for footprint: up to
~1.35× on a non-separable element at `uint32_t`.** The cause is legible rather
than mysterious — `ops/shift.hpp`'s row kernel hoists its shift amount out of the
word loop, while the fused kernel's inner loop runs over element cells with a
data-dependent shift count. It is a Phase 5 vectorisation question, not a reason
to hand every 3×3 erosion a third frame.

**2. Against OpenCV, with the footprint beside it, including where binCV loses.**
At 640×480 on the reference device and `BORDER_CONSTANT`, binCV is **a tie with
`cv::erode` at `uint32_t` (0.99×) and 1.89× at `uint64_t`, at 8× less memory** for
the 3×3 rect; **1.48× / 2.73×** for the 3×3 dilation; and **1.14× / 2.13×** for
`morphologyEx` OPEN, at **5.33×** less memory there rather than 8×. **On the 5×5
ellipse binCV is SLOWER — 0.52× at `uint32_t` and 0.73× at `uint64_t`** — at the
same 8× less memory: that is the general path, 17 shifted folds per word, scalar,
against OpenCV's vectorised morphology. **And on the four non-constant border
types binCV is slower again — 0.76× at `uint32_t`** — because the edge band is
recomputed per pixel. binCV's advantage on this operation is the footprint, the
`uint64_t` word width and the dilation; the general path and the non-constant
borders are at a disadvantage until Phase 5.

**3. Cache residency: NO — this ladder is the denominator's fixed cost.** The
largest ratio anywhere is 4.43× (dilate, `uint64_t`, 94×60), and the ladder rises
as the frame **shrinks**, which is the opposite of the shape
[X-6](#x-6--is-the-t22-logic-speedup-real--done) and
[X-12](#x-12--t31-denoise-against-the-reference-implementation--done) found. The
mechanism is measured, not inferred: at 94×60 `cv::erode` pays **2.77 µs per
call** against binCV's **0.22 µs**, which is **18% of an entire 94×60 erode frame
time (15.25 µs)** and 1% of a 640×480 one; for OPEN at 94×60 the same figure is
**19%**. (This entry previously said 10%; that was the 5×5 ellipse row, not the
erode row the argument is about.) binCV's own ns/pixel is nearly flat — 0.720 at
640×480 against 0.926 at 94×60 — so **the number that may be quoted for T3.3 is
the 640×480 one, with its working set, and never the 94×60 ratio.**

**One asymmetry the ladder does not control for, stated rather than argued.**
Hazard 2 rotates four input images, so what is RESIDENT in the timed loop is four
sources plus a destination — 1500 KiB on the OpenCV side against 188 KiB at
`uint32_t` at 640×480 — and only the OpenCV side straddles the Cortex-A72's 1 MiB
L2. The reported working set (614400 B against 76800 B) is one call's, which is
what CLAUDE.md asks for and is not what the loop keeps live. The effect can only
flatter binCV, so every ratio above is an upper bound in that respect; the
benchmark now prints both numbers per size so a reader can bound it.

**Decision:** [D-16](ARCHITECTURE.md#d-16-morphology-fuses-the-shift-and-the-fold-and-only-the-compound-ops-take-scratch)
— `erode` and `dilate` fuse the shift and the fold and take no scratch; the five
compound operations take exactly one caller-provided frame. The ~1.35× cost on
non-separable elements is part of that record, not a footnote to it.


---

### X-14 · Horizontal decimation for the pyramid · `DONE`

**Gates:** [E-8](ARCHITECTURE.md#9-open-questions-and-planned-experiments) —
whether `ops/` gains a resample primitive, and whether it is **word-local** (word
literals only) or **frame-masked** (materialised masks at frame width). Nothing
in `ops/` expresses output bit *j* ← input bit 2*j* today
([§6.1](ARCHITECTURE.md#61-bit-parallel-primitives)), so [T3.4](TASKS.md)'s
`pyrDown` cannot be written until this is answered. Vertical decimation is not in
question: it is a stride-doubled `BinMatConstView` and costs nothing.

**Question:** which route produces output bit *j* from input bit 2*j* — a
per-pixel gather loop, a **word-local** unshuffle (log2(WordBits) mask/shift steps
in registers), or the **frame-masked** big-integer unshuffle E-8 names
(log2(rowBits) passes over the row against materialised frame-width masks and a
scratch row)?

**Hypothesis:** the word-local unshuffle wins on time — it does the same log-depth
gather as the frame-masked route, but each step is a register operation instead of
a pass over memory, and it needs no mask storage at all. If that holds, **the
trade E-8 was registered on does not exist**: E-8 assumes speed (word-parallel
masks) is bought with footprint (frame-sized constants), and a word-local
unshuffle is word-parallel *and* zero-byte. The gather loop is expected to be the
slow one, by roughly the ratio of one branchy bit-test per pixel to one
mask/shift chain per 32 or 64 pixels. All of that is argument, which is precisely
what CLAUDE.md says may not settle it.

**Decision rule** *(written and committed before measuring):*

1. **Frame-masked ships only if it is decisively faster.** It costs a mask table
   and a scratch row per (width, word type) and forces a prepared-plan API that
   T3.4 would have to thread scratch through; the word-local routes cost zero
   bytes and take `(src, dst)`. So it is adopted only if its median beats the
   better word-local variant by **≥ 1.5×, with non-overlapping [min, max] batch
   spreads, at both word types, on the 640×480 → 320×240 case**. Below that bar
   **memory wins** — CLAUDE.md's tiebreak, applied as written rather than
   after seeing which side it favours.
2. **Gather loop versus word-local unshuffle** is decided on speed alone: both
   cost zero auxiliary bytes, so the footprint tiebreak has nothing to weigh.
   Take the faster **if the median difference exceeds the larger of the two
   spreads** at 640×480 at both word types. If it is inside the spread the result
   is null and **the gather loop wins on simplicity** — it is ~10 lines and needs
   no per-word-width mask constants.
3. **The winner becomes the shipped primitive** `decimateColumnsBy2` and is its
   *default*, not a special case. E-8's second half is answered by the winner's
   class. A frame-masked win additionally means `ops/` gains a prepared-plan
   entry point and T3.4 carries caller-provided scratch through `pyrDown`; a
   word-local win means the primitive is `(src, dst)` and the pyramid needs no
   scratch for the subsample half.
4. **Any ratio above 8× is checked against the physical bound below before it is
   believed.** A gather loop that looks 40× slower than a mask chain is plausible;
   an unshuffle that looks 40× faster than DRAM allows is a dead-code measurement.

**Variants:** three, required to produce an identical destination before anything
is timed.

| | Route | Auxiliary memory |
|---|---|---|
| **A** | **Gather loop.** Per destination pixel, read source bit 2*j*, accumulate into a local word, one store per destination word. | none |
| **B** | **Word-local unshuffle.** Per destination word, deinterleave the even bits of source words 2*i* and 2*i*+1 with log2(WordBits) mask/shift steps in registers, then combine the halves. Destination word *i* covers source columns [2*i*·WordBits, 2(*i*+1)·WordBits), so the pairing is exact and no cross-word carry arises. **This is the arm E-8 did not list**, and it is why the hypothesis above doubts the register's framing. | none |
| **C** | **Frame-masked unshuffle.** The row as one big integer: log2(paddedRowBits) passes, each a masked shift-or over the whole row, against a caller-built mask table at frame width and a caller-provided scratch row. | mask table + scratch row |

**Workload:** the pyramid ladder this operation exists for — 640×480 → 320×240,
320×240 → 160×120, 160×120 → 80×60, and 94×60 → 47×30 (odd source width, a
non-word-multiple at every word type). ~50% fill, four distinct random images
rotated through so nothing constant-folds, at `uint32_t`
([D-14](ARCHITECTURE.md#d-14-uint32_t-is-the-default-word-type)) and `uint64_t`.
All three variants read the same stride-doubled source view, so what separates
them is the horizontal half alone.

**Metric:** ns per destination pixel and per call — median of ≥ 7 calibrated
batches with min/max printed beside it, interleaved round-robin — **and** the
auxiliary bytes each variant needs, in the same table. Speed and memory together,
because that is the pair rule 1 weighs.

**Sanity bound:** at 640×480 → 320×240 with `uint32_t` one call touches 19200 B of
source (only even rows are read) plus 9600 B of destination = **28.1 KiB**, and
every smaller ladder step less. That is the **per-call** footprint, and it is what
the bound below is built from; it is *not* a claim that the benchmark runs out of
L1. The harness rotates `kInputs = 4` distinct source frames so nothing
constant-folds, so a batch's resident set is ~163 KB at 640×480 and consecutive
calls of the same variant land in L2, not L1. The useful ceiling is therefore a
range rather than a single number: L1 load throughput on the order of 8–16 B/cycle
at 1.5 GHz = **12–24 GB/s** at the top, L2 below it, and DRAM (~4–6 GB/s) at the
bottom. **A variant reporting more than the highest of those is measuring dead
code** — that is the check this bound exists to enable, and it does not depend on
which level the working set actually sits in.

**Method:** `bincv-cpp/benchmark/decimate_benchmark.cpp` — binCV against binCV, so
[§10.3](ARCHITECTURE.md#103-benchmark-denominator)'s OpenCV denominator does not
apply and it builds in the reference device's default core-only configuration.
Run with `./scripts/run_on_pi.sh pi4 './benchmark/decimate_benchmark'`.
Correctness comes first and is separate: `bincv-cpp/tests/test_resample.cpp`
compares **all three** variants against a per-pixel reference at `uint8_t`,
`uint16_t`, `uint32_t` and `uint64_t` over widths that are and are not word
multiples, including the padding-bit invariant
([D-13](ARCHITECTURE.md#d-13-a-reduction-counts-pixels-never-padding)); the
benchmark additionally re-checks agreement in-binary before timing anything.

**Environment:** Raspberry Pi 4 Model B Rev 1.5 · Cortex-A72 · aarch64,
kernel 6.18.34+rpt-rpi-v8 · g++ (Debian 14.2.0-19) 14.2.0 · Release, no `-march`
(X-7) · governor `performance` · `taskset -c 3` · `throttled=0x0` **before and
after** · commit `f77d8f1`. Correctness on the same device first:
`test_resample` 245312/245312.

**Result** — median ns per destination pixel, 9 interleaved batches, min/max in
the spread column. Auxiliary bytes are the mask table plus the scratch row,
built once per width, which is the most favourable accounting variant C can have.

*640×480 → 320×240, the case rule 1 names:*

| Variant | `uint32_t` | spread | `uint64_t` | spread | aux B |
|---|---|---|---|---|---|
| A gather loop | 4.0188 | 0.1% | 4.8823 | 0.1% | **0** |
| **B word-local unshuffle** | **0.2750** | 0.2% | **0.1847** | 0.2% | **0** |
| C frame-masked | 3.0938 | 1.0% | 1.5416 | 0.1% | 1408 |
| A ÷ B | **14.61×** | | **26.43×** | | |
| C ÷ B | **11.25×** | | **8.35×** | | |

*The whole pyramid ladder, medians, `uint32_t` / `uint64_t`:*

| Source | A gather | B unshuffle | C frame-masked | aux B (C) |
|---|---|---|---|---|
| 640×480 | 4.0188 / 4.8823 | 0.2750 / 0.1847 | 3.0938 / 1.5416 | 1408 |
| 320×240 | 4.0051 / 4.9045 | 0.2681 / 0.1704 | 2.7273 / 1.6396 | 640 |
| 160×120 | 4.0724 / 4.8959 | 0.2742 / 0.2200 | 2.7981 / 1.8072 | 288 |
| 94×60 | 4.0631 / 4.7239 | 0.3029 / 0.2643 | 2.6156 / 1.7650 | 128 |

Every batch spread was ≤ 3.1% and all but two ≤ 1.2%. **Run-to-run scatter was
measured rather than assumed**: the whole benchmark was run twice on the device,
and the headline medians reproduced to 0.1% (0.2750 → 0.2750, 4.0188 → 4.0187,
3.0938 → 3.0972). Both bounds are three orders of magnitude below the smallest
gap the rule weighs.

**A third run confirms the numbers belong to the SHIPPED code, not to a
candidate.** Runs 1 and 2 were taken at commit `f77d8f1`, where all three arms
were `impl::` candidates; run 3 was taken at `f3235b5`, after the winner became
the public `decimateColumnsBy2()` and the benchmark was re-pointed at it —
0.2753 / 0.1842 against 0.2750 / 0.1847, inside the batch spread. This is X-11b's
lesson applied without waiting to be bitten by it: a benchmark that times a copy
of the winner stops describing the library the moment the copy drifts. All three
runs are in `bincv-cpp/results/decimate_benchmark_pi4.log`.

**The sanity bound holds (rule 4).** Every ratio here is above 8×, so all of them
were checked. The winner moves 28800 B per call in 0.2750 ns/pixel × 76800
pixels = 21.1 µs, i.e. **1.36 GB/s at `uint32_t` and 2.03 GB/s at `uint64_t`** —
an order of magnitude below **every** ceiling the bound listed, the ~4–6 GB/s DRAM
floor included, so it does not matter that the four-frame rotation puts the batch
in L2 rather than L1: no variant is fast enough for the question "was this deleted
by the optimizer?" to be live. The second check
is per-word cost: 0.1847 ns/pixel × 64 pixels is **11.8 ns = ~17.7 cycles at
1.5 GHz** for one destination word, against roughly 28 ALU operations (two
six-step deinterleaves plus the combine, load, store) — about 1.6 ops/cycle on a
3-wide core. Fast, and not impossibly fast.

**Conclusion:** **E-8's premise was wrong, and that is the result.** The register
framed horizontal decimation as speed bought with footprint — "a per-pixel gather
loop, or a log2(width) word-parallel unshuffle that needs frame-sized constant
masks" — so the interesting question was where the project's tiebreak fell. There
was no tiebreak to apply: the word-local unshuffle is word-parallel **and**
zero-byte, and it beat both alternatives by 8.3× to 26.4× on the device. Rule 1's
bar (frame-masked must be ≥ 1.5× *faster*) was missed in the opposite direction
by an order of magnitude, and rule 2 was decided far outside both the batch
spread and the run-to-run scatter.

Two things are worth recording beyond the branch that fired:

1. **The frame-masked route loses for a structural reason, not a tuning one.** It
   performs the same log-depth gather as B, but each step is a pass over memory
   instead of a register operation, and its recurrence needs the row padded to a
   power-of-two bit count. **Stated at one word type**, because an earlier version
   of this sentence mixed two and produced a ratio neither one gives: at
   `uint64_t`, 640 columns is 10 words padded to 16, so C runs **10 passes over 16
   words** where B makes **one pass over 5** destination words. At `uint32_t` the
   same counts are 20 → 32, 10 passes over 32, against one pass over 10. Either
   way the word-count ratio is 32×, against 8.35× measured — the gap is B's
   per-word cost, which is a six-step deinterleave rather than a single move, so
   the count alone was never going to predict the ratio and no predicted figure is
   claimed here. What the counts do establish is the direction, and it is
   structural: more passes over more words. No amount of tuning closes that; it is
   more work.
2. **A word-width result that belongs to [E-9](ARCHITECTURE.md#9-open-questions-and-planned-experiments), not to this decision.**
   The winner is **1.49× faster at `uint64_t`** (0.1847 against 0.2750) because
   its fixed per-word cost amortises over 64 pixels instead of 32 — while the
   gather loop is **1.21× SLOWER** at `uint64_t`, having no per-word cost to
   amortise. That is the shape E-9 asks about and this entry does not settle it:
   [D-14](ARCHITECTURE.md#d-14-uint32_t-is-the-default-word-type) stands, and one
   kernel's preference is not a default.

**Decision:** [D-17](ARCHITECTURE.md#d-17-horizontal-decimation-is-word-local).
`ops/resample.hpp` ships `decimateColumnsBy2()` — the word-local unshuffle,
taking `(src, dst)` and nothing else. `ops/` gains a resample row in
[§6.1](ARCHITECTURE.md#61-bit-parallel-primitives)'s table, T3.4's `pyrDown`
needs no scratch and no prepared plan for the subsample half, and E-8 is
resolved. The two losing arms stay in `impl::` so the experiment can be re-run:
`tests/test_resample.cpp` checks all three against one per-pixel reference, and
`benchmark/decimate_benchmark.cpp` times the **shipped** function against them
rather than a copy of it.

---

### X-15 · Pyramid bit growth and footprint, against the reference path · `DONE`

**Gates:** nothing on its own — it is the measurement half of
[T3.4](TASKS.md)'s done-when clauses, and the data
[E-7](ARCHITECTURE.md#9-open-questions-and-planned-experiments) (T4.1) will weigh.
It also **completes [X-2](#x-2--pyramid-bit-growth--done)**, whose caveat asked for
a re-run against the reference's actual `PyrDownInvoker` path rather than against
`cv::resize(INTER_NEAREST)` as a stand-in.

**No decision rule, and that is not an omission.** There is no choice being made
here: the bit depth per level is a *parameter* of `pyrDown`, and E-7 is the entry
that decides it, in Phase 4, against tracking accuracy. Parameterizing a contested
choice is what buys the right to defer measuring it
([ARCHITECTURE §9](ARCHITECTURE.md#9-open-questions-and-planned-experiments)).
What T3.4 owes is the numbers E-7 will need and the evidence for the cost claim
its second blocking gap turned on.

**Question:** three of them. (a) How many bits does each pyramid level actually
need, and how many does a frame contain? (b) What does a four-level pyramid cost
in bytes at several `NOut` caps, against the `CV_8U` pyramid a user has today?
(c) Is the shipped 2×2 box sum really linear in `NIn` where the replication route
is exponential?

**Method:** `bincv-cpp/benchmark/pyramid_benchmark.cpp` (core-only, no OpenCV, so
it builds in the reference device's default configuration) and the OpenCV half of
`bincv-cpp/tests/test_pyramid.cpp`. Growth and footprint are
architecture-independent and close anywhere; the timing half was closed **on the
reference device** (`./scripts/run_on_pi.sh pi4 './benchmark/pyramid_benchmark'`),
with the x86 pre-run kept beside it as a non-authoritative cross-check.

---

**Result (a) — THREE FINDINGS, and two of them correct a documented claim.**

**1. `cv::blur` on `CV_8U` does not round the mean to nearest. It rounds it UP.**
Measured over 59 940 operand quadruples, its 2×2 box is exactly
`ceil((a + b + c + d) / 4)`. That is where [§7.2](ARCHITECTURE.md#72-pyramid-downsample--box-22)'s
level-1 value `192` comes from: the exact mean of `{0, 255, 255, 255}` is 191.25,
and rounded to *nearest* it would be 191. X-2's table was right and the obvious
reading of it — "the reference rounds the mean" — was not.
`tests/test_pyramid.cpp` now checks OpenCV's rule as well as binCV's, so a change
in either fails rather than being absorbed. binCV rounds once, half up: rounding a
mean up is a systematic brightening of every level, and Tier 2 is what buys the
right to differ.

**2. X-2's 1/3/4/5 is a FRAME STATISTIC, not the precision the operation needs.**
X-2 counted distinct values in a 256×256 frame; its level 3 is 32×32, i.e. 1024
pixels. The alphabet the arithmetic can *reach* is larger, and it does not shrink
with the frame:

| Level | reachable, reference rule (`ceil`) | reachable, binCV rule (half up) | in a 640×480-derived frame (binCV, `1-8-8-8`) | X-2, 256² frame |
|---|---|---|---|---|
| 0 | 2 — 1 bit | 2 — 1 bit | 2 | 2 |
| 1 | 5 — 3 bits | 5 — 3 bits | 5 | 5 |
| 2 | 17 — 5 bits | 21 — 5 bits | 20 | 15 |
| 3 | 65 — 7 bits | 95 — 7 bits | 34 | 26 |

So an *uncapped* box mean adds exactly two bits per level — 1, 3, 5, 7 — which is
what a 4-input sum of `NIn`-bit values must, and the reference's "1/3/4/5" is what
one small frame happened to contain. [§7.2](ARCHITECTURE.md#72-pyramid-downsample--box-22)
is corrected: the table now says which column is which. **The conclusion X-2 drew
is unchanged and strengthened** — binary survives exactly one level and
`QuantMat<N>` is mandatory — but E-7's lever is larger than it looked, because the
uncapped ladder wants 7 bits at level 3 rather than 5.

**3. The reference's 2×2 window is half a pixel up and to the left of the aligned
block.** `cv::blur(src, dst, cv::Size(2, 2))` takes OpenCV's *default* anchor,
which for an even kernel size is (1, 1), so the reference's output at (y, x)
averages source rows 2y−1…2y and columns 2x−1…2x. binCV uses the aligned,
non-overlapping block, whose centre maps to source coordinate 2·(y + ½) with no
offset. Pinned in `tests/test_pyramid.cpp` OpenCV-against-OpenCV, so that only the
anchor differs between the two sides of the check.

---

**Result (b) — bit growth and peak footprint, 640×480, four levels, `uint32_t`.**

Every level coexists, because a tracker reads all of them, so this is a peak
working set and not a per-buffer ratio. The `CV_8U` denominator is exact
arithmetic — one byte per pixel per level — and is **408 000 bytes**.

| ladder (`NOut` caps) | distinct in the frame | reachable | bits needed | bytes | vs `CV_8U` |
|---|---|---|---|---|---|
| 1-3-5-7 uncapped | 2/5/24/50 | 2/5/26/121 | 1/3/5/7 | 84 240 | **4.84×** |
| 1-3-4-5 reference-shaped | 2/5/14/15 | 2/5/16/32 | 1/3/4/5 | 80 400 | **5.07×** |
| 1-3-3-3 | 2/5/7/4 | 2/5/8/8 | 1/3/3/3 | 76 560 | 5.33× |
| 1-2-2-2 | 2/4/4/3 | 2/4/4/4 | 1/2/2/2 | 63 840 | 6.39× |
| 1-1-1-1 re-binarized | 2/2/2/2 | 2/2/2/2 | 1/1/1/1 | 51 120 | 7.98× |
| 1-8-8-8 `CV_8U`-shaped | 2/5/20/34 | 2/5/21/95 | 1/3/5/7 | 140 160 | 2.91× |

**The cap is worth less than it looks, and that is the useful part.** Level 0 is
38 400 of those bytes and no cap touches it, so the whole range from "keep every
bit the box produces" to "re-binarize every level" spans 84 240 → 51 120 bytes —
**1.65×**, against the 4.84×–7.98× the pyramid already wins over `CV_8U`. E-7 is
therefore trading a 1.65× footprint band against tracking accuracy, not an order
of magnitude, and it should be run knowing that.

---

**Result (c) — the box sum, linear against exponential.** 640×480 → 320×240,
`uint32_t`, `NOut = NIn + 1` throughout so the requantizer is the same shape in
every row and the pair differs only in the sum. **Reference device**, spreads
≤ 1.1%; `bincv-cpp/results/pyramid_benchmark_pi4.log`.

```
device: pi4   Raspberry Pi 4 Model B Rev 1.5
arch:   aarch64 / 6.18.34+rpt-rpi-v8      compiler: g++ (Debian 14.2.0-19) 14.2.0
governor: performance (restored)          pinning: taskset -c 3
throttled before: 0x0                     throttled after: 0x0
commit: 7bbe65d
```

| `NIn` | linear adder (ns/dst px) | spread | replicated (ns/dst px) | spread | ratio | stages `3·NIn+1` | inputs `4·(2^NIn−1)` |
|---|---|---|---|---|---|---|---|
| 1 | 1.1301 | 0.1% | 2.3462 | 0.5% | **2.08×** | 4 | 4 |
| 2 | 3.0387 | 0.1% | 6.4503 | 0.2% | **2.12×** | 7 | 12 |
| 3 | 4.3974 | 0.1% | 14.0389 | 1.1% | **3.19×** | 10 | 28 |
| 4 | 9.0190 | 0.2% | 31.2323 | 0.2% | **3.46×** | 13 | 60 |

The ratio widens monotonically, which is the shape the two formulas predict. The
x86 pre-run agreed on the shape and disagreed on the level (1.39× / 2.12× / 3.10×
/ 3.51×, spreads up to 97% on a loaded desktop) — recorded in
`bincv-cpp/results/pyramid_benchmark.log` and non-authoritative, as
[X-7](#x-7--what-__builtin_popcountll-actually-compiles-to-in-bincvs-own-build--done)'s
caveat requires. The replicated arm **refuses to compile above `NIn = 5`**, where
its per-destination-word input array is already 124 words; at `NIn = 8` it would
be 1020 against the shipped route's 25 adder stages. Both routes are checked to
agree pixel for pixel before either is timed.

**One thing this does NOT show, and it is worth saying.** "Linear in NIn" is a
statement about the **operation count**, and it is exact: the loops run to
`NIn + 2`. The measured wall time of the shipped route grows faster than that —
8.0× from NIn = 1 to NIn = 4 where the stage count
(`3·NIn + 1` plus `(NOut+2)(NIn+NOut+2)`, i.e. 24 → 90) predicts 3.75×. The likely
cause is register pressure rather than arithmetic, **and the live-word count this
paragraph first quoted was itself too low** — it counted the four phase arrays and
`scaled` and missed `boxSum4`'s two partial sums and `value`. The corrected
inventory is `impl::pyrDownAutomaticWords(NIn, NOut) = 8·NIn + 2·NOut + 6` words
per destination word: **18 at NIn = 1, NOut = 2 and 48 at NIn = 4, NOut = 5**,
against a Cortex-A72's 31 general registers. So the register-pressure explanation
is *stronger* than it read, not weaker: the kernel is already over the register
file at NIn = 1 by this count and is 1.5× over it at NIn = 4. That is a **tuning**
observation, not an algorithmic one — the comparison T3.4 turned on is against
`4·(2^NIn − 1)`, which grows 13.3× over the same range — but it is the number to
look at first if a pyramid level ever needs to be faster, and it is left here
rather than smoothed away.

**The kernel's automatic storage, measured rather than asserted.** The same
correction applies to what `ops/pyramid.hpp` promised a caller. Its header said
"the whole arithmetic runs in NIn + NOut + 2 words of automatic storage"; that is
the widest *single* intermediate (`scaled`), not the total, and it understated the
emitted frame by 5×–10×. Measured with
`g++ -std=c++17 -O2 -DNDEBUG -fstack-usage` on non-inlined instantiations of
`impl::pyrDownRoute<NOut, NIn, W, false>`:

| NIn / NOut | 1 / 3 | 3 / 4 | 4 / 5 | 8 / 8 |
|---|---|---|---|---|
| **aarch64**, `uint32_t` | 224 B | 416 B | 448 B | 640 B |
| **aarch64**, `uint64_t` | **272 B** | **544 B** | **592 B** | **912 B** |
| x86_64, `uint32_t` | 288 B | 480 B | 544 B | 720 B |
| x86_64, `uint64_t` | 368 B | 640 B | 704 B | 1040 B |
| *claimed*, `uint64_t` | *48 B* | *72 B* | *88 B* | *144 B* |
| declared words, `uint64_t` | 160 B | 304 B | 384 B | 688 B |

The aarch64 row is the authoritative one — it was taken on the reference device
itself (g++ 14.2.0, environment block above), and it is smaller than x86_64's
because a Cortex-A72 has twice the registers to spill into. The old sentence was
low by **5.7×–7.7×** across the whole table, and low against the *declared* word
count too. The frame does not depend on image size.

**The code was not changed to fit the number.** Two restructurings that would have
made the source-level count tight — grouping the nine arrays into one struct, and
hoisting the helpers' locals into the caller — were both tried and both measured
**larger** (784 B and 816 B against 704 B at NIn = 4 / NOut = 5 / `uint64_t`,
x86_64), because the compiler can no longer overlap the lifetimes that do not
meet. Memory wins ([CLAUDE.md](CLAUDE.md)), so the kernel kept its shape and the
documentation was corrected instead — which is also what
"[a measurement contradicts a documented claim] → report it rather than adjusting
the code to fit the doc" asks for. The heap half of the promise was
true and is now checked rather than asserted: `tests/test_pyramid.cpp`
(`Pyramid.FootprintClaims`) counts `operator new` across `pyrDown` and
`Pyramid::build` and requires zero.

**Correctness on the reference device.** `tests/test_pyramid.cpp` reports
**236429/236429** under `aarch64` on the same device, byte-identical to the three
core x86 configurations — so nothing in the bit-sliced arithmetic, the word-local
gather or the tail masking depends on the host architecture.

**Conclusion:** the numbers E-7 needs exist, X-2's caveat is discharged, and two
of its readings are corrected. The cost claim T3.4's second blocking gap turned on
holds: linear in `NIn` for the sum, quadratic in `NOut` and linear in `NIn` for
the requantization, exponential in neither.

**Decision:** promoted to
[D-18](ARCHITECTURE.md#d-18-the-n-bit-box-is-a-multi-bit-adder-and-the-requantization-is-a-documented-rescale).
E-7 stays open and stays in Phase 4; it now has its footprint axis measured and
knows the band is 1.65× rather than an order of magnitude.

---

### X-16 · T3.5 derivative against `cv::filter2D` · `DONE`

**Gates:** nothing that was open. T3.5's done-when requires a committed benchmark
against `cv::filter2D` with the same kernel, and CLAUDE.md requires the decision
rule to exist before the numbers do. As with
[X-12](#x-12--t31-denoise-against-the-reference-implementation--done), this entry
records a measurement that **cannot change the shipped kernel**, and says so
rather than dressing a confirmation up as an experiment. It has two live branches
all the same, and both are stated at full strength below.

**Question:** what does the bit-parallel `[-1, 0, 1]` derivative cost against
`cv::filter2D` on the reference device; how much of any ratio is cache residency
rather than arithmetic; and is the N-bit path's cost linear in N as
`derivativeAdderStages(N) == 2N` claims?

**Decision rule** *(written into `benchmark/derivative_benchmark.cpp`'s header and
into this entry before the device ran):*

1. **Fused against composed.** If the composed spelling — two `ops/shift.hpp`
   calls plus `ops/logic.hpp` per axis, four passes and **two frame-sized scratch
   buffers** — is **faster** than the fused kernel, the fused kernel still ships:
   it is strictly smaller (5 bit-planes of working set against 7, 1.40×) and
   CLAUDE.md's tiebreak is that memory wins when no explicit choice has been made.
   The result would then be recorded as **a known speed cost accepted for
   footprint**, with the number attached, exactly as
   [D-16](ARCHITECTURE.md#d-16-morphology-fuses-the-shift-and-the-fold-and-only-the-compound-ops-take-scratch)
   records morphology's. If the fused kernel is faster or equal, nothing is traded.
2. **Is the headline a cache result?** binCV's working set for both axes is 5
   BIT-planes where OpenCV's is a byte-plane and two 16-bit planes — 8× smaller —
   so at 640×480 one side fits a 1 MiB L2 and the other does not. **The ratio at
   94×60 and 160×120, where both sides fit comfortably, is the arithmetic ratio.**
   If the 640×480 ratio exceeds it by more than ~30%, the excess is residency and
   the entry says so in those words, as
   [X-6](#x-6--is-the-t22-logic-speedup-real--done),
   [X-12](#x-12--t31-denoise-against-the-reference-implementation--done) and
   [X-13](#x-13--t33-morphology-against-cverode--cvdilate--done) each had to.
   The fixed per-call cost of both sides is measured on a 2×2 frame and printed
   beside every size, because a ns/pixel ladder divides it by the pixel count and
   it dominates the small end.
3. **Linear in N.** From N = 1 to N = 5 the stage count rises 5× and the
   destination plane count rises 3× (2·(N+1) planes across both axes, 4 → 12), so
   a linear formulation should land **roughly in the 5×–15× band**. The
   replication route T3.4 rejected would rise **31×** (`2·(2^N − 1)`, 2 → 62). A
   measured growth at or above 31× would mean the shipped formulation is not the
   linear one it is documented as, which **is** a finding and would reopen the
   N-bit path rather than be absorbed.

**Variants:** `OpenCV filter2D x2` (the **denominator** — the derivative and
nothing else, into pre-allocated `CV_16S`); `OpenCV as-written` (the same plus the
reference's `*= 16` on each result and its `cv::merge`, neither of which binCV
reproduces, so charging them to the baseline would flatter binCV); `binCV u32`;
`binCV u64`; `binCV composed u32`. Every row computes **both axes**, because that
is what a VIO frontend needs before it can form the T3.6 covariance.

**Workload:** 640×480 and the pyramid ladder below it (320×240, 160×120, 94×60),
~50% fill, four distinct images rotated through, batches calibrated to a 40 ms
budget with the minimum of five batches reported. The N-bit ladder runs at
640×480 for N = 1…5.

**Metric:** ns/pixel **and** the working set of one call, together (CLAUDE.md),
plus the measured fixed per-call cost of each side.

**Method:** `bincv-cpp/benchmark/derivative_benchmark.cpp`. The denominator is
ARCHITECTURE 10.3's and is not a judgement call for this operation: it is
`SEAL/src/keypoint_tracking/gradients.cpp`'s `calcBinarizedDeriv` — two
`cv::filter2D` calls with `[-1, 0, 1]` as a 1×3 and a 3×1 — on the same binary
content stored as `CV_8U`. The two kernel `cv::Mat`s and both `CV_16S`
destinations are hoisted out of the timed region as a caller in a frame loop would
hoist them; `cv::filter2D`'s own per-call kernel analysis is **not**, because it is
not something a caller can hoist. Every implementation is compared pixel for pixel
before anything is timed and each destination is folded into a
representation-independent checksum afterwards; a disagreement skips the size and
exits non-zero. For the N-bit rows the denominator is `cv::filter2D` on a `CV_8U`
image holding the pixel VALUES, which is the same operation with no scale factor,
and each row is checked against it before its time is reported. **See the
amendment at the end of this entry: when this was written the N-bit rows' OpenCV
calls ran only as a correctness oracle, outside every timed region, so the ladder
had no denominator despite this sentence and two others calling it one. The
benchmark now times them; the device number for that column is still outstanding.**

**Environment** *(one run, `bincv-cpp/results/derivative_benchmark_pi4.log`)*:

```
device pi4 · Raspberry Pi 4 Model B Rev 1.5 · aarch64 / 6.18.34+rpt-rpi-v8
g++ (Debian 14.2.0-19) 14.2.0 · governor performance · taskset -c 3
throttled before 0x0 · throttled after 0x0 · commit 6d05ec3
```

**Fixed per-call cost, measured on a 2×2 frame:** OpenCV (2 × `cv::filter2D`)
**9.301 µs**, binCV (`derivativeX` + `derivativeY`) **0.048 µs** — 194× apart, and
the reason the small end of the ladder cannot be read raw. `cv::filter2D`
re-analyses and re-separates its 1×3 kernel on every call; nothing a caller can
hoist.

**RESULT — ns/pixel, both axes, and the working set of one call:**

```
                    640x480          320x240          160x120           94x60
                 ns/px    vs      ns/px    vs      ns/px    vs      ns/px    vs
OpenCV filter2D  5.007  1.00x     4.961  1.00x     5.840  1.00x     8.514  1.00x
OpenCV as-written 10.033 0.50x    8.853  0.56x     9.674  0.60x    12.605  0.68x
binCV u32        0.201 24.90x     0.221 22.42x     0.271 21.57x     0.345 24.66x
binCV u64        0.114 43.75x     0.138 35.96x     0.205 28.54x     0.279 30.48x
binCV composed   0.590  8.48x     0.602  8.24x     0.715  8.17x     1.346  6.33x

working set of one call, both axes (bytes)
binCV u32       192000            48000            12000             3600
binCV composed  268800  (1.40x)   67200            16800             5040
OpenCV filter2D 1536000 (8.0x)   384000           96000            28200
OpenCV as-writ  2764800 (14.4x)  691200          172800            50760
```

All five rows of every size printed the same pixel-value checksum, and all five
were compared pixel for pixel before anything was timed.

**Rule 1 — fused against composed: the live branch did NOT fire, and nothing is
traded.** The fused kernel is **2.94× faster at 640×480** (2.72×, 2.64× and 3.90×
down the ladder) *and* 1.40× smaller. Memory and speed do not conflict here, so
there is no cost to record and no tiebreak to invoke — the same shape
[X-12](#x-12--t31-denoise-against-the-reference-implementation--done) found for
denoise, and the opposite of what
[D-16](ARCHITECTURE.md#d-16-morphology-fuses-the-shift-and-the-fold-and-only-the-compound-ops-take-scratch)
had to record for a non-separable morphological element. The composed row's eight
passes against the fused kernel's two are most of it.

**Rule 2 — the headline is NOT mainly a cache result, and that is the finding.**
The rule said the ratio where both sides fit in cache is the arithmetic ratio, and
that anything above ~30% of it at 640×480 is residency. Subtracting each side's
measured per-call floor, the ratios are:

```
640x480  24.8x      320x240  21.9x      160x120  20.0x      94x60  20.4x
```

**24.8 / 20.0 = 1.24× — inside the threshold.** So roughly **20× of the 24.9× is
the operation** and at most ~24% is the 8× smaller working set staying resident;
at 640×480 OpenCV's 1.5 MB does not fit the Pi 4's 1 MiB L2 and binCV's 192 KB
does, and that buys about a quarter. This is a **different answer** from
[X-6](#x-6--is-the-t22-logic-speedup-real--done),
[X-12](#x-12--t31-denoise-against-the-reference-implementation--done) and
[X-13](#x-13--t33-morphology-against-cverode--cvdilate--done), where residency
was the larger part of the story, and the reason is visible in the table rather
than speculative: `cv::filter2D` spends most of its time on per-pixel
multiply-accumulate arithmetic that binCV replaces with three word operations,
so the arithmetic gap survives into cache. The raw 94×60 ratio (24.66×) is
*higher* than 160×120's only because OpenCV's 9.3 µs floor is 19% of that frame;
corrected, the ladder is flat.

**Rule 3 — linear in N, confirmed.** From N = 1 to N = 5 the measured cost rises
**6.93×**, against 5× from the stage count and 3× from the destination planes
(2·(N+1) across both axes, 4 → 12) — inside the 5×–15× band the rule named, and
nowhere near the **31×** the replication route would cost. Every row was checked
against `cv::filter2D` on the same values before its time was reported. **Checked
against, not measured against** — see the amendment below. The rule-3 verdict does
not depend on the missing column, because it reads binCV's own curve against the
stage count; what was claimed and never measured is the ladder's *ratio*.

```
N        1      2      3      4      5
ns/px  0.200  0.463  0.783  1.044  1.388
vs N=1 1.00x  2.31x  3.91x  5.21x  6.93x     (2N stages: 1, 2, 3, 4, 5x)
                                             (replicated: 1, 3, 7, 15, 31x)
```

**One thing this measured that it was not asked to, and it is registered rather
than acted on.** `uint64_t` beat `uint32_t` by **1.75× at 640×480** (43.75× against
24.90×), narrowing to 1.32× at 160×120 and 1.24× at 94×60 — a bigger word-width
gap than [X-10](#x-10--default-word-width--done) measured for `bitwiseAnd`
(null, memory-bound) and closer to what it measured for `countNonZero` (1.94×).
At 640×480 the two word widths have **identical footprint** (38400 B/plane), so
[D-14](ARCHITECTURE.md#d-14-uint32_t-is-the-default-word-type)'s conjunction —
faster *and* no footprint increase — is satisfied at that size and fails only at
the upper pyramid levels, where `uint64_t` costs +20% and +33% per plane. That is
**exactly E-9**, the per-level word width question X-10 spun out, and this entry
adds a second operation to its evidence. **D-14 is unchanged and nothing here
overrides it**; the derivative ships at whatever `WordType` its caller's container
uses (D-1), so no code decision was deferred by leaving it open.

**Conclusion:** T3.5's done-when is satisfied — the benchmark exists, is committed,
and runs on the reference device against the denominator ARCHITECTURE 10.3
specifies. `derivativeX` + `derivativeY` cost **0.201 ns/pixel at 640×480 in
192 000 B**, against **5.007 ns/pixel in 1 536 000 B** for the two `cv::filter2D`
calls the reference pipeline runs: **24.9× faster in 8.0× less memory**, of which
about 20× is arithmetic and about a quarter is cache residency.

**Decision:** nothing to promote that
[D-19](ARCHITECTURE.md#d-19-the-derivatives-border-is-reflect-101-and-its-sign-is-the-borrow)
does not already record; the fused-versus-composed branch resolved with no trade,
so D-19 states the choice without a cost attached. E-9 gains a second data point
and stays open and unscheduled.

---

#### X-16 amendment · the N-bit ladder had no denominator · `OPEN`

*(Added by the T3.5 review, after the entry above was written and measured. It
corrects a claim, so it is an amendment in place rather than an edit over the
original text — the original is what the device actually measured.)*

**What was wrong.** Three places said the N-bit ladder's denominator is
`cv::filter2D` on a `CV_8U` value image: this entry's **Method**,
`benchmark/derivative_benchmark.cpp`'s header, and the ladder's own comment. The
two `cv::filter2D` calls were real, but they ran **once, outside every timed
region**, purely to check that binCV and OpenCV computed the same picture. The
ladder's printed columns were N, ns/pixel, vs N=1, 2N stages, replicated, planes —
no OpenCV row and no ratio. So the **N ≥ 2 path, which is what every pyramid level
above 0 runs, had no timed OpenCV comparison anywhere**, against ARCHITECTURE 10.3
and against T3.5's done-when. The word "denominator" was doing work no measurement
supported.

**What changed.** `timeNBit()` now runs the two `cv::filter2D` calls in their own
`measureNs`, over the same four images, with the destinations hoisted the way the
main table hoists them, and the ladder prints a **vs OpenCV** column. The ladder
also gained the **working-set columns** CLAUDE.md asks for — N is the one axis in
this benchmark along which binCV's footprint moves (3(N+1) − 1 bit-planes against
OpenCV's flat byte-plane plus two 16-bit planes), so a ns/pixel-only table was
reporting speed without memory on precisely the table where the two diverge.

**Status: the device number is NOT TAKEN, and nothing from the attempted run is
recorded as a result.** The re-run on `pi4` tripped the **soft temperature limit
during the benchmark** — `throttled before 0x0`, `throttled after 0x80000` — which
`run_on_pi.sh` reports as `RESULTS INVALID`. A throttled measurement is wrong
rather than merely slow. That flag is sticky until the device is rebooted, so no
further device run was possible in the same session. **The x86 run is indicative
only** ("Measurement platforms") and is written here as a shape, not a result:

```
x86_64, INDICATIVE ONLY -- not a result, do not quote
N          1       2       3       4       5
binCV  0.048   0.110   0.205   0.357   0.388   ns/px
OpenCV 0.519   0.520   0.522   0.519   0.526   ns/px
vs OCV 10.82x   4.72x   2.55x   1.46x   1.36x
bytes 192000  307200  422400  537600  652800   B   (OpenCV flat at 1536000 B)
```

**What closes it:** reboot the device (which is what clears the sticky flag), let
it reach a cold start, and re-run

```
BINCV_PI_OPENCV=1 ./scripts/run_on_pi.sh pi4 \
    './benchmark/derivative_benchmark > derivative_benchmark.log'
```

then replace `bincv-cpp/results/derivative_benchmark_pi4.log` — **which records
the PRE-AMENDMENT binary and therefore has no ladder ratio column** — and fill the
table above.

**What this does NOT change.** Rules 1 and 2 and the headline (24.9× faster in
8.0× less memory at 640×480) come from the main size table, which the amendment
touched only in a volatile-sink read. Rule 3's verdict — linear in N — reads
binCV's own curve against the stage count and the replication count, so it stands
on the device numbers already recorded.

What is missing is the ratio alone, and the indicative shape says something worth
measuring properly: **binCV's advantage over `cv::filter2D` shrinks steeply with
N**, because OpenCV's cost is flat in the pixel depth while binCV's is linear in
it. If that survives on the device it belongs to **E-7** (bits per pyramid level),
which is where each level's depth is decided. Registered here, not acted on.

**One more sink asymmetry, fixed with it.** The OpenCV rows' volatile sink read
`dx.at<short>(0, 0)`, which `BORDER_REFLECT_101` pins to **exactly 0 for every
input** — both taps read column 1 — while the binCV rows' sink
(`dx32.data()[0]`, spanning columns 0–31) is content-dependent. Measured on six
random 640×480 draws: `dx(0,0) = 0` every time, `dx(1,1)` alternating 0 and 4080.
All three OpenCV sinks now read an interior pixel. The 2×2 call-floor rows are
left constant on **both** sides, and the benchmark says why: on a 2×2 frame
reflect-101 maps column 2 back to column 0, so the whole derivative is identically
zero whatever the content — for binCV exactly as for OpenCV.

---

### X-17 · The LK gradient covariance, fused against composed at T3.6's own level · `DONE`

**Gates:** nothing that is open. [D-15](ARCHITECTURE.md#d-15-window-reductions-get-incremental-state-and-a-fused-covariance)
axis 2 is settled and `ops/covariance.hpp` is already written against the fused
entry point. [T3.6](TASKS.md)'s done-when nevertheless asks for the ratio to be
measured **at this level** rather than inherited, and CLAUDE.md requires the
decision rule to exist before the numbers do — so this entry exists, and says
plainly that it is a confirmation rather than a decision.

**Why it is not simply X-11 axis 2 again.** X-11 measured fused-versus-composed
one level down, on the reduction entry points, **with a precomputed
`sign_x ^ sign_y` plane on both sides**. `gradientCovariance` ships neither: it
calls the **four-argument** `countCovariance`, which XORs the two sign planes
inside the word loop (axis 3 — memory wins CLAUDE.md's tiebreak). That form loads a
fourth stream per word, so the redundancy a composition pays is a different
fraction of a larger number. Inheriting the ratio would be assuming the answer.

**Question:** at the T3.6 call site, on the scratch-free four-argument form, is one
fused traversal still faster than the three composed calls that produce the same
three numbers — and by how much, at W = 7, 15 and 31?

**Decision rule** *(written into `benchmark/covariance_benchmark.cpp`'s header
before any number existed)*:

1. **Fused beats composed at W = 31** → D-15 axis 2 holds where T3.6 calls it;
   `ops/covariance.hpp`'s "reach for the fused entry point" note is confirmed and
   nothing moves.
2. **Fused within noise of composed, or slower** → that **contradicts a documented
   claim** (D-15 axis 2, ARCHITECTURE §7.5, `ops/reduce.hpp`). CLAUDE.md's rule for
   that case is explicit: report it, do not adjust the code to fit the doc. T3.6
   would then be resting on a ratio that does not exist at its own level, and the
   spec's "built on the fused entry point" would need re-deciding.

No threshold is attached, deliberately. T2.10's 15% line selected an interface that
did not exist yet; this checks that an already-selected interface behaves as
recorded where it is actually called, so the question is direction and magnitude
against the measured spread, not a gate.

**Variants:** `fused` (`gradientCovariance` — what T3.6 ships, 0 B scratch);
`composed` (`countNonZero` ×2 + the four-argument `countAndSplit`, three
traversals, 6 word loads per word index against the fused pass's 4, also 0 B
scratch — so this pair is speed against speed with **memory held equal**, which is
what makes it a clean test of axis 2 rather than a mixture of axes 2 and 3);
`fused+plane` and `composed+plane`, the same two with a caller-held selector plane,
carried because CLAUDE.md requires memory and speed on one page. The plane's
formation cost is **not** charged to the timed loop, which flatters the plane forms
on purpose.

**Workload:** 640×480, 200 keypoints (the reference pipeline's
`gftt_max_corners`), one window each, scattered so that windows near the border
clip — the LK access pattern of [§7.5](ARCHITECTURE.md#75-lk-gradient-covariance).
Windows are deliberately **not** swept in a column: a caller that sweeps a column
should be calling `SlidingWindowCount` for `sumXX` and `sumYY` (X-11b axis 1,
5.96×–15.9×, which are single-plane `countNonZero` sweeps and not covariance
sweeps — the cross term has no incremental form and is recomputed per position),
and `ops/covariance.hpp` says so in its docstring. `uint32_t` and `uint64_t`,
W ∈ {7, 15, 31}, four rotating inputs, batches calibrated to 50 ms, 11 batches,
variants interleaved, spread reported beside every median. All four variants are
compared window for window and must agree before anything is timed.

**Metric:** ns per window, and the scratch each form needs, together.

**Method:** `bincv-cpp/benchmark/covariance_benchmark.cpp`.

**Status: THE DEVICE NUMBER IS NOT TAKEN, and nothing below is recorded as a
result.** `./scripts/run_on_pi.sh pi4` refused the run at preflight:
`throttled=0x80000` — the **soft temperature limit has occurred** bit, sticky since
a previous session (the device was idle at 45.7 °C and clocked at 1.8 GHz when
queried, so it is not throttling *now*; the flag records that it once did). The
flag clears only on a reboot, which this session was not permitted to perform. That
is the same wall the [X-16 amendment](#x-16-amendment--the-n-bit-ladder-had-no-denominator--open)
hit, and the same conclusion follows: a throttled measurement is wrong rather than
merely slow, so no number is recorded and the entry stays `PARTIAL`.

**The x86 run is INDICATIVE ONLY** ("Measurement platforms"), and doubly so here:
X-7 measured that on binCV's shipped x86 baseline `__builtin_popcountll` is a
libgcc CALL per word, which is the single instruction every variant in this table
is dominated by. It is written as a shape, not a result
(`bincv-cpp/results/covariance_benchmark_x86_indicative.log`):

```
x86_64, INDICATIVE ONLY -- not a result, do not quote
                fused    composed  composed/  fused+plane  comp+plane   plane/   spread
word      W   ns/window  ns/window     fused    ns/window   ns/window     4arg    f / c
uint32_t  7       99.8      128.4     1.29x         97.9       115.3    1.02x  36.0% / 37.5%
uint32_t  15     247.2      280.2     1.13x        260.3       291.0    0.95x  39.7% / 24.3%
uint32_t  31     712.0      756.2     1.06x        643.0       698.1    1.11x  38.8% / 69.4%
uint64_t  7       89.6      114.5     1.28x         92.2       108.7    0.97x  67.1% / 38.1%
uint64_t  15     203.0      244.7     1.21x        219.5       224.0    0.92x  33.2% / 69.0%
uint64_t  31     482.4      560.3     1.16x        508.5       573.2    0.95x  41.8% / 62.1%

The plane/4arg column does not even hold its SIGN here -- 0.92x to 1.11x,
straddling 1.00 -- which is most of why the table is not a result. The
composed/fused column is no better founded: every one of its ratios is
INSIDE its own within-run spread (the tightest row, uint32_t W=31, is 1.06x
against 38.8% / 69.4%), so the table is CONSISTENT WITH the direction D-15
axis 2 predicts and establishes nothing about it. An earlier run of the same
committed binary on the same host reported 1.14x/1.20x/1.06x/1.23x/1.18x/1.13x
with spreads to 124%: the ranking is not stable run to run, which is the
finding, not the ratios.
```

**MEMORY, which is measured rather than timed and therefore stands.** The scratch
column is arithmetic — a plane's size is not in doubt — and the `0 B` rows are now
an `operator new` count taken over one pass of each variant inside the benchmark
binary itself, plain and over-aligned forms both, with the counter's own teeth
printed beside the table. Printed as a literal it would have read `0 B` for a
`gradientCovariance` that allocated on every call, which is the one number D-15
axis 3 traded 11–14% of speed for:

```
fused (SHIPPED)        0 B        beyond the four derivative planes it reads
composed               0 B
fused+plane        38400 B        one sign_x^sign_y plane at 640x480
composed+plane     38400 B        ... and one at EVERY pyramid level: ~51 kB
                                  over four levels, a FIFTH plane on top of the
                                  four the covariance already reads, +25% of the
                                  derivative working set, held for the frame
```

**What closes it:** reboot the device (which is what clears the sticky flag), let
it reach a cold start, and run

```
./scripts/run_on_pi.sh pi4 './benchmark/covariance_benchmark > covariance_benchmark.log'
```

then commit `bincv-cpp/results/covariance_benchmark_pi4.log` and fill the table
above. The benchmark needs no OpenCV — it compares binCV against binCV — so the
device's default core-only build produces it.

**What this does NOT block.** T3.6's correctness bar is closed and is the part that
carries the project's central claim: the popcount identity of §7.5 agrees
**exactly** with a per-pixel float oracle at **383 200** window positions (95 800
per word type: four full-frame sweeps × W ∈ {7, 15, 31} × origins from a full
window outside each edge to a full window past it, on a frame taller than the
largest window so every size has fully-interior positions as well as clipped
ones), with **459 280** further positions checked against an invariant, in all
four verification configurations.
The no-scratch property that D-15 axis 3 traded speed for is checked by an
`operator new` counter and is **0 allocations**. Neither of those is a device
question.


**Result — reference device, `throttled` unchanged at `0x80000` (sticky history
from a previous session; no active bit, 45.7 °C idle — see the runner fix in
commit `73af779` for why that is not an invalidating condition), governor
`performance`, `taskset -c 3`, 640×480, 200 keypoints, four variants interleaved
and all agreeing before timing:**

| word | W | fused ns/win | composed ns/win | composed/fused | spread f / c |
|---|---|---|---|---|---|
| `uint32_t` | 7 | 226.9 | 296.9 | **1.31×** | 4.1% / 3.8% |
| `uint32_t` | 15 | 452.6 | 540.0 | **1.19×** | 0.8% / 4.2% |
| `uint32_t` | 31 | 945.7 | 1108.9 | **1.17×** | 0.8% / 1.5% |
| `uint64_t` | 7 | 217.3 | 268.8 | **1.24×** | 1.3% / 2.0% |
| `uint64_t` | 15 | 435.1 | 476.4 | **1.10×** | 1.5% / 1.2% |
| `uint64_t` | 31 | 857.6 | 937.3 | **1.09×** | 0.8% / 0.6% |

Memory, measured in this binary rather than printed as a claim: fused **0 B**,
composed **0 B**, either plane form **38400 B per pyramid level** at 640×480.

**Conclusion.** The fused entry point wins at every window size and both word
widths, by 1.09–1.31×, every gap comfortably outside its own spread. That
confirms [X-11](#x-11--e-3--window-reductions--done) axis 2 **at T3.6's own
level** rather than inheriting a reduction-level number — which is what the
Done-when asked for, because a ratio measured on bare reductions need not survive
being wrapped in an operation.

The **plane** form is faster again (1.20–1.44× over the four-argument form) and is
still not what T3.6 ships, because it costs a fifth plane at every pyramid level.
That is X-11 axis 3's tradeoff re-confirmed here, and it lands the same way:
memory wins ([CLAUDE.md](CLAUDE.md)).

Raw log: [bincv-cpp/results/covariance_benchmark_pi4.log](bincv-cpp/results/covariance_benchmark_pi4.log).

---

### X-18 · Does the incremental window form still pay inside T3.7's dense sweep? · `DONE`

**Gates:** nothing that is written. `ops/corner.hpp` is already built on
`SlidingWindowCount`, because [D-15](ARCHITECTURE.md#d-15-window-reductions-get-incremental-state-and-a-fused-covariance),
`ops/reduce.hpp`'s "WHICH SHAPE TO REACH FOR" table and `ops/covariance.hpp`'s
docstring all send a **dense sweep** there, and [T3.7](TASKS.md)'s spec repeats it.
This entry exists because that guidance had never been measured **at a caller**,
and CLAUDE.md requires the decision rule to exist before the numbers do.

**Why it is not X-11 axis 1 again.** X-11b measured `SlidingWindowCount` against
`countNonZero` on **one plane, one number**, and reported **15.9×** on a dense scan
at 31×31. T3.7 sweeps a **2×2 covariance**, and only two of its three numbers have
an incremental form: `sumXX` and `sumYY` slide, `sumXY` needs `magX & magY` split
by `signX ^ signY` and nothing in `ops/reduce.hpp` slides a split, so the cross
term is recomputed per position **on both sides of the comparison**. The saving is
therefore bounded by the share of the work the other two numbers represent — and
the accumulator also **forces a column-major traversal**, because it only slides
downward. Inheriting 15.9× would be assuming the answer twice over.

**Question:** at 640×480, is `cornerMinEigenVal`'s sliding sweep faster than the
obvious `gradientCovariance`-per-position recomputation — and by how much, at
`blockSize` ∈ {3, 7, 15, 31}?

**Decision rule** *(written into `benchmark/corner_benchmark.cpp`'s header before
any number existed)*:

1. **Sliding faster at every block size** → X-11 axis 1's advantage survives being
   embedded in a caller that can slide only two thirds of its state;
   `ops/corner.hpp`'s "this is the sliding form" note stands and the magnitude
   recorded here — not 15.9× — is what a caller plans with.
2. **Sliding within the measured spread of recompute, or slower** → that
   **contradicts a documented claim**. CLAUDE.md's rule is explicit: report it, do
   not adjust the code to fit the doc.
3. **A ratio near 15.9× would also be a surprise**, and would mean the cross term
   is not the dominant cost the argument above assumes. Written down so it could
   not be quietly welcomed as a good result.

**Variants:** `sliding` (`cornerMinEigenVal` — what T3.7 ships); `recompute`
(`gradientCovariance` per position, **row-major**, which is how anyone would write
it); and `recompute-col`, the same recomputation swept **column-major**. The third
is a control and the entry is not interpretable without it: `sliding` differs from
`recompute` in *two* ways at once — incremental state and traversal order — and on
a 32 KiB L1 those pull in opposite directions. `sliding / recompute-col` isolates
the incremental effect; `recompute / recompute-col` isolates the traversal effect.

**Workload:** 640×480, four rotating frames with real corner structure
(overlapping blocks, a diagonal, sparse texture), `uint32_t`, batches calibrated
to 200 ms, 5 batches, variants interleaved. All three maps are compared **bit for
bit** and must agree before anything is timed.

**Metric:** ns per pixel of response map, and the working set each form needs,
together.

**Method:** `bincv-cpp/benchmark/corner_benchmark.cpp`.

**Result — reference device, `throttled=0x0` before AND after (a clean baseline,
not sticky history), governor `performance`, `taskset -c 3`, g++ 14.2.0, commit
`5158341`. Within-run spreads 0.04–3.4%; run-to-run scatter measured separately
over four runs and reported below the table:**

| blockSize | sliding ns/px | recompute ns/px | recompute-col ns/px | net (rec/slide) | incremental alone | traversal alone |
|---|---|---|---|---|---|---|
| 3 | 101.25 | **84.83** | 95.19 | **0.84×** | **0.94×** | 1.12× |
| 7 | 142.84 | 138.55 | 148.49 | **0.97×** | 1.04× | 1.07× |
| 15 | 252.89 | 278.78 | 288.22 | **1.10×** | 1.14× | 1.03× |
| 31 | 581.44 | 704.39 | 711.91 | **1.21×** | 1.22× | 1.01× |

**RUN-TO-RUN SCATTER, FROM THREE FURTHER RUNS OF THE SAME BINARY.** The spreads
above are `measure_util.hpp`'s **within-run** figure — five interleaved batches
inside one process. That is not the number a difference has to clear, and this
entry originally quoted it as though it were. X-17 (immediately above) recorded
that re-runs move ratios; X-14 took three device runs for the same reason. So
three more were taken, same binary, same protocol, `throttled=0x0` before and
after each ([`corner_benchmark_pi4_scatter.log`](bincv-cpp/results/corner_benchmark_pi4_scatter.log)):

| blockSize | sliding ns/px (min…max) | recompute ns/px (min…max) | net ratio (min…max) | net scatter |
|---|---|---|---|---|
| 3 | 101.25 … 101.54 | 84.83 … 84.88 | 0.836 … 0.838 | 0.28% |
| 7 | 142.54 … 143.06 | 138.55 … 138.64 | 0.969 … 0.972 | 0.34% |
| 15 | 252.89 … 253.72 | 278.78 … 279.45 | 1.100 … 1.102 | 0.18% |
| 31 | 558.89 … 581.44 | 687.34 … 704.39 | 1.203 … 1.243 | 3.32% |

**Run-to-run is the LARGER number at `blockSize` 31 (3.3% on the ratio, against a
1.9–3.8% within-run spread) and the smaller one below it (0.06–0.76%).** The
ranking is preserved in every run at every block size: the net ratio never reaches
1.00 at 3 or 7 and never falls to 1.00 at 15 or 31. **The `blockSize` 7 row — the
one the crossover claim rests on, and the smallest gap in the table — survives:**
its 3.1% net gap is about seven times the combined run-to-run range of the two
rows it is taken from. So the crossover between 7 and 15 is a measured boundary
rather than one-run evidence, which is what these runs were taken to establish.

**And they correct one statement in the conclusion below.** "The incremental state
itself is a loss below `blockSize` 15" does not follow from the numbers even in
run 1, which already showed **1.04× at `blockSize` 7 — a win**. It replicates at
1.0382–1.0395 across four runs with 0.13% scatter. The **incremental** effect
crosses over between 3 and 7; the **net** effect crosses between 7 and 15, because
the column-major traversal the accumulator forces (1.072× at 7) more than eats the
incremental win there. That distinction is exactly what the `recompute-col` control
was added to make, and the headline is unchanged: at `seal_params.yaml`'s
`blockSize` 3 the shipped sliding sweep is **1.20× slower** than a plain row-major
recomputation. The bullets below are corrected accordingly.

**Memory, identical for all three variants and measured rather than printed:** four
one-bit derivative planes **153 600 B**, the `float` response map **1 228 800 B**,
caller scratch **0 B** (two stack accumulators), total **1 382 400 B**. The
`operator new` count — plain and C++17 over-aligned — is **0** for every variant at
every block size. So this is speed against speed with **footprint held exactly
equal**, and nothing was traded for the result below.

**Conclusion — rule 2 fired, and it fired at the block size the reference pipeline
actually runs.** `gftt_block_size` is **3** in `SEAL/seal_params.yaml`, and there
the shipped sliding sweep is **19% SLOWER** than a plain row-major recomputation.
The control variant says why, and it is two independent losses rather than one:

* **The incremental state itself is a loss at `blockSize` 3 and only marginal at
  7** — 0.93× at 3, 1.039× at 7, then 1.14× and 1.24× at 15 and 31 (four-run
  medians). At a 3-row window, sliding replaces three row counts with two while
  paying a per-column construction and per-position bookkeeping, and the cross term
  traverses all three rows either way. The 4% it wins back at 7 is less than the
  traversal costs there, which is why the NET crossover sits higher than this one.
* **The column-major traversal the accumulator forces costs 12% at `blockSize` 3**,
  decaying to 1% at 31 as per-position work comes to dominate. `SlidingWindowCount`
  slides only downward, so a caller that wants the incremental form has no choice
  about this.

Both effects shrink as the window grows, which is why the two of them together
cross over between 7 and 15 — higher than the incremental effect crosses on its
own, because the traversal penalty is still 7% at `blockSize` 7. All four runs put
the net ratio below 1.00 at 7 and above it at 15.

**This contradicts documented guidance, and the documents are what change, not the
kernel.** `ops/reduce.hpp`'s table, `ops/covariance.hpp`'s docstring, D-15 and
T3.7's own spec all point a dense sweep at the incremental form with **no
window-size qualification**. That qualification now exists and is recorded in
`ops/corner.hpp`'s docstring, in the table above, and in D-15's amendment.

**What is NOT decided here, deliberately.** Whether `cornerMinEigenVal` should
select on `blockSize` — recompute row-major below ~15, slide above — is an open
decision and this entry does not take it. One device at one frame size is thin
evidence for a permanent branch, and the x86 run has the **opposite sign** at
`blockSize` 3 (1.19× *in the sliding form's favour*) with spreads past 50%
(`bincv-cpp/results/corner_benchmark_x86_indicative.log`, filed as indicative only
per "Measurement platforms"). Closing it needs the same table on a second Cortex-A
part and at a second frame size, plus a decision about whether a window-size branch
inside a kernel is a shape this project wants at all. Until then T3.7 ships the
form its spec named, with the cost written down beside it.

Raw logs: [bincv-cpp/results/corner_benchmark_pi4.log](bincv-cpp/results/corner_benchmark_pi4.log)
(run 1) and
[bincv-cpp/results/corner_benchmark_pi4_scatter.log](bincv-cpp/results/corner_benchmark_pi4_scatter.log)
(runs 2–4, with the cross-run summary).

---

### X-19 · The tier 2 denominator: `goodFeaturesToTrack` against OpenCV · `DONE`

**Question:** T3.7 is API tier 2 — it has a direct `cv::` counterpart — so
CLAUDE.md's denominator rule applies to it: *OpenCV doing the same semantic
operation on the same binary content stored as `CV_8U`*, with **peak working set
reported beside speed**. X-18 answered a binCV-versus-binCV question and recorded
"memory identical for all three variants", which is binCV against binCV. The
1 228 800 B `float` response map — the operation's whole memory cost per
`ops/corner.hpp`, and eight times the four one-bit planes it reads — had therefore
never been weighed against what a byte-per-pixel pipeline pays for the same
answer. This closes that.

**Decision rule, written before measuring.** This is not a gate on a shipped
interface; it is the missing denominator for a claim the project makes about
footprint. Written down first anyway, because "record what result favours which
conclusion" applies to a characterization too:

1. **binCV smaller AND faster** → the tier 2 claim is unqualified and the response
   map's cost is bought back.
2. **binCV smaller and SLOWER** → the trade is real and must be stated as a trade,
   with both numbers, wherever T3.7's benefit is claimed. CLAUDE.md's tie-break
   applies (*memory wins* when the two conflict and no explicit choice was made),
   but the speed number gets printed, not buried.
3. **binCV larger** → the `float` response map is not affordable and T3.7's shape
   needs re-deciding, not re-measuring.

**Variants:** `binCV` (pack → `derivativeX`/`derivativeY` → `goodFeaturesToTrack`);
**`OpenCV binarized`, THE DENOMINATOR** — the same semantics in stock OpenCV over
`CV_8U`: two `filter2D` calls with the reference's `[-1, 0, 1]` tap, the three
product planes, a `boxFilter` **sum** (`BORDER_CONSTANT`, which for a sum is
exactly T3.6's clipped window), the min eigenvalue, then gftt.cpp's selection
including `greaterThanPtr` and the greedy spacing filter; and `OpenCV Sobel`,
stock `cv::goodFeaturesToTrack`, which is **not** the same numerics and is timed
only because it is the call a reader reaches for.

**Method:** `bincv-cpp/benchmark/corner_opencv_benchmark.cpp`, 640×480, four
rotating frames, `seal_params.yaml`'s parameters verbatim, `measure_util.hpp`'s
protocol. Both selections are compared **before** anything is timed.

**Result — reference device, `throttled=0x0` before AND after, governor
`performance`, `taskset -c 3`, g++ 14.2.0, 640×480:**

| variant | ns/frame | ns/pixel | spread | vs denominator | B/pixel |
|---|---|---|---|---|---|
| binCV | 42 428 078 | 138.11 | 0.15% | **0.55×** | **16.54** |
| **OpenCV binarized** (denominator) | 23 365 791 | 76.06 | 1.27% | 1.00× | 36.94 |
| OpenCV Sobel (stock, different numerics) | 18 277 527 | 59.50 | 1.15% | 1.28× | 29.00 |

**Peak working set, itemized rather than ratioed** (307 200 pixels):

| buffer | binCV | OpenCV binarized |
|---|---|---|
| source | 38 400 B (1 bit/px) | 307 200 B (`CV_8U`) |
| derivatives | 153 600 B (four 1-bit planes) | 2 457 600 B (`dx`, `dy` `CV_32F`) |
| covariance planes | — (popcounts, no plane) | 3 686 400 B (`xx`, `yy`, `xy` `CV_32F`) |
| response map + NMS scratch | 1 228 800 B (one `float` map, no dilate buffer) | 2 457 600 B (`eig` + dilate destination) |
| candidate list, worst case | 3 659 568 B (**is** the output array) | 2 439 712 B (`vector<const float*>`) |
| **total** | **5 080 368 B (16.54 B/px)** | **11 348 512 B (36.94 B/px)** |
| total at the **measured** survivor count (13 272) | **1 580 064 B (5.14 B/px)** | 9 014 976 B (29.35 B/px) |

Both candidate buffers are charged their worst case `(w−2)(h−2)` in the first
total and the measured survivor count in the second, symmetrically — binCV's
output array *is* its candidate buffer and OpenCV's pointer vector is the same
list. **2.23× smaller at the worst case, 5.71× at the measured one.** binCV
allocates nothing (`operator new` count 0, X-18); OpenCV's buffers are the
allocation.

Raw logs:
[bincv-cpp/results/corner_opencv_benchmark_pi4.log](bincv-cpp/results/corner_opencv_benchmark_pi4.log),
and
[bincv-cpp/results/corner_opencv_benchmark_x86_indicative.log](bincv-cpp/results/corner_opencv_benchmark_x86_indicative.log)
(indicative only — spreads past 50%, per "Measurement platforms").

**AGREEMENT FIRST, BECAUSE IT IS THE SURPRISE.** Over four frames binCV and the
OpenCV binarized pipeline return the **same number of corners at exactly the same
positions** — 723 of 723, worst displacement 0.00 px. That is not bit-exactness
and is not claimed as it (tier 2 stands: the response *maps* differ in their last
bits, `double` against `float`), but it says the two sides agree on every decision
the selection makes on this content, ties included. It is also independent
evidence for the tie order: the OpenCV side sorts with `greaterThanPtr`, so an
implementation whose tie rule disagreed with the reference could not produce this
number.

**Rule 2 fired.** binCV is **2.23–5.71× smaller and 1.82× slower** than the
denominator, and both halves are recorded wherever T3.7's benefit is claimed. Part
of that 1.82× is already priced: X-18 measured the shipped sliding sweep as 1.20×
slower than a row-major recomputation *at this very block size*, so roughly a
third of the gap is an internal choice X-18 registered as an open decision rather
than anything intrinsic to bit-parallel corner detection. The rest is that the
denominator spends 28 bytes per pixel of `float` planes to buy locality binCV
declines to buy.

**Conclusion.** The `float` response map is affordable. It is the single largest
fixed binCV buffer — 1 228 800 B, eight times the four one-bit planes, and
`ops/corner.hpp` calls it the operation's whole memory cost — and the pipeline
still costs 5.14 B/pixel against the denominator's 29.35, because the OpenCV side
needs **seven** `float` planes where binCV needs **one**. Speed goes the other way,
by 1.82×. Neither number settles the question alone, which is why CLAUDE.md asks
for both and why *memory wins* is the project's stated tie-break.

**What this does NOT close.** T3.7's own "select on `blockSize`" question stays
open (X-18) — this entry prices its cost against an external baseline but takes no
new decision. And the denominator here is the reference pipeline *expressed in
OpenCV*, not the reference binary itself, which this repo cannot run.


---

### X-20 · Hybrid LK: accuracy against ground truth, and the frontend's peak footprint · `DONE`

**Gates:** [T3.8](TASKS.md#t38--hybrid-lk-tracking--done)'s two Done-when
criteria, and it is the measurement that
[E-7](ARCHITECTURE.md#register) / [T4.1](TASKS.md) now depends on ·
confirms [E-10](ARCHITECTURE.md#register)
**Question:** (a) How close to a KNOWN displacement does route (b) hybrid LK get
over 1-bit frames? (b) What is the peak working set of the whole frontend at
640×480, by stage?
**Hypothesis:** (a) sub-pixel, and better than the fractional part an integer-only
tracker would be stuck with — that is the entire purpose of route (b) over route
(a). (b) the float response map dominates, which is what E-10 was registered on.

**Decision rule** — *written in `tests/test_opticalflow.cpp` before any error was
measured, and derived from the REPRESENTATION rather than from a run:*
- **RMS endpoint error ≤ 0.25 px.** A 1-bit frame locates an edge crossing to
  ±0.5 px; a 31×31 window averages many crossings, so the aggregate must beat the
  single-crossing bound by at least a factor of two (an effective count of four
  independent crossings — the modest form of the claim).
- **Max endpoint error ≤ 1.0 px** — one whole pixel of the grid the estimate is
  read off: the same ±0.5 px per-axis bound as above, doubled, so that no SINGLE
  point may miss the single-crossing bound by more than a factor of two.
- **≥ 80% of eligible points tracked, and no tracked point may be STUCK** — ground
  truth moved it by ≥ 0.5 px (the 1-bit localisation bound) while the tracker
  reports a total displacement ≤ `lk_term_criteria_eps` = 0.03 px, the step at
  which the iteration calls itself converged. Without the second half the rule is
  vacuous: on the real frame 141 of 141 points come back tracked in every case,
  including ones that never moved at all.
- **On a translation with fractional part `q`, RMS strictly below `min(q, 1−q)`** —
  the error a tracker restricted to whole-pixel displacements cannot avoid, since
  it can only return `round(d)`.

**TWO OF THOSE BULLETS WERE RESTATED AFTER A REVIEW, WITHOUT MOVING A NUMBER.**
The max bound and the sub-pixel criterion were originally justified by route (a) —
"integer block matching gives 1.0 px" and "an integer-only tracker's error is
exactly `q`". Both halves of that are wrong: a minimising integer matcher returns
`round(d)`, so its error is `min(q, 1−q) ≤ 0.5` per axis rather than `q` or 1.0,
and **binCV contains no route (a) implementation** — route (a) is E-6 / T4.2 — so
nothing had been measured to justify either figure. The 1.0 px is unchanged and now
rests on the representation alone; the sub-pixel criterion moved to the tighter,
derived `min(q, 1−q)`, which the operation still clears at q = 0.75 by 2.6×. No
tolerance was widened at any point in this entry's life.
- **One allowance, also derived:** LK's model is a pure translation, so under a
  rotation `θ` or a scale `s` the true displacement varies across the window by up
  to `halfWin·θ` and `halfWin·|s−1|`. That is model error, added to both bounds
  for those cases only — 0.26 px at 1° and 0.30 px at 1.02×.

**Variants:** synthetic texture and the repo's real test image, at 1 and 4 pyramid
levels; sub-pixel translations, integer translations, rotation, scale.
**Workload:** 320×240 synthetic, 752×480 real; 31×31 windows, `lk_max_level 3`,
20 iterations, eps 0.03, minEig 0.001 — `seal_params.yaml` verbatim. 30–141
eligible keypoints per case from `goodFeaturesToTrack` with its own
`seal_params.yaml` defaults.
**Metric:** RMS and maximum endpoint error in pixels against ANALYTIC ground
truth; peak bytes by stage with `operator new` counted.

**Method — and the harness is the part that matters.** Ground truth is never
another estimator. Frame 1 is the binarization of a WARPED CONTINUOUS FIELD:
`frame1(z) = [f(A⁻¹z) > 0]` against `frame0(z) = [f(z) > 0]`, so every point's
displacement is `Az − z` exactly. For the real image the same shape is used on the
decoded grayscale — warp the continuous-valued thing, binarize both frames
afterwards with the SAME function. That function is the reference pipeline's own
`rl_fast_edge_filter_wide(edge_threshold = 17)`
(`SEAL/src/temporal_processing/edge_filter.cpp`), ported and checked: it
reproduces the repo's shipped `_bin_normalized.png` to within **0.024%** of
pixels. Warping the BITS instead would mean resampling binary content, which
cannot be done without inventing information. Code: `tests/test_opticalflow.cpp`,
registered as a core suite.

**Result (a) — synthetic texture, 320×240, four 1-bit levels:**

| case | RMS px | max px | stuck | tolerance | verdict |
|---|---|---|---|---|---|
| shift (0.25, 0.25) | 0.1117 | 0.2366 | 0/0 | 0.25 / 1.0 | within |
| shift (0.50, 0.50) | 0.1166 | 0.3108 | 0/35 | 0.25 / 1.0 | within |
| shift (0.75, 0.75) | 0.1500 | 0.4622 | 0/35 | 0.25 / 1.0 | within |
| shift (0.75, 0.25) | 0.1129 | 0.3617 | 0/35 | 0.25 / 1.0 | within |
| shift (2.25, −1.50) | 0.1231 | 0.3183 | 0/35 | 0.25 / 1.0 | within |
| shift (1, 0) | 0.0004 | 0.0013 | 0/35 | 0.25 / 1.0 | within |
| shift (0, −2) | 0.0001 | 0.0003 | 0/35 | 0.25 / 1.0 | within |
| shift (3, 2) | 0.0005 | 0.0021 | 0/35 | 0.25 / 1.0 | within |
| shift (−5, 4) | 0.0004 | 0.0013 | 0/34 | 0.25 / 1.0 | within |
| rotate +1° | 0.1519 | 0.4892 | 0/35 | 0.512 / 1.262 | within |
| rotate −1° | 0.1348 | 0.3385 | 0/34 | 0.512 / 1.262 | within |
| scale 1.02 | 0.1452 | 0.3130 | 0/34 | 0.550 / 1.300 | within |
| scale 0.98 | 0.1497 | 0.3455 | 0/35 | 0.550 / 1.300 | within |

("stuck" is `stuck / ground truth moved ≥ 0.5 px`. The 0.25 px diagonal shift moves
truth by 0.354 px, below the bound the representation resolves at all, so no point
in that row is eligible to be called stuck — which is why the denominator is 0.)

**And the criterion that separates route (b) from a whole-pixel tracker:** at
`q` = 0.25, 0.50, 0.75 the RMS is **0.0756, 0.1018, 0.0950** against the
`min(q, 1−q)` = 0.25, 0.50, 0.25 a whole-pixel tracker cannot avoid — **3.3×, 4.9×
and 2.6× better**.

**These numbers replace the ones this entry first recorded** (0.1176 / 0.2458 for
the 0.25 px shift, and so on). Two shipped changes moved them, both from the same
review: the reference's pyramid cap is now reproduced, so 320×240 uses three levels
rather than four — the fourth would have been 40×30 under a 31×31 window — and
`err` is measured at the returned position. Re-measured, not re-derived.

**Result (a) — the repo's real frame, 752×480, reference binarization (10.36%
set), 141 eligible keypoints:**

| case | 1 level RMS / max / stuck | 4 levels RMS / max |
|---|---|---|
| stationary | **0.0000 / 0.0000** / 0 | **0.0000 / 0.0000** |
| shift (1, 0) — axis-aligned | 0.0017 / 0.0124 / 0 | **3.2530 / 18.68** |
| shift (0, 1) — axis-aligned | 0.0004 / 0.0017 / 0 | — |
| **shift (1, 1) — DIAGONAL** | **0.7532 / 1.4142 / 40 of 141** | — |
| **shift (2, 2) — DIAGONAL** | **1.7459 / 3.5823 / 45 of 140** | — |
| shift (0.25, 0.25) | **0.2860** / 1.3417 / — | **1.2645** / 7.78 |
| shift (0.25, 0) | **0.2161** / 1.5323 / — | — |
| shift (0.50, 0.50) | **0.4587** / 2.5691 / 34 | **2.2311** / 11.43 |
| shift (0.75, 0.75) | **0.6800** / 3.2123 / 37 | **3.5093** / 19.58 |
| shift (2, −3) | — | 5.8461 / 33.81 |
| rotate 1° | — | 4.5949 / 19.52 |
| scale 1.02 | — | 8.2501 / 47.15 |

**THE STATED TOLERANCE IS NOT MET ON REAL CONTENT.** It was not widened and the
`minEigThreshold` was not changed. Four facts locate the cause, and **the first
version of this entry attributed all of the miss to the third of them. A review
asked for the control measurements, and they do not support that.**

1. **It is not the bit-parallel arithmetic.** An independently written per-pixel
   FLOAT implementation of the same algorithm — the multiply-and-accumulate
   formulation the popcount residual identity replaces — agrees with the shipped
   kernel to **0.000e+00 px** on this same real content, with zero status
   mismatches, and the residual identity itself agrees to **6.3e-14** over 864
   window positions per word type.
2. **It is not the propagation.** A stationary frame tracks EXACTLY at every level
   count.
3. **PART OF IT IS A LEVEL-0 FAILURE MODE WITH NO PYRAMID INVOLVED.** At ONE level
   the axis-aligned 1 px translation tracks to 0.0017 px — and the DIAGONAL one to
   **0.7532 px, with 40 of 141 points returning EXACTLY zero flow** while ground
   truth moved 1.41 px. Those points are a stationary point of the iteration, not
   a rejection: `b1 = b2 = 0` at zero displacement on a one-pixel-wide edge map
   whose two gradient components live on the same pixels, and the weakest of them
   still scores 0.033 against the 0.001 rejection threshold. **The sub-pixel rows
   miss the tolerance at one level too**, where no pyramid exists to blame. The
   original entry reported only `(1, 0)` and generalised from it; `(1, 0)` is a
   near-exact special case of this content and `(1, 1)` is the same easy
   displacement, one axis apart, failing both halves of the tolerance.
4. **AND PART OF IT IS THE PYRAMID — HALF BIT DEPTH, HALF CLIPPED WINDOWS.**
   Accuracy still degrades **monotonically** as 1-bit levels are added: 0.0017 px
   at one level to 3.2530 px at four for a 1 px shift. Two effects move together
   there, and the control below separates them: the same warp over the subset of
   points whose 31×31 window is inside EVERY level, so that deviation (ii) — binCV
   clips where the reference pads — cannot contribute.

   | levels | all 141 eligible points | the 58 whose window never clips |
   |---|---|---|
   | 1 | 0.0017 / 0.0124 | 0.0024 / 0.0124 |
   | 2 | 0.8985 / 7.01 | 0.7441 / 4.22 |
   | 3 | 3.1327 / 18.69 | 1.4764 / 9.10 |
   | 4 | **3.2530 / 18.68** | **1.4742 / 9.11** |

   **Roughly half of the headline 3.25 px is the clipped coarse-level window**, an
   accuracy cost of deviation (ii) that had never been measured — the decision to
   decline the reference's 1.24× padded levels was taken on footprint alone. The
   other half survives the control: **0.0024 px to 1.47 px on windows that never
   clip is still 600× and still six times the tolerance**, so a level whose pixels
   are BITS genuinely cannot localise sub-pixel motion, and that error is
   multiplied by 2^level on the way down. Synthetic texture never shows either
   effect: its level sets are smooth, its coarse levels still carry the geometry,
   and its level-0 basin is wide.

**Reference device, same suite, `./scripts/run_on_pi.sh pi4 './tests/test_opticalflow'`
— 148/148 checks (15 cases; the real-frame case needs OpenCV and is not built
there), `throttled=0x0` before and after:**

```
  device:           pi4
  cpu:              Raspberry Pi 4 Model B Rev 1.5
  arch / kernel:    aarch64 / 6.18.34+rpt-rpi-v8
  compiler:         g++ (Debian 14.2.0-19) 14.2.0
  governor:         performance (restored to ondemand on exit)
  core pinning:     taskset -c 3
  throttled before: throttled=0x0
  throttled after:  throttled=0x0
  commit:           9956596 + the T3.8 review fixes in the working tree
```

The footprint table is **byte-identical** on aarch64 — it is integer arithmetic
over container sizes. **AND SO IS EVERY ACCURACY NUMBER**: all thirteen synthetic
cases and all three T4 rows print the same values to the four decimals the suite
reports, on both platforms, from the same source. The only quantity that differs
anywhere in the suite is the residual identity's worst rounding gap — 6.306e-14 on
x86-64 against 6.573e-14 on the device — which is the float-summation-order effect
showing up in the one place small enough to see it.

**THIS CORRECTS THIS ENTRY'S OWN EARLIER TEXT**, which reported `q` = 0.25 as
0.0756 on the device against 0.0765 on x86-64 and explained the gap by aarch64's
fused multiply-add. Re-measured on both platforms from one tree, there is no gap:
the two figures came from two different builds of the tracker, one of them from
before the tap-displacement fix. The CAUTION still stands as a caution — the solve
is float, and D-20 declines to promise cross-ISA bit-identity for it, where
`cornerMinEigenVal` does promise it — but it is not currently visible in any
reported accuracy number, and saying that it was is what this paragraph is here to
withdraw.

**Result (b) — peak footprint of the full frontend, 640×480, `uint32_t`, four
1-bit levels, 200 tracked points. `operator new` inside every kernel: ZERO.**

| stage | buffers it owns | bytes | share |
|---|---|---|---|
| denoise | 2 incoming frames, 1 bit/px (dst is pyramid L0) | 76 800 | 4.5% |
| pyramid | 2 × 4 levels, 1 bit/px | 102 240 | 5.9% |
| derivative | dx+dy ternary, 2 bits/px, prev pyramid only | 204 480 | 11.9% |
| corner | float response map (1 228 800) + 8 754 candidates (105 048) | 1 333 848 | **77.5%** |
| track | prevPts/nextPts/status/err, 200 points | 4 200 | 0.2% |
| **TOTAL** | | **1 721 568** | |

> **THIS TABLE IS SUPERSEDED AS THE PROJECT'S CURRENT FRONTEND FOOTPRINT.** It is
> the frame-map shape, which
> [D-22](ARCHITECTURE.md#d-22-the-corner-response-streams-over-a-three-row-ring-and-that-is-the-recommended-path)
> no longer recommends. On the recommended path the corner row is **112 744 B** and
> the **TOTAL is 500 464 B** — same content, same five stages, identical corners,
> re-measured by the same case
> ([X-23](#x-23--the-rolling-response-ring-against-the-frame-sized-response-map--done),
> T3.11). **Phase 4 quotes 500 464 B**, not the total above. The table stays because
> it is the measurement that made the case for changing the shape, and because the
> 77.5% row is the finding.

**E-10 IS CONFIRMED AND IT IS NOT MARGINAL: the float response map alone is
71.4% of the frontend and more than everything else combined.** Four bytes per
pixel where every other plane is one or two BITS. The tracker itself — the
operation this task added — is 0.2%.

**TWO QUALIFICATIONS THE FIRST VERSION OF THIS TABLE DID NOT CARRY**, both from
the same review:

- **The candidate array is a PER-FRAME READING, not a bound**, and it is the only
  row that is: every other row is fixed by the frame size. 8 754 survivors here,
  **9 774 on the 752×480 real frame** (117 288 B) — the suite now prints both. A
  deployed caller cannot know the count in advance and must provision or accept
  truncation; provisioning the structural maximum `W·H` would be 3 686 400 B and
  would make the candidate array, not the response map, the dominant term.
  Related, and now fixed: the `W·H` buffer used to MEASURE the survivor count was
  still in scope when the table was printed — 3 686 400 B, 2.1× the reported
  total, excluded from a number labelled "peak". It is scoped and destroyed before
  the accounting.
- **`5.60× a single CV_8U 640×480 frame` is a SCALE REFERENCE, NOT CLAUDE.md's
  DENOMINATOR, and must not be quoted as a memory result.** CLAUDE.md's denominator
  is "OpenCV doing the same semantic operation on the same binary content stored as
  `CV_8U`" — for this frontend that is two `CV_8U` frames, two `winSize`-padded
  `CV_8U` pyramids, `CV_16S` derivatives and the same float response map, none of
  which is built or measured here. **That comparison is E-5 / T4.3's**, and until
  it is run this entry states no memory win.

**Conclusion.**
- Route (b) meets its stated accuracy tolerance on well-textured binary content
  and **beats the whole-pixel bound by 2.6–4.9× on exactly the sub-pixel cases
  that justify its existence**. The hybrid is doing what §7.9 chose it to do.
- It does **not** meet that tolerance on the reference pipeline's own edge-map
  content, and **three distinguishable things are wrong there, not one**: a
  level-0 stationary point that leaves ~28% of points exactly where they started
  on a diagonal displacement; the 1-bit level's own quantisation; and the clipped
  coarse-level window, which is about half of the four-level number.
- The frontend's footprint is dominated by a `float` scratch buffer, not by any
  image plane.

**Decision.** Three, and one deliberate non-decision.
1. **E-7 / T4.1 is promoted from an optimisation to a precondition — but it is no
   longer the whole fix.** Per-level bit depth is not a footprint tuning knob; on
   windows that never clip, four 1-bit levels are still 600× worse than one. T4.1
   must also produce the bit-sliced weighted-sum covariance of
   [§7.5](ARCHITECTURE.md#75-lk-gradient-covariance), because without it binCV
   cannot build an N-bit level at all. **What T4.1 must NOT be asked to close on
   its own** are the other two terms this entry now separates: the level-0
   stationary point, which is present with no pyramid at all, and the clipped
   coarse-level window, which is deviation (ii)'s accuracy cost.
2. **THE COST OF DEVIATION (ii) IS NOW MEASURED, AND IT WAS NOT WHEN THE
   DEVIATION WAS TAKEN.** Declining the reference's `winSize`-wide padded levels
   was decided on footprint alone (1.24× per level); its accuracy cost on real
   content is about half of the four-level error. CLAUDE.md's loop wants both
   sides of a trade weighed, so the decision is now weighable — and it is NOT
   reversed here: re-adding the border is a footprint decision that belongs with
   T4.1, which is already rebuilding the level representation, and reversing it in
   this task would change a measurement while reporting it.
3. **E-10 should be scheduled.** A rolling three-row response ring would take the
   frontend from 1.72 MB to about 0.49 MB — **3.5×** — ~~for roughly 2× the response
   compute~~, on a project whose tiebreak is memory.
   **SCHEDULED, RUN AND CLOSED as T3.11 /
   [X-23](#x-23--the-rolling-response-ring-against-the-frame-sized-response-map--done),
   AND THE COMPUTE HALF OF THIS ITEM WAS WRONG — corrected here by name, because
   X-23's pre-registered rule required exactly that of a `T < 1.00` result.** The
   footprint half was right and slightly conservative: measured
   **1 721 568 B → 500 464 B, 3.44×**. The compute is **0.774×, i.e. FASTER**, not 2×
   slower — a ring forces a row-major sweep, which [X-18](#x-18--does-the-incremental-window-form-still-pay-inside-t37s-dense-sweep--done)
   had already measured as the quicker traversal at `blockSize` 3, and no second pass
   is needed to recover the global threshold. Even the two-pass shape this sentence
   described costs **1.327×**, not 2×. The corners are identical.
   (Both figures are X-23's **reported** device run,
   `results/corner_streaming_benchmark_pi4.log`; its second run of the same source
   reads 0.764× and 1.344×. An earlier version of this paragraph quoted the second
   run's numbers while D-22, ARCHITECTURE §9's E-10 row, TASKS.md T3.11 and
   `ops/corner.hpp` all quoted the reported one, so a reader following the named
   correction met two different figures for one decision cell.)
4. **Nothing about `lk_min_eig_threshold` was changed**, although this entry
   measured it to be nearly vacuous over exact popcount covariances on binarized
   content: the weakest window among 141 real-frame keypoints scores 0.033 in the
   reference's units against a 0.001 threshold, i.e. 33× clear, while its ΣIy² is
   10 pixels out of 961. Aperture-limited points therefore survive rejection and
   slide along their edge. Changing a parameter `seal_params.yaml` sets is a
   frontend decision, not a kernel one, and it belongs with T4.3a's outlier
   handling.


---

# Pre-registered — rule recorded first, then measured

The entry below was written **before any benchmark for it existed** and committed
on its own, so the history shows the rule predates the data — the same discipline
[X-9, X-10 and X-11](#phase-2-experiments--rules-recorded-first-then-measured)
were held to at 4245210. Its **Decision rule** is copied **verbatim** from its
[TASKS.md](TASKS.md) task entry and was not re-scaled, re-scoped or softened once
the numbers arrived — which matters here, because the numbers landed partly
outside the bands the rule anticipated and the entry says so rather than bending
one of them to fit.

**That paragraph describes X-21, and X-22 and X-23 have since joined this section.
All three are now closed**, so this section is no longer distinguished from
"Completed" by having an open entry in it — it is distinguished by *how* its entries
were written, which is the property worth keeping visible.
**[X-23](#x-23--the-rolling-response-ring-against-the-frame-sized-response-map--done)
is the strongest instance of the discipline in the log so far**, and it is worth
saying why: its Decision rule was NOT copied from [TASKS.md](TASKS.md) —
[T3.11](TASKS.md#t311--rolling-response-map-e-10--done) deliberately stated only
that a rule must be written first — so the bands, their numbers and the
justification for those numbers were written in the entry itself, at a commit where
`ops/corner.hpp` contained no streaming form at all. **And the rule earned its
keep**: it pre-declared `T < 1.00` as a live outcome requiring three documents to be
corrected by name, that outcome is what happened (`T` = 0.774×), and the correction
was therefore a step in a written procedure rather than an awkward discovery. A rule
that only anticipates the answer you expect is not doing this job.

---

### X-21 · Does generic-N cost the specialized N=1 and ternary paths anything? · `DONE`

> **CORRECTED AND COMPLETED AT TRIAGE.** The first version of this entry was run
> and reported without a committed device log, and three of its claims did not
> survive review. All three are corrected below **in place, with the correction
> named**, because a log that quietly reads differently on a second visit is worse
> than one that was wrong:
>
> 1. **"the same machine code" / "the same 567 aarch64 instructions"** — the
>    derivative pair is the same instruction COUNT (567) and the same function SIZE
>    (2264 B, `nm` to the byte), but the streams are **not identical**: GCC allocates
>    different registers in the row loop. And the claim was only ever about the
>    derivative — the covariance and count functions differ in size and in
>    instruction count. Both are now shown by a committed script rather than asserted.
> 2. **the attribution of the kernel-shape gap to the `a[N]`/`b[N]`/`srcRow[N]`
>    arrays** — measured at triage with a new decomposition point that removes N's
>    array plumbing **and nothing else**. The arrays are a MINORITY of the gap:
>    about a fifth of the per-row cost and a third of the per-word cost. The rest is
>    genericity the entry never separated — runtime `BorderType`, the word type, the
>    argument contract.
> 3. **"2.84× in code size"** — an exceptions-enabled figure, weighed against a
>    Tier 2 claim that rests on `-fno-exceptions`. Measured there at triage: **2.63×**.
>
> Two further claims were checked and **stand**: the hand-written arm is a genuinely
> independent control, and the decision rule was not softened — band 2 fired, was
> reported, and nothing was acted on. One reviewer hypothesis was **rejected on
> evidence**; see conclusion 3. Every number in this entry is reproducible from
> [`bincv-cpp/results/genericn_benchmark_pi4.log`](bincv-cpp/results/genericn_benchmark_pi4.log),
> produced by [`scripts/genericn_evidence.sh`](scripts/genericn_evidence.sh).

**Gates:** [E-4](ARCHITECTURE.md#9-open-questions-and-planned-experiments) · task
[T3.9](TASKS.md) — whether N
stays arbitrary or gets capped, and whether the specialization strategy chosen at
T1.5 is confirmed or revised.

**Numbering, and a correction.** T3.9's Done-when says "logged as **X-6**". That
is **stale**: [X-6](#x-6--is-the-t22-logic-speedup-real--done) is the T2.2
logic-speedup entry, written long before this task came up for scheduling. This
entry therefore takes the **next free X-number, X-21** — X-1…X-20 and X-7b are
taken — exactly as the Phase 2 batch did for the same kind of staleness ("T2.9's
older 'logged as X-4' is likewise superseded — X-4 is taken"). TASKS.md T3.9's
Done-when has since been corrected to X-21, which is the number to quote.

**Question:** Does the bit-sliced generic-N implementation regress the
specialized N=1 and ternary paths?

**Hypothesis:** it should not, and the reason is structural rather than hopeful —
`N` is a compile-time template parameter, so a generic loop over `N` planes is
fully unrolled at `N = 1` and the specialization exists to remove work the
compiler may already have removed. The interesting outcome is therefore the null
result, which under the rule's first band confirms arbitrary N at no cost. The
side this hypothesis does **not** cover is **code size**: three routes that are
speed-equivalent are not size-equivalent, and every extra instantiation is
charged against the constraint [ARCHITECTURE §2](ARCHITECTURE.md#2-target-platforms)
names as *often binding before RAM* on Tier 2. A null on ns/pixel and a
regression on bytes is a live outcome of this experiment, not an unexpected one.

> **The hypothesis was right about the mechanism and wrong about the size.** `N`
> being a compile-time parameter does collapse the plane loop — further than
> predicted: the two routes compile to the *same instructions*, not merely to
> equally fast ones. But the anticipated "null on ns/pixel, regression on bytes"
> did **not** happen between generic-N and the specialization; generic-N's object
> is 90 B **smaller**. The size regression the hypothesis was watching for turned
> up somewhere it was not looking — against the hand-written control, at 2.84×,
> on both binCV routes equally.

**Decision rule** *(written before measuring; verbatim from
[T3.9](TASKS.md))*:
- Specialized paths within 5% of a hand-written binary-only implementation →
  arbitrary N confirmed at no cost to the common cases
- Regression > 5% → report before acting; options are stronger specialization or
  capping N, and which is right depends on where the cost comes from

**Variants — three arms, and the third is the one that makes this meaningful:**
1. `QuantMat<1>` and `TernaryMat` through the **generic** path (`ForceGeneric`);
2. the same content through their **specializations**, i.e. the routes
   `ops/derivative.hpp` and `ops/reduce.hpp` select by default;
3. a **hand-written binary-only reference** — the same operation written with no
   genericity at all, no `N`, no plane loop, no route selection.

Arms 1 and 2 alone would only show whether the specialization is *selected*.
Only arm 3 shows whether genericity *costs* anything, and the rule is written
against arm 3 ("within 5% of a hand-written binary-only implementation"), so
omitting it would leave the rule unevaluable.

**Workload:** [T3.5](TASKS.md)'s derivative and the reductions — [T2.5](TASKS.md)'s
whole-frame `countNonZero` and, because the LK covariance is the reduction the
frontend actually spends its time in, [T3.6](TASKS.md)'s fused `countCovariance`.
T3.9's spec says "T2.5's reductions"; only the count row is a T2.5 kernel, and
**that is stated rather than left for a reader to notice** — four of the six rule
comparisons below are carried by the T3.6 entry point.
At the frame sizes those entries already use so the numbers sit alongside
[X-16](#x-16--t35-derivative-against-cvfilter2d--done) and
[X-7b](#x-7b--the-same-question-on-aarch64-where-d-6-comes-from--done) rather
than beside a workload invented here. Inputs varied and results consumed through
a `volatile` sink, per the Rules above; all three arms compared for identical
output on every image before anything is timed.

**Metric: ns/pixel AND code size — both, and neither alone settles it.**
`size` on the built object, reported per arm, because
[D-2](ARCHITECTURE.md#d-2-bit-planes-over-swar-packing) rests on bit-planes
making "the 1-bit case the base case rather than a special case" and
[ARCHITECTURE §2](ARCHITECTURE.md#2-target-platforms) names code size as often
the binding constraint on Tier 2, before RAM — so a speed result alone cannot
close E-4. Peak working set is identical across the three arms by construction
(same containers, same buffers) and is reported once rather than per arm.

**Platform: the reference device closes this.** Pi 4 / Cortex-A72, via
`scripts/run_on_pi.sh pi4`, with architecture, governor, core pinning and
`vcgencmd get_throttled` **before and after** recorded per the "Measuring on the
Pi 4" rules. **Code size is measured there too, not on x86** — `size` on an
aarch64 object answers a different question from `size` on an x86_64 one, and
aarch64 is the tier the answer is for. An x86 pre-run is a cheap signal only and
cannot close E-4.

**The X-7 / X-7b confound applies and is not to be "fixed".** binCV builds with
no `-march` flags, so `__builtin_popcountll` lowers to `call __popcountdi2@PLT`
per word on x86_64 and to `fmov`/`cnt`/`uaddlv`/`fmov` on aarch64. The reduction
arm is therefore measuring that lowering as much as the design variant under
test. **No `-march` flag and no intrinsics enter the library to settle this** —
baseline-ISA dispatch is an unsettled decision (ROADMAP 2.3), and changing it
mid-experiment would confound precisely the three-arm comparison above.

**Why this is not a tidy-up.** [X-20](#x-20--hybrid-lk-accuracy-against-ground-truth-and-the-frontends-peak-footprint--done)
found that a 1-bit pyramid cannot localise sub-pixel motion, promoting E-7 from
an optimisation to a **precondition**, and recorded that binCV cannot form the LK
covariance at an N-bit level at all today. [T4.1](TASKS.md) must therefore build
N-bit paths that matter for **accuracy**, not only for footprint. Whether
generic-N is cheap is a live question about the next phase's design, not a
retrospective on this one.

**Method:** three arms, **one translation unit each**, committed at `066a339`
**before a single number was taken** — `benchmark/genericn_arms.hpp` (the shared
interface), `genericn_arm_generic.cpp`, `genericn_arm_specialized.cpp`,
`genericn_arm_handwritten.cpp`, driven by `benchmark/genericn_benchmark.cpp`.

The separation is not tidiness. `size` on one object per arm is a number **per
arm** rather than for the sum, which the code-size half of the metric needs; and
[morphology_path_benchmark.cpp](bincv-cpp/benchmark/morphology_path_benchmark.cpp)
records, measured, that two instantiations of one kernel in a single object move
each other's timings by ~10% through code layout alone. The cost — no arm inlines
into the timing loop — is paid identically by all three, one call per frame
against 307 200 pixels of work inside it.

**What the hand-written arm actually is**, since the experiment is worthless if it
is the specialization under another name: it includes **no binCV header**. No
`QuantMat`, no `BinMatView`, no template over `N`, no template over the word type,
no route selector, no `BINCV_ASSERT` contract, no `impl::` helpers —
`BORDER_REFLECT_101` and `uint32_t` are compiled in, and `bitMask`, `rowTailMask`,
`minRowWords`, `borderIndex` and `regionFromExtent` are re-derived inline as the
two or three lines of arithmetic they are. **The word-level arithmetic is equal across the arms — and at triage that was
checked rather than asserted.** The first version of this entry said "there is no
second way to compute `mag = a ^ b`", which is plainly false as prose: the library
spells it `pos = a & ~b; neg = b & ~a; mag = pos | neg` and the hand-written arm
spells it `mag = a ^ b`, the same function in four operations and in two. Whether
those are the same *instructions* is a question about the compiler, so it is now
compiled: § 6 of the committed log builds both spellings at `-O3` and GCC 14.2
emits **`eor` + `bic` for each**, differing only in the operand order of a
commutative `eor`. So the premise holds — with the arithmetic held equal in the
machine code, a difference is the machinery around it — but it holds **because the
compiler folds the spellings**, not because there is one way to write them, and
that is a fact about this toolchain at this optimisation level rather than a
tautology.
Its popcount is deliberately spelled `__builtin_popcountll` exactly as
`impl::popcountWord` is, because X-7/X-7b's no-`-march` lowering is a confound to
**hold fixed**, not to vary; two arms differing in the builtin would be measuring
the builtin.

**Three further measurement points were added afterwards** — `views only` in
`benchmark/genericn_diag.cpp` (`abf3504`), then at triage `scalarized` in the same
file and the accumulator twins in `benchmark/genericn_diag_accum.cpp`. **None is an
arm of the rule comparison** — the rule and the three arms are untouched, and the comparison the
rule is written against was run and recorded before that file existed. It exists
because the rule's second band makes the remedy depend on *where the cost comes
from*, and the headline run localizes nothing. It calls the same kernel the
specialized arm calls, through the public view entry points, with the container
removed, so

```
views_only  − hand_written  =  the kernel's generic SHAPE at N = 1
specialized − views_only    =  the CONTAINER
```

**`scalarized` splits the first of those, and it is the point the original entry
needed and did not have.** The hand-written arm drops *three* kinds of genericity at
once — genericity in **N**, genericity in the **BorderType** (a runtime parameter in
`ops/`, which pays for itself inside the word loop: `derivativeYRoute`'s
`a[p] = haveA ? rowA[p][i] : fill` select exists only so `BORDER_CONSTANT` can be
asked for at run time) and genericity in the **word type** (`B - 1` from
`bitsPerWord<WordType>()` where the hand-written arm writes the literal `31`).
Nothing in the original run separated them, and the entry nevertheless charged the
whole gap to N's array plumbing. `scalarized` is the shipped kernel with the plane
arrays replaced by scalars and the `for p < N` loops deleted, and **nothing else
changed** — same views, same `impl::` helpers, same `BINCV_ASSERT` contract, same
runtime `BorderType`, same template over `WordType`, same `ternaryDifference`. So

```
scalarized − hand_written  =  every OTHER kind of genericity
views_only − scalarized    =  N's array plumbing, alone
```

It is also exactly the remedy this entry's Decision Q3 named as a candidate, so
measuring it costs one decomposition point and says whether that remedy is worth an
experiment at all.

**The accumulator twins** exist because conclusion 3 originally explained the
library's count *win* by `impl::visitRowWords`' head/interior/tail skeleton — which
the hand-written count already has, running its interior unmasked and masking only
the tail. The variable the original run never isolated is D-15's **per-row partial
sum**: `impl::countRowRegion` returns a value that `countViewRegion` adds, so a
640×480 count is 480 independent popcount chains against the hand-written arm's one
9600-long chain, and X-11b measured that shape separately at 1.03–1.09× on a target
where popcount latency is the bottleneck (D-6). `genericn_diag_accum.cpp` holds two
pairs of twins — a count and a covariance, each in a one-chain and a per-row
spelling, identical in every other character — **both twins of a pair in one object**,
so the A/B cannot be a code-layout artefact, and the one-chain copies double as a
layout control against the arm they were copied from

**Reproducing it — one command, and a committed log.** The original entry gave
four shell commands and described the disassembly comparison in prose, with no
script and no diff output in the tree; every other reference-device entry in this
file cites a committed `results/*_pi4.log`, and this file twice records an entry
being faulted for the omission ([X-6](#x-6--is-the-t22-logic-speedup-real--done),
[X-8](#x-8--what-composing-the-lk-covariance-out-of-t26-costs--done)). It is
repaired the same way:

```
./scripts/run_on_pi.sh pi4 '../../scripts/genericn_evidence.sh'
```

[`scripts/genericn_evidence.sh`](scripts/genericn_evidence.sh) emits, in one
stream, the benchmark, `size`, `size -A`, `nm -C -S`, the address-stripped
instruction-identity diff for **all three** generic/specialized function pairs, the
hand-written instruction counts, the arithmetic-spelling check, and the
`-fno-exceptions` sizes — and writes them to
[`bincv-cpp/results/genericn_benchmark_pi4.log`](bincv-cpp/results/genericn_benchmark_pi4.log),
which is committed. Its normalisation is documented in the script and removes only
the instruction address, the redundant hex branch target, and the arm's name inside
a symbol; a differing register, mnemonic or within-function offset survives and
diffs — which is how the correction to conclusion 1 was found.
**Code size and the disassembly are taken on the device, not on x86** — `size` on an
aarch64 object answers a different question.

**Validity.** Volatile sink; four rotated pseudo-random frames; padding kept clear
so no arm over-counts; and all three arms **and both decomposition points**
required to produce bit-identical `dx`/`dy` buffers and identical counts and
covariances on every input before anything is timed — a disagreement prints as a
defect and nothing is timed. Batches are interleaved round-robin. The three
destination buffers are re-filled from one input before the covariance rows,
because popcount cost is content-dependent and the derivative rows leave each
buffer holding whichever frame ran last.

**Environment** (the committed triage run, whose full output is
[`bincv-cpp/results/genericn_benchmark_pi4.log`](bincv-cpp/results/genericn_benchmark_pi4.log);
the two earlier runs at `066a339` and `abf3504` were taken on the same device with
the same block and agree to within 0.5%):

```
device:           pi4
cpu:              Raspberry Pi 4 Model B Rev 1.5
arch / kernel:    aarch64 / 6.18.34+rpt-rpi-v8
compiler:         g++ (Debian 14.2.0-19) 14.2.0
governor:         performance (restored to ondemand on exit)
core pinning:     taskset -c 3
throttled before: throttled=0x0
throttled after:  throttled=0x0
build:            Release, core-only, no -march (X-7 confound held fixed)
```

Throttle clean before **and** after, and unchanged across the run. The headline
rows have now been measured **three** times — `066a339`, `abf3504`, and the triage
run whose log is committed — across three different binaries, and reproduce to
**within 0.5% worst case**, on the specialized covariance row at 640×480 (1.0045,
1.0093, 1.0068); every other row agrees to within 0.15%. The first version of this
entry claimed 0.1%, which its own two tables already contradicted on that row. 0.5%
is still inside the 0.1–1.3% batch spread and no band moves, but the smaller
number was not true and is not what a later reader should plan a re-run against.
The arm most sensitive to a rebuild is the one whose object layout moves when a
decomposition point joins `genericn_arms` — the ~10% layout effect
`morphology_path_benchmark` records, showing up here as half a percent.

---

**Result — speed. ns/pixel, median of 15 batches, `uint32_t`, Pi 4.**

Batch spread `(max − min)/median` was **0.0–0.4%** on every row of the committed
triage run (0.1–1.3% on the two earlier runs). The original entry called every
difference "an order of magnitude outside it"; against the worst spread of 1.3%
the real margins are **4.5× to 33×** — the 94×60 count row at 5.8% is 4.5× and the
640×480 covariance at 7.6% is 5.8×, not 10×. Both are still far outside it and no
band moves, but the overstatement is corrected rather than kept.

*640×480 (stride 20 words):*

| workload | hand-written | specialized | generic-N | spec ÷ hand | **generic ÷ spec** |
|---|---|---|---|---|---|
| derivative dx+dy | 0.1704 | 0.2060 | 0.2061 | **1.209×** | **1.000×** |
| count whole frame | 0.0723 | 0.0637 | 0.0636 | **0.881×** | 0.998× |
| covariance 31×31 ×200 | 0.9359 | 1.0068 | 1.0059 | **1.076×** | 0.999× |

*94×60 — pyramid level 3, the level LK touches every frame (stride 3 words):*

| workload | hand-written | specialized | generic-N | spec ÷ hand | **generic ÷ spec** |
|---|---|---|---|---|---|
| derivative dx+dy | 0.2508 | 0.3579 | 0.3580 | **1.427×** | **1.000×** |
| count whole frame | 0.0897 | 0.0844 | 0.0846 | **0.941×** | 1.002× |
| covariance 31×31 ×200 | 0.6592 | 0.7314 | 0.7321 | **1.109×** | 1.001× |

**Result — code size. `size` on one aarch64 object per arm.**

| arm | text | data | dec |
|---|---|---|---|
| hand-written | 1809 | 32 | 1841 |
| specialized | 5144 | 40 | 5184 |
| generic-N | 5054 | 40 | 5094 |

| comparison | text (default build) | text (`-fno-exceptions`) |
|---|---|---|
| hand-written | 1809 | 1809 |
| specialized | 5144 | 4767 |
| generic-N | 5054 | 4677 |
| specialized ÷ hand-written | **2.84×** | **2.63×** |
| **generic-N ÷ specialized** | **0.98× — generic-N is 90 B SMALLER** | 0.98× |

**The right-hand column is the one the Tier 2 constraint is about, and the original
entry did not measure it.** [ARCHITECTURE §2](ARCHITECTURE.md#2-target-platforms)
names code size as often binding before RAM on Tier 2, and the Tier 2 claim rests
on the core-only `-fno-exceptions` build `verify.sh` already builds — which emits
none of the `.gcc_except_table`, message strings or `.eh_frame` the split below
attributes 972 B to. Measured there, the gap is **2.63×**, not 2.84×. The
hand-written object is byte-for-byte unchanged between the two configurations,
having no throw site; both binCV arms lose ~377 B. Every quotation of this figure
must carry which build it came from.

Per function, `nm -S` on the same objects (bytes):

| function | hand-written | specialized | generic-N |
|---|---|---|---|
| derivative dx+dy | 568 | **2264** | **2264** |
| covariance window | 812 | 1256 | 1216 |
| count whole frame | 184 | 344 | 320 |
| `QuantMat<1>` wrapping ctor, out of line | — | 380 | 380 |

`size -A` on the same objects splits the container arms' extra bytes: `.text`
1580 (hand) against 3876 (specialized) and 3812 (generic), plus **972 B of
plumbing the hand-written arm has no throw site to need** — the out-of-line
`QuantMat` constructor (380 B), `.gcc_except_table` (116 B), its `.rodata`
message strings (236 B) and 240 B more `.eh_frame`.

**Result — memory.** Peak working set is **identical across the three arms by
construction** and is therefore reported once, not per arm: the hand-written arm
indexes the *same* `SignedQuantMat<1>` buffer layout the containers name. At
640×480, `uint32_t`: source 38 400 B + `dx` 76 800 B + `dy` 76 800 B = **192 000 B**
for all three.

**Result — the decomposition, i.e. where the gap actually is.** The `scalarized`
column is the triage addition; without it the two right-hand columns are one
undifferentiated "kernel shape", which is what the original entry reported.

| workload / size | hand | scalarized | views only | specialized | **not-N genericity** | **N's arrays** | **container** |
|---|---|---|---|---|---|---|---|
| derivative, 640×480 | 0.1707 | 0.1894 | 0.1999 | 0.2062 | **+11.0%** | **+6.2%** | +3.7% |
| derivative, 94×60 | 0.2508 | 0.3239 | 0.3498 | 0.3579 | **+29.1%** | **+10.3%** | +3.2% |
| covariance, 640×480 | 0.9357 | — | 0.9744 | 1.0062 | \*+4.1% | \*combined | +3.4% |
| covariance, 94×60 | 0.6592 | — | 0.6993 | 0.7315 | \*+6.1% | \*combined | +4.9% |

The three right-hand columns are **percentage points of the total gap**, not
ratios — they sum to `specialized ÷ hand − 1` — and the original entry printed
them with a `%` sign and then quoted them in prose as ratios. \*The covariance rows
have no `scalarized` point, so their kernel-shape column is still the two effects
combined; only the derivative is split.

Solving the two derivative sizes as `t_row = a·words + b` — **an exact two-point
fit, not a validated model: two equations, two unknowns, no residual to check, and
a third frame size is what would test it.** This caveat travels with the numbers
wherever they are quoted:

| | per word | per row |
|---|---|---|
| hand-written | 5.04 ns | 8.46 ns |
| `scalarized` (all genericity but N) | 5.34 ns | 14.43 ns |
| `views only` (+ N's arrays) | 5.59 ns | 16.10 ns |
| specialized (+ container) | 5.78 ns | 16.29 ns |
| **total ratio** | **1.15×** | **1.93×** |
| of which N's arrays | +5.0 pts | +19.8 pts |
| of which everything else generic | +6.0 pts | **+70.6 pts** |
| of which the container | +3.8 pts | +2.2 pts |

**Result — the accumulator, isolated.** Two pairs of twins differing only in where
the popcount sum lands, both twins of a pair in one object:

| workload / size | one chain | per row (D-15) | ratio | the arm it was copied from |
|---|---|---|---|---|
| count, 640×480 | 0.0722 | 0.0722 | **1.000×** | 0.0723 (layout control agrees to 0.1%) |
| count, 94×60 | 0.0895 | 0.1917 | **2.141×** | 0.0897 (agrees to 0.2%) |
| covariance, 640×480 | 0.9402 | 0.8959 | **0.953×** | 0.9357 (agrees to 0.5%) |
| covariance, 94×60 | 0.6441 | 0.6022 | **0.935×** | 0.6592 (agrees to 2.3%) |

**Bound check.** The whole-frame count reads 37.5 KiB at **1.73 GB/s** — roughly a
third of the Pi 4's ~4–6 GB/s DRAM figure and far below L2, on a frame that fits
in the 1 MiB L2 but not the 32 KiB L1D. So the count is popcount-throughput bound,
not memory bound, exactly as X-7b predicts for a no-`-march` build, and nothing
here is above a physical bound — which is what a dead-code-eliminated loop would
have looked like.

---

**Conclusion**

**1. The question E-4 registered has a clean null answer — and the answer is
"same size, same instruction count, same time", not "the same code".** At N = 1 the
generic route and the specialization produce a derivative of **2264 B in both
objects, `nm` to the byte, and 567 instructions each**, and they time to within
0.1%. They are **not** the same instruction stream: the address-stripped diff in
§ 4 of the committed log shows GCC allocating different registers through the row
loop (`mov x3, x17` against `mov x2, x17`, and the stores that follow). The first
version of this entry said every differing line was "a symbol name or a branch
target offset"; that was read off a comparison this entry never committed, and it
is wrong. The claim was also **only ever true of the derivative** — the same entry's
own `nm` table showed the covariance at 1256 B against 1216 B and the count at
344 B against 320 B, which the log now confirms as 314 against 304 instructions and
86 against 80. Those two functions time within 0.2%, inside the batch spread.

What survives is the finding, in its correct and slightly weaker form:
**bit-sliced generic-N does not regress the specialized N = 1 and ternary paths.
Same bytes, same instruction count, same time, and generic-N's whole object is
90 B smaller.** `N` being a compile-time parameter collapses the plane loop exactly
as the hypothesis said. **N is not capped**, and nothing in this measurement
argues for capping it.

The uncomfortable half stands too: `impl::signedDifference`'s `if constexpr (N == 1)`
branch — the ternary spelling the specialization exists for — **buys nothing
measurable**. GCC 14.2 at `-O3` already reduces the N-generic ripple to it. The
specialization is not harmful; it is, on this evidence, redundant. It is also what
`Derivative.RoutesAgree_*` compares against, so removing it would remove a test.

**2. The rule's second band fired, on four of six rows, and neither remedy it names
fits the cause — but the cause is not what this entry first said it was.**
Specialized against hand-written: **+20.9%** and **+42.7%** on the derivative,
**+7.6%** and **+10.9%** on the covariance; on code size **2.84×** with exceptions
and **2.63×** without. Capping N cannot recover a nanosecond of a gap the N-generic
and N-specialized routes pay equally, and strengthening the specialization cannot
either. That part is unchanged.

**What changed is the attribution.** The original entry named the per-plane arrays
`a[N]`, `b[N]`, `m[N]`, `srcRow[N]`, `magRow[N]`, `prev[N]` and the `for p < N`
loops as the cost, on an argument rather than a measurement — and the argument had
a hole its own conclusion 1 should have made visible: if those arrays were the
cost, and the N-generic and N-specialized routes are byte-identical *because
`ForceGeneric` never touches them*, then the arrays are common to both arms and no
comparison in the run could isolate them. The `scalarized` point isolates them, and
they are a **minority** of the gap:

* **everything generic EXCEPT N — +11.0 points at 640×480 and +29.1 at 94×60**,
  which is 64% and 74% of the kernel-shape gap. Runtime `BorderType`, the word
  type, the `BINCV_ASSERT` contract, and the view structs, **not separated from one
  another here** — three candidates, one measurement, and this entry does not
  pretend to rank them;
* **N's array plumbing — +6.2 points and +10.3 points.** Real, and worth about a
  fifth of the per-row cost. This is Decision Q3's candidate remedy, now sized:
  it is worth having, and it is not the fix;
* **the container — a flat 3.2–4.9%** at both sizes and both workloads. Unchanged.

**3. Two rows land outside BOTH bands, the rule has no branch for them, and the
reviewer's explanation for them is REJECTED on evidence.** The whole-frame count
has the library **11.9% and 5.9% FASTER** than the hand-written control. The
original entry credited `impl::visitRowWords`' head/interior/tail skeleton; review
objected that the hand-written count already has that skeleton and that the real
untested difference is D-15's per-row partial sum, worth 1.03–1.09× at X-11b.
**Measured, the accumulator is not it.** Isolated in a twin pair, the per-row form
is *exactly* even at 640×480 (1.000×) and **2.141× SLOWER** at 94×60 — the wrong
sign, by a wide margin, on the row the library wins. A variable that costs time
cannot explain a win.

So the count win is **real, reproduced across three runs, and unexplained**. What
is known: the library issues *more* instructions for it — 86 against the
hand-written arm's 47 — and is faster anyway, which on a target where the popcount
is a four-instruction NEON round trip (X-7b, D-6) points at scheduling and
dependency structure rather than at work done. Both the skeleton hypothesis and the
accumulator hypothesis are now measured or eliminated, and neither is the answer.
Recorded as an open observation rather than rounded into the nearer band. **The
generic machinery is not uniformly a tax** — on the operation D-6 is built around,
it is a win, whatever the mechanism.

**3b. The covariance regression was UNDERSTATED, and the correction makes it
larger.** The hand-written covariance carries its four sums across all 31 window
rows where the library builds a fresh `CovarianceCount` per row, and the twin pair
measures that shape at **0.953× and 0.935×** — so the control is ~5–6% slower than
a hand-written kernel written with the library's accumulator. Against that fairer
control the library sits **+12.3%** and **+21.5%** rather than +7.6% and +10.9%.
Band 2 fired either way; the point of recording it is that **the direction of every
correction in this triage is against the library**, not for it.

**4. The size dependence is the finding with teeth for Phase 4, and the split
sharpens it.** The derivative gap **doubles** from 640×480 to 94×60, and all of the
growth is in the kernel shape (17.2 → 39.4 points), none in the container
(3.7 → 3.2). The two-point fit — **exact, unvalidated, two equations and two
unknowns** — puts it at **+15% per word but +93% per row**, and now says which
genericity: **+70.6 of those 93 points are the not-N genericity, +19.8 are N's
arrays, +2.2 the container.** A pyramid level 3 row is 3 words, so a per-row cost
is paid 640×480/94×60 = 5.4× more often per pixel up there. **The frames that pay
most are the upper pyramid levels, which LK touches every frame** — the same levels
[X-20](#x-20--hybrid-lk-accuracy-against-ground-truth-and-the-frontends-peak-footprint--done)
found cannot localise sub-pixel motion, promoting
[E-7](ARCHITECTURE.md#9-open-questions-and-planned-experiments) from an optimisation
to a precondition. [T4.1](TASKS.md)'s N-bit paths run on exactly these levels.

**5. What this does NOT license.**

* **Band 1's sentence is not claimed.** The rule was written for a null and this is
  not one. Against *binCV's own specialized path* arbitrary N is free, provably.
  Against *code with no genericity at all* the common cases pay 8–43% in time and
  2.6–2.8× in size, and that is a cost to the common cases whatever its cause.
* **This is an N = 1 result and only an N = 1 result.** Every arm here is
  `QuantMat<1>` / `TernaryMat`, because that is the question E-4 asked. It says
  nothing about what N = 3 or N = 5 costs — and the shapes differ in kind, not only
  in degree: the derivative's ripple-borrow work is linear in N, while §7.5's
  bit-sliced covariance contributes plane **pairs** and is quadratic. **"Generic-N
  is free" means "free against the specialization at N = 1". It does not mean
  N-bit levels are free**, and T4.1 must not read it that way.
* **The three genericity axes are not separated.** +11.0 and +29.1 points are
  charged to "runtime `BorderType`, word type, contract and views" as a group. Which
  of them dominates is unmeasured, and the fix for one is not the fix for another.

**Decision:** **E-4 is RESOLVED on the question it asked, and the larger cost it
uncovered is re-registered as its own open question rather than left in a log.**
The rule's second band says *report before acting*; the report was made, nothing
was acted on, and this triage did not act either — it measured what the report
said it could not localize, and the localization moved the remedy.

1. **E-4 → resolved. `N` stays arbitrary; no cap.** The question was "does
   bit-sliced generic-N ever regress the specialized N = 1 and ternary paths", and
   the answer is no, at the strongest resolution this project can measure: same
   function size to the byte, same instruction count, same time, smaller object.
   Promoted to **[D-21](ARCHITECTURE.md#d-21-generic-n-is-not-capped-and-the-n--1-specialization-is-kept-as-a-test-oracle)**,
   which also amends **[D-2](ARCHITECTURE.md#d-2-bit-planes-over-swar-packing)**:
   D-2's "1-bit case | the base case" cell was an argument, and it is now a
   measurement.
2. **The `N == 1` specialization is KEPT, on a test-oracle argument rather than a
   speed one** (Q1). It is measured redundant for speed and costs 90 B; what it
   buys is `Derivative.RoutesAgree_*`, which compares two independent formulations
   of the same operation and would otherwise compare a route against itself. That
   is a cheap oracle for a bit-parallel kernel, and D-21 records the reason so a
   later reader does not delete it as dead weight.
3. **The code-size figure is 2.63×, not 2.84×, wherever the Tier 2 constraint is
   what is being weighed** (Q2) — measured at triage in the core-only
   `-fno-exceptions` configuration. Both numbers are recorded and each carries its
   build. No decision is taken on whether 2.63× is acceptable on Tier 2: it is a
   ratio against a control that supports one border and one word width, which is
   not the library binCV is, and turning it into a budget needs a target and a
   denominator this experiment does not have.
4. **The kernel-shape cost becomes [E-12](ARCHITECTURE.md#9-open-questions-and-planned-experiments),
   gated on [T4.1](TASKS.md)** (Q3). E-4 could not close on it — it is a different
   question from the one E-4 asked, and closing E-4 on it would have filed the
   finding under the wrong heading, which is why the first version of this entry
   left the row open. Registering it fixes that properly: **+93% per row against a
   hand-written control, of which +70.6 points is genericity that is not in N**,
   worst on the upper pyramid levels E-7 and T4.1 both live on. Its pre-registered
   rule is E-12's to write, not this entry's; what this entry hands it is the
   sizing — scalarizing N's arrays recovers about a fifth, so the experiment worth
   running is the one that separates `BorderType`, word type and contract, and
   whichever it finds is the fix.
5. **The count win goes with it as an open observation**, not as a claim: the
   library beats a hand-written control by 5.9–11.9% on the whole-frame count with
   *more* instructions, both proposed explanations are eliminated, and it reproduces
   across three runs on the reference device.

**What did NOT change, and was checked because it is the failure that would matter
most.** The hand-written arm is a genuinely independent control — it includes no
binCV header, and its derivative is 142 instructions against the library's 567, a
different function by any measure. The decision rule is the one written in
TASKS.md T3.9 before any code existed, quoted verbatim, with no threshold moved:
band 2 fired at 5%, and every correction made here moved a number **away** from the
library (the covariance regression grew from +7.6/+10.9% to +12.3/+21.5%, the
"same code" claim weakened, the "0.1% reproduction" claim weakened). Nothing was
softened to fit.

---

### X-22 · What an N-bit pyramid level costs the LK covariance · `DONE`

**Gates:** [T3.10](TASKS.md) — the price
[T4.1](TASKS.md) / [E-7](ARCHITECTURE.md#9-open-questions-and-planned-experiments)
has to weigh a per-level bit depth against · spawned
[E-13](ARCHITECTURE.md#9-open-questions-and-planned-experiments)
**Question:** How much does the bit-sliced covariance of
[§7.5](ARCHITECTURE.md#75-lk-gradient-covariance) cost per LK window at N = 1, 2, 3
and 4, on the reference device?

**Why it exists.** [X-20](#x-20--hybrid-lk-accuracy-against-ground-truth-and-the-frontends-peak-footprint--done)
found the tracker's accuracy failure **is the 1-bit pyramid** — on windows that
never clip, four 1-bit levels are still ~600× worse than one — and
[X-2](#x-2--pyramid-bit-growth--done) had measured the levels needing 1/3/4/5 bits.
So T4.1 has to choose a bit depth per level, and a choice needs a price. Until
T3.10 there was no N-bit covariance to price.

**Hypothesis:** the cost is **quadratic in N**, because a product of two N-bit
values is a sum over plane PAIRS. That is inherent, not an implementation choice:
anything linear in N computes a different quantity.
[D-21](ARCHITECTURE.md#d-21-generic-n-is-not-capped-and-the-n--1-specialization-is-kept-as-a-test-oracle)
closed E-4 for **N = 1 only** and flagged exactly this asymmetry — the derivative
is linear in N where the covariance is quadratic — so "generic-N is free" must not
be read onto this table.

**The cost model, written out before measuring.** Counting the popcounts the
shipped kernel issues per word, with the two diagonal entries counted on the upper
triangle and doubled:

```
N(N+1)/2  for sumXX  +  N(N+1)/2  for sumYY  +  2N^2 for sumXY  =  3N^2 + N
N = 1: 4     N = 2: 14     N = 3: 30     N = 4: 52
ratio: 1.00        3.50          7.50          13.00
```

At N = 1 that is exactly `countCovariance`'s four popcounts, which is the
arithmetic statement of "ternary is the N = 1 instance".

**Decision rule** *(written before measuring, in
`bincv-cpp/benchmark/covariance_nbit_benchmark.cpp`'s header)*. **Nothing here
chooses between two implementations**, so the rule is a falsifiable prediction
about the curve rather than a selection between arms:
- **Band A — ratios within ±25% of 1.00 / 3.50 / 7.50 / 13.00:** the popcount count
  IS the cost model; T4.1 may price a bit depth with `3N² + N` and no code moves.
- **Band B — ratios systematically BELOW:** the kernel is not purely popcount-bound
  (the N² pairs come off 2N+2 loads, so there is ILP the count does not model).
  Report the measured curve as the price and mark `3N² + N` an upper bound.
- **Band C — ratios ABOVE:** something is quadratic that should not be. That
  contradicts the shipped kernel's documented cost and
  [CLAUDE.md](CLAUDE.md)'s rule applies — report it, do not adjust the doc.

**Variants:** the T3.6 ternary entry point; the T3.10 bit-sliced entry point at
N = 1, 2, 3, 4; and — added after the first run, see the caveat — the same
bit-sliced kernel with its **per-row partial accumulator replaced by a
window-wide one**, which is the shipped code with exactly one thing changed.
**The N = 1 bit-sliced arm is the same kernel as the N = 2..4 arms**, not the
ternary one, so the ratio column is one kernel's curve rather than a change of
kernel at the first column.
**Workload:** 640×480, 200 keypoints (the reference pipeline's
`gftt_max_corners`), one window each, scattered so border windows clip;
W = 7, 15, 31; `uint32_t` and `uint64_t`. The same frame, keypoint count and
window generator [X-11](#x-11--incremental-versus-recomputed-window-reductions--done)
and X-17 use.
**Metric:** ns per window (median of 11 interleaved batches, spread reported),
**and bytes per level beside it** — an N-bit level is (N+1) bits per pixel per
derivative against ternary's 2, which is the other half of the trade.
**Method:** `benchmark/covariance_nbit_benchmark.cpp` through
`measure_util.hpp`'s protocol. Every arm's ANSWER is checked against a per-pixel
reference at every timed window before anything is timed. Log:
[`bincv-cpp/results/covariance_nbit_benchmark_pi4.log`](bincv-cpp/results/covariance_nbit_benchmark_pi4.log).

**Result — reference device, `./scripts/run_on_pi.sh pi4 ./benchmark/covariance_nbit_benchmark`,
ns per window, W = 31, `throttled=0x0` before and after:**

| arm | uint32_t ns | vs N=1 | uint64_t ns | vs N=1 | predicted | bits/px/deriv |
|---|---|---|---|---|---|---|
| ternary (T3.6) | 977.0 | 1.08× | 797.4 | 1.06× | 1.00× | 2 |
| bit-sliced N=1 | 903.3 | 1.00× | 754.7 | 1.00× | 1.00× | 2 |
| bit-sliced N=2 | 3186.7 | 3.53× | 3161.1 | 4.19× | 3.50× | 3 |
| bit-sliced N=3 | 5906.8 | 6.54× | 6803.4 | 9.01× | 7.50× | 4 |
| bit-sliced N=4 | 11023.1 | 12.20× | 11719.8 | 15.53× | 13.00× | 5 |

The footprint column is flat in W: 153 600 B at N = 1 against 384 000 B at N = 4
for both derivatives of one 640×480 level.

Spreads are **0.4–4.1%** on every row of the quoted run (run 3, the same binary,
reaches 5.0%), so the differences above are far outside the noise they were
measured against.

**THE BAND VERDICT IS PER CELL, NOT PER WORD TYPE — and the first version of this
entry got it wrong in BOTH directions.** The pre-registered band is ±25% of
1.00 / 3.50 / 7.50 / 13.00, and applying that arithmetic to every measured cell of
the quoted run gives:

| ratio vs N=1 | `uint32_t` N=2 | N=3 | N=4 | `uint64_t` N=2 | N=3 | N=4 |
|---|---|---|---|---|---|---|
| W = 7  | **4.76× (C, +36%)** | 7.22× (A) | 12.96× (A) | **4.89× (C, +40%)** | **9.68× (C, +29%)** | 15.69× (A) |
| W = 15 | 4.06× (A) | 6.61× (A) | 12.02× (A) | **4.52× (C, +29%)** | 9.24× (A) | 15.31× (A) |
| W = 31 | 3.53× (A) | 6.54× (A) | 12.20× (A) | 4.19× (A) | 9.01× (A) | 15.53× (A) |

Two corrections follow, and both are corrections to this entry rather than to the
kernel:

1. **"Band A holds at `uint32_t` at every window size and every N" was false.**
   `uint32_t` / W = 7 / N = 2 is 4.76× against 3.50× predicted — +36%, past band A's
   upper edge of 4.375× and squarely in **band C**, whose pre-registered rule is
   "something is quadratic that should not be … report it, do not adjust the doc."
   It reproduces inside the binary (4.65×, run 3), so it is not noise.
2. **"It does not hold at `uint64_t`: N = 3 at +20% and N = 4 at +19%" was also
   false, in the opposite direction.** ±20% is *inside* ±25%. Those W = 31 cells
   are band A by the rule this entry wrote down before measuring, and calling them
   band C was fitting the verdict to an impression instead of to the rule. The
   genuine `uint64_t` band-C cells are at the SMALL windows: W = 7 at N = 2 and
   N = 3, and W = 15 at N = 2.

**WHAT THE BAND-C CELLS ARE, AND WHY THEY ARE NOT ATTRIBUTED HERE.** Every one of
them is at W = 7 or W = 15 and concentrated at N = 2 — the corner where a window is
1–3 words per row, so the per-window and per-row FIXED costs (the clip ladder, the
row prologue, the 4N² per-row counters E-13 is registered against) are largest
relative to the word work the model counts. They are also the cells the code-layout
effect below moves most: the same `uint32_t` / W = 7 / N = 2 cell reads **3.27×**
in the five-arm binary, i.e. band A. So the band-C readings are **reported and left
open**, not explained — the same treatment the `uint64_t` crossover gets, and for
the same reason.

What survives all of it and does not depend on a band label: at W = 31 the 64-bit
word is **SLOWER in absolute terms than the 32-bit one at N = 4** (11 720 ns
against 11 023 ns) after being faster at every N below. That crossover is the
finding with the most consequence for [E-9](ARCHITECTURE.md#9-open-questions-and-planned-experiments)
and for T4.1, which will be choosing bit depth and word width on the same levels.

**THE OBVIOUS EXPLANATION IS WRONG, AND IT WAS MEASURED RATHER THAN ASSUMED.**
Register pressure — 2N = 8 live magnitude words at N = 4, a 64-bit word being a
whole GPR where two 32-bit words are not — predicts spills at exactly that corner.
[`scripts/covariance_nbit_codegen.sh`](scripts/covariance_nbit_codegen.sh) compiles
the kernel out of line per (N, word type) on the device and counts stack traffic in
its own instruction stream
([log](bincv-cpp/results/covariance_nbit_codegen_pi4.log)):

| word | N=1 | N=2 | N=3 | N=4 |
|---|---|---|---|---|
| `uint32_t` stack ld/st | 20 | 60 | 69 | 69 |
| `uint64_t` stack ld/st | 15 | 60 | 69 | 70 |

Identical within one instruction, and the instruction counts within 2% (399 against
406 at N = 4). **The hypothesis is rejected**; whatever `uint64_t` is paying at
N ≥ 3 is not extra spill code. What remains is that a 31-pixel window is 1–2
`uint64_t` words per row against 2–3 `uint32_t` words, so per-ROW costs are
amortized over less work — which is what the next paragraph is about, and which
this entry does **not** close.

**THE PER-ROW ACCUMULATOR IS O(N²) PER ROW, AND THAT IS AN OPEN OBSERVATION RATHER
THAN A RESULT.** The shipped kernel gives each row its own `BitSlicedPairCounts<N>`
and folds it in — the per-row-partial shape T2.11 item 4 adopted on measurement at
N = 1 (X-11b: 1.08× at W = 31). At N = 4 that is 4N² = 64 counters zeroed and 64
added **per row**, independent of how many words the row has. The window-accumulator
arm removes exactly that and nothing else:

| W = 31 | uint32_t shipped | uint32_t window-acc | uint64_t shipped | uint64_t window-acc |
|---|---|---|---|---|
| N=1 | 903.3 | 898.1 | 754.7 | 766.0 |
| N=2 | 3186.7 | 2739.8 | 3161.1 | 2323.7 |
| N=3 | 5906.8 | 4835.7 | 6803.4 | 4266.2 |
| N=4 | 11023.1 | 8791.2 | 11719.8 | 8431.8 |

**That is 1.14–1.60× and it is NOT claimed as a result**, for a reason this entry
found the hard way and reports rather than hides. See the caveat.

**THE CAVEAT, AND IT LIMITS EVERY ABSOLUTE NUMBER ABOVE.** The first run of this
benchmark had five arms — the shipped kernel only. Adding the four
window-accumulator arms to the same translation unit **moved the shipped arms'
timings, with no change to their source**: `uint64_t`, N = 3, W = 31 went from
4652.6 ns to 6803.4 ns (**1.46×**), and its ratio column from 6.06× to 9.01×.
A third run of the *same binary* as the second reproduces it to ~1% on every row,
so this is **code layout between binaries, not run-to-run noise** — the effect
`benchmark/morphology_path_benchmark.cpp` already recorded at ~10%, here at up to
1.46×. All four runs are in the committed log with the caveat at the top.

Three consequences, and they are the honest reading:
1. **Within a binary this benchmark is reproducible to ~1%; between binaries the
   same kernel's cost moves by up to ~1.5×.** T4.1 must re-measure in its own
   binary rather than quote a number from this one.
2. **The `uint64_t` band-C reading is not safe to attribute.** It is present in the
   nine-arm binary and absent in the five-arm one (6.06× at N = 3 there, inside
   band A). **And it is not only `uint64_t` that moves**: the `uint32_t` /
   W = 7 / N = 2 cell goes 3.27× → 4.76× between the two binaries, a 1.46× swing
   that crosses the band boundary — the same magnitude this caveat first attributed
   to `uint64_t` alone. What is stable across both binaries is `uint32_t` at
   **W = 15 and W = 31**, inside band A in both (though N = 2 still moves 1.18–1.31×
   between them). Nothing at W = 7 is stable.
3. **The window-accumulator comparison is confounded by the same effect** and
   cannot be settled here: it is an interleaved within-binary comparison, and
   interleaving controls for machine state, not for how the two arms' code was
   laid out. The part that survives both binaries is `uint32_t` at N = 4, where the
   shipped arm measured 11 393 ns (five-arm binary) and 11 023 ns (nine-arm)
   against the window-accumulator's 8791 ns — a gap larger than the layout drift
   between the two shipped readings.

**RUN 4 — THE SHIPPED KERNEL AFTER TRIAGE, RE-MEASURED SO THIS TABLE DESCRIBES IT.**
Triage removed dead work from `impl::BitSlicedPairCounts<N>::add()`: it folded the
full N × N of `xx` and `yy` per window ROW, where the row body only ever writes the
upper triangle and the combine only ever reads it — N² − N adds per row with two
provably-zero operands, 12 of 64 at N = 4. The answers are bit-identical by
construction. The benchmark was re-run on the device (`throttled=0x0` before and
after) so that no number here describes code that is no longer shipped:

| W = 31, `uint32_t` | N=1 | N=2 | N=3 | N=4 |
|---|---|---|---|---|
| run 2 (pre-triage) | 903.3 | 3186.7 | 5906.8 | 11023.1 |
| run 4 (shipped) | 895.7 | 2929.7 | 6188.5 | 10551.3 |

**Run 4 is NOT an A/B of that change and must not be read as one** — it is a fourth
binary, so the 1.46× layout effect applies in full, and it moves in both directions
(−4.3% at N = 4, **+4.8% at N = 3**), which a removal of dead work cannot do. What
run 4 is good for is confirming the curve and the corrected band verdicts, and it
does: the only band-C cells are again W = 7 at N = 2 (+26% at `uint32_t`, +30% at
`uint64_t`); `uint64_t` at W = 31 is inside the band at every N (4.03× / 8.47× /
15.18×), as the pre-registered rule says; and the ternary arm is again slower than
the generic arm at N = 1 in all six of its rows. Pricing the per-row accumulator
itself is still [E-13](ARCHITECTURE.md#9-open-questions-and-planned-experiments)'s
job and still needs a binary per arm.

**Conclusion.**
1. **The covariance is quadratic in N and the quadratic is the popcount count.**
   At the shipped word width and the LK window size the measured curve is 3.5×,
   6.5×, 12.2× at N = 2, 3, 4 against a predicted 3.50×, 7.50×, 13.00× — band A,
   and inside it on the low side at N = 3 and 4, which is band B's direction.
   **`3N² + N` is a good model and a slight over-estimate AT W = 15 AND W = 31,
   and an under-estimate at N = 2 on a 7×7 window** (+36%, band C, above). A price
   quoted from this entry has to carry the window size it was measured at; T4.1's
   interpolation to N = 5 is licensed at W = 15 and W = 31 and not below.
2. **The price of the fix X-20 called for, stated plainly.** Going from a 1-bit to
   a 4-bit level costs **12.2× the covariance time and 2.5× the derivative
   footprint** (153 600 B → 384 000 B per level at 640×480). That is the number
   T4.1 weighs against X-20's accuracy finding; **this entry takes no bit-depth
   decision**, because the accuracy side of that trade is T4.1's to measure.
3. **The ternary kernel is not made redundant — but level 0 PAYS 3–8% for it, and
   the sign of that difference was buried by an absolute value.** "Within 1.08×"
   is true and uninformative: in **all 24 measured rows** (4 runs × 2 word types ×
   3 window sizes, three different binaries) the T3.6 ternary arm is **SLOWER** than the generic bit-sliced
   arm at N = 1, by 3–8%, with the direction unanimous and the worst case
   `uint32_t` / W = 31 / run 2 at 977.0 ns against 903.3 ns (8.2%, spreads 0.7–0.8%).
   The two arms are like-for-like — `benchmark/covariance_nbit_benchmark.cpp` feeds
   the ternary arm the same `LevelSet<1>` planes and the same window list inside the
   same interleaved batch — so this is a real, if small, cost of routing level 0
   through the specialization. **The table therefore does not support keeping the
   ternary path for SPEED**, and it is not kept for speed:
   [D-21](ARCHITECTURE.md#d-21-generic-n-is-not-capped-and-the-n--1-specialization-is-kept-as-a-test-oracle)
   keeps it as a **test oracle**, which is exactly the property T3.10's 61 232
   bit-identical positions per word type exercise. Whether the frontend should
   *dispatch* level 0 to the bit-sliced kernel is left open here rather than decided:
   it is one binary's reading of a 3–8% difference in the presence of a 1.46×
   layout effect, and it belongs with E-13 in T4.1's own binary.

**Decision.**
1. **No shipped code changes**, and no D-record is promoted: the formulation was
   already decided in [§7.5](ARCHITECTURE.md#75-lk-gradient-covariance) and this
   entry priced it rather than choosing it.
2. **[E-13](ARCHITECTURE.md#9-open-questions-and-planned-experiments) is registered
   and gated on T4.1** — does the per-row partial accumulator still pay above
   N = 1, where it is O(N²) per row against work that is O(N²) per word? It needs a
   binary per arm to escape the layout confound this entry hit, which is why it is
   an experiment and not a patch.
3. **The layout sensitivity is recorded against
   [E-12](ARCHITECTURE.md#9-open-questions-and-planned-experiments) as well.** X-21
   sized the kernel-shape cost from a two-point fit on one binary each; this entry
   is evidence that such a fit can move by 1.5× for reasons that are not in the
   source, and E-12's design should account for it.

---

### X-23 · The rolling response ring against the frame-sized response map · `DONE`

> **CORRECTED AT TRIAGE, and the corrections are named where they land rather than
> summarised here.** The *decision* is unchanged — band A fired, the streaming form
> ships, and 3 655 device checks plus an independent re-derivation of the answer
> both re-confirmed the equality precondition. What was wrong was reporting: (1) the
> `blockSize` crossover was stated as one number when the two word types cross in
> different places — `uint64_t` is already above 1.00 at 15, where X-18 put its own
> boundary, so conclusion 4's "the boundary moved" held on half the data; (2) two
> cross-checks against X-18 and the arm-order-swap control quoted the *scatter* run's
> numbers, and one pair of them (1.083 / 2.058) appears in neither log; (3) X-20's
> decision 3 carried the scatter run's 0.76× / 1.344× while four other sites carried
> the reported run's 0.774× / 1.327×; (4) the registered 752×480 *real frame* was
> replaced by synthetic content of the same size and the substitution was not
> recorded; (5) the frontend footprint was an **enumeration over listed buffers
> printed while the frame map was still live**, not the reading it claimed to be —
> now a live-byte high-water mark, in RESULT (a). A fourth site of the "~2×" figure
> (TASKS.md T3.8's X-20 write-up) was also missed by the rule's list of three and now
> carries the same named correction.

> **PRE-REGISTERED. Written and committed BEFORE the streaming form exists** —
> nothing in `ops/corner.hpp` implements it at this commit, and this commit touches
> this file and nothing else, so the history shows the rule predates the data. Same
> discipline as [X-9, X-10 and X-11](#phase-2-experiments--rules-recorded-first-then-measured)
> at 4245210 and as [X-21](#x-21--does-generic-n-cost-the-specialized-n1-and-ternary-paths-anything--done).
> **Result, Conclusion and Decision were deliberately empty at that commit.** They
> are filled in below, under a horizontal rule, and NOTHING ABOVE THAT RULE WAS
> TOUCHED when the numbers arrived — no band was re-scaled, re-scoped or softened.

**Numbering:** X-1…X-22 and X-7b are taken, so this entry is **X-23**. (Duplicates
have happened here — X-21 records two tasks whose Done-when pointed at an X-number
that was already in use.)

**Gates:** [E-10](ARCHITECTURE.md#9-open-questions-and-planned-experiments) · task
[T3.11](TASKS.md#t311--rolling-response-map-e-10--done) — whether
`cornerMinEigenVal` / `goodFeaturesToTrack` gain a streaming form, and if so whether
it becomes the recommended path or a second shape.

**Question:** What does replacing [T3.7](TASKS.md)'s frame-sized `float` response
map with a rolling three-row ring cost in time, and what does it actually save once
everything the streaming form must carry to preserve the selection's **global**
properties is counted against the ring?

**Why it exists.**
[X-20](#x-20--hybrid-lk-accuracy-against-ground-truth-and-the-frontends-peak-footprint--done)
measured the frontend's peak working set at **1 721 568 B**, of which T3.7's
response map is **1 228 800 B — 71.4%, more than everything else combined** — at
4 bytes per pixel where every other plane in the frontend is one or two BITS and the
tracker itself is 0.2%. On a project whose thesis is footprint, the largest buffer
in the frontend is a float scratch.

**And the trade is NOT settled by that fact.** [CLAUDE.md](CLAUDE.md)'s "memory
wins" is the tiebreak for a conflict **where no explicit choice has been made**;
this entry is where the choice gets made, so the tiebreak applies only if the
measurement leaves it genuinely close — and the bands below say what "close" is
before any number exists.

---

**TWO PRECONDITIONS, STATED FIRST, BECAUSE NEITHER IS ON THE TRADE CURVE.**

**1. Corner equality is a precondition, not a tradeable metric.** The streaming form
must return **identical** corners to the frame-map form: same count, same
coordinates, in the same order, with the same `CornerResult` triple (`count`,
`candidatesRanked`, `candidatesTruncated`). Proven by comparing the full output
arrays element for element over whole frames — not by sampling, not by counting how
many matched, not by a displacement tolerance. **An arm that returns different
corners has not solved the problem and is disqualified whatever its bytes and
nanoseconds say.** There is no band below that trades a corner for a byte or for a
nanosecond.

That is not pedantry, because **the selection pipeline is not local** and a
three-row ring gives none of it for free:

- the quality threshold is `qualityLevel × the maximum over the WHOLE map`, border
  included (`cv::minMaxLoc`'s region, `gftt.cpp` step 1);
- the greedy minimum-distance filter needs the survivors in descending order across
  the **whole frame**, with the reference's own tie rule — response descending, then
  `y` descending, then `x` descending, which is `greaterThanPtr` on pointers into a
  contiguous map spelled on coordinates (`impl::CornerStronger`).

Ties are common in this operation rather than exotic — a checkerboard makes the
entire interior tie, and a 3×3 window of popcounts takes few distinct values — so a
streaming implementation that changes which corner wins a tie changes real output on
real content. NMS ordering has teeth too: `Corner.SelectionOrder_PinsNmsBeforeDistance`
already pins a case where two stage orders give different survivors.

**2. The kernel rules are preconditions too**, not things to be bought back with
speed: views and never owning containers (D-5); **never throws**; **no heap
anywhere in the kernel** — the `operator new` counter T3.7's suite already runs
(plain and C++17 over-aligned) stays at zero; padding never counted
([D-13](ARCHITECTURE.md#d-13-a-reduction-counts-pixels-never-padding)), inherited
from the reductions. An arm that buys its numbers with an allocation is not an arm.

---

**WHAT COUNTS AS THE STREAMING FORM'S TRUE PEAK — defined now, so it cannot be
defined afterwards to flatter a result.** The peak is every byte alive at the
operation's high-water mark, namely:

1. **the response ring** — however many rows the implementation actually needs, not
   the three NMS needs in principle;
2. **everything carried to preserve the global properties.** The candidate/heap
   array, which the frame-map form also carries and which the streaming form may
   need **larger** (see the hypothesis); any per-column accumulator array a
   row-major sweep needs to keep an incremental form — `width` counters is 5 120 B
   at 640 px, and [X-11](#x-11--incremental-versus-recomputed-window-reductions--done)
   declined exactly that shape once already; any carried derivative rows or per-row
   bookkeeping;
3. counted the way X-20 counted it: `operator new` instrumented, every
   measurement-only buffer **scoped and destroyed before the accounting point** —
   the mistake X-20 caught in its own first table, where a `W·H` sizing buffer
   2.1× the reported total was still live — and the frontend total re-stated in
   X-20's five-row stage table so the 71.4% row can be read directly against its
   replacement.

**The frame-map form is re-measured in the same binary**, not quoted from X-20, so
both sides of the ratio come from one build.

**The arithmetic the ring estimate rests on, written out so the measurement can
contradict it.** Three `float` rows at 640 px is **7 680 B**; TASKS.md's "~15 kB"
budgets roughly twice that, and the difference is precisely the carry this entry
insists on counting rather than a disagreement about the ring. If the candidate
carry stays at X-20's 105 048 B (8 754 survivors), the corner stage goes
1 333 848 B → **112 728 B** and the frontend 1 721 568 B → **500 448 B, 3.44×** —
which would put the measured frontend **under**
[§4.6](ARCHITECTURE.md#46-memory-arithmetic)'s own "~0.6 MiB" projection for two
frames plus derivatives, a figure written before anything in this project counted a
float scratch. The frame-map frontend is **2.7×** that projection. That is the size
of the claim at stake, and **every byte of carry comes off it**.

---

**Hypothesis — two-sided on purpose, because the "~2× compute" estimate rests on an
assumption that may not hold.**

- **The second pass may not be forced.** The estimate assumes the global max
  requires evaluating every response twice. The shipped `selectGoodFeatures` already
  proves the threshold is a **pure post-filter** over the raw 3×3 maxima — its NMS
  is fused with the threshold on exactly that argument ("val > threshold and val is
  the maximum of its 3×3 neighbourhood in the RAW map"). So a single pass that
  carries raw maxima and applies `qualityLevel × max` after the last row is possible
  in principle: the pass count is an implementation choice, not a property of the
  operation.
- **The reason it may cost two passes anyway is memory, and it lands on the carry.**
  Un-thresholded, **every pixel of a flat plateau is a 3×3 maximum**, and a zero
  plateau is most of an edge-map response — so the raw-maxima set can be far larger
  than X-20's 8 754 thresholded survivors, with a structural worst case of
  `(w−2)(h−2)` candidates = **3 659 568 B** at 640×480, three times the map the
  streaming form set out to remove. A one-pass arm therefore has to prune against a
  **running** maximum (monotone non-decreasing, so anything below
  `qualityLevel × running max` is permanently dead) or degenerate into the two-pass
  arm. Whichever way it goes, the bytes land in the peak defined above.
- **And the traversal cuts the other way.** A ring forces a **row-major** sweep, and
  [X-18](#x-18--does-the-incremental-window-form-still-pay-inside-t37s-dense-sweep--done)
  measured the shipped column-major sliding sweep **1.19× SLOWER than a plain
  row-major recomputation at `blockSize` 3** — the reference pipeline's own block
  size. So the streaming form gets a discount on the shape change before it pays for
  any extra pass.
- **Expectation:** two-pass near 1.7–2.0× (two evaluations, less what row-major gives
  back); one-pass at or below 1.0× **if** its carry stays bounded. The outcome the
  bands are written to catch is a one-pass arm that is *faster* and yet saves *less*
  than 3.4×, because of what it carries.

---

**Decision rule** *(written before measuring; nothing below is evaluated against
data that exists at this commit).*

Define, all on the reference device, `uint32_t`, 640×480, `blockSize` 3 — SEAL's own
value and the point the decision is taken at:

- **`T`** = (streaming arm) ÷ (frame-map arm) for one whole `goodFeaturesToTrack`
  call, response **and** selection, medians of interleaved batches;
- **`P_frame`, `P_stream`** = the frontend peak under X-20's accounting with each
  form, the streaming one including all carry as defined above; **`R = P_frame ÷ P_stream`**.

**Gate on the saving, evaluated FIRST.** If **`P_stream > 750 000 B`** at 640×480
(equivalently `R < 2.3×`), the carry has eaten the saving E-10 was registered on,
**none of the time bands apply**, and the streaming form does not become the
recommended path on a footprint claim it does not have. *Why 750 000 B:* X-20's
non-corner stages total 387 720 B and are fixed by the frame size, so this line
leaves the whole corner stage ≤ ~362 000 B — under a third of the frontend instead
of 71.4%, and the float scratch no longer its dominant term. It also keeps the
result inside §4.6's "~0.6 MiB"-class projection rather than merely below the
status quo. Above that line the experiment has measured a different trade from the
one it set out to measure, and says so.

**The time bands, evaluated only if the saving gate passes.**

- **Band A — `T ≤ 1.25`: the streaming form ships AND becomes the recommended path.**
  The frame-map entry point stays — T3.7 made the map caller-provided, and a caller
  who wants to select twice over one map, or to mask it (the documented route for a
  mask), still needs it.
  *Why 1.25 and not 1.00:* this operation **already pays 1.19× at this exact block
  size** for a shape chosen on ergonomics rather than on a measurement — X-18 found
  the shipped column-major sliding sweep 20% slower than row-major recomputation at
  `blockSize` 3, and it ships that way because the alternative wanted a width-long
  accumulator array the caller would have to own. A project that has accepted 1.19×
  on this kernel to avoid a 5 kB scratch cannot coherently refuse a ~3.4× frontend
  footprint cut at 1.25×. *Why the floor is set by relevance and not by resolution:*
  anything under ~1.10 is not distinguishable from the code-layout effect this
  repository has measured twice (~10% between two instantiations in one TU; **1.46×**
  between binaries, [X-22](#x-22--what-an-n-bit-pyramid-level-costs-the-lk-covariance--done)).

- **Band B — `1.25 < T ≤ 2.20`: it ships as a second shape, and the recommended path
  does NOT move.** Both peaks and both times go in the D-record and in the header, so
  a caller picks with the numbers in front of them.
  *Why it ships at all:* a caller who cannot fit 1 228 800 B has **no** alternative
  today, and CLAUDE.md's benchmarking rule is that a target either fits or it does
  not. For that caller a 2× is unambiguously worth paying, and refusing to ship the
  shape would be deciding for them.
  *Why the recommendation does not move:* at `blockSize` 3 and 640×480 the response
  sweep alone measures ~101 ns/pixel on the reference device (X-18) — about **31 ms
  for one frame**. Beyond 1.25× the added time is a material fraction of a whole
  frame, and a caller with the megabyte to spare should not pay it silently.
  *Why 2.20 is the top:* the estimate this task was scheduled on is ~2×
  (TASKS.md T3.11, ARCHITECTURE §9's E-10 row, X-20's decision 3). 2.20 is that
  estimate plus roughly one layout-effect's margin, so a reading marginally above the
  estimate is judged against the estimate rather than against noise.

- **Band C — `T > 2.20`: it does not ship.** The measurement is recorded, `ops/corner.hpp`
  keeps only the caller-provided frame map, and E-10 closes as *"not at this price"*
  rather than as unanswered.
  *Why there is a ceiling at all:* memory-first is a **tiebreak, not a licence for an
  unbounded slowdown** — a rule that accepts any cost is not a rule.
  *Why it sits at 2.20:* past the estimate, the trade on offer is not the trade the
  project agreed to weigh. And this is already the operation furthest from
  [ROADMAP](ROADMAP.md#success-criteria) success criterion 4:
  [X-19](#x-19--the-tier-2-denominator-goodfeaturestotrack-against-opencv--done)
  measured binCV's whole detector **1.82× slower** than the byte-per-pixel OpenCV
  denominator (for 2.23× less memory, 5.71× once both candidate buffers are sized).
  At `T = 2.2` that becomes ~4× slower than the denominator; trading a stated success
  criterion that far against another one is a decision to be taken deliberately and in
  the open, not absorbed inside a footprint task.

- **Tie between two passing streaming arms:** the smaller true peak wins, unless it is
  more than **1.10×** slower than the other arm, in which case the faster one wins.
  That is CLAUDE.md's tiebreak applied at the resolution the layout effect leaves.

- **The outcome the rule must not silently swallow: `T < 1.00`.** If the streaming
  form is *faster*, then the "~2× the response compute" figure is wrong, and **three
  documents state it** — X-20's decision 3, ARCHITECTURE §9's E-10 row, and TASKS.md
  T3.11. CLAUDE.md's rule for a measurement that contradicts a documented claim
  applies to a claim this project made about itself: all three get corrected to the
  measured number, and the correction is **named**, not quietly applied. It is a live
  possibility rather than a courtesy — see the traversal bullet in the hypothesis.

---

**Variants — three arms, and each one lives in its OWN translation unit.**

1. **F, the control:** the shipped `goodFeaturesToTrack`, caller's 1 228 800 B map,
   column-major sliding sweep.
2. **S2, streaming two-pass:** three-row ring; pass 1 evaluates every response purely
   to find the global maximum; pass 2 re-evaluates, thresholds, runs the 3×3 NMS over
   the ring, and feeds the existing bounded heap.
3. **S1, streaming one-pass:** three-row ring; one evaluation per pixel; running global
   maximum; raw 3×3 maxima carried and pruned against that running maximum; the final
   threshold, the sort and the spacing filter applied after the last row.

Arms 2 and 3 answer different halves of the question and neither does alone: **S2
prices the naive shape the estimate describes; S1 prices whether the second pass is
needed at all.** If S1's carry is unbounded in practice it shows up as a truncation
against the equality precondition — which is a result, not a failure of the
experiment.

**"Own TU" is in this list rather than in an implementation note because the hazard
has been measured twice here:** ~10% between two instantiations of one kernel in a
single translation unit, and 1.46× for the same kernel between two binaries (X-22),
with T3.10 seeing 1.46× from adding arms to a shared TU. X-21 split its arms across
`genericn_arm_*.cpp` for exactly this reason and this entry copies that shape. The
A/B is additionally **re-run with the arm order in the batch swapped**; a band verdict
that moves when the order moves is reported as layout, not as a result.

**Workload:** 640×480 — X-20's own frontend configuration, so the footprint table
lines up row for row — and the repository's real 752×480 frame, whose survivor count
X-20 measured separately at 9 774 **because the candidate row is a per-frame reading
and not a bound**. `blockSize` 3 for the decision, with 7, 15 and 31 reported beside
it, because X-18 found the traversal and incremental effects cross over at different
sizes and a streaming form changes exactly the traversal. `uint32_t` (the shipped
default) and `uint64_t`. Selection parameters are `GoodFeaturesParams`' defaults —
`SEAL/seal_params.yaml` verbatim.

**Metric — memory and speed together (CLAUDE.md), with equality as a pass/fail beside
them:**
- ns/pixel for the whole detector call **and** for the response stage alone; medians
  of 11 interleaved batches, spreads reported;
- **ms/frame at 640×480 beside the ratio**, because a ratio does not say whether the
  frame budget moved;
- the **true peak** in bytes, as defined above, for every arm, at both frame sizes,
  with the structural `(w−2)(h−2)` candidate worst case stated beside the measured
  per-frame reading;
- the frontend total in X-20's five-row stage table;
- corner equality: pass/fail, with the compared counts printed.

**Method:** `benchmark/corner_streaming_benchmark.cpp` plus one file per arm, through
`measure_util.hpp`'s protocol, Release only. Every arm's ANSWER is checked against the
frame-map form before anything is timed. **Equality lives in `tests/test_corner.cpp`
as core-suite cases**, so it runs in all four `verify.sh` configurations rather than
only where a benchmark builds. Footprint via the `operator new` counter T3.7 and X-20
already use. Device:
`./scripts/run_on_pi.sh pi4 ./benchmark/corner_streaming_benchmark`, with
architecture, compiler, governor, core pinning and `vcgencmd get_throttled` **before
and after** recorded — a non-zero value, or one that CHANGES during the run,
invalidates it, and **exit 77 is a skip, not a pass**. Log:
`bincv-cpp/results/corner_streaming_benchmark_pi4.log`.

---

**THE MEASUREMENT. Everything above this line is unchanged from 79db8f8, where it
was committed with no streaming form in the tree; `git log -p EXPERIMENTS.md` shows
the bands, their numbers and their justifications predating every number below.**

**Environment.** Reference device `pi4`, Raspberry Pi 4 Model B Rev 1.5, aarch64 /
6.18.34+rpt-rpi-v8, `g++ (Debian 14.2.0-19) 14.2.0`, governor `performance`, pinned
with `taskset -c 3`, `throttled=0x0` **before and after** both runs (unchanged during
either), exit 0 — not 77. Release, core-only. Code:
`benchmark/corner_streaming_benchmark.cpp` plus one translation unit per arm.

**TWO DEVICE RUNS, AND BOTH LOGS ARE COMMITTED**, because the within-run spread here
is 0.1–0.3% and that is *not* what a reader should hold a nanosecond to:

- `bincv-cpp/results/corner_streaming_benchmark_pi4.log` — the reported run, taken
  at commit `f253a2e`, which is the commit containing the code it measured;
- `bincv-cpp/results/corner_streaming_benchmark_pi4_scatter.log` — an independent
  earlier run of the same source, whose environment block names the parent commit
  `79db8f8` because the tree was not yet committed when it ran.

**Run-to-run scatter is ~1.3% on the ratio `T` and up to ~3.4% on an individual
ns/pixel column** — an order of magnitude above the within-run spread, and the same
lesson X-18 recorded with its own scatter log. **Every band verdict is identical in
the two runs, at every block size, both word types and both frame sizes**, so the
ratio is the quotable quantity and the third digit of a nanosecond is not.

---

**PRECONDITION 1 — CORNER EQUALITY. PASSED, AND PROVEN BY FULL-ARRAY COMPARISON
RATHER THAN BY SAMPLING.**

`Corner.Streaming_*` in `tests/test_corner.cpp` — **core-suite cases, so they run in
all four `verify.sh` configurations**, not only where a benchmark builds:

| case | what it compares | scale |
|---|---|---|
| `Streaming_IdenticalCorners_{uint8,16,32,64}_t` | `count`, `candidatesRanked`, `candidatesTruncated`, and the whole `[0, candidatesRanked)` prefix — coordinates and exact `float` bits | **1 080 cells and 133 098 corner records per word type**; 8 frames × 6 block sizes (3, 4, 5, 7, 15, 31) × 4 parameter sets × 6 capacities, **less the 72 cells whose frame has no survivor at that block size, where the capacity sweep is five wide rather than six** (`survivors − 1` is not a capacity when `survivors` is 0) — 1 152 − 72 = 1 080 |
| `Streaming_IdenticalCorners_LargeFrames` | the same, at 160×120, 129×97 and 151×113 | **184 cells and 566 270 corner records** at `uint32_t` and again at `uint64_t` |
| `Streaming_RowMatchesFrameMap_*` | `cornerMinEigenValRow` against `cornerMinEigenVal`'s row `y` | **65 910 positions per word type, bit-identical** |
| `Streaming_DegenerateShapes` | 1×1, 1×9, 9×1, 2×2, 3×3, 4×3, 3×4, 33×2 — frames with no NMS row at all | 8 shapes |
| `Flow.FrontendFootprint_640x480` | the two shapes on **X-20's own frontend content**, 640×480 | 8 754 candidates, 0 differing |

**1 664 932 corner records compared, 0 differing.** The frame list is deliberately
tie-dominated rather than random — a checkerboard at block 1 (the whole interior one
response), a checkerboard at block 4, stripes, a uniform frame (every response
exactly 0, so `maxVal` is 0 and nothing survives), and a single dot — because a
partial-maximum threshold or a reordered tie is invisible on content with distinct
responses. **82 of the 1 080 cells per word type contain a repeated response and 476
truncate**, and the suite asserts that both counts are non-zero, so a zero mismatch
count is a statement about a sweep that actually entered those paths.

**AND IT IS PROVEN ON THE PRIMARY TARGET, NOT ONLY ON x86-64.** The whole suite was
re-run on the reference device (`throttled=0x0` before and after, exit 0, commit
`6fa1a91` — re-run at triage, over the whole of `test_corner` **and** the whole of
`test_opticalflow` rather than one filtered case; log
`bincv-cpp/results/corner_streaming_tests_pi4.log`): **3655/3655 checks passed**,
`test_opticalflow` **169/169**, with the four `IdenticalCorners` cases reporting the **same 1 080
cells, 133 098 records, 476 truncating cells and 82 tie-containing cells** as the
x86-64 run, `LargeFrames` the same 566 270 records at each word type, the row kernel
bit-identical over the same 65 910 positions, and `operator new` = 0.
`Flow.FrontendFootprint_640x480` reads the same peaks on aarch64 as on x86-64 —
**1 723 232 B → 502 112 B**, i.e. the same 1 721 568 B → 500 464 B attribution plus
the same 1 664 B of bookkeeping — with the accounting identities holding on both. That is what makes the equality a property of the *operation* rather
than of one compiler's floating-point code generation — which matters here, because
the response's one rounding is a `std::sqrt` that IEEE-754 requires to be correctly
rounded and D-20 declines to promise cross-ISA bit-identity for the LK solve.

**And the suite can fail.** Three mutants of the shipped streaming form, built and
run: dropping the post-threshold filter (34 mismatching cells, 6 failed checks);
failing to record an evicted candidate in the discarded-maximum (6 failed checks);
breaking ties by ASCENDING raster position in the streaming sort alone — the exact
failure a ring invites — (7 failed checks). All three were caught by
`Streaming_IdenticalCorners_*` alone.

**PRECONDITION 2 — THE KERNEL RULES. PASSED.** Views only (D-5); never throws; the
`operator new` counter reads **0** across `goodFeaturesToTrackStreaming` and
`cornerMinEigenValRow` in the test suite and **0** for all three arms in the
benchmark, on the device; padding never counted (D-13, inherited from the
reductions). The ring is the caller's, as the map is.

---

**HOW THE GLOBAL PROPERTIES ARE PRESERVED — the hard part, and it turned out not to
need a second pass.**

The registered hypothesis was two-sided: a one-pass form is possible in principle
because the threshold is a pure post-filter, but its candidate carry might be
unbounded, with a structural worst case of `(w−2)(h−2)` = 3 659 568 B. **The carry
is bounded, and it is bounded exactly, by an argument rather than by a heuristic.**
Write `A` for the raw 3×3 maxima, `S = {a ∈ A : a.response > threshold}`, `K` for
the caller's `capacity`:

1. **`S` is UPWARD CLOSED in `A`** under `CornerStronger`, which orders on response
   first — anything stronger than a member of `S` is itself above the threshold.
2. For an up-set, **`topK(A) ∩ S = topK(S)` when `|S| > K`, and `= S` when
   `|S| ≤ K`.** So keeping the `K` strongest RAW maxima and applying the threshold
   after the last row yields **exactly** the set the frame-map form ranks, and `K`
   is the caller's own candidate array. **The carry for the global sort is zero
   extra bytes.**
3. **`candidatesTruncated` — which means `|S| > K` — is one `float`.** If any
   discarded maximum is above the threshold then every retained one is at least as
   strong and hence also above it, so `|S| ≥ K+1`; if none is, `S` is contained in
   the retained set and `|S| ≤ K`. So `truncated == (max response among discarded
   candidates > threshold)`.
4. **The plateau problem is a RUNNING threshold, and the prune is provably
   answer-preserving.** Un-thresholded every pixel of a flat plateau is a 3×3
   maximum. The running maximum is monotone non-decreasing and
   `x ↦ float(double(x)·q)` is monotone for `q > 0`, so the running threshold never
   exceeds the final one and everything it rejects is permanently outside `S`.
   Removing non-members of `S` from `A` changes neither `topK(A) ∩ S` nor the test
   in item 3, since both arguments used only `S`'s up-set property inside whatever
   `A` is.

**So the answer is invariant to how aggressive the prune is, and to the order
candidates are visited in** — which is why equality holds on a checkerboard whose
entire interior ties. The whole extra carry is **16 B**: a running maximum, a
running retained count and the strongest discarded response. The two-pass arm S2 was
still built and measured, because "the second pass is unnecessary" is a claim that
needs a price beside it.

---

**RESULT (a) — TRUE PEAK. The saving gate is evaluated FIRST, as the rule says.**

640×480, `uint32_t`, `blockSize` 3, `operator new` instrumented, every
measurement-only buffer scoped and destroyed before the accounting point (the
mistake X-20 caught in its own first table):

| term | frame map | streaming |
|---|---|---|
| response storage | 1 228 800 B | **7 680 B** |
| candidate array (also the output) | 105 048 B | 105 048 B |
| carry for the GLOBAL properties | 0 B | **16 B** |
| **corner stage TRUE PEAK** | **1 333 848 B** | **112 744 B** — 11.83× |

and the frontend in X-20's five-row stage table, re-measured end to end by
`Flow.FrontendFootprint_640x480` on X-20's own content and with the two shapes
asserted to return identical corners:

| stage | frame map | share | streaming | share |
|---|---|---|---|---|
| denoise | 76 800 | 4.5% | 76 800 | 15.3% |
| pyramid | 102 240 | 5.9% | 102 240 | 20.4% |
| derivative | 204 480 | 11.9% | 204 480 | 40.9% |
| **corner** | **1 333 848** | **77.5%** | **112 744** | **22.5%** |
| track | 4 200 | 0.2% | 4 200 | 0.8% |
| **TOTAL** | **1 721 568** | | **500 464** | |

**`P_stream` = 500 464 B ≤ 750 000 B, `R` = 3.44× → THE SAVING GATE PASSES**, so the
time bands apply. The response storage is 1.5% of the streaming frontend and the
**candidate array is now the corner stage's dominant term** — which is also the only
content-dependent row in the table, and the reason the structural worst case is
stated beside it: at `(w−2)(h−2)` candidates the two forms are 4 888 368 B and
3 667 264 B, i.e. **the ring stops being the interesting term long before the
candidate array does.** At 752×480 the corner-stage peak is 1 630 524 B → 195 724 B
(8.33×) on this benchmark's frame.

**The pre-registered arithmetic predicted 500 448 B; the measurement is 500 464 B.**
The 16 B difference is the carry the entry insisted on counting and the estimate had
not enumerated. §4.6's "~0.6 MiB" projection for two frames plus derivatives — written
before anything in this project counted a float scratch — is **exceeded 2.7× by the
frame-map frontend and met by the streaming one.**

**AND THE TWO TOTALS ARE NOW READ RATHER THAN ADDED UP — CORRECTED AT TRIAGE,
BECAUSE THE FIRST VERSION OF THIS ROW WAS NOT THE MEASUREMENT IT CLAIMED TO BE.**
`Flow.FrontendFootprint_640x480` used to compute the streaming total as
`ring + candidates + carry` — an enumeration of the buffers its author had listed, so
no buffer nobody listed could move it — **and it printed that total while the
1 228 800 B frame map was still live**, which is the very mistake the case's own
candidate-probe note says it is avoiding. Both are fixed:

- the test's replaced `operator new`/`delete` now track **live bytes and their
  high-water mark**, not just call counts (every block carries a 16 B header so the
  requested size is recoverable at `free`);
- the frame-map stage and the streaming stage each run in **their own scope with the
  other shape's buffers destroyed**, and each peak is read *inside* its scope at the
  high-water moment;
- the per-stage rows are then required to **account for the reading to the byte** —
  `BINCV_CHECK_EQ(framePeak, total + bookkeeping)` and the same for the streaming
  peak — with `bookkeeping` (**1 664 B**: the `std::vector`s of `BinMat`/`TernaryMat`
  inside the test's `Frontend` and its `LKLevel` bundle) named, printed, and required
  to be **identical in both readings**, so that
  `framePeak − streamPeak == responseBytes − ringBytes` is a difference of two
  measurements of the same thing;
- nothing asserts *inside* a measured window, because `BINCV_CHECK_EQ` builds its
  message eagerly (`std::to_string` allocates whether the check passes or not) and a
  check inside the window lands in the mark being read. That is not hypothetical:
  the first version of this rewrite asserted inside the window and the two readings
  disagreed by 313 B. The windows record; the assertions follow.

Read this way the frontend measures **1 723 232 B → 502 112 B**, which is the
1 721 568 B → 500 464 B table above plus the 1 664 B of bookkeeping and minus the
16 B of carry that is on the stack and therefore cannot appear in a heap reading.
**The quoted numbers are unchanged**; what changed is that they are now falsifiable.
Verified by mutation: a `new float[640*480] … delete[]` transiently inside
`goodFeaturesToTrackStreaming` takes the streaming reading to 1 730 912 B and fails
two checks. **And the limit is stated rather than glossed:** this is a HEAP
high-water mark, so a `static float[640*480]` inside the kernel does *not* move it —
measured, the test binary's BSS goes 648 B → 1 229 464 B and all 29 checks still
pass. What covers that case is D-5's caller-provided-scratch contract and reading the
header, not this case.

---

**RESULT (b) — TIME. `T` is the whole `goodFeaturesToTrack` call, medians of 11
interleaved batches, three arms in three translation units.**

640×480, `uint32_t`, the decision configuration. Spreads are **within-run full
scatter**, `(max − min)/median`:

| block | arm | detector ns/px | spread | ms/frame | response stage ns/px | spread | `T` |
|---|---|---|---|---|---|---|---|
| **3** | F frame map | 132.790 | 0.18% | **40.79** | 107.119 | 0.15% | 1.000 |
| **3** | S2 two-pass | 176.192 | 0.17% | 54.13 | 145.889 | 0.06% | 1.327 |
| **3** | **S1 one-pass** | **102.845** | 0.27% | **31.59** | **77.123** | 0.09% | **0.774** |
| 7 | F | 265.979 | 0.14% | 81.71 | 152.632 | 0.15% | 1.000 |
| 7 | S2 | 369.456 | 0.08% | 113.50 | 256.211 | 0.06% | 1.389 |
| 7 | S1 | 243.803 | 0.15% | 74.90 | 129.749 | 0.14% | **0.917** |
| 15 | F | 392.017 | 0.69% | 120.43 | 265.320 | 0.04% | 1.000 |
| 15 | S2 | 651.213 | 0.62% | 200.05 | 523.679 | 0.19% | 1.661 |
| 15 | S1 | 391.083 | 0.68% | 120.14 | 263.424 | 0.04% | **0.998** |
| 31 | F | 653.055 | 1.61% | 200.62 | 585.104 | 1.52% | 1.000 |
| 31 | S2 | 1341.101 | 6.39% | 411.99 | 1275.768 | 6.79% | 2.054 |
| 31 | S1 | 706.207 | 6.00% | 216.95 | 653.194 | 1.65% | **1.081** |

**THE ARM ORDER WAS SWAPPED IN THE BATCH AND THE RUN REPEATED**, as the rule
requires. In order (S1, S2, F) the same cells read `T(S1)` = **0.775, 0.917, 0.998,
1.081** and `T(S2)` = 1.327, 1.389, 1.661, 2.054 — **the largest movement anywhere is
0.001**, at `blockSize` 3; at 15 and 31 it is 0.000. **The verdict is not a
batch-position effect**, and this is the control the twice-measured layout hazard
demanded. (An earlier version of this paragraph transcribed the `blockSize` 31 cells
as 1.083 and 2.058 and bounded the movement at 0.002. Neither figure is in either
committed log — both swapped-order runs read 1.081/2.054 and 1.086/2.063 — and
TASKS.md T3.11 already carried the correct 0.001. The log lines are
`corner_streaming_benchmark_pi4.log:134` and `..._scatter.log:134`.)

**AND THE OTHER DEVICE RUN AGREES ON EVERY VERDICT** (the scatter log): `T(S1)` =
**0.764 / 0.914 / 0.997 / 1.085**, `T(S2)` = 1.344 / 1.382 / 1.661 / 2.063. The
largest disagreement between the two runs is **1.3% on `T` at `blockSize` 3** and
**3.4% on one ns/pixel column** (S2's response stage, 145.889 against 150.859) — which
is why there are two logs rather than one.

`uint64_t`, 640×480: `T(S1)` = **0.771 / 0.931 / 1.025 / 1.125** at blockSize 3 / 7 /
15 / 31, `T(S2)` = 1.332 / 1.397 / 1.654 / 2.117 — the same shape, and half a point
to four points worse for the streaming form. (Scatter run: 0.775 / 0.932 / 1.018 /
1.122.)

**THE CROSSOVER IS NOT AT THE SAME PLACE IN THE TWO WORD TYPES, AND AN EARLIER
VERSION OF THIS ENTRY SAID IT WAS.** At `uint32_t` the streaming form is still at
parity at `blockSize` 15 (0.998) and the crossover falls between 15 and 31. At
`uint64_t` it is **already above 1.00 at 15** — 1.025 in the reported run, 1.018 in
the scatter run, against within-run spreads of 0.58% and 0.69% on the two arms, so
the 2.5% gap is about four times the noise — and the crossover falls between **7 and
15**. Both runs agree on that, so the reading is:

| word type | `T(S1)` at 3 / 7 / 15 / 31 | crossover |
|---|---|---|
| `uint32_t` | 0.774 / 0.917 / 0.998 / 1.081 | between **15 and 31** |
| `uint64_t` | 0.771 / 0.931 / 1.025 / 1.125 | between **7 and 15** |

This matters for one thing in particular and it is flagged rather than buried:
[X-18](#x-18--does-the-incremental-window-form-still-pay-inside-t37s-dense-sweep--done)
put the sliding-versus-recompute crossover between 7 and 15, and **at `uint64_t` this
entry reproduces that boundary exactly rather than moving it**. So conclusion 4
below — "X-18's boundary moved" — holds for `uint32_t` only, on half the measured
data, and [E-11](ARCHITECTURE.md#9-open-questions-and-planned-experiments) (should
the sweep select its traversal on `blockSize`?) gets a *word-type-dependent* second
data point, not a clean disagreement. That is a further reason E-11 stays open.

752×480, `blockSize` 3: `T(S1)` = **0.749** (`uint32_t`) and **0.755** (`uint64_t`);
`T(S2)` = 1.311 and 1.331. The saving is slightly *larger* on the wider frame, which
is the direction a row-major sweep should move in. The corner-stage peak there is
1 630 524 B → 195 724 B (**8.33×**), on 15 557 survivors of this benchmark's own
frame.

**A DEVIATION FROM THE REGISTERED WORKLOAD, RECORDED RATHER THAN LEFT TO BE
NOTICED.** The pre-registration above names "the repository's real 752×480 frame,
whose survivor count X-20 measured separately at 9 774". What the committed
benchmark runs at 752×480 is the **frame SIZE** with this benchmark's own synthetic
content (`makeFrame`), which yields **15 557** survivors — the decoded repository
frame is behind `BINCV_WITH_OPENCV` and the arms are built core-only. The
consequence is bounded and stated: 15 557 is **not** comparable to X-20's 9 774, so
the candidate-array row of the 752×480 footprint block is this benchmark's frame and
nothing else, and conclusion 3's "9 774 on the real frame" comes from
[X-20](#x-20--the-lucaskanade-tracker-on-binary-pyramids--done), not from here. The
*ratio* `T` is unaffected — both arms see the same content — and the 640×480 cells,
which are the decision, run the same content as each other and are compared with the
frontend number through `Flow.FrontendFootprint_640x480`, which does run X-20's own
content.

---

**TWO CROSS-CHECKS AGAINST X-18, BECAUSE A CONTROL THAT DOES NOT REPRODUCE A KNOWN
NUMBER IS NOT A CONTROL. BOTH RUNS ARE QUOTED, BECAUSE THE TWO CROSS-CHECKS ARE
EXACTLY THE PLACE WHERE PICKING ONE WOULD BE PICKING THE FLATTERING ONE.** X-18
measured the shipped sliding response sweep at **101.25 ns/px** at `blockSize` 3 on
this device. Arm F's response column reads **107.119** in the reported run and
**105.360** in the scatter run — the same sweep plus the `minMaxLoc` pass this entry
deliberately charges to it. That surcharge is therefore **4.1–5.9 ns/px**, and one
linear pass over a 1 228 800 B `float` map is the right size for it at either end. So
the control is the shipped code behaving as previously measured. (An earlier version
of this paragraph quoted only 105.360 while declaring the reported run its source,
which made the surcharge look 43% tighter than the bracket the two runs actually
give.)

The second cross-check does **not** land on X-18's number, and it is reported rather
than reconciled. X-18's row-major recomputation arm read **84.83 ns/px** at
`blockSize` 3; S1's row kernel does the same arithmetic in **77.123** (reported run)
and **74.654** (scatter run) — **9.1% to 12% less**. Two candidate explanations, and
**this entry does not have the measurement that separates them**:

- **The argument spelling.** X-18's arm calls `gradientCovariance(dx, dy, rect)` on
  the **containers**, which rebuilds four views **per pixel**;
  `cornerMinEigenValRow` takes the views once and loops. Both end in the same
  `countCovariance`, four popcounts per word, so the reduction is not the
  difference. This is a plausible 12%-sized effect and it is the same *kind* of cost
  [E-12](ARCHITECTURE.md#9-open-questions-and-planned-experiments) was registered for.
- **Code layout.** The two numbers come from **different binaries**, and this
  repository has measured **1.46×** of cross-binary drift on unchanged source
  ([X-22](#x-22--what-an-n-bit-pyramid-level-costs-the-lk-covariance--done)). 12% is
  well inside that.

So **12% is an upper bound on the argument-spelling effect, not a measurement of
it** — and it is the upper end of a 9.1–12% run-to-run bracket, which is itself the
point — and the sentence to take away is only that S1's row kernel is at least as fast
as X-18's recomputation arm — which is what the comparison in this entry needs, since
both of *its* arms are in this binary. It is a further reason not to close
[E-11](ARCHITECTURE.md#9-open-questions-and-planned-experiments) from either entry's
numbers alone.

**BAND A FIRES, AND SO DOES THE OUTCOME THE RULE SAID MUST NOT BE SWALLOWED.**

`T` = **0.774** at the decision point (0.764 in the other run), which is not merely
inside band A's `T ≤ 1.25` — **it is below 1.00**, in both runs, at both word types
and at both frame sizes. The rule pre-declared that case, called it live rather than
courteous, and required the correction to be named:

> "If the streaming form is *faster*, then the '~2× the response compute' figure is
> wrong, and **three documents state it** — X-20's decision 3, ARCHITECTURE §9's E-10
> row, and TASKS.md T3.11. […] all three get corrected to the measured number, and
> the correction is **named**, not quietly applied."

**All three are corrected in this commit, by name.** The streaming form is
**1.29× FASTER** at the reference pipeline's own block size, not 2× slower; the
whole detector goes from 40.79 ms/frame to 31.59 ms/frame at 640×480, and the
frontend's peak falls 3.44×.

**Why the estimate was wrong is not a mystery and was written into the hypothesis
before the run.** A ring FORCES a row-major sweep, and
[X-18](#x-18--does-the-incremental-window-form-still-pay-inside-t37s-dense-sweep--done)
had already measured the shipped column-major sliding sweep **1.19× slower than
row-major recomputation at `blockSize` 3**. The streaming form collects that discount
before paying for anything, and the one-pass argument above means it never pays for a
second evaluation. The estimate assumed the second pass; **S2 is what the estimate
actually described, and S2 is 1.33× — so even the naive shape came in well under the
"~2×" the task was scheduled on, at `blockSize` 3.**

**THE ARM TIE RULE IS WHAT PICKS S1 OVER S2, AND IT IS APPLIED AS WRITTEN.** The
two streaming arms have **the same true peak**: the same three-row ring — S2 uses
ring row 0 as its first pass's scratch, so its extra pass costs no bytes — and the
same candidate array, differing only by a scalar or two of live state, which is far
inside the resolution of any footprint claim this project makes. So the rule's first
clause, "the smaller true peak wins", **cannot separate them**, and its second,
"unless it is more than 1.10× slower", decides: S2 is **1.71× S1**. S1 ships. This
is the tie the clause was written for, and it is worth noting that the clause was
written for a case nobody expected to arise — the pre-registration assumed the two
arms would differ in *carry*, and they do not.

**THE COST, STATED WITH THE WIN.** The streaming form is *slower* at large blocks —
at `uint32_t` 1.00× at 15 and 1.08× at 31; at `uint64_t` **1.03× already at 15** and
1.13× at 31 — because that is where the sliding accumulator earns its keep, and X-18
measured that crossover from the other side. **The crossover is between 15 and 31 at
`uint32_t` and between 7 and 15 at `uint64_t`, not between 3 and 7 in either**, so
the frame-map form is the faster shape only at block sizes above the one the MVP
pipeline runs. It also remains the only shape that produces a map, which the
documented mask route and any caller selecting twice over one map both need.

**ONE READING WORTH FLAGGING RATHER THAN ABSORBING.** The `blockSize` 31 rows have
**6.0–6.8% within-run spreads** against 0.04–0.7% everywhere else, on both streaming
arms and in both orders (the scatter run shows the same thing at 4.2–4.4%). The
verdict there — 1.081× here, 1.085× in the other run — is well outside that scatter
and the order swap moves it by 0.002, so it is reported as measured; but a reader
interpolating a 31×31 number from this table should re-measure rather than trust the
third digit.

**AND ONE NON-RESULT.** The x86-64 run of the same binary is filed as indicative
only, and it disagrees: `T(S1)` = 1.044× at `blockSize` 3, with spreads of 12–52%.
That is the same sign reversal X-18 recorded on x86 for the underlying traversal
question ("the x86 run has the opposite sign at `blockSize` 3"), and it is why the
device is where this is closed.

**Conclusion.**

1. **The frame-sized `float` response map is not needed, and removing it is not a
   trade.** The streaming form is 11.83× smaller in the corner stage, **3.44× smaller
   across the whole frontend**, and **1.29× faster** at the reference pipeline's block
   size, with corners identical to the byte. E-10's premise — that the ring costs
   compute — is wrong at `blockSize` 3, and the reason is the traversal the ring
   forces rather than anything clever in the implementation.
2. **The second pass is unnecessary, and that is the substantive finding.** The
   threshold is a pure post-filter, the candidate set is a top-K over an up-set, and
   the truncation flag is one `float`. The naive two-pass shape costs **1.33×** for
   **the same** footprint — it buys nothing at all.
3. **The dominant term in the corner stage is now the candidate array, and it is the
   one content-dependent row in the table.** 8 754 survivors at 640×480 and 9 774 on
   the real frame; the structural maximum is 3 659 568 B, which would be 7.3× the
   whole streaming frontend. **The next footprint question in this operation is the
   candidate array, not the response storage** — but it is a CONTRACT question (what
   a caller provisions, and what `candidatesTruncated` costs them), not a buffer
   question, and it is not opened here.
4. **X-18's crossover is confirmed from the other side, and its boundary moves with
   the WORD TYPE rather than simply moving.** X-18 measured sliding-versus-recompute
   inside the response sweep alone and put the net crossover between 7 and 15.
   Measured through the whole detector with the streaming form's own row kernel, the
   crossover is between **15 and 31 at `uint32_t`** and between **7 and 15 at
   `uint64_t`** — i.e. X-18's own boundary, unmoved, at the wider word. Both device
   runs agree on both cells. This is one data point on
   [E-11](ARCHITECTURE.md#9-open-questions-and-planned-experiments), not a settled
   disagreement with X-18, and the word-type dependence is itself a reason E-11
   should not be closed from either entry's numbers.

**Decision.**

1. **[D-22](ARCHITECTURE.md#d-22-the-corner-response-streams-over-a-three-row-ring-and-that-is-the-recommended-path)
   is promoted to ARCHITECTURE §8**: `goodFeaturesToTrackStreaming` ships and is the
   **recommended path**; `cornerMinEigenVal` + `selectGoodFeatures` stay, as T3.7's
   caller-provided map already allowed, for callers who need the map itself or who
   run a large `blockSize`. Both peaks and both times are in the header so a caller
   picks with the numbers in front of them.
2. **E-10 CLOSES** and leaves the register.
3. **Three documents are corrected by name**, as the rule required: X-20's decision 3,
   ARCHITECTURE §9's E-10 row, and TASKS.md T3.11 all said "roughly 2× the response
   compute". The measured figure is **0.774×** (0.764× in the other device run),
   and none of the three is edited silently — each carries the correction.
   **A FOURTH SITE EXISTED AND THE RULE'S LIST DID NOT NAME IT**: TASKS.md T3.8's
   X-20 write-up carries the same sentence ("E-10 should be scheduled … for roughly
   2× the response compute"). It was found at triage and now carries the same named
   correction. The lesson is about the rule, not about the result — an enumeration of
   sites written from memory is not a search, and a rule that requires a correction
   "in three documents" should have said "wherever the figure appears", which is what
   `grep` can actually check.
4. **X-20's footprint table is restated** with the number this experiment measured:
   1 721 568 B → **500 464 B**, and the 71.4% row becomes 1.5%.
5. **The two-pass arm is NOT shipped.** It lives in
   `benchmark/corner_streaming_arm_stream2.cpp` as the priced alternative, because
   nothing should ship two implementations of one answer.
6. **E-11 is not touched.** Whether `cornerMinEigenVal` should select its traversal
   on `blockSize` is still open and still unscheduled; this entry adds a second data
   point on the same crossover from a different direction and deliberately takes no
   decision on it.

---

### X-24 · Pyramid level bit depths · `PARTIAL`

**THIS RULE IS COMMITTED ON ITS OWN, BEFORE THE HARNESS THAT MEASURES IT EXISTS.**
Same discipline as X-9/X-10/X-11 and X-23. The kernel that makes the question
askable at all — the generic-`N` tracker — landed first and is measured only for
*correctness* in that commit; not one accuracy, footprint or timing number below
had been produced when this was written.

**Gates:** [E-7](ARCHITECTURE.md#register) ·
[T4.1](TASKS.md#t41--e-7--pyramid-level-bit-depths--todo)
**Question:** How many bits does each pyramid level need to preserve tracking
accuracy, and what do those bits cost in footprint and in time?

**Hypothesis** — three parts, and the third is the one most likely to be wrong:

1. **Accuracy improves with coarse-level depth**, because that is what
   [X-20](#x-20--hybrid-lk-accuracy-against-ground-truth-and-the-frontends-peak-footprint--done)
   diagnosed: a level whose pixels are BITS cannot localise a sub-pixel motion
   better than its own quantisation, and that error is multiplied by `2^level` on
   the way down. Deepening the coarse levels should remove exactly that term.
2. **It will NOT reach the one-level number** (0.0017 px). X-20 separated two other
   terms that a deeper alphabet does not touch — a **level-0 stationary point** with
   no pyramid in the picture, and the **clipped coarse-level window** (deviation
   (ii)), about half of the four-level error. So the expected outcome is "inside
   X-20's tolerance", not "as good as one level", and a result that merely fails to
   reach 0.0017 px is NOT a failure of this experiment.
3. **The two cost terms weight the levels OPPOSITELY, and the expensive one is
   time, not bytes.** Stated now because it determines which end of the ladder is
   worth deepening and it would be easy to rationalise afterwards:
   * the **pyramid build** and the **derivative** are per-pixel, so level `l` costs
     `1/4^l` as much — deepening the COARSE levels is nearly free in both bytes and
     build time, and X-15 already measured the whole uncapped ladder at just
     **1.65×** the re-binarized one, because level 0 dominates the footprint;
   * the **tracker** is per-point-per-window, and **every level tracks the same
     points through the same 31×31 window**, so `residualSums`' `20N²` popcounts
     per word are paid IN FULL at every level regardless of how small it is. A
     `1/3/5/7` ladder pays `1 + 9 + 25 + 49 = 84` units of tracking popcount against
     `4` for `1/1/1/1` — **21×** — while costing only 1.65× the pyramid bytes.

   So the cheapest passing ladder is expected to be the one that deepens the coarse
   levels **as little as possible**, and the binding constraint is expected to be
   TRACKER TIME rather than footprint. If that inverts — if footprint binds first —
   it means the tracking cost model above is wrong and the entry must say so.

**Decision rule** *(written before measuring)*

**The accuracy gate is [X-20](#x-20--hybrid-lk-accuracy-against-ground-truth-and-the-frontends-peak-footprint--done)'s,
inherited verbatim, and NOTHING in it may be widened by this entry:** RMS endpoint
error ≤ 0.25 px, max endpoint error ≤ 1.0 px, ≥ 80% of eligible points tracked, and
no tracked point STUCK, with X-20's derived model-error allowance for rotation and
scale. It is evaluated at **four levels on the reference pipeline's own edge maps**
— the configuration that MISSED in X-20 — **and** on X-20's synthetic cases, so a
ladder cannot pass by trading real-content accuracy for synthetic.

Among ladders that pass:

* **Band A — a shallow ladder suffices.** Some ladder with no level deeper than
  **3 bits** passes. Adopt the smallest such ladder by frontend peak footprint;
  where footprints tie within 2%, the faster wins ([CLAUDE.md](CLAUDE.md): memory
  wins, then speed). E-7 closes and the fix is cheap.
* **Band B — depth is needed but bounded.** No ≤3-bit ladder passes, but some ladder
  within the natural growth `1/3/5/7` does. Adopt the smallest-footprint passing
  ladder and report the whole curve. **If that ladder's tracker time at 640×480
  exceeds 2× the `1/1/1/1` ladder's, that goes in the conclusion as a headline, not
  a footnote** — a 2× tracker is a real cost against the project's speed goal and
  the decision to pay it must be explicit rather than absorbed.
* **Band C — nothing passes.** No ladder up to `1/3/5/7` brings the four-level
  tracker inside X-20's tolerance. Then **depth is not the whole cause**, X-20's
  other two terms dominate, and the required output is (i) the best achievable
  curve, (ii) an explicit statement that a deeper alphabet does not fix T3.8's
  miss, and (iii) a new experiment registered for the remainder. **The tolerance is
  not widened and the kernel is not tuned to fit it.**
* **Band D — accuracy is NON-MONOTONIC in depth.** Pre-declared because it is
  plausible rather than perverse: depth changes `sumXX`/`sumYY`/`sumXY` and
  therefore which points the `minEigThreshold` test rejects, so a deeper ladder can
  end up tracking a *different and harder* point set. If this fires, the tracked
  point SETS must be reported beside the errors — **a curve measured over different
  point sets is not a curve** — and the comparison re-run over the intersection.

**Reporting is mandatory in every band**, per [CLAUDE.md](CLAUDE.md)'s "report
memory and speed together": pyramid + derivative bytes, frontend peak bytes,
tracker ns/frame and pyramid-build ns/frame.

**Variants:** `1` (single level — X-20's best case, the control), `1/1/1/1` (what
ships today — X-20's failing case), `1/2/2/2`, `1/3/3/3`, `1/3/4/4`, `1/3/5/5`,
`1/3/5/7` (natural growth, uncapped). Level 0 is 1 bit in every ladder and is not a
variable: it is the binary frame, which is the project's premise.
**Workload:** X-20's, unchanged — 320×240 synthetic and the repo's real 752×480
test image binarized by the reference pipeline's own `rl_fast_edge_filter_wide` at
`edge_threshold 17`; 31×31 windows, `lk_max_level 3`, 20 iterations, eps 0.03,
minEig 0.001, `seal_params.yaml` verbatim. Timing at 640×480 and at 94×60.
**Metric:** RMS and max endpoint error in px against ANALYTIC ground truth, percent
tracked and stuck count; bytes by stage with `operator new` counted; ns/frame.
**Method:** `Pyramid<WordType, LevelBits...>` for both frames, per-level
`derivativeX`/`derivativeY` into `SignedQuantMat<N, WordType>`, assembled into
`LKLevels<WordType, LevelBits...>` and run through `calcOpticalFlowPyrLK`. Harness
committed with the entry.

**PLATFORM, AND WHY THIS ENTRY WILL BE `PARTIAL` BEFORE IT IS `DONE`.** The
accuracy and footprint axes are **exact and device-independent** — deterministic
integer and `double` arithmetic, and byte counts — so they close on the development
machine and are authoritative there. The **ns/frame axis is the reference
device's** and nothing else may close it: X-22's caveat 1 measured the same kernel
moving **1.46×** between two binaries built from unchanged source, so a timing
taken anywhere else would not survive contact with the ladder that has to be chosen
on it. This entry therefore stays **PARTIAL** until the speed axis runs on the Pi,
and a ladder is not adopted into a D-record until it does.

**PLATFORM:** development machine (x86_64). Accuracy and footprint only — both are
exact and device-independent, which is why they close here. **The ns/frame axis has
NOT been measured**, so this entry is `PARTIAL` and no ladder is promoted to a
D-record, exactly as the rule pre-declared.

**Harness validity, checked before any row was read.** Every `1/1/1/1` row below
reproduces [X-20](#x-20--hybrid-lk-accuracy-against-ground-truth-and-the-frontends-peak-footprint--done)'s
published number **exactly** — 3.2530, 1.2645, 2.2311, 3.5093, 5.8461, 4.5949,
8.2501 — through the generic-`N` code path rather than the hand-written one. The
tolerance, binarization, warps, eligibility rule and stuck rule are reached through
X-20's own functions, not re-derived.

**Result (a) — the real frame, 752×480, reference binarization, 141 eligible
points, ALL of them (this is the gate):**

| ladder | (1,0) | (0.25,0.25) | (0.50,0.50) | (0.75,0.75) | (2,−3) | rot 1° | scale 1.02 | bytes |
|---|---|---|---|---|---|---|---|---|
| 1 (one level) | 0.0017 | 0.2860 | 0.4587 | 0.6800 | 2.5358 | 3.4934 | 3.7678 | 276 480 |
| **1/1/1/1** *(ships)* | 3.2530 | 1.2645 | 2.2311 | 3.5093 | 5.8461 | 4.5949 | 8.2501 | 367 200 |
| 1/2/2/2 | **0.8356** | 1.6655 | 1.8877 | 1.8394 | 3.1247 | 7.1873 | 6.6770 | 427 680 |
| 1/3/3/3 | 1.1092 | 1.6837 | 2.0384 | 1.8483 | **3.0311** | 6.8964 | **5.6894** | 488 160 |
| 1/3/4/4 | 1.4708 | 1.6776 | 2.0351 | 1.9732 | 3.2101 | 6.9827 | 5.9344 | 502 560 |
| 1/3/5/5 | 1.5133 | 1.6784 | 2.0407 | 1.9687 | 3.4883 | 6.9169 | 6.1521 | 516 960 |
| 1/3/5/7 | 1.5151 | 1.6783 | 2.0412 | 1.9654 | 3.4867 | 6.9191 | 6.1376 | 522 720 |

RMS px. **Not one non-stationary cell is inside the 0.25 px tolerance at any
depth.** Stationary is 0.0000 at every ladder. Every ladder tracked 141/141.

**Result (b) — the SAME points restricted to the 58 whose 31×31 window is fully
inside EVERY level (X-20's own control for deviation (ii)):**

| ladder | (1,0) | (0.25,0.25) | (0.75,0.75) | (2,−3) | (6,4) | (12,−8) |
|---|---|---|---|---|---|---|
| 1 (one level) | 0.0024 | **0.2278** | 0.5916 | 2.2820 | 6.7355 | 14.1441 |
| **1/1/1/1** *(ships)* | 1.4742 | 0.3994 | 1.3482 | 4.4581 | 1.7816 | **0.0009** |
| **1/2/2/2** | **0.0010** | 0.3847 | 0.6061 | **0.0014** | **0.0002** | **0.0001** |
| 1/3/3/3 | 0.5334 | 0.3312 | 0.8172 | **0.0013** | **0.0003** | **0.0001** |
| 1/3/4/4 | 0.8585 | 0.3441 | 0.8182 | 1.0820 | **0.0003** | **0.0001** |
| 1/3/5/5 | 0.8567 | 0.3443 | 0.8498 | 2.0650 | **0.0002** | **0.0001** |
| 1/3/5/7 | 0.8595 | 0.3443 | 0.8484 | 2.0641 | **0.0003** | **0.0001** |

Bold is inside tolerance. 58/58 tracked in every cell.

**Conclusion — and the hypothesis was wrong in BOTH directions, which is why the
rule was written down first.**

1. **BAND C FIRES ON THE GATE.** No ladder brings the four-level tracker inside
   X-20's tolerance on the full 141-point set, at any depth up to `1/3/5/7`. **A
   deeper alphabet is not what fixes T3.8's miss**, and E-7 as posed — "how many
   bits does each level need to preserve accuracy" — presupposed a cause that the
   measurement does not support. The tolerance was not widened and nothing was
   tuned.
2. **BAND D ALSO FIRES, AND ITS REMEDY DOES NOT DISSOLVE IT.** Accuracy is not
   monotone in depth: it is **peaked at 2 bits** and gets WORSE with more. On
   `(1,0)` unclipped, 1 bit gives 1.4742, **2 bits gives 0.0010 — a factor of
   1474** — and 5 bits gives back 0.8567. Band D's prescribed check was applied and
   excludes the obvious artifact: **every ladder tracked every point** (141/141,
   58/58), so the rows are measured over IDENTICAL point sets and the
   non-monotonicity is real rather than a changing denominator. `minEigThreshold`
   rejection is likewise not involved, for the same reason.
3. **Hypothesis 2 is refuted too, in the good direction.** It predicted no ladder
   would reach the one-level number. On `(1,0)` unclipped `1/2/2/2` returns
   **0.0010 against one level's 0.0024**, and on every large motion it beats one
   level by four orders of magnitude.
4. **THE DOMINANT RESIDUAL IS CLIPPING, NOT QUANTISATION — and that relocates the
   problem onto a decision the project took deliberately.** X-20 attributed "about
   half" the four-level error to the clipped coarse window. Measured directly: on
   `(1,0)`, `1/2/2/2` goes from 0.8356 over all 141 points to **0.0010** over the
   58 that never clip. Clipping was not half of that ladder's error, it was
   essentially all of it. Only **58 of 141 points (41%)** have a 31×31 window
   inside all four levels — which is exactly the population the reference's
   `winSize`-wide reflected border exists to serve, at the 1.24×-per-level cost
   [deviation (ii)](../bincv-cpp/include/bincv-cpp/ops/opticalFlow.hpp) declined.
5. **1-bit coarse levels ARE genuinely broken, so X-20 was half right.** `1/1/1/1`
   fails unclipped at `(1,0)`, `(2,−3)` and `(6,4)` where `1/2/2/2` is exact. The
   histogram says why, and it is not about sub-pixel localisation: down a 1-bit
   ladder the edge map is **thinned away** — level 3 retains 154 set pixels of
   5 640 against `1/2/2/2`'s 1 028. What the second bit buys is **content
   survival**, not precision.
6. **The depth regression is real but its MECHANISM is not established, and two
   explanations remain open.** (i) The bit-sliced covariance and residual weight
   plane pairs by `2^(i+j)`, so a few high-magnitude pixels dominate a window whose
   sub-pixel accuracy comes from averaging many edge crossings — which predicts
   degradation at SMALL motion and none at large, and that is exactly the observed
   pattern (`(12,−8)` is exact at every depth; `(2,−3)` fails from 4 bits;
   `(1,0)` from 3). (ii) `1/2/2/2`'s upper levels collapse to two distinct values
   anyway, so its advantage may be density preservation rather than precision.
   **These were not separated here** and the entry does not choose between them.
7. **`requantizeBoxSum` is excluded as the cause.** It is
   `round(sum/4 · (2^NOut−1)/(2^NIn−1))` — a faithful average renormalized to the
   output alphabet — at every depth, verified against the level histograms, so the
   ladders differ in precision and not in transfer function.

**Decision:** **None taken, and that is the pre-registered outcome, not a stall.**
Band C requires the remainder to be registered rather than absorbed, and the rule
forbids adopting any ladder before the speed axis runs on the reference device.
Two experiments are registered from this entry:
[E-14](ARCHITECTURE.md#register) (the coarse-level window border — now known to be
the dominant term, not a secondary one) and
[E-15](ARCHITECTURE.md#register) (why accuracy peaks at 2 bits). **`1/2/2/2` is the
provisional leader on accuracy and is also the cheapest non-1-bit ladder** — 427 680
B against `1/3/5/7`'s 522 720 B — but it is not adopted until E-14 is settled,
because a 41% keypoint loss is a larger effect than anything this entry measured.

**Method:** `tests/test_opticalflow.cpp`, `Flow.X24_LadderSweep_RealFrame_uint32_t`
and `Flow.X24_LadderSweep_Synthetic_uint32_t`, built on `LadderFrontend` /
`runLadder`, which reuse X-20's `measure()`, `eligiblePoints()` and
`unclippedAtEveryLevel()` verbatim.

---

# Pending

Registered in [ARCHITECTURE §9](ARCHITECTURE.md#9-open-questions-and-planned-experiments),
scheduled as tasks in [TASKS.md](TASKS.md). Each runs **in the phase whose code it
gates**, not at the end.

**E-8 has closed too**, in Phase 3 and in the task whose code it gates
([X-14](#x-14--horizontal-decimation-for-the-pyramid--done),
[D-17](ARCHITECTURE.md#d-17-horizontal-decimation-is-word-local)) — with the
answer that its own framing was wrong.

**E-1, E-2 and E-3 have closed** on the reference device —
[X-9](#x-9--does-row-alignment-earn-its-memory--done),
[X-10](#x-10--default-word-width--done),
[X-11](#x-11--incremental-versus-recomputed-window-reductions--done). Phase 2 has
no open experiment left, and the project has no provisional decision left.

**E-10 has now closed as well**, in Phase 3 and in the task whose code it gates
([T3.11](TASKS.md#t311--rolling-response-map-e-10--done),
[X-23](#x-23--the-rolling-response-ring-against-the-frame-sized-response-map--done),
[D-22](ARCHITECTURE.md#d-22-the-corner-response-streams-over-a-three-row-ring-and-that-is-the-recommended-path)),
and it leaves this table. **Phase 3 has no SCHEDULED experiment left** — every row
below runs in Phase 4, and the only Phase 3 question still open, E-11, is
unscheduled and was left untouched by X-23 deliberately. Like E-8, E-10 closed with
part of its own framing wrong: it was registered as a footprint win to be **bought**
with compute, and the streaming form is **smaller AND faster**, so there was no
purchase to make.

| ID | Question | Task | Runs during |
|---|---|---|---|
| E-13 | Does the per-row partial accumulator still pay above N = 1, where it is O(N²) per row against work that is O(N²) per word — and can the answer be measured free of the code-layout drift [X-22](#x-22--what-an-n-bit-pyramid-level-costs-the-lk-covariance--done) hit? | T4.1 | Phase 4 |
| E-12 | How much of the `ops/` kernel's per-row cost is genericity that is not in N — runtime `BorderType`, the word type, the argument contract — and which of them? | T4.1 | Phase 4 |
| E-7 | Bits needed per pyramid level | T4.1 | Phase 4 |
| E-6 | Hybrid LK versus binary block matching | T4.2 | Phase 4 |
| E-5 | End-to-end accuracy, footprint, speed | T4.3 | Phase 4 |
| E-9 | Per-level word width down the pyramid | — | unscheduled; spun out of [X-10](#x-10--default-word-width--done), which priced both sides |

(E-8 was registered in ARCHITECTURE §9 and never listed here until it was
about to run; it is now closed and has left the table. **E-4 has now left it
too** — [X-21](#x-21--does-generic-n-cost-the-specialized-n1-and-ternary-paths-anything--done)
closed it and [D-21](ARCHITECTURE.md#d-21-generic-n-is-not-capped-and-the-n--1-specialization-is-kept-as-a-test-oracle)
records the decision. **E-12 is what X-21 found while answering E-4** and could not
close on, because it is a different question: the cost is real, it is worst on the
upper pyramid levels T4.1 targets, and it is not genericity in N. It runs in the
phase whose code it gates, alongside E-7, rather than being carried as a note in a
log.)
