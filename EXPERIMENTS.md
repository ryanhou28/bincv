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

### X-24 · Pyramid level bit depths · `DONE`

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

**PLATFORM:** accuracy and footprint on the development machine (x86_64) — both are
exact and device-independent, which is why they close there. **The ns/frame axis is
the reference device's and ran there**, environment block in result (c). The entry
was `PARTIAL` between those two runs, exactly as the rule pre-declared.

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

**Result (c) — THE SPEED AXIS, on the reference device.**

Raspberry Pi 4 Model B Rev 1.5, aarch64, kernel 6.18.34+rpt-rpi-v8, g++ 14.2.0,
governor `performance`, pinned `taskset -c 3`, `throttled=0x0` **before and
after**, commit `110bf22`. Batch spread 0% on every cell. Harness
`benchmark/pyramid_depth_benchmark.cpp`, interleaved round-robin.

Level 0 = 640×480, four levels, 140 keypoints, 31×31 window — the only block in
which the coarse levels are actually used:

| ladder | Σ N² | build µs | vs 1 bit | track µs | vs 1 bit | **predicted** | bytes |
|---|---|---|---|---|---|---|---|
| 1/1/1/1 | 4 | 282.2 | 1.00× | 20 485.6 | 1.00× | 1.00× | 306 720 |
| **1/2/2/2** | 13 | 427.1 | 1.51× | 27 571.5 | **1.35×** | *3.25×* | 357 600 |
| 1/3/3/3 | 28 | 537.2 | 1.90× | 82 229.1 | 4.01× | *7.00×* | 408 480 |
| 1/3/4/4 | 42 | 588.8 | 2.09× | 82 358.0 | 4.02× | *10.50×* | 420 960 |
| 1/3/5/5 | 60 | 745.2 | 2.64× | 95 682.3 | 4.67× | *15.00×* | 433 440 |
| 1/3/5/7 | 84 | 790.9 | 2.80× | 117 898.4 | 5.76× | *21.00×* | 439 200 |

**THE COST MODEL WAS WRONG, AND IN THE DIRECTION THAT MATTERS FOR THE DECISION.**
Hypothesis 3 predicted tracking time scaling as `Σ_l N_l²` — 3.25× at `1/2/2/2`
and 21× at `1/3/5/7`. Measured: **1.35× and 5.76×**, an over-prediction of 2.4×
at the low end and 3.6× at the high end. The `20N²` popcounts per word are real
(they are counted in the kernel) but they do **not** dominate a tracked frame.
Three terms dilute them, and the entry does not claim to have separated them:
per-level work independent of `N` (region clipping, the 2×2 float solve, the
iteration and oscillation tests), the `4N` displaced row readers per row which are
**linear** in N, and — the one that is not a cost at all — **iteration count is
data-dependent and the ladders converge differently**, so this is an end-to-end
frame cost rather than a per-iteration one. End-to-end is the right metric for the
decision; it is simply not the same quantity the model predicted.

**The 94×60 block's track column is 1.00× at every ladder, exactly as
pre-declared, and it is not a refutation** — at that size the next level down is
47×30, whose height is under the 31-pixel window, so `usableLevelCount` stops at
one and the tracker never reads a level deeper than level 0 ([deviation
(vi)](../bincv-cpp/include/bincv-cpp/ops/opticalFlow.hpp)). Its informative column
is BUILD, where the per-row prologue [X-21](#x-21--does-generic-n-cost-the-specialized-n1-and-ternary-paths-anything--done)
flagged shows up slightly worse than at full frame — **3.06× against 2.80×** at
`1/3/5/7` — because a per-row cost is paid 5.4× more often per pixel there.

**The development machine gives a different set of ratios** — 1.94 / 6.51 / 6.93 /
8.04 / 10.66 against the device's 1.35 / 4.01 / 4.02 / 4.67 / 5.76 — consistently
higher, which is what D-6 predicts (aarch64 has no scalar popcount; `CNT` runs in
the NEON domain) and is why the device closes this axis and the laptop does not.

**What this changes:** `1/2/2/2`, X-24's accuracy leader, is **also cheap** —
1.35× the shipped ladder's tracking time, 1.51× its build, 1.17× its bytes. The
pre-registered band-B headline condition (*"if the chosen ladder's tracker exceeds
2× the 1/1/1/1 ladder, say so as a headline"*) **does not fire**. The binding
constraint on E-7 is therefore neither footprint nor speed; it is
[E-14](ARCHITECTURE.md#register), the coarse-level window border, which no depth
can address.

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

### X-25 · The coarse-level window border · `DONE`

**COMMITTED ON ITS OWN, BEFORE ANY ARM EXISTS.** Arms C and B are not written at
the time of writing; arms A and D can be evaluated from code that already ships.

**Gates:** [E-14](ARCHITECTURE.md#register) — and through it
[E-7](ARCHITECTURE.md#register)'s ladder, which
[X-24](#x-24--pyramid-level-bit-depths--done) could not adopt because this is the
binding constraint.
**Question:** Does the tracker need a border on its coarse pyramid levels — and if
so, is it the reference's padded copy, or something that costs nothing?

**Why this is now the main event.** X-24 measured `1/2/2/2` at **0.8356 px over all
141 real-frame keypoints and 0.0010 px over the 58 that never clip**. The ladder is
not the problem and the arithmetic is not the problem; **83 of 141 keypoints (59%)
have a 31×31 window that leaves some pyramid level**, and on those the clipped
window gives an ill-conditioned `A` and a one-sided `b` whose error is then
multiplied by `2^level` on the way down. binCV declined the reference's
`winSize`-wide reflected border deliberately — 1.24× per level at 640×480 — and
this entry is where that decision is re-examined against a measurement rather than
an argument.

**Hypothesis.** A point near a coarse level's edge does not need that level: it
needs to not be *ruined* by it. So the cheap arm (C) is expected to recover most of
the gap at zero bytes, because the information a clipped coarse window supplies is
worse than no information. The reference's border is expected to win on raw
per-point accuracy and to lose on footprint — and the coarse levels are where the
padding is most expensive in RELATIVE terms, which is the opposite of where
intuition puts it: at 640×480 a 31-pixel border is 1.24× the level, and at 80×60 it
is **3.6×**.

**Metric — pre-registered, and it must PENALIZE LOSING KEYPOINTS, because three of
the four arms trade points for accuracy and a per-point error alone would reward
throwing points away.**

* **YIELD** = (eligible keypoints whose endpoint error ≤ **1.0 px**, X-20's max
  tolerance) / (eligible keypoints). A keypoint dropped by policy counts as **not
  usable**. A keypoint returned but tracked badly also counts as **not usable**.
  This is the frontend's actual product.
* **RMS over the usable set**, reported beside it — a policy that yields many
  marginal points is not obviously better than one that yields fewer excellent
  ones, and neither number alone shows that.
* **Peak bytes** of the tracking stage, and **ns/frame** on the reference device.

**This is a deliberate strengthening of X-20's "≥ 80% of eligible points tracked",
and it is written down rather than slipped in.** That criterion counted a point as
tracked if `status == 1`, which on real content is **vacuous — 141 of 141 come back
tracked in every configuration X-24 ran, including ones with 18 px of error.** Yield
asks the question the old criterion was trying to ask.

**Arms:**
* **A — clip, all levels.** What ships. The baseline.
* **B — per-level `winSize` border, REPLICATE fill.** The reference's shape, with
  binCV's cheaper border: reflect-101 is a per-pixel index map, replicate is two
  mask-selects per word (deviation (iii) already made this choice for taps).
* **C — per-point start level.** A keypoint enters the pyramid at the **coarsest
  level whose window fully contains it** and is simply not tracked above that. No
  padding, no keypoint loss, **zero bytes**. This is the arm that did not exist
  before this entry.
* **D — reject any keypoint that would clip at any level.** Costs nothing and
  discards 59% of them. The honest lower bound on yield and upper bound on
  per-point accuracy.

**ORDER, AND IT IS PART OF THE RULE.** C and D cost **zero bytes**, so they are
measured FIRST, and **B is built only if neither reaches the gate below**. That is
[CLAUDE.md](CLAUDE.md)'s memory-wins tiebreak applied to the experiment's own scope
rather than only to its conclusion: a padded pyramid is a large, invasive change and
building it before knowing whether a free arm suffices would bias the comparison
toward using it. **If C reaches the gate, B is not built and this entry says so
explicitly rather than leaving it looking unexamined.**

**Decision rule** *(written before measuring)* — the accuracy gate stays X-20's,
unwidened: RMS ≤ 0.25 px over the usable set, max ≤ 1.0 px (which is yield's own
definition), on the real reference edge maps at four levels, measured on **both**
the shipped `1/1/1/1` ladder and X-24's `1/2/2/2` leader so E-14's answer and E-7's
cannot be chosen independently.

* **Band A — a free arm wins.** C or D reaches RMS ≤ 0.25 px at **yield ≥ 0.80**.
  Adopt it; B is not built; E-7's ladder unblocks and X-24's leader is adopted with
  it.
* **Band B — the border has to be paid for.** No free arm reaches it but B does.
  Then measure B's real peak footprint — including the coarse levels, where the
  relative cost is worst — and decide against CLAUDE.md's memory-wins rule **with
  the numbers in front of us**, reporting the yield-per-byte of all four arms.
* **Band C — nothing reaches it.** Then the border is not the whole story either,
  and this entry reports what remains rather than widening anything. Given X-24
  already relocated the cause once, a second relocation must be treated as evidence
  that the failure is in the pyramid's *use* rather than in any of its parameters.
* **Band D — D beats C.** Pre-declared because it would be surprising and
  informative: D discards points C keeps, so D can only win if C's retained points
  are tracked so badly they fall outside tolerance — which would mean **a partial
  pyramid is worse for a point than no pyramid at all**, and would make the
  per-point start level a bad idea rather than a cheap one.

**Variants:** ladders `1/1/1/1` and `1/2/2/2` × arms A, C, D (and B only under the
order rule above).
**Workload:** X-24's, unchanged — the repo's real 752×480 frame under the reference
binarization, 141 eligible keypoints, 31×31 windows, four levels,
`seal_params.yaml` verbatim; the same warps X-24 reports.
**Metric:** yield, RMS over the usable set, peak bytes, ns/frame.
**Method:** harness committed with the entry. Accuracy and footprint on the
development machine (exact, device-independent); ns/frame on the reference device
(`pi4`).

**PLATFORM:** development machine. Accuracy and byte counts only, both exact and
device-independent. **No speed axis was needed and none is reported**, because the
decision below is *keep what ships* — there is no new code on the default path to
time. Arm C's cost is discussed under "what is offered" and is **not measured**.

**Result — ladder `1/2/2/2` (X-24's leader), 752×480 real frame, 141 eligible
keypoints of which 58 never clip. YIELD %, then RMS over the usable set:**

| case | A clip *(ships)* | C per-point entry | D reject clipping | B padded *(bound)* |
|---|---|---|---|---|
| shift (1, 0) | 98.6 / 0.0009 | **100.0** / 0.0010 | 41.1 / 0.0010 | 97.9 / 0.0009 |
| shift (0.25, 0.25) | 92.9 / 0.2502 | **95.7** / 0.2490 | 39.0 / 0.2172 | 93.6 / 0.2513 |
| shift (0.75, 0.75) | 89.4 / 0.2609 | 87.2 / 0.2543 | 37.6 / 0.2577 | **90.1** / 0.2705 |
| shift (2, −3) | 94.3 / 0.0010 | 85.8 / 0.0010 | 41.1 / 0.0014 | **95.7** / 0.0010 |
| shift (6, 4) | **99.3** / 0.0006 | 77.9 / 0.0002 | 41.4 / 0.0002 | 93.6 / 0.0003 |
| rotate 1° | **89.2** / 0.3208 | 78.4 / 0.3027 | 41.0 / 0.3240 | 85.6 / 0.3011 |
| scale 1.02 | **88.7** / 0.3157 | 81.2 / 0.2950 | 42.1 / 0.3299 | 85.7 / 0.3162 |
| **bytes** | **427 680** | **427 680** | **427 680** | **589 968** |

Arm B's byte figure is the **reference's own per-level `winSize` border scheme**,
computed analytically; the measurement scaffold that produced its accuracy column
pads level 0 instead and is larger still (1 425 936 B), which is why the two are
reported separately and the scaffold's number is not used for comparison.

**Conclusion — BAND C, and the answer to E-14 is NO.**

1. **THE BIGGEST FINDING IS METHODOLOGICAL, AND IT CORRECTS X-24 RATHER THAN
   EXTENDING IT.** `rms(all)` — the statistic X-20, X-24 and this project's whole
   reading of T3.8's "miss" rested on — describes a distribution it does not fit.
   Arm A on `(1, 0)` has **rms(all) = 0.8356 px and yield 98.6% at rms(usable)
   = 0.0009 px**: 139 of 141 keypoints are tracked to a *thousandth of a pixel* and
   two are catastrophically wrong. The tracker was never broadly inaccurate. It has
   a small catastrophic tail, and an RMS over everything reports the tail as though
   it were the body.
2. **THAT INVALIDATES X-24's CLIPPING CONCLUSION, WHICH THIS ENTRY WITHDRAWS.**
   X-24 concluded clipping was "essentially all" of the error because restricting to
   the 58 never-clipping points moved `rms(all)` from 0.8356 to 0.0010. Measured by
   yield, **clipping costs about two keypoints out of 141 on that case, not 59%** —
   81 of the 83 clipped points are perfectly usable. The unclipped subset simply
   happened to exclude the outliers. The 59% figure was real as a count of clipped
   points and wrong as an attribution of error.
3. **A BORDER BUYS NOTHING, AND THAT VINDICATES DEVIATION (ii).** Arm B is
   **worse than or equal to arm A on yield in five of seven cases** and better by at
   most 1.4 points in the other two, for **1.38× the bytes** under the reference's
   own scheme. binCV declined the reference's `winSize`-wide padded copy of every
   level as an argued footprint decision; it is now a measured one. **Keep clipping.**
4. **Arm C is a real trade and is offered rather than adopted.** It is the best arm
   on small motion — **100% yield at 0.0010 px on `(1, 0)`, the only cell in the
   whole table that is perfect** — and the worst on large motion (77.9% at
   `(6, 4)` against arm A's 99.3%). The mechanism is not mysterious and was not
   anticipated: a point denied its coarse levels cannot capture a large
   displacement, which is what the pyramid is *for*. So `LKEntryLevel::DeepestFitting`
   ships as an option with that trade documented, and `Coarsest` stays the default.
   **Band D did not fire** — C beats D everywhere, so a partial pyramid is better
   than no pyramid, which is what band D was watching for.
5. **WHAT IS ACTUALLY LEFT IS THE LEVEL-0 FLOOR, AND IT IS NOT A PYRAMID
   PARAMETER.** `rms(usable)` sits at **0.25–0.32 px on the sub-pixel and
   non-translational cases in every arm**, including the padded one. X-20's own
   single-level real-frame number for `(0.25, 0.25)` — **no pyramid in the picture
   at all** — is 0.2860 px. So this residual is what a 1-bit level 0 gives on real
   edge maps, and no border, depth or entry policy moves it. X-20's 0.25 px
   tolerance was derived from "an effective count of four independent crossings" in
   a 31×31 window; on a 10.36%-set edge map that assumption is the thing now in
   doubt. Registered as **[E-16](ARCHITECTURE.md#register)**.
6. **Two relocations in a row is itself the finding.** X-24 moved the cause from
   quantisation to clipping; this entry moves it from clipping to a statistic and a
   representation floor. X-25's rule pre-declared that a second relocation should be
   read as evidence the failure is in the pyramid's *use* rather than in any of its
   parameters, and that is how it is read: **three pyramid parameters have now been
   measured and none of them is the problem.**

**Decision:** **E-14 answered NO — no border, no padding; deviation (ii) stands and
is now measured.** `LKEntryLevel` ships with `Coarsest` as the default and
`DeepestFitting` as a documented option. **E-7 unblocks**: with yield as the metric
`1/2/2/2` delivers 88.7–99.3% usable keypoints against `1/1/1/1`'s 75.9–88.7% at
1.17× the bytes and 1.35× the tracking time, so X-24's leader is confirmed on the
metric that matters. Promoted to [D-23](ARCHITECTURE.md#8-design-decisions).

**Method:** `tests/test_opticalflow.cpp`, `Flow.X25_CoarseLevelBorder_uint32_t`.

---

### X-26 · Hybrid LK against binary block matching · `DONE`

**COMMITTED BEFORE `ops/blockMatch.hpp` EXISTS.** Route (a) is not written at the
time of writing.

**Gates:** [E-6](ARCHITECTURE.md#register) · [T4.2](TASKS.md) — and it closes
[ARCHITECTURE §7.9](#79-known-hard-problems)'s two-route split, of which only
route (b) has ever been built.
**Question:** Does fully bit-parallel tracking — census/Hamming block matching at
integer pixels — match hybrid LK's accuracy, and what does it cost?

**Scope note, because [CLAUDE.md](CLAUDE.md) forbids template matching.** It does,
as an *operation*: `cv::matchTemplate` is deliberately out of scope. This is not
that. Route (a) is an internal tracker search named in §7.9 and scheduled as T4.2
since the roadmap was written, and nothing here exposes a template-matching API.

**THE FLOOR OF ROUTE (a) IS DERIVABLE BEFORE MEASURING, AND IS WRITTEN DOWN HERE SO
THE EXPERIMENT CANNOT BE READ AS DISCOVERING IT.** A matcher restricted to whole
pixels returns `round(d)`, so on a translation with fractional part `q` its per-axis
error is exactly `min(q, 1−q)`. Over `q` uniform on `[0, 1)` the per-axis RMS is
`sqrt(2 ∫₀^0.5 q² dq)` = **0.2887 px**, and over two independent axes
**0.408 px**. So **arm (a1) cannot meet X-20's 0.25 px tolerance, by construction
and not by implementation quality** — and any measurement of it near 0.41 px is a
confirmation that the search works, not a finding about accuracy.

**What makes this worth running anyway is [X-25](#x-25--the-coarse-level-window-border--done).**
Route (b)'s `rms(usable)` on real edge maps is **0.25–0.32 px**, not the 0.0009 px
its best cells suggest — so the gap between "irreducibly continuous" LK and a
whole-pixel matcher is **much smaller on real content than the tolerance implies**,
and route (a) pays no floating point at all. That comparison was not available
before X-25 and is the reason this entry is worth the kernel.

**Hypothesis.** (a1) lands near 0.41 px and fails the tolerance as derived. (a2) —
the same search with a **parabolic fit to the Hamming cost surface** around the
integer minimum, three costs per axis, no extra search — recovers most of the
sub-pixel term for a handful of arithmetic operations per point and lands close to
route (b). Route (a) is expected to WIN on speed at small search radii and lose
badly as the radius grows, because its cost is `O(R²)` per level where LK's is
`O(iterations)`.

**THE ASYMMETRY THAT MUST NOT BE HIDDEN.** Hamming distance is defined on bits, so
route (a) is a **1-bit** algorithm; route (b) has just been shown to do better on
X-25's `1/2/2/2` ladder than on `1/1/1/1`. Comparing route (a) against route (b)'s
best ladder would therefore be comparing an algorithm against a representation. So
**both comparisons are reported**: route (a) against route (b) on the SAME `1/1/1/1`
ladder (the algorithm question) and against route (b) on `1/2/2/2` (the practical
question, which route (a) cannot enter without an N-bit cost function).

**Decision rule** *(written before measuring)* — metric is
[X-25](#x-25--the-coarse-level-window-border--done)'s **yield** (eligible keypoints
tracked within X-20's 1.0 px, over ALL eligible keypoints) with `rms(usable)`
beside it, plus peak bytes and ns/frame on the reference device.

* **Band A — route (a) wins outright.** An arm of (a) reaches yield within 2 points
  of route (b) on the same ladder **and** `rms(usable)` within tolerance **and** a
  material win — ≥ 1.3× — on footprint or speed. Then route (a) replaces route (b)
  and §7.9's route (b) becomes the fallback.
* **Band B — route (a) is cheaper but less accurate.** It wins speed or bytes
  materially but loses accuracy. Then **neither is adopted blindly**: report the
  yield-per-millisecond of both, and record that the choice belongs to the
  integrating pipeline, since a VIO frontend that RANSACs its correspondences may
  rationally prefer more, cheaper, noisier points. This is the outcome the old
  one-line rule in TASKS.md ("switch only if accuracy is within tolerance and the
  win is material; otherwise hybrid stands") would have thrown away, and it is the
  most likely one.
* **Band C — route (a) is not cheaper.** Then route (a) is **closed**, §7.9's split
  resolves to route (b), and the roadmap stops carrying it.
* **Band D — (a2) beats route (b) on accuracy.** Pre-declared because it would
  overturn §7.9's central claim that LK's accuracy comes from its continuous
  formulation. If a parabolic fit to a Hamming surface matches a Gauss-Newton solve
  on real content, then the continuity was never load-bearing and D-20 needs
  revisiting.

**Variants:** (a1) integer block matching, (a2) + parabolic sub-pixel, (b) hybrid LK
— on ladders `1/1/1/1` and, for (b) only, `1/2/2/2`. Search radius swept, since it
is route (a)'s whole cost story.
**Workload:** X-25's, unchanged — the repo's real 752×480 frame, reference
binarization, 141 eligible keypoints, 31×31 windows, four levels, the same warps.
**Metric:** yield, `rms(usable)`, peak bytes, ns/frame.
**Method:** `ops/blockMatch.hpp` and a harness, both committed with the entry.
Coarse-to-fine over the same ladder route (b) uses, so the two differ in the SEARCH
and in nothing else.

**PLATFORM:** accuracy and bytes on the development machine (exact,
device-independent); **speed on the reference device** — Pi 4 Model B Rev 1.5,
governor `performance`, `taskset -c 3`, `throttled=0x0` before and after, 0% batch
spread.

**Result (a) — accuracy, 752×480 real frame, YIELD % / `rms(usable)`:**

| case | (b) LK `1/1/1/1` | (b) LK `1/2/2/2` | (a1) R=2 integer | (a2) R=2 sub-pixel | (a2) R=4 |
|---|---|---|---|---|---|
| shift (1, 0) | 85.1 / 0.0014 | **98.6** / 0.0009 | 68.8 / 0.0000 | 68.8 / 0.0180 | 73.8 / 0.0181 |
| shift (0.25, 0.25) | 88.7 / 0.2900 | **92.9** / 0.2502 | 70.9 / 0.3873 | 69.5 / **0.2408** | 70.9 / 0.2420 |
| shift (0.75, 0.75) | 75.9 / 0.2725 | **89.4** / 0.2609 | 57.4 / 0.3869 | 56.7 / **0.2347** | 60.3 / 0.2410 |
| shift (2, −3) | 81.6 / 0.0006 | **94.3** / 0.0010 | 73.8 / 0.0981 | 73.8 / 0.0999 | 66.7 / 0.0193 |
| shift (6, 4) | 87.1 / 0.0015 | **99.3** / 0.0006 | 73.6 / 0.0000 | 73.6 / 0.0190 | 75.0 / 0.0180 |
| rotate 1° | 76.3 / 0.3036 | **89.2** / 0.3208 | 63.3 / 0.4449 | 63.3 / **0.2594** | 64.7 / 0.2804 |
| scale 1.02 | 75.9 / 0.3122 | **88.7** / 0.3157 | 62.4 / 0.4583 | 62.4 / **0.2504** | 63.2 / 0.2446 |
| **bytes** | 367 200 | 427 680 | **122 400** | **122 400** | **122 400** |

**Result (b) — speed, reference device, 640×480, 140 keypoints, same binary:**

| stage | route (b) LK | route (a) R=2 | route (a) R=4 |
|---|---|---|---|
| build | 285.0 µs | **196.9 µs** (1.45× cheaper) | 196.9 µs |
| track | 14 024.5 µs | **13 006.3 µs** (0.93×) | 40 674.2 µs (2.90×) |
| pyramid-stage bytes | 306 720 | **102 240** (3.00×) | 102 240 |

**Conclusion — BAND B, the outcome the rule called most likely, and the one the old
one-line rule in TASKS.md would have discarded.**

1. **THE DERIVED FLOOR IS CONFIRMED, WHICH IS HOW THE SEARCH IS KNOWN TO WORK.**
   X-26 derived, before any code existed, that an integer matcher's error is
   `min(q, 1−q)` per axis — **0.408 px over two axes for uniform `q`**. Measured:
   (a1) gives **0.3873 and 0.3869 px** on the two sub-pixel translations, and
   **exactly 0.0000 px** on both integer translations. That is the prediction, to
   two figures, from both ends.
2. **ROUTE (a) IS 3.00× SMALLER AND THAT IS ITS REAL RESULT.** It forms no
   derivative, so it carries two frame ladders where route (b) carries two frames
   *and* two `SignedQuantMat` ladders. 122 400 B against 367 200 B at 752×480;
   102 240 against 306 720 on the device at 640×480. Its build is **1.45× cheaper**
   for the same reason.
3. **BUT IT LOSES ON THE FRONTEND'S PRODUCT.** Yield is **56.7–75.0%** against
   route (b)'s 75.9–88.7% on the *same* `1/1/1/1` ladder and 88.7–99.3% on
   `1/2/2/2`. Route (a) is worse than route (b) even where the representation is
   identical, so this is an algorithm result and not a ladder artefact — which is
   exactly why both comparisons were pre-registered.
4. **PER MILLISECOND THEY ARE A WASH; PER BYTE ROUTE (a) WINS OUTRIGHT.** On
   `(2, −3)`, route (b) at `1/1/1/1` returns 115 usable points in 14.02 ms
   (**8.2 pts/ms**) and route (a) at R=2 returns 104 in 13.01 ms (**8.0 pts/ms**).
   Per byte: **1.04 usable points per KB against 0.32** — route (a) is **3.2× more
   keypoint-efficient per byte** at equal keypoint-efficiency per millisecond.
   *(The `1/2/2/2` timing is X-24's, from a different binary; per X-22's caveat 1 a
   cross-binary ratio can move 1.46×, so it is indicative and the same-binary
   `1/1/1/1` comparison is the one to read.)*
5. **`searchRadius` IS THE WHOLE COST STORY AND BIGGER IS WORSE TWICE OVER.**
   R = 4 costs **2.90× LK** against R = 2's 0.93× — the `O(R²)` was predicted — and
   it is also LESS accurate in aggregate (`rms(all)` 10.3–13.4 px against R = 2's
   5.5–7.9), because a wider search finds more false minima. There is no radius at
   which route (a) becomes competitive by searching harder.
6. **BAND D PARTIALLY FIRED, AND IT REFINES [D-20](ARCHITECTURE.md#d-20-the-trackers-per-pixel-work-is-all-popcounts-only-the-solve-is-float)
   RATHER THAN OVERTURNING IT.** A parabolic fit to a Hamming cost surface —
   integer arithmetic and four extra window scores — reaches `rms(usable)` of
   **0.2347–0.2594 px, BETTER than route (b)'s 0.2502–0.3208** on every sub-pixel
   and non-translational case. So **LK's continuous formulation does not buy
   PRECISION on the points it matches; it buys ROBUSTNESS — which points match at
   all.** §7.9's claim that "LK's accuracy comes from its continuous formulation"
   is too coarse: the continuity is load-bearing for the 20–30 points per frame
   that route (a) loses, not for the localisation of the ones both find.

**Decision:** **NEITHER IS ADOPTED AS THE ONE TRACKER, and route (a) is NOT closed.**
Route (b) stays the default because yield is what a frontend produces and it is
materially higher. `ops/blockMatch.hpp` ships as the memory-constrained
alternative, documented with its 3.00× footprint advantage, its equal
yield-per-millisecond, and its 15–25 point yield deficit — because
[CLAUDE.md](CLAUDE.md)'s tiebreak covers speed against footprint and **this is
accuracy against footprint, which is the integrating pipeline's call and not this
repo's**: a VIO frontend that RANSACs its correspondences may rationally prefer
more, cheaper, noisier points, and ARCHITECTURE §1 puts that decision on the other
side of the boundary. Recorded as
[D-24](ARCHITECTURE.md#8-design-decisions). §7.9's two-route split **resolves to
both routes existing**, which is not what either arm of it anticipated.

**Method:** `ops/blockMatch.hpp`; `tests/test_opticalflow.cpp`,
`Flow.X26_BlockMatchVersusLK_uint32_t`; `benchmark/blockmatch_benchmark.cpp`.

---

## CORRECTION, 2026-08-21 — RE-MEASURED ON THE REFERENCE PIPELINE'S ACTUAL CONTENT

**The numbers above were measured with the DENOISE STAGE MISSING.**
`SEALProcessor::temporal_process` runs `median_filter(THREE_PIX_MEDIAN)` *then*
`rl_fast_edge_filter_wide`, and `seal_params.yaml` enables both;
`tests/test_opticalflow.cpp` implemented only the second until it was fixed. Over
1710 EuRoC V1_02_medium frames the stage moves the content from **14.14% set to
13.04% set**. The eligible keypoint count on the repo's frame drops **141 → 102**,
because a median filter removes exactly the isolated noise that was manufacturing
corners.

**BOTH ENTRIES' CONCLUSIONS SURVIVE, AND THE RE-RUN WAS DONE BEFORE THAT WAS
ASSERTED.** Every arm was compared *within* one content set, so the rankings were
predicted to hold; they do. What moved is stated below rather than quietly
substituted.

### X-25 — the conclusion sharpens

| case | A clip | C per-point | D reject | B padded |
|---|---|---|---|---|
| shift (1, 0) | **98.0** / 0.0013 | **98.0** / 0.0013 | 42.2 / 0.0016 | **98.0** / 0.0013 |
| shift (0.25, 0.25) | **95.1** / 0.2779 | **95.1** / 0.2744 | 42.2 / 0.2393 | **95.1** / 0.2780 |
| shift (0.75, 0.75) | 90.2 / 0.2552 | **91.2** / 0.2569 | 41.2 / 0.2428 | **91.2** / 0.2551 |
| shift (2, −3) | **97.1** / 0.0008 | 92.2 / 0.0008 | 42.2 / 0.0010 | **97.1** / 0.0008 |
| shift (6, 4) | **97.0** / 0.0001 | 84.2 / 0.0001 | 41.6 / 0.0000 | 94.1 / 0.0001 |
| rotate 1° | **93.1** / 0.2921 | 84.2 / 0.2766 | 41.6 / 0.2919 | **93.1** / 0.2923 |
| scale 1.02 | **86.6** / 0.2647 | 82.5 / 0.2483 | 40.2 / 0.1914 | 84.5 / 0.2634 |

**Arm B now equals arm A exactly in four of seven cases and loses the other
three.** The padded pyramid does not win a single cell. E-14's answer — *no border*
— is unchanged and now rests on a cleaner comparison. Arm C's large-motion
weakness is unchanged and so is its mechanism.

`rms(usable)` still sits at **0.25–0.29 px** on every sub-pixel and
non-translational case in every arm including the padded one, so
[E-16](ARCHITECTURE.md#register) — the level-0 floor — is untouched by the
correction. That is the finding this entry ended on and it survives intact.

### X-26 — ROUTE (a) IS SUBSTANTIALLY BETTER THAN REPORTED

| case | (b) `1/1/1/1` | (b) `1/2/2/2` | (a1) R=2 int | (a2) R=2 sub | (a2) R=4 sub |
|---|---|---|---|---|---|
| shift (1, 0) | 91.2 / 0.0015 | **98.0** / 0.0013 | 82.4 / 0.1543 | 80.4 / 0.0244 | 85.3 / 0.0241 |
| shift (0.25, 0.25) | 93.1 / 0.2809 | **95.1** / 0.2779 | 81.4 / 0.4233 | 80.4 / 0.2800 | 80.4 / 0.2784 |
| shift (0.75, 0.75) | 86.3 / 0.2737 | **90.2** / 0.2552 | 69.6 / 0.3914 | 69.6 / **0.2512** | 68.6 / **0.2432** |
| shift (2, −3) | 85.3 / 0.0012 | **97.1** / 0.0008 | 83.3 / 0.0000 | 83.3 / 0.0244 | 78.4 / 0.0236 |
| shift (6, 4) | 87.1 / 0.0002 | **97.0** / 0.0001 | 83.2 / 0.0000 | 83.2 / 0.0245 | **88.1** / 0.0242 |
| rotate 1° | 83.2 / 0.2535 | **93.1** / 0.2921 | 74.3 / 0.4590 | 74.3 / **0.2704** | 70.3 / **0.2650** |
| scale 1.02 | 81.4 / **0.2238** | **86.6** / 0.2647 | 74.2 / 0.4849 | 74.2 / 0.2857 | 71.1 / 0.2948 |

**Route (a)'s yield rises from 56.7–75.0% to 68.6–88.1%**, and the gap to route (b)
on the SAME `1/1/1/1` ladder narrows from 15–25 points to **2–12**. On `(6, 4)` at
R = 4 route (a) reaches **88.1% against route (b)'s 87.1%** — the first cell where
route (a) wins on the same ladder.

**The mechanism is the obvious one and it cuts route (a)'s way**: a median filter
removes isolated noise pixels, and isolated noise is exactly what manufactures
false minima on a Hamming cost surface. Route (b)'s gradient-based solve was
already averaging that noise away over a 31×31 window; route (a), which takes the
single lowest score, was not. **So the previous measurement understated route (a)
specifically, not both routes equally** — the correction is not a wash and it would
have been wrong to assume it was.

D-24's decision is **unchanged**: route (b) still leads on yield everywhere but one
cell, and route (a) still holds its **3.00× footprint** advantage, so both still
ship with route (b) as the default. But the trade is **materially more favourable
to route (a)** than D-24 recorded, and its claimed 15–25 point yield deficit is
corrected to **2–12 points**.

Band D's finding is reinforced: (a2)'s `rms(usable)` beats route (b)'s on
`(0.75, 0.75)`, `rotate` and — at R = 4 — several others, on a cleaner content set.
A parabolic fit to a Hamming surface remains more precise than the Gauss-Newton
solve on the points both find.


---

### X-27 · The 1-bit level-0 localisation floor · `DONE`

**COMMITTED BEFORE THE HARNESS EXISTS.**

**Gates:** [E-16](ARCHITECTURE.md#register) — and through it
[T3.8](TASKS.md)'s standing accuracy criterion, which has been a documented MISS
since X-20 and which three separate experiments have now failed to explain.
**Question:** Is X-20's **0.25 px RMS** tolerance reachable at all from a 1-bit
edge map — and if not, what is the representation's actual floor?

**Why it has come to this.** Three pyramid parameters have been measured and none
is the cause: [X-24](#x-24--pyramid-level-bit-depths--done) ruled out level bit
depth, [X-25](#x-25--the-coarse-level-window-border--done) ruled out the border and
the entry policy, and in every arm of X-25 — **including the padded one** —
`rms(usable)` sits at **0.25–0.29 px** on the sub-pixel and non-translational
cases. X-20's own single-level figure, with no pyramid in the picture at all, is
**0.2860 px**. The residual has stopped moving when the pyramid changes, which is
the signature of a limit that is not in the pyramid.

**WHERE THE 0.25 CAME FROM, AND WHICH HALF OF IT IS IN DOUBT.** X-20 derived it as:
*"A 1-bit frame locates an edge crossing to ±0.5 px; a 31×31 window averages many
crossings, so the aggregate must beat the single-crossing bound by at least a
factor of two (an effective count of four independent crossings — the modest form
of the claim)."* The ±0.5 px half is sound — it is quantisation. **The suspect half
is "four independent crossings"**, which was asserted, never measured, and has two
ways of being wrong on a 13%-set edge map: the set pixels lie on **connected
contours** rather than being independent samples, and an edge constrains motion
only **perpendicular to itself** (the aperture problem), so `N` edge pixels do not
supply `N` independent one-dimensional constraints.

**Method — AN ORACLE ESTIMATOR, WITH THE TRACKER ENTIRELY OUT OF THE LOOP.** For a
window and a known sub-pixel displacement `d`, the only thing a binary frontend
observes is `B_d = binarize(warp(gray, d))`. Sweep candidate displacements `δ` on a
fine grid, form `B_δ` the same way, and take `δ* = argmin_δ Hamming(B_δ, B_d)` over
the window. That is the maximum-likelihood estimate **under the exact forward
model**, from the binary observation alone.

**It is deliberately unfair to the tracker, and that is the point.** The oracle
knows the grayscale, the warp family and the binarization; no real estimator does.
So `RMS|δ* − d|` is a **floor**: it is what the representation permits, and nothing
that sees only the bits can do better. If the oracle cannot reach 0.25 px, the
tolerance was never reachable by anything.

**Decision rule** *(written before measuring)* — let `F` be the oracle RMS at the
shipped 31×31 window on the reference pipeline's own content.

* **Band A — `F` ≤ 0.20 px.** The tolerance WAS reachable and the tracker is
  leaving accuracy on the table. T3.8's criterion stands unchanged, the residual
  becomes a **tracker** question, and it is registered as one rather than being
  absorbed here.
* **Band B — 0.20 < `F` ≤ 0.35 px.** The tolerance sat inside the representation's
  own noise. Restate T3.8's criterion at a bound **derived from `F`** — not fitted
  to any tracker output — recording that X-20's number rested on an assumption this
  entry measured and contradicted. Report the split: how much of the tracker's
  0.25–0.29 px is representation and how much is tracker.
* **Band C — `F` > 0.35 px.** The tolerance was badly unreachable AND `F` would
  then exceed the tracker's own measured error, which is **impossible**: an
  estimator cannot beat the floor of the data it sees. **A band-C reading is
  therefore a bug in the oracle, not a finding**, and the entry must report a
  methodology failure and stop rather than restate any tolerance on the strength of
  it. Written down now so that the most flattering-looking outcome — "the
  representation is even worse than we thought" — cannot be reported as a result.
* **Band D — the scaling law, measured alongside.** `F` is also measured at 11×11,
  21×21, 31×31 and 41×41. If `F` falls as `1/√(area)` the independent-crossings
  model is right in KIND and only its constant was wrong. If `F` **plateaus**, the
  model is wrong in kind — the crossings are not independent — and that is the more
  interesting statement, because it means a bigger window buys nothing and
  [D-15](ARCHITECTURE.md#8-design-decisions)'s window sizing is affected.

**Also measured, because the derivation turns on it:** the actual count of set
pixels per 31×31 window on real content, and the count of distinct edge
ORIENTATIONS in it — the two numbers "four independent crossings" was standing in
for.

**Variants:** window 11/21/31/41; displacements spanning a full pixel in both axes.
**Workload:** the reference pipeline's own content through the corrected two-stage
preprocessing, on EuRoC V1_02_medium frames.
**Metric:** RMS and max of `|δ* − d|` in px; set-pixels and orientation count per
window. Accuracy only — this is a property of the representation, so there is no
speed axis and none is reported.

**PLATFORM:** development machine. Accuracy only — this is a property of the
representation, so there is no speed axis, as pre-registered.

**METHOD CHANGED BEFORE MEASURING, AND THE REASON IS THAT THE RULE'S SKETCH WAS
DEGENERATE.** X-27's rule proposed an oracle that formed candidates *the same way
as the observation*; the Hamming-nearest candidate is then the observation itself
and the floor would have been **exactly zero by construction**. The flaw was found
while writing the harness, before any number existed. The replacement inverts no
forward model, so it cannot recover `d` trivially. **The decision bands were not
touched.**

**Arm 1 — the partition method.** As `d` varies continuously the binarized window
changes only when some pixel's gradient crosses the threshold, so over `d ∈ [0,1)`
the observation takes finitely many values and `d` is partitioned into intervals
that are *indistinguishable from the bits alone*. That partition is the floor: the
best any estimator can do is report an interval's midpoint. 250 samples at
0.004 px; the repo's real frame through the corrected two-stage preprocessing.

| window | windows | mean set px | **distinct states / px** | mean interval | **FLOOR rms** | max |
|---|---|---|---|---|---|---|
| 11×11 | 50 | 44.2 | 17.3 | 0.0578 | 0.0648 px | 0.5000 |
| 21×21 | 64 | 127.4 | 27.2 | 0.0368 | 0.0231 px | 0.2980 |
| **31×31** | 85 | 217.5 | **29.3** | 0.0341 | **0.0254 px** | 0.3440 |
| 41×41 | 95 | 323.7 | 30.8 | 0.0324 | 0.0131 px | 0.1400 |

**Arm 2 — with sensor noise**, since arm 1 is noise-free and some of its state
transitions might be uninformative in practice. Observation binarized from a
**noisy** frame, candidates from clean ones, 31×31, 204 windows:

| σ (gray levels) | FLOOR rms | max | median |
|---|---|---|---|
| 0.0 | 0.0625 px | 0.6500 | 0.0100 |
| 0.5 | 0.0730 px | 0.6500 | 0.0200 |
| **1.0** | **0.1000 px** | 0.7100 | 0.0300 |
| 2.0 | 0.1120 px | 0.6500 | 0.0500 |
| 4.0 | **0.1742 px** | 0.8400 | 0.0900 |

**Conclusion — BAND A, and it is the opposite of what E-16 supposed.**

1. **THE 0.25 px TOLERANCE WAS ALWAYS REACHABLE, AND BY A WIDE MARGIN.** The
   representation permits **0.025 px** noise-free and **0.10 px at σ = 1** gray
   level — a realistic figure for the global-shutter sensor this content comes
   from. Even at **σ = 4**, which is a poor sensor, the floor is **0.174 px**,
   still inside the tolerance. **No tolerance is restated and none is weakened.**
2. **X-20's "four independent crossings" WAS WRONG — BUT CONSERVATIVE BY ~7×.** A
   31×31 window on real reference content resolves **29.3 distinct binary states
   per pixel of displacement**, not 4. The derivation's suspect half was suspect in
   the right place and wrong in the safe direction, which is why nothing downstream
   of it broke.
3. **BAND D FIRED: THE CROSSINGS ARE NOT INDEPENDENT, AND THE FLOOR PLATEAUS.**
   From 11×11 to 41×41 the set pixels grow **7.3×** (44 → 324) while distinct
   states grow only **1.8×** (17.3 → 30.8). Averaging does not go as `1/√area`,
   because the set pixels lie on **connected contours** and an edge constrains
   motion only perpendicular to itself. So a bigger window buys very little
   localisation — which matters for window sizing, and is registered rather than
   acted on here. *(It happens not to matter for the tolerance question, because
   the floor is already an order of magnitude below it at every window size
   measured.)*
4. **THEREFORE THE TRACKER IS THE LIMIT, AND THAT IS NOW A LOCATED PROBLEM RATHER
   THAN A DIFFUSE ONE.** The tracker delivers 0.25–0.29 px where the representation
   permits 0.10. **The gap is a factor of 2.5–3 and it belongs to the algorithm.**
   Registered as **[E-17](ARCHITECTURE.md#register)**, with the prime suspect named:
   **deviation (i)**, the previous window anchored on the integer grid.
   `ops/opticalFlow.hpp` already calls that "the concrete thing route (b) trades
   away", and it displaces the aperture by up to half a pixel — the right order of
   magnitude for a 2.5× gap on a sub-pixel measurement.

**Decision:** **T3.8's criterion STANDS UNCHANGED at 0.25 px RMS.** E-16 is closed
with the answer that the representation was never the constraint. The residual is
handed to E-17 as a tracker question, which is where three experiments' worth of
elimination now points. Recorded as
[D-25](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/level0_floor.cpp` (arm 1) and
`benchmark/level0_floor_noise.cpp` (arm 2).

---

### X-28 · T4.3a — the frontend end to end, over a real sequence · `PARTIAL`

**Gates:** [E-5](ARCHITECTURE.md#register) · [T4.3a](TASKS.md) — three of the four
[ROADMAP success criteria](ROADMAP.md#success-criteria).
**Decision rule:** **the ROADMAP's success criteria, unchanged.** They were written
before Phase 1 and are not restated here; this entry reports against them verbatim
so there is no opportunity to shape a criterion around a result.
**Workload:** EuRoC **V1_02_medium, all 1709 frame pairs**, 752×480, through the
reference pipeline's **two-stage** preprocessing (`median_filter` then
`rl_fast_edge_filter_wide(17)`). `seal_params.yaml` parameters verbatim.
**Denominator:** [CLAUDE.md](CLAUDE.md)'s — OpenCV doing the same semantic
operation **on the same binary content stored as `CV_8U`**. Both frontends see
bit-identical input, detect their own corners, maintain their own track sets and
re-detect on their own schedule, so **track lifetime is comparable** rather than
being one frontend's points scored by the other.

**Result — criterion 2, agreement with the reference frontend:**

| | binCV | OpenCV |
|---|---|---|
| corners detected (first frame, `maxCorners` 200) | 193 | 200 |
| tracks observed over the sequence | 10 279 | 10 108 |
| **median track lifetime** | **11 frames** | **12 frames** |
| per-frame track survival | 96.4% | 96.6% |

Flow-vector difference over 2 029 matched pairs: **median 0.0437 px, p90 0.1614 px,
p99 22.49 px, max 213.80 px**; **95.6% agree within 1 px**.

**REPORTED AS PERCENTILES, AND THAT IS THIS PROJECT'S OWN LESSON APPLIED.** The RMS
over all comparisons is **7.03 px** and it describes nothing: the body of the
distribution is at 0.04 px and a ~1% tail is at 22 px and beyond.
[X-25](#x-25--the-coarse-level-window-border--done) established that an RMS over
this distribution reports the tail as though it were the body, and that error cost
two experiments' worth of misattribution before it was caught. The RMS is printed
for completeness and is not the summary.

**Result — criterion 3, peak footprint over the frontend operation set:**

| | bytes |
|---|---|
| binCV — `1/2/2/2` pyramid ×2, derivative ladders, 3-row response ring | **436 704** |
| OpenCV — `CV_8U` pyramid ×2 with a 31 px border per level, `CV_32F` eigen map | 2 719 832 |
| **ratio** | **6.23× smaller** |

**Result — criterion 4, speed against the byte-per-pixel denominator:**

| | ms/frame |
|---|---|
| binCV | 21.43 |
| OpenCV | 1.54 |
| **ratio** | **0.07× — binCV is 14× SLOWER** |

**Conclusion.**

1. **CRITERION 3 IS MET, AND COMFORTABLY: 6.23×.** "Several-fold smaller peak
   footprint over the frontend operation set" is the criterion; 6.23× is several
   fold. Most of it is structural rather than clever — binCV carries no
   `winSize` border on any level ([D-23](ARCHITECTURE.md), measured) and no
   frame-sized float response map ([D-22](ARCHITECTURE.md)).
2. **CRITERION 2 IS MET FOR THE BODY OF THE DISTRIBUTION AND HAS A 1% TAIL.**
   Detection agrees to 3.5%, track lifetime to one frame in twelve, survival to
   0.2 points, and 90% of flow vectors agree to **0.16 px**. The ~1% beyond 22 px
   is the same catastrophic tail X-25 found and
   [E-17](ARCHITECTURE.md#register) is chartered to explain; it is not a new
   phenomenon and it is not swept up here.
3. **CRITERION 4 IS NOT MET, AND NOT NEARLY — BUT THE FIRST NUMBER CONFLATED TWO
   THINGS AND HAS BEEN SPLIT.** binCV is **14× slower** than OpenCV as it runs by
   default, and **6.3× slower than OpenCV pinned to ONE thread** (22.82 vs
   3.64 ms/frame over the same 400 frames). So **threading accounts for ~2.1× of
   the gap and SIMD plus algorithm for the remaining 6.3×**, which is the
   like-for-like figure and the one to quote. Both are reported because neither is
   honest alone: a multi-core denominator is what a user actually has, and a
   single-thread one is what isolates the code. The
   comparison is honest and unflattering: **binCV is scalar and single-threaded**,
   while OpenCV's `calcOpticalFlowPyrLK` and `goodFeaturesToTrack` are
   SIMD-vectorized and were running on **12 threads**. That is a like-for-like
   measurement of what ships today and it must be read that way — it is not a
   comparison of the two algorithms. **Phase 5 is exactly the work this number
   demands**, and the earlier device profiling says where: 99% of frontend time is
   in two windowed popcount reductions, and the per-pixel primitives are under 1%.
   **The criterion stays open and is NOT restated.**

**Why `PARTIAL`.** Criterion 4 is unmet, so E-5 cannot close. Criteria 2 and 3 are
measured and reported. This entry re-runs once Phase 5.1 exists.

**TWO HARNESS DEFECTS WERE FOUND AND FIXED WHILE BUILDING THIS, AND THE SECOND
LOOKED EXACTLY LIKE A binCV DEFECT.**
* Flow pairs were matched **by array index**. Two independently-detecting frontends
  share no ordering, so this compared unrelated points. It reported **zero**
  comparisons rather than wrong ones, which is how it was caught — a failure mode
  worth preferring.
* Corner capacity was passed as `maxCorners`. `goodFeaturesToTrack` ranks NMS
  survivors into the caller's array and applies the `minDistance` spacing filter
  **afterwards**, so a capacity of `maxCorners` truncates the pool before spacing
  thins it: capacity 200 yields **61** corners, capacity 20 000 yields **193**.
  The first reading looked like a 3.3× detection shortfall in binCV and was
  nearly recorded as one. **`CornerResult::candidatesTruncated` was reporting the
  truncation the whole time** — T3.11 added that flag for precisely this, and it
  was ignored. The harness now allocates 20 000 and reports the flag every run.

**Method:** `benchmark/frontend_sequence.cpp`.

---

### X-29 · The per-row partial accumulator above N = 1 · `DONE`

**COMMITTED BEFORE EITHER ARM EXISTS.**

**Gates:** [E-13](ARCHITECTURE.md#register) ·
[D-15](ARCHITECTURE.md#8-design-decisions) item 4.
**Question:** Does the per-row partial accumulator still pay above `N = 1` in
[§7.5](ARCHITECTURE.md#75-lk-gradient-covariance)'s bit-sliced covariance?

**Why it is worth the effort now, and it is a different reason than when E-13 was
registered.** [X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial)
measured the frontend end to end: **criterion 4 is unmet by 14×**, and the device
profiling says **99% of frontend time sits in two windowed popcount reductions** —
the corner response and LK's covariance/residual. This accumulator is inside one of
them. It is no longer a tidy-up; it is the first piece of Phase 5.1's target.

**What is in question.** `BitSlicedPairCounts<N>` is `4N²` counters — **4 at
`N = 1`, 64 at `N = 4`**. The per-row form zeroes all of them per row, fills them,
and adds them into the window total: roughly `3N² + N` adds plus `4N²` words of
zeroing **per row**, against work that is `O(N²)` **per word** — and a 31-pixel
window is **1–2 `uint64_t` words per row**. [D-15](ARCHITECTURE.md) item 4 chose the
per-row shape on X-11b's measurement **at `N = 1`** (1.08× at W = 31), where the
structure is 4 counters. Nothing has re-tested it at 64.

**THE CONFOUND IS THE POINT OF THIS ENTRY, AND IT IS MEASURED RATHER THAN
ASSUMED.** [X-22](#x-22--what-an-n-bit-pyramid-level-costs-the-lk-covariance--done)
already measured a window-wide accumulator **1.14–1.60× faster at N = 2, 3, 4** —
and declined to close on it, because that same entry measured **the same kernel
moving 1.46× between two binaries built from unchanged source**, and
`morphology_path_benchmark` records two instantiations in one object moving each
other ~10%. A 1.14× reading inside a 1.46× confound is not a result.

So this entry **measures the noise floor first**: the *same* arm is compiled into
**two different translation units** and both are timed. Call the resulting spread
**`L`**. Every comparison below is judged against `L`, not against zero. Each arm
also gets its own translation unit, the pattern `genericn_arms` and
`corner_streaming_arms` already use for exactly this reason.

**Decision rule** *(written before measuring)* — arms: **per-row** (ships) and
**window-wide**, at `N = 1, 2, 3, 4`, `uint32_t` and `uint64_t`, on the reference
device.

* **Band A — the choice is N-dependent.** Window-wide beats per-row at `N ≥ 2` by
  more than `L`, and per-row is not beaten by more than `L` at `N = 1`. Then
  `D-15` item 4 becomes **an `N = 1` statement** and the kernel selects on `N` at
  compile time — which costs nothing, since `N` is already a template parameter.
* **Band B — window-wide wins everywhere.** Faster by more than `L` at every `N`
  including 1. Then D-15 item 4 is **revised outright** and X-11b's 1.08× is
  re-examined, because it would mean the per-row shape never paid in this kernel.
* **Band C — inside the noise.** `|difference| ≤ L` at every `N`. Then **D-15 item 4
  stands, E-13 closes as "no measurable effect", and X-22's 1.14–1.60× is
  attributed to code layout** — which is the outcome X-22 itself suspected.
* **Band D — per-row wins at `N > 1`.** The opposite of X-22's reading. Then X-22's
  number was layout, and that gets recorded prominently as a caution: **a
  single-binary A/B in this codebase can invert a real ordering**, and two entries
  would then have demonstrated it.

**`L` IS REPORTED IN EVERY BAND, INCLUDING THE ONES WHERE IT IS NOT DECISIVE.** A
comparison whose noise floor is not stated cannot be checked by a reader.

**Variants:** per-row vs window-wide × `N` ∈ {1,2,3,4} × `uint32_t`, `uint64_t`;
plus the same-arm-twice pair that measures `L`.
**Workload:** 31×31 windows over a 640×480 level, the shipped default; sweep of
window positions so the region masks vary.
**Metric:** ns per window, interleaved round-robin, median with spread.
**Method:** one translation unit per arm; `benchmark/` following
`corner_streaming_arms`' shape. Reference device closes it.

**PLATFORM:** reference device — Pi 4 Model B Rev 1.5, governor `performance`,
`taskset -c 3`, `throttled=0x0` before and after. The development machine's run is
reported beside it because **the difference between the two platforms is itself a
result**.

**Result — `W vs P` is the ratio `P / W`, so above 1.0 means window-wide is
faster. `L` is `|P − P'| / min(P, P')`: the same algorithm in two translation
units, i.e. pure code layout.**

**Reference device (Cortex-A72):**

| N | P (per-row) µs | P′ µs | W (window-wide) µs | **L** | W vs P | verdict |
|---|---|---|---|---|---|---|
| 1 | 288.6 | 289.4 | 314.8 | **0.3%** | 0.917× | **P wins** |
| 2 | 924.4 | 923.8 | 829.8 | **0.1%** | **1.114×** | W wins |
| 3 | 2420.5 | 2421.5 | 1795.3 | **0.0%** | **1.348×** | W wins |
| 4 | 4355.0 | 4353.5 | 3489.0 | **0.0%** | **1.248×** | W wins |

**Development machine (x86_64), same source, same binary layout discipline:**

| N | **L** | W vs P | verdict |
|---|---|---|---|
| 1 | 5.8% | 0.846× | P wins |
| 2 | **10.6%** | 1.017× | **IN NOISE** |
| 3 | 4.3% | 1.208× | W wins |
| 4 | 0.0% | 1.116× | W wins |

**Conclusion — BAND A, cleanly, and the noise-floor arm earned its place.**

1. **THE PER-ROW SHAPE PAYS AT `N = 1` AND COSTS ABOVE IT.** The crossover is
   between 1 and 2, not somewhere in the middle: 0.917× at N = 1 and 1.114× at
   N = 2, against a noise floor of 0.3% and 0.1%. **D-15 item 4 becomes an `N = 1`
   statement**, and the kernel now selects with `if constexpr` — free, since `N` is
   already a template parameter.
2. **THE MECHANISM MATCHES THE STRUCTURE.** `BitSlicedPairCounts<N>` is `4N²`
   counters — 4 at N = 1, **64 at N = 4**. Per-row costs `~3N² + N` adds plus `4N²`
   words of zeroing **per row**, against 1–2 `uint64_t` words of real work per row.
   At four counters that is cheap enough for the dependency-chain break X-11b
   measured to win; at sixty-four it is not.
3. **X-22 WAS RIGHT TO DECLINE, AND FOR THE RIGHT REASON.** It measured
   1.14–1.60× and would not close because the same kernel moved 1.46× between
   binaries. Measured directly here, **the code-layout noise floor on the
   development machine reaches 10.6% at N = 2 — larger than the entire effect at
   that N**, which reads `IN NOISE` there and `W wins` on the device. A
   single-binary A/B would have been indistinguishable from a result.
4. **AND THE CONFOUND IS LARGELY AN x86 PHENOMENON, WHICH IS NEW.** `L` is
   **0.0–0.3% on the Cortex-A72** and **0.0–10.6% on x86_64** — a difference of an
   order of magnitude in the *noise*, not the signal. Every prior caution about
   code layout in this repository (X-22's 1.46×,
   `morphology_path_benchmark`'s ~10%) was measured on x86. **That does not retire
   the caution** — the arms must still be split, and this entry's own device
   numbers were only trustworthy because they were — but it does explain why the
   device has consistently given tighter spreads, and it is a reason to prefer the
   device for A/B work beyond the ISA argument already in "Measurement platforms".
5. **IT LANDS ON THE ADOPTED LADDER.** [D-23](ARCHITECTURE.md) adopted `1/2/2/2`,
   so **three of four levels run at N = 2 and take the 1.114×**, while level 0 at
   N = 1 keeps the per-row shape that suits it. The change is worth having exactly
   where the frontend spends its time.

**Bit-identity:** both forms add the same integers in a different order and
`size_t` addition is associative, so results are identical by construction.
`tests/test_covariance.cpp` passes 17 704 checks unchanged, and
`tests/test_opticalflow.cpp` 194.

**Decision:** `gradientCovariance<N>` selects the accumulator on `N`.
[D-15](ARCHITECTURE.md#8-design-decisions) item 4 is amended to say it is an
`N = 1` result. E-13 closes. Recorded as
[D-26](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/covacc_benchmark.cpp` with `covacc_arm_perrow.cpp`,
`covacc_arm_perrow_b.cpp` (the noise-floor arm) and `covacc_arm_window.cpp`, one
translation unit each; the two per-row arms share a body by inclusion so they
cannot drift.

---

### X-30 · Where the frontend's time goes, and Phase 5.1's target list · `DONE`

**Gates:** [E-12](ARCHITECTURE.md#register) — its second half — and
[X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial)'s unmet
criterion 4.
**Question:** How much of an `ops/` kernel's per-row cost is genericity that is not
in `N`, and which kernels is that worth asking about?

**THE REGISTERED TARGET TURNED OUT TO BE WORTH ALMOST NOTHING, AND THAT IS REPORTED
RATHER THAN QUIETLY SUBSTITUTED.** E-12 was registered against T3.5's derivative
(**+93% per row** against a hand-written control) *and* against "every `ops/` kernel
with a per-row prologue". X-28 then measured the frontend end to end, and the
derivative sits inside a build stage worth **0.7%**: **eliminating the entire build
stage caps the frontend gain at 1.0062×.** Answering E-12 precisely on the
derivative would be optimising six tenths of a percent. So this entry asks E-12's
question of the 99% instead, which is the half of its registration that still bites.

**Method:** splits taken **by difference**, so nothing is perturbed by a timer
inside a loop. `maxIterations = 0` runs LK's per-point setup, clipping, covariance
and `minEig` test and **no residual**; subtracting it from the full call separates
the 2×2 matrix from the iteration. `cornerMinEigenVal` is the response sweep alone;
`goodFeaturesToTrackStreaming` adds NMS, ranking and the spacing filter.

**Result — reference device, 640×480, ladder `1/2/2/2`, 140 keypoints, 31×31
window, four levels. Governor `performance`, `throttled=0x0`, spread ≤ 1%.**

| stage | ms/frame | share |
|---|---|---|
| **corner: response sweep** (`cornerMinEigenVal`) | **30.367** | **52.7%** |
| **LK: residual + solve** (`residualSums` × iterations) | **25.182** | **43.7%** |
| LK: covariance + setup (`gradientCovariance`, `minEig`, clipping) | 0.833 | 1.4% |
| corner: selection (NMS, ranking, spacing) | 0.773 | 1.3% |
| build: `pyrDown` ×2 + both derivative ladders | 0.424 | 0.7% |
| **total** | **57.579** | |

**Conclusion — PHASE 5.1's TARGET LIST IS TWO FUNCTIONS, AND THEY ARE THE SAME
SHAPE.**

1. **`cornerMinEigenVal`'s response sweep is 52.7% and `residualSums` is 43.7% —
   together 96.4% of the frontend.** Everything else in the operation set,
   summed, is 3.6%. Both are **windowed popcount reductions**, which is precisely
   what [D-6](ARCHITECTURE.md#d-6-bulk-only-reductions) reserved the NEON domain
   for: on aarch64 `CNT` lives in the vector registers and every word currently
   pays two register-domain crossings. **Phase 5.1 is one kernel shape, not a
   catalogue.**
2. **THE COVARIANCE IS 1.4%, WHICH PUTS [X-29](#x-29--the-per-row-partial-accumulator-above-n--1--done)
   IN PERSPECTIVE AND THE ENTRY SAYS SO.** X-29's accumulator win is real,
   device-measured and bit-identical — and at 1.114× on 1.4% of the frontend it is
   worth about **0.17% end to end**. It was the right answer to E-13's question; it
   was not the biggest lever available, and reporting it as though it were would
   misrepresent the profile that was measured immediately afterwards.
3. **The LK iteration, not the matrix, is what costs.** The residual is **96.8% of
   LK time** — `20N²` popcounts per word paid once per iteration per level, against
   the covariance's `3N² + N` paid once. Any further work on LK's arithmetic belongs
   in `residualSums`.
4. **X-23's streaming ring is confirmed as a footprint decision, not a speed one.**
   Corner selection — NMS, ranking, spacing — is **2.5% of detection**; the response
   sweep is 97.5%. [D-22](ARCHITECTURE.md) claimed the ring cost nothing in time and
   saved 11.83× in bytes, and this profile is consistent with that from the other
   direction.
5. **E-12's original question is answered by not needing an answer.** The
   derivative's +93% per-row genericity cost stands as measured by
   [X-21](#x-21--does-generic-n-cost-the-specialized-n1-and-ternary-paths-anything--done);
   it is simply not worth removing, because the whole stage it lives in is 0.7%.
   **That is a real result about where NOT to spend effort**, and it is the kind of
   thing a profile is for.

**Decision:** **E-12 closes.** No change is made to the derivative. Phase 5.1's
scope is fixed to the two windowed reductions above, in that order, and is recorded
as [D-27](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/frontend_profile.cpp`.

---

### X-31 · The corner response as bit-sliced box sums · `DONE`

**COMMITTED BEFORE THE KERNEL EXISTS.**

**Gates:** [D-27](ARCHITECTURE.md#8-design-decisions)'s Phase 5.1 target list, and
through it [X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial)'s
**unmet criterion 4**.
**Question:** `cornerMinEigenVal` is **52.7% of the frontend**. How much of it is
recoverable by computing the 3×3 covariance **row-at-a-time in bit-sliced form**
instead of per pixel, and how much more by skipping words with no gradient?

**WHY THIS AND NOT SIMD, MEASURED RATHER THAN ARGUED.** The obvious response to
"14× slower" is to vectorize. Two diagnostics say that would be the wrong first
move:

* **The corner response is 84% per-pixel FIXED cost.** Sweeping `blockSize`
  3/5/7/9 gives 14.42 / 18.40 / 24.99 / 32.97 ms while the window area goes
  1× / 2.8× / 5.4× / 9×. Fitting `T = A + B·bs²`: **A ≈ 12.1 ms, B·9 ≈ 2.3 ms.**
  Nine times the window work costs 2.29×. The fixed part is `clipRegion` +
  `blockWindow` + mask construction + call, **once per pixel**; the `sqrt` is only
  ~0.5 ms of it.
* **`residualSums` spends ~9.4 cycles per popcount** (measured directly), where a
  popcount is 1 cycle throughput — so it too is dominated by addressing, not
  arithmetic.

**Vectorizing the popcounts therefore attacks 10–16% of either kernel, and Amdahl
caps it near 1.2×.** The overhead has to go first, and removing it is portable
where SIMD is not.

**The reformulation, and it is the library's own existing technique.** For
`blockSize` 3 every quantity the response needs is a **3×3 box sum of a bit-plane**:
`xx = Σ magX`, `yy = Σ magY`, and `crossTerm = Σ(magX & magY & ~sel) −
Σ(magX & magY & sel)` with `sel = signX ^ signY`. A box sum of bits is computed
word-at-a-time with shifts and **full adders** — `h = a₋₁ + a₀ + a₊₁` is one full
adder into two planes, then three of those sum vertically into four planes (0..9).
**32 pixels per word instead of one**, against a current form that issues ~12
popcounts per pixel with **3 valid bits out of 32** in each.
[D-2](ARCHITECTURE.md) is this technique and `pyrDown`'s `boxSum4`
(`boxSumFullAdders(n) = 3n + 1`) is it already in the library; it was simply never
applied to the biggest kernel.

**BIT-EXACTNESS IS A PRECONDITION, NOT A BAND.** Box sums of bits are **exact
integers**, and `minEigenValue` takes the same three integers, so the `float` it
returns is identical *by construction* rather than to a tolerance — and therefore
so are the corners, their order and their count, exactly as
[D-22](ARCHITECTURE.md) held the streaming ring. **If exact equality turns out not
to be achievable, that is a finding to stop and report on, not a reason to relax to
a tolerance.** The border is where it will fail if it fails: a clipped window counts
only pixels inside the frame, which the bit-sliced form must reproduce by
zero-filling rows above/below the frame and relying on D-13's guarantee that
padding bits past `width` are zero.

**Arms, each in its OWN translation unit** — house practice, and
[X-29](#x-29--the-per-row-partial-accumulator-above-n--1--done) has just
demonstrated why (`morphology_path_benchmark` measured two instantiations in one
object moving each other ~10%):

* **C** — the shipped per-pixel `countCovariance` form. The control.
* **B1** — bit-sliced box sums, **no** sparsity skip.
* **B2** — bit-sliced box sums **plus** a word-level skip: if `magX | magY` is zero
  across the 3×3 neighbourhood of a word, all four box sums are zero, so `minEig`
  is exactly 0 for **32 pixels at once** and their `sqrt` is never taken. On a
  6.5–13%-set edge map that should fire often.

**Separating B1 from B2 is deliberate**: they are independent ideas and a combined
number would not say which one worked.

**Decision rule** *(written before measuring)* — `R` = C/B2 on `cornerMinEigenVal`
at `blockSize` 3, on the reference device, with bit-exact corners.

* **Band A — `R` ≥ 2.0×.** Adopt. Report B1 and B2 separately so the split between
  reformulation and sparsity is on the record, and re-run
  [X-30](#x-30--where-the-frontends-time-goes-and-phase-51s-target-list--done)'s
  profile, because the frontend's shape will have changed and Phase 5.1's target
  list may need reordering.
* **Band B — 1.2× ≤ `R` < 2.0×.** Adopt if the code is not materially harder to
  hold correct, and **state that the 84%-fixed-cost model over-predicted**, with
  where the remaining time went.
* **Band C — `R` < 1.2×.** **Do not ship it.** The cost model was wrong, a more
  complex kernel would be being adopted for nothing, and the entry reports that the
  A/B fit misled — which is the outcome the `blockSize` sweep was supposed to
  protect against and would mean it did not.
* **Band D — B1 is SLOWER than C.** Pre-declared because it would contradict
  `pyrDown`'s precedent: it would mean the full-adder network costs more than the
  per-pixel popcounts it replaces, and that needs explaining before S3 or SIMD is
  attempted, since both rest on the same reading of where time goes.

**The sparsity skip rate is reported as a number, not implied by the timing**, and
on real reference content as well as synthetic — a skip rate is a property of the
data, and synthetic texture would flatter or punish it arbitrarily.

**Variants:** C, B1, B2 × `uint32_t`, `uint64_t`; `blockSize` 3 (the shipped
default) and 5.
**Workload:** 640×480, the reference pipeline's own binarized content and the
synthetic control.
**Metric:** ms/frame on the reference device; skip rate; and corner-array equality
against C.

**PLATFORM:** reference device for the kernel timings (Pi 4, governor
`performance`, `throttled=0x0` before and after); development machine for the
sequence-level frontend measurement, which is a ratio against OpenCV on the same
machine.

**BIT-EXACTNESS — the precondition, met.** B1 and B2 are **bit-identical to the
control over the whole response map**, compared as raw `float` bits, on synthetic
texture and on four real reference frames spanning 8.69%–26.67% set. The library's
own `test_corner` passes **3 655/3 655** with the fast path wired in, including the
frame-map-versus-streaming equality suites.

**Result — kernel, reference device, 752×480 real reference content (10.23% set):**

| arm | ms | vs C |
|---|---|---|
| C — per-pixel (shipped) | 37.934 | 1.00× |
| B1 — bit-sliced box sums | 7.886 | **4.81×** |
| B2 — B1 + sparsity skip | **5.433** | **6.98×** |

Skip rate **39.3%** of words. **Band A fires** (R = 6.98 ≥ 2.0).

**The skip is real, and I nearly discarded it on one frame.** The first real frame I
sampled was **26.67% set** — an outlier against the sequence's ~13% average — and
skipped **1.7%** of words, so B2 looked like a dud. On three typical frames
(8.69–10.23% set) the skip fires **22.4%, 30.2% and 39.3%** of the time and is worth
**1.2–1.45× on top of the reformulation**. Separating B1 from B2, which the rule
required, is what made the correction visible rather than folded into one number.

**Result — the frontend. AND THIS IS WHERE THE ENTRY STOPS BEING GOOD NEWS.**

The streaming detector — the path the frontend uses — went **31.14 → 13.66 ms**
(2.28×) on the device. But the **sequence-level frontend barely moved**: 22.82 →
22.01 ms/frame over 400 real frames, **1.04×**.

**X-30's PROFILE WAS MEASURED AT AN OPERATING POINT THE FRONTEND DOES NOT RUN AT,
AND THAT INVALIDATES D-27's TARGET ORDERING.** X-30 timed **one detection and one
track per frame**. A real frontend re-detects only when tracks run down: measured
here, **12 re-detections in 399 frames — a 3.0% duty cycle**. So detection
contributes `13.66 × 0.030 ≈ 0.41 ms` of the 22.0 ms/frame — **under 2%**, not the
52.7% X-30 reported. The profile over-weighted it by roughly **33×**.

Corrected weighting of the real frontend:

| stage | share of real frontend |
|---|---|
| **LK tracking (`residualSums`)** | **~97%** |
| corner detection (amortized at 3% duty) | ~2% |
| build | ~2% |

**Conclusion.**

1. **The kernel result stands and the code ships.** 6.98× on the response, bit-exact,
   and `test_corner` unchanged at 3 655 checks. It is free at runtime and it is a
   large win for any caller that detects often — a frontend re-seeding every frame,
   or `goodFeaturesToTrack` used directly. It also validates the *method*: the
   `A + B·bs²` fit predicted 84% addressing overhead and removing it delivered
   ~5–7×, so the cost model was right about the kernel.
2. **BUT IT MOVES THE FRONTEND BY 1.04%, AND THE REASON IS THAT I PROFILED THE
   WRONG WORKLOAD.** This is the third time in this project a summary statistic has
   misdirected effort — after X-25's RMS-over-a-tailed-distribution and X-24's
   clipping attribution — and it is the same failure each time: **a number measured
   on something adjacent to the real thing.** X-30's per-frame profile was not the
   frontend's duty cycle, and nothing in it said so.
3. **D-27 IS CORRECTED, NOT DELETED.** Its ordering was wrong; its method was not.
   The right target is **`residualSums`, at ~97% of real frontend time**, and the
   diagnostic that matters is the one already measured: **~9.4 cycles per popcount**
   where a popcount is 1 cycle throughput, i.e. that kernel is also
   addressing-bound, by tap extraction. **S3 — deriving `t01` from `t00` and `t11`
   from `t10` by one shift instead of four independent extractions — is now the
   whole ballgame.**
4. **The SIMD conclusion is unchanged and strengthened.** Both kernels are
   addressing-bound; vectorizing popcounts still attacks ~10–15%. S3 comes first.

**Decision:** adopt the fast path — `cornerMinEigenValRow` dispatches to it at
`blockSize == 3`, which is `seal_params.yaml`'s value and the frontend's; other
block sizes keep the per-pixel form, the same shape D-22 uses. **The frame-map
`cornerMinEigenVal` has its own column-major sliding implementation and is
deliberately untouched** — it is not on the frontend's path and changing it would
be a second equality surface for no measured gain. Recorded as
[D-28](ARCHITECTURE.md#8-design-decisions).

**Not done, and named rather than left implicit:** the rule asked for `blockSize` 5
as well. The bit-sliced form implements **3 only**; a 5×5 box sum needs wider adder
networks (0..25, five planes) and the frontend does not use it. That is a scope cut,
not an oversight.

**Method:** `benchmark/cornerresp_benchmark.cpp` with `cornerresp_arm_perpixel.cpp`,
`cornerresp_arm_sliced.cpp` and `cornerresp_arm_sliced_skip.cpp`, one translation
unit each; B1 and B2 share a body by inclusion so the `Skip` template argument is
their only difference.

---

### X-32 · `residualSums`' tap extraction · `DONE`

**COMMITTED BEFORE THE ARM EXISTS.**

**Gates:** [D-28](ARCHITECTURE.md#8-design-decisions)'s corrected target list, and
[X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial)'s unmet
criterion 4.
**Question:** `residualSums` is **94.7% of the real frontend**. Its four bilinear
taps are extracted independently per word, but `t01` **is** `t00` shifted one pixel
and `t11` **is** `t10` shifted one pixel. What does deriving two from the other two
buy?

**THE TARGET IS CORRECTED, AND THE CORRECTION IS THE POINT.**
[X-30](#x-30--where-the-frontends-time-goes-and-phase-51s-target-list--done) timed
**one detection and one track per frame** and reported corner detection at 52.7%;
[X-31](#x-31--the-corner-response-as-bit-sliced-box-sums--done) then made that
kernel **6.98× faster, bit-exact**, and moved the frontend **1.04%**. Measured
inside the real sequence loop at the actual duty cycle — 25 re-detections in 599
frames — the split is:

| stage | ms/frame | share |
|---|---|---|
| **track (LK)** | **21.725** | **94.7%** |
| build | 1.036 | 4.5% |
| detect | 0.173 | 0.8% |

**Every optimisation from here is judged against this table, not against X-30's.**

**The mechanism, and why it should be nearly free.** `t01`'s pixel `x` is
`next[x + tapX + 1]`, which is `t00`'s pixel `x + 1`. On the word grid that is
`t01_i = (t00_i >> 1) | (t00_{i+1} << (bits-1))` — **exact**, including the
replicate border, because both taps clamp on the same absolute column. So two of
the four `ReplicatedShiftedRow::word()` calls per word can become a shift and an
or, and `t00_{i+1}` is the next iteration's `t00_i`, so it is computed once.

**Why it should matter:** `residualSums` was measured directly at **~9.4 cycles per
popcount** where a popcount is 1 cycle throughput, so ~90% of it is tap extraction
and addressing, not arithmetic. Halving the extraction should therefore be worth
substantially more than vectorizing the popcounts would.

**Decision rule** *(written before measuring)* — `R` = shipped / hoisted on
`residualSums`, reference device, and **bit-exact sums** as a precondition (the
derivation is an identity, so the ten integers must be identical, not close).

* **Band A — `R` ≥ 1.4×.** Adopt, and re-measure the frontend end to end rather
  than quoting the kernel ratio, because X-31 has just demonstrated how far those
  two can diverge.
* **Band B — 1.15× ≤ `R` < 1.4×.** Adopt if bit-exact and not materially harder to
  hold correct, and state that the ~90%-extraction reading over-predicted.
* **Band C — `R` < 1.15×.** **Do not ship.** The ~9.4-cycles-per-popcount reading
  would then not localise to tap extraction, and the entry must say where it does
  go before anything else is tried — including SIMD, which rests on the same
  reading.
* **Band D — slower.** The extra `word(i+1)` costs more than the `word(i)` it
  saves, i.e. the calls were already being CSE'd or inlined away. That would mean
  the cost is elsewhere in the loop and would invalidate the tap-extraction
  hypothesis outright.

**REPORTED ALONGSIDE, AND IT IS A CALLER-FACING TRADE RATHER THAN AN
OPTIMISATION:** `maxIterations` is the single largest speed lever in the frontend
and nothing had measured it. Over 599 real frames, LK time against track quality:

| iterations | track ms/frame | median track lifetime | flow within 1 px |
|---|---|---|---|
| 1 | 6.078 | 8 frames | 90.1% |
| 2 | 11.090 | 9 | 97.1% |
| 3 | 14.457 | 11 | 95.1% |
| 5 | 16.943 | 12 | 96.8% |
| **20** *(`seal_params.yaml`)* | **21.246** | **13** | **97.5%** |

**It is not free**: 3 iterations is **1.47× faster for 15% shorter tracks**. Track
lifetime is what a VIO estimator consumes, so this is accuracy against speed —
[CLAUDE.md](CLAUDE.md)'s tiebreak covers speed against footprint and
[§1](ARCHITECTURE.md#what-bincv-is-not) puts this one on the integrating pipeline's
side, exactly as [D-24](ARCHITECTURE.md) put route (a) there. **binCV keeps
`seal_params.yaml`'s 20 and documents the curve.** It is recorded here because a
caller cannot make that trade without the numbers, and because the points that
consume all 20 iterations are presumably the ones that never converge — the same
~1% tail [E-17](ARCHITECTURE.md#register) is chartered on.

**Variants:** shipped vs hoisted × `N` = 1, 2; `uint32_t`.
**Workload:** 31×31 windows over a 640×480 level, the shipped default.
**Metric:** ns per window on the reference device, bit-exact sums, then the frontend
end to end.

**PLATFORM:** development machine. The arm was rejected before it earned a device
run — see below.

**Result — BAND D. The hoisted form is SLOWER, and bit-exact.**

| arm | µs | vs S |
|---|---|---|
| S — shipped, 4 `word()` per word | 1861.3 | 1.000× |
| H — hoisted, 2 `word()` + 2 shifts | 1911.0 | **0.974×** |

Equality: **0 of 130 windows differ**, so the identity holds exactly as derived —
the derivation was right and the *premise* was wrong.

**Band D required finding where the time actually goes before anything else is
tried, including SIMD. It goes nowhere in particular:**

| variant | µs | share of full |
|---|---|---|
| full `residualSums` | 1878.1 | 100% |
| **taps only, no popcounts** | 257.9 | **13.7%** |
| popcounts only, no taps | 355.2 | 18.9% |

**TAP EXTRACTION IS 13.7%, NOT ~90%.** Halving it caps the kernel gain at ~1.07×,
and the branch that carries `word(i+1)` between iterations costs more than that.
The arm is **not shipped**.

**HOW I GOT THAT WRONG, BECAUSE THE ERROR IS REUSABLE.** I measured
`residualSums` at **~9.4 cycles per popcount**, observed that a popcount is 1 cycle
throughput, and concluded that ~90% of the kernel was "addressing". That inference
does not follow. The loop issues roughly `20N²` popcounts *and* a comparable number
of masks, ANDs and accumulates: at `N = 2` that is ~240 operations per word of which
popcounts are ~33%. **9.4 cycles per popcount is what a loop with ~5 other
operations per popcount plus dependency stalls looks like — it is not evidence that
any one thing dominates.** The ratio was real; the localisation was invented.

*(The two isolations sum to 32.6%, not 100%. The remainder is the two combined —
the dependency chain from `word()` through the mask and popcount into the
accumulator, and the register pressure of ten live tap words at `N = 2`. The
`popcounts only` arm reuses one word for all five taps, so it is register-friendly
in a way the real kernel is not; **18.9% is a floor on the arithmetic, not a
measurement of it**, and this entry does not claim otherwise.)*

**Conclusion — and it inverts the SIMD recommendation FOR THIS KERNEL.**

1. **`residualSums` is not addressing-bound the way the corner response was.** The
   corner response was **84% pure per-pixel overhead** producing nothing, which is
   why removing it gave 6.98×. This kernel has no comparable dead weight: it is
   doing a large amount of distributed work — masks, popcounts, accumulates —
   spread across a long dependency chain.
2. **So SIMD is now the reasonable lever HERE, and it was not there.** Masks,
   popcounts and accumulates all vectorize, and NEON would additionally relieve the
   register pressure by holding four words per vector. That is the opposite of the
   conclusion for the corner response, and the difference is measured rather than
   assumed: 84% removable overhead against 13.7%.
3. **The `9.4 cycles per popcount` figure should not be cited again as evidence of
   addressing cost**, in this entry or elsewhere. It appears in X-31's rationale and
   in [D-27](ARCHITECTURE.md#8-design-decisions); both are annotated.

**Decision:** **S3 is rejected and not shipped.** No change to `residualSums`.
`benchmark/residual_benchmark.cpp` and its two arms are committed anyway, because a
rejected optimisation with a measurement attached is what stops it being tried
again. Recorded as [D-29](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/residual_benchmark.cpp` with `residual_arm_shipped.cpp` and
`residual_arm_hoisted.cpp`, one translation unit each.

---

### X-33 · NEON for the bit-sliced signed sum · `DONE`

**COMMITTED BEFORE THE KERNEL EXISTS.** Phase 5.1's first vectorized kernel.

**Gates:** [D-29](ARCHITECTURE.md#8-design-decisions) — which established that
`residualSums` is arithmetic-bound and therefore *is* a SIMD target, unlike the
corner response — and [X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial)'s
unmet criterion 4.
**Question:** What does keeping the window's popcounts in NEON registers buy
`residualSums`, which is **94.7% of the real frontend**?

**WHY THIS KERNEL AND NOT THE OTHER ONE, WHICH IS THE WHOLE POINT OF THE LAST TWO
ENTRIES.** [X-31](#x-31--the-corner-response-as-bit-sliced-box-sums--done) showed the
corner response was **84% removable per-pixel overhead** — reformulating gave 6.98×
where vectorizing would have given ~1.2×. [X-32](#x-32--residualsums-tap-extraction--done)
showed `residualSums` is **not** like that: tap extraction is **13.7%**, there is no
comparable dead weight, and the work is masks, popcounts and accumulates that all
vectorize. **Same question, opposite answers, and the difference is measured.**

**The design, and it is [D-6](ARCHITECTURE.md#d-6-bulk-only-reductions)'s own
argument spent rather than restated.** aarch64 has no scalar popcount: `CNT` lives
in the vector registers, so every scalar `popcountWord` pays `fmov` in and `fmov`
out. `slicedSignedSum<N>` issues `2N²` popcounts per call — **8 at `N = 2`**, which
is the depth three of four levels of the adopted `1/2/2/2` ladder run at. Batching
them means:

* form the `N²` plane-pair ANDs in vector lanes rather than one at a time;
* `vcntq_u8` → `vpaddlq_u8` → `vpaddlq_u16` to get four per-word counts in four
  `u32` lanes;
* accumulate lane-wise, and **cross the register domain ONCE per call instead of
  `2N²` times**.

**This is why D-6 forbade a per-word popcount in the public API in the first
place** — so that the reductions would be shaped to allow exactly this later. This
entry is the first time that reservation is cashed in.

**BIT-EXACTNESS IS A PRECONDITION.** The sums are exact integers and the vector path
computes the same integers in the same weighting, so `residualSums`' ten values must
be **identical**, not close. Checked against the scalar path before any timing.

**Decision rule** *(written before measuring)* — `R` = scalar / NEON on
`residualSums` at `N = 2`, reference device, bit-exact.

* **Band A — `R` ≥ 1.5×.** Adopt behind `BINCV_HAVE_NEON`, keep the scalar path as
  the portable one and as the equality oracle, and **re-measure the frontend end to
  end** rather than quoting the kernel ratio — X-31 demonstrated how far those two
  can diverge.
* **Band B — 1.15× ≤ `R` < 1.5×.** Adopt only if the scalar path stays the
  reference and the NEON path is confined to one function. A dual implementation is
  a permanent correctness cost, and below 1.5× it has to earn that.
* **Band C — `R` < 1.15×.** **Do not ship.** Two implementations of one identity,
  for nothing, on the library's hottest kernel. Report where the time went instead
  — and note that this would be the *third* reading in a row (after S3 and after
  the 9.4-cycles inference) to say the cost is not where it was expected.
* **Band D — NEON is slower.** Then the scalar `popcountWord` is already being
  compiled to something better than the hand-written intrinsics — plausible, since
  GCC can auto-vectorize and the operands are contiguous — and **that is worth
  knowing before any further Phase 5.1 work**, because the whole phase assumes hand
  vectorization beats the compiler here.

**A CEILING IS MEASURED FIRST, AND IT CAN CANCEL THE ARM.** Before the real kernel,
a stripped loop that performs only the vector popcounts and accumulates — no taps,
no masks, no weighting — gives an upper bound on what any vectorization of this
inner loop can deliver. **If the ceiling is under 1.5×, the arm is not written**;
[X-32](#x-32--residualsums-tap-extraction--done) was a day spent on an optimisation
whose ceiling was 1.07× and which a five-minute bound would have killed.

**Variants:** scalar vs NEON × `N` = 1, 2 × `uint32_t`, `uint64_t`.
**Workload:** 31×31 windows over a 640×480 level, the shipped default.
**Metric:** ns per window on the reference device; bit-exact sums; then the frontend
end to end at the real duty cycle.
**Platform note:** the development machine is x86_64 and **cannot measure this at
all**. Correctness is checkable there only through the scalar fallback; every number
in this entry comes from the reference device.

**PLATFORM:** reference device throughout. The development machine is x86_64 and
cannot measure or even compile this path; its only role was checking that the
scalar fallback still builds and passes.

**The ceiling, measured FIRST as the rule required:** batched NEON popcount with
lane accumulators against scalar `__builtin_popcount`, everything else stripped —
**3.42×**, bit-identical. Above the 1.5× cancel threshold, so the arm was written.

**Result — BAND B.**

| arm | µs | vs NEON |
|---|---|---|
| NEON, `slicedSignedSum` batched | **2104.8** | 1.000× |
| scalar, per plane pair | 2618.5 | 0.804× |

**`R` = 1.24×**, and **0 of 130 windows differ**. On-device `ctest` passes
`test_opticalflow` with the vector path live — 194 checks including the per-pixel
oracle that compares the bit-sliced residual against a `long long` control at
`N = 1..5`.

**Stage effect, reference device:**

| | before | after | |
|---|---|---|---|
| LK, `maxIterations` 20 | 25.540 ms | **21.088 ms** | **1.21×** |
| LK residual + solve | 24.710 ms | 20.258 ms | 1.22× |

LK is 94.7% of the real frontend, so this is **~1.20× end to end**.

**Conclusion.**

1. **It works, it is bit-exact, and it is worth 1.24× — not 3.42×.** The ceiling
   measured the popcounts alone; the real kernel dilutes them with tap extraction
   (13.7%, [X-32](#x-32--residualsums-tap-extraction--done)), masks, accumulator
   updates and the loop structure. **The ceiling did its job — it authorised the
   work and it correctly bounded it** — and quoting 3.42× as the result would have
   been the same error as quoting X-31's 6.98× kernel win as a frontend number.
2. **Band B's condition is met, so it ships.** The vector path is confined to **one
   function**, the scalar path remains the reference *and* the equality oracle, and
   `UseNeon` exists so both spellings can be compared on the same machine. That
   flag is not a tuning knob — it is how the bit-exactness claim is checkable.
3. **[D-6](ARCHITECTURE.md#d-6-bulk-only-reductions) is cashed in, and this is the
   first time.** D-6 forbade exposing a per-word popcount so that reductions would
   be *shaped* to allow batching later. None of this would have been possible if
   callers held `popcountWord`: the eight plane-pair counts had to be inside one
   function for the domain crossing to be collapsible from eight to one.
4. **MOST OF THE CEILING IS STILL ON THE TABLE, AND WHERE IT IS IS KNOWN.** The
   horizontal add still runs **once per call** — ~620 domain crossings per window
   — where the ceiling amortized its extraction across the whole buffer. Carrying
   **vector accumulators across the window**, and reducing once per window instead
   of once per word per tap, is the remaining 2–3×. It is a larger change: `TapSums`
   becomes vector state and `residualSums` restructures around it. Registered as
   **[E-18](ARCHITECTURE.md#register)** rather than attempted here.

**Decision:** adopt. `slicedSignedSum` gains a NEON path at `N == 2, uint32_t`
behind `BINCV_HAVE_NEON && __aarch64__`; every other `N`, word type and platform
keeps the scalar path unchanged. Recorded as
[D-30](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/neon_ceiling.cpp` (the bound) and
`benchmark/residual_benchmark.cpp` with `residual_arm_shipped.cpp` /
`residual_arm_hoisted.cpp`, one translation unit each — the latter repurposed from
X-32's rejected arm into the scalar arm, since a rejected optimisation belongs in
the log as a number rather than in the tree as a second implementation.

---

### X-34 · The straddling window · `DONE`

**COMMITTED BEFORE THE ARM EXISTS. A CEILING IS MEASURED FIRST AND CAN CANCEL IT** —
X-33's procedure, which X-32 is the argument for.

**Gates:** [X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial)'s
unmet criterion 4, on the kernel that is **94.7% of the real frontend**.
**Question:** A 31-pixel window at an arbitrary offset spans **1.94 `uint32_t`
words on average** — it fits in one only when `x0 % 32 ≤ 1`, which is 2 cases in
32. So `residualSums` issues **twice the popcounts it needs**, each covering 15.5
useful pixels instead of 31. What does aligning the window into a single word buy?

**WHERE THIS CAME FROM.** An LK-against-LK measurement — same points, same window,
same bits, OpenCV pinned to one thread — put binCV at **4.11× slower** on the
shipped `1/2/2/2` ladder and **2.00×** on `1/1/1/1`, against an op-count model of
**1.29 popcounts per pixel at N = 1** where OpenCV spends ~0.4–0.9 SIMD ops. The
model matches the N = 1 measurement almost exactly. **1.29 rather than 0.625 is the
straddle**, and it is the one factor of the four that is pure waste rather than a
design choice:

* **the straddle — 1.94× of pure waste**, this entry;
* the **five-tap decomposition** — ten accumulated sums where OpenCV does two —
  which is [D-20](ARCHITECTURE.md)'s boundary being paid for and is *not* waste;
* the **N² plane pairs**, 2.06× measured for the `1/2/2/2` ladder
  ([D-23](ARCHITECTURE.md) bought that with accuracy);
* OpenCV being vectorized, which [D-30](ARCHITECTURE.md) has started answering.

**The mechanism.** Extract the window's 31 bits into bits `[0, 30]` of one word:
`(row[w0] >> s) | (row[w0+1] << (32 - s))` with `s = x0 % 32`, masked to 31 bits and
**guarded at `s == 0`, where the second shift is undefined**. Then every popcount
covers the whole window instead of half of it. **For the four taps the alignment is
FREE** — `ReplicatedShiftedRow` already shifts, so it only has to shift by a
different amount. Only `magX`, `magY`, `signX`, `signY` and `prev` need explicit
alignment: **8 planes at N = 2**, about 32 operations per row, against 80 popcounts
saved per row.

**THE CEILING, AND IT IS CHEAP BECAUSE THE HARDWARE ALREADY OFFERS IT.** Windows at
`x0 % 32 == 0` already occupy one word; windows at `x0 % 32 == 5` occupy two. Timing
`residualSums` on each is an upper bound on this optimisation **with the alignment
cost removed entirely**, and it needs no new kernel. **If that bound is under 1.3×,
the arm is not written.**

**Decision rule** *(written before measuring)* — `R` = unaligned / aligned on
`residualSums`, reference device, `N = 2`, bit-exact sums.

* **Band A — `R` ≥ 1.3×.** Adopt, and re-measure the LK stage and the frontend
  rather than quoting the kernel ratio.
* **Band B — 1.1× ≤ `R` < 1.3×.** Adopt only if the alignment stays inside
  `residualSums` and adds no interface. The window extraction is fiddly at `s == 0`
  and at the row's last word, and below 1.3× that risk has to be earned.
* **Band C — `R` < 1.1×.** **Do not ship**, and record that the straddle is not
  where the model said it was — which would make the *third* op-count inference in
  this project to fail against measurement, after S3's tap extraction and the
  9.4-cycles reading.
* **Band D — the aligned case is SLOWER.** Then the two-word path is being helped by
  something the one-word path loses — most plausibly that `visitRowWords`'
  head/tail masking is cheaper than a shift-based extraction — and that needs
  saying before the arm is written.

**Also to be reported, because it is now a different question than when it was
decided:** [D-23](ARCHITECTURE.md) adopted the `1/2/2/2` ladder on accuracy, with
its speed cost *estimated* at 1.35× from a confounded measurement. Isolated, it is
**2.06×**, and it was chosen when corner detection was believed to be 52.7% of the
frontend rather than 2%. **That decision is not reversed here** — it was an
accuracy decision and this entry measures speed — but the corrected price is put on
the record next to it.

**Variants:** aligned vs straddling windows (the ceiling); then shipped vs aligning
kernel, `N` = 1, 2.
**Workload:** 31×31 windows over a 640×480 level.
**Metric:** ns per window on the reference device, bit-exact sums, then the LK stage.

**Ceiling, measured first as the rule required:** aligned windows (`x0 % 32 == 0`,
one word) against straddling ones (`x0 % 32 == 5`, two words), shipped kernel
unchanged — word-visits **2.00×**, time **1.463×**. Above the 1.3× gate, so the arm
was written.

**Result — BAND A, and it EXCEEDS its own ceiling.**

| arm | device µs | vs shipped |
|---|---|---|
| shipped (NEON, per-word loop) | 2147.8 | 1.000× |
| **aligned, one word per row** | **1007.1** | **2.133×** |

Bit-exact: **0 of 130 windows differ**, and on-device `ctest` passes with the path
live.

**It beats the ceiling because the ceiling only measured the word count.** Halving
the words was worth 1.463×; the aligned kernel also deletes the **per-word loop and
its head/tail mask construction** — `visitRowWords` is gone from this path — and
that is the rest. A ceiling that bounds one mechanism does not bound a change that
happens to remove two.

**Stage and end-to-end effect, reference device and 400 real frames:**

| | before X-33 | after X-33 | **after X-34** |
|---|---|---|---|
| LK stage | 25.540 ms | 21.088 ms | **11.638 ms** |
| frontend | — | 22.01 ms/frame | **13.55 ms/frame** |
| vs OpenCV, 1 thread | — | 0.17× | **0.26×** |

**LK against LK, same points, same bits, OpenCV pinned to one thread:**

| | before | now |
|---|---|---|
| binCV `1/2/2/2` | 4.11× slower | **3.08× slower** |
| binCV `1/1/1/1` | 2.00× slower | **1.34× slower** |
| cost of the N=2 ladder | 2.06× | **2.30×** |

Accuracy is untouched — median track lifetime **18 vs OpenCV's 18**, flow median
**0.0348 px** — because every change in this sequence is bit-exact.

**Conclusion.**

1. **The straddle was real waste and removing it is the largest single win so far:
   2.13× on the kernel, 1.81× on the LK stage, 1.62× on the frontend.** The
   op-count model that predicted it — 1.29 popcounts per pixel at `N = 1` against a
   packed-word ideal of 0.625 — was right, and this is the first op-count inference
   in this project to survive measurement after S3's and the 9.4-cycles reading both
   failed.
2. **AT `1/1/1/1` binCV IS NOW 1.34× SLOWER THAN SIMD OpenCV WHILE USING 8× LESS
   MEMORY.** That is close to the honest statement of what bit-parallelism buys on
   this workload, and it is a very different claim than "14× slower", which is where
   this began.
3. **THE LADDER IS NOW THE DOMINANT SPEED FACTOR, AND IT WAS CHOSEN UNDER A
   MISTAKEN PROFILE.** [D-23](ARCHITECTURE.md) adopted `1/2/2/2` on accuracy, with
   its speed cost estimated at 1.35×. Isolated, it is **2.30×** — and it was chosen
   when corner detection was believed to be 52.7% of the frontend rather than 2%.
   **The decision is not reversed here**: it bought real accuracy (yield 88.7–99.3%
   against `1/1/1/1`'s 75.9–88.7%, [X-25](#x-25--the-coarse-level-window-border--done))
   and this entry measures speed, not accuracy. But it is now **the single largest
   speed lever left**, larger than [E-18](ARCHITECTURE.md#register), and it should
   be re-decided against the corrected profile rather than left standing on the old
   one. Registered as **[E-19](ARCHITECTURE.md#register)**.
4. **`RegionWords` gained `x0`/`x1`.** `regionFromExtent` is handed them and threw
   them away; recovering them from the masks afterwards would have cost a
   count-trailing-zeros to rediscover what was already known.

**Decision:** adopt. `residualSums` dispatches to the aligned path when the clipped
region fits one word — which at `seal_params.yaml`'s 31×31 is every window at every
word type binCV supports — and keeps the general path for wider windows. Recorded as
[D-31](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/straddle_ceiling.cpp` (the bound) and
`benchmark/residual_benchmark.cpp` with `residual_arm_aligned.cpp`;
`benchmark/lk_headtohead.cpp` for the OpenCV comparison.

---

### X-35 · The tap machinery around the arithmetic · `DONE`

**COMMITTED BEFORE THE ARMS EXIST.**

**Gates:** [X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial)'s
unmet criterion 4.
**Question:** After [X-34](#x-34--the-straddling-window--done), `residualSums`'
**arithmetic is already better than OpenCV's** — 0.65 popcounts per pixel at `N = 1`
against OpenCV's ~1.2 SIMD ops. binCV is nonetheless 1.34× slower at `1/1/1/1`.
**All of the remaining gap is machinery around the arithmetic. What is it, and how
much comes off?**

**The accounting, per window row at `N = 1` on the aligned path:**

| | ops |
|---|---|
| 4 × `displacedRow` construction | **60** — clamp, `row()`, `minRowWords`, `rowTailMask`, **`edgeFill` twice (two loads)** |
| 4 × `word(0)` | 24 |
| 5 × `alignedWord` (prev, mag, sign) | 20 |
| **10 × `slicedSignedSum`** | **60** ← the actual arithmetic |
| 10 × accumulate | 10 |
| **total** | **~174 for 31 pixels = 5.6 ops/pixel** |

**The machinery is eight times the arithmetic.** That is the whole remaining
answer to "why isn't a 1-bit tracker faster than a byte-per-pixel one".

**Arm T — the `+1` tap is a shift, and X-34 is what makes that TRUE.**
[X-32](#x-32--residualsums-tap-extraction--done) tried exactly this identity and it
**lost**, because in the per-word path `t01` at word `i` needs a bit from word
`i + 1` and the extra read cost more than it saved. **Alignment removes that.** The
window is 31 pixels and a `uint32_t` read covers 32, so `t01`'s bits for window
positions `0..30` are source columns `[c+1, c+31]` — **entirely inside the word
`t00` already read**. So `t01 = t00 >> 1` exactly, one operation, and **two of the
four `displacedRow` constructions disappear**. A rejected optimisation becoming
correct because an unrelated change moved the ground under it is worth recording as
such.

**Arm I — the interior fast path.** `displacedRow` builds the replicate border
unconditionally: two `edgeFill` calls, each a load and a test. **A window whose
displaced extent is entirely inside the frame needs none of it** and can use the
same cheap `alignedWord` the previous-frame planes use. Most windows are interior;
the border machinery is being paid for the minority.

**Arms are separated deliberately** — they are independent, and a combined number
would not say which worked. **Guarded on `width < bitsPerWord`** for arm T (at
`width == bits` the `+1` tap leaves the word); at `seal_params.yaml`'s 31 with
`uint32_t` that holds, and the general path remains for anything else.

**Decision rule** *(written before measuring)* — `R` = shipped / (T+I) on
`residualSums`, reference device, `N = 1` **and** `N = 2`, **bit-exact sums as a
precondition**.

* **Band A — `R` ≥ 1.4×.** Adopt both, report T and I separately, and re-measure
  the LK stage and the frontend.
* **Band B — 1.15× ≤ `R` < 1.4×.** Adopt whichever arm individually clears 1.1×;
  drop the other. Neither is worth its complexity below that.
* **Band C — `R` < 1.15×.** **Do not ship.** The op accounting above would then be
  wrong in the same way S3's was, and the entry must say where the time really goes
  before any further work on this kernel.
* **Band D — arm T is slower, as it was in X-32.** Then alignment did *not* make the
  identity free and X-32's result stands for a reason not yet understood — which
  matters more than the optimisation, because the same reasoning is what X-34 rests
  on.

**Reported regardless:** ops per pixel before and after, against OpenCV's ~1.2, so
the remaining gap is attributable rather than merely smaller.

**Variants:** shipped, +T, +I, +both × `N` = 1, 2.
**Workload:** 31×31 windows over 640×480.
**Metric:** ns per window on the reference device; bit-exact sums; then the LK stage
and the frontend.

**Result — BAND A, and the arms are separated as the rule required.**

Reference device, LK stage: **11.638 → 7.421 ms, 1.57×**; residual + solve
**10.516 → 6.267 ms, 1.68×**. Bit-exact — `test_opticalflow`'s per-pixel oracle
compares `residualSums` against a `long long` control at `N = 1..5` over random
windows with **negative and out-of-range taps**, which is exactly what exercises
the non-interior path, and on-device `ctest` is green.

**Arm T fired, and it is X-32's rejected identity working because X-34 moved the
ground under it.** In the per-word path `t01` at word `i` needed a bit from word
`i + 1` and the extra read cost more than it saved — 0.974×, rejected. Aligned,
the window is 31 pixels and one `uint32_t` read covers 32, so `t01`'s bits are
**already inside the word `t00` holds**: `t01 = t00 >> 1`, and two of four
displaced-row constructions vanish.

**Arm I fired too.** `displacedRow` built the replicate border unconditionally —
two `edgeFill` calls, each a load and a test — for windows that are mostly interior.

**Cumulative effect of the whole sequence on the LK stage (reference device):**

| | ms | |
|---|---|---|
| before X-33 | 25.540 | |
| after X-33 (NEON) | 21.088 | 1.21× |
| after X-34 (alignment) | 11.638 | 2.19× |
| **after X-35 (tap machinery)** | **7.421** | **3.44×** |

**LK AGAINST LK, SAME POINTS, SAME BITS, OpenCV PINNED TO ONE THREAD — median of
seven repeats on an otherwise idle machine:**

| arm | median ms | min | max |
|---|---|---|---|
| binCV `1/2/2/2` (shipped) | 9.819 | 9.377 | 11.878 |
| **binCV `1/1/1/1`** | **4.216** | 4.040 | 4.470 |
| **OpenCV, `CV_8U`, 1 thread** | **4.134** | 3.896 | 7.708 |

**binCV at `1/1/1/1` is 1.02× of single-threaded SIMD OpenCV — level — on 8× less
memory.** Against 2.00× at the start of this sequence and 14× where the session
began.

**AN EARLIER READING OF THIS SAME COMPARISON WAS CONTAMINATED AND IS WITHDRAWN.**
A run taken while `verify.sh` was building in the background reported 1.00×, and
OpenCV's own time had swung 4.425 → 3.803 → 5.480 ms across runs **on identical
code**. That is a 1.44× spread from machine load, larger than most of the effects
this project measures. The numbers above are medians of seven repeats at load
average ~1.2. **A single timing run on a busy development machine is not a
measurement**, and this is the second time in this project that a number measured
on the wrong conditions nearly became a result.

**Conclusion.**

1. **The machinery was the gap, and the accounting was right.** Per row at `N = 1`,
   arithmetic was 0.65 popcounts/pixel against machinery of ~5 ops/pixel. Removing
   two of four displaced-row constructions and the border work for interior windows
   took 1.57× off the stage. **This is the second op-count inference in a row to
   survive measurement**, after four in this project that did not.
2. **At `N = 1` the original expectation is now met: bit-parallel tracking is level
   with vectorized byte-per-pixel tracking, at an eighth of the memory.** The
   arithmetic was always ahead — 0.65 popcounts/pixel against ~1.2 SIMD ops — and
   what stood in the way was addressing, not the idea.
3. **ALL of the remaining gap at the shipped ladder is the ladder.** `1/2/2/2` costs
   **2.33×**, and that is now the entire difference between parity and 2.38× slower.
   [E-19](ARCHITECTURE.md#register) is no longer one lever among several; it is the
   only one left of this size.

**Decision:** adopt both arms. Recorded as
[D-32](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/lk_headtohead.cpp`, `benchmark/frontend_profile.cpp`.

---

### X-36 · Batching across TAPS, and what the footprint does not buy · `DONE`

**COMMITTED BEFORE THE ARM EXISTS.**

**Gates:** [X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial)'s
criterion 4, and a standing expectation this entry settles first.

---

**PART ONE — MEASURED BEFORE ANY OPTIMISATION, BECAUSE IT DETERMINES WHETHER THE
OPTIMISATION IS THE RIGHT KIND: IS LK MEMORY-BOUND?**

The project's footprint result is **6.23×** end to end, and the natural expectation
is that moving 8× less data makes tracking faster. **It does not, and this is
measured rather than reasoned.** Reference device, `N = 1`, one level, 31×31:

| points | µs per point | | frame | KB @ 1 bit | µs per point |
|---|---|---|---|---|---|
| 35 | 21.91 | | 320×240 | 9.4 | 22.93 |
| 140 | 21.38 | | 640×480 | 37.5 | 21.38 |
| 560 | 22.26 | | 1280×960 | 150.0 | 24.19 |
| 1160 | 22.96 | | 1920×1440 | 337.5 | 24.23 |

**33× more points and 36× more data move the per-point cost by under 13%.** A 31×31
window is **120 bytes at 1 bit** — two to four cache lines either way — and the
compute per window dwarfs the load. **LK is compute-bound, and the 8× footprint
advantage does not convert into tracking speed.**

That is worth stating plainly because it has been implicit in this project's framing
and is wrong: **the memory result and the speed result are independent.** The
footprint matters for what fits on a device; it does not make this kernel faster.
Any further speed has to come from doing less work, not from touching less data.

---

**PART TWO — THE ARM.**

**Question:** `slicedSignedSum` batches its popcounts across the **`N²` plane
pairs**. At `N = 1` there is exactly one pair, so **the NEON path does nothing and
level 0 — the largest level of every ladder — runs fully scalar even on aarch64.**
What does batching across the **five taps** instead buy, given that they exist at
every `N`?

**The structure being exploited.** The residual needs, per component, five sums:
`t00`, `t01`, `t10`, `t11` and `self`, each against the same magnitude and sign.
**Four of them fit one 128-bit vector.** So at `N = 1` a row's 20 popcounts become
~4 vector operations plus 4 scalar, instead of 20 scalar — and the grouping is by
TAP, which exists at every depth, rather than by plane pair, which does not exist
at `N = 1`.

**Two further facts frame what this can and cannot do.**
* **binCV has NO x86 vector path at all.** The parity result in
  [X-35](#x-35--the-tap-machinery-around-the-arithmetic--done) —
  binCV 4.216 ms against OpenCV's 4.134 ms — was **binCV SCALAR against OpenCV
  SSE**. Anything this arm gains on x86 comes from instruction-level parallelism
  alone.
* [ROADMAP Phase 5.3](ROADMAP.md#phase-5--platform-hardening) makes x86 a
  comparison platform, not a deployment target, so **the arm is judged on the
  reference device** and any x86 gain is reported as a secondary observation.

**Decision rule** *(written before measuring)* — `R` = shipped / batched on
`residualSums`, reference device, **`N = 1`** (the case that currently gets nothing)
and `N = 2`, bit-exact sums as a precondition.

* **Band A — `R` ≥ 1.4× at `N = 1`.** Adopt. Re-measure the LK stage, the frontend
  and the OpenCV comparison, and report the `N = 2` effect separately — the two
  depths use different batching and one may regress.
* **Band B — 1.15× ≤ `R` < 1.4×.** Adopt only if `N = 2` does not regress. The
  restructure replaces a small, well-tested function with a wider one, and below
  1.4× that has to be earned.
* **Band C — `R` < 1.15×.** **Do not ship.** The remaining cost is then not the
  popcounts at all, and the entry must locate it before any further work — the
  five-tap decomposition itself would become the thing to question, and that is
  [D-20](ARCHITECTURE.md)'s boundary rather than an implementation detail.
* **Band D — the batched form is SLOWER at `N = 2`.** Then batching across pairs and
  batching across taps are in conflict and only one can be had; report which and
  why, and keep the better per depth rather than forcing one shape on both.

**A ceiling is measured first**, per the procedure X-33 established and X-32 argued
for: a stripped loop that batches four popcounts against one mask, against four
scalar popcounts. **Under 1.4× and the arm is not written.**

**Variants:** shipped vs tap-batched × `N` = 1, 2.
**Workload:** 31×31 windows over 640×480.
**Metric:** ns per window on the reference device; bit-exact sums; then the LK stage
and the OpenCV comparison.

**Ceiling, measured first:** four popcounts against one mask, scalar against vector
— **2.414×**, bit-identical. Above the 1.4× gate, so the arm was written.

**Result — BAND A on the kernel, and a small effect on the stage. Both are
reported, because the difference between them is the finding.**

| | device µs | |
|---|---|---|
| `residualSums` at `N = 1`, scalar | 354.2 | 1.000× |
| **`residualSums` at `N = 1`, tap-batched** | **204.0** | **1.736×** |

Bit-exact: 0 of 130 windows differ, on-device `ctest` green.

**LK stage: 7.421 → 7.129 ms, only 1.04×** — because the shipped `1/2/2/2` ladder
has **one level at `N = 1` and three at `N = 2`**, and every level costs the same in
LK (same points, same window). A 1.736× on a quarter of the work is 1.13× at best,
and the measured 1.04× is inside that.

**Ladder sweep on the reference device with the new kernel:**

| ladder | track µs | vs `1/1/1/1` |
|---|---|---|
| **`1/1/1/1`** | **5 479.8** | 1.00× |
| `1/2/2/2` *(shipped)* | 9 639.6 | **1.76×** |
| `1/3/3/3` | 29 608.4 | 5.40× |

**Cumulative across the whole optimisation sequence, reference device, LK track:**

| ladder | before X-33 | now | |
|---|---|---|---|
| `1/1/1/1` | 20 485.6 µs | **5 479.8** | **3.74×** |
| `1/2/2/2` | 27 571.5 µs | **9 639.6** | **2.86×** |

**Conclusion.**

1. **PART ONE'S FINDING IS THE MORE IMPORTANT ONE: LK IS COMPUTE-BOUND AND THE 8×
   FOOTPRINT BUYS NO TRACKING SPEED.** 33× more points and 36× more data moved the
   per-point cost under 13%. This project has carried an implicit assumption that
   the two results reinforce each other. **They do not — they are independent**, and
   saying so is worth more than the optimisation below it.
2. **The arm works and is adopted, at 1.736× on the kernel.** `N = 1` is level 0 of
   every ladder and had been running **fully scalar even on aarch64**, because the
   existing NEON path batches `N²` plane pairs and at `N = 1` there is one.
   Batching across the four **taps** is the structure that exists at every depth,
   and D-31's alignment is what lets the lane accumulators run the whole window
   instead of extracting per row.
3. **THE LADDER NOW GATES THE OPTIMISATION, NOT JUST THE ARITHMETIC.** At `1/1/1/1`
   all four levels would take the 1.736×; at `1/2/2/2` one does. So
   [E-19](ARCHITECTURE.md#register) is no longer only about the `N²` cost — the
   ladder also decides how much of the vectorized path is reachable at all. The
   measured ladder cost is now **1.76×** on the device track.
4. **Two gaps stay open and are named rather than left implicit.** `N = 2` gets
   nothing from tap batching — the two batchings compete for the same registers, and
   band D anticipated that they might not compose. And **binCV still has no x86
   vector path at all**, so [X-35](#x-35--the-tap-machinery-around-the-arithmetic--done)'s
   parity was binCV-scalar against OpenCV-SSE.

**Decision:** adopt for `N == 1, uint32_t` on aarch64; `N == 2` keeps the plane-pair
batching, which is the better shape at that depth. Recorded as
[D-33](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/tapbatch_ceiling.cpp`, `benchmark/residual_n1.cpp`,
`benchmark/lk_memorybound.cpp`, `benchmark/pyramid_depth_benchmark.cpp`.

---

### X-37 · binCV against OpenCV on the DEPLOYMENT TARGET, SIMD against SIMD · `DONE`

**Gates:** [ROADMAP](ROADMAP.md#success-criteria) criterion 4, and it **reverses the
sign of every previous reading of it**.
**Question:** Every binCV-versus-OpenCV tracking comparison in this project has been
taken on **x86, where binCV has no vector path at all** — so "parity" in
[X-35](#x-35--the-tap-machinery-around-the-arithmetic--done) was binCV **scalar**
against OpenCV **SSE**. On the reference device binCV has NEON
([D-30](ARCHITECTURE.md), [D-33](ARCHITECTURE.md)) and OpenCV has NEON. What is the
answer there?

**Platform check first, because assuming it is what bites.** The Pi's OpenCV 4.10
reports `Baseline: NEON FP16` with `Dispatched: NEON_DOTPROD NEON_FP16 NEON_BF16`.
It is genuinely vectorized. Pinned to one thread.

**THE FIRST RUN WAS THROWN AWAY BY THE HARNESS AND THAT WAS CORRECT.** Building
OpenCV's benchmarks on four cores drove the device into its soft temperature limit —
`throttled` went `0x0` → `0x80000` **during** the run — and `run_on_pi.sh` refused
the numbers. Re-run after cooling to 53 °C, with `throttled` unchanged at `0x80000`
(sticky history, no active bit) before and after. Every figure below is from a run
the harness certified valid.

**ITERATION COUNT IS A CONFOUND AND IS CONTROLLED.** Both trackers stop early on
their own convergence rules, so at `maxIterations = 20` they may do different amounts
of work and the ratio would not be of the kernels. `epsilon = 0` forces both to run
exactly the stated count.

**Result — reference device, 140 points, 31×31, four levels, ms:**

| iterations | binCV `1/1/1/1` | binCV `1/2/2/2` | OpenCV `CV_8U` |
|---|---|---|---|
| 1 | **0.961** | 2.849 | 11.989 |
| 2 | **1.390** | 4.281 | 14.326 |
| 4 | **2.226** | 6.986 | 18.307 |
| 8 | **3.317** | 10.855 | 22.582 |
| 20 | **5.416** | 15.326 | 26.575 |

**Linear fit `T = setup + iterations × slope`:**

| arm | setup ms | ms/iteration |
|---|---|---|
| binCV `1/1/1/1` | **1.077** | **0.2264** |
| binCV `1/2/2/2` | 3.666 | 0.6276 |
| OpenCV `CV_8U` | **13.810** | **0.7065** |

**binCV is faster at every iteration count:**

| iterations | `1/1/1/1` | `1/2/2/2` |
|---|---|---|
| 1 | **11.1× faster** | 3.4× |
| 4 | **8.4×** | 2.7× |
| 20 | **5.0×** | 1.7× |
| 50 | **4.0×** | 1.4× |

**Conclusion — and it locates the advantage precisely.**

1. **ON THE DEPLOYMENT TARGET binCV IS 1.4×–11× FASTER THAN SINGLE-THREADED SIMD
   OpenCV**, depending on ladder and iteration count, **while using 6.23× less
   memory**. Every prior reading of criterion 4 — 14× slower, then 6.3×, then
   3.8×, then parity — was taken on x86 where binCV runs **scalar**. Those readings
   were correct about x86 and **wrong as statements about the product**, whose
   target is Cortex-A.
2. **OpenCV's SETUP IS 12.8× binCV's, AND THAT IS THE REAL ADVANTAGE.** OpenCV copies
   the warped patch into `IWinBuf`/`derivIWinBuf` — **961 pixels × 3 shorts per point
   per level** — before iterating. binCV **copies nothing**; it reads the frame in
   place through the region walk. That is a data-movement win, and
   [X-36](#x-36--batching-across-taps-and-what-the-footprint-does-not-buy--done)
   is what makes it legible: the kernel is compute-bound, so the footprint does not
   speed up the *arithmetic* — it removes an entire **stage**.
3. **The per-iteration advantage is real but smaller: 3.1×.** So the shape of the
   win is "no setup, and cheaper steady state", and the first term dominates at the
   iteration counts a frontend actually uses.
4. **The ladder costs 1.7×–3.4× here**, consistent with the 1.76× measured on the
   track alone, and [E-19](ARCHITECTURE.md#register) is unaffected as a question.

**WHAT THIS DOES NOT SHOW.** This is **LK against LK**, not the whole frontend.
[X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial)'s
end-to-end comparison ran on x86 and included detection, build and preprocessing.
**The frontend comparison has never been run on the device**, because the EuRoC
sequence is not there. Until it is, criterion 4 is answered **for the tracker on the
deployment target** and not for the frontend. Registered as
[E-20](ARCHITECTURE.md#register).

**Method:** `benchmark/lk_headtohead.cpp` with `BINCV_FORCE_ITERS` / `BINCV_ITERS`,
run through `scripts/run_on_pi.sh pi4` with `BINCV_PI_OPENCV=1`.

---

### X-38 · E-20 — the WHOLE FRONTEND against OpenCV, on the deployment target · `DONE`

**Gates:** [E-20](ARCHITECTURE.md#register) · [T4.3a](TASKS.md) · **ROADMAP
criterion 4**, which this closes.
**Question:** [X-37](#x-37--bincv-against-opencv-on-the-deployment-target-simd-against-simd--done)
measured **LK against LK** on the device and found binCV 1.4×–11× faster. Every
*end-to-end* reading, including [X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial)'s,
was taken on **x86 where binCV runs scalar**. What does the whole frontend —
detection, pyramid build, preprocessing and tracking — do on the reference device?

**Workload:** EuRoC V1_02_medium, the **full 1710-frame sequence**, 752×480, both
frontends on bit-identical input through the reference pipeline's two-stage
preprocessing, each detecting and tracking independently, OpenCV pinned to **one
thread**. Pi 4, governor `performance`, `throttled` unchanged at `0x80000` (sticky
history, no active bit) before and after, `taskset -c 3`, commit `82daca6`.

*(This entry first closed on **692** frames — all that had transferred before the
Windows drive holding the dataset dropped mid-copy, `/mnt/g` going to `d?????????`.
The drive is back; the full sequence has now run, and so has the 692-frame prefix
again, as a control. Both are reported below because they **disagree**.)*

**Result — full sequence.**

| criterion | binCV | OpenCV | |
|---|---|---|---|
| 2 · median track lifetime | **11 frames** | 12 | one frame short |
| 2 · per-frame survival | **96.4%** | 96.6% | 0.2 points short |
| 2 · tracks observed | 10 279 | 10 129 | within 2% |
| 2 · flow difference | **median 0.0434 px, p90 0.1614** | — | **95.4% within 1 px** |
| 3 · peak footprint | **436 704 B** | 2 719 832 B | **6.23× smaller** |
| **4 · speed** | **11.198 ms/frame** | 16.324 ms/frame | **1.46× FASTER** |

**THE 692-FRAME PREFIX WAS AN EASY ONE, AND CRITERION 2's "EQUAL" WAS ITS ARTIFACT.**
Re-running the prefix on the current commit is the control, and it reproduces the
original accuracy figures **exactly**:

| over | lifetime | survival | flow median | p90 | p99 | within 1 px |
|---|---|---|---|---|---|---|
| 692 prefix, first run (`0cde718`) | 13 vs 13 | 97.1 vs 97.1 | 0.0386 | 0.1177 | 14.478 | 97.4% |
| **692 prefix, control (`82daca6`)** | **13 vs 13** | **97.1 vs 97.1** | **0.0386** | **0.1177** | **14.478** | **97.4%** |
| **full 1710** | **11 vs 12** | **96.4 vs 96.6** | **0.0434** | **0.1614** | **22.494** | **95.4%** |

Nothing regressed between the two commits — **the extra 1018 frames are harder**, and
**both** frontends degrade on them (OpenCV's own lifetime drops 13 → 12, its survival
97.1% → 96.6%). binCV degrades slightly more. Backing the prefix out of the totals
puts the tail at roughly **95.9% survival for binCV against 96.3% for OpenCV** — a
0.34-point gap where the prefix had exactly none. That is an inference from
aggregates, not a measurement: `frontend_sequence` reports whole-run statistics, so
the split is arithmetic on the two runs rather than a third run over frames 693–1710.

**binCV per stage, at the real duty cycle** (82 re-detections in 1709 frames):

| stage | ms/frame | share |
|---|---|---|
| track (LK) | 7.815 | 69.8% |
| **build (`pyrDown` + derivatives)** | **2.811** | **25.1%** |
| detect | 0.571 | 5.1% |

**The speed figures barely moved, and the ratio's movement is OpenCV's.** binCV lands
at **11.169 / 11.195 / 11.198 ms** across the three runs — a **0.26%** spread, well
inside this device's noise floor. OpenCV moves **16.324–17.060 ms** (±2.3%) on
identical input. So 1.46× and 1.52× are the same measurement seen through OpenCV's
run-to-run variance, and the conservative one is quoted. The duty cycle also held:
4.8% here against 4.1% on the prefix, both far from the 100% X-30 assumed (D-28).

**Conclusion — ROADMAP CRITERION 4 IS MET, AND ALL FOUR NOW ARE.**

1. **1.46× faster and 6.23× smaller, simultaneously, on the deployment target.**
   Criterion 4 asked for "faster execution on the bit-parallel operations against
   the byte-per-pixel denominator" and it is now satisfied end to end, not just for
   the tracker.
2. **Every previous reading of this criterion was a fact about x86, not about the
   product.** 14× slower → 6.3× → 3.8× → parity → **1.46× faster**: the first four
   were measured where binCV has **no vector path at all** (ROADMAP 5.3 is
   unwritten). The measurements were correct; **the platform was wrong**, and it
   took X-37 to notice. `frontend_sequence` now prints which case it is in rather
   than a fixed disclaimer that had gone false.
3. **Criterion 2 holds end to end, but NOT at parity — and the parity claim was
   this entry's own, so its withdrawal is recorded here rather than quietly edited.**
   Over the full sequence binCV's median track lifetime is **11 against OpenCV's 12**
   and survival **96.4% against 96.6%**, with flow agreeing to 0.0434 px at the
   median. Criterion 2 asks for *agreement frame by frame* — one lifetime frame out
   of twelve, 0.2 survival points and a 0.043 px median meet that. **Equality was a
   property of the 692-frame prefix, which the control above reproduces exactly.**
   The 4.6% beyond 1 px is the same tail [E-17](ARCHITECTURE.md#register) is
   chartered on, now measured over 2.5× the data and correspondingly heavier
   (p99 14.5 → 22.5 px).
4. **THE PROFILE HAS MOVED AGAIN, AND THIS TIME IT MOVES THE NEXT TARGET.** Tracking
   is 69.8% and **build is 25.1%** — up from 4.5%, because LK got 3.44× faster
   ([D-32](ARCHITECTURE.md)) and `pyrDown` did not. **`pyrDown` is now a quarter of
   the frontend**, which is exactly where the downsampling-filter design space
   ([E-21](ARCHITECTURE.md#register)) lands. Detection is 5.1% and stays
   uninteresting.
5. **WHERE THE GAP OPENS IS A LEAD, NOT A MYSTERY.** It opens on the harder,
   faster-motion tail — and large motion is exactly what coarse pyramid levels carry,
   which is exactly what the downsampling filter builds.
   [X-39](#x-39--the-pyramid-design-space-downsampling-filter--bit-depth--done) found
   the filter arms spreading furthest apart at its **largest** shifts:
   `DIRECT_SUBSAMPLE` collapses to 59.4% yield at `shift (6, 4)` where the shipped box
   holds 94.1%, and `BOX_3x3` reaches 100%. So
   [D-36](ARCHITECTURE.md#8-design-decisions)'s `BOX_2x2`/`BOX_3x3` trade and this
   0.34-point tail plausibly meet. **Plausibly**: no measurement has yet run the
   filter arms *through the frontend over a sequence*, and this is a hypothesis being
   registered, not a result.

**Decision:** criterion 4 closed; E-20 closed. Recorded as
[D-35](ARCHITECTURE.md#8-design-decisions). **Criterion 2's parity claim is
withdrawn**; criterion 2 itself still holds.

**Method:** `benchmark/frontend_sequence.cpp` via `scripts/run_on_pi.sh pi4` with
`BINCV_PI_OPENCV=1` and `BINCV_OPENCV_THREADS=1`, run twice — once over the whole
directory and once with the `692` argument, which selects the same prefix the
original run saw.

---

### X-39 · The pyramid design space: downsampling filter × bit depth · `DONE`

**COMMITTED BEFORE ANY FILTER BEYOND `BOX_2x2` EXISTS.**

**Gates:** [E-21](ARCHITECTURE.md#register), and it re-opens
[E-19](ARCHITECTURE.md#register) as one axis of a larger question.

**Question:** binCV implements **one** of the reference's six
`LKPyrDownFilterType` variants. What does the full space of
**(downsampling filter × bit depth)** look like in accuracy, footprint and speed —
and which point matches standard-LK accuracy?

**WHY THIS IS NOT A TIDY-UP.**

1. **EVERY ACCURACY NUMBER IN THIS PROJECT WAS MEASURED ON A BOX-DOWNSAMPLED
   PYRAMID.** X-20's 0.25 px tolerance and its miss, X-24's ladder sweep, X-25's
   border arms, X-27's representation floor — all of them. **A 2×2 box is a poor
   lowpass and aliases**, and aliasing at coarse levels is a classic cause of
   pyramid-LK failure. So a cause this project chased through bit depth, borders and
   the representation itself **has never been excluded**, because the filter was
   never a variable.
2. **There has never been a Gaussian reference point**, so "how much worse than
   standard LK is this?" has never had an answer on binCV's own content.
3. **[X-38](#x-38--e-20--the-whole-frontend-against-opencv-on-the-deployment-target--done)
   makes it the hot stage**: `pyrDown` is now **25.8%** of the frontend, up from
   4.5%, because LK got 3.44× faster and the build did not.
4. **binCV's shipped `1/2/2/2` is not the paper's configuration either.** SEAL
   §4.2.2 stores **3 bits** for Box 2×2 (a box sum of four bits is 0..4); binCV caps
   at 2. The paper's own axis: Gaussian 5×5 → **8 bits/pixel**, Box 2×2 → **3 bits**
   at "comparable accuracy (~1 cm)", direct subsample → **1 bit** but **>2.5 cm
   worse** ATE.

**Arms — all six variants, as `LKPyrDownFilterType` defines them.** Five are
weighted sums of shifted taps and share one bit-sliced framework — the same full
adders `boxSum4` already uses (D-2); only the weights and the tap count change:

| filter | weights | sum range on 1-bit input |
|---|---|---|
| `DIRECT_SUBSAMPLE` | pick one of four | 0..1 |
| `BOX_2x2` *(shipped)* | 1,1,1,1 | 0..4 |
| `BOX_3x3` | nine 1s | 0..9 |
| `GAUSSIAN_3x3` | `[1,2,1] ⊗ [1,2,1]`, Σ=16 | 0..16 |
| `GAUSSIAN_5x5` | `[1,4,6,4,1] ⊗ [1,4,6,4,1]`, Σ=256 | 0..256 |

`MEDIAN_3x3` is **structurally different** — an order statistic, not a weighted sum.
At `NIn = 1` it collapses to a **majority vote** (popcount ≥ 5), which is cheap; at
`NIn > 1` it needs a bit-sliced sorting network and is priced separately.

**Output depth stays the existing `NOut` requantization**, so the space is genuinely
two-dimensional: any filter can be stored at any depth, and the paper's
"Gaussian needs 8 bits" is a *consequence* of its sum range rather than a
constraint.

**Decision rule** *(written before measuring)*

**The accuracy anchor is `GAUSSIAN_5x5` at its natural depth**, because that is what
`cv::buildOpticalFlowPyramid` applies and therefore what "standard LK accuracy"
means. Everything else is reported as a displacement from it.

* **Band A — a cheap point matches the anchor.** Some (filter, depth) within 1
  accuracy point of `GAUSSIAN_5x5` costs materially less in footprint or time.
  **Make it the default**, expose the rest, and record the anchor's numbers beside
  it so the trade is legible.
* **Band B — nothing matches, and the anchor is affordable.** Then the **default
  becomes `GAUSSIAN_5x5`** — the user's framing, and the paper's baseline — with the
  cheaper points offered as documented trade-downs. binCV would then ship
  standard-LK accuracy by default and low-bitwidth options by choice.
* **Band C — nothing matches and the anchor is NOT affordable** at `pyrDown`'s 25.8%
  share. Then report the frontier and **bring the choice to the caller**, as
  [D-24](ARCHITECTURE.md) did for route (a) and [D-32](ARCHITECTURE.md) for
  `maxIterations`. **Do not pick silently.**
* **Band D — the filter changes X-20's accuracy story.** Pre-declared as the most
  consequential outcome: if `GAUSSIAN_5x5` materially improves the tracking accuracy
  that X-24/X-25/X-27 chased elsewhere, then **aliasing from the box filter was a
  cause those entries could not see**, and their conclusions need re-reading — not
  retracting, since each was correct about its own variable, but re-weighting.

**Bit-exactness is NOT a precondition here** — unlike every optimisation entry in
this project, these filters compute **different functions on purpose**. What must
hold instead: each filter reproduces its *own* definition exactly, checked against a
per-pixel integer reference.

**Variants:** 6 filters × depths {1, 2, 3, 5, natural}.
**Workload:** the repo's real frame and the 692 EuRoC frames already on the device.
*(The full 1710-frame sequence needs the Windows drive back.)*
**Metric:** yield and flow agreement ([X-25](#x-25--the-coarse-level-window-border--done)'s),
peak bytes, and ms/frame on the reference device.

**PARTIAL — accuracy and footprint measured; the SPEED axis needs the bit-sliced
kernels, which is the point of measuring accuracy first.** If the filter had not
mattered, no kernel would need writing.

**Method note:** the filters are built with a **reference implementation**
(`cv::sepFilter2D` on the binarized frame, then subsample and quantize), not a
bit-sliced kernel. Level 0 is the binary frame in every arm; **only levels 1–3
vary**. `N` is capped at **7**, not 8: the derivative of an `N`-bit level is
`SignedQuantMat<N>`, which needs `N+1` planes.

**Result — mean YIELD across six warps on the repo's real frame:**

| filter | N=2 | N=3 | N=5 | N=7 | vs anchor | gain N=2→7 |
|---|---|---|---|---|---|---|
| **`GAUSSIAN_5x5` (anchor)** | 93.10% | 96.22% | **96.87%** | 97.03% | — | **+3.93** |
| `BOX_3x3` | 92.80% | 95.73% | 96.07% | 96.07% | −0.80 | +3.27 |
| `GAUSSIAN_3x3` | 94.43% | 95.73% | 95.58% | 95.58% | −1.28 | +1.15 |
| **`BOX_2x2` (shipped)** | **93.78%** | 94.77% | 94.60% | 94.60% | **−2.27** | **+0.82** |
| `MEDIAN_3x3` | 89.33% | 89.33% | 89.33% | 89.33% | −7.53 | +0.00 |
| `DIRECT_SUBSAMPLE` | 77.18% | 77.18% | 77.18% | 77.18% | **−19.68** | +0.00 |

**Conclusion.**

1. **THE TWO AXES ARE NOT INDEPENDENT, AND THAT IS THE FINDING.** Look at the last
   column. **`BOX_2x2` gains +0.82 points from N=2 to N=7; `GAUSSIAN_5x5` gains
   +3.93.** A 2×2 box sum of four values has **five possible outcomes** — past 3
   bits there is nothing left to store, and the table shows it saturating exactly
   there. A 5×5 Gaussian produces a genuinely graded value and keeps paying to 5
   bits. **The filter determines how much depth is USEFUL**, so E-19's ladder
   question was never separable from the filter it was asked about, and every
   bit-depth result in this project was measured at the one filter that benefits
   least from depth.
2. **BAND D FIRES, BUT MILDLY — AND THAT MATTERS FOR THE EARLIER ENTRIES.** The
   filter does affect accuracy: the shipped `BOX_2x2 @ N=2` sits **2.27 points**
   below the anchor, and up to 5 points on individual large-motion cases
   (`(6,4)`: 94.1% against 100.0%). **But it is not the hidden cause of T3.8's
   miss.** Box does not fail where Gaussian succeeds; it is a few points worse.
   **[X-24](#x-24--pyramid-level-bit-depths--done),
   [X-25](#x-25--the-coarse-level-window-border--done) and
   [X-27](#x-27--the-1-bit-level-0-localisation-floor--done) stand**, with the
   caveat that they were measured on the filter least sensitive to the axis they
   were varying.
3. **`DIRECT_SUBSAMPLE` is catastrophic and the paper is confirmed on binCV's own
   content.** 77.18% mean, and **63.7% / 59.4%** on the two largest motions against
   94–100% for every filtered arm. SEAL §4.2.2 reports ">2.5 cm worse" ATE for this
   arm; the mechanism is visible here — with no lowpass, coarse levels alias and
   large-motion capture collapses.
4. **`MEDIAN_3x3` is the surprise, and it is bad.** 7.53 points below the anchor and
   **completely flat in N** — worse than `BOX_2x2`. A median is an order statistic,
   so on a sparse edge map a 3×3 median of mostly-zero neighbourhoods returns zero:
   it **erodes the edges** rather than blurring them. It is in the reference's
   option set for *denoising*, where it is the right tool (SEAL uses it in the
   temporal processor, Table 3), and this measures it somewhere it does not belong.
5. **The cheapest point that nearly matches the anchor is the anchor at lower
   depth.** `GAUSSIAN_5x5 @ N=3` is **0.65 below** `@ N=5` — 3 bits instead of 5, so
   materially less footprint for almost no accuracy. `BOX_3x3 @ N=3` is 1.14 below
   the anchor and is a cheaper filter.

**Decision: NONE YET, and deliberately.** The rule's bands are about accuracy
*against cost*, and **the cost side does not exist until the filters have bit-sliced
kernels**. What is established is the accuracy frontier and that
`GAUSSIAN_5x5 @ N=3` and `BOX_3x3 @ N=3` are the points worth pricing. Writing the
kernels is now justified — which is what measuring accuracy first was for.

**Still owed:** the full 1710-frame sequence (the Windows drive holding it dropped
mid-session), and the flow-agreement metric alongside yield.

**Method:** `tests/test_opticalflow.cpp`,
`Flow.X39_PyramidFilterDesignSpace_uint32_t`.

---

**THE SPEED AXIS — measured, and it settles the entry.**

Five of the six filters are now real bit-sliced kernels
(`impl::pyrDownFilteredRoute`), verified **exact against a per-pixel integer
reference** at several `(NIn, NOut)` pairs — X-39's rule asked each filter to
reproduce *its own* definition, not OpenCV's border, and it does.
`MEDIAN_3x3` was not implemented: X-39 measured it **7.53 points below the box**
and flat in `N`, so a kernel for it would price something already excluded.

**Reference device, 640×480 → 320×240, `NIn = 1`:**

| arm | µs | vs shipped | mean yield vs anchor |
|---|---|---|---|
| `pyrDown` (shipped hand-written `BOX_2x2` route) N=3 | **93.7** | 1.00× | −2.27 |
| filtered `DIRECT_SUBSAMPLE` N=1 | 20.9 | 0.22× | −19.68 |
| filtered `BOX_2x2` N=3 | 277.8 | 2.96× | −2.27 |
| filtered `BOX_3x3` N=3 | 398.0 | 4.25× | **−0.80** |
| filtered `GAUSSIAN_3x3` N=3 | 497.7 | 5.31× | −1.28 |
| filtered **`GAUSSIAN_5x5`** N=3 (anchor) | **2 352.9** | **25.10×** | 0.00 |

**Estimated frontend effect**, scaling the single-level cost by the level geometry
(3 transitions × 2 frames, each level ¼ the pixels) against X-38's 11.169 ms:

| filter | frontend ms | vs OpenCV |
|---|---|---|
| `BOX_2x2` shipped | **11.169** | **1.48× faster** |
| `BOX_3x3` | 11.968 | 1.38× faster |
| `GAUSSIAN_3x3` | 12.230 | 1.35× faster |
| **`GAUSSIAN_5x5`** | **17.099** | **0.97× — SLOWER** |

*(Baselines are X-38's 692-frame figures, 11.169 against 16.509 ms, which were the
numbers on record when this ran. X-38 has since re-measured over the full 1710-frame
sequence at 11.198 against 16.324 — the "vs OpenCV" column shifts by about 0.02× and
`GAUSSIAN_5x5` stays on the slow side of parity, so nothing here changes.)*

**Conclusion — BAND C: the anchor is not affordable, and the frontier is clear.**

> **THE BAND C VERDICT WAS LATER OVERTURNED — see
> [X-42](#x-42--e-22--is-the-filter-frameworks-cost-genericity-or-structure--done).**
> Every speed number below was measured on a framework carrying a **3× genericity
> tax**, which this entry names as a caveat and registers as E-22. Removing that tax
> made `GAUSSIAN_5x5` **4.28× faster**, so the anchor costs **+1.20 ms** and leaves
> binCV **1.32× faster than OpenCV** rather than 0.97× slower. **The caveat was larger
> than the effect being decided.** The accuracy axis is unaffected; the speed axis and
> the Band C verdict are superseded.


1. **`GAUSSIAN_5x5` WOULD COST THE ENTIRE CRITERION-4 RESULT.** At 25.10× it adds
   ~5.9 ms to an 11.169 ms frontend and puts binCV **behind** OpenCV again. Standard-LK
   accuracy is reachable and **it costs more than it is worth here** — which is the
   answer the paper reached too, by a different route and for different reasons
   (SRAM, §4.2.2).
2. **`BOX_3x3` IS THE INTERESTING POINT AND IT WAS NOT ON ANYONE'S LIST.** It
   recovers **1.47 of the 2.27 yield points** the shipped filter gives up — 65% of
   the gap to standard LK — for **+0.8 ms**, leaving binCV at **1.38× faster** than
   OpenCV. It is also *cheaper and more accurate than `GAUSSIAN_3x3`*, which is
   therefore **dominated** and can be dropped from consideration.
3. **BUT 3× OF EVERY FILTERED NUMBER IS FRAMEWORK, NOT FILTER.** The generic route
   runs `BOX_2x2` at **2.96×** the hand-written one **computing the same
   function**. So the honest reading of `BOX_3x3`'s 4.25× is roughly **1.4× of
   filter and 3× of genericity**, and a hand-written `BOX_3x3` would likely land
   near 130 µs — about +0.15 ms on the frontend rather than +0.8. **The frontier
   above is measured, and it is measured on a framework that has not been optimised
   at all.**
4. **`DIRECT_SUBSAMPLE` is 0.22× and −19.68 points.** It is on the frontier only in
   the sense that nothing is cheaper; the paper's ">2.5 cm worse" is the same
   verdict.

**Decision: the shipped `BOX_2x2` stays the default, and `pyrDownFiltered` ships as
the option set.** Nothing else is within reach of the anchor at a cost that
preserves criterion 4, and **the accuracy/speed trade between `BOX_2x2` and
`BOX_3x3` — 1.47 yield points for ~0.8 ms — is the caller's**, exactly as
[D-24](ARCHITECTURE.md) put route (a) and [D-32](ARCHITECTURE.md) put
`maxIterations` there. Recorded as [D-36](ARCHITECTURE.md#8-design-decisions).

**Owed, and named rather than implied:** the frontend figures are **estimates** from
a single-level measurement scaled by geometry, not a measured frontend run; and the
framework's 3× genericity is unexamined ([E-22](ARCHITECTURE.md#register)). *(The
third item — the full-sequence accuracy sweep — is no longer owed; see immediately
below.)*

---

### X-39, sequence arm · the same design space over 1710 frames · `DONE`

**Question:** every accuracy number above came from **one image**, six warps and
about 102 eligible keypoints per cell.
[D-36](ARCHITECTURE.md#8-design-decisions) chose between `BOX_2x2` and `BOX_3x3` on a
**1.47-point** difference measured at that sample size, which is not obviously larger
than the frame-to-frame spread. **Does the ranking survive a sequence?**

**Rule, written before the merge.** The ordering is what D-36 rests on, so:
**(A)** ordering preserved and gaps of the same order → D-36 stands as written;
**(B)** ordering preserved but gaps materially different → D-36's *decision* stands
and its *numbers* are restated from the sequence; **(C)** ordering changes → D-36 is
re-opened.

**Workload:** EuRoC V1_02_medium, **all 1710 frames**, the same six warps and the
same six filters as above, run as **10 stride-10 shards** merged by summation —
yields are ratios and do not average. **1 180 133 eligible keypoint-cases per cell**,
against 611 in the single-frame table: a **1 900× larger sample**. Accuracy is
deterministic arithmetic, so this runs on the development machine; only speed needs
the reference device.

**Result — yield, 1710 frames.**

| filter | N=2 | N=3 | N=5 | N=7 |
|---|---|---|---|---|
| `GAUSSIAN_5x5` (anchor) | 95.22% | **95.83%** | 95.95% | 95.95% |
| `GAUSSIAN_3x3` | 95.39% | 95.46% | 95.61% | 95.58% |
| **`BOX_3x3`** | 95.27% | **95.73%** | 95.69% | 95.69% |
| `MEDIAN_3x3` | 90.76% | 90.76% | 90.76% | 90.76% |
| **`BOX_2x2`** (shipped) | 94.49% | **94.58%** | 94.55% | 94.51% |
| `DIRECT_SUBSAMPLE` | 83.18% | 83.18% | 83.18% | 83.18% |

**Yield relative to the anchor — one frame against 1710, at N=3:**

| filter | single frame | **1710 frames** | |
|---|---|---|---|
| `GAUSSIAN_3x3` | −1.28 | **−0.37** | gap shrinks 3.5× |
| **`BOX_3x3`** | −0.80 | **−0.10** | gap shrinks 8× |
| `MEDIAN_3x3` | −7.53 | **−5.07** | |
| **`BOX_2x2`** (shipped) | −2.27 | **−1.26** | gap shrinks 1.8× |
| `DIRECT_SUBSAMPLE` | −19.68 | **−12.65** | |

**Conclusion — BAND B: the ordering is exactly preserved, every gap is smaller, and
one of them collapses.**

1. **THE SINGLE FRAME OVERSTATED EVERY DIFFERENCE, BY 1.8× TO 8×.** Not one arm
   changed rank — the ordering `GAUSSIAN_5x5 > BOX_3x3 > GAUSSIAN_3x3 > BOX_2x2 >
   MEDIAN_3x3 > DIRECT_SUBSAMPLE` is reproduced at N=3, N=5 and N=7 — but the whole
   table compresses toward the anchor. **This is the ordinary failure mode of a
   design-space table read off one image**, and it is worth recording that the
   *decision* survived it while the *numbers* did not.
2. **`BOX_3x3` IS THE GAUSSIAN ANCHOR, FOR PRACTICAL PURPOSES.** −0.10 points at
   1.18 M samples. X-39 read it as closing 65% of the gap to standard LK; over the
   sequence it closes **92%**, at **4.25×** the filter cost rather than **25.10×**.
   The strongest claim this project can now make about its pyramid is that
   **standard-LK accuracy is available in a bit-sliced kernel at a sixth of the
   anchor's cost** — and that is a *stronger* result than the entry above reported.
3. **`BOX_2x2`'s DEFICIT IS HALF WHAT WAS THOUGHT: 1.26 points, not 2.27.** D-36
   priced the `BOX_2x2` → `BOX_3x3` upgrade at 1.47 yield points for ~0.8 ms; the
   sequence prices it at **1.16** for the same 0.8 ms. **The decision does not move —
   it moves slightly in the default's favour** — but the number a caller weighs does,
   so D-36 is restated rather than left standing on the single frame.
4. **`GAUSSIAN_3x3` IS DOMINATED MORE CLEANLY THAN BEFORE.** It is below `BOX_3x3` at
   every depth from N=3 up *and* costs 1.25× more (497.7 vs 398.0 µs). One image
   suggested it; 1710 confirm it. It stays in the option set because it is what a
   caller asking for "a Gaussian" will reach for, and being told it is dominated is
   more useful than not offering it.
5. **BIT DEPTH IS SETTLED AT N=3 AND FLAT FOR THE SHIPPED FILTER.** The anchor gains
   0.61 points from N=2→3, **0.12** from 3→5 and **0.00** from 5→7. `BOX_2x2` is flat
   across the whole axis (94.49 / 94.58 / 94.55 / 94.51, a 0.09-point band) —
   **binCV's shipped 2-bit levels lose nothing**, which the single frame could not
   have established. SEAL §4.2.2's 3 bits for Box 2×2 and this measurement agree.
   `MEDIAN_3x3` and `DIRECT_SUBSAMPLE` are exactly flat in N, as they must be: a
   median or a sample of 1-bit values is a 1-bit value, so the extra planes carry
   nothing. **That exactness is a self-check on the harness**, not a finding.
6. **NO SINGLE FRAME COULD HAVE DECIDED THIS, AND THE SPREAD SAYS SO.** Per-frame
   yield at N=3 runs p10 88.7–89.5 / median 94.1–94.6 / p90 98.3–98.6 for `BOX_2x2`
   against 91.4–92.3 / 95.2–95.7 / 98.7–98.8 for `BOX_3x3` (ranges across the 10
   shards). **The percentile bands separate cleanly across shards and overlap almost
   entirely within one** — the frame-to-frame spread is ~7 points wide, six times the
   1.16-point gap being measured. The ranking is a claim about the mean, and it takes
   a sequence to make it.
7. **The ten shards are a free replication.** Each is an independent stride-10 sample
   of 171 frames and they reproduce every percentile of every arm to within 0.7
   points. Nothing here rests on one draw.

**Decision: BAND B — [D-36](ARCHITECTURE.md#8-design-decisions) stands, restated on
the sequence.** `BOX_2x2` remains the default; the option set ships; the trade a
caller weighs is **1.16 yield points for ~0.8 ms**, and `BOX_3x3` is now reported as
reaching the Gaussian anchor rather than 65% of the way to it.

**Method:** `tests/test_opticalflow.cpp`,
`Flow.X39_PyramidFilterDesignSpaceSequence_uint32_t`, gated on `BINCV_X39_FRAMES` so
`verify.sh` stays hermetic and independent of a dataset outside the repository;
`BINCV_X39_SHARD=i/10` for the shards.

---

### X-40 · E-18 — window-carried vector accumulators at N = 2 · `DONE`

**COMMITTED BEFORE THE ARM IS WRITTEN.** Ceiling before arm, as
[X-33](#x-33--the-ceiling-for-batched-neon-popcounts--done) and
[X-36](#x-36--batching-across-taps-and-what-the-footprint-does-not-buy--done)
established: measure the bound, then decide whether the code gets written.

**Gates:** [E-18](ARCHITECTURE.md#register).

**Question.** [D-33](ARCHITECTURE.md#8-design-decisions) gave **N = 1** lane
accumulators that carry across the whole window, crossing the register domain
**once per window** instead of once per row. **N = 2 never got it.** Three of the
four levels of the shipped `1/2/2/2` ladder run at N = 2, so the depth that does most
of the tracking is still on the older shape: `slicedSignedSum` batches the `N²` plane
pairs into lanes and then does `vaddvq_s32` **per call** — ten calls per row, 31 rows,
**310 domain crossings per window** where D-33's shape needs about ten.

**The two shapes, precisely.** Both compute the same ten integers.

| | lanes hold | accumulators | reduces per window |
|---|---|---|---|
| **A — shipped** | the 4 plane pairs `(i, j)` | none; scalar after each call | **310** |
| **B — proposed** | the 4 taps `t00 t01 t10 t11` | **one `int32x4_t` per component** | **~10** |

B loops the four plane pairs *inside* the row and folds each into the same
accumulator with `vmlaq_n_s32(acc, diff_ij, 2^(i+j))`. That is exact, not
approximate: the weight is constant across rows, so
`Σ_rows Σ_pairs w·d = Σ_pairs w·Σ_rows d`. Two accumulator registers total, so
**no spill risk** — the naive "16 accumulators, one per (tap, pair)" layout is
rejected before measuring for that reason.

**Range check, so the arm cannot be wrong silently.** Per lane and pair,
`diff ∈ [−32, 32]` at `uint32_t`; weighted ≤ 4 and summed over 4 pairs and 31 rows
gives **|acc| ≤ 15 872**, comfortably inside `int32`.

**DECISION RULE, WRITTEN BEFORE MEASURING.** The benchmark runs both shapes on the
same data and **asserts all ten sums equal** — a ceiling that is also a
correctness check, which X-36's was not.

- **Band A — B/A ≥ 2.0×:** write the arm in `ops/opticalFlow.hpp` for N = 2.
- **Band B — 1.4× ≤ B/A < 2.0×:** write it, and report the frontend effect as
  modest rather than quoting the kernel ratio.
- **Band C — B/A < 1.4×:** **do not write it.** Record why and close E-18 negative.
  This is X-36's own threshold, reused deliberately.
- **Band D — B is SLOWER:** the domain-crossing account in E-18 is wrong, which is a
  finding about the cost model and not just about this kernel. Report it as such.

**A correction this entry carries.** E-18 was registered saying LK is *"94.7% of the
real frontend"*. [X-38](#x-38--e-20--the-whole-frontend-against-opencv-on-the-deployment-target--done)
measured the real duty cycle and LK is **69.8%** (7.815 of 11.198 ms). So a 2× on this
kernel is worth about **1.53×** on the frontend, not 1.9×, and a 3× is worth **1.87×**.
Still the largest item left in Phase 5.1 — but priced honestly before the work, not
after.

**Method:** `benchmark/windowaccum_ceiling.cpp`, reference device, via
`scripts/run_on_pi.sh pi4`.

**RESULT — BAND B, the arm was written, and the interesting number is not the one
the rule was about.**

**Ceiling, the two shapes alone** (20 windows, `benchmark/windowaccum_ceiling.cpp`):

| arm | ns / 20 windows | vs A |
|---|---|---|
| A — shipped, reduce per call | 57 655.0 | 1.000× |
| **B — proposed, reduce per window** | **39 458.4** | **1.461×** |

All ten sums identical over 20 windows. **1.461× → Band B: write the arm, price the
frontend effect as modest.** It was written as `impl::alignedResidualSumsNeon2`.

**Delivered, in the real kernel** (130 windows of 31×31 on an N = 2 level,
`benchmark/residual_n2.cpp`):

| arm | µs | vs scalar | vs shipped |
|---|---|---|---|
| scalar (`UseNeon=false`) | 842.3 | 1.000× | 0.721× |
| shipped NEON, reduce per call | 607.6 | 1.386× | 1.000× |
| **X-40, reduce per window** | **568.5** | **1.482×** | **1.069×** |
| **extraction only, no counting** | **275.6** | **3.057×** | **2.205×** |

**1. THE CEILING OVERPROMISED AGAIN, AND BY ABOUT THE SAME FACTOR AS LAST TIME.**
1.461× measured on the shapes, **1.069×** delivered. [X-33](#x-33--the-ceiling-for-batched-neon-popcounts--done)
got 1.24× against a 3.42× ceiling. **Two ceilings in a row have overstated the
delivered result**, and this one did so even after being built to be as close to the
real shape as possible — same window layout, same per-row data, same loads in both
arms. Isolating the counting does not rescue it either: the counting portion is
607.6 − 275.6 = 332.0 µs shipped against 568.5 − 275.6 = 292.9 µs here, a **1.133×**
on the part the reshaping actually touches. **In situ the accumulators compete for
registers with the tap machinery**, and no ceiling that omits the tap machinery can
see that. The lesson is not "stop measuring ceilings" — it is that a ceiling bounds
the *shape*, not the *kernel*, and this project should stop reading it as the latter.

**2. THE BINDING CONSTRAINT IS NO LONGER THE COUNTING. IT IS THE EXTRACTION, AND
THAT IS NOW MEASURED RATHER THAN INFERRED.** The floor arm runs the entire per-row
tap machinery — `alignedWord`, the interior test, the `t01 = t00 >> 1` identity, the
masks — with the counting removed and the words XORed into a sink. It costs
**275.6 µs, 45.4% of the shipped kernel**. So:

- **If counting became FREE, the kernel would gain 2.205× and no more.**
- E-18 was chartered on a remaining **"2–3×"**. That is not available from reshaping
  counts: **2.205× is the whole budget**, and this reshaping collected 1.069× of it.
- [D-29](ARCHITECTURE.md#8-design-decisions) measured tap extraction at **13.7%**.
  It is now **45.4%** — not because extraction got slower, but because
  [D-30](ARCHITECTURE.md), [D-31](ARCHITECTURE.md), [D-33](ARCHITECTURE.md) and
  [X-35](#x-35--the-tap-machinery-two-identities-the-alignment-made-true--done) made
  the counting roughly three times faster and left it alone. **The same thing that
  happened to `pyrDown` in [X-38](#x-38--e-20--the-whole-frontend-against-opencv-on-the-deployment-target--done)
  has now happened inside `residualSums`.**

**3. THE FRONTEND EFFECT IS SMALL, AND IS QUOTED AS SUCH.** At 1.069× on N = 2 —
levels 1–3 of the shipped ladder, roughly 6/7 of LK time by
[X-34](#x-34--the-ladder-in-isolation--done)'s 2.30× ladder ratio — LK improves about
**1.06×**, 7.815 → 7.374 ms, and the frontend 11.198 → **10.757 ms**, or **1.52×
against OpenCV** from 1.46×. **Under four percent.** Worth keeping because it is
exact, tested and free at runtime; not worth quoting as 1.5×.

**4. A COST THIS ENTRY DECLARES RATHER THAN HIDES.** `alignedResidualSumsNeon2`
duplicates the per-row tap-extraction block from `alignedResidualSums`, as
`alignedResidualSumsNeon1` already did — so **the extraction code now exists in three
copies, and extraction is exactly what the next optimisation must change.** That is a
real tax on the work item this entry just identified, and it is named here so the
next entry starts by paying it down rather than discovering it.

**Decision:** the arm ships. **E-18 is resolved, and resolved NEGATIVELY on its own
terms**: the 2–3× it was chartered on does not exist in the counting. Recorded as
[D-37](ARCHITECTURE.md#8-design-decisions). The extraction question it uncovers is
registered as [E-23](ARCHITECTURE.md#register).

**Bit-exactness** is now enforced by the gate rather than asserted in a comment.
`slicedSignedSum`, `alignedResidualSumsNeon1` and `alignedResidualSumsNeon2` all said
"the scalar path is the equality oracle and `tests/test_opticalflow.cpp` compares
them" — **and it did not.** The comparison existed only in `benchmark/residual_n1.cpp`,
which nothing in `verify.sh` runs. `Flow.ResidualNeonMatchesScalar_{N1,N2,N3}` now
sweeps 728 windows per depth across the borders and the interior with taps of both
signs; on x86 it compares the scalar path with itself and says so, and
`scripts/verify_arm.sh` gives it teeth. **0 of 728 differ at every depth on aarch64.**

**Method:** `benchmark/windowaccum_ceiling.cpp` and `benchmark/residual_n2.cpp` via
`scripts/run_on_pi.sh pi4`; `tests/test_opticalflow.cpp` for the equality oracle.

---

### X-41 · E-23 — what the extraction is actually spending · `DONE`

**COMMITTED BEFORE THE ARM IS WRITTEN.** Ceiling before arm — and this time with
[D-37](ARCHITECTURE.md#8-design-decisions)'s correction applied: **the ceiling bounds
the shape, not the kernel**, so the bands below are set on the *extraction floor arm*,
which is the thing being changed, and the delivered kernel number is expected to be
lower.

**Gates:** [E-23](ARCHITECTURE.md#register).

**Question.** [X-40](#x-40--e-18--window-carried-vector-accumulators-at-n--2--done)
measured the per-row tap machinery at **275.6 µs, 45.4% of `residualSums`**, with
zero counting in it. What is that 275.6 µs *made of*, and how much of it survives an
attempt to remove it?

**THE HYPOTHESIS, WRITTEN BEFORE MEASURING.** Almost everything `alignedWord` decides
is **invariant across the whole window**, and the kernel decides it again every row:

| quantity | recomputed | distinct values per window |
|---|---|---|
| `w0 = x / bits`, `s = x % bits`, `bits - s` | ~12× per row × 31 rows = **372×** | **2** — one for `r.x0`, one for `srcX` |
| `s == 0` branch | per call | **2**, both loop-invariant |
| `w0 + 1 < words` bounds test | per call | **2**, both loop-invariant |
| `interior` (`colsInside && rowsInside`) | per row | `colsInside` invariant; `rowsInside` false only on the **first and last few rows** |
| `.row(y)` → `data + y * stride` | 12 multiplies per row | consecutive rows differ by **one stride add** |

So the arm is: hoist the two `(w0, s)` pairs and their branches out of the y-loop,
carry **strided row pointers** instead of recomputing `.row(y)`, and **split the
y-loop** into border/interior/border so the bulk pays no `interior` test at all.

**Why the compiler may not already do this.** The row pointers come from *different
objects* (`lv.next[k]`, `lv.prev[k]`, `lv.dxMag[k]`, `lv.dyMag[k]`, `lv.dxSign`), so
hoisting the address arithmetic requires proving they do not alias — which nothing in
the signature says. That is the part most likely to pay, and the part a ceiling can
confirm cheaply before any of it is written.

**DECISION RULE, WRITTEN BEFORE MEASURING.** Bands are on the **extraction arm**
(hoisted vs the X-40 floor arm), both computing the same sink so equality is checked
first:

- **Band A — ≥ 1.6×:** write it into the kernel. At 45.4% of `residualSums` this is
  worth ≥ 1.2× on the kernel even after D-37's delivery discount.
- **Band B — 1.3× ≤ … < 1.6×:** write it **only as part of collapsing the three
  copies**, never as a fourth. The refactor has to pay for itself.
- **Band C — < 1.3×:** do not write it. **The extraction is loads, not addressing**,
  and E-23's successor is a memory-layout question rather than an arithmetic one —
  which would be a more valuable answer than a small speedup.
- **Band D — hoisting is SLOWER:** the compiler was already doing it and the
  three-copy duplication bought nothing. Say so, and drop the refactor's speed
  justification while keeping its maintenance one.

**A LIMIT DECLARED IN ADVANCE.** Even if extraction became **free**, X-40 measured the
cap at **2.205×** on `residualSums`, which is about **1.55×** on LK and **1.29×** on
the frontend — reaching **1.9× against OpenCV**. That is the *entire* remaining budget
for this kernel, and no result from this experiment can exceed it. Stating it now
means the number cannot be quietly inflated afterwards.

**Method:** `benchmark/residual_n2.cpp`, a new hoisted arm beside the existing floor
arm; reference device via `scripts/run_on_pi.sh pi4`.

**RESULT — BAND C, TWICE. THE HYPOTHESIS WAS WRONG, AND SO WAS THE BAND'S OWN
PREDICTION ABOUT WHY.**

| arm | µs | vs shipped |
|---|---|---|
| scalar (`UseNeon=false`) | 832.6 | 0.719× |
| shipped NEON | 598.6 | 1.000× |
| X-40, reduce per window | 559.9 | 1.069× |
| extraction only | 264.5 | 2.263× |
| **X-41 extraction, hoisted + strided** | **258.9** | **2.312×** |

**Hoisting bought 1.023×.** Every loop-invariant the hypothesis named — the two
`(w0, s)` descriptors, their `s == 0` case, their bounds test, the `.row(y)`
multiplies, the split y-loop that removes the `interior` branch from the bulk — is
worth **two percent**. Either the compiler was already doing it, or it was never the
cost. **Band C: the arm is not written.**

**SO THE BAND C BRANCH SAID "IT IS LOADS, NOT ADDRESSING", AND THAT IS ALSO WRONG.**
The rule pre-committed a successor — a memory-layout question — on the strength of a
prediction, so the prediction was tested rather than adopted. A 31×31 window touches
31 rows of **ten separate planes** (`prev`×2, `next`×2, `dxMag`×2, `dyMag`×2, two sign
planes); consecutive rows of one plane are a stride apart, so that is **310 distinct
cache lines, ~19.8 KB fetched for ~2.5 KB of useful bits** — an 8× overfetch against a
32 KB L1D. If the layout were the cost, shrinking the level until all ten planes fit
in L1 together should collapse it:

| same extraction, same window count | µs | vs large |
|---|---|---|
| 640×480 level (384 KB working set) | 264.8 | 1.000× |
| **128×96 level (~15 KB, inside L1D)** | **234.4** | **1.129×** |

**Fitting the entire working set in L1 buys 13%.** The 8× overfetch is real and it is
**not** what the kernel is waiting on.

**WHAT IT IS.** 264.5 µs over 130 windows is 2.035 µs per window; at the
`performance` governor's 1.8 GHz that is **~3 660 cycles per window, ~118 cycles per
row** for roughly a hundred instructions of shifts, ors, masks, branches and border
machinery. Neither the address arithmetic (2%) nor the memory system (13%) is the
constraint — **the instruction stream is**. The only remaining lever on this kernel is
to *issue fewer instructions*, and there is an obvious one: the twelve `alignedWord`
extractions in a row share exactly **two** `(w0, s)` descriptors, so they are twelve
scalar load-shift-or sequences that could be about **three vector ones**. That is
[E-24](ARCHITECTURE.md#register), and it is a different kind of change from anything
X-41 tried.

**Decision:** **Band C — nothing is written.** The hoisted arm stays in the benchmark
as the evidence, not in the kernel. **E-23 is resolved NEGATIVELY on its stated
hypothesis**, and the layout successor its own rule pre-committed is **withdrawn
before it was started** — which is the whole point of writing the rule down and then
testing its prediction instead of acting on it. Recorded as
[D-38](ARCHITECTURE.md#8-design-decisions); successor registered as
[E-24](ARCHITECTURE.md#register).

**The declared limit still stands and still binds.** X-40 capped this kernel at
**2.205×** if counting were free; X-41 now adds that extraction is not cheaply
removable either. Whatever E-24 delivers, the frontend cannot pass **~1.9× against
OpenCV** from `residualSums` alone.

**Method:** `benchmark/residual_n2.cpp` via `scripts/run_on_pi.sh pi4`. Both new arms
are checked for equality against the shipped extraction before anything is timed —
**0 of 130 windows differ.**

---

### X-42 · E-22 — is the filter framework's cost genericity, or structure? · `DONE`

**COMMITTED BEFORE THE CHANGE IS WRITTEN**, and with
[D-38](ARCHITECTURE.md#8-design-decisions)'s correction applied twice over: a rule
must state not just its bands but **what each band implies**, because
[X-41](#x-41--e-23--what-the-extraction-is-actually-spending--done)'s Band C
pre-committed a successor on a prediction that turned out false.

**Gates:** [E-22](ARCHITECTURE.md#register), and it re-prices
[D-36](ARCHITECTURE.md#8-design-decisions).

**Question.** [X-39](#x-39--the-pyramid-design-space-downsampling-filter--bit-depth--done)
measured `impl::pyrDownFilteredRoute` running `BOX_2x2` at **2.96× the hand-written
`pyrDown` computing the same function**. That factor is the tax on every filter in the
set, so `BOX_3x3`'s 4.25× is roughly 1.4× of filter and 3× of framework. **Is that tax
genericity, or is it structure?**

**IT SHOULD NOT BE GENERICITY, AND THAT IS WHY THIS IS WORTH MEASURING.** `F` is
already a **template parameter**; `filterTaps(F)` is `constexpr`, and `T`, `kTaps`,
`hPlanes`, `vPlanes` are all compile-time constants in the route. **The helpers then
throw that away at their own signatures:**

| helper | discards | consequence |
|---|---|---|
| `weightedAxis(taps, tapCount, weights, out, outN)` | `tapCount`, `weights`, `outN` as runtime arguments | the weight-bit decomposition loops at runtime over compile-time values |
| `requantizeWeighted(sum, kSum, out)` | `kSum` as a runtime `unsigned` | restoring division cannot fold `divisor << q` into literals, and `thresholdGE` cannot specialise |
| `addShifted(acc, accN, v, vN, shift, tmp)` | `accN`, `vN`, `shift` | every add stages its operand through a 9-word `tmp` and ripples with a per-plane bounds test — and at `shift == 0` the "shifted operand" **is** the operand, so the copy is pure overhead |

**The change is therefore a signature change, not an algorithm change**: make those
parameters template parameters, and drop `tmp` by indexing directly. **It costs zero
flexibility** — `F` was always compile-time — which is what distinguishes this from a
genericity/speed trade. There isn't one to make.

**Structural costs it CANNOT remove**, named in advance so they are not later
mistaken for a failure of the change:

1. **Serial accumulation instead of an adder tree.** `weightedAxis` folds taps one at
   a time into a worst-case-width accumulator; `boxSum4` pairs them, `(a+b) + (c+d)`,
   in 3N+1 stages.
2. **A materialised intermediate.** The route runs horizontal-then-vertical through
   `hRes`; `boxSum4` fuses both axes in one tree.
3. **Worst-case plane widths** from `axisPlanes`, which a filter-specific kernel sizes
   exactly.

**DECISION RULE, WRITTEN BEFORE MEASURING.** Arms are the generic `BOX_2x2` route
before and after, against the hand-written `pyrDown` as the control, all three exact
against each other (`tests/test_pyramid.cpp` already proves the route exact against a
per-pixel reference, and that test must still pass unchanged).

- **Band A — generic route lands within 1.3× of hand-written.** The tax was
  genericity. **Implication, stated now:** `BOX_3x3` re-prices from ~398 µs toward
  ~130 µs, D-36's caller trade drops from **+0.8 ms to roughly +0.15 ms**, and the
  hand-written `pyrDown` becomes a **candidate for deletion** — one implementation
  covering all filters. That last consequence needs its own decision and is *not*
  taken by this experiment.
- **Band B — 1.3× to 2.0×.** Partial. Re-price D-36 with the measured number; keep
  both implementations; record which of the three structural costs the residual is,
  measured and not guessed.
- **Band C — still above 2.0×.** The tax is **structural, not genericity**. Say so
  plainly, leave the framework alone, and D-36's `BOX_3x3` estimate stands as measured
  rather than being revised downward on a hope.
- **Band D — slower after the change.** Constant-folding hurt, most likely through
  code growth from full specialisation. That would be a fact about this compiler on
  this target and must be reported as one, not generalised.

**A limit declared in advance.** `pyrDown` is **25.1%** of the frontend
([X-38](#x-38--e-20--the-whole-frontend-against-opencv-on-the-deployment-target--done)).
This experiment changes the **generic** route, which the shipped default does **not**
call — so **its effect on the shipped frontend is exactly zero.** What it buys is the
price of the *options*, and therefore whether D-36's trade is worth what D-36 says it
is. That is the whole claim, and it is not a speed claim about binCV as shipped.

**Method:** `benchmark/pyrfilter_benchmark.cpp`, reference device via
`scripts/run_on_pi.sh pi4`; `tests/test_pyramid.cpp` unchanged, as the exactness
oracle.

**RESULT — BAND A, AND IT OVERTURNS [D-36](ARCHITECTURE.md#8-design-decisions)'s
CENTRAL CLAIM.**

Three signatures changed — `addShifted` (extents and shift become template
parameters, staging buffer deleted), `weightedAxis` (tap count, weights and output
width come from `F`), `requantizeWeighted` + new `divideByConstantT` (the divisor
becomes a template parameter so each restoring-division step folds to a literal).
**No algorithm changed.** `tests/test_pyramid.cpp` passes with the **identical**
262 322 checks, still exact against the per-pixel integer reference.

| arm | X-39 | **X-42** | speedup | vs hand-written |
|---|---|---|---|---|
| hand-written `pyrDown` N=3 *(control, untouched)* | 93.7 | **93.8** | 1.00× | 1.00× |
| **generic `BOX_2x2` N=3** | 277.8 | **111.9** | **2.48×** | **1.19×** |
| `BOX_3x3` N=3 | 398.0 | **228.0** | 1.75× | 2.43× |
| `GAUSSIAN_3x3` N=3 | 497.7 | **225.7** | 2.21× | 2.41× |
| **`GAUSSIAN_5x5` N=3** (anchor) | 2 352.9 | **549.8** | **4.28×** | 5.86× |

Reproduced across two runs to within 0.2 µs. The control moved 93.7 → 93.8 µs,
confirming nothing outside the generic route changed.

**1. THE TAX WAS GENERICITY, AND IT WAS NEVER NECESSARY.** The generic route ran
`BOX_2x2` at **2.96×** the hand-written one; it now runs it at **1.19×** — Band A.
`F` was a template parameter the whole time; the helpers simply threw the constants
away at their own signatures. **This bought a 2.48× speedup with no algorithm change,
no accuracy change and no loss of flexibility.** The residual 1.19× is the three
structural costs named in the rule, and *only* those.

**2. THE STANDARD-LK ANCHOR IS NOW AFFORDABLE, WHICH IS THE OPPOSITE OF WHAT X-39
CONCLUDED.** Scaling by the level geometry as X-39 did (2 frames × (1 + ¼ + ¹⁄₁₆) =
2.625) against X-38's full-sequence 11.198 ms and OpenCV's 16.324 ms:

| filter | X-39 said | **X-42 says** | |
|---|---|---|---|
| `BOX_2x2` shipped | 11.169 ms, 1.48× | **11.198 ms, 1.46×** | unchanged |
| `BOX_3x3` | 11.968 ms, 1.38× | **11.550 ms, 1.41×** | +0.35 ms, was +0.80 |
| `GAUSSIAN_3x3` | 12.230 ms, 1.35× | **11.544 ms, 1.41×** | |
| **`GAUSSIAN_5x5`** | **17.099 ms, 0.97× — SLOWER** | **12.395 ms, 1.32× FASTER** | **+1.20 ms, was +5.93** |

**X-39 closed in Band C — "the anchor is not affordable" — and that is now false.**
The Gaussian anchor costs **+1.20 ms** and leaves binCV **1.32× faster than OpenCV**,
not 0.97×. **binCV can have standard-LK pyramid accuracy *and* criterion 4 at the same
time**, giving up 0.14× of speed for 1.25 yield points
([X-39 sequence arm](#x-39-sequence-arm--the-same-design-space-over-1710-frames--done):
95.83% against `BOX_2x2`'s 94.58%). That was the central trade D-36 declared
unavailable.

**3. `BOX_3x3` NO LONGER DOMINATES `GAUSSIAN_3x3` ON COST.** D-36 recorded it as
"both cheaper and more accurate". It is now **228.0 against 225.7 µs — one percent
the other way.** More accurate still (−0.10 against −0.37 vs the anchor), so it
remains the one to prefer, but **at equal cost rather than at a 1.25× discount**, and
the recorded reason was wrong.

**4. A METHODOLOGICAL POINT THIS PROJECT SHOULD KEEP.** X-39 measured a design space
**on an unoptimised framework** and drew a decision from it, naming the framework tax
as a caveat and registering E-22 — and the caveat turned out to be **larger than the
effect being decided**. Two of X-39's four conclusions do not survive. The
registration is what saved it: the number was flagged as provisional at the time, so
this is a correction rather than a discovery of a hidden error. **Measure the
framework before mapping a design space on it**, or say loudly that the map is
provisional.

**Decision: Band A.** D-36 is **restated** — see
[D-39](ARCHITECTURE.md#8-design-decisions) — with the anchor's affordability reversed
and the caller's `BOX_2x2`/`BOX_3x3` trade re-priced from **+0.8 ms to +0.35 ms**.
E-22 resolved. **The hand-written `pyrDown` is now a deletion candidate at 1.19×**;
that is a separate decision and is *not* taken here, because it is the route every
prior result in this project was measured on. Registered as
[E-25](ARCHITECTURE.md#register).

**What this does NOT claim.** The shipped default calls the hand-written route, so
**the effect on binCV as shipped is exactly zero** — as the rule said in advance. What
changed is the price of the *options*, and therefore what D-36's trade is worth.

**Method:** `benchmark/pyrfilter_benchmark.cpp` via `scripts/run_on_pi.sh pi4`, run
twice; `tests/test_pyramid.cpp` unchanged as the exactness oracle.

---

### X-43 · E-24 — can the extraction be vectorised, and what stops it? · `DONE`

**COMMITTED BEFORE THE ARM IS WRITTEN.**

**Gates:** [E-24](ARCHITECTURE.md#register). Bounded before it starts:
[X-40](#x-40--e-18--window-carried-vector-accumulators-at-n--2--done) capped
`residualSums` at **2.205×** even with counting free, so **nothing here can take the
frontend past ~1.9× against OpenCV.**

**Question.** [X-41](#x-41--e-23--what-the-extraction-is-actually-spending--done) ruled
out addressing (1.023×) and the memory system (1.129×) and left **instruction count**
as the only lever on the 45.4% of `residualSums` that is extraction. The twelve
`alignedWord` calls in a row share exactly **two** `(w0, s)` descriptors — eight on
`r.x0` (`prev`×2, `dxMag`×2, `dyMag`×2, two sign planes) and four on `srcX`
(`next`×2 at two rows) — so twelve scalar load-shift-or sequences look like they should
be about three vector ones. **Should they?**

**THE OBSTACLE, NAMED BEFORE MEASURING.** `QuantMat` stacks its planes: they share one
stride and plane `p` begins at word offset `p × height × stride`
([§4.1](ARCHITECTURE.md)). So the eight words a vector would want are in **eight
unrelated cache lines**, and **aarch64 NEON has no gather** — each element costs a
scalar load plus a lane insert. That is ~2 instructions per element against a scalar
extraction's ~6 for the whole word, so the gather plausibly costs more than the shifts
it saves. **If that is what happens, the finding is not "vectorisation does not work"
— it is "the layout forbids it", which is a different and more useful answer.**

**ARMS.** All three compute the same sink, equality checked before timing.

| arm | what it is |
|---|---|
| **A** | the shipped scalar extraction — X-41's floor arm, 263–265 µs |
| **B** | vector extraction with a **real gather**: scalar loads plus lane inserts, then one `vshlq` pair per descriptor. **This is what could actually be written today.** |
| **C** | vector extraction from **contiguous** memory — the same shifts with the gather removed by staging the words beforehand. Not implementable on today's layout; it is the **upper bound a word-interleaved plane layout would unlock.** |

**DECISION RULE, WRITTEN BEFORE MEASURING**, with each band's implication stated now —
[D-38](ARCHITECTURE.md#8-design-decisions)'s lesson from X-41's false Band C
prediction:

- **Band A — B ≥ 1.4× over A.** Write it. **And only then** collapse the three copies
  of the extraction block, since that is the block being changed. The refactor is
  justified by the change, not the reverse.
- **Band B — 1.15× ≤ B < 1.4×.** Marginal against the ~1.9× frontend cap. Write it
  **only** as part of the three-copy collapse, so the churn buys maintenance as well
  as speed.
- **Band C — B < 1.15×, and C is materially better than B.** **The gather is the
  obstacle and the layout is the cause.** Do not write B. The successor is a
  **word-interleaved plane layout** question — and note that this would be an
  *instruction-count* argument for relayout, which is **not** the cache argument X-41
  already refuted at 1.129×. Those must not be conflated.
- **Band D — B < 1.15× and C is no better either.** The extraction is irreducible on
  this ISA at this word width. **Close the whole line of work**, and say plainly that
  `residualSums` is finished: 2.205× was never reachable and what remains is not
  reachable at all.

**The refactor is deliberately NOT done first.** X-41's Band C already recorded that
the three-copy collapse is worth doing *for maintenance, not for speed*. Doing it
before knowing whether E-24 pays would be refactoring for a change that may never be
written — so the ceiling runs first, in the benchmark, touching no kernel.

**Method:** `benchmark/residual_n2.cpp`, two new arms beside X-41's floor arm;
reference device via `scripts/run_on_pi.sh pi4`.

**RESULT — BAND C. The prediction written above held, and the two halves of it
separate cleanly.**

| arm | µs | vs shipped | vs extraction A |
|---|---|---|---|
| shipped NEON | 596.8 | 1.000× | |
| X-40, reduce per window | 552.1 | 1.081× | |
| **A — scalar extraction** | **254.8** | 2.342× | **1.000×** |
| X-41 hoisted + strided | 247.7 | 2.410× | 1.029× |
| **B — vector, real gather** | **288.0** | 2.072× | **0.885× — SLOWER** |
| **C — vector, gather removed** | **155.6** | 3.836× | **1.638×** |

Arm B reproduces arm A exactly over 130 windows; arm C is a **cost model, not a
computation**, and is labelled as one in the code.

**1. THE SHIFTS ARE CHEAP AND THE GATHER IS NOT.** Removing the gather makes the
extraction **1.638×** faster — the vector spelling of "twelve load-shift-ors become
three" is real and large. Paying for the gather makes it **0.885×**, i.e. **13%
slower than scalar.** Eight scalar loads plus eight lane inserts cost more than the
eight shift-or sequences they replace. **Arm B is not written.**

**2. THE OBSTACLE IS THE LAYOUT, EXACTLY AS THE RULE PREDICTED — and this time the
prediction is worth noting, because [X-41](#x-41--e-23--what-the-extraction-is-actually-spending--done)'s
was wrong.** `QuantMat` stacks planes at word offset `p × height × stride`
([§4.1](ARCHITECTURE.md)), so the eight words a vector wants are in eight unrelated
lines and aarch64 has no gather instruction. The rule named that mechanism before the
measurement and the measurement confirmed it.

**3. THIS IS AN INSTRUCTION-COUNT ARGUMENT FOR RELAYOUT, NOT A CACHE ONE — AND THE
TWO MUST NOT BE CONFLATED.** X-41 refuted the *cache* case at **1.129×**: fitting the
entire working set in L1 buys 13%. This is a different case for a different reason and
is worth **1.638×** on the extraction. The rule pre-registered the distinction so that
the successor could not inherit X-41's refutation by association.

**4. THE PRIZE, AND WHY ARM C IS AN UPPER BOUND AND NOT A TARGET.** At arm C the
kernel would run 552.1 → 452.9 µs, **1.219×**, which is about **1.18× on LK** and
would put the frontend near **9.6 ms, ~1.70× against OpenCV** from 1.52×. **But arm C
assumes all eight words are contiguous, and they belong to FIVE SEPARATE CONTAINERS** —
`prev`, `dxMag`, `dyMag` (three `QuantMat`s) and `dxSign`, `dySign` (two `BinMat`s).
Interleaving *within* a `QuantMat` gives 2-wide contiguity at best, a fraction of arm
C's benefit. **The full 1.638× requires merging the five containers of an `LKLevelN`
into one interleaved allocation** — a far larger change than "interleave the planes",
and one that would make every single-plane bulk operation stride instead of stream.
**Arm C is the ceiling for a design that does not exist, and D-38's lesson applies to
it: a ceiling bounds the shape, not the kernel.**

**Decision: Band C — arm B is not written, and the three-copy collapse is not
triggered** (its justification was to be the change, and there is no change). E-24
resolved. Recorded as [D-40](ARCHITECTURE.md#8-design-decisions). The successor —
whether an `LKLevelN` should be one interleaved allocation — is registered as
[E-26](ARCHITECTURE.md#register) **with its cost side named**: every kernel that reads
one plane contiguously would then stride, and [CLAUDE.md](../CLAUDE.md)'s rule is that
the alternatives get measured together, not the winner alone.

**Method:** `benchmark/residual_n2.cpp` via `scripts/run_on_pi.sh pi4`. Arm B is
checked exact against arm A before timing — 0 of 130 windows differ.

---

### X-44 · E-26 — is INTERLEAVED a layout binCV should support? · `DONE — DECLINED`

**COMMITTED BEFORE ANY OF IT IS WRITTEN.**

**Gates:** [E-26](ARCHITECTURE.md#register). This is a **storage-layout** question, not
a tracker optimisation, and the framing is deliberate: if more than one operation
wants planes interleaved, binCV should support the pattern rather than special-case
`residualSums`.

**A PREMISE THIS ENTRY CORRECTS BEFORE USING IT.** Earlier drafts of this question
argued that binCV's design "rests on" planar bit-planes and that interleaving reopens
[D-2](ARCHITECTURE.md#8-design-decisions). **That is wrong and is withdrawn.**
Bit-planes are the representation this project *started* from, not its thesis — which
is processing low-bit-width frames at their true bit width. A layout is a means. So
the question is not whether interleaving is permissible; it is what it costs and what
it buys, measured on both sides.

**Question.** [X-43](#x-43--e-24--can-the-extraction-be-vectorised-and-what-stops-it--done)
found the extraction **1.638×** faster once its eight words are contiguous, and
**0.885× — slower** when they are gathered from `QuantMat`'s stacked planes. Arm C was
a **cost model with a fabricated buffer**. What does a **real** interleaved layout
cost and buy, end to end, including the conversion and including the operations it
makes worse?

**WHY A CONVERSION CAN PAY, WHICH AN EARLIER ESTIMATE GOT WRONG.** A conversion pass
is per **level per frame**; the window extractions that benefit are per **keypoint per
iteration**. [X-38](#x-38--e-20--the-whole-frontend-against-opencv-on-the-deployment-target--done)'s
LK time divided by this benchmark's per-window cost puts the frontend at roughly
**1 800 window-evaluations per frame**, so each converted word is re-read by hundreds
of windows. An earlier note priced the conversion as marginal by comparing it against
a single window. **It amortises, and by a large factor.** That is the thing this
experiment must confirm rather than assume.

**ARMS.**

| # | measures | why it decides something |
|---|---|---|
| **1** | **conversion cost**, planar → interleaved, one N=2 level, per frame | the price of entry; if this dominates, nothing else matters |
| **2** | **extraction from a REAL interleaved buffer** | replaces X-43 arm C's fabricated buffer with the real thing |
| **3** | **`residualSums` end to end** on interleaved input, conversion included | the only number that is actually about the kernel |
| **4** | **a streaming single-plane consumer** reading planar vs interleaved | the cost side — what every other operation pays |
| **5** | **footprint**, computed exactly, both variants | co-equal with speed ([CLAUDE.md](../CLAUDE.md)) |

Arms 2 and 3 must be **bit-exact** against the planar path before being timed.

**Two variants are priced, not one.** **(P) Producers write interleaved** — `pyrDown`
and the derivative kernels emit the layout directly, so there is no conversion and no
extra memory, and arm 4's cost lands on whoever reads those planes.
**(C) Conversion** — storage stays planar and the tracker converts per level per
frame, so arm 4's cost is zero and arm 1's is not. **These have opposite cost
profiles and the rule must not collapse them.**

**DECISION RULE, WRITTEN BEFORE MEASURING.** `N` is the **net** frontend ratio against
OpenCV including every cost above; today's baseline is **1.52×**, and
[X-40](#x-40--e-18--window-carried-vector-accumulators-at-n--2--done)'s cap says
`residualSums` alone cannot pass ~1.9×.

- **Band A — N ≥ 1.65× and arm 4 costs < 1.10× on the affected consumers.**
  Interleaved becomes a **supported layout**: a documented `QuantMat` storage order
  with the conversion in the public API, not a private tracker buffer. Adopt the
  cheaper of (P) and (C) and say which and why.
- **Band B — N ≥ 1.65× but arm 4 costs ≥ 1.10×.** The gain is real and it is **paid
  for elsewhere**. Adopt **(C) only**, because a conversion confines the cost to the
  one operation that benefits. Interleaved stays internal, and the entry states the
  cost it would have imposed had it gone global.
- **Band C — 1.58× ≤ N < 1.65×.** Under a 4% frontend gain for a storage-layout
  change. **Do not adopt.** Record the measured numbers so the next operation that
  wants this layout inherits them rather than re-deriving them.
- **Band D — N < 1.58×, or the conversion does not amortise.** The premise above is
  wrong and the correction is the finding: report **why** the amortisation argument
  failed, since it is the load-bearing part of the case.

**A SEPARATE OUTPUT THIS EXPERIMENT OWES REGARDLESS OF BAND.** The recurrence
criterion, stated as a rule rather than case by case: **an operation wants interleaved
when it re-reads the same words many times to reconstruct pixel values, and planar
when it streams one plane.** X-44 must report the measured crossover — **how many
re-reads a word needs before conversion pays** — because that number, not this one
kernel's verdict, is what the next operation will need.

**Method:** `benchmark/interleave_layout.cpp` (new) and `benchmark/residual_n2.cpp`;
reference device via `scripts/run_on_pi.sh pi4`.

**RESULT — BAND B on speed, and THE RULE ITSELF HAS A HOLE that has to be reported
before any band is applied.**

**Arm 3 — the real kernel** (`residual_n2`, 130 windows of 31×31, N=2):

| arm | µs | vs planar extraction |
|---|---|---|
| extraction, planar (shipped) | 255.7 | 1.000× |
| [X-43](#x-43--e-24--can-the-extraction-be-vectorised-and-what-stops-it--done) C, fabricated buffer | 155.4 | *1.645× — the old ceiling* |
| **arm 3, REAL interleaved buffer** | **177.0** | **1.445×** |

Bit-exact against the planar path over 130 windows. **The fabricated buffer overstated
by 1.14×** — much closer than [D-37](ARCHITECTURE.md#8-design-decisions)'s and
[D-40](ARCHITECTURE.md#8-design-decisions)'s ceilings, because this one differed from
the real thing only in where the memory lived.

Kernel: extraction 255.7 → 177.0 with counting unchanged at 294.5 µs, so
**550.2 → 471.5 µs = 1.167×.**

**Arms 1, 4 and 5 — the cost side** (`interleave_layout`, 376×240 level, 8 planes):

| | |
|---|---|
| **arm 1** conversion, planar → interleaved | **23.7 µs** per level per frame |
| **arm 4** streaming one plane: planar 0.605 µs, interleaved 3.129 µs | **5.17× COST** |
| **arm 5** interleaved buffer for the largest N=2 level | **92 160 B** |

**1. THE STREAMING COST IS 5.17×, WHICH ELIMINATES BAND A OUTRIGHT.** Band A required
under 1.10× on the affected consumers. Striding by four words touches one useful word
per cache line and discards the rest — **interleaving cannot be binCV's general layout
on this device**, and that is now measured rather than argued.

**2. THE CROSSOVER — ONE DATA POINT, AND DELIBERATELY NOT PROMOTED TO A CRITERION.**
Conversion costs 23.7 µs; interleaving saves `(255.7 − 177.0)/130 = 0.605 µs` per
31-row window, so **on this level it paid after ≈ 39 windows**. The frontend evaluates
roughly 600 windows per level per frame, amortising ~15×.

**This entry's rule promised to "report the measured crossover" as something the next
operation could use. It is reported, and the promotion is withdrawn.** One level size
(376×240), one plane count, one word type, one cache hierarchy — and the conversion
scales with level *area* while the saving scales with *window count*, two quantities
that vary independently, so the ratio is not portable. What generalises is the
*shape* of the argument — a conversion amortises when the converted data is re-read
many times — **not the number**. Any future operation asking this gets measured on its
own terms.

**3. THE NET FRONTEND ESTIMATE LANDS ON THE BAND BOUNDARY, AND SAYS SO.** N = 2 levels
are ~6/7 of LK by [X-34](#x-34--the-ladder-in-isolation--done)'s ladder ratio, so
6.32 of 7.374 ms improves 1.167× → saving **0.90 ms**, less 0.031 ms of conversion
across the three levels: frontend **10.757 → 9.89 ms, ≈ 1.65× against OpenCV** from
1.52×. **Band B's threshold is 1.65×.** The estimate chains three prior measurements
and its uncertainty straddles the boundary — **it is not a measured frontend number**,
and this entry does not pretend otherwise. Band A is excluded on arm 4 regardless, so
the boundary does not change the verdict.

**4. THE RULE DID NOT PRE-REGISTER A FOOTPRINT BAND. THAT IS A DEFECT IN THE RULE AND
IT IS DECISIVE.** [CLAUDE.md](../CLAUDE.md): *"Performance and memory footprint are
co-equal goals. When they conflict and no explicit choice has been made, memory
wins."* Converting one level at a time and reusing the buffer needs the largest N = 2
level only — **+92 160 B on a 436 704 B peak, +21%**, taking criterion 3 from
**6.23× to 5.15×**.

**So the trade is +21% footprint for +8% speed, and the project's own tie-breaker says
memory wins.** X-44's bands were written on speed alone and therefore **cannot settle
this**. Under [CLAUDE.md](../CLAUDE.md)'s "stop and ask" — *a decision is needed that
isn't recorded in ARCHITECTURE §8* — **this is escalated rather than decided.**

**Decision: the trade is DECLINED.** Escalated as a goals question rather than
decided by a band — and answered: **binCV does not spend 21% of its footprint
advantage on 8% of speed.** The measurements stand as the record of what was on offer
and at what price. What *is* decided, on arm 4's 5.17×:
**interleaving will not become binCV's general storage layout** — any adoption is
variant (C), a conversion confined to the operation that benefits. Recorded as
[D-41](ARCHITECTURE.md#8-design-decisions); the open half stays as
[E-26](ARCHITECTURE.md#register), reduced to a single yes/no with both numbers on it.

**A premise this entry confirms.** The conversion **does** amortise, by ~15×, exactly
as the rule argued and against the earlier note that priced it against a single
window. The amortisation was never the problem; the footprint is.

**Method:** `benchmark/interleave_layout.cpp` and `benchmark/residual_n2.cpp` via
`scripts/run_on_pi.sh pi4`. Arm 3 is bit-exact against the planar extraction over 130
windows; `interleave_layout`'s two extractions agree over 2 695 cases.

---

### X-45 · `pyrDown` against `cv::pyrDown` across BIT WIDTH · `DONE`

**A CHARACTERISATION, NOT A DECISION — and therefore deliberately without a
pre-registered rule.** [CLAUDE.md](../CLAUDE.md) requires the rule before a
*measurement that settles a choice*. Nothing is chosen here: this measures a property
of the representation. It **informs** the open `pyrDown`-default question, and if that
question is settled by numbers, its rule gets written separately and on top of these.

**Question.** binCV's footprint claim obviously vanishes at 8 bits per pixel — both
sides store a byte. **What happens to the SPEED claim?** Bit-slicing pays per bit of
intermediate precision where SIMD pays a flat rate per vector, so the two should cross
somewhere. **Where?**

**Workload:** 640×480 → 320×240, reference device, `cv::pyrDown` pinned to one thread
as the denominator ([CLAUDE.md](../CLAUDE.md): OpenCV doing the same semantic
operation). `pyrDownFiltered<Gaussian5x5, 8, 8>` is **verified exact** against a
per-pixel integer reference before being timed — it is the framework's widest point
(12-plane horizontal accumulator, 16-plane vertical, divisor 256×255), so if anything
overflowed it would overflow there. `tests/test_pyramid.cpp` now carries that case.

| arm | µs | **vs `cv::pyrDown`** |
|---|---|---|
| **`cv::pyrDown`, 8U** (denominator) | **517.8** | **1.00×** |
| binCV `BOX_2x2`, **1 → 3** *(shipped)* | **93.8** | **5.52× FASTER** |
| binCV `GAUSSIAN_5x5`, **1 → 3** | 549.7 | 0.94× — **parity** |
| binCV `BOX_2x2`, **8 → 8** | 2 614.3 | 0.20× — 5.0× slower |
| **binCV `GAUSSIAN_5x5`, 8 → 8** *(cv::pyrDown's exact shape)* | **7 111.7** | **0.073× — 13.7× SLOWER** |

**1. THE CROSSOVER IS BIT WIDTH, AND IT IS STEEP.** Same filter, same image, same
device: **1 → 3 bits costs 549.7 µs and 8 → 8 costs 7 111.7 µs — 12.9×.** Bit width
dominates filter choice by roughly five to one (filter alone, at fixed 8 → 8, is
2.7×). This is the project's thesis stated as a single measurement: **binCV is 5.5×
faster than OpenCV at the bit width it exists for, and 13.7× slower at OpenCV's.**

**2. WHY, AND IT IS STRUCTURAL RATHER THAN AN OPTIMISATION GAP.** A bit-sliced
weighted sum's cost scales with the *number of planes in the accumulator*, which
`axisPlanes` puts at `bits(weightSum × (2^N − 1))` — 5 and 9 planes at N = 1, **12 and
16 at N = 8**. SIMD pays the same price for an 8-bit lane as for a 1-bit one, because
the lane is 8 bits wide either way. **Bit-slicing buys its advantage by not paying for
bits it does not use, and at 8 bits there are none to skip.**

**3. THE CONSEQUENCE FOR API COMPATIBILITY IS SHARP AND SHOULD NOT BE SOFTENED.** An
`8 → 8` `cv::pyrDown`-compatible call has **no footprint advantage by construction**
and is **13.7× slower**. It is worth shipping — an API that borrows a name should mean
it — but it must be documented as **correct, not fast**, in the docstring and not only
in this log. A user who benchmarks that configuration and concludes binCV is pointless
would be reading it correctly.

**4. AND AT binCV's OWN BIT WIDTHS, OPENCV's FILTER IS AFFORDABLE.** `GAUSSIAN_5x5` at
1 → 3 is **0.94× of `cv::pyrDown`** — parity in time, at **⅜ the stored bits** and with
[X-39](#x-39-sequence-arm--the-same-design-space-over-1710-frames--done)'s accuracy
1.26 yield points above the shipped box. So *matching OpenCV's filter* costs binCV
nothing against OpenCV; only *matching OpenCV's bit width* does. **Those two are
routinely conflated and this entry separates them.**

**Decision:** none — see the header. What it establishes is the price list the
`pyrDown`-default question needs.

**Method:** `benchmark/pyrfilter_benchmark.cpp` via `scripts/run_on_pi.sh pi4` with
`BINCV_PI_OPENCV=1`; `tests/test_pyramid.cpp` for the 8 → 8 exactness case.

---

### X-46 · WHERE does bit-slicing stop paying? · `DONE`

**A CHARACTERISATION, like [X-45](#x-45--pyrdown-against-cvpyrdown-across-bit-width--done),
and for the same reason: it measures a property of the representation and settles
nothing by itself.**

**Question.** X-45 gave two endpoints — 5.5× faster at 1 → 3 bits, 13.7× slower at
8 → 8. **Where is the line?** It decides what "low bit width" means as an engineering
claim rather than a slogan, and it is the number any 8-bit-specialisation proposal has
to be argued against.

**Workload:** 640×480 → 320×240, reference device, `cv::pyrDown` on `CV_8U` at one
thread as the flat denominator. **One process per arm** — see the method note below.

| N | **box 2×2** N→N | vs `cv::pyrDown` | **Gaussian 5×5** N→N | vs `cv::pyrDown` |
|---|---|---|---|---|
| **1** | **84.5 µs** | **5.98× faster** | 447.5 µs | **1.13× faster** |
| **2** | 197.1 | **2.56× faster** | 1 285.7 | 0.39× |
| **3** | 299.3 | **1.69× faster** | 1 909.3 | 0.26× |
| **4** | 434.5 | **1.16× faster** | 2 454.9 | 0.21× |
| **5** | 632.2 | 0.80× | 4 159.0 | 0.12× |
| **8** | 2 616.0 | 0.19× | 7 093.6 | 0.07× |

Binary input, which is binCV's actual shape: **box 1 → 3 = 111.9 µs (4.52× faster)**,
gauss 1 → 3 = 550.0 (0.92×), gauss 1 → 5 = 566.1 (0.89×). `cv::pyrDown` = 505.5 µs.

**1. THERE IS NO SINGLE CROSSOVER — IT IS FILTER-DEPENDENT, AND THE SPREAD IS THREE
BITS.** The box crosses **between 4 and 5 bits**; the Gaussian crosses **between 1 and
2**. Any rule of the form *"specialise at N = k"* is wrong for one of them. What sets
the line is the accumulator width `bits(weightSum × (2^N − 1))`, so the filter's weight
sum and the bit depth trade against each other directly.

**2. INPUT WIDTH DOMINATES OUTPUT WIDTH BY ABOUT 7:1.** Gaussian **1 → 5 costs 566 µs;
5 → 5 costs 4 159** — same output, 7.3× apart. The horizontal pass runs over `NIn`
planes per tap, so **what binCV charges for is the precision it READS**, not the
precision it writes. That is a more useful statement of the library's advantage than
"low bit width": **binCV is fast when its INPUT is narrow.** The shipped ladder
(binary in, 2–3 bits out) sits exactly there, and it is not a coincidence — it is what
the representation is good at.

**3. THE SHIPPED CONFIGURATION IS 4.5× FASTER THAN OPENCV** at the operation OpenCV
would have to do at 8 bits regardless. The denominator is flat across the whole sweep
**because OpenCV has no cheaper mode for a caller who only needs three bits**, and that
asymmetry is the entire product.

**METHOD NOTE — THE FIRST VERSION OF THIS BENCHMARK WAS WRONG AND IS RECORDED AS
WRONG.** It declared all eight source widths up front — 1 + 2 + … + 8 planes of
640×480, about **1.4 MB against this device's 1 MB L2** — and `measureInterleaved`
pumped that whole set between samples. Every cheap arm ran cache-cold: it reported
`box 1 → 3` at **352.7 µs** where `pyrfilter_benchmark` measures the *identical call*
at **112.1 µs**, a **3.1× inflation**, and it would have put the box crossover at 2
bits instead of 4–5. **The disagreement between two benchmarks measuring the same call
is what caught it**, which is an argument for keeping overlapping arms rather than
trimming them. A second version scoped the matrices per arm and left the lambdas
holding dangling references, aborting in `malloc`. The shipped version runs **one arm
per process** (`benchmark/crossover_sweep.sh`), and its `box 1 → 3` reads 111.9 µs
against the independent 112.1.

**Decision:** none — see the header. It is the price list that any proposal to
specialise wide-`N` paths has to be argued against, and it says such a proposal cannot
key on `N == 8`.

**Method:** `benchmark/bitwidth_crossover.cpp` + `benchmark/crossover_sweep.sh` via
`scripts/run_on_pi.sh pi4` with `BINCV_PI_OPENCV=1`.

---

### X-47 · Interop or specialisation above the crossover? · `DONE`

**COMMITTED BEFORE THE CONVERSION EXISTS.** Unlike
[X-45](#x-45--pyrdown-against-cvpyrdown-across-bit-width--done) and
[X-46](#x-46--where-does-bit-slicing-stop-paying--done), this one **settles a
choice**, so it gets a rule first.

**Gates:** the wide-`N` question X-46 priced. binCV's operations are 2.5–14× slower
than OpenCV above the (filter-dependent) crossover. The proposal on the table was to
**specialise** wide-`N` cases internally to a byte representation; the counter-proposal
is **interop** — make `QuantMat<N>` ↔ `cv::Mat` conversion first-class, and let callers
hand wide intermediates to OpenCV, which already exists and is already optimal at
8 bits. Specialising internally would mean a second storage layout and a second
implementation of every kernel — OpenCV rebuilt inside binCV — so it has to beat
interop by a margin that pays for that, not merely tie it.

**What gets built first (and is the experiment's subject):** `fromCVMat` / `toCVMat` /
`toCVMatNormalized` on the general `QuantMat<N>`, today `QuantMat<1>`-only. The
conversion is **transpose-based, not per-pixel** — an 8×8 bit-matrix transpose
(three delta-swaps) moves 8 pixels × 8 planes per step. This is deliberate and is
[X-42](#x-42--e-22--is-the-filter-frameworks-cost-genericity-or-structure--done)'s
lesson applied in advance: measuring the round trip through a naive per-pixel
conversion would price the *strawman*, and the caveat would exceed the effect.
The transpose's orientation and the quantise/dequantise round-trip law
(`round(round(v·255/maxV)·maxV/255) == v`, exact for every `v` at every `N`, no ties
because 255 and `2^N−1` are odd) are **verified numerically before the C++ exists**.

**Semantics, fixed before measuring.** `toCVMat` = raw values 0..`MaxValue` (exact,
one-way at `N < 8`); `toCVMatNormalized` = `round(v·255/maxV)` (the OpenCV bridge);
`fromCVMat` = `round(v·maxV/255)` (its exact inverse). **The `QuantMat<1>`
specialisation keeps its established nonzero-threshold `fromCVMat`** — the two
disagree for bytes 1..127 at N = 1, recorded as a deliberate difference, not
unified retroactively.

**Measurement:** reference device, one arm per process (X-46's method note), OpenCV
at one thread. `R` = the 8→8 round trip `toCVMatNormalized` → `cv::pyrDown` →
`fromCVMat`, against `B` = native `pyrDownFiltered<Gaussian5x5, 8, 8>` = **7 094 µs**
(X-46) and the floor `cv::pyrDown` ≈ 505 µs. Conversion is also timed per direction at
N = 8 and N = 3, because that per-frame tax `T` is what generalises: **any operation's
interop decision is `native_binCV − native_OpenCV` against `T`**, which is the
"per-operation crossover" published as a formula instead of a table.

**DECISION RULE, WRITTEN BEFORE MEASURING.**

- **Band A — R ≤ B/2 (≤ ~3 550 µs):** **interop is the answer.** No internal wide-`N`
  specialisation, ever, on this evidence; the conversion ships as the documented
  pattern for wide intermediates. D-record.
- **Band B — B/2 < R < B:** interop wins but by under 2×. Still no specialisation —
  a parallel byte implementation of the library cannot be justified by less than 2× —
  but the thin margin is stated, and a NEON transpose is named as the follow-up if a
  real pipeline needs more.
- **Band C — R ≥ B:** the conversion is too expensive and the interop recommendation
  was wrong as offered. **Contingency named now:** first check whether the conversion
  dominates `R` and whether a vector transpose plausibly moves the band — the
  conversion is new code and X-42 showed new-framework numbers can carry a 3×
  removable tax. Only if that fails does specialisation come back on the table.

**Method:** `QuantMat<N>` conversions in `quantMat.hpp` (OpenCV-guarded); exactness
tests in `tests/test_opencv_interop.cpp` — transpose vs per-pixel reference, the
round-trip law at every `N`, padding-bit invariant after `fromCVMat`;
`benchmark/interop_roundtrip.cpp` via `scripts/run_on_pi.sh pi4` with
`BINCV_PI_OPENCV=1`.

**RESULT — BAND A, but the entry that first recorded it had SIX DEFECTS an
adversarial review caught, and the corrections are below rather than folded away.**

Reference device, one arm per process, OpenCV at one thread, throttle bits unchanged
across the run. **Spread across repeat batches is reported next to every median** —
`measure_util.hpp`'s own hazard 3, which the first version of this benchmark violated
by printing medians alone to 0.1 µs:

| arm | µs (median) | spread |
|---|---|---|
| `toCVMatNormalized`, N=8, 640×480 | 952.6 | 0.2% |
| `fromCVMat`, N=8, 640×480 | 1 616.6 | 0.8% |
| **`fromCVMat`, N=8, 320×240** | **389.3** | 0.2% |
| **R — round trip `to` → `cv::pyrDown` → `from`** | **1 906.2** | 0.7% |
| **B — native `GAUSSIAN_5x5` 8→8** | **7 092.8** | 0.9% |
| `cv::pyrDown` alone (floor) | 516.1 | 0.2% |
| `toCVMatNormalized`, N=3, 640×480 | 748.1 | 0.1% |
| `fromCVMat`, N=3, 640×480 | 875.3 | 0.3% |

**R = 1 906.2 ≤ B/2 = 3 546.4 — Band A, at 0.27·B.** All spreads under 1%, so every
gap here is far outside the noise.

**1. THE ROUND TRIP IS 3.7× FASTER THAN binCV's OWN PATH — AND 3.7× SLOWER THAN A
SPECIALISATION'S CEILING. BOTH HALVES MATTER.** The floor arm is the ceiling any
internal byte specialisation could reach: **516.1 µs**. So the honest ordering is
`specialisation (516) < interop (1 906) < native bit-sliced (7 093)`. **A
specialisation would be faster, and the reason not to build one is the cost model,
not the speed** — that model was pre-registered above ("a second storage layout and a
second implementation of every kernel — OpenCV rebuilt inside binCV"), so the argument
is not post-hoc, but the first version of this entry let its table imply interop won
on time. It does not. It wins on **time-per-unit-of-machinery**.

**2. FOOTPRINT — THE HALF THIS ENTRY ORIGINALLY OMITTED, WHICH IS THE SAME DEFECT
[X-44](#x-44--e-26--is-interleaved-a-layout-bincv-should-support--done--declined)
REPORTED IN ITS OWN RULE, ONE EXPERIMENT LATER.** Peak live bytes, computed exactly:

| path | peak | |
|---|---|---|
| native bit-sliced 8→8 | **384 000 B** | 37% of the device's 1 MiB L2 |
| **interop round trip** | **844 800 B** | **2.20×**, 81% of L2 |

**The interop path materialises a full byte-per-pixel frame, which is precisely what
binCV exists to avoid.** [CLAUDE.md](../CLAUDE.md) makes footprint co-equal and gives
memory the tie-break — so this number had to be on the record **before** the decision,
not after. It does not reverse Band A, because at 8 bits binCV has **no footprint
advantage to protect** ([X-45](#x-45--pyrdown-against-cvpyrdown-across-bit-width--done):
8 bpp on both sides by construction) and the byte buffers are transient rather than
pipeline-resident. **But that is an argument, and it should have been made in the
rule.** Recorded as a rule defect, not silently repaired.

**3. R AND B DO NOT COMPUTE THE SAME ANSWER, AND NOW THAT IS QUANTIFIED RATHER THAN
UNSTATED.** `measure_util.hpp`'s hazard 4 requires that whatever is compared agrees
before it is timed; this benchmark did not check. It does now:

> **1 114 of 76 800 destination pixels differ — and ZERO of them are interior.**
> Max |Δ| 73 of 255, entirely on a 2-pixel rim.

Exactly the predicted deviation: `cv::pyrDown` uses `BORDER_REFLECT_101`,
`pyrDownFiltered` reads outside the frame as zero (Tier 3, documented in
`ops/pyramid.hpp`). **The substitute changes the answer at the border and nowhere
else** — which for a caller sending a wide intermediate to OpenCV is an *improvement*,
since OpenCV's border is the reference one. But "3.7× faster" without this line was an
incomplete claim.

**4. THE TAX `T` NOW HAS AN ARM PER GEOMETRY, WHERE THE FIRST VERSION QUOTED ONE
NUMBER THAT NO ARM MEASURED.** "T ≈ 1.4 ms" was 953.5 (export at 640×480) plus a
320×240 import that only existed inside R and was never timed. Measured directly:

| | export | import |
|---|---|---|
| 640×480 | 952.6 | 1 616.6 |
| 320×240 | — | 389.3 |

R's decomposition now closes: 952.6 + 516.1 + 389.3 = **1 858.0 against 1 906.2
measured**, a 2.5% residual. **A size-preserving 640×480 operation pays 2 569 µs both
ways, not 1.4 ms** — the original figure under-counted by 1.8× for that case, because
it silently mixed a full-size export with a decimated import.

**The formula, now stated with its geometry:** send an operation to OpenCV when
`native_binCV − native_OpenCV > T(in) + T(out)`, each term taken at the size that
side actually processes. For the 8→8 Gaussian: 7 093 − 516 = 6 577 against 1 342 —
**4.9× over**, and for a chain of wide operations the tax is paid once at each end
rather than per operation.

**5. `fromCVMat` COSTS 1.7× `toCVMatNormalized`, AND THIS BENCHMARK CANNOT SAY WHY.**
The first version attributed it to the per-call allocation. That is **not separable
here**: the export arms hoist their destination so `cv::Mat::create` is a no-op from
warm-up on, while the import allocates every call by API contract. The asymmetry is
real and is the first place to look if the tax ever matters; **the causal claim is
withdrawn** as unmeasured.

**6. WHAT THE CONVERSION BUYS BEYOND THIS EXPERIMENT.** `QuantMat<N>` was previously
unreachable from `cv::Mat` at any `N > 1` — a caller holding a wide intermediate had
**no way in at all**.

**Decision: BAND A.** No internal wide-`N` specialisation on this evidence — closed,
not deferred. Interop ships as the documented pattern.
[D-42](ARCHITECTURE.md#8-design-decisions).

**A DEFECT IN THE CODE, FOUND BY THE SAME REVIEW AND FIXED.** `fromCVMat` read
`empty() ? DefaultRowAlignment : getRowAlignment()`, silently downgrading an opt-in
row alignment. The guard was **redundant** — every constructor establishes a valid
alignment — and the trigger was **buffer reuse**, not the degenerate case: a moved-from
matrix is empty but keeps its alignment, so `dst = std::move(src); src.fromCVMat(f);`
rebuilt `src` at word granularity and dropped a Tier 2 / DMA stride. No test saw it,
because every destination in the suite was default-aligned. Fixed, and
`OpenCVInterop.QuantMatAlignmentReuse` **fails on the old code and passes on the
new** — checked by reverting.

**Method:** `benchmark/interop_roundtrip.cpp` + `benchmark/interop_sweep.sh` via
`scripts/run_on_pi.sh pi4` with `BINCV_PI_OPENCV=1`; exactness in
`tests/test_opencv_interop.cpp`.

---

### X-48 · `BORDER_REFLECT_101`, and the Tier 1 claim made good · `DONE`

**IMPLEMENTATION AND VERIFICATION, NOT A DECISION — so no pre-registered rule.** The
decision was made explicitly: *same-named functions supply OpenCV's behaviour by
default; our alternatives are documented options.* This builds it and checks it.

**Question.** `pyrDownFiltered` read outside the frame as ZERO; `cv::pyrDown` uses
`BORDER_REFLECT_101`. Reflect-101 was rejected once in this file's own header as *"a
per-pixel index map, and not word-parallel"* — the same objection that kept it out of
the LK taps. **Does that objection actually apply here?**

**IT APPLIES TO ONE AXIS AND NOT THE OTHER, WHICH IS WHY THIS IS AFFORDABLE.**

- **Vertical reflection is FREE.** The filter reads whole rows, so reflecting a row
  index picks a different **row pointer** and changes nothing per pixel. The
  word-parallel body is untouched.
- **Horizontal reflection is genuinely per-pixel** — the original objection stands —
  **but it only reaches the output columns whose source support crosses an edge**:
  `ceil(Radius/2)` per side, computed rather than assumed, which is **1** for the 5×5
  Gaussian and **0 on the left** for `Box2x2` and `DirectSubsample`. Those columns are
  recomputed from the per-pixel definition; the interior keeps the word-parallel path.

The per-pixel definition is `impl::pyrDownPixel`, and **the shipped path calls it** on
the rim rather than it being a test scaffold — so the reference and the implementation
cannot drift apart.

**RESULT 1 — `pyrDownFiltered<Gaussian5x5, 8, 8, W, Reflect101>` IS `cv::pyrDown`,
BIT FOR BIT.**

| source | destination | |
|---|---|---|
| 64×48 (even) | 32×24 | **0 of 768 differ** |
| 63×47 (odd both) | 32×24 | **0 of 768 differ** |
| 65×32 (odd width) | 33×16 | **0 of 528 differ** |
| 32×65 (odd height) | 16×33 | **0 of 528 differ** |
| **9×7** (taps reach past **both** edges at once) | 5×4 | **0 of 20 differ** |

The 9×7 case is the one that matters: at that size a single tap folds past one edge
*and then the other*, which is why `reflect101` loops rather than folding once. **The
Tier 1 claim is now proven rather than asserted**, and `pyrDown` may legitimately
carry `cv::pyrDown`'s name at that configuration.

**RESULT 2 — the border costs 9%, on the filters that have a rim.** Same run, so drift
cancels:

| arm (640×480 → 320×240, 1 → 3 bits) | µs | |
|---|---|---|
| `GAUSSIAN_5x5`, `Zero` | 549.9 | the previous behaviour |
| **`GAUSSIAN_5x5`, `Reflect101`** | **599.2** | **+9.0%** |
| `BOX_2x2`, `Reflect101` | 116.3 | no rim at all — `lo == 0`, and at even width no right rim either |

**Nine percent for exact OpenCV semantics is a good price**, and it is charged only
where geometry demands it. Against `cv::pyrDown` (516.6 µs in this run), the matching
Gaussian at binCV's own bit widths is **0.86×** — near parity, at ⅜ the stored bits —
while the shipped box is **4.44× faster**. **Matching OpenCV's FILTER AND BORDER costs
binCV roughly parity; matching OpenCV's BIT WIDTH is what costs 13.7×**
([X-45](#x-45--pyrdown-against-cvpyrdown-across-bit-width--done)). The two remain
separate, and this entry keeps them separate.

**Both borders are verified, at both parities.** `Zero` did not become untested when it
stopped being the default: `test_pyramid.cpp` now checks every filter against a
per-pixel reference under **`Reflect101` and `Zero`, at even and odd extents** — 5
size/border combinations × 8 `(filter, NIn, NOut)` points. The odd extents are not
decoration: the right rim is where an odd width makes the last output column read a
source column that does not exist, and **the even-only test that shipped before could
not see it**. The test's reflection is spelled independently of `impl::reflect101`, so
it cannot inherit a fold bug from the code it checks.

**Method:** `ops/pyramid.hpp` (`PyrDownBorder`, `reflect101`, `pyrDownPixel`, the rim
fixup); `tests/test_pyramid.cpp`; `benchmark/pyrfilter_benchmark.cpp` via
`scripts/run_on_pi.sh pi4` with `BINCV_PI_OPENCV=1`.

---

### X-49 · The frontend after the API swap: a control, and a new headline · `DONE`

**A CONTROL, not a new question.** `pyrDown` changed meaning
([X-48](#x-48--border_reflect_101-and-the-tier-1-claim-made-good--done) and the API
swap): it is now `cv::pyrDown`, and the frontend asks for `pyrDownBox` explicitly.
**The swap compiled silently at all 25 call sites**, so "the migration was complete"
is a claim that has to be measured, not asserted.

**Result 1 — the swap cost the frontend NOTHING.** 692-frame prefix, against the same
prefix at `82daca6`:

| | before | after |
|---|---|---|
| flow median / p90 / p99 | 0.0386 / 0.1177 / 14.4781 | **identical** |
| within 1 px | 97.4% | **97.4%** |
| lifetime, survival | 13 vs 13, 97.1%/97.1% | **identical** |
| footprint | 6.23× | **6.23×** |
| **build (`pyrDown` + derivatives)** | 2.884 ms | **2.887 ms** |

Build moved by **0.1%**, which is noise. Every accuracy figure reproduces to the last
digit. The migration is complete and the box path is byte-for-byte what it was.

**Result 2 — X-40 landed exactly as forecast, and this is the first end-to-end
confirmation of it.** [X-40](#x-40--e-18--window-carried-vector-accumulators-at-n--2--done)
measured a **1.069×** kernel gain on the N = 2 levels and predicted **~1.06× on LK**
from the ladder share. Measured on the prefix: **track 7.774 → 7.342 ms, 1.059×.**
A forecast from a micro-benchmark, confirmed end to end, within 0.001.

**Result 3 — the criterion-4 headline improves.** Full 1710-frame sequence:

| criterion | binCV | OpenCV | |
|---|---|---|---|
| 2 · lifetime, survival, flow median | 11 / 96.4% / 0.0434 px | 12 / 96.6% | **unchanged from X-38** |
| 3 · peak footprint | 436 704 B | 2 719 832 B | **6.23× smaller** |
| **4 · speed** | **10.644 ms/frame** | 16.289 ms/frame | **1.53× FASTER** (was 1.46×) |

**Every criterion-2 figure is bit-identical to X-38's full-sequence run**, so the
gain is pure speed with no accuracy cost. The stage profile: track 68.3%, build
26.3%, detect 5.4%.

**Decision:** none. This is the record that
[D-35](ARCHITECTURE.md#d-35-all-four-roadmap-success-criteria-are-met-on-the-deployment-target)'s
criterion-4 figure now points at.

**Method:** `benchmark/frontend_sequence.cpp` via `scripts/run_on_pi.sh pi4` with
`BINCV_PI_OPENCV=1`, run twice — the whole directory, and the `692` prefix as the
control.

---

### X-50 · E-19 — is `1/2/2/2` + `BOX_2x2` still the operating point? · `DONE`

**COMMITTED BEFORE MEASURING.**

**Gates:** [E-19](ARCHITECTURE.md#register), the last open question from the pyramid
thread.

**THREE CORRECTIONS TO E-19's OWN PREMISE, MADE BEFORE USING IT.** The register entry
was written in the middle of the x86 era and its framing does not survive:

1. *"LK is 94.7% of the frontend"* — it is **68.3%** ([X-49](#x-49--the-frontend-after-the-api-swap-a-control-and-a-new-headline--done)),
   because D-30…D-37 made LK faster and `pyrDown` did not. Build is now **26.3%**, so
   the ladder's cost is split across two stages rather than concentrated in one.
2. *"at `1/1/1/1` binCV is 1.34× slower than OpenCV against 3.08× at `1/2/2/2`"* —
   **both are x86 numbers**, from the era [D-35](ARCHITECTURE.md#8-design-decisions)
   corrected. On the reference device binCV is **1.53× FASTER** at `1/2/2/2`. The
   ladder is therefore not buying its way out of a deficit; it is spending a surplus.
3. *"This is the largest single speed lever left, larger than E-18."* E-18 is closed
   and delivered 1.069×; this claim was never measured and is not assumed here.

**THE BIT-DEPTH AXIS IS ALREADY ANSWERED, and re-deriving it would be waste.**
[X-39's sequence arm](#x-39-sequence-arm--the-same-design-space-over-1710-frames--done)
measured `BOX_2x2` **flat across N = 2→7** — 94.49 / 94.58 / 94.55 / 94.51, a
0.09-point band over 1.18 M keypoint-cases. **Under the box, bits past 2 buy nothing**,
so `1/3/3/3` and deeper are excluded on evidence rather than re-measured. What is NOT
answered is the opposite direction: **`1/2/2/2` says every coarse level needs 2 bits,
and nobody has ever measured whether they do.** `1/2/1/1` and `1/2/2/1` have never been
run, on either axis.

**AND THE TWO AXES ARE COUPLED, which is why this is one experiment and not two.**
X-39 found the *filter* decides how much depth pays: the box saturates immediately, the
Gaussian keeps paying. [X-42](#x-42--e-22--is-the-filter-frameworks-cost-genericity-or-structure--done)
then re-priced `BOX_3x3` from +0.8 ms to **+0.35 ms** at **−0.10 yield points from the
Gaussian anchor**. So the live question is not "which ladder" but **"which (ladder,
filter) point"** — and a shallower ladder with a better filter may beat a deeper ladder
with a cheap one. Arms: `{1/1/1/1, 1/2/1/1, 1/2/2/1, 1/2/2/2} × {BOX_2x2, BOX_3x3}`.

**MEASURED ON THREE AXES, NOT TWO.** X-44's rule was written on speed alone and X-47's
repeated the mistake one experiment later. **The ladder changes footprint directly** —
fewer planes, fewer bytes — and [CLAUDE.md](../CLAUDE.md) makes footprint co-equal with
speed and gives it the tie-break. So every arm reports **yield, time and bytes**, and no
band fires on time alone.

- **Yield**: the X-39 sequence harness over the EuRoC sequence, ≥ 100 frames, six warps,
  which is the statistic X-25's own conclusions had to be withdrawn for lacking.
- **Time**: `benchmark/pyramid_depth_benchmark.cpp`, build and track separately, on the
  reference device.
- **Bytes**: exact, from the ladder's own `bytes()`.

**DECISION RULE, WRITTEN BEFORE MEASURING.** Reference is the shipped point,
`1/2/2/2` + `BOX_2x2`. An arm **dominates** it if it is no worse on all three axes and
better on at least one, with "no worse on yield" meaning **within 0.5 points** — a
tenth of the `1/1/1/1` gap X-25 measured, and comfortably above this harness's
frame-to-frame noise.

- **Band A — some arm DOMINATES the shipped point.** Switch the default to it. D-23 was
  chosen on a confounded speed estimate and a single-frame accuracy read; being
  overturned by a three-axis sequence measurement is the system working.
- **Band B — no arm dominates, but an arm is within 0.5 yield points at materially
  lower cost** (≥ 15% on frontend time **or** ≥ 15% on bytes). The frontier is a
  caller's choice: keep the default, **ship the alternative as a documented operating
  point** with all three numbers, as [D-24](ARCHITECTURE.md) put route (a) and
  [D-36](ARCHITECTURE.md) put the filter set.
- **Band C — the shipped point is on the frontier and nothing comes close.** D-23 is
  **confirmed on evidence it never had**: its speed basis was confounded and its
  accuracy basis was seven synthetic warps of one image. Confirming a decision with
  better evidence is a result, not a null.
- **Band D — the shipped point is DOMINATED on yield**, i.e. some arm is *more accurate*
  and cheaper. That would mean `1/2/2/2` is over-provisioned and the extra bits are
  actively hurting; report the mechanism, not just the number.

**A limit declared in advance.** Track is 68.3% of the frontend and build 26.3%, so even
eliminating the ladder's entire cost differential cannot move the frontend more than the
arms' measured spread — this experiment is bounded by what the table shows and no
extrapolation beyond it will be made.

**Method:** `benchmark/pyramid_depth_benchmark.cpp` (new intermediate arms);
`tests/test_opticalflow.cpp` X-39 sequence harness (new ladder × filter sweep);
reference device via `scripts/run_on_pi.sh pi4`.

**RESULT — BAND A. THE SHIPPED POINT IS THE ONLY ONE OF THE SEVEN THAT IS NOT ON THE
PARETO FRONTIER.**

Yield over the **full 1710-frame sequence**, 1.18 M eligible keypoint-cases per cell;
time on the reference device, build and track summed, all spreads **0%**; bytes exact.

| ladder | filter | build+track | yield | bytes | vs shipped |
|---|---|---|---|---|---|
| `1/1/1/1` | `BOX_2x2` | 3 311 µs | 90.69% | 306 720 | −42.7% t, **−3.80 y** |
| `1/2/1/1` | `BOX_2x2` | 4 553 | 92.39% | 345 120 | −21.2% t, −2.10 y |
| `1/2/1/1` | `BOX_3x3` | 5 054 | 93.71% | 345 120 | −12.5% t, −0.78 y, −3.5% b |
| `1/2/2/1` | `BOX_2x2` | 4 849 | 93.80% | 354 720 | **−16.1% t**, −0.69 y |
| **`1/2/2/1`** | **`BOX_3x3`** | **5 642** | **94.97%** | **354 720** | **−2.4% t, +0.48 y, −0.8% b** |
| `1/2/2/2` | `BOX_2x2` | 5 778 | 94.49% | 357 600 | **(shipped)** |
| `1/2/2/2` | `BOX_3x3` | 6 005 | 95.27% | 357 600 | +3.9% t, +0.78 y |

**`1/2/2/1` + `BOX_3x3` is faster, more accurate AND smaller than the shipped point** —
all three axes, no trade. Every other arm is on the frontier; **only `1/2/2/2` +
`BOX_2x2` is dominated.**

**1. WHY THE SHIPPED POINT IS OFF THE FRONTIER, AND IT IS NOT AN ERROR IN D-23.** D-23
chose the ladder in 2 dimensions at a time when the third was priced differently: the
filter was fixed at `BOX_2x2` because `BOX_3x3` cost **+0.8 ms**, and
[X-42](#x-42--e-22--is-the-filter-frameworks-cost-genericity-or-structure--done) re-priced
it to **+0.35 ms** by removing a genericity tax nobody had looked for. **The decision
was right on the prices it had.** What moved is that a bit at level 3 and a wider
filter now buy accuracy at different rates than they did, and the swap — spend the
level-3 bit, buy the better filter — is *free on time and bytes and positive on yield*.

**2. THE COUPLING X-39 PREDICTED IS VISIBLE IN THE TABLE.** `BOX_3x3` is worth **+1.32
points at `1/2/1/1`**, **+1.17 at `1/2/2/1`**, **+0.78 at `1/2/2/2`** and **−0.02 at
`1/1/1/1`**. The better filter pays *more* the shallower the ladder, and **nothing at
all** when every level is 1 bit — because a 1-bit level cannot represent the smoother
result. **Filter and depth are substitutes over part of the range**, which is exactly
why pricing them on separate axes produced a dominated point.

**3. EVERY COARSE LEVEL'S BITS DO MATTER — E-19's OPEN SUB-QUESTION, ANSWERED.**
`1/2/1/1` loses **2.10** points and `1/2/2/1` **0.69** against `1/2/2/2` at the same
filter. Both exceed the 0.5-point bar, so **no level's second bit is free**, and
`1/2/2/2`'s shape was right even though its point is dominated. The gain from
`BOX_3x3` is what pays for dropping level 3's bit, not the bit being redundant.

**4. A METHODOLOGY FINDING THAT NEARLY INVERTED THIS RESULT.** The depth benchmark
seeded level 0 from a **synthetic lattice** (`(x*7+y*13)%29==0 || (x+y)%37==0`). LK's
cost is dominated by **iteration count**, not per-iteration work, and on a lattice the
coarse levels alias into false minima. Measured that way:

| | lattice seed | **real frame** |
|---|---|---|
| `1/1/1/1` track | 1.00× | 1.00× |
| `1/2/1/1` | **0.61× — FASTER with more bits** | 1.38× |
| `1/2/2/1` | 1.19× | 1.46× |
| `1/2/2/2` | 1.34× | 1.77× |

The lattice column is **non-monotonic** and would have made `1/2/1/1` look like a free
win. The benchmark now seeds from `benchmark/realframe.bin`, the real binarized frame,
and **refuses to fall back to a synthetic pattern** if it cannot read it. The build
column never had this problem — it is per-pixel and has no convergence behaviour.
**A benchmark whose arms differ in convergence needs content whose convergence is
real.**

**5. AND E-19's OWN COST FIGURE WAS INFLATED.** The register says the ladder costs
**2.30×**; measured here on real content it is **1.77×** on track and **1.50×** on
build. Part of that is the x86-era measurement D-35 corrected; part is the lattice.

**Decision — BAND A: `1/2/2/1` + `BOX_3x3` is the new operating point**, and D-23 is
superseded on evidence it could not have had. Recorded as
[D-43](ARCHITECTURE.md#8-design-decisions). **The frontier ships as documented operating
points**, since `1/2/2/1` + `BOX_2x2` at **−16.1% time for −0.69 yield points** is a
trade a footprint- or power-bound caller may well want, and `1/1/1/1` at −42.7% time
and −14.2% bytes remains the floor.

**NOT YET DONE, AND NAMED RATHER THAN IMPLIED:** the switch itself. Changing the
shipped ladder re-bases **every** performance number in this project, exactly as the
`pyrDown` swap did, so it needs the same treatment
[X-49](#x-49--the-frontend-after-the-api-swap-a-control-and-a-new-headline--done) gave
that one — a frontend re-measure that confirms accuracy is unchanged and re-states
criterion 4 — before the records can be updated. **This entry establishes the
operating point; it does not claim the frontend has moved.**

---

### X-51 · The frontend REFUTES X-50, and the accuracy harness is why · `DONE`

**A CONFIRMING RUN THAT DID NOT CONFIRM.** [X-50](#x-50--e-19--is-1222--box_2x2-still-the-operating-point--done)
concluded `1/2/2/1` + `BOX_3x3` dominates the shipped point on all three axes and
[D-43](ARCHITECTURE.md#8-design-decisions) recorded it. X-50 also said, in its own
words, that the switch *"does not claim the frontend has moved"* and needed
[X-49](#x-49--the-frontend-after-the-api-swap-a-control-and-a-new-headline--done)'s
treatment first. **That measurement was run and it refutes the conclusion.**

Full 1710-frame sequence, reference device, OpenCV at one thread:

| config | within 1 px | lifetime | p90 / p99 | speed | build | track |
|---|---|---|---|---|---|---|
| **`1/2/2/2` + `BOX_2x2`** *(shipped)* | **95.4%** | **11** | 0.161 / 22.5 | **10.644 ms** | 2.805 | 7.270 |
| `1/2/2/2` + `BOX_3x3` | 95.2% | 11 | 0.218 / 25.8 | 10.879 | 3.370 | 6.926 |
| **`1/2/2/1` + `BOX_3x3`** *(X-50's winner)* | **90.6%** | **9** | 0.648 / 50.6 | 10.787 | 3.371 | 6.753 |

**BOTH of X-50's accuracy claims fail, in the same direction.**

| | X-50's harness said | the frontend says |
|---|---|---|
| `BOX_3x3` at `1/2/2/2` | **+0.78** yield points | **−0.2** points, and **+0.235 ms** |
| dropping level 3's bit | −0.69 points | **−4.6 points**, lifetime **11 → 9** |

**THE MECHANISM, AND IT IS A DEFECT IN THE HARNESS RATHER THAN NOISE.**
`seedFiltered` builds the pyramid **entirely in floating point** — `p = downOnce(p, f)`
cascading on `CV_32F` — and quantizes **each level from the float chain**
(`tests/test_opticalflow.cpp`). binCV's real pyramid quantizes level 1 to N₁ bits,
then filters **that quantized level** to make level 2, and so on. **The harness models
a pyramid with no cascaded quantization error; the shipped one has three rounds of it.**

That explains the direction exactly: the harness **systematically understates the cost
of taking bits away**, because in the harness a coarse level is a fresh quantization of
an exact float, while in the pipeline it is a quantization of a quantization of a
quantization. At `1/2/2/1` the coarsest level is 1 bit at the end of that chain, and
the real loss is **6.7× larger** than the harness predicted.

**WHAT THIS DOES AND DOES NOT INVALIDATE.**

- **D-43 is WITHDRAWN.** `1/2/2/2` + `BOX_2x2` is the operating point;
  [D-23](ARCHITECTURE.md#8-design-decisions) stands, now on a frontend measurement
  rather than on X-50's proxy. The frontend is reverted to the exact file X-49
  measured, verified by diff rather than re-run.
- **E-19 is still answered**, but the answer is the opposite of X-50's: the shipped
  point is **not** dominated. Every coarse level's second bit earns its place *by more
  than X-50 could see*.
- **[X-39](#x-39-sequence-arm--the-same-design-space-over-1710-frames--done)'s accuracy
  axis rests on the same harness**, so [D-36](ARCHITECTURE.md) and
  [D-39](ARCHITECTURE.md)'s accuracy figures — including *"`BOX_3x3` is −0.10 from the
  Gaussian anchor"* — describe the **idealised** chain, not binCV's. Their *relative*
  filter comparisons at fixed ladder are less affected, because the idealisation is
  symmetric across filters; their absolute yields are not binCV's yields. **Flagged on
  those records, not quietly left.** The speed axes of D-36/D-39 are unaffected —
  those were measured on the real kernels.
- **X-50's speed and footprint tables are unaffected**: they were measured on the real
  kernels and reproduce here (track *did* fall, 7.270 → 6.753). What failed is the
  accuracy proxy, and it failed hard enough to invert a three-axis dominance claim.

**AND ONE COST ESTIMATE WAS LOW.** `BOX_3x3` was priced at **+0.35 ms** on build from
X-42/X-48 scaling; measured in place it is **+0.565 ms** (2.805 → 3.370). The scaling
used a 640×480 single-level number against a 752×480 three-transition two-frame build.
Another argument for measuring in place rather than scaling.

**Registered as [E-27](ARCHITECTURE.md#register): make the accuracy harness build its
levels with binCV's own `pyrDownFiltered` cascade**, so that ladder and filter accuracy
are measured on the pipeline that ships. Until then, **no accuracy conclusion from that
harness should be promoted to a shipped default without a frontend confirmation** —
which is the rule X-50 followed and is the only reason this was caught.

**Method:** `benchmark/frontend_sequence.cpp` via `scripts/run_on_pi.sh pi4` with
`BINCV_PI_OPENCV=1`, three configurations over the full sequence.

---

### X-52 · binCV on x86: the whole deficit is one stage · `DONE`

**A CHARACTERISATION, not a decision — no pre-registered rule.** It measures where the
library stands on a platform it does not target, which is where most people will first
judge it.

**Question.** Every criterion-4 reading before [X-37](#x-37--bincv-against-opencv-on-the-deployment-target-simd-against-simd--done)
was an x86 fact, and the last one recorded was **21.43 vs 1.54 ms — 13.9× slower**.
Since then D-31, D-32, X-35 and X-42 landed, and **most of those are
platform-independent algorithm changes** — only the accumulators (D-33, X-40) are NEON
intrinsics. **Where is x86 now, and what is left there?**

**Workload:** full 1710-frame sequence, x86_64 desktop core, OpenCV pinned to one thread;
the same binary and the same content as [X-49](#x-49--the-frontend-after-the-api-swap-a-control-and-a-new-headline--done)'s
aarch64 run.

**THE RESULT, AGAINST THE DENOMINATOR** ([CLAUDE.md](../CLAUDE.md): OpenCV doing the
same semantic operation) — this is the claim, and the only one:

| platform | binCV | OpenCV | |
|---|---|---|---|
| aarch64 ([X-49](#x-49--the-frontend-after-the-api-swap-a-control-and-a-new-headline--done)) | 10.644 ms | 16.289 | **1.53× FASTER** |
| **x86** | **14.155 ms** | **3.961** | **3.57× SLOWER** |

Much better than the 13.9× last recorded, and still a real gap.

**1. THE DEFICIT IS ONE STAGE — and OpenCV is what makes that legible.** A
binCV-on-x86 against binCV-on-Pi comparison would mix machine speed with
implementation quality and claim nothing about the product. **What makes the
cross-platform figures diagnostic is that OpenCV is the control for how much faster
the machine is:**

| | aarch64 | x86 | scaling |
|---|---|---|---|
| **OpenCV total — the control** | 16.289 ms | 3.961 | **4.11×** |
| binCV detect | 0.570 | 0.223 | 2.56× |
| binCV build | 2.805 | 0.984 | 2.85× |
| **binCV track (LK)** | **7.270** | **12.948** | **0.56× — INVERTED** |

Everything scales in the machine's direction except **LK**, which goes backwards. LK is
the only stage with a NEON-only fast path, and on x86 that path compiles out to scalar.
**The platform-independent optimisations transferred for free; the vector ones did not
transfer at all**, and measuring each stage against how OpenCV itself moved separates
the two without asserting anything about one machine versus another.

**2. LK IS 91.5% OF THE x86 FRONTEND** against 68.3% on aarch64 — not because LK got
worse, but because everything around it got faster and it did not.

**3. THE HYPOTHESIS THIS SETS UP, STATED AS A HYPOTHESIS.** If an AVX2 port of the tap
batching (D-33, X-40) bought x86 what NEON bought aarch64, LK would scale in the
machine's direction like its neighbours rather than against it — landing near
**~2.7 ms**, and the frontend near **3.9 ms against OpenCV's 3.961: parity.** **That is an extrapolation and this project
has been wrong three times tonight extrapolating** (the `BOX_3x3` build cost, X-43's
fabricated buffer, X-50's proxy). It is written down as a target to measure against,
not a result.

**4. WHAT IS NOT AT STAKE.** The **footprint** claim is platform-independent and
reproduces exactly — **6.23×**, identical bytes on both machines. [X-46](#x-46--where-does-bit-slicing-stop-paying--done)'s
bit-width crossover is a property of the representation. **Only the speed claim is
aarch64-only**, and ROADMAP and D-35 already say "on the reference device" — this entry
puts a number on what that qualifier is worth.

**Decision:** none. It scopes the x86 work: port the accumulators, expect LK to be the
whole of it, and measure rather than assume the rest.

**Method:** `benchmark/frontend_sequence.cpp`, run directly on the development machine
with `BINCV_OPENCV_THREADS=1`.

---

### X-53 · E-27 — the harness now measures the shipped pipeline, and it STILL does not predict the frontend · `DONE`

**THIS ENTRY REFUTES [X-51](#x-51--the-frontend-refutes-x-50-and-the-accuracy-harness-is-why--done)'s
OWN DIAGNOSIS.** X-51 traced the frontend's disagreement with X-50 to the accuracy
harness building its pyramid in floating point. E-27 fixed exactly that — `seedFiltered`
now runs **binCV's own `pyrDownFiltered` cascade**, quantizing each level from the
quantized level above it, which is the pipeline that ships. **The fix is right and the
diagnosis was wrong.**

**THE TEST, and it fails.** Level 3's second bit, `1/2/2/1` against `1/2/2/2` at
`BOX_3x3`:

| | says |
|---|---|
| old float-cascade harness | **−0.30** points |
| **corrected harness** | **−0.42** points |
| **frontend** ([X-51](#x-51--the-frontend-refutes-x-50-and-the-accuracy-harness-is-why--done)) | **−4.60** points |

**Removing the cascade moved the number by 0.12 where the gap is 4.2.** The float
cascade was not the cause, or was a small part of it.

**AND THE CORRECTIONS RUN IN BOTH DIRECTIONS**, which the proposed mechanism does not
predict. X-51 argued the float harness *understates the cost of removing bits*, so
correcting it should push low-bit ladders **down**. Full sequence, 1.18 M
keypoint-cases per cell:

| ladder | filter | corrected | old harness | Δ |
|---|---|---|---|---|
| `1/1/1/1` | `BOX_2x2` | 92.58% | 90.69% | **+1.89** |
| `1/1/1/1` | `BOX_3x3` | 90.73% | 90.67% | +0.06 |
| `1/2/1/1` | `BOX_2x2` | 93.15% | 92.39% | +0.76 |
| `1/2/1/1` | `BOX_3x3` | 93.39% | 93.71% | −0.32 |
| `1/2/2/1` | `BOX_2x2` | 94.32% | 93.80% | +0.52 |
| `1/2/2/1` | `BOX_3x3` | 94.84% | 94.97% | −0.13 |
| `1/2/2/2` | `BOX_2x2` | 94.72% | 94.49% | +0.23 |
| `1/2/2/2` | `BOX_3x3` | 95.26% | 95.27% | −0.01 |

**`1/1/1/1` — the ladder with the MOST cascaded quantization — moved UP by 1.89
points.** Under X-51's mechanism it should have moved down hardest. **The mechanism is
withdrawn.**

**THE FILTER AXIS BARELY MOVED, WHICH VINDICATES ONE HEDGE.** X-51 flagged D-36/D-39's
filter figures as resting on the idealised chain but predicted that *relative
comparisons between filters at a fixed ladder* would be least affected, "since the
idealisation is symmetric across filters." Measured against the `GAUSSIAN_5x5` anchor
at N=3:

| filter | corrected | old |
|---|---|---|
| `GAUSSIAN_3x3` | −0.37 | −0.37 |
| `BOX_3x3` | **−0.09** | −0.10 |
| `BOX_2x2` | −1.10 | −1.26 |
| `DIRECT_SUBSAMPLE` | −12.65 | −12.65 |

**Nothing moved by more than 0.16 points.** D-36 and D-39's filter rankings stand, now
on the shipped pipeline rather than a proxy, and their flags can be narrowed from "the
accuracy figures are suspect" to "the ladder figures were, the filter figures were not."

**SO WHAT DOES EXPLAIN THE FRONTEND?** Not measured here, and therefore not claimed.
The leading candidate is structural rather than numerical: **the harness warps a single
frame and asks whether LK can recover a known warp**, so `prev` and `next` are
binarizations of *the same image* and their edge maps are nearly identical. The
frontend tracks **real consecutive frames**, whose binarizations differ wherever a
pixel sits near the threshold, across a sequence where errors compound and tracks are
re-detected. **Those are different questions**, and coarse-level quality plausibly
matters far more for the second — but "plausibly" is the honest word and this entry
stops there.

**The tension is fundamental, not a bug:** the harness uses synthetic warps *because
it needs ground truth*, and that is exactly what makes it unrepresentative.

**Decision: E-27's fix SHIPS** — the harness measuring the pipeline that ships is
strictly better than measuring a float idealisation, and the filter numbers are now
first-hand. **But E-27's PURPOSE is not achieved.** The rule tightens rather than
relaxes:

> **No accuracy conclusion from the synthetic-warp harness may be promoted to a
> shipped default, with or without a corrected cascade.** It answers a sensitivity
> question — can LK recover a known warp — not a tracking one. Frontend accuracy is
> measured at the frontend.

Recorded as [D-44](ARCHITECTURE.md#8-design-decisions); the open half is
[E-28](ARCHITECTURE.md#register).

**Method:** `tests/test_opticalflow.cpp` (`seedPyramid` replacing the float cascade,
`downOnce`/`quantizeInto` deleted); both sequence sweeps re-run over the full 1710
frames in 10 shards.

---

### X-54 · E-9 — should the word type vary down the pyramid? · `DONE`

**Gates:** [E-9](ARCHITECTURE.md#register), unscheduled since
[X-10](#x-10--e-2--default-word-width--done).

**Question.** X-10 measured `uint64_t` reducing **1.94× faster** and costing **+33%**
at 94×60 but **0%** at 640×480, so the right answer might not be one type.
[D-1](ARCHITECTURE.md) makes the width a per-object template parameter, so a per-level
choice costs no new machinery — only a decision.

**THE FOOTPRINT SIDE IS ALREADY SETTLED, BY ARITHMETIC RATHER THAN MEASUREMENT.** A row
costs `ceil(width/bits) × bits/8` bytes, so `uint64_t` is free wherever the row is a
multiple of 64 pixels wide and costs at most one word otherwise. Over the shipped
frontend ladder:

| level | `uint32_t` | `uint64_t` | |
|---|---|---|---|
| L0 752×480 | 46 080 B | 46 080 | **+0.0%** |
| L1 376×240 | 23 040 | 23 040 | **+0.0%** |
| L2 188×120 | 5 760 | 5 760 | **+0.0%** |
| L3 94×60 | 1 440 | 1 920 | +33.3% |
| **whole ladder** | **76 320** | **76 800** | **+0.6%** |

**So this is a speed question**, and X-10's +33% headline is a property of the smallest
level rather than of the pyramid.

**A COLLISION THIS RULE NAMES BEFORE MEASURING, because it may decide the answer on its
own.** Every NEON path in the tracker is guarded on **`sizeof(WordType) == 4`** —
`slicedSignedSum`'s plane-pair batching, `alignedResidualSumsNeon1` (D-33) and
`alignedResidualSumsNeon2` (X-40). **A `uint64_t` level would silently fall back to
scalar for the whole of LK.** That is [D-1](ARCHITECTURE.md)'s genericity in the word
type colliding with D-33's specialisation at one width, and it is visible by reading the
guards — but *how much it costs* is not, so it is measured.

**Arms** (reference device, real-frame seed, build and track separately, per
[X-50](#x-50--e-19--is-1222--box_2x2-still-the-operating-point--done)'s method note):
the shipped `1/2/2/2` ladder at `uint32_t` and at `uint64_t`.

**DECISION RULE, WRITTEN BEFORE MEASURING.**

- **Band A — `uint64_t` is faster end to end** (build + track) despite losing NEON.
  Then the reduction win outweighs the specialisation, and a per-level choice is worth
  designing: `uint64_t` at L0–L2 where it is byte-free, `uint32_t` at L3.
- **Band B — `uint64_t` wins on build but loses on track.** The per-level answer is
  real but the *kernels that walk several levels* would need two instantiations, which
  E-9 itself names as the cost. Report the split and leave the decision to a caller
  who knows which stage dominates their pipeline.
- **Band C — `uint64_t` loses overall.** The NEON guards decide it. **Answer NO**, and
  record the collision as the reason: binCV's word-type genericity is real in the API
  and *not* real in the tracker's fast path, which is a fact about the library worth
  stating plainly rather than leaving implicit in three `if constexpr`s.
- **Band D — the two are within 5%.** Then the guards are not costing what they appear
  to, which would mean the NEON paths are worth less than D-33 and X-40 measured, and
  that contradiction is the finding.

**Method:** `benchmark/pyramid_depth_benchmark.cpp`, word-type arms added beside the
ladder arms; `scripts/run_on_pi.sh pi4`.

**RESULT — BAND B. THE SAME LIBRARY IS 1.66× FASTER AND 1.32× SLOWER AT THE SAME TIME,
DEPENDING ON WHICH KERNEL YOU ASK.**

Reference device, real-frame seed, spreads 0%:

| `1/2/2/2` | build | track | bytes |
|---|---|---|---|
| `uint32_t` | 424.5 µs | 4 838.9 | 357 600 |
| **`uint64_t`** | **255.5 — 1.66× FASTER** | **6 368.2 — 1.32× SLOWER** | +2.0% |
| `1/1/1/1` `uint32_t` | 276.1 | 3 059.6 | 306 720 |
| `1/1/1/1` **`uint64_t`** | **173.4 — 1.59× faster** | 3 410.3 — 1.11× slower | +1.4% |

**1. THE SPLIT IS EXACTLY THE COLLISION THE RULE NAMED.** Build is word-parallel
kernels — `pyrDown` and the derivatives — and a wider word does strictly less work per
pixel, so it wins by **1.6×**, which is X-10's reduction result showing up in a real
stage. Track is `residualSums`, whose three NEON paths are guarded on
**`sizeof(WordType) == 4`**, so `uint64_t` runs it **fully scalar** and loses **1.32×**.
**[D-1](ARCHITECTURE.md)'s genericity in the word type is real in the API and not real
in the tracker's fast path**, and this is that stated as a number instead of left
implicit in three `if constexpr`s.

**2. AT binCV's OWN BALANCE, `uint64_t` LOSES — but the balance is the whole argument.**
This benchmark's build:track ratio is 0.088; the frontend's is **0.386**
([X-49](#x-49--the-frontend-after-the-api-swap-a-control-and-a-new-headline--done)), so
it under-weights build **4.4×**. Weighting by the frontend's shares — track 68.3%
costing +31.6%, build 26.3% saving 39.8% — gives **≈ +11% frontend time**. `uint64_t`
loses, **but by a margin that a differently-balanced pipeline would reverse.** That
arithmetic combines two measurements rather than being one, and after three
extrapolation failures tonight it is offered as a direction, not a number.

**3. THE FOOTPRINT OBJECTION IS DEAD EITHER WAY.** +2.0% on the shipped ladder, +1.4%
at `1/1/1/1` — X-10's "+33%" was the 94×60 level in isolation and does not survive
being weighed against the levels above it.

**Decision — BAND B: NO per-level word type, and the reason is the split rather than a
verdict.** binCV's frontend is track-dominated, so the stage that loses is the stage
that matters, and E-9's own named cost — kernels walking several levels needing two
instantiations — buys nothing here. **A build-dominated pipeline would want the
opposite**, and that is now a documented operating point rather than an unexamined
assumption. Recorded as [D-45](ARCHITECTURE.md#8-design-decisions).

**What would change the answer:** giving `residualSums` a `uint64_t` NEON path. The
guards are a specialisation gap, not a property of the ISA — aarch64 counts bits in a
128-bit register regardless of how the caller sliced them. Registered as
[E-29](ARCHITECTURE.md#register), and it is the same shape as the x86 work
([X-52](#x-52--bincv-on-x86-the-whole-deficit-is-one-stage--done)): one kernel,
missing one specialisation.

---

### X-55 · E-11 — is the window strategy already gated on `blockSize`? · `DONE`

**A CHARACTERISATION that turned into a stale-premise correction**, so no rule: nothing
is decided here that the code has not already decided.

**E-11 asked** whether `cornerMinEigenVal` should select its window strategy on
`blockSize`, because [X-18](#x-18--does-the-incremental-window-form-still-pay-inside-t37s-dense-sweep--done)
measured the incremental form **losing below blockSize 15** — 0.84× at 3, which is what
`seal_params.yaml` configures.

**X-18 REPRODUCES, ON HEAVILY CHANGED CODE, TO TWO DECIMAL PLACES.** Reference device,
within-run spreads 0.05–1.6%:

| blockSize | sliding ns/px | recompute | net | X-18 |
|---|---|---|---|---|
| **3** | 103.573 | **88.395** | **0.85×** | 0.84× |
| 7 | 144.621 | 140.169 | 0.97× | — |
| 15 | 254.948 | 281.635 | **1.10×** | 1.10× |
| 31 | 559.895 | 686.000 | **1.23×** | — |

The crossover sits **between 7 and 15**, and it is now measured twice, months apart,
across D-27…D-45 worth of change.

**BUT E-11's PREMISE IS STALE, AND THAT IS THE ANSWER.** The path the frontend
*actually* runs is not the one X-18 measured. Since [X-31](#x-31--corner-response-698x-bit-exact--and-d-27s-target-ordering-was-wrong--done),
`impl::cornerMinEigenValRow` — the streaming row form behind the response ring
([D-26](ARCHITECTURE.md), E-10) — **already dispatches on `blockSize == 3`** to a
bit-sliced box-sum path, and its own comment says why: *"blockSize 3 is
`seal_params.yaml`'s value and the whole frontend's."* **The gating E-11 asks for exists,
on the path that matters, and has since before E-11 was written.**

What X-18 and this entry measure is the **frame-map** `cornerMinEigenVal`, which keeps
an unconditional column-major slide. That is deliberate — a frame map slides *down
columns*, which is a different traversal from the row form — and **no measured path in
this project calls it**: the frontend uses the ring.

**Decision: E-11 answered YES, and it is already implemented where it counts.** The
frame-map API keeps its unconditional slide rather than growing a branch no measured
path exercises; the crossover is recorded in its docstring so a caller sweeping
`blockSize < 7` densely can choose. **Adding an unexercised branch to buy 1.17× on a
path nobody takes is churn, not optimisation.**

**A note on what this cost.** The first run for this entry was
`corner_streaming_benchmark`, which measures E-10's ring against the frame map — a
different question. Two benchmarks in this repo have "corner" in the name and answer
different things, which is worth knowing before reaching for one.

**Method:** `benchmark/corner_benchmark.cpp` via `scripts/run_on_pi.sh pi4`.

---

### X-56 · T4.3b — a real VIO frontend loop, and what it says about D-28 · `DONE`

**THE SUFFICIENCY CHECK [T4.3](TASKS.md) SPLIT OFF AND NEVER RAN.** Every end-to-end
measurement in this project — [X-28](#x-28--t43a--the-frontend-end-to-end-over-a-real-sequence--partial),
[X-38](#x-38--e-20--the-whole-frontend-against-opencv-on-the-deployment-target--done),
[X-49](#x-49--the-frontend-after-the-api-swap-a-control-and-a-new-headline--done) — runs
a **benchmark loop**: detect wholesale every N frames, track, compare against OpenCV. A
VIO frontend does something structurally different, and `examples/vio_frontend.cpp` is
that loop, modelled on HybVIO's (`src/tracker/optical_flow.cpp`,
`feature_detector_legacy.cpp`) which is what the SEAL paper's pipeline drives.

**What it exercises that a benchmark loop does not:** a **persistent track set** carried
across frames rather than a per-frame rematch; **culling** on LK status and on leaving
the frame (HybVIO's `FAILED_FLOW` / `FLOW_OUT_OF_RANGE`); and **topping up** by
detection only when the count falls below a target, with `applyMinDistance` against the
**survivors**. binCV has no mask parameter by design — `ops/corner.hpp` documents the
spacing filter as the route — and this is that route taken.

**RESULT 1 — THE KERNEL SET IS SUFFICIENT.** The loop runs 1710 frames with no gap
requiring an operation binCV lacks. Sensor stage (median + edge filter) in OpenCV,
because that is not binCV's claim — in SEAL it is dedicated hardware and binCV's domain
starts at the binary frame. Everything after is binCV.

**RESULT 2 — D-28's DUTY CYCLE IS A PROPERTY OF THE BENCHMARK'S POLICY, NOT OF A VIO
FRONTEND.** [D-28](ARCHITECTURE.md) corrected X-30 by measuring detection at a **4.8%
duty cycle** and concluded it was "uninteresting". That figure comes from
`frontend_sequence` re-detecting every N frames. A frontend that **maintains a target
feature count** detects far more often, and the profile inverts:

| | `frontend_sequence` | **VIO loop, top up below target** | **VIO loop, 60% hysteresis** |
|---|---|---|---|
| detections | 4.8% of frames | **91.0%** | 45.0% |
| **detect** | 0.570 ms | **18.070** | 7.831 |
| track (LK) | 7.270 | 12.062 | 10.658 |
| build | 2.805 | 1.509 | 1.507 |
| **binCV total** | 10.644 | **31.641** | **19.996** |
| mean live features | — | 153.3 | 133.9 |
| lifetime p50 / p90 | — | 4 / 49 | 5 / 51 |

**Detection is between 39% and 57% of the frontend here, against D-28's 4.8%.** It is
the largest stage under the aggressive policy and comparable to LK under the relaxed
one. **D-28 is not wrong — it measured what it measured — but its conclusion does not
transfer**, and every optimisation priority derived from that profile (D-28's own target
list, and the reason detection was left alone from X-31 onward) rests on a detection
policy nobody had written down.

**RESULT 3 — THE POLICY IS WORTH 1.58× AND IT IS NOT binCV's.** Moving the low-water
mark from 100% to 60% of target takes binCV from **31.6 to 20.0 ms** for a 13% drop in
mean live features and a **slightly better** lifetime (p50 4 → 5). **A caller tuning one
number outside the library moves the frontend more than any optimisation this project
has landed.** That belongs in the documentation, not in a kernel.

**RESULT 4 — A SIZING TRAP THE API'S OWN FLAG CAUGHT.** The first run truncated the NMS
pool on **all 299 detections**: `candidatesTruncated` was set every frame. The pool peak
is **70 831 survivors on a 752×480 frame — 19.6% of all pixels.** A binarized
min-eigenvalue map takes few distinct values, so enormous numbers of pixels tie and
survive non-maximum suppression. **Size the pool from `candidatesRanked`, not from
`maxCorners`** — truncation happens *before* the spacing filter, so the corners kept
are the first found rather than the strongest. The same trap produced a false binCV
finding once before, and the flag `ops/corner.hpp` added for it is what caught it both
times.

**Decision:** T4.3b's sufficiency question is **answered YES**. Recorded as
[D-46](ARCHITECTURE.md#8-design-decisions), together with the correction to D-28's
scope. The detection-policy sensitivity is registered as
[E-30](ARCHITECTURE.md#register), because 39–57% of the frontend is now the largest
unexamined term in it.

**Method:** `bincv-cpp/examples/vio_frontend.cpp` and `examples/vio_sweep.sh` via
`scripts/run_on_pi.sh pi4` with `BINCV_PI_OPENCV=1`, full 1710-frame sequence at two
detection policies.

---

### X-57 · The entire x86 deficit is a compile flag · `DONE`

**[X-52](#x-52--bincv-on-x86-the-whole-deficit-is-one-stage--done) SCOPED THIS WRONG,
AND THE CORRECTION IS THE RESULT.** X-52 found LK to be the whole x86 deficit and
concluded the fix was porting D-33/X-40's NEON tap batching to AVX2 — reasoning that
those wins were vector wins and had not transferred. **They are not vector wins on
x86, and no AVX2 code was needed.**

**The NEON batching exists because aarch64 has NO SCALAR POPCOUNT** — `CNT` lives in
the vector registers, so every scalar count pays `fmov` in and out (D-6). **x86-64 has
`POPCNT` as a scalar instruction.** So the trick that wins on aarch64 answers a problem
x86 does not have — *provided the instruction is actually emitted.*

**IT IS NOT.** Baseline x86-64 predates SSE4.2, so `POPCNT` is not in the default ISA
and `__builtin_popcountll` compiles to a **software fallback**. Measured directly:
**zero `popcnt` instructions in the shipped `frontend_sequence` binary.** binCV counts
bits for a living, and it was doing it in software.

**RESULT.** Full 1710-frame sequence, same machine, two runs of each build back to
back because the first pass showed OpenCV itself moving 27% between runs:

| build | binCV | OpenCV | ratio |
|---|---|---|---|
| default (portable) | 12.851 / 12.991 ms | 3.454 / 3.396 | **0.27× / 0.26×** |
| **`-mpopcnt`** | **3.398 / 3.507** | 3.108 / 3.185 | **0.91× / 0.91×** |

**One flag: binCV 12.92 → 3.45 ms, a 3.75× speedup, and from 3.8× slower than OpenCV
to 0.91× — near parity, at 6.23× less memory.** The stage profile also snaps to the
aarch64 shape: track 67.3% / build 26.7% / detect 6.0% against aarch64's 68.3 / 26.3 /
5.4. **The library was never mis-shaped on x86; it was mis-compiled.**

**X-52's EXTRAPOLATION WAS RIGHT ABOUT THE DESTINATION AND WRONG ABOUT THE ROUTE.** It
predicted LK near ~2.7 ms and the frontend near ~3.9 ms at parity, flagged as an
extrapolation after three failures. Measured: LK **2.307 ms**, frontend **3.43 ms**,
parity. **The number was right and the mechanism was wrong**, which is worth recording
as its own lesson — a correct prediction is not evidence of a correct model.

**WHAT SHIPS.** `BINCV_X86_POPCNT` is **ON by default**: binCV's x86 baseline is a
POPCNT-capable CPU. Nehalem (2008) and Barcelona (2007) both have it, so the minimum
excludes roughly nothing still in service — and **shipping a bit-counting library that
counts bits in software is a worse default than a 2008 minimum.** It can be turned OFF
for pre-SSE4.2 targets (some Bonnell-era Atom), at 3.75×. The configure summary prints
which side it is on, because a 3.75× factor should not be invisible.

**The principled fix is runtime dispatch** (ROADMAP 2.3), which this does not pre-empt.
It is genuinely awkward for `popcountWord`: it is an inline function in hot loops, so a
dispatch that defeats inlining could cost more than it saves. **Registered as
[E-31](ARCHITECTURE.md#register)**, with the note that the pragmatic alternative —
simply requiring a 2008-era instruction — is what several vision libraries already do
and deserves weighing against the machinery.

**Decision:** the **specific** port X-52 proposed — batching taps into lanes to dodge a
domain crossing — is **cancelled**, because x86 has no such crossing to dodge.
Recorded as [D-47](ARCHITECTURE.md#8-design-decisions).

**THAT IS NOT "x86 NEEDS NO VECTOR WORK", AND AN EARLIER DRAFT OF THIS ENTRY SAID SO.**
With the flag on, binCV is **3.429 ms against OpenCV's 3.150** — 9% short of parity —
and **binCV still has no x86 vector code at all** against an OpenCV whose LK and gftt
are hand-tuned AVX2. The asymmetry is stark and is the honest summary of where binCV
stands: **NEON paths exist, x86 paths do not.**

The headroom is concrete rather than hopeful: **binCV processes `uint32_t` words on a
machine with 256-bit registers — 32 bits per operation where AVX2 offers 256.** The
build stage (`pyrDown` + derivatives) is **0.915 ms, 26.7%** of the frontend and is
long contiguous loops over whole planes, which is exactly what a vector unit is for;
halving it alone is 13% and would flip the ratio. Registered as
[E-32](ARCHITECTURE.md#register).

**Method:** `benchmark/frontend_sequence.cpp` built twice from the same tree, once with
`-mpopcnt`; `objdump | grep popcnt` to confirm the instruction is present or absent
rather than inferring it from timings.

---

### X-58 · E-32 — x86 vector paths: how much is the compiler's, how much is ours? · `DONE`

**COMMITTED BEFORE ANY INTRINSIC IS WRITTEN**, because
[X-57](#x-57--the-entire-x86-deficit-is-a-compile-flag--done) has just shown this
project reaching for hand-written vector code when the answer was a build flag. That
mistake cost nothing only because the flag was measured first.

**Gates:** [E-32](ARCHITECTURE.md#register). binCV is **0.91× of OpenCV on x86** with
POPCNT on, and has **no x86 vector code at all** against an OpenCV whose LK and gftt
are hand-tuned AVX2.

**THE STRUCTURAL OPPORTUNITY, and why it is not the port X-52 proposed.** binCV's
kernels are loops over `WordType` words. **x86-64's baseline is SSE2 — 128 bits — so
GCC can already auto-vectorize those loops to 4 `uint32_t` lanes**; `-mavx2` would take
it to 8. **The question is therefore not "should binCV have AVX2 code" but "how much of
the available win does the compiler already take, and what is left that only hand-written
code can reach?"** Asking it the other way round is what X-52 did.

**Arms**, full 1710-frame sequence, each built from the same tree and each confirmed by
`objdump` to contain the instructions it claims:

| arm | flags | minimum CPU |
|---|---|---|
| **A** | today's default (`-mpopcnt`) | Nehalem 2008 |
| **B** | `+ -mavx2` | Haswell 2013 |
| **C** | `-march=native` | this machine only |

Reported per stage, because [X-52](#x-52--bincv-on-x86-the-whole-deficit-is-one-stage--done)
showed the frontend total hiding which stage moved: `build` is long contiguous plane
loops and should vectorise well; `track` is a 31-pixel window — **one word per row** —
and may not vectorise at all, which would be the finding.

**DECISION RULE, WRITTEN BEFORE MEASURING.**

- **Band A — B reaches or beats OpenCV (≤ 3.15 ms) and the gain is broad.** The
  compiler had it all along and **no intrinsics get written**. The decision becomes a
  baseline question — is Haswell-2013 an acceptable x86 minimum — which is the same
  shape as X-57's and is the caller's, not mine.
- **Band B — B closes some of the 9% but leaves `track` untouched.** That localises the
  hand-written work to exactly one kernel and is the outcome that most deserves
  intrinsics, because a 31-pixel window is precisely what a compiler cannot vectorise
  and a human can. Write them **for `residualSums` only**, and price them separately.
- **Band C — B changes little anywhere.** The loops are not auto-vectorisable as
  written — most likely aliasing or the bit-sliced plane indirection — and the finding
  is *why*, not a speedup. Report the obstruction before writing code around it.
- **Band D — C beats B substantially.** Something other than AVX2 is carrying it (BMI2,
  `tzcnt`, wider shifts). **Identify it before adopting `-march=native`**, which is not
  a shippable flag and would otherwise smuggle an unattributed win into the record.

**A LIMIT DECLARED IN ADVANCE.** Even a perfect result here makes binCV *competitive*
on x86, not dominant: the footprint claim (6.23×) is unaffected and universal, and
**x86 is not the deployment target** — CLAUDE.md targets Cortex-A/M. This work buys
evaluability and honesty, and no x86 number will be promoted to a headline claim.

**Method:** `benchmark/frontend_sequence.cpp` built three ways from one tree;
`objdump` to confirm the ISA actually present in each binary, as X-57 did, rather than
inferring it from timings.

**RESULT — BAND C. `-mavx2` CHANGES NOTHING, AND THE REASON IS THE REPRESENTATION.**

Two runs of each arm, all three confirmed by `objdump` (default: **0** `%ymm`; avx2 and
native: **228** and **227**):

| run | arm | detect | build | track | binCV | OpenCV |
|---|---|---|---|---|---|---|
| 1 | default | 0.223 | 1.059 | 2.668 | 3.951 | 3.873 |
| 1 | `-mavx2` | 0.216 | 1.049 | 2.629 | 3.894 | 3.960 |
| 1 | `native` | 0.209 | 0.966 | 2.523 | 3.698 | 3.766 |
| 2 | default | 0.237 | 0.990 | 2.473 | **3.699** | 3.704 |
| 2 | `-mavx2` | 0.198 | 0.996 | 2.470 | **3.664** | 3.750 |
| 2 | `native` | 0.214 | 0.972 | 2.515 | 3.701 | 3.751 |

**1–2%, inside the 6.8% run-to-run spread of the SAME binary.** Band D did not fire
either: `native` and `avx2` are indistinguishable, so nothing outside AVX2 was carrying
anything.

**WHICH KERNELS THE COMPILER VECTORISED, AND WHICH IT DID NOT:**

| kernel | `%ymm` instructions |
|---|---|
| `derivativeX/Y` | **63 — vectorised** |
| `pyrDownRoute` | **0** |
| `boxSum4` | **0** |
| `cornerMinEigenValRow` | **0** |
| `residualSums` | 2 |

**THE OBSTRUCTION, IN GCC'S OWN WORDS** (`-fopt-info-vec-missed` on a TU instantiating
`pyrDownBox`): *"not vectorized: **multiple nested loops**"* and *"not vectorized:
**control flow in loop**"*.

**THAT IS THE BIT-PLANE REPRESENTATION ITSELF.** A bit-sliced kernel's outer loop walks
**words** — the dimension a vector unit wants — and its **body is a nest over planes**,
plus the bounds tests `srcWord` needs. GCC will not vectorise an outer loop whose body
is a loop nest. **The one kernel that is a plain word loop, `derivative`, vectorises
without being asked.** So:

> **The KERNELS' loop order stops the compiler; the LAYOUT does not.** An earlier draft
> of this entry conflated them, which would have sent the next experiment at the
> representation instead of at the loop.

**THE LAYOUT IS ALREADY WHAT SIMD WANTS.** A plane row is `ptr + y * stride` with
consecutive words, so eight consecutive words are **32 contiguous bytes — one
`_mm256_loadu_si256`**. Nothing in the storage resists vectorisation; only the loop
order does.

**This is not a flag and it is not a drop-in intrinsic.** Reaching AVX2 means
**restructuring each kernel to process 8 destination words at once**, with the plane
arrays held as vectors rather than as `WordType[N]` locals — a redesign per kernel, and
**no change to the bytes on the heap**.

**The upside is that bit-sliced arithmetic is IDEALLY suited to it once restructured:**
`boxSum4`'s ripple adds are pure `AND`/`XOR`/`OR`, which AVX2 does **256 bits at a
time** against the current 32. Nothing about the maths resists vectorisation; only the
loop order does.

**Decision — Band C: no intrinsics on this evidence, and the finding is the
obstruction.** `-mavx2` is **not** adopted: it would raise the minimum CPU to Haswell
2013 and buy nothing measurable. The kernel-restructuring work is scoped and registered
as [E-33](ARCHITECTURE.md#register) rather than started at the end of a long session,
and it needs a ceiling before an arm — the discipline
[X-33](#x-33--the-ceiling-for-batched-neon-popcounts--done) established and X-52 forgot.

**Where binCV actually stands on x86:** **3.70 ms against OpenCV's 3.70** in the
cleaner run — **parity**, from POPCNT alone, at 6.23× less memory. Beating OpenCV needs
E-33.

**Method:** `benchmark/frontend_sequence.cpp` built three ways from one tree; `objdump`
per-symbol for the vectorisation audit; `-fopt-info-vec-missed` for GCC's reason.

---

### X-59 · E-33's ceiling: what AVX2 would buy a restructured kernel · `DONE`

**CEILING BEFORE ARM.** [X-58](#x-58--e-32--x86-vector-paths-how-much-is-the-compilers-how-much-is-ours--done)
localised the x86 obstruction to loop order and registered [E-33](ARCHITECTURE.md#register)
— restructure the kernels so AVX2 can reach them. That is a redesign per kernel, so it
gets a ceiling first: the discipline [X-33](#x-33--the-ceiling-for-batched-neon-popcounts--done)
established, [X-52](#x-52--bincv-on-x86-the-whole-deficit-is-one-stage--done) forgot,
and [X-57](#x-57--the-entire-x86-deficit-is-a-compile-flag--done) paid for.

**The two operations a restructured kernel rests on**, each against the scalar form
binCV runs today, each **bit-exact against it before being timed**:

| | scalar | AVX2 | ceiling |
|---|---|---|---|
| **adder** — `boxSum4`'s shape, three ripple adds over 1-bit planes | 26.3–30.6 µs | 5.8–6.6 | **≈ 4.7×** |
| **popcount** — what `residualSums` rests on | 14.0–16.2 | 1.8–2.1 | **≈ 7.9×** |

**1. THE ADDER CEILING IS ~4.7×, AND IT IS THE UNSURPRISING HALF.** Bit-sliced
arithmetic is pure `AND`/`XOR`/`OR`; AVX2 does 256 bits where binCV does 32. **8× the
width delivering 4.7× is load/store bound**, not compute bound — which is what a
restructured `pyrDown` would actually see.

**2. THE POPCOUNT CEILING IS ~7.9×, AND THAT ONE IS A SURPRISE WORTH STATING.** It beats
**hardware `POPCNT`** — the instruction [X-57](#x-57--the-entire-x86-deficit-is-a-compile-flag--done)
just enabled and which x86 has natively. Mula's `pshufb` nibble-table counts **32 bytes
per pass** against `POPCNT`'s 8, so the vector form wins despite the scalar one being a
single instruction. **This entry set out to check whether a win was even available here
and expected the answer to be no**; `residualSums` is 67% of the x86 frontend, which
makes it the opposite of a footnote.

**3. THE HONEST CAVEAT ON THE NUMBERS.** Within-run spreads are **45–165%** — this is a
noisy desktop, not the reference device, and the reference device is aarch64 so **x86
ceilings can only be measured here.** What survives that is the **ratio**: three
independent runs give 4.48 / 4.66 / 4.96 and 7.91 / 8.44 / 7.42. A 4.7× gap against a
~100% spread is robust — even at scalar's fastest batch it would still be several times
the vector median — but the absolute times are not quotable and are given as ranges.

**4. AND CEILINGS OVERSTATE — THIS PROJECT HAS MEASURED THAT THREE TIMES.**
[D-37](ARCHITECTURE.md): 1.461× ceiling, 1.069× delivered. [D-40](ARCHITECTURE.md):
1.638× ceiling from a fabricated buffer, 0.885× real. [D-41](ARCHITECTURE.md): 1.14×
overstatement from a buffer that differed only in where it lived. **These are shape
ceilings, not kernel forecasts**, and E-33 should expect materially less.

**Decision:** none — this is a ceiling. It says E-33 is **worth doing**: both of the
operations binCV's hot path rests on have real headroom, the arithmetic is ideally
suited to vectorisation, and only the loop order is in the way. It does **not** say how
much survives the restructuring.

**Method:** `benchmark/avx2_ceiling.cpp`, `__attribute__((target("avx2")))` so the
default build stays portable; equality checked on all 16 384 words and on the popcount
total before timing; three independent runs on an idle machine.

---

### X-60 · E-33 attempted on `residualSums`: written, bit-exact, and 1.88× SLOWER · `DONE`

**THE ARM X-59's CEILING AUTHORISED, BUILT AND MEASURED AND REVERTED.** X-59 priced a
batched `pshufb` popcount at **≈7.9× — beating hardware `POPCNT`** — on a 16 384-word
array. This applies it to `slicedSignedSum` at N = 2, the shape three of the four
shipped ladder levels run, mirroring the NEON path that already wins there.

**IT IS BIT-EXACT AND IT LOSES.** One interleaved run, both arms under identical
conditions, `Flow.ResidualNeonMatchesScalar_N2` reporting **AVX2 vs scalar, 728 windows,
0 differ**:

| arm | µs | vs scalar |
|---|---|---|
| **scalar** | **182.5** | 1.000× |
| **AVX2, inlined** | **344.0** | **0.53× — 1.88× SLOWER** |

**TWO SEPARATE FAILURES, AND THE FIRST ONE IS A TRAP WORTH PUBLISHING.**

**1. `__attribute__((target("avx2")))` BLOCKS INLINING.** The first version used it so
the portable build would still compile the path, with a cached
`__builtin_cpu_supports` guard. A `target` function **cannot be inlined into callers
compiled for a different target**, so `slicedSignedSum` — previously inlined into a
tight row loop — became **310 real calls per window** plus a store and reload of the
lane array. Confirmed by `objdump`: **20 call sites and a standalone symbol.** That
version measured **1.9× slower** on the frontend. **The mechanism E-31 proposes for
runtime dispatch is exactly this, and it is now measured as unusable for an inline hot
function** — which is the concrete evidence E-31 was registered without.

**2. FIXING THE INLINING DID NOT FIX THE RESULT.** Rebuilt with compile-time `__AVX2__`
and no `target` attribute, the path inlines — `nm` shows **zero standalone symbols** —
and it is **still 1.88× slower.** The reason is the one X-43 already found on NEON, in
a new costume:

> **The eight words are COMPUTED IN REGISTERS, not loaded from memory.** X-59's ceiling
> used `_mm256_loadu_si256` on a contiguous array. Here the vector must be *assembled*
> from eight scalar values (`_mm256_setr_epi32` is inserts, not a load) and then
> *disassembled* through a store and eight scalar reloads. **The pack and unpack cost
> more than the eight `POPCNT` instructions they replace.**

**THIS IS THE FIFTH CEILING IN THIS PROJECT TO OVERSTATE, AND THE PATTERN IS NOW THE
FINDING:**

| | ceiling | delivered |
|---|---|---|
| [X-33](#x-33--the-ceiling-for-batched-neon-popcounts--done) | 3.42× | 1.24× |
| [X-40](#x-40--e-18--window-carried-vector-accumulators-at-n--2--done) | 1.461× | 1.069× |
| [X-43](#x-43--e-24--can-the-extraction-be-vectorised-and-what-stops-it--done) | 1.638× | **0.885×** |
| [X-44](#x-44--e-26--is-interleaved-a-layout-bincv-should-support--done--declined) | 1.638× | 1.445× |
| **X-60** | **7.9×** | **0.53×** |

**Every one was measured on bulk contiguous data and applied to a kernel that works on
a handful of register-resident words.** That is not five unlucky estimates; it is one
structural mismatch measured five times. **binCV's hot kernels are not array
operations** — `residualSums` touches one word per row of a 31-row window — and **a
vector unit wants arrays.** The mismatch is **granularity, not layout**, which is a
sharper statement than [D-48](ARCHITECTURE.md)'s and supersedes the part of it that
implicated the representation.

**WHERE THIS LEAVES E-33.** Narrowed, not closed. `residualSums` — **67% of the x86
frontend** — is refuted for vectorisation at its current granularity. **`build` is
not**: `pyrDown` and the derivatives are genuine bulk passes over contiguous plane
rows, which is the shape X-59's adder ceiling (4.7×) was actually measured on, and
`derivative` already auto-vectorises. **The untested half is the half the ceiling
applies to**, and it is 27% of the frontend rather than 67%.

**Decision:** the `slicedSignedSum` AVX2 path is **reverted** — bit-exact but 1.88×
slower, and a 1.88× regression is not worth keeping for a platform binCV does not
target. E-33 stays open for `build` only. Recorded as
[D-49](ARCHITECTURE.md#8-design-decisions).

**A note on measurement conditions.** Mid-experiment the machine's load average reached
**8–10 from processes not mine**, and `frontend_sequence` reported OpenCV alone ranging
**3.7–10.9 ms** on identical work. Those numbers were discarded rather than reported.
The verdict above rests on `residual_n2`, which interleaves its arms in one run so both
see the same conditions — the only measurement shape that survives a loaded machine.

**Method:** `ops/opticalFlow.hpp` (added and reverted), `benchmark/residual_n2.cpp`,
`tests/test_opticalflow.cpp`'s `ResidualNeonMatchesScalar_N2` for exactness.

---

### X-61 · Batching across KEYPOINTS: the third granularity, and the third refutation · `DONE`

**THE ONE AXIS X-59 AND X-60 BOTH MISSED.** A window row is 31 pixels — **one `uint32`
word** — so binCV's packing, which can hold **256 pixels in an AVX2 register**, is
**8/9ths unused at that granularity, and no reshaping inside a row can recover it.**
But LK tracks **150–200 keypoints doing the identical computation on different
windows**, and eight of them fill a register exactly. That is the natural vector
dimension and neither prior attempt used it.

**THE ARITHMETIC THAT MOTIVATED IT.** binCV holds 31 pixels in a word where OpenCV's
AVX2 holds 32 per lane-row — no advantage — and then spends **~120 operations per
window row against OpenCV's ~18**: N² = 4 plane pairs × 5 taps ([D-20](ARCHITECTURE.md))
× 2 components × 2 for sign-magnitude ([D-3](ARCHITECTURE.md)). **An 8× packing
advantage divided by a ~6.7× op-count disadvantage is ≈1.2×**, which is where binCV
actually sits. Batching eight keypoints would cash the packing advantage at the level
where it is real.

**RESULT — PARITY. 1.00× / 0.99× / 1.11× over three runs**, bit-exact against the
scalar arm:

| arm | ns | vs scalar |
|---|---|---|
| A — 8 keypoints, sequential, scalar `POPCNT` | 1 261–1 512 | 1.00× |
| B — 8 keypoints in 8 lanes, gathered | 1 134–1 535 | **0.99–1.11×** |

**THE GATHER IS THE BLOCKER, AND THE ACCOUNTING IS SIMPLE.** Five
`_mm256_i32gather_epi32` per row at ~15 cycles each on this core — gathers are
microcoded — is **~75 cycles, against the ~40 single-cycle loads the scalar arm
issues for all eight keypoints.** The vector arithmetic *does* win (≈48 vector ops
against ≈100 scalar), and the gathers give it straight back. **X-43 found gathering a
net loss when it fed one popcount; here it feeds sixteen per row and is still a net
loss.**

**THREE ATTEMPTS, THREE MEASURED REFUTATIONS, THREE DIFFERENT REASONS:**

| attempt | granularity | why it failed |
|---|---|---|
| [X-59](#x-59--e-33s-ceiling-what-avx2-would-buy-a-restructured-kernel--done) ceiling | bulk contiguous array | **not the kernel's shape** — 7.9× that never applied |
| [X-60](#x-60--e-33-attempted-on-residualsums-written-bit-exact-and-188-slower--done) | within one window row | values are **register-resident**; pack/unpack > 8 `POPCNT`s |
| **X-61** | **across 8 keypoints** | words are **scattered**; gather > the loads it replaces |

**The common thread is now unmistakable: `residualSums` reads SCATTERED SINGLE WORDS,
and SIMD needs CONTIGUOUS RUNS.** That is a property of the **access pattern**, not of
the bit-plane layout ([D-48](ARCHITECTURE.md) corrected) and not of the granularity
alone ([D-49](ARCHITECTURE.md) refined). Making those words contiguous is exactly what
[E-26](ARCHITECTURE.md#register) priced — **+21% footprint, declined.**

**WHAT THIS DOES NOT SAY.** It does not say binCV cannot beat OpenCV. binCV **is**
1.53× faster on aarch64, the deployment target, where the same scattered access is a
*win* because `CNT` is cheap and OpenCV's NEON coverage is thinner. And **`build`
remains untested and is the one shape that should work** — `pyrDown` and the
derivatives are bulk contiguous passes, exactly what X-59's 4.7× adder ceiling was
measured on, and `derivative` already auto-vectorises unaided. That is 27% of the x86
frontend and the only untried avenue with a favourable prior.

**Decision:** `residualSums` is **closed for x86 vectorisation** at every granularity
tried, and the reason is recorded per attempt rather than as a single verdict.
[D-50](ARCHITECTURE.md#8-design-decisions). E-33 narrows to `build` alone.

**Method:** `benchmark/kpbatch_ceiling.cpp`, `-mavx2`, equality checked before timing;
three runs, spreads 33–143% on a loaded machine, which is why the verdict rests on the
ratio being ~1.0 across all three rather than on any single figure.

---

### X-62 · E-34 — can the four tap correlations become one? · `DONE`

**COMMITTED BEFORE ANY KERNEL CHANGE, AND WITH A CEILING BEFORE THE ARM.** Five
ceilings in this project have overstated ([D-49](ARCHITECTURE.md#8-design-decisions)),
and the last three x86 attempts each cost a working implementation that was then
reverted. This one gets priced first.

**Gates:** [E-34](ARCHITECTURE.md#register), the largest of the three multipliers
[D-51](ARCHITECTURE.md#8-design-decisions) identified, in the kernel that is **67% of
the frontend**.

**THE ACCOUNT THIS COMES FROM.** binCV's LK issues **5.6× OpenCV's operations** per
window per iteration — 3 720 against 660 — and holds a **2× packing advantage**
(32 px/op at N = 2 in `uint32_t`, against OpenCV's 16 at `CV_16S`), so it costs **2.8×**.
Of the three multipliers, **N² is a property of bit-sliced arithmetic** and **×2 is
sign-magnitude ([D-3](ARCHITECTURE.md))**. **Only the ×5 is a design choice**, and
[D-20](ARCHITECTURE.md) records it as a consequence of bits not being interpolable
rather than as an option that was weighed.

**THE IDEA.** Correlation is linear, so
`Σ(w₀₀t₀₀ + w₀₁t₀₁ + w₁₀t₁₀ + w₁₁t₁₁)·G` may be computed as **one** weighted patch
correlated **once**, instead of four correlations combined afterwards — *if* the
bilinear weights are **quantised** so the weighted sum can be formed in bit-sliced
arithmetic. That is `weightedAxis`'s existing machinery (`ops/pyramid.hpp`), applied to
taps instead of filter positions.

**THE COLLAPSE IS NOT FREE, AND THE CEILING EXISTS TO PRICE THAT.** Forming the
weighted patch costs bit-sliced adds over `N + k` planes plus a requantise back to N
bits, once per row — but **shared across both gradient components**, where the four
correlations are paid per component. Rough account: **~120 ops/row → ~60**, i.e. **~2×**,
not the ×5 the tap count alone suggests.

**ARMS** (`benchmark/tapcollapse_ceiling.cpp`, reference device):

| | |
|---|---|
| **A** | today: four correlations, combined with float weights afterwards |
| **B** | quantised weighted patch formed once, then **one** correlation |

Both compute the same residual sums; **B is an approximation of A, not an identity**,
so the arms are compared for *closeness*, not equality — which is the opposite of every
other ceiling in this project and is why the accuracy band below is mandatory.

**DECISION RULE, WRITTEN BEFORE MEASURING.**

- **Band A — B is ≥1.6× faster AND the weight quantisation costs <0.02 px** of
  `rms(usable)` at k = 3 bits. Write the arm. **Then measure accuracy on the sequence
  before it ships** — [X-51](#x-51--the-frontend-refutes-x-50-and-the-accuracy-harness-is-why--done)
  is the standing reason a proxy does not settle a shipped default.
- **Band B — ≥1.6× faster but accuracy costs more.** The trade is real and is the
  caller's, as [D-24](ARCHITECTURE.md) put route (a) and [D-36](ARCHITECTURE.md) put the
  filter set. Report both axes; do not change the default.
- **Band C — <1.6×.** The weighted-sum construction eats the tap saving. **Do not write
  it**, and record where the ops went — that is the useful half, because it would mean
  bit-sliced weighted sums are too expensive to be worth reaching for elsewhere.
- **Band D — B is SLOWER.** Then `weightedAxis` at this shape costs more than four
  popcount correlations, which contradicts the arithmetic above and means the op-count
  model in D-51 is wrong. **Report the model failure, not the timing.**

**A LIMIT DECLARED IN ADVANCE.** Even Band A leaves binCV at roughly **1.4× OpenCV's
cost on LK**, not ahead of it — the N² and sign-magnitude factors remain. What it would
buy is the **frontend**, where LK is 67%: a 2× on LK is roughly **1.5× on the frontend**
against OpenCV on x86, and correspondingly more on aarch64 where binCV already leads.
**No claim beyond that will be made from this experiment.**

---

#### THE RESULT: THE SPEED BAND FIRED, AND THE FRONTEND REFUTED IT ANYWAY

**THE CEILING.** Reference device, three runs, `benchmark/tapcollapse_ceiling.cpp`:

| arm | ns / 20 windows | vs A |
|---|---|---|
| **A** shipped: 4 correlations/component | 58 311 – 59 666 | 1.000× |
| **B** k=3, compile-time weights, generic offset | 30 664 – 32 819 | **1.82 – 1.91×** |
| **B** k=3, worst offset `(3,3,1,1)` | 33 161 – 34 830 | 1.71 – 1.76× |
| **B** k=3, **RUNTIME weights — the implementable one** | 33 115 – 33 568 | **1.75 – 1.76×** |

**Band A or B on speed, decisively.** The runtime-weight arm — the only shape a
real kernel can have, since the subpixel offset moves every iteration — is within
7% of the compile-time ceiling. **It was priced rather than assumed**, which is
the correction [D-49](ARCHITECTURE.md#8-design-decisions) demanded.

**THE ACCURACY AXIS,** `Flow.X62_TapCollapseSequence_uint32_t`, 93 frames × 6
warps, shipped `1/2/2/2` ladder and `BOX_2x2`, `tapCollapseBits` the only
difference:

| case | yield A | yield B | rms A | rms B | Δrms |
|---|---|---|---|---|---|
| shift (1, 0) | 99.27% | 99.26% | 0.0007 | 0.2131 | +0.2124 |
| shift (0.25,0.25) | 96.85% | 97.11% | 0.2131 | 0.2975 | +0.0844 |
| shift (0.75,0.75) | 96.11% | 96.34% | 0.2024 | 0.2918 | +0.0894 |
| shift (2, -3) | 99.31% | 99.26% | 0.0009 | 0.2117 | +0.2108 |
| shift (6, 4) | 99.42% | 99.46% | 0.0003 | 0.2080 | +0.2077 |
| rotate 1 deg | 97.49% | 97.65% | 0.2051 | 0.3053 | +0.1002 |
| **ALL** | **98.07%** | **98.18%** | **0.1455** | **0.2579** | **+0.1124** |

**+0.112 px, which is 5.6× the 0.02 px band — so Band B on the pre-registered
rule.** Note what is *not* hurt: **yield is unchanged** (98.07 → 98.18%). The
collapse costs sub-pixel precision, not track survival. And note the integer
shifts: the exact route returns **0.0007 px** there, because at a whole-pixel
displacement there is no interpolation to do — the collapse turns that into
0.21 px, which is the **floor** its requantise imposes at every offset.

#### AND THEN THE FRONTEND, WHICH NO BAND ANTICIPATED

`benchmark/frontend_sequence`, `track (LK)` ms/frame, 400 frames, both platforms:

| `maxIterations` | ref. device A | ref. device B | B vs A | x86 A | x86 B | B vs A |
|---|---|---|---|---|---|---|
| 1 | 2.764 | 2.601 | **1.06×** | 0.740 | 0.683 | **1.08×** |
| 2 | 4.209 | 3.970 | 1.06× | 1.160 | 1.071 | 1.08× |
| 3 | 5.202 | 5.422 | 0.96× | 1.509 | 1.534 | 0.98× |
| 5 | 5.827 | 7.572 | 0.77× | 2.218 | 2.431 | 0.91× |
| 10 | 7.022 | 13.757 | 0.51× | 2.580 | 4.514 | 0.57× |
| **20 (shipped)** | **7.459** | **24.409** | **0.31×** | **2.766** | **6.736** | **0.41×** |

**AT THE SHIPPED ITERATION CAP THE COLLAPSE IS 3.3× SLOWER ON THE REFERENCE
DEVICE.** The crossover is between 2 and 3 iterations, on both platforms
independently.

**THE MECHANISM IS IN THE SHAPE OF THE TWO CURVES.** A's cost *flattens* —
5.83 → 7.02 → 7.46 across a 4× rise in the cap — because its points **converge and
stop**. B's keeps climbing linearly — 7.57 → 13.76 → 24.41 — because its points
**never converge**, so every one of them runs to the cap.

> **ROUNDING THE INTERPOLATED PATCH BACK INTO THE PIXELS' OWN N-BIT ALPHABET
> DESTROYS EXACTLY THE SIGNAL LK CONVERGES ON.** Sub-pixel displacement lives in
> the interpolation; that is what [D-20](ARCHITECTURE.md)'s five integer sums keep
> at full precision. Quantise it away and `delta` never falls under `eps`, so the
> per-iteration saving is spent several times over on iterations that would not
> otherwise have happened.

#### WHAT WAS WRONG WITH THE RULE, WHICH IS THE PART WORTH KEEPING

**None of the four pre-registered bands can express this result**, and forcing it
into one would be fitting the conclusion to the numbers. Every band was a
statement about a **per-iteration ratio** — Band D is even phrased as "B is
SLOWER", which B is *not*: at a fixed iteration count it is 1.06–1.08× faster on
the frontend and 1.75× faster in the kernel, exactly as the op-count model said.
**[D-51](ARCHITECTURE.md#8-design-decisions)'s model was right. The ceiling's
SCOPE was wrong.**

This is a new failure mode for this project's ceilings, and it is worse than the
five [D-49](ARCHITECTURE.md#8-design-decisions) records. Those **mispriced the
operation** and a careful ceiling catches them — this one **priced the operation
correctly and mispriced the algorithm**, and no amount of care *inside* the
ceiling would have caught it. Promoted to
[D-53](ARCHITECTURE.md#8-design-decisions).

**Decision: E-34 CLOSED NEGATIVE. The arm is reverted; `LKParams` gains nothing.**
There is no regime worth offering a caller: at `maxIterations ≤ 2` the gain is
1.06×, inside run-to-run scatter, and at the shipped 20 it is a 3.3× loss. An
option that never wins is a trap, not a trade — unlike
[D-24](ARCHITECTURE.md)'s route (a) or [D-36](ARCHITECTURE.md)'s filter set, both
of which ship because each has a regime.

**The arm itself is recoverable, and deliberately so.** `benchmark/tapcollapse_ceiling.cpp`
stays in the tree and reproduces the 1.75×. The kernel arm, the `LKParams` field
and `Flow.X62_TapCollapseSequence_uint32_t` live in commit **`4351bd6`**, reverted
by the commit that carries this entry — `git revert` it to re-run the accuracy and
iteration tables above.

**AND ONE WARNING ABOUT THE MEASUREMENT ITSELF.** The first reference-device sweep
reported the two arms **identical at every iteration count**, which was a **stale
binary**: `frontend_sequence` needs OpenCV, `run_on_pi.sh` builds core-only unless
`BINCV_PI_OPENCV=1`, and the run silently executed a build predating the knob
(`strings … | grep -c BINCV_LK_TAPCOLLAPSE` → 0). It did not fail; it produced a
plausible null result. `benchmark/CMakeLists.txt` already warns that "a failed
reproduction then looks like a missing binary" — **a stale one looks like a
finding**, which is worse.

**Method:** `benchmark/tapcollapse_ceiling.cpp` via `scripts/run_on_pi.sh pi4`,
three runs; accuracy via `Flow.X62_TapCollapseSequence_uint32_t` on
`BINCV_X62_FRAMES` at stride 40 (93 frames); frontend via
`BINCV_LK_TAPCOLLAPSE` × `BINCV_LK_ITERS` on 400 frames, **`BINCV_PI_OPENCV=1`**.
Reference device: Cortex-A72, `performance` governor, `taskset -c 3`,
`throttled=0x80000` unchanged across every run.

---

### X-63 · The 31-pixel window is the cap, and it explains every failure · `DONE`

**ONE MEASUREMENT THAT UNIFIES FOUR.** `uint64_t` doubles the word and therefore
doubles the pixels per operation — the packing argument that motivated
[X-58](#x-58--e-32--x86-vector-paths-how-much-is-the-compilers-how-much-is-ours--done),
[X-60](#x-60--e-33-attempted-on-residualsums-written-bit-exact-and-188-slower--done)
and [X-61](#x-61--batching-across-keypoints-the-third-granularity-and-the-third-refutation--done).
[X-54](#x-54--e-9--should-the-word-type-vary-down-the-pyramid--done) measured it on
aarch64 and it lost **1.32× on track**, but that was the NEON paths compiling out at
`sizeof(WordType) != 4`. **On x86 there are no such guards**, so this is the clean test
of the packing argument with nothing else attached.

Full sequence, idle machine (load 1.53), three runs:

| | build | track | binCV | vs OpenCV | bytes |
|---|---|---|---|---|---|
| `uint32` (shipped) | 0.995–1.073 | 2.496–2.641 | 3.709–3.952 | **1.00×** | 436 704 |
| `uint64` | 1.203–1.227 | **2.447–2.574** | 3.886–4.035 | 0.93–0.97× | 439 104 |

**Track moved 2% for a doubled word.** The packing argument fails cleanly, with no
guards, no gather and no intrinsics to blame — and the reason is embarrassingly simple:

> **THE WINDOW IS 31 PIXELS WIDE. A 31-pixel window occupies ONE `uint32` word.**
> Widening the word does not let `residualSums` do more per operation; it only wastes
> more bits.

| word / register | bits | px used | utilisation |
|---|---|---|---|
| **`uint32` (shipped)** | 32 | 31 | **97%** |
| `uint64` | 64 | 31 | **48%** |
| SSE2 | 128 | 31 | 24% |
| **AVX2** | 256 | 31 | **12%** |

**THIS IS THE COMMON CAUSE BEHIND TWO PRIOR REFUTATIONS** — X-58 (compiler AVX2) and
X-60 (hand-written within a row). Both fed a 256-bit register **from a 31-pixel
window**; the proximate reasons were real, and this is the one underneath them.

**IT DOES NOT COVER [X-61](#x-61--batching-across-keypoints-the-third-granularity-and-the-third-refutation--done),
AND THIS ENTRY ORIGINALLY CLAIMED IT DID.** X-61 put **eight keypoints** in the lanes,
so window width does not enter — the register is full by construction. Its blocker was
the **gather**, and X-61 itself records that **the vector arithmetic won** and the
gathers gave it back. Corrected in [D-52](ARCHITECTURE.md#8-design-decisions); the
distinction matters because it is the difference between a wall and a layout bug.

**SO binCV's REAL RATE IN LK IS 31 px/op — NOT 32, 64 OR 256.** Against OpenCV's 16
(`CV_16S` in AVX2 lanes) that is a **1.94× packing advantage, and it is CAPPED BY THE
ALGORITHM, not by the word type or the ISA.** Against **5.6×** the operations, that is
**2.9× cost** — which is the gap, fully accounted, with nothing left over.

**WHAT THIS RULES OUT AND WHAT IT LEAVES.** It rules out *every* widening approach:
wider words, wider registers, and — since X-61 already measured it — batching windows
to fill them. **The only remaining lever is the operation count**, which is
[E-34](ARCHITECTURE.md#register)'s ×5 tap factor. The 2.9× is 1.94 packing against 5.6
operations; nothing can improve the numerator, so the denominator is the whole game.

**A note on `build`.** `uint64` made build **worse on x86 (1.2×)** where
[X-54](#x-54--e-9--should-the-word-type-vary-down-the-pyramid--done) measured it
**1.66× better on aarch64**. Build is a bulk pass over full rows, so the window cap
does not apply and the packing argument should hold — that it reverses by platform is
unexplained and is **not** claimed either way here. It does mean the "restructure
`build` for AVX2" avenue ([E-33](ARCHITECTURE.md#register)) has a weaker prior than
X-59's adder ceiling suggested.

**Decision:** none — this is an explanation, and its value is that it closes a line of
enquiry rather than opening one. Recorded as
[D-52](ARCHITECTURE.md#8-design-decisions). `frontend_sequence` keeps
`BINCV_BENCH_WORD` so the word type stays measurable without editing source.

**Method:** `benchmark/frontend_sequence.cpp` with `-DBINCV_BENCH_WORD=uint64_t`, full
1710-frame sequence, three runs each on an idle machine.

---

### X-64 · The x86 deficit was THREADS, and the benchmark let it drift · `DONE`

**A MEASUREMENT THAT CONTRADICTS WHAT THIS REPOSITORY SAYS, WHICH IS WHY IT IS HERE**
([CLAUDE.md](../CLAUDE.md): report it rather than adjusting the code to fit the doc).
`benchmark/frontend_sequence` sets `cv::setNumThreads` **only if
`BINCV_OPENCV_THREADS` is set**, and it was not set for any x86 run in the session
that produced [X-63](#x-63--the-31-pixel-window-is-the-cap-and-it-explains-every-failure--done).
So those runs compared **single-threaded binCV against twelve-threaded OpenCV** —
**binCV has no threading at all** — and the resulting `0.65–1.00×` was read, for most
of a working session, as a **SIMD** deficit.

**The reference device never had this problem and that is why it was never noticed:**
`run_on_pi.sh` runs under `taskset -c 3`, and OpenCV's threads cannot escape a single
pinned core. Every entry in this file that *states* its thread count states **one**
([X-37](#x-37--the-frontend-on-the-reference-device--done),
[X-49](#x-49--the-frontend-after-the-api-swap-a-control-and-a-new-headline--done),
[X-52](#x-52--where-x86-is-now--done), [X-38](#x-38--the-full-sequence--done)). The
x86 runs are the ones that drifted.

**Controlled, full 1710-frame sequence, three runs each, idle machine:**

| OpenCV threads | binCV ms/frame | OpenCV ms/frame | ratio |
|---|---|---|---|
| **1** | 2.794 / 2.651 / 3.187 | 3.333 / 3.032 / 3.791 | **1.19× / 1.14× / 1.19×** |
| 12 | 3.030 / 3.198 / 3.190 | 1.981 / 2.065 / 2.051 | 0.65× / 0.65× / 0.64× |

**~~binCV IS AHEAD OF OpenCV ON x86 AT EQUAL CORE COUNT — 1.14–1.19×~~ — CORRECTED
BELOW.** The thread-control finding stands. **The headline does not: it was measured on
`MH_01_easy`, not on the sequence every other entry in this file uses**, and it does not
survive the change.

#### CORRECTION: THE HEADLINE WAS SEQUENCE-DEPENDENT, AND THE SEQUENCE WAS WRONG

`/mnt/g` dropped, and the local copy that replaced it is **`euroc-v1_02-cam0`** — the
**V1_02** sequence [X-38](#x-38--the-full-sequence--done),
[X-49](#x-49--the-frontend-after-the-api-swap-a-control-and-a-new-headline--done) and
[X-52](#x-52--where-x86-is-now--done) all used. Re-measured there, two runs:

| OpenCV threads | binCV | OpenCV | ratio on **V1_02** | ratio on MH_01 |
|---|---|---|---|---|
| **1** | 3.325 / 3.295 | 2.967 / 2.937 | **0.89× / 0.89×** | 1.14 – 1.19× |
| 12 | 3.367 / 3.475 | 1.681 / 1.667 | 0.50× / 0.48× | 0.64 – 0.65× |

**On the canonical sequence binCV is BEHIND single-threaded OpenCV, 0.89×.** V1_02 is
the harder sequence — `track` costs **2.28 ms/frame against MH_01's 1.66**, 37% more
work — and the ratio moves further than most experiments in this file measure.

**WHAT SURVIVES AND WHAT DOES NOT.** The *reason* X-64 exists is unaffected and is
confirmed on both sequences: **OpenCV gains 1.68× (MH_01) / 1.77× (V1_02) from threads**,
so an uncontrolled comparison measures parallelism and reads as implementation. The
benchmark's one-thread default, the disclaimer and the corrected NOTE all stand.
**What does not stand is "binCV leads at equal core count"** — true on one sequence,
false on the other, and stated as if it were a property of the library.

**THIS IS THE SECOND HEADLINE IN TWO DAYS TO REST ON AN UNCONTROLLED VARIABLE** —
threads, then the sequence. Recorded as such in
[D-54](ARCHITECTURE.md#8-design-decisions); the rule that follows is in
[D-57](ARCHITECTURE.md#8-design-decisions).

**WHAT THREADING IS WORTH, AND WHAT SIMD IS WORTH.** OpenCV goes 3.33 → 1.98 ms on
twelve threads: **1.68× from parallelism**. Its SIMD is fully active in *both* rows —
so the row that isolates the implementations is the one-thread row, and there **scalar
binCV beats vectorised OpenCV**. That is [D-52](ARCHITECTURE.md#8-design-decisions)'s
SWAR argument cashing out: one 32-bit `AND` covers 32 pixels, and binCV gets
**31 px/op** from ordinary integer instructions where OpenCV gets 16 from `CV_16S` in
AVX2 lanes.

**THIS DOES NOT REOPEN [E-32](ARCHITECTURE.md#register) OR OVERTURN
[D-52](ARCHITECTURE.md#8-design-decisions).** X-58/X-60/X-61 measured *vector width*
and D-52 explains why all three failed; none of that is threads, and none of it
changes. What changes is the **framing**: "binCV is behind on x86" was an artefact of
an uncontrolled denominator, so the open x86 question is not vector width — it is that
**binCV is single-threaded and OpenCV is not**, an axis this project has never
examined.

**Decision — three changes, none of them to a kernel.**

1. **`frontend_sequence` now defaults `cv::setNumThreads(1)`.** This codifies existing
   practice rather than choosing a new denominator: it is what every recorded entry
   used and what the pinned reference device gets for free.
2. **A ratio at any other thread count prints a loud disclaimer** naming this entry. An
   unpinned x86 box silently changes what the ratio means; the device could not.
3. **The criterion-4 NOTE is corrected.** It read "binCV has NO VECTOR PATH ON x86 …
   so this is binCV SCALAR against OpenCV SSE", offering vector width as the cause of a
   deficit that was threads. The vector-path fact is true and stays; the causal claim
   is gone.

**WHAT IS DELIBERATELY NOT DECIDED HERE.** Whether **multi-core OpenCV** is the fairer
denominator for a multi-core target is a real question and a different one — a VIO
frontend on a phone has more than one core, and binCV using one of them is a genuine
limitation, not a measurement artefact. **It is left open rather than settled by this
entry**, because it is a question about the product's shape and no experiment has
addressed it.

**Also corrected while here:** [X-52](#x-52--where-x86-is-now--done)'s method note
named a specific x86 part. CLAUDE.md keeps platform language generic and the rest of
the file says `x86_64`; it now does too.

**Method:** `benchmark/frontend_sequence.cpp`, full 1710-frame sequence,
`BINCV_OPENCV_THREADS` ∈ {1, 12}, three runs each, load average 0.33 at start.
**No aarch64 arm, and none is needed** — `taskset -c 3` already pins the reference
device to one core, which is what makes its recorded ratios unaffected.

---

### X-65 · E-35 — what does threading the frontend buy, and what does it cost? · `DONE`

**COMMITTED BEFORE ANY CODE.** The rule is written first because
[D-53](ARCHITECTURE.md#8-design-decisions) is one day old and its whole content is that
a kernel ratio can point the opposite way from the workload.

**Gates:** [E-35](ARCHITECTURE.md#register). **binCV has never been threaded, and no
record anywhere chooses that** ([D-56](ARCHITECTURE.md#8-design-decisions)).
[X-64](#x-64--the-x86-deficit-was-threads-and-the-benchmark-let-it-drift--done)
measured OpenCV gaining **1.68×** from threads on the same workload.

**THE ARMS.** `track` is 59.5% of the frontend and `build` 38.8%.

| | |
|---|---|
| **A** | shipped: one thread |
| **B** | `track` parallel over **keypoints**, `build` serial |
| **C** | B, plus `build` parallel over **row bands** within a level, barrier per level |

**MEASURED ON BOTH, AND THE PROTOCOL HAS TO CHANGE TO ASK THE QUESTION.**
`run_on_pi.sh` pins with `taskset -c 3`, which makes a threading arm unmeasurable; this
experiment runs it **unpinned across all four cores**, and reports the single-core
control from the same binary so the two are comparable. **That is a deviation from the
standing protocol and is confined to this experiment.**

**BOTH AXES, AND THE MEMORY ONE IS NOT A FORMALITY.** Peak working set is reported per
arm. The claim under test is that the shared pyramids and ladders are **read-only**, so
only per-thread stack scales — if that is wrong, [CLAUDE.md](../CLAUDE.md)'s tiebreak
applies and **memory wins**, whatever the speed says.

**DECISION RULE, WRITTEN BEFORE MEASURING.** `T` = threads, on a 4-core device.

- **Band A — arm C reaches ≥ 2.5× at T=4 AND peak working set grows < 5%.** Threading
  is the frontend's largest remaining lever. **Ship it ON BY DEFAULT in hosted builds**,
  with a swappable backend and a `parallelFor` hook — see the amendment below for why
  this band no longer reads "caller-supplied hook, not threads binCV spawns".
- **Band B — ≥ 2.5× but the working set grows ≥ 5%.** Report both; **do not ship**
  without an explicit decision from the project owner, because this is exactly the
  conflict CLAUDE.md's tiebreak names and E-26 is the precedent for declining.
- **Band C — 1.5–2.5×.** Real but not transformative. Ship the hook for `track` only;
  `build`'s barrier is not worth its complexity at that return.
- **Band D — < 1.5× at T=4.** Something dominates that is not compute — most likely
  memory bandwidth on the shared pyramids, which would be a **finding about the
  representation**, not about threading, and the far more interesting result.
- **ANY BAND is void if `track`'s output is not bit-identical to arm A.** Keypoints are
  independent, so parallelism must not change a single flow vector. A difference means
  a data race, and a data race means the timing is meaningless.

#### AMENDMENT, BEFORE ANY MEASUREMENT: the default polarity flips

**This changes a band, and the protocol requires saying exactly when.** X-65 is
`RULE ONLY` — **no arm has been built and no number collected** — so this is a rule
being sharpened before measuring, not a rule being fitted to a result. The speed and
footprint thresholds are untouched; only what Band A *ships* changes.

**THE ORIGINAL BAND SAID "a library at this level does not own the caller's thread
policy" AND CITED THE REFERENCE IMPLEMENTATION.** HybVIO does run
`Processor::createThreadPool(1)` per stage — single-worker pools, pipeline parallelism
across stages, no data parallelism inside a kernel. That is real evidence and it still
stands **for that integrator**.

**WHAT IT MISSED IS EVERYONE ELSE, AND [X-64](#x-64--the-x86-deficit-was-threads-and-the-benchmark-let-it-drift--done)
IS THE PROOF.** OpenCV ships parallel by default. If binCV ships serial by default,
then **every casual comparison anyone runs is single-threaded binCV against
twelve-threaded OpenCV** — which is precisely the trap X-64 documents this project
falling into, with its own benchmark, for most of a working session. A default that
makes the library lose its own benchmark is not a neutral default.

**AND THE OBJECTIONS THAT LOOKED STRUCTURAL MOSTLY ARE NOT:**

- **Determinism survives.** Keypoints are independent and `build`'s row bands write
  disjoint memory, so both arms are **bit-exact** against serial — which is already a
  precondition of this experiment, not a new requirement.
- **The footprint objection is bounded and already gated.** Pyramids and ladders are
  read-only and shared ([D-56](ARCHITECTURE.md#8-design-decisions)); only per-thread
  stack and pool state scale. Band B still declines at ≥5% whatever the speed.
- **The core-only claim is untouched, because the pool cannot live in core anyway.**
  `bincv_core` is allocation-free and builds `-fno-exceptions`, where `std::thread`
  is not usable. **That constraint does not argue for a serial default — it argues for
  a PROFILE-SCOPED one**, and this repository already builds four profiles.

**SO THE DEFAULT FOLLOWS THE BUILD PROFILE, WHICH IS THE ONLY PLACE IT CAN BE BOTH
HONEST AND FAST:**

| profile | default |
|---|---|
| hosted (Release, Release+OpenCV) | **parallel, sized to hardware concurrency** |
| core-only / `-fno-exceptions` / freestanding | **serial, no pool, no allocation — unchanged** |

with `bincv::setNumThreads(n)` (`1` serialises), a swappable backend, and a
`parallelFor` hook an integrator installs to hand binCV their existing pool. **That is
OpenCV's surface, so an integrator finds what they expect — and HybVIO's model is one
`setNumThreads(1)` call away**, which is exactly what such codebases already do to
OpenCV.

**PROVISIONAL, AND SAYING SO.** [CLAUDE.md](../CLAUDE.md) requires a decision made
without the measurement loop to be marked provisional. This is a decision about API
shape taken **ahead of** X-65's numbers; if Band B or D fires, the shape ships serial
by default regardless of how good the argument reads.

**Method:** `benchmark/frontend_sequence.cpp` with a thread-count knob; full sequence;
OpenCV held at **one thread throughout**, per [X-64](#x-64--the-x86-deficit-was-threads-and-the-benchmark-let-it-drift--done),
so the arms compare against a fixed denominator and against each other. **The headline
ratio is additionally reported at OpenCV's own default**, because that is the
comparison a user actually runs and X-64's whole lesson is that leaving it unstated is
how a denominator drifts.

---

#### THE RESULT: BAND A ON ARM B, AND THE PROFILE MOVES TO `build`

**BIT-EXACTNESS FIRST, BECAUSE IT IS A PRECONDITION AND NOT A BAND.** Splitting the
point array across four threads reproduced the serial result on **every one of 300
frames, 0 differed.** Keypoints are independent and the pyramids are read-only, so
this needed **no library change at all** — the arm is a split of the existing
`calcOpticalFlowPyrLK` call, which is why it could answer the bands before any API was
designed for it.

**V1_02, full 1710 frames, two passes, OpenCV pinned to one thread:**

| threads | `track` ms | speedup | `build` ms | binCV ms | vs OpenCV |
|---|---|---|---|---|---|
| **1** | 2.294 / 2.259 | 1.00× | 0.893 / 0.883 | 3.379 / 3.329 | 0.90× |
| 2 | 1.355 / 1.331 | 1.70× | 0.852 / 0.833 | 2.401 / 2.344 | 1.20× / 1.22× |
| **4** | 0.895 / 0.858 | **2.60×** | 0.857 / 0.838 | 1.938 / 1.884 | **1.49× / 1.51×** |
| 6 | 0.750 / 0.748 | 3.04× | 0.871 / 0.863 | 1.810 / 1.797 | 1.62× / 1.58× |
| 12 | 0.618 / 0.608 | 3.71× | 0.877 / 0.864 | 1.682 / 1.666 | 1.74× / 1.75× |

**FOOTPRINT: FLAT.** Peak RSS is **29 848 / 29 828 / 29 844 KB** at T = 1 / 4 / 12 — a
**0.07%** spread, which is noise. [D-56](ARCHITECTURE.md#8-design-decisions)'s claim
that only per-thread stack scales is confirmed with three orders of magnitude of margin
against Band A's 5%.

**BAND A FIRES — with one precision.** Its numeric thresholds are met: **2.60× on
`track` at T=4** against a ≥2.5× bar, footprint growth **0.07%** against <5%, bit-exact.
But the band names **arm C** (track *and* build), and **only arm B was built.** Arm C is
**unmeasured**, and the table shows why that now matters more than it did when the rule
was written.

> **`build` DOES NOT SCALE, AND IT HAS BECOME THE BOTTLENECK.** It is **26.4% of the
> frontend at one thread and 52.2% at twelve**, essentially unchanged in absolute terms
> — 0.89 → 0.86 ms. Threading `track` moved the constraint rather than removing it.

**AND THE HONEST COMPARISON AT EQUAL THREAD COUNTS IS PARITY, NOT A LEAD.** binCV at
12 threads is **1.674 ms**; OpenCV at 12 threads is **1.674 ms**. Threading takes binCV
from 0.89× to **1.00×** against a like-configured OpenCV — a real and large gain, and
**not** a lead. The 1.49–1.75× figures above are against **one-thread** OpenCV and must
always be labelled as such.

**Decision: threading is the largest single lever this project has measured** — 0.89× →
1.50× against the recorded one-thread denominator, at no footprint cost and bit-exact.
[D-57](ARCHITECTURE.md#8-design-decisions). The provisional API shape
([D-56](ARCHITECTURE.md#8-design-decisions)) survives its gate: Band B and Band D did
not fire, so **hosted builds default parallel**. **Arm C stays owed**, and `build` is now
the highest-value target in the frontend — which is [E-33](ARCHITECTURE.md#register),
by a different route.

**Measured on:** V1_02, `~/bincv-data/euroc-v1_02-cam0`, x86_64, 12 cores, unpinned —
the protocol deviation this experiment declared in advance.

---

### X-66 · E-36 — staging, not gathering: the x86 keypoint batch, second attempt · `RULE ONLY`

**COMMITTED BEFORE ANY CODE, AND WITH A CEILING BEFORE THE ARM.**

**Gates:** [E-36](ARCHITECTURE.md#register). [X-61](#x-61--batching-across-keypoints-the-third-granularity-and-the-third-refutation--done)
found the vector arithmetic **wins** (≈48 vector ops against ≈100 scalar) and the
gathers give it back. [D-55](ARCHITECTURE.md#8-design-decisions) says why that is a
**data-movement** result rather than the packing wall
[D-52](ARCHITECTURE.md#8-design-decisions) filed it as.

**TWO CHANGES, NEITHER OF WHICH IS A PORT OF THE NEON PATH.**

1. **STAGE ONCE, REUSE ACROSS ITERATIONS.** Eight of the twelve words read per row —
   `self`, `magX`, `magY`, `signX`, `signY` — belong to the **previous** frame, which LK
   linearises about and never re-reads at a new offset. Transpose eight keypoints' worth
   into `[row][plane][lane]` **once per keypoint per level**; the inner loop then issues
   `_mm256_loadu_si256`, not `_mm256_i32gather_epi32`. **LK runs up to 20 iterations**,
   so the movement is paid once rather than twenty times.
2. **CARRY-SAVE, NOT POPCOUNT-PER-ROW.** AVX2 has no vector popcount and this class of
   core has no `AVX512-VPOPCNTDQ`, which is what made X-60's emulation lose. But the
   kernel needs **the sum over 31 rows**, not each row's count — so compress 31
   one-bit-per-lane values into a **5-plane bit-sliced sum** with `AND`/`XOR` only, then
   popcount **five** words with weights 1,2,4,8,16. `maj3` and `addShifted` in
   ops/bitslice.hpp are already the full adder this needs.

**THE FOOTPRINT NUMBER, STATED BEFORE MEASURING.** 8 keypoints × 31 rows × 8 words ×
4 B ≈ **8 KB on a 436 704 B peak, +1.8%** — transient, not a second copy of anything.
[E-26](ARCHITECTURE.md#register) declined **+21%** for a whole-level conversion; **this
is an order of magnitude smaller and that decision does not settle it.** If the measured
figure exceeds **+5%**, the arm is declined on footprint regardless of speed.

**DECISION RULE, WRITTEN BEFORE MEASURING.**

- **Band A — ceiling ≥ 2.0× on `residualSums` AND the whole-frontend arm ≥ 1.3×.**
  Write it. [D-53](ARCHITECTURE.md#8-design-decisions) makes the frontend arm
  **mandatory, not confirmatory** — X-62 was 1.75× in the kernel and 0.31× on the
  frontend.
- **Band B — ceiling ≥ 2.0× but the frontend gains < 1.3×.** Report where it went. That
  would mean `residualSums` is no longer the frontend's constraint, which redirects
  Phase 5 rather than closing it.
- **Band C — ceiling 1.2–2.0×.** Below what staging plus CSA should deliver on paper;
  **report the gap between the op-count model and the measurement** before writing
  anything.
- **Band D — ceiling < 1.2×, or slower.** Then the operands were never the problem and
  [D-50](ARCHITECTURE.md#8-design-decisions)'s verdict stands at every granularity.
  **Record it as the fourth refutation and close E-36.**

**BIT-EXACTNESS IS A PRECONDITION, NOT A BAND.** CSA reassociates a sum of integers,
which is exact — unlike [X-62](#x-62--e-34--can-the-four-tap-correlations-become-one--done),
this arm computes **the same integers**, and the arms are compared for **equality**
before they are timed. A mismatch is a bug, not a trade.

---

#### THE CEILING: 2.9×, BIT-EXACT — AND THE CREDIT IS NOT WHERE THE PROPOSAL PUT IT

Three runs, idle machine, 20 batches of 8 keypoints, **all arms produce the same ten
integers per keypoint and are checked for equality before timing**:

| arm | ns / 20 batches | vs A |
|---|---|---|
| **A** shipped: scalar, one keypoint at a time, `POPCNT` | 111 586 – 122 727 | 1.000× |
| **B** staged + **per-row emulated popcount** | 39 174 – 41 569 | **2.85 – 2.95×** |
| **C** staged + **Harley-Seal CSA tree** | 36 835 – 42 398 | **2.90 – 3.03×** |
| *(the staging transpose alone)* | 12 363 – 14 031 | — |

**B AND C ARE THE SAME WITHIN NOISE, AND THAT REFUTES HALF OF
[D-55](ARCHITECTURE.md#8-design-decisions).** The proposal argued that AVX2's missing
vector popcount was a first-order obstacle and that a CSA tree — pure `AND`/`XOR`,
replacing 31 emulated popcounts with five — would be worth ~1.7× over emulation.
**Measured: 0–5%, inside the spread.** Arm B exists precisely to split the credit, and
it took all of it.

> **CONTIGUITY WAS THE ENTIRE STORY. The popcount was never the bottleneck — the
> ADDRESSING was**, which is [X-41](#x-41--e-23--is-the-extraction-addressable--done)'s
> finding arriving from a third direction.

**Amortised over LK iterations** — the transpose is paid once per batch per level, the
inner loop once per iteration:

| iterations | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| **C + staging, vs A** | **2.18×** | 2.48× | 2.61× | 2.72× | 2.80× | **2.85×** |

**It clears the 2.0× bar at ONE iteration**, so the amortisation argument, which was
the proposal's load-bearing claim, turns out not to be load-bearing either. Staging
pays for itself immediately.

**FOOTPRINT: 15 872 B per 8-keypoint batch, one batch live at a time — +3.6% on a
436 704 B peak.** Under X-66's +5% decline threshold, and an order of magnitude below
[E-26](ARCHITECTURE.md#register)'s declined +21%.

**A NOTE ON `target("avx2")`, BECAUSE IT IS WHY X-60 LOST AND IT IS NOT ABOUT AVX2.**
GCC and Clang refuse to inline a callee whose target features are **not a subset of the
caller's** — the attribute exists for runtime multiversioning, and inlining an AVX2 body
into a baseline caller would defeat it. X-60 marked leaf helpers, so every one became a
real call: 310 per window. This benchmark puts `-mavx2` on the **whole translation
unit** instead, and `nm` confirms `popcnt32`, `harleySeal32` and `csa` have **no
standalone symbols** while `avx2_ceiling.cpp`'s attribute-marked functions still do.
**The rule for shipping: mark ONE COARSE entry point, never leaf helpers.**

#### THE GUARD ARM: 2.9× WAS OPTIMISTIC BY 28%, AND THIS WOULD HAVE BEEN THE SIXTH

**ARMS B AND C STAGE EVERYTHING, TAPS INCLUDED. A KERNEL CANNOT.** The four tap words
depend on each keypoint's own integer displacement `(tapX, tapY)`, which **differs
across lanes and moves between iterations**. Only the **eight previous-frame words** —
`self`, `magX`, `magY`, `signX`, `signY` — are genuinely invariant. So arm **D** stages
those eight and **gathers the four taps**, which is the honest bound on a shipped path.

| arm | ns / 20 batches | vs A |
|---|---|---|
| **C** staged **everything** (optimistic) | 38 402 – 43 478 | 2.71 – 3.06× |
| **D** staged invariants, **taps gathered** | **54 651 – 56 215** | **2.085 / 2.092 / 2.122×** |

**Bit-exact, and tight across three runs.** The difference between C and D is **entirely
the tap gathers**, which reproduces [X-61](#x-61--batching-across-keypoints-the-third-granularity-and-the-third-refutation--done)'s
finding at a smaller scale: gathers cost, and every word moved off them is a win.

**[D-49](ARCHITECTURE.md#8-design-decisions) SAYS FIVE CEILINGS HAVE OVERSTATED. THIS
WOULD HAVE BEEN THE SIXTH** — 2.9× reported where 2.1× is available. It was caught by
writing the arm that models what the kernel can actually do, before writing the kernel.

**Amortised over LK iterations** — and unlike arm C, here the amortisation *does*
matter, because arm D's inner loop is slower so the fixed transpose is a larger share:

| iterations | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| **D + staging, vs A** | **1.655×** | 1.837× | 1.907× | 1.967× | 2.015× | **2.039×** |

**Footprint is smaller than the buffer this benchmark declares**: only the eight
invariant words need staging, 31 × 8 × 32 B = **7 936 B, +1.8%** on a 436 704 B peak.
The 15 872 B the harness prints includes the taps, which the shipped path would gather.

**WHAT THIS PROJECTS ONTO THE FRONTEND, STATED BEFORE MEASURING IT.**
[X-41](#x-41--e-23--is-the-extraction-addressable--done) puts `residualSums` at
**43.7%** of the frontend. At 2.0×, Amdahl gives **1/(0.563 + 0.437/2.0) = 1.28×** —
**below Band A's 1.3× line, marginally.** So the frontend arm is expected to land on
the boundary, and Band B (report where it went, do not ship) is a live outcome rather
than a formality.

**Status: the ceiling clears Band A's first half (≥2.0×) on the implementable arm, but
only just. The whole-frontend arm (≥1.3×) is owed**, and [D-53](ARCHITECTURE.md#8-design-decisions) makes it mandatory
rather than confirmatory — X-62 was 1.75× in the kernel and 0.31× on the frontend.

**Method:** `benchmark/kpbatch_staged_ceiling.cpp` for the ceiling, `-mavx2` on the
target rather than on the functions; whole-frontend via
`benchmark/frontend_sequence.cpp`; both with OpenCV at one thread.
x86 only — aarch64 has `CNT` and its own measured paths, and this changes nothing there.

---

### X-67 · `build` decomposed — and E-33 is worth almost nothing · `DONE`

**THIS CORRECTS A CLAIM MADE SIX HOURS EARLIER IN [D-57](ARCHITECTURE.md#8-design-decisions).**
[X-65](#x-65--e-35--what-does-threading-the-frontend-buy-and-what-does-it-cost--done)
found `build` rising to **52.2%** of the frontend at twelve threads and concluded that
[E-33](ARCHITECTURE.md#register) — AVX2 for `pyrDown` — was "the highest-value target
left." **That inference treated `build` as one thing. It is not.**

Splitting the stage, V1_02, 900 frames:

| | T = 1 | T = 4 |
|---|---|---|
| `track` | 2.225 (69.1%) | 0.905 (47.1%) |
| **`build` total** | **0.813 (25.2%)** | **0.821 (42.7%)** |
| … `fromCVMat` (the harness's input conversion) | **0.657 (20.4%)** | **0.662 (34.4%)** |
| … `pyrDown` | **0.117 (3.6%)** | **0.118 (6.1%)** |
| … derivatives | 0.039 (1.2%) | 0.040 (2.1%) |
| `detect` | 0.184 (5.7%) | 0.196 (10.2%) |

> **`pyrDown` IS 3.6% OF THE FRONTEND AT ONE THREAD AND 6.1% AT FOUR. An INFINITE
> speedup on it is worth 1.037× and 1.065×.** [E-33](ARCHITECTURE.md#register) is not
> the highest-value target; it is close to the lowest.

**EIGHTY PERCENT OF `build` IS `fromCVMat` — THE `CV_8U` → BIT-PLANE CONVERSION.** That
is the harness's input path, not a kernel: it exists because the reference preprocessing
emits an OpenCV `Mat`, and **a binary-frame frontend receives bits from its sensor
stage and never performs it.** OpenCV pays nothing equivalent — it works on the `CV_8U`
directly — so this is a cost binCV carries **in this comparison only**.

**BOTH NUMBERS, LABELLED, BECAUSE EITHER ALONE MISLEADS** — which is
[D-58](ARCHITECTURE.md#8-design-decisions) applied to the variable this experiment
found:

| | binCV | vs 1-thread OpenCV |
|---|---|---|
| T=1, **as measured** (conversion included) | 3.223 | **0.94×** |
| T=1, conversion excluded | 2.566 | 1.18× |
| T=4, **as measured** | 1.922 | **1.62×** |
| T=4, conversion excluded | 1.260 | 2.47× |

**Reporting only the second would be self-serving; only the first understates by
20–34%.** The headline stays the **as-measured** row, and the excluded row is reported
beside it as what a deployed binary-frame pipeline would see.

**AND A REDUNDANCY THE HARNESS CARRIES.** `loadLevel0` converts **both** `prev` and
`next` every frame, and `build()` builds **both** pyramids — but this frame's `next` is
next frame's `prev`. **Roughly half of both costs is recomputation of something already
computed**, which a ping-pong buffer removes. Registered as
[E-37](ARCHITECTURE.md#register) rather than fixed here, because changing the harness
changes every recorded frontend number and that is a decision, not a cleanup.

**Decision: [E-33](ARCHITECTURE.md#register) is DEMOTED, not closed** — `pyrDown` may
still matter on the reference device, where the profile differs and X-30 put build at
25.8%. On x86 it is 3.6%. [D-59](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/frontend_sequence.cpp` with per-sub-stage timers; V1_02, 900
frames, OpenCV at one thread.

---

### X-68 · `track` decomposed — 91.5% is iterated `residualSums` · `DONE`

**[D-59](ARCHITECTURE.md#8-design-decisions) SAYS A STAGE TOTAL IS NOT A TARGET, SO
`track` GETS THE SAME TREATMENT `build` JUST GOT** before anything is written against
it. `maxIterations = 0` runs every per-point step — covariance, clipping, entry-level
selection, the min-eigenvalue test — and **no iterations at all**, which isolates setup
exactly rather than by extrapolation.

V1_02, 600 frames, one thread:

| `maxIterations` | `track` ms | what it adds |
|---|---|---|
| **0** | **0.178** | per-point setup, all of it |
| 1 | 0.626 | +0.448 — one iteration for every point |
| 2 | 1.039 | +0.413 |
| **20 (shipped)** | **2.099** | — |

- **Per-point setup is 0.178 ms — 8.5% of `track`.**
- **Iterated work is 1.921 ms — 91.5%**, and it is `residualSums` plus a 2×2 solve of a
  handful of flops.
- **Mean effective iterations = 1.921 / 0.448 = 4.29** per point per level, which is
  also the amortisation factor [X-66](#x-66--e-36--staging-not-gathering-the-x86-keypoint-batch-second-attempt--rule-only)'s
  staging gets: its arm D reads **1.967×** at five iterations.

**WHAT [X-66](#x-66--e-36--staging-not-gathering-the-x86-keypoint-batch-second-attempt--rule-only)'s
2.09× WOULD BE WORTH, COMPUTED BEFORE THE ARM IS WRITTEN.** Applying it to the 91.5%
and leaving setup alone: `track` 2.099 → **1.093 ms, 1.92×**. On the frontend:

| | binCV now | with X-66 | vs 1-thread OpenCV |
|---|---|---|---|
| T=1 | 3.223 | **≈2.16** | 0.94× → **≈1.40×** |
| T=4 | 1.922 | **≈1.49** | 1.62× → **≈2.03×** |

**That clears X-66's Band A (≥1.3× on the frontend) with room** — where the earlier
Amdahl estimate using [X-41](#x-41--e-23--is-the-extraction-addressable--done)'s 43.7%
put it at 1.28×, *below* the line. **The difference is that 43.7% was measured on the
reference device and this is x86**, where `track` is a larger share. Both are recorded;
the projection that matters for an x86 decision is this one.

**A caveat kept in view:** these are projections from a decomposition, not a
measurement of the arm. [D-53](ARCHITECTURE.md#8-design-decisions) exists because a
kernel ratio pointed the opposite way from the workload, and nothing here substitutes
for building it.

**Method:** `benchmark/frontend_sequence.cpp`, `BINCV_LK_ITERS` sweep including 0;
V1_02, 600 frames, OpenCV at one thread. **`perf` is not usable on this kernel** (WSL2
without matching `linux-tools`), which is why the decomposition is done with an
iteration sweep rather than a profile — the sweep is a direct measurement, not an
estimate.

---

### X-69 · Staging WITHOUT vectorising — 1.27× on `track`, bit-exact · `DONE`

**[X-66](#x-66--e-36--staging-not-gathering-the-x86-keypoint-batch-second-attempt--rule-only)
FOUND THAT CONTIGUITY, NOT THE POPCOUNT, WAS THE WHOLE STORY** — its arms B and C were
identical within noise. If the win is *addressing*, then it should be available
**without any vector code at all**, and this is that arm.

**THE IDEA, WHICH IS SMALLER THAN X-66's AND SHIPS TODAY.** Of the twelve words
`alignedResidualSums` reads per row, **eight belong to the previous frame** — `self`,
`magX`, `magY`, `signX`, `signY` — and LK linearises about the previous frame, so they
are **identical on every iteration**. `region` is fixed per point per level. So extract
them **once** into a stack buffer and let all
[X-68](#x-68--track-decomposed--915-is-iterated-residualsums--done)'s **4.29 mean
iterations** read from it. No lanes, no batching, no AVX2 — and it therefore works on
**aarch64 too**, where X-66's AVX2 arm never would.

**A/B on the same binary, V1_02, 900 frames, two runs each:**

| | `track` ms | binCV ms | OpenCV ms | ratio |
|---|---|---|---|---|
| unstaged | 2.190 / 2.223 | 3.170 / 3.222 | 2.859 / 2.992 | 0.90× / 0.93× |
| **staged** | **1.721 / 1.753** | **2.895 / 2.945** | 2.834 / 2.877 | **0.98× / 0.98×** |

**`track` 1.27×, the frontend 1.09×, and the ratio against one-thread OpenCV moves
0.90–0.93 → 0.98.** Measured as a stash-and-rebuild A/B rather than against an earlier
run, because OpenCV's own figure drifted 2.86–2.99 between them and that drift is
larger than some of this project's decisions.

**THE MODEL PREDICTED IT.** [D-37](ARCHITECTURE.md#8-design-decisions) put extraction at
**45.4%** of the kernel; staging removes 8 of 12 words (30%) on all but the first of
4.29 iterations (77%) → **1.30× expected, 1.27× measured.** After
[X-62](#x-62--e-34--can-the-four-tap-correlations-become-one--done) and
[X-66](#x-66--e-36--staging-not-gathering-the-x86-keypoint-batch-second-attempt--rule-only)
both broke their models, one that holds is worth noting.

**COST: 2 048 B OF STACK** at the shipped `N = 2` (6 656 B at the `N = 8` ceiling), and
**zero heap** — [CLAUDE.md](../CLAUDE.md) forbids a kernel allocating and this operation
has no caller scratch. `stageWindow` **declines** rather than overrunning: windows wider
than a word, or taller than 64 rows, take the unstaged path unchanged. Peak working set
is untouched.

**BIT-EXACT, AND PINNED BY A TEST THAT WAS WATCHED TO FAIL.**
`Flow.StagedMatchesUnstaged_{N1,N2,N3}` compares the staged and unstaged paths over 624
windows each — swept off every edge, four tap signs. **A one-bit fault injected into one
staged row makes 374 / 487 / 522 of 624 windows differ.** The tracker-level tests would
have caught a gross error; they would not have caught a stale row.

**Decision: SHIPPED.** It is the default for any window that can be staged. This is the
**scalar half** of [E-36](ARCHITECTURE.md#register); X-66's AVX2 keypoint batch stacks
on top of it and its 2.09× ceiling was measured against the *unstaged* scalar arm, so
**X-66's remaining headroom over this is smaller than 2.09× and is not yet measured.**
[D-60](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/frontend_sequence.cpp`, V1_02, 900 frames, OpenCV at one thread;
A/B by `git stash` on `ops/opticalFlow.hpp` with a full rebuild between arms. Gate green
on all four configurations, `expected-checks.txt` floors raised by the 6 checks the new
oracle adds.

---

### X-70 · The taps too — cached on the integer displacement · `DONE`

**[X-69](#x-69--staging-without-vectorising--127-on-track-bit-exact--done) STAGED EIGHT
OF TWELVE WORDS AND LEFT FOUR** — the taps, because they move. **They move as
`floor(offX)`, and the iteration is SHRINKING `off`.** Once the estimate settles inside
a pixel the integer part stops changing, and the same four words per row are
re-extracted every remaining iteration for nothing.

So the taps are **cached against `(tapX, tapY)`** rather than staged. Sound by
construction: the tap words are a pure function of `lv.next`, `region` and the integer
displacement; the first two are fixed for the point, and the third is the key.

**THREE ARMS, ONE SESSION, TWO RUNS EACH** — V1_02, 900 frames, OpenCV at one thread and
stable at **2.80–2.85 ms** across all six, which is what makes the ratios comparable:

| arm | `track` ms | binCV ms | vs OpenCV |
|---|---|---|---|
| **A** unstaged (before tonight) | 2.180 / 2.155 | 3.168 / 3.134 | 0.90× / 0.91× |
| **B** staged ([X-69](#x-69--staging-without-vectorising--127-on-track-bit-exact--done)) | 1.748 / 1.743 | 2.917 / 2.929 | 0.97× / 0.96× |
| **C** staged + tap cache | **1.447 / 1.524** | **2.619 / 2.682** | **1.07× / 1.05×** |

- **Tap cache alone: 1.17× on `track`.**
- **Cumulative A → C: `track` 1.46×, the frontend 1.19×.**
- **The ratio against single-threaded OpenCV crosses 1.0: 0.905 → 1.06×.**

> **binCV NOW LEADS SINGLE-THREADED OpenCV ON THE CANONICAL SEQUENCE.**
> [X-64](#x-64--the-x86-deficit-was-threads-and-the-benchmark-let-it-drift--done) claimed
> that and had to withdraw it — the claim came from an easier sequence. **This time it is
> V1_02, the sequence every recorded frontend entry uses, and it was earned by two
> changes rather than by a choice of input.**

**BIT-EXACT, AND THE REUSE PATH IS TESTED — NOT JUST THE FILL PATH.**
`Flow.StagedMatchesUnstaged_{N1,N2,N3}` now drives **one** cache through a six-tap
sequence that forces fill, **reuse**, and **invalidation**: `(2,-1) (2,-1) (-3,4)
(-3,4) (2,-1) (-3,-1)`. A fresh cache per call would have tested only the fill path,
which is the one that cannot be wrong. **Watched to fail:** a fault that ignores the key
and always reuses makes **exactly 468 of 936** windows differ — the four taps of six that
follow a key change.

**COST: STACK, AND IT IS NOT FREE AT HIGH N.** `TapCache` adds 4 × N × 64 words on top of
`StagedWindow`'s 8 × N × 64. At the **shipped `N = 2` that is 4 KB total**; at the `N = 8`
ceiling it is **≈15 KB**, which is a lot of stack for a Cortex-M and is **stated rather
than hidden**. The shipped ladder is `1/2/2/2`, so the real figure is 4 KB. Both
structures decline above 64 rows rather than overrunning, and **zero heap** either way.

**Decision: SHIPPED**, and [D-61](ARCHITECTURE.md#8-design-decisions) records the
stack-versus-N trade with it. A byte-bounded cap instead of a row-bounded one is
registered as [E-38](ARCHITECTURE.md#register) rather than guessed at here.

**Method:** `benchmark/frontend_sequence.cpp`, V1_02, 900 frames, OpenCV at one thread;
three arms measured back-to-back from `git checkout` of the single header, full rebuild
between each. Gate green on all four configurations, check counts unchanged.

---

### X-71 · E-40 — the input conversion, on both architectures · `DONE`

**COMMITTED BEFORE ANY CODE.**

**Gates:** [E-40](ARCHITECTURE.md#register).
[X-67](#x-67--build-decomposed--and-e-33-is-worth-almost-nothing--done) found
`fromCVMat` at **20.4%** of the x86 frontend; after
[X-69](#x-69--staging-without-vectorising--127-on-track-bit-exact--done)/[X-70](#x-70--the-taps-too--cached-on-the-integer-displacement--done)
shrank `track` it is **33% at one thread and 53% at twelve**, and it does not scale.
**It is the frontend's largest single item.**

**WHAT IT DOES TODAY.** The 1-bit path — the hot one, since level 0 of the shipped
ladder is `QuantMat<1>` — is a **per-pixel branch and read-modify-write**:

```
for (x) if (rowIn[x]) rowOut[wordIndex(x)] |= bitMask(x);
```

A data-dependent branch per pixel, a load-or-store per set pixel, and nothing a
vectoriser can touch. The N > 1 path packs 8 pixels, transposes 8×8 and scatters N
planes — **~34 operations per 8 pixels**, better but still scalar.

**THE OBSERVATION THIS RESTS ON.** Bit-plane extraction from bytes is exactly what a
**move-mask** instruction does. `_mm256_movemask_epi8` takes the top bit of each of 32
bytes and returns 32 bits — **one plane of 32 pixels in one instruction.** So the 1-bit
conversion becomes compare-to-zero, movemask, invert: **three instructions per 32
pixels.** aarch64 has no movemask, but the AND-with-bit-weights plus pairwise-add idiom
gives a 16-bit mask in about six.

**ARMS** (`benchmark/convert_ceiling.cpp`, and the shipped paths):

| | |
|---|---|
| **A** | today: per-pixel branch (N=1) / 8×8 transpose (N>1) |
| **B** | portable branchless — no intrinsics, so every platform gets something |
| **C** | x86 `movemask` |
| **D** | aarch64 NEON bitmask |

**BIT-EXACTNESS IS A PRECONDITION, NOT A BAND.** Every arm must produce **identical
bits** to arm A. This is a repacking, not an approximation — unlike
[X-62](#x-62--e-34--can-the-four-tap-correlations-become-one--done) there is no trade to
weigh, and a mismatch is a bug.

**DECISION RULE, WRITTEN BEFORE MEASURING.**

- **Band A — ≥5× on the conversion AND ≥1.10× on the frontend.** Ship the platform
  paths. The frontend arm is **mandatory**, not confirmatory
  ([D-53](ARCHITECTURE.md#8-design-decisions)).
- **Band B — ≥5× on the conversion but <1.10× on the frontend.** Then the conversion was
  not the constraint the profile said it was, and **that** is the finding — report where
  the frontend time actually went, and ship only the portable arm.
- **Band C — 2–5×.** Ship whichever arms are bit-exact, and **report arm B separately**:
  if the portable branchless arm gets most of it, the intrinsics are not worth their
  maintenance and only B ships.
- **Band D — <2×.** The per-pixel loop was not the problem. Report where the time goes;
  the likely candidate is the **allocation** (`fromCVMat` builds a fresh `Storage` and
  moves it) rather than the packing, which would redirect this to
  [E-37](ARCHITECTURE.md#register)'s ping-pong instead.

**WHY THE ALLOCATION MATTERS AND IS MEASURED SEPARATELY.** `fromCVMat` **allocates a new
buffer every call** — commit-last, for the exception-safety reason its comment gives —
and the frontend calls it **twice per frame, 1710 frames**. An arm that fixes the packing
and leaves the allocation would hit Band D and look like a failure of the *idea* rather
than of the *scope*. So arm A is additionally measured with the allocation hoisted, and
that number is reported whatever the bands say.

**BOTH ARCHITECTURES, AND THE aarch64 HALF IS NOT OPTIONAL** —
[D-62](ARCHITECTURE.md#8-design-decisions) already carries one unmeasured platform.
**Unlike [E-39](ARCHITECTURE.md#register), this displaces no measured optimisation**:
the current path is a scalar per-pixel loop on every platform, so a NEON arm is strictly
additive and can ship on correctness while its speed number waits for a device window.
**That difference is the whole reason one waits and this does not.**

---

#### THE RESULT: BAND A, AND THE PORTABLE ARM ALONE IS 10×

**The arms, 752×480 at 10% set — a real edge map's density, because a dense or empty
image would make arm A's branch predictable and flatter it:**

| arm | ns/frame | vs A | ns/px |
|---|---|---|---|
| **A** shipped: per-pixel branch | 401 487 | 1.00× | 1.1123 |
| **B** portable branchless, **no intrinsics** | 39 131 | **10.26×** | 0.1084 |
| **C** x86 `movemask` | 8 703 | **46.13×** | 0.0241 |
| *(allocate + zero, no packing)* | 477 | — | 0.0013 |

**All bit-identical to A.** And **the allocation is 0.1% of arm A** — X-71 required that
number in advance precisely so a packing win could not be mistaken for a scope failure,
and it rules out Band D's alternative hypothesis outright: **the packing really was the
problem.**

**ARM B IS THE RESULT THAT MATTERS MOST.** Ten times faster **with no intrinsics at
all** — it ships on every target binCV supports, including one with no vector unit, and
it is the *whole* path there. The `46×` needs AVX2; the `10×` needs nothing.

**THE MANDATORY FRONTEND ARM** ([D-53](ARCHITECTURE.md#8-design-decisions)), V1_02, 900
frames, two runs each:

| | `fromCVMat` ms | binCV ms | vs 1-thread OpenCV |
|---|---|---|---|
| old conversion | 0.878 / 0.899 (32%) | 2.725 / 2.828 | 1.13× / 1.11× |
| **new conversion** | **0.051 / 0.064 (3%)** | **1.869 / 2.004** | **1.56× / 1.63×** |
| **new, 4 threads** | 0.055 / 0.053 (5%) | **1.035 / 0.997** | **3.04× / 3.08×** |

**`fromCVMat` 15.5× on the real workload; the frontend 1.43×.** Band A on both halves.
The conversion falls from **32% of the frontend to 3%**.

> **binCV IS NOW 3.06× SINGLE-THREADED OpenCV ON THE CANONICAL SEQUENCE, AT 6.23× LESS
> MEMORY.** [X-67](#x-67--build-decomposed--and-e-33-is-worth-almost-nothing--done)
> projected exactly 3.04× for "conversion excluded" — and now that the conversion is
> nearly free, **the projection and the measurement agree to two decimals.**

**A BUG THE `uint64_t` ARM CAUGHT AND NOTHING NARROWER COULD HAVE.** The first
implementation let the vector loop stop at a multiple of **32** and handed the portable
tail a start that was not word-aligned; at 64-bit words a 32-pixel group is *half* a
word, so the tail's bits landed at the wrong offsets — 336 set where 398 were expected.
Fixed by consuming whole **words** (`kGroup`), and the precondition is now a
`BINCV_ASSERT` rather than a comment. **`-Wconversion` and the four-word-type sweep are
[CLAUDE.md](../CLAUDE.md)'s load-bearing rule and this is what they are for.**

**HOW AVX2 SHIPS WITHOUT AN `-mavx2` BUILD.** `movemask32` is marked
`target("avx2")` and selected by a cached `__builtin_cpu_supports`, so **the baseline
ISA is unchanged** — the project's stated x86 floor is SSE4.2-era
([D-47](ARCHITECTURE.md#8-design-decisions)), and AVX2 is a 2013 part. This is the rule
[X-66](#x-66--e-36--staging-not-gathering-the-x86-keypoint-batch-second-attempt--rule-only)
derived from X-60's failure, applied: **mark one coarse entry point, never leaf
helpers.** Here the marked function *is* the unit of work — load, compare, movemask, 32
pixels — not a helper inside one, so nothing that mattered was prevented from inlining.

#### THE aarch64 HALF, MEASURED

**The arms on the reference device**, `taskset -c 3`, governor `performance`,
`throttled` unchanged:

| arm | ns/px | vs A | *(x86 was)* |
|---|---|---|---|
| **A** shipped: per-pixel branch | 3.3819 | 1.00× | 1.1123 |
| **B** portable branchless | 0.6525 | **5.18×** | 10.26× |
| **D** NEON bitmask | **0.2416** | **14.00×** | 46.13× (AVX2) |

Both bit-identical to A. **Both clear the ≥5× bar on the deployment target**, and the
portable arm does it with no intrinsics — which matters most on the platform where a
Cortex-M variant has no vector unit at all.

**THE FRONTEND ARM**, V1_02, 900 frames, two runs each, OpenCV at one thread:

| | `fromCVMat` ms | binCV ms | OpenCV ms | ratio |
|---|---|---|---|---|
| old conversion | 2.304 / 2.303 (**21%**) | 10.965 / 11.024 | 16.804 / 16.824 | 1.53× |
| **new conversion** | **0.290 / 0.293 (3.2%)** | **9.002 / 9.008** | 16.646 / 16.709 | **1.85×** |

**`fromCVMat` 7.9× on the real workload, the frontend 1.22×, and the ratio against
OpenCV goes 1.53× → 1.85× on the deployment target** — above
[X-38](#x-38--the-full-sequence--done)'s recorded 1.46× and
[X-49](#x-49--the-frontend-after-the-api-swap-a-control-and-a-new-headline--done)'s
1.53×. **Band A on both architectures.**

**The standalone arm reads 14.0× and the frontend 7.9×**, exactly as x86's 46× read
15.5×: the gap is the allocation and the `cv::Mat` row-pointer work around the packing,
not the packing. It is the same ratio on both machines, which is what makes it an
explanation rather than an excuse.

**~~aarch64 SHIPS TOO, AND ITS SPEED IS UNMEASURED.~~ — NOW MEASURED, ABOVE.** NEON has no move-mask, so the arm
ANDs per-lane bit weights and folds sixteen bytes with three pairwise adds. **Unlike
[E-39](ARCHITECTURE.md#register) this displaces no measured optimisation** — the old
path was a scalar per-pixel loop on every platform — so it is strictly additive and
ships on correctness. **The number waits for a device window**, and until then no
aarch64 claim is made.

**Decision: SHIPPED, all three paths.** [D-63](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/convert_ceiling.cpp` for the arms; `benchmark/frontend_sequence.cpp`
for the mandatory frontend arm; V1_02, OpenCV at one thread. Bit-exactness in
`tests/test_opencv_interop.cpp`, across all four word types. Gate green on all four
configurations.

---

### X-72 · E-39 — the staged NEON variants: measured 2.13×, code DEFERRED · `PARTIAL`

**THE MEASUREMENT SUCCEEDED AND THE IMPLEMENTATION DID NOT.** Both halves are recorded,
because the number is worth having and the reason for stopping is worth more.

**WHAT IT IS WORTH ON THE DEPLOYMENT TARGET.** A working prototype — one `RowReader`
serving scalar and NEON, staged and unstaged — was built, **passed all 298 checks
including the staged-vs-unstaged oracle at N = 1/2/3 with 0 declined**, and measured on
the reference device against the shipped hold, same session, two runs each:

| | `track` ms | binCV ms | OpenCV ms | ratio |
|---|---|---|---|---|
| **A** hold in place (shipped) | 7.602 / 7.624 | 8.996 / 9.022 | 16.639 / 16.688 | **1.85×** |
| **B** staging + tap cache active | **6.474 / 6.470** | **7.881 / 7.874** | 16.807 / 16.725 | **2.13×** |

**`track` 1.18×, the frontend 1.14×, and the ratio 1.85× → 2.13× on the reference
device.** Smaller than x86's 1.46× on `track`, which is expected: D-33's and X-40's NEON
accumulators had already removed part of the per-row cost that staging targets.

**AND A REAL x86 FINDING THAT COST TWO WRONG GUESSES.** The first refactor — one body
with a runtime `staged != nullptr` test — cost **17% of `track` on x86** (1.462 → 1.710
ms). Two hypotheses were measured and **both were wrong**:

| change | x86 `track` | |
|---|---|---|
| pointer operands, runtime flag | 1.710 | the regression |
| `__restrict` on the operands | 1.750 / 1.811 | **no help** |
| array operands + copy, runtime flag | 1.745 / 1.780 | **no help** |
| array operands, **compile-time** `Staged` | 1.540 / 1.626 / 1.545 | most of it back |
| **pointers aliasing + compile-time `Staged`** | **1.465 / 1.458 / 1.473** | **parity** |

> **NEITHER CHANGE ALONE SUFFICED.** A runtime flag stops the compiler specialising the
> row loop; copying the operands gives back exactly what staging bought. **Both the
> compile-time flag AND zero-copy aliasing are required**, and that is a fact about the
> shape any future attempt must have.

**WHY THE CODE IS NOT COMMITTED.** Collapsing four extraction blocks into one is a
structural edit to functions that **only compile on aarch64**, and this environment has
**no cross-compiler, no working Docker for `verify_arm.sh`, and a 5–10 minute
round-trip per device build.** The prototype accumulated brace-level damage inside the
`#if BINCV_HAVE_NEON` region that x86 cannot see and that three repair attempts did not
clear. **Continuing to patch blind on a platform that cannot be compiled locally is how
a subtle bug ships**, so the working tree was reverted to the last gate-green state:
x86 `track` 1.459 / 1.476 ms, ratio 1.55–1.56×, 298/298 checks.

**WHAT REMAINS TRUE AND SHIPPED:** the aarch64 hold
([D-60](ARCHITECTURE.md#8-design-decisions)) stays, so the deployment target keeps its
measured NEON accumulators and loses nothing. **This experiment costs nothing and
defers a known 1.15× on the device.**

**WHAT THE NEXT ATTEMPT NEEDS, IN ORDER.** (1) A working `verify_arm.sh` — Docker, or a
cross-compiler — so the NEON region compiles in seconds rather than minutes. (2) The
row reader written **once, whole**, not spliced into existing bodies. (3) `Staged` as a
template parameter and operands as aliasing pointers, per the table above. The
[E-39](ARCHITECTURE.md#register) row records all three.

**Method:** prototype measured on the reference device via `scripts/run_on_pi.sh pi4`,
`taskset -c 3`, governor `performance`, throttle unchanged; x86 arms by `git checkout`
of the single header with a full rebuild between each, V1_02, 900 frames, OpenCV at one
thread.

---

### X-73 · T5.1's threading, on both architectures and at equal thread counts · `DONE`

**[X-65](#x-65--e-35--what-does-threading-the-frontend-buy-and-what-does-it-cost--done)
MEASURED THE IDEA; THIS MEASURES THE SHIPPED API**, and adds the two things X-65 was
missing: the reference device, and OpenCV at a matched thread count.

**REFERENCE DEVICE**, 4 cores, **unpinned** — the protocol deviation X-65 declared,
since a threading arm is unmeasurable under `taskset -c 3`. Governor `performance`,
`throttled=0x80000` unchanged before and after. V1_02, 900 frames:

| threads | `track` ms | speedup | binCV ms | vs **1-thread** OpenCV |
|---|---|---|---|---|
| 1 | 8.168 | 1.00× | 9.772 | 1.95× |
| 2 | 3.120 | 2.62× | 4.679 | 3.94× |
| **4** | **2.213** | **3.69×** | **3.632** | 4.66× |

**`track` scales 3.69× on four cores — better than x86's 2.50×**, which is what a
smaller cache and a simpler core do to a workload that was already memory-light.

#### THE HONEST HEADLINE IS THE EQUAL-THREADS ROW, AND IT IS THE SAME AT BOTH COUNTS

[D-58](ARCHITECTURE.md#8-design-decisions) exists because an unstated thread count
produced a wrong headline once already. So:

| | binCV | OpenCV | ratio |
|---|---|---|---|
| **reference device, 1 thread each** | 9.772 | 19.023 | **1.95×** |
| **reference device, 4 threads each** | 3.633 | 7.065 | **1.94×** |
| *reference device, binCV 4 vs OpenCV 1* | 3.599 | 16.984 | *4.72×* |
| **x86_64, 4 threads each** | 1.082 | 1.462 | **1.35×** |
| *x86_64, binCV 4 vs OpenCV 1* | 1.094 | 3.259 | *2.98×* |

> **1.95× at one thread and 1.94× at four, on the deployment target.** The advantage
> is the implementation, not the parallelism — **both sides scale about equally**, so
> threading moves the absolute numbers and leaves the ratio where it was. **That is a
> better claim than 4.72×**, which mixes a parallelism difference into what reads as
> an implementation one.

**x86 is the weaker of the two at equal threads (1.35×)** and that is the expected
shape: OpenCV's LK is AVX2-vectorised there and NEON coverage on aarch64 is thinner.

**Decision:** the shipped figures are the **equal-thread** ones. The
binCV-4-vs-OpenCV-1 rows stay in the record because they are what a caller sees if
they leave OpenCV at its default — but they are labelled, never quoted bare.
[D-64](ARCHITECTURE.md#8-design-decisions).

**Method:** `benchmark/frontend_sequence.cpp` with `BINCV_LK_THREADS` and
`BINCV_OPENCV_THREADS`; V1_02, 900 frames. Reference device via direct `ssh` rather
than `run_on_pi.sh`, because that script pins with `taskset -c 3` and a nested
`taskset` cannot widen the mask it sets — governor and throttle state checked by hand
in its place. x86 load average 0.38 at start, and the arms were run back to back so a
drift moves both.

---

### X-74 · E-39 done — the staged NEON variants, and the device reaches 2.48× · `DONE`

**[X-72](#x-72--e-39--the-staged-neon-variants-measured-213-code-deferred--partial)
MEASURED THIS AND REVERTED IT**, reporting "no way to compile for aarch64". That was
false — `scripts/check_arm_syntax.sh` compiles the NEON region **in 2.5 seconds** using
the device as a compiler, and with that loop the work is routine.

**The three things X-72 learned the hard way, all obeyed:**

1. **`Staged` is a template parameter.** A runtime `staged != nullptr` per row cost
   **17% of `track` on x86** — the compiler stops specialising the loop.
2. **Operands alias, they do not copy.** Copying them into a per-row struct gives back
   exactly what staging bought.
3. **The reader is written ONCE AND WHOLE.** X-72's damage came from splicing it into
   four existing bodies inside a region x86 never compiles. This time the entire
   region — `StagedWindow`, `TapCache`, `RowReader`, both NEON kernels and the scalar
   one — was replaced in a single edit.

**RESULT, reference device, `taskset -c 3`, V1_02, 900 frames, two runs:**

| | `track` ms | binCV ms | OpenCV ms | ratio |
|---|---|---|---|---|
| staging held off ([X-71](#x-71--e-40--the-input-conversion-on-both-architectures--done)) | 7.602 / 7.624 | 9.002 / 9.008 | 16.6 | 1.85× |
| **staged NEON** | **5.356 / 5.388** | **6.754 / 6.788** | 16.7 | **2.48× / 2.46×** |

**`track` 1.42×, the frontend 1.33×, and the ratio 1.85× → 2.47× on the deployment
target** — and **above X-72's own prototype at 2.13×**, because that one reached the
NEON kernels through a runtime flag rather than a template parameter.

**x86 pays nothing:** `track` 1.498 / 1.510 ms against 1.544 before, ratio 1.54× /
1.53× against 1.57× — inside the run-to-run spread at load 0.48, and if anything
slightly better on `track`.

**298/298 checks pass ON THE DEVICE**, including
`Flow.StagedMatchesUnstaged_{N1,N2,N3}` — which now exercises the NEON paths, since
the aarch64 hold is gone and `stageWindow` accepts at every depth.

**THE HOLD IS LIFTED.** [D-60](ARCHITECTURE.md#8-design-decisions) declined staging on
aarch64 for `N` ∈ {1,2} at `uint32_t` — the shipped ladder's whole depth range —
because the staged path had neither D-33's tap batching nor X-40's accumulator. It has
both now: one `RowReader` serves scalar and NEON, staged and unstaged, and X-41's
**three copies** of the extraction block are **one**.

**Decision: E-39 CLOSED.** [D-65](ARCHITECTURE.md#8-design-decisions).

**Method:** `scripts/check_arm_syntax.sh` as the inner loop; device tests and timing
via `scripts/run_on_pi.sh pi4` (pinned, matching every recorded entry); x86 control on
the same tree, load 0.48.

---

### X-75 · T5.3 — re-baselining the AVX2 batch, and an x86 non-regression · `DONE`

#### THE DOUBT WAS UNFOUNDED: ARM A WAS ALWAYS THE SHIPPED SHAPE

[D-60](ARCHITECTURE.md#8-design-decisions) warned that
[X-66](#x-66--e-36--staging-not-gathering-the-x86-keypoint-batch-second-attempt--rule-only)'s
**2.09×** was "measured against the *unstaged* scalar arm", so the headroom over what
now ships was "smaller than 2.09× and not yet measured". **Reading the arm settles
it:** `armScalar` calls `slicedSignedSum` over a **pre-extracted, per-keypoint
contiguous buffer** — which is exactly what
[X-69](#x-69--staging-without-vectorising--127-on-track-bit-exact--done)/[X-70](#x-70--the-taps-too--cached-on-the-integer-displacement--done)
now ship: staged invariants and a hit tap cache, read through the same
`slicedSignedSum`. **The baseline was already current; the note in D-60 was wrong.**

Re-run on the current tree, three runs:

| arm | vs A | |
|---|---|---|
| **A** scalar over pre-staged data — **the shipped inner loop** | 1.000× | |
| B staged **everything** + per-row emulated popcount | 2.77 – 2.94× | optimistic |
| C staged **everything** + CSA tree | 3.07 – 3.25× | optimistic |
| **D** staged invariants, **taps gathered** | **2.06 – 2.14×** | the implementable one |

**T5.3's bar was ≥1.5× and D clears it at 2.1×**, so the arm is worth writing.

**AND THE ACHIEVABLE FACTOR MAY BE ABOVE D.** Arm D gathers taps every row because the
ceiling has no tap cache. **The shipped path does** — with 4.29 mean iterations
([X-68](#x-68--track-decomposed--915-is-iterated-residualsums--done)) the cache hits
about 77% of the time, so a batched kernel that also caches taps in `[row][plane][lane]`
layout sits **between D and C**. The complication is real and should not be waved away:
**the eight lanes have different integer taps**, so a refresh for one is a scatter into
the batched layout rather than a shared reload.

#### AND AN x86 NON-REGRESSION, BECAUSE "WE WERE FASTER BEFORE" DESERVED A NUMBER

[X-74](#x-74--e-39-done--the-staged-neon-variants-and-the-device-reaches-248--done)
gave the device 1.33× and x86 nothing, which reads like a regression against the 1.63×
recorded after [X-71](#x-71--e-40--the-input-conversion-on-both-architectures--done).
**It is not, and the ratio was the wrong thing to look at:**

| | binCV ms | OpenCV ms | ratio |
|---|---|---|---|
| after X-71 | 1.869 / **2.004** | 2.918 / **3.271** | 1.56× / **1.63×** |
| now | 1.877 / 1.907 | 2.885 / 2.912 | 1.53 – 1.55× |

> **binCV's own time is unchanged — 1.877/1.907 against 1.869/2.004.** The 1.63× came
> from an OpenCV run at **3.271 ms** against its usual 2.9: **the denominator was slow,
> not the numerator fast.** Five runs on the current tree give `track` 1.482–1.530 and
> the ratio 1.53–1.55×, a tighter spread than the difference being questioned.

**E-39 gaining nothing on x86 is correct and expected**, not a cost: x86 has had
staging and the tap cache since X-69/X-70, and E-39's whole content was **bringing them
to the NEON kernels**, which x86 does not have.

**WHERE x86 GOES NEXT, SIZED.** `track` is 79% of the x86 frontend and
[X-68](#x-68--track-decomposed--915-is-iterated-residualsums--done) put iterated
`residualSums` at 91.5% of `track`. At arm D's 2.1×: `track` 1.50 → **0.81 ms**, the
frontend 1.89 → **1.20**, and the ratio **1.54× → ≈2.4×**. That is the prize, it is on
x86 specifically, and [E-36](ARCHITECTURE.md#register) is the only thing left that
reaches it.

**Decision:** T5.3 closed, [E-36](ARCHITECTURE.md#register) confirmed worth writing,
D-60's baseline note corrected. **The frontend arm remains mandatory**
([D-53](ARCHITECTURE.md#8-design-decisions)) — the projection above is Amdahl, not a
measurement.

**Method:** `benchmark/kpbatch_staged_ceiling.cpp`, three runs, load 0.92; frontend
control five runs at load 0.44 on the same tree.

---

### X-76 · How fast are the descriptor family and FAST? · `DONE`

**FIRST ANSWER: ONE WIN AND TWO LOSSES. FINAL ANSWER: THREE WINS**, once the losses
were treated as bugs rather than as facts. 752×480, 256-bit descriptors, OpenCV at one
thread, three runs, load 0.33:

| | binCV | OpenCV | | *(first measured)* |
|---|---|---|---|---|
| **matching**, kNN=2 over 1000×1000 | 1.96–2.04 ms | 9.46–10.18 (`cv::BFMatcher`) | **4.65 – 5.21×** | 4.79× |
| **describe**, 1000 keypoints | **0.134–0.136 ms** | 0.685–0.706 (`cv::ORB`) | **5.02 – 5.26×** | *1.4× SLOWER* |
| **FAST**, 4144 corners | **0.355–0.369 ms** | 0.369–0.373 (`cv::FAST`) | **1.01 – 1.05×** | *3.9× SLOWER* |

> **THESE ARE COMPARISON OPERATIONS, NOT ARITHMETIC ONES, AND BEING SLOWER AT THEM WAS
> NEVER PLAUSIBLE.** FAST is sixteen byte comparisons against two thresholds; BRIEF is
> 256 comparisons; matching is `popcount(a ^ b)`. A library built on bit-parallel
> comparison losing at all three should have been read as a defect on sight, and the
> first version of this entry recorded two of them as findings.

#### WHAT WAS ACTUALLY WRONG, AND NEITHER WAS EXOTIC

**`computeBrief`: 0.983 → 0.134 ms, 7.3×.** The inner loop computed
`q.ay * stride + q.ax` **per pair** — two multiplies inside a 256-iteration loop, half
a million of them for a thousand keypoints, all recomputing the same 512 numbers. The
pattern is now flattened to offsets **once per call**, the bounds test moved from
per-pair to per-keypoint, and the descriptor word accumulated in a register instead of
read-modify-written. **No intrinsics involved.**

**`detectFast`: 1.43 → 0.356 ms, 4.0×, on top of an earlier 16× → 3.9×.** The vector
path exists because **the ring loads are contiguous**: for a horizontal run of 32
pixels, ring position `k` is 32 consecutive bytes at a fixed offset. Sixteen vector
loads per 32 pixels, where the scalar loop paid sixteen scalar loads *per pixel* — a
**32× reduction in loads before any comparison happens**. The contiguity test then runs
*vertically* across the sixteen masks: `run = (run + 1) & mask` resets wherever a ring
pixel fails, and a lane reaching `arcLength` is a corner. No transpose, no per-lane
branch.

Comparisons are unsigned via **saturating arithmetic** (`v > hi` is
`subs_epu8(v, hi) != 0`) rather than the sign-bias ops/pack.hpp uses, which would cost
two extra ops per ring position — and saturation gives the clamp for free: `c + t`
stopping at 255 means "nothing is brighter", which is exactly right.

**Correctness held throughout: `test_fast` still reports 1818 / 1818 with zero on
either side** against `cv::FAST`, through three successive rewrites.

#### AND A MEASUREMENT THAT WAS MEASURING NOTHING

The first run reported `describe` at **21.8 µs, "1.05× OpenCV"**. The source was white
noise at threshold 40, which makes **14% of pixels FAST corners** against about **1.1%**
in a real frame — and the first 1000 corners in scan order were all in the top rows, so
their patches fell **outside the image** and `computeBrief` rejected them after **one of
256 pairs**. The benchmark was timing early exit and calling it a descriptor.

Smoothing to a realistic density moved it from 21.8 µs to 983 µs — **a factor of 45** —
and only then did it measure the operation, which is also when the real defect became
visible. **A benchmark whose input is unrepresentative does not report a pessimistic
number; it reports a number about something else.**

#### THE NEON PATH, AND A COMPASS REJECT BOTH ARCHITECTURES WERE MISSING

**FAST's vector path now exists on aarch64 too**, and NEON makes two parts of it
cheaper than x86: the unsigned compares are **native** (`vcgtq_u8` / `vcltq_u8`, no
sign bias, no saturating-subtract trick), and only the final move-mask needs the
bit-weight fold ops/pack.hpp already carries.

**AND BOTH PATHS WERE PAYING FULL PRICE ON EVERY GROUP.** The scalar loop has rejected
on the four compass points since it was written; the vector versions loaded all sixteen
ring positions and ran the whole run-length loop regardless. **About 1% of pixels on a
real frame are corners**, so nearly every group can be dismissed from four loads — and
dismissing it there skips twelve loads *and* the 24-step loop, which is most of the
function.

**Final, both architectures:**

| | binCV | OpenCV | |
|---|---|---|---|
| **x86 FAST** | **0.346 ms** | 0.368 (`cv::FAST`) | **1.06×** |
| **x86 describe** | **0.128–0.136** | 0.660–0.706 (`cv::ORB`) | **5.0 – 5.3×** |
| **x86 matching** | 1.96–2.04 | 9.46–10.18 (`cv::BFMatcher`) | **4.65 – 5.21×** |
| **device FAST** | 3.04 | 2.92 | **0.96×** — parity, from 0.75× |
| **device describe** | **0.653** | 6.97 | **10.7×** |
| **device matching** | 19.7 | 38.4 | **1.95×** |

**Matching is 4.7× on x86 but 1.95× on the device**, and that asymmetry is
[D-6](ARCHITECTURE.md#8-design-decisions) showing through from the other side: x86 has a
**scalar** `POPCNT` that a Hamming loop over contiguous words uses directly, while
aarch64's `CNT` is a vector instruction whose result must be reduced. The op that most
suits binCV's thesis is the one where the deployment target's ISA helps least.

**Decision: all three ship as strengths on both architectures.**
[E-41](ARCHITECTURE.md#register) is closed. **FAST at parity is the honest outcome** --
`cv::FAST` is a mature vectorised kernel and matching it is the result, not beating it.

**Method:** `benchmark/feature_benchmark.cpp`, seven interleaved repeats, three runs,
OpenCV pinned to one thread per [D-58](ARCHITECTURE.md#8-design-decisions); source
smoothed with a 5×5 Gaussian to a corner density matching a real frame.


---

### X-77 · Where FAST's time actually goes, and why parity is the honest answer · `DONE`

**"WE SHOULD HAVE MORE PARALLELISM THAN OpenCV" — AND FOR THIS OPERATION WE DO NOT.**
The question deserved a measurement rather than an argument, so: sweep the threshold so
the corner density runs from nothing to a third of the image, and see which end binCV
is losing at.

| threshold | corners | binCV | `cv::FAST` | ratio |
|---|---|---|---|---|
| 200 | 0 | 104 µs | 44 µs | **0.42×** |
| 80 | 1 | 96 | 42 | 0.44× |
| **40** | **4144 (1.1%, realistic)** | **407** | **444** | **1.09×** |
| 20 | 48 114 | 1 092 | 1 829 | **1.67×** |
| 10 | 112 741 | 2 477 | 4 602 | **1.86×** |

> **binCV WINS 1.9× WHERE THE CONTIGUITY TEST RUNS AND LOSES 2.4× WHERE ONLY THE
> REJECT RUNS.** At a realistic density the two cancel. **The reject path was the whole
> gap**, and the part that looked like the interesting problem — the arc scan — was
> already the part binCV was better at.

**THE FIX WAS TWO LOADS INSTEAD OF FOUR.** Compass points 0 and 8 are opposite, and
**any window of nine consecutive ring positions contains at least one of them** —
1..9 holds 8, and 9..1 wrapping holds 0. So a group where neither passes cannot contain
a 9-arc, and two loads settle what four were being paid for. Reject path **104 → 53 µs**;
the zero-corner ratio **0.42 → 0.80×**, realistic **1.09 → 1.04×**, high density
**1.86 → 2.02×**.

#### WHY PARITY IS THE CEILING **FOR THIS SIGNATURE** — AND THE FIRST VERSION OF THIS SECTION SAID SOMETHING WRONG

**`detectFast` TAKES `const SrcT* img` WITH A BYTE STRIDE.** Its own header line reads
"FAST corner detection on a **wide image**". So both implementations load the same bytes
into the same 32-byte registers and compare them the same way, and there is no packing
advantage to have.

> **THIS PARAGRAPH ORIGINALLY READ "FAST's input is 8-bit, so binCV has no packing
> advantage" AS THOUGH THAT WERE A PROPERTY OF THE OPERATION. IT IS A PROPERTY OF THE
> SIGNATURE THIS PROJECT CHOSE.** FAST runs perfectly well on a binary image — on
> `{0, 1}` pixels there is exactly one meaningful threshold, and the whole test collapses
> to booleans. Writing binCV's FAST against a byte pointer was a decision, and defending
> its consequence as a ceiling was the error. **See
> [X-80](#x-80--fast-on-a-bit-plane-where-the-thesis-actually-applies) — the corrected
> question is measured there, not argued here.**

**Within the byte signature the finding stands**, and it is the useful half: the
contiguity test is one bit per (pixel, ring position) and is exactly where the 1.9×
comes from, while the reject is byte-shaped and is where the loss was. That is what the
sweep measured and it is why the two-point reject was the fix.

**Final for the wide-image entry point, both architectures:** x86 **1.04×** at realistic
density and **2.02×** at high density; reference device **0.96×**. Correctness unmoved:
**1818 / 1818** against `cv::FAST` through five successive rewrites.

**Decision:** `detectFast` **on a wide image** ships at parity, which for a mature
vectorised byte kernel is the honest result — and it stays, because a caller holding
bytes should not be told to pack them first. **What does not follow is that FAST is
unsuited to binCV**, which is X-80's question.

**Method:** `benchmark/feature_benchmark.cpp` and a threshold sweep; five interleaved
repeats; OpenCV at one thread; x86 load 1.3–1.7 (noted — the realistic-density row moves
±5% with it, which is why the verdict rests on the shape of the sweep and not on one
cell).

---

### X-79 · The AVX2 keypoint batch, with lane refill · `DONE — SHIPPED` · E-36

**THE LAST LEVER ON x86, AND [X-78](#x-78--lockstep-waste-priced-before-the-batch-is-written--rule-pre-registered)
SAID IT ONLY WORKS WITH THE REFILL.** Eight keypoints in AVX2 lanes, staged
`[row][plane][lane]` so eight keypoints' words at the same row and plane are eight
adjacent `uint32_t` — **one load, no gather.** [X-61](#x-61) lost this fight once with
vector arithmetic that WON on operation count and gathers that gave it back; the fix
was never a better gather.

#### THE KERNEL, AND THE TWO THINGS THAT NEARLY MADE IT LOOK 6× FASTER THAN IT IS

| | scalar ×8 | AVX2 batch | ratio |
|---|---|---|---|
| N=1, 31 rows | 1 617–2 023 ns | 468–564 ns | **3.0–3.6×** |
| **N=2, 31 rows (shipped)** | 4 934–5 300 | 1 688–2 970 | **1.8–3.1×** |
| N=2, 15 rows | 2 443–4 584 | 734–1 372 | **3.2–3.3×** |

**The first denominator was wrong twice, and both errors flattered the vector path:**

1. **The scratch build had no `-mpopcnt`.** binCV's own CMake turns it ON by default
   (X-57 measured it worth 3.75×), so a scalar arm without it is not the shipped scalar
   arm — it is calling libgcc's `__popcountdi2`. Reported **6.2–8.4×**; with the flag,
   **0.6–0.7×**, an inversion.
2. **Then the scalar arm was dead-code-eliminated.** Only two of its ten sums fed the
   sink, so the compiler deleted the other eight. **Fixing that alone moved the ratio
   from 0.7× to 3.0×.**

> **A benchmark whose arms differ in what the OPTIMISER can delete is measuring the
> optimiser.** Both numbers were believable, and neither was the answer.

The scalar arm also reads a **contiguous** per-keypoint layout, because that is what
`StagedWindow` actually has. Feeding it the batch's `[row][plane][lane]` arrays would
have strided it across a cache line per word and measured the transpose.

#### WHAT THE VECTOR PATH ACTUALLY DOES, AND WHY IT WINS WITHOUT A POPCOUNT

**AVX2 HAS NO POPCOUNT AT ALL** — `VPOPCNTDQ` is AVX-512 — so the nibble table through
`vpshufb` costs **six operations for thirty-two bytes** against `POPCNT`'s one
instruction for eight. That is a loss per word and a win per keypoint: it covers eight
keypoints, so it is six operations where the scalar path issues eight.

The rest is arranging not to give that back:

- **The counts stay in BYTES until four plane pairs have been folded.** A weighted byte
  is at most `8 + 2·16 + 4·8 = 72` at `N = 2`, so the `2^(i+j)` weighting is done with
  byte adds and the widening happens **once per (row, value, component)** instead of
  once per popcount. Accumulating bytes *across* rows would overflow at 255 and is not
  attempted.
- **The sign is split out of the inner loop** — `P = mag & ~sign`, `N = mag & sign` —
  so nothing goes negative in the byte domain and the subtraction happens once, on
  32-bit lane sums.
- **Ten separate passes over the staged rows, deliberately.** One row loop computing
  all ten sums needs ten accumulators plus both components' split magnitudes plus four
  constants — past sixteen `ymm` registers, and the spill costs more than the extra
  L1 loads do.
- **`target("avx2")` is on ONE function covering the whole window.** X-60 measured a
  leaf-level attribute blocking inlining and costing **1.9×**.

#### THE REFILL, WHICH IS THE DESIGN AND NOT A REFINEMENT

X-78 measured **39.9% of lane slots wasted** by naive lockstep. So a lane that converges
**takes the next untracked point** rather than idling. The refill re-stages one lane —
**the same staging the scalar path does for every point anyway.** The work is not new,
it happens at a different time.

**AND CLIPPED WINDOWS NEED NO SPECIAL CASE, WHICH IS THE PART THAT COULD HAVE BEEN A
MESS.** Lanes in a batch have different heights and the kernel runs the tallest. A short
lane's remaining rows get **zero magnitude**, and `popcount(V & 0)` is zero whatever `V`
is — so a half-clipped window batches with a full one and the answer is unchanged, with
no per-lane row masking anywhere. Column clipping needs even less: the region mask is
applied to `magX`/`magY` once, at staging, and every product is taken against a masked
magnitude.

Windows the batch cannot hold — wider than a word, or taller than the 32-row cap — are
**not skipped, they are tracked by `trackOnePoint` at the moment the refill reaches
them.** The cap is 32 rows because eleven `[row][plane][lane]` arrays at `N = 2` are
**~20 KB at 32 and ~41 KB at 64**, and the working set has to stay in L1 or the
transpose the layout exists to avoid comes back as cache traffic. The shipped window is
31.

#### THE WHOLE-FRONTEND ARM, WHICH [D-53](ARCHITECTURE.md#8-design-decisions) MAKES MANDATORY

Same binary, `BINCV_LK_BATCH=0/1`, five interleaved repeats over 80 V1_02 frames, one
thread each side:

| | batch OFF | batch ON | |
|---|---|---|---|
| `track` (min) | 1.444 ms | **1.054** | **1.37×** |
| `track` (median) | 1.563 | 1.179 | 1.33× |
| frontend (min) | 1.733 | **1.345** | **1.29×** |
| **ratio vs one-thread OpenCV** | **2.10–2.21×** | **2.76–2.88×** | |
| tracks observed | 193 | **193** | unchanged |

**At four threads each side: `track` 1.051 → 0.568 ms, and the headline against a
four-thread OpenCV is 1.67×** (it was 1.35× before this).

> **A 3.1× KERNEL BOUGHT 1.37× ON `track`, AND THE GAP IS THE HONEST PART OF THE
> RESULT.** Inverting Amdahl on the measured numbers puts the vectorised arithmetic at
> **~44% of `track`**, not the ~91% [X-68](#x-68--track-decomposed--915-is-iterated-residualsums--done)'s
> decomposition implied. The other 56% is the scalar half the batch does not touch:
> staging and tap extraction as **scatters** into the lane layout, the covariance, the
> per-lane 2×2 solve. X-68's 91.5% counted the tap extraction inside `residualSums` as
> part of the thing being replaced; only the arithmetic was replaced. **This is the same
> correction X-78 made to X-68's iteration mean, from the other side.**

#### BIT-EXACTNESS, WHICH T5.16 MAKES A PRECONDITION RATHER THAN A BAND

`impl::lkBatchEnabled()` runs both spellings on identical input in one process — the
device `slicedSignedSum`'s `UseNeon` established (X-33) and for the same reason.
**2 208 points on a grid over the whole frame, borders included: 0 positions differ, 0
status, 0 err**, compared as BITS and not to a tolerance.

**And the test was watched to fail**, twice, because one that never has proves nothing:

| injected fault | caught as |
|---|---|
| padded rows given magnitude 1 instead of 0 | **46 positions** differ — and it is 46 rather than 2 208 because only *clipped* windows have padded rows, which is the confirmation that the border cases are in the sample |
| the `(1,1)` plane pair weighted ×2 instead of ×4 | **2 205 positions** differ |

**Method:** `impl/lkBatch_impl.hpp` (kernel plus its scalar oracle),
`trackRangeBatched` in ops/opticalFlow.hpp, `Flow.X79_KeypointBatchIsBitExact`,
`benchmark/frontend_sequence` with `BINCV_LK_BATCH`. Runtime-dispatched on
`__builtin_cpu_supports("avx2")`, so the baseline ISA is unchanged and no `-mavx2` build
is asked of a consumer. aarch64 is untouched and still compiles
(`check_arm_syntax.sh`). **Machine shared with other work** — hence interleaved repeats
and minima, and hence the kernel table's spread.

---

### X-80 · FAST on a bit-plane — where the thesis actually applies · `DONE — SHIPPED` · E-43

**THIS EXPERIMENT EXISTS BECAUSE [X-77](#x-77--where-fasts-time-actually-goes-and-why-parity-is-the-honest-answer)
CONCLUDED SOMETHING IT HAD NOT MEASURED.** X-77 wrote that "FAST's input is 8-bit, so
binCV has no packing advantage" and treated that as a property of the operation. It is a
property of `detectFast(const SrcT*, ...)`, whose own header line says **"on a wide
image"**. Writing binCV's FAST against a byte pointer was a decision; defending its
consequence as a ceiling was the error, and it was **the user who caught it**.

> **ON A ONE-BIT FRAME THE DETECTOR IS BOOLEAN ALGEBRA.** Pixels are `{0, 1}`, so
> `p_ring > p_centre + t` can hold only for `t = 0` with `ring = 1, centre = 0`, and
> `p_ring < p_centre - t` only for `ring = 0, centre = 1`. **There is exactly one
> meaningful threshold** — which is why the bit-plane entry point takes none.

#### THE IDENTITY THAT HALVED THE KERNEL, FOUND ONLY BY WRITING IT OUT

The obvious form is `arc9(ring & ~centre) | arc9(~ring & centre)` — two trees. But a
brighter arc needs a **clear** centre and a darker arc a **set** one, so for any given
pixel they are the same test:

```
    corner  =  arc9( ring XOR centre )
```

**One tree, not two**, and no polarity anywhere in the kernel. Measured: **182 → 145 µs.**

#### AND THE EQUIVALENCE THAT MAKES THIS TIER-1-TIGHT AGAINST OpenCV

For binary content stored as `CV_8U` in `{0, 255}`, `cv::FAST` at **any** threshold in
`[1, 254]` accepts precisely these corners — `255 > 0 + t` holds for every such `t`, and
`0 < 255 - t` likewise. **Checked rather than argued** (`Fast.BitPlaneThresholdIsUnique
OnOneBitContent`): six thresholds from 1 to 254, all 508 corners, zero disagreements.

So the comparison is **corner for corner and in scan order**, not set against set:
**2 860 / 2 860, 4 085 / 4 085, 6 724 / 6 724, 0 mismatched**, over four sizes chosen to
exercise the fully-scalar path, the chunk-plus-leftovers path and the exact-chunk path.

#### THE RESULT

One thread, twelve interleaved rounds, minimum reported, quiet machine:

| content | corners | `cv::FAST` | binCV **wide** | binCV **bit-plane** | input |
|---|---|---|---|---|---|
| EuRoC frame, thresholded | 2 860 (0.8%) | 135.2 µs | 145.8 (0.93×) | **111.8 (1.21×)** | **7.8× smaller** |
| `realframe.bin`, the frontend's own binarised frame | 6 724 (1.9%) | 266.8 | 262.8 (1.02×) | **192.3 (1.39×)** | **7.8× smaller** |

**And on the REFERENCE DEVICE** — Cortex-A72, `taskset -c 3`, performance governor, not
throttled, same frame:

| | `cv::FAST` | binCV wide | binCV **bit-plane** |
|---|---|---|---|
| aarch64, `realframe.bin` | 2 055.8 µs | 2 053.6 (1.00×) | **870.1 (2.36×)** |

> **X-77'S "PARITY IS THE HONEST CEILING" WAS WRONG. ON binCV'S OWN TYPE THE SAME
> DETECTOR IS 1.4× ON x86 AND 2.36× ON THE DEPLOYMENT TARGET, ON A SEVENTH OF THE
> MEMORY, BIT-EXACT.**

**THE DEVICE BEATS x86 AND THE REASON WAS PREDICTED BEFORE IT WAS MEASURED:** the arc
tree needs sixteen live vectors, **aarch64 has thirty-two registers and x86 has
sixteen**. The AVX2 form spends its win on spill traffic; the NEON form does not.
For once the embedded target is where the operation looks best, which is the right way
round for this project.

#### WHAT IT COST TO GET THERE, BECAUSE THE FIRST VERSION LOST BY 5×

**690 → 383 → 235 → 182 → 145 → 112 µs.** Every step was a measurement, and the first
number is the one worth keeping:

| | µs | why |
|---|---|---|
| scalar `uint32_t` words | 690 | **0.20× — SLOWER THAN THE BYTE KERNEL** |
| + AVX2, four-array arc tree | 383 | |
| + in-place tree, ring words kept | 235 | scoring was rebuilding 16 displaced reads per corner-bearing word |
| + compile-time doubling step | 182 | a runtime step makes every index variable, so `v[16]` cannot stay in registers |
| + `ring XOR centre`, one tree | 145 | the identity above |
| + last step folded into the OR | 112 | 16 stores per chunk for a value only ever ORed |

**THE FIRST ROW IS THE WHOLE LESSON. A `uint32_t` HOLDS THIRTY-TWO PIXELS — WHICH IS
EXACTLY WHAT AN AVX2 REGISTER OF BYTES HOLDS.** So bit-packing buys *nothing* against a
vectorised byte kernel until the boolean algebra itself moves into a vector register,
where one `vpand` decides **256** pixels. The packing is not the advantage; **the
packing plus the vector register** is.

**The four-array tree ran at 0.7 operations per cycle** — sixty-four live `__m256i`
against a register file of sixteen. Restructuring to one in-place array of sixteen is
most of the difference between 383 and 145, and it is why the NEON form should fare
better still: **aarch64 has thirty-two vector registers.**

#### WHAT IS NOT CLAIMED

**The score is binCV's own and it costs about 40% of the operation.** With scoring
stubbed out the same frame runs at **1.75×** rather than 1.39×. OpenCV's score — the
largest surviving threshold — is **the same number for every corner** on binary content
and orders nothing, so this reports the **longest qualifying arc**, 9 to 16. That is a
real Tier 2 difference and a real cost, filed as **E-44** rather than optimised here or
quietly dropped: a corner list with no strength ordering is not obviously better.

**Method:** `benchmark/fast_bitplane_benchmark.cpp` (defaults to the committed
`realframe.bin`, so it runs on the reference device with no dataset),
`Fast.BitPlaneMatchesCvFastExactly`, `Fast.BitPlaneThresholdIsUniqueOnOneBitContent`.
The AVX2 path is runtime-dispatched; the NEON path is baseline on aarch64 and was
**watched to fail** under `check_arm_syntax.sh` before being trusted. The vector path
declines the first and last row of the sweep: a bit-plane row has no stride padding
(752 px is exactly 96 bytes) and the ring read touches one byte either side.

---

### X-81 · The bit-plane FAST's score: a crossover, not a winner · `DONE — SHIPPED` · E-44

**[X-80](#x-80--fast-on-a-bit-plane--where-the-thesis-actually-applies--done--shipped--e-43)
LEFT THE SCORE AT ~40% OF THE OPERATION AND FILED IT INSTEAD OF FIXING IT.** This fixes
it. Every corner is returned as `{x, y, score}`, and binCV's score is the **longest
qualifying arc** (9–16) because OpenCV's — the largest surviving threshold — is *the same
number for every corner* on binary content and therefore orders nothing, which is no use
to the non-maximum suppression a detector feeds.

Computing it meant, per corner, pulling one bit out of each of sixteen words to rebuild
that pixel's ring — a bit transpose — then measuring the run. **~78 operations per
corner, and 2% of pixels are corners on a real binarised frame.**

#### THE SCORE FALLS OUT OF THE DETECTION, WHICH ALREADY COMPUTES MOST OF IT

After the three doublings `v[k]` is `AND(diff[k .. k+7])`, and for any `L` in 9..16

```
    AND(diff[k .. k+L-1])  ==  v[k] & v[(k + L - 8) & 15]
```

because two overlapping runs of eight cover any `L <= 16`. So each arc length is one
pass of sixteen ANDs and an OR-reduce — and the masks are **nested**, so a pixel's score
is `8 + the number of masks holding its bit`. **A population count instead of a
transpose**, and `L = 9` is the corner mask the detector wanted anyway.

#### THE MEASUREMENT, AND IT DOES NOT PICK A WINNER

Density swept by moving the binarisation level; one thread, ten interleaved rounds,
minimum, quiet machine:

| corners | density | **A** per-corner transpose | **B** arc masks | **ADAPTIVE** | vs A | scores differ |
|---|---|---|---|---|---|---|
| 246 | 0.07% | **56.9 µs** | 75.7 | 56.8 | 1.00× | 0 |
| 745 | 0.21% | **68.0** | 83.8 | 67.1 | 1.01× | 0 |
| 2 860 | 0.79% | 109.8 | 112.8 | **101.2** | **1.08×** | 0 |
| 9 163 | 2.54% | 195.3 | 157.2 | **147.9** | **1.32×** | 0 |
| 13 809 | 3.83% | 257.1 | 188.7 | **180.4** | **1.42×** | 0 |

> **THE TWO ARMS CROSS AT ABOUT 1% CORNER DENSITY AND EACH LOSES BY UP TO 1.4× ON THE
> WRONG SIDE OF IT.** B's seven extra mask passes are ~217 vector operations per chunk
> **whatever the density**; A's transpose is ~78 scalar operations **per corner**. A
> library that picked one would be 1.4× slow on half its inputs.

**E-44'S REGISTERED PREDICTION WAS "wins only above roughly 3% corner density"; measured,
the crossover is nearer 1%.** The shape was right and the number pessimistic — the
prediction is recorded as written, not adjusted.

So the choice is **per chunk**: `L = 9` is computed first, its population counted, and
the other seven masks produced only if the chunk holds at least three corners. **Never
worse than the better fixed arm, and up to 1.42× better than the shipped X-80 path.**

#### AND IT IS CLUSTERING, NOT DENSITY, THAT DECIDES HOW MUCH IT IS WORTH

On the frontend's own `realframe.bin` — an **edge map**, 1.86% corners — the adaptive
path is worth only **~3%**, against 1.32× on thresholded grayscale at a similar overall
density. **Corners on an edge map are concentrated on the edges**, so most chunks sit
below the threshold and take the transpose path anyway. Density alone does not predict
the gain; how corners are spread across chunks does.

#### THE NEON PORT WAS A REGRESSION AND IT IS NOT SHIPPED

Ported to NEON the same change made the reference device **SLOWER: 2.36× → 2.10×**. The
threshold sweep is what settled it — the loss showed up **even at a threshold that never
takes the mask path**, so it was the restructuring and not the masks:

| device, `realframe.bin` | X-80 | X-81 ported | X-81 with NEON reverted |
|---|---|---|---|
| bit-plane | **870.1 µs (2.36×)** | 978.8 (2.10×) | **863.8 (2.37×)** |

**THE CAUSE IS THE REGISTER FILE, WHICH IS ALSO WHAT MADE X-80 GOOD ON THIS DEVICE.**
The mask form must keep all sixteen `v` vectors **live across up to eight passes**; the
X-80 fold **consumes them in place** and they are dead after. x86 was spilling those
sixteen anyway, so it lost nothing; aarch64's thirty-two registers were holding them,
and that is exactly what X-80's 2.36× was made of.

**A wrong turn on the way, recorded because it cost a device round trip:** the first
diagnosis was the corner-population count, sixteen `__builtin_popcount` calls per chunk
on a machine [D-6](ARCHITECTURE.md#8-design-decisions) says **has no scalar popcount**.
Replacing them with `cnt`/`addv` was correct and changed nothing measurable (978.8 →
971.3, inside noise). **The plausible cause was not the cause**, and the sweep — not the
reasoning — is what found the real one.

> **SO THE TWO BACKENDS KEEP DIFFERENT CODE, BECAUSE THE MEASUREMENT SAID TO.** A change
> that helps one architecture is not thereby an improvement, and this one would have
> shipped as a 10% device regression on the strength of an x86 number.

**Final: x86 up to 1.42× over X-80's path and never worse; the reference device
unchanged at 2.37×.**

**Method:** `fastBitChunk256` (AVX2, adaptive) and `fastBitMask128` (NEON, X-80's fold,
unchanged); `Fast.BitPlaneScoringArmsAgree`; the density sweep above and the threshold
sweep now built into `fast_bitplane_benchmark`, which is how the NEON regression was
located. Quiet machine, load < 0.2; device pinned, performance governor, not throttled.

---


### X-82 · The keypoint batch on aarch64 — and PART A ended the question · `DONE`

**PART A KILLED PART B, WHICH IS WHY IT WENT FIRST.** The rule said decompose the device
frontend before pricing the port, because
[X-67](#x-67)/[D-59](ARCHITECTURE.md#8-design-decisions) had already caught this project
calling a 3.6% stage the biggest target. It caught it again.

#### FIRST, A CORRECTION: x86 HAD NOT OVERTAKEN THE DEVICE

X-82's own premise was wrong. [D-65](ARCHITECTURE.md#8-design-decisions) records the
device at **2.48×**, and that figure was compared against a *freshly measured* x86
2.81×. Measured **today, same harness, same 80 frames, one thread each side**:

| | binCV | OpenCV | ratio | `track` share |
|---|---|---|---|---|
| x86 | 1.345 ms | ~3.87 | 2.81× | 78% |
| **reference device** | **6.945** | **22.014** | **3.17×** | **83.8%** |

**The device was ahead the whole time.** D-65's 2.48× is not wrong, it is *old* — and
comparing a fresh number against a recorded one is exactly what
[D-58](ARCHITECTURE.md#8-design-decisions) exists to forbid. **The error was made and
repeated to the user before it was caught.**

#### PART A — WHERE DEVICE `track` GOES

An iteration-cap sweep first, pinned:

| cap | device `track` | marginal |
|---|---|---|
| 1 | 4.207 ms | — |
| 2 | 5.522 | +1.315 |
| 4 | 5.759 | +0.237 |
| 20 | 5.766 | +0.007 |

Mean iterations is 1.98, so **roughly 45% of `track` is outside the iteration loop** —
and nothing had ever measured which part. So `BINCV_LK_STAGE_TIMING` and
`benchmark/lk_stage_profile`, on the scalar path both architectures share:

| stage | x86 ns/point-level | device | **device / x86** |
|---|---|---|---|
| setup (bounds, clip) | 39.5 | 81.4 | 2.1× |
| staging | 236.0 | 600.4 | 2.5× |
| **covariance + eigen** | **366.3** | **2151.3** | **5.9×** |
| iteration loop | 1369.8 | 4981.9 | 3.6× |

> **THE COVARIANCE IS 27.5% OF DEVICE `track` AND IS 5.9× SLOWER THAN x86 WHERE THE
> ITERATION LOOP — WHICH HAS A NEON PATH — IS 3.6×.** That gap is the whole finding.
> `gradientCovariance` had **no NEON path at all**: `3N^2 + N` **scalar**
> `popcountWord` calls per word, fourteen at `N = 2`, on the one architecture
> [D-6](ARCHITECTURE.md#8-design-decisions) says has no scalar popcount. **binCV was
> breaking its own rule, in the operation sitting next to the one that rule was written
> to enable.**

#### PART B — NOT MEASURED, AND THE RULE SAYS SO

Part A's decomposition makes the keypoint-batch ceiling the wrong next measurement: the
iteration loop is 66% of `track`, the batch's own x86 result converted a **3.1× kernel
into 1.37× on `track`**, and the covariance was sitting untouched at 27.5% with a known
cause and a known fix. **E-45 stays open, unmeasured, and honestly labelled** — the
prediction against it is still only an argument.

---

### X-83 · The covariance gets the NEON kernel D-6 was written for · `DONE — SHIPPED`

**Three guesses preceded this and the scoreboard is worth keeping**, because it is why
`lk_stage_profile` exists:

| change | device `track` |
|---|---|
| `self` term folded into the NEON lanes (a real D-6 violation: four scalar `popcountWord` a row on the full-resolution level) | **1.9%** |
| four accumulators instead of one, to break a 248-long dependent `vmla` chain | **0.0% — reverted** |
| **the covariance NEON kernel, after measuring** | **7.8%** |

The chain was not the bottleneck; the compiler was already scheduling around it. **The
change was reverted rather than kept for tidiness** — it measured zero and cost eight
registers and a combine step.

#### THE KERNEL

Fourteen plane-pair counts a word become **four vectors**, accumulated in lanes to the
end of the window, so the register domain is crossed **once per point per level instead
of `14 x rows` times** — X-40's accumulator shape, applied to the operation next door
that never got it. At `N = 1` the four counts `{xx, yy, xy total, xy set}` are *exactly*
one vector.

**And the second version was 1.5× the first, for a reason worth recording.** The first
built each of the four operand vectors through its own stack array — **sixteen stores
and four loads a word, each load waiting on its stores** — and got only 1.17×.
Everything is now a **shuffle of one vector** `{ax0, ax1, ay0, ay1}`:

- that vector **is already** `xx[0][0]`, `xx[1][1]`, `yy[0][0]`, `yy[1][1]`, because
  `a & a` is `a` and the diagonal needs no AND at all;
- rotating it one lane (`vextq`) puts `ax1` under `ax0` and `ay1` under `ay0`, so lanes
  0 and 2 are the two off-diagonal terms;
- `vzip1q` and a high-half duplicate give `{ax0,ax0,ax1,ax1}` against
  `{ay0,ay1,ay0,ay1}` — every ordered cross pair in one AND.

| | before | after | |
|---|---|---|---|
| covariance, ns/point-level | 2151.3 | **1613.9** | **1.33×** |
| device `track` | 5.766 ms | **5.310** | **1.09×** |
| device frontend | 6.892 | **6.431** | |
| **device vs one-thread OpenCV** | **3.17×** | **3.42×** | |

*(Confirmed on the shipped code after the 0.0% change was reverted: 5.347 → 5.310 and
3.43× → 3.42× across the revert, which is the noise floor — the revert cost nothing, as
the 0.0% measurement predicted it would.)*

**Bit-exact and unmoved:** `test_covariance` **17 704 / 17 704**, `test_opticalflow`
**303 / 303**, **193 tracks** before and after. The portable body is still the oracle
and still compiled everywhere.

**x86 is untouched** — the dispatch is `if constexpr` inside the NEON guard — which is
the point: [X-81](#x-81) had just finished demonstrating that a change helping one
architecture is not thereby an improvement.

**Method:** `benchmark/lk_stage_profile.cpp` (`BINCV_LK_STAGE_TIMING`, off by default),
the cap sweep above, `frontend_sequence` over 80 V1_02 frames. Device pinned to core 3,
performance governor, not throttled. `check_arm_syntax.sh` **watched to fail** on the
new region before it was trusted.

---

### X-84 · Stop reading the same window twice · `DONE — SHIPPED`

**TWO REDUNDANT TRAVERSALS, FOUND BY READING [X-83](#x-83--the-covariance-gets-the-neon-kernel-d-6-was-written-for--done--shipped)'s
PROFILE INSTEAD OF THE CODE.** Neither is a clever kernel; both are the same mistake —
walking memory that had just been walked.

#### 1. THE COVARIANCE WAS RE-READING THE STAGED WINDOW

`stageWindow` extracts `magX`, `magY`, `signX`, `signY` for every row of the window.
That is **exactly and only** what `gradientCovariance` reads. `trackOnePoint` was doing
both, in that order, over the same rows.

**Bit-exact for a reason worth stating: popcounts do not care where a bit sits.** The
staged word is the region extracted to bit 0 and masked; the general path reads it in
place under `visitRowWords`' mask. Every operand is shifted by the same amount, so every
`popcount(a & b)` is the same integer.

The batch path needed it too — x86's shipped path is
[D-66](ARCHITECTURE.md#8-design-decisions)'s keypoint batch, which stages into
`[row][plane][lane]` and would otherwise have got nothing — so `refill` now stages
**before** taking the covariance off the staged lane. A rejected point pays one staging
it does not use, against a whole second traversal for every point that is accepted.

#### 2. THE TAP EXTRACTION WAS READING EVERY `next` ROW TWICE

Row `i`'s lower tap and row `i+1`'s upper tap are the same level row at the same
displacement. `RowReader` read both. It now carries the word forward, halving the reads
of `lv.next` — for **both** architectures, staged and unstaged alike.

#### RESULT

| reference device | before | after | |
|---|---|---|---|
| covariance, ns/point-level | 1613.9 | **492.0** | **3.3×** |
| `track` | 5.310 ms | **4.187** | **1.27×** |
| frontend | 6.431 | **5.309** | |
| **vs one-thread OpenCV** | **3.42×** | **4.17×** | |

| x86 | before | after | |
|---|---|---|---|
| `track` | 1.054 ms | **0.779** | **1.35×** |
| frontend | 1.345 | **1.051** | |
| **vs one-thread OpenCV** | **2.81×** | **3.46×** | |

**Bit-exact everywhere:** `test_covariance` 17 704 / 17 704, `test_opticalflow`
303 / 303, the batched-versus-serial oracle **0 positions differ**, **193 tracks**
unchanged.

> **THE COVARIANCE IS NOW 8.4% OF DEVICE `track`, DOWN FROM 27.5%** — 2151 → 492 ns
> across X-83 and this, **4.4× in total**. Half was the NEON kernel it never had; the
> other half was not calling it on memory already sitting in a stack buffer.

**And the profile has moved:** the iteration loop is now **80.4%** of device `track`,
staging 9.8%, covariance 8.4%, setup 1.4%.

---

### X-85 · The tap layout, and two hoists that did not pay · `DONE — SHIPPED`

[X-84](#x-84--stop-reading-the-same-window-twice--done--shipped) left the iteration loop
at **80.4% of device `track`**. Three things were tried in it. **One shipped.**

#### WHAT SHIPPED: `[plane][tap]`, WHICH IS THE ORDER THE LANES WANT

The NEON kernels put a plane's **four taps in the four lanes** of a vector. `TapCache`
stored them as four separate `[row][plane]` arrays, so every row marshalled them through
a stack array — **eight stores and two loads a row at `N = 2`, each load waiting on its
stores.** Storing `[row][plane][tap]` makes it `vld1q_u32(o.taps[k])`: **two loads, no
stores.**

This is the same store-to-load round trip [X-83](#x-83) found costing 1.5× in the
covariance. **It is the third time this pattern has cost something in this project**, and
it is now worth stating as a rule: *on aarch64, marshalling operands through a stack
array can cost more than the arithmetic they feed.*

The generic scalar fallback transposes the `4 x N` block once a row, because
`slicedSignedSum` wants a value's `N` planes contiguous. That path is not the shipped
ladder's — `1/2/2/2` at `uint32_t` takes the NEON kernels or the AVX2 batch — and `4N`
moves a row is cheaper than the eight stores a row the old layout charged the paths that
*do* ship.

| device | before | after |
|---|---|---|
| iteration loop, ns/point-level | 4710.1 | **4489.3** |
| `track` | 4.187 ms | **4.048** |
| **vs one-thread OpenCV** | **4.17×** | **4.28×** |

> **CORRECTION.** This table first read `4489.3 → 4132`. The **4132 was never measured**
> — the logged pair is 4710.1 before and 4489.3 after, and the "before" cell had been
> shifted as well. The `track` and ratio rows were always right. Corrected against the
> device logs; recorded rather than silently amended, because a number nobody measured
> is worse in this log than no number at all.

#### WHAT DID NOT: HOISTING THE ITERATION-INVARIANT `self` TERM

`b1 = w00*t00 + w01*t01 + w10*t10 + w11*t11 - self`. The four taps move with the
estimate; **`self` is the previous frame against its own gradient and does not depend on
the displacement at all.** It is two of the ten window sums — 20% of the residual
arithmetic — recomputed on every iteration of a loop that averages two. It looks like
free money.

**It is not, and the measurement is the interesting part:**

| device, ns/point-level | |
|---|---|
| iteration loop, self removed | 4489 → **3797** (−692) |
| the hoisted computation | **+897** |
| **net** | **+205, i.e. `track` 4.048 → 4.210 ms** |

**Removing 1.98 evaluations saved 692 ns; adding one back cost 897.** So the in-kernel
evaluation is **2.6× cheaper than the standalone one** — because inside the row loop
`magX`, `signX` and `self` are already in registers, and the hoisted version re-reads
them from the staged buffer for no other purpose.

**The first suspect was wrong, too.** The standalone helper used `slicedSignedSum`, whose
NEON path ends in a `vaddvq_s32` — a horizontal reduce, and 62 of them per point.
Rewriting it with lane accumulators and one reduce changed the number by **5 ns**. The
reduces were not the cost; the loads were.

**On x86 the same hoist measured no change at all**, for a different reason: the batch
computes `self` **eight lanes at a time**, so hoisting it turns vector work into a scalar
per-lane loop. **A hoist is only free when what it removes costs more than what it
adds**, and here it never did on either architecture.

**Reverted.** Both the kernels and the test that had been updated to the new contract.

#### AND THE ONE BEFORE THAT

X-83 also recorded a four-accumulator split of the NEON `vmla` chain at **0.0%**, also
reverted. **Three attempts at the iteration loop, one shipped, and the two that failed
failed for reasons that were not visible without measuring.** The loop is now **79.3% of
device `track`** and remains the largest item.

---

### X-86 · Keep the counts in BYTES to the end of the window · `DONE — SHIPPED`

**THE ITERATION LOOP WAS 80% OF DEVICE `track` AND THE ARITHMETIC WAS NEAR THE MACHINE'S
ISSUE LIMIT, SO THE ONLY WAY DOWN WAS FEWER OPERATIONS.** A counter first, because two
guesses had already missed:

| per point-level, measured | |
|---|---|
| `residualSums` calls | **2.003** |
| tap ROWS extracted | **41.6** (a window is 31, so 1.34 refreshes) |
| tap refreshes per call | 0.671 — X-70's cache absorbs a third |

So tap extraction is ~12% of the loop and **the arithmetic is ~88%**. At ~124 NEON
operations a row it was running at roughly 1.4 per cycle against a Cortex-A72's 2 —
scheduling was not the problem, the operation count was.

#### THE OPERATION COUNT, HALVED

`vcntq_u8` counts **per byte**. Turning that into a per-tap total takes two `vpaddlq`
widenings, and the old kernel paid them **on every row**, then subtracted, shifted and
multiply-accumulated — **eleven operations per plane pair per row**:

```
    b = vand(taps, mag);  sv = vand(b, sign)
    ct = paddl(paddl(cnt(b)));  co = paddl(paddl(cnt(sv)))
    acc = mla(acc, sub(ct, shl(co, 1)), weight)
```

**A byte count is at most 8 and a window is 31 rows, so 248 fits in a byte.** The
widening can wait for the end of the window and the row body collapses to **AND, `cnt`,
byte-add — six operations**:

```
    tX[k] = vaddq_u8(tX[k], cnt(vand(taps, mag)))
    oX[k] = vaddq_u8(oX[k], cnt(vand(vand(taps, mag), sign)))
```

with the widening, the `- 2 x` and the `2^(i+j)` weight applied **once per window**.
Sixteen byte accumulators at `N = 2`, plus four for the previous-frame term —
**aarch64's thirty-two vector registers fit them, and x86's sixteen would not**, which
is why the AVX2 batch stays a different kernel. Rows are flushed every 31 so a taller
window cannot overflow a byte.

This is [X-80](#x-80)'s trick, from the bit-plane FAST, applied to the tracker.

#### RESULT

| reference device | before | after | |
|---|---|---|---|
| `track` | 4.048 ms | **3.683** | **1.10×** |
| frontend | 5.176 | **4.813** | |
| **vs one-thread OpenCV** | **4.28×** | **4.58×** | |

**Bit-exact:** `test_opticalflow` **303 / 303**, **193 tracks** unchanged. x86 is
untouched — this is inside the NEON kernels.

#### AND THE FOURTH INSTANCE OF THE SAME ROUND TRIP

The previous-frame term still built its four operands through a stack array — four ANDs,
four stores and a load that waited on all of them. But `self`, `magX` and `magY` are each
**two contiguous words** in the staged row (and in the unstaged scratch), so a 64-bit load
and two lane moves give `{s0,s1,s0,s1}` against `{m0,m0,m1,m1}` with no store at all.

| reference device | | |
|---|---|---|
| `track` | 3.683 → **3.525 ms** | **1.045×** |
| frontend | 4.813 → **4.648** | |
| **vs one-thread OpenCV** | **4.58× → 4.73×** | |

**That is the fourth time marshalling operands through a stack array has cost this
project something measurable** — the covariance kernel (X-83, 1.5×), the tap layout
(X-85), and twice here. It is now a rule worth stating plainly: **on aarch64, if a vector
operand can be built by a shuffle of something already loaded, building it through memory
will cost more than the arithmetic it feeds.**

**A note on the profiler:** `lk_stage_profile` reports the iteration loop *higher* after
this change while `frontend_sequence` reports `track` lower. The profiler adds four clock
reads per point-level and this kernel's register footprint is larger, so its absolute
numbers are not comparable across builds. **The uninstrumented `frontend_sequence`
number is the result**; the profiler is for shares within one run.

---

### X-87 · The keypoint batch does not transfer to NEON · `DONE — E-45 CLOSES NEGATIVE`

**THE PREDICTION [X-82](#x-82) WROTE AGAINST THIS PORT WAS RIGHT, AND THE MARGIN IS
BIGGER THAN "NO WIN".**

| one 31-row window, `N = 2`, four keypoints, device, 12 interleaved rounds, minimum | |
|---|---|
| **(A)** the shipped kernel, four keypoints in sequence | **4 102.2 ns** |
| **(B)** four keypoints in NEON lanes | **4 825.3 ns** |
| **ceiling on the residual arithmetic** | **0.85×** |

**BAND D. The batched form is 1.18× SLOWER than the one that ships.**

**[D-6](ARCHITECTURE.md#8-design-decisions) IN REVERSE, EXACTLY AS REGISTERED.** x86
needed keypoint batching because **AVX2 has no popcount at all** — the nibble-table
emulation costs six operations per thirty-two bytes and only pays off spread across eight
keypoints. **aarch64 has `cnt`, and the shipped kernel already fills all four lanes with
the four taps.** So one `vand` + `cnt` + byte-add covers **four taps × one plane pair**;
batching keypoints instead makes it **one keypoint set × one (tap, plane pair)** — the
same lane-work, rearranged into **four times as many vector operations**.

It is worse rather than equal because the per-lane form also loses the broadcasts: the
shipped kernel `ld1r`s one magnitude word and reuses it across four taps, while the
batched one must load a different magnitude, sign and tap word **per lane**.

**The denominator honoured all three traps X-82 named in advance:** arm A is the shipped
kernel's shape including [X-86](#x-86)'s byte accumulation, it reads the **contiguous**
per-keypoint layout `StagedWindow` really has, and **all ten of its sums are consumed** —
the omission that flattered a vector arm by 4× in [X-79](#x-79).

**E-45 closes NEGATIVE and the port is not written.** `benchmark/kpbatch_neon_ceiling.cpp`
is committed so the number can be reproduced rather than re-argued.

---

### X-88 · Where the tracker's iteration loop actually stops · `ASSESSMENT`

The user asked to stop when the remaining time is dominated by something an optimisation
cannot move. **It is, and here is the arithmetic rather than the impression.**

At `N = 2` the residual needs `20 N^2 = 80` popcounts of a 31-bit word per window row —
**2 480 bits counted per row**, and that is the algorithm, not the implementation. After
[X-86](#x-86) the kernel issues **20 `vcntq_u8` per row, each counting 128 bits = 2 560
bits.**

> **97% OF EVERY BIT THE KERNEL COUNTS IS A BIT THE ALGORITHM ASKED FOR.** There is no
> packing left to find. On a Cortex-A72 `CNT` is one per cycle, so twenty of them is a
> **twenty-cycle floor per row** before a single AND or accumulate.

The three attempts that remain unexplained by op count were all aimed at scheduling and
all failed: four accumulators to break a dependent `vmla` chain (**0.0%**), hoisting the
iteration-invariant `self` term (**−4%**), and now keypoint batching (**0.85×**). What
*did* work was always **fewer operations or fewer memory round trips** — the byte domain,
the tap layout, the shuffle-built operands, the covariance kernel.

**So the loop is called done at this shape.** Going further needs a different algorithm —
a lower `N`, or a residual that does not need every plane pair — and that is
[E-7](ARCHITECTURE.md#9-open-questions-and-planned-experiments)'s accuracy-versus-depth
question, not a kernel question.

---

### X-89 · T5.8: binCV ships the sensor stage, and the denominator changes with it · `DONE — SHIPPED`

**THE BENCHMARK AND [ARCHITECTURE §7.3](ARCHITECTURE.md#73-edge-filter--threshold)
DISAGREED, AND §7.3 WAS RIGHT.** §7.3 puts the edge filter inside the MVP set;
`frontend_sequence` ran the reference pipeline's median + gradient-threshold stage **in
OpenCV for both sides**, with a comment calling it "deliberately NOT binCV's claim".
binCV has had `medianWide` and `edgeThreshold` since T5.10/T5.11 — **bit-exact against the
reference, 0 of 1219 and 0 of 3367 pixels differing** — and they were tested but never
*used*.

**THIS IS A CHANGE OF DENOMINATOR, NOT A SPEEDUP.** Before, *neither* total included the
sensor stage: binCV was handed a `CV_8U` binary frame and paid `fromCVMat` to unpack it,
OpenCV was handed the same frame free. Now **each side builds its own binary frame from
the same grayscale input**, which is what end-to-end means — and `fromCVMat` leaves
binCV's pipeline entirely, because the edge filter writes bit-planes directly.

**OpenCV's spelling becomes the CONTROL**: `binaryFramesAgree` checks binCV's frame
against it every frame. **BIT-EXACT, 0 pixels differing over 79 frames**, on both
architectures.

#### THE FIRST MEASUREMENT WAS THE POINT OF THE TASK

| x86, first run with the stage wired in | |
|---|---|
| sensor stage | **3.743 ms/frame — 78.4% of the frontend** |
| headline | **3.46× → 0.99×** |

The two kernels were written for correctness and never optimised: a per-pixel sorting
network with three bounds-checked gathers, and an edge predicate re-deriving
`BORDER_REFLECT_101` for every sample.

#### WHAT THEY BECAME

**`medianWide` at K = 3 is min and max and nothing else** —
`med3(a,b,c) = max(min(a,b), min(max(a,b), c))`, five register operations for 32 pixels
on AVX2 or 16 on NEON.

**`edgeThreshold` never forms a byte, even internally.** `|a - b| >= t` on unsigned bytes
is `subs_epu8(a,b) | subs_epu8(b,a)` for the magnitude and `subs_epu8(t, d) == 0` for the
comparison — no widening, no sign, no branch — and `movemask_epi8` turns 32 byte masks
into the 32 bits of an output word **LSB first, which is already binCV's bit order**
(X-71). aarch64 folds 16 masks with bit weights and pairwise adds, the same substitute.

The tail word is taken by an **overlapping window** anchored at `width - 32` and shifted
into place — the padding bits shift in as zero, which is the invariant that would
otherwise need a mask — so only the two border columns per row stay scalar.

| | before | after | |
|---|---|---|---|
| `edgeThreshold`, 752×480, standalone | 861.3 µs | **22.8 µs** | **38×** |
| sensor stage, x86 | 3.743 ms | **0.081** | **46×** |
| sensor stage, device | — | **0.531 ms** (10.9%) | |

#### THE RESULT, AND IT IS A STRONGER CLAIM THAN THE ONE IT REPLACES

| end-to-end, one thread each, 80 V1_02 frames | binCV | OpenCV | ratio |
|---|---|---|---|
| **x86** | **1.072 ms/frame** | 4.46 | **4.16×** |
| **reference device** | **4.879** | 29.633 | **6.07×** |

**binCV absorbed the entire sensor stage for +0.02 ms** on x86 — its frontend went 1.051
(excluding preprocessing) to 1.072 (including it) — while OpenCV's total grew by the
cost of its own. Tracking behaviour unmoved: **1039 / 1037 flow vectors, 193 tracks, 1
re-detection.**

#### A MEASUREMENT ERROR WORTH RECORDING, BECAUSE IT SURVIVED THREE RUNS

`BINCV_EDGE_AVX2` was **never defined** — the `#define` was attached to an include line
`edge.hpp` does not have — so the whole vector block was compiled out. Three successive
"improvements" were measured and reported against it, and **every one of the gains came
from elsewhere** (the median's SIMD, and a cleanup loop that had been visiting 360 000
pixels a frame to skip them).

It was caught by **timing the kernel in isolation**: 861 µs standalone, and `-mavx2` on
the whole translation unit changing the number by 0.8% — which cannot happen if vector
code is running. **A kernel that does not respond to its own ISA flag is not using it.**

---

### X-90 · T5.9: N-bit ingestion without OpenCV · `DONE — SHIPPED`

**AT `N > 1` THE ONLY WAY INTO binCV WAS `QuantMat<N>::fromCVMat`, WHICH TAKES A
`cv::Mat`.** So the core-only build — the whole embedded claim — could receive a **1-bit**
frame (T5.6's `packBits`, `edgeThreshold`, `readPgm`) and **could not receive a 2-bit
one**. `packQuant` closes that.

#### THE RULE HAD TO NOT MOVE, WHICH IS THE HARD PART

`round(v · MaxValue / 255)` is not an arbitrary choice: it is `toCVMatNormalized`'s
**exact inverse**, and [D-42](ARCHITECTURE.md#8-design-decisions) records a deliberate
divergence from OpenCV at bytes 1..127 inside it. Writing it out a second time in core
would have been a load-bearing expression with two spellings and no test between them.

**So it now has ONE definition** — `impl::quantScale` in `impl/kernel_util.hpp` — and
`fromCVMat` builds its lookup table from it. `Pack.QuantScaleReproducesFromCVMatsRule`
pins the two paths equal at **N in {1, 2, 3, 4, 8}: 0 pixels differ.**

**And `transpose8x8` was locked inside `#ifdef BINCV_WITH_OPENCV`** — three delta-swaps
of a `uint64_t`, nothing to do with OpenCV, declared beside the conversions that happened
to use it. **A core-only build could not reach the one primitive that makes N-bit packing
cheap.** The same shape of gap [T5.6](TASKS.md) found with `packRowCmp`. Moved to
`kernel_util.hpp`.

#### THE SCALE BECOMES COMPARISONS, WHICH IS WHY IT VECTORISES

`quantScale` is monotonic, so **the value is the number of thresholds a pixel clears** —
`MaxValue` byte compares, three at `N = 2`. Extracting plane `p` is then one AND, one
compare and one **move-mask**, which is [X-71](#x-71)'s trick with the comparison
swapped. `cmpeq(max(v,t), v)` is `v >= t` on unsigned bytes; `cmpgt_epi8` is signed and
would invert above 127. aarch64 has no move-mask and folds sixteen byte masks with bit
weights, X-71's substitute unchanged.

**A 256-entry lookup table — which is what `fromCVMat` uses — cannot be done in a vector
register at all**, which is why the fast policy is a compile-time `QuantRule` and the
arbitrary map is a separate function.

| 752×480, 10 interleaved rounds, minimum | | x86 | device |
|---|---|---|---|
| **N = 1** | vector vs portable | **10.60×** | **9.50×** |
| **N = 2** | | **8.81×** | **4.35×** |
| **N = 4** | | **4.08×** | **2.60×** |
| **N = 8** | | **1.01×** | **1.00×** |

**N = 8 IS THE LIVENESS CHECK AND IT READS 1.00× EXACTLY AS PREDICTED.** `MaxValue` is
255 there, above the gate, so both arms take the portable transpose. A vector arm that
reports a speedup where its own gate excludes it is not running the code you think it is
— which is precisely how [X-89](#x-89) shipped a block that had been compiled out.

`packQuantWith` — the arbitrary map — is **0.06–0.34×**, i.e. 3–17× slower, and says so
in its docstring. That is the documented cost of a predicate the compiler cannot see.

**Portable path:** `transpose8x8`, eight pixels and **all N planes** per call — ~3
operations per pixel for every plane rather than one bit test per (pixel, plane). It is
also the tail of the vector path, so a width that is not a multiple of 32 costs nothing
extra.

**Method:** `benchmark/pack_quant_benchmark.cpp`, which **links no OpenCV** — that being
the claim it measures. `impl::packQuantSimdEnabled()` switches the arms.
`check_arm_syntax.sh` watched to fail on the new NEON region before it was trusted.

---

### X-91 · The audit: which operations were correct, tested, and never timed · `DONE`

[X-89](#x-89) found `medianWide` and `edgeThreshold` at **78% of the frontend the day
something first called them** — written bit-exact against the reference, benchmarked by
nobody, and therefore unoptimised. **That is not bad luck, it is a hole in the rules:**
CLAUDE.md's benchmarking rules trigger on a performance *claim*, and a kernel with no
caller makes none. So it ships correct and untimed, and nothing notices.

**The rule is now written down** — *a new operation gets a benchmark arm when it is
written, even with no caller* — and this is the sweep that came with it.

#### THE SWEEP

Every header in `ops/` and `io/` against every benchmark. Sixteen of eighteen already had
an arm. The exceptions:

| | state | verdict |
|---|---|---|
| `threshold(cv::Mat, …)` | delegates to `ops/pack.hpp` (X-71 unified it) | **covered** — its vector path is the packer's |
| `binarize<N>` | word-wise already, gathers N plane words per output word | **untimed** |
| `packBits` | vectorised since X-71, but measured only *inside* `fromCVMat` | **untimed as an entry point** |
| `unpackTo8Bit` | one word load per 32 pixels, then a scalar per-bit loop | **untimed** |
| `readPgm` | parses a header and memcpies, once per file | **not on a per-frame path — stated rather than left implicit** |

#### WHAT THE NUMBERS SAID

| 752×480 | x86 | device |
|---|---|---|
| `binarize<2>` | 1.9 µs (0.01 ns/px) | 9.7 µs |
| `binarize<4>` | 4.3 | 18.9 |
| `packBits<GreaterEqual>` | 31.0 | 94.6 |
| **`unpackTo8Bit`, before** | **126.9** | — |
| **`unpackTo8Bit`, after** | **36.3 (3.5×)** | **95.1** |

**Three of the four were already fine and stay untouched** — which is the right outcome
for an audit and worth saying, because a sweep that "finds" something everywhere is
usually finding its own expectations.

**`unpackTo8Bit` was the one.** It was **4× slower than `packBits` doing the same job in
the other direction**, and the reason was thirty-two **branches** per word, not the
shifts. Unpacking is the *inverse of a move-mask* and vectorises the same way: broadcast
the word so byte `i` holds the byte containing bit `i`, AND with per-lane bit weights,
compare, select. Six operations for thirty-two branches.

**It is now symmetric with `packBits` on both architectures** — 36.3 against 31.0 on x86,
95.1 against 94.6 on the device — which is the shape an operation and its inverse should
have, and a better check than any absolute number.

#### AND A GUARD THAT WAS WRONG, CAUGHT BY THE TOOL BUILT FOR IT

The first version gated the AVX2 branch on `BINCV_HAVE_VECTOR_PACK`. **That macro means
"a vector row packer exists" and BOTH backends define it**, so aarch64 compiled the AVX2
branch. x86's whole four-configuration gate passed — `-Werror` and all — because that
region is invisible to it. `check_arm_syntax.sh` caught it in two seconds.

**Third time this session a guard has been wrong in code x86 never compiles**, after
X-89's mis-attached `#define` and X-85's NEON regression. The tool exists precisely
because a third of `ops/opticalFlow.hpp` is in that state; it is worth running on every
edit that touches a `#if`, not only ones that touch NEON intrinsics.

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

---

### X-78 · Lockstep waste, priced BEFORE the batch is written · `RULE PRE-REGISTERED`

**T5.16 NAMES ITS OWN KILL CONDITION AND IT COSTS ALMOST NOTHING TO CHECK.** Eight
keypoints in AVX2 lanes iterate **in lockstep**, so a batch runs until its *last* lane
converges. [X-68](#x-68--track-decomposed--915-is-iterated-residualsums--done) put the
**mean** at 4.29 iterations per point per level — but the batch pays the **maximum over
eight**, and nothing measured so far says what that is.

> **THE WHOLE 2.1× KERNEL WIN IS MULTIPLIED BY `mean / mean-of-max-8`.** If that ratio
> is 0.5, arm D lands at 1.05× and T5.16 is not worth writing. **This is knowable from
> a histogram, before any AVX2 exists.**

**Predicted end-to-end factor on `track`:**

```
    naive lockstep  =  2.1  x  mean(iters) / mean(batch max over 8)
    with lane refill =  2.1  x  mean(iters) / (mean(iters) + refill overhead)
```

**DECISION RULE, WRITTEN FIRST:**

| measured `mean / mean-of-max-8` | band | what gets built |
|---|---|---|
| **≥ 0.70** | **A** | naive lockstep — ≥ 1.47× on `track`, and refill is not worth its complexity |
| **0.45 – 0.70** | **B** | **lane refill**: a converged lane takes the next untracked point instead of idling |
| **< 0.45** | **C** | refill is **mandatory**; naive lockstep would be a regression and must not ship |
| refill also under 1.3× projected | **D** | **T5.16 CLOSES NEGATIVE.** Record it and stop — `track` is already 5.5× faster than where Phase 5 started |

**Measured on the same EuRoC sequences as [X-64](#x-64), at the shipped 1/2/2/2 ladder
and `seal_params.yaml`'s iteration cap**, counting iterations actually executed per
point per level — the loop's own `it`, including the points that exit at 1.

**Instrumentation:** `BINCV_LK_ITERATION_HISTOGRAM`, off by default, writing one
`unsigned` per (level, point). It changes no shipped code path.

#### RESULT · `BAND B` · naive lockstep would waste 40% of every lane slot

**87 246 point-levels over 120 V1_02 frames, at `seal_params.yaml`'s cap of 20.**

| iterations | share | cumulative |
|---|---|---|
| **1** | 14.4% | 14.4% |
| **2** | **58.2%** | **72.6%** |
| 3 | 13.8% | 86.4% |
| 4–19 | 9.8% | 96.4% |
| **20 (the cap — never converged)** | **3.6%** | 100% |

> **THE DISTRIBUTION IS BIMODAL AND THAT IS THE WHOLE PROBLEM.** Nearly three
> quarters of point-levels are finished after **two** iterations, and a 3.6% tail runs
> the cap. Batch eight of those together and **one straggler pins seven converged
> lanes to its own iteration count.**

|  |  |
|---|---|
| mean iterations per point-level | **3.235** |
| mean **maximum** over a batch of 8 | **5.204** |
| **ratio `mean / mean-of-max-8`** | **0.622** |
| lane slots run / useful | 469 424 / 282 213 — **39.9% wasted** |

**Projected `track`: naive lockstep 1.31×, lane refill 2.10×.** **BAND B — build the
refill.** A batch that idles a converged lane throws away two thirds of what the AVX2
kernel earns, and the fix is not an optimisation on top of the batch, it is **part of
the batch's design**: when a lane converges it takes the next untracked point rather
than waiting.

**REFILL IS VERY NEARLY FREE, WHICH IS WHY THE BAND IS ACTIONABLE.** A refill re-stages
one lane's window — and **the serial path stages every point exactly once too**. The
work is the same work at a different time; what refill adds is the interleaved scatter
and the bookkeeping, not a new pass over the data.

#### TWO THINGS THIS CONTRADICTS, BOTH REPORTED RATHER THAN ABSORBED

1. **X-68's 4.29 mean is high. The direct count is 3.235.** X-68 *derived* its figure
   from a timing decomposition — iterated time over single-iteration time — which
   attributes every fixed per-point cost that scales with the loop to the loop. **The
   counter is the better number and X-68's decomposition should be read as an upper
   bound.** Nothing built on X-68 changes: 3.235 iterations still put iterated
   `residualSums` at ~91% of `track`, and the conclusion it gated was *which* stage to
   optimise.
2. **3.6% of point-levels burn 22% of all iterations and converge to nothing.** They
   run the cap and are then almost all dropped. **Lowering the cap is worth its own
   experiment** — it is an accuracy change, not a free one — and it is filed as
   **E-42** rather than taken here.

**Method:** `benchmark/lk_iteration_histogram.cpp`, the frontend and re-detection
schedule of `frontend_sequence` exactly, serial (the hook writes from inside
`parallelFor`). **One sequence only** — V1_02 is the only one on this machine — and
D-58 applies: **the ratio is a property of this sequence's convergence behaviour**, so
the batch's own whole-frontend arm still has to be measured, not projected.

