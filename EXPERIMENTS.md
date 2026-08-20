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

### X-17 · The LK gradient covariance, fused against composed at T3.6's own level · `PARTIAL`

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

| ID | Question | Task | Runs during |
|---|---|---|---|
| E-4 | Does generic-N regress the specialized paths? | T3.9 | Phase 3 |
| E-7 | Bits needed per pyramid level | T4.1 | Phase 4 |
| E-6 | Hybrid LK versus binary block matching | T4.2 | Phase 4 |
| E-5 | End-to-end accuracy, footprint, speed | T4.3 | Phase 4 |
| E-9 | Per-level word width down the pyramid | — | unscheduled; spun out of [X-10](#x-10--default-word-width--done), which priced both sides |

(E-8 was registered in ARCHITECTURE §9 and never listed here until it was
about to run; it is now closed and has left the table.)
