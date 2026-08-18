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

### X-1 · Row alignment memory cost · `PARTIAL`

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

**Why this is `PARTIAL`:** only the *cost* side was measured. The *benefit* side —
whether aligned rows measurably speed up any NEON kernel — is untested, and there
were no bulk kernels to test it on. D-4 is therefore **provisional** until
[T2.8](TASKS.md) completes it. This is exactly the situation the log exists to
make visible rather than let pass as settled.

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

# Pending

Registered in [ARCHITECTURE §9](ARCHITECTURE.md#9-open-questions-and-planned-experiments),
scheduled as tasks in [TASKS.md](TASKS.md). Each runs **in the phase whose code it
gates**, not at the end.

| ID | Question | Task | Runs during |
|---|---|---|---|
| E-1 | Does alignment beyond word granularity help any kernel? | T2.8 | Phase 2 |
| E-2 | Best default word width on aarch64 | T2.9 | Phase 2 |
| E-3 | Incremental versus recomputed window reductions | T2.10 | Phase 2 |
| E-4 | Does generic-N regress the specialized paths? | T3.9 | Phase 3 |
| E-7 | Bits needed per pyramid level | T4.1 | Phase 4 |
| E-6 | Hybrid LK versus binary block matching | T4.2 | Phase 4 |
| E-5 | End-to-end accuracy, footprint, speed | T4.3 | Phase 4 |
