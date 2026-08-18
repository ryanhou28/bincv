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
combinations was 1.008×, inside its own batch spread. D-4 is confirmed and no
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
currently **provisional**, and the only such decision in the project ·
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

**Environment** (identical for all three runs; `scripts/run_on_pi.sh pi4`):

```
device:           pi4
cpu:              Raspberry Pi 4 Model B Rev 1.5 (Cortex-A72)
arch / kernel:    aarch64 / 6.18.34+rpt-rpi-v8
compiler:         g++ (Debian 14.2.0-19) 14.2.0, Release (-O2 -DNDEBUG)
governor:         performance (restored to ondemand on exit)
core pinning:     taskset -c 3
throttled before / after: throttled=0x0 / throttled=0x0   (every run)
benchmark commit: 3383996
```

**Result** — speedup against word granularity, three runs. **>1.00× means the
wider alignment is faster**, which is the direction the rule reads. Per-run batch
spread never exceeded 7%.

| Size | Alignment | bytes/image | bitwiseAnd | countNonZero |
|---|---|---|---|---|
| 640×480 | 4 (word) | 38400 | 1.000× | 1.000× |
| 640×480 | 16 | 38400 (+0%) | 1.008 / 0.983 / 1.001× | 1.000 / 1.000 / 1.001× |
| 640×480 | 32 | 46080 (+20%) | **0.321 / 0.311 / 0.321×** | 0.999 / 0.999 / 0.998× |
| 640×480 | 64 | 61440 (+60%) | **0.211 / 0.205 / 0.209×** | 0.996 / 0.996 / 0.996× |
| 94×60 | 4 (word) | 720 | 1.000× | 1.000× |
| 94×60 | 16 | 960 (+33%) | 0.988 / 1.000 / 0.993× | 1.009 / 1.000 / 0.990× |
| 94×60 | 32 | 1920 (+167%) | 0.991 / 0.980 / 0.994× | 0.974 / 1.000 / 0.999× |
| 94×60 | 64 | 3840 (+433%) | 0.913 / 0.874 / 0.929× | 0.956 / 0.947 / 0.992× |

**Conclusion:** **No alignment beats word granularity anywhere, on either kernel,
in any run.** The largest number in the table is 1.008×, which sits inside its own
row's 5–6% batch spread — a null result, not a small win. The hypothesis held.

Two of the slowdowns are large enough to be worth naming, and both are
consequences of choosing the alignment rather than confounds:

1. `ops/logic.hpp`'s contiguous fast path requires every stride to equal the words
   a row needs, so **over-aligning disables it**: `bitwiseAnd` at 640×480 runs
   3.1× slower at alignment 32 and 4.8× slower at alignment 64, while using 20%
   and 60% more memory. At 94×60 the width is not a whole number of words, no
   variant takes that path, and the cliff does not appear.
2. Padding words are allocated and never read, so a wider stride spends cache
   lines on nothing — the residual 0.87–0.99× at 94×60.

`countNonZero` has no fast path and walks rows unconditionally, so its column is
the clean isolation of alignment alone: **flat to within 0.5% at 640×480 across
all four alignments.** On this core the alignment effect by itself is zero.

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
benchmark commit: 3383996
```

**Result — speed**, three runs, against `uint32_t`. >1.00× is faster than the
current default.

| Size | Word | bitwiseAnd | countNonZero | batch spread |
|---|---|---|---|---|
| 640×480 | `uint8_t` | 0.959 / 0.962 / 1.065× | 0.247 / 0.247 / 0.247× | 10–17% / 0.1% |
| 640×480 | `uint16_t` | 0.989 / 0.962 / 1.010× | 0.463 / 0.463 / 0.463× | 7–12% / 0.1% |
| 640×480 | `uint64_t` | 0.974 / 0.959 / 1.035× | **1.941 / 1.942 / 1.939×** | 3–13% / 0.2% |
| 94×60 | `uint8_t` | 0.604 / 0.607 / 0.607× | 0.315 / 0.316 / 0.316× | 0.2% |
| 94×60 | `uint16_t` | 0.579 / 0.579 / 0.574× | 0.580 / 0.580 / 0.580× | 0.2% |
| 94×60 | `uint64_t` | **1.285 / 1.285 / 1.285×** | **1.562 / 1.562 / 1.561×** | 0.1% |

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
1.56× at 94×60**, stable to 0.2% across three runs. That clears ">10% on bulk
kernels" outright. `bitwiseAnd` is a null result at 640×480 (0.96–1.07× against a
3–17% batch spread) because it is memory-bound and all four widths move identical
bytes there; at 94×60, where per-row overhead dominates, `uint64_t` wins 1.29×.
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
benchmark commit: 3383996
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
within 2% on every cell; per-run batch spread was 0.0–2.2%):

| Pattern | W | recompute | INC-COL | INC-ROW | INC-COL × | INC-ROW × | scratch |
|---|---|---|---|---|---|---|---|
| SPARSE | 7 | 67.9 | 251.8 | 69.1 | 0.27× | 0.98× | 28 B |
| SEARCH | 7 | 47.2 | 18.1 | 13.2 | 2.61× | 3.58× | 56 B |
| DENSE | 7 | 44.9 | 6.9 | 8.8 | 6.48× | 5.11× | 2560 B |
| SPARSE | 15 | 138.0 | 976.3 | 118.9 | 0.14× | 1.16× | 60 B |
| SEARCH | 15 | 110.8 | 39.7 | 20.4 | 2.79× | 5.44× | 88 B |
| DENSE | 15 | 106.7 | 7.1 | 10.2 | 15.06× | 10.47× | 2560 B |
| SPARSE | **31** | 314.1 | 3879.6 | 238.1 | 0.08× | **1.32×** | 124 B |
| SEARCH | **31** | 269.8 | 104.1 | 36.6 | 2.59× | **7.37×** | 152 B |
| DENSE | **31** | 267.3 | 7.4 | 13.2 | **36.35×** | 20.27× | 2560 B |

Across runs at W=31: SPARSE INC-ROW 1.31/1.32/1.32×, SEARCH INC-ROW
7.31/7.33/7.37×, DENSE INC-COL 36.37/36.35/36.35×.

**Result — axis 2**, ns/keypoint, composed ÷ fused:

| Word | W | composed | fused | run 1 | run 2 | run 3 |
|---|---|---|---|---|---|---|
| `uint32_t` | 7 | 193.7 | 152.9 | 1.26× | 1.27× | 1.27× |
| `uint32_t` | 15 | 432.6 | 344.0 | 1.26× | 1.25× | 1.26× |
| `uint32_t` | **31** | 1011.4 | 796.3 | **1.27×** | **1.27×** | **1.27×** |
| `uint64_t` | 7 | 200.3 | 144.6 | 1.38× | 1.39× | 1.38× |
| `uint64_t` | 15 | 390.7 | 300.0 | 1.30× | 1.31× | 1.30× |
| `uint64_t` | **31** | 822.3 | 637.0 | **1.29×** | **1.29×** | **1.29×** |

**Result — axis 3**, ns/keypoint, uint32_t, and the memory each form needs:

| W | plane (shipped) | four-arg XOR | plane ÷ 4arg |
|---|---|---|---|
| 7 | 109.4 | 152.1 | 0.72 / 0.74 / 0.78× |
| 15 | 247.4 | 328.3 | 0.75 / 0.77 / 0.80× |
| 31 | 580.3 | 731.9 | 0.79 / 0.79 / 0.80× |

Per frame at W=31 and 200 keypoints, **with the plane's formation cost included**:
plane 116.1 µs + 6.5 µs = **122.6 µs**; four-arg **146.4 µs**. The plane is 16%
faster per level even after paying to build it. Memory: the plane is **38400 B per
pyramid level**, held for the frame's lifetime, against **0 B** for the four-argument
form — and for scale, the ternary `dx`/`dy` the covariance reads are 4 planes =
153600 B per level, so the selector is a fifth plane, **+25% on the derivative
working set of every level**.

**Conclusion:**

*Axis 1.* At 31×31 an accumulator beats recompute in **every** pattern — 1.32×
where windows barely overlap, 7.4× and 36× where they do. Past the 15% line in all
three. Which accumulator wins depends on the pattern and they are not
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

*Axis 2.* 1.27× (`uint32_t`) and 1.29× (`uint64_t`) at 31×31, in all three runs,
past the 15% line — reproducing X-8's 1.30× in a separate session and extending it
to two word widths and three window sizes, where it holds from 1.25× to 1.39×. The
popcount count is identical on both sides, so the delta is redundant traversal:
three calls visit and load each region word three times.

*Axis 3.* The two goals disagree, which is why T2.10 set no threshold: the plane is
16% faster per frame and costs 38400 B per level; the four-argument form is 16%
slower and costs nothing.

**Decision:**

1. **Axis 1 → the rule's second branch: extend T2.6 with incremental state before
   T3.6 is written.** INC-ROW is the form to expose — it wins or ties everywhere,
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
[T2.11](TASKS.md#t211--t26-api-extensions-mandated-by-e-3--todo).

E-3 is **resolved** — with the answer that the simpler API does *not* survive,
which is the opposite of what the first branch would have given and is why the
rule was written down first.

---

# Pending

Registered in [ARCHITECTURE §9](ARCHITECTURE.md#9-open-questions-and-planned-experiments),
scheduled as tasks in [TASKS.md](TASKS.md). Each runs **in the phase whose code it
gates**, not at the end.

**E-1, E-2 and E-3 have closed** on the reference device —
[X-9](#x-9--does-row-alignment-earn-its-memory--done),
[X-10](#x-10--default-word-width--done),
[X-11](#x-11--incremental-versus-recomputed-window-reductions--done). Phase 2 has
no open experiment left, and the project has no provisional decision left.

| ID | Question | Task | Runs during |
|---|---|---|---|
| E-8 | Horizontal decimation for `pyrDown`: gather loop or frame-masked unshuffle | T3.4 | Phase 3 |
| E-4 | Does generic-N regress the specialized paths? | T3.9 | Phase 3 |
| E-7 | Bits needed per pyramid level | T4.1 | Phase 4 |
| E-6 | Hybrid LK versus binary block matching | T4.2 | Phase 4 |
| E-5 | End-to-end accuracy, footprint, speed | T4.3 | Phase 4 |
| E-9 | Per-level word width down the pyramid | — | unscheduled; spun out of [X-10](#x-10--default-word-width--done), which priced both sides |

(E-8 was registered in ARCHITECTURE §9 but had never been listed here.)
