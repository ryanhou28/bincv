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
| **Cortex-A device** (Pi 4/5, Jetson) | ✅ | ✅ | ✅ authoritative |

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
