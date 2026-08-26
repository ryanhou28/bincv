# Overnight session — 2026-08-21

Plan agreed before starting: **(0)** housekeeping · **(1)** validate the
binarization port against real paired data · **(2)** E-16 / X-27, the level-0
floor · **(3)** T4.3a / E-5 on V1_02_medium's 1710 frames · **(4)** E-13 then
E-12, the inner loop of the two kernels that own 99% of frontend time.

Guardrails: rule before measurement in its own commit; no commit unless
`verify.sh` is green on all four configurations; a measurement that contradicts a
document gets reported, never absorbed; `/mnt/g` is read-only.

---

## 0 · Housekeeping — done

* `T4.1`'s header still said "blocked on E-14"; X-25 resolved that. Corrected.
* **All 3 420 V1_02_medium files decode** (1710 raw + 1710 reference-binarized),
  uniform 752×480 grayscale, timestamps aligned one-to-one. The
  `NOTE V1_02 HAS ERRORS WHILE EXTRACTING` marker sitting beside the directory
  appears to be stale — nothing in the pair set is corrupt.

---

## 1 · The binarization port — TWO FINDINGS, ONE OF THEM A REAL DEFECT

### 1a · THE HARNESS HAS BEEN SKIPPING THE DENOISE STAGE (defect, being fixed)

`SEALProcessor::temporal_process` is **two** stages, and `seal_params.yaml`
enables both:

```
if (cfg.seal_denoiser_on)    median_filter(img, cfg.denoiser_type);   // THREE_PIX_MEDIAN
if (cfg.seal_edge_filter_on) rl_fast_edge_filter_wide(img, cfg.edge_threshold);
```

`tests/test_opticalflow.cpp`'s `referenceEdgeFilter` implements **only the second**.
Every "the reference pipeline's own edge maps" claim in X-20, X-24, X-25 and X-26
was therefore measured on content the reference pipeline does not produce — it is
the right filter applied to an un-denoised frame.

Measured effect at `edge_threshold 17` over 1710 frames: **14.14% set without the
denoise, 13.04% with it.** So the stage is real but small, and the four entries'
comparisons were all *within* one content set — which is why their rankings are
expected to survive. **That is a prediction, and it is being re-run rather than
assumed.**

### 1b · THE `Edge Filtered` DATASET DOES NOT MATCH THE REFERENCE SOURCE, AT ANY THRESHOLD

The paired binarized set was the reason to prefer V1_02_medium, so this matters.
Sweeping `edge_threshold` from 17 to 70, with and without the denoise:

| thr | plain set% | plain diff% | denoise set% | denoise diff% |
|---|---|---|---|---|
| 17 | 14.140 | 10.608 | 13.041 | 9.701 |
| 25 | 8.381 | 5.312 | 7.780 | 5.093 |
| 35 | 4.924 | 3.100 | 4.636 | 3.245 |
| **40** | 3.911 | **2.871** | 3.699 | 3.051 |
| 50 | 2.569 | 3.047 | 2.441 | 3.196 |
| 70 | 1.216 | 3.671 | 1.155 | 3.751 |

The reference set is **4.455% set**. A threshold near 35 reproduces the DENSITY,
but the disagreement never approaches zero — it bottoms out at **2.87%** of pixels
and rises on both sides. **The spatial pattern differs, not just the density**, so
no threshold of `rl_fast_edge_filter_wide` produces that dataset.

**Conclusion: the `Edge Filtered` set's provenance is unknown and it is NOT used as
ground truth.** What IS established is that the port faithfully reproduces the
reference source — it was checked line by line against
`SEAL/src/temporal_processing/edge_filter.cpp`, and the repo's own shipped
`_bin_normalized.png` agrees with it to 0.024% of pixels. The dataset is a third
artefact, possibly from another version or tool, and treating it as truth would
have silently redefined the content every prior entry was measured on.

T4.3a therefore runs on the **raw** V1_02_medium frames through the corrected
two-stage preprocessing, which is self-consistent and matches the reference source.

---

## 2 · E-16 / X-27 — the level-0 floor · **DONE, Band A**

**The representation was never the limit, which is the opposite of what E-16
supposed.** A 31×31 window of 1-bit reference content resolves **29.3 distinct
binary states per pixel of displacement** — floor **0.025 px** noise-free,
**0.10 px** at σ = 1 gray level, **0.174 px** even at σ = 4.

**X-20's 0.25 px criterion therefore stands unchanged.** This was the entry most
likely to end in a widened tolerance and it did not need one.

The rule's proposed method had to be replaced before measuring: it formed
candidates the same way as the observation, so the Hamming-nearest candidate would
have *been* the observation and the floor would have read exactly zero by
construction. The replacement measures the **partition** of displacement into
intervals indistinguishable from the bits, inverting nothing. Bands untouched.

Band D fired: from 11×11 to 41×41 set pixels grow **7.3×** while distinct states
grow only **1.8×**. The crossings lie on connected contours and an edge constrains
only motion perpendicular to itself, so **a bigger window buys almost no
localisation** — window sizing cannot be justified by averaging.

What remains is a factor of **2.5–3 that belongs to the tracker**, now a located
problem: [E-17](ARCHITECTURE.md#register), prime suspect **deviation (i)**, the
integer-grid previous window. Recorded as D-25.

---

## 3 · T4.3a / E-5 — the frontend end to end · **PARTIAL**

EuRoC V1_02_medium, **all 1709 frame pairs**, both frontends on bit-identical
input, each detecting and tracking independently so lifetime is comparable.

| criterion | binCV | OpenCV | verdict |
|---|---|---|---|
| 2 · detection | 193 corners | 200 | agrees to 3.5% |
| 2 · median track lifetime | **11 frames** | 12 | agrees to one frame |
| 2 · per-frame survival | 96.4% | 96.6% | agrees to 0.2 pt |
| 2 · flow | **median 0.0437 px, p90 0.1614** | — | 95.6% within 1 px |
| 3 · peak footprint | **436 704 B** | 2 719 832 B | **MET, 6.23×** |
| 4 · speed | 21.43 ms/frame | 1.54 | **NOT MET, 14× slower** |

Criterion 4 is unmet and is **not restated**. binCV is scalar and single-threaded
against a SIMD, 12-threaded OpenCV. Phase 5 is the answer and the target is already
located: 99% of frontend time sits in two windowed popcount reductions.

Flow is reported as **percentiles, not RMS** — the RMS is 7.03 px and describes
nothing, because the body is at 0.04 px and a ~1% tail is beyond 22 px. That is
X-25's lesson applied to a new measurement rather than re-learned.

**Two harness defects found, and the second nearly became a false binCV finding.**
Flow pairs were matched by array index (two independent detectors share no
ordering — it reported zero comparisons rather than wrong ones, which is how it was
caught). And corner capacity was passed as `maxCorners`, truncating the NMS pool
before the spacing filter: capacity 200 gives **61** corners, capacity 20 000 gives
**193**. The first reading looked like a 3.3× detection shortfall in binCV.
`CornerResult::candidatesTruncated` had been reporting it all along.

---

## 4a · E-13 / X-29 — the per-row accumulator above N = 1 · **DONE, Band A**

**Per-row pays at N = 1 and costs above it**, with the crossover between 1 and 2
rather than somewhere in the middle. Reference device, window-wide vs per-row:

| N | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| W vs P | **0.917×** | **1.114×** | **1.348×** | **1.248×** |

`gradientCovariance<N>` now selects with `if constexpr` — free, since `N` is
already a template parameter — and results are **bit-identical** (same integers,
different order, associative addition). D-15 item 4 amended to be an `N = 1`
statement. It lands on the adopted `1/2/2/2` ladder, where three of four levels run
at N = 2.

**The noise-floor arm is the part worth copying.** X-29 compiled the *same*
algorithm into two translation units, so their spread is pure code layout:

* **Cortex-A72: 0.0–0.3%**
* **x86_64: 0.0–10.6%** — at N = 2 the layout noise is *larger than the entire
  effect*, which reads `IN NOISE` on the laptop and `W wins` on the device.

X-22 declined to close this question on a single-binary A/B and was right to. The
corollary: every prior code-layout caution in this repo (X-22's 1.46×,
`morphology_path_benchmark`'s ~10%) was measured **on x86**, and the device's floor
is an order of magnitude smaller. That does not retire the split-arms discipline —
these numbers are trustworthy *because* the arms were split — but it is a further
reason to prefer the device for A/B work.

---

## 4b · E-12 / X-30 — where the time goes, and Phase 5.1's target list · **DONE**

**E-12's registered target turned out to be worth almost nothing, and that is
reported rather than quietly substituted.** It was registered against T3.5's
derivative (+93% per row) *and* against "every `ops/` kernel with a per-row
prologue". The derivative sits in a build stage worth **0.7%** — eliminating the
whole stage caps the frontend gain at **1.0062×** — so the question was asked of
the 99% instead.

Reference device, 640×480, ladder `1/2/2/2`, 140 keypoints. Splits taken **by
difference**, so nothing is perturbed by a timer inside a loop:

| stage | ms/frame | share |
|---|---|---|
| **corner response sweep** | **30.367** | **52.7%** |
| **LK residual + solve** | **25.182** | **43.7%** |
| LK covariance + setup | 0.833 | 1.4% |
| corner selection | 0.773 | 1.3% |
| build | 0.424 | 0.7% |

**Two functions are 96.5% of the frontend and they are the same kernel shape** — a
windowed popcount reduction, exactly what D-6 reserved the NEON domain for. Phase
5.1 is one piece of work applied twice, recorded as D-27.

**Two of my own results get re-weighted, and I would rather say so than not.**
X-29's accumulator win (last night's work) is 1.114× on a stage worth 1.4% —
**~0.17% end to end**. It was the right answer to E-13 and it was not a big lever.
And D-22's streaming corner ring is confirmed as a *footprint* decision: selection
is 2.5% of detection, the response sweep 97.5%.

---

## Follow-up · criterion 4, with threading separated from SIMD

The 14× figure compared binCV on one thread against OpenCV on twelve, which
conflates "OpenCV has more cores" with "OpenCV has better code". Pinned to one
thread over the same 400 frames:

| | ms/frame | ratio |
|---|---|---|
| binCV (scalar, 1 thread) | 22.82 | — |
| OpenCV, **1 thread** | 3.64 | **6.3× faster** |
| OpenCV, 12 threads | 1.72 | 14× faster |

**Threading is ~2.1× of the gap; SIMD plus algorithm is the remaining 6.3×.** The
like-for-like deficit is **6.3×**, and that is the number to quote. Both are
reported because neither is honest alone — a multi-core denominator is what a user
actually has, a single-thread one is what isolates the code.

---

## 5 · X-31 — the corner response as bit-sliced box sums · **DONE, Band A, with a correction**

**Kernel: 6.98× on the reference device, bit-exact.** Per-pixel 37.93 ms →
bit-sliced 7.89 ms (4.81×) → with the sparsity skip 5.43 ms (6.98×), on real
reference content. `test_corner` passes 3 655 checks unchanged.

The `A + B·bs²` fit predicted 84% addressing overhead; removing it delivered ~5–7×.
The cost model was right about the kernel.

**I nearly discarded the sparsity skip on one frame.** The first real frame I
sampled was 26.67% set — an outlier against the ~13% average — and skipped 1.7% of
words. On three typical frames it skips 22–39% and is worth 1.2–1.45×. Separating
B1 from B2, which the rule required, is what made that visible.

**AND THEN IT MOVED THE FRONTEND BY 1.04%.** X-30's profile timed **one detection
per frame**; the real frontend re-detects on a **3.0% duty cycle** (12 in 399
frames). Detection is **under 2%** of frontend time, not 52.7% — the profile
over-weighted it ~33×. D-27's ordering is corrected by D-28.

**Third time in this project a summary statistic has misdirected effort**, after
X-25's RMS over a tailed distribution and X-24's clipping attribution. Same failure
each time: a number measured on something *adjacent* to the real thing, with
nothing in the number itself to say so.

**What survives:** both hot kernels are addressing-bound, so SIMD is still not the
first move; and the real target is `residualSums` at ~97%, where tap extraction
costs ~9.4 cycles per popcount against a 1-cycle throughput.

---

## 6 · X-32 — S3 rejected, and the SIMD recommendation splits by kernel

**S3 lost.** Deriving `t01` from `t00` and `t11` from `t10` by one shift is
bit-exact (0 of 130 windows differ) and **0.974× — slower**. Not shipped.

**Because the premise was wrong, and the premise was mine.**

| variant | share of `residualSums` |
|---|---|
| taps only, no popcounts | **13.7%** |
| popcounts only, no taps | 18.9% (a floor, not a measurement) |

Tap extraction is **13.7%, not ~90%**. I had measured ~9.4 cycles per popcount,
noted a popcount is 1 cycle throughput, and concluded "90% is addressing". **That
inference does not follow**: the loop issues `20N²` popcounts *and* a comparable
number of masks, ANDs and accumulates — ~240 ops per word at `N = 2`, of which
popcounts are ~33%. 9.4 cycles per popcount is just what a loop with ~5 other ops
per popcount and a long dependency chain looks like. The ratio was real; the
localisation was invented.

**The useful result is that the SIMD answer is per-kernel, and now measured both
ways:**

* **corner response — 84% removable per-pixel overhead** → reformulate, not
  vectorize. Did: **6.98× bit-exact** (D-28).
* **`residualSums` — 13.7%** → no comparable dead weight, work is distributed
  across masks/popcounts/accumulates that all vectorize. **SIMD is the lever here**
  (D-29).

So "should we go SIMD?" has two different right answers depending on which kernel,
and the difference is 84% against 13.7%.

---

## 7 · X-33 — NEON for the sliced signed sum · **DONE, Band B, adopted**

**Ceiling measured first, as X-33's rule required, and it authorised the work:**
batched NEON popcount with lane accumulators against scalar, everything else
stripped — **3.42×**, bit-identical. Above the 1.5× cancel threshold.

**Result: 1.24× on `residualSums`, 1.21× on the LK stage** (25.540 → 21.088 ms),
and LK is 94.7% of the real frontend, so **~1.20× end to end**. Bit-exact: 0 of 130
windows differ, and on-device `ctest` passes `test_opticalflow` with the vector path
live, including the per-pixel oracle at `N = 1..5`.

**The ceiling did its job twice** — it authorised the arm, and it bounded it. The
real kernel gets 1.24× not 3.42× because the popcounts are diluted by tap extraction
(13.7%), masks and accumulator updates. **3.42× is not quoted as the result**; that
would repeat D-28's error exactly.

**D-6 is cashed in for the first time.** It forbade exposing a per-word popcount so
that reductions would be *shaped* to allow batching later. The eight plane-pair
counts had to be inside one function for the domain crossing to collapse from eight
to one — impossible if callers held `popcountWord`.

**Most of the ceiling is still there and its location is known:** the horizontal add
runs once per call, ~620 domain crossings per window. Vector accumulators carried
across the window is the remaining 2–3×, registered as **E-18**.

**The gate caught a build failure my targeted builds could not.** `neon_ceiling.cpp`
compiles its whole body out on x86, leaving `scalarSum` unused and
`-Wunused-function` fatal. `cmake --build --target` never compiled that file on x86;
`verify.sh`'s clean build of every target did.

---

## 8 · X-34 — the straddling window · **DONE, Band A, and it beat its ceiling**

A 31-pixel window at an arbitrary offset spans **1.94 `uint32_t` words** — it fits
in one only when `x0 % 32 ≤ 1`. So `residualSums` was issuing **twice the popcounts
it needed**, each covering 15.5 useful pixels instead of 31.

| | device |
|---|---|
| kernel | **2.13×** |
| LK stage | 21.088 → **11.638 ms** (1.81×) |
| frontend | 22.01 → **13.55 ms/frame** (1.62×) |

Bit-exact — 0 of 130 windows differ, on-device `ctest` green, track lifetime
unchanged at 18 vs OpenCV's 18.

**It beat its own 1.463× ceiling** because the ceiling measured only the word count;
the aligned path also deletes the per-word loop and its head/tail masking. A bound
on one mechanism does not bound a change that removes two.

**vs OpenCV, LK against LK, one thread:**

| ladder | before | now |
|---|---|---|
| `1/2/2/2` | 4.11× slower | **3.08×** |
| `1/1/1/1` | 2.00× slower | **1.34×** |

**At `1/1/1/1` binCV is within 1.34× of SIMD OpenCV while using 8× less memory.**
That is a very different claim from the 14× this sequence started at.

**And it makes the ladder the dominant speed factor.** D-23 chose `1/2/2/2` on
accuracy with its speed cost *estimated* at 1.35×; isolated it is **2.30×**, and it
was chosen when corner detection was believed to be 52.7% of the frontend rather
than 2%. Not reversed — it bought real accuracy — but registered as **E-19**, and it
is now a larger lever than E-18. The intermediate ladders (`1/2/1/1`, `1/2/2/1`)
have never been measured for speed at all.

---

## 9 · X-35 — the tap machinery · **DONE, Band A, and it reaches parity**

The arithmetic was already ahead of OpenCV after X-34 — 0.65 popcounts/pixel at
`N = 1` against ~1.2 SIMD ops — and binCV was still slower. **The whole remaining
gap was machinery: ~5 ops/pixel of addressing around 0.65 ops/pixel of work.**

**Arm T: the `+1` tap is a shift — and X-34 is what made that true.** X-32 tried
this identity and lost at 0.974×, because in the per-word path `t01` needed a bit
from the next word. Aligned, `t01`'s bits are inside the word `t00` already holds.
**A rejected optimisation became correct because an unrelated change moved the
ground under it.**

**Arm I: an interior fast path**, since `displacedRow` built the replicate border
unconditionally for windows that are mostly interior.

**LK stage, reference device, cumulative: 25.540 → 7.421 ms, 3.44×.**

**LK vs LK, OpenCV 1 thread, median of seven repeats on an idle machine:**

| | median ms |
|---|---|
| binCV `1/2/2/2` | 9.819 |
| **binCV `1/1/1/1`** | **4.216** |
| **OpenCV `CV_8U`** | **4.134** |

**Parity — 1.02× — on an eighth of the memory.** From 14× slower at the start of
the session.

**A wrong number was nearly reported.** An earlier run of this comparison, taken
while `verify.sh` was building in the background, said 1.00× — and OpenCV's own time
swung **4.425 → 3.803 → 5.480 ms on identical code**, a 1.44× spread from load
alone, larger than most effects this project measures. The numbers above are medians
of seven repeats at load ~1.2.

**All that remains at the shipped ladder is the ladder.** `1/2/2/2` costs 2.33×,
which is now the entire difference between parity and 2.38× slower. E-19.

---

## 10 · X-36 — the footprint buys no speed, and tap batching · **DONE**

**The important half is the negative result.** LK is **compute-bound**: 33× more
points and 36× more data move the per-point cost **under 13%**. A 31×31 window is
120 bytes at one bit — two to four cache lines either way. **The 8× footprint
advantage does not convert into tracking speed.** This project had been carrying the
opposite assumption implicitly; the two results are independent.

**The optimisation.** The NEON path batched `N²` plane pairs, so at `N = 1` — level
0 of every ladder — it did **nothing**, and that level ran fully scalar on aarch64.
Batching across the four **taps** works at every depth, and D-31's alignment lets the
lane accumulators run the whole window rather than extracting per row.

**1.736× on the kernel, bit-exact.** But **1.04× on the LK stage**, because
`1/2/2/2` has one level at `N = 1` and three at `N = 2`.

**Cumulative, reference device, LK track:**

| ladder | before X-33 | now | |
|---|---|---|---|
| `1/1/1/1` | 20 485.6 µs | **5 479.8** | **3.74×** |
| `1/2/2/2` | 27 571.5 µs | **9 639.6** | **2.86×** |

**The ladder now gates the optimisation, not just the arithmetic** — at `1/1/1/1`
all four levels would take the 1.736×. E-19 grew accordingly.

---

## 11 · X-37 — the comparison on the DEPLOYMENT TARGET reverses the sign

**Every previous reading of criterion 4 was taken on x86, where binCV has no vector
path at all.** X-35's "parity" was binCV **scalar** against OpenCV **SSE**. On the
Pi, binCV has NEON and so does OpenCV (`Baseline: NEON FP16`, verified before
trusting it).

**binCV is faster at every iteration count:**

| iterations | `1/1/1/1` | `1/2/2/2` |
|---|---|---|
| 1 | **11.1×** | 3.4× |
| 4 | **8.4×** | 2.7× |
| 20 | **5.0×** | 1.7× |

**And the advantage is located, not just measured.** Fitting
`T = setup + iters × slope`:

| arm | setup ms | ms/iteration |
|---|---|---|
| binCV `1/1/1/1` | **1.077** | **0.2264** |
| OpenCV `CV_8U` | **13.810** | **0.7065** |

**OpenCV's setup is 12.8× binCV's.** It copies the warped patch into
`IWinBuf`/`derivIWinBuf` — 961 px × 3 shorts per point per level — before iterating.
**binCV copies nothing**; it reads the frame in place.

**That is the data-movement advantage, and X-36 is what makes it legible.** The
kernel is compute-bound, so 8× less data does not speed the *arithmetic* — it
**removes an entire stage**. The two entries only make sense together.

**The harness threw away the first run and was right to**: building OpenCV on four
cores drove the Pi into its soft temperature limit mid-run (`0x0` → `0x80000`).
Re-run at 53 °C with the sticky bit unchanged before and after.

**Iteration count was controlled** — both trackers stop early on their own rules, so
at `maxIterations = 20` they do different work and the ratio would not be of the
kernels.

**Not claimed:** this is LK against LK. The *frontend* comparison has never run on
the device (the EuRoC sequence is not there). E-20.

---

## 12 · X-38 / E-20 — ALL FOUR ROADMAP CRITERIA MET

Whole frontend against OpenCV **on the reference device**, 692 consecutive EuRoC
frames, bit-identical input, OpenCV pinned to one thread, `throttled` unchanged
either side.

| criterion | binCV | OpenCV | |
|---|---|---|---|
| 2 · median track lifetime | **13 frames** | 13 | equal |
| 2 · per-frame survival | **97.1%** | 97.1% | equal |
| 2 · flow difference | **median 0.0386 px** | — | 97.4% within 1 px |
| 3 · peak footprint | **436 704 B** | 2 719 832 B | **6.23×** |
| **4 · speed** | **11.169 ms/frame** | 16.509 | **1.48× FASTER** |

**1.48× faster and 6.23× smaller, simultaneously.**

**Criterion 4 read 14× SLOWER for most of this project's life.** 14× → 6.3× → 3.8×
→ parity → **1.48× faster**. The first four were all measured on **x86, where binCV
has no vector path at all**. The measurements were right; the platform was wrong,
and it took X-37 to notice after four entries had reported the gap as a property of
the library. `frontend_sequence` now prints which case it is in — the fixed
disclaimer it carried had gone false the moment D-30 landed.

**The profile moved and so did the next target:** track 69.6%, **build 25.8%**
(up from 4.5%, because LK got 3.44× faster and `pyrDown` did not), detect 4.6%.
`pyrDown` is now a quarter of the frontend — exactly where E-21's filter design
space lands.

**Caveat:** 692 frames not 1710 — the Windows drive holding the dataset dropped
mid-copy (`/mnt/g` → `d?????????`). Frames are consecutive so lifetimes are intact.

---

## 13 · X-39 / E-21 — the pyramid design space

Filters built with a **reference implementation**, deliberately: the question was
whether the filter matters at all, and if it did not, no bit-sliced kernel needed
writing. Mean yield across six warps:

| filter | N=2 | N=3 | N=5 | N=7 | vs anchor | gain N=2→7 |
|---|---|---|---|---|---|---|
| **`GAUSSIAN_5x5` (anchor)** | 93.10% | 96.22% | **96.87%** | 97.03% | — | **+3.93** |
| `BOX_3x3` | 92.80% | 95.73% | 96.07% | 96.07% | −0.80 | +3.27 |
| `GAUSSIAN_3x3` | 94.43% | 95.73% | 95.58% | 95.58% | −1.28 | +1.15 |
| **`BOX_2x2` (shipped)** | **93.78%** | 94.77% | 94.60% | 94.60% | **−2.27** | **+0.82** |
| `MEDIAN_3x3` | 89.33% | 89.33% | 89.33% | 89.33% | −7.53 | +0.00 |
| `DIRECT_SUBSAMPLE` | 77.18% | 77.18% | 77.18% | 77.18% | **−19.68** | +0.00 |

**THE AXES ARE NOT INDEPENDENT — that is the finding.** `BOX_2x2` gains **+0.82**
from N=2→7; `GAUSSIAN_5x5` gains **+3.93**. A 2×2 box sum has five possible
outcomes, so past 3 bits there is nothing to store. **The filter determines how much
depth is useful**, and every bit-depth result in this project was measured at the
filter that benefits least from depth.

**Band D fires mildly.** The shipped config is 2.27 points below the anchor (up to 5
on large motion), so aliasing is real — but box does not *fail* where Gaussian
succeeds. X-24, X-25 and X-27 stand, with the caveat that they were measured at the
filter least sensitive to their own axis.

**`DIRECT_SUBSAMPLE` confirms the paper on binCV's content**: 63.7% / 59.4% on the
two largest motions against 94–100% filtered — the mechanism behind SEAL §4.2.2's
">2.5 cm worse".

**`MEDIAN_3x3` is the surprise: worse than box and flat in N.** A median of a
mostly-zero neighbourhood returns zero, so it **erodes** a sparse edge map rather
than blurring it. It belongs in the temporal denoiser, where SEAL uses it.

**No decision yet, deliberately**: the bands weigh accuracy against cost, and the
cost side does not exist until the filters have bit-sliced kernels. The points worth
pricing are `GAUSSIAN_5x5 @ N=3` (0.65 below anchor at 3 bits) and `BOX_3x3 @ N=3`.

---

## 14 · X-39 speed axis — the design space closes

Five of six filters are now bit-sliced kernels, **verified exact against a per-pixel
integer reference** at several `(NIn, NOut)` pairs. `MEDIAN_3x3` was deliberately not
implemented — X-39 measured it 7.53 points below the box and flat in N.

| filter | µs | vs shipped | yield vs anchor | est. frontend |
|---|---|---|---|---|
| **`BOX_2x2` (default)** | **93.7** | 1.00× | −2.27 | **11.169 ms, 1.48× faster** |
| `DIRECT_SUBSAMPLE` | 20.9 | 0.22× | −19.68 | 10.978 ms |
| `BOX_3x3` | 398.0 | 4.25× | **−0.80** | 11.968 ms, 1.38× faster |
| `GAUSSIAN_3x3` | 497.7 | 5.31× | −1.28 | *dominated by `BOX_3x3`* |
| **`GAUSSIAN_5x5`** | **2 352.9** | **25.10×** | 0.00 | **17.099 ms — SLOWER** |

**Standard-LK accuracy is reachable and costs criterion 4.** The anchor would put
binCV behind OpenCV. SEAL §4.2.2 reached the same choice by a different route.

**`BOX_3x3` is the point nobody had listed** — 65% of the gap for +0.8 ms, and it
**dominates `GAUSSIAN_3x3`** on both axes.

**Three quarters of every filtered number is framework**: the generic route runs
`BOX_2x2` at 2.96× the hand-written one *computing the same function*. The frontier
is measured on a framework with no optimisation at all (E-22).

`BOX_2x2` stays the default; the set ships as options, and the `BOX_2x2`/`BOX_3x3`
trade is the caller's. D-36.

---

## 15 · The drive comes back — and takes a claim with it

Two things were owed on the dataset. Both are now measured, and **each moved a
number this project had already written down.**

### X-38, full 1710 frames: criterion 2's *parity* is withdrawn

| over | lifetime | survival | flow median | p99 | <1 px |
|---|---|---|---|---|---|
| 692 prefix, first run (`0cde718`) | 13 vs 13 | 97.1 / 97.1 | 0.0386 | 14.478 | 97.4% |
| **692 prefix, control (`82daca6`)** | **13 vs 13** | **97.1 / 97.1** | **0.0386** | **14.478** | **97.4%** |
| **full 1710** | **11 vs 12** | **96.4 / 96.6** | **0.0434** | **22.494** | **95.4%** |

The control reproduces the prefix **exactly**, so nothing regressed — the extra
1018 frames are harder and **both** frontends degrade (OpenCV's own lifetime drops
13 → 12). binCV degrades slightly more: backing the prefix out puts the tail near
**95.9% vs 96.3%** survival, a 0.34-point gap where the prefix had none.

**"Equal" was an artifact of an easy prefix.** It was this project's own claim, in
ROADMAP's banner and D-35's table, so the withdrawal is recorded where the claim
was made. Criterion 2 still holds; **parity is no longer asserted.**

Criteria 3 and 4 are unchanged — 6.23× smaller, **1.46× faster**. binCV lands at
11.169 / 11.195 / 11.198 ms across three runs (0.26% spread) while OpenCV moves
±2.3%, so **the ratio's movement is OpenCV's variance, not binCV's.**

### X-39, sequence arm: the single frame overstated everything by 1.8×–8×

1710 frames, **1.18 M eligible keypoint-cases per cell** against 611 — a 1900×
larger sample. Yield vs the Gaussian anchor at N=3:

| filter | one frame | **1710 frames** |
|---|---|---|
| `GAUSSIAN_3x3` | −1.28 | **−0.37** |
| **`BOX_3x3`** | −0.80 | **−0.10** |
| `MEDIAN_3x3` | −7.53 | −5.07 |
| **`BOX_2x2`** (shipped) | −2.27 | **−1.26** |
| `DIRECT_SUBSAMPLE` | −19.68 | −12.65 |

**Not one arm changed rank** — the decision survived; the numbers did not. Band B.

**`BOX_3x3` IS the Gaussian anchor**, −0.10 points at 1.18 M samples: standard-LK
accuracy in a bit-sliced kernel at a **sixth** of the anchor's cost. That is a
*stronger* result than D-36 first recorded.

**binCV's 2-bit levels give up nothing**: `BOX_2x2` is flat across N=2→7 (+0.02),
while the Gaussian gains +0.73. `1/2/2/2` is not a compromise — it is the right
depth for that filter.

**The per-frame spread is ~7 points, six times the gap being measured.** No single
frame could have decided this, which is exactly why the first table was wrong.

---

## 16 · E-18 answered — and answered negatively

X-40 gave N = 2 the window-carried lane accumulators D-33 gave N = 1. Three of
the four levels of the shipped ladder run at N = 2, so this was the depth doing
most of the tracking.

**Ceiling, the two shapes alone: 1.461×.** Band B → write the arm. Written,
bit-exact, and the gate now proves it (`ResidualNeonMatchesScalar_{N1,N2,N3}`,
728 windows per depth, **0 differ on aarch64**).

**Delivered, in the real kernel:**

| arm | µs | vs shipped |
|---|---|---|
| scalar (`UseNeon=false`) | 842.3 | 0.721× |
| shipped NEON, reduce per call | 607.6 | 1.000× |
| **X-40, reduce per window** | **568.5** | **1.069×** |
| **extraction only, no counting** | **275.6** | **2.205×** |

**The floor arm is the finding.** The per-row tap machinery with the counting
**removed** is **45.4%** of the kernel. So if counting were *free*, the cap is
**2.205×** — and E-18 was chartered on a remaining "2–3×". **That does not exist
in the counting.**

D-29 put tap extraction at **13.7%**. It is now **45.4%** — not because it got
slower, but because D-30, D-31, D-33 and X-35 made the counting ~3× faster and
never touched the addressing. **The same thing that happened to `pyrDown` in
X-38 has happened inside `residualSums`.**

**Two ceilings in a row have overstated the delivered result.** X-33: 3.42× →
1.24×. X-40: 1.461× → 1.069×, and this ceiling was built deliberately close to
the real shape. Even isolating the counting it gained only 1.133×, because *in
situ* the accumulators compete for registers with the tap machinery. **A ceiling
bounds the shape, not the kernel.**

Frontend effect: **1.52× against OpenCV**, from 1.46× — under four percent, and
quoted that way. E-18 resolved negative (D-37); successor **E-23** is the
extraction, whose first task is collapsing the three copies of the block.

---

## 17 · E-23: the extraction is instruction-bound — both hypotheses wrong

X-40 left 45.4% of `residualSums` as extraction with zero counting in it. X-41
asked what that is made of. **Two pre-registered hypotheses, both false.**

| what was removed | result |
|---|---|
| every loop-invariant — both `(w0, s)` descriptors, their `s == 0` case, their bounds test, the `.row(y)` multiplies, the `interior` branch | **1.023×** |
| the memory system — same code on a level small enough that all ten planes fit **L1D** | **1.129×** |

A 31×31 window touches 31 rows of **ten separate planes**: 310 distinct cache
lines, **~19.8 KB fetched for ~2.5 KB of useful bits**. That 8× overfetch is real
— and removing it entirely buys **13%**.

**What binds is the instruction stream.** ~3 660 cycles/window, ~118 per row, for
about a hundred instructions of shifts, ors, masks and border machinery. Neither
address arithmetic (2%) nor memory (13%) is the constraint.

**A prediction withdrawn before it was acted on.** X-41's Band C pre-committed a
successor — *"it is loads, not addressing; the successor is a layout question"*.
The second measurement **tested that prediction instead of adopting it**, and it
is false. Writing the rule down is what made the difference visible.

**Third relocation in a row, and the pattern is now the finding.** D-28 moved the
target detection → tracking; D-35 tracking → `pyrDown`; D-37 counting →
extraction; D-38 finds extraction isn't addressable by either obvious means.
**Every optimisation this project lands relocates the bottleneck rather than
removing it** — which is what a kernel with no remaining stalls looks like from
the inside.

**Budget closed:** 2.205× on this kernel even with counting free, so
`residualSums` cannot take the frontend past **~1.9× against OpenCV**. E-24 (the
twelve scalar extractions per row share two descriptors → could be three vector
ones) is the only lever left, and it is bounded by that.

---

## 18 · E-22: the framework tax was genericity — and it overturns D-36

Three **signatures** changed, no algorithm: `addShifted`'s extents and shift,
`weightedAxis`' tap count/weights/width, `requantizeWeighted`'s divisor. `F` was
**already** a template parameter — the helpers were throwing the constants away.
`test_pyramid` passes with the **identical** 262 322 checks.

| arm | X-39 | **X-42** | speedup |
|---|---|---|---|
| hand-written `pyrDown` *(control)* | 93.7 | 93.8 | 1.00× |
| **generic `BOX_2x2`** | 277.8 | **111.9** | **2.48×** |
| `BOX_3x3` | 398.0 | **228.0** | 1.75× |
| `GAUSSIAN_3x3` | 497.7 | **225.7** | 2.21× |
| **`GAUSSIAN_5x5`** (anchor) | 2 352.9 | **549.8** | **4.28×** |

**The generic route ran `BOX_2x2` at 2.96× the hand-written one; it now runs it at
1.19×.** Band A.

**THE STANDARD-LK ANCHOR IS AFFORDABLE.** D-36 said it *"costs more than it is
worth"* and would put binCV **behind** OpenCV at 0.97×:

| filter | D-36 recorded | **now** |
|---|---|---|
| `BOX_3x3` | 11.968 ms, 1.38× | **11.550 ms, 1.41×** — +0.35 ms, was +0.80 |
| **`GAUSSIAN_5x5`** | **17.099 ms, 0.97× SLOWER** | **12.395 ms, 1.32× FASTER** |

**binCV can have standard-LK pyramid accuracy *and* criterion 4** — 0.14× of speed
for 1.25 yield points. That was the exact trade D-36 declared unavailable.

`BOX_3x3` also no longer *dominates* `GAUSSIAN_3x3` on cost — 228.0 vs 225.7, one
percent the other way. Still preferable, on accuracy, at **equal cost**.

**The caveat was larger than the effect being decided.** X-39 mapped a design space
on an unoptimised framework and two of its four conclusions don't survive. The
registration is what saved it — the number was flagged provisional at the time, so
this is a correction, not a hidden error.

Effect on binCV **as shipped: exactly zero** — the default calls the hand-written
route. What changed is the price of the options. That route is now a deletion
candidate at 1.19× (**E-25**).

---

## 19 · E-24: the shifts are cheap, the gather is not

| arm | µs | vs scalar extraction |
|---|---|---|
| **A — scalar extraction** (shipped) | 254.8 | 1.000× |
| **B — vector, real gather** | 288.0 | **0.885× — SLOWER** |
| **C — vector, gather removed** | 155.6 | **1.638×** |

Twelve scalar load-shift-ors really do become three vector ones, and that's worth
**1.638×**. But `QuantMat` stacks planes, so the eight words a vector wants sit in
eight unrelated cache lines — and **aarch64 has no gather**. Eight loads plus eight
lane inserts cost *more* than the shift-ors they replace. **Arm B is not written.**

**The mechanism was predicted before measuring**, which matters because X-41's
prediction was wrong. The rule named the stacked-plane layout and the missing
gather, and named the consequence — *"the layout forbids it"*, not *"vectorisation
doesn't work"*. Both held.

**This is an instruction-count argument for relayout, not a cache one.** X-41
refuted the cache case at 1.129×. This is a different case for a different reason,
worth 1.638×. The rule pre-registered the distinction so the successor couldn't
inherit that refutation by association.

**Why the prize is smaller than arm C.** The eight words belong to **five separate
containers**. Interleaving within one `QuantMat` gives 2-wide contiguity at best;
the full 1.638× needs an `LKLevelN`'s five containers merged into one allocation —
which makes every single-plane bulk op stride instead of stream. **Arm C is the
ceiling for a design that does not exist.**

### `residualSums` is finished under the current layout

| | |
|---|---|
| cap if counting were free (D-37) | **2.205×** |
| collected by reshaping counts | **1.069×** |
| available from addressing / cache (D-38) | 1.023× / 1.129× |
| available from vectorising extraction as laid out (D-40) | **0.885×** |

Four experiments, one small win, and the remainder sits behind a container
redesign (**E-26**) whose cost side is unmeasured.

---

## 20 · E-26 measured on both sides — and escalated, not decided

| | measured |
|---|---|
| extraction, planar → **real** interleaved (bit-exact) | 255.7 → **177.0 µs, 1.445×** |
| `residualSums` overall | 550.2 → **471.5 µs, 1.167×** |
| conversion, per level per frame | **23.7 µs** |
| **streaming one plane** | 0.605 → **3.129 µs, 5.17× COST** |
| interleaved buffer, largest N=2 level | **92 160 B** |

**Decided:** interleaving does **not** become binCV's general layout. The 5.17×
streaming cost settles it — striding by four uses one word per cache line and
throws the rest away.

**The reusable output — the crossover.** Conversion costs 23.7 µs; each 31-row
window saves 0.605 µs. It pays after **≈40 windows** on a level. The frontend does
~600 per level per frame → amortises **~15×**. *Rule for any future operation:
interleave when a level is re-read more than ~40 windows' worth; stay planar
otherwise.*

**Not decided, and not mine to decide.** Net frontend **~1.65×** from 1.52× —
**+8% speed for +21% peak footprint** (criterion 3: 6.23× → 5.15×). CLAUDE.md makes
these co-equal and gives memory the tie-break absent an explicit choice. **X-44's
bands were written on speed alone** — a defect in my own rule — so the experiment
cannot settle its own question.

**A ceiling that behaved, and a sharper rule.** X-43's fabricated buffer overstated
by only **1.14×**, against D-37's 1.37×. It differed from the real thing only in
where the memory lived. **A ceiling's accuracy tracks how few things it abstracts
away** — better than "ceilings overstate".

---

## 21 · The thesis, as one measurement

`pyrDown` against `cv::pyrDown`, 640×480 → 320×240, OpenCV pinned to one thread:

| arm | µs | **vs `cv::pyrDown`** |
|---|---|---|
| **`cv::pyrDown`, 8U** (denominator) | **517.8** | **1.00×** |
| binCV `BOX_2x2`, **1 → 3** *(shipped)* | **93.8** | **5.52× FASTER** |
| binCV `GAUSSIAN_5x5`, **1 → 3** | 549.7 | 0.94× — **parity** |
| binCV `BOX_2x2`, **8 → 8** | 2 614.3 | 5.0× slower |
| **binCV `GAUSSIAN_5x5`, 8 → 8** | **7 111.7** | **13.7× SLOWER** |

**The crossover is bit width, and it is steep.** Same filter, same image, same
device: 1→3 bits is 549.7 µs, 8→8 is 7 111.7 — **12.9×**. Bit width dominates
filter choice by ~5:1.

**Why it's structural, not an optimisation gap.** A bit-sliced accumulator is
`bits(weightSum × (2^N − 1))` planes wide — 5 and 9 at N=1, **12 and 16 at N=8**.
SIMD pays the same for an 8-bit lane as a 1-bit one. **Bit-slicing buys its
advantage by not paying for bits it doesn't use; at 8 bits there are none to skip.**

**Two things routinely conflated, now separated:** matching OpenCV's *filter* costs
binCV nothing (parity at 1→3, at ⅜ the bits and +1.26 yield points). Matching
OpenCV's *bit width* costs 13.7×.

The 8→8 path is **verified exact** — it's the framework's widest point (12/16-plane
accumulators, divisor 256×255), so if anything overflowed it would be there.

---

## 22 · Interop beats specialisation, 3.7×

X-46 left a question: binCV is 2.5–14× slower than OpenCV above the crossover, so
should wide-`N` cases be **specialised** internally to a byte representation? X-47
built the alternative first and priced both.

`QuantMat<N>` ↔ `cv::Mat` conversion is now first-class for every `N` — previously
`QuantMat<1>` only, so a caller with an 8-bit intermediate had **no way in at all**.
Transpose-based (8 pixels × 8 planes per step), not per-pixel: pricing the round
trip through a naive conversion would have measured the strawman.

| arm | µs |
|---|---|
| **round trip `to` → `cv::pyrDown` → `from`, 8→8** | **1 901.5** |
| native bit-sliced `GAUSSIAN_5x5` 8→8 | 7 101.8 |
| conversion tax, out / back | 953.5 / 1 619.3 |
| `cv::pyrDown` alone (floor) | 514.2 |

**R = 0.27·B — Band A.** Converting out, running OpenCV, and converting back is
**3.7× faster than binCV's own path** at the configuration where binCV is weakest —
cheaper even than the native *box* at 8→8, while computing the Gaussian.

**Specialisation is closed, not deferred.** Its theoretical best saves only the
~1.4 ms tax, once per operation *chain*, and would cost a second storage layout plus
a second implementation of every kernel. The margin cannot fund the machinery.

**The general answer is a formula, not a table:** send an operation to OpenCV when
`native_binCV − native_OpenCV > T`. With X-46's per-width table that settles every
wide-`N` question without another sweep.

Exactness verified before timing: the transpose orientation and round-trip law were
checked numerically *before the C++ existed*; the tests then hold the code to them at
N ∈ {2,3,5,7,8} across four word types, plus the padding-bit invariant and all 256
byte values.

---

## 23 · The review caught me repeating X-44's error

An adversarial review of the X-47 work confirmed **ten findings**, including a real
code bug and — worst of the lot — that **X-47's rule was written on speed alone,
which is the exact defect X-44 reported in its own rule one experiment earlier.**

**Corrections now in the record, not folded away:**

| | |
|---|---|
| **Footprint** | interop **844 800 B** vs native **384 000 B** — **2.20×**, 81% of L2 vs 37%. The interop path materialises a byte-per-pixel frame, which is what binCV exists to avoid. Doesn't reverse Band A (at 8 bits there is no footprint advantage to protect, and the buffers are transient) — **but that is an argument the rule should have made in advance.** |
| **Ordering** | `specialisation (516 µs) < interop (1 906) < native (7 093)`. **A specialisation would be FASTER.** The reason not to build one is the pre-registered cost model, not the clock. The first table let interop look like the speed winner; it isn't. |
| **Agreement** | R and B differ on **1 114 of 76 800 pixels, ZERO interior**, max |Δ| 73/255 — exactly the `BORDER_REFLECT_101` vs zero-fill rim. `measure_util`'s hazard 4 requires this check; the benchmark didn't have one. |
| **Spread** | every median now carries it — all arms under **1%**. Hazard 3; the first version printed medians alone to 0.1 µs. |
| **The tax `T`** | "≈1.4 ms" had **no arm behind it** — it mixed a 640×480 export with a 320×240 import that was only ever hidden inside R. Both now measured; a size-preserving 640×480 op pays **2 569 µs**, not 1.4 ms — 1.8× under-counted. |
| **Causal claim** | "`fromCVMat` is 1.7× slower *because* it allocates" — **withdrawn**, not separable by this design. |

**And a real bug.** `fromCVMat` read `empty() ? DefaultRowAlignment : getRowAlignment()`,
silently downgrading an opt-in row alignment. The guard was redundant, and the trigger
was **buffer reuse** — a moved-from matrix is empty but keeps its alignment, so
`dst = std::move(src); src.fromCVMat(f);` rebuilt at word granularity and dropped a
Tier 2 / DMA stride. No test saw it: every destination in the suite was
default-aligned. Fixed; the regression test **fails on the old code and passes on the
new**, verified by reverting.

Also fixed: three missing API-tier statements (a CLAUDE.md hard rule), the stale
"differs in exactly two observable ways" docstring (it is three now), an
"exact inverse" claim that asserted the false direction, an undocumented
allocate-and-detach, a vacuous half of the padding check, and an untested
empty-matrix branch.

---

## 24 · Reflect-101, and `pyrDown` earns OpenCV's name

The file header had rejected reflect-101 once, as *"a per-pixel index map, and not
word-parallel"*. **That objection applies to one axis and not the other:**

- **Vertical is free** — the filter reads whole rows, so reflecting picks a different
  **row pointer**. The word-parallel body is untouched.
- **Horizontal is genuinely per-pixel** — the objection stands — **but reaches only
  `ceil(Radius/2)` output columns per side**: 1 for the 5×5 Gaussian, **0 on the left**
  for `Box2x2`. Those get the per-pixel definition; the interior keeps the fast path.

The per-pixel definition is `impl::pyrDownPixel`, and **the shipped path calls it** on
the rim — so reference and implementation can't drift apart.

### `pyrDownFiltered<Gaussian5x5, 8, 8, Reflect101>` **IS** `cv::pyrDown`

| source | | source | |
|---|---|---|---|
| 64×48 even | **0 of 768 differ** | 63×47 odd both | **0 of 768** |
| 65×32 odd w | **0 of 528** | 32×65 odd h | **0 of 528** |
| **9×7** — taps fold past **both** edges | **0 of 20** | | |

The 9×7 case is why `reflect101` loops instead of folding once.

### The border costs 9%

| 640×480 → 320×240, 1→3 bits | µs |
|---|---|
| `GAUSSIAN_5x5`, `Zero` (old behaviour) | 549.9 |
| **`GAUSSIAN_5x5`, `Reflect101`** | **599.2 — +9.0%** |
| `BOX_2x2`, `Reflect101` | 116.3 — no rim, no charge |

**Matching OpenCV's filter *and* border costs ~parity** (0.86× of `cv::pyrDown`, at ⅜
the bits). **Matching OpenCV's bit width is what costs 13.7×.** Separate axes, kept
separate.

`Zero` didn't become untested when it stopped being the default — every filter is now
checked under both borders at even *and* odd extents. The odd ones matter: that's where
the last output column reads a source column that doesn't exist, and the even-only test
that shipped before couldn't see it.

---

## 25 · `pyrDown` means `cv::pyrDown` — and the headline moves to 1.53×

| name | what it is |
|---|---|
| **`pyrDown`** | 5×5 `[1,4,6,4,1]` Gaussian, `BORDER_REFLECT_101`. **Exactly `cv::pyrDown`** — Tier 1 at 8→8, proven at even, odd and both-odd extents |
| **`pyrDownBox`** | 2×2 box, `BORDER_REPLICATE` — what `pyrDown` used to be, and binCV's own operating point |
| `Pyramid::build<F, Bo>()` | defaults to the OpenCV pair; the frontend asks for the box explicitly |

**A third border existed and nobody had noticed.** The hand-optimised box route
implements `BORDER_REPLICATE`, not the `Zero` the filtered route used — at 63×47 it
differs from `Zero` in 39 of 768 pixels and from `Reflect101` in 35. Without adding
`Replicate`, dispatching into that route would have silently changed the generic
route's meaning at odd extents.

**It was nearly missed.** My first check used `(x*7+y*13)%2`, which makes adjacent
columns exact complements, so replicate and reflect sums coincide — it reported *zero*
differences everywhere but one corner. **Periodic test data can hide a border bug
completely.**

**Per-filter specialization is now the shape**, which was the right architectural
call: `F` is a closed enum, so `pyrDownFiltered` dispatches at compile time —
`Box2x2`+`Replicate` to the hand-optimised route, everything else generic until
someone measures a case worth specialising. **E-25 answered without deleting
anything**: the fast path is kept and demoted from a competing public API to an
internal dispatch.

**The swap compiled silently at all 25 call sites** — only `test_pyramid` caught it.
So "the migration is complete" had to be measured:

| 692-frame control | before | after |
|---|---|---|
| flow median / p90 / p99, within 1 px | 0.0386 / 0.1177 / 14.478, 97.4% | **identical** |
| lifetime, survival, footprint | 13v13, 97.1%, 6.23× | **identical** |
| **build (`pyrDown` + derivatives)** | 2.884 ms | **2.887 ms** |

**And X-40's forecast confirmed end to end**: it predicted ~1.06× on LK from a 1.069×
kernel gain; measured **7.774 → 7.342 ms, 1.059×**.

### Criterion 4 is now 1.53×

Full 1710 frames: **10.644 ms vs OpenCV's 16.289**, with **every criterion-2 figure
bit-identical to X-38**. Pure speed, no accuracy cost — from the accumulators, not
from the pyramid work.

---

## 26 · E-19: the shipped point was the only one off the frontier

Swept **ladder × filter** on three axes — yield over all 1710 frames (1.18 M
keypoint-cases per cell), build+track on the reference device (spreads **0%**),
exact bytes.

| ladder | filter | build+track | yield | bytes |
|---|---|---|---|---|
| **`1/2/2/1`** | **`BOX_3x3`** | **5 642 µs (−2.4%)** | **94.97% (+0.48)** | **354 720 (−0.8%)** |
| `1/2/2/2` | `BOX_2x2` *(shipped)* | 5 778 | 94.49% | 357 600 |
| `1/2/2/1` | `BOX_2x2` | 4 849 (**−16.1%**) | 93.80% (−0.69) | 354 720 |
| `1/1/1/1` | `BOX_2x2` | 3 311 (−42.7%) | 90.69% (−3.80) | 306 720 |

**Faster, more accurate *and* smaller. No trade.** Of the seven points measured,
**only the shipped one is dominated.**

**D-23 was right on the prices it had.** It fixed the filter at `BOX_2x2` because
`BOX_3x3` cost +0.8 ms; X-42 re-priced it to +0.35 ms by removing a genericity tax
nobody had looked for. Spending level 3's bit to buy the wider filter only became
free when that tax went.

**Filter and depth are substitutes over part of the range** — `BOX_3x3` is worth
+1.32 points at `1/2/1/1`, +1.17 at `1/2/2/1`, +0.78 at `1/2/2/2`, and **−0.02 at
`1/1/1/1`**, because a 1-bit level can't represent the smoother result. Pricing the
two axes separately is what produced a dominated point.

### A methodology finding that nearly inverted the result

The depth benchmark seeded level 0 from a **synthetic lattice**. LK's cost is
dominated by **iteration count**, and on a lattice the coarse levels alias into false
minima:

| track, vs `1/1/1/1` | lattice seed | **real frame** |
|---|---|---|
| `1/2/1/1` | **0.61× — faster with more bits** | 1.38× |
| `1/2/2/1` | 1.19× | 1.46× |
| `1/2/2/2` | 1.34× | 1.77× |

The lattice column is non-monotonic and would have made `1/2/1/1` look like a free
win. The benchmark now seeds from the real binarized frame and **refuses to fall back**
to a synthetic pattern. **A benchmark whose arms differ in convergence needs content
whose convergence is real.** (E-19's own "2.30×" ladder cost was inflated the same way
— it's 1.77× on track.)

**Not yet enacted:** switching re-bases every performance number, so it needs X-49's
treatment first — a frontend re-measure confirming accuracy and re-stating criterion 4.

---

## 27 · The frontend refuted X-50 — and the accuracy harness is why

X-50 said `1/2/2/1` + `BOX_3x3` dominates on all three axes, and required a frontend
confirmation before switching. **That run refuted it.**

| config | within 1 px | lifetime | speed |
|---|---|---|---|
| **`1/2/2/2` + `BOX_2x2`** *(shipped)* | **95.4%** | **11** | **10.644 ms** |
| `1/2/2/2` + `BOX_3x3` | 95.2% | 11 | 10.879 |
| **`1/2/2/1` + `BOX_3x3`** *(X-50's winner)* | **90.6%** | **9** | 10.787 |

Both of X-50's accuracy claims fail in the same direction:

| | harness said | frontend says |
|---|---|---|
| `BOX_3x3` at `1/2/2/2` | **+0.78** points | **−0.2**, and +0.235 ms |
| dropping level 3's bit | −0.69 points | **−4.6**, lifetime 11 → 9 |

**The mechanism is a harness defect, not noise.** `seedFiltered` builds the pyramid
**entirely in floating point** and quantizes each level *from the float chain*.
binCV's pyramid quantizes level 1, then filters **that quantized level** to make
level 2. **The harness models a pyramid with no cascaded quantization error; the
shipped one has three rounds of it** — so it systematically **understates the cost of
removing bits**, here by **6.7×**.

**D-43 withdrawn before it shipped.** D-23 stands, now on a frontend measurement
instead of a proxy. X-50's speed and footprint tables are sound — track *did* fall
7.270 → 6.753 as predicted; only the accuracy proxy failed.

**X-39's accuracy axis rests on the same harness**, so D-36/D-39's yield figures
describe the idealised chain. Flagged on those records rather than left. Their speed
figures were measured on real kernels and are unaffected.

**E-27**: rebuild the harness on binCV's own `pyrDownFiltered` cascade. Until then, no
accuracy conclusion from it may become a shipped default without a frontend
confirmation — **the rule that caught this one.**

Also: `BOX_3x3`'s build cost was estimated at +0.35 ms by scaling; measured in place it
is **+0.565 ms**. Another argument against scaling.

---

## 28 · E-27 shipped the fix — and refuted the diagnosis that motivated it

X-51 blamed the accuracy harness's float cascade for mispricing level 3's bit by
6.7×. E-27 fixed exactly that: `seedFiltered` now runs **binCV's own
`pyrDownFiltered` cascade**. **The fix is right and the diagnosis was wrong.**

| level 3's second bit, `1/2/2/1` vs `1/2/2/2` at `BOX_3x3` | |
|---|---|
| old float-cascade harness | −0.30 pts |
| **corrected harness** | **−0.42** |
| **frontend** | **−4.60** |

**0.12 points of movement against a 4.2-point gap.** And the corrections ran in
**both directions** — `1/1/1/1`, the ladder with the *most* cascaded quantization,
moved **up 1.89 points** where the mechanism predicted the largest fall. Mechanism
withdrawn.

**What the fix did settle:** the filter axis moved by **≤0.16 points**, exactly as
X-51 hedged. D-36/D-39's filter rankings stand and are now first-hand. Their warning
narrows from *"the accuracy figures are suspect"* to **"the ladder figures were; the
filter figures were not."**

**What's left is structural and not measured, so not claimed.** The harness warps
**one frame** — `prev` and `next` are binarizations of the same image with
near-identical edge maps. The frontend tracks **real consecutive frames**, whose
binarizations differ near the threshold, over a sequence where error compounds. Those
are different questions. And it may be irreducible: **the harness uses synthetic warps
because it needs ground truth, which is what makes it unrepresentative.**

**The rule tightened rather than relaxed:** no accuracy conclusion from that harness
becomes a shipped default, corrected cascade or not. Frontend accuracy is measured at
the frontend. **E-28** carries the open half.

---

## 29 · T4.3b, and then the x86 deficit turns out to be a compile flag

### The VIO frontend loop runs — and inverts the profile

`examples/vio_frontend.cpp`, modelled on HybVIO's: persistent track set,
`FAILED_FLOW`/`FLOW_OUT_OF_RANGE` culling, top-up detection with `applyMinDistance`
against survivors. **1710 frames, no gap requiring an operation binCV lacks.**

| | `frontend_sequence` | **top up below target** | **60% hysteresis** |
|---|---|---|---|
| detections | 4.8% of frames | **91.0%** | 45.0% |
| **detect** | 0.570 ms | **18.070** | 7.831 |
| **binCV total** | 10.644 | **31.641** | **19.996** |

**Detection is 39–57% of a real frontend, against D-28's 4.8%** — that figure was a
property of the benchmark's re-detect policy, and every optimisation priority since
X-31 rested on it. **The policy alone is worth 1.58×** and lives outside binCV.

### The x86 deficit: one flag, 3.75×

X-52 said port the NEON accumulators to AVX2. **Wrong — and the correction is the
result.** The NEON batching exists because aarch64 has *no scalar popcount*. x86 has
`POPCNT`… except baseline x86-64 predates SSE4.2, so it wasn't being emitted:
**zero `popcnt` instructions in the shipped binary.**

| build | binCV | OpenCV | ratio |
|---|---|---|---|
| default | 12.92 ms | 3.43 | **0.27×** |
| **`-mpopcnt`** | **3.45** | 3.15 | **0.91×** |

**3.75× from one flag**, from 3.8× slower than OpenCV to near parity — at 6.23× less
memory. The stage profile snaps to the aarch64 shape. **binCV was never mis-shaped on
x86; it was mis-compiled.** The AVX2 port is cancelled.

**And a lesson worth more than the flag:** X-52 predicted LK ≈2.7 ms and the frontend
≈3.9 ms at parity. Measured: **2.307 and 3.43.** The number was right and **the
mechanism was wrong** — had the AVX2 port been written, it would have "confirmed" the
hypothesis while the real cause went unfound.

---

## 30 · I wrote the x86 vector path. It's bit-exact and 1.88× slower.

X-59's ceiling said a batched `pshufb` popcount would be **7.9×** — beating hardware
`POPCNT`. I built it for `slicedSignedSum` at N=2 (three of four shipped levels),
mirroring the NEON path that wins there.

| arm | µs | vs scalar |
|---|---|---|
| **scalar** | **182.5** | 1.000× |
| **AVX2, inlined, bit-exact** | **344.0** | **0.53× — 1.88× SLOWER** |

**Two failures, and the first is a trap worth publishing.**
`__attribute__((target("avx2")))` **blocks inlining** — `slicedSignedSum` became **310
real calls per window** (`objdump`: 20 call sites, standalone symbol). That's *exactly*
the mechanism **E-31** proposes for runtime dispatch, now measured as unusable for an
inline hot function.

Fixing the inlining didn't fix the result. Compile-time `__AVX2__`, zero standalone
symbols — **still 1.88× slower**:

> The eight words are **computed in registers**, not loaded from memory. The ceiling
> used a contiguous-array load. Here the vector must be **assembled** from eight
> scalars and **disassembled** through a store and eight reloads. **The pack and unpack
> cost more than the eight `POPCNT`s they replace.**

### Five ceilings have now overstated, and that's the finding

| | ceiling | delivered |
|---|---|---|
| X-33 | 3.42× | 1.24× |
| D-37 | 1.461× | 1.069× |
| D-40 | 1.638× | **0.885×** |
| D-41 | 1.638× | 1.445× |
| **X-60** | **7.9×** | **0.53×** |

**Every one measured on bulk contiguous data, applied to a kernel that works on a
handful of register-resident words.** Not five unlucky estimates — **one structural
mismatch measured five times.** The problem is **granularity, not layout**, which
supersedes the part of D-48 that blamed the representation.

**E-33 narrowed, not closed.** `residualSums` (67%) is refuted. `build` (27%) is *not*
— `pyrDown` and the derivatives are genuine bulk passes over contiguous rows, which is
the shape the 4.7× adder ceiling was actually measured on. **The untested half is the
half the ceiling applies to.**

*Measurement note: mid-experiment the machine hit load average 8–10 from foreign
processes and `frontend_sequence` reported OpenCV alone ranging 3.7–10.9 ms on
identical work. Those runs were discarded. The verdict rests on `residual_n2`, which
interleaves its arms in one run.*
