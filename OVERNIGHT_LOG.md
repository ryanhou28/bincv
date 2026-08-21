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
