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
