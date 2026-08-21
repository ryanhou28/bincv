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
