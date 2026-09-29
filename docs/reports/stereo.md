# Dense stereo disparity

`denseDisparityBinary` against `cv::StereoBM` — the same job, a rectified pair to a
disparity map, winner-take-all over a window-aggregated cost. One thread on both sides.
Setup and denominator rule: [README.md](README.md).

This is where the library's premise is most literal: a binCV pipeline already holds packed
binary frames, and on bits the dense matching cost is one XOR per word — 64 pixels per host
word (the CUDA backend's device word holds 32). The binary path is faster than `cv::StereoBM`
on both measured architectures and holds a smaller working set — at least 1.84× smaller, a
lower bound — and it does so in 32,352 B of scratch where a dense cost volume at this
geometry would be 23,101,440 B.

## Summary

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

752×480 · 64 disparities · 9×9 aggregation on binCV's side · one thread on both sides ·
`cv::StereoBM` at its default `blockSize 21`, the only configuration measured. binCV
aggregates over 9×9, so this is a comparison of role — a disparity map from a rectified
pair — and not of window size. Time is milliseconds per frame and working set is bytes held
live, so the smaller number is the better one. **x86-64 and aarch64 are separate columns** —
different OpenCV builds on different machines, never averaged. Each time is the median of
thirty pinned launches of its binary on x86-64 and seven on aarch64, taken at `80ff0a8`
(on `main` as `086428c`).

| arm | x86-64 (ms) | x86-64 ratio | aarch64 (ms) | aarch64 ratio | working set (B) | memory ratio |
|---|---|---|---|---|---|---|
| `cv::StereoBM` | 12.68 | — | 79.90 | — | ≥ 721,920 (its `CV_16S` output alone) | — |
| **`denseDisparityBinary`** (packed frames in) | 10.41 | 1.22× [1.20, 1.24], unpaired: two binaries | 60.57 | 1.32× [1.32, 1.32], unpaired: two binaries | 393,312 (360,960 output + 32,352 scratch) | ≥ 1.84×, a lower bound |
| `denseDisparity` (census 5×5, 8-bit frames in) | 166.2 | 0.0763×, unpaired | 730.6 | 0.109×, unpaired | 456,864 (360,960 output + 95,904 scratch) | ≥ 1.58×, a lower bound |

**The ratios are unpaired.** The two arms live in different binaries (`dense_benchmark` and
`dense_opencv_benchmark`), so their launches cannot be matched up. The x86-64 interval comes
from resampling the two thirty-launch sweeps independently, which makes it the widest x86-64
interval in these reports. The aarch64 arms scatter 0.4% and 0.6% across their seven
launches.

**The census row exists for callers arriving with 8-bit frames** and is not the operating
point a binCV pipeline runs: it pays a 24-bit census transform per pixel to compete on
`cv::StereoBM`'s own terms, and it is behind — 13.1× slower on x86-64 and 9.14× on the
device. The row times the `uint32` sliding arm on both architectures; the `uint64` arm
reads 109.9 ms on x86-64 and 458.0 ms on the device in the same logs.

**`cv::StereoBM`'s working set is a lower bound.** ≥ 721,920 B is its `CV_16S` output
(2 B/px) alone; its internal buffers were not measured — the allocator interposer in
[methodology-memory.md](methodology-memory.md#heap--allocator-interposition) can see them,
and nobody has run it on this arm. binCV's figures are its 1 B/px output plus the scratch
the caller provides, read off the objects. Both memory ratios are therefore lower bounds.

## What the number is made of

The binary path was built and priced in stages, each committed with its measurement
(reference device, pinned clock). This is binCV against its own earlier arms, so there is
no OpenCV column and no ratio — the smaller the number gets, the better:

| stage | time, aarch64 (ms/frame) |
|---|---|
| census v1 (per-pixel, the shape that prompted the rebuild) | 1842 |
| census v2, sliding vertical accumulator | 489 |
| binary-native path, scalar winner-take-all | 167 |
| bit-sliced winner-take-all | 110 |
| plane-outer carry rows | 99 |
| NEON plane ops | 79 |
| two independent carry chains per iteration | 72 |
| per-stage plane bounds in the doubling tree | 59 |
| the runtime arm switch (costs 1.02×; kept, because one binary then times and tests both arms) | 60.57 |

The vector arms exist at `uint64` on both architectures (NEON pairs, AVX2 quads),
runtime-dispatched on x86 so the baseline ISA is unchanged, switchable off
(`denseSimdEnabled`), and held byte-identical to the portable arm by a test that runs both
from one binary. The portable arm measures 84.47 ms on the device against the vector arm's
60.57 — 1.39× [1.39, 1.40] over seven launches — and 21.22 ms on x86-64 against 10.41 —
2.05× [2.01, 2.09] over thirty. Both of those ratios are paired per launch, because the two
arms are in one binary.

## The third architecture

The same kernel, unchanged, executes on a Cortex-M7 (STM32H753, no vector unit, no hardware
popcount): a 320×240, 32-disparity map in 6,480 B of scratch, bit-exact against the host —
825 ms at `uint32` at the 64 MHz floor clock. The word-type guidance inverts at 32-bit
pointer width (`uint64` is 1.30× slower there), and the header carries the width-qualified
guidance. Details: [targets/stm32h753/README.md](../../targets/stm32h753/README.md).

## Memory

The refused allocation is the point: a dense cost volume at this configuration is
23,101,440 B. The kernel streams instead — a band of rows and one accumulator ring per
disparity — in 32,352 B of caller-provided scratch. That is the memory claim: the scratch,
against the cost volume the design does not allocate.

Like for like, each side's output plus what it holds to produce it:

| | working set | what it is |
|---|---|---|
| `cv::StereoBM` | ≥ 721,920 B | its `CV_16S` output (2 B/px) alone; its internal buffers were not measured |
| **`denseDisparityBinary`** | 393,312 B | 360,960 B output (1 B/px) + 32,352 B streaming scratch |
| `denseDisparity` (census 5×5) | 456,864 B | 360,960 B output (1 B/px) + 95,904 B streaming scratch |

`cv::StereoBM` ÷ binCV: ≥ 1.84× for the binary path and ≥ 1.58× for the census path — both
lower bounds, because the StereoBM side is one.

## Reproduce

```
./build/benchmark/dense_benchmark          # binCV arms, census and binary, both vector arms
./build/benchmark/dense_opencv_benchmark   # the cv::StereoBM denominator
./build/benchmark/dense_stage_profile      # where the binary path's time goes, by stage
./build/benchmark/census_benchmark         # the census transform on its own, with and without its vector rows

# the columns above, for each of the first two: thirty launches on x86-64, seven on the device
./scripts/run_launches.sh -n 30 ./build/benchmark/dense_benchmark
./scripts/aggregate_launches.py dense_benchmark-x86_64-launches.log
```

No launch sweep of `census_benchmark` is committed, so the census transform's own cost is
not published here.

Logs: [binCV arms, x86-64](logs/dense-x86_64-launches.log) ·
[`cv::StereoBM` and census, x86-64](logs/dense_opencv-x86_64-launches.log) ·
[binCV arms, aarch64](logs/dense-aarch64-launches.log) ·
[`cv::StereoBM` and census, aarch64](logs/dense_opencv-aarch64-launches.log).
Logs marked stale in [expected-stale.txt](logs/expected-stale.txt) were taken before a
bit-identical change to the code beneath them; the file records which files moved and why
the figure stands.

## Changes to published figures

| date | row | previous | current | why |
|---|---|---|---|---|
| 2026-09-21 | x86-64 column | floors of interleaved single runs | median of thirty pinned launches per binary, with interval | protocol change; logs committed |
| 2026-09-21 | aarch64 column | no committed log | seven pinned launches per binary | first committed device logs |
| 2026-09-21 | `denseDisparity` census, aarch64 | 462 ms | 730.6 ms | the earlier cell matched the `uint64` arm (458.0 ms in the new log); the row now times the `uint32` arm on both sides |
| 2026-09-21 | portable arm, aarch64 | 100.2 ms (1.66× vs the vector arm) | 84.47 ms (1.39×) | the portable arm had got faster while untimed; re-taken |
| 2026-09-28 | memory ratio | ~22× | ≥ 1.84× | the earlier figure divided `cv::StereoBM`'s output by binCV's scratch alone; now output + scratch on both sides |
| 2026-09-28 | census ratios | not published | 0.0763× / 0.109× / ≥ 1.58× | the arms were already printed; the ratios follow from them |
