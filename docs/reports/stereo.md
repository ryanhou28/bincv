# Dense stereo disparity

`denseDisparityBinary` against `cv::StereoBM` — the same job, a rectified pair to a
disparity map, winner-take-all over a window-aggregated cost. One thread on both sides.
Setup and denominator rule: [README.md](README.md).

This is where the library's premise is most literal: a binCV pipeline already holds packed
binary frames, and on bits the dense matching cost is one XOR per word of 64 pixels. The binary
path is faster than StereoBM on both measured architectures **and** holds roughly a
twenty-second of its working set.

## Summary

**Every `ratio` column below is OpenCV ÷ binCV: above 1× means binCV is ahead, below 1× means OpenCV is.** Where a table divides something else, its header says so.

752×480 · 64 disparities · 9×9 aggregation · one thread on both sides · `cv::StereoBM` at
its default `blockSize 21`, the strongest configuration measured for it here. Time is
milliseconds per frame and working set is bytes held live, so the smaller number is the
better one. **x86-64 and aarch64 are separate columns** — different OpenCV builds on
different machines, never averaged.

| arm | x86-64 (ms) | x86-64 ratio | aarch64 (ms) | aarch64 ratio | working set | memory ratio |
|---|---|---|---|---|---|---|
| `cv::StereoBM` | 12.675 | — | 79.8 | — | ≥ 722 KB, its output alone (2 B/px) | — |
| **`denseDisparityBinary`** (packed frames in) | **10.405** | **1.218× [1.199, 1.240]** | **60.4** | **1.32×** | **32.4 KB scratch** + 1 B/px out | **~22×**, a lower bound |
| `denseDisparity` (census 5×5, wide frames in) | 166.2 | not published | 462 | not published | 95.9 KB scratch + 1 B/px out | not published |

The census row's times are here and they lose; its ratio cells read `not published` because
this report has never published a ratio for that arm, and deriving one now would add a figure
no measurement record states. That spelling exists for callers arriving with wide images: it
pays a 24-bit census transform per pixel to compete on StereoBM's own terms, and it is behind.
The binary path is the operating point a binCV pipeline runs.

**The x86-64 column used to be floors of interleaved runs on a host with 20–100% spreads,
claimed only at that granularity. It is now thirty pinned launches of each binary**, median
with a bootstrap 95% interval ([`cv::StereoBM` and census](logs/dense_opencv-x86_64-launches.log),
[the binary path](logs/dense-x86_64-launches.log)). This is the one x86 ratio in these
reports that **cannot be paired**: the two arms live in different binaries, so the launches
cannot be matched up and the interval comes from resampling the two sweeps independently.
That makes it the weakest of the x86 intervals here, and it is wide enough to say so —
[1.199, 1.240] against a median of 1.218×. aarch64 figures are a single pinned-clock launch,
0–1% spread.

## What the number is made of

The path was built and priced in stages, each committed with its measurement (reference
device, pinned clock). This is binCV against its own earlier arms, so there is no OpenCV
column and no ratio — the smaller the number gets, the better:

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
| the runtime arm switch (1.7%, kept: one binary times and tests both arms) | **60.4** |

The vector arms exist at `uint64` on both architectures (NEON pairs, AVX2 quads),
runtime-dispatched on x86 so the baseline ISA is unchanged, switchable off
(`denseSimdEnabled`), and held byte-identical to the portable arm by a test that runs both
from one binary. The scalar arm measures 100.2 ms on the reference device and **21.22 ms over
thirty launches on x86-64, against the vector arm's 10.405** — the switch is real and the
sweep shows it on.

## The third architecture

The same kernel, unchanged, executes on a Cortex-M7 (STM32H753, no vector unit, no hardware
popcount): a 320×240, 32-disparity map in **6.5 KB of scratch**, bit-exact against the host —
825 ms at `uint32` at the 64 MHz floor clock. That run also measured the word-type guidance
inverting at 32-bit pointer width (`uint64` 1.30× slower there); the header carries the
width-qualified guidance. Details and the pre-registered rule:
[targets/stm32h753/README.md](../../targets/stm32h753/README.md).

## Memory

The refused allocation is the point: a dense cost volume at this configuration is 23 MB. The
kernel streams instead — a band of rows and one accumulator ring per disparity:

|  | working set | what it is | ratio |
|---|---|---|---|
| `cv::StereoBM` | ≥ 722 KB | its output buffer alone, before its internal buffers | — |
| **`denseDisparityBinary`** | **32.4 KB** | streaming scratch — a row band and one accumulator ring per disparity | **~22×** |

Both paths write 1 byte per pixel out on top of that. StereoBM's figure is a **lower bound**:
its internal buffers are not observable from outside with the tooling
[methodology-memory.md](methodology-memory.md) describes, so binCV's ~22× lead is a lower
bound too.

## Reproduce

```
./build/benchmark/dense_benchmark          # binCV arms, census and binary, both vector arms
./build/benchmark/dense_opencv_benchmark   # the cv::StereoBM denominator
./build/benchmark/dense_stage_profile      # where the binary path's time goes, by stage

# what the x86-64 column above is, for each of the first two
./scripts/run_launches.sh -n 30 ./build/benchmark/dense_benchmark
./scripts/aggregate_launches.py dense_benchmark-x86_64-launches.log
```

Logs: [binCV arms](logs/dense-x86_64-launches.log) ·
[the `cv::StereoBM` denominator](logs/dense_opencv-x86_64-launches.log)

The census transform's own price (27.5 → 4.7 ms on the reference device with its vector rows,
which the census dense path streams) is `census_benchmark`.
