# Dense stereo disparity

`denseDisparityBinary` against `cv::StereoBM` — the same job, a rectified pair to a
disparity map, winner-take-all over a window-aggregated cost. One thread on both sides.
Setup and denominator rule: [README.md](README.md).

This is the operation where the library's premise is most literal: a binCV pipeline
already holds packed binary frames, and on bits the dense matching cost is one XOR per
word of 64 pixels. The binary path is faster than StereoBM on both measured
architectures **and** holds roughly a twenty-second of its working set — the first
operation in these reports to lead its role bar on speed and memory at once.

## Summary

752×480 · 64 disparities · 9×9 aggregation · one thread on both sides · `cv::StereoBM` at
its default `blockSize 21`, the strongest configuration measured for it here.

Time is milliseconds per frame and working set is bytes held live, so the smaller number is
the better one in every column. **x86-64 and aarch64 are separate columns and are never
averaged** — different OpenCV builds on different machines.

| arm | time, x86-64 (ms) | time, aarch64 (ms) | working set |
|---|---|---|---|
| `cv::StereoBM` | ~14.7 | 79.8 | ≥ 722 KB, its output alone (2 B/px) |
| **`denseDisparityBinary`** (packed frames in) | **~12.0** | **60.4** | **32.4 KB scratch** + 1 B/px out |
| `denseDisparity` (census 5×5, wide frames in) | ~200 | 462 | 95.9 KB scratch + 1 B/px out |

The binary path is **~1.2× faster than `cv::StereoBM` on x86-64 and 1.32× on aarch64**, in
**~22× less working set**. The census row loses on speed and is in the table for that
reason rather than in spite of it.

x86-64 figures are floors of interleaved runs on a host with 20–100% spreads and are
claimed only at this granularity; aarch64 figures are pinned-clock, 0–1% spread. The
census spelling exists for callers arriving with wide images: it pays a 24-bit census
transform per pixel to compete on StereoBM's own terms, and it is behind — stated here
rather than averaged away. The binary path is the operating point a binCV pipeline runs.

## What the number is made of

The path was built and priced in stages, each committed with its measurement
(reference device, pinned clock). This is binCV against its own earlier arms, so there is
no OpenCV column — the column is one number per stage and the smaller it gets, the better:

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
(`denseSimdEnabled`), and held byte-identical to the portable arm by a test that runs
both from one binary. The scalar arm measures 100.2 ms on the reference device.

## The third architecture

The same kernel, unchanged, executes on a Cortex-M7 (STM32H753, no vector unit, no
hardware popcount): a 320×240, 32-disparity map in **6.5 KB of scratch**, bit-exact
against the host — 825 ms at `uint32` at the 64 MHz floor clock. That run also
measured the word-type guidance inverting at 32-bit pointer width (`uint64` 1.30×
slower there); the header carries the width-qualified guidance. Details and the
pre-registered rule: [targets/stm32h753/README.md](../../targets/stm32h753/README.md).

## Memory

The refused allocation is the point: a dense cost volume at this configuration is
23 MB. The kernel streams — a band of rows and one accumulator ring per disparity — so
the two sides read:

| | working set | what it is |
|---|---|---|
| `cv::StereoBM` | ≥ 722 KB | its output buffer alone, before its internal buffers |
| **`denseDisparityBinary`** | **32.4 KB** | streaming scratch — a row band and one accumulator ring per disparity |

Both paths write 1 byte per pixel out on top of that. StereoBM's figure is a **lower
bound**: its internal buffers are not observable from outside with the tooling
[methodology-memory.md](methodology-memory.md) describes, so binCV's ~22× lead is a lower
bound too.

## Reproduce

```
./build/benchmark/dense_benchmark          # binCV arms, census and binary, both vector arms
./build/benchmark/dense_opencv_benchmark   # the cv::StereoBM denominator
./build/benchmark/dense_stage_profile      # where the binary path's time goes, by stage
```

The census transform's own price (27.5 → 4.7 ms on the reference device with its
vector rows, which the census dense path streams) is `census_benchmark`.
