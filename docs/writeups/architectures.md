# How bits map onto the machines binCV runs on

[The premise](premise.md) is machine-independent: store a pixel in one bit and a word
instruction processes a word's worth of pixels. How much that is worth is not. It depends
on how wide the machine's integers are, where its population count lives, what its vector
unit already does to bytes, and how strong the byte-based alternative is on that machine.

This writeup covers the four machines binCV has been measured on, and only those. Each
figure quoted here comes from a report or target page linked beside it, and every chart is
drawn from the report table it cites.

## The four machines

| | x86-64 | aarch64 | Cortex-M7 | GPU |
|---|---|---|---|---|
| part | AMD Ryzen 5 5600X | Broadcom BCM2711 (Raspberry Pi 4), Cortex-A72 at 1.8 GHz | STM32H753ZI, run at its 64 MHz reset clock | NVIDIA RTX 3070 Ti, SM 8.6 |
| native integer width | 64 bits | 64 bits | 32 bits | 32 bits; no 64-bit integer datapath |
| population count | `POPCNT`, in general registers | `cnt`, in NEON registers only | none: a software sequence | `__popc`, on a 32-bit register |
| vector unit binCV uses | AVX2, 256 bits, selected at run time | NEON, 128 bits | none | the 32-lane warp |
| memory | 32 KiB L1d, 512 KiB L2 per core, 32 MiB L3 | 32 KiB L1d, 1 MiB shared L2 | 512 KB AXI SRAM, 16 KiB stack as configured | ~608 GB/s device memory |

Every row below differs down that table, and so do the results.

## The word should be the machine's integer width

binCV is templated on its word type, from `uint8_t` to `uint64_t`. A wider word does less
loop overhead per pixel, so on a 64-bit core it is faster. On a 32-bit core every 64-bit
operation is two operations and a register pair, so it is slower:

<!-- figure-check values="uint64_t against uint32_t" source="source" -->
| machine | operation | uint64_t against uint32_t | source |
|---|---|---|---|
| x86-64 | `countNonZero`, 640×480 | 0.005895 against 0.009270 ns/pixel | [primitives.md](../reports/primitives.md) |
| aarch64 | `countNonZero`, 640×480 | 1.95× faster | [footprint.md](../reports/footprint.md) |
| Cortex-M7 | dense disparity, 320×240 | 1.30× slower | [targets/stm32h753](../../targets/stm32h753/README.md) |
| GPU | every kernel | not offered: `uint64_t` is two 32-bit operations | [ARCHITECTURE.md §8.5](../ARCHITECTURE.md) |

The default is nevertheless `uint32_t` on every machine, because the word type also sets
how coarsely each row's stride rounds up, and at the small upper levels of a pyramid a
64-bit stride costs up to 1.33× the bytes. Where speed and footprint conflict, footprint
wins. A kernel whose scratch is a band rather than a frame is free to recommend otherwise,
and dense disparity does: `uint64_t` on 64-bit cores, the native word elsewhere.

A caller who chooses 64-bit words is not locked out of the 32-bit kernels, because on a
little-endian machine the two layouts are the same bytes:

![A 64-bit row word laid over the same eight bytes as two 32-bit words: bits 0–63 are pixels
0–63 either way, so a 64-bit plane reads as a 32-bit plane at twice the stride](figures/architectures-narrowing.svg)

`narrowPlane` reinterprets the view. Measured on `edgeThreshold`, a narrowed 64-bit buffer
runs at the native 32-bit speed on the Cortex-A72 and within 4% of it on x86-64
([footprint.md](../reports/footprint.md#where-speed-was-declined-to-protect-it)).

## Where the population count lives

Counting set bits is the reduction under nearly every measurement binCV makes — a pixel
count, a Hamming distance, a gradient covariance. The four machines put it in three
different places.

![The population count's path on x86-64, all in general registers; on aarch64 one word at a
time, crossing into the NEON register file and back; and on aarch64 over a whole window,
crossing once](figures/architectures-popcount.svg)

**On x86-64** `POPCNT` is an ordinary integer instruction and a per-word count costs what it
looks like it costs.

**On aarch64** there is no scalar population count. `cnt` works on NEON registers, so a
word in a general register has to cross into the vector register file and its count has to
cross back — and those crossings cost about as much as the count. This is the reason binCV
has no public `popcount(word)`: reductions are offered over regions, masks and windows, so a
kernel can load straight into vector registers, accumulate there, and cross once. The rule
applies on every machine; aarch64 is the reason for it.

**On the Cortex-M7** there is no instruction at all. GCC calls libgcc's software sequence,
once per word. Four ways of counting a word, timed inside `countNonZero` on a 752×480 frame:

<!-- figure-check values="cycles|against the shipped arm" source="@targets/stm32h753/README.md" -->
| per-word counter | cycles | against the shipped arm |
|---|---|---|
| libgcc `__popcountsi2`, the shipped arm | 201,600 | 1.00× |
| the same sequence, inlined | 213,300 | 1.05× slower |
| a 64-bit sequence, inlined | 374,270 | 1.85× slower |
| inlined, with the DSP extension's `USAD8` for the final sum | 178,700 | 1.13× faster |

Almost nothing is available in the per-word counter. What moves the number is the
**loop**: keeping the software count's per-byte partial sums in an accumulator and
collapsing them once per sixteen words, rather than once per word, is **2.6× faster** on
the M7, in plain C++. It helps only where the count is software and only on long
contiguous reductions, and the target page
([targets/stm32h753](../../targets/stm32h753/README.md)) records why it does nothing for the
tracking pipeline, which performs none.

**On the GPU** `__popc` counts a 32-bit register, which is one format word.

## A word of bits against a vector of bytes

The comparison that decides most results on the two application processors is not bits
against bytes. It is bits against a byte kernel that has already been vectorized:

![To scale in bits: an AVX2 register of bytes holds 32 pixels, one uint32_t word of bits
also holds 32, and an AVX2 register of bits holds 256](figures/architectures-register.svg)

A `uint32_t` word of bits holds 32 pixels, and so does a 256-bit AVX2 register of bytes.
Packing alone therefore gains nothing against a vectorized byte kernel. The gain appears
only when the bit logic also moves into vector registers, and where OpenCV's byte kernel is
already well vectorized, binCV ties:

<!-- figure-check values="x86-64|aarch64" source="@docs/reports/limits.md" -->
| operation, 640×480 | x86-64 | aarch64 | why |
|---|---|---|---|
| `erode`, 3×3 rect | 1.05× | 1.02× | a dead heat against a mature vectorized kernel |
| FAST, on bytes | 1.04× | 0.962× | parity; the bit-plane overload is where packing applies |
| `countNonZero` | 1.62× | 2.66× | OpenCV is bandwidth-bound; binCV reads an eighth of the data |

Ratios are OpenCV's time over binCV's, one thread each side; FAST is 752×480. The vector
arms binCV does have are switchable at run time, so their worth is measured rather than
assumed: the eight-keypoint AVX2 batch in the tracker is worth 1.84× on tracking on x86-64,
bit-exact with the scalar path ([limits.md](../reports/limits.md#the-vector-arms-and-proving-they-are-on)).

### The window, not the word, caps the gain

Inside Lucas–Kanade the unit of work is one row of a 31-pixel window:

![A 31-pixel window in a uint32_t word uses 31 of 32 bits; in a uint64_t, 31 of 64; OpenCV's
CV_16S lanes in an AVX2 register process 16 pixels per operation](figures/architectures-window.svg)

In a 32-bit word the row is 31 pixels per operation against OpenCV's 16 — a 1.94× packing
advantage — and that is the ceiling at this window size. A wider word lowers the
utilisation to 48% without processing any more of the window.

## The crossover moves with the machine

Bit-slicing pays at low bit depth and loses at high bit depth, because the adder needs more
planes as its input gets wider. Where it stops paying is not a property of the algorithm.
The same box-filter downsample, against `cv::pyrDown` on each machine, at equal input and
output widths:

![binCV's box-filter downsample against cv::pyrDown from 1 to 8 bits per pixel: on the
Cortex-A72 binCV leads through 4 bits and crosses between 4 and 5; on x86-64 it is behind at
every equal width](figures/architectures-crossover.svg)

On the Cortex-A72, binCV stays ahead through four bits per pixel. On x86-64 it is behind at
every equal-width depth, and only the shipped shape — one bit in, three bits out — is ahead,
at 1.56× there and 5.51× on the Cortex-A72 ([limits.md](../reports/limits.md)).

The difference is the denominator. OpenCV's x86-64 build dispatches at run time up through
AVX-512 code paths and its pyramid is very good; its aarch64 pyramid is relatively weaker on
the same silicon. binCV's own times scale about as expected between the two machines;
OpenCV's do not. **A ratio measured on a desktop is not a ratio on a deployment part, in
either direction.**

## Less data does not make a compute-bound kernel faster

An eighth of the bytes decides what fits on a device. It does not by itself make a kernel
faster. Lucas–Kanade at a fixed 140 points, with the frame grown 36-fold:

![Lucas–Kanade cost per point relative to 320×240, across frame sizes from 9.4 to 337.5 KiB:
no trend with frame size on either machine](figures/architectures-lk-frame-size.svg)

From the smallest frame to the largest, the per-point cost moves 0.4% on x86-64 and 6% on
the Cortex-A72, and the points between show no trend with size. A 31×31 window is two to
four cache lines at one bit per pixel, and it would be two to four as bytes too. The
tracker is compute-bound, so its speed has to come from doing less work, and the footprint
result stands independently of it.

## On the microcontroller

The Cortex-M7 has no population count, no vector unit and a 32-bit `size_t`, so it is the
machine where binCV has the least help from the hardware. The library runs there unchanged:
no kernel allocates, so none needs a heap; the one threading header is left out, because
binCV threads only through a backend the caller installs.

<!-- figure-check values="measured" source="source" -->
| on the STM32H753ZI | measured | source |
|---|---|---|
| a 752×480 frame, as bits and as `CV_8U` | 46,080 against 360,960 bytes | [limits.md](../reports/limits.md#what-is-not-measured-at-all) |
| tracker staging on the stack, at 2 bits per pixel | 4,120 bytes of a 16 KiB stack | [limits.md](../reports/limits.md#what-is-not-measured-at-all) |
| dense disparity, 320×240, 32 disparities, `uint32_t` | 825 ms at 64 MHz, in 6,480 B of scratch | [targets/stm32h753](../../targets/stm32h753/README.md) |

The stack budget is a compile-time setting that fails the build rather than overflowing at
run time, which matters on a part where an overflow is silent corruption rather than a
crash. The dense-disparity kernel's first run on this core returned the exact disparity on
its test pair, and the same map at both word types.

What has **not** been measured there is as important: no comparison against OpenCV, no
pipeline or tracker timing, and nothing at the part's full 480 MHz clock. The disparity
figure is a floor at an eighth of the clock, not the part's number.

## On the GPU

A CUDA core is a 32-bit integer machine, and the hardware's own primitives work on 32 bits,
so the device word is `uint32_t` and nothing else. One of those primitives packs the format
directly:

![Thirty-two threads of a warp each test one pixel, and __ballot_sync returns the 32 answers
as one word in which lane i is bit i](figures/architectures-ballot.svg)

Lane *i* sets bit *i*, which is where the format stores pixel *i*, so a warp produces a
32-pixel word of the host's own layout in a single instruction. Device planes are
byte-identical to host planes, and every device kernel is proven bit-exact against the host
library. Against `cv::cuda` on the same GPU:

<!-- figure-check values="cv::cuda|binCV|ratio" source="@docs/reports/cuda.md" -->
| operation, 752×480 | cv::cuda | binCV | ratio |
|---|---|---|---|
| dense disparity from bits, time (ms) | 0.7134 | 0.06400 | 11.2× |
| dense disparity from bits, device memory (KiB) | 3072.0 | 448.0 | 6.86× |
| dense disparity through census, time (ms) | 0.7101 | 0.4789 | 1.50× |
| dense disparity through census, device memory (KiB) | 3072.0 | 4512.0 | 0.681× |

The census row is the exception that shows the rule: census expands each 8-bit pixel into a
32-bit descriptor word, so its input is wider than the image it came from, and binCV is
larger there. On one-bit input the dense cost is one XOR per 32-pixel word, and binCV leads
on both axes. The full set, including where binCV does not lead, is in
[cuda.md](../reports/cuda.md).

## What holds on every machine

- **Memory barely moves.** A plane's size is arithmetic on its geometry, so a footprint
  figure is identical on every CPU, and on the GPU differs only by the driver's allocation
  rounding. Speed is the axis that moves.
- **The right word is the native integer width**, and the default trades that away only for
  footprint.
- **Reductions stay bulk**, because on one of the four machines a per-word count is a
  round trip between register files.
- **The byte alternative sets the bar**, and its strength differs by machine. binCV's lead
  is largest where the data is genuinely narrow and the byte kernel is weakest, and it
  disappears where the input is wide.
