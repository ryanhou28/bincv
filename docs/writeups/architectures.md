# How binCV uses what each architecture provides

Processors differ in what they offer a kernel: how wide their integer registers are, which
bit-manipulation instructions they have, and which vector extensions they add. Each of these
can speed up some of binCV's kernels and does nothing for others.

This writeup has two parts. The first runs through those hardware features and the technique
each one enables for packed bits. The second looks at the four machines binCV has been
measured on — a desktop x86-64 CPU, a Raspberry Pi 4, an STM32 microcontroller and an
NVIDIA GPU — and at what the measurements show. It makes no claim about machines binCV has
not run on.

Every figure quoted here comes from a report or target page linked beside it, and every
chart is drawn from the report table it cites. [The premise](premise.md) explains the
representation itself.

## Part 1: hardware features, and what each does for packed bits

### Integer width: how many pixels one instruction handles

A bit-packed word instruction processes as many pixels as the register has bits: 32 pixels
per AND on a 32-bit core, 64 on a 64-bit one. So the natural technique is to **match the
word type to the register width**. binCV is templated on its word type, from `uint8_t` to
`uint64_t`, so the choice is the caller's.

Two things limit it. On a 32-bit core a 64-bit word is emulated with two instructions and a
register pair, so it costs more than it saves. And a wider word rounds each row's stride up
more coarsely, which costs memory; binCV's default is `uint32_t` for that reason.

### Population count: counting a word's set bits in one instruction

Many binCV operations end in counting set bits. `countNonZero` counts the pixels in a region,
a Hamming distance between two descriptors is the count of their XOR, and the gradient
covariance in the tracker is four counts over masks. A **population count instruction**
counts every set bit in a word at once, so a reduction costs one instruction per word.

Without one, the count is a software sequence of about a dozen shifts, masks and adds per
word — a "SIMD within a register" trick that sums bits in pairs, then nibbles, then bytes.
Reductions then cost several times more per word, and the useful technique changes: keep the
per-byte partial sums across many words and finish the sum once, instead of finishing it for
every word.

Where the instruction lives matters as well as whether it exists. Some instruction sets
count bits only in vector registers. There, counting a single word from a general register
means moving it into the vector unit and the result back out, and those moves can cost as
much as the count. The technique is to **count in bulk**: load data straight into vector
registers, accumulate there, and move a result out once per region. binCV offers reductions
only over regions, masks and windows, never over a single word, so that every kernel can do
this.

### SIMD extensions: many words per instruction

Vector extensions — SSE and AVX2 on x86-64, NEON on Arm — apply one instruction to a 128-
or 256-bit register. For packed bits that is 128 or 256 pixels per AND. binCV has vector
paths for AVX2, selected at run time, and for NEON, and each one can be switched off so a
benchmark can measure what it is worth. SIMD also lets a kernel process several independent
small problems side by side, as the tracker does with eight keypoints at once.

SIMD also speeds up the byte-based alternative. A 256-bit register of bytes holds 32
pixels — exactly what one `uint32_t` of bits holds. So on a machine with SIMD, the
comparison that decides a result is bits in vector registers against bytes in vector
registers, not bits against a byte-at-a-time loop.

### DSP extensions on microcontrollers

Microcontroller cores without SIMD often have a DSP extension that works on four bytes
packed in one 32-bit register. On Armv7E-M, `USAD8` — a sum of absolute byte differences —
adds up four bytes in one instruction when one operand is zero, which can replace the last
steps of a software population count.

### GPU warp primitives

A GPU runs threads in groups — a warp of 32 on NVIDIA hardware — and provides instructions
that work across the group. `__ballot_sync` collects one true-or-false answer from each of
the 32 threads into a single 32-bit word. That word is the packed format, so a GPU can turn
32 per-pixel comparisons into one packed word in one instruction. `__popc` counts the bits
of a 32-bit register.

### Summary

| feature | what it speeds up | x86-64 | aarch64 | Cortex-M7 | GPU |
|---|---|---|---|---|---|
| native integer width | pixels per scalar instruction | 64 bits | 64 bits | 32 bits | 32 bits |
| population count | reductions, Hamming distance, covariance | `POPCNT`, general registers | `cnt`, vector registers only | none | `__popc`, 32 bits |
| SIMD | logic and counting over many words at once | AVX2, 256 bits | NEON, 128 bits | none | the 32-thread warp |
| DSP byte instructions | the tail of a software population count | — | — | `USAD8` and others | — |
| warp primitives | packing pixels from a comparison | — | — | — | `__ballot_sync` |

## Part 2: what the measurements show

### The four machines

| | x86-64 | aarch64 | Cortex-M7 | GPU |
|---|---|---|---|---|
| part | AMD Ryzen 5 5600X | Broadcom BCM2711 (Raspberry Pi 4), Cortex-A72 at 1.8 GHz | STM32H753ZI, run at its 64 MHz reset clock | NVIDIA RTX 3070 Ti, SM 8.6 |
| memory | 32 KiB L1d, 512 KiB L2 per core, 32 MiB L3 | 32 KiB L1d, 1 MiB shared L2 | 512 KB AXI SRAM, 16 KiB stack as configured | ~608 GB/s device memory |

### Integer width: 64-bit words on 64-bit cores, 32-bit words elsewhere

On kernels whose cost is arithmetic, the word that matches the register wins, and the same
choice loses on a core of the other width:

<!-- figure-check values="measured" source="source" -->
| machine | operation | `uint64_t` against `uint32_t` | measured | source |
|---|---|---|---|---|
| x86-64 | `countNonZero`, 640×480 | 1.57× faster | 0.005895 against 0.009270 ns/pixel | [primitives.md](../reports/primitives.md) |
| aarch64 | `countNonZero`, 640×480 | 1.95× faster | 1.95× [1.95, 1.95] | [footprint.md](../reports/footprint.md) |
| Cortex-M7 | dense disparity, 320×240 | 1.30× slower | 68,807,055 against 52,823,363 cycles | [targets/stm32h753](../../targets/stm32h753/README.md) |
| GPU | every kernel | not offered | `uint64_t` is two 32-bit operations | [ARCHITECTURE.md §8.5](../ARCHITECTURE.md) |

The x86-64 speedup is computed from the two published medians; the others are published as
ratios. Pointwise logic is the exception: AND, OR and XOR run at memory bandwidth on both
sides, and the word type makes no consistent difference to them
([primitives.md](../reports/primitives.md#logic)).

binCV's default is still `uint32_t` on every machine, because at the small upper levels of a
pyramid a 64-bit stride costs up to 1.33× the bytes, and where speed and footprint conflict,
footprint wins. A kernel whose scratch is a band rather than a frame is free to recommend
otherwise, and dense disparity does: `uint64_t` on 64-bit cores, the native word elsewhere.

A caller who chooses 64-bit words is not locked out of the 32-bit kernels, because on a
little-endian machine the two layouts are the same bytes:

![A 64-bit row word laid over the same eight bytes as two 32-bit words: bits 0–63 are pixels
0–63 either way, so a 64-bit plane reads as a 32-bit plane at twice the stride](figures/architectures-narrowing.svg)

`narrowPlane` reinterprets the view. Measured on `edgeThreshold`, a narrowed 64-bit buffer
runs at the native 32-bit speed on the Cortex-A72 and within 4% of it on x86-64
([footprint.md](../reports/footprint.md#where-speed-was-declined-to-protect-it)).

### Population count: one instruction, a round trip, or software

The four machines handle the count in three different ways.

![The population count's path on x86-64, all in general registers; on aarch64 one word at a
time, crossing into the NEON register file and back; and on aarch64 over a whole window,
crossing once](figures/architectures-popcount.svg)

**On x86-64** `POPCNT` is an ordinary integer instruction, so a per-word count costs one
instruction.

**On aarch64** there is no scalar population count. `cnt` works on NEON registers, so a
word in a general register has to cross into the vector register file and its count has to
cross back — and those crossings cost about as much as the count. This is the reason binCV
has no public `popcount(word)`: reductions are offered over regions, masks and windows, so a
kernel can load straight into vector registers, accumulate there, and cross once. The rule
applies on every machine; aarch64 is the reason for it.

**On the Cortex-M7** there is no instruction at all. GCC calls libgcc's software sequence
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

### SIMD extensions: bits against an already vectorized byte kernel

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

#### The window, not the word, caps the gain

Inside Lucas–Kanade the unit of work is one row of a 31-pixel window:

![A 31-pixel window in a uint32_t word uses 31 of 32 bits; in a uint64_t, 31 of 64; OpenCV's
CV_16S lanes in an AVX2 register process 16 pixels per operation](figures/architectures-window.svg)

In a 32-bit word the row is 31 pixels per operation against OpenCV's 16 — a 1.94× packing
advantage — and that is the ceiling at this window size. A wider word lowers the
utilisation to 48% without processing any more of the window.

### The bit-depth crossover moves with the machine

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

### Less data does not make a compute-bound kernel faster

An eighth of the bytes decides what fits on a device. It does not by itself make a kernel
faster. Lucas–Kanade at a fixed 140 points, with the frame grown 36-fold:

![Lucas–Kanade cost per point relative to 320×240, across frame sizes from 9.4 to 337.5 KiB:
no trend with frame size on either machine](figures/architectures-lk-frame-size.svg)

From the smallest frame to the largest, the per-point cost moves 0.4% on x86-64 and 6% on
the Cortex-A72, and the points between show no trend with size. A 31×31 window is two to
four cache lines at one bit per pixel, and it would be two to four as bytes too. The
tracker is compute-bound, so its speed has to come from doing less work, and the footprint
result stands independently of it.

### The microcontroller: no population count, no SIMD

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

### The GPU: warp primitives that produce the format

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

### What holds on every machine

- **Memory barely moves.** A plane's size is arithmetic on its geometry, so a footprint
  figure is identical on every CPU, and on the GPU differs only by the driver's allocation
  rounding. Speed is the axis that moves.
- **Match the word to the register** for kernels whose cost is arithmetic. The default
  gives that up only for footprint.
- **Count in bulk**, because on one of the four machines a per-word count is a round trip
  between register files.
- **The byte alternative sets the bar**, and SIMD makes it stronger on some machines than
  others. binCV's lead is largest where the data is genuinely narrow and the byte kernel is
  weakest, and it disappears where the input is wide.
