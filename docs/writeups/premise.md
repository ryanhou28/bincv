# Pixels at their true bit width

A binary image is one bit of information per pixel. Conventional vision libraries store it
in a byte and compute on it in bytes, so a mask, a thresholded edge map or a census bit
costs eight times the memory it holds and eight times the work it needs. binCV stores such
images at their real width and computes on them a machine word at a time. This writeup
shows how that works, why the representation is shaped the way it is, and where the idea
stops paying.

The rules themselves live in [ARCHITECTURE.md](../ARCHITECTURE.md); this is the illustrated
walk through them. Every figure here is drawn by
[`scripts/gen_writeup_figures.py`](../../scripts/gen_writeup_figures.py), which computes each
bit pattern it shows rather than drawing it by hand.

## One bit per pixel

![A 16×4 binary mask stored as 64 bytes and as eight 8-bit words. Below, pixels 0–7 of row 1
drawn both ways: as eight bytes, 64 bits of which one per byte carries the pixel, and as one
8-bit word](figures/premise-bytes-vs-bits.svg)

Stored as bytes, every pixel in the figure takes a whole byte, and seven of its eight bits
are always zero. Stored as bits, each 16-pixel row takes two 8-bit words, and nothing is
wasted. Pixel `x` lives at bit `x % W` of word `x / W`, where `W` is the word width.

The figures use 8-bit words so the bits stay legible. binCV's default word is `uint32_t`,
for a reason given [below](#memory-is-decided-by-the-design), so a word holds 32 pixels: a
640-pixel VGA row is 20 words where it would be 640 bytes, and a 640×480 frame is 38,400
bytes where it would be 307,200.

Pixel 0 is the word's **lowest** bit. Written out as a binary number, a row's pixels
therefore appear right to left. It matters in one place below: shifting a word *left* (`<<`)
moves its pixels *right* in the image.

## One instruction, many pixels

The speed comes from one fact: **a word holds many pixels, so one instruction processes
many pixels.** ANDing two masks shows it most directly:

![Two 16-pixel rows ANDed: sixteen instructions when each pixel is a byte, two when each
pixel is a bit in an 8-bit word, with the corresponding loops for a 640-pixel
row](figures/premise-and.svg)

A byte loop does one AND per pixel. A packed loop does one AND per word, and each word is 32
pixels at the default width, so the same row takes 20 instructions instead of 640. A
compiler vectorizes the byte loop too, which narrows that gap; the same vector registers
then hold eight times as many pixels as bits, and
[architectures.md](architectures.md#simd-extensions-bits-against-an-already-vectorized-byte-kernel)
follows that comparison onto real machines.

Most binary operations become word instructions the same way:

| on pixels | on words |
|---|---|
| AND, OR, XOR, NOT of two masks | the same instruction, W pixels per instruction |
| count the set pixels in a region | a population count over the region's words |
| dilate or erode by a neighbour | shift the word, carry in from its neighbour, OR or AND |
| median of three binary values | the bitwise majority `(a & b) \| (b & c) \| (a & c)` |

The shift is the one step with a subtlety, because a pixel near a word boundary has its
neighbour in the next word:

![A row of two 8-bit words dilated by one pixel horizontally: shifted right, shifted left,
and ORed, with the two pixels that cross the word boundary highlighted](figures/premise-shift-or.svg)

Each shifted word is its own bits moved by one, plus the one bit that crosses in from the
neighbouring word. That is two shifts and an OR per word for each direction, and it is the
whole of a horizontal dilation. The vertical direction is cheaper still: the neighbour row is
simply a different pointer.

**Counting is bulk-only.** binCV has no public per-word population count. On aarch64 the
count instruction works on a vector register, so counting one general-purpose word costs two
crossings between register files — about the price of the count itself. Reductions are
offered over regions, masks and windows instead, so those crossings are paid once per
traversal rather than once per word.

## Few-bit images are bit-planes

A pyramid level, a quantized gradient or a small count needs more than one bit per pixel.
The obvious layout packs each pixel's N bits side by side. binCV stores an N-bit image as
**N separate one-bit planes** instead:

![Eight 3-bit pixels stored as three bit-planes, one per bit, and as packed 3-bit fields
where two fields straddle a word boundary](figures/premise-bitplanes.svg)

With packed fields, adding two images means masking every field, guarding every carry from
spilling into its neighbour, and handling the fields that straddle a word boundary. With
bit-planes every pixel sits at the same bit position in every plane, so each plane is still
an ordinary one-bit image and every word operation above still applies. The one-bit image is
not a special case; it is the N = 1 case, and every kernel is written once.

## Arithmetic becomes a circuit

With bit-planes, arithmetic is done the way hardware does it: as a network of logic gates,
except that each gate is a word instruction and processes W pixels at once.

![A bit-sliced adder adding two 2-bit images: four input planes, seven word operations, three
output planes, and a column showing that pixel 0 computes 3 + 2 = 5](figures/premise-adder.svg)

Seven instructions add every pixel in the word, and the count does not depend on how many
pixels the word holds. No carry ever crosses between pixels, because each pixel's carry
lives in its own bit position of the carry word.

The same idea counts per pixel. A **bit-sliced sum** of k one-bit inputs answers, for every
bit position separately, how many of the inputs are set — and it returns planes, not a
number, so the next operation is still word-parallel. That is the opposite of a population
count, which collapses a word to one scalar. The pyramid is built on it:

![A 2×2 box over a 1-bit image: each block's count of 0 to 4 is rescaled to a 3-bit value
and stored as three planes](figures/premise-pyramid.svg)

Four one-bit pixels sum to one of five values, which fit in three bits. The rescale keeps
white at full scale, and it is a comparison against constants rather than a division. Row
pairs are addressed by index, so the vertical half of the downsample moves no bits at all;
the horizontal half is an unshuffle local to each word.

## Signed values, and a covariance with no multiplies

A derivative is signed. In two's complement the sign bit would join every carry, so binCV
stores a signed value as **magnitude planes plus one sign plane**. The case that matters most
is the ternary derivative — values in {−1, 0, +1}, one magnitude plane and a sign plane — and
it turns the gradient covariance that Lucas–Kanade tracking needs into population counts:

![A binary ellipse, its ternary x and y derivatives and their product, with the three
covariance sums computed as population counts over masks](figures/premise-ternary-covariance.svg)

`Ix²` is 1 wherever `Ix` is nonzero, so its sum is the population count of the magnitude
plane. `IxIy` is +1 where the two signs agree and −1 where they differ, so its sum is one
masked count minus another. The whole 2×2 matrix is four population counts, with no multiply
anywhere.

## The invariant that makes it safe

A row whose width is not a multiple of W has padding bits after its last pixel. Word-wise
operations write those bits too:

![A 13-pixel row stored in two 8-bit words: NOT over whole words sets the three padding bits
and the count reads 10, while NOT followed by the tail mask reads the correct 7](figures/premise-padding.svg)

Nothing looks wrong at the NOT — every real pixel is correct, and a bit-exact comparison
against OpenCV passes. The error appears in the next reduction, which counts the padding as
pixels. So **padding bits are always zero**: every operation that writes whole words masks
the last one. That invariant is what lets every reduction run over whole words without
checking where the row ends.

## Memory is decided by the design

A kernel takes **views** — a pointer, a width, a height and a stride — never an owning
container, and it never allocates. Where an operation needs scratch, the caller passes it, so
the memory an operation costs is visible in its signature, and a target with no heap can run
every kernel.

![Peak working set of one call at 640×480 for five operations, binCV against OpenCV, each
row drawn to its own OpenCV total](figures/premise-memory.svg)

The same design refuses allocations rather than shrinking them. Dense stereo disparity
streams a band of rows instead of building a cost volume:

<!-- figure-check values="bytes" source="@docs/reports/stereo.md" -->
| dense disparity, 640×480 | bytes |
|---|---|
| a dense cost volume at this configuration, which binCV does not allocate | 23,101,440 |
| binCV's whole caller-provided scratch | 32,352 |

The default word is `uint32_t` rather than `uint64_t` for the same reason. Wider words are
faster in bulk, but each row's stride rounds up to a whole word, and at the small upper
levels of a pyramid that rounding costs measurably more bytes. Where speed and footprint
conflict, footprint wins.
[footprint.md](../reports/footprint.md#where-speed-was-declined-to-protect-it) prices that
choice. A caller who holds 64-bit words still gets the 32-bit kernels: on little-endian
machines a 64-bit plane already is a 32-bit plane with twice the stride, and binCV reads it
as one without a copy.

## What it buys, and where it stops

Measured against OpenCV on the same content stored as bytes, one thread on each side. Every
ratio is OpenCV's time divided by binCV's, so above 1× binCV is ahead:

<!-- figure-check values="ratio, x86-64|ratio, aarch64" source="source" -->
| operation | ratio, x86-64 | ratio, aarch64 | source |
|---|---|---|---|
| `bitwiseAnd` | 9.97× | 26.7× | [primitives.md](../reports/primitives.md) |
| optical flow, 140 points | 7.19× | 8.27× | [features.md](../reports/features.md) |
| Hamming matching, kNN=2 over 1000×1000 | 4.70× | 1.95× | [features.md](../reports/features.md) |
| `countNonZero` | 1.62× | 2.66× | [primitives.md](../reports/primitives.md) |
| `goodFeaturesToTrack` | 1.38× | 2.42× | [features.md](../reports/features.md) |
| `erode`, 5×5 ellipse | 0.319× | 0.514× | [primitives.md](../reports/primitives.md) |

x86-64 is a desktop Ryzen 5 5600X and aarch64 a Raspberry Pi 4. [The reports](../reports/README.md)
give the intervals, the method and the command behind each row.

The pyramid shows how the result depends on bit depth. The same 2×2 box downsample from
640×480, against `cv::pyrDown` on bytes, at each input and output width:

<!-- figure-check values="x86-64|aarch64" source="@docs/reports/limits.md" -->
| `pyrDown`, bits in → bits out | x86-64 | aarch64 |
|---|---|---|
| **1 → 3, the shipped kernel** | 1.56× | 5.51× |
| 1 → 1 | 0.605× | 5.89× |
| 2 → 2 | 0.772× | 2.52× |
| 3 → 3 | 0.508× | 1.68× |
| 4 → 4 | 0.358× | 1.16× |
| 5 → 5 | 0.272× | 0.797× |
| 8 → 8 | 0.0736× | 0.201× |

On the Raspberry Pi binCV is ahead at every depth through four bits, by up to 5.89×. On the
desktop only the shipped one-bit-in, three-bits-out kernel is ahead: OpenCV's x86-64 pyramid
is a much stronger baseline than its aarch64 one, which
[architectures.md](architectures.md#the-bit-depth-crossover-moves-with-the-machine) looks at.
The rows other than the first run the generic filtered path rather than the shipped kernel;
[limits.md](../reports/limits.md) has both.

Two limits are structural:

- **A structuring element that does not separate** costs one shifted word operation per set
  element, so a 5×5 ellipse pays for every cell it covers. binCV ships it at that speed
  because it holds an eighth of the memory, and where speed and footprint conflict, footprint
  wins.
- **Wide inputs defeat bit-slicing.** An adder needs enough output planes to hold its sum,
  so the work per pixel grows with the input's bit depth, and the pyramid table falls
  steadily as the depth rises. At eight bits in, the bit-sliced form is doing gate by gate
  what a byte kernel's vector unit does in one instruction.

The premise holds where the data is genuinely narrow. [limits.md](../reports/limits.md)
finds each place it does not, on both machines, and
[architectures.md](architectures.md) follows the idea onto each machine binCV has been
measured on.
