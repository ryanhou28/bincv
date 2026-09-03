# binCV API reference

**Generated** by `scripts/gen_api_index.py` from the headers — do not edit.
Every entry is the `@brief` from the declaration itself, so this cannot drift
from the code without the code changing.

## API tiers

| tier | meaning |
|---|---|
| **1** | **bit-exact against OpenCV**, proven by a test |
| **2** | same role and call shape as an OpenCV function, different numerics |
| **3** | no OpenCV equivalent; deliberately does not borrow an OpenCV name |

Anything marked INTERNAL in its docstring is omitted here.

## Contents

- [`binMat.hpp`](#binMathpp) — 14 entries
- [`quantMat.hpp`](#quantMathpp) — 29 entries
- [`ops/bitslice.hpp`](#opsbitslicehpp) — 6 entries
- [`ops/blockMatch.hpp`](#opsblockMatchhpp) — 5 entries
- [`ops/corner.hpp`](#opscornerhpp) — 17 entries
- [`ops/covariance.hpp`](#opscovariancehpp) — 3 entries
- [`ops/denoise.hpp`](#opsdenoisehpp) — 2 entries
- [`ops/derivative.hpp`](#opsderivativehpp) — 6 entries
- [`ops/descriptor.hpp`](#opsdescriptorhpp) — 8 entries
- [`ops/edge.hpp`](#opsedgehpp) — 6 entries
- [`ops/essential.hpp`](#opsessentialhpp) — 6 entries
- [`ops/fast.hpp`](#opsfasthpp) — 6 entries
- [`ops/logic.hpp`](#opslogichpp) — 6 entries
- [`ops/medianWide.hpp`](#opsmedianWidehpp) — 5 entries
- [`ops/morphology.hpp`](#opsmorphologyhpp) — 27 entries
- [`ops/occupancy.hpp`](#opsoccupancyhpp) — 6 entries
- [`ops/opticalFlow.hpp`](#opsopticalFlowhpp) — 21 entries
- [`ops/pack.hpp`](#opspackhpp) — 9 entries
- [`ops/pyramid.hpp`](#opspyramidhpp) — 41 entries
- [`ops/ransac.hpp`](#opsransachpp) — 11 entries
- [`ops/reduce.hpp`](#opsreducehpp) — 19 entries
- [`ops/resample.hpp`](#opsresamplehpp) — 6 entries
- [`ops/shift.hpp`](#opsshifthpp) — 12 entries
- [`ops/subpix.hpp`](#opssubpixhpp) — 3 entries
- [`ops/threshold.hpp`](#opsthresholdhpp) — 2 entries
- [`io/pnm.hpp`](#iopnmhpp) — 3 entries
- [`core/parallel.hpp`](#coreparallelhpp) — 4 entries
- [`core/simd.hpp`](#coresimdhpp) — 3 entries
- [`core/storage.hpp`](#corestoragehpp) — 11 entries
- [`core/types.hpp`](#coretypeshpp) — 6 entries
- [`core/view.hpp`](#coreviewhpp) — 6 entries
- [`threads/pool.hpp`](#threadspoolhpp) — 2 entries

## `binMat.hpp`

[`bincv-cpp/include/bincv-cpp/binMat.hpp`](../bincv-cpp/include/bincv-cpp/binMat.hpp)

| | tier | |
|---|---|---|
| `QuantMat` *(class)* | — | A binary matrix storing one bit per pixel, packed into words |
| `getRowAlignment` | — | Byte alignment this matrix rounds its row stride up to when it allocates |
| `getAlignedWidth` | — | Row stride in words: the distance from one row to the next |
| `empty` | — | True if the matrix has no pixels |
| `ownsMemory` | — | True if this matrix will free its storage; false when it wraps a caller-provided buffer (or is empty) |
| `data` | — | Raw access to the packed storage, for bulk/SIMD operations |
| `sizeInWords` | — | Total number of words in the backing store (height * alignedWidth) |
| `view` | — | Non-owning mutable view over this matrix's pixels |
| `constView` | — | Non-owning read-only view over this matrix's pixels |
| `plane` | — | Bit-plane `i` as a view |
| `constPlane` | — | Plane `i` as a read-only view, from a NON-const matrix |
| `planeWords` | — | Words occupied by one plane -- here, by the whole matrix |
| `at` | — | Gets the value of a single element at (row, col) |
| `set` | — | Sets a single element at (row, col) to value |

## `quantMat.hpp`

[`bincv-cpp/include/bincv-cpp/quantMat.hpp`](../bincv-cpp/include/bincv-cpp/quantMat.hpp)

| | tier | |
|---|---|---|
| `signedMagnitude` | — | |value| as an unsigned, computed WITHOUT negating a signed int |
| `QuantMat` *(class)* | — | An N-bit image, stored as N bit-planes in ONE contiguous allocation |
| `wrap` | — | Wraps a caller-provided buffer, CHECKING that it is long enough |
| `getWidth` | — | Width in pixels |
| `getHeight` | — | Height in pixels, of ONE plane -- not of the plane stack |
| `getAlignedWidth` | — | Row stride in words, shared by every plane |
| `getRowAlignment` | — | Byte alignment rows are rounded up to when this matrix allocates |
| `empty` | — | True if the matrix has no pixels |
| `ownsMemory` | — | True if this matrix will free its storage |
| `data` | — | First word of plane 0 |
| `sizeInWords` | — | Total words backing ALL N planes: N * planeWords |
| `planeWords` | — | Words in one plane: height * stride |
| `plane` | — | Plane `i` as a mutable view, plane 0 being the LEAST significant bit |
| `constPlane` | — | Plane `i` as a read-only view, from a NON-const matrix |
| `at` | — | Reads the N-bit value at (row, col), plane 0 contributing bit 0 |
| `set` | — | Writes the N-bit value at (row, col), plane 0 taking bit 0 |
| `fromCVMat` | 3 | Replaces this matrix with a quantized copy of an 8-bit cv::Mat: each byte v becomes round(v * MaxValue / 255) |
| `toCVMat` | 3 | Writes this matrix as CV_8U holding the RAW values 0..MaxValue |
| `toCVMatNormalized` | 3 | Writes this matrix as CV_8U scaled to the full byte range: round(v * 255 / MaxValue) |
| `toCVMatWith` | — | The shared export loop: 8 pixels x N planes per transpose, then a table lookup per pixel |
| `checkedStackHeight` | — | Rows the plane stack needs: N per image row |
| `SignedQuantMat` *(class)* | — | A signed N-bit image: N magnitude planes plus one sign plane |
| `planes` | — | The underlying uninterpreted container -- this object's only member |
| `magnitude` | — | Magnitude plane `i`, plane 0 being the least significant bit |
| `constMagnitude` | — | Magnitude plane `i` as a read-only view, from a NON-const matrix |
| `sign` | — | The sign plane: a set bit means NEGATIVE |
| `constSign` | — | The sign plane as a read-only view, from a NON-const matrix |
| `magnitudeAt` | — | Reads the magnitude at (row, col), ignoring the sign plane |
| `SignedQuantMat` | — | Adopts an already-validated container |

## `ops/bitslice.hpp`

[`bincv-cpp/include/bincv-cpp/ops/bitslice.hpp`](../bincv-cpp/include/bincv-cpp/ops/bitslice.hpp)

| | tier | |
|---|---|---|
| `bitSlicedSumPlanes` | 3 | Planes a bit-sliced sum of `k` one-bit inputs needs: ceil(log2(k+1)) |
| `maj3` | 3 | Bitwise majority of three words: `(a & b) | (b & c) | (a & c)` |
| `bitSlicedSum` | 3 | Bit-sliced sum of `k` single-bit inputs, lane by lane |
| `thresholdGE` | 3 | Lanes whose bit-sliced value is >= `threshold`, as a 1-bit mask |
| `applyMajority3` | — | The majority3 kernel body: dst = maj3(a, b, c), word-wise, padding cleared |
| `majority3` | 3 | dst = the per-pixel MAJORITY of a, b and c -- which for binary pixels is their MEDIAN |

## `ops/blockMatch.hpp`

[`bincv-cpp/include/bincv-cpp/ops/blockMatch.hpp`](../bincv-cpp/include/bincv-cpp/ops/blockMatch.hpp)

| | tier | |
|---|---|---|
| `BlockMatchParams` *(struct)* | — | Search and window parameters for `calcOpticalFlowBlockMatch` |
| `BlockMatchLevel` *(struct)* | — | One pyramid level for route (a): both frames, and no derivative |
| `blockMatchLevel` | 3 | Names two frames' level into a BlockMatchLevel |
| `parabolicOffset` | — | The vertex of the parabola through `(-1, cm)`, `(0, c0)`, `(+1, cp)` |
| `calcOpticalFlowBlockMatch` | 3 | Pyramidal keypoint tracking by integer Hamming block matching |

## `ops/corner.hpp`

[`bincv-cpp/include/bincv-cpp/ops/corner.hpp`](../bincv-cpp/include/bincv-cpp/ops/corner.hpp)

| | tier | |
|---|---|---|
| `ResponseMap` *(struct)* | 3 | A caller-owned, non-owning view of a `float` response map |
| `ConstResponseMap` *(struct)* | — | The read-only spelling of ResponseMap (the design rule’s two-view-types rule) |
| `Corner` *(struct)* | — | One detected corner: integer pixel coordinates and its response |
| `GoodFeaturesParams` *(struct)* | — | The four parameters `goodFeaturesToTrack` takes, defaulted to the values the reference pipeline actually runs |
| `CornerResult` *(struct)* | — | What `goodFeaturesToTrack` / `selectGoodFeatures` report back |
| `minEigenValue` | — | The smaller eigenvalue of `[[xx, xy], [xy, yy]]`, from exact integers |
| `blockWindow` | — | The window OpenCV's box filter of side `blockSize` reduces at pixel `(x, y)`, anchored where `cv::Point(-1, -1)` puts it |
| `CornerStronger` *(struct)* | — | Strict weak ordering over corners: response DESCENDING, ties broken by DESCENDING raster position -- larger `y` first, then larger `x` |
| `cornerMinEigenVal` | 2 | The minimum-eigenvalue corner response at every pixel, from binarized ternary derivatives |
| `selectGoodFeatures` | 2 | The quality threshold, 3x3 non-maximum suppression and minimum-distance spacing filter `cv::goodFeaturesToTrack` performs, over an existing response map |
| `goodFeaturesToTrack` | 2 | `goodFeaturesToTrack` over a binarized ternary derivative pair: the response map, then the selection |
| `boxHorizontal3` | — | `h = L + C + R` for one bit-plane: one full adder, two output planes |
| `boxVertical3` | — | Sum three 2-bit numbers into four planes (0..9) |
| `boxValueAt` | — | The 0..9 value carried by four bit-planes at bit `bit` |
| `boxWordAt` | — | A word of `row`, or 0 when the row or the index does not exist |
| `cornerMinEigenValRowSliced` | — | One row of `cornerMinEigenVal` at `blockSize == 3`, bit-sliced |
| `goodFeaturesToTrackStreaming` | 2 | `goodFeaturesToTrack` over a THREE-ROW ring instead of a frame-sized response map |

## `ops/covariance.hpp`

[`bincv-cpp/include/bincv-cpp/ops/covariance.hpp`](../bincv-cpp/include/bincv-cpp/ops/covariance.hpp)

| | tier | |
|---|---|---|
| `GradientCovariance` *(struct)* | 3 | The 2x2 Lucas-Kanade gradient covariance over one window: `[sumXX, sumXY; sumXY, sumYY]` |
| `gradientCovariance` | 3 | The 2x2 gradient covariance of a ternary derivative pair over `window`, from ONE traversal and with no scratch |
| `add` | — | Adds a row's partial counts into this one |

## `ops/denoise.hpp`

[`bincv-cpp/include/bincv-cpp/ops/denoise.hpp`](../bincv-cpp/include/bincv-cpp/ops/denoise.hpp)

| | tier | |
|---|---|---|
| `medianRow3` | — | One destination row of the three-pixel median |
| `denoiseMedian3` | 3 | dst[y][x] = median(src[y-1][x], src[y][x], src[y][x+1]), with the out-of-image neighbours reading 0 |

## `ops/derivative.hpp`

[`bincv-cpp/include/bincv-cpp/ops/derivative.hpp`](../bincv-cpp/include/bincv-cpp/ops/derivative.hpp)

| | tier | |
|---|---|---|
| `derivativeAdderStages` | 3 | Adder-class stages one destination word of the derivative costs |
| `derivativeReplicatedInputs` | 3 | Single-bit inputs the REJECTED replication route would need |
| `checkDerivativeArgs` | — | The shape and aliasing contract both derivative kernels take |
| `derivativeX` | 3 | Horizontal binarized derivative: `dst(x, y) = src(x+1, y) - src(x-1, y)`, as sign and magnitude |
| `derivativeY` | 3 | Vertical binarized derivative: `dst(x, y) = src(x, y+1) - src(x, y-1)`, as sign and magnitude |
| `derivativeContainer` | — | Names `src`'s and `dst`'s planes into the arrays the kernels take |

## `ops/descriptor.hpp`

[`bincv-cpp/include/bincv-cpp/ops/descriptor.hpp`](../bincv-cpp/include/bincv-cpp/ops/descriptor.hpp)

| | tier | |
|---|---|---|
| `BriefPair` *(struct)* | — | One intensity comparison, as offsets from the keypoint |
| `BriefPattern` *(struct)* | — | `Bits` comparisons |
| `descriptorWords` | — | Words a `Bits`-bit descriptor occupies |
| `makeBriefPattern` | 3 | Fills a pattern by deterministic Gaussian sampling -- BRIEF's own construction |
| `computeBrief` | 3 | Computes descriptors for `count` keypoints |
| `hammingDistance` | 3 | `popcount(a ^ b)` over `words` |
| `DescriptorMatch` *(struct)* | — | One query's best and second-best match |
| `matchDescriptors` | 3 | Brute-force nearest neighbour with Lowe's ratio test |

## `ops/edge.hpp`

[`bincv-cpp/include/bincv-cpp/ops/edge.hpp`](../bincv-cpp/include/bincv-cpp/ops/edge.hpp)

| | tier | |
|---|---|---|
| `EdgeCombine` *(enum)* | — | How the two axes' results are combined |
| `EdgeRelation` *(enum)* | — | How a gradient is compared with the threshold |
| `EdgeSpatial` *(enum)* | — | Which pixels are differenced |
| `__attribute__` | — | Thirty-two pixels of the edge predicate, as thirty-two bits |
| `edgeFold16` | — | Sixteen byte masks into sixteen bits, LSB first |
| `edgeThreshold` | 3 | Gradient-magnitude edge extraction straight into bits |

## `ops/essential.hpp`

[`bincv-cpp/include/bincv-cpp/ops/essential.hpp`](../bincv-cpp/include/bincv-cpp/ops/essential.hpp)

| | tier | |
|---|---|---|
| `EssentialMatrix` *(struct)* | 2 | A 3x3 essential matrix, row-major |
| `essentialSolverStackBytes` | 3 | Stack the five-point solver uses for one call, in bytes |
| `fivePointEssential` | 2 | Up to ten essential matrices through five correspondences |
| `EssentialModel` *(struct)* | 2 | The five-point model policy, for `bincv::ransac` |
| `residual` | — | Sampson distance -- the first-order approximation of geometric reprojection error, which is what `cv::findEssentialMat`'s threshold is in |
| `findEssentialMat` | 2 | `cv::findEssentialMat(..., cv::RANSAC, ...)`'s role over caller-owned scratch |

## `ops/fast.hpp`

[`bincv-cpp/include/bincv-cpp/ops/fast.hpp`](../bincv-cpp/include/bincv-cpp/ops/fast.hpp)

| | tier | |
|---|---|---|
| `FastCorner` *(struct)* | — | One detected corner |
| `hasFastAvx2` | — | NEON is baseline on aarch64, so there is nothing to dispatch on |
| `detectFast` | 2 | Detects FAST corners |
| `hasFastBitAvx2` | — | Is AVX2 present? |
| `fastArcStepNeon` | — | One doubling step of the arc test, the step a compile-time constant |
| `fastRingLoadNeon` | — | The sixteen ring reads, unrolled so every displacement is an immediate |

## `ops/logic.hpp`

[`bincv-cpp/include/bincv-cpp/ops/logic.hpp`](../bincv-cpp/include/bincv-cpp/ops/logic.hpp)

| | tier | |
|---|---|---|
| `applyBinary` | — | The two-input kernel body: dst = Op(a, b), word-wise, padding cleared |
| `applyUnary` | — | The one-input kernel body: dst = Op(src), word-wise, padding cleared |
| `bitwiseAnd` | 1 | dst = a & b, pixel for pixel |
| `bitwiseOr` | 1 | dst = a | b, pixel for pixel |
| `bitwiseXor` | 1 | dst = a ^ b, pixel for pixel |
| `bitwiseNot` | 1 | dst = ~src, pixel for pixel |

## `ops/medianWide.hpp`

[`bincv-cpp/include/bincv-cpp/ops/medianWide.hpp`](../bincv-cpp/include/bincv-cpp/ops/medianWide.hpp)

| | tier | |
|---|---|---|
| `MedianOffset` *(struct)* | — | One sample position, relative to the pixel being written |
| `MedianPattern` *(struct)* | — | A neighbourhood: `K` offsets, `K` odd so the median is a single element |
| `hasMedianSimd` | — | Is AVX2 present? |
| `med3Store` | — | `med3` for sixteen pixels |
| `medianWide` | 3 | Median filter over a caller-chosen neighbourhood |

## `ops/morphology.hpp`

[`bincv-cpp/include/bincv-cpp/ops/morphology.hpp`](../bincv-cpp/include/bincv-cpp/ops/morphology.hpp)

| | tier | |
|---|---|---|
| `StructuringElement` *(struct)* | — | A morphological structuring element: a shape, an extent and an anchor |
| `rect` | — | `cv::getStructuringElement(MORPH_RECT, {c, r}, anchor)` |
| `cross` | — | `cv::getStructuringElement(MORPH_CROSS, {c, r}, anchor)` |
| `ellipse` | — | `cv::getStructuringElement(MORPH_ELLIPSE, {c, r}, anchor)` |
| `custom` | — | An arbitrary caller-owned mask; `m` must outlive the element |
| `anchorCol` | — | The anchor column with OpenCV's `-1 == center` resolved |
| `anchorRow` | — | The anchor row with OpenCV's `-1 == center` resolved |
| `activeAt` | — | True when cell (col, row) is part of the element |
| `spanOfRow` | — | The half-open column range `[first, last)` of row `row` that MAY be set: exact for the parametric shapes, `[0, cols)` for a mask |
| `spanIsDense` | — | True when every cell inside `spanOfRow` is set, so a kernel that iterates the span needs no per-cell test at all |
| `valid` | — | Extents positive, anchor inside the element, at least one set cell |
| `rect3x3` | — | The 3x3 rectangle -- `cv::Mat` passed to `cv::erode`, i.e |
| `cross3x3` | — | The 3x3 plus -- what BOTH `MORPH_CROSS` and `MORPH_ELLIPSE` give at 3x3 |
| `MorphPath` *(enum)* | — | Which implementation `morphApply` may take |
| `MorphFold` *(struct)* | — | The combining operation and its identity, as a compile-time choice |
| `morphShiftedWord` | — | Word `i` of `srcRow` shifted so that destination column `c` reads source column `c + dx`, with everything outside the row reading `fill` |
| `morphMaxOffsetX` | — | The element's horizontal reach: max |cell - anchor| over set cells |
| `morphFixupPixel` | — | One destination pixel, recomputed from the whole element with every source coordinate mapped through `borderIndex` |
| `morphFixupRowBorder` | — | Rewrites the destination columns whose source column can leave the row, one pixel at a time, for the four NON-CONSTANT border types |
| `morphRowGeneric` | — | One destination row, general element |
| `morphRow3x3` | — | One destination row, 3x3 element anchored at its center |
| `morphApply` | — | erode (IsErode) or dilate, whole image |
| `morphArgumentsAreSane` | — | The preconditions `erode` and `dilate` share, in one place |
| `erode` | 1 | Morphological erosion: `dst(x,y) = AND over the element of src(x+dx, y+dy)` |
| `dilate` | 1 | Morphological dilation: `dst(x,y) = OR over the element of src(x+dx, y+dy)` |
| `morphologyExNeedsScratch` | — | True when `morphologyEx(op,...)` reads and writes its scratch view |
| `morphologyEx` | 1 | The seven `MorphOp` compositions |

## `ops/occupancy.hpp`

[`bincv-cpp/include/bincv-cpp/ops/occupancy.hpp`](../bincv-cpp/include/bincv-cpp/ops/occupancy.hpp)

| | tier | |
|---|---|---|
| `spaceCandidates` | 3 | Keeps the candidates that are at least `radius` from every live point and from every candidate already kept |
| `clearOccupancy` | 3 | Zeroes an occupancy mask |
| `markDisc` | 3 | Sets every pixel strictly within `radius` of `(cx, cy)` |
| `markOccupied` | 3 | Stamps `markDisc` for every point |
| `occupied` | 3 | Is the pixel `(x, y)` claimed? |
| `spaceCandidatesMasked` | 3 | `spaceCandidates` through an occupancy mask: test one bit, and stamp the disc of every candidate kept |

## `ops/opticalFlow.hpp`

[`bincv-cpp/include/bincv-cpp/ops/opticalFlow.hpp`](../bincv-cpp/include/bincv-cpp/ops/opticalFlow.hpp)

| | tier | |
|---|---|---|
| `LKEntryLevel` *(enum)* | 2 | A subpixel point |
| `LKLevel` *(struct)* | — | One pyramid level's six planes: both frames, and the previous frame's ternary derivative |
| `lkLevel` | 2 | Names a level's containers into an LKLevel |
| `LKLevelN` *(struct)* | — | One pyramid level at N bits per pixel: both frames' bit-planes, and the previous frame's N-bit signed derivative |
| `StageTiming` *(struct)* | — | Where `track`'s time actually goes, by stage |
| `referenceMinEigScale` | — | `kReferenceMinEigScale` at an arbitrary bit depth |
| `floorDiv` | — | `floor(a / b)` for integers with `b > 0`, rounding toward MINUS infinity |
| `sourceWord` | — | The source word `k`, with the trailing partial word masked and any index outside the row reading as zero (the replicate fill covers it) |
| `word` | — | Bits of the displaced row lying under word `i` of the window grid |
| `displacedRow` | — | Builds a displaced reader for row `y` of `plane`, clamped vertically |
| `TapSums` *(struct)* | — | The five integer sums one gradient component's residual needs |
| `combine` | — | `w00*t00 + w01*t01 + w10*t10 + w11*t11 - self` |
| `floorToLL` | — | `floor(v)` as a `long long`, for a value already known to be finite and within the frame's range |
| `IterationTrace` *(struct)* | — | that measurement’s iteration counter |
| `windowFitsAtLevel` | — | Is point `p`'s window entirely inside level `li`? |
| `entryLevelFor` | — | The coarsest usable level whose window contains point `p`, or 0 |
| `calcOpticalFlowPyrLK` | 2 | Pyramidal Lucas-Kanade tracking of sparse keypoints between two binary frames |
| `narrowLevel` | 2 | Pyramidal Lucas-Kanade over a ladder of levels that are all the SAME depth `N` |
| `lkPathName` | 3 | Which residual kernel this level type will actually run, as a string |
| `LKLevels` *(struct)* | 2 | A tracking ladder whose levels have DIFFERENT bit depths, level 0 first |
| `stagingStackBytes` | 3 | Stack bytes the tracker's staging buffers occupy at `(N, WordType)` |

## `ops/pack.hpp`

[`bincv-cpp/include/bincv-cpp/ops/pack.hpp`](../bincv-cpp/include/bincv-cpp/ops/pack.hpp)

| | tier | |
|---|---|---|
| `PackRule` *(enum)* | — | How a source pixel becomes a bit |
| `packRows` | 3 | Packs `rowCount` rows into `dst` starting at `dstRow` |
| `packBits` | 3 | Packs a pixel array to one bit per pixel |
| `QuantRule` *(enum)* | — | How a source pixel becomes an N-bit value |
| `packQuant` | 3 | Packs a pixel array to N bits per pixel, no OpenCV |
| `packQuantWith` | 3 | `packQuant` with an arbitrary per-pixel map |
| `packBitsIf` | 3 | `packBits` with an arbitrary per-pixel predicate |
| `unpackTo8Bit` | 3 | The reverse: one bit per pixel out to one byte per pixel |
| `writePgm` | 3 | Writes a binary image as a binary PGM (`P5`) to a caller-supplied buffer |

## `ops/pyramid.hpp`

[`bincv-cpp/include/bincv-cpp/ops/pyramid.hpp`](../bincv-cpp/include/bincv-cpp/ops/pyramid.hpp)

| | tier | |
|---|---|---|
| `pyrDownWidth` | 3 | Destination width of one pyramid level: ceil(srcWidth / 2) |
| `pyrDownHeight` | 3 | Destination height of one pyramid level: ceil(srcHeight / 2) |
| `boxSumFullAdders` | — | Full-adder stages the 2x2 box sum costs at a given source depth |
| `boxSum4ReplicatedInputs` | — | Single-bit inputs the REJECTED replication route would need |
| `pyrDownAdderStages` | — | Total adder-class stages one destination word costs: box plus rescale |
| `pyrDownAutomaticWords` | — | Words of automatic storage the shipped `pyrDown` DECLARES per destination word |
| `addPlanes` | — | `out = a + b`, bit-sliced |
| `multiplyByAllOnes` | — | `out = (v << shift) - v`, i.e |
| `addConstant` | — | `v += c` in every lane, bit-sliced, `c` an ordinary integer constant |
| `subtractConstantWhere` | — | `v -= c` in the lanes where `mask` is set, bit-sliced |
| `divideByConstant` | — | `quotient = floor(value / divisor)`, bit-sliced, divisor a constant |
| `boxSum4` | — | `sum = a + b + c + d`, four NIn-bit bit-sliced operands, NIn+2 planes |
| `boxSum4Replicated` | — | The same sum through ops/bitslice.hpp's SINGLE-BIT adder network |
| `requantizeBoxSum` | — | `dst = round(sum / 4 * (2^NOut - 1) / (2^NIn - 1))`, bit-sliced |
| `checkPyrDownArgs` | — | The shape and aliasing contract pyrDown's kernel takes |
| `gatherPhases` | — | The four 2x2 phases of one destination word, for one source plane |
| `PyrDownFilter` *(enum)* | — | Which downsampling filter `pyrDown` applies |
| `PyrDownBorder` *(enum)* | — | What a filter reads outside the frame |
| `FilterTaps` *(struct)* | — | Tap offsets and weights for one separable filter |
| `replicateIndex` | — | Planes needed to hold one axis of the weighted sum of `NIn`-bit values |
| `srcPixelValue` | — | One source pixel's value across all NIn planes |
| `setPixelValue` | — | Writes value `v` at `(y, x)` across NOut destination planes |
| `addShifted` | — | `acc += (v << Shift)`, bit-sliced |
| `weightedAxisStage` | — | One (tap, weight-bit) stage of `weightedAxis`, unrolled at compile time |
| `weightedAxis` | — | One axis of a separable weighted sum, bit-sliced |
| `divideStage` | — | `requantizeBoxSum` for an arbitrary kernel weight sum |
| `divideByConstantT` | — | `divideByConstant` with the divisor and the quotient width known at compile time |
| `pyrDownBoxViews` | 2 | One pyramid level: 2x2 box mean of `src`, subsampled, at NOut bits |
| `pyrDownReplicated` | — | The same, taking containers, for the benchmark's convenience |
| `pyrDownBoxContainers` | 2 | The QuantMat spelling of pyrDown |
| `Pyramid` *(class)* | 2 | A pyramid: one QuantMat per level, each at its own bit depth |
| `levelBits` | — | Bits per pixel at level I |
| `level` | — | Level I, mutable |
| `build` | — | Fills levels 1..N-1 by running pyrDown down the ladder |
| `sizeInWords` | — | Total words across every level -- the pyramid's whole footprint |
| `sizeInBytes` | — | Total bytes across every level |
| `phaseAtMinus1` | — | The value at output column x-1, and at x+1, of a phase word |
| `leftRimColumns` | — | Output columns at each edge whose source support crosses the frame, and which therefore need the per-pixel border rule rather than the word-parallel path |
| `pyrDownFiltered` | 3 | One pyramid level under a chosen downsampling filter |
| `pyrDownBox` | 2 | One pyramid level by 2x2 box mean, `BORDER_REPLICATE` |
| `pyrDown` | 1 | One pyramid level, EXACTLY as `cv::pyrDown` computes it: a 5x5 `[1,4,6,4,1]` Gaussian with `BORDER_REFLECT_101`, subsampled by 2 |

## `ops/ransac.hpp`

[`bincv-cpp/include/bincv-cpp/ops/ransac.hpp`](../bincv-cpp/include/bincv-cpp/ops/ransac.hpp)

| | tier | |
|---|---|---|
| `RansacParams` *(struct)* | 2 | The four parameters a RANSAC call takes |
| `RansacResult` *(struct)* | 2 | What a RANSAC call reports back |
| `RansacScratch` *(struct)* | 3 | Caller-owned scratch: one flag per correspondence, twice |
| `ransacScratchWords` | — | Words in one inlier set over `correspondences` points |
| `ransacScratchBytes` | — | Bytes of scratch a call over `correspondences` points needs |
| `ransac` | 2 | Fit `Model` to the correspondences by random sample consensus |
| `Affine2D` *(struct)* | 2 | A 2D affine transform, row-major: `[m[0] m[1] m[2]; m[3] m[4] m[5]]` |
| `Affine2DModel` *(struct)* | 2 | The 3-point affine model policy |
| `estimate` | — | Solve the affine through the three sampled correspondences |
| `residual` | — | Euclidean reprojection error, in pixels -- the units `cv::estimateAffine2D`'s `ransacReprojThreshold` is in |
| `estimateAffine2D` | 2 | `cv::estimateAffine2D(..., RANSAC, ...)`'s role over caller-owned scratch |

## `ops/reduce.hpp`

[`bincv-cpp/include/bincv-cpp/ops/reduce.hpp`](../bincv-cpp/include/bincv-cpp/ops/reduce.hpp)

| | tier | |
|---|---|---|
| `SplitCount` *(struct)* | — | The two halves of a split count: pixels where the selector `c` was clear, and pixels where it was set |
| `crossTerm` | — | The LK cross term: `whenClear - whenSet`, signed (the design notes) |
| `CovarianceCount` *(struct)* | 3 | The four numbers of a 2x2 gradient covariance over one region: popcount(a), popcount(b), and the split of `a & b` by the selector |
| `RegionWords` *(struct)* | — | A region clipped to a view, expressed in the words a row loop walks |
| `regionFromExtent` | — | Region geometry from an already-clipped, non-empty pixel extent |
| `wholeViewWords` | — | The whole of a view: every pixel, no region |
| `clipColumns` | — | The COLUMN half of a clip: a band of columns over EVERY row of a view |
| `clipRegion` | — | Intersects a Rect with a view's extent, in words |
| `visitRowWords` | — | Visits every word index of one region-row, ascending, exactly once |
| `countViewRegion` | — | popcount of one view over an already-clipped region |
| `countNonZero` | 1 | Number of set pixels in `src` |
| `countAnd` | 3 | Number of pixels set in BOTH `a` and `b` inside `region` |
| `countAndSplit` | 3 | popcount(a & b & ~c) and popcount(a & b & c) over `region`, in ONE pass |
| `countCovariance` | 3 | All four numbers of the 2x2 gradient covariance over `region`, from ONE traversal |
| `SlidingWindowCount` *(class)* | 3 | A window count slid DOWNWARD one pixel row at a time: the sum gains the incoming row's windowed popcount and loses the outgoing row's |
| `count` | — | Set pixels inside the current window, intersected with the image |
| `slideDown` | — | Advances the window one pixel row down |
| `window` | — | The window this accumulator is currently reporting, unclipped |
| `alive` | — | False when no position of this column can ever count anything: an empty column band, or a non-positive window height |

## `ops/resample.hpp`

[`bincv-cpp/include/bincv-cpp/ops/resample.hpp`](../bincv-cpp/include/bincv-cpp/ops/resample.hpp)

| | tier | |
|---|---|---|
| `decimatedWidth` | 3 | Destination width for a horizontal decimation by two |
| `rowsDecimatedBy2` | 3 | The FREE half of a 2x2 subsample: every other row, as a view |
| `checkDecimateArgs` | — | The shape and aliasing contract every decimation kernel here shares |
| `decimateColumnsBy2Gather` | — | variant A: horizontal decimation one destination pixel at a time |
| `decimateColumnsBy2FrameMasked` | — | variant C: horizontal decimation as a big-integer unshuffle |
| `decimateColumnsBy2` | 3 | Horizontal decimation by two: `dst(y, j) = src(y, 2j)` |

## `ops/shift.hpp`

[`bincv-cpp/include/bincv-cpp/ops/shift.hpp`](../bincv-cpp/include/bincv-cpp/ops/shift.hpp)

| | tier | |
|---|---|---|
| `maxShiftOffset` | — | Largest shift magnitude these kernels accept, in pixels |
| `isKnownBorderType` | — | True for the five BorderType values core/types.hpp defines |
| `borderIndex` | — | cv::borderInterpolate: the source index a coordinate outside [0, len) extrapolates to, or -1 when the border is a constant |
| `extendedRowWord` | — | Word `j` of a row, with everything outside the row's PIXELS reading `fill` -- both whole words past the row and the padding bits of the last |
| `fillRowWords` | — | Writes one destination row as a constant, padding bits left zero |
| `shiftRowHorizontal` | — | One row of horizontal shift by `dx` columns, filling with `fill` |
| `fixupHorizontalBorder` | — | Rewrites the destination columns whose source column lies outside the row, for the four NON-CONSTANT border types |
| `shift` | 3 | dst[y][x] = src[y + dy][x + dx], extrapolating outside the image |
| `shiftLeft` | 3 | dst[y][x] = src[y][x + k] -- moves the image LEFT by k columns |
| `shiftRight` | 3 | dst[y][x] = src[y][x - k] -- moves the image RIGHT by k columns |
| `shiftUp` | 3 | dst[y][x] = src[y + k][x] -- moves the image UP by k rows |
| `shiftDown` | 3 | dst[y][x] = src[y - k][x] -- moves the image DOWN by k rows |

## `ops/subpix.hpp`

[`bincv-cpp/include/bincv-cpp/ops/subpix.hpp`](../bincv-cpp/include/bincv-cpp/ops/subpix.hpp)

| | tier | |
|---|---|---|
| `SubPixParams` *(struct)* | — | `cv::cornerSubPix`'s `winSize`, `zeroZone` and `criteria`, in one struct |
| `SubPixResult` *(struct)* | 3 | What `cornerSubPix` did, per corner |
| `cornerSubPix` | 2 | Refines corner positions to sub-pixel accuracy |

## `ops/threshold.hpp`

[`bincv-cpp/include/bincv-cpp/ops/threshold.hpp`](../bincv-cpp/include/bincv-cpp/ops/threshold.hpp)

| | tier | |
|---|---|---|
| `binarize` | 3 | dst = (src > thresh), pixel for pixel, over an N-plane bit-sliced source |
| `threshold` | 1 | dst = (src > thresh), packing a CV_8U image into one bit per pixel |

## `io/pnm.hpp`

[`bincv-cpp/include/bincv-cpp/io/pnm.hpp`](../bincv-cpp/include/bincv-cpp/io/pnm.hpp)

| | tier | |
|---|---|---|
| `PgmHeader` *(struct)* | — | What a `readPgm` call found, or why it did not |
| `readPgmHeader` | 3 | Parses a binary PGM (`P5`) header |
| `readPgm` | 3 | Reads a binary PGM straight into bits, under a `PackRule` |

## `core/parallel.hpp`

[`bincv-cpp/include/bincv-cpp/core/parallel.hpp`](../bincv-cpp/include/bincv-cpp/core/parallel.hpp)

| | tier | |
|---|---|---|
| `setParallelForBackend` | — | Installs a parallel-for backend |
| `setNumThreads` | — | How many threads binCV may use |
| `getNumThreads` | — | The current thread count |
| `parallelFor` | — | Runs `body(i, ctx)` for `i` in `[0, n)` |

## `core/simd.hpp`

[`bincv-cpp/include/bincv-cpp/core/simd.hpp`](../bincv-cpp/include/bincv-cpp/core/simd.hpp)

| | tier | |
|---|---|---|
| `SimdStatus` *(struct)* | — | What this translation unit compiled, and what the CPU under it supports |
| `simdStatus` | 3 | What vector paths are actually active |
| `simdStatusString` | 3 | One line naming every fast path and whether it is on |

## `core/storage.hpp`

[`bincv-cpp/include/bincv-cpp/core/storage.hpp`](../bincv-cpp/include/bincv-cpp/core/storage.hpp)

| | tier | |
|---|---|---|
| `Storage` *(class)* | — | Backing memory for a bit-packed matrix: {pointer, word count, ownership} |
| `Storage` | — | Allocates and zero-fills `words` words, owned by this object |
| `data` | — | First word of the buffer, or nullptr when empty |
| `size` | — | Buffer size in WORDS, not bytes |
| `empty` | — | True when the buffer holds no words |
| `ownsMemory` | — | True when this object will free the buffer on destruction |
| `copyWords` | — | Copies `words` words |
| `aliasesOwnedBlock` | — | True if `p` points into the block this object owns |
| `adoptThenFree` | — | Installs a new descriptor, then frees the block this object held |
| `release` | — | Releases the buffer if owned, and resets to the empty state so a freed pointer can never survive the call |
| `clear` | — | Resets to the empty, non-owning state without freeing anything |

## `core/types.hpp`

[`bincv-cpp/include/bincv-cpp/core/types.hpp`](../bincv-cpp/include/bincv-cpp/core/types.hpp)

| | tier | |
|---|---|---|
| `Size` *(struct)* | — | Size structure representing width and height |
| `area` | — | Calculate the area (width * height) |
| `empty` | — | Check if the size is empty (zero width or height) |
| `Rect` *(struct)* | — | An axis-aligned rectangle in PIXELS: origin (x, y), extent (width, height) |
| `QuantMat` *(class)* | — | Forward declaration of the QuantMat template -- the N-bit container |
| `Point2f` *(struct)* | — | A point with sub-pixel coordinates -- the tracker's and the refiner's |

## `core/view.hpp`

[`bincv-cpp/include/bincv-cpp/core/view.hpp`](../bincv-cpp/include/bincv-cpp/core/view.hpp)

| | tier | |
|---|---|---|
| `BinMatView` *(struct)* | — | Non-owning, mutable view of a bit-packed matrix: {ptr, size, stride} |
| `empty` | — | True if the view addresses no pixels |
| `row` | — | First word of row y |
| `BinMatConstView` *(struct)* | — | Non-owning, read-only view of a bit-packed matrix |
| `narrowPlane` | 3 | Reads a 64-bit bit-plane as a 32-bit one |
| `narrowPlaneMutable` | 3 | The same reinterpretation for a WRITABLE plane |

## `threads/pool.hpp`

[`bincv-cpp/include/bincv-cpp/threads/pool.hpp`](../bincv-cpp/include/bincv-cpp/threads/pool.hpp)

| | tier | |
|---|---|---|
| `ThreadPool` *(class)* | — | A minimal fixed-size pool that serves `bincv::parallelFor` |
| `install` | — | Makes this pool binCV's backend and sets the thread count to match |

