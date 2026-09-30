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

- [`binMat.hpp`](#binMathpp) — 29 entries
- [`quantMat.hpp`](#quantMathpp) — 28 entries
- [`util.hpp`](#utilhpp) — 1 entries
- [`ops/bitslice.hpp`](#opsbitslicehpp) — 5 entries
- [`ops/blockMatch.hpp`](#opsblockMatchhpp) — 4 entries
- [`ops/census.hpp`](#opscensushpp) — 5 entries
- [`ops/corner.hpp`](#opscornerhpp) — 11 entries
- [`ops/covariance.hpp`](#opscovariancehpp) — 2 entries
- [`ops/denoise.hpp`](#opsdenoisehpp) — 1 entries
- [`ops/denseDisparity.hpp`](#opsdenseDisparityhpp) — 7 entries
- [`ops/derivative.hpp`](#opsderivativehpp) — 4 entries
- [`ops/descriptor.hpp`](#opsdescriptorhpp) — 14 entries
- [`ops/edge.hpp`](#opsedgehpp) — 4 entries
- [`ops/essential.hpp`](#opsessentialhpp) — 7 entries
- [`ops/fast.hpp`](#opsfasthpp) — 2 entries
- [`ops/logic.hpp`](#opslogichpp) — 4 entries
- [`ops/medianWide.hpp`](#opsmedianWidehpp) — 3 entries
- [`ops/morphology.hpp`](#opsmorphologyhpp) — 17 entries
- [`ops/occupancy.hpp`](#opsoccupancyhpp) — 6 entries
- [`ops/opticalFlow.hpp`](#opsopticalFlowhpp) — 10 entries
- [`ops/orbPattern.hpp`](#opsorbPatternhpp) — 1 entries
- [`ops/orientation.hpp`](#opsorientationhpp) — 1 entries
- [`ops/pack.hpp`](#opspackhpp) — 10 entries
- [`ops/pyramid.hpp`](#opspyramidhpp) — 15 entries
- [`ops/ransac.hpp`](#opsransachpp) — 12 entries
- [`ops/reduce.hpp`](#opsreducehpp) — 12 entries
- [`ops/resample.hpp`](#opsresamplehpp) — 3 entries
- [`ops/shift.hpp`](#opsshifthpp) — 5 entries
- [`ops/stereo.hpp`](#opsstereohpp) — 5 entries
- [`ops/subpix.hpp`](#opssubpixhpp) — 3 entries
- [`ops/threshold.hpp`](#opsthresholdhpp) — 2 entries
- [`io/pnm.hpp`](#iopnmhpp) — 7 entries
- [`io/sequence.hpp`](#iosequencehpp) — 9 entries
- [`core/parallel.hpp`](#coreparallelhpp) — 4 entries
- [`core/simd.hpp`](#coresimdhpp) — 3 entries
- [`core/storage.hpp`](#corestoragehpp) — 11 entries
- [`core/types.hpp`](#coretypeshpp) — 6 entries
- [`core/view.hpp`](#coreviewhpp) — 6 entries
- [`threads/pool.hpp`](#threadspoolhpp) — 2 entries

## `binMat.hpp`

[`include/bincv/binMat.hpp`](../include/bincv/binMat.hpp)

| | tier | |
|---|---|---|
| `QuantMat` *(class)* | 3 | A binary matrix storing one bit per pixel, packed into words |
| `getRowAlignment` | — | Byte alignment this matrix rounds its row stride up to when it allocates |
| `getAlignedWidth` | — | Row stride in words: the distance from one row to the next |
| `empty` | — | True if the matrix has no pixels |
| `ownsMemory` | — | True if this matrix will free its storage; false when it wraps a caller-provided buffer (or is empty) |
| `rows` | — | The dimensions under the names cv::Mat gives its `rows` and `cols` data members, as accessors |
| `data` | — | Raw access to the packed storage, for bulk/SIMD operations |
| `sizeInWords` | — | Total number of words in the backing store (height * alignedWidth) |
| `view` | — | Non-owning mutable view over this matrix's pixels |
| `constView` | — | Non-owning read-only view over this matrix's pixels |
| `plane` | — | Bit-plane `i` as a view |
| `constPlane` | — | Plane `i` as a read-only view, from a NON-const matrix |
| `planeWords` | — | Words occupied by one plane -- here, by the whole matrix |
| `fromCVMat` | 3 | Replaces this matrix with a binarized copy of a `CV_8UC1` cv::Mat: any nonzero byte becomes 1 |
| `toCVMat` | 3 | Writes this matrix as a `CV_8UC1` cv::Mat holding 0 or 1 per pixel |
| `toCVMatNormalized` | 3 | Writes this matrix as a `CV_8UC1` cv::Mat holding 0 or 255 per pixel, which is what an image viewer or an OpenCV operation expects of a binary image |
| `at` | 1 | Gets the value of a single element at (row, col) |
| `set` | 1 | Sets a single element at (row, col) to value |
| `ptr` | 3 | First word of row `row`, read-only |
| `resize` | 3 | Reshapes the matrix to `newWidth` x `newHeight`, keeping the pixels that still fit and zero-filling the rest |
| `pad` | 3 | Adds `top`, `bottom`, `left` and `right` pixels of border, filled with `value` |
| `transposed` | 3 | A transposed copy of this matrix |
| `transpose` | 3 | Transposes this matrix |
| `forEachNonZero` | 3 | Iterates over all non-zero pixels, invoking callback(row, col) |
| `printMatrix` | 3 | Prints the pixels as 0/1 characters, one row per line, to stdout |
| `printInternalData` | 3 | Prints the packed words of the backing store, one row per line, to stdout |
| `fill` | 3 | Sets every pixel to `value` |
| `countNonZero` | 1 | The number of set pixels |
| `sparsity` | 3 | The fraction of pixels that are zero, in [0.0, 1.0] |

## `quantMat.hpp`

[`include/bincv/quantMat.hpp`](../include/bincv/quantMat.hpp)

| | tier | |
|---|---|---|
| `QuantMat` *(class)* | 3 | An N-bit image, stored as N bit-planes in ONE contiguous allocation |
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
| `SignedQuantMat` *(class)* | 3 | A signed N-bit image: N magnitude planes plus one sign plane |
| `planes` | — | The underlying uninterpreted container -- this object's only member |
| `magnitude` | — | Magnitude plane `i`, plane 0 being the least significant bit |
| `constMagnitude` | — | Magnitude plane `i` as a read-only view, from a NON-const matrix |
| `sign` | — | The sign plane: a set bit means NEGATIVE |
| `constSign` | — | The sign plane as a read-only view, from a NON-const matrix |
| `magnitudeAt` | — | Reads the magnitude at (row, col), ignoring the sign plane |
| `SignedQuantMat` | — | Adopts an already-validated container |

## `util.hpp`

[`include/bincv/util.hpp`](../include/bincv/util.hpp)

| | tier | |
|---|---|---|
| `save_test_image` | 3 | Writes an 8-bit image to `tests/output/<imageName>` through `cv::imwrite` |

## `ops/bitslice.hpp`

[`include/bincv/ops/bitslice.hpp`](../include/bincv/ops/bitslice.hpp)

| | tier | |
|---|---|---|
| `bitSlicedSumPlanes` | 3 | Planes a bit-sliced sum of `k` one-bit inputs needs: ceil(log2(k+1)) |
| `maj3` | 3 | Bitwise majority of three words: `(a & b) | (b & c) | (a & c)` |
| `bitSlicedSum` | 3 | Bit-sliced sum of `k` single-bit inputs, lane by lane |
| `thresholdGE` | 3 | Lanes whose bit-sliced value is >= `threshold`, as a 1-bit mask |
| `majority3` | 3 | dst = the per-pixel MAJORITY of a, b and c -- which for binary pixels is their MEDIAN |

## `ops/blockMatch.hpp`

[`include/bincv/ops/blockMatch.hpp`](../include/bincv/ops/blockMatch.hpp)

| | tier | |
|---|---|---|
| `BlockMatchParams` *(struct)* | 3 | Search and window parameters for `calcOpticalFlowBlockMatch` |
| `BlockMatchLevel` *(struct)* | 3 | One pyramid level for block matching: both frames, and no derivative |
| `blockMatchLevel` | 3 | Names two frames' level into a BlockMatchLevel |
| `calcOpticalFlowBlockMatch` | 3 | Pyramidal keypoint tracking by integer Hamming block matching |

## `ops/census.hpp`

[`include/bincv/ops/census.hpp`](../include/bincv/ops/census.hpp)

| | tier | |
|---|---|---|
| `CensusOffset` *(struct)* | 3 | One census comparison offset, relative to the pixel being written |
| `CensusPattern` *(struct)* | 3 | A census neighbourhood: `K` offsets, none of them (0, 0) |
| `kCensus3x3` *(constant)* | 3 | The 8-neighbour census (3x3 minus center), raster order |
| `kCensus5x5` *(constant)* | 3 | The 24-comparison census (5x5 minus center), raster order -- the neighbourhood the dense-stereo design is written against |
| `censusTransform` | 3 | Census transform: plane `k` of `planes` gets `I(p + pattern.at[k]) > I(p)` at every pixel `p` |

## `ops/corner.hpp`

[`include/bincv/ops/corner.hpp`](../include/bincv/ops/corner.hpp)

| | tier | |
|---|---|---|
| `ResponseMap` *(struct)* | 3 | A caller-owned, non-owning view of a `float` response map |
| `ConstResponseMap` *(struct)* | 3 | The read-only spelling of ResponseMap (the two-view-types rule) |
| `Corner` *(struct)* | 2 | One detected corner: integer pixel coordinates and its response |
| `GoodFeaturesParams` *(struct)* | 2 | The four parameters `goodFeaturesToTrack` takes, defaulted to the values the reference pipeline actually runs |
| `CornerResult` *(struct)* | 3 | What `goodFeaturesToTrack` / `selectGoodFeatures` report back |
| `cornerMinEigenVal` | 2 | The minimum-eigenvalue corner response at every pixel, from binarized ternary derivatives |
| `selectGoodFeatures` | 2 | The quality threshold, 3x3 non-maximum suppression and minimum-distance spacing filter `cv::goodFeaturesToTrack` performs, over an existing response map |
| `goodFeaturesToTrack` | 2 | `goodFeaturesToTrack` over a binarized ternary derivative pair: the response map, then the selection |
| `kResponseRingRows` *(constant)* | 3 | Rows the streaming form's ring must have |
| `cornerMinEigenValRow` | 2 | One ROW of the minimum-eigenvalue response map |
| `goodFeaturesToTrackStreaming` | 2 | `goodFeaturesToTrack` over a THREE-ROW ring instead of a frame-sized response map |

## `ops/covariance.hpp`

[`include/bincv/ops/covariance.hpp`](../include/bincv/ops/covariance.hpp)

| | tier | |
|---|---|---|
| `GradientCovariance` *(struct)* | 3 | The 2x2 Lucas-Kanade gradient covariance over one window: `[sumXX, sumXY; sumXY, sumYY]` |
| `gradientCovariance` | 3 | The 2x2 gradient covariance of a ternary derivative pair over `window`, from ONE traversal and with no scratch |

## `ops/denoise.hpp`

[`include/bincv/ops/denoise.hpp`](../include/bincv/ops/denoise.hpp)

| | tier | |
|---|---|---|
| `denoiseMedian3` | 3 | dst[y][x] = median(src[y-1][x], src[y][x], src[y][x+1]), with the out-of-image neighbours reading 0 |

## `ops/denseDisparity.hpp`

[`include/bincv/ops/denseDisparity.hpp`](../include/bincv/ops/denseDisparity.hpp)

| | tier | |
|---|---|---|
| `kDenseDisparityInvalid` *(constant)* | — | The disparity byte written where no candidate could be evaluated |
| `DenseDisparityParams` *(struct)* | 3 | Search and aggregation parameters for `denseDisparity` |
| `denseDisparityScratchWords` | 3 | WordType units of scratch `denseDisparity` needs: the two census bands and the accumulator ladder |
| `denseDisparityScratchRows` | 3 | uint16_t units of scratch `denseDisparity` needs: the extraction row and the two running-best rows |
| `denseDisparity` | 3 | Dense disparity over a rectified pair: census cost, box aggregation, winner-take-all, one byte per pixel |
| `denseDisparityBinaryScratchWords` | 3 | WordType units of scratch `denseDisparityBinary` needs |
| `denseDisparityBinary` | 3 | Dense disparity over an ALREADY-BINARY rectified pair: the cost is `popcount((L ^ shift(R, d)) over window)` -- one XOR per word of 64 pixels, no census, no wide image anywhere |

## `ops/derivative.hpp`

[`include/bincv/ops/derivative.hpp`](../include/bincv/ops/derivative.hpp)

| | tier | |
|---|---|---|
| `derivativeAdderStages` | 3 | Adder-class stages one destination word of the derivative costs |
| `derivativeReplicatedInputs` | 3 | Single-bit inputs the REJECTED replication route would need |
| `derivativeX` | 3 | Horizontal binarized derivative: `dst(x, y) = src(x+1, y) - src(x-1, y)`, as sign and magnitude |
| `derivativeY` | 3 | Vertical binarized derivative: `dst(x, y) = src(x, y+1) - src(x, y-1)`, as sign and magnitude |

## `ops/descriptor.hpp`

[`include/bincv/ops/descriptor.hpp`](../include/bincv/ops/descriptor.hpp)

| | tier | |
|---|---|---|
| `BriefPair` *(struct)* | 3 | One intensity comparison, as offsets from the keypoint |
| `BriefPattern` *(struct)* | 3 | `Bits` comparisons |
| `descriptorWords` | 3 | Words a `Bits`-bit descriptor occupies |
| `makeBriefPattern` | 3 | Fills a pattern by deterministic Gaussian sampling -- BRIEF's own construction |
| `computeBrief` | 3 | Computes descriptors for `count` keypoints |
| `kBriefAngleBins` *(constant)* | 3 | Rotation bins a steered pattern is built at: 12-degree steps, the ORB paper's own discretization |
| `briefAngleBin` | 3 | Which rotation bin an angle selects: the nearest 12-degree step, wrapped |
| `SteeredBriefPattern` *(struct)* | 3 | `Bits` comparisons at each of the 30 rotations: ~30 KB at 256 bits, built once and reused for every frame |
| `makeSteeredBriefPattern` | 3 | Builds the 30 rotated copies of `base` |
| `computeBriefSteered` | 3 | `computeBrief` steered by per-keypoint angles |
| `hammingDistance` | 3 | `popcount(a ^ b)` over `words` |
| `DescriptorMatch` *(struct)* | 3 | One query's best and second-best match |
| `matchDescriptors` | 3 | Brute-force nearest neighbour with Lowe's ratio test |
| `matchDescriptorsGated` | 3 | `matchDescriptors` restricted to candidates a pipeline's priors admit: a position window, and optionally an octave band |

## `ops/edge.hpp`

[`include/bincv/ops/edge.hpp`](../include/bincv/ops/edge.hpp)

| | tier | |
|---|---|---|
| `EdgeCombine` *(enum)* | 3 | How the two axes' results are combined |
| `EdgeRelation` *(enum)* | 3 | How a gradient is compared with the threshold |
| `EdgeSpatial` *(enum)* | 3 | Which pixels are differenced |
| `edgeThreshold` | 3 | Gradient-magnitude edge extraction straight into bits |

## `ops/essential.hpp`

[`include/bincv/ops/essential.hpp`](../include/bincv/ops/essential.hpp)

| | tier | |
|---|---|---|
| `EssentialMatrix` *(struct)* | 2 | A 3x3 essential matrix, row-major |
| `essentialSolverStackBytes` | 3 | Stack the five-point solver uses for one call, in bytes |
| `fivePointEssential` | 2 | Up to ten essential matrices through five correspondences |
| `EssentialModel` *(struct)* | 2 | The five-point model policy, for `bincv::ransac` |
| `refine` | — | No refit |
| `residual` | — | Sampson distance -- the first-order approximation of geometric reprojection error, which is what `cv::findEssentialMat`'s threshold is in |
| `findEssentialMat` | 2 | `cv::findEssentialMat(..., cv::RANSAC, ...)`'s role over caller-owned scratch |

## `ops/fast.hpp`

[`include/bincv/ops/fast.hpp`](../include/bincv/ops/fast.hpp)

| | tier | |
|---|---|---|
| `FastCorner` *(struct)* | 2 | One detected corner |
| `detectFast` | 2 | Detects FAST corners |

## `ops/logic.hpp`

[`include/bincv/ops/logic.hpp`](../include/bincv/ops/logic.hpp)

| | tier | |
|---|---|---|
| `bitwiseAnd` | 1 | dst = a & b, pixel for pixel |
| `bitwiseOr` | 1 | dst = a | b, pixel for pixel |
| `bitwiseXor` | 1 | dst = a ^ b, pixel for pixel |
| `bitwiseNot` | 1 | dst = ~src, pixel for pixel |

## `ops/medianWide.hpp`

[`include/bincv/ops/medianWide.hpp`](../include/bincv/ops/medianWide.hpp)

| | tier | |
|---|---|---|
| `MedianOffset` *(struct)* | 3 | One sample position, relative to the pixel being written |
| `MedianPattern` *(struct)* | 3 | A neighbourhood: `K` offsets, `K` odd so the median is a single element |
| `medianWide` | 3 | Median filter over a caller-chosen neighbourhood |

## `ops/morphology.hpp`

[`include/bincv/ops/morphology.hpp`](../include/bincv/ops/morphology.hpp)

| | tier | |
|---|---|---|
| `StructuringElement` *(struct)* | 1 | A morphological structuring element: a shape, an extent and an anchor |
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
| `rect3x3` | 1 | The 3x3 rectangle -- `cv::Mat()` passed to `cv::erode`, i.e |
| `cross3x3` | 1 | The 3x3 plus -- what BOTH `MORPH_CROSS` and `MORPH_ELLIPSE` give at 3x3 |
| `erode` | 1 | Morphological erosion: `dst(x,y) = AND over the element of src(x+dx, y+dy)` |
| `dilate` | 1 | Morphological dilation: `dst(x,y) = OR over the element of src(x+dx, y+dy)` |
| `morphologyExNeedsScratch` | 3 | True when `morphologyEx(op,...)` reads and writes its scratch view |
| `morphologyEx` | 1 | The seven `MorphOp` compositions |

## `ops/occupancy.hpp`

[`include/bincv/ops/occupancy.hpp`](../include/bincv/ops/occupancy.hpp)

| | tier | |
|---|---|---|
| `spaceCandidates` | 3 | Keeps the candidates that are at least `radius` from every live point and from every candidate already kept |
| `clearOccupancy` | 3 | Zeroes an occupancy mask |
| `markDisc` | 3 | Sets every pixel strictly within `radius` of `(cx, cy)` |
| `markOccupied` | 3 | Stamps `markDisc` for every point |
| `occupied` | 3 | Is the pixel `(x, y)` claimed? |
| `spaceCandidatesMasked` | 3 | `spaceCandidates` through an occupancy mask: test one bit, and stamp the disc of every candidate kept |

## `ops/opticalFlow.hpp`

[`include/bincv/ops/opticalFlow.hpp`](../include/bincv/ops/opticalFlow.hpp)

| | tier | |
|---|---|---|
| `LKEntryLevel` *(enum)* | 3 | Which pyramid level a keypoint ENTERS at |
| `LKParams` *(struct)* | 2 | The tracker's parameters, defaulted to the reference pipeline's verbatim |
| `LKLevel` *(struct)* | 2 | One pyramid level's six planes: both frames, and the previous frame's ternary derivative |
| `lkLevel` | 2 | Names a level's containers into an LKLevel |
| `LKLevelN` *(struct)* | 2 | One pyramid level at N bits per pixel: both frames' bit-planes, and the previous frame's N-bit signed derivative |
| `calcOpticalFlowPyrLK` | 2 | Pyramidal Lucas-Kanade tracking of sparse keypoints between two binary frames |
| `narrowLevel` | 3 | Reads a 64-bit pyramid level as a 32-bit one, so the vector kernels apply |
| `lkPathName` | 3 | Which residual kernel this level type will actually run, as a string |
| `LKLevels` *(struct)* | 2 | A tracking ladder whose levels have DIFFERENT bit depths, level 0 first |
| `stagingStackBytes` | 3 | Stack bytes the tracker's staging buffers occupy at `(N, WordType)` |

## `ops/orbPattern.hpp`

[`include/bincv/ops/orbPattern.hpp`](../include/bincv/ops/orbPattern.hpp)

| | tier | |
|---|---|---|
| `kOrbBriefPattern` *(constant)* | 3 | `cv::ORB`'s learned sampling table as a `BriefPattern<256>`: pair i is OpenCV's points (x1, y1) -> a and (x2, y2) -> b, in OpenCV's order |

## `ops/orientation.hpp`

[`include/bincv/ops/orientation.hpp`](../include/bincv/ops/orientation.hpp)

| | tier | |
|---|---|---|
| `keypointOrientation` | 3 | Orientation of `count` keypoints on a WIDE image, from the intensity centroid over a disc of `radius` |

## `ops/pack.hpp`

[`include/bincv/ops/pack.hpp`](../include/bincv/ops/pack.hpp)

| | tier | |
|---|---|---|
| `PackRule` *(enum)* | 3 | How a source pixel becomes a bit |
| `packRows` | 3 | Packs `rowCount` rows into `dst` starting at `dstRow` |
| `packBits` | 3 | Packs a pixel array to one bit per pixel |
| `QuantRule` *(enum)* | 3 | How a source pixel becomes an N-bit value |
| `packQuant` | 3 | Packs a pixel array to N bits per pixel, no OpenCV |
| `packQuantWith` | 3 | `packQuant` with an arbitrary per-pixel map |
| `packBitsIf` | 3 | `packBits` with an arbitrary per-pixel predicate |
| `unpackTo8Bit` | 3 | The reverse: one bit per pixel out to one byte per pixel |
| `writePbm` | 3 | Writes a binary image as a binary PBM (`P4`) to a caller-supplied buffer |
| `writePgm` | 3 | Writes a binary image as a binary PGM (`P5`) to a caller-supplied buffer |

## `ops/pyramid.hpp`

[`include/bincv/ops/pyramid.hpp`](../include/bincv/ops/pyramid.hpp)

| | tier | |
|---|---|---|
| `pyrDownWidth` | 3 | Destination width of one pyramid level: ceil(srcWidth / 2) |
| `pyrDownHeight` | 3 | Destination height of one pyramid level: ceil(srcHeight / 2) |
| `pyrLevelToBase` | 3 | Where a level-`level` pixel CENTER sits in level-0 coordinates, one axis |
| `pyrBaseToLevel` | 3 | The inverse: a level-0 coordinate in level-`level` pixels |
| `PyrDownFilter` *(enum)* | 3 | Which downsampling filter `pyrDownFiltered` applies |
| `PyrDownBorder` *(enum)* | 3 | What a filter reads outside the frame |
| `Pyramid` *(class)* | 2 | A pyramid: one QuantMat per level, each at its own bit depth |
| `levelBits` | — | Bits per pixel at level I |
| `level` | — | Level I, mutable |
| `build` | — | Fills levels 1..N-1 by running the chosen filter down the ladder |
| `sizeInWords` | — | Total words across every level -- the pyramid's whole footprint |
| `sizeInBytes` | — | Total bytes across every level |
| `pyrDownFiltered` | 3 | One pyramid level under a chosen downsampling filter |
| `pyrDownBox` | 2 | One pyramid level by 2x2 box mean, `BORDER_REPLICATE` |
| `pyrDown` | 1 | One pyramid level, EXACTLY as `cv::pyrDown` computes it: a 5x5 `[1,4,6,4,1]` Gaussian with `BORDER_REFLECT_101`, subsampled by 2 |

## `ops/ransac.hpp`

[`include/bincv/ops/ransac.hpp`](../include/bincv/ops/ransac.hpp)

| | tier | |
|---|---|---|
| `RansacParams` *(struct)* | 2 | The four parameters a RANSAC call takes |
| `RansacResult` *(struct)* | 2 | What a RANSAC call reports back |
| `RansacScratch` *(struct)* | 3 | Caller-owned scratch: one flag per correspondence, twice |
| `ransacScratchWords` | 3 | Words in one inlier set over `correspondences` points |
| `ransacScratchBytes` | 3 | Bytes of scratch a call over `correspondences` points needs |
| `ransac` | 2 | Fit `Model` to the correspondences by random sample consensus |
| `Affine2D` *(struct)* | 2 | A 2D affine transform, row-major: `[m[0] m[1] m[2]; m[3] m[4] m[5]]` |
| `Affine2DModel` *(struct)* | 2 | The 3-point affine model policy |
| `estimate` | — | Solve the affine through the three sampled correspondences |
| `refine` | — | Least-squares refit over the consensus set, which is what makes this agree with `cv::estimateAffine2D` on noisy data rather than merely on clean data |
| `residual` | — | Euclidean reprojection error, in pixels -- the units `cv::estimateAffine2D`'s `ransacReprojThreshold` is in |
| `estimateAffine2D` | 2 | `cv::estimateAffine2D(..., RANSAC, ...)`'s role over caller-owned scratch |

## `ops/reduce.hpp`

[`include/bincv/ops/reduce.hpp`](../include/bincv/ops/reduce.hpp)

| | tier | |
|---|---|---|
| `SplitCount` *(struct)* | 3 | The two halves of a split count: pixels where the selector `c` was clear, and pixels where it was set |
| `crossTerm` | — | The LK cross term: `whenClear - whenSet`, signed |
| `CovarianceCount` *(struct)* | 3 | The four numbers of a 2x2 gradient covariance over one region: popcount(a), popcount(b), and the split of `a & b` by the selector |
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

[`include/bincv/ops/resample.hpp`](../include/bincv/ops/resample.hpp)

| | tier | |
|---|---|---|
| `decimatedWidth` | 3 | Destination width for a horizontal decimation by two |
| `rowsDecimatedBy2` | 3 | The FREE half of a 2x2 subsample: every other row, as a view |
| `decimateColumnsBy2` | 3 | Horizontal decimation by two: `dst(y, j) = src(y, 2j)` |

## `ops/shift.hpp`

[`include/bincv/ops/shift.hpp`](../include/bincv/ops/shift.hpp)

| | tier | |
|---|---|---|
| `shift` | 3 | dst[y][x] = src[y + dy][x + dx], extrapolating outside the image |
| `shiftLeft` | 3 | dst[y][x] = src[y][x + k] -- moves the image LEFT by k columns |
| `shiftRight` | 3 | dst[y][x] = src[y][x - k] -- moves the image RIGHT by k columns |
| `shiftUp` | 3 | dst[y][x] = src[y + k][x] -- moves the image UP by k rows |
| `shiftDown` | 3 | dst[y][x] = src[y - k][x] -- moves the image DOWN by k rows |

## `ops/stereo.hpp`

[`include/bincv/ops/stereo.hpp`](../include/bincv/ops/stereo.hpp)

| | tier | |
|---|---|---|
| `StereoMatchParams` *(struct)* | 3 | Search and window parameters for the sparse rectified stereo matcher |
| `StereoMatch` *(struct)* | 3 | One left keypoint's stereo result |
| `stereoDescriptorMatch` | 3 | COARSE stage: each left descriptor against the right keypoints in its row band and disparity range |
| `stereoRefineDisparity` | 3 | FINE stage: slide a window along the epipolar row around each valid match's disparity, score by Hamming distance on the packed frames, and refine to sub-pixel |
| `stereoMatchRectified` | 3 | Both stages: descriptor search, then window refinement |

## `ops/subpix.hpp`

[`include/bincv/ops/subpix.hpp`](../include/bincv/ops/subpix.hpp)

| | tier | |
|---|---|---|
| `SubPixParams` *(struct)* | 2 | `cv::cornerSubPix`'s `winSize`, `zeroZone` and `criteria`, in one struct |
| `SubPixResult` *(struct)* | 3 | What `cornerSubPix` did, per corner |
| `cornerSubPix` | 2 | Refines corner positions to sub-pixel accuracy |

## `ops/threshold.hpp`

[`include/bincv/ops/threshold.hpp`](../include/bincv/ops/threshold.hpp)

| | tier | |
|---|---|---|
| `binarize` | 3 | dst = (src > thresh), pixel for pixel, over an N-plane bit-sliced source |
| `threshold` | 1 | dst = (src > thresh), packing a CV_8U image into one bit per pixel |

## `io/pnm.hpp`

[`include/bincv/io/pnm.hpp`](../include/bincv/io/pnm.hpp)

| | tier | |
|---|---|---|
| `PgmHeader` *(struct)* | 3 | What a `readPgm` call found, or why it did not |
| `PbmHeader` *(struct)* | 3 | What a `readPbm` call found, or why it did not |
| `readPgmHeader` | 3 | Parses a binary PGM (`P5`) header |
| `readPgmHeaderFromPrefix` | 3 | The same parse, from a prefix of the file |
| `readPbmHeader` | 3 | Parses a binary PBM (`P4`) header |
| `readPgm` | 3 | Reads a binary PGM straight into bits, under a `PackRule` |
| `readPbm` | 3 | Reads a binary PBM (`P4`) into a bit matrix |

## `io/sequence.hpp`

[`include/bincv/io/sequence.hpp`](../include/bincv/io/sequence.hpp)

| | tier | |
|---|---|---|
| `kSequenceMode8Bit` *(constant)* | — | `mode` value for 8-bit (`P5`-shaped) frame bodies |
| `kSequenceModePacked` *(constant)* | — | `mode` value for packed 1-bit (`P4`-shaped) frame bodies |
| `kSequenceHeaderBytes` *(constant)* | — | The fixed header size; frame 0's body starts here |
| `SequenceHeader` *(struct)* | 3 | What a `readSequenceHeader` call found, or why it did not |
| `SequenceFrameRange` *(struct)* | 3 | One frame's body within a blob, or `valid == false` |
| `readSequenceHeader` | 3 | Parses a `BSQ1` header from the first 32 bytes |
| `sequenceFrame` | 3 | Frame `i`'s body bytes, bounds-checked |
| `readSequenceFrameBody` | 3 | Unpacks ONE mode-1 frame body into a bit matrix |
| `readSequenceFrame` | 3 | Reads mode-1 frame `i` of a whole blob into a bit matrix |

## `core/parallel.hpp`

[`include/bincv/core/parallel.hpp`](../include/bincv/core/parallel.hpp)

| | tier | |
|---|---|---|
| `setParallelForBackend` | — | Installs a parallel-for backend |
| `setNumThreads` | — | How many threads binCV may use |
| `getNumThreads` | — | The current thread count |
| `parallelFor` | — | Runs `body(i, ctx)` for `i` in `[0, n)` |

## `core/simd.hpp`

[`include/bincv/core/simd.hpp`](../include/bincv/core/simd.hpp)

| | tier | |
|---|---|---|
| `SimdStatus` *(struct)* | 3 | What this translation unit compiled, and what the CPU under it supports |
| `simdStatus` | 3 | What vector paths are actually active |
| `simdStatusString` | 3 | One line naming every fast path and whether it is on |

## `core/storage.hpp`

[`include/bincv/core/storage.hpp`](../include/bincv/core/storage.hpp)

| | tier | |
|---|---|---|
| `Storage` *(class)* | 3 | Backing memory for a bit-packed matrix: {pointer, word count, ownership} |
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

[`include/bincv/core/types.hpp`](../include/bincv/core/types.hpp)

| | tier | |
|---|---|---|
| `Size` *(struct)* | 1 | A width and a height, in pixels |
| `area` | — | Calculate the area (width * height) |
| `empty` | — | Check if the size is empty (zero width or height) |
| `Rect` *(struct)* | 1 | An axis-aligned rectangle in PIXELS: origin (x, y), extent (width, height) |
| `QuantMat` *(class)* | — | Forward declaration of the QuantMat template -- the N-bit container |
| `Point2f` *(struct)* | 1 | A point with sub-pixel coordinates -- the tracker's and the refiner's |

## `core/view.hpp`

[`include/bincv/core/view.hpp`](../include/bincv/core/view.hpp)

| | tier | |
|---|---|---|
| `BinMatView` *(struct)* | 3 | Non-owning, mutable view of a bit-packed matrix: {ptr, size, stride} |
| `empty` | — | True if the view addresses no pixels |
| `row` | — | First word of row y |
| `BinMatConstView` *(struct)* | 3 | Non-owning, read-only view of a bit-packed matrix |
| `narrowPlane` | 3 | Reads a 64-bit bit-plane as a 32-bit one |
| `narrowPlaneMutable` | 3 | The same reinterpretation for a WRITABLE plane |

## `threads/pool.hpp`

[`include/bincv/threads/pool.hpp`](../include/bincv/threads/pool.hpp)

| | tier | |
|---|---|---|
| `ThreadPool` *(class)* | 3 | A minimal fixed-size pool that serves `bincv::parallelFor` |
| `install` | — | Makes this pool binCV's backend and sets the thread count to match |

