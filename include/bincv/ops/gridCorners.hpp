#pragma once

/// @file gridCorners.hpp
/// @brief Corners chosen cell by cell over a grid. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHAT THIS IS, AND WHY IT IS NOT goodFeaturesToTrack
///
/// `goodFeaturesToTrack` ranks every corner in the frame and walks them strongest
/// first, keeping one when no kept corner is within `minDistance`. That walk is global:
/// it needs every candidate ranked before it can start, so it holds a pool of them --
/// and on a frame whose walk outruns the pool it sweeps the frame again. Frontends that
/// spread their features by construction instead divide the frame into cells and take
/// the best corner or corners of each, skipping the cells that already hold a track.
/// That is a different answer to a different question, so it is a separate operation
/// with its own name rather than a mode of the other.
///
/// It is also a cheap one to compute. A cell's best `perCell` corners are known the
/// moment the sweep has passed the cell, so ONE sweep decides them, and the memory is
/// `perCell` eight-byte slots a cell: 2,880 bytes for one corner per 32-pixel cell at
/// 752x480, where `goodFeaturesToTrackStreaming`'s pool is whatever the caller can give
/// it and the walk's depth is the image's.
///
/// ---------------------------------------------------------------------------
/// THE RESPONSE IS goodFeaturesToTrack's
///
/// The same minimum-eigenvalue response from the same ternary derivatives, the same
/// 3x3 suppression and the same running maximum -- `impl::sweepCornerResponses`, the
/// sweep every corner selection here shares -- and the same meanings for `qualityLevel`
/// (a fraction of the frame's strongest admitted response, kept strictly above) and
/// `minDistance` (the integer spacing bound `goodFeaturesToTrack` uses). Every choice a
/// grid detector makes is a parameter: the cell size, how many corners a cell keeps,
/// a relative threshold, an absolute one, a spacing across cell borders, a cap on the
/// total, which cells already have features, and a pixel mask.
///
/// ---------------------------------------------------------------------------
/// WHAT IS EXACT
///
/// Each cell's corners are exactly its `perCell` strongest local maxima above the
/// thresholds -- the threshold that depends on the frame's maximum is applied after
/// the sweep, and a candidate the running threshold drops during it is below the final
/// one too. The spacing filter, when on, then walks every kept corner strongest first,
/// as `goodFeaturesToTrack`'s does; a corner it drops is not replaced by its cell's
/// next one, because the cell kept only `perCell`. Nothing allocates.
///
/// With `skipFullCells`, a full cell is what a mask leaving it out would make it, and
/// the corners are exactly the ones the same call returns given that mask instead. The
/// on-demand form then clears the gradient more than two pixels from any open cell, so
/// a full cell's responses are computed only where an open neighbour's suppression
/// reads them.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "corner.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief What `gridCorners` keeps, and where. **API TIER 3.** Every field is the
/// caller's choice; the defaults are one corner per 32-pixel cell above 1% of the
/// frame's strongest response, unspaced and uncapped.
struct GridCornerParams {
    int cellWidth = 32;          ///< pixels; the last column of cells takes what is left
    int cellHeight = 32;         ///< pixels; the last row of cells takes what is left
    int perCell = 1;             ///< corners a cell keeps, strongest first; at least 1
    double qualityLevel = 0.01;  ///< kept above this fraction of the frame's maximum; 0: none
    float minResponse = 0.0f;    ///< kept above this response; 0: none (a corner is above 0)
    double minDistance = 0.0;    ///< below 1: no spacing; else as `goodFeaturesToTrack`'s
    int maxCorners = 0;          ///< at most this many in all, strongest first; <= 0: no cap
    int blockSize = 3;           ///< the covariance window, as `goodFeaturesToTrack`'s
    /// A cell `existing` says is full is left out of the search altogether: none of its
    /// pixels can be a corner or counts toward the frame's strongest response, exactly
    /// as if the mask left the cell out -- so `qualityLevel` is then a fraction of the
    /// strongest response among the cells that can still take a corner. The form that
    /// derives its gradient also computes a full cell's responses only where a
    /// neighbouring cell's suppression reads them. Off, a full cell only takes no new
    /// corner. No effect without `existing`.
    bool skipFullCells = false;
};

/// @brief Cells across a frame `width` pixels wide. **API TIER 3.**
constexpr size_t gridCellsAcross(size_t width, const GridCornerParams& params) {
    const size_t c = params.cellWidth > 0 ? static_cast<size_t>(params.cellWidth) : 1;
    return (width + c - 1) / c;
}

/// @brief Cells down a frame `height` pixels tall. **API TIER 3.**
constexpr size_t gridCellsDown(size_t height, const GridCornerParams& params) {
    const size_t c = params.cellHeight > 0 ? static_cast<size_t>(params.cellHeight) : 1;
    return (height + c - 1) / c;
}

/// @brief The slots `gridCorners` needs: `perCell` for every cell. **API TIER 3.**
/// Eight bytes each -- 2,880 bytes for one corner per 32-pixel cell at 752x480.
constexpr size_t gridCornerSlots(size_t width, size_t height, const GridCornerParams& params) {
    const size_t k = params.perCell > 0 ? static_cast<size_t>(params.perCell) : 1;
    return gridCellsAcross(width, params) * gridCellsDown(height, params) * k;
}

namespace impl {

/// @brief The pixels of the cells that can still take a corner, under an optional pixel
/// mask: the sweep's admit policy when `skipFullCells` is on. **INTERNAL.**
template <typename WordType>
struct AdmitOpenCells {
    BinMatConstView<WordType> mask;
    const uint8_t* existing;
    size_t perCell, cellWidth, cellHeight, across, width;

    bool open(size_t cx, size_t cy) const { return existing[cy * across + cx] < perCell; }

    bool operator()(int x, int y) const {
        if (!open(static_cast<size_t>(x) / cellWidth, static_cast<size_t>(y) / cellHeight))
            return false;
        return mask.empty() || AdmitMasked<WordType>{mask}(x, y);
    }

    /// Every maximal run `[x0, x1)` of admitted pixels in row `y`, to `fn`: runs of open
    /// cells, split where the mask is clear.
    template <typename Fn>
    void runs(size_t y, Fn&& fn) const {
        constexpr size_t kBits = bitsPerWord<WordType>();
        const size_t cy = y / cellHeight;
        size_t x = 0;
        while (x < width) {
            const size_t cx = x / cellWidth;
            const size_t cellEnd = (cx + 1) * cellWidth < width ? (cx + 1) * cellWidth : width;
            if (!open(cx, cy)) {
                x = cellEnd;
                continue;
            }
            size_t end = cellEnd;
            while (end < width && open(end / cellWidth, cy))
                end = end + cellWidth < width ? end + cellWidth : width;
            if (mask.empty()) {
                fn(x, end);
            } else {
                const WordType* row = mask.row(y);
                size_t start = end;
                for (size_t i = x; i < end; ++i) {
                    const bool set = ((row[i / kBits] >> (i % kBits)) & WordType{1}) != 0;
                    if (set && start == end) start = i;
                    if (!set && start != end) {
                        fn(start, i);
                        start = end;
                    }
                }
                if (start != end) fn(start, end);
            }
            x = end;
        }
    }

    /// Which pixels of word `w` lie within two pixels of an open cell in rows `cr0`..`cr1`
    /// of cells: the gradient a sweep over open cells can read, through a suppression
    /// one pixel out and a 3x3 window one pixel beyond that.
    WordType keep(size_t cr0, size_t cr1, size_t w) const {
        constexpr size_t kBits = bitsPerWord<WordType>();
        constexpr WordType kAll = static_cast<WordType>(~WordType{0});
        const long long base = static_cast<long long>(w * kBits);
        const long long lo = base - 2;
        const size_t hi = w * kBits + kBits + 1 < width ? w * kBits + kBits + 1 : width - 1;
        const size_t c0 = lo < 0 ? 0 : static_cast<size_t>(lo) / cellWidth;
        const size_t c1 = hi / cellWidth;
        WordType out = 0;
        for (size_t cy = cr0; cy <= cr1; ++cy) {
            for (size_t cx = c0; cx <= c1; ++cx) {
                if (!open(cx, cy)) continue;
                long long a = static_cast<long long>(cx * cellWidth) - 2 - base;
                long long b = static_cast<long long>((cx + 1) * cellWidth) + 1 - base;
                if (a < 0) a = 0;
                if (b > static_cast<long long>(kBits) - 1) b = static_cast<long long>(kBits) - 1;
                if (a > b) continue;
                const WordType upTo =
                    static_cast<WordType>(kAll >> (kBits - 1 - static_cast<size_t>(b)));
                out = static_cast<WordType>(out | (upTo & static_cast<WordType>(
                                                              kAll << static_cast<size_t>(a))));
            }
        }
        return out;
    }
};

/// @brief The selection, over any response sweep. **INTERNAL.** `sweep(onCandidate)`
/// runs `sweepCornerResponses` with the caller's row source and returns the frame's
/// maximum admitted response, or -1 when a mask admitted nothing.
template <typename Sweep>
inline CornerResult selectPerCell(size_t width, size_t height, const GridCornerParams& params,
                                  Sweep&& sweep, CornerCandidate* slots, size_t slotCount,
                                  Corner* corners, size_t capacity, const uint8_t* existing) {
    CornerResult out;
    if (width == 0 || height == 0) return out;
    const size_t across = gridCellsAcross(width, params);
    const size_t cells = across * gridCellsDown(height, params);
    const size_t perCell = static_cast<size_t>(params.perCell);
    const size_t cw = static_cast<size_t>(params.cellWidth);
    const size_t ch = static_cast<size_t>(params.cellHeight);
    BINCV_ASSERT(slotCount >= cells * perCell,
                 "gridCorners: the slots must number gridCornerSlots(width, height, params)");
    static_cast<void>(slotCount);

    // An empty slot holds a negative response; a corner's is never negative.
    for (size_t i = 0; i < cells * perCell; ++i) slots[i] = CornerCandidate{0, 0, -1.0f};

    const impl::CandidateStronger stronger;
    const float minResponse = params.minResponse;
    const float frameMax = sweep([&](int x, int y, float val) {
        if (!(val > minResponse)) return;
        const size_t cell = (static_cast<size_t>(y) / ch) * across + static_cast<size_t>(x) / cw;
        const size_t have = existing != nullptr ? existing[cell] : 0;
        if (have >= perCell) return;
        // The cell's slots, strongest first; it keeps `perCell - have` of them.
        CornerCandidate* s = slots + cell * perCell;
        const size_t quota = perCell - have;
        const CornerCandidate c{static_cast<uint16_t>(x), static_cast<uint16_t>(y), val};
        if (!(s[quota - 1].response < 0.0f) && !stronger(c, s[quota - 1])) return;
        size_t at = quota - 1;
        while (at > 0 && (s[at - 1].response < 0.0f || stronger(c, s[at - 1]))) {
            s[at] = s[at - 1];
            --at;
        }
        s[at] = c;
    });
    if (!(frameMax > 0.0f)) return out;

    // The threshold that needed the frame's maximum, as goodFeaturesToTrack forms it.
    const float threshold = static_cast<float>(static_cast<double>(frameMax) * params.qualityLevel);
    size_t ranked = 0;
    for (size_t i = 0; i < cells * perCell; ++i) {
        if (slots[i].response >= 0.0f && slots[i].response > threshold) slots[ranked++] = slots[i];
    }
    out.candidatesRanked = ranked;
    std::sort(slots, slots + ranked, stronger);

    size_t limit = capacity;
    if (params.maxCorners > 0 && static_cast<size_t>(params.maxCorners) < limit) {
        limit = static_cast<size_t>(params.maxCorners);
    }
    const bool spaced = params.minDistance >= 1.0;
    if (!spaced) {
        const size_t n = ranked < limit ? ranked : limit;
        for (size_t i = 0; i < n; ++i) {
            corners[i].x = static_cast<int>(slots[i].x);
            corners[i].y = static_cast<int>(slots[i].y);
            corners[i].response = slots[i].response;
        }
        out.count = n;
        out.candidatesTruncated = ranked > limit;
        return out;
    }

    // The spacing walk, strongest first, with goodFeaturesToTrack's integer bounds: a
    // corner is dropped when a kept one lies at dx^2 + dy^2 < ceil(minDistance^2). The
    // kept set is held in row order, so the corners that could conflict are one run.
    const double sq = params.minDistance * params.minDistance;
    constexpr double kCap = 4.0e18;
    const long long limitSq =
        (sq >= kCap) ? static_cast<long long>(kCap) : static_cast<long long>(std::ceil(sq));
    long long band = static_cast<long long>(std::sqrt(static_cast<double>(limitSq)));
    while (band > 0 && band * band >= limitSq) --band;
    while ((band + 1) * (band + 1) < limitSq) ++band;
    size_t kept = 0;
    for (size_t i = 0; i < ranked; ++i) {
        const long long cy = static_cast<long long>(slots[i].y);
        const long long cx = static_cast<long long>(slots[i].x);
        size_t lo = 0, hi = kept;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (static_cast<long long>(corners[mid].y) < cy - band) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        bool good = true;
        for (size_t j = lo; j < kept; ++j) {
            const long long dy = static_cast<long long>(corners[j].y) - cy;
            if (dy > band) break;
            const long long dx = static_cast<long long>(corners[j].x) - cx;
            if (dx > band || dx < -band) continue;
            if (dx * dx + dy * dy < limitSq) {
                good = false;
                break;
            }
        }
        if (!good) continue;
        if (kept == limit) {   // a corner the walk keeps, and no room for it
            out.candidatesTruncated = true;
            break;
        }
        size_t ip = lo, ihi = kept;
        while (ip < ihi) {
            const size_t mid = ip + (ihi - ip) / 2;
            if (static_cast<long long>(corners[mid].y) <= cy) {
                ip = mid + 1;
            } else {
                ihi = mid;
            }
        }
        if (ip < kept) std::memmove(corners + ip + 1, corners + ip, (kept - ip) * sizeof(Corner));
        corners[ip].x = static_cast<int>(slots[i].x);
        corners[ip].y = static_cast<int>(slots[i].y);
        corners[ip].response = slots[i].response;
        ++kept;
    }
    std::sort(corners, corners + kept, impl::CornerStronger());
    out.count = kept;
    return out;
}

/// @brief The argument checks every `gridCorners` form makes. **INTERNAL.**
inline void checkGridCornerArgs(size_t width, size_t height, const GridCornerParams& params,
                                const CornerCandidate* slots, const Corner* corners,
                                size_t capacity) {
    BINCV_ASSERT(params.cellWidth > 0 && params.cellHeight > 0,
                 "gridCorners: cells must be at least one pixel on a side");
    BINCV_ASSERT(params.perCell > 0, "gridCorners: a cell must keep at least one corner");
    BINCV_ASSERT(params.qualityLevel >= 0.0, "gridCorners: qualityLevel must not be negative");
    BINCV_ASSERT(params.minResponse >= 0.0f, "gridCorners: minResponse must not be negative");
    BINCV_ASSERT(params.minDistance >= 0.0, "gridCorners: minDistance must not be negative");
    BINCV_ASSERT(params.blockSize > 0, "gridCorners: blockSize must be positive");
    BINCV_ASSERT(width <= 65535 && height <= 65535,
                 "gridCorners: a slot holds a position in 16 bits");
    BINCV_ASSERT(slots != nullptr, "gridCorners: the slots are required");
    BINCV_ASSERT(corners != nullptr || capacity == 0,
                 "gridCorners: a non-zero capacity needs a non-null corner array");
    static_cast<void>(width);
    static_cast<void>(height);
    static_cast<void>(params);
    static_cast<void>(slots);
    static_cast<void>(corners);
    static_cast<void>(capacity);
}

/// @brief `goodFeaturesToTrack`'s parameters for the shared sweep: the relative
/// threshold, which only prunes, and the window. **INTERNAL.**
inline GoodFeaturesParams gridSweepParams(const GridCornerParams& params) {
    GoodFeaturesParams g;
    g.qualityLevel = params.qualityLevel;
    g.blockSize = params.blockSize;
    return g;
}

} // namespace impl

/// @brief The corners of each cell of a grid, over stored derivative planes.
/// **API TIER 3.**
/// @param magX, magY, signX, signY The level's ternary derivative planes, as
/// `goodFeaturesToTrackStreaming` takes them.
/// @param params The grid and what each cell keeps; see `GridCornerParams`.
/// @param ring Caller scratch: three response rows of the frame's width.
/// @param slots Caller scratch of at least `gridCornerSlots(width, height, params)`.
/// @param corners Where the corners go, strongest first; `capacity` entries.
/// @param existing Optional: one byte per cell, cells in row order -- how many features
/// the caller already has there. A cell keeps at most `perCell - existing` new ones, so
/// a full cell takes none -- and with `params.skipFullCells` is not searched at all.
/// Null: every cell is empty.
/// @param mask Optional pixel mask, as `goodFeaturesToTrack`'s: only set pixels can be
/// corners, and the relative threshold is a fraction of their strongest response.
/// @note `count` is how many corners were written; `candidatesRanked` how many cells'
/// corners passed the thresholds; `candidatesTruncated` that `capacity` or `maxCorners`
/// cut the answer short. One sweep, always.
/// @note Never throws; allocates nothing.
template <typename WordType>
inline CornerResult gridCorners(BinMatConstView<WordType> magX, BinMatConstView<WordType> magY,
                                BinMatConstView<WordType> signX,
                                BinMatConstView<WordType> signY,
                                const GridCornerParams& params, ResponseMap ring,
                                CornerCandidate* slots, size_t slotCount, Corner* corners,
                                size_t capacity, const uint8_t* existing = nullptr,
                                BinMatConstView<WordType> mask = {}) {
    BINCV_ASSERT(magX.width == magY.width && magX.height == magY.height &&
                     magX.width == signX.width && magX.height == signX.height &&
                     magX.width == signY.width && magX.height == signY.height,
                 "gridCorners: the four derivative planes must have the same dimensions");
    impl::checkGridCornerArgs(magX.width, magX.height, params, slots, corners, capacity);
    const GoodFeaturesParams sweepParams = impl::gridSweepParams(params);
    if (params.skipFullCells && existing != nullptr) {
        const impl::AdmitOpenCells<WordType> open{
            mask,
            existing,
            static_cast<size_t>(params.perCell),
            static_cast<size_t>(params.cellWidth),
            static_cast<size_t>(params.cellHeight),
            gridCellsAcross(magX.width, params),
            magX.width};
        return impl::selectPerCell(
            magX.width, magX.height, params,
            [&](auto&& onCandidate) {
                return impl::sweepCornerCandidates<WordType>(magX, magY, signX, signY,
                                                             sweepParams, ring, open, onCandidate);
            },
            slots, slotCount, corners, capacity, existing);
    }
    return impl::selectPerCell(
        magX.width, magX.height, params,
        [&](auto&& onCandidate) {
            return impl::sweepCornerCandidates<WordType>(magX, magY, signX, signY, sweepParams,
                                                         ring, mask, onCandidate);
        },
        slots, slotCount, corners, capacity, existing);
}

/// @brief The container spelling of `gridCorners`. **API TIER 3.**
template <typename WordType>
inline CornerResult gridCorners(const TernaryMat<WordType>& dx, const TernaryMat<WordType>& dy,
                                const GridCornerParams& params, ResponseMap ring,
                                CornerCandidate* slots, size_t slotCount, Corner* corners,
                                size_t capacity, const uint8_t* existing = nullptr,
                                BinMatConstView<WordType> mask = {}) {
    return gridCorners<WordType>(dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(),
                                 dy.constSign(), params, ring, slots, slotCount, corners,
                                 capacity, existing, mask);
}

/// @brief `gridCorners` deriving the gradient as it sweeps instead of reading stored
/// derivative planes. **API TIER 3** -- the corners the stored form returns from
/// `derivativeX`/`derivativeY` of the same image with the same border, bit for bit.
/// @param src The image, and the derivative's border.
/// @param gradientRows Caller scratch of at least `gradientRowWords<N, WordType>(width)`
/// words, as `goodFeaturesToTrackStreaming`'s on-demand form takes.
/// @note blockSize 3 only, the bit-sliced response's. Any other returns no corners (and
/// asserts in a checked build).
template <size_t N, typename WordType>
inline CornerResult gridCorners(const GradientSource<N, WordType>& src,
                                const GridCornerParams& params, ResponseMap ring,
                                WordType* gradientRows, size_t gradientRowWordsAvailable,
                                CornerCandidate* slots, size_t slotCount, Corner* corners,
                                size_t capacity, const uint8_t* existing = nullptr,
                                BinMatConstView<WordType> mask = {}) {
    const size_t width = src.planes[0].width;
    const size_t height = src.planes[0].height;
    impl::checkGridCornerArgs(width, height, params, slots, corners, capacity);
    BINCV_ASSERT(params.blockSize == 3,
                 "gridCorners: the on-demand gradient serves blockSize 3, the bit-sliced response");
    BINCV_ASSERT((gradientRows != nullptr &&
                  gradientRowWordsAvailable >= gradientRowWords<N, WordType>(width)),
                 "gridCorners: the gradient scratch must hold gradientRowWords(width) words");
    static_cast<void>(gradientRowWordsAvailable);
    if (params.blockSize != 3) {
        CornerResult none;
        none.passes = 0;
        return none;
    }

    // The derivative rows the 3x3 window reads, made as the sweep reaches them: the
    // bookkeeping goodFeaturesToTrackStreaming's on-demand form does, over the same
    // row arithmetic (impl::gradientRow, impl::cornerMinEigenValRowSlicedRows).
    const size_t words = impl::minRowWords<WordType>(width);
    const ptrdiff_t leftSrc = impl::borderIndex(static_cast<ptrdiff_t>(-1), width, src.border);
    const ptrdiff_t rightSrc =
        impl::borderIndex(static_cast<ptrdiff_t>(width), width, src.border);
    const auto rowOf = [&](size_t slot, size_t axis, size_t plane) {
        return gradientRows + ((slot * 2 + axis) * (N + 1) + plane) * words;
    };
    size_t derived = 0;
    const bool skip = params.skipFullCells && existing != nullptr;
    const impl::AdmitOpenCells<WordType> open{mask,
                                              existing,
                                              static_cast<size_t>(params.perCell),
                                              static_cast<size_t>(params.cellWidth),
                                              static_cast<size_t>(params.cellHeight),
                                              gridCellsAcross(width, params),
                                              width};
    const auto derive = [&](size_t r) {
        const size_t slot = r % kResponseRingRows;
        const WordType* srcRow[N];
        const WordType* rowA[N];
        const WordType* rowB[N];
        WordType* magX[N];
        WordType* magY[N];
        const ptrdiff_t ya = impl::borderIndex(static_cast<ptrdiff_t>(r) + 1, height, src.border);
        const ptrdiff_t yb = impl::borderIndex(static_cast<ptrdiff_t>(r) - 1, height, src.border);
        for (size_t p = 0; p < N; ++p) {
            srcRow[p] = src.planes[p].row(r);
            rowA[p] = ya >= 0 ? src.planes[p].row(static_cast<size_t>(ya)) : nullptr;
            rowB[p] = yb >= 0 ? src.planes[p].row(static_cast<size_t>(yb)) : nullptr;
            magX[p] = rowOf(slot, 0, p);
            magY[p] = rowOf(slot, 1, p);
        }
        impl::gradientRow<N, WordType>(srcRow, rowA, rowB, width, leftSrc, rightSrc,
                                       src.borderValue, magX, rowOf(slot, 0, N), magY,
                                       rowOf(slot, 1, N));
        if (!skip) return;
        // The gradient no sweep over open cells reads, cleared: a group of eight with
        // none left is then skipped as empty, and every response the sweep reads -- an
        // admitted pixel's, and its suppression's neighbours' -- is the unmasked one.
        const size_t cr0 = (r >= 2 ? r - 2 : 0) / static_cast<size_t>(params.cellHeight);
        const size_t cr1 = (r + 2 < height ? r + 2 : height - 1) /
                           static_cast<size_t>(params.cellHeight);
        for (size_t w = 0; w < words; ++w) {
            const WordType k = open.keep(cr0, cr1, w);
            magX[0][w] = static_cast<WordType>(magX[0][w] & k);
            magY[0][w] = static_cast<WordType>(magY[0][w] & k);
        }
    };
#if BINCV_CORNER_GROUP_SKIP
    const auto rowAt = [&](int yi, float* dst, const auto&... groups) {
#else
    const auto rowAt = [&](int yi, float* dst) {
#endif
        const size_t y = static_cast<size_t>(yi);
        const size_t need = y + 1 < height ? y + 1 : height - 1;
        while (derived <= need) derive(derived++);
        const WordType* mxr[3];
        const WordType* myr[3];
        const WordType* sxr[3];
        const WordType* syr[3];
        for (size_t k = 0; k < 3; ++k) {
            const bool ok = y + k >= 1 && y + k - 1 < height;
            const size_t slot = ok ? (y + k - 1) % kResponseRingRows : 0;
            mxr[k] = ok ? rowOf(slot, 0, 0) : nullptr;
            myr[k] = ok ? rowOf(slot, 1, 0) : nullptr;
            sxr[k] = ok ? rowOf(slot, 0, N) : nullptr;
            syr[k] = ok ? rowOf(slot, 1, N) : nullptr;
        }
#if BINCV_CORNER_GROUP_SKIP
        impl::cornerMinEigenValRowSlicedRows<WordType>(mxr, myr, sxr, syr, width,
                                                       y > 0 && y + 1 < height, dst, groups...);
#else
        impl::cornerMinEigenValRowSlicedRows<WordType>(mxr, myr, sxr, syr, width,
                                                       y > 0 && y + 1 < height, dst);
#endif
    };
    const GoodFeaturesParams sweepParams = impl::gridSweepParams(params);
    if (skip) {
        return impl::selectPerCell(
            width, height, params,
            [&](auto&& onCandidate) {
                return impl::sweepCornerResponses<WordType>(width, height, sweepParams, ring,
                                                            open, rowAt, onCandidate);
            },
            slots, slotCount, corners, capacity, existing);
    }
    return impl::selectPerCell(
        width, height, params,
        [&](auto&& onCandidate) {
            return impl::sweepCornerResponses<WordType>(width, height, sweepParams, ring, mask,
                                                        rowAt, onCandidate);
        },
        slots, slotCount, corners, capacity, existing);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
