#pragma once

/// @file opticalFlowOnDemand.hpp
/// @brief Pyramidal Lucas–Kanade without stored derivatives: each window's derivative
/// words are computed from the previous frame as the tracker stages the window.
/// **API TIER 2** -- `calcOpticalFlowPyrLK`'s, and the same points, bit for bit.
///
/// ---------------------------------------------------------------------------
/// THE TRADE
///
/// The stored form takes each level's derivative planes, `4N + 2` planes beside the two
/// frames' `2N`: at 752x480 on the shipped 1/2/2/2 ladder that is 275,040 bytes, more
/// than the two pyramids together (152,640). Nothing reads them but the window staging,
/// which copies each tracked window's rows onto the stack anyway. Here staging computes
/// those rows instead -- the derivative's own `signedDifference`, over the same words,
/// with the same border -- so the derivative planes are never held.
///
/// The cost is compute: a window's derivative words are made once per point and level,
/// where the stored form makes every pixel's once per frame, and at the coarse levels the
/// windows of different points overlap. Which costs less depends on how many points
/// track and how big the frame is, so it is the caller's to choose, by which level type
/// they build; the stored form is unchanged.
///
/// ---------------------------------------------------------------------------
/// THE LIMIT
///
/// **The window must fit one word wide and `impl::kStagedMaxRows` tall** -- 32x64 at
/// `uint32_t`, which holds the reference 31x31 window. Those are the windows the tracker
/// stages; a larger one reads the derivative planes directly, and there are none here.
/// A call with a larger window marks every point lost and returns (and asserts in a
/// checked build): kernels do not throw.
///
/// The derivative's border is the level's `border` and `borderValue`, defaulting to
/// what `derivativeX` and `derivativeY` default to, so the points are the stored form's
/// when its planes came from those calls with the same border.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
#include "../quantMat.hpp"
#include "derivative.hpp"
#include "opticalFlow.hpp"
#include "shift.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief One pyramid level at N bits per pixel for the on-demand tracker: both
/// frames' bit-planes and the border the derivative is taken with. **API TIER 2.**
/// @note Views, not containers; plane 0 is the least significant, as `LKLevelN`.
template <size_t N, typename WordType>
struct LKLevelOnDemand {
    static_assert(N >= 1 && N <= 7, "LKLevelOnDemand: N outside SignedQuantMat's range");
    static constexpr size_t Bits = N;
    using Word = WordType;

    BinMatConstView<WordType> prev[N];   ///< previous frame's planes, LSB first
    BinMatConstView<WordType> next[N];   ///< next frame's planes, LSB first
    BorderType border = BORDER_REFLECT_101;  ///< the derivative's, as `derivativeX` takes it
    bool borderValue = false;                ///< BORDER_CONSTANT's fill

    size_t width() const { return prev[0].width; }
    size_t height() const { return prev[0].height; }
};

/// @brief An on-demand level over two `QuantMat` frames. **API TIER 2.**
template <size_t N, typename WordType>
inline LKLevelOnDemand<N, WordType> lkLevelOnDemand(const QuantMat<N, WordType>& prev,
                                                    const QuantMat<N, WordType>& next,
                                                    BorderType border = BORDER_REFLECT_101,
                                                    bool borderValue = false) {
    LKLevelOnDemand<N, WordType> lv;
    for (size_t i = 0; i < N; ++i) {
        lv.prev[i] = prev.constPlane(i);
        lv.next[i] = next.constPlane(i);
    }
    lv.border = border;
    lv.borderValue = borderValue;
    return lv;
}

namespace impl {

/// @brief The tracker's view of an on-demand level. **INTERNAL.** An `LKLevelN` whose
/// derivative views are empty, so every routine that takes a level takes it, and a type
/// of its own, so `stageWindow` below is the overload that stages it. Nothing reads
/// the empty views: the entry points admit only windows that stage.
template <size_t N, typename WordType>
struct LKLevelOnDemandImpl : LKLevelN<N, WordType> {
    BorderType border = BORDER_REFLECT_101;
    bool borderValue = false;
};

template <size_t N, typename WordType>
inline LKLevelOnDemandImpl<N, WordType> onDemandImpl(const LKLevelOnDemand<N, WordType>& lv) {
    LKLevelOnDemandImpl<N, WordType> out;
    for (size_t i = 0; i < N; ++i) {
        out.prev[i] = lv.prev[i];
        out.next[i] = lv.next[i];
    }
    out.border = lv.border;
    out.borderValue = lv.borderValue;
    return out;
}

/// @brief `checkLevelPlanes` for an on-demand level: its frames' `2N` planes share the
/// level's dimensions, and it has no others. **INTERNAL.**
template <size_t N, typename WordType>
inline void checkLevelPlanes(const LKLevelOnDemandImpl<N, WordType>& lv) {
    const size_t w = lv.prev[0].width;
    const size_t h = lv.prev[0].height;
    for (size_t k = 0; k < N; ++k) {
        BINCV_ASSERT(lv.prev[k].width == w && lv.prev[k].height == h &&
                         lv.next[k].width == w && lv.next[k].height == h,
                     "opticalFlow: a level's planes must share its dimensions");
    }
    (void)w;
    (void)h;
}

/// @brief `stageWindow` for an on-demand level: the window's derivative words made
/// from the previous frame instead of read from planes. **INTERNAL.**
/// The same words the stored form stages, inside the window: `derivativeX`'s taps
/// `src(x+1)` and `src(x-1)`, `derivativeY`'s `src(y+1)` and `src(y-1)`, its border at
/// the frame's edges, and its `signedDifference` between them. A sign outside the
/// window's columns may differ from the stored plane's, where nothing reads it: every
/// consumer of a staged sign ANDs it with the staged magnitude, which is masked to the
/// window.
template <size_t N, typename WordType>
inline bool stageWindow(const LKLevelOnDemandImpl<N, WordType>& lv,
                        const RegionWords<WordType>& r, StagedWindow<N, WordType>& s) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    const size_t width = r.x1 - r.x0;
    if (width == 0 || width > kBits) return false;
    const size_t rows = r.y1 - r.y0;
    if (rows > kStagedMaxRows) return false;

    const size_t frameW = lv.prev[0].width;
    const size_t frameH = lv.prev[0].height;
    const size_t words = minRowWords<WordType>(frameW);
    const WordType mask = lowBitsMask<WordType>(width);
    const WordType fill = lv.borderValue ? static_cast<WordType>(~static_cast<WordType>(0))
                                         : static_cast<WordType>(0);

    // The two columns whose horizontal tap is outside the frame, when the window
    // reaches them: src(-1) for column 0, src(frameW) for column frameW - 1.
    const ptrdiff_t leftSrc = borderIndex(static_cast<ptrdiff_t>(-1), frameW, lv.border);
    const ptrdiff_t rightSrc =
        borderIndex(static_cast<ptrdiff_t>(frameW), frameW, lv.border);
    const bool atLeft = r.x0 == 0;
    const size_t lastCol = frameW - 1;
    const bool atRight = lastCol - r.x0 < kBits;
    const WordType rightBit =
        atRight ? static_cast<WordType>(static_cast<WordType>(1) << (lastCol - r.x0))
                : static_cast<WordType>(0);

    // Each source row's window word is extracted once: the window's own rows are
    // `self`, which the vertical derivative reads as its rows above and below, so only
    // the row past each end of the window is extracted again. The horizontal taps come
    // from the centre word: one place over, plus the single bit outside it.
    for (size_t y = r.y0, i = 0; y < r.y1; ++y, ++i) {
        for (size_t k = 0; k < N; ++k) s.self[i][k] = alignedWord<WordType>(lv.prev[k].row(y), words, r.x0);
    }
    const auto outside = [&](ptrdiff_t y, size_t k) -> WordType {
        return y >= 0 ? alignedWord<WordType>(lv.prev[k].row(static_cast<size_t>(y)), words, r.x0)
                      : fill;
    };
    // The one bit each horizontal tap needs from outside the word: column x0 - 1 for the
    // left tap of bit 0, and column x0 + kBits for the right tap of the top bit -- which
    // only a window the whole word wide reads; narrower, that bit is masked away.
    const bool fullWord = width == kBits && r.x0 + kBits < frameW;
    const size_t leftWord = atLeft ? 0 : (r.x0 - 1) / kBits;
    const WordType leftMask = atLeft ? static_cast<WordType>(0) : bitMask<WordType>(r.x0 - 1);
    const size_t topWord = fullWord ? (r.x0 + kBits) / kBits : 0;
    const WordType topMask = fullWord ? bitMask<WordType>(r.x0 + kBits) : static_cast<WordType>(0);
    constexpr WordType kTop = static_cast<WordType>(static_cast<WordType>(1) << (kBits - 1));

    for (size_t y = r.y0, i = 0; y < r.y1; ++y, ++i) {
        const ptrdiff_t ya = borderIndex(static_cast<ptrdiff_t>(y) + 1, frameH, lv.border);
        const ptrdiff_t yb = borderIndex(static_cast<ptrdiff_t>(y) - 1, frameH, lv.border);
        const bool belowInWindow = i + 1 < rows && ya == static_cast<ptrdiff_t>(y) + 1;
        const bool aboveInWindow = i > 0 && yb == static_cast<ptrdiff_t>(y) - 1;
        WordType ax[N], bx[N], ay[N], by[N];
        for (size_t k = 0; k < N; ++k) {
            const WordType* row = lv.prev[k].row(y);
            const WordType c = s.self[i][k];

            WordType a = static_cast<WordType>(c >> 1);
            if (fullWord && (row[topWord] & topMask) != 0) a = static_cast<WordType>(a | kTop);
            if (atRight) {
                const bool bit = rightSrc < 0
                                     ? lv.borderValue
                                     : rowBit<WordType>(row, static_cast<size_t>(rightSrc));
                a = static_cast<WordType>(static_cast<WordType>(a & static_cast<WordType>(~rightBit)) |
                                          (bit ? rightBit : static_cast<WordType>(0)));
            }
            bool leftBit;
            if (atLeft) {
                leftBit = leftSrc < 0 ? lv.borderValue
                                      : rowBit<WordType>(row, static_cast<size_t>(leftSrc));
            } else {
                leftBit = (row[leftWord] & leftMask) != 0;
            }
            ax[k] = a;
            bx[k] = static_cast<WordType>(static_cast<WordType>(c << 1) |
                                          (leftBit ? static_cast<WordType>(1) : static_cast<WordType>(0)));
            ay[k] = belowInWindow ? s.self[i + 1][k] : outside(ya, k);
            by[k] = aboveInWindow ? s.self[i - 1][k] : outside(yb, k);
        }
        WordType mx[N], my[N];
        WordType sx = static_cast<WordType>(0), sy = static_cast<WordType>(0);
        signedDifference<N, WordType, false>(ax, bx, mx, sx);
        signedDifference<N, WordType, false>(ay, by, my, sy);
        for (size_t k = 0; k < N; ++k) {
            s.magX[i][k] = static_cast<WordType>(mx[k] & mask);
            s.magY[i][k] = static_cast<WordType>(my[k] & mask);
        }
        s.signX[i] = sx;
        s.signY[i] = sy;
    }
    return true;
}

/// @brief Whether the on-demand tracker can take this window; marks every point lost
/// when it cannot. **INTERNAL.**
template <typename WordType>
inline bool onDemandWindowFits(const LKContext& c) {
    // lkPrologue has asserted both are above 2, so the casts are of positive values.
    const bool fits = static_cast<size_t>(c.winW) <= bitsPerWord<WordType>() &&
                      static_cast<size_t>(c.winH) <= kStagedMaxRows;
    BINCV_ASSERT(fits, "opticalFlow: the on-demand tracker stages every window, so the window "
                       "must fit one word wide and kStagedMaxRows tall");
    if (!fits) {
        for (size_t i = 0; i < c.pointCount; ++i) c.status[i] = 0;
    }
    return fits;
}

} // namespace impl

/// @brief Pyramidal Lucas–Kanade over on-demand levels of one depth, level 0 first.
/// **API TIER 2** -- the stored form's points, status and error, bit for bit.
/// @note The window must fit one word wide and `impl::kStagedMaxRows` tall; see the
/// file header.
template <size_t N, typename WordType>
inline void calcOpticalFlowPyrLK(const LKLevelOnDemand<N, WordType>* levels, size_t levelCount,
                                 const Point2f* prevPts, Point2f* nextPts, uint8_t* status,
                                 float* err, size_t pointCount,
                                 const LKParams& params = LKParams()) {
    impl::LKContext c;
    if (!impl::lkPrologue(levelCount, prevPts, nextPts, status, err, pointCount, params, c)) return;
    BINCV_ASSERT(levels != nullptr, "opticalFlow: levels must be non-null");
    if (!impl::onDemandWindowFits<WordType>(c)) return;
    c.usableLevels = impl::usableLevelCount(levelCount, c.winW, c.winH, [&](size_t i) {
        return impl::LevelDims{levels[i].width(), levels[i].height()};
    });
    if (c.usableLevels > impl::kMaxLevels) c.usableLevels = impl::kMaxLevels;
    for (size_t i = 0; i < c.usableLevels; ++i) {
        c.dims[i] = impl::LevelDims{levels[i].width(), levels[i].height()};
    }
    for (size_t li = c.usableLevels; li-- > 0;) {
        impl::trackOneLevel(impl::onDemandImpl(levels[li]), li, c);
    }
}

/// @brief A ladder of on-demand levels of different depths, level 0 first: `LKLevels`
/// without the derivative planes. **API TIER 2.**
template <typename WordType, size_t... LevelBits>
struct LKLevelsOnDemand;

template <typename WordType, size_t N0>
struct LKLevelsOnDemand<WordType, N0> {
    static constexpr size_t Levels = 1;
    LKLevelOnDemand<N0, WordType> level;

    template <size_t I>
    LKLevelOnDemand<N0, WordType>& get() {
        static_assert(I == 0, "LK ladder level index out of range");
        return level;
    }
    template <size_t I>
    const LKLevelOnDemand<N0, WordType>& get() const {
        static_assert(I == 0, "LK ladder level index out of range");
        return level;
    }
    impl::LevelDims dimsAt(size_t i) const {
        (void)i;
        return impl::LevelDims{level.width(), level.height()};
    }
    template <typename Fn>
    void visitCoarseToFine(size_t usable, Fn& f, size_t index) const {
        (void)usable;
        f(level, index);
    }
};

template <typename WordType, size_t N0, size_t N1, size_t... Rest>
struct LKLevelsOnDemand<WordType, N0, N1, Rest...> {
    static constexpr size_t Levels = 2 + sizeof...(Rest);
    LKLevelOnDemand<N0, WordType> level;
    LKLevelsOnDemand<WordType, N1, Rest...> rest;

    template <size_t I>
    auto& get() {
        if constexpr (I == 0) {
            return level;
        } else {
            return rest.template get<I - 1>();
        }
    }
    template <size_t I>
    const auto& get() const {
        if constexpr (I == 0) {
            return level;
        } else {
            return rest.template get<I - 1>();
        }
    }
    impl::LevelDims dimsAt(size_t i) const {
        return i == 0 ? impl::LevelDims{level.width(), level.height()} : rest.dimsAt(i - 1);
    }
    template <typename Fn>
    void visitCoarseToFine(size_t usable, Fn& f, size_t index) const {
        if (index + 1 < usable) rest.visitCoarseToFine(usable, f, index + 1);
        f(level, index);
    }
};

/// @brief Pyramidal Lucas–Kanade over a ladder of on-demand levels of mixed depths.
/// **API TIER 2** -- the stored form's points, status and error, bit for bit.
/// @note The window must fit one word wide and `impl::kStagedMaxRows` tall; see the
/// file header.
template <typename WordType, size_t... LevelBits>
inline void calcOpticalFlowPyrLK(const LKLevelsOnDemand<WordType, LevelBits...>& levels,
                                 const Point2f* prevPts, Point2f* nextPts, uint8_t* status,
                                 float* err, size_t pointCount,
                                 const LKParams& params = LKParams()) {
    impl::LKContext c;
    if (!impl::lkPrologue(sizeof...(LevelBits), prevPts, nextPts, status, err, pointCount, params,
                          c)) {
        return;
    }
    if (!impl::onDemandWindowFits<WordType>(c)) return;
    c.usableLevels = impl::usableLevelCount(sizeof...(LevelBits), c.winW, c.winH,
                                            [&](size_t i) { return levels.dimsAt(i); });
    if (c.usableLevels > impl::kMaxLevels) c.usableLevels = impl::kMaxLevels;
    for (size_t i = 0; i < c.usableLevels; ++i) c.dims[i] = levels.dimsAt(i);
    auto visit = [&](const auto& lv, size_t li) {
        impl::trackOneLevel(impl::onDemandImpl(lv), li, c);
    };
    levels.visitCoarseToFine(c.usableLevels, visit, 0);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
