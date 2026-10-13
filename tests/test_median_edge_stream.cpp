// The sensor stage over rows as they arrive (ops/medianEdgeStream.hpp), against the
// two whole-frame calls it replaces and against a definition of both written here
// pixel by pixel.
//
// WHY THE ORACLE IS WRITTEN HERE AND NOT BORROWED. The stream and the whole-frame
// calls now share their row bodies, so comparing one with the other would pass even
// if the shared body were wrong. Every path is therefore held to two naive
// per-pixel definitions -- the median with zero fill, the edge test with
// BORDER_REFLECT_101 -- that share nothing with the library but the arithmetic.
//
// WHAT IS SWEPT, AND WHY THOSE NUMBERS.
//  * widths around every word and vector boundary: the vector edge row takes its
//    first word from column 1 and words near the right edge from column w - 33,
//    and those anchors change at 32, 33, 34, 64, 65; the vector median steps 32 or
//    16 pixels and leaves a tail.
//  * heights 1 to 5: the reflected row above row 0 and below the last are the same
//    row at height 2, and the row itself at height 1.
//  * every way of cutting the frame into pushes -- one row at a time, two, three,
//    five, irregular, the whole frame at once -- because a stream that is right for
//    one chunking and wrong for another is the failure this shape invites.
//  * the vector arms ON and OFF, where they exist.
//  * rows pushed from the image itself, and from a staging buffer of another stride
//    that is overwritten after every push: the stream reads a push's rows in place,
//    so one that kept a pointer past the call would read the garbage.
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <vector>

#include "bincv/ops/edge.hpp"
#include "bincv/ops/medianEdgeStream.hpp"
#include "bincv/ops/medianWide.hpp"
#include "test_util.hpp"

namespace {

using namespace bincv;
using W = uint32_t;

template <typename T>
std::vector<T> randomImage(size_t w, size_t h, uint64_t seed, unsigned span) {
    std::vector<T> img(w * h);
    uint64_t s = seed;
    for (auto& v : img) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<T>((s >> 33) % span);
    }
    return img;
}

template <size_t K, typename T>
std::vector<T> naiveMedian(const std::vector<T>& in, size_t w, size_t h,
                           const MedianPattern<K>& pat) {
    std::vector<T> out(w * h);
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            std::vector<T> v;
            for (size_t k = 0; k < K; ++k) {
                const long long sy = static_cast<long long>(y) + pat.offset[k].dy;
                const long long sx = static_cast<long long>(x) + pat.offset[k].dx;
                const bool inside = sy >= 0 && sx >= 0 && sy < static_cast<long long>(h) &&
                                    sx < static_cast<long long>(w);
                v.push_back(inside ? in[static_cast<size_t>(sy) * w + static_cast<size_t>(sx)]
                                   : T{0});
            }
            std::sort(v.begin(), v.end());
            out[y * w + x] = v[K / 2];
        }
    }
    return out;
}

long long reflect101(long long i, long long n) {
    if (n <= 1) return 0;
    while (i < 0 || i > n - 1) {
        if (i < 0) i = -i;
        if (i > n - 1) i = 2 * (n - 1) - i;
    }
    return i;
}

/// The edge predicate, pixel by pixel, as its documentation states it.
template <EdgeCombine C, EdgeRelation R, EdgeSpatial S, typename T>
std::vector<uint8_t> naiveEdge(const std::vector<T>& in, size_t w, size_t h, long long t) {
    const long long n = static_cast<long long>(w), m = static_cast<long long>(h);
    const auto at = [&](long long y, long long x) {
        return static_cast<long long>(in[static_cast<size_t>(reflect101(y, m) * n +
                                                             reflect101(x, n))]);
    };
    const long long a = (S == EdgeSpatial::Backward) ? 0 : 1;
    const long long b = (S == EdgeSpatial::Forward) ? 0 : 1;
    std::vector<uint8_t> out(w * h);
    for (long long y = 0; y < m; ++y) {
        for (long long x = 0; x < n; ++x) {
            const long long dh = std::llabs(at(y, x + a) - at(y, x - b));
            const long long dv = std::llabs(at(y + a, x) - at(y - b, x));
            const bool ph = (R == EdgeRelation::Ge) ? dh >= t : dh > t;
            const bool pv = (R == EdgeRelation::Ge) ? dv >= t : dv > t;
            out[static_cast<size_t>(y * n + x)] = (C == EdgeCombine::Or) ? (ph || pv) : (ph && pv);
        }
    }
    return out;
}

/// Bits that differ from `want`, plus any padding bit that is set.
size_t bitsWrong(const std::vector<W>& bits, size_t stride, size_t w, size_t h,
                 const std::vector<uint8_t>& want) {
    size_t bad = 0;
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < stride * 32; ++x) {
            const bool got = (bits[y * stride + x / 32] >> (x % 32)) & 1u;
            const bool exp = x < w ? want[y * w + x] != 0 : false;
            if (got != exp) ++bad;
        }
    }
    return bad;
}

void setVectorArms(bool on) {
#if defined(BINCV_EDGE_AVX2) || defined(BINCV_EDGE_NEON) || defined(BINCV_EDGE_DSP)
    impl::edgeSimdEnabled() = on;
#endif
#if defined(BINCV_MEDIAN_AVX2) || defined(BINCV_MEDIAN_NEON) || defined(BINCV_MEDIAN_DSP)
    impl::medianSimdEnabled() = on;
#endif
    (void)on;
}

/// The irregular chunking: 1, 4, 2, 7, 3, ... rows per push.
std::vector<size_t> chunks(size_t h, size_t mode) {
    std::vector<size_t> c;
    static const size_t kIrregular[] = {1, 4, 2, 7, 3, 1, 5};
    size_t done = 0, i = 0;
    while (done < h) {
        size_t n = mode == 0 ? h : mode == 99 ? kIrregular[i++ % 7] : mode;
        if (n > h - done) n = h - done;
        c.push_back(n);
        done += n;
    }
    return c;
}

/// Everything for one image and one pattern: whole frame with the vector arms on and
/// off, then the stream in every chunking, each held to the naive oracle.
template <size_t K, typename T, EdgeCombine C = EdgeCombine::Or,
          EdgeRelation R = EdgeRelation::Ge, EdgeSpatial S = EdgeSpatial::Wide>
size_t sweepOne(size_t w, size_t h, const MedianPattern<K>& pat, T threshold, uint64_t seed,
                unsigned span) {
    const std::vector<T> img = randomImage<T>(w, h, seed, span);
    const std::vector<uint8_t> want =
        naiveEdge<C, R, S, T>(naiveMedian<K, T>(img, w, h, pat), w, h, threshold);
    const size_t stride = (w + 31) / 32;
    size_t wrong = 0;

    for (int arms = 1; arms >= 0; --arms) {
        setVectorArms(arms != 0);
        std::vector<T> med(w * h);
        std::vector<W> bits(stride * h, ~W{0});   // padding must be CLEARED, not left
        medianWide<K, T>(img.data(), w, h, w, med.data(), w, pat);
        edgeThreshold<C, R, S, T, W>(med.data(), w, h, w, BinMatView<W>(bits.data(), w, h, stride),
                                     threshold);
        wrong += bitsWrong(bits, stride, w, h, want);

        size_t which = 0;
        for (size_t mode : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, size_t{0}, size_t{99}}) {
            using Stream = MedianEdgeStream<K, T, W, C, R, S>;
            std::vector<T> scratch(Stream::scratchElements(w, pat));
            std::vector<W> sb(stride * h, ~W{0});
            Stream st(scratch.data(), scratch.size(), pat, BinMatView<W>(sb.data(), w, h, stride),
                      threshold);
            st.reset();
            const bool staged = (which++ % 2) == 1;
            const size_t sstride = w + 3;
            std::vector<T> staging(sstride * h);
            size_t row = 0, lastDone = 0;
            for (size_t n : chunks(h, mode)) {
                if (staged) {
                    for (size_t r = 0; r < n; ++r) {
                        std::copy(img.begin() + static_cast<std::ptrdiff_t>((row + r) * w),
                                  img.begin() + static_cast<std::ptrdiff_t>((row + r + 1) * w),
                                  staging.begin() + static_cast<std::ptrdiff_t>(r * sstride));
                    }
                    st.push(staging.data(), n, sstride);
                    std::fill(staging.begin(), staging.end(), static_cast<T>(0x5A5A));
                } else {
                    st.push(img.data() + row * w, n, w);
                }
                row += n;
                if (st.rowsDone() < lastDone) ++wrong;   // never goes backwards
                lastDone = st.rowsDone();
            }
            if (st.rowsDone() != h) ++wrong;
            wrong += bitsWrong(sb, stride, w, h, want);
        }
    }
    setVectorArms(true);
    return wrong;
}

} // namespace

BINCV_TEST(MedianEdgeStream, ReferenceShapeEveryWidthAndHeight) {
    static const size_t kWidths[] = {1,  2,  3,  15, 16, 17, 31, 32, 33, 34, 35,
                                     47, 63, 64, 65, 66, 95, 96, 97, 100, 129};
    size_t cases = 0;
    for (size_t w : kWidths) {
        for (size_t h = 1; h <= 5; ++h) {
            for (uint8_t t : {uint8_t{17}, uint8_t{0}, uint8_t{255}}) {
                const size_t bad = sweepOne<3, uint8_t>(w, h, kMedianReferenceL, t,
                                                        0x5EED0000ULL + w * 31 + h, 256);
                BINCV_CHECK(bad == 0);
                if (bad) std::printf("  %zux%zu t=%u: %zu bits wrong\n", w, h, t, bad);
                ++cases;
            }
        }
    }
    std::printf(" reference L, 8-bit: %zu sizes x thresholds, all paths agree with the oracle\n",
                cases);
}

BINCV_TEST(MedianEdgeStream, ReferenceFrameSize) {
    // The frame the pipeline runs on, and a low-contrast image whose gradients sit
    // around the threshold so both answers occur everywhere.
    BINCV_CHECK((sweepOne<3, uint8_t>(752, 480, kMedianReferenceL, uint8_t{17}, 0xA11CEULL,
                                      40)) == 0);
    BINCV_CHECK((sweepOne<3, uint8_t>(376, 240, kMedianReferenceL, uint8_t{17}, 0xB0BULL, 40)) ==
                0);
}

BINCV_TEST(MedianEdgeStream, PatternsThatReachBelow) {
    // The plus reaches the row below, so median rows complete one push later; the
    // diagonal reaches both ways; the last reaches two rows down.
    constexpr MedianPattern<3> diagonal{{{-1, -1}, {0, 0}, {1, 1}}};
    constexpr MedianPattern<3> deep{{{0, 0}, {2, 0}, {0, -2}}};
    for (size_t w : {size_t{1}, size_t{33}, size_t{34}, size_t{70}}) {
        for (size_t h : {size_t{1}, size_t{2}, size_t{3}, size_t{6}}) {
            BINCV_CHECK((sweepOne<5, uint8_t>(w, h, kMedianReferencePlus, uint8_t{17}, 7 + w + h,
                                              256)) == 0);
            BINCV_CHECK((sweepOne<3, uint8_t>(w, h, diagonal, uint8_t{17}, 11 + w + h, 256)) == 0);
            BINCV_CHECK((sweepOne<3, uint8_t>(w, h, deep, uint8_t{17}, 13 + w + h, 256)) == 0);
        }
    }
}

BINCV_TEST(MedianEdgeStream, EveryPredicateVariant) {
    // The scalar row body carries every combination; the vector row only the shipped one.
    for (size_t w : {size_t{5}, size_t{34}, size_t{67}}) {
        const size_t h = 4;
        BINCV_CHECK((sweepOne<3, uint8_t, EdgeCombine::And, EdgeRelation::Ge, EdgeSpatial::Wide>(
                        w, h, kMedianReferenceL, uint8_t{9}, 21 + w, 64)) == 0);
        BINCV_CHECK((sweepOne<3, uint8_t, EdgeCombine::Or, EdgeRelation::Gt, EdgeSpatial::Wide>(
                        w, h, kMedianReferenceL, uint8_t{9}, 22 + w, 64)) == 0);
        BINCV_CHECK((sweepOne<3, uint8_t, EdgeCombine::Or, EdgeRelation::Ge,
                              EdgeSpatial::Forward>(w, h, kMedianReferenceL, uint8_t{9}, 23 + w,
                                                    64)) == 0);
        BINCV_CHECK((sweepOne<3, uint8_t, EdgeCombine::And, EdgeRelation::Gt,
                              EdgeSpatial::Backward>(w, h, kMedianReferenceL, uint8_t{9}, 24 + w,
                                                     64)) == 0);
    }
}

BINCV_TEST(MedianEdgeStream, SixteenBitSource) {
    // Wider than eight bits, which is in scope precisely so a small gradient survives
    // to the threshold; the vector arms do not apply and the scalar body must.
    for (size_t w : {size_t{3}, size_t{34}, size_t{100}}) {
        for (size_t h : {size_t{1}, size_t{4}}) {
            BINCV_CHECK((sweepOne<3, uint16_t>(w, h, kMedianReferenceL, uint16_t{300}, 31 + w + h,
                                               4096)) == 0);
            BINCV_CHECK((sweepOne<5, uint16_t>(w, h, kMedianReferencePlus, uint16_t{40000},
                                               37 + w + h, 65536)) == 0);
        }
    }
}

BINCV_TEST(MedianEdgeStream, ScratchIsTheNeighbourhood) {
    // What the header promises the stream holds: the input rows a later push's medians
    // reach back to, and the three median rows the edge test reads -- 4 x 752 at the
    // reference's shape. A row is rounded up to whole 4-byte words.
    BINCV_CHECK((MedianEdgeStream<3, uint8_t>::scratchElements(752, kMedianReferenceL)) == 3008);
    BINCV_CHECK((MedianEdgeStream<5, uint8_t>::scratchElements(752, kMedianReferencePlus)) ==
                5 * 752);
    BINCV_CHECK((MedianEdgeStream<3, uint8_t>::scratchElements(33, kMedianReferenceL)) == 4 * 36);
    BINCV_CHECK((MedianEdgeStream<3, uint16_t>::scratchElements(33, kMedianReferenceL)) ==
                4 * 34);
}

BINCV_TEST_MAIN("test_median_edge_stream")
