// ===========================================================================
// ops/subpix.hpp -- sub-pixel corner refinement on bit-planes.
//
// The operation claims to BE cv::cornerSubPix on the image its derivative planes came
// from, and rests on two linear facts: resampling commutes with central differencing (so
// the window's gradient at a sub-pixel position is a bilinear blend of the integer
// planes), and the solution is invariant to a scale on the image (so +/-1 refines where
// +/-255 does).
//
// The claims, in the order they matter:
// 1. ORACLE EQUALITY -- the bit-plane kernel equals a dense, obvious spelling of
//    cv::cornerSubPix's iteration TO THE BIT, over every word type, two plane depths,
//    window sizes, zero zones, iteration caps and tolerances, with seeds to the image's
//    edges and between pixels. The kernel's trick is skipping samples word-wise, and a
//    skip that drops a contributing sample is the bug that would not show as a crash.
// 2. AGREEMENT WITH cv::cornerSubPix -- on synthetic corners and on a REAL binarized
//    frame, where a refinement that agreed on clean synthetic corners has been seen to
//    part from OpenCV by a median of 0.73 px. The bounds are what a correct
//    implementation reaches, measured, and OpenCV's own disagreement with itself between
//    0/1 and 0/255 content is printed beside them as the floor.
// 3. THE BORDER CLAIM -- derivatives taken with BORDER_REFLECT_101 give the answer
//    BORDER_REPLICATE gives wherever the docstring says they do.
// 4. SCALE INVARIANCE and IT ACTUALLY REFINES.
// ===========================================================================

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bincv/ops/derivative.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/subpix.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/imgproc.hpp>
#endif

using bincv::Point2f;

namespace {

/// An integer gradient field with cv::getRectSubPix's replicated border: the image is
/// flat across the left and right edges, so `gx` is zero outside the columns and the
/// edge rows' `gx` is carried up and down; `gy` is the transpose.
struct Field {
    std::vector<int> gx, gy;
    long long w = 0, h = 0;

    int GX(long long u, long long v) const {
        if (u < 0 || u >= w) return 0;
        v = std::min(std::max(v, 0LL), h - 1);
        return gx[static_cast<size_t>(v * w + u)];
    }
    int GY(long long u, long long v) const {
        if (v < 0 || v >= h) return 0;
        u = std::min(std::max(u, 0LL), w - 1);
        return gy[static_cast<size_t>(v * w + u)];
    }
};

template <size_t N, typename W>
Field toField(const bincv::SignedQuantMat<N, W>& dx, const bincv::SignedQuantMat<N, W>& dy) {
    Field f;
    f.w = static_cast<long long>(dx.getWidth());
    f.h = static_cast<long long>(dx.getHeight());
    f.gx.resize(static_cast<size_t>(f.w * f.h));
    f.gy.resize(static_cast<size_t>(f.w * f.h));
    for (long long y = 0; y < f.h; ++y) {
        for (long long x = 0; x < f.w; ++x) {
            f.gx[static_cast<size_t>(y * f.w + x)] = dx.at(static_cast<int>(y), static_cast<int>(x));
            f.gy[static_cast<size_t>(y * f.w + x)] = dy.at(static_cast<int>(y), static_cast<int>(x));
        }
    }
    return f;
}

enum class Outcome { Refined, Singular, Outside, Diverged };

/// cv::cornerSubPix's iteration, spelled densely and obviously on an integer field: every
/// sample of every window, read through the replicated border, nothing skipped. It is
/// written from cornersubpix.cpp, not from the kernel -- the float mask, the float
/// position, the `|det| <= DBL_EPSILON^2` test, the image test, the poor-convergence
/// rule -- with the bilinear blend standing in for getRectSubPix. That blend's order, and
/// the order of the five sums, are the operation's definition, so the kernel must equal
/// this to the bit.
/// @param scale multiplies every gradient -- the knob `ScaleInvariance` turns.
/// @param trackBand if non-null, set when any window's samples read column 0 or
/// `w - 1`, or row 0 or `h - 1` -- where the derivative's border type shows.
Point2f oracleRefine(const Field& F, Point2f seed, const bincv::SubPixParams& p, double scale,
                     Outcome& outcome, bool* trackBand = nullptr) {
    const int win = p.winHalf;
    const int side = 2 * win + 1;
    std::vector<float> mask(static_cast<size_t>(side * side));
    for (int i = 0; i < side; ++i) {
        const float y = static_cast<float>(i - win) / static_cast<float>(win);
        const float vy = std::exp(-y * y);
        for (int j = 0; j < side; ++j) {
            const float x = static_cast<float>(j - win) / static_cast<float>(win);
            mask[static_cast<size_t>(i * side + j)] = vy * std::exp(-x * x);
        }
    }
    if (p.zeroHalf >= 0 && p.zeroHalf < win) {
        for (int i = win - p.zeroHalf; i <= win + p.zeroHalf; ++i)
            for (int j = win - p.zeroHalf; j <= win + p.zeroHalf; ++j)
                mask[static_cast<size_t>(i * side + j)] = 0.0f;
    }
    const int maxIters = std::min(std::max(p.maxIterations, 1), 100);
    const double eps = std::max(p.epsilon, 0.0);
    const double eps2 = eps * eps;
    const float fw = static_cast<float>(F.w), fh = static_cast<float>(F.h);

    if (!(seed.x >= 0.0f && seed.x < fw && seed.y >= 0.0f && seed.y < fh)) {
        outcome = Outcome::Outside;
        return seed;
    }
    outcome = Outcome::Refined;
    Point2f c = seed;
    int iter = 0;
    double err = 0.0;
    do {
        const float flx = std::floor(c.x), fly = std::floor(c.y);
        const long long x0 = static_cast<long long>(flx), y0 = static_cast<long long>(fly);
        const double a = static_cast<double>(c.x - flx), b = static_cast<double>(c.y - fly);
        if (trackBand != nullptr &&
            (x0 - win <= 0 || x0 + win + 1 >= F.w - 1 || y0 - win <= 0 || y0 + win + 1 >= F.h - 1))
            *trackBand = true;
        const double w00 = (1.0 - a) * (1.0 - b), w10 = a * (1.0 - b), w01 = (1.0 - a) * b,
                     w11 = a * b;
        double A = 0, B = 0, C = 0, bb1 = 0, bb2 = 0;
        for (int i = 0; i < side; ++i) {
            const long long v = y0 + i - win;
            const double py = i - win;
            for (int j = 0; j < side; ++j) {
                const long long u = x0 + j - win;
                // The blend row by row: each row's pair, then the two rows.
                const double tgx = (w00 * (scale * F.GX(u, v)) + w10 * (scale * F.GX(u + 1, v))) +
                                   (w01 * (scale * F.GX(u, v + 1)) +
                                    w11 * (scale * F.GX(u + 1, v + 1)));
                const double tgy = (w00 * (scale * F.GY(u, v)) + w10 * (scale * F.GY(u + 1, v))) +
                                   (w01 * (scale * F.GY(u, v + 1)) +
                                    w11 * (scale * F.GY(u + 1, v + 1)));
                const double m = mask[static_cast<size_t>(i * side + j)];
                const double gxx = tgx * tgx * m, gxy = tgx * tgy * m, gyy = tgy * tgy * m;
                const double px = j - win;
                A += gxx;
                B += gxy;
                C += gyy;
                bb1 += gxx * px + gxy * py;
                bb2 += gxy * px + gyy * py;
            }
        }
        const double det = A * C - B * B;
        if (std::fabs(det) <= DBL_EPSILON * DBL_EPSILON) {
            outcome = Outcome::Singular;
            break;
        }
        const double s = 1.0 / det;
        Point2f n;
        n.x = static_cast<float>(static_cast<double>(c.x) + C * s * bb1 - B * s * bb2);
        n.y = static_cast<float>(static_cast<double>(c.y) - B * s * bb1 + A * s * bb2);
        err = static_cast<double>((n.x - c.x) * (n.x - c.x) + (n.y - c.y) * (n.y - c.y));
        c = n;
        if (c.x < 0.0f || c.x >= fw || c.y < 0.0f || c.y >= fh) {
            outcome = Outcome::Outside;
            break;
        }
    } while (++iter < maxIters && err > eps2);
    if (std::fabs(c.x - seed.x) > static_cast<float>(win) ||
        std::fabs(c.y - seed.y) > static_cast<float>(win)) {
        outcome = Outcome::Diverged;
        return seed;
    }
    return c;
}

/// A quadrant corner at (cornerX, cornerY): set where both coordinates are past it.
void quadrant(bincv::BinMat<uint32_t>& m, int cornerX, int cornerY) {
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            m.set(y, x, (x >= cornerX && y >= cornerY) ? 1u : 0u);
        }
    }
}

/// The quadrant with a short notch cut from one edge two pixels from the corner: the
/// two edges no longer meet symmetrically, so the refinement has something to get wrong.
void asymmetricCorner(bincv::BinMat<uint32_t>& m, int cornerX, int cornerY) {
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            bool v = (x >= cornerX && y >= cornerY);
            if (x >= cornerX + 2 && x <= cornerX + 3 && y >= cornerY && y <= cornerY + 1) v = false;
            m.set(y, x, v ? 1u : 0u);
        }
    }
}

/// Blobs over a sparse speckle, with one flat band: straight edges, corners, isolated
/// pixels, and windows with no gradient at all.
template <typename W>
void speckleAndBlobs(bincv::BinMat<W>& img, uint64_t seed) {
    const int Wd = img.cols(), H = img.rows();
    uint64_t st = seed;
    const auto next = [&st]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return st >> 33;
    };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < Wd; ++x) img.set(y, x, next() % 23u == 0u ? 1u : 0u);
    for (int b = 0; b < 40; ++b) {
        const int x0 = static_cast<int>(next() % static_cast<uint64_t>(Wd));
        const int y0 = static_cast<int>(next() % static_cast<uint64_t>(H));
        const int w = 3 + static_cast<int>(next() % 20), h = 3 + static_cast<int>(next() % 20);
        for (int y = y0; y < y0 + h && y < H; ++y)
            for (int x = x0; x < x0 + w && x < Wd; ++x) img.set(y, x, 1u);
    }
    for (int y = H / 2 - 12; y < H / 2 + 12; ++y)
        for (int x = Wd / 3; x < Wd / 3 + 40 && x < Wd; ++x) img.set(y, x, 0u);
}

/// Seeds scattered over the frame on a quarter-pixel grid -- integer and fractional
/// positions -- plus every seed within three pixels of each edge on a coarse grid, the
/// four corners, and a few outside the image altogether.
std::vector<Point2f> scatteredSeeds(int W, int H, uint64_t seed, size_t n) {
    uint64_t st = seed;
    const auto next = [&st]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return st >> 33;
    };
    std::vector<Point2f> s;
    for (size_t i = 0; i < n; ++i) {
        s.push_back(Point2f{static_cast<float>(next() % static_cast<uint64_t>(W * 4)) / 4.0f,
                            static_cast<float>(next() % static_cast<uint64_t>(H * 4)) / 4.0f});
    }
    for (int k = 0; k < W; k += 9) {
        s.push_back(Point2f{static_cast<float>(k), 0.0f});
        s.push_back(Point2f{static_cast<float>(k) + 0.5f, static_cast<float>(H) - 1.25f});
    }
    for (int k = 0; k < H; k += 7) {
        s.push_back(Point2f{0.75f, static_cast<float>(k)});
        s.push_back(Point2f{static_cast<float>(W) - 1.0f, static_cast<float>(k) + 0.25f});
    }
    s.push_back(Point2f{0.0f, 0.0f});
    s.push_back(Point2f{static_cast<float>(W) - 0.5f, static_cast<float>(H) - 0.5f});
    s.push_back(Point2f{-1.0f, 5.0f});
    s.push_back(Point2f{static_cast<float>(W), 5.0f});
    s.push_back(Point2f{5.0f, NAN});
    return s;
}

struct OracleTally {
    size_t cases = 0, corners = 0, mismatched = 0;
    size_t refined = 0, singular = 0, outside = 0, diverged = 0, fractional = 0, band = 0;
};

/// Holds the kernel to the oracle on one field, over a family of parameter sets.
template <size_t N, typename W>
void holdToOracle(const bincv::SignedQuantMat<N, W>& dx, const bincv::SignedQuantMat<N, W>& dy,
                  const std::vector<Point2f>& seeds, const std::vector<bincv::SubPixParams>& sets,
                  OracleTally& t) {
    const Field F = toField(dx, dy);
    for (const bincv::SubPixParams& p : sets) {
        std::vector<Point2f> mine = seeds;
        const bincv::SubPixResult r =
            bincv::cornerSubPix<N, W>(dx, dy, mine.data(), mine.size(), p);
        size_t counts[4] = {0, 0, 0, 0};
        for (size_t i = 0; i < seeds.size(); ++i) {
            Outcome o;
            bool band = false;
            const Point2f ref = oracleRefine(F, seeds[i], p, 1.0, o, &band);
            ++counts[static_cast<int>(o)];
            // Bit for bit: NaN seeds are returned as they came, and compare by bytes.
            if (std::memcmp(&ref, &mine[i], sizeof(Point2f)) != 0) ++t.mismatched;
            if (o == Outcome::Refined && (ref.x != std::floor(ref.x) || ref.y != std::floor(ref.y)))
                ++t.fractional;
            t.band += band;
        }
        BINCV_CHECK_EQ(r.refined, counts[0]);
        BINCV_CHECK_EQ(r.singular, counts[1]);
        BINCV_CHECK_EQ(r.outside, counts[2]);
        BINCV_CHECK_EQ(r.diverged, counts[3]);
        BINCV_CHECK_EQ(r.refined + r.singular + r.outside + r.diverged, seeds.size());
        t.refined += r.refined;
        t.singular += r.singular;
        t.outside += r.outside;
        t.diverged += r.diverged;
        t.corners += seeds.size();
        ++t.cases;
    }
}

std::vector<bincv::SubPixParams> paramFamily() {
    std::vector<bincv::SubPixParams> sets;
    for (const int winHalf : {1, 3, 5, 10, 15}) {
        // {zeroHalf, maxIterations, epsilon}: the defaults; a hole and a loose tolerance;
        // a hole as wide as the window, which OpenCV ignores; one iteration and an exact
        // tolerance; and the caps on both sides of OpenCV's [1, 100].
        const struct { int zero, iters; double eps; } kSets[] = {
            {-1, 40, 0.001}, {winHalf > 2 ? 2 : 0, 5, 0.03}, {winHalf, 20, 0.001},
            {-1, 1, 0.0},    {-1, 0, 0.03},                   {-1, 500, 1e-9}};
        for (const auto& k : kSets) {
            bincv::SubPixParams p;
            p.winHalf = winHalf;
            p.zeroHalf = k.zero;
            p.maxIterations = k.iters;
            p.epsilon = k.eps;
            sets.push_back(p);
        }
    }
    return sets;
}

template <typename W>
void ternaryOracleCase(uint64_t seed, OracleTally& t) {
    constexpr int Wd = 157, H = 101;
    bincv::BinMat<W> img(Wd, H);
    speckleAndBlobs(img, seed);
    bincv::SignedQuantMat<1, W> dx(Wd, H), dy(Wd, H);
    bincv::derivativeX(img, dx, bincv::BORDER_REPLICATE);
    bincv::derivativeY(img, dy, bincv::BORDER_REPLICATE);
    holdToOracle<1, W>(dx, dy, scatteredSeeds(Wd, H, seed ^ 0x9E37u, 60), paramFamily(), t);
}

#ifdef BINCV_REALFRAME_PATH
/// The committed real frame: 752x480 bits, one byte a pixel. Empty if unreadable.
std::vector<uint8_t> loadRealFrame(int& w, int& h) {
    std::vector<uint8_t> px;
    FILE* f = std::fopen(BINCV_REALFRAME_PATH, "rb");
    if (!f) return px;
    uint32_t fw = 0, fh = 0;
    if (std::fread(&fw, 4, 1, f) == 1 && std::fread(&fh, 4, 1, f) == 1) {
        px.resize(static_cast<size_t>(fw) * fh);
        if (std::fread(px.data(), 1, px.size(), f) != px.size()) px.clear();
    }
    std::fclose(f);
    w = static_cast<int>(fw);
    h = static_cast<int>(fh);
    return px;
}
#endif

}  // namespace

BINCV_TEST(SubPix, MatchesDenseOracle_EveryWordTypeAndWindow) {
    // The kernel finds contributing samples a WORD at a time -- the planes OR-ed with
    // themselves one column right and one row down -- and visits only those; on an
    // integer position it reads one row and no neighbours; a window past the image goes
    // through the replicated border instead. Every one of those must be the dense
    // oracle's answer to the bit, and every counter the oracle's outcome.
    OracleTally t;
    ternaryOracleCase<uint8_t>(0x5B9, t);
    ternaryOracleCase<uint32_t>(0x5BA, t);
    ternaryOracleCase<uint64_t>(0x5BB, t);
    std::printf(" ternary: %zu parameter sets, %zu corners, %zu mismatched; refined %zu (%zu"
                " fractional), singular %zu, outside %zu, diverged %zu; %zu reached the band\n",
                t.cases, t.corners, t.mismatched, t.refined, t.fractional, t.singular, t.outside,
                t.diverged, t.band);
    BINCV_CHECK_EQ(t.mismatched, size_t{0});
    // Not vacuous: every outcome happens, most walks end between pixels, and the border
    // path runs.
    BINCV_CHECK(t.refined > 5000);
    BINCV_CHECK(t.fractional * 2 > t.refined);
    BINCV_CHECK(t.singular > 0);
    BINCV_CHECK(t.outside > 0);
    BINCV_CHECK(t.diverged > 0);
    BINCV_CHECK(t.band > 1000);
}

BINCV_TEST(SubPix, MatchesDenseOracle_TwoBitPlanes) {
    // N = 2: values in [-3, 3], so the blend's products round and the planes are
    // assembled from two magnitude bits. A field that is not any image's derivative --
    // the oracle does not care, and neither may the kernel.
    constexpr int Wd = 131, H = 77;
    bincv::SignedQuantMat<2, uint32_t> dx(Wd, H), dy(Wd, H);
    uint64_t st = 0xC0FFEE;
    const auto next = [&st]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return st >> 33;
    };
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < Wd; ++x) {
            // Mostly zero, so the skip has something to skip.
            const bool on = (next() % 5u) == 0u || ((x / 9 + y / 7) % 4 == 0);
            dx.set(y, x, on ? static_cast<int>(next() % 7u) - 3 : 0);
            dy.set(y, x, on ? static_cast<int>(next() % 7u) - 3 : 0);
        }
    }
    OracleTally t;
    std::vector<bincv::SubPixParams> sets;
    for (const int winHalf : {2, 5, 9}) {
        bincv::SubPixParams p;
        p.winHalf = winHalf;
        sets.push_back(p);
        p.epsilon = 0.03;
        p.maxIterations = 7;
        sets.push_back(p);
    }
    holdToOracle<2, uint32_t>(dx, dy, scatteredSeeds(Wd, H, 0xAB, 80), sets, t);
    std::printf(" two-bit: %zu parameter sets, %zu corners, %zu mismatched; refined %zu (%zu"
                " fractional)\n", t.cases, t.corners, t.mismatched, t.refined, t.fractional);
    BINCV_CHECK_EQ(t.mismatched, size_t{0});
    BINCV_CHECK(t.fractional > 100);
}

#ifdef BINCV_REALFRAME_PATH
BINCV_TEST(SubPix, MatchesDenseOracle_OnARealFrame) {
    // Real content: edges a binarized camera frame actually has, crossing word
    // boundaries at every phase. Seeds on a grid, so most are not corners and walk.
    int w = 0, h = 0;
    const std::vector<uint8_t> frame = loadRealFrame(w, h);
    BINCV_CHECK(!frame.empty());
    if (frame.empty()) return;
    bincv::BinMat<uint32_t> bin(w, h);
    bincv::packBits<bincv::PackRule::NonZero>(frame.data(), static_cast<size_t>(w),
                                              static_cast<size_t>(h), static_cast<size_t>(w),
                                              bin.view());
    bincv::SignedQuantMat<1, uint32_t> dx(w, h), dy(w, h);
    bincv::derivativeX(bin, dx, bincv::BORDER_REPLICATE);
    bincv::derivativeY(bin, dy, bincv::BORDER_REPLICATE);
    std::vector<Point2f> seeds;
    for (int y = 0; y < h; y += 23)
        for (int x = 0; x < w; x += 19) seeds.push_back(Point2f{static_cast<float>(x),
                                                                static_cast<float>(y)});
    bincv::SubPixParams a, b;
    b.winHalf = 10;
    b.maxIterations = 20;
    b.epsilon = 0.03;
    OracleTally t;
    holdToOracle<1, uint32_t>(dx, dy, seeds, {a, b}, t);
    std::printf(" real frame: %zu corners, %zu mismatched; refined %zu (%zu fractional),"
                " singular %zu, outside %zu, diverged %zu\n", t.corners, t.mismatched, t.refined,
                t.fractional, t.singular, t.outside, t.diverged);
    BINCV_CHECK_EQ(t.mismatched, size_t{0});
    BINCV_CHECK(t.fractional > 300);
}
#endif

BINCV_TEST(SubPix, ScaleInvariance_TernaryRefinesWhereBytesDo) {
    // Same gradients, scaled by 1 and by 255: G and b both scale by s^2, so the solution
    // must not move beyond rounding -- which is what lets +/-1 stand in for 0/255.
    constexpr int W = 60, H = 60;
    bincv::BinMat<uint32_t> img(W, H);
    asymmetricCorner(img, 30, 30);
    bincv::SignedQuantMat<1, uint32_t> dx(W, H), dy(W, H);
    bincv::derivativeX(img, dx, bincv::BORDER_REPLICATE);
    bincv::derivativeY(img, dy, bincv::BORDER_REPLICATE);
    const Field F = toField(dx, dy);

    double worst = 0.0, total = 0.0;
    size_t n = 0;
    bincv::SubPixParams p;
    for (int sy = -8; sy <= 8; ++sy) {
        for (int sx = -8; sx <= 8; ++sx) {
            const Point2f start{30.0f + 0.25f * static_cast<float>(sx),
                                30.0f + 0.25f * static_cast<float>(sy)};
            Outcome o1, o2;
            const Point2f one = oracleRefine(F, start, p, 1.0, o1);
            const Point2f big = oracleRefine(F, start, p, 255.0, o2);
            const double d = std::hypot(static_cast<double>(one.x) - big.x,
                                        static_cast<double>(one.y) - big.y);
            worst = d > worst ? d : worst;
            total += d;
            ++n;
        }
    }
    std::printf(" +/-1 vs +/-255 over %zu starts: mean %.3e px, worst %.3e px\n", n,
                total / static_cast<double>(n), worst);
    BINCV_CHECK(n == 289);
    BINCV_CHECK(worst < 1e-4);
}

BINCV_TEST(SubPix, ActuallyRefines_TowardsAKnownCorner) {
    // A quadrant corner whose two edges meet exactly between pixels (31.5, 26.5) in the
    // image's own coordinates: the central difference puts the gradient on the pixels
    // either side of each edge. Start off it and check the refinement moves TOWARD it --
    // an operation that runs, converges and reports success while making localisation
    // worse would pass the oracle tests above.
    constexpr int W = 70, H = 60;
    const int cxTrue = 32, cyTrue = 27;
    const double tx = cxTrue - 0.5, ty = cyTrue - 0.5;
    bincv::BinMat<uint32_t> img(W, H);
    quadrant(img, cxTrue, cyTrue);
    bincv::SignedQuantMat<1, uint32_t> dx(W, H), dy(W, H);
    bincv::derivativeX(img, dx);
    bincv::derivativeY(img, dy);

    double beforeSum = 0.0, afterSum = 0.0;
    size_t improved = 0, cases = 0;
    for (int sy = -2; sy <= 2; ++sy) {
        for (int sx = -2; sx <= 2; ++sx) {
            Point2f pt{static_cast<float>(cxTrue + sx), static_cast<float>(cyTrue + sy)};
            const double before = std::hypot(static_cast<double>(pt.x) - tx,
                                             static_cast<double>(pt.y) - ty);
            bincv::cornerSubPix<1, uint32_t>(dx, dy, &pt, 1);
            const double after = std::hypot(static_cast<double>(pt.x) - tx,
                                            static_cast<double>(pt.y) - ty);
            beforeSum += before;
            afterSum += after;
            if (after < before) ++improved;
            ++cases;
        }
    }
    std::printf(" %zu starts around a known corner: mean distance %.3f -> %.3f px,"
                " %zu improved\n", cases, beforeSum / static_cast<double>(cases),
                afterSum / static_cast<double>(cases), improved);
    BINCV_CHECK(cases == 25);
    BINCV_CHECK(afterSum < beforeSum * 0.1);
    BINCV_CHECK(improved == cases);
}

BINCV_TEST(SubPix, ReflectedBorderAgreesWhereTheDocstringSaysItDoes) {
    // ops/derivative.hpp defaults to BORDER_REFLECT_101, which makes dx zero in the first
    // and last column and dy in the first and last row, where cv::cornerSubPix's
    // replicated border has the one-sided difference. The docstring says a walk whose
    // windows never read those columns or rows gets the same answer from either -- held
    // here, corner by corner -- and that the rest may differ, which is also checked, so
    // the band is shown to be the one that matters.
    constexpr int Wd = 157, H = 101;
    bincv::BinMat<uint32_t> img(Wd, H);
    speckleAndBlobs(img, 0x77);
    for (int y = 0; y < H; ++y) {   // edges that run INTO the border
        img.set(y, 0, (y / 5) % 2 ? 1u : 0u);
        img.set(y, Wd - 1, (y / 4) % 2 ? 1u : 0u);
    }
    for (int x = 0; x < Wd; ++x) {
        img.set(0, x, (x / 6) % 2 ? 1u : 0u);
        img.set(H - 1, x, (x / 3) % 2 ? 1u : 0u);
    }
    bincv::SignedQuantMat<1, uint32_t> rx(Wd, H), ry(Wd, H), fx(Wd, H), fy(Wd, H);
    bincv::derivativeX(img, rx, bincv::BORDER_REPLICATE);
    bincv::derivativeY(img, ry, bincv::BORDER_REPLICATE);
    bincv::derivativeX(img, fx);
    bincv::derivativeY(img, fy);
    const Field F = toField(rx, ry);

    const std::vector<Point2f> seeds = scatteredSeeds(Wd, H, 0x31, 400);
    size_t clear = 0, inBand = 0, bandDiffers = 0;
    for (const int winHalf : {3, 5, 10}) {
        bincv::SubPixParams p;
        p.winHalf = winHalf;
        std::vector<Point2f> rep = seeds, ref = seeds;
        bincv::cornerSubPix<1, uint32_t>(rx, ry, rep.data(), rep.size(), p);
        bincv::cornerSubPix<1, uint32_t>(fx, fy, ref.data(), ref.size(), p);
        for (size_t i = 0; i < seeds.size(); ++i) {
            Outcome o;
            bool band = false;
            oracleRefine(F, seeds[i], p, 1.0, o, &band);
            const bool same = std::memcmp(&rep[i], &ref[i], sizeof(Point2f)) == 0;
            if (band) {
                ++inBand;
                bandDiffers += !same;
            } else {
                ++clear;
                BINCV_CHECK(same);
            }
        }
    }
    std::printf(" %zu walks clear of the outermost rows and columns, all identical; %zu reach"
                " them, %zu of those differ\n", clear, inBand, bandDiffers);
    BINCV_CHECK(clear > 500);
    BINCV_CHECK(bandDiffers > 0);
}

BINCV_TEST(SubPix, SeedsOutsideTheImageAreLeftAndCounted) {
    constexpr int W = 40, H = 30;
    bincv::BinMat<uint32_t> img(W, H);
    quadrant(img, 20, 15);
    bincv::SignedQuantMat<1, uint32_t> dx(W, H), dy(W, H);
    bincv::derivativeX(img, dx, bincv::BORDER_REPLICATE);
    bincv::derivativeY(img, dy, bincv::BORDER_REPLICATE);
    Point2f pts[] = {{-0.5f, 3.0f}, {40.0f, 3.0f}, {3.0f, 30.0f}, {NAN, 3.0f}, {20.0f, 15.0f}};
    Point2f before[5];
    std::memcpy(before, pts, sizeof pts);
    const bincv::SubPixResult r = bincv::cornerSubPix<1, uint32_t>(dx, dy, pts, 5);
    BINCV_CHECK_EQ(r.outside, size_t{4});
    BINCV_CHECK_EQ(r.refined, size_t{1});
    BINCV_CHECK(std::memcmp(pts, before, 4 * sizeof(Point2f)) == 0);
}

#ifdef BINCV_WITH_OPENCV
// ---------------------------------------------------------------------------
// AGAINST cv::cornerSubPix, AND THE TWO PLACES ITS VERSIONS DISAGREE
//
// cv::cornerSubPix resamples its window with cv::getRectSubPix, and two builds measured
// here do not resample alike. 4.5.4's 8-bit path clamps the horizontal fraction to at
// least 1e-4, so a window at an integer x is sampled 1e-4 px right of it; 4.8.0's reads
// column `width - 2` for the part of a window that lies above the image and right of its
// last column. This kernel computes the exact replicated-border blend that both
// approximate, and with a loose epsilon a 1e-4 px first step is enough to end a walk a
// step apart: on 34,200 EuRoC corners at window 21 and eps 0.03, 4.5.4 and 4.8.0 put
// 4.3% of them more than 1e-3 px apart, and binCV is that far from 4.5.4 and within 0.15%
// of 4.8.0 (benchmark/subpix_agreement.cpp measures a sequence, version effects and all).
//
// The cases below hold the kernel to whichever OpenCV the build links, so they keep
// their seeds off the integer grid and their windows away from the image's top-right
// corner -- the two places where the answer depends on the OpenCV version rather than
// on this kernel.
// ---------------------------------------------------------------------------
namespace {

/// One comparison: binCV against cv::cornerSubPix on the 0/255 image, and -- the floor --
/// cv::cornerSubPix on the same bits as 0/1 against itself on 0/255. Per-corner distances.
struct Agreement {
    std::vector<double> bincv, floor;
    double worst() const { return bincv.empty() ? 0.0 : *std::max_element(bincv.begin(), bincv.end()); }
    double floorWorst() const {
        return floor.empty() ? 0.0 : *std::max_element(floor.begin(), floor.end());
    }
    size_t beyond(double r) const {
        return static_cast<size_t>(std::count_if(bincv.begin(), bincv.end(), [r](double d) { return d > r; }));
    }
    size_t floorBeyond(double r) const {
        return static_cast<size_t>(std::count_if(floor.begin(), floor.end(), [r](double d) { return d > r; }));
    }
};

double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size() - 1) + 0.5))];
}

/// Two float ulps of a position below `extent` px: the resolution at which a float
/// position can agree at all.
double positionUlps(int extent) {
    return 2.0 * std::ldexp(1.0, std::ilogb(static_cast<double>(extent)) - 23);
}

/// Whether a walk from `s` can sample the part of a window above the image and right of
/// its last column, where OpenCV 4.8.0 reads the wrong column. A walk that stays within
/// half a window of its seed -- the ones the poor-convergence rule keeps -- samples at
/// most `2 * winHalf + 1` columns and rows from it.
bool nearTopRight(const cv::Point2f& s, int width, int winHalf) {
    return s.x > static_cast<float>(width - 3 * winHalf - 3) &&
           s.y < static_cast<float>(3 * winHalf + 2);
}

Agreement againstOpenCV(const bincv::BinMat<uint32_t>& img, const std::vector<cv::Point2f>& seeds,
                        const bincv::SubPixParams& p) {
    const int W = img.cols(), H = img.rows();
    bincv::SignedQuantMat<1, uint32_t> dx(W, H), dy(W, H);
    bincv::derivativeX(img, dx, bincv::BORDER_REPLICATE);
    bincv::derivativeY(img, dy, bincv::BORDER_REPLICATE);
    cv::Mat bytes(H, W, CV_8U, cv::Scalar(0)), ones(H, W, CV_8U, cv::Scalar(0));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            bytes.at<uint8_t>(y, x) = img.at(y, x) ? uint8_t{255} : uint8_t{0};
            ones.at<uint8_t>(y, x) = img.at(y, x) ? uint8_t{1} : uint8_t{0};
        }
    }
    std::vector<Point2f> mine;
    for (const cv::Point2f& s : seeds) mine.push_back(Point2f{s.x, s.y});
    std::vector<cv::Point2f> theirs = seeds, theirsOnes = seeds;
    bincv::cornerSubPix<1, uint32_t>(dx, dy, mine.data(), mine.size(), p);
    const cv::TermCriteria crit(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                p.maxIterations, p.epsilon);
    const cv::Size win(p.winHalf, p.winHalf), zero(p.zeroHalf, p.zeroHalf);
    cv::cornerSubPix(bytes, theirs, win, zero, crit);
    cv::cornerSubPix(ones, theirsOnes, win, zero, crit);
    Agreement a;
    for (size_t i = 0; i < mine.size(); ++i) {
        a.bincv.push_back(std::hypot(static_cast<double>(mine[i].x) - theirs[i].x,
                                     static_cast<double>(mine[i].y) - theirs[i].y));
        a.floor.push_back(std::hypot(static_cast<double>(theirsOnes[i].x) - theirs[i].x,
                                     static_cast<double>(theirsOnes[i].y) - theirs[i].y));
    }
    return a;
}

/// Seeds on a quarter-pixel grid around (cx, cy), an eighth of a pixel off it in x.
std::vector<cv::Point2f> gridSeeds(int cx, int cy) {
    std::vector<cv::Point2f> s;
    for (int sy = -8; sy <= 8; ++sy)
        for (int sx = -8; sx <= 8; ++sx)
            s.push_back(cv::Point2f(static_cast<float>(cx) + 0.25f * static_cast<float>(sx) + 0.125f,
                                    static_cast<float>(cy) + 0.25f * static_cast<float>(sy)));
    return s;
}

}  // namespace

BINCV_TEST(SubPix, AgreesWithOpenCV_OnSyntheticCorners) {
    // Symmetric and asymmetric corners, seeds on a quarter-pixel grid around them -- and a
    // corner at the image's top-left corner, where every window reaches past the border.
    //
    // THE BOUND IS OpenCV's OWN. OpenCV resamples in float and this blends ternary values
    // in double, so they agree to rounding -- and where a walk is ill-conditioned,
    // rounding ends it a step apart. cv::cornerSubPix does the same to ITSELF between the
    // same bits as 0/1 and as 0/255, so each case is held to that floor plus a couple of
    // float ulps, rather than to a number chosen from what this code happened to produce.
    constexpr int W = 70, H = 60;
    bincv::BinMat<uint32_t> sym(W, H), asym(W, H), edge(W, H);
    quadrant(sym, 32, 27);
    asymmetricCorner(asym, 32, 27);
    quadrant(edge, 3, 2);
    size_t n = 0;
    double worst = 0.0;
    for (const bincv::BinMat<uint32_t>* img : {&sym, &asym, &edge}) {
        const bool atEdge = img == &edge;
        for (const int winHalf : {3, 5, 8}) {
            bincv::SubPixParams p;
            p.winHalf = winHalf;
            const Agreement a = againstOpenCV(*img, gridSeeds(atEdge ? 4 : 32, atEdge ? 4 : 27), p);
            std::printf("  %-12s window %2d: worst %.3e px (OpenCV against itself %.3e)\n",
                        img == &sym ? "symmetric" : img == &asym ? "asymmetric" : "image corner",
                        2 * winHalf + 1, a.worst(), a.floorWorst());
            BINCV_CHECK(a.worst() <= a.floorWorst() + positionUlps(W));
            worst = std::max(worst, a.worst());
            n += a.bincv.size();
        }
    }
    BINCV_CHECK_EQ(n, size_t{2601});
    // Most of these are well-conditioned, and there the floor is a float ulp: a
    // refinement anchored on the rounded position was 0.47 px from OpenCV on the
    // asymmetric corner, which no floor here comes near.
    BINCV_CHECK(worst < 1e-3);
}

BINCV_TEST(SubPix, AgreesWithOpenCV_ZeroZoneAndIterationCaps) {
    // OpenCV applies a zero zone only when it is smaller than the window, and takes the
    // iteration count into [1, 100]; the kernel must do both as it does. A 5x5 hole over
    // a small corner leaves walks that wander rather than converge -- OpenCV against
    // itself is a tenth of a pixel apart there -- so the floor is what holds it.
    constexpr int W = 70, H = 60;
    bincv::BinMat<uint32_t> img(W, H);
    asymmetricCorner(img, 32, 27);
    const struct { int win, zero, iters; double eps; } kCases[] = {
        {5, 2, 40, 0.001}, {5, 5, 40, 0.001}, {5, 9, 40, 0.001},
        {5, -1, 0, 0.001}, {5, -1, 150, 0.0}, {3, 1, 1, 0.0}};
    for (const auto& k : kCases) {
        bincv::SubPixParams p;
        p.winHalf = k.win;
        p.zeroHalf = k.zero;
        p.maxIterations = k.iters;
        p.epsilon = k.eps;
        const Agreement a = againstOpenCV(img, gridSeeds(32, 27), p);
        std::printf("  window %d, zero zone %2d, %3d iterations, eps %g: worst %.3e px (OpenCV"
                    " against itself %.3e)\n", 2 * k.win + 1, k.zero, k.iters, k.eps, a.worst(),
                    a.floorWorst());
        BINCV_CHECK(a.worst() <= a.floorWorst() + positionUlps(W));
    }
}

#ifdef BINCV_REALFRAME_PATH
namespace {

/// The committed real frame as an image, as its bits, and its detector's corners -- moved
/// an eighth of a pixel off the integer grid in x and clear of the top-right corner.
struct RealFrameCase {
    int w = 0, h = 0;
    bincv::BinMat<uint32_t> img{1, 1};
    std::vector<cv::Point2f> seeds;

    bool load(int maxWinHalf) {
        const std::vector<uint8_t> frame = loadRealFrame(w, h);
        if (frame.empty()) return false;
        img = bincv::BinMat<uint32_t>(w, h);
        bincv::packBits<bincv::PackRule::NonZero>(frame.data(), static_cast<size_t>(w),
                                                  static_cast<size_t>(h), static_cast<size_t>(w),
                                                  img.view());
        cv::Mat bytes(h, w, CV_8U);
        for (size_t i = 0; i < frame.size(); ++i) bytes.data[i] = frame[i] ? uint8_t{255} : uint8_t{0};
        std::vector<cv::Point2f> found;
        cv::goodFeaturesToTrack(bytes, found, 400, 0.01, 8.0);
        for (cv::Point2f s : found) {
            s.x += 0.125f;
            if (!nearTopRight(s, w, maxWinHalf)) seeds.push_back(s);
        }
        return true;
    }
};

}  // namespace

BINCV_TEST(SubPix, AgreesWithOpenCV_OneStepOnARealFrame) {
    // ONE ITERATION, which is the window's sums and nothing else: no termination to flip,
    // no walk to amplify a difference. Every corner a detector finds on a real frame,
    // borders included, must take OpenCV's step to the float position's resolution.
    RealFrameCase c;
    BINCV_CHECK(c.load(10));
    if (c.seeds.empty()) return;
    for (const int winHalf : {5, 10}) {
        bincv::SubPixParams p;
        p.winHalf = winHalf;
        p.maxIterations = 1;
        const Agreement a = againstOpenCV(c.img, c.seeds, p);
        std::printf(" real frame, one step, window %d: %zu corners, worst %.2e px (OpenCV against"
                    " itself %.2e)\n", 2 * winHalf + 1, a.bincv.size(), a.worst(), a.floorWorst());
        BINCV_CHECK(a.bincv.size() > 300);
        BINCV_CHECK(a.worst() <= positionUlps(c.w));
    }
}

BINCV_TEST(SubPix, AgreesWithOpenCV_OnARealFrame) {
    // THE WHOLE WALK, on the corners a detector finds on a real binarized frame, at this
    // struct's defaults and at a 21-pixel window. A refinement that anchored each window
    // on the rounded position passed the synthetic cases of its day and was a median of
    // 0.98 px from cv::cornerSubPix on this frame at the defaults, 1.39 px at window 21.
    //
    // The residual is rounding, made visible by an occasional walk that is ill-conditioned
    // enough to end one step apart. OpenCV has the same residual against ITSELF between
    // 0/1 and 0/255 content -- printed beside it as the floor no reimplementation can go
    // under.
    RealFrameCase c;
    BINCV_CHECK(c.load(10));
    if (c.seeds.empty()) return;
    bincv::SubPixParams defaults, wide;
    wide.winHalf = 10;
    wide.maxIterations = 20;
    wide.epsilon = 0.03;
    for (const bincv::SubPixParams& p : {defaults, wide}) {
        const Agreement a = againstOpenCV(c.img, c.seeds, p);
        std::printf(" real frame, window %d: %zu corners, median %.2e px, p99 %.2e px, max %.2e"
                    " px, %zu beyond 1e-3 px (OpenCV against itself: max %.2e, %zu beyond)\n",
                    2 * p.winHalf + 1, a.bincv.size(), quantile(a.bincv, 0.5),
                    quantile(a.bincv, 0.99), a.worst(), a.beyond(1e-3), a.floorWorst(),
                    a.floorBeyond(1e-3));
        BINCV_CHECK(a.bincv.size() > 300);
        BINCV_CHECK(quantile(a.bincv, 0.5) <= positionUlps(c.w));
        BINCV_CHECK(quantile(a.bincv, 0.99) < 1e-3);
        BINCV_CHECK(a.beyond(1e-3) <= a.floorBeyond(1e-3) + 1);
    }
}
#endif  // BINCV_REALFRAME_PATH
#endif  // BINCV_WITH_OPENCV

// `impl::ctzWord` is the compiler builtin where one exists and `impl::ctzPortable`
// elsewhere; the two must agree on every bit position, or a build without the
// builtin would read a different first set bit and refine a different pixel.
BINCV_TEST(SubPix, CtzPortableMatchesTheBuiltin) {
    size_t wrong = 0;
    for (unsigned bit = 0; bit < 64; ++bit) {
        const unsigned long long lone = 1ull << bit;
        // the lone bit, and the same bit with everything above it set
        const unsigned long long crowded = lone | (~0ull << bit);
        if (bincv::impl::ctzPortable(lone) != bit) ++wrong;
        if (bincv::impl::ctzPortable(crowded) != bit) ++wrong;
        if (bincv::impl::ctzWord(lone) != bincv::impl::ctzPortable(lone)) ++wrong;
        if (bincv::impl::ctzWord(crowded) != bincv::impl::ctzPortable(crowded)) ++wrong;
#if defined(__GNUC__) || defined(__clang__)
        if (bincv::impl::ctzWord(crowded) != static_cast<size_t>(__builtin_ctzll(crowded)))
            ++wrong;
#endif
    }
    std::printf(" ctz: %zu disagreements over 64 bit positions\n", wrong);
    BINCV_CHECK(wrong == 0);
}

BINCV_TEST_MAIN("test_subpix")
