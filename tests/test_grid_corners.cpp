// test_grid_corners.cpp -- gridCorners against a brute-force oracle, and its two forms
// against each other.
//
// The oracle is the operation written the obvious way: the whole response map, every
// 3x3 local maximum, each cell's candidates sorted, the thresholds, the spacing walk,
// the cap. The kernel does it in one sweep over a three-row ring with a few slots a
// cell; the two must agree on every corner, in order, to the bit.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/gridCorners.hpp"
#include "test_util.hpp"

namespace {

using namespace bincv;

struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 33);
    }
};

/// Blobs over a sparse speckle: straight edges, corners and isolated pixels.
BinMat<uint32_t> scene(int w, int h, uint64_t seed) {
    BinMat<uint32_t> img(w, h);
    Lcg r(seed);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) img.set(y, x, r.next() % 29u == 0u ? 1u : 0u);
    for (int b = 0; b < (w * h) / 300; ++b) {
        const int x0 = static_cast<int>(r.next() % static_cast<uint32_t>(w));
        const int y0 = static_cast<int>(r.next() % static_cast<uint32_t>(h));
        const int bw = 2 + static_cast<int>(r.next() % 14u), bh = 2 + static_cast<int>(r.next() % 14u);
        for (int y = y0; y < y0 + bh && y < h; ++y)
            for (int x = x0; x < x0 + bw && x < w; ++x) img.set(y, x, 1u);
    }
    return img;
}

bool stronger(const Corner& a, const Corner& b) {
    uint32_t ra, rb;
    std::memcpy(&ra, &a.response, sizeof ra);
    std::memcpy(&rb, &b.response, sizeof rb);
    if (ra != rb) return ra > rb;
    if (a.y != b.y) return a.y > b.y;
    return a.x > b.x;
}

/// The operation, spelled the obvious way.
std::vector<Corner> oracle(const std::vector<float>& map, int w, int h,
                           const GridCornerParams& p, const uint8_t* existing,
                           const BinMat<uint32_t>* mask, size_t capacity) {
    const auto at = [&](int x, int y) { return map[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)]; };
    const auto admitted = [&](int x, int y) { return mask == nullptr || mask->at(y, x) != 0; };
    float maxResp = -1.0f;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (admitted(x, y)) maxResp = std::max(maxResp, at(x, y));
    std::vector<Corner> none;
    if (!(maxResp > 0.0f)) return none;
    const float threshold = static_cast<float>(static_cast<double>(maxResp) * p.qualityLevel);

    const int across = (w + p.cellWidth - 1) / p.cellWidth;
    const int down = (h + p.cellHeight - 1) / p.cellHeight;
    std::vector<std::vector<Corner>> cells(static_cast<size_t>(across * down));
    for (int y = 1; y + 1 < h; ++y) {
        for (int x = 1; x + 1 < w; ++x) {
            const float v = at(x, y);
            bool peak = true;
            for (int dy = -1; dy <= 1 && peak; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    if (at(x + dx, y + dy) > v) peak = false;
            if (!peak || !admitted(x, y) || !(v > p.minResponse)) continue;
            cells[static_cast<size_t>((y / p.cellHeight) * across + x / p.cellWidth)].push_back(
                Corner{x, y, v});
        }
    }
    std::vector<Corner> ranked;
    for (size_t c = 0; c < cells.size(); ++c) {
        std::sort(cells[c].begin(), cells[c].end(), stronger);
        const int have = existing != nullptr ? existing[c] : 0;
        const int quota = std::max(0, p.perCell - have);
        for (int i = 0; i < quota && i < static_cast<int>(cells[c].size()); ++i) {
            if (cells[c][static_cast<size_t>(i)].response > threshold)
                ranked.push_back(cells[c][static_cast<size_t>(i)]);
        }
    }
    std::sort(ranked.begin(), ranked.end(), stronger);
    size_t limit = capacity;
    if (p.maxCorners > 0) limit = std::min(limit, static_cast<size_t>(p.maxCorners));
    std::vector<Corner> out;
    const long long limitSq = static_cast<long long>(std::ceil(p.minDistance * p.minDistance));
    for (const Corner& c : ranked) {
        if (out.size() == limit) break;
        bool good = true;
        if (p.minDistance >= 1.0) {
            for (const Corner& k : out) {
                const long long dx = k.x - c.x, dy = k.y - c.y;
                if (dx * dx + dy * dy < limitSq) good = false;
            }
        }
        if (good) out.push_back(c);
    }
    return out;
}

void derivatives(const BinMat<uint32_t>& img, TernaryMat<uint32_t>& dx, TernaryMat<uint32_t>& dy) {
    derivativeX(img, dx);
    derivativeY(img, dy);
}

}  // namespace

BINCV_TEST(GridCorners, MatchesTheObviousSpelling) {
    struct Shape {
        int w, h;
    };
    const Shape shapes[] = {{160, 120}, {97, 61}, {33, 200}};
    size_t cases = 0, cornersCompared = 0, truncations = 0;
    Lcg r(0x6121D);
    for (const Shape& sh : shapes) {
        const BinMat<uint32_t> img = scene(sh.w, sh.h, 0x9A1D + static_cast<uint64_t>(sh.w));
        TernaryMat<uint32_t> dx(sh.w, sh.h);
        TernaryMat<uint32_t> dy(sh.w, sh.h);
        derivatives(img, dx, dy);
        std::vector<float> map(static_cast<size_t>(sh.w * sh.h));
        cornerMinEigenVal<uint32_t>(dx, dy, 3,
                                    ResponseMap(map.data(), static_cast<size_t>(sh.w),
                                                static_cast<size_t>(sh.h),
                                                static_cast<size_t>(sh.w)));
        BinMat<uint32_t> mask(sh.w, sh.h);
        for (int y = 0; y < sh.h; ++y)
            for (int x = 0; x < sh.w; ++x) mask.set(y, x, (x / 7 + y / 5) % 3 != 0 ? 1u : 0u);
        std::vector<float> ring(static_cast<size_t>(sh.w) * kResponseRingRows);

        for (const int cell : {8, 16, 37, 400}) {
            for (const int perCell : {1, 2, 4}) {
                for (const double q : {0.0, 0.01, 0.3}) {
                    for (const float minResp : {0.0f, 0.75f}) {
                        for (const double dist : {0.0, 6.0, 15.5}) {
                            for (const int cap : {0, 7}) {
                                GridCornerParams p;
                                p.cellWidth = cell;
                                p.cellHeight = cell / 2 + 3;
                                p.perCell = perCell;
                                p.qualityLevel = q;
                                p.minResponse = minResp;
                                p.minDistance = dist;
                                p.maxCorners = cap;
                                const size_t nCells = gridCellsAcross(static_cast<size_t>(sh.w), p) *
                                                      gridCellsDown(static_cast<size_t>(sh.h), p);
                                std::vector<uint8_t> existing(nCells);
                                for (auto& e : existing) e = static_cast<uint8_t>(r.next() % 4u);
                                const bool useExisting = (cases % 3) == 1;
                                const bool useMask = (cases % 4) == 2;
                                const size_t capacity = (cases % 5 == 3) ? 3u : 2000u;
                                std::vector<CornerCandidate> slots(
                                    gridCornerSlots(static_cast<size_t>(sh.w),
                                                    static_cast<size_t>(sh.h), p));
                                std::vector<Corner> got(capacity);
                                const CornerResult res = gridCorners<uint32_t>(
                                    dx, dy, p,
                                    ResponseMap(ring.data(), static_cast<size_t>(sh.w),
                                                kResponseRingRows, static_cast<size_t>(sh.w)),
                                    slots.data(), slots.size(), got.data(), capacity,
                                    useExisting ? existing.data() : nullptr,
                                    useMask ? mask.constView() : BinMatConstView<uint32_t>{});
                                const std::vector<Corner> want =
                                    oracle(map, sh.w, sh.h, p,
                                           useExisting ? existing.data() : nullptr,
                                           useMask ? &mask : nullptr, capacity);
                                BINCV_CHECK_EQ(res.count, want.size());
                                const size_t n = std::min(res.count, want.size());
                                for (size_t i = 0; i < n; ++i) {
                                    BINCV_CHECK(got[i].x == want[i].x && got[i].y == want[i].y);
                                    BINCV_CHECK(std::memcmp(&got[i].response, &want[i].response,
                                                            sizeof(float)) == 0);
                                }
                                BINCV_CHECK_EQ(res.passes, size_t{1});
                                cornersCompared += n;
                                truncations += res.candidatesTruncated ? 1u : 0u;
                                ++cases;
                            }
                        }
                    }
                }
            }
        }
    }
    std::printf(" %zu parameter sets, %zu corners compared; %zu cut short by the cap\n", cases,
                cornersCompared, truncations);
    BINCV_CHECK(cornersCompared > 20000);
    BINCV_CHECK(truncations > 0);
}

BINCV_TEST(GridCorners, OnDemandGradientMatchesStoredDerivatives) {
    // The form that derives each gradient row as the sweep reaches it, against the one
    // that reads stored derivative planes: the same corners, for every border.
    size_t compared = 0;
    for (const BorderType border : {BORDER_REFLECT_101, BORDER_REPLICATE, BORDER_CONSTANT}) {
        for (const bool fill : {false, true}) {
            if (border != BORDER_CONSTANT && fill) continue;
            const int w = 131, h = 77;
            const BinMat<uint32_t> img = scene(w, h, 0xB0D3 + static_cast<uint64_t>(border));
            TernaryMat<uint32_t> dx(w, h);
            TernaryMat<uint32_t> dy(w, h);
            derivativeX(img, dx, border, fill);
            derivativeY(img, dy, border, fill);
            std::vector<float> ring(static_cast<size_t>(w) * kResponseRingRows);
            const ResponseMap rm(ring.data(), static_cast<size_t>(w), kResponseRingRows,
                                 static_cast<size_t>(w));
            for (const int perCell : {1, 3}) {
                GridCornerParams p;
                p.cellWidth = 12;
                p.cellHeight = 10;
                p.perCell = perCell;
                std::vector<CornerCandidate> slots(gridCornerSlots(static_cast<size_t>(w),
                                                                   static_cast<size_t>(h), p));
                std::vector<Corner> a(1000), b(1000);
                const CornerResult ra = gridCorners<uint32_t>(dx, dy, p, rm, slots.data(),
                                                              slots.size(), a.data(), a.size());
                GradientSource<1, uint32_t> src;
                src.planes[0] = img.constView();
                src.border = border;
                src.borderValue = fill;
                std::vector<uint32_t> rows(gradientRowWords<1, uint32_t>(static_cast<size_t>(w)));
                const CornerResult rb = gridCorners<1, uint32_t>(
                    src, p, rm, rows.data(), rows.size(), slots.data(), slots.size(), b.data(),
                    b.size());
                BINCV_CHECK_EQ(ra.count, rb.count);
                for (size_t i = 0; i < std::min(ra.count, rb.count); ++i) {
                    BINCV_CHECK(a[i].x == b[i].x && a[i].y == b[i].y);
                    BINCV_CHECK(std::memcmp(&a[i].response, &b[i].response, sizeof(float)) == 0);
                }
                compared += ra.count;
            }
        }
    }
    std::printf(" %zu corners compared across borders\n", compared);
    BINCV_CHECK(compared > 200);
}

BINCV_TEST(GridCorners, EmptyFramesAndFullCells) {
    // Nothing to find, and nowhere to put it: an empty image, a mask that admits
    // nothing, and every cell already full all return no corners from one sweep.
    const int w = 64, h = 48;
    std::vector<float> ring(static_cast<size_t>(w) * kResponseRingRows);
    const ResponseMap rm(ring.data(), static_cast<size_t>(w), kResponseRingRows,
                         static_cast<size_t>(w));
    GridCornerParams p;
    std::vector<CornerCandidate> slots(gridCornerSlots(static_cast<size_t>(w),
                                                       static_cast<size_t>(h), p));
    std::vector<Corner> out(100);

    const BinMat<uint32_t> blank(w, h);
    TernaryMat<uint32_t> dx(w, h);
    TernaryMat<uint32_t> dy(w, h);
    derivatives(blank, dx, dy);
    BINCV_CHECK_EQ(gridCorners<uint32_t>(dx, dy, p, rm, slots.data(), slots.size(), out.data(),
                                         out.size())
                       .count,
                   size_t{0});

    const BinMat<uint32_t> img = scene(w, h, 77);
    derivatives(img, dx, dy);
    const CornerResult some = gridCorners<uint32_t>(dx, dy, p, rm, slots.data(), slots.size(),
                                                    out.data(), out.size());
    BINCV_CHECK(some.count > 0);
    const BinMat<uint32_t> noMask(w, h);
    BINCV_CHECK_EQ(gridCorners<uint32_t>(dx, dy, p, rm, slots.data(), slots.size(), out.data(),
                                         out.size(), nullptr, noMask.constView())
                       .count,
                   size_t{0});
    std::vector<uint8_t> full(gridCellsAcross(static_cast<size_t>(w), p) *
                                  gridCellsDown(static_cast<size_t>(h), p),
                              static_cast<uint8_t>(p.perCell));
    BINCV_CHECK_EQ(gridCorners<uint32_t>(dx, dy, p, rm, slots.data(), slots.size(), out.data(),
                                         out.size(), full.data())
                       .count,
                   size_t{0});
    std::printf(" %zu corners on the scene, none blank, masked out or full\n", some.count);
}

BINCV_TEST(GridCorners, SkipFullCellsIsTheMaskOfOpenCells) {
    // `skipFullCells` leaves a full cell out exactly as a mask leaving it out would: the
    // same call with that mask -- AND the caller's, when there is one -- and the switch
    // off must return the same corners, counts and flags. Both forms; cell sizes that do
    // and do not divide a word; occupancy from sparse to nearly full; a cell that keeps
    // one corner and one that keeps three, partly filled.
    struct Shape {
        int w, h, cw, ch;
    };
    const Shape shapes[] = {{160, 120, 32, 32}, {131, 77, 12, 10}, {97, 61, 7, 9},
                            {200, 33, 64, 8}};
    size_t cases = 0, compared = 0, differ = 0;
    Lcg r(0x5C1F);
    for (const Shape& sh : shapes) {
        const BinMat<uint32_t> img = scene(sh.w, sh.h, 0x5EED + static_cast<uint64_t>(sh.w));
        TernaryMat<uint32_t> dx(sh.w, sh.h);
        TernaryMat<uint32_t> dy(sh.w, sh.h);
        derivatives(img, dx, dy);
        std::vector<float> ring(static_cast<size_t>(sh.w) * kResponseRingRows);
        const ResponseMap rm(ring.data(), static_cast<size_t>(sh.w), kResponseRingRows,
                             static_cast<size_t>(sh.w));
        GradientSource<1, uint32_t> src;
        src.planes[0] = img.constView();
        std::vector<uint32_t> rows(gradientRowWords<1, uint32_t>(static_cast<size_t>(sh.w)));
        BinMat<uint32_t> callerMask(sh.w, sh.h);
        for (int y = 0; y < sh.h; ++y)
            for (int x = 0; x < sh.w; ++x)
                if ((x * 7 + y * 3) % 11 != 0) callerMask.set(y, x, 1u);
        for (const int perCell : {1, 3}) {
            for (const uint32_t fullPct : {10u, 50u, 90u}) {
                for (const bool withMask : {false, true}) {
                    GridCornerParams p;
                    p.cellWidth = sh.cw;
                    p.cellHeight = sh.ch;
                    p.perCell = perCell;
                    const size_t across = gridCellsAcross(static_cast<size_t>(sh.w), p);
                    const size_t cells = across * gridCellsDown(static_cast<size_t>(sh.h), p);
                    std::vector<uint8_t> existing(cells);
                    for (uint8_t& e : existing) {
                        const bool full = r.next() % 100u < fullPct;
                        e = static_cast<uint8_t>(full ? perCell
                                                      : static_cast<int>(r.next() % static_cast<uint32_t>(perCell)));
                    }
                    BinMat<uint32_t> openMask(sh.w, sh.h);
                    for (int y = 0; y < sh.h; ++y) {
                        for (int x = 0; x < sh.w; ++x) {
                            const size_t c = static_cast<size_t>(y / sh.ch) * across +
                                             static_cast<size_t>(x / sh.cw);
                            const bool admit = existing[c] < perCell &&
                                               (!withMask || callerMask.at(y, x) != 0u);
                            if (admit) openMask.set(y, x, 1u);
                        }
                    }
                    const BinMatConstView<uint32_t> given =
                        withMask ? callerMask.constView() : BinMatConstView<uint32_t>{};
                    // The policy's runs are its per-pixel answer, row by row, and the
                    // sweep's maximum over them is the mask's.
                    const impl::AdmitOpenCells<uint32_t> open{
                        given, existing.data(), static_cast<size_t>(perCell),
                        static_cast<size_t>(sh.cw), static_cast<size_t>(sh.ch), across,
                        static_cast<size_t>(sh.w)};
                    for (int y = 0; y < sh.h; ++y) {
                        std::vector<uint8_t> fromRuns(static_cast<size_t>(sh.w), 0);
                        open.runs(static_cast<size_t>(y), [&](size_t x0, size_t x1) {
                            for (size_t x = x0; x < x1; ++x) fromRuns[x] = 1;
                        });
                        size_t wrong = 0;
                        for (int x = 0; x < sh.w; ++x)
                            wrong += (fromRuns[static_cast<size_t>(x)] != 0) != open(x, y);
                        differ += wrong != 0 ? 1 : 0;
                    }
                    GoodFeaturesParams g;
                    const auto rowOfDx = [&](int y, float* dst) {
                        cornerMinEigenValRow<uint32_t>(dx.constMagnitude(0), dy.constMagnitude(0),
                                                       dx.constSign(), dy.constSign(), 3, y, dst);
                    };
                    const auto none = [](int, int, float) {};
                    const float maxMask = impl::sweepCornerResponses<uint32_t>(
                        static_cast<size_t>(sh.w), static_cast<size_t>(sh.h), g, rm,
                        openMask.constView(), rowOfDx, none);
                    const float maxRuns = impl::sweepCornerResponses<uint32_t>(
                        static_cast<size_t>(sh.w), static_cast<size_t>(sh.h), g, rm, open,
                        rowOfDx, none);
                    differ += std::memcmp(&maxMask, &maxRuns, sizeof(float)) != 0 ? 1 : 0;
                    std::vector<CornerCandidate> slots(gridCornerSlots(
                        static_cast<size_t>(sh.w), static_cast<size_t>(sh.h), p));
                    std::vector<Corner> want(2000), got(2000);
                    for (const bool onDemand : {false, true}) {
                        const auto run = [&](const GridCornerParams& q, BinMatConstView<uint32_t> m,
                                             std::vector<Corner>& out) {
                            if (onDemand)
                                return gridCorners<1, uint32_t>(src, q, rm, rows.data(), rows.size(),
                                                                slots.data(), slots.size(),
                                                                out.data(), out.size(),
                                                                existing.data(), m);
                            return gridCorners<uint32_t>(dx, dy, q, rm, slots.data(), slots.size(),
                                                         out.data(), out.size(), existing.data(), m);
                        };
                        const CornerResult ra = run(p, openMask.constView(), want);
                        GridCornerParams q = p;
                        q.skipFullCells = true;
                        const CornerResult rb = run(q, given, got);
                        ++cases;
                        bool same = ra.count == rb.count &&
                                    ra.candidatesRanked == rb.candidatesRanked &&
                                    ra.candidatesTruncated == rb.candidatesTruncated;
                        for (size_t i = 0; same && i < ra.count; ++i) {
                            same = want[i].x == got[i].x && want[i].y == got[i].y &&
                                   std::memcmp(&want[i].response, &got[i].response,
                                               sizeof(float)) == 0;
                        }
                        differ += same ? 0 : 1;
                        compared += ra.count;
                    }
                }
            }
        }
    }
    std::printf(" skipFullCells against the open-cell mask: %zu cases, %zu corners, %zu differ\n",
                cases, compared, differ);
    BINCV_CHECK_EQ(differ, size_t{0});
    BINCV_CHECK(compared > 500);
}

BINCV_TEST_MAIN("test_grid_corners")
