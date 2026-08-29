// ===========================================================================
// ops/occupancy.hpp -- spacing new detections against the tracks already live.
//
// TWO ARMS THAT MUST RETURN THE SAME POINTS. One does `O(new x live)` float distance
// tests; the other stamps a disc per live track into a 1-bit frame and reads one bit
// per candidate. E-46 chooses between them on SPEED, which only means anything if the
// answer is identical -- so that is the first and largest thing in this file.
//
// The disagreement to worry about is not a wild one. It is a single boundary pixel:
// a candidate at distance exactly `radius` from a live point, or one where the disc's
// row bound comes from a `sqrt` that rounded the wrong way. `markDisc` takes no square
// root for precisely this reason, and `MaskMatchesFloatDistance` checks every pixel of
// a sweep of sub-pixel centres against the float predicate rather than sampling.
//
// Four claims:
//   1. THE MASK IS THE DISTANCE TEST -- pixel for pixel, no tolerance.
//   2. THE ARMS AGREE -- same kept points, same order, over random populations.
//   3. THE VECTOR ARM IS THE SCALAR ARM -- and can be switched off, so it is checked.
//   4. IT DOES THE JOB IT WAS ADDED FOR -- a candidate on top of a live track is
//      rejected, which is the thing binCV could not do before T5.20.
// ===========================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/core/storage.hpp"
#include "bincv-cpp/ops/occupancy.hpp"
#include "test_util.hpp"

using bincv::Point2f;

namespace {

// A small deterministic LCG. The suite must not depend on <random>'s engine, which
// is implementation-defined in ways that would make a failure unreproducible.
struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(s >> 33);
    }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * (static_cast<float>(next() & 0xFFFFFF) / 16777215.0f);
    }
};

/// The float oracle: exactly what `spaceCandidates` promises, written the obvious way.
std::vector<Point2f> oracle(const std::vector<Point2f>& cand, const std::vector<Point2f>& live,
                            float radius, size_t limit) {
    std::vector<Point2f> kept;
    if (radius < 1.0f) {
        for (size_t i = 0; i < cand.size() && kept.size() < limit; ++i) kept.push_back(cand[i]);
        return kept;
    }
    const float r2 = radius * radius;
    auto near = [&](const std::vector<Point2f>& set, Point2f p) {
        for (const Point2f& q : set) {
            const float dx = q.x - p.x, dy = q.y - p.y;
            if (dx * dx + dy * dy < r2) return true;
        }
        return false;
    };
    for (size_t i = 0; i < cand.size() && kept.size() < limit; ++i) {
        if (near(live, cand[i])) continue;
        if (near(kept, cand[i])) continue;
        kept.push_back(cand[i]);
    }
    return kept;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. THE MASK IS THE DISTANCE TEST
// ---------------------------------------------------------------------------

BINCV_TEST(Occupancy, MaskMatchesFloatDistance) {
    using W = uint32_t;
    constexpr size_t kW = 71, kH = 53;   // neither a multiple of 32: padding is in play
    bincv::BinMat<W> mask(kW, kH);

    size_t checked = 0, disagreements = 0;
    Rng rng;
    // Sub-pixel centres deliberately including .0 and .5, where a boundary pixel is
    // exactly at distance `radius` and the strict `<` decides it.
    const float kCentres[] = {0.0f, 0.25f, 0.5f, 0.75f};
    const float kRadii[] = {1.0f, 2.5f, 4.0f, 7.5f, 13.0f, 32.0f};
    for (float rad : kRadii) {
        for (float fx : kCentres) {
            for (float fy : kCentres) {
                const float cx = static_cast<float>(11 + static_cast<int>(rng.next() % 40)) + fx;
                const float cy = static_cast<float>(7 + static_cast<int>(rng.next() % 30)) + fy;
                bincv::clearOccupancy<W>(mask.view());
                bincv::markDisc<W>(mask.view(), cx, cy, rad);

                const float r2 = rad * rad;
                for (size_t y = 0; y < kH; ++y) {
                    for (size_t x = 0; x < kW; ++x) {
                        const float dx = static_cast<float>(x) - cx;
                        const float dy = static_cast<float>(y) - cy;
                        const bool want = (dx * dx + dy * dy) < r2;
                        const bool got = bincv::occupied<W>(mask.constView(),
                                                            static_cast<long long>(x),
                                                            static_cast<long long>(y));
                        ++checked;
                        if (want != got) ++disagreements;
                    }
                }
                // PADDING BITS STAY ZERO (CLAUDE.md). The disc is clipped to `width`,
                // so no word past the last pixel may be touched -- and a set padding
                // bit is invisible to `occupied` while corrupting any word-wise
                // reduction over the same frame.
                const size_t lastWord = (kW - 1) / 32;
                const W padMask = static_cast<W>(~static_cast<W>(0)) << (kW % 32);
                for (size_t y = 0; y < kH; ++y) {
                    BINCV_CHECK((mask.constView().row(y)[lastWord] & padMask) == 0);
                }
            }
        }
    }
    std::printf("  %zu pixels against the float predicate: %zu disagreements\n", checked,
                disagreements);
    BINCV_CHECK(checked > 300000);
    BINCV_CHECK_EQ(disagreements, size_t{0});
}

BINCV_TEST(Occupancy, DiscIsClippedNotWrapped) {
    using W = uint32_t;
    constexpr size_t kW = 40, kH = 24;
    bincv::BinMat<W> mask(kW, kH);
    // A centre outside the frame on every side in turn. A disc that wrapped would set
    // pixels on the opposite edge; one that ran off the end would corrupt the next row.
    const Point2f kCentres[] = {{-5.0f, 12.0f}, {44.0f, 12.0f}, {20.0f, -6.0f}, {20.0f, 30.0f}};
    for (const Point2f& c : kCentres) {
        bincv::clearOccupancy<W>(mask.view());
        bincv::markDisc<W>(mask.view(), c.x, c.y, 9.0f);
        const float r2 = 81.0f;
        for (size_t y = 0; y < kH; ++y) {
            for (size_t x = 0; x < kW; ++x) {
                const float dx = static_cast<float>(x) - c.x;
                const float dy = static_cast<float>(y) - c.y;
                const bool want = (dx * dx + dy * dy) < r2;
                BINCV_CHECK(bincv::occupied<W>(mask.constView(), static_cast<long long>(x),
                                               static_cast<long long>(y)) == want);
            }
        }
    }
    // A centre far enough out that nothing is set at all -- the sweep must terminate
    // rather than walk the whole frame or, worse, not terminate.
    bincv::clearOccupancy<W>(mask.view());
    bincv::markDisc<W>(mask.view(), -100.0f, -100.0f, 9.0f);
    size_t set = 0;
    for (size_t y = 0; y < kH; ++y) {
        for (size_t x = 0; x < kW; ++x) {
            if (bincv::occupied<W>(mask.constView(), static_cast<long long>(x),
                                   static_cast<long long>(y))) ++set;
        }
    }
    BINCV_CHECK_EQ(set, size_t{0});
}

// ---------------------------------------------------------------------------
// 2. THE ARMS AGREE
// ---------------------------------------------------------------------------

BINCV_TEST(Occupancy, ArmsAgreeExactly) {
    using W = uint32_t;
    constexpr size_t kW = 640, kH = 480;
    bincv::BinMat<W> mask(kW, kH);

    Rng rng;
    size_t populations = 0, totalKept = 0;
    for (int trial = 0; trial < 24; ++trial) {
        const float radius = (trial % 4 == 0) ? 8.0f : (trial % 4 == 1) ? 16.0f
                             : (trial % 4 == 2) ? 32.0f : 3.0f;
        const size_t liveCount = static_cast<size_t>(10 + (trial * 17) % 190);
        const size_t candCount = static_cast<size_t>(20 + (trial * 29) % 280);

        // Live points are SUB-PIXEL, as tracked positions are. Candidates are integer,
        // as a detector's are -- which is the contract the mask arm is exact under.
        std::vector<Point2f> live;
        for (size_t i = 0; i < liveCount; ++i) {
            live.push_back(Point2f{rng.uniform(0.0f, 639.0f), rng.uniform(0.0f, 479.0f)});
        }
        std::vector<Point2f> cand;
        for (size_t i = 0; i < candCount; ++i) {
            cand.push_back(Point2f{std::floor(rng.uniform(0.0f, 639.0f)),
                                   std::floor(rng.uniform(0.0f, 479.0f))});
        }
        const size_t limit = static_cast<size_t>(30 + (trial * 13) % 120);

        const std::vector<Point2f> want = oracle(cand, live, radius, limit);

        std::vector<Point2f> a = cand;
        const size_t na = bincv::spaceCandidates(a.data(), a.size(), live.data(), live.size(),
                                                 radius, limit);

        bincv::clearOccupancy<W>(mask.view());
        bincv::markOccupied<W>(mask.view(), live.data(), live.size(), radius);
        std::vector<Point2f> b = cand;
        const size_t nb = bincv::spaceCandidatesMasked<W>(mask.view(), b.data(), b.size(),
                                                          radius, limit);

        BINCV_CHECK_EQ(na, want.size());
        BINCV_CHECK_EQ(nb, want.size());
        for (size_t i = 0; i < want.size(); ++i) {
            BINCV_CHECK(a[i].x == want[i].x && a[i].y == want[i].y);
            BINCV_CHECK(b[i].x == want[i].x && b[i].y == want[i].y);
        }
        ++populations;
        totalKept += want.size();
    }
    std::printf("  %zu populations, %zu kept points: both arms match the float oracle\n",
                populations, totalKept);
    BINCV_CHECK_EQ(populations, size_t{24});
    BINCV_CHECK(totalKept > 200);
}

// ---------------------------------------------------------------------------
// 3. THE VECTOR ARM IS THE SCALAR ARM
// ---------------------------------------------------------------------------

BINCV_TEST(Occupancy, VectorDistanceArmMatchesScalar) {
    Rng rng;
    // Counts straddling the vector width in both directions -- 8 on AVX2, 4 on NEON --
    // so the tail loop is exercised at every remainder, which is where a lane-count
    // error hides.
    size_t compared = 0;
    for (size_t liveCount = 0; liveCount <= 20; ++liveCount) {
        std::vector<Point2f> live;
        for (size_t i = 0; i < liveCount; ++i) {
            live.push_back(Point2f{rng.uniform(0.0f, 100.0f), rng.uniform(0.0f, 100.0f)});
        }
        std::vector<Point2f> cand;
        for (size_t i = 0; i < 40; ++i) {
            cand.push_back(Point2f{std::floor(rng.uniform(0.0f, 100.0f)),
                                   std::floor(rng.uniform(0.0f, 100.0f))});
        }
        for (float radius : {2.0f, 9.0f, 25.0f}) {
            std::vector<Point2f> vec = cand;
            bincv::impl::spacingSimdEnabled() = true;
            const size_t nv = bincv::spaceCandidates(vec.data(), vec.size(), live.data(),
                                                     live.size(), radius, vec.size());
            std::vector<Point2f> sca = cand;
            bincv::impl::spacingSimdEnabled() = false;
            const size_t ns = bincv::spaceCandidates(sca.data(), sca.size(), live.data(),
                                                     live.size(), radius, sca.size());
            bincv::impl::spacingSimdEnabled() = true;

            BINCV_CHECK_EQ(nv, ns);
            for (size_t i = 0; i < ns; ++i) {
                BINCV_CHECK(vec[i].x == sca[i].x && vec[i].y == sca[i].y);
            }
            ++compared;
        }
    }
    std::printf("  %zu (liveCount, radius) pairs: vector arm == scalar arm\n", compared);
    BINCV_CHECK_EQ(compared, size_t{63});
}

// ---------------------------------------------------------------------------
// 4. THE JOB IT WAS ADDED FOR
// ---------------------------------------------------------------------------

BINCV_TEST(Occupancy, RejectsCandidatesOnTopOfLiveTracks) {
    using W = uint32_t;
    bincv::BinMat<W> mask(128, 128);

    // Three live tracks and five candidates: one exactly on a track, one just inside
    // the radius, one just outside, and two far away. This is the case binCV could not
    // express before T5.20 -- ops/corner.hpp spaces candidates against each other and
    // has never seen `live`.
    const std::vector<Point2f> live = {{20.0f, 20.0f}, {60.4f, 33.7f}, {100.0f, 90.0f}};
    const float radius = 10.0f;
    std::vector<Point2f> cand = {
        {20.0f, 20.0f},    // exactly on a track            -> rejected
        {66.0f, 33.0f},    // 5.6 px from the second track  -> rejected
        {71.0f, 34.0f},    // 10.6 px from it               -> KEPT
        {10.0f, 110.0f},   // far from everything           -> KEPT
        {14.0f, 112.0f},   // 4.5 px from the one just kept -> rejected (self-spacing)
    };

    std::vector<Point2f> a = cand;
    const size_t na = bincv::spaceCandidates(a.data(), a.size(), live.data(), live.size(),
                                             radius, 100);
    BINCV_CHECK_EQ(na, size_t{2});
    BINCV_CHECK(a[0].x == 71.0f && a[0].y == 34.0f);
    BINCV_CHECK(a[1].x == 10.0f && a[1].y == 110.0f);

    bincv::clearOccupancy<W>(mask.view());
    bincv::markOccupied<W>(mask.view(), live.data(), live.size(), radius);
    std::vector<Point2f> b = cand;
    const size_t nb = bincv::spaceCandidatesMasked<W>(mask.view(), b.data(), b.size(), radius, 100);
    BINCV_CHECK_EQ(nb, size_t{2});
    BINCV_CHECK(b[0].x == 71.0f && b[0].y == 34.0f);
    BINCV_CHECK(b[1].x == 10.0f && b[1].y == 110.0f);

    // `limit` is the free-slot count, and it stops the filter rather than the scan.
    std::vector<Point2f> c = cand;
    BINCV_CHECK_EQ(bincv::spaceCandidates(c.data(), c.size(), live.data(), live.size(), radius, 1),
                   size_t{1});

    // radius < 1 disables the filter, as GoodFeaturesParams::minDistance documents --
    // a caller forwarding that field must not get a different rule at 0.5 here.
    std::vector<Point2f> d = cand;
    BINCV_CHECK_EQ(bincv::spaceCandidates(d.data(), d.size(), live.data(), live.size(), 0.5f, 100),
                   size_t{5});

    // Degenerate but legal: no live points at all is the FIRST detection of a session.
    // Three survive, not four -- with `live` gone, (71,34) and (14,112) are still
    // rejected by the SELF-spacing against (66,33) and (10,110), which are now kept.
    std::vector<Point2f> e = cand;
    BINCV_CHECK_EQ(bincv::spaceCandidates(e.data(), e.size(), nullptr, 0, radius, 100), size_t{3});
}

BINCV_TEST_MAIN("test_occupancy")
