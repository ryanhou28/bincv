// ===========================================================================
// earlier work -- SPACING NEW DETECTIONS AGAINST LIVE TRACKS: WHICH ARM?
//
// A VIO frontend detects to top up, so every fresh corner must be rejected if it lands
// on a track already being followed. Two shapes, and the DECISION RULE WAS FIXED BEFORE
// EITHER EXISTED :
//
// (a) EXHAUSTIVE -- O(new x live) float distance tests. Vectorized eight-wide on
// AVX2, four-wide on NEON. COSTS NO MEMORY.
// (b) OCCUPANCY MASK -- stamp a disc of `radius` per live track into a 1-bit frame,
// then one bit test per candidate. COSTS ONE 1-BIT FRAME: 38 400 B at 640x480.
//
// The rule: the mask becomes the recommended path only if it is FASTER at the
// frontend's own operating point on BOTH architectures. Parity is a loss -- CLAUDE.md
// settles unclaimed speed/footprint conflicts in favor of memory, and 38 400 B has to
// buy something.
//
// The pre-registered prediction, recorded so that agreeing with it is not evidence:
// the mask LOSES at small candidate counts and wins past new ~ pi*r^2/WordBits ~ 100.
//
// TWO CONTROLS, BOTH REQUIRED BY CLAUDE.md's BENCHMARKING RULES:
// - the exhaustive arm is timed with the vector path forced OFF as well as on, so
// the vector arm is shown to be RUNNING rather than assumed to be;
// - `live = 4` is below the vector width in both ISAs, so its vector/scalar ratio
// must come back at ~1.00x. If it does not, the "vector" arm is not what it says.
//
// Parameters are the reference's, not chosen: radius 32 is HybVIO's
// relativeMaskRadius 0.0667 x min(640,480), and maxTracks is 200.
//
// Usage: spacing_benchmark
// ===========================================================================

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/core/storage.hpp"
#include "bincv-cpp/ops/occupancy.hpp"

using Clock = std::chrono::steady_clock;
using bincv::Point2f;
using W = uint32_t;

namespace {

constexpr size_t kWidth = 640, kHeight = 480;
constexpr float kRadius = 32.0f;   // relativeMaskRadius 0.0667 x 480
constexpr int kRounds = 7;

struct Rng {
    uint64_t s = 0x243F6A8885A308D3ull;
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(s >> 33);
    }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * (static_cast<float>(next() & 0xFFFFFF) / 16777215.0f);
    }
};

double minOf(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }

}  // namespace

int main() {
    std::printf("=== spacing new detections against live tracks ===\n");
    std::printf("%zux%zu, radius %.0f, %d interleaved rounds, minimum reported\n",
                kWidth, kHeight, static_cast<double>(kRadius), kRounds);
#if defined(BINCV_OCCUPANCY_AVX2)
    std::printf("distance arm: AVX2, eight live points per register\n");
#elif defined(BINCV_OCCUPANCY_NEON)
    std::printf("distance arm: NEON, four live points per register\n");
#else
    std::printf("distance arm: portable scalar (no vector arm compiled)\n");
#endif
    std::printf("\nmemory: exhaustive 0 B | mask %zu B (one 1-bit frame)\n\n",
                (kWidth / 32) * kHeight * sizeof(W));

    bincv::BinMat<W> mask(kWidth, kHeight);

    std::printf("%6s %6s | %11s %11s %8s | %11s %11s %11s | %8s %6s\n",
                "live", "new", "exh(vec)", "exh(scalar)", "vec/sca",
                "mask clear", "mask mark", "mask TOTAL", "exh/mask", "kept");
    std::printf("%s\n", "-------------------------------------------------------------------"
                        "-------------------------------------");

    const size_t kLive[] = {4, 50, 100, 200};
    // Out to 5 000 deliberately. The frontend never sees more than a few hundred
    // candidates, but the pre-registered rule asks WHERE the mask crosses over, and an
    // extrapolated crossover is not a measured one -- 1 000 and 5 000 bracket it on
    // both architectures, which is what the rule needs. It stops there rather than at
    // 20 000 for a physical reason: the reference device SOFT-THROTTLED partway through
    // the 20 000 rows, `run_on_pi.sh` flagged the whole run INVALID, and a longer sweep
    // that cannot be measured on the device it has to be measured on is not a sweep.
    const size_t kNew[] = {50, 200, 1000, 5000};

    for (size_t liveCount : kLive) {
        for (size_t newCount : kNew) {
            Rng rng;
            std::vector<Point2f> live;
            for (size_t i = 0; i < liveCount; ++i) {
                live.push_back(Point2f{rng.uniform(0.0f, static_cast<float>(kWidth - 1)),
                                       rng.uniform(0.0f, static_cast<float>(kHeight - 1))});
            }
            std::vector<Point2f> cand;
            for (size_t i = 0; i < newCount; ++i) {
                cand.push_back(
                    Point2f{static_cast<float>(rng.next() % kWidth),
                            static_cast<float>(rng.next() % kHeight)});
            }
            // UNCAPPED. `limit` is the frontend's free-slot count, and capping it here
            // would stop the scan early and time the CAP rather than the filter -- at
            // live = 200 the cap is zero and the first version of this benchmark
            // measured a filter that exits immediately, reporting `kept = 1` for four
            // whole rows. The frontend's own operating point, cap included, is the
            // separate section at the bottom.
            const size_t limit = newCount;

            std::vector<double> tv, ts, tc, tm, tl, tsel;
            size_t keptV = 0, keptS = 0, keptM = 0;
            std::vector<Point2f> work(newCount);

            // Reps scale with the work so every cell gets a usable clock, and the
            // three arms are interleaved WITHIN a round so a frequency excursion or
            // another process on the machine lands on all of them alike.
            const int reps = static_cast<int>(
                std::max<size_t>(20, 400000 / (newCount * (liveCount + 20))));

            for (int r = 0; r < kRounds; ++r) {
                bincv::impl::spacingSimdEnabled() = true;
                auto t0 = Clock::now();
                for (int rep = 0; rep < reps; ++rep) {
                    work = cand;
                    keptV = bincv::spaceCandidates(work.data(), work.size(), live.data(),
                                                   live.size(), kRadius, limit);
                }
                tv.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count() /
                             reps);

                bincv::impl::spacingSimdEnabled() = false;
                t0 = Clock::now();
                for (int rep = 0; rep < reps; ++rep) {
                    work = cand;
                    keptS = bincv::spaceCandidates(work.data(), work.size(), live.data(),
                                                   live.size(), kRadius, limit);
                }
                ts.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count() /
                             reps);
                bincv::impl::spacingSimdEnabled() = true;

                // The mask arm's three parts, timed apart. `clear` is a fixed cost per
                // detect that does not depend on either count, and reporting it inside
                // one total would hide which term actually moves.
                t0 = Clock::now();
                for (int rep = 0; rep < reps; ++rep) bincv::clearOccupancy<W>(mask.view());
                tc.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count() /
                             reps);

                t0 = Clock::now();
                for (int rep = 0; rep < reps; ++rep) {
                    bincv::clearOccupancy<W>(mask.view());
                    bincv::markOccupied<W>(mask.view(), live.data(), live.size(), kRadius);
                }
                const double clearPlusMark =
                    std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / reps;
                tm.push_back(clearPlusMark - tc.back());

                t0 = Clock::now();
                for (int rep = 0; rep < reps; ++rep) {
                    bincv::clearOccupancy<W>(mask.view());
                    bincv::markOccupied<W>(mask.view(), live.data(), live.size(), kRadius);
                    work = cand;
                    keptM = bincv::spaceCandidatesMasked<W>(mask.view(), work.data(), work.size(),
                                                            kRadius, limit);
                }
                const double total =
                    std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / reps;
                tl.push_back(total);
                // WITHIN the round, so the difference is of two numbers taken under the
                // same conditions. Taking the minimum of each part independently and
                // subtracting produced a NEGATIVE column in the first run of this
                // benchmark -- the parts' minima came from different rounds.
                tsel.push_back(total - clearPlusMark);
            }

            const double V = minOf(tv), S = minOf(ts), C = minOf(tc), M = minOf(tm),
                         L = minOf(tl);
            // The SELECT phase is deliberately NOT reported as its own column. It is a
            // difference of two large numbers -- stamping dominates both -- and at
            // live >= 100 the noise in `mark` exceeds it, so the derived column came
            // out NEGATIVE. A number that can be negative is not a measurement.
            (void)tsel;
            // A benchmark whose arms disagree is measuring two different operations.
            const char* agree = (keptV == keptS && keptV == keptM) ? "" : " <-- ARMS DISAGREE";
            std::printf("%6zu %6zu | %11.0f %11.0f %8.2f | %11.0f %11.0f %11.0f | "
                        "%8.2f %6zu%s\n",
                        liveCount, newCount, V, S, S / V, C, M, L, L > 0 ? V / L : 0.0,
                        keptV, agree);
        }
    }

    // -----------------------------------------------------------------------
    // THE GATE CONTROL, AND THE FIRST VERSION OF IT WAS WRONG.
    //
    // `live = 4` is below the vector width in both ISAs, so those rows LOOK like the
    // control -- and they read 1.5x to 3.4x, not 1.00x. The reason is that
    // `spaceCandidates` makes TWO scans, and the second one is over the candidates
    // already KEPT, which grows into the hundreds and vectorizes perfectly. A control
    // has to starve BOTH scans, so this one caps `limit` at 1 as well: at most one
    // kept point, four live points, nothing for either scan to vectorize.
    //
    // If this reads far from 1.00x, the vec/sca column above is not measuring the
    // vector arm and every ratio in this table is against the wrong denominator.
    // -----------------------------------------------------------------------
    {
        Rng rng;
        std::vector<Point2f> live;
        for (size_t i = 0; i < 4; ++i) {
            live.push_back(Point2f{rng.uniform(0.0f, 639.0f), rng.uniform(0.0f, 479.0f)});
        }
        std::vector<Point2f> cand;
        for (size_t i = 0; i < 1000; ++i) {
            cand.push_back(Point2f{static_cast<float>(rng.next() % kWidth),
                                   static_cast<float>(rng.next() % kHeight)});
        }
        std::vector<double> tv, ts;
        std::vector<Point2f> work(cand.size());
        constexpr int kReps = 2000;
        for (int r = 0; r < kRounds; ++r) {
            bincv::impl::spacingSimdEnabled() = true;
            auto t0 = Clock::now();
            for (int rep = 0; rep < kReps; ++rep) {
                work = cand;
                bincv::spaceCandidates(work.data(), work.size(), live.data(), live.size(),
                                       kRadius, 1);
            }
            tv.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count() /
                         kReps);
            bincv::impl::spacingSimdEnabled() = false;
            t0 = Clock::now();
            for (int rep = 0; rep < kReps; ++rep) {
                work = cand;
                bincv::spaceCandidates(work.data(), work.size(), live.data(), live.size(),
                                       kRadius, 1);
            }
            ts.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count() /
                         kReps);
            bincv::impl::spacingSimdEnabled() = true;
        }
        std::printf("\nGATE CONTROL live=4, limit=1 (neither scan reaches the vector width)\n");
        std::printf(" vector %8.0f ns scalar %8.0f ns ratio %5.2fx (must be ~1.00)\n",
                    minOf(tv), minOf(ts), minOf(ts) / minOf(tv));
    }

    // -----------------------------------------------------------------------
    // THE FRONTEND'S OWN OPERATING POINT, cap included.
    //
    // examples/vio_frontend.cpp targets 200 live tracks and detects when the count
    // falls below a hysteresis low-water mark, so a detect sees roughly 120 live
    // tracks, a few hundred candidates and about 80 free slots. This is the row the
    // decision rule is actually about; everything above is the shape of the curve.
    // -----------------------------------------------------------------------
    {
        Rng rng;
        std::vector<Point2f> live;
        for (size_t i = 0; i < 120; ++i) {
            live.push_back(Point2f{rng.uniform(0.0f, 639.0f), rng.uniform(0.0f, 479.0f)});
        }
        std::vector<Point2f> cand;
        for (size_t i = 0; i < 300; ++i) {
            cand.push_back(Point2f{static_cast<float>(rng.next() % kWidth),
                                   static_cast<float>(rng.next() % kHeight)});
        }
        std::vector<double> ta, tb;
        std::vector<Point2f> work(cand.size());
        size_t keptA = 0, keptB = 0;
        constexpr int kReps = 3000;
        for (int r = 0; r < kRounds; ++r) {
            auto t0 = Clock::now();
            for (int rep = 0; rep < kReps; ++rep) {
                work = cand;
                keptA = bincv::spaceCandidates(work.data(), work.size(), live.data(), live.size(),
                                               kRadius, 80);
            }
            ta.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count() /
                         kReps);
            t0 = Clock::now();
            for (int rep = 0; rep < kReps; ++rep) {
                bincv::clearOccupancy<W>(mask.view());
                bincv::markOccupied<W>(mask.view(), live.data(), live.size(), kRadius);
                work = cand;
                keptB = bincv::spaceCandidatesMasked<W>(mask.view(), work.data(), work.size(),
                                                        kRadius, 80);
            }
            tb.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count() /
                         kReps);
        }
        const double A = minOf(ta), B = minOf(tb);
        std::printf("\nOPERATING POINT live=120, new=300, limit=80, radius=32\n");
        std::printf(" (a) exhaustive %9.0f ns 0 B\n", A);
        std::printf(" (b) mask %9.0f ns %zu B\n", B, (kWidth / 32) * kHeight * sizeof(W));
        std::printf(" exhaustive is %.2fx %s (kept %zu / %zu)%s\n", (B > A) ? B / A : A / B,
                    (B > A) ? "FASTER" : "SLOWER", keptA, keptB,
                    (keptA == keptB) ? "" : " <-- ARMS DISAGREE");
    }
    return 0;
}
