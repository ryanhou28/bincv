// ===========================================================================
// E-48 -- CAN THE RESIDUAL binCV ALREADY COMPUTES TELL A GOOD TRACK FROM A LOST ONE?
//
// X-94 found the tracker returning a mean endpoint error of 54.7 px on a 24 px
// translation while reporting `status = 1` for every point. Two candidate fixes were
// registered, and they do not cost the same:
//
//   (a) a bidirectional consistency check -- needs the NEXT frame's derivative, two more
//       planes per pyramid level, which is the footprint trade `LKLevel` declines by
//       construction;
//   (b) a reject on the residual AT CONVERGENCE -- and `err` IS that residual.
//       `windowMeanAbsDiff` over the warped window, already computed at level 0 for
//       every tracked point whenever the caller passes an `err` array. **Free.**
//
// THIS PROGRAM DECIDES WHETHER (b) IS EVEN VIABLE, BEFORE ANY RULE IS WRITTEN. If `err`
// does not separate the two populations there is nothing to threshold and the question
// is only about (a); if it does, the cheap arm wins on the memory rule before a single
// timing is taken.
//
// It reports, per translation, the SEPARATION rather than a mean: the err distribution
// of points that landed within 1 px of truth against those that did not, and what a
// threshold chosen to keep 99% of the good ones would do to the bad ones. A mean would
// hide exactly the overlap that decides this.
//
// Usage: lk_loss_diagnostic
// ===========================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"

using bincv::Point2f;
using W = uint32_t;

namespace {

constexpr int kWidth = 640, kHeight = 480;

double field(double x, double y) {
    return std::sin(x / 7.3 + 0.4) * std::cos(y / 5.1) + 0.6 * std::sin((x + y) / 11.7) +
           0.5 * std::cos((x - 2.0 * y) / 9.3) + 0.4 * std::sin(x / 3.1) * std::sin(y / 3.7);
}

double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = static_cast<size_t>(q * static_cast<double>(v.size() - 1) + 0.5);
    return v[i];
}

}  // namespace

int main() {
    std::printf("=== E-48: does `err` separate a good track from a lost one? ===\n");
    std::printf("%dx%d, ladder 1/2/2/2, 31x31 window, seal_params.yaml\n", kWidth, kHeight);
    std::printf("`err` is windowMeanAbsDiff over the warped window: 0 = identical, "
                "0.5 = uncorrelated\n\n");

    std::printf("%6s %7s %7s %7s | %-27s | %-27s | %s\n", "motion", "pts", "good", "bad",
                "err of GOOD (p50/p90/p99)", "err of BAD  (p1/p10/p50)", "threshold @99% good");
    std::printf("%s\n", "--------------------------------------------------------------------"
                        "-------------------------------------------------------");

    for (const double t : {2.0, 4.0, 8.0, 16.0, 24.0, 32.0}) {
        const double tx = t, ty = t * 0.6;

        bincv::Pyramid<W, 1, 2, 2, 2> prev(kWidth, kHeight), next(kWidth, kHeight);
        bincv::SignedQuantMat<1, W> dx0(kWidth, kHeight), dy0(kWidth, kHeight);
        bincv::SignedQuantMat<2, W> dx1(320, 240), dy1(320, 240), dx2(160, 120), dy2(160, 120),
            dx3(80, 60), dy3(80, 60);
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                prev.level<0>().set(y, x, field(x, y) > 0.0 ? 1u : 0u);
                next.level<0>().set(y, x, field(static_cast<double>(x) - tx,
                                                static_cast<double>(y) - ty) > 0.0 ? 1u : 0u);
            }
        }
        prev.build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
        next.build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
        bincv::derivativeX(prev.level<0>(), dx0); bincv::derivativeY(prev.level<0>(), dy0);
        bincv::derivativeX(prev.level<1>(), dx1); bincv::derivativeY(prev.level<1>(), dy1);
        bincv::derivativeX(prev.level<2>(), dx2); bincv::derivativeY(prev.level<2>(), dy2);
        bincv::derivativeX(prev.level<3>(), dx3); bincv::derivativeY(prev.level<3>(), dy3);
        bincv::LKLevels<W, 1, 2, 2, 2> levels;
        levels.get<0>() = bincv::lkLevel<1>(prev.level<0>(), next.level<0>(), dx0, dy0);
        levels.get<1>() = bincv::lkLevel<2>(prev.level<1>(), next.level<1>(), dx1, dy1);
        levels.get<2>() = bincv::lkLevel<2>(prev.level<2>(), next.level<2>(), dx2, dy2);
        levels.get<3>() = bincv::lkLevel<2>(prev.level<3>(), next.level<3>(), dx3, dy3);

        std::vector<Point2f> pts;
        for (int y = 48; y < kHeight - 48; y += 16) {
            for (int x = 48; x < kWidth - 48; x += 16) {
                pts.push_back(Point2f{static_cast<float>(x), static_cast<float>(y)});
            }
        }
        const size_t n = pts.size();
        std::vector<Point2f> out(n);
        std::vector<uint8_t> status(n);
        std::vector<float> err(n);
        bincv::calcOpticalFlowPyrLK(levels, pts.data(), out.data(), status.data(), err.data(), n,
                                    bincv::LKParams{});

        std::vector<double> good, bad;
        for (size_t i = 0; i < n; ++i) {
            if (!status[i]) continue;
            const double ex = static_cast<double>(out[i].x) - (static_cast<double>(pts[i].x) + tx);
            const double ey = static_cast<double>(out[i].y) - (static_cast<double>(pts[i].y) + ty);
            (std::sqrt(ex * ex + ey * ey) <= 1.0 ? good : bad).push_back(
                static_cast<double>(err[i]));
        }

        // The threshold that keeps 99% of the good points, and what it costs the bad
        // ones. This is the only number that decides arm (b): a rule that removes the
        // failures by also removing the tracks is not a fix, it is a shorter sequence.
        const double thr = good.empty() ? 0.0 : quantile(good, 0.99);
        size_t badRemoved = 0;
        for (double e : bad) if (e > thr) ++badRemoved;
        const double pct = bad.empty() ? 0.0
                                       : 100.0 * static_cast<double>(badRemoved) /
                                             static_cast<double>(bad.size());

        std::printf("%5.0fpx %7zu %7zu %7zu | %7.4f %7.4f %7.4f     | %7.4f %7.4f %7.4f     |"
                    " %.4f -> removes %.1f%% of bad\n",
                    std::sqrt(tx * tx + ty * ty), n, good.size(), bad.size(),
                    quantile(good, 0.50), quantile(good, 0.90), quantile(good, 0.99),
                    quantile(bad, 0.01), quantile(bad, 0.10), quantile(bad, 0.50), thr, pct);
    }

    std::printf("\nGOOD = tracked and within 1 px of ground truth; BAD = tracked and not.\n"
                "If the GOOD p99 column sits below the BAD p10 column, the two populations\n"
                "are separable and arm (b) costs nothing but a comparison. If they overlap,\n"
                "no threshold on `err` can do this job and only arm (a) is left.\n");
    return 0;
}
