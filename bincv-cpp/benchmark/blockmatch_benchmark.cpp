// ===========================================================================
// earlier work -- THE SPEED AXIS OF ROUTE (a) AGAINST ROUTE (b).
//
// that measurement’s band B requires yield-per-millisecond for both routes, so this supplies
// the millisecond. Interleaved round-robin; the reference device closes it.
//
// PRE-WRITTEN COST MODEL. Route (a) scores (2R+1)^2 windows per point per level,
// each one XOR + popcount per word. Route (b) runs up to `maxIterations` window
// passes per point per level, each 20N^2 popcounts per word plus a covariance and
// a float solve. At R = 2 that is 25 scores against up to 20 iterations, so route
// (a) should be COMPARABLE or slightly cheaper per level -- and its per-word body
// is far cheaper (1 popcount against 20 at N=1). Route (a) also builds no
// derivative, so its BUILD stage is pyrDown alone. The prediction is that route
// (a) wins clearly on build and modestly on track at R = 2, and loses as R grows,
// since its cost is O(R^2) and route (b)'s is not.
// ===========================================================================

#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "bincv-cpp/ops/blockMatch.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"
#include "measure_util.hpp"

using bincv::Point2f;
using W = uint32_t;

namespace {

constexpr int kLevels = 4;

struct Frames {
    std::vector<bincv::BinMat<W>> prev, next;
    std::vector<bincv::TernaryMat<W>> dx, dy;
    std::vector<bincv::LKLevel<W>> lkLevels;
    std::vector<bincv::BlockMatchLevel<W>> bmLevels;

    Frames(int w, int h) {
        int cw = w, ch = h;
        for (int i = 0; i < kLevels; ++i) {
            prev.emplace_back(cw, ch);
            next.emplace_back(cw, ch);
            dx.emplace_back(cw, ch);
            dy.emplace_back(cw, ch);
            cw = static_cast<int>(bincv::pyrDownWidth(static_cast<size_t>(cw)));
            ch = static_cast<int>(bincv::pyrDownHeight(static_cast<size_t>(ch)));
        }
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                prev[0].set(y, x, ((x * 7 + y * 13) % 29 == 0 || (x + y) % 37 == 0) ? 1u : 0u);
                next[0].set(y, x,
                            (((x - 1) * 7 + y * 13) % 29 == 0 || (x - 1 + y) % 37 == 0) ? 1u : 0u);
            }
        }
        buildBlock();
        buildLK();
        for (int i = 0; i < kLevels; ++i) {
            lkLevels.push_back(bincv::lkLevel(prev[static_cast<size_t>(i)],
                                              next[static_cast<size_t>(i)],
                                              dx[static_cast<size_t>(i)],
                                              dy[static_cast<size_t>(i)]));
            bmLevels.push_back(bincv::blockMatchLevel(prev[static_cast<size_t>(i)],
                                                      next[static_cast<size_t>(i)]));
        }
    }
    /// Route (a)'s build: pyrDown only. NO derivative.
    void buildBlock() {
        for (size_t i = 1; i < prev.size(); ++i) {
            bincv::pyrDownBox<1, 1, W>(prev[i - 1], prev[i]);
            bincv::pyrDownBox<1, 1, W>(next[i - 1], next[i]);
        }
    }
    /// Route (b) additionally needs both derivative ladders.
    void buildLK() {
        for (size_t i = 0; i < prev.size(); ++i) {
            bincv::derivativeX(prev[i], dx[i]);
            bincv::derivativeY(prev[i], dy[i]);
        }
    }
    size_t blockBytes() const {
        size_t words = 0;
        for (size_t i = 0; i < prev.size(); ++i) {
            words += prev[i].sizeInWords() + next[i].sizeInWords();
        }
        return words * sizeof(W);
    }
    size_t lkBytes() const {
        size_t words = 0;
        for (size_t i = 0; i < prev.size(); ++i) {
            words += prev[i].sizeInWords() + next[i].sizeInWords() + dx[i].sizeInWords() +
                     dy[i].sizeInWords();
        }
        return words * sizeof(W);
    }
};

std::vector<Point2f> gridPoints(int w, int h, int step, int margin) {
    std::vector<Point2f> pts;
    for (int y = margin; y < h - margin; y += step) {
        for (int x = margin; x < w - margin; x += step) {
            pts.push_back(Point2f{static_cast<float>(x), static_cast<float>(y)});
        }
    }
    return pts;
}

} // namespace

int main() {
    const int w = 640, h = 480;
    Frames fr(w, h);
    const std::vector<Point2f> pts = gridPoints(w, h, 40, 40);
    std::vector<Point2f> out(pts.size());
    std::vector<uint8_t> status(pts.size());

    std::printf("=== route (a) block matching vs route (b) hybrid LK ===\n");
    std::printf("640x480, %zu keypoints, 31x31 window, 4 levels\n\n", pts.size());

    auto lkTrack = [&](int) {
        bincv::LKParams p;
        bincv::calcOpticalFlowPyrLK<W>(fr.lkLevels.data(), fr.lkLevels.size(), pts.data(),
                                       out.data(), status.data(), nullptr, pts.size(), p);
    };
    auto blockTrack = [&](int radius) {
        return [&, radius](int) {
            bincv::BlockMatchParams p;
            p.searchRadius = radius;
            p.subPixel = true;
            bincv::calcOpticalFlowBlockMatch<W>(fr.bmLevels.data(), fr.bmLevels.size(), pts.data(),
                                               out.data(), status.data(), pts.size(), p);
        };
    };

    std::vector<measure::Bench> track = {
        {"(b) LK", lkTrack},
        {"(a) block R=2", blockTrack(2)},
        {"(a) block R=4", blockTrack(4)},
    };
    std::vector<measure::Bench> build = {
        {"(b) build: pyrDown + 2 derivative ladders", [&](int) { fr.buildBlock(); fr.buildLK(); }},
        {"(a) build: pyrDown only", [&](int) { fr.buildBlock(); }},
    };

    const auto bt = measure::measureInterleaved(build, 7, 60.0);
    const auto tt = measure::measureInterleaved(track, 7, 60.0);

    std::printf(" BUILD\n");
    for (size_t i = 0; i < build.size(); ++i) {
        std::printf(" %-44s %9.1f us spread %.0f%%\n", build[i].name.c_str(),
                    bt[i].medianNs / 1000.0, bt[i].spreadPct());
    }
    std::printf(" route (b) build / route (a) build = %.2fx\n",
                bt[0].medianNs / bt[1].medianNs);

    std::printf("\n TRACK\n");
    for (size_t i = 0; i < track.size(); ++i) {
        std::printf(" %-44s %9.1f us %.2fx LK spread %.0f%%\n", track[i].name.c_str(),
                    tt[i].medianNs / 1000.0, tt[i].medianNs / tt[0].medianNs, tt[i].spreadPct());
    }

    std::printf("\n FOOTPRINT (pyramid stage, 640x480, 4 levels)\n");
    std::printf(" route (b): %zu B route (a): %zu B ratio %.2fx\n", fr.lkBytes(),
                fr.blockBytes(),
                static_cast<double>(fr.lkBytes()) / static_cast<double>(fr.blockBytes()));
    return 0;
}
