// Is LK memory-bound or compute-bound? The 8x smaller footprint only buys speed if
// it does. Two probes:
//
// (a) PER-POINT COST vs POINT COUNT. More points touch more of the frame. If the
// kernel were bandwidth- or cache-bound the per-point cost would RISE as the
// working set grows past L1 and then L2; if it is compute-bound it stays flat.
//
// (b) FRAME SIZE at a fixed point count. Same compute, more spread-out data.
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "measure_util.hpp"

using W = uint32_t;

namespace {
void fill(bincv::BinMat<W>& m, int w, int h, int dx) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            m.set(y, x, (((x - dx) * 7 + y * 13) % 29 == 0 || (x - dx + y) % 37 == 0) ? 1u : 0u);
}
} // namespace

int main() {
    std::printf("=== Is LK memory-bound? N=1, one level, 31x31 window ===\n\n");

    // (a) point count sweep at 640x480. Frame at 1 bit = 38.4 KB; Pi L1D is 32 KB.
    {
        const int w = 640, h = 480;
        bincv::BinMat<W> p(w, h), n(w, h);
        fill(p, w, h, 0); fill(n, w, h, 1);
        bincv::TernaryMat<W> dx(w, h), dy(w, h);
        bincv::derivativeX(p, dx); bincv::derivativeY(p, dy);
        const auto lv = bincv::lkLevel(p, n, dx, dy);
        bincv::LKParams lk; lk.maxIterations = 4; lk.epsilon = 0.0f;

        std::printf(" (a) point count sweep, 640x480 (frame = %.1f KB at 1 bit)\n",
                    static_cast<double>(w) * h / 8.0 / 1024.0);
        std::printf(" %8s %12s %14s\n", "points", "ms", "us per point");
        for (int step : {80, 56, 40, 28, 20, 14}) {
            std::vector<bincv::Point2f> pts;
            for (int y = 40; y < h - 40; y += step)
                for (int x = 40; x < w - 40; x += step)
                    pts.push_back(bincv::Point2f{static_cast<float>(x), static_cast<float>(y)});
            std::vector<bincv::Point2f> out(pts.size());
            std::vector<uint8_t> st(pts.size());
            std::vector<measure::Bench> b = {{"x", [&](int) {
                bincv::calcOpticalFlowPyrLK<W>(&lv, 1, pts.data(), out.data(), st.data(),
                                               nullptr, pts.size(), lk); }}};
            const auto t = measure::measureInterleaved(b, 5, 50.0);
            std::printf(" %8zu %12.4f %14.4f\n", pts.size(), t[0].medianNs / 1e6,
                        t[0].medianNs / 1e3 / static_cast<double>(pts.size()));
        }
    }

    // (b) frame size at a FIXED point count -- same compute, data more spread out.
    {
        std::printf("\n (b) frame size at 140 points (same compute, wider spread)\n");
        std::printf(" %12s %10s %12s %14s\n", "frame", "KB @1bit", "ms", "us per point");
        for (auto wh : {std::pair<int,int>{320,240}, {640,480}, {1280,960}, {1920,1440}}) {
            const int w = wh.first, h = wh.second;
            bincv::BinMat<W> p(w, h), n(w, h);
            fill(p, w, h, 0); fill(n, w, h, 1);
            bincv::TernaryMat<W> dx(w, h), dy(w, h);
            bincv::derivativeX(p, dx); bincv::derivativeY(p, dy);
            const auto lv = bincv::lkLevel(p, n, dx, dy);
            bincv::LKParams lk; lk.maxIterations = 4; lk.epsilon = 0.0f;
            std::vector<bincv::Point2f> pts;
            const int nx = 14, ny = 10;
            for (int iy = 0; iy < ny; ++iy)
                for (int ix = 0; ix < nx; ++ix)
                    pts.push_back(bincv::Point2f{
                        static_cast<float>(40 + ix * (w - 80) / nx),
                        static_cast<float>(40 + iy * (h - 80) / ny)});
            std::vector<bincv::Point2f> out(pts.size());
            std::vector<uint8_t> st(pts.size());
            std::vector<measure::Bench> b = {{"x", [&](int) {
                bincv::calcOpticalFlowPyrLK<W>(&lv, 1, pts.data(), out.data(), st.data(),
                                               nullptr, pts.size(), lk); }}};
            const auto t = measure::measureInterleaved(b, 5, 50.0);
            char name[24]; std::snprintf(name, sizeof(name), "%dx%d", w, h);
            std::printf(" %12s %10.1f %12.4f %14.4f\n", name,
                        static_cast<double>(w) * h / 8.0 / 1024.0, t[0].medianNs / 1e6,
                        t[0].medianNs / 1e3 / static_cast<double>(pts.size()));
        }
    }
    std::printf("\n FLAT us/point => compute-bound, and the 8x footprint buys nothing here.\n"
                " RISING => memory-bound, and it should.\n");
    return 0;
}
