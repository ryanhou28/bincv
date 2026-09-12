// ===========================================================================
// WHERE ONE STREAMING DETECTION'S TIME ACTUALLY SITS -- the profile that has to
// exist before any optimisation of it is allowed to start.
//
// `goodFeaturesToTrackStreaming` has four stages: the response row sweep, the
// NMS scan with its top-K heap, the rank (std::sort over the ranked pool), and
// the greedy spacing filter. The named suspect has always been the POOL -- a
// binarized response takes few distinct values, ties are everywhere, and most
// interior pixels are raw 3x3 maxima -- but which stage pays for the pool has
// never been measured. The spacing filter is O(examined * kept) and the sort is
// O(pool log pool); which one owns the time decides what gets optimised.
//
// METHOD: prefix arms, timed whole, split by difference -- nothing instrumented
// inside a loop. Arms 2 and 3 REPLICATE the shipped kernel's prefix (mask-free
// path, copied verbatim); the replication is held honest by comparing its pool
// counts against the shipped call's own CornerResult every run, and the split
// is only printed when they agree.
//
// CONTENT: realframe.bin -- a real binarized reference frame, the same content
// rule fast_bitplane_benchmark.cpp states: synthetic noise gets the candidate
// density wrong, and the candidate density is the whole question here.
//
// Core-only: no OpenCV anywhere, so the reference device's default build runs
// this as-is.
// ===========================================================================

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/corner.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/quantMat.hpp"
#include "measure_util.hpp"

using W = uint32_t;

namespace {

int gW = 0, gH = 0;

std::vector<uint8_t> loadRealFrame() {
    // BINCV_REALFRAME_PATH is the build system's absolute path to the frame --
    // the same macro every realframe consumer here uses, so this benchmark runs
    // from any working directory and cannot drift onto a different file.
    std::vector<uint8_t> px;
    FILE* f = std::fopen(BINCV_REALFRAME_PATH, "rb");
    if (!f) return px;
    uint32_t fw = 0, fh = 0;
    if (std::fread(&fw, 4, 1, f) != 1 || std::fread(&fh, 4, 1, f) != 1) {
        std::fclose(f);
        return px;
    }
    px.resize(static_cast<size_t>(fw) * fh);
    if (std::fread(px.data(), 1, px.size(), f) != px.size()) px.clear();
    std::fclose(f);
    gW = static_cast<int>(fw);
    gH = static_cast<int>(fh);
    return px;
}

} // namespace

int main() {
    const std::vector<uint8_t> frame = loadRealFrame();
    if (frame.empty()) {
        std::fprintf(stderr, "realframe.bin unreadable -- refusing a synthetic fallback;\n"
                             "the candidate density IS the question this profile answers.\n");
        return 1;
    }
    const size_t w = static_cast<size_t>(gW), h = static_cast<size_t>(gH);

    bincv::BinMat<W> bin(gW, gH);
    bincv::packBits<bincv::PackRule::NonZero>(frame.data(), w, h, w, bin.view());
    bincv::SignedQuantMat<1, W> dx(gW, gH), dy(gW, gH);
    bincv::derivativeX(bin, dx);
    bincv::derivativeY(bin, dy);
    const bincv::BinMatConstView<W> magX = dx.constMagnitude(0), magY = dy.constMagnitude(0);
    const bincv::BinMatConstView<W> signX = dx.constSign(), signY = dy.constSign();

    bincv::GoodFeaturesParams params;   // the reference frontend's values, verbatim
    std::vector<float> ring(bincv::kResponseRingRows * w);
    const bincv::ResponseMap ringMap{ring.data(), w, bincv::kResponseRingRows, w};
    const size_t capacity = 20000;      // frontend_sequence.cpp's own pool size
    std::vector<bincv::Corner> corners(capacity);

    // The shipped call once, for the counts every arm is checked against.
    const bincv::CornerResult ref = bincv::goodFeaturesToTrackStreaming<W>(
        magX, magY, signX, signY, params, ringMap, corners.data(), capacity);

    // Raw-maxima census, outside any timed arm: how many pixels ever reach the
    // heap, which is the pool the sort and the spacing filter are blamed for.
    size_t rawMaxima = 0;
    {
        std::vector<float> full(w * h);
        const bincv::ResponseMap fullMap{full.data(), w, h, w};
        bincv::cornerMinEigenVal<W>(dx, dy, params.blockSize, fullMap);
        for (size_t y = 1; y + 1 < h; ++y) {
            for (size_t x = 1; x + 1 < w; ++x) {
                const float v = full[y * w + x];
                if (!(v > 0.0f)) continue;   // the zero plateau never enters the heap:
                                             // the running threshold kills it at row 0
                bool isMax = true;
                for (int dxo = -1; dxo <= 1 && isMax; ++dxo) {
                    const size_t c = x + static_cast<size_t>(dxo);
                    if (full[(y - 1) * w + c] > v || full[y * w + c] > v ||
                        full[(y + 1) * w + c] > v)
                        isMax = false;
                }
                if (isMax) ++rawMaxima;
            }
        }
    }

    // ---- arm bodies ------------------------------------------------------
    // Stage 1: the response sweep exactly as the streaming kernel runs it.
    const auto sweep = [&]() {
        for (int y = 0; y < gH; ++y) {
            bincv::cornerMinEigenValRow<W>(magX, magY, signX, signY, params.blockSize, y,
                                           ring.data() +
                                               (static_cast<size_t>(y) %
                                                bincv::kResponseRingRows) * w);
        }
    };

    // Stages 1-2: the kernel's prefix, replicated (mask-free path). Returns
    // {retained, maxDiscarded} so stage 3 can continue from it and the counts
    // can be checked against the shipped call.
    const auto sweepNmsHeap = [&](size_t& retainedOut, float& maxDiscardedOut) {
        const int width = gW, height = gH;
        float* first = ring.data();
        bincv::cornerMinEigenValRow<W>(magX, magY, signX, signY, params.blockSize, 0, first);
        float runningMax = first[0];
        for (int x = 1; x < width; ++x)
            if (first[static_cast<size_t>(x)] > runningMax)
                runningMax = first[static_cast<size_t>(x)];
        size_t retained = 0;
        float maxDiscarded = -1.0f;
        for (int y = 1; y < height; ++y) {
            float* cur = ring.data() + (static_cast<size_t>(y) % bincv::kResponseRingRows) * w;
            bincv::cornerMinEigenValRow<W>(magX, magY, signX, signY, params.blockSize, y, cur);
            for (int x = 0; x < width; ++x)
                if (cur[static_cast<size_t>(x)] > runningMax)
                    runningMax = cur[static_cast<size_t>(x)];
            if (y < 2) continue;
            const float running =
                static_cast<float>(static_cast<double>(runningMax) * params.qualityLevel);
            const int cy = y - 1;
            const float* above =
                ring.data() + (static_cast<size_t>(cy - 1) % bincv::kResponseRingRows) * w;
            const float* mid =
                ring.data() + (static_cast<size_t>(cy) % bincv::kResponseRingRows) * w;
            const float* below = cur;
            for (int x = 1; x + 1 < width; ++x) {
                const float val = mid[static_cast<size_t>(x)];
                if (!(val > running)) continue;
                bool isMax = true;
                for (int dxo = -1; dxo <= 1 && isMax; ++dxo) {
                    const size_t c = static_cast<size_t>(x + dxo);
                    if (above[c] > val || mid[c] > val || below[c] > val) isMax = false;
                }
                if (!isMax) continue;
                bincv::Corner candidate;
                candidate.x = x;
                candidate.y = cy;
                candidate.response = val;
                if (retained < capacity) {
                    corners[retained++] = candidate;
                    std::push_heap(corners.begin(),
                                   corners.begin() + static_cast<ptrdiff_t>(retained),
                                   bincv::impl::CornerStronger());
                } else if (bincv::impl::CornerStronger()(candidate, corners[0])) {
                    if (corners[0].response > maxDiscarded) maxDiscarded = corners[0].response;
                    std::pop_heap(corners.begin(),
                                  corners.begin() + static_cast<ptrdiff_t>(retained),
                                  bincv::impl::CornerStronger());
                    corners[retained - 1] = candidate;
                    std::push_heap(corners.begin(),
                                   corners.begin() + static_cast<ptrdiff_t>(retained),
                                   bincv::impl::CornerStronger());
                } else if (val > maxDiscarded) {
                    maxDiscarded = val;
                }
            }
        }
        retainedOut = retained;
        maxDiscardedOut = maxDiscarded;
        // The final threshold, formed as the kernel forms it -- needed by stage 3.
        ring[0] = static_cast<float>(static_cast<double>(runningMax) * params.qualityLevel);
    };

    size_t armRetained = 0, armRanked = 0;
    std::vector<measure::Bench> bs = {
        {"1    response row sweep", [&](int) { sweep(); measure::g_sink += static_cast<size_t>(ring[8]); }},
        {"1-2  + NMS scan + top-K heap",
         [&](int) {
             size_t retained;
             float md;
             sweepNmsHeap(retained, md);
             armRetained = retained;
             measure::g_sink += retained;
         }},
        {"1-3  + threshold filter + sort",
         [&](int) {
             size_t retained;
             float md;
             sweepNmsHeap(retained, md);
             const float threshold = ring[0];
             size_t ranked = 0;
             for (size_t i = 0; i < retained; ++i)
                 if (corners[i].response > threshold) corners[ranked++] = corners[i];
             std::sort(corners.begin(), corners.begin() + static_cast<ptrdiff_t>(ranked),
                       bincv::impl::CornerStronger());
             armRanked = ranked;
             measure::g_sink += ranked;
         }},
        {"1-4  the shipped kernel (adds spacing)",
         [&](int) {
             const bincv::CornerResult r = bincv::goodFeaturesToTrackStreaming<W>(
                 magX, magY, signX, signY, params, ringMap, corners.data(), capacity);
             measure::g_sink += r.count;
         }},
    };
    const auto t = measure::measureInterleaved(bs, 7, 60.0);

    // The replication check BEFORE the numbers: a split against a drifted
    // replica would name the wrong stage with full confidence.
    const bool consistent = armRetained >= ref.candidatesRanked && armRanked == ref.candidatesRanked;

    std::printf("=== one streaming detection, split by stage ===\n");
    std::printf(" %dx%d real frame, blockSize %d, quality %.2f, minDistance %.1f,"
                " capacity %zu\n\n",
                gW, gH, params.blockSize, params.qualityLevel, params.minDistance, capacity);
    std::printf(" the pool: %zu raw 3x3 maxima (%.1f%% of pixels) -> %zu retained ->"
                " %zu ranked -> %zu kept%s\n\n",
                rawMaxima, 100.0 * static_cast<double>(rawMaxima) /
                               (static_cast<double>(w) * static_cast<double>(h)),
                armRetained, ref.candidatesRanked, ref.count,
                ref.candidatesTruncated ? " (TRUNCATED)" : "");
    for (size_t i = 0; i < bs.size(); ++i) {
        std::printf(" %-42s %10.3f ms  spread %.0f%%\n", bs[i].name.c_str(),
                    t[i].medianNs / 1e6, t[i].spreadPct());
    }
    if (!consistent) {
        std::printf("\n *** REPLICA DISAGREES WITH THE SHIPPED KERNEL (retained %zu,"
                    " ranked %zu vs %zu) -- the splits below would lie; fix the replica"
                    " first. ***\n",
                    armRetained, armRanked, ref.candidatesRanked);
        return 1;
    }
    std::printf("\n DERIVED SPLITS (by difference)\n");
    std::printf(" response sweep     : %10.3f ms (%5.1f%%)\n", t[0].medianNs / 1e6,
                100.0 * t[0].medianNs / t[3].medianNs);
    std::printf(" NMS + top-K heap   : %10.3f ms (%5.1f%%)\n",
                (t[1].medianNs - t[0].medianNs) / 1e6,
                100.0 * (t[1].medianNs - t[0].medianNs) / t[3].medianNs);
    std::printf(" threshold + sort   : %10.3f ms (%5.1f%%)\n",
                (t[2].medianNs - t[1].medianNs) / 1e6,
                100.0 * (t[2].medianNs - t[1].medianNs) / t[3].medianNs);
    std::printf(" spacing filter     : %10.3f ms (%5.1f%%)\n",
                (t[3].medianNs - t[2].medianNs) / 1e6,
                100.0 * (t[3].medianNs - t[2].medianNs) / t[3].medianNs);
    std::printf(" whole detection    : %10.3f ms\n", t[3].medianNs / 1e6);
    std::printf("\n memory: ring %zu B + corner buffer %zu B, both caller-owned and\n"
                " constant across every arm -- no arm may quietly grow them.\n",
                ring.size() * sizeof(float), corners.size() * sizeof(bincv::Corner));
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
