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
// METHOD: arms that differ by exactly one stage, timed whole, split by
// difference -- nothing instrumented inside a loop.
//
// THE ARMS RUN THE LIBRARY'S OWN CODE, NOT A COPY OF IT. An earlier version of
// this file replicated the kernel's prefix verbatim so it could stop half way,
// and the replica went stale the first time the kernel changed: it kept the
// scalar running maximum and the per-candidate heap after the kernel had
// dropped both, so the arm that was supposed to be a PREFIX of the shipped call
// ran SLOWER than the whole of it and the spacing filter came out at MINUS 8%
// of a detection. A profile that can print a negative stage is not measuring
// the shipped kernel.
//
// Three of the four stages are library functions that can be timed directly --
// the response sweep is `cornerMinEigenValRow`, the rank is `std::sort` under
// `impl::CornerStronger`, the spacing filter is `impl::spacingFilter` -- so the
// candidate pool is rebuilt into a scratch array and the sort and spacing arms
// differ from the rebuild arm by exactly one stage each. What is left over from
// the whole call is the suppression scan and its top-K heap, which is the one
// stage that has no entry point of its own.
//
// The pool the sort and spacing arms are given is checked against the shipped
// call's own `candidatesRanked` before anything is timed.
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

    bincv::GoodFeaturesParams params;   // the reference pipeline's values, verbatim
    std::vector<float> ring(bincv::kResponseRingRows * w);
    const bincv::ResponseMap ringMap{ring.data(), w, bincv::kResponseRingRows, w};
    const size_t capacity = 20000;      // feature_tracking_sequence.cpp's own pool size
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

    // ---- the candidate pool the ranking stages are given ------------------
    // The kernel hands `std::sort` the raw 3x3 maxima that clear the FINAL
    // threshold, in raster order -- that is what its suppression sweep appends
    // and what its threshold pass compacts. Rebuilt here from the frame map so
    // the sort and spacing arms are given the same corners in the same order,
    // and checked against the shipped call's own count below.
    std::vector<bincv::Corner> pool;
    {
        std::vector<float> full(w * h);
        const bincv::ResponseMap fullMap{full.data(), w, h, w};
        bincv::cornerMinEigenVal<W>(dx, dy, params.blockSize, fullMap);
        float maxVal = 0.0f;
        for (size_t i = 0; i < full.size(); ++i) maxVal = full[i] > maxVal ? full[i] : maxVal;
        const float threshold =
            static_cast<float>(static_cast<double>(maxVal) * params.qualityLevel);
        for (size_t y = 1; y + 1 < h; ++y) {
            for (size_t x = 1; x + 1 < w; ++x) {
                const float v = full[y * w + x];
                if (!(v > threshold)) continue;
                bool isMax = true;
                for (int dxo = -1; dxo <= 1 && isMax; ++dxo) {
                    const size_t c = x + static_cast<size_t>(dxo);
                    if (full[(y - 1) * w + c] > v || full[y * w + c] > v ||
                        full[(y + 1) * w + c] > v)
                        isMax = false;
                }
                if (!isMax) continue;
                bincv::Corner candidate;
                candidate.x = static_cast<int>(x);
                candidate.y = static_cast<int>(y);
                candidate.response = v;
                pool.push_back(candidate);
            }
        }
    }
    const size_t poolSize = pool.size();
    const size_t limit = (params.maxCorners > 0)
                             ? std::min(capacity, static_cast<size_t>(params.maxCorners))
                             : capacity;

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

    // The three ranking arms share this rebuild, so every difference between
    // them is one stage and nothing else. The copy is real work and it is
    // charged to all three identically.
    const auto refill = [&]() {
        std::copy(pool.begin(), pool.end(), corners.begin());
    };

    size_t armKept = 0;
    std::vector<measure::Bench> bs = {
        {"A    response row sweep",
         [&](int) { sweep(); measure::g_sink += static_cast<size_t>(ring[8]); }},
        {"B    the shipped kernel, whole",
         [&](int) {
             const bincv::CornerResult r = bincv::goodFeaturesToTrackStreaming<W>(
                 magX, magY, signX, signY, params, ringMap, corners.data(), capacity);
             measure::g_sink += r.count;
         }},
        {"C    rebuild the candidate pool",
         [&](int) { refill(); measure::g_sink += static_cast<size_t>(corners[0].x); }},
        {"C+   + rank",
         [&](int) {
             refill();
             bincv::impl::rankCandidates(corners.data(), poolSize, capacity, true);
             measure::g_sink += static_cast<size_t>(corners[0].x);
         }},
        {"C++  + spacing filter",
         [&](int) {
             refill();
             bincv::impl::rankCandidates(corners.data(), poolSize, capacity, true);
             armKept = bincv::impl::spacingFilter(corners.data(), poolSize, params.minDistance,
                                                  limit);
             measure::g_sink += armKept;
         }},
        // The rank's fallback arm, on the same pool. Two things it is here for:
        // the counting passes and `std::sort` must produce the SAME order, and a
        // ratio near 1.00x between these two rows would mean the counting arm is
        // not the one running.
        {"C+   + rank, std::sort arm",
         [&](int) {
             refill();
             std::sort(corners.begin(), corners.begin() + static_cast<ptrdiff_t>(poolSize),
                       bincv::impl::CornerStronger());
             measure::g_sink += static_cast<size_t>(corners[0].x);
         }},
    };
    const auto t = measure::measureInterleaved(bs, 7, 60.0);

    // The pool check BEFORE the numbers: a split taken over a pool that is not
    // the kernel's would name the wrong stage with full confidence. The two rank
    // arms are checked against each other too, since one of them is timed
    // against the other below.
    bool ranksAgree = true;
    {
        std::vector<bincv::Corner> viaCounting(pool), viaSort(pool);
        bincv::impl::rankCandidates(viaCounting.data(), poolSize, viaCounting.size(), true);
        std::sort(viaSort.begin(), viaSort.end(), bincv::impl::CornerStronger());
        for (size_t i = 0; i < poolSize; ++i) {
            if (viaCounting[i].x != viaSort[i].x || viaCounting[i].y != viaSort[i].y) {
                ranksAgree = false;
                break;
            }
        }
    }
    const bool consistent =
        poolSize == ref.candidatesRanked && armKept == ref.count && ranksAgree;

    const double whole = t[1].medianNs;
    const double sweepNs = t[0].medianNs;
    const double sortNs = t[3].medianNs - t[2].medianNs;
    const double sortStdNs = t[5].medianNs - t[2].medianNs;
    const double spacingNs = t[4].medianNs - t[3].medianNs;
    const double nmsNs = whole - sweepNs - sortNs - spacingNs;

    std::printf("=== one streaming detection, split by stage ===\n");
    std::printf(" %dx%d real frame, blockSize %d, quality %.2f, minDistance %.1f,"
                " capacity %zu\n\n",
                gW, gH, params.blockSize, params.qualityLevel, params.minDistance, capacity);
    std::printf(" the pool: %zu raw 3x3 maxima (%.1f%% of pixels) -> %zu ranked -> %zu kept%s\n\n",
                rawMaxima, 100.0 * static_cast<double>(rawMaxima) /
                               (static_cast<double>(w) * static_cast<double>(h)),
                ref.candidatesRanked, ref.count,
                ref.candidatesTruncated ? " (TRUNCATED)" : "");
    for (size_t i = 0; i < bs.size(); ++i) {
        std::printf(" %-42s %10.3f ms  spread %.0f%%\n", bs[i].name.c_str(),
                    t[i].medianNs / 1e6, t[i].spreadPct());
    }
    if (!consistent) {
        std::printf("\n *** THE REBUILT POOL IS NOT THE KERNEL'S (ranked %zu vs %zu,"
                    " kept %zu vs %zu) -- the splits below would lie. ***\n",
                    poolSize, ref.candidatesRanked, armKept, ref.count);
        return 1;
    }
    std::printf("\n DERIVED SPLITS\n");
    std::printf(" response sweep     : %10.3f ms (%5.1f%%)   A\n", sweepNs / 1e6,
                100.0 * sweepNs / whole);
    std::printf(" NMS + top-K heap   : %10.3f ms (%5.1f%%)   B - A - sort - spacing\n",
                nmsNs / 1e6, 100.0 * nmsNs / whole);
    std::printf(" rank               : %10.3f ms (%5.1f%%)   C+ - C\n", sortNs / 1e6,
                100.0 * sortNs / whole);
    std::printf("   the std::sort arm: %10.3f ms           %.2fx -- 1.00x would mean the\n"
                "                       counting arm is not the one running\n",
                sortStdNs / 1e6, sortStdNs / sortNs);
    std::printf(" spacing filter     : %10.3f ms (%5.1f%%)   C++ - C+\n", spacingNs / 1e6,
                100.0 * spacingNs / whole);
    std::printf(" whole detection    : %10.3f ms\n", whole / 1e6);
    std::printf("\n A stage at or below zero means this machine cannot resolve it --"
                " read the\n spreads above before reading the percentages.\n");
    std::printf("\n memory: ring %zu B + corner buffer %zu B, both caller-owned and\n"
                " constant across every arm -- no arm may quietly grow them.\n",
                ring.size() * sizeof(float), corners.size() * sizeof(bincv::Corner));
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
