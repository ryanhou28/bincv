// T3.7 -- goodFeaturesToTrack against OPENCV. This is the tier 2 DENOMINATOR.
//
// WHY THIS FILE EXISTS BESIDE corner_benchmark.cpp
//
// corner_benchmark.cpp is binCV against binCV: the sliding response map against a
// covariance call per position (X-18). That is an internal question and no OpenCV
// denominator applies to it. But T3.7 is **API TIER 2** -- it has a direct cv::
// counterpart -- and CLAUDE.md's rule for that case is not optional:
//
//     Denominator: OpenCV doing the *same semantic operation on the same binary
//     content stored as CV_8U* -- that is what a user does today without binCV.
//     Report peak working set, not per-buffer ratios.
//
// T3.7 shipped without one. The only footprint figure recorded for it was binCV
// against binCV ("identical for all three variants"), so the 1 228 800 B float
// response map -- eight times the four one-bit derivative planes, and per
// ops/corner.hpp "the operation's whole memory cost" -- had never been weighed
// against what the byte-per-pixel pipeline costs for the same answer. That is the
// gap this file closes, and it is the memory column that closes it: on speed the
// answer is not flattering, and it is printed anyway.
//
// THE THREE ROWS, AND WHICH ONE IS THE DENOMINATOR
//
//   binCV            pack -> derivativeX/derivativeY -> goodFeaturesToTrack.
//                    Four one-bit derivative planes, one float response map, a
//                    caller-owned corner array, and NO other buffer.
//
//   OpenCV binarized THE DENOMINATOR. The same semantics, expressed in stock
//                    OpenCV over the same content stored as CV_8U: two filter2D
//                    calls with the reference's [-1, 0, 1] tap into CV_32F, the
//                    three product planes, a boxFilter SUM over the block, the
//                    min-eigenvalue per pixel, then gftt.cpp's selection --
//                    minMaxLoc, THRESH_TOZERO, dilate, the `val != 0 && val ==
//                    tmp[x]` scan, greaterThanPtr, and the greedy spacing filter.
//                    This is what the reference pipeline does
//                    (`gftt_corner_derivative_type: BINARIZED`), so it is the
//                    operation a user runs today without binCV, and every ratio
//                    below is taken against it.
//
//                    Its box filter uses BORDER_CONSTANT, not the reference's
//                    BORDER_REPLICATE, because a SUM with a zero fill is exactly
//                    T3.6's clipped window (D-13) -- the two sides then compute the
//                    same numbers and the agreement column below means something.
//                    ops/corner.hpp records the REPLICATE deviation separately.
//
//   OpenCV Sobel     stock `cv::goodFeaturesToTrack` on the same CV_8U image,
//                    blockSize 3, useHarrisDetector false. NOT the same numerics --
//                    it runs a 3x3 Sobel where the reference runs the binarized tap
//                    -- so it is NOT the denominator and no correctness claim is
//                    made against it. It is here because it is the call a reader
//                    reaches for, and leaving it out would invite the assumption
//                    that it is faster or slower than it is.
//
// THE MEMORY COLUMN IS THE POINT, AND IT IS AN ACCOUNTING, NOT A GUESS
//
// Peak working set is the live buffers ONE call needs, per CLAUDE.md. For binCV
// and for the binarized OpenCV row every one of those buffers is allocated in this
// file, so the total is read off the buffers themselves. For the stock
// `cv::goodFeaturesToTrack` row it cannot be: that function allocates its
// intermediates internally. Its accounting is stated from what the call is known
// to materialize -- `cornerMinEigenVal`'s Dx, Dy (CV_32F) and cov (CV_32FC3), then
// gftt's own eig and the dilate destination -- and is LABELLED as an accounting
// rather than a reading, which is why it is not the denominator either.
//
// Validity: measure_util.hpp's protocol -- volatile sink, calibrated batches,
// interleaved variants, spread printed beside the median. Four rotating frames.
// The two selections are compared BEFORE anything is timed and the agreement is
// printed, because a tier 2 operation owes a reader the size of the gap it is
// trading against.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "bench_util.hpp"
#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/corner.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "measure_util.hpp"

namespace {

using Word = uint32_t;  // D-14's default, and what a VIO frontend would run

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kInputs = 4;
constexpr int kBlockSize = 3;              // seal_params.yaml's gftt_block_size
constexpr int kMaxCorners = 200;           // gftt_max_corners
constexpr double kQualityLevel = 0.01;     // gftt_quality_level
constexpr double kMinDistance = 33.33333333333;  // gftt_min_distance

/// @brief gftt.cpp's comparator, from SEAL/opencv_internal/include/gftt.hpp, with
///        its comment. Ties go to the LATER raster position, because a larger
///        address in a contiguous CV_32F map is a later pixel.
struct GreaterThanPtr {
    bool operator()(const float* a, const float* b) const

    // Ensure a fully deterministic result of the sort
    { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
};

/// @brief A frame with real corner structure rather than salt-and-pepper noise --
///        the same generator corner_benchmark.cpp uses, so the two entries in
///        EXPERIMENTS.md describe the same content.
void makeFrame(uint64_t seed, bincv::BinMat<Word>& bin, cv::Mat& bytes) {
    uint64_t state = seed;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const unsigned block = static_cast<unsigned>((x / 37) + (y / 29)) % 2u;
            const unsigned diag = (x > y + 40) ? 1u : 0u;
            unsigned v = block ^ diag;
            if ((measure::nextRandom(state) & 63ULL) == 0ULL) v ^= 1u;  // sparse texture
            bin.set(y, x, v);
            bytes.at<uint8_t>(y, x) = static_cast<uint8_t>(v);
        }
    }
}

/// @brief Every buffer the OpenCV binarized pipeline needs, allocated once so the
///        timed body reuses them and the footprint is a reading rather than a
///        guess. Named individually because the memory table lists them.
struct CvBuffers {
    cv::Mat dx, dy;        // CV_32F derivative planes
    cv::Mat xx, yy, xy;    // CV_32F products, box-filtered in place
    cv::Mat eig, tmp;      // CV_32F response map and the dilate destination
    std::vector<const float*> candidates;

    CvBuffers()
        : dx(kHeight, kWidth, CV_32F), dy(kHeight, kWidth, CV_32F),
          xx(kHeight, kWidth, CV_32F), yy(kHeight, kWidth, CV_32F),
          xy(kHeight, kWidth, CV_32F), eig(kHeight, kWidth, CV_32F),
          tmp(kHeight, kWidth, CV_32F) {
        // The candidate list is the reference's unbounded std::vector<const
        // float*>. Reserving the worst case up front keeps allocation out of the
        // timed region -- and the worst case IS (w-2)(h-2), because every interior
        // pixel of an all-equal map is a 3x3 maximum. binCV pays 0 for this; the
        // reservation is charged to the OpenCV row's working set below.
        candidates.reserve(static_cast<size_t>(kWidth - 2) * static_cast<size_t>(kHeight - 2));
    }

    size_t bytes() const {
        return dx.total() * dx.elemSize() + dy.total() * dy.elemSize() +
               xx.total() * xx.elemSize() + yy.total() * yy.elemSize() +
               xy.total() * xy.elemSize() + eig.total() * eig.elemSize() +
               tmp.total() * tmp.elemSize() +
               candidates.capacity() * sizeof(const float*);
    }
};

/// @brief The reference pipeline expressed in stock OpenCV: binarized derivatives,
///        a box-filter SUM over the block, the min eigenvalue, then gftt.cpp's
///        selection. THIS IS THE DENOMINATOR.
void openCvBinarized(const cv::Mat& src, CvBuffers& b, std::vector<cv::Point>& out) {
    static const cv::Mat kx = (cv::Mat_<float>(1, 3) << -1.0f, 0.0f, 1.0f);
    static const cv::Mat ky = (cv::Mat_<float>(3, 1) << -1.0f, 0.0f, 1.0f);
    // BORDER_REFLECT_101 is D-19's choice and filter2D's default.
    cv::filter2D(src, b.dx, CV_32F, kx, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
    cv::filter2D(src, b.dy, CV_32F, ky, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);

    cv::multiply(b.dx, b.dx, b.xx);
    cv::multiply(b.dy, b.dy, b.yy);
    cv::multiply(b.dx, b.dy, b.xy);

    // normalize = false, so a SUM over the block. BORDER_CONSTANT (zero fill) is
    // T3.6's clipped window exactly, for a sum.
    const cv::Size k(kBlockSize, kBlockSize);
    const int border = cv::BORDER_CONSTANT | cv::BORDER_ISOLATED;
    cv::boxFilter(b.xx, b.xx, CV_32F, k, cv::Point(-1, -1), false, border);
    cv::boxFilter(b.yy, b.yy, CV_32F, k, cv::Point(-1, -1), false, border);
    cv::boxFilter(b.xy, b.xy, CV_32F, k, cv::Point(-1, -1), false, border);

    for (int y = 0; y < kHeight; ++y) {
        const float* rxx = b.xx.ptr<float>(y);
        const float* ryy = b.yy.ptr<float>(y);
        const float* rxy = b.xy.ptr<float>(y);
        float* re = b.eig.ptr<float>(y);
        for (int x = 0; x < kWidth; ++x) {
            const double s = static_cast<double>(rxx[x]) + static_cast<double>(ryy[x]);
            const double d = static_cast<double>(rxx[x]) - static_cast<double>(ryy[x]);
            const double c = static_cast<double>(rxy[x]);
            re[x] = static_cast<float>(0.5 * (s - std::sqrt(d * d + 4.0 * c * c)));
        }
    }

    double maxVal = 0.0;
    cv::minMaxLoc(b.eig, nullptr, &maxVal, nullptr, nullptr);
    cv::threshold(b.eig, b.eig, maxVal * kQualityLevel, 0.0, cv::THRESH_TOZERO);
    cv::dilate(b.eig, b.tmp, cv::Mat());

    b.candidates.clear();
    for (int y = 1; y + 1 < kHeight; ++y) {
        const float* e = b.eig.ptr<float>(y);
        const float* t = b.tmp.ptr<float>(y);
        for (int x = 1; x + 1 < kWidth; ++x) {
            if (e[x] != 0.0f && e[x] == t[x]) b.candidates.push_back(e + x);
        }
    }
    out.clear();
    if (b.candidates.empty()) return;
    std::sort(b.candidates.begin(), b.candidates.end(), GreaterThanPtr());

    const double minDistSq = kMinDistance * kMinDistance;
    const float* base = b.eig.ptr<float>(0);
    const size_t stride = b.eig.step / sizeof(float);
    for (size_t i = 0; i < b.candidates.size(); ++i) {
        const size_t ofs = static_cast<size_t>(b.candidates[i] - base);
        const int y = static_cast<int>(ofs / stride);
        const int x = static_cast<int>(ofs % stride);
        bool good = true;
        for (size_t j = 0; j < out.size(); ++j) {
            const double ddx = static_cast<double>(x) - static_cast<double>(out[j].x);
            const double ddy = static_cast<double>(y) - static_cast<double>(out[j].y);
            if (ddx * ddx + ddy * ddy < minDistSq) {
                good = false;
                break;
            }
        }
        if (!good) continue;
        out.push_back(cv::Point(x, y));
        if (out.size() == static_cast<size_t>(kMaxCorners)) break;
    }
}

} // namespace

int main() {
    std::printf("binCV T3.7 -- goodFeaturesToTrack against OpenCV (the TIER 2 denominator)\n");
    std::printf("frame %dx%d, blockSize %d, maxCorners %d, qualityLevel %.2f, minDistance %.5f\n",
                kWidth, kHeight, kBlockSize, kMaxCorners, kQualityLevel, kMinDistance);
    std::printf("(SEAL/seal_params.yaml verbatim; word uint32_t)\n\n");

    std::vector<bincv::BinMat<Word>> bins;
    std::vector<cv::Mat> bytes;
    std::vector<bincv::TernaryMat<Word>> dxs, dys;
    bins.reserve(kInputs);
    dxs.reserve(kInputs);
    dys.reserve(kInputs);
    for (int i = 0; i < kInputs; ++i) {
        bins.emplace_back(kWidth, kHeight);
        bytes.emplace_back(kHeight, kWidth, CV_8U);
        makeFrame(uint64_t{0x9E3779B9} + static_cast<uint64_t>(i) * uint64_t{7919},
                  bins.back(), bytes.back());
        dxs.emplace_back(kWidth, kHeight);
        dys.emplace_back(kWidth, kHeight);
    }

    std::vector<float> mapStorage(static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight),
                                  0.0f);
    bincv::ResponseMap mapView{mapStorage.data(), static_cast<size_t>(kWidth),
                               static_cast<size_t>(kHeight), static_cast<size_t>(kWidth)};
    // CAPACITY IS NOT maxCorners (ops/corner.hpp's capacity contract): the array is
    // also the candidate buffer, so it is sized to the worst-case survivor count.
    // That is the honest binCV footprint and it is what the memory table charges.
    const size_t capacity = static_cast<size_t>(kWidth - 2) * static_cast<size_t>(kHeight - 2);
    std::vector<bincv::Corner> corners(capacity);

    CvBuffers cvb;
    std::vector<cv::Point> cvBinarized;
    std::vector<cv::Point2f> cvSobel;

    bincv::GoodFeaturesParams params;
    params.blockSize = kBlockSize;
    params.maxCorners = kMaxCorners;
    params.qualityLevel = kQualityLevel;
    params.minDistance = kMinDistance;

    // -----------------------------------------------------------------------
    // AGREEMENT, BEFORE ANY TIMING. Two sides that disagree are not comparable.
    // -----------------------------------------------------------------------
    size_t binCvTotal = 0, cvTotal = 0, exactMatches = 0, worstGap = 0;
    size_t maxSurvivors = 0;
    for (int k = 0; k < kInputs; ++k) {
        const size_t ki = static_cast<size_t>(k);
        bincv::derivativeX(bins[ki], dxs[ki]);
        bincv::derivativeY(bins[ki], dys[ki]);
        const bincv::CornerResult r = bincv::goodFeaturesToTrack(
            dxs[ki], dys[ki], params, mapView, corners.data(), corners.size());
        openCvBinarized(bytes[ki], cvb, cvBinarized);
        binCvTotal += r.count;
        cvTotal += cvBinarized.size();
        if (r.candidatesRanked > maxSurvivors) maxSurvivors = r.candidatesRanked;
        if (cvb.candidates.size() > maxSurvivors) maxSurvivors = cvb.candidates.size();
        for (size_t i = 0; i < r.count; ++i) {
            size_t best = ~size_t{0};
            for (size_t j = 0; j < cvBinarized.size(); ++j) {
                const long long ddx = corners[i].x - cvBinarized[j].x;
                const long long ddy = corners[i].y - cvBinarized[j].y;
                const size_t d2 = static_cast<size_t>(ddx * ddx + ddy * ddy);
                if (d2 < best) best = d2;
            }
            if (best == 0) ++exactMatches;
            if (best != ~size_t{0} && best > worstGap) worstGap = best;
        }
    }
    std::printf("  AGREEMENT with the OpenCV binarized pipeline over %d frames:\n", kInputs);
    std::printf("    binCV corners %zu, OpenCV corners %zu, EXACT position matches %zu\n",
                binCvTotal, cvTotal, exactMatches);
    std::printf("    worst displacement of a binCV corner from the nearest OpenCV one: "
                "%.2f px\n", std::sqrt(static_cast<double>(worstGap)));
    std::printf("    (not bit-exact and not claimed to be -- TIER 2. The two sides share the\n"
                "     covariance integers exactly; the remaining gap is the float eig map and\n"
                "     how ties fall out of it.)\n\n");

    // -----------------------------------------------------------------------
    // FOOTPRINT, BEFORE THE TIMINGS -- the column this file was added for.
    // -----------------------------------------------------------------------
    const bincv::BinMatConstView<Word> plane = dxs[0].constMagnitude(0);
    const size_t planeBytes = plane.stride * plane.height * sizeof(Word);
    const size_t srcPlane = bins[0].sizeInWords() * sizeof(Word);
    const size_t binCvMap = mapStorage.size() * sizeof(float);
    const size_t binCvCorners = corners.size() * sizeof(bincv::Corner);
    const size_t binCvSet = srcPlane + 4 * planeBytes + binCvMap + binCvCorners;

    const size_t cvSrc = bytes[0].total() * bytes[0].elemSize();
    const size_t cvSet = cvSrc + cvb.bytes();

    const size_t pixels = static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight);
    // OpenCV's own goodFeaturesToTrack: cornerMinEigenVal materializes Dx, Dy
    // (CV_32F) and cov (CV_32FC3), and gftt adds eig and the dilate destination.
    // ACCOUNTED, not read -- those buffers are internal to the call.
    const size_t cvSobelSet = cvSrc + pixels * (4 + 4 + 12 + 4 + 4);

    std::printf("  PEAK WORKING SET -- live buffers for ONE call (CLAUDE.md), %zu pixels\n",
                pixels);
    std::printf("    binCV\n");
    std::printf("      source, 1 bit/pixel                  %9zu B\n", srcPlane);
    std::printf("      four one-bit derivative planes       %9zu B\n", 4 * planeBytes);
    std::printf("      float response map (caller scratch)  %9zu B\n", binCvMap);
    std::printf("      corner array = candidate buffer      %9zu B\n", binCvCorners);
    std::printf("      TOTAL                                %9zu B  (%5.2f B/pixel)\n",
                binCvSet, static_cast<double>(binCvSet) / static_cast<double>(pixels));
    std::printf("    OpenCV binarized (the denominator), every buffer read off itself\n");
    std::printf("      source, CV_8U                        %9zu B\n", cvSrc);
    std::printf("      dx, dy CV_32F                        %9zu B\n", 2 * pixels * 4);
    std::printf("      xx, yy, xy CV_32F                    %9zu B\n", 3 * pixels * 4);
    std::printf("      eig + dilate destination CV_32F      %9zu B\n", 2 * pixels * 4);
    std::printf("      candidate pointer vector             %9zu B\n",
                cvb.candidates.capacity() * sizeof(const float*));
    std::printf("      TOTAL                                %9zu B  (%5.2f B/pixel)\n",
                cvSet, static_cast<double>(cvSet) / static_cast<double>(pixels));
    std::printf("    OpenCV Sobel (stock call), ACCOUNTED from what the call materializes\n");
    std::printf("      TOTAL                                %9zu B  (%5.2f B/pixel)\n",
                cvSobelSet, static_cast<double>(cvSobelSet) / static_cast<double>(pixels));
    std::printf("    -> binCV footprint vs the denominator: %.2fx smaller\n\n",
                static_cast<double>(cvSet) / static_cast<double>(binCvSet));

    // BOTH SIDES ABOVE ARE SIZED TO THE WORST CASE, symmetrically: binCV's array is
    // also its candidate buffer and OpenCV's pointer vector is the same list, and
    // both worst cases are (w-2)(h-2) because every interior pixel of an all-equal
    // map is a 3x3 maximum. That is the honest bound but it is not what a frame
    // costs, and on THIS content it dominates binCV's total. So the same accounting
    // is repeated at the MEASURED survivor count -- what a caller who has run the
    // detector once would actually reserve. The fixed buffers do not move.
    const size_t binCvSized = binCvSet - binCvCorners + maxSurvivors * sizeof(bincv::Corner);
    const size_t cvSized = cvSet - cvb.candidates.capacity() * sizeof(const float*) +
                           maxSurvivors * sizeof(const float*);
    std::printf("  THE SAME ACCOUNTING AT THE MEASURED SURVIVOR COUNT (%zu over %d frames),\n",
                maxSurvivors, kInputs);
    std::printf("  which is what a caller reserves once it knows the content:\n");
    std::printf("    binCV                                  %9zu B  (%5.2f B/pixel)\n",
                binCvSized, static_cast<double>(binCvSized) / static_cast<double>(pixels));
    std::printf("    OpenCV binarized (the denominator)     %9zu B  (%5.2f B/pixel)\n",
                cvSized, static_cast<double>(cvSized) / static_cast<double>(pixels));
    std::printf("    -> binCV footprint vs the denominator: %.2fx smaller\n\n",
                static_cast<double>(cvSized) / static_cast<double>(binCvSized));

    // -----------------------------------------------------------------------
    // SPEED
    // -----------------------------------------------------------------------
    std::vector<measure::Bench> benches;
    benches.push_back({"binCV", [&](int i) {
                           const size_t k = static_cast<size_t>(i % kInputs);
                           bincv::derivativeX(bins[k], dxs[k]);
                           bincv::derivativeY(bins[k], dys[k]);
                           const bincv::CornerResult r = bincv::goodFeaturesToTrack(
                               dxs[k], dys[k], params, mapView, corners.data(), corners.size());
                           measure::g_sink += r.count;
                       }});
    benches.push_back({"OpenCV binarized", [&](int i) {
                           const size_t k = static_cast<size_t>(i % kInputs);
                           openCvBinarized(bytes[k], cvb, cvBinarized);
                           measure::g_sink += cvBinarized.size();
                       }});
    benches.push_back({"OpenCV Sobel", [&](int i) {
                           const size_t k = static_cast<size_t>(i % kInputs);
                           cv::goodFeaturesToTrack(bytes[k], cvSobel, kMaxCorners, kQualityLevel,
                                                   kMinDistance, cv::noArray(), kBlockSize,
                                                   false, 0.04);
                           measure::g_sink += cvSobel.size();
                       }});

    const std::vector<measure::Timing> t = measure::measureInterleaved(benches, 5, 200.0);
    const double denom = t[1].medianNs;

    std::printf("  %-18s %12s %10s %8s %12s %10s\n", "variant", "ns/frame", "ns/pixel", "spread",
                "vs OpenCV", "B/pixel");
    const size_t setBytes[3] = {binCvSet, cvSet, cvSobelSet};
    for (size_t i = 0; i < benches.size(); ++i) {
        std::printf("  %-18s %12.0f %10.3f %7.2f%% %11.2fx %10.2f\n", benches[i].name.c_str(),
                    t[i].medianNs, t[i].medianNs / static_cast<double>(pixels), t[i].spreadPct(),
                    denom / t[i].medianNs,
                    static_cast<double>(setBytes[i]) / static_cast<double>(pixels));
    }

    std::printf("\n  READ THE TWO COLUMNS TOGETHER, per CLAUDE.md -- neither settles the\n");
    std::printf("  question alone, and for this operation they point opposite ways.\n");
    std::printf("  sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
