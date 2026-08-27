// Gradient-magnitude edge extraction (T5.11).
//
// TWO HALVES, the split tests/test_bitslice.cpp and tests/test_threshold.cpp use.
//
//   1. The CORE half needs no OpenCV, so it runs in all four verification
//      configurations. It checks ALL TWELVE combinations -- combine {Or, And} x
//      relation {Ge, Gt} x spatial {Wide, Forward, Backward} -- against a naive
//      per-pixel oracle that shares no code with the kernel.
//
//      A twelve-way option set with one tested combination is a one-combination
//      operation with eleven untested branches, which is why this is a cross-product
//      rather than a spot check.
//
//   2. The OPENCV half is the one that matters most: it checks that the DEFAULTS
//      reproduce the reference's own spelling, `rl_fast_edge_filter_wide`, written
//      out as the OpenCV calls it uses -- two filter2D in CV_32F, two abs, two
//      compares, an OR. Bit-exact, not approximately.
//
//      That is what makes "the defaults are the reference" a checked claim rather
//      than a comment.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/edge.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

using namespace bincv;

/// The oracle: reflect-101 indexing and the arithmetic written the obvious way.
/// Deliberately NOT sharing the kernel's helpers -- an oracle that calls the thing it
/// checks proves nothing.
template <typename SrcT>
long long refAt(const std::vector<SrcT>& img, size_t w, size_t h, long long y, long long x) {
    auto rf = [](long long i, size_t n) -> size_t {
        if (n <= 1) return 0;
        const long long last = static_cast<long long>(n) - 1;
        while (i < 0 || i > last) { if (i < 0) i = -i; if (i > last) i = 2 * last - i; }
        return static_cast<size_t>(i);
    };
    return static_cast<long long>(img[rf(y, h) * w + rf(x, w)]);
}

template <EdgeCombine C, EdgeRelation R, EdgeSpatial S, typename SrcT>
void checkCombination(const char* label, SrcT t) {
    constexpr size_t kW = 71, kH = 29;   // odd both ways: tail word and both borders
    std::vector<SrcT> img(kW * kH);
    uint64_t st = 0xBEEFULL;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<SrcT>(st >> 40);
    }
    BinMat<uint32_t> dst(static_cast<int>(kW), static_cast<int>(kH));
    edgeThreshold<C, R, S, SrcT, uint32_t>(img.data(), kW, kH, kW, dst.view(), t);

    size_t wrong = 0, dirty = 0;
    for (size_t y = 0; y < kH; ++y) {
        for (size_t x = 0; x < kW; ++x) {
            const long long yy = static_cast<long long>(y), xx = static_cast<long long>(x);
            long long dx, dy;
            if (S == EdgeSpatial::Wide) {
                dx = refAt(img, kW, kH, yy, xx + 1) - refAt(img, kW, kH, yy, xx - 1);
                dy = refAt(img, kW, kH, yy + 1, xx) - refAt(img, kW, kH, yy - 1, xx);
            } else if (S == EdgeSpatial::Forward) {
                dx = refAt(img, kW, kH, yy, xx + 1) - refAt(img, kW, kH, yy, xx);
                dy = refAt(img, kW, kH, yy + 1, xx) - refAt(img, kW, kH, yy, xx);
            } else {
                dx = refAt(img, kW, kH, yy, xx) - refAt(img, kW, kH, yy, xx - 1);
                dy = refAt(img, kW, kH, yy, xx) - refAt(img, kW, kH, yy - 1, xx);
            }
            if (dx < 0) dx = -dx;
            if (dy < 0) dy = -dy;
            const long long tt = static_cast<long long>(t);
            const bool px = (R == EdgeRelation::Ge) ? (dx >= tt) : (dx > tt);
            const bool py = (R == EdgeRelation::Ge) ? (dy >= tt) : (dy > tt);
            const bool want = (C == EdgeCombine::Or) ? (px || py) : (px && py);
            const bool got = ((dst.constView().row(y)[x / 32] >> (x % 32)) & 1u) != 0;
            if (want != got) ++wrong;
        }
        const uint32_t tail = dst.constView().row(y)[(kW + 31) / 32 - 1] >> (kW % 32);
        if (tail != 0) ++dirty;
    }
    std::printf("  %-38s %4zu wrong, %zu dirty-padding rows\n", label, wrong, dirty);
    BINCV_CHECK(wrong == 0);
    BINCV_CHECK(dirty == 0);
}

} // namespace

BINCV_TEST(Edge, AllTwelveCombinations_uint8) {
    using EC = EdgeCombine; using ER = EdgeRelation; using ES = EdgeSpatial;
    checkCombination<EC::Or,  ER::Ge, ES::Wide,     uint8_t>("Or  Ge Wide     (the reference)", 17);
    checkCombination<EC::Or,  ER::Gt, ES::Wide,     uint8_t>("Or  Gt Wide", 17);
    checkCombination<EC::And, ER::Ge, ES::Wide,     uint8_t>("And Ge Wide", 17);
    checkCombination<EC::And, ER::Gt, ES::Wide,     uint8_t>("And Gt Wide", 17);
    checkCombination<EC::Or,  ER::Ge, ES::Forward,  uint8_t>("Or  Ge Forward", 17);
    checkCombination<EC::Or,  ER::Gt, ES::Forward,  uint8_t>("Or  Gt Forward", 17);
    checkCombination<EC::And, ER::Ge, ES::Forward,  uint8_t>("And Ge Forward", 17);
    checkCombination<EC::And, ER::Gt, ES::Forward,  uint8_t>("And Gt Forward", 17);
    checkCombination<EC::Or,  ER::Ge, ES::Backward, uint8_t>("Or  Ge Backward", 17);
    checkCombination<EC::Or,  ER::Gt, ES::Backward, uint8_t>("Or  Gt Backward", 17);
    checkCombination<EC::And, ER::Ge, ES::Backward, uint8_t>("And Ge Backward", 17);
    checkCombination<EC::And, ER::Gt, ES::Backward, uint8_t>("And Gt Backward", 17);
}

BINCV_TEST(Edge, WideSource_uint16) {
    // ARCHITECTURE 7.8.1: the wide-source path exists BECAUSE of this operation, so
    // a 12-bit-shaped threshold is checked rather than only an 8-bit one.
    using EC = EdgeCombine; using ER = EdgeRelation; using ES = EdgeSpatial;
    checkCombination<EC::Or,  ER::Ge, ES::Wide, uint16_t>("Or  Ge Wide  u16 t=15", 15);
    checkCombination<EC::And, ER::Gt, ES::Wide, uint16_t>("And Gt Wide  u16 t=4095", 4095);
}

BINCV_TEST(Edge, TruncatingTo8BitLosesEdges) {
    // The measured form of ARCHITECTURE 7.8.1's argument, so it is a CHECKED claim
    // rather than a paragraph: a 12-bit image whose gradients are all smaller than
    // 16 counts has real edges at 12-bit precision and NONE after `v >> 4`.
    constexpr size_t kW = 64, kH = 16;
    std::vector<uint16_t> wide(kW * kH);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x)
            wide[y * kW + x] = static_cast<uint16_t>(2048u + ((x / 4u) % 2u) * 12u);
    BinMat<uint32_t> full(kW, kH), truncated(kW, kH);
    edgeThreshold<EdgeCombine::Or, EdgeRelation::Ge, EdgeSpatial::Wide, uint16_t, uint32_t>(
        wide.data(), kW, kH, kW, full.view(), 8);
    std::vector<uint8_t> narrow(kW * kH);
    for (size_t i = 0; i < wide.size(); ++i) narrow[i] = static_cast<uint8_t>(wide[i] >> 4);
    edgeThreshold<EdgeCombine::Or, EdgeRelation::Ge, EdgeSpatial::Wide, uint8_t, uint32_t>(
        narrow.data(), kW, kH, kW, truncated.view(), 1);
    const size_t keptFull = static_cast<size_t>(full.countNonZero());
    const size_t keptTrunc = static_cast<size_t>(truncated.countNonZero());
    std::printf("  12-bit source: %zu edge pixels; after v>>4: %zu\n", keptFull, keptTrunc);
    BINCV_CHECK(keptFull > 0);
    BINCV_CHECK(keptTrunc == 0);
}

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Edge, DefaultsMatchTheReferenceSpelling) {
    // rl_fast_edge_filter_wide, written as the OpenCV calls it actually makes.
    // THIS is what makes "the defaults are the reference" a checked claim.
    constexpr int kW = 91, kH = 37;
    cv::Mat img(kH, kW, CV_8U);
    uint64_t st = 7;
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            img.at<uint8_t>(y, x) = static_cast<uint8_t>(st >> 40);
        }
    const int t = 17;
    cv::Mat kx = (cv::Mat_<float>(1, 3) << -1, 0, 1);
    cv::Mat ky = (cv::Mat_<float>(3, 1) << -1, 0, 1);
    cv::Mat dx, dy;
    cv::filter2D(img, dx, CV_32F, kx);
    cv::filter2D(img, dy, CV_32F, ky);
    dx = cv::abs(dx);
    dy = cv::abs(dy);
    const cv::Mat mask = (dx >= t) | (dy >= t);

    BinMat<uint32_t> got(kW, kH);
    edgeThreshold(img.ptr<uint8_t>(0), static_cast<size_t>(kW), static_cast<size_t>(kH),
                  img.step, got.view(), static_cast<uint8_t>(t));

    size_t diff = 0;
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            const bool want = mask.at<uint8_t>(y, x) != 0;
            const bool have = ((got.constView().row(static_cast<size_t>(y))[x / 32] >>
                                (x % 32)) & 1u) != 0;
            if (want != have) ++diff;
        }
    std::printf("  vs rl_fast_edge_filter_wide(17): %zu of %d pixels differ\n", diff, kW * kH);
    BINCV_CHECK(diff == 0);
}
#endif

BINCV_TEST_MAIN("test_edge")
