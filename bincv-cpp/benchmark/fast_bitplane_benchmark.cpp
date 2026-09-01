// ===========================================================================
// X-80 / E-43 -- FAST ON A BIT-PLANE, WHICH IS THE OPERATION binCV IS FOR.
//
// [X-77](../../docs/EXPERIMENTS.md) concluded that FAST could only match `cv::FAST`
// because "FAST's input is 8-bit". That is true of `detectFast(const SrcT*, ...)`,
// whose own header says "on a WIDE image" -- and it is a property of the SIGNATURE,
// not of FAST. On a one-bit frame the detector collapses to boolean algebra: there is
// exactly one meaningful threshold, and the test becomes
//
//     corner = arc9(ring & ~centre) | arc9(~ring & centre)
//
// This measures the three arms on identical content:
//
//   (a) `cv::FAST` on the binary frame stored as CV_8U       -- CLAUDE.md's denominator
//   (b) binCV's WIDE detectFast on the same CV_8U buffer      -- what X-77 measured
//   (c) binCV's BIT-PLANE detectFast on the same content      -- the question
//
// MEMORY IS REPORTED WITH SPEED because they trade off and (c)'s input is eight times
// smaller, which is half of what is being asked.
//
// THE ARMS ARE INTERLEAVED AND THE MINIMUM IS REPORTED. This machine is shared; a
// mean over a contended run measures the contention, and interleaving keeps a slow
// stretch from landing on one arm.
//
// Usage: fast_bitplane_benchmark <image> [more images...]
// ===========================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/fast.hpp"

using Clock = std::chrono::steady_clock;

namespace {

double minOf(const std::vector<double>& v) {
    return *std::min_element(v.begin(), v.end());
}

void runOne(const cv::Mat& gray, const char* label) {
    // `realframe.bin` holds {0, 1}; a photograph holds {0..255}. Binarise at a level
    // that means the same thing for both, and land on {0, 255} either way -- the
    // equivalence with `cv::FAST` is stated for {0, 255} content.
    double lo = 0.0, hi = 0.0;
    cv::minMaxLoc(gray, &lo, &hi);
    cv::Mat bin;
    cv::threshold(gray, bin, hi <= 1.0 ? 0.0 : 110.0, 255, cv::THRESH_BINARY);
    bincv::BinMat<uint32_t> plane(bin.cols, bin.rows);
    plane.fromCVMat(bin);

    std::vector<bincv::FastCorner> corners(400000);
    std::vector<cv::KeyPoint> kp;
    bool truncated = false;
    size_t nBit = 0, nWide = 0, nCv = 0;

    constexpr int kRounds = 12;
    constexpr int kReps = 8;
    std::vector<double> tCv, tWide, tBit;
    for (int r = 0; r < kRounds; ++r) {
        auto t = Clock::now();
        for (int i = 0; i < kReps; ++i) cv::FAST(bin, kp, 100, false);
        tCv.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t).count() / kReps);
        nCv = kp.size();

        t = Clock::now();
        for (int i = 0; i < kReps; ++i) {
            nWide = bincv::detectFast<uint8_t>(bin.data, static_cast<size_t>(bin.cols),
                                               static_cast<size_t>(bin.rows), bin.step, 100,
                                               corners.data(), corners.size(), &truncated);
        }
        tWide.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t).count() / kReps);

        t = Clock::now();
        for (int i = 0; i < kReps; ++i) {
            nBit = bincv::detectFast(plane.constView(), corners.data(), corners.size(),
                                     &truncated);
        }
        tBit.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t).count() / kReps);
    }

    const double cvMin = minOf(tCv), wideMin = minOf(tWide), bitMin = minOf(tBit);
    const size_t planeBytes = plane.sizeInWords() * sizeof(uint32_t);
    const size_t byteBytes = bin.total();

    std::printf("\n  %s  %dx%d\n", label, bin.cols, bin.rows);
    std::printf("    corners: cv::FAST %zu   binCV wide %zu   binCV bit-plane %zu%s\n", nCv,
                nWide, nBit, nBit == nCv ? "   (bit-plane == cv::FAST)" : "   MISMATCH");
    std::printf("    %-26s %9.1f us   %6s   %9zu B\n", "(a) cv::FAST on CV_8U", cvMin, "1.00x",
                byteBytes);
    std::printf("    %-26s %9.1f us   %5.2fx   %9zu B\n", "(b) binCV wide, CV_8U", wideMin,
                cvMin / wideMin, byteBytes);
    std::printf("    %-26s %9.1f us   %5.2fx   %9zu B   (%.1fx smaller input)\n",
                "(c) binCV BIT-PLANE", bitMin, cvMin / bitMin, planeBytes,
                static_cast<double>(byteBytes) / static_cast<double>(planeBytes));

    // X-81: WHERE THE TWO SCORING ARMS CROSS, MEASURED PER ARCHITECTURE RATHER THAN
    // DERIVED ONCE. The arithmetic says ~2.8 corners per chunk on both -- but a chunk
    // is 256 pixels on AVX2 and 128 on NEON, and a Cortex-A72 has two vector pipes
    // against a modern x86 core's four-plus, so the same operation count is not the
    // same time. The first NEON threshold was taken from the x86 arithmetic and cost
    // the reference device 2.36x -> 2.12x. This sweep is why it is no longer guessed.
    const int savedThreshold = bincv::impl::fastScoreMaskThreshold();
    std::printf("    scoring-arm crossover sweep (corners per chunk before the arc masks"
                " are used):\n");
    for (int th : {0, 2, 3, 4, 6, 8, 12, 1 << 30}) {
        bincv::impl::fastScoreMaskThreshold() = th;
        std::vector<double> ts;
        for (int r = 0; r < kRounds; ++r) {
            auto t = Clock::now();
            for (int i = 0; i < kReps; ++i) {
                nBit = bincv::detectFast(plane.constView(), corners.data(), corners.size(),
                                         &truncated);
            }
            ts.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t).count() /
                         kReps);
        }
        const double m = minOf(ts);
        const std::string tag = th == 0 ? "0 (always)"
                                        : (th > 1000 ? "never" : std::to_string(th));
        std::printf("      threshold %-10s %9.1f us   %5.2fx vs cv::FAST\n", tag.c_str(), m,
                    cvMin / m);
    }
    bincv::impl::fastScoreMaskThreshold() = savedThreshold;
}

/// The committed binarized frame, so this runs on the reference device with no dataset
/// present. Raw `{uint32 w, uint32 h, w*h bytes}` -- the format every other benchmark
/// here reads it in.
cv::Mat loadRealFrame() {
    std::FILE* f = std::fopen(BINCV_REALFRAME_PATH, "rb");
    if (f == nullptr) return cv::Mat();
    uint32_t w = 0, h = 0;
    if (std::fread(&w, 4, 1, f) != 1 || std::fread(&h, 4, 1, f) != 1) {
        std::fclose(f);
        return cv::Mat();
    }
    cv::Mat m(static_cast<int>(h), static_cast<int>(w), CV_8U);
    const size_t want = static_cast<size_t>(w) * h;
    if (std::fread(m.data, 1, want, f) != want) m = cv::Mat();
    std::fclose(f);
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    cv::setNumThreads(1);
    std::printf("=== X-80 / E-43: FAST on a bit-plane against cv::FAST ===\n");
    std::printf("one thread; 12 interleaved rounds, minimum reported\n");
    if (argc < 2) {
        const cv::Mat frame = loadRealFrame();
        if (frame.empty()) {
            std::printf("realframe.bin unreadable and no image given\n");
            return 2;
        }
        runOne(frame, "benchmark/realframe.bin");
        return 0;
    }
    for (int i = 1; i < argc; ++i) {
        const cv::Mat gray = cv::imread(argv[i], cv::IMREAD_GRAYSCALE);
        if (gray.empty()) continue;
        runOne(gray, argv[i]);
    }
    return 0;
}
