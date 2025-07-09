#pragma once

#include <chrono>
#include <opencv2/opencv.hpp>
#include "bincv-cpp/binMat.hpp"

namespace bench {

using Clock = std::chrono::high_resolution_clock;

inline void benchmark(const std::string& name, int iterations, const std::function<void()>& func) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[BENCH] " << name << " - Avg: " << (ms / iterations) << " ms over " << iterations << " runs\n";
}

inline void fillCVMat(cv::Mat& mat, uint8_t value) {
    mat.setTo(cv::Scalar(value));
}

inline void randomizeCVMat(cv::Mat& mat, int maxVal) {
    cv::randu(mat, 0, maxVal);
}

inline void setRandomBinMat(bincv::BinMat& mat, float fillRatio = 0.1f) {
    int w = mat.width();
    int h = mat.height();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if ((rand() / float(RAND_MAX)) < fillRatio) {
                mat.set(y, x, true);
            }
        }
    }
}

} // namespace bench
