#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/error.hpp"

namespace bench {

struct Config {
    int width = 640;
    int height = 480;
    int iterations = 100;
    float sparsity = 1.0f;
    std::string dtype = "uint8";  // "binary", "uint8", "float32"
    // Note: "binary" is still uint8, except the values are 0 or 1
};

inline Config parseArgs(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width") cfg.width = std::stoi(argv[++i]);
        else if (arg == "--height") cfg.height = std::stoi(argv[++i]);
        else if (arg == "--iterations") cfg.iterations = std::stoi(argv[++i]);
        else if (arg == "--dtype") cfg.dtype = argv[++i];
        else if (arg == "--sparsity") cfg.sparsity = std::stof(argv[++i]);
        // Through the project's error policy like every other check, so that
        // BINCV_THROW really is the single one (T1.4). The .c_str() is what lets
        // one spelling serve both expansions: the throw path takes the message by
        // std::string or by const char*, the abort path prints a const char*.
        else BINCV_THROW(std::invalid_argument, ("Unknown argument: " + arg).c_str());
    }

    std::cout << "=== Benchmark Config ===\n";
    std::cout << "Width: " << cfg.width << ", Height: " << cfg.height
              << ", Iterations: " << cfg.iterations
              << ", DType: " << cfg.dtype << ", Sparsity: " << cfg.sparsity << "\n";

    return cfg;
}

// --- Matrix Init Helpers ---

inline void fillCVMat(cv::Mat& mat, float value) {
    if (mat.type() == CV_8UC1)
        mat.setTo(cv::Scalar(static_cast<uint8_t>(value)));
    else if (mat.type() == CV_32FC1)
        mat.setTo(cv::Scalar(static_cast<float>(value)));
}

inline void randomizeCVMat(cv::Mat& mat, float maxVal = 1.0f) {
    if (mat.type() == CV_8UC1)
        cv::randu(mat, 0, static_cast<uint8_t>(maxVal));
    else if (mat.type() == CV_32FC1)
        cv::randu(mat, 0.0f, maxVal);
}

template <typename WordType>
inline void setRandomBinMat(bincv::BinMat<WordType>& mat, float fillRatio = 0.1f) {
    int w = mat.cols(), h = mat.rows();
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) < fillRatio)
                mat.set(y, x, true);
}

// --- Benchmark Wrapper ---

using Clock = std::chrono::high_resolution_clock;

inline void benchmark(const std::string& name, int iterations, const std::function<void()>& fn) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i)
        fn();
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[BENCH] " << name << " - Avg: " << (ms / iterations) << " ms over " << iterations << " runs\n";
}

// --- Memory Usage (Linux only) ---
// @TODO: Memory usage might be better captured by other tools (such as valgrind). Need more investigation.
#ifdef __linux__
#include <sys/resource.h>
inline size_t getCurrentRSS() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // ru_maxrss is a signed long; the kernel never reports it negative, but the
    // conversion has to say so rather than wrap silently.
    return usage.ru_maxrss < 0 ? size_t(0) : static_cast<size_t>(usage.ru_maxrss); // kilobytes
}
inline void printMemoryUsage(const std::string& label = "") {
    std::cout << "[MEM] " << label << " RSS: "
              << (static_cast<double>(getCurrentRSS()) / 1024.0) << " MB\n";
}
#endif

}  // namespace bench
