#include <iostream>
#include "bench_util.hpp"

int main() {
    int width = 640;
    int height = 480;
    int newWidth = 1280;
    int newHeight = 960;
    int iterations = 100;

    // Binary CV_8U matrix
    cv::Mat cvBin(height, width, CV_8UC1);
    bench::fillCVMat(cvBin, 0);
    bincv::BinMat binmat(width, height);

    std::cout << "=== Resize Benchmark ===\n";

    bench::benchmark("OpenCV resize (binary)", iterations, [&] {
        cv::Mat resized;
        cv::resize(cvBin, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_NEAREST);
    });

    bench::benchmark("BinMat resize", iterations, [&] {
        bincv::BinMat tmp = binmat;
        tmp.resize(newWidth, newHeight);
    });

    return 0;
}
