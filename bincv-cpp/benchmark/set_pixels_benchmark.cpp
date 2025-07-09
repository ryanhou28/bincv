#include <iostream>
#include "bench_util.hpp"

int main() {
    int width = 640;
    int height = 480;
    int iterations = 100;

    // Binary CV_8U matrix
    cv::Mat cvBin(height, width, CV_8UC1);
    bench::fillCVMat(cvBin, 0);
    bincv::BinMat binmat(width, height);

    std::vector<std::pair<int, int>> coords;
    for (int i = 0; i < 1000; ++i)
        coords.emplace_back(rand() % height, rand() % width);

    bench::benchmark("OpenCV set pixels", iterations, [&] {
        for (const auto& [y, x] : coords)
            cvBin.at<uint8_t>(y, x) = 255;
    });

    bench::benchmark("BinMat set pixels", iterations, [&] {
        for (const auto& [y, x] : coords)
            binmat.set(y, x, true);
    });


    return 0;
}
