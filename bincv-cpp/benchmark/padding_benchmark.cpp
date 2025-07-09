#include <iostream>
#include "bench_util.hpp"

int main() {
    int width = 640;
    int height = 480;
    int iterations = 100;
    int padTop = 1, padBottom = 1, padLeft = 2, padRight = 2;

    // Binary CV_8U matrix
    cv::Mat cvBin(height, width, CV_8UC1);
    bench::fillCVMat(cvBin, 0);
    bincv::BinMat binmat(width, height);

    bench::benchmark("BinMat pad", iterations, [&] {
        bincv::BinMat tmp = binmat;
        tmp.pad(padTop, padBottom, padLeft, padRight);
    });

    return 0;
}
