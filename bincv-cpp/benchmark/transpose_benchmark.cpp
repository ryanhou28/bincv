#include <iostream>
#include "bench_util.hpp"

int main() {
    int width = 1024, height = 512, iterations = 100;
    cv::Mat cvBin(height, width, CV_8UC1); bench::fillCVMat(cvBin, 1);
    bincv::BinMat binmat(width, height); binmat.fill(true);

    bench::benchmark("OpenCV transpose", iterations, [&] {
        cv::Mat transposed;
        cv::transpose(cvBin, transposed);
    });

    bench::benchmark("BinMat transposed()", iterations, [&] {
        auto t = binmat.transposed();
    });

    return 0;
}
