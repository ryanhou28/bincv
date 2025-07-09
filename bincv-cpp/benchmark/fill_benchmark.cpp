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

    bench::benchmark("OpenCV fill", iterations, [&] {
        cvBin.setTo(cv::Scalar(255));
    });

    bench::benchmark("BinMat fill", iterations, [&] {
        binmat.fill(true);
    });


    return 0;
}
