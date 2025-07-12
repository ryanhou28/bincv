#include <iostream>
#include "bench_util.hpp"

int main(int argc, char* argv[]) {
    bench::Config cfg = bench::parseArgs(argc, argv);

    bincv::BinMat binmat(cfg.width, cfg.height);
    cv::Mat cvmat;

    if (cfg.dtype == "binary" || cfg.dtype == "uint8")
        cvmat = cv::Mat(cfg.height, cfg.width, CV_8UC1);
    else if (cfg.dtype == "float32")
        cvmat = cv::Mat(cfg.height, cfg.width, CV_32FC1);
    else {
        std::cerr << "Unsupported dtype: " << cfg.dtype << "\n";
        return 1;
    }

    // Initialize data
    if (cfg.dtype == "binary") {
        binmat.fill(false);
        bench::setRandomBinMat(binmat, cfg.sparsity);
        bench::fillCVMat(cvmat, 0);
    } else {
        bench::randomizeCVMat(cvmat, cfg.dtype == "uint8" ? 255 : 1.0f);
    }

    bench::benchmark("OpenCV fill", cfg.iterations, [&] {
        cvmat.setTo(cv::Scalar(255));
    });

    bench::benchmark("BinMat fill", cfg.iterations, [&] {
        binmat.fill(true);
    });

    return 0;
}
