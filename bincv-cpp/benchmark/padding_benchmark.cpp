#include <iostream>
#include "bench_util.hpp"

int main(int argc, char* argv[]) {
    // Parse args: width, height, dtype, sparsity, iterations
    bench::Config cfg = bench::parseArgs(argc, argv);

    // Define fixed padding for now
    // @TODO: Make padding configurable via command line arguments
    int padTop = 1, padBottom = 1, padLeft = 2, padRight = 2;

    // Construct input BinMat and cv::Mat based on dtype
    bincv::BinMat<> binmat(cfg.width, cfg.height);
    cv::Mat cvmat;

    if (cfg.dtype == "binary" || cfg.dtype == "uint8")
        cvmat = cv::Mat(cfg.height, cfg.width, CV_8UC1);
    else if (cfg.dtype == "float32")
        cvmat = cv::Mat(cfg.height, cfg.width, CV_32FC1);
    else {
        std::cerr << "Unsupported dtype: " << cfg.dtype << "\n";
        return 1;
    }

    // Fill data based on sparsity
    if (cfg.dtype == "binary") {
        binmat.fill(false);
        bench::setRandomBinMat(binmat, cfg.sparsity);
        bench::fillCVMat(cvmat, 0);
    } else {
        bench::randomizeCVMat(cvmat, cfg.dtype == "uint8" ? 255 : 1.0f);
    }

    // Run BinMat pad benchmark
    bench::benchmark("BinMat pad", cfg.iterations, [&] {
        bincv::BinMat<> tmp = binmat;
        tmp.pad(padTop, padBottom, padLeft, padRight);
    });

    return 0;
}
