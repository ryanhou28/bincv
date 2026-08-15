#include <iostream>
#include "bench_util.hpp"

int main(int argc, char* argv[]) {
    // Parse args: width, height, dtype, sparsity, iterations
    bench::Config cfg = bench::parseArgs(argc, argv);

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

    // Initialize values based on dtype and sparsity
    if (cfg.dtype == "binary") {
        binmat.fill(false);
        bench::setRandomBinMat(binmat, cfg.sparsity);
        bench::fillCVMat(cvmat, 0);
    } else {
        bench::randomizeCVMat(cvmat, cfg.dtype == "uint8" ? 255 : 1.0f);
    }

    std::cout << "\n=== Transpose Benchmark ===\n";

    bench::benchmark("OpenCV transpose", cfg.iterations, [&] {
        cv::Mat transposed;
        cv::transpose(cvmat, transposed);
    });

    bench::benchmark("BinMat transposed()", cfg.iterations, [&] {
        auto t = binmat.transposed();
    });

    return 0;
}
