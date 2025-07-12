#include <iostream>
#include "bench_util.hpp"

int main(int argc, char* argv[]) {
    // Parse command-line config: width, height, dtype, sparsity, iterations
    bench::Config cfg = bench::parseArgs(argc, argv);

    // Define new target size (e.g., 2× upscaling)
    int newWidth = cfg.width * 2;
    int newHeight = cfg.height * 2;

    // Construct input BinMat and cv::Mat based on dtype
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

    // Initialize values based on dtype and sparsity
    if (cfg.dtype == "binary") {
        binmat.fill(false);
        bench::setRandomBinMat(binmat, cfg.sparsity);
        bench::fillCVMat(cvmat, 0);
    } else {
        bench::randomizeCVMat(cvmat, cfg.dtype == "uint8" ? 255 : 1.0f);
    }

    std::cout << "\n=== Resize Benchmark ===\n";

    bench::benchmark("OpenCV resize", cfg.iterations, [&] {
        cv::Mat resized;
        cv::resize(cvmat, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_NEAREST);
    });

    bench::benchmark("BinMat resize", cfg.iterations, [&] {
        bincv::BinMat tmp = binmat;
        tmp.resize(newWidth, newHeight);
    });

    return 0;
}
