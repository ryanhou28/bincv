#include <iostream>
#include <vector>
#include "bench_util.hpp"

int main(int argc, char* argv[]) {
    // Parse CLI arguments: width, height, dtype, sparsity, iterations
    bench::Config cfg = bench::parseArgs(argc, argv);

    // Construct matrices
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

    // Fill input with optional sparsity control
    if (cfg.dtype == "binary") {
        binmat.fill(false);
        bench::setRandomBinMat(binmat, cfg.sparsity);
        bench::fillCVMat(cvmat, 0);
    } else {
        bench::randomizeCVMat(cvmat, cfg.dtype == "uint8" ? 255 : 1.0f);
    }

    // Prepare random coordinates to set (sparsity does not affect this)
    const int numSet = 1000;
    std::vector<std::pair<int, int>> coords;
    coords.reserve(numSet);
    for (int i = 0; i < numSet; ++i)
        coords.emplace_back(rand() % cfg.height, rand() % cfg.width);

    std::cout << "\n=== Set Pixels Benchmark ===\n";

    bench::benchmark("OpenCV set pixels", cfg.iterations, [&] {
        for (const auto& [y, x] : coords)
            cvmat.at<uint8_t>(y, x) = 255;
    });

    bench::benchmark("BinMat set pixels", cfg.iterations, [&] {
        for (const auto& [y, x] : coords)
            binmat.set(y, x, true);
    });

    return 0;
}
