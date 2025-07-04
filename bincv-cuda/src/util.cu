#include "bincv-cuda/util.hpp"
#include <cuda_runtime.h>
#include <stdint.h>

namespace bincv {
namespace util {

__global__ void norm_1b_to_uint8Kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        // if the pixel is 0, set to 0, otherwise set to 255
        // This assumes input is a 1-bit image where 0 is black and 1
        output[idx] = input[idx] ? 255 : 0;
    }
}

void norm_1b_to_uint8(const uint8_t* h_input, uint8_t* h_output, int width, int height) {
    size_t imageSize = width * height * sizeof(uint8_t);
    uint8_t *d_input, *d_output;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    norm_1b_to_uint8Kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void save_test_image(const std::string& imageName, const uint8_t* h_input, int width, int height) {
    // Create a directory for test outputs if it doesn't exist
    std::string outputDir = std::filesystem::current_path().string() + "/tests/output";
    std::filesystem::create_directories(outputDir);

    // Construct the full path for the output image
    std::string outputPath = outputDir + "/" + imageName;

    // Create an OpenCV Mat from the input data
    cv::Mat outputImage(height, width, CV_8UC1, const_cast<uint8_t*>(h_input));

    // Save the image using OpenCV
    cv::imwrite(outputPath, outputImage);
}

} // namespace util
} // namespace bincv