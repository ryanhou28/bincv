#include "bincv-cuda/edge_filter.hpp"
#include <cuda_runtime.h>
#include <stdint.h>

namespace bincv {

__global__ void horizontalEdgeFilter8bKernel(const uint8_t* input, uint8_t* output, int width, int height, uint8_t threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip out-of-bounds and border pixels
    if (x <= 0 || x >= width - 1 || y >= height)
        return;

    int idx = y * width + x;
    int left = input[y * width + (x - 1)];
    int right = input[y * width + (x + 1)];

    output[idx] = (abs(right - left) >= threshold) ? 1 : 0;
}

void horizontalEdgeFilter8b(const uint8_t* h_input, uint8_t* h_output, int width, int height, uint8_t threshold) {
    size_t imageSize = width * height * sizeof(uint8_t);
    uint8_t *d_input, *d_output;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    horizontalEdgeFilter8bKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, threshold);

    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void verticalEdgeFilter8bKernel(const uint8_t* input, uint8_t* output, int width, int height, uint8_t threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip out-of-bounds and border pixels
    if (x <= 0 || x >= width - 1 || y >= height)
        return;

    int idx = y * width + x;
    int top = input[(y - 1) * width + x];
    int bottom = input[(y + 1) * width + x];

    output[idx] = (abs(top - bottom) >= threshold) ? 1 : 0;
}

void verticalEdgeFilter8b(const uint8_t* h_input, uint8_t* h_output, int width, int height, uint8_t threshold) {
    size_t imageSize = width * height * sizeof(uint8_t);
    uint8_t *d_input, *d_output;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    verticalEdgeFilter8bKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, threshold);

    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

}
