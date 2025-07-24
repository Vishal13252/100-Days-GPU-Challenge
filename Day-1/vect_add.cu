#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* input1, const float* input2, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input1[idx] + input2[idx];
    }
}

int main() {
    const int size = 10;
    float hostInput1[size], hostInput2[size], hostOutput[size];

    // Initialize input arrays
    for (int i = 0; i < size; ++i) {
        hostInput1[i] = i * 1.0f;
        hostInput2[i] = (size - i) * 1.0f;
    }

    // Device pointers
    float *deviceInput1, *deviceInput2, *deviceOutput;
    cudaMalloc(&deviceInput1, size * sizeof(float));
    cudaMalloc(&deviceInput2, size * sizeof(float));
    cudaMalloc(&deviceOutput, size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceInput1, hostInput1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, size);

    // Copy result back to host
    cudaMemcpy(hostOutput, deviceOutput, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Input Array 1: ";
    for (int i = 0; i < size; ++i) std::cout << hostInput1[i] << " ";
    std::cout << "\nInput Array 2: ";
    for (int i = 0; i < size; ++i) std::cout << hostInput2[i] << " ";
    std::cout << "\nOutput Array (Sum): ";
    for (int i = 0; i < size; ++i) std::cout << hostOutput[i] << " ";
    std::cout << std::endl;

    // Free device memory
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    return 0;
}
