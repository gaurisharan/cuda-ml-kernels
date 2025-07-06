#include <iostream>
#include <cuda_runtime.h>

__global__ void dotProductKernel(float* A, float* B, float* result, int N) {
    __shared__ float cache[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0;
    while (idx < N) {
        temp += A[idx] * B[idx];
        idx += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while(i != 0) {
        if(cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
        i /= 2;
    }

    if(cacheIdx == 0)
        atomicAdd(result, cache[0]);
}

int main() {
    const int N = 1024;
    float A[N], B[N], h_result = 0, *d_A, *d_B, *d_result;

    for(int i=0;i<N;i++) { A[i]=1; B[i]=2; }

    size_t size = N * sizeof(float);
    cudaMalloc(&d_A, size); cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, size); cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    dotProductKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_result, N);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Dot Product: " << h_result << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_result);
    return 0;
}
// This code performs dot product of two vectors on the GPU using CUDA.
// It initializes two vectors A and B, allocates memory on the GPU, copies the vectors to the GPU, launches a kernel to perform the dot product, and then copies the result back to the host.
// Finally, it prints the result and frees the allocated GPU memory.

//Output obtained-
// Dot Product: 2048