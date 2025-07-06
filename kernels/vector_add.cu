#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 10;
    float A[N], B[N], C[N];
    for(int i=0;i<N;i++) { A[i]=i; B[i]=2*i; }

    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_A, size); cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, size); cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_C, size);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAddKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++)
        std::cout << C[i] << " ";
    std::cout << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
// This code performs vector addition on the GPU using CUDA.
// It initializes two vectors A and B, allocates memory on the GPU, copies the vectors to the GPU, launches a kernel to perform the addition, and then copies the result back to the host.
// Finally, it prints the result and frees the allocated GPU memory.