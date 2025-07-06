#include <iostream>
#include <cuda_runtime.h>

__global__ void matMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N) {
        float sum = 0;
        for(int i = 0; i < N; ++i)
            sum += A[row * N + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

void matMul(float* A, float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size); cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, size); cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_C, size);

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((N + 15)/16, (N + 15)/16);
    matMulKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    const int N = 4;
    float A[N*N], B[N*N], C[N*N];

    // Initialize A and B with simple values
    for (int i = 0; i < N*N; i++) {
        A[i] = i + 1;  // 1,2,3,4,...
        B[i] = (i + 1) * 2;  // 2,4,6,8,...
    }

    matMul(A, B, C, N);

    std::cout << "Result matrix:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << C[i*N + j] << " ";
        std::cout << "\n";
    }

    return 0;
}
// This code performs matrix multiplication on the GPU using CUDA.
// It initializes two matrices A and B, performs the multiplication in a kernel, and then copies the result back to the host.
// Finally, it prints the result matrix to the console.
