#include <iostream>
#include <cuda_runtime.h>

__global__ void reluKernel(float* A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        A[idx] = fmaxf(0.0f, A[idx]);
    }
}

int main() {
    const int N = 10;
    float A[N] = {-1, -0.5, 0, 0.5, 1, -2, 3, -4, 5, -6};

    float *d_A;
    size_t size = N * sizeof(float);
    cudaMalloc(&d_A, size); cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    reluKernel<<<blocks, threadsPerBlock>>>(d_A, N);

    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++)
        std::cout << A[i] << " ";
    std::cout << std::endl;

    cudaFree(d_A);
    return 0;
}
// This code performs ReLU activation on a vector A using CUDA.
// It initializes a vector with some negative and positive values, allocates memory on the GPU,
// copies the vector to the GPU, launches a kernel to apply the ReLU function, and then copies the result back to the host.
// Finally, it prints the result and frees the allocated GPU memory.        

//Output obtained:
// 0 0 0 0.5 1 0 3 0 5 0