//
// Created by puneet on 20/05/25.
//

#include "mat.cuh"


#include <device_launch_parameters.h>
#include <iostream>
#include <crt/host_defines.h>

using namespace std;


#define N 400

// Kernel definition
__global__ void MatAdd(float* A, float* B,
                       float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }}


int main_gpu() {
    float *A, *B, *C;
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    int threadsPerBlock = 32;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        cout << C[i] << " ";
    }


    // Kernel invocation with one block of N * N * 1 threads
    // int numBlocks = 1;
    // dim3 threadsPerBlock(N, N);
    // MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

}
