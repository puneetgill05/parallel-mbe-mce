#include <device_launch_parameters.h>
#include <crt/host_defines.h>

#define N 4

// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    // int i = threadIdx.x;
    // int j = threadIdx.y;
    // C[i][j] = A[i][j] + B[i][j];
}

int run()
{

    float A[N*N], B[N*N], C[N*N];

    // Initialize input matrices
    for (int i = 0; i < N * N; ++i) {
        A[i] = i;
        B[i] = i * 2;
    }


    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    // MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);

}