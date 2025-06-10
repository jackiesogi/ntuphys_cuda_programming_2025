// Vector addition: C = 1/A + 1/B.
// compile with the following command:
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 -o vecAdd vecAdd.cu
// (for GTX4090)
// nvcc -arch=compute_89 -code=sm_89 -O3 -o vecAdd vecAdd.cu

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define NUM_REPEATS 10

// Device code

__global__ void compute(float* A, float* B, float* C, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        int idx = i * n + j;
        C[idx] = 1.0f / A[idx] + 1.0f / B[idx];
    }
}

// Host code

int main() {
    int gid;
    cudaError_t err = cudaSuccess;

    printf("Enter the GPU_ID: ");
    scanf("%d", &gid);
    printf("%d\n", gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    printf("Matrix Addition: C = 1/A + 1/B\n");

    int mem = 1024 * 1024 * 1024; // 1GB
    int N;
    printf("Enter the size of the matrices: ");
    scanf("%d", &N);
    printf("(%d, %d)\n", N, N);

    if (sizeof(float) * N * N > mem) {
        printf("The size of these 3 matrices cannot be fitted into 1 Gbyte\n");
        exit(2);
    }

    size_t size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    srand(time(NULL));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = ((float)rand() / (RAND_MAX)) + 0.0001f;
        h_B[i] = ((float)rand() / (RAND_MAX)) + 0.0001f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    FILE* f = fopen("./result/result.csv", "w");
    fprintf(f, "threads_per_block,avg_time_ms,gflops\n");

    for (int tpb = 1; tpb <= 32; ++tpb) {
        dim3 blockDim(tpb, tpb);
        dim3 gridDim((N + tpb - 1) / tpb, (N + tpb - 1) / tpb);

        float totalTime = 0.0f;
        for (int i = 0; i < NUM_REPEATS; ++i) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            compute<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            totalTime += ms;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        float avgTime = totalTime / NUM_REPEATS;
        float gflops = (2.0f * N * N / 1e9f) / (avgTime / 1000.0f);
        printf("TPB=%d, Avg Time=%.4f ms, GFLOPS=%.4f\n", tpb, avgTime, gflops);
        fprintf(f, "%d,%.4f,%.4f\n", tpb, avgTime, gflops);
    }

    fclose(f);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

