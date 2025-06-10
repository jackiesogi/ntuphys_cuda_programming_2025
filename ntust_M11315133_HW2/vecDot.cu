#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 6400
#define NUM_REPEATS 10

float* h_A;
float* h_C;
float* d_A;
float* d_C;

void RandomInit(float*, int);

// CUDA kernel for trace with parallel reduction
__global__ void MatrixTrace(const float* A, float* C, int N) {
    extern __shared__ float cache[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0.0f;
    while (i < N) {
        temp += A[i * N + i];  // access diagonal element
        i += blockDim.x * gridDim.x;
    }

    cache[tid] = temp;
    __syncthreads();

    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (tid < ib)
            cache[tid] += cache[tid + ib];
        __syncthreads();
        ib /= 2;
    }

    if (tid == 0)
        C[blockIdx.x] = cache[0];
}

int main(void) {
    cudaError_t err;

    // Allocate host memory
    int matrixSize = N * N;
    int size = matrixSize * sizeof(float);

    h_A = (float*)malloc(size);
    RandomInit(h_A, matrixSize);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Prepare result file
    FILE* fp = fopen("result.csv", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open result.csv for writing\n");
        return 1;
    }
    fprintf(fp, "block_size,avg_time_ms,gflops\n");

    for (int tpb = 1; tpb <= 32; ++tpb) {
        int threadsPerBlock = tpb;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        int sb = blocksPerGrid * sizeof(float);
        h_C = (float*)malloc(sb);
        cudaMalloc((void**)&d_C, sb);

        float totalTime = 0.0f;

        // CUDA event timers
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int sharedMemSize = threadsPerBlock * sizeof(float);

        for (int repeat = 0; repeat < NUM_REPEATS; ++repeat) {
            cudaEventRecord(start, 0);
            MatrixTrace<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_C, N);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            totalTime += ms;
        }

        float avgTime = totalTime / NUM_REPEATS;

        cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);

        // sum partial results from each block to get final trace
        double gpu_result = 0.0;
        for (int i = 0; i < blocksPerGrid; i++)
            gpu_result += (double)h_C[i];

        // Compute GFLOPS: 2*N operations (sum N elements with add)
        // Here the operation count is N (sum), but to be consistent with original: 2*N*N (matrix multiply), 
        // but trace sum is only N additions, so GFLOPS is low.
        // For meaningful performance, we can report:
        // FLOPS = N operations (N additions), but GPU FLOPS is low here.
        // We'll just report time here.
        float gflops = (float)(N) / (avgTime / 1000.0f) / 1e9f;  // in GFLOPS (very small)

        fprintf(fp, "%d,%.5f,%.9f\n", threadsPerBlock, avgTime, gflops);

        printf("Block size: %d, Avg time: %.5f ms, GPU Trace: %.15e\n", threadsPerBlock, avgTime, gpu_result);

        cudaFree(d_C);
        free(h_C);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // CPU verification
    double cpu_result = 0.0;
    for (int i = 0; i < N; i++)
        cpu_result += h_A[i * N + i];
    printf("CPU Trace: %.15e\n", cpu_result);

    free(h_A);
    cudaFree(d_A);
    fclose(fp);

    cudaDeviceReset();

    return 0;
}

void RandomInit(float* data, int n) {
    for (int i = 0; i < n; ++i)
        data[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f;
}

