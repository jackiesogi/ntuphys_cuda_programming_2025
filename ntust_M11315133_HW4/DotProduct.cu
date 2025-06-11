/*
    Vector dot product with N GPUs: C = A · B

    Compile with (for GTX1060):
        nvcc -Xcompiler -fopenmp -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 \
            -o vectorDotProduct vectorDotProduct.cu -lcudart
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

void RandomInit(float *data, int M);

/* Device reduction kernel: each block computes a partial dot‐product */
__global__ void compute(const float* A, const float* B, float* result, int M) {
    extern __shared__ float cache[];
    int global_index = blockDim.x * blockIdx.x + threadIdx.x;
    int local_index  = threadIdx.x;
    int stride       = blockDim.x * gridDim.x;

    float temp = 0.0f;
    /* Strided load+multiply loop */
    while (global_index < M) {
        temp += A[global_index] * B[global_index];
        global_index += stride;
    }

    cache[local_index] = temp;
    __syncthreads();

    /* Binary reduction in shared memory */
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (local_index < offset) {
            cache[local_index] += cache[local_index + offset];
        }
        __syncthreads();
    }

    /* Write block result */
    if (local_index == 0) {
        result[blockIdx.x] = cache[0];
    }
}

int main() {
    int gpu_count;
    scanf("%d", &gpu_count);

    int *gpu_ids = (int*)malloc(sizeof(int) * gpu_count);
    for (int i = 0; i < gpu_count; ++i) {
        scanf("%d", &gpu_ids[i]);
    }

    int threads_per_block, blocks_per_grid;
    scanf("%d", &threads_per_block);
    printf("Threads Per Block: %d\n", threads_per_block);
    scanf("%d", &blocks_per_grid);
    printf("Blocks Per Grid: %d\n", blocks_per_grid);

    const int M = 40960000;
    const int N = (M + gpu_count - 1) / gpu_count;  // slice length per GPU

    /* Allocate page-locked host memory */
    float *A_h, *B_h, *result_h;
    cudaMallocHost((void**)&A_h, M * sizeof(float));
    cudaMallocHost((void**)&B_h, M * sizeof(float));
    /* need gpu_count * blocks_per_grid entries for partial sums */
    cudaMallocHost((void**)&result_h, gpu_count * blocks_per_grid * sizeof(float));

    srand((unsigned)time(NULL));
    RandomInit(A_h, M);
    RandomInit(B_h, M);

    omp_set_num_threads(gpu_count);

    float *host_to_device_times = (float*)malloc(gpu_count * sizeof(float));
    float *kernel_times = (float*)malloc(gpu_count * sizeof(float));
    float *device_to_host_times = (float*)malloc(gpu_count * sizeof(float));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(gpu_ids[tid]);

        int this_M = N;
        int offset = tid * N;
        if (offset + this_M > M) {
            this_M = M - offset;
        }

        /* Allocate device memory for this slice */
        float *A_d, *B_d, *partial_d;
        cudaMalloc(&A_d, this_M * sizeof(float));
        cudaMalloc(&B_d, this_M * sizeof(float));
        cudaMalloc(&partial_d, blocks_per_grid * sizeof(float));

        /* Create per-thread CUDA events */
        cudaEvent_t gpu_start, gpu_stop;

        /* H2D copy timing */
        cudaEventCreate(&gpu_start);
        cudaEventCreate(&gpu_stop);
        cudaEventRecord(gpu_start, 0);
        cudaMemcpy(A_d, A_h + offset, this_M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h + offset, this_M * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(gpu_stop, 0);
        cudaEventSynchronize(gpu_stop);
        cudaEventElapsedTime(&host_to_device_times[tid], gpu_start, gpu_stop);

        /* Kernel timing */
        cudaEventRecord(gpu_start, 0);
        compute
            <<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(
                A_d, B_d, partial_d, this_M);
        cudaEventRecord(gpu_stop, 0);
        cudaEventSynchronize(gpu_stop);
        cudaEventElapsedTime(&kernel_times[tid], gpu_start, gpu_stop);

        /* D2H copy timing */
        cudaEventRecord(gpu_start, 0);
        cudaMemcpy(result_h + tid * blocks_per_grid, partial_d,
                   blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(gpu_stop, 0);
        cudaEventSynchronize(gpu_stop);
        cudaEventElapsedTime(&device_to_host_times[tid], gpu_start, gpu_stop);

        /* Cleanup per-thread resources */
        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_stop);
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(partial_d);
    }

    /* Max time for each stage */
    float host_to_device_time = 0.0f, kernel_time = 0.0f, device_to_host_time = 0.0f;
    for (int i = 0; i < gpu_count; ++i) {
        host_to_device_time = fmax(host_to_device_time, host_to_device_times[i]);
        kernel_time = fmax(kernel_time, kernel_times[i]);
        device_to_host_time = fmax(device_to_host_time, device_to_host_times[i]);
    }
    printf("Host to Device Time: %f ms\n", host_to_device_time);
    printf("Kernel Time: %f ms\n", kernel_time);
    printf("Device to Host Time: %f ms\n", device_to_host_time);

    /* Merge partial sums on CPU (in ms) */
    clock_t cpu_start = clock();
    double gpu_result = 0.0;
    for (int i = 0; i < gpu_count * blocks_per_grid; ++i) {
        gpu_result += result_h[i];
    }
    clock_t cpu_end = clock();
    float merge_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("Merge Time: %f ms\n", merge_time);

    /* Aggregate GPU total time (all in ms) */
    float gpu_time = host_to_device_time + kernel_time + device_to_host_time + merge_time;
    printf("GPU Time: %f ms\n", gpu_time);

    /* CPU reference dot‐product timing (in ms) */
    cpu_start = clock();
    double cpu_result = 0.0;
    for (int i = 0; i < M; ++i) {
        cpu_result += A_h[i] * B_h[i];
    }
    cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU Time: %f ms\n", cpu_time);

    /* Speedup and GFLOPS */
    printf("Speedup With Data Transfer: %.2fx\n", cpu_time / gpu_time);
    printf("Speedup Without Data Transfer: %.2fx\n", cpu_time / (gpu_time - host_to_device_time - device_to_host_time));
    printf("GPU GFLOPS (kernel): %f\n", 2.0f * M / (kernel_time * 1e-6f) / 1e9f);
    printf("CPU GFLOPS: %f\n", 2.0f * M / (cpu_time * 1e-6f) / 1e9f);

    /* Verify correctness */
    double diff = fabs(cpu_result - gpu_result);
    printf("Norm(cpu_result - gpu_result): %e\n\n", diff);

    /* Cleanup */
    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFreeHost(result_h);
    free(gpu_ids);
    cudaDeviceReset();

    return 0;
}

void RandomInit(float *data, int M) {
    /* Fill host array with random floats in [0,1) */
    for (int i = 0; i < M; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}
