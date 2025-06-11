#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* initialize random values in the matrix: [-1, 1] */
void randomInit(float *data, int N) {
    for (int i = 0; i < N; ++i) {
        data[i] = 2.0 * (rand() / (float) RAND_MAX) - 1.0;
    }
}

/* CUDA kernel to sum up diagonal entries using parallel reduction */
__global__ void matrixTrace(const float* diagonal, float* output, int N) {
    extern __shared__ float sharedCache[];

    int localIdx = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0f;
    while (globalIdx < N) {
        sum += (double) diagonal[globalIdx];
        globalIdx += stride;
    }

    sharedCache[localIdx] = sum;
    __syncthreads();

    // blockDim.x must be a power of 2
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (localIdx < offset) {
            sharedCache[localIdx] += sharedCache[localIdx + offset];
        }
        __syncthreads();
    }

    if (localIdx == 0)
        output[blockIdx.x] = (float) sharedCache[0];
}

/* main program */
int main(void) {
    cudaError_t errorCode;

    int gpuId = 0;
    scanf("%d", &gpuId);
    if (cudaSuccess != (errorCode = cudaSetDevice(gpuId))) {
        printf("[Error] %s\n", cudaGetErrorString(errorCode));
        exit(1);
    }

    int threadsPerBlock;
    scanf("%d", &threadsPerBlock);
    if (threadsPerBlock > 1024) {
        printf("[Error] The number of threads per block must be less than 1024!\n");
        exit(1);
    } else if (threadsPerBlock <= 0) {
        printf("[Error] The number of threads per block must be positive!\n");
        exit(1);
    } else if ((threadsPerBlock & (threadsPerBlock - 1)) != 0) {
        printf("[Error] The number of threads per block must be a power of 2!\n");
        exit(1);
    } else {
        printf("Number of threads per block: %d\n", threadsPerBlock);
    }

    int blocksPerGrid;
    scanf("%d", &blocksPerGrid);
    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks per grid must be less than 2147483647!\n");
        exit(1);
    } else if (blocksPerGrid <= 0) {
        printf("The number of blocks per grid must be positive!\n");
        exit(1);
    } else {
        printf("Number of blocks per grid: %d\n", blocksPerGrid);
    }

    /* allocate pinned host memory for diagonal and partial output */
    int N = 1 << 30;
    printf("Size of diagonal: %d\n", N);

    float *hostDiagonal, *hostPartialOutput;
    if (cudaSuccess != (errorCode = cudaMallocHost((void**) &hostDiagonal, N * sizeof(float))) ||
        cudaSuccess != (errorCode = cudaMallocHost((void**) &hostPartialOutput, blocksPerGrid * sizeof(float)))) {
        printf("[Error] %s\n", cudaGetErrorString(errorCode));
        exit(1);
    }

    /* allocate and initialize matrix, then extract the diagonal */
    // int M = N * N;
    // float *hostMatrix = (float*) malloc(M * sizeof(float));
    // randomInit(hostMatrix, M);

    // for (int i = 0; i < N; i++) {
    //     hostDiagonal[i] = hostMatrix[i * N + i];
    // }
    // free(hostMatrix);

    /* allocate diagonal and initialize */
    randomInit(hostDiagonal, N);

    /* allocate device memory */
    float *deviceDiagonal, *devicePartialOutput;
    if (cudaSuccess != (errorCode = cudaMalloc((void**) &deviceDiagonal, N * sizeof(float))) ||
        cudaSuccess != (errorCode = cudaMalloc((void**) &devicePartialOutput, blocksPerGrid * sizeof(float)))) {
        printf("[Error] %s\n", cudaGetErrorString(errorCode));
        exit(1);
    }

    /* create GPU timer events */
    cudaEvent_t gpuStart, gpuEnd;
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuEnd);

    /* copy the diagonal data from the host to the device */
    cudaEventRecord(gpuStart, 0);
    cudaMemcpy(deviceDiagonal, hostDiagonal, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(gpuEnd, 0);
    cudaEventSynchronize(gpuEnd);

    float host2DeviceTime;
    cudaEventElapsedTime(&host2DeviceTime, gpuStart, gpuEnd);
    printf("Time for data transfer from host to device: %f ms\n", host2DeviceTime);

    /* launch kernel */
    int sharedMemSize = threadsPerBlock * sizeof(float);
    cudaEventRecord(gpuStart, 0);
    matrixTrace<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(deviceDiagonal, devicePartialOutput, N);
    cudaEventRecord(gpuEnd, 0);
    cudaEventSynchronize(gpuEnd);

    float kernelTime;
    cudaEventElapsedTime(&kernelTime, gpuStart, gpuEnd);
    printf("Time for kernel execution: %f ms\n", kernelTime);

    /* copy the result from the device to the host */
    cudaEventRecord(gpuStart, 0);
    cudaMemcpy(hostPartialOutput, devicePartialOutput, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(gpuEnd, 0);
    cudaEventSynchronize(gpuEnd);

    float device2HostTime;
    cudaEventElapsedTime(&device2HostTime, gpuStart, gpuEnd);
    printf("Time for data transfer from device to host: %f ms\n", device2HostTime);

    /* clean up GPU memory */
    cudaFree(deviceDiagonal);
    cudaFree(devicePartialOutput);

    /* destroy GPU timers */
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuEnd);

    /* sum up the partial outputs */
    struct timespec cpuStart, cpuEnd;

    clock_gettime(CLOCK_MONOTONIC, &cpuStart);
    double gpuResult = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        gpuResult += (double) hostPartialOutput[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &cpuEnd);

    float mergeTime = (cpuEnd.tv_sec - cpuStart.tv_sec) * 1000.0 + (cpuEnd.tv_nsec - cpuStart.tv_nsec) / 1000000.0;
    printf("Time for merging partial outputs: %f ms\n", mergeTime);

    float gpuTime = host2DeviceTime + kernelTime + device2HostTime + mergeTime;
    printf("Time for GPU execution: %f ms\n", gpuTime);

    /* CPU baseline for correctness check */
    double cpuResult = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &cpuStart);
    for (int i = 0; i < N; i++)
        cpuResult += (double) hostDiagonal[i];
    clock_gettime(CLOCK_MONOTONIC, &cpuEnd);

    float cpuTime = (cpuEnd.tv_sec - cpuStart.tv_sec) * 1000.0 + (cpuEnd.tv_nsec - cpuStart.tv_nsec) / 1000000.0;
    printf("Time for CPU execution: %f ms\n", cpuTime);

    /* calculate speedup and GFLOPS */
    printf("GPU speedup with data transfer: %.2fx\n", cpuTime / gpuTime);
    printf("GPU speedup without data transfer: %.2fx\n", cpuTime / (kernelTime + mergeTime));

    printf("GFLOPS of GPU: %f\n", N / (1000000.0 * kernelTime));
    printf("GFLOPS of CPU: %f\n", N / (1000000.0 * cpuTime));

    /* accuracy check */
    double diff = fabs((cpuResult - gpuResult) / cpuResult);
    printf("|(CPU - GPU)/CPU| = %.15e\n\n", diff);

    /* free memory */
    cudaFreeHost(hostDiagonal);
    cudaFreeHost(hostPartialOutput);

    cudaDeviceReset();
    return 0;
}
