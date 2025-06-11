// To compute histogram with atomic operations
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Variables
float *data_h; // host vectors
unsigned int *hist_h;
float *data_d; // device vectors
unsigned int *hist_d;
unsigned int *hist_c; // CPU solution

// Functions
void RandomUniform(float *, long);
void RandomNormal(float *, long);
void RandomExp(float *, long);

__global__ void hist_gmem(float *data, const long N, unsigned int *hist,
                          const int bins, const float Rmin, const float binsize)
{
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long stride = blockDim.x * gridDim.x;

    while (i < N)
    {
        int index = (int)((data[i] - Rmin) / binsize);
        if (index >= 0 && index < bins)
        {
            atomicAdd(&hist[index], 1);
        }
        i += stride;
    }
}

int main(void)
{
    int gid;
    cudaError_t err = cudaSuccess;

    scanf("%d", &gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess)
    {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    printf("To find the histogram of a data set (with real numbers): \n");

    long N;
    int bins;
    float Rmin, Rmax, binsize;

    printf("Enter the size of the data vector: ");
    scanf("%ld", &N);
    printf("%ld\n", N);
    long size = N * sizeof(float);

    printf("Enter the data range [Rmin, Rmax] for the histogram: ");
    scanf("%f %f", &Rmin, &Rmax);
    printf("%f %f\n", Rmin, Rmax);

    printf("Enter the number of bins of the histogram: ");
    scanf("%d", &bins);
    printf("%d\n", bins);
    int bsize = bins * sizeof(int);
    binsize = (Rmax - Rmin) / (float)bins;

    data_h = (float *)malloc(size);
    hist_h = (unsigned int *)malloc(bsize);

    if (data_h == NULL || hist_h == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < bins; i++)
        hist_h[i] = 0;

    srand(12345);
    printf("Starting to generate data by RNG\n");

    // Choose one of the following RNGs:
    // RandomUniform(data_h, N);
    // RandomNormal(data_h, N);
    RandomExp(data_h, N);

    printf("Finish the generation of data\n");

    int threadsPerBlock;
    printf("Enter the number of threads per block: ");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if (threadsPerBlock > 1024)
    {
        printf("The number of threads per block must be <= 1024\n");
        exit(0);
    }

    int blocksPerGrid;
    printf("Enter the number of blocks per grid: ");
    scanf("%d", &blocksPerGrid);
    printf("%d\n", blocksPerGrid);
    if (blocksPerGrid > 2147483647)
    {
        printf("The number of blocks must be < 2147483647\n");
        exit(0);
    }

    int CPU;
    printf("To compute the histogram with CPU (1/0)? ");
    scanf("%d", &CPU);
    printf("%d\n", CPU);

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory
    cudaMalloc((void **)&data_d, size);
    cudaMalloc((void **)&hist_d, bsize);

    cudaEventRecord(start, 0);

    // Copy to device
    cudaMemcpy(data_d, data_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(hist_d, hist_h, bsize, cudaMemcpyHostToDevice);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float Intime;
    cudaEventElapsedTime(&Intime, start, stop);
    printf("Input time for GPU: %f (ms)\n", Intime);

    // Launch kernel
    cudaEventRecord(start, 0);
    hist_gmem<<<blocksPerGrid, threadsPerBlock>>>(data_d, N, hist_d, bins, Rmin, binsize);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gputime;
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Processing time for GPU: %f (ms)\n", gputime);
    printf("GPU Gflops: %f\n", 2 * N / (1000000.0 * gputime));

    // Copy result back to host
    cudaEventRecord(start, 0);
    cudaMemcpy(hist_h, hist_d, bsize, cudaMemcpyDeviceToHost);
    cudaFree(data_d);
    cudaFree(hist_d);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float Outime;
    cudaEventElapsedTime(&Outime, start, stop);
    float gputime_tot = Intime + gputime + Outime;

    printf("Output time for GPU: %f (ms)\n", Outime);
    printf("Total time for GPU: %f (ms)\n", gputime_tot);

    // Write GPU result
    FILE *out = fopen("hist_gmem.txt", "w");
    for (int i = 0; i < bins; i++)
    {
        float x = Rmin + (i + 0.5f) * binsize;
        fprintf(out, "%f %d\n", x, hist_h[i]);
    }
    fclose(out);

    printf("Histogram (GPU):\n");
    for (int i = 0; i < bins; i++)
    {
        float x = Rmin + (i + 0.5f) * binsize;
        printf("%f %d\n", x, hist_h[i]);
    }

    if (CPU == 0)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(data_h);
        free(hist_h);
        return 0;
    }

    // CPU histogram
    hist_c = (unsigned int *)malloc(bsize);
    for (int i = 0; i < bins; i++)
        hist_c[i] = 0;

    cudaEventRecord(start, 0);
    for (int i = 0; i < N; i++)
    {
        int index = (int)((data_h[i] - Rmin) / binsize);
        if (index >= 0 && index < bins)
        {
            hist_c[index]++;
        }
        else
        {
            printf("data[%d]=%f, index=%d out of range\n", i, data_h[i], index);
            exit(0);
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime(&cputime, start, stop);
    printf("Processing time for CPU: %f (ms)\n", cputime);
    printf("CPU Gflops: %f\n", 2 * N / (1000000.0 * cputime));
    printf("Speed up of GPU = %f\n", cputime / gputime_tot);

    // Compare results
    int sum = 0;
    for (int i = 0; i < bins; i++)
        sum += hist_c[i];
    if (sum != N)
    {
        printf("Error: histogram sum = %d != N = %ld\n", sum, N);
        exit(0);
    }

    for (int i = 0; i < bins; i++)
    {
        if (hist_h[i] != hist_c[i])
            printf("Mismatch at bin %d: GPU=%d, CPU=%d\n", i, hist_h[i], hist_c[i]);
    }

    // Save CPU histogram
    FILE *out1 = fopen("hist_cpu.txt", "w");
    for (int i = 0; i < bins; i++)
    {
        float x = Rmin + (i + 0.5f) * binsize;
        fprintf(out1, "%f %d\n", x, hist_c[i]);
    }
    fclose(out1);

    printf("Histogram (CPU):\n");
    for (int i = 0; i < bins; i++)
    {
        float x = Rmin + (i + 0.5f) * binsize;
        printf("%f %d\n", x, hist_c[i]);
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(data_h);
    free(hist_h);
    free(hist_c);

    return 0;
}

void RandomUniform(float *data, long n)
{
    for (long i = 0; i < n; i++)
        data[i] = rand() / (float)RAND_MAX;
}

void RandomExp(float *data, long n)
{
    for (long i = 0; i < n; i++)
    {
        double y = rand() / (double)RAND_MAX;
        data[i] = (float)(-log(1.0 - y));
    }
}

void RandomNormal(float *data, long n)
{
    const float Pi = acos(-1.0f);
    for (long i = 0; i < n; i++)
    {
        double y = rand() / (double)RAND_MAX;
        double x = -log(1.0 - y);
        double z = rand() / (double)RAND_MAX;
        double theta = 2 * Pi * z;
        data[i] = (float)(sqrt(2.0 * x) * cos(theta));
    }
}

