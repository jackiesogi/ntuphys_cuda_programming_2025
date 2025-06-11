#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Variables
float *hostData;           // host vectors
unsigned int *hostHist;    // GPU solution back to the CPU
float *deviceData;         // device vectors
unsigned int *deviceHist;
unsigned int *cpuHist;     // CPU solution

// Functions
void InitRandomData(float *, long);
void InitRandomNormal(float *, long);
void InitRandomExp(float *, long);

__global__ void hist_smem(float *inputData, const long dataSize, unsigned int *histogram,
                          const int numBins, const float minRange, const float binWidth)
{

  // use shared memory and atomic addition

  extern __shared__ unsigned int sharedHist[]; // assume block size == numBins
  sharedHist[threadIdx.x] = 0;
  __syncthreads();

  long idx = threadIdx.x + blockIdx.x * blockDim.x;
  long step = blockDim.x * gridDim.x;

  while (idx < dataSize)
  {
    int binIndex = (int)((inputData[idx] - minRange) / binWidth);
    atomicAdd(&sharedHist[binIndex], 1);
    idx += step;
  }

  __syncthreads();
  atomicAdd(&(histogram[threadIdx.x]), sharedHist[threadIdx.x]);
}

int main(void)
{

  int gpuId;

  // Error code to check return values for CUDA calls
  cudaError_t cudaStatus = cudaSuccess;

  scanf("%d", &gpuId);
  cudaStatus = cudaSetDevice(gpuId);
  if (cudaStatus != cudaSuccess)
  {
    printf("!!! Cannot select GPU with device ID = %d\n", gpuId);
    exit(1);
  }
  printf("Set GPU with device ID = %d\n", gpuId);

  cudaSetDevice(gpuId);

  printf("To find the histogram of a data set (with real numbers): \n");
  long dataLength;
  int bins, binIndex;
  float rangeMin, rangeMax, binSize;

  printf("Enter the size of the data vector: ");
  scanf("%ld", &dataLength);
  printf("%ld\n", dataLength);
  long dataBytes = dataLength * sizeof(float);

  printf("Enter the data range [min, max] for the histogram: ");
  scanf("%f %f", &rangeMin, &rangeMax);
  printf("%f %f\n", rangeMin, rangeMax);

  printf("Enter the number of bins of the histogram: ");
  scanf("%d", &bins);
  printf("%d\n", bins);
  if (bins > 1024)
  {
    printf("The number of bins is set to # of threads per block < 1024 ! \n");
    exit(0);
  }
  int histBytes = bins * sizeof(int);
  binSize = (rangeMax - rangeMin) / (float)bins;

  hostData = (float *)malloc(dataBytes);
  hostHist = (unsigned int *)malloc(histBytes);

  // Check memory allocations
  if (hostData == NULL || hostHist == NULL)
  {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < bins; i++)
    hostHist[i] = 0;

  // initialize the data_h vector
  srand(time(NULL)); // initialize the seed with the current time
  srand(12345);

  //    InitRandomData(hostData, dataLength);      // uniform deviate in (0,1)
  //    InitRandomNormal(hostData, dataLength);    // Gaussian deviate with sigma = 1
  InitRandomExp(hostData, dataLength); // Exponential Distribution

  int threadsPerBlock;
  printf("Enter the number of threads per block: ");
  scanf("%d", &threadsPerBlock);
  printf("%d\n", threadsPerBlock);
  if (threadsPerBlock != bins)
  {
    printf("The number of threads per block must be equal to the number of bins ! \n");
    exit(0);
  }
  fflush(stdout);

  int blocksPerGrid;
  printf("Enter the number of blocks per grid: ");
  scanf("%d", &blocksPerGrid);
  printf("%d\n", blocksPerGrid);
  if (blocksPerGrid > 2147483647)
  {
    printf("The number of blocks must be less than 2147483647 ! \n");
    exit(0);
  }
  printf("The number of blocks is %d\n", blocksPerGrid);
  fflush(stdout);

  int runCPU;
  printf("To compute the histogram with CPU (1/0) ? ");
  scanf("%d", &runCPU);
  printf("%d\n", runCPU);
  fflush(stdout);

  // create the timer
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  // start the timer
  cudaEventRecord(startEvent, 0);

  // Allocate vectors in device memory

  cudaMalloc((void **)&deviceHist, histBytes);
  cudaMalloc((void **)&deviceData, dataBytes);

  // Copy vectors from host memory to device memory

  cudaMemcpy(deviceData, hostData, dataBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceHist, hostHist, histBytes, cudaMemcpyHostToDevice);

  // stop the timer
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);

  float inputTime;
  cudaEventElapsedTime(&inputTime, startEvent, stopEvent);
  printf("Input time for GPU: %f (ms) \n", inputTime);

  // start the timer
  cudaEventRecord(startEvent, 0);

  int sharedMemSize = threadsPerBlock * sizeof(int);

  hist_smem<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(deviceData, dataLength, deviceHist, bins, rangeMin, binSize);

  // stop the timer
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);

  float gpuProcessTime;
  cudaEventElapsedTime(&gpuProcessTime, startEvent, stopEvent);
  printf("Processing time for GPU: %f (ms) \n", gpuProcessTime);
  printf("GPU Gflops: %f\n", 2 * dataLength / (1000000.0 * gpuProcessTime));

  // Copy result from device memory to host memory
  // hostHist contains the result in host memory

  // start the timer
  cudaEventRecord(startEvent, 0);

  cudaMemcpy(hostHist, deviceHist, histBytes, cudaMemcpyDeviceToHost);

  cudaFree(deviceData);
  cudaFree(deviceHist);

  // stop the timer
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);

  float outputTime;
  cudaEventElapsedTime(&outputTime, startEvent, stopEvent);
  printf("Output time for GPU: %f (ms) \n", outputTime);

  float totalGPUTime = inputTime + gpuProcessTime + outputTime;
  printf("Total time for GPU: %f (ms) \n", totalGPUTime);

  FILE *outfile;
  outfile = fopen("histogram_sharedmem.txt", "w");

  for (int i = 0; i < bins; i++)
  {
    float center = rangeMin + (i + 0.5f) * binSize;
    fprintf(outfile, "%f %d \n", center, hostHist[i]);
  }
  fclose(outfile);

  printf("Histogram (GPU):\n");
  for (int i = 0; i < bins; i++)
  {
    float center = rangeMin + (i + 0.5f) * binSize;
    printf("%f %d \n", center, hostHist[i]);
  }

  if (runCPU == 0)
  {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    free(hostData);
    free(hostHist);
    return 0;
  }

  // To compute the CPU reference solution

  cpuHist = (unsigned int *)malloc(histBytes);
  for (int i = 0; i < bins; i++)
    cpuHist[i] = 0;

  // start the timer
  cudaEventRecord(startEvent, 0);

  for (int i = 0; i < dataLength; i++)
  {
    binIndex = (int)((hostData[i] - rangeMin) / binSize);
    if ((binIndex > bins - 1) || (binIndex < 0))
    {
      printf("hostData[%d]=%f, binIndex=%d\n", i, hostData[i], binIndex);
      exit(0);
    }
    cpuHist[binIndex]++;
  }

  // stop the timer
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);

  float cpuTime;
  cudaEventElapsedTime(&cpuTime, startEvent, stopEvent);
  printf("Processing time for CPU: %f (ms) \n", cpuTime);
  printf("CPU Gflops: %f\n", 2 * dataLength / (1000000.0 * cpuTime));
  printf("Speed up of GPU = %f\n", cpuTime / totalGPUTime);

  // destroy the timer
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

  // check histogram sum equal to the total number of data

  int totalCount = 0;
  for (int i = 0; i < bins; i++)
  {
    totalCount += cpuHist[i];
  }
  if (totalCount != dataLength)
  {
    printf("Error, sum = %d\n", totalCount);
    exit(0);
  }

  // compare histograms from CPU and GPU

  for (int i = 0; i < bins; i++)
  {
    if (hostHist[i] != cpuHist[i])
      printf("i=%d, hostHist=%d, cpuHist=%d \n", i, hostHist[i], cpuHist[i]);
  }

  FILE *outfile_cpu;
  outfile_cpu = fopen("histogram_cpu.txt", "w");

  for (int i = 0; i < bins; i++)
  {
    float center = rangeMin + (i + 0.5f) * binSize;
    fprintf(outfile_cpu, "%f %d \n", center, cpuHist[i]);
  }
  fclose(outfile_cpu);

  printf("Histogram (CPU):\n");
  for (int i = 0; i < bins; i++)
  {
    float center = rangeMin + (i + 0.5f) * binSize;
    printf("%f %d \n", center, cpuHist[i]);
  }

  free(hostData);
  free(hostHist);
  free(cpuHist);

  return 0;
}

void InitRandomData(float *data, long n) // RNG with uniform distribution in (0,1)
{
  for (long i = 0; i < n; i++)
    data[i] = rand() / (float)RAND_MAX;
}

void InitRandomExp(float *data, long n)
{
  for (long i = 0; i < n; i++)
  {
    double y = (double)rand() / (float)RAND_MAX;
    double x = -log(1.0 - y);
    data[i] = (float)(x);
  }
}

void InitRandomNormal(float *data, long n) // RNG with normal distribution, mu=0, sigma=1
{
  const float Pi = acos(-1.0);

  for (long i = 0; i < n; i++)
  {
    double y = (double)rand() / (float)RAND_MAX;
    double x = -log(1.0 - y);
    double z = (double)rand() / (float)RAND_MAX;
    double theta = 2.0 * Pi * z;
    data[i] = (float)(sqrt(2.0 * x) * cos(theta));
  }
}

