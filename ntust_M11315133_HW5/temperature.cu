/*
    Thermal equilibrium temperature distribution with N GPUs

    Compile with (for GTX1060):
        nvcc -Xcompiler -fopenmp -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 \
            -o temperatureDistribution temperatureDistribution.cu
*/

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define GRID_SIZE 1024
#define MAX_ITERATIONS 100000
#define ERROR_TOLERANCE 1e-3f

__global__ void temperature_update_kernel(float* new_temperature_d, const float* old_temperature_d, int start_index) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    int row = global_idx / GRID_SIZE;
    int col = global_idx % GRID_SIZE;

    if (row > 0 && row < GRID_SIZE - 1 && col > 0 && col < GRID_SIZE - 1) {
        new_temperature_d[global_idx] = 0.25f * (old_temperature_d[global_idx - 1] + old_temperature_d[global_idx + 1] +
                                                old_temperature_d[global_idx - GRID_SIZE] + old_temperature_d[global_idx + GRID_SIZE]);
    }
}

int main() {
    int number_of_gpus;
    scanf("%d", &number_of_gpus);

    int* gpu_device_ids = (int*)malloc(sizeof(int) * number_of_gpus);
    for (int i = 0; i < number_of_gpus; ++i) {
        scanf("%d", &gpu_device_ids[i]);
    }

    int threads_per_block;
    scanf("%d", &threads_per_block);

    const int total_grid_points = GRID_SIZE * GRID_SIZE;

    float* host_temperature_old;
    float* host_temperature_new;
    cudaMallocHost((void**)&host_temperature_old, total_grid_points * sizeof(float));
    memset(host_temperature_old, 0, total_grid_points * sizeof(float));
    cudaMallocHost((void**)&host_temperature_new, total_grid_points * sizeof(float));
    memset(host_temperature_new, 0, total_grid_points * sizeof(float));

    /* Initialize boundary temperature */
    for (int i = 0; i < GRID_SIZE; ++i) {
        host_temperature_old[i * GRID_SIZE] = host_temperature_new[i * GRID_SIZE] = 273.0f;
        host_temperature_old[(GRID_SIZE - 1) + i * GRID_SIZE] = host_temperature_new[(GRID_SIZE - 1) + i * GRID_SIZE] = 273.0f;
        host_temperature_old[i + (GRID_SIZE - 1) * GRID_SIZE] = host_temperature_new[i + (GRID_SIZE - 1) * GRID_SIZE] = 273.0f;
        host_temperature_old[i] = host_temperature_new[i] = 400.0f;
    }

    omp_set_num_threads(number_of_gpus);
    const int points_per_gpu = total_grid_points / number_of_gpus;

    float** device_temperature_old_arr = (float**)malloc(sizeof(float*) * number_of_gpus);
    float** device_temperature_new_arr = (float**)malloc(sizeof(float*) * number_of_gpus);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        cudaSetDevice(gpu_device_ids[thread_id]);
        cudaMalloc((void**)&device_temperature_old_arr[thread_id], total_grid_points * sizeof(float));
        cudaMalloc((void**)&device_temperature_new_arr[thread_id], total_grid_points * sizeof(float));
    }

    float* host_to_device_durations = (float*)calloc(number_of_gpus, sizeof(float));
    float* kernel_durations = (float*)calloc(number_of_gpus, sizeof(float));
    float* device_to_host_durations = (float*)calloc(number_of_gpus, sizeof(float));

    float max_error = 1.0f;
    int iteration_count = 0;

    /* GPU Jacobi iteration */
    while (max_error > ERROR_TOLERANCE && iteration_count < MAX_ITERATIONS) {
        max_error = 0.0f;

        #pragma omp parallel reduction(max:max_error)
        {
            int thread_id = omp_get_thread_num();
            cudaSetDevice(gpu_device_ids[thread_id]);

            cudaEvent_t event_start, event_stop;
            cudaEventCreate(&event_start);
            cudaEventCreate(&event_stop);

            float* device_temperature_old = device_temperature_old_arr[thread_id];
            float* device_temperature_new = device_temperature_new_arr[thread_id];
            float elapsed_time;

            cudaEventRecord(event_start);
            cudaMemcpy(device_temperature_old, host_temperature_old, total_grid_points * sizeof(float), cudaMemcpyHostToDevice);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
            host_to_device_durations[thread_id] += elapsed_time;

            int gpu_offset = thread_id * points_per_gpu;
            int grid_blocks = (points_per_gpu + threads_per_block - 1) / threads_per_block;

            cudaEventRecord(event_start);
            temperature_update_kernel<<<grid_blocks, threads_per_block>>>(device_temperature_new, device_temperature_old, gpu_offset);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
            kernel_durations[thread_id] += elapsed_time;

            cudaEventRecord(event_start);
            cudaMemcpy(host_temperature_new + gpu_offset, device_temperature_new + gpu_offset, points_per_gpu * sizeof(float), cudaMemcpyDeviceToHost);
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&elapsed_time, event_start, event_stop);
            device_to_host_durations[thread_id] += elapsed_time;

            for (int i = gpu_offset; i < gpu_offset + points_per_gpu; ++i) {
                float difference = fabsf(host_temperature_new[i] - host_temperature_old[i]);
                if (difference > max_error) max_error = difference;
            }

            cudaEventDestroy(event_start);
            cudaEventDestroy(event_stop);
        }

        std::swap(host_temperature_old, host_temperature_new);

        /* Reapply boundary conditions */
        for (int i = 0; i < GRID_SIZE; ++i) {
            host_temperature_old[i * GRID_SIZE] = host_temperature_new[i * GRID_SIZE] = 273.0f;
            host_temperature_old[(GRID_SIZE - 1) + i * GRID_SIZE] = host_temperature_new[(GRID_SIZE - 1) + i * GRID_SIZE] = 273.0f;
            host_temperature_old[i + (GRID_SIZE - 1) * GRID_SIZE] = host_temperature_new[i + (GRID_SIZE - 1) * GRID_SIZE] = 273.0f;
            host_temperature_old[i] = host_temperature_new[i] = 400.0f;
        }

        iteration_count++;
    }

    float max_host_to_device_time = 0.0f, max_kernel_time = 0.0f, max_device_to_host_time = 0.0f;
    for (int i = 0; i < number_of_gpus; ++i) {
        max_host_to_device_time = fmax(max_host_to_device_time, host_to_device_durations[i]);
        max_kernel_time = fmax(max_kernel_time, kernel_durations[i]);
        max_device_to_host_time = fmax(max_device_to_host_time, device_to_host_durations[i]);
    }

    float total_gpu_time = max_host_to_device_time + max_kernel_time + max_device_to_host_time;

    /* Output performance information */
    printf("Number of GPUs: %d\n", number_of_gpus);
    printf("Threads per block: %d\n", threads_per_block);
    printf("Blocks per grid: %d\n", (points_per_gpu + threads_per_block - 1) / threads_per_block);
    printf("Host to device time: %.2f s\n", max_host_to_device_time / 1000.0f);
    printf("Kernel execution time: %.2f s\n", max_kernel_time / 1000.0f);
    printf("Device to host time: %.2f s\n", max_device_to_host_time / 1000.0f);
    printf("Total GPU time: %.2f s\n", total_gpu_time / 1000.0f);
    printf("GFLOPS: %.2f\n\n", 2.0f * total_grid_points * iteration_count / (total_gpu_time * 1e-3f) * 1e-9f);

    /* Clean up */
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        cudaSetDevice(gpu_device_ids[thread_id]);
        cudaFree(device_temperature_old_arr[thread_id]);
        cudaFree(device_temperature_new_arr[thread_id]);
    }

    free(device_temperature_old_arr);
    free(device_temperature_new_arr);
    free(gpu_device_ids);
    free(host_to_device_durations);
    free(kernel_durations);
    free(device_to_host_durations);
    cudaFreeHost(host_temperature_old);
    cudaFreeHost(host_temperature_new);

    return 0;
}

