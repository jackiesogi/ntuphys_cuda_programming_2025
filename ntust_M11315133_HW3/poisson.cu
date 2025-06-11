/*
    Poisson equation calculation

    Compile with (for GTX1060):
        nvcc -arch=compute_61 -code=sm_61 -O2 -m64 -o poisson poisson.cu
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <map>
#include <vector>

#define BLOCK_SIZE 8

// CUDA kernel to perform one Jacobi iteration step to solve ∇²φ = -ρ
__global__ void compute_phi(float* phi_new_d, const float* phi_old_d, const float* rho_d, int L) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    // Only process interior grid points (avoid boundaries at 0 and L-1)
    if (i < L - 1 && j < L - 1 && k < L - 1) {
        
        int idx = i * L * L + j * L + k;

        // Compute indices of 6 direct neighbors (±x, ±y, ±z)
        int xm = idx - 1, xp = idx + 1;
        int ym = idx - L, yp = idx + L;
        int zm = idx - L * L, zp = idx + L * L;

        // Jacobi update:
        phi_new_d[idx] = (
            phi_old_d[xm] + phi_old_d[xp] +
            phi_old_d[ym] + phi_old_d[yp] +
            phi_old_d[zm] + phi_old_d[zp] +
            rho_d[idx]
        ) / 6.0f;
    }
}

std::map<int, float> log_states(const float* phi, int L) {
    std::map<int, std::vector<float>> r_to_phis;

    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < L; ++k) {
                int idx = i * L * L + j * L + k;
                int dx = i - L / 2;
                int dy = j - L / 2;
                int dz = k - L / 2;
                int r = static_cast<int>(round(sqrt(dx * dx + dy * dy + dz * dz)));
                if (r > 0)
                    r_to_phis[r].push_back(phi[idx]);
            }
        }
    }

    std::map<int, float> r_to_avg_phi;
    for (const auto& pair : r_to_phis) {
        int r = pair.first;
        const std::vector<float>& phis = pair.second;
        float sum = 0.0f;
        for (float phi : phis)
            sum += phi;
        r_to_avg_phi[r] = sum / phis.size();
    }

    return r_to_avg_phi;
}

/* Main program */
int main(void) {
    cudaError_t error_code;

    int gpu_id = 0;
    scanf("%d", &gpu_id);
    if (cudaSuccess != (error_code = cudaSetDevice(gpu_id))) {
        printf("[Error] %s\n", cudaGetErrorString(error_code));
        exit(1);
    }

    int L;
    scanf("%d", &L);
    printf("Cube size: %d * %d * %d\n", L, L, L);
    int size = L * L * L;

    /* Allocate pinned host memory */
    float *phi_new_h, *phi_old_h, *rho_h;
    if (cudaSuccess != (error_code = cudaMallocHost((void**)&phi_new_h, size * sizeof(float))) ||
        cudaSuccess != (error_code = cudaMallocHost((void**)&phi_old_h, size * sizeof(float))) ||
        cudaSuccess != (error_code = cudaMallocHost((void**)&rho_h, size * sizeof(float)))) {
        printf("[Error] %s\n", cudaGetErrorString(error_code));
        exit(1);
    }

    /* Initialize all to zero, except center charge */
    for (int i = 0; i < size; i++) {
        phi_new_h[i] = 0.0f;
        phi_old_h[i] = 0.0f;
        rho_h[i] = 0.0f;
    }

    /* Set center charge to 1.0 */
    int center = (L / 2) * L * L + (L / 2) * L + (L / 2);
    rho_h[center] = 1.0f;

    /* Allocate device memory */
    float *phi_new_d, *phi_old_d, *rho_d;
    if (cudaSuccess != (error_code = cudaMalloc((void**)&phi_new_d, size * sizeof(float))) ||
        cudaSuccess != (error_code = cudaMalloc((void**)&phi_old_d, size * sizeof(float))) ||
        cudaSuccess != (error_code = cudaMalloc((void**)&rho_d, size * sizeof(float)))) {
        printf("[Error] %s\n", cudaGetErrorString(error_code));
        exit(1);
    }

    /* Create GPU timer events */
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    /* Copy phi and rho from host to device */
    cudaEventRecord(gpu_start, 0);
    cudaMemcpy(phi_new_d, phi_new_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(phi_old_d, phi_old_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_d, rho_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(gpu_end, 0);
    cudaEventSynchronize(gpu_end);

    float host_to_device_time;
    cudaEventElapsedTime(&host_to_device_time, gpu_start, gpu_end);
    printf("Host to device time: %f ms\n", host_to_device_time);

    /* Launch kernel */
    int iterations = L * L;
    printf("Iterations: %d\n", iterations);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((L + (block.x - 1)) / block.x, (L + (block.y - 1)) / block.y, (L + (block.z - 1)) / block.z);

    cudaEventRecord(gpu_start, 0);
    for (int i = 0; i < iterations; i++) {
        compute_phi<<<grid, block>>>(phi_new_d, phi_old_d, rho_d, L);
        cudaDeviceSynchronize();
        std::swap(phi_new_d, phi_old_d);
    }
    cudaEventRecord(gpu_end, 0);
    cudaEventSynchronize(gpu_end);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, gpu_start, gpu_end);
    printf("Kernel execution time: %f ms\n", kernel_time);

    /* Copy result from device to host */
    cudaEventRecord(gpu_start, 0);
    cudaMemcpy(phi_new_h, phi_new_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(gpu_end, 0);
    cudaEventSynchronize(gpu_end);

    float device_to_host_time;
    cudaEventElapsedTime(&device_to_host_time, gpu_start, gpu_end);
    printf("Device to host time: %f ms\n", device_to_host_time);

    /* Clean up GPU memory */
    cudaFree(phi_new_d);
    cudaFree(phi_old_d);
    cudaFree(rho_d);

    /* Destroy GPU timers */
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);

    /* Collect statistics from GPU results */
    auto device_stats = log_states(phi_new_h, L);

    /* CPU baseline for correctness check */
    for (int i = 0; i < size; i++) {
        phi_new_h[i] = 0.0f;
        phi_old_h[i] = 0.0f;
        rho_h[i] = 0.0f;
    }
    rho_h[center] = 1.0f;

    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    for (int it = 0; it < iterations; ++it) {
        for (int i = 1; i < L - 1; ++i) {
            for (int j = 1; j < L - 1; ++j) {
                for (int k = 1; k < L - 1; ++k) {
                    int idx = i * L * L + j * L + k;
                    int xm = idx - 1, xp = idx + 1;
                    int ym = idx - L, yp = idx + L;
                    int zm = idx - L * L, zp = idx + L * L;

                    phi_new_h[idx] = 0.1667f * (
                        phi_old_h[xm] + phi_old_h[xp] +
                        phi_old_h[ym] + phi_old_h[yp] +
                        phi_old_h[zm] + phi_old_h[zp] +
                        rho_h[idx]
                    );
                }
            }
        }
        std::swap(phi_new_h, phi_old_h);
    }
    cpu_end = clock();

    float cpu_time = (cpu_end - cpu_start) / (float)CLOCKS_PER_SEC * 1000.0f;
    printf("CPU execution time: %f ms\n", cpu_time);

    /* Calculate speedup and GFLOPS */
    printf("GPU speedup with data transfer: %.2fx\n", cpu_time / kernel_time);
    printf("GPU speedup without data transfer: %.2fx\n", cpu_time / (host_to_device_time +kernel_time + device_to_host_time));

    printf("GFLOPS of GPU: %f\n", L * L * L * 7.0f * iterations / (1000000.0f * kernel_time));
    printf("GFLOPS of CPU: %f\n", L * L * L * 7.0f * iterations / (1000000.0f * cpu_time));

    /* Output results */
    for (const auto& pair : device_stats) {
        int r = pair.first;
        float avg_phi = pair.second;
        printf("r = %d, avg_phi = %f\n", r, avg_phi);
    }
    printf("\n");

    FILE* fout = fopen("phi_output.txt", "w");
    for (int i = 0; i < size; ++i) {
        fprintf(fout, "%f\n", phi_new_h[i]);
    }
    fclose(fout);

    /* Free memory */
    cudaFreeHost(phi_new_h);
    cudaFreeHost(phi_old_h);
    cudaFreeHost(rho_h);

    cudaDeviceReset();

    return 0;
}
