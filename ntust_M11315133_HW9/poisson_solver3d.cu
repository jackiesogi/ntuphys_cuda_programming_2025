// ------------------- poisson_solver3d.cu -------------------

#include <cstdio>
#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>

using complexd = cufftDoubleComplex;

__device__ inline int freq_to_index(int idx, int dim)
{
    return (idx <= dim / 2) ? idx : (idx - dim);
}

// 在頻率空間套用離散 Green 函數
__global__ void ApplyGreenFunction3D(complexd *freq_data, int dim)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size_t(dim) * dim * dim)
        return;

    int area = dim * dim;
    int z = tid / area;
    int rem = tid % area;
    int y = rem / dim;
    int x = rem % dim;

    int fx = freq_to_index(x, dim);
    int fy = freq_to_index(y, dim);
    int fz = freq_to_index(z, dim);

    double wx = 2.0 * M_PI * (double)fx / (double)dim;
    double wy = 2.0 * M_PI * (double)fy / (double)dim;
    double wz = 2.0 * M_PI * (double)fz / (double)dim;
    double denominator = 2.0 * (cos(wx) - 1.0) + 2.0 * (cos(wy) - 1.0) + 2.0 * (cos(wz) - 1.0);

    complexd ρk = freq_data[tid];
    if (fx == 0 && fy == 0 && fz == 0)
    {
        // DC 組件設為零
        freq_data[tid].x = 0.0;
        freq_data[tid].y = 0.0;
    }
    else
    {
        double inv_denom = -1.0 / denominator;
        freq_data[tid].x = ρk.x * inv_denom;
        freq_data[tid].y = ρk.y * inv_denom;
    }
}

// 對反向 FFT 結果做正規化
__global__ void NormalizeIFFT(complexd *data, int dim)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = size_t(dim) * dim * dim;
    if (tid >= total)
        return;

    double factor = 1.0 / ((double)dim * dim * dim);
    data[tid].x *= factor;
    data[tid].y *= factor;
}

// ---------------------------------------------------
// 主程式區段（Host Side）
// ---------------------------------------------------
int main()
{
    int dim;
    printf("Input grid dimension dim (dim x dim x dim): ");
    scanf("%d", &dim);

    size_t total_voxels = size_t(dim) * dim * dim;

    // 1) 配置 pinned host 記憶體
    complexd *h_density = nullptr, *h_potential = nullptr;
    if (cudaHostAlloc((void **)&h_density, sizeof(complexd) * total_voxels, cudaHostAllocDefault) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate host memory for density\n");
        return EXIT_FAILURE;
    }
    if (cudaHostAlloc((void **)&h_potential, sizeof(complexd) * total_voxels, cudaHostAllocDefault) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate host memory for potential\n");
        cudaFreeHost(h_density);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < total_voxels; ++i)
    {
        h_density[i].x = 0.0;
        h_density[i].y = 0.0;
    }
    // 在原點設一個單位點電荷
    h_density[0].x = 1.0;
    h_density[0].y = 0.0;

    // 2) 配置 GPU 裝置記憶體
    complexd *d_volume = nullptr;
    cudaMalloc((void **)&d_volume, sizeof(complexd) * total_voxels);

    // 傳送密度資料到 GPU
    cudaMemcpy(d_volume, h_density, sizeof(complexd) * total_voxels, cudaMemcpyHostToDevice);

    // 3) 建立 cuFFT 計劃（3D 雙精度）
    cufftHandle plan_forward, plan_inverse;
    if (cufftPlan3d(&plan_forward, dim, dim, dim, CUFFT_Z2Z) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cuFFT forward plan creation failed\n");
        return EXIT_FAILURE;
    }
    if (cufftPlan3d(&plan_inverse, dim, dim, dim, CUFFT_Z2Z) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cuFFT inverse plan creation failed\n");
        return EXIT_FAILURE;
    }

    // 4) 執行前向傅立葉轉換：密度 → 頻域
    cufftExecZ2Z(plan_forward, d_volume, d_volume, CUFFT_FORWARD);

    // 5) 套用 Green 函數解頻域方程（就地更新）
    int threadsPerBlock = 256;
    int totalBlocks = (int)((total_voxels + threadsPerBlock - 1) / threadsPerBlock);
    ApplyGreenFunction3D<<<totalBlocks, threadsPerBlock>>>(d_volume, dim);
    cudaDeviceSynchronize();

    // 6) 執行反向傅立葉轉換：頻域 → 潛勢
    cufftExecZ2Z(plan_inverse, d_volume, d_volume, CUFFT_INVERSE);

    // 7) 對結果做縮放正規化
    NormalizeIFFT<<<totalBlocks, threadsPerBlock>>>(d_volume, dim);
    cudaDeviceSynchronize();

    // 8) 從 GPU 把潛勢資料取回
    cudaMemcpy(h_potential, d_volume, sizeof(complexd) * total_voxels, cudaMemcpyDeviceToHost);

    // 9) 印出對角線與 x 軸方向上的潛勢值
    printf("  n   φ(n,n,n)             φ(n,0,0)\n");
    for (int n = 0; n < dim; ++n)
    {
        size_t id_diag = size_t(n) * (1 + dim + (size_t)dim * dim);
        size_t id_xaxis = size_t(n);
        double φ_diag = h_potential[id_diag].x;
        double φ_x = h_potential[id_xaxis].x;
        printf("%2d  % .8e   % .8e\n", n, φ_diag, φ_x);
    }

    // 10) 計算全域平均（應接近 0）
    double sum = 0.0;
    for (size_t i = 0; i < total_voxels; ++i)
    {
        sum += h_potential[i].x;
    }
    printf("\nAverage φ over grid = % .3e  (should be ≈ 0)\n\n", sum / double(total_voxels));

    // 11) 資源釋放
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cudaFree(d_volume);
    cudaFreeHost(h_density);
    cudaFreeHost(h_potential);

    return EXIT_SUCCESS;
}

