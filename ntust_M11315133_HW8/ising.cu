#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Device: dimension parameters and weight
static const int D = 10;                          // number of dimensions (n)
static const double c_const = 1.5822955636702438; // weight function normalization constant
static const double a_const = 1.0;                // weight function exponential constant

// Device function f(x):  f(x) = 1.0 / ( sum_{i=0..D-1}( x[i]^2 ) + 1.0 )
__device__ double dev_f(const double *x) {
    double sum = 0.0;
#pragma unroll
    for (int i = 0; i < D; i++) {
        sum += x[i] * x[i];
    }
    return 1.0 / (sum + 1.0);
}

// Device function w(x):  w(x) = ∏_{i=0..D-1} [ c_const * exp( -a_const * x[i] ) ]
__device__ double dev_w(const double *x) {
    double ret = 1.0;
#pragma unroll
    for (int i = 0; i < D; i++)
    {
        // c_const * exp(-a_const * x_i)
        ret *= c_const * exp(-a_const * x[i]);
    }
    return ret;
}

// Simple (plain) Monte Carlo sampling
__global__ void kernel_simple_sampling(unsigned long long dnum_per_gpu,
                                       unsigned long long seed_offset,
                                       double *d_sum_f,
                                       double *d_sum_f2) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(
        /* seed     */ 12345ULL,
        /* subsequence */ (unsigned long long)(tid + seed_offset),
        /* offset   */ 0,
        &state);

    double x[D];

    // Loop over all dnum_per_gpu samples with stride = total_threads
    for (unsigned long long i = tid; i < dnum_per_gpu; i += total_threads)
    {
// Generate D uniform(0,1) in x[0..D-1]
#pragma unroll
        for (int j = 0; j < D; j++)
        {
            x[j] = curand_uniform_double(&state);
        }
        // Evaluate f(x)
        double fx = dev_f(x);

        // Atomically accumulate into global sums
        atomicAdd(d_sum_f, fx);
        atomicAdd(d_sum_f2, fx * fx);
    }
}

// Metropolis importance sampling (one chain, run entirely on 1 thread)
__global__ void kernel_metropolis_sampling(
    unsigned long long dnum_per_gpu,
    unsigned long long seed_offset,
    double *d_sum_fx, // one double per GPU: Σ [ f(x)/w(x) ]
    double *d_sum_fx2 // one double per GPU: Σ [ (f(x)/w(x))^2 ]
)
{
    // Only one thread does everything
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // Initialize curand for a single thread
    curandStatePhilox4_32_10_t state;
    curand_init(
        /* seed       */ 12345ULL,
        /* subsequence */ (unsigned long long)(seed_offset),
        /* offset     */ 0,
        &state);

    // Local arrays on device stack
    double x_old[D], x_new[D], x_curr[D];

    // Initialize x_old[j] ~ U(0,1)
    for (int j = 0; j < D; j++)
    {
        x_old[j] = curand_uniform_double(&state);
    }
    double w_old = dev_w(x_old);

    // Local accumulators
    double sum_fx = 0.0;
    double sum_fx2 = 0.0;

    // Metropolis loop: dnum_per_gpu steps
    for (unsigned long long i = 0; i < dnum_per_gpu; i++)
    {
        // Propose a new uniform point x_new
        for (int j = 0; j < D; j++)
        {
            x_new[j] = curand_uniform_double(&state);
        }
        double w_new = dev_w(x_new);

        // Accept/reject
        bool accept = false;
        if (w_new > w_old)
        {
            accept = true;
        }
        else
        {
            double r = curand_uniform_double(&state);
            if (r < (w_new / w_old))
            {
                accept = true;
            }
        }
        if (accept)
        {
            // copy x_new → x_old
            for (int j = 0; j < D; j++)
            {
                x_old[j] = x_new[j];
            }
            w_old = w_new;
        }
        // Evaluate f(x_old)/w(x_old)
        double fx_over_w = dev_f(x_old) / w_old;

        sum_fx += fx_over_w;
        sum_fx2 += fx_over_w * fx_over_w;
    }

    // Write back into global memory (per-GPU slot)
    atomicAdd(d_sum_fx, sum_fx);
    atomicAdd(d_sum_fx2, sum_fx2);
}

////////////////////////////////////////////////////////////////////////////////
// Host code: set up multi-GPU loops, aggregate results, print final output
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
    int NGx, NGy; // The partition of the lattice (NGx*NGy=NGPU).
    int NGPU;
    int *Dev;    // GPU device numbers.
    double dnum; // Number of samples to be generated.
    printf("  Enter the number of GPUs (NGx, NGy): ");
    scanf("%d %d", &NGx, &NGy);
    printf("%d %d\n", NGx, NGy);
    NGPU = NGx * NGy;
    Dev = (int *)malloc(sizeof(int) * NGPU);
    for (int i = 0; i < NGPU; i++)
    {
        printf("  * Enter the GPU ID (0/1/...): ");
        scanf("%d", &(Dev[i]));
        printf("%d\n", Dev[i]);
    }

    printf("How many samples to be generated ?\n");
    scanf("%lf", &dnum);
    if (dnum == 0ULL)
    {
        fprintf(stderr, "Number of samples must be > 0.\n");
        return EXIT_FAILURE;
    }
    printf("Samples size: %llu\n", (unsigned long long)dnum);
    printf("c = %lf\n", c_const);
    printf("a = %lf\n", a_const);

    // make dnum_d from dnum to integer
    unsigned long long dnum_d = (unsigned long long)dnum;

    // 3) Determine how many samples per GPU (ceil division)
    unsigned long long base = dnum / (unsigned long long)NGPU;
    unsigned long long rem = dnum_d % (unsigned long long)NGPU;
    // We'll assign [ base+1 ] samples to the first 'rem' GPUs, and [ base ] to the rest.
    unsigned long long *dnum_per_gpu = (unsigned long long *)malloc(sizeof(unsigned long long) * NGPU);
    for (int i = 0; i < NGPU; i++)
    {
        dnum_per_gpu[i] = base + ((unsigned long long)i < rem ? 1ULL : 0ULL);
    }

    // Host arrays to gather per-GPU partial sums (for simple and Metropolis)
    double *h_sum_f = (double *)malloc(sizeof(double) * NGPU);
    double *h_sum_f2 = (double *)malloc(sizeof(double) * NGPU);
    double *h_sum_fx = (double *)malloc(sizeof(double) * NGPU);
    double *h_sum_fx2 = (double *)malloc(sizeof(double) * NGPU);

    // Initialize host accumulators to zero
    for (int i = 0; i < NGPU; i++)
    {
        h_sum_f[i] = 0.0;
        h_sum_f2[i] = 0.0;
        h_sum_fx[i] = 0.0;
        h_sum_fx2[i] = 0.0;
    }

// 4) Launch per-GPU work in parallel using OpenMP
#pragma omp parallel num_threads(NGPU)
    {
        int thread_id = omp_get_thread_num();
        int gpu_id = Dev[thread_id];

        // 4.1) Pin this OpenMP thread to the chosen GPU
        cudaSetDevice(gpu_id);

        // 4.2) Allocate device memory for this GPU's partial sums
        double *d_sum_f, *d_sum_f2, *d_sum_fx, *d_sum_fx2;
        cudaMalloc((void **)&d_sum_f, sizeof(double));
        cudaMalloc((void **)&d_sum_f2, sizeof(double));
        cudaMalloc((void **)&d_sum_fx, sizeof(double));
        cudaMalloc((void **)&d_sum_fx2, sizeof(double));

        // Initialize device sums to 0
        cudaMemset(d_sum_f, 0, sizeof(double));
        cudaMemset(d_sum_f2, 0, sizeof(double));
        cudaMemset(d_sum_fx, 0, sizeof(double));
        cudaMemset(d_sum_fx2, 0, sizeof(double));

        // 4.3) SIMPLE SAMPLING: choose a launch configuration
        //      We pick e.g. 256 threads per block and enough blocks so that
        //      (blocks * 256) ≥ min(1024, dnum_per_gpu).
        unsigned long long S = dnum_per_gpu[thread_id];
        int threadsPerBlock = 256;
        unsigned long long minThreads = min((unsigned long long)1024ULL, S);
        int numBlocks = (int)ceil((double)minThreads / (double)threadsPerBlock);
        if (numBlocks < 1)
            numBlocks = 1;

        // Launch the simple-sampling kernel
        kernel_simple_sampling<<<
            numBlocks,
            threadsPerBlock>>>(
            S,                             // dnum_per_gpu for this GPU
            (unsigned long long)thread_id, // seed_offset so each GPU+thread has a distinct subsequence
            d_sum_f,
            d_sum_f2);
        cudaDeviceSynchronize();

        // 4.4) METROPOLIS SAMPLING: single-threaded chain on this GPU
        // Launch <<<1,1>>> so that thread 0 does all S Metropolis steps
        kernel_metropolis_sampling<<<1, 1>>>(
            S,
            (unsigned long long)thread_id, // seed_offset
            d_sum_fx,
            d_sum_fx2);
        cudaDeviceSynchronize();

        // 4.5) Copy back the four doubles to host arrays
        cudaMemcpy(&h_sum_f[thread_id], d_sum_f, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum_f2[thread_id], d_sum_f2, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum_fx[thread_id], d_sum_fx, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum_fx2[thread_id], d_sum_fx2, sizeof(double), cudaMemcpyDeviceToHost);

        // 4.6) Free device memory
        cudaFree(d_sum_f);
        cudaFree(d_sum_f2);
        cudaFree(d_sum_fx);
        cudaFree(d_sum_fx2);
    } // end OpenMP parallel region

    // 5) AGGREGATE HOST-SIDE: combine per-GPU sums
    double total_sum_f = 0.0;
    double total_sum_f2 = 0.0;
    double total_sum_fx = 0.0;
    double total_sum_fx2 = 0.0;
    unsigned long long total_samples_simple = 0ULL;
    unsigned long long total_samples_metropolis = 0ULL;

    for (int i = 0; i < NGPU; i++)
    {
        total_sum_f += h_sum_f[i];
        total_sum_f2 += h_sum_f2[i];
        total_sum_fx += h_sum_fx[i];
        total_sum_fx2 += h_sum_fx2[i];
        total_samples_simple += dnum_per_gpu[i];
        total_samples_metropolis += dnum_per_gpu[i];
    }
    // Sanity check: total_samples_simple and total_samples_metropolis should each == dnum
    // (up to rounding).  If rem>0, some GPUs get one extra sample, but sum of dnum_per_gpu = dnum.

    // 6) Compute means & standard deviations exactly as in CPU version
    double mean_simple = total_sum_f / (double)total_samples_simple;
    double var_simple = (total_sum_f2 / (double)total_samples_simple - mean_simple * mean_simple) / (double)total_samples_simple;
    if (var_simple < 0.0)
        var_simple = 0.0; // guard against tiny negative due to round-off
    double sigma_simple = sqrt(var_simple);

    double mean_metropolis = total_sum_fx / (double)total_samples_metropolis;
    double var_metropolis = (total_sum_fx2 / (double)total_samples_metropolis - mean_metropolis * mean_metropolis) / (double)total_samples_metropolis;
    if (var_metropolis < 0.0)
        var_metropolis = 0.0;
    double sigma_metropolis = sqrt(var_metropolis);

    // 7) Print results, and also write to "results/mc_int_cuda.txt"
    printf("Simple sampling (GPU‐parallel):     %.10f +/- %.10f\n",
           mean_simple, sigma_simple);
    printf("Metropolis sampling (GPU‐parallel): %.10f +/- %.10f\n",
           mean_metropolis, sigma_metropolis);

    // Create output directory if needed
    system("mkdir -p results");
    FILE *outc = fopen("results/mc_int_cuda.txt", "w");
    if (outc == NULL)
    {
        fprintf(stderr, "Cannot open output file results/mc_int_cuda.txt\n");
        // but still continue
    }
    else
    {
        fprintf(outc, "Monte Carlo integration results (CUDA multi-GPU):\n");
        fprintf(outc, "Number of Sampling: %llu\n", (unsigned long long)dnum);
        fprintf(outc, "Simple sampling (GPU‐parallel):     %.10f +/- %.10f\n",
                mean_simple, sigma_simple);
        fprintf(outc, "Metropolis sampling (GPU‐parallel): %.10f +/- %.10f\n",
                mean_metropolis, sigma_metropolis);
        fprintf(outc, "==================================================\n");
        fclose(outc);
    }

    // Free host memory
    free(dnum_per_gpu);
    free(Dev);
    free(h_sum_f);
    free(h_sum_f2);
    free(h_sum_fx);
    free(h_sum_fx2);

    return 0;
}
