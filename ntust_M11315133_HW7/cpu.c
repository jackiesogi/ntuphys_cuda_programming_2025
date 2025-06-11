#include <stdio.h>
#include <stdlib.h> // for malloc, free
#include <string.h> // for memcpy
#include <math.h>
#include <gsl/gsl_rng.h> /* header for gsl_rng */

double f(double *x, int n)
{
    // integrand: 1 / (x_1^2 + x_2^2 + ... + x_n^2 + 1)
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += x[i] * x[i];
    }
    return 1.0 / (sum + 1.0);
}

double w(double *x, double c, double a, int n)
{
    // weight function: ∏_{i=1}^n [c * exp(-a * x[i])]
    double ret = 1.0;
    for (int i = 0; i < n; i++)
    {
        ret *= c * exp(-a * x[i]);
    }
    return ret;
}

int main(void)
{
    gsl_rng *rng; /* pointer to gsl_rng random number generator */

    int n;
    double c, a;
    long long dnum; /* number of samples (must be integer) */
    double *x_old;  /* old point in Metropolis */
    double *x_new;  /* proposed new point */
    double *x;      /* accepted point, to evaluate integrand */
    double *y;      /* for simple sampling */

    double r;
    double mean, sigma;
    double mean1, sigma1;
    double fx, w_old, w_new;

    n = 10;
    c = 1.5822955636702438; /* should be a/(1 - exp(-a)) for a=1 in 1D; for n-dim, the product yields a normalized distribution */
    a = 1.0;

    printf("How many samples to be generated ?\n");
    scanf("%lld", &dnum);
    printf("Samples size: %lld\n", dnum);
    printf("Normalization constant (c) = %lf\n", c);
    printf("Exponential decay constant (a) = %lf\n", a);

    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 12345);

    /*************************************/
    /* 1) Simple (uniform) Monte Carlo  */
    /*************************************/
    y = (double *)malloc(n * sizeof(double));
    if (y == NULL)
    {
        fprintf(stderr, "Error allocating y.\n");
        return 1;
    }

    mean = 0.0;
    sigma = 0.0;

    for (long long i = 0; i < dnum; i++)
    {
        for (int j = 0; j < n; j++)
        {
            y[j] = gsl_rng_uniform(rng); /* uniform in [0,1] */
        }
        fx = f(y, n);
        mean += fx;
        sigma += fx * fx;
    }
    mean /= (double)dnum;
    /* Standard error: sqrt[ ( ⟨f^2⟩ - ⟨f⟩^2 ) / N ] */
    {
        double e_fx2 = sigma / (double)dnum; /* ⟨f^2⟩ */
        double e_fx = mean;                  /* ⟨f⟩ */
        sigma = sqrt((e_fx2 - e_fx * e_fx) / (double)dnum);
    }
    printf("Simple sampling:     %.10f +/- %.10f\n", mean, sigma);

    free(y);

    /***************************************************/
    /* 2) Importance sampling via Metropolis algorithm */
    /***************************************************/
    x_old = (double *)malloc(n * sizeof(double));
    x_new = (double *)malloc(n * sizeof(double));
    x = (double *)malloc(n * sizeof(double));

    if (x_old == NULL || x_new == NULL || x == NULL)
    {
        fprintf(stderr, "Error allocating Metropolis arrays.\n");
        return 1;
    }

    /* Initialize x_old to a random point in [0,1]^n */
    for (int i = 0; i < n; i++)
    {
        x_old[i] = gsl_rng_uniform(rng);
    }
    w_old = w(x_old, c, a, n);

    mean1 = 0.0;
    sigma1 = 0.0;

    for (long long i = 0; i < dnum; i++)
    {
        /* Propose x_new uniformly in [0,1]^n (symmetric proposal) */
        for (int j = 0; j < n; j++)
        {
            x_new[j] = gsl_rng_uniform(rng);
        }
        w_new = w(x_new, c, a, n);

        /* Metropolis acceptance */
        if (w_new >= w_old)
        {
            /* Accept unconditionally */
            memcpy(x_old, x_new, n * sizeof(double));
            w_old = w_new;
        }
        else
        {
            r = gsl_rng_uniform(rng);
            if (r < (w_new / w_old))
            {
                memcpy(x_old, x_new, n * sizeof(double));
                w_old = w_new;
            }
            /* else: reject, keep x_old as is */
        }

        /* Copy accepted x_old into x for evaluating f(x)/w(x) */
        memcpy(x, x_old, n * sizeof(double));

        fx = f(x, n) / w(x, c, a, n);
        mean1 += fx;
        sigma1 += fx * fx;
    }

    mean1 /= (double)dnum;
    {
        double e_fx2 = sigma1 / (double)dnum; /* ⟨(f/w)^2⟩ */
        double e_fx = mean1;                  /* ⟨f/w⟩ */
        /* Standard error = sqrt[ ( ⟨(f/w)^2⟩ - ⟨f/w⟩^2 ) / N ] */
        sigma1 = sqrt((e_fx2 - e_fx * e_fx) / (double)dnum);
    }

    printf("Metropolis sampling with Importance: %.10f +/- %.10f\n", mean1, sigma1);

    {
        FILE *outc = fopen("results/mc_int_cpu.txt", "w");
        if (outc == NULL)
        {
            fprintf(stderr, "Cannot open output file in results/ folder.\n");
            /* Not a fatal error; maybe directory doesn’t exist */
        }
        else
        {
            fprintf(outc, "Monte Carlo integration results:\n");
            fprintf(outc, "Number of Samples:   %lld\n", dnum);
            fprintf(outc, "Simple sampling:     %.10f +/- %.10f\n", mean, sigma);
            fprintf(outc, "Metropolis sampling with Importance: %.10f +/- %.10f\n", mean1, sigma1);
            fprintf(outc, "==================================================\n");
            fclose(outc);
        }
    }

    free(x_old);
    free(x_new);
    free(x);
    gsl_rng_free(rng);

    return 0;
}
