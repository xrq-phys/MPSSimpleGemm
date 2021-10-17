//
//  Example.c
//  MPSSimpleGemm
//
//  © RuQing (G) Xu, Univ. Tokyo, 2021~
//

#include "MPSSimpleGemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern const unsigned metal_page_size;

int main(int argc, const char * argv[]) {
    long m   = 4096;
    long n   = 4096;
    long k   = 4096;
    long tda = k;
    long tdb = n;
    long tdc = n;
    fprintf(stderr, "Problem size: m=%4ld n=%4ld k=%4ld\n", m, n, k);

    float *a;
    float *b;
    float *c;
    posix_memalign((void **)&a, metal_page_size, m * tda * sizeof(float) + metal_page_size);
    posix_memalign((void **)&b, metal_page_size, k * tdb * sizeof(float) + metal_page_size);
    posix_memalign((void **)&c, metal_page_size, m * tdc * sizeof(float) + metal_page_size);

    for (int i = 0; i < m * tda; ++i) a[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < k * tdb; ++i) b[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < m * tdc; ++i) c[i] = 1.0;

#ifdef DEBUG
    FILE *fidA = fopen("A.dat", "w+");
    FILE *fidB = fopen("B.dat", "w+");
    FILE *fidC = fopen("C.dat", "w+");

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j)
            fprintf(fidA, "%15.12e ", a[i * tda + j]);
        fprintf(fidA, "\n");
    }
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j)
            fprintf(fidB, "%15.12e ", b[i * tdb + j]);
        fprintf(fidB, "\n");
    }
    fclose(fidA);
    fclose(fidB);
#endif

    double alpha = 1.0;
    double beta  = 0.2;

    int nexp_inner = 1; ///< Num of tests within.
    double elapsed =
        MPSSimpleGemm_32F_32F
        (
         0, 0,
         m, n, k, alpha,
         a, tda, b, tdb, beta,
         c, tdc
        );
    fprintf(stderr, "Elapsed: %f ms\n", elapsed * 1000);
    fprintf(stderr, "%f GFlOps\n",
            (double)2 * nexp_inner * m * n * k / 1000000000 / elapsed);

#ifdef DEBUG
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            fprintf(fidC, "%15.12e ", c[i * tdc + j]);
        fprintf(fidC, "\n");
    }
    fclose(fidC);
#endif
    fprintf(stderr, "Exec completed.\n");

    free(a);
    free(b);
    free(c);

    return 0;
}
