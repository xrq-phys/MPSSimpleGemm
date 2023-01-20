//
//  Example.c
//  MPSSimpleGemm
//
//  © RuQing (G) Xu, Univ. Tokyo, 2021~
//

#import "MPSSimpleGemm.h"
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern const unsigned metal_page_size;

#define FUNCNAME(name, suffix_src, suffix_dst) name##_##suffix_src##_##suffix_dst

#define DECLFUNC(mpstype_src, ctype_src, mpstype_dst, ctype_dst, typechar_src, typechar_dst) \
MPSMatrixMultiplication *FUNCNAME(MPSGetGemmHandler, typechar_src, typechar_dst) \
    (id<MTLDevice> *iDev, \
     bool tA, \
     bool tB, \
     unsigned long m, \
     unsigned long n, \
     unsigned long k, \
     double alpha, double beta); \
void FUNCNAME(MPSGetGemmMatrixDescriptor, typechar_src, typechar_dst) \
    (id<MTLDevice> *iDev, \
     bool tA, \
     bool tB, \
     unsigned long m, \
     unsigned long n, \
     unsigned long k, \
     ctype_src *addrA, unsigned long tdA_, \
     ctype_src *addrB, unsigned long tdB_, \
     ctype_dst *addrC, unsigned long tdC_, \
     MPSMatrix **mA, unsigned long *tdA, \
     MPSMatrix **mB, unsigned long *tdB, \
     MPSMatrix **mC, unsigned long *tdC); \
void FUNCNAME(MPSFlushGemmMatrix, typechar_src, typechar_dst) \
    (ctype_src *addrA, unsigned long tdA_, \
     ctype_src *addrB, unsigned long tdB_, \
     ctype_dst *addrC, unsigned long tdC_, \
     MPSMatrix *mA, unsigned long tdA, \
     MPSMatrix *mB, unsigned long tdB, \
     MPSMatrix *mC, unsigned long tdC);
DECLFUNC( MPSDataTypeFloat32, float32_t, MPSDataTypeFloat32, float32_t, 32F, 32F )
DECLFUNC( MPSDataTypeFloat16, float16_t, MPSDataTypeFloat16, float16_t, 16F, 16F )
DECLFUNC( MPSDataTypeFloat16, float16_t, MPSDataTypeFloat32, float32_t, 16F, 32F )
DECLFUNC( MPSDataTypeInt16,   int16_t,   MPSDataTypeFloat32, float32_t, 16I, 32F )
DECLFUNC( MPSDataTypeInt8,    int8_t,    MPSDataTypeFloat16, float16_t, 8I,  16F )
DECLFUNC( MPSDataTypeInt8,    int8_t,    MPSDataTypeFloat32, float32_t, 8I,  32F )

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
    FILE *fidA = fopen("/tmp/A.dat", "w+");
    FILE *fidB = fopen("/tmp/B.dat", "w+");
    FILE *fidC = fopen("/tmp/C.dat", "w+");

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

    int nexp_inner = 200; ///< Num of tests within.
#if 0
    double elapsed =
        MPSSimpleGemm_32F_32F
        (
         0, 0,
         m, n, k, alpha,
         a, tda, b, tdb, beta,
         c, tdc
        );
#else
    id<MTLDevice> iDev = MTLCreateSystemDefaultDevice();
    id<MTLCommandBuffer> iCmd = [[iDev newCommandQueue] commandBuffer];
    unsigned long tda_, tdb_, tdc_;
    MPSMatrix *mA, *mB, *mC;
    MPSMatrixMultiplication *iGemm = MPSGetGemmHandler_32F_32F
        ( &iDev,
          0, 0,
          m, n, k,
          alpha, beta );
    MPSGetGemmMatrixDescriptor_32F_32F
        ( &iDev,
          0, 0,
          m, n, k,
          a, tda,
          b, tdb,
          c, tdc,
          &mA, &tda_,
          &mB, &tdb_,
          &mC, &tdc_ );
    for ( int i = 0; i < nexp_inner; ++i )
        [iGemm
         encodeToCommandBuffer:iCmd
         leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    [iCmd commit];
    [iCmd waitUntilCompleted];
    double elapsed = [iCmd GPUEndTime] - [iCmd GPUStartTime];
#endif
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
