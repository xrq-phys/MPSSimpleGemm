//
//  matrix_realloc_rowmaj.c
//  MPSSimpleGemm
//
//  © RuQing (G) Xu, Univ. Tokyo, 2021~
//

#include <stdlib.h>
#include "matrix_realloc_rowmaj.h"

const unsigned metal_page_size = 0x4000;

#define FUNCNAME(name, suffix) name##_##suffix

#define DEFFUNC(ctype, typechar) \
void FUNCNAME(matrix_realloc_rowmaj, typechar) \
    (unsigned long m, \
     unsigned long n, \
     ctype *A, unsigned long tdA, \
     ctype **Ae_, unsigned long *tdAe, \
     unsigned long *lbuff) \
{ \
    ctype *Ae = 0x0; \
    unsigned long ldata = m * n * sizeof(ctype); \
    if (((unsigned long long)A % metal_page_size) || \
        (ldata % metal_page_size)) { \
        /* Reallocate matrices. */ \
        posix_memalign((void **)&Ae, metal_page_size, m * n * sizeof(ctype)); \
        for (unsigned long i = 0; i < m; ++i) \
            for (unsigned long j = 0; j < n; ++j) \
                Ae[i*n + j] = A[i*tdA + j]; \
        *Ae_ = Ae; \
        *tdAe = n; \
        if (ldata % metal_page_size) \
            *lbuff = metal_page_size * (ldata / metal_page_size + 1); \
        else \
            *lbuff = ldata; \
    } else { \
        /* In case allocation is not required. */ \
        *Ae_ = A; \
        *tdAe = tdA; \
        *lbuff = ldata; \
    } \
}
DEFFUNC( float32_t, 32F )
DEFFUNC( float16_t, 16F )
DEFFUNC( int64_t,   64I )
DEFFUNC( int32_t,   32I )
DEFFUNC( int16_t,   16I )
DEFFUNC( int8_t,    8I  )
DEFFUNC( uint64_t,  64U )
DEFFUNC( uint32_t,  32U )
DEFFUNC( uint16_t,  16U )
DEFFUNC( uint8_t,   8U  )
#undef DEFFUNC

#define DEFFUNC(ctype, typechar) \
void FUNCNAME(matrix_writeback_rowmaj, typechar) \
    (unsigned long m, \
     unsigned long n, \
     ctype *A, unsigned long tdA, \
     ctype *Ae, unsigned long tdAe) \
{ \
    if (A != Ae) { \
        /* In case reallocation has taken place. */ \
        for (unsigned long i = 0; i < m; ++i) \
            for (unsigned long j = 0; j < n; ++j) \
                A[i*tdA + j] = Ae[i*tdAe + j]; \
        free(Ae); \
    } \
}
DEFFUNC( float32_t, 32F )
DEFFUNC( float16_t, 16F )
DEFFUNC( int64_t,   64I )
DEFFUNC( int32_t,   32I )
DEFFUNC( int16_t,   16I )
DEFFUNC( int8_t,    8I  )
DEFFUNC( uint64_t,  64U )
DEFFUNC( uint32_t,  32U )
DEFFUNC( uint16_t,  16U )
DEFFUNC( uint8_t,   8U  )
#undef DEFFUNC

#undef FUNCNAME

