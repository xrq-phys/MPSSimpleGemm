//
//  matrix_realloc_rowmaj.h
//  MPSSimpleGemm
//
//  © RuQing (G) Xu, Univ. Tokyo, 2021~
//

#pragma once
#include <stdint.h>

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#else
typedef float float32_t;
typedef int16_t float16_t;
#endif

#define FUNCNAME(name, suffix) name##_##suffix

#define DECFUNC(ctype, typechar) \
EXTERNC void FUNCNAME(matrix_realloc_rowmaj, typechar) \
    (unsigned long m, \
     unsigned long n, \
     ctype *A, unsigned long tdA, \
     ctype **Ae_, unsigned long *tdAe, \
     unsigned long *lbuff);
DECFUNC( float32_t, 32F )
DECFUNC( float16_t, 16F )
DECFUNC( int64_t,   64I )
DECFUNC( int32_t,   32I )
DECFUNC( int16_t,   16I )
DECFUNC( int8_t,    8I  )
DECFUNC( uint64_t,  64U )
DECFUNC( uint32_t,  32U )
DECFUNC( uint16_t,  16U )
DECFUNC( uint8_t,   8U  )
#undef DECFUNC

#define DECFUNC(ctype, typechar) \
EXTERNC void FUNCNAME(matrix_writeback_rowmaj, typechar) \
    (unsigned long m, \
     unsigned long n, \
     ctype *A, unsigned long tdA, \
     ctype *Ae, unsigned long tdAe);
DECFUNC( float32_t, 32F )
DECFUNC( float16_t, 16F )
DECFUNC( int64_t,   64I )
DECFUNC( int32_t,   32I )
DECFUNC( int16_t,   16I )
DECFUNC( int8_t,    8I  )
DECFUNC( uint64_t,  64U )
DECFUNC( uint32_t,  32U )
DECFUNC( uint16_t,  16U )
DECFUNC( uint8_t,   8U  )
#undef DECFUNC

#undef FUNCNAME

