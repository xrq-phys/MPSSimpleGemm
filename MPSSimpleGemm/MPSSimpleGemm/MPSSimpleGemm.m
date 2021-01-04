//
//  MPSSimpleGemm.m
//  MPSSimpleGemm
//
//  © RuQing (G) Xu, Univ. Tokyo, 2021~
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import "matrix_realloc_rowmaj.h"

#define FUNCNAME_AUX(name, suffix) name##_##suffix
#define FUNCNAME(name, suffix_src, suffix_dst) name##_##suffix_src##_##suffix_dst

#define DEFFUNC(mpstype_src, ctype_src, mpstype_dst, ctype_dst, typechar_src, typechar_dst) \
void FUNCNAME(MPSSimpleGemm, typechar_src, typechar_dst) \
	(bool tA, \
     bool tB, \
     unsigned long m, \
     unsigned long n, \
     unsigned long k, \
     double alpha, \
     ctype_src *addrA, unsigned long tdA_, \
     ctype_src *addrB, unsigned long tdB_, \
     double beta, \
     ctype_dst *addrC, unsigned long tdC_) \
{ \
    MPSMatrixDescriptor *shapeA; \
    MPSMatrixDescriptor *shapeB; \
    MPSMatrixDescriptor *shapeC; \
\
    /* Shape */ \
    unsigned long nrowA, ncolA; \
    unsigned long nrowB, ncolB; \
    unsigned long nrowC, ncolC; \
    if (!tA) { \
        nrowA = m; \
        ncolA = k; \
    } else { \
        nrowA = k; \
        ncolA = m; \
    } \
    if (!tB) { \
        nrowB = k; \
        ncolB = n; \
    } else { \
        nrowB = n; \
        ncolB = k; \
    } \
    nrowC = m; \
    ncolC = n; \
\
    /* Memory */ \
    ctype_src *buffA; \
    ctype_src *buffB; \
    ctype_dst *buffC; \
    unsigned long tdA, lbuffA; \
    unsigned long tdB, lbuffB; \
    unsigned long tdC, lbuffC; \
    FUNCNAME_AUX(matrix_realloc_rowmaj, typechar_src)(nrowA, ncolA, addrA, tdA_, &buffA, &tdA, &lbuffA); \
    FUNCNAME_AUX(matrix_realloc_rowmaj, typechar_src)(nrowB, ncolB, addrB, tdB_, &buffB, &tdB, &lbuffB); \
    FUNCNAME_AUX(matrix_realloc_rowmaj, typechar_dst)(nrowC, ncolC, addrC, tdC_, &buffC, &tdC, &lbuffC); \
\
    /* Descriptor */ \
    shapeA = [MPSMatrixDescriptor \
              matrixDescriptorWithDimensions:nrowA columns:ncolA \
              rowBytes:tdA * sizeof(ctype_src) dataType:mpstype_src]; \
    shapeB = [MPSMatrixDescriptor \
              matrixDescriptorWithDimensions:nrowB columns:ncolB \
              rowBytes:tdB * sizeof(ctype_src) dataType:mpstype_src]; \
    shapeC = [MPSMatrixDescriptor \
              matrixDescriptorWithDimensions:nrowC columns:ncolC \
              rowBytes:tdC * sizeof(ctype_dst) dataType:mpstype_dst]; \
\
    /* Device */ \
    id<MTLDevice> iDev = MTLCreateSystemDefaultDevice(); \
    id<MTLCommandQueue> iQueue = [iDev newCommandQueue]; \
    id<MTLCommandBuffer> iCmd = [iQueue commandBuffer]; \
\
    /* Buffer */ \
    id<MTLBuffer> inpA = [iDev \
                          newBufferWithBytesNoCopy:buffA \
                          length:lbuffA \
                          options:MTLResourceOptionCPUCacheModeDefault \
                          deallocator:nil]; \
    id<MTLBuffer> inpB = [iDev \
                          newBufferWithBytesNoCopy:buffB \
                          length:lbuffB \
                          options:MTLResourceOptionCPUCacheModeDefault \
                          deallocator:nil]; \
    id<MTLBuffer> inpC = [iDev \
                          newBufferWithBytesNoCopy:buffC \
                          length:lbuffC \
                          options:MTLResourceOptionCPUCacheModeDefault \
                          deallocator:nil]; \
\
    MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:inpA descriptor:shapeA]; \
    MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:inpB descriptor:shapeB]; \
    MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:inpC descriptor:shapeC]; \
\
    MPSMatrixMultiplication *iGemm = [[MPSMatrixMultiplication alloc] \
                                      initWithDevice:iDev \
                                      transposeLeft:tA transposeRight:tB \
                                      resultRows:m resultColumns:n interiorColumns:k \
                                      alpha:alpha beta:beta]; \
\
    /* Attach buffer */ \
    [iGemm \
     encodeToCommandBuffer:iCmd \
     leftMatrix:mA rightMatrix:mB resultMatrix:mC]; \
\
    /* Commit */ \
    [iCmd commit]; \
    [iCmd waitUntilCompleted]; \
\
    FUNCNAME_AUX(matrix_writeback_rowmaj, typechar_src)(nrowA, ncolA, addrA, tdA_, buffA, tdA); \
    FUNCNAME_AUX(matrix_writeback_rowmaj, typechar_src)(nrowB, ncolB, addrB, tdB_, buffB, tdB); \
    FUNCNAME_AUX(matrix_writeback_rowmaj, typechar_dst)(nrowC, ncolC, addrC, tdC_, buffC, tdC); \
}
DEFFUNC( MPSDataTypeFloat32, float32_t, MPSDataTypeFloat32, float32_t, 32F, 32F )
DEFFUNC( MPSDataTypeFloat16, float16_t, MPSDataTypeFloat16, float16_t, 16F, 16F )
DEFFUNC( MPSDataTypeFloat16, float16_t, MPSDataTypeFloat32, float32_t, 16F, 32F )
DEFFUNC( MPSDataTypeInt16,   int16_t,   MPSDataTypeFloat32, float32_t, 16I, 32F )
DEFFUNC( MPSDataTypeInt8,    int8_t,    MPSDataTypeFloat16, float16_t, 8I,  16F )
DEFFUNC( MPSDataTypeInt8,    int8_t,    MPSDataTypeFloat32, float32_t, 8I,  32F )
#undef DEFFUNC

#undef FUNCNAME_AUX
#undef FUNCNAME

