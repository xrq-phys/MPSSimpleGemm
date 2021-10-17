#include <stdbool.h>
#include <arm_neon.h>

#define FUNCNAME_AUX(name, suffix) name##_##suffix
#define FUNCNAME(name, suffix_src, suffix_dst) name##_##suffix_src##_##suffix_dst

#define DEFFUNC(mpstype_src, ctype_src, mpstype_dst, ctype_dst, typechar_src, typechar_dst) \
double FUNCNAME(MPSSimpleGemm, typechar_src, typechar_dst) \
    (bool tA, \
     bool tB, \
     unsigned long m, \
     unsigned long n, \
     unsigned long k, \
     double alpha, \
     ctype_src *addrA, unsigned long tdA_, \
     ctype_src *addrB, unsigned long tdB_, \
     double beta, \
     ctype_dst *addrC, unsigned long tdC_);
DEFFUNC( MPSDataTypeFloat32, float32_t, MPSDataTypeFloat32, float32_t, 32F, 32F )
DEFFUNC( MPSDataTypeFloat16, float16_t, MPSDataTypeFloat16, float16_t, 16F, 16F )
DEFFUNC( MPSDataTypeFloat16, float16_t, MPSDataTypeFloat32, float32_t, 16F, 32F )
DEFFUNC( MPSDataTypeInt16,   int16_t,   MPSDataTypeFloat32, float32_t, 16I, 32F )
DEFFUNC( MPSDataTypeInt8,    int8_t,    MPSDataTypeFloat16, float16_t, 8I,  16F )
DEFFUNC( MPSDataTypeInt8,    int8_t,    MPSDataTypeFloat32, float32_t, 8I,  32F )
#undef FUNCNAME_AUX
#undef FUNCNAME
#undef DEFFUNC

