//
//  Example.m
//  MPSSimpleGemm
//
//  © RuQing (G) Xu, Univ. Tokyo, 2021~
//

#import <Foundation/Foundation.h>
void MPSSimpleGemm_32F_32F
    (bool tA,
     bool tB,
     unsigned long m,
     unsigned long n,
     unsigned long k,
     double alpha,
     float *addrA, unsigned long tdA_,
     float *addrB, unsigned long tdB_,
     double beta,
     float *addrC, unsigned long tdC_);
void MPSSimpleGemm_8I_32F
    (bool tA,
     bool tB,
     unsigned long m,
     unsigned long n,
     unsigned long k,
     double alpha,
     int8_t *addrA, unsigned long tdA_,
     int8_t *addrB, unsigned long tdB_,
     double beta,
     float *addrC, unsigned long tdC_);


int main(int argc, const char * argv[]) {
    @autoreleasepool {
        float vA[6] = {1, 2, 3, 4, 5, 6};
        float vB[6] = {1, 2, 3, 4, 5, 6};
        float vC[6] = {1, 1, 1, 1, 1, 1};
        int8_t iA[6]= {1, 2, 3, 4, 5, 6};
        int8_t iB[6]= {1, 2, 3, 4, 5, 6};
        float iC[6] = {1, 1, 1, 1, 1, 1};

        NSLog(@"Begin Metal SGEMM");
        MPSSimpleGemm_32F_32F(false, true,
                              2, 2, 2,
                              1.0,
                              vA, 3,
                              vB, 3,
                              1.0,
                              vC, 2);
        NSLog(@"End Metal SGEMM");
        fprintf(stdout, "C = [");
        for (int i = 0; i < 6; ++i)
            fprintf(stdout, " %10.6e", vC[i]);
        fprintf(stdout, "]\n");

        NSLog(@"Begin Metal I8GEMM");
        MPSSimpleGemm_8I_32F(false, true,
                             2, 2, 3,
                             1.0,
                             iA, 3,
                             iB, 3,
                             1.0,
                             iC, 2);
        NSLog(@"End Metal SGEMM");
        fprintf(stdout, "C = [");
        for (int i = 0; i < 6; ++i)
            fprintf(stdout, " %10.6e", iC[i]);
        fprintf(stdout, "]\n");
    }
    return 0;
}
