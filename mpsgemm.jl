using Libdl

if !(@isdefined libmpsgemm)
    libmpsgemm = dlopen("MPSSimpleGemm.dylib")
end

custom_typechars = Dict(Float32 => "32F",
                        Float16 => "16F",
                        Int16   => "16I",
                        Int8    => "8I")

"Compute SGEMM with Metal Performance Shader."
mmul_simple!(tA::Bool,
             tB::Bool,
             α::Float64,
             A::StridedMatrix{TSrc},
             B::StridedMatrix{TSrc},
             β::Float64,
             C::StridedMatrix{TDst}) where {TSrc<:Number,
                                            TDst<:Number} = begin
    if !tA
        m, k = size(A)
    else
        k, m = size(A)
    end
    if !tB
        k_,n = size(B)
    else
        n, k_= size(B)
    end
    m_, n_ = size(C)
    (k == k_ && m == m_ && n == n_) ||
        throw(DimensionMismatch("Array dimension mismatch."))

    _, ldA = strides(A)
    _, ldB = strides(B)
    _, ldC = strides(C)

    # Lookup type symbols.
    funcname = string("MPSSimpleGemm_",
                      custom_typechars[TSrc], "_",
                      custom_typechars[TDst])
    funcptr = dlsym(libmpsgemm, funcname)
    funcptr != 0 || throw(DomainError(TDst),
                          "Multiplication of $TSrc and $TSrc into $TDst is not supported by Metal.")

    # ML Compute is row-major. Exchange A and B.
    ccall(funcptr,
          Cvoid,
          (Bool, Bool,
           Culong, Culong, Culong,
           Float64,
           Ptr{TSrc}, Culong,
           Ptr{TSrc}, Culong,
           Float64,
           Ptr{TDst}, Culong),
          tB, tA,
          n, m, k,
          α,
          B, ldB,
          A, ldA,
          β,
          C, ldC);
    C
end

