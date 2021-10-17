Use Metal Performance Shaders to compute GEMM. Supports:

| A         | B         | C         |
|-----------|-----------|-----------|
| `Float32` | `Float32` | `Float32` |
| `Float16` | `Float16` | `Float16` |
| `Float16` | `Float16` | `Float32` |
| `Int16`   | `Int16`   | `Float32` |
| `Int8`    | `Int8`    | `Float16` |
| `Int8`    | `Int8`    | `Float32` |

Have a try if you're using Mac with a on-chip GPU (e.g. M1).

Compilation one-liner for Julia integration.
```bash
# Use ARCHFLAG="-arch arm64" if you are using a ARM64 native build of Julia.
# (As of Jan. 3 2021, Native ARM64 Julia is not vendored but can already be built from master.)
ARCHFLAG="-arch x86_64" \
  clang $ARCHFLAG \
    MPSSimpleGemm/MPSSimpleGemm/MPSSimpleGemm.m \
    MPSSimpleGemm/MPSSimpleGemm/matrix_realloc_rowmaj.c \
    -framework CoreGraphics \
    -framework Foundation \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -shared -o MPSSimpleGemm.dylib
```

The GPU timer that skipps launch overhead shows a performance of ~1.4TFlOps/sec.
From this result I severely doubt that AMX (cf. [BLIS for Apple](https://github.com/xrq-phys/blis_apple))
 is sharing half of M1's GPU (Any idea how to test?).
```
Problem size: m=2048 n=2048 k=2048
Elapsed: 11.9637083 ms
1435.998666 GFlOps
Exec completed.
```

