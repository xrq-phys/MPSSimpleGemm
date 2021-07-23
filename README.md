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

