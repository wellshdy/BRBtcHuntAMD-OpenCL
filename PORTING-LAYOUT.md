# BRBtcHuntAMD — Conversão HIP → OpenCL C 2.0 (RDNA1 / Windows)

## 1. Visão geral do porte

O projeto original é uma única TU HIP (`GPUWorker.cpp`) que inclui
`HashPipeline.cpp` e depende de `AMDMath.h`. O porte OpenCL separa o
código em três camadas distintas, porque o OpenCL exige que o kernel
seja compilado em tempo de execução a partir de um arquivo `.cl`
externo, e o host usa uma API C completamente diferente (CL em vez de
HIP).

```
BRBtcHuntAMD-OpenCL/
├── kernels/                 ← Código que roda na GPU (compilado em runtime)
│   ├── secp256k1_math.cl    ← uint256, ModMul, ModSqr, ModInv (Bernstein-Yang)
│   ├── hash_pipeline.cl     ← SHA-256 + RIPEMD-160 inline
│   ├── gpu_worker.cl        ← kernel_point_add_and_check_oneinv + helpers
│   └── kernel_top.cl        ← #include dos três acima + __kernel entry points
│
├── include/                 ← Headers C++ compartilhados entre host e device
│   ├── ocl_common.h         ← tipos uint256, ECPointA, FoundResult, VanityResult
│   ├── ocl_helpers.h        ← utilitários de host (hex, base58, 256-bit arith)
│   ├── gpu_worker_host.h    ← declaração de run_on_gpu_opencl()
│   └── lang.h               ← i18n ( reaproveitado do original)
│
├── src/                     ← Host C++ (usa API OpenCL C pura)
│   ├── main_opencl.cpp      ← entry point (CLI, range split, progress)
│   ├── gpu_worker_opencl.cpp← inicializa CL context/queue, lança kernels
│   ├── ocl_utils.cpp        ← wrapper de erro + carregamento de arquivo .cl
│   └── hash_check.cpp       ← SHA-256/RIPEMD-160 do host (para validação)
│
├── scripts/
│   ├── build-windows.ps1    ← compila com cl.exe + liga contra OpenCL.lib
│   └── run.bat              ← lança o executável
│
├── CMakeLists.txt           ← build portável (Windows + Linux)
└── README.md
```

## 2. Mapeamento arquivo-a-arquivo (HIP → OpenCL)

| Original (HIP)                 | Destino OpenCL                  | Observações |
|--------------------------------|---------------------------------|-------------|
| `include/AMDMath.h`            | `kernels/secp256k1_math.cl`     | C structs + funções explícitas; sem `__uint128_t`, sem `__builtin_addcll` |
| `include/AMDStructures.h`      | `include/ocl_common.h`          | `FoundResult`, `VanityResult` viram POD C |
| `include/AMDHash.h`            | (merge em `hash_pipeline.cl`)   | Apenas declarações, fica embutido no `.cl` |
| `include/AMDUtils.h`           | `include/ocl_helpers.h` + `kernels/warp_helpers.cl` | Parte host vai para `ocl_helpers.h`; parte device vai para o `.cl` |
| `include/GpuPlatform.h`        | **removido**                    | Substituído por `#define CL_HPP_TARGET_OPENCL_VERSION 200` e `#include <CL/opencl.h>` |
| `include/GPUWorker.h`          | `include/gpu_worker_host.h`     | Apenas a parte host; `GpuShared` continua como struct C++ |
| `src/HashPipeline.cpp`         | `kernels/hash_pipeline.cl`      | `__device__` → qualificador vazio; macros ROL etc. preservadas |
| `src/GPUWorker.cpp` (device)   | `kernels/gpu_worker.cl`         | `__global__` → `__kernel`; `<<<...>>>` → `clEnqueueNDRangeKernel` |
| `src/GPUWorker.cpp` (host)     | `src/gpu_worker_opencl.cpp`     | `hipMalloc`→`clCreateBuffer`, `hipMemcpy`→`clEnqueueWriteBuffer`, etc. |
| `src/main.cpp`                 | `src/main_opencl.cpp`           | `hipGetDeviceCount` → `clGetPlatformIDs` + `clGetDeviceIDs` |

## 3. Decisões de portabilidade crítica

### 3.1 Multiplicação 64×64 → 128 bits

O original depende de `__uint128_t` (Clang). OpenCL C 2.0 não tem
`__uint128_t`, mas oferece os builtins `mul_hi` e `mad_hi`:

```c
// Em OpenCL C (size_t = 32 bits na GPU por default, ulong = 64 bits):
ulong lo = a * b;             // v_mul_lo_u32 (GCN)
ulong hi = mul_hi(a, b);      // v_mul_hi_u32 (GCN)
```

Para a cadeia de carry (em vez de `__builtin_addcll`), usa-se
`add3`-estilo manual:

```c
ulong add_with_carry(ulong a, ulong b, ulong carry_in, ulong *carry_out) {
    ulong s = a + b;
    ulong c = (s < a) ? 1UL : 0UL;
    ulong s2 = s + carry_in;
    if (s2 < s) c = 1UL;
    *carry_out = c;
    return s2;
}
```

### 3.2 Sub-group shuffles (wavefront 64 / warp 32)

RDNA1 tem wavefront de 32 lanes (modo "wave32"). O OpenCL 2.0 expõe
`cl_khr_subgroups` que deve ser habilitado no `.cl`:

```c
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
```

Mapeamento:

| HIP / CUDA              | OpenCL C (cl_khr_subgroups)        | Fallback (sem extensão)         |
|-------------------------|------------------------------------|---------------------------------|
| `__shfl(v, src)`        | `sub_group_shuffle(v, src)`        | `__local` array + barreira      |
| `__shfl_down(v, delta)` | `sub_group_shuffle_down(v, delta)` | `__local` array + barreira      |
| `__any(pred)`           | `sub_group_any(pred)`              | `atomic_or` em `__local`        |

Para RDNA1 (wave32), é **obrigatório** forçar o compilador a usar
wave32 adicionando `-cl-std=CL2.0` e queryando
`CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` (deve retornar 32).

### 3.3 Memória `__constant__` do HIP → `__constant` do OpenCL

O HIP permite `__device__ __constant__` declarado em header. OpenCL
exige que constantes sejam passadas como `clSetKernelArg` ou
declaradas como `__constant` no topo do `.cl`:

```c
__constant ulong SECP_P_LE[4] = { ... };
__constant ulong K_SHA256[64] = { ... };
```

Para arrays dinâmicos (`c_Gx`, `c_Gy`), passa-se como `__global` +
`__attribute__((max_constant_size))` ou como buffer normal.

### 3.4 Inlining

O HIP usa `__forceinline__`. No OpenCL C, usamos:

```c
#define __forceinline__ inline __attribute__((always_inline))
```

Aplicado a todas as funções matemáticas críticas (`_ModMult`, `_ModSqr`,
`UMult`, `ModSub256`, etc.). Sem isso, o compilador AMD OpenCL insere
chamadas de função que destroem a performance (cada
`v_s_setreg_b32` custa ~4 ciclos).

## 4. Fluxo de inicialização do host (substitui o `hip*` original)

```cpp
// main_opencl.cpp (resumo)
cl_platform_id plat;
cl_device_id   dev;
cl_context     ctx;
cl_command_queue queue;
cl_program     program;
cl_kernel      krn_point_add;

clGetPlatformIDs(1, &plat, NULL);
clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
ctx   = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
queue = clCreateCommandQueueWithProperties(ctx, dev, 0, &err);

// Carrega e compila o kernel em runtime:
char* src = load_file("kernels/kernel_top.cl");
program = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, &err);
clBuildProgram(program, 1, &dev,
    "-cl-std=CL2.0 -cl-mad-enable -DOPENCL_RDNA1", NULL, NULL);
krn_point_add = clCreateKernel(program, "kernel_point_add_and_check_oneinv", &err);
```

## 5. O que ainda falta após esta entrega

Esta primeira entrega cobre:
- ✅ Estrutura de diretórios (este arquivo)
- ✅ `include/ocl_common.h` — structs C puras (uint256, ECPointA, etc.)
- ✅ `kernels/secp256k1_math.cl` — toda a aritmética modular 256-bit
- ✅ Macros UADDO/USUBC/MADDO reescritas para OpenCL (sem `__uint128_t`)

Próximas entregas sugeridas:
- 🔲 `kernels/hash_pipeline.cl` (SHA-256 + RIPEMD-160)
- 🔲 `kernels/gpu_worker.cl` (kernel principal + batch inversion)
- 🔲 `src/gpu_worker_opencl.cpp` (host: buffers, queue, launch)
- 🔲 `src/main_opencl.cpp` (CLI + range distribution)
- 🔲 `CMakeLists.txt` + `scripts/build-windows.ps1`
