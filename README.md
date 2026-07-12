# BRBtcHuntAMD-OpenCL

Porte do [BRBtcHuntAMD](https://github.com/jmr2704/BRBtcHuntAMD) (HIP/ROCm) para **OpenCL C 2.0**, otimizado para **GPUs AMD RDNA1** (RX 5700 XT, RX 5600 XT, RX 5500 XT) rodando em **Windows** com driver AMD Adrenalin.

## Status do porte

| Componente | Status | Linhas |
|---|---|---|
| `include/ocl_common.h` — Structs C puros | ✅ | 138 |
| `include/ocl_helpers.h` — Utilitários host 256-bit | ✅ | 116 |
| `include/gpu_worker_host.h` — GpuShared + declarações | ✅ | 75 |
| `include/Lang.h` — i18n (reaproveitado) | ✅ | 76 |
| `kernels/secp256k1_math.cl` — Aritmética modular 256-bit | ✅ | 1010 |
| `kernels/hash_pipeline.cl` — SHA-256 + RIPEMD-160 | ✅ | 350 |
| `kernels/gpu_worker.cl` — Kernel principal + batch inv | ✅ | 600 |
| `src/ocl_utils.cpp` — Loader/concat/build | ✅ | 220 |
| `src/gpu_worker_opencl.cpp` — Host: context, buffers, launch | ✅ | 700 |
| `src/main_opencl.cpp` — CLI + range split + progress | ✅ | 450 |
| `CMakeLists.txt` | ✅ | 90 |
| `scripts/build-windows.ps1` | ✅ | 130 |
| **Compila em GPU RDNA1 real** | ⚠️ Não testado | — |

## Pré-requisitos

### Windows
- **Visual Studio 2022** com workload "Desktop development with C++"
- **CMake 3.16+** — `choco install cmake`
- **AMD APP SDK 3.0** (para headers OpenCL) — baixar de https://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/
  - Ou: `choco install opencl-headers` (alternativa)
- **OpenSSL** (para validação de endereços P2PKH)
  - Via vcpkg: `vcpkg install openssl:x64-windows`
  - Ou: `choco install openssl`
- **Driver AMD Adrenalin 23+** (já vem com runtime OpenCL)

### Linux (testes/CI)
- `apt install opencl-headers ocl-icd-opencl-dev libssl-dev cmake g++`

## Build

### Windows (PowerShell)
```powershell
cd BRBtcHuntAMD-OpenCL
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1
```

Saída: `build\BRBtcHuntAMD-OpenCL.exe` + `build\kernels\*.cl`

### Linux / Manual
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Uso

```bash
# Busca por endereço (P2PKH)
BRBtcHuntAMD-OpenCL.exe --range 200000000:3FFFFFFFF \
                         --address 1HBtApAwR7JgqgzqERiA6R5T5o5mYh1k3j \
                         --grid 128,256 --slices 64

# Busca por hash160 raw
BRBtcHuntAMD-OpenCL.exe --range 200000000:3FFFFFFFF \
                         --target-hash160 79fbfc3e62c4d3e9b6d1c1c5d5d8c5e9a6b3c2d1 \
                         --gpus 0,1 --random --slices 16

# Modo vanity (salva chaves com N hex chars iniciais do hash160)
BRBtcHuntAMD-OpenCL.exe --range 1:FFFFFFFF \
                         --target-hash160 0000000000000000000000000000000000000000 \
                         --vanity 4 --random

# Listar GPUs detectadas
BRBtcHuntAMD-OpenCL.exe --help
```

### Argumentos

| Flag | Descrição | Default |
|---|---|---|
| `--range <hex>:<hex>` | Intervalo de chaves privadas em hex (obrigatório) | — |
| `--address <base58>` | Endereço P2PKH alvo (Bitcoin mainnet) | — |
| `--target-hash160 <hex>` | Hash160 raw em hex (alternativa a --address) | — |
| `--grid <P,T>` | Pontos por batch, threads por bloco | `128,256` |
| `--slices <N>` | Batches por thread por launch | `64` |
| `--gpus <all\|0\|0,1>` | Seleciona GPUs | `all` |
| `--random` | Modo loteria (saltos aleatórios) | off |
| `--vanity <N>` | Salva chaves cujos N hex chars do hash160 casam com target | off |
| `--kernel-dir <path>` | Caminho para os .cl | `kernels` |
| `--lang pt\|en` | Idioma | `en` |

## Arquitetura do porte

```
┌──────────────────────────────────────────────────────┐
│                  Host (C++ / OpenCL API)              │
│  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │  main_opencl.cpp │  │  gpu_worker_opencl.cpp   │  │
│  │  - CLI parsing   │  │  - clGetPlatformIDs      │  │
│  │  - range split   │  │  - clCreateContext       │  │
│  │  - progress      │  │  - clCreateBuffer        │  │
│  │  - thread launch │  │  - clEnqueueNDRangeKernel│  │
│  └────────┬─────────┘  └────────────┬─────────────┘  │
│           │                          │                │
│           ▼                          ▼                │
│  ┌────────────────────────────────────────────────┐  │
│  │           ocl_utils.cpp                         │  │
│  │  - concat_kernel_sources() — concatena .cl     │  │
│  │  - build_program_with_logs() — clBuildProgram  │  │
│  │  - enumerate_opencl_gpus()                     │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
                         │
                         │  clEnqueueNDRangeKernel
                         ▼
┌──────────────────────────────────────────────────────┐
│              Device (OpenCL C 2.0 / RDNA1)            │
│  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │ secp256k1_math.cl│  │  hash_pipeline.cl        │  │
│  │  - uint256 ops   │  │  - SHA-256 (64 rounds)   │  │
│  │  - mod_mult/sqr  │  │  - RIPEMD-160 (80 round) │  │
│  │  - mod_inv (Ferm)│  │  - Hash160 from limbs    │  │
│  └────────┬─────────┘  └────────────┬─────────────┘  │
│           └──────────┬──────────────┘                │
│                      ▼                                │
│  ┌────────────────────────────────────────────────┐  │
│  │           gpu_worker.cl                         │  │
│  │  - kernel_point_add_and_check_oneinv           │  │
│  │  - Batch inversion (Fermat tree)               │  │
│  │  - sub_group_shuffle / sub_group_any           │  │
│  │  - atomic_cmpxchg para "found"                 │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## Decisões técnicas do porte

### 1. Sem `__uint128_t` — usando `mul_hi()`
OpenCL C 2.0 não tem `__uint128_t`. A multiplicação 64×64→128 é feita com:
```c
ulong lo = a * b;            // v_mul_lo_u32 no GCN
ulong hi = mul_hi(a, b);     // v_mul_hi_u32 no GCN
```

### 2. Sem `__builtin_addcll` — função `awc()` manual
```c
__forceinline__ ulong awc(ulong a, ulong b, ulong cin, ulong *cout) {
    ulong s = a + b;  ulong c1 = (s < a) ? 1 : 0;
    ulong s2 = s + cin;  if (s2 < s) c1 = 1;
    *cout = c1;  return s2;
}
```
O compilador AMD OpenCL gera `v_add_co_u32` + `v_addc_co_u32` a partir desta construção.

### 3. Sub-group shuffles (cl_khr_subgroups)
HIP `__shfl`/`__shfl_down`/`__any` → OpenCL `sub_group_shuffle`/`sub_group_shuffle_down`/`sub_group_any`.
RDNA1 roda em **wave32** (não wave64), então o sub_group tem 32 lanes.

Fallback via `__local` memory está disponível compilando com `-DUSE_LOCAL_FALLBACK`.

### 4. `__constant` do HIP → buffers `__global`
O HIP permite `__device__ __constant__` de tamanho dinâmico definido em runtime. OpenCL não — arrays grandes passam como `__global const` com `__constant` para casos pequenos. O host cria `cl_mem` com `CL_MEM_READ_ONLY` e passa como kernel arg.

### 5. Concatenação de `.cl` em runtime
O compilador OpenCL da AMD não processa `#include`. O host (`ocl_utils.cpp → concat_kernel_sources`) lê os 4 arquivos, remove guards `#ifndef/#endif`, junta em uma única string e passa para `clCreateProgramWithSource`.

## Limitações conhecidas (NÃO testado em GPU real)

1. **`__attribute__((reqd_work_group_size(256,1,1)))`** — se o driver reclamar, remover do kernel.
2. **`cl_khr_subgroups`** — em drivers AMD antigos pode não estar disponível; usar `-DUSE_LOCAL_FALLBACK`.
3. **`atomic_work_item_fence` com `memory_scope_device`** — em algumas implementaçõess só funciona com `memory_scope_all_svm_devices`.
4. **OpenSSL link** — em Windows pode dar problema de DLL não encontrada; usar static link (`vcpkg install openssl:x64-windows-static`).
5. **ThreadsPerBlock=256** — RDNA1 tem 64 CUs × 4 SIMDs × 16 = 4096 work-items por CU; 256/block é seguro mas pode não ser ótimo. Testar com 128, 256, 512.
6. **`subp[MAX_BATCH_SIZE/2][4]`** no kernel — aloca `MAX_BATCH_SIZE/2 * 4 * 8 = 8192 bytes` no private. RDNA1 tem 65536 bytes de register file por work-item; com 256 threads/CU pode dar **register spill**. Se ocorrer, reduzir MAX_BATCH_SIZE para 1024 ou 512.

## Roadmap pós-porte (não implementado)

1. **Teste em RX 5700 XT real** — verificar clBuildProgram passa sem erros
2. **Benchmark** — comparar Mkeys/s vs HIP/ROCm original
3. **Bernstein-Yang `_ModInvBY`** — ~1.6× mais rápido que Fermat (já portado em secp256k1_math.cl, mas não conectado ao kernel principal)
4. **Endomorfismo secp256k1** — ganho ~1.3× (lambda split)
5. **Jacobian coordinates** — evita inversão modular na adição de pontos (ganho ~2× em batches pequenos)
6. **Optimização RDNA1-specific** — usar `__builtin_amdgcn_s_barrier` e wave32 explicit

## Créditos

- Original: [jmr2704/BRBtcHuntAMD](https://github.com/jmr2704/BRBtcHuntAMD) (HIP/ROCm)
- Inspiração: [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch) (CUDA)
- Algoritmo Bernstein-Yang: [eprint 2019/266](https://eprint.iacr.org/2019/266)

## Licença

Mesma licença do projeto original (MIT).
