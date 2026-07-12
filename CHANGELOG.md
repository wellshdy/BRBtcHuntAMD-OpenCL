# Changelog

## [Unreleased] — Revisão pós-entrega inicial

### Correções críticas de compilação

1. **`include/ocl_common.h`** — Adicionados typedefs para tipos stdint em modo OpenCL device.
   - `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`, `int8_t`, `int16_t`, `int32_t`, `int64_t`
   - Sem estes typedefs, structs como `FoundResult`, `VanityResult`, `ECPointA` falhariam ao compilar em OpenCL C (que só tem `uchar/uint/ulong/etc.` nativos).
   - Reduzido `MAX_BATCH_SIZE` default de 2048 para 128 (32KB → 2KB por thread). RDNA1 tem apenas 2KB de VGPR por thread; 2048 causaria spill massivo.
   - Corrigido comentário sobre `2^256 ≡ C256 (mod p)` — era `-C256`, agora `+C256` (POSITIVO).

2. **`kernels/secp256k1_math.cl`** — Várias correções:
   - Removidos inicializadores de array com valores não-const (proibidos em alguns drivers OpenCL):
     - `ulong res[5] = {1, 0, 0, 0, 0};` → atribuição explícita
     - `ulong base[5] = {R[0], R[1], ...};` → atribuição explícita
     - `ulong exp[4] = {...};` → atribuição explícita
     - `ulong t[5] = {in[0], ...};` em `fieldInv` → atribuição explícita
   - Corrigido `fieldAdd`: reimplementado sem macros para evitar bug de variável `borrow`/`carry` confundidas.
   - Corrigido `fieldSub`: mesmo bug — `borrow` era declarado mas as macros USUBO/USUBC usam `carry`.
   - Corrigido `scalarMulKernelBase`: `(size_t)idx * 4` → `(ulong)idx * 4` (size_t não existe em OpenCL C).
   - Corrigida macro `_IsOne`: parênteses faltando em torno de comparações.
   - Corrigido comentário sobre redução modular em `mod_mult`.

3. **`kernels/hash_pipeline.cl`** — Corrigido cast de `__constant` para `__private`:
   - `__private const uchar *t = (__private const uchar *)(target_hash160 + 4);` — PROIBIDO em OpenCL C.
   - Substituído por loop `for (int k = 0; k < 16; ++k) if (h[k] != target_hash160[k + 4]) return false;`.
   - Alterado parâmetro `h160_out` de `__private uint h160_out[5]` para `__private uint *h160_out` (compatível com NULL).

4. **`kernels/gpu_worker.cl`** — Várias correções:
   - Adicionados casts explícitos em todas as 8 chamadas atômicas:
     - `atomic_cmpxchg(d_found_flag, ...)` → `atomic_cmpxchg((__global atomic_int *)d_found_flag, ...)`
     - `atomic_xchg(d_found_flag, ...)` → `atomic_xchg((__global atomic_int *)d_found_flag, ...)`
   - Substituído `(size_t)` por `(ulong)` em todos os índices de array (size_t não existe em OpenCL C).
   - Kernel signature: `__global atomic_uint * d_vanity_count` → `__global uint * d_vanity_count` (cast feito na chamada atômica).
   - Função `vanity_check_and_save`: mesma correção no parâmetro `vanity_count`.
   - Fallback `USE_LOCAL_FALLBACK` simplificado (placeholder, com cast `(void)` para evitar warnings).

5. **`src/ocl_utils.cpp`** — Simplificação da concatenação de kernels:
   - Removida lógica de "drop first 3 lines" que era frágil (ocl_common.h tem 15 linhas de comentário antes do `#ifndef`).
   - Agora faz stream direto do conteúdo de cada arquivo; include guards internos evitam redefinições.
   - Adicionado `#include "ocl_helpers.h"` para usar definição compartilhada de `OclDeviceInfo`.
   - Em caso de erro de build, agora faz dump do source para `build_failed_source.cl` para debug.

6. **`include/ocl_helpers.h`** — Movido struct `OclDeviceInfo` para cá (compartilhado entre `ocl_utils.cpp` e `gpu_worker_opencl.cpp`):
   - Antes: `struct OclDeviceInfo;` forward declaration em `gpu_worker_opencl.cpp` era INCOMPLETA, não permitia `std::vector<OclDeviceInfo>`.
   - Agora: struct completo definido em header compartilhado.
   - Inclui `<CL/opencl.h>` diretamente (necessário para `cl_platform_id`, `cl_device_id`, `cl_uint`, `cl_ulong`).
   - Adicionado `<cmath>` (necessário para `std::ldexp`).

7. **`src/gpu_worker_opencl.cpp`**:
   - Removido `struct OclDeviceInfo;` forward declaration (agora vem de `ocl_helpers.h`).
   - Adicionado `#include <csignal>` (necessário para `sig_atomic_t`).
   - Removido `-Werror` das opções de build (drivers AMD têm warnings benignos).

8. **`src/main_opencl.cpp`**:
   - Removido `struct OclDeviceInfo;` forward declaration (agora vem de `ocl_helpers.h`).

### Validação

- ✅ Sintaxe C dos kernels validada com `gcc -c -std=c11` (apenas warnings de harness, zero erros).
- ✅ `ocl_utils.cpp` compila com `g++ -std=c++17`.
- ✅ `gpu_worker_opencl.cpp` compila com `g++ -std=c++17`.
- ✅ `main_opencl.cpp` compila com `g++ -std=c++17`.
- ❌ Ainda NÃO testado em GPU RDNA1 real com `clBuildProgram`.

### Pendências conhecidas

- `cl_khr_subgroups` deve estar disponível em RDNA1 + Adrenalin 23+; fallback `USE_LOCAL_FALLBACK` é placeholder.
- `MAX_BATCH_SIZE=128` pode ser aumentado para 256 ou 512 em RDNA2/3 (registradores extras).
- `__attribute__((reqd_work_group_size(256,1,1)))` no kernel força work-group size 256 — não usar `--grid A,B` com B≠256.
