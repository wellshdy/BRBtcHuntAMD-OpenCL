/*
 * ocl_common.h — Tipos compartilhados entre host (C++) e device (OpenCL C).
 *
 * Substitui AMDStructures.h + partes de AMDMath.h + GPUWorker.h do original
 * HIP, mas usando apenas C puro (sem classes, sem templates, sem operadores
 * sobrecarregados) para que os mesmos tipos possam ser usados em:
 *   - Host: #include em .cpp (compilado com cl.exe / g++)
 *   - Device: #include em .cl (compilado em runtime pelo driver OpenCL)
 *
 * Convenção:
 *   - Todos os campos são little-endian (limb 0 = bits 0..63).
 *   - bool é evitado nos structs que cruzam host/device (uso de uint8_t).
 *   - Alinhamento de 8 bytes para evitar padding inconsistente entre
 *     host x86_64 e device GCN/RDNA.
 */

#ifndef OCL_COMMON_H
#define OCL_COMMON_H

/*
 * Quando incluído por um .cl, __OPENCL_C_VERSION__ está definida.
 * Em OpenCL C os tipos stdint (uint8_t, uint32_t, uint64_t, ...) NÃO
 * existem por padrão — apenas uchar/uint/ulong/etc. Precisamos typedef'ar.
 */
#ifdef __OPENCL_C_VERSION__
  #define OPENCL_DEVICE 1
  #define OCL_HOST 0
  /* Mapear tipos stdint para tipos nativos OpenCL C. */
  typedef uchar  uint8_t;
  typedef char   int8_t;
  typedef ushort uint16_t;
  typedef short  int16_t;
  typedef uint   uint32_t;
  typedef int    int32_t;
  typedef ulong  uint64_t;
  typedef long   int64_t;
#else
  #define OPENCL_DEVICE 0
  #define OCL_HOST 1
  #include <cstdint>
  #include <cstddef>
#endif

/* ── Constantes de runtime ──────────────────────────────────────────── */
#ifndef WARP_SIZE
  /* RDNA1 roda em wave32; RDNA2/3 podem usar wave32 ou wave64. */
  #define WARP_SIZE 32
#endif

/*
 * MAX_BATCH_SIZE controla quantas chaves são processadas por batch
 * inversion. Cada thread aloca subp[MAX_BATCH_SIZE/2][4] ulong em
 * registradores — isso é 32 * (MAX_BATCH_SIZE/2) bytes por thread.
 *
 *   MAX_BATCH_SIZE = 64  → 1.0 KB por thread  (seguro)
 *   MAX_BATCH_SIZE = 128 → 2.0 KB por thread  (limite RDNA1)
 *   MAX_BATCH_SIZE = 256 → 4.0 KB por thread  (spill em RDNA1)
 *   MAX_BATCH_SIZE = 512 → 8.0 KB por thread  (spill pesado)
 *   MAX_BATCH_SIZE = 2048 → 32 KB por thread  (não compila em RDNA1)
 *
 * Default: 128. Pode ser override via -DMAX_BATCH_SIZE=N no clBuildProgram.
 */
#ifndef MAX_BATCH_SIZE
  #define MAX_BATCH_SIZE 256
#endif

#define FOUND_NONE  0
#define FOUND_LOCK  1
#define FOUND_READY 2

/* ── uint256: valor 256-bit little-endian (4 limbs × 64 bits) ───────── */
/*
 * No original C++ isto era uma classe com operator+ etc. Aqui é um
 * struct C puro; toda operação é feita por funções explícitas
 * (uint256_add, uint256_sub, ...) no arquivo secp256k1_math.cl.
 */
typedef struct {
    uint64_t v[4];   /* v[0] = bits 0..63, ..., v[3] = bits 192..255 */
} uint256;

/* ── uint320: 5 limbs (256-bit + 64-bit overflow para IMult/UMult) ──── */
typedef struct {
    uint64_t v[5];
} uint320;

/* ── Ponto afim na curva secp256k1 (X,Y) ou ponto-em-infinito ───────── */
/*
 * Equivalente ao `struct ECPointA` do AMDMath.h original, mas sem o
 * campo bool (que tem tamanho implementation-defined). Usamos uint8_t.
 */
typedef struct {
    uint64_t X[4];
    uint64_t Y[4];
    uint8_t  infinity;   /* 0 = ponto finito, 1 = infinito */
    uint8_t  _pad[7];    /* alinhamento de 8 bytes */
} ECPointA;

/* ── Resultado de "chave encontrada" (host ↔ device) ────────────────── */
typedef struct {
    int32_t  threadId;
    int32_t  iter;
    uint64_t scalar[4];
    uint64_t Rx[4];
    uint64_t Ry[4];
} FoundResult;

/* ── Resultado de vanity address (apenas device → host) ─────────────── */
typedef struct {
    uint64_t privkey[4];
    uint64_t pubkey_x[4];
    uint32_t hash160[5];
    uint8_t  prefix;       /* 0x02 ou 0x03 */
    uint8_t  _pad[3];
} VanityResult;

/* ── MatchResult (legacy do AMDHash.h) ──────────────────────────────── */
typedef struct {
    int32_t  found;
    uint8_t  publicKey[33];
    uint8_t  sha256[32];
    uint8_t  ripemd160[20];
} MatchResult;

/* ── Constantes da curva secp256k1 ──────────────────────────────────── */
/* Acessíveis de host e device. No .cl, vão para __constant. */
#define SECP256K1_P0 0xFFFFFFFEFFFFFC2FULL
#define SECP256K1_P1 0xFFFFFFFFFFFFFFFFULL
#define SECP256K1_P2 0xFFFFFFFFFFFFFFFFULL
#define SECP256K1_P3 0xFFFFFFFFFFFFFFFFULL

/* Gerador G */
#define SECP256K1_GX0 0x59F2815B16F81798ULL
#define SECP256K1_GX1 0x029BFCDB2DCE28D9ULL
#define SECP256K1_GX2 0x55A06295CE870B07ULL
#define SECP256K1_GX3 0x79BE667EF9DCBBACULL
#define SECP256K1_GY0 0x9C47D08FFB10D4B8ULL
#define SECP256K1_GY1 0xFD17B448A6855419ULL
#define SECP256K1_GY2 0x5DA4FBFC0E1108A8ULL
#define SECP256K1_GY3 0x483ADA7726A3C465ULL

/* Ordem n do grupo (não usado nos hot paths, mas mantido para futuras
   otimizações como endomorfismo). */
#define SECP256K1_N0 0xBFD25E8CD0364141ULL
#define SECP256K1_N1 0xBAAEDCE6AF48A03BULL
#define SECP256K1_N2 0xFFFFFFFFFFFFFFFEULL
#define SECP256K1_N3 0xFFFFFFFFFFFFFFFFULL

/*
 * Constante da redução modular.
 *
 *   p = 2^256 - 2^32 - 977 = 2^256 - 0x1000003D1
 *
 * Logo: 2^256 ≡ 0x1000003D1 (mod p)  (POSITIVE, não negativo)
 *
 * Na redução de um 512-bit r512[0..7]:
 *   high = r512[4..7] (256 bits altos)
 *   low  = r512[0..3] (256 bits baixos)
 *   valor total = low + high * 2^256 ≡ low + high * 0x1000003D1 (mod p)
 *
 * Portanto SOMAMOS high*C256 a low (não subtraímos).
 */
#define SECP256K1_C256 0x1000003D1ULL

/* ── Macros de alinhamento de buffer (host) ─────────────────────────── */
#if OCL_HOST
  #define OCL_ALIGNED(x) alignas(x)
#else
  #define OCL_ALIGNED(x)
#endif

#endif /* OCL_COMMON_H */
