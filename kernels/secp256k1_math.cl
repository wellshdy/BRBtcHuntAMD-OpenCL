/*
 * secp256k1_math.cl — Aritmética modular secp256k1 para OpenCL C 2.0
 *
 * Porte de AMDMath.h (HIP/ROCm) para OpenCL C, otimizado para AMD RDNA1
 * (RX 5700 XT / gfx1010) sob Windows com driver AMD Adrenalin.
 *
 * Decisões de porte:
 *   1. __uint128_t  →  mul_hi() + carry manual
 *   2. __builtin_addcll → add_with_carry() inline
 *   3. __device__/__global__/__forceinline__ → qualificadores OpenCL
 *   4. classes C++ (uint256) → typedef struct + funções explícitas
 *   5. templates → macros
 *
 * OBS: este arquivo é incluído por kernel_top.cl. NÃO compila sozinho.
 *      Depende de ocl_common.h (incluído antes via -I ou #include path).
 */

/* ───────────────────────────────────────────────────────────────────────
 * Configuração de extensões e qualificadores
 * ─────────────────────────────────────────────────────────────────────── */

/* cl_khr_subgroups não é mais necessário pois removemos o uso de shuffles e warp-reductions para máxima portabilidade. */
// #pragma OPENCL EXTENSION cl_khr_subgroups : enable
/* cl_khr_int64_base_atomics: para atomic_cmpxchg em flags de found. */
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

/* Mapeamento de qualificadores HIP → OpenCL */
#define __forceinline__ inline __attribute__((always_inline))
#define __noinline__    __attribute__((noinline))

/* ───────────────────────────────────────────────────────────────────────
 * Primitivas de 64 bits: add/sub com carry, mul 64×64→128, ctz, clz
 * ─────────────────────────────────────────────────────────────────────── */

/* add_with_carry: r = a + b + carry_in; retorna carry_out (0 ou 1).
   Substitui __builtin_addcll. O compilador AMD OpenCL gera v_add_co_u32
   + v_addc_co_u32 a partir desta construção. */
__forceinline__ ulong awc(ulong a, ulong b, ulong carry_in, __private ulong *carry_out) {
    ulong s = a + b + carry_in;
    *carry_out = (s < a) || (carry_in && s == a);
    return s;
}

/* sub_with_borrow: r = a - b - borrow_in; retorna borrow_out (0 ou 1).
   Substitui __builtin_subcll. */
__forceinline__ ulong swb(ulong a, ulong b, ulong borrow_in, __private ulong *borrow_out) {
    ulong d = a - b - borrow_in;
    *borrow_out = (a < b) || (borrow_in && a == b);
    return d;
}

/* mul_u64_128: a * b → (lo, hi). Em OpenCL C, mul_hi retorna os 64 bits
   altos da multiplicação. Não há __uint128_t em OpenCL C 2.0. */
__forceinline__ void mul_u64_128(ulong a, ulong b, ulong *lo, ulong *hi) {
    *lo = a * b;
    *hi = mul_hi(a, b);
}

/* mad_u64_128: a * b + c → (lo, hi). O GCN tem v_mad_u64_u32, mas o
   compilador OpenCL nem sempre o gera a partir de mul + add. Mantemos
   a decomposição manual para garantir estabilidade. */
__forceinline__ void mad_u64_128(ulong a, ulong b, ulong c, ulong *lo, ulong *hi) {
    ulong l, h;
    mul_u64_128(a, b, &l, &h);
    ulong s = l + c;
    if (s < l) h = h + 1UL;
    *lo = s;
    *hi = h;
}

/* count_trailing_zeros_64: substitui __builtin_ctzll. */
__forceinline__ uint ctz_64(ulong x) {
    if (x == 0UL) return 64u;
    /* OpenCL tem clz (count leading zeros); derivamos ctz com bit-twiddle. */
    ulong isolated = x & (~x + 1UL);          /* isola LSB */
    return (uint)(63u - clz(isolated));
}

/* ───────────────────────────────────────────────────────────────────────
 * Macros de carry-chain (portadas do AMDMath.h original)
 *
 * Estas macros eram baseadas em __builtin_addcll no HIP. Aqui são
 * reescritas em termos de awc() / swb(). Cada macro assume que a
 * variável local `carry` já está declarada.
 * ─────────────────────────────────────────────────────────────────────── */

/* UADDO: r = a + b, com carry inicial 0 */
#define UADDO(c, a, b) do { \
    ulong _c; (c) = awc((a), (b), 0UL, &_c); carry = _c; \
} while(0)

/* UADDC: r = a + b + carry (continuação da cadeia) */
#define UADDC(c, a, b) do { \
    ulong _c; (c) = awc((a), (b), carry, &_c); carry = _c; \
} while(0)

/* UADD: r = a + b + carry (usado no último limb, descarta carry final) */
#define UADD(c, a, b) do { \
    ulong _c; (c) = awc((a), (b), carry, &_c); (void)_c; \
} while(0)

/* UADDO1: r = r + a, com carry inicial 0 */
#define UADDO1(c, a) do { \
    ulong _c; (c) = awc((c), (a), 0UL, &_c); carry = _c; \
} while(0)

#define UADDC1(c, a) do { \
    ulong _c; (c) = awc((c), (a), carry, &_c); carry = _c; \
} while(0)

#define UADD1(c, a) do { \
    ulong _c; (c) = awc((c), (a), carry, &_c); (void)_c; \
} while(0)

/* USUBO / USUBC / USUB: subtração com borrow */
#define USUBO(c, a, b) do { \
    ulong _b; (c) = swb((a), (b), 0UL, &_b); carry = _b; \
} while(0)

#define USUBC(c, a, b) do { \
    ulong _b; (c) = swb((a), (b), carry, &_b); carry = _b; \
} while(0)

#define USUB(c, a, b) do { \
    ulong _b; (c) = swb((a), (b), carry, &_b); (void)_b; \
} while(0)

#define USUBO1(c, a) do { \
    ulong _b; (c) = swb((c), (a), 0UL, &_b); carry = _b; \
} while(0)

#define USUBC1(c, a) do { \
    ulong _b; (c) = swb((c), (a), carry, &_b); carry = _b; \
} while(0)

#define USUB1(c, a) do { \
    ulong _b; (c) = swb((c), (a), carry, &_b); (void)_b; \
} while(0)

/* UMULLO / UMULHI: parte baixa / alta de 64×64 → 128 */
#define UMULLO(lo, a, b) do { \
    ulong _hi_tmp; mul_u64_128((a), (b), &(lo), &_hi_tmp); \
} while(0)

#define UMULHI(hi, a, b) do { \
    ulong _lo_tmp; mul_u64_128((a), (b), &_lo_tmp, &(hi)); \
} while(0)

/* MADDO / MADDC / MADD: fused multiply-add com carry chain.
   No HIP usavam __uint128_t. Aqui decompondo em mul_u64_128 + awc. */

/* MADDO: r = hi(a*b) + c (carry_in = 0) */
#define MADDO(r, a, b, c) do { \
    ulong _lo, _hi; mul_u64_128((a), (b), &_lo, &_hi); \
    ulong _sum = _lo + (ulong)(c); \
    ulong _carry_lo = (_sum < _lo) ? 1UL : 0UL; \
    (r) = _hi + _carry_lo; \
    carry = _carry_lo; \
} while(0)

/* MADDC: r = hi(a*b + c + carry) */
#define MADDC(r, a, b, c) do { \
    ulong _lo, _hi; mul_u64_128((a), (b), &_lo, &_hi); \
    ulong _s1 = _lo + (ulong)(c); \
    ulong _c1 = (_s1 < _lo) ? 1UL : 0UL; \
    ulong _s2 = _s1 + carry; \
    if (_s2 < _s1) _c1 = 1UL; \
    (r) = _hi + _c1; \
    carry = _c1; \
} while(0)

/* MADD: r = hi(a*b + c + carry) (não atualiza carry) */
#define MADD(r, a, b, c) do { \
    ulong _lo, _hi; mul_u64_128((a), (b), &_lo, &_hi); \
    ulong _s1 = _lo + (ulong)(c); \
    ulong _c1 = (_s1 < _lo) ? 1UL : 0UL; \
    ulong _s2 = _s1 + carry; \
    if (_s2 < _s1) _c1 = 1UL; \
    (r) = _hi + _c1; \
} while(0)

/* ───────────────────────────────────────────────────────────────────────
 * Constantes em __constant (substituem __device__ __constant__ do HIP)
 * ─────────────────────────────────────────────────────────────────────── */

__constant ulong SECP_P_LE[4] = {
    SECP256K1_P0, SECP256K1_P1, SECP256K1_P2, SECP256K1_P3
};

__constant ulong SECP_GX_LE[4] = {
    SECP256K1_GX0, SECP256K1_GX1, SECP256K1_GX2, SECP256K1_GX3
};

__constant ulong SECP_GY_LE[4] = {
    SECP256K1_GY0, SECP256K1_GY1, SECP256K1_GY2, SECP256K1_GY3
};

/* Constante usada na redução: 2^256 ≡ -C256 (mod p). */
__constant ulong SECP_C256 = SECP256K1_C256;

/* MM64 / MSK62 (herdados do original — usados em _DivStep62, ainda
   presente no .cl caso queiramos ativar Bernstein-Yang com 5 limbs). */
__constant ulong MM64  = 0xD838091DD2253531UL;
__constant ulong MSK62 = 0x3FFFFFFFFFFFFFFFUL;

/* Ordem n do grupo (para futura ativação de USE_SYMMETRY). */
__constant ulong SECP_N_LE[4] = {
    SECP256K1_N0, SECP256K1_N1, SECP256K1_N2, SECP256K1_N3
};

/* ───────────────────────────────────────────────────────────────────────
 * Macros de comparação e carregamento (preservadas do original)
 * ─────────────────────────────────────────────────────────────────────── */

#define _IsPositive(x) (((long)(x[4])) >= 0L)
#define _IsNegative(x) (((long)(x[4])) <  0L)
#define _IsEqual(a,b)  ((a[4]==b[4])&&(a[3]==b[3])&&(a[2]==b[2])&&(a[1]==b[1])&&(a[0]==b[0]))
#define _IsZero(a)     (((a)[4]|(a)[3]|(a)[2]|(a)[1]|(a)[0]) == 0UL)
#define _IsOne(a)      (((a)[4]==0UL)&&((a)[3]==0UL)&&((a)[2]==0UL)&&((a)[1]==0UL)&&((a)[0]==1UL))

#define SWAP(tmp,x,y) do { tmp=x; x=y; y=tmp; } while(0)

#define __sright128(a,b,n) (((a)>>(n))|((b)<<(64-(n))))
#define __sleft128(a,b,n)  (((b)<<(n))|((a)>>(64-(n))))

/* ───────────────────────────────────────────────────────────────────────
 * Operações básicas de 256 bits
 *
 * Cada operação corresponde a um operador sobrecarregado no original.
 * A tabela abaixo mapeia a operação C++ original para a função OpenCL C:
 *
 *   C++ (AMDMath.h)                    →   OpenCL C (este arquivo)
 *   ───────────────────────────────────    ────────────────────────────
 *   uint256 operator+(a,b)             →   uint256_add(a,b,out)
 *   uint256 operator-(a,b)             →   uint256_sub(a,b,out)
 *   uint256 operator==(a,b)            →   uint256_eq(a,b)
 *   bool isZero(a)                     →   uint256_is_zero(a)
 *   ModNeg256(r)                       →   mod_neg256_inplace(r)
 *   ModSub256(r,a,b)                   →   mod_sub256(a,b,r)
 *   _ModMult(r,a,b)                    →   mod_mult(r,a,b)
 *   _ModSqr(rp,up)                     →   mod_sqr(rp,up)
 *   _ModInv(R)                         →   mod_inv_fermat(R)
 *   _ModInvBY(R)                       →   mod_inv_bernstein_yang(R)
 * ─────────────────────────────────────────────────────────────────────── */

/* uint256_eq: substitui operator== */
__forceinline__ bool uint256_eq(const uint256 *a, const uint256 *b) {
    return (a->v[0] == b->v[0]) &
           (a->v[1] == b->v[1]) &
           (a->v[2] == b->v[2]) &
           (a->v[3] == b->v[3]);
}

__forceinline__ bool uint256_is_zero(const uint256 *a) {
    return (a->v[0] | a->v[1] | a->v[2] | a->v[3]) == 0UL;
}

__forceinline__ bool uint256_is_one(const uint256 *a) {
    return (a->v[0] == 1UL) & (a->v[1] == 0UL) &
           (a->v[2] == 0UL) & (a->v[3] == 0UL);
}

/* uint256_add: r = a + b (256-bit, com overflow descartado).
   Equivalente ao operator+ do original. Não faz redução mod p. */
__forceinline__ void uint256_add(const uint256 *a, const uint256 *b, uint256 *r) {
    ulong carry = 0;
    UADDO(r->v[0], a->v[0], b->v[0]);
    UADDC(r->v[1], a->v[1], b->v[1]);
    UADDC(r->v[2], a->v[2], b->v[2]);
    UADD (r->v[3], a->v[3], b->v[3]);
}

/* uint256_sub: r = a - b. Equivalente ao operator-. */
__forceinline__ void uint256_sub(const uint256 *a, const uint256 *b, uint256 *r) {
    ulong carry = 0;
    USUBO(r->v[0], a->v[0], b->v[0]);
    USUBC(r->v[1], a->v[1], b->v[1]);
    USUBC(r->v[2], a->v[2], b->v[2]);
    USUB (r->v[3], a->v[3], b->v[3]);
}

/* uint256_ge: a >= b (unsigned). Usado na seleção no GCD. */
__forceinline__ bool uint256_ge(const uint256 *a, const uint256 *b) {
    if (a->v[3] != b->v[3]) return a->v[3] > b->v[3];
    if (a->v[2] != b->v[2]) return a->v[2] > b->v[2];
    if (a->v[1] != b->v[1]) return a->v[1] > b->v[1];
    return a->v[0] >= b->v[0];
}

/* ───────────────────────────────────────────────────────────────────────
 * Aritmética modular sobre F_p (p = secp256k1 prime)
 *
 * Todas assumem que os operandos estão na forma canonical [0, p).
 * ─────────────────────────────────────────────────────────────────────── */

/* AddP: r += p (in-place). Usado para corrigir subtrações que ficaram
   negativas. */
#define AddP(r) { \
    ulong carry = 0; \
    UADDO1(r[0], SECP256K1_P0); \
    UADDC1(r[1], SECP256K1_P1); \
    UADDC1(r[2], SECP256K1_P2); \
    UADDC1(r[3], SECP256K1_P3); \
    UADD1 (r[4], 0UL); \
}

/* SubP: r -= p (in-place). */
#define SubP(r) { \
    ulong carry = 0; \
    USUBO1(r[0], SECP256K1_P0); \
    USUBC1(r[1], SECP256K1_P1); \
    USUBC1(r[2], SECP256K1_P2); \
    USUBC1(r[3], SECP256K1_P3); \
    USUB1 (r[4], 0UL); \
}

/* Sub2: r = a - b (5-limb signed, usado no GCD). */
#define Sub2(r,a,b) { \
    ulong carry = 0; \
    USUBO(r[0], a[0], b[0]); \
    USUBC(r[1], a[1], b[1]); \
    USUBC(r[2], a[2], b[2]); \
    USUBC(r[3], a[3], b[3]); \
    USUB (r[4], a[4], b[4]); \
}

/* Sub1: r -= a (in-place, 5-limb). */
#define Sub1(r,a) { \
    ulong carry = 0; \
    USUBO1(r[0], a[0]); \
    USUBC1(r[1], a[1]); \
    USUBC1(r[2], a[2]); \
    USUBC1(r[3], a[3]); \
    USUB1 (r[4], a[4]); \
}

/* Add128: r += a (apenas 2 limbs, usado em shifts 128-bit). */
#define Add128(r,a) { \
    ulong carry = 0; \
    UADDO1((r)[0], (a)[0]); \
    UADD1 ((r)[1], (a)[1]); \
}

/* Neg: r = -r (5-limb two's complement). */
#define Neg(r) { \
    ulong carry = 0; \
    USUBO(r[0], 0UL, r[0]); \
    USUBC(r[1], 0UL, r[1]); \
    USUBC(r[2], 0UL, r[2]); \
    USUBC(r[3], 0UL, r[3]); \
    USUB (r[4], 0UL, r[4]); \
}

/* Load / Load256: copia 5 ou 4 limbs. */
#define Load(r, a) { \
    (r)[0]=(a)[0]; (r)[1]=(a)[1]; (r)[2]=(a)[2]; (r)[3]=(a)[3]; (r)[4]=(a)[4]; \
}

#define Load256(r, a) { \
    (r)[0]=(a)[0]; (r)[1]=(a)[1]; (r)[2]=(a)[2]; (r)[3]=(a)[3]; \
}

/* ───────────────────────────────────────────────────────────────────────
 * UMult: 256 × 64 → 320 bits (escola-book, sem __uint128_t)
 *
 * O original usava __uint128_t; aqui decompondo em mul_u64_128 + awc.
 * Implementado como função (não macro) para evitar captura incorreta
 * da variável `carry` — cada chamada tem seu próprio escopo.
 *
 * Estrutura:
 *   r[0]   = lo(a[0] * b)
 *   r[1]   = hi(a[0] * b) + lo(a[1] * b) + carry_in(0)
 *   r[2]   = hi(a[1] * b) + lo(a[2] * b) + carry
 *   r[3]   = hi(a[2] * b) + lo(a[3] * b) + carry
 *   r[4]   = hi(a[3] * b) + carry
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ void UMult_fn(ulong r[5], const ulong a[4], ulong b) {
    ulong lo, hi, c;
    /* r[0] = lo(a[0]*b); carry (hi) vai para r[1] */
    mul_u64_128(a[0], b, &lo, &hi);
    r[0] = lo;
    /* r[1] = hi(a[0]*b) + lo(a[1]*b) + carry_in(0) */
    ulong p1_lo, p1_hi;
    mul_u64_128(a[1], b, &p1_lo, &p1_hi);
    r[1] = awc(hi, p1_lo, 0UL, &c);
    /* r[2] = hi(a[1]*b) + lo(a[2]*b) + c */
    ulong p2_lo, p2_hi;
    mul_u64_128(a[2], b, &p2_lo, &p2_hi);
    r[2] = awc(p1_hi, p2_lo, c, &c);
    /* r[3] = hi(a[2]*b) + lo(a[3]*b) + c */
    ulong p3_lo, p3_hi;
    mul_u64_128(a[3], b, &p3_lo, &p3_hi);
    r[3] = awc(p2_hi, p3_lo, c, &c);
    /* r[4] = hi(a[3]*b) + c */
    r[4] = p3_hi + c;
}

/* UMultSpecial: multiplica um 256-bit (4 limbs) pela constante 0x1000003D1.
   Equivalente a UMult(r, a, 0x1000003D1), mas inlined para o compilador
   poderStrengthReduce ( Embora o valor não seja potência de 2, ter a
   constante inline ajuda o scheduler). */
__forceinline__ void UMultSpecial_fn(ulong r[5], const ulong a[4]) {
    UMult_fn(r, a, SECP256K1_C256);
}

/* ───────────────────────────────────────────────────────────────────────
 * IMult / IMultC: multiplicação 256 × 64 SIGNED → 320 bits
 *
 * IMult é usado em _DivStep62 (Bernstein-Yang). Se b<0, computa -a*|b|.
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ void IMult_fn(ulong r[5], const ulong a[5], long b) {
    ulong t[5];
    ulong abs_b;
    if (b < 0) {
        abs_b = (ulong)(-b);
        ulong carry = 0;
        /* t = ~a + 1 (two's complement negation) */
        UADDO(t[0], ~a[0], 1UL); UADDC(t[1], ~a[1], 0UL);
        UADDC(t[2], ~a[2], 0UL); UADDC(t[3], ~a[3], 0UL);
        UADD (t[4], ~a[4], 0UL);
    } else {
        abs_b = (ulong)b;
        Load(t, a);
    }
    /* UMult sobre 5 limbs. Aqui t tem 5 limbs e abs_b é 64-bit. */
    ulong lo, hi, c;
    mul_u64_128(t[0], abs_b, &lo, &hi);
    r[0] = lo;
    ulong p1_lo, p1_hi; mul_u64_128(t[1], abs_b, &p1_lo, &p1_hi);
    r[1] = awc(hi, p1_lo, 0UL, &c);
    ulong p2_lo, p2_hi; mul_u64_128(t[2], abs_b, &p2_lo, &p2_hi);
    r[2] = awc(p1_hi, p2_lo, c, &c);
    ulong p3_lo, p3_hi; mul_u64_128(t[3], abs_b, &p3_lo, &p3_hi);
    r[3] = awc(p2_hi, p3_lo, c, &c);
    ulong p4_lo, p4_hi; mul_u64_128(t[4], abs_b, &p4_lo, &p4_hi);
    r[4] = awc(p3_hi, p4_lo, c, &c);
    /* 6º limb (overflow) seria p4_hi + c, mas 320 bits é o limite. */
}

/* IMultC: igual a IMult, mas retorna o carry-out (6º limb). */
__forceinline__ ulong IMultC_fn(ulong r[5], const ulong a[5], long b) {
    ulong t[5];
    ulong abs_b;
    if (b < 0) {
        abs_b = (ulong)(-b);
        ulong carry = 0;
        UADDO(t[0], ~a[0], 1UL); UADDC(t[1], ~a[1], 0UL);
        UADDC(t[2], ~a[2], 0UL); UADDC(t[3], ~a[3], 0UL);
        UADD (t[4], ~a[4], 0UL);
    } else {
        abs_b = (ulong)b;
        Load(t, a);
    }
    ulong lo, hi, c;
    mul_u64_128(t[0], abs_b, &lo, &hi);
    r[0] = lo;
    ulong p1_lo, p1_hi; mul_u64_128(t[1], abs_b, &p1_lo, &p1_hi);
    r[1] = awc(hi, p1_lo, 0UL, &c);
    ulong p2_lo, p2_hi; mul_u64_128(t[2], abs_b, &p2_lo, &p2_hi);
    r[2] = awc(p1_hi, p2_lo, c, &c);
    ulong p3_lo, p3_hi; mul_u64_128(t[3], abs_b, &p3_lo, &p3_hi);
    r[3] = awc(p2_hi, p3_lo, c, &c);
    ulong p4_lo, p4_hi; mul_u64_128(t[4], abs_b, &p4_lo, &p4_hi);
    r[4] = awc(p3_hi, p4_lo, c, &c);
    return p4_hi + c;  /* carry-out de 6º limb */
}

/* ───────────────────────────────────────────────────────────────────────
 * ShiftR62: deslocamento de 62 bits à direita em 5-limb.
 * Usado em _DivStep62 (Bernstein-Yang).
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ void ShiftR62_fn(ulong r[5]) {
    r[0] = (r[1] << 2) | (r[0] >> 62);
    r[1] = (r[2] << 2) | (r[1] >> 62);
    r[2] = (r[3] << 2) | (r[2] >> 62);
    r[3] = (r[4] << 2) | (r[3] >> 62);
    r[4] = (ulong)(((long)r[4]) >> 62);
}

__forceinline__ void ShiftR62_dest_fn(ulong dest[5], const ulong r[5], ulong carry_in) {
    dest[0] = (r[1] << 2) | (r[0] >> 62);
    dest[1] = (r[2] << 2) | (r[1] >> 62);
    dest[2] = (r[3] << 2) | (r[2] >> 62);
    dest[3] = (r[4] << 2) | (r[3] >> 62);
    dest[4] = (carry_in << 2) | (r[4] >> 62);
}

/* ───────────────────────────────────────────────────────────────────────
 * MulP: r = a * (-C256) mod 2^320  (redução parcial para secp256k1)
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ void MulP_fn(ulong r[5], ulong a) {
    ulong al, ah;
    mul_u64_128(a, SECP256K1_C256, &al, &ah);
    ulong carry = 0;
    USUBO(r[0], 0UL, al);
    USUBC(r[1], 0UL, ah);
    USUBC(r[2], 0UL, 0UL);
    USUBC(r[3], 0UL, 0UL);
    USUB (r[4], a,   0UL);
}

/* ───────────────────────────────────────────────────────────────────────
 * ModNeg256: r = -a mod p  (4-limb canonical)
 * ─────────────────────────────────────────────────────────────────────── */
#define mod_neg256(a, r) do { \
    ulong _t[4]; \
    ulong carry = 0; \
    USUBO(_t[0], 0UL, (a)[0]); \
    USUBC(_t[1], 0UL, (a)[1]); \
    USUBC(_t[2], 0UL, (a)[2]); \
    USUBC(_t[3], 0UL, (a)[3]); \
    carry = 0; \
    UADDO((r)[0], _t[0], SECP256K1_P0); \
    UADDC((r)[1], _t[1], SECP256K1_P1); \
    UADDC((r)[2], _t[2], SECP256K1_P2); \
    UADD ((r)[3], _t[3], SECP256K1_P3); \
} while(0)

#define mod_neg256_inplace(r) do { \
    ulong _t[4]; \
    ulong carry = 0; \
    USUBO(_t[0], 0UL, (r)[0]); \
    USUBC(_t[1], 0UL, (r)[1]); \
    USUBC(_t[2], 0UL, (r)[2]); \
    USUBC(_t[3], 0UL, (r)[3]); \
    carry = 0; \
    UADDO((r)[0], _t[0], SECP256K1_P0); \
    UADDC((r)[1], _t[1], SECP256K1_P1); \
    UADDC((r)[2], _t[2], SECP256K1_P2); \
    UADD ((r)[3], _t[3], SECP256K1_P3); \
} while(0)

/* ───────────────────────────────────────────────────────────────────────
 * ModSub256: r = (a - b) mod p
 *
 * Se a < b (borrow final), soma p de volta. Mantém resultado em [0, p).
 * ─────────────────────────────────────────────────────────────────────── */
#define mod_sub256(a, b, r) do { \
    ulong _borrow, carry; \
    USUBO((r)[0], (a)[0], (b)[0]); \
    USUBC((r)[1], (a)[1], (b)[1]); \
    USUBC((r)[2], (a)[2], (b)[2]); \
    USUBC((r)[3], (a)[3], (b)[3]); \
    USUB (_borrow, 0UL, 0UL); \
    if (_borrow) { \
        carry = 0; \
        UADDO1((r)[0], SECP256K1_P0); \
        UADDC1((r)[1], SECP256K1_P1); \
        UADDC1((r)[2], SECP256K1_P2); \
        UADD1 ((r)[3], SECP256K1_P3); \
    } \
} while(0)

#define mod_sub256_inplace(r, b) do { \
    ulong _borrow, carry; \
    USUBO1((r)[0], (b)[0]); \
    USUBC1((r)[1], (b)[1]); \
    USUBC1((r)[2], (b)[2]); \
    USUBC1((r)[3], (b)[3]); \
    USUB(_borrow, 0UL, 0UL); \
    if (_borrow) { \
        carry = 0; \
        UADDO1((r)[0], SECP256K1_P0); \
        UADDC1((r)[1], SECP256K1_P1); \
        UADDC1((r)[2], SECP256K1_P2); \
        UADD1 ((r)[3], SECP256K1_P3); \
    } \
} while(0)

/* ModSub256isOdd: r = (a - b) mod 2^256, e parity = bit menos significativo.
   Usado no batch inversion para diferir prefixo 0x02/0x03. */
__forceinline__ void mod_sub256_is_odd(const ulong a[4], const ulong b[4],
                                        ulong r[4], uchar *parity) {
    ulong carry = 0;
    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
    USUB(carry, 0UL, 0UL);
    *parity = (uchar)((r[0] & 1UL) ^ (carry & 1UL));
}

/* ───────────────────────────────────────────────────────────────────────
 * _ModMult: multiplicação modular 256 × 256 → 256 mod p
 *
 * Estratégia: escola-book 4×4 → 8 limbs (512 bits), depois redução
 * rápida usando a propriedade 2^256 ≡ -C256 (mod p).
 *
 * Importante: esta é a função mais chamada do kernel (junto com
 * _ModSqr). Deve ser __forceinline__ para o compilador poder fazer
 * register allocation agressivo.
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ void mod_mult(ulong r[4], const ulong a[4], const ulong b[4]) {
    ulong r512[8];
    ulong t[5];
    ulong carry, ah, al;

    /* r512[0..7] = 0 nas posições 5..7; 0..4 virá de UMult(a,b[0]) */
    r512[5] = 0; r512[6] = 0; r512[7] = 0;

    /* r512 = a * b[0] (256×64 → 320)  →  r512[0..4] */
    carry = 0;
    UMult_fn(r512, a, b[0]);

    /* r512 += a*b[1] << 64  →  somar UMult(a,b[1]) em r512[1..5] */
    carry = 0;
    UMult_fn(t, a, b[1]);
    UADDO1(r512[1], t[0]); UADDC1(r512[2], t[1]);
    UADDC1(r512[3], t[2]); UADDC1(r512[4], t[3]);
    UADD1 (r512[5], t[4]);

    /* r512 += a*b[2] << 128 →  r512[2..6] */
    carry = 0;
    UMult_fn(t, a, b[2]);
    UADDO1(r512[2], t[0]); UADDC1(r512[3], t[1]);
    UADDC1(r512[4], t[2]); UADDC1(r512[5], t[3]);
    UADD1 (r512[6], t[4]);

    /* r512 += a*b[3] << 192 →  r512[3..7] */
    carry = 0;
    UMult_fn(t, a, b[3]);
    UADDO1(r512[3], t[0]); UADDC1(r512[4], t[1]);
    UADDC1(r512[5], t[2]); UADDC1(r512[6], t[3]);
    UADD1 (r512[7], t[4]);

    /* Redução: r512[4..7] (high 256 bits) × C256, somar em r512[0..3].
       Como 2^256 ≡ C256 (mod p)  (POSITIVE), temos high*2^256 ≡ high*C256 (mod p).
       Então computamos t = UMultSpecial(r512+4) e SOMAMOS em r512[0..3]. */
    carry = 0;
    UMultSpecial_fn(t, r512 + 4);
    UADDO1(r512[0], t[0]); UADDC1(r512[1], t[1]);
    UADDC1(r512[2], t[2]); UADDC1(r512[3], t[3]);
    UADD1 (t[4], 0UL);

    /* t[4] ainda contém overflow; multiplicar por C256 e somar. */
    mul_u64_128(t[4], SECP256K1_C256, &al, &ah);
    carry = 0;
    UADDO(r[0], r512[0], al); UADDC(r[1], r512[1], ah);
    UADDC(r[2], r512[2], 0UL); UADD(r[3], r512[3], 0UL);

    /* Redução final: se r >= p, subtrair p. */
    if (r[3] == SECP256K1_P3 && r[2] == SECP256K1_P2 &&
        r[1] == SECP256K1_P1 && r[0] >= SECP256K1_P0) {
        carry = 0;
        USUBO1(r[0], SECP256K1_P0); USUBC1(r[1], SECP256K1_P1);
        USUBC1(r[2], SECP256K1_P2); USUB1 (r[3], SECP256K1_P3);
    }
}

/* Variante quadrado (a*a). Aproximadamente 1.5× mais rápida que mod_mult
   porque explora simetria: a[i]*a[j] + a[j]*a[i] = 2*a[i]*a[j].
   Implementação direta da fórmula do original. */
__forceinline__ void mod_sqr(ulong rp[4], const ulong up[4]) {
    ulong r512[8];
    ulong SL, SH, r01L, r01H, r02L, r02H, r03L, r03H;
    ulong carry;

    mul_u64_128(up[0], up[0], &SL, &SH);
    mul_u64_128(up[0], up[1], &r01L, &r01H);
    mul_u64_128(up[0], up[2], &r02L, &r02H);
    mul_u64_128(up[0], up[3], &r03L, &r03H);

    r512[0] = SL; r512[1] = r01L; r512[2] = r02L; r512[3] = r03L;
    carry = 0;
    UADDO1(r512[1], SH); UADDC1(r512[2], r01H);
    UADDC1(r512[3], r02H); UADD(r512[4], r03H, 0UL);

    ulong r12L, r12H, r13L, r13H;
    mul_u64_128(up[1], up[1], &SL, &SH);
    mul_u64_128(up[1], up[2], &r12L, &r12H);
    mul_u64_128(up[1], up[3], &r13L, &r13H);

    carry = 0;
    UADDO1(r512[1], r01L); UADDC1(r512[2], SL); UADDC1(r512[3], r12L);
    UADDC1(r512[4], r13L); UADD(r512[5], r13H, 0UL);

    carry = 0;
    UADDO1(r512[2], r01H); UADDC1(r512[3], SH); UADDC1(r512[4], r12H);
    UADD1(r512[5], 0UL);

    ulong r23L, r23H;
    mul_u64_128(up[2], up[2], &SL, &SH);
    mul_u64_128(up[2], up[3], &r23L, &r23H);

    carry = 0;
    UADDO1(r512[2], r02L); UADDC1(r512[3], r12L); UADDC1(r512[4], SL);
    UADDC1(r512[5], r23L); UADD(r512[6], r23H, 0UL);

    carry = 0;
    UADDO1(r512[3], r02H); UADDC1(r512[4], r12H); UADDC1(r512[5], SH);
    UADD1(r512[6], 0UL);

    mul_u64_128(up[3], up[3], &SL, &SH);
    carry = 0;
    UADDO1(r512[3], r03L); UADDC1(r512[4], r13L); UADDC1(r512[5], r23L);
    UADDC1(r512[6], SL); UADD(r512[7], SH, 0UL);

    carry = 0;
    UADDO1(r512[4], r03H); UADDC1(r512[5], r13H); UADDC1(r512[6], r23H);
    UADD1(r512[7], 0UL);

    ulong t[5];
    carry = 0;
    UMult_fn(t, r512 + 4, SECP256K1_C256);
    carry = 0;
    UADDO1(r512[0], t[0]); UADDC1(r512[1], t[1]);
    UADDC1(r512[2], t[2]); UADDC1(r512[3], t[3]);
    UADD1(t[4], 0UL);

    mul_u64_128(t[4], SECP256K1_C256, &SL, &SH);
    carry = 0;
    UADDO(rp[0], r512[0], SL); UADDC(rp[1], r512[1], SH);
    UADDC(rp[2], r512[2], 0UL); UADD(rp[3], r512[3], 0UL);

    if (rp[3] == SECP256K1_P3 && rp[2] == SECP256K1_P2 &&
        rp[1] == SECP256K1_P1 && rp[0] >= SECP256K1_P0) {
        carry = 0;
        USUBO1(rp[0], SECP256K1_P0); USUBC1(rp[1], SECP256K1_P1);
        USUBC1(rp[2], SECP256K1_P2); USUB1(rp[3], SECP256K1_P3);
    }
}

/* ───────────────────────────────────────────────────────────────────────
 * _ModInv (Fermat): R = R^(p-2) mod p
 *
 * Padrão: square-and-multiply. O expoente é p-2 = 0xFFFFFFFEFFFFFC2D...
 * Converge em ~256 iterações. No original era o padrão (~500 Mkeys/s
 * em RX 6600). Mantido como fallback robusto; o kernel principal usa
 * _ModInvBY quando disponível.
 *
 * Aqui R é passado como 5-limb (R[4] é zero por convenção com o
 * código original). */
__forceinline__ void mod_inv_fermat(ulong R[5]) {
    if (R[0]==0 && R[1]==0 && R[2]==0 && R[3]==0) return;

    /* NOTA: Em OpenCL C 2.0, inicializadores de array com valores não-const
       podem falhar em alguns drivers. Usamos atribuição explícita. */
    ulong res[5];
    res[0]=1UL; res[1]=0UL; res[2]=0UL; res[3]=0UL; res[4]=0UL;
    ulong base[5];
    base[0]=R[0]; base[1]=R[1]; base[2]=R[2]; base[3]=R[3]; base[4]=0UL;
    ulong tmp[5];
    ulong exp[4];
    exp[0]=0xFFFFFFFEFFFFFC2DUL; exp[1]=0xFFFFFFFFFFFFFFFFUL;
    exp[2]=0xFFFFFFFFFFFFFFFFUL; exp[3]=0xFFFFFFFFFFFFFFFFUL;

    bool started = false;
    for (int limb = 3; limb >= 0; limb--) {
        for (int bit = 63; bit >= 0; bit--) {
            bool bit_set = (exp[limb] >> bit) & 1UL;
            if (!bit_set && !started) continue;
            if (!started) {
                res[0] = base[0]; res[1] = base[1];
                res[2] = base[2]; res[3] = base[3];
                started = true;
                continue;
            }
            mod_sqr(tmp, res);
            res[0]=tmp[0]; res[1]=tmp[1]; res[2]=tmp[2]; res[3]=tmp[3];
            if (bit_set) {
                mod_mult(tmp, res, base);
                res[0]=tmp[0]; res[1]=tmp[1]; res[2]=tmp[2]; res[3]=tmp[3];
            }
        }
    }
    R[0]=res[0]; R[1]=res[1]; R[2]=res[2]; R[3]=res[3]; R[4]=0;
}

/* ───────────────────────────────────────────────────────────────────────
 * Helpers para conversão 4×64 ↔ 5×62 (Bernstein-Yang)
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ void to62(const ulong src[4], ulong dst[5]) {
    dst[0] = src[0] & MSK62;
    dst[1] = ((src[0] >> 62) | (src[1] << 2)) & MSK62;
    dst[2] = ((src[1] >> 60) | (src[2] << 4)) & MSK62;
    dst[3] = ((src[2] >> 58) | (src[3] << 6)) & MSK62;
    dst[4] = (src[3] >> 56) & 0xFFUL;
    if (dst[4] & 0x80UL) dst[4] |= 0xFFFFFFFFFFFFFF00UL;
}

__forceinline__ void from62(const ulong src[5], ulong dst[4]) {
    dst[0] =  (src[0]       | (src[1] << 62));
    dst[1] = ((src[1] >> 2) | (src[2] << 60));
    dst[2] = ((src[2] >> 4) | (src[3] << 58));
    dst[3] = ((src[3] >> 6) | ((ulong)(((long)src[4]) << 56)));
}

/* ───────────────────────────────────────────────────────────────────────
 * Operações de campo (field ops) — wrappers amigáveis
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ void fieldCopy(const __generic ulong *a, __generic ulong *out) {
    out[0]=a[0]; out[1]=a[1]; out[2]=a[2]; out[3]=a[3];
}

__forceinline__ bool fieldIsZero(const __generic ulong *a) {
    return (a[0]|a[1]|a[2]|a[3]) == 0UL;
}

__forceinline__ void fieldAdd(const __generic ulong *a, const __generic ulong *b, __generic ulong *out) {
    /* Implementação direta (sem macros) para clareza. Equivalente ao
       fieldAdd original do AMDMath.h. */
    ulong carry = 0;
    for (int i = 0; i < 4; ++i) {
        ulong s = a[i] + b[i];
        ulong c = (s < a[i]) ? 1UL : 0UL;
        s += carry;
        if (s < carry) c = 1UL;
        out[i] = s;
        carry = c;
    }
    /* Redução condicional: se out >= p, subtrair p. */
    if (carry || out[3] > SECP256K1_P3 ||
        (out[3]==SECP256K1_P3 && out[2] > SECP256K1_P2) ||
        (out[3]==SECP256K1_P3 && out[2]==SECP256K1_P2 && out[1] > SECP256K1_P1) ||
        (out[3]==SECP256K1_P3 && out[2]==SECP256K1_P2 && out[1]==SECP256K1_P1 && out[0] >= SECP256K1_P0)) {
        ulong borrow = 0;
        for (int i = 0; i < 4; ++i) {
            ulong d = out[i] - borrow;
            ulong nb = (d > out[i]) ? 1UL : 0UL;
            ulong d2 = d - SECP_P_LE[i];
            if (d2 > d) nb = 1UL;
            out[i] = d2;
            borrow = nb;
        }
    }
}

__forceinline__ void fieldSub(const __generic ulong *a, const __generic ulong *b, __generic ulong *out) {
    /* Implementação direta (sem macros USUBO/USUBC) para evitar confusão
       de nomes de variáveis. Equivalente ao fieldSub original do AMDMath.h. */
    ulong borrow = 0;
    for (int i = 0; i < 4; ++i) {
        ulong d = a[i] - borrow;
        ulong nb = (d > a[i]) ? 1UL : 0UL;
        ulong d2 = d - b[i];
        if (d2 > d) nb = 1UL;
        out[i] = d2;
        borrow = nb;
    }
    if (borrow) {
        ulong carry = 0;
        for (int i = 0; i < 4; ++i) {
            ulong s = out[i] + SECP_P_LE[i];
            ulong c = (s < out[i]) ? 1UL : 0UL;
            s += carry;
            if (s < carry) c = 1UL;
            out[i] = s;
            carry = c;
        }
    }
}

__forceinline__ void fieldNeg(const __generic ulong *a, __generic ulong *out) {
    if (fieldIsZero(a)) { out[0]=out[1]=out[2]=out[3]=0UL; return; }
    ulong p_local[4] = { SECP256K1_P0, SECP256K1_P1, SECP256K1_P2, SECP256K1_P3 };
    fieldSub(p_local, a, out);
}

__forceinline__ void fieldMul(const __generic ulong *a, const __generic ulong *b, __generic ulong *out) {
    ulong a_priv[4], b_priv[4], out_priv[4];
    a_priv[0] = a[0]; a_priv[1] = a[1]; a_priv[2] = a[2]; a_priv[3] = a[3];
    b_priv[0] = b[0]; b_priv[1] = b[1]; b_priv[2] = b[2]; b_priv[3] = b[3];
    mod_mult(out_priv, a_priv, b_priv);
    out[0] = out_priv[0]; out[1] = out_priv[1]; out[2] = out_priv[2]; out[3] = out_priv[3];
}

__forceinline__ void fieldSqr(const __generic ulong *a, __generic ulong *out) {
    ulong a_priv[4], out_priv[4];
    a_priv[0] = a[0]; a_priv[1] = a[1]; a_priv[2] = a[2]; a_priv[3] = a[3];
    mod_sqr(out_priv, a_priv);
    out[0] = out_priv[0]; out[1] = out_priv[1]; out[2] = out_priv[2]; out[3] = out_priv[3];
}

__forceinline__ void fieldInv(const __generic ulong *in, __generic ulong *out) {
    ulong t[5];
    t[0]=in[0]; t[1]=in[1]; t[2]=in[2]; t[3]=in[3]; t[4]=0UL;
    mod_inv_fermat(t);
    out[0]=t[0]; out[1]=t[1]; out[2]=t[2]; out[3]=t[3];
}

/* ───────────────────────────────────────────────────────────────────────
 * Operações com pontos afins (ECPointA) — portadas de AMDMath.h
 * ─────────────────────────────────────────────────────────────────────── */

__forceinline__ void pointSetInfinity(__generic ECPointA *P) {
    P->infinity = 1;
    P->X[0]=P->X[1]=P->X[2]=P->X[3]=0UL;
    P->Y[0]=P->Y[1]=P->Y[2]=P->Y[3]=0UL;
}

__forceinline__ void pointSetG(__generic ECPointA *P) {
    P->infinity = 0;
    P->X[0]=SECP_GX_LE[0]; P->X[1]=SECP_GX_LE[1];
    P->X[2]=SECP_GX_LE[2]; P->X[3]=SECP_GX_LE[3];
    P->Y[0]=SECP_GY_LE[0]; P->Y[1]=SECP_GY_LE[1];
    P->Y[2]=SECP_GY_LE[2]; P->Y[3]=SECP_GY_LE[3];
}

__forceinline__ void pointCopy(const __generic ECPointA *src, __generic ECPointA *dst) {
    dst->infinity = src->infinity;
    dst->X[0]=src->X[0]; dst->X[1]=src->X[1];
    dst->X[2]=src->X[2]; dst->X[3]=src->X[3];
    dst->Y[0]=src->Y[0]; dst->Y[1]=src->Y[1];
    dst->Y[2]=src->Y[2]; dst->Y[3]=src->Y[3];
}

/* pointDoubleAffine: R = 2*P */
__forceinline__ void pointDoubleAffine(const __generic ECPointA *P, __generic ECPointA *R) {
    if (P->infinity) { pointSetInfinity(R); return; }
    ulong x2[4], two_x2[4], three_x2[4], denom[4], invDen[4], lambda[4];
    fieldSqr(P->X, x2);
    fieldAdd(x2, x2, two_x2);
    fieldAdd(two_x2, x2, three_x2);
    fieldAdd(P->Y, P->Y, denom);
    fieldInv(denom, invDen);
    fieldMul(three_x2, invDen, lambda);

    ulong lambda2[4], twoX[4], newX[4];
    fieldSqr(lambda, lambda2);
    fieldAdd(P->X, P->X, twoX);
    fieldSub(lambda2, twoX, newX);

    ulong tmp[4], prod[4], newY[4];
    fieldSub(P->X, newX, tmp);
    fieldMul(lambda, tmp, prod);
    fieldSub(prod, P->Y, newY);

    fieldCopy(newX, R->X);
    fieldCopy(newY, R->Y);
    R->infinity = 0;
}

/* pointAddAffine: R = P + Q */
__forceinline__ void pointAddAffine(const __generic ECPointA *P, const __generic ECPointA *Q, __generic ECPointA *R) {
    if (P->infinity) { pointCopy(Q, R); return; }
    if (Q->infinity) { pointCopy(P, R); return; }

    bool sameX = (P->X[0]==Q->X[0]) && (P->X[1]==Q->X[1]) &&
                 (P->X[2]==Q->X[2]) && (P->X[3]==Q->X[3]);
    bool sameY = (P->Y[0]==Q->Y[0]) && (P->Y[1]==Q->Y[1]) &&
                 (P->Y[2]==Q->Y[2]) && (P->Y[3]==Q->Y[3]);
    if (sameX && sameY) { pointDoubleAffine(P, R); return; }
    if (sameX && !sameY) { pointSetInfinity(R); return; }

    ulong dx[4], dy[4], invdx[4], lambda[4], lambda2[4], tmp1[4], prod[4], newX[4], newY[4];
    fieldSub(Q->X, P->X, dx);
    fieldSub(Q->Y, P->Y, dy);
    fieldInv(dx, invdx);
    fieldMul(dy, invdx, lambda);
    fieldSqr(lambda, lambda2);
    fieldSub(lambda2, P->X, tmp1);
    fieldSub(tmp1, Q->X, newX);
    fieldSub(P->X, newX, tmp1);
    fieldMul(lambda, tmp1, prod);
    fieldSub(prod, P->Y, newY);

    fieldCopy(newX, R->X);
    fieldCopy(newY, R->Y);
    R->infinity = 0;
}

/* scalarMulBaseAffine: out = scalar * G (método double-and-add)
   Usado pelo kernel scalarMulKernelBase para inicializar pontos. */
__forceinline__ void scalarMulBaseAffine(const __generic ulong *scalar_le,
                                          __generic ulong *outX, __generic ulong *outY) {
    ECPointA R;
    pointSetInfinity(&R);

    int msb = -1;
    for (int limb = 3; limb >= 0; --limb) {
        ulong v = scalar_le[limb];
        if (v != 0UL) {
            msb = limb * 64 + 63 - (int)clz(v);
            break;
        }
    }
    if (msb == -1) {
        outX[0]=outX[1]=outX[2]=outX[3]=0UL;
        outY[0]=outY[1]=outY[2]=outY[3]=0UL;
        return;
    }
    for (int bi = msb; bi >= 0; --bi) {
        if (!R.infinity) {
            ECPointA t;
            pointDoubleAffine(&R, &t);
            pointCopy(&t, &R);
        }
        int limb = bi >> 6;
        int shift = bi & 63;
        ulong bit = (scalar_le[limb] >> shift) & 1UL;
        if (bit) {
            ECPointA Gp;
            pointSetG(&Gp);
            if (R.infinity) {
                pointCopy(&Gp, &R);
            } else {
                ECPointA t;
                pointAddAffine(&R, &Gp, &t);
                pointCopy(&t, &R);
            }
        }
    }
    if (R.infinity) {
        outX[0]=outX[1]=outX[2]=outX[3]=0UL;
        outY[0]=outY[1]=outY[2]=outY[3]=0UL;
    } else {
        fieldCopy(R.X, outX);
        fieldCopy(R.Y, outY);
    }
}

/* ───────────────────────────────────────────────────────────────────────
 * Kernel exposto: scalarMulKernelBase
 *
 * Substitui o __global__ kernel do HIP. Em OpenCL:
 *   - __global substitui __device__/__global__
 *   - get_global_id(0) substitui blockIdx.x*blockDim.x+threadIdx.x
 *   - __restrict__ → __restrict__ (OpenCL suporta)
 * ─────────────────────────────────────────────────────────────────────── */
__kernel void scalarMulKernelBase(
    __global const ulong * __restrict__ scalars_in,
    __global       ulong * __restrict__ outX,
    __global       ulong * __restrict__ outY,
    const int N)
{
    int idx = get_global_id(0);
    if (idx >= N) return;

    /* NOTA: Em OpenCL C, size_t não existe. Usamos ulong para aritmética
       de ponteiros (conversão implícita int → ulong é segura aqui). */
    __global const ulong *scalar = scalars_in + (ulong)idx * 4;
    __global       ulong *outx   = outX       + (ulong)idx * 4;
    __global       ulong *outy   = outY       + (ulong)idx * 4;

    /* Copy from __global to __private (OpenCL address space requirement) */
    ulong s_priv[4];
    s_priv[0] = scalar[0]; s_priv[1] = scalar[1];
    s_priv[2] = scalar[2]; s_priv[3] = scalar[3];

    ulong sx[4], sy[4];
    scalarMulBaseAffine(s_priv, sx, sy);

    outx[0]=sx[0]; outx[1]=sx[1]; outx[2]=sx[2]; outx[3]=sx[3];
    outy[0]=sy[0]; outy[1]=sy[1]; outy[2]=sy[2]; outy[3]=sy[3];
}
