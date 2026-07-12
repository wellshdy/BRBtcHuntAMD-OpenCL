/*
 * hash_pipeline.cl — SHA-256 + RIPEMD-160 inline para OpenCL C 2.0
 *
 * Porte direto de HashPipeline.cpp (HIP). Quase não há mudanças
 * porque o código já era majoritariamente C puro com macros.
 *
 * Mudanças:
 *   - __device__ → removido (qualificador padrão em .cl é __private)
 *   - __forceinline__ → inline __attribute__((always_inline))
 *   - __constant__ → __constant
 *   - ulonglong2 + memcpy → loads manuais (sem vector types HIP)
 *   - __builtin_bswap32 → macro própria
 */

/* ───────────────────────────────────────────────────────────────────────
 * Helpers de bitwise
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ uint ror32(uint x, int n) {
    return (x >> n) | (x << (32 - n));
}

__forceinline__ uint bigS0(uint x)  { return ror32(x, 2)  ^ ror32(x, 13) ^ ror32(x, 22); }
__forceinline__ uint bigS1(uint x)  { return ror32(x, 6)  ^ ror32(x, 11) ^ ror32(x, 25); }
__forceinline__ uint smallS0(uint x){ return ror32(x, 7)  ^ ror32(x, 18) ^ (x >> 3);     }
__forceinline__ uint smallS1(uint x){ return ror32(x, 17) ^ ror32(x, 19) ^ (x >> 10);   }

__forceinline__ uint Ch (uint x, uint y, uint z) { return (x & y) ^ (~x & z); }
__forceinline__ uint Maj(uint x, uint y, uint z) { return (x & y) | (x & z) | (y & z); }

/* bswap32: troca bytes de um uint32. OpenCL não tem __builtin_bswap32. */
__forceinline__ uint bswap32(uint x) {
    return ((x & 0x000000FFu) << 24) |
           ((x & 0x0000FF00u) <<  8) |
           ((x & 0x00FF0000u) >>  8) |
           ((x & 0xFF000000u) >> 24);
}

__forceinline__ uint pack_be4(uchar a, uchar b, uchar c, uchar d) {
    return ((uint)a << 24) | ((uint)b << 16) | ((uint)c << 8) | (uint)d;
}

/* ───────────────────────────────────────────────────────────────────────
 * SHA-256 constant table
 * ─────────────────────────────────────────────────────────────────────── */
__constant uint K_SHA256[64] = {
    0x428A2F98u,0x71374491u,0xB5C0FBCFu,0xE9B5DBA5u,0x3956C25Bu,0x59F111F1u,0x923F82A4u,0xAB1C5ED5u,
    0xD807AA98u,0x12835B01u,0x243185BEu,0x550C7DC3u,0x72BE5D74u,0x80DEB1FEu,0x9BDC06A7u,0xC19BF174u,
    0xE49B69C1u,0xEFBE4786u,0x0FC19DC6u,0x240CA1CCu,0x2DE92C6Fu,0x4A7484AAu,0x5CB0A9DCu,0x76F988DAu,
    0x983E5152u,0xA831C66Du,0xB00327C8u,0xBF597FC7u,0xC6E00BF3u,0xD5A79147u,0x06CA6351u,0x14292967u,
    0x27B70A85u,0x2E1B2138u,0x4D2C6DFCu,0x53380D13u,0x650A7354u,0x766A0ABBu,0x81C2C92Eu,0x92722C85u,
    0xA2BFE8A1u,0xA81A664Bu,0xC24B8B70u,0xC76C51A3u,0xD192E819u,0xD6990624u,0xF40E3585u,0x106AA070u,
    0x19A4C116u,0x1E376C08u,0x2748774Cu,0x34B0BCB5u,0x391C0CB3u,0x4ED8AA4Au,0x5B9CCA4Fu,0x682E6FF3u,
    0x748F82EEu,0x78A5636Fu,0x84C87814u,0x8CC70208u,0x90BEFFFAu,0xA4506CEBu,0xBEF9A3F7u,0xC67178F2u
};

/* ───────────────────────────────────────────────────────────────────────
 * SHA-256 init + transform (64 rounds, message schedule in-place)
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ void SHA256Initialize(uint s[8]) {
    s[0] = 0x6a09e667u;
    s[1] = 0xbb67ae85u;
    s[2] = 0x3c6ef372u;
    s[3] = 0xa54ff53au;
    s[4] = 0x510e527fu;
    s[5] = 0x9b05688cu;
    s[6] = 0x1f83d9abu;
    s[7] = 0x5be0cd19u;
}

__forceinline__ void SHA256Transform(uint state[8], const uint W_in[16]) {
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];

    uint w[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = W_in[i];

    /* 64 rounds. O pragma unroll do OpenCL C funciona, mas o compilador
       AMD pode usar muito VGPR; mantemos o loop manual caso queira trocar. */
    for (int t = 0; t < 64; ++t) {
        if (t >= 16) {
            uint s0 = smallS0(w[(t + 1)  & 15]);
            uint s1 = smallS1(w[(t + 14) & 15]);
            uint newW = w[t & 15] + s1 + w[(t + 9) & 15] + s0;
            w[t & 15] = newW;
        }
        uint Wt = w[t & 15];
        uint T1 = h + bigS1(e) + Ch(e, f, g) + K_SHA256[t] + Wt;
        uint T2 = bigS0(a) + Maj(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* ───────────────────────────────────────────────────────────────────────
 * RIPEMD-160
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ void RIPEMD160Initialize(uint s[5]) {
    s[0] = 0x67452301u;
    s[1] = 0xEFCDAB89u;
    s[2] = 0x98BADCFEu;
    s[3] = 0x10325476u;
    s[4] = 0xC3D2E1F0u;
}

#define ROL(x,n) ((x >> (32 - n)) | (x << n))
#define f1(x, y, z) (x ^ y ^ z)
#define f2(x, y, z) ((x & y) | (~x & z))
#define f3(x, y, z) ((x | ~y) ^ z)
#define f4(x, y, z) ((x & z) | (~z & y))
#define f5(x, y, z) (x ^ (y | ~z))

#define RPRound(a,b,c,d,e,f,x,k,r) \
    u = a + f + x + k; \
    a = ROL(u, r) + e; \
    c = ROL(c, 10);

#define R11(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0,        r)
#define R21(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f2(b, c, d), x, 0x5A827999u, r)
#define R31(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6ED9EBA1u, r)
#define R41(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f4(b, c, d), x, 0x8F1BBCDCu, r)
#define R51(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f5(b, c, d), x, 0xA953FD4Eu, r)
#define R12(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f5(b, c, d), x, 0x50A28BE6u, r)
#define R22(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f4(b, c, d), x, 0x5C4DD124u, r)
#define R32(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6D703EF3u, r)
#define R42(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f2(b, c, d), x, 0x7A6D76E9u, r)
#define R52(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0,        r)

__forceinline__ void RIPEMD160Transform(uint s[5], const uint w_in[16]) {
    uint u;
    uint a1 = s[0], b1 = s[1], c1 = s[2], d1 = s[3], e1 = s[4];
    uint a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;

    /* Copia para w local para evitar aliasing */
    uint w[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = w_in[i];

    R11(a1, b1, c1, d1, e1, w[0], 11);  R12(a2, b2, c2, d2, e2, w[5], 8);
    R11(e1, a1, b1, c1, d1, w[1], 14);  R12(e2, a2, b2, c2, d2, w[14], 9);
    R11(d1, e1, a1, b1, c1, w[2], 15);  R12(d2, e2, a2, b2, c2, w[7], 9);
    R11(c1, d1, e1, a1, b1, w[3], 12);  R12(c2, d2, e2, a2, b2, w[0], 11);
    R11(b1, c1, d1, e1, a1, w[4], 5);   R12(b2, c2, d2, e2, a2, w[9], 13);
    R11(a1, b1, c1, d1, e1, w[5], 8);   R12(a2, b2, c2, d2, e2, w[2], 15);
    R11(e1, a1, b1, c1, d1, w[6], 7);   R12(e2, a2, b2, c2, d2, w[11], 15);
    R11(d1, e1, a1, b1, c1, w[7], 9);   R12(d2, e2, a2, b2, c2, w[4], 5);
    R11(c1, d1, e1, a1, b1, w[8], 11);  R12(c2, d2, e2, a2, b2, w[13], 7);
    R11(b1, c1, d1, e1, a1, w[9], 13);  R12(b2, c2, d2, e2, a2, w[6], 7);
    R11(a1, b1, c1, d1, e1, w[10], 14); R12(a2, b2, c2, d2, e2, w[15], 8);
    R11(e1, a1, b1, c1, d1, w[11], 15); R12(e2, a2, b2, c2, d2, w[8], 11);
    R11(d1, e1, a1, b1, c1, w[12], 6);  R12(d2, e2, a2, b2, c2, w[1], 14);
    R11(c1, d1, e1, a1, b1, w[13], 7);  R12(c2, d2, e2, a2, b2, w[10], 14);
    R11(b1, c1, d1, e1, a1, w[14], 9);  R12(b2, c2, d2, e2, a2, w[3], 12);
    R11(a1, b1, c1, d1, e1, w[15], 8);  R12(a2, b2, c2, d2, e2, w[12], 6);

    R21(e1, a1, b1, c1, d1, w[7], 7);   R22(e2, a2, b2, c2, d2, w[6], 9);
    R21(d1, e1, a1, b1, c1, w[4], 6);   R22(d2, e2, a2, b2, c2, w[11], 13);
    R21(c1, d1, e1, a1, b1, w[13], 8);  R22(c2, d2, e2, a2, b2, w[3], 15);
    R21(b1, c1, d1, e1, a1, w[1], 13);  R22(b2, c2, d2, e2, a2, w[7], 7);
    R21(a1, b1, c1, d1, e1, w[10], 11); R22(a2, b2, c2, d2, e2, w[0], 12);
    R21(e1, a1, b1, c1, d1, w[6], 9);   R22(e2, a2, b2, c2, d2, w[13], 8);
    R21(d1, e1, a1, b1, c1, w[15], 7);  R22(d2, e2, a2, b2, c2, w[5], 9);
    R21(c1, d1, e1, a1, b1, w[3], 15);  R22(c2, d2, e2, a2, b2, w[10], 11);
    R21(b1, c1, d1, e1, a1, w[12], 7);  R22(b2, c2, d2, e2, a2, w[14], 7);
    R21(a1, b1, c1, d1, e1, w[0], 12);  R22(a2, b2, c2, d2, e2, w[15], 7);
    R21(e1, a1, b1, c1, d1, w[9], 15);  R22(e2, a2, b2, c2, d2, w[8], 12);
    R21(d1, e1, a1, b1, c1, w[5], 9);   R22(d2, e2, a2, b2, c2, w[12], 7);
    R21(c1, d1, e1, a1, b1, w[2], 11);  R22(c2, d2, e2, a2, b2, w[4], 6);
    R21(b1, c1, d1, e1, a1, w[14], 7);  R22(b2, c2, d2, e2, a2, w[9], 15);
    R21(a1, b1, c1, d1, e1, w[11], 13); R22(a2, b2, c2, d2, e2, w[1], 13);
    R21(e1, a1, b1, c1, d1, w[8], 12);  R22(e2, a2, b2, c2, d2, w[2], 11);

    R31(d1, e1, a1, b1, c1, w[3], 11);  R32(d2, e2, a2, b2, c2, w[15], 9);
    R31(c1, d1, e1, a1, b1, w[10], 13); R32(c2, d2, e2, a2, b2, w[5], 7);
    R31(b1, c1, d1, e1, a1, w[14], 6);  R32(b2, c2, d2, e2, a2, w[1], 15);
    R31(a1, b1, c1, d1, e1, w[4], 7);   R32(a2, b2, c2, d2, e2, w[3], 11);
    R31(e1, a1, b1, c1, d1, w[9], 14);  R32(e2, a2, b2, c2, d2, w[7], 8);
    R31(d1, e1, a1, b1, c1, w[15], 9);  R32(d2, e2, a2, b2, c2, w[14], 6);
    R31(c1, d1, e1, a1, b1, w[8], 13);  R32(c2, d2, e2, a2, b2, w[6], 6);
    R31(b1, c1, d1, e1, a1, w[1], 15);  R32(b2, c2, d2, e2, a2, w[9], 14);
    R31(a1, b1, c1, d1, e1, w[2], 14);  R32(a2, b2, c2, d2, e2, w[11], 12);
    R31(e1, a1, b1, c1, d1, w[7], 8);   R32(e2, a2, b2, c2, d2, w[8], 13);
    R31(d1, e1, a1, b1, c1, w[0], 13);  R32(d2, e2, a2, b2, c2, w[12], 5);
    R31(c1, d1, e1, a1, b1, w[6], 6);   R32(c2, d2, e2, a2, b2, w[2], 14);
    R31(b1, c1, d1, e1, a1, w[13], 5);  R32(b2, c2, d2, e2, a2, w[10], 13);
    R31(a1, b1, c1, d1, e1, w[11], 12); R32(a2, b2, c2, d2, e2, w[0], 13);
    R31(e1, a1, b1, c1, d1, w[5], 7);   R32(e2, a2, b2, c2, d2, w[4], 7);
    R31(d1, e1, a1, b1, c1, w[12], 5);  R32(d2, e2, a2, b2, c2, w[13], 5);

    R41(c1, d1, e1, a1, b1, w[1], 11);  R42(c2, d2, e2, a2, b2, w[8], 15);
    R41(b1, c1, d1, e1, a1, w[9], 12);  R42(b2, c2, d2, e2, a2, w[6], 5);
    R41(a1, b1, c1, d1, e1, w[11], 14); R42(a2, b2, c2, d2, e2, w[4], 8);
    R41(e1, a1, b1, c1, d1, w[10], 15); R42(e2, a2, b2, c2, d2, w[1], 11);
    R41(d1, e1, a1, b1, c1, w[0], 14);  R42(d2, e2, a2, b2, c2, w[3], 14);
    R41(c1, d1, e1, a1, b1, w[8], 15);  R42(c2, d2, e2, a2, b2, w[11], 14);
    R41(b1, c1, d1, e1, a1, w[12], 9);  R42(b2, c2, d2, e2, a2, w[15], 6);
    R41(a1, b1, c1, d1, e1, w[4], 8);   R42(a2, b2, c2, d2, e2, w[0], 14);
    R41(e1, a1, b1, c1, d1, w[13], 9);  R42(e2, a2, b2, c2, d2, w[5], 6);
    R41(d1, e1, a1, b1, c1, w[3], 14);  R42(d2, e2, a2, b2, c2, w[12], 9);
    R41(c1, d1, e1, a1, b1, w[7], 5);   R42(c2, d2, e2, a2, b2, w[2], 12);
    R41(b1, c1, d1, e1, a1, w[15], 6);  R42(b2, c2, d2, e2, a2, w[13], 9);
    R41(a1, b1, c1, d1, e1, w[14], 8);  R42(a2, b2, c2, d2, e2, w[9], 12);
    R41(e1, a1, b1, c1, d1, w[5], 6);   R42(e2, a2, b2, c2, d2, w[7], 5);
    R41(d1, e1, a1, b1, c1, w[6], 5);   R42(d2, e2, a2, b2, c2, w[10], 15);
    R41(c1, d1, e1, a1, b1, w[2], 12);  R42(c2, d2, e2, a2, b2, w[14], 8);

    R51(b1, c1, d1, e1, a1, w[4], 9);   R52(b2, c2, d2, e2, a2, w[12], 8);
    R51(a1, b1, c1, d1, e1, w[0], 15);  R52(a2, b2, c2, d2, e2, w[15], 5);
    R51(e1, a1, b1, c1, d1, w[5], 5);   R52(e2, a2, b2, c2, d2, w[10], 12);
    R51(d1, e1, a1, b1, c1, w[9], 11);  R52(d2, e2, a2, b2, c2, w[4], 9);
    R51(c1, d1, e1, a1, b1, w[7], 6);   R52(c2, d2, e2, a2, b2, w[1], 12);
    R51(b1, c1, d1, e1, a1, w[12], 8);  R52(b2, c2, d2, e2, a2, w[5], 5);
    R51(a1, b1, c1, d1, e1, w[2], 13);  R52(a2, b2, c2, d2, e2, w[8], 14);
    R51(e1, a1, b1, c1, d1, w[10], 12); R52(e2, a2, b2, c2, d2, w[7], 6);
    R51(d1, e1, a1, b1, c1, w[14], 5);  R52(d2, e2, a2, b2, c2, w[6], 8);
    R51(c1, d1, e1, a1, b1, w[1], 12);  R52(c2, d2, e2, a2, b2, w[2], 13);
    R51(b1, c1, d1, e1, a1, w[3], 13);  R52(b2, c2, d2, e2, a2, w[13], 6);
    R51(a1, b1, c1, d1, e1, w[8], 14);  R52(a2, b2, c2, d2, e2, w[14], 5);
    R51(e1, a1, b1, c1, d1, w[11], 11); R52(e2, a2, b2, c2, d2, w[0], 15);
    R51(d1, e1, a1, b1, c1, w[6], 8);   R52(d2, e2, a2, b2, c2, w[3], 13);
    R51(c1, d1, e1, a1, b1, w[15], 5);  R52(c2, d2, e2, a2, b2, w[9], 11);
    R51(b1, c1, d1, e1, a1, w[13], 6);  R52(b2, c2, d2, e2, a2, w[11], 11);

    uint t = s[0];
    s[0] = s[1] + c1 + d2;
    s[1] = s[2] + d1 + e2;
    s[2] = s[3] + e1 + a2;
    s[3] = s[4] + a1 + b2;
    s[4] = t    + b1 + c2;
}

/* ───────────────────────────────────────────────────────────────────────
 * Funções de hashing de alto nível
 * ─────────────────────────────────────────────────────────────────────── */

/* getSHA256_33bytes: computa SHA-256 dos primeiros 33 bytes de pubkey.
   Equivalente ao do HashPipeline.cpp. */
__forceinline__ void getSHA256_33bytes(const __private uchar *pubkey33,
                                        __private uchar sha[32]) {
    uint M[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) M[i] = 0;

    #pragma unroll
    for (int i = 0; i < 33; ++i) {
        M[i >> 2] |= (uint)pubkey33[i] << (24 - ((i & 3) << 3));
    }
    M[8]  |= (uint)0x80 << (24 - ((33 & 3) << 3));
    M[14] = 0;
    M[15] = 33u * 8u;

    uint state[8];
    SHA256Initialize(state);
    SHA256Transform(state, M);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        sha[4*i + 0] = (uchar)(state[i] >> 24);
        sha[4*i + 1] = (uchar)(state[i] >> 16);
        sha[4*i + 2] = (uchar)(state[i] >>  8);
        sha[4*i + 3] = (uchar)(state[i] >>  0);
    }
}

/* getRIPEMD160_32bytes: RIPEMD-160 sobre 32 bytes (output do SHA-256). */
__forceinline__ void getRIPEMD160_32bytes(const __private uchar *sha,
                                           __private uchar ripemd[20]) {
    uchar block[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) block[i] = 0;

    #pragma unroll
    for (int i = 0; i < 32; ++i) block[i] = sha[i];
    block[32] = 0x80;
    const uint bitLen = 256u;

    block[56] = (uchar)(bitLen & 0xFF);
    block[57] = (uchar)((bitLen >> 8)  & 0xFF);
    block[58] = (uchar)((bitLen >> 16) & 0xFF);
    block[59] = (uchar)((bitLen >> 24) & 0xFF);

    uint W[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint)block[4*i+3] << 24) |
               ((uint)block[4*i+2] << 16) |
               ((uint)block[4*i+1] <<  8) |
               ((uint)block[4*i+0] <<  0);
    }

    uint state[5];
    RIPEMD160Initialize(state);
    RIPEMD160Transform(state, W);

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        ripemd[4*i + 0] = (uchar)(state[i] >>  0);
        ripemd[4*i + 1] = (uchar)(state[i] >>  8);
        ripemd[4*i + 2] = (uchar)(state[i] >> 16);
        ripemd[4*i + 3] = (uchar)(state[i] >> 24);
    }
}

/* getHash160_33bytes: pipeline completo SHA-256 → RIPEMD-160. */
__forceinline__ void getHash160_33bytes(const __private uchar *pubkey33,
                                         __private uchar *hash20) {
    uchar sha256[32];
    getSHA256_33bytes(pubkey33, sha256);
    getRIPEMD160_32bytes(sha256, hash20);
}

/* ───────────────────────────────────────────────────────────────────────
 * Fast path: SHA-256 + RIPEMD-160 a partir de X em limbs BE (4×u64)
 *
 * Esta é a função hot do kernel. Recebe X como 4 limbs big-endian
 * (Rx[3] = MSB), monta o pubkey 33 bytes virtualmente (prefix + X) sem
 * nunca materializar o array, e computa Hash160 direto.
 *
 * Equivalente a SHA256_33_from_limbs + RIPEMD160_from_SHA256_state_words
 * do HashPipeline.cpp original.
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ void SHA256_33_from_limbs(uchar prefix02_03,
                                           const ulong x_be_limbs[4],
                                           uint out_state[8]) {
    const ulong v3 = x_be_limbs[3];
    const ulong v2 = x_be_limbs[2];
    const ulong v1 = x_be_limbs[1];
    const ulong v0 = x_be_limbs[0];

    uint M[16];
    M[0] = pack_be4(prefix02_03,
                    (uchar)(v3 >> 56), (uchar)(v3 >> 48), (uchar)(v3 >> 40));
    M[1] = pack_be4((uchar)(v3 >> 32), (uchar)(v3 >> 24),
                    (uchar)(v3 >> 16), (uchar)(v3 >> 8));
    M[2] = pack_be4((uchar)(v3 >> 0),
                    (uchar)(v2 >> 56), (uchar)(v2 >> 48), (uchar)(v2 >> 40));
    M[3] = pack_be4((uchar)(v2 >> 32), (uchar)(v2 >> 24),
                    (uchar)(v2 >> 16), (uchar)(v2 >> 8));
    M[4] = pack_be4((uchar)(v2 >> 0),
                    (uchar)(v1 >> 56), (uchar)(v1 >> 48), (uchar)(v1 >> 40));
    M[5] = pack_be4((uchar)(v1 >> 32), (uchar)(v1 >> 24),
                    (uchar)(v1 >> 16), (uchar)(v1 >> 8));
    M[6] = pack_be4((uchar)(v1 >> 0),
                    (uchar)(v0 >> 56), (uchar)(v0 >> 48), (uchar)(v0 >> 40));
    M[7] = pack_be4((uchar)(v0 >> 32), (uchar)(v0 >> 24),
                    (uchar)(v0 >> 16), (uchar)(v0 >> 8));
    M[8] = pack_be4((uchar)(v0 >> 0), 0x80u, 0x00u, 0x00u);
    #pragma unroll
    for (int i = 9; i < 16; ++i) M[i] = 0;
    M[15] = 33u * 8u;

    uint st[8];
    SHA256Initialize(st);
    SHA256Transform(st, M);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out_state[i] = st[i];
}

/* RIPEMD160_from_SHA256_state: assume que o SHA state são 8 palavras
   big-endian (como saem do SHA256Transform). Faz bswap para little
   e roda RIPEMD-160. Output: 20 bytes little-endian. */
__forceinline__ void RIPEMD160_from_SHA256_state(const uint sha_state_be[8],
                                                   __private uchar ripemd20[20]) {
    uint W[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) W[i] = bswap32(sha_state_be[i]);
    W[8] = 0x00000080u;
    #pragma unroll
    for (int i = 9; i < 14; ++i) W[i] = 0;
    W[14] = 256u;
    W[15] = 0u;

    uint s[5];
    RIPEMD160Initialize(s);
    RIPEMD160Transform(s, W);
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        ripemd20[4*i + 0] = (uchar)(s[i] >>  0);
        ripemd20[4*i + 1] = (uchar)(s[i] >>  8);
        ripemd20[4*i + 2] = (uchar)(s[i] >> 16);
        ripemd20[4*i + 3] = (uchar)(s[i] >> 24);
    }
}

/* RIPEMD160_from_SHA256_state_words: mesma coisa, mas retorna 5 uint. */
__forceinline__ void RIPEMD160_from_SHA256_state_words(const uint sha_state_be[8],
                                                         uint h160[5]) {
    uint W[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) W[i] = bswap32(sha_state_be[i]);
    W[8] = 0x00000080u;
    #pragma unroll
    for (int i = 9; i < 14; ++i) W[i] = 0;
    W[14] = 256u;
    W[15] = 0u;

    uint s[5];
    RIPEMD160Initialize(s);
    RIPEMD160Transform(s, W);
    #pragma unroll
    for (int i = 0; i < 5; ++i) h160[i] = s[i];
}

/* hash_load_u32_le: lê 4 bytes como uint little-endian. */
__forceinline__ uint hash_load_u32_le(const __private uchar *p) {
    return (uint)p[0] | ((uint)p[1] << 8) |
           ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

/* getHash160_33_from_limbs: pipeline completo a partir de limbs. */
__forceinline__ void getHash160_33_from_limbs(uchar prefix02_03,
                                               const ulong x_be_limbs[4],
                                               __private uchar out20[20]) {
    uint sha_state[8];
    SHA256_33_from_limbs(prefix02_03, x_be_limbs, sha_state);
    RIPEMD160_from_SHA256_state(sha_state, out20);
}

/* getHash160_33_from_limbs_matches: pipeline + comparação contra target.
   Retorna true se h160 == target_hash160 (todos os 20 bytes).
   O parâmetro h160_out é opcional (passar NULL se não quiser). */
__forceinline__ bool getHash160_33_from_limbs_matches(
    uchar prefix02_03,
    const ulong x_be_limbs[4],
    __global const uchar *target_hash160,
    uint target_prefix_le,
    __private uint *h160_out)
{
    uint sha_state[8];
    uint h160[5];
    SHA256_33_from_limbs(prefix02_03, x_be_limbs, sha_state);
    RIPEMD160_from_SHA256_state_words(sha_state, h160);

    if (h160_out) {
        h160_out[0] = h160[0]; h160_out[1] = h160[1];
        h160_out[2] = h160[2]; h160_out[3] = h160[3];
        h160_out[4] = h160[4];
    }

    /* Comparação prefix-first: se o primeiro uint (4 bytes LE) não
       bater, falha rápido sem ler o resto. */
    if (h160[0] != target_prefix_le) return false;

    /* Compara bytes 4..19 (16 bytes restantes).
       Importante: NÃO podemos castar __constant uchar* para __private uchar*
       em OpenCL C. Comparamos byte-a-byte lendo diretamente de target_hash160
       (__constant) e de h160 (__private via cast para uchar*). */
    __private const uchar *h = (__private const uchar *)(&h160[1]);
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (h[k] != target_hash160[k + 4]) return false;
    }
    return true;
}
