/*
 * gpu_worker.cl — Kernel principal de busca + warp intrinsics (OpenCL C)
 *
 * Porte de GPUWorker.cpp (parte device) para OpenCL C 2.0.
 *
 * O kernel original kernel_point_add_and_check_oneinv implementa:
 *   1. Endereço atual P = (x1,y1), scalar S
 *   2. Loop de batches de tamanho B (cada batch = B chaves)
 *   3. Para cada batch:
 *      - Computa todos os pontos P+i*G (i=0..B-1) com uma única inversão
 *        (Fermat Batch Inversion via árvore de produtos)
 *      - Hash160 de cada ponto, compara com target
 *      - Em match: atomic CAS para tomar o lock de "found"
 *      - Após o batch: atualiza P = P + B*G via kernel point addition
 *
 * Mudanças do porte:
 *   - threadIdx/blockIdx → get_local_id/get_global_id
 *   - __shfl / __any → sub_group_shuffle / sub_group_any (com fallback)
 *   - __threadfence_system → atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE)
 *   - atomicCAS → atomic_cmpxchg
 *   - atomicExch → atomic_xchg
 *   - atomicAdd → atomic_fetch_add
 *   - c_Gx / c_Gy / c_Jx / c_Jy / c_target_*: passados como __global buffers
 *     (OpenCL não permite __constant de tamanho dinâmico definido em runtime)
 */

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

/* ───────────────────────────────────────────────────────────────────────
 * Warp / sub-group helpers
 *
 * cl_khr_subgroups expõe sub_group_shuffle, sub_group_shuffle_down,
 * sub_group_any. Em RDNA1 (wave32) o sub_group tem 32 lanes.
 *
 * Se a extensão não estiver disponível, definir USE_LOCAL_FALLBACK e
 * usar memória __local + barreira.
 * ─────────────────────────────────────────────────────────────────────── */
#ifdef USE_LOCAL_FALLBACK

/* Fallback: cada work-group compartilha um __local int via barreira.
   Mais lento, mas funciona em qualquer driver OpenCL 2.0. */

#define BTC_SHFL_SYNC(mask, value, src_lane) \
    btc_local_shuffle(value, src_lane)

#define BTC_SHFL_DOWN_SYNC(mask, value, delta) \
    btc_local_shuffle_down(value, delta)

#define BTC_ANY_SYNC(mask, predicate) \
    btc_local_any(predicate)

/* Estas funções exigem que o work-group tenha __local scratch buffer
   alocado pelo host via clSetKernelArg. Como a API é complexa e RDNA1
   tem suporte nativo a cl_khr_subgroups, o fallback abaixo é
   PLACEHOLDER — não use USE_LOCAL_FALLBACK em produção sem implementar
   a passagem de __local buffer via clSetKernelArg. */
__forceinline__ ulong btc_local_shuffle(ulong value, uint src_lane) {
    (void)src_lane;
    /* Sem implementação correta; retorna o próprio valor (INCORRETO). */
    return value;
}

__forceinline__ ulong btc_local_shuffle_down(ulong value, uint delta) {
    (void)delta;
    /* Sem implementação correta; retorna o próprio valor (INCORRETO). */
    return value;
}

__forceinline__ bool btc_local_any(bool predicate) {
    /* Em GPU, sem shuffle, usamos atomic em __local uint. */
    /* Implementação simplificada — para uso real, fornecer __local scratch. */
    return predicate;
}

#else /* USE_LOCAL_FALLBACK — usar sub_group intrinsics */

#define BTC_SHFL_SYNC(mask, value, src_lane) \
    sub_group_shuffle(value, src_lane)

#define BTC_SHFL_DOWN_SYNC(mask, value, delta) \
    sub_group_shuffle(value, get_sub_group_local_id() + (delta))
#endif /* USE_LOCAL_FALLBACK */

/* BTC_SYNCWARP: no-op em OpenCL */
#define BTC_SYNCWARP(mask) ((void)0)

/* ───────────────────────────────────────────────────────────────────────
 * Helpers de 256-bit no device
 *
 * Estes substituem as funções inline de AMDUtils.h (parte __device__).
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ bool ge256_u64(const ulong a[4], ulong b) {
    if (a[3] | a[2] | a[1]) return true;
    return a[0] >= b;
}

__forceinline__ void sub256_u64_inplace(ulong a[4], ulong dec) {
    ulong borrow = (a[0] < dec) ? 1UL : 0UL;
    a[0] = a[0] - dec;
    #pragma unroll
    for (int i = 1; i < 4; ++i) {
        ulong ai = a[i];
        ulong bi = borrow;
        a[i] = ai - bi;
        borrow = (ai < bi) ? 1UL : 0UL;
        if (!borrow) break;
    }
}

__forceinline__ bool eq256_u64(const ulong a[4], ulong b) {
    return (a[0]==b) & (a[1]==0UL) & (a[2]==0UL) & (a[3]==0UL);
}

__forceinline__ uint load_u32_le(const __private uchar *p) {
    return (uint)p[0] | ((uint)p[1] << 8)
         | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

/* ───────────────────────────────────────────────────────────────────────
 * load_found_flag_relaxed: leitura não-coalesced do flag de found.
 *
 * Em OpenCL, volatile em __global é o equivalente a threadfence_read.
 * Equivalente ao "const volatile int*" do HIP.
 * ─────────────────────────────────────────────────────────────────────── */
__forceinline__ int load_found_flag_relaxed(__global const volatile int *p) {
    return *p;
}

__forceinline__ bool warp_found_ready(__global const volatile int *d_found_flag,
                                        uint full_mask,
                                        uint lane) {
    (void)full_mask; (void)lane;
    return load_found_flag_relaxed(d_found_flag) == FOUND_READY;
}

/* ───────────────────────────────────────────────────────────────────────
 * vanity_check_and_save
 *
 * Compara primeiro N bytes (+ optional nibble) do h160 contra target.
 * Em match, grava no buffer de vanity via atomic.
 *
 * Equivalente à função inline em GPUWorker.cpp original.
 * ─────────────────────────────────────────────────────────────────────── */

__forceinline__ void vanity_check_and_save(
    const uint h160[5],
    uchar prefix,
    const ulong privkey[4],
    const ulong pubkey_x[4],
    __global uint *vanity_count,
    __global VanityResult *vanity_buf,
    uint max_results,
    __global const uchar *target_hash160,
    uint vanity_compare_bytes,
    uint vanity_match_nibble)
{
    __private const uchar *h = (__private const uchar *)h160;

    /* Compara os primeiros N bytes */
    for (uint b = 0; b < vanity_compare_bytes; ++b) {
        if (h[b] != target_hash160[b]) return;
    }
    /* Optional: compara high nibble do próximo byte */
    if (vanity_match_nibble) {
        uchar hn = (h[vanity_compare_bytes] >> 4) & 0x0f;
        uchar tn = (target_hash160[vanity_compare_bytes] >> 4) & 0x0f;
        if (hn != tn) return;
    }

    /* Match — aloca slot atômico (cast explícito para atomic_uint). */
    uint slot = atomic_fetch_add((__global atomic_uint *)vanity_count, 1u);
    if (slot < max_results && vanity_buf) {
        __global VanityResult *r = vanity_buf + slot;
        #pragma unroll
        for (int k = 0; k < 4; ++k) r->privkey[k] = privkey[k];
        #pragma unroll
        for (int k = 0; k < 4; ++k) r->pubkey_x[k] = pubkey_x[k];
        #pragma unroll
        for (int k = 0; k < 5; ++k) r->hash160[k] = h160[k];
        r->prefix = prefix;
    }
}

/* ───────────────────────────────────────────────────────────────────────
 * KERNEL PRINCIPAL: kernel_point_add_and_check_oneinv
 *
 * Substitui o __global__ kernel homônimo do GPUWorker.cpp.
 *
 * Args (mapeados via clSetKernelArg):
 *   Px, Py            : pontos de partida atuais (4 limbs por thread)
 *   Rx, Ry            : pontos finais (output)
 *   start_scalars     : scalar atual de cada thread
 *   counts256         : contador restante de batches por thread
 *   threadsTotal      : número total de threads (NxWorkItems)
 *   batch_size        : B (deve ser par e <= MAX_BATCH_SIZE)
 *   max_batches       : slices_per_launch
 *   d_found_flag      : flag global de "found" (atomic)
 *   d_found_result    : buffer de FoundResult
 *   hashes_accum      : acumulador de hashes (atomic_add em ulong)
 *   d_any_left        : flag "ainda há trabalho" (atomic)
 *   d_vanity_count    : contador de vanity hits
 *   d_vanity_buf      : buffer de VanityResult
 *   vanity_max_results: limite do buffer vanity
 *   c_Gx, c_Gy        : pontos pré-computados G*1..G*half (constant-like)
 *   c_Jx, c_Jy        : ponto J = G*B (jump base)
 *   c_target_hash160  : 20 bytes do hash160 alvo
 *   c_target_prefix   : primeiros 4 bytes do hash160 alvo como uint LE
 *   vanity_compare_bytes, vanity_match_nibble: config de vanity
 *
 * Importante: cada thread executa B/(2*1) iterações de batch inversion,
 * processando B chaves por iteração com UMA inversão modular.
 * ─────────────────────────────────────────────────────────────────────── */
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void kernel_point_add_and_check_oneinv(
    __global const ulong * __restrict__ Px,
    __global const ulong * __restrict__ Py,
    __global       ulong * __restrict__ Rx,
    __global       ulong * __restrict__ Ry,
    __global       ulong * __restrict__ start_scalars,
    __global       ulong * __restrict__ counts256,
    const  ulong           threadsTotal,
    const  uint            batch_size,
    const  uint            max_batches,
    __global       int    * __restrict__ d_found_flag,
    __global       FoundResult * __restrict__ d_found_result,
    __global       uint   * __restrict__ hashes_accum,
    __global       uint   * __restrict__ d_any_left,
    __global       uint   * __restrict__ d_vanity_count,
    __global       VanityResult * __restrict__ d_vanity_buf,
    const  uint            vanity_max_results,
    __global const ulong * __restrict__ c_Gx,
    __global const ulong * __restrict__ c_Gy,
    __global const ulong * __restrict__ c_Jx,
    __global const ulong * __restrict__ c_Jy,
    __global const uchar * __restrict__ c_target_hash160,
    const  uint            c_target_prefix,
    const  uint            c_vanity_compare_bytes,
    const  uint            c_vanity_match_nibble)
{
#ifdef BATCH_SIZE_CONST
    #define B BATCH_SIZE_CONST
    #define half_B HALF_B_CONST
#else
    const int B = (int)batch_size;
    const int half_B = B >> 1;
#endif

    const ulong gid = (ulong)get_global_id(0);
    if (gid >= threadsTotal) return;

    const uint lane      = (uint)(get_local_id(0) & (WARP_SIZE - 1));
    const uint full_mask = 0xFFFFFFFFu;
    if (warp_found_ready((__global const volatile int *)d_found_flag, full_mask, lane))
        return;

    const uint target_prefix = c_target_prefix;
    const bool _vanity_active = (c_vanity_compare_bytes > 0 || c_vanity_match_nibble > 0);

    /* Acumulador local de hashes (para evitar contention no atomic global) */
    uint local_hashes = 0;
    #define FLUSH_THRESHOLD 1024u
    #define WARP_FLUSH_HASHES() do { \
        if (local_hashes) { \
            atomic_fetch_add((__global atomic_uint *)hashes_accum, (uint)local_hashes); \
            local_hashes = 0; \
        } \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { \
        if (local_hashes >= FLUSH_THRESHOLD) WARP_FLUSH_HASHES(); \
    } while (0)
    /* Carrega estado da thread */
    ulong x1[4], y1[4], S[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const ulong idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        S[i]  = start_scalars[idx];
    }
    ulong rem[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

    /* Caso trivial: rem == 0 → nada a fazer */
    if ((rem[0] | rem[1] | rem[2] | rem[3]) == 0UL) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            Rx[gid*4+i] = x1[i];
            Ry[gid*4+i] = y1[i];
        }
        WARP_FLUSH_HASHES();
        return;
    }

    uint batches_done = 0;

    while (batches_done < max_batches && ge256_u64(rem, (ulong)B)) {
        if (warp_found_ready((__global const volatile int *)d_found_flag, full_mask, lane)) {
            WARP_FLUSH_HASHES(); return;
        }

        /* ── Passo 1: checa o ponto atual (P, prefix determinado por y) ── */
        {
            uchar prefix = (uchar)((y1[0] & 1UL) ? 0x03 : 0x02);
            uint _h160_i[5];
            bool matched = getHash160_33_from_limbs_matches(
                prefix, x1, c_target_hash160, target_prefix,
                _vanity_active ? _h160_i : (__private uint *)0);
            ++local_hashes; MAYBE_WARP_FLUSH();
            if (_vanity_active) {
                vanity_check_and_save(_h160_i, prefix, S, x1,
                                      d_vanity_count, d_vanity_buf, vanity_max_results,
                                      c_target_hash160, c_vanity_compare_bytes, c_vanity_match_nibble);
            }

            if (matched) {
                if (atomic_cmpxchg((volatile __global int *)d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                    d_found_result->threadId = (int)gid;
                    d_found_result->iter     = 0;
                    #pragma unroll
                    for (int k=0; k<4; ++k) d_found_result->scalar[k] = S[k];
                    #pragma unroll
                    for (int k=0; k<4; ++k) d_found_result->Rx[k] = x1[k];
                    #pragma unroll
                    for (int k=0; k<4; ++k) d_found_result->Ry[k] = y1[k];
                    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
                    atomic_xchg((volatile __global int *)d_found_flag, FOUND_READY);
                }
                BTC_SYNCWARP(full_mask);
                WARP_FLUSH_HASHES();
                return;
            }
        }

        /* ── Passo 2: batch inversion (Fermat tree) ──
           Monta subp[half-1..0] tal que subp[i] = prod_{k=i+1}^{half-1} (Gx[k] - x1)
           inverse = (Gx[0] - x1) * subp[0]   → inverte uma vez, depois
           propaga para todos via multiplicação. */
        ulong subp[MAX_BATCH_SIZE/2][4];
        ulong acc[4], tmp[4];

        #pragma unroll
        for (int j=0; j<4; ++j) acc[j] = c_Jx[j];
        mod_sub256_inplace(acc, x1);
        #pragma unroll
        for (int j=0; j<4; ++j) subp[half_B-1][j] = acc[j];

        #pragma unroll
        for (int i = half_B - 2; i >= 0; --i) {
            #pragma unroll
            for (int j=0; j<4; ++j) tmp[j] = c_Gx[(ulong)(i+1)*4 + j];
            mod_sub256_inplace(tmp, x1);
            mod_mult(acc, acc, tmp);
            #pragma unroll
            for (int j=0; j<4; ++j) subp[i][j] = acc[j];
        }

        ulong d0[4], inverse[5];
        #pragma unroll
        for (int j=0; j<4; ++j) d0[j] = c_Gx[0*4 + j];
        mod_sub256_inplace(d0, x1);
        #pragma unroll
        for (int j=0; j<4; ++j) inverse[j] = d0[j];
        mod_mult(inverse, inverse, subp[0]);
        inverse[4] = 0UL;
        mod_inv_fermat(inverse);

        ulong sy_neg[4], sx_neg[4];
        mod_neg256(y1, sy_neg);
        mod_neg256(x1, sx_neg);

        /* ── Passo 3: para cada i=0..half_B-2, computa P+i*G e P-i*G e checa ── */
        #pragma unroll
        for (int i = 0; i < half_B - 1; ++i) {
            if (warp_found_ready((__global const volatile int *)d_found_flag, full_mask, lane)) {
                WARP_FLUSH_HASHES(); return;
            }

            ulong dx_inv_i[4];
            mod_mult(dx_inv_i, subp[i], inverse);

            /* ── P + i*G (lambda positivo) ── */
            {
                ulong px3[4], s[4], lam[4];
                ulong px_i[4], py_i[4];
                #pragma unroll
                for (int j=0; j<4; ++j) {
                    px_i[j] = c_Gx[(ulong)i*4 + j];
                    py_i[j] = c_Gy[(ulong)i*4 + j];
                }

                mod_sub256(py_i, y1, s);
                mod_mult(lam, s, dx_inv_i);

                mod_sqr(px3, lam);
                mod_sub256_inplace(px3, x1);
                mod_sub256_inplace(px3, px_i);

                /* Deferred parity: try 0x02, then 0x03 */
                uint _h160_a[5];
                bool m02 = getHash160_33_from_limbs_matches(
                    0x02, px3, c_target_hash160, target_prefix,
                    _vanity_active ? _h160_a : (__private uint *)0);
                ++local_hashes; MAYBE_WARP_FLUSH();
                if (_vanity_active) {
                    vanity_check_and_save(_h160_a, 0x02, S, px3,
                                          d_vanity_count, d_vanity_buf, vanity_max_results,
                                          c_target_hash160, c_vanity_compare_bytes, c_vanity_match_nibble);
                }
                bool matched;
                if (!m02) {
                    uint _h160_b[5];
                    bool m03 = getHash160_33_from_limbs_matches(
                        0x03, px3, c_target_hash160, target_prefix,
                        _vanity_active ? _h160_b : (__private uint *)0);
                    ++local_hashes; MAYBE_WARP_FLUSH();
                    if (_vanity_active) {
                        vanity_check_and_save(_h160_b, 0x03, S, px3,
                                              d_vanity_count, d_vanity_buf, vanity_max_results,
                                              c_target_hash160, c_vanity_compare_bytes, c_vanity_match_nibble);
                    }
                    matched = m03;
                } else {
                    matched = true;
                }

                if (matched) {
                    if (atomic_cmpxchg((volatile __global int *)d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        ulong fs[4];
                        #pragma unroll
                        for (int k=0; k<4; ++k) fs[k] = S[k];
                        ulong sub = (ulong)(i+1);
                        for (int k=0; k<4 && sub; ++k) {
                            ulong old = fs[k]; fs[k] = old - sub;
                            sub = (old < sub) ? 1UL : 0UL;
                        }
                        #pragma unroll
                        for (int k=0; k<4; ++k) d_found_result->scalar[k] = fs[k];
                        #pragma unroll
                        for (int k=0; k<4; ++k) d_found_result->Rx[k] = px3[k];

                        ulong y3[4], t[4];
                        mod_sub256(x1, px3, t);
                        mod_mult(y3, t, lam);
                        mod_sub256_inplace(y3, y1);
                        #pragma unroll
                        for (int k=0; k<4; ++k) d_found_result->Ry[k] = y3[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
                        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
                        atomic_xchg((volatile __global int *)d_found_flag, FOUND_READY);
                    }
                    BTC_SYNCWARP(full_mask);
                    WARP_FLUSH_HASHES();
                    return;
                }
            }

            /* ── P - i*G (lambda negativo) ── */
            {
                ulong px3[4], s[4], lam[4];
                ulong px_i[4], py_i[4];
                #pragma unroll
                for (int j=0; j<4; ++j) {
                    px_i[j] = c_Gx[(ulong)i*4 + j];
                    py_i[j] = c_Gy[(ulong)i*4 + j];
                }
                mod_neg256_inplace(py_i);

                mod_sub256(py_i, y1, s);
                mod_mult(lam, s, dx_inv_i);

                mod_sqr(px3, lam);
                mod_sub256_inplace(px3, x1);
                mod_sub256_inplace(px3, px_i);

                uint _h160_a[5];
                bool m02 = getHash160_33_from_limbs_matches(
                    0x02, px3, c_target_hash160, target_prefix,
                    _vanity_active ? _h160_a : (__private uint *)0);
                ++local_hashes; MAYBE_WARP_FLUSH();
                if (_vanity_active) {
                    vanity_check_and_save(_h160_a, 0x02, S, px3,
                                          d_vanity_count, d_vanity_buf, vanity_max_results,
                                          c_target_hash160, c_vanity_compare_bytes, c_vanity_match_nibble);
                }
                bool matched;
                if (!m02) {
                    uint _h160_b[5];
                    bool m03 = getHash160_33_from_limbs_matches(
                        0x03, px3, c_target_hash160, target_prefix,
                        _vanity_active ? _h160_b : (__private uint *)0);
                    ++local_hashes; MAYBE_WARP_FLUSH();
                    if (_vanity_active) {
                        vanity_check_and_save(_h160_b, 0x03, S, px3,
                                              d_vanity_count, d_vanity_buf, vanity_max_results,
                                              c_target_hash160, c_vanity_compare_bytes, c_vanity_match_nibble);
                    }
                    matched = m03;
                } else {
                    matched = true;
                }

                if (matched) {
                    if (atomic_cmpxchg((volatile __global int *)d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        ulong fs[4];
                        #pragma unroll
                        for (int k=0; k<4; ++k) fs[k] = S[k];
                        ulong sub = (ulong)(i+1);
                        for (int k=0; k<4 && sub; ++k) {
                            ulong old = fs[k]; fs[k] = old - sub;
                            sub = (old < sub) ? 1UL : 0UL;
                        }
                        #pragma unroll
                        for (int k=0; k<4; ++k) d_found_result->scalar[k] = fs[k];
                        #pragma unroll
                        for (int k=0; k<4; ++k) d_found_result->Rx[k] = px3[k];

                        ulong y3[4], t[4];
                        mod_sub256(x1, px3, t);
                        mod_mult(y3, t, lam);
                        mod_sub256_inplace(y3, y1);
                        #pragma unroll
                        for (int k=0; k<4; ++k) d_found_result->Ry[k] = y3[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
                        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
                        atomic_xchg((volatile __global int *)d_found_flag, FOUND_READY);
                    }
                    BTC_SYNCWARP(full_mask);
                    WARP_FLUSH_HASHES();
                    return;
                }
            }

            /* Atualiza inverse = inverse * (Gx[i] - x1) */
            {
                ulong gxmi[4];
                #pragma unroll
                for (int j=0; j<4; ++j) gxmi[j] = c_Gx[(ulong)i*4 + j];
                mod_sub256_inplace(gxmi, x1);
                mod_mult(inverse, inverse, gxmi);
            }
        }

        /* ── Passo 4: último i = half_B-1 (apenas P - (half_B-1)*G, sem P+) ── */
        {
            const int i = half_B - 1;
            ulong dx_inv_i[4];
            mod_mult(dx_inv_i, subp[i], inverse);

            ulong px3[4], s[4], lam[4];
            ulong px_i[4], py_i[4];
            #pragma unroll
            for (int j=0; j<4; ++j) {
                px_i[j] = c_Gx[(ulong)i*4 + j];
                py_i[j] = c_Gy[(ulong)i*4 + j];
            }
            mod_neg256_inplace(py_i);

            mod_sub256(py_i, y1, s);
            mod_mult(lam, s, dx_inv_i);

            mod_sqr(px3, lam);
            mod_sub256_inplace(px3, x1);
            mod_sub256_inplace(px3, px_i);

            uint _h160_a[5];
            bool m02 = getHash160_33_from_limbs_matches(
                0x02, px3, c_target_hash160, target_prefix,
                _vanity_active ? _h160_a : (__private uint *)0);
            ++local_hashes; MAYBE_WARP_FLUSH();
            if (_vanity_active) {
                vanity_check_and_save(_h160_a, 0x02, S, px3,
                                      d_vanity_count, d_vanity_buf, vanity_max_results,
                                      c_target_hash160, c_vanity_compare_bytes, c_vanity_match_nibble);
            }
            bool matched;
            if (!m02) {
                uint _h160_b[5];
                bool m03 = getHash160_33_from_limbs_matches(
                    0x03, px3, c_target_hash160, target_prefix,
                    _vanity_active ? _h160_b : (__private uint *)0);
                ++local_hashes; MAYBE_WARP_FLUSH();
                if (_vanity_active) {
                    vanity_check_and_save(_h160_b, 0x03, S, px3,
                                          d_vanity_count, d_vanity_buf, vanity_max_results,
                                          c_target_hash160, c_vanity_compare_bytes, c_vanity_match_nibble);
                }
                matched = m03;
            } else {
                matched = true;
            }

            if (matched) {
                if (atomic_cmpxchg((volatile __global int *)d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                    ulong fs[4];
                    #pragma unroll
                    for (int k=0; k<4; ++k) fs[k] = S[k];
                    ulong sub = (ulong)half_B;
                    for (int k=0; k<4 && sub; ++k) {
                        ulong old = fs[k]; fs[k] = old - sub;
                        sub = (old < sub) ? 1UL : 0UL;
                    }
                    #pragma unroll
                    for (int k=0; k<4; ++k) d_found_result->scalar[k] = fs[k];
                    #pragma unroll
                    for (int k=0; k<4; ++k) d_found_result->Rx[k] = px3[k];

                    ulong y3[4], t[4];
                    mod_sub256(x1, px3, t);
                    mod_mult(y3, t, lam);
                    mod_sub256_inplace(y3, y1);
                    #pragma unroll
                    for (int k=0; k<4; ++k) d_found_result->Ry[k] = y3[k];
                    d_found_result->threadId = (int)gid;
                    d_found_result->iter     = 0;
                    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
                    atomic_xchg((volatile __global int *)d_found_flag, FOUND_READY);
                }
                BTC_SYNCWARP(full_mask);
                WARP_FLUSH_HASHES();
                return;
            }

            /* Atualiza inverse = inverse * (Gx[i] - x1) */
            {
                ulong last_dx[4];
                #pragma unroll
                for (int j=0; j<4; ++j) last_dx[j] = c_Gx[(ulong)i*4 + j];
                mod_sub256_inplace(last_dx, x1);
                mod_mult(inverse, inverse, last_dx);
            }
        }

        /* ── Passo 5: jump P = P + B*G (usa a inverse já computada) ── */
        {
            ulong lam[4], s[4], x3[4], y3[4];

            ulong Jy_minus_y1[4];
            #pragma unroll
            for (int j=0; j<4; ++j) Jy_minus_y1[j] = c_Jy[j];
            mod_sub256_inplace(Jy_minus_y1, y1);

            mod_mult(lam, Jy_minus_y1, inverse);
            mod_sqr(x3, lam);
            mod_sub256_inplace(x3, x1);

            ulong Jx_local[4];
            #pragma unroll
            for (int j=0; j<4; ++j) Jx_local[j] = c_Jx[j];
            mod_sub256_inplace(x3, Jx_local);

            mod_sub256(x1, x3, s);
            mod_mult(y3, s, lam);
            mod_sub256_inplace(y3, y1);

            #pragma unroll
            for (int j=0; j<4; ++j) { x1[j] = x3[j]; y1[j] = y3[j]; }
        }

        /* ── Passo 6: atualiza scalar S e contador rem ── */
        {
            ulong addv = (ulong)B;
            for (int k=0; k<4 && addv; ++k) {
                ulong old = S[k]; S[k] = old + addv;
                addv = (S[k] < old) ? 1UL : 0UL;
            }
            sub256_u64_inplace(rem, (ulong)B);
        }
        ++batches_done;
    }

    /* Persiste estado de volta para próxima launch */
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0UL) {
        atomic_fetch_add((__global atomic_uint *)d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
#ifdef BATCH_SIZE_CONST
    #undef B
    #undef half_B
#endif
}

__kernel void test_hash_kernel(__global uint *out_h160) {
    ulong x[4];
    x[0] = 0xd7efe2315fbc7671UL;
    x[1] = 0x743f1bc852858e32UL;
    x[2] = 0xd20291ce1798f490UL;
    x[3] = 0x8e3d1248c7657211UL;
    uint h160[5];
    uint sha[8];
    SHA256_33_from_limbs(0x02, x, sha);
    RIPEMD160_from_SHA256_state_words(sha, h160);
    for (int i = 0; i < 5; ++i) out_h160[i] = h160[i];
}
