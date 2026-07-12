/*
 * ocl_helpers.h — Utilitários do host (aritmética 256-bit, hex, formatação)
 *
 * Porte da parte host de AMDUtils.h. Remove tudo que era __device__.
 * Mantém add256, sub256, divmod_256_by_u64, hexToLE64, formatHex256, etc.
 */

#ifndef OCL_HELPERS_H
#define OCL_HELPERS_H

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>

/* Inclui OpenCL C API headers — necessário para OclDeviceInfo conter
   cl_platform_id e cl_device_id. Em Windows/Linux com AMD APP SDK
   instalado, este header está no include path padrão. */
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#include <CL/opencl.h>

/* ── OclDeviceInfo: info de uma GPU OpenCL enumerada ────────────────── */
struct OclDeviceInfo {
    cl_platform_id platform;
    cl_device_id   device;
    std::string    name;
    cl_uint        compute_units;
    cl_ulong       global_mem;
    cl_uint        max_clock;
    cl_uint        max_work_group_size;
};

#if defined(_MSC_VER) && !defined(__clang__)
  #include <intrin.h>
#endif

/* ── 256-bit arithmetic (host-side) ─────────────────────────────────── */

inline void add256_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]) {
    uint64_t tmp[4];
    tmp[0] = a[0] + b;
    uint64_t carry = (tmp[0] < a[0]) ? 1ULL : 0ULL;
    for (int i = 1; i < 4; ++i) {
        tmp[i] = a[i] + carry;
        carry = (tmp[i] < a[i]) ? 1ULL : 0ULL;
    }
    memcpy(out, tmp, 4 * sizeof(uint64_t));
}

inline void add256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t tmp[4];
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t s = a[i] + b[i];
        uint64_t c = (s < a[i]) ? 1ULL : 0ULL;
        uint64_t s2 = s + carry;
        if (s2 < s) c = 1ULL;
        tmp[i] = s2;
        carry = c;
    }
    memcpy(out, tmp, 4 * sizeof(uint64_t));
}

inline void sub256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t tmp[4];
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t diff = a[i] - borrow;
        uint64_t nb = (diff > a[i]) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - b[i];
        if (diff2 > diff) nb = 1ULL;
        tmp[i] = diff2;
        borrow = nb;
    }
    memcpy(out, tmp, 4 * sizeof(uint64_t));
}

inline void inc256(uint64_t a[4], uint64_t inc) {
    a[0] += inc;
    uint64_t carry = (a[0] < inc) ? 1ULL : 0ULL;
    for (int i = 1; i < 4 && carry; ++i) {
        ++a[i];
        carry = (a[i] == 0ULL) ? 1ULL : 0ULL;
    }
}

inline void divmod_256_by_u64(const uint64_t value[4], uint64_t divisor,
                               uint64_t quotient[4], uint64_t &remainder) {
#if defined(_MSC_VER) && !defined(__clang__)
    remainder = 0;
    for (int i = 3; i >= 0; --i) {
        quotient[i] = _udiv128(remainder, value[i], divisor, &remainder);
    }
#else
    remainder = 0;
    for (int i = 3; i >= 0; --i) {
        __uint128_t cur = ((__uint128_t)remainder << 64) | value[i];
        quotient[i] = (uint64_t)(cur / divisor);
        remainder = (uint64_t)(cur % divisor);
    }
#endif
}

/* ── Hex/format utilities ───────────────────────────────────────────── */

inline bool hexToLE64(const std::string& h_in, uint64_t w[4]) {
    std::string h = h_in;
    if (h.size() >= 2 && (h[0] == '0') && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
    if (h.size() > 64) return false;
    if (h.size() < 64) h = std::string(64 - h.size(), '0') + h;
    if (h.size() != 64) return false;
    for (int i = 0; i < 4; ++i) {
        std::string part = h.substr(i * 16, 16);
        w[3 - i] = std::stoull(part, nullptr, 16);
    }
    return true;
}

inline bool hexToHash160(const std::string& h, uint8_t hash160[20]) {
    if (h.size() != 40) return false;
    for (int i = 0; i < 20; ++i) {
        std::string byteStr = h.substr(i * 2, 2);
        hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
    }
    return true;
}

inline std::string formatHex256(const uint64_t limbs[4]) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (int i = 3; i >= 0; --i) oss << std::setw(16) << limbs[i];
    return oss.str();
}

/* ── Host utilities ─────────────────────────────────────────────────── */

inline std::string human_bytes(double bytes) {
    static const char* u[]={"B","KB","MB","GB","TB","PB"};
    int k=0;
    while(bytes>=1024.0 && k<5){ bytes/=1024.0; ++k; }
    std::ostringstream o; o.setf(std::ios::fixed);
    o<<std::setprecision(bytes<10?2:1)<<bytes<<" "<<u[k];
    return o.str();
}

inline long double ld_from_u256(const uint64_t v[4]) {
    return std::ldexp((long double)v[3],192)
         + std::ldexp((long double)v[2],128)
         + std::ldexp((long double)v[1],64)
         + (long double)v[0];
}

inline std::string formatCompressedPubHex(const uint64_t Rx[4], const uint64_t Ry[4]) {
    uint8_t out[33];
    out[0] = (Ry[0] & 1ULL) ? 0x03 : 0x02;
    int off=1;
    for (int limb=3; limb>=0; --limb) {
        uint64_t v = Rx[limb];
        out[off+0]=(uint8_t)(v>>56); out[off+1]=(uint8_t)(v>>48);
        out[off+2]=(uint8_t)(v>>40); out[off+3]=(uint8_t)(v>>32);
        out[off+4]=(uint8_t)(v>>24); out[off+5]=(uint8_t)(v>>16);
        out[off+6]=(uint8_t)(v>> 8); out[off+7]=(uint8_t)(v>> 0);
        off+=8;
    }
    static const char* hexd="0123456789ABCDEF";
    std::string s; s.resize(66);
    for (int i=0;i<33;++i){ s[2*i]=hexd[(out[i]>>4)&0xF]; s[2*i+1]=hexd[out[i]&0xF]; }
    return s;
}

#endif // OCL_HELPERS_H
