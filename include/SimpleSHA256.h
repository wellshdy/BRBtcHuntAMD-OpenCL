#ifndef SIMPLE_SHA256_H
#define SIMPLE_SHA256_H

#include <cstdint>
#include <cstring>

namespace SimpleSHA256 {

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define Ch(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define Sigma0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define Sigma1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sigma0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define sigma1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline void sha256_transform(uint32_t state[8], const uint8_t data[64]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    uint32_t W[64];

    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)data[i * 4] << 24) |
               ((uint32_t)data[i * 4 + 1] << 16) |
               ((uint32_t)data[i * 4 + 2] << 8) |
               ((uint32_t)data[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }

    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
        uint32_t T2 = Sigma0(a) + Maj(a, b, c);
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

inline void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint8_t buf[128];
    size_t bitlen = len * 8;
    size_t rem = len % 64;
    size_t limit = (rem < 56) ? 64 : 128;

    size_t i = 0;
    for (; i + 64 <= len; i += 64) {
        sha256_transform(state, data + i);
    }

    std::memcpy(buf, data + i, rem);
    buf[rem] = 0x80;
    std::memset(buf + rem + 1, 0, limit - rem - 9);
    
    // Append bit length in big-endian
    buf[limit - 8] = (uint8_t)(bitlen >> 56);
    buf[limit - 7] = (uint8_t)(bitlen >> 48);
    buf[limit - 6] = (uint8_t)(bitlen >> 40);
    buf[limit - 5] = (uint8_t)(bitlen >> 32);
    buf[limit - 4] = (uint8_t)(bitlen >> 24);
    buf[limit - 3] = (uint8_t)(bitlen >> 16);
    buf[limit - 2] = (uint8_t)(bitlen >> 8);
    buf[limit - 1] = (uint8_t)(bitlen);

    sha256_transform(state, buf);
    if (limit == 128) {
        sha256_transform(state, buf + 64);
    }

    for (int idx = 0; idx < 8; ++idx) {
        out[idx * 4]     = (uint8_t)(state[idx] >> 24);
        out[idx * 4 + 1] = (uint8_t)(state[idx] >> 16);
        out[idx * 4 + 2] = (uint8_t)(state[idx] >> 8);
        out[idx * 4 + 3] = (uint8_t)(state[idx]);
    }
}

#undef ROTR
#undef Ch
#undef Maj
#undef Sigma0
#undef Sigma1
#undef sigma0
#undef sigma1

} // namespace SimpleSHA256

#endif // SIMPLE_SHA256_H
