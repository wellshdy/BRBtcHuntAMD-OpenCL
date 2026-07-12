/*
 * main_opencl.cpp — Entry point do porte OpenCL
 *
 * Porte de main.cpp. Substitui hipGetDeviceCount por clGetPlatformIDs +
 * clGetDeviceIDs. Resto (CLI, range split, progresso) é praticamente idêntico.
 */

#include "ocl_common.h"
#include "ocl_helpers.h"
#include "gpu_worker_host.h"
#include "Lang.h"

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.h>

#include "SimpleSHA256.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <vector>
#include <array>
#include <fstream>

/* ocl_utils.cpp expõe esta função.
   OclDeviceInfo já é definido em ocl_helpers.h (incluído acima). */
extern std::vector<OclDeviceInfo> enumerate_opencl_gpus();

/* ── Base58 decode (idêntico ao original) ───────────────────────────── */
static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static const int8_t BASE58_MAP[128] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,-1,-1,-1,-1,-1,-1,
    -1, 9,10,11,12,13,14,15,16,-1,17,18,19,20,21,-1,
    22,23,24,25,26,27,28,29,30,31,32,-1,-1,-1,-1,-1,
    -1,33,34,35,36,37,38,39,40,41,42,43,-1,44,45,46,
    47,48,49,50,51,52,53,54,55,56,57,-1,-1,-1,-1,-1
};

static bool decode_base58(const char* b58, uint8_t* out_25bytes) {
    size_t len = strlen(b58);
    if (len < 5 || len > 40) return false;
    uint8_t digits[128] = {0};
    size_t ndigits = 0;
    for (size_t i = 0; i < len; ++i) {
        int val = (b58[i] & 0x7f) < 128 ? BASE58_MAP[b58[i] & 0x7f] : -1;
        if (val < 0) return false;
        uint32_t carry = (uint32_t)val;
        for (size_t j = 0; j < ndigits; ++j) {
            carry += (uint32_t)digits[j] * 58;
            digits[j] = (uint8_t)(carry & 0xFF);
            carry >>= 8;
        }
        while (carry) {
            digits[ndigits++] = (uint8_t)(carry & 0xFF);
            carry >>= 8;
        }
    }
    size_t leading = 0;
    while (leading < len && b58[leading] == '1') ++leading;
    size_t total = leading + ndigits;
    if (total != 25) return false;
    size_t pos = 0;
    for (size_t z = 0; z < leading; ++z) out_25bytes[pos++] = 0;
    for (size_t i = ndigits; i > 0; --i) out_25bytes[pos++] = digits[i - 1];
    return true;
}

static bool decode_p2pkh_address(const char* b58, uint8_t hash20_out[20]) {
    uint8_t raw[25];
    if (!decode_base58(b58, raw)) return false;
    if (raw[0] != 0x00) return false;
    uint8_t hash1[32], hash2[32];
    SimpleSHA256::sha256(raw, 21, hash1);
    SimpleSHA256::sha256(hash1, 32, hash2);
    if (memcmp(hash2, raw + 21, 4) != 0) return false;
    memcpy(hash20_out, raw + 1, 20);
    return true;
}

/* ── Language globals ──────────────────────────────────────────────── */
LangId g_lang = LangId::EN;

static void load_lang_config() {
    std::ifstream f("lang.cfg");
    if (!f.is_open()) return;
    std::string s; f >> s;
    if (s == "pt" || s == "PT") g_lang = LangId::PT;
}

static void save_lang_config() {
    std::ofstream f("lang.cfg");
    if (!f.is_open()) return;
    f << (g_lang == LangId::PT ? "pt" : "en") << "\n";
}

/* ── Signal handling ───────────────────────────────────────────────── */
volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

/* ── Entry point ───────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string target_hash_hex, range_hex, address_b58, vanity_hex;
    uint32_t runtime_points_batch_size = 32;
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t slices_per_launch         = 64;
    bool     random_mode               = false;
    std::string kernel_dir             = "kernels";

    auto parse_grid = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        if (a_str.empty() || b_str.empty()) return false;
        char* endp=nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10); if (*endp) return false;
        endp=nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10); if (*endp) return false;
        if (aa == 0ul || bb == 0ul) return false;
        if (aa > (1ul<<20) || bb > (1ul<<20)) return false;
        a_out=(uint32_t)aa; b_out=(uint32_t)bb; return true;
    };

    load_lang_config();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--lang" && i + 1 < argc) {
            std::string l = argv[++i];
            if (l == "pt" || l == "PT") g_lang = LangId::PT;
        }
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "BRBtcHuntAMD-OpenCL — GPU Satoshi Puzzle Solver (OpenCL C 2.0)\n"
                      << "\n"
                      << "Usage: " << argv[0]
                      << " --range <start_hex>:<end_hex> --address <base58>\n"
                      << "       [--grid A,B] [--slices N] [--gpus all|0|0,1] [--random]\n"
                      << "       [--kernel-dir PATH]\n"
                      << "\n"
                      << "Required:\n"
                      << "  --range <start:end>        Search range in hex (e.g. 2000000000:3FFFFFFFFF)\n"
                      << "  --address <base58>         P2PKH address to search for\n"
                      << "  --target-hash160 <hex>     Alternative to --address (raw hash160)\n"
                      << "\n"
                      << "Options:\n"
                      << "  --grid <P,T>               Points per batch, threads per block (default 32,256)\n"
                      << "  --slices <N>               Batches per thread per kernel launch\n"
                      << "  --gpus <all|0|0,1>         Select which GPUs to use (default: all)\n"
                      << "  --random                   Lottery mode: random jumps across the range\n"
                      << "  --vanity <N>               Save keys whose first N hex chars match target\n"
                      << "  --kernel-dir <path>        Path to .cl files (default: kernels)\n"
                      << "  --lang pt|en               Language (default: en)\n"
                      << "  -h, --help                 Show this help\n"
                      << "\n"
                      << "Examples:\n"
                      << "  BRBtcHuntAMD.exe --range 200000000:3FFFFFFFF --address 1HBtAp... --grid 128,128\n"
                      << "  BRBtcHuntAMD.exe --range 200000000:3FFFFFFFF --gpus 0,1 --random --slices 16\n";
            return EXIT_SUCCESS;
        }
        if      (arg == "--target-hash160" && i + 1 < argc) target_hash_hex = argv[++i];
        else if (arg == "--address"        && i + 1 < argc) address_b58     = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << ERR_GRID_FMT() << "\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm    = b;
        }
        else if (arg == "--slices" && i + 1 < argc) {
            char* endp=nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0ul || v > (1ul<<20)) {
                std::cerr << ERR_SLICES() << " 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = (uint32_t)v;
        }
        else if (arg == "--gpus" && i + 1 < argc) ++i;
        else if (arg == "--random") random_mode = true;
        else if (arg == "--kernel-dir" && i + 1 < argc) kernel_dir = argv[++i];
        else if (arg == "--vanity" && i + 1 < argc) {
            vanity_hex = argv[++i];
            char* end = nullptr;
            long n = std::strtol(vanity_hex.c_str(), &end, 10);
            if (end == vanity_hex.c_str() || *end != '\0' || n < 1 || n > 40) {
                std::cerr << ERR_VANITY() << "\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "--lang" && i + 1 < argc) {
            std::string l = argv[++i];
            if (l == "pt" || l == "PT") g_lang = LangId::PT;
            else if (l == "en" || l == "EN") g_lang = LangId::EN;
            else { std::cerr << "Error: --lang pt|en\n"; return EXIT_FAILURE; }
            save_lang_config();
        }
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>) [--grid A,B] [--slices N] [--gpus all|0|0,1] [--random]\n";
        return EXIT_FAILURE;
    }
    if (!target_hash_hex.empty() && !address_b58.empty()) {
        std::cerr << ERR_BOTH_TARGET() << "\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) {
        std::cerr << ERR_RANGE_FORMAT() << "\n";
        return EXIT_FAILURE;
    }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << ERR_INVALID_RANGE() << "\n";
        return EXIT_FAILURE;
    }

    uint8_t target_hash160[20];
    if (!address_b58.empty()) {
        if (!decode_p2pkh_address(address_b58.c_str(), target_hash160)) {
            std::cerr << "Error: invalid P2PKH address\n";
            return EXIT_FAILURE;
        }
    } else {
        if (!hexToHash160(target_hash_hex, target_hash160)) {
            std::cerr << "Error: invalid target hash160 hex\n";
            return EXIT_FAILURE;
        }
    }

    if (runtime_points_batch_size < 2 || (runtime_points_batch_size & 1u)) {
        std::cerr << ERR_BATCH_EVEN() << "\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << ERR_BATCH_LIMIT() << "\n";
        return EXIT_FAILURE;
    }

    /* ── Enumerate OpenCL GPUs ──────────────────────────────────────── */
    auto gpus = enumerate_opencl_gpus();
    int num_gpus_avail = (int)gpus.size();
    if (num_gpus_avail == 0) {
        std::cerr << ERR_NO_GPU() << "\n";
        return EXIT_FAILURE;
    }

    /* Parse --gpus */
    std::vector<int> selected_gpus;
    {
        std::string gpus_arg = "all";
        for (int _i = 1; _i < argc; ++_i) {
            if (std::string(argv[_i]) == "--gpus" && _i + 1 < argc) {
                gpus_arg = argv[++_i];
                break;
            }
        }
        if (gpus_arg == "all") {
            for (int g = 0; g < num_gpus_avail; ++g) selected_gpus.push_back(g);
        } else {
            std::stringstream ss(gpus_arg);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                char* endp = nullptr;
                unsigned long idx = std::strtoul(tok.c_str(), &endp, 10);
                if (*endp != '\0' || idx >= (unsigned long)num_gpus_avail) {
                    std::cerr << ERR_GPU_INDEX() << " " << tok
                              << "'. Available GPUs: 0.." << (num_gpus_avail - 1) << "\n";
                    return EXIT_FAILURE;
                }
                selected_gpus.push_back((int)idx);
            }
            if (selected_gpus.empty()) {
                std::cerr << ERR_GPU_FORMAT() << "\n";
                return EXIT_FAILURE;
            }
        }
    }
    int num_gpus = (int)selected_gpus.size();

    /* Range split (idêntico ao original) */
    uint64_t range_len[4];
    sub256(range_end, range_start, range_len);
    add256_u64(range_len, 1ULL, range_len);

    std::vector<std::array<uint64_t,4>> gpu_starts(num_gpus), gpu_ends(num_gpus);
    if (random_mode) {
        for (int gi = 0; gi < num_gpus; ++gi) {
            gpu_starts[gi] = { range_start[0], range_start[1], range_start[2], range_start[3] };
            gpu_ends[gi]   = { range_end[0],   range_end[1],   range_end[2],   range_end[3]   };
        }
    } else {
        uint64_t per_gpu_len[4]; uint64_t r_gpu = 0ULL;
        divmod_256_by_u64(range_len, (uint64_t)num_gpus, per_gpu_len, r_gpu);
        uint64_t cur[4] = { range_start[0], range_start[1], range_start[2], range_start[3] };
        for (int gi = 0; gi < num_gpus; ++gi) {
            gpu_starts[gi] = { cur[0], cur[1], cur[2], cur[3] };
            if (gi == num_gpus - 1) {
                gpu_ends[gi] = { range_end[0], range_end[1], range_end[2], range_end[3] };
            } else {
                uint64_t next[4]; add256(cur, per_gpu_len, next);
                uint64_t one[4] = {1,0,0,0};
                uint64_t end[4]; sub256(next, one, end);
                gpu_ends[gi] = { end[0], end[1], end[2], end[3] };
                cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
            }
        }
    }

    /* Print GPU info */
    std::cout << HDR_PREPHASE() << " (" << num_gpus
              << " GPU" << (num_gpus > 1 ? "s" : "") << ") ===\n";
    for (int gi = 0; gi < num_gpus; ++gi) {
        int dev = selected_gpus[gi];
        std::cout << "  GPU " << dev << " : " << gpus[dev].name
                  << "  |  " << gpus[dev].compute_units << " CUs"
                  << "  |  " << human_bytes((double)gpus[dev].global_mem) << "\n";
    }
    std::cout << "======================================================= \n\n";
    std::cout.flush();

    GpuShared shared;
    std::atomic<int> gpus_running{num_gpus};

    /* Vanity setup */
    if (!vanity_hex.empty()) {
        shared.vanity_nibbles = (uint32_t)std::strtol(vanity_hex.c_str(), nullptr, 10);
        uint32_t vbytes = shared.vanity_nibbles / 2;
        if (shared.vanity_nibbles % 2) ++vbytes;
        std::cout << VANITY_MATCHING() << " " << shared.vanity_nibbles
                  << " " << VANITY_HEX_CHARS() << " (" << vbytes << " bytes)\n";
    }

    /* Launch worker threads */
    std::vector<std::thread> workers;
    uint8_t target_copy[20];
    memcpy(target_copy, target_hash160, 20);
    for (int gi = 0; gi < num_gpus; ++gi) {
        int dev = selected_gpus[gi];
        workers.emplace_back([dev, gi, &gpu_starts, &gpu_ends, target_copy,
                              runtime_points_batch_size, runtime_batches_per_sm,
                              slices_per_launch, random_mode, &kernel_dir, &shared, &gpus_running]()
        {
            run_on_gpu_opencl(gi, dev,
                               gpu_starts[gi].data(), gpu_ends[gi].data(),
                               target_copy,
                               runtime_points_batch_size, runtime_batches_per_sm,
                               slices_per_launch,
                               random_mode,
                               kernel_dir,
                               shared);
            gpus_running.fetch_sub(1, std::memory_order_relaxed);
        });
    }

    std::cout << "\n" << HDR_PHASE1() << ": "
              << (random_mode ? (g_lang == LangId::PT ? "Loteria" : "Lottery / Random Jump")
                              : "BruteForce")
              << " (" << num_gpus << " GPU" << (num_gpus > 1 ? "s" : "") << ") =====\n";
    if (random_mode) {
        uint64_t ck = (uint64_t)runtime_points_batch_size * slices_per_launch;
        std::string ck_s;
        if      (ck >= 1000000000ULL) ck_s = std::to_string(ck/1000000000ULL) + "G";
        else if (ck >= 1000000ULL)    ck_s = std::to_string(ck/1000000ULL)    + "M";
        else if (ck >= 1000ULL)       ck_s = std::to_string(ck/1000ULL)       + "K";
        else                          ck_s = std::to_string(ck);
        std::cout << "(random mode: ~" << ck_s << " keys/thread per chunk)\n";
    }
    std::cout.flush();

    /* Progress loop (idêntico ao original) */
    auto t0 = std::chrono::high_resolution_clock::now();
    double total_elapsed = 0.0;
    auto tLast = t0;
    unsigned long long lastHashes = 0ULL;
    long double total_keys_ld = ld_from_u256(range_len);

    while (gpus_running.load(std::memory_order_relaxed) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - tLast).count();
        if (dt >= 1.0) {
            unsigned long long h_hashes = shared.total_hashes.load(std::memory_order_relaxed);
            double delta = (double)(h_hashes - lastHashes);
            double mkeys = delta / (dt * 1e6);
            total_elapsed = std::chrono::duration<double>(now - t0).count();
            long double range_total = (shared.total_keys_adjusted > 0.0L)
                                       ? shared.total_keys_adjusted : total_keys_ld;
            long double prog = range_total > 0.0L
                                ? ((long double)h_hashes / range_total) * 100.0L : 0.0L;
            if (prog > 100.0L) prog = 100.0L;

            double speed_val = mkeys;
            const char* speed_unit = "Mkeys/s";
            if (speed_val >= 1000000.0)  { speed_val /= 1000000.0; speed_unit = "Tkeys/s"; }
            else if (speed_val >= 1000.0) { speed_val /= 1000.0;  speed_unit = "Gkeys/s"; }

            if (!shared.setup_done.load(std::memory_order_relaxed)) {
                std::cout << "\rInitializing EC points... "
                          << std::fixed << std::setprecision(1) << total_elapsed << " s     ";
            } else if (random_mode) {
                unsigned long long chunks = shared.chunks_tried.load(std::memory_order_relaxed);
                uint64_t s_lo = shared.cur_scalar_lo.load(std::memory_order_relaxed);
                uint64_t s_hi = shared.cur_scalar_hi.load(std::memory_order_relaxed);
                std::cout << "\r" << LBL_TIME() << ": "
                          << std::fixed << std::setprecision(1) << std::setw(6) << total_elapsed
                          << " s | " << LBL_SPEED() << ": "
                          << std::fixed << std::setprecision(2) << std::setw(7) << speed_val
                          << " " << speed_unit << " | " << LBL_COUNT() << ": "
                          << std::setw(14) << h_hashes
                          << " | " << LBL_KEY() << ": " << std::hex;
                if (s_hi) { std::cout << s_hi << s_lo; }
                else      { std::cout << s_lo; }
                std::cout << std::dec << " | " << LBL_CHUNKS() << ": " << std::setw(6) << chunks;
                if (shared.vanity_nibbles)
                    std::cout << " | " << LBL_VANITY() << ": "
                              << shared.vanity_total.load(std::memory_order_relaxed);
                std::cout << "   ";
            } else {
                std::cout << "\r" << LBL_TIME() << ": "
                          << std::fixed << std::setprecision(1) << std::setw(6) << total_elapsed
                          << " s | " << LBL_SPEED() << ": "
                          << std::fixed << std::setprecision(2) << std::setw(7) << speed_val
                          << " " << speed_unit << " | " << LBL_COUNT() << ": "
                          << std::setw(14) << h_hashes
                          << " | " << LBL_PROGRESS() << ": "
                          << std::fixed << std::setprecision(2) << std::setw(6) << (double)prog << " %";
                if (shared.vanity_nibbles)
                    std::cout << " | " << LBL_VANITY() << ": "
                              << shared.vanity_total.load(std::memory_order_relaxed);
                std::cout << "   ";
            }
            std::cout.flush();
            lastHashes = h_hashes; tLast = now;
        }

        if (g_sigint) break;
    }

    for (auto& t : workers) { if (t.joinable()) t.join(); }

    /* Save vanity results */
    if (!shared.vanity_results.empty()) {
        auto hex_limbs = [](const uint64_t limbs[4]) -> std::string {
            char buf[65];
            for (int i = 3; i >= 0; --i)
                sprintf(buf + (3-i)*16, "%016llx",
                        static_cast<unsigned long long>(limbs[i]));
            buf[64] = '\0';
            return std::string(buf);
        };
        auto fmt_compressed = [&](const VanityResult& vr) -> std::string {
            std::string s;
            s += (vr.prefix == 0x03 ? "03" : "02");
            s += hex_limbs(vr.pubkey_x);
            return s;
        };
        auto h160_str = [](const uint32_t h160[5]) -> std::string {
            char buf[41];
            const uint8_t* b = (const uint8_t*)h160;
            for (int i = 0; i < 20; ++i)
                sprintf(buf + i*2, "%02x", b[i]);
            buf[40] = '\0';
            return std::string(buf);
        };

        std::string fname = VANITY_RESULTS_FILE();
        std::ofstream ofs(fname, std::ios::app);
        for (const auto& vr : shared.vanity_results) {
            std::string priv_s = formatHex256(vr.privkey);
            std::string pub_s  = fmt_compressed(vr);
            std::string h160_s = h160_str(vr.hash160);
            std::string line = priv_s + " " + pub_s + " " + h160_s + "\n";
            if (ofs.is_open()) ofs << line;
        }
        if (ofs.is_open()) {
            ofs.close();
            std::cout << VANITY_SAVED() << " " << shared.vanity_results.size()
                      << " vanity results to " << VANITY_RESULTS_FILE() << "\n";
        }
    }

    std::cout << "\n";

    int exit_code = EXIT_SUCCESS;

    auto print_summary = [&]() {
        unsigned long long h_total = shared.total_hashes.load(std::memory_order_relaxed);
        if (h_total == 0) return;
        double avg_speed = total_elapsed > 0.0 ? (double)h_total / total_elapsed / 1e6 : 0.0;
        const char* avg_unit = "Mkeys/s";
        double avg_disp = avg_speed;
        if (avg_disp >= 1000000.0)  { avg_disp /= 1000000.0; avg_unit = "Tkeys/s"; }
        else if (avg_disp >= 1000.0) { avg_disp /= 1000.0; avg_unit = "Gkeys/s"; }
        std::cout << "\n" << SUM_HEADER() << "\n";
        std::cout << SUM_KEYS() << "  : " << h_total << "\n";
        std::cout << SUM_TIME() << "        : "
                  << std::fixed << std::setprecision(2) << total_elapsed << " s\n";
        std::cout << SUM_AVG_SPEED() << "   : "
                  << std::fixed << std::setprecision(2) << avg_disp << " " << avg_unit << "\n";
        uint32_t nv = shared.vanity_total.load(std::memory_order_relaxed);
        if (nv > 0)
            std::cout << SUM_VANITY() << "   : " << nv
                      << " (" << VANITY_SAVED() << " " << VANITY_RESULTS_FILE() << ")\n";
    };

    if (shared.has_result) {
        std::cout << "\n" << RSLT_FOUND() << "\n";
        std::cout << RSLT_PRIVKEY() << "   : "
                  << formatHex256(shared.best_result.scalar) << "\n";
        std::cout << RSLT_PUBKEY() << "    : "
                  << formatCompressedPubHex(shared.best_result.Rx,
                                             shared.best_result.Ry) << "\n";
        
        // Escreve automaticamente em found.txt no diretório atual
        std::ofstream found_file("found.txt", std::ios::app);
        if (found_file.is_open()) {
            found_file << "=== KEY FOUND ===\n";
            found_file << "Private Key (HEX): " << formatHex256(shared.best_result.scalar) << "\n";
            found_file << "Public Key (HEX):  " << formatCompressedPubHex(shared.best_result.Rx, shared.best_result.Ry) << "\n";
            found_file << "=================\n\n";
            found_file.close();
        }
        print_summary();
    } else if (g_sigint) {
        std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
        print_summary();
        exit_code = 130;
    } else if (shared.gpus_exhausted.load() >= num_gpus) {
        std::cout << RSLT_NOT_FOUND() << "\n";
        print_summary();
    } else {
        std::cout << "======== TERMINATED ===================================\n";
        print_summary();
    }

    return exit_code;
}
