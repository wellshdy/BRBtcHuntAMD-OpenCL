/*
 * gpu_worker_host.h — Tipos compartilhados e declaração de run_on_gpu_opencl
 *
 * Porte de GPUWorker.h. Mantém a struct GpuShared quase idêntica ao
 * original (apenas remove referências a tipos HIP).
 */

#ifndef GPU_WORKER_HOST_H
#define GPU_WORKER_HOST_H

#include <cstdint>
#include <atomic>
#include <mutex>
#include <vector>
#include "ocl_common.h"

/* ── Found-result descriptor (mesmo conteúdo do original) ───────────── */
/* FoundResult e VanityResult já estão em ocl_common.h. */

/* ── Shared state between GPU threads and main loop ─────────────────── */
struct GpuShared {
    std::atomic<int>                any_found{0};
    std::mutex                      result_mtx;
    FoundResult                     best_result{};
    bool                            has_result{false};
    std::atomic<unsigned long long> total_hashes{0};
    std::atomic<unsigned long long> chunks_tried{0};
    std::atomic<int>                gpus_exhausted{0};
    std::atomic<int>                init_done{0};
    std::atomic<uint64_t>           cur_scalar_lo{0};
    std::atomic<uint64_t>           cur_scalar_hi{0};
    std::atomic<int>                setup_done{0};
    long double                     total_keys_adjusted{0.0L};
    std::atomic<uint32_t>           vanity_total{0};

    /* Vanity config (host-side) */
    uint32_t                  vanity_nibbles{0};
    uint32_t                  vanity_max_results{65536};
    std::vector<VanityResult> vanity_results;
    std::mutex                vanity_mtx;
};

/* Constants WARP_SIZE, MAX_BATCH_SIZE, FOUND_NONE/LOCK/READY are
   already defined in ocl_common.h (included above via this header). */

/* ── Worker entry point (definido em gpu_worker_opencl.cpp) ─────────── */
/*
 * gpu_id              : índice lógico (0..N-1) dentro da lista de GPUs selecionadas
 * physical_device_id  : índice da cl_device_id no array retornado por clGetDeviceIDs
 * range_start, range_end : 4 limbs little-endian
 * target_hash160      : 20 bytes do hash160 alvo
 * runtime_points_batch_size : B (pontos por batch)
 * runtime_batches_per_sm    : SMs × batches/SM = upper bound de threads
 * slices_per_launch    : batches por thread por launch
 * random_mode          : modo loteria
 * kernel_dir           : diretório onde estão os arquivos .cl
 * shared               : estado compartilhado
 */
void run_on_gpu_opencl(
    int            gpu_id,
    int            physical_device_id,
    const uint64_t range_start[4],
    const uint64_t range_end[4],
    const uint8_t  target_hash160[20],
    uint32_t       runtime_points_batch_size,
    uint32_t       runtime_batches_per_sm,
    uint32_t       slices_per_launch,
    bool           random_mode,
    const std::string& kernel_dir,
    GpuShared&     shared
);

#endif // GPU_WORKER_HOST_H
