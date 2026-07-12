/*
 * gpu_worker_opencl.cpp — Worker GPU principal (porta de GPUWorker.cpp)
 *
 * Substitui TODA a API HIP pela API OpenCL C padrão:
 *   hipSetDevice         → (não existe; context já vincula device)
 *   hipGetDeviceProperties → clGetDeviceInfo
 *   hipMalloc            → clCreateBuffer
 *   hipMemcpy H2D        → clEnqueueWriteBuffer
 *   hipMemcpy D2H        → clEnqueueReadBuffer
 *   hipMemcpyToSymbol    → clEnqueueWriteBuffer em buffer __constant
 *   hipLaunchKernelGGL   → clSetKernelArg + clEnqueueNDRangeKernel
 *   hipDeviceSynchronize → clFinish(queue)
 *   hipStreamCreate      → clCreateCommandQueueWithProperties
 *   hipStreamSynchronize → clFinish(queue)
 *   hipStreamQuery       → clGetEventInfo
 *   hipFree              → clReleaseMemObject
 *
 * O fluxo geral é idêntico ao original:
 *   1. Cria context + queue para o device
 *   2. Carrega kernel source (.cl), compila com clBuildProgram
 *   3. Pré-computa G*1..G*half, J=G*B (via scalarMulKernelBase)
 *   4. Copia para constant buffers (simula __constant do HIP)
 *   5. Loop principal:
 *      - Lança kernel_point_add_and_check_oneinv
 *      - Poll: lê d_found_flag, hashes_accum, vanity_count
 *      - Em FOUND_READY: copia FoundResult, sinaliza shared.any_found
 *      - Swap Px/Py ↔ Rx/Ry para próxima iteração (modo brute-force)
 *   6. Em random_mode: a cada iteração, regenera pontos via pick_random_start
 */

#include "ocl_common.h"
#include "ocl_helpers.h"
#include "gpu_worker_host.h"
#include "Lang.h"

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#include <CL/opencl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <chrono>
#include <thread>
#include <random>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <atomic>
#include <algorithm>

/* ocl_utils.cpp é compilado separadamente, mas expõe estas funções. */
extern std::string concat_kernel_sources(const std::string& kernel_dir);
extern cl_program build_program_with_logs(cl_context ctx, cl_device_id dev,
                                           const std::string& source,
                                           const std::string& options);
/* OclDeviceInfo já é definido em ocl_helpers.h (incluído acima). */
extern std::vector<OclDeviceInfo> enumerate_opencl_gpus();

/* ── Print mutex (shared across GPU worker threads) ──────────────── */
std::mutex g_print_mutex;

/* ── Signal flag (defined in main_opencl.cpp) ─────────────────────── */
extern volatile sig_atomic_t g_sigint;

/* Macro local de checagem de erro */
#define CK_CL(ans, msg) do { \
    cl_int _e = (ans); \
    if (_e != CL_SUCCESS) { \
        std::lock_guard<std::mutex> lk(g_print_mutex); \
        fprintf(stderr, "\n[GPU %d] %s: OpenCL error %d\n", gpu_id, msg, _e); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

/* ───────────────────────────────────────────────────────────────────────
 * run_on_gpu_opencl — Entry point
 * ─────────────────────────────────────────────────────────────────────── */
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
    GpuShared&     shared)
{
    /* ── 1. Enumerate devices and pick ours ─────────────────────────── */
    auto gpus = enumerate_opencl_gpus();
    if (physical_device_id < 0 || (size_t)physical_device_id >= gpus.size()) {
        std::lock_guard<std::mutex> lk(g_print_mutex);
        fprintf(stderr, "[GPU %d] Invalid physical device id %d\n", gpu_id, physical_device_id);
        std::exit(EXIT_FAILURE);
    }
    auto& dev_info = gpus[physical_device_id];

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev_info.device, nullptr, nullptr, &err);
    CK_CL(err, "clCreateContext");

    /* OpenCL 2.0+: command queues são criadas via clCreateCommandQueueWithProperties */
    const cl_queue_properties qprops[] = {
        CL_QUEUE_PROPERTIES, (cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
        0
    };
    /* Out-of-order pode dar problema em drivers antigos; use simple queue. */
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, dev_info.device, nullptr, &err);
    CK_CL(err, "clCreateCommandQueueWithProperties");
    cl_command_queue queue_read = clCreateCommandQueueWithProperties(ctx, dev_info.device, nullptr, &err);
    CK_CL(err, "clCreateCommandQueueWithProperties (queue_read)");

    /* ── 2. Compute launch parameters (espelha GPUWorker.cpp) ───────── */
    int threadsPerBlock = 256;
    if ((cl_uint)threadsPerBlock > dev_info.max_work_group_size)
        threadsPerBlock = (int)dev_info.max_work_group_size;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    /* GPU range length (256-bit) */
    uint64_t gpu_range_len[4];
    sub256(range_end, range_start, gpu_range_len);
    add256_u64(gpu_range_len, 1ULL, gpu_range_len);

    /* Memory budget for thread buffers (8 uint64 per thread = 64 bytes) */
    const uint64_t bytesPerThread = 2ULL * 4ULL * sizeof(uint64_t);
    cl_ulong totalGlobalMem = dev_info.global_mem;
    const uint64_t reserveBytes = 64ULL * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes)
                                                         : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    /* Divide range by batch_size */
    uint64_t q_div_batch[4]; uint64_t r_batch = 0ULL;
    divmod_256_by_u64(gpu_range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_batch);
    if (r_batch != 0ULL) {
        uint64_t adjust = (uint64_t)runtime_points_batch_size - r_batch;
        add256_u64(gpu_range_len, adjust, gpu_range_len);
        divmod_256_by_u64(gpu_range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_batch);
    }
    if ((q_div_batch[3] | q_div_batch[2] | q_div_batch[1]) != 0ULL) {
        std::lock_guard<std::mutex> lk(g_print_mutex);
        fprintf(stderr, "[GPU %d] %s\n", gpu_id, ERR_RANGE_LARGE());
        std::exit(EXIT_FAILURE);
    }
    uint64_t total_batches_u64 = q_div_batch[0];

    uint64_t userUpper = (uint64_t)dev_info.compute_units
                       * (uint64_t)runtime_batches_per_sm
                       * (uint64_t)threadsPerBlock;
    if (userUpper == 0ULL) userUpper = UINT64_MAX;

    uint64_t desired_upper = maxThreadsByMem;
    if (userUpper < desired_upper) desired_upper = userUpper;
    uint64_t threadsTotal = (desired_upper / (uint64_t)threadsPerBlock)
                          * (uint64_t)threadsPerBlock;
    if (threadsTotal < (uint64_t)threadsPerBlock) threadsTotal = (uint64_t)threadsPerBlock;

    if (total_batches_u64 < threadsTotal) {
        threadsTotal = (total_batches_u64 / (uint64_t)threadsPerBlock)
                     * (uint64_t)threadsPerBlock;
        if (threadsTotal < (uint64_t)threadsPerBlock) threadsTotal = (uint64_t)threadsPerBlock;
    }
    if ((total_batches_u64 % threadsTotal) != 0ULL) {
        uint64_t rem = total_batches_u64 % threadsTotal;
        total_batches_u64 += threadsTotal - rem;
        add256_u64(gpu_range_len,
                   (threadsTotal - rem) * (uint64_t)runtime_points_batch_size,
                   gpu_range_len);
    }

    shared.total_keys_adjusted = ld_from_u256(gpu_range_len);

    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    uint64_t per_thread_cnt[4]; uint64_t r_u64 = 0ULL;
    if (random_mode) {
        per_thread_cnt[0] = (uint64_t)runtime_points_batch_size * slices_per_launch;
        per_thread_cnt[1] = per_thread_cnt[2] = per_thread_cnt[3] = 0ULL;
    } else {
        divmod_256_by_u64(gpu_range_len, threadsTotal, per_thread_cnt, r_u64);
    }

    const uint32_t B    = runtime_points_batch_size;
    const uint32_t half = B >> 1;

    /* ── 3. Load + build kernel program ─────────────────────────────── */
    std::string src = concat_kernel_sources(kernel_dir);
    /* Opções de build do kernel OpenCL.
       - -cl-std=CL2.0: obrigatório para atomic_fetch_add, atomic_work_item_fence, etc.
       - -cl-mad-enable: permite v_mad_* (fused multiply-add) no GCN/RDNA.
       - -DOPENCL_RDNA1=1: ativa código específico RDNA1 (wave32).
       NOTA: -Werror foi removido — drivers AMD têm warnings benignos que
       impediriam a compilação na primeira tentativa. */
    std::string build_opts = "-cl-std=CL2.0 -cl-mad-enable -DOPENCL_RDNA1=1"
                           + std::string(" -DMAX_BATCH_SIZE=") + std::to_string(B)
                           + std::string(" -DBATCH_SIZE_CONST=") + std::to_string(B)
                           + std::string(" -DHALF_B_CONST=") + std::to_string(half);
    cl_program program = build_program_with_logs(ctx, dev_info.device, src, build_opts);

    cl_kernel krn_scalar_mul = clCreateKernel(program, "scalarMulKernelBase", &err);
    CK_CL(err, "clCreateKernel(scalarMulKernelBase)");
    cl_kernel krn_main = clCreateKernel(program, "kernel_point_add_and_check_oneinv", &err);
    CK_CL(err, "clCreateKernel(kernel_point_add_and_check_oneinv)");

    {
        cl_kernel krn_test = clCreateKernel(program, "test_hash_kernel", &err);
        if (err == CL_SUCCESS) {
            cl_mem d_test_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 5 * sizeof(uint32_t), nullptr, &err);
            CK_CL(err, "clCreateBuffer(d_test_out)");
            CK_CL(clSetKernelArg(krn_test, 0, sizeof(cl_mem), &d_test_out), "clSetKernelArg(krn_test)");
            size_t g_test = 1, l_test = 1;
            CK_CL(clEnqueueNDRangeKernel(queue, krn_test, 1, nullptr, &g_test, &l_test, 0, nullptr, nullptr), "enqueue krn_test");
            clFinish(queue);
            uint32_t h_test[5] = {0};
            clEnqueueReadBuffer(queue, d_test_out, CL_TRUE, 0, 5 * sizeof(uint32_t), h_test, 0, nullptr, nullptr);
            printf("[DEBUG] GPU computed Hash160 (key 144): ");
            for (int i = 0; i < 5; ++i) {
                // print bytes of each word in little-endian byte order
                uint32_t w = h_test[i];
                printf("%02x%02x%02x%02x", w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF);
            }
            printf("\n");
            clReleaseMemObject(d_test_out);
            clReleaseKernel(krn_test);
        } else {
            printf("[ERROR] Failed to create test_hash_kernel: %d\n", err);
        }
    }

    /* ── 4. Allocate device buffers ─────────────────────────────────── */
    size_t per_thread_bytes = (size_t)threadsTotal * 4 * sizeof(uint64_t);

    auto make_buf = [&](size_t bytes, const char* msg) -> cl_mem {
        cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        CK_CL(err, msg);
        return m;
    };

    cl_mem d_start_scalars = make_buf(per_thread_bytes, "clCreateBuffer(start_scalars)");
    if (threadsTotal == 0) {
        shared.init_done.fetch_add(1, std::memory_order_release);
        return;
    }
    cl_mem d_Px        = make_buf(per_thread_bytes, "clCreateBuffer(Px)");
    cl_mem d_Py        = make_buf(per_thread_bytes, "clCreateBuffer(Py)");
    cl_mem d_Rx        = make_buf(per_thread_bytes, "clCreateBuffer(Rx)");
    cl_mem d_Ry        = make_buf(per_thread_bytes, "clCreateBuffer(Ry)");
    cl_mem d_counts256 = make_buf(per_thread_bytes, "clCreateBuffer(counts256)");

    cl_mem d_found_flag = make_buf(sizeof(cl_int), "clCreateBuffer(found_flag)");
    cl_mem d_found_result = make_buf(sizeof(FoundResult), "clCreateBuffer(found_result)");
    cl_mem d_hashes_accum = make_buf(sizeof(cl_uint), "clCreateBuffer(hashes_accum)");
    cl_mem d_any_left    = make_buf(sizeof(cl_uint), "clCreateBuffer(any_left)");

    /* Vanity buffers */
    cl_mem d_vanity_count = nullptr;
    cl_mem d_vanity_buf   = nullptr;
    if (shared.vanity_nibbles > 0) {
        d_vanity_count = make_buf(sizeof(cl_uint), "clCreateBuffer(vanity_count)");
        d_vanity_buf   = make_buf((size_t)shared.vanity_max_results * sizeof(VanityResult),
                                  "clCreateBuffer(vanity_buf)");
        cl_uint vz = 0;
        CK_CL(clEnqueueWriteBuffer(queue, d_vanity_count, CL_TRUE, 0, sizeof(cl_uint), &vz,
                                    0, nullptr, nullptr), "init vanity_count");
    }

    /* Constant buffers (emulam __constant do HIP) */
    /* c_Gx, c_Gy: half * 4 * sizeof(u64) */
    size_t half_bytes = (size_t)half * 4 * sizeof(uint64_t);
    cl_mem c_Gx = clCreateBuffer(ctx, CL_MEM_READ_WRITE, half_bytes, nullptr, &err);
    CK_CL(err, "clCreateBuffer(c_Gx)");
    cl_mem c_Gy = clCreateBuffer(ctx, CL_MEM_READ_WRITE, half_bytes, nullptr, &err);
    CK_CL(err, "clCreateBuffer(c_Gy)");
    /* c_Jx, c_Jy: 4 * sizeof(u64) */
    cl_mem c_Jx = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(uint64_t), nullptr, &err);
    CK_CL(err, "clCreateBuffer(c_Jx)");
    cl_mem c_Jy = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(uint64_t), nullptr, &err);
    CK_CL(err, "clCreateBuffer(c_Jy)");

    /* c_target_hash160: 20 bytes; c_target_prefix: cl_uint */
    cl_mem c_target_hash160 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 20, nullptr, &err);
    CK_CL(err, "clCreateBuffer(c_target_hash160)");

    /* ── 5. Fill host buffers ───────────────────────────────────────── */
    std::vector<uint64_t> h_counts256(threadsTotal * 4);
    std::vector<uint64_t> h_start_scalars(threadsTotal * 4);

    for (uint64_t i = 0; i < threadsTotal; ++i) {
        h_counts256[i*4+0] = per_thread_cnt[0];
        h_counts256[i*4+1] = per_thread_cnt[1];
        h_counts256[i*4+2] = per_thread_cnt[2];
        h_counts256[i*4+3] = per_thread_cnt[3];
    }
    {
        uint64_t cur[4] = { range_start[0], range_start[1], range_start[2], range_start[3] };
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            uint64_t Sc[4]; add256_u64(cur, (uint64_t)half, Sc);
            h_start_scalars[i*4+0] = Sc[0];
            h_start_scalars[i*4+1] = Sc[1];
            h_start_scalars[i*4+2] = Sc[2];
            h_start_scalars[i*4+3] = Sc[3];
            uint64_t next[4]; add256(cur, per_thread_cnt, next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
        shared.cur_scalar_lo.store(h_start_scalars[0], std::memory_order_relaxed);
        shared.cur_scalar_hi.store(h_start_scalars[1], std::memory_order_relaxed);
    }

    /* ── 6. Upload initial buffers ──────────────────────────────────── */
    CK_CL(clEnqueueWriteBuffer(queue, d_start_scalars, CL_TRUE, 0, per_thread_bytes,
                                h_start_scalars.data(), 0, nullptr, nullptr),
          "upload start_scalars");
    CK_CL(clEnqueueWriteBuffer(queue, d_counts256, CL_TRUE, 0, per_thread_bytes,
                                h_counts256.data(), 0, nullptr, nullptr),
          "upload counts256");
    {
        cl_int z = FOUND_NONE;
        cl_uint z32 = 0u;
        CK_CL(clEnqueueWriteBuffer(queue, d_found_flag, CL_TRUE, 0, sizeof(cl_int), &z,
                                    0, nullptr, nullptr), "init found_flag");
        CK_CL(clEnqueueWriteBuffer(queue, d_hashes_accum, CL_TRUE, 0, sizeof(cl_uint), &z32,
                                    0, nullptr, nullptr), "init hashes_accum");
    }

    /* Target hash160 + prefix */
    cl_uint prefix_le = (cl_uint)target_hash160[0]
                      | ((cl_uint)target_hash160[1] << 8)
                      | ((cl_uint)target_hash160[2] << 16)
                      | ((cl_uint)target_hash160[3] << 24);
    CK_CL(clEnqueueWriteBuffer(queue, c_target_hash160, CL_TRUE, 0, 20,
                                target_hash160, 0, nullptr, nullptr),
          "upload c_target_hash160");

    /* ── 7. Helper: launch scalarMulKernelBase ──────────────────────── */
    auto launch_scalar_mul = [&](cl_mem scalars, cl_mem outX, cl_mem outY, int N) {
        int bs = (N + threadsPerBlock - 1) / threadsPerBlock;
        size_t global = (size_t)bs * (size_t)threadsPerBlock;
        size_t local  = (size_t)threadsPerBlock;

        int arg = 0;
        CK_CL(clSetKernelArg(krn_scalar_mul, arg++, sizeof(cl_mem), &scalars), "arg scalars");
        CK_CL(clSetKernelArg(krn_scalar_mul, arg++, sizeof(cl_mem), &outX),    "arg outX");
        CK_CL(clSetKernelArg(krn_scalar_mul, arg++, sizeof(cl_mem), &outY),    "arg outY");
        CK_CL(clSetKernelArg(krn_scalar_mul, arg++, sizeof(int),     &N),      "arg N");

        CK_CL(clEnqueueNDRangeKernel(queue, krn_scalar_mul, 1, nullptr, &global, &local,
                                      0, nullptr, nullptr),
              "launch scalarMulKernelBase");
        CK_CL(clFinish(queue), "sync scalarMulKernelBase");
    };

    /* ── 8. Compute initial EC points (Px, Py = scalar*G) ───────────── */
    launch_scalar_mul(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
    {
        uint64_t dbg_x[4] = {0};
        uint64_t dbg_y[4] = {0};
        clEnqueueReadBuffer(queue, d_Px, CL_TRUE, 0, 4 * sizeof(uint64_t), dbg_x, 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, d_Py, CL_TRUE, 0, 4 * sizeof(uint64_t), dbg_y, 0, nullptr, nullptr);
        std::lock_guard<std::mutex> lk(g_print_mutex);
        printf("[DEBUG] Thread 0 start scalar point:\n");
        printf("  X: %016llx %016llx %016llx %016llx\n",
               (unsigned long long)dbg_x[3], (unsigned long long)dbg_x[2],
               (unsigned long long)dbg_x[1], (unsigned long long)dbg_x[0]);
        printf("  Y: %016llx %016llx %016llx %016llx\n",
               (unsigned long long)dbg_y[3], (unsigned long long)dbg_y[2],
               (unsigned long long)dbg_y[1], (unsigned long long)dbg_y[0]);
    }

    /* ── 9. Compute G*1..G*half → c_Gx, c_Gy ────────────────────────── */
    {
        std::vector<uint64_t> h_scalars_half((size_t)half * 4, 0);
        for (uint32_t k = 0; k < half; ++k)
            h_scalars_half[(size_t)k*4] = (uint64_t)(k + 1);

        cl_mem d_sh  = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  half_bytes, nullptr, &err);
        CK_CL(err, "clCreateBuffer(d_sh)");
        cl_mem d_Gxh = clCreateBuffer(ctx, CL_MEM_READ_WRITE, half_bytes, nullptr, &err);
        CK_CL(err, "clCreateBuffer(d_Gxh)");
        cl_mem d_Gyh = clCreateBuffer(ctx, CL_MEM_READ_WRITE, half_bytes, nullptr, &err);
        CK_CL(err, "clCreateBuffer(d_Gyh)");

        CK_CL(clEnqueueWriteBuffer(queue, d_sh, CL_TRUE, 0, half_bytes,
                                    h_scalars_half.data(), 0, nullptr, nullptr),
              "upload half scalars");

        launch_scalar_mul(d_sh, d_Gxh, d_Gyh, (int)half);

        /* Copia d_Gxh → c_Gx, d_Gyh → c_Gy (ambos são device buffers) */
        CK_CL(clEnqueueCopyBuffer(queue, d_Gxh, c_Gx, 0, 0, half_bytes, 0, nullptr, nullptr),
              "copy Gx → c_Gx");
        CK_CL(clEnqueueCopyBuffer(queue, d_Gyh, c_Gy, 0, 0, half_bytes, 0, nullptr, nullptr),
              "copy Gy → c_Gy");
        CK_CL(clFinish(queue), "sync copy half tables");

        clReleaseMemObject(d_sh);
        clReleaseMemObject(d_Gxh);
        clReleaseMemObject(d_Gyh);
    }

    /* ── 10. Compute J = G*B → c_Jx, c_Jy ───────────────────────────── */
    {
        uint64_t h_scB[4] = { (uint64_t)B, 0, 0, 0 };
        cl_mem d_scB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sizeof(uint64_t), nullptr, &err);
        CK_CL(err, "clCreateBuffer(d_scB)");

        CK_CL(clEnqueueWriteBuffer(queue, d_scB, CL_TRUE, 0, 4*sizeof(uint64_t), h_scB,
                                    0, nullptr, nullptr), "upload scB");

        int N1 = 1;
        int arg = 0;
        CK_CL(clSetKernelArg(krn_scalar_mul, arg++, sizeof(cl_mem), &d_scB), "J arg1");
        CK_CL(clSetKernelArg(krn_scalar_mul, arg++, sizeof(cl_mem), &c_Jx),  "J arg2");
        CK_CL(clSetKernelArg(krn_scalar_mul, arg++, sizeof(cl_mem), &c_Jy),  "J arg3");
        CK_CL(clSetKernelArg(krn_scalar_mul, arg++, sizeof(int),     &N1),   "J arg4");
        size_t g1=1, l1=1;
        CK_CL(clEnqueueNDRangeKernel(queue, krn_scalar_mul, 1, nullptr, &g1, &l1,
                                      0, nullptr, nullptr), "launch scalarMulKernelBase(J)");
        CK_CL(clFinish(queue), "sync J");

        clReleaseMemObject(d_scB);
    }

    /* ── 11. Print GPU info block ───────────────────────────────────── */
    {
        std::lock_guard<std::mutex> lk(g_print_mutex);
        std::cout << "======== GPU " << gpu_id << " : " << dev_info.name
                  << " (OpenCL 2.0) ========\n";
        std::cout << std::left << std::setw(20) << LBL_SM()              << " : " << dev_info.compute_units << "\n";
        std::cout << std::left << std::setw(20) << LBL_THREADS_BLOCK()   << " : " << threadsPerBlock << "\n";
        std::cout << std::left << std::setw(20) << LBL_BLOCKS()          << " : " << blocks << "\n";
        std::cout << std::left << std::setw(20) << LBL_TOTAL_THREADS()   << " : " << threadsTotal << "\n";
        std::cout << std::left << std::setw(20) << LBL_BATCH_SIZE()      << " : " << B << "\n";
        std::cout << std::left << std::setw(20) << LBL_BATCHES_SM()      << " : " << runtime_batches_per_sm << "\n";
        std::cout << std::left << std::setw(20) << LBL_BATCHES_LAUNCH()  << " : " << slices_per_launch << " (per thread)\n";
        std::cout << std::left << std::setw(20) << LBL_MEM_UTIL()        << " : "
                  << human_bytes((double)per_thread_bytes * 2.0) << " (thread buffers)\n";
        std::cout << "------------------------------------------------------- \n";
        std::cout.flush();
    }

    /* Signal init complete */
    shared.init_done.fetch_add(1, std::memory_order_release);

    /* ── 12. Random-mode setup (espelha GPUWorker.cpp) ──────────────── */
    unsigned long long last_hashes_gpu = 0ULL;
    bool stop_all = false;
    bool completed_all = false;
    uint32_t last_vanity_count = 0;

    uint64_t full_range_len[4];
    sub256(range_end, range_start, full_range_len);
    add256_u64(full_range_len, 1ULL, full_range_len);
    uint64_t chunk_span = (uint64_t)threadsTotal * per_thread_cnt[0];

    std::mt19937_64 rng_state(
        (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count()
        ^ ((uint64_t)gpu_id * 0x9e3779b97f4a7c15ULL)
    );

    auto pick_random_start = [&](uint64_t chunk_start[4]) {
        uint64_t rl_lo = full_range_len[0];
        uint64_t rl_hi = full_range_len[1];
        if (rl_lo < chunk_span) { if (rl_hi > 0) --rl_hi; }
        rl_lo -= chunk_span;
        if (rl_hi == 0 && rl_lo == 0) rl_lo = 1;

        uint64_t r_lo = rng_state();
        uint64_t r_hi = rng_state();

        uint64_t off_lo, off_hi;
        if (rl_hi == 0) {
#if defined(_MSC_VER) && !defined(__clang__)
            uint64_t rem = 0;
            (void)_udiv128(r_hi % rl_lo, r_lo, rl_lo, &rem);
            off_lo = rem;
            off_hi = 0;
#else
            __uint128_t rr = ((__uint128_t)r_hi << 64) | r_lo;
            off_lo = (uint64_t)(rr % rl_lo);
            off_hi = 0;
#endif
        } else {
            uint64_t rm_lo = 0, rm_hi = 0;
            off_lo = 0; off_hi = 0;
            for (int _i = 0; _i < 128; ++_i) {
                uint64_t top = (r_hi >> 63);
                rm_lo = (rm_lo << 1) | (rm_hi >> 63);
                rm_hi = (rm_hi << 1) | top;
                off_lo = (off_lo << 1) | (off_hi >> 63);
                off_hi = (off_hi << 1);
                r_hi = (r_hi << 1) | (r_lo >> 63);
                r_lo = (r_lo << 1);
                if (rm_hi > rl_hi || (rm_hi == rl_hi && rm_lo >= rl_lo)) {
                    uint64_t diff = rm_lo - rl_lo;
                    uint64_t brw = (diff > rm_lo) ? 1ULL : 0ULL;
                    rm_lo = diff;
                    rm_hi = rm_hi - rl_hi - brw;
                    off_lo |= 1;
                }
            }
        }
        uint64_t offset[4] = {off_lo, off_hi, 0, 0};
        add256(range_start, offset, chunk_start);
    };

    auto reinit_chunk = [&](const uint64_t chunk_start[4]) {
        uint64_t cur[4] = {chunk_start[0], chunk_start[1], chunk_start[2], chunk_start[3]};
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            uint64_t Sc[4]; add256_u64(cur, (uint64_t)half, Sc);
            h_start_scalars[i*4+0]=Sc[0]; h_start_scalars[i*4+1]=Sc[1];
            h_start_scalars[i*4+2]=Sc[2]; h_start_scalars[i*4+3]=Sc[3];
            uint64_t next[4]; add256(cur, per_thread_cnt, next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            h_counts256[i*4+0]=per_thread_cnt[0]; h_counts256[i*4+1]=per_thread_cnt[1];
            h_counts256[i*4+2]=per_thread_cnt[2]; h_counts256[i*4+3]=per_thread_cnt[3];
        }
        shared.cur_scalar_lo.store(h_start_scalars[0], std::memory_order_relaxed);
        shared.cur_scalar_hi.store(h_start_scalars[1], std::memory_order_relaxed);

        CK_CL(clEnqueueWriteBuffer(queue, d_start_scalars, CL_TRUE, 0, per_thread_bytes,
                                    h_start_scalars.data(), 0, nullptr, nullptr),
              "reinit start_scalars");
        CK_CL(clEnqueueWriteBuffer(queue, d_counts256, CL_TRUE, 0, per_thread_bytes,
                                    h_counts256.data(), 0, nullptr, nullptr),
              "reinit counts256");
        launch_scalar_mul(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
    };
    shared.setup_done.store(1, std::memory_order_release);

    /* ── 13. Pre-compute vanity constants (passed as kernel args) ───── */
    uint32_t vanity_compare_bytes = shared.vanity_nibbles / 2;
    uint32_t vanity_match_nibble  = shared.vanity_nibbles % 2;

    /* ── 14. Main loop ──────────────────────────────────────────────── */
    while (!stop_all) {
        if (shared.any_found.load(std::memory_order_relaxed)) break;
        if (g_sigint) break;

        if (random_mode) {
            uint64_t chunk_start[4];
            pick_random_start(chunk_start);
            reinit_chunk(chunk_start);
            shared.chunks_tried.fetch_add(1, std::memory_order_relaxed);
            if (shared.any_found.load(std::memory_order_relaxed)) break;
            if (g_sigint) break;
        }

        /* Zero d_any_left and d_hashes_accum, and reset last_hashes_gpu */
        cl_uint zeroU = 0u;
        CK_CL(clEnqueueWriteBuffer(queue, d_any_left, CL_FALSE, 0, sizeof(cl_uint), &zeroU,
                                    0, nullptr, nullptr), "zero d_any_left");
        CK_CL(clEnqueueWriteBuffer(queue, d_hashes_accum, CL_FALSE, 0, sizeof(cl_uint), &zeroU,
                                    0, nullptr, nullptr), "zero d_hashes_accum");
        last_hashes_gpu = 0u;

        uint32_t slices_done = 0;
        while (slices_done < slices_per_launch && !stop_all && !g_sigint) {
            uint32_t current_sub_slices = std::min((uint32_t)4, slices_per_launch - slices_done);

            /* Set kernel args (must re-set every launch because some buffers swap) */
            int arg = 0;
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem), &d_Px),              "arg Px");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem), &d_Py),              "arg Py");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem), &d_Rx),              "arg Rx");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem), &d_Ry),              "arg Ry");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem), &d_start_scalars),   "arg start_scalars");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem), &d_counts256),       "arg counts256");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_ulong), &threadsTotal),    "arg threadsTotal");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_uint),  &B),               "arg B");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_uint),  &current_sub_slices), "arg max_batches");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &d_found_flag),     "arg found_flag");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &d_found_result),   "arg found_result");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &d_hashes_accum),   "arg hashes_accum");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &d_any_left),       "arg any_left");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &d_vanity_count),   "arg vanity_count");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &d_vanity_buf),     "arg vanity_buf");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_uint), &shared.vanity_max_results), "arg vanity_max");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &c_Gx),             "arg c_Gx");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &c_Gy),             "arg c_Gy");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &c_Jx),             "arg c_Jx");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &c_Jy),             "arg c_Jy");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_mem),  &c_target_hash160), "arg target_hash160");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_uint), &prefix_le),        "arg target_prefix");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_uint), &vanity_compare_bytes), "arg vanity_cmp_bytes");
            CK_CL(clSetKernelArg(krn_main, arg++, sizeof(cl_uint), &vanity_match_nibble), "arg vanity_match_nib");

            size_t global = (size_t)blocks * (size_t)threadsPerBlock;
            size_t local  = (size_t)threadsPerBlock;
            cl_event ev_kernel = nullptr;
            err = clEnqueueNDRangeKernel(queue, krn_main, 1, nullptr, &global, &local,
                                          0, nullptr, &ev_kernel);
            if (err != CL_SUCCESS) {
                std::lock_guard<std::mutex> lk(g_print_mutex);
                fprintf(stderr, "\n[GPU %d] Kernel launch error: %d\n", gpu_id, err);
                stop_all = true;
                break;
            }

            /* Poll until kernel finishes */
            int poll_count = 0;
            while (!stop_all) {
                if (shared.any_found.load(std::memory_order_relaxed)) {
                    cl_int ready = FOUND_READY;
                    clEnqueueWriteBuffer(queue, d_found_flag, CL_TRUE, 0, sizeof(cl_int), &ready,
                                          0, nullptr, nullptr);
                    stop_all = true;
                    break;
                }
                if (g_sigint) { stop_all = true; break; }

                // Periodically read hashes_accum (every 100ms) to update speed smoothly
                poll_count++;
                if (poll_count >= 10) {
                    poll_count = 0;
                    cl_uint h_hashes = 0u;
                    cl_int readErr = clEnqueueReadBuffer(queue_read, d_hashes_accum, CL_TRUE, 0, sizeof(cl_uint), &h_hashes,
                                                         0, nullptr, nullptr);
                    if (readErr == CL_SUCCESS) {
                        uint32_t diff = h_hashes - (uint32_t)last_hashes_gpu;
                        if (diff > 0) {
                            shared.total_hashes.fetch_add(diff, std::memory_order_relaxed);
                            last_hashes_gpu = h_hashes;
                        }
                    }
                }

                cl_int status = 0;
                cl_int infoErr = clGetEventInfo(ev_kernel, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, nullptr);
                if (infoErr == CL_SUCCESS) {
                    if (status == CL_COMPLETE) {
                        break;
                    } else if (status < 0) {
                        std::lock_guard<std::mutex> lk(g_print_mutex);
                        fprintf(stderr, "\n[GPU %d] Erro de execucao: %d (A placa de video pode ter resetado ou sofrido timeout TDR do Windows)\n", gpu_id, status);
                        stop_all = true;
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            if (ev_kernel) {
                clReleaseEvent(ev_kernel);
            }

            if (!stop_all) {
                cl_uint h_hashes = 0u;
                CK_CL(clEnqueueReadBuffer(queue, d_hashes_accum, CL_TRUE, 0, sizeof(cl_uint), &h_hashes,
                                           0, nullptr, nullptr), "read hashes_accum");
                uint32_t diff = h_hashes - (uint32_t)last_hashes_gpu;
                if (diff > 0) {
                    shared.total_hashes.fetch_add(diff, std::memory_order_relaxed);
                    last_hashes_gpu = h_hashes;
                }

                cl_int host_found = 0;
                CK_CL(clEnqueueReadBuffer(queue, d_found_flag, CL_TRUE, 0, sizeof(cl_int), &host_found,
                                           0, nullptr, nullptr), "read d_found_flag");
                if (host_found == FOUND_READY) {
                    FoundResult res{};
                    CK_CL(clEnqueueReadBuffer(queue, d_found_result, CL_TRUE, 0, sizeof(FoundResult), &res,
                                               0, nullptr, nullptr), "read d_found_result");
                    {
                        std::lock_guard<std::mutex> lk(shared.result_mtx);
                        if (!shared.has_result) {
                            shared.best_result = res;
                            shared.has_result  = true;
                        }
                    }
                    shared.any_found.store(1, std::memory_order_release);
                    stop_all = true;
                }
            }

            /* Swap Px/Py ↔ Rx/Ry for next sub-launch to start from the outputs of current sub-launch */
            std::swap(d_Px, d_Rx);
            std::swap(d_Py, d_Ry);

            slices_done += current_sub_slices;
        }

        /* Read back vanity results */
        if (shared.vanity_nibbles > 0 && d_vanity_count) {
            cl_uint current_count = 0;
            clEnqueueReadBuffer(queue, d_vanity_count, CL_TRUE, 0, sizeof(cl_uint), &current_count,
                                 0, nullptr, nullptr);
            cl_uint capped = std::min(current_count, shared.vanity_max_results);
            if (capped > last_vanity_count) {
                cl_uint n_new = capped - last_vanity_count;
                std::vector<VanityResult> results(n_new);
                clEnqueueReadBuffer(queue, d_vanity_buf, CL_TRUE,
                                     (size_t)last_vanity_count * sizeof(VanityResult),
                                     (size_t)n_new * sizeof(VanityResult),
                                     results.data(), 0, nullptr, nullptr);
                {
                    std::lock_guard<std::mutex> lk(shared.vanity_mtx);
                    shared.vanity_results.insert(shared.vanity_results.end(),
                                                  results.begin(), results.end());
                }
                shared.vanity_total.fetch_add(n_new, std::memory_order_relaxed);
            }
            last_vanity_count = capped;
        }

        if (stop_all || g_sigint) break;

        cl_uint h_any = 0u;
        clEnqueueReadBuffer(queue, d_any_left, CL_TRUE, 0, sizeof(cl_uint), &h_any,
                             0, nullptr, nullptr);

        if (random_mode) {
            /* Chunk done — loop back to pick a new random position */
        } else {
            if (h_any == 0u) { completed_all = true; break; }
        }
    }

    CK_CL(clFinish(queue), "final clFinish");

    /* Read back remaining vanity results */
    if (shared.vanity_nibbles > 0 && d_vanity_count) {
        cl_uint current_count = 0;
        clEnqueueReadBuffer(queue, d_vanity_count, CL_TRUE, 0, sizeof(cl_uint), &current_count,
                             0, nullptr, nullptr);
        cl_uint capped = std::min(current_count, shared.vanity_max_results);
        if (capped > last_vanity_count) {
            cl_uint n_new = capped - last_vanity_count;
            std::vector<VanityResult> results(n_new);
            clEnqueueReadBuffer(queue, d_vanity_buf, CL_TRUE,
                                 (size_t)last_vanity_count * sizeof(VanityResult),
                                 (size_t)n_new * sizeof(VanityResult),
                                 results.data(), 0, nullptr, nullptr);
            {
                std::lock_guard<std::mutex> lk(shared.vanity_mtx);
                shared.vanity_results.insert(shared.vanity_results.end(),
                                              results.begin(), results.end());
            }
            shared.vanity_total.fetch_add(n_new, std::memory_order_relaxed);
        }
    }

    /* ── 15. Cleanup ────────────────────────────────────────────────── */
    clReleaseMemObject(d_start_scalars);
    clReleaseMemObject(d_Px); clReleaseMemObject(d_Py);
    clReleaseMemObject(d_Rx); clReleaseMemObject(d_Ry);
    clReleaseMemObject(d_counts256);
    clReleaseMemObject(d_found_flag); clReleaseMemObject(d_found_result);
    clReleaseMemObject(d_hashes_accum); clReleaseMemObject(d_any_left);
    if (d_vanity_count) clReleaseMemObject(d_vanity_count);
    if (d_vanity_buf)   clReleaseMemObject(d_vanity_buf);
    clReleaseMemObject(c_Gx); clReleaseMemObject(c_Gy);
    clReleaseMemObject(c_Jx); clReleaseMemObject(c_Jy);
    clReleaseMemObject(c_target_hash160);

    clReleaseKernel(krn_main);
    clReleaseKernel(krn_scalar_mul);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseCommandQueue(queue_read);
    clReleaseContext(ctx);

    if (completed_all)
        shared.gpus_exhausted.fetch_add(1, std::memory_order_relaxed);
}
