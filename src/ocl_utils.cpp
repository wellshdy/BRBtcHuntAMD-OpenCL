/*
 * ocl_utils.cpp — Utilitários OpenCL: carregamento, concatenação, build
 *
 * Funções:
 *   - load_file_to_string: lê arquivo .cl do disco
 *   - concat_kernel_sources: concatena ocl_common.h + secp256k1_math.cl +
 *                            hash_pipeline.cl + gpu_worker.cl em uma string
 *   - build_program_with_logs: clBuildProgram + extrai log de build
 *   - check_cl_error: macro/função de erro
 */

#include "ocl_common.h"
#include "ocl_helpers.h"
#include <CL/opencl.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

/* ── Error checking ─────────────────────────────────────────────────── */
#define CL_CHECK(ans, msg) do { \
    cl_int _e = (ans); \
    if (_e != CL_SUCCESS) { \
        std::cerr << "[OpenCL] " << (msg) << " failed: error " << _e \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

/* ── File loader ────────────────────────────────────────────────────── */
static std::string load_file_to_string(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[OpenCL] Cannot open file: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/* ── Kernel source concatenation ──────────────────────────────────────
 *
 * O compilador OpenCL da AMD NÃO processa #include por padrão (apesar
 * de a spec OpenCL C 2.0 permitir, o suporte é inconsistente entre
 * drivers). Para garantir portabilidade, o host concatena manualmente
 * os arquivos em uma única string.
 *
 * Importante: ocl_common.h tem include guards (#ifndef OCL_COMMON_H /
 * #define / #endif). Como vamos pré-definir OCL_COMMON_H antes de
 * injetar o conteúdo, o guard #ifndef fará o conteúdo ser skipado caso
 * o mesmo header seja injetado novamente (nautilus de segurança).
 *
 * Ordem obrigatória (respeitando dependências):
 *   1. ocl_common.h   — typedefs (uint256, ECPointA, etc.)
 *   2. secp256k1_math.cl  — aritmética modular
 *   3. hash_pipeline.cl   — SHA-256 + RIPEMD-160
 *   4. gpu_worker.cl      — kernel principal
 *
 * kernel_dir: caminho para a pasta "kernels/". O include ocl_common.h
 *             é procurado em "../include/" relativo a kernel_dir.
 */
std::string concat_kernel_sources(const std::string& kernel_dir) {
    /* Caminho para ocl_common.h: <kernel_dir>/../include/ocl_common.h */
    std::string common_path = kernel_dir + "/../include/ocl_common.h";
    std::string common    = load_file_to_string(common_path);
    std::string math_cl   = load_file_to_string(kernel_dir + "/secp256k1_math.cl");
    std::string hash_cl   = load_file_to_string(kernel_dir + "/hash_pipeline.cl");
    std::string worker_cl = load_file_to_string(kernel_dir + "/gpu_worker.cl");

    std::ostringstream out;

    /* Pré-definições para o dispositivo */
    out << "/* === Auto-concatenated by ocl_utils.cpp === */\n";
    out << "#define OPENCL_DEVICE 1\n";
    out << "#define OPENCL_RDNA1 1\n\n";

    /* 1. ocl_common.h — stream direto. O include guard interno
       (#ifndef OCL_COMMON_H / #define / #endif) evita redefinições. */
    out << "/* === ocl_common.h === */\n";
    out << common << "\n\n";

    /* 2. secp256k1_math.cl — sem #includes próprios, stream direto. */
    out << "/* === secp256k1_math.cl === */\n";
    out << math_cl << "\n\n";

    /* 3. hash_pipeline.cl */
    out << "/* === hash_pipeline.cl === */\n";
    out << hash_cl << "\n\n";

    /* 4. gpu_worker.cl */
    out << "/* === gpu_worker.cl === */\n";
    out << worker_cl << "\n";

    return out.str();
}

/* ── Program builder com log detalhado ──────────────────────────────── */
cl_program build_program_with_logs(
    cl_context ctx,
    cl_device_id dev,
    const std::string& source,
    const std::string& options)
{
    cl_int err;
    const char* src = source.c_str();
    size_t src_len = source.size();
    cl_program program = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
    CL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &dev, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        /* Extrai log de build */
        size_t log_size = 0;
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, 0);
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "=== OpenCL Build Error ===\n";
        std::cerr << "Error code: " << err << "\n";
        std::cerr << "Build log:\n" << log.data() << "\n";
        std::cerr << "=========================\n";

        /* Dump do source para debug (em arquivo temporário) */
        std::ofstream dbg("build_failed_source.cl");
        if (dbg.is_open()) {
            dbg << source;
            dbg.close();
            std::cerr << "Source dumped to build_failed_source.cl for inspection.\n";
        }

        clReleaseProgram(program);
        std::exit(EXIT_FAILURE);
    }

    /* Avisa sobre warnings (log mesmo em sucesso pode ter avisos úteis) */
    size_t log_size = 0;
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    if (log_size > 1) {
        std::vector<char> log(log_size + 1, 0);
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::string s(log.data());
        if (s.find("warning") != std::string::npos ||
            s.find("error")   != std::string::npos) {
            std::cerr << "[OpenCL build log]:\n" << s << std::endl;
        }
    }

    return program;
}

/* ── Platform/device enumeration ────────────────────────────────────── */
/* OclDeviceInfo é definido em ocl_helpers.h. */

std::vector<OclDeviceInfo> enumerate_opencl_gpus() {
    std::vector<OclDeviceInfo> result;
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0)
        return result;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (cl_platform_id plat : platforms) {
        cl_uint num_devs = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devs) != CL_SUCCESS)
            continue;
        std::vector<cl_device_id> devs(num_devs);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devs, devs.data(), nullptr);

        for (cl_device_id d : devs) {
            OclDeviceInfo info{};
            info.platform = plat;
            info.device   = d;

            char name[256] = {0};
            clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            info.name = name;

            clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &info.compute_units, nullptr);
            if (info.compute_units <= 1) {
                continue; // Skip phantom/ghost devices or weak fallback layers
            }
            clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &info.global_mem, nullptr);
            clGetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &info.max_clock, nullptr);
            size_t max_wgs = 0;
            clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wgs, nullptr);
            info.max_work_group_size = (cl_uint)max_wgs;

            result.push_back(info);
        }
    }
    return result;
}
