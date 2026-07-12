/*
 * kernel_top.cl — Ponto de entrada único para o compilador OpenCL
 *
 * Este arquivo existe apenas como documentação. Em runtime, o host
 * (ocl_utils.cpp → concat_kernel_sources) carrega e concatena os 4
 * arquivos .cl nesta ordem:
 *
 *   1. include/ocl_common.h     — tipos C puros
 *   2. kernels/secp256k1_math.cl — aritmética modular 256-bit
 *   3. kernels/hash_pipeline.cl  — SHA-256 + RIPEMD-160 inline
 *   4. kernels/gpu_worker.cl     — kernel principal + batch inversion
 *
 * Não compile este arquivo diretamente. O host faz o pré-processamento.
 *
 * Kernels exportados (acessíveis via clCreateKernel):
 *   - scalarMulKernelBase       — usado na inicialização (P = scalar*G)
 *   - kernel_point_add_and_check_oneinv — kernel de busca principal
 */

#error "kernel_top.cl é apenas documentação. Não compile este arquivo diretamente — use o concatenador em ocl_utils.cpp"
