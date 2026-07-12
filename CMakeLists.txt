# CMakeLists.txt — BRBtcHuntAMD-OpenCL
#
# Build portável (Windows + Linux). Requer:
#   - OpenCL 1.2+ headers/runtime (AMD APP SDK ou vendor SDK)
#   - C++17

cmake_minimum_required(VERSION 3.16)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(BRBtcHuntAMD-OpenCL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── OpenCL ────────────────────────────────────────────────────────────
# Tenta achar o header OpenCL via find_path; em Windows normalmente em
# "C:\Program Files\AMD APP SDK\3.0\include" ou via pacote chocolatey
# "opencl-headers" no path padrão.
find_path(OPENCL_INCLUDE_DIR
    NAMES CL/opencl.h
    PATHS
        "$ENV{AMDAPPSDKROOT}/include"
        "$ENV{INTELOCLSDKROOT}/include"
        "$ENV{CUDA_PATH}/include"
        /usr/include
        /usr/local/include
    DOC "Path to OpenCL headers"
)

find_library(OPENCL_LIBRARY
    NAMES OpenCL
    PATHS
        "$ENV{AMDAPPSDKROOT}/lib/x86_64"
        "$ENV{AMDAPPSDKROOT}/lib/x64"
        "$ENV{INTELOCLSDKROOT}/lib/x64"
        "$ENV{CUDA_PATH}/lib/x64"
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
    DOC "Path to OpenCL library"
)

if(NOT OPENCL_INCLUDE_DIR OR NOT OPENCL_LIBRARY)
    message(STATUS "OpenCL not found locally. Fetching Khronos OpenCL headers and loader from GitHub...")
    include(FetchContent)
    
    FetchContent_Declare(
        opencl_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
        GIT_TAG        v2024.05.08
    )
    FetchContent_MakeAvailable(opencl_headers)
    
    FetchContent_Declare(
        opencl_icd_loader
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
        GIT_TAG        v2024.05.08
    )
    set(OPENCL_ICD_LOADER_HEADERS_DIR "${opencl_headers_SOURCE_DIR}" CACHE PATH "Path to OpenCL headers")
    FetchContent_MakeAvailable(opencl_icd_loader)
    
    set(OPENCL_INCLUDE_DIR "${opencl_headers_SOURCE_DIR}")
    set(OPENCL_LIBRARY OpenCL)
endif()

message(STATUS "OpenCL include: ${OPENCL_INCLUDE_DIR}")
message(STATUS "OpenCL library: ${OPENCL_LIBRARY}")

# ── OpenSSL removed — using local SimpleSHA256.h instead ─────────────

# ── Sources ───────────────────────────────────────────────────────────
set(SOURCES
    src/main_opencl.cpp
    src/gpu_worker_opencl.cpp
    src/ocl_utils.cpp
)

set(HEADERS
    include/ocl_common.h
    include/ocl_helpers.h
    include/gpu_worker_host.h
    include/SimpleSHA256.h
    include/Lang.h
)

set(KERNELS
    kernels/secp256k1_math.cl
    kernels/hash_pipeline.cl
    kernels/gpu_worker.cl
)

add_executable(BRBtcHuntAMD-OpenCL ${SOURCES} ${HEADERS} ${KERNELS})

target_include_directories(BRBtcHuntAMD-OpenCL PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENCL_INCLUDE_DIR}
)

target_link_libraries(BRBtcHuntAMD-OpenCL PRIVATE
    ${OPENCL_LIBRARY}
)

# ── Windows-specific: link against static runtime to avoid MSVCRedist ─
if(MSVC)
    set_property(TARGET BRBtcHuntAMD-OpenCL PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_options(BRBtcHuntAMD-OpenCL PRIVATE /W3 /EHsc /O2)
endif()

# ── Linux: enable pthread for std::thread ─────────────────────────────
if(UNIX AND NOT APPLE)
    target_link_libraries(BRBtcHuntAMD-OpenCL PRIVATE pthread)
endif()

# ── Copy kernels folder to build directory (so the exe finds them) ────
add_custom_command(TARGET BRBtcHuntAMD-OpenCL POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        $<TARGET_FILE_DIR:BRBtcHuntAMD-OpenCL>/kernels
)

# ── Install (opcional) ────────────────────────────────────────────────
install(TARGETS BRBtcHuntAMD-OpenCL RUNTIME DESTINATION bin)
install(DIRECTORY kernels DESTINATION share/BRBtcHuntAMD-OpenCL)
