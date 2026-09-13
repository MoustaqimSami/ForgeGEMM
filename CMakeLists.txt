cmake_minimum_required(VERSION 3.18)

# CUDA is enabled only if a CUDA compiler is actually found. This lets the
# host-side library, tests, and structure build and run on machines without a
# GPU (e.g. CI, a laptop) while still supporting the real CUDA build on
# development/benchmark machines.
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(ForgeGEMM LANGUAGES CXX CUDA)
    set(FORGEGEMM_HAVE_CUDA ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    project(ForgeGEMM LANGUAGES CXX)
    set(FORGEGEMM_HAVE_CUDA OFF)
    message(STATUS "No CUDA compiler found -- building host-only (Matrix/MatrixView, tests). "
                    "GPU kernels will be skipped until this is built on a CUDA-capable machine.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# Capture hardware/compiler info at configure time so benchmark results can
# always be traced back to what produced them.
find_program(NVIDIA_SMI nvidia-smi)
if(NVIDIA_SMI)
    execute_process(
        COMMAND ${NVIDIA_SMI} --query-gpu=name,driver_version,memory.total --format=csv,noheader
        OUTPUT_VARIABLE GPU_INFO
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "GPU detected: ${GPU_INFO}")
else()
    message(STATUS "nvidia-smi not found -- no GPU detected on this machine.")
endif()

add_library(forgegemm_core INTERFACE)
target_include_directories(forgegemm_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()
add_subdirectory(tests)

# Benchmarks depend on CUDA kernels that don't exist yet -- wired up once
# the naive/tiled kernels land (Day 3/5 of the sprint plan), not before.
if(FORGEGEMM_HAVE_CUDA)
    add_subdirectory(benchmarks)
endif()
