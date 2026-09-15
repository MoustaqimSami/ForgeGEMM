#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

namespace forgegemm {

// Wraps a CUDA runtime API call, throwing a std::runtime_error with a clear
// message (call, file, line, CUDA error string) if the call did not succeed.
// Every CUDA API call in this codebase should go through this macro rather
// than being checked ad hoc -- one consistent failure path.
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t forgegemm_err__ = (call);                                \
        if (forgegemm_err__ != cudaSuccess) {                                \
            std::ostringstream oss;                                         \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -- " \
                << #call << " failed: "                                      \
                << cudaGetErrorString(forgegemm_err__);                      \
            throw std::runtime_error(oss.str());                            \
        }                                                                    \
    } while (0)

}  // namespace forgegemm