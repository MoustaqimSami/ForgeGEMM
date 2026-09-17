#pragma once

#include <cstddef>

namespace forgegemm {

// Describes a single GEMM operation: C = alpha * op(A) * op(B) + beta * C
//
// Backend-neutral by design: this header must never include CUDA types
// (no cudaStream_t, no device pointers) so it can be included anywhere,
// including files that never touch the GPU. The naive kernel we write next
// will take one of these as its config, instead of a long list of loose
// parameters.
struct GemmConfig {
    std::size_t M = 0;  // rows of A / rows of C
    std::size_t N = 0;  // cols of B / cols of C
    std::size_t K = 0;  // cols of A / rows of B
    double alpha = 1.0;
    double beta = 0.0;
    bool transposeA = false;
    bool transposeB = false;
};

}  // namespace forgegemm