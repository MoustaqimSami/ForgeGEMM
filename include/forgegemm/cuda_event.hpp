#pragma once

#include <cuda_runtime.h>

#include "forgegemm/cuda_check.hpp"

namespace forgegemm {

// RAII wrapper for a cudaEvent_t, used for GPU-side timing. Move-only, same
// reasoning as DeviceBuffer: exactly one owner, destructor cleans up,
// copying would risk two owners destroying the same event.
class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }

    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() { CUDA_CHECK(cudaEventSynchronize(event_)); }

    // Elapsed time in milliseconds between two recorded, synchronized events.
    static float elapsedMs(CudaEvent& start, CudaEvent& end) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
        return ms;
    }

    cudaEvent_t handle() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

}  // namespace forgegemm