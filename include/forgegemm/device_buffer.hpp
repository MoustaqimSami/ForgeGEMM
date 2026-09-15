#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

#include "forgegemm/cuda_check.hpp"

namespace forgegemm {

// RAII wrapper for a device (GPU) memory allocation. Move-only: exactly one
// DeviceBuffer owns a given allocation at a time, and it frees it in the
// destructor. Copy is deleted deliberately -- copying would either mean a
// silent, expensive deep device-to-device copy, or two owners racing to
// free the same pointer (a double-free). Neither should happen silently,
// so we don't allow it.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count), ptr_(nullptr) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        // Destructors must not throw. cudaFree can fail, but there's
        // nothing useful to do about it here -- free-and-ignore rather
        // than risk crashing the program during unwinding.
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : count_(other.count_), ptr_(other.ptr_) {
        other.count_ = 0;
        other.ptr_ = nullptr;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            count_ = other.count_;
            ptr_ = other.ptr_;
            other.count_ = 0;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    void copyFromHost(const T* hostSrc, std::size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, hostSrc, count * sizeof(T),
                               cudaMemcpyHostToDevice));
    }

    void copyToHost(T* hostDst, std::size_t count) const {
        CUDA_CHECK(cudaMemcpy(hostDst, ptr_, count * sizeof(T),
                               cudaMemcpyDeviceToHost));
    }

private:
    std::size_t count_;
    T* ptr_;
};

}  // namespace forgegemm