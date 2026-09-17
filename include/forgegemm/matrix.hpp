#pragma once

#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace forgegemm {

// Owning, row-major, host-side matrix. Rule-of-Zero: all resource management
// is delegated to std::vector<T>, so the compiler-generated copy/move/dtor
// are all correct and cheap to reason about.
template <typename T>
class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(checked_size(rows, cols)) {}

    static Matrix Zero(std::size_t rows, std::size_t cols) {
        return Matrix(rows, cols);
    }

    static Matrix Identity(std::size_t n) {
        Matrix m(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            m(i, i) = static_cast<T>(1);
        }
        return m;
    }

    // Deterministic seeded random init - reproducibility matters more than
    // "real" randomness for correctness tests and benchmark repeatability.
    static Matrix Random(std::size_t rows, std::size_t cols,
                          unsigned seed = 42,
                          T low = static_cast<T>(-1),
                          T high = static_cast<T>(1)) {
        Matrix m(rows, cols);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(static_cast<double>(low),
                                                      static_cast<double>(high));
        for (auto& v : m.data_) {
            v = static_cast<T>(dist(rng));
        }
        return m;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return data_.size(); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    T& operator()(std::size_t r, std::size_t c) {
        return data_[index(r, c)];
    }
    const T& operator()(std::size_t r, std::size_t c) const {
        return data_[index(r, c)];
    }

    T& at(std::size_t r, std::size_t c) {
        bounds_check(r, c);
        return data_[index(r, c)];
    }
    const T& at(std::size_t r, std::size_t c) const {
        bounds_check(r, c);
        return data_[index(r, c)];
    }

    // Element-wise equality. Mainly for tests: comparing a kernel's output
    // matrix against a CPU reference matrix.
    bool operator==(const Matrix& other) const {
        return rows_ == other.rows_ && cols_ == other.cols_ &&
               data_ == other.data_;
    }
    bool operator!=(const Matrix& other) const { return !(*this == other); }

private:
    std::size_t index(std::size_t r, std::size_t c) const {
        return r * cols_ + c;
    }

    void bounds_check(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
    }

    // Guards against rows * cols overflowing size_t on pathological inputs --
    // cheap to check, and the kind of thing that's easy to skip and only
    // matters once, badly.
    static std::size_t checked_size(std::size_t rows, std::size_t cols) {
        if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) {
            throw std::overflow_error("Matrix dimensions overflow allocation size");
        }
        return rows * cols;
    }

    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;
};

}  // namespace forgegemm
