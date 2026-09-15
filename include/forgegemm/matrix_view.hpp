#pragma once

#include <cstddef>
#include <stdexcept>

#include "forgegemm/matrix.hpp"

namespace forgegemm {

// Non-owning view into row-major matrix data. Does not manage lifetime --
// the caller is responsible for ensuring the viewed data outlives the view.
// Exists as a distinct type from Matrix<T> specifically so "who owns this
// memory" is answered by the type system, not by convention.
template <typename T>
class MatrixView {
public:
    MatrixView(T* data, std::size_t rows, std::size_t cols)
        : data_(data), rows_(rows), cols_(cols) {}

    // Views the full contents of an existing Matrix<T> without copying.
    explicit MatrixView(Matrix<T>& m)
        : data_(m.data()), rows_(m.rows()), cols_(m.cols()) {}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return rows_ * cols_; }

    T* data() { return data_; }
    const T* data() const { return data_; }

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

    // Sets every element in the view to the same value. Writes through to
    // whatever the view is looking at -- there is no separate storage here.
    void fill(T value) {
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < cols_; ++c) {
                (*this)(r, c) = value;
            }
        }
    }

    bool operator==(const MatrixView& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) return false;
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < cols_; ++c) {
                if ((*this)(r, c) != other(r, c)) return false;
            }
        }
        return true;
    }
    bool operator!=(const MatrixView& other) const { return !(*this == other); }

private:
    std::size_t index(std::size_t r, std::size_t c) const {
        return r * cols_ + c;
    }

    void bounds_check(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("MatrixView index out of range");
        }
    }

    T* data_;
    std::size_t rows_;
    std::size_t cols_;
};

// const-view convenience overload.
template <typename T>
class ConstMatrixView {
public:
    ConstMatrixView(const T* data, std::size_t rows, std::size_t cols)
        : data_(data), rows_(rows), cols_(cols) {}

    explicit ConstMatrixView(const Matrix<T>& m)
        : data_(m.data()), rows_(m.rows()), cols_(m.cols()) {}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    const T& operator()(std::size_t r, std::size_t c) const {
        return data_[r * cols_ + c];
    }

private:
    const T* data_;
    std::size_t rows_;
    std::size_t cols_;
};

}  // namespace forgegemm
