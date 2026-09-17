// Minimal dependency-free test harness -- no gtest yet. Good enough for
// Day 2 correctness checks; worth swapping for Catch2/gtest once the test
// matrix grows past a handful of cases (see Day 9 of the sprint plan).

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "forgegemm/matrix.hpp"
#include "forgegemm/matrix_view.hpp"
#include "forgegemm/gemm_config.hpp"

using forgegemm::Matrix;
using forgegemm::MatrixView;

static int g_failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__,    \
                          __LINE__);                                       \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

static void test_zero_matrix() {
    Matrix<double> m = Matrix<double>::Zero(3, 4);
    CHECK(m.rows() == 3);
    CHECK(m.cols() == 4);
    for (std::size_t r = 0; r < m.rows(); ++r) {
        for (std::size_t c = 0; c < m.cols(); ++c) {
            CHECK(m(r, c) == 0.0);
        }
    }
}

static void test_identity_matrix() {
    Matrix<double> m = Matrix<double>::Identity(4);
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            CHECK(m(r, c) == (r == c ? 1.0 : 0.0));
        }
    }
}

static void test_1x1_matrix() {
    Matrix<double> m(1, 1);
    m(0, 0) = 7.0;
    CHECK(m.rows() == 1);
    CHECK(m.cols() == 1);
    CHECK(m(0, 0) == 7.0);
}

static void test_rectangular_matrix() {
    Matrix<double> m(2, 5);
    CHECK(m.rows() == 2);
    CHECK(m.cols() == 5);
    CHECK(m.size() == 10);
    m(1, 4) = 3.5;
    CHECK(m(1, 4) == 3.5);
}

static void test_bounds_checking() {
    Matrix<double> m(2, 2);
    bool threw = false;
    try {
        m.at(2, 0);  // out of range on purpose
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_matrix_view_reads_through() {
    Matrix<double> m(2, 2);
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(1, 0) = 3.0;
    m(1, 1) = 4.0;

    MatrixView<double> view(m);
    CHECK(view(0, 0) == 1.0);
    CHECK(view(1, 1) == 4.0);

    // View writes through to the original -- it does not own or copy data.
    view(0, 0) = 99.0;
    CHECK(m(0, 0) == 99.0);
}

static void test_random_matrix_is_deterministic() {
    Matrix<double> a = Matrix<double>::Random(4, 4, /*seed=*/7);
    Matrix<double> b = Matrix<double>::Random(4, 4, /*seed=*/7);
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            CHECK(a(r, c) == b(r, c));
        }
    }
}

static void test_gemm_config_defaults() {
    forgegemm::GemmConfig cfg;
    CHECK(cfg.M == 0 && cfg.N == 0 && cfg.K == 0);
    CHECK(cfg.alpha == 1.0 && cfg.beta == 0.0);
    CHECK(!cfg.transposeA && !cfg.transposeB);
}

static void test_gemm_config_explicit_values() {
    forgegemm::GemmConfig cfg;
    cfg.M = 128; cfg.N = 64; cfg.K = 32;
    cfg.alpha = 2.0; cfg.beta = 0.5;
    cfg.transposeA = true;
    CHECK(cfg.M == 128 && cfg.N == 64 && cfg.K == 32);
    CHECK(cfg.alpha == 2.0 && cfg.beta == 0.5);
    CHECK(cfg.transposeA && !cfg.transposeB);
}

int main() {
    test_zero_matrix();
    test_identity_matrix();
    test_1x1_matrix();
    test_rectangular_matrix();
    test_bounds_checking();
    test_matrix_view_reads_through();
    test_random_matrix_is_deterministic();

    if (g_failures == 0) {
        std::printf("All Matrix/MatrixView tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d test(s) failed.\n", g_failures);
    return 1;
}
