# ForgeGEMM

A CUDA/HIP general matrix multiplication (GEMM) library, built to go deep on
GPU performance fundamentals: memory coalescing, occupancy, shared-memory
tiling, and honest benchmarking against cuBLAS.

**Status: early development.** This is not a finished library yet. See
"What's actually done" below for exactly what exists right now versus what's
planned -- kept deliberately separate so this README stays accurate as the
project grows.

Inspired by and built on the general approach described in Simon Boehm's
["How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"](https://siboehm.com/articles/22/CUDA-MMM),
adapted here with a CUDA/HIP portability goal from the start.

## What's actually done

- [x] CMake project structure (CUDA optional -- host code builds without a GPU present)
- [x] `Matrix<T>`: owning, row-major, bounds-checked, deterministic seeded random init
- [x] `MatrixView<T>` / `ConstMatrixView<T>`: non-owning views, separate types from `Matrix<T>` by design
- [x] Host-side correctness tests for both (zero/identity/1x1/rectangular/bounds/view-writes-through/deterministic-random)
- [x] Matrix/MatrixView equality operators and MatrixView::fill, with tests
- [x] `DeviceBuffer<T>` RAII wrapper for GPU memory
- [x] CudaEvent RAII wrapper for GPU timing
- [x] GemmConfig struct describing a GEMM operation (M/N/K, alpha/beta, transpose flags)
- [ ] Naive CUDA SGEMM kernel
- [ ] Shared-memory tiled SGEMM kernel
- [ ] cuBLAS reference backend + correctness comparison
- [ ] Benchmark harness (warm-up, median/min timing, GFLOP/s)
- [ ] HIP/ROCm backend

## Building

Host-only (no GPU required):

```bash
mkdir build && cd build
cmake ..
cmake --build .
ctest
```

On a CUDA-capable machine, CUDA support is detected automatically at
configure time -- no separate flag needed. GPU kernels and benchmarks will
build once they exist (see status above).

## Design notes

- `Matrix<T>` is Rule-of-Zero: it holds a `std::vector<T>` internally, so
  copy/move/destroy are all compiler-generated and correct.
- `MatrixView<T>` exists as a separate type specifically so ownership is
  answered by the type system -- a function taking a `MatrixView<T>` cannot
  accidentally take ownership of data it shouldn't.
- CUDA is an optional language in the CMake build so the host-side library
  and tests can be built and checked anywhere, not just on a GPU machine.

## License

MIT
