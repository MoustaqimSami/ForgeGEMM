# No CUDA kernels exist yet. This file exists so `add_subdirectory(benchmarks)`
# doesn't fail once a CUDA compiler is found -- the naive SGEMM kernel and its
# benchmark target get added here once they're implemented and correctness-
# tested (naive kernel: Day 3-4; benchmark harness: Day 4 of the sprint plan).
message(STATUS "ForgeGEMM benchmarks: no kernels implemented yet.")
