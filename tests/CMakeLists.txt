add_executable(forgegemm_tests test_matrix.cpp)
target_link_libraries(forgegemm_tests PRIVATE forgegemm_core)

add_test(NAME MatrixTests COMMAND forgegemm_tests)
