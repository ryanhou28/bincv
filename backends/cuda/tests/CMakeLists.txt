# Device-versus-host bit-exactness. Built whenever the backend is, run
# meaningfully only where a GPU is: the binary exits 77 without one, and ctest
# reports the skip rather than a hollow pass.
bincv_add_test_target(test_cuda_backend test_cuda_backend.cpp)
target_link_libraries(test_cuda_backend PRIVATE bincv_cuda)
target_include_directories(test_cuda_backend PRIVATE ${BINCV_POLICY_ROOT}/tests)

add_test(NAME cuda_backend COMMAND test_cuda_backend)
set_tests_properties(cuda_backend PROPERTIES SKIP_RETURN_CODE 77)
