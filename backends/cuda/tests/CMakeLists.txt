# Device-versus-host bit-exactness. Built whenever the backend is, run
# meaningfully only where a GPU is: the binaries exit 77 without one, and ctest
# reports the skip rather than a hollow pass.
bincv_add_test_target(test_cuda_backend test_cuda_backend.cpp)
target_link_libraries(test_cuda_backend PRIVATE bincv_cuda)
target_include_directories(test_cuda_backend PRIVATE ${BINCV_POLICY_ROOT}/tests)

add_test(NAME cuda_backend COMMAND test_cuda_backend)
set_tests_properties(cuda_backend PROPERTIES SKIP_RETURN_CODE 77)

# The custom-packer suite is itself a CUDA translation unit, because
# packCustom.cuh can only be instantiated by one -- so this target is also the
# standing proof that its documented contract is usable.
bincv_add_test_target(test_cuda_custom test_cuda_custom.cu)
target_link_libraries(test_cuda_custom PRIVATE bincv_cuda)
target_include_directories(test_cuda_custom PRIVATE ${BINCV_POLICY_ROOT}/tests)
set_target_properties(test_cuda_custom PROPERTIES CUDA_STANDARD 17
                                                  CUDA_STANDARD_REQUIRED ON)

add_test(NAME cuda_custom_packers COMMAND test_cuda_custom)
set_tests_properties(cuda_custom_packers PROPERTIES SKIP_RETURN_CODE 77)
