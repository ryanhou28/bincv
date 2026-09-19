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

# The shared device vocabulary -- plane blocks, keypoint and descriptor sets,
# the compaction contract -- exercised FROM THE DEVICE. A CUDA translation unit
# because the failure these types exist to prevent is a kernel addressing the
# wrong plane or the wrong descriptor, which no host-side arithmetic can catch.
bincv_add_test_target(test_cuda_vocabulary test_cuda_vocabulary.cu)
target_link_libraries(test_cuda_vocabulary PRIVATE bincv_cuda)
target_include_directories(test_cuda_vocabulary PRIVATE ${BINCV_POLICY_ROOT}/tests)
set_target_properties(test_cuda_vocabulary PROPERTIES CUDA_STANDARD 17
                                                      CUDA_STANDARD_REQUIRED ON)

add_test(NAME cuda_device_vocabulary COMMAND test_cuda_vocabulary)
set_tests_properties(cuda_device_vocabulary PROPERTIES SKIP_RETURN_CODE 77)

# The BINCV_HOST_DEVICE helpers, called on both targets and compared. A CUDA
# translation unit because the claim is about what NVCC produces for a HOST
# header's function: the annotation buys one definition instead of two, and this
# is where "the two targets agree" stops being an argument about source text.
bincv_add_test_target(test_cuda_shared_helpers test_cuda_shared_helpers.cu)
target_link_libraries(test_cuda_shared_helpers PRIVATE bincv_cuda)
target_include_directories(test_cuda_shared_helpers PRIVATE ${BINCV_POLICY_ROOT}/tests)
set_target_properties(test_cuda_shared_helpers PROPERTIES CUDA_STANDARD 17
                                                          CUDA_STANDARD_REQUIRED ON)

add_test(NAME cuda_shared_helpers COMMAND test_cuda_shared_helpers)
set_tests_properties(cuda_shared_helpers PROPERTIES SKIP_RETURN_CODE 77)
