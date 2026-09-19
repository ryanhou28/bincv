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

# Morphology: erode / dilate / morphologyEx against the host twin. A CUDA
# translation unit, because one case here cannot be written any other way --
# impl::borderIndex carries BINCV_HOST_DEVICE so the kernel calls the Tier 1
# border promise itself rather than a transcription of it, and proving that
# means calling it ON THE DEVICE and comparing.
bincv_add_test_target(test_cuda_morphology test_cuda_morphology.cu)
target_link_libraries(test_cuda_morphology PRIVATE bincv_cuda)
target_include_directories(test_cuda_morphology PRIVATE ${BINCV_POLICY_ROOT}/tests)
set_target_properties(test_cuda_morphology PROPERTIES CUDA_STANDARD 17
                                                      CUDA_STANDARD_REQUIRED ON)

add_test(NAME cuda_morphology COMMAND test_cuda_morphology)
set_tests_properties(cuda_morphology PROPERTIES SKIP_RETURN_CODE 77)

# The median family: denoiseMedian3 over packed bits and medianWide over a wide
# image, both against their host twins. A CUDA translation unit because two of
# its cases cannot be written any other way -- maj3 now carries
# BINCV_HOST_DEVICE and the binary kernel CALLS it, so what is worth pinning is
# that nvcc's device pass gives the host's answer for the host's own function,
# which only a kernel can ask.
bincv_add_test_target(test_cuda_median test_cuda_median.cu)
target_link_libraries(test_cuda_median PRIVATE bincv_cuda)
target_include_directories(test_cuda_median PRIVATE ${BINCV_POLICY_ROOT}/tests)
set_target_properties(test_cuda_median PROPERTIES CUDA_STANDARD 17
                                                  CUDA_STANDARD_REQUIRED ON)

add_test(NAME cuda_median COMMAND test_cuda_median)
set_tests_properties(cuda_median PROPERTIES SKIP_RETURN_CODE 77)

# The pyramid, the resident ladder and shift. A CUDA translation unit because
# four of its cases cannot be written any other way: the geometry twins and the
# extendedRowWord restatement are only worth anything if the DEVICE agrees with
# the host, and the requantization table compares what a kernel computes against
# the host's bit-sliced arithmetic on every reachable sum.
bincv_add_test_target(test_cuda_pyramid test_cuda_pyramid.cu)
target_link_libraries(test_cuda_pyramid PRIVATE bincv_cuda)
target_include_directories(test_cuda_pyramid PRIVATE ${BINCV_POLICY_ROOT}/tests)
set_target_properties(test_cuda_pyramid PROPERTIES CUDA_STANDARD 17
                                                   CUDA_STANDARD_REQUIRED ON)

add_test(NAME cuda_pyramid COMMAND test_cuda_pyramid)
set_tests_properties(cuda_pyramid PROPERTIES SKIP_RETURN_CODE 77)

# The sensor stage: threshold, binarize, edgeThreshold against their host twins.
#
# HOST C++ rather than a CUDA translation unit, which is the one place this suite
# differs from its neighbours. Nothing in it needs a kernel of its own: the three
# BINCV_HOST_DEVICE helpers this family calls from device code --
# impl::reflect101Edge, thresholdGE and impl::thresholdCutoff -- are SHARED, not
# forked, and test_cuda_shared_helpers.cu already pins nvcc's device compilation
# of all three against the host's. What is left is launcher-level, and compiling
# it as host C++ also proves the header-only cuda::threshold is usable from a
# consumer's own .cpp.
bincv_add_test_target(test_cuda_sensor test_cuda_sensor.cpp)
target_link_libraries(test_cuda_sensor PRIVATE bincv_cuda)
target_include_directories(test_cuda_sensor PRIVATE ${BINCV_POLICY_ROOT}/tests)

add_test(NAME cuda_sensor COMMAND test_cuda_sensor)
set_tests_properties(cuda_sensor PROPERTIES SKIP_RETURN_CODE 77)
