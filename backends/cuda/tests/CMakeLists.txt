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

# The packer's three arms -- grid-stride, row grid and byte lane -- held to ONE
# answer, with both runtime switches toggled inside this one binary.
#
# A SEPARATE SUITE, not an addition to any existing one, because what it proves
# is different in kind: every other suite here runs whichever arm the switches
# default to, so two of the three are invisible to all of them. This is where a
# switch that quietly stopped switching, or an arm that ran outside the domain
# its gate claims to refuse, is observable. It also drives BOTH gates from the
# excluded side -- a 70,000-row image past `gridDim.y`, an odd-strided view, a
# uint16 source, a cutoff of 256 -- which are the four shapes a fast arm would
# corrupt and nothing else would notice.
#
# HOST C++ rather than a CUDA translation unit, for test_cuda_sensor's reason:
# nothing here needs a kernel of its own, and compiling it as host C++ also
# proves the arm switches are reachable from a consumer's own .cpp.
bincv_add_test_target(test_cuda_packfast test_cuda_packfast.cpp)
target_link_libraries(test_cuda_packfast PRIVATE bincv_cuda)
target_include_directories(test_cuda_packfast PRIVATE ${BINCV_POLICY_ROOT}/tests)

add_test(NAME cuda_packer_arms COMMAND test_cuda_packfast)
set_tests_properties(cuda_packer_arms PROPERTIES SKIP_RETURN_CODE 77)

# The ternary/N-bit derivative and the gradient covariance against their host
# twins. A CUDA translation unit because two of its claims can only be asked of
# a kernel: that nvcc's device pass gives the HOST's answer for the host's own
# impl::signedDifference -- both branches of its route selector, the ternary
# three-op spelling at N == 1 and the ripple-borrow subtract above it -- and for
# impl::combineBitSlicedPairs, which the covariance epilogue CALLS rather than
# forking. It also holds all three covariance batch arms and both derivative
# arms to one answer in ONE binary, which is where a switch that stopped
# switching becomes observable.
bincv_add_test_target(test_cuda_derivcov test_cuda_derivcov.cu)
target_link_libraries(test_cuda_derivcov PRIVATE bincv_cuda)
target_include_directories(test_cuda_derivcov PRIVATE ${BINCV_POLICY_ROOT}/tests)
set_target_properties(test_cuda_derivcov PROPERTIES CUDA_STANDARD 17
                                                    CUDA_STANDARD_REQUIRED ON)

add_test(NAME cuda_derivative_covariance COMMAND test_cuda_derivcov)
set_tests_properties(cuda_derivative_covariance PROPERTIES SKIP_RETURN_CODE 77)

# Keypoint orientation and BRIEF against their host twins. A CUDA translation
# unit because two of its claims can only be asked from a kernel.
# `bincv::briefAngleBin` now carries BINCV_HOST_DEVICE and the describe kernel
# CALLS it rather than restating it, so what is worth pinning is that nvcc's
# device pass selects the HOST's bin -- walked here in single-ULP steps either
# side of all 30 boundaries, because one ULP there replaces a whole 256-bit
# descriptor rather than nudging an angle. The other is the moment claim: the
# angle goes through atan2f on one side and std::atan2 on the other, so the
# suite pins the INTEGER moments exactly and measures the float's ULP
# divergence rather than asserting a bit-exactness no transcendental can give.
bincv_add_test_target(test_cuda_orientation_descriptor test_cuda_orientation_descriptor.cu)
target_link_libraries(test_cuda_orientation_descriptor PRIVATE bincv_cuda)
target_include_directories(test_cuda_orientation_descriptor PRIVATE
                           ${BINCV_POLICY_ROOT}/tests)
set_target_properties(test_cuda_orientation_descriptor PROPERTIES
                      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

add_test(NAME cuda_orientation_descriptor COMMAND test_cuda_orientation_descriptor)
set_tests_properties(cuda_orientation_descriptor PROPERTIES SKIP_RETURN_CODE 77)

# The frontend's entry point: FAST, the minimum-eigenvalue response,
# good-features selection and sub-pixel refinement, each against its host twin.
#
# HOST C++ RATHER THAN A CUDA TRANSLATION UNIT, and for this family that is
# load-bearing rather than a preference. ops/fast.hpp gates its AVX2 bit-plane
# kernels off under __CUDACC__ on purpose; compiled by nvcc the host arm of the
# comparison would silently become the portable one, and the suite would be
# holding the device to a kernel nobody runs. Everything the device side needs
# here is a host-callable launcher, and the BINCV_HOST_DEVICE helpers this
# family calls from device code -- impl::minEigenValue, fastShiftedWord,
# fastArcAny, fastLongestRun, boxHorizontal3, boxVertical3 -- are SHARED rather
# than forked, with test_cuda_shared_helpers.cu already pinning nvcc's device
# compilation of the response against the host's.
bincv_add_test_target(test_cuda_frontend_corner test_cuda_frontend_corner.cpp)
target_link_libraries(test_cuda_frontend_corner PRIVATE bincv_cuda)
target_include_directories(test_cuda_frontend_corner PRIVATE ${BINCV_POLICY_ROOT}/tests)

add_test(NAME cuda_frontend_corner COMMAND test_cuda_frontend_corner)
set_tests_properties(cuda_frontend_corner PROPERTIES SKIP_RETURN_CODE 77)

# The detector-to-keypoint-set link, and the padding contract the RESIDENT
# pipeline rests on. A .cpp for the reason the frontend-corner suite gives:
# everything here is a host-callable launcher, and a host header compiled by
# nvcc is not the host arm a caller runs.
#
# One case in this suite deliberately reaches into ANOTHER family's kernel --
# keypointOrientation -- because the assumption being tested is about that
# kernel: a launch sized by capacity must reject the slots this op zeroed, or
# a frame with no mid-pipeline synchronize is not available at all. That is an
# integration claim, and an integration claim asserted only in a comment is one
# nobody has checked.
bincv_add_test_target(test_cuda_keypoints test_cuda_keypoints.cpp)
target_link_libraries(test_cuda_keypoints PRIVATE bincv_cuda)
target_include_directories(test_cuda_keypoints PRIVATE ${BINCV_POLICY_ROOT}/tests)

add_test(NAME cuda_keypoints COMMAND test_cuda_keypoints)
set_tests_properties(cuda_keypoints PROPERTIES SKIP_RETURN_CODE 77)
