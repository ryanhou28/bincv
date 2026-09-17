# The CUDA backend: one definition of the format, an independent kernel tree
# (ARCHITECTURE, backends). Compiled as a static library because kernels are
# .cu translation units; the headers under include/ are the public surface and
# every type in them is device-typed, so no call can hide where its memory
# lives.

# sm_86 is the backend's reference GPU (RTX 3070 Ti, the machine the committed
# numbers come from). Override for other parts -- 87 covers Jetson Orin with no
# source change. Must be chosen before enable_language(CUDA).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(bincv_cuda STATIC
    src/logic.cu
    src/reduce.cu
    src/pack.cu
    src/census.cu
    src/denseDisparity.cu)

set_target_properties(bincv_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

target_include_directories(bincv_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# The core headers by path rather than through bincv_core: the backend's own
# translation units compile without BINCV_WITH_OPENCV (nothing here touches
# interop), and a consumer already links bincv_core for the host half.
target_include_directories(bincv_cuda PRIVATE ${BINCV_POLICY_ROOT}/include)
# Consumers reach cudaMemcpy and friends from the inline transfer header.
target_link_libraries(bincv_cuda PUBLIC CUDA::cudart)
target_link_libraries(bincv_cuda PRIVATE bincv_warnings)

if(BINCV_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(BINCV_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
