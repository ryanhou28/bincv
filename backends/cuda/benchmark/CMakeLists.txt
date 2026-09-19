# The backend's benchmark arms: every device op priced at birth, next to the
# host library's arm on the same machine and frames.

# cuda_bench_null.cu carries the launch floor's empty kernel and nothing else.
# It is compiled into every benchmark rather than into bincv_cuda: an empty
# kernel is measurement apparatus, and the shipped library has no business
# carrying one.
add_executable(cuda_dense_benchmark cuda_dense_benchmark.cpp cuda_bench_null.cu)
target_link_libraries(cuda_dense_benchmark PRIVATE bincv_core bincv_cuda bincv_warnings)
target_include_directories(cuda_dense_benchmark PRIVATE ${BINCV_POLICY_ROOT}/benchmark)
set_target_properties(cuda_dense_benchmark PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_foundation_benchmark cuda_foundation_benchmark.cpp cuda_bench_null.cu)
target_link_libraries(cuda_foundation_benchmark PRIVATE bincv_core bincv_cuda bincv_warnings)
target_include_directories(cuda_foundation_benchmark PRIVATE ${BINCV_POLICY_ROOT}/benchmark)
set_target_properties(cuda_foundation_benchmark PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# The GPU-vs-GPU role comparison needs an OpenCV built WITH cudastereo, which
# no packaged OpenCV ships. Point BINCV_CUDA_OPENCV_DIR at such a build's
# install prefix. Refused when the main build already linked a different
# OpenCV: two OpenCV builds in one process is an ODR hazard, not a benchmark.
set(BINCV_CUDA_OPENCV_DIR "" CACHE PATH
    "Install prefix of an OpenCV built with the cudastereo module")
if(BINCV_CUDA_OPENCV_DIR)
    if(BINCV_OPENCV_FOUND)
        message(WARNING "cuda_stereobm_benchmark skipped: the build already links "
                        "another OpenCV (configure with -DBINCV_USE_OPENCV=OFF).")
    else()
        find_package(OpenCV QUIET COMPONENTS core calib3d cudastereo
                     PATHS ${BINCV_CUDA_OPENCV_DIR} NO_DEFAULT_PATH)
        if(OpenCV_FOUND)
            add_executable(cuda_stereobm_benchmark cuda_stereobm_benchmark.cpp
                                                   cuda_bench_null.cu)
            target_link_libraries(cuda_stereobm_benchmark PRIVATE
                bincv_core bincv_cuda bincv_warnings ${OpenCV_LIBS})
            target_include_directories(cuda_stereobm_benchmark SYSTEM PRIVATE
                ${OpenCV_INCLUDE_DIRS})
            set_target_properties(cuda_stereobm_benchmark PROPERTIES
                CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
            message(STATUS "cuda_stereobm_benchmark: OpenCV ${OpenCV_VERSION} "
                           "from ${BINCV_CUDA_OPENCV_DIR}")
        else()
            message(WARNING "BINCV_CUDA_OPENCV_DIR set but no cudastereo-capable "
                            "OpenCV found there; cuda_stereobm_benchmark skipped.")
        endif()
    endif()
endif()
