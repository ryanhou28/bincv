# The backend's benchmark arms: every device op priced at birth, next to the
# host library's arm on the same machine and frames.

add_executable(cuda_dense_benchmark cuda_dense_benchmark.cpp)
target_link_libraries(cuda_dense_benchmark PRIVATE bincv_core bincv_cuda bincv_warnings)
target_include_directories(cuda_dense_benchmark PRIVATE ${BINCV_POLICY_ROOT}/benchmark)

add_executable(cuda_foundation_benchmark cuda_foundation_benchmark.cpp)
target_link_libraries(cuda_foundation_benchmark PRIVATE bincv_core bincv_cuda bincv_warnings)
target_include_directories(cuda_foundation_benchmark PRIVATE ${BINCV_POLICY_ROOT}/benchmark)

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
            add_executable(cuda_stereobm_benchmark cuda_stereobm_benchmark.cpp)
            target_link_libraries(cuda_stereobm_benchmark PRIVATE
                bincv_core bincv_cuda bincv_warnings ${OpenCV_LIBS})
            target_include_directories(cuda_stereobm_benchmark SYSTEM PRIVATE
                ${OpenCV_INCLUDE_DIRS})
            message(STATUS "cuda_stereobm_benchmark: OpenCV ${OpenCV_VERSION} "
                           "from ${BINCV_CUDA_OPENCV_DIR}")
        else()
            message(WARNING "BINCV_CUDA_OPENCV_DIR set but no cudastereo-capable "
                            "OpenCV found there; cuda_stereobm_benchmark skipped.")
        endif()
    endif()
endif()
