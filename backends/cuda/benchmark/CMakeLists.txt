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

# ---------------------------------------------------------------------------
# The window family: morphology's arms, its role comparison, and its memory.
#
# ONE executable, not two. The GPU-vs-GPU role comparison needs an OpenCV built
# with cudafilters, which no packaged OpenCV ships -- but a second, OpenCV-only
# target would be one the CUDA gate tries to build on every machine that has
# not built one. So the OpenCV arms live behind BINCV_CUDA_HAVE_OPENCV inside
# this single always-built binary, and the target simply gains a definition and
# some libraries when such an OpenCV is pointed at.
# ---------------------------------------------------------------------------
add_executable(cuda_window_benchmark cuda_window_benchmark.cpp cuda_bench_null.cu)
target_link_libraries(cuda_window_benchmark PRIVATE bincv_core bincv_cuda bincv_warnings)
target_include_directories(cuda_window_benchmark PRIVATE ${BINCV_POLICY_ROOT}/benchmark)
set_target_properties(cuda_window_benchmark PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

if(BINCV_CUDA_OPENCV_DIR AND NOT BINCV_OPENCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core imgproc cudafilters cudaarithm
                 PATHS ${BINCV_CUDA_OPENCV_DIR} NO_DEFAULT_PATH)
    if(OpenCV_FOUND AND TARGET opencv_cudafilters)
        target_compile_definitions(cuda_window_benchmark PRIVATE BINCV_CUDA_HAVE_OPENCV=1)
        target_link_libraries(cuda_window_benchmark PRIVATE
            opencv_core opencv_imgproc opencv_cudafilters opencv_cudaarithm)
        target_include_directories(cuda_window_benchmark SYSTEM PRIVATE
            ${OpenCV_INCLUDE_DIRS})
        message(STATUS "cuda_window_benchmark: role comparison against "
                       "cv::cuda::createMorphologyFilter enabled "
                       "(OpenCV ${OpenCV_VERSION} from ${BINCV_CUDA_OPENCV_DIR})")
    else()
        message(WARNING "BINCV_CUDA_OPENCV_DIR set but no cudafilters-capable OpenCV "
                        "found there; cuda_window_benchmark keeps its binCV arms and "
                        "prints the role comparison as UNMEASURED.")
    endif()
endif()

# ---------------------------------------------------------------------------
# The median family's arms, its bar and its memory.
#
# PLAIN C++, NOT A CUDA TRANSLATION UNIT, and that is load-bearing: this file
# times the HOST library's arms next to the device's, and ops/medianWide.hpp
# gates its AVX2 kernel off under __CUDACC__ on purpose. Compiled by nvcc the
# host rows would silently become the portable arm and the comparison would be
# against a kernel nobody runs. Only cuda_bench_null.cu needs nvcc here.
# ---------------------------------------------------------------------------
add_executable(cuda_median_benchmark cuda_median_benchmark.cpp cuda_bench_null.cu)
target_link_libraries(cuda_median_benchmark PRIVATE bincv_core bincv_cuda bincv_warnings)
target_include_directories(cuda_median_benchmark PRIVATE ${BINCV_POLICY_ROOT}/benchmark)
set_target_properties(cuda_median_benchmark PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# The pyramid family: pyrDownBox's two arms, the resident ladder's footprint,
# the resident-pipeline stand-in, and shift against the byte side's DMA.
#
# ONE executable, for the reason the window family gives above: a second,
# OpenCV-only target is one the CUDA gate tries to build on every machine that
# has not built a cudawarping-capable OpenCV. The cv::cuda::resize and
# cv::cuda::pyrDown role arms therefore live behind BINCV_CUDA_BENCH_OPENCV
# inside this always-built binary, and the binary says so in its own output
# when they are not compiled in -- an unmeasured role bar is reported as
# OUTSTANDING rather than replaced with a substitute.
# ---------------------------------------------------------------------------
add_executable(cuda_pyramid_benchmark cuda_pyramid_benchmark.cpp cuda_bench_null.cu)
target_link_libraries(cuda_pyramid_benchmark PRIVATE bincv_core bincv_cuda bincv_warnings)
target_include_directories(cuda_pyramid_benchmark PRIVATE ${BINCV_POLICY_ROOT}/benchmark)
set_target_properties(cuda_pyramid_benchmark PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

if(BINCV_CUDA_OPENCV_DIR AND NOT BINCV_OPENCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core cudawarping
                 PATHS ${BINCV_CUDA_OPENCV_DIR} NO_DEFAULT_PATH)
    if(OpenCV_FOUND AND TARGET opencv_cudawarping)
        target_compile_definitions(cuda_pyramid_benchmark PRIVATE
            BINCV_CUDA_BENCH_OPENCV=1)
        target_link_libraries(cuda_pyramid_benchmark PRIVATE
            opencv_core opencv_cudawarping)
        target_include_directories(cuda_pyramid_benchmark SYSTEM PRIVATE
            ${OpenCV_INCLUDE_DIRS})
        message(STATUS "cuda_pyramid_benchmark: role comparison against "
                       "cv::cuda::resize(INTER_AREA) and cv::cuda::pyrDown enabled "
                       "(OpenCV ${OpenCV_VERSION} from ${BINCV_CUDA_OPENCV_DIR})")
    else()
        message(WARNING "BINCV_CUDA_OPENCV_DIR set but no cudawarping-capable OpenCV "
                        "found there; cuda_pyramid_benchmark keeps its binCV arms and "
                        "reports the GPU role comparison as OUTSTANDING.")
    endif()
endif()

# ---------------------------------------------------------------------------
# The sensor stage: threshold, binarize, edgeThreshold -- both edge arms through
# their runtime switch, two gate-excluded controls that must read ~1.00x, and
# the memory formulas beside every time.
#
# ONE executable, always built, with the cv::cuda role arms behind
# BINCV_CUDA_SENSOR_OPENCV inside it. A second, OpenCV-only target would be one
# the CUDA gate derives from this file's text and then tries to build on every
# machine that has not built a cuda-module OpenCV. When the role arms are not
# compiled in the binary says so and reports threshold and edgeThreshold as
# BLOCKED, which is what this project's both-axes ship rule requires -- a
# missing role comparison is not a memory argument that carries them anyway.
#
# Its own find_package with its own component list, deliberately NOT an
# extension of the cudastereo one above: adding components to that call would
# make OpenCV_FOUND false on any build carrying cudastereo but not these three,
# and cuda_stereobm_benchmark -- which is where the binary-vs-cv::cuda::StereoBM
# headline is reproduced -- would vanish behind a warning.
# ---------------------------------------------------------------------------
add_executable(cuda_sensor_benchmark cuda_sensor_benchmark.cpp cuda_bench_null.cu)
target_link_libraries(cuda_sensor_benchmark PRIVATE bincv_core bincv_cuda bincv_warnings)
target_include_directories(cuda_sensor_benchmark PRIVATE ${BINCV_POLICY_ROOT}/benchmark)
set_target_properties(cuda_sensor_benchmark PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

if(BINCV_CUDA_OPENCV_DIR AND NOT BINCV_OPENCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core imgproc cudaarithm cudafilters cudaimgproc
                 PATHS ${BINCV_CUDA_OPENCV_DIR} NO_DEFAULT_PATH)
    if(OpenCV_FOUND AND TARGET opencv_cudaarithm AND TARGET opencv_cudafilters
                    AND TARGET opencv_cudaimgproc)
        target_compile_definitions(cuda_sensor_benchmark PRIVATE
            BINCV_CUDA_SENSOR_OPENCV=1)
        target_link_libraries(cuda_sensor_benchmark PRIVATE
            opencv_core opencv_imgproc opencv_cudaarithm opencv_cudafilters
            opencv_cudaimgproc)
        target_include_directories(cuda_sensor_benchmark SYSTEM PRIVATE
            ${OpenCV_INCLUDE_DIRS})
        message(STATUS "cuda_sensor_benchmark: role comparison against "
                       "cv::cuda::threshold, the composed cv::cuda edge chain and "
                       "cv::cuda::CannyEdgeDetector enabled "
                       "(OpenCV ${OpenCV_VERSION} from ${BINCV_CUDA_OPENCV_DIR})")
    else()
        message(WARNING "BINCV_CUDA_OPENCV_DIR set but no OpenCV with cudaarithm + "
                        "cudafilters + cudaimgproc found there; cuda_sensor_benchmark "
                        "keeps its binCV arms and reports threshold and edgeThreshold "
                        "as BLOCKED on their role comparison.")
    endif()
endif()

# ---------------------------------------------------------------------------
# THE ROLE COMPARISONS, all of them, in one binary.
#
# The family benchmarks each carry their own cv::cuda arms, and each was
# written against its own find_package at its own moment. That gives four
# answers to "how does binCV compare to OpenCV on this GPU", taken in four
# processes, with the two sides' memory read on DIFFERENT meters. This target
# makes the cross-library claim once: one process, one launch floor, every pair
# interleaved, and cudaMemGetInfo on BOTH sides of every memory figure.
#
# ALWAYS BUILT, with the OpenCV arms behind BINCV_CUDA_ROLE_OPENCV inside it --
# the same shape cuda_window_benchmark and cuda_sensor_benchmark use, and for
# the same reason those two give: scripts/verify_cuda.sh derives its benchmark
# list from this file's TEXT and hand-excludes exactly one name
# (`grep -v cuda_stereobm_benchmark`), so a target that exists only when an
# OpenCV is pointed at is one the gate tries to build on every machine that has
# not built one. That is not hypothetical -- this target was written guarded
# first and turned the gate red with
# "No rule to make target 'cuda_role_benchmark'".
#
# Every row in this binary IS a comparison against OpenCV, so without one it
# has no arms left to run; what it does instead is print each verdict as
# BLOCKED (role bar unmeasured) or OUTSTANDING (no counterpart exists, ruling
# R2), which is what the both-axes ship rule requires it to say.
#
# Its own find_package with its own component list, appended after the others
# and linking IMPORTED TARGETS rather than ${OpenCV_LIBS}: adding components to
# an earlier call would make OpenCV_FOUND false for every target that call
# feeds, and the cudastereo guard above is where the binary-vs-StereoBM
# headline is reproduced.
# ---------------------------------------------------------------------------
add_executable(cuda_role_benchmark cuda_role_benchmark.cpp cuda_bench_null.cu)
target_link_libraries(cuda_role_benchmark PRIVATE bincv_core bincv_cuda bincv_warnings)
target_include_directories(cuda_role_benchmark PRIVATE ${BINCV_POLICY_ROOT}/benchmark)
set_target_properties(cuda_role_benchmark PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

if(BINCV_CUDA_OPENCV_DIR AND NOT BINCV_OPENCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core imgproc cudaarithm cudafilters cudawarping
                 cudastereo PATHS ${BINCV_CUDA_OPENCV_DIR} NO_DEFAULT_PATH)
    if(OpenCV_FOUND AND TARGET opencv_cudaarithm AND TARGET opencv_cudafilters
                    AND TARGET opencv_cudawarping AND TARGET opencv_cudastereo)
        target_compile_definitions(cuda_role_benchmark PRIVATE BINCV_CUDA_ROLE_OPENCV=1)
        target_link_libraries(cuda_role_benchmark PRIVATE
            opencv_core opencv_imgproc opencv_cudaarithm opencv_cudafilters
            opencv_cudawarping opencv_cudastereo)
        target_include_directories(cuda_role_benchmark SYSTEM PRIVATE
            ${OpenCV_INCLUDE_DIRS})
        message(STATUS "cuda_role_benchmark: every cv::cuda role bar in one binary "
                       "(OpenCV ${OpenCV_VERSION} from ${BINCV_CUDA_OPENCV_DIR})")
    else()
        message(WARNING "BINCV_CUDA_OPENCV_DIR set but no OpenCV with cudaarithm + "
                        "cudafilters + cudawarping + cudastereo found there; "
                        "cuda_role_benchmark keeps building and reports every role "
                        "bar as UNMEASURED.")
    endif()
endif()
