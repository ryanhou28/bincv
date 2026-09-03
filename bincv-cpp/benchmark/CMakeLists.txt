# Two kinds of benchmark live here, and only one of them needs OpenCV.
#
#   * binCV-versus-OpenCV comparisons (ARCHITECTURE 10.3's denominator) include
#     bench_util.hpp, which pulls in <opencv2/opencv.hpp>. Those need OpenCV
#     present and are skipped without it.
#   * The Phase 2 experiment benchmarks (E-1 / E-2 / E-3) compare binCV against
#     binCV. No OpenCV denominator applies to "which of our own alternatives is
#     faster", so they include measure_util.hpp and nothing external.
#
# The whole directory used to be gated on OpenCV, which meant the reference
# device's DEFAULT build -- core-only, which is the configuration the embedded
# claim rests on -- could not produce the three binaries that close E-1, E-2 and
# E-3 at all. A failed reproduction then looks like a missing binary rather than a
# misconfiguration. Partitioning by what a file actually includes fixes that and
# keeps working as benchmarks are added.

file(GLOB BENCHMARK_SOURCES CONFIGURE_DEPENDS *.cpp)

# heap_probe.cpp is a component, not a benchmark: it DEFINES malloc and friends so
# the executable's definitions preempt libc's for every dynamically linked caller,
# OpenCV included. It is linked into the benchmarks that measure heap footprint,
# and must not become a target of its own.
set(BINCV_HEAP_PROBE ${CMAKE_CURRENT_SOURCE_DIR}/heap_probe.cpp)
list(REMOVE_ITEM BENCHMARK_SOURCES ${BINCV_HEAP_PROBE})

# X-32: residualSums' tap extraction, one TU per arm (see residual_arms.hpp).

# X-31: the corner response's three arms, one TU each (see cornerresp_arms.hpp).

set(BINCV_BENCH_CORE "")
set(BINCV_BENCH_OPENCV "")
foreach(benchmark_src ${BENCHMARK_SOURCES})
    file(STRINGS ${benchmark_src} _bincv_needs_opencv
         REGEX "^[ \t]*#[ \t]*include.*(bench_util\\.hpp|opencv2/)")
    if(_bincv_needs_opencv)
        list(APPEND BINCV_BENCH_OPENCV ${benchmark_src})
    else()
        list(APPEND BINCV_BENCH_CORE ${benchmark_src})
    endif()
endforeach()

set(BINCV_BENCH_BUILD ${BINCV_BENCH_CORE})
if(BINCV_OPENCV_FOUND)
    list(APPEND BINCV_BENCH_BUILD ${BINCV_BENCH_OPENCV})
else()
    list(LENGTH BINCV_BENCH_OPENCV _bincv_skipped)
    message(STATUS "Benchmarks: skipping ${_bincv_skipped} OpenCV-comparison benchmark(s); "
                   "the binCV-versus-binCV experiment benchmarks still build")
endif()

# TODO: Consider moving bench_util.hpp to a common include directory rather than in /benchmark
foreach(benchmark_src ${BINCV_BENCH_BUILD})
    get_filename_component(benchmark_name ${benchmark_src} NAME_WE)

    # Benchmarks that report a heap figure link the allocator-level probe; a
    # replaced `operator new` cannot see cv::fastMalloc and under-reports OpenCV
    # by an order of magnitude (see heap_probe.hpp).
    file(STRINGS ${benchmark_src} _bincv_needs_heap_probe REGEX "heap_probe\\.hpp")
    if(_bincv_needs_heap_probe)
        add_executable(${benchmark_name} ${benchmark_src} ${BINCV_HEAP_PROBE})
    else()
        add_executable(${benchmark_name} ${benchmark_src})
    endif()
    target_link_libraries(${benchmark_name} PRIVATE bincv_core bincv_warnings)
    # The stack probe runs each trial on a thread inside a forked child, so it needs
    # Threads. Nothing else here does, and bincv_core deliberately does not.
    if(benchmark_name STREQUAL "essential_stack_benchmark")
        find_package(Threads REQUIRED)
        target_link_libraries(${benchmark_name} PRIVATE Threads::Threads)
    endif()
    target_include_directories(${benchmark_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    # The real binarized frame, for benchmarks whose arms differ in CONVERGENCE
    # behavior and therefore cannot use synthetic content (X-50).
    target_compile_definitions(${benchmark_name} PRIVATE
        BINCV_REALFRAME_PATH="${CMAKE_CURRENT_SOURCE_DIR}/realframe.bin")

endforeach()
