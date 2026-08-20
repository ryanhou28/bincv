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

# ---------------------------------------------------------------------------
# T3.9 / E-4 (X-21): the one MULTI-FILE benchmark here.
#
# genericn_benchmark compares three arms -- the generic-N route, the shipped
# specialization, and a hand-written binary-only reference with no binCV in it at
# all -- and each arm gets its OWN translation unit for two measured reasons,
# both spelled out in benchmark/genericn_arms.hpp:
#
#   * code size is half of that experiment's metric, and `size` on one object per
#     arm is a number per arm rather than a number for the sum;
#   * morphology_path_benchmark.cpp records, measured, that two instantiations of
#     one kernel in a single object move each other's timings by ~10% through code
#     layout alone.
#
# So the arms are carved out of the one-executable-per-source glob and built as an
# OBJECT library, whose per-arm .o files are exactly what `size` is run on.
# ---------------------------------------------------------------------------
set(BINCV_GENERICN_ARMS
    ${CMAKE_CURRENT_SOURCE_DIR}/genericn_arm_generic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genericn_arm_specialized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genericn_arm_handwritten.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genericn_diag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genericn_diag_accum.cpp)
list(REMOVE_ITEM BENCHMARK_SOURCES ${BINCV_GENERICN_ARMS})

add_library(genericn_arms OBJECT ${BINCV_GENERICN_ARMS})
target_link_libraries(genericn_arms PRIVATE bincv_core bincv_warnings)
target_include_directories(genericn_arms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

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

    add_executable(${benchmark_name} ${benchmark_src})
    target_link_libraries(${benchmark_name} PRIVATE bincv_core bincv_warnings)
    target_include_directories(${benchmark_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if(benchmark_name STREQUAL "genericn_benchmark")
        target_link_libraries(${benchmark_name} PRIVATE genericn_arms)
    endif()
endforeach()
