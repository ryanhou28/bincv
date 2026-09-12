# Worked examples. Not tests and not benchmarks: they exist to be READ, and to
# prove the kernel set composes into something a real pipeline shape can use
# (T4.3b).
#
# NOT gated on OpenCV. The core-only build is the configuration the whole memory
# argument rests on, and this directory used to be skipped there -- so the one
# thing a reader could RUN to check the embedded claim needed the dependency the
# claim is about not needing. vio_frontend now builds in every configuration: its
# OpenCV pieces (cv::imread over a directory, the cv:: sensor arm) sit behind
# BINCV_WITH_OPENCV inside the one file, and the core-only build reads frames
# from a sequence blob (scripts/make_sequence_blob.py) and runs binCV's own
# sensor stage. An example that cannot compile core-clean with its OpenCV half
# guarded off does not belong in this directory.
#
# Freestanding targets are the one real exclusion. The examples are HOSTED
# programs -- they fopen a blob or a directory, allocate, and read a clock --
# so on a bare-metal configure (CMAKE_SYSTEM_NAME Generic, the Cortex-M7
# toolchain file) they cannot link: newlib with no OS has no syscalls under
# fopen. The library they demonstrate still compiles there; that is what
# scripts/verify_cortex_m.sh checks, suite by suite.
if(CMAKE_SYSTEM_NAME STREQUAL "Generic")
    message(STATUS "Examples: skipping; hosted programs (files, heap, clock) on a freestanding target")
    return()
endif()

# Examples that REQUIRE OpenCV, named explicitly rather than detected from an
# include scan: vio_frontend also mentions opencv2 headers, but behind a guard,
# so the benchmark directory's include-grep partition would misclassify it. An
# example belongs on this list only when its OpenCV use is structural -- here,
# slam_frontend's frame source and sensor stage. It should come OFF the list the
# way vio_frontend did: by growing a blob-fed core arm, not by weakening the gate.
set(BINCV_EXAMPLES_NEED_OPENCV slam_frontend)

file(GLOB EXAMPLE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(example_src ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    if(NOT BINCV_OPENCV_FOUND AND example_name IN_LIST BINCV_EXAMPLES_NEED_OPENCV)
        message(STATUS "Examples: skipping ${example_name}; it needs OpenCV")
        continue()
    endif()
    add_executable(${example_name} ${example_src})
    target_link_libraries(${example_name} PRIVATE bincv_core bincv_warnings)
endforeach()
