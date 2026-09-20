# Worked examples for the CUDA backend. Not tests and not benchmarks: they
# exist to be READ, and to prove the device kernel set composes into a pipeline
# shape a caller would actually write -- which for a GPU backend is the whole
# argument, because a per-kernel ratio says nothing about whether the frame ever
# had to leave the device.
#
# PLAIN C++, NOT A CUDA TRANSLATION UNIT, and that is the point rather than a
# convenience. A caller of this backend writes .cpp, links bincv_cuda, and never
# opens nvcc; an example compiled by nvcc would quietly be allowed to write a
# kernel, and a reader could not tell which side of the line they were on. It
# would also silently swap the HOST library's arms for their __CUDACC__
# fallbacks -- ops/medianWide.hpp and ops/fast.hpp both gate their AVX2 kernels
# off under nvcc -- so the host comparison arm would be against code nobody
# runs. Same reason cuda_median_benchmark gives.
#
# NOT GATED ON OpenCV, for the reason the top-level examples/ directory gives:
# the configuration the memory argument rests on is the one without it. Frames
# come from a BSQ1 sequence blob (scripts/make_sequence_blob.py), which is how a
# core-only build runs real dataset frames at all.
#
# Freestanding targets are excluded as they are upstairs: these are HOSTED
# programs, and a CUDA backend on a target with no OS is not a configuration
# this repository has.
if(CMAKE_SYSTEM_NAME STREQUAL "Generic")
    message(STATUS "CUDA examples: skipping; hosted programs on a freestanding target")
    return()
endif()

file(GLOB BINCV_CUDA_EXAMPLE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(example_src ${BINCV_CUDA_EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    add_executable(${example_name} ${example_src})
    target_link_libraries(${example_name} PRIVATE bincv_core bincv_cuda bincv_warnings)
endforeach()
