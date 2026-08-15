# Benchmarks compare binCV against OpenCV, so they are only configured when
# OpenCV is available (guarded by the caller in the top-level CMakeLists.txt).

file(GLOB BENCHMARK_SOURCES CONFIGURE_DEPENDS *.cpp)

# TODO: Consider moving bench_util.hpp to a common include directory rather than in /benchmark
foreach(benchmark_src ${BENCHMARK_SOURCES})
    get_filename_component(benchmark_name ${benchmark_src} NAME_WE)

    add_executable(${benchmark_name} ${benchmark_src})
    target_link_libraries(${benchmark_name} PRIVATE bincv_core)
    target_include_directories(${benchmark_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
