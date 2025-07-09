file(GLOB BENCHMARK_SOURCES CONFIGURE_DEPENDS *.cpp)

# TODO: Consider moving bench_util.hpp to a common include directory rather than in /benchmark
foreach(benchmark_src ${BENCHMARK_SOURCES})
    get_filename_component(benchmark_name ${benchmark_src} NAME_WE)

    add_executable(${benchmark_name} ${benchmark_src})
    target_link_libraries(${benchmark_name} PRIVATE bincv_cpp ${OpenCV_LIBS})
    target_include_directories(${benchmark_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endforeach()
