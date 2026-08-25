# Worked examples. Not tests and not benchmarks: they exist to be READ, and to
# prove the kernel set composes into something a real pipeline shape can use
# (T4.3b). They need OpenCV for the sensor stage and for image I/O.
if(NOT BINCV_OPENCV_FOUND)
    message(STATUS "Examples: skipping; they need OpenCV for the sensor stage")
    return()
endif()

file(GLOB EXAMPLE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(example_src ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    add_executable(${example_name} ${example_src})
    target_link_libraries(${example_name} PRIVATE bincv_core bincv_warnings)
endforeach()
