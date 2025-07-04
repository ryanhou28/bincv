# Get all test source files
file(GLOB TEST_SOURCES "*.cpp")

foreach(TEST_SRC ${TEST_SOURCES})
    # Extract filename without extension
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} bincv_cpp ${OpenCV_LIBS})
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
endforeach()
