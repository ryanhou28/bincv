# Core tests: must build and run without OpenCV.
set(BINCV_CORE_TESTS test_binMat test_storage)

# Tests that require OpenCV interop.
set(BINCV_OPENCV_TESTS test_opencv_interop)

foreach(TEST_NAME ${BINCV_CORE_TESTS})
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE bincv_core)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

if(BINCV_OPENCV_FOUND)
    foreach(TEST_NAME ${BINCV_OPENCV_TESTS})
        add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
        target_link_libraries(${TEST_NAME} PRIVATE bincv_cpp)
        target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        # save_test_image writes to ${cwd}/tests/output
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endforeach()
endif()
