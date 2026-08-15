# Core tests: must build and run without OpenCV.
set(BINCV_CORE_TESTS test_binMat test_storage test_error test_error_checked)

# Tests that require OpenCV interop.
set(BINCV_OPENCV_TESTS test_opencv_interop)

foreach(TEST_NAME ${BINCV_CORE_TESTS})
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE bincv_core)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# ---------------------------------------------------------------------------
# Death tests
#
# A failed binCV check terminates the process -- by throwing where exceptions
# exist, by abort() where they do not -- so no assertion inside the process can
# observe it. These binaries run as their own processes instead, one ctest test
# per case, driven by expect_fatal.cmake, which requires that the child died
# abnormally AND that it named the reason.
#
# The message is the same in both configurations, which is what lets one case
# cover the exceptions and the no-exceptions builds: with exceptions the runtime
# prints what() while terminating, without them BINCV_THROW prints it itself.
#
# EXPECT_OUTPUT is a CMake regex, so the strings below avoid regex metacharacters
# rather than escaping them -- each is a distinctive literal substring of the
# diagnostic, not the whole line.
# ---------------------------------------------------------------------------

function(bincv_add_death_test TARGET CASE_NAME EXPECTED)
    add_test(NAME ${TARGET}.${CASE_NAME}
             COMMAND ${CMAKE_COMMAND}
             -DCHILD=$<TARGET_FILE:${TARGET}>
             -DCHILD_ARGS=${CASE_NAME}
             "-DEXPECT_OUTPUT=${EXPECTED}"
             -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_fatal.cmake)
endfunction()

# --- BINCV_THROW: one case per validation site in the library ---------------
#
# Not one representative case: without exceptions BINCV_CHECK_THROWS cannot
# evaluate its expression, so the in-process suites skip every validation check
# in that configuration. These are what make the -fno-exceptions build able to
# fail when a check is deleted. See tests/test_error_abort.cpp.
add_executable(test_error_abort test_error_abort.cpp)
target_link_libraries(test_error_abort PRIVATE bincv_core)
target_include_directories(test_error_abort PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

bincv_add_death_test(test_error_abort alignment-power-of-two
                     "rowAlignment must be a positive power of two")
bincv_add_death_test(test_error_abort alignment-word-multiple
                     "rowAlignment must be a multiple of the word size")
bincv_add_death_test(test_error_abort negative-dims
                     "BinMat dimensions must be non-negative")
bincv_add_death_test(test_error_abort wrap-negative-dims
                     "BinMat dimensions must be non-negative")
bincv_add_death_test(test_error_abort wrap-short-stride
                     "strideWords must be at least")
bincv_add_death_test(test_error_abort wrap-null-pointer
                     "cannot wrap a null pointer as a non-empty matrix")
bincv_add_death_test(test_error_abort resize-negative
                     "BinMat dimensions must be non-negative")
bincv_add_death_test(test_error_abort pad-negative
                     "Padding values must be non-negative")
bincv_add_death_test(test_error_abort for-each-on-empty
                     "cannot iterate over non-zero pixels")
bincv_add_death_test(test_error_abort sparsity-on-empty
                     "Sparsity is undefined for empty BinMat")

if(BINCV_OPENCV_FOUND)
    bincv_add_death_test(test_error_abort cvmat-empty
                         "Input cv::Mat is empty")
    bincv_add_death_test(test_error_abort cvmat-wrong-type
                         "Input cv::Mat must be of type CV_8UC1")
endif()

# --- BINCV_ASSERT: the debug-checked preconditions --------------------------
#
# This binary forces BINCV_DEBUG_CHECKS on from inside the source (#undef NDEBUG),
# so these run in all three configurations rather than only in a Debug build that
# no verification command configures. It is the replacement for the four
# out-of-range at()/set() assertions T1.4 deleted from the in-process suites.
add_executable(test_assert_abort test_assert_abort.cpp)
target_link_libraries(test_assert_abort PRIVATE bincv_core)
target_include_directories(test_assert_abort PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

bincv_add_death_test(test_assert_abort at-row "BinMat::at: index out of range")
bincv_add_death_test(test_assert_abort at-col "BinMat::at: index out of range")
bincv_add_death_test(test_assert_abort at-negative "BinMat::at: index out of range")
bincv_add_death_test(test_assert_abort set-row "BinMat::set: index out of range")
bincv_add_death_test(test_assert_abort set-col "BinMat::set: index out of range")
bincv_add_death_test(test_assert_abort view-stride
                     "BinMatView: multi-row view needs a non-zero stride")
bincv_add_death_test(test_assert_abort const-view-stride
                     "BinMatConstView: multi-row view needs a non-zero stride")

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
