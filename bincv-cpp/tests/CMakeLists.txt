# ---------------------------------------------------------------------------
# Test backend (T1.7)
#
# Google Test is used wherever it can be obtained, EXCEPT in the configuration
# whose whole claim is that binCV needs nothing but a C++17 compiler: core-only,
# -fno-exceptions (ARCHITECTURE 2, Tier 2). There the built-in harness in
# tests/test_util.hpp stays the backend. Both run the same case bodies through
# the same check macros -- see the header for the full argument.
#
# NOT because GTest cannot compile there. TASKS.md T1.7 says "GTest needs
# exceptions"; measured, that is wrong -- googletest v1.14.0 builds from source
# under -fno-exceptions, runs, and reports failures correctly, EXPECT_DEATH
# included. `-DBINCV_USE_GTEST=ON` therefore works in that configuration too and
# is supported. The default declines it for a different reason: the embedded gate
# is the one this project can least afford to have go dark, and putting it behind
# a network fetch and a 30k-line desktop framework is how that happens.
# ---------------------------------------------------------------------------

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#if !defined(__cpp_exceptions) && !defined(__EXCEPTIONS) && !defined(_CPPUNWIND)
#error exceptions are disabled
#endif
int main() { return 0; }
" BINCV_EXCEPTIONS_AVAILABLE)

set(BINCV_GTEST_ENABLED FALSE)

if(BINCV_USE_GTEST STREQUAL "OFF")
    set(BINCV_TEST_BACKEND_DESC "built-in (opted out)")
elseif(BINCV_USE_GTEST STREQUAL "AUTO" AND NOT BINCV_EXCEPTIONS_AVAILABLE)
    set(BINCV_TEST_BACKEND_DESC "built-in (dependency-free)")
else()
    # FetchContent with FIND_PACKAGE_ARGS: an installed GTest satisfies the
    # dependency without a download, and only a machine that has neither reaches
    # the network. On CMake older than 3.24 that argument does not exist, so try
    # find_package first by hand.
    include(FetchContent)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
            GIT_SHALLOW    TRUE
            FIND_PACKAGE_ARGS NAMES GTest)
        FetchContent_MakeAvailable(googletest)
        set(BINCV_GTEST_ENABLED TRUE)
    else()
        find_package(GTest QUIET)
        if(GTest_FOUND)
            set(BINCV_GTEST_ENABLED TRUE)
        else()
            FetchContent_Declare(googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG        v1.14.0
                GIT_SHALLOW    TRUE)
            FetchContent_MakeAvailable(googletest)
            set(BINCV_GTEST_ENABLED TRUE)
        endif()
    endif()

    if(BINCV_GTEST_ENABLED)
        # A downloaded googletest is built as a normal target, so its headers are
        # not system headers and -Wconversion would report on them. Nothing in
        # binCV can act on those, and a gate that prints warnings nobody can fix
        # is a gate people learn to ignore.
        if(TARGET gtest)
            get_target_property(_bincv_gtest_inc gtest INTERFACE_INCLUDE_DIRECTORIES)
            if(_bincv_gtest_inc)
                set_target_properties(gtest PROPERTIES
                    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_bincv_gtest_inc}")
            endif()
        endif()
        set(BINCV_TEST_BACKEND_DESC "Google Test")
    endif()
endif()

if(NOT BINCV_GTEST_ENABLED AND BINCV_USE_GTEST STREQUAL "ON")
    message(FATAL_ERROR "BINCV_USE_GTEST=ON but Google Test could not be obtained")
endif()

# Reported in the top-level configuration summary, so a build says which backend
# it verified with rather than leaving it to be inferred.
set(BINCV_TEST_BACKEND_DESC "${BINCV_TEST_BACKEND_DESC}" PARENT_SCOPE)

# Everything a binCV test target needs, in one place: the library, the harness
# header, the warning policy, and the backend selection macro.
function(bincv_add_test_target TARGET)
    add_executable(${TARGET} ${ARGN})
    target_link_libraries(${TARGET} PRIVATE bincv_core bincv_warnings)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(BINCV_GTEST_ENABLED)
        target_compile_definitions(${TARGET} PRIVATE BINCV_TEST_WITH_GTEST=1)
        target_link_libraries(${TARGET} PRIVATE GTest::gtest)
    else()
        target_compile_definitions(${TARGET} PRIVATE BINCV_TEST_WITH_GTEST=0)
    endif()
endfunction()

# Core tests: must build and run without OpenCV.
set(BINCV_CORE_TESTS test_binMat test_quantMat test_storage test_error test_error_checked)

# Tests that require OpenCV interop.
set(BINCV_OPENCV_TESTS test_opencv_interop)

foreach(TEST_NAME ${BINCV_CORE_TESTS})
    bincv_add_test_target(${TEST_NAME} ${TEST_NAME}.cpp)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# ---------------------------------------------------------------------------
# The harness's own regression check
#
# T1.7 asks that a deliberately failing assertion still produce a non-zero exit
# code. Asking is not verifying, so it is a test: the same source is built twice,
# once ordinary and once with a case that fails on purpose, and the second is
# registered WILL_FAIL so ctest passes only when it exits non-zero. Both are
# built in every configuration, which is what checks BOTH backends.
# ---------------------------------------------------------------------------
bincv_add_test_target(test_harness test_harness.cpp)
add_test(NAME test_harness COMMAND test_harness)

bincv_add_test_target(test_harness_failing test_harness.cpp)
target_compile_definitions(test_harness_failing PRIVATE BINCV_HARNESS_EXPECT_FAILURE)
add_test(NAME test_harness_failing COMMAND test_harness_failing)
set_tests_properties(test_harness_failing PROPERTIES WILL_FAIL TRUE)

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
# Deliberately NOT ported to GTest's EXPECT_DEATH by T1.7. EXPECT_DEATH exists in
# every configuration binCV builds (verified, -fno-exceptions included), but it
# works by forking the test process and re-running it, which is materially more
# fragile under the QEMU user-mode emulation scripts/verify_arm.sh runs in than
# "start this process and check how it ended". These cases are the only coverage
# the no-exceptions build has for the validation checks, so they are the last
# thing that should acquire a new way to fail.
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
#
# These two binaries are driven by argv and must die rather than report, so they
# have no harness and are built directly rather than through
# bincv_add_test_target.
add_executable(test_error_abort test_error_abort.cpp)
target_link_libraries(test_error_abort PRIVATE bincv_core bincv_warnings)
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
bincv_add_death_test(test_error_abort quant-height-overflow
                     "height x N planes does not fit in an int")

# Plane indices (T1.5/T1.6). Validation rather than element access, so these are
# live in Release too -- which is the whole point: unchecked, plane(N) handed out
# a writable view of the next allocation, and magnitude(N) handed back the sign
# plane. One case per overload, mutable and const, on all three spellings of the
# accessor, because each overload carries its own copy of the check.
bincv_add_death_test(test_error_abort quant-plane-index
                     "QuantMat::plane: index out of range")
bincv_add_death_test(test_error_abort quant-plane-index-const
                     "QuantMat::plane: index out of range")
bincv_add_death_test(test_error_abort binmat-plane-index
                     "QuantMat<1>::plane: index out of range")
bincv_add_death_test(test_error_abort binmat-plane-index-const
                     "QuantMat<1>::plane: index out of range")
bincv_add_death_test(test_error_abort signed-magnitude-index
                     "SignedQuantMat::magnitude: index out of range")
bincv_add_death_test(test_error_abort signed-magnitude-index-const
                     "SignedQuantMat::magnitude: index out of range")
bincv_add_death_test(test_error_abort quant-wrap-short-buffer
                     "buffer is shorter than N x height x strideWords")
bincv_add_death_test(test_error_abort signed-wrap-short-buffer
                     "buffer is shorter than N x height x strideWords")
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
# so these run in all configurations rather than only in the Debug build. Since
# T1.8 there IS a Debug configuration in the verification set, which compiles the
# same assertions the ordinary way; these cases remain the coverage for the three
# Release configurations, where NDEBUG would otherwise delete them.
add_executable(test_assert_abort test_assert_abort.cpp)
target_link_libraries(test_assert_abort PRIVATE bincv_core bincv_warnings)
target_include_directories(test_assert_abort PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

bincv_add_death_test(test_assert_abort at-row "BinMat::at: index out of range")
bincv_add_death_test(test_assert_abort at-col "BinMat::at: index out of range")
bincv_add_death_test(test_assert_abort at-negative "BinMat::at: index out of range")
bincv_add_death_test(test_assert_abort set-row "BinMat::set: index out of range")
bincv_add_death_test(test_assert_abort set-col "BinMat::set: index out of range")
bincv_add_death_test(test_assert_abort quant-at-row "QuantMat::at: index out of range")
bincv_add_death_test(test_assert_abort quant-at-col "QuantMat::at: index out of range")
bincv_add_death_test(test_assert_abort quant-set-row "QuantMat::set: index out of range")
bincv_add_death_test(test_assert_abort quant-set-value
                     "QuantMat::set: value does not fit in N bits")
# The same precondition at N == 1, where BinMat::set's bool parameter used to
# swallow it: an out-of-range value narrowed to 1 instead of being reported.
bincv_add_death_test(test_assert_abort binmat-set-value
                     "QuantMat<1>::set: value does not fit in N bits")
bincv_add_death_test(test_assert_abort signed-set-value
                     "SignedQuantMat::set: value does not fit in N magnitude bits")
# INT_MIN specifically: the value whose magnitude used to be computed as `-value`,
# which is signed-overflow UB rather than merely a wrong number.
bincv_add_death_test(test_assert_abort signed-set-int-min
                     "SignedQuantMat::set: value does not fit in N magnitude bits")
bincv_add_death_test(test_assert_abort view-stride
                     "BinMatView: multi-row view needs a non-zero stride")
bincv_add_death_test(test_assert_abort const-view-stride
                     "BinMatConstView: multi-row view needs a non-zero stride")

if(BINCV_OPENCV_FOUND)
    foreach(TEST_NAME ${BINCV_OPENCV_TESTS})
        bincv_add_test_target(${TEST_NAME} ${TEST_NAME}.cpp)
        target_link_libraries(${TEST_NAME} PRIVATE bincv_cpp)
        # save_test_image writes to ${cwd}/tests/output
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endforeach()
endif()
