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
#
# test_logic (T2.2), test_shift (T2.3/T2.4) and test_reduce (T2.5/T2.6) are core
# tests even though half of each is the Tier 1 comparison against OpenCV --
# cv::bitwise_* for the first, cv::borderInterpolate and cv::copyMakeBorder for
# the second, cv::countNonZero for the third. That half sits behind
# BINCV_WITH_OPENCV inside the file, so the kernels are still compiled and checked
# against a per-pixel reference in the three configurations that have no OpenCV --
# including the Debug one, which is the only place their BINCV_ASSERT
# preconditions are live.
set(BINCV_CORE_TESTS test_binMat test_quantMat test_storage test_error test_error_checked
                     test_logic test_shift test_reduce)

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

# T2.2 kernel preconditions. A kernel never throws (ARCHITECTURE 5.3), so these
# are the only place the logic kernels' argument checking can be observed
# failing -- and the alias case is a defect no sanitizer can see, since every
# address involved is valid memory and only the answer is wrong.
bincv_add_death_test(test_assert_abort logic-dims
                     "a, b and dst must have the same dimensions")
bincv_add_death_test(test_assert_abort logic-dims-not
                     "src and dst must have the same dimensions")
bincv_add_death_test(test_assert_abort logic-alias
                     "dst must alias an input exactly or not overlap it")
bincv_add_death_test(test_assert_abort logic-short-stride
                     "every view's stride must cover a whole row")

# T2.3/T2.4 shift preconditions. The overlap case is the one that differs from
# the logic kernels and is the reason it gets its own case rather than being
# assumed covered by logic-alias: a shift is NOT pointwise in the word index, so
# the in-place spelling ops/logic.hpp supports is undefined here, and the two
# kernels therefore accept different sets of views. An unknown BorderType is the
# other: BorderType is an enum, so an int can reach these kernels as one, and
# without the check it would silently mean BORDER_CONSTANT.
bincv_add_death_test(test_assert_abort shift-dims
                     "shift: src and dst must have the same dimensions")
bincv_add_death_test(test_assert_abort shift-in-place
                     "dst must share no word with src")
bincv_add_death_test(test_assert_abort shift-overlap
                     "dst must share no word with src")
bincv_add_death_test(test_assert_abort shift-short-stride
                     "shift: every view's stride must cover a whole row")
bincv_add_death_test(test_assert_abort shift-border-type
                     "shift: unknown BorderType")

# T2.5/T2.6 reduction preconditions. There is deliberately NO alias case: a
# reduction reads only, so D-11 does not bind it and every overlap -- including
# countAndSplit(m, m, m, r), which is how a caller asks "how many pixels are set"
# through the split -- is legal. The two that remain are dimensions and stride,
# and the dimension check has one case per entry point because countAnd compares
# two views and countAndSplit three: a third view nobody compared would simply be
# read at the first two's geometry, inside its own buffer, with only the answer
# wrong.
bincv_add_death_test(test_assert_abort reduce-dims
                     "reduce: masked reductions need views of the same size")
bincv_add_death_test(test_assert_abort reduce-split-dims
                     "reduce: masked reductions need views of the same size")
bincv_add_death_test(test_assert_abort reduce-short-stride
                     "reduce: every view's stride must cover a whole row")
# impl::regionFromExtent's non-empty precondition. Unreachable through the public
# entry points -- both callers guard it -- so it is driven directly: an empty
# extent underflows x1 - 1 into a ~5.8e17-word row range with isEmpty == false.
bincv_add_death_test(test_assert_abort reduce-empty-extent
                     "reduce: regionFromExtent needs a non-empty extent")

if(BINCV_OPENCV_FOUND)
    foreach(TEST_NAME ${BINCV_OPENCV_TESTS})
        bincv_add_test_target(${TEST_NAME} ${TEST_NAME}.cpp)
        target_link_libraries(${TEST_NAME} PRIVATE bincv_cpp)
        # save_test_image writes to ${cwd}/tests/output
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endforeach()

    # -----------------------------------------------------------------------
    # Equivalence harness (T2.1)
    #
    # Needs cv::Mat, so it is an OpenCV test. It does no image I/O, so unlike the
    # suites above it does not link bincv_cpp and does not need a working
    # directory.
    # -----------------------------------------------------------------------
    bincv_add_test_target(test_equivalence test_equivalence.cpp)
    add_test(NAME test_equivalence COMMAND test_equivalence)

    # --- the harness's own regression check --------------------------------
    #
    # T2.1: "a harness that passes everything is worse than no harness, because
    # it licenses every kernel built on it." The same source is rebuilt once per
    # deliberate conversion fault and registered WILL_FAIL, so ctest passes only
    # when the harness REPORTS the fault. Same shape as test_harness_failing
    # above, and for the same reason: asking for a property is not verifying it.
    #
    # Measured, with each fault compiled in (checks passed, of 9922):
    #   1 column off-by-one       8405 -- PackingAnchor and Transpose go red
    #   2 dropped trailing word   7339 -- all three swept cases go red
    #   3 transposed row/column   6737 -- PackingAnchor goes red
    #   4 dirty padding           6346 -- every padding verdict goes red
    #
    # The gaps between the faults are the point. Fault 1 leaves countNonZero
    # GREEN, a cyclic column rotation being count-preserving; fault 3 leaves the
    # transpose case green, because a transposing conversion applied to both sides
    # of cv::transpose cancels exactly. Only the anchor sweep -- whose expectation
    # comes from a generator that never touches the conversion -- sees all three.
    # That is why tests/equivalence.hpp carries a second content generator.
    #
    # Fault 4 is the odd one out and the reason it was added: it corrupts the
    # GENERATOR's padding rather than the conversion, because no pixel comparison
    # can see a padding bit -- unpackTo8U never reads one. Before
    # expectPaddingClean() existed, a stand-in word-wise bitwiseNot with no
    # clearTrailingBits() passed 240 of 240 swept cases at uint64_t while leaving
    # 826,200 phantom set bits behind. Fault 4 is what keeps that check honest.
    #
    # Excluded from the check-count harvest by scripts/verify.sh, which skips
    # WILL_FAIL tests, so these need no row in tests/expected-checks.txt.
    function(bincv_add_faulted_equivalence_test SUFFIX FAULT_ID)
        set(FAULT_TARGET test_equivalence_fault_${SUFFIX})
        bincv_add_test_target(${FAULT_TARGET} test_equivalence.cpp)
        target_compile_definitions(${FAULT_TARGET} PRIVATE
                                   BINCV_EQUIVALENCE_INJECT_FAULT=${FAULT_ID})
        add_test(NAME ${FAULT_TARGET} COMMAND ${FAULT_TARGET})
        set_tests_properties(${FAULT_TARGET} PROPERTIES WILL_FAIL TRUE)
    endfunction()

    bincv_add_faulted_equivalence_test(column_off_by_one    1)
    bincv_add_faulted_equivalence_test(dropped_trailing_word 2)
    bincv_add_faulted_equivalence_test(transposed_row_col   3)
    bincv_add_faulted_equivalence_test(dirty_padding        4)

    # -----------------------------------------------------------------------
    # The same proof for the suite that CONSUMES the harness (T2.2)
    #
    # test_equivalence going red under a fault says the harness can detect one.
    # It says nothing about whether a kernel suite built on the harness can, and
    # MEASURED, tests/test_logic.cpp could not: built with fault 1 it passed
    # 56044 of 56044 checks and exited 0, and with fault 3 likewise. Its Tier 1
    # half fed OpenCV through toCvMask() and read binCV back through
    # expectBitExact(), so a fault in the conversion applied to both sides of a
    # pointwise operation and cancelled exactly -- which is the failure mode
    # equivalence.hpp property 2 predicts in writing.
    #
    # test_logic now builds its OpenCV inputs with randomCvMask() instead, so the
    # two sides no longer share a conversion, and these three targets are what
    # keep that true. Fault 4 is deliberately NOT among them: it dirties the
    # GENERATOR's padding, and a destination the kernels wrote is clean whatever
    # the sources' padding held -- that is the invariant, and the in-process
    # Logic.DirtySources_* sweep is what asserts it in all four configurations.
    # -----------------------------------------------------------------------
    function(bincv_add_faulted_logic_test SUFFIX FAULT_ID)
        set(FAULT_TARGET test_logic_fault_${SUFFIX})
        bincv_add_test_target(${FAULT_TARGET} test_logic.cpp)
        target_compile_definitions(${FAULT_TARGET} PRIVATE
                                   BINCV_EQUIVALENCE_INJECT_FAULT=${FAULT_ID})
        add_test(NAME ${FAULT_TARGET} COMMAND ${FAULT_TARGET})
        set_tests_properties(${FAULT_TARGET} PROPERTIES WILL_FAIL TRUE)
    endfunction()

    bincv_add_faulted_logic_test(column_off_by_one     1)
    bincv_add_faulted_logic_test(dropped_trailing_word 2)
    bincv_add_faulted_logic_test(transposed_row_col    3)
endif()
