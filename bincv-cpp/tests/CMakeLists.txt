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
# test_logic (T2.2), test_shift (T2.3/T2.4), test_reduce (T2.5/T2.6) and
# test_bitslice (T2.7) are core tests even though part of each is a comparison
# against OpenCV -- cv::bitwise_* for the first, cv::borderInterpolate and
# cv::copyMakeBorder for the second, cv::countNonZero for the third, and the
# reference pipeline's cv::min/cv::max median for the fourth. That part sits
# behind BINCV_WITH_OPENCV inside the file, so the kernels are still compiled and
# checked against a per-pixel reference in the three configurations that have no
# OpenCV -- including the Debug one, which is the only place their BINCV_ASSERT
# preconditions are live.
#
# test_bitslice is core for a second reason: T2.7's word-level primitives have no
# OpenCV counterpart at all (bit-sliced arithmetic is Tier 3, ARCHITECTURE 5.1),
# so the exhaustive pattern enumeration that proves them is the ONLY thing
# standing behind maj3, bitSlicedSum and thresholdGE -- and it has to run in the
# configuration the embedded claim rests on.
#
# test_denoise (T3.1) and test_threshold (T3.2) are core for the same reason and
# with one addition. test_denoise's OpenCV half PORTS the reference pipeline's own
# cv::min/cv::max median, which is the correctness bar T3.1 is written against;
# its core half carries the per-pixel neighbourhood reference, the zero-border
# cases and the fused-equals-composed check, all of which have to run where the
# embedded claim does. test_threshold is split by TIER rather than by
# configuration: `binarize` (QuantMat<N> -> 1 bit) is Tier 3 and core, `threshold`
# (CV_8U -> 1 bit) is Tier 1 and needs cv::threshold, so half the file is behind
# the guard and half is not.
#
# test_morphology (T3.3) is core for the same reason and carries one thing the
# others do not: BORDER_WRAP. cv::morphologyEx REFUSES that border type -- it is
# an OpenCV assertion, not an omission -- so there is no Tier 1 denominator for
# it at all, and the only thing standing behind binCV's wrapped morphology is
# this file's core half (a per-pixel reference over an independently written
# border mapping, plus agreement with ops/shift.hpp for a one-cell element).
# That coverage would vanish if the suite were OpenCV-only.
#
# test_resample (E-8 / X-14) is core for the third reason and the strongest one:
# horizontal decimation has no OpenCV counterpart at all -- cv::resize resamples
# both axes on a byte image and rounds rather than keeping the even columns -- so
# there is no Tier 1 denominator, and this suite is the ONLY thing standing behind
# the primitive T3.4's pyramid is built on.
# test_pyramid (T3.4) is core for the same reason once more, and for a fourth
# that is specific to it: pyrDown is TIER 2 (ARCHITECTURE 5.1) -- cv::pyrDown's
# name and role with deliberately different numerics -- so there is no
# bit-exactness promise anywhere in it and no Tier 1 denominator at all. What
# stands behind it is a per-pixel reference, an enumeration of the requantizer's
# entire input space, and agreement with the EXPONENTIAL formulation T3.4
# replaced (impl::boxSum4Replicated, ops/bitslice.hpp's single-bit adder at
# k = 4*(2^NIn - 1)). None of those needs OpenCV, and all of them have to run
# where the embedded claim does -- including Debug, the only configuration in
# which pyrDown's BINCV_ASSERT preconditions are live. Its OpenCV half is one
# suite: binCV at NOut = 8 against the reference pipeline's own BOX_2x2 path, and
# the phase deviation that path's default anchor introduces.
# test_derivative (T3.5) is core for the same reasons and for one more. The
# binarized derivative is TIER 3 -- OpenCV has no operation with these semantics
# (see ops/derivative.hpp) -- so there is no bit-exactness promise to a cv:: call
# and no Tier 1 denominator. What stands behind it is a per-pixel reference over
# an INDEPENDENTLY WRITTEN border mapping, the canonical-zero sweep, the
# dirty-source-padding cases, and agreement between three formulations of one
# operation (the ternary spelling, the generic ripple, and the composed
# shift+logic route). None of those needs OpenCV, and all of them have to run
# where the embedded claim does -- including Debug, the only configuration in
# which the kernels' BINCV_ASSERT preconditions are live. The OpenCV half is the
# reference pipeline's own calcBinarizedDeriv, ported call for call, plus the two
# probes that pin cv::filter2D's correlation direction and default border --
# the two properties this operation would otherwise differ on silently.
# test_covariance (T3.6) is core for the strongest version of the same reason.
# The LK gradient covariance is TIER 3 -- cv:: has no operation with these
# semantics (cornerEigenValsAndVecs computes something else, from byte images,
# with a Sobel and a box filter) -- so there is no bit-exactness promise to a cv::
# call and no Tier 1 denominator anywhere in it. What stands behind it is a
# per-pixel FLOAT oracle written before the kernel, swept at every window position
# from a full window outside each edge to a full window past it, at W = 7, 15 and
# 31, over four independently built frames per word type. And what it is proving is
# not one operation's arithmetic but ARCHITECTURE 7.5 -- the claim that the whole
# 2x2 covariance reduces to masked population counts, which is the project's
# central technical claim. That has to be checked where the embedded claim runs:
# core-only, -fno-exceptions, and Debug, the only configuration in which the
# reduction's BINCV_ASSERT preconditions are live. It also counts `operator new`
# across the calls, which is what keeps D-15's axis-3 trade honest -- the
# four-argument selector form was taken over the 11-14% FASTER plane form purely
# because it needs no scratch, and an implementation that allocated would pass
# every value check in the file.
# test_corner (T3.7) is core, and for a reason none of the suites above has. The
# corner response is API TIER 2 -- `cv::goodFeaturesToTrack`'s role with the
# reference pipeline's BINARIZED derivatives -- so there is no bit-exactness
# promise to a cv:: call, and `cv::cornerMinEigenVal` works in float over float
# Sobel outputs, which means an exact comparison against it is not available even
# in principle. What stands behind it is a per-pixel reference written before the
# kernel, two exact INTEGER properties of the eigenvalue that need no square root
# (zero response iff singular matrix; an exact half-integer wherever the
# discriminant is a perfect square), a literal port of gftt.cpp's selection
# including the dilate buffer and the cell grid, and a hand-built map that pins
# the ORDER of NMS and the spacing filter. It also carries the check that D-19's
# reasoning holds: BORDER_REFLECT_101 must give no border ring and BORDER_CONSTANT
# must give one. None of that needs OpenCV, all of it has to run where the
# embedded claim does -- including Debug, the only configuration in which the
# kernels' BINCV_ASSERT preconditions are live -- and it counts `operator new`,
# because replacing the reference's unbounded candidate vector and its grid with
# the caller's own array is the whole no-heap claim.
# test_opticalflow (T3.8) is core, and it is the suite with the least excuse to
# be anywhere else. The tracker is API TIER 2 -- cv::calcOpticalFlowPyrLK's role
# over 1-bit content, with the previous window anchored on the integer grid --
# so there is no cv:: call to be exact against and no Tier 1 denominator. What
# stands behind it is GROUND TRUTH rather than another estimator: synthetic warps
# of a CONTINUOUS field, binarized after the warp, so every point's displacement
# is known analytically and the tolerance is stated with its derivation before any
# error is measured. Alongside that sits an independent per-pixel FLOAT
# implementation of the same algorithm -- the multiply-and-accumulate formulation
# the popcount residual identity is asserted to replace -- compared both on a
# single residual at 864 window positions per word type and on a whole tracked
# frame. None of that needs OpenCV, and all of it has to run where the embedded
# claim does: core-only, -fno-exceptions, and Debug, the only configuration in
# which the tracker's BINCV_ASSERT preconditions are live. It also counts
# `operator new` -- the tracker takes no scratch buffer AT ALL, where the
# reference copies the warped patch and its derivative into a winSize.area()*3
# AutoBuffer, so an implementation that allocated would pass every value check in
# the file. And it carries the frontend FOOTPRINT table, which is the number
# Phase 4 needs; it is a test rather than a benchmark because the thing it pins --
# which stage dominates -- is a structural property that must fail loudly when it
# changes, not a timing that is allowed to drift.
set(BINCV_CORE_TESTS test_binMat test_quantMat test_storage test_error test_error_checked
                     test_logic test_shift test_reduce test_bitslice
                     test_denoise test_threshold test_morphology test_resample
                     test_pyramid test_derivative test_covariance test_corner
                     test_opticalflow test_pack test_edge)

# Tests that require OpenCV interop.
set(BINCV_OPENCV_TESTS test_opencv_interop)

foreach(TEST_NAME ${BINCV_CORE_TESTS})
    bincv_add_test_target(${TEST_NAME} ${TEST_NAME}.cpp)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# ---------------------------------------------------------------------------
# A translation unit that MUST NOT COMPILE (T3.10 triage)
#
# The plane-ARRAY overload of gradientCovariance() is templated on the plane count
# and shipped without a bound on it, so a hand-assembled 33-plane view array reached
# `int64_t(1) << (i + j)` with a shift exponent of 64 -- undefined behaviour,
# reproduced under -fsanitize=undefined before the fix. The fix is a static_assert
# bounding N at QuantMat's 1..8, matching pyrDownRoute and derivativeX.
#
# A static_assert inside a function body is invisible to SFINAE, so the detection
# traits test_covariance.cpp uses for its other dispatch cases cannot test this one.
# What can is a target that is EXCLUDE_FROM_ALL -- an ordinary build never touches
# it -- and a ctest case whose command BUILDS it, registered WILL_FAIL. ctest passes
# only when the compile fails. Remove the static_assert and the file compiles, links
# and runs, `cmake --build` exits 0, and the case goes red.
#
# Excluded from the check-count harvest twice over: verify.sh skips WILL_FAIL tests
# and skips tests whose command is not a single executable. No expected-checks row.
# ---------------------------------------------------------------------------
bincv_add_test_target(test_covariance_n_bound test_covariance_n_bound.cpp)
set_target_properties(test_covariance_n_bound PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_test(NAME test_covariance_n_bound
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                  --target test_covariance_n_bound)
set_tests_properties(test_covariance_n_bound PROPERTIES WILL_FAIL TRUE)

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
# T2.11 added a FOURTH view (c1) and two more entry points. Each carries its own
# copy of the dimension assert, so each is its own deletion -- and a deletion no
# in-process suite can see: measured, removing c1's half from BOTH four-argument
# overloads left all four configurations green with byte-identical CHECKS
# (Debug test_reduce 594184/594184 before and after). countCovariance gets its own
# case for the same reason countAnd and countAndSplit do: a third view nobody
# compared is read at the first two's geometry, off the end of its buffer.
bincv_add_death_test(test_assert_abort reduce-xor-split-dims
                     "reduce: masked reductions need views of the same size")
bincv_add_death_test(test_assert_abort reduce-covariance-dims
                     "reduce: masked reductions need views of the same size")
bincv_add_death_test(test_assert_abort reduce-covariance-xor-dims
                     "reduce: masked reductions need views of the same size")
bincv_add_death_test(test_assert_abort reduce-short-stride
                     "reduce: every view's stride must cover a whole row")
# SlidingWindowCount checks the same stride precondition in a CONSTRUCTOR, which
# is a site no call-shaped case reaches.
bincv_add_death_test(test_assert_abort reduce-sliding-short-stride
                     "reduce: every view's stride must cover a whole row")
# impl::regionFromExtent's non-empty precondition. Unreachable through the public
# entry points -- both callers guard it -- so it is driven directly: an empty
# extent underflows x1 - 1 into a ~5.8e17-word row range with isEmpty == false.
bincv_add_death_test(test_assert_abort reduce-empty-extent
                     "reduce: regionFromExtent needs a non-empty extent")

# T2.7 bit-sliced preconditions. `bitslice-dims` is not a duplicate of logic-dims:
# majority3 compares THREE sources, and a dimension check that forgot the third
# left all 348061 of test_bitslice's checks green in every configuration when it
# was deleted. `bitslice-sum-overlap` is the only precondition in the project
# stated over raw pointers rather than views -- bitSlicedSum accumulates in its
# destination planes, so an outPlanes array overlapping inputs corrupts inputs the
# ripple has not read yet, with every address valid and only the answer wrong.
bincv_add_death_test(test_assert_abort bitslice-dims
                     "majority3: a, b, c and dst must have the same dimensions")
bincv_add_death_test(test_assert_abort bitslice-alias
                     "majority3: dst must alias an input exactly or not overlap it")
bincv_add_death_test(test_assert_abort bitslice-short-stride
                     "majority3: every view's stride must cover a whole row")
bincv_add_death_test(test_assert_abort bitslice-sum-overlap
                     "bitSlicedSum: outPlanes must not overlap inputs")

# T3.1 denoise preconditions. `denoise-in-place` is the one that is NOT covered by
# any case above, and it is the reason denoiseMedian3 takes viewsShareNoWord
# rather than destinationAliasIsSafe: destination row y reads SOURCE ROW y - 1, so
# the in-place spelling ops/logic.hpp supports would feed the filter its own
# output from the second row onwards -- every address valid, only the image wrong.
# The reference implementation has the same property and answers it the same way
# (it assigns `img = median_img` at the end rather than filtering in place).
bincv_add_death_test(test_assert_abort denoise-dims
                     "denoiseMedian3: src and dst must have the same dimensions")
bincv_add_death_test(test_assert_abort denoise-in-place
                     "denoiseMedian3: dst must share no word with src")
bincv_add_death_test(test_assert_abort denoise-short-stride
                     "denoiseMedian3: every view's stride must cover a whole row")

# T3.2 binarize preconditions. `binarize-dims` is a PER-PLANE check -- the loop
# compares every one of the N planes against dst, so a check that only looked at
# plane 0 would read planes 1..N-1 at plane 0's geometry, off the end of a shorter
# buffer, and an N-bit source arrives as N separate views under D-5.
# `binarize-alias` is the exact-alias case ops/logic.hpp permits and this kernel
# does not: plane p is read at word i AFTER dst's word i has been written for the
# lower planes, so writing into a source plane corrupts values still to be read.
bincv_add_death_test(test_assert_abort binarize-dims
                     "binarize: every source plane must have dst's dimensions")
bincv_add_death_test(test_assert_abort binarize-alias
                     "binarize: dst must share no word with any source plane")
bincv_add_death_test(test_assert_abort binarize-short-stride
                     "binarize: every view's stride must cover a whole row")

# T3.3 morphology preconditions. TWO OF THESE HAVE NO ANALOGUE ANYWHERE ELSE IN
# THE PROJECT, which is why the list is longer than denoise's or binarize's:
#
#   `morphex-scratch-size` / `morphex-scratch-overlap` are the D-16 SCRATCH
#   CONTRACT, and morphologyEx is the only kernel in binCV that takes a
#   caller-provided intermediate. An undersized scratch is not a wrong answer but
#   an overrun of the caller's array -- OPEN erodes into every word of scratch and
#   then reads every word back -- and a scratch aliasing dst makes the second step
#   read what the first wrote.
#
#   `morph-element-invalid` is the only check anywhere that validates an
#   argument's own internal consistency rather than a relationship between views,
#   and StructuringElement::valid() has NO other caller: without this case it can
#   be deleted with every configuration still green, and an element with no set
#   cell yields a uniform frame -- the fold's identity everywhere -- which reads
#   as a plausible image rather than as a fault.
#
# Measured before these existed: neutering every BINCV_ASSERT in ops/morphology.hpp
# at once (`BINCV_ASSERT(true || ...)`) left tests/test_morphology.cpp reporting
# 172632/172632 checks passed with assertions live. Preconditions with no death
# test are not covered by a correctness suite that never violates them.
bincv_add_death_test(test_assert_abort morph-dims
                     "erode: src and dst must have the same dimensions")
bincv_add_death_test(test_assert_abort morph-short-stride
                     "erode: every view's stride must cover a whole row")
bincv_add_death_test(test_assert_abort morph-in-place
                     "erode: dst must share no word with src")
bincv_add_death_test(test_assert_abort morph-border-type
                     "dilate: unknown BorderType")
bincv_add_death_test(test_assert_abort morph-element-invalid
                     "erode: the structuring element needs positive extents")
bincv_add_death_test(test_assert_abort morphex-op
                     "morphologyEx: unknown MorphOp")
bincv_add_death_test(test_assert_abort morphex-scratch-size
                     "morphologyEx: this op needs a caller-provided scratch view of src's size")
bincv_add_death_test(test_assert_abort morphex-scratch-overlap
                     "morphologyEx: scratch must share no word with src or dst")

# T3.5 derivative preconditions. TWO OF THESE ARE A CLASS NO OTHER KERNEL HAS,
# and it is the class T3.5 was set to discover: the derivative is the first
# operation with a MULTI-PLANE destination, so "no aliasing" stops meaning
# src-versus-dst alone and starts meaning DESTINATION-VERSUS-DESTINATION as well.
#
#   `derivative-sign-alias` is the one that matters most. A sign plane sharing
#   words with a magnitude plane compiles, runs, and leaves pixels at magnitude 0
#   with the sign bit SET -- SignedQuantMat's canonical-zero rule broken, which
#   corrupts T3.6's sumXY while leaving sumXX and sumYY exactly right. Every
#   magnitude-only test passes over it.
#   `derivative-mag-alias` is the N > 1 half of the same class: the ripple writes
#   magnitude plane p and reads it back in the conditional negate.
#
#   `derivative-border-type` is driven through derivativeY so the shared
#   impl::checkDerivativeArgs is entered from both public kernels, the same way
#   `morph-border-type` above uses dilate where its neighbours use erode.
#
# Measured before these existed: replacing all FOURTEEN conditions inside
# impl::checkDerivativeArgs with a literal `true` left tests/test_derivative.cpp
# reporting a byte-identical 47593/47593 checks passed in the Debug core-only
# build -- the only configuration in which BINCV_ASSERT is live -- and left
# ./scripts/verify.sh ALL CONFIGURATIONS GREEN. The entire precondition block was
# deletable. That is the same measurement the morphology block above records, and
# the same conclusion: preconditions with no death test are not covered by a
# correctness suite that never violates them.
bincv_add_death_test(test_assert_abort derivative-dims
                     "derivative: every magnitude plane must have the source's dimensions")
bincv_add_death_test(test_assert_abort derivative-short-stride
                     "derivative: a source plane's stride is shorter than one of its own rows")
bincv_add_death_test(test_assert_abort derivative-in-place
                     "derivative: no destination plane may share a word with a source plane")
bincv_add_death_test(test_assert_abort derivative-sign-alias
                     "derivative: the sign plane must not share a word with a magnitude plane")
bincv_add_death_test(test_assert_abort derivative-mag-alias
                     "derivative: the magnitude planes must not share a word with each other")
bincv_add_death_test(test_assert_abort derivative-border-type
                     "derivative: unknown BorderType")

# The tracker's destination-aliasing predicate (T3.8). Its destination is an array
# of points rather than a view, which is why it fell outside the D-11 pattern every
# other kernel follows and why it had no check at all until a review found it: an
# in-place call returned status 1 for every point and answers up to 23 px wrong, in
# the Debug build, with no diagnostic.
bincv_add_death_test(test_assert_abort flow-alias
                     "opticalFlow: nextPts must not overlap prevPts")

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
