cmake_minimum_required(VERSION 3.15)
project(bincv_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release; an unoptimized build makes benchmark numbers meaningless.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BINCV_USE_OPENCV "Enable OpenCV interop and OpenCV-based tests/benchmarks" ON)
option(BINCV_BUILD_TESTS "Build tests" ON)
option(BINCV_BUILD_BENCHMARKS "Build benchmarks (requires OpenCV)" ON)

# AUTO uses Google Test wherever it can be obtained AND the configuration is not
# the dependency-free one; see tests/CMakeLists.txt for why that exception exists.
set(BINCV_USE_GTEST "AUTO" CACHE STRING "Test backend: AUTO, ON (require GTest), OFF (built-in harness)")
set_property(CACHE BINCV_USE_GTEST PROPERTY STRINGS AUTO ON OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(BincvWarnings)

# ---------------------------------------------------------------------------
# Gate self-checks (T1.8 follow-up)
#
# Both are OFF in every ordinary build and both make the build FAIL when on.
# That is the point: scripts/verify.sh turns each on in a throwaway build tree
# and requires the failure, so "warnings are fatal" and "an unwired target cannot
# slip through" are observed rather than assumed. Before these existed, both
# claims were true only of the code that happened to be in the tree.
# ---------------------------------------------------------------------------
option(BINCV_SELFTEST_WARNING "Gate self-check: build a wired target that warns (build MUST fail under -DBINCV_WERROR=ON)" OFF)
option(BINCV_SELFTEST_UNWIRED "Gate self-check: build a target that omits bincv_warnings (configure MUST fail)" OFF)

if(BINCV_SELFTEST_WARNING)
    add_executable(bincv_selftest_warning ${CMAKE_CURRENT_SOURCE_DIR}/tests/selftest_warning.cpp)
    target_link_libraries(bincv_selftest_warning PRIVATE bincv_warnings)
endif()

if(BINCV_SELFTEST_UNWIRED)
    # The missing `bincv_warnings` is the whole point -- do not "fix" it.
    add_executable(bincv_selftest_unwired ${CMAKE_CURRENT_SOURCE_DIR}/tests/selftest_warning.cpp)
    target_link_libraries(bincv_selftest_unwired PRIVATE bincv_core)
endif()

# ---------------------------------------------------------------------------
# Core library: header-only, zero dependencies. This is what embedded targets use.
# ---------------------------------------------------------------------------
add_library(bincv_core INTERFACE)
target_include_directories(bincv_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(bincv_core INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
# Optional OpenCV interop
# ---------------------------------------------------------------------------
set(BINCV_OPENCV_FOUND FALSE)
if(BINCV_USE_OPENCV)
    find_package(OpenCV QUIET)
    if(OpenCV_FOUND)
        set(BINCV_OPENCV_FOUND TRUE)
        target_compile_definitions(bincv_core INTERFACE BINCV_WITH_OPENCV)
        # SYSTEM, so -Wconversion describes binCV rather than OpenCV. Measured:
        # a plain -I put ~107 warnings from <opencv2/...> into every interop
        # translation unit, none of them about first-party code, which would have
        # made the warning gate unreadable and therefore ignorable.
        target_include_directories(bincv_core SYSTEM INTERFACE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(bincv_core INTERFACE ${OpenCV_LIBS})
    endif()
endif()

# ---------------------------------------------------------------------------
# SIMD / platform feature detection
# ---------------------------------------------------------------------------
include(CheckCXXCompilerFlag)
set(BINCV_SIMD "scalar")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM|AARCH64")
    check_cxx_compiler_flag("-mfpu=neon" HAVE_NEON_FLAG)
    if(HAVE_NEON_FLAG OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64")
        target_compile_definitions(bincv_core INTERFACE BINCV_HAVE_NEON)
        set(BINCV_SIMD "NEON")
    endif()
else()
    check_cxx_compiler_flag("-mavx512f" HAVE_AVX512)
    check_cxx_compiler_flag("-mavx2" HAVE_AVX2)
    # Detection only: the flags are applied per-kernel once runtime dispatch lands
    # (ROADMAP 2.3), so the default build stays portable across machines.
    if(HAVE_AVX512)
        target_compile_definitions(bincv_core INTERFACE BINCV_HAVE_AVX512)
        set(BINCV_SIMD "AVX-512 (compiler-capable)")
    elseif(HAVE_AVX2)
        target_compile_definitions(bincv_core INTERFACE BINCV_HAVE_AVX2)
        set(BINCV_SIMD "AVX2 (compiler-capable)")
    endif()
endif()

# ---------------------------------------------------------------------------
# Support library: helpers that genuinely need OpenCV (image I/O for tests).
# Not part of the embedded core.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE BINCVC_SRC_FILES CONFIGURE_DEPENDS src/*.cpp)
if(BINCV_OPENCV_FOUND AND BINCVC_SRC_FILES)
    add_library(bincv_cpp STATIC ${BINCVC_SRC_FILES})
    target_link_libraries(bincv_cpp PUBLIC bincv_core PRIVATE bincv_warnings)
endif()

if(BINCV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BINCV_BUILD_BENCHMARKS AND BINCV_OPENCV_FOUND)
    add_subdirectory(benchmark)
endif()

# Every first-party target now exists, so the warning policy can be checked.
# FATAL_ERRORs if any of them compiles without the flags -- see the function's
# comment for why a build-log scan cannot substitute for this.
bincv_assert_warning_policy()

# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========== binCV Configuration ==========")
message(STATUS "Platform:   ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SIMD:       ${BINCV_SIMD}")
message(STATUS "Warnings:   -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion")
if(BINCV_WERROR)
    message(STATUS "            -Werror ON (gate build)")
else()
    message(STATUS "            -Werror OFF (set -DBINCV_WERROR=ON; scripts/verify.sh does)")
endif()
message(STATUS "            wired into ${BINCV_WARNING_POLICY_TARGETS} first-party target(s), all verified")
if(BINCV_BUILD_TESTS)
    message(STATUS "Tests:      ${BINCV_TEST_BACKEND_DESC}")
endif()
if(BINCV_OPENCV_FOUND)
    message(STATUS "OpenCV:     YES (${OpenCV_VERSION}) - interop enabled")
else()
    message(STATUS "OpenCV:     NO - core-only mode (embedded)")
    if(BINCV_BUILD_BENCHMARKS)
        message(STATUS "            benchmarks skipped (they compare against OpenCV)")
    endif()
endif()
message(STATUS "=========================================")
message(STATUS "")
