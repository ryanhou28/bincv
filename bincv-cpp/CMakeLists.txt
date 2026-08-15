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
        target_include_directories(bincv_core INTERFACE ${OpenCV_INCLUDE_DIRS})
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
    target_link_libraries(bincv_cpp PUBLIC bincv_core)
endif()

if(BINCV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BINCV_BUILD_BENCHMARKS AND BINCV_OPENCV_FOUND)
    add_subdirectory(benchmark)
endif()

# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========== binCV Configuration ==========")
message(STATUS "Platform:   ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SIMD:       ${BINCV_SIMD}")
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
