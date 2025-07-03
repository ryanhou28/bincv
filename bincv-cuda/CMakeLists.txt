cmake_minimum_required(VERSION 3.18)
project(bincv_cuda LANGUAGES CXX CUDA)

# Set language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include path for public headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files (both .cpp and .cu)
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
)

# Create static library target
add_library(bincv_cuda STATIC ${SRC_FILES})

# Enable CUDA features
set_target_properties(bincv_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Apply compile options per language
target_compile_options(bincv_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# === Add OpenCV ===
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

enable_testing()
add_subdirectory(tests)