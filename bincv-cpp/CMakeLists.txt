cmake_minimum_required(VERSION 3.12)
project(bincv_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Automatically gather all headers and sources in src/
file(GLOB_RECURSE BINCVC_SRC_FILES CONFIGURE_DEPENDS src/*.cpp src/*.hpp)

# Define static library
add_library(bincv_cpp STATIC ${BINCVC_SRC_FILES})

# Link OpenCV and expose include directory
target_link_libraries(bincv_cpp PUBLIC ${OpenCV_LIBS})
target_include_directories(bincv_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add tests
add_subdirectory(tests)