# Collect all test sources
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cpp)

# Build the test executable
add_executable(bincv_tests ${TEST_SOURCES})

# Link the main bincv_cuda library and OpenCV
target_link_libraries(bincv_tests
    PRIVATE bincv_cuda ${OpenCV_LIBS}
)

# Include OpenCV headers
target_include_directories(bincv_tests PRIVATE ${OpenCV_INCLUDE_DIRS})

# Add compile warnings for tests too
target_compile_options(bincv_tests PRIVATE
    -Wall -Wextra -Werror
)

# Add to CTest
add_test(NAME AllTests COMMAND bincv_tests)
