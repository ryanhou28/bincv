# Bare-metal images for the NUCLEO-H753ZI. Built only when cross-compiling for
# M-profile -- see the guard in the top-level CMakeLists.
enable_language(C)

# The serial rate the firmware and scripts/m7_host.py talk at. Frames for the
# tracker are streamed over this link, outside every timed region, so it sets how
# long a sweep takes and nothing a sweep reports. 4 Mbaud is 30 x the 120 MHz
# peripheral clock exactly, and moved 3.7 MB through the ST-LINK without a CRC
# failure.
set(BINCV_M7_BAUD 4000000 CACHE STRING "USART3 rate for the host link")

# Leave the part on its 64 MHz HSI reset clock instead of bringing it to 480 MHz
# (board.h). The first measurements on this board were taken that way.
option(BINCV_M7_RESET_CLOCK "Run the images on the 64 MHz reset clock" OFF)

set(BINCV_M7_LDSCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/stm32h753.ld")

# Everything an image needs besides its own sources. One image per benchmark, each
# named after its translation unit, because scripts/check_figure_staleness.py ages
# a figure when a file its image compiles changes -- and it reads each image's
# sources out of the literal add_executable below, so those stay spelled out.
function(bincv_m7_image target)
    # bincv_warnings is not optional for a first-party target
    # (cmake/BincvWarnings.cmake), and it is wanted here for its own sake: this is
    # code compiled at 32-bit pointer width, where -Wconversion earns its keep.
    target_link_libraries(${target} PRIVATE bincv_core bincv_warnings)

    # Arms shared with a host benchmark live in benchmark/, so the board and the
    # host time the SAME source. Two copies would be two things to keep bit-exact.
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmark)

    # The linker script reserves a 16 KB stack in DTCM; the tracker static_asserts
    # its staging buffers against this number. Set from one place so the two cannot
    # drift -- a budget larger than the stack turns a compile-time check into a
    # silent overflow.
    target_compile_definitions(${target} PRIVATE BINCV_STAGING_BUDGET_BYTES=16384
                                                 BINCV_M7_BAUD=${BINCV_M7_BAUD}u)
    if(BINCV_M7_RESET_CLOCK)
        target_compile_definitions(${target} PRIVATE BINCV_M7_RESET_CLOCK)
    endif()

    target_link_options(${target} PRIVATE
        -T${BINCV_M7_LDSCRIPT}
        # newlib-nano: snprintf, which core/simd.hpp uses for its status line,
        # otherwise brings in full newlib stdio. No kernel allocates, throws, or does
        # I/O; but nano's memcpy, memmove and memset are byte loops, and kernels do
        # call those, so string.c replaces them with word-wide ones.
        --specs=nano.specs
        -nostartfiles
        -Wl,--gc-sections
        -Wl,-Map=$<TARGET_FILE_DIR:${target}>/${target}.map
        -Wl,--print-memory-usage)

    set_target_properties(${target} PROPERTIES
        SUFFIX ".elf"
        LINK_DEPENDS "${BINCV_M7_LDSCRIPT}")

    # A .bin is what the ST-LINK's mass-storage interface flashes when it is copied
    # to the board's drive, which is the path that needs no debugger attached.
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${target}>
                $<TARGET_FILE_DIR:${target}>/${target}.bin
        COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${target}>
        COMMENT "${target}.bin + size")
endfunction()

add_executable(bincv_m7_reduce startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_reduce.cpp)
bincv_m7_image(bincv_m7_reduce)

add_executable(bincv_m7_dense startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_dense.cpp)
bincv_m7_image(bincv_m7_dense)

add_executable(bincv_m7_track startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_track.cpp)
bincv_m7_image(bincv_m7_track)

# The tracker with its derivatives computed where they are read instead of stored
# (ops/opticalFlowOnDemand.hpp and goodFeaturesToTrackStreaming's GradientSource): the
# same source, so the two images differ in that and nothing else.
add_executable(bincv_m7_track_ondemand startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_track.cpp)
bincv_m7_image(bincv_m7_track_ondemand)
target_compile_definitions(bincv_m7_track_ondemand PRIVATE BINCV_M7_DERIVATIVES_ON_DEMAND)

# The same tracker detecting by gridCorners, its derivatives made where they are read.
add_executable(bincv_m7_track_grid startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_track.cpp)
bincv_m7_image(bincv_m7_track_grid)
target_compile_definitions(bincv_m7_track_grid PRIVATE BINCV_M7_DERIVATIVES_ON_DEMAND BINCV_M7_GRID)

# DIAGNOSTIC, never a published figure: the tracker with opticalFlow.hpp's stage
# timers on, reading the board's cycle counter. It answers where `track` goes.
add_executable(bincv_m7_lkstages startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_track.cpp)
bincv_m7_image(bincv_m7_lkstages)
target_compile_definitions(bincv_m7_lkstages PRIVATE BINCV_LK_STAGE_TIMING
                                                     "BINCV_LK_STAGE_CLOCK=boardCycles64()")

add_executable(bincv_m7_sensor startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_sensor.cpp)
bincv_m7_image(bincv_m7_sensor)

# Not a kernel: what single instructions cost on this core, which the DSP arms are
# written against.
add_executable(bincv_m7_cycles startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_cycles.cpp)
bincv_m7_image(bincv_m7_cycles)

add_executable(bincv_m7_stereo startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_stereo.cpp)
bincv_m7_image(bincv_m7_stereo)

add_executable(bincv_m7_essential startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_essential.cpp)
bincv_m7_image(bincv_m7_essential)

add_executable(bincv_m7_subpix startup.c board.c syscalls.c string.c m7_harness.cpp bincv_m7_subpix.cpp)
bincv_m7_image(bincv_m7_subpix)
