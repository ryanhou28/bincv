# Bare-metal target for the NUCLEO-H753ZI. Built only when cross-compiling for
# M-profile -- see the guard in the top-level CMakeLists.
enable_language(C)

add_executable(bincv_m7
    startup.c
    board.c
    syscalls.c
    main.cpp)

# bincv_warnings is not optional for a first-party target (cmake/BincvWarnings.cmake),
# and it is wanted here for its own sake: this is the first code in the project
# compiled at 32-bit pointer width, which is exactly where -Wconversion earns its
# keep.
target_link_libraries(bincv_m7 PRIVATE bincv_core bincv_warnings)

# The loop arms live with the host benchmark that also times them, so both targets
# time the SAME source. Two copies of four arms would be two things to keep
# bit-exact, and the comparison between host and board would stop meaning anything
# the moment they drifted.
target_include_directories(bincv_m7 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmark)

# The linker script reserves a 16 KB stack in DTCM; the tracker static_asserts its
# staging buffers against this number. Set from one place so the two cannot drift --
# a budget larger than the stack turns a compile-time check into a silent overflow.
target_compile_definitions(bincv_m7 PRIVATE BINCV_STAGING_BUDGET_BYTES=16384)

set(BINCV_M7_LDSCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/stm32h753.ld")

target_link_options(bincv_m7 PRIVATE
    -T${BINCV_M7_LDSCRIPT}
    # newlib-nano: snprintf, which core/simd.hpp uses for its status line, otherwise
    # brings in full newlib stdio. Nothing binCV itself needs is affected -- no
    # kernel allocates, throws, or does I/O.
    --specs=nano.specs
    -nostartfiles
    -Wl,--gc-sections
    -Wl,-Map=$<TARGET_FILE_DIR:bincv_m7>/bincv_m7.map
    -Wl,--print-memory-usage)

set_target_properties(bincv_m7 PROPERTIES
    SUFFIX ".elf"
    LINK_DEPENDS "${BINCV_M7_LDSCRIPT}")

# A .bin is what the ST-LINK's mass-storage interface flashes when it is copied to
# the board's drive, which is the path that needs no debugger attached.
add_custom_command(TARGET bincv_m7 POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:bincv_m7>
            $<TARGET_FILE_DIR:bincv_m7>/bincv_m7.bin
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:bincv_m7>
    COMMENT "bincv_m7.bin + size")
