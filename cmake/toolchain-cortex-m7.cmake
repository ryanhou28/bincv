# Cross-compilation toolchain for Cortex-M7 (ARMv7E-M), bare metal.
#
# Reference part: STM32H753ZI on a NUCLEO-H753ZI -- Cortex-M7 at 480 MHz with a
# double-precision FPU, 2 MB flash and 1 MB SRAM.
#
#   cmake -S . -B build-m7 \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-cortex-m7.cmake \
#         -DBINCV_USE_OPENCV=OFF -DBINCV_BUILD_BENCHMARKS=OFF
#
# Point BINCV_ARM_TOOLCHAIN_DIR at an arm-none-eabi install, or put one on PATH.

set(CMAKE_SYSTEM_NAME Generic)

# "arm", not "cortex-m7". The top-level CMakeLists branches on this string, and the
# arm branch is the correct one to take: it probes `-mfpu=neon`, which a Cortex-M
# compiler rejects, so BINCV_HAVE_NEON stays undefined and BINCV_SIMD reports
# "scalar". Naming the part instead would fall through to the x86 branch, reach the
# same answer by having every x86 probe fail, and be right by accident.
set(CMAKE_SYSTEM_PROCESSOR arm)

set(BINCV_ARM_TOOLCHAIN_DIR "" CACHE PATH "Directory holding arm-none-eabi-* binaries")
if(BINCV_ARM_TOOLCHAIN_DIR)
    set(_bincv_arm_prefix "${BINCV_ARM_TOOLCHAIN_DIR}/arm-none-eabi-")
else()
    set(_bincv_arm_prefix "arm-none-eabi-")
endif()

set(CMAKE_C_COMPILER   "${_bincv_arm_prefix}gcc")
set(CMAKE_CXX_COMPILER "${_bincv_arm_prefix}g++")
set(CMAKE_ASM_COMPILER "${_bincv_arm_prefix}gcc")
set(CMAKE_AR           "${_bincv_arm_prefix}ar"      CACHE FILEPATH "")
set(CMAKE_OBJCOPY      "${_bincv_arm_prefix}objcopy" CACHE FILEPATH "")
set(CMAKE_SIZE         "${_bincv_arm_prefix}size"    CACHE FILEPATH "")

# Without this CMake's compiler check tries to LINK a hosted executable, which needs
# a linker script and reset vector this file deliberately does not supply -- the
# library is the artefact here, and the application that links it brings its own.
# The check would fail on a perfectly good compiler.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# -mfloat-abi=hard: the H7's FPU is double-precision (fpv5-d16), and binCV's geometry
# (ops/essential.hpp, ops/ransac.hpp) is the only double-heavy code. Soft-float would
# make that arithmetic a library call per operation.
set(_bincv_m7_arch "-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard")

# Section-per-symbol so --gc-sections can drop what the firmware never calls; the
# footprint this target reports should be binCV's, not the whole library's.
set(_bincv_m7_lang "-ffunction-sections -fdata-sections")

# No OS, so no exceptions, no RTTI and no unwinder. ARCHITECTURE 6 already requires
# that nothing in a kernel throws, so this removes tables rather than behaviour.
# C++ only -- gcc rejects both flags for C, and the warning gate makes that fatal.
set(CMAKE_C_FLAGS_INIT   "${_bincv_m7_arch} ${_bincv_m7_lang}")
set(CMAKE_CXX_FLAGS_INIT "${_bincv_m7_arch} ${_bincv_m7_lang} -fno-exceptions -fno-rtti")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_bincv_m7_arch} -Wl,--gc-sections")

# Nothing on the host is a candidate for this target.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BEFORE)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
