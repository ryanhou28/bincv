# Project warning policy (T1.8).
#
# Until T1.8 nothing in this project enabled a single warning flag, so the
# "all three configurations must build warning-free" gate in CLAUDE.md,
# TASKS.md (V-ALL) and GETTING_STARTED.md passed vacuously -- see
# OVERNIGHT_LOG.md finding 5. This file is what gives that sentence teeth.
#
# Usage: link `bincv_warnings` PRIVATE into every FIRST-PARTY target. It is
# deliberately NOT attached to the bincv_core INTERFACE target: a downstream
# consumer's warning policy is theirs to choose, and forcing -Wconversion onto
# someone else's translation unit is how a header-only library earns a reputation
# for being unusable.

add_library(bincv_warnings INTERFACE)

option(BINCV_WERROR "Promote warnings to errors (scripts/verify.sh turns this on)" OFF)

# The first-party source root, captured where this module is included so that
# bincv_assert_warning_policy() below can tell our directories from a fetched
# dependency's. Not CMAKE_SOURCE_DIR: that is the parent project's when binCV is
# vendored into one.
set(BINCV_POLICY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "first-party source root")

# ---------------------------------------------------------------------------
# Why these flags
#
#   -Wall -Wextra -Wpedantic   the baseline. Zero warnings across every
#                              translation unit before this file existed, which
#                              is the good news buried in finding 5: the code was
#                              clean by care, it just was not clean by check.
#
#   -Wshadow                   a shadowed `width`/`stride`/`x` inside a packing
#                              loop reads correct and is not.
#
#   -Wconversion               this is the load-bearing one for binCV. The
#   -Wsign-conversion          library is templated on the word TYPE (D-1), so
#                              every mask and shift is written once and compiled
#                              at 8, 16, 32 and 64 bits. An expression that is
#                              exact at uint64_t can truncate at uint8_t, and
#                              integer promotion means the narrow instantiations
#                              are the ones that silently narrow back -- `~mask`
#                              on a uint8_t is an int. Turning these on found
#                              exactly that shape in QuantMat<1>::pad().
#
#                              The cost is that every deliberate narrowing needs
#                              a static_cast. In a bit-packing container that is
#                              the right price: a cast is where a reader should
#                              be told the truncation is intended.
#
# Not enabled, deliberately:
#   -Wpadded                   fires on every view struct by design (D-9 pins
#                              their layout; the test asserts the size).
#   -Wold-style-cast           would fire inside OpenCV-adjacent code far more
#                              than it would find anything here.
#   -Wuseless-cast             conflicts head-on with -Wconversion: the casts the
#                              latter demands at uint8_t are redundant at
#                              uint64_t, and both instantiations are compiled.
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_options(bincv_warnings INTERFACE /W4)
    if(BINCV_WERROR)
        target_compile_options(bincv_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(bincv_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion)
    # -Werror is a GATE flag, not a development flag.
    #
    # Always-on -Werror makes an unrelated compiler upgrade break every working
    # tree at once, and it stops the build at the first diagnostic when a
    # mid-edit developer wants to see all of them. Off by default is therefore
    # the kinder default -- but "kinder" is only defensible because the flag is
    # not optional where it counts: scripts/verify.sh configures every
    # configuration with -DBINCV_WERROR=ON, and nothing may be committed until
    # verify.sh is green (CLAUDE.md).
    #
    # verify.sh ALSO greps its build logs for "warning:". That scan catches
    # diagnostics -Werror does not turn into errors -- linker warnings, CMake's
    # own warnings, and anything a third-party target emits. It does NOT catch a
    # first-party target that forgets to link bincv_warnings: such a target is
    # compiled with no warning flags at all, so it emits nothing for the scan to
    # find. bincv_assert_warning_policy() below is what covers that case, and it
    # covers it structurally rather than by inspection of output.
    if(BINCV_WERROR)
        target_compile_options(bincv_warnings INTERFACE -Werror)
    endif()
endif()

# ---------------------------------------------------------------------------
# bincv_assert_warning_policy() -- the wiring is not optional
#
# `target_link_libraries(foo PRIVATE bincv_warnings)` is one line, and a target
# that omits it compiles with NO warning flags. That failure is silent in the
# worst possible way: the build log contains no diagnostics, so a log scan reads
# it as clean, and -Werror never sees the translation unit. Measured before this
# function existed: a test target linking only bincv_core, whose source emits
# three warnings under the project flag set, produced `WARN 0 ... PASS` and
# `ALL CONFIGURATIONS GREEN`.
#
# So the wiring is asserted at configure time instead. Every first-party target
# that compiles anything must link bincv_warnings, or configuration fails and
# names the target.
#
# Why not `add_compile_options()` at directory scope, which cannot be opted out
# of at all: FetchContent adds googletest as a subdirectory, and directory-scope
# flags would land on its 30k lines too. Under -Werror that is a build failure in
# code nobody here can fix. The interface target plus this assertion gets the same
# guarantee for first-party code without reaching into a dependency.
#
# Call after every add_subdirectory(), so that every target exists.
# ---------------------------------------------------------------------------
function(bincv_assert_warning_policy)
    set(_dirs "${BINCV_POLICY_ROOT}")
    set(_unwired "")
    set(_checked 0)

    while(_dirs)
        list(POP_FRONT _dirs _dir)

        # First-party only. A fetched dependency is configured from
        # ${CMAKE_BINARY_DIR}/_deps/..., outside the source root, and its warning
        # policy is its own.
        string(FIND "${_dir}" "${BINCV_POLICY_ROOT}" _under)
        if(NOT _under EQUAL 0 OR "${_dir}" MATCHES "/_deps/")
            continue()
        endif()

        get_property(_subs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
        list(APPEND _dirs ${_subs})

        get_property(_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(_t IN LISTS _targets)
            get_target_property(_type ${_t} TYPE)
            # INTERFACE and UTILITY targets compile no translation unit, so there
            # is nothing for a warning flag to apply to.
            if(NOT _type MATCHES "^(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY)$")
                continue()
            endif()
            math(EXPR _checked "${_checked} + 1")
            get_target_property(_libs ${_t} LINK_LIBRARIES)
            if(NOT _libs OR NOT "bincv_warnings" IN_LIST _libs)
                list(APPEND _unwired "${_t}")
            endif()
        endforeach()
    endwhile()

    if(_unwired)
        list(JOIN _unwired "\n    " _pretty)
        message(FATAL_ERROR
            "binCV warning policy: these first-party targets do not link "
            "bincv_warnings, so they compile with no warning flags and neither "
            "-Werror nor a build-log scan can see anything they emit:\n"
            "    ${_pretty}\n"
            "Add `target_link_libraries(<target> PRIVATE bincv_warnings)`, or "
            "build them through bincv_add_test_target() which does it for you.")
    endif()

    set(BINCV_WARNING_POLICY_TARGETS "${_checked}" PARENT_SCOPE)
endfunction()
