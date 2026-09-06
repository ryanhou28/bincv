# Runs a program that is expected to die, and passes only if it both died and
# said why.
#
# ctest cannot express this on its own. PASS_REGULAR_EXPRESSION ignores the
# return code but still reports a signal-killed process as a failure, and
# abort() -- a signal -- is exactly how BINCV_THROW and BINCV_ASSERT report a
# failed check where exceptions are unavailable. Checking it here also makes the
# test stricter than either ctest property alone: the process must terminate
# abnormally AND name the reason, not one or the other.
#
# Usage:
#   cmake -DCHILD=<exe> [-DCHILD_ARGS=<arg>] -DEXPECT_OUTPUT=<regex> -P expect_fatal.cmake
#
# CHILD_ARGS is how one death-test binary covers many checks: each validation or
# assertion site is a case selected by argv[1], registered as its own ctest test
# with its own expected diagnostic.

if(NOT DEFINED CHILD OR NOT DEFINED EXPECT_OUTPUT)
    message(FATAL_ERROR "expect_fatal.cmake requires -DCHILD= and -DEXPECT_OUTPUT=")
endif()

if(NOT DEFINED CHILD_ARGS)
    set(CHILD_ARGS "")
endif()

# TIMEOUT so that a check which stops terminating hangs this script rather than
# the whole ctest run; without it the failure mode of "the process no longer
# dies" is an unbounded wait.
execute_process(
    COMMAND "${CHILD}" ${CHILD_ARGS}
    TIMEOUT 60
    RESULT_VARIABLE result
    OUTPUT_VARIABLE childStdout
    ERROR_VARIABLE childStderr)

set(combined "${childStdout}${childStderr}")
message(STATUS "child output:\n${combined}")

# execute_process reports a normal exit as the numeric exit code and an abnormal
# one as a description ("Subprocess aborted"), so the distinction this test is
# about -- died vs. returned -- is exactly the distinction between a number and
# anything else. Verified for both configurations: abort() from BINCV_THROW and
# an uncaught std::invalid_argument through std::terminate both report
# "Subprocess aborted".
#
# Rejecting every numeric result, not just 0, is the point. A regression that
# turned a fatal check into a clean `return 1` used to pass here, which made the
# script's own promise ("must terminate abnormally") untrue.
if(result MATCHES "[Tt]imeout")
    message(FATAL_ERROR
        "'${CHILD} ${CHILD_ARGS}' did not terminate within the timeout (${result})")
endif()

if(result MATCHES "^-?[0-9]+$")
    message(FATAL_ERROR
        "expected '${CHILD} ${CHILD_ARGS}' to terminate abnormally (abort, or an "
        "uncaught exception), but it returned normally with exit code ${result}")
endif()

if(NOT combined MATCHES "${EXPECT_OUTPUT}")
    message(FATAL_ERROR
        "'${CHILD} ${CHILD_ARGS}' terminated (${result}) without the expected diagnostic.\n"
        "  expected output matching: ${EXPECT_OUTPUT}")
endif()

message(STATUS "OK: terminated as expected (${result}), with the expected diagnostic")
