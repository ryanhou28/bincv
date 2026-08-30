#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# A 2.5-SECOND aarch64 SYNTAX CHECK, USING THE REFERENCE DEVICE AS A COMPILER.
#
# WHY THIS EXISTS. Roughly a third of ops/opticalFlow.hpp -- D-33's tap batching,
# X-40's window-carried accumulator -- lives inside
# `#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)` and is therefore INVISIBLE
# to every x86 build, including all four configurations of verify.sh. An edit there
# can be structurally broken and still pass the whole gate.
#
# scripts/verify_arm.sh covers that, but it needs Docker (it EMULATES aarch64, which
# is what makes it hermetic and device-independent). When the daemon is not running
# it skips, and the NEON region goes unchecked.
#
# THE DEVICE IS A REAL aarch64 COMPILER AND IS NOT THE SLOW PART. X-72 reverted a
# working refactor after reporting "no way to compile for aarch64" -- which was
# false. What was slow was `run_on_pi.sh`, which rsyncs and runs a FULL cmake build
# of every target. A syntax-only compile of one translation unit is SECONDS, and it
# was never tried.
#
# This is NOT a substitute for verify_arm.sh: it checks that the NEON region
# COMPILES, not that it computes the right answer, and it needs the device where
# verify_arm.sh does not. Use it as the inner loop; use verify_arm.sh (or the device
# test run) before committing.
# ---------------------------------------------------------------------------
set -uo pipefail
TARGET="${1:-pi4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="bincv-syntax"

command -v rsync >/dev/null 2>&1 || { echo "  rsync is not installed"; exit 77; }
ssh -4 -o BatchMode=yes -o ConnectTimeout=10 "$TARGET" true 2>/dev/null || {
    echo "  SKIPPED: $TARGET is not reachable over ssh."
    echo "  The NEON region was NOT checked. Not a failure -- but not a pass."
    exit 77
}

rsync -az --delete --exclude 'build*/' --exclude '.git/' --exclude 'results/' \
      -e 'ssh -4 -o BatchMode=yes -o ConnectTimeout=10' \
      "${REPO_ROOT}/" "${TARGET}:${REMOTE_DIR}/" || { echo "  rsync failed"; exit 1; }

# The SAME warning set the gate uses, -Werror included. A NEON path that only
# compiles without -Wconversion is not compiling (CLAUDE.md: -Wconversion is the
# load-bearing one, and every mask and shift here is templated on the word type).
OUT=$(ssh -4 -o BatchMode=yes "$TARGET" "cd ${REMOTE_DIR}/bincv-cpp && \
    g++ -std=c++17 -fsyntax-only -Iinclude -Itests \
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror \
        -DBINCV_HAVE_NEON -O2 -x c++ tests/test_opticalflow_syntax.cpp" 2>&1)
RC=$?
if [[ $RC -ne 0 ]]; then
    echo "$OUT" | head -40
    echo
    echo "  aarch64 SYNTAX CHECK FAILED (this is code x86 never compiles)."
    exit 1
fi
echo "  aarch64 syntax OK -- NEON region compiled with the gate's warning set."

# ---------------------------------------------------------------------------
# F-5 REGRESSION GUARD: THE NEON PATH MUST NOT DEPEND ON THE CMAKE TARGET.
#
# The compile above passes -DBINCV_HAVE_NEON, exactly as the bincv_core target does,
# so it proves the NEON region compiles and NOTHING about how a consumer reaches it.
# An integrator who adds the include path and forgets `target_link_libraries` used to
# get a correct library with every NEON kernel compiled out -- no warning, no different
# answer, and 1.78x slower `track` measured on this device.
#
# So this second compile passes NO define at all. core/simd.hpp is supposed to derive
# BINCV_HAVE_NEON from __ARM_NEON and __aarch64__, which the compiler defines by itself
# on ARMv8. If that stops working, this fails.
# ---------------------------------------------------------------------------
OUT=$(ssh -4 -o BatchMode=yes "$TARGET" "cd ${REMOTE_DIR}/bincv-cpp && \
    printf '%s\\n' '#include \"bincv-cpp/ops/opticalFlow.hpp\"' \
        '#if !defined(BINCV_HAVE_NEON)' \
        '#error \"F-5: BINCV_HAVE_NEON is not defined without the CMake target -- an include-only integration would silently lose every NEON kernel\"' \
        '#endif' \
        'int main() { return 0; }' > /tmp/bincv_f5_guard.cpp && \
    g++ -std=c++17 -fsyntax-only -Iinclude -O2 /tmp/bincv_f5_guard.cpp" 2>&1)
RC=$?
if [[ $RC -ne 0 ]]; then
    echo "$OUT" | head -20
    echo
    echo "  F-5 GUARD FAILED: an include-only build no longer gets NEON."
    exit 1
fi
echo "  F-5 guard OK -- include-only build auto-detects NEON (no CMake target needed)."
