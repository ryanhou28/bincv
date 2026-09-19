#!/usr/bin/env bash
#
# verify_cuda.sh -- CUDA backend gate.
#
# The CUDA backend is invisible to every other gate: verify.sh builds no .cu
# translation unit, and its bit-exactness claim -- every device kernel gives
# the host kernel's answer -- can only be observed where a GPU is. This gate
# configures the backend with warnings fatal, builds it, and runs the
# device-vs-host suite.
#
# Two things are true here and are the reason this gate is separate:
#
#   * The kernels are forked from the host but the FORMAT is shared, so the
#     test is a raw comparison: run both, download, compare bytes. A device
#     kernel that is faster and different fails here, which is the only place
#     it can.
#   * nvcc is a second compiler with its own diagnostics. -Werror is turned on
#     for both the host and device halves (via -Xcompiler), so a warning in a
#     .cu file is as fatal as one in a .cpp file is under verify.sh.
#
# The suites are not listed in this file. They are read out of
# backends/cuda/tests/CMakeLists.txt, which is where a suite is declared -- see
# the block that does it for why naming them here, twice, was a silent skip
# waiting on the next op family.
#
#   ./scripts/verify_cuda.sh
#
# The toolkit is found through CMake, or pointed at explicitly:
#
#   BINCV_CUDA_NVCC=/usr/local/cuda-11.1/bin/nvcc ./scripts/verify_cuda.sh
#   BINCV_CUDA_HOST_COMPILER=g++-9 ./scripts/verify_cuda.sh   # nvcc's -ccbin
#
# A CUDA build tree is large and a machine's root filesystem may not be where it
# belongs, so the build directory is overridable:
#
#   BINCV_CUDA_BUILD_DIR=/scratch/bincv-cuda-gate ./scripts/verify_cuda.sh
#
# EXIT CODES
#   0   CUDA backend built and the device-vs-host suite passed
#   1   verification FAILED (build error, a warning under -Werror, or a check)
#   77  could not run at all -- no nvcc, or no CUDA device to run the suite on.
#       NOT a pass: the same contract verify_cross.sh and verify_cortex_m.sh
#       use, so a caller can tell a skipped run from a verified one.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BINCV_CUDA_BUILD_DIR:-${REPO_ROOT}/build-cuda-gate}"

echo "============================================================"
echo "  binCV -- CUDA backend (device-vs-host bit-exactness)"
echo "============================================================"
echo

skip() {
    echo
    echo "  SKIPPED: $1"
    echo "  The CUDA backend was NOT verified. This is not a failure -- but it is"
    echo "  also not a pass. Install the CUDA toolkit and ensure a device is"
    echo "  visible, or set BINCV_CUDA_NVCC to an nvcc the host compiler accepts."
    echo "  (exit 77, so a caller can tell this apart from a verified pass.)"
    echo
    exit 77
}

# nvcc: explicit, on PATH, or the reference install.
NVCC="${BINCV_CUDA_NVCC:-}"
if [ -z "${NVCC}" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        NVCC="$(command -v nvcc)"
    elif [ -x /usr/local/cuda-11.1/bin/nvcc ]; then
        NVCC=/usr/local/cuda-11.1/bin/nvcc
    else
        skip "no nvcc found (set BINCV_CUDA_NVCC)"
    fi
fi
echo "  nvcc: ${NVCC}"

# ---------------------------------------------------------------------------
# The suites, named once
#
# They used to be named twice -- once in the `cmake --build --target` list and
# once in the loop that runs them -- and the two lists could disagree in
# silence. A suite missing from the first is run by nobody; a suite missing from
# the second is built by nobody; either way this gate stays green over a suite
# that never executed, which is the one failure a verification script must not
# have. Every op family added to the backend adds a suite, so the two lists had
# a drift scheduled for the next one.
#
# Derived from the tests' own CMakeLists.txt rather than written here, for the
# reason scripts/verify_cross.sh gives for deriving its manifest: a list kept in
# the script goes stale the moment a suite is added the intended way, and
# nothing notices. `bincv_add_test_target(<name> ...)` is that declaration.
# ---------------------------------------------------------------------------
CUDA_TESTS_CMAKE="${REPO_ROOT}/backends/cuda/tests/CMakeLists.txt"
SUITES=()
while IFS= read -r line; do
    [ -n "${line}" ] && SUITES+=("${line}")
done < <(sed -n 's/^[[:space:]]*bincv_add_test_target([[:space:]]*\([A-Za-z0-9_][A-Za-z0-9_]*\).*/\1/p' \
             "${CUDA_TESTS_CMAKE}" 2>/dev/null)

if [ ${#SUITES[@]} -eq 0 ]; then
    echo "  NO SUITES DERIVED from ${CUDA_TESTS_CMAKE}"
    echo "  This is a broken gate, not a skip: it would build the backend, run"
    echo "  nothing, and report a pass. Check that the suites are still declared"
    echo "  with bincv_add_test_target()."
    exit 1
fi

# The derivation is a text match, so it is only as good as the spelling it
# expects. That file names its binaries a SECOND time, independently, in the
# add_test(NAME ... COMMAND <binary>) calls -- so requiring every binary ctest is
# told to run to be one this gate builds turns "the sed missed a suite" from a
# silent skip into a failure. A name behind a variable, or a call split across
# lines, lands here rather than nowhere.
while IFS= read -r bin; do
    [ -n "${bin}" ] || continue
    case " ${SUITES[*]} " in
        *" ${bin} "*) ;;
        *)  echo "  add_test() in ${CUDA_TESTS_CMAKE} runs '${bin}', which is not in"
            echo "  the suite list derived from bincv_add_test_target(). This gate would"
            echo "  build it with nobody and run it with nobody. Declare the target that"
            echo "  way, or fix the derivation above -- do not leave it unrun."
            exit 1 ;;
    esac
done < <(sed -n 's/^[[:space:]]*add_test([[:space:]]*NAME[[:space:]][^)]*COMMAND[[:space:]]\{1,\}\([A-Za-z0-9_][A-Za-z0-9_]*\).*/\1/p' \
             "${CUDA_TESTS_CMAKE}" 2>/dev/null)

echo "  suites: ${SUITES[*]}"
echo "          (derived from backends/cuda/tests/CMakeLists.txt -- built and run"
echo "           from one list, so neither half can quietly lose one)"

CMAKE_ARGS=(
    -S "${REPO_ROOT}" -B "${BUILD_DIR}"
    -DBINCV_CUDA=ON
    -DBINCV_WERROR=ON
    -DBINCV_USE_OPENCV=OFF
    -DBINCV_BUILD_BENCHMARKS=OFF
    -DCMAKE_CUDA_COMPILER="${NVCC}"
)
if [ -n "${BINCV_CUDA_HOST_COMPILER:-}" ]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER="${BINCV_CUDA_HOST_COMPILER}")
fi

rm -rf "${BUILD_DIR}"
echo "  configuring (warnings fatal)..."
if ! cmake "${CMAKE_ARGS[@]}" >"${BUILD_DIR}.configure.log" 2>&1; then
    cat "${BUILD_DIR}".configure.log
    echo "  CONFIGURE FAILED"
    exit 1
fi

echo "  building..."
if ! cmake --build "${BUILD_DIR}" --target bincv_cuda "${SUITES[@]}" -j"$(nproc)" \
        >"${BUILD_DIR}.build.log" 2>&1; then
    cat "${BUILD_DIR}".build.log
    echo "  BUILD FAILED"
    exit 1
fi
# A warning nvcc emitted but did not turn into an error (it forwards only the
# host half to -Werror): the same belt-and-braces log scan verify.sh runs.
if grep -q "warning:" "${BUILD_DIR}".build.log; then
    grep "warning:" "${BUILD_DIR}".build.log
    echo "  BUILD EMITTED WARNINGS"
    exit 1
fi

echo "  running device-vs-host suites..."
for suite in "${SUITES[@]}"; do
    "${BUILD_DIR}/backends/cuda/tests/${suite}"
    RC=$?
    if [ ${RC} -eq 77 ]; then
        skip "built cleanly, but no CUDA device is available to run ${suite}"
    elif [ ${RC} -ne 0 ]; then
        echo "  DEVICE-VS-HOST SUITE FAILED: ${suite}"
        exit 1
    fi
done

echo
echo "  CUDA BACKEND VERIFIED"
echo "  Built with -Werror on both host and device halves; every device kernel"
echo "  matched the host library byte for byte."
echo
exit 0
