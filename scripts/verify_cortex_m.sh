#!/usr/bin/env bash
#
# verify_cortex_m.sh -- Cortex-M (ARMv7E-M) correctness gate.
#
# Cortex-M is the target binCV's memory claim is aimed at, and it is the one the
# other gates cannot see at all. Three things are true here and nowhere else in
# this repository's verification:
#
#   * `size_t` is 32 bits. Every index, stride and planeWords() is a size_t, and
#     the four-word-type sweep is otherwise only ever compiled at 64-bit pointer
#     width. The first run of this gate found a test asserting `size_t{1} << 63`.
#   * There is no population count instruction and no NEON, so the software path
#     is the only path and every vector `#if` takes its other branch.
#   * There is no OS. No heap beyond what the link provides, no exceptions, no
#     threads -- `-fno-exceptions -fno-rtti`, newlib, no vendor SDK.
#
# CORRECTNESS ONLY, AND COMPILE-ONLY. This gate does not execute anything: a
# Cortex-M image cannot run on the host, and qemu-system emulation of a specific
# part is a heavier dependency than a gate should carry. Issue #13 allows exactly
# this ("qemu-user or a cross-compiler suffices, since this is a correctness
# axis"). Execution is verified on real hardware -- see
# bincv-cpp/embedded/stm32h753/README.md -- and nothing here is a timing result.
#
#   ./scripts/verify_cortex_m.sh
#
# The toolchain is found on PATH, or via BINCV_ARM_TOOLCHAIN_DIR:
#
#   BINCV_ARM_TOOLCHAIN_DIR=~/toolchains/arm-gnu-.../bin ./scripts/verify_cortex_m.sh
#
# EXIT CODES
#   0   Cortex-M compilation verified
#   1   verification FAILED
#   77  could not run at all (no arm-none-eabi toolchain) -- NOT a pass, the same
#       contract scripts/verify_arm.sh uses so that a caller can tell a skipped
#       run from a verified one.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/bincv-cpp"

MCPU="-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard"
WARN="-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror"
LANG_FLAGS="-std=c++17 -O2 -fno-exceptions -fno-rtti"

echo "============================================================"
echo "  binCV -- Cortex-M7 correctness (compile-only, 32-bit size_t)"
echo "============================================================"
echo

skip() {
    echo
    echo "  SKIPPED: $1"
    echo "  Cortex-M compilation was NOT verified. This is not a failure -- but it"
    echo "  is also not a pass, and Cortex-M is the target the memory claim is for."
    echo "  Install an arm-none-eabi toolchain, or set BINCV_ARM_TOOLCHAIN_DIR."
    echo "  (exit 77, so a caller can tell this apart from a verified pass.)"
    echo
    exit 77
}

# --- availability ------------------------------------------------------------
if [[ -n "${BINCV_ARM_TOOLCHAIN_DIR:-}" ]]; then
    CXX="${BINCV_ARM_TOOLCHAIN_DIR}/arm-none-eabi-g++"
    [[ -x "${CXX}" ]] || skip "BINCV_ARM_TOOLCHAIN_DIR is set but ${CXX} is not executable"
else
    command -v arm-none-eabi-g++ >/dev/null 2>&1 || \
        skip "arm-none-eabi-g++ is not on PATH and BINCV_ARM_TOOLCHAIN_DIR is unset"
    CXX="$(command -v arm-none-eabi-g++)"
fi

echo "  toolchain: ${CXX}"
echo "             $("${CXX}" --version | head -1)"

# The compiler must actually be an M-profile one. A host g++ symlinked into place,
# or an aarch64 cross-compiler, would compile most of this and prove nothing about
# the target -- and __ARM_ARCH_PROFILE is the same macro core/simd.hpp gates on.
if ! echo 'int main(){return 0;}' | \
     "${CXX}" ${MCPU} -x c++ -dM -E - 2>/dev/null | grep -q "__ARM_ARCH_PROFILE 77"; then
    skip "${CXX} does not report an M-profile target for ${MCPU}"
fi
echo "  ok -- reports __ARM_ARCH_PROFILE 'M'"
echo

# --- what is excluded, and why -----------------------------------------------
# Named individually. A gate that silently skips what it cannot build reports a
# pass for a shrinking set of files, which is the vacuity this project's warning
# policy exists to prevent.
declare -A EXCLUDED=(
  [test_parallel]="threads/pool.hpp needs <thread>/<mutex>; newlib has none. ARCHITECTURE 9: binCV is serial by default and threads through a caller-installed backend, so a target with no threads never installs one."
  [test_opencv_interop]="needs OpenCV, which is not an embedded dependency."
  [test_equivalence]="needs OpenCV; guarded by its own #error."
  [test_covariance_n_bound]="a deliberate compile FAILURE, driven by tests/expect_fatal.cmake. Checked below rather than skipped."
)

echo "  not compiled here, by design:"
for name in "${!EXCLUDED[@]}"; do
    printf '    %-24s %s\n' "${name}" "${EXCLUDED[$name]}"
done | sort
echo

# --- compile ------------------------------------------------------------------
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

INCLUDES="-I${SRC_DIR}/include -I${SRC_DIR}/tests"
compiled=0
failed=0
warned=0
FAILED_LIST=()

echo "  compiling the suites for cortex-m7..."
for src in "${SRC_DIR}"/tests/test_*.cpp; do
    base="$(basename "${src}" .cpp)"
    [[ -n "${EXCLUDED[$base]:-}" ]] && continue

    log="${TMP_DIR}/${base}.log"
    if "${CXX}" ${MCPU} ${LANG_FLAGS} ${WARN} ${INCLUDES} -c "${src}" -o /dev/null 2>"${log}"; then
        compiled=$((compiled + 1))
        if grep -q 'warning:' "${log}"; then
            warned=$((warned + 1))
            echo "    WARNING in ${base}:"
            head -5 "${log}" | sed 's/^/      /'
        fi
    else
        failed=$((failed + 1))
        FAILED_LIST+=("${base}")
        echo "    FAILED ${base}:"
        grep -E 'error:' "${log}" | head -4 | sed 's/^/      /'
    fi
done
echo "  ${compiled} suite(s) compiled clean, ${failed} failed"

# --- the negative test must still fail ---------------------------------------
# The same argument as verify.sh's gate self-check: a check nobody has watched
# fail is not known to work, and a static_assert that stopped firing on this
# target would look exactly like a pass.
echo
echo "  checking that the deliberate-failure suite still fails to compile..."
if "${CXX}" ${MCPU} ${LANG_FLAGS} ${WARN} ${INCLUDES} \
        -c "${SRC_DIR}/tests/test_covariance_n_bound.cpp" -o /dev/null 2>/dev/null; then
    echo "    FAILED: test_covariance_n_bound COMPILED. Its static_assert is not firing"
    echo "            on this target, so the bound it guards is unguarded here."
    failed=$((failed + 1))
else
    echo "    ok -- it fails, as it must"
fi

# --- the library must link into a real image ---------------------------------
# Compiling every suite proves the headers are portable. It does not prove the
# library links with no OS underneath it, which is a separate failure mode:
# a missing symbol appears only at link time.
echo
echo "  linking the bare-metal image (proves it links with no OS)..."
FW_DIR="${TMP_DIR}/fw"
fw_ok=0
if cmake -S "${SRC_DIR}" -B "${FW_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${SRC_DIR}/cmake/toolchain-cortex-m7.cmake" \
        ${BINCV_ARM_TOOLCHAIN_DIR:+-DBINCV_ARM_TOOLCHAIN_DIR="${BINCV_ARM_TOOLCHAIN_DIR}"} \
        -DBINCV_USE_OPENCV=OFF -DBINCV_BUILD_TESTS=OFF -DBINCV_BUILD_BENCHMARKS=OFF \
        -DBINCV_BUILD_EMBEDDED=ON -DBINCV_WERROR=ON > "${TMP_DIR}/fw-configure.log" 2>&1; then
    if cmake --build "${FW_DIR}" -j"$(nproc 2>/dev/null || echo 4)" \
            > "${TMP_DIR}/fw-build.log" 2>&1; then
        fw_ok=1
        # The SIMD line the firmware would print is the one GETTING_STARTED tells a
        # reader to trust, and on this target it must not claim a fast path.
        if grep -q 'scalar only' "${FW_DIR}/embedded/stm32h753/bincv_m7.elf" 2>/dev/null || \
           strings "${FW_DIR}/embedded/stm32h753/bincv_m7.elf" 2>/dev/null | grep -q 'scalar only'; then
            echo "    ok -- image links, and its SIMD status reports 'scalar only'"
        else
            echo "    FAILED: the image links but does not carry the 'scalar only' status"
            echo "            string, so simdStatusString may be claiming a fast path here."
            failed=$((failed + 1))
        fi
        size_line="$("${CXX%g++}size" "${FW_DIR}/embedded/stm32h753/bincv_m7.elf" 2>/dev/null | tail -1)"
        [[ -n "${size_line}" ]] && echo "    size (text/data/bss): ${size_line}"
    else
        echo "    FAILED to build:"
        grep -E 'error:|Error' "${TMP_DIR}/fw-build.log" | head -5 | sed 's/^/      /'
        failed=$((failed + 1))
    fi
else
    echo "    FAILED to configure:"
    tail -5 "${TMP_DIR}/fw-configure.log" | sed 's/^/      /'
    failed=$((failed + 1))
fi

# --- verdict ------------------------------------------------------------------
echo
echo "============================================================"
if [[ ${failed} -eq 0 && ${warned} -eq 0 ]]; then
    echo "  CORTEX-M OK -- ${compiled} suites compile clean at 32-bit size_t,"
    echo "  the negative test still fails, and the bare-metal image links."
    echo "  Compile-only: nothing was executed. Execution is verified on hardware,"
    echo "  see bincv-cpp/embedded/stm32h753/README.md."
    echo "============================================================"
    exit 0
fi
echo "  CORTEX-M FAILED -- ${failed} error(s), ${warned} warning(s)"
[[ ${#FAILED_LIST[@]} -gt 0 ]] && echo "  suites that did not compile: ${FAILED_LIST[*]}"
echo "============================================================"
exit 1
