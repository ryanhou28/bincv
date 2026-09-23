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
# Every suite's CHECK COUNT is held to a floor, per configuration, in
# backends/cuda/tests/expected-checks.txt -- the same contract, and the same row
# format, as tests/expected-checks.txt is for verify.sh. Raising a floor is a
# reviewed edit:
#
#   ./scripts/verify_cuda.sh --update-checks-baseline   # then commit the diff
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
BASELINE_FILE="${REPO_ROOT}/backends/cuda/tests/expected-checks.txt"

UPDATE_BASELINE=0
for arg in "$@"; do
    case "${arg}" in
        --update-checks-baseline) UPDATE_BASELINE=1 ;;
        -h|--help) sed -n '3,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "verify_cuda.sh: unknown argument '${arg}'" >&2; exit 2 ;;
    esac
done

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

# ---------------------------------------------------------------------------
# Two configurations, for the reason verify.sh runs four
#
#   Release  the shipped build, and the one that prices anything. It also
#            builds the BENCHMARKS, compile-only: every operation here gets a
#            benchmark arm the day it is written, and an arm no gate compiles
#            rots at the speed the harness underneath it changes. They are not
#            RUN -- a timing run needs an idle GPU it has no right to assume.
#
#   Debug    the only configuration where BINCV_ASSERT survives to nvcc's
#            device pass. Every other CUDA build in this project carries
#            -DNDEBUG, so a device op's domain assertions -- the ones a
#            narrowed device domain is required to make -- were compiled out
#            before anything could check they even build, let alone fire. This
#            is the CUDA analogue of verify.sh's fourth configuration and it
#            exists for the same reason: an assertion nobody has compiled is
#            not known to work.
# ---------------------------------------------------------------------------
run_configuration() {
    local name="$1" build_type="$2" benchmarks="$3"
    local dir="${BUILD_DIR}-${name}"

    echo
    echo "  --- ${name} (CMAKE_BUILD_TYPE=${build_type}, benchmarks=${benchmarks}) ---"

    local args=(
        -S "${REPO_ROOT}" -B "${dir}"
        -DBINCV_CUDA=ON
        -DBINCV_WERROR=ON
        -DBINCV_USE_OPENCV=OFF
        -DCMAKE_BUILD_TYPE="${build_type}"
        -DBINCV_BUILD_BENCHMARKS="${benchmarks}"
        -DCMAKE_CUDA_COMPILER="${NVCC}"
    )
    if [ -n "${BINCV_CUDA_HOST_COMPILER:-}" ]; then
        args+=(-DCMAKE_CUDA_HOST_COMPILER="${BINCV_CUDA_HOST_COMPILER}")
    fi

    rm -rf "${dir}"
    echo "  configuring (warnings fatal)..."
    if ! cmake "${args[@]}" >"${dir}.configure.log" 2>&1; then
        cat "${dir}.configure.log"
        echo "  CONFIGURE FAILED (${name})"
        exit 1
    fi

    local targets=(bincv_cuda "${SUITES[@]}")
    # The worked examples, in BOTH configurations. An example is the one thing a
    # reader can run to check a claim, and an example no gate compiles rots at
    # the speed the headers under it change -- the same argument that made the
    # benchmarks a gate target. They are built, not run: the resident tracking pipeline
    # needs a frame blob and an idle GPU, neither of which this gate may assume.
    #
    # DERIVED FROM THE FILENAMES rather than listed here, because that directory
    # declares its targets with a file(GLOB) -- so there is no add_executable()
    # name to match, and a list written here would go stale the moment an
    # example is added the intended way. Same reason the suites and the
    # benchmarks are derived.
    while IFS= read -r example; do
        [ -n "${example}" ] && targets+=("${example}")
    done < <(find "${REPO_ROOT}/backends/cuda/examples" -maxdepth 1 -name '*.cpp' \
                 -exec basename {} .cpp \; 2>/dev/null | sort)
    if [ "${benchmarks}" = "ON" ]; then
        # Derived, not listed, for the same reason the suites are: a benchmark
        # added the intended way must not be one this gate silently skips.
        #
        # The text names every benchmark the file DECLARES; CMake knows which of
        # them this configuration CREATED. A target inside an `if()` guard -- one
        # that needs an OpenCV with the cv::cuda modules, say -- is declared and
        # not created, and asking for it by name fails the build with "No rule to
        # make target". So the declared list is intersected with CMake's own, and
        # whatever the guard left out is printed by name rather than skipped in
        # silence or excluded by a name written here.
        local configured
        configured="$(cmake --build "${dir}" --target help 2>/dev/null)"
        local built=() unconfigured=()
        while IFS= read -r bench; do
            [ -n "${bench}" ] || continue
            if grep -qw -- "${bench}" <<<"${configured}"; then
                built+=("${bench}")
            else
                unconfigured+=("${bench}")
            fi
        done < <(sed -n 's/^[[:space:]]*add_executable([[:space:]]*\([A-Za-z0-9_][A-Za-z0-9_]*\).*/\1/p' \
                     "${REPO_ROOT}/backends/cuda/benchmark/CMakeLists.txt" 2>/dev/null)
        if [ ${#built[@]} -eq 0 ]; then
            echo "  NO BENCHMARK TARGETS CONFIGURED -- the derivation or CMake's target"
            echo "  list is broken, and this gate would compile no benchmark arm at all."
            exit 1
        fi
        targets+=("${built[@]}")
        echo "  benchmark targets: ${built[*]}"
        if [ ${#unconfigured[@]} -gt 0 ]; then
            echo "  declared but NOT CONFIGURED here (their CMake guard is off): ${unconfigured[*]}"
        fi
    fi

    echo "  building..."
    if ! cmake --build "${dir}" --target "${targets[@]}" -j"$(nproc)" \
            >"${dir}.build.log" 2>&1; then
        cat "${dir}.build.log"
        echo "  BUILD FAILED (${name})"
        exit 1
    fi
    # A warning nvcc emitted but did not turn into an error (it forwards only
    # the host half to -Werror): the same belt-and-braces log scan verify.sh
    # runs.
    if grep -q "warning:" "${dir}.build.log"; then
        grep "warning:" "${dir}.build.log"
        echo "  BUILD EMITTED WARNINGS (${name})"
        exit 1
    fi

    echo "  running device-vs-host suites..."
    local counts="${dir}.checks.txt"
    : > "${counts}"
    for suite in "${SUITES[@]}"; do
        local out="${dir}.${suite}.out"
        "${dir}/backends/cuda/tests/${suite}" 2>&1 | tee "${out}"
        RC=${PIPESTATUS[0]}
        if [ ${RC} -eq 77 ]; then
            skip "built cleanly, but no CUDA device is available to run ${suite}"
        elif [ ${RC} -ne 0 ]; then
            echo "  DEVICE-VS-HOST SUITE FAILED: ${suite} (${name})"
            exit 1
        fi
        local line n k
        line="$( { grep -oE '[0-9]+/[0-9]+ checks passed' "${out}" || true; } | head -1)"
        if [ -z "${line}" ]; then
            echo "  ${suite} passed but printed no check summary -- it is not reporting"
            echo "  through tests/test_util.hpp, so its coverage cannot be counted."
            exit 1
        fi
        n="${line%%/*}"
        k="$( { grep -oE ', [0-9]+ skipped' "${out}" || true; } | head -1 | { grep -oE '[0-9]+' || true; })"
        printf '%s\t%s\t%s\n' "${suite}" "${n}" "${k:-0}" >> "${counts}"
    done
    check_floors "${name}" "${counts}"
}

# ---------------------------------------------------------------------------
# Check-count floors
#
# Exit codes say whether every check that RAN passed; they say nothing about
# how many ran. An edit that narrows a width sweep, drops a border mode from a
# loop or removes a MorphOp still exits 0 -- the failure tests/expected-checks.txt
# exists to catch on the host, and one this backend had no defence against while
# its suites went from 784 checks to tens of thousands.
#
# The floors are PER CONFIGURATION because the two legitimately differ: a
# deliberate domain violation that trips an assertion can only have its error
# return checked where the assertion is compiled out, so Debug reports one fewer
# check per such call (BINCV_CHECK_EQ_UNLESS_CHECKED in tests/test_util.hpp).
# ---------------------------------------------------------------------------
baseline_rows() {   # config -> "suite<TAB>checks<TAB>skipped"
    { grep -v '^[[:space:]]*#' "${BASELINE_FILE}" 2>/dev/null || true; } \
        | awk -F'\t' -v c="$1" 'NF>=4 && $1==c {printf "%s\t%s\t%s\n", $2, $3, $4}'
}

FLOOR_DRIFT=0
check_floors() {
    local name="$1" counts="$2" drift=0 rises=""
    while IFS=$'\t' read -r b_suite b_checks b_skipped; do
        [ -n "${b_suite}" ] || continue
        local row got_c got_s
        row="$(awk -F'\t' -v s="${b_suite}" '$1==s {print; exit}' "${counts}")"
        if [ -z "${row}" ]; then
            echo "  MISSING SUITE ${b_suite} -- ${BASELINE_FILE#"${REPO_ROOT}"/} lists it for"
            echo "  '${name}' and it did not run. That is ${b_checks} checks gone."
            drift=1
            continue
        fi
        got_c="$(cut -f2 <<<"${row}")"
        got_s="$(cut -f3 <<<"${row}")"
        if [ "${got_c}" -lt "${b_checks}" ] || [ "${got_s}" -lt "${b_skipped}" ]; then
            echo "  CHECK COUNT DROPPED  ${b_suite} (${name}): ${got_c}+${got_s}s,"
            echo "  expected at least ${b_checks}+${b_skipped}s."
            drift=1
        elif [ "${got_c}" -gt "${b_checks}" ] || [ "${got_s}" -gt "${b_skipped}" ]; then
            rises="${rises}    ${b_suite}: ${b_checks}+${b_skipped}s -> ${got_c}+${got_s}s\n"
        fi
    done < <(baseline_rows "${name}")

    # A suite with no row has no floor, so it could lose every assertion it has
    # without this gate noticing.
    while IFS=$'\t' read -r r_suite _ _; do
        [ -n "${r_suite}" ] || continue
        if ! baseline_rows "${name}" | awk -F'\t' -v s="${r_suite}" '$1==s{f=1} END{exit !f}'; then
            echo "  UNRECORDED SUITE ${r_suite} ran in '${name}' but has no floor."
            drift=1
        fi
    done < "${counts}"

    if [ -n "${rises}" ]; then
        echo "  check counts ROSE (not a failure):"
        printf "%b" "${rises}"
        echo "    raise the floor with: ./scripts/verify_cuda.sh --update-checks-baseline"
    fi
    if [ ${drift} -eq 1 ]; then
        if [ ${UPDATE_BASELINE} -eq 1 ]; then
            echo "  (--update-checks-baseline: recording the current counts anyway)"
        else
            echo "  CHECK-COUNT FLOOR FAILED (${name})"
            FLOOR_DRIFT=1
        fi
    fi
    local total
    total="$(awk -F'\t' '{t+=$2} END{print t+0}' "${counts}")"
    echo "  ${name}: ${total} checks across ${#SUITES[@]} suites"
}

run_configuration release Release ON
run_configuration debug   Debug   OFF

if [ ${UPDATE_BASELINE} -eq 1 ]; then
    # Keep the header block -- it explains why the configurations differ, and
    # is worth more than the numbers under it.
    {
        awk 'BEGIN{done=0} !done && (/^[[:space:]]*#/ || /^[[:space:]]*$/) {print; next} {done=1}' \
            "${BASELINE_FILE}" 2>/dev/null
        for cfg in release debug; do
            while IFS=$'\t' read -r s c k; do
                [ -n "${s}" ] && printf '%s\t%s\t%s\t%s\n' "${cfg}" "${s}" "${c}" "${k}"
            done < "${BUILD_DIR}-${cfg}.checks.txt"
            echo
        done
    } > "${BASELINE_FILE}.new" && mv "${BASELINE_FILE}.new" "${BASELINE_FILE}"
    echo
    echo "  floors rewritten: ${BASELINE_FILE#"${REPO_ROOT}"/} -- review and commit the diff"
fi

if [ ${FLOOR_DRIFT} -eq 1 ]; then
    echo
    echo "  CUDA BACKEND NOT VERIFIED: a suite ran fewer checks than its floor."
    exit 1
fi

echo
echo "  CUDA BACKEND VERIFIED"
echo "  Built with -Werror on both host and device halves, Release and Debug;"
echo "  every device kernel matched the host library byte for byte, every suite"
echo "  met its check-count floor, and the benchmark arms compile."
echo
exit 0
