#!/usr/bin/env bash
#
# verify.sh -- everything a session should check before committing (T1.8).
#
# Builds and tests every supported configuration, fails on any compiler warning,
# prints a summary table, and exits non-zero if anything at all went wrong.
#
#   ./scripts/verify.sh                        # the gate: clean builds, warnings fatal
#   ./scripts/verify.sh --incremental          # fast inner-loop re-run (see the caveat)
#   ./scripts/verify.sh --only core            # one configuration by name
#   ./scripts/verify.sh --arm                  # also run scripts/verify_arm.sh at the end
#   ./scripts/verify.sh --update-checks-baseline   # raise tests/expected-checks.txt
#
# ---------------------------------------------------------------------------
# Why a fourth configuration
#
# Until T1.8 the verification set was three RELEASE builds, so NDEBUG was defined
# in every one of them and BINCV_DEBUG_CHECKS was 0 everywhere. Half of the error
# policy -- every BINCV_ASSERT, which is to say the bounds checks in at() and
# set() and every kernel precondition -- was therefore never compiled by anything
# that could fail. `debug` closes that.
#
# It is not redundant with tests/test_assert_abort.cpp. That binary forces the
# checks on for a handful of deliberately-out-of-range cases; this configuration
# compiles them into the whole library and runs all ~1900 checks through them, so
# an assertion whose own condition is wrong -- rejecting something legal -- is
# caught. Measured, by mutating BinMat::at's bounds check to reject the last legal
# column: the Debug configuration fails 5 tests, the Release core configuration
# fails 1.
#
# Being a Debug build is not the same as HAVING the checks, and this script no
# longer assumes it: each configuration declares the BINCV_DEBUG_CHECKS and
# BINCV_EXCEPTIONS_ENABLED values it must produce, and the values are read back
# out of the built test_error binary. `CXXFLAGS=-DNDEBUG ./scripts/verify.sh
# --only debug` used to pass while compiling every assertion away; it now fails.
#
# ---------------------------------------------------------------------------
# Why warnings are fatal here and not by default
#
# Before T1.8 nothing in this project enabled a single warning flag, so the
# "must build warning-free" requirement in CLAUDE.md, TASKS.md and
# GETTING_STARTED.md passed vacuously (OVERNIGHT_LOG finding 5). The flags now
# live in bincv-cpp/cmake/BincvWarnings.cmake and this script configures every
# build with -DBINCV_WERROR=ON, so a warning stops the build here even though a
# plain `cmake --build` still only prints it.
#
# The build log is ALSO scanned for "warning:" independently of -Werror, which
# catches what -Werror does not turn into an error: linker warnings, CMake's own
# warnings, third-party targets. It does NOT catch a first-party target that
# forgets to link bincv_warnings -- such a target compiles with no warning flags,
# so it emits nothing to scan, and this script reported WARN 0 / PASS for exactly
# that case until bincv_assert_warning_policy() was added. That check is
# structural, at configure time, and the "gate self-check" phase below proves it
# still fires.
#
# ---------------------------------------------------------------------------
# What this script refuses to call a pass
#
# A gate is only worth running if it can fail, so every one of these is red:
#
#   - a configuration that ran no tests (ctest exits 0 on an empty test set)
#   - a suite that reported fewer checks than tests/expected-checks.txt records
#   - a suite that ran but has no row in that file, or a row whose suite vanished
#   - a configuration whose BINCV_DEBUG_CHECKS/BINCV_EXCEPTIONS_ENABLED are not
#     the values it is supposed to be verifying
#   - a build tree that could not be cleaned (that configuration only)
#   - either gate self-check failing to fail

set -euo pipefail

# Resolve through symlinks first: invoked as ~/bin/bincv-verify, a bare dirname
# put SRC_DIR and LOG_DIR next to the symlink and created a build-logs/ directory
# outside the repo.
SELF="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1 && readlink -f "${SELF}" >/dev/null 2>&1; then
    SELF="$(readlink -f "${SELF}")"
fi
REPO_ROOT="$(cd "$(dirname "${SELF}")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/bincv-cpp"
LOG_DIR="${SRC_DIR}/build-logs"
BASELINE_FILE="${SRC_DIR}/tests/expected-checks.txt"

if [[ ! -f "${SRC_DIR}/CMakeLists.txt" ]]; then
    echo "verify.sh: ${SRC_DIR}/CMakeLists.txt not found -- this does not look like the binCV repo" >&2
    exit 2
fi

JOBS="$( (command -v nproc >/dev/null && nproc) || echo 4)"
INCREMENTAL=0
RUN_ARM=0
UPDATE_BASELINE=0
ONLY=""

# ---------------------------------------------------------------------------
# Configurations
#
# The first is the desktop build. The other three are the embedded claim, and
# they regress silently if not run -- which is the entire reason this file exists.
# Declared before argument parsing so --help can name them.
# ---------------------------------------------------------------------------
CONFIG_NAMES=(opencv core noexcept debug)

need_value() {   # flag -- errors the way an unknown flag does, not silently
    if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "verify.sh: '$1' needs a value" >&2
        echo "verify.sh: configurations are: ${CONFIG_NAMES[*]}" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --incremental|-i) INCREMENTAL=1 ;;
        --arm)            RUN_ARM=1 ;;
        --update-checks-baseline) UPDATE_BASELINE=1 ;;
        --only)           need_value "$1" "${2:-}"; ONLY="$2"; shift ;;
        -j)               need_value "$1" "${2:-}"; JOBS="$2"; shift ;;
        -h|--help)
            sed -n '3,14p' "${SELF}" | sed 's/^# \{0,1\}//'
            echo
            echo "Configurations: ${CONFIG_NAMES[*]}"
            exit 0 ;;
        *) echo "verify.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
    shift
done

config_dir() {
    case "$1" in
        opencv)   echo "${SRC_DIR}/build" ;;
        core)     echo "${SRC_DIR}/build-core" ;;
        noexcept) echo "${SRC_DIR}/build-noexcept" ;;
        debug)    echo "${SRC_DIR}/build-debug" ;;
    esac
}

config_args() {
    # BINCV_BUILD_TESTS is passed explicitly rather than left to the default:
    # a build tree configured once with -DBINCV_BUILD_TESTS=OFF keeps that value
    # in its cache forever, and this script then reported PASS for a run in which
    # ctest found no tests at all.
    case "$1" in
        opencv)   echo "-DCMAKE_BUILD_TYPE=Release -DBINCV_BUILD_TESTS=ON" ;;
        core)     echo "-DCMAKE_BUILD_TYPE=Release -DBINCV_BUILD_TESTS=ON -DBINCV_USE_OPENCV=OFF" ;;
        noexcept) echo "-DCMAKE_BUILD_TYPE=Release -DBINCV_BUILD_TESTS=ON -DBINCV_USE_OPENCV=OFF -DCMAKE_CXX_FLAGS=-fno-exceptions" ;;
        debug)    echo "-DCMAKE_BUILD_TYPE=Debug -DBINCV_BUILD_TESTS=ON -DBINCV_USE_OPENCV=OFF" ;;
    esac
}

config_label() {
    case "$1" in
        opencv)   echo "Release + OpenCV" ;;
        core)     echo "Release core-only" ;;
        noexcept) echo "-fno-exceptions core" ;;
        debug)    echo "Debug core-only" ;;
    esac
}

# The error-policy macros each configuration exists to exercise. Read back out of
# the built binary rather than inferred from the CMake arguments -- an exported
# CXXFLAGS=-DNDEBUG turned the Debug configuration into a second copy of `core`
# and nothing noticed, which is the same vacuity T1.8 was written to remove.
config_expect_macros() {
    case "$1" in
        opencv|core) echo "BINCV_EXCEPTIONS_ENABLED=1 BINCV_DEBUG_CHECKS=0" ;;
        noexcept)    echo "BINCV_EXCEPTIONS_ENABLED=0 BINCV_DEBUG_CHECKS=0" ;;
        debug)       echo "BINCV_EXCEPTIONS_ENABLED=1 BINCV_DEBUG_CHECKS=1" ;;
    esac
}

if [[ -n "${ONLY}" ]]; then
    found=0
    for n in "${CONFIG_NAMES[@]}"; do if [[ "$n" == "${ONLY}" ]]; then found=1; fi; done
    if [[ $found -eq 0 ]]; then
        echo "verify.sh: --only expects one of: ${CONFIG_NAMES[*]}" >&2
        exit 2
    fi
    CONFIG_NAMES=("${ONLY}")
fi

if [[ ${UPDATE_BASELINE} -eq 1 && -n "${ONLY}" ]]; then
    echo "verify.sh: --update-checks-baseline rewrites every configuration's rows," >&2
    echo "           so it cannot be combined with --only (the others' rows would be lost)." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Removing a build tree is the only destructive thing this script does, so it
# refuses to remove anything that is not demonstrably one. Returns non-zero
# instead of exiting: one unusable build directory should fail its own
# configuration, not abort the run before the other three are built.
remove_build_dir() {
    local dir="$1"
    [[ -d "${dir}" ]] || return 0
    if [[ ! -f "${dir}/CMakeCache.txt" ]]; then
        echo "  '${dir}' has no CMakeCache.txt, so it is not a build tree and this"
        echo "  script will not delete it. Remove it by hand if it is yours:"
        echo "      rm -rf '${dir}'"
        return 1
    fi
    rm -rf "${dir}"
}

# Compiler warnings, excluding anything that is not ours to fix. A downloaded
# googletest is built from source in the build tree and its warnings are not
# actionable here; binCV's own headers are what this gate is about.
#
# Written to always succeed and always print a number: `grep` exits non-zero when
# it matches nothing, which under `set -o pipefail` would abort the run on the
# happy path -- a verification script that dies when there is nothing wrong.
# `[-Werror=` is matched as well as `warning:` on purpose: with -Werror on, GCC
# reports a warning as `error: ... [-Werror=unused-variable]`, so a scan for the
# word "warning" alone would report zero warnings for a build that failed on one.
warning_lines() {
    local log="$1"
    { grep -E "(^|[[:space:]])warning:|: warning [A-Z]|\[-Werror" "${log}" 2>/dev/null || true; } \
        | { grep -v "_deps/googletest" || true; } \
        | { grep -v "/usr/include/" || true; } \
        | sort -u
}

count_warnings() {
    warning_lines "$1" | wc -l | tr -d ' '
}

# Print at most N lines, indented. Deliberately NOT `| head -n N`: head closes the
# pipe early, the producer dies of SIGPIPE, and under `set -o pipefail` plus
# `set -e` that aborted the entire run with exit 141 -- no summary table, no
# "VERIFICATION FAILED", remaining configurations never built. Reproduced 3/3 on a
# build with ~300 distinct diagnostics, i.e. exactly when the gate had the most to
# say. `sed -n '1,Np'` reads its whole input, so nothing gets a closed pipe.
print_first() {
    local n="$1"
    sed -n "1,${n}{s/^/    /;p}"
}

# A content hash of everything that determines the check counts. Stamped into
# build-logs/checks-<cfg>.txt so scripts/verify_arm.sh can tell a reference that
# describes the tree it just built from one left over by an earlier, greener run.
# git rev-parse HEAD would not do: the counts follow the working tree, dirty or not.
source_stamp() {
    local root="$1"
    ( cd "${root}" && find include tests -type f \
        \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' -o -name '*.txt' -o -name '*.cmake' \) \
        -print0 | LC_ALL=C sort -z | xargs -0 sha1sum ) | sha1sum | cut -d' ' -f1
}

# The suites a configuration actually registered, and where their binaries are.
#
# Derived from ctest rather than hardcoded. A list written out by hand here went
# out of date the moment a suite was added the intended way (appending to
# BINCV_CORE_TESTS): ctest picked the new suite up, this harvest did not, and the
# CHECKS column stayed put while 100 new assertions ran.
#
# Excluded: WILL_FAIL tests (test_harness_failing is supposed to fail) and tests
# whose command is not a single executable (death tests run cmake -P).
list_suites() {
    local build_dir="$1"
    (cd "${build_dir}" && ctest --show-only=json-v1 2>/dev/null) | python3 -c '
import json, os, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(0)
TRUE = ("TRUE", "ON", "1", "YES", "Y")
for t in doc.get("tests", []):
    props = {p.get("name"): p.get("value") for p in t.get("properties", [])}
    if str(props.get("WILL_FAIL", "")).upper() in TRUE:
        continue
    cmd = t.get("command") or []
    if len(cmd) != 1 or not os.path.isfile(cmd[0]):
        continue
    print("%s\t%s" % (t["name"], cmd[0]))
' || true
}

# ctest exits 0 when it finds no tests at all, which is how a build configured
# with -DBINCV_BUILD_TESTS=OFF once reported PASS with "No tests were found!!!"
# in its log. Available since CMake 3.20; the tests=="-" guard further down is
# the fallback for anything older.
CTEST_NO_TESTS_FLAG=""
if ctest --help 2>/dev/null | grep -q -- "--no-tests="; then
    CTEST_NO_TESTS_FLAG="--no-tests=error"
fi

# list_suites parses ctest's JSON with python3. Say so plainly rather than
# harvesting zero suites and blaming the tests.
if ! command -v python3 >/dev/null 2>&1; then
    echo "verify.sh: python3 is required (it reads the suite list out of ctest's JSON)" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Baseline (tests/expected-checks.txt)
# ---------------------------------------------------------------------------
baseline_rows() {   # config -> "suite<TAB>checks<TAB>skipped"
    local cfg="$1"
    { grep -v '^[[:space:]]*#' "${BASELINE_FILE}" || true; } \
        | awk -F'\t' -v c="${cfg}" 'NF>=4 && $1==c {printf "%s\t%s\t%s\n", $2, $3, $4}'
}

RESULT_LABEL=()
RESULT_BACKEND=()
RESULT_TESTS=()
RESULT_CHECKS=()
RESULT_WARNINGS=()
RESULT_STATUS=()
RESULT_SECONDS=()
FAILED=0

mkdir -p "${LOG_DIR}"
STAMP="$(source_stamp "${SRC_DIR}")"

echo
echo "binCV verification -- $(date '+%Y-%m-%d %H:%M:%S')"
echo "  repo      : ${REPO_ROOT}"
echo "  compiler  : $(${CXX:-c++} --version 2>/dev/null | head -1)"
echo "  jobs      : ${JOBS}"
echo "  sources   : ${STAMP}  (include/ + tests/ content hash)"
if [[ ${INCREMENTAL} -eq 1 ]]; then
    echo "  mode      : INCREMENTAL -- fast, but the warning scan is NOT trustworthy:"
    echo "              an unchanged translation unit is not recompiled, so a warning"
    echo "              it emits is not re-emitted. Re-run without --incremental"
    echo "              before committing."
else
    echo "  mode      : clean build (every translation unit recompiled, so every"
    echo "              warning is emitted where the scan can see it)"
fi
echo

# ---------------------------------------------------------------------------
# Gate self-check: prove the warning gate can still fail
#
# Both halves of the warning policy are supposed to reject something, and neither
# had ever been observed rejecting anything -- which is how the pre-T1.8 gate
# passed for months while enabling no warning flags at all. These two builds are
# supposed to FAIL; the run is red if either one succeeds.
# ---------------------------------------------------------------------------
SELFCHECK_STATUS="PASS"
SELFCHECK_DETAIL=""
run_gate_selfcheck() {
    local base=("-DBINCV_USE_OPENCV=OFF" "-DBINCV_BUILD_TESTS=OFF"
                "-DBINCV_BUILD_BENCHMARKS=OFF" "-DBINCV_WERROR=ON"
                "-DCMAKE_BUILD_TYPE=Release")
    local dir_w="${LOG_DIR}/selfcheck-werror"
    local dir_u="${LOG_DIR}/selfcheck-unwired"
    local log="${LOG_DIR}/selfcheck.log"
    rm -rf "${dir_w}" "${dir_u}"
    : > "${log}"

    # 1. A wired target whose source warns must not build.
    if cmake -S "${SRC_DIR}" -B "${dir_w}" "${base[@]}" -DBINCV_SELFTEST_WARNING=ON >> "${log}" 2>&1 \
       && cmake --build "${dir_w}" >> "${log}" 2>&1; then
        SELFCHECK_STATUS="FAILED"
        SELFCHECK_DETAIL="a target with three deliberate warnings built cleanly -- -Werror is not reaching first-party sources"
        return 1
    fi

    # 2. A target that omits bincv_warnings must not configure.
    if cmake -S "${SRC_DIR}" -B "${dir_u}" "${base[@]}" -DBINCV_SELFTEST_UNWIRED=ON >> "${log}" 2>&1; then
        SELFCHECK_STATUS="FAILED"
        SELFCHECK_DETAIL="a target that does not link bincv_warnings configured successfully -- bincv_assert_warning_policy() is not enforcing the wiring"
        return 1
    fi
    if ! grep -q "binCV warning policy" "${log}"; then
        SELFCHECK_STATUS="FAILED"
        SELFCHECK_DETAIL="the unwired target failed to configure, but not with the warning-policy error -- something else broke"
        return 1
    fi

    # 3. This script must survive having a lot to report.
    #
    # The diagnostic printer used to be `... | head -15 | sed`, which gives its
    # producer a SIGPIPE; under `set -o pipefail` plus `set -e` that killed the
    # whole run with exit 141 -- no table, no "VERIFICATION FAILED", remaining
    # configurations never built -- and it did so precisely when the build had the
    # most warnings to show. Reproduced 3/3 at ~300 diagnostics.
    if ! ( set -eo pipefail; seq 1 5000 | print_first 15 > /dev/null ); then
        SELFCHECK_STATUS="FAILED"
        SELFCHECK_DETAIL="print_first died on a long input -- this script would abort mid-run on a build with many diagnostics"
        return 1
    fi

    rm -rf "${dir_w}" "${dir_u}"
    return 0
}

echo "=============================================================="
echo "  Gate self-check"
echo "=============================================================="
selfcheck_started=${SECONDS}
if run_gate_selfcheck; then
    echo "  -Werror rejects a warning in a wired target        ok"
    echo "  configure rejects a target missing bincv_warnings  ok"
    echo "  this script survives a long diagnostic list        ok"
else
    echo "  GATE SELF-CHECK FAILED: ${SELFCHECK_DETAIL}"
    echo "  see ${LOG_DIR}/selfcheck.log"
    FAILED=1
fi
SELFCHECK_SECONDS=$((SECONDS - selfcheck_started))
echo "  -> ${SELFCHECK_STATUS}  (${SELFCHECK_SECONDS}s)"
echo

for name in "${CONFIG_NAMES[@]}"; do
    label="$(config_label "${name}")"
    build_dir="$(config_dir "${name}")"
    cfg_log="${LOG_DIR}/${name}-configure.log"
    build_log="${LOG_DIR}/${name}-build.log"
    test_log="${LOG_DIR}/${name}-test.log"
    checks_file="${LOG_DIR}/checks-${name}.txt"
    checks_tmp="${LOG_DIR}/.checks-${name}.partial"
    started=${SECONDS}

    echo "=============================================================="
    echo "  ${label}"
    echo "=============================================================="

    # An out-of-date reference is worse than none: scripts/verify_arm.sh diffs
    # against these, and one left behind by an earlier green run described a tree
    # that no longer exists. Removed now, rewritten only if this configuration
    # passes.
    rm -f "${checks_file}" "${checks_tmp}"

    status="PASS"
    backend="?"
    tests="-"
    checks="-"
    warnings="-"

    if [[ ${INCREMENTAL} -eq 0 ]]; then
        if ! remove_build_dir "${build_dir}"; then
            status="DIRTY-BUILD-DIR"
        fi
    fi

    # --- configure ---------------------------------------------------------
    if [[ "${status}" == "PASS" ]]; then
        # shellcheck disable=SC2046  # config_args is a deliberate word-split list
        if cmake -S "${SRC_DIR}" -B "${build_dir}" $(config_args "${name}") \
                 -DBINCV_WERROR=ON > "${cfg_log}" 2>&1; then
            backend="$( { grep -oE 'Tests: +.*' "${cfg_log}" || true; } | head -1 | sed 's/Tests: *//')"
            if [[ -z "${backend}" ]]; then backend="?"; fi
        else
            echo "  CONFIGURE FAILED -- see ${cfg_log}"
            tail -20 "${cfg_log}" | sed 's/^/    /'
            status="CONFIG-FAIL"
        fi
    fi

    # The OpenCV configuration silently degrades to core-only when OpenCV is not
    # installed, which would let the interop suite vanish while the gate stayed
    # green. Say so rather than report a pass for tests that did not run.
    if [[ "${status}" == "PASS" && "${name}" == "opencv" ]] \
       && ! grep -q "OpenCV:     YES" "${cfg_log}"; then
        echo "  OpenCV not found -- the interop suite is NOT being verified."
        status="NO-OPENCV"
    fi

    # --- build -------------------------------------------------------------
    if [[ "${status}" == "PASS" || "${status}" == "NO-OPENCV" ]]; then
        if cmake --build "${build_dir}" -j"${JOBS}" > "${build_log}" 2>&1; then
            :
        else
            echo "  BUILD FAILED -- see ${build_log}"
            { grep -E "error:" "${build_log}" || true; } | print_first 15
            status="BUILD-FAIL"
        fi
        warnings="$(count_warnings "${build_log}")"
        if [[ "${warnings}" != "0" ]]; then
            # -Werror should have turned these into build failures already; if one
            # reaches here it came from outside the -Werror path (a linker or CMake
            # warning), and the gate still refuses it.
            echo "  ${warnings} WARNING(S) -- the build is not warning-free:"
            { warning_lines "${build_log}" || true; } | print_first 15
            if [[ "${status}" == "PASS" ]]; then status="WARNINGS"; fi
        fi
    fi

    # --- test --------------------------------------------------------------
    if [[ "${status}" == "PASS" || "${status}" == "NO-OPENCV" || "${status}" == "WARNINGS" ]]; then
        if (cd "${build_dir}" && ctest --output-on-failure ${CTEST_NO_TESTS_FLAG}) > "${test_log}" 2>&1; then
            :
        else
            echo "  TESTS FAILED -- see ${test_log}"
            { grep -E "tests failed|\(Failed\)|\(Subprocess aborted\)|No tests were found" "${test_log}" || true; } \
                | print_first 15
            status="TEST-FAIL"
        fi
        # ctest prints "100% tests passed, 0 tests failed out of 43"; report the
        # passing count over the total.
        summary_line="$( { grep -oE '[0-9]+ tests? failed out of [0-9]+' "${test_log}" || true; } | head -1)"
        if [[ -n "${summary_line}" ]]; then
            failed_n="$(echo "${summary_line}" | awk '{print $1}')"
            total_n="$(echo "${summary_line}" | awk '{print $NF}')"
            tests="$((total_n - failed_n))/${total_n}"
        else
            tests="-"
        fi
    fi

    # --- check totals, per suite and summed --------------------------------
    #
    # ctest reports how many test CASES ran; the number this project has tracked
    # across the T1.x tasks is how many CHECKS ran (845 / 544 / 443), and it is the
    # one that catches a suite quietly losing assertions. The suites print it on
    # their last line in both harness backends, so harvest it by running them once
    # more -- under a second in total, and worth it to keep "the test count did not
    # drop" a measured claim rather than a printed number nobody compares.
    observed_macros=""
    if [[ "${status}" == "PASS" || "${status}" == "NO-OPENCV" || "${status}" == "WARNINGS" ]]; then
        total=0
        skipped=0
        any=0
        {
            printf '# stamp\t%s\n' "${STAMP}"
            printf '# config\t%s\n' "${name}"
            printf '# generated\t%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
        } > "${checks_tmp}"

        while IFS=$'\t' read -r suite_name suite_path; do
            [[ -n "${suite_name}" ]] || continue
            out="$( (cd "${build_dir}" && "${suite_path}" 2>/dev/null) || true)"
            line="$( { echo "${out}" | grep -oE '[0-9]+/[0-9]+ checks passed' || true; } | head -1)"
            if [[ -z "${line}" ]]; then
                echo "  ${suite_name} ran but printed no check summary -- it is not"
                echo "  reporting through tests/test_util.hpp, so its coverage cannot be counted."
                status="CHECKS-DRIFT"
                continue
            fi
            n="${line%%/*}"
            total=$((total + n))
            # Skips are reported separately on purpose: without exceptions
            # BINCV_CHECK_THROWS cannot evaluate its expression, and a run that
            # counted 1815 where another counted 1892 would read as lost coverage
            # rather than as 60 checks that moved to the death tests. Summing them
            # here is what makes the two columns comparable.
            s="$( { echo "${out}" | grep -oE ', [0-9]+ skipped' || true; } | head -1 | { grep -oE '[0-9]+' || true; })"
            if [[ -n "${s}" ]]; then skipped=$((skipped + s)); fi
            printf '%s\t%s\t%s\n' "${suite_name}" "${n}" "${s:-0}" >> "${checks_tmp}"
            any=1
            # The error-policy suite prints the macro values its translation unit
            # actually compiled with. That is the only direct evidence this script
            # has that the configuration is the one it claims to be.
            #
            # The header line is what distinguishes it from test_error_checked,
            # which prints the same two macros but #undef NDEBUG's first and would
            # report BINCV_DEBUG_CHECKS=1 in every configuration. Do not add that
            # header there.
            case "${out}" in
                *"Error policy configuration"*)
                    observed_macros="$(echo "${out}" \
                        | { grep -oE 'BINCV_(EXCEPTIONS_ENABLED|DEBUG_CHECKS) *= *[01]' || true; } \
                        | tr -d ' ' | sort -u | tr '\n' ' ')" ;;
            esac
        done < <(list_suites "${build_dir}")

        if [[ ${any} -eq 1 ]]; then
            if [[ ${skipped} -gt 0 ]]; then
                checks="${total}+${skipped}s"
            else
                checks="${total}"
            fi
        fi
    fi

    # --- the configuration must have actually verified something -----------
    if [[ "${status}" == "PASS" || "${status}" == "NO-OPENCV" ]]; then
        if [[ "${tests}" == "-" || "${checks}" == "-" || "${backend}" == "?" ]]; then
            echo "  NOTHING WAS VERIFIED -- ctest ran ${tests} cases and ${checks} checks."
            echo "  ctest exits 0 on an empty test set, so this used to read as a pass."
            status="NO-TESTS"
        fi
    fi

    # --- the configuration must be the configuration it claims to be -------
    if [[ "${status}" == "PASS" || "${status}" == "NO-OPENCV" ]]; then
        want_macros="$(config_expect_macros "${name}")"
        for pair in ${want_macros}; do
            if [[ " ${observed_macros}" != *" ${pair} "* ]]; then
                echo "  CONFIGURATION MISMATCH -- expected ${pair}, and the built"
                echo "  test_error reports: ${observed_macros:-nothing}"
                echo "  (an exported CXXFLAGS or a stale cache can do this; the"
                echo "  '${name}' configuration is then not verifying what it exists for.)"
                status="MACRO-MISMATCH"
            fi
        done
    fi

    # --- the check counts must not have dropped ----------------------------
    if [[ "${status}" == "PASS" || "${status}" == "NO-OPENCV" ]]; then
        drift=0
        rises=""
        while IFS=$'\t' read -r b_suite b_checks b_skipped; do
            [[ -n "${b_suite}" ]] || continue
            row="$(awk -F'\t' -v s="${b_suite}" '$1==s {print; exit}' "${checks_tmp}")"
            if [[ -z "${row}" ]]; then
                echo "  MISSING SUITE ${b_suite} -- tests/expected-checks.txt lists it for"
                echo "  '${name}', and it did not run. That is ${b_checks} checks gone."
                drift=1
                continue
            fi
            got_checks="$(echo "${row}" | awk -F'\t' '{print $2}')"
            got_skipped="$(echo "${row}" | awk -F'\t' '{print $3}')"
            if [[ "${got_checks}" -lt "${b_checks}" || "${got_skipped}" -lt "${b_skipped}" ]]; then
                echo "  CHECK COUNT DROPPED  ${b_suite}: ${got_checks}+${got_skipped}s,"
                echo "  expected at least ${b_checks}+${b_skipped}s (tests/expected-checks.txt)."
                drift=1
            elif [[ "${got_checks}" -gt "${b_checks}" || "${got_skipped}" -gt "${b_skipped}" ]]; then
                rises="${rises}    ${b_suite}: ${b_checks}+${b_skipped}s -> ${got_checks}+${got_skipped}s\n"
            fi
        done < <(baseline_rows "${name}")

        # A suite with no row has no floor, so it could lose every assertion it has
        # without this gate noticing. Adding one is a one-line reviewed edit.
        while IFS=$'\t' read -r r_suite _ _; do
            [[ -n "${r_suite}" && "${r_suite}" != \#* ]] || continue
            if ! baseline_rows "${name}" | awk -F'\t' -v s="${r_suite}" '$1==s{f=1} END{exit !f}'; then
                echo "  UNRECORDED SUITE ${r_suite} ran in '${name}' but has no row in"
                echo "  tests/expected-checks.txt, so nothing would notice it losing checks."
                drift=1
            fi
        done < "${checks_tmp}"

        if [[ -n "${rises}" ]]; then
            echo "  check counts ROSE (not a failure):"
            printf "%b" "${rises}"
            echo "    raise the floor with: ./scripts/verify.sh --update-checks-baseline"
        fi
        if [[ ${drift} -eq 1 ]]; then
            if [[ ${UPDATE_BASELINE} -eq 1 ]]; then
                echo "  (--update-checks-baseline: recording the current counts anyway)"
            else
                status="CHECKS-DRIFT"
            fi
        fi
    fi

    # The reference scripts/verify_arm.sh diffs its emulated aarch64 run against
    # is published only by a configuration that passed. A partial or stale file
    # made that script report DIFFERS for two configurations that were both fine.
    if [[ -f "${checks_tmp}" ]]; then
        if [[ "${status}" == "PASS" ]]; then
            mv "${checks_tmp}" "${checks_file}"
        else
            rm -f "${checks_tmp}"
        fi
    fi

    elapsed=$((SECONDS - started))
    if [[ "${status}" != "PASS" ]]; then FAILED=1; fi
    echo "  -> ${status}  (${elapsed}s)"
    echo

    RESULT_LABEL+=("${label}")
    RESULT_BACKEND+=("${backend}")
    RESULT_TESTS+=("${tests}")
    RESULT_CHECKS+=("${checks}")
    RESULT_WARNINGS+=("${warnings}")
    RESULT_STATUS+=("${status}")
    RESULT_SECONDS+=("${elapsed}")
done

# ---------------------------------------------------------------------------
# Baseline update
# ---------------------------------------------------------------------------
if [[ ${UPDATE_BASELINE} -eq 1 ]]; then
    echo "=============================================================="
    echo "  Updating ${BASELINE_FILE}"
    echo "=============================================================="
    missing=0
    for name in "${CONFIG_NAMES[@]}"; do
        [[ -s "${LOG_DIR}/checks-${name}.txt" ]] || { echo "  ${name}: no counts harvested"; missing=1; }
    done
    if [[ ${missing} -eq 1 ]]; then
        echo "  refusing to rewrite the baseline from an incomplete run"
        FAILED=1
    else
        # Keep everything above the first data row -- that block is the file's
        # explanation of why the four configurations legitimately differ, and it
        # is worth more than the numbers under it.
        { awk 'BEGIN{done=0} !done && (/^[[:space:]]*#/ || /^[[:space:]]*$/) {print; next} {done=1}' "${BASELINE_FILE}"
          for name in "${CONFIG_NAMES[@]}"; do
              while IFS=$'\t' read -r s c k; do
                  [[ -n "${s}" && "${s}" != \#* ]] || continue
                  printf '%s\t%s\t%s\t%s\n' "${name}" "${s}" "${c}" "${k}"
              done < "${LOG_DIR}/checks-${name}.txt"
              echo
          done
        } > "${BASELINE_FILE}.new"
        mv "${BASELINE_FILE}.new" "${BASELINE_FILE}"
        echo "  written. Review and commit the diff with the change that earned it."
    fi
    echo
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "================================================================================"
printf "  %-20s  %-27s  %7s  %10s  %4s  %5s  %s\n" \
       CONFIGURATION "TEST BACKEND" "CTEST" "CHECKS" "WARN" "TIME" "RESULT"
echo "--------------------------------------------------------------------------------"
printf "  %-20s  %-27s  %7s  %10s  %4s  %4ss  %s\n" \
    "Gate self-check" "warning policy" "3/3" "-" "-" "${SELFCHECK_SECONDS}" "${SELFCHECK_STATUS}"
for i in "${!RESULT_LABEL[@]}"; do
    printf "  %-20s  %-27s  %7s  %10s  %4s  %4ss  %s\n" \
        "${RESULT_LABEL[$i]}" "${RESULT_BACKEND[$i]}" "${RESULT_TESTS[$i]}" \
        "${RESULT_CHECKS[$i]}" "${RESULT_WARNINGS[$i]}" "${RESULT_SECONDS[$i]}" \
        "${RESULT_STATUS[$i]}"
done
echo "================================================================================"
echo "  CHECKS is assertions executed, not ctest cases; '+Ns' is checks the"
echo "  configuration cannot express in-process and covers as death tests instead."
echo "  Floors live in bincv-cpp/tests/expected-checks.txt; a drop there is red."
echo "  logs: ${LOG_DIR}"

ARM_NOTE=""
if [[ ${RUN_ARM} -eq 1 ]]; then
    echo
    # verify_arm.sh exits 77 when it could not run at all. That is not a failure
    # and not a pass: printing "aarch64: OK" under its own "aarch64 correctness
    # was NOT verified" message is the one thing this must not do.
    arm_rc=0
    BINCV_ARM_REQUIRE_REFERENCE=1 "${REPO_ROOT}/scripts/verify_arm.sh" || arm_rc=$?
    case ${arm_rc} in
        0)  echo "  aarch64: OK" ;;
        77) echo "  aarch64: SKIPPED -- NOT verified (see the message above)"
            ARM_NOTE="skipped" ;;
        *)  echo "  aarch64: FAILED"
            FAILED=1 ;;
    esac
fi

echo
if [[ ${FAILED} -eq 0 ]]; then
    echo "  ALL CONFIGURATIONS GREEN"
    if [[ ${RUN_ARM} -eq 0 || -n "${ARM_NOTE}" ]]; then
        echo "  x86_64 only. ./scripts/verify_arm.sh checks the primary target"
        echo "  (aarch64) for correctness under emulation -- cheap, and the bugs it"
        echo "  finds are expensive to find later."
    fi
    echo
    exit 0
fi

echo "  VERIFICATION FAILED"
echo
exit 1
