#!/usr/bin/env bash
#
# verify_arm.sh -- aarch64 correctness verification under emulation (T1.9).
#
# aarch64 is the primary target (ARCHITECTURE 2). The bugs that only appear there
# -- type-width assumptions, char signedness, alignment faults, intrinsics that
# fail to compile -- are cheap to catch now and expensive to attribute later.
# This runs the core-only and no-exceptions suites inside an emulated aarch64
# container so that every commit can be checked against the real target's type
# model without ARM hardware.
#
#   ./scripts/verify_arm.sh
#
# CORRECTNESS ONLY. Nothing measured in this environment is a timing result --
# see the banner the script prints, and EXPERIMENTS.md "Measurement platforms".
#
# EXIT CODES
#   0   aarch64 verified
#   1   aarch64 verification FAILED
#   77  could not run at all (no Docker, no arm64 emulation) -- NOT a pass.
#       scripts/verify.sh branches on 77 so that a skipped run cannot be printed
#       as "aarch64: OK" directly underneath this script saying it verified
#       nothing, which is what it used to do.
#
# ---------------------------------------------------------------------------
# Why this compiles directly with g++ instead of configuring CMake
#
# arm64v8/gcc:12 ships a compiler and make, and no cmake. Installing one means
# apt inside an emulated container on every run: slow, and it puts a network
# dependency on the gate that is meant to catch problems before they are
# expensive. Compiling the suites directly needs neither, and it keeps this
# script hermetic -- which matters most for exactly the configuration whose claim
# is that binCV needs nothing but a C++17 compiler.
#
# The cost is that the Google Test backend is not exercised here. That is the
# right thing to drop: the harness is not what aarch64 is being asked about, the
# library is.
#
# What IS covered is derived from tests/CMakeLists.txt rather than listed here.
# A hardcoded suite list meant a suite added the intended way -- appending to
# BINCV_CORE_TESTS -- was never compiled under aarch64, was left out of the
# comparison, and the run still printed "identical to x86_64".
#
# The build also runs with the full warning set and -Werror. An aarch64 compiler
# disagrees with an x86_64 one about the width of long and the signedness of
# char, so -Wconversion here is not a repeat of the x86 run -- it is the part of
# the warning gate that only this environment can perform.

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1 && readlink -f "${SELF}" >/dev/null 2>&1; then
    SELF="$(readlink -f "${SELF}")"
fi
REPO_ROOT="$(cd "$(dirname "${SELF}")/.." && pwd)"
IMAGE="arm64v8/gcc:12"
PLATFORM="linux/arm64"

# Set by scripts/verify.sh --arm: there, a missing x86_64 reference means the
# comparison this script exists to make did not happen, and the gate should say
# so rather than pass.
REQUIRE_REFERENCE="${BINCV_ARM_REQUIRE_REFERENCE:-0}"

if [[ ! -f "${REPO_ROOT}/bincv-cpp/CMakeLists.txt" ]]; then
    echo "verify_arm.sh: ${REPO_ROOT}/bincv-cpp/CMakeLists.txt not found -- this does not look like the binCV repo" >&2
    exit 2
fi

skip() {
    echo
    echo "  SKIPPED: $1"
    echo "  aarch64 correctness was NOT verified. This is not a failure -- but it"
    echo "  is also not a pass, and the primary target is aarch64."
    echo "  (exit 77, so a caller can tell this apart from a verified pass.)"
    echo
    exit 77
}

# Must match scripts/verify.sh's source_stamp() byte for byte: the stamp written
# into build-logs/checks-<cfg>.txt is compared against the tree this run built.
source_stamp() {
    ( cd "$1" && find include tests -type f \
        \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' -o -name '*.txt' -o -name '*.cmake' \) \
        -print0 | LC_ALL=C sort -z | xargs -0 sha1sum ) | sha1sum | cut -d' ' -f1
}

echo
echo "=============================================================================="
echo "  aarch64 correctness verification (emulated)"
echo "=============================================================================="
echo
echo "  ###   DO NOT BENCHMARK IN THIS ENVIRONMENT   ###"
echo
echo "  QEMU translates instructions and models no cache hierarchy, no instruction"
echo "  latency and no memory bandwidth. Measured on this project: the same"
echo "  popcount loop takes ~15.7 ms native x86 and ~54 ms emulated -- and the"
echo "  slowdown is not uniform across instruction mixes, so A/B rankings between"
echo "  design variants can INVERT here. A number produced in this container is"
echo "  not a slow number, it is not a number at all."
echo
echo "  It answers correctness perfectly, and nothing else. Performance decisions"
echo "  close on a deployment-class Cortex-A device -- EXPERIMENTS.md, section"
echo "  \"Measurement platforms\"."
echo

# --- availability ------------------------------------------------------------
command -v docker >/dev/null 2>&1 || skip "docker is not installed"
docker info >/dev/null 2>&1 || skip "the Docker daemon is not reachable (is it running, and are you in the docker group?)"

echo "  checking that ${PLATFORM} can run here..."
if ! docker run --rm --platform "${PLATFORM}" "${IMAGE}" uname -m 2>/dev/null | grep -q aarch64; then
    skip "cannot run ${PLATFORM} images (no binfmt/qemu emulation, or ${IMAGE} is unavailable)"
fi
echo "  ok -- ${IMAGE} reports aarch64"

STAMP="$(source_stamp "${REPO_ROOT}/bincv-cpp")"
echo "  sources: ${STAMP}"
echo

# Read-only mount: this gate must not be able to write into the working tree, and
# a container writing as root into a bind mount is how a build tree ends up
# undeletable. Everything is built in the container's own /tmp.
docker run --rm -i \
    --platform "${PLATFORM}" \
    -v "${REPO_ROOT}":/src:ro \
    -w /src \
    -e "BINCV_EXPECT_STAMP=${STAMP}" \
    -e "BINCV_REQUIRE_REFERENCE=${REQUIRE_REFERENCE}" \
    "${IMAGE}" bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail

SRC=/src/bincv-cpp
OUT=/tmp/bincv-arm
LOGS=/tmp/bincv-arm-logs
mkdir -p "$OUT" "$LOGS"

WARN="-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror"
BASE="-std=c++17 -O2 -DNDEBUG -DBINCV_TEST_WITH_GTEST=0 -I$SRC/include -I$SRC/tests"

JOBS="$(nproc 2>/dev/null || echo 4)"
FAILED=0
TAB="$(printf '\t')"

echo "  compiler: $(g++ --version | head -1)"
echo "  arch:     $(uname -m)"
echo "  jobs:     $JOBS"
echo

# ---------------------------------------------------------------------------
# What to build, derived from tests/CMakeLists.txt
#
# SUITES        every target built through bincv_add_test_target() that is not
#               the deliberately-failing harness copy, plus BINCV_CORE_TESTS.
#               The OpenCV interop suite drops out on its own: it is registered
#               through a foreach over a variable, so no literal name matches.
# DEATH_BINS    the targets bincv_add_death_test() drives.
# expected.txt  target, case, expected diagnostic -- and whether the case sits
#               inside `if(BINCV_OPENCV_FOUND)`, since this container has no
#               OpenCV and those cases are not compiled in.
# ---------------------------------------------------------------------------
python3 - <<'PY' > "$LOGS/manifest.txt"
import re
src = open('/src/bincv-cpp/tests/CMakeLists.txt').read()

# Regions guarded by if(BINCV_OPENCV_FOUND) ... endif()
guarded = []
for m in re.finditer(r'if\(BINCV_OPENCV_FOUND\)', src):
    end = src.find('\nendif()', m.end())
    guarded.append((m.start(), end if end != -1 else len(src)))
def is_guarded(pos):
    return any(a <= pos <= b for a, b in guarded)

suites = []
m = re.search(r'set\(BINCV_CORE_TESTS([^)]*)\)', src)
if m:
    suites += m.group(1).split()

# Literal-named targets built through the helper (test_harness and friends).
for m in re.finditer(r'bincv_add_test_target\(\s*([A-Za-z0-9_]+)\s', src):
    name = m.group(1)
    if name.startswith('${') or is_guarded(m.start()):
        continue
    # The copy compiled with BINCV_HARNESS_EXPECT_FAILURE is supposed to fail and
    # is run separately below, so it is not a suite.
    if re.search(r'target_compile_definitions\(\s*%s\b[^)]*BINCV_HARNESS_EXPECT_FAILURE' % name, src):
        continue
    if name not in suites:
        suites.append(name)

deaths = []
for m in re.finditer(r'bincv_add_death_test\(\s*(\S+)\s+(\S+)\s*\n?\s*"([^"]*)"\s*\)', src):
    if is_guarded(m.start()):
        continue
    deaths.append(m.groups())

print("SUITES\t" + " ".join(suites))
print("DEATH_BINS\t" + " ".join(sorted({d[0] for d in deaths})))
for d in deaths:
    print("CASE\t%s\t%s\t%s" % d)
PY

SUITES="$(awk -F'\t' '$1=="SUITES"{print $2}' "$LOGS/manifest.txt")"
DEATH_BINS="$(awk -F'\t' '$1=="DEATH_BINS"{print $2}' "$LOGS/manifest.txt")"
awk -F'\t' '$1=="CASE"{printf "%s\t%s\t%s\n", $2, $3, $4}' "$LOGS/manifest.txt" > "$LOGS/expected.txt"

if [ -z "$SUITES" ] || [ -z "$DEATH_BINS" ]; then
    echo "  FAIL: could not derive the suite list from tests/CMakeLists.txt"
    exit 1
fi
echo "  suites derived from tests/CMakeLists.txt: $SUITES"
echo "  death-test binaries:                      $DEATH_BINS"
echo "  death-test cases expected (no OpenCV):    $(wc -l < "$LOGS/expected.txt")"
echo

expected_for() {   # target case -> diagnostic
    awk -F'\t' -v t="$1" -v c="$2" '$1==t && $2==c {print $3; found=1} END{if(!found) exit 1}' \
        "$LOGS/expected.txt"
}

expected_case_count() {   # target -> how many cases it must enumerate
    awk -F'\t' -v t="$1" '$1==t {n++} END{print n+0}' "$LOGS/expected.txt"
}

build_config() {
    local name="$1"; shift
    local extra="$*"
    local dir="$OUT/$name"
    mkdir -p "$dir"

    echo "  building [$name] ($extra)"
    local rc=0
    # One compile per line, run in parallel. -Werror is on, so a warning is a
    # build failure and shows up here rather than scrolling past.
    {
        for s in $SUITES $DEATH_BINS; do
            echo "g++ $BASE $WARN $extra $SRC/tests/$s.cpp -o $dir/$s"
        done
        echo "g++ $BASE $WARN $extra -DBINCV_HARNESS_EXPECT_FAILURE $SRC/tests/test_harness.cpp -o $dir/test_harness_failing"
    } | xargs -P "$JOBS" -I{} bash -c '{}' > "$LOGS/$name-build.log" 2>&1 || rc=$?

    if [ "$rc" -ne 0 ]; then
        echo "    BUILD FAILED"
        sed -n '1,40{s/^/      /;p}' "$LOGS/$name-build.log"
        return 1
    fi
    if grep -q "warning:" "$LOGS/$name-build.log"; then
        echo "    WARNINGS (aarch64 build is not warning-free)"
        grep "warning:" "$LOGS/$name-build.log" | sort -u | sed -n '1,20{s/^/      /;p}'
        return 1
    fi
    echo "    build ok, warning-free"
    return 0
}

run_config() {
    local name="$1"
    local dir="$OUT/$name"
    local rc=0
    local total=0 skipped=0

    echo "  running  [$name]"
    : > "$LOGS/$name-checks.txt"

    for s in $SUITES; do
        local out
        if ! out="$("$dir/$s" 2>&1)"; then
            echo "    FAIL $s"
            echo "$out" | tail -20 | sed 's/^/      /'
            rc=1
            continue
        fi
        local line
        line="$(echo "$out" | grep -oE '[0-9]+/[0-9]+ checks passed' | head -1 || true)"
        local sk
        sk="$(echo "$out" | grep -oE ', [0-9]+ skipped' | head -1 | grep -oE '[0-9]+' || true)"
        [ -n "$sk" ] && skipped=$((skipped + sk)) || true
        if [ -n "$line" ]; then
            total=$((total + ${line%%/*}))
            printf "    %-20s %s\n" "$s" "$line"
            printf '%s\t%s\t%s\n' "$s" "${line%%/*}" "${sk:-0}" >> "$LOGS/$name-checks.txt"
        else
            echo "    $s: no check summary printed"
            rc=1
        fi
    done

    # The harness must still be able to fail. Same claim ctest makes with
    # WILL_FAIL on x86: a non-zero exit when a check does not hold.
    if "$dir/test_harness_failing" > /dev/null 2>&1; then
        echo "    FAIL test_harness_failing exited 0 -- a failing check is not failing the build"
        rc=1
    else
        echo "    test_harness_failing exits non-zero, as it must"
    fi

    # Death tests. Each case must terminate abnormally AND name the reason --
    # the same two conditions tests/expect_fatal.cmake enforces on x86.
    #
    # The COUNT is checked too. The case list is read out of the binary's own
    # usage output, and a zero-iteration loop over a list that failed to parse
    # printed "death tests: 0 passed" and left the configuration PASSing.
    local deaths=0
    for bin in $DEATH_BINS; do
        local cases want_n got_n
        cases="$("$dir/$bin" 2>&1 | sed -n 's/^  \([A-Za-z0-9_-][A-Za-z0-9_-]*\)$/\1/p')"
        got_n="$(printf '%s\n' "$cases" | grep -c . || true)"
        want_n="$(expected_case_count "$bin")"
        if [ "$got_n" -ne "$want_n" ]; then
            echo "    FAIL $bin enumerated $got_n cases, tests/CMakeLists.txt registers $want_n"
            rc=1
        fi
        for c in $cases; do
            local wanted
            if ! wanted="$(expected_for "$bin" "$c")"; then
                echo "    FAIL $bin/$c has no expected diagnostic in tests/CMakeLists.txt"
                rc=1
                continue
            fi
            local co status=0
            co="$("$dir/$bin" "$c" 2>&1)" || status=$?
            # Killed by a signal: 128 + signum. A clean return -- any clean
            # return, including a non-zero one -- means the check stopped being
            # fatal, which is the regression these cases exist to catch.
            if [ "$status" -lt 128 ]; then
                echo "    FAIL $bin/$c returned normally (exit $status) instead of dying"
                rc=1
                continue
            fi
            case "$co" in
                *"$wanted"*) deaths=$((deaths + 1)) ;;
                *)
                    echo "    FAIL $bin/$c died without the expected diagnostic"
                    echo "         expected: $wanted"
                    rc=1 ;;
            esac
        done
    done
    local want_deaths
    want_deaths="$(wc -l < "$LOGS/expected.txt")"
    echo "    death tests: $deaths of $want_deaths passed"
    if [ "$deaths" -ne "$want_deaths" ]; then
        rc=1
    fi

    if [ "$skipped" -gt 0 ]; then
        echo "    TOTAL: $total checks, $skipped skipped (covered as death tests)"
    else
        echo "    TOTAL: $total checks"
    fi
    echo
    return $rc
}

STATUS_CORE="not run"
STATUS_NOEXC="not run"

if build_config core "" && run_config core; then
    STATUS_CORE="PASS"
else
    STATUS_CORE="FAIL"; FAILED=1
fi

if build_config noexcept "-fno-exceptions" && run_config noexcept; then
    STATUS_NOEXC="PASS"
else
    STATUS_NOEXC="FAIL"; FAILED=1
fi

# --- comparison against the x86_64 run ---------------------------------------
#
# Always reports its outcome. This block used to sit inside a test for the
# reference's existence, so on a fresh clone -- where build-logs/ is gitignored
# and absent -- the run printed PASS/PASS having compared nothing at all.
STATUS_REF="PASS"
echo "  comparing check counts against the x86_64 run"
for cfg in core noexcept; do
    ref="/src/bincv-cpp/build-logs/checks-$cfg.txt"
    mine="$LOGS/$cfg-checks.txt"

    if [ ! -f "$ref" ]; then
        echo "    $cfg: NOT PERFORMED -- no x86_64 reference (run scripts/verify.sh first)"
        if [ "${BINCV_REQUIRE_REFERENCE:-0}" = "1" ]; then
            STATUS_REF="FAIL"; FAILED=1
        elif [ "$STATUS_REF" = "PASS" ]; then
            STATUS_REF="NOT PERFORMED"
        fi
        continue
    fi

    # A reference is only a reference for the tree it was produced from. Without
    # this, a checks-*.txt left over from an earlier and greener tree was diffed
    # against the current one and reported "identical to x86_64".
    ref_stamp="$(awk -F'\t' '$1=="# stamp"{print $2}' "$ref")"
    if [ "$ref_stamp" != "${BINCV_EXPECT_STAMP:-}" ]; then
        echo "    $cfg: NOT PERFORMED -- the x86_64 reference describes a different tree"
        echo "         reference: ${ref_stamp:-<unstamped>}"
        echo "         this tree: ${BINCV_EXPECT_STAMP:-<unknown>}"
        echo "         re-run scripts/verify.sh, then this script."
        if [ "${BINCV_REQUIRE_REFERENCE:-0}" = "1" ]; then
            STATUS_REF="FAIL"; FAILED=1
        elif [ "$STATUS_REF" = "PASS" ]; then
            STATUS_REF="NOT PERFORMED"
        fi
        continue
    fi

    # Compare every suite present in both, and say plainly what is present in
    # only one. x86 also runs the OpenCV interop suite, which is deliberately out
    # of scope here; anything else appearing on one side only is a real finding,
    # not something a hardcoded filter should hide.
    grep -v '^#' "$ref" | awk -F'\t' 'NF>=3' | LC_ALL=C sort -t"$TAB" -k1,1 > /tmp/ref.txt
    LC_ALL=C sort -t"$TAB" -k1,1 "$mine" > /tmp/mine.txt
    cut -f1 /tmp/ref.txt  | LC_ALL=C sort > /tmp/ref-names.txt
    cut -f1 /tmp/mine.txt | LC_ALL=C sort > /tmp/mine-names.txt

    arm_only="$(comm -13 /tmp/ref-names.txt /tmp/mine-names.txt)"
    ref_only="$(comm -23 /tmp/ref-names.txt /tmp/mine-names.txt)"

    ok=1
    if [ -n "$arm_only" ]; then
        echo "    $cfg: suites that ran on aarch64 but not on x86_64:"
        printf '%s\n' "$arm_only" | sed 's/^/      /'
        ok=0
    fi
    if [ -n "$ref_only" ]; then
        # Expected exactly for the OpenCV interop suite. Reported either way so
        # that a core suite silently missing from the aarch64 build is visible.
        echo "    $cfg: x86_64-only suites (not built here):"
        printf '%s\n' "$ref_only" | sed 's/^/      /'
        for s in $ref_only; do
            case " $SUITES " in
                *" $s "*) echo "      ^ $s IS in the aarch64 suite list but produced no counts"; ok=0 ;;
                *) : ;;
            esac
        done
    fi

    LC_ALL=C join -t"$TAB" -j1 -o 1.1,1.2,1.3,2.2,2.3 /tmp/ref.txt /tmp/mine.txt > /tmp/joined.txt
    if [ ! -s /tmp/joined.txt ]; then
        echo "    $cfg: no suite ran on both sides -- nothing was compared"
        ok=0
    fi
    awk -F'\t' '$2!=$4 || $3!=$5 {printf "%s: x86_64 %s+%ss, aarch64 %s+%ss\n", $1, $2, $3, $4, $5}' \
        /tmp/joined.txt > /tmp/diff.txt

    if [ "$ok" = "1" ] && [ ! -s /tmp/diff.txt ]; then
        echo "    $cfg: identical to x86_64 on all $(wc -l < /tmp/joined.txt) suites both ran"
    else
        [ -s /tmp/diff.txt ] && { echo "    $cfg: DIFFERS from x86_64 --"; sed 's/^/      /' /tmp/diff.txt; }
        STATUS_REF="FAIL"; FAILED=1
    fi
done
echo

echo "=============================================================================="
printf "  %-24s %s\n" "core-only" "$STATUS_CORE"
printf "  %-24s %s\n" "-fno-exceptions" "$STATUS_NOEXC"
printf "  %-24s %s\n" "x86_64 count match" "$STATUS_REF"
echo "=============================================================================="
echo "  Correctness only. No timing here means anything."
if [ "$FAILED" -ne 0 ]; then
    echo
    echo "  AARCH64 VERIFICATION FAILED"
fi

exit $FAILED
CONTAINER_SCRIPT
