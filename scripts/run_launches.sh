#!/usr/bin/env bash
#
# run_launches.sh -- run one benchmark N times as N SEPARATE PROCESSES, pinned.
#
# WHY THIS EXISTS. benchmark/measure_util.hpp calibrates a batch, repeats it, and
# prints the spread across those repeats. It says what that spread is worth:
#
#     The spread bounds WITHIN-run noise only; run-to-run scatter is a separate
#     and sometimes larger number, and an entry that calls a difference real
#     should clear the larger of the two.
#
# A process cannot see the second half of that. Whatever a launch pays once --
# where the allocator put the buffers, which core the scheduler parked a
# neighbour on, what the frequency was doing when calibrate() picked the batch
# size -- is constant inside the process and therefore invisible to every number
# the process prints. It moves between processes, so only processes can measure
# it.
#
# It is not a small correction on this host. On the x86-64 desktop under WSL2,
# goodFeaturesToTrack reports a 5% within-run spread and the same figure scatters
# 57% across thirty launches: the honest bound is eleven times the one the
# harness can print. The Pi 4 with the governor locked is the other extreme --
# 0.05-0.14% within-run, 0.41% across launches -- which is why the reference
# device is the reference device.
#
# The companion is scripts/aggregate_launches.py, which reads what this writes.
#
# USAGE
#     scripts/run_launches.sh -n 30 [options] <benchmark> [benchmark args...]
#
#     -n N            launches. REQUIRED, and deliberately without a default:
#                     how many launches a row needs is a property of the row and
#                     the difference being claimed, which `aggregate_launches.py
#                     --ladder` answers from the data. A number baked in here
#                     would be a project-wide bar that nobody decided.
#     -c CPU          core to pin to (default 2 on x86-64, 3 on aarch64 -- what
#                     the committed logs used).
#     -o FILE         output log (default <benchmark>-<arch>-launches.log here).
#     -s SECONDS      sleep between launches (default 0), for a thermally
#                     marginal device. Recorded in the header either way.
#     -g              lock the scaling governor to performance for the run and
#                     restore it on exit. Needs write access to sysfs.
#     --any-governor  run even though the governor is not performance. The
#                     numbers are then not comparable with the committed ones --
#                     ondemand inflates by 15-70% on the Pi -- so this has to be
#                     asked for.
#
# EXIT CODES
#     0   N launches completed, log written
#     1   a launch failed, or the governor is wrong and was not overridden
#     2   usage error
#
# WHAT IT DOES NOT DO. It does not discard the first launch. A cold page cache is
# part of what varies between launches, and dropping the launch that pays for it
# would trim exactly the tail this exists to measure. measure_util.hpp already
# discards a warm-up ROUND inside each process, which is the right place for it.

set -uo pipefail

usage() { sed -n '3,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

LAUNCHES=""
CPU=""
OUT=""
SLEEP=0
LOCK_GOV=0
ANY_GOV=0

while [ $# -gt 0 ]; do
    case "$1" in
        -n) LAUNCHES="${2-}"; shift 2 ;;
        -c) CPU="${2-}"; shift 2 ;;
        -o) OUT="${2-}"; shift 2 ;;
        -s) SLEEP="${2-}"; shift 2 ;;
        -g) LOCK_GOV=1; shift ;;
        --any-governor) ANY_GOV=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "run_launches.sh: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

[ $# -ge 1 ] || { echo "run_launches.sh: no benchmark given" >&2; usage >&2; exit 2; }
BENCH="$1"; shift
[ -x "$BENCH" ] || { echo "run_launches.sh: not executable: $BENCH" >&2; exit 2; }

case "$LAUNCHES" in
    ''|*[!0-9]*) echo "run_launches.sh: -n N is required (a positive integer)" >&2
                 echo "  There is no default on purpose: see the header." >&2; exit 2 ;;
esac
[ "$LAUNCHES" -ge 1 ] || { echo "run_launches.sh: -n must be at least 1" >&2; exit 2; }

ARCH="$(uname -m)"
if [ -z "$CPU" ]; then
    case "$ARCH" in
        aarch64) CPU=3 ;;
        *)       CPU=2 ;;
    esac
fi
[ -n "$OUT" ] || OUT="$(basename "$BENCH")-${ARCH}-launches.log"

# ---------------------------------------------------------------- the governor
GOVDIR=/sys/devices/system/cpu/cpu${CPU}/cpufreq
GOV="none (this host exposes no cpufreq)"
RESTORE=""
if [ -r "${GOVDIR}/scaling_governor" ]; then
    GOV="$(cat "${GOVDIR}/scaling_governor")"
    if [ "$LOCK_GOV" = 1 ]; then
        RESTORE="$GOV"
        for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo performance > "$g" 2>/dev/null
        done
        GOV="$(cat "${GOVDIR}/scaling_governor")"
        if [ "$GOV" != performance ]; then
            echo "run_launches.sh: could not set the governor (need write access to sysfs)" >&2
            exit 1
        fi
        GOV="performance (restored to ${RESTORE} on exit)"
    elif [ "$GOV" != performance ] && [ "$ANY_GOV" != 1 ]; then
        echo "run_launches.sh: governor is '${GOV}', not performance." >&2
        echo "  Pass -g to lock it for the run, or --any-governor to accept it." >&2
        echo "  Timings taken under ondemand are not comparable with the committed logs." >&2
        exit 1
    fi
fi
restore_governor() {
    [ -n "$RESTORE" ] || return 0
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "$RESTORE" > "$g" 2>/dev/null
    done
}
trap restore_governor EXIT INT TERM

# ------------------------------------------------------------------ provenance
# x86 spells it "model name"; the Pi puts the board in "Model" and has no
# "model name" line at all, so both are tried and the first hit wins.
host_model() {
    local m
    m="$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -1)"
    [ -n "$m" ] || m="$(sed -n 's/^Model[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -1)"
    echo "$m"
}

{
    echo "# launch sweep -- ${LAUNCHES} separate processes of one benchmark"
    echo "# benchmark: ${BENCH} $*"
    echo "# launches: ${LAUNCHES}"
    echo "# arch:     ${ARCH}"
    echo "# host:     $(host_model)"
    echo "# kernel:   $(uname -r)"
    echo "# compiler: $(g++ --version 2>/dev/null | head -1)"
    echo "# commit:   $(git -C "$(dirname "$0")" rev-parse --short HEAD 2>/dev/null)$(git -C "$(dirname "$0")" diff --quiet 2>/dev/null || echo ' (dirty)')"
    echo "# governor: ${GOV}"
    command -v vcgencmd >/dev/null 2>&1 && echo "# throttled before: $(vcgencmd get_throttled)"
    command -v vcgencmd >/dev/null 2>&1 && echo "# temp before: $(vcgencmd measure_temp)"
    echo "# pinning:  taskset -c ${CPU}"
    echo "# sleep between launches: ${SLEEP}s"
    echo "# date:     $(date -Iseconds)"
    echo "# loadavg at start: $(cat /proc/loadavg)"
    echo "# NO launch is discarded -- a cold start is part of what varies between"
    echo "# launches, and trimming it would trim the tail this sweep measures."
    echo
} > "$OUT"

# --------------------------------------------------------------------- the run
FAILED=0
for i in $(seq 1 "$LAUNCHES"); do
    echo "### run ${i}" >> "$OUT"
    taskset -c "$CPU" "$BENCH" "$@" >> "$OUT" 2>&1
    rc=$?
    if [ "$rc" != 0 ]; then
        echo "### run ${i} EXIT ${rc}" >> "$OUT"
        echo "run_launches.sh: launch ${i} exited ${rc}" >&2
        FAILED=1
    fi
    printf '\r  launch %d/%d' "$i" "$LAUNCHES" >&2
    [ "$SLEEP" = 0 ] || sleep "$SLEEP"
done
printf '\n' >&2

{
    echo
    echo "# loadavg at end: $(cat /proc/loadavg)"
    command -v vcgencmd >/dev/null 2>&1 && echo "# throttled after: $(vcgencmd get_throttled)"
    command -v vcgencmd >/dev/null 2>&1 && echo "# temp after: $(vcgencmd measure_temp)"
    echo "# date:     $(date -Iseconds)"
    echo "# done"
} >> "$OUT"

echo "wrote ${OUT}  (${LAUNCHES} launches)" >&2
echo "  scripts/aggregate_launches.py ${OUT}" >&2
exit "$FAILED"
