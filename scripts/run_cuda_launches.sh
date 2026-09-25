#!/usr/bin/env bash
#
# run_cuda_launches.sh -- run one CUDA benchmark N times as N SEPARATE
# PROCESSES, and write ONE stamped log the staleness gate can read.
#
# WHY THIS EXISTS. docs/reports/cuda.md promises "the median of 7 independent
# process runs" for every figure in it, and committed nothing: no raw output,
# no commit stamp, so nothing could tell whether a published CUDA figure still
# described the code. It was not hypothetical. Re-run at its own published
# commit, `goodFeaturesToTrack` read 5.3x on the page and 8.40x on the machine,
# its two arms moving in OPPOSITE directions -- which a session can do and a
# kernel cannot. A reader had no way to separate those two stories because the
# session that produced the page left nothing behind.
#
# WHY A SECOND SCRIPT RATHER THAN A FLAG ON run_launches.sh. The same reason
# the backend forks its kernels: what is SHARED here is the log's header --
# `# benchmark:` and `# commit:` are the contract scripts/check_figure_staleness.py
# reads, and a second spelling of them would put a whole sweep outside the gate
# without saying so. What is FORKED is the protocol, because almost none of the
# host runner's applies. Its long argument is about what a LAUNCH pays once:
# the allocator, the sibling core, the governor when calibrate() picked a batch
# size. A kernel-resident figure pays for a GPU whose clock ramps on its own
# schedule, a driver that may be a different version from the toolkit that
# built the kernels, and a profiler that -- if it is attached -- replays every
# kernel several times and reports the replay. Folding that into the host
# script would give one file two incompatible essays and one flag nobody could
# read the header of. The shared half is instead made structural at the end of
# this script: it asks the gate to map the log it has just written, so a header
# that has drifted out of the gate's reach is caught here and not in review.
#
# WHAT THE HEADER RECORDS, AND WHY EACH FIELD IS THERE. A CUDA figure is a
# claim about a specific GPU under a specific driver, compiled by a specific
# nvcc for a specific architecture. Change any of those and the number is about
# something else:
#
#   device       name, compute capability and memory. The backend's reference
#                GPU is an RTX 3070 Ti at sm_86; the same kernel on sm_75 has a
#                different occupancy and on Jetson a different memory system.
#   driver       the driver schedules every launch, and on WSL2 the launch cost
#                IS the figure for the several ops that sit on the launch floor.
#                Its CUDA version is recorded beside it because it need not be
#                the toolkit's -- here it is 12.6 over an 11.1 nvcc.
#   nvcc         which compiler produced the SASS, read from the build tree
#                that produced the binary rather than from PATH, because those
#                are routinely different on a machine with two toolkits.
#   cuda arch    what the binary was actually built for. A binary built for
#                another architecture JITs at first launch, and a JIT is not
#                the kernel a figure claims to measure.
#   opencv       which OpenCV supplies the role comparison's denominators. Half
#                of every ratio in the report's speed table comes from it.
#   gpu clock    before and after, with the throttle reasons and temperature --
#                the device analogue of the host runner's governor and the Pi's
#                vcgencmd readings. The clock is NOT lockable on every host
#                (WSL2 exposes no application clocks at all), so the header says
#                which case this was rather than implying a locked clock.
#   profiler     that no profiler was attached. ncu serialises kernels and
#                REPLAYS each one to collect its counters; a timing taken with
#                it attached is a timing of the replay. Timing and profiling do
#                not mix, so the run refuses to start when either tool is
#                running and the log records that it refused to.
#   compute apps what else held device memory when the sweep started. A second
#                process on the device is not a neighbour on a sibling core --
#                it competes for the SMs the figure is about.
#
# THE OUTPUT IS ONE FILE, with `### run N` between launches, which is the shape
# scripts/aggregate_launches.py established and scripts/aggregate_cuda_runs.py
# now reads: one committed artifact per sweep, and the per-process structure
# the project's rule needs kept inside it.
#
# USAGE
#     scripts/run_cuda_launches.sh -n 7 <benchmark> [benchmark args...]
#
#     -n N            launches. REQUIRED, and deliberately without a default,
#                     for the reason run_launches.sh gives: how many a row needs
#                     is a property of the row and the difference claimed.
#     -c CPU          core to pin the enqueueing thread to (default 2). The
#                     enqueue path is host code and WSL2 inflates it.
#     -o FILE         output log (default <benchmark>-cuda-launches.log here).
#     -s SECONDS      sleep between launches (default 0). A GPU that has just
#                     run for a minute is not at the clock it started at;
#                     recorded in the header either way.
#     -b DIR          the CMake build tree the binary came from. Found by
#                     walking up from the binary when not given.
#     --any-governor  run even though the CPU governor is not performance.
#     --any-device    run even though another process holds device memory. The
#                     numbers are then not comparable with the committed ones.
#
# EXIT CODES
#     0   N launches completed, log written, and the gate could map it
#     1   a launch failed, the governor is wrong, or the device was busy
#     2   usage error
#     3   a profiler is running -- nothing was run and nothing was written
#
# WHAT IT DOES NOT DO. It does not lock the GPU clock, because no host this
# backend has run on lets it: `nvidia-smi -lgc` needs administrative rights and
# WSL2's driver reports application clocks as N/A. So it records the clock
# instead of pretending to hold it, and the spread across launches is where an
# unlocked clock shows up. It also does not discard the first launch, for the
# reason the host runner gives -- and here there is a second: the first launch
# pays for context creation and any JIT, which is part of what a caller pays.

set -uo pipefail

usage() { sed -n '3,100p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

LAUNCHES=""
CPU=2
OUT=""
SLEEP=0
BUILD_DIR=""
ANY_GOV=0
ANY_DEVICE=0

while [ $# -gt 0 ]; do
    case "$1" in
        -n) LAUNCHES="${2-}"; shift 2 ;;
        -c) CPU="${2-}"; shift 2 ;;
        -o) OUT="${2-}"; shift 2 ;;
        -s) SLEEP="${2-}"; shift 2 ;;
        -b) BUILD_DIR="${2-}"; shift 2 ;;
        --any-governor) ANY_GOV=1; shift ;;
        --any-device) ANY_DEVICE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "run_cuda_launches.sh: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

[ $# -ge 1 ] || { echo "run_cuda_launches.sh: no benchmark given" >&2; usage >&2; exit 2; }
BENCH="$1"; shift
[ -x "$BENCH" ] || { echo "run_cuda_launches.sh: not executable: $BENCH" >&2; exit 2; }

case "$LAUNCHES" in
    ''|*[!0-9]*) echo "run_cuda_launches.sh: -n N is required (a positive integer)" >&2
                 echo "  There is no default on purpose: see the header." >&2; exit 2 ;;
esac
[ "$LAUNCHES" -ge 1 ] || { echo "run_cuda_launches.sh: -n must be at least 1" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARCH="$(uname -m)"
[ -n "$OUT" ] || OUT="$(basename "$BENCH")-cuda-launches.log"

# -------------------------------------------------------------- the profiler
# Before anything else, because a run taken under ncu is not a timing at all
# and there is no point collecting one. nsys perturbs less than ncu and still
# adds a tracing hook to every launch, so both are refused.
PROFILER=""
for tool in ncu nsys nv-nsight-cu-cli nvprof; do
    if pgrep -x "$tool" >/dev/null 2>&1; then PROFILER="$tool"; break; fi
done
if [ -n "$PROFILER" ]; then
    echo "run_cuda_launches.sh: '${PROFILER}' is running." >&2
    echo "  A profiler replays each kernel to collect its counters, so a timing" >&2
    echo "  taken beside one is a timing of the replay. Stop it and re-run." >&2
    exit 3
fi

# ---------------------------------------------------------------- the device
nvq() {
    # One --query-gpu field, or "unavailable". Names moved between driver
    # branches (clocks_throttle_reasons -> clocks_event_reasons), so a caller
    # passes the spellings it knows and the first one the driver accepts wins.
    local v
    for field in "$@"; do
        v="$(nvidia-smi --query-gpu="$field" --format=csv,noheader 2>/dev/null | head -1)"
        [ -n "$v" ] && { echo "$v"; return 0; }
    done
    echo "unavailable"
}
gpu_state() {
    printf 'sm %s, mem %s, pstate %s' \
        "$(nvq clocks.sm)" "$(nvq clocks.mem)" "$(nvq pstate)"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "run_cuda_launches.sh: no nvidia-smi -- cannot record what device this is." >&2
    echo "  A CUDA figure is a claim about a specific GPU; an unrecorded one is not" >&2
    echo "  a figure. Install the driver utilities or run this where the GPU is." >&2
    exit 1
fi

COMPUTE_APPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                --format=csv,noheader 2>/dev/null | paste -sd';' -)"
[ -n "$COMPUTE_APPS" ] || COMPUTE_APPS="none"
if [ "$COMPUTE_APPS" != none ] && [ "$ANY_DEVICE" != 1 ]; then
    echo "run_cuda_launches.sh: another process holds device memory:" >&2
    echo "    ${COMPUTE_APPS}" >&2
    echo "  It competes for the SMs this figure is about. Wait for it, or pass" >&2
    echo "  --any-device to accept numbers that are not comparable with the" >&2
    echo "  committed ones." >&2
    exit 1
fi

# HOST CPU LOAD, WHICH IS NOT A CPU CONCERN HERE. Under WSL2 the GPU is reached
# through a virtualization shim and every launch is enqueued by the host, so the
# LAUNCH FLOOR is host-dispatch-bound -- and several ops in this backend sit on
# that floor at the reference geometry. Measured cost of not checking: two sweeps
# of one commit, one at load 0.52 and one at 7.41, put the floor at 0.0063 and
# 0.0079 ms and moved the launch-bound rows 10-24% with the kernels untouched.
# The rows far from the floor moved 0.7%, and the paired RATIOS survived -- both
# arms pay the same contention, which is what pairing is for -- so this warns
# rather than refuses: a ratio taken here is still sound and an absolute is not.
LOAD1="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
if awk -v l="$LOAD1" 'BEGIN{exit !(l > 2.0)}'; then
    echo "run_cuda_launches.sh: host load average is ${LOAD1}." >&2
    echo "  The launch floor is host-dispatch-bound through WSL2's GPU shim, so the" >&2
    echo "  ABSOLUTE times of launch-bound ops will read high and are not comparable" >&2
    echo "  with committed figures taken on a quiet host. Paired ratios are fine." >&2
    echo "  Proceeding -- the header records the load either way." >&2
fi

# The clock cannot be held on any host this backend has run on, so the header
# says which case applies rather than leaving a reader to assume the good one.
APP_CLOCK="$(nvq clocks.applications.graphics)"
case "$APP_CLOCK" in
    *N/A*|unavailable) CLOCK_LOCK="not locked -- this host exposes no application clocks" ;;
    *)                 CLOCK_LOCK="application clock ${APP_CLOCK}" ;;
esac

# ---------------------------------------------------------------- the build
# What compiled the kernels, read from the tree that produced the binary. The
# nvcc on PATH is routinely not the one CMake used: this backend builds only
# under cuda-11.1 while the driver exposes 12.6.
if [ -z "$BUILD_DIR" ]; then
    d="$(cd "$(dirname "$BENCH")" && pwd)"
    while [ "$d" != / ]; do
        [ -f "$d/CMakeCache.txt" ] && { BUILD_DIR="$d"; break; }
        d="$(dirname "$d")"
    done
fi
cache_value() {
    [ -n "$BUILD_DIR" ] && [ -f "$BUILD_DIR/CMakeCache.txt" ] || { echo "unknown"; return; }
    local v
    v="$(sed -n "s/^$1:[A-Z]*=//p" "$BUILD_DIR/CMakeCache.txt" | head -1)"
    [ -n "$v" ] && echo "$v" || echo "unset"
}
NVCC="$(cache_value CMAKE_CUDA_COMPILER)"
[ "$NVCC" = unknown ] || [ "$NVCC" = unset ] || [ ! -x "$NVCC" ] || \
    NVCC="$NVCC ($("$NVCC" --version 2>/dev/null | sed -n 's/.*release \([^,]*\),.*/release \1/p' | head -1))"

# ------------------------------------------------------------- the governor
# The enqueue path is host code, so the host's clock still moves part of an
# end-to-end figure. Recorded always; refused only where there is a governor to
# refuse -- this backend's reference host exposes no cpufreq at all.
GOVDIR=/sys/devices/system/cpu/cpu${CPU}/cpufreq
GOV="none (this host exposes no cpufreq)"
if [ -r "${GOVDIR}/scaling_governor" ]; then
    GOV="$(cat "${GOVDIR}/scaling_governor")"
    if [ "$GOV" != performance ] && [ "$ANY_GOV" != 1 ]; then
        echo "run_cuda_launches.sh: governor is '${GOV}', not performance." >&2
        echo "  Pass --any-governor to accept it; the host half of an end-to-end" >&2
        echo "  figure is then not comparable with the committed logs." >&2
        exit 1
    fi
fi

# ----------------------------------------------------------------- the header
host_model() {
    local m
    m="$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -1)"
    [ -n "$m" ] || m="$(sed -n 's/^Model[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -1)"
    echo "$m"
}

{
    echo "# launch sweep -- ${LAUNCHES} separate processes of one CUDA benchmark"
    echo "# benchmark: ${BENCH} $*"
    echo "# launches: ${LAUNCHES}"
    echo "# arch:     ${ARCH}"
    echo "# host:     $(host_model)"
    echo "# kernel:   $(uname -r)"
    echo "# compiler: $(g++ --version 2>/dev/null | head -1)"
    echo "# commit:   $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null)$(git -C "$REPO_ROOT" diff --quiet 2>/dev/null || echo ' (dirty)')"
    echo "# device:   $(nvq name) (compute $(nvq compute_cap), $(nvq memory.total))"
    echo "# driver:   $(nvq driver_version), exposing CUDA $(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
    echo "# nvcc:     ${NVCC}"
    echo "# cuda arch: $(cache_value CMAKE_CUDA_ARCHITECTURES)"
    echo "# nvcc ccbin: $(cache_value CMAKE_CUDA_HOST_COMPILER)"
    echo "# build type: $(cache_value CMAKE_BUILD_TYPE)"
    echo "# build dir: ${BUILD_DIR:-not found -- nvcc and architecture unread}"
    echo "# opencv for role arms: $(cache_value BINCV_CUDA_OPENCV_DIR)"
    echo "# profiler: none running -- ncu, nsys, nvprof all absent at start"
    echo "# compute apps at start: ${COMPUTE_APPS}"
    echo "# gpu clock lock: ${CLOCK_LOCK}"
    echo "# gpu clock before: $(gpu_state)"
    echo "# gpu throttle before: $(nvq clocks_event_reasons.active clocks_throttle_reasons.active)"
    echo "# gpu temp before: $(nvq temperature.gpu) C"
    echo "# governor: ${GOV}"
    echo "# pinning:  taskset -c ${CPU}"
    echo "# sleep between launches: ${SLEEP}s"
    echo "# date:     $(date -Iseconds)"
    echo "# loadavg at start: $(cat /proc/loadavg)"
    echo "# NO launch is discarded -- the first pays for context creation and any"
    echo "# JIT, which is part of what a caller pays and part of what varies."
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
        echo "run_cuda_launches.sh: launch ${i} exited ${rc}" >&2
        FAILED=1
    fi
    printf '\r  launch %d/%d' "$i" "$LAUNCHES" >&2
    [ "$SLEEP" = 0 ] || sleep "$SLEEP"
done
printf '\n' >&2

# A profiler started mid-sweep would have perturbed the launches after it, and
# the log would say nothing. Cheap to ask again; the answer belongs in the file.
PROFILER_AFTER=""
for tool in ncu nsys nv-nsight-cu-cli nvprof; do
    if pgrep -x "$tool" >/dev/null 2>&1; then PROFILER_AFTER="$tool"; break; fi
done

{
    echo
    echo "# loadavg at end: $(cat /proc/loadavg)"
    echo "# gpu clock after: $(gpu_state)"
    echo "# gpu throttle after: $(nvq clocks_event_reasons.active clocks_throttle_reasons.active)"
    echo "# gpu temp after: $(nvq temperature.gpu) C"
    echo "# profiler at end: ${PROFILER_AFTER:-none running}"
    echo "# date:     $(date -Iseconds)"
    echo "# done"
} >> "$OUT"

if [ -n "$PROFILER_AFTER" ]; then
    echo "run_cuda_launches.sh: '${PROFILER_AFTER}' started DURING the sweep." >&2
    echo "  The launches after it are timings of a replay. Do not commit this log." >&2
    FAILED=1
fi

echo "wrote ${OUT}  (${LAUNCHES} launches)" >&2
echo "  scripts/aggregate_cuda_runs.py ${OUT}" >&2

# ------------------------------------------------- the gate reads what we wrote
# The half this script SHARES with run_launches.sh is the header the staleness
# gate parses, and a shared format with two writers is a format that drifts. So
# the gate is asked, here, whether it can map the file just written: a sweep it
# cannot map is a sweep whose figures nothing can ever call stale.
echo >&2
python3 "${SCRIPT_DIR}/check_figure_staleness.py" --explain "$OUT" >&2
case $? in
    0) ;;
    1) echo "  (stale against the working tree -- expected only if the tree moved" >&2
       echo "   since the stamp; a sweep for publication is taken at a clean tree.)" >&2 ;;
    *) echo "  This log cannot be mapped to the code it measured, so committing it" >&2
       echo "  would publish a figure no gate can ever check. Fix that first." >&2
       FAILED=1 ;;
esac

exit "$FAILED"
