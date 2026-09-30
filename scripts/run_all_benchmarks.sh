#!/usr/bin/env bash
#
# run_all_benchmarks.sh -- run every built benchmark once, each into its own log.
#
# A smoke run, not a measurement: one launch per benchmark, unpinned, with the
# governor wherever it is. A figure that can be published comes from
# scripts/run_launches.sh, which launches ONE benchmark N times pinned to a core
# and stamps the log with the commit.
#
# USAGE
#     scripts/run_all_benchmarks.sh [options] [OUT_DIR]
#
#     OUT_DIR         where the logs go, one <benchmark>.log each
#                     (default: results/run_all-<arch>-<timestamp>, which is
#                     gitignored).
#     -b BUILD_DIR    the build tree whose benchmark/ holds the binaries
#                     (default: build).
#     -d FRAME_DIR    a directory of grayscale PNG frames for the sequence
#                     benchmarks -- feature_tracking_sequence, accuracy_realframes,
#                     lk_iteration_histogram, lk_stage_profile. Without it they are
#                     skipped, and their log says so.
#     -t SECONDS      per-benchmark timeout (default: none).
#     -h              this text.
#
# WHAT EACH BENCHMARK IS GIVEN. Most take no arguments and get none. The eleven
# built on bench_util.hpp parse --width/--height/--iterations and get the values
# below. The two one-arm-per-process sweeps (bitwidth_crossover, interop_roundtrip)
# run through their sweep scripts. The two localisation-floor probes get the
# repository's test image. The sequence benchmarks get FRAME_DIR or are skipped.
# Handing a flag to a binary that does not parse it is not harmless -- a
# positional-argument benchmark would take "--width" as its frame directory -- so
# the lists below are explicit.
#
# EXIT CODES
#     0   every benchmark that ran exited 0
#     1   at least one benchmark exited non-zero (named at the end; its log has the output)
#     2   usage error, or no benchmark binaries under BUILD_DIR

set -uo pipefail

usage() { awk 'NR > 2 && !/^#/ { exit } NR > 2 { print }' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
FRAME_DIR=""
TIMEOUT=""
WIDTH=640
HEIGHT=480
ITERATIONS=100

while [ $# -gt 0 ]; do
    case "$1" in
        -b) BUILD_DIR="${2-}"; shift 2 ;;
        -d) FRAME_DIR="${2-}"; shift 2 ;;
        -t) TIMEOUT="${2-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "run_all_benchmarks.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
        *) break ;;
    esac
done
[ $# -le 1 ] || { echo "run_all_benchmarks.sh: expected at most one OUT_DIR" >&2; usage >&2; exit 2; }

ARCH="$(uname -m)"
OUT_DIR="${1:-$ROOT_DIR/results/run_all-${ARCH}-$(date +%Y%m%d-%H%M%S)}"
BIN_DIR="$BUILD_DIR/benchmark"

if [ ! -d "$BIN_DIR" ]; then
    echo "run_all_benchmarks.sh: no benchmark directory at $BIN_DIR" >&2
    echo "  Build first (cmake -S . -B build && cmake --build build), or pass -b BUILD_DIR." >&2
    exit 2
fi
if [ -n "$TIMEOUT" ] && ! command -v timeout >/dev/null 2>&1; then
    echo "run_all_benchmarks.sh: -t needs the timeout(1) utility" >&2
    exit 2
fi
if [ -n "$FRAME_DIR" ] && [ ! -d "$FRAME_DIR" ]; then
    echo "run_all_benchmarks.sh: no such frame directory: $FRAME_DIR" >&2
    exit 2
fi

# The benchmarks built on bench_util.hpp, which parse --width/--height/--iterations.
FLAGGED="corner_opencv_benchmark denoise_benchmark derivative_benchmark fill_benchmark
logic_benchmark morphology_benchmark padding_benchmark reduce_benchmark resize_benchmark
set_pixels_benchmark transpose_benchmark"
# A directory of frames as the first argument.
SEQUENCE="accuracy_realframes feature_tracking_sequence lk_iteration_histogram lk_stage_profile"
# A grayscale image path as the first argument.
IMAGE="level0_floor level0_floor_noise"
IMAGE_PATH="$ROOT_DIR/tests/images/1403715887284058112.png"

listed() { case " $(echo $1) " in *" $2 "*) return 0 ;; esac; return 1; }

mkdir -p "$OUT_DIR" || exit 2
echo "benchmarks: $BIN_DIR"
echo "logs:       $OUT_DIR"

RAN=0
FAILED=()
SKIPPED=()
for bench_path in "$BIN_DIR"/*; do
    [ -f "$bench_path" ] && [ -x "$bench_path" ] || continue
    name="$(basename "$bench_path")"
    log="$OUT_DIR/$name.log"
    if listed "$FLAGGED" "$name"; then
        cmd=("$bench_path" --width "$WIDTH" --height "$HEIGHT" --iterations "$ITERATIONS")
    elif listed "$SEQUENCE" "$name"; then
        if [ -z "$FRAME_DIR" ]; then
            echo "=== [$name] skipped: needs a directory of frames (-d FRAME_DIR)" | tee "$log"
            SKIPPED+=("$name")
            continue
        fi
        cmd=("$bench_path" "$FRAME_DIR")
    elif listed "$IMAGE" "$name"; then
        if [ ! -f "$IMAGE_PATH" ]; then
            echo "=== [$name] skipped: test image missing at $IMAGE_PATH" | tee "$log"
            SKIPPED+=("$name")
            continue
        fi
        cmd=("$bench_path" "$IMAGE_PATH")
    else
        case "$name" in
            bitwidth_crossover) cmd=(bash "$ROOT_DIR/benchmark/crossover_sweep.sh") ;;
            interop_roundtrip)  cmd=(bash "$ROOT_DIR/benchmark/interop_sweep.sh") ;;
            *)                  cmd=("$bench_path") ;;
        esac
    fi
    echo "=== [$name] ${cmd[*]}" | tee "$log"
    # The sweep scripts address ./benchmark/<name>, so everything runs from the
    # build directory.
    if [ -n "$TIMEOUT" ]; then
        (cd "$BUILD_DIR" && timeout "$TIMEOUT" "${cmd[@]}") >> "$log" 2>&1
    else
        (cd "$BUILD_DIR" && "${cmd[@]}") >> "$log" 2>&1
    fi
    rc=$?
    RAN=$((RAN + 1))
    if [ "$rc" != 0 ]; then
        echo "    exit $rc" | tee -a "$log"
        FAILED+=("$name")
    fi
done

echo
echo "ran $RAN benchmark(s), ${#SKIPPED[@]} skipped, ${#FAILED[@]} failed. Logs in $OUT_DIR"
[ "${#SKIPPED[@]}" -eq 0 ] || echo "  skipped: ${SKIPPED[*]}"
if [ "${#FAILED[@]}" -ne 0 ]; then
    echo "  failed:  ${FAILED[*]}"
    exit 1
fi
if [ "$RAN" -eq 0 ]; then
    echo "run_all_benchmarks.sh: no benchmark binaries under $BIN_DIR" >&2
    exit 2
fi
exit 0
