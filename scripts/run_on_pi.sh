#!/usr/bin/env bash
#
# run_on_pi.sh -- run a command on the reference measurement device (T1.10).
#
# The Cortex-A device is where binCV's performance experiments are CLOSED
# (EXPERIMENTS.md "Measurement platforms"). E-1, E-2 and E-3 are cache-hierarchy
# questions, and a laptop hides the effect they measure: a Pi 4's Cortex-A72 has
# 32 KiB L1D and 1 MiB shared L2, against roughly 128 KiB and 12 MiB on an
# M-series core.
#
#   ./scripts/run_on_pi.sh <target> <command...>
#   ./scripts/run_on_pi.sh bincv-pi ./tests/test_binMat
#
# <target> is any ssh destination -- a ~/.ssh/config alias is strongly preferred,
# and pass -4 (see remote() below): mDNS .local resolves dual-stack and the IPv6
# leg is unroutable from WSL2. See
# docs/MEASUREMENT_HARDWARE.md. It may also be given as BINCV_PI_TARGET.
#
# ---------------------------------------------------------------------------
# Why this script exists rather than a documented procedure
#
# A Pi 4 will happily produce stable-looking numbers that are wrong. Four
# hazards, each of which silently corrupts a measurement:
#
#   1. A 32-bit OS. On armv7l every uint64_t operation is synthesised from
#      32-bit pairs, so E-2 would measure the compiler's 64-bit emulation
#      rather than the hardware. Not noise -- a different question.
#   2. Thermal or undervoltage throttling. An uncooled Pi 4 reaches ~80 C
#      during a sustained benchmark; a PC USB port cannot supply the ~1.2-1.5 A
#      it draws. Either sets a vcgencmd flag, and numbers taken in that state
#      are INVALID rather than merely slow.
#   3. The ondemand governor, scaling 600 MHz-1.5 GHz. Left alone, a short
#      benchmark measures the governor's ramp.
#   4. Core migration and desktop background load.
#
# Documentation cannot enforce any of that; a script can. Each is a hard refusal
# or an automatic INVALID, not a warning someone reads past.
#
# EXIT CODES
#   0   command ran, device state was valid throughout
#   1   command failed, or the device state INVALIDATES the run
#   77  could not run at all (no target configured, unreachable, wrong arch)
#       -- NOT a pass. Callers must branch on this rather than treat it as OK.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT

# Remote scratch. Deliberately not /tmp: some Pi images mount it as tmpfs, and a
# build tree there competes for the RAM the measurement is about.
# Relative on purpose. An earlier version used "\$HOME/bincv-measure", which ssh
# expanded but rsync did NOT -- rsync received the literal string and created a
# directory named '$HOME' on the device. A relative path resolves against the
# remote home for both, so there is nothing to quote and nothing to disagree about.
readonly REMOTE_DIR="bincv-measure"
readonly RESULTS_DIR="${REPO_ROOT}/results"

GOVERNOR_SAVED=""
TARGET=""

# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
hr()   { printf '%s\n' "------------------------------------------------------------"; }
info() { printf '  %s\n' "$*"; }
fail() { printf '\n  FAILED: %s\n' "$*" >&2; exit 1; }

skip() {
    printf '\n  SKIPPED: %s\n' "$*" >&2
    printf '  Setup instructions: docs/MEASUREMENT_HARDWARE.md\n' >&2
    printf '  This is exit 77, not a pass -- nothing was measured.\n' >&2
    exit 77
}

# Run a command on the device. Kept in one place so every remote call gets the
# same options: no interactive prompts, and a bounded connect time so an absent
# device fails in seconds rather than hanging a session.
remote() {
    # -4 is not optional. An mDNS .local name resolves dual-stack, and a WSL2 host
    # typically has no IPv6 route, so ssh intermittently picks the AAAA record and
    # dies with "Network is unreachable". Measured here: 1 failure in a handful of
    # connections, which is precisely the flake that kills an unattended run at 3am.
    ssh -4 -o BatchMode=yes -o ConnectTimeout=10 "$TARGET" "$@"
}

# --------------------------------------------------------------------------
# Governor restoration
#
# Runs on EVERY exit path including failure and interrupt. Leaving a device
# pinned to `performance` would silently change every later measurement taken by
# anyone, which is a worse failure than the one that got us here.
# --------------------------------------------------------------------------
restore_governor() {
    local rc=$?
    if [[ -n "$GOVERNOR_SAVED" && -n "$TARGET" ]]; then
        # RETRY. The restore itself needs the network, so a transient ssh failure
        # would otherwise leave the device pinned to `performance` -- which
        # silently changes every later measurement anyone takes on it. Observed
        # in the first real run against hardware: the restore lost a name
        # resolution and the Pi stayed pinned.
        local attempt restored=0
        for attempt in 1 2 3; do
            if remote "echo '$GOVERNOR_SAVED' | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null" 2>/dev/null; then
                restored=1; break
            fi
            sleep 2
        done
        if [[ $restored -eq 1 ]]; then
            info "governor restored to $GOVERNOR_SAVED"
        else
            printf '\n  ####################################################################\n' >&2
            printf '  ## WARNING: COULD NOT RESTORE THE GOVERNOR after 3 attempts.\n' >&2
            printf '  ## The device is probably still pinned to `performance`, which will\n' >&2
            printf '  ## silently affect every later measurement taken on it.\n' >&2
            printf '  ## Fix by hand:\n' >&2
            printf '  ##   ssh -4 %s "echo %s | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"\n' "$TARGET" "$GOVERNOR_SAVED" >&2
            printf '  ####################################################################\n\n' >&2
        fi
    fi
    exit $rc
}
trap restore_governor EXIT INT TERM

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    TARGET="${BINCV_PI_TARGET:-}"
    [[ -n "$TARGET" ]] || skip "no target given and BINCV_PI_TARGET is unset

  usage: ./scripts/run_on_pi.sh <ssh-target> <command...>
     eg: ./scripts/run_on_pi.sh bincv-pi ./tests/test_binMat"
else
    TARGET="$1"; shift
fi

if [[ $# -lt 1 ]]; then
    REMOTE_CMD="ctest --output-on-failure"
    info "no command given; defaulting to: $REMOTE_CMD"
else
    REMOTE_CMD="$*"
fi

printf '\n'
hr
printf '  binCV reference-device run\n'
hr
info "target:  $TARGET"
info "command: $REMOTE_CMD"
printf '\n'

# --------------------------------------------------------------------------
# Preflight -- REFUSE rather than warn
# --------------------------------------------------------------------------
printf '  preflight\n'

command -v ssh   >/dev/null 2>&1 || skip "ssh is not installed"
command -v rsync >/dev/null 2>&1 || skip "rsync is not installed"

remote true 2>/dev/null || skip "cannot reach '$TARGET' over ssh with BatchMode

  Key-based auth is required (no password prompts) -- see
  docs/MEASUREMENT_HARDWARE.md. If the router pings but the device does not,
  check whether a VPN client is routing LAN traffic through its tunnel."

# 1. Architecture. A hard refusal: a 32-bit result answers a different question.
ARCH="$(remote uname -m | tr -d '\r')"
if [[ "$ARCH" != "aarch64" ]]; then
    skip "device reports '$ARCH', not aarch64

  On 32-bit ARM every uint64_t operation is synthesised from 32-bit pairs, so a
  word-width measurement would describe the compiler rather than the hardware.
  Reflash with Raspberry Pi OS Lite (64-bit) -- docs/MEASUREMENT_HARDWARE.md."
fi
info "arch: $ARCH"

# 2. Throttle state BEFORE the run. Non-zero means undervoltage or thermal
#    throttling has already occurred, so anything measured now is invalid.
THROTTLE_BEFORE="$(remote 'vcgencmd get_throttled 2>/dev/null || echo unavailable' | tr -d '\r')"
case "$THROTTLE_BEFORE" in
    throttled=0x0)  info "throttle before: 0x0 (clean)" ;;
    unavailable)    info "throttle before: vcgencmd unavailable (not a Pi? continuing)" ;;
    *)              fail "device is already throttled ($THROTTLE_BEFORE)

  Non-zero means undervoltage or thermal throttling. Measurements taken in this
  state are INVALID, not merely slower. Use the official 5.1V 3A supply -- a PC
  USB port cannot power a Pi 4 -- add cooling, let it cool, and re-run." ;;
esac

# 3. Governor. Saved first so the trap can put it back.
GOVERNOR_SAVED="$(remote 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown' | tr -d '\r')"
if [[ "$GOVERNOR_SAVED" != "unknown" ]]; then
    remote "echo performance | sudo -n tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null" \
        || fail "could not set the performance governor (passwordless sudo required)"
    info "governor: $GOVERNOR_SAVED -> performance"
else
    GOVERNOR_SAVED=""
    info "governor: cpufreq unavailable, leaving alone"
fi

# --------------------------------------------------------------------------
# Sync and build
# --------------------------------------------------------------------------
printf '\n  sync\n'
remote "mkdir -p $REMOTE_DIR"
rsync -az --delete \
      --exclude 'build*/' --exclude '.git/' --exclude 'results/' \
      -e 'ssh -4 -o BatchMode=yes -o ConnectTimeout=10' \
      "${REPO_ROOT}/" "${TARGET}:${REMOTE_DIR}/" \
    || fail "rsync failed"
info "repository synced to ${REMOTE_DIR}"

printf '\n  build (Release, core-only)\n'
remote "cd $REMOTE_DIR && cmake -S bincv-cpp -B bincv-cpp/build-pi \
            -DCMAKE_BUILD_TYPE=Release -DBINCV_USE_OPENCV=OFF >/dev/null" \
    || fail "cmake configure failed on the device"
remote "cd $REMOTE_DIR && cmake --build bincv-cpp/build-pi -j\$(nproc)" \
    || fail "build failed on the device"
info "build complete"

# --------------------------------------------------------------------------
# Run, pinned to one core
# --------------------------------------------------------------------------
printf '\n  run\n'
hr
set +e
remote "cd $REMOTE_DIR/bincv-cpp/build-pi && taskset -c 3 $REMOTE_CMD"
RUN_RC=$?
set -e
hr

# --------------------------------------------------------------------------
# Throttle state AFTER -- this is what makes a corrupted run visible
# --------------------------------------------------------------------------
THROTTLE_AFTER="$(remote 'vcgencmd get_throttled 2>/dev/null || echo unavailable' | tr -d '\r')"
INVALID=0
case "$THROTTLE_AFTER" in
    throttled=0x0|unavailable) ;;
    *) INVALID=1 ;;
esac

# --------------------------------------------------------------------------
# Environment block -- paste straight into an EXPERIMENTS.md entry
# --------------------------------------------------------------------------
CPU_MODEL="$(remote "grep -m1 'model name\|Model' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'" | tr -d '\r' || true)"
KERNEL="$(remote uname -r | tr -d '\r')"
COMPILER="$(remote 'g++ --version | head -1' | tr -d '\r')"
GOV_ACTIVE="$(remote 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a' | tr -d '\r')"

printf '\n'
hr
printf '  ENVIRONMENT -- record this with any measurement\n'
hr
info "device:           $TARGET"
info "cpu:              ${CPU_MODEL:-unknown}"
info "arch / kernel:    $ARCH / $KERNEL"
info "compiler:         $COMPILER"
info "governor:         $GOV_ACTIVE (restored to $GOVERNOR_SAVED on exit)"
info "core pinning:     taskset -c 3"
info "throttled before: $THROTTLE_BEFORE"
info "throttled after:  $THROTTLE_AFTER"
info "commit:           $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
hr

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
mkdir -p "$RESULTS_DIR"
rsync -az -e 'ssh -4 -o BatchMode=yes' \
      "${TARGET}:${REMOTE_DIR}/bincv-cpp/build-pi/*.log" "$RESULTS_DIR/" 2>/dev/null || true

printf '\n'
if [[ $INVALID -eq 1 ]]; then
    printf '  RESULTS INVALID -- device throttled during the run (%s)\n' "$THROTTLE_AFTER" >&2
    printf '  Do NOT record these numbers. Add cooling, check the power supply,\n' >&2
    printf '  and re-run. A throttled measurement is wrong, not merely slow.\n' >&2
    exit 1
fi

if [[ $RUN_RC -ne 0 ]]; then
    printf '  COMMAND FAILED (exit %d). Device state was valid throughout.\n' "$RUN_RC" >&2
    exit 1
fi

printf '  OK -- command succeeded, device state valid throughout\n'
printf '  Timings from this device ARE authoritative (unlike emulation).\n'
printf '  Record the environment block above in EXPERIMENTS.md.\n\n'
