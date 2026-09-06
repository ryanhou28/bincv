#!/usr/bin/env bash
#
# verify_arm.sh -- forwarding wrapper around scripts/verify_cross.sh.
#
# This used to be the aarch64-under-emulation gate, hardcoded to linux/arm64.
# That shape assumed an x86_64 host: on an aarch64 machine its "emulated" run
# was native, and its count comparison diffed one architecture against itself
# and called that a pass. The gate now lives in scripts/verify_cross.sh, which
# detects the host and emulates the other architecture; this name stays so
# existing docs and habits keep working.

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1 && readlink -f "${SELF}" >/dev/null 2>&1; then
    SELF="$(readlink -f "${SELF}")"
fi

echo "verify_arm.sh is now a wrapper: running scripts/verify_cross.sh, which emulates"
echo "whichever architecture this host is not."
exec "$(dirname "${SELF}")/verify_cross.sh" "$@"
