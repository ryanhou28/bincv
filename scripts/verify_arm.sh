#!/usr/bin/env bash
#
# verify_arm.sh -- forwarding wrapper around scripts/verify_cross.sh.
#
# The emulated-architecture gate lives in scripts/verify_cross.sh, which detects
# the host and emulates whichever architecture it is not. A gate hardcoded to
# linux/arm64 would assume an x86_64 host: on an aarch64 machine its "emulated"
# run would be native, and its count comparison would diff one architecture
# against itself and call that a pass. This name stays so existing docs and
# habits keep working.

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1 && readlink -f "${SELF}" >/dev/null 2>&1; then
    SELF="$(readlink -f "${SELF}")"
fi

echo "verify_arm.sh is now a wrapper: running scripts/verify_cross.sh, which emulates"
echo "whichever architecture this host is not."
exec "$(dirname "${SELF}")/verify_cross.sh" "$@"
