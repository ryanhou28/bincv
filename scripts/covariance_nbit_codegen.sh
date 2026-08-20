#!/usr/bin/env bash
#
# covariance_nbit_codegen.sh -- where the N-bit covariance's registers go (X-22).
#
# X-22 measured the bit-sliced covariance's cost curve in N on the reference
# device and found it tracking the 3N^2 + N popcount model everywhere EXCEPT one
# corner: at N = 4 with `uint64_t` words it is 15.2-15.7x the N = 1 arm where the
# model says 13.0x, while the same N = 4 at `uint32_t` comes in UNDER the model at
# 12.1-12.5x. The obvious hypothesis is register pressure -- at N = 4 the word loop
# holds 2N = 8 live magnitude words plus a selector, and a 64-bit word is a whole
# GPR where two 32-bit words are not -- but a hypothesis is not a measurement, and
# X-21 is this project's record of what happens when an attribution is reported
# without one.
#
# So this counts the SPILLS. It compiles one translation unit per (N, word type)
# with the kernel forced out of line, and counts stack traffic in the kernel's own
# instruction stream: `ldr`/`str` whose operand addresses [sp, ...] or [x29, ...].
# A kernel that fits in registers has a prologue and an epilogue and little else;
# one that does not has stack traffic inside the loop.
#
# Run it on the device, from a build directory:
#   ./scripts/run_on_pi.sh pi4 'bash ../../scripts/covariance_nbit_codegen.sh'
#
# It needs objdump and g++ on the device, and it writes nothing outside its own
# temporary directory.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCLUDE_DIR="${REPO_ROOT}/bincv-cpp/include"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

command -v objdump >/dev/null 2>&1 || { echo "objdump not available -- nothing measured"; exit 77; }

printf 'covariance N-bit codegen: stack traffic inside the kernel\n'
printf '  arch: %s   compiler: %s\n' "$(uname -m)" "$(g++ --version | head -1)"
printf '\n  %-10s %-4s %10s %10s %10s\n' "word" "N" "insns" "stack ld/st" "bytes"

for WORD in uint32_t uint64_t; do
  for N in 1 2 3 4; do
    cat > "${WORK}/probe.cpp" <<EOF
#include <cstdint>
#include "bincv-cpp/ops/covariance.hpp"
// Forced out of line, so \`objdump\` has one symbol that is exactly this kernel
// and nothing else. Everything the kernel calls is a header-inline template and
// lands inside it.
__attribute__((noinline)) bincv::GradientCovariance probe(
    const bincv::BinMatConstView<${WORD}> (&magX)[${N}],
    const bincv::BinMatConstView<${WORD}> (&magY)[${N}],
    bincv::BinMatConstView<${WORD}> signX, bincv::BinMatConstView<${WORD}> signY,
    bincv::Rect window) {
    return bincv::gradientCovariance<${N}, ${WORD}>(magX, magY, signX, signY, window);
}
EOF
    g++ -std=c++17 -O2 -DNDEBUG -I"${INCLUDE_DIR}" -c "${WORK}/probe.cpp" -o "${WORK}/probe.o"
    DIS="$(objdump -d --no-show-raw-insn "${WORK}/probe.o")"
    INSNS="$(printf '%s\n' "$DIS" | grep -cE '^\s+[0-9a-f]+:' || true)"
    STACK="$(printf '%s\n' "$DIS" | grep -cE '^\s+[0-9a-f]+:\s+(ldr|ldp|str|stp)[a-z]*\s+.*\[(sp|x29)' || true)"
    BYTES="$(size -A "${WORK}/probe.o" | awk '$1==".text"{print $2}')"
    printf '  %-10s %-4s %10s %10s %10s\n' "$WORD" "$N" "$INSNS" "$STACK" "$BYTES"
  done
done
printf '\n  "stack ld/st" counts every load/store through sp or x29, prologue and\n'
printf '  epilogue included, so the number to read is its GROWTH with N rather than\n'
printf '  its absolute value.\n'
