#!/usr/bin/env bash
#
# genericn_evidence.sh -- produce X-21's (T3.9 / E-4) complete evidence artifact.
#
# WHY THIS EXISTS. X-21 as first written committed no raw log, and its two most
# load-bearing claims -- "the generic route and the specialization compile to the
# same 567 aarch64 instructions" and "2.84x in code size" -- lived only as prose
# transcribed by hand. Every other reference-device entry in EXPERIMENTS.md cites a
# committed bincv-cpp/results/*_pi4.log, and the file records X-6 and X-8 being
# faulted for exactly this omission. A claim about machine code that no committed
# command reproduces is an assertion, not a measurement.
#
# This script emits, in one stream:
#
#   1. the benchmark itself (ns/pixel, all arms and decomposition points);
#   2. `size` and `size -A` on the three arm objects -- the code-size half of the
#      metric, with the Berkeley `text` column split so the exception plumbing is
#      visible separately;
#   3. `nm -C -S` per function on the same objects;
#   4. the INSTRUCTION-IDENTITY check, address-stripped and diffed, for all three
#      generic/specialized function pairs -- not just the derivative;
#   5. the same disassembly for the HAND-WRITTEN derivative, whose instruction
#      count is the datum behind the "+15% per word" attribution;
#   6. the ARITHMETIC-SPELLING check. X-21 claimed the word arithmetic is equal
#      across the arms "because there is no second way to compute mag = a ^ b",
#      while the library spells it `(a & ~b) | (b & ~a)` and the hand-written arm
#      spells it `a ^ b`. Whether those are the same INSTRUCTIONS is a compiler
#      question, and this compiles both and shows the answer rather than asserting
#      one.
#
# RUN IT THROUGH THE DEVICE RUNNER, from the build directory it creates:
#
#   ./scripts/run_on_pi.sh pi4 '../../scripts/genericn_evidence.sh'
#
# It writes genericn_benchmark_pi4.log into the CURRENT directory, which is
# build-pi -- run_on_pi.sh rsyncs build-pi/*.log back into bincv-cpp/results/, so
# the artifact lands in the tree without a second copy step.
#
# It is safe to run on x86_64 too, and says so in the header it writes: the shape
# of the evidence is the same, the numbers are INDICATIVE ONLY, and `size` on an
# x86_64 object answers a different question from `size` on an aarch64 one.

set -uo pipefail

readonly LOG="genericn_benchmark_pi4.log"
readonly ARMS_DIR="benchmark/CMakeFiles/genericn_arms.dir"

if [[ ! -x ./benchmark/genericn_benchmark ]]; then
    printf 'genericn_evidence.sh: run me from the build directory (no ./benchmark/genericn_benchmark here)\n' >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Disassembly helpers
#
# A function's disassembly is compared with ADDRESSES REMOVED, because two objects
# place the same code at different offsets and a raw diff would report every line.
# What survives the stripping is the instruction stream: mnemonics, registers, and
# branch targets rewritten as offsets RELATIVE TO THE FUNCTION's own entry, so a
# branch is compared as "forward 0x40" rather than as an absolute address.
# ---------------------------------------------------------------------------
slice() {   # slice <object> <symbol-substring>
    objdump -d --no-show-raw-insn -C "$1" |
        awk -v sym="$2" '
            /^[0-9a-f]+ </ { inside = (index($0, sym) > 0); next }
            inside && NF   { print }
        '
}

# Normalisation, and exactly what it does and does not remove:
#   1. the leading "  4a8:" address of each instruction  -- pure placement;
#   2. the bare hex branch/call TARGET that objdump prints immediately before the
#      <symbol> it resolves to -- also pure placement, since the <symbol+0xoff>
#      that follows carries the same information relative to the target;
#   3. the arm's name inside a symbol, so derivativeGeneric and
#      derivativeSpecialized compare as derivativeARM. This is the one substitution
#      that removes a real difference, and it removes only the NAME: an offset,
#      a register or a mnemonic that differed would still show.
# Nothing else is touched. A differing register allocation, a differing
# instruction, or a differing within-function offset all survive and diff.
strip_addr() {
    sed -E -e 's/^[[:space:]]*[0-9a-f]+:[[:space:]]*//' \
           -e 's/[0-9a-f]+ </</' \
           -e 's/Generic/ARM/g' -e 's/Specialized/ARM/g' \
           -e 's/libSpelling/SPELL/g' -e 's/handSpelling/SPELL/g' \
           -e 's/[[:space:]]+/ /g' -e 's/[[:space:]]*$//'
}

count_insns() { slice "$1" "$2" | grep -c . ; }

compare_pair() {  # compare_pair <label> <objA> <symA> <objB> <symB>
    local label="$1" objA="$2" symA="$3" objB="$4" symB="$5"
    local a b na nb
    a="$(mktemp)"; b="$(mktemp)"
    slice "$objA" "$symA" | strip_addr > "$a"
    slice "$objB" "$symB" | strip_addr > "$b"
    na="$(grep -c . < "$a")"; nb="$(grep -c . < "$b")"
    printf -- '--- %s\n' "$label"
    printf '    %-34s %s instructions\n' "$symA" "$na"
    printf '    %-34s %s instructions\n' "$symB" "$nb"
    if diff -q "$a" "$b" >/dev/null; then
        printf '    IDENTICAL after address stripping\n'
    else
        printf '    DIFFERS -- %s differing lines:\n' "$(diff "$a" "$b" | grep -c '^[<>]')"
        diff "$a" "$b" | sed 's/^/      /' | head -40
    fi
    rm -f "$a" "$b"
}

# ---------------------------------------------------------------------------
{
printf '========================================================================\n'
printf '  X-21 (T3.9 / E-4) -- complete evidence artifact\n'
printf '========================================================================\n'
printf '  generated: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf '  host arch: %s\n' "$(uname -m)"
printf '  compiler:  %s\n' "$(g++ --version | head -1)"
if [[ "$(uname -m)" != "aarch64" ]]; then
    printf '\n  *** NOT aarch64 -- INDICATIVE ONLY. E-4 closes on the reference device;\n'
    printf '      `size` on an x86_64 object answers a different question.\n'
fi
printf '\n\n'

printf '### 1. BENCHMARK\n\n'
./benchmark/genericn_benchmark

printf '\n\n### 2. CODE SIZE -- size (Berkeley) on one object per arm\n\n'
size ${ARMS_DIR}/genericn_arm_*.cpp.o

printf '\n### 2b. size -A -- the same bytes, split by section\n\n'
size -A ${ARMS_DIR}/genericn_arm_*.cpp.o

printf '\n### 3. nm -C -S --size-sort -- bytes per function\n\n'
nm -C -S --size-sort ${ARMS_DIR}/genericn_arm_*.cpp.o

printf '\n\n### 4. INSTRUCTION IDENTITY -- generic-N against the specialization\n'
printf '\n  All three function pairs, not only the derivative. X-21 first claimed\n'
printf '  "the same machine code" for the pair as a whole; the nm table in the same\n'
printf '  entry showed two of the three functions differing in size.\n\n'
compare_pair "derivative" \
    ${ARMS_DIR}/genericn_arm_generic.cpp.o     "derivativeGeneric" \
    ${ARMS_DIR}/genericn_arm_specialized.cpp.o "derivativeSpecialized"
printf '\n'
compare_pair "covariance window" \
    ${ARMS_DIR}/genericn_arm_generic.cpp.o     "covarianceWindowGeneric" \
    ${ARMS_DIR}/genericn_arm_specialized.cpp.o "covarianceWindowSpecialized"
printf '\n'
compare_pair "count whole frame" \
    ${ARMS_DIR}/genericn_arm_generic.cpp.o     "countWholeGeneric" \
    ${ARMS_DIR}/genericn_arm_specialized.cpp.o "countWholeSpecialized"

printf '\n\n### 5. THE HAND-WRITTEN DERIVATIVE, for the same treatment\n'
printf '\n  Two different functions, so a diff would be noise -- the datum is the\n'
printf '  instruction COUNT, which is what the per-word and per-row attribution\n'
printf '  rests on.\n\n'
printf '    %-34s %s instructions\n' "derivativeHandWritten" \
    "$(count_insns ${ARMS_DIR}/genericn_arm_handwritten.cpp.o derivativeHandWritten)"
printf '    %-34s %s instructions\n' "derivativeSpecialized" \
    "$(count_insns ${ARMS_DIR}/genericn_arm_specialized.cpp.o derivativeSpecialized)"
printf '    %-34s %s instructions\n' "countWholeHandWritten" \
    "$(count_insns ${ARMS_DIR}/genericn_arm_handwritten.cpp.o countWholeHandWritten)"
printf '    %-34s %s instructions\n' "countWholeSpecialized" \
    "$(count_insns ${ARMS_DIR}/genericn_arm_specialized.cpp.o countWholeSpecialized)"
printf '    %-34s %s instructions\n' "covarianceWindowHandWritten" \
    "$(count_insns ${ARMS_DIR}/genericn_arm_handwritten.cpp.o covarianceWindowHandWritten)"
printf '    %-34s %s instructions\n' "covarianceWindowSpecialized" \
    "$(count_insns ${ARMS_DIR}/genericn_arm_specialized.cpp.o covarianceWindowSpecialized)"

printf '\n\n### 6. THE ARITHMETIC-SPELLING CHECK\n'
printf '\n  impl::ternaryDifference writes  pos = a & ~b; neg = b & ~a; mag = pos | neg\n'
printf '  the hand-written arm writes      mag = a ^ b;  sign = b & ~a\n'
printf '  Same function of the same inputs, spelled in four operations and in two.\n'
printf '  X-21 asserted the arms hold the arithmetic equal; whether they do is a\n'
printf '  question about the compiler, so here it is compiled.\n\n'
SPELL="$(mktemp -d)"
cat > "$SPELL/spell.cpp" <<'CPP'
#include <cstdint>
void libSpelling(uint32_t a, uint32_t b, uint32_t* m, uint32_t* s) {
    const uint32_t pos = a & static_cast<uint32_t>(~b);
    const uint32_t neg = b & static_cast<uint32_t>(~a);
    *m = pos | neg;
    *s = neg;
}
void handSpelling(uint32_t a, uint32_t b, uint32_t* m, uint32_t* s) {
    *m = a ^ b;
    *s = b & static_cast<uint32_t>(~a);
}
CPP
g++ -O3 -c "$SPELL/spell.cpp" -o "$SPELL/spell.o"
objdump -d --no-show-raw-insn -C "$SPELL/spell.o" | sed -n '/libSpelling/,$p' | sed 's/^/    /'
printf '\n'
compare_pair "the two spellings, side by side" \
    "$SPELL/spell.o" "libSpelling" "$SPELL/spell.o" "handSpelling"
rm -rf "$SPELL"

printf '\n\n### 7. -fno-exceptions CODE SIZE\n'
printf '\n  The 2.84x figure above is from the DEFAULT build, which has exceptions on.\n'
printf '  ARCHITECTURE 2 names code size as often binding before RAM on Tier 2, and the\n'
printf '  Tier 2 claim rests on the core-only -fno-exceptions configuration -- which\n'
printf '  emits none of the .gcc_except_table, message strings or .eh_frame that the\n'
printf '  split above attributes 972 B to. So the ratio is measured there too.\n\n'
NOEXC="$(mktemp -d)"
for arm in generic specialized handwritten; do
    g++ -std=c++17 -O3 -DNDEBUG -fno-exceptions \
        -I ../include -I ../benchmark \
        -c "../benchmark/genericn_arm_${arm}.cpp" -o "$NOEXC/${arm}.o" 2>"$NOEXC/${arm}.err"
    if [[ ! -f "$NOEXC/${arm}.o" ]]; then
        printf '    %-14s COMPILE FAILED:\n' "$arm"; sed 's/^/      /' "$NOEXC/${arm}.err" | head -10
    fi
done
size "$NOEXC"/*.o 2>/dev/null
printf '\n'
size -A "$NOEXC"/*.o 2>/dev/null
rm -rf "$NOEXC"

printf '\n\n### END\n'
} 2>&1 | tee "$LOG"

printf '\nwrote %s\n' "$LOG"
