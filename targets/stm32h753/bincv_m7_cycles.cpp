/// @file bincv_m7_cycles.cpp
/// @brief What single instructions cost on this core, the numbers the DSP arms are
/// shaped by.
///
///   CYCLES          cycles per instruction for each block below, x100, best of 3
///
/// Each block is 32 instructions repeated 64 times in a loop: "ind" blocks write four
/// registers in turn from the same two inputs, so nothing waits on anything and the
/// figure is the issue rate (0.5 is two a cycle); "dep" blocks feed each instruction
/// its predecessor's result, so the figure is the latency; a block naming two
/// instructions alternates them, so a figure near 0.5 says they issue together. Loads
/// read one 64-byte block of the cache over and over, so they never miss. Nothing here
/// is a kernel; it is the cost model a kernel for this core is written against.

#include <cstddef>
#include <cstdint>

#include "board.h"
#include "m7_harness.hpp"

namespace {

__attribute__((section(".frames"), aligned(32))) uint8_t g_buf[256];

#define REP8(x) x x x x x x x x

#define BLOCK(name, body)                                                               \
    __attribute__((noinline)) uint32_t name(const uint8_t* p) {                         \
        const uint32_t t0 = boardCycles();                                              \
        for (int i = 0; i < 64; ++i) {                                                  \
            __asm__ volatile(body::"r"(p)                                               \
                             : "r0", "r1", "r2", "r3", "r4", "r8", "r9", "r10", "r11", \
                               "memory", "cc");                                         \
        }                                                                               \
        return boardCycles() - t0;                                                      \
    }

#define FOUR(a, b, c, d) REP8(a "\n" b "\n" c "\n" d "\n")

BLOCK(orrInd, FOUR("orr r0, r8, r9", "orr r1, r8, r9", "orr r2, r8, r9", "orr r3, r8, r9"))
BLOCK(orrDep, FOUR("orr r0, r0, r9", "orr r0, r0, r9", "orr r0, r0, r9", "orr r0, r0, r9"))
BLOCK(orrShiftedInd, FOUR("orr r0, r8, r9, lsl #8", "orr r1, r8, r9, lsl #8",
                          "orr r2, r8, r9, lsr #8", "orr r3, r8, r9, lsr #8"))
BLOCK(lsrInd, FOUR("lsr r0, r8, #8", "lsr r1, r8, #8", "lsl r2, r8, #8", "lsl r3, r8, #8"))
BLOCK(uqsub8Ind, FOUR("uqsub8 r0, r8, r9", "uqsub8 r1, r8, r9", "uqsub8 r2, r8, r9",
                      "uqsub8 r3, r8, r9"))
BLOCK(uqsub8Dep, FOUR("uqsub8 r0, r0, r9", "uqsub8 r0, r0, r9", "uqsub8 r0, r0, r9",
                      "uqsub8 r0, r0, r9"))
BLOCK(usub8Ind, FOUR("usub8 r0, r8, r9", "usub8 r1, r8, r9", "usub8 r2, r8, r9",
                     "usub8 r3, r8, r9"))
BLOCK(usub8Dep, FOUR("usub8 r0, r0, r9", "usub8 r0, r0, r9", "usub8 r0, r0, r9",
                     "usub8 r0, r0, r9"))
BLOCK(selDep, FOUR("sel r0, r0, r9", "sel r0, r0, r9", "sel r0, r0, r9", "sel r0, r0, r9"))
BLOCK(usub8SelChain, FOUR("usub8 r0, r0, r9", "sel r0, r0, r9", "usub8 r0, r0, r9",
                          "sel r0, r0, r9"))
BLOCK(uqsub8Orr, FOUR("uqsub8 r0, r8, r9", "orr r1, r8, r9", "uqsub8 r2, r8, r9",
                      "orr r3, r8, r9"))
BLOCK(uqsub8OrrShifted, FOUR("uqsub8 r0, r8, r9", "orr r1, r8, r9, lsl #8",
                             "uqsub8 r2, r8, r9", "orr r3, r8, r9, lsr #8"))
BLOCK(uqsub8Usub8, FOUR("uqsub8 r0, r8, r9", "usub8 r1, r8, r9", "uqsub8 r2, r8, r9",
                        "usub8 r3, r8, r9"))
BLOCK(pkhbtInd, FOUR("pkhbt r0, r8, r9, lsl #16", "pkhbt r1, r8, r9, lsl #16",
                     "pkhtb r2, r8, r9, asr #16", "pkhtb r3, r8, r9, asr #16"))
BLOCK(mulOrr, FOUR("mul r0, r8, r9", "orr r1, r8, r9", "mul r2, r8, r9", "orr r3, r8, r9"))
BLOCK(ldrAligned, FOUR("ldr r0, [%0, #0]", "ldr r1, [%0, #4]", "ldr r2, [%0, #8]",
                       "ldr r3, [%0, #12]"))
BLOCK(ldrPlus1, FOUR("ldr r0, [%0, #1]", "ldr r1, [%0, #5]", "ldr r2, [%0, #9]",
                     "ldr r3, [%0, #13]"))
BLOCK(ldrPlus2, FOUR("ldr r0, [%0, #2]", "ldr r1, [%0, #6]", "ldr r2, [%0, #10]",
                     "ldr r3, [%0, #14]"))
BLOCK(ldrdAligned, FOUR("ldrd r0, r1, [%0, #0]", "ldrd r2, r3, [%0, #8]",
                        "ldrd r0, r1, [%0, #16]", "ldrd r2, r3, [%0, #24]"))
BLOCK(ldrUqsub8, FOUR("ldr r0, [%0, #0]", "uqsub8 r1, r8, r9", "ldr r2, [%0, #4]",
                      "uqsub8 r3, r8, r9"))
BLOCK(ldrbInd, FOUR("ldrb r0, [%0, #0]", "ldrb r1, [%0, #5]", "ldrb r2, [%0, #9]",
                    "ldrb r3, [%0, #13]"))
BLOCK(strUqsub8, FOUR("str r8, [%0, #64]", "uqsub8 r1, r8, r9", "str r8, [%0, #68]",
                      "uqsub8 r3, r8, r9"))

struct Block {
    const char* name;
    uint32_t (*fn)(const uint8_t*);
};

const Block kBlocks[] = {
    {"orr, independent", orrInd},
    {"orr, dependent", orrDep},
    {"orr with a shifted operand, independent", orrShiftedInd},
    {"lsr/lsl by an immediate, independent", lsrInd},
    {"uqsub8, independent", uqsub8Ind},
    {"uqsub8, dependent", uqsub8Dep},
    {"usub8, independent", usub8Ind},
    {"usub8, dependent", usub8Dep},
    {"sel, dependent", selDep},
    {"usub8 -> sel -> usub8 chain", usub8SelChain},
    {"uqsub8 with orr", uqsub8Orr},
    {"uqsub8 with a shifted-operand orr", uqsub8OrrShifted},
    {"uqsub8 with usub8", uqsub8Usub8},
    {"pkhbt/pkhtb, independent", pkhbtInd},
    {"mul with orr", mulOrr},
    {"ldr, aligned", ldrAligned},
    {"ldr, 1 byte off alignment", ldrPlus1},
    {"ldr, 2 bytes off alignment", ldrPlus2},
    {"ldrd, aligned", ldrdAligned},
    {"ldr with uqsub8", ldrUqsub8},
    {"ldrb", ldrbInd},
    {"str with uqsub8", strUqsub8},
};

void cmdCycles(const char*) {
    boardPuts("-- cycles per instruction x100, 64 x 32 instructions, best of 3 --\n");
    for (const Block& b : kBlocks) {
        uint32_t best = 0xFFFFFFFFu;
        for (int r = 0; r < 3; ++r) {
            const uint32_t c = b.fn(g_buf);
            best = c < best ? c : best;
        }
        boardPuts("  ");
        boardPuts(b.name);
        boardPuts("  ");
        boardPutU32(best * 100u / (64u * 32u));
        boardPutc('\n');
    }
    boardPuts("OK\n");
}

const m7::Command kCommands[] = {{"CYCLES", &cmdCycles}};

}  // namespace

int main() { m7::serve("cycles", kCommands, sizeof(kCommands) / sizeof(kCommands[0])); }
