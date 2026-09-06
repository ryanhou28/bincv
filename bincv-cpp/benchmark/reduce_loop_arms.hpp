/// @file reduce_loop_arms.hpp
/// @brief Four shapes for the interior of a bulk bit-count, compared against each
/// other. Benchmark code, not library code -- these are a prototype of a change to
/// ops/reduce.hpp's loops, not the change itself.
///
/// ---------------------------------------------------------------------------
/// WHAT IS BEING ASKED
///
/// reduce.hpp says of its per-word counter: "Phase 5 replaces the LOOPS below, not
/// this function." Measuring the per-word counter on a Cortex-M7 said the same thing
/// from the other side -- hand-writing the arithmetic gained nothing, because the
/// compiler already emits a good SWAR. So the question is what the LOOP costs, and
/// whether a better loop is portable rather than per-chip.
///
/// The four arms differ ONLY in loop structure. None uses an intrinsic, inline
/// assembly, or a target `#if`, so whatever they show applies to an architecture
/// family rather than to one part.
///
/// ---------------------------------------------------------------------------
/// THE TWO FAMILIES, WHICH IS WHY THIS IS TIMED WITH POPCOUNT BOTH ON AND OFF
///
/// - **No hardware population count** (Cortex-M, RISC-V without Zbb, x86 without
///   POPCNT): the per-word count is a ~10-operation SWAR ending in a horizontal
///   collapse. That collapse is what arm L2 amortizes, so this is the family where
///   it can pay.
/// - **Hardware population count** (x86 with POPCNT, aarch64): the per-word count is
///   one instruction. There is no collapse to amortize and L2 should LOSE, because
///   it replaces one instruction with ten. What can still help there is breaking the
///   accumulator dependency chain, which is L1.
///
/// aarch64 is in the second family, so a result from the first does not carry to it.
/// Building x86 both ways is what makes that second family measurable here.
#ifndef BINCV_BENCHMARK_REDUCE_LOOP_ARMS_HPP
#define BINCV_BENCHMARK_REDUCE_LOOP_ARMS_HPP

#include <cstddef>
#include <cstdint>

#include "bincv-cpp/ops/reduce.hpp"

namespace loopbench {

using bincv::impl::popcountWord;

/// **L0 -- the shipped shape.** One accumulator, one call per word. This is what
/// `impl::countRowRegion` does for a row's interior; the all-ones mask that
/// `visitRowWords` ANDs onto every interior word is omitted because every compiler
/// folds it away, and keeping it would time the fold rather than the loop.
inline uint64_t l0Shipped(const uint32_t* p, size_t n) {
    uint64_t total = 0;
    for (size_t i = 0; i < n; ++i) total += popcountWord<uint32_t>(p[i]);
    return total;
}

/// **L1 -- four independent accumulators.** Same per-word count as L0; the only
/// change is that four partial sums remove the serial dependency between adjacent
/// adds, so a core that can issue more than one at a time is allowed to.
///
/// reduce.hpp already does this ACROSS ROWS -- each row returns its own partial sum,
/// measured at 1.03-1.09x -- on the argument that the accumulator, not the work, sets
/// the pace. L1 asks whether the same argument holds WITHIN a row, which is where the
/// words actually are.
inline uint64_t l1FourAccumulators(const uint32_t* p, size_t n) {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 += popcountWord<uint32_t>(p[i + 0]);
        a1 += popcountWord<uint32_t>(p[i + 1]);
        a2 += popcountWord<uint32_t>(p[i + 2]);
        a3 += popcountWord<uint32_t>(p[i + 3]);
    }
    for (; i < n; ++i) a0 += popcountWord<uint32_t>(p[i]);
    return a0 + a1 + a2 + a3;
}

/// The group size L2 and L3 accumulate over, chosen by what the lanes can hold.
///
/// After the nibble step each of the four bytes of a word holds that byte's count,
/// so at most 8. Adding `kGroup` words lane-wise leaves at most `8 * kGroup` in a
/// byte, and a byte holds 255, so the arithmetic is exact for `kGroup <= 31`.
/// 16 is used: comfortably inside the bound, a power of two, and already 16x fewer
/// horizontal collapses.
constexpr size_t kGroup = 16;
static_assert(8 * kGroup <= 255, "byte lanes would overflow before the collapse");
static_assert(kGroup % 2 == 0, "L3 consumes the group two words at a time");

/// **L2 -- amortize the horizontal collapse.** The SWAR count is two parts: reduce a
/// word to four per-byte counts (cheap, unavoidable), then collapse those four bytes
/// into one number (the expensive tail). L0 pays the tail once per word. L2 keeps the
/// byte lanes in an accumulator and pays it once per `kGroup` words.
///
/// This is plain C++ and helps any target whose count is a software SWAR. On a target
/// with a population count instruction there is no tail to amortize and this should
/// lose -- which is a prediction the measurement can falsify.
inline uint64_t l2GroupedCollapse(const uint32_t* p, size_t n) {
    uint64_t total = 0;
    size_t i = 0;
    for (; i + kGroup <= n; i += kGroup) {
        uint32_t acc = 0;
        for (size_t k = 0; k < kGroup; ++k) {
            uint32_t v = p[i + k];
            v = v - ((v >> 1) & 0x55555555u);
            v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
            v = (v + (v >> 4)) & 0x0F0F0F0Fu;
            acc += v;  // byte lanes; each <= 8 * kGroup
        }
        // Collapse once for the whole group. 16-bit lanes hold <= 16 * kGroup.
        acc = (acc & 0x00FF00FFu) + ((acc >> 8) & 0x00FF00FFu);
        total += (acc & 0xFFFFu) + (acc >> 16);
    }
    for (; i < n; ++i) total += popcountWord<uint32_t>(p[i]);
    return total;
}

/// **L3 -- L2, plus merging two words before the nibble step.** After the 2-bit step
/// each 4-bit field holds at most 4, so two words' fields can be added and still fit
/// a nibble (4 + 4 = 8 <= 15). The nibble step is then paid once per PAIR instead of
/// once per word, on top of L2's amortized collapse. It saves about one operation per
/// word out of eleven, so it is carried as a diagnostic rather than as a candidate:
/// if L3 does not beat L2, what remains is not in the body.
///
/// **The nibble step must be mask-then-add here, not the cheaper add-then-mask.**
/// `(v + (v >> 4)) & 0x0F0F0F0F` is only correct while nibbles are <= 4, because it
/// lets adjacent nibbles sum inside the field first. Merged pairs make them <= 8, so
/// that form reaches 16 and carries into the neighbouring nibble; masking first puts
/// each sum in a whole byte, where <= 16 is comfortable. One extra AND per pair, and
/// without it the arm is silently wrong on dense inputs.
inline uint64_t l3PairMerged(const uint32_t* p, size_t n) {
    uint64_t total = 0;
    size_t i = 0;
    for (; i + kGroup <= n; i += kGroup) {
        uint32_t acc = 0;
        for (size_t k = 0; k < kGroup; k += 2) {
            uint32_t a = p[i + k];
            uint32_t b = p[i + k + 1];
            a = a - ((a >> 1) & 0x55555555u);
            b = b - ((b >> 1) & 0x55555555u);
            a = (a & 0x33333333u) + ((a >> 2) & 0x33333333u);
            b = (b & 0x33333333u) + ((b >> 2) & 0x33333333u);
            uint32_t v = a + b;  // nibbles <= 8
            v = (v & 0x0F0F0F0Fu) + ((v >> 4) & 0x0F0F0F0Fu);  // bytes <= 16 per pair
            acc += v;  // byte lanes; <= 16 * (kGroup / 2) == 8 * kGroup
        }
        acc = (acc & 0x00FF00FFu) + ((acc >> 8) & 0x00FF00FFu);
        total += (acc & 0xFFFFu) + (acc >> 16);
    }
    for (; i < n; ++i) total += popcountWord<uint32_t>(p[i]);
    return total;
}

struct Arm {
    const char* name;
    uint64_t (*run)(const uint32_t*, size_t);
};

inline const Arm* arms(size_t& count) {
    static const Arm kArms[] = {
        {"L0 shipped (1 acc, call/word)", &l0Shipped},
        {"L1 four accumulators         ", &l1FourAccumulators},
        {"L2 grouped collapse          ", &l2GroupedCollapse},
        {"L3 grouped + pair merge      ", &l3PairMerged},
    };
    count = sizeof(kArms) / sizeof(kArms[0]);
    return kArms;
}

}  // namespace loopbench

#endif
