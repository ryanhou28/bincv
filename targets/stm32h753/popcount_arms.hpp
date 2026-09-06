/// @file popcount_arms.hpp
/// @brief The four ways of counting one word that this target compares, and the one
/// bulk loop they are compared inside. Harness code -- not part of binCV.
///
/// Every arm is compiled into every image. The build does not select one: the
/// firmware runs all four and requires them to agree, which is what makes a
/// mis-attached `#define` impossible to mistake for a result (CLAUDE.md).
#ifndef BINCV_EMBEDDED_STM32H753_POPCOUNT_ARMS_HPP
#define BINCV_EMBEDDED_STM32H753_POPCOUNT_ARMS_HPP

#include <cstddef>
#include <cstdint>

#include "bincv/ops/reduce.hpp"

namespace armbench {

/// Arm A -- what binCV ships. `impl::popcountWord` is `__builtin_popcountll` on GCC.
/// The `unsigned long long` cast does NOT cost a 64-bit count here: GCC sees the top
/// half is zero and calls libgcc's 32-bit `__popcountsi2`. That routine is arm C's
/// SWAR almost instruction for instruction, so A and C differ by a function call per
/// word and essentially nothing else -- which is what this benchmark prices.
inline uint32_t popcountA(uint32_t w) {
    return static_cast<uint32_t>(bincv::impl::popcountWord<uint32_t>(w));
}

/// Arm B -- `impl::popcountWordPortable`, which reduce.hpp names as "the sequence a
/// Cortex-M build compiles to". It widens to `uint64_t` too, so on this machine its
/// last step is a 64-bit multiply.
inline uint32_t popcountB(uint32_t w) {
    return static_cast<uint32_t>(bincv::impl::popcountWordPortable<uint32_t>(w));
}

/// Arm C -- the same SWAR shape that never leaves 32 bits.
inline uint32_t popcountC(uint32_t v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;
    return (v * 0x01010101u) >> 24;
}

/// Arm D -- arm C with the final byte-sum done by the DSP extension. After the
/// nibble step each byte holds its own count, and `USAD8` against zero is exactly
/// "add the four bytes", replacing a multiply and a shift with one instruction.
///
/// Written as inline asm rather than `__usad8` from <arm_acle.h> so the arm does not
/// depend on which ACLE intrinsics a given GCC exposes; the encoding is stable.
inline uint32_t popcountD(uint32_t v) {
#if defined(__ARM_FEATURE_SIMD32)
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;
    uint32_t r;
    __asm volatile("usad8 %0, %1, %2" : "=r"(r) : "r"(v), "r"(0u));
    return r;
#else
    return popcountC(v);
#endif
}

/// The bulk walk every arm is timed inside: whole words, no masking, which is the
/// interior of `countNonZero`'s row loop. Templated on the counter so all four see
/// an identical loop and the only difference is the arm.
///
/// `noinline` matters. Without it the compiler inlines one arm's body into the timing
/// site and not another's, and the comparison becomes a comparison of inlining
/// decisions.
template <uint32_t (*Pop)(uint32_t)>
__attribute__((noinline)) uint64_t sumWords(const uint32_t* p, size_t n) {
    uint64_t total = 0;
    for (size_t i = 0; i < n; ++i) total += Pop(p[i]);
    return total;
}

}  // namespace armbench

#endif
