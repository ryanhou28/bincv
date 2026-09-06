#pragma once

/// @file simd.hpp
/// @brief Which vector paths this build actually compiled, and the auto-detection that
/// stops one of them from going missing. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// F-5: THE FAST PATH USED TO RIDE ON A CMAKE TARGET, AND MISSING IT WAS SILENT
///
/// `BINCV_HAVE_NEON` and `-mpopcnt` are INTERFACE properties of the `bincv_core` CMake
/// target. binCV is header-only, so an integrator can do the natural thing --
/// `-I.../include`, no `target_link_libraries` -- and get a **correct** library with
/// every NEON kernel `#ifdef`-ed out. Nothing warns, and nothing computes a different
/// answer, because the vector kernels are bit-exact with the scalar ones.
///
/// Reported from outside, measured on a Pi 4:
///
/// | keypoint tracking | |
/// |---|---|
/// | include path only | **42.83 ms** |
/// | linking `bincv_core` | **19.03 ms** |
///
/// **2.25x from one CMake line**, and without catching it they would have reported that
/// binCV's tracker is slower than OpenCV's on ARM. Same shape as the `uint64_t` trap
/// and invisible for the same reason.
///
/// ---------------------------------------------------------------------------
/// THE FIX IS DETECTION, NOT A DIAGNOSTIC
///
/// **On aarch64 the compiler already tells us.** NEON is mandatory in ARMv8, so
/// `__ARM_NEON` and `__aarch64__` are defined with no flags at all -- which means
/// `BINCV_HAVE_NEON` never needed to come from CMake on that target, and making it do so
/// is what tied the fast path to a link line. This header defines it from the
/// compiler's own macros, so **an include-only integration on aarch64 now gets the NEON
/// kernels**. The CMake definition stays for armv7, where `__ARM_NEON` appears only with
/// `-mfpu=neon` and the flag genuinely is a build-system choice.
///
/// **`-mpopcnt` cannot be fixed this way and is not pretended away.** It changes code
/// generation rather than gating a `#if`: without it `__builtin_popcount` becomes a
/// table lookup, worth 3.75x. No header can add a compiler flag to a translation
/// unit it is being included into. What this header does instead is make the omission
/// **visible** -- `simdStatus` reports it, so a consumer can log one line and see it.
///
/// ---------------------------------------------------------------------------
/// USE IT
///
/// ```cpp
/// std::printf("binCV: %s\n", bincv::simdStatusString);
/// // binCV SIMD: NEON=yes AVX2=n/a popcount=hardware (all fast paths active)
/// // binCV SIMD: NEON=NO AVX2=n/a popcount=software (SLOW -- link bincv_core)
/// ```

// -------------------------------------------------------------------------------
// THE AUTO-DETECTION. Must come before any `#if defined(BINCV_HAVE_NEON)`, which is
// why this header is included explicitly at the top of every file that has one --
// three of them gate BEFORE their first core include, so relying on transitive
// inclusion would have re-created F-5 in a new place.
// -------------------------------------------------------------------------------
// `BINCV_NO_NEON` forces the scalar arm, and it is not a convenience: CLAUDE.md
// requires that a vector arm be switchable off so a benchmark can time both and show
// which one it is running. `BINCV_HAVE_NEON` used to be that switch by accident --
// leaving it undefined disabled NEON -- and auto-defining it would have taken the
// ability away. This restores it deliberately instead.
#if !defined(BINCV_HAVE_NEON) && !defined(BINCV_NO_NEON) && defined(__ARM_NEON) && \
    defined(__aarch64__)
#define BINCV_HAVE_NEON 1
#endif
#if defined(BINCV_NO_NEON) && defined(BINCV_HAVE_NEON)
#error "BINCV_NO_NEON and BINCV_HAVE_NEON are both defined -- pick one"
#endif
// -------------------------------------------------------------------------------
// M-PROFILE HAS NO NEON, AND THE COMPILER WILL NOT SAY SO ON ITS OWN.
//
// `arm-none-eabi-g++ -mcpu=cortex-m7 -mfpu=neon` exits 0, emits no diagnostic, and
// defines `__ARM_NEON`. So a build system that concludes "this target has NEON"
// from a `-mfpu=neon` compile check concludes it on a Cortex-M7, and `simdStatus`
// then reports NEON=yes on a part that has no vector unit at all. That is F-5 with
// the sign reversed: not a fast path silently missing, but a fast path silently
// claimed -- and it is worse, because the number it would corrupt is the population
// count measurement that the M-profile port exists to make.
//
// `__ARM_ARCH_PROFILE` is the macro that does know: 'M' on Cortex-M, 'A' on
// Cortex-A. Consulting it keeps the gate on the compiler's own macros, which is
// where this project puts a gate the compiler can decide.
//
// An #error rather than a quiet #undef, because on an M-profile target the define
// can only have come from a build system asserting something untrue, and the line
// GETTING_STARTED tells a reader to log at start-up is the one place that must not
// lie about which paths are live.
#if defined(BINCV_HAVE_NEON) && defined(__ARM_ARCH_PROFILE) && (__ARM_ARCH_PROFILE == 'M')
#error "BINCV_HAVE_NEON is defined for an M-profile target, which has no NEON. A -mfpu=neon compile check is not a NEON test on Cortex-M: it succeeds there. Drop the define; the scalar path is the only path on this part."
#endif

#include "error.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief What this translation unit compiled, and what the CPU under it supports.
/// @note Every field is about the BUILD except `avx2Runtime`, which is about the CPU.
struct SimdStatus {
    bool neon = false;             ///< NEON kernels compiled in (aarch64, or armv7 + flag)
    bool avx2Compiled = false;     ///< AVX2 kernels compiled in (x86 with a GNU-ish compiler)
    bool avx2Runtime = false;      ///<...and this CPU supports AVX2
    bool hardwarePopcount = false; ///< `-mpopcnt` on x86; always true on aarch64 (`cnt`)
    bool isX86 = false;
    bool isAarch64 = false;
};

/// @brief What vector paths are actually active. **API TIER 3.**
/// @note Cheap and safe to call at start-up; the AVX2 runtime probe is cached.
inline SimdStatus simdStatus() {
    SimdStatus s;
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
    s.neon = true;
#elif defined(BINCV_HAVE_NEON)
    s.neon = true;
#endif
#if defined(__aarch64__)
    s.isAarch64 = true;
    // ARMv8's `cnt` is unconditional -- there is no aarch64 without it, so there is no
    // flag to forget.
    s.hardwarePopcount = true;
#endif
#if defined(__x86_64__) || defined(__i386__)
    s.isX86 = true;
#if defined(__POPCNT__)
    s.hardwarePopcount = true;
#endif
#if defined(__GNUC__) || defined(__clang__)
    s.avx2Compiled = true;
    static const bool kAvx2 = __builtin_cpu_supports("avx2");
    s.avx2Runtime = kAvx2;
#endif
#endif
    return s;
}

/// @brief One line naming every fast path and whether it is on. **API TIER 3.**
/// @note **LOG THIS ONCE AT START-UP.** It is the whole answer to "why is binCV slower
/// than I expected" for the two failure modes that produce no other symptom --
/// and both of them are silent because the fast and slow paths agree exactly.
/// @note Returns a pointer to a function-local static; valid for the program's lifetime
/// and not to be freed.
inline const char* simdStatusString() {
    static char buf[160];
    const SimdStatus s = simdStatus();
    const char* avx2 = !s.isX86          ? "n/a"
                       : !s.avx2Compiled ? "NOT COMPILED"
                       : s.avx2Runtime   ? "yes"
                                         : "compiled, unsupported by this CPU";
    // The verdict is spelled out because a reader who has to work out which combination
    // is bad is a reader who will not notice the bad one.
    //
    // Three verdicts, not two. A target with no vector unit and no population count
    // instruction is not misconfigured -- the software path is the ONLY path, and
    // "link bincv_core" is advice that would change nothing there. Cortex-M is that
    // target, and with only the two flags below this line read
    // "popcount=SOFTWARE (fast paths active)" on a Cortex-M7: self-contradictory in
    // its own sentence, on the one line GETTING_STARTED tells a reader to trust.
    // `neon` keeps armv7-A out of this branch, where NEON is a real fast path that
    // the aarch64 flag does not cover.
    const bool noFastPath = !s.isX86 && !s.isAarch64 && !s.neon;
    const bool slow = (s.isAarch64 && !s.neon) || (s.isX86 && !s.hardwarePopcount);
    const char* verdict =
        noFastPath ? "scalar only -- this target has no vector or popcount instruction"
        : slow     ? "SLOW -- link the bincv_core target, do not just add its include path"
                   : "fast paths active";
    std::snprintf(buf, sizeof(buf),
                  "binCV SIMD: NEON=%s AVX2=%s popcount=%s (%s)", s.neon ? "yes" : "NO",
                  avx2, s.hardwarePopcount ? "hardware" : "SOFTWARE", verdict);
    return buf;
}

}  // namespace BINCV_ABI_NAMESPACE
}  // namespace bincv
