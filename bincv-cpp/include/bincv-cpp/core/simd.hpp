#pragma once

/// @file simd.hpp
/// @brief Which vector paths this build actually compiled, and the auto-detection that
///        stops one of them from going missing. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// F-5: THE FAST PATH USED TO RIDE ON A CMAKE TARGET, AND MISSING IT WAS SILENT
///
/// `BINCV_HAVE_NEON` and `-mpopcnt` are INTERFACE properties of the `bincv_core` CMake
/// target. binCV is header-only, so an integrator can do the natural thing --
/// `-I .../include`, no `target_link_libraries` -- and get a **correct** library with
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
/// (D-73) and invisible for the same reason.
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
/// table lookup, worth 3.75x (X-57). No header can add a compiler flag to a translation
/// unit it is being included into. What this header does instead is make the omission
/// **visible** -- `simdStatus()` reports it, so a consumer can log one line and see it.
///
/// ---------------------------------------------------------------------------
/// USE IT
///
/// ```cpp
/// std::printf("binCV: %s\n", bincv::simdStatusString());
/// // binCV SIMD: NEON=yes AVX2=n/a popcount=hardware  (all fast paths active)
/// // binCV SIMD: NEON=NO   AVX2=n/a popcount=software (SLOW -- link bincv_core)
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

#include "error.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief What this translation unit compiled, and what the CPU under it supports.
/// @note Every field is about the BUILD except `avx2Runtime`, which is about the CPU.
struct SimdStatus {
    bool neon = false;             ///< NEON kernels compiled in (aarch64, or armv7 + flag)
    bool avx2Compiled = false;     ///< AVX2 kernels compiled in (x86 with a GNU-ish compiler)
    bool avx2Runtime = false;      ///< ...and this CPU supports AVX2
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
///       than I expected" for the two failure modes that produce no other symptom --
///       and both of them are silent because the fast and slow paths agree exactly.
/// @note Returns a pointer to a function-local static; valid for the program's lifetime
///       and not to be freed.
inline const char* simdStatusString() {
    static char buf[160];
    const SimdStatus s = simdStatus();
    const char* avx2 = !s.isX86          ? "n/a"
                       : !s.avx2Compiled ? "NOT COMPILED"
                       : s.avx2Runtime   ? "yes"
                                         : "compiled, unsupported by this CPU";
    // The verdict is spelled out because a reader who has to work out which combination
    // is bad is a reader who will not notice the bad one.
    const bool slow = (s.isAarch64 && !s.neon) || (s.isX86 && !s.hardwarePopcount);
    std::snprintf(buf, sizeof(buf),
                  "binCV SIMD: NEON=%s AVX2=%s popcount=%s  (%s)", s.neon ? "yes" : "NO",
                  avx2, s.hardwarePopcount ? "hardware" : "SOFTWARE",
                  slow ? "SLOW -- link the bincv_core target, do not just add its include path"
                       : "fast paths active");
    return buf;
}

}  // namespace BINCV_ABI_NAMESPACE
}  // namespace bincv
