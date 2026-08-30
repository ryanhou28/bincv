// ===========================================================================
// core/simd.hpp -- which fast paths this build actually compiled.
//
// F-5: a binCV user added `-I .../include` and never linked `bincv_core`, so every NEON
// kernel was #ifdef-ed out. Nothing warned and nothing computed a different answer --
// the vector kernels are bit-exact with the scalar ones -- and their tracker ran 2.25x
// slower than it should have on a Pi 4. Measured on this repo's own diagnostic, on the
// reference device: 20.31 ms with NEON against 36.16 ms without, 1.78x.
//
// THE REAL GUARD FOR THAT IS NOT IN THIS FILE, and it cannot be: this suite links
// `bincv_core`, so it gets the define either way and would pass through the whole bug.
// `scripts/check_arm_syntax.sh` compiles a translation unit with NO defines at all and
// fails if `BINCV_HAVE_NEON` is absent -- that is the test, and it has been watched to
// fail. What this file pins is the reported STATUS being consistent with the build it
// is reporting on, which is what a consumer logs and acts on.
// ===========================================================================

#include <cstdio>
#include <cstring>

#include "bincv-cpp/core/simd.hpp"
#include "test_util.hpp"

BINCV_TEST(Simd, StatusAgreesWithTheBuildItDescribes) {
    const bincv::SimdStatus s = bincv::simdStatus();
    std::printf("  %s\n", bincv::simdStatusString());

    // Exactly one architecture, and it is the one the compiler thinks it is.
#if defined(__aarch64__)
    BINCV_CHECK(s.isAarch64);
    BINCV_CHECK(!s.isX86);
    // ARMv8 makes NEON mandatory, so there is no aarch64 build that legitimately lacks
    // it -- which is precisely why deriving the define from the compiler works and why
    // taking it from a CMake target was the bug.
    BINCV_CHECK(s.neon);
    // `cnt` is unconditional on ARMv8; there is no flag to forget.
    BINCV_CHECK(s.hardwarePopcount);
#elif defined(__x86_64__) || defined(__i386__)
    BINCV_CHECK(s.isX86);
    BINCV_CHECK(!s.isAarch64);
    BINCV_CHECK(!s.neon);
#endif

    // The struct and the string cannot disagree: the string is what a consumer reads.
    const char* line = bincv::simdStatusString();
    BINCV_CHECK(line != nullptr && std::strlen(line) > 0);
    BINCV_CHECK(std::strstr(line, "binCV SIMD:") != nullptr);
    BINCV_CHECK(std::strstr(line, s.neon ? "NEON=yes" : "NEON=NO") != nullptr);
    BINCV_CHECK(std::strstr(line, s.hardwarePopcount ? "popcount=hardware"
                                                     : "popcount=SOFTWARE") != nullptr);

    // THE VERDICT IS THE POINT OF THE LINE. A build missing a fast path must say so in
    // words -- a reader who has to work out which combination is bad will not notice
    // the bad one.
    const bool slow = (s.isAarch64 && !s.neon) || (s.isX86 && !s.hardwarePopcount);
    BINCV_CHECK((std::strstr(line, "SLOW") != nullptr) == slow);
    if (!slow) BINCV_CHECK(std::strstr(line, "fast paths active") != nullptr);

    // The define and the report are the same fact, so they cannot drift apart.
#if defined(BINCV_HAVE_NEON)
    BINCV_CHECK(s.neon);
#else
    BINCV_CHECK(!s.neon);
#endif
}

BINCV_TEST_MAIN("test_simd")
