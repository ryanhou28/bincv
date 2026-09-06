// Deliberately warning-emitting translation unit -- the warning gate's own test.
//
// This file is NOT part of any normal build. Two options in the top-level
// CMakeLists.txt compile it, and scripts/verify.sh turns each on in a throwaway
// build tree and requires the build to FAIL:
//
// -DBINCV_SELFTEST_WARNING=ON builds it as a properly wired target, so
// -Werror must reject it. Proves the flags are
// reaching first-party sources and are fatal.
//
// -DBINCV_SELFTEST_UNWIRED=ON builds it as a target that omits
// bincv_warnings, so bincv_assert_warning_policy
// must reject it at CONFIGURE time. Proves the
// wiring assertion fires -- this is the case a
// build-log scan structurally cannot catch, since
// an unwired target emits nothing to scan.
//
// A gate that has never been observed to fail is not known to work; both of
// these have to stay compilable-but-unclean for that observation to be possible.
// If a future compiler stops warning about one of these three, the self-check
// will say so by passing when it should fail, and another diagnostic goes here.

#include <cstddef>
#include <cstdint>

// -Wconversion: narrows without a cast.
static std::uint8_t bincvSelftestNarrow(std::uint32_t v) { return v; }

int main() {
    int unusedLocal = 0;              // -Wunused-variable (-Wall)
    std::size_t n = 4;
    int i = 2;
    if (i < n) {                      // -Wsign-compare (-Wall/-Wextra)
        return static_cast<int>(bincvSelftestNarrow(7u));
    }
    return 0;
}
