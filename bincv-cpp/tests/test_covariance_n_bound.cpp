// ---------------------------------------------------------------------------
// tests/test_covariance_n_bound.cpp -- a translation unit that MUST NOT COMPILE.
//
// T3.10's plane-ARRAY overload of gradientCovariance() is templated on the plane
// count N and, unlike every sibling N-templated entry point in the library
// (QuantMat 1..8, SignedQuantMat 1..7, pyrDownRoute, derivativeX/derivativeY), it
// shipped without a bound on it. Nothing in the library can reach an out-of-range
// N -- a derivative caps at 7 planes -- but the overload's own docstring blesses
// HAND-ASSEMBLED view arrays, and there the weighting in combineBitSlicedPairs,
//
//     const int64_t weight = static_cast<int64_t>(1) << (i + j);
//
// is undefined behaviour as soon as 2N - 2 >= 64, and the weights overflow int64_t
// well before that. Measured before the fix, with a 40-element view array under
// -fsanitize=undefined: "covariance.hpp:536:60: runtime error: shift exponent 64 is
// too large for 64-bit type 'long int'".
//
// The fix is a static_assert. A static_assert in a function body is not
// SFINAE-detectable, so the SFINAE traits test_covariance.cpp uses for its dispatch
// cases cannot see it -- which is why this is a whole translation unit and a ctest
// case that BUILDS it, registered WILL_FAIL. ctest passes only when this file fails
// to compile. Delete the static_assert and this file compiles, links and runs, and
// the case goes red: that is the property, and it is the reason to prefer a build
// that fails over a comment that asks.
//
// N = 9 rather than 40: it is the FIRST rejected value, so the case pins where the
// bound is and not merely that one exists. Everything above 9 is rejected by the
// same assert.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/covariance.hpp"

int main() {
    using Word = uint32_t;
    bincv::BinMat<Word> plane(64, 8);

    bincv::BinMatConstView<Word> magX[9];
    bincv::BinMatConstView<Word> magY[9];
    for (int p = 0; p < 9; ++p) {
        magX[static_cast<size_t>(p)] = plane.constView();
        magY[static_cast<size_t>(p)] = plane.constView();
    }

    // N is deduced as 9 from the array type, which is one past the bound.
    const bincv::GradientCovariance c = bincv::gradientCovariance<9, Word>(
        magX, magY, plane.constView(), plane.constView(), bincv::Rect(0, 0, 8, 8));
    return static_cast<int>(c.sumXX);
}
