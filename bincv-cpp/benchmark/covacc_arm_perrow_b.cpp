// X-29 arm P' -- BIT-IDENTICAL SOURCE TO P, in a second translation unit.
//
// This arm measures nothing about the algorithm. Any difference between P and P'
// is pure code layout, and that difference is the NOISE FLOOR L against which
// X-29's rule judges the real comparison. Without it, a 1.14x reading inside a
// 1.46x confound would be indistinguishable from a result -- which is exactly why
// X-22 declined to close on this question.
#include "covacc_arm_body.inc"

namespace covacc {
int64_t perRowB(const bincv::BinMatConstView<uint32_t>* magX,
                const bincv::BinMatConstView<uint32_t>* magY,
                bincv::BinMatConstView<uint32_t> signX, bincv::BinMatConstView<uint32_t> signY,
                size_t n, const bincv::Rect* windows, size_t windowCount) {
    return dispatch(magX, magY, signX, signY, n, windows, windowCount);
}
} // namespace covacc
