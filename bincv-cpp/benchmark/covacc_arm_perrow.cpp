// X-29 arm P -- the shipped per-row accumulator. Body shared with P' by inclusion.
#include "covacc_arm_body.inc"

namespace covacc {
int64_t perRow(const bincv::BinMatConstView<uint32_t>* magX,
               const bincv::BinMatConstView<uint32_t>* magY,
               bincv::BinMatConstView<uint32_t> signX, bincv::BinMatConstView<uint32_t> signY,
               size_t n, const bincv::Rect* windows, size_t windowCount) {
    return dispatch(magX, magY, signX, signY, n, windows, windowCount);
}
} // namespace covacc
