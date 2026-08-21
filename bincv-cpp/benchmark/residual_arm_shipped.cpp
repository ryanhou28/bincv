// X-32 arm S -- the shipped residualSums. The control.
#include "residual_arms.hpp"

namespace residual {
void shipped(const bincv::LKLevelN<2, uint32_t>& lv,
             const bincv::impl::RegionWords<uint32_t>& r, long long tapX, long long tapY,
             bincv::impl::TapSums& sx, bincv::impl::TapSums& sy) {
    bincv::impl::residualSums(lv, r, tapX, tapY, sx, sy);
}
} // namespace residual
