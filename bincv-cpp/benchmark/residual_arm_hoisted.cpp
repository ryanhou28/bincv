// X-33 arm SCALAR -- residualSums with the NEON path forced off.
//
// This file previously held X-32's rejected tap-hoisting arm. That arm is recorded
// in X-32 with its measurement and is not kept as code: a rejected optimisation
// needs a number in the log, not a second implementation to maintain. The slot now
// holds the SCALAR arm, which is what the NEON path must be bit-identical to.
#include "residual_arms.hpp"

namespace residual {
void hoisted(const bincv::LKLevelN<2, uint32_t>& lv,
             const bincv::impl::RegionWords<uint32_t>& r, long long tapX, long long tapY,
             bincv::impl::TapSums& sx, bincv::impl::TapSums& sy) {
    bincv::impl::residualSums<2, uint32_t, false>(lv, r, tapX, tapY, sx, sy);
}
} // namespace residual
