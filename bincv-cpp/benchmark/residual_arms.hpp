#pragma once

// X-32 -- the two arms, one translation unit each.
//
//   residual_arm_shipped.cpp   S, the control: impl::residualSums as it ships,
//                              four ReplicatedShiftedRow::word() calls per word.
//   residual_arm_hoisted.cpp   H: two calls per word plus two shifts, deriving
//                              t01 from t00 and t11 from t10.
//
// The derivation is an IDENTITY, not an approximation: t01's pixel x is
// next[x + tapX + 1], which is t00's pixel x + 1, so on the word grid
// t01_i = (t00_i >> 1) | (t00_{i+1} << (bits-1)). It holds through the replicate
// border because both taps clamp on the same absolute column. The ten sums must
// therefore come out IDENTICAL, and the benchmark checks that before it times
// anything.

#include <cstddef>
#include <cstdint>

#include "bincv-cpp/ops/opticalFlow.hpp"

namespace residual {

void shipped(const bincv::LKLevelN<2, uint32_t>& lv,
             const bincv::impl::RegionWords<uint32_t>& r, long long tapX, long long tapY,
             bincv::impl::TapSums& sx, bincv::impl::TapSums& sy);
void hoisted(const bincv::LKLevelN<2, uint32_t>& lv,
             const bincv::impl::RegionWords<uint32_t>& r, long long tapX, long long tapY,
             bincv::impl::TapSums& sx, bincv::impl::TapSums& sy);

} // namespace residual
