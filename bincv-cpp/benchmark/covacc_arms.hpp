#pragma once

// X-29 / E-13 -- the interface the FOUR translation units share.
//
// The question is whether D-15 item 4's PER-ROW PARTIAL ACCUMULATOR still pays
// above N = 1 in the bit-sliced covariance. `BitSlicedPairCounts<N>` is 4N^2
// counters -- 4 at N = 1, 64 at N = 4 -- and the per-row form zeroes all of them
// per row, fills them, and adds them into the window total, against work that is
// O(N^2) per WORD with only 1-2 uint64_t words per row at a 31-pixel window.
//
//   covacc_arm_perrow.cpp    P, the control: the shipped shape -- a fresh
//                            BitSlicedPairCounts per row, added into the total.
//   covacc_arm_window.cpp    W: one accumulator for the whole window, rows
//                            accumulated straight into it. No per-row zero, no add.
//   covacc_arm_perrow_b.cpp  P', BIT-IDENTICAL SOURCE TO P, in a second
//                            translation unit. This arm measures nothing about the
//                            algorithm; it measures the NOISE FLOOR.
//
// WHY P' EXISTS, AND WHY IT IS THE MOST IMPORTANT ARM HERE
//
// X-22 already measured a window-wide accumulator 1.14-1.60x faster at N = 2,3,4
// and DECLINED TO CLOSE ON IT, because that same entry measured the SAME kernel
// moving 1.46x between two binaries built from unchanged source, and
// morphology_path_benchmark records two instantiations in one object moving each
// other ~10% through code layout alone. A 1.14x reading inside a 1.46x confound is
// not a result.
//
// P and P' are the same algorithm compiled into different objects, so any
// difference between them is PURE LAYOUT. That difference is the noise floor L,
// and X-29's decision rule judges W against P by L rather than against zero. A
// comparison whose noise floor is not stated cannot be checked by a reader, so L
// is reported in every band -- including the ones where it is not decisive.

#include <cstddef>
#include <cstdint>

#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/covariance.hpp"

namespace covacc {

/// One timed call: the covariance over `windows` window positions.
/// @return a checksum, so nothing can be dead-code eliminated.
using Fn = int64_t (*)(const bincv::BinMatConstView<uint32_t>* magX,
                       const bincv::BinMatConstView<uint32_t>* magY,
                       bincv::BinMatConstView<uint32_t> signX,
                       bincv::BinMatConstView<uint32_t> signY, size_t n,
                       const bincv::Rect* windows, size_t windowCount);

/// P -- the shipped per-row shape.
int64_t perRow(const bincv::BinMatConstView<uint32_t>* magX,
               const bincv::BinMatConstView<uint32_t>* magY,
               bincv::BinMatConstView<uint32_t> signX, bincv::BinMatConstView<uint32_t> signY,
               size_t n, const bincv::Rect* windows, size_t windowCount);

/// P' -- bit-identical source, second translation unit. Measures layout only.
int64_t perRowB(const bincv::BinMatConstView<uint32_t>* magX,
                const bincv::BinMatConstView<uint32_t>* magY,
                bincv::BinMatConstView<uint32_t> signX, bincv::BinMatConstView<uint32_t> signY,
                size_t n, const bincv::Rect* windows, size_t windowCount);

/// W -- one accumulator for the whole window.
int64_t windowWide(const bincv::BinMatConstView<uint32_t>* magX,
                   const bincv::BinMatConstView<uint32_t>* magY,
                   bincv::BinMatConstView<uint32_t> signX, bincv::BinMatConstView<uint32_t> signY,
                   size_t n, const bincv::Rect* windows, size_t windowCount);

} // namespace covacc
