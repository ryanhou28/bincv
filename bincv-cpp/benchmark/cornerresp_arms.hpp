#pragma once

// X-31 -- the interface the THREE ARMS share, and the box-sum body two of them
// use. Each arm is compiled into its own translation unit (X-29 measured why).
//
//   cornerresp_arm_perpixel.cpp     C, the control: the shipped
//                                   cornerMinEigenVal -- one clipRegion +
//                                   ~12 popcounts PER PIXEL, over a window three
//                                   bits wide in a 32-bit word.
//   cornerresp_arm_sliced.cpp       B1: 3x3 box sums computed word-at-a-time with
//                                   shifts and full adders. 32 pixels per word.
//   cornerresp_arm_sliced_skip.cpp  B2: B1 plus a word-level sparsity skip.

#include <cstddef>
#include <cstdint>

#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/corner.hpp"

namespace cornerresp {

using Fn = void (*)(const bincv::TernaryMat<uint32_t>& dx, const bincv::TernaryMat<uint32_t>& dy,
                    int blockSize, bincv::ResponseMap out, size_t* skippedWords);

void perPixel(const bincv::TernaryMat<uint32_t>& dx, const bincv::TernaryMat<uint32_t>& dy,
              int blockSize, bincv::ResponseMap out, size_t* skippedWords);
void sliced(const bincv::TernaryMat<uint32_t>& dx, const bincv::TernaryMat<uint32_t>& dy,
            int blockSize, bincv::ResponseMap out, size_t* skippedWords);
void slicedSkip(const bincv::TernaryMat<uint32_t>& dx, const bincv::TernaryMat<uint32_t>& dy,
                int blockSize, bincv::ResponseMap out, size_t* skippedWords);

} // namespace cornerresp
