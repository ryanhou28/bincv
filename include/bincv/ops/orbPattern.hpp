#pragma once

/// @file orbPattern.hpp
/// @brief cv::ORB's learned 256-pair sampling table, vendored with its notice.
/// **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHY A TABLE IS VENDORED WHEN THE ALGORITHM IS FREE
///
/// ORB (Rublee et al., ICCV 2011) is free to reimplement, and ops/descriptor.hpp
/// and ops/orientation.hpp do. But BRIEF-family descriptors are only COMPARABLE
/// when both sides sample with the IDENTICAL pair table -- bit i must mean the
/// same comparison on both -- and re-running the paper's learning procedure
/// yields a different table than the one every deployed ORB pipeline actually
/// uses. Matching against a cv::ORB-family map therefore requires OpenCV's own
/// `bit_pattern_31_`, copied verbatim, not a table like it.
///
/// The pairs sit within [-13, 12] per axis -- norms up to sqrt(338) ~ 18.4, the
/// value the tests pin -- so `makeSteeredBriefPattern` rotates this table with a
/// reach of at most 19: tighter than the default square-sampled pattern's ~21,
/// and comfortably inside the int8 offsets the steered builder requires.
///
/// The comparison predicate and the bit packing match OpenCV's descriptor loop
/// (bit set when img[a] < img[b], eight LSB-first bits per byte), so descriptors
/// from this table at angle 0 over the SAME blurred pixels are cv::ORB's bytes
/// exactly -- tests/test_descriptor.cpp pins that byte for byte against
/// cv::ORB::compute. What cv::ORB adds around the sampling -- its 7x7 Gaussian
/// blur, per-keypoint exact rotation rather than 30 bins, its pyramid -- is the
/// caller's to reproduce where full interchange matters; the table is the part
/// that cannot be reproduced from the paper.
///
/// ---------------------------------------------------------------------------
/// PROVENANCE AND LICENSE
///
/// The 1024 values below are `bit_pattern_31_` from OpenCV's
/// modules/features2d/src/orb.cpp, which carries the BSD 3-clause license
/// (Copyright (c) 2009, Willow Garage, Inc.) -- NOT OpenCV's newer Apache-2.0;
/// the file-level notice governs, and an earlier comment in ops/descriptor.hpp
/// mis-stated it. Retaining that notice is the license's condition on source
/// redistribution, so it is reproduced here, where the copied material lives,
/// and in THIRD_PARTY_NOTICES.md:
///
///   Software License Agreement (BSD License)
///   Copyright (c) 2009, Willow Garage, Inc.  All rights reserved.
///
///   Redistribution and use in source and binary forms, with or without
///   modification, are permitted provided that the following conditions are met:
///     * Redistributions of source code must retain the above copyright notice,
///       this list of conditions and the following disclaimer.
///     * Redistributions in binary form must reproduce the above copyright
///       notice, this list of conditions and the following disclaimer in the
///       documentation and/or other materials provided with the distribution.
///     * Neither the name of the Willow Garage nor the names of its contributors
///       may be used to endorse or promote products derived from this software
///       without specific prior written permission.
///
///   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
///   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
///   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
///   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
///   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
///   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
///   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
///   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
///   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
///   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
///   POSSIBILITY OF SUCH DAMAGE.

#include "descriptor.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief `cv::ORB`'s learned sampling table as a `BriefPattern<256>`: pair i is
/// OpenCV's points (x1, y1) -> a and (x2, y2) -> b, in OpenCV's order.
/// **API TIER 3** -- binCV's name for OpenCV's data; see the file header for
/// what "compatible" does and does not promise.
inline constexpr BriefPattern<256> kOrbBriefPattern = {{
    {8, -3, 9, 5}, {4, 2, 7, -12}, {-11, 9, -8, 2}, {7, -12, 12, -13},
    {2, -13, 2, 12}, {1, -7, 1, 6}, {-2, -10, -2, -4}, {-13, -13, -11, -8},
    {-13, -3, -12, -9}, {10, 4, 11, 9}, {-13, -8, -8, -9}, {-11, 7, -9, 12},
    {7, 7, 12, 6}, {-4, -5, -3, 0}, {-13, 2, -12, -3}, {-9, 0, -7, 5},
    {12, -6, 12, -1}, {-3, 6, -2, 12}, {-6, -13, -4, -8}, {11, -13, 12, -8},
    {4, 7, 5, 1}, {5, -3, 10, -3}, {3, -7, 6, 12}, {-8, -7, -6, -2},
    {-2, 11, -1, -10}, {-13, 12, -8, 10}, {-7, 3, -5, -3}, {-4, 2, -3, 7},
    {-10, -12, -6, 11}, {5, -12, 6, -7}, {5, -6, 7, -1}, {1, 0, 4, -5},
    {9, 11, 11, -13}, {4, 7, 4, 12}, {2, -1, 4, 4}, {-4, -12, -2, 7},
    {-8, -5, -7, -10}, {4, 11, 9, 12}, {0, -8, 1, -13}, {-13, -2, -8, 2},
    {-3, -2, -2, 3}, {-6, 9, -4, -9}, {8, 12, 10, 7}, {0, 9, 1, 3},
    {7, -5, 11, -10}, {-13, -6, -11, 0}, {10, 7, 12, 1}, {-6, -3, -6, 12},
    {10, -9, 12, -4}, {-13, 8, -8, -12}, {-13, 0, -8, -4}, {3, 3, 7, 8},
    {5, 7, 10, -7}, {-1, 7, 1, -12}, {3, -10, 5, 6}, {2, -4, 3, -10},
    {-13, 0, -13, 5}, {-13, -7, -12, 12}, {-13, 3, -11, 8}, {-7, 12, -4, 7},
    {6, -10, 12, 8}, {-9, -1, -7, -6}, {-2, -5, 0, 12}, {-12, 5, -7, 5},
    {3, -10, 8, -13}, {-7, -7, -4, 5}, {-3, -2, -1, -7}, {2, 9, 5, -11},
    {-11, -13, -5, -13}, {-1, 6, 0, -1}, {5, -3, 5, 2}, {-4, -13, -4, 12},
    {-9, -6, -9, 6}, {-12, -10, -8, -4}, {10, 2, 12, -3}, {7, 12, 12, 12},
    {-7, -13, -6, 5}, {-4, 9, -3, 4}, {7, -1, 12, 2}, {-7, 6, -5, 1},
    {-13, 11, -12, 5}, {-3, 7, -2, -6}, {7, -8, 12, -7}, {-13, -7, -11, -12},
    {1, -3, 12, 12}, {2, -6, 3, 0}, {-4, 3, -2, -13}, {-1, -13, 1, 9},
    {7, 1, 8, -6}, {1, -1, 3, 12}, {9, 1, 12, 6}, {-1, -9, -1, 3},
    {-13, -13, -10, 5}, {7, 7, 10, 12}, {12, -5, 12, 9}, {6, 3, 7, 11},
    {5, -13, 6, 10}, {2, -12, 2, 3}, {3, 8, 4, -6}, {2, 6, 12, -13},
    {9, -12, 10, 3}, {-8, 4, -7, 9}, {-11, 12, -4, -6}, {1, 12, 2, -8},
    {6, -9, 7, -4}, {2, 3, 3, -2}, {6, 3, 11, 0}, {3, -3, 8, -8},
    {7, 8, 9, 3}, {-11, -5, -6, -4}, {-10, 11, -5, 10}, {-5, -8, -3, 12},
    {-10, 5, -9, 0}, {8, -1, 12, -6}, {4, -6, 6, -11}, {-10, 12, -8, 7},
    {4, -2, 6, 7}, {-2, 0, -2, 12}, {-5, -8, -5, 2}, {7, -6, 10, 12},
    {-9, -13, -8, -8}, {-5, -13, -5, -2}, {8, -8, 9, -13}, {-9, -11, -9, 0},
    {1, -8, 1, -2}, {7, -4, 9, 1}, {-2, 1, -1, -4}, {11, -6, 12, -11},
    {-12, -9, -6, 4}, {3, 7, 7, 12}, {5, 5, 10, 8}, {0, -4, 2, 8},
    {-9, 12, -5, -13}, {0, 7, 2, 12}, {-1, 2, 1, 7}, {5, 11, 7, -9},
    {3, 5, 6, -8}, {-13, -4, -8, 9}, {-5, 9, -3, -3}, {-4, -7, -3, -12},
    {6, 5, 8, 0}, {-7, 6, -6, 12}, {-13, 6, -5, -2}, {1, -10, 3, 10},
    {4, 1, 8, -4}, {-2, -2, 2, -13}, {2, -12, 12, 12}, {-2, -13, 0, -6},
    {4, 1, 9, 3}, {-6, -10, -3, -5}, {-3, -13, -1, 1}, {7, 5, 12, -11},
    {4, -2, 5, -7}, {-13, 9, -9, -5}, {7, 1, 8, 6}, {7, -8, 7, 6},
    {-7, -4, -7, 1}, {-8, 11, -7, -8}, {-13, 6, -12, -8}, {2, 4, 3, 9},
    {10, -5, 12, 3}, {-6, -5, -6, 7}, {8, -3, 9, -8}, {2, -12, 2, 8},
    {-11, -2, -10, 3}, {-12, -13, -7, -9}, {-11, 0, -10, -5}, {5, -3, 11, 8},
    {-2, -13, -1, 12}, {-1, -8, 0, 9}, {-13, -11, -12, -5}, {-10, -2, -10, 11},
    {-3, 9, -2, -13}, {2, -3, 3, 2}, {-9, -13, -4, 0}, {-4, 6, -3, -10},
    {-4, 12, -2, -7}, {-6, -11, -4, 9}, {6, -3, 6, 11}, {-13, 11, -5, 5},
    {11, 11, 12, 6}, {7, -5, 12, -2}, {-1, 12, 0, 7}, {-4, -8, -3, -2},
    {-7, 1, -6, 7}, {-13, -12, -8, -13}, {-7, -2, -6, -8}, {-8, 5, -6, -9},
    {-5, -1, -4, 5}, {-13, 7, -8, 10}, {1, 5, 5, -13}, {1, 0, 10, -13},
    {9, 12, 10, -1}, {5, -8, 10, -9}, {-1, 11, 1, -13}, {-9, -3, -6, 2},
    {-1, -10, 1, 12}, {-13, 1, -8, -10}, {8, -11, 10, -6}, {2, -13, 3, -6},
    {7, -13, 12, -9}, {-10, -10, -5, -7}, {-10, -8, -8, -13}, {4, -6, 8, 5},
    {3, 12, 8, -13}, {-4, 2, -3, -3}, {5, -13, 10, -12}, {4, -13, 5, -1},
    {-9, 9, -4, 3}, {0, 3, 3, -9}, {-12, 1, -6, 1}, {3, 2, 4, -8},
    {-10, -10, -10, 9}, {8, -13, 12, 12}, {-8, -12, -6, -5}, {2, 2, 3, 7},
    {10, 6, 11, -8}, {6, 8, 8, -12}, {-7, 10, -6, 5}, {-3, -9, -3, 9},
    {-1, -13, -1, 5}, {-3, -7, -3, 4}, {-8, -2, -8, 3}, {4, 2, 12, 12},
    {2, -5, 3, 11}, {6, -9, 11, -13}, {3, -1, 7, 12}, {11, -1, 12, 4},
    {-3, 0, -3, 6}, {4, -11, 4, 12}, {2, -4, 2, 1}, {-10, -6, -8, 1},
    {-13, 7, -11, 1}, {-13, 12, -11, -13}, {6, 0, 11, -13}, {0, -1, 1, 4},
    {-13, 3, -9, -2}, {-9, 8, -6, -3}, {-13, -6, -8, -2}, {5, -9, 8, 10},
    {2, 7, 3, -9}, {-1, -6, -1, -1}, {9, 5, 11, -2}, {11, -3, 12, -8},
    {3, 0, 3, 5}, {-1, 4, 0, 10}, {3, -6, 4, 5}, {-13, 0, -10, 5},
    {5, 8, 12, 11}, {8, 9, 9, -6}, {7, -4, 8, -12}, {-10, 4, -10, 9},
    {7, 3, 12, 4}, {9, -7, 10, -2}, {7, 0, 12, -2}, {-1, -6, 0, -11}
}};

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
