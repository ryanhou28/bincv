#pragma once

// T3.9 / E-4 (EXPERIMENTS.md X-21) -- the interface the THREE ARMS share.
//
// The experiment asks whether binCV's generic-N machinery costs the specialized
// N = 1 / ternary paths anything against code written with no genericity at all.
// Three arms answer that, and each lives in its OWN translation unit:
//
//   genericn_arm_generic.cpp      the library's generic-N route, forced on at
//                                 N = 1 (impl::derivative*Generic, ForceGeneric
//                                 = true) and a compile-time plane loop around
//                                 the reductions.
//   genericn_arm_specialized.cpp  what ships: derivativeX/derivativeY on a
//                                 QuantMat<1> / SignedQuantMat<1>, which is
//                                 TernaryMat, and the single-plane reductions.
//   genericn_arm_handwritten.cpp  binary-only C++ with NO QuantMat, no view
//                                 struct, no template over N or over the word
//                                 type, no route selector and no argument
//                                 contract -- what someone writes if binCV does
//                                 not exist. See that file's header for exactly
//                                 what it does and does not share.
//
// WHY THREE TRANSLATION UNITS RATHER THAN ONE.
//
//   1. CODE SIZE IS HALF THE METRIC. `size` on one object per arm is a number
//      per arm; `size` on a single object holding all three is a number for the
//      sum, and D-2 / ARCHITECTURE 2 want the per-arm figure.
//   2. benchmark/morphology_path_benchmark.cpp records, measured, that two
//      instantiations of one kernel in a single object file move each other's
//      timings by ~10% through code layout alone. Separate objects keep each
//      arm's layout its own.
//
// The cost of the separation is that no arm can inline into the timing loop.
// That cost is paid IDENTICALLY by all three -- one non-inlined call per frame,
// against 307 200 pixels of work inside it -- so it cannot move the comparison.
//
// EVERY ARM TAKES RAW POINTERS, and that is not a convenience. A signature
// mentioning QuantMat would already be genericity, so the hand-written arm could
// not honestly implement it. The container arms wrap the caller's buffer in a
// NON-OWNING QuantMat / SignedQuantMat inside the call, which allocates nothing
// (quantMat.hpp's wrapping constructor) and is the Tier 2 spelling anyway.
//
// THE DESTINATION LAYOUT IS SignedQuantMat<1>'s, so one buffer serves all three
// arms: plane 0 is the magnitude, plane 1 is the sign, laid out contiguously as
//     dst[0 .. height*stride)            magnitude
//     dst[height*stride .. 2*height*stride)   sign
// The hand-written arm indexes those two halves directly; it does not learn the
// layout from a container.

#include <cstddef>
#include <cstdint>

namespace t39 {

/// @brief The word type the whole experiment runs at: D-14's shipped default.
/// @note The hand-written arm hardcodes it, because a person writing binary-only
///       code picks one word width and writes it down. That is not a handicap
///       imposed on the arm -- it is the thing the arm is a control for.
using Word = uint32_t;

/// @brief Bits per Word, spelled out rather than derived, for the same reason.
constexpr size_t kWordBits = 32;

/// @brief The four numbers of a 2x2 gradient covariance (ARCHITECTURE 7.5).
/// @note A plain aggregate rather than bincv::CovarianceCount: the hand-written
///       arm must not include a binCV header to return its result.
struct Cov {
    size_t xx = 0;
    size_t yy = 0;
    size_t whenClear = 0;
    size_t whenSet = 0;

    bool operator==(const Cov& o) const {
        return xx == o.xx && yy == o.yy && whenClear == o.whenClear && whenSet == o.whenSet;
    }
    bool operator!=(const Cov& o) const { return !(*this == o); }
};

/// @brief One arm's four entry points. Identical signatures across the arms, so
///        the driver holds three of these and treats them the same way.
/// @note A struct of function pointers, filled in by each arm's TU. It keeps the
///       driver from naming twelve symbols and, more to the point, makes the
///       three arms interchangeable at the call site so the timing loop is
///       literally the same code for each.
struct Arm {
    const char* name;

    /// dx and dy of one binary frame, into two SignedQuantMat<1>-shaped buffers.
    void (*derivative)(const Word* src, size_t strideWords, int width, int height, Word* dstX,
                       Word* dstY);

    /// Set pixels in the whole frame.
    size_t (*countWhole)(const Word* src, size_t strideWords, int width, int height);

    /// The LK covariance over one window, from the two ternary derivatives.
    Cov (*covarianceWindow)(const Word* dx, const Word* dy, size_t strideWords, int width,
                            int height, int wx, int wy, int wsize);
};

const Arm& genericArm();
const Arm& specializedArm();
const Arm& handWrittenArm();

// The decomposition points, from genericn_diag.cpp. Not arms of the rule
// comparison -- they exist only to split the gap the rule fires on into the
// kernel's generic SHAPE and the CONTAINER around it. See that file's header.
void derivativeViewsOnly(const Word* src, size_t strideWords, int width, int height, Word* dstX,
                         Word* dstY);
Cov covarianceWindowViewsOnly(const Word* dx, const Word* dy, size_t strideWords, int width,
                              int height, int wx, int wy, int wsize);

}  // namespace t39
