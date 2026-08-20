#pragma once

// T3.11 / E-10 (EXPERIMENTS.md X-23) -- the interface the THREE ARMS share.
//
// The experiment asks what replacing T3.7's frame-sized `float` response map with
// a rolling three-row ring costs in time, and what it actually saves once
// everything the streaming form must carry to preserve the selection's GLOBAL
// properties is counted against the ring. Three arms answer it, and each lives in
// its OWN translation unit:
//
//   corner_streaming_arm_frame.cpp    F, the control: the shipped
//                                     `cornerMinEigenVal` (column-major, two slid
//                                     accumulators) into a caller-owned
//                                     1 228 800 B map, then `selectGoodFeatures`.
//   corner_streaming_arm_stream2.cpp  S2, streaming TWO-pass: three-row ring; one
//                                     sweep purely to find the global maximum,
//                                     then a second that thresholds, suppresses
//                                     and ranks. NOT shipped -- this arm exists
//                                     to price the naive shape the task estimate
//                                     describes.
//   corner_streaming_arm_stream1.cpp  S1, streaming ONE-pass: the shipped
//                                     `goodFeaturesToTrackStreaming` -- one
//                                     evaluation per pixel, a running maximum, a
//                                     top-K over RAW 3x3 maxima, threshold
//                                     applied last.
//
// WHY THREE TRANSLATION UNITS RATHER THAN ONE, AND WHY IT IS IN THE INTERFACE
// HEADER RATHER THAN IN A NOTE
//
// The hazard is measured twice in this repository. benchmark/morphology_path_-
// benchmark.cpp records two instantiations of one kernel in a single object file
// moving each other's timings by ~10% through code layout alone, and X-22
// measured 1.46x for the SAME kernel between two binaries built from unchanged
// source -- with T3.10 seeing 1.46x from adding arms to a shared TU. X-21 split
// its arms across genericn_arm_*.cpp for exactly this reason; this file copies
// that shape. An A/B taken inside one TU here would be a code-layout artefact.
//
// The cost of the separation is that no arm inlines into the timing loop. That
// cost is paid IDENTICALLY by all three -- one non-inlined call per frame against
// 307 200 pixels of work inside it -- so it cannot move the comparison.
//
// EVERY ARM TAKES THE SAME VIEWS. Unlike X-21's arms there is no "no-binCV"
// control here: the question is binCV against binCV, and all three arms compute
// the same three popcount sums through the same reductions. What differs is the
// TRAVERSAL, the number of passes, and what is kept alive between rows.
//
// TWO ENTRY POINTS PER ARM, because the task asks for both numbers:
//
//   *Detect  -- the WHOLE `goodFeaturesToTrack` call: response AND selection.
//               This is what the decision rule's `T` is taken on.
//   *Respond -- the response stage alone, defined as EVERYTHING THE FORM MUST DO
//               BEFORE IT CAN START SUPPRESSING: for F that is the full map sweep
//               PLUS the `minMaxLoc` pass over the map (`selectGoodFeatures` step
//               1, which the frame-map form cannot avoid); for S1 the single row
//               sweep with its running maximum; for S2 the maximum-finding pass
//               plus the re-evaluation sweep. Defining it as "the sweep only"
//               would charge S1 for a running maximum that F pays for separately
//               and hide F's second traversal, so the two columns would not be
//               comparable. Each returns the global maximum, which the driver
//               folds into measure_util's volatile sink.

#include <cstddef>
#include <cstdint>

#include "bincv-cpp/ops/corner.hpp"
#include "bincv-cpp/quantMat.hpp"

namespace t311 {

/// @brief The four derivative planes an arm reads, as views -- the kernel
///        contract (D-5). No arm sees a container.
template <typename W>
struct Planes {
    bincv::BinMatConstView<W> magX;
    bincv::BinMatConstView<W> magY;
    bincv::BinMatConstView<W> signX;
    bincv::BinMatConstView<W> signY;
};

template <typename W>
inline Planes<W> planesOf(const bincv::TernaryMat<W>& dx, const bincv::TernaryMat<W>& dy) {
    return Planes<W>{dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(), dy.constSign()};
}

// --------------------------------------------------------------------------
// F -- the control. `scratch` is the frame-sized map.
// --------------------------------------------------------------------------
template <typename W>
float frameRespond(const Planes<W>& p, int blockSize, bincv::ResponseMap scratch);
template <typename W>
bincv::CornerResult frameDetect(const Planes<W>& p, const bincv::GoodFeaturesParams& params,
                                bincv::ResponseMap scratch, bincv::Corner* corners,
                                std::size_t capacity);

// --------------------------------------------------------------------------
// S2 -- streaming, two passes. `ring` is three rows.
// --------------------------------------------------------------------------
template <typename W>
float stream2Respond(const Planes<W>& p, int blockSize, bincv::ResponseMap ring);
template <typename W>
bincv::CornerResult stream2Detect(const Planes<W>& p, const bincv::GoodFeaturesParams& params,
                                  bincv::ResponseMap ring, bincv::Corner* corners,
                                  std::size_t capacity);

// --------------------------------------------------------------------------
// S1 -- streaming, one pass. This arm is the SHIPPED entry point.
// --------------------------------------------------------------------------
template <typename W>
float stream1Respond(const Planes<W>& p, int blockSize, bincv::ResponseMap ring);
template <typename W>
bincv::CornerResult stream1Detect(const Planes<W>& p, const bincv::GoodFeaturesParams& params,
                                  bincv::ResponseMap ring, bincv::Corner* corners,
                                  std::size_t capacity);

// The instantiations that exist. `uint32_t` is D-14's shipped default and the
// word type the decision is taken at; `uint64_t` is the other half of X-23's
// stated workload. Declared `extern` so the driver's TU emits none of this code
// and every arm's layout stays inside its own object file.
#define BINCV_T311_DECLARE_ARM(W)                                                                \
    extern template float frameRespond<W>(const Planes<W>&, int, bincv::ResponseMap);            \
    extern template bincv::CornerResult frameDetect<W>(                                          \
        const Planes<W>&, const bincv::GoodFeaturesParams&, bincv::ResponseMap, bincv::Corner*,  \
        std::size_t);                                                                            \
    extern template float stream2Respond<W>(const Planes<W>&, int, bincv::ResponseMap);          \
    extern template bincv::CornerResult stream2Detect<W>(                                        \
        const Planes<W>&, const bincv::GoodFeaturesParams&, bincv::ResponseMap, bincv::Corner*,  \
        std::size_t);                                                                            \
    extern template float stream1Respond<W>(const Planes<W>&, int, bincv::ResponseMap);          \
    extern template bincv::CornerResult stream1Detect<W>(                                        \
        const Planes<W>&, const bincv::GoodFeaturesParams&, bincv::ResponseMap, bincv::Corner*,  \
        std::size_t)

BINCV_T311_DECLARE_ARM(uint32_t);
BINCV_T311_DECLARE_ARM(uint64_t);

#undef BINCV_T311_DECLARE_ARM

}  // namespace t311
