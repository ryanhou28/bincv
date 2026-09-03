#pragma once

/// @file ransac.hpp
/// @brief Random sample consensus with a bounded, caller-owned working set.
/// **API TIER 2** -- `cv::estimateAffine2D(..., RANSAC, ...)`'s role and call
/// shape, with an agreement bound rather than bit-exactness (see PRECISION).
///
/// ---------------------------------------------------------------------------
/// WHAT THIS IS FOR, AND WHAT IT IS NOT
///
/// A caller chooses their estimator. binCV does not have an opinion about whether
/// RANSAC, MSAC or MAGSAC is the right consensus rule for their data, and nothing
/// here argues for one. What this file offers is the same operation with binCV's
/// memory contract attached:
///
///   * **Nothing here allocates.** The sample indices, the inlier flags and the
///     candidate models all live in scratch the caller provides, sized by
///     `ransacScratchBytes()` before the call rather than discovered during it.
///   * **The working set is computable from the signature.** `cv::estimateAffine2D`
///     allocates internally, so what one call costs is not visible to its caller.
///     Here it is `ransacScratchBytes(correspondences)` and nothing else.
///   * **It needs no OpenCV.** `-DBINCV_USE_OPENCV=OFF` is a supported build, and
///     before this file a caller on that path had features and flow with no way to
///     consume them.
///
/// **THE SPEED IS NOT THE POINT AND IS NOT CLAIMED TO BE.** RANSAC's cost is
/// dominated by the minimal solver -- dense floating-point arithmetic on very small
/// matrices -- which is not work bit-packing has anything to say about. The one
/// structurally bit-parallel step, counting inliers, runs over a few hundred to a
/// few thousand flags: 25 to 250 bytes, resident in L1 either way. ops/opticalFlow's
/// window measurement already established what happens at that scale, and it is
/// nothing. `benchmark/ransac_benchmark.cpp` reports the honest figure beside the
/// footprint rather than leaving the reader to assume one.
///
/// ---------------------------------------------------------------------------
/// THE MODEL IS A POLICY, SO THE LOOP IS WRITTEN ONCE
///
/// Sampling, scoring, consensus and termination do not depend on what is being
/// estimated. A model supplies three things and the driver supplies the rest:
///
///   static constexpr size_t kMinimalSetSize;   // correspondences per hypothesis
///   static constexpr size_t kMaxModels;        // hypotheses one minimal set yields
///   static size_t estimate(const Point2f* from, const Point2f* to,
///                          const size_t* idx, Model* out);   // -> count written
///   static float residual(const Model& m, Point2f from, Point2f to);
///
/// `estimate` returns how many models it produced, because a minimal solver may
/// produce several (a 3-point pose solver yields up to four) or none (a degenerate
/// sample). The driver scores every one of them.
///
/// This is the shape the harder solvers slot into unchanged. `Affine2D` below is
/// the one this file ships; an essential-matrix or perspective-pose model is a new
/// policy and no change to the loop.
///
/// ---------------------------------------------------------------------------
/// PRECISION, AND WHY THERE IS NO TIER 1 HERE
///
/// **RANSAC is randomised, so bit-exactness against OpenCV is not available even in
/// principle** -- two implementations that sample differently see different
/// hypotheses and stop at different iterations. That is a property of the algorithm,
/// not a shortfall of this one, and it is why this is Tier 2 with a stated bound
/// rather than Tier 1 with a proof.
///
/// What IS pinned:
///
///   * **Determinism.** The same seed, correspondences and parameters give the same
///     model, bit for bit, on every run and on both architectures. A caller who
///     cannot reproduce a result cannot debug one, so the sampler is a counter-based
///     generator seeded from `RansacParams::seed` and nothing reads a global.
///   * **The consensus rule.** A correspondence is an inlier when its residual is
///     strictly below `threshold`, which is what OpenCV's `ransacReprojThreshold`
///     means, so the two agree about what a given model's support is even when they
///     disagree about which model they found.
///
/// tests/test_ransac.cpp holds both, and measures agreement with
/// `cv::estimateAffine2D` on the same correspondences.

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <new>  // std::nothrow -- the owning overloads at the bottom, and nothing else

#include "../core/error.hpp"
#include "../core/types.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief The four parameters a RANSAC call takes. **API TIER 2** -- the same
/// meanings `cv::estimateAffine2D` gives them.
struct RansacParams {
    /// @brief Inlier cutoff, in the residual's own units. A correspondence is an
    /// inlier when its residual is STRICTLY below this, matching OpenCV's
    /// `ransacReprojThreshold`.
    double threshold = 3.0;
    /// @brief Probability that the returned model is free of outliers, used by the
    /// adaptive stopping rule. OpenCV spells this `confidence`.
    double confidence = 0.99;
    /// @brief Hard cap on hypotheses, whatever the stopping rule says. OpenCV
    /// spells this `maxIters`.
    int maxIterations = 2000;
    /// @brief Seeds the sampler. **The same seed gives the same answer**, which is
    /// the whole reason this is a parameter rather than a global.
    uint64_t seed = UINT64_C(0x9E3779B97F4A7C15);
};

/// @brief What a RANSAC call reports back. **API TIER 2.**
struct RansacResult {
    /// @brief Correspondences supporting the returned model.
    size_t inliers = 0;
    /// @brief Hypotheses actually scored, which the stopping rule usually cuts well
    /// below `maxIterations`. Reported because it is the honest cost of the call.
    int iterations = 0;
    /// @brief False when no model was found -- too few correspondences, or every
    /// minimal sample degenerate. The output model is then untouched.
    bool found = false;
    /// @brief The owning overload could not obtain its flags. Nothing else sets it:
    /// every scratch-taking entry point takes the caller's buffer and cannot
    /// fail this way.
    bool allocationFailed = false;
};

/// @brief Caller-owned scratch: one flag per correspondence, twice.
/// @note **API TIER 3 as a TYPE** -- OpenCV allocates this internally and has no
/// equivalent to name. Two arrays rather than one because the driver has to
/// compare a candidate's support against the best so far without having
/// scored the best again.
struct RansacScratch {
    uint8_t* best = nullptr;       ///< inlier flags of the best model so far
    uint8_t* candidate = nullptr;  ///< inlier flags of the model being scored
    size_t capacity = 0;           ///< entries in each, >= the correspondence count

    bool empty() const { return best == nullptr || candidate == nullptr || capacity == 0; }
};

/// @brief Bytes of scratch a call over `correspondences` points needs.
/// @note **This is the operation's whole memory cost**, and it is here so a caller
/// can size a buffer before the call rather than discover it during one. Two
/// bytes per correspondence: 4 000 B at 2 000 correspondences, against an
/// OpenCV call whose internal allocation is not visible from its signature.
inline constexpr size_t ransacScratchBytes(size_t correspondences) {
    return 2 * correspondences;
}

// ---------------------------------------------------------------------------
// The sampler
// ---------------------------------------------------------------------------

namespace impl {

/// @brief Counter-based generator: state advances, output is mixed from it.
/// **INTERNAL.**
/// @note Counter-based rather than stateful-stream so that a run is reproducible
/// from the seed alone, with no dependence on how many values were drawn
/// before it.
inline uint64_t ransacMix(uint64_t x) {
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

/// @brief `k` distinct indices below `n`, written to `out`. **INTERNAL.**
/// @return false when `n` cannot supply `k` distinct indices.
/// @note Rejection against the indices already drawn. `k` is 3 to 8 for every
/// minimal solver worth having, so the quadratic scan is a handful of
/// comparisons and needs no set.
inline bool ransacSample(size_t n, size_t k, uint64_t counter, size_t* out) {
    if (n < k) return false;
    for (size_t i = 0; i < k; ++i) {
        // A fresh draw per attempt, so a collision does not bias the retry.
        for (size_t attempt = 0; attempt < 64; ++attempt) {
            const size_t pick =
                static_cast<size_t>(ransacMix(counter * UINT64_C(0x100000001B3) + attempt) % n);
            bool clash = false;
            for (size_t j = 0; j < i; ++j) {
                if (out[j] == pick) {
                    clash = true;
                    break;
                }
            }
            if (!clash) {
                out[i] = pick;
                break;
            }
            if (attempt == 63) return false;
        }
        ++counter;
    }
    return true;
}

/// @brief The adaptive stopping rule: iterations needed to see one clean sample
/// with probability `confidence`, given the support seen so far. **INTERNAL.**
/// @note This is what keeps a 2 000-iteration cap from costing 2 000 iterations.
/// With 80% inliers and a 3-point model it stops at about 7.
inline int ransacIterationsNeeded(size_t inliers, size_t total, size_t minimalSet,
                                  double confidence) {
    if (total == 0 || inliers == 0) return INT32_MAX;
    const double w = static_cast<double>(inliers) / static_cast<double>(total);
    double wk = 1.0;
    for (size_t i = 0; i < minimalSet; ++i) wk *= w;
    if (wk >= 1.0) return 1;
    const double denom = std::log(1.0 - wk);
    if (denom >= 0.0) return INT32_MAX;
    const double needed = std::log(1.0 - confidence) / denom;
    if (!(needed > 0.0)) return 1;
    if (needed >= 2147483647.0) return INT32_MAX;
    return static_cast<int>(needed) + 1;
}

} // namespace impl

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// @brief Fit `Model` to the correspondences by random sample consensus.
/// **API TIER 2** -- the role and call shape of OpenCV's RANSAC estimators, with
/// caller-owned scratch and no allocation.
/// @tparam Model The model policy; see "THE MODEL IS A POLICY" at the top of this file.
/// @param from Source points. `count` entries.
/// @param to Destination points, paired with `from` by index. `count` entries.
/// @param count Correspondences. Fewer than `Model::kMinimalSetSize` returns `found = false`.
/// @param params Threshold, confidence, iteration cap and seed.
/// @param scratch Caller-owned, `capacity >= count`. See `ransacScratchBytes`.
/// @param bestModel Written only when the result reports `found`.
/// @param inlierMask Optional, `count` entries; receives 1 for each inlier of the
/// returned model and 0 elsewhere. Pass `nullptr` if not wanted --
/// unlike `cv::estimateAffine2D`, not asking for it costs nothing.
/// @return `{inliers, iterations, found}`.
///
/// @note **Deterministic.** Same seed, same inputs, same answer, on both
/// architectures. Nothing here reads a global generator.
/// @note **No allocation and no throw**, as everywhere else in this library.
/// @note **The returned model is the minimal-set fit, not a refit over its
/// inliers.** OpenCV's estimators follow RANSAC with a least-squares
/// refinement over the consensus set, so their model will differ slightly
/// even when the two agree on which correspondences are inliers. A caller who
/// wants the refinement can do it over `inlierMask`; doing it here would mean
/// carrying a solver this file deliberately does not have.
template <typename Model>
inline RansacResult ransac(const Point2f* from, const Point2f* to, size_t count,
                           const RansacParams& params, RansacScratch scratch,
                           typename Model::Type* bestModel, uint8_t* inlierMask = nullptr) {
    BINCV_ASSERT(from != nullptr || count == 0, "ransac: a non-zero count needs source points");
    BINCV_ASSERT(to != nullptr || count == 0, "ransac: a non-zero count needs destination points");
    BINCV_ASSERT(bestModel != nullptr, "ransac: needs somewhere to put the model");
    BINCV_ASSERT(params.threshold > 0.0, "ransac: threshold must be positive");
    BINCV_ASSERT(params.confidence > 0.0 && params.confidence < 1.0,
                 "ransac: confidence must be in (0, 1)");
    BINCV_ASSERT(params.maxIterations > 0, "ransac: maxIterations must be positive");

    RansacResult out;
    if (count < Model::kMinimalSetSize) return out;

    BINCV_ASSERT(!scratch.empty(), "ransac: a non-empty problem needs scratch");
    BINCV_ASSERT(scratch.capacity >= count,
                 "ransac: scratch capacity must cover every correspondence");

    const float threshold = static_cast<float>(params.threshold);
    size_t indices[Model::kMinimalSetSize];
    typename Model::Type models[Model::kMaxModels];

    size_t bestInliers = 0;
    int limit = params.maxIterations;

    for (int iter = 0; iter < limit && iter < params.maxIterations; ++iter) {
        out.iterations = iter + 1;

        const uint64_t counter = params.seed + static_cast<uint64_t>(iter) * UINT64_C(0x9E3779B9);
        if (!impl::ransacSample(count, Model::kMinimalSetSize, counter, indices)) continue;

        const size_t produced = Model::estimate(from, to, indices, models);
        for (size_t m = 0; m < produced; ++m) {
            size_t support = 0;
            for (size_t i = 0; i < count; ++i) {
                const bool in = Model::residual(models[m], from[i], to[i]) < threshold;
                scratch.candidate[i] = in ? uint8_t{1} : uint8_t{0};
                support += in ? size_t{1} : size_t{0};
            }
            if (support <= bestInliers) continue;

            bestInliers = support;
            *bestModel = models[m];
            out.found = true;
            for (size_t i = 0; i < count; ++i) scratch.best[i] = scratch.candidate[i];

            // Tighten the cap as the support estimate improves. This is where the
            // iteration count actually comes from; maxIterations is a backstop.
            const int needed = impl::ransacIterationsNeeded(support, count,
                                                            Model::kMinimalSetSize,
                                                            params.confidence);
            if (needed < limit) limit = needed;
        }
    }

    out.inliers = bestInliers;
    if (inlierMask != nullptr) {
        for (size_t i = 0; i < count; ++i) {
            inlierMask[i] = out.found ? scratch.best[i] : uint8_t{0};
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// The model this file ships
// ---------------------------------------------------------------------------

/// @brief A 2D affine transform, row-major: `[m[0] m[1] m[2]; m[3] m[4] m[5]]`.
/// **API TIER 2** -- what `cv::estimateAffine2D` returns, in the same layout as
/// its `CV_64F` 2x3 `cv::Mat` read row by row.
struct Affine2D {
    float m[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
};

/// @brief The 3-point affine model policy. **API TIER 2.**
/// @note Three correspondences determine a 2D affine exactly, and the solve is a
/// pair of 3x3 linear systems sharing one matrix -- no eigen decomposition,
/// no iteration, one determinant. That is why this is the model this file
/// ships first: it is the one whose numerics can be checked by inspection.
struct Affine2DModel {
    using Type = Affine2D;
    static constexpr size_t kMinimalSetSize = 3;
    static constexpr size_t kMaxModels = 1;

    /// @brief Solve the affine through the three sampled correspondences.
    /// @return 1, or 0 when the three source points are collinear.
    static size_t estimate(const Point2f* from, const Point2f* to, const size_t* idx,
                           Affine2D* out) {
        const Point2f a = from[idx[0]], b = from[idx[1]], c = from[idx[2]];
        const Point2f p = to[idx[0]], q = to[idx[1]], r = to[idx[2]];

        // Twice the signed area of the source triangle. Zero means collinear, and a
        // near-zero one means a model that fits three points and nothing else -- the
        // sample is rejected rather than producing a hypothesis the scorer must waste
        // a full pass on.
        const double d = static_cast<double>(b.x - a.x) * static_cast<double>(c.y - a.y) -
                         static_cast<double>(c.x - a.x) * static_cast<double>(b.y - a.y);
        if (!(std::fabs(d) > 1e-8)) return 0;

        const double inv = 1.0 / d;
        // Cramer's rule on the shared 3x3, once per output row.
        const double x1 = static_cast<double>(a.x), y1 = static_cast<double>(a.y);
        const double x2 = static_cast<double>(b.x), y2 = static_cast<double>(b.y);
        const double x3 = static_cast<double>(c.x), y3 = static_cast<double>(c.y);

        const double u1 = static_cast<double>(p.x), v1 = static_cast<double>(p.y);
        const double u2 = static_cast<double>(q.x), v2 = static_cast<double>(q.y);
        const double u3 = static_cast<double>(r.x), v3 = static_cast<double>(r.y);

        const double au = ((u2 - u1) * (y3 - y1) - (u3 - u1) * (y2 - y1)) * inv;
        const double bu = ((x2 - x1) * (u3 - u1) - (x3 - x1) * (u2 - u1)) * inv;
        const double cu = u1 - au * x1 - bu * y1;

        const double av = ((v2 - v1) * (y3 - y1) - (v3 - v1) * (y2 - y1)) * inv;
        const double bv = ((x2 - x1) * (v3 - v1) - (x3 - x1) * (v2 - v1)) * inv;
        const double cv = v1 - av * x1 - bv * y1;

        out->m[0] = static_cast<float>(au);
        out->m[1] = static_cast<float>(bu);
        out->m[2] = static_cast<float>(cu);
        out->m[3] = static_cast<float>(av);
        out->m[4] = static_cast<float>(bv);
        out->m[5] = static_cast<float>(cv);
        return 1;
    }

    /// @brief Euclidean reprojection error, in pixels -- the units
    /// `cv::estimateAffine2D`'s `ransacReprojThreshold` is in.
    static float residual(const Affine2D& t, Point2f a, Point2f b) {
        const float px = t.m[0] * a.x + t.m[1] * a.y + t.m[2];
        const float py = t.m[3] * a.x + t.m[4] * a.y + t.m[5];
        const float dx = px - b.x;
        const float dy = py - b.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

/// @brief `cv::estimateAffine2D(..., RANSAC, ...)`'s role over caller-owned scratch.
/// **API TIER 2** -- see PRECISION at the top of this file for what agrees and
/// what cannot.
/// @param from Source points, `count` entries.
/// @param to Destination points, `count` entries.
/// @param count Correspondences.
/// @param params Threshold, confidence, iteration cap and seed.
/// @param scratch Caller-owned, `capacity >= count`; `ransacScratchBytes(count)`.
/// @param model Written only when the result reports `found`.
/// @param inlierMask Optional, `count` entries. `nullptr` costs nothing.
/// @note Never throws; allocates nothing.
inline RansacResult estimateAffine2D(const Point2f* from, const Point2f* to, size_t count,
                                     const RansacParams& params, RansacScratch scratch,
                                     Affine2D* model, uint8_t* inlierMask = nullptr) {
    return ransac<Affine2DModel>(from, to, count, params, scratch, model, inlierMask);
}

/// @brief `ransac` with no scratch parameter: it takes the flags itself.
/// **API TIER 2** -- the call shape OpenCV's estimators have.
/// @note **THIS IS THE ONLY PART OF THIS FILE THAT ALLOCATES**, and it exists so a
/// caller porting from OpenCV does not have to restructure to make one call.
/// It is a WRAPPER over the form above, not a second implementation.
/// @note **It does not reduce the memory the operation needs** -- the same two
/// flags per correspondence are live either way. What the scratch-taking form
/// buys is that the number is knowable before the call and the buffer can be
/// reused across frames; what this form buys is convenience.
/// @note A target with no heap never instantiates it: this is a template, so a
/// build that does not call it does not emit it.
/// @note Allocation failure is a RETURN VALUE (`allocationFailed`), not a throw --
/// the library builds with exceptions disabled.
template <typename Model>
inline RansacResult ransac(const Point2f* from, const Point2f* to, size_t count,
                           const RansacParams& params, typename Model::Type* bestModel,
                           uint8_t* inlierMask = nullptr) {
    RansacResult out;
    if (count == 0) return out;
    uint8_t* flags = new (std::nothrow) uint8_t[ransacScratchBytes(count)];
    if (flags == nullptr) {
        out.allocationFailed = true;
        return out;
    }
    const RansacScratch scratch{flags, flags + count, count};
    out = ransac<Model>(from, to, count, params, scratch, bestModel, inlierMask);
    delete[] flags;
    return out;
}

/// @brief `estimateAffine2D` with no scratch parameter. **API TIER 2.**
/// @note See the owning `ransac` above for what this does and does not buy.
inline RansacResult estimateAffine2D(const Point2f* from, const Point2f* to, size_t count,
                                     const RansacParams& params, Affine2D* model,
                                     uint8_t* inlierMask = nullptr) {
    return ransac<Affine2DModel>(from, to, count, params, model, inlierMask);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
