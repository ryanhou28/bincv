#pragma once

/// @file essentialKnownRotation.hpp
/// @brief The essential matrix when the rotation is already known -- measured by a
/// gyroscope, say -- estimated by RANSAC over two-point samples. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHAT IT IS FOR
///
/// A frontend with an inertial sensor knows the rotation between two frames before it
/// looks at them. With the rotation known, the epipolar constraint is LINEAR in the
/// translation: for `E = [t]x R`,
///
/// q2^T E q1 = t . ((R q1) x q2)
///
/// so each correspondence says `t` is perpendicular to `n = (R q1) x q2`, and two of
/// them give `t = n1 x n2`. A hypothesis is three cross products where the five-point
/// solve (ops/essential.hpp) eliminates through a degree-10 polynomial, and it yields
/// one model where that yields up to ten. A two-point sample is also clean far more
/// often than a five-point one, so at the same confidence the stopping rule asks for
/// fewer of them: 7 against 26 at 70% inliers and 99% confidence, 17 against 146 at 50%
/// (`impl::ransacIterationsNeeded`, which is arithmetic, not a measurement).
///
/// ---------------------------------------------------------------------------
/// WHAT IT GIVES UP
///
/// **The rotation is taken as exact.** Whatever error it carries -- the gyroscope's
/// bias and noise, the camera-to-sensor calibration, the time offset between the two --
/// goes into every residual, so the threshold has to absorb it as well as the features'
/// error, and a rotation error larger than the threshold allows turns inliers into
/// outliers. Whether that trade pays depends on the caller's sensor and calibration,
/// which nothing here can see. `findEssentialMat` makes no such assumption.
///
/// **A frame that did not translate has no translation to find.** Every correspondence
/// then satisfies every candidate -- `R q1` is parallel to `q2` -- so a model comes back
/// with every point its inlier, and that count is no evidence of anything. The
/// five-point estimator cannot tell pure rotation either; a caller that needs to know
/// has to ask whether a rotation alone explains the frame.
///
/// ---------------------------------------------------------------------------
/// WHAT IS THE FIVE-POINT ESTIMATOR'S
///
/// Everything but the minimal solve and the refit is `EssentialModel`'s, inherited: the
/// Sampson residual, the threshold's units (normalised camera coordinates), the strict
/// `<` consensus rule, the deterministic sampler, and on M-profile the squared test and
/// the single-precision scorer. The model returned is an `EssentialMatrix` of unit
/// Frobenius norm, `[t]x R / sqrt(2)` for a unit `t`, as `findEssentialMat` returns
/// one, so a caller can put this where that was.
///
/// OpenCV has no known-rotation estimator, so there is nothing to agree with: API tier
/// 3, and a name of its own.
///
/// ---------------------------------------------------------------------------
/// THE CONVENTION
///
/// `rotation` is row-major and takes the `from` camera's frame to the `to` camera's: a
/// point `X` in the first camera's coordinates is `R X + t` in the second's. That is
/// the `R` of `E = [t]x R` and of `cv::recoverPose`, and the transpose of what a
/// column-major matrix library hands over as its raw data.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../core/types.hpp"
#include "essential.hpp"
#include "ransac.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief The two-point model policy under a known rotation, for `bincv::ransac`.
/// **API TIER 3.** Stateful: construct it with the rotation and hand the driver the
/// instance -- `ransac(KnownRotationEssentialModel(R), ...)`.
struct KnownRotationEssentialModel : EssentialModel {
    static constexpr size_t kMinimalSetSize = 2;
    static constexpr size_t kMaxModels = 1;

    /// @param rotation Row-major, `from` camera to `to` camera; see the file header.
    explicit KnownRotationEssentialModel(const double (&rotation)[9]) {
        for (int i = 0; i < 9; ++i) r[i] = rotation[i];
    }

    /// @brief The translation through two correspondences, as `E = [t]x R / sqrt(2)`.
    /// @return 1, or 0 when the two constraints are the same plane to within
    /// `1e-6` rad -- the sample cannot fix a direction -- including a sample with no
    /// parallax at all, whose `n` vanishes.
    size_t estimate(const Point2f* from, const Point2f* to, const size_t* idx,
                    EssentialMatrix* out) const {
        double n[2][3];
        for (size_t s = 0; s < 2; ++s) constraint(from[idx[s]], to[idx[s]], n[s]);
        double t[3];
        cross(n[0], n[1], t);
        const double tt = dot(t, t);
        // |n1 x n2|^2 = |n1|^2 |n2|^2 sin^2: the angle between the two planes, squared.
        if (!(tt > kMinSinSquared * dot(n[0], n[0]) * dot(n[1], n[1]))) return 0;
        const double inv = 1.0 / std::sqrt(tt);
        for (double& v : t) v *= inv;
        compose(t, out);
        return 1;
    }

    /// @brief Least-squares refit of the translation over the consensus set: the unit
    /// `t` minimising `sum (t . n_i)^2`, the smallest eigenvector of `sum n_i n_i^T`.
    /// **Closed form**, because the constraint is linear in `t` once `R` is fixed --
    /// unlike the five-point model, which has no linear refit and does none.
    /// @return false, leaving `model` as the minimal-set fit, when fewer than two
    /// correspondences are flagged or their constraints do not fix a direction.
    bool refine(const Point2f* from, const Point2f* to, const uint32_t* flags, size_t count,
                EssentialMatrix* model) const {
        double m[3][3] = {};
        size_t used = 0;
        for (size_t i = 0; i < count; ++i) {
            if (((flags[i >> 5] >> (i & 31u)) & 1u) == 0u) continue;
            double n[3];
            constraint(from[i], to[i], n);
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) m[a][b] += n[a] * n[b];
            ++used;
        }
        if (used < kMinimalSetSize) return false;
        double v[3][3], w[3];
        impl::eJacobi<3>(m, v, w);
        int lo = 0, hi = 0;
        for (int k = 1; k < 3; ++k) {
            if (w[k] < w[lo]) lo = k;
            if (w[k] > w[hi]) hi = k;
        }
        const int mid = 3 - lo - hi;
        // Two vanishing eigenvalues: every constraint is the same plane, and any
        // direction in it fits as well as any other.
        if (lo == hi || !(w[mid] > kMinSinSquared * w[hi])) return false;
        double t[3] = {v[0][lo], v[1][lo], v[2][lo]};
        const double tn = std::sqrt(dot(t, t));
        if (!(tn > 0.0)) return false;
        for (double& x : t) x /= tn;
        compose(t, model);
        return true;
    }

    double r[9];   ///< the rotation, row-major

private:
    /// Below this squared sine the two constraint planes are taken as one.
    static constexpr double kMinSinSquared = 1e-12;

    static double dot(const double (&a)[3], const double (&b)[3]) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    static void cross(const double (&a)[3], const double (&b)[3], double (&c)[3]) {
        c[0] = a[1] * b[2] - a[2] * b[1];
        c[1] = a[2] * b[0] - a[0] * b[2];
        c[2] = a[0] * b[1] - a[1] * b[0];
    }
    /// `n = (R q1) x q2`, the plane one correspondence confines `t` to.
    void constraint(Point2f a, Point2f b, double (&n)[3]) const {
        const double x = static_cast<double>(a.x), y = static_cast<double>(a.y);
        const double p[3] = {r[0] * x + r[1] * y + r[2], r[3] * x + r[4] * y + r[5],
                             r[6] * x + r[7] * y + r[8]};
        const double q[3] = {static_cast<double>(b.x), static_cast<double>(b.y), 1.0};
        cross(p, q, n);
    }
    /// `E = [t]x R / sqrt(2)` for a unit `t`, row by row: `|[t]x R| = |[t]x| = sqrt(2)`,
    /// so this is the unit Frobenius norm the five-point solver returns.
    void compose(const double (&unit)[3], EssentialMatrix* e) const {
        constexpr double kInvSqrt2 = 0.70710678118654752440;
        const double t[3] = {unit[0] * kInvSqrt2, unit[1] * kInvSqrt2, unit[2] * kInvSqrt2};
        for (int c = 0; c < 3; ++c) {
            e->m[c] = -t[2] * r[3 + c] + t[1] * r[6 + c];
            e->m[3 + c] = t[2] * r[c] - t[0] * r[6 + c];
            e->m[6 + c] = -t[1] * r[c] + t[0] * r[3 + c];
        }
    }
};

/// @brief The essential matrix `[t]x R` for a known rotation, by RANSAC over two-point
/// samples, over caller-owned scratch. **API TIER 3.**
/// @param from Source points in NORMALISED camera coordinates, `count` entries.
/// @param to Destination points, `count` entries.
/// @param count Correspondences. Fewer than two returns `found = false`.
/// @param rotation Row-major, `from` camera to `to` camera; see the file header.
/// @param params Threshold (in normalised units, the Sampson distance), confidence,
/// cap, seed, and whether to refit the translation over the consensus set.
/// @param scratch Caller-owned, `capacity >= count`; `ransacScratchBytes(count)`.
/// @param model Written only when the result reports `found`: `[t]x R / sqrt(2)` for a
/// unit `t`, unit Frobenius norm.
/// @param inlierMask Optional, `count` entries. `nullptr` costs nothing.
/// @note Never throws; allocates nothing.
inline RansacResult essentialFromKnownRotation(const Point2f* from, const Point2f* to,
                                               size_t count, const double (&rotation)[9],
                                               const RansacParams& params,
                                               RansacScratch scratch, EssentialMatrix* model,
                                               uint8_t* inlierMask = nullptr) {
    return ransac(KnownRotationEssentialModel(rotation), from, to, count, params, scratch,
                  model, inlierMask);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
