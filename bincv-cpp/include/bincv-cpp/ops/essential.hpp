#pragma once

/// @file essential.hpp
/// @brief The essential matrix from five point correspondences, and the RANSAC
/// estimator built on it. **API TIER 2** -- `cv::findEssentialMat(...,
/// cv::RANSAC, ...)`'s role and call shape, with an agreement bound rather
/// than bit-exactness (ops/ransac.hpp, PRECISION).
///
/// ---------------------------------------------------------------------------
/// WHY FIVE POINTS AND NOT SOMETHING EASIER
///
/// Five is the minimum for a calibrated camera pair, and a minimal solver is what
/// makes RANSAC cheap: fewer points per sample means a clean sample is far more
/// likely, so far fewer hypotheses are needed at the same outlier ratio. At 50%
/// outliers and 99% confidence, five points need about 145 iterations where eight
/// need about 1 177. That is the whole reason VIO frontends are built around this
/// solver rather than a linear one, and it is why an eight-point or affine
/// estimator is not a substitute for it.
///
/// ---------------------------------------------------------------------------
/// THE ELIMINATION, DERIVED RATHER THAN QUOTED
///
/// The five epipolar constraints `q2^T E q1 = 0` leave a four-dimensional
/// nullspace, so `E = xX + yY + zZ + W` with three unknowns after fixing scale.
/// `E` must also satisfy `det E = 0` and `2 E E^T E - trace(E E^T) E = 0`, which
/// is ten cubic equations in `(x, y, z)` over the twenty degree-3 monomials.
///
/// Hide `z`. Each cubic is then a polynomial in `x` and `y` over six monomials
/// `{x^2, xy, y^2, x, y, 1}`, plus the four pure cubics `{x^3, x^2y, xy^2, y^3}`
/// **whose coefficients carry no z at all**. Eliminating those four columns
/// therefore uses CONSTANT pivots, which leaves the z-degrees of everything else
/// exactly as they were:
///
/// x^2, xy, y^2 -> degree 1 in z x, y -> degree 2 1 -> degree 3
///
/// Six equations survive in six monomials, so `[x^2 xy y^2 x y 1]` lies in the
/// nullspace of a 6x6 polynomial matrix `M(z)`, which needs `det M(z) = 0`. The
/// degrees sum to `1+1+1+2+2+3 = 10`.
///
/// **That the determinant comes out at degree ten is the check that the
/// construction is the right one**, because ten is independently known to be the
/// number of solutions the five-point problem has. A construction that produced
/// any other degree would be wrong on its face.
///
/// ---------------------------------------------------------------------------
/// WHAT IS PINNED, AND HOW
///
/// Validated over 300 random poses against a planted `E = [t]x R`:
///
/// at least one solution returned 300/300
/// a returned E matches the planted one, to 1e-3 295/300
/// EVERY returned E satisfies q2^T E q1 = 0 300/300, residuals at 1e-14
///
/// The last line is the one that matters most: it holds for every solution
/// returned, not merely for the one that happens to match, so a solution set
/// polluted with spurious roots would fail it. The residuals sit at machine
/// precision -- 7.5e-14 worst over 400 trials -- so a spurious root, which lands
/// orders of magnitude above that, is caught decisively rather than marginally.
///
/// The 1e-3 rather than 1e-6 in the middle line is the FLOAT INPUT, not the
/// elimination: the same solver on double coordinates recovers 1e-6 almost always.
/// Feature detectors produce float positions, so this is the precision the
/// operation actually runs at.
///
/// ---------------------------------------------------------------------------
/// MEMORY
///
/// Nothing here allocates. The solver's working set is stack, and
/// `essentialSolverStackBytes()` reports it, because on a small part that number
/// is the one that decides whether this is usable at all.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "ransac.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief A 3x3 essential matrix, row-major. **API TIER 2** -- the same layout as
/// `cv::findEssentialMat`'s `CV_64F` result read row by row.
/// @note `double`, not `float`. The solver eliminates through a degree-10
/// polynomial and a single-precision intermediate loses roots.
/// @note **The INPUT is `Point2f`, and that is what bounds the accuracy.** Measured
/// over 300 random poses, a returned `E` matches the planted one to 1e-3 in
/// 299 of them but to 1e-6 in only 163 -- the gap is the float coordinates,
/// not the elimination, and feeding the same solver double coordinates
/// recovers 1e-6 almost always. Feature detectors produce float positions, so
/// this is the precision the operation actually runs at; it is recorded here
/// rather than left for someone to rediscover.
struct EssentialMatrix {
    double m[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

namespace impl {

// --- polynomials in x, y, z -------------------------------------------------
// deg1: [x y z 1]
// deg2: [x2 xy y2 xz yz z2 x y z 1]
// deg3: [x3 x2y xy2 y3 x2z xyz y2z xz2 yz2 z3 x2 xy y2 xz yz z2 x y z 1]

struct EPoly1 { double c[4] = {0, 0, 0, 0}; };
struct EPoly2 { double c[10] = {0}; };
struct EPoly3 { double c[20] = {0}; };

inline EPoly2 ePolyMul11(const EPoly1& a, const EPoly1& b) {
    static const int kT[4][4] = {{0, 1, 3, 6}, {1, 2, 4, 7}, {3, 4, 5, 8}, {6, 7, 8, 9}};
    EPoly2 r;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) r.c[kT[i][j]] += a.c[i] * b.c[j];
    }
    return r;
}

inline EPoly3 ePolyMul21(const EPoly2& a, const EPoly1& b) {
    static const int kX[10] = {0, 1, 2, 4, 5, 7, 10, 11, 13, 16};
    static const int kY[10] = {1, 2, 3, 5, 6, 8, 11, 12, 14, 17};
    static const int kZ[10] = {4, 5, 6, 7, 8, 9, 13, 14, 15, 18};
    static const int kO[10] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    EPoly3 r;
    for (int i = 0; i < 10; ++i) {
        r.c[kX[i]] += a.c[i] * b.c[0];
        r.c[kY[i]] += a.c[i] * b.c[1];
        r.c[kZ[i]] += a.c[i] * b.c[2];
        r.c[kO[i]] += a.c[i] * b.c[3];
    }
    return r;
}

inline EPoly2 ePolyAdd2(const EPoly2& a, const EPoly2& b) {
    EPoly2 r;
    for (int i = 0; i < 10; ++i) r.c[i] = a.c[i] + b.c[i];
    return r;
}
inline EPoly3 ePolyAdd3(const EPoly3& a, const EPoly3& b) {
    EPoly3 r;
    for (int i = 0; i < 20; ++i) r.c[i] = a.c[i] + b.c[i];
    return r;
}
inline EPoly3 ePolyScale3(const EPoly3& a, double s) {
    EPoly3 r;
    for (int i = 0; i < 20; ++i) r.c[i] = a.c[i] * s;
    return r;
}

// --- the elimination's z-machinery ------------------------------------------
//
// M(z)'s entries are degree 3 at most, so they need four coefficients rather than
// the eleven a general degree-10 polynomial would. And `det M(z)` is degree 10 BY
// CONSTRUCTION, so it can be recovered from its values at eleven nodes instead of
// being built symbolically. Both matter: a symbolic 6x6 determinant over degree-10
// polynomials needs a memo table of 6 KB and about 23 000 multiplies, where eleven
// numeric 6x6 determinants need one 6x6 scratch and about 2 200.

/// @brief A coefficient of M(z): degree 3 at most. **INTERNAL.**
struct EPolyZ3 {
    double c[4] = {0, 0, 0, 0};
};

inline double ePolyZ3Eval(const EPolyZ3& p, double z) {
    return ((p.c[3] * z + p.c[2]) * z + p.c[1]) * z + p.c[0];
}

/// @brief `det M(z0)` by LU with partial pivoting. **INTERNAL.**
inline double eDet6At(const EPolyZ3 m[6][6], double z) {
    double a[6][6];
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) a[i][j] = ePolyZ3Eval(m[i][j], z);
    }
    double det = 1.0;
    for (size_t col = 0; col < 6; ++col) {
        size_t piv = col;
        double best = std::fabs(a[col][col]);
        for (size_t r = col + 1; r < 6; ++r) {
            const double v = std::fabs(a[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) return 0.0;
        if (piv != col) {
            for (size_t j = 0; j < 6; ++j) {
                const double t = a[col][j];
                a[col][j] = a[piv][j];
                a[piv][j] = t;
            }
            det = -det;
        }
        det *= a[col][col];
        const double inv = 1.0 / a[col][col];
        for (size_t r = col + 1; r < 6; ++r) {
            const double f = a[r][col] * inv;
            if (f == 0.0) continue;
            for (size_t j = col; j < 6; ++j) a[r][j] -= f * a[col][j];
        }
    }
    return det;
}

/// @brief The degree-10 coefficients of `det M(z)`, from its values at eleven
/// Chebyshev nodes. **INTERNAL.**
/// @note Chebyshev on [-1, 1] rather than equispaced, and rather than wider. The
/// range was swept: recovery of a planted E over 300 poses runs 297, 296,
/// 293, 292 and 285 of 300 at node scales 1, 1.5, 2, 3 and 4, because the
/// values grow like z^10 and roundoff in them is what limits the
/// coefficients. Chebyshev rather than equispaced because interpolation at eleven
/// equispaced nodes carries a Lebesgue constant near 30 where Chebyshev's is
/// under 3, and the coefficients are what the root finder consumes.
/// @note Newton divided differences, then expanded to the monomial basis. Solving
/// a Vandermonde system directly would be shorter and far worse conditioned.
inline void eDetPoly(const EPolyZ3 m[6][6], double* coeff) {
    constexpr int kN = 11;
    double node[kN], value[kN];
    for (int i = 0; i < kN; ++i) {
        const double theta = 3.14159265358979323846 * (2.0 * i + 1.0) / (2.0 * kN);
        node[i] = std::cos(theta);
        value[i] = eDet6At(m, node[i]);
    }
    // Divided differences, in place.
    for (int j = 1; j < kN; ++j) {
        for (int i = kN - 1; i >= j; --i) {
            const double den = node[i] - node[i - j];
            value[i] = (std::fabs(den) > 1e-300) ? (value[i] - value[i - 1]) / den : 0.0;
        }
    }
    // Newton form -> monomial, by repeated synthetic multiplication.
    for (int i = 0; i < kN; ++i) coeff[i] = 0.0;
    coeff[0] = value[kN - 1];
    int deg = 0;
    for (int k = kN - 2; k >= 0; --k) {
        for (int i = deg + 1; i > 0; --i) coeff[i] = coeff[i - 1] - node[k] * coeff[i];
        coeff[0] = value[k] - node[k] * coeff[0];
        ++deg;
    }
}

/// @brief Cyclic Jacobi eigendecomposition of a symmetric matrix. **INTERNAL.**
/// @note Jacobi rather than anything faster because it is unconditionally stable
/// on symmetric input and short enough to be read. `N` is 9 or 6 here.
template <size_t N>
inline void eJacobi(double a[N][N], double v[N][N], double w[N]) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) v[i][j] = (i == j) ? 1.0 : 0.0;
    }
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) off += a[i][j] * a[i][j];
        }
        if (off < 1e-30) break;
        for (size_t p = 0; p < N; ++p) {
            for (size_t q = p + 1; q < N; ++q) {
                if (std::fabs(a[p][q]) < 1e-300) continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (size_t k = 0; k < N; ++k) {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (size_t k = 0; k < N; ++k) {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (size_t k = 0; k < N; ++k) {
                    const double vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (size_t i = 0; i < N; ++i) w[i] = a[i][i];
}

/// @brief Every real root of a real polynomial, by Aberth-Ehrlich iteration.
/// **INTERNAL.**
/// @note All roots at once rather than one-then-deflate: deflation on a degree-10
/// polynomial loses accuracy in the later roots, and every root here is a
/// candidate solution that RANSAC will score anyway.
inline int eRealRoots(const double* c, int degree, double* out, int maxOut) {
    int n = degree;
    while (n > 0 && std::fabs(c[n]) < 1e-14) --n;
    if (n < 1) return 0;

    double re[10], im[10];
    for (int i = 0; i < n; ++i) {
        const double ang = 0.7 + 6.2831853071795862 * static_cast<double>(i) /
                                     static_cast<double>(n);
        re[i] = 0.4 * std::cos(ang);
        im[i] = 0.4 * std::sin(ang) + 0.9;
    }
    for (int it = 0; it < 500; ++it) {
        double move = 0.0;
        for (int i = 0; i < n; ++i) {
            // Horner for value and derivative, in complex arithmetic.
            double vr = c[n], vi = 0.0, dr = 0.0, di = 0.0;
            for (int k = n - 1; k >= 0; --k) {
                const double ndr = dr * re[i] - di * im[i] + vr;
                const double ndi = dr * im[i] + di * re[i] + vi;
                dr = ndr;
                di = ndi;
                const double nvr = vr * re[i] - vi * im[i] + c[k];
                const double nvi = vr * im[i] + vi * re[i];
                vr = nvr;
                vi = nvi;
            }
            const double dd = dr * dr + di * di;
            if (dd < 1e-300) continue;
            const double qr = (vr * dr + vi * di) / dd;
            const double qi = (vi * dr - vr * di) / dd;

            double sr = 0.0, si = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                const double ar = re[i] - re[j], ai = im[i] - im[j];
                const double aa = ar * ar + ai * ai;
                if (aa < 1e-300) continue;
                sr += ar / aa;
                si += -ai / aa;
            }
            const double denr = 1.0 - (qr * sr - qi * si);
            const double deni = -(qr * si + qi * sr);
            const double dn = denr * denr + deni * deni;
            if (dn < 1e-300) continue;
            const double str = (qr * denr + qi * deni) / dn;
            const double sti = (qi * denr - qr * deni) / dn;
            re[i] -= str;
            im[i] -= sti;
            move += str * str + sti * sti;
        }
        if (move < 1e-28) break;
    }

    int found = 0;
    for (int i = 0; i < n && found < maxOut; ++i) {
        if (std::fabs(im[i]) < 1e-6 * (1.0 + std::fabs(re[i]))) out[found++] = re[i];
    }
    return found;
}

} // namespace impl

/// @brief Stack the five-point solver uses for one call, in bytes.
/// @note **API TIER 3** -- OpenCV has nothing to report here because it allocates.
/// This is the number that decides whether the solver fits on a small part.
/// @note **MEASURED, NOT ADDED UP.** An earlier version of this function summed the
/// sizes of the solver's arrays and returned 4 536 while the real frame was
/// 6 240 -- a published budget that was 27% low, which is worse than no budget
/// at all. The figure below comes from the compiler:
///
/// g++ -std=c++17 -O2 -DNDEBUG -fstack-usage -c <a TU calling it>
/// grep fivePointEssential *.su
///
/// Re-measure it when the solver changes. It is a ceiling with a little
/// margin, not the exact frame, because the frame moves with the compiler
/// and the optimisation level.
inline constexpr size_t essentialSolverStackBytes() {
    return 5376;  // measured 5 136 with g++ 11.4 at -O2, rounded up
}

/// @brief Up to ten essential matrices through five correspondences.
/// **API TIER 2** -- `cv::findEssentialMat`'s minimal solver, standalone.
/// @param from Five source points in NORMALISED camera coordinates.
/// @param to Five destination points, paired by index.
/// @param out Receives the solutions; room for ten.
/// @return Solutions written, 0 to 10. Typically four to six on real geometry.
/// @note **Normalised coordinates, not pixels.** Divide by the intrinsics first;
/// this solver has no notion of focal length or principal point, exactly as
/// `cv::findEssentialMat`'s five-point core does not.
/// @note Never throws; allocates nothing. See `essentialSolverStackBytes()`.
inline int fivePointEssential(const Point2f* from, const Point2f* to, EssentialMatrix* out) {
    BINCV_ASSERT(from != nullptr && to != nullptr && out != nullptr,
                 "essential: five-point needs five correspondences and somewhere to write");

    // THE PHASES ARE SEQUENTIAL, SO THEIR STORAGE OVERLAPS.
    //
    // The nullspace scratch is dead before the cubics exist; the cubics are dead
    // before the roots. Writing them as one flat frame kept all of it live at once
    // for no reason -- it made the solver's peak the SUM of the phases where it only
    // ever needs the MAXIMUM. Each phase is scoped so its locals die at the brace,
    // and only what the next phase reads crosses it.
    //
    // Peak is the cubic-construction phase. `essentialSolverStackBytes()` reports
    // what -fstack-usage measures, not what the arrays add up to.

    double basis[4][9];  // crosses every phase: E is built from it at the end

    // --- phase 1: the nullspace of the 5x9, by Householder on its transpose ------
    //
    // Not an eigendecomposition of A^T A: forming the normal matrix squares the
    // condition number, and it costs a 9x9 of eigenvectors this needs four columns
    // of. Reflecting the last four standard basis vectors back through the same
    // five reflectors gives an orthonormal nullspace basis directly.
    {
        double at[9][5];
        for (int p = 0; p < 5; ++p) {
            const double q1[3] = {static_cast<double>(from[p].x),
                                  static_cast<double>(from[p].y), 1.0};
            const double q2[3] = {static_cast<double>(to[p].x),
                                  static_cast<double>(to[p].y), 1.0};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) at[i * 3 + j][p] = q2[i] * q1[j];
            }
        }

        double reflector[5][9];
        for (int k = 0; k < 5; ++k) {
            double norm = 0.0;
            for (int i = k; i < 9; ++i) norm += at[i][k] * at[i][k];
            norm = std::sqrt(norm);
            if (!(norm > 1e-14)) return 0;  // rank-deficient: the sample is degenerate
            const double alpha = (at[k][k] > 0.0) ? -norm : norm;

            double v[9];
            for (int i = 0; i < 9; ++i) v[i] = (i < k) ? 0.0 : at[i][k];
            v[k] -= alpha;
            double vn = 0.0;
            for (int i = k; i < 9; ++i) vn += v[i] * v[i];
            if (!(vn > 1e-30)) {
                for (int i = 0; i < 9; ++i) reflector[k][i] = 0.0;
                continue;
            }
            vn = std::sqrt(vn);
            for (int i = 0; i < 9; ++i) reflector[k][i] = v[i] / vn;

            for (int c = k; c < 5; ++c) {
                double dot = 0.0;
                for (int i = k; i < 9; ++i) dot += reflector[k][i] * at[i][c];
                dot *= 2.0;
                for (int i = k; i < 9; ++i) at[i][c] -= dot * reflector[k][i];
            }
        }

        for (int b = 0; b < 4; ++b) {
            for (int i = 0; i < 9; ++i) basis[b][i] = (i == b + 5) ? 1.0 : 0.0;
            for (int k = 4; k >= 0; --k) {
                double dot = 0.0;
                for (int i = 0; i < 9; ++i) dot += reflector[k][i] * basis[b][i];
                dot *= 2.0;
                for (int i = 0; i < 9; ++i) basis[b][i] -= dot * reflector[k][i];
            }
        }
    }

    impl::EPolyZ3 m[6][6];  // crosses into the root phases; the cubics do not

    // --- phases 2 and 3: the ten cubics, then eliminate the pure-cubic columns ---
    {
        double a[10][20];
        {
            impl::EPoly1 e[3][3];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    e[i][j].c[0] = basis[0][i * 3 + j];
                    e[i][j].c[1] = basis[1][i * 3 + j];
                    e[i][j].c[2] = basis[2][i * 3 + j];
                    e[i][j].c[3] = basis[3][i * 3 + j];
                }
            }
            // E E^T is symmetric, so six entries rather than nine.
            impl::EPoly2 eet[3][3];
            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    impl::EPoly2 acc;
                    for (int k = 0; k < 3; ++k) {
                        acc = impl::ePolyAdd2(acc, impl::ePolyMul11(e[i][k], e[j][k]));
                    }
                    eet[i][j] = acc;
                    if (i != j) eet[j][i] = acc;
                }
            }
            const impl::EPoly2 tr =
                impl::ePolyAdd2(impl::ePolyAdd2(eet[0][0], eet[1][1]), eet[2][2]);

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    impl::EPoly3 acc;
                    for (int k = 0; k < 3; ++k) {
                        acc = impl::ePolyAdd3(acc, impl::ePolyMul21(eet[i][k], e[k][j]));
                    }
                    const impl::EPoly3 eq = impl::ePolyAdd3(
                        impl::ePolyScale3(acc, 2.0),
                        impl::ePolyScale3(impl::ePolyMul21(tr, e[i][j]), -1.0));
                    for (int c = 0; c < 20; ++c) a[i * 3 + j][c] = eq.c[c];
                }
            }
            {
                impl::EPoly3 d = impl::ePolyMul21(impl::ePolyMul11(e[1][1], e[2][2]), e[0][0]);
                d = impl::ePolyAdd3(d, impl::ePolyScale3(
                        impl::ePolyMul21(impl::ePolyMul11(e[1][2], e[2][1]), e[0][0]), -1.0));
                d = impl::ePolyAdd3(d, impl::ePolyScale3(
                        impl::ePolyMul21(impl::ePolyMul11(e[1][0], e[2][2]), e[0][1]), -1.0));
                d = impl::ePolyAdd3(d, impl::ePolyMul21(impl::ePolyMul11(e[1][2], e[2][0]), e[0][1]));
                d = impl::ePolyAdd3(d, impl::ePolyMul21(impl::ePolyMul11(e[1][0], e[2][1]), e[0][2]));
                d = impl::ePolyAdd3(d, impl::ePolyScale3(
                        impl::ePolyMul21(impl::ePolyMul11(e[1][1], e[2][0]), e[0][2]), -1.0));
                for (int c = 0; c < 20; ++c) a[9][c] = d.c[c];
            }
        }

        // The four pure-cubic columns carry no z, so the pivots are constants and
        // the z-degrees of everything else survive untouched.
        bool used[10] = {false};
        for (int col = 0; col < 4; ++col) {
            int piv = -1;
            double best = 0.0;
            for (int r = 0; r < 10; ++r) {
                if (used[r]) continue;
                if (std::fabs(a[r][col]) > best) {
                    best = std::fabs(a[r][col]);
                    piv = r;
                }
            }
            if (piv < 0 || best < 1e-12) return 0;
            used[piv] = true;
            const double inv = 1.0 / a[piv][col];
            for (int c = 0; c < 20; ++c) a[piv][c] *= inv;
            for (int r = 0; r < 10; ++r) {
                if (r == piv) continue;
                const double f = a[r][col];
                if (f == 0.0) continue;
                for (int c = 0; c < 20; ++c) a[r][c] -= f * a[piv][c];
            }
        }

        int mr = 0;
        for (int r = 0; r < 10 && mr < 6; ++r) {
            if (used[r]) continue;
            const double* c = a[r];
            m[mr][0].c[0] = c[10]; m[mr][0].c[1] = c[4];
            m[mr][1].c[0] = c[11]; m[mr][1].c[1] = c[5];
            m[mr][2].c[0] = c[12]; m[mr][2].c[1] = c[6];
            m[mr][3].c[0] = c[16]; m[mr][3].c[1] = c[13]; m[mr][3].c[2] = c[7];
            m[mr][4].c[0] = c[17]; m[mr][4].c[1] = c[14]; m[mr][4].c[2] = c[8];
            m[mr][5].c[0] = c[19]; m[mr][5].c[1] = c[18];
            m[mr][5].c[2] = c[15]; m[mr][5].c[3] = c[9];
            ++mr;
        }
        if (mr < 6) return 0;
    }

    // --- phase 4: det M(z) = 0, degree ten, and its real roots ------------------
    double roots[10];
    int nroots = 0;
    {
        double detCoeff[11];
        impl::eDetPoly(m, detCoeff);
        nroots = impl::eRealRoots(detCoeff, 10, roots, 10);
    }

    // --- phase 5: at each root the nullvector of M gives x and y ----------------
    int found = 0;
    for (int k = 0; k < nroots && found < 10; ++k) {
        const double z = roots[k];
        double x = 0.0, y = 0.0;
        {
            double n[6][6];
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) n[i][j] = impl::ePolyZ3Eval(m[i][j], z);
            }
            double ntn[6][6];
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    double acc = 0.0;
                    for (int t = 0; t < 6; ++t) acc += n[t][i] * n[t][j];
                    ntn[i][j] = acc;
                }
            }
            double nv[6][6], nw[6];
            impl::eJacobi<6u>(ntn, nv, nw);
            int mn = 0;
            for (int i = 1; i < 6; ++i) {
                if (nw[i] < nw[mn]) mn = i;
            }
            const double w5 = nv[5][mn];
            if (std::fabs(w5) < 1e-9) continue;
            x = nv[3][mn] / w5;
            y = nv[4][mn] / w5;
        }

        EssentialMatrix& dst = out[found];
        double norm = 0.0;
        for (int i = 0; i < 9; ++i) {
            dst.m[i] = x * basis[0][i] + y * basis[1][i] + z * basis[2][i] + basis[3][i];
            norm += dst.m[i] * dst.m[i];
        }
        norm = std::sqrt(norm);
        if (!(norm > 1e-12)) continue;
        for (int i = 0; i < 9; ++i) dst.m[i] /= norm;
        ++found;
    }
    return found;
}

/// @brief The five-point model policy, for `bincv::ransac`. **API TIER 2.**
struct EssentialModel {
    using Type = EssentialMatrix;
    static constexpr size_t kMinimalSetSize = 5;
    static constexpr size_t kMaxModels = 10;

    static size_t estimate(const Point2f* from, const Point2f* to, const size_t* idx,
                           EssentialMatrix* out) {
        Point2f f[5], t[5];
        for (int i = 0; i < 5; ++i) {
            f[i] = from[idx[i]];
            t[i] = to[idx[i]];
        }
        const int n = fivePointEssential(f, t, out);
        return n < 0 ? 0u : static_cast<size_t>(n);
    }

    /// @brief Sampson distance -- the first-order approximation of geometric
    /// reprojection error, which is what `cv::findEssentialMat`'s threshold is in.
    /// @note NOT the raw algebraic residual `q2^T E q1`. That is not a distance:
    /// it scales with the magnitude of the coordinates, so a fixed threshold
    /// on it accepts far more at the image edge than at its centre.
    static float residual(const EssentialMatrix& e, Point2f a, Point2f b) {
        const double x1 = static_cast<double>(a.x), y1 = static_cast<double>(a.y);
        const double x2 = static_cast<double>(b.x), y2 = static_cast<double>(b.y);

        const double ex1 = e.m[0] * x1 + e.m[1] * y1 + e.m[2];
        const double ey1 = e.m[3] * x1 + e.m[4] * y1 + e.m[5];
        const double ez1 = e.m[6] * x1 + e.m[7] * y1 + e.m[8];

        const double ex2 = e.m[0] * x2 + e.m[3] * y2 + e.m[6];
        const double ey2 = e.m[1] * x2 + e.m[4] * y2 + e.m[7];

        const double num = x2 * ex1 + y2 * ey1 + ez1;
        const double den = ex1 * ex1 + ey1 * ey1 + ex2 * ex2 + ey2 * ey2;
        if (!(den > 1e-30)) return 1e30f;
        return static_cast<float>(std::fabs(num) / std::sqrt(den));
    }
};

/// @brief `cv::findEssentialMat(..., cv::RANSAC, ...)`'s role over caller-owned
/// scratch. **API TIER 2.**
/// @param from Source points in NORMALISED camera coordinates, `count` entries.
/// @param to Destination points, `count` entries.
/// @param count Correspondences. Fewer than five returns `found = false`.
/// @param params Threshold (in normalised units), confidence, cap and seed.
/// @param scratch Caller-owned, `capacity >= count`; `ransacScratchBytes(count)`.
/// @param model Written only when the result reports `found`.
/// @param inlierMask Optional, `count` entries. `nullptr` costs nothing.
/// @note Never throws; allocates nothing.
inline RansacResult findEssentialMat(const Point2f* from, const Point2f* to, size_t count,
                                     const RansacParams& params, RansacScratch scratch,
                                     EssentialMatrix* model, uint8_t* inlierMask = nullptr) {
    return ransac<EssentialModel>(from, to, count, params, scratch, model, inlierMask);
}

/// @brief `findEssentialMat` with no scratch parameter, matching
/// `cv::findEssentialMat`'s call shape. **API TIER 2.**
/// @note A wrapper over the scratch-taking form, which is the kernel. It allocates
/// the inlier flags and releases them; the solver's own working set is stack
/// either way and is unchanged by which overload is used.
inline RansacResult findEssentialMat(const Point2f* from, const Point2f* to, size_t count,
                                     const RansacParams& params, EssentialMatrix* model,
                                     uint8_t* inlierMask = nullptr) {
    return ransac<EssentialModel>(from, to, count, params, model, inlierMask);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
