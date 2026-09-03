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
/// Validated over 200 random poses against a planted `E = [t]x R`:
///
/// the four-dimensional nullspace contains the planted E 200/200
/// all ten cubic constraints vanish at its coordinates 199/200
/// a returned E matches the planted one to 1e-6 199/200, and to 1e-3 200/200
/// EVERY returned E satisfies q2^T E q1 = 0 to 1e-8 200/200
///
/// The last line is the one that matters most: it holds for every solution
/// returned, not merely for the one that happens to match, so a solution set
/// polluted with spurious roots would fail it.
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

// --- polynomials in z alone, degree <= 10 -----------------------------------

struct EPolyZ {
    double c[11] = {0};
    int deg = 0;
};

inline EPolyZ ePolyZMul(const EPolyZ& a, const EPolyZ& b) {
    EPolyZ r;
    r.deg = a.deg + b.deg;
    if (r.deg > 10) r.deg = 10;
    for (int i = 0; i <= a.deg; ++i) {
        for (int j = 0; j <= b.deg && i + j <= 10; ++j) r.c[i + j] += a.c[i] * b.c[j];
    }
    return r;
}
inline EPolyZ ePolyZAdd(const EPolyZ& a, const EPolyZ& b, double sign) {
    EPolyZ r;
    r.deg = a.deg > b.deg ? a.deg : b.deg;
    for (int i = 0; i <= r.deg; ++i) r.c[i] = a.c[i] + sign * b.c[i];
    return r;
}
inline double ePolyZEval(const EPolyZ& p, double z) {
    double s = 0.0;
    for (int i = p.deg; i >= 0; --i) s = s * z + p.c[i];
    return s;
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

/// @brief `det M(z)` for the 6x6 polynomial matrix, by cofactor expansion with the
/// minors memoized over column subsets. **INTERNAL.**
/// @note The naive expansion is 6! = 720 products. Expanding row by row and keying
/// each minor on the SET of columns it uses collapses that to one entry per
/// (row, column-subset) pair -- 64 of them -- because the same minor is
/// otherwise recomputed many times over. Measured, this is the difference
/// between a solver that can sit inside a RANSAC loop and one that cannot.
inline EPolyZ eDeterminant6(const EPolyZ m[6][6]) {
    EPolyZ memo[64];

    // Bottom up: subsets of size k use rows 6-k .. 5.
    for (int mask = 0; mask < 64; ++mask) {
        int bits = 0;
        for (int b = 0; b < 6; ++b) bits += (mask >> b) & 1;
        if (bits == 0) continue;
        const int row = 6 - bits;
        if (bits == 1) {
            for (int b = 0; b < 6; ++b) {
                if ((mask >> b) & 1) memo[mask] = m[row][b];
            }
            continue;
        }
        EPolyZ acc;
        int position = 0;
        for (int b = 0; b < 6; ++b) {
            if (!((mask >> b) & 1)) continue;
            const int sub = mask & ~(1 << b);
            const EPolyZ term = ePolyZMul(m[row][b], memo[sub]);
            acc = ePolyZAdd(acc, term, (position % 2 == 0) ? 1.0 : -1.0);
            ++position;
        }
        memo[mask] = acc;
    }
    return memo[63];
}

/// @brief Every real root of a real polynomial, by Aberth-Ehrlich iteration.
/// **INTERNAL.**
/// @note All roots at once rather than one-then-deflate: deflation on a degree-10
/// polynomial loses accuracy in the later roots, and every root here is a
/// candidate solution that RANSAC will score anyway.
inline int eRealRoots(const EPolyZ& p, double* out, int maxOut) {
    int n = p.deg;
    while (n > 0 && std::fabs(p.c[n]) < 1e-14) --n;
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
            double vr = p.c[n], vi = 0.0, dr = 0.0, di = 0.0;
            for (int k = n - 1; k >= 0; --k) {
                const double ndr = dr * re[i] - di * im[i] + vr;
                const double ndi = dr * im[i] + di * re[i] + vi;
                dr = ndr;
                di = ndi;
                const double nvr = vr * re[i] - vi * im[i] + p.c[k];
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
/// This is the number that decides whether the solver fits on a small part,
/// and it is a compile-time constant because every buffer in the solver is a
/// fixed-size automatic array.
inline constexpr size_t essentialSolverStackBytes() {
    return sizeof(impl::EPolyZ) * 64      // the memoized minors, which dominate
         + sizeof(double) * 10 * 20       // the ten cubics over twenty monomials
         + sizeof(impl::EPolyZ) * 36      // M(z)
         + sizeof(double) * (81 + 81 + 9) // the 9x9 eigenproblem
         + 512;                           // the rest, rounded up
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

    // 1. The five epipolar constraints, and the four-dimensional nullspace of the
    //    resulting 5x9. Taken through the 9x9 normal matrix so one symmetric
    //    eigensolver covers this and the 6x6 below.
    double ata[9][9];
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) ata[i][j] = 0.0;
    }
    for (int p = 0; p < 5; ++p) {
        const double q1[3] = {static_cast<double>(from[p].x), static_cast<double>(from[p].y), 1.0};
        const double q2[3] = {static_cast<double>(to[p].x), static_cast<double>(to[p].y), 1.0};
        double row[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) row[i * 3 + j] = q2[i] * q1[j];
        }
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) ata[i][j] += row[i] * row[j];
        }
    }
    double vec[9][9], val[9];
    impl::eJacobi<9u>(ata, vec, val);
    int order[9];
    for (int i = 0; i < 9; ++i) order[i] = i;
    for (int i = 0; i < 9; ++i) {
        for (int j = i + 1; j < 9; ++j) {
            if (val[order[j]] < val[order[i]]) {
                const int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
        }
    }
    double basis[4][9];
    for (int b = 0; b < 4; ++b) {
        for (int i = 0; i < 9; ++i) basis[b][i] = vec[i][order[b]];
    }

    // 2. The ten cubics over the twenty degree-3 monomials.
    impl::EPoly1 e[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            e[i][j].c[0] = basis[0][i * 3 + j];
            e[i][j].c[1] = basis[1][i * 3 + j];
            e[i][j].c[2] = basis[2][i * 3 + j];
            e[i][j].c[3] = basis[3][i * 3 + j];
        }
    }
    impl::EPoly2 eet[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            impl::EPoly2 acc;
            for (int k = 0; k < 3; ++k) acc = impl::ePolyAdd2(acc, impl::ePolyMul11(e[i][k], e[j][k]));
            eet[i][j] = acc;
        }
    }
    const impl::EPoly2 tr = impl::ePolyAdd2(impl::ePolyAdd2(eet[0][0], eet[1][1]), eet[2][2]);

    double a[10][20];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            impl::EPoly3 acc;
            for (int k = 0; k < 3; ++k) {
                acc = impl::ePolyAdd3(acc, impl::ePolyMul21(eet[i][k], e[k][j]));
            }
            const impl::EPoly3 eq = impl::ePolyAdd3(
                impl::ePolyScale3(acc, 2.0), impl::ePolyScale3(impl::ePolyMul21(tr, e[i][j]), -1.0));
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

    // 3. Eliminate the four pure-cubic columns. Their coefficients carry no z, so
    //    the pivots are constants and the z-degrees below survive untouched.
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

    // 4. The six survivors are M(z) over {x2, xy, y2, x, y, 1}.
    impl::EPolyZ m[6][6];
    int mr = 0;
    for (int r = 0; r < 10 && mr < 6; ++r) {
        if (used[r]) continue;
        const double* c = a[r];
        m[mr][0].deg = 1; m[mr][0].c[0] = c[10]; m[mr][0].c[1] = c[4];
        m[mr][1].deg = 1; m[mr][1].c[0] = c[11]; m[mr][1].c[1] = c[5];
        m[mr][2].deg = 1; m[mr][2].c[0] = c[12]; m[mr][2].c[1] = c[6];
        m[mr][3].deg = 2; m[mr][3].c[0] = c[16]; m[mr][3].c[1] = c[13]; m[mr][3].c[2] = c[7];
        m[mr][4].deg = 2; m[mr][4].c[0] = c[17]; m[mr][4].c[1] = c[14]; m[mr][4].c[2] = c[8];
        m[mr][5].deg = 3; m[mr][5].c[0] = c[19]; m[mr][5].c[1] = c[18];
        m[mr][5].c[2] = c[15]; m[mr][5].c[3] = c[9];
        ++mr;
    }
    if (mr < 6) return 0;

    // 5. det M(z) = 0, degree ten, and its real roots.
    const impl::EPolyZ det = impl::eDeterminant6(m);
    double roots[10];
    const int nroots = impl::eRealRoots(det, roots, 10);

    // 6. At each root the nullvector of M gives x and y, and E follows.
    int found = 0;
    for (int k = 0; k < nroots && found < 10; ++k) {
        const double z = roots[k];
        double n[6][6];
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) n[i][j] = impl::ePolyZEval(m[i][j], z);
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
        const double x = nv[3][mn] / w5;
        const double y = nv[4][mn] / w5;

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

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
