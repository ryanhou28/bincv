#pragma once

/// @file cornerSqrtMemo.cuh
/// @brief The square root inside the minimum-eigenvalue response, memoized over
/// the integer domain a blockSize-3 window confines it to.
///
/// ---------------------------------------------------------------------------
/// A .cuh, AND WHY
///
/// The table is `__constant__` and the lookup is `__device__`, so only a CUDA
/// translation unit can name either. It is a header rather than a static block
/// inside `corner.cu` because the claim that makes the memo exact is a claim
/// about EVERY entry, and a table in one `.cu`'s anonymous namespace can only
/// be reached through whatever responses a frame happens to produce -- which
/// covers 55 of the 100 cells (see the domain note below). The suite sweeps all
/// 100. `sparseMatch.cuh` and `packCustom.cuh` are the precedent.
///
/// ---------------------------------------------------------------------------
/// A MEMO OF `sqrt`, NOT A SECOND DEFINITION OF THE RESPONSE
///
/// `bincv::impl::minEigenValue` is unchanged, and every caller whose
/// discriminant is not a small integer still evaluates it: the 31x31 windows in
/// `ops/opticalFlow.hpp` and `opticalFlow.cu`, and the per-pixel window arm in
/// `corner.cu`, which takes its blockSize at runtime. What is memoized here is
/// ONE `sqrt` over the integers a 3x3 window confines it to. The response
/// formula is the same expression in the same order around it, so there is no
/// second rule for the response to drift from.
///
/// THE DOMAIN. At blockSize 3, `xx` and `yy` are popcounts of a 3x3 window of a
/// one-bit plane, so both lie in [0, 9] and `|xx - yy| <= 9`. `xy` is
/// `pos - neg`, where `pos + neg` is the popcount of the two planes' overlap in
/// the same window, so `|xy| <= 9` as well. `minEigenValue` forms
/// `disc = d*d + 4.0*c*c` from those: both products are exact integers (9^2 and
/// 4*9^2 are far under 2^53), so `disc` is exactly the integer `d^2 + 4c^2`.
/// The table is therefore indexed by `(|d|, |c|)` rather than by `disc` -- 100
/// cells of 8 bytes against 406, one cache line's worth of difference in the
/// constant bank, and the index costs the same absolute values either way.
///
/// `pos + neg <= min(xx, yy)` narrows the REACHABLE cells further, to
/// `|d| + |c| <= 9` -- 55 of the 100. The table is sized for the whole square
/// anyway, because `|d| <= 9` and `|c| <= 9` are the bounds the argument above
/// rests on and the tighter one is an accident of which plane pair a caller
/// happens to pass.
///
/// THE ENTRIES ARE EXACT, NOT APPROXIMATE. IEEE-754 requires `sqrt` to be
/// correctly rounded -- the same property that lets `minEigenValue` carry
/// BINCV_HOST_DEVICE and give one answer on two targets. So
/// `std::sqrt(d^2 + 4c^2)` on the host, `sqrt` in a kernel, and a table entry
/// holding that double are three spellings of one value. Verified rather than
/// argued: `test_cuda_shared_helpers.cu` sweeps all 100 entries against
/// `std::sqrt` and `sqrt(0 .. 405)` device against host, and the corner suite
/// holds the memoized response map to the host's byte for byte.
///
/// AND IT CANNOT BE READ BEFORE IT IS FILLED. The table is initialized where it
/// is declared, so it is part of the module image and the loader places it
/// before any kernel of this module can run. A `cudaMemcpyToSymbol` at first
/// use would be a convention instead -- an ordering that every launch path has
/// to remember, on whichever stream the caller handed it -- and a table that is
/// still zero produces plausible responses rather than an error.

#include <cuda_runtime.h>

#include "bincv/core/error.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace impl {

/// @brief One side of the memo's square index domain: `|d|` and `|c|` in [0, 9].
constexpr int kSqrtMemoSide = 10;

/// @brief `sqrt(d*d + 4*c*c)` for `|d| = row`, `|c| = column`, both in [0, 9].
/// @note Written out rather than uploaded so that the symbol is filled by the
/// module loader -- see the file header. Every value is `std::sqrt` of the
/// integer named in the comment beside its row, printed to 17 significant
/// digits, which round-trips a double exactly.
/// @note `static`, so each translation unit that includes this header gets its
/// own device symbol rather than a duplicate definition at host link. There is
/// still ONE set of values, here, which is what makes the suite's sweep of its
/// copy a statement about the kernels' copy.
static __constant__ double kSqrtMemo[kSqrtMemoSide * kSqrtMemoSide] = {
    // sqrt(d^2 + 4c^2), |c| = 0 .. 9 across each row
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18,                                             // |d| = 0
    1, 2.2360679774997898, 4.1231056256176606, 6.0827625302982193,                 // |d| = 1
    8.0622577482985491, 10.04987562112089, 12.041594578792296, 14.035668847618199,
    16.031219541881399, 18.027756377319946,
    2, 2.8284271247461903, 4.4721359549995796, 6.324555320336759,                  // |d| = 2
    8.2462112512353212, 10.198039027185569, 12.165525060596439, 14.142135623730951,
    16.124515496597098, 18.110770276274835,
    3, 3.6055512754639891, 5, 6.7082039324993694,                                  // |d| = 3
    8.5440037453175304, 10.440306508910551, 12.369316876852981, 14.317821063276353,
    16.278820596099706, 18.248287590894659,
    4, 4.4721359549995796, 5.6568542494923806, 7.2111025509279782,                 // |d| = 4
    8.9442719099991592, 10.770329614269007, 12.649110640673518, 14.560219778561036,
    16.492422502470642, 18.439088914585774,
    5, 5.3851648071345037, 6.4031242374328485, 7.810249675906654,                  // |d| = 5
    9.4339811320566032, 11.180339887498949, 13, 14.866068747318506,
    16.763054614240211, 18.681541692269406,
    6, 6.324555320336759, 7.2111025509279782, 8.4852813742385695,                  // |d| = 6
    10, 11.661903789690601, 13.416407864998739, 15.231546211727817,
    17.088007490635061, 18.973665961010276,
    7, 7.2801098892805181, 8.0622577482985491, 9.2195444572928871,                 // |d| = 7
    10.63014581273465, 12.206555615733702, 13.892443989449804, 15.652475842498529,
    17.464249196572979, 19.313207915827967,
    8, 8.2462112512353212, 8.9442719099991592, 10,                                 // |d| = 8
    11.313708498984761, 12.806248474865697, 14.422205101855956, 16.124515496597098,
    17.888543819998318, 19.697715603592208,
    9, 9.2195444572928871, 9.8488578017961039, 10.816653826391969,                 // |d| = 9
    12.041594578792296, 13.45362404707371, 15, 16.643316977093239,
    18.357559750685819, 20.124611797498108};

/// @brief `impl::minEigenValue(xx, yy, xy)` with the square root looked up.
/// @note THE CALLER OWNS THE DOMAIN. Valid only where `|xx - yy| <= 9` and
/// `|xy| <= 9`, which is what a 3x3 window of one-bit planes gives and what the
/// bit-sliced response arm passes. Anything wider must call `minEigenValue`,
/// and does.
/// @note The arithmetic around the root is `minEigenValue`'s, unchanged:
/// `s` is formed from the same integer sum and the result is rounded to float
/// from the same double expression, so the two agree bit for bit rather than
/// closely.
__device__ inline float minEigenValueMemo3(long long xx, long long yy, long long xy) {
    const long long d = xx - yy;
    const long long ad = d < 0 ? -d : d;
    const long long ac = xy < 0 ? -xy : xy;
    BINCV_ASSERT(ad < kSqrtMemoSide && ac < kSqrtMemoSide,
                 "cuda minEigenValueMemo3: the discriminant is outside the blockSize-3 domain");
    const double s = static_cast<double>(xx + yy);
    return static_cast<float>(0.5 * (s - kSqrtMemo[ad * kSqrtMemoSide + ac]));
}

} // namespace impl
} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
