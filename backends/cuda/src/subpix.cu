// The device arm of ops/subpix.hpp. One kernel, two window arms, one switch.
//
// The kernel is the host loop transcribed -- deliberately, and the shape is the
// bit-exactness requirement rather than an oversight. ONE THREAD PER CORNER: the
// five accumulators are doubles, double addition is not associative, and any
// warp- or block-level reduction sums them in an order the host did not use.
// That changes their last bits, which changes `q`, which can change the
// converged position and can flip a corner between refined, clamped and
// diverged. A serial per-thread walk in the host's row-major, LSB-first order is
// the only shape that agrees by construction.
//
// TWO WINDOW ARMS, held to one answer. The default visits SET BITS: `nz = |dx| |
// |dy|` a word at a time, trimmed to the window's columns, peeled low-first with
// __ffs. The reference arm visits every pixel of the window and looks its bits
// up. They are bit-identical rather than nearly so, and the reason is worth
// stating: the pixels the skip removes contribute exactly zero to all five
// accumulators, every accumulator starts at +0.0 and can only reach -0.0 by
// adding -0.0 to -0.0, so adding the skipped terms is an exact no-op.
//
// THIS FILE IS COMPILED -fmad=false (backends/cuda/CMakeLists.txt). `bx += xx*px
// + xy*py` has two multiplies and an add, `xx*px` rounds, and nvcc's default
// -fmad=true would contract it where the host cannot. `det = gxx*gyy - gxy*gxy`
// is the same shape. `gxx += w*gx*gx` is immune because `gx` is in {-1, 0, +1}
// and the product is exact -- so the flag is covering three expressions, not
// waved at the file.

#include "bincv/cuda/subpix.hpp"

#include <cstdint>

#include "bincv/impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief The spread launch's block, in threads. One warp.
constexpr unsigned kSpreadThreads = 32u;

namespace {

struct SubPixViews {
    DeviceBinMatConstView magX;
    DeviceBinMatConstView magY;
    DeviceBinMatConstView signX;
    DeviceBinMatConstView signY;
};

/// @brief The five weighted sums over one window, accumulated in the host's
/// exact order: rows top to bottom, columns left to right.
/// @tparam Skip true for the set-bit arm, false for the dense reference arm.
template <bool Skip>
__device__ void accumulate(const SubPixViews& v, const double* mask, int winHalf, long long ix0,
                           long long iy0, double& gxx, double& gxy, double& gyy, double& bx,
                           double& by) {
    const int side = 2 * winHalf + 1;
    const size_t xLo = static_cast<size_t>(ix0 - winHalf);
    const size_t xHi = static_cast<size_t>(ix0 + winHalf);  // inclusive

    for (int wy = -winHalf; wy <= winHalf; ++wy) {
        const size_t y = static_cast<size_t>(iy0 + wy);
        const double* mrow = mask + static_cast<size_t>(wy + winHalf) * static_cast<size_t>(side);
        const uint32_t* mx = v.magX.row(y);
        const uint32_t* my = v.magY.row(y);
        const uint32_t* sx = v.signX.row(y);
        const uint32_t* sy = v.signY.row(y);

        for (size_t wi = xLo / 32; wi <= xHi / 32; ++wi) {
            // Which pixels of this word are inside the window's columns. `wi`
            // may span either end.
            const size_t lo = (wi * 32 > xLo) ? 0 : (xLo - wi * 32);
            const size_t hiExcl = (xHi >= (wi + 1) * 32) ? 32 : (xHi - wi * 32 + 1);

            if (Skip) {
                // A pixel with zero gradient in both axes adds nothing to any of
                // the five sums, and this finds all of them a word at a time.
                uint32_t nz = mx[wi] | my[wi];
                if (lo != 0) nz &= ~bincv::impl::lowBitsMask<uint32_t>(lo);
                if (hiExcl < 32) nz &= bincv::impl::lowBitsMask<uint32_t>(hiExcl);
                while (nz != 0u) {
                    const unsigned b = static_cast<unsigned>(__ffs(static_cast<int>(nz)) - 1);
                    nz &= nz - 1u;
                    const size_t x = wi * 32 + b;
                    const int wx = static_cast<int>(static_cast<long long>(x) - ix0);
                    const double w = mrow[static_cast<size_t>(wx + winHalf)];
                    if (w == 0.0) continue;  // the zero zone
                    const uint32_t bit = uint32_t{1} << b;
                    double vx = (mx[wi] & bit) ? 1.0 : 0.0;
                    double vy = (my[wi] & bit) ? 1.0 : 0.0;
                    if (sx[wi] & bit) vx = -vx;
                    if (sy[wi] & bit) vy = -vy;
                    const double xx = w * vx * vx;
                    const double xy = w * vx * vy;
                    const double yy = w * vy * vy;
                    const double px = static_cast<double>(wx);
                    const double py = static_cast<double>(wy);
                    gxx += xx;
                    gxy += xy;
                    gyy += yy;
                    bx += xx * px + xy * py;
                    by += xy * px + yy * py;
                }
            } else {
                for (size_t b = lo; b < hiExcl; ++b) {
                    const size_t x = wi * 32 + b;
                    const int wx = static_cast<int>(static_cast<long long>(x) - ix0);
                    const double w = mrow[static_cast<size_t>(wx + winHalf)];
                    if (w == 0.0) continue;
                    const uint32_t bit = uint32_t{1} << b;
                    double vx = (mx[wi] & bit) ? 1.0 : 0.0;
                    double vy = (my[wi] & bit) ? 1.0 : 0.0;
                    if (sx[wi] & bit) vx = -vx;
                    if (sy[wi] & bit) vy = -vy;
                    const double xx = w * vx * vx;
                    const double xy = w * vx * vy;
                    const double yy = w * vy * vy;
                    const double px = static_cast<double>(wx);
                    const double py = static_cast<double>(wy);
                    gxx += xx;
                    gxy += xy;
                    gyy += yy;
                    bx += xx * px + xy * py;
                    by += xy * px + yy * py;
                }
            }
        }
    }
}

template <bool Skip>
__global__ void subPixKernel(SubPixViews v, float* cornersXY, uint32_t count,
                             const double* mask, int winHalf, int maxIterations, double eps2,
                             DeviceSubPixResult* result) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= count) return;

    const long long width = static_cast<long long>(v.magX.width);
    const long long height = static_cast<long long>(v.magX.height);

    const double startX = static_cast<double>(cornersXY[2 * static_cast<size_t>(c)]);
    const double startY = static_cast<double>(cornersXY[2 * static_cast<size_t>(c) + 1]);
    double cx = startX;
    double cy = startY;

    for (int it = 0; it < maxIterations; ++it) {
        // The window is anchored on the ROUNDED position: a bit-plane derivative
        // cannot be interpolated, so the samples are integer pixels and the
        // refinement is the offset within them. The host says the same.
        const long long ix0 = static_cast<long long>(floor(cx + 0.5));
        const long long iy0 = static_cast<long long>(floor(cy + 0.5));
        if (ix0 - winHalf < 0 || iy0 - winHalf < 0 || ix0 + winHalf >= width ||
            iy0 + winHalf >= height) {
            atomicAdd(&result->clamped, 1u);
            break;
        }

        double gxx = 0.0, gxy = 0.0, gyy = 0.0, bx = 0.0, by = 0.0;
        accumulate<Skip>(v, mask, winHalf, ix0, iy0, gxx, gxy, gyy, bx, by);

        const double det = gxx * gyy - gxy * gxy;
        if (det == 0.0) {
            atomicAdd(&result->singular, 1u);
            break;
        }
        const double qx = (gyy * bx - gxy * by) / det;
        const double qy = (gxx * by - gxy * bx) / det;
        // Outside the window is a divergence, not an answer -- OpenCV's rule.
        if (fabs(qx) > static_cast<double>(winHalf) || fabs(qy) > static_cast<double>(winHalf)) {
            atomicAdd(&result->clamped, 1u);
            break;
        }

        const double nx = static_cast<double>(ix0) + qx;
        const double ny = static_cast<double>(iy0) + qy;
        const double stepX = nx - cx;
        const double stepY = ny - cy;
        cx = nx;
        cy = ny;
        cornersXY[2 * static_cast<size_t>(c)] = static_cast<float>(cx);
        cornersXY[2 * static_cast<size_t>(c) + 1] = static_cast<float>(cy);
        if (stepX * stepX + stepY * stepY <= eps2) {
            atomicAdd(&result->refined, 1u);
            break;
        }
        if (it + 1 == maxIterations) atomicAdd(&result->refined, 1u);
    }

    // POOR CONVERGENCE -- cv::cornerSubPix's rule, applied after the loop exactly
    // as it applies it. A point that has walked further than its own half-window
    // from the seed has been captured by different structure, and the seed is the
    // better answer.
    if (fabs(cx - startX) > static_cast<double>(winHalf) ||
        fabs(cy - startY) > static_cast<double>(winHalf)) {
        cornersXY[2 * static_cast<size_t>(c)] = static_cast<float>(startX);
        cornersXY[2 * static_cast<size_t>(c) + 1] = static_cast<float>(startY);
        atomicAdd(&result->diverged, 1u);
    }
}

bool viewsAgree(const SubPixViews& v) {
    return v.magX.width == v.magY.width && v.magX.height == v.magY.height &&
           v.magX.width == v.signX.width && v.magX.height == v.signX.height &&
           v.magX.width == v.signY.width && v.magX.height == v.signY.height;
}

} // namespace

namespace impl {

bool& subPixSkipEnabled() {
    static bool on = true;
    return on;
}

bool& subPixSpreadEnabled() {
    static bool on = true;
    return on;
}

bool subPixSpreadApplies(uint32_t count) { return count > kSpreadThreads; }

} // namespace impl

DeviceSubPixMask::DeviceSubPixMask(const SubPixParams& params) {
    if (params.winHalf < 1 || params.winHalf > bincv::impl::kMaxWinHalf) {
        BINCV_THROW(std::invalid_argument,
                    "DeviceSubPixMask: winHalf outside [1, kMaxWinHalf]");
    }
    winHalf_ = params.winHalf;
    zeroHalf_ = params.zeroHalf;
    const size_t side = static_cast<size_t>(2 * winHalf_ + 1);
    // THE HOST'S OWN MASK BUILDER. CUDA's exp(double) is documented at 1-2 ulp
    // where glibc's is about 0.5, so a device-built mask is not required to be
    // the same weights -- and one ulp of weight lands in the accumulators.
    double host[(2 * bincv::impl::kMaxWinHalf + 1) * (2 * bincv::impl::kMaxWinHalf + 1)];
    bincv::impl::subPixMask(winHalf_, zeroHalf_, host);
    mask_ = DeviceArray<double>(side * side);
    BINCV_CUDA_CHECK(cudaMemcpy(mask_.data(), host, side * side * sizeof(double),
                                cudaMemcpyHostToDevice));
}

cudaError_t cornerSubPixAsync(DeviceBinMatConstView magX, DeviceBinMatConstView magY,
                              DeviceBinMatConstView signX, DeviceBinMatConstView signY,
                              float* cornersXY, uint32_t count, const SubPixParams& params,
                              const double* mask, DeviceSubPixResult* result,
                              cudaStream_t stream) {
    const SubPixViews v{magX, magY, signX, signY};
    BINCV_ASSERT(viewsAgree(v),
                 "cuda cornerSubPix: the four derivative planes must have the same dimensions");
    BINCV_ASSERT(params.winHalf >= 1 && params.winHalf <= bincv::impl::kMaxWinHalf,
                 "cuda cornerSubPix: winHalf outside [1, kMaxWinHalf]");
    BINCV_ASSERT(count == 0u || cornersXY != nullptr,
                 "cuda cornerSubPix: a non-zero count needs a corner array");
    BINCV_ASSERT(count == 0u || mask != nullptr,
                 "cuda cornerSubPix: a non-zero count needs an uploaded mask");
    BINCV_ASSERT(result != nullptr, "cuda cornerSubPix: a result pointer is required");
    // R4: the narrowed domain is REFUSED, not truncated, in release too.
    if (!viewsAgree(v)) return cudaErrorInvalidValue;
    if (params.winHalf < 1 || params.winHalf > bincv::impl::kMaxWinHalf) {
        return cudaErrorInvalidValue;
    }
    if (result == nullptr) return cudaErrorInvalidValue;
    if (count == 0u) return cudaSuccess;
    if (cornersXY == nullptr || mask == nullptr) return cudaErrorInvalidValue;
    if (magX.width == 0 || magX.height == 0) return cudaSuccess;

    // ONE CORNER PER THREAD is the bit-exactness requirement, so the only lever
    // over the launch is how many SMs the corners land on -- and on this part
    // that lever is the whole operation. A block of 256 puts a 200-corner
    // refinement on ONE SM, and sm_86 runs FP64 at 1/64 of FP32: two FP64 pipes
    // for the entire call. A block of one WARP puts the same 200 corners on
    // seven SMs and seven times the FP64 issue. It is not smaller than a warp,
    // because the pipe's cost is per warp-instruction rather than per active
    // lane -- a block of 8 would engage more SMs and issue a quarter of the work
    // per instruction, which is the trade running the wrong way.
    const unsigned threads = impl::subPixSpreadEnabled() ? kSpreadThreads : 256u;
    const dim3 grid((count + threads - 1u) / threads);
    const double eps2 = params.epsilon * params.epsilon;
    if (impl::subPixSkipEnabled()) {
        subPixKernel<true><<<grid, threads, 0, stream>>>(
            v, cornersXY, count, mask, params.winHalf, params.maxIterations, eps2, result);
    } else {
        subPixKernel<false><<<grid, threads, 0, stream>>>(
            v, cornersXY, count, mask, params.winHalf, params.maxIterations, eps2, result);
    }
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
