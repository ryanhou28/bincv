// The device arm of ops/subpix.hpp. One kernel, two window arms, one switch.
//
// The kernel is the host loop transcribed -- deliberately, and the shape is the
// bit-exactness requirement rather than an oversight. ONE THREAD PER CORNER: the
// five accumulators are doubles, double addition is not associative, and any
// warp- or block-level reduction sums them in an order the host did not use.
// That changes their last bits, which changes `q`, which can change the
// converged position and can flip a corner between refined, singular, outside
// and diverged. A serial per-thread walk in the host's row-major, column-ascending
// order is the only shape that agrees by construction.
//
// THE WINDOW IS SAMPLED AT THE SUB-PIXEL POSITION, as the host samples it: each
// sample blends the four integer pixels around it, row by row, with the weights of
// the position's fractional part, and the rows a window reads are staged once per
// integer cell as 32-bit spans carrying cv::getRectSubPix's replicated border. The
// host looks each row's half of the blend up in a table; this computes it. The two
// are the same bits, because every product in the blend is a weight times -1, 0 or
// +1, which is exact.
//
// TWO WINDOW ARMS, held to one answer, and ONE CODE PATH. The default visits
// CANDIDATES: samples one of whose four neighbours has a gradient, found a word at
// a time and peeled low-first with __ffs. The reference arm ORs the whole window
// into that candidate set, so it visits every sample through the same loop -- the
// switch changes which samples run and nothing about how, which is what lets its
// gate-excluded control read 1.00x. The arms are bit-identical rather than nearly
// so, and the reason is worth stating: the samples the skip removes contribute
// exactly zero to all five accumulators, every accumulator starts at +0.0 and can
// only reach -0.0 by adding -0.0 to -0.0, so adding the skipped terms is an exact
// no-op.
//
// THIS FILE IS COMPILED -fmad=false (backends/cuda/CMakeLists.txt). The window's
// gradient is a blend, not -1, 0 or +1, so `gxx += gx*gx*m` rounds its product
// before the add on the host and would not under nvcc's default contraction;
// `bx += xx*px + xy*py`, `det = gxx*gyy - gxy*gxy` and the position update are the
// same shape.

#include "bincv/cuda/subpix.hpp"

#include <cfloat>
#include <cstdint>

#include "bincv/impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace {

/// @brief The spread launch's block, in threads. One warp.
constexpr unsigned kSpreadThreads = 32u;

/// @brief The most rows, and columns, a window reads: `2 * kMaxWinHalf + 2`.
constexpr int kSpanRows = 2 * bincv::impl::kMaxWinHalf + 2;

struct SubPixViews {
    DeviceBinMatConstView magX;
    DeviceBinMatConstView magY;
    DeviceBinMatConstView signX;
    DeviceBinMatConstView signY;
};

/// @brief Bits `[x, x + len)` of a plane row, `len <= 32`, read from the image's
/// columns only: a column outside them reads 0, or -- `replicate` -- the edge
/// column's bit. The host's `impl::subPixSpan`, over device words.
__device__ uint32_t span(const uint32_t* row, long long x, unsigned len, long long width,
                         bool replicate) {
    const long long lo = x < 0 ? 0 : x;
    const long long hi = (x + len < width) ? x + len : width;  // exclusive
    uint64_t acc = 0;
    if (lo < hi) {
        size_t wi = static_cast<size_t>(lo) / 32u;
        const unsigned off = static_cast<unsigned>(static_cast<size_t>(lo) % 32u);
        const unsigned n = static_cast<unsigned>(hi - lo);
        uint64_t v = static_cast<uint64_t>(row[wi]) >> off;
        unsigned got = 32u - off;
        while (got < n) {
            v |= static_cast<uint64_t>(row[++wi]) << got;
            got += 32u;
        }
        acc = (v & ((uint64_t{1} << n) - 1u)) << (lo - x);
    }
    if (replicate) {
        if (x < 0 && (row[0] & 1u)) acc |= (uint64_t{1} << (-x)) - 1u;
        if (x + len > width) {
            const size_t last = static_cast<size_t>(width - 1);
            if ((row[last / 32u] >> (last % 32u)) & 1u) {
                acc |= ((uint64_t{1} << len) - 1u) & ~((uint64_t{1} << (width - x)) - 1u);
            }
        }
    }
    return static_cast<uint32_t>(acc);
}

/// @brief The rows one window reads, staged per integer cell: `dx` reads zero left
/// and right of the image and the edge rows above and below it, `dy` the transpose.
struct StagedRows {
    uint32_t magX[kSpanRows], signX[kSpanRows], magY[kSpanRows], signY[kSpanRows];
    long long x0, y0;
    bool valid;
};

__device__ void stage(const SubPixViews& v, int winHalf, long long x0, long long y0,
                      long long width, long long height, StagedRows& r) {
    if (r.valid && r.x0 == x0 && r.y0 == y0) return;
    const unsigned len = static_cast<unsigned>(2 * winHalf + 2);
    const long long xLo = x0 - winHalf;
    for (int i = 0; i < 2 * winHalf + 2; ++i) {
        const long long y = y0 - winHalf + i;
        const size_t yc = static_cast<size_t>(y < 0 ? 0 : (y >= height ? height - 1 : y));
        r.magX[i] = span(v.magX.row(yc), xLo, len, width, false);
        r.signX[i] = span(v.signX.row(yc), xLo, len, width, false);
        if (y < 0 || y >= height) {
            r.magY[i] = 0u;
            r.signY[i] = 0u;
        } else {
            r.magY[i] = span(v.magY.row(static_cast<size_t>(y)), xLo, len, width, true);
            r.signY[i] = span(v.signY.row(static_cast<size_t>(y)), xLo, len, width, true);
        }
    }
    r.x0 = x0;
    r.y0 = y0;
    r.valid = true;
}

/// @brief A staged pixel's value; a zero magnitude reads zero whatever its sign.
__device__ double value(uint32_t mag, uint32_t sign, unsigned j) {
    return ((mag >> j) & 1u) ? (((sign >> j) & 1u) ? -1.0 : 1.0) : 0.0;
}

struct Sums {
    double gxx, gxy, gyy, bx, by;
};

/// @brief The five weighted sums over one window, accumulated in the host's exact
/// order: rows top to bottom, columns left to right.
/// @param fill 0 for the candidate arm; the whole window for the dense reference arm,
/// which visits every sample through the same loop.
__device__ Sums accumulate(const StagedRows& r, const float* mask, int winHalf, bool integer,
                           uint32_t fill, double w00, double w10, double w01, double w11) {
    Sums s{0.0, 0.0, 0.0, 0.0, 0.0};
    const int side = 2 * winHalf + 1;
    const uint32_t inWindow = static_cast<uint32_t>((uint64_t{1} << side) - 1u);
    for (int i = 0; i < side; ++i) {
        const double py = static_cast<double>(i - winHalf);
        uint32_t cand;
        if (integer) {
            cand = r.magX[i] | r.magY[i];
        } else {
            const uint32_t m = r.magX[i] | r.magY[i] | r.magX[i + 1] | r.magY[i + 1];
            cand = m | (m >> 1);
        }
        cand = (cand | fill) & inWindow;
        while (cand != 0u) {
            const unsigned j = static_cast<unsigned>(__ffs(static_cast<int>(cand)) - 1);
            cand &= cand - 1u;
            double gx, gy;
            if (integer) {
                gx = value(r.magX[i], r.signX[i], j);
                gy = value(r.magY[i], r.signY[i], j);
            } else {
                gx = (w00 * value(r.magX[i], r.signX[i], j) +
                      w10 * value(r.magX[i], r.signX[i], j + 1)) +
                     (w01 * value(r.magX[i + 1], r.signX[i + 1], j) +
                      w11 * value(r.magX[i + 1], r.signX[i + 1], j + 1));
                gy = (w00 * value(r.magY[i], r.signY[i], j) +
                      w10 * value(r.magY[i], r.signY[i], j + 1)) +
                     (w01 * value(r.magY[i + 1], r.signY[i + 1], j) +
                      w11 * value(r.magY[i + 1], r.signY[i + 1], j + 1));
            }
            const double m = static_cast<double>(mask[i * side + static_cast<int>(j)]);
            const double px = static_cast<double>(static_cast<int>(j) - winHalf);
            const double xx = gx * gx * m;
            const double xy = gx * gy * m;
            const double yy = gy * gy * m;
            s.gxx += xx;
            s.gxy += xy;
            s.gyy += yy;
            s.bx += xx * px + xy * py;
            s.by += xy * px + yy * py;
        }
    }
    return s;
}

__global__ void subPixKernel(SubPixViews v, float* cornersXY, uint32_t count, const float* mask,
                             int winHalf, int maxIterations, double eps2, uint32_t fill,
                             DeviceSubPixResult* result) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= count) return;

    const long long width = static_cast<long long>(v.magX.width);
    const long long height = static_cast<long long>(v.magX.height);
    const float widthF = static_cast<float>(width);
    const float heightF = static_cast<float>(height);
    const float startX = cornersXY[2 * static_cast<size_t>(c)];
    const float startY = cornersXY[2 * static_cast<size_t>(c) + 1];
    if (!(startX >= 0.0f && startX < widthF && startY >= 0.0f && startY < heightF)) {
        atomicAdd(&result->outside, 1u);
        return;
    }

    StagedRows rows;
    rows.valid = false;
    float cx = startX;
    float cy = startY;
    uint32_t* outcome = &result->refined;
    int iter = 0;
    double err = 0.0;
    do {
        const float fx = floorf(cx);
        const float fy = floorf(cy);
        const long long x0 = static_cast<long long>(fx);
        const long long y0 = static_cast<long long>(fy);
        const double a = static_cast<double>(cx - fx);
        const double b = static_cast<double>(cy - fy);
        stage(v, winHalf, x0, y0, width, height, rows);
        const Sums s = accumulate(rows, mask, winHalf, a == 0.0 && b == 0.0, fill,
                                  (1.0 - a) * (1.0 - b), a * (1.0 - b), (1.0 - a) * b, a * b);

        const double det = s.gxx * s.gyy - s.gxy * s.gxy;
        if (fabs(det) <= DBL_EPSILON * DBL_EPSILON) {
            outcome = &result->singular;
            break;
        }
        const double scale = 1.0 / det;
        const float nx = static_cast<float>(static_cast<double>(cx) + s.gyy * scale * s.bx -
                                            s.gxy * scale * s.by);
        const float ny = static_cast<float>(static_cast<double>(cy) - s.gxy * scale * s.bx +
                                            s.gxx * scale * s.by);
        const float ex = nx - cx;
        const float ey = ny - cy;
        err = static_cast<double>(ex * ex + ey * ey);
        cx = nx;
        cy = ny;
        if (cx < 0.0f || cx >= widthF || cy < 0.0f || cy >= heightF) {
            outcome = &result->outside;
            break;
        }
    } while (++iter < maxIterations && err > eps2);

    // POOR CONVERGENCE -- cv::cornerSubPix's rule, applied after the loop exactly
    // as it applies it.
    const float limit = static_cast<float>(winHalf);
    if (fabsf(cx - startX) > limit || fabsf(cy - startY) > limit) {
        cx = startX;
        cy = startY;
        outcome = &result->diverged;
    }
    cornersXY[2 * static_cast<size_t>(c)] = cx;
    cornersXY[2 * static_cast<size_t>(c) + 1] = cy;
    atomicAdd(outcome, 1u);
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
    // THE HOST'S OWN MASK BUILDER. CUDA's expf is not required to round as the
    // host's C library does, so a device-built mask is not required to be the same
    // weights -- and one ulp of weight lands in the accumulators.
    float host[(2 * bincv::impl::kMaxWinHalf + 1) * (2 * bincv::impl::kMaxWinHalf + 1)];
    bincv::impl::subPixMask(winHalf_, zeroHalf_, host);
    mask_ = DeviceArray<float>(side * side);
    BINCV_CUDA_CHECK(cudaMemcpy(mask_.data(), host, side * side * sizeof(float),
                                cudaMemcpyHostToDevice));
}

cudaError_t cornerSubPixAsync(DeviceBinMatConstView magX, DeviceBinMatConstView magY,
                              DeviceBinMatConstView signX, DeviceBinMatConstView signY,
                              float* cornersXY, uint32_t count, const SubPixParams& params,
                              const float* mask, DeviceSubPixResult* result,
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
    // cv::cornerSubPix's own normalizations, as the host makes them.
    const int maxIterations =
        params.maxIterations < 1
            ? 1
            : (params.maxIterations > bincv::impl::kSubPixMaxIterations
                   ? bincv::impl::kSubPixMaxIterations
                   : params.maxIterations);
    const double eps = params.epsilon > 0.0 ? params.epsilon : 0.0;
    const double eps2 = eps * eps;
    const uint32_t fill = impl::subPixSkipEnabled() ? 0u : ~0u;
    subPixKernel<<<grid, threads, 0, stream>>>(v, cornersXY, count, mask, params.winHalf,
                                               maxIterations, eps2, fill, result);
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
