// The device arm of ops/edge.hpp. Two kernels, one switch.
//
// REFERENCE ARM: one lane, one pixel, `__ballot_sync` IS the packed word -- the
// format's 32-bit granule and the warp width coinciding, as in pack.cu. Border
// pixels are handled by reflect-101 in index arithmetic, so the whole image goes
// through one body and there is no second path to keep correct. This arm is the
// oracle, and the vector arm below is held to its map bit for bit.
//
// BYTE-LANE ARM: one lane, FOUR pixels. This is the half of the host's AVX2
// structure that a per-lane device kernel throws away. The host header says it
// in as many words -- "`subs_epu8` ... the 8-bit intermediate this operation
// exists to avoid never appears even inside the kernel" -- and `__ballot_sync`
// alone only replaces the `movemask_epi8` half of that. The device's byte-lane
// instructions are the other half, and on sm_86 the ones this arm is built on
// are SINGLE instructions, measured on this machine:
//
//     __vabsdiffu4  -> VABSDIFF4.U8   1   four |a-b| at once
//     __byte_perm   -> PRMT           1   the shifted neighbour quad
//     __dp4a        -> IDP.4A         1   four byte flags folded into a nibble
//
// THEY ARE NOT ALL SINGLE, and the ones that are not decide how this kernel is
// written. Counted out of cuobjdump -sass on this part, same probe:
//
//     __vminu4 + __vmaxu4  14 together   emulated -- which is why the absolute
//                                        difference is __vabsdiffu4 and not the
//                                        min/max pair it would obviously be
//     __vsetgeu4            6            emulated: LOP3 x3, SHF x2, IADD3
//     __vcmpgeu4            5            emulated
//
// So the four byte comparisons are the expensive part of this arm, not the
// arithmetic they compare -- 12 of the roughly 18 instructions a lane spends on
// its four pixels. Three spellings of "four (d >= tp), combined and folded to a
// nibble" land within one instruction of each other (vsetgeu4 pair 14,
// vcmpgeu4 pair 13, vmaxu4-then-vsetgeu4 13), so the choice among them is not
// where this arm is won or lost.
//
// The arm also cuts the warp's load instructions fourfold -- 128 bytes per load
// instruction against 32 -- which is the second mechanism, and the two are not
// separable by any measurement this machine can take with no profiler.

#include "bincv/cuda/edge.hpp"

#include <cstdint>

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// @brief One axis' absolute difference at (y, x), with the host's border rule.
/// @note `bincv::impl::reflect101Edge` is the HOST's function, called. It is
/// BINCV_HOST_DEVICE, and every offset here is in {-1, 0, +1}, so its fold
/// runs at most once -- there is no device twin to keep in step.
template <EdgeSpatial S, bool Horizontal, typename SrcT>
__device__ inline unsigned axisDiffDevice(const DeviceImageConstView<SrcT>& src, size_t y,
                                          size_t x) {
    long long aI, bI;
    const long long here = static_cast<long long>(Horizontal ? x : y);
    if constexpr (S == EdgeSpatial::Wide) {
        aI = here + 1;
        bI = here - 1;
    } else if constexpr (S == EdgeSpatial::Forward) {
        aI = here + 1;
        bI = here;
    } else {
        aI = here;
        bI = here - 1;
    }
    const size_t n = Horizontal ? src.width : src.height;
    const size_t a = bincv::impl::reflect101Edge(aI, n);
    const size_t b = bincv::impl::reflect101Edge(bI, n);
    const unsigned va = static_cast<unsigned>(Horizontal ? src.row(y)[a] : src.row(a)[x]);
    const unsigned vb = static_cast<unsigned>(Horizontal ? src.row(y)[b] : src.row(b)[x]);
    return va >= vb ? (va - vb) : (vb - va);
}

/// @brief Is (y, x) an edge? The whole predicate, in one place, used by BOTH
/// arms -- the vector arm calls this for the quad at the right frame edge, so
/// there is one spelling of the arithmetic and not two.
/// @param tp The threshold ALREADY folded for the relation: `t` for `Ge`,
/// `t + 1` for `Gt`, so the comparison has one spelling.
/// @note Both axes are computed unconditionally where the host short-circuits.
/// Inside a warp the host's saving is illusory -- the lanes diverge and the
/// warp pays both paths anyway, plus the branch. Same output; the suite
/// holds the two bodies to it.
template <EdgeCombine C, EdgeSpatial S, typename SrcT>
__device__ inline bool edgePixel(const DeviceImageConstView<SrcT>& src, size_t y, size_t x,
                                 unsigned tp) {
    const bool ph = axisDiffDevice<S, true, SrcT>(src, y, x) >= tp;
    const bool pv = axisDiffDevice<S, false, SrcT>(src, y, x) >= tp;
    // ONE return: `C` is a template parameter, so the ternary folds away, and
    // nvcc's frontend does not warn about a missing return the way it does for
    // an `if constexpr` chain whose every branch returns.
    return (C == EdgeCombine::Or) ? (ph || pv) : (ph && pv);
}

// ---------------------------------------------------------------------------
// The reference arm
// ---------------------------------------------------------------------------

template <typename SrcT, EdgeCombine C, EdgeSpatial S>
__global__ void edgeKernelRef(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                              unsigned tp, size_t words) {
    const unsigned lane = threadIdx.x;  // 0..31, the pixel within the word
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = words * src.height;
    for (size_t wordIdx = blockIdx.x * warpsPerBlock + threadIdx.y; wordIdx < total;
         wordIdx += gridDim.x * warpsPerBlock) {
        const size_t y = wordIdx / words;
        const size_t i = wordIdx - y * words;
        const size_t x = i * 32 + lane;
        // A lane past `width` contributes 0, which is the padding invariant
        // holding by construction rather than by masking.
        const bool pred = (x < src.width) && edgePixel<C, S, SrcT>(src, y, x, tp);
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, pred);
        if (lane == 0) dst.row(y)[i] = word;
    }
}

// ---------------------------------------------------------------------------
// The byte-lane arm
// ---------------------------------------------------------------------------

/// @brief Pixels one warp of the byte-lane arm covers: 32 lanes x 4 = 4 words.
constexpr unsigned kEdgeVecPixelsPerWarp = 128;

template <EdgeCombine C>
__global__ void edgeKernelVec(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                              unsigned tp, size_t words, size_t groups) {
    const unsigned lane = threadIdx.x;
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = groups * src.height;
    // tp <= 255 by this arm's gate: a byte-lane comparison cannot express 256.
    const uint32_t tpv = tp * 0x01010101u;
    const size_t rowW32 = (src.width + 3) / 4;

    for (size_t unit = blockIdx.x * warpsPerBlock + threadIdx.y; unit < total;
         unit += gridDim.x * warpsPerBlock) {
        const size_t y = unit / groups;
        const size_t g = unit - y * groups;
        const size_t x0 = g * kEdgeVecPixelsPerWarp + static_cast<size_t>(lane) * 4;
        const size_t w32 = x0 / 4;

        // The vertical neighbours' rows, reflected. `y` is uniform across the
        // warp, so these are too: reflect-101 costs nothing here and buys the
        // whole absence of a border code path.
        const size_t yUp =
            bincv::impl::reflect101Edge(static_cast<long long>(y) - 1, src.height);
        const size_t yDn =
            bincv::impl::reflect101Edge(static_cast<long long>(y) + 1, src.height);

        const uint32_t* r32 = reinterpret_cast<const uint32_t*>(src.row(y));
        // The shuffles must be reached by EVERY lane, so the load that feeds
        // them is guarded and they are not.
        const uint32_t mid = (w32 < rowW32) ? r32[w32] : 0u;
        uint32_t prev = __shfl_up_sync(0xFFFFFFFFu, mid, 1);
        uint32_t next = __shfl_down_sync(0xFFFFFFFFu, mid, 1);

        unsigned nib = 0;
        if (x0 + 4 < src.width) {
            // Every pixel of this quad, and both of its horizontal neighbours,
            // is inside the frame. Lane 0 and lane 31 have no shuffle partner
            // for their out-of-word neighbour and load it: two extra loads per
            // warp per row, against the 64 a per-lane arm would issue.
            if (lane == 0) prev = (w32 > 0) ? r32[w32 - 1] : mid;
            if (lane == 31) next = r32[w32 + 1];

            // left4  = v[x0-1], v[x0], v[x0+1], v[x0+2]
            // right4 = v[x0+1], v[x0+2], v[x0+3], v[x0+4]
            uint32_t left4 = __byte_perm(prev, mid, 0x6543u);
            const uint32_t right4 = __byte_perm(mid, next, 0x4321u);
            // Pixel 0 is the only one in the frame whose left neighbour is
            // outside it: reflect-101 sends -1 to 1.
            if (x0 == 0) left4 = __byte_perm(left4, mid, 0x3215u);

            const uint32_t* r32up = reinterpret_cast<const uint32_t*>(src.row(yUp));
            const uint32_t* r32dn = reinterpret_cast<const uint32_t*>(src.row(yDn));
            const uint32_t dh = __vabsdiffu4(right4, left4);
            const uint32_t dv = __vabsdiffu4(r32dn[w32], r32up[w32]);

            // Bytes of 0 or 1, four comparisons per instruction group.
            const uint32_t sh = __vsetgeu4(dh, tpv);
            const uint32_t sv = __vsetgeu4(dv, tpv);
            const uint32_t s = (C == EdgeCombine::Or) ? (sh | sv) : (sh & sv);
            // Four byte flags into the low four bits, LSB = the lowest x:
            // 1*s.b0 + 2*s.b1 + 4*s.b2 + 8*s.b3, one IDP.4A.
            nib = static_cast<unsigned>(__dp4a(s, 0x08040201u, 0u));
        } else {
            // The quad holding the last pixel, and every quad past it. The SAME
            // predicate the reference arm runs -- the same function, not a
            // second spelling of it -- so the right frame edge cannot drift.
            for (unsigned k = 0; k < 4; ++k) {
                const size_t x = x0 + k;
                if (x < src.width &&
                    edgePixel<C, EdgeSpatial::Wide, uint8_t>(src, y, x, tp)) {
                    nib |= (1u << k);
                }
            }
        }

        // Eight lanes' nibbles are one output word. A three-step butterfly OR
        // leaves it in every lane of the group; lane 0 of the group stores it.
        uint32_t val = nib << (4u * (lane & 7u));
        val |= __shfl_xor_sync(0xFFFFFFFFu, val, 1);
        val |= __shfl_xor_sync(0xFFFFFFFFu, val, 2);
        val |= __shfl_xor_sync(0xFFFFFFFFu, val, 4);
        const size_t wordIdx = g * 4 + (lane >> 3);
        if ((lane & 7u) == 0 && wordIdx < words) dst.row(y)[wordIdx] = val;
    }
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

dim3 warpGrid(size_t units, unsigned warpsPerBlock) {
    const size_t warps = (units + warpsPerBlock - 1) / warpsPerBlock;
    return dim3(static_cast<unsigned>(warps < 4096 ? (warps ? warps : 1) : 4096));
}

template <typename SrcT, EdgeCombine C>
cudaError_t launchRefSpatial(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                             unsigned tp, EdgeSpatial spatial, size_t words,
                             cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid = warpGrid(words * src.height, block.y);
    switch (spatial) {
        case EdgeSpatial::Wide:
            edgeKernelRef<SrcT, C, EdgeSpatial::Wide>
                <<<grid, block, 0, stream>>>(src, dst, tp, words);
            break;
        case EdgeSpatial::Forward:
            edgeKernelRef<SrcT, C, EdgeSpatial::Forward>
                <<<grid, block, 0, stream>>>(src, dst, tp, words);
            break;
        case EdgeSpatial::Backward:
            edgeKernelRef<SrcT, C, EdgeSpatial::Backward>
                <<<grid, block, 0, stream>>>(src, dst, tp, words);
            break;
    }
    return cudaGetLastError();
}

template <typename SrcT>
cudaError_t launchEdge(DeviceImageConstView<SrcT> src, DeviceBinMatView dst, SrcT t,
                       EdgeCombine combine, EdgeRelation relation, EdgeSpatial spatial,
                       cudaStream_t stream) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "cuda edgeThreshold: src and dst must have the same dimensions");
    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda edgeThreshold: a non-empty image needs non-null pointers");
    BINCV_ASSERT(dst.stride >= rowWords(dst.width),
                 "cuda edgeThreshold: dst's stride must cover a whole row");
    BINCV_ASSERT(src.stride >= src.width,
                 "cuda edgeThreshold: src's stride must cover a whole row");

    // The relation folds into the threshold, as the host's own vector arm folds
    // it. `tp == 256` at t == 255 with Gt is simply "nothing passes": no unsigned
    // byte difference reaches it, so no special case is needed.
    const unsigned tp =
        static_cast<unsigned>(t) + (relation == EdgeRelation::Gt ? 1u : 0u);
    const size_t words = rowWords(dst.width);

    if constexpr (sizeof(SrcT) == 1) {
        if (impl::edgeVectorEnabled() &&
            impl::edgeVectorApplies(src.width, src.stride, src.ptr, sizeof(SrcT), spatial,
                                    tp)) {
            const dim3 block(32, 8);
            const size_t groups = (words + 3) / 4;
            const dim3 grid = warpGrid(groups * src.height, block.y);
            DeviceImageConstView<uint8_t> src8{reinterpret_cast<const uint8_t*>(src.ptr),
                                               src.width, src.height, src.stride};
            if (combine == EdgeCombine::Or) {
                edgeKernelVec<EdgeCombine::Or>
                    <<<grid, block, 0, stream>>>(src8, dst, tp, words, groups);
            } else {
                edgeKernelVec<EdgeCombine::And>
                    <<<grid, block, 0, stream>>>(src8, dst, tp, words, groups);
            }
            return cudaGetLastError();
        }
    }

    if (combine == EdgeCombine::Or)
        return launchRefSpatial<SrcT, EdgeCombine::Or>(src, dst, tp, spatial, words,
                                                       stream);
    return launchRefSpatial<SrcT, EdgeCombine::And>(src, dst, tp, spatial, words, stream);
}

} // namespace

namespace impl {

bool& edgeVectorEnabled() {
    static bool on = true;
    return on;
}

bool edgeVectorApplies(size_t width, size_t stride, const void* base, size_t srcElemSize,
                       EdgeSpatial spatial, unsigned tp) {
    if (srcElemSize != 1) return false;
    if (spatial != EdgeSpatial::Wide) return false;
    if (tp > 255u) return false;
    if (width < kEdgeVecPixelsPerWarp) return false;
    if (stride % 4u != 0u) return false;
    return (reinterpret_cast<std::uintptr_t>(base) % 4u) == 0u;
}

} // namespace impl

cudaError_t edgeThreshold(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                          uint8_t t, EdgeCombine combine, EdgeRelation relation,
                          EdgeSpatial spatial, cudaStream_t stream) {
    return launchEdge<uint8_t>(src, dst, t, combine, relation, spatial, stream);
}

cudaError_t edgeThreshold(DeviceImageConstView<uint16_t> src, DeviceBinMatView dst,
                          uint16_t t, EdgeCombine combine, EdgeRelation relation,
                          EdgeSpatial spatial, cudaStream_t stream) {
    return launchEdge<uint16_t>(src, dst, t, combine, relation, spatial, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
