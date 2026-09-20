// THE PACKER'S THREE ARMS, HELD TO ONE ANSWER.
//
// `cuda::packBits`, `cuda::packRows`, `cuda::packQuant` and `cuda::threshold`
// all reach the device through the same kernel, and that kernel now has three
// shapes behind two runtime switches: the grid-stride arm it shipped with (the
// ORACLE), a row-grid arm that carries the image row in `blockIdx.y`, and a
// byte-lane arm that resolves four pixels per lane. Which one runs is a
// performance decision. That it produces a different matrix would be a
// correctness failure, and this suite is where that can be observed.
//
// WHAT THIS SUITE DOES THAT THE OTHERS CANNOT
//
//   * IT TOGGLES THE ARMS. Every other backend suite exercises whatever the
//     switches default to, so two of the three arms are invisible to all of
//     them. Here every case runs the host twin and ALL THREE device arms in ONE
//     binary and compares the three to the host and to each other. Two binaries
//     built with different defaults would not be the same check: a switch that
//     silently stopped switching would pass that and fail this.
//
//   * IT EXERCISES BOTH GATES, INCLUDING FROM THE EXCLUDED SIDE. Each fast arm
//     has a domain it cannot express, and the value of a gate is entirely in
//     what it refuses. The row grid puts the row in `gridDim.y`, which the
//     hardware caps at 65535, so a 70,000-row image must fall to the
//     grid-stride arm AND STILL BE RIGHT. The byte lane needs a 4-byte-aligned
//     base and stride, a uint8 source and a cutoff a byte can hold, so an
//     odd-strided view, a uint16 source and `GreaterThan` at 255 must each fall
//     to the row grid AND STILL BE RIGHT. Those four cases are the ones an arm
//     that quietly ran outside its domain would corrupt, and nothing else here
//     would notice.
//
//   * IT CHECKS PADDING EXPLICITLY, in every destination of every case.
//     `mismatchWords` compares only `rowWords` words per row, so a set bit past
//     `width` reads as a match on both sides. The byte-lane arm makes that a
//     live question rather than a formality: its trailing quad straddles
//     `width`, and its store covers four words at a time, so a row whose word
//     count is not a multiple of four is exactly where it could write one word
//     too many.
//
// The size matrix is chosen for the two arms' seams, not for coverage as such:
// widths that are and are not multiples of 4 (the quad), of 32 (the word) and
// of 128 (the byte-lane warp group), the degenerate extents, and two sizes
// whose `rowWords` is not a multiple of four so the group's last store is
// partly out of range.
//
// Exits 77 when no CUDA device is present, like the other suites here.

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/pack.hpp"
#include "bincv/cuda/threshold.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/threshold.hpp"
#include "bincv/quantMat.hpp"
#include "test_util.hpp"

namespace {

namespace bc = bincv::cuda;

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct Size {
    size_t w, h;
};

/// @brief The seams both fast arms have, spelled as widths.
/// @note 752 is the reference frame (24 words per row -- NOT a multiple of 4,
/// so the byte-lane group's last store runs past the row). 1000 gives 32
/// words, which is. 129 is one pixel past a whole warp group; 128 is
/// exactly one; 127 is one short. 97, 33, 31, 9 and 1 are not multiples of
/// 4, so every one of them puts a lane's quad across `width`. 64 and 96 are
/// multiples of 4 and of 32 but not of 128.
const Size kSizes[] = {{752, 480}, {1000, 17}, {129, 5}, {128, 3}, {127, 4},
                       {97, 13},   {96, 2},    {64, 5},  {33, 2},  {31, 7},
                       {9, 1},     {1, 9},     {1, 1}};

template <typename T>
std::vector<T> randomFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<T> img(w * h);
    for (auto& v : img) v = static_cast<T>(splitmix(seed));
    return img;
}

/// @brief A source containing EVERY value the type can hold, cycled, so that
/// `t - 1`, `t` and `t + 1` are all present for every threshold swept.
template <typename T>
std::vector<T> everyValueFrame(size_t w, size_t h) {
    std::vector<T> img(w * h);
    const size_t period = size_t{1} << (8 * sizeof(T));
    for (size_t i = 0; i < img.size(); ++i) img[i] = static_cast<T>(i % period);
    return img;
}

size_t mismatchWords(const bincv::BinMat<uint32_t>& expect,
                     const bincv::BinMat<uint32_t>& got) {
    size_t bad = 0;
    const size_t words = bc::rowWords(expect.getWidth());
    for (size_t y = 0; y < expect.getHeight(); ++y) {
        const uint32_t* a = expect.constView().row(y);
        const uint32_t* b = got.constView().row(y);
        for (size_t i = 0; i < words; ++i)
            if (a[i] != b[i]) ++bad;
    }
    return bad;
}

/// @brief Set bits PAST `width` in any row's trailing word. Must be zero.
size_t dirtyPaddingBits(const bincv::BinMat<uint32_t>& m) {
    const size_t words = bc::rowWords(m.getWidth());
    const uint32_t tail = bc::rowTailMask(m.getWidth());
    if (tail == 0xFFFFFFFFu) return 0;  // the row ends on a word boundary
    size_t bad = 0;
    for (size_t y = 0; y < m.getHeight(); ++y) {
        const uint32_t v = m.constView().row(y)[words - 1] & ~tail;
        bad += static_cast<size_t>(__builtin_popcount(v));
    }
    return bad;
}

/// @brief Runs a device op into a POISONED device matrix and downloads it.
/// @note The poison is load-bearing for this suite specifically: the byte-lane
/// arm stores four words per group and guards the tail, so an arm that skips
/// the row's last word would otherwise inherit whatever the allocation held.
template <typename Launch>
bincv::BinMat<uint32_t> runDevice(size_t w, size_t h, Launch&& launch) {
    bc::DeviceBinMat d(static_cast<int>(w), static_cast<int>(h));
    bincv::BinMat<uint32_t> poison(static_cast<int>(w), static_cast<int>(h));
    for (size_t y = 0; y < h; ++y) {
        uint32_t* row = poison.view().row(y);
        for (size_t i = 0; i < d.getAlignedWidth(); ++i) row[i] = 0xA5A5A5A5u;
    }
    bincv::BinMat<uint32_t> got(static_cast<int>(w), static_cast<int>(h));
    // EVERY CUDA CALL HERE IS BOUND TO A LOCAL FIRST. BINCV_CHECK_EQ expands its
    // first argument twice -- once for the comparison and once for the failure
    // message -- so a call written inline runs twice. That is merely slow for an
    // idempotent upload and it is a DOUBLE FREE for cudaFree, which is how this
    // suite found it.
    const cudaError_t up = bc::upload(poison.constView(), d.view());
    BINCV_CHECK_EQ(up, cudaSuccess);
    const cudaError_t rc = launch(d.view());
    BINCV_CHECK_EQ(rc, cudaSuccess);
    const cudaError_t down = bc::download(d.constView(), got.view());
    BINCV_CHECK_EQ(down, cudaSuccess);
    const cudaError_t sync = cudaDeviceSynchronize();
    BINCV_CHECK_EQ(sync, cudaSuccess);
    return got;
}

/// @brief The three arms, selected by the two switches, in the order they are
/// tried at runtime. `restore` puts the switches back.
struct ArmSelection {
    const char* name;
    bool rowGrid;
    bool byteLane;
};
const ArmSelection kArms[] = {
    {"grid-stride (oracle)", false, false},
    {"row grid", true, false},
    {"byte lane", true, true},
};

/// @brief Sets both switches and returns them to their defaults on scope exit.
class ArmScope {
public:
    explicit ArmScope(const ArmSelection& a)
        : rowGrid_(bc::impl::packRowGridEnabled()),
          byteLane_(bc::impl::packByteLaneEnabled()) {
        bc::impl::packRowGridEnabled() = a.rowGrid;
        bc::impl::packByteLaneEnabled() = a.byteLane;
    }
    ~ArmScope() {
        bc::impl::packRowGridEnabled() = rowGrid_;
        bc::impl::packByteLaneEnabled() = byteLane_;
    }
    ArmScope(const ArmScope&) = delete;
    ArmScope& operator=(const ArmScope&) = delete;

private:
    bool rowGrid_, byteLane_;
};

/// @brief The host packer, which is the truth for every arm below.
template <bincv::PackRule R, typename SrcT>
bincv::BinMat<uint32_t> hostPack(const std::vector<SrcT>& src, size_t w, size_t h,
                                 SrcT t) {
    bincv::BinMat<uint32_t> out(static_cast<int>(w), static_cast<int>(h));
    bincv::packBits<R, SrcT, uint32_t>(src.data(), w, h, w, out.view(), t);
    return out;
}

template <typename SrcT>
bincv::BinMat<uint32_t> hostPackRule(const std::vector<SrcT>& src, size_t w, size_t h,
                                     bincv::PackRule rule, SrcT t) {
    switch (rule) {
        case bincv::PackRule::NonZero:
            return hostPack<bincv::PackRule::NonZero, SrcT>(src, w, h, t);
        case bincv::PackRule::GreaterThan:
            return hostPack<bincv::PackRule::GreaterThan, SrcT>(src, w, h, t);
        case bincv::PackRule::GreaterEqual:
            return hostPack<bincv::PackRule::GreaterEqual, SrcT>(src, w, h, t);
    }
    return hostPack<bincv::PackRule::NonZero, SrcT>(src, w, h, t);
}

const bincv::PackRule kRules[] = {bincv::PackRule::NonZero, bincv::PackRule::GreaterThan,
                                  bincv::PackRule::GreaterEqual};

} // namespace

// ---------------------------------------------------------------------------
// packBits -- every arm, every rule, every seam
// ---------------------------------------------------------------------------

BINCV_TEST(CudaPackArms, EveryArmMatchesTheHostPacker) {
    // Named here rather than left implicit: a reader of a failure needs to know
    // which three shapes were compared, and there is no other place in the run
    // where the arm roster appears.
    std::printf("  arms held to one answer:");
    for (const ArmSelection& arm : kArms)
        std::printf(" [%s rowGrid=%d byteLane=%d]", arm.name, arm.rowGrid ? 1 : 0,
                    arm.byteLane ? 1 : 0);
    std::printf("\n");
    size_t bad = 0, dirty = 0, cases = 0;
    for (const Size& s : kSizes) {
        const auto frame = randomFrame<uint8_t>(s.w, s.h, 0xC0FFEEu + s.w);
        bc::DeviceImage<uint8_t> dImg(static_cast<int>(s.w), static_cast<int>(s.h));
        const cudaError_t upImg =
            bc::uploadImage<uint8_t>(frame.data(), s.w, s.h, s.w, dImg.view());
BINCV_CHECK_EQ(upImg, cudaSuccess);
        // Thresholds at the ends AND in the middle: 0 and 255 are where the
        // folded cutoff reaches 0 and 256, the two values a byte lane cannot
        // hold, and 1 and 254 are their neighbours.
        const uint8_t kT[] = {0, 1, 17, 128, 254, 255};
        for (bincv::PackRule rule : kRules) {
            for (uint8_t t : kT) {
                const auto expect = hostPackRule<uint8_t>(frame, s.w, s.h, rule, t);
                for (const ArmSelection& arm : kArms) {
                    const ArmScope sel(arm);
                    const auto got = runDevice(s.w, s.h, [&](bc::DeviceBinMatView dst) {
                        return bc::packBits(dImg.constView(), dst, rule, t);
                    });
                    bad += mismatchWords(expect, got);
                    dirty += dirtyPaddingBits(got);
                    ++cases;
                }
            }
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
    BINCV_CHECK(cases > 700);
}

BINCV_TEST(CudaPackArms, EveryArmMatchesTheHostPackerOnUint16) {
    // The byte-lane arm's gate REFUSES this source type, so what this case
    // proves is that the refusal reaches the row grid and the row grid is
    // right -- with both switches on, which is the shipped configuration.
    size_t bad = 0, dirty = 0;
    for (const Size& s : kSizes) {
        const auto frame = randomFrame<uint16_t>(s.w, s.h, 0xBEEFu + s.w);
        bc::DeviceImage<uint16_t> dImg(static_cast<int>(s.w), static_cast<int>(s.h));
        const cudaError_t upImg =
            bc::uploadImage<uint16_t>(frame.data(), s.w, s.h, s.w, dImg.view());
BINCV_CHECK_EQ(upImg, cudaSuccess);
        const uint16_t kT[] = {0, 1, 4095, 32768, 65534, 65535};
        for (bincv::PackRule rule : kRules) {
            for (uint16_t t : kT) {
                const auto expect = hostPackRule<uint16_t>(frame, s.w, s.h, rule, t);
                for (const ArmSelection& arm : kArms) {
                    const ArmScope sel(arm);
                    const auto got = runDevice(s.w, s.h, [&](bc::DeviceBinMatView dst) {
                        return bc::packBits(dImg.constView(), dst, rule, t);
                    });
                    bad += mismatchWords(expect, got);
                    dirty += dirtyPaddingBits(got);
                }
            }
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
}

// ---------------------------------------------------------------------------
// The gates, from the side that is REFUSED
// ---------------------------------------------------------------------------

BINCV_TEST(CudaPackArms, ByteLaneGateRefusesWhatItCannotExpress) {
    // Four refusals, each for a reason the kernel would otherwise get wrong:
    // a 16-bit element (`__vsetgeu4` compares bytes), a cutoff of 256 (no byte
    // is >= 256, and no byte lane can say so), a stride that breaks the aligned
    // 32-bit load, and a base pointer that does the same.
    alignas(4) uint8_t buf[16] = {0};
    const void* aligned = buf;
    const void* offByOne = buf + 1;
    BINCV_CHECK(bc::impl::packByteLaneApplies(64, aligned, 1, 128));
    BINCV_CHECK(bc::impl::packByteLaneApplies(64, aligned, 1, 0));
    BINCV_CHECK(bc::impl::packByteLaneApplies(64, aligned, 1, 255));
    BINCV_CHECK(!bc::impl::packByteLaneApplies(64, aligned, 2, 128));
    BINCV_CHECK(!bc::impl::packByteLaneApplies(64, aligned, 1, 256));
    BINCV_CHECK(!bc::impl::packByteLaneApplies(65, aligned, 1, 128));
    BINCV_CHECK(!bc::impl::packByteLaneApplies(66, aligned, 1, 128));
    BINCV_CHECK(!bc::impl::packByteLaneApplies(64, offByOne, 1, 128));
}

BINCV_TEST(CudaPackArms, RowGridGateStopsAtTheHardwareLimit) {
    // `gridDim.y` is 65535. The gate is that number and not a rounder one,
    // because an image of exactly 65535 rows must still take the fast arm.
    BINCV_CHECK(bc::impl::packRowGridApplies(1));
    BINCV_CHECK(bc::impl::packRowGridApplies(65535));
    BINCV_CHECK(!bc::impl::packRowGridApplies(65536));
    BINCV_CHECK(!bc::impl::packRowGridApplies(70000));
}

BINCV_TEST(CudaPackArms, AnOddStrideFallsToTheRowGridAndIsStillRight) {
    // A source VIEW whose stride is not a multiple of four -- the shape a
    // caller gets from a sub-view of a wider buffer. The byte-lane arm cannot
    // load an aligned quad from row 1 onwards, its gate says so, and the answer
    // must be unchanged.
    const size_t w = 129, h = 7, stride = w;  // 129 is odd, so row 1 is odd too
    BINCV_CHECK(!bc::impl::packByteLaneApplies(stride, reinterpret_cast<void*>(4), 1, 17));
    auto frame = randomFrame<uint8_t>(stride, h, 0x5150u);
    uint8_t* dev = nullptr;
    const cudaError_t alloc = cudaMalloc(&dev, stride * h);
    BINCV_CHECK_EQ(alloc, cudaSuccess);
    const cudaError_t copy =
        cudaMemcpy(dev, frame.data(), stride * h, cudaMemcpyHostToDevice);
    BINCV_CHECK_EQ(copy, cudaSuccess);
    const bc::DeviceImageConstView<uint8_t> src{dev, w, h, stride};

    std::vector<uint8_t> tight(w * h);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) tight[y * w + x] = frame[y * stride + x];
    const auto expect =
        hostPack<bincv::PackRule::GreaterEqual, uint8_t>(tight, w, h, uint8_t{17});

    size_t bad = 0, dirty = 0;
    for (const ArmSelection& arm : kArms) {
        const ArmScope sel(arm);
        const auto got = runDevice(w, h, [&](bc::DeviceBinMatView dst) {
            return bc::packBits(src, dst, bincv::PackRule::GreaterEqual, uint8_t{17});
        });
        bad += mismatchWords(expect, got);
        dirty += dirtyPaddingBits(got);
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
    const cudaError_t freed = cudaFree(dev);
    BINCV_CHECK_EQ(freed, cudaSuccess);
}

BINCV_TEST(CudaPackArms, AnImageTallerThanGridDimYFallsToGridStride) {
    // 70,000 rows is past `gridDim.y`, so the row grid cannot express this
    // launch at all and the grid-stride arm -- the one shape with no bound --
    // has to answer. 32 pixels wide keeps it to 2.2 MB of source.
    const size_t w = 32, h = 70000;
    BINCV_CHECK(!bc::impl::packRowGridApplies(h));
    const auto frame = randomFrame<uint8_t>(w, h, 0x7AAAu);
    bc::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    const cudaError_t upImg =
        bc::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view());
BINCV_CHECK_EQ(upImg, cudaSuccess);
    const auto expect =
        hostPack<bincv::PackRule::GreaterEqual, uint8_t>(frame, w, h, uint8_t{128});
    size_t bad = 0;
    for (const ArmSelection& arm : kArms) {
        const ArmScope sel(arm);
        const auto got = runDevice(w, h, [&](bc::DeviceBinMatView dst) {
            return bc::packBits(dImg.constView(), dst, bincv::PackRule::GreaterEqual,
                                uint8_t{128});
        });
        bad += mismatchWords(expect, got);
    }
    BINCV_CHECK_EQ(bad, 0u);
}

// ---------------------------------------------------------------------------
// The rule fold -- the byte-lane arm's one semantic liberty
// ---------------------------------------------------------------------------

BINCV_TEST(CudaPackArms, TheFoldedCutoffAgreesWithTheHostAtEveryBoundary) {
    // The byte-lane arm does not evaluate three rules; it evaluates `v >= c`
    // for one cutoff folded host-side. The three identities that fold has to
    // satisfy are checkable as WHOLE-IMAGE claims against a frame that holds
    // every byte, which is what makes an off-by-one here visible rather than
    // rare:
    //     NonZero            == GreaterEqual at 1
    //     GreaterThan at t   == GreaterEqual at t + 1
    //     GreaterThan at 255 == nothing passes  (cutoff 256, a byte cannot)
    //     GreaterEqual at 0  == everything passes, padding still clean
    const size_t w = 257, h = 9;
    const auto frame = everyValueFrame<uint8_t>(w, h);
    bc::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    const cudaError_t upImg =
        bc::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view());
BINCV_CHECK_EQ(upImg, cudaSuccess);
    const auto pack = [&](bincv::PackRule rule, uint8_t t) {
        return runDevice(w, h, [&](bc::DeviceBinMatView dst) {
            return bc::packBits(dImg.constView(), dst, rule, t);
        });
    };
    for (const ArmSelection& arm : kArms) {
        const ArmScope sel(arm);
        const auto nonZero = pack(bincv::PackRule::NonZero, 0);
        const auto geOne = pack(bincv::PackRule::GreaterEqual, 1);
        BINCV_CHECK_EQ(mismatchWords(nonZero, geOne), 0u);

        for (unsigned t = 0; t < 255; ++t) {
            const auto gt = pack(bincv::PackRule::GreaterThan, static_cast<uint8_t>(t));
            const auto ge =
                pack(bincv::PackRule::GreaterEqual, static_cast<uint8_t>(t + 1));
            BINCV_CHECK_EQ(mismatchWords(gt, ge), 0u);
        }

        const auto nothing = pack(bincv::PackRule::GreaterThan, 255);
        size_t set = 0;
        for (size_t y = 0; y < h; ++y)
            for (size_t i = 0; i < bc::rowWords(w); ++i)
                set += static_cast<size_t>(__builtin_popcount(nothing.constView().row(y)[i]));
        BINCV_CHECK_EQ(set, 0u);

        const auto everything = pack(bincv::PackRule::GreaterEqual, 0);
        size_t lit = 0;
        for (size_t y = 0; y < h; ++y)
            for (size_t i = 0; i < bc::rowWords(w); ++i)
                lit += static_cast<size_t>(
                    __builtin_popcount(everything.constView().row(y)[i]));
        BINCV_CHECK_EQ(lit, w * h);
        BINCV_CHECK_EQ(dirtyPaddingBits(everything), 0u);
    }
}

// ---------------------------------------------------------------------------
// packRows -- the banded destination the arms address through `dstRow`
// ---------------------------------------------------------------------------

BINCV_TEST(CudaPackArms, EveryArmFillsABandOfALargerDestination) {
    const size_t w = 97, chunk = 5, bands = 4, h = chunk * bands;
    const auto frame = randomFrame<uint8_t>(w, h, 0xBA5Eu);
    bc::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    const cudaError_t upImg =
        bc::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view());
BINCV_CHECK_EQ(upImg, cudaSuccess);
    const auto expect =
        hostPack<bincv::PackRule::GreaterEqual, uint8_t>(frame, w, h, uint8_t{64});

    size_t bad = 0, dirty = 0;
    for (const ArmSelection& arm : kArms) {
        const ArmScope sel(arm);
        const auto got = runDevice(w, h, [&](bc::DeviceBinMatView dst) {
            // One band at a time, so `dstRow` is non-zero for three of the four.
            for (size_t b = 0; b < bands; ++b) {
                const bc::DeviceImageConstView<uint8_t> band{
                    dImg.constView().ptr + b * chunk * dImg.getStride(), w, chunk,
                    dImg.getStride()};
                const cudaError_t rc = bc::packRows(band, dst, b * chunk,
                                                    bincv::PackRule::GreaterEqual,
                                                    uint8_t{64});
                if (rc != cudaSuccess) return rc;
            }
            return cudaDeviceSynchronize();
        });
        bad += mismatchWords(expect, got);
        dirty += dirtyPaddingBits(got);
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
}

// ---------------------------------------------------------------------------
// packQuant -- the N-deep twin, which got the row grid and not the byte lane
// ---------------------------------------------------------------------------

namespace {

/// @brief One (N, size) case of the quant packer, every arm, against the host.
template <size_t N>
void quantCase(const Size& s, size_t& bad, size_t& dirty) {
    const auto frame = randomFrame<uint8_t>(s.w, s.h, 0x9911u + s.h + N);
    bc::DeviceImage<uint8_t> dImg(static_cast<int>(s.w), static_cast<int>(s.h));
    const cudaError_t upImg =
        bc::uploadImage<uint8_t>(frame.data(), s.w, s.h, s.w, dImg.view());
BINCV_CHECK_EQ(upImg, cudaSuccess);

    // The host's own quantizer, plane by plane, as the truth.
    std::vector<bincv::BinMat<uint32_t>> planes;
    planes.reserve(N);
    for (size_t p = 0; p < N; ++p)
        planes.emplace_back(static_cast<int>(s.w), static_cast<int>(s.h));
    bincv::BinMatView<uint32_t> views[N];
    for (size_t p = 0; p < N; ++p) views[p] = planes[p].view();
    bincv::packQuant<bincv::QuantRule::Scale, N, uint8_t, uint32_t>(frame.data(), s.w,
                                                                    s.h, s.w, views);

    for (const ArmSelection& arm : kArms) {
        const ArmScope sel(arm);
        const auto got = runDevice(s.w, N * s.h, [&](bc::DeviceBinMatView dst) {
            return bc::packQuant(dImg.constView(), dst, N);
        });
        for (size_t p = 0; p < N; ++p) {
            bincv::BinMat<uint32_t> plane(static_cast<int>(s.w), static_cast<int>(s.h));
            for (size_t y = 0; y < s.h; ++y)
                for (size_t i = 0; i < bc::rowWords(s.w); ++i)
                    plane.view().row(y)[i] = got.constView().row(p * s.h + y)[i];
            bad += mismatchWords(planes[p], plane);
            dirty += dirtyPaddingBits(plane);
        }
    }
}

} // namespace

BINCV_TEST(CudaPackArms, EveryArmMatchesTheHostQuantPacker) {
    size_t bad = 0, dirty = 0;
    for (const Size& s : kSizes) {
        quantCase<1>(s, bad, dirty);
        quantCase<2>(s, bad, dirty);
        quantCase<4>(s, bad, dirty);
        quantCase<8>(s, bad, dirty);
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
}

// ---------------------------------------------------------------------------
// threshold -- the Tier 1 caller, through all three arms
// ---------------------------------------------------------------------------

namespace {

/// @brief The host oracle for `cuda::threshold`: the host's OWN cutoff reduction
/// composed with the host's OWN packer -- the body of
/// `bincv::threshold(const cv::Mat&, ...)`, reachable with no OpenCV.
bincv::BinMat<uint32_t> hostThreshold(const std::vector<uint8_t>& src, size_t w, size_t h,
                                      double thresh) {
    bincv::BinMat<uint32_t> out(static_cast<int>(w), static_cast<int>(h));
    const size_t words = bc::rowWords(w);
    const int cutoff = bincv::impl::thresholdCutoff(thresh);
    if (cutoff <= 0) {
        const uint32_t tail = bc::rowTailMask(w);
        for (size_t y = 0; y < h; ++y) {
            uint32_t* row = out.view().row(y);
            for (size_t i = 0; i < words; ++i) row[i] = (i + 1 == words) ? tail : ~0u;
        }
        return out;
    }
    if (cutoff > 255) {
        for (size_t y = 0; y < h; ++y) {
            uint32_t* row = out.view().row(y);
            for (size_t i = 0; i < words; ++i) row[i] = 0;
        }
        return out;
    }
    bincv::packBits<bincv::PackRule::GreaterEqual, uint8_t, uint32_t>(
        src.data(), w, h, w, out.view(), static_cast<uint8_t>(cutoff));
    return out;
}

} // namespace

BINCV_TEST(CudaPackArms, ThresholdIsTheHostsAnswerOnEveryArm) {
    // The whole integer range plus the fractional and out-of-domain values the
    // host header pins, on a frame holding every byte -- run three times, once
    // per arm. What the fast arms could break here and nowhere else is the
    // BOUNDARY: `p >= cutoff` is `p > thresh`, and a byte-lane compare that
    // folded one off would move exactly the pixels equal to `thresh`.
    const size_t w = 129, h = 11;  // one pixel past a whole byte-lane group
    const auto frame = everyValueFrame<uint8_t>(w, h);
    bc::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    const cudaError_t upImg =
        bc::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view());
BINCV_CHECK_EQ(upImg, cudaSuccess);

    const double kInf = std::numeric_limits<double>::infinity();
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    size_t bad = 0, dirty = 0;
    for (const ArmSelection& arm : kArms) {
        const ArmScope sel(arm);
        for (int t = -2; t <= 257; ++t) {
            const double thresh = static_cast<double>(t);
            const auto expect = hostThreshold(frame, w, h, thresh);
            const auto got = runDevice(w, h, [&](bc::DeviceBinMatView dst) {
                return bc::threshold(dImg.constView(), dst, thresh);
            });
            bad += mismatchWords(expect, got);
            dirty += dirtyPaddingBits(got);
        }
        const double odd[] = {127.5, 0.5, 254.5, -0.5, 1e300, -1e300, kInf, -kInf, kNaN};
        for (double thresh : odd) {
            const auto expect = hostThreshold(frame, w, h, thresh);
            const auto got = runDevice(w, h, [&](bc::DeviceBinMatView dst) {
                return bc::threshold(dImg.constView(), dst, thresh);
            });
            bad += mismatchWords(expect, got);
            dirty += dirtyPaddingBits(got);
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
}

BINCV_TEST(CudaPackArms, ThresholdAgreesAcrossArmsAtTheReferenceFrame) {
    // The reference geometry, whose 24 words per row is NOT a multiple of four:
    // the byte-lane group's last store addresses words 24..27, three of which
    // are past the row. This is the case that would corrupt the NEXT row if the
    // store guard were missing, which a small frame may not reveal.
    const size_t w = 752, h = 480;
    const auto frame = randomFrame<uint8_t>(w, h, 0xFEEDu);
    bc::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    const cudaError_t upImg =
        bc::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view());
BINCV_CHECK_EQ(upImg, cudaSuccess);
    size_t bad = 0;
    for (double thresh : {0.0, 17.0, 127.0, 254.0}) {
        const auto expect = hostThreshold(frame, w, h, thresh);
        for (const ArmSelection& arm : kArms) {
            const ArmScope sel(arm);
            const auto got = runDevice(w, h, [&](bc::DeviceBinMatView dst) {
                return bc::threshold(dImg.constView(), dst, thresh);
            });
            bad += mismatchWords(expect, got);
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
}

// ---------------------------------------------------------------------------
// The switches themselves
// ---------------------------------------------------------------------------

BINCV_TEST(CudaPackArms, BothSwitchesDefaultOnAndRestore) {
    // A mis-attached switch that never selects anything is the failure this
    // project has already had once, and a default that quietly flipped would
    // make every number in the benchmark a measurement of the wrong arm.
    BINCV_CHECK(bc::impl::packRowGridEnabled());
    BINCV_CHECK(bc::impl::packByteLaneEnabled());
    {
        const ArmScope sel(kArms[0]);
        BINCV_CHECK(!bc::impl::packRowGridEnabled());
        BINCV_CHECK(!bc::impl::packByteLaneEnabled());
    }
    BINCV_CHECK(bc::impl::packRowGridEnabled());
    BINCV_CHECK(bc::impl::packByteLaneEnabled());
}

// ---------------------------------------------------------------------------
// Entry point: probe the device first, and report "not performed" as 77 --
// a pass this binary did not earn is worse than a skip it announces.
// ---------------------------------------------------------------------------
namespace {
bool cudaDevicePresent() {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device available (%s)\n",
                    err == cudaSuccess ? "zero devices" : cudaGetErrorString(err));
        return false;
    }
    return true;
}
} // namespace

#if BINCV_TEST_WITH_GTEST
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    const int summaryRc = ::bincv::test::summarize("CUDA packer-arm tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA packer-arm tests", argc, argv);
}
#endif
