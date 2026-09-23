// THE PACKER'S ARMS, HELD TO ONE ANSWER.
//
// `cuda::packBits`, `cuda::packRows`, `cuda::packQuant` and `cuda::threshold`
// all reach the device through the same kernels, and those have several shapes
// behind two runtime switches: the grid-stride arm packBits shipped with (the
// ORACLE), a row-grid arm that carries the row and the word in the launch
// shape, and for packQuant a wide-lane arm that scales sixteen pixels per
// lane. Which one runs is a performance decision. That it
// produces a different matrix would be a correctness failure, and this suite is
// where that can be observed.
//
// WHAT THIS SUITE DOES THAT THE OTHERS CANNOT
//
//   * IT TOGGLES THE ARMS. Every other backend suite exercises whatever the
//     switches default to, so every arm but the fastest is invisible to all of
//     them. Here every case runs the host twin and EVERY device arm in ONE
//     binary and compares each to the host. Two binaries built with different
//     defaults would not be the same check: a switch that silently stopped
//     switching would pass that and fail this.
//
//   * IT EXERCISES EVERY GATE, INCLUDING FROM THE EXCLUDED SIDE. Each fast arm
//     has a domain it cannot express, and the value of a gate is entirely in
//     what it refuses. The row grid is capped at 65535 rows by a grid
//     dimension, so a 70,000-row image must fall to the grid-stride arm AND
//     STILL BE RIGHT. packQuant's wide lane needs a 16-byte-aligned base and
//     stride and a uint8 source, so a stride of 4 mod 16 and a uint16 source
//     must each fall to the row grid AND STILL BE RIGHT. Those are the cases an
//     arm that quietly ran outside its domain would corrupt, and nothing else
//     here would notice.
//
//   * IT PUTS THE WIDE LANE WHERE ITS TAILS ARE. A tight stride is 16-byte
//     aligned only when the width is a multiple of 16, which is exactly when no
//     lane's sixteen pixels straddle `width`. So the quant size matrix is also
//     run on sources whose stride is padded to 16 bytes, with the padding
//     filled with 0xFF: every width then reaches the wide lane, and a lane that
//     read past `width` would set a bit that is not there.
//
//   * IT CHECKS PADDING EXPLICITLY, in every destination of every case.
//     `mismatchWords` compares only `rowWords` words per row, so a set bit past
//     `width` reads as a match on both sides. The fast arms make that a live
//     question rather than a formality: their stores cover several words at a
//     time, so a row whose word count is not a multiple of the group is exactly
//     where one could write a word too many.
//
// The size matrix is chosen for the arms' seams, not for coverage as such:
// widths that are and are not multiples of 4 (a 32-bit quarter of the load),
// of 16 (the wide lane), of 32 (the word), of 256 (a ballot unit of eight
// words) and of 512 (a wide-lane warp), and the degenerate extents -- so that
// a unit's last store is partly out of range at most of them.
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

/// @brief The seams the fast arms have, spelled as widths.
/// @note 752 is the reference frame: 24 words per row, three whole ballot units
/// and a wide-lane warp and a half, the second warp's last eight words past
/// the row. 1000 gives 32 words and a wide lane straddling `width` at a padded
/// stride. 129, 128 and 127 sit on either side of a word boundary; 97, 33,
/// 31, 9 and 1 are not multiples of 4, so a wide lane's quarter straddles
/// `width`. 64 and 96 are multiples of 16 and of 32 but fill no ballot unit.
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
/// @note The poison is load-bearing for this suite specifically: the row grid
/// stores eight words per warp and packQuant's wide lane sixteen, each guarding
/// the row's end, so an arm that skipped the row's last word would otherwise
/// inherit whatever the allocation held.
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

/// @brief The arms, selected by the switches, in the order they are tried at
/// runtime. Each switch sits one level below the one before it.
struct ArmSelection {
    const char* name;
    bool rowGrid;
    bool quantWide;
};
/// @brief packBits' two arms.
const ArmSelection kArms[] = {
    {"grid-stride (oracle)", false, false},
    {"row grid", true, false},
};
/// @brief packQuant's three arms.
const ArmSelection kQuantArms[] = {
    {"grid-stride (oracle)", false, false},
    {"row grid", true, false},
    {"wide lane", true, true},
};

/// @brief Sets every switch and returns them to their defaults on scope exit.
class ArmScope {
public:
    explicit ArmScope(const ArmSelection& a)
        : rowGrid_(bc::impl::packRowGridEnabled()),
          quantWide_(bc::impl::packQuantWideLaneEnabled()) {
        bc::impl::packRowGridEnabled() = a.rowGrid;
        bc::impl::packQuantWideLaneEnabled() = a.quantWide;
    }
    ~ArmScope() {
        bc::impl::packRowGridEnabled() = rowGrid_;
        bc::impl::packQuantWideLaneEnabled() = quantWide_;
    }
    ArmScope(const ArmScope&) = delete;
    ArmScope& operator=(const ArmScope&) = delete;

private:
    bool rowGrid_, quantWide_;
};

/// @brief A device source whose stride is `width` rounded up to 16 bytes, so
/// packQuant's wide-lane gate admits it at EVERY width -- including the widths whose last
/// sixteen-pixel lane straddles `width`, which a tight stride never shows it.
/// @note The bytes between `width` and the stride are filled with 0xFF, so a
/// lane that let one of them into its bits would set a padding bit or, for a
/// threshold, a pixel that is not there.
class PaddedSource {
public:
    PaddedSource(const std::vector<uint8_t>& frame, size_t w, size_t h)
        : w_(w), h_(h), stride_((w + 15) / 16 * 16 + 16) {
        std::vector<uint8_t> host(stride_ * h, 0xFF);
        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x) host[y * stride_ + x] = frame[y * w + x];
        const cudaError_t alloc = cudaMalloc(&dev_, stride_ * h);
        BINCV_CHECK_EQ(alloc, cudaSuccess);
        const cudaError_t copy =
            cudaMemcpy(dev_, host.data(), stride_ * h, cudaMemcpyHostToDevice);
        BINCV_CHECK_EQ(copy, cudaSuccess);
    }
    ~PaddedSource() { cudaFree(dev_); }
    PaddedSource(const PaddedSource&) = delete;
    PaddedSource& operator=(const PaddedSource&) = delete;
    bc::DeviceImageConstView<uint8_t> view() const { return {dev_, w_, h_, stride_}; }
    size_t stride() const { return stride_; }
    const void* base() const { return dev_; }

private:
    size_t w_, h_, stride_;
    uint8_t* dev_ = nullptr;
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
    // which shapes were compared, and there is no other place in the run where
    // the arm roster appears.
    std::printf("  arms held to one answer:");
    for (const ArmSelection& arm : kArms)
        std::printf(" [%s rowGrid=%d]", arm.name, arm.rowGrid ? 1 : 0);
    std::printf("\n");
    size_t bad = 0, dirty = 0, cases = 0;
    for (const Size& s : kSizes) {
        const auto frame = randomFrame<uint8_t>(s.w, s.h, 0xC0FFEEu + s.w);
        bc::DeviceImage<uint8_t> dImg(static_cast<int>(s.w), static_cast<int>(s.h));
        const cudaError_t upImg =
            bc::uploadImage<uint8_t>(frame.data(), s.w, s.h, s.w, dImg.view());
BINCV_CHECK_EQ(upImg, cudaSuccess);
        // Thresholds at the ends AND in the middle: 0 and 255 are where
        // GreaterEqual passes everything and GreaterThan nothing, and 1 and 254
        // are their neighbours.
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
    // Every (size, rule, threshold, arm), so a loop that silently ran short --
    // an arm dropped from the roster -- fails here rather than passing on less.
    BINCV_CHECK_EQ(cases, (sizeof(kSizes) / sizeof(kSizes[0])) * 3 * 6 *
                              (sizeof(kArms) / sizeof(kArms[0])));
}

BINCV_TEST(CudaPackArms, EveryArmMatchesTheHostPackerOnUint16) {
    // The row grid on its second source type, with every switch on, which is
    // the shipped configuration.
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

BINCV_TEST(CudaPackArms, WideLaneGateRefusesWhatItCannotExpress) {
    // The load is one aligned 16-byte access per lane, so the base AND every row
    // it strides to must be 16-byte aligned -- 4-byte alignment is not enough.
    // And the scale is of bytes.
    alignas(16) uint8_t buf[32] = {0};
    BINCV_CHECK(bc::impl::packWideLaneApplies(64, buf, 1));
    BINCV_CHECK(bc::impl::packWideLaneApplies(16, buf, 1));
    BINCV_CHECK(!bc::impl::packWideLaneApplies(68, buf, 1));
    BINCV_CHECK(!bc::impl::packWideLaneApplies(72, buf, 1));
    BINCV_CHECK(!bc::impl::packWideLaneApplies(76, buf, 1));
    BINCV_CHECK(!bc::impl::packWideLaneApplies(65, buf, 1));
    BINCV_CHECK(!bc::impl::packWideLaneApplies(64, buf + 4, 1));
    BINCV_CHECK(!bc::impl::packWideLaneApplies(64, buf + 8, 1));
    BINCV_CHECK(!bc::impl::packWideLaneApplies(64, buf, 2));
}

BINCV_TEST(CudaPackArms, AFourByteStrideFallsToTheQuantRowGridAndIsStillRight) {
    // Stride 132: a multiple of 4 and not of 16. packQuant's wide-lane gate
    // refuses it, so with every switch on this is the row grid answering -- and
    // it must be the host's answer.
    const size_t w = 129, h = 7, stride = 132;
    auto frame = randomFrame<uint8_t>(stride, h, 0x4B4Bu);
    uint8_t* dev = nullptr;
    const cudaError_t alloc = cudaMalloc(&dev, stride * h);
    BINCV_CHECK_EQ(alloc, cudaSuccess);
    const cudaError_t copy =
        cudaMemcpy(dev, frame.data(), stride * h, cudaMemcpyHostToDevice);
    BINCV_CHECK_EQ(copy, cudaSuccess);
    BINCV_CHECK(!bc::impl::packWideLaneApplies(stride, dev, 1));
    const bc::DeviceImageConstView<uint8_t> src{dev, w, h, stride};

    std::vector<uint8_t> tight(w * h);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) tight[y * w + x] = frame[y * stride + x];

    size_t bad = 0, dirty = 0;
    bincv::BinMat<uint32_t> planes[2] = {
        bincv::BinMat<uint32_t>(static_cast<int>(w), static_cast<int>(h)),
        bincv::BinMat<uint32_t>(static_cast<int>(w), static_cast<int>(h))};
    bincv::BinMatView<uint32_t> views[2] = {planes[0].view(), planes[1].view()};
    bincv::packQuant<bincv::QuantRule::Scale, 2, uint8_t, uint32_t>(tight.data(), w, h, w,
                                                                    views);
    for (const ArmSelection& arm : kQuantArms) {
        const ArmScope sel(arm);
        const auto got = runDevice(w, 2 * h, [&](bc::DeviceBinMatView dst) {
            return bc::packQuant(src, dst, 2);
        });
        for (size_t p = 0; p < 2; ++p) {
            bincv::BinMat<uint32_t> plane(static_cast<int>(w), static_cast<int>(h));
            for (size_t y = 0; y < h; ++y)
                for (size_t i = 0; i < bc::rowWords(w); ++i)
                    plane.view().row(y)[i] = got.constView().row(p * h + y)[i];
            bad += mismatchWords(planes[p], plane);
            dirty += dirtyPaddingBits(plane);
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
    const cudaError_t freed = cudaFree(dev);
    BINCV_CHECK_EQ(freed, cudaSuccess);
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
    // A source VIEW whose stride is odd -- the shape a caller gets from a
    // sub-view of a wider buffer. Every row after the first starts unaligned,
    // and the answer must be unchanged.
    const size_t w = 129, h = 7, stride = w;  // 129 is odd, so row 1 is odd too
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
// The rule identities -- every boundary of every rule
// ---------------------------------------------------------------------------

BINCV_TEST(CudaPackArms, TheRuleIdentitiesHoldAtEveryBoundary) {
    // Three rules over one byte are four identities, checkable as WHOLE-IMAGE
    // claims against a frame that holds every byte, which is what makes an
    // off-by-one here visible rather than rare:
    //     NonZero            == GreaterEqual at 1
    //     GreaterThan at t   == GreaterEqual at t + 1
    //     GreaterThan at 255 == nothing passes
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
// packQuant -- the N-deep twin: the row grid, and a wide-lane arm of its own
// ---------------------------------------------------------------------------

namespace {

/// @brief Every quant arm on `src` against the host's own quantizer on `frame`.
template <size_t N>
void quantCompare(const std::vector<uint8_t>& frame, size_t w, size_t h,
                  bc::DeviceImageConstView<uint8_t> src, size_t& bad, size_t& dirty) {
    std::vector<bincv::BinMat<uint32_t>> planes;
    planes.reserve(N);
    for (size_t p = 0; p < N; ++p)
        planes.emplace_back(static_cast<int>(w), static_cast<int>(h));
    bincv::BinMatView<uint32_t> views[N];
    for (size_t p = 0; p < N; ++p) views[p] = planes[p].view();
    bincv::packQuant<bincv::QuantRule::Scale, N, uint8_t, uint32_t>(frame.data(), w, h, w,
                                                                    views);

    for (const ArmSelection& arm : kQuantArms) {
        const ArmScope sel(arm);
        const auto got = runDevice(w, N * h, [&](bc::DeviceBinMatView dst) {
            return bc::packQuant(src, dst, N);
        });
        for (size_t p = 0; p < N; ++p) {
            bincv::BinMat<uint32_t> plane(static_cast<int>(w), static_cast<int>(h));
            for (size_t y = 0; y < h; ++y)
                for (size_t i = 0; i < bc::rowWords(w); ++i)
                    plane.view().row(y)[i] = got.constView().row(p * h + y)[i];
            bad += mismatchWords(planes[p], plane);
            dirty += dirtyPaddingBits(plane);
        }
    }
}

/// @brief One (N, size) case of the quant packer on a TIGHT source.
template <size_t N>
void quantCase(const Size& s, size_t& bad, size_t& dirty) {
    const auto frame = randomFrame<uint8_t>(s.w, s.h, 0x9911u + s.h + N);
    bc::DeviceImage<uint8_t> dImg(static_cast<int>(s.w), static_cast<int>(s.h));
    const cudaError_t upImg =
        bc::uploadImage<uint8_t>(frame.data(), s.w, s.h, s.w, dImg.view());
BINCV_CHECK_EQ(upImg, cudaSuccess);
    quantCompare<N>(frame, s.w, s.h, dImg.constView(), bad, dirty);
}

/// @brief The same on a 16-byte-strided source, which the wide lane admits at
/// every width.
template <size_t N>
void quantCasePadded(const Size& s, size_t& bad, size_t& dirty) {
    const auto frame = randomFrame<uint8_t>(s.w, s.h, 0x7A11u + s.w + N);
    const PaddedSource src(frame, s.w, s.h);
    quantCompare<N>(frame, s.w, s.h, src.view(), bad, dirty);
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

BINCV_TEST(CudaPackArms, EveryQuantArmMatchesTheHostOnSixteenByteStrides) {
    // Every depth, odd ones included: the wide lane stores two planes per
    // instruction, so an odd N ends on a store with only its even lanes live.
    size_t bad = 0, dirty = 0;
    for (const Size& s : kSizes) {
        quantCasePadded<1>(s, bad, dirty);
        quantCasePadded<2>(s, bad, dirty);
        quantCasePadded<3>(s, bad, dirty);
        quantCasePadded<4>(s, bad, dirty);
        quantCasePadded<5>(s, bad, dirty);
        quantCasePadded<6>(s, bad, dirty);
        quantCasePadded<7>(s, bad, dirty);
        quantCasePadded<8>(s, bad, dirty);
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
}

BINCV_TEST(CudaPackArms, TheWideLaneScaleIsTheHostsAtEveryValueAndDepth) {
    // The wide lane does not call impl::quantScale: it evaluates the same
    // expression, `(v * maxValue + 127) / 255`, for four bytes at once in two
    // 16-bit fields. That is a second spelling of the host's one definition,
    // so it is pinned here over its WHOLE domain rather than argued: every
    // byte value, in every byte position of a 32-bit load (the two fields
    // treat positions 0/2 and 1/3 differently), at every depth 1..8.
    const size_t w = 1024, h = 8;
    std::vector<uint8_t> frame(w * h);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) frame[y * w + x] = static_cast<uint8_t>(x + 5 * y);
    bc::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    const cudaError_t upImg = bc::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view());
    BINCV_CHECK_EQ(upImg, cudaSuccess);
    BINCV_CHECK(bc::impl::packWideLaneApplies(dImg.getStride(), dImg.constView().ptr, 1));
    size_t bad = 0, dirty = 0;
    quantCompare<1>(frame, w, h, dImg.constView(), bad, dirty);
    quantCompare<2>(frame, w, h, dImg.constView(), bad, dirty);
    quantCompare<3>(frame, w, h, dImg.constView(), bad, dirty);
    quantCompare<4>(frame, w, h, dImg.constView(), bad, dirty);
    quantCompare<5>(frame, w, h, dImg.constView(), bad, dirty);
    quantCompare<6>(frame, w, h, dImg.constView(), bad, dirty);
    quantCompare<7>(frame, w, h, dImg.constView(), bad, dirty);
    quantCompare<8>(frame, w, h, dImg.constView(), bad, dirty);
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
    // host header pins, on a frame holding every byte -- run once per arm.
    // What an arm could break here and nowhere else is the BOUNDARY:
    // `p >= cutoff` is `p > thresh`, and a compare that folded one off would
    // move exactly the pixels equal to `thresh`.
    const size_t w = 129, h = 11;  // one pixel past a word
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
    // The reference geometry, the launch the published threshold rows time:
    // 480 rows are sixty row bands and 24 words three ballot units exactly, so
    // any band or unit arithmetic that was off by one would land in the next
    // row -- which a small frame may not reveal.
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

BINCV_TEST(CudaPackArms, EverySwitchDefaultsOnAndRestores) {
    // A mis-attached switch that never selects anything is the failure this
    // project has already had once, and a default that quietly flipped would
    // make every number in the benchmark a measurement of the wrong arm.
    BINCV_CHECK(bc::impl::packRowGridEnabled());
    BINCV_CHECK(bc::impl::packQuantWideLaneEnabled());
    {
        const ArmScope sel(kArms[0]);
        BINCV_CHECK(!bc::impl::packRowGridEnabled());
        BINCV_CHECK(!bc::impl::packQuantWideLaneEnabled());
    }
    BINCV_CHECK(bc::impl::packRowGridEnabled());
    BINCV_CHECK(bc::impl::packQuantWideLaneEnabled());
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
