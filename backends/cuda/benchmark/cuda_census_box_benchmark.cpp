// The packed census matcher's two arms, priced against each other in ONE
// binary, on ONE explicit stream.
//
// WHAT THIS FILE IS FOR. `denseDisparityCensusPacked` now has two kernels
// behind it: the shipped per-pixel sliding kernel and the warp-cooperative
// separable-box arm. The project rule is that a fast arm must be switchable
// off and the benchmark must SHOW it is on, so every number here comes from
// `timeKernelPaired` through `impl::densePackedBoxEnabled()`, with both arms
// bracketed inside every round and the round order alternating.
//
// THE THREE THINGS A READER SHOULD CHECK, in order:
//
//   1. THE MAPS AGREE. The ratio below compares nothing unless the two arms
//      produce the same bytes, so the maps are downloaded and compared before
//      anything is timed. (The test suite is what PROVES this across shapes;
//      the check here is the benchmark refusing to quote a ratio between two
//      different answers.)
//   2. THE GATE-EXCLUDED CASE READS ~1.00x. winWidth 19 is outside the box
//      arm's own gate -- `33 - winWidth` lanes of a warp produce output, so at
//      19 fewer than half of them would -- and both switch positions therefore
//      run the same kernel. If that line is not ~1.00x, the switch is not
//      selecting what the lines above it claim.
//   3. MEMORY AND SPEED TOGETHER. Both meters are printed here rather than
//      deferred to another binary: this arm's claim is that it adds ZERO bytes,
//      and a claim nobody's benchmark prints is not reproducible.
//
// THE FRAME IS REAL where one is available. Point BINCV_CUDA_DENSE_FRAMES at a
// sequence blob (scripts/make_sequence_blob.py --mode 8bit) and frame 0 of it
// is the left image, with the right image a shift of it -- the same synthetic
// disparity the rest of the dense benchmarks use, on real texture. Without a
// blob the frame is smoothed noise and every line says so. This matcher has no
// data-dependent branch, so content moves the timing very little; the label is
// still printed, because a reader should not have to assume that.
//
// NO STRUCTURAL ADVANTAGE IS CLAIMED HERE. Census EXPANDS data -- 8 bits per
// pixel in, 24 out -- and this kernel reads a conventional one-word-per-pixel
// descriptor, not binCV's bit-planes. The arm is a standard separable-box
// technique, measured for completeness on the entry a wide-input caller meets
// first.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bincv/cuda/census.hpp"
#include "bincv/cuda/denseCensusBox.hpp"
#include "bincv/cuda/denseDisparity.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/io/sequence.hpp"
#include "cuda_bench_util.hpp"

namespace {

namespace bc = bincv::cuda;

constexpr size_t kW = 752, kH = 480;  // the reference frame
constexpr int kShift = 21;
constexpr size_t kK = 24;  // census 5x5

std::vector<uint8_t> smoothFrame(size_t w, size_t h) {
    std::vector<uint8_t> img(w * h);
    uint64_t st = 0xFEEDFACEULL;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<uint8_t>(st >> 40);
    }
    std::vector<uint8_t> tmp(w * h);
    for (size_t y = 1; y + 1 < h; ++y)
        for (size_t x = 1; x + 1 < w; ++x)
            tmp[y * w + x] = static_cast<uint8_t>(
                (img[y * w + x - 1] + 2u * img[y * w + x] + img[y * w + x + 1]) / 4u);
    for (size_t y = 1; y + 1 < h; ++y)
        for (size_t x = 1; x + 1 < w; ++x)
            img[y * w + x] = static_cast<uint8_t>(
                (tmp[(y - 1) * w + x] + 2u * tmp[y * w + x] + tmp[(y + 1) * w + x]) / 4u);
    return img;
}

/// @brief Frame 0 of the blob BINCV_CUDA_DENSE_FRAMES names, or smoothed noise.
std::vector<uint8_t> leftFrame(const char** source) {
    *source = "synthetic smoothed noise (set BINCV_CUDA_DENSE_FRAMES for real frames)";
    const char* path = std::getenv("BINCV_CUDA_DENSE_FRAMES");
    if (path != nullptr) {
        std::FILE* fh = std::fopen(path, "rb");
        if (fh != nullptr) {
            std::fseek(fh, 0, SEEK_END);
            const long len = std::ftell(fh);
            std::fseek(fh, 0, SEEK_SET);
            std::vector<uint8_t> blob;
            if (len > 0) {
                blob.resize(static_cast<size_t>(len));
                if (std::fread(blob.data(), 1, blob.size(), fh) != blob.size())
                    blob.clear();
            }
            std::fclose(fh);
            if (!blob.empty()) {
                const bincv::SequenceHeader h =
                    bincv::readSequenceHeader(blob.data(), blob.size());
                const bincv::SequenceFrameRange f0 =
                    bincv::sequenceFrame(h, blob.data(), blob.size(), 0);
                if (h.valid && h.mode == bincv::kSequenceMode8Bit && h.width == kW &&
                    h.height == kH && f0.valid) {
                    *source = "REAL SEQUENCE FRAME 0 from BINCV_CUDA_DENSE_FRAMES";
                    return std::vector<uint8_t>(f0.data, f0.data + kW * kH);
                }
            }
        }
    }
    return smoothFrame(kW, kH);
}

} // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    std::printf("=== packed census matcher: warp-box arm vs the shipped per-pixel arm"
                " ===\n");
    cudabench::printDevice();

    // ONE EXPLICIT STREAM for everything timed. Events recorded on a stream the
    // work does not use time the wrong thing (cuda_bench_util.hpp), and the
    // legacy default stream additionally carries cross-blocking semantics that
    // have produced 32-second outliers on this host.
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::printf("SKIP: could not create a stream\n");
        return 77;
    }

    const char* source = nullptr;
    const auto lw = leftFrame(&source);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kShift < kW; ++x) rw[y * kW + x] = lw[y * kW + x + kShift];
    std::printf(" frame: %zux%zu, %s\n", kW, kH, source);
    std::printf(" right image is the left shifted by %d px -- a synthetic disparity on"
                " whatever texture the left frame has\n\n",
                kShift);

    const auto floor = cudabench::measureLaunchFloor(dim3(1), dim3(32), 100, 25, 250.0,
                                                     stream);
    cudabench::printLaunchFloor(floor);
    const size_t meterStep = cudabench::measureDriverMeterStep();
    std::printf("\n");

    bincv::DenseDisparityParams p;
    p.maxDisparity = 64;

    bc::DeviceImage<uint8_t> dLw(static_cast<int>(kW), static_cast<int>(kH));
    bc::DeviceImage<uint8_t> dRw(static_cast<int>(kW), static_cast<int>(kH));
    bc::DeviceImage<uint32_t> descL(static_cast<int>(kW), static_cast<int>(kH));
    bc::DeviceImage<uint32_t> descR(static_cast<int>(kW), static_cast<int>(kH));
    bc::DeviceImage<uint8_t> dDisp(static_cast<int>(kW), static_cast<int>(kH));
    bc::uploadImage<uint8_t>(lw.data(), kW, kH, kW, dLw.view(), stream);
    bc::uploadImage<uint8_t>(rw.data(), kW, kH, kW, dRw.view(), stream);
    bc::censusTransformPacked<kK>(dLw.constView(), bincv::kCensus5x5, descL.view(),
                                  stream);
    bc::censusTransformPacked<kK>(dRw.constView(), bincv::kCensus5x5, descR.view(),
                                  stream);
    cudaStreamSynchronize(stream);

    // ------------------------------------------------------------------
    // MEMORY, both meters, named -- the second half of "report memory and
    // speed together". This arm allocates nothing, so the two numbers below
    // are identical with the box arm on and off; that is the claim, and it is
    // printed rather than asserted in prose.
    // ------------------------------------------------------------------
    cudabench::printMemoryHeader("the census entry's working set");
    const size_t workingSet = kW * kH * (2 * sizeof(uint8_t) + 8 + 1);
    cudabench::printAllocSum("two wide frames + two descriptors + map", workingSet);
    cudabench::printAllocSum("global scratch the warp-box arm adds", 0);
    cudabench::printAllocSum("the cost volume this design REFUSES", kW * kH * 65);
    {
        cudabench::DeviceMemMeter meter;
        bc::DeviceImage<uint8_t> w1(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint8_t> w2(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint32_t> d1(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint32_t> d2(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint8_t> m1(static_cast<int>(kW), static_cast<int>(kH));
        cudabench::printDriverDelta("the same arrays, allocated again",
                                    meter.deltaBytes(), meterStep);
    }
    std::printf("   shared memory per block, warp-box arm: 0 B (launched with 0);"
                " caller scratch: none.\n\n");

    // ------------------------------------------------------------------
    // HAZARD 4 OF THE PROTOCOL: what is compared must AGREE before it is timed.
    // ------------------------------------------------------------------
    const auto mapOf = [&](const bincv::DenseDisparityParams& q, bool box) {
        bc::impl::densePackedBoxEnabled() = box;
        bc::denseDisparityCensusPacked(descL.constView(), descR.constView(), q,
                                       dDisp.view(), stream);
        std::vector<uint8_t> out(kW * kH, 0x55);
        bc::downloadImage<uint8_t>(dDisp.constView(), out.data(), kW, stream);
        cudaStreamSynchronize(stream);
        return out;
    };
    const bool agree = mapOf(p, true) == mapOf(p, false);
    std::printf(" the two arms' maps at 9x9 are %s\n\n",
                agree ? "IDENTICAL (byte for byte)"
                      : "DIFFERENT -- every ratio below compares nothing");

    // ------------------------------------------------------------------
    // THE NUMBER THE DECISION RULE NAMES: kernel-resident time of the matcher.
    // ------------------------------------------------------------------
    std::printf("--- the matcher alone, 752x480, D=0..64, 9x9, census 5x5 (K=24) ---\n");
    const auto paired = cudabench::timeKernelPaired(
        [&] {
            bc::impl::densePackedBoxEnabled() = true;
            bc::denseDisparityCensusPacked(descL.constView(), descR.constView(), p,
                                           dDisp.view(), stream);
        },
        [&] {
            bc::impl::densePackedBoxEnabled() = false;
            bc::denseDisparityCensusPacked(descL.constView(), descR.constView(), p,
                                           dDisp.view(), stream);
        },
        20, 8, 9, stream);
    bc::impl::densePackedBoxEnabled() = true;
    cudabench::printArmVsFloor("  A: warp-box arm (switch ON)", paired.a, floor,
                               "kernel");
    cudabench::printArmVsFloor("  B: per-pixel sliding arm (switch OFF)", paired.b,
                               floor, "kernel");
    std::printf("   ratio B/A, interleaved rounds: median %5.2fx <- QUOTE THIS"
                "   geomean %5.2fx   per-round range %5.2f-%5.2fx (%d rounds)\n",
                paired.ratioMedian, paired.ratioGeoMean, paired.ratioMin, paired.ratioMax,
                paired.rounds);
    cudabench::printPairedSignAndSeparation(paired);
    cudabench::printPairedVerdict(paired);
    std::printf("   the bar (the owner's, written before any measurement of this arm)"
                " is 1.50x.\n"
                "   ONE RUN IS NOT A NUMBER: the figure that ships is the median of at\n"
                "   least seven independent runs of this binary.\n\n");

    // ------------------------------------------------------------------
    // THE GATE-EXCLUDED CASE. winWidth 19 leaves the box arm's own gate
    // (33 - winWidth output lanes per warp), so both switch positions run the
    // same kernel and the line MUST read ~1.00x. The in-gate control at 17
    // differs in exactly one variable and must not.
    // ------------------------------------------------------------------
    std::printf("--- the gate's own exclusion: winWidth 19 (outside) vs 17 (inside) ---\n");
    {
        bincv::DenseDisparityParams pOut = p;
        pOut.winWidth = 19;
        bincv::DenseDisparityParams pIn = p;
        pIn.winWidth = 17;
        const bool agree17 = mapOf(pIn, true) == mapOf(pIn, false);
        bc::impl::densePackedBoxEnabled() = true;

        const auto outside = cudabench::timeKernelPaired(
            [&] {
                bc::impl::densePackedBoxEnabled() = true;
                bc::denseDisparityCensusPacked(descL.constView(), descR.constView(),
                                               pOut, dDisp.view(), stream);
            },
            [&] {
                bc::impl::densePackedBoxEnabled() = false;
                bc::denseDisparityCensusPacked(descL.constView(), descR.constView(),
                                               pOut, dDisp.view(), stream);
            },
            4, 4, 7, stream);
        bc::impl::densePackedBoxEnabled() = true;
        std::printf(" winWidth 19 -- OUTSIDE the box arm's gate (winWidth <= 17):\n");
        cudabench::printPaired("  A: switch ON  (the gate rejects it anyway)",
                               "  B: switch OFF", outside, "kernel", /*expect1x=*/true);

        const auto inside = cudabench::timeKernelPaired(
            [&] {
                bc::impl::densePackedBoxEnabled() = true;
                bc::denseDisparityCensusPacked(descL.constView(), descR.constView(), pIn,
                                               dDisp.view(), stream);
            },
            [&] {
                bc::impl::densePackedBoxEnabled() = false;
                bc::denseDisparityCensusPacked(descL.constView(), descR.constView(), pIn,
                                               dDisp.view(), stream);
            },
            8, 4, 7, stream);
        bc::impl::densePackedBoxEnabled() = true;
        std::printf(" winWidth 17 -- INSIDE it, the positive control. The two arms'"
                    " maps are %s.\n",
                    agree17 ? "IDENTICAL"
                            : "DIFFERENT, so the ratio below compares nothing");
        cudabench::printPaired("  A: switch ON  (warp-box arm runs)",
                               "  B: switch OFF (per-pixel sliding arm)", inside,
                               "kernel");
        std::printf("   Read the two together: the 19 line at ~1.00x says the switch is\n"
                    "   real and the gate excludes what it claims; the 17 line away from\n"
                    "   1.00x says the arm above it is the one being timed.\n");
    }

    cudaStreamDestroy(stream);
    std::printf("\n role note: this path has NO structural advantage for binCV. Census\n"
                " expands data and this kernel reads a conventional descriptor word, not\n"
                " a bit-plane. The bar it answers to is binCV's own shipped kernel.\n");
    return 0;
}
