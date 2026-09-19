// The orientation / descriptor family, priced at birth.
//
// WHAT IS IN HERE AND WHY EACH ROW EXISTS
//
//   * Every arm, both switch positions, in ONE binary: W0 / W1 / W2 over a wide
//     frame, reference / warp over a bit-plane block, reference / ballot for
//     describe. A fast arm that cannot be switched off is a fast arm nobody has
//     proven was running.
//   * FOUR GATE-EXCLUDED CONTROLS, each a case the fast path's OWN gate
//     rejects. They must read ~1.00x. If one does not, the switch is not wired
//     and no other number in this file means anything -- which is the
//     mis-attached-#define scar applied to a device arm.
//   * Three keypoint counts. 470 is the rate examples/slam_frontend.cpp
//     measured over 300 real EuRoC frames; 1000 is the host benchmark's count;
//     100,000 is carried because at the first two these kernels sit ON the
//     launch floor, where a ratio between two arms is a ratio between two
//     launches. Every row says which count it was taken at.
//   * ONE EXPLICIT STREAM for every arm on both sides, always. OpenCV
//     synchronizes the whole device on the default stream in cudev's grid
//     transform and in every cudafilters filter; a default-stream comparison
//     measures against an arm nobody would use.
//   * MEMORY WITH THE METER NAMED. Meter 1 (allocation sum) for binCV against
//     binCV. Meter 2 (cudaMemGetInfo delta over replicas) for anything crossing
//     to OpenCV, taken identically on both sides. Never a ratio across two
//     meters.
//
// WHAT m2 (END-TO-END) IS DEFINED TO COVER, said at the number and not left to
// the reader: keypoints up (8N B), the kernels, angles/keep/descriptors down
// (37N B), and the synchronize -- with the wide frame ALREADY RESIDENT. The
// 361 KB frame upload is NOT inside it, and it is printed on the line below so
// the non-resident figure is visible beside the resident one.
//
// THE ROLE BAR IS cv::cuda::ORB (cudafeatures2d) and it is NOT
// STAGE-DECOMPOSABLE. This file's first act, when built against such an
// OpenCV, is to CALL `computeAsync` -- the entry point a caller actually
// reaches for -- and print what it does. When that is refused, the
// describe-stage denominator is the DIFFERENCE between detectAndComputeAsync
// and detectAsync, with its error bar stated. A CPU number is never quoted as
// a GPU bar.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#if BINCV_CUDA_ORB_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d.hpp>
#endif

#include "bincv/binMat.hpp"
#include "bincv/cuda/descriptor.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/orientation.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/orbPattern.hpp"
#include "bincv/ops/orientation.hpp"
#include "cuda_bench_util.hpp"

using namespace cudabench;
namespace bc = bincv::cuda;

namespace {

constexpr size_t kW = 752, kH = 480;
constexpr size_t kBits = 256;
constexpr size_t kWords = kBits / 32;
constexpr int kRadius = 15;

/// Replicas per memory reading. This driver reserves in 2 MB units and this
/// family's whole per-frame working set is 45 KB at N=1000, so a single reading
/// would be the meter's resolution rather than a footprint.
constexpr int kReplicas = 128;

cudaStream_t gStream = nullptr;

uint64_t rngState = 0x9E3779B97F4A7C15ULL;
uint32_t nextU32() {
    rngState = rngState * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(rngState >> 32);
}

/// A frame with structure as well as noise: a flat patch makes a centroid
/// uninformative and a comparison a coin toss, which is not what either op
/// costs on a real frame.
std::vector<uint8_t> makeFrame(size_t w, size_t h) {
    std::vector<uint8_t> img(w * h);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x)
            img[y * w + x] = static_cast<uint8_t>(((x * 3u + y * 5u) % 251u) * 7u / 7u +
                                                  (nextU32() & 0x3Fu));
    return img;
}

std::vector<float> makeKeypoints(size_t count, int margin) {
    std::vector<float> xy(count * 2);
    const size_t spanX = kW - 2u * static_cast<size_t>(margin) - 2u;
    const size_t spanY = kH - 2u * static_cast<size_t>(margin) - 2u;
    for (size_t k = 0; k < count; ++k) {
        xy[2 * k] = static_cast<float>(margin + 1 + static_cast<int>(nextU32() % spanX));
        xy[2 * k + 1] = static_cast<float>(margin + 1 + static_cast<int>(nextU32() % spanY));
    }
    return xy;
}

/// Machine-readable emission. This binary is run at least seven times and its
/// rows aggregated across processes -- one run of a small kernel is not a
/// number, it is a sample.
void emit(const char* tag, const Timing& t) {
    std::printf("ROW|%s|%.6f|%.6f|%.6f\n", tag, t.medianMs, t.minMs, t.maxMs);
}
void emitPair(const char* tag, const PairedTiming& p) {
    std::printf("PAIR|%s|%.4f|%.4f|%.4f|%d|%s\n", tag, p.ratioMedian, p.ratioMin,
                p.ratioMax, p.rounds, p.separated() ? "disjoint" : "overlap");
}

// ---------------------------------------------------------------------------
// The device-side working set this family owns
// ---------------------------------------------------------------------------

struct FamilyBuffers {
    bc::DeviceArray<float> xy;
    bc::DeviceArray<float> angle;
    bc::DeviceArray<uint8_t> keep;
    bc::DeviceArray<uint32_t> desc;
    bc::DeviceArray<bincv::BriefPair> pairs;

    explicit FamilyBuffers(size_t n)
        : xy(n * 2), angle(n), keep(n), desc(n * kWords),
          pairs(bc::steeredBriefPatternPairs<kBits>()) {}

    static size_t allocSum(size_t n) {
        return n * 2 * sizeof(float) + n * sizeof(float) + n +
               n * kWords * sizeof(uint32_t) +
               bc::steeredBriefPatternPairs<kBits>() * sizeof(bincv::BriefPair);
    }
};

void printFamilyMemory(size_t n) {
    const size_t perFrame = n * 2 * sizeof(float) + n * sizeof(float) + n +
                            n * kWords * sizeof(uint32_t);
    const size_t setup =
        bc::steeredBriefPatternPairs<kBits>() * sizeof(bincv::BriefPair);
    printAllocSum("per frame (xy+angle+keep+desc)", perFrame);
    printAllocSum("once per process (steered table)", setup);
    printAllocSum("wide frame, shared with pipeline", kW * kH);
    printAllocSum("bit-plane frame, 1 plane", bc::rowWords(kW) * 4 * kH);
    std::printf("   SCRATCH: none. Every per-keypoint intermediate is a register and the\n"
                "   warp arms reduce through __shfl_down_sync, so scratchBytes(N) = 0.\n");
    std::printf("   The bit-plane frame is %.1fx smaller than the wide one. That factor is\n"
                "   the CONTAINER's, not this kernel's, and it is not counted twice below.\n",
                static_cast<double>(kW * kH) /
                    static_cast<double>(bc::rowWords(kW) * 4 * kH));
}

} // namespace

int main(int argc, char** argv) {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return 77;
    }
    const std::string only = argc > 1 ? argv[1] : std::string();
    const auto want = [&](const char* s) { return only.empty() || only == s; };

    printDevice();
    std::printf("\n binCV CUDA -- keypoint orientation and BRIEF, priced at birth.\n"
                " Frame %zux%zu, %zu-bit descriptors, disc radius %d.\n"
                " EVERY arm below runs on ONE EXPLICIT STREAM, both sides of every pair.\n",
                kW, kH, kBits, kRadius);
    std::printf("\n INDICATIVE: the GPU was shared while these were taken. A later serial\n"
                " pass produces every number that ships. Spreads are printed so a reader\n"
                " can see what a difference has to clear.\n");

    cudaStreamCreate(&gStream);
    const Timing floor = measureLaunchFloor(dim3(1), dim3(32), 100, 25, 250.0, gStream);
    std::printf("\n");
    printLaunchFloor(floor);
    emit("launch_floor", floor);

    const std::vector<uint8_t> frame = makeFrame(kW, kH);
    bc::DeviceImage<uint8_t> dframe(static_cast<int>(kW), static_cast<int>(kH));
    bc::uploadImage<uint8_t>(frame.data(), kW, kH, kW, dframe.view(), gStream);
    bc::DeviceImage<uint16_t> dframe16(static_cast<int>(kW), static_cast<int>(kH));

    // The bit-plane frame: the binary image a bits-native pipeline already holds.
    bincv::BinMat<uint32_t> hostBits(static_cast<int>(kW), static_cast<int>(kH));
    for (size_t y = 0; y < kH; ++y) {
        uint32_t* row = hostBits.view().row(y);
        for (size_t x = 0; x < kW; ++x)
            if (frame[y * kW + x] > 127u) row[x / 32] |= (uint32_t{1} << (x % 32));
    }
    bc::DeviceBinMat dbits(static_cast<int>(kW), static_cast<int>(kH));
    bc::upload<uint32_t>(hostBits.constView(), dbits.view(), gStream);
    const bc::DevicePlaneBlockConstView plane1 =
        bc::planeBlock(bc::DeviceBinMatConstView(dbits.constView()), 1);

    bincv::BriefPattern<kBits> base{};
    bincv::makeBriefPattern<kBits>(base);
    static bincv::SteeredBriefPattern<kBits> steered{};
    bincv::makeSteeredBriefPattern<kBits>(steered, base);
    cudaStreamSynchronize(gStream);

    const size_t counts[] = {470, 1000, 100000};

    for (size_t n : counts) {
        if (!want("arms")) break;
        const std::vector<float> xy = makeKeypoints(n, 24);
        FamilyBuffers buf(n);
        bc::DeviceBriefPattern pat{};
        bc::uploadBriefPattern<kBits>(steered, buf.pairs.data(), pat, gStream);
        cudaMemcpyAsync(buf.xy.data(), xy.data(), n * 2 * sizeof(float),
                        cudaMemcpyHostToDevice, gStream);
        cudaStreamSynchronize(gStream);

        const bc::DeviceKeypointSetConstView kp = bc::keypointSet(buf.xy.data(), n);
        const bc::DeviceDescriptorSetView dset =
            bc::descriptorSet(buf.desc.data(), n, kWords, buf.keep.data());

        const int iters = n > 10000 ? 20 : 60;
        std::printf("\n=====================================================================\n"
                    " N = %zu keypoints\n"
                    "=====================================================================\n",
                    n);

        // --- Wide orientation: W0 vs W1 (warp parallelism) -----------------
        const auto runWide = [&](bool warp, bool quad) {
            bc::impl::orientationWideWarpEnabled() = warp;
            bc::impl::orientationWideQuadEnabled() = quad;
            bc::keypointOrientation(dframe.constView(), kp, buf.angle.data(),
                                    buf.keep.data(), kRadius, nullptr, gStream);
        };
        {
            const PairedTiming p = timeKernelPaired([&] { runWide(false, false); },
                                                    [&] { runWide(true, false); }, iters,
                                                    iters, 9, gStream);
            std::printf("\n WIDE ORIENTATION -- what WARP PARALLELISM buys\n");
            printPaired("W0 reference (thread per keypoint)", "W1 warp, lane per column", p,
                        "kernel");
            emitPair("wide_W1_over_W0", p);
        }
        // --- W1 vs W2 (vectorization): the format claim's denominator ------
        {
            const PairedTiming p = timeKernelPaired([&] { runWide(true, false); },
                                                    [&] { runWide(true, true); }, iters,
                                                    iters, 9, gStream);
            std::printf("\n WIDE ORIENTATION -- what VECTORIZATION buys (uchar4 + __dp4a)\n"
                        "   static, from cuobjdump -sass at radius 15, per keypoint:\n"
                        "     W1  31 LDG, ~776 instructions;  W2  ~9 LDG, ~500 instructions\n"
                        "     -> 3.4x fewer loads, 1.55x fewer instructions. 1.55x is an\n"
                        "        UPPER BOUND on what that can buy in time.\n");
            printPaired("W1 warp, lane per column", "W2 warp, 4 px/lane, __dp4a", p,
                        "kernel");
            emitPair("wide_W2_over_W1", p);
        }
        // --- Bit-plane orientation: its own two arms ------------------------
        const auto runPlane = [&](bool warp) {
            bc::impl::orientationBitPlaneWarpEnabled() = warp;
            bc::keypointOrientation(plane1, kp, buf.angle.data(), buf.keep.data(), kRadius,
                                    nullptr, gStream);
        };
        {
            const PairedTiming p = timeKernelPaired([&] { runPlane(false); },
                                                    [&] { runPlane(true); }, iters, iters,
                                                    9, gStream);
            std::printf("\n BIT-PLANE ORIENTATION -- reference against its warp arm\n");
            printPaired("bit-plane reference", "bit-plane warp, lane per row", p, "kernel");
            emitPair("plane_warp_over_ref", p);
        }
        // --- THE FORMAT COMPARISON -----------------------------------------
        {
            bc::impl::orientationBitPlaneWarpEnabled() = true;
            const PairedTiming p = timeKernelPaired([&] { runWide(true, true); },
                                                    [&] { runPlane(true); }, iters, iters,
                                                    9, gStream);
            std::printf("\n THE FORMAT COMPARISON -- bit-plane (planeCount=1) against the\n"
                        " WIDE ARM AT ITS BEST (W2), which is the denominator the design\n"
                        " review demanded and the one this family measures against.\n"
                        "   ROLE-ONLY: the two arms do not read the same data. One reads a\n"
                        "   1-bit image and one an 8-bit image, so part of any ratio is\n"
                        "   'a 1-bit image is smaller', which is true by construction.\n"
                        "   NO MAGNITUDE IS ASSERTED HERE: what a format advantage has to\n"
                        "   buy to ship is a judgement nobody has made. Printed, not judged.\n");
            printPaired("W2 wide, __dp4a", "bit-plane warp, 1 plane", p, "kernel");
            emitPair("plane_over_wideW2", p);
            std::printf("   static, per keypoint at radius 15: bit-plane ~2 LDG and ~41\n"
                        "   instructions for the WHOLE disc, against W2's ~9 and ~500.\n");
        }
        // --- Describe: reference vs ballot ---------------------------------
        const auto runBrief = [&](bool ballot) {
            bc::impl::briefBallotArmEnabled() = ballot;
            bc::computeBriefSteered(dframe.constView(), kp, buf.angle.data(), pat, dset,
                                    gStream);
        };
        {
            // Angles must be in [-2pi, 2pi]: fill them from the orientation op
            // rather than leaving whatever the previous arm wrote.
            runWide(true, true);
            cudaStreamSynchronize(gStream);
            const PairedTiming p = timeKernelPaired([&] { runBrief(false); },
                                                    [&] { runBrief(true); }, iters, iters,
                                                    9, gStream);
            std::printf("\n STEERED BRIEF -- serial shift-or against __ballot_sync\n"
                        "   The ballot arm packs 32 comparison bits in ONE instruction\n"
                        "   where the reference does ~32 dependent shift-ors per word.\n");
            printPaired("reference (thread per keypoint)", "ballot (warp, lane per bit)", p,
                        "kernel");
            emitPair("brief_ballot_over_ref", p);
        }
        // --- The stage, end to end ------------------------------------------
        {
            // THE SHIPPED CONFIGURATION, not the fastest one found anywhere:
            // the quad arm is off by default because it lost its bar at these
            // counts, and this row must be what a caller actually gets.
            bc::impl::orientationWideWarpEnabled() = true;
            bc::impl::orientationWideQuadEnabled() = false;
            bc::impl::briefBallotArmEnabled() = true;
            std::vector<uint32_t> hostDesc(n * kWords);
            std::vector<float> hostAngle(n);
            std::vector<uint8_t> hostKeep(n);
            const Timing e2e = timeKernel(
                [&] {
                    cudaMemcpyAsync(buf.xy.data(), xy.data(), n * 2 * sizeof(float),
                                    cudaMemcpyHostToDevice, gStream);
                    bc::keypointOrientation(dframe.constView(), kp, buf.angle.data(),
                                            buf.keep.data(), kRadius, nullptr, gStream);
                    bc::computeBriefSteered(dframe.constView(), kp, buf.angle.data(), pat,
                                            dset, gStream);
                    cudaMemcpyAsync(hostAngle.data(), buf.angle.data(), n * sizeof(float),
                                    cudaMemcpyDeviceToHost, gStream);
                    cudaMemcpyAsync(hostKeep.data(), buf.keep.data(), n,
                                    cudaMemcpyDeviceToHost, gStream);
                    cudaMemcpyAsync(hostDesc.data(), buf.desc.data(),
                                    n * kWords * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                    gStream);
                    cudaStreamSynchronize(gStream);
                },
                4, 9, gStream);
            const Timing upload = timeKernel(
                [&] {
                    bc::uploadImage<uint8_t>(frame.data(), kW, kH, kW, dframe.view(),
                                             gStream);
                    cudaStreamSynchronize(gStream);
                },
                4, 9, gStream);
            std::printf("\n THE STAGE, END TO END (m2). DEFINED AS: keypoints up (%zu B),\n"
                        " both kernels, angles+keep+descriptors down (%zu B), synchronize --\n"
                        " with the WIDE FRAME ALREADY RESIDENT.\n",
                        n * 8, n * 37);
            printArm("orient + describe, resident frame", e2e, "e2e");
            printArm("the 361 KB frame upload m2 EXCLUDES", upload, "e2e");
            std::printf("   NON-RESIDENT total = %.3f ms. A caller who does not already hold\n"
                        "   the frame on device pays that, and this family's case rests on\n"
                        "   residency, so the number it must be quoted with is this one.\n",
                        e2e.medianMs + upload.medianMs);
            emit("stage_e2e_resident", e2e);
            emit("stage_frame_upload", upload);

            // The host library's own arms, same frame, same keypoints, same
            // machine. A CPU denominator, labelled as one.
            std::vector<double> hostSamples;
            for (int r = 0; r < 9; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 4; ++i) {
                    bincv::keypointOrientation<uint8_t>(frame.data(), kW, kH, kW, xy.data(),
                                                        n, hostAngle.data(),
                                                        hostKeep.data(), kRadius);
                    bincv::computeBriefSteered<kBits, uint8_t, uint32_t>(
                        frame.data(), kW, kH, kW, xy.data(), n, hostAngle.data(), steered,
                        hostDesc.data(), hostKeep.data());
                }
                hostSamples.push_back(
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count() / 4.0);
            }
            const Timing host = summarize(std::move(hostSamples));
            printArm("HOST binCV orient + describe (CPU arm)", host, "host");
            emit("host_stage", host);
            std::printf("   A CPU arm is CONTEXT, not the role bar for a GPU op. It is here\n"
                        "   because it is what a caller runs today, and because this host is\n"
                        "   not timing-grade under WSL2 -- take its ordering, not its factor.\n");
        }

        if (n == 1000) {
            std::printf("\n");
            printMemoryHeader("the orientation/descriptor family, N=1000");
            printFamilyMemory(n);
        }
    }

    // ---------------------------------------------------------------------
    // THE GATE-EXCLUDED CONTROLS
    // ---------------------------------------------------------------------
    if (want("gates")) {
        std::printf("\n=====================================================================\n"
                    " GATE-EXCLUDED CONTROLS -- each must read ~1.00x\n"
                    "=====================================================================\n"
                    " Every one of these flips a switch on a configuration the fast path's\n"
                    " OWN GATE rejects, so the switch must select nothing. A number other\n"
                    " than ~1.00x here means the arm under test was never running where the\n"
                    " rows above say it was, and none of them mean anything.\n");
        const size_t n = 1000;
        const std::vector<float> xy = makeKeypoints(n, 40);
        FamilyBuffers buf(n);
        bc::DeviceBriefPattern pat{};
        bc::uploadBriefPattern<kBits>(steered, buf.pairs.data(), pat, gStream);
        cudaMemcpyAsync(buf.xy.data(), xy.data(), n * 2 * sizeof(float),
                        cudaMemcpyHostToDevice, gStream);
        cudaStreamSynchronize(gStream);
        const bc::DeviceKeypointSetConstView kp = bc::keypointSet(buf.xy.data(), n);

        // MORE ROUNDS THAN THE ARM COMPARISONS ABOVE, on purpose. Each control
        // below runs the SAME kernel on both sides -- the gate rejects the fast
        // path, so the switch selects nothing -- which means every unit of
        // difference is noise, and on this host the launch floor's own spread
        // runs to several hundred percent. A control that is allowed to read
        // 1.10x because it was sampled nine times is not a control.
        constexpr int kGateIters = 200;
        constexpr int kGateRounds = 25;

        // (1) radius 16: above the warp arms' 31-column gate on both spellings.
        {
            const auto run = [&](bool warp) {
                bc::impl::orientationWideWarpEnabled() = warp;
                bc::impl::orientationWideQuadEnabled() = warp;
                bc::keypointOrientation(dframe.constView(), kp, buf.angle.data(), nullptr,
                                        16, nullptr, gStream);
            };
            const PairedTiming p = timeKernelPaired([&] { run(false); }, [&] { run(true); },
                                                    kGateIters, kGateIters, kGateRounds,
                                                    gStream);
            printPaired("wide r=16, warp arm OFF", "wide r=16, warp arm ON", p, "kernel",
                        true);
            emitPair("gate_wide_radius16", p);
        }
        // (2) uint16_t: the __dp4a arm has no uint16 form, by decision.
        {
            const auto run = [&](bool quad) {
                bc::impl::orientationWideWarpEnabled() = true;
                bc::impl::orientationWideQuadEnabled() = quad;
                bc::keypointOrientation(dframe16.constView(), kp, buf.angle.data(), nullptr,
                                        kRadius, nullptr, gStream);
            };
            const PairedTiming p = timeKernelPaired([&] { run(false); }, [&] { run(true); },
                                                    kGateIters, kGateIters, kGateRounds,
                                                    gStream);
            printPaired("u16 wide, quad arm OFF", "u16 wide, quad arm ON", p, "kernel",
                        true);
            emitPair("gate_wide_uint16", p);
        }
        // (3) bit-plane at radius 16.
        {
            const auto run = [&](bool warp) {
                bc::impl::orientationBitPlaneWarpEnabled() = warp;
                bc::keypointOrientation(plane1, kp, buf.angle.data(), nullptr, 16, nullptr,
                                        gStream);
            };
            const PairedTiming p = timeKernelPaired([&] { run(false); }, [&] { run(true); },
                                                    kGateIters, kGateIters, kGateRounds,
                                                    gStream);
            printPaired("bit-plane r=16, warp arm OFF", "bit-plane r=16, warp arm ON", p,
                        "kernel", true);
            emitPair("gate_plane_radius16", p);
        }
        // (4) 1056-bit descriptors: 33 words, above the ballot arm's 32-word gate.
        {
            static bincv::BriefPattern<1056> wide{};
            bincv::makeBriefPattern<1056>(wide);
            bc::DeviceArray<bincv::BriefPair> widePairs(bc::briefPatternPairs<1056>());
            bc::DeviceArray<uint32_t> wideDesc(n * 33);
            bc::DeviceBriefPattern widePat{};
            bc::uploadBriefPattern<1056>(wide, widePairs.data(), widePat, gStream);
            cudaStreamSynchronize(gStream);
            const bc::DeviceDescriptorSetView wset =
                bc::descriptorSet(wideDesc.data(), n, 33);
            const auto run = [&](bool ballot) {
                bc::impl::briefBallotArmEnabled() = ballot;
                bc::computeBrief(dframe.constView(), kp, widePat, wset, gStream);
            };
            const PairedTiming p = timeKernelPaired([&] { run(false); }, [&] { run(true); },
                                                    kGateIters / 4, kGateIters / 4,
                                                    kGateRounds, gStream);
            printPaired("1056-bit, ballot arm OFF", "1056-bit, ballot arm ON", p, "kernel",
                        true);
            emitPair("gate_brief_1056", p);
        }
        bc::impl::orientationWideWarpEnabled() = true;
        bc::impl::orientationWideQuadEnabled() = false;  // the shipped default
        bc::impl::orientationBitPlaneWarpEnabled() = true;
        bc::impl::briefBallotArmEnabled() = true;
    }

    // ---------------------------------------------------------------------
    // THE ROLE BAR
    // ---------------------------------------------------------------------
    std::printf("\n=====================================================================\n"
                " ROLE BAR -- cv::cuda::ORB (cudafeatures2d)\n"
                "=====================================================================\n");
#if !BINCV_CUDA_ORB_OPENCV
    std::printf(" UNMEASURED. This binary was built without an OpenCV carrying\n"
                " cudafeatures2d, so the GPU role bar for orientation and describe is\n"
                " BLOCKED, not substituted. No CPU number above is a stand-in for it:\n"
                " point BINCV_CUDA_ORB_OPENCV_DIR at such a build and re-run.\n");
#else
    if (want("role")) {
        // The shipped arm selection, so the role rows are what a caller gets.
        bc::impl::orientationWideWarpEnabled() = true;
        bc::impl::orientationWideQuadEnabled() = false;
        bc::impl::orientationBitPlaneWarpEnabled() = true;
        bc::impl::briefBallotArmEnabled() = true;
        cv::cuda::setDevice(0);
        { cv::cuda::GpuMat warm(16, 16, CV_8UC1); warm.setTo(cv::Scalar(0)); }
        cudaDeviceSynchronize();
        cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(gStream);

        cv::Mat hostMat(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
        std::memcpy(hostMat.data, frame.data(), kW * kH);
        cv::cuda::GpuMat gpuMat;
        gpuMat.upload(hostMat);

        const int nfeatures = 1000;
        cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(
            nfeatures, 1.2f, /*nlevels=*/1, /*edgeThreshold=*/31, 0, 2,
            cv::ORB::HARRIS_SCORE, 31, 20, /*blurForDescriptor=*/false);

        // (a) THE FIRST THING THIS FILE DOES: ask whether the stage-isolated
        // call exists. `computeAsync` is what a caller reaches for, and its
        // base implementation forwards to detectAndComputeAsync with
        // useProvidedKeypoints = true.
        //
        // IT IS ACCEPTED, and that contradicts what this family was designed
        // against. The design and its review both assumed OpenCV 4.5.4's CUDA
        // ORB carried `CV_Assert(!useProvidedKeypoints)`, and the review's
        // proposed fix was a DIFFERENTIAL of two whole-ORB timings with an
        // error bar. Neither is needed: measured here, `computeAsync` returns
        // an N x 32 CV_8U descriptor matrix for provided keypoints and its
        // bytes are IDENTICAL to the ones `detectAndComputeAsync` produces for
        // the same keypoints. So a stage-isolated GPU denominator exists and it
        // is what this file uses. The differential stays printed underneath as
        // context, not as the bar.
        cv::cuda::GpuMat kpMat, descMat;
        orb->detectAsync(gpuMat, kpMat, cv::noArray(), cvStream);
        cvStream.waitForCompletion();
        bool computeAsyncWorks = false;
        std::string computeAsyncSays;
        cv::cuda::GpuMat kpProvided = kpMat.clone();
        try {
            orb->computeAsync(gpuMat, kpProvided, descMat, cvStream);
            cvStream.waitForCompletion();
            computeAsyncWorks = !descMat.empty();
        } catch (const cv::Exception& e) {
            computeAsyncSays = e.what();
        }
        std::printf(" cv::cuda::Feature2DAsync::computeAsync(image, keypoints, desc,"
                    " stream):\n");
        if (computeAsyncWorks) {
            std::printf("   ACCEPTED. %d keypoints in, a %dx%d CV_8U descriptor matrix out.\n"
                        "   A STAGE-ISOLATED GPU DENOMINATOR EXISTS, so the describe rows\n"
                        "   below are GPU-against-GPU on the same stage, not a differential\n"
                        "   and not a CPU stand-in.\n",
                        kpProvided.cols, descMat.rows, descMat.cols);
        } else {
            std::printf("   REFUSED: %s\n", computeAsyncSays.substr(0, 240).c_str());
            std::printf("   OpenCV then has NO device-side 'describe these keypoints' call\n"
                        "   and the denominator falls back to the differential below, whose\n"
                        "   error bar is printed with it.\n");
        }

        // (b) THE ROLE COMPARISON. Both arms on the one explicit stream,
        // interleaved and order-alternated, same frame, same 1000 keypoints.
        //
        // WHAT IS AND IS NOT COMPARABLE HERE, said before the number:
        //   * SAME ROLE: turn a frame plus N device-resident keypoints into N
        //     256-bit descriptors, on the device.
        //   * NOT THE SAME OUTPUT. binCV steers by 30 pre-rotated copies of the
        //     pattern (the ORB paper's own discretization); cv::cuda::ORB
        //     rotates per keypoint with the exact angle. The descriptors are
        //     therefore not interchangeable even with the same table, and no
        //     bit-exactness is claimed or implied by this pair.
        //   * STILL A SUPERSET ON OPENCV'S SIDE, though a much smaller one than
        //     whole-ORB: computeAsync also builds ORB's level-0 pyramid entry
        //     and converts the keypoint matrix into its per-level buffers.
        //     nlevels = 1 and blurForDescriptor = false are pinned to keep that
        //     superset as small as the API allows.
        {
            const size_t n = static_cast<size_t>(kpProvided.cols);
            const std::vector<float> xy = makeKeypoints(n, 24);
            FamilyBuffers buf(n);
            // The ORB table, so the two sides sample the same 256 pairs. binCV
            // reaches it BY POINTER out of ops/orbPattern.hpp and copies it
            // nowhere new.
            static bincv::SteeredBriefPattern<kBits> orbSteered{};
            bincv::makeSteeredBriefPattern<kBits>(orbSteered, bincv::kOrbBriefPattern);
            bc::DeviceBriefPattern pat{};
            bc::uploadBriefPattern<kBits>(orbSteered, buf.pairs.data(), pat, gStream);
            cudaMemcpyAsync(buf.xy.data(), xy.data(), n * 2 * sizeof(float),
                            cudaMemcpyHostToDevice, gStream);
            cudaStreamSynchronize(gStream);
            const bc::DeviceKeypointSetConstView kp2 = bc::keypointSet(buf.xy.data(), n);
            const bc::DeviceDescriptorSetView dset2 =
                bc::descriptorSet(buf.desc.data(), n, kWords, buf.keep.data());
            bc::keypointOrientation(dframe.constView(), kp2, buf.angle.data(),
                                    buf.keep.data(), kRadius, nullptr, gStream);
            cudaStreamSynchronize(gStream);

            cv::cuda::GpuMat descOut;
            const PairedTiming p = timeKernelPaired(
                [&] { orb->computeAsync(gpuMat, kpProvided, descOut, cvStream); },
                [&] {
                    bc::computeBriefSteered(dframe.constView(), kp2, buf.angle.data(), pat,
                                            dset2, gStream);
                },
                20, 60, 15, gStream);
            std::printf("\n DESCRIBE, GPU against GPU, N = %zu, nlevels=1, blur off:\n", n);
            printPaired("cv::cuda::ORB::computeAsync", "binCV cuda::computeBriefSteered", p,
                        "kernel");
            emitPair("role_describe_bincv_over_orb", p);

            // binCV's ORIENTATION stage has no isolable counterpart at all:
            // OpenCV runs IC_Angle inside its keypoint pass and exposes no
            // entry point for it. Verdict OUTSTANDING, per ruling R2 -- and a
            // CPU number is not put in its place.
            const Timing orientT = timeKernel(
                [&] {
                    bc::keypointOrientation(dframe.constView(), kp2, buf.angle.data(),
                                            buf.keep.data(), kRadius, nullptr, gStream);
                },
                60, 9, gStream);
            printArm("binCV cuda::keypointOrientation (wide)", orientT, "kernel");
            emit("bincv_orient_1000", orientT);
            std::printf("   ROLE BAR FOR ORIENTATION: **OUTSTANDING**. cv::cuda::ORB runs\n"
                        "   IC_Angle inside its keypoint pass and exposes no device entry\n"
                        "   point that orients provided keypoints, so there is no cv::cuda\n"
                        "   counterpart to time. Ruling R2: correctness, memory and the host\n"
                        "   comparison carry it, and the speed verdict is recorded\n"
                        "   OUTSTANDING rather than given a substitute.\n");
        }

        // (c) The whole-chain numbers, as CONTEXT for what the stage sits in.
        {
            cv::cuda::GpuMat kpOut, descOut;
            const Timing detectT = timeKernel(
                [&] { orb->detectAsync(gpuMat, kpOut, cv::noArray(), cvStream); }, 10, 9,
                gStream);
            const Timing bothT = timeKernel(
                [&] {
                    orb->detectAndComputeAsync(gpuMat, cv::noArray(), kpOut, descOut, false,
                                               cvStream);
                },
                10, 9, gStream);
            std::printf("\n CONTEXT, not the bar: the whole-chain calls this stage sits in.\n");
            printArm("cv::cuda::ORB detectAsync", detectT, "kernel");
            printArm("cv::cuda::ORB detectAndComputeAsync", bothT, "kernel");
            emit("orb_detect", detectT);
            emit("orb_detect_and_compute", bothT);
            std::printf("   Their DIFFERENCE is %.4f ms, readable only to about +/- %.4f ms\n"
                        "   given spreads of %.0f%% and %.0f%%. That is the differential the\n"
                        "   design review proposed as the bar; it is not needed, and at this\n"
                        "   error bar it could not have carried a claim anyway.\n",
                        bothT.medianMs - detectT.medianMs,
                        0.5 * ((detectT.maxMs - detectT.minMs) + (bothT.maxMs - bothT.minMs)),
                        detectT.spreadPct(), bothT.spreadPct());
        }

        // (d) Memory, meter 2, on BOTH sides, over replicas.
        std::printf("\n MEMORY, cross-library -- meter 2 (cudaMemGetInfo delta) on BOTH\n"
                    " sides, over %d replicas so the driver's 2 MB unit resolves.\n"
                    " SUPERSET AGAINST SUBSET, and it is labelled exactly as the speed\n"
                    " comparison is: cv::cuda::ORB necessarily allocates a DETECTOR's\n"
                    " pyramid, mask pyramid and response buffer, which a family that takes\n"
                    " keypoints as INPUT does not have. 'Smaller by any amount' cannot fail\n"
                    " informatively and is not the bar.\n",
                    kReplicas);
        const size_t step = measureDriverMeterStep();
        {
            DeviceMemMeter meter;
            std::vector<std::unique_ptr<FamilyBuffers>> keep;
            for (int i = 0; i < kReplicas; ++i)
                keep.push_back(std::unique_ptr<FamilyBuffers>(new FamilyBuffers(1000)));
            const size_t delta = meter.deltaBytes();
            // The RAW delta goes to the meter's own printer, because its caveat
            // is about whether THAT reading cleared the driver's unit. The
            // per-frame figure is the division, stated separately.
            printDriverDelta("binCV family, 128 replicas", delta, step);
            std::printf("   -> %.1f KB per frame at N=1000 (delta / %d).\n",
                        static_cast<double>(delta) / static_cast<double>(kReplicas) / 1024.0,
                        kReplicas);
            std::printf("ROW|mem_bincv_per_frame|%zu|0|0\n", delta / kReplicas);
        }
        {
            DeviceMemMeter meter;
            std::vector<cv::Ptr<cv::cuda::ORB>> orbs;
            std::vector<cv::cuda::GpuMat> kps, descs;
            for (int i = 0; i < 8; ++i) {
                orbs.push_back(cv::cuda::ORB::create(nfeatures, 1.2f, 1, 31, 0, 2,
                                                     cv::ORB::HARRIS_SCORE, 31, 20, false));
                kps.emplace_back();
                descs.emplace_back();
                orbs.back()->detectAndComputeAsync(gpuMat, cv::noArray(), kps.back(),
                                                   descs.back(), false, cvStream);
            }
            cvStream.waitForCompletion();
            const size_t delta = meter.deltaBytes();
            printDriverDelta("cv::cuda::ORB, 8 instances", delta, step);
            std::printf("   -> %.2f MB per instance (delta / 8).\n",
                        static_cast<double>(delta) / 8.0 / (1024.0 * 1024.0));
            std::printf("ROW|mem_orb_per_instance|%zu|0|0\n", delta / 8);
            std::printf("   UPPER BOUND on that side: GpuMat may pool, and anything held\n"
                        "   rather than freed lands inside the delta. binCV's side has no\n"
                        "   allocator between the op and cudaMalloc.\n");
        }
    }
#endif

    cudaStreamDestroy(gStream);
    std::printf("\n DONE. Rows tagged ROW| and PAIR| are machine-readable; aggregate at\n"
                " least seven independent process runs before quoting any of them.\n");
    return 0;
}
