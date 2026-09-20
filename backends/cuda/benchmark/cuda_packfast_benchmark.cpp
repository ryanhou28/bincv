// THE PACKER'S ARMS, PRICED AGAINST EACH OTHER.
//
// `cuda::threshold` lost its role comparison against `cv::cuda::threshold` --
// 2.22x slower at 3840x2160 while moving 1.78x LESS traffic, which is the fail
// condition its own author wrote before measuring. The mechanism was located
// with `cuobjdump` rather than a profiler, because no profiler runs on this
// machine: `packKernel<uint8_t, GreaterEqual>` assembled to 184 instructions
// around ONE load and ONE store, two of them software divides, from the
// `wordIdx / words` a flat grid-stride loop needs to recover (row, word).
//
// This binary prices the two fixes SEPARATELY, because they are two claims:
//
//   arm 0  grid-stride, one lane one pixel   -- what shipped, and the oracle
//   arm 1  2-D grid, blockIdx.y IS the row   -- both divides deleted
//   arm 2  arm 1 plus four pixels per lane   -- one 32-bit load where the warp
//                                              was issuing thirty-two byte ones
//
// and reports arm1/arm0, arm2/arm1 and arm2/arm0 at every geometry. A single
// end-to-end ratio would say the packer got faster without saying which of the
// two changes did it, and the second one costs a hand-written kernel that has
// to stay bit-exact forever -- a price only its own measured share can justify.
//
// WHAT THIS BINARY IS NOT. It carries NO OpenCV arm. The role comparison
// against `cv::cuda::threshold` -- the bar this work exists to clear -- lives in
// cuda_role_benchmark, in one process with every other cv::cuda pair, on one
// explicit stream and one meter. Reproducing it here would be a fifth answer to
// a question that file exists to answer once.
//
// THE PROTOCOL, and every part of it is load-bearing on this host:
//
//   * ONE EXPLICIT STREAM for every arm and for the launch floor. Events
//     recorded on the legacy default stream while the body enqueues elsewhere
//     bracket the wrong work.
//   * INTERLEAVED pairs. `timeKernelPaired` brackets both arms inside every
//     round and alternates their order, so each round yields one observation of
//     the ratio with the drift divided out. Whether a difference is real is
//     decided by measure_util.hpp's rule -- the difference must exceed the
//     larger of the within-run spread and the run-to-run scatter -- and the
//     two arms' RANGE separation is printed beside it as a fact, not as a
//     second veto. See paired_stats.hpp: one round slow in BOTH arms overlaps
//     the ranges while leaving every per-round ratio untouched.
//   * THE LAUNCH FLOOR is measured, not quoted. At 752x480 this op is a single
//     launch over 400 KB and sits ON the floor; the 4K geometry is where the
//     kernel is visible at all, and every arm prints its floor share.
//   * ONE RUN IS NOT A NUMBER. Every timed pair also emits a machine-readable
//     ROW line so a cross-process aggregation over seven or more runs can take
//     the median of medians. The prose is for reading one run.
//
// BOTH HALVES OF THE SWITCH RULE ARE HERE. The off-switch ratio is printed at
// every admitted geometry (it must NOT be ~1.00x -- that would mean the switch
// is not switching and one arm was timed twice), and FOUR gate-excluded
// controls are printed, each a shape a fast arm's own gate refuses: a uint16
// source, an odd-strided view and a cutoff of 256 for the byte lane, and an
// image taller than `gridDim.y` for the row grid. Those must read ~1.00x.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/pack.hpp"
#include "bincv/cuda/threshold.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/pack.hpp"
#include "cuda_bench_util.hpp"

namespace bc = bincv::cuda;
using namespace cudabench;

namespace {

/// @brief ONE EXPLICIT STREAM for every arm and the floor -- see the file note.
cudaStream_t gStream = nullptr;

/// @brief Rounds per paired comparison inside ONE process. The median across
/// seven-plus processes is what is finally quoted; this is the inner median.
constexpr int kRounds = 15;

struct Geometry {
    size_t w, h;
    const char* label;
    int iters;
};

// 752x480 is the reference frame every other table in this backend uses, and it
// is where the op sits on the launch floor. 3840x2160 is where the published
// failure was measured and where the kernel is visible; 1920x1080 is between
// them. All three are on the ladder the ship bar applies to.
const Geometry kGeometries[] = {
    {752, 480, "752x480 -- the reference frame (ON the launch floor)", 40},
    {1920, 1080, "1920x1080", 30},
    {3840, 2160, "3840x2160 -- where the published failure was measured", 20},
};

uint64_t rngState = 0x9E3779B97F4A7C15ULL;
uint8_t nextByte() {
    rngState = rngState * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint8_t>(rngState >> 40);
}

std::vector<uint8_t> makeFrame(size_t w, size_t h) {
    std::vector<uint8_t> img(w * h);
    for (auto& v : img) v = nextByte();
    return img;
}

double peakBytesPerMs() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);
    const double gbPerSec = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8.0) / 1.0e6;
    return gbPerSec * 1.0e9 / 1000.0;
}

size_t bitBytes(size_t w, size_t h) { return h * bc::rowWords(w) * 4; }

/// @brief An enqueue body that FIRST selects its arm, so the two arms of a pair
/// are the same call with the switches in two positions -- one binary, one
/// call site, which is what the both-arms rule asks for.
std::function<void()> withArm(bool rowGrid, bool byteLane,
                              const std::function<void()>& body) {
    return [rowGrid, byteLane, body] {
        bc::impl::packRowGridEnabled() = rowGrid;
        bc::impl::packByteLaneEnabled() = byteLane;
        body();
    };
}

void emitRow(const char* key, const char* geom, const PairedTiming& p) {
    std::printf("ROW,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d"
                ",%.6f,%d,%d,%d,%.6f,%.6f,%.4g\n",
                key, geom, p.a.minMs, p.a.medianMs, p.a.maxMs, p.b.minMs, p.b.medianMs,
                p.b.maxMs, p.ratioMin, p.ratioMedian, p.ratioMax, p.separated() ? 1 : 0,
                p.rounds, p.ratioGeoMean, p.roundsFavouringA, p.roundsFavouringB,
                p.roundsTied, p.differenceFactor(), p.ratioSwingFactor(), p.signTestP());
}

/// @brief A pair whose ratio is a CLAIM: B is meant to beat A, and the printer
/// says whether this run can tell them apart at all.
void printGain(const char* key, const char* geom, const char* nameA, const char* nameB,
               const PairedTiming& p) {
    cudabench::pairedScope() = geom;
    printPaired(nameA, nameB, p, "kernel");
    const double faster = p.ratioMedian > 0.0 ? 1.0 / p.ratioMedian : 0.0;
    // The verdict is printPaired's, which is measure_util.hpp's rule. This line
    // reads the SAME number the other way up for a human, and says nothing
    // about whether it is real -- the line above it has already said that.
    std::printf("   B is %.2fx %s than A this run.\n",
                faster >= 1.0 ? faster : 1.0 / faster,
                faster >= 1.0 ? "FASTER" : "SLOWER");
    emitRow(key, geom, p);
}

/// @brief A GATE-EXCLUDED control: a shape the fast arm's own gate refuses, so
/// flipping the switch must select nothing and the ratio must read ~1.00x.
void printControl(const char* key, const char* geom, const char* nameA, const char* nameB,
                  const PairedTiming& p, const Timing& floor, bool gateSaysApplies) {
    cudabench::pairedScope() = geom;
    std::printf("   the gate itself says: applies = %s\n",
                gateSaysApplies ? "TRUE  <-- WRONG, this control is not excluded"
                                : "FALSE (excluded, as required)");
    printPaired(nameA, nameB, p, "kernel", /*expect1x=*/true);
    const double share = p.a.medianMs > 0.0 ? floor.medianMs / p.a.medianMs : 0.0;
    if (share >= 0.25) {
        std::printf("   READING: both arms are within %.0f%% of the launch floor, so the\n"
                    "   +/-5%% verdict above is noise either way at this geometry. The\n"
                    "   decidable reading is the one taken where the arms are visible.\n",
                    share * 100.0);
    } else {
        std::printf("   READING: the arms are %.1fx the launch floor, so this control's\n"
                    "   verdict IS decidable here.\n",
                    1.0 / share);
    }
    emitRow(key, geom, p);
}

void printArmWithFloors(const char* name, const Timing& t, const Timing& floor,
                        size_t trafficBytes, double bytesPerMs) {
    printArmVsFloor(name, t, floor, "kernel");
    const double bwFloor = static_cast<double>(trafficBytes) / bytesPerMs;
    const double realized = t.medianMs > 0.0
                                ? static_cast<double>(trafficBytes) / t.medianMs / 1.0e6
                                : 0.0;
    std::printf("   %-42s %9.4f ms   (%zu B at peak; this arm realizes %.0f GB/s,\n"
                "   %sor %.1fx that floor)\n",
                "bandwidth floor for this op's traffic", bwFloor, trafficBytes, realized,
                "                                           ",
                bwFloor > 0.0 ? t.medianMs / bwFloor : 0.0);
}

} // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    const std::string only = argc > 1 ? argv[1] : "";

    std::printf("=======================================================================\n"
                " THE PACKER'S THREE ARMS -- grid-stride, row grid, byte lane\n"
                "=======================================================================\n");
    printDevice();
    std::printf(" Every time below is KERNEL-RESIDENT, both arms of every pair on ONE\n"
                " explicit stream, interleaved and order-alternated. Arm A is always the\n"
                " arm being replaced; a ratio UNDER 1.00x means B is faster.\n"
                " NO OpenCV arm is in this binary -- the role bar lives in\n"
                " cuda_role_benchmark, measured once for every op in one process.\n\n");

    cudaStreamCreate(&gStream);
    const Timing floor = measureLaunchFloor(dim3(1), dim3(32), 100, 25, 250.0, gStream);
    printLaunchFloor(floor);
    std::printf("FLOOR,%.6f,%.6f,%.6f\n", floor.minMs, floor.medianMs, floor.maxMs);
    const double bytesPerMs = peakBytesPerMs();

    for (const Geometry& geo : kGeometries) {
        if (!only.empty() && only != geo.label && only != std::to_string(geo.w)) continue;
        const size_t w = geo.w, h = geo.h;
        char geom[32];
        std::snprintf(geom, sizeof(geom), "%zux%zu", w, h);
        const int iters = geo.iters;

        std::printf("\n===========================================================\n");
        std::printf(" %s\n", geo.label);
        std::printf("===========================================================\n");

        const std::vector<uint8_t> frame = makeFrame(w, h);
        bc::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
        bc::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view());
        bc::DeviceBinMat dBits(static_cast<int>(w), static_cast<int>(h));

        const size_t traffic = w * h + bitBytes(w, h);

        // ---------------- packBits: the ladder, one rung at a time ----------
        std::printf("\n -- packBits(GreaterEqual, 127): the ladder, one rung at a time\n");
        {
            const auto body = [&] {
                bc::packBits(dImg.constView(), dBits.view(), bincv::PackRule::GreaterEqual,
                             uint8_t{127}, gStream);
            };
            const auto p10 = timeKernelPaired(withArm(false, false, body),
                                              withArm(true, false, body), iters, iters,
                                              kRounds, gStream);
            printGain("pack_rowgrid_vs_gridstride", geom,
                      "A: arm 0, grid-stride (what shipped)",
                      "B: arm 1, row grid (divides deleted)", p10);

            const auto p21 = timeKernelPaired(withArm(true, false, body),
                                              withArm(true, true, body), iters, iters,
                                              kRounds, gStream);
            printGain("pack_bytelane_vs_rowgrid", geom, "A: arm 1, row grid",
                      "B: arm 2, byte lane (4 px/lane)", p21);

            const auto p20 = timeKernelPaired(withArm(false, false, body),
                                              withArm(true, true, body), iters, iters,
                                              kRounds, gStream);
            printGain("pack_bytelane_vs_gridstride", geom,
                      "A: arm 0, grid-stride (what shipped)",
                      "B: arm 2, byte lane (what ships now)", p20);
            printArmWithFloors("packBits, the shipped arm", p20.b, floor, traffic,
                               bytesPerMs);
            bc::impl::packRowGridEnabled() = true;
            bc::impl::packByteLaneEnabled() = true;
        }

        // ---------------- cuda::threshold, the Tier 1 caller ----------------
        std::printf("\n -- cuda::threshold(127.0): the caller whose role bar this work\n"
                    "    exists to clear. It is packBits with one host-side cutoff in\n"
                    "    front of it, so it inherits the ladder above -- priced here\n"
                    "    anyway, because a caller pays the composition and not the kernel.\n");
        {
            const auto body = [&] {
                bc::threshold(dImg.constView(), dBits.view(), 127.0, gStream);
            };
            const auto p = timeKernelPaired(withArm(false, false, body),
                                            withArm(true, true, body), iters, iters,
                                            kRounds, gStream);
            printGain("threshold_bytelane_vs_gridstride", geom,
                      "A: cuda::threshold on arm 0 (grid-stride)",
                      "B: cuda::threshold on arm 2 (byte lane)", p);
            printArmWithFloors("cuda::threshold, the shipped arm", p.b, floor, traffic,
                               bytesPerMs);
            bc::impl::packRowGridEnabled() = true;
            bc::impl::packByteLaneEnabled() = true;
        }

        // ---------------- packQuant: the shared blast radius ----------------
        std::printf("\n -- packQuant(n = 2): the N-deep twin in the same file. It got the\n"
                    "    row grid and NOT the byte lane, and the question here is whether\n"
                    "    a change made for threshold cost this op anything.\n");
        {
            bc::DeviceBinMat dBlock(static_cast<int>(w), static_cast<int>(2 * h));
            const auto body = [&] {
                bc::packQuant(dImg.constView(), dBlock.view(), 2, gStream);
            };
            const auto p = timeKernelPaired(withArm(false, false, body),
                                            withArm(true, false, body), iters, iters,
                                            kRounds, gStream);
            printGain("packquant_rowgrid_vs_gridstride", geom,
                      "A: packQuant n=2, grid-stride", "B: packQuant n=2, row grid", p);
            printArmWithFloors("packQuant n=2, the shipped arm", p.b, floor,
                               w * h + 2 * bitBytes(w, h), bytesPerMs);
            bc::impl::packRowGridEnabled() = true;
            bc::impl::packByteLaneEnabled() = true;
        }

        // ---------------- gate-excluded control: a uint16 source ------------
        std::printf("\n -- GATE-EXCLUDED CONTROL 1: a uint16 source. `__vsetgeu4` compares\n"
                    "    BYTES, so the byte-lane gate refuses this and flipping its switch\n"
                    "    must select nothing.\n");
        {
            bc::DeviceImage<uint16_t> dImg16(static_cast<int>(w), static_cast<int>(h));
            std::vector<uint16_t> wide(w * h);
            for (size_t i = 0; i < wide.size(); ++i)
                wide[i] = static_cast<uint16_t>(frame[i]) * 257u;
            bc::uploadImage<uint16_t>(wide.data(), w, h, w, dImg16.view());
            const auto body = [&] {
                bc::packBits(dImg16.constView(), dBits.view(),
                             bincv::PackRule::GreaterEqual, uint16_t{30000}, gStream);
            };
            const auto p = timeKernelPaired(withArm(true, false, body),
                                            withArm(true, true, body), iters, iters,
                                            kRounds, gStream);
            printControl("control_uint16", geom, "A: uint16, byte lane OFF",
                         "B: uint16, byte lane ON", p, floor,
                         bc::impl::packByteLaneApplies(dImg16.getStride(),
                                                       dImg16.constView().ptr, 2, 30000));
            bc::impl::packRowGridEnabled() = true;
            bc::impl::packByteLaneEnabled() = true;
        }

        // ---------------- gate-excluded control: an odd stride --------------
        std::printf("\n -- GATE-EXCLUDED CONTROL 2: a source VIEW whose stride is not a\n"
                    "    multiple of 4. The quad load is an ALIGNED 32-bit access, so from\n"
                    "    row 1 onward the byte-lane arm cannot make it; the gate refuses.\n");
        {
            const size_t oddStride = w + 1;  // w is even here, so w+1 is odd
            uint8_t* dev = nullptr;
            cudaMalloc(&dev, oddStride * h);
            cudaMemset(dev, 0x40, oddStride * h);
            const bc::DeviceImageConstView<uint8_t> odd{dev, w, h, oddStride};
            const auto body = [&] {
                bc::packBits(odd, dBits.view(), bincv::PackRule::GreaterEqual,
                             uint8_t{127}, gStream);
            };
            const auto p = timeKernelPaired(withArm(true, false, body),
                                            withArm(true, true, body), iters, iters,
                                            kRounds, gStream);
            printControl("control_odd_stride", geom, "A: odd stride, byte lane OFF",
                         "B: odd stride, byte lane ON", p, floor,
                         bc::impl::packByteLaneApplies(oddStride, dev, 1, 127));
            cudaFree(dev);
            bc::impl::packRowGridEnabled() = true;
            bc::impl::packByteLaneEnabled() = true;
        }

        // ---------------- gate-excluded control: a cutoff of 256 ------------
        std::printf("\n -- GATE-EXCLUDED CONTROL 3: GreaterThan at t = 255. The rule folds\n"
                    "    to `v >= 256`, which is \"nothing passes\" -- true arithmetic that no\n"
                    "    byte lane can express, so the gate refuses it and the row grid\n"
                    "    answers. This is the one gate a caller can trip with a VALUE.\n");
        {
            const auto body = [&] {
                bc::packBits(dImg.constView(), dBits.view(), bincv::PackRule::GreaterThan,
                             uint8_t{255}, gStream);
            };
            const auto p = timeKernelPaired(withArm(true, false, body),
                                            withArm(true, true, body), iters, iters,
                                            kRounds, gStream);
            printControl("control_cutoff_256", geom, "A: GreaterThan 255, byte lane OFF",
                         "B: GreaterThan 255, byte lane ON", p, floor,
                         bc::impl::packByteLaneApplies(dImg.getStride(),
                                                       dImg.constView().ptr, 1, 256));
            bc::impl::packRowGridEnabled() = true;
            bc::impl::packByteLaneEnabled() = true;
        }
    }

    // ---------------- gate-excluded control: taller than gridDim.y ----------
    std::printf("\n===========================================================\n"
                " GATE-EXCLUDED CONTROL 4: an image taller than gridDim.y\n"
                "===========================================================\n"
                " The row-grid arm carries the image row in `blockIdx.y`, which the\n"
                " hardware caps at 65535. A 70,000-row image cannot be expressed that\n"
                " way at all, so BOTH switch positions run the grid-stride arm and the\n"
                " ratio must read ~1.00x. This is the row grid's own gate, and it is the\n"
                " reason the arm it replaced still exists.\n");
    {
        const size_t w = 32, h = 70000;
        const std::vector<uint8_t> tall = makeFrame(w, h);
        bc::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
        bc::uploadImage<uint8_t>(tall.data(), w, h, w, dImg.view());
        bc::DeviceBinMat dBits(static_cast<int>(w), static_cast<int>(h));
        const auto body = [&] {
            bc::packBits(dImg.constView(), dBits.view(), bincv::PackRule::GreaterEqual,
                         uint8_t{127}, gStream);
        };
        const auto p = timeKernelPaired(withArm(false, false, body),
                                        withArm(true, true, body), 20, 20, kRounds,
                                        gStream);
        printControl("control_tall_image", "32x70000", "A: 70,000 rows, fast arms OFF",
                     "B: 70,000 rows, fast arms ON", p, floor,
                     bc::impl::packRowGridApplies(h));
        bc::impl::packRowGridEnabled() = true;
        bc::impl::packByteLaneEnabled() = true;
    }

    // ---------------- memory: meter 1, with the formulas --------------------
    std::printf("\n===========================================================\n"
                " MEMORY -- meter 1 (allocation sum), binCV against binCV\n"
                "===========================================================\n"
                " NOTHING HERE CHANGED, and that is the point of printing it: three\n"
                " kernel shapes, one footprint. No arm allocates, none takes caller\n"
                " scratch, and none asks for a byte of shared memory -- the byte-lane\n"
                " arm's reuse lives in registers and three warp shuffles. The\n"
                " cross-library figure is a cudaMemGetInfo delta and lives in\n"
                " cuda_role_benchmark; meter 1 and meter 2 are never divided.\n");
    for (const Geometry& geo : kGeometries) {
        const size_t w = geo.w, h = geo.h;
        std::printf("   %-11zux%-6zu src %8.1f KB (h*w)      bits %8.1f KB"
                    " (h*rowWords(w)*4)\n"
                    "   %-18s shared memory per block: 0 B   device scratch: 0 B\n",
                    w, h, static_cast<double>(w * h) / 1024.0,
                    static_cast<double>(bitBytes(w, h)) / 1024.0, "");
        std::printf("MEM,%zux%zu,%zu,%zu,0,0\n", w, h, w * h, bitBytes(w, h));
    }

    cudaStreamDestroy(gStream);
    std::printf("\n Every number above is from ONE process on a machine whose GPU may be\n"
                " shared. One run is not a number: aggregate the ROW lines over seven or\n"
                " more runs and take the median of the medians.\n");
    return 0;
}
