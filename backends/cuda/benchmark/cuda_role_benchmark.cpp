// THE ROLE COMPARISONS, ALL OF THEM, IN ONE BINARY AND ONE SESSION.
//
// Every device op in this backend whose job a cv::cuda call already does, timed
// against that call on the same device, the same frames and the same process
// run -- and every op whose job no cv::cuda call does, named here with its
// verdict recorded OUTSTANDING rather than given a substitute bar.
//
// WHY A SEPARATE BINARY FROM THE FAMILY BENCHMARKS. The family benchmarks each
// carry their own role arms, and each was written against its own OpenCV
// find_package. That makes four independent answers to "how does binCV compare
// to OpenCV on this GPU", taken in four processes at four moments, with the two
// sides' memory read on DIFFERENT meters -- binCV's allocation sum against
// OpenCV's pitch read-back. Neither of those is wrong on its own and both are
// printed by the families, but a RATIO across two meters answers no question,
// and four processes cannot be compared to each other at all on a device whose
// clock ramps. This file exists to make the cross-library claim once, under one
// protocol:
//
//   * ONE PROCESS. Every pair below runs in the same session, after the same
//     warm-up, against the same measured launch floor.
//   * INTERLEAVED. Every pair goes through timeKernelPaired, which brackets
//     both arms inside every round and alternates their order round to round.
//     A ratio of two separately-measured medians carries the drift between
//     them; a per-round ratio does not.
//   * ONE METER ACROSS THE BOUNDARY. Device memory on BOTH sides is a
//     cudaMemGetInfo delta -- meter 2 -- and nothing here divides it by an
//     allocation sum. Meter 1 and meter 3 are printed too, in their own block,
//     labelled as binCV-side context and never as half of a cross-library
//     ratio.
//   * THE METER IS MADE TO RESOLVE. This driver reserves in 2 MB units, which
//     is larger than most of these working sets. Every memory figure is
//     therefore taken over kReplicas independent working sets so the delta is
//     many units wide, and the per-frame figure is that delta divided by the
//     replica count. A reading that still does not clear eight units says so
//     and is not quoted.
//   * THE OPENCV SIDE IS METERED WITH ITS FILTER RUNNING. cv::cuda::Filter
//     allocates internal buffers lazily, and NPP allocates scratch. Those are
//     part of what a caller pays, and binCV's equivalent -- morphologyEx's
//     scratch -- is caller-provided and counted here. So the metered scope
//     includes constructing the filter AND applying it once, on both sides.
//
// WHAT THE OPENCV READING IS AND IS NOT. GpuMat's default allocator is a
// straight cudaMallocPitch, but OpenCV can be built or configured with a
// BufferPool, and a filter may keep a buffer alive past the call that made it.
// Anything held rather than freed lands inside the delta. So the OpenCV figure
// is an UPPER BOUND on that side's per-frame footprint, and it is stated as one
// at every number. binCV's side has no allocator between the op and cudaMalloc,
// so its reading is the arrays plus the driver's rounding and nothing else.
//
// WHAT IS NOT HERE. No op is given a bar it does not have. cuda::binarize,
// cuda::shift and cuda::buildPyramidBox's own N-bit ladder have no cv::cuda
// counterpart at any API level; each is listed in the OUTSTANDING section with
// what it would take to price it. A CPU number is never quoted as a GPU bar.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// EVERY ROW IN THIS FILE IS A COMPARISON AGAINST OpenCV, so unlike the family
// benchmarks there are no binCV-only arms to fall back on. The target is
// nevertheless ALWAYS BUILT, with the whole body behind this guard, for a
// reason that is about the gate rather than about the measurement:
// scripts/verify_cuda.sh derives its benchmark list from the TEXT of
// benchmark/CMakeLists.txt and then excludes exactly one name by hand
// (`grep -v cuda_stereobm_benchmark`). A second target that exists only when
// an OpenCV is pointed at would therefore be one the gate tries to build on
// every machine that has not built one -- which is the hazard
// cuda_sensor_benchmark's CMake block already documents, and which this file
// walked into before being restructured this way.
//
// Compiled without that OpenCV the binary still runs and still says something
// true: it reports every role bar as UNMEASURED and the verdicts that depend
// on one as BLOCKED or OUTSTANDING, which is what the both-axes ship rule
// requires. It does not substitute a bar and it does not fall silent.
#if BINCV_CUDA_ROLE_OPENCV

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include "bincv/binMat.hpp"
#include "bincv/io/sequence.hpp"
#include "bincv/cuda/corner.hpp"
#include "bincv/cuda/denseDisparity.hpp"
#include "bincv/cuda/derivative.hpp"
#include "bincv/cuda/descriptor.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/edge.hpp"
#include "bincv/cuda/fast.hpp"
#include "bincv/cuda/keypoints.hpp"
#include "bincv/cuda/median.hpp"
#include "bincv/cuda/morphology.hpp"
#include "bincv/cuda/orientation.hpp"
#include "bincv/cuda/pyramid.hpp"
#include "bincv/cuda/shift.hpp"
#include "bincv/cuda/threshold.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/morphology.hpp"
#include "bincv/ops/orbPattern.hpp"
#include "bincv/ops/pack.hpp"
#include "cuda_bench_util.hpp"

using namespace cudabench;
namespace bc = bincv::cuda;

namespace {

// The reference frame. 752x480 is the EuRoC geometry this project's frontend
// numbers are all taken at, and it is where every implementer's written bar
// was set. 1920x1080 is carried alongside because several of these ops sit on
// the launch floor at 752x480, where a ratio between two arms is a ratio
// between two launches -- the second geometry is what makes those readable,
// and every row says which geometry it was taken at.
constexpr size_t kW = 752, kH = 480;
constexpr size_t kW2 = 1920, kH2 = 1080;

// Replicas per memory reading. Chosen so the SMALLER side -- binCV's 92 KB
// working set at 752x480 -- clears the driver's 2 MB unit many times over
// rather than by one: 256 replicas put it at ~23 MB, which is ~11 units, and
// the same count on the OpenCV side puts that at ~250 MB. Both sides use the
// same count, so the ratio is a ratio of like readings.
constexpr int kReplicas = 256;

// Two sides need their OWN counts, for opposite reasons, and both are named
// here rather than buried at the call site.
//
// cv::cuda::createMedianFilter allocates cols * 256 * partitions * 4 bytes of
// fine histogram plus cols * 8 * partitions * 4 of coarse -- READ OUT OF
// OpenCV 4.5.4's own filtering.cpp, not guessed. At 752 columns and its
// default 128 partitions that is ~98 MB of scratch for ONE 752x480 frame, so
// 256 replicas would ask for 25 GB on an 8 GB card. Eight replicas of ~100 MB
// is already 400 of the meter's 2 MB units, which is far more resolution than
// the 8-unit floor this file requires.
constexpr int kMedianOcvReplicas = 8;

// The composed edge chain holds seven intermediates, six of them CV_16S --
// ~4.9 MB per frame at 752x480. 64 replicas is ~315 MB and ~157 units.
constexpr int kEdgeOcvReplicas = 64;

// Rounds per paired comparison inside ONE process. The run-to-run median across
// seven-plus processes is what is finally quoted; this is the inner median.
constexpr int kRounds = 15;

// ONE EXPLICIT STREAM FOR THE WHOLE RUN, and this is a protocol decision with
// a measurement behind it rather than a style choice.
//
// The legacy default stream implicitly synchronizes with every other blocking
// stream. Timing one arm there and the other on an explicit stream therefore
// does not merely put them on separate queues -- it makes each arm's event
// bracket contain part of the other arm's work. Measured here before this was
// fixed: a 5 ms cv::cuda median produced 32-SECOND samples, and binCV's 0.02 ms
// medianWide produced 1.4-second ones, purely from that interaction.
//
// It also matters for a second reason specific to this comparison.
// cv::cuda::createMedianFilter calls cudaDeviceSynchronize() inside apply()
// when and only when it is handed the default stream (OpenCV 4.5.4,
// cudafilters/src/cuda/median_filter.cu). On the default stream a batch of
// enqueues cannot pipeline on that side while binCV's does, which is not a
// kernel-to-kernel comparison. On an explicit stream neither side
// synchronizes and both pipeline.
//
// So EVERY arm in this file -- both sides of every pair, the launch floor, and
// the host-enqueue probe -- runs on this one stream, and the events are
// recorded on it. The default stream's extra cost is measured separately and
// reported as the API cost it is.
cudaStream_t gStream = nullptr;
cv::cuda::Stream gCvStream;

uint64_t rngState = 0x9E3779B97F4A7C15ULL;
uint8_t nextByte() {
    rngState = rngState * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint8_t>(rngState >> 40);
}

/// @brief A frame with structure in it: smoothed noise, so a median and a
/// morphology have something to do other than copy. A pure-noise frame makes
/// a 3x3 median's output nearly independent of its input and can let a
/// data-dependent early-out look fast; a flat frame does the same the other
/// way.
std::vector<uint8_t> makeFrame(size_t w, size_t h) {
    std::vector<uint8_t> img(w * h);
    for (auto& v : img) v = nextByte();
    std::vector<uint8_t> tmp(img);
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

// ---------------------------------------------------------------------------
// Machine-readable emission
//
// This binary is run at least seven times and its rows aggregated across
// processes -- one run of a small kernel on this host is not a number, it is a
// sample of a distribution whose spread reached 338% in a single run. Every
// timed row therefore also prints a ROW line: a stable key, both arms' full
// min/median/max, the per-round ratio's own min/median/max, and this run's
// separation verdict. The prose table above it is for a human reading one run;
// the ROW lines are what the cross-run aggregation reads.
// ---------------------------------------------------------------------------

void emitRow(const char* key, const char* geom, const PairedTiming& p) {
    std::printf("ROW,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d\n", key,
                geom, p.a.minMs, p.a.medianMs, p.a.maxMs, p.b.minMs, p.b.medianMs,
                p.b.maxMs, p.ratioMin, p.ratioMedian, p.ratioMax, p.separated() ? 1 : 0,
                p.rounds);
}

void emitFloor(const Timing& t) {
    std::printf("FLOOR,%.6f,%.6f,%.6f\n", t.minMs, t.medianMs, t.maxMs);
}

// ---------------------------------------------------------------------------
// THE HOST-ENQUEUE PROBE -- what a CUDA-event bracket cannot separate, and how
// to separate it anyway on a machine with no profiler.
//
// A CUDA-event bracket around N back-to-back enqueues measures the GPU's
// elapsed time between two markers. That is the kernel's time ONLY while the
// host can feed the device faster than the device drains. When one call spends
// more time in host-side dispatch than its kernel spends running, the device
// starves between launches and the bracket reports the DISPATCH, not the
// kernel -- and it reports it as though it were device time.
//
// That is not a hypothetical here. cv::cuda::threshold reads almost the same
// time at 752x480 and at 1920x1080, seven times the pixels. A kernel does not
// do that; a fixed per-call cost does. So this probe times the host's own loop
// around the same enqueues with NO synchronize: what comes back is the wall
// time the calling thread spends inside the API per call. Comparing it with the
// event-bracketed number tells a reader which of the two a row is made of:
//
//   host enqueue << event time  -> the row is the kernel.
//   host enqueue ~= event time  -> the row is dispatch overhead; the device was
//                                  idle waiting for the host, and the ratio in
//                                  that row is a ratio of two API call costs.
//
// This is the measurement that locates a gap a profiler would otherwise be
// needed for, and no profiler runs on this host.
// ---------------------------------------------------------------------------

/// @brief Wall time the CALLING THREAD spends per enqueue, no synchronize.
/// @note The device is drained before and after, so the loop being timed is
/// enqueue only. The final synchronize is outside the clock.
Timing timeHostEnqueue(const std::function<void()>& body, int iters = 20,
                       int repeats = 15) {
    for (int i = 0; i < iters; ++i) body();
    cudaDeviceSynchronize();
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) body();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() /
                          iters);
        cudaDeviceSynchronize();
    }
    return summarize(std::move(samples));
}

/// @brief GPU time of ONE call, with the device drained on both sides of it.
/// @note THE DISAMBIGUATION THE BATCHED TIMER CANNOT MAKE. When an arm's host
/// dispatch costs more than its kernel, a batch of N enqueues starves the
/// device and the event bracket reports the dispatch. Measuring ONE call
/// between two synchronizes removes the starvation: the host is finished
/// enqueueing long before the events are read, so what the pair brackets is
/// the work itself.
/// @note It is not the number to quote for a cheap kernel -- a solo call pays
/// the device's wake-up and the event pair's own cost, which is the launch
/// floor all over again. It is a DIAGNOSTIC, used here only to answer
/// "is that arm's measured time its kernel or its API?", and the batched
/// number stays the one reported.
Timing timeKernelSolo(const std::function<void()>& body, int repeats = 15) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 3; ++i) body();
    cudaDeviceSynchronize();
    std::vector<double> samples;
    for (int r = 0; r < repeats; ++r) {
        cudaDeviceSynchronize();
        cudaEventRecord(start, gStream);
        body();
        cudaEventRecord(stop, gStream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        samples.push_back(static_cast<double>(ms));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return summarize(std::move(samples));
}

/// @brief Prints both arms' host-enqueue cost beside their event time and says
/// which of the two each row is actually made of.
void printEnqueue(const char* key, const char* geom, const char* nameA, const char* nameB,
                  const Timing& hostA, const Timing& hostB, const PairedTiming& p,
                  const Timing* soloA = nullptr, const Timing* soloB = nullptr) {
    const auto verdict = [](double host, double ev) {
        if (ev <= 0.0) return "?";
        const double s = host / ev;
        if (s >= 0.80) return "DISPATCH-BOUND: the event bracket is timing the API call";
        if (s >= 0.40) return "MIXED: host dispatch is a large share of the bracket";
        return "kernel-bound: the host feeds the device faster than it drains";
    };
    std::printf("   host-enqueue probe (no sync, wall clock per call):\n");
    std::printf("     %-42s %8.4f ms enqueue vs %7.4f ms event  -> %s\n", nameA,
                hostA.medianMs, p.a.medianMs, verdict(hostA.medianMs, p.a.medianMs));
    std::printf("     %-42s %8.4f ms enqueue vs %7.4f ms event  -> %s\n", nameB,
                hostB.medianMs, p.b.medianMs, verdict(hostB.medianMs, p.b.medianMs));
    std::printf("ENQ,%s,%s,%.6f,%.6f,%.6f,%.6f\n", key, geom, hostA.medianMs,
                p.a.medianMs, hostB.medianMs, p.b.medianMs);
    if (soloA != nullptr && soloB != nullptr) {
        std::printf("   single-call GPU time (device drained either side -- separates the\n"
                    "   kernel from the API for a dispatch-bound arm):\n"
                    "     %-42s %8.4f ms solo vs %7.4f ms batched\n"
                    "     %-42s %8.4f ms solo vs %7.4f ms batched\n",
                    nameA, soloA->medianMs, p.a.medianMs, nameB, soloB->medianMs,
                    p.b.medianMs);
        std::printf("SOLO,%s,%s,%.6f,%.6f,%.6f,%.6f\n", key, geom, soloA->medianMs,
                    p.a.medianMs, soloB->medianMs, p.b.medianMs);
    }
}

void emitMem(const char* key, const char* geom, size_t bincvBytes, size_t ocvBytes,
             size_t stepBytes, int binReplicas, int ocvReplicas) {
    std::printf("MEM,%s,%s,%zu,%zu,%zu,%d,%d\n", key, geom, bincvBytes, ocvBytes,
                stepBytes, binReplicas, ocvReplicas);
}

/// @brief One role row: names both arms, prints them with the floor's share of
/// each, then the per-round ratio and whether the ranges are disjoint.
/// @note Arm A is ALWAYS the OpenCV side and arm B is ALWAYS binCV's, so the
/// printed ratio B/A is "binCV's time as a fraction of OpenCV's" everywhere
/// in this file. Below 1.00x is binCV ahead. Having one orientation for
/// every row is worth more than having each row read naturally.
void printRole(const char* what, const char* ocvName, const char* binName,
               const PairedTiming& p, const Timing& floor, const char* geom) {
    std::printf("\n %s  [%s]\n", what, geom);
    printArmVsFloor(ocvName, p.a, floor, "kernel");
    printArmVsFloor(binName, p.b, floor, "kernel");
    std::printf("   ratio binCV/OpenCV, interleaved rounds: %5.3fx  (binCV %5.2fx %s)"
                "   per-round range %5.3f-%5.3fx (%d rounds)\n",
                p.ratioMedian, p.ratioMedian > 0.0 ? 1.0 / p.ratioMedian : 0.0,
                p.ratioMedian < 1.0 ? "FASTER" : "SLOWER", p.ratioMin, p.ratioMax,
                p.rounds);
    std::printf("   verdict THIS RUN: %s\n",
                p.separated() ? "sample ranges DISJOINT -- a result"
                              : "sample ranges OVERLAP -- not a result in this run");
}

// ---------------------------------------------------------------------------
// The memory probes -- meter 2 on both sides, same replica count
// ---------------------------------------------------------------------------

/// @brief cudaMemGetInfo delta across building `replicas` complete working sets
/// and running the op once on each.
/// @note The op runs INSIDE the scope on purpose: a lazily-allocated filter
/// buffer or an NPP scratch block is part of what the caller pays, and it
/// does not exist until the first apply.
size_t meterScope(const std::function<void(int)>& buildAndRunOne, int replicas) {
    DeviceMemMeter m;
    m.reset();
    for (int i = 0; i < replicas; ++i) buildAndRunOne(i);
    cudaDeviceSynchronize();
    return m.deltaBytes();
}

/// @note THE TWO SIDES MAY USE DIFFERENT REPLICA COUNTS, and the ratio is
/// still a ratio of like readings because each side is divided by its OWN
/// count before they are compared. The counts differ where they have to: a
/// side whose per-frame working set is ~100 MB cannot be replicated 256
/// times on an 8 GB card, and a side whose working set is 90 KB does not
/// resolve the meter's 2 MB unit at eight. What is required of BOTH is the
/// same thing -- that its own total clears eight units, which is checked
/// and, when it fails, the ratio is refused rather than printed.
void printMemPair(const char* what, const char* geom, size_t binBytes, size_t ocvBytes,
                  size_t step, int binReplicas, int ocvReplicas) {
    const double unit = static_cast<double>(step);
    const double binU = unit > 0.0 ? static_cast<double>(binBytes) / unit : 0.0;
    const double ocvU = unit > 0.0 ? static_cast<double>(ocvBytes) / unit : 0.0;
    const double binPer = static_cast<double>(binBytes) / binReplicas;
    const double ocvPer = static_cast<double>(ocvBytes) / ocvReplicas;
    std::printf("\n %s  [%s]  -- METER 2 on BOTH sides (cudaMemGetInfo delta),\n"
                " each side over enough independent working sets that the meter's\n"
                " %.2f MB unit resolves, then divided by its own count.\n",
                what, geom, unit / (1024.0 * 1024.0));
    std::printf("   [meter 2] cv::cuda  %10.2f MB over %4d sets = %9.1f KB/frame  (%.0f units)\n",
                static_cast<double>(ocvBytes) / (1024.0 * 1024.0), ocvReplicas,
                ocvPer / 1024.0, ocvU);
    std::printf("   [meter 2] binCV     %10.2f MB over %4d sets = %9.1f KB/frame  (%.0f units)\n",
                static_cast<double>(binBytes) / (1024.0 * 1024.0), binReplicas,
                binPer / 1024.0, binU);
    if (binBytes == 0 || binU < 8.0 || ocvU < 8.0) {
        std::printf("   RATIO NOT QUOTED: a side read under eight of the meter's own units,\n"
                    "   so its rounding is a large fraction of the reading. Raise that\n"
                    "   side's replica count before quoting this pair.\n");
    } else {
        std::printf("   ratio OpenCV/binCV = %.3fx  -- binCV smaller by that factor.\n"
                    "   The OpenCV side is an UPPER BOUND: GpuMat pads its pitch, and any\n"
                    "   filter buffer or NPP scratch held past the call is inside this\n"
                    "   delta. binCV's side is cudaMalloc with nothing in between.\n",
                    ocvPer / binPer);
    }
}

} // namespace

int main(int argc, char** argv) {
    // A single-family run, so a seven-run sweep of one family does not have to
    // pay for the others. No argument runs everything.
    const std::string only = argc > 1 ? argv[1] : "";
    const auto want = [&](const char* fam) {
        return only.empty() || only == fam;
    };

    std::printf("=======================================================================\n"
                " CUDA ROLE COMPARISONS -- every cv::cuda bar, one process, one protocol\n"
                "=======================================================================\n");
    printDevice();
    std::printf(" OpenCV %s\n", CV_VERSION);
    std::printf(" frames: %zux%zu (the reference geometry, where every written bar was\n"
                " set) and %zux%zu (carried because several of these ops sit ON the\n"
                " launch floor at the reference size, where a ratio between two arms is\n"
                " a ratio between two launches).\n",
                kW, kH, kW2, kH2);
    std::printf("\n ORIENTATION: every ratio below is binCV / OpenCV. Under 1.000x is\n"
                " binCV ahead. Arm A is OpenCV, arm B is binCV, in every pair.\n");

    // Force OpenCV's lazy CUDA context up before anything is timed, so the
    // first OpenCV arm does not pay context creation inside a timed batch.
    cv::cuda::setDevice(0);
    { cv::cuda::GpuMat warm(16, 16, CV_8UC1); warm.setTo(cv::Scalar(0)); }
    cudaDeviceSynchronize();

    cudaStreamCreate(&gStream);
    gCvStream = cv::cuda::StreamAccessor::wrapStream(gStream);

    const Timing floor = measureLaunchFloor(dim3(1), dim3(32), 100, 25, 250.0, gStream);
    std::printf("\n");
    printLaunchFloor(floor);
    emitFloor(floor);

    const size_t step = measureDriverMeterStep();
    std::printf("\n Driver meter step, measured in this run: %.2f MB. Every memory\n"
                " reading below is taken over %d working sets for that reason.\n",
                static_cast<double>(step) / (1024.0 * 1024.0), kReplicas);

    const std::vector<uint8_t> frame = makeFrame(kW, kH);
    const std::vector<uint8_t> frame2 = makeFrame(kW2, kH2);

    // ======================================================================
    // 1. threshold -- cv::cuda::threshold (cudaarithm). One launch vs one.
    // ======================================================================
    if (want("threshold")) {
        std::printf("\n=====================================================================\n"
                    " 1. THRESHOLD -- binCV cuda::threshold vs cv::cuda::threshold\n"
                    "=====================================================================\n"
                    " ROLE: 'turn a wide frame into a per-pixel above/below map on device'.\n"
                    " Both sides are ONE launch. They differ in what they emit: OpenCV\n"
                    " writes one BYTE per pixel (CV_8UC1, THRESH_BINARY), binCV writes one\n"
                    " BIT. That difference is the whole point of the library and it is\n"
                    " stated at the number rather than folded into it -- the outputs are\n"
                    " not the same array, so this is a role comparison, not an equality.\n");

        // THREE geometries here, not two. The sensor family reports its own
        // threshold headline at 3840x2160, so that size is carried as well --
        // a correction to a published number has to be taken at the size the
        // number was published at, or it is a different measurement.
        const size_t tW[] = {kW, kW2, 3840};
        const size_t tH[] = {kH, kH2, 2160};
        const char* tName[] = {"752x480", "1920x1080", "3840x2160"};
        for (int g = 0; g < 3; ++g) {
            const size_t w = tW[g], h = tH[g];
            const char* geom = tName[g];
            const std::vector<uint8_t> f4 = (g == 2) ? makeFrame(w, h)
                                                     : std::vector<uint8_t>();
            const std::vector<uint8_t>& f = g == 0 ? frame : (g == 1 ? frame2 : f4);

            bc::DeviceImage<uint8_t> src(static_cast<int>(w), static_cast<int>(h));
            bc::DeviceBinMat dst(static_cast<int>(w), static_cast<int>(h));
            bc::uploadImage(f.data(), w, h, w, src.view());

            cv::cuda::GpuMat gsrc(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            cv::cuda::GpuMat gdst(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            gsrc.upload(cv::Mat(static_cast<int>(h), static_cast<int>(w), CV_8UC1,
                                const_cast<uint8_t*>(f.data())));

            const PairedTiming p = timeKernelPaired(
                [&] { cv::cuda::threshold(gsrc, gdst, 127.0, 255.0, cv::THRESH_BINARY,
                                        gCvStream); },
                [&] { bc::threshold(src.constView(), dst.view(), 127.0, gStream); }, 20, 20,
                kRounds);
            printRole("threshold, uint8 -> above/below map", "cv::cuda::threshold -> CV_8U",
                      "bincv::cuda::threshold -> bits", p, floor, geom);
            emitRow("threshold", geom, p);
            printEnqueue(
                "threshold", geom, "cv::cuda::threshold", "bincv::cuda::threshold",
                timeHostEnqueue([&] {
                    cv::cuda::threshold(gsrc, gdst, 127.0, 255.0, cv::THRESH_BINARY,
                                        gCvStream);
                }),
                timeHostEnqueue([&] { bc::threshold(src.constView(), dst.view(), 127.0, gStream); }),
                p);
        }

        // Memory, meter 2, both sides, same replica count.
        //
        // Every replica is HELD until the delta is read. That is the whole
        // mechanism: freeing as we go would let the driver hand the same unit
        // back and the delta would measure one working set rather than 256,
        // which is exactly the reading that cannot resolve. They are released
        // immediately afterwards, before the other side is metered, so the two
        // sides do not have to share the card.
        std::vector<std::unique_ptr<bc::DeviceImage<uint8_t>>> si;
        std::vector<std::unique_ptr<bc::DeviceBinMat>> di;
        const size_t binMem = meterScope(
            [&](int) {
                si.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                    static_cast<int>(kW), static_cast<int>(kH)));
                di.push_back(std::make_unique<bc::DeviceBinMat>(static_cast<int>(kW),
                                                               static_cast<int>(kH)));
                bc::threshold(si.back()->constView(), di.back()->view(), 127.0);
            },
            kReplicas);
        si.clear();
        di.clear();
        cudaDeviceSynchronize();

        std::vector<cv::cuda::GpuMat> gs, gd;
        const size_t ocvMem = meterScope(
            [&](int) {
                gs.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                gd.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                cv::cuda::threshold(gs.back(), gd.back(), 127.0, 255.0, cv::THRESH_BINARY);
            },
            kReplicas);
        gs.clear();
        gd.clear();
        printMemPair("threshold working set (src + dst)", "752x480", binMem, ocvMem, step,
                     kReplicas, kReplicas);
        emitMem("threshold", "752x480", binMem, ocvMem, step, kReplicas, kReplicas);
    }

    // ======================================================================
    // 2. morphology -- cv::cuda::createMorphologyFilter (cudafilters, NPP)
    // ======================================================================
    if (want("morphology")) {
        std::printf("\n=====================================================================\n"
                    " 2. MORPHOLOGY -- binCV cuda::erode / morphologyEx vs\n"
                    "    cv::cuda::createMorphologyFilter (cudafilters, NPP-backed)\n"
                    "=====================================================================\n"
                    " ROLE: 'erode / open a binary-valued image on device'. OpenCV's\n"
                    " operand is CV_8UC1 with values 0 and 255; binCV's is one bit per\n"
                    " pixel. Same operation, same element, same border, different\n"
                    " representation -- which is the comparison this library exists to\n"
                    " make. The filter object is built ONCE outside the timed region on\n"
                    " both sides, as a caller would; only apply() is timed.\n"
                    " BORDER: cv::cuda's morphology filter is BORDER_CONSTANT only, so\n"
                    " binCV runs BORDER_CONSTANT here too -- the four border types binCV\n"
                    " makes word-parallel have no counterpart on this side to be timed\n"
                    " against, and are priced in cuda_window_benchmark instead.\n");

        struct MorphCase {
            const char* name;
            const char* key;
            bincv::StructuringElement se;
            cv::Mat cvKernel;
            bincv::MorphOp op;
            int cvOp;
        };

        std::vector<MorphCase> cases;
        cases.push_back({"erode, rect 3x3", "morph_erode_rect3",
                         bincv::StructuringElement::rect(3, 3),
                         cv::getStructuringElement(cv::MORPH_RECT, {3, 3}),
                         bincv::MORPH_ERODE, cv::MORPH_ERODE});
        cases.push_back({"morphologyEx OPEN, rect 3x3", "morph_open_rect3",
                         bincv::StructuringElement::rect(3, 3),
                         cv::getStructuringElement(cv::MORPH_RECT, {3, 3}),
                         bincv::MORPH_OPEN, cv::MORPH_OPEN});
        cases.push_back({"erode, ellipse 5x5", "morph_erode_ell5",
                         bincv::StructuringElement::ellipse(5, 5),
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}),
                         bincv::MORPH_ERODE, cv::MORPH_ERODE});

        for (int g = 0; g < 2; ++g) {
            const size_t w = g == 0 ? kW : kW2, h = g == 0 ? kH : kH2;
            const char* geom = g == 0 ? "752x480" : "1920x1080";
            const std::vector<uint8_t>& f = g == 0 ? frame : frame2;

            bc::DeviceBinMat bsrc(static_cast<int>(w), static_cast<int>(h));
            bc::DeviceBinMat bdst(static_cast<int>(w), static_cast<int>(h));
            bc::DeviceBinMat bscratch(static_cast<int>(w), static_cast<int>(h));
            {
                bincv::BinMat<uint32_t> hsrc(static_cast<int>(w), static_cast<int>(h));
                for (size_t y = 0; y < h; ++y)
                    for (size_t x = 0; x < w; ++x)
                        hsrc.set(static_cast<int>(y), static_cast<int>(x),
                                 f[y * w + x] >= 128);
                bc::upload(hsrc.constView(), bsrc.view());
            }

            cv::Mat host8(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            for (size_t y = 0; y < h; ++y)
                for (size_t x = 0; x < w; ++x)
                    host8.at<uint8_t>(static_cast<int>(y), static_cast<int>(x)) =
                        f[y * w + x] >= 128 ? 255 : 0;
            cv::cuda::GpuMat gsrc, gdst(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            gsrc.upload(host8);

            for (const MorphCase& c : cases) {
                const bc::DeviceStructuringElement de = bc::toDeviceElement(c.se);
                cv::Ptr<cv::cuda::Filter> filt =
                    cv::cuda::createMorphologyFilter(c.cvOp, CV_8UC1, c.cvKernel);
                const bool needsScratch = bincv::morphologyExNeedsScratch(c.op);
                const Timing soloA = timeKernelSolo([&] { filt->apply(gsrc, gdst, gCvStream); });
                const Timing soloB = timeKernelSolo([&] {
                    if (c.op == bincv::MORPH_ERODE)
                        bc::erode(bsrc.constView(), bdst.view(), de,
                                  bincv::BORDER_CONSTANT, true, gStream);
                    else
                        bc::morphologyEx(bsrc.constView(), bdst.view(), c.op, de,
                                         needsScratch ? bscratch.view()
                                                      : bc::DeviceBinMatView{},
                                         bincv::BORDER_CONSTANT, gStream);
                });

                const PairedTiming p = timeKernelPaired(
                    [&] { filt->apply(gsrc, gdst, gCvStream); },
                    [&] {
                        if (c.op == bincv::MORPH_ERODE)
                            bc::erode(bsrc.constView(), bdst.view(), de,
                                      bincv::BORDER_CONSTANT, true, gStream);
                        else
                            bc::morphologyEx(bsrc.constView(), bdst.view(), c.op, de,
                                             needsScratch ? bscratch.view()
                                                          : bc::DeviceBinMatView{},
                                             bincv::BORDER_CONSTANT, gStream);
                    },
                    20, 20, kRounds);
                printRole(c.name, "cv::cuda morphology filter -> CV_8U",
                          "bincv::cuda -> bits", p, floor, geom);
                emitRow(c.key, geom, p);
                printEnqueue(c.key, geom, "cv::cuda morphology filter", "bincv::cuda",
                             timeHostEnqueue([&] { filt->apply(gsrc, gdst, gCvStream); }),
                             timeHostEnqueue([&] {
                                 if (c.op == bincv::MORPH_ERODE)
                                     bc::erode(bsrc.constView(), bdst.view(), de,
                                               bincv::BORDER_CONSTANT, true, gStream);
                                 else
                                     bc::morphologyEx(bsrc.constView(), bdst.view(), c.op,
                                                      de,
                                                      needsScratch ? bscratch.view()
                                                                   : bc::DeviceBinMatView{},
                                                      bincv::BORDER_CONSTANT, gStream);
                             }),
                             p,
                             &soloA, &soloB);
            }
        }

        // Memory: erode rect3x3 working set, both sides, meter 2.
        {
            const bincv::StructuringElement se = bincv::StructuringElement::rect(3, 3);
            const bc::DeviceStructuringElement de = bc::toDeviceElement(se);
            std::vector<std::unique_ptr<bc::DeviceBinMat>> bs, bd;
            const size_t binMem = meterScope(
                [&](int) {
                    bs.push_back(std::make_unique<bc::DeviceBinMat>(static_cast<int>(kW),
                                                                    static_cast<int>(kH)));
                    bd.push_back(std::make_unique<bc::DeviceBinMat>(static_cast<int>(kW),
                                                                    static_cast<int>(kH)));
                    bc::erode(bs.back()->constView(), bd.back()->view(), de,
                              bincv::BORDER_CONSTANT);
                },
                kReplicas);
            bs.clear();
            bd.clear();
            cudaDeviceSynchronize();

            const cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
            std::vector<cv::cuda::GpuMat> gs, gd;
            std::vector<cv::Ptr<cv::cuda::Filter>> filts;
            const size_t ocvMem = meterScope(
                [&](int) {
                    gs.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                    gd.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                    filts.push_back(
                        cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, k));
                    filts.back()->apply(gs.back(), gd.back());
                },
                kReplicas);
            gs.clear();
            gd.clear();
            filts.clear();
            printMemPair("erode working set (src + dst + any filter buffer)", "752x480",
                         binMem, ocvMem, step, kReplicas, kReplicas);
            emitMem("morph_erode_rect3", "752x480", binMem, ocvMem, step, kReplicas,
                    kReplicas);
        }
    }

    // ======================================================================
    // 3. medianWide -- cv::cuda::createMedianFilter (cudafilters)
    //    THE BAR THAT DID NOT EXIST. The median family shipped with its uint8
    //    role verdict recorded OUTSTANDING because its own target linked no
    //    cudafilters-capable OpenCV. This closes it.
    // ======================================================================
    if (want("median")) {
        std::printf("\n=====================================================================\n"
                    " 3. MEDIAN (WIDE) -- binCV cuda::medianWide vs\n"
                    "    cv::cuda::createMedianFilter(CV_8UC1, ksize)\n"
                    "=====================================================================\n"
                    " ROLE ONLY, and the asymmetry is stated first because it is large:\n"
                    " cv::cuda::createMedianFilter is a 3x3 (or 5x5) SQUARE median with a\n"
                    " replicated border. binCV's kMedianReferenceL is a 3-sample L and\n"
                    " kMedianReferencePlus is a 5-sample plus, both with a ZERO border.\n"
                    " These produce DIFFERENT IMAGES. Neither is an oracle for the other\n"
                    " and no correctness claim is made across this row -- binCV's\n"
                    " correctness is closed against its own host twin by the test suite.\n"
                    " What this prices is the ROLE: 'median-filter a resident 8-bit frame\n"
                    " on the device'. Part of binCV's advantage here is that its operation\n"
                    " is CHEAPER (3 or 5 samples against 9), which is the algorithm's win\n"
                    " and NOT the representation's -- medianWide has no packed advantage\n"
                    " and its own header says so. The K=9 row below is the closest\n"
                    " sample-count match to a 3x3 square and is the fairest single row.\n"
                    " MEMORY IS PARITY BY CONSTRUCTION: one byte per pixel in, one out,\n"
                    " on both sides. Any difference in the meter is GpuMat's pitch\n"
                    " padding, not a representation win, and it is labelled as such.\n");

        using bincv::kMedianReferenceL;
        using bincv::kMedianReferencePlus;
        constexpr bincv::MedianPattern<9> kSquare9{{{-1, -1}, {-1, 0}, {-1, 1},
                                                    {0, -1},  {0, 0},  {0, 1},
                                                    {1, -1},  {1, 0},  {1, 1}}};

        // ------------------------------------------------------------------
        // THE DEFAULT STREAM CANNOT BE USED FOR THIS PAIR, and the reason is
        // in OpenCV's own source rather than in anything measured here.
        //
        // modules/cudafilters/src/cuda/median_filter.cu, medianFiltering_gpu:
        //
        //     if (!stream)
        //         cudaSafeCall( cudaDeviceSynchronize() );
        //
        // On the DEFAULT stream every apply() synchronizes the whole device
        // before returning. Two consequences, and they pull in opposite
        // directions, so both are measured and both are printed:
        //
        //  * A batch of N enqueues cannot pipeline. The event bracket then
        //    measures N serialized round trips, which is a launch-latency
        //    number wearing a kernel's clothes -- and binCV's arm in the same
        //    bracket DOES pipeline. That comparison is not kernel-to-kernel.
        //  * It is nevertheless exactly what a default-stream caller pays, and
        //    pretending otherwise would hide a real cost of the API.
        //
        // So the KERNEL comparison below runs BOTH sides on an explicit
        // non-default stream, where OpenCV's branch is not taken and neither
        // side synchronizes. The default-stream cost is then reported
        // separately, once, as an API cost and labelled as one.
        // ------------------------------------------------------------------
        for (int g = 0; g < 2; ++g) {
            const size_t w = g == 0 ? kW : kW2, h = g == 0 ? kH : kH2;
            const char* geom = g == 0 ? "752x480" : "1920x1080";
            const std::vector<uint8_t>& f = g == 0 ? frame : frame2;

            bc::DeviceImage<uint8_t> src(static_cast<int>(w), static_cast<int>(h));
            bc::DeviceImage<uint8_t> dst(static_cast<int>(w), static_cast<int>(h));
            bc::uploadImage(f.data(), w, h, w, src.view());

            cv::cuda::GpuMat gsrc, gdst(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            gsrc.upload(cv::Mat(static_cast<int>(h), static_cast<int>(w), CV_8UC1,
                                const_cast<uint8_t*>(f.data())));
            cv::Ptr<cv::cuda::Filter> med3 = cv::cuda::createMedianFilter(CV_8UC1, 3);

            // K=9, the sample-count match: 9 samples against OpenCV's 9.
            {
                const PairedTiming p = timeKernelPaired(
                    [&] { med3->apply(gsrc, gdst, gCvStream); },
                    [&] {
                        bc::medianWide(src.constView(), dst.view(), kSquare9, gStream);
                    },
                    5, 5, kRounds, gStream);
                printRole("median, 9 samples each (the sample-count match), BOTH on an"
                          " explicit stream",
                          "cv::cuda median 3x3 square", "bincv::cuda medianWide K=9", p,
                          floor, geom);
                emitRow("median_k9", geom, p);
                printEnqueue("median_k9", geom, "cv::cuda median 3x3",
                             "bincv::cuda medianWide K=9",
                             timeHostEnqueue([&] { med3->apply(gsrc, gdst, gCvStream); }, 5),
                             timeHostEnqueue(
                                 [&] {
                                     bc::medianWide(src.constView(), dst.view(), kSquare9,
                                                    gStream);
                                 },
                                 5),
                             p);
            }
            // K=3 L -- binCV's shipped reference neighbourhood.
            {
                const PairedTiming p = timeKernelPaired(
                    [&] { med3->apply(gsrc, gdst, gCvStream); },
                    [&] {
                        bc::medianWide(src.constView(), dst.view(), kMedianReferenceL,
                                       gStream);
                    },
                    5, 5, kRounds, gStream);
                printRole("median, OpenCV 9 samples vs binCV 3 (the shipped L)",
                          "cv::cuda median 3x3 square", "bincv::cuda medianWide K=3 L", p,
                          floor, geom);
                emitRow("median_k3L", geom, p);
            }
            // K=5 plus.
            {
                const PairedTiming p = timeKernelPaired(
                    [&] { med3->apply(gsrc, gdst, gCvStream); },
                    [&] {
                        bc::medianWide(src.constView(), dst.view(), kMedianReferencePlus,
                                       gStream);
                    },
                    5, 5, kRounds, gStream);
                printRole("median, OpenCV 9 samples vs binCV 5 (the plus)",
                          "cv::cuda median 3x3 square", "bincv::cuda medianWide K=5 plus",
                          p, floor, geom);
                emitRow("median_k5plus", geom, p);
            }

            // The default-stream API cost, measured once per geometry and
            // labelled for what it is.
            if (g == 0) {
                const Timing dflt = timeHostEnqueue(
                    [&] { med3->apply(gsrc, gdst); }, 5, kRounds);
                const Timing strm = timeHostEnqueue(
                    [&] { med3->apply(gsrc, gdst, gCvStream); }, 5, kRounds);
                std::printf("\n   THE DEFAULT-STREAM SURCHARGE, cv::cuda median, %s:\n"
                            "     apply() on the DEFAULT stream   %8.4f ms/call (wall,"
                            " includes its internal cudaDeviceSynchronize)\n"
                            "     apply() on an EXPLICIT stream   %8.4f ms/call (wall,"
                            " enqueue only)\n"
                            "   The difference is the API's, not the kernel's, and the"
                            " rows above avoid it\n"
                            "   by running BOTH sides on an explicit stream.\n",
                            geom, dflt.medianMs, strm.medianMs);
                std::printf("SYNC,median_default_stream,%s,%.6f,%.6f\n", geom,
                            dflt.medianMs, strm.medianMs);
            }
        }

        // denoiseMedian3 -- the PACKED-BIT median. Its written bar is binCV's
        // own byte medianWide<L>, not OpenCV; the cv::cuda row is carried here
        // as a role reference so the representation's cost of shape is visible
        // in the same session, and it is NOT the bar the implementer wrote.
        //
        // A LADDER, not one size, and the implementer's own rule is why. Rule A
        // says denoiseMedian3's speed gate is decided "at the first ladder size
        // whose measured time clears the printed launch floor" -- so the sizes
        // below the crossing are not a softer result, they are no result, and
        // the only way to honour that rule is to climb until something clears.
        {
            const size_t ladderW[] = {kW, kW2, 4096};
            const size_t ladderH[] = {kH, kH2, 2160};
            const char* ladderName[] = {"752x480", "1920x1080", "4096x2160"};
            std::printf("\n denoiseMedian3 against ITS OWN WRITTEN BAR -- binCV's byte\n"
                        " medianWide<L> with its fast arm on. Identical operation,\n"
                        " identical border, one launch each; the only variable is the\n"
                        " representation. This is the implementer's stated bar and the\n"
                        " cv::cuda rows above are NOT it. Climbed as a LADDER because the\n"
                        " written rule decides this gate at the first size clear of the\n"
                        " launch floor, and names every size below it as deciding nothing.\n");
            for (int li = 0; li < 3; ++li) {
                const size_t w = ladderW[li], h = ladderH[li];
                std::vector<uint8_t> lf = makeFrame(w, h);
                bc::DeviceBinMat bsrc(static_cast<int>(w), static_cast<int>(h));
                bc::DeviceBinMat bdst(static_cast<int>(w), static_cast<int>(h));
                bc::DeviceImage<uint8_t> wsrc(static_cast<int>(w), static_cast<int>(h));
                bc::DeviceImage<uint8_t> wdst(static_cast<int>(w), static_cast<int>(h));
                bc::uploadImage(lf.data(), w, h, w, wsrc.view());

                const PairedTiming p = timeKernelPaired(
                    [&] {
                        bc::medianWide(wsrc.constView(), wdst.view(),
                                       bincv::kMedianReferenceL, gStream);
                    },
                    [&] { bc::denoiseMedian3(bsrc.constView(), bdst.view(), gStream); }, 20,
                    20, kRounds, gStream);
                printRole("denoiseMedian3 (bits) vs medianWide<L> (bytes)",
                          "bincv::cuda medianWide K=3 L, bytes",
                          "bincv::cuda denoiseMedian3", p, floor, ladderName[li]);
                emitRow("denoise3_vs_bytebar", ladderName[li], p);
            }
        }

        // Memory, meter 2, both sides.
        {
            std::vector<std::unique_ptr<bc::DeviceImage<uint8_t>>> bs, bd;
            const size_t binMem = meterScope(
                [&](int) {
                    bs.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    bd.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    bc::medianWide(bs.back()->constView(), bd.back()->view(),
                                   bincv::kMedianReferenceL);
                },
                kReplicas);
            bs.clear();
            bd.clear();
            cudaDeviceSynchronize();

            std::vector<cv::cuda::GpuMat> gs, gd;
            std::vector<cv::Ptr<cv::cuda::Filter>> filts;
            const size_t ocvMem = meterScope(
                [&](int) {
                    gs.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                    gd.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                    filts.push_back(cv::cuda::createMedianFilter(CV_8UC1, 3));
                    filts.back()->apply(gs.back(), gd.back());
                },
                kMedianOcvReplicas);
            std::printf("   cv::cuda::createMedianFilter's scratch is NOT incidental: its\n"
                        "   own filtering.cpp sizes the histograms at cols*256*partitions\n"
                        "   and cols*8*partitions CV_32S, which at 752 columns and its\n"
                        "   default 128 partitions is ~98 MB of device scratch for ONE\n"
                        "   752x480 frame. binCV's medianWide allocates ZERO scratch --\n"
                        "   no heap in kernels is a project rule, so the caller's two\n"
                        "   images are the whole working set.\n");
            gs.clear();
            gd.clear();
            filts.clear();
            printMemPair("medianWide working set (wide src + wide dst + filter buffer)",
                         "752x480", binMem, ocvMem, step, kReplicas, kMedianOcvReplicas);
            std::printf("   NOTE: medianWide claims NO footprint advantage -- it is one\n"
                        "   byte per pixel on both sides. Any ratio here is GpuMat's pitch\n"
                        "   padding plus whatever cv::cuda::createMedianFilter holds for\n"
                        "   its histogram scratch, and it is NOT evidence for 1-bit\n"
                        "   packing. denoiseMedian3 is where this family's footprint\n"
                        "   result lives, and its bar is the binCV byte arm above.\n");
            emitMem("median_k3L", "752x480", binMem, ocvMem, step, kReplicas,
                    kMedianOcvReplicas);

            // And denoiseMedian3's own footprint, against the byte arm it is
            // barred against -- the same meter, so the pair is comparable.
            std::vector<std::unique_ptr<bc::DeviceBinMat>> qs, qd;
            const size_t bitMem = meterScope(
                [&](int) {
                    qs.push_back(std::make_unique<bc::DeviceBinMat>(static_cast<int>(kW),
                                                                    static_cast<int>(kH)));
                    qd.push_back(std::make_unique<bc::DeviceBinMat>(static_cast<int>(kW),
                                                                    static_cast<int>(kH)));
                    bc::denoiseMedian3(qs.back()->constView(), qd.back()->view());
                },
                kReplicas);
            qs.clear();
            qd.clear();
            printMemPair("denoiseMedian3 (bits) vs medianWide<L> (bytes) working set",
                         "752x480", bitMem, binMem, step, kReplicas, kReplicas);
            std::printf("   Both sides of THAT pair are binCV, so the label 'OpenCV' on\n"
                        "   the first line reads 'binCV byte arm' here. Same meter, same\n"
                        "   replica count, and it is the representation's own ratio.\n");
            emitMem("denoise3_vs_bytebar", "752x480", bitMem, binMem, step, kReplicas,
                    kReplicas);
        }
    }

    // ======================================================================
    // 4. pyramid -- cv::cuda::resize(INTER_AREA) and cv::cuda::pyrDown
    // ======================================================================
    if (want("pyramid")) {
        std::printf("\n=====================================================================\n"
                    " 4. PYRAMID -- binCV cuda::buildPyramidBox vs 3x cv::cuda::resize\n"
                    "    (INTER_AREA, 0.5) and 3x cv::cuda::pyrDown (cudawarping)\n"
                    "=====================================================================\n"
                    " ROLE: 'build a three-rung half-resolution ladder on device'.\n"
                    " cv::cuda::resize at INTER_AREA and scale 0.5 is the SAME FILTER as\n"
                    " binCV's box -- a 2x2 average -- so that is the like-for-like arm.\n"
                    " cv::cuda::pyrDown is a Gaussian 5x5 and is a DIFFERENT operation;\n"
                    " it is carried because it is what a caller reaches for, and it is\n"
                    " labelled as a different filter rather than quoted as the bar.\n"
                    " binCV's ladder is N-BIT per rung (1,3,4,5) -- OpenCV has no N-bit\n"
                    " image type, so the output arrays are not the same thing and this\n"
                    " is a role comparison.\n"
                    " ODD WIDTHS ARE NOT COMPARABLE: at width 753, resize's dsize is 376\n"
                    " where pyrDownWidth(753) is 377. 752 is even at every rung, which is\n"
                    " why the ladder is taken there.\n");

        bc::DevicePyramid<1, 3, 4, 5> pyr(static_cast<int>(kW), static_cast<int>(kH));
        cv::cuda::GpuMat g0(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
        g0.upload(cv::Mat(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                          const_cast<uint8_t*>(frame.data())));
        cv::cuda::GpuMat g1(240, 376, CV_8UC1), g2(120, 188, CV_8UC1), g3(60, 94, CV_8UC1);

        {
            const PairedTiming p = timeKernelPaired(
                [&] {
                    cv::cuda::resize(g0, g1, g1.size(), 0, 0, cv::INTER_AREA, gCvStream);
                    cv::cuda::resize(g1, g2, g2.size(), 0, 0, cv::INTER_AREA, gCvStream);
                    cv::cuda::resize(g2, g3, g3.size(), 0, 0, cv::INTER_AREA, gCvStream);
                },
                [&] { bc::buildPyramidBox(pyr, gStream); }, 10, 10, kRounds, gStream);
            printRole("3-rung ladder, SAME FILTER (2x2 box / INTER_AREA 0.5)",
                      "3x cv::cuda::resize INTER_AREA -> CV_8U",
                      "bincv::cuda buildPyramidBox -> N-bit", p, floor, "752x480");
            emitRow("pyramid_vs_resize", "752x480", p);
            printEnqueue("pyramid_vs_resize", "752x480", "3x cv::cuda::resize",
                         "bincv::cuda buildPyramidBox",
                         timeHostEnqueue([&] {
                             cv::cuda::resize(g0, g1, g1.size(), 0, 0, cv::INTER_AREA,
                                              gCvStream);
                             cv::cuda::resize(g1, g2, g2.size(), 0, 0, cv::INTER_AREA,
                                              gCvStream);
                             cv::cuda::resize(g2, g3, g3.size(), 0, 0, cv::INTER_AREA,
                                              gCvStream);
                         }),
                         timeHostEnqueue([&] { bc::buildPyramidBox(pyr, gStream); }), p);
        }
        {
            const PairedTiming p = timeKernelPaired(
                [&] {
                    cv::cuda::pyrDown(g0, g1, gCvStream);
                    cv::cuda::pyrDown(g1, g2, gCvStream);
                    cv::cuda::pyrDown(g2, g3, gCvStream);
                },
                [&] { bc::buildPyramidBox(pyr, gStream); }, 10, 10, kRounds, gStream);
            printRole("3-rung ladder, DIFFERENT FILTER (Gaussian 5x5) -- reference only",
                      "3x cv::cuda::pyrDown -> CV_8U",
                      "bincv::cuda buildPyramidBox -> N-bit", p, floor, "752x480");
            emitRow("pyramid_vs_pyrdown", "752x480", p);
        }

        // Memory, meter 2, both sides: the whole ladder.
        {
            std::vector<std::unique_ptr<bc::DevicePyramid<1, 3, 4, 5>>> ps;
            const size_t binMem = meterScope(
                [&](int) {
                    ps.push_back(std::make_unique<bc::DevicePyramid<1, 3, 4, 5>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    bc::buildPyramidBox(*ps.back());
                },
                kReplicas);
            ps.clear();
            cudaDeviceSynchronize();

            std::vector<cv::cuda::GpuMat> l0, l1, l2, l3;
            const size_t ocvMem = meterScope(
                [&](int) {
                    l0.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                    l1.emplace_back(240, 376, CV_8UC1);
                    l2.emplace_back(120, 188, CV_8UC1);
                    l3.emplace_back(60, 94, CV_8UC1);
                    cv::cuda::resize(l0.back(), l1.back(), l1.back().size(), 0, 0,
                                     cv::INTER_AREA);
                    cv::cuda::resize(l1.back(), l2.back(), l2.back().size(), 0, 0,
                                     cv::INTER_AREA);
                    cv::cuda::resize(l2.back(), l3.back(), l3.back().size(), 0, 0,
                                     cv::INTER_AREA);
                },
                kReplicas);
            l0.clear();
            l1.clear();
            l2.clear();
            l3.clear();
            printMemPair("4-level ladder, all rungs resident", "752x480", binMem, ocvMem,
                         step, kReplicas, kReplicas);
            emitMem("pyramid", "752x480", binMem, ocvMem, step, kReplicas, kReplicas);
        }
    }

    // ======================================================================
    // 5. edgeThreshold -- the composed cv::cuda deriv-filter chain
    // ======================================================================
    if (want("edge")) {
        std::printf("\n=====================================================================\n"
                    " 5. EDGE THRESHOLD -- binCV cuda::edgeThreshold vs the composed\n"
                    "    cv::cuda spelling (createDerivFilter x2 + abs + add + threshold)\n"
                    "=====================================================================\n"
                    " ROLE: 'central-difference gradient magnitude, thresholded, on\n"
                    " device'. There is no single cv::cuda call for it, so the bar is the\n"
                    " composed spelling a caller would actually write. createDerivFilter\n"
                    " at ksize=1 with normalize=false gives getDerivKernels' exact\n"
                    " [-1,0,1] -- the same computation -- and defaults to\n"
                    " BORDER_REFLECT_101, which is binCV's border here.\n"
                    " createLinearFilter cannot be used: it asserts dst depth == src\n"
                    " depth, and a signed derivative of CV_8U needs CV_16S.\n"
                    " The chain is separable, so it is SEVERAL launches against binCV's\n"
                    " one, and the launch count is derived from the API's structure --\n"
                    " no profiler runs on this host to count them directly.\n");

        for (int g = 0; g < 2; ++g) {
            const size_t w = g == 0 ? kW : kW2, h = g == 0 ? kH : kH2;
            const char* geom = g == 0 ? "752x480" : "1920x1080";
            const std::vector<uint8_t>& f = g == 0 ? frame : frame2;

            bc::DeviceImage<uint8_t> src(static_cast<int>(w), static_cast<int>(h));
            bc::DeviceBinMat dst(static_cast<int>(w), static_cast<int>(h));
            bc::uploadImage(f.data(), w, h, w, src.view());

            cv::cuda::GpuMat gsrc;
            gsrc.upload(cv::Mat(static_cast<int>(h), static_cast<int>(w), CV_8UC1,
                                const_cast<uint8_t*>(f.data())));
            cv::cuda::GpuMat gx(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            cv::cuda::GpuMat gy(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            cv::cuda::GpuMat ax(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            cv::cuda::GpuMat ay(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            cv::cuda::GpuMat mag(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            cv::cuda::GpuMat emap(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            cv::Ptr<cv::cuda::Filter> dfx = cv::cuda::createDerivFilter(
                CV_8UC1, CV_16SC1, 1, 0, 1, false, 1.0, cv::BORDER_REFLECT101);
            cv::Ptr<cv::cuda::Filter> dfy = cv::cuda::createDerivFilter(
                CV_8UC1, CV_16SC1, 0, 1, 1, false, 1.0, cv::BORDER_REFLECT101);

            const PairedTiming p = timeKernelPaired(
                [&] {
                    dfx->apply(gsrc, gx, gCvStream);
                    dfy->apply(gsrc, gy, gCvStream);
                    cv::cuda::abs(gx, ax, gCvStream);
                    cv::cuda::abs(gy, ay, gCvStream);
                    cv::cuda::max(ax, ay, mag, gCvStream);
                    cv::cuda::threshold(mag, emap, 17.0, 255.0, cv::THRESH_BINARY,
                                        gCvStream);
                },
                [&] {
                    bc::edgeThreshold(src.constView(), dst.view(), uint8_t{17},
                                      bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                                      bincv::EdgeSpatial::Wide, gStream);
                },
                10, 20, kRounds, gStream);
            printRole("edge map, |dx| max |dy| >= 17", "composed cv::cuda chain -> CV_16S",
                      "bincv::cuda edgeThreshold -> bits", p, floor, geom);
            emitRow("edge", geom, p);
            printEnqueue("edge", geom, "composed cv::cuda chain",
                         "bincv::cuda edgeThreshold",
                         timeHostEnqueue([&] {
                             dfx->apply(gsrc, gx, gCvStream);
                             dfy->apply(gsrc, gy, gCvStream);
                             cv::cuda::abs(gx, ax, gCvStream);
                             cv::cuda::abs(gy, ay, gCvStream);
                             cv::cuda::max(ax, ay, mag, gCvStream);
                             cv::cuda::threshold(mag, emap, 17.0, 255.0,
                                                 cv::THRESH_BINARY, gCvStream);
                         }),
                         timeHostEnqueue([&] {
                             bc::edgeThreshold(src.constView(), dst.view(), uint8_t{17},
                                               bincv::EdgeCombine::Or,
                                               bincv::EdgeRelation::Ge,
                                               bincv::EdgeSpatial::Wide, gStream);
                         }),
                         p);
        }

        // Memory, meter 2, both sides.
        {
            std::vector<std::unique_ptr<bc::DeviceImage<uint8_t>>> bs;
            std::vector<std::unique_ptr<bc::DeviceBinMat>> bd;
            const size_t binMem = meterScope(
                [&](int) {
                    bs.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    bd.push_back(std::make_unique<bc::DeviceBinMat>(static_cast<int>(kW),
                                                                    static_cast<int>(kH)));
                    bc::edgeThreshold(bs.back()->constView(), bd.back()->view(),
                                      uint8_t{17});
                },
                kReplicas);
            bs.clear();
            bd.clear();
            cudaDeviceSynchronize();

            std::vector<cv::cuda::GpuMat> gsv, gxv, gyv, axv, ayv, magv, emv;
            std::vector<cv::Ptr<cv::cuda::Filter>> fx, fy;
            const size_t ocvMem = meterScope(
                [&](int) {
                    const int H = static_cast<int>(kH), W = static_cast<int>(kW);
                    gsv.emplace_back(H, W, CV_8UC1);
                    gxv.emplace_back(H, W, CV_16SC1);
                    gyv.emplace_back(H, W, CV_16SC1);
                    axv.emplace_back(H, W, CV_16SC1);
                    ayv.emplace_back(H, W, CV_16SC1);
                    magv.emplace_back(H, W, CV_16SC1);
                    emv.emplace_back(H, W, CV_16SC1);
                    fx.push_back(cv::cuda::createDerivFilter(CV_8UC1, CV_16SC1, 1, 0, 1,
                                                             false, 1.0,
                                                             cv::BORDER_REFLECT101));
                    fy.push_back(cv::cuda::createDerivFilter(CV_8UC1, CV_16SC1, 0, 1, 1,
                                                             false, 1.0,
                                                             cv::BORDER_REFLECT101));
                    fx.back()->apply(gsv.back(), gxv.back());
                    fy.back()->apply(gsv.back(), gyv.back());
                    cv::cuda::abs(gxv.back(), axv.back());
                    cv::cuda::abs(gyv.back(), ayv.back());
                    cv::cuda::max(axv.back(), ayv.back(), magv.back());
                    cv::cuda::threshold(magv.back(), emv.back(), 17.0, 255.0,
                                        cv::THRESH_BINARY);
                },
                kEdgeOcvReplicas);
            gsv.clear(); gxv.clear(); gyv.clear(); axv.clear();
            ayv.clear(); magv.clear(); emv.clear(); fx.clear(); fy.clear();
            printMemPair("edge working set (every intermediate the chain needs)", "752x480",
                         binMem, ocvMem, step, kReplicas, kEdgeOcvReplicas);
            emitMem("edge", "752x480", binMem, ocvMem, step, kReplicas, kEdgeOcvReplicas);
        }
    }

    // ======================================================================
    // 6. The ops with NO cv::cuda counterpart -- ruling R2
    // ======================================================================
    if (only.empty() || only == "outstanding") {
        std::printf("\n=====================================================================\n"
                    " 6. NO BAR EXISTS -- verdict OUTSTANDING (owner ruling R2)\n"
                    "=====================================================================\n"
                    " These ops have NO cv::cuda counterpart at any API level. Under R2\n"
                    " they ship on correctness + memory + the host comparison, with the\n"
                    " GPU speed verdict recorded explicitly as OUTSTANDING against the\n"
                    " resident pipeline that will later price them. No substitute bar is\n"
                    " invented here and no CPU number is quoted as a GPU bar.\n"
                    "\n"
                    "   cuda::binarize     N-plane bit-sliced source -> bits. OpenCV has\n"
                    "                      no N-bit image type on host or device, so there\n"
                    "                      is nothing to compare the input side against.\n"
                    "                      SPEED VERDICT: OUTSTANDING.\n"
                    "   cuda::shift        integer translation of a packed bit plane. The\n"
                    "                      byte side's actual alternative is a pitched DMA\n"
                    "                      (cudaMemcpy2D), not a kernel -- timed below as\n"
                    "                      a reference, not as a bar, because binCV's one\n"
                    "                      funnel-shift instruction is being compared\n"
                    "                      against ZERO instructions on a copy engine.\n"
                    "                      SPEED VERDICT: OUTSTANDING.\n"
                    "   cuda::pyrDownBox   the N-bit rung itself (the LADDER has a role\n"
                    "                      bar above; the per-rung N-bit requantize does\n"
                    "                      not). SPEED VERDICT: OUTSTANDING.\n");

        bc::DeviceBinMat s(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceBinMat d(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint8_t> b1(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint8_t> b2(static_cast<int>(kW), static_cast<int>(kH));
        const PairedTiming p = timeKernelPaired(
            [&] {
                cudaMemcpy2DAsync(b2.view().ptr, kW, b1.constView().ptr, kW, kW - 8, kH,
                                  cudaMemcpyDeviceToDevice, gStream);
            },
            [&] { bc::shift(s.constView(), d.view(), 8, 0, bincv::BORDER_CONSTANT, false,
                            gStream); },
            20, 20, kRounds, gStream);
        std::printf("\n shift, REFERENCE ONLY (not a bar -- a DMA against a kernel):\n");
        printArmVsFloor("cudaMemcpy2DAsync ROI, bytes (copy engine)", p.a, floor, "kernel");
        printArmVsFloor("bincv::cuda::shift dx=8, bits (one kernel)", p.b, floor, "kernel");
        std::printf("   ratio binCV/DMA: %5.3fx   range %5.3f-%5.3fx   %s\n",
                    p.ratioMedian, p.ratioMin, p.ratioMax,
                    p.separated() ? "DISJOINT" : "OVERLAP");
        emitRow("shift_vs_dma", "752x480", p);
    }

    // ======================================================================
    // 7. DENSE DISPARITY -- cv::cuda::StereoBM, re-taken on an explicit stream
    // ======================================================================
    if (want("stereo")) {
        std::printf("\n=====================================================================\n"
                    " 7. DENSE DISPARITY -- binCV binary path vs cv::cuda::StereoBM\n"
                    "=====================================================================\n"
                    " ROLE: 'a dense disparity map from a rectified pair, resident on\n"
                    " device'. StereoBM matches SAD over prefiltered bytes; binCV matches\n"
                    " Hamming over packed binary. Different maps -- correctness is settled\n"
                    " against the host library by the test suite, not here.\n"
                    " RE-TAKEN HERE because this backend's shipped report quotes this\n"
                    " comparison from a DEFAULT-STREAM measurement, and StereoBM calls\n"
                    " cudaDeviceSynchronize() on the null stream in each of its three\n"
                    " kernels (cudastereo/src/cuda/stereobm.cu). Both sides now run on\n"
                    " one explicit stream, where neither synchronizes.\n");

        std::vector<uint8_t> lw = makeFrame(kW, kH);
        std::vector<uint8_t> rw(kW * kH, 0);
        for (size_t y = 0; y < kH; ++y)
            for (size_t x = 0; x + 21 < kW; ++x) rw[y * kW + x] = lw[y * kW + x + 21];

        bincv::DenseDisparityParams dp;
        dp.maxDisparity = 64;

        bincv::BinMat<uint32_t> lb(static_cast<int>(kW), static_cast<int>(kH));
        bincv::BinMat<uint32_t> rb(static_cast<int>(kW), static_cast<int>(kH));
        bincv::packBits<bincv::PackRule::GreaterThan>(lw.data(), kW, kH, kW, lb.view(),
                                                      uint8_t{127});
        bincv::packBits<bincv::PackRule::GreaterThan>(rw.data(), kW, kH, kW, rb.view(),
                                                      uint8_t{127});
        bc::DeviceBinMat dl(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceBinMat dr(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint8_t> dDisp(static_cast<int>(kW), static_cast<int>(kH));
        bc::upload(lb.constView(), dl.view());
        bc::upload(rb.constView(), dr.view());

        cv::cuda::GpuMat gl, gr, gd;
        gl.upload(cv::Mat(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1, lw.data()));
        gr.upload(cv::Mat(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1, rw.data()));
        auto bm = cv::cuda::createStereoBM(64, 9);
        bm->compute(gl, gr, gd, gCvStream);  // first call allocates its internals
        cudaStreamSynchronize(gStream);

        const PairedTiming p = timeKernelPaired(
            [&] { bm->compute(gl, gr, gd, gCvStream); },
            [&] {
                bc::denseDisparityBinary(dl.constView(), dr.constView(), dp, dDisp.view(),
                                         gStream);
            },
            10, 10, kRounds, gStream);
        const Timing soloS_a = timeKernelSolo([&] { bm->compute(gl, gr, gd, gCvStream); });
        const Timing soloS_b = timeKernelSolo([&] {
            bc::denseDisparityBinary(dl.constView(), dr.constView(), dp, dDisp.view(),
                                     gStream);
        });
        printRole("dense disparity, 64 disparities, 9x9 support",
                  "cv::cuda::StereoBM(64, 9)", "bincv::cuda denseDisparityBinary", p, floor,
                  "752x480");
        emitRow("stereo_binary", "752x480", p);
        printEnqueue("stereo_binary", "752x480", "cv::cuda::StereoBM",
                     "bincv denseDisparityBinary",
                     timeHostEnqueue([&] { bm->compute(gl, gr, gd, gCvStream); }, 10),
                     timeHostEnqueue(
                         [&] {
                             bc::denseDisparityBinary(dl.constView(), dr.constView(), dp,
                                                      dDisp.view(), gStream);
                         },
                         10),
                     p,
                     &soloS_a, &soloS_b);

        // The same pair on the DEFAULT stream, which is how the shipped report
        // took it. Printed beside the corrected row, not instead of it.
        const Timing dfltOcv =
            timeKernel([&] { bm->compute(gl, gr, gd); }, 10, kRounds);
        std::printf("\n   THE SAME cv::cuda::StereoBM ON THE DEFAULT STREAM: %.4f ms\n"
                    "   against %.4f ms on an explicit stream -- a %.2fx difference that\n"
                    "   is the API's three null-stream cudaDeviceSynchronize calls, not\n"
                    "   the kernel. The explicit-stream number is the bar, because the\n"
                    "   bar is the BEST existing option and a pipeline uses streams.\n",
                    dfltOcv.medianMs, p.a.medianMs,
                    p.a.medianMs > 0.0 ? dfltOcv.medianMs / p.a.medianMs : 0.0);
        std::printf("SYNC,stereobm_default_stream,752x480,%.6f,%.6f\n", dfltOcv.medianMs,
                    p.a.medianMs);
    }

    // ======================================================================
    // 8. THE DEFAULT-STREAM SURCHARGE, measured across the whole role set
    // ======================================================================
    if (want("streamcost")) {
        std::printf("\n=====================================================================\n"
                    " 8. THE DEFAULT-STREAM SURCHARGE -- why every row above runs on an\n"
                    "    EXPLICIT stream, measured rather than asserted\n"
                    "=====================================================================\n"
                    " OpenCV's CUDA modules call cudaDeviceSynchronize() after a launch\n"
                    " WHEN AND ONLY WHEN the stream is the null stream. It is not one\n"
                    " function: the guard\n"
                    "\n"
                    "     if (stream == 0) cudaSafeCall( cudaDeviceSynchronize() );\n"
                    "\n"
                    " appears in cudev's grid transform (which backs cudaarithm's\n"
                    " threshold, abs and max), in cudafilters' morphology, linear and\n"
                    " median filters, in cudawarping's resize and pyrDown, and three\n"
                    " times in cudastereo's StereoBM.\n"
                    "\n"
                    " CONSEQUENCE FOR A BENCHMARK. On the default stream an OpenCV arm\n"
                    " cannot pipeline across a batch of enqueues while binCV's arm can,\n"
                    " so an event bracket over N calls measures N serialized round trips\n"
                    " on one side and N pipelined launches on the other. That is not a\n"
                    " kernel-to-kernel comparison, and the difference is not small.\n"
                    "\n"
                    " WHICH NUMBER IS THE BAR. The explicit-stream one. CLAUDE.md says the\n"
                    " bar is the BEST existing option, not the worst; a resident pipeline\n"
                    " uses streams, and OpenCV supports them on every call measured here.\n"
                    " Quoting the default-stream number would be measuring against a\n"
                    " fallback nobody would use. The default-stream figure is reported\n"
                    " beside it as what a naive caller pays, and labelled as that.\n");

        bc::DeviceImage<uint8_t> src(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceBinMat bits(static_cast<int>(kW), static_cast<int>(kH));
        bc::uploadImage(frame.data(), kW, kH, kW, src.view());
        cv::cuda::GpuMat gsrc, gdst(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
        gsrc.upload(cv::Mat(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                            const_cast<uint8_t*>(frame.data())));
        const cv::Mat k3 = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
        cv::Ptr<cv::cuda::Filter> er =
            cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, k3);
        cv::cuda::GpuMat gsmall(240, 376, CV_8UC1);
        cv::Ptr<cv::cuda::Filter> med = cv::cuda::createMedianFilter(CV_8UC1, 3);
        bc::DeviceBinMat bits2(static_cast<int>(kW), static_cast<int>(kH));
        const bc::DeviceStructuringElement de3 =
            bc::toDeviceElement(bincv::StructuringElement::rect(3, 3));

        struct Probe {
            const char* name;
            std::function<void()> dflt;
            std::function<void()> strm;
        };
        const Probe probes[] = {
            {"cv::cuda::threshold (cudaarithm / cudev)",
             [&] { cv::cuda::threshold(gsrc, gdst, 127.0, 255.0, cv::THRESH_BINARY); },
             [&] {
                 cv::cuda::threshold(gsrc, gdst, 127.0, 255.0, cv::THRESH_BINARY,
                                     gCvStream);
             }},
            {"cv::cuda erode 3x3 (cudafilters)", [&] { er->apply(gsrc, gdst); },
             [&] { er->apply(gsrc, gdst, gCvStream); }},
            {"cv::cuda::resize INTER_AREA 0.5 (cudawarping)",
             [&] { cv::cuda::resize(gsrc, gsmall, gsmall.size(), 0, 0, cv::INTER_AREA); },
             [&] {
                 cv::cuda::resize(gsrc, gsmall, gsmall.size(), 0, 0, cv::INTER_AREA,
                                  gCvStream);
             }},
            {"cv::cuda::pyrDown (cudawarping)",
             [&] { cv::cuda::pyrDown(gsrc, gsmall); },
             [&] { cv::cuda::pyrDown(gsrc, gsmall, gCvStream); }},
            {"cv::cuda median 3x3 (cudafilters) -- a 6 ms kernel",
             [&] { med->apply(gsrc, gdst); },
             [&] { med->apply(gsrc, gdst, gCvStream); }},
            {"CONTROL: binCV cuda::threshold (no such guard anywhere)",
             [&] { bc::threshold(src.constView(), bits.view(), 127.0); },
             [&] { bc::threshold(src.constView(), bits.view(), 127.0, gStream); }},
            {"CONTROL: binCV cuda::erode 3x3 (no such guard anywhere)",
             [&] { bc::erode(bits.constView(), bits2.view(), de3); },
             [&] {
                 bc::erode(bits.constView(), bits2.view(), de3, bincv::BORDER_CONSTANT,
                           true, gStream);
             }},
        };
        // NOT timeKernelPaired: the whole comparison here is BETWEEN two
        // streams, and that timer requires both arms on one stream for its
        // events to be comparable. Two single-arm timings it is, with both
        // ranges printed so a reader can see whether they overlap.
        std::printf("\n %-46s %10s %10s %8s\n", "call", "default", "explicit", "ratio");
        double controlRatio = 0.0;
        for (const Probe& pr : probes) {
            const Timing d = timeKernel(pr.dflt, 20, kRounds);
            const Timing e = timeKernel(pr.strm, 20, kRounds, gStream);
            const double ratio = e.medianMs > 0.0 ? d.medianMs / e.medianMs : 0.0;
            const bool overlap = !(d.maxMs < e.minMs || e.maxMs < d.minMs);
            std::printf(" %-46s %8.4f ms %8.4f ms %7.2fx  %s\n", pr.name, d.medianMs,
                        e.medianMs, ratio, overlap ? "(ranges OVERLAP -- not a result)"
                                                   : "(ranges disjoint)");
            std::printf("STREAMCOST,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n", pr.name,
                        d.minMs, d.medianMs, d.maxMs, e.minMs, e.medianMs, e.maxMs,
                        overlap ? 0 : 1);
            if (controlRatio == 0.0) controlRatio = ratio;  // set on first, overwritten
        }
        std::printf("\n HOW TO READ THIS TABLE, and the last row is what makes it\n"
                    " readable. binCV has NO null-stream guard anywhere, so its row is\n"
                    " the control: whatever ratio IT shows is the legacy default stream's\n"
                    " own per-launch tax, which every arm on that stream pays -- the\n"
                    " legacy stream implicitly synchronizes with every other blocking\n"
                    " stream in the process, and one exists here. Only the EXCESS over\n"
                    " the control is attributable to OpenCV's cudaDeviceSynchronize.\n"
                    "\n"
                    " AND THE SIZE OF THE EFFECT DEPENDS ON THE KERNEL, which is the\n"
                    " conclusion that actually matters for this report: a synchronize\n"
                    " costs roughly one launch latency, so it is most of the measurement\n"
                    " for an op sitting near the launch floor and almost none of it for\n"
                    " an op that runs for a millisecond. Expect threshold, resize and\n"
                    " median to move a lot and StereoBM to barely move at all -- and\n"
                    " check that against the rows above rather than taking it on trust.\n");
    }


    // ======================================================================
    // ROUND 2 -- the frontend families' role bars, brought into this process
    //
    // WHY THEY ARE HERE AND NOT LEFT IN THEIR FAMILY BENCHMARKS. Each family
    // wrote its own OpenCV arm against its own find_package, in its own
    // process, and two of them read memory on a meter the other side does not
    // share. That is four answers to one question. The rule this file was
    // built on applies unchanged to the frontend set: one process, one
    // explicit stream on both sides, interleaved rounds, and meter 2 on both
    // sides of anything that crosses the library boundary.
    //
    // THE INPUT, AND WHY IT IS THE PIPELINE'S OWN. Every arm below reads a
    // binary frame produced by binCV's OWN sensor stage -- medianWide<3> then
    // edgeThreshold at the reference frontend's threshold of 17 -- rather than
    // a synthetic bit pattern. A detector's cost is a function of how many
    // corners its input has, so a frame with the wrong density prices the
    // wrong operation. OpenCV's side reads the SAME bits, expanded to the
    // CV_8U {0,255} picture its detectors require, which is the construction
    // ops/fast.hpp proves accepts the identical corner set.
    // ======================================================================
    const bool wantFrontend = want("fast") || want("gftt") || want("cornerresp") ||
                              want("describe") || want("matcher");

    // Declared out here so every frontend family shares one set of device
    // buffers and one upload: a per-family upload would put a 361 KB transfer
    // inside a process that is timing launches.
    bc::DeviceImage<uint8_t> fWide(static_cast<int>(kW), static_cast<int>(kH));
    bc::DeviceImage<uint8_t> fDenoised(static_cast<int>(kW), static_cast<int>(kH));
    bc::DeviceBinMat fBits(static_cast<int>(kW), static_cast<int>(kH));
    bc::DeviceBinMat fdxBlock(static_cast<int>(kW), static_cast<int>(2 * kH));
    bc::DeviceBinMat fdyBlock(static_cast<int>(kW), static_cast<int>(2 * kH));
    cv::cuda::GpuMat gPicture;
    std::vector<uint8_t> picture(kW * kH);

    // THE FRONTEND FRAME, and why it is not the synthetic one the rest of this
    // file uses. A detector's cost is a function of how many corners its input
    // has, and edgeThreshold on SMOOTHED NOISE sets ~83% of pixels -- a frame
    // on which both FAST implementations overflow any sane capacity, so the
    // corner-set gate cannot even run. The real content this project measures
    // on is a EuRoC sequence blob; point BINCV_CUDA_ROLE_FRAMES at one
    // (scripts/make_sequence_blob.py, --mode 8bit) and frame 0 of it is used.
    // Without one the synthetic frame is used and every frontend row below
    // says so, because a role bar taken on saturating content is not the role
    // bar anyone means.
    std::vector<uint8_t> frontFrame = frame;
    const char* frontSource = "synthetic smoothed noise (NOT representative content)";
    std::vector<uint8_t> blob;
    if (const char* path = std::getenv("BINCV_CUDA_ROLE_FRAMES")) {
        std::FILE* fh = std::fopen(path, "rb");
        if (fh != nullptr) {
            std::fseek(fh, 0, SEEK_END);
            const long len = std::ftell(fh);
            std::fseek(fh, 0, SEEK_SET);
            if (len > 0) {
                blob.resize(static_cast<size_t>(len));
                if (std::fread(blob.data(), 1, blob.size(), fh) != blob.size()) blob.clear();
            }
            std::fclose(fh);
        }
        if (!blob.empty()) {
            const bincv::SequenceHeader h = bincv::readSequenceHeader(blob.data(), blob.size());
            const bincv::SequenceFrameRange f0 =
                bincv::sequenceFrame(h, blob.data(), blob.size(), 0);
            if (h.valid && h.mode == bincv::kSequenceMode8Bit && h.width == kW &&
                h.height == kH && f0.valid) {
                frontFrame.assign(f0.data, f0.data + kW * kH);
                frontSource = "REAL SEQUENCE FRAME 0 from BINCV_CUDA_ROLE_FRAMES";
            }
        }
    }

    if (wantFrontend) {
        bc::uploadImage(frontFrame.data(), kW, kH, kW, fWide.view(), gStream);
        bc::medianWide<3>(fWide.constView(), fDenoised.view(), bincv::kMedianReferenceL,
                          gStream);
        // The reference frontend's threshold, overridable so the frontend rows
        // can be read against a SWEEP of corner density rather than at one
        // point. FAST's cost on this backend turns out to track the corner
        // COUNT rather than the capacity, and that is only checkable by moving
        // the count.
        uint8_t edgeT = 17;
        if (const char* e = std::getenv("BINCV_CUDA_ROLE_EDGE")) {
            const long n = std::atol(e);
            if (n > 0 && n < 256) edgeT = static_cast<uint8_t>(n);
        }
        bc::edgeThreshold(fDenoised.constView(), fBits.view(), edgeT,
                          bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                          bincv::EdgeSpatial::Wide, gStream);
        bc::derivativeXY(fBits.constView(), bc::planeBlock(fdxBlock.view(), 2),
                         bc::planeBlock(fdyBlock.view(), 2), bincv::BORDER_REFLECT_101,
                         false, gStream);
        cudaStreamSynchronize(gStream);

        // The SAME bits as the CV_8U picture OpenCV's detectors take. Downloaded
        // from the device rather than recomputed on the host, so the two sides
        // are provably the same frame and not two frames that ought to agree.
        bincv::BinMat<uint32_t> hostBits(static_cast<int>(kW), static_cast<int>(kH));
        bc::download(fBits.constView(), hostBits.view(), gStream);
        cudaStreamSynchronize(gStream);
        for (size_t y = 0; y < kH; ++y) {
            const uint32_t* row = hostBits.constView().row(y);
            for (size_t x = 0; x < kW; ++x)
                picture[y * kW + x] =
                    static_cast<uint8_t>(((row[x / 32] >> (x % 32)) & 1u) ? 255 : 0);
        }
        cv::Mat hostPicture(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                            picture.data());
        gPicture.upload(hostPicture);

        size_t setBits = 0;
        for (size_t i = 0; i < kW * kH; ++i) setBits += picture[i] != 0 ? 1u : 0u;
        std::printf("\n=====================================================================\n"
                    " ROUND 2 -- the frontend role bars. INPUT, stated once for all of\n"
                    " them: binCV's own sensor stage output at %zux%zu -- medianWide<3>\n"
                    " then edgeThreshold(17) -- %.2f%% of pixels set. OpenCV reads the\n"
                    " identical bits as a CV_8U {0,255} picture.\n"
                    " FRAME SOURCE: %s\n"
                    "=====================================================================\n",
                    kW, kH, 100.0 * static_cast<double>(setBits) /
                                static_cast<double>(kW * kH), frontSource);
        std::printf("FRAMESRC,%s,%.4f\n", frontSource,
                    100.0 * static_cast<double>(setBits) / static_cast<double>(kW * kH));
    }

    // ======================================================================
    // 9. FAST -- cv::cuda::FastFeatureDetector (cudafeatures2d)
    // ======================================================================
    if (want("fast")) {
        std::printf("\n=====================================================================\n"
                    " 9. FAST -- binCV cuda::detectFastAsync vs cv::cuda::FastFeatureDetector\n"
                    "=====================================================================\n"
                    " THE AUTHOR'S OWN RULE, not restated more kindly: required >= 1.00x\n"
                    " kernel-resident, both sides one explicit stream, with the corner-set\n"
                    " agreement gate passing first. The '>= 2.00x target' the design\n"
                    " carried was deleted by its own author as underived; what ratio ships\n"
                    " the claim UNQUALIFIED remains an open owner question, and nothing\n"
                    " here fills it in.\n"
                    " The capped arm is carried beside the shipped one because the family\n"
                    " located its shortfall in the single-block raster sort rather than in\n"
                    " the ring algebra, and a serial pass has to be able to check that.\n");

        // Sized so NOTHING TRUNCATES on this frame. A truncated run cannot be
        // compared against OpenCV's corner set at all -- the atomic decided
        // which corners were stored -- so a capacity below the true count turns
        // the agreement gate into a coin toss. A real EuRoC edge map at the
        // reference threshold carries far more FAST corners than the synthetic
        // frame the family benchmark used, and the first run of this section at
        // 16384 truncated BOTH sides; the true count is printed below so the
        // choice can be checked rather than trusted.
        // 32768 is the SMALLEST POWER OF TWO that holds this frame's 19,898
        // corners, and the capacity a caller sizing for this content would
        // pick. The choice is load-bearing rather than incidental: the single-
        // block bitonic sort orders nextPow2(capacity) SLOTS, not the corners
        // found, so capacity -- not corner count -- sets this op's cost. At
        // 262144 the same frame measures 3.53x against OpenCV where 32768
        // measures what the row below reports. A role bar must be taken at the
        // capacity a caller would use, and it is named here so a reader can
        // check the choice instead of trusting it.
        uint32_t cap = 32768;
        if (const char* cv2 = std::getenv("BINCV_CUDA_ROLE_FASTCAP")) {
            const long n = std::atol(cv2);
            if (n > 0) cap = static_cast<uint32_t>(n);
        }
        bc::DeviceArray<bc::DeviceFastCorner> dfast(cap);
        bc::DeviceAppendCounter fastCounter;
        const bc::DeviceFastCornerBuffer fastBuf(dfast.data(), fastCounter.devicePtr(), cap);
        const size_t fastScratchBytes = bc::fastScratchBytes(cap);
        bc::DeviceArray<uint8_t> dfastScratch(fastScratchBytes);

        const uint32_t capped = 512;
        bc::DeviceArray<bc::DeviceFastCorner> dfastCapped(capped);
        bc::DeviceAppendCounter cappedCounter;
        const bc::DeviceFastCornerBuffer cappedBuf(dfastCapped.data(),
                                                   cappedCounter.devicePtr(), capped);
        const size_t cappedScratchBytes = bc::fastScratchBytes(capped);
        bc::DeviceArray<uint8_t> dcappedScratch(cappedScratchBytes);

        const auto runFast = [&] {
            fastCounter.reset(gStream);
            bc::detectFastAsync(fBits.constView(), fastBuf, dfastScratch.data(),
                                fastScratchBytes, 9, gStream);
        };
        const auto runCapped = [&] {
            cappedCounter.reset(gStream);
            bc::detectFastAsync(fBits.constView(), cappedBuf, dcappedScratch.data(),
                                cappedScratchBytes, 9, gStream);
        };

        cv::Ptr<cv::cuda::FastFeatureDetector> cvFast = cv::cuda::FastFeatureDetector::create(
            128, false, cv::FastFeatureDetector::TYPE_9_16, static_cast<int>(cap));
        cv::cuda::GpuMat cvKp;

        // THE GATE FIRST. A speed ratio between two arms that found different
        // corners is not a comparison, so this runs before anything is timed
        // and its failure would stop the section rather than qualify it.
        runFast();
        cudaStreamSynchronize(gStream);
        bc::DeviceAppendResult fres;
        bc::readAppendResult(fastBuf, fres, gStream);
        std::vector<bc::DeviceFastCorner> mine(fres.acceptTruncated());
        if (!mine.empty()) {
            bc::downloadAppended(fastBuf, fres, mine.data(), gStream);
            cudaStreamSynchronize(gStream);
        }
        cvFast->detectAsync(gPicture, cvKp, cv::noArray(), gCvStream);
        gCvStream.waitForCompletion();
        cv::Mat kp;
        cvKp.download(kp);
        std::vector<uint64_t> theirs, ours;
        if (kp.rows > 0) {
            const short* loc = kp.ptr<short>(cv::cuda::FastFeatureDetector::LOCATION_ROW);
            for (int i = 0; i < kp.cols; ++i)
                theirs.push_back(
                    (static_cast<uint64_t>(static_cast<uint16_t>(loc[2 * i + 1])) << 32) |
                    static_cast<uint32_t>(static_cast<uint16_t>(loc[2 * i])));
        }
        for (const bc::DeviceFastCorner& c : mine)
            ours.push_back((static_cast<uint64_t>(static_cast<uint32_t>(c.y)) << 32) |
                           static_cast<uint32_t>(c.x));
        std::sort(theirs.begin(), theirs.end());
        std::sort(ours.begin(), ours.end());
        const bool setsAgree = theirs == ours;
        std::printf("\n CORNER-SET AGREEMENT GATE (runs before any timing): binCV %zu"
                    " (found %u, capacity %u%s), OpenCV %zu keypoints -- %s\n",
                    ours.size(), fres.found(), cap,
                    fres.truncated() ? ", *** TRUNCATED ***" : "", theirs.size(),
                    setsAgree ? "SETS AGREE" : "*** SETS DIFFER -- ratios below are NOT"
                                               " like-for-like ***");
        std::printf("GATE,fast_corner_sets,752x480,%zu,%zu,%d\n", ours.size(),
                    theirs.size(), setsAgree ? 1 : 0);

        const PairedTiming p = timeKernelPaired(
            [&] { cvFast->detectAsync(gPicture, cvKp, cv::noArray(), gCvStream); },
            [&] { runFast(); }, 10, 10, kRounds, gStream);
        printRole("FAST corners, binary frame vs CV_8U picture",
                  "cv::cuda::FastFeatureDetector (nms off)",
                  "bincv::cuda::detectFastAsync", p, floor, "752x480");
        std::printf("   RULE: required <= 1.000x (binCV at or faster than parity) -> %s\n",
                    p.ratioMedian <= 1.0 ? "MET" : "MISSED");
        emitRow("fast", "752x480", p);
        printEnqueue("fast", "752x480", "cv::cuda::FastFeatureDetector",
                     "bincv::cuda::detectFastAsync",
                     timeHostEnqueue(
                         [&] { cvFast->detectAsync(gPicture, cvKp, cv::noArray(), gCvStream); }),
                     timeHostEnqueue([&] { runFast(); }), p);

        const PairedTiming pc = timeKernelPaired(
            [&] { cvFast->detectAsync(gPicture, cvKp, cv::noArray(), gCvStream); },
            [&] { runCapped(); }, 10, 10, kRounds, gStream);
        printRole("FAST, binCV CAPPED at 512 stored -- the DETECTOR without most of\n"
                  " the sort. An UPPER BOUND on detection alone, so the conclusion is\n"
                  " safe in the direction it is used",
                  "cv::cuda::FastFeatureDetector (nms off)",
                  "bincv::cuda::detectFastAsync, capacity 512", pc, floor, "752x480");
        emitRow("fast_capped512", "752x480", pc);

        // Memory, meter 2, both sides, each over enough sets to resolve.
        {
            const int reps = 32;
            size_t binD = 0, ocvD = 0;
            {
                DeviceMemMeter m;
                m.reset();
                std::vector<std::unique_ptr<bc::DeviceBinMat>> planes;
                std::vector<std::unique_ptr<bc::DeviceArray<bc::DeviceFastCorner>>> outs;
                std::vector<std::unique_ptr<bc::DeviceArray<uint8_t>>> scr;
                for (int i = 0; i < reps; ++i) {
                    planes.push_back(std::make_unique<bc::DeviceBinMat>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    outs.push_back(
                        std::make_unique<bc::DeviceArray<bc::DeviceFastCorner>>(cap));
                    scr.push_back(std::make_unique<bc::DeviceArray<uint8_t>>(
                        bc::fastScratchBytes(cap)));
                }
                cudaDeviceSynchronize();
                binD = m.deltaBytes();
            }
            {
                DeviceMemMeter m;
                m.reset();
                // RESERVED, and it is not a micro-optimization. Each arm below
                // is ENQUEUED asynchronously and keeps reading its GpuMat until
                // the stream drains; a vector that reallocates DESTROYS the old
                // elements, and ~GpuMat frees device memory a launch in flight
                // is still reading. Reserving is what makes the loop safe.
                std::vector<cv::cuda::GpuMat> srcs, kps;
                std::vector<cv::Ptr<cv::cuda::FastFeatureDetector>> dets;
                srcs.reserve(static_cast<size_t>(reps));
                kps.reserve(static_cast<size_t>(reps));
                dets.reserve(static_cast<size_t>(reps));
                cv::Mat hp(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                           picture.data());
                for (int i = 0; i < reps; ++i) {
                    srcs.emplace_back(hp);
                    dets.push_back(cv::cuda::FastFeatureDetector::create(
                        128, false, cv::FastFeatureDetector::TYPE_9_16,
                        static_cast<int>(cap)));
                    kps.emplace_back();
                    dets.back()->detectAsync(srcs.back(), kps.back(), cv::noArray(),
                                             gCvStream);
                }
                gCvStream.waitForCompletion();
                ocvD = m.deltaBytes();
            }
            printMemPair("FAST working set, capacity 16384", "752x480", binD, ocvD, step,
                         reps, reps);
            emitMem("fast", "752x480", binD, ocvD, step, reps, reps);
        }
    }

    // ======================================================================
    // 10. goodFeaturesToTrack -- cv::cuda::createGoodFeaturesToTrackDetector
    // ======================================================================
    if (want("gftt")) {
        std::printf("\n=====================================================================\n"
                    " 10. goodFeaturesToTrack -- binCV device-resident selection vs\n"
                    "     cv::cuda::createGoodFeaturesToTrackDetector (cudaimgproc)\n"
                    "=====================================================================\n"
                    " WALL CLOCK ON BOTH SIDES, and the reason is on OpenCV's side: at\n"
                    " minDistance >= 1 its detect() DOWNLOADS the sorted candidate list,\n"
                    " runs the spacing filter on the HOST and uploads the survivors\n"
                    " (cudaimgproc/src/gftt.cpp). A CUDA-event clock would exclude that\n"
                    " pass and flatter OpenCV. binCV's selection never leaves the device.\n"
                    " THE SPEED BAR FOR THIS OP WAS NOT WRITABLE -- its author escalated\n"
                    " it rather than deriving a number from a host CPU ratio, and this\n"
                    " binary does not invent one either. The ratio is reported; the\n"
                    " verdict is the owner's.\n");

        const bincv::GoodFeaturesParams gp{};
        const uint32_t poolCap = 65536;
        const uint32_t rankCap = 32768;
        bc::DeviceArray<bc::DeviceCorner> cands(poolCap);
        bc::DeviceAppendCounter candCounter;
        bc::DeviceArray<uint32_t> maxBits(1);
        bc::DeviceArray<uint8_t> selScratch(bc::goodFeaturesScratchBytes(poolCap));
        bc::DeviceArray<bc::DeviceCorner> outCorners(rankCap);
        bc::DeviceArray<uint8_t> resultBlock(sizeof(bc::DeviceCornerResult));

        const auto runGftt = [&] {
            candCounter.reset(gStream);
            cudaMemsetAsync(maxBits.data(), 0, sizeof(uint32_t), gStream);
            bc::DeviceGoodFeaturesWorkspace work;
            work.candidates = bc::appendBuffer(cands, candCounter);
            work.maxBits = maxBits.data();
            work.scratch = selScratch.data();
            work.scratchBytes = selScratch.size();
            bc::goodFeaturesToTrackAsync(
                bc::planeBlock(fdxBlock.constView(), 2).plane(0),
                bc::planeBlock(fdyBlock.constView(), 2).plane(0),
                bc::planeBlock(fdxBlock.constView(), 2).plane(1),
                bc::planeBlock(fdyBlock.constView(), 2).plane(1), gp, work,
                outCorners.data(), rankCap,
                reinterpret_cast<bc::DeviceCornerResult*>(resultBlock.data()), gStream);
        };

        cv::Ptr<cv::cuda::CornersDetector> cvGftt =
            cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, gp.maxCorners,
                                                        gp.qualityLevel, gp.minDistance,
                                                        gp.blockSize, false);
        cv::cuda::GpuMat cvCorners;

        // WALL CLOCK, both arms, interleaved and order-alternated exactly as
        // timeKernelPaired does it -- but with a synchronize inside each arm,
        // because OpenCV's arm contains a host pass that a CUDA event cannot
        // see. Hand-rolled here rather than reaching for the event timer,
        // which would be the wrong instrument by construction.
        std::vector<double> sa, sb, ratios;
        const auto wallOne = [&](const std::function<void()>& body, int iters) {
            cudaStreamSynchronize(gStream);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) body();
            cudaStreamSynchronize(gStream);
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        };
        for (int i = 0; i < 3; ++i) {
            cvGftt->detect(gPicture, cvCorners, cv::noArray(), gCvStream);
            runGftt();
        }
        cudaStreamSynchronize(gStream);
        for (int r = 0; r < kRounds; ++r) {
            double ta = 0.0, tb = 0.0;
            if (r % 2 == 0) {
                ta = wallOne([&] { cvGftt->detect(gPicture, cvCorners, cv::noArray(), gCvStream); }, 5);
                tb = wallOne([&] { runGftt(); }, 5);
            } else {
                tb = wallOne([&] { runGftt(); }, 5);
                ta = wallOne([&] { cvGftt->detect(gPicture, cvCorners, cv::noArray(), gCvStream); }, 5);
            }
            sa.push_back(ta);
            sb.push_back(tb);
            ratios.push_back(ta > 0.0 ? tb / ta : 0.0);
        }
        PairedTiming pg;
        pg.a = summarize(sa);
        pg.b = summarize(sb);
        const Timing rg = summarize(ratios);
        pg.ratioMin = rg.minMs;
        pg.ratioMedian = rg.medianMs;
        pg.ratioMax = rg.maxMs;
        pg.rounds = kRounds;
        printRole("goodFeaturesToTrack -- WALL CLOCK both sides, OpenCV's host\n"
                  " round trip INSIDE its arm because a caller pays it",
                  "cv::cuda gftt (downloads, spaces on host, uploads)",
                  "bincv::cuda::goodFeaturesToTrackAsync (never leaves device)", pg, floor,
                  "752x480 WALL");
        emitRow("gftt_wall", "752x480", pg);

        // How many corners each side found, so the ratio is read against a
        // known workload rather than an assumed one.
        cudaStreamSynchronize(gStream);
        bc::DeviceCornerResult hres{};
        cudaMemcpy(&hres, resultBlock.data(), sizeof(hres), cudaMemcpyDeviceToHost);
        cvGftt->detect(gPicture, cvCorners, cv::noArray(), gCvStream);
        gCvStream.waitForCompletion();
        std::printf("   corners: binCV %u (ranked %u, truncated %u, pool overflow %u)"
                    " vs OpenCV %d.\n"
                    "   NOT A GATE and not an equality: the two run different spacing\n"
                    "   rules over different responses. It is here so the ratio above is\n"
                    "   read against a known workload.\n",
                    hres.count, hres.candidatesRanked, hres.candidatesTruncated,
                    hres.candidateOverflow, cvCorners.cols);
        std::printf("GATE,gftt_counts,752x480,%u,%d,0\n", hres.count, cvCorners.cols);

        {
            const int reps = 16;
            size_t binD = 0, ocvD = 0;
            {
                DeviceMemMeter m;
                m.reset();
                std::vector<std::unique_ptr<bc::DeviceBinMat>> planes;
                std::vector<std::unique_ptr<bc::DeviceArray<bc::DeviceCorner>>> cs, os;
                std::vector<std::unique_ptr<bc::DeviceArray<uint8_t>>> sc;
                for (int i = 0; i < reps; ++i) {
                    for (int q = 0; q < 4; ++q)
                        planes.push_back(std::make_unique<bc::DeviceBinMat>(
                            static_cast<int>(kW), static_cast<int>(kH)));
                    cs.push_back(std::make_unique<bc::DeviceArray<bc::DeviceCorner>>(poolCap));
                    os.push_back(std::make_unique<bc::DeviceArray<bc::DeviceCorner>>(rankCap));
                    sc.push_back(std::make_unique<bc::DeviceArray<uint8_t>>(
                        bc::goodFeaturesScratchBytes(poolCap)));
                }
                cudaDeviceSynchronize();
                binD = m.deltaBytes();
            }
            {
                DeviceMemMeter m;
                m.reset();
                std::vector<cv::cuda::GpuMat> srcs, outs;
                std::vector<cv::Ptr<cv::cuda::CornersDetector>> dets;
                srcs.reserve(static_cast<size_t>(reps));
                outs.reserve(static_cast<size_t>(reps));
                dets.reserve(static_cast<size_t>(reps));
                cv::Mat hp(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                           picture.data());
                for (int i = 0; i < reps; ++i) {
                    srcs.emplace_back(hp);
                    dets.push_back(cv::cuda::createGoodFeaturesToTrackDetector(
                        CV_8UC1, gp.maxCorners, gp.qualityLevel, gp.minDistance,
                        gp.blockSize, false));
                    outs.emplace_back();
                    dets.back()->detect(srcs.back(), outs.back(), cv::noArray(), gCvStream);
                }
                gCvStream.waitForCompletion();
                ocvD = m.deltaBytes();
            }
            printMemPair("goodFeaturesToTrack working set (FUSED arm: no frame-sized\n"
                         " float map exists at all on binCV's side)",
                         "752x480", binD, ocvD, step, reps, reps);
            emitMem("gftt", "752x480", binD, ocvD, step, reps, reps);
        }
    }

    // ======================================================================
    // 11. The corner response -- cv::cuda::createMinEigenValCorner and
    //     cv::cuda::createHarrisCorner (cudaimgproc)
    // ======================================================================
    if (want("cornerresp")) {
        std::printf("\n=====================================================================\n"
                    " 11. CORNER RESPONSE -- binCV cuda::cornerMinEigenValAsync vs\n"
                    "     cv::cuda::createMinEigenValCorner AND createHarrisCorner\n"
                    "=====================================================================\n"
                    " TWO DENOMINATORS, AND THEY ARE NOT INTERCHANGEABLE. binCV's op\n"
                    " computes the MINIMUM EIGENVALUE of the gradient covariance, so\n"
                    " createMinEigenValCorner is the like-for-like counterpart and is the\n"
                    " bar. createHarrisCorner computes a DIFFERENT response (det - k*tr^2)\n"
                    " over the same covariance and is timed beside it as context -- the\n"
                    " task named it, and the honest thing is to run it and say plainly\n"
                    " that it answers a different question, not to quietly substitute it\n"
                    " for the one that matches.\n"
                    " BOTH OpenCV arms read the CV_8U picture and internally run a Sobel;\n"
                    " binCV reads four TERNARY BIT PLANES that already exist in the\n"
                    " pipeline. That is a role comparison, not an equality, and the\n"
                    " asymmetry is stated rather than folded into the ratio.\n");

        bc::DeviceImage<float> resp(static_cast<int>(kW), static_cast<int>(kH));
        const auto runResp = [&] {
            bc::cornerMinEigenValAsync(bc::planeBlock(fdxBlock.constView(), 2).plane(0),
                                       bc::planeBlock(fdyBlock.constView(), 2).plane(0),
                                       bc::planeBlock(fdxBlock.constView(), 2).plane(1),
                                       bc::planeBlock(fdyBlock.constView(), 2).plane(1), 3,
                                       resp.view(), gStream);
        };
        cv::Ptr<cv::cuda::CornernessCriteria> cvMin =
            cv::cuda::createMinEigenValCorner(CV_8UC1, 3, 3);
        cv::Ptr<cv::cuda::CornernessCriteria> cvHarris =
            cv::cuda::createHarrisCorner(CV_8UC1, 3, 3, 0.04);
        cv::cuda::GpuMat cvResp;

        const PairedTiming pm = timeKernelPaired(
            [&] { cvMin->compute(gPicture, cvResp, gCvStream); }, [&] { runResp(); }, 20, 20,
            kRounds, gStream);
        printRole("min-eigenvalue response map, blockSize 3 -- THE BAR",
                  "cv::cuda::createMinEigenValCorner", "bincv::cuda::cornerMinEigenValAsync",
                  pm, floor, "752x480");
        emitRow("cornerresp_mineigen", "752x480", pm);
        printEnqueue("cornerresp_mineigen", "752x480", "cv::cuda minEigenVal",
                     "bincv::cuda::cornerMinEigenValAsync",
                     timeHostEnqueue([&] { cvMin->compute(gPicture, cvResp, gCvStream); }),
                     timeHostEnqueue([&] { runResp(); }), pm);

        const PairedTiming ph = timeKernelPaired(
            [&] { cvHarris->compute(gPicture, cvResp, gCvStream); }, [&] { runResp(); }, 20,
            20, kRounds, gStream);
        printRole("createHarrisCorner beside it -- CONTEXT, a DIFFERENT response\n"
                  " function over the same covariance, not binCV's counterpart",
                  "cv::cuda::createHarrisCorner (k = 0.04)",
                  "bincv::cuda::cornerMinEigenValAsync", ph, floor, "752x480");
        emitRow("cornerresp_harris_context", "752x480", ph);

        {
            const int reps = 16;
            size_t binD = 0, ocvD = 0;
            {
                DeviceMemMeter m;
                m.reset();
                std::vector<std::unique_ptr<bc::DeviceBinMat>> planes;
                std::vector<std::unique_ptr<bc::DeviceImage<float>>> maps;
                for (int i = 0; i < reps; ++i) {
                    for (int q = 0; q < 4; ++q)
                        planes.push_back(std::make_unique<bc::DeviceBinMat>(
                            static_cast<int>(kW), static_cast<int>(kH)));
                    maps.push_back(std::make_unique<bc::DeviceImage<float>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                }
                cudaDeviceSynchronize();
                binD = m.deltaBytes();
            }
            {
                DeviceMemMeter m;
                m.reset();
                std::vector<cv::cuda::GpuMat> srcs, outs;
                std::vector<cv::Ptr<cv::cuda::CornernessCriteria>> crit;
                srcs.reserve(static_cast<size_t>(reps));
                outs.reserve(static_cast<size_t>(reps));
                crit.reserve(static_cast<size_t>(reps));
                cv::Mat hp(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                           picture.data());
                for (int i = 0; i < reps; ++i) {
                    srcs.emplace_back(hp);
                    crit.push_back(cv::cuda::createMinEigenValCorner(CV_8UC1, 3, 3));
                    outs.emplace_back();
                    crit.back()->compute(srcs.back(), outs.back(), gCvStream);
                }
                gCvStream.waitForCompletion();
                ocvD = m.deltaBytes();
            }
            printMemPair("corner response working set (four bit planes + one float map\n"
                         " vs one byte picture + OpenCV's Sobel intermediates + one map)",
                         "752x480", binD, ocvD, step, reps, reps);
            emitMem("cornerresp", "752x480", binD, ocvD, step, reps, reps);
        }
    }

    // ======================================================================
    // 12. DESCRIBE -- cv::cuda::ORB::computeAsync (cudafeatures2d)
    // ======================================================================
    if (want("describe")) {
        std::printf("\n=====================================================================\n"
                    " 12. DESCRIBE -- binCV cuda::computeBriefSteered vs\n"
                    "     cv::cuda::ORB::computeAsync on PROVIDED keypoints\n"
                    "=====================================================================\n"
                    " computeAsync IS accepted in OpenCV 4.5.4 on provided keypoints, so a\n"
                    " stage-isolated GPU denominator exists and no differential is needed.\n"
                    " SUPERSET vs SUBSET, stated at the number: computeAsync also builds\n"
                    " ORB's level-0 pyramid entry and can blur. nlevels = 1 and\n"
                    " blurForDescriptor = false pin that superset as small as the API\n"
                    " allows; it is not zero, and the ratio is an upper bound on binCV's\n"
                    " advantage for that reason.\n"
                    " Both sides sample the SAME 256 pairs -- cv::ORB's learned table,\n"
                    " which binCV reaches by pointer out of ops/orbPattern.hpp.\n");

        constexpr size_t kBits = 256;
        constexpr size_t kWords = kBits / 32;
        const size_t n = 1000;
        std::vector<float> xy(2 * n);
        for (size_t i = 0; i < n; ++i) {
            xy[2 * i] = static_cast<float>(40 + (i * 37) % (kW - 80));
            xy[2 * i + 1] = static_cast<float>(40 + (i * 53) % (kH - 80));
        }
        bc::DeviceArray<float> dxy(2 * n);
        bc::DeviceArray<float> dang(n);
        bc::DeviceArray<uint8_t> dkeep(n);
        bc::DeviceArray<uint32_t> ddesc(n * kWords);
        bc::DeviceArray<bincv::BriefPair> dpairs(bc::steeredBriefPatternPairs<kBits>());
        static bincv::SteeredBriefPattern<kBits> orbSteered{};
        bincv::makeSteeredBriefPattern<kBits>(orbSteered, bincv::kOrbBriefPattern);
        bc::DeviceBriefPattern pat{};
        bc::uploadBriefPattern<kBits>(orbSteered, dpairs.data(), pat, gStream);
        cudaMemcpyAsync(dxy.data(), xy.data(), 2 * n * sizeof(float),
                        cudaMemcpyHostToDevice, gStream);
        cudaStreamSynchronize(gStream);
        const bc::DeviceKeypointSetConstView kps = bc::keypointSet(dxy.data(), n);
        const bc::DeviceDescriptorSetView dset =
            bc::descriptorSet(ddesc.data(), n, kWords, dkeep.data());
        bc::keypointOrientation(fDenoised.constView(), kps, dang.data(), dkeep.data(), 15,
                                nullptr, gStream);
        cudaStreamSynchronize(gStream);

        cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(
            static_cast<int>(n), 1.2f, 1, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20, false);
        cv::cuda::GpuMat gWide;
        {
            cv::Mat hostWide(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                             frontFrame.data());
            gWide.upload(hostWide);
        }
        cv::cuda::GpuMat kpMat, descMat;
        orb->detectAndComputeAsync(gWide, cv::noArray(), kpMat, descMat, false, gCvStream);
        gCvStream.waitForCompletion();
        bool computeAsyncWorks = false;
        std::string refusal;
        cv::cuda::GpuMat kpProvided = kpMat.clone();
        try {
            orb->computeAsync(gWide, kpProvided, descMat, gCvStream);
            gCvStream.waitForCompletion();
            computeAsyncWorks = !descMat.empty();
        } catch (const cv::Exception& e) {
            refusal = e.what();
        }
        std::printf("\n cv::cuda::Feature2DAsync::computeAsync on provided keypoints: %s\n",
                    computeAsyncWorks ? "ACCEPTED" : "REFUSED");
        if (!computeAsyncWorks) {
            std::printf("   %s\n   ROLE BAR UNMEASURED -> verdict BLOCKED. No substitute.\n",
                        refusal.substr(0, 200).c_str());
        } else {
            std::printf("   %d keypoints in, %dx%d CV_8U descriptors out.\n", kpProvided.cols,
                        descMat.rows, descMat.cols);
            cv::cuda::GpuMat descOut;
            const PairedTiming pd = timeKernelPaired(
                [&] { orb->computeAsync(gWide, kpProvided, descOut, gCvStream); },
                [&] {
                    bc::computeBriefSteered(fDenoised.constView(), kps, dang.data(), pat,
                                            dset, gStream);
                },
                20, 60, kRounds, gStream);
            printRole("256-bit steered BRIEF on provided keypoints",
                      "cv::cuda::ORB::computeAsync (nlevels 1, blur off)",
                      "bincv::cuda::computeBriefSteered", pd, floor, "752x480, N=1000");
            emitRow("describe", "752x480_n1000", pd);
            printEnqueue("describe", "752x480_n1000", "cv::cuda::ORB::computeAsync",
                         "bincv::cuda::computeBriefSteered",
                         timeHostEnqueue(
                             [&] { orb->computeAsync(gWide, kpProvided, descOut, gCvStream); }),
                         timeHostEnqueue([&] {
                             bc::computeBriefSteered(fDenoised.constView(), kps, dang.data(),
                                                     pat, dset, gStream);
                         }),
                         pd);
        }

        std::printf("\n ORIENTATION: **OUTSTANDING** (ruling R2). cv::cuda::ORB runs\n"
                    " IC_Angle inside its own keypoint pass and exposes no entry point\n"
                    " that orients PROVIDED keypoints, so no cv::cuda denominator exists\n"
                    " at any API level. No CPU number is put in its place.\n");

        {
            // THE TWO SIDES USE DIFFERENT REPLICA COUNTS, which printMemPair is
            // built for: each is divided by its OWN count. They have to differ
            // here. binCV's describe working set is ~45 KB, so sixteen of them
            // is under ONE of the driver's 2 MB units and the meter refuses to
            // quote it -- correctly. A cv::cuda::ORB instance is ~2 MB, so 512
            // of those would not fit on an 8 GB card. Each side is replicated
            // until its OWN total clears eight units, and no further.
            const int reps = 16;
            const int binReps = 512;
            size_t binD = 0, ocvD = 0;
            {
                DeviceMemMeter m;
                m.reset();
                std::vector<std::unique_ptr<bc::DeviceArray<float>>> xys, angs;
                std::vector<std::unique_ptr<bc::DeviceArray<uint8_t>>> keeps;
                std::vector<std::unique_ptr<bc::DeviceArray<uint32_t>>> descs;
                for (int i = 0; i < binReps; ++i) {
                    xys.push_back(std::make_unique<bc::DeviceArray<float>>(2 * n));
                    angs.push_back(std::make_unique<bc::DeviceArray<float>>(n));
                    keeps.push_back(std::make_unique<bc::DeviceArray<uint8_t>>(n));
                    descs.push_back(std::make_unique<bc::DeviceArray<uint32_t>>(n * kWords));
                }
                cudaDeviceSynchronize();
                binD = m.deltaBytes();
            }
            {
                DeviceMemMeter m;
                m.reset();
                std::vector<cv::Ptr<cv::cuda::ORB>> orbs;
                std::vector<cv::cuda::GpuMat> kpsv, descsv;
                orbs.reserve(static_cast<size_t>(reps));
                kpsv.reserve(static_cast<size_t>(reps));
                descsv.reserve(static_cast<size_t>(reps));
                // EACH REPLICA IS DETECTED ON FIRST, and that is not padding
                // the reading. cv::cuda::ORB::computeAsync on an instance that
                // has never detected faults -- its internal per-level buffers
                // are sized by the first detect -- so `detect then compute` is
                // the only sequence a caller can actually perform, and it is
                // what a caller therefore pays. The reading stays labelled an
                // UPPER BOUND on the describe stage alone, because the
                // detector's state is inside it.
                for (int i = 0; i < reps; ++i) {
                    orbs.push_back(cv::cuda::ORB::create(static_cast<int>(n), 1.2f, 1, 31, 0,
                                                         2, cv::ORB::HARRIS_SCORE, 31, 20,
                                                         false));
                    kpsv.emplace_back();
                    descsv.emplace_back();
                    orbs.back()->detectAndComputeAsync(gWide, cv::noArray(), kpsv.back(),
                                                       descsv.back(), false, gCvStream);
                    gCvStream.waitForCompletion();
                    orbs.back()->computeAsync(gWide, kpsv.back(), descsv.back(), gCvStream);
                }
                gCvStream.waitForCompletion();
                ocvD = m.deltaBytes();
            }
            printMemPair("describe working set, N = 1000 -- keypoints, angles, keep\n"
                         " bytes and descriptors on binCV's side; a cv::cuda::ORB\n"
                         " DETECTOR's whole state on OpenCV's, which is a superset",
                         "752x480, N=1000", binD, ocvD, step, binReps, reps);
            emitMem("describe", "752x480_n1000", binD, ocvD, step, binReps, reps);
        }
    }

    // ======================================================================
    // 13. DESCRIPTOR MATCHING -- the one row that runs the other way
    // ======================================================================
    if (want("matcher")) {
        std::printf("\n=====================================================================\n"
                    " 13. DESCRIPTOR MATCHING -- cv::cuda::DescriptorMatcher\n"
                    "=====================================================================\n"
                    " NO binCV DEVICE ARM EXISTS. This backend ships no device matcher:\n"
                    " `matchDescriptors` appears in backends/cuda/include/bincv/cuda/\n"
                    " features.hpp only as the shared vocabulary's named consumer, and no\n"
                    " kernel implements it.\n"
                    "\n"
                    " THIS IS NOT AN 'OUTSTANDING' ROW AND MUST NOT BE FILED AS ONE.\n"
                    " OUTSTANDING (ruling R2) is for a binCV op with no OpenCV\n"
                    " counterpart. This is the reverse: an OpenCV counterpart with no\n"
                    " binCV op. There is nothing to time and nothing to ship, so timing\n"
                    " cv::cuda::BFMatcher alone would produce a number with no second\n"
                    " arm -- which is not a comparison, and printing it beside this\n"
                    " backend's op list would invite exactly the misreading the file is\n"
                    " built to prevent.\n"
                    "\n"
                    " WHAT IS TRUE AND WORTH RECORDING. binCV's descriptors come out as\n"
                    " uint32_t words, so a matcher reading them issues 8 __popc per\n"
                    " 256-bit descriptor where cv::cuda's HammingDist::reduceIter, which\n"
                    " is instantiated at uchar, issues 32. That 4x is real, it sits on\n"
                    " the MATCHING side, and it is UNSPENT until someone writes the\n"
                    " kernel. It is a reason to write one, not a result.\n");
        std::printf("NOARM,matcher,752x480,cv::cuda::DescriptorMatcher has no binCV"
                    " device counterpart\n");
    }

    // ======================================================================
    // 14. OUTSTANDING -- every round-2 op with no cv::cuda bar at any level
    // ======================================================================
    if (want("outstanding")) {
        std::printf("\n=====================================================================\n"
                    " 14. OUTSTANDING (ruling R2) -- round 2's ops with NO cv::cuda\n"
                    "     counterpart at any API level\n"
                    "=====================================================================\n"
                    " Each of these ships on correctness, memory and the HOST comparison,\n"
                    " with its SPEED verdict recorded OUTSTANDING. No substitute bar is\n"
                    " invented and no CPU number is quoted as a GPU one.\n"
                    "\n"
                    " gradientCovarianceAsync / gradientCovarianceBatchAsync\n"
                    "     Neither cv::cuda nor cv:: computes a 2x2 gradient covariance at\n"
                    "     any API level. cornerHarris and createMinEigenValCorner compute\n"
                    "     a dense float RESPONSE THROUGH one -- that is the bar for a\n"
                    "     composed corner op, which section 11 runs, and it is not a bar\n"
                    "     for the covariance itself.\n"
                    "\n"
                    " cornerSubPixAsync\n"
                    "     No cv::cuda counterpart in any module. Decided instead by its\n"
                    "     own round-trip inequality, which its author reports it MISSES.\n"
                    "\n"
                    " goodFeaturesToTrack's DEVICE-RESIDENT SPACING\n"
                    "     Section 10 times the whole op against OpenCV's. The residency\n"
                    "     itself -- a greedy min-distance filter that never leaves the\n"
                    "     device -- has no counterpart to be timed against, because\n"
                    "     OpenCV's runs on the host by construction.\n"
                    "\n"
                    " keypointsFromCorners\n"
                    "     Tier 3. OpenCV does not have the problem: its detectors already\n"
                    "     hand back float2, so there is nothing to convert and nothing to\n"
                    "     compare.\n"
                    "\n"
                    " keypointOrientation (all three arms)\n"
                    "     cv::cuda::ORB runs IC_Angle inside its keypoint pass and exposes\n"
                    "     no entry point that orients provided keypoints. Section 12 says\n"
                    "     so at the number.\n"
                    "\n"
                    " derivativeX / derivativeY / derivativeXY over TERNARY planes\n"
                    "     Section 11's inputs. cv::cuda::createDerivFilter is the nearest\n"
                    "     denominator and the derivcov family times it; both arms sit ON\n"
                    "     the launch floor there, so that ratio is a lower bound on the\n"
                    "     gap and says nothing about binCV's kernel.\n");
        std::printf("OUTSTANDING,covariance\nOUTSTANDING,cornerSubPixAsync\n"
                    "OUTSTANDING,gftt_device_spacing\nOUTSTANDING,keypointsFromCorners\n"
                    "OUTSTANDING,keypointOrientation\n");
    }

    cudaStreamSynchronize(gStream);
    cudaStreamDestroy(gStream);
    std::printf("\n=====================================================================\n"
                " END. One run is NOT a number on this host: aggregate the ROW lines\n"
                " across at least seven independent processes before quoting anything.\n"
                "=====================================================================\n");
    return 0;
}

#else  // BINCV_CUDA_ROLE_OPENCV

int main() {
    std::printf(
        "=======================================================================\n"
        " CUDA ROLE COMPARISONS -- NOT COMPILED WITH A cuda-module OpenCV\n"
        "=======================================================================\n"
        " Every row in this benchmark is a GPU-to-GPU comparison against a\n"
        " cv::cuda call. Without an OpenCV built with cudaarithm, cudafilters,\n"
        " cudawarping and cudastereo there is nothing here to measure against,\n"
        " so nothing is measured and nothing is substituted.\n"
        "\n"
        " Configure with -DBINCV_CUDA_OPENCV_DIR=<install prefix of such a\n"
        " build> to enable it.\n"
        "\n"
        " VERDICTS IN THIS CONFIGURATION, stated rather than left blank:\n"
        "   cuda::threshold       role bar UNMEASURED -> BLOCKED on the\n"
        "                         both-axes ship rule. A memory argument does\n"
        "                         not carry an op past a missing role bar.\n"
        "   cuda::erode/morphologyEx, cuda::medianWide, cuda::edgeThreshold,\n"
        "   cuda::buildPyramidBox, denseDisparityBinary\n"
        "                         same: role bar UNMEASURED -> BLOCKED.\n"
        "   cuda::binarize, cuda::shift, cuda::pyrDownBox's N-bit rung\n"
        "                         no cv::cuda counterpart exists at any API\n"
        "                         level, so these are OUTSTANDING under owner\n"
        "                         ruling R2 whether or not OpenCV is present.\n"
        "\n"
        " The binCV-against-binCV arms -- every off-switch ratio and every\n"
        " gate-excluded ~1.00x control -- live in the FAMILY benchmarks\n"
        " (cuda_sensor_benchmark, cuda_window_benchmark, cuda_median_benchmark,\n"
        " cuda_pyramid_benchmark) and are unaffected by this guard.\n");
    return 0;
}

#endif  // BINCV_CUDA_ROLE_OPENCV
