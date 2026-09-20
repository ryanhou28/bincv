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
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include "bincv/binMat.hpp"
#include "bincv/io/sequence.hpp"
#include "bincv/cuda/corner.hpp"
#include "bincv/cuda/census.hpp"
#include "bincv/cuda/denseCensusBox.hpp"
#include "bincv/cuda/denseDisparity.hpp"
#include "bincv/cuda/derivative.hpp"
#include "bincv/cuda/descriptor.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/edge.hpp"
#include "bincv/cuda/fast.hpp"
#include "bincv/cuda/keypoints.hpp"
#include "bincv/cuda/median.hpp"
#include "bincv/cuda/opticalFlow.hpp"
#include "bincv/cuda/morphology.hpp"
#include "bincv/cuda/orientation.hpp"
#include "bincv/cuda/pyramid.hpp"
#include "bincv/cuda/shift.hpp"
#include "bincv/cuda/sparseMatch.hpp"
#include "bincv/cuda/threshold.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/census.hpp"
#include "bincv/ops/corner.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/edge.hpp"
#include "bincv/ops/medianWide.hpp"
#include "bincv/ops/morphology.hpp"
#include "bincv/ops/opticalFlow.hpp"
#include "bincv/ops/pyramid.hpp"
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

/// @brief Makes a free-text label safe to put in a COMMA-SEPARATED field.
/// @note NOT COSMETIC. The LK sweep passes its geometry as prose --
/// "752x480, 1024 pts (prefix of minDistance 6)" -- which put two extra
/// commas inside one field and shifted every column after it. Seven runs of
/// that row were therefore unaggregatable, and a parser that did not notice
/// read the keypoint count as the first timing. The label is the only field
/// here a caller composes, so it is the only one that needs this.
const char* csvSafe(const char* text, char* buf, size_t n) {
    size_t i = 0;
    for (; text[i] != '\0' && i + 1 < n; ++i) buf[i] = text[i] == ',' ? ';' : text[i];
    buf[i] = '\0';
    return buf;
}

void emitRow(const char* key, const char* geom, const PairedTiming& p) {
    char safeGeom[256];
    geom = csvSafe(geom, safeGeom, sizeof(safeGeom));
    // The trailing fields are what the cross-process aggregation now decides
    // on: the geometric mean, the sign split and the per-round ratio's own
    // spread. The separation bit stays in the row because it is a fact worth
    // carrying, but it is no longer the verdict -- see paired_stats.hpp.
    std::printf("ROW,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d"
                ",%.6f,%d,%d,%d,%.6f,%.6f,%.4g\n",
                key, geom, p.a.minMs, p.a.medianMs, p.a.maxMs, p.b.minMs, p.b.medianMs,
                p.b.maxMs, p.ratioMin, p.ratioMedian, p.ratioMax, p.separated() ? 1 : 0,
                p.rounds, p.ratioGeoMean, p.roundsFavouringA, p.roundsFavouringB,
                p.roundsTied, p.differenceFactor(), p.ratioSwingFactor(), p.signTestP());
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
    std::printf("   ratio binCV/OpenCV, interleaved rounds: median %5.3fx <- QUOTE THIS"
                "  (binCV %5.2fx %s)\n"
                "                                           geomean %5.3fx"
                "   per-round range %5.3f-%5.3fx (%d rounds)\n",
                p.ratioMedian, p.ratioMedian > 0.0 ? 1.0 / p.ratioMedian : 0.0,
                p.ratioMedian < 1.0 ? "FASTER" : "SLOWER", p.ratioGeoMean, p.ratioMin,
                p.ratioMax, p.rounds);
    printPairedSignAndSeparation(p);
    printPairedVerdict(p);
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
    // WHAT ONE UNIT IS WORTH PER FRAME, which is the rounding each per-frame
    // figure above carries. It is the number that decides whether a replica
    // count was high enough, and it is printed rather than left to be derived:
    // at 64 replicas one unit is 32 KB/frame, which on a 430 KB reading was
    // enough for two harnesses to disagree by 6%.
    std::printf("   [meter 2] ROUNDING: one %.2f MB unit is %.1f KB/frame at %d sets"
                " (%.2f%% of binCV's\n"
                "             figure) and %.1f KB/frame at %d sets (%.2f%% of"
                " OpenCV's).\n",
                unit / (1024.0 * 1024.0), unit / binReplicas / 1024.0, binReplicas,
                binPer > 0.0 ? unit / binReplicas / binPer * 100.0 : 0.0,
                unit / ocvReplicas / 1024.0, ocvReplicas,
                ocvPer > 0.0 ? unit / ocvReplicas / ocvPer * 100.0 : 0.0);
    if (binBytes == 0 || binU < 8.0 || ocvU < 8.0) {
        std::printf("   RATIO NOT QUOTED: a side read under eight of the meter's own units,\n"
                    "   so its rounding is a large fraction of the reading. Raise that\n"
                    "   side's replica count before quoting this pair.\n");
    } else {
        // THE DIRECTION IS PRINTED, NOT ASSUMED. This helper used to say
        // "binCV smaller by that factor" whatever the ratio was, which reads as
        // a claim rather than a reading on any row where OpenCV is the smaller
        // side -- and round 3 produced one (the census entry). The ratio is
        // still OpenCV/binCV everywhere so the rows stay comparable; only the
        // sentence under it follows the number.
        std::printf("   ratio OpenCV/binCV = %.3fx  -- %s by %.3fx.\n"
                    "   The OpenCV side is an UPPER BOUND: GpuMat pads its pitch, and any\n"
                    "   filter buffer or NPP scratch held past the call is inside this\n"
                    "   delta. binCV's side is cudaMalloc with nothing in between.\n",
                    ocvPer / binPer,
                    ocvPer >= binPer ? "binCV is SMALLER" : "OpenCV is SMALLER",
                    ocvPer >= binPer ? ocvPer / binPer : binPer / ocvPer);
    }
}

// ---------------------------------------------------------------------------
// ROUND 3: the tracker state, on device and on the host, so section 14's role
// row and section 16's sequence row are built from ONE definition of "what a
// tracker holds". A second definition would let the two sections quietly
// measure two different working sets.
// ---------------------------------------------------------------------------

using Ladder = bc::DevicePyramid<1, 2, 2, 2>;
constexpr int kLkLevels = 4;
constexpr int kLkWin = 31;        ///< the reference frontend's window
constexpr int kLkIterCap = 20;    ///< ...and its iteration cap
constexpr int kEdgeThr = 17;      ///< ...and its edge threshold
constexpr uint32_t kTrackCapacity = 2048;
constexpr uint32_t kRankCapacity = 32768;

/// The derivative planes of ONE ladder: `bits + 1` planes per axis per level.
/// LK linearises about the PREVIOUS frame, so only one ladder's derivatives
/// ever exist -- which is what halves this footprint against a naive reading.
struct DerivLadder {
    bc::DeviceBinMat dx[kLkLevels];
    bc::DeviceBinMat dy[kLkLevels];
    size_t bits[kLkLevels] = {1, 2, 2, 2};

    DerivLadder() {
        int w = static_cast<int>(kW), h = static_cast<int>(kH);
        for (int i = 0; i < kLkLevels; ++i) {
            const int rows = static_cast<int>((bits[i] + 1) * static_cast<size_t>(h));
            dx[i] = bc::DeviceBinMat(w, rows);
            dy[i] = bc::DeviceBinMat(w, rows);
            w = static_cast<int>(bc::pyrDownWidth(static_cast<size_t>(w)));
            h = static_cast<int>(bc::pyrDownHeight(static_cast<size_t>(h)));
        }
    }
    bc::DevicePlaneBlockView dxAt(int i) { return bc::planeBlock(dx[i].view(), bits[i] + 1); }
    bc::DevicePlaneBlockView dyAt(int i) { return bc::planeBlock(dy[i].view(), bits[i] + 1); }
    size_t bytes() const {
        size_t t = 0;
        for (int i = 0; i < kLkLevels; ++i) {
            t += dx[i].getHeight() * dx[i].getAlignedWidth() * sizeof(uint32_t);
            t += dy[i].getHeight() * dy[i].getAlignedWidth() * sizeof(uint32_t);
        }
        return t;
    }
};

/// THE DEVICE TRACKER'S WHOLE RESIDENT STATE -- both ladders, the previous
/// frame's derivatives, the keypoint arrays, and the sensor stage's two wide
/// staging frames. The wide frames ARE counted here, unlike in the optical-flow
/// family's own Gate 3 where the comparison was tracker-to-tracker: a SEQUENCE
/// number is what a caller pays per frame end to end, and the frame has to
/// arrive somewhere.
struct DeviceTracker {
    Ladder prev, next;
    DerivLadder deriv;
    bc::DeviceImage<uint8_t> wide, denoised;
    bc::DeviceArray<float> prevXY, nextXY;
    bc::DeviceArray<uint8_t> status;
    // The detector's own working set, allocated once and reused -- a tracker
    // that allocated per detection would be timing cudaMalloc.
    bc::DeviceArray<bc::DeviceCorner> candidates, corners;
    bc::DeviceAppendCounter counter;
    bc::DeviceArray<uint32_t> maxBits;
    bc::DeviceArray<uint8_t> gfScratch;
    bc::DeviceArray<bc::DeviceCornerResult> result;

    DeviceTracker()
        : prev(static_cast<int>(kW), static_cast<int>(kH)),
          next(static_cast<int>(kW), static_cast<int>(kH)),
          wide(static_cast<int>(kW), static_cast<int>(kH)),
          denoised(static_cast<int>(kW), static_cast<int>(kH)),
          prevXY(2 * kTrackCapacity), nextXY(2 * kTrackCapacity), status(kTrackCapacity),
          // THE CAPACITY CONTRACT, AND IT IS NOT `maxCorners`. Both headers
          // say so and this harness got it wrong first: `capacity` bounds the
          // survivors that can be RANKED, and the greedy spacing filter then
          // accepts from that ranked list in rank order. Sizing it to
          // `maxCorners` silently shortens the list the filter sees -- measured
          // here as 132 device corners against the host's 204 on the same
          // frame, with the 132 an exact PREFIX of the 204. Both arms now rank
          // the same 32768.
          candidates(kRankCapacity), corners(kRankCapacity), maxBits(1),
          gfScratch(bc::goodFeaturesScratchBytes(kRankCapacity)), result(1) {}

    /// METER 1: the allocation sum from the containers' own closed formulas.
    /// binCV to binCV only; never divided into the driver meter's reading.
    size_t bytes() const {
        return prev.sizeInBytes() + next.sizeInBytes() + deriv.bytes() +
               2 * kW * kH +                                  // wide + denoised
               4 * kTrackCapacity * sizeof(float) +           // prevXY + nextXY
               kTrackCapacity +                               // status
               2 * kRankCapacity * sizeof(bc::DeviceCorner) +
               bc::goodFeaturesScratchBytes(kRankCapacity) + sizeof(uint32_t) +
               sizeof(bc::DeviceCornerResult);
    }
    /// The same without the detector's ranking pool and without the wide staging
    /// frames: what the TRACKER alone holds, for the reader who wants the two
    /// separated rather than folded.
    size_t trackerOnlyBytes() const {
        return prev.sizeInBytes() + next.sizeInBytes() + deriv.bytes() +
               4 * kTrackCapacity * sizeof(float) + kTrackCapacity;
    }
};

/// One frame's sensor stage and ladder, ENQUEUED. Everything after the upload
/// reads memory that is already on the device; nothing comes back.
void deviceLoadFrame(DeviceTracker& t, const uint8_t* hostFrame, Ladder& ladder,
                     cudaStream_t s) {
    bc::uploadImage<uint8_t>(hostFrame, kW, kH, kW, t.wide.view(), s);
    bc::medianWide<3>(t.wide.constView(), t.denoised.view(), bincv::kMedianReferenceL, s);
    bc::edgeThreshold(t.denoised.constView(), ladder.levelAt(0).plane(0),
                      static_cast<uint8_t>(kEdgeThr), bincv::EdgeCombine::Or,
                      bincv::EdgeRelation::Ge, bincv::EdgeSpatial::Wide, s);
    bc::buildPyramidBox(ladder, s);
}

void deviceDerivatives(DeviceTracker& t, Ladder& of, cudaStream_t s) {
    for (int i = 0; i < kLkLevels; ++i) {
        bc::derivativeXY(of.levelAt(static_cast<size_t>(i)), t.deriv.dxAt(i),
                         t.deriv.dyAt(i), bincv::BORDER_REFLECT_101, false, s);
    }
}

void deviceBuildLevels(DeviceTracker& t, bc::DeviceLKLevel (&levels)[kLkLevels]) {
    for (int i = 0; i < kLkLevels; ++i) {
        const size_t li = static_cast<size_t>(i);
        levels[i] = bc::deviceLkLevel(t.prev.levelAt(li), t.next.levelAt(li),
                                      t.deriv.dxAt(i), t.deriv.dyAt(i));
    }
}

/// The detector, ENQUEUED except for the one 4-byte read of its own count.
/// A detection frame is the ONLY frame on which a resident tracker has to know
/// something the device knows -- how many corners there are -- and that read is
/// inside the timed region on both arms because both arms pay it.
uint32_t deviceDetect(DeviceTracker& t, double minDistance, float* dstXY,
                      cudaStream_t s) {
    bincv::GoodFeaturesParams gf;
    gf.maxCorners = static_cast<int>(kTrackCapacity);
    gf.minDistance = minDistance;
    t.counter.reset(s);
    cudaMemsetAsync(t.maxBits.data(), 0, sizeof(uint32_t), s);

    bc::DeviceGoodFeaturesWorkspace work;
    work.candidates = bc::appendBuffer(t.candidates, t.counter);
    work.maxBits = t.maxBits.data();
    work.scratch = t.gfScratch.data();
    work.scratchBytes = t.gfScratch.size();

    bc::DevicePlaneBlockView dx = t.deriv.dxAt(0);
    bc::DevicePlaneBlockView dy = t.deriv.dyAt(0);
    bc::goodFeaturesToTrackAsync(dx.plane(0), dy.plane(0), dx.plane(1), dy.plane(1), gf,
                                 work, t.corners.data(), kRankCapacity, t.result.data(), s);
    bc::keypointsFromCorners(t.corners.data(), &t.result.data()->count, dstXY,
                             kTrackCapacity, s);
    bc::DeviceCornerResult hr{};
    cudaMemcpyAsync(&hr, t.result.data(), sizeof(hr), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    return hr.count;
}

// ---- the host tracker, the same ladder and the same policy ----------------

using HW = uint32_t;

/// The host arm's whole state, allocated once. This mirrors
/// benchmark/frontend_sequence.cpp's `BincvFrontend` -- the same 1/2/2/2
/// ladder, the same swap scheme, the same streaming response ring -- so the
/// host number here is the host frontend's number and not a second spelling
/// of it that might have drifted.
struct HostTracker {
    bincv::Pyramid<HW, 1, 2, 2, 2> prev, next;
    bincv::SignedQuantMat<1, HW> dx0, dy0;
    bincv::SignedQuantMat<2, HW> dx1, dy1, dx2, dy2, dx3, dy3;
    bincv::LKLevels<HW, 1, 2, 2, 2> levels;
    std::vector<float> ring;
    std::vector<uint8_t> medianScratch;
    int w, h;

    HostTracker(int width, int height)
        : prev(width, height), next(width, height), dx0(width, height), dy0(width, height),
          dx1(width / 2 + (width & 1), height / 2 + (height & 1)),
          dy1(width / 2 + (width & 1), height / 2 + (height & 1)),
          dx2((width + 3) / 4, (height + 3) / 4), dy2((width + 3) / 4, (height + 3) / 4),
          dx3((width + 7) / 8, (height + 7) / 8), dy3((width + 7) / 8, (height + 7) / 8),
          ring(bincv::kResponseRingRows * static_cast<size_t>(width)),
          medianScratch(static_cast<size_t>(width) * static_cast<size_t>(height)), w(width),
          h(height) {}

    void sensorStage(const uint8_t* gray) {
        const size_t ww = static_cast<size_t>(w), hh = static_cast<size_t>(h);
        bincv::medianWide<3, uint8_t>(gray, ww, hh, ww, medianScratch.data(), ww,
                                      bincv::kMedianReferenceL);
        bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                             bincv::EdgeSpatial::Wide, uint8_t, HW>(
            medianScratch.data(), ww, hh, ww, next.level<0>().plane(0),
            static_cast<uint8_t>(kEdgeThr));
    }
    void seed(const uint8_t* gray) {
        sensorStage(gray);
        next.build<bincv::PyrDownFilter::Box2x2,
                            bincv::PyrDownBorder::Replicate>();
    }
    void loadFrame(const uint8_t* gray) {
        std::swap(prev, next);
        sensorStage(gray);
        next.build<bincv::PyrDownFilter::Box2x2,
                            bincv::PyrDownBorder::Replicate>();
    }
    void derivatives() {
        bincv::derivativeX(prev.level<0>(), dx0);
        bincv::derivativeY(prev.level<0>(), dy0);
        bincv::derivativeX(prev.level<1>(), dx1);
        bincv::derivativeY(prev.level<1>(), dy1);
        bincv::derivativeX(prev.level<2>(), dx2);
        bincv::derivativeY(prev.level<2>(), dy2);
        bincv::derivativeX(prev.level<3>(), dx3);
        bincv::derivativeY(prev.level<3>(), dy3);
        levels.get<0>() = bincv::lkLevel<1>(prev.level<0>(),
                                                    next.level<0>(), dx0, dy0);
        levels.get<1>() = bincv::lkLevel<2>(prev.level<1>(),
                                                    next.level<1>(), dx1, dy1);
        levels.get<2>() = bincv::lkLevel<2>(prev.level<2>(),
                                                    next.level<2>(), dx2, dy2);
        levels.get<3>() = bincv::lkLevel<2>(prev.level<3>(),
                                                    next.level<3>(), dx3, dy3);
    }
    /// METER 3 (HOST bytes), and it is NEVER divided into a device figure.
    size_t bytes() const {
        const size_t pyr = prev.sizeInBytes() + next.sizeInBytes();
        const size_t der = (dx0.sizeInWords() + dy0.sizeInWords() + dx1.sizeInWords() +
                            dy1.sizeInWords() + dx2.sizeInWords() + dy2.sizeInWords() +
                            dx3.sizeInWords() + dy3.sizeInWords()) * sizeof(HW);
        return pyr + der + ring.size() * sizeof(float) + medianScratch.size();
    }
};

/// Wall-clock median over `reps` passes of a whole-sequence loop.
struct SeqResult {
    double msPerFrame = 0.0;
    double msMin = 0.0;
    double msMax = 0.0;
    size_t frames = 0;
    size_t detections = 0;
    size_t tracked = 0;   ///< points tracked, summed over frames
};

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

    // THE REAL FRAME, LOADED ONCE AT THE TOP, and why it is not the synthetic
    // one the rest of this file uses where content matters.
    //
    // HOISTED HERE rather than declared beside the frontend families because
    // two sections need it and both need it for the same reason: a figure that
    // depends on what is IN the picture cannot be taken on smoothed noise.
    // Section 7b meters cv::cuda::StereoBM on it, to answer whether StereoBM's
    // footprint moves with content the way OpenCV's FAST turned out to.
    // The frontend families detect on it, because a detector's cost is a
    // function of how many corners its input has, and edgeThreshold on
    // SMOOTHED NOISE sets ~83% of pixels -- a frame
    // on which both FAST implementations overflow any sane capacity, so the
    // corner-set gate cannot even run. The real content this project measures
    // on is a EuRoC sequence blob; point BINCV_CUDA_ROLE_FRAMES at one
    // (scripts/make_sequence_blob.py, --mode 8bit) and frame 0 of it is used.
    // Without one the synthetic frame is used and every frontend row below
    // says so, because a role bar taken on saturating content is not the role
    // bar anyone means.
    std::vector<uint8_t> frontFrame = frame;
    bool haveRealFrame = false;
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
                haveRealFrame = true;
            }
        }
    }


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

        // ------------------------------------------------------------------
        // MEMORY: all three working sets, ONE region, ONE meter, ONE count
        // ------------------------------------------------------------------
        //
        // WHY THIS BLOCK EXISTS. StereoBM's footprint has been read twice in
        // this project at 3.3x apart -- 10.0 MB in one round and 3.0 MB in
        // another -- and nothing was wrong with either reading. They were
        // taken on different regions, at different replica counts, in
        // different processes. Two readings of one library that far apart
        // are not two measurements of a footprint; they are one measurement
        // of how much a protocol matters. So the three working sets that this
        // report compares are metered HERE, in this scope, against this
        // process's own meter step, at one replica count:
        //
        //   * binCV binary entry -- two DeviceBinMat bit planes and the
        //     disparity map. This is what a caller who already holds bits has
        //     resident.
        //   * binCV census entry -- two wide uint8 frames, two packed
        //     uint32 descriptor images and the disparity map. This is what a
        //     caller arriving with ordinary camera frames has resident.
        //   * cv::cuda::StereoBM(64, 9) -- two CV_8UC1 GpuMats, the output
        //     GpuMat and whatever compute() allocates internally, which is
        //     why the first compute() is INSIDE the metered scope.
        //
        // THE REPLICA COUNT IS CHOSEN BY THE ROUNDING, not by habit. This
        // driver reserves in 2 MB units, so a reading carries up to one unit
        // of rounding however large it is, and what matters is that unit
        // divided by the count against the per-frame figure. The smallest of
        // the three sets is the binary entry at roughly 450 KB. At 64
        // replicas one unit is 32 KB/frame, which is 7% of it -- and 32
        // KB/frame on a 430 KB reading is exactly what let two FAST harnesses
        // disagree 1.615x against 1.714x while both cleared the eight-unit
        // rule. At kReplicas = 256 one unit is 8.0 KB/frame: 1.8% of the
        // binary entry, 0.2% of the census entry, 0.3% of StereoBM. Every
        // per-frame figure below states its own rounding.
        //
        // WHAT IS AND IS NOT INSIDE EACH SET. binCV's two sets are cudaMalloc
        // with nothing between the op and the driver, so their readings are
        // the arrays plus the driver's rounding. StereoBM's is an UPPER
        // READING: GpuMat pads its pitch, it may be backed by a BufferPool,
        // and anything compute() holds past the call lands inside the delta.
        // The direction of that asymmetry is stated at the number rather than
        // folded into it.
        {
            std::printf("\n---------------------------------------------------------------\n"
                        " 7b. MEMORY -- THE THREE WORKING SETS, ONE REGION, ONE METER,\n"
                        "     ONE REPLICA COUNT, THIS PROCESS\n"
                        "---------------------------------------------------------------\n"
                        " Meter 2 (cudaMemGetInfo delta) on all three, %d replicas each,\n"
                        " every set held until its delta is read. Region: %zux%zu, 64\n"
                        " disparities, 9x9 support -- the same region section 7 timed.\n",
                        kReplicas, kW, kH);

            constexpr size_t kCensusK2 = 24;  // kCensus5x5

            // --- binCV binary entry ---
            std::vector<std::unique_ptr<bc::DeviceBinMat>> bl, br;
            std::vector<std::unique_ptr<bc::DeviceImage<uint8_t>>> bm2;
            const size_t binaryMem = meterScope(
                [&](int) {
                    bl.push_back(std::make_unique<bc::DeviceBinMat>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    br.push_back(std::make_unique<bc::DeviceBinMat>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    bm2.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    bc::denseDisparityBinary(bl.back()->constView(), br.back()->constView(),
                                             dp, bm2.back()->view(), gStream);
                },
                kReplicas);
            cudaStreamSynchronize(gStream);
            const size_t binaryAllocSum =
                (2 * kH * bl.front()->getAlignedWidth() * sizeof(uint32_t)) + kW * kH;
            bl.clear(); br.clear(); bm2.clear();
            cudaDeviceSynchronize();

            // THE COUNT IS CHECKED, NOT ASSERTED. The arithmetic above says 256
            // replicas put one meter unit at 8 KB/frame, but the arithmetic is
            // only a bound on the rounding -- it does not prove the reading has
            // converged, and the FAST re-meter is the reason that distinction
            // is in this file. There, two harnesses at 64 replicas BOTH cleared
            // the eight-unit rule and still disagreed 1.615x against 1.714x.
            // So the smallest of the three sets -- the one where a unit is the
            // largest fraction -- is also read at a quarter of the count, and
            // the two per-frame figures are printed together. If they differ,
            // the count is too low and neither is quotable.
            std::vector<std::unique_ptr<bc::DeviceBinMat>> ql, qr;
            std::vector<std::unique_ptr<bc::DeviceImage<uint8_t>>> qm;
            const int kQuarter = kReplicas / 4;
            const size_t binaryMemQuarter = meterScope(
                [&](int) {
                    ql.push_back(std::make_unique<bc::DeviceBinMat>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    qr.push_back(std::make_unique<bc::DeviceBinMat>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    qm.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    bc::denseDisparityBinary(ql.back()->constView(), qr.back()->constView(),
                                             dp, qm.back()->view(), gStream);
                },
                kQuarter);
            cudaStreamSynchronize(gStream);
            ql.clear(); qr.clear(); qm.clear();
            cudaDeviceSynchronize();

            // --- binCV census entry ---
            std::vector<std::unique_ptr<bc::DeviceImage<uint8_t>>> cw;
            std::vector<std::unique_ptr<bc::DeviceImage<uint32_t>>> cdsc;
            std::vector<std::unique_ptr<bc::DeviceImage<uint8_t>>> cmap;
            const size_t censusMem = meterScope(
                [&](int) {
                    cw.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    cw.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    cdsc.push_back(std::make_unique<bc::DeviceImage<uint32_t>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    cdsc.push_back(std::make_unique<bc::DeviceImage<uint32_t>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    cmap.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                        static_cast<int>(kW), static_cast<int>(kH)));
                    bc::censusTransformPacked<kCensusK2>(cw[cw.size() - 2]->constView(),
                                                         bincv::kCensus5x5,
                                                         cdsc[cdsc.size() - 2]->view(),
                                                         gStream);
                    bc::censusTransformPacked<kCensusK2>(cw.back()->constView(),
                                                         bincv::kCensus5x5,
                                                         cdsc.back()->view(), gStream);
                    bc::denseDisparityCensusPacked(cdsc[cdsc.size() - 2]->constView(),
                                                   cdsc.back()->constView(), dp,
                                                   cmap.back()->view(), gStream);
                },
                kReplicas);
            cudaStreamSynchronize(gStream);
            cw.clear(); cdsc.clear(); cmap.clear();
            cudaDeviceSynchronize();

            // --- cv::cuda::StereoBM(64, 9), on the SAME frames ---
            // CONTENT DEPENDENCE IS CHECKED, NOT ASSUMED. OpenCV's FAST sizes
            // its output by corners FOUND, so its footprint moves with the
            // picture; that was found the hard way in this project. StereoBM
            // is metered twice here, once on the synthetic pair and once on a
            // real EuRoC frame pair, and the two readings are printed beside
            // each other. If they differ, the real one is the number.
            const auto meterStereoBM = [&](const uint8_t* lp, const uint8_t* rp) {
                std::vector<cv::cuda::GpuMat> ml, mr, md;
                std::vector<cv::Ptr<cv::cuda::StereoBM>> mbm;
                cv::Mat hl(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                           const_cast<uint8_t*>(lp));
                cv::Mat hr(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                           const_cast<uint8_t*>(rp));
                const size_t got = meterScope(
                    [&](int) {
                        ml.emplace_back(hl);
                        mr.emplace_back(hr);
                        md.emplace_back();
                        mbm.push_back(cv::cuda::createStereoBM(64, 9));
                        mbm.back()->compute(ml.back(), mr.back(), md.back(), gCvStream);
                        cudaStreamSynchronize(gStream);
                    },
                    kReplicas);
                ml.clear(); mr.clear(); md.clear(); mbm.clear();
                cudaDeviceSynchronize();
                return got;
            };
            const size_t bmSynthetic = meterStereoBM(lw.data(), rw.data());

            // The real pair: EuRoC frame 0 as the left image and the same
            // frame shifted as the right, so only the CONTENT differs from
            // the synthetic run and the geometry does not.
            size_t bmReal = 0;
            bool haveReal = false;
            if (haveRealFrame) {
                std::vector<uint8_t> rl(frontFrame.begin(), frontFrame.end());
                std::vector<uint8_t> rr(kW * kH, 0);
                for (size_t y = 0; y < kH; ++y)
                    for (size_t x = 0; x + 21 < kW; ++x)
                        rr[y * kW + x] = rl[y * kW + x + 21];
                bmReal = meterStereoBM(rl.data(), rr.data());
                haveReal = true;
            }

            const double unit = static_cast<double>(step);
            const double perUnitKB = unit / kReplicas / 1024.0;
            const auto row = [&](const char* name, size_t total, const char* note) {
                const double per = static_cast<double>(total) / kReplicas;
                std::printf("   %-38s %8.2f MB / %d = %8.1f KB/frame  (%4.0f units,"
                            " rounding %.2f%%)  %s\n",
                            name, static_cast<double>(total) / (1024.0 * 1024.0),
                            kReplicas, per / 1024.0,
                            unit > 0.0 ? static_cast<double>(total) / unit : 0.0,
                            per > 0.0 ? perUnitKB * 1024.0 / per * 100.0 : 0.0, note);
            };
            std::printf("\n   one meter unit = %.2f MB = %.1f KB/frame at %d replicas\n",
                        unit / (1024.0 * 1024.0), perUnitKB, kReplicas);
            row("binCV BINARY entry", binaryMem, "2 bit planes + map");
            {
                const double per256 = static_cast<double>(binaryMem) / kReplicas;
                const double per64 = static_cast<double>(binaryMemQuarter) / kQuarter;
                const double drift = per256 > 0.0 ? per64 / per256 : 0.0;
                std::printf("     CONVERGENCE, on the smallest of the three: the same set"
                            " reads %.1f KB/frame\n"
                            "     at %d replicas against %.1f KB/frame at %d -- %.4fx."
                            " %s\n",
                            per64 / 1024.0, kQuarter, per256 / 1024.0, kReplicas, drift,
                            (drift > 1.005 || drift < 0.995)
                                ? "THE READING HAS NOT CONVERGED; raise the count before"
                                  " quoting it."
                                : "It has stopped moving, so the count is high enough.");
            }
            row("binCV CENSUS entry", censusMem, "2 wide + 2 descriptors + map");
            row("cv::cuda::StereoBM(64,9) synthetic", bmSynthetic, "UPPER reading");
            if (haveReal) row("cv::cuda::StereoBM(64,9) REAL frame", bmReal, "UPPER reading");

            std::printf("   [meter 1, binCV context only, never a cross-library"
                        " numerator] binary\n"
                        "   entry allocation sum = %.1f KB. Meter 1 and meter 2 do not"
                        " divide.\n",
                        static_cast<double>(binaryAllocSum) / 1024.0);

            const size_t bmQuote = haveReal ? bmReal : bmSynthetic;
            if (haveReal) {
                const double drift =
                    bmSynthetic > 0
                        ? static_cast<double>(bmReal) / static_cast<double>(bmSynthetic)
                        : 0.0;
                std::printf("   CONTENT DEPENDENCE: StereoBM reads %.3fx on the real frame"
                            " against the\n"
                            "   synthetic one. %s\n",
                            drift,
                            (drift > 1.02 || drift < 0.98)
                                ? "IT IS CONTENT-DEPENDENT -- the real reading is the"
                                  " number quoted."
                                : "It is NOT content-dependent: StereoBM sizes its output"
                                  " and its\n   internals by the image geometry and the"
                                  " disparity count, not by what\n   it finds. The real"
                                  " reading is still the one quoted.");
            } else {
                std::printf("   CONTENT DEPENDENCE: NOT CHECKED -- no real sequence blob"
                            " was pointed at\n"
                            "   BINCV_CUDA_ROLE_FRAMES, so only the synthetic reading"
                            " exists and it is\n   quoted as such.\n");
            }

            const auto verdict = [&](const char* name, size_t mine) {
                const double per = static_cast<double>(mine) / kReplicas;
                const double theirs = static_cast<double>(bmQuote) / kReplicas;
                const double u = unit > 0.0 ? static_cast<double>(mine) / unit : 0.0;
                const double tu = unit > 0.0 ? static_cast<double>(bmQuote) / unit : 0.0;
                if (u < 8.0 || tu < 8.0 || per <= 0.0) {
                    std::printf("   %s vs StereoBM: RATIO NOT QUOTED -- a side read under"
                                " eight of the\n   meter's own units.\n", name);
                    return;
                }
                std::printf("   %s vs StereoBM: %.3fx -- %s by %.3fx"
                            " (%.1f KB against %.1f KB).\n",
                            name, theirs / per,
                            theirs >= per ? "binCV SMALLER" : "StereoBM SMALLER",
                            theirs >= per ? theirs / per : per / theirs, per / 1024.0,
                            theirs / 1024.0);
            };
            verdict("BINARY entry", binaryMem);
            verdict("CENSUS entry", censusMem);
            std::printf("   Both ratios are LOWER BOUNDS on binCV's side and UPPER"
                        " readings on\n"
                        "   StereoBM's, for the reason stated above: GpuMat may pool and"
                        " pads its\n   pitch; binCV has nothing between the op and"
                        " cudaMalloc.\n");
            emitMem("stereo_binary_vs_stereobm", "752x480", binaryMem, bmQuote, step,
                    kReplicas, kReplicas);
            emitMem("stereo_census_vs_stereobm", "752x480", censusMem, bmQuote, step,
                    kReplicas, kReplicas);
            std::printf("MEM3,stereo_one_region,752x480,%zu,%zu,%zu,%zu,%zu,%d\n",
                        binaryMem, censusMem, bmSynthetic, bmReal, step, kReplicas);
        }
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
                        e.medianMs, ratio, overlap ? "(ranges OVERLAP)"
                                                   : "(ranges DISJOINT)");
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
        // THE SHIPPED ARM'S NUMBER. This binary runs only the arm production runs,
        // so it sizes for that arm: 380 bytes at 752x480 rather than the reference
        // arm's nextPow2(capacity) records. The frontend benchmark, which times
        // both arms against one buffer, asks for FastArm::Reference instead.
        const size_t fastScratchBytes =
            bc::fastScratchBytes(kW, kH, cap, bc::FastArm::Ordered);
        bc::DeviceArray<uint8_t> dfastScratch(fastScratchBytes);

        const uint32_t capped = 512;
        bc::DeviceArray<bc::DeviceFastCorner> dfastCapped(capped);
        bc::DeviceAppendCounter cappedCounter;
        const bc::DeviceFastCornerBuffer cappedBuf(dfastCapped.data(),
                                                   cappedCounter.devicePtr(), capped);
        const size_t cappedScratchBytes =
            bc::fastScratchBytes(kW, kH, capped, bc::FastArm::Ordered);
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
            // 256, not 32 and not 64. The footprint this row measures fell by ~2.5x
            // when the shipped arm stopped allocating the reference arm's scratch and
            // the corner record went from 16 bytes to 12, and at 32 sets binCV's side
            // no longer cleared the eight meter units printMemPair requires before it
            // will quote a ratio. But clearing eight units is not the same as
            // resolving: at 64 sets this side reads 13 units, so one unit of rounding
            // is 32 KB a frame on a 430 KB reading, and two harnesses that BOTH
            // cleared the rule quoted 1.615x and 1.714x for the same pair. At 256 the
            // unit is 8 KB a frame, this binary reads 54 and 85 units, the reading is
            // identical across five processes, and binCV's side lands within 3 KB of
            // its own allocation sum. The replica count follows the reading; the
            // reading does not follow the replica count.
            const int reps = 256;
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
                        bc::fastScratchBytes(kW, kH, cap, bc::FastArm::Ordered)));
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
            // The capacity is PRINTED from the variable rather than restated: the
            // label read "16384" for a run that used 32768, which is exactly the
            // kind of drift a hand-written label produces.
            char fastMemLabel[96];
            std::snprintf(fastMemLabel, sizeof(fastMemLabel),
                          "FAST working set, capacity %u (record %zu B, scratch %zu B)", cap,
                          sizeof(bc::DeviceFastCorner), fastScratchBytes);
            printMemPair(fastMemLabel, "752x480", binD, ocvD, step, reps, reps);
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
                    " 13. DESCRIPTOR MATCHING -- binCV cuda::matchDescriptors vs\n"
                    "     cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING)\n"
                    "=====================================================================\n"
                    " ROLE: 'nearest neighbour over binary descriptors with Lowe's ratio\n"
                    " test, for a whole query set, on device'. A binCV device matcher\n"
                    " EXISTS as of this round (cuda/sparseMatch.hpp), so the row this\n"
                    " file used to print -- 'no binCV device counterpart' -- is retired.\n"
                    "\n"
                    " WHAT THE TIMED REGION IS, AND WHY IT FAVOURS OpenCV. binCV's arm\n"
                    " is ONE launch that produces the ratio-tested result. OpenCV's arm\n"
                    " is knnMatchAsync(k=2) ALONE -- the ratio test that would turn its\n"
                    " two distances into the same answer is NOT in the bracket, and\n"
                    " neither is knnMatchConvert. So the OpenCV arm is timed doing\n"
                    " strictly LESS work than binCV's, and every ratio below is an\n"
                    " UPPER BOUND on OpenCV's standing. That is deliberate: a bar the\n"
                    " challenger cannot be accused of shaving is worth more than a bar\n"
                    " that is exactly fair, when the challenger wins anyway.\n"
                    "\n"
                    " TWO OpenCV ARMS, because the bar is the BEST existing option.\n"
                    " cv::cuda instantiates matchHamming_gpu at <unsigned char> and at\n"
                    " <int>, and cv::cuda::ORB emits CV_8U. Both are run over the SAME\n"
                    " descriptor bytes, reinterpreted, so the two arms differ only in\n"
                    " the type the kernel was instantiated at.\n"
                    "\n"
                    " CONTENT. Pseudorandom descriptor words. Neither kernel has a\n"
                    " data-dependent early-out -- both evaluate every (query, train)\n"
                    " pair -- so the cost is content-independent and a real descriptor\n"
                    " set would move neither arm. What content DOES decide is accuracy,\n"
                    " which this row does not measure and does not claim.\n");

        constexpr size_t kDescWords = 8;   // 256 bits, the ORB/BRIEF width
        const size_t mq[] = {470, 5000};
        const size_t mt[] = {470, 5000};
        const char* mName[] = {"470x470 (frontend scale -- DOES NOT DECIDE)",
                               "5000x5000 (map scale -- the deciding row)"};
        for (int g = 0; g < 2; ++g) {
            const size_t qn = mq[g], tn = mt[g];
            std::vector<uint32_t> qw(qn * kDescWords), tw(tn * kDescWords);
            for (auto& v : qw) v = static_cast<uint32_t>(nextByte()) |
                                   (static_cast<uint32_t>(nextByte()) << 8) |
                                   (static_cast<uint32_t>(nextByte()) << 16) |
                                   (static_cast<uint32_t>(nextByte()) << 24);
            for (auto& v : tw) v = static_cast<uint32_t>(nextByte()) |
                                   (static_cast<uint32_t>(nextByte()) << 8) |
                                   (static_cast<uint32_t>(nextByte()) << 16) |
                                   (static_cast<uint32_t>(nextByte()) << 24);

            bc::DeviceArray<uint32_t> dq(qw.size()), dt(tw.size());
            bc::DeviceArray<bc::DeviceDescriptorMatch> dm(qn);
            cudaMemcpyAsync(dq.data(), qw.data(), qw.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, gStream);
            cudaMemcpyAsync(dt.data(), tw.data(), tw.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, gStream);
            cudaStreamSynchronize(gStream);

            const auto qv = bc::descriptorSet(static_cast<const uint32_t*>(dq.data()), qn,
                                              kDescWords);
            const auto tv = bc::descriptorSet(static_cast<const uint32_t*>(dt.data()), tn,
                                              kDescWords);

            // The SAME BYTES on OpenCV's side, twice: as the CV_8U rows
            // cv::cuda::ORB emits, and as the CV_32S rows matchHamming_gpu<int>
            // is instantiated for.
            cv::Mat qm8(static_cast<int>(qn), static_cast<int>(kDescWords * 4), CV_8UC1,
                        qw.data());
            cv::Mat tm8(static_cast<int>(tn), static_cast<int>(kDescWords * 4), CV_8UC1,
                        tw.data());
            cv::Mat qm32(static_cast<int>(qn), static_cast<int>(kDescWords), CV_32SC1,
                         qw.data());
            cv::Mat tm32(static_cast<int>(tn), static_cast<int>(kDescWords), CV_32SC1,
                         tw.data());
            cv::cuda::GpuMat gq8, gt8, gq32, gt32, gRes;
            gq8.upload(qm8, gCvStream);
            gt8.upload(tm8, gCvStream);
            gq32.upload(qm32, gCvStream);
            gt32.upload(tm32, gCvStream);
            gCvStream.waitForCompletion();

            cv::Ptr<cv::cuda::DescriptorMatcher> bf =
                cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

            const auto binArm = [&] {
                bc::matchDescriptors(qv, tv, dm.data(), 80, gStream);
            };
            const auto ocv8 = [&] {
                bf->knnMatchAsync(gq8, gt8, gRes, 2, cv::noArray(), gCvStream);
            };
            bool has32 = true;
            try {
                bf->knnMatchAsync(gq32, gt32, gRes, 2, cv::noArray(), gCvStream);
                gCvStream.waitForCompletion();
            } catch (const cv::Exception&) {
                has32 = false;
            }

            const int it = g == 0 ? 20 : 5;
            const PairedTiming p8 = timeKernelPaired(ocv8, binArm, it, it, kRounds, gStream);
            printRole("descriptor match, 256-bit, ratio 0.8 -- OpenCV arm is CV_8U",
                      "cv::cuda BFMatcher knnMatchAsync(k=2), CV_8U",
                      "bincv::cuda::matchDescriptors (ratio test INCLUDED)", p8, floor,
                      mName[g]);
            emitRow("matcher_u8", mName[g], p8);
            if (has32) {
                const auto ocv32 = [&] {
                    bf->knnMatchAsync(gq32, gt32, gRes, 2, cv::noArray(), gCvStream);
                };
                const PairedTiming p32 =
                    timeKernelPaired(ocv32, binArm, it, it, kRounds, gStream);
                printRole("the same, OpenCV arm instantiated at CV_32S",
                          "cv::cuda BFMatcher knnMatchAsync(k=2), CV_32S",
                          "bincv::cuda::matchDescriptors (ratio test INCLUDED)", p32, floor,
                          mName[g]);
                emitRow("matcher_s32", mName[g], p32);
            } else {
                std::printf("\n   cv::cuda BFMatcher refused CV_32S in this build -- the\n"
                            "   CV_8U arm is the only bar here, and it is the one\n"
                            "   cv::cuda::ORB emits anyway.\n");
                std::printf("NOARM,matcher_s32,%s,BFMatcher refused CV_32S\n", mName[g]);
            }

            if (g == 1) {
                // MEMORY, meter 2 on both sides. A 256-bit descriptor is a 32-byte
                // row and GpuMat pitches; that is the whole of this reading and it
                // is printed rather than argued.
                std::vector<std::unique_ptr<bc::DeviceArray<uint32_t>>> bq, bt;
                std::vector<std::unique_ptr<bc::DeviceArray<bc::DeviceDescriptorMatch>>> bo;
                // TWO REPLICA COUNTS, for the reason printMemPair documents: at
                // 5000 descriptors binCV's whole set is ~400 KB and needs many
                // replicas to clear the driver's 2 MB unit eight times, while
                // OpenCV's is ~8 MB and 128 of those would not fit on this card.
                // Each side is divided by its own count.
                const int reps = 128;
                const int ocvReps = 48;
                const size_t binMem = meterScope(
                    [&](int) {
                        bq.push_back(std::make_unique<bc::DeviceArray<uint32_t>>(qn * kDescWords));
                        bt.push_back(std::make_unique<bc::DeviceArray<uint32_t>>(tn * kDescWords));
                        bo.push_back(std::make_unique<bc::DeviceArray<bc::DeviceDescriptorMatch>>(qn));
                        bc::matchDescriptors(
                            bc::descriptorSet(static_cast<const uint32_t*>(bq.back()->data()),
                                              qn, kDescWords),
                            bc::descriptorSet(static_cast<const uint32_t*>(bt.back()->data()),
                                              tn, kDescWords),
                            bo.back()->data(), 80, gStream);
                    },
                    reps);
                cudaStreamSynchronize(gStream);
                bq.clear(); bt.clear(); bo.clear();
                cudaDeviceSynchronize();

                std::vector<cv::cuda::GpuMat> oq, ot, orr;
                const size_t ocvMem = meterScope(
                    [&](int) {
                        oq.emplace_back();
                        ot.emplace_back();
                        orr.emplace_back();
                        oq.back().upload(qm8);
                        ot.back().upload(tm8);
                        bf->knnMatchAsync(oq.back(), ot.back(), orr.back(), 2);
                        cudaDeviceSynchronize();
                    },
                    ocvReps);
                oq.clear(); ot.clear(); orr.clear();
                printMemPair("descriptor matching working set (query + train + result)",
                             "5000x5000", binMem, ocvMem, step, reps, ocvReps);
                emitMem("matcher", "5000x5000", binMem, ocvMem, step, reps, ocvReps);
                std::printf("   A 256-bit descriptor is a 32-BYTE ROW and GpuMat pitches a\n"
                            "   row to a 512-byte multiple. That pitch is most of this\n"
                            "   reading, and it is a property of the container rather than\n"
                            "   of the matcher -- said here so the number is not read as a\n"
                            "   claim about the kernel.\n");
            }
        }
    }

    // ======================================================================
    // 14, 15, 16 -- ROUND 3'S BARS. Every one of them needs the real sequence,
    // so they share one parse of it and one device tracker state.
    // ======================================================================
    const bool wantR3 = want("lk") || want("sequence");
    if (want("censusstereo")) {
        std::printf("\n=====================================================================\n"
                    " 15. CENSUS DENSE DISPARITY -- binCV's census entry vs\n"
                    "     cv::cuda::StereoBM, RE-TAKEN after the matcher changed\n"
                    "=====================================================================\n"
                    " ROLE: 'a dense disparity map from a rectified wide-input pair,\n"
                    " resident on device'. Section 7 runs the BINARY path, which is\n"
                    " where this library's structural claim lives. This section runs the\n"
                    " CENSUS path, which is the entry a wide-input caller meets first and\n"
                    " where binCV has NO structural advantage -- census EXPANDS data,\n"
                    " 8 bits per pixel in and 24 out, and the layout that makes it fast\n"
                    " is the conventional one-word-per-pixel descriptor.\n"
                    "\n"
                    " WHY RE-TAKEN. The packed census matcher was replaced this round by\n"
                    " a warp-cooperative separable box arm, and the shipped report's\n"
                    " census rows predate it. Three rows are printed: the matcher alone,\n"
                    " the whole entry (two transforms plus the match, which is what a\n"
                    " caller pays), and the arm's own off-switch ratio inside THIS\n"
                    " protocol so the family's 2.4-2.5x is reproduced or contradicted\n"
                    " here rather than quoted from another process.\n"
                    " Both sides on one explicit stream: StereoBM calls\n"
                    " cudaDeviceSynchronize() on the null stream in each of its three\n"
                    " kernels.\n");

        constexpr size_t kCensusK = 24;   // kCensus5x5
        std::vector<uint8_t> clw = makeFrame(kW, kH);
        std::vector<uint8_t> crw(kW * kH, 0);
        for (size_t y = 0; y < kH; ++y)
            for (size_t x = 0; x + 21 < kW; ++x) crw[y * kW + x] = clw[y * kW + x + 21];

        bincv::DenseDisparityParams cp;
        cp.maxDisparity = 64;
        cp.winWidth = 9;
        cp.winHeight = 9;

        bc::DeviceImage<uint8_t> cdL(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint8_t> cdR(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint32_t> cDescL(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint32_t> cDescR(static_cast<int>(kW), static_cast<int>(kH));
        bc::DeviceImage<uint8_t> cDisp(static_cast<int>(kW), static_cast<int>(kH));
        bc::uploadImage<uint8_t>(clw.data(), kW, kH, kW, cdL.view(), gStream);
        bc::uploadImage<uint8_t>(crw.data(), kW, kH, kW, cdR.view(), gStream);
        bc::censusTransformPacked<kCensusK>(cdL.constView(), bincv::kCensus5x5,
                                            cDescL.view(), gStream);
        bc::censusTransformPacked<kCensusK>(cdR.constView(), bincv::kCensus5x5,
                                            cDescR.view(), gStream);
        cudaStreamSynchronize(gStream);

        cv::cuda::GpuMat cgl, cgr, cgd;
        cgl.upload(cv::Mat(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1, clw.data()));
        cgr.upload(cv::Mat(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1, crw.data()));
        auto cbm = cv::cuda::createStereoBM(64, 9);
        cbm->compute(cgl, cgr, cgd, gCvStream);
        cudaStreamSynchronize(gStream);

        const auto ocvArm = [&] { cbm->compute(cgl, cgr, cgd, gCvStream); };
        const auto matchArm = [&] {
            bc::denseDisparityCensusPacked(cDescL.constView(), cDescR.constView(), cp,
                                           cDisp.view(), gStream);
        };
        const auto entryArm = [&] {
            bc::censusTransformPacked<kCensusK>(cdL.constView(), bincv::kCensus5x5,
                                                cDescL.view(), gStream);
            bc::censusTransformPacked<kCensusK>(cdR.constView(), bincv::kCensus5x5,
                                                cDescR.view(), gStream);
            bc::denseDisparityCensusPacked(cDescL.constView(), cDescR.constView(), cp,
                                           cDisp.view(), gStream);
        };

        bc::impl::densePackedBoxEnabled() = true;
        const PairedTiming pm = timeKernelPaired(ocvArm, matchArm, 10, 10, kRounds, gStream);
        printRole("census MATCHER only, 64 disparities, 9x9, K=24",
                  "cv::cuda::StereoBM(64, 9)",
                  "bincv denseDisparityCensusPacked (warp-box arm)", pm, floor, "752x480");
        emitRow("census_match", "752x480", pm);

        const PairedTiming pe = timeKernelPaired(ocvArm, entryArm, 10, 10, kRounds, gStream);
        printRole("census ENTRY: two transforms + the match -- what a caller pays",
                  "cv::cuda::StereoBM(64, 9)",
                  "bincv censusTransformPacked x2 + denseDisparityCensusPacked", pe, floor,
                  "752x480");
        emitRow("census_entry", "752x480", pe);

        // The arm's own off-switch, binCV to binCV, inside THIS protocol.
        const auto refArm = [&] {
            bc::impl::densePackedBoxEnabled() = false;
            bc::denseDisparityCensusPacked(cDescL.constView(), cDescR.constView(), cp,
                                           cDisp.view(), gStream);
            bc::impl::densePackedBoxEnabled() = true;
        };
        const PairedTiming po = timeKernelPaired(refArm, matchArm, 10, 10, kRounds, gStream);
        std::printf("\n OFF-SWITCH, binCV to binCV: the shipped packed matcher against\n"
                    " the warp-box arm that replaced it.\n");
        printArmVsFloor("reference arm (densePackedBoxEnabled = false)", po.a, floor,
                        "kernel");
        printArmVsFloor("warp-box arm (densePackedBoxEnabled = true)", po.b, floor,
                        "kernel");
        std::printf("   ratio box/reference: median %5.3fx <- QUOTE THIS  (box %5.2fx"
                    " faster)  geomean %5.3fx  range %5.3f-%5.3fx\n",
                    po.ratioMedian, po.ratioMedian > 0.0 ? 1.0 / po.ratioMedian : 0.0,
                    po.ratioGeoMean, po.ratioMin, po.ratioMax);
        printPairedSignAndSeparation(po);
        printPairedVerdict(po);
        emitRow("census_offswitch", "752x480", po);
        bc::impl::densePackedBoxEnabled() = true;

        // Memory, meter 2 on both sides.
        std::vector<std::unique_ptr<bc::DeviceImage<uint8_t>>> ci;
        std::vector<std::unique_ptr<bc::DeviceImage<uint32_t>>> cd;
        std::vector<std::unique_ptr<bc::DeviceImage<uint8_t>>> cm;
        const int cReps = 64;
        const size_t cBinMem = meterScope(
            [&](int) {
                ci.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                    static_cast<int>(kW), static_cast<int>(kH)));
                ci.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                    static_cast<int>(kW), static_cast<int>(kH)));
                cd.push_back(std::make_unique<bc::DeviceImage<uint32_t>>(
                    static_cast<int>(kW), static_cast<int>(kH)));
                cd.push_back(std::make_unique<bc::DeviceImage<uint32_t>>(
                    static_cast<int>(kW), static_cast<int>(kH)));
                cm.push_back(std::make_unique<bc::DeviceImage<uint8_t>>(
                    static_cast<int>(kW), static_cast<int>(kH)));
                bc::censusTransformPacked<kCensusK>(ci[ci.size() - 2]->constView(),
                                                    bincv::kCensus5x5, cd[cd.size() - 2]->view(),
                                                    gStream);
                bc::denseDisparityCensusPacked(cd[cd.size() - 2]->constView(),
                                               cd.back()->constView(), cp, cm.back()->view(),
                                               gStream);
            },
            cReps);
        cudaStreamSynchronize(gStream);
        ci.clear(); cd.clear(); cm.clear();
        cudaDeviceSynchronize();

        std::vector<cv::cuda::GpuMat> sl, sr, sd;
        std::vector<cv::Ptr<cv::cuda::StereoBM>> sbm;
        const size_t cOcvMem = meterScope(
            [&](int) {
                sl.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                sr.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                sd.emplace_back();
                sbm.push_back(cv::cuda::createStereoBM(64, 9));
                sbm.back()->compute(sl.back(), sr.back(), sd.back());
                cudaDeviceSynchronize();
            },
            cReps);
        sl.clear(); sr.clear(); sd.clear(); sbm.clear();
        printMemPair("census entry working set (two wide frames + two descriptor"
                     " images + map) vs StereoBM's",
                     "752x480", cBinMem, cOcvMem, step, cReps, cReps);
        emitMem("census_entry", "752x480", cBinMem, cOcvMem, step, cReps, cReps);
        std::printf("   SCOPE, and it is not a footnote: the census path is where binCV\n"
                    "   has no structural advantage. Whatever this row says, the\n"
                    "   library's claim lives in section 7's binary path.\n");
    }

    if (wantR3) {
        bincv::SequenceHeader seqH{};
        bool seqOk = false;
        if (!blob.empty()) {
            seqH = bincv::readSequenceHeader(blob.data(), blob.size());
            seqOk = seqH.valid && seqH.mode == bincv::kSequenceMode8Bit &&
                    seqH.width == kW && seqH.height == kH && seqH.frameCount >= 2;
        }
        const auto frameAt = [&](size_t i) -> const uint8_t* {
            const bincv::SequenceFrameRange r =
                bincv::sequenceFrame(seqH, blob.data(), blob.size(), i);
            return r.valid ? r.data : nullptr;
        };

        if (!seqOk) {
            std::printf("\n=====================================================================\n"
                        " 14 and 16 -- SKIPPED: NO REAL SEQUENCE\n"
                        "=====================================================================\n"
                        " Both the Lucas-Kanade role row and the sequence-level number are\n"
                        " decided by corner density, and a synthetic frame is a different\n"
                        " workload -- measured elsewhere in this project as a verdict that\n"
                        " INVERTED between synthetic and real content. So neither is run\n"
                        " on makeFrame's smoothed noise and neither is substituted.\n"
                        " Point BINCV_CUDA_ROLE_FRAMES at a %zux%zu 8-bit BSQ1 blob with\n"
                        " at least two frames (scripts/make_sequence_blob.py --mode 8bit).\n",
                        kW, kH);
            std::printf("SKIP,lk,no real sequence blob\nSKIP,sequence,no real sequence blob\n");
        } else {
        DeviceTracker dev;
        bincv::LKParams lkp;
        lkp.winWidth = kLkWin;
        lkp.winHeight = kLkWin;
        lkp.maxIterations = kLkIterCap;

        deviceLoadFrame(dev, frameAt(0), dev.prev, gStream);
        deviceLoadFrame(dev, frameAt(1), dev.next, gStream);
        deviceDerivatives(dev, dev.prev, gStream);
        cudaStreamSynchronize(gStream);

        bc::DeviceLKLevel lkLevels[kLkLevels];
        deviceBuildLevels(dev, lkLevels);

        // TWO real keypoint sets from the detector's own output, at the
        // reference frontend's spacing and at a denser one. Corner density is
        // what decides this comparison, so the row prints the count it got
        // rather than a round number reached by tuning the detector.
        const uint32_t sparseN = deviceDetect(dev, 33.33333333333, dev.prevXY.data(), gStream);
        bc::DeviceArray<float> denseXY(2 * kTrackCapacity);
        const uint32_t denseN = deviceDetect(dev, 6.0, denseXY.data(), gStream);

        // =================================================================
        // 14. LUCAS-KANADE
        // =================================================================
        if (want("lk")) {
            std::printf("\n=====================================================================\n"
                        " 14. SPARSE LUCAS-KANADE -- binCV cuda::calcOpticalFlowPyrLKAsync\n"
                        "     vs cv::cuda::SparsePyrLKOpticalFlow (cudaoptflow)\n"
                        "=====================================================================\n"
                        " ROLE: 'track a keypoint set from one frame to the next over a\n"
                        " pyramid, on device'. BOTH SIDES' PYRAMIDS ARE RESIDENT AT ENTRY\n"
                        " and neither arm builds one inside the bracket -- cv::cuda's\n"
                        " calc() overload that takes std::vector<GpuMat> pyramids is used\n"
                        " for exactly that reason, so the row is a like-for-like rather\n"
                        " than a subtraction estimate.\n"
                        " SAME WINDOW BOTH SIDES: 31x31, %d levels, iteration cap %d,\n"
                        " err off on both, both free-running (cv::cuda breaks on its own\n"
                        " 0.01-pixel convergence test; binCV on its own epsilon rule --\n"
                        " forcing equal iteration counts would be forcing a DIFFERENT\n"
                        " algorithm on one of them).\n"
                        " keypoints: %u at the reference minDistance of 33.33 px, %u at\n"
                        " 6 px, both from goodFeaturesToTrack on frame 0 of the REAL\n"
                        " sequence.\n",
                        kLkLevels, kLkIterCap, sparseN, denseN);

            cv::Mat hp(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                       const_cast<uint8_t*>(frameAt(0)));
            cv::Mat hn(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                       const_cast<uint8_t*>(frameAt(1)));
            std::vector<cv::cuda::GpuMat> prevPyr(static_cast<size_t>(kLkLevels)),
                nextPyr(static_cast<size_t>(kLkLevels));
            prevPyr[0].upload(hp, gCvStream);
            nextPyr[0].upload(hn, gCvStream);
            for (size_t l = 1; l < static_cast<size_t>(kLkLevels); ++l) {
                cv::cuda::pyrDown(prevPyr[l - 1], prevPyr[l], gCvStream);
                cv::cuda::pyrDown(nextPyr[l - 1], nextPyr[l], gCvStream);
            }
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> cvlk =
                cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(kLkWin, kLkWin),
                                                         kLkLevels - 1, kLkIterCap, false);

            const auto lkRow = [&](const char* geom, const float* dXY, uint32_t count) {
                std::vector<float> hostPts(2 * count);
                cudaMemcpy(hostPts.data(), dXY, 2 * count * sizeof(float),
                           cudaMemcpyDeviceToHost);
                cv::Mat ptsMat(1, static_cast<int>(count), CV_32FC2, hostPts.data());
                cv::cuda::GpuMat gPts, gNextPts, gStatus;
                gPts.upload(ptsMat, gCvStream);
                gCvStream.waitForCompletion();

                bc::DeviceLKTracks tr;
                tr.dPrevXY = dXY;
                tr.dNextXY = dev.nextXY.data();
                tr.dStatus = dev.status.data();
                tr.dErr = nullptr;
                tr.count = count;

                const auto cvArm = [&] {
                    cvlk->calc(prevPyr, nextPyr, gPts, gNextPts, gStatus, cv::noArray(),
                               gCvStream);
                };
                const auto binArm = [&] {
                    bc::calcOpticalFlowPyrLKAsync(lkLevels, kLkLevels, tr, lkp, gStream);
                };
                const PairedTiming pl = timeKernelPaired(cvArm, binArm, 50, 50, kRounds,
                                                         gStream);
                char lbl[128];
                std::snprintf(lbl, sizeof(lbl), "sparse LK, 31x31, %d levels, %u keypoints",
                              kLkLevels, count);
                printRole(lbl, "cv::cuda::SparsePyrLKOpticalFlow (pyramids resident)",
                          "bincv::cuda::calcOpticalFlowPyrLKAsync (ladder resident)", pl,
                          floor, geom);
                emitRow("lk", geom, pl);
                printEnqueue("lk", geom, "cv::cuda SparsePyrLK", "bincv LK",
                             timeHostEnqueue(cvArm, 20), timeHostEnqueue(binArm, 20), pl);
                std::printf("   READ THIS RATIO WITH THE LAUNCH FLOOR UNDER IT. binCV\n"
                            "   issues ONE launch for all four levels; cv::cuda issues one\n"
                            "   per level plus multiply and setTo. The optical-flow family\n"
                            "   profiled both kernels in one ncu run and found binCV's\n"
                            "   KERNEL WORK is a 1.56x LOSS at 61 keypoints -- the wall\n"
                            "   clock advantage is the launch shape, and on a host whose\n"
                            "   launch is cheap it would shrink.\n");
                return pl;
            };
            // A KEYPOINT-COUNT SWEEP, not two points. binCV issues one launch
            // with one warp per keypoint; cv::cuda issues one launch per level
            // with 256 threads per keypoint. Those two shapes cross somewhere,
            // and a bar quoted at one count on either side of that crossing is
            // a bar quoted at the count that flattered it. The sweep runs on the
            // PREFIX of one detected set, so every row is the same corners with
            // more of them and nothing else changes.
            std::printf("\n THE SWEEP, and why it is a sweep: binCV = one launch, one warp\n"
                        " per keypoint; cv::cuda = one launch per level, 256 threads per\n"
                        " keypoint. Those shapes cross. Rows below are PREFIXES of the\n"
                        " minDistance-6 set, so only the count moves.\n");
            lkRow("752x480, minDistance 33.33 (the reference frontend's spacing)",
                  dev.prevXY.data(), sparseN);
            const uint32_t sweep[] = {64, 128, 256, 512, 1024};
            for (uint32_t n : sweep) {
                if (n > denseN) continue;
                char g[64];
                std::snprintf(g, sizeof(g), "752x480, %u pts (prefix of minDistance 6)", n);
                lkRow(g, denseXY.data(), n);
            }
            lkRow("752x480, minDistance 6, the whole set", denseXY.data(), denseN);

            // Memory, meter 2 on both sides: the tracker's resident state.
            // binCV's side is the two binary ladders, the previous frame's
            // derivative planes and the keypoint arrays. cv::cuda's is the two
            // CV_8U pyramids and its keypoint GpuMats. The SENSOR STAGE's wide
            // frames are excluded on binCV's side because cv::cuda has no
            // counterpart to them in this row; section 16 puts them back,
            // because a SEQUENCE pays for them.
            // Separate counts again: binCV's tracker state is ~384 KB and
            // OpenCV's ~1.3 MB, so eight of the meter's units cost very
            // different replica counts. Each side is divided by its own.
            const int lkReps = 96;
            const int lkOcvReps = 32;
            std::vector<std::unique_ptr<Ladder>> bp, bn;
            std::vector<std::unique_ptr<DerivLadder>> bd;
            std::vector<std::unique_ptr<bc::DeviceArray<float>>> bxy;
            std::vector<std::unique_ptr<bc::DeviceArray<uint8_t>>> bst;
            const size_t lkBin = meterScope(
                [&](int) {
                    bp.push_back(std::make_unique<Ladder>(static_cast<int>(kW),
                                                          static_cast<int>(kH)));
                    bn.push_back(std::make_unique<Ladder>(static_cast<int>(kW),
                                                          static_cast<int>(kH)));
                    bd.push_back(std::make_unique<DerivLadder>());
                    bxy.push_back(std::make_unique<bc::DeviceArray<float>>(4 * denseN));
                    bst.push_back(std::make_unique<bc::DeviceArray<uint8_t>>(denseN));
                },
                lkReps);
            bp.clear(); bn.clear(); bd.clear(); bxy.clear(); bst.clear();
            cudaDeviceSynchronize();

            std::vector<std::vector<cv::cuda::GpuMat>> op, on;
            std::vector<cv::cuda::GpuMat> opts, onext, ostat;
            std::vector<cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>> olk;
            const size_t lkOcv = meterScope(
                [&](int) {
                    op.emplace_back(static_cast<size_t>(kLkLevels));
                    on.emplace_back(static_cast<size_t>(kLkLevels));
                    op.back()[0].upload(hp);
                    on.back()[0].upload(hn);
                    for (size_t l = 1; l < static_cast<size_t>(kLkLevels); ++l) {
                        cv::cuda::pyrDown(op.back()[l - 1], op.back()[l]);
                        cv::cuda::pyrDown(on.back()[l - 1], on.back()[l]);
                    }
                    opts.emplace_back();
                    onext.emplace_back();
                    ostat.emplace_back();
                    std::vector<float> pts(2 * denseN, 100.0f);
                    opts.back().upload(cv::Mat(1, static_cast<int>(denseN), CV_32FC2,
                                               pts.data()));
                    olk.push_back(cv::cuda::SparsePyrLKOpticalFlow::create(
                        cv::Size(kLkWin, kLkWin), kLkLevels - 1, kLkIterCap, false));
                    olk.back()->calc(op.back(), on.back(), opts.back(), onext.back(),
                                     ostat.back(), cv::noArray());
                    cudaDeviceSynchronize();
                },
                lkOcvReps);
            op.clear(); on.clear(); opts.clear(); onext.clear(); ostat.clear(); olk.clear();
            printMemPair("sparse LK tracker resident state (both frames at every"
                         " level + keypoint arrays)",
                         "752x480", lkBin, lkOcv, step, lkReps, lkOcvReps);
            emitMem("lk", "752x480", lkBin, lkOcv, step, lkReps, lkOcvReps);
            std::printf("   HONEST WEAKNESS, printed at the number: binCV STORES the\n"
                        "   ternary derivative planes where cv::cuda recomputes them\n"
                        "   in-kernel, which is most of binCV's side of this reading.\n"
                        "   A fused-derivative variant would take it further and is not\n"
                        "   in this measurement.\n");
        }

        // =================================================================
        // 16. THE SEQUENCE-LEVEL NUMBER
        // =================================================================
        if (want("sequence")) {
            size_t nFrames = 400;
            if (const char* e = std::getenv("BINCV_CUDA_ROLE_SEQ_FRAMES")) {
                const long v = std::atol(e);
                if (v > 1) nFrames = static_cast<size_t>(v);
            }
            if (nFrames > seqH.frameCount) nFrames = seqH.frameCount;
            int passes = 3;
            if (const char* e = std::getenv("BINCV_CUDA_ROLE_SEQ_PASSES")) {
                const long v = std::atol(e);
                if (v > 0) passes = static_cast<int>(v);
            }
            const size_t redetect = 10;

            std::printf("\n=====================================================================\n"
                        " 16. THE SEQUENCE-LEVEL NUMBER -- a resident device tracker\n"
                        "     against the host tracker, over %zu real frames\n"
                        "=====================================================================\n"
                        " THE RULE, WRITTEN HERE BEFORE THE LOOP RUNS. This row is not a\n"
                        " kernel ratio and must not be read as one. It is the whole\n"
                        " per-frame cost a caller pays, wall clock, on both arms:\n"
                        "   sensor stage (median + edge threshold) -> pyramid ladder ->\n"
                        "   the previous frame's derivative planes -> LK track,\n"
                        " with a detection every %zu frames on BOTH arms. The device arm\n"
                        " additionally pays the host-to-device upload of every frame,\n"
                        " inside its timing, because the frame has to arrive.\n"
                        "\n"
                        " ONE SYNCHRONIZE PER FRAME on the device arm, and it is in the\n"
                        " clock. A tracker whose output nobody waits for is not a tracker;\n"
                        " removing that synchronize would measure enqueue.\n"
                        "\n"
                        " WHAT DECIDES IT: speed AND peak device memory, together. Faster\n"
                        " but larger does not settle it, and neither does smaller but\n"
                        " slower -- the two are printed side by side and the verdict names\n"
                        " both. There is no project-wide ratio that ships this and none is\n"
                        " invented here.\n"
                        "\n"
                        " A DECLARED SIMPLIFICATION, stated before measuring: the fixed\n"
                        " re-detection cadence replaces the frontend's adaptive\n"
                        " 'detect when live tracks fall below 60' policy. The adaptive\n"
                        " policy needs the surviving count on the host every frame, which\n"
                        " would put a device-to-host round trip in the device arm's loop\n"
                        " and make the two arms do different work. The cadence is run at\n"
                        " TWO values so it can be seen not to decide the answer.\n"
                        "\n"
                        " THE HOST ARM IS NOT TIMING-GRADE ON THIS MACHINE (30-130%%\n"
                        " spread under WSL2, per this project's own record). The device\n"
                        " arm's wall clock is taken on the same host and the same loop,\n"
                        " so the RATIO carries that noise too. It is reported with both\n"
                        " arms' full ranges over %d passes and the reader is told when\n"
                        " they overlap.\n",
                        nFrames, redetect, passes);

            HostTracker host(static_cast<int>(kW), static_cast<int>(kH));

            std::vector<bincv::Corner> hostCorners(kRankCapacity);
            std::vector<bincv::Point2f> hPrevPts(kTrackCapacity), hNextPts(kTrackCapacity);
            std::vector<uint8_t> hStatus(kTrackCapacity);
            bincv::GoodFeaturesParams hgf;
            hgf.maxCorners = static_cast<int>(kTrackCapacity);
            hgf.minDistance = 33.33333333333;

            // ============================================================
            // WHAT IS COMPARED MUST AGREE BEFORE IT IS TIMED.
            // Two arms that track different keypoint sets are not two
            // measurements of one operation, they are two operations. This
            // block runs ONE frame through each arm and compares, stage by
            // stage, WHOLE WORDS -- so padding bits, which word-wise
            // reductions count, are compared too and not only pixels. It
            // prints a mismatch rather than asserting one away.
            // ============================================================
            {
                deviceLoadFrame(dev, frameAt(0), dev.prev, gStream);
                deviceLoadFrame(dev, frameAt(1), dev.next, gStream);
                deviceDerivatives(dev, dev.prev, gStream);
                cudaStreamSynchronize(gStream);
                host.seed(frameAt(0));
                host.loadFrame(frameAt(1));
                host.derivatives();

                bincv::BinMat<HW> dBits(static_cast<int>(kW), static_cast<int>(kH));
                bc::download(dev.prev.levelAt(0).block(), dBits.view(), gStream);
                cudaStreamSynchronize(gStream);
                size_t bitWords = 0, bitDiff = 0;
                {
                    const HW* a = dBits.data();
                    const HW* b = host.prev.level<0>().data();
                    const size_t n = dBits.sizeInWords();
                    bitWords = n;
                    for (size_t i = 0; i < n; ++i)
                        if (a[i] != b[i]) ++bitDiff;
                }

                bincv::SignedQuantMat<1, HW> dDx(static_cast<int>(kW), static_cast<int>(kH));
                bincv::SignedQuantMat<1, HW> dDy(static_cast<int>(kW), static_cast<int>(kH));
                bc::download(dev.deriv.dx[0].constView(),
                             bincv::BinMatView<HW>(dDx.data(), kW, 2 * kH,
                                                   dDx.getAlignedWidth()), gStream);
                bc::download(dev.deriv.dy[0].constView(),
                             bincv::BinMatView<HW>(dDy.data(), kW, 2 * kH,
                                                   dDy.getAlignedWidth()), gStream);
                cudaStreamSynchronize(gStream);
                size_t derWords = 0, derDiff = 0;
                {
                    const size_t n = dDx.sizeInWords();
                    derWords = 2 * n;
                    for (size_t i = 0; i < n; ++i) {
                        if (dDx.data()[i] != host.dx0.data()[i]) ++derDiff;
                        if (dDy.data()[i] != host.dy0.data()[i]) ++derDiff;
                    }
                }

                const uint32_t dN = deviceDetect(dev, hgf.minDistance, dev.prevXY.data(),
                                                 gStream);
                std::vector<float> dPts(2 * dN);
                cudaMemcpy(dPts.data(), dev.prevXY.data(), 2 * dN * sizeof(float),
                           cudaMemcpyDeviceToHost);
                bincv::ResponseMap ring0{host.ring.data(), kW, bincv::kResponseRingRows, kW};
                const bincv::CornerResult hr = bincv::goodFeaturesToTrackStreaming<HW>(
                    host.dx0, host.dy0, hgf, ring0, hostCorners.data(), hostCorners.size());
                size_t posDiff = 0;
                const size_t common = dN < hr.count ? dN : hr.count;
                for (size_t i = 0; i < common; ++i) {
                    if (static_cast<double>(dPts[2 * i]) !=
                            static_cast<double>(hostCorners[i].x) ||
                        static_cast<double>(dPts[2 * i + 1]) !=
                            static_cast<double>(hostCorners[i].y))
                        ++posDiff;
                }

                std::printf("\n AGREEMENT CHECK, frame 0/1 of the real sequence, before a\n"
                            " single timing number is taken:\n");
                std::printf("   level-0 binary frame        %8zu words compared, %zu differ\n",
                            bitWords, bitDiff);
                std::printf("   dx0 / dy0 ternary planes    %8zu words compared, %zu differ\n",
                            derWords, derDiff);
                std::printf("   detector, minDistance %.2f: device %u corners, host %zu"
                            " corners%s\n",
                            hgf.minDistance, dN, hr.count,
                            hr.candidatesTruncated ? "  (host reports its candidate pool"
                                                     " TRUNCATED)" : "");
                std::printf("   positions over the %zu ranks both produced: %zu differ\n",
                            common, posDiff);
                std::printf("AGREE,%zu,%zu,%zu,%zu,%u,%zu,%zu,%d\n", bitWords, bitDiff,
                            derWords, derDiff, dN, hr.count, posDiff,
                            hr.candidatesTruncated ? 1 : 0);
                if (bitDiff != 0 || derDiff != 0 || static_cast<size_t>(dN) != hr.count ||
                    posDiff != 0) {
                    std::printf("   *** THE TWO ARMS DO NOT SEE THE SAME THING. Every ratio\n"
                                "   *** below is reported anyway, and is reported as NOT\n"
                                "   *** like-for-like. This is a finding, not a nuisance:\n"
                                "   *** the device corner op is documented bit-exact\n"
                                "   *** against the host's, and here it is not.\n");
                }
            }


            const auto runCadence = [&](size_t cadence) {
                std::vector<double> devSamples, hostSamples;
                size_t devTracked = 0, hostTracked = 0, dets = 0;
                uint32_t devCount = 0;
                size_t hostCount = 0;

                for (int pass = 0; pass < passes; ++pass) {
                    // ---- the device arm -------------------------------------
                    deviceLoadFrame(dev, frameAt(0), dev.next, gStream);
                    cudaStreamSynchronize(gStream);
                    devCount = 0;
                    devTracked = 0;
                    dets = 0;
                    const auto d0 = std::chrono::steady_clock::now();
                    for (size_t f = 1; f < nFrames; ++f) {
                        std::swap(dev.prev, dev.next);
                        deviceLoadFrame(dev, frameAt(f), dev.next, gStream);
                        deviceDerivatives(dev, dev.prev, gStream);
                        if ((f - 1) % cadence == 0) {
                            devCount = deviceDetect(dev, hgf.minDistance,
                                                    dev.prevXY.data(), gStream);
                            ++dets;
                        }
                        bc::DeviceLKLevel lv[kLkLevels];
                        deviceBuildLevels(dev, lv);
                        bc::DeviceLKTracks tr;
                        tr.dPrevXY = dev.prevXY.data();
                        tr.dNextXY = dev.nextXY.data();
                        tr.dStatus = dev.status.data();
                        tr.dErr = nullptr;
                        tr.count = devCount;
                        bc::calcOpticalFlowPyrLKAsync(lv, kLkLevels, tr, lkp, gStream);
                        cudaStreamSynchronize(gStream);
                        devTracked += devCount;
                    }
                    const auto d1 = std::chrono::steady_clock::now();
                    devSamples.push_back(
                        std::chrono::duration<double, std::milli>(d1 - d0).count() /
                        static_cast<double>(nFrames - 1));

                    // ---- the host arm ---------------------------------------
                    host.seed(frameAt(0));
                    hostCount = 0;
                    hostTracked = 0;
                    const auto h0 = std::chrono::steady_clock::now();
                    for (size_t f = 1; f < nFrames; ++f) {
                        host.loadFrame(frameAt(f));
                        host.derivatives();
                        if ((f - 1) % cadence == 0) {
                            bincv::ResponseMap ringMap{host.ring.data(), kW,
                                                       bincv::kResponseRingRows, kW};
                            const bincv::CornerResult r =
                                bincv::goodFeaturesToTrackStreaming<HW>(
                                    host.dx0, host.dy0, hgf, ringMap, hostCorners.data(),
                                    hostCorners.size());
                            hostCount = r.count < kTrackCapacity ? r.count : kTrackCapacity;
                            for (size_t i = 0; i < hostCount; ++i)
                                hPrevPts[i] = bincv::Point2f{
                                    static_cast<float>(hostCorners[i].x),
                                    static_cast<float>(hostCorners[i].y)};
                        }
                        if (hostCount > 0) {
                            bincv::calcOpticalFlowPyrLK(host.levels, hPrevPts.data(),
                                                        hNextPts.data(), hStatus.data(),
                                                        nullptr, hostCount, lkp);
                        }
                        hostTracked += hostCount;
                    }
                    const auto h1 = std::chrono::steady_clock::now();
                    hostSamples.push_back(
                        std::chrono::duration<double, std::milli>(h1 - h0).count() /
                        static_cast<double>(nFrames - 1));
                }

                const Timing dt = summarize(devSamples);
                const Timing ht = summarize(hostSamples);
                const bool disjoint = dt.maxMs < ht.minMs || ht.maxMs < dt.minMs;
                std::printf("\n cadence %zu -- detection every %zu frames, %zu detections\n",
                            cadence, cadence, dets);
                std::printf("   %-46s %9.3f ms/frame  range %8.3f-%8.3f\n",
                            "HOST binCV tracker (x86, wall clock)", ht.medianMs, ht.minMs,
                            ht.maxMs);
                std::printf("   %-46s %9.3f ms/frame  range %8.3f-%8.3f\n",
                            "DEVICE binCV tracker (wall clock, 1 sync/frame)", dt.medianMs,
                            dt.minMs, dt.maxMs);
                std::printf("   ratio device/host: %6.4fx  (device %5.2fx %s)"
                            "   ranges %s\n",
                            dt.medianMs > 0.0 && ht.medianMs > 0.0
                                ? dt.medianMs / ht.medianMs : 0.0,
                            dt.medianMs > 0.0 ? ht.medianMs / dt.medianMs : 0.0,
                            dt.medianMs < ht.medianMs ? "FASTER" : "SLOWER",
                            disjoint ? "DISJOINT" : "OVERLAP");
                std::printf("   keypoints tracked: device %u/frame, host %zu/frame%s\n",
                            devCount, hostCount,
                            static_cast<size_t>(devCount) == hostCount
                                ? "  (the two detectors agree, as bit-exactness requires)"
                                : "  <-- THE TWO ARMS ARE NOT TRACKING THE SAME COUNT;"
                                  " the ratio above is NOT like-for-like");
                std::printf("SEQ,%zu,%zu,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%u,%zu\n",
                            cadence, nFrames, passes, ht.minMs, ht.medianMs, ht.maxMs,
                            dt.minMs, dt.medianMs, dt.maxMs, disjoint ? 1 : 0, devCount,
                            hostCount);
                (void)devTracked;
                (void)hostTracked;
            };

            runCadence(redetect);
            runCadence(1);

            // ---- MEMORY, reported WITH the speed, never after it -----------
            std::printf("\n MEMORY, and it is half the verdict.\n");
            std::printf("   [meter 1, allocation sum, binCV to binCV] device tracker whole\n"
                        "   resident state, including the detector's ranking pool and the\n"
                        "   sensor stage's two wide frames: %8.1f KB\n",
                        static_cast<double>(dev.bytes()) / 1024.0);
            std::printf("   [meter 1] ...of which the TRACKER alone (two ladders, the\n"
                        "   previous frame's derivatives, the keypoint arrays): %8.1f KB\n",
                        static_cast<double>(dev.trackerOnlyBytes()) / 1024.0);
            std::printf("   [meter 3, HOST bytes -- a DIFFERENT METER, printed beside the\n"
                        "   device figure and NEVER divided into it] host tracker's own\n"
                        "   working set: %8.1f KB\n",
                        static_cast<double>(host.bytes()) / 1024.0);

            const int sReps = 32;
            std::vector<std::unique_ptr<DeviceTracker>> reps;
            const size_t seqBin = meterScope(
                [&](int) { reps.push_back(std::make_unique<DeviceTracker>()); }, sReps);
            reps.clear();
            cudaDeviceSynchronize();
            std::printf("   [meter 2, cudaMemGetInfo delta over %d independent tracker\n"
                        "   states] %8.2f MB total = %8.1f KB per tracker (%.0f of the\n"
                        "   driver's own %.2f MB units -- %s)\n",
                        sReps, static_cast<double>(seqBin) / (1024.0 * 1024.0),
                        static_cast<double>(seqBin) / sReps / 1024.0,
                        static_cast<double>(seqBin) / static_cast<double>(step),
                        static_cast<double>(step) / (1024.0 * 1024.0),
                        static_cast<double>(seqBin) / static_cast<double>(step) >= 8.0
                            ? "resolves"
                            : "DOES NOT RESOLVE -- do not quote this one");
            std::printf("MEMSEQ,%zu,%zu,%zu,%zu,%d,%zu\n", dev.bytes(),
                        dev.trackerOnlyBytes(), host.bytes(), seqBin, sReps, step);
            std::printf("\n   PEAK is what is printed: every allocation above is made once\n"
                        "   at construction and held for the whole sequence -- no kernel\n"
                        "   here allocates, and the per-frame loop calls no cudaMalloc. So\n"
                        "   the resident state IS the peak, and that is a property of the\n"
                        "   design rather than a reading that happened to come out flat.\n");
        }
        }
    }

    // ======================================================================
    // 17. OUTSTANDING -- every round-2 op with no cv::cuda bar at any level
    // ======================================================================
    if (want("outstanding")) {
        std::printf("\n=====================================================================\n"
                    " 17. OUTSTANDING (ruling R2) -- round 2's ops with NO cv::cuda\n"
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
        std::printf("\n ROUND 3'S ADDITIONS TO THIS LIST.\n"
                    "\n"
                    " stereoDescriptorMatch / stereoRefineDisparity /\n"
                    " stereoMatchRectified\n"
                    "     Sparse rectified stereo by descriptor match and a one-bit\n"
                    "     window refinement. cv::cuda ships dense block matchers\n"
                    "     (StereoBM, StereoBeliefPropagation, StereoConstantSpaceBP)\n"
                    "     and no SPARSE stereo at any API level -- a dense map for\n"
                    "     500 keypoints is a different operation, not a slower\n"
                    "     spelling of this one. Priced instead against binCV's OWN\n"
                    "     device dense path, which is a binCV-to-binCV bar and lives\n"
                    "     in cuda_sparse_benchmark. SPEED VERDICT: OUTSTANDING.\n"
                    "\n"
                    " calcOpticalFlowBlockMatch\n"
                    "     Pyramidal tracking by integer Hamming block matching. The\n"
                    "     nearest cv::cuda call is SparsePyrLKOpticalFlow, which solves\n"
                    "     a different equation -- and it is already section 14's bar\n"
                    "     for the op that DOES solve the same one. Quoting it twice\n"
                    "     would make one denominator answer two questions. SPEED\n"
                    "     VERDICT: OUTSTANDING; the role row that exists is the\n"
                    "     sparse family's own, stated as a role comparison there.\n"
                    "\n"
                    " matchDescriptorsGated\n"
                    "     Section 13 prices the UNGATED matcher, which is the one with\n"
                    "     a counterpart. The gate changes the ADMITTED SET, and\n"
                    "     cv::cuda::DescriptorMatcher has no mask that reproduces it,\n"
                    "     so the gated form has no bar. SPEED VERDICT: OUTSTANDING.\n"
                    "\n"
                    " the RANSAC geometry stage\n"
                    "     NOT an outstanding row and must not be filed as one. OpenCV\n"
                    "     has no cv::cuda essential-matrix estimator, but it DOES ship\n"
                    "     a GPU RANSAC (cv::cuda::solvePnPRansac, cudalegacy) whose\n"
                    "     shape is the finding: it keeps the solver on the host and\n"
                    "     moves only scoring. binCV's own measurement came out the\n"
                    "     same way and the stage stays on the host. There is nothing\n"
                    "     here to ship and therefore nothing to leave outstanding.\n");
        std::printf("OUTSTANDING,covariance\nOUTSTANDING,cornerSubPixAsync\n"
                    "OUTSTANDING,gftt_device_spacing\nOUTSTANDING,keypointsFromCorners\n"
                    "OUTSTANDING,keypointOrientation\n"
                    "OUTSTANDING,stereoDescriptorMatch\nOUTSTANDING,stereoRefineDisparity\n"
                    "OUTSTANDING,stereoMatchRectified\n"
                    "OUTSTANDING,calcOpticalFlowBlockMatch\n"
                    "OUTSTANDING,matchDescriptorsGated\n");
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
