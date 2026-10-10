// grid_corners_benchmark.cpp -- gridCorners beside goodFeaturesToTrackStreaming, on real
// frames.
//
// WHAT IS COMPARED, WRITTEN BEFORE MEASURING
//
// Two detectors a frontend chooses between, not two spellings of one: they return
// different corners, so there is no ratio of like for like here and none is printed.
// What a caller weighs is printed side by side, per detection on the same frames: the
// time, the memory each holds, and how many corners each returns. The frames are the
// reference sequence's, through the reference sensor stage -- `medianWide`, then
// `edgeThreshold` at 17 -- because a detector's cost is the image's, and a synthetic
// frame would price a different image.
//
//   ./grid_corners_benchmark <blob.bsq> [frames sampled, default 16]
//
// The blob is the 8-bit BSQ1 file scripts/make_sequence_blob.py writes; frames are
// sampled evenly across it.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/io/sequence.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/edge.hpp"
#include "bincv/ops/gridCorners.hpp"
#include "bincv/ops/medianWide.hpp"
#include "bincv/quantMat.hpp"
#include "measure_util.hpp"

namespace {

using W = uint32_t;
constexpr uint8_t kEdgeThreshold = 17;  // the reference pipeline's
constexpr size_t kPool = 8192;          // the board tracker's candidate pool

struct Frame {
    bincv::TernaryMat<W> dx, dy;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: grid_corners_benchmark <blob.bsq> [frames sampled]\n");
        return 2;
    }
    const size_t samples = argc > 2 ? static_cast<size_t>(std::atol(argv[2])) : 16;
    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) {
        std::printf("cannot open %s\n", argv[1]);
        return 2;
    }
    uint8_t head[bincv::kSequenceHeaderBytes];
    if (std::fread(head, 1, sizeof head, in) != sizeof head) return 2;
    const bincv::SequenceHeader sh = bincv::readSequenceHeader(head, sizeof head);
    if (!sh.valid || sh.mode != bincv::kSequenceMode8Bit || samples == 0) {
        std::printf("not an 8-bit BSQ1 blob\n");
        return 2;
    }
    const size_t w = sh.width, h = sh.height;

    std::vector<Frame> frames;
    std::vector<uint8_t> gray(w * h), median(w * h);
    for (size_t k = 0; k < samples; ++k) {
        const size_t f = k * sh.frameCount / samples;
        std::fseek(in, static_cast<long>(bincv::kSequenceHeaderBytes + f * w * h), SEEK_SET);
        if (std::fread(gray.data(), 1, gray.size(), in) != gray.size()) return 2;
        bincv::medianWide<3, uint8_t>(gray.data(), w, h, w, median.data(), w,
                                      bincv::kMedianReferenceL);
        bincv::BinMat<W> bits(static_cast<int>(w), static_cast<int>(h));
        bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                             bincv::EdgeSpatial::Wide, uint8_t, W>(median.data(), w, h, w,
                                                                   bits.view(), kEdgeThreshold);
        Frame fr{bincv::TernaryMat<W>(static_cast<int>(w), static_cast<int>(h)),
                 bincv::TernaryMat<W>(static_cast<int>(w), static_cast<int>(h))};
        bincv::derivativeX(bits, fr.dx);
        bincv::derivativeY(bits, fr.dy);
        frames.push_back(std::move(fr));
    }
    std::fclose(in);

    const bincv::GoodFeaturesParams gftt;   // the reference pipeline's four values
    bincv::GridCornerParams coarse;         // the reference's detection, gridded
    coarse.cellWidth = coarse.cellHeight = 34;
    coarse.maxCorners = gftt.maxCorners;
    bincv::GridCornerParams fine = coarse;  // a denser grid: 16-pixel cells, two a cell
    fine.cellWidth = fine.cellHeight = 16;
    fine.perCell = 2;
    fine.maxCorners = 0;

    std::vector<float> ring(w * bincv::kResponseRingRows);
    const bincv::ResponseMap rm(ring.data(), w, bincv::kResponseRingRows, w);
    std::vector<bincv::CornerCandidate> pool(kPool);
    std::vector<bincv::CornerCandidate> slotsCoarse(bincv::gridCornerSlots(w, h, coarse));
    std::vector<bincv::CornerCandidate> slotsFine(bincv::gridCornerSlots(w, h, fine));
    std::vector<bincv::Corner> corners(slotsFine.size() > 4096 ? slotsFine.size() : 4096);

    // How many corners each returns, and how many sweeps the pool form takes, before
    // anything is timed.
    size_t nG = 0, nC = 0, nF = 0, sweeps = 0;
    for (const Frame& fr : frames) {
        const bincv::CornerResult g = bincv::goodFeaturesToTrackStreaming<W>(
            fr.dx.constMagnitude(0), fr.dy.constMagnitude(0), fr.dx.constSign(),
            fr.dy.constSign(), gftt, rm, pool.data(), pool.size(), corners.data(),
            corners.size());
        nG += g.count;
        sweeps += g.passes;
        nC += bincv::gridCorners<W>(fr.dx, fr.dy, coarse, rm, slotsCoarse.data(),
                                    slotsCoarse.size(), corners.data(), corners.size())
                  .count;
        nF += bincv::gridCorners<W>(fr.dx, fr.dy, fine, rm, slotsFine.data(), slotsFine.size(),
                                    corners.data(), corners.size())
                  .count;
    }

    // One timed call detects on EVERY sampled frame, so each batch holds the same mix
    // of frames -- the pool form's cost is the frame's, one to three sweeps -- and the
    // figure is per frame after dividing.
    std::vector<measure::Bench> benches;
    benches.push_back({"goodFeaturesToTrackStreaming", [&](int) {
                           for (const Frame& fr : frames) {
                               measure::g_sink += bincv::goodFeaturesToTrackStreaming<W>(
                                                      fr.dx.constMagnitude(0),
                                                      fr.dy.constMagnitude(0), fr.dx.constSign(),
                                                      fr.dy.constSign(), gftt, rm, pool.data(),
                                                      pool.size(), corners.data(), corners.size())
                                                      .count;
                           }
                       }});
    benches.push_back({"gridCorners, 34-px cells, 1", [&](int) {
                           for (const Frame& fr : frames) {
                               measure::g_sink +=
                                   bincv::gridCorners<W>(fr.dx, fr.dy, coarse, rm,
                                                         slotsCoarse.data(), slotsCoarse.size(),
                                                         corners.data(), corners.size())
                                       .count;
                           }
                       }});
    benches.push_back({"gridCorners, 16-px cells, 2", [&](int) {
                           for (const Frame& fr : frames) {
                               measure::g_sink +=
                                   bincv::gridCorners<W>(fr.dx, fr.dy, fine, rm, slotsFine.data(),
                                                         slotsFine.size(), corners.data(),
                                                         corners.size())
                                       .count;
                           }
                       }});
    const std::vector<measure::Timing> t = measure::measureInterleaved(benches, 7, 150.0);

    const double n = static_cast<double>(frames.size());
    std::printf("Corner detection on %zu frames of %zux%zu, sampled evenly from %s\n",
                frames.size(), w, h, argv[1]);
    std::printf("the reference sensor stage; blockSize %d, qualityLevel %.2f\n\n",
                gftt.blockSize, gftt.qualityLevel);
    std::printf(" %-30s %12s %8s %10s %14s\n", "detector", "us/frame", "spread", "corners",
                "holds, bytes");
    const size_t ringBytes = ring.size() * sizeof(float);
    const size_t holds[3] = {ringBytes + pool.size() * sizeof(bincv::CornerCandidate),
                             ringBytes + slotsCoarse.size() * sizeof(bincv::CornerCandidate),
                             ringBytes + slotsFine.size() * sizeof(bincv::CornerCandidate)};
    const double counts[3] = {static_cast<double>(nG) / n, static_cast<double>(nC) / n,
                              static_cast<double>(nF) / n};
    for (size_t i = 0; i < benches.size(); ++i) {
        std::printf(" %-30s %12.1f %7.1f%% %10.1f %14zu\n", benches[i].name.c_str(),
                    t[i].medianNs / 1000.0 / n, t[i].spreadPct(), counts[i], holds[i]);
    }
    std::printf("\n goodFeaturesToTrackStreaming: maxCorners %d, minDistance %.2f, pool %zu,"
                " %.2f sweeps a detection\n",
                gftt.maxCorners, gftt.minDistance, kPool, static_cast<double>(sweeps) / n);
    std::printf(" 'holds' is the response ring plus the pool or the slots; the corners"
                " written are the caller's either way.\n");
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
