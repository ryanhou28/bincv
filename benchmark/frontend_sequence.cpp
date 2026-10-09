// frontend_sequence.cpp -- the binCV feature tracker over a frame-sequence blob,
// from frontend_pipeline.hpp: the source the Cortex-M7 image runs, run here.
//
// THREE JOBS, ONE BINARY.
//
//  1. THE REFERENCE THE BOARD IS CHECKED AGAINST. With --out it writes, per frame,
//     the line the board prints -- detection, counts, a digest of the integer
//     outputs -- and Lucas–Kanade's points as exact float bits. scripts/m7_host.py
//     holds the board to the digests exactly and reports how far its points sit
//     from these.
//
//  2. THE CHECK THAT THE SHARED SOURCE IS THE PUBLISHED PIPELINE. Run on the
//     full-resolution blob it must reproduce feature_tracking_sequence.cpp's
//     binCV statistics -- detections, tracks observed, median lifetime, survival
//     -- because it is meant to be that pipeline written against views. A
//     difference there is a bug here, not a finding.
//
//  3. THE WHOLE WORKING SET, COUNTED. It prints Layout::bytes() itemized: the
//     published figure's buffers, and beside them the sensor stream's rows and the
//     corner array the published figure leaves out.
//
// Frames are pushed one row at a time, as a camera delivers them; the frame buffer
// here is this tool's, read from the blob, and is not the pipeline's.
//
// No OpenCV: frames come from a BSQ1 blob (io/sequence.hpp), which
// scripts/make_sequence_blob.py writes.
//
//   ./frontend_sequence <blob.bsq> [frames] [--pool N] [--out FILE]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "frontend_pipeline.hpp"

#include "bincv/core/simd.hpp"
#include "bincv/io/sequence.hpp"

namespace {

using W = uint32_t;

struct NanoClock {
    uint64_t operator()() const {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    }
};

void collect(void* ctx, int frames) { static_cast<std::vector<int>*>(ctx)->push_back(frames); }

/// One frame's points as the board prints them: x, y as IEEE bits in hex, then the
/// status byte, for every point LK was given.
void writePoints(std::FILE* f, size_t frame, const bincv::Point2f* pts, const uint8_t* status,
                 size_t n) {
    std::fprintf(f, "P %zu", frame);
    for (size_t i = 0; i < n; ++i) {
        uint32_t x, y;
        std::memcpy(&x, &pts[i].x, 4);
        std::memcpy(&y, &pts[i].y, 4);
        std::fprintf(f, " %08x%08x%u", x, y, static_cast<unsigned>(status[i] ? 1 : 0));
    }
    std::fputc('\n', f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: frontend_sequence <blob.bsq> [frames] [--pool N] [--out FILE]\n");
        return 2;
    }
    // The corner pool, in 8-byte entries: a cost in sweeps, never in the corners.
    size_t maxFrames = 0, poolCap = 8192;
    const char* outPath = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--pool") == 0 && i + 1 < argc) {
            poolCap = static_cast<size_t>(std::atol(argv[++i]));
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            outPath = argv[++i];
        } else {
            maxFrames = static_cast<size_t>(std::atol(argv[i]));
        }
    }

    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) {
        std::printf("cannot open %s\n", argv[1]);
        return 2;
    }
    uint8_t head[bincv::kSequenceHeaderBytes];
    if (std::fread(head, 1, sizeof head, in) != sizeof head) {
        std::printf("short read on the header\n");
        return 2;
    }
    const bincv::SequenceHeader sh = bincv::readSequenceHeader(head, sizeof head);
    if (!sh.valid || sh.mode != bincv::kSequenceMode8Bit) {
        std::printf("not an 8-bit BSQ1 blob: the sensor stage needs 8-bit frames\n");
        return 2;
    }
    size_t frames = sh.frameCount;
    if (maxFrames && maxFrames < frames) frames = maxFrames;

    const frontend::Layout<W> L(sh.width, sh.height, poolCap,
                                static_cast<size_t>(bincv::GoodFeaturesParams{}.maxCorners));
    std::vector<uint8_t> frame(L.w * L.h), sensor(L.sensorScratchBytes());
    std::vector<W> pyrA(L.pyramidWords()), pyrB(L.pyramidWords()), deriv(L.derivativeWords());
    std::vector<float> ring(L.ringFloats());
    std::vector<bincv::CornerCandidate> pool(L.poolCapacity);
    std::vector<bincv::Corner> corners(L.maxTracks);
    std::vector<bincv::Point2f> points(L.maxTracks), tracked(L.maxTracks);
    std::vector<uint8_t> status(L.maxTracks);
    std::vector<int> age(L.maxTracks);
    frontend::Memory<W> mem;
    mem.sensor = sensor.data();
    mem.pyramidA = pyrA.data();
    mem.pyramidB = pyrB.data();
    mem.derivatives = deriv.data();
    mem.ring = ring.data();
    mem.pool = pool.data();
    mem.corners = corners.data();
    mem.points = points.data();
    mem.tracked = tracked.data();
    mem.status = status.data();
    mem.age = age.data();

    std::vector<int> lifetimes;
    frontend::Tracker<W, NanoClock> tr(L, mem, NanoClock{}, &collect, &lifetimes);
    tr.clear();

    std::FILE* out = outPath ? std::fopen(outPath, "w") : nullptr;
    if (outPath && !out) {
        std::printf("cannot write %s\n", outPath);
        return 2;
    }

    std::printf("%s\n", bincv::simdStatusString());
    std::printf("LK residual kernel: %s\n", bincv::lkPathName<bincv::LKLevelN<2, W>>());
    std::printf("=== the frontend over %zu frames, %zux%zu, corner pool %zu ===\n\n", frames,
                sh.width, sh.height, L.poolCapacity);

    const bincv::GoodFeaturesParams gftt;
    const bincv::LKParams lk;
    size_t truncated = 0, multiPass = 0, maxPasses = 0;
    for (size_t f = 0; f < frames; ++f) {
        if (std::fread(frame.data(), 1, frame.size(), in) != frame.size()) {
            std::printf("short read at frame %zu\n", f);
            return 2;
        }
        tr.beginFrame();
        for (size_t y = 0; y < L.h; ++y) tr.pushRows(frame.data() + y * L.w, 1, L.w);
        if (f == 0) {
            tr.seed();
            continue;
        }
        const frontend::FrameResult r = tr.step(gftt, lk);
        if (r.truncated) ++truncated;
        if (r.detected && r.passes > 1) ++multiPass;
        if (r.detected && r.passes > maxPasses) maxPasses = r.passes;
        if (out) {
            std::fprintf(out, "F %zu det=%d corners=%zu trunc=%d tracked=%zu alive=%zu digest=%08x\n",
                         f, r.detected ? 1 : 0, r.corners, r.truncated ? 1 : 0, r.tracked, r.alive,
                         r.digest);
            writePoints(out, f, tr.trackedPoints(), tr.trackedStatus(), r.tracked);
        }
    }
    tr.finish();
    if (out) std::fclose(out);
    std::fclose(in);

    const double n = static_cast<double>(tr.frames());
    const frontend::StageTimes& t = tr.times();
    auto ms = [&](uint64_t ns) { return static_cast<double>(ns) / 1e6 / n; };
    std::printf("--- per frame, mean over %zu frame pairs ---\n", tr.frames());
    std::printf("  sensor stage      %.4f ms\n", ms(t.sensor));
    std::printf("  pyrDown           %.4f ms\n", ms(t.pyrDown));
    std::printf("  derivatives       %.4f ms\n", ms(t.derivatives));
    std::printf("  detect            %.4f ms\n", ms(t.detect));
    std::printf("  track             %.4f ms\n", ms(t.track));
    std::printf("  total             %.4f ms\n\n", ms(t.total()));

    std::vector<int> sorted = lifetimes;
    std::sort(sorted.begin(), sorted.end());
    std::printf("--- tracking ---\n");
    std::printf("detections               : %zu (%zu truncated, %zu took more than one sweep, "
                "at most %zu)\n",
                tr.detections(), truncated, multiPass, maxPasses);
    std::printf("tracks observed          : %zu\n", lifetimes.size());
    std::printf("median track lifetime    : %d frames\n",
                sorted.empty() ? 0 : sorted[sorted.size() / 2]);
    std::printf("per-frame survival       : %.1f%%\n\n",
                tr.tried() ? 100.0 * static_cast<double>(tr.survived()) /
                                 static_cast<double>(tr.tried())
                           : 0.0);

    std::printf("--- working set, bytes ---\n");
    std::printf("pyramids, two            : %zu\n", L.pyramidBytes());
    std::printf("derivative ladders       : %zu\n", L.derivativeBytes());
    std::printf("response ring            : %zu\n", L.ringBytes());
    std::printf("  = the published figure : %zu\n", L.publishedBytes());
    std::printf("sensor stream rows       : %zu\n", L.sensorScratchBytes());
    std::printf("corner pool + kept       : %zu (%zu x %zu B + %zu x %zu B)\n", L.cornerBytes(),
                L.poolCapacity, sizeof(bincv::CornerCandidate), L.maxTracks, sizeof(bincv::Corner));
    std::printf("tracks                   : %zu\n", L.trackBytes());
    std::printf("  = everything held      : %zu\n", L.bytes());
    return 0;
}
