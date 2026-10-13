/// @file bincv_m7_track.cpp
/// @brief The assembled feature tracker on the M7: benchmark/frontend_pipeline.hpp,
/// the source the host reference runs, fed a real sequence row by row.
///
///   TRACK <frames> <width> <height>
///   SET lkwc <0|1>     Lucas–Kanade's window-count arm off or on (on by default on
///                      this core): one image times both, so an A/B never compares
///                      two builds
///   SET rt <0|1>       goodFeaturesToTrack's square-root table off or on (on by
///                      default on this core), for the same reason
///   SET nc <0|1>       Lucas–Kanade's conversions through 32 bits off or on (on by
///                      default on this core), for the same reason
///   SET gs <0|1>       the corner sweep's skip of groups whose response is zero off
///                      or on (on by default on this core), for the same reason
///   SET topup <0|1>    bincv_m7_track_grid only: after every frame, also time a top-up
///                      detection over the cells the live tracks hold, with
///                      `skipFullCells` off and on, and count the frames where the two
///                      return different corners (off by default; the tracker is
///                      unchanged either way)
///
/// For each frame the board asks for rows -- `NEED <frame> <row> <count>` -- and the
/// host answers with exactly those bytes and their CRC-32, so nothing arrives while
/// the core computes and no transfer is inside a timed region. Rows go into the
/// sensor stream as they arrive, the way a camera's lines would. After each frame it
/// prints what the host reference prints (frontend_sequence --out): an `F` line with
/// a digest of the frame's integer outputs and a `P` line with Lucas–Kanade's points
/// as exact float bits, plus a `T` line of that frame's cycles per stage. At the end,
/// totals per stage, the tracking statistics and the memory the pipeline held.
///
/// Placement: pyramids, derivatives and the response ring in the AXI SRAM (436,704
/// bytes at 752x480); the corner pool, tracks, sensor rows and the receive buffer in
/// the D2 SRAM. Nothing here allocates.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "board.h"
#include "frontend_pipeline.hpp"
#include "m7_harness.hpp"

namespace {

using W = uint32_t;

constexpr size_t kMaxW = 752, kMaxH = 480;
/// Detection by gridCorners over frontend::gridParams instead of
/// goodFeaturesToTrackStreaming: the image bincv_m7_track_grid is this source built with
/// BINCV_M7_GRID (and BINCV_M7_DERIVATIVES_ON_DEMAND).
#if defined(BINCV_M7_GRID)
constexpr bool kGrid = true;
#else
constexpr bool kGrid = false;
#endif
/// Candidate pool entries, 8 bytes each. The pool's size is a cost in response
/// sweeps, never a difference in the corners (goodFeaturesToTrackStreaming's
/// bounded-pool form); on the reference sequence 8,192 sweeps once for most
/// detections and at most three times. A grid detector needs one slot a cell instead.
constexpr size_t kPoolCapacity =
    kGrid ? bincv::gridCornerSlots(752, 480, frontend::gridParams(bincv::GoodFeaturesParams{}))
          : 8192;
constexpr size_t kMaxTracks = 200;
/// Rows per request: large enough that the round trip is amortized, small enough that
/// the buffer fits what the D2 SRAM has left.
constexpr size_t kChunkRows = 40;
/// Lifetimes longer than this are counted in the last bin; the sequence is 1,709
/// frames, so nothing reaches it.
constexpr size_t kMaxAge = 2048;

/// Derivatives computed where they are read instead of stored: the image
/// bincv_m7_track_ondemand is this source built with BINCV_M7_DERIVATIVES_ON_DEMAND.
#if defined(BINCV_M7_DERIVATIVES_ON_DEMAND)
constexpr bool kOnDemand = true;
constexpr const char* kImageName = kGrid ? "track-grid" : "track-ondemand";
#else
constexpr bool kOnDemand = false;
constexpr const char* kImageName = "track";
#endif

constexpr frontend::Layout<W> kMax(kMaxW, kMaxH, kPoolCapacity, kMaxTracks, kOnDemand);
/// A buffer the build does not use still needs a size the language allows.
constexpr size_t atLeastOne(size_t n) { return n > 0 ? n : 1; }

// AXI SRAM: the buffers the published figure counts.
__attribute__((section(".frames"), aligned(32))) W g_pyrA[kMax.pyramidWords()];
__attribute__((section(".frames"), aligned(32))) W g_pyrB[kMax.pyramidWords()];
__attribute__((section(".frames"), aligned(32))) W g_deriv[atLeastOne(kMax.derivativeWords())];
__attribute__((section(".frames"), aligned(32))) float g_ring[kMax.ringFloats()];

// D2 SRAM: what the published figure leaves out, and the harness's own.
__attribute__((section(".sram2"), aligned(32))) bincv::CornerCandidate g_pool[kPoolCapacity];
__attribute__((section(".sram2"), aligned(32))) bincv::Corner g_corners[kMaxTracks];
__attribute__((section(".sram2"), aligned(32))) uint8_t g_sensor[kMax.sensorScratchBytes()];
__attribute__((section(".sram2"), aligned(32))) bincv::Point2f g_points[kMaxTracks];
__attribute__((section(".sram2"), aligned(32))) bincv::Point2f g_tracked[kMaxTracks];
__attribute__((section(".sram2"), aligned(32))) uint8_t g_status[kMaxTracks];
__attribute__((section(".sram2"), aligned(32))) int g_age[kMaxTracks];
__attribute__((section(".sram2"), aligned(32))) uint8_t g_rx[kChunkRows * kMaxW];
__attribute__((section(".sram2"), aligned(32))) uint16_t g_lifetimes[kMaxAge];
__attribute__((section(".sram2"), aligned(32))) W g_gradient[atLeastOne(kMax.gradientWords())];
__attribute__((section(".sram2"), aligned(32))) uint8_t g_existing[kGrid ? kPoolCapacity : 1];

bool g_topUp = false;
bool& topUpEnabled() { return g_topUp; }

struct CycleClock {
    uint64_t operator()() const { return boardCycles64(); }
};

void recordLifetime(void*, int frames) {
    const size_t i = frames < 0 ? 0 : static_cast<size_t>(frames);
    uint16_t& bin = g_lifetimes[i < kMaxAge ? i : kMaxAge - 1];
    if (bin != 0xFFFFu) ++bin;
}

void putHexFloat(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    boardPutHex32(u);
}

/// The host reference's `P` line: x, y as IEEE bits, then the status, per point.
void putPoints(size_t frame, const bincv::Point2f* pts, const uint8_t* status, size_t n) {
    boardPuts("P ");
    boardPutU32(static_cast<uint32_t>(frame));
    for (size_t i = 0; i < n; ++i) {
        boardPutc(' ');
        putHexFloat(pts[i].x);
        putHexFloat(pts[i].y);
        boardPutc(status[i] ? '1' : '0');
    }
    boardPutc('\n');
}

void putStage(const char* name, uint64_t total, size_t frames) {
    boardPuts("  ");
    boardPuts(name);
    boardPuts("  ");
    boardPutU64(frames ? total / frames : 0);
    boardPuts(" cyc\n");
}

void putBytes(const char* name, size_t bytes) {
    boardPuts(name);
    boardPutU32(static_cast<uint32_t>(bytes));
    boardPutc('\n');
}

/// Receive one chunk of rows into g_rx, with up to three attempts on a CRC mismatch.
bool receiveRows(size_t frame, size_t row, size_t count, size_t width) {
    const size_t bytes = count * width;
    for (int attempt = 0; attempt < 3; ++attempt) {
        boardPuts("NEED ");
        boardPutU32(static_cast<uint32_t>(frame));
        boardPutc(' ');
        boardPutU32(static_cast<uint32_t>(row));
        boardPutc(' ');
        boardPutU32(static_cast<uint32_t>(count));
        boardPutc('\n');
        uint32_t crc = 0;
        bool overrun = false;
        for (size_t i = 0; i < bytes; ++i) {
            const int c = boardGetc();
            if (c < 0) {
                overrun = true;
                break;
            }
            g_rx[i] = static_cast<uint8_t>(c);
            crc = m7::crc32Update(crc, &g_rx[i], 1);
        }
        uint32_t sent = 0;
        if (!overrun && boardRead(&sent, sizeof sent) == 0 && sent == crc) return true;
    }
    return false;
}

void cmdTrack(const char* args) {
    uint32_t frames = 0, w = 0, h = 0;
    if (!m7::parseU32(args, frames) || !m7::parseU32(args, w) || !m7::parseU32(args, h) ||
        w == 0 || h == 0 || w > kMaxW || h > kMaxH || frames < 2) {
        boardPuts("ERR TRACK <frames >= 2> <width <= 752> <height <= 480>\n");
        return;
    }
    const frontend::Layout<W> L(w, h, kPoolCapacity, kMaxTracks, kOnDemand);
    frontend::Memory<W> mem;
    mem.sensor = g_sensor;
    mem.pyramidA = g_pyrA;
    mem.pyramidB = g_pyrB;
    mem.derivatives = g_deriv;
    mem.gradientRows = g_gradient;
    mem.ring = g_ring;
    mem.pool = g_pool;
    mem.corners = g_corners;
    mem.points = g_points;
    mem.tracked = g_tracked;
    mem.status = g_status;
    mem.age = g_age;
    std::memset(g_lifetimes, 0, sizeof g_lifetimes);

    frontend::Tracker<W, CycleClock, kOnDemand, kGrid> tr(L, mem, CycleClock{},
                                                          &recordLifetime, nullptr);
    tr.clear();
    boardStackPaint();

    const bincv::GoodFeaturesParams gftt;
    const bincv::LKParams lk;
    size_t truncated = 0, multiPass = 0, maxPasses = 0;
    uint64_t topUpCycles[2] = {0, 0};
    size_t topUpCorners[2] = {0, 0}, topUpFrames = 0, topUpDiffer = 0;
    for (size_t f = 0; f < frames; ++f) {
        const frontend::StageTimes before = tr.times();
        tr.beginFrame();
        for (size_t row = 0; row < h; row += kChunkRows) {
            const size_t n = (h - row < kChunkRows) ? h - row : kChunkRows;
            if (!receiveRows(f, row, n, w)) {
                boardPuts("ERR rows failed their CRC three times\n");
                return;
            }
            tr.pushRows(g_rx, n, w);
        }
        if (f == 0) {
            tr.seed();
            continue;
        }
        const frontend::FrameResult r = tr.step(gftt, lk);
        if (r.truncated) ++truncated;
        if (r.detected && r.passes > 1) ++multiPass;
        if (r.detected && r.passes > maxPasses) maxPasses = r.passes;

        boardPuts("F ");
        boardPutU32(static_cast<uint32_t>(f));
        boardPuts(" det=");
        boardPutc(r.detected ? '1' : '0');
        boardPuts(" corners=");
        boardPutU32(static_cast<uint32_t>(r.corners));
        boardPuts(" trunc=");
        boardPutc(r.truncated ? '1' : '0');
        boardPuts(" tracked=");
        boardPutU32(static_cast<uint32_t>(r.tracked));
        boardPuts(" alive=");
        boardPutU32(static_cast<uint32_t>(r.alive));
        boardPuts(" digest=");
        boardPutHex32(r.digest);
        boardPutc('\n');
        putPoints(f, tr.trackedPoints(), tr.trackedStatus(), r.tracked);

        if constexpr (kGrid && kOnDemand) {
            if (g_topUp) {
                const auto off = tr.topUp(gftt, false, g_existing);
                const auto on = tr.topUp(gftt, true, g_existing);
                topUpCycles[0] += off.cycles;
                topUpCycles[1] += on.cycles;
                topUpCorners[0] += off.corners;
                topUpCorners[1] += on.corners;
                ++topUpFrames;
                if (off.digest != on.digest || off.corners != on.corners) ++topUpDiffer;
            }
        }

        const frontend::StageTimes& t = tr.times();
        boardPuts("T ");
        boardPutU32(static_cast<uint32_t>(f));
        for (uint64_t v : {t.sensor - before.sensor, t.pyrDown - before.pyrDown,
                           t.derivatives - before.derivatives, t.detect - before.detect,
                           t.track - before.track}) {
            boardPutc(' ');
            boardPutU64(v);
        }
        boardPutc('\n');
    }
    tr.finish();

    const frontend::StageTimes& t = tr.times();
    const size_t n = tr.frames();
    boardPuts("-- per frame, mean cycles over ");
    boardPutU32(static_cast<uint32_t>(n));
    boardPuts(" frame pairs --\n");
    putStage("sensor stage", t.sensor, n);
    putStage("pyrDown", t.pyrDown, n);
    putStage("derivatives", t.derivatives, n);
    putStage("detect", t.detect, n);
    putStage("track", t.track, n);
    putStage("total", t.total(), n);

    size_t observed = 0;
    for (size_t i = 0; i < kMaxAge; ++i) observed += g_lifetimes[i];
    size_t median = 0, seen = 0;
    for (size_t i = 0; i < kMaxAge; ++i) {
        seen += g_lifetimes[i];
        if (seen > observed / 2) {
            median = i;
            break;
        }
    }
    if (topUpFrames > 0) {
        boardPuts("-- top-up detection over the live tracks' cells, mean over ");
        boardPutU32(static_cast<uint32_t>(topUpFrames));
        boardPuts(" frames --\n");
        putStage("skipFullCells off", topUpCycles[0], topUpFrames);
        putStage("skipFullCells on", topUpCycles[1], topUpFrames);
        putBytes("corners, off             : ", topUpCorners[0]);
        putBytes("corners, on              : ", topUpCorners[1]);
        putBytes("frames that differ       : ", topUpDiffer);
    }
    boardPuts("-- tracking --\ndetections               : ");
    boardPutU32(static_cast<uint32_t>(tr.detections()));
    boardPuts(" (");
    boardPutU32(static_cast<uint32_t>(truncated));
    boardPuts(" truncated, ");
    boardPutU32(static_cast<uint32_t>(multiPass));
    boardPuts(" took more than one sweep, at most ");
    boardPutU32(static_cast<uint32_t>(maxPasses));
    boardPuts(")\ntracks observed          : ");
    boardPutU32(static_cast<uint32_t>(observed));
    boardPuts("\nmedian track lifetime    : ");
    boardPutU32(static_cast<uint32_t>(median));
    boardPuts(" frames\nper-frame survival       : ");
    m7::putRatio(static_cast<uint32_t>(tr.survived()), static_cast<uint32_t>(tr.tried()));
    boardPuts(" of tracks tried\n");

    boardPuts("-- working set, bytes --\n");
    putBytes("pyramids, two            : ", L.pyramidBytes());
    putBytes("derivative ladders       : ", L.derivativeBytes());
    putBytes("response ring            : ", L.ringBytes());
    putBytes("  = the published figure : ", L.publishedBytes());
    if (L.derivativesOnDemand) putBytes("gradient rows (on demand): ", L.gradientBytes());
    putBytes("sensor stream rows       : ", L.sensorScratchBytes());
    putBytes("corner pool + kept       : ", L.cornerBytes());
    putBytes("tracks                   : ", L.trackBytes());
    putBytes("  = everything held      : ", L.bytes());
    putBytes("receive buffer (harness) : ", sizeof g_rx);
    putBytes("stack used               : ", boardStackUsed());
#ifdef BINCV_LK_STAGE_TIMING
    // Diagnostic builds only (bincv_m7_lkstages): where `track`'s cycles go.
    const bincv::impl::StageTiming& st = bincv::impl::stageTiming();
    boardPuts("-- track by stage, cycles over the run --\n");
    putStage("setup", st.setup, 1);
    putStage("staging", st.staging, 1);
    putStage("covariance", st.covariance, 1);
    putStage("iterations", st.residual, 1);
    putBytes("points x levels          : ", static_cast<size_t>(st.points));
    putBytes("residual calls           : ", static_cast<size_t>(st.iterations));
    putBytes("tap rows extracted       : ", static_cast<size_t>(st.tapRows));
#endif
    boardPuts("OK\n");
}

struct Setting {
    const char* name;
    bool& (*arm)();
};

const Setting kSettings[] = {
    {"lkwc", &bincv::impl::lkWindowCountEnabled},
    {"rt", &bincv::impl::cornerRootTableEnabled},
    {"nc", &bincv::impl::lkNarrowConvertEnabled},
#if BINCV_CORNER_GROUP_SKIP
    {"gs", &bincv::impl::cornerGroupSkipEnabled},
#endif
    {"topup", &topUpEnabled},
};

void cmdSet(const char* args) {
    for (const Setting& setting : kSettings) {
        const size_t n = std::strlen(setting.name);
        if (std::strncmp(args, setting.name, n) != 0 || (args[n] != ' ' && args[n] != '\0'))
            continue;
        uint32_t on = 0;
        const char* v = args + n;
        if (!m7::parseU32(v, on) || on > 1) {
            boardPuts("ERR SET ");
            boardPuts(setting.name);
            boardPuts(" <0|1>\n");
            return;
        }
        setting.arm() = on != 0;
        boardPuts("SET ");
        boardPuts(setting.name);
        boardPutc('=');
        boardPutc(on ? '1' : '0');
        boardPuts("\nOK\n");
        return;
    }
    boardPuts("ERR unknown setting\n");
}

const m7::Command kCommands[] = {{"TRACK", &cmdTrack}, {"SET", &cmdSet}};

}  // namespace

int main() { m7::serve(kImageName, kCommands, sizeof(kCommands) / sizeof(kCommands[0])); }
