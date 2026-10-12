/// @file frontend_pipeline.hpp
/// @brief The binCV half of the assembled feature tracker, over memory the caller
/// provides, so a microcontroller with no heap and a host run the SAME source.
///
/// This is the pipeline `feature_tracking_sequence.cpp` times on x86-64 and the Pi,
/// stage for stage and parameter for parameter -- sensor stage, a 1/2/2/2 pyramid,
/// derivatives, `goodFeaturesToTrack` when fewer than 60 tracks survive, pyramidal
/// Lucas–Kanade, then the cull -- written against views instead of containers.
/// Containers allocate; a part with a 4 KB heap cannot use them, and a second copy
/// of the pipeline written for that part would be a second thing to keep identical.
/// So the board image (targets/stm32h753/bincv_m7_track.cpp) and the host reference
/// (frontend_sequence.cpp) both compile this, and each frame's integer outputs are
/// compared between the two by digest.
///
/// It computes nothing the benchmark does not compute, with one change of SHAPE: the
/// sensor stage takes the frame row by row (ops/medianEdgeStream.hpp), as a camera
/// delivers it, instead of as a whole 8-bit frame with an 8-bit median buffer beside
/// it. The stream is bit-identical to the two whole-frame calls, so every output is
/// the benchmark's; what it changes is that those two buffers -- 721,920 bytes at
/// 752x480, more than everything downstream -- are never held.
///
/// Every byte it holds is in `Memory`, named, so the footprint is that struct's
/// arithmetic and nothing else: `Layout::bytes()` is the pipeline's whole working
/// set. The rows a camera's DMA lands are the caller's, as they would be.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bincv/ops/corner.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/gridCorners.hpp"
#include "bincv/ops/medianEdgeStream.hpp"
#include "bincv/ops/opticalFlow.hpp"
#include "bincv/ops/pyramid.hpp"
#include "bincv/ops/opticalFlowOnDemand.hpp"

namespace frontend {

/// The reference pipeline's constants, as `feature_tracking_sequence.cpp` sets them.
inline constexpr uint8_t kEdgeThreshold = 17;
inline constexpr size_t kMinTracks = 60;
inline constexpr size_t kLevels = 4;
inline constexpr size_t kLevelBits[kLevels] = {1, 2, 2, 2};

/// The grid detector's parameters for this pipeline: the reference's detection,
/// gridded -- one corner a cell, cells the reference spacing wide, the same quality
/// level, window and cap.
constexpr bincv::GridCornerParams gridParams(const bincv::GoodFeaturesParams& g) {
    bincv::GridCornerParams p;
    const int whole = static_cast<int>(g.minDistance);
    const int cell = static_cast<double>(whole) < g.minDistance ? whole + 1 : whole;
    p.cellWidth = cell > 0 ? cell : 1;
    p.cellHeight = p.cellWidth;
    p.perCell = 1;
    p.qualityLevel = g.qualityLevel;
    p.maxCorners = g.maxCorners;
    p.blockSize = g.blockSize;
    return p;
}

/// Where every buffer goes, for a frame of `w` x `h`, and how big each is. Strides
/// are whole words with no row alignment beyond that, which is a `QuantMat`'s
/// default and so the same bytes the published footprint counts.
template <typename W>
struct Layout {
    static constexpr size_t kBits = sizeof(W) * 8;

    size_t w = 0, h = 0;
    size_t lw[kLevels] = {}, lh[kLevels] = {}, stride[kLevels] = {};
    size_t poolCapacity = 0;   ///< candidate pool entries, or the grid's slots (8 bytes each)
    size_t maxTracks = 0;
    /// Derivatives computed where they are read (ops/opticalFlowOnDemand.hpp and the
    /// `GradientSource` detector) instead of stored: no derivative planes, three
    /// gradient rows of scratch for detection.
    bool derivativesOnDemand = false;

    constexpr Layout(size_t width, size_t height, size_t pool, size_t tracks,
                     bool onDemand = false)
        : w(width), h(height), poolCapacity(pool), maxTracks(tracks),
          derivativesOnDemand(onDemand) {
        size_t cw = width, ch = height;
        for (size_t l = 0; l < kLevels; ++l) {
            lw[l] = cw;
            lh[l] = ch;
            stride[l] = (cw + kBits - 1) / kBits;
            cw = bincv::pyrDownWidth(cw);
            ch = bincv::pyrDownHeight(ch);
        }
    }

    constexpr size_t planeWords(size_t l) const { return stride[l] * lh[l]; }

    /// One pyramid: kLevelBits[l] planes at each level.
    constexpr size_t pyramidWords() const {
        size_t n = 0;
        for (size_t l = 0; l < kLevels; ++l) n += kLevelBits[l] * planeWords(l);
        return n;
    }
    /// dx and dy at every level: a sign plane plus kLevelBits[l] magnitude planes each;
    /// none on demand.
    constexpr size_t derivativeWords() const {
        if (derivativesOnDemand) return 0;
        size_t n = 0;
        for (size_t l = 0; l < kLevels; ++l) n += 2 * (kLevelBits[l] + 1) * planeWords(l);
        return n;
    }
    constexpr size_t ringFloats() const { return bincv::kResponseRingRows * w; }
    /// The on-demand detector's derivative rows at level 0; none when they are stored.
    constexpr size_t gradientWords() const {
        return derivativesOnDemand ? bincv::gradientRowWords<1, W>(w) : 0;
    }
    constexpr size_t sensorScratchBytes() const {
        using Stream = bincv::MedianEdgeStream<3, uint8_t, W>;
        return Stream::scratchElements(w, bincv::kMedianReferenceL);
    }

    constexpr size_t pyramidBytes() const { return 2 * pyramidWords() * sizeof(W); }
    constexpr size_t derivativeBytes() const { return derivativeWords() * sizeof(W); }
    constexpr size_t ringBytes() const { return ringFloats() * sizeof(float); }
    constexpr size_t gradientBytes() const { return gradientWords() * sizeof(W); }
    /// The candidate pool plus the kept corners: the bounded-pool selection takes the
    /// pool's size as a cost in sweeps, never in the answer.
    constexpr size_t cornerBytes() const {
        return poolCapacity * sizeof(bincv::CornerCandidate) + maxTracks * sizeof(bincv::Corner);
    }
    constexpr size_t trackBytes() const {
        return maxTracks * (2 * sizeof(bincv::Point2f) + sizeof(uint8_t) + sizeof(int));
    }

    /// What the published figure counts: two pyramids, the derivative ladders and
    /// the response ring.
    constexpr size_t publishedBytes() const {
        return pyramidBytes() + derivativeBytes() + ringBytes();
    }
    /// Everything this pipeline holds at once, which is what a device has to have.
    constexpr size_t bytes() const {
        return sensorScratchBytes() + publishedBytes() + gradientBytes() + cornerBytes() +
               trackBytes();
    }
};

/// The caller's buffers. Sizes are Layout's; nothing here is allocated by the pipeline.
template <typename W>
struct Memory {
    uint8_t* sensor = nullptr;      ///< Layout::sensorScratchBytes(): the stream's rows
    W* pyramidA = nullptr;          ///< Layout::pyramidWords()
    W* pyramidB = nullptr;          ///< Layout::pyramidWords()
    W* derivatives = nullptr;       ///< Layout::derivativeWords()
    W* gradientRows = nullptr;      ///< Layout::gradientWords(), on demand only
    float* ring = nullptr;          ///< Layout::ringFloats()
    bincv::CornerCandidate* pool = nullptr;  ///< Layout::poolCapacity
    bincv::Corner* corners = nullptr;        ///< Layout::maxTracks: the kept corners
    bincv::Point2f* points = nullptr;    ///< Layout::maxTracks: live tracks
    bincv::Point2f* tracked = nullptr;   ///< Layout::maxTracks: LK's output
    uint8_t* status = nullptr;           ///< Layout::maxTracks
    int* age = nullptr;                  ///< Layout::maxTracks
};

/// Per-stage time, in whatever unit the caller's clock counts -- cycles on the board,
/// nanoseconds on a host. Accumulated over every frame `step` processes.
struct StageTimes {
    uint64_t sensor = 0, pyrDown = 0, derivatives = 0, detect = 0, track = 0;
    uint64_t total() const { return sensor + pyrDown + derivatives + detect + track; }
};

/// What one frame did, and a digest of its integer outputs.
struct FrameResult {
    bool detected = false;
    bool truncated = false;    ///< the corner array could not hold every NMS survivor
    size_t corners = 0;        ///< corners detected this frame, when `detected`
    size_t passes = 0;         ///< response sweeps the detection took, when `detected`
    size_t tracked = 0;        ///< points Lucas–Kanade was given
    size_t alive = 0;          ///< points that survived the cull
    uint32_t digest = 0;       ///< FNV-1a over the binary frame, pyramid, derivatives, corners
};

/// FNV-1a, 32-bit. Small enough to have one definition both sides compile.
inline uint32_t fnv1a(uint32_t h, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}
inline constexpr uint32_t kFnvSeed = 2166136261u;

/// Where a track's lifetime goes when it ends, in frames: on a cull, on being
/// discarded at re-detection, or still alive when the sequence ends (`finish`). The
/// benchmark records the same three; a host keeps a list, a board a histogram.
using LifetimeSink = void (*)(void* ctx, int frames);

/// `OnDemand` selects the derivatives' handling at compile time, so an image carries
/// one form's code; the layout must have been made for the same one. `Grid` detects
/// with `gridCorners` over `gridParams` instead of `goodFeaturesToTrackStreaming`, the
/// pool then holding the grid's slots; the lifecycle is the same.
template <typename W, typename Clock, bool OnDemand = false, bool Grid = false>
class Tracker {
public:
    Tracker(const Layout<W>& layout, const Memory<W>& mem, Clock clock,
            LifetimeSink sink = nullptr, void* sinkCtx = nullptr)
        : L_(layout), m_(mem), clock_(clock), prev_(mem.pyramidA), next_(mem.pyramidB),
          sink_(sink), sinkCtx_(sinkCtx) {
        BINCV_ASSERT(layout.derivativesOnDemand == OnDemand,
                     "frontend: the layout was made for the other derivative form");
    }

    /// Zero every buffer: padding bits must start clear, and a caller's static
    /// buffer need not be (a NOLOAD section is not).
    void clear() {
        std::memset(m_.pyramidA, 0, L_.pyramidWords() * sizeof(W));
        std::memset(m_.pyramidB, 0, L_.pyramidWords() * sizeof(W));
        // On demand there are none, and an image may pass a one-word placeholder. GCC 13
        // does not prove the guard and reports this memset as overflowing it. Silenced
        // rather than restated on the template argument: that would be the same code,
        // but GCC would inline the board images differently around it.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
        if (L_.derivativeWords() > 0) std::memset(m_.derivatives, 0, L_.derivativeWords() * sizeof(W));
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
        std::memset(m_.ring, 0, L_.ringBytes());
        prev_ = m_.pyramidA;
        next_ = m_.pyramidB;
        started_ = false;
        count_ = 0;
        times_ = StageTimes{};
        frames_ = 0;
        detections_ = 0;
        survivedSum_ = 0;
        triedSum_ = 0;
    }

    /// Starts a frame: last frame's pyramid becomes `prev` -- all of it, since pyrDown
    /// is a pure function of level 0 -- and the stream starts writing the new level 0.
    /// The first frame of a sequence swaps nothing.
    void beginFrame() {
        if (started_) {
            W* const t = prev_;
            prev_ = next_;
            next_ = t;
        }
        started_ = true;
        stream_ = Stream(m_.sensor, L_.sensorScratchBytes(), bincv::kMedianReferenceL,
                         plane(next_, 0, 0), kEdgeThreshold);
        stream_.reset();
    }

    /// The frame's next `count` rows, `stride` bytes apart, as they arrive.
    void pushRows(const uint8_t* rows, size_t count, size_t stride) {
        const uint64_t t0 = clock_();
        stream_.push(rows, count, stride);
        times_.sensor += clock_() - t0;
    }

    /// Frame 0's whole pyramid, so frame 1's swap hands it a fully built `prev`.
    /// Untimed, as in the benchmark -- and so its sensor stage is taken back out.
    void seed() {
        times_ = StageTimes{};
        StageTimes discard;
        buildPyramid(next_, discard);
    }

    /// One frame, once every row of it has been pushed. Order and contents as the
    /// benchmark's loop.
    FrameResult step(const bincv::GoodFeaturesParams& gftt, const bincv::LKParams& lk) {
        FrameResult r;
        buildPyramid(next_, times_);
        if constexpr (!OnDemand) derivatives(prev_, times_);

        uint64_t t0 = clock_();
        if (count_ < kMinTracks) {
            bincv::ResponseMap ring{m_.ring, L_.w, bincv::kResponseRingRows, L_.w};
            bincv::CornerResult cr;
            if constexpr (Grid) {
                const bincv::GridCornerParams gp = gridParams(gftt);
                if constexpr (OnDemand) {
                    bincv::GradientSource<1, W> gs;
                    gs.planes[0] = plane(prev_, 0, 0);
                    cr = bincv::gridCorners(gs, gp, ring, m_.gradientRows, L_.gradientWords(),
                                            m_.pool, L_.poolCapacity, m_.corners, L_.maxTracks);
                } else {
                    cr = bincv::gridCorners<W>(derivMag(0, 0, 0), derivMag(1, 0, 0),
                                               derivSign(0, 0), derivSign(1, 0), gp, ring,
                                               m_.pool, L_.poolCapacity, m_.corners,
                                               L_.maxTracks);
                }
            } else if constexpr (OnDemand) {
                bincv::GradientSource<1, W> gs;
                gs.planes[0] = plane(prev_, 0, 0);
                cr = bincv::goodFeaturesToTrackStreaming(gs, gftt, ring, m_.gradientRows,
                                                         L_.gradientWords(), m_.pool,
                                                         L_.poolCapacity, m_.corners,
                                                         L_.maxTracks);
            } else {
                cr = bincv::goodFeaturesToTrackStreaming<W>(
                    derivMag(0, 0, 0), derivMag(1, 0, 0), derivSign(0, 0), derivSign(1, 0),
                    gftt, ring, m_.pool, L_.poolCapacity, m_.corners, L_.maxTracks);
            }
            for (size_t i = 0; i < count_; ++i) record(m_.age[i]);  // discarded, not immortal
            count_ = cr.count < L_.maxTracks ? cr.count : L_.maxTracks;
            for (size_t i = 0; i < count_; ++i) {
                m_.points[i] = bincv::Point2f{static_cast<float>(m_.corners[i].x),
                                              static_cast<float>(m_.corners[i].y)};
                m_.age[i] = 0;
            }
            r.detected = true;
            r.truncated = cr.candidatesTruncated;
            r.corners = cr.count;
            r.passes = cr.passes;
            ++detections_;
        }
        times_.detect += clock_() - t0;

        t0 = clock_();
        if (count_ > 0) {
            if constexpr (OnDemand) {
                bincv::LKLevelsOnDemand<W, 1, 2, 2, 2> levels;
                levels.template get<0>() = levelOnDemand<1>(0);
                levels.template get<1>() = levelOnDemand<2>(1);
                levels.template get<2>() = levelOnDemand<2>(2);
                levels.template get<3>() = levelOnDemand<2>(3);
                bincv::calcOpticalFlowPyrLK(levels, m_.points, m_.tracked, m_.status, nullptr,
                                            count_, lk);
            } else {
                bincv::LKLevels<W, 1, 2, 2, 2> levels;
                levels.template get<0>() = level<1>(0);
                levels.template get<1>() = level<2>(1);
                levels.template get<2>() = level<2>(2);
                levels.template get<3>() = level<2>(3);
                bincv::calcOpticalFlowPyrLK(levels, m_.points, m_.tracked, m_.status, nullptr,
                                            count_, lk);
            }
        }
        times_.track += clock_() - t0;

        r.tracked = count_;
        r.digest = digest(r);
        cull();
        r.alive = count_;
        triedSum_ += r.tracked;
        survivedSum_ += r.alive;
        ++frames_;
        return r;
    }

    /// The tracks still alive at the end of the sequence, recorded as the benchmark
    /// records them.
    void finish() {
        for (size_t i = 0; i < count_; ++i) record(m_.age[i]);
    }

    const StageTimes& times() const { return times_; }
    size_t frames() const { return frames_; }
    size_t detections() const { return detections_; }
    size_t tried() const { return triedSum_; }
    size_t survived() const { return survivedSum_; }

    /// LK's output for the frame just stepped, before the cull: `tracked()` points.
    const bincv::Point2f* trackedPoints() const { return m_.tracked; }
    const uint8_t* trackedStatus() const { return m_.status; }

    /// What a top-up detection costs on the frame just stepped, as a frontend that keeps
    /// its tracks makes one: `gridParams`'s grid over the cells the live tracks hold, with
    /// `skipFullCells` as asked. A measurement beside the pipeline, which goes on exactly
    /// as it would have: it writes only scratch the next detection rewrites. Grid and
    /// on-demand configurations only. `existing` is the caller's, one byte a cell.
    struct TopUp {
        uint64_t cycles = 0;
        uint32_t digest = 0;
        size_t corners = 0;
    };
    TopUp topUp(const bincv::GoodFeaturesParams& gftt, bool skip, uint8_t* existing) {
        static_assert(Grid && OnDemand, "topUp measures the on-demand grid detector");
        bincv::GridCornerParams gp = gridParams(gftt);
        gp.skipFullCells = skip;
        const size_t across = bincv::gridCellsAcross(L_.w, gp);
        const size_t cells = across * bincv::gridCellsDown(L_.h, gp);
        for (size_t c = 0; c < cells; ++c) existing[c] = 0;
        for (size_t i = 0; i < count_; ++i) {
            const size_t c = static_cast<size_t>(m_.points[i].y) / static_cast<size_t>(gp.cellHeight) * across +
                             static_cast<size_t>(m_.points[i].x) / static_cast<size_t>(gp.cellWidth);
            if (existing[c] < 255) ++existing[c];
        }
        bincv::ResponseMap ring{m_.ring, L_.w, bincv::kResponseRingRows, L_.w};
        bincv::GradientSource<1, W> gs;
        gs.planes[0] = plane(next_, 0, 0);
        TopUp t;
        const uint64_t t0 = clock_();
        const bincv::CornerResult cr =
            bincv::gridCorners(gs, gp, ring, m_.gradientRows, L_.gradientWords(), m_.pool,
                               L_.poolCapacity, m_.corners, L_.maxTracks, existing);
        t.cycles = clock_() - t0;
        t.corners = cr.count;
        t.digest = fnv1a(kFnvSeed, m_.corners, cr.count * sizeof(bincv::Corner));
        return t;
    }

private:
    /// Views onto a pyramid buffer: level l's plane p.
    bincv::BinMatView<W> plane(W* pyr, size_t l, size_t p) const {
        size_t off = 0;
        for (size_t k = 0; k < l; ++k) off += kLevelBits[k] * L_.planeWords(k);
        off += p * L_.planeWords(l);
        return bincv::BinMatView<W>(pyr + off, L_.lw[l], L_.lh[l], L_.stride[l]);
    }
    /// The derivative block: for level l, axis a (0 = x, 1 = y), the sign plane then
    /// kLevelBits[l] magnitude planes.
    W* derivBase(size_t a, size_t l) const {
        size_t off = 0;
        for (size_t k = 0; k < l; ++k) off += 2 * (kLevelBits[k] + 1) * L_.planeWords(k);
        off += a * (kLevelBits[l] + 1) * L_.planeWords(l);
        return m_.derivatives + off;
    }
    bincv::BinMatView<W> derivSign(size_t a, size_t l) const {
        return bincv::BinMatView<W>(derivBase(a, l), L_.lw[l], L_.lh[l], L_.stride[l]);
    }
    bincv::BinMatView<W> derivMag(size_t a, size_t l, size_t p) const {
        return bincv::BinMatView<W>(derivBase(a, l) + (1 + p) * L_.planeWords(l), L_.lw[l],
                                    L_.lh[l], L_.stride[l]);
    }

    /// Box2x2 + Replicate down the 1/2/2/2 ladder, as `Pyramid::build<Box2x2,
    /// Replicate>` does: level 0's one plane to level 1's two, then two to two.
    void buildPyramid(W* pyr, StageTimes& t) {
        const uint64_t t0 = clock_();
        down<2, 1>(pyr, 1);
        down<2, 2>(pyr, 2);
        down<2, 2>(pyr, 3);
        t.pyrDown += clock_() - t0;
    }
    template <size_t NOut, size_t NIn>
    void down(W* pyr, size_t l) {
        bincv::BinMatConstView<W> src[NIn];
        bincv::BinMatView<W> dst[NOut];
        for (size_t p = 0; p < NIn; ++p) src[p] = plane(pyr, l - 1, p);
        for (size_t p = 0; p < NOut; ++p) dst[p] = plane(pyr, l, p);
        bincv::pyrDownFiltered<bincv::PyrDownFilter::Box2x2, NOut, NIn, W,
                               bincv::PyrDownBorder::Replicate>(src, dst);
    }

    /// derivativeX and derivativeY of every level of `pyr`, the defaults the
    /// container spelling passes.
    void derivatives(W* pyr, StageTimes& t) {
        const uint64_t t0 = clock_();
        deriv<1>(pyr, 0);
        deriv<2>(pyr, 1);
        deriv<2>(pyr, 2);
        deriv<2>(pyr, 3);
        t.derivatives += clock_() - t0;
    }
    template <size_t N>
    void deriv(W* pyr, size_t l) {
        bincv::BinMatConstView<W> src[N];
        bincv::BinMatView<W> mx[N], my[N];
        for (size_t p = 0; p < N; ++p) {
            src[p] = plane(pyr, l, p);
            mx[p] = derivMag(0, l, p);
            my[p] = derivMag(1, l, p);
        }
        bincv::derivativeX<N, W>(src, mx, derivSign(0, l));
        bincv::derivativeY<N, W>(src, my, derivSign(1, l));
    }

    template <size_t N>
    bincv::LKLevelOnDemand<N, W> levelOnDemand(size_t l) const {
        bincv::LKLevelOnDemand<N, W> lv;
        for (size_t p = 0; p < N; ++p) {
            lv.prev[p] = plane(prev_, l, p);
            lv.next[p] = plane(next_, l, p);
        }
        return lv;
    }

    template <size_t N>
    bincv::LKLevelN<N, W> level(size_t l) const {
        bincv::LKLevelN<N, W> lv;
        for (size_t p = 0; p < N; ++p) {
            lv.prev[p] = plane(prev_, l, p);
            lv.next[p] = plane(next_, l, p);
            lv.dxMag[p] = derivMag(0, l, p);
            lv.dyMag[p] = derivMag(1, l, p);
        }
        lv.dxSign = derivSign(0, l);
        lv.dySign = derivSign(1, l);
        return lv;
    }

    /// The benchmark's `compact`: keep a point when LK found it and it is inside
    /// the frame.
    void cull() {
        size_t k = 0;
        for (size_t i = 0; i < count_; ++i) {
            const bincv::Point2f& o = m_.tracked[i];
            const bool alive = m_.status[i] && o.x >= 0 && o.y >= 0 &&
                               o.x < static_cast<float>(L_.w) && o.y < static_cast<float>(L_.h);
            if (alive) {
                m_.points[k] = o;
                m_.age[k] = m_.age[i] + 1;
                ++k;
            } else {
                record(m_.age[i] + 1);
            }
        }
        count_ = k;
    }

    void record(int frames) {
        if (sink_) sink_(sinkCtx_, frames);
    }

    /// The integer outputs of this frame: the new binary frame and the rest of its
    /// pyramid, the derivatives of the previous one, and the corners if detection
    /// ran. Lucas–Kanade's float outputs are compared separately: whether the
    /// compiler contracts a multiply-add into an FMA legitimately changes their last
    /// bits between targets, and nothing integer can.
    uint32_t digest(const FrameResult& r) const {
        uint32_t h = kFnvSeed;
        h = fnv1a(h, next_, L_.pyramidWords() * sizeof(W));
        if (L_.derivativeWords() > 0) h = fnv1a(h, m_.derivatives, L_.derivativeWords() * sizeof(W));
        if (r.detected) {
            for (size_t i = 0; i < r.corners && i < L_.maxTracks; ++i) {
                const int32_t xy[2] = {m_.corners[i].x, m_.corners[i].y};
                h = fnv1a(h, xy, sizeof xy);
            }
        }
        const uint32_t n = static_cast<uint32_t>(r.tracked);
        return fnv1a(h, &n, sizeof n);
    }

    using Stream = bincv::MedianEdgeStream<3, uint8_t, W>;

    Layout<W> L_;
    Memory<W> m_;
    Clock clock_;
    W* prev_;
    W* next_;
    Stream stream_{nullptr, 0, bincv::kMedianReferenceL, bincv::BinMatView<W>(), 0};
    bool started_ = false;
    LifetimeSink sink_;
    void* sinkCtx_;
    size_t count_ = 0;
    StageTimes times_;
    size_t frames_ = 0, detections_ = 0, triedSum_ = 0, survivedSum_ = 0;
};

}  // namespace frontend
