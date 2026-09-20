// ===========================================================================
// A DEVICE-RESIDENT BINARY-FRAME VIO FRONTEND, END TO END, ON binCV'S CUDA
// KERNELS -- the GPU twin of examples/vio_frontend.cpp.
//
// A VIO *frontend* is the image-processing half of a visual-inertial odometry
// system -- sensor stage, pyramid, detection, tracking, track lifecycle -- the
// part that turns camera frames into feature tracks for the estimator behind it.
//
// WHAT "RESIDENT" MEANS HERE, BECAUSE IT IS THE ENTIRE ARGUMENT
//
// A GPU kernel that is 3x faster than its CPU twin is worth nothing to a caller
// who has to send the frame across the bus to reach it and pull the answer back
// to use it. So this example is built around one rule, and every structural
// choice below follows from it:
//
//     THE WIDE FRAME CROSSES THE BUS ONCE PER FRAME, IN ONE DIRECTION, AND
//     NOTHING FRAME-SIZED EVER COMES BACK.
//
// Per frame, 360,960 bytes go up and 11,536 bytes come down, and there is
// EXACTLY ONE host<->device synchronization -- at the end, after every launch
// of the frame is already enqueued. The sensor stage, the pyramid, the
// derivatives, detection, orientation and description all run on data that is
// already there. The program prints both numbers and the sync count, because a
// residency claim that is not counted is not a claim.
//
// THE PIPELINE, AND WHERE EACH PIECE LIVES
//
//   host: one frame of a BSQ1 blob            (the caller's wide pixel array)
//     |  uploadImage                                            360,960 B up
//     v
//   [wide 8-bit frame, device]
//     |  cuda::medianWide<3>  (the reference L neighbourhood)
//     v
//   [denoised 8-bit frame, device] ------------------------------+
//     |  cuda::edgeThreshold  (8 bits in, 1 BIT out, no wide     |
//     v                        intermediate at any point)        |
//   [binary frame = pyramid level 0, device]                     |
//     |  cuda::buildPyramidBox   (levels 1..3, 1/2/2/2 ladder)   |
//     |  cuda::derivativeXY      (both axes, one traversal)      |
//     v                                                          |
//   [ternary dx, dy]                                             |
//     |  cuda::goodFeaturesToTrackAsync  (fused arm: NO          |
//     v                                   frame-sized float map) |
//   [corner records, device]                                     |
//     |  cuda::keypointsFromCorners                              |
//     v                                                          |
//   [keypoint set (x, y), device] <------------------------------+
//     |  cuda::keypointOrientation  (intensity centroid, reads
//     |                              the denoised WIDE frame)
//     |  cuda::computeBriefSteered  (256-bit steered BRIEF, also
//     v                              on the wide frame)
//   [angles + descriptors, device]  --> 11,536 B down, once.
//
// WHY THE LAST TWO STAGES READ A WIDE FRAME, SAID PLAINLY
//
// They have to. ops/descriptor.hpp is explicit that the BRIEF test is
// `img[a] < img[b]` on the GRAYSCALE image, because a comparison between two
// one-bit pixels carries almost nothing -- so the byte frame stays resident for
// the whole pipeline and the working set is NOT one bit per pixel end to end.
// That is a real cost of this frontend and it is printed in the memory table
// rather than left for a reader to discover. The binary representation pays in
// the sensor stage, the pyramid, the derivatives and detection; it pays nothing
// in orientation and description, and this example does not pretend otherwise.
//
// WHAT THIS DOES NOT ESTABLISH
//
// * It is NOT a GPU-vs-GPU speed result. There is no `cv::cuda` whole-frontend
//   counterpart to compare against -- `cv::cuda::ORB` is a different algorithm
//   over different input -- so the end-to-end GPU speed verdict is OUTSTANDING,
//   and no substitute bar is invented here.
// * The optional host arm (BINCV_CUDA_VIO_HOST=1) is a same-machine CPU
//   measurement against a GPU one. It is CONTEXT, printed with that label.
// * It is a DETECT-AND-DESCRIBE frontend, not a TRACKING one. A device
//   Lucas-Kanade now exists (cuda/opticalFlow.hpp), but this program does not
//   call it: the stage that dominates the host frontend (62-67% of it) is
//   absent from BOTH arms here, so the two totals stay comparable to each
//   other and to nothing else. The tracking pipeline is timed end to end by
//   `cuda_role_benchmark sequence`. Comparing this program's total against
//   docs/reports/feature-tracking.md's would be comparing two different pipelines.
//
// FRAMES
//
// A BSQ1 sequence blob (scripts/make_sequence_blob.py), which is how a build
// with no OpenCV reads real dataset frames -- and this backend has no OpenCV in
// it at all. Make one from the EuRoC stream this project measures on:
//
//     scripts/make_sequence_blob.py <euroc-V1_02-cam0-dir> -o v1_02.bsq --mode 8bit
//     ./cuda_vio_frontend v1_02.bsq
//
// A mode-1 (packed) blob is REFUSED rather than silently accepted: its sensor
// stage already ran on the host, so it cannot exercise the device sensor stage
// that is a third of what this example exists to show.
//
// ENVIRONMENT
//
//   BINCV_CUDA_VIO_HOST=1      also run binCV's HOST frontend on every frame,
//                              compare the two answers exactly, and print both
//                              totals. Off by default: the device pipeline is
//                              what this example is for.
//   BINCV_CUDA_VIO_ROUNDTRIP=1 replace cuda::keypointsFromCorners with the
//                              download-convert-upload round trip it exists to
//                              remove, so the two can be timed in ONE process.
//   BINCV_CUDA_VIO_STAGES=0    record only the outer pair of CUDA events, so
//                              the frame total is measured WITHOUT the
//                              per-stage instrumentation in it.
// ===========================================================================
#include <algorithm>
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

#include "bincv/cuda/compaction.hpp"
#include "bincv/cuda/corner.hpp"
#include "bincv/cuda/derivative.hpp"
#include "bincv/cuda/descriptor.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/edge.hpp"
#include "bincv/cuda/keypoints.hpp"
#include "bincv/cuda/median.hpp"
#include "bincv/cuda/orientation.hpp"
#include "bincv/cuda/pyramid.hpp"
#include "bincv/cuda/transfer.hpp"

#include "bincv/io/sequence.hpp"
#include "bincv/ops/corner.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/edge.hpp"
#include "bincv/ops/medianWide.hpp"
#include "bincv/ops/orbPattern.hpp"
#include "bincv/ops/orientation.hpp"
#include "bincv/ops/pyramid.hpp"
#include "bincv/quantMat.hpp"

namespace {

using W = uint32_t;
namespace bc = bincv::cuda;

// ---- the frontend's parameters, all of them the reference frontend's --------
constexpr int kEdgeThreshold = 17;       ///< the reference frontend's edge_threshold
constexpr int kOrientationRadius = 15;   ///< pairs with the 31-pixel BRIEF patch
constexpr size_t kDescriptorBits = 256;  ///< cv::ORB's length, and the vendored pattern's
constexpr uint32_t kCapacity = 256;      ///< keypoint slots; >= GoodFeaturesParams::maxCorners

// THE RANKING CAPACITY, which is NOT the keypoint-slot count and is the second
// number a reader has to get right.
//
// `goodFeaturesToTrack` ranks the survivors of the quality threshold and THEN
// runs the greedy spacing filter over them, so a capacity that cannot hold
// every survivor drops the weakest ones BEFORE the spacing filter ever sees
// them -- and a dropped survivor might have been accepted once the ranked ones
// ran out. The count that comes back is then a LOWER BOUND on the reference's,
// and `candidatesTruncated` is the only way a caller learns of it. The final
// corner count is capped by `GoodFeaturesParams::maxCorners` (200) regardless,
// which is why this can be large while the keypoint arrays stay small.
constexpr uint32_t kDefaultRankCapacity = 32768;

uint32_t rankCapacity() {
    if (const char* v = std::getenv("BINCV_CUDA_VIO_RANK")) {
        const long n = std::atol(v);
        if (n > 0) return static_cast<uint32_t>(n);
    }
    return kDefaultRankCapacity;
}

// THE DEVICE CANDIDATE POOL, and it is the one number in this file a reader
// has to choose for their own content.
//
// It holds every NONZERO 3x3 LOCAL MAXIMUM of the response -- not `maxCorners`,
// and not the survivors of the quality threshold, which is applied afterwards.
// On a binary edge map the response takes only a few hundred distinct values,
// so plateaus are wide and maxima are common; a textured frame produces tens of
// thousands. The run prints the peak it actually saw, and an overflow is
// reported rather than absorbed, because the selection cannot report a ranked
// count at all once its candidate buffer overflows.
//
// This is ALSO the pipeline's largest single allocation, and that is worth
// knowing: the host's streaming selection prunes candidates against a RUNNING
// threshold as it walks rows and never holds the whole set, while the device's
// parallel append has no running state to prune against. The device arm
// therefore pays memory the host arm does not, and the memory table says so.
constexpr uint32_t kDefaultCandidateCapacity = 32768;

uint32_t candidatePoolSize() {
    if (const char* v = std::getenv("BINCV_CUDA_VIO_POOL")) {
        const long n = std::atol(v);
        if (n > 0) return static_cast<uint32_t>(n);
    }
    return kDefaultCandidateCapacity;
}

/// ms since a steady-clock mark -- the HOST clock, used only where a host clock
/// is the honest one (the CPU arm, and the wall time around a whole frame).
double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}

bool envOn(const char* name, bool fallback = false) {
    const char* v = std::getenv(name);
    if (v == nullptr) return fallback;
    return std::strcmp(v, "0") != 0;
}

/// PAGE-LOCKED host memory, and it is not an optimization detail -- it is the
/// difference between a transfer that is asynchronous and one that is not.
/// A copy out of pageable memory makes the driver stage it through an internal
/// pinned buffer, which serializes against the stream; measured here, the
/// 11.5 KB result block cost 0.386 ms from pageable memory and 0.060 ms from
/// pinned -- a 6.5x difference on a transfer small enough that it should have
/// been free. A resident pipeline whose whole point is that the host does not
/// wait cannot land its results in a `std::vector`.
template <typename T>
class Pinned {
public:
    explicit Pinned(size_t count) : count_(count) {
        if (count_ == 0) return;
        void* p = nullptr;
        BINCV_CUDA_CHECK(cudaHostAlloc(&p, count_ * sizeof(T), cudaHostAllocDefault));
        ptr_ = static_cast<T*>(p);
        // Value-initialized rather than memset: some of these element types
        // carry default member initializers, and -Wclass-memaccess is right to
        // refuse a raw fill of one.
        for (size_t i = 0; i < count_; ++i) ptr_[i] = T{};
    }
    ~Pinned() {
        if (ptr_ != nullptr) cudaFreeHost(ptr_);
    }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// ===========================================================================
// THE STAGE CLOCK
//
// CUDA events recorded ON THE PIPELINE'S OWN STREAM, between stages. They are
// enqueued, not waited on, so the frame is never stalled to read a clock; the
// elapsed times are read after the frame's single synchronize, from events that
// have long since completed.
//
// Every event still costs a small amount of stream time. Rather than argue
// about how much, BINCV_CUDA_VIO_STAGES=0 records only the outer pair, and the
// summary prints both totals so the instrumentation's own cost is visible.
// ===========================================================================
class StageClock {
public:
    StageClock(const char* const* names, size_t count, bool perStage)
        : names_(names), count_(count), perStage_(perStage), totals_(count, 0.0) {
        events_.resize(count + 1);
        for (cudaEvent_t& e : events_) BINCV_CUDA_CHECK(cudaEventCreate(&e));
    }
    ~StageClock() {
        for (cudaEvent_t e : events_) cudaEventDestroy(e);
    }
    StageClock(const StageClock&) = delete;
    StageClock& operator=(const StageClock&) = delete;

    /// Enqueue the frame's first mark.
    void begin(cudaStream_t s) { BINCV_CUDA_CHECK(cudaEventRecord(events_[0], s)); }
    /// Enqueue the mark that closes stage `i`. A no-op for the interior marks
    /// when per-stage timing is off; the last one always records.
    void mark(size_t i, cudaStream_t s) {
        if (perStage_ || i + 1 == count_) {
            BINCV_CUDA_CHECK(cudaEventRecord(events_[i + 1], s));
        }
    }
    /// Read the frame's elapsed times back. Call AFTER the frame's synchronize.
    void accumulate() {
        float ms = 0.0f;
        if (perStage_) {
            for (size_t i = 0; i < count_; ++i) {
                BINCV_CUDA_CHECK(cudaEventElapsedTime(&ms, events_[i], events_[i + 1]));
                totals_[i] += static_cast<double>(ms);
            }
        }
        BINCV_CUDA_CHECK(cudaEventElapsedTime(&ms, events_[0], events_[count_]));
        wholeTotal_ += static_cast<double>(ms);
        ++frames_;
    }

    bool perStage() const { return perStage_; }
    size_t count() const { return count_; }
    const char* name(size_t i) const { return names_[i]; }
    double meanMs(size_t i) const { return frames_ ? totals_[i] / double(frames_) : 0.0; }
    double meanWholeMs() const { return frames_ ? wholeTotal_ / double(frames_) : 0.0; }

private:
    const char* const* names_;
    size_t count_;
    bool perStage_;
    std::vector<cudaEvent_t> events_;
    std::vector<double> totals_;
    double wholeTotal_ = 0.0;
    size_t frames_ = 0;
};

// The stages, named once. The order is the order they are enqueued in.
enum Stage : size_t {
    kUpload = 0,
    kMedian,
    kEdge,
    kPyramid,
    kDerivative,
    kDetect,
    kKeypoints,
    kOrientation,
    kDescribe,
    kDownload,
    kStageCount
};
const char* const kStageNames[kStageCount] = {
    "upload wide frame        (H2D)", "sensor: medianWide<3> L       ",
    "sensor: edgeThreshold -> bits ", "pyramid: 3 x pyrDownBox       ",
    "derivatives: derivativeXY     ", "detect: goodFeaturesToTrack   ",
    "keypointsFromCorners          ", "orientation: centroid r=15    ",
    "describe: steered BRIEF-256   ", "download results         (D2H)"};

// ===========================================================================
// THE RESIDENT FRONTEND
//
// Everything the pipeline owns on the device, allocated ONCE outside the frame
// loop. This is the part of the file to read: a caller's device frontend is
// this struct and the seven calls in `runFrame`.
//
// The containers are binCV's own -- DeviceImage, DevicePyramid, DeviceBinMat,
// DeviceArray -- so allocation lives in containers and the kernels take views
// and allocate nothing, exactly as on the host.
// ===========================================================================
struct ResidentFrontend {
    // --- the two wide frames. Both are needed: medianWide cannot run in place
    // (its destination would feed its own neighbourhood), and the denoised one
    // stays resident afterwards because orientation and BRIEF read it.
    bc::DeviceImage<uint8_t> wide;
    bc::DeviceImage<uint8_t> denoised;

    // --- the binary ladder. Level 0 is written by the sensor stage; 1..3 are
    // built on device and never touched by the host.
    bc::DevicePyramid<1, 2, 2, 2> pyramid;

    // --- the ternary derivatives of level 0: 2 planes each (magnitude, sign).
    bc::DeviceBinMat dxBlock;
    bc::DeviceBinMat dyBlock;

    // --- detection. `frameMap` is deliberately NOT allocated: the fused arm
    // materialises no frame-sized float map, and leaving the field empty is
    // what makes that a checkable property of this program rather than a claim.
    bc::DeviceArray<bc::DeviceCorner> candidates;
    bc::DeviceAppendCounter candidateCounter;
    bc::DeviceArray<uint32_t> maxBits;
    bc::DeviceArray<uint8_t> selectScratch;
    bc::DeviceArray<bc::DeviceCorner> corners;

    // --- EVERYTHING THE FRAME PRODUCES, IN ONE DEVICE ALLOCATION.
    //
    // "One download per frame" is not a figure of speech, and five separate
    // `cudaMemcpyAsync` calls are not one download: each carries its own
    // latency, and measured here the five cost 0.089 ms against 0.060 ms for
    // the same 11,536 bytes as a single contiguous copy -- on a host whose
    // device-to-host latency is high enough that even the single copy is not
    // bandwidth-bound. So the result set is
    // ONE array that the kernels write into at fixed offsets -- the same thing
    // DevicePyramid does with its levels, and for the same reason.
    //
    // The offsets are ordered so every one of them is 4-byte aligned; `keep`
    // is last because it is the only byte-granular member.
    bc::DeviceArray<uint8_t> resultBlock;
    bc::DeviceArray<bincv::BriefPair> patternPairs;
    bc::DeviceBriefPattern pattern;

    static constexpr size_t kResultOffset = 0;
    static constexpr size_t kXYOffset = 16;
    static constexpr size_t kAnglesOffset = kXYOffset + 2 * kCapacity * sizeof(float);
    static constexpr size_t kDescriptorOffset = kAnglesOffset + kCapacity * sizeof(float);
    static constexpr size_t kKeepOffset =
        kDescriptorOffset + size_t{kCapacity} * (kDescriptorBits / 32) * sizeof(uint32_t);
    static constexpr size_t kResultBlockBytes = kKeepOffset + kCapacity;

    bc::DeviceCornerResult* result() {
        return reinterpret_cast<bc::DeviceCornerResult*>(resultBlock.data() + kResultOffset);
    }
    float* keypointXY() {
        return reinterpret_cast<float*>(resultBlock.data() + kXYOffset);
    }
    float* angles() { return reinterpret_cast<float*>(resultBlock.data() + kAnglesOffset); }
    uint32_t* descriptors() {
        return reinterpret_cast<uint32_t*>(resultBlock.data() + kDescriptorOffset);
    }
    uint8_t* keep() { return resultBlock.data() + kKeepOffset; }

    cudaStream_t stream = nullptr;
    size_t width = 0;
    size_t height = 0;
    uint32_t candidateCapacity = 0;
    uint32_t rankCapacity = 0;

    // --- the host-side landing area for the one download per frame, and the
    // staging buffer the frame is read into. BOTH PAGE-LOCKED: see `Pinned`.
    Pinned<uint8_t> frameStaging;
    Pinned<uint8_t> landing;                 // the device result block, mirrored
    Pinned<bc::DeviceCorner> cornerStaging;  // the ROUND-TRIP arm's only buffer

    // The five views into the landed block. Typed accessors rather than raw
    // offsets at every use, for the reason the device side has them.
    const bc::DeviceCornerResult& hostResult() const {
        return *reinterpret_cast<const bc::DeviceCornerResult*>(landing.data() + kResultOffset);
    }
    const float* hostXY() const {
        return reinterpret_cast<const float*>(landing.data() + kXYOffset);
    }
    float* hostXYMutable() { return reinterpret_cast<float*>(landing.data() + kXYOffset); }
    const float* hostAngles() const {
        return reinterpret_cast<const float*>(landing.data() + kAnglesOffset);
    }
    const uint32_t* hostDescriptors() const {
        return reinterpret_cast<const uint32_t*>(landing.data() + kDescriptorOffset);
    }
    const uint8_t* hostKeep() const { return landing.data() + kKeepOffset; }

    ResidentFrontend(int w, int h, uint32_t poolSize, uint32_t rankCap,
                     const bincv::SteeredBriefPattern<kDescriptorBits>& hostPattern)
        : wide(w, h),
          denoised(w, h),
          pyramid(w, h),
          dxBlock(w, 2 * h),
          dyBlock(w, 2 * h),
          candidates(poolSize),
          maxBits(1),
          selectScratch(bc::goodFeaturesScratchBytes(poolSize)),
          corners(rankCap),
          resultBlock(kResultBlockBytes),
          patternPairs(bc::steeredBriefPatternPairs<kDescriptorBits>()),
          width(static_cast<size_t>(w)),
          height(static_cast<size_t>(h)),
          candidateCapacity(poolSize),
          rankCapacity(rankCap),
          frameStaging(static_cast<size_t>(w) * static_cast<size_t>(h)),
          landing(kResultBlockBytes),
          cornerStaging(kCapacity) {
        // AN EXPLICIT STREAM, NOT THE DEFAULT ONE. The default stream has
        // implicit synchronization with every other stream in the process; a
        // pipeline that means to be asynchronous says so in its own type.
        BINCV_CUDA_CHECK(cudaStreamCreate(&stream));
        // The BRIEF pattern is uploaded ONCE, not per frame: it is 30 rotated
        // copies of cv::ORB's learned table, 30,720 pairs, and it does not
        // change between frames.
        BINCV_CUDA_CHECK(bc::uploadBriefPattern<kDescriptorBits>(
            hostPattern, patternPairs.data(), pattern, stream));
        BINCV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    ~ResidentFrontend() {
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
    ResidentFrontend(const ResidentFrontend&) = delete;
    ResidentFrontend& operator=(const ResidentFrontend&) = delete;

    /// The four derivative planes, in the order the corner family takes them:
    /// magnitudes first, then signs. All four are the same type, so a
    /// transposed pair compiles silently -- naming them once, here, is the
    /// defence a call site cannot offer.
    bc::DevicePlaneBlockView dx() { return bc::planeBlock(dxBlock.view(), 2); }
    bc::DevicePlaneBlockView dy() { return bc::planeBlock(dyBlock.view(), 2); }

    /// Every device byte this frontend holds, from the containers' own closed
    /// formulas. METER: the allocation sum, and it compares binCV to binCV
    /// only -- nothing here crosses to another library.
    size_t detectionPoolBytes() const {
        return candidateCapacity * sizeof(bc::DeviceCorner) +
               bc::goodFeaturesScratchBytes(candidateCapacity);
    }
    size_t binaryStageBytes() const {
        return pyramid.sizeInBytes() +
               2 * (2 * height * dxBlock.getAlignedWidth() * sizeof(uint32_t));
    }

    size_t deviceBytes() const {
        const size_t wideBytes = wide.getWidth() * wide.getHeight();
        return 2 * wideBytes                                  // wide + denoised
               + binaryStageBytes()                           // ladder + derivatives
               + detectionPoolBytes()
               + sizeof(uint32_t)                             // maxBits
               + rankCapacity * sizeof(bc::DeviceCorner)
               + kResultBlockBytes                            // the one result block
               + bc::steeredBriefPatternPairs<kDescriptorBits>() * sizeof(bincv::BriefPair);
    }

    /// Bytes that cross the bus per frame, counted from the geometry.
    size_t uploadBytesPerFrame() const { return width * height; }
    size_t downloadBytesPerFrame() const { return kResultBlockBytes; }
};

/// One frame, enqueued end to end. **Every call below is asynchronous on
/// `fe.stream`, and there is no synchronize anywhere in this function.**
///
/// `roundTrip` selects the alternative `keypointsFromCorners` replaces -- a
/// download, a host loop and an upload -- so the two can be timed in one
/// process. It is the only path in this file that synchronizes mid-frame, and
/// it returns the number of extra synchronizations it performed.
int runFrame(ResidentFrontend& fe, const uint8_t* hostFrame, StageClock& clock,
             bool roundTrip) {
    const bincv::GoodFeaturesParams gf{};  // the reference frontend's four values
    cudaStream_t s = fe.stream;
    int extraSyncs = 0;

    clock.begin(s);

    // 1. THE FRAME CROSSES THE BUS. Once, in one direction. Everything after
    //    this point reads memory that is already on the device.
    BINCV_CUDA_CHECK(bc::uploadImage<uint8_t>(hostFrame, fe.width, fe.height, fe.width,
                                              fe.wide.view(), s));
    clock.mark(kUpload, s);

    // 2. THE SENSOR STAGE, in binCV's own spelling: the reference pipeline's
    //    L-shaped three-pixel median, then the gradient-magnitude edge filter
    //    whose no-argument defaults ARE the reference's operation. 8 bits in,
    //    1 BIT out, and the wide edge image never exists at any point.
    BINCV_CUDA_CHECK(bc::medianWide<3>(fe.wide.constView(), fe.denoised.view(),
                                       bincv::kMedianReferenceL, s));
    clock.mark(kMedian, s);
    BINCV_CUDA_CHECK(bc::edgeThreshold(fe.denoised.constView(),
                                       fe.pyramid.level<0>().plane(0),
                                       static_cast<uint8_t>(kEdgeThreshold),
                                       bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                                       bincv::EdgeSpatial::Wide, s));
    clock.mark(kEdge, s);

    // 3. THE LADDER. Levels 1..3 from level 0, three launches, nothing
    //    allocated and nothing transferred.
    BINCV_CUDA_CHECK(bc::buildPyramidBox(fe.pyramid, s));
    clock.mark(kPyramid, s);

    // 4. BOTH DERIVATIVES FROM ONE TRAVERSAL. Level 0 is a bit matrix, so the
    //    binary spelling applies with no adapter; each destination is a
    //    2-plane block (magnitude, then sign).
    BINCV_CUDA_CHECK(bc::derivativeXY(fe.pyramid.level<0>().plane(0), fe.dx(), fe.dy(),
                                      bincv::BORDER_REFLECT_101, false, s));
    clock.mark(kDerivative, s);

    // 5. DETECTION. The counter and the frame maximum are the two pieces of
    //    device state that must start each frame clean; `frameMap` is left
    //    empty because the fused arm materialises no frame-sized float map.
    BINCV_CUDA_CHECK(fe.candidateCounter.reset(s));
    BINCV_CUDA_CHECK(cudaMemsetAsync(fe.maxBits.data(), 0, sizeof(uint32_t), s));
    bc::DeviceGoodFeaturesWorkspace work;
    work.candidates = bc::appendBuffer(fe.candidates, fe.candidateCounter);
    work.maxBits = fe.maxBits.data();
    work.scratch = fe.selectScratch.data();
    work.scratchBytes = fe.selectScratch.size();
    BINCV_CUDA_CHECK(bc::goodFeaturesToTrackAsync(
        fe.dx().plane(0), fe.dy().plane(0), fe.dx().plane(1), fe.dy().plane(1), gf, work,
        fe.corners.data(), fe.rankCapacity, fe.result(), s));
    clock.mark(kDetect, s);

    // 6. THE LINK. The detector wrote corner RECORDS; the orientation and
    //    descriptor families read interleaved (x, y) FLOATS. On the device
    //    that is one launch; off it, it is a download, a host loop, an upload
    //    and -- the part that actually costs -- a synchronize in the middle of
    //    a frame whose remaining work is already enqueued behind it.
    if (!roundTrip) {
        BINCV_CUDA_CHECK(bc::keypointsFromCorners(fe.corners.data(),
                                                  bc::deviceCornerCount(fe.result()),
                                                  fe.keypointXY(), kCapacity, s));
    } else {
        BINCV_CUDA_CHECK(cudaMemcpyAsync(fe.landing.data(), fe.resultBlock.data(),
                                         sizeof(bc::DeviceCornerResult),
                                         cudaMemcpyDeviceToHost, s));
        BINCV_CUDA_CHECK(cudaMemcpyAsync(fe.cornerStaging.data(), fe.corners.data(),
                                         kCapacity * sizeof(bc::DeviceCorner),
                                         cudaMemcpyDeviceToHost, s));
        BINCV_CUDA_CHECK(cudaStreamSynchronize(s));
        ++extraSyncs;
        const uint32_t n = std::min(fe.hostResult().count, kCapacity);
        for (uint32_t i = 0; i < kCapacity; ++i) {
            fe.hostXYMutable()[2 * i] =
                i < n ? static_cast<float>(fe.cornerStaging[i].x) : 0.0f;
            fe.hostXYMutable()[2 * i + 1] =
                i < n ? static_cast<float>(fe.cornerStaging[i].y) : 0.0f;
        }
        BINCV_CUDA_CHECK(cudaMemcpyAsync(fe.keypointXY(), fe.hostXY(),
                                         2 * kCapacity * sizeof(float),
                                         cudaMemcpyHostToDevice, s));
    }
    clock.mark(kKeypoints, s);

    // 7. ORIENTATION AND DESCRIPTION, both over the DENOISED WIDE FRAME that
    //    has been resident since stage 2. The launches are sized by CAPACITY,
    //    not by the corner count -- the count is still on the device, and that
    //    is exactly why this frame needs no synchronize. Slots past the count
    //    hold (0, 0), which every consumer's bounds test rejects: keep = 0,
    //    angle 0, descriptor zero, and nothing read outside the image.
    const bc::DeviceKeypointSetConstView kps = bc::keypointSet(fe.keypointXY(), kCapacity);
    BINCV_CUDA_CHECK(bc::keypointOrientation(fe.denoised.constView(), kps, fe.angles(),
                                             fe.keep(), kOrientationRadius, nullptr, s));
    clock.mark(kOrientation, s);
    BINCV_CUDA_CHECK(bc::computeBriefSteered(
        fe.denoised.constView(), kps, fe.angles(), fe.pattern,
        bc::descriptorSet(fe.descriptors(), kCapacity, kDescriptorBits / 32, fe.keep()), s));
    clock.mark(kDescribe, s);

    // 8. THE ONE DOWNLOAD. A fixed-size block -- the result triple, the
    //    keypoints, the angles, the keep bytes and the descriptors -- because
    //    sizing it by the count would need the count on the host, which is the
    //    synchronize this pipeline exists not to perform. Only the first
    //    `count` entries mean anything, and `count` arrives in the same block.
    BINCV_CUDA_CHECK(cudaMemcpyAsync(fe.landing.data(), fe.resultBlock.data(),
                                     ResidentFrontend::kResultBlockBytes,
                                     cudaMemcpyDeviceToHost, s));
    clock.mark(kDownload, s);

    // THE FRAME'S ONE SYNCHRONIZATION. Everything above is enqueued; this is
    // where the host waits, once, for all of it.
    BINCV_CUDA_CHECK(cudaStreamSynchronize(s));
    clock.accumulate();
    return extraSyncs;
}

// ===========================================================================
// THE HOST ARM -- binCV's own CPU kernels, same stages, same frames.
//
// It is here for TWO reasons, and only the first is about speed:
//
//  1. IT IS THE CORRECTNESS ORACLE. The device claim is bit-exactness against
//     the host library, and this arm is what turns that from a property of
//     twelve unit tests into a property of the whole pipeline on real frames.
//  2. It gives a CPU number beside the GPU one. That number is context, not a
//     bar: it is a CPU measurement on the same machine as the GPU one, and a
//     desktop under a hypervisor does not measure CPU time well.
// ===========================================================================
struct HostFrontend {
    std::vector<uint8_t> denoised;
    bincv::Pyramid<W, 1, 2, 2, 2> pyramid;
    bincv::SignedQuantMat<1, W> dx0, dy0;
    std::vector<float> ring;
    std::vector<bincv::Corner> candidates;
    std::vector<float> xy;
    std::vector<float> angles;
    std::vector<uint8_t> keep;
    std::vector<W> descriptors;
    const bincv::SteeredBriefPattern<kDescriptorBits>& pattern;
    size_t width, height;
    bincv::CornerResult result{};

    /// The same stage split as the device arm's, on the host clock. It is here
    /// so the comparison can say WHERE each side spends its frame rather than
    /// only that one total is larger: a CPU-vs-GPU total with no split cannot
    /// tell "the GPU is slow" from "one stage is slow on both".
    double msSensor = 0, msPyramid = 0, msDerivative = 0, msDetect = 0, msOrient = 0,
           msDescribe = 0;

    HostFrontend(int w, int h, const bincv::SteeredBriefPattern<kDescriptorBits>& p)
        : denoised(static_cast<size_t>(w) * static_cast<size_t>(h)),
          pyramid(w, h),
          dx0(w, h),
          dy0(w, h),
          ring(bincv::kResponseRingRows * static_cast<size_t>(w)),
          candidates(rankCapacity()),
          xy(2 * kCapacity),
          angles(kCapacity),
          keep(kCapacity),
          descriptors(size_t{kCapacity} * (kDescriptorBits / 32)),
          pattern(p),
          width(static_cast<size_t>(w)),
          height(static_cast<size_t>(h)) {}

    void runFrame(const uint8_t* frame) {
        const bincv::GoodFeaturesParams gf{};
        auto t = std::chrono::steady_clock::now();
        bincv::medianWide<3, uint8_t>(frame, width, height, width, denoised.data(), width,
                                      bincv::kMedianReferenceL);
        bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                             bincv::EdgeSpatial::Wide, uint8_t, W>(
            denoised.data(), width, height, width, pyramid.level<0>().plane(0),
            static_cast<uint8_t>(kEdgeThreshold));
        msSensor += msSince(t);
        t = std::chrono::steady_clock::now();
        pyramid.build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
        msPyramid += msSince(t);
        t = std::chrono::steady_clock::now();
        bincv::derivativeX(pyramid.level<0>(), dx0);
        bincv::derivativeY(pyramid.level<0>(), dy0);
        msDerivative += msSince(t);
        bincv::ResponseMap ringView{ring.data(), width, bincv::kResponseRingRows, width};
        t = std::chrono::steady_clock::now();
        result = bincv::goodFeaturesToTrackStreaming<W>(dx0, dy0, gf, ringView,
                                                        candidates.data(), candidates.size());
        msDetect += msSince(t);
        const size_t n = result.count;
        for (size_t i = 0; i < kCapacity; ++i) {
            xy[2 * i] = i < n ? static_cast<float>(candidates[i].x) : 0.0f;
            xy[2 * i + 1] = i < n ? static_cast<float>(candidates[i].y) : 0.0f;
        }
        // Held to the DEVICE's shape deliberately: capacity keypoints, not
        // `count`, so the two arms compute the same thing and a difference is
        // a difference in the kernels rather than in how they were driven.
        t = std::chrono::steady_clock::now();
        bincv::keypointOrientation<uint8_t>(denoised.data(), width, height, width, xy.data(),
                                            kCapacity, angles.data(), keep.data(),
                                            kOrientationRadius);
        msOrient += msSince(t);
        t = std::chrono::steady_clock::now();
        bincv::computeBriefSteered<kDescriptorBits, uint8_t, W>(
            denoised.data(), width, height, width, xy.data(), kCapacity, angles.data(),
            pattern, descriptors.data(), keep.data());
        msDescribe += msSince(t);
    }
};

/// What the two arms disagreed about, over the whole run. Every field is a
/// COUNT rather than a bool: "they differ" is not a finding, "they differ on 3
/// of 118,000 descriptors, all of them at a rotation-bin boundary" is.
struct Divergence {
    size_t frames = 0;
    size_t cornerCountMismatch = 0;
    size_t cornerPositionMismatch = 0;
    size_t cornerResponseMismatch = 0;
    size_t keepMismatch = 0;
    size_t binMismatch = 0;
    size_t descriptorWordMismatch = 0;
    size_t descriptorsCompared = 0;
    float maxAngleDiff = 0.0f;
};

void compareArms(const ResidentFrontend& dev, const std::vector<bincv::Corner>& hostCorners,
                 const HostFrontend& host, Divergence& d) {
    ++d.frames;
    const uint32_t dn = std::min(dev.hostResult().count, kCapacity);
    const size_t hn = std::min(host.result.count, size_t{kCapacity});
    if (dn != hn) {
        ++d.cornerCountMismatch;
        return;  // nothing below is comparable once the sets differ in size
    }
    for (size_t i = 0; i < hn; ++i) {
        if (dev.hostXY()[2 * i] != static_cast<float>(hostCorners[i].x) ||
            dev.hostXY()[2 * i + 1] != static_cast<float>(hostCorners[i].y)) {
            ++d.cornerPositionMismatch;
        }
    }
    for (size_t i = 0; i < kCapacity; ++i) {
        if (dev.hostKeep()[i] != host.keep[i]) ++d.keepMismatch;
        const float diff = std::fabs(dev.hostAngles()[i] - host.angles[i]);
        if (diff > d.maxAngleDiff) d.maxAngleDiff = diff;
        // The bin is what a descriptor actually depends on, so a bin
        // disagreement is counted separately from a word disagreement: one
        // ULP of angle at a bin boundary replaces the whole descriptor, and
        // reporting that as "256 bits differ" would hide the mechanism.
        const bool sameBin = bincv::briefAngleBin(dev.hostAngles()[i]) ==
                             bincv::briefAngleBin(host.angles[i]);
        if (!sameBin) ++d.binMismatch;
        if (dev.hostKeep()[i] == 0 || host.keep[i] == 0) continue;
        ++d.descriptorsCompared;
        if (!sameBin) continue;
        for (size_t w = 0; w < kDescriptorBits / 32; ++w) {
            if (dev.hostDescriptors()[i * (kDescriptorBits / 32) + w] !=
                host.descriptors[i * (kDescriptorBits / 32) + w]) {
                ++d.descriptorWordMismatch;
            }
        }
    }
}

/// THE LAUNCH FLOOR ON AN IDLE STREAM: `denoiseMedian3` on a 32x1 matrix is one
/// block of one warp doing three integer operations, so what this measures is
/// not the kernel -- it is what it costs to START one when nothing else is
/// queued. An upper bound on the empty-launch floor.
///
/// **A PIPELINE STAGE IS ALLOWED TO READ BELOW THIS, AND SEVERAL HERE DO.**
/// That is not a contradiction, it is the residency argument showing up in the
/// measurement: with an idle stream the host's enqueue call IS the cost, while
/// in the frame loop every launch of the frame is already queued behind a long
/// kernel, so the events measure GPU time with the enqueue latency hidden. The
/// floor is what a launch costs when the HOST is the bottleneck; the stage times
/// are what the work costs when it is not.
double measureLaunchFloorMs(cudaStream_t s) {
    bc::DeviceBinMat tiny(32, 1);
    bc::DeviceBinMat tinyDst(32, 1);
    cudaEvent_t a, b;
    BINCV_CUDA_CHECK(cudaEventCreate(&a));
    BINCV_CUDA_CHECK(cudaEventCreate(&b));
    std::vector<double> samples;
    for (int round = 0; round < 32; ++round) {
        BINCV_CUDA_CHECK(cudaEventRecord(a, s));
        for (int i = 0; i < 16; ++i) {
            BINCV_CUDA_CHECK(bc::denoiseMedian3(tiny.constView(), tinyDst.view(), s));
        }
        BINCV_CUDA_CHECK(cudaEventRecord(b, s));
        BINCV_CUDA_CHECK(cudaStreamSynchronize(s));
        float ms = 0.0f;
        BINCV_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        if (round > 0) samples.push_back(static_cast<double>(ms) / 16.0);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(samples.begin(), samples.end());
    const double floorMs = samples.empty() ? 0.0 : samples[samples.size() / 2];
    return floorMs;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf(
            "usage: cuda_vio_frontend <sequence.bsq> [max-frames]\n"
            "  The blob comes from scripts/make_sequence_blob.py --mode 8bit, which is\n"
            "  how a build with no OpenCV reads real dataset frames:\n"
            "    scripts/make_sequence_blob.py <euroc-cam0-dir> -o v1_02.bsq --mode 8bit\n");
        return 2;
    }
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::printf("no CUDA device available\n");
        return 77;
    }

    const std::string path = argv[1];
    const size_t maxFrames = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 0;
    const bool withHost = envOn("BINCV_CUDA_VIO_HOST");
    const bool roundTrip = envOn("BINCV_CUDA_VIO_ROUNDTRIP");
    const bool perStage = envOn("BINCV_CUDA_VIO_STAGES", true);

    // ---- the frame source ---------------------------------------------------
    std::FILE* blob = std::fopen(path.c_str(), "rb");
    if (blob == nullptr) {
        std::printf("cannot open %s\n", path.c_str());
        return 2;
    }
    uint8_t head[bincv::kSequenceHeaderBytes];
    bincv::SequenceHeader sh;
    if (std::fread(head, 1, sizeof(head), blob) != sizeof(head) ||
        !(sh = bincv::readSequenceHeader(head, sizeof(head))).valid) {
        std::printf("%s is not a BSQ1 sequence blob\n", path.c_str());
        std::fclose(blob);
        return 2;
    }
    if (sh.mode != bincv::kSequenceMode8Bit) {
        // Refused rather than accepted: a packed blob's sensor stage already
        // ran on the host, so it cannot exercise the device sensor stage that
        // is a third of what this example exists to show.
        std::printf("%s is a PACKED blob. This example needs --mode 8bit: its first two\n"
                    "stages ARE the sensor stage, and a packed blob has already run it.\n",
                    path.c_str());
        std::fclose(blob);
        return 2;
    }
    const int w = static_cast<int>(sh.width);
    const int h = static_cast<int>(sh.height);
    size_t totalFrames = sh.frameCount;
    if (maxFrames != 0 && totalFrames > maxFrames) totalFrames = maxFrames;
    if (totalFrames == 0) {
        std::printf("the blob holds no frames\n");
        std::fclose(blob);
        return 2;
    }

    // ---- the BRIEF pattern: cv::ORB's learned table, rotated into 30 bins ----
    // On the heap because 30 x 256 pairs is 30,720 bytes, which does not belong
    // on a stack. Built once, uploaded once, used by both arms -- so a
    // descriptor difference cannot be a difference in the pattern.
    auto steered = std::make_unique<bincv::SteeredBriefPattern<kDescriptorBits>>();
    bincv::makeSteeredBriefPattern<kDescriptorBits>(*steered, bincv::kOrbBriefPattern);

    // ---- the memory meter, read across construction -------------------------
    size_t freeBefore = 0, totalMem = 0;
    BINCV_CUDA_CHECK(cudaFree(nullptr));  // force context creation BEFORE the reading
    BINCV_CUDA_CHECK(cudaMemGetInfo(&freeBefore, &totalMem));

    ResidentFrontend fe(w, h, candidatePoolSize(), rankCapacity(), *steered);

    size_t freeAfter = 0;
    BINCV_CUDA_CHECK(cudaMemGetInfo(&freeAfter, &totalMem));
    const size_t driverDelta = freeBefore - freeAfter;

    std::unique_ptr<HostFrontend> host;
    std::vector<bincv::Corner> hostCorners;
    if (withHost) {
        host = std::make_unique<HostFrontend>(w, h, *steered);
        hostCorners.resize(kCapacity);
    }

    // ---- the banner ---------------------------------------------------------
    std::printf("=== A DEVICE-RESIDENT binary-frame VIO frontend on binCV's CUDA kernels ===\n");
    std::printf("  %zu frames, %dx%d, 1/2/2/2 ladder, %u keypoint slots, BRIEF-%zu\n",
                totalFrames, w, h, kCapacity, kDescriptorBits);
    std::printf("  frames from a BSQ1 blob, 8-bit bodies, one frame resident at a time\n");
    std::printf("  keypoint link: %s\n",
                roundTrip ? "DOWNLOAD-CONVERT-UPLOAD (the round trip, for comparison)"
                          : "cuda::keypointsFromCorners (device-resident)");
    std::printf("  stage timing : %s\n",
                perStage ? "per-stage CUDA events on the pipeline's own stream"
                         : "OUTER PAIR ONLY (per-stage instrumentation off)");
    std::printf("  host arm     : %s\n\n",
                withHost ? "ON -- binCV's CPU kernels on the same frames"
                         : "off (BINCV_CUDA_VIO_HOST=1 to compare)");

    // ---- the frame loop -----------------------------------------------------
    StageClock clock(kStageNames, kStageCount, perStage);
    Divergence div;
    size_t frames = 0, overflows = 0, truncated = 0, extraSyncs = 0;
    size_t sumLive = 0, sumKept = 0, maxRanked = 0;
    double hostMs = 0.0, wallMs = 0.0;

    for (size_t f = 0; f < totalFrames; ++f) {
        // ONE frame body resident at a time, read straight into PAGE-LOCKED
        // memory. The blob is never resident: an 8-bit frame is the buffer
        // binCV exists not to hold, and a 217 MB blob is 600 of them.
        if (std::fread(fe.frameStaging.data(), 1, fe.frameStaging.size(), blob) !=
            fe.frameStaging.size()) {
            std::printf("  blob truncated at frame %zu; stopping\n", f);
            break;
        }
        const auto t0 = std::chrono::steady_clock::now();
        extraSyncs +=
            static_cast<size_t>(runFrame(fe, fe.frameStaging.data(), clock, roundTrip));
        wallMs += msSince(t0);

        if (fe.hostResult().candidateOverflow) ++overflows;
        if (fe.hostResult().candidatesTruncated) ++truncated;
        maxRanked = std::max(maxRanked, static_cast<size_t>(fe.hostResult().candidatesRanked));
        const uint32_t live = std::min(fe.hostResult().count, kCapacity);
        sumLive += live;
        for (uint32_t i = 0; i < live; ++i) sumKept += fe.hostKeep()[i] != 0 ? 1u : 0u;

        if (host) {
            const auto t1 = std::chrono::steady_clock::now();
            host->runFrame(fe.frameStaging.data());
            hostMs += msSince(t1);
            const size_t hn = std::min(host->result.count, size_t{kCapacity});
            for (size_t i = 0; i < hn; ++i) hostCorners[i] = host->candidates[i];
            compareArms(fe, hostCorners, *host, div);
        }
        ++frames;
        if (frames % 200 == 0) std::printf("  ... %zu frames, %u keypoints live\n", frames,
                                           live);
    }
    std::fclose(blob);
    if (frames == 0) {
        std::printf("no frames processed\n");
        return 2;
    }
    const double fd = static_cast<double>(frames);
    // MEASURED AFTER THE LOOP, DELIBERATELY. Taken before it, this reads a cold
    // and contended device and comes back LARGER than stages the loop actually
    // timed -- which would make the floor look like a bound the pipeline beats.
    const double floorMs = measureLaunchFloorMs(fe.stream);

    // ---- what a VIO backend is handed ---------------------------------------
    std::printf("\n--- WHAT THE BACKEND IS HANDED ---\n");
    std::printf("  frames processed      : %zu\n", frames);
    std::printf("  keypoints per frame   : %.1f of %u slots\n",
                static_cast<double>(sumLive) / fd, kCapacity);
    std::printf("  describable (keep = 1): %.1f per frame (%.1f%% -- the rest are within\n"
                "                          the 31-pixel patch of a border)\n",
                static_cast<double>(sumKept) / fd,
                100.0 * static_cast<double>(sumKept) / static_cast<double>(std::max<size_t>(sumLive, 1)));
    std::printf("  NMS pool peak         : %zu ranked, device pool %u%s\n", maxRanked,
                fe.candidateCapacity, overflows ? "  <-- OVERFLOWED" : "");
    std::printf("  ranking capacity      : %u (BINCV_CUDA_VIO_RANK)\n", fe.rankCapacity);
    if (overflows) {
        std::printf("  *** %zu frames overflowed the DEVICE candidate pool. The selection\n"
                    "      cannot report a ranked count at all in that case -- size the pool\n"
                    "      from the peak above and re-run. ***\n", overflows);
    }
    if (truncated) {
        std::printf("  *** %zu frames could not rank every NMS survivor in %u ranking\n"
                    "      slots; the count returned is a LOWER BOUND on the reference's. ***\n",
                    truncated, fe.rankCapacity);
    }

    // ---- where the frame goes ----------------------------------------------
    double computeMs = 0.0;
    if (perStage) {
        for (size_t i = kMedian; i <= kDescribe; ++i) computeMs += clock.meanMs(i);
    }
    std::printf("\n--- WHERE THE FRAME GOES (CUDA events, ONE explicit stream) ---\n");
    if (perStage) {
        std::printf("  %-30s %9s %8s\n", "stage", "ms/frame", "share");
        for (size_t i = 0; i < kStageCount; ++i) {
            const bool isCompute = i >= kMedian && i <= kDescribe;
            const double ms = clock.meanMs(i);
            if (isCompute) {
                std::printf("  %-30s %9.4f %7.1f%%%s\n", clock.name(i), ms,
                            computeMs > 0.0 ? 100.0 * ms / computeMs : 0.0,
                            ms < floorMs ? "   <-- under the idle-stream launch floor" : "");
            } else {
                std::printf("  %-30s %9.4f %8s\n", clock.name(i), ms, "transfer");
            }
        }
        std::printf("  %-30s %9.4f %7.1f%%\n", "DEVICE COMPUTE (the 8 above)     ", computeMs,
                    100.0);
    } else {
        std::printf("  per-stage timing is OFF; only the frame total below was measured.\n");
    }
    std::printf("  %-30s %9.4f\n", "FRAME TOTAL, device            ", clock.meanWholeMs());
    std::printf("  %-30s %9.4f\n", "FRAME TOTAL, host wall clock  ", wallMs / fd);
    std::printf("\n  idle-stream launch floor (proxy: the cheapest real launch in this\n"
                "  library, denoiseMedian3 on a 32x1 matrix): %.4f ms -- what STARTING a\n"
                "  kernel costs when nothing is queued. Stages here read below it because\n"
                "  the frame's launches are all queued behind a long kernel, so their\n"
                "  enqueue latency is hidden. That hiding is the residency argument, and\n"
                "  it stops working the moment the long kernel goes away.\n",
                floorMs);
    // THE COST OF DRIVING THE PIPELINE, which is not on any stage's line and is
    // bigger than most of them. The device clock measures the stream; the wall
    // clock measures the stream PLUS what the host spent enqueueing 15 launches
    // and waiting on one synchronize. On this host the difference is larger than
    // the whole pipeline outside detection, which is a bound no per-kernel
    // benchmark can show and no kernel change can move.
    std::printf("\n  host-side cost of DRIVING the frame (wall - device): %.4f ms/frame,\n"
                "  over %d launches and one synchronize. Compare it against the stages\n"
                "  above before optimizing any of them.\n",
                wallMs / fd - clock.meanWholeMs(), 15);
    if (perStage) {
        std::printf("  The per-stage marks are themselves stream work. Re-run with\n"
                    "  BINCV_CUDA_VIO_STAGES=0 for a frame total with no marks in it.\n");
    }

    // ---- the bus ------------------------------------------------------------
    std::printf("\n--- WHAT RESIDENCY COSTS THE BUS (counted from the geometry) ---\n");
    std::printf("  up, per frame   : %8zu B   the wide frame, once\n",
                fe.uploadBytesPerFrame());
    std::printf("  down, per frame : %8zu B   result + keypoints + angles + keep + %zu-bit\n"
                "                                descriptors, all %u slots, one fixed block\n",
                fe.downloadBytesPerFrame(), kDescriptorBits, kCapacity);
    std::printf("  ratio           : %8.1fx more up than down\n",
                static_cast<double>(fe.uploadBytesPerFrame()) /
                    static_cast<double>(fe.downloadBytesPerFrame()));
    std::printf("  syncs per frame : %8.2f   %s\n", 1.0 + static_cast<double>(extraSyncs) / fd,
                roundTrip ? "(the round-trip arm stalls mid-frame)"
                          : "(one, after every launch is enqueued)");
    std::printf("  NOTHING FRAME-SIZED COMES BACK. The binary frame, the ladder, the\n"
                "  derivatives and the response never cross the bus in either direction.\n");

    // ---- memory -------------------------------------------------------------
    std::printf("\n--- PEAK DEVICE MEMORY ---\n");
    std::printf("  METER 1, allocation sum (binCV to binCV, closed formula): %zu B (%.2f MB)\n",
                fe.deviceBytes(), static_cast<double>(fe.deviceBytes()) / (1024.0 * 1024.0));
    std::printf("    of which the two WIDE frames            : %8zu B\n",
                2 * static_cast<size_t>(w) * static_cast<size_t>(h));
    std::printf("    the 1/2/2/2 binary ladder               : %8zu B\n",
                fe.pyramid.sizeInBytes());
    std::printf("    the two ternary derivative blocks       : %8zu B\n",
                2 * (2 * fe.height * fe.dxBlock.getAlignedWidth() * sizeof(uint32_t)));
    std::printf("    the detection pool + its sort scratch   : %8zu B\n",
                fe.detectionPoolBytes());
    std::printf("    the steered BRIEF pattern (once, not per frame): %zu B\n",
                bc::steeredBriefPatternPairs<kDescriptorBits>() * sizeof(bincv::BriefPair));
    std::printf("  METER 2, cudaMemGetInfo across construction            : %zu B (%.2f MB)\n",
                driverDelta, static_cast<double>(driverDelta) / (1024.0 * 1024.0));
    std::printf("    This driver reserves in 2 MB units, so meter 2 is meter 1 rounded up to\n"
                "    a multiple of 2 MB and the two are NOT a ratio. They are printed\n"
                "    together so a reader can see they agree to within one granule.\n");
    std::printf("  THE WIDE FRAMES DOMINATE, and that is the honest shape of this pipeline:\n"
                "  BRIEF tests `img[a] < img[b]` on GRAYSCALE, so a byte frame has to stay\n"
                "  resident. The binary representation pays for the sensor stage, the\n"
                "  ladder, the derivatives and detection -- %zu B of the total -- and pays\n"
                "  nothing for orientation and description.\n",
                fe.pyramid.sizeInBytes() +
                    2 * (2 * fe.height * fe.dxBlock.getAlignedWidth() * sizeof(uint32_t)));

    // ---- the host arm -------------------------------------------------------
    if (host) {
        std::printf("\n--- THE HOST ARM: binCV's OWN CPU KERNELS, SAME STAGES, SAME FRAMES ---\n");
        std::printf("  CORRECTNESS, which is what this arm is really for:\n");
        std::printf("    frames compared                 : %zu\n", div.frames);
        std::printf("    corner COUNT differed on         : %zu frames\n", div.cornerCountMismatch);
        std::printf("    corner POSITION differed on      : %zu keypoints\n",
                    div.cornerPositionMismatch);
        std::printf("    `keep` byte differed on          : %zu keypoints\n", div.keepMismatch);
        std::printf("    rotation BIN differed on         : %zu keypoints  (the angle is a\n"
                    "                                       transcendental and is NOT claimed\n"
                    "                                       bit-exact; one ULP at a 12-degree\n"
                    "                                       boundary replaces a descriptor)\n",
                    div.binMismatch);
        std::printf("    descriptor WORDS differed on     : %zu of %zu compared (same-bin only)\n",
                    div.descriptorWordMismatch, div.descriptorsCompared * (kDescriptorBits / 32));
        std::printf("    max |angle difference|           : %.3e rad\n",
                    static_cast<double>(div.maxAngleDiff));
        std::printf("\n  WHERE THE HOST FRAME GOES (steady clock, one thread):\n");
        const double hostTotal = host->msSensor + host->msPyramid + host->msDerivative +
                                 host->msDetect + host->msOrient + host->msDescribe;
        const char* hostNames[6] = {"sensor: medianWide + edgeThreshold",
                                    "pyramid: 3 x pyrDownBox           ",
                                    "derivatives: derivativeX + Y      ",
                                    "detect: goodFeaturesToTrack       ",
                                    "orientation: centroid r=15        ",
                                    "describe: steered BRIEF-256       "};
        const double hostStages[6] = {host->msSensor, host->msPyramid, host->msDerivative,
                                      host->msDetect, host->msOrient, host->msDescribe};
        for (int i = 0; i < 6; ++i) {
            std::printf("    %-36s %8.4f ms %6.1f%%\n", hostNames[i], hostStages[i] / fd,
                        hostTotal > 0.0 ? 100.0 * hostStages[i] / hostTotal : 0.0);
        }
        std::printf("\n  SPEED, and read the label before the number:\n");
        std::printf("    host  (CPU, one thread)         : %8.3f ms/frame\n", hostMs / fd);
        std::printf("    device (GPU, resident)          : %8.3f ms/frame  [wall clock]\n",
                    wallMs / fd);
        std::printf("    ratio                           : %8.1fx\n",
                    (wallMs > 0.0) ? (hostMs / wallMs) : 0.0);
        std::printf("    *** THIS IS NOT A GPU-vs-GPU RESULT AND IS NOT A SHIP BAR. It is a\n"
                    "    same-machine CPU measurement against a GPU one, and this host is a\n"
                    "    desktop under a hypervisor whose CPU timings carry tens of percent\n"
                    "    of spread. There is no cv::cuda whole-frontend counterpart to put\n"
                    "    here, so the GPU speed verdict for this pipeline is OUTSTANDING. ***\n");
    } else {
        std::printf("\n  No host arm was run, so nothing here checks the device answers\n"
                    "  against the host library. Re-run with BINCV_CUDA_VIO_HOST=1.\n");
    }

    std::printf("\n--- WHAT THIS DOES AND DOES NOT SHOW ---\n");
    std::printf("  SHOWS: the wide frame crosses once per frame and nothing frame-sized\n"
                "  comes back; the whole detect-and-describe chain composes on device; and\n"
                "  where the frame budget actually goes, which is what decides what is\n"
                "  worth optimizing next. A stage at 3%% of the frame cannot be worth more\n"
                "  than 1.03x however fast it gets.\n");
    std::printf("  DOES NOT SHOW: a GPU-vs-GPU speed result (no counterpart exists); a\n"
                "  TRACKING frontend (a device Lucas-Kanade exists but this program does\n"
                "  not call it, so the stage that is 62-67%% of the host frontend is absent\n"
                "  from BOTH arms here -- cuda_role_benchmark sequence times that one); or\n"
                "  anything about pose error, which is outside this library.\n");
    return 0;
}
