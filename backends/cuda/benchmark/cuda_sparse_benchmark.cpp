// Sparse stereo, block matching and descriptor matching: every arm priced at
// birth, the decision rule printed before the first number, and the role bars
// run on ONE EXPLICIT STREAM on both sides.
//
// PLAIN C++, NOT A CUDA TRANSLATION UNIT, apart from two small .cu files. This
// file times the HOST library's matcher next to the device's, and a host header
// compiled by nvcc is not the host arm a caller runs -- the reason
// cuda_median_benchmark and cuda_orientation_descriptor_benchmark give. Only
// the launch floor's empty kernel and the OpenCV arm's ratio pass need nvcc.
//
// ONE EXECUTABLE, ALWAYS BUILT, with the cv::cuda arms behind
// BINCV_CUDA_SPARSE_OPENCV inside it -- the shape cuda_sensor_benchmark and
// cuda_frontend_benchmark use, and for the reason their CMake blocks give:
// scripts/verify_cuda.sh derives its benchmark list from the TEXT of
// benchmark/CMakeLists.txt and hand-excludes exactly one name, so a target that
// exists only when an OpenCV is pointed at is one the gate tries to build on
// every machine that has not built one. Without that OpenCV every binCV arm
// still runs and the role verdicts print BLOCKED rather than being given a
// substitute bar.
//
// ---------------------------------------------------------------------------
// WHY BOTH SIDES OF EVERY PAIR RUN ON ONE EXPLICIT STREAM
//
// OpenCV synchronizes the WHOLE DEVICE on the default stream -- a guard of the
// form `if (stream == 0) cudaSafeCall(cudaDeviceSynchronize())` sits in cudev's
// grid transform, in every cudafilters filter, in cudawarping and three times
// in cudastereo. Measured elsewhere in this backend at up to 7.18x against
// binCV controls at 1.03x. Two published headlines had to be withdrawn over it.
// So every arm here, the launch floor included, runs on `gStream` and the
// events are recorded on it.
//
// ---------------------------------------------------------------------------
// WHAT THE FRAMES ARE
//
// Real EuRoC frames when a directory is given (argv[1], or $BINCV_FRAME_DIR),
// because the GATE question in CASE B turns on the spatial distribution of real
// keypoints and a uniform scatter would answer a different question. The stereo
// pair is a real frame against ITSELF SHIFTED by a known disparity -- real image
// content with ground truth, and it is labelled as that rather than as a real
// stereo pair, because no rectified EuRoC pair is on this machine (only cam0
// was fetched). Without a frame directory every arm falls back to a synthetic
// frame and the header says so in as many words.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/sparseMatch.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/blockMatch.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/fast.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/stereo.hpp"
#include "cuda_bench_util.hpp"
#include "measure_util.hpp"

#if BINCV_CUDA_SPARSE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

/// @brief The device-side ratio pass that completes the cv::cuda BFMatcher arm.
/// Defined in cuda_sparse_bench_kernels.cu; see that file for why it is
/// committed rather than replaced by `knnMatchConvert`.
void launchBfRatioPass(const void* trainIdxRow, const void* distanceRow, int nQuery,
                       unsigned maxRatio, void* dOut, cudaStream_t stream);

using cudabench::PairedTiming;
using cudabench::Timing;

namespace {

// ---------------------------------------------------------------------------
// Shapes and constants
// ---------------------------------------------------------------------------

constexpr size_t kW = 752, kH = 480;      ///< the reference frame geometry
constexpr size_t kBits = 256;             ///< descriptor length
constexpr size_t kWords = kBits / 32;
constexpr size_t kFrontendCount = 470;    ///< the frontend's measured operating point
constexpr size_t kMapCount = 5000;        ///< the relocalisation / loop-closure regime
constexpr size_t kStereoKeypoints = 500;
constexpr int kStereoShift = 21;          ///< the known disparity of the synthetic pair
constexpr int kRounds = 15;
constexpr int kReplicas = 32;             ///< working sets per meter-2 reading

/// ONE EXPLICIT STREAM FOR THE WHOLE RUN. See the header comment.
cudaStream_t gStream = nullptr;

uint64_t gRng = 0x9E3779B97F4A7C15ULL;
uint32_t nextU32() {
    gRng = gRng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(gRng >> 32);
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

struct Frames {
    std::vector<uint8_t> a, b;   ///< two consecutive frames, kW x kH, 8-bit
    bool real = false;
    std::string source = "synthetic (no frame directory given)";
};

/// @brief A frame with structure in it, for the fallback path. Pure noise makes
/// a corner detector's output nearly independent of its input, and a flat
/// frame does the same the other way.
std::vector<uint8_t> syntheticFrame(size_t w, size_t h, uint64_t seed) {
    gRng = seed;
    std::vector<uint8_t> img(w * h);
    for (auto& v : img) v = static_cast<uint8_t>(nextU32() >> 24);
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

Frames loadFrames(const char* dir) {
    Frames f;
    f.a = syntheticFrame(kW, kH, 0xFEEDFACEULL);
    f.b = syntheticFrame(kW, kH, 0xFEEDFACFULL);
#if BINCV_CUDA_SPARSE_OPENCV
    if (dir != nullptr && dir[0] != '\0') {
        std::vector<cv::String> names;
        cv::glob(std::string(dir) + "/*.png", names, false);
        std::sort(names.begin(), names.end());
        if (names.size() >= 2) {
            const cv::Mat m0 = cv::imread(names[0], cv::IMREAD_GRAYSCALE);
            const cv::Mat m1 = cv::imread(names[1], cv::IMREAD_GRAYSCALE);
            if (!m0.empty() && !m1.empty() &&
                static_cast<size_t>(m0.cols) >= kW && static_cast<size_t>(m0.rows) >= kH) {
                for (size_t y = 0; y < kH; ++y) {
                    std::memcpy(f.a.data() + y * kW, m0.ptr<uint8_t>(static_cast<int>(y)), kW);
                    std::memcpy(f.b.data() + y * kW, m1.ptr<uint8_t>(static_cast<int>(y)), kW);
                }
                f.real = true;
                f.source = std::string("real frames from ") + dir + " (first two, cropped to " +
                           std::to_string(kW) + "x" + std::to_string(kH) + ")";
            }
        }
    }
#else
    (void)dir;
#endif
    return f;
}

/// @brief The `count` strongest FAST corners of a wide frame, as the
/// interleaved (x, y) floats the descriptor family takes.
/// @note REAL corner positions, because CASE B's whole question is what
/// fraction of candidates a 48-pixel window admits, and that is a property
/// of where features actually sit. A uniform scatter answers a different
/// question. The threshold is lowered until at least `count` corners exist,
/// and the strongest `count` are kept -- which is the set a frontend's
/// selection stage would hand a matcher.
std::vector<float> strongestCorners(const std::vector<uint8_t>& img, size_t w, size_t h,
                                    size_t count, long long& thresholdUsed,
                                    int margin = 0) {
    std::vector<bincv::FastCorner> corners(200000);
    size_t n = 0;
    long long t = 40;
    for (; t >= 2; t = t / 2) {
        n = bincv::detectFast<uint8_t>(img.data(), w, h, w, t, corners.data(), corners.size());
        if (margin > 0) {
            // Keep only corners whose BRIEF patch -- and, for the stereo pair,
            // its shifted twin -- lies wholly inside the frame. A border
            // keypoint gets an all-zero descriptor by the family's own rule,
            // and a set of all-zero descriptors matches everything at distance
            // zero, which would make an accuracy number about the border
            // instead of about the operation.
            size_t k = 0;
            for (size_t i = 0; i < n; ++i) {
                const long long x = corners[i].x, y = corners[i].y;
                if (x >= margin && x + margin < static_cast<long long>(w) && y >= margin &&
                    y + margin < static_cast<long long>(h))
                    corners[k++] = corners[i];
            }
            n = k;
        }
        if (n >= count) break;
    }
    thresholdUsed = t;
    if (n > count) {
        std::nth_element(corners.begin(), corners.begin() + static_cast<long>(count),
                         corners.begin() + static_cast<long>(n),
                         [](const bincv::FastCorner& a, const bincv::FastCorner& b) {
                             return a.score > b.score;
                         });
        n = count;
    }
    std::vector<float> xy;
    xy.reserve(2 * count);
    for (size_t i = 0; i < n; ++i) {
        xy.push_back(static_cast<float>(corners[i].x));
        xy.push_back(static_cast<float>(corners[i].y));
    }
    // A synthetic frame can run out of corners before `count`; pad with a
    // spread so the shapes stay comparable, and the header says when that
    // happened.
    uint64_t s = 0x1234ULL;
    while (xy.size() < 2 * count) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        xy.push_back(static_cast<float>((s >> 33) % (w - 40) + 20));
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        xy.push_back(static_cast<float>((s >> 33) % (h - 40) + 20));
    }
    return xy;
}

std::vector<uint32_t> briefFor(const std::vector<uint8_t>& img, size_t w, size_t h,
                               const std::vector<float>& xy,
                               const bincv::BriefPattern<kBits>& pattern) {
    const size_t count = xy.size() / 2;
    std::vector<uint32_t> d(count * kWords, 0);
    bincv::computeBrief<kBits, uint8_t, uint32_t>(img.data(), w, h, w, xy.data(), count,
                                                  pattern, d.data(), nullptr);
    return d;
}

std::vector<int32_t> octavesFor(size_t count) {
    std::vector<int32_t> o(count);
    for (auto& v : o) v = static_cast<int32_t>(nextU32() % 4u);
    return o;
}

/// @brief One box-downsample of a wide frame -- benchmark setup, not a measured
/// op: it is how the block matcher's ladder is built on both sides.
std::vector<uint8_t> halve(const std::vector<uint8_t>& src, size_t w, size_t h, size_t& dw,
                           size_t& dh) {
    dw = w > 1 ? w / 2 : 1;
    dh = h > 1 ? h / 2 : 1;
    std::vector<uint8_t> out(dw * dh);
    for (size_t y = 0; y < dh; ++y)
        for (size_t x = 0; x < dw; ++x) {
            const size_t sy = 2 * y, sx = 2 * x;
            const unsigned a = src[sy * w + sx];
            const unsigned b = src[sy * w + (sx + 1 < w ? sx + 1 : sx)];
            const unsigned c = src[(sy + 1 < h ? sy + 1 : sy) * w + sx];
            const unsigned e = src[(sy + 1 < h ? sy + 1 : sy) * w + (sx + 1 < w ? sx + 1 : sx)];
            out[y * dw + x] = static_cast<uint8_t>((a + b + c + e) / 4u);
        }
    return out;
}

bincv::BinMat<uint32_t> packOf(const std::vector<uint8_t>& img, size_t w, size_t h) {
    bincv::BinMat<uint32_t> m(static_cast<int>(w), static_cast<int>(h));
    bincv::packBits<bincv::PackRule::GreaterThan>(img.data(), w, h, w, m.view(), uint8_t{127});
    return m;
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

void rule() {
    std::printf("---------------------------------------------------------------------------\n");
}

void printDecisionRule() {
    std::printf(
"===========================================================================\n"
" THE DECISION RULE, WRITTEN BEFORE ANY OF THE NUMBERS BELOW WERE TAKEN\n"
"===========================================================================\n"
" Per case, because there is no project-wide bar and inventing one is\n"
" forbidden. Where a threshold is a judgement nobody has made it is a\n"
" STOP AND ASK and is left empty rather than filled in.\n"
"\n"
" THE ONE MAGNITUDE EVERY CASE SHARES, and where it comes from: a difference\n"
" smaller than the larger of the two arms' printed sample RANGES is not a\n"
" difference. That is derived from the measurement's own noise rather than\n"
" chosen -- every pair below goes through timeKernelPaired, which brackets\n"
" both arms inside every round and alternates their order, and prints whether\n"
" the two ranges are disjoint.\n"
"\n"
" METRICS: (1) kernel-resident median with its spread, both sides of every\n"
" pair on ONE EXPLICIT STREAM; (2) device working set -- allocation sum for\n"
" binCV against binCV, cudaMemGetInfo delta over %d replicas for anything\n"
" crossing to OpenCV; (3) accuracy, PUBLISHED but not decided on for the two\n"
" ops whose output differs from their bar. Code size and portability decide\n"
" nothing here.\n"
"\n"
" CASE A -- the brute-force matcher against cv::cuda's BFMatcher.\n"
"   Arms: binCV's tiled matcher; knnMatchAsync(k=2) on CV_8U descriptors (what\n"
"   cv::cuda::ORB emits, and the path whose HammingDist::reduceIter issues one\n"
"   __popc per BYTE -- 32 per 256-bit descriptor); and knnMatchAsync on CV_32S\n"
"   over the IDENTICAL bytes, where matchHamming_gpu<int> issues eight. The\n"
"   CV_32S arm is the BAR, because the bar is the best existing option, not\n"
"   the worst; CV_8U is reported beside it as what the default path costs.\n"
"   TIMED REGION: knnMatchAsync plus a committed device-side ratio kernel on\n"
"   the same stream. knnMatchConvert is a HOST download and is OUTSIDE the\n"
"   event window on every arm -- timing it would compare binCV's resident\n"
"   kernel against an OpenCV round trip.\n"
"   DECIDES AT %zu x %zu. The frontend point %zu x %zu is reported and does\n"
"   NOT decide, because both arms are predicted to sit on the launch floor\n"
"   there -- which is why the floor is printed next to every arm.\n"
"   SHIPS as the recommended device arm if, at map scale, binCV is not slower\n"
"   than the CV_32S arm by more than the larger printed range AND its working\n"
"   set is not larger. Otherwise it does not ship as a recommendation: the two\n"
"   working sets are the same arrays, so no memory-side argument is available\n"
"   in either direction.\n"
"   PREDICTION: binCV ~= CV_32S within the spread; ~4x ahead of CV_8U; memory\n"
"   a wash.\n"
"\n"
" CASE B -- is the gate still the right answer on device? (issue #61's own\n"
" question). Four arms on identical inputs: device brute force, device gated,\n"
" host brute force, host gated. TWO-SIDED:\n"
"   * SPEED: the gate's speed rationale survives only if the device gated arm\n"
"     beats the device brute-force arm by more than both printed ranges at the\n"
"     frontend point. PREDICTION: it does not -- the whole match there is a\n"
"     launch's worth of work.\n"
"   * MEMORY: arithmetic, not a measurement. The gated arm carries two\n"
"     position arrays (and two octave arrays) the ungated one does not. Under\n"
"     'memory wins' that counts AGAINST the gate on device, and this rule says\n"
"     so before the numbers arrive.\n"
"   * ACCURACY: untouched by the device and not re-measured here. The device\n"
"     computes the same admitted set bit for bit; the suite pins it.\n"
"   EXPECTED FINDING, pre-registered: on device the gate is an ACCURACY\n"
"   feature and not a speed feature. If the measurement contradicts this it is\n"
"   reported as a finding, not smoothed.\n"
"   WHAT IT COVERS: a kernel-resident microbenchmark. The number that would\n"
"   decide ADOPTION is the match stage's share of an end-to-end frontend; no\n"
"   committed report records it and no resident device frontend calls these\n"
"   kernels. That is a deferral, not a number to invent.\n"
"\n"
" CASE C -- sparse stereo. cv::cuda has NO sparse stereo API (cudastereo's\n"
" four entries are all dense), so ruling R2 would make the speed verdict\n"
" OUTSTANDING -- but a stronger bar exists and is used instead, because the\n"
" bar is the best existing option a caller actually has: binCV's own device\n"
" denseDisparityBinary on the same frame, then index the map at the keypoints,\n"
" recorded in docs/reports/cuda.md at 0.39 ms and 442 KB.\n"
"   SHIPS as the recommended keypoint-shaped stereo path if coarse + refine\n"
"   together are below 0.39 ms AND the working set is below 442 KB. Both are\n"
"   recorded measurements from this repo, not judgements. If sparse is not\n"
"   faster, that is the finding and the recommendation becomes 'run the dense\n"
"   kernel and index it'.\n"
"   ACCURACY published beside the number and not decided on: the sub-pixel\n"
"   residual against a pair of KNOWN disparity.\n"
"\n"
" CASE D -- block matching against cv::cuda::SparsePyrLKOpticalFlow. ROLE\n"
" ONLY: route (b)'s algorithm on bytes against route (a)'s on bits, and\n"
" correctness is settled against the HOST binCV kernel, never against OpenCV.\n"
"   (i) FOOTPRINT, derived and falsifiable, with no threshold to choose: two\n"
"       binary pyramids are exactly 1/8 the bytes of two 8-bit ones, and the\n"
"       points, the status bytes and the scratch are added to give a COMPUTED\n"
"       ratio. The condition is that the MEASURED cudaMemGetInfo ratio\n"
"       reproduces the computed one within the meter's own measured step. If\n"
"       it does not, the footprint claim does not rest on the format and that\n"
"       is the finding. (A '4x' bar was proposed for this case and is DELETED:\n"
"       8x is derived from bits per byte, the word 'half' came from nobody.)\n"
"   (ii) SPEED: not slower by more than the larger printed range.\n"
"   (iii) ACCURACY: reported, never decided on here -- the magnitude is a\n"
"       judgement nobody has made. Case D produces numbers and a\n"
"       recommendation, not a ship.\n"
"\n"
" CASE E -- the fast-arm gate check, with the excluding input NAMED per op, so\n"
" the ~1.00x control is runnable rather than asserted:\n"
"   * the matchers and the coarse stereo stage: a descriptor wider than\n"
"     impl::kMatchTileMaxWords words, which the shared-memory staging refuses;\n"
"   * stereo refinement and block matching: a window impl::windowSpanFits\n"
"     rejects.\n"
" Anything other than ~1.00x there means the switch is not switching.\n"
"\n"
" STOP AND ASK -- thresholds nobody has set, left empty on purpose:\n"
"   1. CASE D's accuracy floor: how much tracking yield may route (a) give up\n"
"      against cv::cuda LK and still be worth recommending? Case D is\n"
"      ship-blocked on it.\n"
"   2. CASE C's sub-pixel residual floor: the residual is published; what\n"
"      residual would make the sparse arm not worth running is unset.\n"
"   3. CASE A's fallback: if binCV lands slower than the CV_32S arm, does the\n"
"      op ship for residency's sake with the gap documented, or is 'cv::cuda's\n"
"      matcher is the better option for a caller who already has OpenCV' the\n"
"      honest answer? The census entry was rescued by a memory argument; this\n"
"      one has none.\n"
"===========================================================================\n\n",
        kReplicas, kMapCount, kMapCount, kFrontendCount, kFrontendCount);
}

void printHostArm(const char* name, const measure::Timing& t) {
    std::printf(" %-44s %9.3f ms  spread %4.0f%%  [host clock]\n", name,
                t.medianNs / 1.0e6, t.spreadPct());
}

} // namespace

// ===========================================================================

int main(int argc, char** argv) {
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }

    const char* frameDir = argc > 1 ? argv[1] : std::getenv("BINCV_FRAME_DIR");

    std::printf("===========================================================================\n");
    std::printf(" SPARSE STEREO, BLOCK MATCHING AND DESCRIPTOR MATCHING -- device arms\n");
    std::printf("===========================================================================\n");
    cudabench::printDevice();
#if BINCV_CUDA_SPARSE_OPENCV
    std::printf(" OpenCV %s -- cv::cuda role arms COMPILED IN\n", CV_VERSION);
#else
    std::printf(" cv::cuda role arms NOT compiled in: CASE A and CASE D print BLOCKED.\n"
                " Point BINCV_CUDA_OPENCV_DIR at an OpenCV with cudafeatures2d and\n"
                " cudaoptflow to price them. No substitute bar is invented.\n");
#endif

    printDecisionRule();

    cudaStreamCreate(&gStream);
#if BINCV_CUDA_SPARSE_OPENCV
    cv::cuda::setDevice(0);
    { cv::cuda::GpuMat warm(16, 16, CV_8UC1); warm.setTo(cv::Scalar(0)); }
    cudaDeviceSynchronize();
    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(gStream);
#endif

    const Timing floor =
        cudabench::measureLaunchFloor(dim3(1), dim3(32), 100, 25, 250.0, gStream);
    cudabench::printLaunchFloor(floor);
    const size_t meterStep = cudabench::measureDriverMeterStep();
    std::printf("\n Driver meter step, measured in this run: %.2f MB. Every CROSS-LIBRARY\n"
                " memory reading below is taken over %d working sets for that reason;\n"
                " binCV-against-binCV figures use the allocation sum instead.\n\n",
                static_cast<double>(meterStep) / (1024.0 * 1024.0), kReplicas);

    // -----------------------------------------------------------------------
    // Inputs
    // -----------------------------------------------------------------------
    const Frames frames = loadFrames(frameDir);
    std::printf(" FRAMES: %s\n", frames.source.c_str());
    if (!frames.real)
        std::printf("   *** SYNTHETIC. Corner density and the spatial distribution of\n"
                    "   *** keypoints drive CASE B's admitted fraction, and a synthetic\n"
                    "   *** frame answers a different question. Pass a EuRoC frame\n"
                    "   *** directory as argv[1] for the numbers that decide.\n");

    bincv::BriefPattern<kBits> pattern;
    bincv::makeBriefPattern<kBits>(pattern);

    long long thrA = 0, thrB = 0, thrMapA = 0, thrMapB = 0;
    const std::vector<float> qxy =
        strongestCorners(frames.a, kW, kH, kFrontendCount, thrA);
    const std::vector<float> txy =
        strongestCorners(frames.b, kW, kH, kFrontendCount, thrB);
    const std::vector<float> qxyMap = strongestCorners(frames.a, kW, kH, kMapCount, thrMapA);
    const std::vector<float> txyMap = strongestCorners(frames.b, kW, kH, kMapCount, thrMapB);
    std::printf(" FAST thresholds used to reach the two counts: %lld / %lld at %zu, "
                "%lld / %lld at %zu\n\n",
                thrA, thrB, kFrontendCount, thrMapA, thrMapB, kMapCount);

    const std::vector<uint32_t> qd = briefFor(frames.a, kW, kH, qxy, pattern);
    const std::vector<uint32_t> td = briefFor(frames.b, kW, kH, txy, pattern);
    const std::vector<uint32_t> qdMap = briefFor(frames.a, kW, kH, qxyMap, pattern);
    const std::vector<uint32_t> tdMap = briefFor(frames.b, kW, kH, txyMap, pattern);
    const std::vector<int32_t> qo = octavesFor(kFrontendCount);
    const std::vector<int32_t> to = octavesFor(kFrontendCount);

    // Device copies.
    using bincv::cuda::DeviceArray;
    DeviceArray<uint32_t> dQd(qd.size()), dTd(td.size());
    DeviceArray<uint32_t> dQdMap(qdMap.size()), dTdMap(tdMap.size());
    DeviceArray<float> dQxy(qxy.size()), dTxy(txy.size());
    DeviceArray<float> dQxyMap(qxyMap.size()), dTxyMap(txyMap.size());
    DeviceArray<int32_t> dQo(qo.size()), dTo(to.size());
    DeviceArray<bincv::cuda::DeviceDescriptorMatch> dMatch(kFrontendCount);
    DeviceArray<bincv::cuda::DeviceDescriptorMatch> dMatchMap(kMapCount);
    const auto up = [](auto& dst, const auto& src) {
        cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(src[0]),
                   cudaMemcpyHostToDevice);
    };
    up(dQd, qd); up(dTd, td); up(dQdMap, qdMap); up(dTdMap, tdMap);
    up(dQxy, qxy); up(dTxy, txy); up(dQxyMap, qxyMap); up(dTxyMap, txyMap);
    up(dQo, qo); up(dTo, to);
    cudaDeviceSynchronize();

    const auto qSet = [&](bool map) {
        return bincv::cuda::descriptorSet(map ? dQdMap.data() : dQd.data(),
                                          map ? kMapCount : kFrontendCount, kWords);
    };
    const auto tSet = [&](bool map) {
        return bincv::cuda::descriptorSet(map ? dTdMap.data() : dTd.data(),
                                          map ? kMapCount : kFrontendCount, kWords);
    };
    const auto qPts = [&](bool withOctave) {
        return bincv::cuda::keypointSet(dQxy.data(), kFrontendCount,
                                        withOctave ? dQo.data() : nullptr);
    };
    const auto tPts = [&](bool withOctave) {
        return bincv::cuda::keypointSet(dTxy.data(), kFrontendCount,
                                        withOctave ? dTo.data() : nullptr);
    };

    // =======================================================================
    // CASE A / CASE E -- the matcher's own arms
    // =======================================================================
    rule();
    std::printf(" CASE A -- descriptor matching. binCV's two arms first, then the bar.\n");
    std::printf(" ORIENTATION: every paired ratio below is B/A. Arm A is the arm being\n"
                " compared against (the reference arm, or OpenCV); arm B is the one being\n"
                " priced. Under 1.00x means B is ahead.\n");
    rule();

    // The TILE WIDTH is measured here rather than inherited from another
    // kernel: three instantiations, timed against each other at map scale
    // where the kernel dominates the launch.
    std::printf("\n TILE WIDTH SWEEP (queries a block owns; the train set streams once\n"
                " for all of them). MEASURED, not inherited -- and measured at BOTH\n"
                " scales, because a wider tile trades blocks for reuse and the two\n"
                " points can disagree about which way that goes.\n");
    for (int map = 0; map < 2; ++map) {
        const bool m = map == 1;
        const size_t n = m ? kMapCount : kFrontendCount;
        auto* out = m ? dMatchMap.data() : dMatch.data();
        std::printf("   at %zu x %zu:\n", n, n);
        for (unsigned tile : {1u, 4u, 8u}) {
            bincv::cuda::impl::matchTiledArmEnabled() = true;
            bincv::cuda::impl::matchQueryTile() = tile;
            const Timing t = cudabench::timeKernel(
                [&] {
                    bincv::cuda::matchDescriptors(qSet(m), tSet(m), out, 80, gStream);
                },
                m ? 5 : 20, kRounds, gStream);
            char label[64];
            std::snprintf(label, sizeof(label), "binCV tiled, tile = %u", tile);
            cudabench::printArmVsFloor(label, t, floor, "kernel");
        }
    }
    bincv::cuda::impl::matchQueryTile() = bincv::cuda::impl::kMatchDefaultTile;
    std::printf("   The shipped default is tile = %u, and the lines above are why.\n",
                bincv::cuda::impl::kMatchDefaultTile);

    std::printf("\n THE OFF-SWITCH RATIO, both arms in this binary on the same inputs.\n");
    for (int map = 0; map < 2; ++map) {
        const bool m = map == 1;
        const size_t n = m ? kMapCount : kFrontendCount;
        auto* out = m ? dMatchMap.data() : dMatch.data();
        const PairedTiming p = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::impl::matchTiledArmEnabled() = false;
                bincv::cuda::matchDescriptors(qSet(m), tSet(m), out, 80, gStream);
            },
            [&] {
                bincv::cuda::impl::matchTiledArmEnabled() = true;
                bincv::cuda::matchDescriptors(qSet(m), tSet(m), out, 80, gStream);
            },
            m ? 3 : 20, m ? 3 : 20, kRounds, gStream);
        char a[64], b[64];
        std::snprintf(a, sizeof(a), "reference arm (thread per query), %zux%zu", n, n);
        std::snprintf(b, sizeof(b), "tiled arm, %zux%zu", n, n);
        cudabench::printPaired(a, b, p, "kernel");
    }
    bincv::cuda::impl::matchTiledArmEnabled() = true;

    // CASE E for the matcher: the fast path's OWN gate excludes a descriptor
    // wider than the staging accepts, so both switch positions run the same
    // kernel and the ratio must read ~1.00x.
    {
        const size_t wideWords = bincv::cuda::impl::kMatchTileMaxWords + 1;
        const size_t n = 470;
        DeviceArray<uint32_t> wq(n * wideWords), wt(n * wideWords);
        DeviceArray<bincv::cuda::DeviceDescriptorMatch> wOut(n);
        const auto q = bincv::cuda::descriptorSet(wq.data(), n, wideWords);
        const auto t = bincv::cuda::descriptorSet(wt.data(), n, wideWords);
        const PairedTiming p = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::impl::matchTiledArmEnabled() = false;
                bincv::cuda::matchDescriptors(q, t, wOut.data(), 80, gStream);
            },
            [&] {
                bincv::cuda::impl::matchTiledArmEnabled() = true;
                bincv::cuda::matchDescriptors(q, t, wOut.data(), 80, gStream);
            },
            20, 20, kRounds, gStream);
        std::printf("\n CASE E -- the matcher's gate-excluded control: a %zu-bit descriptor,\n"
                    " which the shared-memory staging refuses. Both switch positions run the\n"
                    " reference kernel, so this MUST read ~1.00x.\n", wideWords * 32);
        cudabench::printPaired("switch OFF, 544-bit descriptor",
                               "switch ON, 544-bit descriptor", p, "kernel", true);
        bincv::cuda::impl::matchTiledArmEnabled() = true;
    }

    // -----------------------------------------------------------------------
    // CASE A -- the role bar
    // -----------------------------------------------------------------------
#if BINCV_CUDA_SPARSE_OPENCV
    {
        std::printf("\n THE ROLE BAR: cv::cuda::DescriptorMatcher::createBFMatcher"
                    "(NORM_HAMMING),\n knnMatchAsync(k = 2) + the committed device ratio pass,"
                    " both on gStream.\n");
        auto bf = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        for (int map = 0; map < 2; ++map) {
            const bool m = map == 1;
            const int n = static_cast<int>(m ? kMapCount : kFrontendCount);
            const std::vector<uint32_t>& hq = m ? qdMap : qd;
            const std::vector<uint32_t>& ht = m ? tdMap : td;

            // CV_8U: what cv::cuda::ORB emits, and the path whose Hamming
            // reduction popcounts one BYTE at a time.
            cv::cuda::GpuMat q8(n, static_cast<int>(kBits / 8), CV_8UC1);
            cv::cuda::GpuMat t8(n, static_cast<int>(kBits / 8), CV_8UC1);
            q8.upload(cv::Mat(n, static_cast<int>(kBits / 8), CV_8UC1,
                              const_cast<uint32_t*>(hq.data())));
            t8.upload(cv::Mat(n, static_cast<int>(kBits / 8), CV_8UC1,
                              const_cast<uint32_t*>(ht.data())));
            // CV_32S: the IDENTICAL bytes, reinterpreted, where OpenCV
            // dispatches matchHamming_gpu<int> and issues eight popcounts.
            cv::cuda::GpuMat q32(n, static_cast<int>(kWords), CV_32SC1);
            cv::cuda::GpuMat t32(n, static_cast<int>(kWords), CV_32SC1);
            q32.upload(cv::Mat(n, static_cast<int>(kWords), CV_32SC1,
                               const_cast<uint32_t*>(hq.data())));
            t32.upload(cv::Mat(n, static_cast<int>(kWords), CV_32SC1,
                               const_cast<uint32_t*>(ht.data())));

            cv::cuda::GpuMat matches8, matches32;
            DeviceArray<uint32_t> cvOut(static_cast<size_t>(n) * 4);
            auto* out = m ? dMatchMap.data() : dMatch.data();
            cudaDeviceSynchronize();

            const auto ocvArm = [&](cv::cuda::GpuMat& qm, cv::cuda::GpuMat& tm,
                                    cv::cuda::GpuMat& res) {
                bf->knnMatchAsync(qm, tm, res, 2, cv::noArray(), cvStream);
                launchBfRatioPass(res.ptr(0), res.ptr(1), n, 80, cvOut.data(), gStream);
            };
            const auto bincvArm = [&](bool mm) {
                bincv::cuda::matchDescriptors(qSet(mm), tSet(mm), out, 80, gStream);
            };

            const int iters = m ? 3 : 20;
            PairedTiming p8 = cudabench::timeKernelPaired(
                [&] { ocvArm(q8, t8, matches8); }, [&] { bincvArm(m); }, iters, iters,
                kRounds, gStream);
            PairedTiming p32 = cudabench::timeKernelPaired(
                [&] { ocvArm(q32, t32, matches32); }, [&] { bincvArm(m); }, iters, iters,
                kRounds, gStream);
            std::printf("\n  %d x %d descriptors, %zu-bit:\n", n, n, kBits);
            cudabench::printPaired("cv::cuda BFMatcher, CV_8U + ratio pass",
                                   "binCV matchDescriptors", p8, "kernel");
            cudabench::printPaired("cv::cuda BFMatcher, CV_32S + ratio pass",
                                   "binCV matchDescriptors", p32, "kernel");
            std::printf("   CV_8U / CV_32S on OpenCV's own side: %.2fx -- the instruction\n"
                        "   ratio the byte-wise HammingDist costs a caller who hands it\n"
                        "   what cv::cuda::ORB produces.\n",
                        p32.a.medianMs > 0.0 ? p8.a.medianMs / p32.a.medianMs : 0.0);
        }

        // Memory, the cross-library figure, meter 2 on both sides.
        // Its OWN replica count, an order above kReplicas: one matcher working
        // set is ~37 KB, so 32 of them do not clear a single 2 MB driver unit
        // and the reading would be the meter's resolution rather than a
        // footprint. 256 of them clear several.
        constexpr int kMatchReplicas = 256;
        std::printf("\n MEMORY, CASE A, meter 2 on BOTH sides, %d replicas of the %zu x %zu\n"
                    " working set -- an order more than elsewhere, because one set is ~37 KB\n"
                    " and 32 of them would not clear one 2 MB driver unit.\n",
                    kMatchReplicas, kFrontendCount, kFrontendCount);
        const size_t descBytes = kFrontendCount * kWords * 4;
        {
            cudabench::DeviceMemMeter meter;
            std::vector<DeviceArray<uint32_t>> qs, ts;
            std::vector<DeviceArray<bincv::cuda::DeviceDescriptorMatch>> os;
            for (int i = 0; i < kMatchReplicas; ++i) {
                qs.emplace_back(kFrontendCount * kWords);
                ts.emplace_back(kFrontendCount * kWords);
                os.emplace_back(kFrontendCount);
            }
            const size_t used = meter.deltaBytes();
            cudabench::printDriverDelta("binCV ungated, per working set",
                                        used / kMatchReplicas, meterStep);
        }
        {
            cudabench::DeviceMemMeter meter;
            std::vector<cv::cuda::GpuMat> qs, ts, ms;
            auto bf2 = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
            for (int i = 0; i < kMatchReplicas; ++i) {
                qs.emplace_back(static_cast<int>(kFrontendCount),
                                static_cast<int>(kBits / 8), CV_8UC1);
                ts.emplace_back(static_cast<int>(kFrontendCount),
                                static_cast<int>(kBits / 8), CV_8UC1);
                ms.emplace_back();
                bf2->knnMatchAsync(qs.back(), ts.back(), ms.back(), 2, cv::noArray(),
                                   cvStream);
            }
            cvStream.waitForCompletion();
            const size_t used = meter.deltaBytes();
            cudabench::printDriverDelta("cv::cuda BFMatcher, per working set",
                                        used / kMatchReplicas, meterStep);
            std::printf("   OpenCV's reading is an UPPER bound: GpuMat may pool, and the\n"
                        "   metered scope includes one knnMatchAsync so the match buffer it\n"
                        "   creates is counted -- as binCV's record array is.\n"
                        "   THE GAP IS THE ROW PITCH, and that part is arithmetic rather than\n"
                        "   allocator behaviour: a %zu-bit descriptor is a %zu-BYTE ROW, and\n"
                        "   GpuMat pitches a row to a 512-byte multiple, so OpenCV's two\n"
                        "   descriptor matrices carry ~16x their own data. A packed device\n"
                        "   array does not. The design this family came from computed both\n"
                        "   sides UNPITCHED and predicted a wash; the measurement says\n"
                        "   otherwise, and the prediction is what was wrong.\n",
                        kBits, kBits / 8);
        }
        cudabench::printAllocSum("binCV descriptors + records (meter 1)",
                                 2 * descBytes + kFrontendCount * 16);
    }
#else
    std::printf("\n CASE A ROLE BAR: BLOCKED -- no cv::cuda::DescriptorMatcher compiled in.\n"
                " No substitute is quoted, and a CPU number is not a GPU bar.\n");
#endif

    // =======================================================================
    // CASE B -- the gate question
    // =======================================================================
    rule();
    std::printf(" CASE B -- is the gate still the right answer on device?\n");
    rule();
    {
        // The admitted fraction, computed rather than guessed, because it is
        // what the gate's whole claim rests on.
        size_t admitted = 0;
        for (size_t q = 0; q < kFrontendCount; ++q)
            for (size_t t = 0; t < kFrontendCount; ++t) {
                const float dx = txy[2 * t] - qxy[2 * q];
                const float dy = txy[2 * t + 1] - qxy[2 * q + 1];
                if (dx > 48.0f || dx < -48.0f || dy > 48.0f || dy < -48.0f) continue;
                const int od = qo[q] - to[t];
                if (od > 1 || od < -1) continue;
                ++admitted;
            }
        std::printf("\n Admitted fraction at window 48 px, octave band 1, on these\n"
                    " keypoints: %.2f%% of %zu candidate pairs. That is the only work the\n"
                    " gate can save, and it is the number the speed side of this case is\n"
                    " a test of.\n",
                    100.0 * static_cast<double>(admitted) /
                        static_cast<double>(kFrontendCount * kFrontendCount),
                    kFrontendCount * kFrontendCount);

        const PairedTiming p = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::matchDescriptors(qSet(false), tSet(false), dMatch.data(), 75,
                                              gStream);
            },
            [&] {
                bincv::cuda::matchDescriptorsGated(qSet(false), qPts(true), tSet(false),
                                                   tPts(true), 48.0f, 48.0f, dMatch.data(),
                                                   75, 1, gStream);
            },
            20, 20, kRounds, gStream);
        cudabench::printPaired("device brute force, 470x470",
                               "device GATED (48 px, octave band 1)", p, "kernel");

        // The host arms, on the same inputs. INDICATIVE: this host is not
        // timing-grade under WSL2, and the spread printed beside each number is
        // what says so.
        std::vector<bincv::DescriptorMatch> hostOut(kFrontendCount);
        std::vector<measure::Bench> hb;
        hb.push_back({"host brute force", [&](int) {
                          bincv::matchDescriptors<uint32_t>(qd.data(), kFrontendCount,
                                                            td.data(), kFrontendCount, kWords,
                                                            hostOut.data(), 75);
                      }});
        hb.push_back({"host GATED (48 px, octave band 1)", [&](int) {
                          bincv::matchDescriptorsGated<uint32_t>(
                              qd.data(), qxy.data(), kFrontendCount, td.data(), txy.data(),
                              kFrontendCount, kWords, 48.0f, 48.0f, hostOut.data(), 75,
                              qo.data(), to.data(), 1);
                      }});
        const std::vector<measure::Timing> ht =
            measure::measureInterleaved(hb, 9, 60.0);
        std::printf("\n The same two on the HOST, same inputs, same ratio:\n");
        for (size_t i = 0; i < hb.size(); ++i) printHostArm(hb[i].name.c_str(), ht[i]);
        std::printf("   host gated / host brute: %.2fx. The host numbers are INDICATIVE:\n"
                    "   this x86 host is not timing-grade under WSL2 and the spread says so.\n"
                    "   They are here because the gate exists to save work a CPU pays for,\n"
                    "   and that is the comparison the device one has to be read against.\n",
                    ht[0].medianNs > 0.0 ? ht[1].medianNs / ht[0].medianNs : 0.0);

        std::printf("\n MEMORY, arithmetic and not a measurement: the gated arm carries\n"
                    "   queryXY + trainXY = %zu B and the two octave arrays = %zu B that the\n"
                    "   ungated arm does not. Under 'memory wins' that counts against the\n"
                    "   gate on device.\n",
                    2 * kFrontendCount * 8, 2 * kFrontendCount * 4);
    }

    // =======================================================================
    // CASE C -- sparse stereo
    // =======================================================================
    rule();
    std::printf(" CASE C -- sparse rectified stereo, against binCV's own dense device\n"
                " path (0.39 ms / 442 KB, docs/reports/cuda.md). No cv::cuda sparse\n"
                " stereo API exists to compare against.\n");
    rule();
    {
        // A real frame against itself shifted by a KNOWN disparity: real image
        // content with ground truth. Labelled as that, not as a stereo pair.
        std::vector<uint8_t> rightWide(kW * kH, 0);
        for (size_t y = 0; y < kH; ++y)
            for (size_t x = 0; x + static_cast<size_t>(kStereoShift) < kW; ++x)
                rightWide[y * kW + x] = frames.a[y * kW + x + static_cast<size_t>(kStereoShift)];

        long long thrL = 0;
        // A margin that covers the BRIEF patch reach AND the disparity, so both
        // sides of every pair have a real descriptor.
        std::vector<float> lxy =
            strongestCorners(frames.a, kW, kH, kStereoKeypoints, thrL, 24 + kStereoShift);
        // Right keypoints at the true disparity, so the coarse stage has a
        // correct candidate to find and the refinement has something to sharpen.
        std::vector<float> rxy(lxy.size());
        for (size_t i = 0; i < kStereoKeypoints; ++i) {
            rxy[2 * i] = lxy[2 * i] - static_cast<float>(kStereoShift);
            rxy[2 * i + 1] = lxy[2 * i + 1];
        }
        const std::vector<uint32_t> ld = briefFor(frames.a, kW, kH, lxy, pattern);
        const std::vector<uint32_t> rd = briefFor(rightWide, kW, kH, rxy, pattern);

        const bincv::BinMat<uint32_t> leftBits = packOf(frames.a, kW, kH);
        const bincv::BinMat<uint32_t> rightBits = packOf(rightWide, kW, kH);

        bincv::cuda::DeviceBinMat dLeft(kW, kH), dRight(kW, kH);
        bincv::cuda::upload<uint32_t>(leftBits.constView(), dLeft.view());
        bincv::cuda::upload<uint32_t>(rightBits.constView(), dRight.view());
        DeviceArray<float> dLxy(lxy.size()), dRxy(rxy.size());
        DeviceArray<uint32_t> dLd(ld.size()), dRd(rd.size());
        DeviceArray<bincv::cuda::DeviceStereoMatch> dStereo(kStereoKeypoints);
        up(dLxy, lxy); up(dRxy, rxy); up(dLd, ld); up(dRd, rd);
        cudaDeviceSynchronize();

        bincv::StereoMatchParams sp;
        const auto lSet = bincv::cuda::keypointSet(dLxy.data(), kStereoKeypoints);
        const auto rSet = bincv::cuda::keypointSet(dRxy.data(), kStereoKeypoints);
        const auto lDesc = bincv::cuda::descriptorSet(dLd.data(), kStereoKeypoints, kWords);
        const auto rDesc = bincv::cuda::descriptorSet(dRd.data(), kStereoKeypoints, kWords);

        for (int stage = 0; stage < 3; ++stage) {
            const char* names[3] = {"coarse descriptor search", "window refinement",
                                    "BOTH STAGES (what a caller runs)"};
            const auto body = [&] {
                if (stage != 1)
                    bincv::cuda::stereoDescriptorMatch(lSet, lDesc, rSet, rDesc,
                                                       dStereo.data(), sp, gStream);
                if (stage != 0)
                    bincv::cuda::stereoRefineDisparity(dLeft.constView(), dRight.constView(),
                                                       lSet, dStereo.data(), sp, gStream);
            };
            const PairedTiming p = cudabench::timeKernelPaired(
                [&] { bincv::cuda::impl::sparseStereoFastArmEnabled() = false; body(); },
                [&] { bincv::cuda::impl::sparseStereoFastArmEnabled() = true; body(); },
                20, 20, kRounds, gStream);
            char a[80], b[80];
            std::snprintf(a, sizeof(a), "reference arm -- %s", names[stage]);
            std::snprintf(b, sizeof(b), "fast arm      -- %s", names[stage]);
            std::printf("\n");
            cudabench::printPaired(a, b, p, "kernel");
            if (stage == 2)
                std::printf("   CASE C's ship condition is on this line: below 0.39 ms is a\n"
                            "   pass on speed against dense-then-index.\n");
        }
        bincv::cuda::impl::sparseStereoFastArmEnabled() = true;

        // CASE E for the window arms: a window the span bound rejects.
        {
            bincv::StereoMatchParams wide;
            wide.winWidth = 41;
            wide.winHeight = 41;
            wide.refineRadius = 8;
            const PairedTiming p = cudabench::timeKernelPaired(
                [&] {
                    bincv::cuda::impl::sparseStereoFastArmEnabled() = false;
                    bincv::cuda::stereoRefineDisparity(dLeft.constView(), dRight.constView(),
                                                       lSet, dStereo.data(), wide, gStream);
                },
                [&] {
                    bincv::cuda::impl::sparseStereoFastArmEnabled() = true;
                    bincv::cuda::stereoRefineDisparity(dLeft.constView(), dRight.constView(),
                                                       lSet, dStereo.data(), wide, gStream);
                },
                10, 10, kRounds, gStream);
            std::printf("\n CASE E -- refinement's gate-excluded control: a 41x41 window with\n"
                        " refineRadius 8, which windowSpanFits rejects. MUST read ~1.00x.\n");
            cudabench::printPaired("switch OFF, 41x41 window", "switch ON, 41x41 window", p,
                                   "kernel", true);
            bincv::cuda::impl::sparseStereoFastArmEnabled() = true;
        }

        // Accuracy, PUBLISHED and not decided on: the sub-pixel residual
        // against the pair's known disparity.
        {
            bincv::cuda::stereoDescriptorMatch(lSet, lDesc, rSet, rDesc, dStereo.data(), sp,
                                               gStream);
            bincv::cuda::stereoRefineDisparity(dLeft.constView(), dRight.constView(), lSet,
                                               dStereo.data(), sp, gStream);
            cudaStreamSynchronize(gStream);
            std::vector<bincv::cuda::DeviceStereoMatch> got(kStereoKeypoints);
            cudaMemcpy(got.data(), dStereo.data(),
                       kStereoKeypoints * sizeof(bincv::cuda::DeviceStereoMatch),
                       cudaMemcpyDeviceToHost);
            double sum = 0.0, worst = 0.0;
            size_t valid = 0, pinned = 0, coarseExact = 0;
            for (const auto& m : got) {
                if (m.valid == 0u) continue;
                const double e = m.disparity - static_cast<double>(kStereoShift);
                const double ae = e < 0.0 ? -e : e;
                sum += ae;
                if (ae > worst) worst = ae;
                // The scan runs [d0 - R, d0 + R] around the coarse disparity,
                // and the coarse stage gets d0 exactly right here (the right
                // descriptors ARE the left ones, by construction). So a result
                // sitting at the scan's edge is one whose cost surface gave the
                // search nothing to find.
                if (ae >= static_cast<double>(sp.refineRadius) - 0.5001) ++pinned;
                if (ae < 0.5001) ++coarseExact;
                ++valid;
            }
            std::printf("\n ACCURACY, published beside the numbers and NOT decided on:\n"
                        "   %zu of %zu keypoints matched; mean |disparity - %d| = %.4f px,\n"
                        "   worst %.4f px, against a pair whose true disparity is exactly %d.\n"
                        "   %zu landed within half a pixel; %zu landed at the SCAN'S EDGE\n"
                        "   (|error| >= refineRadius = %d).\n"
                        "   THE EDGE COUNT IS THE INTERESTING NUMBER AND IT IS NOT A DEVICE\n"
                        "   RESULT: the suite holds this kernel bit-exact against the host,\n"
                        "   so whatever it does here the host does too. It is a property of a\n"
                        "   ONE-BIT window refinement -- the frames are packed by a single\n"
                        "   threshold, and where the local binary window is uniform EVERY\n"
                        "   disparity scores zero, so the tie rule takes the smallest and the\n"
                        "   result pins to the scan's low edge. A FAST corner in the wide\n"
                        "   image is not a corner in the thresholded one. Reported rather\n"
                        "   than tuned away: the fix is a richer packing (the N-bit or census\n"
                        "   path), not a change to this kernel.\n"
                        "   The dense path this is measured against returns an INTEGER map,\n"
                        "   so the sub-pixel output is the reason to run the sparse arm at\n"
                        "   all. What residual would make it not worth running is STOP AND\n"
                        "   ASK 2.\n",
                        valid, kStereoKeypoints, kStereoShift,
                        valid ? sum / static_cast<double>(valid) : 0.0, worst, kStereoShift,
                        coarseExact, pinned, sp.refineRadius);
        }

        // Memory: allocation sum, binCV against binCV.
        const size_t planeBytes = dLeft.getAlignedWidth() * dLeft.getHeight() * 4;
        const size_t set = 2 * planeBytes + 2 * kStereoKeypoints * 8 +
                           2 * kStereoKeypoints * kWords * 4 + kStereoKeypoints * 16;
        std::printf("\n MEMORY, CASE C. binCV against binCV, so METER 1 (allocation sum):\n");
        cudabench::printAllocSum("sparse: two planes + points + descriptors + records", set);
        cudabench::printAllocSum("dense-then-index (recorded, cuda.md)", 442u * 1024u);
        std::printf("   ratio %.2fx in the sparse arm's favour. Both are allocation sums\n"
                    "   of binCV's own arrays, which is the one comparison meter 1 is for;\n"
                    "   nothing here is divided by a driver reading.\n"
                    "   Scratch: NONE. Both stereo stages reduce in registers and warp\n"
                    "   shuffles, which is this backend's spelling of the host's\n"
                    "   running-minimum rule.\n",
                    static_cast<double>(442u * 1024u) / static_cast<double>(set));
    }

    // =======================================================================
    // CASE D -- block matching
    // =======================================================================
    rule();
    std::printf(" CASE D -- block matching (route a) against"
                " cv::cuda::SparsePyrLKOpticalFlow.\n ROLE ONLY: route (b)'s algorithm on"
                " bytes against route (a)'s on bits.\n");
    rule();
    {
        constexpr size_t kLevels = 4;
        std::vector<std::vector<uint8_t>> wa, wb;
        wa.push_back(frames.a);
        wb.push_back(frames.b);
        std::vector<size_t> lw{kW}, lh{kH};
        for (size_t i = 1; i < kLevels; ++i) {
            size_t dw = 0, dh = 0;
            wa.push_back(halve(wa.back(), lw.back(), lh.back(), dw, dh));
            wb.push_back(halve(wb.back(), lw.back(), lh.back(), dw, dh));
            lw.push_back(dw);
            lh.push_back(dh);
        }
        std::vector<bincv::BinMat<uint32_t>> pa, pb;
        std::vector<bincv::cuda::DeviceBinMat> da, db;
        for (size_t i = 0; i < kLevels; ++i) {
            pa.push_back(packOf(wa[i], lw[i], lh[i]));
            pb.push_back(packOf(wb[i], lw[i], lh[i]));
        }
        for (size_t i = 0; i < kLevels; ++i) {
            da.emplace_back(static_cast<int>(lw[i]), static_cast<int>(lh[i]));
            db.emplace_back(static_cast<int>(lw[i]), static_cast<int>(lh[i]));
            bincv::cuda::upload<uint32_t>(pa[i].constView(), da[i].view());
            bincv::cuda::upload<uint32_t>(pb[i].constView(), db[i].view());
        }
        std::vector<bincv::cuda::DeviceBlockMatchLevel> levels(kLevels);
        for (size_t i = 0; i < kLevels; ++i) {
            levels[i].prev = da[i].constView();
            levels[i].next = db[i].constView();
        }

        long long thrT = 0;
        const std::vector<float> pxy = strongestCorners(frames.a, kW, kH, kStereoKeypoints,
                                                        thrT);
        std::vector<bincv::Point2f> pts(kStereoKeypoints);
        for (size_t i = 0; i < kStereoKeypoints; ++i) {
            pts[i].x = pxy[2 * i];
            pts[i].y = pxy[2 * i + 1];
        }
        DeviceArray<bincv::Point2f> dPrev(kStereoKeypoints), dNext(kStereoKeypoints);
        DeviceArray<uint8_t> dStatus(kStereoKeypoints);
        DeviceArray<uint8_t> dScratch(bincv::cuda::blockMatchScratchBytes(kStereoKeypoints));
        cudaMemcpy(dPrev.data(), pts.data(), kStereoKeypoints * sizeof(bincv::Point2f),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        bincv::BlockMatchParams bp;
        const auto bmBody = [&] {
            bincv::cuda::calcOpticalFlowBlockMatch(levels.data(), kLevels, dPrev.data(),
                                                   dNext.data(), dStatus.data(),
                                                   kStereoKeypoints, dScratch.data(),
                                                   dScratch.size(), bp, gStream);
        };
        const PairedTiming p = cudabench::timeKernelPaired(
            [&] { bincv::cuda::impl::blockMatchFastArmEnabled() = false; bmBody(); },
            [&] { bincv::cuda::impl::blockMatchFastArmEnabled() = true; bmBody(); }, 3, 10,
            kRounds, gStream);
        std::printf("\n");
        cudabench::printPaired("reference arm (thread per keypoint, one launch)",
                               "fast arm (warp per keypoint, one launch per level)", p,
                               "kernel");
        bincv::cuda::impl::blockMatchFastArmEnabled() = true;

        // CASE E for block matching.
        {
            bincv::BlockMatchParams wide;
            wide.winWidth = 48;
            wide.winHeight = 48;
            const PairedTiming pw = cudabench::timeKernelPaired(
                [&] {
                    bincv::cuda::impl::blockMatchFastArmEnabled() = false;
                    bincv::cuda::calcOpticalFlowBlockMatch(
                        levels.data(), kLevels, dPrev.data(), dNext.data(), dStatus.data(),
                        kStereoKeypoints, dScratch.data(), dScratch.size(), wide, gStream);
                },
                [&] {
                    bincv::cuda::impl::blockMatchFastArmEnabled() = true;
                    bincv::cuda::calcOpticalFlowBlockMatch(
                        levels.data(), kLevels, dPrev.data(), dNext.data(), dStatus.data(),
                        kStereoKeypoints, dScratch.data(), dScratch.size(), wide, gStream);
                },
                3, 3, kRounds, gStream);
            std::printf("\n CASE E -- block matching's gate-excluded control: a 48x48 window,\n"
                        " which windowSpanFits rejects. MUST read ~1.00x.\n");
            cudabench::printPaired("switch OFF, 48x48 window", "switch ON, 48x48 window", pw,
                                   "kernel", true);
            bincv::cuda::impl::blockMatchFastArmEnabled() = true;
        }

        // The COMPUTED footprint ratio -- arithmetic from the dimensions the
        // allocations use, with no threshold to choose.
        size_t binWorking = 0;
        for (size_t i = 0; i < kLevels; ++i)
            binWorking += 2 * da[i].getAlignedWidth() * da[i].getHeight() * 4;
        binWorking += 2 * kStereoKeypoints * sizeof(bincv::Point2f) + kStereoKeypoints +
                      bincv::cuda::blockMatchScratchBytes(kStereoKeypoints);
        size_t lkWorking = 0;
        for (size_t i = 0; i < kLevels; ++i) lkWorking += 2 * lw[i] * lh[i];
        lkWorking += 2 * kStereoKeypoints * sizeof(float) * 2 + 2 * kStereoKeypoints * 4;
        std::printf("\n MEMORY, CASE D.\n");
        cudabench::printAllocSum("binCV: two binary pyramids + pts + scratch", binWorking);
        cudabench::printAllocSum("LK: two 8-bit pyramids + pts + status + err (computed)",
                                 lkWorking);
        std::printf("   COMPUTED ratio %.2fx. This is the number the ship condition is\n"
                    "   about: the MEASURED cudaMemGetInfo ratio below has to reproduce it\n"
                    "   within the meter's step, or the footprint claim does not rest on\n"
                    "   the format and THAT is the finding.\n"
                    "   The computed LK figure is a LOWER bound (GpuMat is pitched) and the\n"
                    "   measured one is an UPPER bound (GpuMat may pool): the two caveats\n"
                    "   push opposite ways and both are stated.\n",
                    static_cast<double>(lkWorking) / static_cast<double>(binWorking));

#if BINCV_CUDA_SPARSE_OPENCV
        {
            auto lk = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(31, 31), 3, 30, false);
            cv::cuda::GpuMat gPrev(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
            cv::cuda::GpuMat gNext(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
            gPrev.upload(cv::Mat(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                                 const_cast<uint8_t*>(frames.a.data())));
            gNext.upload(cv::Mat(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                                 const_cast<uint8_t*>(frames.b.data())));
            cv::Mat hostPts(1, static_cast<int>(kStereoKeypoints), CV_32FC2);
            for (size_t i = 0; i < kStereoKeypoints; ++i)
                hostPts.at<cv::Vec2f>(0, static_cast<int>(i)) =
                    cv::Vec2f(pts[i].x, pts[i].y);
            cv::cuda::GpuMat gPts, gNextPts, gStatus, gErr;
            gPts.upload(hostPts);
            cudaDeviceSynchronize();

            const PairedTiming pd = cudabench::timeKernelPaired(
                [&] { lk->calc(gPrev, gNext, gPts, gNextPts, gStatus, gErr, cvStream); },
                [&] { bmBody(); }, 3, 10, kRounds, gStream);
            std::printf("\n");
            cudabench::printPaired("cv::cuda::SparsePyrLKOpticalFlow (31x31, 3 levels)",
                                   "binCV block match (31x31, 4 levels, R=2)", pd, "kernel");

            // Tracking yield, PUBLISHED and never omitted: a speed or footprint
            // win for a tracker that tracks worse is not a win.
            cv::Mat st;
            gStatus.download(st);
            size_t lkTracked = 0;
            for (int i = 0; i < st.cols; ++i)
                if (st.at<uint8_t>(0, i) != 0) ++lkTracked;
            std::vector<uint8_t> binSt(kStereoKeypoints);
            bmBody();
            cudaStreamSynchronize(gStream);
            cudaMemcpy(binSt.data(), dStatus.data(), kStereoKeypoints,
                       cudaMemcpyDeviceToHost);
            size_t binTracked = 0;
            for (uint8_t v : binSt) binTracked += v != 0 ? 1u : 0u;
            std::printf("   TRACKING YIELD, published and never omitted: cv::cuda LK reports\n"
                        "   %zu of %zu tracked; binCV route (a) reports %zu. These are\n"
                        "   DIFFERENT ALGORITHMS and a status byte is not an accuracy\n"
                        "   measurement -- route (a)'s derived integer floor is 0.2887 px per\n"
                        "   axis with sub-pixel off. The floor that would decide Case D is\n"
                        "   STOP AND ASK 1 and is not filled in here.\n",
                        lkTracked, kStereoKeypoints, binTracked);

            // The MEASURED footprint, meter 2 on both sides, replicated.
            size_t binDelta = 0, lkDelta = 0;
            {
                cudabench::DeviceMemMeter meter;
                std::vector<bincv::cuda::DeviceBinMat> ra, rb;
                std::vector<DeviceArray<bincv::Point2f>> rp, rn;
                std::vector<DeviceArray<uint8_t>> rs, rc;
                for (int r = 0; r < kReplicas; ++r) {
                    for (size_t i = 0; i < kLevels; ++i) {
                        ra.emplace_back(static_cast<int>(lw[i]), static_cast<int>(lh[i]));
                        rb.emplace_back(static_cast<int>(lw[i]), static_cast<int>(lh[i]));
                    }
                    rp.emplace_back(kStereoKeypoints);
                    rn.emplace_back(kStereoKeypoints);
                    rs.emplace_back(kStereoKeypoints);
                    rc.emplace_back(bincv::cuda::blockMatchScratchBytes(kStereoKeypoints));
                }
                binDelta = meter.deltaBytes() / kReplicas;
            }
            {
                cudabench::DeviceMemMeter meter;
                std::vector<cv::cuda::GpuMat> gp, gn, gq, gnp, gs, ge;
                std::vector<cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>> lks;
                for (int r = 0; r < kReplicas; ++r) {
                    gp.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                    gn.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                    gq.emplace_back();
                    gq.back().upload(hostPts);
                    gnp.emplace_back();
                    gs.emplace_back();
                    ge.emplace_back();
                    lks.push_back(
                        cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(31, 31), 3, 30,
                                                                 false));
                    lks.back()->calc(gp.back(), gn.back(), gq.back(), gnp.back(), gs.back(),
                                     ge.back(), cvStream);
                }
                cvStream.waitForCompletion();
                lkDelta = meter.deltaBytes() / kReplicas;
            }
            cudabench::printDriverDelta("binCV working set, per replica", binDelta, meterStep);
            cudabench::printDriverDelta("cv::cuda LK working set, per replica", lkDelta,
                                        meterStep);
            std::printf("   MEASURED ratio %.2fx against a COMPUTED %.2fx. The ship\n"
                        "   condition is that these agree within the meter's step; if they\n"
                        "   do not, the footprint claim does not rest on the format.\n",
                        binDelta ? static_cast<double>(lkDelta) / static_cast<double>(binDelta)
                                 : 0.0,
                        static_cast<double>(lkWorking) / static_cast<double>(binWorking));
        }
#else
        std::printf("\n CASE D ROLE BAR: BLOCKED -- no cv::cuda::SparsePyrLKOpticalFlow\n"
                    " compiled in. The footprint arithmetic above stands on its own; the\n"
                    " speed verdict does not, and no substitute bar is invented.\n");
#endif
    }

    rule();
    std::printf(" READ THE SPREADS. Every number here is KERNEL-RESIDENT and covers ONE\n"
                " launch (block matching: one launch per level). A microbenchmark ratio is\n"
                " not an end-to-end result, and no arm here is a share of any pipeline,\n"
                " because no resident device frontend calls these kernels yet.\n");
    rule();

    cudaStreamDestroy(gStream);
    return 0;
}
