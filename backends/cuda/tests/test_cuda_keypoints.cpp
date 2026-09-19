// The detector-to-keypoint-set link, and the CONTRACT THE RESIDENT PIPELINE
// RESTS ON.
//
// `cuda::keypointsFromCorners` is a small kernel, and most of this suite is not
// about the conversion arithmetic -- that is an int-to-float cast and it is
// exact by construction. What is worth testing is the part a caller's
// correctness actually depends on:
//
//  1. The count is read FROM DEVICE MEMORY and clamped. An append counter is
//     unclamped by contract (it reports the true count so a re-run can be
//     sized), so a truncated detection hands this op a count larger than its
//     buffer, and an unclamped read would index past the corner array.
//  2. Slots from the count to the capacity are written (0, 0). Without that
//     they hold the PREVIOUS frame's keypoints, and a consumer sized by
//     capacity would describe last frame's corners at this frame's positions --
//     a plausible wrong answer, which is the kind this project treats as worse
//     than a crash.
//  3. **THE PADDING SLOTS MUST BE REJECTED BY THE CONSUMERS.** The whole reason
//     the resident frontend can enqueue a frame with no mid-pipeline
//     synchronize is that orientation and BRIEF may be launched over `capacity`
//     keypoints rather than `count`, because a keypoint at (0, 0) fails every
//     bounding-box test. That is an assumption about OTHER headers' kernels, so
//     it is checked against them here rather than asserted in a comment.
//  4. The whole chain, in situ: goodFeaturesToTrack -> keypointsFromCorners,
//     with the xy array compared against the corner records the same launch
//     produced.
//
// A .cpp rather than a .cu: everything this needs is a host-callable launcher,
// and ops/fast.hpp and ops/medianWide.hpp gate their AVX2 kernels off under
// __CUDACC__ -- so a host arm compiled by nvcc is not the host arm a caller
// runs. Same reason test_cuda_frontend_corner.cpp gives.
//
// Exits 77 when no CUDA device is present, like the other suites here.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/cuda/compaction.hpp"
#include "bincv/cuda/corner.hpp"
#include "bincv/cuda/derivative.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/keypoints.hpp"
#include "bincv/cuda/orientation.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/corner.hpp"
#include "test_util.hpp"

namespace {

namespace bc = bincv::cuda;

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// A corner set with positions spread over a frame-sized range.
std::vector<bc::DeviceCorner> randomCorners(size_t n, uint64_t seed) {
    std::vector<bc::DeviceCorner> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i].x = static_cast<int>(splitmix(seed) % 752u);
        v[i].y = static_cast<int>(splitmix(seed) % 480u);
        v[i].response = static_cast<float>(splitmix(seed) % 1000u) * 0.125f;
    }
    return v;
}

/// The conversion, on the host, in the most obvious possible spelling. This is
/// the oracle; the kernel is what is on trial.
std::vector<float> hostConversion(const std::vector<bc::DeviceCorner>& corners,
                                  uint32_t count, uint32_t capacity) {
    std::vector<float> xy(2 * static_cast<size_t>(capacity), -1.0f);
    const uint32_t live = count < capacity ? count : capacity;
    for (uint32_t i = 0; i < capacity; ++i) {
        xy[2 * i] = i < live ? static_cast<float>(corners[i].x) : 0.0f;
        xy[2 * i + 1] = i < live ? static_cast<float>(corners[i].y) : 0.0f;
    }
    return xy;
}

/// Runs one conversion on the device and returns the downloaded `xy`.
/// `prefill` is what the destination held before the call, so a test can prove
/// the kernel overwrote it rather than merely agreeing with it.
std::vector<float> runConversion(const std::vector<bc::DeviceCorner>& corners,
                                 uint32_t count, uint32_t capacity, float prefill) {
    bc::DeviceArray<bc::DeviceCorner> dCorners(corners.size());
    bc::DeviceArray<uint32_t> dCount(1);
    bc::DeviceArray<float> dXY(2 * static_cast<size_t>(capacity));
    std::vector<float> seeded(2 * static_cast<size_t>(capacity), prefill);

    BINCV_CUDA_CHECK(cudaMemcpy(dCorners.data(), corners.data(),
                                corners.size() * sizeof(bc::DeviceCorner),
                                cudaMemcpyHostToDevice));
    BINCV_CUDA_CHECK(
        cudaMemcpy(dCount.data(), &count, sizeof(uint32_t), cudaMemcpyHostToDevice));
    BINCV_CUDA_CHECK(cudaMemcpy(dXY.data(), seeded.data(), seeded.size() * sizeof(float),
                                cudaMemcpyHostToDevice));

    BINCV_CHECK_EQ(bc::keypointsFromCorners(dCorners.data(), dCount.data(), dXY.data(),
                                            capacity),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> out(2 * static_cast<size_t>(capacity), -2.0f);
    BINCV_CUDA_CHECK(cudaMemcpy(out.data(), dXY.data(), out.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
    return out;
}

} // namespace

BINCV_TEST(CudaKeypointsFromCorners, MatchesTheHostConversionAtEveryCount) {
    // Capacities chosen to straddle the block size (128) in both directions, so
    // a partial last block and a single-thread launch are both exercised.
    const uint32_t capacities[] = {1u, 7u, 128u, 129u, 200u, 256u, 1000u};
    uint64_t seed = 0x51D3ull;
    for (uint32_t capacity : capacities) {
        const std::vector<bc::DeviceCorner> corners = randomCorners(capacity, seed++);
        const uint32_t counts[] = {0u, 1u, capacity / 2u, capacity};
        for (uint32_t count : counts) {
            const std::vector<float> got = runConversion(corners, count, capacity, -7.5f);
            const std::vector<float> want = hostConversion(corners, count, capacity);
            size_t differing = 0;
            for (size_t i = 0; i < want.size(); ++i) {
                if (got[i] != want[i]) ++differing;
            }
            BINCV_CHECK_EQ(differing, size_t{0});
        }
    }
}

BINCV_TEST(CudaKeypointsFromCorners, ZeroesTheSlotsPastTheCountSoStaleKeypointsCannotSurvive) {
    // The destination is PREFILLED with a plausible previous frame -- not with
    // zeros -- because a kernel that simply never wrote the tail would pass a
    // zero-prefilled test and would then hand a consumer last frame's corners.
    const uint32_t capacity = 256;
    const std::vector<bc::DeviceCorner> corners = randomCorners(capacity, 0xBEEFull);
    const std::vector<float> got = runConversion(corners, 40u, capacity, 123.5f);
    size_t nonZeroTail = 0;
    for (uint32_t i = 40; i < capacity; ++i) {
        if (got[2 * i] != 0.0f || got[2 * i + 1] != 0.0f) ++nonZeroTail;
    }
    BINCV_CHECK_EQ(nonZeroTail, size_t{0});
    // ...and the live half really was written, not left at the prefill.
    BINCV_CHECK_EQ(got[0], static_cast<float>(corners[0].x));
    BINCV_CHECK_EQ(got[79], static_cast<float>(corners[39].y));
}

BINCV_TEST(CudaKeypointsFromCorners, ClampsACountThatExceedsTheCapacity) {
    // An append counter is deliberately unclamped, so this is the shape a
    // TRUNCATED detection hands the op. Reading it unclamped would index past
    // the corner array; the answer must be the first `capacity` corners.
    const uint32_t capacity = 64;
    const std::vector<bc::DeviceCorner> corners = randomCorners(capacity, 0xC0FFEEull);
    const std::vector<float> got = runConversion(corners, 100000u, capacity, -1.0f);
    const std::vector<float> want = hostConversion(corners, capacity, capacity);
    size_t differing = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (got[i] != want[i]) ++differing;
    }
    BINCV_CHECK_EQ(differing, size_t{0});
}

BINCV_TEST(CudaKeypointsFromCorners, TheFastCornerSpellingAgreesWithTheHostConversion) {
    const uint32_t capacity = 300;
    uint64_t seed = 0x1234ull;
    std::vector<bc::DeviceFastCorner> fast(capacity);
    std::vector<bc::DeviceCorner> mirror(capacity);
    for (uint32_t i = 0; i < capacity; ++i) {
        fast[i].x = static_cast<int>(splitmix(seed) % 752u);
        fast[i].y = static_cast<int>(splitmix(seed) % 480u);
        fast[i].score = static_cast<long long>(splitmix(seed) % 16u);
        mirror[i].x = fast[i].x;
        mirror[i].y = fast[i].y;
        mirror[i].response = 0.0f;
    }
    bc::DeviceArray<bc::DeviceFastCorner> dCorners(capacity);
    bc::DeviceArray<uint32_t> dCount(1);
    bc::DeviceArray<float> dXY(2 * static_cast<size_t>(capacity));
    const uint32_t count = 211;
    BINCV_CUDA_CHECK(cudaMemcpy(dCorners.data(), fast.data(),
                                fast.size() * sizeof(bc::DeviceFastCorner),
                                cudaMemcpyHostToDevice));
    BINCV_CUDA_CHECK(
        cudaMemcpy(dCount.data(), &count, sizeof(uint32_t), cudaMemcpyHostToDevice));
    BINCV_CHECK_EQ(
        bc::keypointsFromCorners(dCorners.data(), dCount.data(), dXY.data(), capacity),
        cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> got(2 * static_cast<size_t>(capacity), -2.0f);
    BINCV_CUDA_CHECK(cudaMemcpy(got.data(), dXY.data(), got.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
    const std::vector<float> want = hostConversion(mirror, count, capacity);
    size_t differing = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (got[i] != want[i]) ++differing;
    }
    BINCV_CHECK_EQ(differing, size_t{0});
}

BINCV_TEST(CudaKeypointsFromCorners, RefusesNullPointersInsteadOfLaunching) {
    bc::DeviceArray<bc::DeviceCorner> corners(8);
    bc::DeviceArray<uint32_t> count(1);
    bc::DeviceArray<float> xy(16);
    const bc::DeviceCorner* noCorners = nullptr;
    // Deliberate domain violations: the launcher asserts the pointers and THEN
    // returns the code, so a checked build aborts before there is a code to
    // read. The shared harness runs these unchecked and reports them checked.
    BINCV_CHECK_EQ_UNLESS_CHECKED(
        bc::keypointsFromCorners(noCorners, count.data(), xy.data(), 8u),
        cudaErrorInvalidValue);
    BINCV_CHECK_EQ_UNLESS_CHECKED(
        bc::keypointsFromCorners(corners.data(), nullptr, xy.data(), 8u),
        cudaErrorInvalidValue);
    BINCV_CHECK_EQ_UNLESS_CHECKED(
        bc::keypointsFromCorners(corners.data(), count.data(), nullptr, 8u),
        cudaErrorInvalidValue);
    // A capacity of zero is a no-op, not an error -- the empty-view rule the
    // rest of this backend follows.
    BINCV_CHECK_EQ(bc::keypointsFromCorners(noCorners, nullptr, nullptr, 0u), cudaSuccess);
}

BINCV_TEST(CudaKeypointsFromCorners, DeviceCornerCountAddressesTheResultsOwnCountField) {
    // The helper casts rather than forming `&result->count` on a device
    // pointer, so the offset it assumes has to be checked against a real
    // DeviceCornerResult rather than trusted.
    bc::DeviceArray<bc::DeviceCornerResult> result(1);
    bc::DeviceCornerResult r;
    r.count = 37;
    r.candidatesRanked = 999;
    r.candidatesTruncated = 1;
    r.candidateOverflow = 0;
    BINCV_CUDA_CHECK(cudaMemcpy(result.data(), &r, sizeof(r), cudaMemcpyHostToDevice));

    const std::vector<bc::DeviceCorner> corners = randomCorners(64, 0x777ull);
    bc::DeviceArray<bc::DeviceCorner> dCorners(corners.size());
    bc::DeviceArray<float> dXY(2 * 64);
    BINCV_CUDA_CHECK(cudaMemcpy(dCorners.data(), corners.data(),
                                corners.size() * sizeof(bc::DeviceCorner),
                                cudaMemcpyHostToDevice));
    BINCV_CHECK_EQ(bc::keypointsFromCorners(dCorners.data(),
                                            bc::deviceCornerCount(result.data()),
                                            dXY.data(), 64u),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> got(128, -1.0f);
    BINCV_CUDA_CHECK(cudaMemcpy(got.data(), dXY.data(), got.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
    // 37 live, the rest zero -- which is only true if the helper read `count`
    // and not `candidatesRanked`.
    BINCV_CHECK_EQ(got[2 * 36], static_cast<float>(corners[36].x));
    BINCV_CHECK_EQ(got[2 * 37], 0.0f);
    BINCV_CHECK_EQ(got[2 * 37 + 1], 0.0f);
}

BINCV_TEST(CudaKeypointsFromCorners, ThePaddingSlotsAreRejectedByTheConsumers) {
    // THE ASSUMPTION THE RESIDENT PIPELINE RESTS ON, checked against the other
    // family's kernel rather than asserted in a comment: a launch sized by
    // CAPACITY must treat the padded slots as absent, or a frame with no
    // mid-pipeline synchronize is not available at all.
    const uint32_t capacity = 128;
    const uint32_t count = 20;
    const size_t w = 200, h = 160;
    std::vector<bc::DeviceCorner> corners(capacity);
    for (uint32_t i = 0; i < capacity; ++i) {
        // Every corner well inside the image, so `keep` can only be 0 for a
        // slot the conversion zeroed.
        corners[i].x = static_cast<int>(40 + (i % 100));
        corners[i].y = static_cast<int>(40 + (i % 80));
        corners[i].response = 1.0f;
    }
    bc::DeviceArray<bc::DeviceCorner> dCorners(capacity);
    bc::DeviceArray<uint32_t> dCount(1);
    bc::DeviceArray<float> dXY(2 * static_cast<size_t>(capacity));
    bc::DeviceArray<float> dAngles(capacity);
    bc::DeviceArray<uint8_t> dKeep(capacity);
    bc::DeviceImage<uint8_t> img(static_cast<int>(w), static_cast<int>(h));

    std::vector<uint8_t> frame(w * h);
    uint64_t seed = 0x2468ull;
    for (uint8_t& v : frame) v = static_cast<uint8_t>(splitmix(seed));
    BINCV_CHECK_EQ(bc::uploadImage<uint8_t>(frame.data(), w, h, w, img.view()), cudaSuccess);
    BINCV_CUDA_CHECK(cudaMemcpy(dCorners.data(), corners.data(),
                                corners.size() * sizeof(bc::DeviceCorner),
                                cudaMemcpyHostToDevice));
    BINCV_CUDA_CHECK(
        cudaMemcpy(dCount.data(), &count, sizeof(uint32_t), cudaMemcpyHostToDevice));
    BINCV_CHECK_EQ(bc::keypointsFromCorners(dCorners.data(), dCount.data(), dXY.data(),
                                            capacity),
                   cudaSuccess);
    BINCV_CHECK_EQ(bc::keypointOrientation(img.constView(),
                                           bc::keypointSet(dXY.data(), capacity),
                                           dAngles.data(), dKeep.data(), 15),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint8_t> keep(capacity, 0xFF);
    std::vector<float> angles(capacity, -1.0f);
    BINCV_CUDA_CHECK(cudaMemcpy(keep.data(), dKeep.data(), capacity, cudaMemcpyDeviceToHost));
    BINCV_CUDA_CHECK(cudaMemcpy(angles.data(), dAngles.data(), capacity * sizeof(float),
                                cudaMemcpyDeviceToHost));
    size_t liveKept = 0, padKept = 0, padNonZeroAngle = 0;
    for (uint32_t i = 0; i < capacity; ++i) {
        if (i < count) {
            liveKept += keep[i] != 0 ? 1u : 0u;
        } else {
            padKept += keep[i] != 0 ? 1u : 0u;
            if (angles[i] != 0.0f) ++padNonZeroAngle;
        }
    }
    BINCV_CHECK_EQ(liveKept, size_t{count});
    BINCV_CHECK_EQ(padKept, size_t{0});
    BINCV_CHECK_EQ(padNonZeroAngle, size_t{0});
}

BINCV_TEST(CudaKeypointsFromCorners, TheWholeChainAgreesWithTheCornersTheSameLaunchProduced) {
    // In situ: a real goodFeaturesToTrack selection, then the conversion, with
    // the xy array held to the corner records that very launch wrote. This is
    // the pairing the resident frontend performs, and it is the one place the
    // two families meet.
    const int w = 256, h = 192;
    bc::DeviceBinMat dxBlock(w, 2 * h), dyBlock(w, 2 * h);
    // A deterministic ternary field: magnitude bits from a hash, sign bits from
    // another, uploaded as whole planes.
    const size_t words = bincv::cuda::rowWords(static_cast<size_t>(w));
    std::vector<uint32_t> plane(words * static_cast<size_t>(2 * h));
    const uint32_t tail = bincv::cuda::rowTailMask(static_cast<size_t>(w));
    uint64_t seed = 0x99AAull;
    auto fill = [&](bc::DeviceBinMat& m, uint64_t s) {
        uint64_t local = s;
        for (size_t r = 0; r < static_cast<size_t>(2 * h); ++r) {
            for (size_t i = 0; i < words; ++i) {
                uint32_t v = static_cast<uint32_t>(splitmix(local));
                if (i + 1 == words) v &= tail;  // padding bits stay zero
                plane[r * words + i] = v;
            }
        }
        BINCV_CUDA_CHECK(cudaMemcpy(m.view().ptr, plane.data(),
                                    plane.size() * sizeof(uint32_t),
                                    cudaMemcpyHostToDevice));
    };
    fill(dxBlock, seed);
    fill(dyBlock, seed + 1);

    // The pool is sized to the whole frame: a pseudo-random ternary field is
    // far denser in 3x3 maxima than a real edge map, and an overflow here would
    // make the selection report nothing at all rather than exercise the link.
    const uint32_t rank = 4096, slots = 256;
    const uint32_t pool = static_cast<uint32_t>(w) * static_cast<uint32_t>(h);
    bc::DeviceArray<bc::DeviceCorner> candidates(pool);
    bc::DeviceAppendCounter counter;
    bc::DeviceArray<uint32_t> maxBits(1);
    bc::DeviceArray<uint8_t> scratch(bc::goodFeaturesScratchBytes(pool));
    bc::DeviceArray<bc::DeviceCorner> corners(rank);
    bc::DeviceArray<bc::DeviceCornerResult> result(1);
    bc::DeviceArray<float> dXY(2 * static_cast<size_t>(slots));

    bc::DeviceGoodFeaturesWorkspace work;
    work.candidates = bc::appendBuffer(candidates, counter);
    work.maxBits = maxBits.data();
    work.scratch = scratch.data();
    work.scratchBytes = scratch.size();

    BINCV_CHECK_EQ(counter.reset(), cudaSuccess);
    BINCV_CUDA_CHECK(cudaMemset(maxBits.data(), 0, sizeof(uint32_t)));
    const bc::DevicePlaneBlockView dx = bc::planeBlock(dxBlock.view(), 2);
    const bc::DevicePlaneBlockView dy = bc::planeBlock(dyBlock.view(), 2);
    const bincv::GoodFeaturesParams params{};
    BINCV_CHECK_EQ(bc::goodFeaturesToTrackAsync(dx.plane(0), dy.plane(0), dx.plane(1),
                                                dy.plane(1), params, work, corners.data(),
                                                rank, result.data()),
                   cudaSuccess);
    BINCV_CHECK_EQ(bc::keypointsFromCorners(corners.data(),
                                            bc::deviceCornerCount(result.data()),
                                            dXY.data(), slots),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bc::DeviceCornerResult r{};
    BINCV_CUDA_CHECK(cudaMemcpy(&r, result.data(), sizeof(r), cudaMemcpyDeviceToHost));
    BINCV_CHECK_EQ(r.candidateOverflow, uint32_t{0});
    std::vector<bc::DeviceCorner> hostCorners(rank);
    BINCV_CUDA_CHECK(cudaMemcpy(hostCorners.data(), corners.data(),
                                hostCorners.size() * sizeof(bc::DeviceCorner),
                                cudaMemcpyDeviceToHost));
    std::vector<float> got(2 * static_cast<size_t>(slots), -1.0f);
    BINCV_CUDA_CHECK(cudaMemcpy(got.data(), dXY.data(), got.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
    const std::vector<float> want = hostConversion(hostCorners, r.count, slots);
    size_t differing = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (got[i] != want[i]) ++differing;
    }
    BINCV_CHECK_EQ(differing, size_t{0});
    // A selection that found nothing would pass the comparison above without
    // exercising anything, so the case asserts it actually detected corners.
    BINCV_CHECK(r.count > 0);
}

namespace {
bool cudaDevicePresent() {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return false;
    }
    return true;
}
} // namespace

#if BINCV_TEST_WITH_GTEST
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    const int summaryRc = ::bincv::test::summarize("CUDA keypoint-link tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA keypoint-link tests", argc, argv);
}
#endif
