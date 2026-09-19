// The shared device vocabulary, exercised FROM THE DEVICE.
//
// test_cuda_backend.cpp proves the host-side arithmetic: a plane block names
// the same word offsets a QuantMat does, a descriptor set has the host's pitch,
// the result PODs convert. None of that runs on the GPU, and the failure this
// vocabulary exists to prevent -- a kernel reading the wrong plane, the wrong
// keypoint or the wrong descriptor and returning a plausible answer -- is a
// failure of the DEVICE-side addressing. So this suite is a CUDA translation
// unit, and every case here goes through a kernel that reads only through the
// views and writes back what it saw.
//
// It is also where the capacity contract is held to its word: a compaction
// whose count exceeds the caller's buffer must report the TRUE count and must
// not be able to pass as a complete answer.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/compaction.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/features.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/quantMat.hpp"
#include "test_util.hpp"

namespace {

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::vector<uint8_t> randomFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<uint8_t> img(w * h);
    for (auto& v : img) v = static_cast<uint8_t>(splitmix(seed));
    return img;
}

} // namespace

// ---------------------------------------------------------------------------
// The plane block, read by a kernel
//
// The kernel is handed nothing but the view: no plane pitch, no N * height
// row count, no base pointer of its own. If row(p, y) drifts from the host's
// plane(p).row(y), these words come back from a neighbouring plane and every
// one of them is a legal, plausible bit pattern.
// ---------------------------------------------------------------------------
namespace {

__global__ void planeBlockProbe(bincv::cuda::DevicePlaneBlockConstView blk, uint32_t* out,
                                size_t words) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t perPlane = words * blk.height;
    if (idx >= perPlane * blk.planes) return;
    const size_t p = idx / perPlane;
    const size_t rem = idx - p * perPlane;
    const size_t y = rem / words;
    const size_t i = rem - y * words;
    out[idx] = blk.row(p, y)[i];
}

template <size_t N>
void testPlaneBlockReadFromDevice(size_t w, size_t h) {
    const auto frame = randomFrame(w, h, 0xB10CU + N);

    bincv::QuantMat<N, uint32_t> host(static_cast<int>(w), static_cast<int>(h));
    bincv::BinMatView<uint32_t> planes[N];
    for (size_t p = 0; p < N; ++p) planes[p] = host.plane(p);
    bincv::packQuant<bincv::QuantRule::Scale, N, uint8_t, uint32_t>(frame.data(), w, h, w,
                                                                    planes);

    // The whole stack as one matrix -- the shape a transfer copies, and the
    // shape the plane block is then named over.
    const bincv::BinMatConstView<uint32_t> stack(host.data(), w, N * h,
                                                 host.getAlignedWidth());
    bincv::cuda::DeviceBinMat dBlock(static_cast<int>(w), static_cast<int>(N * h));
    BINCV_CHECK_EQ(bincv::cuda::upload(stack, dBlock.view()), cudaSuccess);

    const size_t words = bincv::impl::minRowWords<uint32_t>(w);
    bincv::cuda::DeviceArray<uint32_t> dOut(N * h * words);
    const bincv::cuda::DevicePlaneBlockConstView blk =
        bincv::cuda::planeBlock(dBlock.constView(), N);
    BINCV_CHECK_EQ(blk.height, h);

    const size_t total = N * h * words;
    const unsigned threads = 128;
    const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);
    planeBlockProbe<<<blocks, threads>>>(blk, dOut.data(), words);
    BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> got(total, 0xDEADBEEFu);
    BINCV_CHECK_EQ(cudaMemcpy(got.data(), dOut.data(), total * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost),
                   cudaSuccess);

    size_t bad = 0;
    for (size_t p = 0; p < N; ++p) {
        const bincv::BinMatConstView<uint32_t> hostPlane = host.constPlane(p);
        for (size_t y = 0; y < h; ++y) {
            const uint32_t* expect = hostPlane.row(y);
            const uint32_t* actual = got.data() + (p * h + y) * words;
            for (size_t i = 0; i < words; ++i)
                if (expect[i] != actual[i]) ++bad;
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
}

} // namespace

BINCV_TEST(CudaVocabulary, PlaneBlockRowsReadTheHostsWords_N2) {
    testPlaneBlockReadFromDevice<2>(133, 41);
}
BINCV_TEST(CudaVocabulary, PlaneBlockRowsReadTheHostsWords_N5) {
    testPlaneBlockReadFromDevice<5>(97, 23);
}
BINCV_TEST(CudaVocabulary, PlaneBlockRowsReadTheHostsWords_N8) {
    testPlaneBlockReadFromDevice<8>(31, 5);
}

// ---------------------------------------------------------------------------
// The keypoint and descriptor sets, read by a kernel
//
// One launch, the whole set -- which is the rule these types exist to enforce.
// The kernel writes back exactly what the views handed it, so a pitch that
// disagrees with computeBrief's `out + k * words` shows up as a descriptor
// belonging to the wrong keypoint rather than as a crash.
// ---------------------------------------------------------------------------
namespace {

__global__ void featureSetProbe(bincv::cuda::DeviceKeypointSetConstView kps,
                                bincv::cuda::DeviceDescriptorSetConstView desc,
                                float* xyOut, int32_t* octaveOut, uint32_t* descOut) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kps.count) return;
    xyOut[2u * i] = kps.x(i);
    xyOut[2u * i + 1u] = kps.y(i);
    octaveOut[i] = kps.hasOctave() ? kps.octave[i] : -1;
    const uint32_t* d = desc.descriptor(i);
    for (uint32_t w = 0; w < desc.wordsPerDescriptor; ++w)
        descOut[i * desc.wordsPerDescriptor + w] = d[w];
}

} // namespace

BINCV_TEST(CudaVocabulary, KeypointAndDescriptorSetsAddressTheHostArrays) {
    constexpr size_t kBits = 256;
    const uint32_t count = 61;
    const uint32_t words = bincv::cuda::descriptorWords<kBits>();
    const size_t w = 96, h = 96;

    // The host arrays, produced by the host family itself: this is the layout
    // under test, not a re-statement of it.
    const auto frame = randomFrame(w, h, 0xF0CU);
    std::vector<float> xy(2 * count);
    std::vector<int32_t> octave(count);
    for (uint32_t i = 0; i < count; ++i) {
        xy[2 * i] = static_cast<float>(30 + (i % 31));
        xy[2 * i + 1] = static_cast<float>(30 + (i % 29));
        octave[i] = static_cast<int32_t>(i % 4);
    }
    bincv::BriefPattern<kBits> pattern;
    bincv::makeBriefPattern<kBits>(pattern);
    std::vector<uint32_t> desc(static_cast<size_t>(count) * words);
    bincv::computeBrief<kBits, uint8_t, uint32_t>(frame.data(), w, h, w, xy.data(), count,
                                                  pattern, desc.data());

    bincv::cuda::DeviceArray<float> dXY(2u * count);
    bincv::cuda::DeviceArray<int32_t> dOctave(count);
    bincv::cuda::DeviceArray<uint32_t> dDesc(static_cast<size_t>(count) * words);
    BINCV_CHECK_EQ(cudaMemcpy(dXY.data(), xy.data(), xy.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaMemcpy(dOctave.data(), octave.data(), octave.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaMemcpy(dDesc.data(), desc.data(), desc.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   cudaSuccess);

    const bincv::cuda::DeviceKeypointSetConstView kps =
        bincv::cuda::keypointSet(dXY.data(), count, dOctave.data());
    // The mutable set converts to the read-only one, as the bit views do.
    const bincv::cuda::DeviceDescriptorSetView mutableSet =
        bincv::cuda::descriptorSet(dDesc.data(), count, words);
    const bincv::cuda::DeviceDescriptorSetConstView set = mutableSet;
    BINCV_CHECK_EQ(set.count, count);
    BINCV_CHECK_EQ(set.wordsPerDescriptor, words);

    bincv::cuda::DeviceArray<float> dXYOut(2u * count);
    bincv::cuda::DeviceArray<int32_t> dOctaveOut(count);
    bincv::cuda::DeviceArray<uint32_t> dDescOut(static_cast<size_t>(count) * words);
    featureSetProbe<<<(count + 63u) / 64u, 64u>>>(kps, set, dXYOut.data(), dOctaveOut.data(),
                                                  dDescOut.data());
    BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> gotXY(2 * count, -1.0f);
    std::vector<int32_t> gotOctave(count, -99);
    std::vector<uint32_t> gotDesc(static_cast<size_t>(count) * words, 0xDEADBEEFu);
    BINCV_CHECK_EQ(cudaMemcpy(gotXY.data(), dXYOut.data(), gotXY.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaMemcpy(gotOctave.data(), dOctaveOut.data(),
                              gotOctave.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaMemcpy(gotDesc.data(), dDescOut.data(),
                              gotDesc.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                   cudaSuccess);

    size_t badXY = 0, badOctave = 0, badDesc = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (gotXY[2 * i] != xy[2 * i]) ++badXY;
        if (gotXY[2 * i + 1] != xy[2 * i + 1]) ++badXY;
        if (gotOctave[i] != octave[i]) ++badOctave;
        for (uint32_t k = 0; k < words; ++k) {
            const size_t at = static_cast<size_t>(i) * words + k;
            if (gotDesc[at] != desc[at]) ++badDesc;
        }
    }
    BINCV_CHECK_EQ(badXY, 0u);
    BINCV_CHECK_EQ(badOctave, 0u);
    BINCV_CHECK_EQ(badDesc, 0u);
}

// ---------------------------------------------------------------------------
// The capacity contract
//
// Every candidate the probe emits is (x = i, y = 2i, response = 1), so a
// downloaded element can be checked against the set of candidates WITHOUT
// depending on order -- which is exactly what the contract says is unspecified.
// ---------------------------------------------------------------------------
namespace {

__global__ void appendProbe(bincv::cuda::DeviceCornerBuffer buf, uint32_t candidates) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= candidates) return;
    bincv::cuda::DeviceCorner c;
    c.x = static_cast<int>(i);
    c.y = static_cast<int>(i) * 2;
    c.response = 1.0f;
    buf.append(c);
}

/// Warp-aggregated appending: one atomic per thread for `perThread` elements.
__global__ void reserveProbe(bincv::cuda::DeviceCornerBuffer buf, uint32_t threads,
                             uint32_t perThread) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= threads) return;
    const uint32_t base = buf.reserve(perThread);
    for (uint32_t k = 0; k < perThread; ++k) {
        bincv::cuda::DeviceCorner c;
        c.x = static_cast<int>(i);
        c.y = static_cast<int>(k);
        c.response = 1.0f;
        buf.store(base + k, c);
    }
}

void launchAppend(const bincv::cuda::DeviceCornerBuffer& buf, uint32_t candidates) {
    appendProbe<<<(candidates + 255u) / 256u, 256u>>>(buf, candidates);
}

/// Every stored corner is one of the probe's candidates, and no candidate
/// appears twice. Returns the number of violations.
size_t checkAppendedCorners(const std::vector<bincv::cuda::DeviceCorner>& got,
                            uint32_t candidates) {
    std::vector<uint8_t> seen(candidates, 0);
    size_t bad = 0;
    for (const auto& c : got) {
        if (c.response != 1.0f) { ++bad; continue; }
        if (c.x < 0 || static_cast<uint32_t>(c.x) >= candidates) { ++bad; continue; }
        if (c.y != c.x * 2) ++bad;
        if (seen[static_cast<size_t>(c.x)] != 0) ++bad;
        seen[static_cast<size_t>(c.x)] = 1;
    }
    return bad;
}

} // namespace

BINCV_TEST(CudaCompaction, CompleteRunReportsTheWholeCount) {
    const uint32_t candidates = 1000, capacity = 4096;
    bincv::cuda::DeviceArray<bincv::cuda::DeviceCorner> buffer(capacity);
    bincv::cuda::DeviceAppendCounter counter;
    BINCV_CHECK_EQ(counter.reset(), cudaSuccess);
    const bincv::cuda::DeviceCornerBuffer view =
        bincv::cuda::appendBuffer(buffer, counter);
    BINCV_CHECK_EQ(view.capacity, capacity);

    launchAppend(view, candidates);
    BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bincv::cuda::DeviceAppendResult result;
    BINCV_CHECK_EQ(bincv::cuda::readAppendResult(view, result), cudaSuccess);
    BINCV_CHECK_EQ(result.found(), candidates);
    BINCV_CHECK(!result.truncated());
    BINCV_CHECK_EQ(result.acceptTruncated(), candidates);
    uint32_t n = 0xA5A5A5A5u;
    BINCV_CHECK(result.completeCount(n));
    BINCV_CHECK_EQ(n, candidates);

    std::vector<bincv::cuda::DeviceCorner> got(candidates);
    BINCV_CHECK_EQ(bincv::cuda::downloadAppended(view, result, got.data()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    BINCV_CHECK_EQ(checkAppendedCorners(got, candidates), 0u);
}

BINCV_TEST(CudaCompaction, TruncatedRunCannotPassAsComplete) {
    const uint32_t candidates = 5000, capacity = 64;
    bincv::cuda::DeviceArray<bincv::cuda::DeviceCorner> buffer(capacity);
    bincv::cuda::DeviceAppendCounter counter;
    BINCV_CHECK_EQ(counter.reset(), cudaSuccess);
    const bincv::cuda::DeviceCornerBuffer view =
        bincv::cuda::appendBuffer(buffer, counter);

    launchAppend(view, candidates);
    BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bincv::cuda::DeviceAppendResult result;
    BINCV_CHECK_EQ(bincv::cuda::readAppendResult(view, result), cudaSuccess);

    // THE COUNTER IS NOT CLAMPED: the true count comes back, which is the
    // capacity a complete re-run needs. A clamped counter would report 64 and
    // leave the caller with no way to size the retry.
    BINCV_CHECK_EQ(result.found(), candidates);
    BINCV_CHECK(result.truncated());
    BINCV_CHECK_EQ(result.capacity(), capacity);

    // completeCount REFUSES, and leaves its output untouched -- so a caller who
    // ignores the refusal is left with their own sentinel, not a plausible
    // count they could mistake for the whole answer.
    uint32_t n = 0xA5A5A5A5u;
    BINCV_CHECK(!result.completeCount(n));
    BINCV_CHECK_EQ(n, 0xA5A5A5A5u);

    // Accepting the partial answer is possible, and says so at the call site.
    BINCV_CHECK_EQ(result.acceptTruncated(), capacity);

    std::vector<bincv::cuda::DeviceCorner> got(capacity);
    BINCV_CHECK_EQ(bincv::cuda::downloadAppended(view, result, got.data()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // The kept elements are real detections -- truncation drops candidates, it
    // does not corrupt the ones that fit. WHICH ones is the atomic's business,
    // so this is a set check, not an order check.
    BINCV_CHECK_EQ(checkAppendedCorners(got, candidates), 0u);
}

BINCV_TEST(CudaCompaction, ExactFillIsNotTruncation) {
    // found == capacity is the boundary the contract is most easily got wrong
    // at: it is a COMPLETE answer, and reporting it as truncated would send
    // every caller who sized exactly right into a pointless re-run.
    const uint32_t candidates = 256, capacity = 256;
    bincv::cuda::DeviceArray<bincv::cuda::DeviceCorner> buffer(capacity);
    bincv::cuda::DeviceAppendCounter counter;
    BINCV_CHECK_EQ(counter.reset(), cudaSuccess);
    const bincv::cuda::DeviceCornerBuffer view =
        bincv::cuda::appendBuffer(buffer, counter);

    launchAppend(view, candidates);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bincv::cuda::DeviceAppendResult result;
    BINCV_CHECK_EQ(bincv::cuda::readAppendResult(view, result), cudaSuccess);
    BINCV_CHECK(!result.truncated());
    uint32_t n = 0;
    BINCV_CHECK(result.completeCount(n));
    BINCV_CHECK_EQ(n, capacity);

    std::vector<bincv::cuda::DeviceCorner> got(capacity);
    BINCV_CHECK_EQ(bincv::cuda::downloadAppended(view, result, got.data()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    BINCV_CHECK_EQ(checkAppendedCorners(got, candidates), 0u);
}

BINCV_TEST(CudaCompaction, TheCounterMustBeResetPerLaunch) {
    // Two launches over one counter accumulate. That is not a defect to hide --
    // it is what makes a banded or multi-tile detection add up to one set --
    // but it is why reset() is a visible member rather than something a
    // launcher does for you.
    const uint32_t candidates = 300, capacity = 2048;
    bincv::cuda::DeviceArray<bincv::cuda::DeviceCorner> buffer(capacity);
    bincv::cuda::DeviceAppendCounter counter;
    BINCV_CHECK_EQ(counter.reset(), cudaSuccess);
    const bincv::cuda::DeviceCornerBuffer view =
        bincv::cuda::appendBuffer(buffer, counter);

    launchAppend(view, candidates);
    launchAppend(view, candidates);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    bincv::cuda::DeviceAppendResult twice;
    BINCV_CHECK_EQ(bincv::cuda::readAppendResult(view, twice), cudaSuccess);
    BINCV_CHECK_EQ(twice.found(), 2u * candidates);

    BINCV_CHECK_EQ(counter.reset(), cudaSuccess);
    launchAppend(view, candidates);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    bincv::cuda::DeviceAppendResult once;
    BINCV_CHECK_EQ(bincv::cuda::readAppendResult(view, once), cudaSuccess);
    BINCV_CHECK_EQ(once.found(), candidates);
}

BINCV_TEST(CudaCompaction, ReserveStoresBlocksAndStillCountsRejections) {
    const uint32_t threads = 500, perThread = 4;
    const uint32_t candidates = threads * perThread;  // 2000
    const uint32_t capacity = 100;
    bincv::cuda::DeviceArray<bincv::cuda::DeviceCorner> buffer(capacity);
    bincv::cuda::DeviceAppendCounter counter;
    BINCV_CHECK_EQ(counter.reset(), cudaSuccess);
    const bincv::cuda::DeviceCornerBuffer view =
        bincv::cuda::appendBuffer(buffer, counter);

    reserveProbe<<<(threads + 127u) / 128u, 128u>>>(view, threads, perThread);
    BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    bincv::cuda::DeviceAppendResult result;
    BINCV_CHECK_EQ(bincv::cuda::readAppendResult(view, result), cudaSuccess);
    // The counter advances by the FULL reservation, rejected elements included.
    BINCV_CHECK_EQ(result.found(), candidates);
    BINCV_CHECK(result.truncated());
    BINCV_CHECK_EQ(result.acceptTruncated(), capacity);

    // Every slot below capacity was reserved by exactly one thread and stored,
    // so all of them carry a written corner rather than the array's zero fill.
    std::vector<bincv::cuda::DeviceCorner> got(capacity);
    BINCV_CHECK_EQ(bincv::cuda::downloadAppended(view, result, got.data()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    size_t unwritten = 0;
    for (const auto& c : got)
        if (c.response != 1.0f) ++unwritten;
    BINCV_CHECK_EQ(unwritten, 0u);
}

// ---------------------------------------------------------------------------
// The owning array the results live in
// ---------------------------------------------------------------------------
BINCV_TEST(CudaVocabulary, DeviceArrayIsZeroFilledAndDeepCopies) {
    bincv::cuda::DeviceArray<uint32_t> a(8);
    BINCV_CHECK_EQ(a.size(), size_t{8});
    std::vector<uint32_t> back(8, 0xFFFFFFFFu);
    BINCV_CHECK_EQ(cudaMemcpy(back.data(), a.data(), back.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost),
                   cudaSuccess);
    size_t nonZero = 0;
    for (uint32_t v : back)
        if (v != 0) ++nonZero;
    BINCV_CHECK_EQ(nonZero, 0u);

    const std::vector<uint32_t> seed(8, 7u);
    BINCV_CHECK_EQ(cudaMemcpy(a.data(), seed.data(), seed.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   cudaSuccess);
    // Copy means deep copy (CLAUDE.md): the copy has its own allocation.
    bincv::cuda::DeviceArray<uint32_t> b(a);
    BINCV_CHECK(b.data() != a.data());
    std::vector<uint32_t> copied(8, 0u);
    BINCV_CHECK_EQ(cudaMemcpy(copied.data(), b.data(), copied.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost),
                   cudaSuccess);
    size_t bad = 0;
    for (uint32_t v : copied)
        if (v != 7u) ++bad;
    BINCV_CHECK_EQ(bad, 0u);
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
    const int summaryRc = ::bincv::test::summarize("CUDA device vocabulary tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA device vocabulary tests", argc, argv);
}
#endif
