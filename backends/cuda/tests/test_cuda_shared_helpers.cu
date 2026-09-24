// The BINCV_HOST_DEVICE helpers, run on BOTH targets and compared.
//
// core/error.hpp defines BINCV_HOST_DEVICE so that a scalar helper can be ONE
// definition compiled twice instead of a host original and a device twin. That
// is only worth anything if the two compilations agree, and "it is the same
// source" does not establish it: the device has its own double-precision
// hardware, its own 64-bit integer emulation, its own libm, and its own
// compiler's licence to contract and reassociate. A helper whose host and
// device answers differ in the last bit is exactly the silent wrongness the
// annotation was introduced to remove, so it is swept here rather than assumed.
//
// Every case has the same shape, and it is the shape the rowWords/rowTailMask
// twin sweep in test_cuda_backend.cpp already uses: build the inputs ONCE on the
// host, upload that array, let a kernel call the helper on each element, download
// the answers, and call the same helper on the same array from host code. The
// inputs are uploaded rather than regenerated inside the kernel on purpose -- a
// sweep that builds its own inputs on each side is two definitions of the sweep,
// which is the mistake this file exists to catch one level down.
//
// Results are compared EXACTLY, including the float ones: minEigenValue is
// compared by bit pattern, because "close enough" is not the claim. A corner
// response that differs in the last bit changes which corners survive a
// quality-level cut.
//
// One case here is not a comparison: the last one calls detail::assertFailed
// from a kernel, because the device branch that makes BINCV_ASSERT legal inside
// a shared helper is otherwise compiled by nothing. See the note above it.
//
// Exits 77 when no CUDA device is present, like the other suites here.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/core/error.hpp"
#include "bincv/core/types.hpp"
#include "bincv/cuda/cornerSqrtMemo.cuh"
#include "bincv/ops/bitslice.hpp"
#include "bincv/ops/corner.hpp"
#include "bincv/ops/edge.hpp"
#include "bincv/ops/reduce.hpp"
#include "bincv/ops/shift.hpp"
#include "bincv/ops/threshold.hpp"
#include "test_util.hpp"

// ---------------------------------------------------------------------------
// The mirror of the assertion in tests/test_error.cpp. That one proves the macro
// expands to NOTHING off nvcc, which is what keeps the library buildable with no
// CUDA installed; this one proves it expands to something here. Either alone
// would be satisfied by a macro that is always empty -- which would compile
// everywhere and put none of the helpers below on the device.
// ---------------------------------------------------------------------------
#define BINCV_TEST_STRINGIZE_(x) #x
#define BINCV_TEST_STRINGIZE(x) BINCV_TEST_STRINGIZE_(x)
static_assert(BINCV_TEST_STRINGIZE(BINCV_HOST_DEVICE)[0] != '\0',
              "nvcc is compiling this file, so BINCV_HOST_DEVICE must carry the "
              "__host__ __device__ annotations -- an empty expansion here would mean "
              "every helper below was tested against itself on the host");

namespace {

/// @brief A device array, allocated and freed with the object.
/// @note Deliberately plain: the backend's own DeviceBinMat is a bit-matrix
/// container, and none of the helpers under test has anything to do with a
/// bit matrix. What is being moved here is a list of scalar cases.
template <typename T>
class DevArray {
public:
    explicit DevArray(size_t count) : n_(count) {
        if (cudaMalloc(&p_, count * sizeof(T)) != cudaSuccess) p_ = nullptr;
    }
    explicit DevArray(const std::vector<T>& host) : DevArray(host.size()) {
        if (p_ != nullptr && n_ != 0)
            cudaMemcpy(p_, host.data(), n_ * sizeof(T), cudaMemcpyHostToDevice);
    }
    ~DevArray() { cudaFree(p_); }
    DevArray(const DevArray&) = delete;
    DevArray& operator=(const DevArray&) = delete;

    T* get() const { return p_; }
    size_t size() const { return n_; }

    std::vector<T> download() const {
        std::vector<T> out(n_);
        if (p_ != nullptr && n_ != 0)
            cudaMemcpy(out.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost);
        return out;
    }

private:
    T* p_ = nullptr;
    size_t n_ = 0;
};

unsigned blocksFor(size_t n) {
    const size_t b = (n + 255u) / 256u;
    return static_cast<unsigned>(b < 1024u ? (b ? b : 1u) : 1024u);
}

/// @brief The launch itself succeeded and the device finished the work.
/// @note Checked per case rather than once: a kernel that never ran leaves the
/// output buffer holding whatever cudaMalloc handed back, and comparing that
/// against the host answers is a test of luck.
bool launchOk() {
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// impl::borderIndex -- ops/shift.hpp, the Tier 1 border promise
// ---------------------------------------------------------------------------

struct BorderCase {
    long long p;
    size_t len;
    int type;
};

__global__ void kBorderIndex(const BorderCase* in, size_t n, long long* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = static_cast<long long>(
            bincv::impl::borderIndex(static_cast<ptrdiff_t>(in[i].p), in[i].len,
                                     static_cast<bincv::BorderType>(in[i].type)));
    }
}

long long borderIndexHost(const BorderCase& c) {
    return static_cast<long long>(bincv::impl::borderIndex(
        static_cast<ptrdiff_t>(c.p), c.len, static_cast<bincv::BorderType>(c.type)));
}

// ---------------------------------------------------------------------------
// impl::reflect101Edge -- ops/edge.hpp
// ---------------------------------------------------------------------------

struct ReflectCase {
    long long i;
    size_t n;
};

__global__ void kReflect(const ReflectCase* in, size_t n, size_t* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = bincv::impl::reflect101Edge(in[i].i, in[i].n);
    }
}

// ---------------------------------------------------------------------------
// maj3 -- ops/bitslice.hpp, at three word widths
//
// All three, because maj3's casts are the -Wconversion tax on integer
// promotion: `a & b` is an `int` at uint8_t and uint16_t, and it is the narrow
// widths where a promotion that is not cast back changes the answer.
// ---------------------------------------------------------------------------

struct Maj3Case {
    uint64_t a, b, c;
};

__global__ void kMaj3(const Maj3Case* in, size_t n, uint8_t* out8, uint32_t* out32,
                      uint64_t* out64) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out8[i] = bincv::maj3<uint8_t>(static_cast<uint8_t>(in[i].a),
                                       static_cast<uint8_t>(in[i].b),
                                       static_cast<uint8_t>(in[i].c));
        out32[i] = bincv::maj3<uint32_t>(static_cast<uint32_t>(in[i].a),
                                         static_cast<uint32_t>(in[i].b),
                                         static_cast<uint32_t>(in[i].c));
        out64[i] = bincv::maj3<uint64_t>(in[i].a, in[i].b, in[i].c);
    }
}

// ---------------------------------------------------------------------------
// thresholdGE -- ops/bitslice.hpp
// ---------------------------------------------------------------------------

constexpr size_t kMaxPlanes = 6;

struct ThresholdGECase {
    uint32_t planes[kMaxPlanes];
    size_t nPlanes;
    unsigned threshold;
};

__global__ void kThresholdGE(const ThresholdGECase* in, size_t n, uint32_t* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = bincv::thresholdGE<uint32_t>(in[i].planes, in[i].nPlanes, in[i].threshold);
    }
}

// ---------------------------------------------------------------------------
// impl::minEigenValue -- ops/corner.hpp. The only float in this file.
// ---------------------------------------------------------------------------

struct EigCase {
    long long xx, yy, xy;
};

__global__ void kMinEig(const EigCase* in, size_t n, float* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = bincv::impl::minEigenValue(in[i].xx, in[i].yy, in[i].xy);
    }
}

// ---------------------------------------------------------------------------
// cuda::impl::minEigenValueMemo3 and its table -- backends/cuda's
// cornerSqrtMemo.cuh. Not a BINCV_HOST_DEVICE helper, and here anyway for this
// file's own reason: the memo is exact only because the device's `sqrt`, the
// host's `std::sqrt` and a written-out double are one value, and "IEEE-754 says
// so" is the kind of claim this file exists to sweep rather than accept. The
// table is `static` per translation unit, so these sweep THIS unit's copy of
// the one definition in that header.
// ---------------------------------------------------------------------------

__global__ void kSqrtMemoTable(double* out) {
    for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         i < bincv::cuda::impl::kSqrtMemoSide * bincv::cuda::impl::kSqrtMemoSide;
         i += static_cast<int>(gridDim.x * blockDim.x)) {
        out[i] = bincv::cuda::impl::kSqrtMemo[i];
    }
}

__global__ void kDeviceSqrt(size_t n, double* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = sqrt(static_cast<double>(i));
    }
}

__global__ void kMemoEig(const EigCase* in, size_t n, float* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = bincv::cuda::impl::minEigenValueMemo3(in[i].xx, in[i].yy, in[i].xy);
    }
}

/// @brief A float's bits, so equality is EXACT and NaN compares equal to itself.
uint32_t bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

// ---------------------------------------------------------------------------
// impl::thresholdCutoff -- ops/threshold.hpp
//
// This is the helper the hoist was for: it used to live inside the
// BINCV_WITH_OPENCV block, and this gate configures -DBINCV_USE_OPENCV=OFF, so
// before the hoist a device threshold could only have been compared against a
// copy of the reduction written in a test.
// ---------------------------------------------------------------------------

__global__ void kCutoff(const double* in, size_t n, int* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = bincv::impl::thresholdCutoff(in[i]);
    }
}

// ---------------------------------------------------------------------------
// impl::squareInsideImage -- impl/kernel_util.hpp, the keypoint-describability
// rule. Shared because orientation and BRIEF must agree about which keypoints
// they may read at all: a device copy drifting by one would not crash and would
// not corrupt a pixel, it would silently shorten the descriptor set.
// ---------------------------------------------------------------------------

struct SquareCase {
    long long cx;
    long long cy;
    int half;
    size_t width;
    size_t height;
};

__global__ void kSquare(const SquareCase* in, size_t n, unsigned char* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const SquareCase c = in[i];
        out[i] = bincv::impl::squareInsideImage(c.cx, c.cy, c.half, c.width, c.height)
                     ? 1u
                     : 0u;
    }
}

// ---------------------------------------------------------------------------
// impl::quantScale -- impl/kernel_util.hpp, the level map backends/cuda/src/pack.cu
// now calls instead of restating
//
// The sweep is EXHAUSTIVE over uint8_t and over every N the packer supports,
// because the claim here is not "these agree on the values I thought of". The
// `+ srcMax/2` rounding diverges from OpenCV at bytes 1..127 deliberately, and a
// device copy drifting toward OpenCV would look like that divergence being
// repaired -- so the whole byte range is checked rather than sampled. uint16_t is
// sampled instead: 65536 values times 8 levels is still small, so it is
// exhaustive too.
// ---------------------------------------------------------------------------

struct QuantCase {
    uint32_t v;
    unsigned maxValue;
};

__global__ void kQuant8(const QuantCase* in, size_t n, unsigned* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = bincv::impl::quantScale<uint8_t>(static_cast<uint8_t>(in[i].v),
                                                  in[i].maxValue);
    }
}

__global__ void kQuant16(const QuantCase* in, size_t n, unsigned* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = bincv::impl::quantScale<uint16_t>(static_cast<uint16_t>(in[i].v),
                                                   in[i].maxValue);
    }
}

// ---------------------------------------------------------------------------
// impl::clipRegion -- ops/reduce.hpp, the geometry backends/cuda/src/reduce.cu
// now calls instead of restating
// ---------------------------------------------------------------------------

struct ClipCase {
    size_t width;
    size_t height;
    bincv::Rect rect;
};

using Region = bincv::impl::RegionWords<uint32_t>;

__global__ void kClip(const ClipCase* in, size_t n, Region* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = bincv::impl::clipRegion<uint32_t>(in[i].width, in[i].height, in[i].rect);
    }
}

/// @brief Every field, because a clip that agrees on the masks and not on the
/// row range is still a different region.
bool sameRegion(const Region& a, const Region& b) {
    if (a.isEmpty != b.isEmpty) return false;
    // An empty region carries no geometry; the constructor's zeros are all that
    // is promised, and both sides return the same default-constructed object.
    if (a.isEmpty) return true;
    return a.y0 == b.y0 && a.y1 == b.y1 && a.firstWord == b.firstWord &&
           a.lastWord == b.lastWord && a.headMask == b.headMask &&
           a.tailMask == b.tailMask && a.x0 == b.x0 && a.x1 == b.x1;
}

// ---------------------------------------------------------------------------
// detail::assertFailed -- core/error.hpp, the device branch
//
// BINCV_ASSERT is what keeps a shared helper's preconditions from vanishing on
// one of its two targets, and it is legal on the device only because
// assertFailed has a `printf` / `__trap()` branch under __CUDA_ARCH__ instead of
// `std::fprintf` / `std::abort`. Nothing else in the tree compiles that branch.
// verify_cuda.sh is the only gate that compiles a .cu at all and it builds
// Release, so NDEBUG removes every BINCV_ASSERT body and nothing calls
// assertFailed from device code -- and nvcc does NOT diagnose the branch of a
// `__host__ __device__` inline function that nobody calls. Measured, by breaking
// the branch: a translation unit that only INCLUDES the header still compiled
// clean, while one that CALLS assertFailed failed with "calling a __host__
// function from a __global__ function".
//
// So it is called here, from device code, under a condition the host never
// satisfies. The point is the compile, not the run: `trigger` is device memory,
// so the branch survives to SASS, and it holds a value that makes it dead at
// runtime -- a test that actually trapped would kill the context and take the
// rest of the suite with it.
// ---------------------------------------------------------------------------

__global__ void kAssertBranch(const uint32_t* trigger, uint32_t* out) {
    if (*trigger == 0xDEADBEEFu) {
        bincv::detail::assertFailed("false", "the device assert branch is compiled",
                                    __FILE__, __LINE__);
    }
    *out = *trigger + 1u;
}

} // namespace

// ---------------------------------------------------------------------------

BINCV_TEST(CudaSharedHelpers, BorderIndexAgreesWithHost) {
    std::vector<BorderCase> cases;
    const int types[] = {bincv::BORDER_CONSTANT, bincv::BORDER_REPLICATE,
                         bincv::BORDER_REFLECT, bincv::BORDER_WRAP,
                         bincv::BORDER_REFLECT_101};
    // len == 1 is the case both reflect flavours answer before their closed form
    // (the period is zero there); the long reach is what the closed form exists
    // for, and is where a loop-based restatement would have differed in cost
    // rather than in answer.
    for (size_t len : {size_t{1}, size_t{2}, size_t{3}, size_t{7}, size_t{32}, size_t{33},
                       size_t{64}, size_t{1000}}) {
        for (long long p = -600; p <= 600; ++p)
            for (int t : types) cases.push_back(BorderCase{p, len, t});
    }

    DevArray<BorderCase> dIn(cases);
    DevArray<long long> dOut(cases.size());
    kBorderIndex<<<blocksFor(cases.size()), 256>>>(dIn.get(), cases.size(), dOut.get());
    BINCV_CHECK(launchOk());
    const std::vector<long long> got = dOut.download();

    size_t bad = 0;
    for (size_t i = 0; i < cases.size(); ++i)
        if (got[i] != borderIndexHost(cases[i])) ++bad;
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK(cases.size() > 40000u);
}

BINCV_TEST(CudaSharedHelpers, Reflect101EdgeAgreesWithHost) {
    std::vector<ReflectCase> cases;
    for (size_t n = 1; n <= 40; ++n)
        for (long long i = -80; i <= 80; ++i) cases.push_back(ReflectCase{i, n});

    DevArray<ReflectCase> dIn(cases);
    DevArray<size_t> dOut(cases.size());
    kReflect<<<blocksFor(cases.size()), 256>>>(dIn.get(), cases.size(), dOut.get());
    BINCV_CHECK(launchOk());
    const std::vector<size_t> got = dOut.download();

    size_t bad = 0;
    for (size_t i = 0; i < cases.size(); ++i)
        if (got[i] != bincv::impl::reflect101Edge(cases[i].i, cases[i].n)) ++bad;
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSharedHelpers, Maj3AgreesWithHostAtEveryWidth) {
    std::vector<Maj3Case> cases;
    uint64_t seed = 0x5EED1234u;
    // The exhaustive 3-bit truth table first -- majority is a per-lane function,
    // so eight lanes covering all eight (a, b, c) patterns IS the whole rule --
    // then random words, which is what catches a mask applied to the wrong lane.
    cases.push_back(Maj3Case{0xF0F0F0F0F0F0F0F0ULL, 0xCCCCCCCCCCCCCCCCULL,
                             0xAAAAAAAAAAAAAAAAULL});
    cases.push_back(Maj3Case{0, 0, 0});
    cases.push_back(Maj3Case{~0ULL, ~0ULL, ~0ULL});
    for (int i = 0; i < 4096; ++i)
        cases.push_back(Maj3Case{splitmix(seed), splitmix(seed), splitmix(seed)});

    DevArray<Maj3Case> dIn(cases);
    DevArray<uint8_t> d8(cases.size());
    DevArray<uint32_t> d32(cases.size());
    DevArray<uint64_t> d64(cases.size());
    kMaj3<<<blocksFor(cases.size()), 256>>>(dIn.get(), cases.size(), d8.get(), d32.get(),
                                            d64.get());
    BINCV_CHECK(launchOk());
    const std::vector<uint8_t> got8 = d8.download();
    const std::vector<uint32_t> got32 = d32.download();
    const std::vector<uint64_t> got64 = d64.download();

    size_t bad = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const Maj3Case& c = cases[i];
        if (got8[i] != bincv::maj3<uint8_t>(static_cast<uint8_t>(c.a),
                                            static_cast<uint8_t>(c.b),
                                            static_cast<uint8_t>(c.c)))
            ++bad;
        if (got32[i] != bincv::maj3<uint32_t>(static_cast<uint32_t>(c.a),
                                              static_cast<uint32_t>(c.b),
                                              static_cast<uint32_t>(c.c)))
            ++bad;
        if (got64[i] != bincv::maj3<uint64_t>(c.a, c.b, c.c)) ++bad;
    }
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSharedHelpers, ThresholdGEAgreesWithHost) {
    std::vector<ThresholdGECase> cases;
    uint64_t seed = 0xB17511CEu;
    // Both degenerate ends are included on purpose: threshold 0 passes every lane
    // whatever the planes hold, and a threshold wider than nPlanes bits passes
    // none -- the two answers a caller sweeping thresholds reaches by arithmetic
    // rather than by choice.
    for (size_t nPlanes = 0; nPlanes <= kMaxPlanes; ++nPlanes) {
        for (unsigned threshold = 0; threshold <= (1u << kMaxPlanes) + 2u; ++threshold) {
            for (int rep = 0; rep < 3; ++rep) {
                ThresholdGECase c{};
                for (size_t p = 0; p < kMaxPlanes; ++p)
                    c.planes[p] = static_cast<uint32_t>(splitmix(seed));
                c.nPlanes = nPlanes;
                c.threshold = threshold;
                cases.push_back(c);
            }
        }
    }

    DevArray<ThresholdGECase> dIn(cases);
    DevArray<uint32_t> dOut(cases.size());
    kThresholdGE<<<blocksFor(cases.size()), 256>>>(dIn.get(), cases.size(), dOut.get());
    BINCV_CHECK(launchOk());
    const std::vector<uint32_t> got = dOut.download();

    size_t bad = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const uint32_t want = bincv::thresholdGE<uint32_t>(cases[i].planes,
                                                           cases[i].nPlanes,
                                                           cases[i].threshold);
        if (got[i] != want) ++bad;
    }
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSharedHelpers, MinEigenValueIsBitIdenticalToHost) {
    std::vector<EigCase> cases;
    uint64_t seed = 0xE16E0001u;
    // The degenerate corner first: xx*yy == xy*xy must give exactly 0.0f, which is
    // what the selection's `!= 0` test depends on (ops/corner.hpp, PRECISION).
    cases.push_back(EigCase{0, 0, 0});
    cases.push_back(EigCase{4, 4, 4});
    cases.push_back(EigCase{9, 1, 3});
    cases.push_back(EigCase{1, 0, 0});
    // A 31x31 window over a binary image bounds the sums at 961, which is the
    // range the tracker actually produces; the large values check that nothing
    // overflows into a different rounding on one side only.
    for (long long xx = 0; xx <= 961; xx += 37)
        for (long long yy = 0; yy <= 961; yy += 53)
            for (long long xy = -961; xy <= 961; xy += 211)
                cases.push_back(EigCase{xx, yy, xy});
    for (int i = 0; i < 4096; ++i) {
        const long long a = static_cast<long long>(splitmix(seed) % 1000000ULL);
        const long long b = static_cast<long long>(splitmix(seed) % 1000000ULL);
        const long long c = static_cast<long long>(splitmix(seed) % 2000001ULL) - 1000000LL;
        cases.push_back(EigCase{a, b, c});
    }

    DevArray<EigCase> dIn(cases);
    DevArray<float> dOut(cases.size());
    kMinEig<<<blocksFor(cases.size()), 256>>>(dIn.get(), cases.size(), dOut.get());
    BINCV_CHECK(launchOk());
    const std::vector<float> got = dOut.download();

    size_t bad = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const float want =
            bincv::impl::minEigenValue(cases[i].xx, cases[i].yy, cases[i].xy);
        if (bits(got[i]) != bits(want)) ++bad;
    }
    BINCV_CHECK_EQ(bad, 0u);
}

// The claim the square-root memo rests on, checked rather than cited: IEEE-754
// requires `sqrt` to be correctly rounded, so there is exactly ONE double for
// each of these arguments and the device's libm must produce it. `disc` at
// blockSize 3 is an integer in [0, 405], so the sweep is the whole domain and
// not a sample of it. A single disagreement here would mean the memo cannot be
// exact -- and equally that `minEigenValue` itself does not give one answer on
// two targets, which is a larger finding than the memo.
BINCV_TEST(CudaSharedHelpers, DeviceSqrtIsTheHostsOverTheDiscriminantDomain) {
    constexpr size_t kDiscCount = 406;
    DevArray<double> dOut(kDiscCount);
    kDeviceSqrt<<<blocksFor(kDiscCount), 256>>>(kDiscCount, dOut.get());
    BINCV_CHECK(launchOk());
    const std::vector<double> got = dOut.download();

    size_t bad = 0;
    for (size_t i = 0; i < kDiscCount; ++i) {
        const double want = std::sqrt(static_cast<double>(i));
        if (std::memcmp(&got[i], &want, sizeof(double)) != 0) ++bad;
    }
    BINCV_CHECK_EQ(bad, 0u);
}

// EVERY entry of the table, not the ones a frame happens to reach. The kernels
// can only index the 55 cells with `|d| + |c| <= 9`, so a wrong value in any of
// the other 45 would sit there until the reachable set changed.
BINCV_TEST(CudaSharedHelpers, SqrtMemoTableIsTheHostsSquareRoot) {
    constexpr int kSide = bincv::cuda::impl::kSqrtMemoSide;
    DevArray<double> dOut(static_cast<size_t>(kSide * kSide));
    kSqrtMemoTable<<<1, 256>>>(dOut.get());
    BINCV_CHECK(launchOk());
    const std::vector<double> got = dOut.download();

    size_t bad = 0;
    for (int a = 0; a < kSide; ++a) {
        for (int c = 0; c < kSide; ++c) {
            const double want = std::sqrt(static_cast<double>(a * a + 4 * c * c));
            if (std::memcmp(&got[static_cast<size_t>(a * kSide + c)], &want,
                            sizeof(double)) != 0) {
                ++bad;
                if (bad == 1u) {
                    std::printf("  sqrt memo [|d|=%d][|c|=%d]: table %.17g host %.17g\n", a, c,
                                got[static_cast<size_t>(a * kSide + c)], want);
                }
            }
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
}

// And the memo in place of the root, over the WHOLE box a 3x3 window can
// produce -- `xx`, `yy` in [0, 9] and `xy` in [-9, 9], every one of the 1900
// codes, which is a superset of the 670 a plane pair can actually realize. The
// comparison is against `impl::minEigenValue` itself, by bit pattern, because
// the memo's whole claim is that it is the same function and not a close one.
BINCV_TEST(CudaSharedHelpers, MemoisedResponseIsMinEigenValueBitForBit) {
    std::vector<EigCase> cases;
    for (long long xx = 0; xx <= 9; ++xx)
        for (long long yy = 0; yy <= 9; ++yy)
            for (long long xy = -9; xy <= 9; ++xy) cases.push_back(EigCase{xx, yy, xy});

    DevArray<EigCase> dIn(cases);
    DevArray<float> dMemo(cases.size());
    DevArray<float> dLive(cases.size());
    kMemoEig<<<blocksFor(cases.size()), 256>>>(dIn.get(), cases.size(), dMemo.get());
    BINCV_CHECK(launchOk());
    kMinEig<<<blocksFor(cases.size()), 256>>>(dIn.get(), cases.size(), dLive.get());
    BINCV_CHECK(launchOk());
    const std::vector<float> memo = dMemo.download();
    const std::vector<float> live = dLive.download();

    size_t badDevice = 0, badHost = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        if (bits(memo[i]) != bits(live[i])) ++badDevice;
        const float want =
            bincv::impl::minEigenValue(cases[i].xx, cases[i].yy, cases[i].xy);
        if (bits(memo[i]) != bits(want)) ++badHost;
    }
    BINCV_CHECK_EQ(cases.size(), size_t{1900});
    BINCV_CHECK_EQ(badDevice, 0u);
    BINCV_CHECK_EQ(badHost, 0u);
}

BINCV_TEST(CudaSharedHelpers, ThresholdCutoffAgreesWithHost) {
    std::vector<double> cases;
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // The values with no caller and a defined answer are the point of the sweep:
    // NaN must land on "nothing passes" through `!(thresh < 255.0)` rather than
    // reaching a cast, and both infinities must land on the whole-image answers.
    for (double v : {-inf, inf, nan, -0.0, 0.0, -1e300, 1e300, -0.5, 0.5, 126.5, 127.0,
                     127.5, 254.0, 254.5, 254.999999, 255.0, 255.5, 1e9, -1e9})
        cases.push_back(v);
    for (int i = -20; i <= 280; ++i) {
        cases.push_back(static_cast<double>(i));
        cases.push_back(static_cast<double>(i) + 0.5);
        cases.push_back(static_cast<double>(i) - 0.0009765625);
    }

    DevArray<double> dIn(cases);
    DevArray<int> dOut(cases.size());
    kCutoff<<<blocksFor(cases.size()), 256>>>(dIn.get(), cases.size(), dOut.get());
    BINCV_CHECK(launchOk());
    const std::vector<int> got = dOut.download();

    size_t bad = 0;
    for (size_t i = 0; i < cases.size(); ++i)
        if (got[i] != bincv::impl::thresholdCutoff(cases[i])) ++bad;
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSharedHelpers, SquareInsideImageAgreesWithHost) {
    // The interesting values are all at the boundary: a square is inside when
    // `cx + half` is width-1 and outside at width, and the four edges have to
    // be tested independently or a transposed copy passes. Radii cover BRIEF's
    // reach and orientation's disc; the extremes check that nothing here
    // narrows to int on one target and not the other.
    std::vector<SquareCase> cases;
    const long long big = (1LL << 40);
    for (size_t width : {size_t{1}, size_t{7}, size_t{32}, size_t{33}, size_t{752}})
        for (size_t height : {size_t{1}, size_t{5}, size_t{31}, size_t{480}})
            for (int half : {0, 1, 3, 15, 16, 31, 64})
                for (long long cx : {-big, -1LL, 0LL,
                                     static_cast<long long>(half),
                                     static_cast<long long>(width) / 2,
                                     static_cast<long long>(width) - half - 1,
                                     static_cast<long long>(width) - half,
                                     static_cast<long long>(width) - 1,
                                     static_cast<long long>(width), big})
                    for (long long cy : {-1LL, 0LL,
                                         static_cast<long long>(half),
                                         static_cast<long long>(height) - half - 1,
                                         static_cast<long long>(height) - half,
                                         static_cast<long long>(height)})
                        cases.push_back(SquareCase{cx, cy, half, width, height});

    DevArray<SquareCase> dIn(cases);
    DevArray<unsigned char> dOut(cases.size());
    kSquare<<<blocksFor(cases.size()), 256>>>(dIn.get(), cases.size(), dOut.get());
    BINCV_CHECK(launchOk());
    const std::vector<unsigned char> got = dOut.download();

    size_t bad = 0;
    size_t inside = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const SquareCase& c = cases[i];
        const bool want =
            bincv::impl::squareInsideImage(c.cx, c.cy, c.half, c.width, c.height);
        if (want) ++inside;
        if ((got[i] != 0u) != want) ++bad;
    }
    // A sweep that accepts everything, or nothing, would pass the equality check
    // while testing nothing -- so both halves are required to be populated.
    std::printf("  squareInsideImage sweep: %zu cases, %zu inside\n", cases.size(),
                inside);
    BINCV_CHECK(inside > 0 && inside < cases.size());
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaSharedHelpers, QuantScaleAgreesWithHost) {
    // maxValue is `(1 << N) - 1` for N = 1..8, which is every N a QuantMat has.
    std::vector<QuantCase> c8, c16;
    for (unsigned n = 1; n <= 8; ++n) {
        const unsigned maxValue = (1u << n) - 1u;
        for (uint32_t v = 0; v <= 255u; ++v) c8.push_back(QuantCase{v, maxValue});
        for (uint32_t v = 0; v <= 65535u; ++v) c16.push_back(QuantCase{v, maxValue});
    }

    DevArray<QuantCase> d8In(c8);
    DevArray<unsigned> d8Out(c8.size());
    kQuant8<<<blocksFor(c8.size()), 256>>>(d8In.get(), c8.size(), d8Out.get());
    BINCV_CHECK(launchOk());
    const std::vector<unsigned> got8 = d8Out.download();

    DevArray<QuantCase> d16In(c16);
    DevArray<unsigned> d16Out(c16.size());
    kQuant16<<<blocksFor(c16.size()), 256>>>(d16In.get(), c16.size(), d16Out.get());
    BINCV_CHECK(launchOk());
    const std::vector<unsigned> got16 = d16Out.download();

    size_t bad = 0;
    for (size_t i = 0; i < c8.size(); ++i)
        if (got8[i] != bincv::impl::quantScale<uint8_t>(static_cast<uint8_t>(c8[i].v),
                                                        c8[i].maxValue))
            ++bad;
    for (size_t i = 0; i < c16.size(); ++i)
        if (got16[i] != bincv::impl::quantScale<uint16_t>(static_cast<uint16_t>(c16[i].v),
                                                          c16[i].maxValue))
            ++bad;
    BINCV_CHECK_EQ(bad, 0u);

    // THE ROUNDING FORM ITSELF, pinned on the DEVICE's answers rather than on
    // the shared source. The sweep above compares the two sides, so it passes
    // whenever they agree -- including if they agreed on a DIFFERENT rule,
    // because both would have moved together. These anchors say which rule.
    //
    // `(v * MaxValue + 127) / 255` rounds half-up; the obvious "simplification"
    // to `v * MaxValue / 255` truncates. The two differ at exactly the values
    // below, so a device that quietly lost the `+ 127` fails here:
    //   N = 1: v = 128 -> 1 half-up, 0 truncating
    //   N = 2: v = 128 -> 2 half-up, 1 truncating; v = 254 -> 3 vs 2
    // c8 is built as N = 1..8 outer, v = 0..255 inner, so N's block starts at
    // (N - 1) * 256.
    BINCV_CHECK_EQ(got8[128], 1u);         // N = 1, v = 128
    BINCV_CHECK_EQ(got8[127], 0u);         // N = 1, v = 127 -- the step is between them
    BINCV_CHECK_EQ(got8[256 + 128], 2u);   // N = 2, v = 128
    BINCV_CHECK_EQ(got8[256 + 254], 3u);   // N = 2, v = 254
    // The ends are exact at every N, which is what makes the scale a scale and
    // not an approximation of one.
    BINCV_CHECK_EQ(got8[0], 0u);
    BINCV_CHECK_EQ(got8[255], 1u);         // N = 1 saturates at the top byte
    BINCV_CHECK_EQ(got8[7 * 256 + 255], 255u);  // N = 8 is the identity
}

BINCV_TEST(CudaSharedHelpers, ClipRegionAgreesWithHost) {
    std::vector<ClipCase> cases;
    // Widths that do and do not end on a 32-bit word boundary, against rectangles
    // that are wholly inside, wholly outside on each side, straddling each edge,
    // degenerate, and negative-origin -- the clipping contract's whole domain.
    const size_t widths[] = {1, 31, 32, 33, 64, 97, 752};
    const size_t heights[] = {1, 2, 17, 480};
    const int xs[] = {-1000, -33, -1, 0, 1, 31, 32, 96, 751, 752, 1000};
    const int ws[] = {-5, 0, 1, 2, 31, 32, 33, 64, 2000};
    for (size_t w : widths) {
        for (size_t h : heights) {
            for (int x : xs) {
                for (int ww : ws) {
                    cases.push_back(ClipCase{w, h, bincv::Rect(x, -1, ww, 3)});
                    cases.push_back(ClipCase{w, h, bincv::Rect(x, 0, ww,
                                                               static_cast<int>(h))});
                    cases.push_back(
                        ClipCase{w, h, bincv::Rect(x, static_cast<int>(h) - 1, ww, 4)});
                    cases.push_back(ClipCase{w, h, bincv::Rect(x, 5000, ww, 2)});
                    cases.push_back(ClipCase{w, h, bincv::Rect(x, -5000, ww, 2)});
                }
            }
        }
    }

    DevArray<ClipCase> dIn(cases);
    DevArray<Region> dOut(cases.size());
    kClip<<<blocksFor(cases.size()), 256>>>(dIn.get(), cases.size(), dOut.get());
    BINCV_CHECK(launchOk());
    const std::vector<Region> got = dOut.download();

    size_t bad = 0, nonEmpty = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const Region want = bincv::impl::clipRegion<uint32_t>(cases[i].width,
                                                              cases[i].height,
                                                              cases[i].rect);
        if (!want.isEmpty) ++nonEmpty;
        if (!sameRegion(got[i], want)) ++bad;
    }
    BINCV_CHECK_EQ(bad, 0u);
    // A sweep that clipped everything to empty would compare equal on both sides
    // and prove nothing about the masks, so the count of surviving regions is
    // reported and floored. The floor is under the measured 2856 rather than at
    // it: this is a guard against the sweep collapsing, not a second baseline to
    // maintain.
    std::printf("  clipRegion sweep: %zu cases, %zu clipped to a non-empty region\n",
                cases.size(), nonEmpty);
    BINCV_CHECK(nonEmpty > 2000u);
}

BINCV_TEST(CudaSharedHelpers, AssertFailedHasALiveDeviceBranch) {
    // Anything but the trigger value -- see the note above the kernel.
    const std::vector<uint32_t> host{7u};
    DevArray<uint32_t> dIn(host);
    DevArray<uint32_t> dOut(size_t{1});
    kAssertBranch<<<1, 1>>>(dIn.get(), dOut.get());
    BINCV_CHECK(launchOk());
    // The kernel ran past the branch rather than trapping, and the context is
    // still usable -- which is also what makes the next suite's launches valid.
    BINCV_CHECK_EQ(dOut.download()[0], 8u);
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
    const int summaryRc = ::bincv::test::summarize("CUDA shared-helper tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA shared-helper tests", argc, argv);
}
#endif
