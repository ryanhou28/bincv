// The minimum-eigenvalue corner response and the good-features selection
// -- ops/corner.hpp.
//
// WHAT THIS SUITE HAS TO STAND BEHIND, GIVEN THAT NOTHING HERE IS BIT-EXACT
// AGAINST OPENCV
//
// The operation is API TIER 2: `cv::goodFeaturesToTrack`'s role with the
// reference pipeline's BINARIZED derivatives, and `cv::cornerMinEigenVal`'s
// response computed from exact integer popcounts rather than from a float box
// filter over float Sobel outputs. There is therefore no cv:: denominator to be
// bit-exact against and no Tier 1 promise anywhere in the file. Four things stand
// in its place, and each is a different kind of evidence:
//
// 1. A PER-PIXEL REFERENCE FOR THE RESPONSE MAP, written before the kernel and
// sharing no code with it: it reads each ternary value as a float in
// {-1, 0, +1} and accumulates `xx += a*a; yy += b*b; xy += a*b` one pixel at
// a time, with its own clipping ladder. Every pixel of every frame is
// compared, at four block sizes and all four word types, with NO TOLERANCE.
// 2. TWO EXACT PROPERTIES OF THE EIGENVALUE THAT NEED NO SQUARE ROOT AT ALL.
// The response is exactly 0.0f iff `det = xx*yy - xy^2` is zero, and where
// `D = (xx-yy)^2 + 4xy^2` is a perfect square the response is the exact
// half-integer `(S - isqrt(D))/2`. Both are checked with INTEGER arithmetic,
// so they do not inherit the library's own rounding, and between them they
// cover every zero-response position of every frame -- which on real content
// is most of the map.
// 3. THE SELECTION ORDER, against a literal port of gftt.cpp: `minMaxLoc`,
// `threshold(THRESH_TOZERO)`, `dilate` into a second buffer, the
// `val != 0 && val == tmp[x]` scan over `[1, h-1) x [1, w-1)`, the descending
// sort, and the CELL-GRID minimum-distance filter. ops/corner.hpp fuses the
// threshold into the scan and replaces the grid with an exhaustive check; the
// port does neither, so agreement is evidence that the two shortcuts are
// shortcuts and not changes. And a hand-built three-point map pins the ORDER
// itself: NMS-before-spacing keeps {A}, spacing-before-NMS keeps {A, C}, and
// the test asserts BOTH -- so the case can fail rather than merely agree.
// 4. STRUCTURE, not only random content. A checkerboard (a corner at every block
// junction), a 45-degree edge (the classic min-eigenvalue discriminator: an
// enormous gradient and NO corner), an isolated dot, a blank frame and a
// uniform frame.
//
// THE BORDER CHECK IS THE ONE THAT VERIFIES A DECISION RATHER THAN A KERNEL
//
// the design rule chose BORDER_REFLECT_101 for the derivative partly BECAUSE a zero fill
// manufactures an edge around the whole frame that THIS operation would select as
// spurious keypoints. `Corner.BorderRing_*` checks that the reasoning holds, and
// checks it in the only way that can fail: reflect-101 must give ZERO corners on a
// blank frame, a uniform frame and a striped frame, AND the same frames through a
// BORDER_CONSTANT derivative must give a ring. Measured at 41x37: uniform gives 4
// spurious corners (one per frame corner) and striped gives 12, all of them in the
// outermost two columns.
//
// WHAT ELSE IS PINNED HERE
//
// * NO HEAP, in any entry point. `operator new` is counted -- the plain AND the
// C++17 over-aligned forms -- across the response map, the selection and the
// whole operation, and must be zero. The selection is where this has teeth:
// the reference grows an unbounded `std::vector<const float*>` of candidates
// and allocates a `vector<vector<Point2f>>` grid, and ops/corner.hpp replaces
// both with the caller's own array.
// * THE MAP DOES NOT DEPEND ON WordType. The same logical frame at uint8_t,
// uint16_t, uint32_t and uint64_t must give BIT-IDENTICAL float maps; the
// popcount arithmetic is exact and the only rounding is a correctly-rounded
// square root, so anything else would be a word-boundary bug.
// * THE `float` STORAGE MARGIN IS MEASURED, NOT ASSERTED. ops/corner.hpp says
// the map is stored as float, that the headroom over `double` is ~1.3e4 at
// blockSize 3 and falls to ~2 by blockSize 31. `Corner.FloatMargin_*` counts
// the positions where the float map merges two responses the double oracle
// separates, and the NMS survivors that differ between the two maps, and
// prints both.
// * THE CAPACITY CONTRACT. Truncation is exercised, and `candidatesTruncated`
// must be set exactly when the buffer could not hold every NMS survivor.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/corner.hpp"
#include "bincv-cpp/ops/covariance.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "test_util.hpp"

// ---------------------------------------------------------------------------
// The allocation counter, in the idiom tests/test_storage.cpp established and
// tests/test_covariance.cpp reuses -- including the C++17 OVER-ALIGNED forms,
// which a counter replacing only the plain pair cannot see.
// ---------------------------------------------------------------------------
namespace {
std::size_t g_newCount = 0;

void* countedAllocate(std::size_t bytes) {
    ++g_newCount;
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    // Cannot throw std::bad_alloc: this file also builds with -fno-exceptions.
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}

void* countedAllocateAligned(std::size_t bytes, std::size_t alignment) {
    ++g_newCount;
    // The bound is not defensive decoration: without it gcc's
    // -Walloc-size-larger-than proves `rounded` can reach SIZE_MAX and FAILS the
    // -fno-exceptions configuration, which is the one build that inlines far
    // enough to see it. tests/test_covariance.cpp carries the same line.
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    const std::size_t wanted = (bytes == 0) ? 1 : bytes;
    const std::size_t rounded = ((wanted + alignment - 1) / alignment) * alignment;
    void* p = std::aligned_alloc(alignment, rounded);
    if (p == nullptr) std::abort();
    return p;
}

void countedFree(void* p) noexcept { std::free(p); }
} // namespace

void* operator new(std::size_t bytes)   { return countedAllocate(bytes); }
void* operator new[](std::size_t bytes) { return countedAllocate(bytes); }
void operator delete(void* p) noexcept                { countedFree(p); }
void operator delete[](void* p) noexcept              { countedFree(p); }
void operator delete(void* p, std::size_t) noexcept   { countedFree(p); }
void operator delete[](void* p, std::size_t) noexcept { countedFree(p); }

void* operator new(std::size_t bytes, std::align_val_t a) {
    return countedAllocateAligned(bytes, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t bytes, std::align_val_t a) {
    return countedAllocateAligned(bytes, static_cast<std::size_t>(a));
}
void operator delete(void* p, std::align_val_t) noexcept                { countedFree(p); }
void operator delete[](void* p, std::align_val_t) noexcept              { countedFree(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept   { countedFree(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { countedFree(p); }

namespace {

using bincv::BinMat;
using bincv::BinMatConstView;
using bincv::BorderType;
using bincv::BORDER_CONSTANT;
using bincv::BORDER_REFLECT_101;
using bincv::Corner;
using bincv::CornerResult;
using bincv::ConstResponseMap;
using bincv::GoodFeaturesParams;
using bincv::GradientCovariance;
using bincv::Rect;
using bincv::ResponseMap;
using bincv::TernaryMat;

// ---------------------------------------------------------------------------
// FRAMES. Built as plain bit arrays first, so that the SAME logical image can be
// packed at all four word types and the maps compared bit for bit.
// ---------------------------------------------------------------------------

// Taller than the largest block size, and pinned. shipped a suite whose
// frame was shorter than two of its three window sizes, so every swept position
// of the two largest windows was clipped and nothing ever reduced a full window.
// The same mistake is available here and these two lines are what prevent it.
constexpr int kMaxBlockSize = 31;
constexpr int kWideW = 71, kWideH = 45;
constexpr int kNarrowW = 40, kNarrowH = 35;
static_assert(kWideH > kMaxBlockSize, "the wide frame must be taller than the largest block");
static_assert(kNarrowH > kMaxBlockSize, "the narrow frame must be taller than the largest block");
static_assert(kNarrowW > kMaxBlockSize, "the narrow frame must be wider than the largest block");

struct Frame {
    std::string name;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> bits;  // one byte per pixel, 0 or 1

    unsigned at(int y, int x) const {
        return bits[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
    }
    void set(int y, int x, unsigned v) {
        bits[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
            static_cast<uint8_t>(v);
    }
};

uint64_t nextRandom(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

Frame makeFrame(const std::string& name, int w, int h) {
    Frame f;
    f.name = name;
    f.width = w;
    f.height = h;
    f.bits.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
    return f;
}

Frame checkerboardFrame(int w, int h, int block) {
    Frame f = makeFrame("checkerboard", w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) f.set(y, x, ((x / block + y / block) % 2) ? 1u : 0u);
    return f;
}

// A 45-degree step edge: the classic case a min-eigenvalue response must REJECT.
// Every gradient inside an interior window is parallel, so the matrix is rank one,
// `det` is zero, and the response is exactly zero -- however large the gradient is.
Frame diagonalFrame(int w, int h) {
    Frame f = makeFrame("diagonal", w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) f.set(y, x, (x > y) ? 1u : 0u);
    return f;
}

Frame randomFrame(int w, int h, uint64_t seed) {
    Frame f = makeFrame("random", w, h);
    uint64_t state = seed;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) f.set(y, x, static_cast<unsigned>(nextRandom(state) & 1ULL));
    return f;
}

Frame uniformFrame(int w, int h, unsigned value) {
    Frame f = makeFrame(value ? "uniform-ones" : "blank", w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) f.set(y, x, value);
    return f;
}

Frame stripeFrame(int w, int h, int period) {
    Frame f = makeFrame("stripes", w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) f.set(y, x, ((y / period) % 2) ? 1u : 0u);
    return f;
}

Frame dotFrame(int w, int h) {
    Frame f = makeFrame("dot", w, h);
    f.set(h / 2, w / 2, 1u);
    return f;
}

template <typename WordType>
BinMat<WordType> pack(const Frame& f) {
    BinMat<WordType> m(f.width, f.height);
    for (int y = 0; y < f.height; ++y)
        for (int x = 0; x < f.width; ++x) m.set(y, x, f.at(y, x));
    return m;
}

// ---------------------------------------------------------------------------
// THE PER-PIXEL REFERENCE.
//
// It reads the derivative pair through the CONTAINER accessor, one pixel at a
// time, into plain signed bytes -- a path that touches no view, no region clip and
// no word arithmetic -- and then accumulates in FLOAT, which is the formulation
// the design notes claims the popcounts replace.
// ---------------------------------------------------------------------------

struct Ternary {
    int width = 0;
    int height = 0;
    std::vector<int8_t> value;  // -1, 0 or +1

    int at(int y, int x) const {
        return value[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
    }
};

template <typename WordType>
Ternary readTernary(const TernaryMat<WordType>& t) {
    Ternary out;
    out.width = t.cols();
    out.height = t.rows();
    out.value.assign(static_cast<size_t>(out.width) * static_cast<size_t>(out.height), 0);
    for (int y = 0; y < out.height; ++y)
        for (int x = 0; x < out.width; ++x)
            out.value[static_cast<size_t>(y) * static_cast<size_t>(out.width) +
                      static_cast<size_t>(x)] = static_cast<int8_t>(t.at(y, x));
    return out;
}

struct Triple {
    long long xx = 0;
    long long yy = 0;
    long long xy = 0;
};

Triple oracleTriple(const Ternary& a, const Ternary& b, int x, int y, int blockSize) {
    const int off = blockSize / 2;
    float xx = 0.0f, yy = 0.0f, xy = 0.0f;
    for (int j = y - off; j < y - off + blockSize; ++j) {
        if (j < 0 || j >= a.height) continue;
        for (int i = x - off; i < x - off + blockSize; ++i) {
            if (i < 0 || i >= a.width) continue;
            const float va = static_cast<float>(a.at(j, i));
            const float vb = static_cast<float>(b.at(j, i));
            xx += va * va;
            yy += vb * vb;
            xy += va * vb;
        }
    }
    Triple t;
    t.xx = static_cast<long long>(xx);
    t.yy = static_cast<long long>(yy);
    t.xy = static_cast<long long>(xy);
    return t;
}

/// @brief floor(sqrt(n)) for n >= 0, exact -- no reliance on the library's sqrt.
long long isqrtExact(long long n) {
    if (n <= 0) return 0;
    long long r = static_cast<long long>(std::sqrt(static_cast<double>(n)));
    while (r > 0 && r * r > n) --r;
    while ((r + 1) * (r + 1) <= n) ++r;
    return r;
}

/// @brief The response in DOUBLE, for the float-margin measurement. Same formula,
/// stored without the narrowing.
double responseDouble(const Triple& t) {
    const double s = static_cast<double>(t.xx) + static_cast<double>(t.yy);
    const double d = static_cast<double>(t.xx) - static_cast<double>(t.yy);
    const double c = static_cast<double>(t.xy);
    return 0.5 * (s - std::sqrt(d * d + 4.0 * c * c));
}

// ---------------------------------------------------------------------------
// A LITERAL PORT OF gftt.cpp's SELECTION -- second buffer, dilate, cell grid.
// ---------------------------------------------------------------------------

/// @brief cvRound: nearest, ties to even. `std::nearbyint` under the default
/// FE_TONEAREST is exactly that.
int cvRoundLike(double v) { return static_cast<int>(std::nearbyint(v)); }

/// @brief gftt.cpp's own comparator, copied from
/// SEAL/opencv_internal/include/gftt.hpp with its comment.
/// @note **THIS MUST NOT BE `impl::CornerStronger` OR A COPY OF IT.** The library's
/// comparator is the thing under test; an oracle that shares it cannot
/// detect a wrong tie order, and equal responses are the RULE in this
/// operation rather than the exception. So the port sorts POINTERS into the
/// `eig` buffer, exactly as the reference does, and its tie rule is the
/// reference's address comparison rather than anything spelled on
/// coordinates. Measured: with the library's tie order inverted this
/// comparator fails `Corner.SelectionMatchesReferencePort` and
/// `Corner.SelectionOnSyntheticMaps`; a copy of `impl::CornerStronger`
/// passes both.
struct greaterThanPtr {
    bool operator()(const float* a, const float* b) const

    // Ensure a fully deterministic result of the sort
    { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
};

std::vector<Corner> referenceSelect(const std::vector<float>& eigIn, int w, int h,
                                    const GoodFeaturesParams& p, size_t* candidatesOut) {
    std::vector<Corner> corners;
    if (w <= 0 || h <= 0) {
        if (candidatesOut != nullptr) *candidatesOut = 0;
        return corners;
    }
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);

    // 1. minMaxLoc over the whole map.
    float maxVal = eigIn[0];
    for (size_t i = 0; i < n; ++i)
        if (eigIn[i] > maxVal) maxVal = eigIn[i];

    // 2. threshold(THRESH_TOZERO), strictly greater, threshold narrowed to float.
    std::vector<float> eig = eigIn;
    const float thr = static_cast<float>(static_cast<double>(maxVal) * p.qualityLevel);
    for (size_t i = 0; i < n; ++i)
        if (!(eig[i] > thr)) eig[i] = 0.0f;

    // 3. dilate into a SECOND buffer, then the val != 0 && val == tmp[x] scan.
    std::vector<float> tmp = eig;
    for (int y = 1; y + 1 < h; ++y) {
        for (int x = 1; x + 1 < w; ++x) {
            float m = eig[static_cast<size_t>(y) * static_cast<size_t>(w) +
                          static_cast<size_t>(x)];
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const float v = eig[static_cast<size_t>(y + dy) * static_cast<size_t>(w) +
                                        static_cast<size_t>(x + dx)];
                    if (v > m) m = v;
                }
            tmp[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = m;
        }
    }
    // Pointers INTO `eig`, pushed in ascending raster order -- gftt.cpp's
    // `std::vector<const float*> tmpCorners` verbatim, because the sort's tie rule
    // is a comparison of those very addresses.
    std::vector<const float*> cand;
    for (int y = 1; y + 1 < h; ++y) {
        for (int x = 1; x + 1 < w; ++x) {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(w) +
                             static_cast<size_t>(x);
            if (eig[i] != 0.0f && eig[i] == tmp[i]) cand.push_back(eig.data() + i);
        }
    }
    if (candidatesOut != nullptr) *candidatesOut = cand.size();
    if (cand.empty()) return corners;

    // 4. sort descending, then the CELL GRID spacing filter.
    std::sort(cand.begin(), cand.end(), greaterThanPtr());

    if (p.minDistance >= 1.0) {
        const int cell = cvRoundLike(p.minDistance);
        const int gw = (w + cell - 1) / cell;
        const int gh = (h + cell - 1) / cell;
        std::vector<std::vector<Corner>> grid(static_cast<size_t>(gw) * static_cast<size_t>(gh));
        const double minDistSq = p.minDistance * p.minDistance;
        for (size_t i = 0; i < cand.size(); ++i) {
            // gftt.cpp recovers (x, y) from the pointer offset. Same arithmetic.
            const size_t ofs = static_cast<size_t>(cand[i] - eig.data());
            const int y = static_cast<int>(ofs / static_cast<size_t>(w));
            const int x = static_cast<int>(ofs % static_cast<size_t>(w));
            Corner here;
            here.x = x;
            here.y = y;
            here.response = *cand[i];
            const int xc = x / cell, yc = y / cell;
            bool good = true;
            const int x1 = std::max(0, xc - 1), y1 = std::max(0, yc - 1);
            const int x2 = std::min(gw - 1, xc + 1), y2 = std::min(gh - 1, yc + 1);
            for (int yy = y1; yy <= y2 && good; ++yy) {
                for (int xx = x1; xx <= x2 && good; ++xx) {
                    const std::vector<Corner>& m =
                        grid[static_cast<size_t>(yy) * static_cast<size_t>(gw) +
                             static_cast<size_t>(xx)];
                    for (size_t j = 0; j < m.size(); ++j) {
                        const double dx = static_cast<double>(x) - static_cast<double>(m[j].x);
                        const double dy = static_cast<double>(y) - static_cast<double>(m[j].y);
                        if (dx * dx + dy * dy < minDistSq) {
                            good = false;
                            break;
                        }
                    }
                }
            }
            if (good) {
                grid[static_cast<size_t>(yc) * static_cast<size_t>(gw) + static_cast<size_t>(xc)]
                    .push_back(here);
                corners.push_back(here);
                if (p.maxCorners > 0 &&
                    corners.size() == static_cast<size_t>(p.maxCorners))
                    break;
            }
        }
    } else {
        for (size_t i = 0; i < cand.size(); ++i) {
            const size_t ofs = static_cast<size_t>(cand[i] - eig.data());
            Corner here;
            here.y = static_cast<int>(ofs / static_cast<size_t>(w));
            here.x = static_cast<int>(ofs % static_cast<size_t>(w));
            here.response = *cand[i];
            corners.push_back(here);
            if (p.maxCorners > 0 && corners.size() == static_cast<size_t>(p.maxCorners)) break;
        }
    }
    return corners;
}

// ---------------------------------------------------------------------------
// Shared plumbing for a frame: derivatives, the map, and the map's oracle.
// ---------------------------------------------------------------------------

template <typename WordType>
struct Derived {
    TernaryMat<WordType> dx;
    TernaryMat<WordType> dy;
    Ternary refX;
    Ternary refY;

    Derived(const Frame& f, BorderType border)
        : dx(f.width, f.height), dy(f.width, f.height) {
        const BinMat<WordType> src = pack<WordType>(f);
        bincv::derivativeX(src, dx, border);
        bincv::derivativeY(src, dy, border);
        refX = readTernary(dx);
        refY = readTernary(dy);
    }
};

std::vector<float> makeMapStorage(int w, int h) {
    return std::vector<float>(static_cast<size_t>(w) * static_cast<size_t>(h), -1.0f);
}

ResponseMap mapView(std::vector<float>& storage, int w, int h) {
    ResponseMap m;
    m.data = storage.data();
    m.width = static_cast<size_t>(w);
    m.height = static_cast<size_t>(h);
    m.stride = static_cast<size_t>(w);
    return m;
}

const int kBlockSizes[] = {3, 7, 15, 31};

} // namespace

// ---------------------------------------------------------------------------
// 1. THE RESPONSE MAP AGAINST THE PER-PIXEL REFERENCE
// ---------------------------------------------------------------------------

namespace {

struct SweepTally {
    size_t positions = 0;        ///< every pixel of the frame, at this block size
    size_t interior = 0;         ///< positions whose window lies fully inside the frame
    size_t clipped = 0;          ///< positions whose window is cut by an edge
    size_t valueMismatch = 0;    ///< map != minEigenValue(reference triple)
    size_t tripleMismatch = 0;   ///< reference triple != that work’s gradientCovariance
    size_t exactChecked = 0;     ///< positions where D is a perfect square
    size_t exactMismatch = 0;    ///<... and the map is not the exact half-integer
    size_t zeroMismatch = 0;     ///< (response == 0) != (det == 0)
    size_t gapViolation = 0;     ///< a non-zero response below 1/(2*blockSize^2)
    size_t precisionMismatch = 0; ///< map != narrow(double evaluation of the same triple)
};

template <typename WordType>
SweepTally sweepFrame(const Frame& f, int blockSize) {
    const Derived<WordType> d(f, BORDER_REFLECT_101);
    std::vector<float> storage = makeMapStorage(f.width, f.height);
    ResponseMap map = mapView(storage, f.width, f.height);
    bincv::cornerMinEigenVal(d.dx, d.dy, blockSize, map);

    const int off = blockSize / 2;
    SweepTally t;
    for (int y = 0; y < f.height; ++y) {
        for (int x = 0; x < f.width; ++x) {
            ++t.positions;
            const bool inside = (x - off >= 0) && (y - off >= 0) &&
                                (x - off + blockSize <= f.width) &&
                                (y - off + blockSize <= f.height);
            if (inside) ++t.interior; else ++t.clipped;

            const Triple ref = oracleTriple(d.refX, d.refY, x, y, blockSize);
            const float got = storage[static_cast<size_t>(y) * static_cast<size_t>(f.width) +
                                      static_cast<size_t>(x)];

            // (a) the map is the eigenvalue of the reference's triple, exactly.
            // This pins the TRIPLE -- it calls the same function the kernel
            // does, so it says nothing about the eigenvalue formula itself.
            if (got != bincv::impl::minEigenValue(ref.xx, ref.yy, ref.xy)) ++t.valueMismatch;

            // (a2)... and the map is what a DOUBLE evaluation of the same triple
            // narrows to. ops/corner.hpp commits to double for the square root
            // and says why; without this the commitment is unbacked, and a
            // mutant computing the whole thing in float passed the rest of this
            // suite unchanged.
            if (got != static_cast<float>(responseDouble(ref))) ++t.precisionMismatch;

            // (b) the reference's triple is that work’s, exactly. This is what tells a
            // reader WHICH side is wrong when (a) fails: the sliding sweep or
            // the oracle.
            const GradientCovariance cov = bincv::gradientCovariance(
                d.dx, d.dy, Rect(x - off, y - off, blockSize, blockSize));
            if (cov.sumXX != ref.xx || cov.sumYY != ref.yy || cov.sumXY != ref.xy)
                ++t.tripleMismatch;

            // (c) where D is a perfect square the answer is an exact half-integer
            // and needs no floating square root to predict.
            const long long S = ref.xx + ref.yy;
            const long long D = (ref.xx - ref.yy) * (ref.xx - ref.yy) + 4 * ref.xy * ref.xy;
            const long long q = isqrtExact(D);
            if (q * q == D) {
                ++t.exactChecked;
                const float expected = static_cast<float>(0.5 * static_cast<double>(S - q));
                if (got != expected) ++t.exactMismatch;
            }

            // (d) zero response iff singular matrix -- integer arithmetic only.
            const long long det = ref.xx * ref.yy - ref.xy * ref.xy;
            if ((got == 0.0f) != (det == 0)) ++t.zeroMismatch;

            // (e) a non-zero response is at least 1/(2*blockSize^2), so it can
            // never round to zero and the `> threshold` test needs no epsilon.
            if (got != 0.0f) {
                const double floorValue =
                    1.0 / (2.0 * static_cast<double>(blockSize) * static_cast<double>(blockSize));
                if (static_cast<double>(got) < floorValue * 0.999) ++t.gapViolation;
            }
        }
    }
    return t;
}

template <typename WordType>
void sweepSuite(const char* wordName) {
    const Frame frames[] = {checkerboardFrame(kWideW, kWideH, 4), diagonalFrame(kWideW, kWideH),
                            randomFrame(kWideW, kWideH, 0x51ED5EEDULL),
                            checkerboardFrame(kNarrowW, kNarrowH, 3),
                            randomFrame(kNarrowW, kNarrowH, 0xC0FFEEULL)};
    size_t positions = 0, interior = 0, clipped = 0, exact = 0;
    for (const Frame& f : frames) {
        for (int blockSize : kBlockSizes) {
            const SweepTally t = sweepFrame<WordType>(f, blockSize);
            BINCV_CHECK_EQ(t.valueMismatch, static_cast<size_t>(0));
            BINCV_CHECK_EQ(t.tripleMismatch, static_cast<size_t>(0));
            BINCV_CHECK_EQ(t.exactMismatch, static_cast<size_t>(0));
            BINCV_CHECK_EQ(t.zeroMismatch, static_cast<size_t>(0));
            BINCV_CHECK_EQ(t.gapViolation, static_cast<size_t>(0));
            BINCV_CHECK_EQ(t.precisionMismatch, static_cast<size_t>(0));
            // Both kinds of coverage are required, not just counted: a sweep that
            // never clipped, or never evaluated a full window, would still report
            // zero mismatches.
            BINCV_CHECK(t.interior > 0);
            BINCV_CHECK(t.clipped > 0);
            positions += t.positions;
            interior += t.interior;
            clipped += t.clipped;
            exact += t.exactChecked;
        }
    }
    // The position count is itself a checked quantity: the one failure mode a
    // check count cannot see is a sweep that quietly got smaller.
    size_t expectedPositions = 0;
    for (const Frame& f : frames)
        expectedPositions += static_cast<size_t>(f.width) * static_cast<size_t>(f.height) *
                             (sizeof(kBlockSizes) / sizeof(kBlockSizes[0]));
    BINCV_CHECK_EQ(positions, expectedPositions);
    BINCV_CHECK(interior > 0 && clipped > 0 && exact > 0);
    std::printf(" [%s] %zu positions (%zu interior, %zu clipped), %zu with an exact square root\n",
                wordName, positions, interior, clipped, exact);
}

} // namespace

BINCV_TEST(Corner, ResponseMap_uint8_t) { sweepSuite<uint8_t>("uint8_t"); }
BINCV_TEST(Corner, ResponseMap_uint16_t) { sweepSuite<uint16_t>("uint16_t"); }
BINCV_TEST(Corner, ResponseMap_uint32_t) { sweepSuite<uint32_t>("uint32_t"); }
BINCV_TEST(Corner, ResponseMap_uint64_t) { sweepSuite<uint64_t>("uint64_t"); }

// ---------------------------------------------------------------------------
// 2. THE MAP DOES NOT DEPEND ON THE WORD TYPE
//
// The three sums are exact integers and the only rounding is a correctly-rounded
// square root, so the same logical frame must give BIT-IDENTICAL float maps at 8,
// 16, 32 and 64 bits. Anything else is a word-boundary bug -- a residue the column
// masks get wrong, or a padding bit counted -- and it would be invisible to
// a suite that only ever ran one width.
// ---------------------------------------------------------------------------

namespace {

template <typename WordType>
std::vector<float> mapFor(const Frame& f, int blockSize, BorderType border) {
    const Derived<WordType> d(f, border);
    std::vector<float> storage = makeMapStorage(f.width, f.height);
    ResponseMap map = mapView(storage, f.width, f.height);
    bincv::cornerMinEigenVal(d.dx, d.dy, blockSize, map);
    return storage;
}

} // namespace

BINCV_TEST(Corner, WordTypeInvariance) {
    const Frame frames[] = {checkerboardFrame(kWideW, kWideH, 4), diagonalFrame(kWideW, kWideH),
                            randomFrame(kWideW, kWideH, 0xA5A5A5A5ULL),
                            randomFrame(kNarrowW, kNarrowH, 0x1234ULL)};
    size_t compared = 0;
    for (const Frame& f : frames) {
        for (int blockSize : kBlockSizes) {
            const std::vector<float> m32 = mapFor<uint32_t>(f, blockSize, BORDER_REFLECT_101);
            const std::vector<float> m8 = mapFor<uint8_t>(f, blockSize, BORDER_REFLECT_101);
            const std::vector<float> m16 = mapFor<uint16_t>(f, blockSize, BORDER_REFLECT_101);
            const std::vector<float> m64 = mapFor<uint64_t>(f, blockSize, BORDER_REFLECT_101);
            size_t differing = 0;
            for (size_t i = 0; i < m32.size(); ++i) {
                if (m8[i] != m32[i] || m16[i] != m32[i] || m64[i] != m32[i]) ++differing;
                ++compared;
            }
            BINCV_CHECK_EQ(differing, static_cast<size_t>(0));
        }
    }
    BINCV_CHECK(compared > 0);
    std::printf(" word-type invariance: %zu positions x 4 word types, bit-identical\n", compared);
}

// ---------------------------------------------------------------------------
// 3. THE `float` STORAGE MARGIN, MEASURED
//
// ops/corner.hpp says the map is stored as float, that the response is computed in
// double, and that the headroom shrinks with blockSize -- ~1.3e4 at 3, ~2 by 31.
// A claim shaped like that has to be a measurement. This counts the two things
// that would go wrong if the margin ran out: 3x3 neighbour pairs the float map
// makes EQUAL that the double values separate (a manufactured NMS tie), and NMS
// survivors that differ between the two maps.
// ---------------------------------------------------------------------------

BINCV_TEST(Corner, FloatMargin) {
    const Frame frames[] = {randomFrame(kWideW, kWideH, 0xBEEF01ULL),
                            checkerboardFrame(kWideW, kWideH, 5)};
    for (int blockSize : kBlockSizes) {
        size_t merged = 0, survivorDiff = 0, positions = 0;
        for (const Frame& f : frames) {
            const Derived<uint32_t> d(f, BORDER_REFLECT_101);
            std::vector<float> storage = makeMapStorage(f.width, f.height);
            ResponseMap map = mapView(storage, f.width, f.height);
            bincv::cornerMinEigenVal(d.dx, d.dy, blockSize, map);

            std::vector<double> exact(static_cast<size_t>(f.width) * static_cast<size_t>(f.height),
                                      0.0);
            for (int y = 0; y < f.height; ++y)
                for (int x = 0; x < f.width; ++x)
                    exact[static_cast<size_t>(y) * static_cast<size_t>(f.width) +
                          static_cast<size_t>(x)] =
                        responseDouble(oracleTriple(d.refX, d.refY, x, y, blockSize));

            for (int y = 1; y + 1 < f.height; ++y) {
                for (int x = 1; x + 1 < f.width; ++x) {
                    ++positions;
                    const size_t i = static_cast<size_t>(y) * static_cast<size_t>(f.width) +
                                     static_cast<size_t>(x);
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            const size_t j = static_cast<size_t>(y + dy) *
                                                 static_cast<size_t>(f.width) +
                                             static_cast<size_t>(x + dx);
                            if (storage[i] == storage[j] && exact[i] != exact[j]) ++merged;
                        }
                    }
                }
            }

            // NMS survivors under the float map against the double map. Both use
            // the same threshold rule, so a difference is purely the storage type.
            float maxF = storage[0];
            double maxD = exact[0];
            for (size_t i = 0; i < storage.size(); ++i) {
                if (storage[i] > maxF) maxF = storage[i];
                if (exact[i] > maxD) maxD = exact[i];
            }
            const float thrF = static_cast<float>(static_cast<double>(maxF) * 0.01);
            const double thrD = maxD * 0.01;
            for (int y = 1; y + 1 < f.height; ++y) {
                for (int x = 1; x + 1 < f.width; ++x) {
                    const size_t i = static_cast<size_t>(y) * static_cast<size_t>(f.width) +
                                     static_cast<size_t>(x);
                    bool okF = storage[i] > thrF, okD = exact[i] > thrD;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const size_t j = static_cast<size_t>(y + dy) *
                                                 static_cast<size_t>(f.width) +
                                             static_cast<size_t>(x + dx);
                            if (storage[j] > storage[i]) okF = false;
                            if (exact[j] > exact[i]) okD = false;
                        }
                    if (okF != okD) ++survivorDiff;
                }
            }
        }
        std::printf(" float margin at blockSize %2d: %zu positions, %zu merged neighbour pairs, "
                    "%zu NMS survivor differences\n",
                    blockSize, positions, merged, survivorDiff);
        BINCV_CHECK(positions > 0);
        if (blockSize == 3) {
            // The reference pipeline's block size. The headroom there is ~1.3e4,
            // so a merge would mean the analysis in ops/corner.hpp is wrong.
            BINCV_CHECK_EQ(merged, static_cast<size_t>(0));
            BINCV_CHECK_EQ(survivorDiff, static_cast<size_t>(0));
        }
    }
}

// ---------------------------------------------------------------------------
// 4. THE SELECTION, AGAINST A LITERAL PORT OF gftt.cpp
// ---------------------------------------------------------------------------

namespace {

void compareSelection(const std::vector<float>& storage, int w, int h,
                      const GoodFeaturesParams& p, size_t capacity, size_t& positionsOut) {
    ConstResponseMap view(storage.data(), static_cast<size_t>(w), static_cast<size_t>(h),
                          static_cast<size_t>(w));
    std::vector<Corner> got(capacity);
    const CornerResult r = bincv::selectGoodFeatures(view, p, got.data(), capacity);

    size_t refCandidates = 0;
    const std::vector<Corner> want = referenceSelect(storage, w, h, p, &refCandidates);

    BINCV_CHECK_EQ(r.candidatesRanked, refCandidates);
    BINCV_CHECK_EQ(r.candidatesTruncated, false);
    BINCV_CHECK_EQ(r.count, want.size());
    size_t differing = 0;
    for (size_t i = 0; i < r.count && i < want.size(); ++i) {
        if (got[i].x != want[i].x || got[i].y != want[i].y || got[i].response != want[i].response)
            ++differing;
    }
    BINCV_CHECK_EQ(differing, static_cast<size_t>(0));
    positionsOut += r.count;
}

} // namespace

BINCV_TEST(Corner, SelectionMatchesReferencePort) {
    const Frame frames[] = {randomFrame(kWideW, kWideH, 0x777ULL),
                            checkerboardFrame(kWideW, kWideH, 4),
                            checkerboardFrame(kWideW, kWideH, 7),
                            diagonalFrame(kWideW, kWideH),
                            randomFrame(kNarrowW, kNarrowH, 0x999ULL)};
    // Spacings chosen around the reference's own 33.33: one below 1 (which turns
    // the filter off entirely, gftt.cpp's `else` branch), one at exactly 1, a
    // fractional one whose cvRound goes DOWN and one whose cvRound goes UP -- the
    // grid's cell size is `cvRound(minDistance)`, so those two are different code
    // paths in the port this is checked against.
    const double spacings[] = {0.0, 0.5, 1.0, 2.5, 3.4, 3.6, 8.0, 33.33333333333};
    const int caps[] = {0, 1, 5, 200};
    size_t total = 0, combos = 0;
    for (const Frame& f : frames) {
        for (int blockSize : {3, 7}) {
            const std::vector<float> storage = mapFor<uint32_t>(f, blockSize, BORDER_REFLECT_101);
            for (double spacing : spacings) {
                for (int cap : caps) {
                    GoodFeaturesParams p;
                    p.blockSize = blockSize;
                    p.minDistance = spacing;
                    p.maxCorners = cap;
                    p.qualityLevel = 0.01;
                    compareSelection(storage, f.width, f.height, p,
                                     static_cast<size_t>(f.width) * static_cast<size_t>(f.height),
                                     total);
                    ++combos;
                }
            }
        }
    }
    BINCV_CHECK(combos > 0);
    std::printf(" selection: %zu parameter combinations, %zu corners agreed with the gftt.cpp "
                "port\n",
                combos, total);
}

// ---------------------------------------------------------------------------
// 5. THE ORDER OF THE STAGES, PINNED BY A CASE WHERE IT MATTERS
//
// NMS's kills are NOT a subset of the spacing filter's: NMS removes a point beside
// a HIGHER one whether or not that higher one is ever accepted. This map is built
// so that the difference is visible, and both answers are asserted -- the one this
// file produces AND the one the other order would produce -- so the case fails if
// either changes.
//
// A = (10, 10) 100 B = (13, 10) 99 C = (14, 10) 98 minDistance 3.5
//
// NMS first (gftt.cpp, and ops/corner.hpp): C dies beside B; {A, B} rank; A is
// accepted; B is 3 away and 9 < 12.25 rejects it. -> {A}
// Spacing first: all three rank; A accepted; B rejected; C is 4 away and
// 16 >= 12.25 accepts it. -> {A, C}
// ---------------------------------------------------------------------------

// Real response maps are not enough to pin the selection, because they never put
// the global maximum where the candidate scan cannot reach it: reflect-101 makes
// the outermost row and column of the derivative zero, so a real map's
// border is weak. `minMaxLoc` in gftt.cpp scans the WHOLE map, border included,
// and the threshold it sets is what every later stage is measured against -- so a
// maximum living on the border is a case only a synthetic map can build. Measured:
// a mutant taking the maximum over the interior only passed every other case in
// this file. These maps also carry heavy TIES -- `rand % 6`, so equal responses are
// the rule -- which is where `impl::CornerStronger`'s tie rule has to agree with
// the port's `greaterThanPtr`. The port sorts POINTERS and breaks ties by address,
// so it cannot agree with an inverted tie order by construction: measured, flipping
// the library's tie direction fails this suite.
BINCV_TEST(Corner, SelectionOnSyntheticMaps) {
    const double spacings[] = {0.0, 1.0, 2.5, 3.4, 3.6, 9.0};
    const int caps[] = {0, 2, 7, 500};
    size_t combos = 0, corners = 0, borderDominant = 0;
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        const int w = 43, h = 31;
        std::vector<float> storage(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
        uint64_t state = seed * 0x9E3779B97F4A7C15ULL;
        for (size_t i = 0; i < storage.size(); ++i) {
            // Small integers, so ties are the rule rather than the exception.
            storage[i] = static_cast<float>(nextRandom(state) % 6ULL);
        }
        // Seeds 4..6 put the map's maximum on the border, where the candidate scan
        // cannot see it but `minMaxLoc` must.
        if (seed >= 4) {
            storage[static_cast<size_t>(seed) % static_cast<size_t>(w)] = 40.0f;
            ++borderDominant;
        }
        for (double spacing : spacings) {
            for (int cap : caps) {
                GoodFeaturesParams p;
                p.blockSize = 3;
                p.qualityLevel = (seed >= 4) ? 0.1 : 0.01;
                p.minDistance = spacing;
                p.maxCorners = cap;
                compareSelection(storage, w, h, p,
                                 static_cast<size_t>(w) * static_cast<size_t>(h), corners);
                ++combos;
            }
        }
    }
    BINCV_CHECK_EQ(borderDominant, static_cast<size_t>(3));
    BINCV_CHECK(combos > 0);
    std::printf(" synthetic maps: %zu parameter combinations (3 with the maximum on the "
                "border), %zu corners agreed with the gftt.cpp port\n",
                combos, corners);
}

BINCV_TEST(Corner, SelectionOrder_PinsNmsBeforeDistance) {
    const int w = 24, h = 21;
    std::vector<float> storage(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
    auto put = [&](int x, int y, float v) {
        storage[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = v;
    };
    put(10, 10, 100.0f);
    put(13, 10, 99.0f);
    put(14, 10, 98.0f);

    GoodFeaturesParams p;
    p.blockSize = 3;
    p.qualityLevel = 0.01;  // threshold 1.0; all three survive it
    p.minDistance = 3.5;
    p.maxCorners = 0;

    ConstResponseMap view(storage.data(), static_cast<size_t>(w), static_cast<size_t>(h),
                          static_cast<size_t>(w));
    std::vector<Corner> got(64);
    const CornerResult r = bincv::selectGoodFeatures(view, p, got.data(), got.size());

    // NMS ranks A and B; C is beside B and dies.
    BINCV_CHECK_EQ(r.candidatesRanked, static_cast<size_t>(2));
    BINCV_CHECK_EQ(r.count, static_cast<size_t>(1));
    BINCV_CHECK_EQ(got[0].x, 10);
    BINCV_CHECK_EQ(got[0].y, 10);

    // The same map through the OTHER order, computed here so that the case has
    // teeth: if this produced {A} too, the assertion above would prove nothing.
    std::vector<Corner> all;
    for (int y = 1; y + 1 < h; ++y)
        for (int x = 1; x + 1 < w; ++x) {
            const float v = storage[static_cast<size_t>(y) * static_cast<size_t>(w) +
                                    static_cast<size_t>(x)];
            if (v > 1.0f) {
                Corner c;
                c.x = x;
                c.y = y;
                c.response = v;
                all.push_back(c);
            }
        }
    // Responses here are 100, 99 and 98 -- all distinct -- so this needs no tie
    // rule and deliberately does not borrow one from the library.
    std::sort(all.begin(), all.end(),
              [](const Corner& a, const Corner& b) { return a.response > b.response; });
    std::vector<Corner> spacingFirst;
    for (size_t i = 0; i < all.size(); ++i) {
        bool good = true;
        for (size_t j = 0; j < spacingFirst.size(); ++j) {
            const double dx = static_cast<double>(all[i].x) -
                              static_cast<double>(spacingFirst[j].x);
            const double dy = static_cast<double>(all[i].y) -
                              static_cast<double>(spacingFirst[j].y);
            if (dx * dx + dy * dy < 3.5 * 3.5) good = false;
        }
        if (good) spacingFirst.push_back(all[i]);
    }
    // Then NMS among the points the spacing filter kept: they are >= 3.5 apart, so
    // none is in another's 3x3 neighbourhood and NMS removes nothing.
    BINCV_CHECK_EQ(all.size(), static_cast<size_t>(3));
    BINCV_CHECK_EQ(spacingFirst.size(), static_cast<size_t>(2));
    BINCV_CHECK_EQ(spacingFirst[1].x, 14);
    BINCV_CHECK_EQ(spacingFirst[1].y, 10);
    //... and that is a corner ops/corner.hpp does NOT return.
    BINCV_CHECK(r.count != spacingFirst.size());
}

// ---------------------------------------------------------------------------
// 5b. THE TIE ORDER, PINNED IN ISOLATION AND IN THE DIRECTION THE REFERENCE SORTS
//
// REGRESSION. `impl::CornerStronger` used to break ties by ASCENDING raster order,
// which is the exact reverse of gftt.cpp's. The reference sorts pointers into the
// `eig` map with `greaterThanPtr`, whose third arm is `(a > b)` on the ADDRESSES --
// a strict total order in which a LATER raster position wins a tie. Nothing in the
// suite caught the inversion, because the oracle in this file was a copy of the
// library's own comparator; it now sorts pointers, like the reference.
//
// This case does not depend on the port at all. Two equal responses, one greedy
// spacing decision, and the answer is one corner or the other:
//
// 8x5 map, 1.0f at (1, 1) and (3, 1), minDistance 3.0
// ascending tie order -> (1, 1) <-- what the bug returned
// descending (reference, and this file) -> (3, 1)
//
// Ties are not a corner case here: a checkerboard makes the entire interior equal,
// and a 3x3 window of {-1, 0, 1} derivatives has few distinct responses to begin
// with, so the tie rule decides most of a real frame's output.
// ---------------------------------------------------------------------------

BINCV_TEST(Corner, TieOrderIsTheReferenceDescendingRasterOrder) {
    // 1. The spacing decision, on a map built so that ONLY the tie rule can move it.
    {
        const int w = 8, h = 5;
        std::vector<float> storage(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
        storage[static_cast<size_t>(1) * static_cast<size_t>(w) + 1] = 1.0f;
        storage[static_cast<size_t>(1) * static_cast<size_t>(w) + 3] = 1.0f;
        const ConstResponseMap view(storage.data(), static_cast<size_t>(w),
                                    static_cast<size_t>(h), static_cast<size_t>(w));
        GoodFeaturesParams p;
        p.blockSize = 3;
        p.qualityLevel = 0.01;
        p.minDistance = 3.0;  // 2 apart < 3, so exactly one of the two survives
        p.maxCorners = 0;
        std::vector<Corner> got(16);
        const CornerResult r = bincv::selectGoodFeatures(view, p, got.data(), got.size());
        BINCV_CHECK_EQ(r.candidatesRanked, static_cast<size_t>(2));
        BINCV_CHECK_EQ(r.count, static_cast<size_t>(1));
        BINCV_CHECK_EQ(got[0].x, 3);  // the LATER position, as greaterThanPtr orders it
        BINCV_CHECK_EQ(got[0].y, 1);

        // And the same map through the literal port, which decides by ADDRESS and
        // therefore cannot agree with an inverted rule by construction.
        size_t refCandidates = 0;
        const std::vector<Corner> want = referenceSelect(storage, w, h, p, &refCandidates);
        BINCV_CHECK_EQ(refCandidates, static_cast<size_t>(2));
        BINCV_CHECK_EQ(want.size(), static_cast<size_t>(1));
        BINCV_CHECK_EQ(want[0].x, got[0].x);
        BINCV_CHECK_EQ(want[0].y, got[0].y);
    }

    // 2. The ROW half of the rule, which an x-only case would not reach: two equal
    // responses in different rows, close enough that one excludes the other.
    {
        const int w = 7, h = 9;
        std::vector<float> storage(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
        storage[static_cast<size_t>(2) * static_cast<size_t>(w) + 3] = 2.0f;
        storage[static_cast<size_t>(5) * static_cast<size_t>(w) + 3] = 2.0f;
        const ConstResponseMap view(storage.data(), static_cast<size_t>(w),
                                    static_cast<size_t>(h), static_cast<size_t>(w));
        GoodFeaturesParams p;
        p.blockSize = 3;
        p.qualityLevel = 0.01;
        p.minDistance = 4.0;  // 3 rows apart < 4
        p.maxCorners = 0;
        std::vector<Corner> got(16);
        const CornerResult r = bincv::selectGoodFeatures(view, p, got.data(), got.size());
        BINCV_CHECK_EQ(r.count, static_cast<size_t>(1));
        BINCV_CHECK_EQ(got[0].y, 5);  // the LATER row
        BINCV_CHECK_EQ(got[0].x, 3);
    }

    // 3. The ordering itself, with no spacing filter at all: an all-equal interior
    // must come out in descending raster order, which is the reference's sort.
    {
        const int w = 6, h = 5;
        std::vector<float> storage(static_cast<size_t>(w) * static_cast<size_t>(h), 3.0f);
        const ConstResponseMap view(storage.data(), static_cast<size_t>(w),
                                    static_cast<size_t>(h), static_cast<size_t>(w));
        GoodFeaturesParams p;
        p.blockSize = 3;
        p.qualityLevel = 0.01;
        p.minDistance = 0.0;  // no spacing: the output IS the ranking
        p.maxCorners = 0;
        std::vector<Corner> got(64);
        const CornerResult r = bincv::selectGoodFeatures(view, p, got.data(), got.size());
        // Interior is [1, h-1) x [1, w-1) == 3 rows x 4 columns.
        BINCV_CHECK_EQ(r.candidatesRanked, static_cast<size_t>(12));
        BINCV_CHECK_EQ(r.count, static_cast<size_t>(12));
        size_t outOfOrder = 0;
        for (size_t i = 1; i < r.count; ++i) {
            const bool descending = (got[i - 1].y > got[i].y) ||
                                    (got[i - 1].y == got[i].y && got[i - 1].x > got[i].x);
            if (!descending) ++outOfOrder;
        }
        BINCV_CHECK_EQ(outOfOrder, static_cast<size_t>(0));
        BINCV_CHECK_EQ(got[0].y, 3);
        BINCV_CHECK_EQ(got[0].x, 4);
        BINCV_CHECK_EQ(got[r.count - 1].y, 1);
        BINCV_CHECK_EQ(got[r.count - 1].x, 1);
    }
}

// ---------------------------------------------------------------------------
// 6. STRUCTURED CONTENT, NOT ONLY RANDOM
//
// Random frames exercise word boundaries and clipping; they do not tell a reader
// that the operation detects CORNERS. These three do, and the middle one is the
// reason a min-eigenvalue response exists at all.
// ---------------------------------------------------------------------------

namespace {

CornerResult detect(const Frame& f, const GoodFeaturesParams& p, std::vector<Corner>& out,
                    BorderType border = BORDER_REFLECT_101) {
    const Derived<uint32_t> d(f, border);
    std::vector<float> storage = makeMapStorage(f.width, f.height);
    ResponseMap map = mapView(storage, f.width, f.height);
    return bincv::goodFeaturesToTrack(d.dx, d.dy, p, map, out.data(), out.size());
}

} // namespace

BINCV_TEST(Corner, Structure_Checkerboard) {
    // A corner at every block junction, and every one of them the same strength.
    const int block = 4;
    const Frame f = checkerboardFrame(kWideW, kWideH, block);
    GoodFeaturesParams p;
    p.blockSize = 3;
    p.minDistance = static_cast<double>(block);
    p.maxCorners = 0;
    p.qualityLevel = 0.01;
    std::vector<Corner> out(4096);
    const CornerResult r = detect(f, p, out);

    BINCV_CHECK(r.count > 20);
    // Every returned corner sits on a junction of the checkerboard -- where four
    // blocks meet. THE JUNCTION IS TWO PIXELS WIDE IN EACH AXIS, not one: the
    // derivative is a `[-1, 0, 1]` tap, so `src(x+1) != src(x-1)` holds at both
    // `x = block*k - 1` and `x = block*k`, and the response has an equal-valued
    // 2x2 plateau at each junction. So the junction test admits `block - 1` and `0`
    // in each axis, and WHICH member of the plateau is returned is decided purely
    // by the tie rule.
    size_t offJunction = 0;
    for (size_t i = 0; i < r.count; ++i) {
        const int xm = out[i].x % block, ym = out[i].y % block;
        if (!((xm == block - 1 || xm == 0) && (ym == block - 1 || ym == 0))) ++offJunction;
    }
    BINCV_CHECK_EQ(offJunction, static_cast<size_t>(0));

    // REGRESSION, AND THE TIE RULE'S SHARPEST CASE. A checkerboard is the frame
    // where ties decide EVERYTHING -- every junction plateau is four equal
    // responses -- so the plateau member the spacing filter accepts is a direct
    // readout of the comparator's tie direction. gftt.cpp's `greaterThanPtr`
    // orders equal responses by DESCENDING address, i.e. later raster position
    // first, so the accepted member is the plateau's bottom-right: `(0, 0)` mod
    // `block`. An ascending tie order returns `(block-1, block-1)` mod `block` for
    // every one of them, and the previous spelling of this test asserted exactly
    // that. Junctions whose full plateau is clipped by the frame edge are excluded,
    // since one of the four positions may not exist there.
    size_t clearOfEdge = 0, notPlateauMax = 0;
    for (size_t i = 0; i < r.count; ++i) {
        const bool clear = out[i].x >= p.blockSize && out[i].y >= p.blockSize &&
                           out[i].x < f.width - p.blockSize && out[i].y < f.height - p.blockSize;
        if (!clear) continue;
        ++clearOfEdge;
        if (out[i].x % block != 0 || out[i].y % block != 0) ++notPlateauMax;
    }
    BINCV_CHECK(clearOfEdge > 20);
    BINCV_CHECK_EQ(notPlateauMax, static_cast<size_t>(0));
    // Every junction whose window and derivative taps are clear of the frame edge
    // is identical to every other, so their responses must be identical too -- the
    // whole result is then decided by the tie-break, which must be stable. Junctions
    // NEAR the edge are legitimately weaker: reflect-101 makes the outermost row
    // and column of the derivative exactly zero, so a window that reaches
    // them sums fewer gradients.
    size_t interiorJunctions = 0, unequal = 0;
    float reference = 0.0f;
    for (size_t i = 0; i < r.count; ++i) {
        const bool clear = out[i].x >= p.blockSize && out[i].y >= p.blockSize &&
                           out[i].x < f.width - p.blockSize && out[i].y < f.height - p.blockSize;
        if (!clear) continue;
        if (interiorJunctions == 0) reference = out[i].response;
        else if (out[i].response != reference) ++unequal;
        ++interiorJunctions;
    }
    BINCV_CHECK(interiorJunctions > 20);
    BINCV_CHECK_EQ(unequal, static_cast<size_t>(0));
    BINCV_CHECK(reference > 0.0f);
    std::printf(" checkerboard(%d): %zu corners, all at junctions, %zu clear of the edge at "
                "response %g\n",
                block, r.count, interiorJunctions, static_cast<double>(reference));
}

BINCV_TEST(Corner, Structure_DiagonalEdgeHasNoCorner) {
    // THE MIN-EIGENVALUE DISCRIMINATOR. A 45-degree step edge has a huge gradient
    // and no corner: every gradient inside an interior window is parallel, the
    // matrix is rank one, `det` is 0, and the response is EXACTLY 0 -- not small,
    // zero, because the determinant is an integer.
    //
    // "Interior" here means a window that sees only INTERIOR derivative rows and
    // columns -- one pixel further in than "the window fits in the frame". The
    // extra pixel is not slack: reflect-101 makes the outermost row and column of
    // the derivative exactly zero, so a window that reaches them holds
    // gradients whose y-component was forced to zero, which is no longer parallel
    // to the rest and gives the matrix a second eigenvalue. Measured on this
    // frame, EVERY non-zero interior response is at exactly that distance from an
    // edge -- 5 of them at blockSize 3, 29 at blockSize 31 -- and all of them are
    // below 1.0 against the checkerboard's 6, so even the boundary effect is not
    // corner-strength.
    const Frame f = diagonalFrame(kWideW, kWideH);
    for (int blockSize : kBlockSizes) {
        const std::vector<float> storage = mapFor<uint32_t>(f, blockSize, BORDER_REFLECT_101);
        const int off = blockSize / 2;
        size_t interiorNonZero = 0, interior = 0;
        float largest = 0.0f;
        for (int y = 0; y < f.height; ++y) {
            for (int x = 0; x < f.width; ++x) {
                const float v = storage[static_cast<size_t>(y) * static_cast<size_t>(f.width) +
                                        static_cast<size_t>(x)];
                if (v > largest) largest = v;
                const bool inside = (x - off >= 1) && (y - off >= 1) &&
                                    (x - off + blockSize <= f.width - 1) &&
                                    (y - off + blockSize <= f.height - 1);
                if (!inside) continue;
                ++interior;
                if (v != 0.0f) ++interiorNonZero;
            }
        }
        BINCV_CHECK(interior > 0);
        BINCV_CHECK_EQ(interiorNonZero, static_cast<size_t>(0));
        // Nothing anywhere on this frame reaches corner strength: the checkerboard
        // at the same block size answers 6.
        BINCV_CHECK(largest < 1.0f);
    }
    // And the same frame really does have an enormous EDGE response, so the zero
    // above is the eigenvalue discriminating, not an absent gradient. A 15x15
    // window centred ON the diagonal: both diagonal entries large, and the
    // determinant exactly zero because every gradient in it is parallel.
    const Derived<uint32_t> d(f, BORDER_REFLECT_101);
    const GradientCovariance cov =
        bincv::gradientCovariance(d.dx, d.dy, Rect(kWideH / 2 - 7, kWideH / 2 - 7, 15, 15));
    BINCV_CHECK(cov.sumXX > 10);
    BINCV_CHECK(cov.sumYY > 10);
    BINCV_CHECK_EQ(cov.sumXX * cov.sumYY - cov.sumXY * cov.sumXY, static_cast<long long>(0));
    std::printf(" diagonal edge: interior response 0 everywhere; a 15x15 window on the edge is "
                "{%lld, %lld, %lld}, det 0\n",
                static_cast<long long>(cov.sumXX), static_cast<long long>(cov.sumYY),
                static_cast<long long>(cov.sumXY));
}

BINCV_TEST(Corner, Structure_IsolatedDot) {
    const Frame f = dotFrame(kWideW, kWideH);
    GoodFeaturesParams p;
    p.blockSize = 3;
    p.minDistance = 2.0;
    p.maxCorners = 0;
    p.qualityLevel = 0.01;
    std::vector<Corner> out(256);
    const CornerResult r = detect(f, p, out);
    BINCV_CHECK_EQ(r.count, static_cast<size_t>(1));
    BINCV_CHECK_EQ(out[0].x, kWideW / 2);
    BINCV_CHECK_EQ(out[0].y, kWideH / 2);
    BINCV_CHECK(out[0].response > 0.0f);
}

// ---------------------------------------------------------------------------
// 7. THE BORDER RING EXISTS TO PREVENT
//
// chose BORDER_REFLECT_101 for the derivative partly BECAUSE a zero fill
// manufactures an edge around the whole frame that this operation would select as
// spurious keypoints. That is checked here, in the only form that can fail: the
// ring must be ABSENT under reflect-101 and PRESENT under BORDER_CONSTANT.
// ---------------------------------------------------------------------------

namespace {

size_t cornersNearBorder(const std::vector<Corner>& c, size_t count, int w, int h, int margin) {
    size_t n = 0;
    for (size_t i = 0; i < count; ++i) {
        if (c[i].x < margin || c[i].y < margin || c[i].x >= w - margin || c[i].y >= h - margin) ++n;
    }
    return n;
}

template <typename WordType>
void borderRingSuite(const char* wordName) {
    const Frame frames[] = {uniformFrame(kWideW, kWideH, 0u), uniformFrame(kWideW, kWideH, 1u),
                            stripeFrame(kWideW, kWideH, 3)};
    // A ZERO fill only manufactures a step where the frame's own border pixels are
    // NOT zero. The blank frame is therefore the one case where BORDER_CONSTANT is
    // harmless -- and it is kept precisely because it says what the hazard is: not
    // "a constant border", but "a constant border that disagrees with the content".
    const bool zeroFillManufacturesAnEdge[] = {false, true, true};
    size_t frameIndex = 0;
    for (const Frame& f : frames) {
        const bool expectRing = zeroFillManufacturesAnEdge[frameIndex++];
        for (int blockSize : {3, 7}) {
            GoodFeaturesParams p;
            p.blockSize = blockSize;
            p.minDistance = 2.0;
            p.maxCorners = 0;
            p.qualityLevel = 0.01;
            std::vector<Corner> out(4096);

            // reflect-101: src(x+1) == src(x-1) at every edge, so the derivative is
            // exactly zero there and a uniform or striped frame has an all-zero
            // response map. No maximum, no threshold, no corners.
            const Derived<WordType> reflect(f, BORDER_REFLECT_101);
            std::vector<float> storageR = makeMapStorage(f.width, f.height);
            ResponseMap mapR = mapView(storageR, f.width, f.height);
            const CornerResult rr =
                bincv::goodFeaturesToTrack(reflect.dx, reflect.dy, p, mapR, out.data(), out.size());
            BINCV_CHECK_EQ(rr.count, static_cast<size_t>(0));
            BINCV_CHECK_EQ(rr.candidatesRanked, static_cast<size_t>(0));

            // BORDER_CONSTANT: the fill manufactures a step at every edge. This is
            // the half that gives the check teeth -- without it, "zero corners"
            // would also be what a broken detector reports.
            const Derived<WordType> zeroFill(f, BORDER_CONSTANT);
            std::vector<float> storageC = makeMapStorage(f.width, f.height);
            ResponseMap mapC = mapView(storageC, f.width, f.height);
            const CornerResult rc = bincv::goodFeaturesToTrack(zeroFill.dx, zeroFill.dy, p, mapC,
                                                               out.data(), out.size());
            BINCV_CHECK_EQ(rc.count > 0, expectRing);
            //... and every one of the spurious corners is ON the border, which is
            // what makes it a ring rather than content this frame happens to have.
            const size_t onBorder =
                cornersNearBorder(out, rc.count, f.width, f.height, blockSize / 2 + 1);
            BINCV_CHECK_EQ(onBorder, rc.count);
            if (blockSize == 3) {
                std::printf(" [%s] %-12s reflect-101: 0 corners; BORDER_CONSTANT: %zu, all on the "
                            "border\n",
                            wordName, f.name.c_str(), rc.count);
            }
        }
    }
}

} // namespace

BINCV_TEST(Corner, BorderRing_uint8_t) { borderRingSuite<uint8_t>("uint8_t"); }
BINCV_TEST(Corner, BorderRing_uint16_t) { borderRingSuite<uint16_t>("uint16_t"); }
BINCV_TEST(Corner, BorderRing_uint32_t) { borderRingSuite<uint32_t>("uint32_t"); }
BINCV_TEST(Corner, BorderRing_uint64_t) { borderRingSuite<uint64_t>("uint64_t"); }

// ---------------------------------------------------------------------------
// 8. THE CAPACITY CONTRACT
//
// `capacity` bounds the NMS survivors that can be RANKED, not the corners
// returned, and a caller who sizes it to `maxCorners` gets a restriction of the
// reference's answer rather than the answer. That is exactly what the flag is for,
// and it has to be exercised in both directions.
// ---------------------------------------------------------------------------

BINCV_TEST(Corner, CapacityContract) {
    const Frame f = randomFrame(kWideW, kWideH, 0x5EED42ULL);
    const std::vector<float> storage = mapFor<uint32_t>(f, 3, BORDER_REFLECT_101);
    ConstResponseMap view(storage.data(), static_cast<size_t>(f.width),
                          static_cast<size_t>(f.height), static_cast<size_t>(f.width));
    GoodFeaturesParams p;
    p.blockSize = 3;
    p.minDistance = 4.0;
    p.maxCorners = 0;
    p.qualityLevel = 0.01;

    const size_t plenty = static_cast<size_t>(f.width) * static_cast<size_t>(f.height);
    std::vector<Corner> big(plenty);
    const CornerResult full = bincv::selectGoodFeatures(view, p, big.data(), plenty);
    BINCV_CHECK_EQ(full.candidatesTruncated, false);
    BINCV_CHECK(full.candidatesRanked > 4);

    // Capacity below the survivor count: truncated, and the survivors kept are the
    // STRONGEST ones -- the same total order the sort uses, so the prefix of the
    // untruncated ranking.
    const size_t small = full.candidatesRanked / 2;
    std::vector<Corner> tight(small);
    const CornerResult cut = bincv::selectGoodFeatures(view, p, tight.data(), small);
    BINCV_CHECK_EQ(cut.candidatesTruncated, true);
    BINCV_CHECK_EQ(cut.candidatesRanked, small);
    BINCV_CHECK(cut.count <= full.count);

    // The first corner cannot change: it is the global maximum either way.
    BINCV_CHECK_EQ(cut.count > 0, full.count > 0);
    if (cut.count > 0 && full.count > 0) {
        BINCV_CHECK_EQ(tight[0].x, big[0].x);
        BINCV_CHECK_EQ(tight[0].y, big[0].y);
    }

    // REGRESSION -- THE WHOLE RETAINED PREFIX, NOT ONLY corners[0].
    //
    // The bounded heap kept `capacity` entries; the docstring says they are the
    // `capacity` STRONGEST NMS survivors, i.e. the exact prefix of the untruncated
    // ranking. Checking `tight[0] == big[0]` alone cannot see that, because the
    // global maximum is the ONE corner a heap built with the reversed comparator
    // still happens to preserve -- measured, this suite reported an identical
    // 2698/2698 with the comparator inverted. The spacing filter destroys the
    // ranking by compacting in place, so both sides run with `minDistance` off:
    // then `count == min(ranked, capacity)` and the buffer IS the ranking.
    GoodFeaturesParams rank = p;
    rank.minDistance = 0.0;  // gftt.cpp's `else` branch: no spacing, just the order
    rank.maxCorners = 0;
    std::vector<Corner> rankedAll(plenty);
    const CornerResult allRanked =
        bincv::selectGoodFeatures(view, rank, rankedAll.data(), plenty);
    BINCV_CHECK_EQ(allRanked.candidatesTruncated, false);
    BINCV_CHECK_EQ(allRanked.count, allRanked.candidatesRanked);
    BINCV_CHECK(allRanked.candidatesRanked > 8);

    // Several capacities, because a single one can be passed by accident.
    size_t prefixChecked = 0;
    for (size_t cap : {static_cast<size_t>(1), static_cast<size_t>(2), static_cast<size_t>(5),
                       static_cast<size_t>(7), allRanked.candidatesRanked / 3,
                       allRanked.candidatesRanked - 1}) {
        if (cap == 0 || cap >= allRanked.candidatesRanked) continue;
        std::vector<Corner> got(cap);
        const CornerResult r = bincv::selectGoodFeatures(view, rank, got.data(), cap);
        BINCV_CHECK_EQ(r.candidatesTruncated, true);
        BINCV_CHECK_EQ(r.candidatesRanked, cap);
        BINCV_CHECK_EQ(r.count, cap);
        size_t differs = 0;
        for (size_t i = 0; i < cap; ++i) {
            if (got[i].x != rankedAll[i].x || got[i].y != rankedAll[i].y ||
                got[i].response != rankedAll[i].response)
                ++differs;
        }
        BINCV_CHECK_EQ(differs, static_cast<size_t>(0));
        ++prefixChecked;
    }
    BINCV_CHECK(prefixChecked >= 4);

    // Capacity zero: no output, no candidates -- and TRUNCATED, because a
    // zero-length buffer could not hold the survivors this map has. Reporting
    // `false` here would make the empty answer indistinguishable from a frame with
    // no corners at all, and `candidatesTruncated` is the only signal a caller has.
    const CornerResult none = bincv::selectGoodFeatures(view, p, nullptr, 0);
    BINCV_CHECK_EQ(none.count, static_cast<size_t>(0));
    BINCV_CHECK_EQ(none.candidatesRanked, static_cast<size_t>(0));
    BINCV_CHECK_EQ(none.candidatesTruncated, true);

    //... and the flag stays false when there is genuinely nothing to truncate: an
    // all-zero map has maxVal 0, threshold 0, and no survivor.
    const std::vector<float> flat(static_cast<size_t>(f.width) * static_cast<size_t>(f.height),
                                  0.0f);
    const ConstResponseMap flatView(flat.data(), static_cast<size_t>(f.width),
                                    static_cast<size_t>(f.height), static_cast<size_t>(f.width));
    const CornerResult blank = bincv::selectGoodFeatures(flatView, p, nullptr, 0);
    BINCV_CHECK_EQ(blank.count, static_cast<size_t>(0));
    BINCV_CHECK_EQ(blank.candidatesRanked, static_cast<size_t>(0));
    BINCV_CHECK_EQ(blank.candidatesTruncated, false);

    // maxCorners caps the OUTPUT without touching the ranking.
    p.maxCorners = 3;
    std::vector<Corner> capped(plenty);
    const CornerResult limited = bincv::selectGoodFeatures(view, p, capped.data(), plenty);
    BINCV_CHECK_EQ(limited.candidatesRanked, full.candidatesRanked);
    BINCV_CHECK_EQ(limited.count, static_cast<size_t>(3));
    size_t prefixDiffers = 0;
    for (size_t i = 0; i < limited.count; ++i)
        if (capped[i].x != big[i].x || capped[i].y != big[i].y) ++prefixDiffers;
    BINCV_CHECK_EQ(prefixDiffers, static_cast<size_t>(0));
}

// ---------------------------------------------------------------------------
// 9. DEGENERATE SHAPES
// ---------------------------------------------------------------------------

BINCV_TEST(Corner, DegenerateShapes) {
    // An empty map: nothing written, nothing read, no corners.
    const CornerResult empty = bincv::selectGoodFeatures(ConstResponseMap(), GoodFeaturesParams(),
                                                         nullptr, 0);
    BINCV_CHECK_EQ(empty.count, static_cast<size_t>(0));

    // A frame too small for the candidate scan (which excludes the outermost row
    // and column) still computes a map and returns nothing.
    for (int side : {1, 2, 3}) {
        Frame f = makeFrame("tiny", side, side);
        for (int y = 0; y < side; ++y)
            for (int x = 0; x < side; ++x) f.set(y, x, static_cast<unsigned>((x + y) & 1));
        const Derived<uint32_t> d(f, BORDER_REFLECT_101);
        std::vector<float> storage = makeMapStorage(side, side);
        ResponseMap map = mapView(storage, side, side);
        bincv::cornerMinEigenVal(d.dx, d.dy, 3, map);
        size_t negative = 0;
        for (float v : storage)
            if (v < 0.0f) ++negative;  // -1.0f is the fill; every pixel must be written
        BINCV_CHECK_EQ(negative, static_cast<size_t>(0));
        std::vector<Corner> out(16);
        GoodFeaturesParams p;
        p.blockSize = 3;
        const CornerResult r =
            bincv::goodFeaturesToTrack(d.dx, d.dy, p, map, out.data(), out.size());
        BINCV_CHECK_EQ(r.count, side <= 2 ? static_cast<size_t>(0) : r.count);
    }

    // blockSize 1: the covariance of a single pixel, so the matrix is rank one
    // everywhere and every response is exactly zero. A real answer, not an error.
    const Frame f = randomFrame(kNarrowW, kNarrowH, 0x1111ULL);
    const std::vector<float> one = mapFor<uint32_t>(f, 1, BORDER_REFLECT_101);
    size_t nonZero = 0;
    for (float v : one)
        if (v != 0.0f) ++nonZero;
    BINCV_CHECK_EQ(nonZero, static_cast<size_t>(0));

    // A blockSize larger than the frame: every window is clipped on every side and
    // the map is one value repeated where the clipped window is the same.
    const std::vector<float> huge = mapFor<uint32_t>(f, 63, BORDER_REFLECT_101);
    BINCV_CHECK_EQ(huge.size(), static_cast<size_t>(kNarrowW) * static_cast<size_t>(kNarrowH));
    size_t unwritten = 0;
    for (float v : huge)
        if (v < 0.0f) ++unwritten;
    BINCV_CHECK_EQ(unwritten, static_cast<size_t>(0));
}

// ---------------------------------------------------------------------------
// 10. THE VIEW AND CONTAINER SPELLINGS AGREE, AND N > 1 DOES NOT COMPILE
// ---------------------------------------------------------------------------

namespace {

template <typename WordType, typename = void>
struct ContainerSpellingAccepts : std::false_type {};

template <typename WordType>
struct ContainerSpellingAccepts<
    WordType, decltype(bincv::cornerMinEigenVal(
                  std::declval<const bincv::SignedQuantMat<3, WordType>&>(),
                  std::declval<const bincv::SignedQuantMat<3, WordType>&>(), 3,
                  std::declval<bincv::ResponseMap>()))> : std::true_type {};

} // namespace

BINCV_TEST(Corner, SpellingsAgree) {
    const Frame f = randomFrame(kWideW, kWideH, 0x2468ULL);
    const Derived<uint32_t> d(f, BORDER_REFLECT_101);
    for (int blockSize : kBlockSizes) {
        std::vector<float> a = makeMapStorage(f.width, f.height);
        std::vector<float> b = makeMapStorage(f.width, f.height);
        ResponseMap ma = mapView(a, f.width, f.height);
        ResponseMap mb = mapView(b, f.width, f.height);
        bincv::cornerMinEigenVal(d.dx, d.dy, blockSize, ma);
        bincv::cornerMinEigenVal<uint32_t>(d.dx.constMagnitude(0), d.dy.constMagnitude(0),
                                           d.dx.constSign(), d.dy.constSign(), blockSize, mb);
        size_t differing = 0;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) ++differing;
        BINCV_CHECK_EQ(differing, static_cast<size_t>(0));
    }

    // the promise 1, inherited: the CONTAINER spelling refuses an N-bit level at
    // compile time rather than returning the LSB plane's response.
    BINCV_CHECK_EQ(ContainerSpellingAccepts<uint32_t>::value, false);
}

// ---------------------------------------------------------------------------
// 11. NO HEAP, ANYWHERE
//
// The reference grows an unbounded `std::vector<const float*>` of candidates and a
// `vector<vector<Point2f>>` grid. ops/corner.hpp uses the caller's array, an
// in-place bounded heap, an in-place sort and an in-place compaction, and this is
// where that is a reading rather than a claim. The counter's own teeth are
// exercised first, including the C++17 over-aligned form -- scratch for a
// vectorized kernel takes exactly that path and a counter replacing only the plain
// operators cannot see it.
// ---------------------------------------------------------------------------

namespace {
struct alignas(64) OverAligned {
    double payload[8];
};
} // namespace

BINCV_TEST(Corner, NoAllocation) {
    // EVERY reading is snapshotted into a local before it is checked. BINCV_CHECK_EQ
    // builds its message with std::to_string, which ALLOCATES -- so `g_newCount`
    // passed directly to the macro is read after the macro's own allocation, and
    // the check compares two different moments. Measured: it reported
    // "got 22834, expected 22834" as a FAILURE.
    {
        const std::size_t before = g_newCount;
        char* p = new char[64];
        delete[] p;
        const std::size_t after = g_newCount;
        BINCV_CHECK_EQ(after, before + 1);
    }
    {
        const std::size_t before = g_newCount;
        OverAligned* p = new OverAligned();
        delete p;
        const std::size_t after = g_newCount;
        BINCV_CHECK_EQ(after, before + 1);
    }

    const Frame f = randomFrame(kWideW, kWideH, 0x13579ULL);
    const Derived<uint32_t> d(f, BORDER_REFLECT_101);
    std::vector<float> storage = makeMapStorage(f.width, f.height);
    ResponseMap map = mapView(storage, f.width, f.height);
    std::vector<Corner> out(static_cast<size_t>(f.width) * static_cast<size_t>(f.height));
    GoodFeaturesParams p;
    p.blockSize = 3;
    p.minDistance = 4.0;
    p.maxCorners = 0;

    for (int blockSize : kBlockSizes) {
        const std::size_t before = g_newCount;
        bincv::cornerMinEigenVal(d.dx, d.dy, blockSize, map);
        const std::size_t after = g_newCount;
        BINCV_CHECK_EQ(after, before);
    }
    {
        const std::size_t before = g_newCount;
        const CornerResult r = bincv::selectGoodFeatures(ConstResponseMap(map), p, out.data(),
                                                         out.size());
        const std::size_t after = g_newCount;
        BINCV_CHECK_EQ(after, before);
        BINCV_CHECK(r.candidatesRanked > 0);
    }
    {
        // And through the whole operation, including the truncating path -- the
        // bounded heap is the piece a `std::vector` would be easiest to reach for.
        const std::size_t before = g_newCount;
        const CornerResult r = bincv::goodFeaturesToTrack(d.dx, d.dy, p, map, out.data(), 8);
        const std::size_t after = g_newCount;
        BINCV_CHECK_EQ(after, before);
        BINCV_CHECK_EQ(r.candidatesTruncated, true);
    }
}

// ---------------------------------------------------------------------------
// 12. THE STREAMING SHAPE: IDENTICAL CORNERS, NOT SIMILAR ONES
//
// `goodFeaturesToTrackStreaming` keeps three rows where `goodFeaturesToTrack`
// keeps a frame-sized float map -- 7 680 B against 1 228 800 B at 640x480. Its
// contract is EQUALITY, and equality is the only thing that makes the trade a
// trade rather than a different operation:
//
// * the quality threshold is relative to the GLOBAL maximum, which a three-row
// ring does not have until the last row;
// * the spacing filter needs the survivors ordered across the WHOLE frame under
// CornerStronger, whose tie rule is DESCENDING raster position.
//
// So the two failure modes this section exists to catch are a threshold taken
// from a partial maximum, and a tie decided by the order candidates happened to
// be visited in. NEITHER is visible on content with distinct responses, which is
// why the frame list here is dominated by frames where the response TIES: a
// checkerboard at block 1 and 4 (the entire interior equal), stripes, a uniform
// frame (every response exactly zero, so `maxVal` is 0, the threshold is 0 and
// nothing survives), and the dot. Random frames are included as well, but they
// are the weak case, not the strong one.
//
// WHAT IS COMPARED: `count`, `candidatesRanked`, `candidatesTruncated`, and the
// whole `[0, candidatesRanked)` prefix of the array element for element --
// coordinates and the response's exact float bits. Not a count of matches, not a
// displacement tolerance, and not just the `count` prefix: the compaction leaves
// the ranked tail in place and it has to agree too.
//
// CAPACITIES ARE SWEPT ACROSS THE TRUNCATION BOUNDARY, because that is where the
// two forms differ most in mechanism. The frame-map form sets
// `candidatesTruncated` when a survivor arrives at a full buffer; the streaming
// form has no such moment -- it reconstructs the flag after the last row from the
// strongest candidate it discarded. `capacity` of 0, 1, half, exactly the
// survivor count, one below it, and far above it exercise both sides of that.
// ---------------------------------------------------------------------------

namespace {

const int kStreamBlockSizes[] = {3, 4, 5, 7, 15, 31};

/// @brief The parameter sets swept beside the frames. `blockSize` is filled in by
/// the caller, so this is only the three SELECTION parameters.
struct SelectionParams {
    const char* name;
    int maxCorners;
    double qualityLevel;
    double minDistance;
};

const SelectionParams kStreamParams[] = {
    // seal_params.yaml verbatim -- the configuration the decision is taken at.
    {"seal-defaults", 200, 0.01, 33.33333333333},
    // The spacing filter DISABLED (gftt.cpp's `else` branch), so the answer is a
    // pure top-`maxCorners` and every tie in the sort is visible in the output.
    {"no-spacing", 200, 0.01, 0.0},
    // No cap and a small spacing: the greedy filter runs over every survivor, so
    // a single reordered tie changes the accepted set downstream of itself.
    {"uncapped-tight", 0, 0.01, 3.5},
    // A severe threshold, which moves the survivor set itself rather than the
    // ranking -- the half of the pipeline that depends on the GLOBAL maximum.
    {"high-quality", 0, 0.5, 5.0},
};

struct StreamTally {
    size_t comparisons = 0;      ///< (frame, blockSize, params, capacity) cells compared
    size_t cornersCompared = 0;  ///< individual Corner records compared
    size_t mismatches = 0;       ///< cells where anything at all differed
    size_t truncatedCells = 0;   ///< cells where the frame-map form reported truncation
    size_t tieCells = 0;         ///< cells whose survivor set contained a repeated response
};

/// @brief Run both shapes over one frame at one blockSize and one parameter set,
/// across a capacity sweep, and compare everything.
template <typename WordType>
void compareShapes(const Frame& f, int blockSize, const SelectionParams& sp, StreamTally& t) {
    const Derived<WordType> d(f, BORDER_REFLECT_101);

    GoodFeaturesParams params;
    params.blockSize = blockSize;
    params.maxCorners = sp.maxCorners;
    params.qualityLevel = sp.qualityLevel;
    params.minDistance = sp.minDistance;

    const size_t pixels = static_cast<size_t>(f.width) * static_cast<size_t>(f.height);

    // How many survivors this frame actually has, measured with a buffer that
    // cannot truncate, so the capacity sweep can be placed ON the boundary rather
    // than near it.
    size_t survivors = 0;
    {
        std::vector<float> storage = makeMapStorage(f.width, f.height);
        ResponseMap map = mapView(storage, f.width, f.height);
        std::vector<Corner> probe(pixels + 1);
        const CornerResult r = bincv::goodFeaturesToTrack(d.dx, d.dy, params, map, probe.data(),
                                                          probe.size());
        BINCV_CHECK_EQ(r.candidatesTruncated, false);
        survivors = r.candidatesRanked;
        // Does this cell actually contain a tie? Recorded so the suite can assert
        // that the tie-breaking path was exercised at all rather than hoping.
        for (size_t i = 1; i < survivors; ++i) {
            if (probe[i].response == probe[i - 1].response) {
                ++t.tieCells;
                break;
            }
        }
    }

    size_t capacities[6];
    size_t nCap = 0;
    capacities[nCap++] = 0;
    capacities[nCap++] = 1;
    capacities[nCap++] = survivors / 2;
    if (survivors > 0) capacities[nCap++] = survivors - 1;
    capacities[nCap++] = survivors;
    capacities[nCap++] = survivors + 8;

    for (size_t ci = 0; ci < nCap; ++ci) {
        const size_t capacity = capacities[ci];

        std::vector<float> storage = makeMapStorage(f.width, f.height);
        ResponseMap map = mapView(storage, f.width, f.height);
        std::vector<Corner> a(capacity + 1);  // +1 so `.data` is never null at 0
        const CornerResult ra = bincv::goodFeaturesToTrack(d.dx, d.dy, params, map, a.data(),
                                                           capacity);

        // The ring: THREE rows, at every block size. Nothing else is carried.
        std::vector<float> ringStorage(bincv::kResponseRingRows *
                                           static_cast<size_t>(f.width), -1.0f);
        ResponseMap ring;
        ring.data = ringStorage.data();
        ring.width = static_cast<size_t>(f.width);
        ring.height = bincv::kResponseRingRows;
        ring.stride = static_cast<size_t>(f.width);
        std::vector<Corner> b(capacity + 1);
        const CornerResult rb = bincv::goodFeaturesToTrackStreaming(d.dx, d.dy, params, ring,
                                                                    b.data(), capacity);

        ++t.comparisons;
        bool bad = false;
        if (ra.count != rb.count) bad = true;
        if (ra.candidatesRanked != rb.candidatesRanked) bad = true;
        if (ra.candidatesTruncated != rb.candidatesTruncated) bad = true;
        if (ra.candidatesTruncated) ++t.truncatedCells;
        if (!bad) {
            // The WHOLE ranked prefix, not just the selected one: the spacing
            // filter compacts in place and leaves the rest of the sorted array
            // behind, and that tail is evidence about the sort's tie order.
            for (size_t i = 0; i < ra.candidatesRanked; ++i) {
                ++t.cornersCompared;
                if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].response != b[i].response) {
                    bad = true;
                    break;
                }
            }
        }
        if (bad) {
            ++t.mismatches;
            std::printf(" MISMATCH %s block %d %s capacity %zu: frame-map "
                        "{%zu, %zu, %d} streaming {%zu, %zu, %d}\n",
                        f.name.c_str(), blockSize, sp.name, capacity, ra.count,
                        ra.candidatesRanked, ra.candidatesTruncated ? 1 : 0, rb.count,
                        rb.candidatesRanked, rb.candidatesTruncated ? 1 : 0);
        }
    }
}

template <typename WordType>
void streamingSuite(const char* wordName) {
    const Frame frames[] = {
        // TIE-DOMINATED, and these are the cases that matter. A checkerboard at
        // block 1 makes the entire interior one response.
        checkerboardFrame(kWideW, kWideH, 1),
        checkerboardFrame(kWideW, kWideH, 4),
        stripeFrame(kWideW, kWideH, 3),
        uniformFrame(kNarrowW, kNarrowH, 1),   // every response 0; maxVal 0; no survivor
        dotFrame(kWideW, kWideH),
        // Structure and noise.
        diagonalFrame(kWideW, kWideH),
        randomFrame(kWideW, kWideH, 0x51ED5EEDULL),
        randomFrame(kNarrowW, kNarrowH, 0xC0FFEEULL),
    };

    StreamTally t;
    for (const Frame& f : frames)
        for (int blockSize : kStreamBlockSizes)
            for (const SelectionParams& sp : kStreamParams) compareShapes<WordType>(f, blockSize, sp, t);

    BINCV_CHECK_EQ(t.mismatches, static_cast<size_t>(0));
    // The sweep has to have EXERCISED what it claims to check, or a zero above is
    // a statement about nothing: corners must have been compared, the truncation
    // path must have been entered, and at least one cell must have contained a
    // tie.
    BINCV_CHECK(t.comparisons > 0);
    BINCV_CHECK(t.cornersCompared > 0);
    BINCV_CHECK(t.truncatedCells > 0);
    BINCV_CHECK(t.tieCells > 0);
    std::printf(" [%s] streaming == frame map: %zu cells, %zu corner records, "
                "%zu truncating cells, %zu cells containing a tied response\n",
                wordName, t.comparisons, t.cornersCompared, t.truncatedCells, t.tieCells);
}

} // namespace

BINCV_TEST(Corner, Streaming_IdenticalCorners_uint8_t)  { streamingSuite<uint8_t>("uint8_t"); }
BINCV_TEST(Corner, Streaming_IdenticalCorners_uint16_t) { streamingSuite<uint16_t>("uint16_t"); }
BINCV_TEST(Corner, Streaming_IdenticalCorners_uint32_t) { streamingSuite<uint32_t>("uint32_t"); }
BINCV_TEST(Corner, Streaming_IdenticalCorners_uint64_t) { streamingSuite<uint64_t>("uint64_t"); }

// ---------------------------------------------------------------------------
// 12b. THE SAME EQUALITY ON FRAMES BIG ENOUGH TO HOLD A REAL SELECTION
//
// The frames above are 71x45 and 40x35, chosen so the whole matrix runs at four
// word types. A frame that size has tens of survivors and a `minDistance` of 33
// selects two or three corners, so the greedy filter barely runs. These frames
// are large enough that the spacing filter does real work and the survivor count
// is in the hundreds -- which is the regime measures and the frontend runs.
// ---------------------------------------------------------------------------

BINCV_TEST(Corner, Streaming_IdenticalCorners_LargeFrames) {
    const Frame frames[] = {
        randomFrame(160, 120, 0xBEEF1234ULL),
        checkerboardFrame(160, 120, 2),        // ties everywhere, at scale
        randomFrame(129, 97, 0xFACE0FFULL),    // width not a multiple of any word width
        stripeFrame(151, 113, 2),
    };
    const int blockSizes[] = {3, 7};
    StreamTally t32, t64;
    for (const Frame& f : frames) {
        for (int blockSize : blockSizes) {
            for (const SelectionParams& sp : kStreamParams) {
                compareShapes<uint32_t>(f, blockSize, sp, t32);
                compareShapes<uint64_t>(f, blockSize, sp, t64);
            }
        }
    }
    BINCV_CHECK_EQ(t32.mismatches, static_cast<size_t>(0));
    BINCV_CHECK_EQ(t64.mismatches, static_cast<size_t>(0));
    BINCV_CHECK(t32.cornersCompared > 1000);
    BINCV_CHECK(t32.truncatedCells > 0);
    BINCV_CHECK(t32.tieCells > 0);
    std::printf(" large frames: uint32_t %zu cells / %zu corner records, "
                "uint64_t %zu cells / %zu corner records, all identical\n",
                t32.comparisons, t32.cornersCompared, t64.comparisons, t64.cornersCompared);
}

// ---------------------------------------------------------------------------
// 12c. THE ROW KERNEL IS THE FRAME MAP'S ROW, BIT FOR BIT
//
// `cornerMinEigenValRow` recomputes a fused `countCovariance` per pixel where
// `cornerMinEigenVal` slides two accumulators down each column and recomputes
// only the cross term. Different traversal, different reduction, SAME three
// integers -- because they are popcounts. If that ever stops being true the
// streaming form's equality is gone, and it would be gone silently.
// ---------------------------------------------------------------------------

namespace {

template <typename WordType>
size_t rowKernelMismatches(const Frame& f, int blockSize) {
    const Derived<WordType> d(f, BORDER_REFLECT_101);
    std::vector<float> storage = makeMapStorage(f.width, f.height);
    ResponseMap map = mapView(storage, f.width, f.height);
    bincv::cornerMinEigenVal(d.dx, d.dy, blockSize, map);

    std::vector<float> row(static_cast<size_t>(f.width), -1.0f);
    size_t differing = 0;
    for (int y = 0; y < f.height; ++y) {
        bincv::cornerMinEigenValRow<WordType>(d.dx.constMagnitude(0), d.dy.constMagnitude(0),
                                              d.dx.constSign(), d.dy.constSign(), blockSize, y,
                                              row.data());
        const float* expected = ConstResponseMap(map).row(static_cast<size_t>(y));
        for (int x = 0; x < f.width; ++x) {
            if (row[static_cast<size_t>(x)] != expected[static_cast<size_t>(x)]) ++differing;
        }
    }
    return differing;
}

template <typename WordType>
void rowKernelSuite(const char* wordName) {
    const Frame frames[] = {checkerboardFrame(kWideW, kWideH, 4), diagonalFrame(kWideW, kWideH),
                            randomFrame(kWideW, kWideH, 0x5EED5EEDULL),
                            randomFrame(kNarrowW, kNarrowH, 0x9911ULL)};
    size_t differing = 0, positions = 0;
    for (const Frame& f : frames) {
        for (int blockSize : kStreamBlockSizes) {
            differing += rowKernelMismatches<WordType>(f, blockSize);
            positions += static_cast<size_t>(f.width) * static_cast<size_t>(f.height);
        }
    }
    BINCV_CHECK_EQ(differing, static_cast<size_t>(0));
    BINCV_CHECK(positions > 0);
    std::printf(" [%s] row kernel == frame map over %zu positions, bit-identical\n", wordName,
                positions);
}

} // namespace

BINCV_TEST(Corner, Streaming_RowMatchesFrameMap_uint8_t)  { rowKernelSuite<uint8_t>("uint8_t"); }
BINCV_TEST(Corner, Streaming_RowMatchesFrameMap_uint16_t) { rowKernelSuite<uint16_t>("uint16_t"); }
BINCV_TEST(Corner, Streaming_RowMatchesFrameMap_uint32_t) { rowKernelSuite<uint32_t>("uint32_t"); }
BINCV_TEST(Corner, Streaming_RowMatchesFrameMap_uint64_t) { rowKernelSuite<uint64_t>("uint64_t"); }

// ---------------------------------------------------------------------------
// 12d. DEGENERATE SHAPES AND NO HEAP, FOR THE STREAMING FORM TOO
//
// A frame shorter than the ring has no NMS row at all, and the one-pass form's
// loop structure is the thing most likely to get that wrong: it evaluates row 0
// before the loop and suppresses row `y-1` inside it.
// ---------------------------------------------------------------------------

BINCV_TEST(Corner, Streaming_DegenerateShapes) {
    struct Shape { int w, h; };
    const Shape shapes[] = {{1, 1}, {1, 9}, {9, 1}, {2, 2}, {3, 3}, {4, 3}, {3, 4}, {33, 2}};
    GoodFeaturesParams params;
    params.blockSize = 3;
    size_t cells = 0;
    for (const Shape& s : shapes) {
        const uint64_t seed = UINT64_C(0x1357) + static_cast<uint64_t>(s.w) * 31u +
                              static_cast<uint64_t>(s.h);
        const Frame f = randomFrame(s.w, s.h, seed);
        const Derived<uint32_t> d(f, BORDER_REFLECT_101);
        std::vector<float> storage = makeMapStorage(s.w, s.h);
        ResponseMap map = mapView(storage, s.w, s.h);
        std::vector<Corner> a(64), b(64);
        const CornerResult ra = bincv::goodFeaturesToTrack(d.dx, d.dy, params, map, a.data(),
                                                           a.size());
        std::vector<float> ringStorage(bincv::kResponseRingRows * static_cast<size_t>(s.w), -1.0f);
        ResponseMap ring{ringStorage.data(), static_cast<size_t>(s.w), bincv::kResponseRingRows,
                         static_cast<size_t>(s.w)};
        const CornerResult rb = bincv::goodFeaturesToTrackStreaming(d.dx, d.dy, params, ring,
                                                                    b.data(), b.size());
        BINCV_CHECK_EQ(ra.count, rb.count);
        BINCV_CHECK_EQ(ra.candidatesRanked, rb.candidatesRanked);
        BINCV_CHECK_EQ(ra.candidatesTruncated, rb.candidatesTruncated);
        for (size_t i = 0; i < ra.candidatesRanked; ++i) {
            BINCV_CHECK_EQ(a[i].x, b[i].x);
            BINCV_CHECK_EQ(a[i].y, b[i].y);
            BINCV_CHECK(a[i].response == b[i].response);
        }
        ++cells;
    }
    BINCV_CHECK_EQ(cells, sizeof(shapes) / sizeof(shapes[0]));
}

BINCV_TEST(Corner, Streaming_NoAllocation) {
    const Frame f = randomFrame(kWideW, kWideH, 0xA11C0DEULL);
    const Derived<uint32_t> d(f, BORDER_REFLECT_101);
    std::vector<float> ringStorage(bincv::kResponseRingRows * static_cast<size_t>(f.width), -1.0f);
    ResponseMap ring{ringStorage.data(), static_cast<size_t>(f.width), bincv::kResponseRingRows,
                     static_cast<size_t>(f.width)};
    std::vector<Corner> corners(512);
    std::vector<float> row(static_cast<size_t>(f.width), 0.0f);

    GoodFeaturesParams params;
    const std::size_t before = g_newCount;
    const CornerResult r = bincv::goodFeaturesToTrackStreaming(d.dx, d.dy, params, ring,
                                                               corners.data(), corners.size());
    bincv::cornerMinEigenValRow<uint32_t>(d.dx.constMagnitude(0), d.dy.constMagnitude(0),
                                          d.dx.constSign(), d.dy.constSign(), 3, 0, row.data());
    const std::size_t during = g_newCount - before;
    BINCV_CHECK_EQ(during, std::size_t{0});
    BINCV_CHECK(r.candidatesRanked > 0);

    // The counter is exercised, so the zero above is a reading and not a blind
    // spot -- the same idiom Corner.NoAllocation uses.
    const std::size_t probeBefore = g_newCount;
    {
        std::vector<double> probe(8, 1.0);
        BINCV_CHECK(probe.size() == 8);
    }
    const std::size_t probeAllocs = g_newCount - probeBefore;
    BINCV_CHECK_EQ(probeAllocs, std::size_t{1});
    std::printf(" streaming: operator new = %zu across the whole call (%zu candidates ranked)\n",
                during, r.candidatesRanked);
}

BINCV_TEST_MAIN("test_corner")
