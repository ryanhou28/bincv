#pragma once

/// @file descriptor.hpp
/// @brief Binary descriptors and Hamming matching. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// THE MOST binCV-NATIVE OPERATION IN COMPUTER VISION
///
/// A BRIEF descriptor **is a bit string**, and matching two of them is
/// `popcount(a ^ b)`. A library whose entire thesis is bit-parallel work at true bit
/// width, which already ships a Hamming block-matcher, had no descriptor extraction
/// and no matcher -- an odd-shaped hole, and the one that separates a VIO frontend
/// from SLAM. LK gives frame-to-frame association; **loop closure, relocalisation and
/// map-point association need descriptors**.
///
/// ---------------------------------------------------------------------------
/// WIDE IN, BITS OUT -- WHICH IS binCV'S SHAPE, NOT AN EXCEPTION TO IT
///
/// The test is `img[a] < img[b]` on the **grayscale** image, exactly as the reference
/// implementations do it, because a comparison between two ONE-BIT pixels carries
/// almost nothing. So this takes `SrcT` like the rest of the sensor stage
/// (the design notes) and emits bits. The intermediate
/// byte never exists.
///
/// ---------------------------------------------------------------------------
/// OpenCV-COMPATIBLE ORB: THE TABLE SHIPS, IN ITS OWN HEADER
///
/// `cv::ORB` samples with a **specific learned 256-pair table**, `bit_pattern_31_`,
/// and descriptors are only comparable across implementations when the table is
/// IDENTICAL -- re-running the paper's learning procedure yields a different one.
/// That table is vendored as `kOrbBriefPattern` in
/// [ops/orbPattern.hpp](orbPattern.hpp), with the notice its license requires
/// carried in the same file and in THIRD_PARTY_NOTICES.md. (An earlier version of
/// this comment called the source Apache-2.0 and said vendoring had to wait for
/// binCV's own license file; both were wrong -- orb.cpp's file-level license is
/// BSD 3-clause, its condition is retaining the notice, and a third-party notice
/// needs no first-party license to live beside. binCV's own license remains an
/// open, deliberately deferred decision.)
///
/// The default pattern here is a deterministic Gaussian sample -- the original
/// BRIEF construction. **Descriptors from two different patterns are not
/// comparable**, which is true of BRIEF generally and is why the pattern is an
/// argument rather than a hidden constant.
///
/// Orientation compensation (ORB's rBRIEF) IS implemented, as the paper describes
/// it rather than as `cv::ORB` does: the orientation
/// ([ops/orientation.hpp](orientation.hpp)) selects one of **30 pre-rotated copies
/// of the pattern** -- 12-degree bins, the paper's own discretization -- where
/// OpenCV rotates per keypoint with the exact angle. The discretized form is the
/// one that fits binCV: the rotated patterns are integer and built once, the
/// kernel needs no per-keypoint trigonometry, and the rotation table is
/// **hardcoded fixed-point** so two platforms' libm cannot disagree about what
/// the pattern is. Descriptors from different bin counts are incomparable
/// exactly as descriptors from different patterns are, which is why the count is
/// a constant of the type rather than a parameter.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../impl/kernel_util.hpp"
#include "reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief One intensity comparison, as offsets from the keypoint.
struct BriefPair {
    int8_t ax, ay, bx, by;
};

/// @brief `Bits` comparisons. One descriptor bit per pair.
/// @note `Bits` must be a multiple of a word so a descriptor occupies whole words --
/// a descriptor that ended mid-word would make `hammingDistance` read padding.
template <size_t Bits>
struct BriefPattern {
    static_assert(Bits % 32 == 0, "descriptor length must be a multiple of 32 bits");
    BriefPair pair[Bits];
};

/// @brief Words a `Bits`-bit descriptor occupies.
template <size_t Bits, typename WordType>
constexpr size_t descriptorWords() {
    return Bits / impl::bitsPerWord<WordType>();
}

namespace impl {

/// @brief Largest absolute offset any of `Bits` pairs reaches. **INTERNAL.**
/// @note Split out of `briefFlattenPattern` because the reach is needed WITHOUT
/// the flat offsets: a kernel that computes each sample's address inline --
/// the GPU arm does, one integer multiply-add per sample being free there --
/// still needs the per-keypoint bounds test, and that test decides `keep`.
/// Two spellings of this loop would be two answers to "is this keypoint
/// describable", which is a correctness difference, not a style one.
template <size_t Bits>
inline int briefPatternReach(const BriefPair* pairs) {
    int reach = 0;
    for (size_t i = 0; i < Bits; ++i) {
        const BriefPair& q = pairs[i];
        const int e[4] = {q.ax < 0 ? -q.ax : q.ax, q.ay < 0 ? -q.ay : q.ay,
                          q.bx < 0 ? -q.bx : q.bx, q.by < 0 ? -q.by : q.by};
        for (int j = 0; j < 4; ++j)
            if (e[j] > reach) reach = e[j];
    }
    return reach;
}

/// @brief The pattern as flat offsets, once per call; returns its reach. **INTERNAL.**
/// @note `q.ay * stride + q.ax` was two MULTIPLIES per pair inside a
/// 256-iteration loop -- half a million of them for a thousand keypoints, all
/// recomputing the same 512 numbers. The same mistake, and the same fix, as
/// ops/fast.hpp's ring offsets. `reach` comes with them: the bounds test
/// belongs per KEYPOINT, not per pair.
template <size_t Bits>
inline int briefFlattenPattern(const BriefPair* pairs, size_t stride, long long* offA,
                               long long* offB) {
    for (size_t i = 0; i < Bits; ++i) {
        const BriefPair& q = pairs[i];
        offA[i] = static_cast<long long>(q.ay) * static_cast<long long>(stride) + q.ax;
        offB[i] = static_cast<long long>(q.by) * static_cast<long long>(stride) + q.bx;
    }
    return briefPatternReach<Bits>(pairs);
}

/// @brief One keypoint's descriptor from prebuilt flat offsets. **INTERNAL.**
/// @note Every sample is in range by construction, so the inner loop is two
/// loads and a compare: no bounds test, no multiply, and no read-modify-write
/// on the descriptor -- the word is ACCUMULATED in a register and stored once
/// per `kBits` pairs.
template <size_t Bits, typename SrcT, typename WordType>
inline void briefDescribeOne(const SrcT* center, const long long* offA,
                             const long long* offB, WordType* d) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    constexpr size_t kWords = Bits / kBits;
    for (size_t w = 0; w < kWords; ++w) {
        WordType acc = 0;
        const size_t base = w * kBits;
        for (size_t b = 0; b < kBits; ++b) {
            const size_t i = base + b;
            acc = static_cast<WordType>(
                acc | (static_cast<WordType>(center[offA[i]] < center[offB[i]]) << b));
        }
        d[w] = acc;
    }
}

} // namespace impl

/// @brief Fills a pattern by deterministic Gaussian sampling -- BRIEF's own
/// construction. **API TIER 3.**
/// @param sigmaOver5 The Gaussian's standard deviation is `patchSize / sigmaOver5`;
/// BRIEF's paper uses `patchSize / 5`.
/// @note Deterministic in `seed`, so two builds agree and a descriptor computed today
/// matches one computed tomorrow. **That matters more than the sampling being
/// optimal** -- descriptors from different patterns are incomparable, so a
/// pattern that silently varied would be a correctness bug, not a quality one.
template <size_t Bits>
inline void makeBriefPattern(BriefPattern<Bits>& out, int patchSize = 31,
                             uint64_t seed = 0x5EA15EEDull, int sigmaOver5 = 5) {
    BINCV_ASSERT(patchSize > 2, "makeBriefPattern: the patch must be wider than 2");
    uint64_t st = seed;
    auto next = [&st]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(st >> 33);
    };
    // Box-Muller would need <cmath> in core; a sum of uniforms is close enough to
    // Gaussian for a sampling pattern and keeps this header dependency-free.
    const int half = patchSize / 2;
    const int sigma = half / (sigmaOver5 > 0 ? sigmaOver5 : 5) + 1;
    auto sample = [&]() -> int8_t {
        int acc = 0;
        for (int k = 0; k < 4; ++k) acc += static_cast<int>(next() % static_cast<uint32_t>(2 * sigma + 1)) - sigma;
        if (acc > half) acc = half;
        if (acc < -half) acc = -half;
        return static_cast<int8_t>(acc);
    };
    for (size_t i = 0; i < Bits; ++i) {
        out.pair[i].ax = sample();
        out.pair[i].ay = sample();
        out.pair[i].bx = sample();
        out.pair[i].by = sample();
    }
}

/// @brief Computes descriptors for `count` keypoints. **API TIER 3.**
/// @param keypointsXY `count` (x, y) pairs, interleaved. A raw float array rather
/// than a point type, so this header depends on nothing but the word helpers:
/// a descriptor extractor should not drag the tracker's types in.
/// @param out `count * descriptorWords<Bits, WordType>` words, filled.
/// @param keep Optional: set to 0 for a keypoint whose patch falls outside the image.
/// **A keypoint too close to the border has no descriptor**, and inventing one
/// by clamping would produce a confident match against nothing.
/// @note Bit `i` is `img[a_i] < img[b_i]`, the reference test. Never allocates.
template <size_t Bits, typename SrcT, typename WordType>
inline void computeBrief(const SrcT* img, size_t width, size_t height, size_t stride,
                         const float* keypointsXY, size_t count,
                         const BriefPattern<Bits>& pattern, WordType* out,
                         uint8_t* keep = nullptr) {
    constexpr size_t kWords = Bits / impl::bitsPerWord<WordType>();
    if (count == 0) return;
    BINCV_ASSERT(img != nullptr && keypointsXY != nullptr && out != nullptr,
                 "computeBrief: null argument");

    long long offA[Bits], offB[Bits];
    const int reach = impl::briefFlattenPattern<Bits>(pattern.pair, stride, offA, offB);

    for (size_t k = 0; k < count; ++k) {
        WordType* d = out + k * kWords;
        for (size_t w = 0; w < kWords; ++w) d[w] = 0;
        const long long cx = static_cast<long long>(keypointsXY[2 * k]);
        const long long cy = static_cast<long long>(keypointsXY[2 * k + 1]);
        const bool inside = impl::squareInsideImage(cx, cy, reach, width, height);
        if (inside) {
            const SrcT* center = img + static_cast<size_t>(cy) * stride +
                                 static_cast<size_t>(cx);
            impl::briefDescribeOne<Bits, SrcT, WordType>(center, offA, offB, d);
        }
        if (keep != nullptr) keep[k] = inside ? uint8_t{1} : uint8_t{0};
    }
}

// ---------------------------------------------------------------------------
// STEERED BRIEF -- the second half of rotation invariance. The first half,
// the intensity-centroid angle, lives in ops/orientation.hpp.
// ---------------------------------------------------------------------------

/// @brief Rotation bins a steered pattern is built at: 12-degree steps, the ORB
/// paper's own discretization.
inline constexpr size_t kBriefAngleBins = 30;

/// @brief Which rotation bin an angle selects: the nearest 12-degree step,
/// wrapped. **API TIER 3.**
/// @param angleRadians An angle in **[-2*pi, 2*pi]** -- `keypointOrientation`'s
/// (-pi, pi] needs no pre-conditioning; an accumulated or otherwise
/// unwrapped angle is the CALLER's to wrap first. The bound is asserted,
/// because outside it the float-to-unsigned cast below is undefined
/// behavior and the two ISAs this library measures resolve it
/// DIFFERENTLY -- the exact cross-platform descriptor divergence the
/// hardcoded rotation table exists to prevent.
/// @note Integer arithmetic after one multiply, no <cmath>: the +30 shift makes
/// the value positive over the asserted domain, so truncation IS floor,
/// and the +0.5 makes floor round-to-nearest.
/// @note **BINCV_HOST_DEVICE**, so a GPU kernel selects the bin by calling THIS
/// function rather than a transcription of it. One ULP of difference here
/// does not change an angle slightly, it changes a whole 256-bit descriptor,
/// and a second copy of this expression is the last place that should be
/// allowed to drift. It is scalar and traversal-free, which is the line.
/// @note `angleRadians * kBinsPerRadian + kBriefAngleBins` is a CONTRACTIBLE
/// multiply-add. Nothing in this project's CMake sets `-ffp-contract`, so
/// GCC's default `fast` applies: an FMA-capable target fuses it and a
/// baseline x86-64 one does not, which means two HOSTS can already disagree
/// about the bin of an angle sitting on a boundary. The CUDA backend pins
/// its own side by compiling the translation unit that calls this with
/// `-fmad=false`, so the device reproduces the unfused evaluation. That
/// makes the device agree with a baseline x86-64 host exactly; it does not
/// remove the host-to-host hole, which is reported rather than patched.
BINCV_HOST_DEVICE inline unsigned briefAngleBin(float angleRadians) {
    constexpr float kTwoPi = 6.28318530717958647692f;
    BINCV_ASSERT(angleRadians >= -kTwoPi && angleRadians <= kTwoPi,
                 "briefAngleBin: angle outside [-2*pi, 2*pi] -- wrap it first");
    constexpr float kBinsPerRadian = static_cast<float>(kBriefAngleBins) / kTwoPi;
    const float t = angleRadians * kBinsPerRadian + static_cast<float>(kBriefAngleBins);
    const unsigned r = static_cast<unsigned>(t + 0.5f);
    // `kBriefAngleBins` is a size_t, so the remainder is one too; the cast says
    // the narrowing is intended, and it is exact -- a remainder modulo 30 fits
    // an unsigned at every width this library compiles at.
    return static_cast<unsigned>(r % kBriefAngleBins);
}

/// @brief `Bits` comparisons at each of the 30 rotations: ~30 KB at 256 bits,
/// built once and reused for every frame.
/// @note A CONTAINER in the descriptor path's sense -- built at setup, read by
/// the kernel. It is plain aggregate data so a bare-metal caller can put it
/// wherever its memory map wants it, flash included.
template <size_t Bits>
struct SteeredBriefPattern {
    BriefPattern<Bits> bin[kBriefAngleBins];
};

namespace impl {

/// @brief cos/sin of each 12-degree bin center, Q16 fixed point. **INTERNAL.**
/// @note HARDCODED, not computed, and that is load-bearing: a pattern that
/// silently varied between two libms would make their descriptors
/// incomparable -- the correctness bug `makeBriefPattern`'s determinism
/// note warns about, arriving through <cmath> instead of the seed.
inline constexpr int kBriefRotQ16[kBriefAngleBins][2] = {
    { 65536,      0}, { 64104,  13626}, { 59870,  26656}, { 53020,  38521},
    { 43852,  48703}, { 32768,  56756}, { 20252,  62328}, {  6850,  65177},
    { -6850,  65177}, {-20252,  62328}, {-32768,  56756}, {-43852,  48703},
    {-53020,  38521}, {-59870,  26656}, {-64104,  13626}, {-65536,      0},
    {-64104, -13626}, {-59870, -26656}, {-53020, -38521}, {-43852, -48703},
    {-32768, -56756}, {-20252, -62328}, { -6850, -65177}, {  6850, -65177},
    { 20252, -62328}, { 32768, -56756}, { 43852, -48703}, { 53020, -38521},
    { 59870, -26656}, { 64104, -13626}};

/// @brief Q16 product back to pixels, rounding half away from zero. **INTERNAL.**
/// @note Division, not a shift: `>>` on a negative value is implementation-defined
/// before C++20, and this number must be THE SAME on every compiler.
inline int briefRotRound(long long v) {
    return static_cast<int>((v >= 0 ? v + 32768 : v - 32768) / 65536);
}

} // namespace impl

/// @brief Builds the 30 rotated copies of `base`. **API TIER 3.**
/// @note Rotation preserves a pair's distance from the keypoint, so a base
/// pattern sampled in the SQUARE (as `makeBriefPattern`'s is, clamped to
/// +/- patchSize/2 per axis) reaches up to sqrt(2) further once rotated --
/// about 21 pixels for a 31-pixel patch -- and keypoints that close to the
/// border lose their descriptor at some angles and not others. A pattern
/// whose samples respect the DISC of radius patchSize/2 (OpenCV's learned
/// table does) keeps the same reach at every angle. Both work; the border
/// behavior is the difference, and it is the base pattern's property.
template <size_t Bits>
inline void makeSteeredBriefPattern(SteeredBriefPattern<Bits>& out,
                                    const BriefPattern<Bits>& base) {
    for (size_t k = 0; k < kBriefAngleBins; ++k) {
        const long long c = impl::kBriefRotQ16[k][0];
        const long long s = impl::kBriefRotQ16[k][1];
        for (size_t i = 0; i < Bits; ++i) {
            const BriefPair& q = base.pair[i];
            const int coords[4] = {q.ax, q.ay, q.bx, q.by};
            int rot[4];
            for (int j = 0; j < 4; j += 2) {
                const long long x = coords[j], y = coords[j + 1];
                rot[j] = impl::briefRotRound(x * c - y * s);
                rot[j + 1] = impl::briefRotRound(x * s + y * c);
                BINCV_ASSERT(rot[j] >= -128 && rot[j] <= 127 && rot[j + 1] >= -128 &&
                                 rot[j + 1] <= 127,
                             "makeSteeredBriefPattern: rotated offset does not fit int8");
            }
            out.bin[k].pair[i] = BriefPair{
                static_cast<int8_t>(rot[0]), static_cast<int8_t>(rot[1]),
                static_cast<int8_t>(rot[2]), static_cast<int8_t>(rot[3])};
        }
    }
}

/// @brief `computeBrief` steered by per-keypoint angles. **API TIER 3.**
/// @param angles One angle per keypoint, radians -- `keypointOrientation`'s
/// output, byte for byte.
/// @param keep As `computeBrief`'s: 0 when the SELECTED bin's pattern falls
/// outside the image at this keypoint. The reach is the bin's own, so a
/// border keypoint can have a descriptor at one angle and not another --
/// see `makeSteeredBriefPattern` on why, and on the base pattern that
/// avoids it.
/// @note Runs bin by bin: one bin's flat offsets on the stack (the same ~4 KB
/// `computeBrief` uses), then every keypoint of that bin -- so the cost
/// over `computeBrief` is 30 pattern flattens per CALL, not a rotation per
/// keypoint, and the stack does not grow with the bin count. Bin 0 is the
/// identity rotation: all-zero angles reproduce `computeBrief` bit for bit
/// (tests/test_descriptor.cpp holds it to that).
template <size_t Bits, typename SrcT, typename WordType>
inline void computeBriefSteered(const SrcT* img, size_t width, size_t height, size_t stride,
                                const float* keypointsXY, size_t count, const float* angles,
                                const SteeredBriefPattern<Bits>& pattern, WordType* out,
                                uint8_t* keep = nullptr) {
    constexpr size_t kWords = Bits / impl::bitsPerWord<WordType>();
    if (count == 0) return;
    BINCV_ASSERT(img != nullptr && keypointsXY != nullptr && angles != nullptr &&
                     out != nullptr,
                 "computeBriefSteered: null argument");

    long long offA[Bits], offB[Bits];
    for (unsigned b = 0; b < kBriefAngleBins; ++b) {
        int reach = -1;   // flattened lazily: most calls populate a few bins
        for (size_t k = 0; k < count; ++k) {
            if (briefAngleBin(angles[k]) != b) continue;
            if (reach < 0)
                reach = impl::briefFlattenPattern<Bits>(pattern.bin[b].pair, stride, offA,
                                                        offB);
            WordType* d = out + k * kWords;
            for (size_t w = 0; w < kWords; ++w) d[w] = 0;
            const long long cx = static_cast<long long>(keypointsXY[2 * k]);
            const long long cy = static_cast<long long>(keypointsXY[2 * k + 1]);
            const bool inside = impl::squareInsideImage(cx, cy, reach, width, height);
            if (inside) {
                const SrcT* center = img + static_cast<size_t>(cy) * stride +
                                     static_cast<size_t>(cx);
                impl::briefDescribeOne<Bits, SrcT, WordType>(center, offA, offB, d);
            }
            if (keep != nullptr) keep[k] = inside ? uint8_t{1} : uint8_t{0};
        }
    }
}

/// @brief `popcount(a ^ b)` over `words`. **API TIER 3.**
/// @note This is the whole of descriptor matching, and it is the operation binCV is
/// built out of. On x86 it is `POPCNT`; on aarch64 `CNT`.
template <typename WordType>
inline unsigned hammingDistance(const WordType* a, const WordType* b, size_t words) {
    unsigned d = 0;
    for (size_t i = 0; i < words; ++i)
        d += static_cast<unsigned>(impl::popcountWord<WordType>(static_cast<WordType>(a[i] ^ b[i])));
    return d;
}

/// @brief One query's best and second-best match.
struct DescriptorMatch {
    size_t trainIndex = 0;
    unsigned distance = 0;
    unsigned secondDistance = 0;   ///< for the ratio test
    bool valid = false;
};

/// @brief Brute-force nearest neighbour with Lowe's ratio test. **API TIER 3.**
/// @param maxRatio Reject unless `best * 100 <= secondBest * maxRatio`. Lowe's 0.8 is
/// `maxRatio == 80`. **An integer percentage, not a float**, so core needs no
/// floating-point comparison and the rule is exact.
/// @note Brute force on purpose: at a few hundred keypoints a k-d tree loses to a
/// linear scan of contiguous words, and Hamming space has no useful metric tree
/// at 256 bits anyway.
template <typename WordType>
inline void matchDescriptors(const WordType* query, size_t queryCount, const WordType* train,
                             size_t trainCount, size_t words, DescriptorMatch* out,
                             unsigned maxRatio = 80) {
    for (size_t q = 0; q < queryCount; ++q) {
        unsigned best = 0xFFFFFFFFu, second = 0xFFFFFFFFu;
        size_t bestIdx = 0;
        for (size_t t = 0; t < trainCount; ++t) {
            const unsigned d = hammingDistance<WordType>(query + q * words, train + t * words,
                                                         words);
            if (d < best) {
                second = best;
                best = d;
                bestIdx = t;
            } else if (d < second) {
                second = d;
            }
        }
        DescriptorMatch m;
        m.trainIndex = bestIdx;
        m.distance = best;
        m.secondDistance = second;
        // With one candidate there is no ratio to test; accepting it unconditionally
        // would make a single-keypoint train set match everything.
        m.valid = trainCount >= 2 && second != 0xFFFFFFFFu &&
                  static_cast<uint64_t>(best) * 100u <=
                      static_cast<uint64_t>(second) * maxRatio;
        out[q] = m;
    }
}

/// @brief `matchDescriptors` restricted to candidates a pipeline's priors admit:
/// a position window, and optionally an octave band. **API TIER 3.**
/// @param queryXY / trainXY (x, y) per keypoint, interleaved -- the descriptor
/// family's raw-array contract. Positions are whatever frame the caller
/// matches in; frame-to-frame association passes both sets in the same
/// image plane and `maxDx/maxDy` bound the motion.
/// @param maxDx,maxDy Half-extents of the admission window, in the positions'
/// own units. The gate is two float compares per candidate, against the
/// `words` XORs and popcounts it saves -- which is the whole point: a
/// pipeline measured brute force at the cost of a full detection level,
/// while knowing priors the matcher ignored.
/// @param queryOctave / trainOctave Optional (both or neither): admit only
/// candidates within `maxOctaveDelta` pyramid levels -- a keypoint rarely
/// jumps more than one octave between consecutive frames.
/// @note THE RATIO TEST RUNS INSIDE THE GATE: best and second-best are the best
/// ADMITTED candidates, and validity needs two of them. That is the honest
/// semantics -- a distant second-best the gate excluded was never a real
/// rival -- and it makes an unbounded window reproduce `matchDescriptors`
/// exactly, which the tests pin.
/// @note Never allocates; O(queryCount * trainCount) gate tests, with the
/// descriptor work paid only inside the window.
template <typename WordType>
inline void matchDescriptorsGated(const WordType* query, const float* queryXY,
                                  size_t queryCount, const WordType* train,
                                  const float* trainXY, size_t trainCount, size_t words,
                                  float maxDx, float maxDy, DescriptorMatch* out,
                                  unsigned maxRatio = 80, const int* queryOctave = nullptr,
                                  const int* trainOctave = nullptr,
                                  int maxOctaveDelta = 1) {
    if (queryCount == 0) return;
    BINCV_ASSERT(query != nullptr && queryXY != nullptr && out != nullptr,
                 "matchDescriptorsGated: null query argument");
    BINCV_ASSERT(trainCount == 0 || (train != nullptr && trainXY != nullptr),
                 "matchDescriptorsGated: null train argument");
    BINCV_ASSERT((queryOctave == nullptr) == (trainOctave == nullptr),
                 "matchDescriptorsGated: octave arrays come as a pair or not at all");
    BINCV_ASSERT(maxDx >= 0.0f && maxDy >= 0.0f,
                 "matchDescriptorsGated: the window must not be negative");

    for (size_t q = 0; q < queryCount; ++q) {
        const float qx = queryXY[2 * q], qy = queryXY[2 * q + 1];
        unsigned best = 0xFFFFFFFFu, second = 0xFFFFFFFFu;
        size_t bestIdx = 0, admitted = 0;
        for (size_t t = 0; t < trainCount; ++t) {
            const float dx = trainXY[2 * t] - qx;
            const float dy = trainXY[2 * t + 1] - qy;
            if (dx > maxDx || dx < -maxDx || dy > maxDy || dy < -maxDy) continue;
            if (queryOctave != nullptr) {
                const int od = queryOctave[q] - trainOctave[t];
                if (od > maxOctaveDelta || od < -maxOctaveDelta) continue;
            }
            ++admitted;
            const unsigned d = hammingDistance<WordType>(query + q * words,
                                                         train + t * words, words);
            if (d < best) {
                second = best;
                best = d;
                bestIdx = t;
            } else if (d < second) {
                second = d;
            }
        }
        DescriptorMatch m;
        m.trainIndex = bestIdx;
        m.distance = best;
        m.secondDistance = second;
        m.valid = admitted >= 2 && second != 0xFFFFFFFFu &&
                  static_cast<uint64_t>(best) * 100u <=
                      static_cast<uint64_t>(second) * maxRatio;
        out[q] = m;
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
