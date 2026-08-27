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
/// ([ARCHITECTURE §7.8](../../../ARCHITECTURE.md)) and emits bits. The intermediate
/// byte never exists.
///
/// ---------------------------------------------------------------------------
/// WHAT THIS IS NOT: OpenCV-COMPATIBLE ORB
///
/// `cv::ORB` uses a **specific learned 256-pair table** from the ORB paper, plus an
/// orientation from the intensity centroid. **binCV does not reproduce that table** --
/// it is data, not an algorithm, and copying it would import a licence question for no
/// technical gain. A caller who needs descriptors that match OpenCV's must pass
/// OpenCV's pattern in; `BriefPattern` exists precisely so that is possible.
///
/// The default pattern here is a deterministic Gaussian sample, which is the original
/// BRIEF construction. **Descriptors from two different patterns are not comparable**,
/// which is true of BRIEF generally and is why the pattern is an explicit argument
/// rather than a hidden constant.
///
/// Orientation compensation (ORB's rBRIEF) is **not implemented**: it needs the
/// intensity centroid and a rotated pattern lookup, and it is a separate piece of work
/// rather than a flag on this one.

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
///       a descriptor that ended mid-word would make `hammingDistance` read padding.
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

/// @brief Fills a pattern by deterministic Gaussian sampling -- BRIEF's own
///        construction. **API TIER 3.**
/// @param sigmaOver5 The Gaussian's standard deviation is `patchSize / sigmaOver5`;
///        BRIEF's paper uses `patchSize / 5`.
/// @note Deterministic in `seed`, so two builds agree and a descriptor computed today
///       matches one computed tomorrow. **That matters more than the sampling being
///       optimal** -- descriptors from different patterns are incomparable, so a
///       pattern that silently varied would be a correctness bug, not a quality one.
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
///        than a point type, so this header depends on nothing but the word helpers:
///        a descriptor extractor should not drag the tracker's types in.
/// @param out `count * descriptorWords<Bits, WordType>()` words, filled.
/// @param keep Optional: set to 0 for a keypoint whose patch falls outside the image.
///        **A keypoint too close to the border has no descriptor**, and inventing one
///        by clamping would produce a confident match against nothing.
/// @note Bit `i` is `img[a_i] < img[b_i]`, the reference test. Never allocates.
template <size_t Bits, typename SrcT, typename WordType>
inline void computeBrief(const SrcT* img, size_t width, size_t height, size_t stride,
                         const float* keypointsXY, size_t count,
                         const BriefPattern<Bits>& pattern, WordType* out,
                         uint8_t* keep = nullptr) {
    constexpr size_t kBits = impl::bitsPerWord<WordType>();
    constexpr size_t kWords = Bits / kBits;
    if (count == 0) return;
    BINCV_ASSERT(img != nullptr && keypointsXY != nullptr && out != nullptr,
                 "computeBrief: null argument");

    for (size_t k = 0; k < count; ++k) {
        WordType* d = out + k * kWords;
        for (size_t w = 0; w < kWords; ++w) d[w] = 0;
        const long long cx = static_cast<long long>(keypointsXY[2 * k]);
        const long long cy = static_cast<long long>(keypointsXY[2 * k + 1]);
        bool inside = true;
        for (size_t i = 0; i < Bits && inside; ++i) {
            const BriefPair& p = pattern.pair[i];
            const long long axx = cx + p.ax, ayy = cy + p.ay;
            const long long bxx = cx + p.bx, byy = cy + p.by;
            if (axx < 0 || ayy < 0 || bxx < 0 || byy < 0 ||
                axx >= static_cast<long long>(width) || bxx >= static_cast<long long>(width) ||
                ayy >= static_cast<long long>(height) || byy >= static_cast<long long>(height)) {
                inside = false;
                break;
            }
            const SrcT va = img[static_cast<size_t>(ayy) * stride + static_cast<size_t>(axx)];
            const SrcT vb = img[static_cast<size_t>(byy) * stride + static_cast<size_t>(bxx)];
            if (va < vb) d[i / kBits] = static_cast<WordType>(d[i / kBits] | (WordType{1} << (i % kBits)));
        }
        if (!inside) {
            for (size_t w = 0; w < kWords; ++w) d[w] = 0;
        }
        if (keep != nullptr) keep[k] = inside ? uint8_t{1} : uint8_t{0};
    }
}

/// @brief `popcount(a ^ b)` over `words`. **API TIER 3.**
/// @note This is the whole of descriptor matching, and it is the operation binCV is
///       built out of. On x86 it is `POPCNT` (D-47); on aarch64 `CNT` (D-6).
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
///        `maxRatio == 80`. **An integer percentage, not a float**, so core needs no
///        floating-point comparison and the rule is exact.
/// @note Brute force on purpose: at a few hundred keypoints a k-d tree loses to a
///       linear scan of contiguous words, and Hamming space has no useful metric tree
///       at 256 bits anyway.
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

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
