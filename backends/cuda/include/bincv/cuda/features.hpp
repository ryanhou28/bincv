#pragma once

/// @file features.hpp
/// @brief The feature families' shared device vocabulary: the sets a kernel
/// READS, and the result PODs it WRITES.
///
/// ---------------------------------------------------------------------------
/// BATCHED, NEVER PER-CALL
///
/// Every type here names a WHOLE SET -- all of a frame's keypoints, all their
/// descriptors, all their matches. That is not a convenience: a launch per
/// keypoint is not a GPU implementation of anything, and a vocabulary that made
/// one keypoint expressible would get one. Nothing in this file describes a
/// single keypoint's inputs, so a per-keypoint entry point cannot be written
/// against it without inventing a second vocabulary first -- which is the thing
/// a reviewer would then see.
///
/// ---------------------------------------------------------------------------
/// THE RAW-ARRAY CONTRACT IS THE HOST'S
///
/// The host descriptor and stereo families take keypoints as interleaved
/// `(x, y)` floats and descriptors as `words` words per keypoint, contiguous --
/// `computeBrief` writes `out + k * words`, `matchDescriptors` reads
/// `query + q * words`, `stereoDescriptorMatch` reads `leftXY[2 * i]` and
/// `leftXY[2 * i + 1]`. The views below address exactly those bytes, so an
/// upload is a raw copy and a device result can be checked against a host run
/// on the same arrays. The layout claims are pinned by `static_assert` where
/// they are compile-time facts and by a sweep in the CUDA suite where they are
/// not.
///
/// ---------------------------------------------------------------------------
/// WHERE THE RESULT PODs NARROW, AND WHY
///
/// `DeviceFastCorner` and `DeviceCorner` are byte-identical to the host
/// `FastCorner` and `Corner`: no field wants narrowing, so identity is free and
/// the download is a `cudaMemcpy` with no per-element pass. `static_assert`
/// holds them to it.
///
/// `DeviceDescriptorMatch` and `DeviceStereoMatch` narrow their keypoint INDEX
/// from the host's `size_t` to `uint32_t` -- a device op accepting a narrower
/// domain than its host twin, with the domain named here, asserted by the
/// factories below, and rejected rather than truncated by a launcher that meets
/// it. The domain is not a restriction in practice: a device keypoint set is
/// counted by a 32-bit atomic (compaction.hpp), so a set these cannot index
/// cannot exist on this backend. The narrowing buys the shape that matters on
/// the device -- 16 bytes, one 128-bit store per match instead of 24 bytes and
/// two stores, and a third off the result buffer. Every field that is NOT
/// narrowed is held to the host's own type by `static_assert`, so a host-side
/// change to `disparity` or `distance` breaks this file rather than quietly
/// producing a device result that means something else.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

#include "bincv/ops/corner.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/fast.hpp"
#include "bincv/ops/stereo.hpp"
#include "compaction.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

// ---------------------------------------------------------------------------
// Keypoints and descriptors: what a kernel reads
// ---------------------------------------------------------------------------

/// @brief Non-owning, read-only view of a whole keypoint set in DEVICE memory.
/// @note `xy` is the host family's own contract -- `count` interleaved `(x, y)`
/// float pairs -- so an upload is a raw copy of the array
/// `goodFeaturesToTrack` or `computeBrief` already works on, and a device
/// run can be compared against a host run over the same bytes.
/// @note Positions are `float` and stay `float`: the frontend's keypoints are
/// sub-pixel once optical flow has touched them, and an integer device
/// twin would silently re-quantize what the host refined.
/// @note `octave` is optional, exactly as `matchDescriptorsGated`'s octave
/// arrays are: null when the caller has no pyramid, and a kernel that reads
/// it must test for null rather than assume a level of zero.
struct DeviceKeypointSetConstView {
    const float* xy = nullptr;       ///< 2 floats per keypoint, interleaved
    const int32_t* octave = nullptr; ///< optional pyramid level per keypoint
    uint32_t count = 0;              ///< keypoints, NOT floats

    DeviceKeypointSetConstView() = default;
    BINCV_CUDA_HD DeviceKeypointSetConstView(const float* xy_, uint32_t count_,
                                             const int32_t* octave_ = nullptr)
        : xy(xy_), octave(octave_), count(count_) {}

    BINCV_CUDA_HD bool empty() const { return xy == nullptr || count == 0; }
    BINCV_CUDA_HD bool hasOctave() const { return octave != nullptr; }

    /// @brief Keypoint `i`'s column. The host's `keypointsXY[2 * i]`.
    BINCV_CUDA_HD float x(uint32_t i) const { return xy[2 * static_cast<size_t>(i)]; }
    /// @brief Keypoint `i`'s row. The host's `keypointsXY[2 * i + 1]`.
    BINCV_CUDA_HD float y(uint32_t i) const { return xy[2 * static_cast<size_t>(i) + 1]; }
};

/// @brief Words a `Bits`-bit descriptor occupies on the device.
/// @note The device word type is `uint32_t` only (core.hpp), so this is a
/// constant rather than a template over the word type. It is asserted equal
/// to the host's `descriptorWords<Bits, uint32_t>()` below.
template <size_t Bits>
BINCV_CUDA_HD constexpr uint32_t descriptorWords() {
    static_assert(Bits % 32 == 0, "descriptor length must be a multiple of 32 bits");
    return static_cast<uint32_t>(Bits / 32);
}

// The device's descriptor pitch IS the host's, at the lengths the ORB family
// uses. A drift here would not fail to compile anywhere -- it would read every
// descriptor after the first from the wrong offset.
static_assert(descriptorWords<128>() == bincv::descriptorWords<128, uint32_t>(),
              "device and host descriptor pitch must agree");
static_assert(descriptorWords<256>() == bincv::descriptorWords<256, uint32_t>(),
              "device and host descriptor pitch must agree");
static_assert(descriptorWords<512>() == bincv::descriptorWords<512, uint32_t>(),
              "device and host descriptor pitch must agree");

/// @brief Non-owning, mutable view of a whole descriptor set in DEVICE memory.
/// @note The mutable spelling exists because a descriptor set is PRODUCED on
/// the device -- it is the BRIEF family's whole output -- and the two-view
/// rule the bit views follow applies for the same reason: a kernel that
/// writes says so in its signature.
/// @note `descriptor(i)` is the host's `out + i * words`. Descriptors are whole
/// words by construction (`BriefPattern` static_asserts `Bits % 32 == 0`),
/// so no descriptor ends mid-word and `hammingDistance` never reads padding.
/// @note A descriptor computed at ANY host word width uploads as a byte copy:
/// bit `i` lands in byte `i / 8`, bit `i % 8` at every width on a
/// little-endian host, the same property `transfer.hpp` relies on for
/// planes. The suite pins it rather than leaving it as an argument.
struct DeviceDescriptorSetView {
    uint32_t* words = nullptr;          ///< count * wordsPerDescriptor words
    uint8_t* keep = nullptr;            ///< optional: computeBrief's per-keypoint validity byte
    uint32_t count = 0;                 ///< descriptors
    uint32_t wordsPerDescriptor = 0;    ///< Bits / 32

    DeviceDescriptorSetView() = default;
    BINCV_CUDA_HD DeviceDescriptorSetView(uint32_t* words_, uint32_t count_,
                                          uint32_t wordsPerDescriptor_,
                                          uint8_t* keep_ = nullptr)
        : words(words_), keep(keep_), count(count_),
          wordsPerDescriptor(wordsPerDescriptor_) {}

    BINCV_CUDA_HD bool empty() const {
        return words == nullptr || count == 0 || wordsPerDescriptor == 0;
    }
    /// @brief First word of descriptor `i`: the host's `out + i * words`.
    BINCV_CUDA_HD uint32_t* descriptor(uint32_t i) const {
        return words + static_cast<size_t>(i) * wordsPerDescriptor;
    }
    /// @brief Words backing the whole set.
    BINCV_CUDA_HD size_t sizeInWords() const {
        return static_cast<size_t>(count) * wordsPerDescriptor;
    }
};

/// @brief Non-owning, read-only view of a whole descriptor set in DEVICE
/// memory -- what the matcher and the stereo coarse stage take.
struct DeviceDescriptorSetConstView {
    const uint32_t* words = nullptr;
    const uint8_t* keep = nullptr;
    uint32_t count = 0;
    uint32_t wordsPerDescriptor = 0;

    DeviceDescriptorSetConstView() = default;
    BINCV_CUDA_HD DeviceDescriptorSetConstView(const uint32_t* words_, uint32_t count_,
                                               uint32_t wordsPerDescriptor_,
                                               const uint8_t* keep_ = nullptr)
        : words(words_), keep(keep_), count(count_),
          wordsPerDescriptor(wordsPerDescriptor_) {}
    /// @brief A mutable set converts, as the bit views do.
    BINCV_CUDA_HD DeviceDescriptorSetConstView(const DeviceDescriptorSetView& v)
        : words(v.words), keep(v.keep), count(v.count),
          wordsPerDescriptor(v.wordsPerDescriptor) {}

    BINCV_CUDA_HD bool empty() const {
        return words == nullptr || count == 0 || wordsPerDescriptor == 0;
    }
    BINCV_CUDA_HD const uint32_t* descriptor(uint32_t i) const {
        return words + static_cast<size_t>(i) * wordsPerDescriptor;
    }
    BINCV_CUDA_HD size_t sizeInWords() const {
        return static_cast<size_t>(count) * wordsPerDescriptor;
    }
};

// ---------------------------------------------------------------------------
// Results: what a kernel writes
// ---------------------------------------------------------------------------

/// @brief One FAST corner on the device. Byte-identical to `bincv::FastCorner`.
/// @note The score is the host's -- the longest qualifying arc, 9 to 16, not
/// `cv::FAST`'s. Keeping the field's type identical is what makes a device
/// result comparable to a host one without a conversion step that could
/// itself be the bug.
struct DeviceFastCorner {
    int x;
    int y;
    long long score;

    /// @brief The host library's corner. Field-for-field, no reinterpretation.
    FastCorner toHost() const {
        FastCorner c;
        c.x = x;
        c.y = y;
        c.score = score;
        return c;
    }
};

/// @brief One good-features corner on the device. Byte-identical to
/// `bincv::Corner`.
struct DeviceCorner {
    int x;
    int y;
    float response;

    Corner toHost() const {
        Corner c;
        c.x = x;
        c.y = y;
        c.response = response;
        return c;
    }
};

/// @brief One query's best and second-best match on the device.
/// @note 16 bytes and 16-byte aligned: one 128-bit store per match, where the
/// host's 24-byte type would be two. See "WHERE THE RESULT PODs NARROW" at
/// the top of this file for the index domain.
struct alignas(16) DeviceDescriptorMatch {
    uint32_t trainIndex;      ///< index into the train set; host type is size_t
    uint32_t distance;        ///< Hamming distance of the best match
    uint32_t secondDistance;  ///< for the ratio test
    uint32_t valid;           ///< 0 or 1; host type is bool

    DescriptorMatch toHost() const {
        DescriptorMatch m;
        m.trainIndex = trainIndex;
        m.distance = distance;
        m.secondDistance = secondDistance;
        m.valid = valid != 0;
        return m;
    }
};

/// @brief One left keypoint's stereo result on the device.
/// @note 16 bytes and 16-byte aligned, for the reason DeviceDescriptorMatch is.
/// `disparity` stays `float`: it is the answer, and the sub-pixel fit the
/// host performs is the whole reason the field is not an integer.
struct alignas(16) DeviceStereoMatch {
    float disparity;      ///< leftX - rightX, level-0 pixels
    uint32_t distance;    ///< descriptor Hamming of the accepted candidate
    uint32_t rightIndex;  ///< index into the right keypoint set; host type is size_t
    uint32_t valid;       ///< 0: no candidate survived; host type is uint8_t

    StereoMatch toHost() const {
        StereoMatch m;
        m.disparity = disparity;
        m.distance = distance;
        m.rightIndex = rightIndex;
        m.valid = static_cast<uint8_t>(valid != 0 ? 1 : 0);
        return m;
    }
};

// The byte-identity claims, asserted rather than trusted: these two types are
// downloaded as raw copies, so a field that moved or changed width would hand
// back a fully-formed corner list with the wrong numbers in it.
static_assert(sizeof(DeviceFastCorner) == sizeof(FastCorner),
              "DeviceFastCorner must be byte-identical to the host FastCorner");
static_assert(alignof(DeviceFastCorner) == alignof(FastCorner),
              "DeviceFastCorner must be byte-identical to the host FastCorner");
static_assert(offsetof(DeviceFastCorner, x) == offsetof(FastCorner, x),
              "DeviceFastCorner must be byte-identical to the host FastCorner");
static_assert(offsetof(DeviceFastCorner, y) == offsetof(FastCorner, y),
              "DeviceFastCorner must be byte-identical to the host FastCorner");
static_assert(offsetof(DeviceFastCorner, score) == offsetof(FastCorner, score),
              "DeviceFastCorner must be byte-identical to the host FastCorner");
static_assert(std::is_same<decltype(DeviceFastCorner::score),
                           decltype(FastCorner::score)>::value,
              "the FAST score's type is the host's; a narrower one loses arcs");

static_assert(sizeof(DeviceCorner) == sizeof(Corner),
              "DeviceCorner must be byte-identical to the host Corner");
static_assert(alignof(DeviceCorner) == alignof(Corner),
              "DeviceCorner must be byte-identical to the host Corner");
static_assert(offsetof(DeviceCorner, x) == offsetof(Corner, x),
              "DeviceCorner must be byte-identical to the host Corner");
static_assert(offsetof(DeviceCorner, y) == offsetof(Corner, y),
              "DeviceCorner must be byte-identical to the host Corner");
static_assert(offsetof(DeviceCorner, response) == offsetof(Corner, response),
              "DeviceCorner must be byte-identical to the host Corner");
static_assert(std::is_same<decltype(DeviceCorner::response),
                           decltype(Corner::response)>::value,
              "the response's type is the host's; the selection compares them");

// The narrowed types cannot claim identity, so they claim the two things that
// matter instead: the fields that were NOT narrowed still have the host's
// types, and the fields that WERE narrowed are still the ones we think they
// are. A host-side change to either breaks this file, which is the point.
static_assert(sizeof(DeviceDescriptorMatch) == 16,
              "one 128-bit store per match is why the indices are narrowed");
static_assert(std::is_same<decltype(DescriptorMatch::trainIndex), size_t>::value,
              "the device narrows this index to uint32_t under the documented domain; "
              "if the host type changed, re-derive the narrowing");
static_assert(std::is_same<decltype(DeviceDescriptorMatch::distance),
                           decltype(DescriptorMatch::distance)>::value,
              "the Hamming distance is not narrowed: it is compared against the host's");
static_assert(std::is_same<decltype(DeviceDescriptorMatch::secondDistance),
                           decltype(DescriptorMatch::secondDistance)>::value,
              "the ratio test's second distance is not narrowed");

static_assert(sizeof(DeviceStereoMatch) == 16,
              "one 128-bit store per match is why the index is narrowed");
static_assert(std::is_same<decltype(DeviceStereoMatch::disparity),
                           decltype(StereoMatch::disparity)>::value,
              "the disparity is the answer and is never narrowed");
static_assert(std::is_same<decltype(DeviceStereoMatch::distance),
                           decltype(StereoMatch::distance)>::value,
              "the Hamming distance is not narrowed: it is compared against the host's");
static_assert(std::is_same<decltype(StereoMatch::rightIndex), size_t>::value,
              "the device narrows this index to uint32_t under the documented domain; "
              "if the host type changed, re-derive the narrowing");

/// @brief The largest keypoint set this backend can index or count.
/// @note The domain in one constant: `uint32_t` counts everywhere, because the
/// append counter is a 32-bit atomic and an index wider than its counter
/// would be a field that can name an element the set cannot hold.
inline constexpr size_t kDeviceMaxKeypoints = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Whole-set conversions to the host library's types
// ---------------------------------------------------------------------------

/// @brief Converts a downloaded device result set to the host library's type.
/// @note The batched spelling, because that is the only spelling the families
/// produce. Both halves are HOST memory: the device-to-host move is
/// `downloadAppended` (compaction.hpp), and this is the type change after
/// it. For the byte-identical corners the loop is a copy the compiler
/// collapses; it is written the same way for all four so a caller does not
/// have to remember which is which.
inline void toHost(const DeviceFastCorner* src, size_t count, FastCorner* dst) {
    for (size_t i = 0; i < count; ++i) dst[i] = src[i].toHost();
}
inline void toHost(const DeviceCorner* src, size_t count, Corner* dst) {
    for (size_t i = 0; i < count; ++i) dst[i] = src[i].toHost();
}
inline void toHost(const DeviceDescriptorMatch* src, size_t count, DescriptorMatch* dst) {
    for (size_t i = 0; i < count; ++i) dst[i] = src[i].toHost();
}
inline void toHost(const DeviceStereoMatch* src, size_t count, StereoMatch* dst) {
    for (size_t i = 0; i < count; ++i) dst[i] = src[i].toHost();
}

// ---------------------------------------------------------------------------
// Checked factories -- where a host `size_t` count meets the device's domain
// ---------------------------------------------------------------------------

/// @brief Names a device keypoint set, checking the count domain.
inline DeviceKeypointSetConstView keypointSet(const float* xy, size_t count,
                                              const int32_t* octave = nullptr) {
    BINCV_ASSERT(count <= kDeviceMaxKeypoints,
                 "keypointSet: count outside the device's uint32 keypoint domain");
    BINCV_ASSERT(count == 0 || xy != nullptr,
                 "keypointSet: a non-empty set needs a non-null position array");
    return DeviceKeypointSetConstView{xy, static_cast<uint32_t>(count), octave};
}

/// @brief Names a mutable device descriptor set, checking the count domain.
inline DeviceDescriptorSetView descriptorSet(uint32_t* words, size_t count,
                                             size_t wordsPerDescriptor,
                                             uint8_t* keep = nullptr) {
    BINCV_ASSERT(count <= kDeviceMaxKeypoints,
                 "descriptorSet: count outside the device's uint32 keypoint domain");
    BINCV_ASSERT(wordsPerDescriptor >= 1 && wordsPerDescriptor <= 64,
                 "descriptorSet: a descriptor is 32 to 2048 bits of whole words");
    BINCV_ASSERT(count == 0 || words != nullptr,
                 "descriptorSet: a non-empty set needs a non-null word array");
    return DeviceDescriptorSetView{words, static_cast<uint32_t>(count),
                                   static_cast<uint32_t>(wordsPerDescriptor), keep};
}

/// @brief Names a read-only device descriptor set, checking the count domain.
inline DeviceDescriptorSetConstView descriptorSet(const uint32_t* words, size_t count,
                                                  size_t wordsPerDescriptor,
                                                  const uint8_t* keep = nullptr) {
    BINCV_ASSERT(count <= kDeviceMaxKeypoints,
                 "descriptorSet: count outside the device's uint32 keypoint domain");
    BINCV_ASSERT(wordsPerDescriptor >= 1 && wordsPerDescriptor <= 64,
                 "descriptorSet: a descriptor is 32 to 2048 bits of whole words");
    BINCV_ASSERT(count == 0 || words != nullptr,
                 "descriptorSet: a non-empty set needs a non-null word array");
    return DeviceDescriptorSetConstView{words, static_cast<uint32_t>(count),
                                        static_cast<uint32_t>(wordsPerDescriptor), keep};
}

// ---------------------------------------------------------------------------
// The append targets the detection families take, named once
// ---------------------------------------------------------------------------

/// @brief What a FAST detection kernel appends into. See compaction.hpp for the
/// capacity contract -- truncate, count the truth, and make the caller say
/// which number they meant.
using DeviceFastCornerBuffer = DeviceAppendBufferView<DeviceFastCorner>;

/// @brief What a good-features selection kernel appends into.
using DeviceCornerBuffer = DeviceAppendBufferView<DeviceCorner>;

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
