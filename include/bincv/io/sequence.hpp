#pragma once

/// @file sequence.hpp
/// @brief Reading a frame-sequence blob -- `BSQ1` -- with **no dependency at all**.
/// **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHY A BLOB AND NOT A DIRECTORY OF FILES
///
/// The design notes move all decoding to the host: nothing on a caller's real
/// path decodes anything, and encoded files turn up only when reading a
/// **dataset** to test against. What carries the decoded frames to a target is
/// this: one flat file, a small header, frames back to back. A bare-metal
/// target has no filesystem, no directory iteration and no `argv`, and one flat
/// byte range works unchanged as a file to `fread` or mmap on desktop, an app
/// asset on mobile, and an `xxd -i` array in flash -- or a stream over
/// USB/UART -- on bare metal. `scripts/make_sequence_blob.py` is the writer;
/// it runs on the host, where the decoders live.
///
/// ---------------------------------------------------------------------------
/// THE FORMAT
///
/// A 32-byte fixed header, then `frameCount` frame bodies with no per-frame
/// framing -- every frame is the same size, so frame `i` is at a computed
/// offset and random access costs nothing:
///
///     bytes  0-3   magic "BSQ1"
///     bytes  4-7   mode: 0 = 8-bit bodies, 1 = packed 1-bit bodies   (u32)
///     bytes  8-11  width in pixels                                   (u32)
///     bytes 12-15  height in pixels                                  (u32)
///     bytes 16-19  frame count                                       (u32)
///     bytes 20-31  reserved, zero
///
/// The u32 fields are **little-endian**. That is a decision, not a hedge: every
/// target binCV runs on -- x86-64, aarch64, Cortex-M, RISC-V -- is
/// little-endian, so the natural byte order is also the portable one here.
/// The parse below reads bytes, so even an exotic host gets the right answer.
///
/// A frame body is EXACTLY the body of a PNM image -- pnm.hpp's formats, reused
/// deliberately so the pixel layout has one definition rather than a second one
/// to keep bit-identical forever:
///
/// * **mode 0** -- a `P5` body: `width * height` bytes, row-major, one byte per
///   pixel. The body IS the input contract (a single-channel, strided pixel
///   array), so a caller feeds it to `packBits`, `edgeThreshold` or `medianWide`
///   directly, from the blob's own bytes. ~353 KB per 752x480 frame, and it
///   exercises the WHOLE pipeline, sensor stage and packing included.
/// * **mode 1** -- a `P4` body: rows padded to a byte boundary, leftmost pixel
///   in the most significant bit. ~44 KB per 752x480 frame -- 8x more frames
///   in the same flash -- at the cost of coverage: the sensor stage ran on the
///   host, so the blob tests only what is downstream of the binary frame.
///
/// The header records which mode a blob is so a reader cannot confuse them, and
/// it records the dimensions and count so a target that cannot allocate can
/// reject a blob it has no room for BEFORE reading a single frame --
/// `frameBytes` and `frameCount` are exactly the two numbers that decision
/// needs.
///
/// ---------------------------------------------------------------------------
/// BUFFERS, NOT PATHS -- AND A PREFIX IS ENOUGH FOR THE HEADER
///
/// Like everything in io/, these take a byte range; where the bytes come from
/// is the caller's. `readSequenceHeader` needs only the first 32 bytes, which
/// is what makes it usable off a UART before anything wide has arrived; the
/// per-frame entry points then bounds-check every access, so a truncated blob
/// reports invalid rather than reading past the end. A streaming caller that
/// holds one frame at a time uses `readSequenceFrameBody` on the body bytes it
/// just received -- the whole blob is never resident, which for mode 0 is the
/// frame binCV exists not to hold.

#include <cstddef>
#include <cstdint>

#include "../binMat.hpp"
#include "../ops/pack.hpp"
#include "pnm.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief `mode` value for 8-bit (`P5`-shaped) frame bodies.
inline constexpr uint32_t kSequenceMode8Bit = 0;
/// @brief `mode` value for packed 1-bit (`P4`-shaped) frame bodies.
inline constexpr uint32_t kSequenceModePacked = 1;
/// @brief The fixed header size; frame 0's body starts here.
inline constexpr size_t kSequenceHeaderBytes = 32;

/// @brief What a `readSequenceHeader` call found, or why it did not.
struct SequenceHeader {
    uint32_t mode = 0;      ///< `kSequenceMode8Bit` or `kSequenceModePacked`
    size_t width = 0;
    size_t height = 0;
    size_t frameCount = 0;
    size_t frameBytes = 0;  ///< bytes of ONE frame's body
    bool valid = false;
};

/// @brief One frame's body within a blob, or `valid == false`.
struct SequenceFrameRange {
    const uint8_t* data = nullptr;
    size_t size = 0;        ///< always `frameBytes` when valid
    bool valid = false;
};

/// @brief Parses a `BSQ1` header from the first 32 bytes. **API TIER 3.**
/// @note Validates the header only, like `readPgmHeaderFromPrefix` and for the
/// same reason: a streaming caller has the first bytes to arrive, not the
/// blob. Whether the buffer also holds frame `i` is `sequenceFrame`'s check.
/// @note Rejects rather than misreads: a wrong magic, an unknown mode, a zero
/// dimension, a non-zero reserved byte, or a frame size that would not fit
/// in a `size_t` all return `valid == false`. The reserved bytes are
/// required zero so a future revision that uses them is REJECTED here
/// instead of being read as if it were plain `BSQ1`.
/// @note The `size_t` overflow guard is what makes the header safe at 32-bit
/// pointer width: a header claiming more bytes per frame than a `size_t`
/// can count is malformed, not large, and without the guard the product
/// wraps and a truncated blob passes every later size check. A 64-bit
/// `size_t` cannot overflow from u32 fields -- `(2^32-1)^2 < 2^64` -- so
/// this branch is live exactly where verify_cortex_m.sh compiles it.
/// @note Never allocates and never throws; this is core, and core has neither
/// exceptions nor an error channel.
inline SequenceHeader readSequenceHeader(const uint8_t* data, size_t size) {
    SequenceHeader h;
    if (data == nullptr || size < kSequenceHeaderBytes) return h;
    if (data[0] != 'B' || data[1] != 'S' || data[2] != 'Q' || data[3] != '1') return h;
    const auto u32 = [&](size_t at) -> uint32_t {
        return static_cast<uint32_t>(data[at]) |
               (static_cast<uint32_t>(data[at + 1]) << 8) |
               (static_cast<uint32_t>(data[at + 2]) << 16) |
               (static_cast<uint32_t>(data[at + 3]) << 24);
    };
    const uint32_t mode = u32(4);
    const uint32_t w = u32(8);
    const uint32_t ht = u32(12);
    const uint32_t count = u32(16);
    if (mode > kSequenceModePacked) return h;
    if (w == 0 || ht == 0) return h;
    for (size_t i = 20; i < kSequenceHeaderBytes; ++i) {
        if (data[i] != 0) return h;
    }

    constexpr size_t kMax = static_cast<size_t>(-1);
    const size_t rowBytes = (mode == kSequenceModePacked)
                                ? impl::pbmRowBytes(static_cast<size_t>(w))
                                : static_cast<size_t>(w);
    if (static_cast<size_t>(ht) > kMax / rowBytes) return h;

    h.mode = mode;
    h.width = static_cast<size_t>(w);
    h.height = static_cast<size_t>(ht);
    h.frameCount = static_cast<size_t>(count);
    h.frameBytes = rowBytes * static_cast<size_t>(ht);
    h.valid = true;
    return h;
}

/// @brief Frame `i`'s body bytes, bounds-checked. **API TIER 3.**
/// @note A blob too short to hold frame `i` -- truncated in transfer, or a
/// header lying about its count -- returns `valid == false` rather than a
/// range past the end. The offset arithmetic is overflow-guarded for the
/// same 32-bit reason `readSequenceHeader`'s is.
/// @note For mode 0 this IS the read: the body is a single-channel, strided
/// pixel array -- the input contract -- so the caller hands it to
/// `packBits` / `medianWide` / `edgeThreshold` directly from the blob's own
/// bytes. There is deliberately no decode-to-wide-buffer convenience here:
/// it would copy out the one frame binCV exists not to hold.
inline SequenceFrameRange sequenceFrame(const SequenceHeader& h, const uint8_t* data,
                                        size_t size, size_t i) {
    SequenceFrameRange r;
    if (!h.valid || data == nullptr || i >= h.frameCount) return r;
    constexpr size_t kMax = static_cast<size_t>(-1);
    if (i > (kMax - kSequenceHeaderBytes) / h.frameBytes) return r;
    const size_t offset = kSequenceHeaderBytes + i * h.frameBytes;
    if (offset > size || size - offset < h.frameBytes) return r;
    r.data = data + offset;
    r.size = h.frameBytes;
    r.valid = true;
    return r;
}

/// @brief Unpacks ONE mode-1 frame body into a bit matrix. **API TIER 3.**
/// @return False if the header is not a valid packed-mode one, `dst` is the
/// wrong size, or `body` is too short.
/// @note **The streaming entry point.** A caller taking frames one at a time --
/// `fread` into a reused buffer, a UART transfer, a flash window -- holds
/// exactly one body, not the blob, so this takes the body's own byte range.
/// `readSequenceFrame` below is a call to this on `sequenceFrame`'s range.
/// @note The body is a `P4` body, and this reuses `readPbm`'s unpack rather
/// than restating the layout -- one definition of the bit order, in
/// io/pnm.hpp. `dst`'s padding bits are zero on return, whatever the
/// file's row padding carried.
template <typename WordType>
inline bool readSequenceFrameBody(const SequenceHeader& h, const uint8_t* body,
                                  size_t bodySize, BinMatView<WordType> dst) {
    if (!h.valid || h.mode != kSequenceModePacked) return false;
    if (h.width != dst.width || h.height != dst.height) return false;
    if (bodySize < h.frameBytes) return false;
    if (body == nullptr || dst.ptr == nullptr) return false;
    impl::readP4Body<WordType>(body, dst);
    return true;
}

/// @brief Reads mode-1 frame `i` of a whole blob into a bit matrix.
/// **API TIER 3.**
/// @return False if the blob is not a valid packed-mode one, `dst` is the wrong
/// size, or frame `i` is out of range or truncated.
template <typename WordType>
inline bool readSequenceFrame(const SequenceHeader& h, const uint8_t* data, size_t size,
                              size_t i, BinMatView<WordType> dst) {
    const SequenceFrameRange f = sequenceFrame(h, data, size, i);
    if (!f.valid) return false;
    return readSequenceFrameBody<WordType>(h, f.data, f.size, dst);
}

/// @brief Reads mode-0 frame `i` straight into bits, under a `PackRule`.
/// **API TIER 3.**
/// @return False if the blob is not a valid 8-bit-mode one, `dst` is the wrong
/// size, or frame `i` is out of range or truncated.
/// @note The mode-0 analogue of `readPgm`, with the same shape and the same
/// property: the pixels never become an 8-bit image inside binCV -- this
/// packs from the blob's own bytes. A plain threshold is NOT the sensor
/// stage; a caller wanting `medianWide` + `edgeThreshold` takes
/// `sequenceFrame`'s raw range and runs them on it directly.
template <PackRule R, typename WordType>
inline bool readSequenceFrame(const SequenceHeader& h, const uint8_t* data, size_t size,
                              size_t i, BinMatView<WordType> dst, uint8_t t = 0) {
    if (h.mode != kSequenceMode8Bit) return false;
    if (h.width != dst.width || h.height != dst.height) return false;
    const SequenceFrameRange f = sequenceFrame(h, data, size, i);
    if (!f.valid) return false;
    if (dst.ptr == nullptr) return false;
    packBits<R, uint8_t, WordType>(f.data, h.width, h.height, h.width, dst, t);
    return true;
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
