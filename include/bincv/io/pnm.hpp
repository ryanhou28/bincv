#pragma once

/// @file pnm.hpp
/// @brief Reading PNM — `P4` and `P5` — with **no dependency at all**. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// binCV LINKS NO CODEC, ON ANY TARGET
///
/// Not austerity for small parts. Every tier's real frame source already **is** the
/// input contract: a capture SDK's buffer, a camera's YUV420 Y plane, a V4L2 buffer
/// and a sensor's DMA rows are all single-channel strided integer arrays, which is
/// why `packBits` and `packRows` take a stride — so they pack with no conversion at
/// all. Nothing on a caller's path decodes anything.
///
/// Encoded files turn up in exactly one place, identically on every tier: reading a
/// **dataset** to test or benchmark against. That is tooling, and tooling runs on a
/// host — including on desktop, where the host already has OpenCV.
///
/// PNM is what a library like this can own outright: `P4` is a header and a copy,
/// `P5` a small parser. That is all these promise — enough to **look at what binCV
/// produced** on a target with no OpenCV, and enough to feed it a test image.
///
/// ---------------------------------------------------------------------------
/// P4 IS THE FORMAT THIS LIBRARY IS SHAPED LIKE
///
/// `P4` stores one bit per pixel, which is binCV's own layout, so the file is the size
/// of the matrix. `P5` stores one **byte** per pixel: a 752x480 frame is 45,120 bytes
/// packed and 360,960 unpacked, so reading or writing it as `P5` moves a buffer eight
/// times larger than the image it represents — on the target where buffers are
/// scarcest. Reach for `readPbm`/`writePbm` unless grey levels are the point.
///
/// The two differ only in bit order: binCV puts pixel `x` at bit `x % WordBits`, least
/// significant first, and `P4` puts a row's leftmost pixel at the most significant bit
/// of its first byte. `impl::reverseByte` is the whole conversion.
///
/// ---------------------------------------------------------------------------
/// BUFFERS, NOT PATHS
///
/// `bincv_core` does no file I/O, has no allocator and builds without exceptions.
/// These take a byte range and give one back; **where the bytes come from is the
/// caller's**, which is also what makes them usable from a filesystem-less target
/// that has the image in flash.
///
/// A `P5` whole-buffer read needs the entire 8-bit frame addressable at once. Out of
/// memory-mapped flash that costs no RAM; off a UART or an SD card it costs exactly
/// the frame binCV exists not to hold. `readPgmHeaderFromPrefix` parses the header
/// from the first bytes to arrive, after which `packRows` consumes the pixels a chunk
/// of rows at a time and the frame is never materialized. `P4` needs no such form:
/// its file already is the matrix, so there is no wide intermediate to stream around.

#include <cstddef>
#include <cstdint>

#include "../binMat.hpp"
#include "../ops/pack.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief What a `readPgm` call found, or why it did not.
struct PgmHeader {
    size_t width = 0;
    size_t height = 0;
    unsigned maxValue = 0;
    size_t pixelOffset = 0;   ///< byte offset of the first pixel
    size_t bodyBytes = 0;     ///< bytes of pixel data the header describes
    bool valid = false;
};

/// @brief What a `readPbm` call found, or why it did not.
/// @note No `maxValue`: `P4` pixels are bits, so the format has no such line.
struct PbmHeader {
    size_t width = 0;
    size_t height = 0;
    size_t pixelOffset = 0;   ///< byte offset of the first pixel
    size_t bodyBytes = 0;     ///< bytes of pixel data the header describes
    bool valid = false;
};

namespace impl {

/// @brief Skips whitespace and `#` comments, PNM's rule. **INTERNAL.**
inline size_t pnmSkip(const uint8_t* d, size_t n, size_t i) {
    for (;;) {
        while (i < n && (d[i] == ' ' || d[i] == '\t' || d[i] == '\n' || d[i] == '\r')) ++i;
        if (i < n && d[i] == '#') {
            while (i < n && d[i] != '\n') ++i;
            continue;
        }
        return i;
    }
}

inline size_t pnmNumber(const uint8_t* d, size_t n, size_t i, size_t& out) {
    out = 0;
    size_t digits = 0;
    while (i < n && d[i] >= '0' && d[i] <= '9') {
        out = out * 10 + static_cast<size_t>(d[i] - '0');
        ++i;
        ++digits;
    }
    return digits == 0 ? n + 1 : i;   // n+1 signals "no number here"
}

/// @brief Everything the two public header parses share. **INTERNAL.**
struct PnmHead {
    size_t width = 0;
    size_t height = 0;
    unsigned maxValue = 0;
    size_t pixelOffset = 0;
    size_t bodyBytes = 0;
    bool valid = false;
};

/// @brief Parses a `P4` or `P5` header. **INTERNAL.**
/// @param magic `'4'` or `'5'`. The two headers differ only in that `P4`, whose
/// pixels are bits, has no maximum-value line.
/// @param requireBody Whether the buffer must also hold the pixels the header
/// describes. False lets a streaming caller parse the header out of the first
/// bytes to arrive and take the pixels later.
inline PnmHead pnmParse(const uint8_t* d, size_t n, char magic, bool requireBody) {
    PnmHead h;
    if (d == nullptr || n < 2 || d[0] != 'P' || d[1] != static_cast<uint8_t>(magic)) return h;
    size_t i = pnmSkip(d, n, 2);
    size_t w = 0, ht = 0, mx = 1;
    i = pnmNumber(d, n, i, w);
    if (i > n) return h;
    i = pnmSkip(d, n, i);
    i = pnmNumber(d, n, i, ht);
    if (i > n) return h;
    if (magic == '5') {
        i = pnmSkip(d, n, i);
        i = pnmNumber(d, n, i, mx);
        if (i > n || mx == 0 || mx > 65535) return h;
    }
    // EXACTLY ONE whitespace byte separates the header from the pixels -- the format
    // says so, and consuming more would eat a pixel whose value happens to be 0x20.
    if (i >= n) return h;
    ++i;

    constexpr size_t kMax = static_cast<size_t>(-1);
    const size_t perPixel = (magic == '5' && mx > 255) ? 2u : 1u;
    if (magic != '4' && w != 0 && perPixel > kMax / w) return h;
    const size_t rowBytes = magic == '4' ? pbmRowBytes(w) : w * perPixel;
    // A header claiming more pixels than a size_t can count is malformed, not large:
    // without this the product wraps and a truncated body passes the size check.
    if (rowBytes != 0 && ht > kMax / rowBytes) return h;
    h.bodyBytes = rowBytes * ht;
    if (requireBody && n - i < h.bodyBytes) return h;

    h.width = w;
    h.height = ht;
    h.maxValue = static_cast<unsigned>(mx);
    h.pixelOffset = i;
    h.valid = true;
    return h;
}

} // namespace impl

/// @brief Parses a binary PGM (`P5`) header. **API TIER 3.**
/// @note `P5` only. `P2` is ASCII pixels and exists mostly in tutorials; supporting it
/// would double the parser to read images nobody produces.
/// @note Never allocates and never throws. A malformed buffer returns `valid == false`
/// rather than reporting an error some other way -- this is core, and core has
/// neither exceptions nor an error channel.
inline PgmHeader readPgmHeader(const uint8_t* data, size_t size) {
    const impl::PnmHead p = impl::pnmParse(data, size, '5', true);
    PgmHeader h;
    h.width = p.width;
    h.height = p.height;
    h.maxValue = p.maxValue;
    h.pixelOffset = p.pixelOffset;
    h.bodyBytes = p.bodyBytes;
    h.valid = p.valid;
    return h;
}

/// @brief The same parse, from a **prefix** of the file. **API TIER 3.**
/// @note **This is what makes a `P5` readable without holding one.** `readPgmHeader`
/// requires the pixels to be present, so a caller taking a frame off a UART or an
/// SD card cannot parse the header until it has buffered the whole 8-bit image --
/// the exact frame binCV exists not to hold. This parses from the first bytes to
/// arrive and reports `pixelOffset` and `bodyBytes`; the caller then hands rows to
/// `packRows` as they come in, and nothing wide is ever allocated. Chunk boundaries
/// cannot change the result, so the streamed image is bit-identical to the
/// whole-buffer one.
/// @note Validates the header only. A buffer holding a good header and no pixels is
/// `valid` here and is not a usable `readPgm` argument.
inline PgmHeader readPgmHeaderFromPrefix(const uint8_t* data, size_t size) {
    const impl::PnmHead p = impl::pnmParse(data, size, '5', false);
    PgmHeader h;
    h.width = p.width;
    h.height = p.height;
    h.maxValue = p.maxValue;
    h.pixelOffset = p.pixelOffset;
    h.bodyBytes = p.bodyBytes;
    h.valid = p.valid;
    return h;
}

/// @brief Parses a binary PBM (`P4`) header. **API TIER 3.**
/// @note `P4` only, for the same reason `readPgmHeader` takes `P5` only: `P1` is
/// ASCII pixels and exists mostly in tutorials.
/// @note Never allocates and never throws; a malformed buffer returns
/// `valid == false`.
inline PbmHeader readPbmHeader(const uint8_t* data, size_t size) {
    const impl::PnmHead p = impl::pnmParse(data, size, '4', true);
    PbmHeader h;
    h.width = p.width;
    h.height = p.height;
    h.pixelOffset = p.pixelOffset;
    h.bodyBytes = p.bodyBytes;
    h.valid = p.valid;
    return h;
}

/// @brief Reads a binary PGM straight into bits, under a `PackRule`. **API TIER 3.**
/// @return False if the buffer is not a usable `P5`, or `dst` is the wrong size.
/// @note **8-bit maxima only.** A 16-bit PGM stores big-endian samples, which is a
/// byte order binCV would have to swap; `readPgmHeader` reports `maxValue` so a
/// caller can detect and reject one rather than being handed silent nonsense.
/// @note The pixels never become an 8-bit image inside binCV -- this packs from the
/// file's own buffer.
/// @note **The whole file must be addressable**, because a `P5` body is a byte per
/// pixel and this reads all of it. Out of memory-mapped flash that costs no RAM.
/// Off a UART or an SD card, pair `readPgmHeaderFromPrefix` with `packRows` instead
/// and the frame is never materialized.
template <PackRule R, typename WordType>
inline bool readPgm(const uint8_t* data, size_t size, BinMatView<WordType> dst,
                    uint8_t t = 0) {
    const PgmHeader h = readPgmHeader(data, size);
    if (!h.valid || h.maxValue > 255) return false;
    if (h.width != dst.width || h.height != dst.height) return false;
    packBits<R, uint8_t, WordType>(data + h.pixelOffset, h.width, h.height, h.width, dst, t);
    return true;
}

/// @brief Reads a binary PBM (`P4`) into a bit matrix. **API TIER 3.**
/// @return False if the buffer is not a usable `P4`, or `dst` is the wrong size.
///
/// @note **No `PackRule` and no threshold, because `P4` pixels are already bits.**
/// That is what makes this the cheap way to get a known image onto a small target:
/// the file is the size of the matrix, so unlike `readPgm` there is no wide
/// intermediate at any point, and no whole-frame buffer to find.
/// @note **`P4` sets a bit for BLACK**, which is the format's convention and the
/// opposite of `writePgm`'s default mapping of a set bit to white. `writePbm` and
/// this agree with each other, so a round trip is exact; a viewer simply shows set
/// pixels dark. Pass `onValue = 0, zeroValue = 255` to `writePgm` if the two need
/// to look alike.
/// @note `dst`'s padding bits are zero on return. The format calls a row's trailing
/// bits padding and ignores them, so a file may carry anything there; binCV may
/// not, since word-wise reductions would over-count.
template <typename WordType>
inline bool readPbm(const uint8_t* data, size_t size, BinMatView<WordType> dst) {
    const PbmHeader h = readPbmHeader(data, size);
    if (!h.valid) return false;
    if (h.width != dst.width || h.height != dst.height) return false;
    if (dst.width == 0 || dst.height == 0) return true;
    if (dst.ptr == nullptr) return false;

    constexpr size_t kWordBits = BinMatView<WordType>::WordBits;
    const size_t rowBytes = impl::pbmRowBytes(dst.width);
    const size_t words = impl::minRowWords<WordType>(dst.width);
    for (size_t y = 0; y < dst.height; ++y) {
        WordType* out = dst.row(y);
        for (size_t i = 0; i < words; ++i) out[i] = 0;
        const uint8_t* in = data + h.pixelOffset + y * rowBytes;
        for (size_t j = 0; j < rowBytes; ++j) {
            const size_t x0 = j * 8;
            uint8_t b = impl::reverseByte(in[j]);
            if (x0 + 8 > dst.width) {
                b = static_cast<uint8_t>(b & ((1u << (dst.width - x0)) - 1u));
            }
            const WordType bits = static_cast<WordType>(static_cast<WordType>(b)
                                                        << (x0 % kWordBits));
            out[x0 / kWordBits] = static_cast<WordType>(out[x0 / kWordBits] | bits);
        }
    }
    return true;
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
