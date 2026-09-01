#pragma once

/// @file pnm.hpp
/// @brief Reading and writing PGM/PNM, with **no dependency at all**. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// THE `none` BACKEND OF `bincv_io`, AND WHY IT IS THE DEFAULT ONE
///
/// A PNG decoder is **eight times the size of everything binCV does** — measured, in
/// [ARCHITECTURE §7.8](../../../../docs/ARCHITECTURE.md): `libpng` + `libz` is 336 KB against
/// binCV's ~41 KB of code, and `libjpeg` is 510 KB. binCV must not carry one.
///
/// PNM is the format whose whole encoder is a header and a copy, and whose decoder is
/// a small parser. That makes it the one image format a library like this can own
/// outright. It is enough to **look at what binCV produced** on a target with no
/// OpenCV, and enough to feed it a test image — which is all the `none` backend
/// promises.
///
/// Encoded formats are a **tooling** concern, not a library one: deployment reads a
/// sensor and needs no codec, and nobody decodes a PNG in a VIO hot loop.
///
/// ---------------------------------------------------------------------------
/// BUFFERS, NOT PATHS
///
/// `bincv_core` does no file I/O, has no allocator and builds without exceptions.
/// These take a byte range and give one back; **where the bytes come from is the
/// caller's**, which is also what makes them usable from a filesystem-less target
/// that has the image in flash.

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

} // namespace impl

/// @brief Parses a binary PGM (`P5`) header. **API TIER 3.**
/// @note `P5` only. `P2` is ASCII pixels and exists mostly in tutorials; supporting it
///       would double the parser to read images nobody produces.
/// @note Never allocates and never throws. A malformed buffer returns `valid == false`
///       rather than reporting an error some other way -- this is core, and core has
///       neither exceptions nor an error channel.
inline PgmHeader readPgmHeader(const uint8_t* data, size_t size) {
    PgmHeader h;
    if (data == nullptr || size < 2 || data[0] != 'P' || data[1] != '5') return h;
    size_t i = impl::pnmSkip(data, size, 2);
    size_t w = 0, ht = 0, mx = 0;
    i = impl::pnmNumber(data, size, i, w);
    if (i > size) return h;
    i = impl::pnmSkip(data, size, i);
    i = impl::pnmNumber(data, size, i, ht);
    if (i > size) return h;
    i = impl::pnmSkip(data, size, i);
    i = impl::pnmNumber(data, size, i, mx);
    if (i > size || mx == 0 || mx > 65535) return h;
    // EXACTLY ONE whitespace byte separates the header from the pixels -- the format
    // says so, and consuming more would eat a pixel whose value happens to be 0x20.
    if (i >= size) return h;
    ++i;
    if (size - i < w * ht * (mx > 255 ? 2u : 1u)) return h;
    h.width = w;
    h.height = ht;
    h.maxValue = static_cast<unsigned>(mx);
    h.pixelOffset = i;
    h.valid = true;
    return h;
}

/// @brief Reads a binary PGM straight into bits, under a `PackRule`. **API TIER 3.**
/// @return False if the buffer is not a usable `P5`, or `dst` is the wrong size.
/// @note **8-bit maxima only.** A 16-bit PGM stores big-endian samples, which is a
///       byte order binCV would have to swap; `readPgmHeader` reports `maxValue` so a
///       caller can detect and reject one rather than being handed silent nonsense.
/// @note The pixels never become an 8-bit image inside binCV -- this packs from the
///       file's own buffer.
template <PackRule R, typename WordType>
inline bool readPgm(const uint8_t* data, size_t size, BinMatView<WordType> dst,
                    uint8_t t = 0) {
    const PgmHeader h = readPgmHeader(data, size);
    if (!h.valid || h.maxValue > 255) return false;
    if (h.width != dst.width || h.height != dst.height) return false;
    packBits<R, uint8_t, WordType>(data + h.pixelOffset, h.width, h.height, h.width, dst, t);
    return true;
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
