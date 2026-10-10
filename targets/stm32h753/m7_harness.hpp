/// @file m7_harness.hpp
/// @brief What every STM32H753 benchmark image shares: bring-up, and the command loop
/// a host drives over the ST-LINK's virtual COM port.
///
/// There is one firmware image per benchmark, each named after its own translation
/// unit (bincv_m7_reduce.cpp is bincv_m7_reduce), for the reason there is one host
/// binary per benchmark: scripts/check_figure_staleness.py ages a figure when a file
/// its image compiles changes, and an image that compiled every kernel would age
/// every figure on every edit.
///
/// The host side is scripts/m7_host.py. Every exchange is a request and a reply: the
/// host sends one command line, the board answers with text lines ending in a line
/// that starts `OK` or `ERR`, and binary data only travels when the board has asked
/// for exactly that many bytes. So the board never receives while it computes,
/// nothing it times includes the link, and the receive FIFO cannot overrun.
///
/// Commands every image answers:
///
///   PING            HELLO line: image, protocol, configured clock, link rate,
///                   stack size, and binCV's SIMD status
///   CLOCK           the core clock as the board's 32.768 kHz crystal measures it
///   ECHO <n>        receive n bytes and a CRC-32, check it -- the link test
///   STACK           stack high-water mark since the last reset
///   RESET           system reset, so a host can take a fresh launch
///
/// plus whatever the image registers.

#pragma once

#include <cstddef>
#include <cstdint>

#include "board.h"

namespace m7 {

struct Command {
    const char* name;               ///< matched against the first word of the line
    void (*run)(const char* args);  ///< the rest of the line; must end with OK or ERR
};

/// Brings the board up and serves commands forever. `image` is the name the HELLO
/// line reports. Never returns.
[[noreturn]] void serve(const char* image, const Command* commands, size_t count);

/// CRC-32 as zlib computes it, so the host checks a transfer with `zlib.crc32`.
uint32_t crc32Update(uint32_t crc, const uint8_t* p, size_t n);

/// Parses an unsigned decimal at `s` and advances past it. No sscanf, so no newlib
/// scanf machinery in the image.
bool parseU32(const char*& s, uint32_t& out);

/// Median of `n` samples; sorts them in place.
uint32_t medianOf(uint32_t* v, size_t n);

/// `num / den` to two decimals, as an integer pair: no floats, so no soft-float
/// formatting in the image.
void putRatio(uint32_t num, uint32_t den);

}  // namespace m7
