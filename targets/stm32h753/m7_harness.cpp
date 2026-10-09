#include "m7_harness.hpp"

#include <cstring>

#include "bincv/core/simd.hpp"

#ifndef BINCV_M7_BAUD
#define BINCV_M7_BAUD 4000000u
#endif

namespace m7 {

namespace {

constexpr uint32_t kProtocol = 1;

bool startsWithWord(const char* line, const char* word, const char*& args) {
    const size_t n = std::strlen(word);
    if (std::strncmp(line, word, n) != 0) return false;
    if (line[n] != '\0' && line[n] != ' ') return false;
    args = line + n;
    while (*args == ' ') ++args;
    return true;
}

void cmdHello(const char* image) {
    boardPuts("HELLO bincv-m7 image=");
    boardPuts(image);
    boardPuts(" proto=");
    boardPutU32(kProtocol);
    boardPuts(" clock_hz=");
    boardPutU32(boardClockHz());
    boardPuts(" baud=");
    boardPutU32(BINCV_M7_BAUD);
    boardPuts(" stack_bytes=");
    boardPutU32(static_cast<uint32_t>(boardStackSize()));
    boardPuts(" caches=ID\nSIMD ");
    boardPuts(bincv::simdStatusString());
    boardPuts("\nOK\n");
}

void cmdClock() {
    const uint32_t hz = boardMeasureClockHz();
    if (hz == 0) {
        boardPuts("ERR the 32.768 kHz crystal did not start; the clock is unverified\n");
        return;
    }
    boardPuts("CLOCK configured_hz=");
    boardPutU32(boardClockHz());
    boardPuts(" measured_hz=");
    boardPutU32(hz);
    boardPuts("\nOK\n");
}

/// The link test. The CRC is folded in byte by byte as each byte arrives -- a few
/// dozen cycles against the ~1200 a byte takes to arrive at 4 Mbaud -- so the test
/// needs no buffer and checks every rate up to the one the core can no longer keep
/// up with.
void cmdEcho(const char* args) {
    uint32_t n = 0;
    if (!parseU32(args, n)) {
        boardPuts("ERR ECHO needs a byte count\n");
        return;
    }
    boardPuts("SEND ");
    boardPutU32(n);
    boardPutc('\n');
    const uint32_t t0 = boardCycles();
    uint32_t crc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const int c = boardGetc();
        if (c < 0) {
            boardPuts("ERR receive overrun\n");
            return;
        }
        const uint8_t b = static_cast<uint8_t>(c);
        crc = crc32Update(crc, &b, 1);
    }
    uint32_t sent = 0;
    if (boardRead(&sent, sizeof sent) != 0) {
        boardPuts("ERR receive overrun\n");
        return;
    }
    const uint32_t cycles = boardCycles() - t0;
    boardPuts("ECHO bytes=");
    boardPutU32(n);
    boardPuts(" crc=");
    boardPutHex32(crc);
    boardPuts(" receive_cycles=");
    boardPutU32(cycles);
    boardPutc('\n');
    boardPuts(crc == sent ? "OK\n" : "ERR crc mismatch\n");
}

}  // namespace

uint32_t crc32Update(uint32_t crc, const uint8_t* p, size_t n) {
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

bool parseU32(const char*& s, uint32_t& out) {
    while (*s == ' ') ++s;
    if (*s < '0' || *s > '9') return false;
    uint64_t v = 0;
    while (*s >= '0' && *s <= '9') {
        v = v * 10u + static_cast<uint64_t>(*s - '0');
        if (v > 0xFFFFFFFFu) return false;
        ++s;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

uint32_t medianOf(uint32_t* v, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        const uint32_t k = v[i];
        size_t j = i;
        while (j > 0 && v[j - 1] > k) {
            v[j] = v[j - 1];
            --j;
        }
        v[j] = k;
    }
    return v[n / 2];
}

void putRatio(uint32_t num, uint32_t den) {
    const uint32_t pct = den ? static_cast<uint32_t>(static_cast<uint64_t>(num) * 100u / den) : 0u;
    boardPutU32(pct / 100u);
    boardPutc('.');
    if ((pct % 100u) < 10u) boardPutc('0');
    boardPutU32(pct % 100u);
}

void serve(const char* image, const Command* commands, size_t count) {
    // The cycle counter first: the clock check is timed on it.
    const int counterOk = boardCycleCounterInit();
    const int clockErr = boardClockInit();
    if (clockErr != 0) {
        boardSerialInitAt(64000000u, 115200u);
        boardPuts("\nFATAL: clock bring-up failed at step ");
        boardPutU32(static_cast<uint32_t>(clockErr));
        boardPuts(" (board.h lists them); nothing will be timed.\n");
        for (;;) {
        }
    }
    boardSerialInit(BINCV_M7_BAUD);
    boardEnableCaches();
    if (!counterOk) {
        boardPuts("\nFATAL: DWT cycle counter does not advance; no timing is possible.\n");
        for (;;) {
        }
    }
    boardStackPaint();

    // Nobody may be listening yet -- flashing over the ST-LINK's mass-storage
    // interface resets the part as the copy completes -- so the boot line is a
    // courtesy, and a host starts every session with PING.
    boardPuts("\nBOOT bincv-m7 image=");
    boardPuts(image);
    boardPutc('\n');

    char line[128];
    for (;;) {
        const int n = boardReadLine(line, sizeof line);
        if (n < 0) {
            boardPuts("ERR receive overrun\n");
            continue;
        }
        if (n == 0) continue;
        const char* args = nullptr;
        if (std::strcmp(line, "PING") == 0) {
            cmdHello(image);
        } else if (std::strcmp(line, "CLOCK") == 0) {
            cmdClock();
        } else if (startsWithWord(line, "ECHO", args)) {
            cmdEcho(args);
        } else if (std::strcmp(line, "STACK") == 0) {
            boardPuts("STACK used_bytes=");
            boardPutU32(static_cast<uint32_t>(boardStackUsed()));
            boardPuts(" of ");
            boardPutU32(static_cast<uint32_t>(boardStackSize()));
            boardPuts("\nOK\n");
        } else if (std::strcmp(line, "RESET") == 0) {
            boardPuts("OK\n");
            boardReset();
        } else {
            bool handled = false;
            for (size_t i = 0; i < count && !handled; ++i) {
                if (startsWithWord(line, commands[i].name, args)) {
                    commands[i].run(args);
                    handled = true;
                }
            }
            if (!handled) boardPuts("ERR unknown command\n");
        }
    }
}

}  // namespace m7
