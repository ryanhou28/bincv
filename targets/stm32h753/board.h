/* Minimal board support for the NUCLEO-H753ZI: clock, serial in and out, cycle
 * counter, caches, stack high-water mark. Register definitions are inline from
 * RM0433 rather than from CMSIS -- see startup.c for why this target carries no
 * vendor tree.
 */
#ifndef BINCV_EMBEDDED_STM32H753_BOARD_H
#define BINCV_EMBEDDED_STM32H753_BOARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The core clock every figure is taken at. 480 MHz is the part's rated maximum:
 * PLL1 from the ST-LINK's 8 MHz clock, voltage scale 0. Defining
 * BINCV_M7_RESET_CLOCK leaves the part on its 64 MHz HSI reset clock instead, which
 * is what the first bring-up measured at.
 *
 * The configured number is not trusted on its own: boardMeasureClockHz() times the
 * core against the board's 32.768 kHz crystal, and the firmware prints both. A PLL
 * misconfigured into a different frequency still runs everything correctly, and
 * every number it produces is then plausible and wrong.
 */
#ifdef BINCV_M7_RESET_CLOCK
#define BINCV_M7_CLOCK_HZ 64000000u
#define BINCV_M7_PCLK1_HZ 64000000u
#else
#define BINCV_M7_CLOCK_HZ 480000000u
#define BINCV_M7_PCLK1_HZ 120000000u
#endif

/* Brings the clock tree to BINCV_M7_CLOCK_HZ. Returns 0 on success, or a nonzero
 * code naming the step that did not complete: 1 supply, 2 voltage scale, 3 HSE,
 * 4 PLL lock, 5 clock switch, 6 flash wait states. On failure the part may still be
 * on HSI at 64 MHz, and the serial port is NOT usable at the configured rates --
 * call boardSerialInitAt(64000000u, ...) to report the failure.
 */
int boardClockInit(void);

uint32_t boardClockHz(void);

/* Core cycles counted across one second of the board's 32.768 kHz LSE crystal, i.e.
 * the core clock in Hz as an independent oscillator sees it. 0 when the crystal
 * does not start. Takes the LSE's start-up time (up to about two seconds) plus one.
 */
uint32_t boardMeasureClockHz(void);

/* USART3 on PD8/PD9, which is what the ST-LINK's virtual COM port is wired to on
 * this board, 8N1, with the 16-byte receive FIFO on.
 */
void boardSerialInit(uint32_t baud);
void boardSerialInitAt(uint32_t pclk1Hz, uint32_t baud);
void boardPutc(char c);
void boardPuts(const char* s);
void boardWrite(const void* data, size_t n);

/* Blocking receive. The protocol is strictly request/response -- the board asks for
 * bytes and then waits for exactly that many -- so nothing ever arrives while the
 * core is busy and the FIFO cannot overrun. Returns -1 on a receive overrun, which
 * would mean that contract was broken.
 */
int boardGetc(void);
int boardRead(void* dst, size_t n);
/* One line, without its terminator; `\r` is dropped. Returns its length, or -1. */
int boardReadLine(char* dst, size_t cap);

/* Unsigned decimal; the harness prints no floats, so no soft-float formatting and no
 * newlib printf lands in the image. Both cost more flash than everything else here.
 */
void boardPutU32(uint32_t v);
void boardPutU64(uint64_t v);
void boardPutHex32(uint32_t v);

/* Cycle-accurate timing from the DWT. Returns false when the counter does not
 * advance, which is the one failure mode that would otherwise report every arm as
 * infinitely fast.
 */
int boardCycleCounterInit(void);

static inline uint32_t boardCycles(void) { return *(volatile uint32_t*)0xE0001004u; }

/* The DWT counter widened to 64 bits. The 32-bit counter wraps every 8.9 s at
 * 480 MHz; this is right as long as it is read at least that often, which every
 * caller here does many times a frame. */
uint64_t boardCycles64(void);

/* Caches are ON for every measurement, and the firmware says so in its banner: an
 * H7 running from flash with the I-cache off is a different machine, and a number
 * taken there does not describe the part anyone ships.
 */
void boardEnableCaches(void);

/* Stack high-water mark. boardStackPaint() fills the unused part of the stack with a
 * pattern; boardStackUsed() is how many bytes from the top have been written since.
 * The stack is the one allocation buffer geometry cannot see, so it is measured.
 */
void boardStackPaint(void);
size_t boardStackUsed(void);
size_t boardStackSize(void);

/* Clocks for the D2 SRAM banks, which are off out of reset; a section placed there
 * reads and writes nothing until this has run. */
void boardEnableSram(void);

/* System reset through the core's AIRCR, so a host can start a fresh launch without
 * re-flashing. */
void boardReset(void);

#ifdef __cplusplus
}
#endif

#endif
