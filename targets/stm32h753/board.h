/* Minimal board support for the NUCLEO-H753ZI: serial out, cycle counter, caches.
 * Register definitions are inline from RM0433 rather than from CMSIS -- see
 * startup.c for why this target carries no vendor tree.
 */
#ifndef BINCV_EMBEDDED_STM32H753_BOARD_H
#define BINCV_EMBEDDED_STM32H753_BOARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SYSCLK out of reset: HSI, no PLL. Everything timed is reported against this, and
 * boardClockHz() is what the firmware prints -- a run that cannot state its clock is
 * not a measurement (see README).
 */
#define BINCV_M7_CLOCK_HZ 64000000u

uint32_t boardClockHz(void);

/* USART3 on PD8/PD9, which is what the ST-LINK's virtual COM port is wired to on
 * this board. 115200 8N1.
 */
void boardSerialInit(void);
void boardPutc(char c);
void boardPuts(const char* s);

/* Unsigned decimal; the harness prints no floats, so no soft-float formatting and no
 * newlib printf lands in the image. Both cost more flash than everything else here.
 */
void boardPutU32(uint32_t v);
void boardPutU64(uint64_t v);

/* Cycle-accurate timing from the DWT. Returns false when the counter does not
 * advance, which is the one failure mode that would otherwise report every arm as
 * infinitely fast.
 */
int boardCycleCounterInit(void);

static inline uint32_t boardCycles(void) { return *(volatile uint32_t*)0xE0001004u; }

/* Caches are ON for every measurement, and the firmware says so in its banner: an
 * H7 running from flash with the I-cache off is a different machine, and a number
 * taken there does not describe the part anyone ships.
 */
void boardEnableCaches(void);

#ifdef __cplusplus
}
#endif

#endif
