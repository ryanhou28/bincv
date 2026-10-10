#include "board.h"

/* RM0433 register addresses. Offsets and bit positions as in ST's stm32h753xx.h. */
#define REG(addr) (*(volatile uint32_t*)(addr))

#define RCC_BASE 0x58024400u
#define RCC_CR REG(RCC_BASE + 0x000u)
#define RCC_CFGR REG(RCC_BASE + 0x010u)
#define RCC_D1CFGR REG(RCC_BASE + 0x018u)
#define RCC_D2CFGR REG(RCC_BASE + 0x01Cu)
#define RCC_D3CFGR REG(RCC_BASE + 0x020u)
#define RCC_PLLCKSELR REG(RCC_BASE + 0x028u)
#define RCC_PLLCFGR REG(RCC_BASE + 0x02Cu)
#define RCC_PLL1DIVR REG(RCC_BASE + 0x030u)
#define RCC_D2CCIP2R REG(RCC_BASE + 0x054u)
#define RCC_BDCR REG(RCC_BASE + 0x070u)
#define RCC_AHB2ENR REG(RCC_BASE + 0x0DCu)
#define RCC_AHB4ENR REG(RCC_BASE + 0x0E0u)
#define RCC_APB1LENR REG(RCC_BASE + 0x0E8u)
#define RCC_APB4ENR REG(RCC_BASE + 0x0F4u)

#define PWR_BASE 0x58024800u
#define PWR_CR1 REG(PWR_BASE + 0x00u)
#define PWR_CSR1 REG(PWR_BASE + 0x04u)
#define PWR_CR3 REG(PWR_BASE + 0x0Cu)
#define PWR_D3CR REG(PWR_BASE + 0x18u)

#define SYSCFG_PWRCR REG(0x58000400u + 0x2Cu)
#define FLASH_ACR REG(0x52002000u + 0x00u)

#define LPTIM1_BASE 0x40002400u
#define LPTIM1_ISR REG(LPTIM1_BASE + 0x00u)
#define LPTIM1_CFGR REG(LPTIM1_BASE + 0x0Cu)
#define LPTIM1_CR REG(LPTIM1_BASE + 0x10u)
#define LPTIM1_ARR REG(LPTIM1_BASE + 0x18u)
#define LPTIM1_CNT REG(LPTIM1_BASE + 0x1Cu)

#define GPIOD_BASE 0x58020C00u
#define GPIOD_MODER REG(GPIOD_BASE + 0x00u)
#define GPIOD_OSPEEDR REG(GPIOD_BASE + 0x08u)
#define GPIOD_AFRH REG(GPIOD_BASE + 0x24u)

#define USART3_BASE 0x40004800u
#define USART3_CR1 REG(USART3_BASE + 0x00u)
#define USART3_BRR REG(USART3_BASE + 0x0Cu)
#define USART3_ISR REG(USART3_BASE + 0x1Cu)
#define USART3_ICR REG(USART3_BASE + 0x20u)
#define USART3_RDR REG(USART3_BASE + 0x24u)
#define USART3_TDR REG(USART3_BASE + 0x28u)

#define USART_CR1_UE (1u << 0)
#define USART_CR1_RE (1u << 2)
#define USART_CR1_TE (1u << 3)
#define USART_CR1_FIFOEN (1u << 29)
#define USART_ISR_ORE (1u << 3)
#define USART_ISR_RXNE (1u << 5) /* RXNE / RXFNE */
#define USART_ISR_TC (1u << 6)
#define USART_ISR_TXE (1u << 7) /* TXE / TXFNF */
#define USART_ICR_ORECF (1u << 3)

#define DEMCR REG(0xE000EDFCu)
#define DEMCR_TRCENA (1u << 24)
#define DWT_CTRL REG(0xE0001000u)
#define DWT_CYCCNT REG(0xE0001004u)
#define DWT_LAR REG(0xE0001FB0u)
#define DWT_CTRL_CYCCNTENA (1u << 0)

#define SCB_AIRCR REG(0xE000ED0Cu)
#define SCB_CCR REG(0xE000ED14u)
#define SCB_CCSIDR REG(0xE000ED80u)
#define SCB_CSSELR REG(0xE000ED84u)
#define SCB_ICIALLU REG(0xE000EF50u)
#define SCB_DCISW REG(0xE000EF60u)
#define SCB_CCR_IC (1u << 17)
#define SCB_CCR_DC (1u << 16)

#define DSB() __asm volatile("dsb" ::: "memory")
#define ISB() __asm volatile("isb" ::: "memory")

/* Loop bound for every wait on a hardware ready flag during clock bring-up. The
 * cycle counter may not be running yet, and the core is on HSI, so this is a count
 * of polls: about a second at 64 MHz, far beyond any flag the reference manual
 * gives a settling time for. A flag that never sets returns an error code rather
 * than a hang, because a hang produces no report at all. */
#define CLOCK_WAIT_POLLS 16000000u

static int waitSet(volatile uint32_t* reg, uint32_t mask) {
    for (uint32_t i = 0; i < CLOCK_WAIT_POLLS; ++i) {
        if ((*reg & mask) == mask) return 1;
    }
    return 0;
}

int boardClockInit(void) {
#ifdef BINCV_M7_RESET_CLOCK
    return 0;
#else
    /* The settings are the ones Zephyr's board file uses for this exact board
     * (boards/st/nucleo_h753zi): HSE in bypass from the ST-LINK's 8 MHz clock,
     * DIVM1 = 2, DIVN1 = 240, DIVP1 = 2 -- a 960 MHz VCO, which the H74x/H75x
     * wide range allows at voltage scale 0 -- and AXI/AHB at 240 MHz with every
     * APB at 120 MHz. */

    /* 1. Supply from the LDO, which is what VOS0 requires on this line. Clearing
     * SCUEN with the write locks the configuration until reset. */
    PWR_CR3 = (PWR_CR3 & ~((1u << 2) | (1u << 1) | (1u << 0))) | (1u << 1);
    if (!waitSet(&PWR_CSR1, 1u << 13)) return 1; /* ACTVOSRDY */

    /* 2. VOS1, then VOS0: on H74x/H75x scale 0 is scale 1 with SYSCFG's
     * overdrive bit set, and it can only be entered from scale 1. */
    PWR_D3CR = (PWR_D3CR & ~(3u << 14)) | (3u << 14);
    if (!waitSet(&PWR_D3CR, 1u << 13)) return 2;
    RCC_APB4ENR |= (1u << 1); /* SYSCFGEN */
    (void)RCC_APB4ENR;
    SYSCFG_PWRCR |= 1u; /* ODEN */
    if (!waitSet(&PWR_D3CR, 1u << 13)) return 2;

    /* 3. Flash wait states before the clock rises: 4 at a 240 MHz AXI clock,
     * programming delay 2. More than the minimum is safe; fewer is not, and
     * instruction fetch is behind the I-cache either way. */
    FLASH_ACR = (FLASH_ACR & ~0x3Fu) | (2u << 4) | 4u;
    if ((FLASH_ACR & 0x3Fu) != ((2u << 4) | 4u)) return 6;

    /* 4. HSE in bypass: the ST-LINK drives the pin with a clock, not a crystal. */
    RCC_CR |= (1u << 18); /* HSEBYP */
    RCC_CR |= (1u << 16); /* HSEON */
    if (!waitSet(&RCC_CR, 1u << 17)) return 3;

    /* 5. PLL1 from HSE: 8 MHz / 2 = 4 MHz reference (input range 2-4 MHz),
     * x 240 = 960 MHz VCO (wide range), / 2 = 480 MHz on P. Q and R stay off. */
    RCC_CR &= ~(1u << 24);
    while (RCC_CR & (1u << 25)) {
    }
    RCC_PLLCKSELR = (RCC_PLLCKSELR & ~((0x3Fu << 4) | 3u)) | (2u << 4) | 2u;
    RCC_PLLCFGR = (RCC_PLLCFGR & ~((7u << 16) | (3u << 2) | (1u << 1) | 1u)) | (1u << 16) |
                  (1u << 2);
    RCC_PLL1DIVR = ((240u - 1u) << 0) | ((2u - 1u) << 9) | ((2u - 1u) << 16) |
                   ((2u - 1u) << 24);
    RCC_CR |= (1u << 24);
    if (!waitSet(&RCC_CR, 1u << 25)) return 4;

    /* 6. Bus prescalers before the switch: core / 1, AXI and AHB / 2, every APB
     * / 2 again. */
    RCC_D1CFGR = (RCC_D1CFGR & ~((0xFu << 8) | (7u << 4) | 0xFu)) | (0u << 8) | (4u << 4) | 8u;
    RCC_D2CFGR = (RCC_D2CFGR & ~((7u << 8) | (7u << 4))) | (4u << 8) | (4u << 4);
    RCC_D3CFGR = (RCC_D3CFGR & ~(7u << 4)) | (4u << 4);

    /* 7. Switch the system clock to PLL1 and wait for the switch to be reported. */
    RCC_CFGR = (RCC_CFGR & ~7u) | 3u;
    for (uint32_t i = 0; i < CLOCK_WAIT_POLLS; ++i) {
        if (((RCC_CFGR >> 3) & 7u) == 3u) return 0;
    }
    return 5;
#endif
}

uint32_t boardClockHz(void) { return BINCV_M7_CLOCK_HZ; }

static uint32_t lptimCount(void) {
    /* LPTIM counts on the LSE, asynchronously to the bus: a read can catch the
     * counter mid-update, so take two equal reads (RM0433, LPTIM_CNT). */
    uint32_t a, b;
    do {
        a = LPTIM1_CNT;
        b = LPTIM1_CNT;
    } while (a != b);
    return a & 0xFFFFu;
}

uint32_t boardMeasureClockHz(void) {
    /* LSE on: the backup domain is write-protected out of reset. */
    PWR_CR1 |= (1u << 8); /* DBP */
    while (!(PWR_CR1 & (1u << 8))) {
    }
    RCC_BDCR = (RCC_BDCR & ~(3u << 3)) | (2u << 3); /* medium-high drive */
    RCC_BDCR |= 1u;                                  /* LSEON */
    const uint32_t t0 = boardCycles();
    while (!(RCC_BDCR & 2u)) { /* LSERDY */
        if ((uint32_t)(boardCycles() - t0) > 3u * BINCV_M7_CLOCK_HZ) return 0;
    }

    /* LPTIM1 counting the LSE, free-running over its full 16 bits. */
    RCC_APB1LENR |= (1u << 9); /* LPTIM1EN */
    (void)RCC_APB1LENR;
    RCC_D2CCIP2R = (RCC_D2CCIP2R & ~(7u << 28)) | (3u << 28); /* LPTIM1SEL = LSE */
    LPTIM1_CR = 0;
    LPTIM1_CFGR = 0;
    LPTIM1_CR = 1u; /* ENABLE */
    LPTIM1_ARR = 0xFFFFu;
    while (!(LPTIM1_ISR & (1u << 4))) { /* ARROK */
    }
    LPTIM1_CR = 1u | (1u << 2); /* continuous */

    /* Start on an edge, stop on the edge one second later, read the cycle counter
     * the moment each is seen. The poll latency is the same at both ends, and one
     * second is 32768 ticks -- half the 16-bit counter's span. */
    const uint32_t ticks = 32768u;
    const uint32_t c0Start = lptimCount();
    uint32_t start;
    while ((start = lptimCount()) == c0Start) {
    }
    const uint32_t cyc0 = boardCycles();
    const uint32_t end = (start + ticks) & 0xFFFFu;
    while (lptimCount() != end) {
    }
    const uint32_t cyc1 = boardCycles();
    return (uint32_t)(cyc1 - cyc0);
}

void boardSerialInitAt(uint32_t pclk1Hz, uint32_t baud) {
    RCC_AHB4ENR |= (1u << 3);   /* GPIODEN */
    RCC_APB1LENR |= (1u << 18); /* USART3EN */
    (void)RCC_APB1LENR;

    /* PD8 = USART3_TX, PD9 = USART3_RX, both AF7. */
    GPIOD_MODER &= ~((3u << (8 * 2)) | (3u << (9 * 2)));
    GPIOD_MODER |= (2u << (8 * 2)) | (2u << (9 * 2)); /* alternate function */
    GPIOD_OSPEEDR |= (3u << (8 * 2)) | (3u << (9 * 2));
    GPIOD_AFRH &= ~((0xFu << ((8 - 8) * 4)) | (0xFu << ((9 - 8) * 4)));
    GPIOD_AFRH |= (7u << ((8 - 8) * 4)) | (7u << ((9 - 8) * 4));

    USART3_CR1 = 0;
    /* OVER8 = 0, so BRR is the integer divisor of the kernel clock, which is PCLK1
     * by default. At 120 MHz: 921600 baud is 130 (0.16% fast), 2 Mbaud exactly 60. */
    USART3_BRR = (pclk1Hz + baud / 2u) / baud;
    USART3_CR1 = USART_CR1_FIFOEN | USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

void boardSerialInit(uint32_t baud) { boardSerialInitAt(BINCV_M7_PCLK1_HZ, baud); }

void boardPutc(char c) {
    while (!(USART3_ISR & USART_ISR_TXE)) {
    }
    USART3_TDR = (uint32_t)(unsigned char)c;
    if (c == '\n') {
        while (!(USART3_ISR & USART_ISR_TC)) {
        }
    }
}

void boardPuts(const char* s) {
    for (; *s; ++s) {
        if (*s == '\n') boardPutc('\r');
        boardPutc(*s);
    }
}

void boardWrite(const void* data, size_t n) {
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < n; ++i) {
        while (!(USART3_ISR & USART_ISR_TXE)) {
        }
        USART3_TDR = p[i];
    }
    while (!(USART3_ISR & USART_ISR_TC)) {
    }
}

int boardGetc(void) {
    for (;;) {
        const uint32_t isr = USART3_ISR;
        if (isr & USART_ISR_ORE) {
            USART3_ICR = USART_ICR_ORECF;
            return -1;
        }
        if (isr & USART_ISR_RXNE) return (int)(USART3_RDR & 0xFFu);
    }
}

int boardRead(void* dst, size_t n) {
    unsigned char* p = (unsigned char*)dst;
    for (size_t i = 0; i < n; ++i) {
        const int c = boardGetc();
        if (c < 0) return -1;
        p[i] = (unsigned char)c;
    }
    return 0;
}

int boardReadLine(char* dst, size_t cap) {
    size_t n = 0;
    for (;;) {
        const int c = boardGetc();
        if (c < 0) return -1;
        if (c == '\n') break;
        if (c == '\r') continue;
        if (n + 1 < cap) dst[n++] = (char)c;
    }
    dst[n] = '\0';
    return (int)n;
}

void boardPutU64(uint64_t v) {
    char buf[21];
    size_t n = 0;
    if (v == 0) {
        boardPutc('0');
        return;
    }
    while (v) {
        buf[n++] = (char)('0' + (v % 10u));
        v /= 10u;
    }
    while (n) boardPutc(buf[--n]);
}

void boardPutU32(uint32_t v) { boardPutU64(v); }

void boardPutHex32(uint32_t v) {
    static const char kHex[] = "0123456789abcdef";
    for (int s = 28; s >= 0; s -= 4) boardPutc(kHex[(v >> s) & 0xFu]);
}

int boardCycleCounterInit(void) {
    DEMCR |= DEMCR_TRCENA;
    /* Cortex-M7 gates DWT behind a lock register that a debugger would normally
     * open. Without this the enable below is silently ignored and every interval
     * reads zero. */
    DWT_LAR = 0xC5ACCE55u;
    DWT_CYCCNT = 0;
    DWT_CTRL |= DWT_CTRL_CYCCNTENA;
    DSB();

    const uint32_t a = DWT_CYCCNT;
    for (volatile int i = 0; i < 64; ++i) {
    }
    return DWT_CYCCNT != a;
}

void boardEnableCaches(void) {
    /* I-cache: invalidate, then enable. */
    DSB();
    ISB();
    SCB_ICIALLU = 0;
    DSB();
    ISB();
    SCB_CCR |= SCB_CCR_IC;
    DSB();
    ISB();

    /* D-cache: every line must be invalidated before enabling, or the first read of
     * a line the cache never filled returns whatever the tag RAM powered up with. */
    SCB_CSSELR = 0;
    DSB();
    const uint32_t ccsidr = SCB_CCSIDR;
    uint32_t sets = (ccsidr >> 13) & 0x7FFFu;
    do {
        uint32_t ways = (ccsidr >> 3) & 0x3FFu;
        do {
            SCB_DCISW = ((sets & 0x1FFu) << 5) | ((ways & 0x3u) << 30);
        } while (ways-- != 0);
    } while (sets-- != 0);
    DSB();
    SCB_CCR |= SCB_CCR_DC;
    DSB();
    ISB();
}

extern uint32_t _stack_limit, _stack_top;
#define STACK_PAINT 0xA5A5A5A5u

void boardStackPaint(void) {
    /* Everything below the current frame, less a margin for this function's own
     * frame, is unused and can be overwritten. */
    uint32_t* const sp = (uint32_t*)__builtin_frame_address(0);
    for (uint32_t* p = &_stack_limit; p < sp - 32; ++p) *p = STACK_PAINT;
}

size_t boardStackUsed(void) {
    const uint32_t* p = &_stack_limit;
    while (p < &_stack_top && *p == STACK_PAINT) ++p;
    return (size_t)((const char*)&_stack_top - (const char*)p);
}

size_t boardStackSize(void) {
    return (size_t)((const char*)&_stack_top - (const char*)&_stack_limit);
}

uint64_t boardCycles64(void) {
    static uint32_t last = 0;
    static uint64_t high = 0;
    const uint32_t now = boardCycles();
    if (now < last) high += (uint64_t)1 << 32;
    last = now;
    return high | now;
}

void boardEnableSram(void) {
    RCC_AHB2ENR |= (1u << 29) | (1u << 30) | (1u << 31); /* SRAM1EN, SRAM2EN, SRAM3EN */
    (void)RCC_AHB2ENR;
    DSB();
}

void boardReset(void) {
    DSB();
    SCB_AIRCR = (0x5FAu << 16) | (1u << 2); /* VECTKEY | SYSRESETREQ */
    DSB();
    for (;;) {
    }
}
