#include "board.h"

/* RM0433 register addresses. */
#define RCC_BASE 0x58024400u
#define RCC_AHB4ENR (*(volatile uint32_t*)(RCC_BASE + 0x0E0u))
#define RCC_APB1LENR (*(volatile uint32_t*)(RCC_BASE + 0x0E8u))

#define GPIOD_BASE 0x58020C00u
#define GPIOD_MODER (*(volatile uint32_t*)(GPIOD_BASE + 0x00u))
#define GPIOD_OSPEEDR (*(volatile uint32_t*)(GPIOD_BASE + 0x08u))
#define GPIOD_AFRH (*(volatile uint32_t*)(GPIOD_BASE + 0x24u))

#define USART3_BASE 0x40004800u
#define USART3_CR1 (*(volatile uint32_t*)(USART3_BASE + 0x00u))
#define USART3_BRR (*(volatile uint32_t*)(USART3_BASE + 0x0Cu))
#define USART3_ISR (*(volatile uint32_t*)(USART3_BASE + 0x1Cu))
#define USART3_TDR (*(volatile uint32_t*)(USART3_BASE + 0x28u))

#define USART_CR1_UE (1u << 0)
#define USART_CR1_TE (1u << 3)
#define USART_ISR_TXE (1u << 7) /* TXE / TXFNF */
#define USART_ISR_TC (1u << 6)

#define DEMCR (*(volatile uint32_t*)0xE000EDFCu)
#define DEMCR_TRCENA (1u << 24)
#define DWT_CTRL (*(volatile uint32_t*)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t*)0xE0001004u)
#define DWT_LAR (*(volatile uint32_t*)0xE0001FB0u)
#define DWT_CTRL_CYCCNTENA (1u << 0)

#define SCB_CCR (*(volatile uint32_t*)0xE000ED14u)
#define SCB_CCSIDR (*(volatile uint32_t*)0xE000ED80u)
#define SCB_CSSELR (*(volatile uint32_t*)0xE000ED84u)
#define SCB_ICIALLU (*(volatile uint32_t*)0xE000EF50u)
#define SCB_DCISW (*(volatile uint32_t*)0xE000EF60u)
#define SCB_CCR_IC (1u << 17)
#define SCB_CCR_DC (1u << 16)

#define DSB() __asm volatile("dsb" ::: "memory")
#define ISB() __asm volatile("isb" ::: "memory")

uint32_t boardClockHz(void) { return BINCV_M7_CLOCK_HZ; }

void boardSerialInit(void) {
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
    /* OVER8 = 0, so BRR is just the integer divisor. At 64 MHz this is 556 for a
     * 115200 target -- 115107 actual, 0.08% error, far inside the 2% a UART
     * tolerates. */
    USART3_BRR = (BINCV_M7_CLOCK_HZ + 115200u / 2u) / 115200u;
    USART3_CR1 = USART_CR1_TE | USART_CR1_UE;
}

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
