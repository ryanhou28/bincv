/* Reset vector and C runtime bring-up for STM32H753ZI.
 *
 * Deliberately minimal: no CMSIS, no HAL, no vendor SDK. The harness needs a stack,
 * initialised data, an FPU and a UART, and every one of those is a handful of
 * register writes. Pulling in a vendor tree to get them would put thousands of lines
 * of code that binCV does not use inside the footprint number this target exists to
 * report.
 */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _stack_top;

extern int main(void);

void Reset_Handler(void);
void Default_Handler(void);

/* A fault here is a bug in the harness, and it must not look like a hang: the
 * firmware's whole output is a serial log, so a silent spin is indistinguishable
 * from a slow benchmark. HardFault parks in a tight loop with a recognisable
 * pattern in r0 for a debugger, and the log's absence is the signal.
 */
void HardFault_Handler(void) {
    __asm volatile("mov r0, #0xDEAD\n bkpt #0\n b .");
}

/* Entry 0 of the table is the initial stack pointer and every other entry is a
 * handler address, so the table holds two different kinds of thing. A union says
 * that directly; casting between object and function pointers to force them into one
 * array type is undefined in ISO C, and -Wpedantic is right to reject it.
 */
typedef union {
    void (*handler)(void);
    void* stack;
} VectorEntry;

__attribute__((section(".isr_vector"), used)) const VectorEntry g_vectors[] = {
    {.stack = &_stack_top},
    {.handler = Reset_Handler},
    {.handler = Default_Handler},  /* NMI          */
    {.handler = HardFault_Handler},
    {.handler = Default_Handler},  /* MemManage    */
    {.handler = Default_Handler},  /* BusFault     */
    {.handler = Default_Handler},  /* UsageFault   */
    {.stack = 0}, {.stack = 0}, {.stack = 0}, {.stack = 0},
    {.handler = Default_Handler},  /* SVCall       */
    {.handler = Default_Handler},  /* DebugMonitor */
    {.stack = 0},
    {.handler = Default_Handler},  /* PendSV       */
    {.handler = Default_Handler},  /* SysTick      */
};

void Default_Handler(void) {
    for (;;) {
    }
}

/* -mfloat-abi=hard emits FPU instructions from the first C function that touches a
 * double, and the FPU is disabled out of reset -- so this must happen before any
 * such code runs, which is why it is here and not in main.
 */
static void enableFpu(void) {
    volatile uint32_t* const cpacr = (volatile uint32_t*)0xE000ED88u;
    *cpacr |= (0xFu << 20);  /* CP10/CP11 full access */
    __asm volatile("dsb" ::: "memory");
    __asm volatile("isb" ::: "memory");
}

void Reset_Handler(void) {
    enableFpu();

    const uint32_t* src = &_sidata;
    for (uint32_t* dst = &_sdata; dst < &_edata;) *dst++ = *src++;
    for (uint32_t* p = &_sbss; p < &_ebss;) *p++ = 0;

    (void)main();
    for (;;) {
    }
}
