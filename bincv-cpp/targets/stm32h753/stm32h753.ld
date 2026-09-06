/* STM32H753ZI. Memory map from RM0433.
 *
 * Data lives in AXI SRAM rather than DTCM on purpose: DTCM is 128 KB and a
 * 752x480 BinMat pair plus a pyramid does not fit there, so measuring out of DTCM
 * would measure a configuration a real frame cannot use. The stack is the
 * exception -- it is small, hot, and the tracker's staging buffers live on it, so
 * it gets DTCM.
 */
ENTRY(Reset_Handler)

MEMORY
{
  FLASH   (rx)  : ORIGIN = 0x08000000, LENGTH = 2048K
  DTCM    (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
  RAM_D1  (rwx) : ORIGIN = 0x24000000, LENGTH = 512K
  RAM_D2  (rwx) : ORIGIN = 0x30000000, LENGTH = 288K
  RAM_D3  (rwx) : ORIGIN = 0x38000000, LENGTH = 64K
}

_stack_top   = ORIGIN(DTCM) + LENGTH(DTCM);
_stack_size  = 16K;   /* BINCV_STAGING_BUDGET_BYTES is checked against this */
_stack_limit = _stack_top - _stack_size;

SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH

  .text :
  {
    *(.text .text.*)
    *(.rodata .rodata.*)
    *(.glue_7) *(.glue_7t) *(.eh_frame)
    KEEP(*(.init)) KEEP(*(.fini))
    . = ALIGN(4);
    _etext = .;
  } > FLASH

  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
  .ARM       : { __exidx_start = .; *(.ARM.exidx*) __exidx_end = .; } > FLASH

  /* Initialised data is stored in flash and copied to AXI SRAM at reset. */
  _sidata = LOADADDR(.data);
  .data : ALIGN(4)
  {
    _sdata = .;
    *(.data .data.*)
    . = ALIGN(4);
    _edata = .;
  } > RAM_D1 AT> FLASH

  .bss : ALIGN(4)
  {
    _sbss = .;
    __bss_start__ = _sbss;
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
    __bss_end__ = _ebss;
  } > RAM_D1

  /* Frame buffers. Named so the map file shows the footprint claim directly. */
  .frames (NOLOAD) : ALIGN(32)
  {
    _sframes = .;
    *(.frames .frames.*)
    . = ALIGN(4);
    _eframes = .;
  } > RAM_D1

  .heap (NOLOAD) : ALIGN(8)
  {
    _sheap = .;
    . = . + 4K;      /* newlib needs a little for its own bookkeeping; binCV
                        kernels allocate nothing (ARCHITECTURE 2). */
    _eheap = .;
  } > RAM_D1

  /DISCARD/ : { *(.ARM.attributes) }
}

ASSERT(_stack_size >= 16K, "stack below the staging budget the tracker asserts against")
