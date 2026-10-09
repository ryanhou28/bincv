/* memcpy, memmove and memset a word at a time, in place of newlib-nano's.
 *
 * newlib-nano is built for size, and its three are byte loops: one byte per
 * iteration, two or three cycles each on the M7. binCV's kernels lean on them --
 * the sensor stream copies every arriving row into its ring, and GCC turns a
 * kernel's own zero-filling loop into a memset call (the tracker image has
 * thirty-two of them) -- so with nano's versions a measurement prices the C
 * library, not the kernel. These move a 32-bit word per access instead, four per
 * iteration. Any word-wide implementation would do; an integrator linking full
 * newlib, or their vendor's, already has one.
 *
 * Unaligned source words are plain loads: the M7 performs an unaligned LDR in
 * hardware (CCR.UNALIGN_TRP is left clear), so only the destination is aligned
 * first. The loops are kept from being recognised as memcpy/memset and compiled
 * back into calls to themselves.
 */
#include <stddef.h>
#include <stdint.h>

#define BINCV_M7_NO_PATTERNS __attribute__((optimize("no-tree-loop-distribute-patterns")))

typedef uint32_t __attribute__((aligned(1), may_alias)) unaligned_u32;

BINCV_M7_NO_PATTERNS void* memcpy(void* restrict dst, const void* restrict src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    while (n > 0 && ((uintptr_t)d & 3u) != 0) {
        *d++ = *s++;
        --n;
    }
    uint32_t* dw = (uint32_t*)(void*)d;
    while (n >= 16) {
        const unaligned_u32* sw = (const unaligned_u32*)s;
        const uint32_t a = sw[0], b = sw[1], c = sw[2], e = sw[3];
        dw[0] = a;
        dw[1] = b;
        dw[2] = c;
        dw[3] = e;
        dw += 4;
        s += 16;
        n -= 16;
    }
    while (n >= 4) {
        *dw++ = *(const unaligned_u32*)s;
        s += 4;
        n -= 4;
    }
    d = (uint8_t*)dw;
    while (n > 0) {
        *d++ = *s++;
        --n;
    }
    return dst;
}

BINCV_M7_NO_PATTERNS void* memmove(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    if (d == s || n == 0) return dst;
    /* Forward is safe unless the destination starts inside the source. */
    if (d < s || d >= s + n) return memcpy(dst, src, n);
    d += n;
    s += n;
    while (n > 0 && ((uintptr_t)d & 3u) != 0) {
        *--d = *--s;
        --n;
    }
    uint32_t* dw = (uint32_t*)(void*)d;
    while (n >= 4) {
        s -= 4;
        *--dw = *(const unaligned_u32*)s;
        n -= 4;
    }
    d = (uint8_t*)dw;
    while (n > 0) {
        *--d = *--s;
        --n;
    }
    return dst;
}

BINCV_M7_NO_PATTERNS void* memset(void* dst, int value, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t byte = (uint8_t)value;
    while (n > 0 && ((uintptr_t)d & 3u) != 0) {
        *d++ = byte;
        --n;
    }
    const uint32_t word = 0x01010101u * byte;
    uint32_t* dw = (uint32_t*)(void*)d;
    while (n >= 16) {
        dw[0] = word;
        dw[1] = word;
        dw[2] = word;
        dw[3] = word;
        dw += 4;
        n -= 16;
    }
    while (n >= 4) {
        *dw++ = word;
        n -= 4;
    }
    d = (uint8_t*)dw;
    while (n > 0) {
        *d++ = byte;
        --n;
    }
    return dst;
}
