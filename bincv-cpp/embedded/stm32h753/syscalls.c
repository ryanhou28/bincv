/* Newlib's link-time dependencies, stubbed.
 *
 * binCV needs none of these -- no kernel allocates and no kernel does I/O
 * (ARCHITECTURE 2). They are here because `snprintf`, which core/simd.hpp uses to
 * build its status line, drags in enough of newlib's stdio to reference them.
 *
 * `_sbrk` is real rather than a stub, because newlib's stdio may take a buffer from
 * the heap. The heap is the fixed region the linker script reserves; running out
 * returns failure rather than colliding with the stack, which is the failure mode
 * that would otherwise corrupt the tracker's staging buffers and look like a binCV
 * bug.
 */
#include <errno.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>

extern char _sheap, _eheap;

void* _sbrk(ptrdiff_t incr) {
    static char* brk = &_sheap;
    if (incr < 0) return (void*)-1;
    if (brk + incr > &_eheap) {
        errno = ENOMEM;
        return (void*)-1;
    }
    char* const prev = brk;
    brk += incr;
    return prev;
}

int _close(int fd) {
    (void)fd;
    return -1;
}
int _fstat(int fd, struct stat* st) {
    (void)fd;
    st->st_mode = S_IFCHR;
    return 0;
}
int _isatty(int fd) {
    (void)fd;
    return 1;
}
off_t _lseek(int fd, off_t off, int whence) {
    (void)fd;
    (void)off;
    (void)whence;
    return 0;
}
ssize_t _read(int fd, void* buf, size_t n) {
    (void)fd;
    (void)buf;
    (void)n;
    return 0;
}
int _getpid(void) { return 1; }
int _kill(int pid, int sig) {
    (void)pid;
    (void)sig;
    errno = EINVAL;
    return -1;
}

/* Anything newlib prints goes to the same UART the harness uses, so a stray
 * printf cannot vanish silently. */
extern void boardPutc(char c);
ssize_t _write(int fd, const void* buf, size_t n) {
    (void)fd;
    const char* p = (const char*)buf;
    for (size_t i = 0; i < n; ++i) {
        if (p[i] == '\n') boardPutc('\r');
        boardPutc(p[i]);
    }
    return (ssize_t)n;
}
