// Interposes the C allocator. Link this into exactly one target per binary:
// it DEFINES malloc and friends, and the executable's definitions preempt
// libc's for every dynamically linked caller, OpenCV included.

#include "heap_probe.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__GLIBC__)
#include <malloc.h>

extern "C" void* __libc_malloc(size_t);
extern "C" void __libc_free(void*);
extern "C" void* __libc_calloc(size_t, size_t);
extern "C" void* __libc_realloc(void*, size_t);
extern "C" void* __libc_memalign(size_t, size_t);

namespace {
bool g_armed = false;
long long g_live = 0;
long long g_peak = 0;
std::size_t g_calls = 0;
std::size_t g_small = 0;

inline void tookBlock(void* p, std::size_t requested) {
    if (!g_armed || p == nullptr) return;
    g_live += static_cast<long long>(malloc_usable_size(p));
    ++g_calls;
    if (requested < 128) ++g_small;
    if (g_live > g_peak) g_peak = g_live;
}
inline void gaveBack(void* p) {
    if (!g_armed || p == nullptr) return;
    g_live -= static_cast<long long>(malloc_usable_size(p));
}
} // namespace

extern "C" void* malloc(size_t n) {
    void* p = __libc_malloc(n);
    tookBlock(p, n);
    return p;
}
extern "C" void free(void* p) {
    gaveBack(p);
    __libc_free(p);
}
extern "C" void* calloc(size_t a, size_t b) {
    void* p = __libc_calloc(a, b);
    tookBlock(p, a * b);
    return p;
}
extern "C" void* realloc(void* q, size_t n) {
    gaveBack(q);
    void* p = __libc_realloc(q, n);
    tookBlock(p, n);
    return p;
}
extern "C" void* memalign(size_t a, size_t n) {
    void* p = __libc_memalign(a, n);
    tookBlock(p, n);
    return p;
}
extern "C" void* aligned_alloc(size_t a, size_t n) {
    void* p = __libc_memalign(a, n);
    tookBlock(p, n);
    return p;
}
extern "C" int posix_memalign(void** out, size_t a, size_t n) {
    void* p = __libc_memalign(a, n);
    if (p == nullptr) return 12;  // ENOMEM
    tookBlock(p, n);
    *out = p;
    return 0;
}

namespace heapprobe {

void begin() {
    g_live = 0;
    g_peak = 0;
    g_calls = 0;
    g_small = 0;
    g_armed = true;
}

Reading end() {
    g_armed = false;
    Reading r;
    r.peakLive = g_peak > 0 ? static_cast<std::size_t>(g_peak) : 0;
    r.calls = g_calls;
    r.smallCalls = g_small;
    r.net = g_live;
    return r;
}

bool selfCheck() {
    bool ok = true;
    auto report = [&ok](const char* what, bool pass, const char* detail) {
        std::printf("   %-46s %s%s%s\n", what, pass ? "ok" : "FAIL",
                    detail[0] ? " -- " : "", detail);
        if (!pass) ok = false;
    };
    char buf[96];

    // 1. A plain malloc of a known size is seen, and freeing gives it back.
    {
        begin();
        void* p = std::malloc(100000);
        std::memset(p, 1, 100000);
        const Reading mid = end();
        begin();
        // Re-arm around the free so `net` reflects only the release.
        std::free(p);
        const Reading after = end();
        std::snprintf(buf, sizeof buf, "peak %zu B, released %lld B", mid.peakLive, -after.net);
        report("sees malloc, and free returns the bytes", mid.peakLive >= 100000 && after.net <= -100000, buf);
    }

    // 2. `new` routes through malloc, so the probe must see it too.
    {
        begin();
        char* p = new char[50000];
        std::memset(p, 2, 50000);
        const Reading r = end();
        delete[] p;
        std::snprintf(buf, sizeof buf, "peak %zu B", r.peakLive);
        report("sees operator new", r.peakLive >= 50000, buf);
    }

    // 3. PEAK LIVE IS NOT THE SUM. Ten blocks taken and released one at a time
    //    must read as one block, not ten -- this is the error that once put
    //    OpenCV at 323 088 B when its peak live was a fraction of that.
    {
        begin();
        for (int i = 0; i < 10; ++i) {
            void* p = std::malloc(10000);
            std::memset(p, 3, 10000);
            std::free(p);
        }
        const Reading r = end();
        std::snprintf(buf, sizeof buf, "peak %zu B over %zu calls", r.peakLive, r.calls);
        report("peak live is not the sum of allocations", r.peakLive < 20000 && r.calls == 10, buf);
    }

    // 4. Balanced work must leave nothing behind, which is what says the
    //    subtract side is as complete as the add side.
    {
        begin();
        void* p = std::malloc(7777);
        std::memset(p, 4, 7777);
        std::free(p);
        const Reading r = end();
        std::snprintf(buf, sizeof buf, "net %lld B", r.net);
        report("balanced work nets to zero", r.net == 0, buf);
    }
    return ok;
}

} // namespace heapprobe

#else  // !__GLIBC__

namespace heapprobe {
void begin() {}
Reading end() { return Reading{}; }
bool selfCheck() {
    std::printf("   heap probe needs glibc's malloc_usable_size; this build has no"
                " instrument\n");
    return false;
}
} // namespace heapprobe

#endif
