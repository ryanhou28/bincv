#pragma once

/// @file parallel.hpp
/// @brief The parallel-for customisation point. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// THE SPEEDUP WAS NEVER MISSING; THE WAY TO ASK FOR IT WAS
///
/// [X-65](../../../../docs/EXPERIMENTS.md) measured **2.60x on `track` at four threads**,
/// bit-exact, with peak RSS flat to 0.07% -- and it needed **no library change at
/// all**. `calcOpticalFlowPyrLK` already takes an ARRAY of points and the pyramids
/// are read-only, so any caller could split the array themselves; the benchmark did
/// exactly that in about thirty lines. What binCV did not offer was a way to ask.
///
/// ---------------------------------------------------------------------------
/// WHY THE POOL IS NOT IN HERE
///
/// `bincv_core` is allocation-free and builds `-fno-exceptions`, and `std::thread` is
/// usable under neither. **So the pool cannot live in core whatever the policy is** --
/// that constraint decides the shape, not a preference. This header holds a function
/// POINTER and a count, both trivially destructible, and nothing else.
///
/// The default is SERIAL, so a core-only or freestanding build is byte-for-byte what
/// it was. A hosted build installs a backend at start-up; an integrator with an
/// existing pool installs theirs instead and binCV never spawns a thread.
///
/// ---------------------------------------------------------------------------
/// THE SURFACE IS OPENCV'S ON PURPOSE
///
/// `setNumThreads` / `getNumThreads`, with `1` serialising. An integrator who already
/// knows `cv::setNumThreads` needs no new vocabulary, and the reference implementation's
/// model -- single-worker pools per pipeline stage, parallelism taken at the stage
/// level -- is one `setNumThreads(1)` call away.

#include <cstddef>
#include <type_traits>

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief Runs `body(i)` for `i` in `[0, n)`, possibly concurrently.
/// @note The callback is a raw function pointer plus a `void*`, not a
///       `std::function`: core allocates nothing, and a `std::function` may.
using ParallelForFn = void (*)(size_t n, void (*body)(size_t, void*), void* ctx);

namespace impl {

/// @brief The installed backend, or null for serial. **INTERNAL.**
/// @note A plain pointer with static storage duration: no constructor runs, so this
///       is safe before `main` and costs nothing in a freestanding image.
inline ParallelForFn& parallelBackend() {
    static ParallelForFn fn = nullptr;
    return fn;
}

inline int& threadCount() {
    static int n = 1;
    return n;
}

} // namespace impl

/// @brief Installs a parallel-for backend. `nullptr` restores serial execution.
/// @note **Not thread-safe, and deliberately not.** A backend is installed once at
///       start-up, like OpenCV's. Guarding every call to make installation racy-safe
///       would put an atomic load in a path that runs per keypoint.
inline void setParallelForBackend(ParallelForFn fn) { impl::parallelBackend() = fn; }

/// @brief How many threads binCV may use. `1` serialises. Mirrors `cv::setNumThreads`.
inline void setNumThreads(int n) { impl::threadCount() = n < 1 ? 1 : n; }

/// @brief The current thread count. `1` unless a backend is installed and asked for more.
inline int getNumThreads() {
    return impl::parallelBackend() == nullptr ? 1 : impl::threadCount();
}

/// @brief Runs `body(i, ctx)` for `i` in `[0, n)`. Serial unless a backend is installed.
/// @note **Serial is not a fallback, it is the default.** Every kernel that uses this
///       must be correct when it runs straight through, because on a core-only build
///       that is the only way it ever runs.
template <typename Body>
inline void parallelFor(size_t n, Body&& body) {
    const ParallelForFn fn = impl::parallelBackend();
    if (fn == nullptr || getNumThreads() <= 1 || n <= 1) {
        for (size_t i = 0; i < n; ++i) body(i);
        return;
    }
    // The lambda is passed by address; `parallelFor` does not return until the
    // backend has finished, so the reference cannot dangle.
    auto trampoline = [](size_t i, void* ctx) {
        (*static_cast<typename std::remove_reference<Body>::type*>(ctx))(i);
    };
    fn(n, trampoline, const_cast<void*>(static_cast<const void*>(&body)));
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
