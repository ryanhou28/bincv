#pragma once

/// @file pool.hpp
/// @brief A thread pool backend for `bincv::parallelFor`. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// THIS HEADER IS NOT PART OF `bincv_core`, AND THAT IS STRUCTURAL
///
/// `bincv_core` is allocation-free and builds `-fno-exceptions`. `std::thread` is
/// usable under neither: it allocates, and its constructor throws
/// `std::system_error` when a thread cannot be created. **So a pool cannot live in
/// core whatever the threading policy is** -- the constraint decides the shape.
///
/// Nothing in `ops/` includes this file. A core-only or freestanding build is
/// byte-for-byte what it was; a hosted caller opts in with one include and one call.
///
/// ---------------------------------------------------------------------------
/// THE INTEGRATOR MAY NOT WANT THIS AT ALL, AND THAT IS EXPECTED
///
/// The reference implementation runs single-worker pools per pipeline stage and takes
/// its parallelism at the stage level. A VIO system that already owns a thread pool
/// should install ITS OWN backend through `setParallelForBackend` and never let binCV
/// spawn anything -- oversubscription is worse than serial. This pool is for callers
/// who have no pool of their own.

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

#include "../core/parallel.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief A minimal fixed-size pool that serves `bincv::parallelFor`.
/// @note Deliberately minimal: work is claimed by index from one counter, which is
///       the right shape when the items are keypoints -- a few hundred of them, each
///       a few microseconds. A work-stealing deque would be more code for no
///       measurable difference at that grain.
class ThreadPool {
public:
    /// @param threads Workers to start. `<= 1` leaves binCV serial.
    explicit ThreadPool(int threads) {
        if (threads <= 1) return;
        workers_.reserve(static_cast<size_t>(threads));
        for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { run(); });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
        if (instance() == this) {
            setParallelForBackend(nullptr);
            instance() = nullptr;
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// @brief Makes this pool binCV's backend and sets the thread count to match.
    /// @note Install once, at start-up, before any tracking call -- the backend
    ///       pointer is not guarded, because guarding it would put an atomic load in
    ///       a path that runs per keypoint.
    void install() {
        instance() = this;
        setNumThreads(static_cast<int>(workers_.size()) + 1);
        setParallelForBackend(&dispatch);
    }

private:
    static ThreadPool*& instance() {
        static ThreadPool* p = nullptr;
        return p;
    }

    static void dispatch(size_t n, void (*body)(size_t, void*), void* ctx) {
        ThreadPool* self = instance();
        if (self == nullptr || self->workers_.empty()) {
            for (size_t i = 0; i < n; ++i) body(i, ctx);
            return;
        }
        self->runJob(n, body, ctx);
    }

    void runJob(size_t n, void (*body)(size_t, void*), void* ctx) {
        {
            std::unique_lock<std::mutex> lk(m_);
            body_ = body;
            ctx_ = ctx;
            total_ = n;
            next_ = 0;
            done_ = 0;
            ++generation_;
        }
        cv_.notify_all();
        // THE CALLING THREAD WORKS TOO. With N workers plus the caller that is N+1
        // hands, and it means a pool of size 1 still does something useful.
        claimAndRun();
        std::unique_lock<std::mutex> lk(m_);
        doneCv_.wait(lk, [this] { return done_ == total_; });
        body_ = nullptr;
    }

    void claimAndRun() {
        for (;;) {
            size_t i;
            {
                std::unique_lock<std::mutex> lk(m_);
                if (next_ >= total_) return;
                i = next_++;
            }
            body_(i, ctx_);
            {
                std::unique_lock<std::mutex> lk(m_);
                ++done_;
                if (done_ == total_) doneCv_.notify_all();
            }
        }
    }

    void run() {
        size_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this, &seen] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            claimAndRun();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, doneCv_;
    void (*body_)(size_t, void*) = nullptr;
    void* ctx_ = nullptr;
    size_t total_ = 0, next_ = 0, done_ = 0, generation_ = 0;
    bool stop_ = false;
};

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
