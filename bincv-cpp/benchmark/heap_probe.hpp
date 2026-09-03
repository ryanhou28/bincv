#ifndef BINCV_BENCHMARK_HEAP_PROBE_HPP
#define BINCV_BENCHMARK_HEAP_PROBE_HPP

/// @file
/// Peak simultaneously-live heap of one call, measured at the C allocator.
///
/// WHY NOT A REPLACED `operator new`. It cannot see OpenCV. `cv::Mat` allocates
/// through `cv::fastMalloc`, which calls `malloc` directly, so a replaced
/// `operator new` never observes the matrix data -- the largest blocks in the
/// call. Measured both ways in one process, `cv::findEssentialMat` on 1 000
/// correspondences reports 2 744 B through `operator new` and 46 968 B through
/// `malloc`: an under-count of 17x, and it does not grow with the input, which
/// is the tell. binCV allocates nothing, so the error fell entirely on OpenCV.
///
/// This probe interposes the C allocator instead, so it sees every path into
/// the heap. It is cross-checked against `valgrind --tool=dhat`, which agrees
/// within allocator rounding.
///
/// WHAT IS COUNTED. `malloc_usable_size`, which is what the allocator actually
/// committed, including its rounding. That is the honest figure for a footprint
/// claim: a caller pays the rounding whether or not the program asked for it.
///
/// THREADING. The counters are plain, so measure single-threaded work only.

#include <cstddef>

namespace heapprobe {

struct Reading {
    std::size_t peakLive = 0;    ///< HIGH-WATER of simultaneously-live bytes
    std::size_t calls = 0;       ///< allocator round-trips in the window
    std::size_t smallCalls = 0;  ///< of those, blocks under 128 B
    long long net = 0;           ///< still held when the window closed; should be ~0
};

/// @brief Open a measurement window. Counters reset; the high-water starts here.
void begin();

/// @brief Close the window and return what happened inside it.
Reading end();

/// @brief Does the probe recover known answers? Prints a line per check.
/// A probe that has never returned a known answer is not evidence, so callers
/// should refuse to print figures when this is false.
bool selfCheck();

/// @brief Peak live bytes attributable to `f`, for a caller that wants one number.
template <typename F>
inline Reading around(F&& f) {
    begin();
    f();
    return end();
}

} // namespace heapprobe

#endif // BINCV_BENCHMARK_HEAP_PROBE_HPP
