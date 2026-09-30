#pragma once

// Image I/O helpers for tests and examples. These genuinely need OpenCV, so they
// are not part of the dependency-free core.

#ifdef BINCV_WITH_OPENCV

#include <cstdint>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace bincv {
namespace util {

/// @brief Writes an 8-bit image to `tests/output/<imageName>` through `cv::imwrite`.
/// **API TIER 3** -- a helper for the tests and examples, not a library operation;
/// it exists only in builds with OpenCV.
/// @param h_input Row-major, `width` bytes per row, `height` rows.
void save_test_image(const std::string& imageName, const uint8_t* h_input, int width, int height);

} // namespace util
} // namespace bincv

#endif // BINCV_WITH_OPENCV