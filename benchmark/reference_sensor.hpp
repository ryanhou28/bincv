#pragma once

// The REFERENCE sensor stage, in one place, because two harnesses stake claims
// on running the identical spelling:
//
//   * feature_tracking_sequence.cpp checks binCV's own sensor stage bit-exact against
//     this, every frame -- the control that makes its comparison one of
//     implementations rather than of inputs.
//   * accuracy_realframes.cpp binarizes its pairs with it so its yield deltas
//     are comparable to the pipeline's -- the one comparison that file exists
//     to make.
//
// A copy in each file is how the two silently diverge while both keep printing
// confident numbers; an include cannot drift.
//
// The spelling is the reference pipeline's own two stages, read from the
// reference rather than inferred: the L-shaped three-pixel median (min/max
// network), then |d/dx| >= t OR |d/dy| >= t over [-1, 0, 1].

#include <opencv2/opencv.hpp>

namespace refsensor {

inline cv::Mat referenceDenoise(const cv::Mat& img) {
    cv::Mat right = cv::Mat::zeros(img.size(), img.type());
    cv::Mat above = cv::Mat::zeros(img.size(), img.type());
    img.colRange(1, img.cols).copyTo(right.colRange(0, img.cols - 1));
    img.rowRange(0, img.rows - 1).copyTo(above.rowRange(1, img.rows));
    cv::Mat a, b, c, out;
    cv::min(above, img, a);
    cv::max(above, img, b);
    cv::min(b, right, c);
    cv::max(a, c, out);
    return out;
}

inline cv::Mat referenceEdgeFilter(const cv::Mat& gray, int thr) {
    const cv::Mat kx = (cv::Mat_<float>(1, 3) << -1, 0, 1);
    const cv::Mat ky = (cv::Mat_<float>(3, 1) << -1, 0, 1);
    cv::Mat dx, dy;
    cv::filter2D(gray, dx, CV_32F, kx);
    cv::filter2D(gray, dy, CV_32F, ky);
    dx = cv::abs(dx);
    dy = cv::abs(dy);
    const cv::Mat mask = (dx >= thr) | (dy >= thr);
    cv::Mat out = cv::Mat::zeros(gray.size(), CV_8U);
    out.setTo(255, mask);
    return out;
}

inline cv::Mat preprocess(const cv::Mat& g, int thr) {
    return referenceEdgeFilter(referenceDenoise(g), thr);
}

} // namespace refsensor
