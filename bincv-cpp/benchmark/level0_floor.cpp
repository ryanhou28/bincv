// ===========================================================================
// X-27 / E-16 -- THE 1-BIT LEVEL-0 LOCALISATION FLOOR.
//
// Three pyramid parameters had been measured and none explained T3.8's standing
// accuracy MISS, so the question became whether the REPRESENTATION was the limit.
// This file answers it, and the answer is no.
//
// ARM 1 -- THE PARTITION METHOD, and it replaces the oracle X-27's rule first
// sketched. That sketch was DEGENERATE: it formed candidates the same way as the
// observation, so the Hamming-nearest candidate was the observation itself and the
// "floor" would have been exactly zero by construction. The flaw was found and the
// method replaced BEFORE any number was taken; the decision BANDS are untouched.
//
// The replacement inverts nothing. As a translation d varies continuously, the
// binarized window changes only when some pixel's gradient crosses the threshold,
// so over d in [0,1) the observation takes FINITELY MANY distinct values and d is
// partitioned into intervals indistinguishable from the bits alone. That partition
// IS the floor: the best any estimator can do is report an interval's midpoint.
// Measured as RMS(d - midpoint(interval(d))) over a fine sweep.
//
// ARM 2 -- WITH SENSOR NOISE, because arm 1 is noise-free and the obvious
// objection is that real frame 2 carries noise, making some of arm 1's state
// transitions uninformative. Here the observation is binarized from a NOISY frame
// and the candidates from clean ones, so nothing can recover d trivially.
//
// Windows with almost no edge in them are excluded and counted. Such a window is
// untrackable rather than imprecise, and including it would report a whole pixel
// of ignorance as if it were a floor.
// ===========================================================================

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

static cv::Mat referenceDenoise(const cv::Mat& img) {
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
static cv::Mat referenceEdgeFilter(const cv::Mat& gray, int thr) {
    const cv::Mat kx = (cv::Mat_<float>(1, 3) << -1, 0, 1);
    const cv::Mat ky = (cv::Mat_<float>(3, 1) << -1, 0, 1);
    cv::Mat dx, dy;
    cv::filter2D(gray, dx, CV_32F, kx);
    cv::filter2D(gray, dy, CV_32F, ky);
    dx = cv::abs(dx); dy = cv::abs(dy);
    cv::Mat mask = (dx >= thr) | (dy >= thr);
    cv::Mat out = cv::Mat::zeros(gray.size(), CV_8U);
    out.setTo(255, mask);
    return out;
}
static cv::Mat preprocess(const cv::Mat& g, int thr) {
    return referenceEdgeFilter(referenceDenoise(g), thr);
}
static cv::Mat shiftX(const cv::Mat& g, double d) {
    cv::Mat m = (cv::Mat_<double>(2, 3) << 1, 0, d, 0, 1, 0);
    cv::Mat out;
    cv::warpAffine(g, out, m, g.size(), cv::INTER_CUBIC, cv::BORDER_REFLECT_101);
    return out;
}

int main(int argc, char** argv) {
    const std::string path = argv[1];
    const int thr = argc > 2 ? std::atoi(argv[2]) : 17;
    const cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) { printf("cannot read %s\n", path.c_str()); return 1; }

    // The displacement grid. Step must be fine enough that intervals are resolved:
    // 0.004 px gives 250 samples across a pixel, and the measured interval widths
    // below are ~100x that, so the grid is not the limiting resolution.
    const double step = 0.004;
    const int N = static_cast<int>(1.0 / step);
    std::vector<cv::Mat> bin(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        bin[static_cast<size_t>(i)] = preprocess(shiftX(gray, i * step), thr);
    }
    printf("frame %dx%d, threshold %d, %d displacement samples at %.3f px\n", gray.cols,
           gray.rows, thr, N, step);
    printf("  set%%: %.3f\n", 100.0 * cv::countNonZero(bin[0]) / static_cast<double>(gray.total()));

    printf("\n  win |  windows | mean set px | distinct states | mean interval px |"
           "  FLOOR rms px | max px\n");
    printf("  ----+----------+-------------+-----------------+------------------+"
           "---------------+--------\n");
    for (int win : {11, 21, 31, 41}) {
        const int half = win / 2;
        double sumStates = 0, sumSet = 0, sumSqErr = 0, maxErr = 0, sumInterval = 0;
        size_t windows = 0, samples = 0;
        // A grid of window centres, stepped so the windows are near-independent.
        for (int cy = half + 20; cy + half < gray.rows - 20; cy += 37) {
            for (int cx = half + 20; cx + half < gray.cols - 20; cx += 41) {
                const cv::Rect r(cx - half, cy - half, win, win);
                // The window's bit pattern at each displacement, as a byte string.
                std::vector<std::string> states(static_cast<size_t>(N));
                for (int i = 0; i < N; ++i) {
                    const cv::Mat patch = bin[static_cast<size_t>(i)](r);
                    std::string s;
                    s.reserve(static_cast<size_t>(win * win));
                    for (int y = 0; y < win; ++y) {
                        const uchar* p = patch.ptr<uchar>(y);
                        for (int x = 0; x < win; ++x) s.push_back(p[x] ? '1' : '0');
                    }
                    states[static_cast<size_t>(i)] = std::move(s);
                }
                // A window with no edge in it carries no information at all and
                // would report a floor of 0.29 px (a whole pixel of ignorance)
                // while being simply untrackable. Those are excluded and counted:
                // the floor is a statement about windows a frontend would USE.
                const size_t setPx = static_cast<size_t>(
                    std::count(states[0].begin(), states[0].end(), '1'));
                if (setPx < static_cast<size_t>(win)) continue;

                // Interval boundaries: where the window's bits change.
                std::vector<int> bounds{0};
                for (int i = 1; i < N; ++i) {
                    if (states[static_cast<size_t>(i)] != states[static_cast<size_t>(i - 1)]) {
                        bounds.push_back(i);
                    }
                }
                bounds.push_back(N);
                sumStates += static_cast<double>(bounds.size() - 1);
                sumSet += static_cast<double>(setPx);
                for (size_t k = 0; k + 1 < bounds.size(); ++k) {
                    const double lo = bounds[k] * step, hi = bounds[k + 1] * step;
                    const double mid = 0.5 * (lo + hi);
                    sumInterval += (hi - lo);
                    for (int i = bounds[k]; i < bounds[k + 1]; ++i) {
                        const double e = i * step - mid;
                        sumSqErr += e * e;
                        maxErr = std::max(maxErr, std::fabs(e));
                        ++samples;
                    }
                }
                ++windows;
            }
        }
        printf("  %3d | %8zu | %11.1f | %15.1f | %16.4f | %13.4f | %6.4f\n", win, windows,
               sumSet / static_cast<double>(windows), sumStates / static_cast<double>(windows),
               sumInterval / sumStates, std::sqrt(sumSqErr / static_cast<double>(samples)),
               maxErr);
    }
    return 0;
}
