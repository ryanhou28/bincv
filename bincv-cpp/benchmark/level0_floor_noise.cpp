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
#include <random>
#include <algorithm>

static cv::Mat referenceDenoise(const cv::Mat& img) {
    cv::Mat right = cv::Mat::zeros(img.size(), img.type());
    cv::Mat above = cv::Mat::zeros(img.size(), img.type());
    img.colRange(1, img.cols).copyTo(right.colRange(0, img.cols - 1));
    img.rowRange(0, img.rows - 1).copyTo(above.rowRange(1, img.rows));
    cv::Mat a, b, c, out;
    cv::min(above, img, a); cv::max(above, img, b);
    cv::min(b, right, c);   cv::max(a, c, out);
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
    if (gray.empty()) return 1;

    const double step = 0.01;
    const int N = 100;
    std::vector<cv::Mat> cand(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) cand[static_cast<size_t>(i)] = preprocess(shiftX(gray, i * step), thr);

    const int win = 31, half = win / 2;
    std::vector<cv::Rect> rects;
    for (int cy = half + 20; cy + half < gray.rows - 20; cy += 37) {
        for (int cx = half + 20; cx + half < gray.cols - 20; cx += 41) {
            rects.emplace_back(cx - half, cy - half, win, win);
        }
    }
    printf("31x31, %zu windows, candidate grid %.2f px, threshold %d\n", rects.size(), step, thr);
    printf("  sigma (gray levels) |  FLOOR rms px | max px | median px\n");
    printf("  --------------------+---------------+--------+----------\n");

    std::mt19937 rng(12345);
    for (double sigma : {0.0, 0.5, 1.0, 2.0, 4.0}) {
        std::vector<double> errs;
        for (int t = 10; t < N; t += 17) {                 // true displacements
            const double d = t * step;
            cv::Mat clean = shiftX(gray, d), noisy = clean.clone();
            if (sigma > 0.0) {
                cv::Mat n(clean.size(), CV_32F);
                std::normal_distribution<float> g(0.0f, static_cast<float>(sigma));
                for (int y = 0; y < n.rows; ++y) {
                    float* p = n.ptr<float>(y);
                    for (int x = 0; x < n.cols; ++x) p[x] = g(rng);
                }
                cv::Mat f; clean.convertTo(f, CV_32F);
                f += n; f.convertTo(noisy, CV_8U);
            }
            const cv::Mat obs = preprocess(noisy, thr);
            for (const cv::Rect& r : rects) {
                const cv::Mat op = obs(r);
                if (cv::countNonZero(op) < win) continue;
                int best = -1; int bestCost = 1 << 30;
                for (int i = 0; i < N; ++i) {
                    cv::Mat diff;
                    cv::compare(op, cand[static_cast<size_t>(i)](r), diff, cv::CMP_NE);
                    const int c = cv::countNonZero(diff);
                    if (c < bestCost) { bestCost = c; best = i; }
                }
                errs.push_back(std::fabs(best * step - d));
            }
        }
        std::sort(errs.begin(), errs.end());
        double sq = 0.0;
        for (double e : errs) sq += e * e;
        printf("  %19.1f | %13.4f | %6.4f | %8.4f\n", sigma,
               std::sqrt(sq / static_cast<double>(errs.size())), errs.back(),
               errs[errs.size() / 2]);
    }
    return 0;
}
