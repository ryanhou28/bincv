// cornerSubPix against cv::cornerSubPix on a real sequence: how far apart the two land.
//
// A role claim, not a speed one. ops/subpix.hpp claims to be cv::cornerSubPix on the
// image its derivative planes came from; this measures that claim on the corners a
// detector finds in real binarized frames, which is where a refinement that agreed on
// clean synthetic corners was once a median of 0.73 px away.
//
// WHAT IS COMPARED, per corner, from identical seeds on identical bits:
//   binCV       bincv::cornerSubPix on the frame's ternary derivatives, taken with
//               BORDER_REPLICATE -- cv::getRectSubPix's border
//   OpenCV      cv::cornerSubPix on the same bits as 0/255
//   the floor   cv::cornerSubPix on the same bits as 0/1, against itself on 0/255.
//               Only rounding separates the two, and an ill-conditioned walk ends a
//               step apart on rounding alone, so no reimplementation can agree with
//               OpenCV more closely than OpenCV agrees with itself.
// Corners seeded within winHalf + 2 pixels of an edge, whose windows can reach past the
// image, are reported apart, since there the border rule is what is being compared.
//
// CONTENT: every `stride`-th frame of a directory of 8-bit PNGs, binarized by the
// reference sensor stage at threshold 17 (benchmark/reference_sensor.hpp), seeded by
// cv::goodFeaturesToTrack (200 corners, quality 0.01, spacing 10) on the 0/255 frame.
// Two parameter sets: this library's defaults (window 11, 40 iterations, eps 0.001)
// and window 21, 20 iterations, eps 0.03.
//
//   usage: subpix_agreement <frame-dir> [stride]     (default stride 10)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bincv/binMat.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/subpix.hpp"
#include "reference_sensor.hpp"

namespace fs = std::filesystem;
using W = uint32_t;

namespace {

double distance(const cv::Point2f& a, const cv::Point2f& b) {
    return std::hypot(static_cast<double>(a.x) - static_cast<double>(b.x),
                      static_cast<double>(a.y) - static_cast<double>(b.y));
}

double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size() - 1) + 0.5))];
}

size_t beyond(const std::vector<double>& v, double r) {
    return static_cast<size_t>(std::count_if(v.begin(), v.end(), [r](double d) { return d > r; }));
}

void row(const char* name, const std::vector<double>& v) {
    if (v.empty()) {
        std::printf("  %-26s (none)\n", name);
        return;
    }
    std::printf("  %-26s %6zu  median %.2e  p99 %.2e  max %.2e  beyond 1e-3 px: %zu (%.2f%%)\n",
                name, v.size(), quantile(v, 0.5), quantile(v, 0.99), quantile(v, 1.0),
                beyond(v, 1e-3), 100.0 * static_cast<double>(beyond(v, 1e-3)) /
                                     static_cast<double>(v.size()));
}

struct Tally {
    std::vector<double> bincv, floor, bincvEdge, floorEdge;
    bincv::SubPixResult outcomes;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <frame-dir> [stride]\n", argv[0]);
        return 2;
    }
    const size_t stride = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 10;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(argv[1])) {
        if (e.path().extension() == ".png") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty() || stride == 0) {
        std::fprintf(stderr, "no frames in %s\n", argv[1]);
        return 2;
    }
    cv::setNumThreads(1);

    bincv::SubPixParams defaults, wide;
    wide.winHalf = 10;
    wide.maxIterations = 20;
    wide.epsilon = 0.03;
    const bincv::SubPixParams sets[2] = {defaults, wide};
    Tally tally[2];
    size_t frames = 0;

    for (size_t f = 0; f < files.size(); f += stride) {
        const cv::Mat gray = cv::imread(files[f].string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            std::fprintf(stderr, "%s unreadable\n", files[f].c_str());
            return 2;
        }
        const cv::Mat bytes = refsensor::preprocess(gray, 17);   // 0/255
        const cv::Mat ones = bytes / 255;
        const int w = bytes.cols, h = bytes.rows;
        std::vector<cv::Point2f> seeds;
        cv::goodFeaturesToTrack(bytes, seeds, 200, 0.01, 10.0);

        bincv::BinMat<W> bits(w, h);
        bincv::packBits<bincv::PackRule::NonZero>(bytes.data, static_cast<size_t>(w),
                                                  static_cast<size_t>(h), bytes.step,
                                                  bits.view());
        bincv::SignedQuantMat<1, W> dx(w, h), dy(w, h);
        bincv::derivativeX(bits, dx, bincv::BORDER_REPLICATE);
        bincv::derivativeY(bits, dy, bincv::BORDER_REPLICATE);

        for (int k = 0; k < 2; ++k) {
            const bincv::SubPixParams& p = sets[k];
            std::vector<bincv::Point2f> mine;
            for (const cv::Point2f& s : seeds) mine.push_back(bincv::Point2f{s.x, s.y});
            const bincv::SubPixResult r =
                bincv::cornerSubPix<1, W>(dx, dy, mine.data(), mine.size(), p);
            tally[k].outcomes.refined += r.refined;
            tally[k].outcomes.singular += r.singular;
            tally[k].outcomes.outside += r.outside;
            tally[k].outcomes.diverged += r.diverged;

            std::vector<cv::Point2f> theirs = seeds, theirsOnes = seeds;
            const cv::TermCriteria crit(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                        p.maxIterations, p.epsilon);
            const cv::Size win(p.winHalf, p.winHalf), zero(-1, -1);
            cv::cornerSubPix(bytes, theirs, win, zero, crit);
            cv::cornerSubPix(ones, theirsOnes, win, zero, crit);

            // A window reads integer pixels floor(x) - winHalf .. floor(x) + winHalf + 1.
            const float reach = static_cast<float>(p.winHalf + 1);
            for (size_t i = 0; i < seeds.size(); ++i) {
                const bool edge = seeds[i].x < reach || seeds[i].y < reach ||
                                  seeds[i].x >= static_cast<float>(w) - reach - 1.0f ||
                                  seeds[i].y >= static_cast<float>(h) - reach - 1.0f;
                const double db = distance(cv::Point2f(mine[i].x, mine[i].y), theirs[i]);
                const double df = distance(theirsOnes[i], theirs[i]);
                (edge ? tally[k].bincvEdge : tally[k].bincv).push_back(db);
                (edge ? tally[k].floorEdge : tally[k].floor).push_back(df);
            }
        }
        ++frames;
    }

    std::printf("=== cornerSubPix against cv::cornerSubPix, %zu frames (every %zu) of %s ===\n",
                frames, stride, argv[1]);
    for (int k = 0; k < 2; ++k) {
        const bincv::SubPixParams& p = sets[k];
        const Tally& t = tally[k];
        std::printf("\nwindow %d, %d iterations, eps %g: refined %zu, singular %zu, outside %zu,"
                    " diverged %zu\n", 2 * p.winHalf + 1, p.maxIterations, p.epsilon,
                    t.outcomes.refined, t.outcomes.singular, t.outcomes.outside,
                    t.outcomes.diverged);
        std::vector<double> all = t.bincv, allFloor = t.floor;
        all.insert(all.end(), t.bincvEdge.begin(), t.bincvEdge.end());
        allFloor.insert(allFloor.end(), t.floorEdge.begin(), t.floorEdge.end());
        row("binCV vs OpenCV, all", all);
        row("  interior", t.bincv);
        row("  window past the image", t.bincvEdge);
        row("OpenCV 0/1 vs 0/255, all", allFloor);
        row("  interior", t.floor);
        row("  window past the image", t.floorEdge);
    }
    return 0;
}
