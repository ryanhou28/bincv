// ===========================================================================
// ops/subpix.hpp -- sub-pixel corner refinement on bit-planes.
//
// THE OPERATION RESTS ON ONE PROPERTY AND THIS FILE PINS IT.
//
//   cv::cornerSubPix solves  G q = b  with  G = sum w (grad I)(grad I)^T.
//   Scaling the image by s scales grad I by s and therefore scales BOTH G and b by
//   s^2, so q = G^-1 b is UNCHANGED.
//
// That is why a ternary derivative carrying +/-1 refines to the same place as a byte
// pipeline carrying +/-255 -- and it is the whole argument for binCV shipping this at
// all. A binCV user checked it before asking for the operation (0.00018 px mean over
// 5924 corners); `ScaleInvariance` below checks it here, where a regression can be
// caught.
//
// Three claims, in the order they matter:
//   1. SCALE INVARIANCE -- the property above, on the formula itself.
//   2. ORACLE EQUALITY -- the bit-plane kernel equals a dense reference on the same
//      data. The kernel's whole trick is skipping zero-gradient pixels word-wise, and
//      a skip that drops a contributing pixel is exactly the bug that would not show
//      up as a crash.
//   3. IT ACTUALLY REFINES -- on a corner whose true position is known, the refined
//      position is closer than the integer one it started from.
// ===========================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/subpix.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/imgproc.hpp>
#endif

using bincv::Point2f;

namespace {

/// A dense, obvious, double-precision spelling of the same refinement. Slow on purpose:
/// it visits every pixel of the window and skips nothing, so it is a genuine oracle for
/// the bit-plane kernel's word-wise skip.
/// @param scale multiplies every gradient -- the knob `ScaleInvariance` turns.
Point2f denseRefine(const std::vector<int>& gx, const std::vector<int>& gy, size_t w,
                    size_t h, Point2f start, int winHalf, double scale, int maxIter,
                    double epsilon) {
    const int side = 2 * winHalf + 1;
    std::vector<double> mask(static_cast<size_t>(side) * static_cast<size_t>(side));
    bincv::impl::subPixMask(winHalf, -1, mask.data());
    double cx = start.x, cy = start.y;
    for (int it = 0; it < maxIter; ++it) {
        const long long ix0 = static_cast<long long>(std::floor(cx + 0.5));
        const long long iy0 = static_cast<long long>(std::floor(cy + 0.5));
        if (ix0 - winHalf < 0 || iy0 - winHalf < 0 ||
            ix0 + winHalf >= static_cast<long long>(w) ||
            iy0 + winHalf >= static_cast<long long>(h)) {
            break;
        }
        double gxx = 0, gxy = 0, gyy = 0, bx = 0, by = 0;
        for (int wy = -winHalf; wy <= winHalf; ++wy) {
            for (int wx = -winHalf; wx <= winHalf; ++wx) {
                const size_t idx = static_cast<size_t>(iy0 + wy) * w +
                                   static_cast<size_t>(ix0 + wx);
                const double a = scale * static_cast<double>(gx[idx]);
                const double b = scale * static_cast<double>(gy[idx]);
                const double weight = mask[static_cast<size_t>(wy + winHalf) *
                                               static_cast<size_t>(side) +
                                           static_cast<size_t>(wx + winHalf)];
                const double xx = weight * a * a, xy = weight * a * b, yy = weight * b * b;
                gxx += xx; gxy += xy; gyy += yy;
                bx += xx * wx + xy * wy;
                by += xy * wx + yy * wy;
            }
        }
        const double det = gxx * gyy - gxy * gxy;
        if (det == 0.0) break;
        const double qx = (gyy * bx - gxy * by) / det;
        const double qy = (gxx * by - gxy * bx) / det;
        if (std::fabs(qx) > winHalf || std::fabs(qy) > winHalf) break;
        const double nx = static_cast<double>(ix0) + qx, ny = static_cast<double>(iy0) + qy;
        const double sx = nx - cx, sy = ny - cy;
        cx = nx; cy = ny;
        if (sx * sx + sy * sy <= epsilon * epsilon) break;
    }
    // cv::cornerSubPix's poor-convergence rule, which the kernel applies and so must
    // the oracle -- otherwise this test reports a disagreement wherever the rule fires
    // and calls a correct kernel wrong.
    if (std::fabs(cx - static_cast<double>(start.x)) > winHalf ||
        std::fabs(cy - static_cast<double>(start.y)) > winHalf) {
        return start;
    }
    return Point2f{static_cast<float>(cx), static_cast<float>(cy)};
}

/// A quadrant corner at (cornerX, cornerY): set where both coordinates are past it.
/// The two straight edges meet exactly there, which is the position the refinement
/// should recover.
void quadrant(bincv::BinMat<uint32_t>& m, int cornerX, int cornerY) {
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            m.set(y, x, (x >= cornerX && y >= cornerY) ? 1u : 0u);
        }
    }
}

/// The signed derivative planes as plain integer arrays, for the dense oracle.
template <size_t N>
void toDense(const bincv::SignedQuantMat<N, uint32_t>& d, std::vector<int>& out, size_t w,
             size_t h) {
    out.assign(w * h, 0);
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            out[y * w + x] = d.at(static_cast<int>(y), static_cast<int>(x));
        }
    }
}

}  // namespace

/// An ASYMMETRIC corner: the quadrant with a short notch cut from one edge.
///
/// **WHY THIS EXISTS, AND IT IS NOT THE REASON IT FIRST LOOKED LIKE.** The obvious story
/// for F-4 is that a symmetric corner cannot see a radially symmetric mask. Measured,
/// that is FALSE: the symmetric case reports 0.0325 px with the wrong mask and 0.0035 px
/// with the right one -- sensitive by a factor of nine. It passed because its BOUND was
/// 0.1, fitted to the value the code produced instead of to the value a correct
/// implementation reaches.
///
/// What this case adds is different and still worth having: on a symmetric corner the
/// two gradient operators agree almost exactly, so the comparison is easy and the
/// residual is tiny. Asymmetric content is where binCV's ternary derivative and OpenCV's
/// byte-domain one actually diverge, so this is the case that states the Tier 2 gap
/// honestly rather than flattering it.
void asymmetricCorner(bincv::BinMat<uint32_t>& m, int cornerX, int cornerY) {
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            bool v = (x >= cornerX && y >= cornerY);
            // A short notch cut out of the HORIZONTAL edge on one side only, two pixels
            // from the corner. It has to be MILD: a strong asymmetry drags the solution
            // several pixels, which puts it on the poor-convergence threshold, and there
            // a 0.2 px difference in the converged position becomes a 5 px difference in
            // the OUTPUT -- so the test would be measuring that discontinuity and not the
            // Gaussian width it exists to check.
            if (x >= cornerX + 2 && x <= cornerX + 3 && y >= cornerY && y <= cornerY + 1) {
                v = false;
            }
            m.set(y, x, v ? 1u : 0u);
        }
    }
}

BINCV_TEST(SubPix, ScaleInvariance_TernaryRefinesWhereBytesDo) {
    // THE PROPERTY THE WHOLE OPERATION RESTS ON. Same gradients, scaled by 1 and by
    // 255: G and b both scale by s^2, so the solution must not move.
    constexpr int W = 60, H = 60;
    bincv::BinMat<uint32_t> img(W, H);
    quadrant(img, 30, 30);
    bincv::SignedQuantMat<1, uint32_t> dx(W, H), dy(W, H);
    bincv::derivativeX(img, dx);
    bincv::derivativeY(img, dy);
    std::vector<int> gx, gy;
    toDense<1>(dx, gx, W, H);
    toDense<1>(dy, gy, W, H);

    double worst = 0.0, total = 0.0;
    size_t n = 0;
    for (int sy = -2; sy <= 2; ++sy) {
        for (int sx = -2; sx <= 2; ++sx) {
            const Point2f start{static_cast<float>(30 + sx), static_cast<float>(30 + sy)};
            const Point2f one = denseRefine(gx, gy, W, H, start, 5, 1.0, 40, 0.001);
            const Point2f big = denseRefine(gx, gy, W, H, start, 5, 255.0, 40, 0.001);
            const double d = std::hypot(static_cast<double>(one.x) - big.x,
                                        static_cast<double>(one.y) - big.y);
            worst = d > worst ? d : worst;
            total += d;
            ++n;
        }
    }
    std::printf("  +/-1 vs +/-255 over %zu starts: mean %.3e px, worst %.3e px\n", n,
                total / static_cast<double>(n), worst);
    BINCV_CHECK(n == 25);
    BINCV_CHECK(worst < 1e-4);   // the user measured 1.8e-4 mean against OpenCV's own
}

BINCV_TEST(SubPix, MatchesDenseOracle_TheWordWiseSkipDropsNothing) {
    // The kernel's trick is finding contributing pixels a WORD at a time and visiting
    // only those. A skip that drops a pixel with a gradient changes the answer quietly,
    // so the dense spelling -- which skips nothing -- is the oracle.
    constexpr int W = 80, H = 70;
    bincv::BinMat<uint32_t> img(W, H);
    // Not a clean quadrant: a diagonal plus a block, so windows straddle word
    // boundaries with irregular gradient patterns rather than tidy straight edges.
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const bool on = (x + y > 60) || (x > 20 && x < 34 && y > 12 && y < 40);
            img.set(y, x, on ? 1u : 0u);
        }
    }
    bincv::SignedQuantMat<1, uint32_t> dx(W, H), dy(W, H);
    bincv::derivativeX(img, dx);
    bincv::derivativeY(img, dy);
    std::vector<int> gx, gy;
    toDense<1>(dx, gx, W, H);
    toDense<1>(dy, gy, W, H);

    // Starts spread across word boundaries: 31, 32, 33 and 63, 64, 65 straddle the
    // 32-bit words the kernel loads.
    std::vector<Point2f> pts;
    for (int x : {14, 20, 31, 32, 33, 40, 63, 64, 65}) {
        for (int y : {18, 25, 31, 32, 44}) {
            pts.push_back(Point2f{static_cast<float>(x), static_cast<float>(y)});
        }
    }
    std::vector<Point2f> mine = pts;
    bincv::SubPixParams p;
    p.winHalf = 5;
    const bincv::SubPixResult r =
        bincv::cornerSubPix<1, uint32_t>(dx, dy, mine.data(), mine.size(), p);

    double worst = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const Point2f ref = denseRefine(gx, gy, W, H, pts[i], 5, 1.0, p.maxIterations,
                                        p.epsilon);
        const double d = std::hypot(static_cast<double>(mine[i].x) - ref.x,
                                    static_cast<double>(mine[i].y) - ref.y);
        worst = d > worst ? d : worst;
    }
    std::printf("  %zu corners vs dense oracle: worst %.3e px  (refined %zu, singular %zu,"
                " clamped %zu)\n", pts.size(), worst, r.refined, r.singular, r.clamped);
    BINCV_CHECK(pts.size() > 40);
    BINCV_CHECK(worst < 1e-5);
    BINCV_CHECK(r.refined > 0);
}

BINCV_TEST(SubPix, ActuallyRefines_TowardsAKnownCorner) {
    // A quadrant corner sits exactly at (32, 27). Start off it and check the refinement
    // moves TOWARD it -- an operation that runs, converges and reports success while
    // making localisation worse would pass both tests above.
    constexpr int W = 70, H = 60;
    const int cxTrue = 32, cyTrue = 27;
    bincv::BinMat<uint32_t> img(W, H);
    quadrant(img, cxTrue, cyTrue);
    bincv::SignedQuantMat<1, uint32_t> dx(W, H), dy(W, H);
    bincv::derivativeX(img, dx);
    bincv::derivativeY(img, dy);

    double beforeSum = 0.0, afterSum = 0.0;
    size_t improved = 0, cases = 0;
    for (int sy = -2; sy <= 2; ++sy) {
        for (int sx = -2; sx <= 2; ++sx) {
            if (sx == 0 && sy == 0) continue;
            Point2f pt{static_cast<float>(cxTrue + sx), static_cast<float>(cyTrue + sy)};
            const double before = std::hypot(static_cast<double>(pt.x) - cxTrue,
                                             static_cast<double>(pt.y) - cyTrue);
            bincv::cornerSubPix<1, uint32_t>(dx, dy, &pt, 1);
            const double after = std::hypot(static_cast<double>(pt.x) - cxTrue,
                                            static_cast<double>(pt.y) - cyTrue);
            beforeSum += before;
            afterSum += after;
            if (after < before) ++improved;
            ++cases;
        }
    }
    std::printf("  %zu starts around a known corner: mean distance %.3f -> %.3f px,"
                " %zu improved\n", cases, beforeSum / static_cast<double>(cases),
                afterSum / static_cast<double>(cases), improved);
    BINCV_CHECK(cases == 24);
    BINCV_CHECK(afterSum < beforeSum);
    BINCV_CHECK(improved * 2 > cases);   // a majority, not one lucky start
}

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(SubPix, AgreesWithOpenCVOnTheSameCorner) {
    // TIER 2, AND THIS IS THE HALF THAT SAYS SO. The refinement rule, the Gaussian
    // window and the termination are cv::cornerSubPix's; the GRADIENT is not -- OpenCV
    // derives its own from the 8-bit image, binCV takes the ternary derivatives the
    // frontend already has. So the two agree in ROLE and land close, and this pins how
    // close rather than leaving "close" as an adjective.
    constexpr int W = 70, H = 60;
    const int cxTrue = 32, cyTrue = 27;
    bincv::BinMat<uint32_t> img(W, H);
    quadrant(img, cxTrue, cyTrue);
    bincv::SignedQuantMat<1, uint32_t> dx(W, H), dy(W, H);
    bincv::derivativeX(img, dx);
    bincv::derivativeY(img, dy);

    cv::Mat bytes(H, W, CV_8U, cv::Scalar(0));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            bytes.at<uint8_t>(y, x) = img.at(y, x) ? 255 : 0;
        }
    }

    std::vector<Point2f> mine;
    std::vector<cv::Point2f> theirs;
    for (int sy = -2; sy <= 2; ++sy) {
        for (int sx = -2; sx <= 2; ++sx) {
            mine.push_back(Point2f{static_cast<float>(cxTrue + sx),
                                   static_cast<float>(cyTrue + sy)});
            theirs.push_back(cv::Point2f(static_cast<float>(cxTrue + sx),
                                         static_cast<float>(cyTrue + sy)));
        }
    }
    bincv::SubPixParams p;
    p.winHalf = 5;
    bincv::cornerSubPix<1, uint32_t>(dx, dy, mine.data(), mine.size(), p);
    cv::cornerSubPix(bytes, theirs, cv::Size(p.winHalf, p.winHalf), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                      p.maxIterations, p.epsilon));

    double worst = 0.0, total = 0.0;
    for (size_t i = 0; i < mine.size(); ++i) {
        const double d = std::hypot(static_cast<double>(mine[i].x) - theirs[i].x,
                                    static_cast<double>(mine[i].y) - theirs[i].y);
        worst = d > worst ? d : worst;
        total += d;
    }
    const double mean = total / static_cast<double>(mine.size());
    std::printf("  binCV vs cv::cornerSubPix over %zu starts: mean %.4f px, worst %.4f px\n",
                mine.size(), mean, worst);
    BINCV_CHECK(mine.size() == 25);
    // A TOLERANCE, NOT AN EQUALITY, and the docstring says why: different gradient
    // operators on the same content. **0.1 px, against a measured 0.0325** -- a bound
    // thirty times looser than the number it guards is not a test, it is a comment.
    // BOUNDS SET FROM WHAT A CORRECT IMPLEMENTATION REACHES (0.0035), NOT FROM WHAT THIS
    // CODE HAPPENS TO PRODUCE. The old bounds were 0.1 and 0.15 against a measured
    // 0.0325 -- and 0.0325 was the WRONG MASK's number. A tolerance fitted to the
    // observation cannot fail, which is how F-4 shipped past this file.
    BINCV_CHECK(mean < 0.01);
    BINCV_CHECK(worst < 0.01);
}
#endif  // BINCV_WITH_OPENCV


// ---------------------------------------------------------------------------
// F-4 -- THE TIER 2 GAP, STATED HONESTLY
//
// Reported from outside: binCV's cornerSubPix disagreed with OpenCV's by 4.53 px mean on
// real frames, while the test above reported 0.0325 px. Two causes, both now fixed -- a
// Gaussian sqrt(2) too narrow, and a MISSING poor-convergence rule (OpenCV reverts to the
// seed when the result lands further than the half-window away; binCV kept walking).
//
// This case is the same comparison on asymmetric content, which is where the two gradient
// operators genuinely differ. It reports ~0.47 px, against ~0.0035 px on an ideal
// symmetric corner -- and that spread is the point: the symmetric number is not
// representative of what a caller gets on a real frame, and quoting it alone is how
// D-74 came to advertise 0.0325 px for an operation a user measured at 4.53.
// ---------------------------------------------------------------------------
#ifdef BINCV_WITH_OPENCV
BINCV_TEST(SubPix, MaskWidthMatchesOpenCV_OnAsymmetricContent) {
    constexpr int W = 70, H = 60;
    const int cxTrue = 32, cyTrue = 27;
    bincv::BinMat<uint32_t> img(W, H);
    asymmetricCorner(img, cxTrue, cyTrue);
    bincv::SignedQuantMat<1, uint32_t> dx(W, H), dy(W, H);
    bincv::derivativeX(img, dx);
    bincv::derivativeY(img, dy);

    cv::Mat bytes(H, W, CV_8U, cv::Scalar(0));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) bytes.at<uint8_t>(y, x) = img.at(y, x) ? 255 : 0;
    }

    std::vector<Point2f> mine;
    std::vector<cv::Point2f> theirs;
    for (int sy = -2; sy <= 2; ++sy) {
        for (int sx = -2; sx <= 2; ++sx) {
            mine.push_back(Point2f{static_cast<float>(cxTrue + sx),
                                   static_cast<float>(cyTrue + sy)});
            theirs.push_back(cv::Point2f(static_cast<float>(cxTrue + sx),
                                         static_cast<float>(cyTrue + sy)));
        }
    }
    bincv::SubPixParams p;
    p.winHalf = 5;
    bincv::cornerSubPix<1, uint32_t>(dx, dy, mine.data(), mine.size(), p);
    cv::cornerSubPix(bytes, theirs, cv::Size(p.winHalf, p.winHalf), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                      p.maxIterations, p.epsilon));

    double worst = 0.0, total = 0.0;
    for (size_t i = 0; i < mine.size(); ++i) {
        const double d = std::hypot(static_cast<double>(mine[i].x) - theirs[i].x,
                                    static_cast<double>(mine[i].y) - theirs[i].y);
        worst = d > worst ? d : worst;
        total += d;
    }
    const double mean = total / static_cast<double>(mine.size());
    std::printf("  ASYMMETRIC, binCV vs cv::cornerSubPix over %zu starts: mean %.4f px, "
                "worst %.4f px\n", mine.size(), mean, worst);
    BINCV_CHECK(mine.size() == 25);
    // The Tier 2 gap on content that shows it. Bound just above the measured 0.4713 --
    // and it is a CEILING on the deviation, so a regression that widened the gap fails
    // here even though nothing crashes.
    BINCV_CHECK(mean < 0.55);
    BINCV_CHECK(worst < 0.60);
}
#endif  // BINCV_WITH_OPENCV

BINCV_TEST_MAIN("test_subpix")
