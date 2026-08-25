// ===========================================================================
// X-24 / E-7 -- THE SPEED AXIS OF THE PYRAMID-DEPTH CHOICE.
//
// X-24's accuracy and footprint axes are exact and device-independent and closed
// on the development machine. This is the third axis, and it is the reference
// device's alone: X-22's caveat 1 measured the SAME kernel moving 1.46x between
// two binaries built from unchanged source, so a ladder chosen on a laptop timing
// would not survive contact with the device it has to run on.
//
// THE COST MODEL, WRITTEN OUT BEFORE MEASURING -- it is X-24's hypothesis 3,
// restated here as numbers so the measurement can contradict it.
//
// The two stages weight the levels OPPOSITELY, and that is the whole point:
//
//   * BUILD (pyrDown down the ladder, then a derivative per level) is PER PIXEL.
//     Level l has 1/4^l of level 0's pixels, so deepening the COARSE levels is
//     nearly free. Against a 1/1/1/1 ladder the per-pixel work grows roughly as
//     sum_l (pixels_l * cost(N_l)) / sum_l pixels_l, and with 4/3 of the pixels in
//     level 0 the coarse levels can barely move it.
//
//   * TRACK is PER POINT PER WINDOW, and EVERY LEVEL TRACKS THE SAME POINTS
//     THROUGH THE SAME 31x31 WINDOW. residualSums issues 20*N^2 popcounts per
//     word and gradientCovariance 3N^2+N, so a level's tracking cost is paid IN
//     FULL however few pixels it has. The prediction is therefore
//
//         relative track cost  ~  sum_l N_l^2 / sum_l 1^2
//
//     ladder     sum N_l^2    predicted
//     1/1/1/1        4          1.00x
//     1/2/2/2       13          3.25x
//     1/3/3/3       28          7.00x
//     1/3/4/4       42         10.50x
//     1/3/5/5       60         15.00x
//     1/3/5/7       84         21.00x
//
// So the binding constraint on E-7 is predicted to be TRACKER TIME, not
// footprint, and 1/2/2/2 -- X-24's accuracy leader -- is predicted to cost 3.25x
// the shipped ladder's tracking time for 1.16x its bytes. IF FOOTPRINT BINDS
// FIRST, OR IF THE TRACK RATIOS COME IN FLAT, THE MODEL ABOVE IS WRONG AND THE
// ENTRY MUST SAY SO RATHER THAN RE-FITTING.
//
// THE 94x60 BLOCK'S TRACK COLUMN IS EXPECTED TO BE FLAT, AND THAT IS NOT A
// REFUTATION -- it is deviation (vi) doing its job. At 94x60 the next level down
// is 47x30, whose height is under the 31-pixel window, so usableLevelCount stops
// at ONE and the tracker never touches a level deeper than level 0 -- which is
// 1 bit in every ladder. The informative column there is BUILD, where the
// per-row prologue X-21 flagged is paid 5.4x more often per pixel than at
// 640x480. The track model is only testable where the coarse levels are actually
// used, i.e. in the 640x480 block.
//
// Interleaved round-robin (measure_util.hpp) rather than one process per ladder:
// the variants are compared against each other within one run, which is what that
// harness exists to make safe.
// ===========================================================================

#include <cstdio>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <initializer_list>

#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"
#include "measure_util.hpp"

using bincv::Point2f;
using W = uint32_t;

namespace {

/// One SignedQuantMat per level, at that level's depth and extent.
template <typename WordType, size_t... LevelBits>
struct DerivLadder;

template <typename WordType, size_t N0>
struct DerivLadder<WordType, N0> {
    bincv::SignedQuantMat<N0, WordType> mat;
    DerivLadder(int w, int h) : mat(w, h) {}
    template <size_t I>
    bincv::SignedQuantMat<N0, WordType>& get() {
        static_assert(I == 0, "index out of range");
        return mat;
    }
    size_t bytes() const { return mat.sizeInWords() * sizeof(WordType); }
};

template <typename WordType, size_t N0, size_t N1, size_t... Rest>
struct DerivLadder<WordType, N0, N1, Rest...> {
    bincv::SignedQuantMat<N0, WordType> mat;
    DerivLadder<WordType, N1, Rest...> rest;
    DerivLadder(int w, int h)
        : mat(w, h),
          rest(static_cast<int>(bincv::pyrDownWidth(static_cast<size_t>(w))),
               static_cast<int>(bincv::pyrDownHeight(static_cast<size_t>(h)))) {}
    template <size_t I>
    auto& get() {
        if constexpr (I == 0) {
            return mat;
        } else {
            return rest.template get<I - 1>();
        }
    }
    size_t bytes() const { return mat.sizeInWords() * sizeof(WordType) + rest.bytes(); }
};

template <typename WordType, size_t... LevelBits>
struct Ladder {
    static constexpr size_t Levels = sizeof...(LevelBits);
    using Pyr = bincv::Pyramid<WordType, LevelBits...>;

    Pyr prev, next;
    DerivLadder<WordType, LevelBits...> dx, dy;
    bincv::LKLevels<WordType, LevelBits...> levels;

    Ladder(int w, int h) : prev(w, h), next(w, h), dx(w, h), dy(w, h) {}

    /// The measured BUILD stage: pyrDown both ladders, derivative every level.
    void buildStage() {
        prev.template build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
        next.template build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
        deriv<0>();
    }
    template <size_t I>
    void deriv() {
        if constexpr (I < Levels) {
            bincv::derivativeX(prev.template level<I>(), dx.template get<I>());
            bincv::derivativeY(prev.template level<I>(), dy.template get<I>());
            deriv<I + 1>();
        }
    }
    template <size_t I>
    void bind() {
        if constexpr (I < Levels) {
            levels.template get<I>() = bincv::lkLevel<Pyr::template levelBits<I>()>(
                prev.template level<I>(), next.template level<I>(), dx.template get<I>(),
                dy.template get<I>());
            bind<I + 1>();
        }
    }
    size_t bytes() const {
        return prev.sizeInBytes() + next.sizeInBytes() + dx.bytes() + dy.bytes();
    }
};

/// An edge-map-like level 0: ~10% set, one-pixel-wide structure, which is what the
/// reference binarization produces. Deterministic, so every ladder sees the same
/// content (validity: the arms must differ in DEPTH and nothing else).
template <typename WordType, size_t... LevelBits>
void seed(Ladder<WordType, LevelBits...>& lad, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const unsigned a = ((x * 7 + y * 13) % 29 == 0 || (x + y) % 37 == 0) ? 1u : 0u;
            const unsigned b = (((x - 1) * 7 + y * 13) % 29 == 0 || (x - 1 + y) % 37 == 0) ? 1u : 0u;
            lad.prev.template level<0>().set(y, x, a);
            lad.next.template level<0>().set(y, x, b);
        }
    }
}

std::vector<Point2f> gridPoints(int w, int h, int step, int margin) {
    std::vector<Point2f> pts;
    for (int y = margin; y < h - margin; y += step) {
        for (int x = margin; x < w - margin; x += step) {
            pts.push_back(Point2f{static_cast<float>(x), static_cast<float>(y)});
        }
    }
    return pts;
}

struct Arm {
    std::string name;
    double sumNsq;   ///< the model's predictor
    size_t bytes;
    std::function<void(int)> buildBody;
    std::function<void(int)> trackBody;
};

/// Holds one ladder alive and hands back its two timed bodies.
template <size_t... LevelBits>
struct ArmHolder {
    Ladder<W, LevelBits...> lad;
    std::vector<Point2f> pts;
    std::vector<Point2f> out;
    std::vector<uint8_t> status;
    bincv::LKParams params;

    ArmHolder(int w, int h, const std::vector<Point2f>& p)
        : lad(w, h), pts(p), out(p.size()), status(p.size()) {
        seed(lad, w, h);
        lad.buildStage();
        lad.template bind<0>();
    }
};

double sumOfSquares(std::initializer_list<size_t> bits) {
    double s = 0.0;
    for (size_t b : bits) s += static_cast<double>(b * b);
    return s;
}

template <size_t... LevelBits>
void addArm(std::vector<Arm>& arms, std::vector<std::shared_ptr<void>>& keep, const char* name,
            double sumNsq, int w, int h, const std::vector<Point2f>& pts) {
    auto holder = std::make_shared<ArmHolder<LevelBits...>>(w, h, pts);
    keep.push_back(holder);
    Arm a;
    a.name = name;
    a.sumNsq = sumNsq;
    a.bytes = holder->lad.bytes();
    a.buildBody = [holder](int) { holder->lad.buildStage(); };
    a.trackBody = [holder](int) {
        bincv::calcOpticalFlowPyrLK(holder->lad.levels, holder->pts.data(), holder->out.data(),
                                    holder->status.data(), nullptr, holder->pts.size(),
                                    holder->params);
    };
    arms.push_back(std::move(a));
    return;
}

void runAt(int w, int h, int step, int margin, int repeats, double targetMs) {
    const std::vector<Point2f> pts = gridPoints(w, h, step, margin);
    std::vector<Arm> arms;
    std::vector<std::shared_ptr<void>> keep;

    addArm<1, 1, 1, 1>(arms, keep, "1/1/1/1", sumOfSquares({1, 1, 1, 1}), w, h, pts);
    addArm<1, 2, 2, 2>(arms, keep, "1/2/2/2", sumOfSquares({1, 2, 2, 2}), w, h, pts);
    addArm<1, 3, 3, 3>(arms, keep, "1/3/3/3", sumOfSquares({1, 3, 3, 3}), w, h, pts);
    addArm<1, 3, 4, 4>(arms, keep, "1/3/4/4", sumOfSquares({1, 3, 4, 4}), w, h, pts);
    addArm<1, 3, 5, 5>(arms, keep, "1/3/5/5", sumOfSquares({1, 3, 5, 5}), w, h, pts);
    addArm<1, 3, 5, 7>(arms, keep, "1/3/5/7", sumOfSquares({1, 3, 5, 7}), w, h, pts);

    std::vector<measure::Bench> buildBenches, trackBenches;
    for (const Arm& a : arms) {
        buildBenches.push_back({a.name, a.buildBody});
        trackBenches.push_back({a.name, a.trackBody});
    }
    const auto bt = measure::measureInterleaved(buildBenches, repeats, targetMs);
    const auto tt = measure::measureInterleaved(trackBenches, repeats, targetMs);

    std::printf("\n  level 0 = %dx%d, %zu keypoints, 31x31 window, 4 levels\n", w, h, pts.size());
    std::printf("  %-9s %6s | %11s %7s | %11s %7s %9s | %9s\n", "ladder", "sumN^2", "build us",
                "vs 1bit", "track us", "vs 1bit", "predicted", "bytes");
    std::printf("  ---------------------------------------------------------------"
                "---------------------\n");
    const double buildBase = bt[0].medianNs;
    const double trackBase = tt[0].medianNs;
    for (size_t i = 0; i < arms.size(); ++i) {
        std::printf("  %-9s %6.0f | %11.1f %6.2fx | %11.1f %6.2fx %8.2fx | %9zu\n",
                    arms[i].name.c_str(), arms[i].sumNsq, bt[i].medianNs / 1000.0,
                    bt[i].medianNs / buildBase, tt[i].medianNs / 1000.0,
                    tt[i].medianNs / trackBase, arms[i].sumNsq / arms[0].sumNsq, arms[i].bytes);
    }
    std::printf("  spread (max-min)/median: build");
    for (const auto& t : bt) std::printf(" %.0f%%", t.spreadPct());
    std::printf(" | track");
    for (const auto& t : tt) std::printf(" %.0f%%", t.spreadPct());
    std::printf("\n");
}

} // namespace

int main() {
    std::printf("=== X-24 / E-7: pyramid depth, the SPEED axis ===\n");
    std::printf("Cost model written before measuring: build is per-pixel and should\n"
                "barely move; track should scale as sum_l N_l^2 (1.00 / 3.25 / 7.00 /\n"
                "10.50 / 13.00 / 21.00). A flat track column refutes it.\n");
    runAt(640, 480, 40, 40, 7, 60.0);
    runAt(94, 60, 8, 12, 7, 60.0);
    return 0;
}
