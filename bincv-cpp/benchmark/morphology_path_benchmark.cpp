// -- what is the 3x3 SPECIAL CASE worth? binCV against binCV.
//
// says "Special-case 3x3 -- it is the common case in practice", and
// ops/morphology.hpp does: morphRow3x3 is a second row kernel that runs when the
// element is 3x3 and centred. A second implementation of one function is a
// maintenance cost forever, and this file is what says what it buys, so that a
// Phase 5 reader deciding whether to vectorise one path or both has a number
// rather than an assertion.
//
// IT IS ALSO A CORRECTION. The special case's docstring used to justify itself by
// load count -- "one extendedRowWord per word per element row where the general
// path pays two per SET CELL". That was wrong: morphRowGeneric's window branch
// hoists exactly the same call out of its cell loop for any element whose row
// reaches less than a word sideways, which every 3x3 element does. What the
// special case actually removes is the per-cell loop itself, the data-dependent
// shift count, and the span queries -- and none of that had ever been measured.
//
// WHY IT IS A SEPARATE BINARY FROM morphology_benchmark.cpp, which is where this
// row naturally belongs: measured, adding a MorphPath::Generic call site to that
// translation unit moved its headline erode 3x3 row by ~10% (0.143-0.159 against
// 0.126-0.129 ns/pixel at 640x480 on x86, identical header). Two instantiations
// of morphApply in one object file change each other's code layout. The number
// that gets published against OpenCV must not depend on what else is in the file
// that measures it, so the two comparisons live in two binaries.
//
// NO OPENCV. Both sides are binCV, so the design notes's denominator does not
// apply and this builds in the reference device's DEFAULT core-only build.
//
// VARIANTS impl::morphApply with MorphPath::Auto (what erode/dilate call) and
// with MorphPath::Generic (the general row kernel, special case off).
// They are required to compute the SAME image before either is timed
// -- the same property tests/test_morphology.cpp's
// Morphology.FastPathEqualsGeneric_* asserts across the whole sweep.
// WORKLOAD erode and dilate, rect3x3 / cross3x3 (== ellipse 3x3, the design rule’s note),
// 640x480 and the pyramid ladder below it, ~50% fill, four rotated
// inputs, at uint32_t (the design rule’s default) and uint64_t.
// METRIC ns/pixel for both paths and the ratio, with the batch spread beside
// it so a difference smaller than the noise reads as one.
//
// On x86_64 this is INDICATIVE ONLY (EXPERIMENTS.md, "Measurement platforms").
// The authoritative run is
//
//./scripts/run_on_pi.sh pi4 './benchmark/morphology_path_benchmark'

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/morphology.hpp"
#include "measure_util.hpp"

namespace {

constexpr int kInputs = 4;
constexpr int kRepeats = 7;
constexpr double kTargetMs = 40.0;

template <typename WordType>
void fill(bincv::BinMat<WordType>& m, uint64_t seed) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            if ((measure::nextRandom(state) >> 63) != 0) m.set(y, x, true);
        }
    }
}

template <typename WordType>
int disagreements(const bincv::BinMat<WordType>& a, const bincv::BinMat<WordType>& b) {
    int n = 0;
    for (int y = 0; y < a.rows(); ++y) {
        for (int x = 0; x < a.cols(); ++x) {
            if (a.at(y, x) != b.at(y, x)) ++n;
        }
    }
    return n;
}

struct Shape {
    const char* name;
    bincv::StructuringElement se;
};

/// @brief One (word type, element, fold) at one size.
/// @return false if the two paths disagree, which is a defect and not a timing.
template <bool IsErode, typename WordType>
bool runOne(const char* wordName, const Shape& shape, int width, int height) {
    using namespace bincv;

    std::vector<BinMat<WordType>> inputs;
    inputs.reserve(static_cast<size_t>(kInputs));
    for (int i = 0; i < kInputs; ++i) {
        inputs.emplace_back(width, height);
        fill(inputs.back(), UINT64_C(0x3033) + static_cast<uint64_t>(i) * UINT64_C(7919));
    }
    BinMat<WordType> autoDst(width, height);
    BinMat<WordType> genericDst(width, height);

    // Validity: the two paths must compute one image before either is timed.
    impl::morphApply<IsErode, impl::MorphPath::Auto>(inputs[0].constView(), autoDst.view(),
                                                     shape.se, BORDER_CONSTANT, IsErode);
    impl::morphApply<IsErode, impl::MorphPath::Generic>(inputs[0].constView(), genericDst.view(),
                                                        shape.se, BORDER_CONSTANT, IsErode);
    const int differing = disagreements(autoDst, genericDst);
    if (differing != 0) {
        std::printf(" %-8s %-9s %-6s THE TWO PATHS DISAGREE on %d pixels -- not timed.\n",
                    wordName, shape.name, IsErode ? "erode" : "dilate", differing);
        return false;
    }

    std::vector<measure::Bench> benches;
    benches.push_back({"auto", [&](int i) {
                           impl::morphApply<IsErode, impl::MorphPath::Auto>(
                               inputs[static_cast<size_t>(i % kInputs)].constView(),
                               autoDst.view(), shape.se, BORDER_CONSTANT, IsErode);
                           measure::g_sink += autoDst.ptr(0)[0];
                       }});
    benches.push_back({"generic", [&](int i) {
                           impl::morphApply<IsErode, impl::MorphPath::Generic>(
                               inputs[static_cast<size_t>(i % kInputs)].constView(),
                               genericDst.view(), shape.se, BORDER_CONSTANT, IsErode);
                           measure::g_sink += genericDst.ptr(0)[0];
                       }});

    const std::vector<measure::Timing> t = measure::measureInterleaved(benches, kRepeats,
                                                                      kTargetMs);
    const double pixels = static_cast<double>(width) * static_cast<double>(height);
    const double a = t[0].medianNs / pixels;
    const double g = t[1].medianNs / pixels;
    std::printf(" %-8s %-9s %-6s %8.5f %8.5f %8.2fx (spread %.1f%% / %.1f%%)\n", wordName,
                shape.name, IsErode ? "erode" : "dilate", a, g, (a > 0.0) ? g / a : 0.0,
                t[0].spreadPct(), t[1].spreadPct());
    return true;
}

bool runSize(int width, int height) {
    const Shape shapes[] = {{"rect3x3", bincv::rect3x3()}, {"cross3x3", bincv::cross3x3()}};
    std::printf("\n================ %d x %d ================\n", width, height);
    std::printf(" %-8s %-9s %-6s %8s %8s %9s\n", "word", "element", "fold", "auto", "generic",
                "generic/auto");
    std::printf(" ------------------------------------------------------------------------\n");
    bool ok = true;
    for (const Shape& s : shapes) {
        if (!runOne<true, uint32_t>("uint32", s, width, height)) ok = false;
        if (!runOne<false, uint32_t>("uint32", s, width, height)) ok = false;
        if (!runOne<true, uint64_t>("uint64", s, width, height)) ok = false;
        if (!runOne<false, uint64_t>("uint64", s, width, height)) ok = false;
    }
    return ok;
}

}  // namespace

int main() {
    std::printf(" -- the 3x3 special case priced against the general row kernel\n");
    std::printf("================================================================================\n\n");
    std::printf("Both columns are ops/morphology.hpp. `auto` is what bincv::erode and\n");
    std::printf("bincv::dilate call; `generic` is impl::MorphPath::Generic, the same kernel\n");
    std::printf("with the 3x3 special case refused. A ratio > 1 is what the special case saves.\n");
    std::printf("The two are required to compute the same image before either is timed.\n");

    const int sizes[][2] = {{640, 480}, {320, 240}, {160, 120}, {94, 60}};
    bool ok = true;
    for (const auto& size : sizes) {
        if (!runSize(size[0], size[1])) ok = false;
    }

    std::printf("\n sink=%llu\n", static_cast<unsigned long long>(measure::g_sink));
    if (!ok) {
        std::printf("\nAT LEAST ONE PAIR DISAGREED -- see above.\n");
        return 1;
    }
    return 0;
}
