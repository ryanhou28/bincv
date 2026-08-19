// T3.4 -- the pyramid's bit growth, its footprint, and the cost of the box sum.
//
// THIS IS NOT AN EXPERIMENT AND HAS NO DECISION RULE. T3.4's done-when clauses
// ask for two numbers that feed E-7 (T4.1) -- bit growth and peak footprint of a
// four-level pyramid at several NOut caps -- plus evidence for the cost claim the
// task's second blocking gap turns on. E-7 is the entry that will WEIGH these
// against tracking accuracy, and it is deliberately deferred: parameterizing the
// cap is what buys the right to defer measuring it (ARCHITECTURE 9).
//
// NO OPENCV, so this builds and runs in the reference device's DEFAULT core-only
// configuration. Three things are measured here and all three are binCV against
// binCV or binCV against arithmetic:
//
//   1. BIT GROWTH. How many distinct values each level of a four-level pyramid
//      actually holds, and therefore how many bits it needs, for several ladders.
//      The uncapped ladder (1 -> 3 -> 5 -> 7, i.e. NOut = NIn + 2 at every step)
//      is the one that answers "how much precision does the box mean create";
//      the capped ones say what is lost by refusing it.
//   2. PEAK FOOTPRINT. Total bytes of the coexisting levels, against the CV_8U
//      byte-per-pixel pyramid a user has today. That denominator is exact
//      arithmetic -- one byte per pixel per level -- and needs no OpenCV to
//      compute. It is a PEAK: the levels coexist because a tracker reads all of
//      them (CLAUDE.md, benchmarking).
//   3. THE COST OF THE BOX SUM, the shipped formulation against the one T3.4
//      rejected, at NIn = 1, 2, 3 and 4. Both are the same kernel with the same
//      requantizer and differ only in how the four NIn-bit operands are summed:
//      3*NIn + 1 full-adder stages against 4*(2^NIn - 1) single-bit accumulate
//      steps. The point is the SHAPE of the two curves, not the ratio at any one
//      NIn -- at NIn = 1 they are the same four inputs and should be close.
//
// VALIDITY (EXPERIMENTS.md "Verify the benchmark measures something"):
//   * measure::g_sink consumes a destination word from every timed call;
//   * four distinct random sources rotate, so nothing constant-folds;
//   * the two routes are checked to agree pixel for pixel on every case BEFORE
//     either is timed -- a benchmark between a right answer and a wrong one is
//     not a measurement;
//   * the reported spread bounds within-run noise; a difference smaller than it
//     is a null result.
//
// On x86_64 the TIMING half is indicative only (EXPERIMENTS.md, "Measurement
// platforms"); the growth and footprint halves are architecture-independent and
// close anywhere. The authoritative timing run is
//
//   ./scripts/run_on_pi.sh pi4 './benchmark/pyramid_benchmark'

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "bincv-cpp/ops/pyramid.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "measure_util.hpp"

namespace {

using bincv::pyrDownHeight;
using bincv::pyrDownWidth;
using bincv::QuantMat;

using Word = uint32_t;  // D-14

constexpr int kWidth = 640;
constexpr int kHeight = 480;

template <size_t N>
void fillRandom(QuantMat<N, Word>& m, uint64_t seed) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            m.set(y, x, static_cast<unsigned>(measure::nextRandom(state)) &
                            QuantMat<N, Word>::MaxValue);
        }
    }
}

template <size_t N>
size_t distinctValues(const QuantMat<N, Word>& m) {
    std::set<unsigned> seen;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) seen.insert(m.at(y, x));
    }
    return seen.size();
}

/// The REACHABLE alphabet of a ladder, by arithmetic rather than by sampling.
///
/// This is the number that says how many bits a level NEEDS; the distinct count
/// of an actual frame says how many it happened to CONTAIN, which is a sample
/// statistic and falls with the frame size. ARCHITECTURE 7.2's 1/3/4/5 is the
/// second kind of number (X-2 counted a 256x256 frame), and the two disagree --
/// see the note under the table.
///
/// The requantized value depends on the SUM alone, so the reachable set at the
/// next level is the image of the four-fold sumset, which is cheap to build with
/// a bitset over [0, 4 * (2^NIn - 1)].
std::vector<unsigned> reachableNext(const std::vector<unsigned>& alphabet, size_t nIn,
                                    size_t nOut) {
    const unsigned maxIn = (1u << nIn) - 1u;
    const unsigned maxOut = (1u << nOut) - 1u;

    std::vector<bool> sums(4 * static_cast<size_t>(maxIn) + 1, false);
    sums[0] = true;
    for (int add = 0; add < 4; ++add) {
        std::vector<bool> next(sums.size(), false);
        for (size_t s = 0; s < sums.size(); ++s) {
            if (!sums[s]) continue;
            for (unsigned v : alphabet) {
                if (s + v < next.size()) next[s + v] = true;
            }
        }
        sums.swap(next);
    }

    std::set<unsigned> values;
    for (size_t s = 0; s < sums.size(); ++s) {
        if (sums[s]) {
            values.insert((static_cast<unsigned>(s) * maxOut + 2u * maxIn) / (4u * maxIn));
        }
    }
    return std::vector<unsigned>(values.begin(), values.end());
}

size_t bitsFor(size_t distinct) {
    size_t bits = 0;
    while ((size_t{1} << bits) < distinct) ++bits;
    return bits == 0 ? 1 : bits;
}

/// The CV_8U pyramid a user has today: one byte per pixel, four levels, all
/// resident. Exact arithmetic, so no OpenCV is needed to state the denominator.
size_t referenceBytes() {
    size_t total = 0;
    size_t w = static_cast<size_t>(kWidth);
    size_t h = static_cast<size_t>(kHeight);
    for (int level = 0; level < 4; ++level) {
        total += w * h;
        w = pyrDownWidth(w);
        h = pyrDownHeight(h);
    }
    return total;
}

// ---------------------------------------------------------------------------
// 1 and 2 -- bit growth and footprint, one ladder at a time
// ---------------------------------------------------------------------------

std::string four(size_t a, size_t b, size_t c, size_t d) {
    return std::to_string(a) + "/" + std::to_string(b) + "/" + std::to_string(c) + "/" +
           std::to_string(d);
}

template <size_t N1, size_t N2, size_t N3>
void reportLadder(const char* label) {
    bincv::Pyramid<Word, 1, N1, N2, N3> pyramid(kWidth, kHeight);
    fillRandom(pyramid.template level<0>(), 0xB17u + N1 * 101 + N2 * 11 + N3);
    pyramid.build();

    const size_t d0 = distinctValues(pyramid.template level<0>());
    const size_t d1 = distinctValues(pyramid.template level<1>());
    const size_t d2 = distinctValues(pyramid.template level<2>());
    const size_t d3 = distinctValues(pyramid.template level<3>());

    const std::vector<unsigned> a0 = {0u, 1u};
    const std::vector<unsigned> a1 = reachableNext(a0, 1, N1);
    const std::vector<unsigned> a2 = reachableNext(a1, N1, N2);
    const std::vector<unsigned> a3 = reachableNext(a2, N2, N3);

    const size_t bytes = pyramid.sizeInBytes();
    const size_t reference = referenceBytes();

    std::printf("  %-22s  %-9s  %-14s  %-14s  %-11s  %8zu  %6.2fx\n", label,
                four(1, N1, N2, N3).c_str(), four(d0, d1, d2, d3).c_str(),
                four(a0.size(), a1.size(), a2.size(), a3.size()).c_str(),
                four(bitsFor(a0.size()), bitsFor(a1.size()), bitsFor(a2.size()),
                     bitsFor(a3.size()))
                    .c_str(),
                bytes, static_cast<double>(reference) / static_cast<double>(bytes));
}

void reportGrowthAndFootprint() {
    std::printf("\nBIT GROWTH AND PEAK FOOTPRINT -- four levels from %dx%d, uint32_t\n",
                kWidth, kHeight);
    std::printf("  the CV_8U denominator is %zu bytes (one byte per pixel, four levels,\n"
                "  all resident)\n\n",
                referenceBytes());
    std::printf("  %-22s  %-9s  %-14s  %-14s  %-11s  %8s  %7s\n", "ladder (NOut caps)",
                "declared", "in the frame", "reachable", "bits needed", "bytes", "vs 8U");
    std::printf("  %-22s  %-9s  %-14s  %-14s  %-11s  %8s  %7s\n", "----------------------",
                "---------", "--------------", "--------------", "-----------", "--------",
                "-------");

    // Uncapped: NOut = NIn + 2 at every step, so nothing the box mean produces is
    // ever thrown away. This row is the answer to "how much precision does a 2x2
    // box actually create", and every other row is a refusal of some of it.
    reportLadder<3, 5, 7>("1-3-5-7  uncapped");
    // The ladder ARCHITECTURE 7.2 measured on the reference pipeline.
    reportLadder<3, 4, 5>("1-3-4-5  reference");
    // Progressively harder caps -- E-7's candidates.
    reportLadder<3, 3, 3>("1-3-3-3");
    reportLadder<2, 2, 2>("1-2-2-2");
    reportLadder<1, 1, 1>("1-1-1-1  re-binarized");
    reportLadder<8, 8, 8>("1-8-8-8  CV_8U-like");

    std::printf("\n  \"in the frame\" counts DISTINCT VALUES PRESENT in one %dx%d-derived\n"
                "  level; \"reachable\" is the alphabet the arithmetic can produce at all, and\n"
                "  \"bits needed\" follows the reachable column. The two differ, and the\n"
                "  difference matters: EXPERIMENTS.md X-2 reported 2/5/15/26 for the CV_8U\n"
                "  ladder, which is what a 256x256 frame CONTAINED -- its level 3 is 32x32,\n"
                "  i.e. 1024 pixels drawn from an alphabet of 65. A frame statistic falls\n"
                "  with the frame size; the reachable alphabet does not.\n",
                kWidth, kHeight);
}

// ---------------------------------------------------------------------------
// 3 -- the cost of the box sum
// ---------------------------------------------------------------------------

constexpr int kInputs = 4;

template <size_t NIn, size_t NOut>
void addCostBenches(std::vector<measure::Bench>& benches,
                    std::vector<std::vector<QuantMat<NIn, Word>>>& sources,
                    std::vector<QuantMat<NOut, Word>>& destinations, size_t& pixels) {
    sources.emplace_back();
    for (int i = 0; i < kInputs; ++i) {
        sources.back().emplace_back(kWidth, kHeight);
        fillRandom(sources.back().back(), UINT64_C(0xC057) + NIn * 977 + static_cast<uint64_t>(i));
    }
    destinations.emplace_back(static_cast<int>(pyrDownWidth(kWidth)),
                              static_cast<int>(pyrDownHeight(kHeight)));
    destinations.emplace_back(static_cast<int>(pyrDownWidth(kWidth)),
                              static_cast<int>(pyrDownHeight(kHeight)));

    auto& src = sources.back();
    QuantMat<NOut, Word>* direct = &destinations[destinations.size() - 2];
    QuantMat<NOut, Word>* replicated = &destinations[destinations.size() - 1];
    pixels = destinations.back().getWidth() * destinations.back().getHeight();

    // Agreement before timing -- validity hazard 4.
    for (int i = 0; i < kInputs; ++i) {
        bincv::pyrDown<NOut, NIn, Word>(src[static_cast<size_t>(i)], *direct);
        bincv::impl::pyrDownReplicated<NOut, NIn, Word>(src[static_cast<size_t>(i)],
                                                        *replicated);
        for (int y = 0; y < direct->rows(); ++y) {
            for (int x = 0; x < direct->cols(); ++x) {
                if (direct->at(y, x) != replicated->at(y, x)) {
                    std::printf("  FATAL: the two box-sum routes disagree at NIn=%zu (%d,%d)\n",
                                NIn, y, x);
                    std::fflush(stdout);
                    std::abort();
                }
            }
        }
    }

    benches.push_back({"NIn=" + std::to_string(NIn) + " linear adder",
                       [&src, direct](int i) {
                           bincv::pyrDown<NOut, NIn, Word>(
                               src[static_cast<size_t>(i) % kInputs], *direct);
                           measure::g_sink += direct->data()[0];
                       }});
    benches.push_back({"NIn=" + std::to_string(NIn) + " replicated  ",
                       [&src, replicated](int i) {
                           bincv::impl::pyrDownReplicated<NOut, NIn, Word>(
                               src[static_cast<size_t>(i) % kInputs], *replicated);
                           measure::g_sink += replicated->data()[0];
                       }});
}

void reportCost() {
    std::printf("\nTHE 2x2 BOX SUM: LINEAR AGAINST EXPONENTIAL -- %dx%d -> %dx%d, uint32_t\n",
                kWidth, kHeight, kWidth / 2, kHeight / 2);
    std::printf("  NOut = NIn + 1 throughout, so the requantizer is the same shape in every\n"
                "  row and the difference between the pair is the SUM alone.\n\n");

    std::vector<measure::Bench> benches;
    std::vector<std::vector<QuantMat<1, Word>>> src1;
    std::vector<std::vector<QuantMat<2, Word>>> src2;
    std::vector<std::vector<QuantMat<3, Word>>> src3;
    std::vector<std::vector<QuantMat<4, Word>>> src4;
    std::vector<QuantMat<2, Word>> dst2;
    std::vector<QuantMat<3, Word>> dst3;
    std::vector<QuantMat<4, Word>> dst4;
    std::vector<QuantMat<5, Word>> dst5;
    size_t pixels = 0;

    addCostBenches<1, 2>(benches, src1, dst2, pixels);
    addCostBenches<2, 3>(benches, src2, dst3, pixels);
    addCostBenches<3, 4>(benches, src3, dst4, pixels);
    addCostBenches<4, 5>(benches, src4, dst5, pixels);

    const std::vector<measure::Timing> timings = measure::measureInterleaved(benches, 7, 60.0);

    std::printf("  %-24s  %10s  %10s  %8s  %8s  %10s\n", "route", "ns/dst px", "spread",
                "stages", "inputs", "vs linear");
    std::printf("  %-24s  %10s  %10s  %8s  %8s  %10s\n", "------------------------",
                "----------", "----------", "--------", "--------", "----------");
    for (size_t i = 0; i < benches.size(); i += 2) {
        const size_t nIn = i / 2 + 1;
        const double linear =
            timings[i].medianNs / static_cast<double>(pixels);
        const double replicated =
            timings[i + 1].medianNs / static_cast<double>(pixels);
        std::printf("  %-24s  %10.4f  %9.1f%%  %8zu  %8s  %10s\n", benches[i].name.c_str(),
                    linear, timings[i].spreadPct(), bincv::impl::boxSumFullAdders(nIn), "-", "-");
        std::printf("  %-24s  %10.4f  %9.1f%%  %8s  %8zu  %9.2fx\n", benches[i + 1].name.c_str(),
                    replicated, timings[i + 1].spreadPct(), "-",
                    bincv::impl::boxSum4ReplicatedInputs(nIn), replicated / linear);
    }
    std::printf("\n  The stage counts are impl::boxSumFullAdders(NIn) = 3*NIn + 1 and\n"
                "  impl::boxSum4ReplicatedInputs(NIn) = 4*(2^NIn - 1). At NIn = 8 they are 25\n"
                "  and 1020; the replicated arm refuses to compile above NIn = 5, where its\n"
                "  input array is already 124 words per destination word.\n");
}

}  // namespace

int main() {
    std::printf("binCV pyramid downsample -- T3.4\n");
    std::printf("================================\n");
    reportGrowthAndFootprint();
    reportCost();
    std::printf("\nsink = %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
