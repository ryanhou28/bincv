// medianWide and edgeThreshold, each against its OWN scalar arm (issue #55).
//
// These are the two ops whose unmeasured state prompted the benchmark-at-birth
// rule -- 78% of the pipeline the day something called them -- and they still
// had no benchmark of their own: timed only as line items inside pipelines,
// with vector arms that were hard-wired on and therefore never proven against
// the scalar paths they replaced.
//
// THE RULE, WRITTEN FIRST: metrics are ms/frame at the reference size and the
// arm-on/arm-off ratio from ONE binary via the new runtime switches. A ratio
// near 1.00x is not a shrug -- it is the mis-attached-#define failure CLAUDE.md
// records (three "improvements" once measured against a compiled-out block),
// and it means the arm is not running where this file says it is.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/edge.hpp"
#include "bincv/ops/medianWide.hpp"
#include "measure_util.hpp"

namespace {
constexpr size_t kW = 752, kH = 480;
}

int main() {
    std::vector<uint8_t> src(kW * kH), med(kW * kH);
    uint64_t st = 0xC0FFEEULL;
    for (auto& v : src) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<uint8_t>(st >> 40);
    }
    bincv::BinMat<uint32_t> edges(kW, kH);

    std::printf("=== sensor ops, %zux%zu, vector arm vs its own scalar arm ===\n\n",
                kW, kH);

    const auto runPair = [&](const char* name, bool& toggle, auto&& fn) {
        std::vector<measure::Bench> bs = {
            {std::string(name) + ", vector arm", [&](int) { fn(); }},
            {std::string(name) + ", scalar arm", [&](int) {
                 toggle = false;
                 fn();
                 toggle = true;
             }}};
        const auto t = measure::measureInterleaved(bs, 7, 30.0);
        for (size_t i = 0; i < 2; ++i)
            std::printf(" %-32s %9.3f ms  spread %.0f%%\n", bs[i].name.c_str(),
                        t[i].medianNs / 1e6, t[i].spreadPct());
        std::printf("   arm buys %.2fx  (~1.00x would mean the arm is NOT running)\n\n",
                    t[1].medianNs / t[0].medianNs);
    };

    runPair("medianWide (reference L)", bincv::impl::medianSimdEnabled(), [&] {
        bincv::medianWide<3, uint8_t>(src.data(), kW, kH, kW, med.data(), kW,
                                      bincv::kMedianReferenceL);
        measure::g_sink += med[kW * (kH / 2) + kW / 2];
    });

    runPair("edgeThreshold (t=24)", bincv::impl::edgeSimdEnabled(), [&] {
        bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                             bincv::EdgeSpatial::Wide>(med.data(), kW, kH, kW,
                                                       edges.view(), uint8_t{24});
        measure::g_sink += edges.view().row(kH / 2)[0];
    });

    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
