// medianWide and edgeThreshold, each against its OWN scalar arm.
//
// These are the two ops whose unmeasured state prompted the benchmark-at-birth
// rule -- 78% of the pipeline the day something called them -- and they still
// had no benchmark of their own: timed only as line items inside pipelines,
// with vector arms that were hard-wired on and therefore never proven against
// the scalar paths they replaced.
//
// THE RULE, WRITTEN FIRST: metrics are ms/frame at the reference size and the
// arm-on/arm-off ratio from ONE binary via the new runtime switches. A ratio
// near 1.00x is not a shrug -- it is the mis-attached-#define failure this project
// has already paid for once (three "improvements" measured against a compiled-out
// block),
// and it means the arm is not running where this file says it is.
//
// THE THIRD ROW is the same stage two ways: the two whole-frame calls, and
// MedianEdgeStream fed one row at a time, as a camera delivers them. Both are
// bit-identical (tests/test_median_edge_stream.cpp), so this measures only what the
// shape costs or saves. It prints the bytes each holds beside the time, because the
// stream exists for the memory and its speed is the price -- which the two numbers
// together state and neither does alone.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/edge.hpp"
#include "bincv/ops/medianEdgeStream.hpp"
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

    {
        using Stream = bincv::MedianEdgeStream<3, uint8_t>;
        std::vector<uint8_t> scratch(Stream::scratchElements(kW, bincv::kMedianReferenceL));
        bincv::BinMat<uint32_t> streamed(kW, kH);
        Stream stream(scratch.data(), scratch.size(), bincv::kMedianReferenceL, streamed.view(),
                      uint8_t{17});
        std::vector<measure::Bench> bs = {
            {"sensor stage, two whole-frame calls", [&](int) {
                 bincv::medianWide<3, uint8_t>(src.data(), kW, kH, kW, med.data(), kW,
                                               bincv::kMedianReferenceL);
                 bincv::edgeThreshold(med.data(), kW, kH, kW, edges.view(), uint8_t{17});
                 measure::g_sink += edges.view().row(kH / 2)[0];
             }},
            {"sensor stage, streamed row by row", [&](int) {
                 stream.reset();
                 for (size_t y = 0; y < kH; ++y) stream.push(src.data() + y * kW, 1, kW);
                 measure::g_sink += streamed.view().row(kH / 2)[0];
             }}};
        const auto t = measure::measureInterleaved(bs, 7, 30.0);
        for (size_t i = 0; i < 2; ++i)
            std::printf(" %-36s %9.3f ms  spread %.0f%%\n", bs[i].name.c_str(),
                        t[i].medianNs / 1e6, t[i].spreadPct());
        std::printf("   stream costs %.2fx the time of the two calls\n",
                    t[1].medianNs / t[0].medianNs);
        std::printf("   held besides the output: whole-frame %zu B (8-bit frame + 8-bit median),"
                    " stream %zu B\n\n",
                    2 * kW * kH, scratch.size());
    }

    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
