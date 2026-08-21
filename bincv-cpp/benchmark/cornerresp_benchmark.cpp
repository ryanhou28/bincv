// ===========================================================================
// X-31 -- the corner response as bit-sliced box sums.
//
// EQUALITY IS CHECKED BEFORE ANY TIMING IS PRINTED, and a failure aborts the
// run. X-31's rule makes bit-exactness a PRECONDITION, not a band: box sums of
// bits are exact integers, minEigenValue takes the same integers, so the float is
// identical by construction. A tolerance here would be admitting the argument is
// wrong.
// ===========================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "cornerresp_arms.hpp"
#include "measure_util.hpp"

using W = uint32_t;

namespace {

void fillSynthetic(bincv::BinMat<W>& f, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            f.set(y, x, ((x * 7 + y * 13) % 29 == 0 || (x + y) % 37 == 0) ? 1u : 0u);
        }
    }
}

/// Exact float equality over the whole map, and the first difference if any.
bool sameMap(const std::vector<float>& a, const std::vector<float>& b, int w, int h,
             const char* label) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(w) +
                             static_cast<size_t>(x);
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                std::printf("  EQUALITY FAILED (%s) at (%d,%d): control %.9g, arm %.9g\n", label,
                            x, y, static_cast<double>(a[i]), static_cast<double>(b[i]));
                return false;
            }
        }
    }
    return true;
}

} // namespace

/// Loads a reference-binarized frame dumped as {int w, int h, w*h bytes of 0/1}.
/// Real content, with no OpenCV dependency in this core-only benchmark -- the skip
/// rate is a property of the DATA, so X-31 requires it on real content and not
/// only on synthetic texture, which would flatter or punish it arbitrarily.
bool loadReal(const char* path, bincv::BinMat<W>*& out, int& w, int& h,
              std::vector<bincv::BinMat<W>>& store) {
    std::FILE* fp = std::fopen(path, "rb");
    if (fp == nullptr) return false;
    int wh[2];
    if (std::fread(wh, sizeof(int), 2, fp) != 2) { std::fclose(fp); return false; }
    w = wh[0]; h = wh[1];
    std::vector<unsigned char> px(static_cast<size_t>(w) * static_cast<size_t>(h));
    if (std::fread(px.data(), 1, px.size(), fp) != px.size()) { std::fclose(fp); return false; }
    std::fclose(fp);
    store.emplace_back(w, h);
    out = &store.back();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            out->set(y, x, px[static_cast<size_t>(y) * static_cast<size_t>(w) +
                              static_cast<size_t>(x)] ? 1u : 0u);
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int w = 640, h = 480;
    std::vector<bincv::BinMat<W>> store;
    bincv::BinMat<W>* real = nullptr;
    const bool haveReal = argc > 1 && loadReal(argv[1], real, w, h, store);
    bincv::BinMat<W> synth(w, h);
    fillSynthetic(synth, w, h);
    bincv::BinMat<W>& f = haveReal ? *real : synth;
    bincv::TernaryMat<W> dx(w, h), dy(w, h);
    bincv::derivativeX(f, dx);
    bincv::derivativeY(f, dy);

    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<float> mc(n), m1(n), m2(n);
    size_t sk = 0, sk2 = 0;
    auto map = [&](std::vector<float>& v) {
        return bincv::ResponseMap{v.data(), static_cast<size_t>(w), static_cast<size_t>(h),
                                  static_cast<size_t>(w)};
    };

    size_t setPx = 0;
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) setPx += f.at(y, x) ? 1u : 0u;
    std::printf("=== X-31: corner response, per-pixel vs bit-sliced box sums ===\n");
    std::printf("%dx%d, blockSize 3, uint32_t, content = %s (%.2f%% set)\n\n", w, h,
                haveReal ? "REAL reference edge map" : "synthetic",
                100.0 * static_cast<double>(setPx) /
                    (static_cast<double>(w) * static_cast<double>(h)));

    cornerresp::perPixel(dx, dy, 3, map(mc), &sk);
    cornerresp::sliced(dx, dy, 3, map(m1), &sk);
    cornerresp::slicedSkip(dx, dy, 3, map(m2), &sk2);

    std::printf("  EQUALITY (exact float bits, whole %dx%d map)\n", w, h);
    bool ok = sameMap(mc, m1, w, h, "B1 vs control");
    ok = sameMap(mc, m2, w, h, "B2 vs control") && ok;
    if (!ok) {
        std::printf("\n  Bit-exactness is a PRECONDITION (X-31). Not timing a wrong kernel.\n");
        return 1;
    }
    const size_t totalWords = ((static_cast<size_t>(w) + 31) / 32) * static_cast<size_t>(h);
    std::printf("    B1 and B2 both bit-identical to the control.\n");
    std::printf("    sparsity skip: %zu of %zu words (%.1f%%)\n\n", sk2, totalWords,
                100.0 * static_cast<double>(sk2) / static_cast<double>(totalWords));

    std::vector<measure::Bench> b = {
        {"C  per-pixel (shipped)", [&](int) { cornerresp::perPixel(dx, dy, 3, map(mc), nullptr); }},
        {"B1 bit-sliced box sums", [&](int) { cornerresp::sliced(dx, dy, 3, map(m1), nullptr); }},
        {"B2 bit-sliced + sparsity skip",
         [&](int) { cornerresp::slicedSkip(dx, dy, 3, map(m2), nullptr); }},
    };
    const auto t = measure::measureInterleaved(b, 7, 60.0);
    std::printf("  %-30s %9s %9s\n", "arm", "ms", "vs C");
    for (size_t i = 0; i < b.size(); ++i) {
        std::printf("  %-30s %9.3f %8.2fx\n", b[i].name.c_str(), t[i].medianNs / 1e6,
                    t[0].medianNs / t[i].medianNs);
    }
    return 0;
}
