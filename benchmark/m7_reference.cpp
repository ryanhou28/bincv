// m7_reference.cpp -- what the board's measurement images print, computed on the host,
// for their outputs to be held to. Each subcommand builds its scene from the same header
// the board image does.
//
//   ./m7_reference essential [--dump]   the digests bincv_m7_essential prints, both
//                                        solvers, or with --dump the lines its
//                                        ESSENTIALDUMP prints
//   ./m7_reference subpix [--dump]      the same for bincv_m7_subpix and SUBPIXDUMP
//
// (The sparse stereo digests come from stereo_benchmark, which builds its pair from the
// same header as bincv_m7_stereo.)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "essential_scene.hpp"
#include "subpix_scene.hpp"

namespace {

int essential(bool dump) {
    using namespace essentialscene;
    std::vector<bincv::Point2f> from(kMaxCount), to(kMaxCount);
    std::vector<uint8_t> mask(kMaxCount);
    std::vector<uint32_t> flags(2 * bincv::ransacScratchWords(kMaxCount));
    if (!dump)
        std::printf("-- five-point RANSAC, normalised coordinates, threshold 0.002, cap 500 --\n");
    for (const Case& c : kCases) {
        const size_t planted = build(from.data(), to.data(), c.count, c.outlierPct, c.seed);
        const bincv::RansacScratch scratch{flags.data(), c.count};
        bincv::EssentialMatrix model;
        const bincv::RansacResult r = bincv::findEssentialMat(from.data(), to.data(), c.count,
                                                              params(), scratch, &model,
                                                              mask.data());
        if (dump) {
            std::printf("E %zu", c.count);
            for (const double v : model.m) {
                uint64_t b;
                std::memcpy(&b, &v, sizeof b);
                std::printf(" %016llx", static_cast<unsigned long long>(b));
            }
            std::printf("\nM %zu ", c.count);
            for (size_t i = 0; i < c.count; ++i) std::putchar(mask[i] ? '1' : '0');
            std::printf("\n");
            bincv::findEssentialMatRealRoots(from.data(), to.data(), c.count, params(), scratch,
                                             &model, mask.data());
            std::printf("R %zu", c.count);
            for (const double v : model.m) {
                uint64_t b;
                std::memcpy(&b, &v, sizeof b);
                std::printf(" %016llx", static_cast<unsigned long long>(b));
            }
            std::printf("\nQ %zu ", c.count);
            for (size_t i = 0; i < c.count; ++i) std::putchar(mask[i] ? '1' : '0');
            std::printf("\n");
            continue;
        }
        std::printf("  %zu correspondences, %d%% outliers: found %u inliers %zu of %zu planted, "
                    "iterations %d, model %08x, mask %08x\n",
                    c.count, c.outlierPct, r.found ? 1u : 0u, r.inliers, planted, r.iterations,
                    fnv1a(kFnvSeed, model.m, sizeof model.m), fnv1a(kFnvSeed, mask.data(), c.count));
        const bincv::RansacResult rr = bincv::findEssentialMatRealRoots(
            from.data(), to.data(), c.count, params(), scratch, &model, mask.data());
        std::printf("    real-root solver: found %u inliers %zu, iterations %d, model %08x, mask %08x\n",
                    rr.found ? 1u : 0u, rr.inliers, rr.iterations,
                    fnv1a(kFnvSeed, model.m, sizeof model.m), fnv1a(kFnvSeed, mask.data(), c.count));
    }
    return 0;
}

void putPoints(const char* tag, const bincv::Point2f* p, size_t n) {
    std::printf("%s", tag);
    for (size_t i = 0; i < n; ++i) {
        uint32_t x, y;
        std::memcpy(&x, &p[i].x, sizeof x);
        std::memcpy(&y, &p[i].y, sizeof y);
        std::printf(" %08x%08x", x, y);
    }
    std::printf("\n");
}

int subpix(bool dump) {
    using namespace subpixscene;
    std::vector<uint8_t> img(kW * kH), ring(3 * kW);
    std::vector<uint32_t> bits(kPlaneWords), dxw(2 * kPlaneWords), dyw(2 * kPlaneWords);
    std::vector<float> response(3 * kW);
    std::vector<bincv::CornerCandidate> pool(kPool);
    std::vector<bincv::Corner> corners(kMaxCorners);
    std::vector<bincv::Point2f> pts(kMaxCorners);
    const size_t n = build(img.data(), ring.data(), bits.data(), dxw.data(), dyw.data(),
                           response.data(), pool.data(), corners.data(), pts.data());
    const std::vector<bincv::Point2f> seeds(pts.begin(), pts.begin() + static_cast<long>(n));
    const bincv::SubPixResult r =
        bincv::cornerSubPix<1, uint32_t>(view(dxw.data()), view(dyw.data()), pts.data(), n,
                                         bincv::SubPixParams());
    if (dump) {
        putPoints("S", seeds.data(), n);
        putPoints("P", pts.data(), n);
        return 0;
    }
    std::printf("-- cornerSubPix, 752x480, winHalf 5, 40 iterations, eps 0.001 --\n");
    std::printf("  %zu seeds, %zu refined, points %08x\n", n, r.refined,
                stereoscene::fnv1a(stereoscene::kFnvSeed, pts.data(),
                                   n * sizeof(bincv::Point2f)));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const bool dump = argc == 3 && std::strcmp(argv[2], "--dump") == 0;
    if (argc == 2 || dump) {
        if (std::strcmp(argv[1], "essential") == 0) return essential(dump);
        if (std::strcmp(argv[1], "subpix") == 0) return subpix(dump);
    }
    std::printf("usage: m7_reference essential|subpix [--dump]\n");
    return 2;
}
