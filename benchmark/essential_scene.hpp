// essential_scene.hpp -- correspondences for the five-point RANSAC, built the same way
// on a host and on a microcontroller, and digests of what the estimator returns.
//
// bincv_m7_essential runs `findEssentialMat` on the board over these scenes and
// m7_reference prints the host's digests for the board's to be held to. Points seen
// by two cameras a known rotation and translation apart, in normalised coordinates,
// with a planted share of outliers.
//
// NO TRIGONOMETRY. The rotation is a unit quaternion's -- three draws and a square
// root -- because the C libraries a host and a board link need not agree on the last
// bit of a sine, and a scene that differs before the estimator starts says nothing
// about the estimator. A square root is correctly rounded everywhere.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "bincv/core/types.hpp"
#include "bincv/ops/essential.hpp"

namespace essentialscene {

inline constexpr double kThreshold = 0.002;  // normalised units, as essential_benchmark

/// The scene: `count` correspondences, the first `outlierPct` of every hundred
/// displaced. Returns how many are planted inliers.
inline size_t build(bincv::Point2f* from, bincv::Point2f* to, size_t count, int outlierPct,
                    uint64_t seed) {
    uint64_t s = seed;
    const auto uni = [&s]() {
        s = s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return 2.0 * (static_cast<double>((s >> 33) % 1000001) / 1000000.0) - 1.0;
    };
    double qw = 1.0, qx = uni() * 0.2, qy = uni() * 0.2, qz = uni() * 0.2;
    const double qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    qw /= qn;
    qx /= qn;
    qy /= qn;
    qz /= qn;
    const double R[3][3] = {
        {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)},
        {2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)},
        {2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)}};
    double t[3] = {uni(), uni(), 1.0};
    const double tn = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    for (int i = 0; i < 3; ++i) t[i] /= tn;

    size_t inliers = 0;
    for (size_t i = 0; i < count; ++i) {
        const double X[3] = {uni() * 2.0, uni() * 2.0, 4.0 + uni()};
        bincv::Point2f a{static_cast<float>(X[0] / X[2]), static_cast<float>(X[1] / X[2])};
        double Xc[3];
        for (int k = 0; k < 3; ++k) {
            double acc = 0.0;
            for (int j = 0; j < 3; ++j) acc += R[k][j] * X[j];
            Xc[k] = acc + t[k];
        }
        bincv::Point2f b{static_cast<float>(Xc[0] / Xc[2]), static_cast<float>(Xc[1] / Xc[2])};
        if (static_cast<int>(i % 100) < outlierPct) {
            b.x += static_cast<float>(uni() * 0.4);
            b.y += static_cast<float>(uni() * 0.4);
        } else {
            ++inliers;
        }
        from[i] = a;
        to[i] = b;
    }
    return inliers;
}

/// The sizes measured: correspondences, outlier percentage, and the scene's seed.
struct Case {
    size_t count;
    int outlierPct;
    uint64_t seed;
};
inline constexpr Case kCases[] = {{200, 30, 0x5EED + 200}, {500, 30, 0x5EED + 500},
                                   {1000, 50, 0x5EED + 1000}};
inline constexpr size_t kMaxCount = 1000;

inline uint32_t fnv1a(uint32_t h, const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}
inline constexpr uint32_t kFnvSeed = 2166136261u;

/// The estimator's parameters for every case: the benchmark's threshold and cap.
inline bincv::RansacParams params() {
    bincv::RansacParams p;
    p.threshold = kThreshold;
    p.maxIterations = 500;
    return p;
}

} // namespace essentialscene
