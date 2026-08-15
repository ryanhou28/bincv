#include "bincv-cpp/binMat.hpp"
#include <cstdio>
template <typename W> void row(const char* n, int w, int h, size_t a) {
    bincv::BinMat<W> m(w, h, a);
    size_t bytes = m.sizeInWords() * sizeof(W);
    size_t ideal = (size_t(w) * h + 7) / 8;
    printf("%-10s %4dx%-4d align=%2zu -> %7zu B (ideal %6zu B, overhead %5.1f%%)\n",
           n, w, h, a, bytes, ideal, 100.0 * (double(bytes) - ideal) / ideal);
}
int main() {
    puts("--- 640x480 VIO frame, 1 bit/px ---");
    row<uint32_t>("u32", 640, 480, 32);
    row<uint32_t>("u32", 640, 480, 4);
    row<uint64_t>("u64", 640, 480, 8);
    puts("--- 752x480 (EuRoC resolution) ---");
    row<uint32_t>("u32", 752, 480, 32);
    row<uint32_t>("u32", 752, 480, 4);
    puts("--- pyramid level 3 (94x60), where padding hurts most ---");
    row<uint32_t>("u32", 94, 60, 32);
    row<uint32_t>("u32", 94, 60, 4);
    return 0;
}
