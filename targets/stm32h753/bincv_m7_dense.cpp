/// @file bincv_m7_dense.cpp
/// @brief `denseDisparityBinary` executed on the M7, at both word types.
///
/// The rule was written before this ran; it is in README.md, section 2.
///
///   DENSE           exactness at both word types, then cycles, median of 3

#include "dense_m7.hpp"
#include "m7_harness.hpp"

namespace {

void cmdDense(const char*) {
    densebench::runDenseBenchmark();
    boardPuts("OK\n");
}

const m7::Command kCommands[] = {{"DENSE", &cmdDense}};

}  // namespace

int main() { m7::serve("dense", kCommands, sizeof(kCommands) / sizeof(kCommands[0])); }
