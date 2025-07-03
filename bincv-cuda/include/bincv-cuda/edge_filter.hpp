#pragma once

#include <cstdint>

namespace bincv {


void runHorizontalEdgeFilter(const uint8_t* h_input, uint8_t* h_output, int width, int height, uint8_t threshold);

}
