#pragma once

#include <cstdint>

namespace bincv {


void horizontalEdgeFilter8b(const uint8_t* h_input, uint8_t* h_output, int width, int height, uint8_t threshold);

void verticalEdgeFilter8b(const uint8_t* h_input, uint8_t* h_output, int width, int height, uint8_t threshold);

void SEALEdgeFilter8b(const uint8_t* h_input, uint8_t* h_output, int width, int height, uint8_t threshold);

}
