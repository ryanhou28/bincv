// X-31 arm C -- the shipped per-pixel form. The control.
#include "cornerresp_arms.hpp"

namespace cornerresp {
void perPixel(const bincv::TernaryMat<uint32_t>& dx, const bincv::TernaryMat<uint32_t>& dy,
              int blockSize, bincv::ResponseMap out, size_t* skippedWords) {
    if (skippedWords != nullptr) *skippedWords = 0;
    bincv::cornerMinEigenVal<uint32_t>(dx, dy, blockSize, out);
}
} // namespace cornerresp
