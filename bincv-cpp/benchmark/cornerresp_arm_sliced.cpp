// X-31 arm B1 -- bit-sliced 3x3 box sums, NO sparsity skip.
#include "cornerresp_sliced_body.inc"

namespace cornerresp {
void sliced(const bincv::TernaryMat<uint32_t>& dx, const bincv::TernaryMat<uint32_t>& dy,
            int blockSize, bincv::ResponseMap out, size_t* skippedWords) {
    slicedImpl<false>(dx, dy, blockSize, out, skippedWords);
}
} // namespace cornerresp
