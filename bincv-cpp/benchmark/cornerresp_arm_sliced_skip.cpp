// X-31 arm B2 -- B1 plus the word-level sparsity skip. Same body by inclusion, so
// the `Skip` template argument is the ONLY difference and the B1/B2 gap measures
// exactly the skip.
#include "cornerresp_sliced_body.inc"

namespace cornerresp {
void slicedSkip(const bincv::TernaryMat<uint32_t>& dx, const bincv::TernaryMat<uint32_t>& dy,
                int blockSize, bincv::ResponseMap out, size_t* skippedWords) {
    slicedImpl<true>(dx, dy, blockSize, out, skippedWords);
}
} // namespace cornerresp
