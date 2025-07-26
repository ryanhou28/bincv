namespace bincv {

// @todo There are several instances of repeated index lookups when doing for-loops.
//       Need to refactor the code to make it cleaner.

namespace impl {

// Helper function to calculate word index
// Computes which word in the row contains the pixel at column x.
static inline int wordIndex(int x, size_t WordSize) {
    return x / WordSize;
}

// Helper function to create a bit mask with a 1 at the bit corresponding to column x.
// Creates a bitmask where the bit corresponding to column x is set to 1, and all others are 0.
// @note Indexing starts with LSB->MSB, so bit 0 is the least significant bit.
static inline typename BinMat<WordSize>::WordType bitMask(int x, size_t WordSize) {
    return static_cast<typename BinMat<WordSize>::WordType>(1) << (x % WordSize);
}

// Determine the number of elements a row should have to align with the chosen memory alignment.
static inline size_t calcAlignedWidth(size_t width, size_t alignment, size_t WordSize) {
    size_t rowBits = width * WordSize;

    // Align rowBits to the specified alignment. I.e., round up to the nearest multiple of alignment.
    size_t remainder = rowBits % alignment;
    if (remainder == 0) {
        return rowBits / WordSize;
    } else {
        return (rowBits + alignment - remainder) / WordSize;
    }
}

// Find a OpenCV data type based on the WordSize.
// @todo: we should do profiling to see which OpenCV type is more performant
static inline int cvTypeFromWordSize(size_t wordSize) {
    switch (wordSize) {
        case 8: return CV_8U;
        case 16: return CV_16U;
        case 32: return CV_32F;
        case 64: return CV_64F;
        default: throw std::invalid_argument("Unsupported WordSize");
    }

}

template <typename PixelTransform>
inline void BinMat<WordSize>::toCVMatHelper(cv::Mat& output, size_t WordSize, PixelTransform transform) const {
    if (width == 0 || height == 0) {
        // Return an empty matrix if dimensions are zero
        output = cv::Mat();
        return;
    }

    output = cv::Mat::zeros(height, width, CV_8U);
    for (size_t y = 0; y < height; ++y) {
        const WordType* row_in = mat.ptr<WordType>(y);
        uint8_t* row_out = output.ptr<uint8_t>(y);
        for (size_t x = 0; x < width; ++x) {
            bool value = (row_in[impl::wordIndex(x)] & impl::bitMask(x)) != 0;
            row_out[x] = transform(value);
        }
    }
}

} // namespace impl

template <size_t WordSize>
BinMat::BinMat()
    : width(0), height(0), rowAlignment(32), alignedWidth(0), mat() {}

template <size_t WordSize>
BinMat::BinMat(size_t width, size_t height, size_t rowAlignment)
    : width(width), height(height), rowAlignment(rowAlignment) {

    alignedWidth = impl::calcAlignedWidth(width, rowAlignment, WordSize);

    int cvType = impl::cvTypeFromWordSize(WordSize);

    mat = cv::Mat::zeros(height, alignedWidth, cvType);
}

template <size_t WordSize>
void BinMat::fromCVMat(const cv::Mat& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input cv::Mat is empty");
    }
    if (input.type() != CV_8UC1) {
        throw std::invalid_argument("Input cv::Mat must be of type CV_8UC1");
    }

    width = input.cols;
    height = input.rows;

    alignedWidth = impl::calcAlignedWidth(width, rowAlignment, WordSize);

    int cvType = impl::cvTypeFromWordSize(WordSize);

    mat = cv::Mat::zeros(height, alignedWidth, cvType);

    for (size_t y = 0; y < height; ++y) {
        const BinMat<WordSize>::WordType* row_in = input.ptr<BinMat<WordSize>::WordType>(y);
        BinMat<WordSize>::WordType* row_out = mat_.ptr<BinMat<WordSize>::WordType>(y);
        for (size_t x = 0; x < alignedWidth; ++x) {
            if (row_in[x]) {
                row_out[impl::wordIndex(x, WordSize)] |= impl::wordIndex(x, WordSize);
            }
        }
    }
}

// @todo: could an approach using OpenCV's resize and scaling functions be more efficient?
//              need to think more about how to do this efficiently
template <size_t WordSize>
void BinMat::toCVMat(cv::Mat& output) const {
    impl::toCVMatHelper(output, WordSize, [](bool value) { return value ? 1 : 0; });
}

template <size_t WordSize>
void BinMat::toCVMatNormalized(cv::Mat& output) const {
    impl::toCVMatHelper(output, WordSize, [](bool value) { return value ? 255 : 0; });
}

template <size_t WordSize>
bool at(int row, int col) const {
    if (row < 0 || row >= height || col < 0 || col >= width) {
        throw std::out_of_range("BinMat::at: index out of range");
    }
    const BinMat<WordSize>::WordType* row_ptr = mat_.ptr<BinMat<WordSize>::WordType>(row);
    return (row_ptr[impl::word_index(col, WordSize)] & impl::bit_mask(col, WordSize)) != 0;
}

template <size_t WordSize>
void set(int row, int col, bool value) {
    if (row < 0 || row >= height || col < 0 || col >= width) {
        throw std::out_of_range("BinMat::set: index out of range");
    }
    BinMat<WordSize>::WordType* row_ptr = mat_.ptr<BinMat<WordSize>::WordType>(row);
    BinMat<WordSize>::WordType& word = row_ptr[impl::wordIndex(col, WordSize)];
    BinMat<WordSize>::WordType mask = impl::bit_mask(col, WordSize);

    if (value) {
        word |= mask;   // set bit
    } else {
        word &= ~mask;  // clear bit
    }
}

template <size_t WordSize>
const typename BinMat<WordSize>::WordType* BinMat<WordSize>::ptr(int row) const {
    return mat.ptr<WordType>(row);
}

template <size_t WordSize>
typename BinMat<WordSize>::WordType* BinMat<WordSize>::ptr(int row) {
    return mat.ptr<WordType>(row);
}

// @todo: This could potentially be optimized
template <size_t WordSize>
void BinMat::resize(int newWidth, int newHeight) {
    if (width < 0 || height < 0)
        throw std::invalid_argument("BinMat dimensions must be non-negative");

    int newAlignedWidth = calcAlignedWidth(newWidth, rowAlignment, WordSize);

    int cvType = impl::cvTypeFromWordSize(WordSize);

    // Create new zero-initialized matrix
    cv::Mat newMat = cv::Mat::zeros(newHeight, newAlignedWidth, cvType);

    // Determine region to copy from old matrix
    int minWidth = std::min(width, newWidth);
    int minHeight = std::min(height, newHeight);

    if (minWidth == 0 || minHeight == 0) {
        // If either dimension is zero, we can skip copying
        width = 0;
        height = 0;
        alignedWidth = 0;
        mat = std::move(newMat);
        return;
    }

    for (size_t y = 0; y < minHeight; ++y) {
        const BinMat<WordSize>::WordType* oldRow = mat_.ptr<BinMat<WordSize>::WordType>(y);
        BinMat<WordSize>::WordType* newRow = new_mat.ptr<BinMat<WordSize>::WordType>(y);

        for (size_t x = 0; x < minWidth; ++x) {
            if (oldRow[impl::wordIndex(x, WordSize)] & impl::bit_mask(x, WordSize)) {
                newRow[impl::wordIndex(x, WordSize)] |= impl::bit_mask(x, WordSize);
            }
        }
    }

    // Update internal state
    width = newWidth;
    height = newHeight;
    alignedWidth = newAlignedWidth;
    mat = std::move(newMat);
}

// @todo: This could potentially be optimized further
// @todo: OpenCV has a copyMakeBorder function, however alignment with our packed representation becomes complicated with 
//        padding towards the left. For now we will implement our own padding function.
void BinMat::pad(int top, int bottom, int left, int right, bool value) {
    if (top < 0 || bottom < 0 || left < 0 || right < 0) {
        throw std::invalid_argument("Padding values must be non-negative");
    }

    // Calculate new dimensions
    size_t newWidth = width + left + right;
    size_t newHeight = height + top + bottom;

    // Calculate new aligned width
    size_t newAlignedWidth = impl::calcAlignedWidth(newWidth, rowAlignment, WordSize);

    // Determine OpenCV type based on WordSize
    int cvType = impl::cvTypeFromWordSize(WordSize);

    // Create a new matrix with the new dimensions
    cv::Mat newMat = cv::Mat::zeros(newHeight, newAlignedWidth, cvType);

    // Fill the new matrix with the padding value
    typename BinMat<WordSize>::WordType fillValue = value ? ~static_cast<typename BinMat<WordSize>::WordType>(0) : 0;

    for (size_t y = 0; y < newHeight; ++y) {
        typename BinMat<WordSize>::WordType* row = newMat.ptr<typename BinMat<WordSize>::WordType>(y);
        std::fill(row, row + newAlignedWidth, fillValue);
    }

    // Copy the original matrix into the new matrix at the correct offset
    for (size_t y = 0; y < height; ++y) {
        const typename BinMat<WordSize>::WordType* oldRow = mat.ptr<typename BinMat<WordSize>::WordType>(y);
        typename BinMat<WordSize>::WordType* newRow = newMat.ptr<typename BinMat<WordSize>::WordType>(y + top);

        for (size_t x = 0; x < width; ++x) {
            if (oldRow[impl::wordIndex(x)] & impl::bitMask(x)) {
                newRow[impl::wordIndex(x + left)] |= impl::bitMask(x + left);
            }
        }
    }

    // Update internal state
    width = newWidth;
    height = newHeight;
    alignedWidth = newAlignedWidth;
    mat = std::move(newMat);
}

template <size_t WordSize>
BinMat<WordSize> BinMat::transposed() const {
    // Skip if empty matrix
    if (width == 0 || height == 0) {
        return BinMat();
    }

    BinMat<WordSize> result(height, width, rowAlignment);
    for (size_t y = 0; y < height; ++y) {
        const BinMat<WordSize>::WordType* row = mat.ptr<BinMat<WordSize>::WordType>(y);
        for (size_t x = 0; x < width; ++x) {
            if (row[impl::wordIndex(x, WordSize)] & impl::bitMask(x, WordSize)) {
                result.set(x, y, true);
            }
        }
    }
    return result;
}

// @todo: this could potentially be optimized further to avoid copying data in the transposed() implementation
template <size_t WordSize>
void BinMat::transpose() {
    *this = this->transposed();
}

template <size_t WordSize>
template <typename Func>
void forEachNonZero(Func callback) const {
    if (width == 0 || height == 0) {
        throw std::runtime_error("BinMat is empty, cannot iterate over non-zero pixels");
    }
    for (size_t y = 0; y < height; ++y) {
        const BinMat<WordSize>::WordType* row = mat.ptr<BinMat<WordSize>::WordType>(y);
        for (size_t x = 0; x < width; ++x) {
            if (row[impl::wordIndex(x, WordSize)] & detail::bitMask(x, WordSize)) {
                callback(y, x);
            }
        }
    }
}

template <size_t WordSize>
void BinMat::printMatrix() const {
    if (width == 0 || height == 0) {
        return;
    }

    for (size_t y = 0; y < height; ++y) {
        const BinMat<WordSize>::WordType* row = mat_.ptr<BinMat<WordSize>::WordType>(y);
        for (size_t x = 0; x < width; ++x) {
            std::cout << ((row[impl::wordIndex(x, WordSize)] & impl::bitMask(x, WordSize)) ? '1' : '0');
        }
        std::cout << '\n';
    }
}

// @todo: consider std::ostringstream or wrapping std::ostream& for reusable logging.
template <size_t WordSize>
std::ostream& operator<<(std::ostream& os, const BinMat<WordSize>& binmat) {
    if (binmat.width() == 0 || binmat.height() == 0) {
        return os; // Nothing to print for empty matrix
    }

    for (size_t y = 0; y < binmat.height(); ++y) {
        const BinMat<WordSize>::WordType* row = binmat.ptr(y);
        for (size_t x = 0; x < binmat.width(); ++x) {
            os << ((row[impl::wordIndex(x, WordSize)] & impl::bitMask(x, WordSize)) ? '1' : '0');
        }
        os << '\n';
    }
    return os;
}

template <size_t WordSize>
void BinMat::printInternalData(bool hex) const {
    if (width == 0 || height == 0) {
        return;
    }

    for (size_t y = 0; y < height; ++y) {
        const BinMat<WordSize>::WordType* row = mat.ptr<BinMat<WordSize>::WordType>(y);
        for (size_t b = 0; b < alignedWidth; ++b) {
            if (hex)
                std::cout << std::hex << static_cast<int>(row[b]) << " ";
            else
                std::cout << std::dec << static_cast<int>(row[b]) << " ";
        }
        std::cout << '\n';
    }
    // Reset output stream to decimal format
    std::cout << std::dec;
}

template <size_t WordSize>
void BinMat::fill(bool value) {
    if (width == 0 || height == 0)
        return;

    BinMat<WordSize>::WordType fillWord = 0;

    if (value) {
        fillWord = ~static_cast<BinMat<WordSize>::WordType>(0);
    }

    for (size_t y = 0; y < height; ++y) {
        BinMat<WordSize>::WordType* row = mat.ptr<BinMat<WordSize>::WordType>(y);
        std::fill(row, row + alignedWidth, fillWord);
    }
}

// @todo: This could probably be optimized
template <size_t WordSize>
int BinMat::countNonZero() const {
    if (width == 0 || height == 0)
        return 0;

    int count = 0;
    for (size_t y = 0; y < height; ++y) {
        const BinMat<WordSize>::WordType* row = mat.ptr<BinMat<WordSize>::WordType>(y);
        for (size_t x = 0; x < width; ++x) {
            if (row[impl::wordIndex(x, WordSize)] & impl::bitMask(x, WordSize)) {
                ++count;
            }
        }
    }
    return count;
}

template <size_t WordSize>
float BinMat::sparsity() const {
    int totalPixels = width * height;
    if (totalPixels == 0)
        throw std::runtime_error("Sparsity is undefined for empty BinMat");

    int nonzero = countNonZero();
    return 1.0f - static_cast<float>(nonzero) / totalPixels;
}

} // namespace bincv
