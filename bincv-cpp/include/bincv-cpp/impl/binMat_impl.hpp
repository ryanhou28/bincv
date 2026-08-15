#pragma once

#include <algorithm>
#include <ostream>

namespace bincv {

namespace impl {

// These helpers are templated directly on the storage word type, so they never
// need to reach back into BinMat to recover it.

/// @brief Number of pixels packed into one word of the given type.
template <typename WordType>
constexpr size_t bitsPerWord() {
    return sizeof(WordType) * 8;
}

/// @brief Index of the word within a row that holds the pixel at column x.
template <typename WordType>
inline size_t wordIndex(size_t x) {
    return x / bitsPerWord<WordType>();
}

/// @brief Mask with a single 1 at the bit corresponding to column x.
/// @note Indexing runs LSB->MSB, so column 0 is the least significant bit.
template <typename WordType>
inline WordType bitMask(size_t x) {
    return static_cast<WordType>(1) << (x % bitsPerWord<WordType>());
}

/// @brief Mask covering bits [0, n) of a word; n == bitsPerWord yields all ones.
template <typename WordType>
inline WordType lowBitsMask(size_t n) {
    if (n == 0) return static_cast<WordType>(0);
    if (n >= bitsPerWord<WordType>()) return static_cast<WordType>(~static_cast<WordType>(0));
    return static_cast<WordType>((static_cast<WordType>(1) << n) - 1);
}

/// @brief Words per row so that the row stride meets the requested byte alignment.
template <typename WordType>
inline size_t calcAlignedWidth(size_t widthPixels, size_t alignmentBytes) {
    constexpr size_t bitsPerWordV = bitsPerWord<WordType>();
    constexpr size_t bytesPerWord = sizeof(WordType);

    // Words needed to hold the row's pixels (one bit each), rounded up
    size_t wordsNeeded = (widthPixels + bitsPerWordV - 1) / bitsPerWordV;
    size_t rowBytes = wordsNeeded * bytesPerWord;

    // Round the row stride up to the requested byte alignment
    size_t alignedBytes = ((rowBytes + alignmentBytes - 1) / alignmentBytes) * alignmentBytes;

    return alignedBytes / bytesPerWord;
}

} // namespace impl

// Constructors
template <typename WordType_>
BinMat<WordType_>::BinMat()
    : width(0), height(0), rowAlignment(32), alignedWidth(0), storage() {}

template <typename WordType_>
BinMat<WordType_>::BinMat(int w, int h, size_t rowAlign)
    : width(0), height(0), rowAlignment(rowAlign), alignedWidth(0), storage() {

    if (w < 0 || h < 0) {
        throw std::invalid_argument("BinMat dimensions must be non-negative");
    }
    if (rowAlign == 0 || (rowAlign & (rowAlign - 1)) != 0) {
        throw std::invalid_argument("BinMat rowAlignment must be a positive power of two");
    }
    if (rowAlign % sizeof(WordType) != 0) {
        throw std::invalid_argument("BinMat rowAlignment must be a multiple of the word size");
    }

    width = static_cast<size_t>(w);
    height = static_cast<size_t>(h);
    alignedWidth = impl::calcAlignedWidth<WordType>(width, rowAlignment);

    // Allocate zero-initialized storage
    storage.assign(height * alignedWidth, 0);
}

#ifdef BINCV_WITH_OPENCV
// OpenCV interoperability implementations

template <typename WordType_>
void BinMat<WordType_>::fromCVMat(const cv::Mat& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input cv::Mat is empty");
    }
    if (input.type() != CV_8UC1) {
        throw std::invalid_argument("Input cv::Mat must be of type CV_8UC1");
    }

    width = static_cast<size_t>(input.cols);
    height = static_cast<size_t>(input.rows);
    alignedWidth = impl::calcAlignedWidth<WordType>(width, rowAlignment);

    // Allocate zero-initialized storage
    storage.assign(height * alignedWidth, 0);

    for (size_t y = 0; y < height; ++y) {
        const uint8_t* rowIn = input.ptr<uint8_t>(static_cast<int>(y));
        WordType* rowOut = &storage[y * alignedWidth];
        for (size_t x = 0; x < width; ++x) {
            if (rowIn[x]) {
                rowOut[impl::wordIndex<WordType>(x)] |= impl::bitMask<WordType>(x);
            }
        }
    }
}

// Shared unpacking loop for the two cv::Mat conversions; `transform` maps a bit
// to the output pixel value.
template <typename WordType, typename PixelTransform>
inline void toCVMatHelper(const BinMat<WordType>& binmat, cv::Mat& output, PixelTransform transform) {
    if (binmat.empty()) {
        output = cv::Mat();
        return;
    }

    output = cv::Mat::zeros(binmat.rows(), binmat.cols(), CV_8U);
    for (int y = 0; y < binmat.rows(); ++y) {
        const WordType* rowIn = binmat.ptr(y);
        uint8_t* rowOut = output.ptr<uint8_t>(y);
        for (size_t x = 0; x < binmat.getWidth(); ++x) {
            bool value = (rowIn[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) != 0;
            rowOut[x] = transform(value);
        }
    }
}

// @todo: could an approach using OpenCV's resize and scaling functions be more efficient?
//        need to think more about how to do this efficiently
template <typename WordType_>
void BinMat<WordType_>::toCVMat(cv::Mat& output) const {
    toCVMatHelper(*this, output, [](bool value) -> uint8_t { return value ? 1 : 0; });
}

template <typename WordType_>
void BinMat<WordType_>::toCVMatNormalized(cv::Mat& output) const {
    toCVMatHelper(*this, output, [](bool value) -> uint8_t { return value ? 255 : 0; });
}

#endif // BINCV_WITH_OPENCV

// clearTrailingBits
template <typename WordType_>
void BinMat<WordType_>::clearTrailingBits() {
    if (empty()) return;

    // Bits [width, alignedWidth * WordBits) are padding and must stay zero.
    size_t lastWord = impl::wordIndex<WordType>(width == 0 ? 0 : width - 1);
    size_t validBitsInLastWord = width - lastWord * WordBits;
    WordType keepMask = impl::lowBitsMask<WordType>(validBitsInLastWord);

    for (size_t y = 0; y < height; ++y) {
        WordType* row = &storage[y * alignedWidth];
        row[lastWord] &= keepMask;
        // Any whole words past the last partially-used one are pure padding
        std::fill(row + lastWord + 1, row + alignedWidth, static_cast<WordType>(0));
    }
}

// at and set
template <typename WordType_>
bool BinMat<WordType_>::at(int row, int col) const {
    if (row < 0 || row >= static_cast<int>(height) || col < 0 || col >= static_cast<int>(width)) {
        throw std::out_of_range("BinMat::at: index out of range");
    }
    const WordType* rowPtr = &storage[static_cast<size_t>(row) * alignedWidth];
    size_t x = static_cast<size_t>(col);
    return (rowPtr[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) != 0;
}

template <typename WordType_>
void BinMat<WordType_>::set(int row, int col, bool value) {
    if (row < 0 || row >= static_cast<int>(height) || col < 0 || col >= static_cast<int>(width)) {
        throw std::out_of_range("BinMat::set: index out of range");
    }
    WordType* rowPtr = &storage[static_cast<size_t>(row) * alignedWidth];
    size_t x = static_cast<size_t>(col);
    WordType& word = rowPtr[impl::wordIndex<WordType>(x)];
    WordType mask = impl::bitMask<WordType>(x);

    if (value) {
        word |= mask;   // set bit
    } else {
        word &= ~mask;  // clear bit
    }
}

// ptr
template <typename WordType_>
const typename BinMat<WordType_>::WordType* BinMat<WordType_>::ptr(int row) const {
    return &storage[static_cast<size_t>(row) * alignedWidth];
}

template <typename WordType_>
typename BinMat<WordType_>::WordType* BinMat<WordType_>::ptr(int row) {
    return &storage[static_cast<size_t>(row) * alignedWidth];
}

// resize
// @todo: This could potentially be optimized (word-wise copy when alignment permits)
template <typename WordType_>
void BinMat<WordType_>::resize(int newWidth, int newHeight) {
    if (newWidth < 0 || newHeight < 0)
        throw std::invalid_argument("BinMat dimensions must be non-negative");

    size_t nw = static_cast<size_t>(newWidth);
    size_t nh = static_cast<size_t>(newHeight);
    size_t newAlignedWidth = impl::calcAlignedWidth<WordType>(nw, rowAlignment);

    // Create new zero-initialized storage
    std::vector<WordType> newData(nh * newAlignedWidth, 0);

    // Determine region to copy from old matrix
    size_t minWidth = std::min(width, nw);
    size_t minHeight = std::min(height, nh);

    for (size_t y = 0; y < minHeight; ++y) {
        const WordType* oldRow = &storage[y * alignedWidth];
        WordType* newRow = &newData[y * newAlignedWidth];

        for (size_t x = 0; x < minWidth; ++x) {
            if (oldRow[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) {
                newRow[impl::wordIndex<WordType>(x)] |= impl::bitMask<WordType>(x);
            }
        }
    }

    // Update internal state
    width = nw;
    height = nh;
    alignedWidth = newAlignedWidth;
    storage = std::move(newData);
}

// pad
// @todo: This could potentially be optimized further
// @todo: OpenCV has a copyMakeBorder function, however alignment with our packed
//        representation becomes complicated when padding towards the left.
//        For now we implement our own padding.
template <typename WordType_>
void BinMat<WordType_>::pad(int top, int bottom, int left, int right, bool value) {
    if (top < 0 || bottom < 0 || left < 0 || right < 0) {
        throw std::invalid_argument("Padding values must be non-negative");
    }

    // Calculate new dimensions
    size_t newWidth = width + static_cast<size_t>(left) + static_cast<size_t>(right);
    size_t newHeight = height + static_cast<size_t>(top) + static_cast<size_t>(bottom);
    size_t newAlignedWidth = impl::calcAlignedWidth<WordType>(newWidth, rowAlignment);

    // Start from a buffer filled with the padding value, then overwrite the
    // interior with the original contents.
    WordType fillValue = value ? static_cast<WordType>(~static_cast<WordType>(0))
                               : static_cast<WordType>(0);
    std::vector<WordType> newData(newHeight * newAlignedWidth, fillValue);

    for (size_t y = 0; y < height; ++y) {
        const WordType* oldRow = &storage[y * alignedWidth];
        WordType* newRow = &newData[(y + static_cast<size_t>(top)) * newAlignedWidth];

        for (size_t x = 0; x < width; ++x) {
            size_t newX = x + static_cast<size_t>(left);
            bool bit = (oldRow[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) != 0;
            if (bit) {
                newRow[impl::wordIndex<WordType>(newX)] |= impl::bitMask<WordType>(newX);
            } else {
                newRow[impl::wordIndex<WordType>(newX)] &= ~impl::bitMask<WordType>(newX);
            }
        }
    }

    // Update internal state
    width = newWidth;
    height = newHeight;
    alignedWidth = newAlignedWidth;
    storage = std::move(newData);

    // A `true` fill writes ones across the whole stride, including padding bits
    if (value) clearTrailingBits();
}

// transposed
// @todo: naive pixel-by-pixel transpose; replace with a cache-blocked / bit-parallel
//        version (see ARCHITECTURE.md 6.4). This is currently the slowest operation.
template <typename WordType_>
BinMat<WordType_> BinMat<WordType_>::transposed() const {
    // Skip if empty matrix
    if (empty()) {
        return BinMat<WordType>();
    }

    BinMat<WordType> result(static_cast<int>(height), static_cast<int>(width), rowAlignment);
    for (size_t y = 0; y < height; ++y) {
        const WordType* row = &storage[y * alignedWidth];
        for (size_t x = 0; x < width; ++x) {
            if (row[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) {
                result.set(static_cast<int>(x), static_cast<int>(y), true);
            }
        }
    }
    return result;
}

// transpose
// @todo: this could avoid the copy made by transposed() for the square case
template <typename WordType_>
void BinMat<WordType_>::transpose() {
    *this = this->transposed();
}

// forEachNonZero
template <typename WordType_>
template <typename Func>
void BinMat<WordType_>::forEachNonZero(Func callback) const {
    if (empty()) {
        throw std::runtime_error("BinMat is empty, cannot iterate over non-zero pixels");
    }
    for (size_t y = 0; y < height; ++y) {
        const WordType* row = &storage[y * alignedWidth];
        for (size_t x = 0; x < width; ++x) {
            if (row[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) {
                callback(static_cast<int>(y), static_cast<int>(x));
            }
        }
    }
}

// printMatrix
template <typename WordType_>
void BinMat<WordType_>::printMatrix() const {
    if (empty()) {
        return;
    }

    for (size_t y = 0; y < height; ++y) {
        const WordType* row = &storage[y * alignedWidth];
        for (size_t x = 0; x < width; ++x) {
            std::cout << ((row[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) ? '1' : '0');
        }
        std::cout << '\n';
    }
}

// operator<<
// @todo: consider std::ostringstream or wrapping std::ostream& for reusable logging.
template <typename WordType>
std::ostream& operator<<(std::ostream& os, const BinMat<WordType>& binmat) {
    if (binmat.empty()) {
        return os; // Nothing to print for empty matrix
    }

    for (int y = 0; y < binmat.rows(); ++y) {
        const WordType* row = binmat.ptr(y);
        for (size_t x = 0; x < binmat.getWidth(); ++x) {
            os << ((row[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) ? '1' : '0');
        }
        os << '\n';
    }
    return os;
}

// printInternalData
template <typename WordType_>
void BinMat<WordType_>::printInternalData(bool hex) const {
    if (empty()) {
        return;
    }

    for (size_t y = 0; y < height; ++y) {
        const WordType* row = &storage[y * alignedWidth];
        for (size_t b = 0; b < alignedWidth; ++b) {
            if (hex)
                std::cout << std::hex << static_cast<uint64_t>(row[b]) << " ";
            else
                std::cout << std::dec << static_cast<uint64_t>(row[b]) << " ";
        }
        std::cout << '\n';
    }
    // Reset output stream to decimal format
    std::cout << std::dec;
}

// fill
template <typename WordType_>
void BinMat<WordType_>::fill(bool value) {
    if (empty())
        return;

    WordType fillWord = value ? static_cast<WordType>(~static_cast<WordType>(0))
                              : static_cast<WordType>(0);

    std::fill(storage.begin(), storage.end(), fillWord);

    // fill(true) sets the row-padding bits too; clear them so that word-wise
    // consumers (countNonZero, future bitwise ops) don't see phantom pixels.
    if (value) clearTrailingBits();
}

// countNonZero
// @todo: replace the per-pixel loop with popcount over whole words
//        (see ARCHITECTURE.md 6.3). Relies on padding bits being zero.
template <typename WordType_>
int BinMat<WordType_>::countNonZero() const {
    if (empty())
        return 0;

    int count = 0;
    for (size_t y = 0; y < height; ++y) {
        const WordType* row = &storage[y * alignedWidth];
        for (size_t x = 0; x < width; ++x) {
            if (row[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) {
                ++count;
            }
        }
    }
    return count;
}

// sparsity
template <typename WordType_>
float BinMat<WordType_>::sparsity() const {
    size_t totalPixels = width * height;
    if (totalPixels == 0)
        throw std::runtime_error("Sparsity is undefined for empty BinMat");

    int nonzero = countNonZero();
    return 1.0f - static_cast<float>(nonzero) / static_cast<float>(totalPixels);
}

} // namespace bincv
