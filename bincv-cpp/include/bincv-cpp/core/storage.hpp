#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

// BINCV_ABI_NAMESPACE (core/error.hpp). Storage itself has no configuration-
// dependent code, but it must sit in the same namespace as everything that
// holds one.
#include "error.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief Backing memory for a bit-packed matrix: {pointer, word count, ownership}.
/// @tparam WordType_ The unsigned integral type the buffer is measured in.
///
/// @note Storage is where allocation lives. Kernels take views (core/view.hpp),
///       never this type, so a kernel never has to care whether its arguments own
///       their memory.
/// @note Two backings are supported, and they are the reason this type exists
///       rather than a std::vector member (ARCHITECTURE 4.3):
///       - owning:     a heap allocation this object created and will free
///       - non-owning: a caller-provided buffer (static, stack, DMA, sensor),
///                     which is the Tier 2 / no-heap path and never allocates
/// @note Value semantics (D-8): copying an owning Storage deep-copies. There is
///       no reference counting -- sharing is expressed by taking a view. Copying
///       a non-owning Storage yields a non-owning Storage over the same memory;
///       it does not promote to owning, because the caller's buffer is the point.
template <typename WordType_>
class Storage {
    static_assert(std::is_same<WordType_, typename std::remove_cv<WordType_>::type>::value,
                  "WordType must not be cv-qualified");
    static_assert(std::is_integral<WordType_>::value && std::is_unsigned<WordType_>::value &&
                      !std::is_same<WordType_, bool>::value &&
                      !std::is_same<WordType_, char>::value,
                  "WordType must be an unsigned integral type, and not bool or char");
    // bool would make WordBits a lie (sizeof is 1 but only one bit is usable), and
    // plain char is unsigned on some targets and signed on others -- accepting it
    // would make the supported type set differ between the project's two tiers.
    static_assert(sizeof(WordType_) == 1 || sizeof(WordType_) == 2 ||
                      sizeof(WordType_) == 4 || sizeof(WordType_) == 8,
                  "WordType must be 8, 16, 32, or 64 bits wide");

public:
    /// The word type the buffer is measured and addressed in.
    using WordType = WordType_;

    // Constructors

    /// @brief Constructs empty storage. Allocates nothing.
    Storage() noexcept : ptr_(nullptr), words_(0), owns_(false) {}

    /// @brief Allocates and zero-fills `words` words, owned by this object.
    /// @param words Number of words to allocate. Zero allocates nothing.
    /// @note Zero-filled because padding bits must start clear -- word-wise
    ///       reductions over-count otherwise.
    /// @note Allocation failure is left to plain new[], which already produces
    ///       exactly the two behaviours BINCV_THROW encodes: it throws
    ///       std::bad_alloc where exceptions are enabled, and aborts where they
    ///       are not. Routing it through the macro would mean allocating with
    ///       new(std::nothrow) and testing the result on every allocation, and
    ///       would force a choice of exception type -- std::bad_alloc is not
    ///       constructible from a message, so the macro cannot express it. That
    ///       choice is not recorded anywhere, so it is not made here (T1.4).
    /// @note Consequence worth knowing: the abort on this path is the runtime's,
    ///       so it carries no binCV diagnostic. Every failure binCV reports
    ///       itself goes through core/error.hpp.
    explicit Storage(size_t words)
        : ptr_(words > 0 ? new WordType[words]() : nullptr),
          words_(words),
          owns_(words > 0) {}

    /// @brief Wraps caller-provided memory without taking ownership of it.
    /// @param ptr First word of the caller's buffer. Must outlive this object.
    /// @param words Number of words available at `ptr`.
    /// @note Performs no allocation at all -- this is the Tier 2 path.
    /// @note The buffer is used as-is; nothing is zeroed, since the caller may
    ///       be wrapping data it has already filled in.
    /// @note A degenerate wrap -- a null pointer, or zero words -- normalizes to
    ///       the empty state, so that `empty()` stays a true emptiness test and
    ///       `!empty()` really does mean `data()` is usable. Without this,
    ///       Storage(nullptr, n) would report n words at a null pointer, and any
    ///       caller trusting size()/empty() would walk it. Note that the literal
    ///       form Storage(0, n) selects this constructor, since 0 is a null
    ///       pointer constant.
    Storage(WordType* ptr, size_t words) noexcept
        : ptr_(words > 0 ? ptr : nullptr),
          words_(ptr != nullptr ? words : 0),
          owns_(false) {}

    // Special members

    /// @brief Deep-copies an owning Storage; aliases a non-owning one.
    Storage(const Storage& other)
        : ptr_(nullptr), words_(other.words_), owns_(other.owns_) {
        if (owns_) {
            ptr_ = new WordType[words_];
            copyWords(ptr_, other.ptr_, words_);
        } else {
            ptr_ = other.ptr_;
        }
    }

    /// @brief Deep-copies an owning Storage; aliases a non-owning one.
    /// @note The new buffer is allocated and filled before the old one is
    ///       released, so a failed allocation leaves this object untouched.
    /// @note Two distinct aliasing cases are handled, and the identity guard
    ///       below only covers the first:
    ///       - `s = s`: caught by `this == &other`.
    ///       - `s = alias`, where `alias` is a *different*, non-owning Storage
    ///         wrapping memory `s` owns. Nothing about the two objects is equal,
    ///         so this reaches the assignment body; releasing first would free
    ///         the block and then adopt the freed pointer. It is defined as a
    ///         no-op instead -- see aliasesOwnedBlock() for why that is the only
    ///         safe answer.
    Storage& operator=(const Storage& other) {
        if (this == &other) return *this;

        if (other.owns_) {
            // An owning source holds its own new[] block, so it cannot overlap
            // this object's: allocate and copy first, then release.
            WordType* fresh = new WordType[other.words_];
            copyWords(fresh, other.ptr_, other.words_);
            release();
            ptr_ = fresh;
            words_ = other.words_;
            owns_ = true;
            return *this;
        }

        // Read `other`'s descriptor before anything is freed: `other` may be a
        // non-owning alias of the very block release() is about to hand back.
        WordType* const otherPtr = other.ptr_;
        const size_t otherWords = other.words_;
        if (aliasesOwnedBlock(otherPtr)) return *this;

        adoptThenFree(otherPtr, otherWords, false);
        return *this;
    }

    /// @brief Takes over `other`'s buffer and ownership, leaving it empty.
    Storage(Storage&& other) noexcept
        : ptr_(other.ptr_), words_(other.words_), owns_(other.owns_) {
        other.clear();
    }

    /// @brief Takes over `other`'s buffer and ownership, leaving it empty.
    /// @note A moved-from Storage is empty, safe to destroy, and safe to
    ///       assign to again.
    /// @note Carries the same aliasing rule as copy-assignment: moving from a
    ///       non-owning Storage that wraps this object's own block leaves this
    ///       object unchanged. `other` is still emptied, so a moved-from
    ///       Storage is empty either way.
    Storage& operator=(Storage&& other) noexcept {
        if (this == &other) return *this;

        // Read the descriptor before anything is freed, for the same reason as
        // in copy-assignment. Only a non-owning source can name this object's
        // block; an owning one always holds a distinct new[] allocation.
        WordType* const otherPtr = other.ptr_;
        const size_t otherWords = other.words_;
        const bool otherOwns = other.owns_;

        if (!otherOwns && aliasesOwnedBlock(otherPtr)) {
            other.clear();
            return *this;
        }

        adoptThenFree(otherPtr, otherWords, otherOwns);
        other.clear();
        return *this;
    }

    /// @brief Frees the buffer only if this object owns it.
    ~Storage() { release(); }

    // Accessors

    /// @brief First word of the buffer, or nullptr when empty.
    WordType* data() { return ptr_; }
    const WordType* data() const { return ptr_; }

    /// @brief Buffer size in WORDS, not bytes.
    size_t size() const { return words_; }

    /// @brief True when the buffer holds no words.
    bool empty() const { return words_ == 0; }

    /// @brief True when this object will free the buffer on destruction.
    /// @note Empty storage owns nothing, so this is false for both a
    ///       default-constructed and a moved-from object.
    bool ownsMemory() const { return owns_; }

private:
    /// @brief Copies `words` words. Word types are trivially copyable by the
    ///        static_assert above, so a byte copy is exact.
    static void copyWords(WordType* dst, const WordType* src, size_t words) {
        if (words > 0) std::memcpy(dst, src, words * sizeof(WordType));
    }

    /// @brief True if `p` points into the block this object owns.
    /// @note Wrapping owned memory is a first-class pattern here -- a non-owning
    ///       Storage over another Storage's buffer is how sharing is expressed --
    ///       so an assignment source can legitimately name memory the target is
    ///       about to free. There is no way to honour such an assignment: dropping
    ///       ownership either leaks the block or leaves both objects dangling, and
    ///       adopting an interior pointer while staying owning would make delete[]
    ///       undefined. The assignment is therefore defined as a no-op, which
    ///       leaves both objects valid and the block owned by exactly one of them.
    /// @note Compared as integers on purpose: relational comparison of pointers
    ///       into different objects is unspecified, and this predicate exists
    ///       precisely to ask about pointers that may be unrelated.
    bool aliasesOwnedBlock(const WordType* p) const {
        if (!owns_ || ptr_ == nullptr || p == nullptr) return false;
        const uintptr_t base = reinterpret_cast<uintptr_t>(ptr_);
        const uintptr_t here = reinterpret_cast<uintptr_t>(p);
        return here >= base && here < base + words_ * sizeof(WordType);
    }

    /// @brief Installs a new descriptor, then frees the block this object held.
    /// @note The order is the point. The obvious spelling of both assignment
    ///       operators is `release(); ptr_ = otherPtr;`, and it is correct --
    ///       each caller has already established, via aliasesOwnedBlock() or via
    ///       new[] returning a fresh block, that the incoming pointer does not
    ///       point into the block being freed. But that reasoning lives in the
    ///       caller, and GCC 12's -Wuse-after-free cannot follow it: all it sees
    ///       is a pointer stored after a free that might have been the same
    ///       pointer, which is precisely the bug the warning exists to catch.
    ///       Freeing last removes the question instead of answering it, and costs
    ///       nothing -- the two orders are otherwise indistinguishable.
    /// @note Found by scripts/verify_arm.sh: its container ships GCC 12, where
    ///       -Wall enables this warning. GCC 11 -- the desktop compiler this
    ///       project has been developed against -- does not have it at all, so
    ///       "builds warning-free" was true of one compiler and not of the next.
    void adoptThenFree(WordType* newPtr, size_t newWords, bool newOwns) {
        WordType* const stale = owns_ ? ptr_ : nullptr;
        ptr_ = newPtr;
        words_ = newWords;
        owns_ = newOwns;
        delete[] stale;   // no-op when null, which is the non-owning case
    }

    /// @brief Releases the buffer if owned, and resets to the empty state so a
    ///        freed pointer can never survive the call.
    void release() {
        if (owns_) delete[] ptr_;
        clear();
    }

    /// @brief Resets to the empty, non-owning state without freeing anything.
    void clear() noexcept {
        ptr_ = nullptr;
        words_ = 0;
        owns_ = false;
    }

    WordType* ptr_;   // first word of the buffer, or nullptr when empty
    size_t words_;    // buffer size in words
    bool owns_;       // true if ptr_ came from this object's own new[]
};

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
