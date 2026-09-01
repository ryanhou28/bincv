// Core tests for the storage model and the view types.
// Deliberately free of OpenCV so this suite also runs in the core-only and
// no-exceptions configurations -- those are the configurations these two types
// exist to serve.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <type_traits>
#include <utility>

#include "bincv-cpp/core/storage.hpp"
#include "bincv-cpp/core/view.hpp"
#include "test_util.hpp"

// ---------------------------------------------------------------------------
// Allocation counters
//
// "A non-owning Storage allocates nothing" is the Tier 2 claim, and it is only a
// claim if something counts. Replacing the global operators is the cheapest way
// to observe every allocation the process makes.
// ---------------------------------------------------------------------------

namespace {

size_t g_newCount = 0;
size_t g_deleteCount = 0;

// Keeps the optimizer from eliding an allocation whose result it can prove is
// unused -- C++14 permits new/delete elision, which would make the counters lie.
const void* volatile g_sink = nullptr;
inline void escape(const void* p) { g_sink = p; }

// Every replacement operator below routes through this pair rather than the
// scalar forms forwarding to each other, so that each operator has its own body
// and every malloc is visibly paired with a free in the same two functions. The
// alternative -- `operator new` calling `::operator new`, the sized deletes
// calling `::operator delete` -- spreads one allocation's lifetime across four
// bodies that a reader has to hold in their head at once, for no gain.
//
// NOT because a compiler rejects the forwarding spelling. An earlier version of
// this comment said -Wall reports -Wmismatched-new-delete on it at -O2; that does
// not reproduce anywhere in this project's verification set. Measured with the
// project flag set
// (-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion), on the
// forwarding spelling, zero warnings in every combination:
//
// g++ 11.4 (x86_64) -O0 -O1 -O2 -O3
// g++ 12.5 (x86_64, gcc:12) -O0 -O1 -O2 -O3
// g++ 12.5 (arm64v8/gcc:12) -O0 -O2 <- the image scripts/verify_arm.sh uses
// clang++ 18 (x86_64) -O0 -O2
//
// The -Walloc-size-larger-than= claim that used to sit on the guard below did not
// reproduce either (g++ 11.4, -O0/-O2, guard removed). Keeping a rule no compiler
// in the gate enforces teaches the next contributor something untrue, so the
// style reason above is the one that stands. The guard itself stays: it is
// behaviour, not warning-appeasement.
//
// The neighbouring -Wuse-after-free finding in core/storage.hpp 
// is a different matter -- that one does reproduce on GCC 12 -- so the two should
// not be read as equally shaky.
void* countedAllocate(std::size_t bytes) {
    ++g_newCount;
    // Reject unsatisfiable requests up front: array-new passes SIZE_MAX as its
    // overflow sentinel and expects the allocator to fail rather than to hand
    // back whatever malloc(SIZE_MAX) does on this platform.
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    // Cannot throw std::bad_alloc: this file also builds with -fno-exceptions.
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}

void countedFree(void* p) noexcept {
    if (p != nullptr) ++g_deleteCount;
    std::free(p);
}

} // namespace

void* operator new(std::size_t bytes)   { return countedAllocate(bytes); }
void* operator new[](std::size_t bytes) { return countedAllocate(bytes); }

void operator delete(void* p) noexcept                 { countedFree(p); }
void operator delete[](void* p) noexcept               { countedFree(p); }
void operator delete(void* p, std::size_t) noexcept    { countedFree(p); }
void operator delete[](void* p, std::size_t) noexcept  { countedFree(p); }

namespace {

// ---------------------------------------------------------------------------
// -- Storage
// ---------------------------------------------------------------------------

// The behavioural contract, run against every supported word width so the
// word-sized arithmetic in size/copy is exercised at 8/16/32/64 bits.
template <typename W>
void testStorage(const char* label) {
    using S = bincv::Storage<W>;
    std::cout << "\n--- Storage<" << label << "> ---\n";

    // Default construction: empty, owns nothing, holds no pointer.
    S defaulted;
    BINCV_CHECK(defaulted.data() == nullptr);
    BINCV_CHECK_EQ(defaulted.size(), size_t(0));
    BINCV_CHECK(defaulted.empty());
    BINCV_CHECK(!defaulted.ownsMemory());

    // Owning construction: sized in WORDS, owned, and zero-initialized -- padding
    // bits must start clear or word-wise reductions over-count.
    S owned(16);
    BINCV_CHECK(owned.data() != nullptr);
    BINCV_CHECK_EQ(owned.size(), size_t(16));
    BINCV_CHECK(!owned.empty());
    BINCV_CHECK(owned.ownsMemory());
    bool allZero = true;
    for (size_t i = 0; i < owned.size(); ++i) {
        if (owned.data()[i] != W(0)) allZero = false;
    }
    BINCV_CHECK(allZero);

    // A zero-word owning request is just empty storage; there is nothing to own.
    S zeroWords(0);
    BINCV_CHECK(zeroWords.data() == nullptr);
    BINCV_CHECK(zeroWords.empty());
    BINCV_CHECK(!zeroWords.ownsMemory());

    // Non-owning construction wraps the caller's buffer and reads/writes it in
    // place: this is the MCU / DMA path.
    W buf[8] = {};
    buf[0] = W(0xA5);
    S wrapped(buf, 8);
    BINCV_CHECK(wrapped.data() == buf);
    BINCV_CHECK_EQ(wrapped.size(), size_t(8));
    BINCV_CHECK(!wrapped.ownsMemory());
    BINCV_CHECK(wrapped.data()[0] == W(0xA5));
    wrapped.data()[1] = W(0x5A);
    BINCV_CHECK(buf[1] == W(0x5A));

    // REGRESSION: a degenerate wrap normalizes to the empty state, so that
    // `!empty` really does mean `data` is usable. Storage(nullptr, n) used to
    // report n words at a null pointer, and a caller following the documented
    // `if (!s.empty) s.data[0] =...` idiom -- or building a view from
    // {s.data, w, h, stride} -- would walk it.
    S nullWrap(nullptr, 10);
    BINCV_CHECK(nullWrap.data() == nullptr);
    BINCV_CHECK(nullWrap.empty());              // was false, with size == 10
    BINCV_CHECK_EQ(nullWrap.size(), size_t(0));
    BINCV_CHECK(!nullWrap.ownsMemory());

    // The literal form selects this constructor too: 0 is a null pointer constant.
    S literalNullWrap(0, 8);
    BINCV_CHECK(literalNullWrap.empty());
    BINCV_CHECK_EQ(literalNullWrap.size(), size_t(0));

    //...and symmetrically, wrapping zero words leaves nothing to address.
    S zeroWordWrap(buf, 0);
    BINCV_CHECK(zeroWordWrap.empty());
    BINCV_CHECK(zeroWordWrap.data() == nullptr);
    BINCV_CHECK(!zeroWordWrap.ownsMemory());

    // --- copy construction ---

    // Copying an owning Storage deep-copies: new buffer, same contents,
    // and the two are independent afterwards in both directions.
    S src(4);
    src.data()[0] = W(1);
    src.data()[3] = W(2);
    S copied(src);
    BINCV_CHECK(copied.data() != src.data());
    BINCV_CHECK(copied.ownsMemory());
    BINCV_CHECK_EQ(copied.size(), size_t(4));
    BINCV_CHECK(copied.data()[0] == W(1));
    BINCV_CHECK(copied.data()[3] == W(2));
    copied.data()[0] = W(9);
    BINCV_CHECK(src.data()[0] == W(1));
    src.data()[3] = W(7);
    BINCV_CHECK(copied.data()[3] == W(2));

    // Copying a non-owning Storage stays non-owning over the SAME memory. It must
    // not promote to owning, or the caller's buffer would silently stop being the
    // buffer being written.
    S alias(wrapped);
    BINCV_CHECK(alias.data() == buf);
    BINCV_CHECK(!alias.ownsMemory());
    BINCV_CHECK_EQ(alias.size(), size_t(8));
    alias.data()[2] = W(3);
    BINCV_CHECK(buf[2] == W(3));

    // --- copy assignment ---

    // Owning source into an owning target: deep copy, target resized.
    S dst(2);
    dst = src;
    BINCV_CHECK(dst.ownsMemory());
    BINCV_CHECK_EQ(dst.size(), size_t(4));
    BINCV_CHECK(dst.data() != src.data());
    BINCV_CHECK(dst.data()[3] == W(7));

    // Non-owning source: the target becomes non-owning over the same memory, and
    // releases whatever it owned before.
    dst = wrapped;
    BINCV_CHECK(dst.data() == buf);
    BINCV_CHECK(!dst.ownsMemory());
    BINCV_CHECK_EQ(dst.size(), size_t(8));

    //...and back the other way, proving a non-owning target does not free the
    // caller's buffer when it is reassigned.
    dst = src;
    BINCV_CHECK(dst.ownsMemory());
    BINCV_CHECK(dst.data() != buf);
    BINCV_CHECK(buf[2] == W(3));

    // Self copy-assignment must not free the buffer it is about to read. The
    // reference alias is only to keep compilers from diagnosing the obvious form.
    S selfCopy(3);
    selfCopy.data()[1] = W(42);
    const W* selfCopyPtr = selfCopy.data();
    S& selfCopyRef = selfCopy;
    selfCopy = selfCopyRef;
    BINCV_CHECK(selfCopy.data() == selfCopyPtr);
    BINCV_CHECK_EQ(selfCopy.size(), size_t(3));
    BINCV_CHECK(selfCopy.ownsMemory());
    BINCV_CHECK(selfCopy.data()[1] == W(42));

    // Self copy-assignment of a non-owning Storage keeps pointing at the caller.
    S selfAlias(buf, 8);
    S& selfAliasRef = selfAlias;
    selfAlias = selfAliasRef;
    BINCV_CHECK(selfAlias.data() == buf);
    BINCV_CHECK(!selfAlias.ownsMemory());

    // REGRESSION: aliasing self-assignment. `wrapAtBase` is a *different* object
    // from `blockOwner`, so the this == &other guard does not fire, but it names
    // the very block blockOwner is about to free. Assigning it used to free the
    // block and then adopt the freed pointer -- a use-after-free that left both
    // objects dangling. Ownership must survive, or nothing frees the block and
    // every read through either object is on freed memory.
    S blockOwner(4);
    blockOwner.data()[0] = W(0x3C);
    const W* ownedBase = blockOwner.data();
    S wrapAtBase(blockOwner.data(), 4);
    blockOwner = wrapAtBase;
    BINCV_CHECK(blockOwner.ownsMemory());          // was false: block leaked AND dangling
    BINCV_CHECK(blockOwner.data() == ownedBase);
    BINCV_CHECK_EQ(blockOwner.size(), size_t(4));
    BINCV_CHECK(blockOwner.data()[0] == W(0x3C));  // still live memory
    blockOwner.data()[1] = W(0x3D);
    BINCV_CHECK(wrapAtBase.data()[1] == W(0x3D));  // and still the same memory

    // Same defect through an interior pointer -- the shape that work’s non-owning
    // BinMat constructor makes reachable as `frame = roiWrapper`.
    S interiorOwner(6);
    interiorOwner.data()[3] = W(0x5E);
    const W* interiorBase = interiorOwner.data();
    S wrapInside(interiorOwner.data() + 2, 3);
    interiorOwner = wrapInside;
    BINCV_CHECK(interiorOwner.ownsMemory());
    BINCV_CHECK(interiorOwner.data() == interiorBase);
    BINCV_CHECK(interiorOwner.data()[3] == W(0x5E));

    // The same rule under move-assignment, which had the identical ordering bug.
    S moveBlockOwner(4);
    moveBlockOwner.data()[2] = W(0x6F);
    const W* moveOwnedBase = moveBlockOwner.data();
    S moveWrap(moveBlockOwner.data(), 4);
    moveBlockOwner = std::move(moveWrap);
    BINCV_CHECK(moveBlockOwner.ownsMemory());
    BINCV_CHECK(moveBlockOwner.data() == moveOwnedBase);
    BINCV_CHECK(moveBlockOwner.data()[2] == W(0x6F));
    BINCV_CHECK(moveWrap.empty());                 // moved-from is empty either way

    // A non-owning source that does NOT alias the target's block still replaces
    // it, as it always did -- the aliasing rule must not swallow the normal case.
    S unrelatedOwner(4);
    unrelatedOwner = wrapAtBase;
    BINCV_CHECK(!unrelatedOwner.ownsMemory());
    BINCV_CHECK(unrelatedOwner.data() == ownedBase);

    // --- move construction ---

    S movedFrom(5);
    movedFrom.data()[0] = W(11);
    const W* movedPtr = movedFrom.data();
    S movedTo(std::move(movedFrom));
    BINCV_CHECK(movedTo.data() == movedPtr);   // buffer transferred, not copied
    BINCV_CHECK(movedTo.ownsMemory());
    BINCV_CHECK_EQ(movedTo.size(), size_t(5));
    BINCV_CHECK(movedTo.data()[0] == W(11));

    // A moved-from Storage is empty, owns nothing, and is safe to destroy.
    BINCV_CHECK(movedFrom.data() == nullptr);
    BINCV_CHECK(movedFrom.empty());
    BINCV_CHECK(!movedFrom.ownsMemory());

    //...and safe to assign to again.
    movedFrom = S(2);
    BINCV_CHECK(movedFrom.ownsMemory());
    BINCV_CHECK_EQ(movedFrom.size(), size_t(2));
    movedFrom.data()[1] = W(6);
    BINCV_CHECK(movedFrom.data()[1] == W(6));

    // --- move assignment ---

    S moveDst(1);
    moveDst = std::move(movedTo);
    BINCV_CHECK(moveDst.data() == movedPtr);
    BINCV_CHECK(moveDst.ownsMemory());
    BINCV_CHECK_EQ(moveDst.size(), size_t(5));
    BINCV_CHECK(moveDst.data()[0] == W(11));
    BINCV_CHECK(movedTo.data() == nullptr);
    BINCV_CHECK(movedTo.empty());
    BINCV_CHECK(!movedTo.ownsMemory());

    // Moving a non-owning Storage transfers the alias, not ownership.
    S aliasSrc(buf, 8);
    S aliasDst(std::move(aliasSrc));
    BINCV_CHECK(aliasDst.data() == buf);
    BINCV_CHECK(!aliasDst.ownsMemory());
    BINCV_CHECK(aliasSrc.empty());

    // Self move-assignment must leave the object intact rather than freeing it.
    S& moveDstRef = moveDst;
    moveDst = std::move(moveDstRef);
    BINCV_CHECK(moveDst.data() == movedPtr);
    BINCV_CHECK_EQ(moveDst.size(), size_t(5));
    BINCV_CHECK(moveDst.ownsMemory());
    BINCV_CHECK(moveDst.data()[0] == W(11));

    // const access yields const words; non-const access does not.
    static_assert(std::is_same<decltype(std::declval<const S&>().data()), const W*>::value,
                  "const Storage::data() must return const WordType*");
    static_assert(std::is_same<decltype(std::declval<S&>().data()), W*>::value,
                  "Storage::data() must return WordType*");
}

// Every allocation Storage does or does not perform, counted rather than argued.
void testAllocationDiscipline() {
    using S = bincv::Storage<uint32_t>;
    std::cout << "\n--- Storage allocation discipline ---\n";

    // Counters are read as deltas around each region, and the delta is captured
    // into a local before anything is asserted: the check macros allocate
    // (std::to_string) and evaluate their argument more than once, so checking
    // the live counter would measure the check.
    uint32_t buf[32] = {};
    size_t news = 0;
    size_t deletes = 0;

    size_t newsBefore = g_newCount;
    size_t deletesBefore = g_deleteCount;
    {
        S wrapped(buf, 32);
        escape(wrapped.data());
        S copiedAlias(wrapped);           // copy of non-owning: alias, no allocation
        escape(copiedAlias.data());
        S assigned;
        assigned = copiedAlias;           // assignment from non-owning: likewise
        escape(assigned.data());
        S moved(std::move(assigned));
        escape(moved.data());
        wrapped.data()[0] = 0xDEADBEEFu;  // and it is really the caller's buffer
    }
    news = g_newCount - newsBefore;
    deletes = g_deleteCount - deletesBefore;
    BINCV_CHECK_EQ(news, size_t(0));      // the Tier 2 claim, measured
    BINCV_CHECK_EQ(deletes, size_t(0));
    BINCV_CHECK_EQ(buf[0], 0xDEADBEEFu);

    // One owning object: exactly one allocation, released exactly once.
    newsBefore = g_newCount;
    deletesBefore = g_deleteCount;
    {
        S owned(64);
        escape(owned.data());
    }
    news = g_newCount - newsBefore;
    deletes = g_deleteCount - deletesBefore;
    BINCV_CHECK_EQ(news, size_t(1));
    BINCV_CHECK_EQ(deletes, size_t(1));

    // Copying an owning object allocates a second buffer (deep copy).
    newsBefore = g_newCount;
    deletesBefore = g_deleteCount;
    {
        S owned(8);
        S copied(owned);
        escape(owned.data());
        escape(copied.data());
    }
    news = g_newCount - newsBefore;
    deletes = g_deleteCount - deletesBefore;
    BINCV_CHECK_EQ(news, size_t(2));
    BINCV_CHECK_EQ(deletes, size_t(2));

    // Moving an owning object allocates nothing and frees nothing extra: one
    // buffer is created and one is released, however many objects it passes
    // through.
    newsBefore = g_newCount;
    deletesBefore = g_deleteCount;
    {
        S a(8);
        S b(std::move(a));
        escape(b.data());
        S c;
        c = std::move(b);
        escape(c.data());
    }
    news = g_newCount - newsBefore;
    deletes = g_deleteCount - deletesBefore;
    BINCV_CHECK_EQ(news, size_t(1));
    BINCV_CHECK_EQ(deletes, size_t(1));

    // Copy-assigning an owning source over an owning target: two buffers exist
    // going in, a third is allocated for the copy, and the target's old one is
    // released -- three allocations, three releases across the block.
    newsBefore = g_newCount;
    deletesBefore = g_deleteCount;
    {
        S a(8);
        S b(4);
        b = a;
        escape(a.data());
        escape(b.data());
    }
    news = g_newCount - newsBefore;
    deletes = g_deleteCount - deletesBefore;
    BINCV_CHECK_EQ(news, size_t(3));
    BINCV_CHECK_EQ(deletes, size_t(3));

    // Empty storage never touches the allocator, however it is handled.
    newsBefore = g_newCount;
    deletesBefore = g_deleteCount;
    {
        S emptyStorage;
        S zeroWords(0);
        S copiedEmpty(emptyStorage);
        S movedEmpty(std::move(copiedEmpty));
        escape(zeroWords.data());
        escape(movedEmpty.data());
    }
    news = g_newCount - newsBefore;
    deletes = g_deleteCount - deletesBefore;
    BINCV_CHECK_EQ(news, size_t(0));
    BINCV_CHECK_EQ(deletes, size_t(0));
}

// ---------------------------------------------------------------------------
// Word-type constraints
//
// REGRESSION: the constraint used to be `is_integral && is_unsigned` alone, which
// admits three things it should not:
// - `const`/`volatile` word types, so BinMatView<const uint32_t> instantiated --
// exactly the const-templated view the design rule forbids, and a fourth view type that
// kernels would silently be instantiated on
// - `bool`, for which WordBits = sizeof * 8 is a lie: the object holds one
// usable bit, so bit packing would discard seven pixels in eight
// - plain `char`, which is unsigned on some targets and signed on others, so
// the accepted type set differed between the project's two named tiers
//
// A failed static_assert is a hard error, so these cannot be exercised from a
// running test. Build with -DBINCV_TEST_MUST_NOT_COMPILE to check that each one
// is still rejected; every line below must fail to compile.
// ---------------------------------------------------------------------------

#ifdef BINCV_TEST_MUST_NOT_COMPILE
namespace mustNotCompile {
bincv::Storage<const uint32_t>      cvQualifiedStorage;
bincv::BinMatView<const uint32_t>   cvQualifiedView;
bincv::BinMatConstView<const uint32_t> cvQualifiedConstView;
bincv::Storage<bool>                boolStorage;
bincv::BinMatView<bool>             boolView;
bincv::BinMatConstView<bool>        boolConstView;
bincv::BinMatView<char>             charView;
} // namespace mustNotCompile
#endif

// ---------------------------------------------------------------------------
// -- views
// ---------------------------------------------------------------------------

// A kernel-shaped free function: takes a read-only view by value, which is how
// every kernel in this library will be declared. Sums the first word of each row.
template <typename W>
size_t sumFirstWords(bincv::BinMatConstView<W> src) {
    size_t total = 0;
    for (size_t y = 0; y < src.height; ++y) total += static_cast<size_t>(src.row(y)[0]);
    return total;
}

// Layout and type-identity guarantees. Almost all of this is static_assert: if it
// compiles, it holds -- the runtime checks exist so the suite reports on it.
void testViewTypeProperties() {
    using V = bincv::BinMatView<uint32_t>;
    using CV = bincv::BinMatConstView<uint32_t>;
    std::cout << "\n--- View type properties ---\n";

    // Two distinct types, not one templated on constness.
    static_assert(!std::is_same<V, CV>::value, "views must be two distinct types");

    // Non-owning and trivially copyable: passing a view by value is a register
    // copy, and no lifetime management rides along.
    static_assert(std::is_trivially_copyable<V>::value, "BinMatView must be trivially copyable");
    static_assert(std::is_trivially_copyable<CV>::value, "BinMatConstView must be trivially copyable");
    static_assert(std::is_trivially_destructible<V>::value, "BinMatView must be trivially destructible");
    static_assert(std::is_trivially_destructible<CV>::value, "BinMatConstView must be trivially destructible");
    static_assert(std::is_standard_layout<V>::value, "BinMatView must be standard layout");
    static_assert(std::is_standard_layout<CV>::value, "BinMatConstView must be standard layout");

    // Pointer plus three size_t, in declaration order, with no padding surprises.
    static_assert(sizeof(V) == sizeof(void*) + 3 * sizeof(size_t), "unexpected BinMatView layout");
    static_assert(sizeof(CV) == sizeof(void*) + 3 * sizeof(size_t), "unexpected BinMatConstView layout");
    BINCV_CHECK_EQ(sizeof(V), sizeof(void*) + 3 * sizeof(size_t));
    BINCV_CHECK_EQ(sizeof(CV), sizeof(void*) + 3 * sizeof(size_t));
    BINCV_CHECK_EQ(offsetof(V, ptr), size_t(0));
    BINCV_CHECK_EQ(offsetof(V, width), sizeof(void*));
    BINCV_CHECK_EQ(offsetof(V, height), sizeof(void*) + sizeof(size_t));
    BINCV_CHECK_EQ(offsetof(V, stride), sizeof(void*) + 2 * sizeof(size_t));
    BINCV_CHECK_EQ(offsetof(CV, ptr), size_t(0));
    BINCV_CHECK_EQ(offsetof(CV, stride), sizeof(void*) + 2 * sizeof(size_t));

    // WordBits derives from the word type, not from a separate parameter.
    static_assert(bincv::BinMatView<uint8_t>::WordBits == 8, "");
    static_assert(bincv::BinMatView<uint16_t>::WordBits == 16, "");
    static_assert(bincv::BinMatView<uint32_t>::WordBits == 32, "");
    static_assert(bincv::BinMatView<uint64_t>::WordBits == 64, "");
    static_assert(bincv::BinMatConstView<uint8_t>::WordBits == 8, "");
    static_assert(bincv::BinMatConstView<uint64_t>::WordBits == 64, "");
    BINCV_CHECK_EQ(bincv::BinMatView<uint64_t>::WordBits, size_t(64));

    // Mutable converts to const implicitly; const never converts back.
    static_assert(std::is_convertible<V, CV>::value, "BinMatView must convert to BinMatConstView");
    static_assert(!std::is_convertible<CV, V>::value, "BinMatConstView must not convert to BinMatView");

    // A const view object still hands out const words from a mutable view.
    static_assert(std::is_same<decltype(std::declval<V&>().row(0)), uint32_t*>::value,
                  "BinMatView::row() must return WordType*");
    static_assert(std::is_same<decltype(std::declval<const V&>().row(0)), const uint32_t*>::value,
                  "const BinMatView::row() must return const WordType*");
    static_assert(std::is_same<decltype(std::declval<const CV&>().row(0)), const uint32_t*>::value,
                  "BinMatConstView::row() must return const WordType*");
}

// Read and write through a view over a raw stack buffer, at every word width.
template <typename W>
void testView(const char* label) {
    using V = bincv::BinMatView<W>;
    using CV = bincv::BinMatConstView<W>;
    std::cout << "\n--- BinMatView<" << label << "> (" << V::WordBits << " bits/word) ---\n";

    // Width spills one pixel into a second word, and the stride carries a third
    // word of slack: a kernel that ignores stride, or assumes it equals the words
    // a row needs, reads the wrong row here.
    constexpr size_t height = 4;
    constexpr size_t stride = 3;
    const size_t width = V::WordBits + 1;
    W buf[height * stride] = {};

    V v{buf, width, height, stride};
    BINCV_CHECK(!v.empty());
    BINCV_CHECK(v.ptr == buf);
    BINCV_CHECK_EQ(v.width, width);
    BINCV_CHECK_EQ(v.height, height);
    BINCV_CHECK_EQ(v.stride, stride);

    // row(y) is exactly ptr + y * stride.
    for (size_t y = 0; y < height; ++y) BINCV_CHECK(v.row(y) == buf + y * stride);

    // Writing through the view lands in the caller's buffer, at the stride the
    // view declares, leaving the slack word untouched.
    for (size_t y = 0; y < height; ++y) {
        v.row(y)[0] = static_cast<W>(y + 1);
        v.row(y)[1] = W(1);   // the word holding the pixel past WordBits
    }
    bool laidOutCorrectly = true;
    for (size_t y = 0; y < height; ++y) {
        if (buf[y * stride + 0] != static_cast<W>(y + 1)) laidOutCorrectly = false;
        if (buf[y * stride + 1] != W(1)) laidOutCorrectly = false;
        if (buf[y * stride + 2] != W(0)) laidOutCorrectly = false;
    }
    BINCV_CHECK(laidOutCorrectly);

    // Individual bits round-trip through the packed words, LSB first.
    v.row(2)[0] = W(0);
    v.row(2)[0] |= static_cast<W>(W(1) << 3);       // column 3
    v.row(2)[1] |= W(1);                            // column WordBits
    BINCV_CHECK(((buf[2 * stride] >> 3) & W(1)) != 0);
    BINCV_CHECK((buf[2 * stride + 1] & W(1)) != 0);

    // Implicit conversion to the read-only view carries every field across.
    CV cv = v;
    BINCV_CHECK(cv.ptr == buf);
    BINCV_CHECK_EQ(cv.width, width);
    BINCV_CHECK_EQ(cv.height, height);
    BINCV_CHECK_EQ(cv.stride, stride);
    BINCV_CHECK(!cv.empty());
    for (size_t y = 0; y < height; ++y) BINCV_CHECK(cv.row(y) == buf + y * stride);

    // The const view sees writes made through the mutable one -- it is the same
    // memory, not a copy.
    v.row(0)[0] = W(0x2A);
    BINCV_CHECK(cv.row(0)[0] == W(0x2A));

    // Conversion also happens at a call boundary. The explicit template argument
    // is required: deduction does not apply user-defined conversions.
    v.row(0)[0] = W(1);
    v.row(1)[0] = W(2);
    v.row(2)[0] = W(3);
    v.row(3)[0] = W(4);
    BINCV_CHECK_EQ(sumFirstWords<W>(v), size_t(10));
    BINCV_CHECK_EQ(sumFirstWords<W>(cv), size_t(10));

    // A view over a const buffer is expressible, and only as the const type.
    const W* constBuf = buf;
    CV overConst{constBuf, width, height, stride};
    BINCV_CHECK_EQ(sumFirstWords<W>(overConst), size_t(10));

    // Emptiness: a default view addresses nothing, and so does any zero extent.
    V defaulted{};
    BINCV_CHECK(defaulted.empty());
    BINCV_CHECK(defaulted.ptr == nullptr);
    BINCV_CHECK_EQ(defaulted.stride, size_t(0));
    CV defaultedConst{};
    BINCV_CHECK(defaultedConst.empty());
    V zeroWidth{buf, 0, height, stride};
    BINCV_CHECK(zeroWidth.empty());
    V zeroHeight{buf, width, 0, stride};
    BINCV_CHECK(zeroHeight.empty());

    // REGRESSION boundary for the debug-only non-zero-stride precondition on
    // row. A stride of zero is legitimate at height <= 1 -- there is no second
    // row for it to collide with -- so the assertion must not fire here. It is
    // the multi-row case that row rejects in debug builds; that one aborts, so
    // it cannot be exercised from a running test until the harness grows death
    // tests.
    V singleRow{buf, width, 1, 0};
    BINCV_CHECK(!singleRow.empty());
    singleRow.row(0)[0] = W(0x15);
    BINCV_CHECK(buf[0] == W(0x15));
    CV singleRowConst = singleRow;
    BINCV_CHECK(singleRowConst.row(0)[0] == W(0x15));
    V noRows{buf, width, 0, 0};
    BINCV_CHECK(noRows.empty());

    // Restore what the checks below expect.
    v.row(0)[0] = W(1);

    // Views are values: copying one copies the descriptor, never the pixels.
    V copiedView = v;
    BINCV_CHECK(copiedView.ptr == v.ptr);
    copiedView.row(3)[0] = W(99);
    BINCV_CHECK(v.row(3)[0] == W(99));
}

// Storage owns, the view borrows -- the pairing every container will use.
void testViewOverStorage() {
    std::cout << "\n--- View over Storage ---\n";

    // 40 pixels needs two 32-bit words per row; three rows, word granularity.
    const size_t width = 40;
    const size_t height = 3;
    const size_t stride = 2;

    bincv::Storage<uint32_t> storage(stride * height);
    bincv::BinMatView<uint32_t> v{storage.data(), width, height, stride};
    BINCV_CHECK(!v.empty());
    BINCV_CHECK(v.row(2) == storage.data() + 4);

    // Writes through the view are writes into the storage words.
    v.row(2)[1] = 0xFFu;
    BINCV_CHECK_EQ(storage.data()[5], 0xFFu);
    BINCV_CHECK_EQ(storage.data()[0], 0u);

    // A const view over const storage reads the same words.
    const bincv::Storage<uint32_t>& constStorage = storage;
    bincv::BinMatConstView<uint32_t> cv{constStorage.data(), width, height, stride};
    BINCV_CHECK_EQ(cv.row(2)[1], 0xFFu);

    // A deep copy of the storage is a different buffer, so the original view must
    // not be pointing into it -- this is seen from the view side.
    bincv::Storage<uint32_t> deepCopy(storage);
    BINCV_CHECK(deepCopy.data() != storage.data());
    BINCV_CHECK_EQ(deepCopy.data()[5], 0xFFu);
    v.row(2)[1] = 0x11u;
    BINCV_CHECK_EQ(deepCopy.data()[5], 0xFFu);

    // The same view type binds to caller-provided memory with no change at all to
    // the kernel-facing interface -- that is the point of views.
    uint32_t raw[stride * height] = {};
    bincv::Storage<uint32_t> wrapped(raw, stride * height);
    bincv::BinMatView<uint32_t> rawView{wrapped.data(), width, height, stride};
    rawView.row(1)[0] = 0x77u;
    BINCV_CHECK_EQ(raw[2], 0x77u);
    BINCV_CHECK(!wrapped.ownsMemory());
}

} // namespace

BINCV_TEST(Storage, Contract_uint8_t)  { testStorage<uint8_t>("uint8_t"); }
BINCV_TEST(Storage, Contract_uint16_t) { testStorage<uint16_t>("uint16_t"); }
BINCV_TEST(Storage, Contract_uint32_t) { testStorage<uint32_t>("uint32_t"); }
BINCV_TEST(Storage, Contract_uint64_t) { testStorage<uint64_t>("uint64_t"); }
BINCV_TEST(Storage, AllocationDiscipline) { testAllocationDiscipline(); }

BINCV_TEST(View, TypeProperties)   { testViewTypeProperties(); }
BINCV_TEST(View, Contract_uint8_t)  { testView<uint8_t>("uint8_t"); }
BINCV_TEST(View, Contract_uint16_t) { testView<uint16_t>("uint16_t"); }
BINCV_TEST(View, Contract_uint32_t) { testView<uint32_t>("uint32_t"); }
BINCV_TEST(View, Contract_uint64_t) { testView<uint64_t>("uint64_t"); }
BINCV_TEST(View, OverStorage)      { testViewOverStorage(); }

BINCV_TEST_MAIN("Storage and view core tests")
