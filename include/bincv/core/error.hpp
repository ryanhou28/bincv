#pragma once

#include <cstdio>
#include <cstdlib>

/// @file error.hpp
/// @brief The project's single error policy (the design notes).
///
/// Two macros, and the split between them is the whole policy:
///
/// BINCV_THROW(ExceptionType, "message") -- VALIDATION. Constructor and
/// argument checking, called at setup rather than per pixel. Throws by
/// default; writes the message to stderr and aborts when exceptions are
/// unavailable.
///
/// BINCV_ASSERT(cond, "message") -- HOT PATH. Element access and
/// kernel preconditions. Checked in debug builds, compiled away entirely
/// under NDEBUG -- no branch, and the condition is not even parsed.
///
/// @note Why not std::assert for the second one: assert reports only the
/// stringified condition, which for a packed bit container reads as an
/// expression about words and strides rather than as the mistake the
/// caller made. This wrapper prints both the condition and a written
/// message, and otherwise does exactly what assert does.
/// @note Why the abort path exists at all: Tier 2 targets build with exceptions
/// disabled (the design notes), and a library that can only report errors by
/// throwing cannot be built for them at all. Aborting is not a graceful
/// answer, but it is a defined one, and it keeps the validation checks in
/// the code rather than compiled out of the configuration that can least
/// afford to skip them.
///
/// @note This header is also where the project's COMPILATION-MODE macros live --
/// the ones every other header has to agree about, each detected from the
/// compiler rather than asserted by a build system: BINCV_EXCEPTIONS_ENABLED,
/// BINCV_DEBUG_CHECKS, BINCV_ABI_NAMESPACE and BINCV_HOST_DEVICE. They are
/// here because this is the one header every binCV header already includes
/// -- opening BINCV_ABI_NAMESPACE requires it -- so a macro defined here is
/// visible everywhere by construction rather than by an include somebody
/// has to remember.

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// @def BINCV_EXCEPTIONS_ENABLED
/// @brief 1 when a `throw` will compile, 0 when it will not.
///
/// @note Detected from the compiler rather than left to the user. Building with
/// -fno-exceptions and forgetting to also define BINCV_NO_EXCEPTIONS would
/// otherwise fail with an error per throw site, which is exactly the state
/// this file was written to fix. Defining BINCV_NO_EXCEPTIONS by hand still
/// works, and still wins: it forces the abort path even where `throw` would
/// have compiled, which is how the Tier 2 behavior is exercised on a
/// desktop toolchain.
/// @note Set it for the WHOLE program if you set it at all. Like NDEBUG it is a
/// per-translation-unit macro, and it changes the body of every function
/// containing a check, so a program with both kinds of object in it has two
/// definitions of each. BINCV_ABI_NAMESPACE below keeps that from resolving
/// silently -- each object gets the behavior it was compiled for, and a
/// genuinely shared interface fails to link rather than misbehaving -- but
/// a half-converted program is still not a configuration anyone wants.
/// @note __cpp_exceptions is the standard feature-test macro; __EXCEPTIONS
/// (GCC/Clang) and _CPPUNWIND (MSVC) cover toolchains that predate it.
#if defined(BINCV_NO_EXCEPTIONS)
#  define BINCV_EXCEPTIONS_ENABLED 0
#elif defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
#  define BINCV_EXCEPTIONS_ENABLED 1
#else
#  define BINCV_EXCEPTIONS_ENABLED 0
// Published so that code outside this header can test one spelling, whether the
// no-exceptions build was requested explicitly or detected from the compiler.
#  define BINCV_NO_EXCEPTIONS 1
#endif

/// @def BINCV_DEBUG_CHECKS
/// @brief 1 when BINCV_ASSERT is live, 0 when it compiles away. Keyed to NDEBUG.
///
/// @note There is no binCV-specific override, but that does NOT make the switch
/// whole-program: NDEBUG is itself a per-translation-unit macro, so two
/// objects of the same program can disagree about it. binCV is header-only,
/// and every function that reads this macro is an inline or template one,
/// so disagreement means the same function has two definitions. Which one
/// survives is otherwise the linker's choice, and the symptom is bounds
/// checks that appear to vanish. BINCV_ABI_NAMESPACE below is what keeps
/// that from happening silently -- see there.
#if defined(NDEBUG)
#  define BINCV_DEBUG_CHECKS 0
#else
#  define BINCV_DEBUG_CHECKS 1
#endif

/// @def BINCV_ABI_NAMESPACE
/// @brief Inline namespace naming the configuration this translation unit was
/// compiled in. Every binCV header opens it; users never spell it.
///
/// @note This exists because the two macros above change the BODY of inline and
/// template functions -- at's bounds check, every BINCV_THROW site's
/// throw-versus-abort -- while binCV is header-only, so those bodies are
/// emitted into whichever objects use them. Two objects compiled with
/// different NDEBUG or different BINCV_NO_EXCEPTIONS define the same symbol
/// differently, which is an ODR violation; the linker keeps one arbitrarily
/// and the loser silently gets the other configuration's behavior, chosen
/// by link order. Measured before this namespace existed: linking a release
/// object ahead of a debug one made an out-of-range set in the DEBUG
/// object return quietly and set a padding bit; reversing the link order
/// made the same line abort. The same experiment with BINCV_NO_EXCEPTIONS
/// made a `catch` in an exceptions-enabled object never run.
/// @note Encoding the configuration in the mangled name is the whole mechanism.
/// Each object then instantiates and calls the definition it was compiled
/// for -- no coin flip -- and any interface a mismatched pair really does
/// share becomes an undefined symbol naming both configurations, which is a
/// link error a reader can act on rather than a behavior change nobody
/// sees. It is transparent otherwise: `bincv::BinMat` still names the type.
/// @note Written out as four cases rather than pasted together from the two
/// macros, so the name that appears in a linker diagnostic is greppable.
#if BINCV_DEBUG_CHECKS
#  if BINCV_EXCEPTIONS_ENABLED
#    define BINCV_ABI_NAMESPACE v1_checked_throwing
#  else
#    define BINCV_ABI_NAMESPACE v1_checked_aborting
#  endif
#else
#  if BINCV_EXCEPTIONS_ENABLED
#    define BINCV_ABI_NAMESPACE v1_unchecked_throwing
#  else
#    define BINCV_ABI_NAMESPACE v1_unchecked_aborting
#  endif
#endif

// ---------------------------------------------------------------------------
// BINCV_HOST_DEVICE -- one scalar helper, compiled for two targets
// ---------------------------------------------------------------------------

/// @def BINCV_HOST_DEVICE
/// @brief Marks a SCALAR helper as compilable for a GPU as well as a CPU:
/// `__host__ __device__` when nvcc is the compiler, and NOTHING otherwise.
///
/// @note WHAT IT IS FOR. backends/cuda/ shares the REPRESENTATION and forks the
/// KERNELS, because a device traversal has nothing in common with a row
/// loop. A scalar helper is not a traversal: `impl::clipRegion`,
/// `impl::borderIndex`, `maj3`, `thresholdGE` and `impl::minEigenValue` are
/// closed-form arithmetic over a handful of integers, with no loop over
/// pixels and no memory of their own. Before this macro the backend had to
/// re-derive such a rule in device code by hand, and a second derivation of
/// one rule is the failure this project keeps finding: the copy that drifts
/// does not crash, it answers a plausible question that nobody asked, and
/// the map comes back subtly wrong. This lets the rule be ONE definition
/// compiled twice, so there is nothing to keep in agreement.
///
/// @note WHAT IT DOES NOT LICENSE. It is not a door to a shared kernel, and it
/// is not a device-side spelling of the host API. A function is a candidate
/// only when it is scalar and traversal-free -- no loop over pixels, rows or
/// words, no pointer into an image, no allocation. Anything that walks an
/// image stays forked, because that is where the host's row-major, popcount,
/// cache-line shape and the device's warp shape genuinely disagree, and
/// pretending otherwise costs the performance the backend exists for.
/// Annotating a traversal would also quietly make a host view type usable
/// from a kernel, which is the one thing the device-typed views exist to
/// prevent.
///
/// @note IT PULLS NOTHING IN. The expansion is decided by `__CUDACC__`, which
/// only nvcc (and clang in CUDA mode) defines, and the annotations it
/// expands to are that compiler's own keywords. No CUDA header is included
/// here, none is required, and on every other compiler the macro expands to
/// an empty token sequence -- so an annotated helper is byte-for-byte the
/// function it was before. binCV remains header-only, dependency-free and
/// buildable by a plain C++17 compiler with no CUDA installed; that property
/// is checked rather than assumed, by a static_assert in tests/test_error.cpp
/// that the expansion here really is empty (and its mirror in the CUDA suite
/// that it really is not, under nvcc).
///
/// @note BINCV_ASSERT WORKS INSIDE ONE. `detail::assertFailed` below has a device
/// branch for exactly this reason: a helper whose preconditions vanish when
/// it is compiled for the device would be a *different* helper, which is the
/// drift this macro exists to remove. BINCV_THROW does NOT work inside one,
/// deliberately -- see the note on `throwFailed`.
#if defined(__CUDACC__)
#  define BINCV_HOST_DEVICE __host__ __device__
#else
#  define BINCV_HOST_DEVICE
#endif

// The default BINCV_THROW expansion constructs the exception type the caller
// names, so that type has to be complete at every call site. Every one in the
// library names a <stdexcept> type, and a caller that reports through its own
// type includes whatever declares it. Guarded because the abort path constructs
// nothing: a Tier 2 target is not made to depend on the exception hierarchy, nor
// to have a <stdexcept> at all.
#if BINCV_EXCEPTIONS_ENABLED
#  include <stdexcept>
#endif

// Deliberately NOT inside BINCV_ABI_NAMESPACE, unlike every other binCV
// declaration. These two have the same definition in all four configurations --
// they print and abort, and read no configuration macro -- so merging them
// across objects is correct rather than hazardous. Versioning them would only
// duplicate them, and the macros below name them absolutely, so nothing about
// the lookup depends on which namespace the caller sits in.
namespace bincv {
namespace detail {

/// @brief Reports a failed validation check and aborts. Never returns.
/// @note Internal. Reached only through BINCV_THROW, and only in builds without
/// exceptions.
/// @note stdio rather than iostream: this has to work in a configuration that
/// has already given up exceptions, where pulling in the iostream static
/// initializers to print one line is the wrong trade.
/// @note HOST ONLY, unlike assertFailed below, and that is the split BINCV_THROW
/// already draws: it reports a SETUP-time validation failure -- a bad
/// dimension, a null buffer, a CUDA call that failed -- and setup happens on
/// the host. A device helper that has to refuse an input returns an error
/// instead of reporting one, which is the shape the backend's launchers
/// already use. So there is no device branch here, and a BINCV_THROW inside
/// a BINCV_HOST_DEVICE function is a compile error naming this function --
/// which is the right answer, not a gap.
[[noreturn]] inline void throwFailed(const char* message, const char* file, int line) {
    std::fprintf(stderr, "bincv: fatal error: %s\n at %s:%d\n", message, file, line);
    std::fflush(stderr);
    std::abort();
}

/// @brief Reports a failed BINCV_ASSERT and aborts. Never returns.
/// @note Internal. The condition text and the written message are passed
/// separately so that neither has to be a literal the other is
/// concatenated onto.
/// @note THE DEVICE BRANCH IS WHAT MAKES BINCV_ASSERT LEGAL IN A
/// BINCV_HOST_DEVICE HELPER. `std::fprintf` and `std::abort` are host
/// functions, and nvcc rejects a call to one from code it is compiling for
/// the device -- so without this branch every shared scalar helper would
/// have to drop its preconditions to cross over, and a helper whose checks
/// vanish on one of its two targets is two helpers again. `printf` is the
/// device's own (it reaches the host through the printf buffer) and
/// `__trap()` is the device's abort: it kills the kernel and leaves the
/// context in an error state, so the failure is not survivable and not
/// silent, which is what the host branch's abort() buys too.
/// @note `__CUDA_ARCH__` rather than `__CUDACC__`: nvcc compiles a .cu file at
/// least twice, and only the DEVICE passes define this. The host pass of a
/// .cu file takes the fprintf branch, exactly like every .cpp.
[[noreturn]] BINCV_HOST_DEVICE inline void assertFailed(const char* expr,
                                                        const char* message,
                                                        const char* file, int line) {
#if defined(__CUDA_ARCH__)
    std::printf("bincv: assertion failed: %s\n condition: %s\n at %s:%d\n", message, expr,
                file, line);
    __trap();
#else
    std::fprintf(stderr, "bincv: assertion failed: %s\n condition: %s\n at %s:%d\n",
                 message, expr, file, line);
    std::fflush(stderr);
    std::abort();
#endif
}

} // namespace detail
} // namespace bincv

// ---------------------------------------------------------------------------
// BINCV_THROW -- validation
// ---------------------------------------------------------------------------

/// @def BINCV_THROW(ExceptionType, message)
/// @brief Reports a validation failure: throws `ExceptionType(message)`, or
/// prints `message` to stderr and aborts where exceptions are disabled.
///
/// @note An expression, not a statement, so it composes the same way `throw`
/// does and needs no trailing-semicolon dance at the call site.
/// @note `ExceptionType` is dropped entirely on the abort path -- deliberately.
/// A Tier 2 build must not be made to depend on the exception hierarchy
/// being available, only on the diagnostic text.
/// @note Consequence, and it is a real one: in a build without exceptions the
/// type argument is not compiled at all, so a misspelled or non-exception
/// type is not diagnosed there. It is diagnosed by the default build, which
/// is why the project's verification runs all three configurations rather
/// than only the one a change was written for (CLAUDE.md). Type-checking it
/// in both would mean naming the type in an unevaluated expression --
/// `sizeof(ExceptionType(message))` -- which reinstates exactly the
/// dependency the previous note refuses: the type would have to be complete
/// and constructible on a target that may have no <stdexcept>.
/// @note This is for setup-time checking only. Anything on a per-pixel path
/// belongs in BINCV_ASSERT; see the design notes.
#if BINCV_EXCEPTIONS_ENABLED
#  define BINCV_THROW(ExceptionType, message) throw ExceptionType(message)
#else
#  define BINCV_THROW(ExceptionType, message) \
      ::bincv::detail::throwFailed((message), __FILE__, __LINE__)
#endif

// ---------------------------------------------------------------------------
// BINCV_ASSERT -- hot-path precondition
// ---------------------------------------------------------------------------

/// @def BINCV_ASSERT(cond, message)
/// @brief Checks a precondition in debug builds; expands to nothing under NDEBUG.
///
/// @note Under NDEBUG the expansion mentions neither argument, so `cond` is not
/// evaluated, not branched on, and not even parsed. That last part is what
/// lets a release build inline BinMat::at down to a shift and a mask,
/// and it is verified by tests/test_error.cpp, which passes a type with no
/// conversion to bool as the condition -- the release build compiles only
/// because the token is discarded.
/// @note The cost of that is the usual one: an expression used only inside a
/// BINCV_ASSERT is unused in release. Compute it into the assert, or mark
/// it (void).
/// @note Exactly two macro arguments, in both configurations. A condition with a
/// top-level comma -- `std::is_same<A, B>::value`, the normal shape of a
/// kernel precondition over template parameters -- is split by the
/// preprocessor into two arguments and fails with "passed 3 arguments, but
/// takes just 2", which does not hint at the cause. Wrap it:
/// `BINCV_ASSERT((std::is_same<A, B>::value), "matching word types")`. The
/// parentheses hide the comma from the preprocessor and change nothing
/// else. Deliberately not made variadic: splitting a trailing message off a
/// variable argument list needs argument-counting macros, and the release
/// expansion has to discard every token it is handed, so the two forms
/// would have to agree about an arity neither of them uses.
#if BINCV_DEBUG_CHECKS
#  define BINCV_ASSERT(cond, message)                                          \
      ((cond) ? static_cast<void>(0)                                           \
              : ::bincv::detail::assertFailed(#cond, (message), __FILE__, __LINE__))
#else
#  define BINCV_ASSERT(cond, message) (static_cast<void>(0))
#endif
