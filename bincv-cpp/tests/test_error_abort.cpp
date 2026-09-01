// Deliberately fails one validation check, and does not survive it.
//
// This is the only way to cover BINCV_THROW's abort path: in a build without
// exceptions the macro prints and calls std::abort, which no same-process
// assertion can observe. So the check lives in the process exit instead --
// tests/CMakeLists.txt registers one ctest test per case below, each driven by
// tests/expect_fatal.cmake, which requires that the child died AND that it named
// the reason.
//
// The same test is meaningful in both configurations, which is why the match is
// on the message and not on the mechanism:
// exceptions on -- the uncaught std::invalid_argument terminates the process,
// and the runtime prints what on the way out
// exceptions off -- BINCV_THROW prints "bincv: fatal error: <message>" itself
// and aborts
// Either way: a bad argument is fatal, and it names the reason before dying. In
// the no-exceptions build that is the entire error-reporting contract.
//
// WHY EVERY SITE AND NOT JUST ONE: BINCV_CHECK_THROWS cannot evaluate its
// expression without exceptions, so the in-process suites skip 44 validation
// checks in that configuration. Measured before these cases existed: deleting
// the strideWords check (and the at/set bounds asserts) from binMat_impl.hpp
// left `-fno-exceptions` reporting 4/4 tests and 801/801 checks passed. The
// configuration exists for could not fail. One case per BINCV_THROW site is
// what closes that, and it closes it in all three configurations at once.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/quantMat.hpp"

namespace {

using Mat = bincv::BinMat<uint32_t>;

// Storage for the wrapping cases. File scope so that no case can be blamed on a
// dangling stack buffer.
uint32_t g_buffer[8] = {0};

// Every case returns something derived from the object it built, and main
// prints it. A case that stops being fatal therefore cannot be optimized into
// nothing -- it has to produce a value, and printing that value is what the
// "REACHED" marker reports.
int caseAlignmentPowerOfTwo() { Mat m(4, 4, 6); return m.cols(); }
int caseAlignmentWordMultiple() { Mat m(4, 4, 2); return m.cols(); }
int caseNegativeDims() { Mat m(-1, 10); return m.cols(); }
int caseWrapNegativeDims() { Mat m(g_buffer, -1, 2, 2); return m.cols(); }
int caseWrapShortStride() { Mat m(g_buffer, 64, 2, 1); return m.cols(); }
int caseWrapNullPointer() { Mat m(nullptr, 8, 2, 1); return m.cols(); }

// QuantMat's only validation of its own: BinMat takes int dimensions, and an
// N-fold plane stack that no longer fits one would otherwise be truncated into a
// plausible smaller matrix rather than reported. Rejected before anything is
// allocated, so the case costs nothing to run.
int caseQuantHeightOverflow() {
    bincv::QuantMat<8, uint32_t> m(1, 300000000);
    return m.cols();
}

// --- Plane indices ---------------------------------------------
//
// These are validation, not element access, so they are checked in EVERY build
// and belong here rather than in test_assert_abort.cpp. What an unchecked one
// did, measured in Release before the fix: QuantMat<3>::plane(3) returned a
// fully-formed writable view of the next allocation (37.5 KiB of heap overflow
// at 640x480), BinMat::plane(7) quietly returned plane 0, and
// SignedQuantMat<2>::magnitude(2) returned the SIGN plane as a magnitude plane --
// three different outcomes for one identical caller mistake, and the third
// invisible to every sanitizer because the address is valid storage.
//
// The const overloads carry their own copies of the check, so each is driven
// separately; a check tested only through the mutable overload is not tested.
using Quant = bincv::QuantMat<3, uint32_t>;
using Signed = bincv::SignedQuantMat<2, uint32_t>;

int caseQuantPlaneIndex() {
    Quant m(20, 4);
    return static_cast<int>(*m.plane(3).ptr);          // planes are 0..2
}

int caseQuantPlaneIndexConst() {
    const Quant m(20, 4);
    return static_cast<int>(*m.plane(3).ptr);
}

int caseBinMatPlaneIndex() {
    Mat m(20, 4);
    return static_cast<int>(*m.plane(1).ptr);          // BinMat is QuantMat<1>
}

int caseBinMatPlaneIndexConst() {
    const Mat m(20, 4);
    return static_cast<int>(*m.plane(1).ptr);
}

int caseSignedMagnitudeIndex() {
    Signed m(20, 4);
    return static_cast<int>(*m.magnitude(2).ptr);      // magnitude planes are 0..1
}

int caseSignedMagnitudeIndexConst() {
    const Signed m(20, 4);
    return static_cast<int>(*m.magnitude(2).ptr);
}

// --- Wrapping a buffer that is too short ------------------------------------
//
// The wrapping constructor is spelled exactly like BinMat's but needs N times as
// many words, and is handed no length, so it cannot check. wrap is the spelling
// that is told the length -- and this is what it is for.
uint32_t g_shortBuffer[8 * 2] = {0};   // 16 words; QuantMat<3> at 64x8 needs 48

int caseQuantWrapShortBuffer() {
    Quant m = Quant::wrap(g_shortBuffer, sizeof(g_shortBuffer) / sizeof(g_shortBuffer[0]),
                          64, 8, 2);
    return m.cols();
}

int caseSignedWrapShortBuffer() {
    // N = 2 magnitude planes plus a sign plane: three planes' worth is required,
    // which is one more than the type parameter reads as.
    Signed m = Signed::wrap(g_shortBuffer, sizeof(g_shortBuffer) / sizeof(g_shortBuffer[0]),
                            64, 8, 2);
    return m.cols();
}

int caseResizeNegative() {
    Mat m(8, 4);
    m.resize(-1, 4);
    return m.cols();
}

int casePadNegative() {
    Mat m(8, 4);
    m.pad(0, 0, -1, 0);
    return m.cols();
}

int caseForEachOnEmpty() {
    Mat empty;
    int seen = 0;
    empty.forEachNonZero([&seen](int, int) { ++seen; });
    return seen;
}

int caseSparsityOnEmpty() {
    Mat empty;
    return static_cast<int>(empty.sparsity() * 100.0f);
}

#ifdef BINCV_WITH_OPENCV
int caseCVMatEmpty() {
    Mat m(8, 4);
    m.fromCVMat(cv::Mat());
    return m.cols();
}

int caseCVMatWrongType() {
    Mat m(8, 4);
    m.fromCVMat(cv::Mat::zeros(4, 4, CV_32FC1));
    return m.cols();
}
#endif

struct Case {
    const char* name;
    int (*run)();
};

const Case kCases[] = {
    {"alignment-power-of-two", caseAlignmentPowerOfTwo},
    {"alignment-word-multiple", caseAlignmentWordMultiple},
    {"negative-dims", caseNegativeDims},
    {"wrap-negative-dims", caseWrapNegativeDims},
    {"wrap-short-stride", caseWrapShortStride},
    {"wrap-null-pointer", caseWrapNullPointer},
    {"quant-height-overflow", caseQuantHeightOverflow},
    {"quant-plane-index", caseQuantPlaneIndex},
    {"quant-plane-index-const", caseQuantPlaneIndexConst},
    {"binmat-plane-index", caseBinMatPlaneIndex},
    {"binmat-plane-index-const", caseBinMatPlaneIndexConst},
    {"signed-magnitude-index", caseSignedMagnitudeIndex},
    {"signed-magnitude-index-const", caseSignedMagnitudeIndexConst},
    {"quant-wrap-short-buffer", caseQuantWrapShortBuffer},
    {"signed-wrap-short-buffer", caseSignedWrapShortBuffer},
    {"resize-negative", caseResizeNegative},
    {"pad-negative", casePadNegative},
    {"for-each-on-empty", caseForEachOnEmpty},
    {"sparsity-on-empty", caseSparsityOnEmpty},
#ifdef BINCV_WITH_OPENCV
    {"cvmat-empty", caseCVMatEmpty},
    {"cvmat-wrong-type", caseCVMatWrongType},
#endif
};

} // namespace

int main(int argc, char** argv) {
    // No argument: the case list, so that a mismatch between this file and
    // tests/CMakeLists.txt is readable rather than mysterious. Exits non-zero,
    // which expect_fatal.cmake treats as a failure -- a case name that no longer
    // exists must not look like a passing death test.
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <case>\ncases:\n", argv[0]);
        for (const Case& c : kCases) std::fprintf(stderr, " %s\n", c.name);
        return 2;
    }

    for (const Case& c : kCases) {
        if (std::strcmp(argv[1], c.name) == 0) {
            const int evidence = c.run();

            // Not reached. Printed rather than returned so that a regression
            // which lets the bad argument through is visible in the test log,
            // not just in the exit code -- and the marker is deliberately not
            // the string ctest matches on.
            std::fprintf(stderr, "REACHED: validation did not reject case '%s' (evidence %d)\n",
                         c.name, evidence);
            std::fflush(stderr);
            return 0;
        }
    }

    std::fprintf(stderr, "unknown case '%s'\n", argv[1]);
    return 2;
}
