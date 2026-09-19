# The CUDA backend: one definition of the format, an independent kernel tree
# (ARCHITECTURE, backends). Compiled as a static library because kernels are
# .cu translation units; the headers under include/ are the public surface and
# every type in them is device-typed, so no call can hide where its memory
# lives.

# sm_86 is the backend's reference GPU (RTX 3070 Ti, the machine the committed
# numbers come from). Override for other parts -- 87 covers Jetson Orin with no
# source change. Must be chosen before enable_language(CUDA).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(bincv_cuda STATIC
    src/logic.cu
    src/reduce.cu
    src/pack.cu
    src/census.cu
    src/denseDisparity.cu
    src/threshold.cu
    src/edge.cu
    src/morphology.cu
    src/median.cu
    src/pyramid.cu
    src/shift.cu
    src/derivative.cu
    src/covariance.cu
    src/orientation.cu
    src/descriptor.cu
    src/fast.cu
    src/corner.cu
    src/subpix.cu
    src/keypoints.cu
    src/opticalFlow.cu
    src/sparseMatch.cu
    src/denseCensusBox.cu)

set_target_properties(bincv_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

target_include_directories(bincv_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# The core headers by path rather than through bincv_core: the backend's own
# translation units compile without BINCV_WITH_OPENCV (nothing here touches
# interop), and a consumer already links bincv_core for the host half.
target_include_directories(bincv_cuda PRIVATE ${BINCV_POLICY_ROOT}/include)
# Consumers reach cudaMemcpy and friends from the inline transfer header.
target_link_libraries(bincv_cuda PUBLIC CUDA::cudart)
target_link_libraries(bincv_cuda PRIVATE bincv_warnings)

if(BINCV_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(BINCV_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

# Worked examples for the backend, and NOT gated on BINCV_BUILD_BENCHMARKS or on
# BINCV_BUILD_TESTS: an example exists to be READ and to be the one thing a
# reader can run to check the residency claim, so it must exist in the plain
# `-DBINCV_CUDA=ON` build a reader configures. The top-level examples/ directory
# is ungated for the same reason.
add_subdirectory(examples)

# ---------------------------------------------------------------------------
# descriptor.cu is compiled WITHOUT fused multiply-add, and that is a
# correctness flag rather than a numerical preference.
#
# The kernel selects a BRIEF rotation bin by calling the host library's own
# `bincv::briefAngleBin`, which carries BINCV_HOST_DEVICE so that one
# definition serves both targets. Its last arithmetic is
# `angle * kBinsPerRadian + kBriefAngleBins` -- a CONTRACTIBLE multiply-add.
# Left to its default (`-fmad=true`) nvcc fuses it into one FFMA; a baseline
# x86-64 host, having no FMA, emits FMUL then FADD. The two differ by up to one
# ULP, and one ULP at a 12-degree bin boundary does not move an angle slightly:
# it selects a different rotation of the pattern and replaces the whole 256-bit
# descriptor. Measured here, with the flag removed, 80 of 4080 descriptor words
# changed across a single-ULP walk of the 30 boundaries -- which is what
# test_cuda_orientation_descriptor's BinBoundariesWalkedInSingleUlpSteps case
# exists to catch.
#
# THIS PINS ONE SIDE OF A HOLE IT DOES NOT CLOSE. GCC's own default is
# `-ffp-contract=fast`, and nothing in this project sets it, so an FMA-capable
# HOST already fuses the same expression while a baseline x86-64 host does not.
# The flag below makes the device agree with the unfused host; it does not make
# the two hosts agree with each other.
set_source_files_properties(src/descriptor.cu PROPERTIES COMPILE_OPTIONS "-fmad=false")

# ---------------------------------------------------------------------------
# src/subpix.cu, and ONLY src/subpix.cu, is compiled -fmad=false.
#
# The sub-pixel refinement accumulates `bx += xx*px + xy*py` and
# `det = gxx*gyy - gxy*gxy`. Both are multiply-then-add over operands that are
# NOT exact: `xx` is a Gaussian weight times a small integer and rounds. Left to
# its default (`-fmad=true`) nvcc fuses each into one FFMA/DFMA with a single
# rounding, where a baseline x86-64 host emits a multiply and an add with two.
# The results differ in the last bits of `q`, which can change the converged
# position and can flip a corner between `refined`, `clamped` and `diverged` --
# and ops/subpix.hpp's own docstring records that a tolerance fitted to an
# observed value cannot fail, so the suite compares positions EXACTLY and there
# is nothing to absorb the difference.
#
# src/corner.cu DELIBERATELY DOES NOT GET THIS FLAG, which is a correction to
# this family's own design. `impl::minEigenValue` computes
# `disc = d*d + 4.0*c*c` where BOTH products are exact integers -- corner.hpp's
# PRECISION section bounds `D <= 5*blockSize^4`, far under 2^53 -- so their sum
# is exact and a fused multiply-add and a separate mul-then-add give
# bit-identical results at every blockSize this library can express.
# `0.5*(s - sqrt(disc))` offers no multiply-add to contract at all. Naming the
# flag there would have been superstition dressed as care, and it would have
# cost the file its FMAs for nothing.
set_source_files_properties(src/subpix.cu PROPERTIES COMPILE_OPTIONS "-fmad=false")

# ---------------------------------------------------------------------------
# src/opticalFlow.cu is compiled -fmad=false, and it is a CORRECTNESS flag.
#
# The device tracker's contract is that it produces the HOST tracker's answer
# bit for bit, and that claim is load-bearing rather than decorative: LK's
# epsilon test (`deltaX*deltaX + deltaY*deltaY <= eps2`) and its oscillation
# test are FLOAT BRANCHES, so a one-ulp disagreement changes a point's iteration
# count and can move its endpoint by PIXELS. A tolerance would have no stable
# meaning; equality is cheaper to defend.
#
# Three expressions in the inner loop are contractible multiply-adds over
# operands that are NOT exact: the residual identity's
# `w00*t00 + w01*t01 + w10*t10 + w11*t11 - self` (impl::TapSums::combine), the
# 2x2 solve's `a12*b2 - a22*b1`, and the epsilon test itself. Left to nvcc's
# default (`-fmad=true`) each fuses into one instruction with a single rounding
# where a baseline x86-64 host emits a multiply and an add with two.
#
# THIS PINS ONE SIDE OF A CONTRACT THE CONSUMER OWNS THE OTHER HALF OF, and the
# device header says so where a caller reads the promise: the guarantee is
# "device equals host when the host header is compiled without FMA contraction".
# GCC's own default is `-ffp-contract=fast`, so the test translation unit is
# compiled `-ffp-contract=off` (backends/cuda/tests/CMakeLists.txt) and
# `impl::lkFmaGuardProbeAsync` exists so that losing either flag breaks the
# suite instead of silently changing which points converge.
set_source_files_properties(src/opticalFlow.cu PROPERTIES COMPILE_OPTIONS "-fmad=false")

# ---------------------------------------------------------------------------
# src/sparseMatch.cu is compiled -fmad=false, and it is a CORRECTNESS flag with
# a WIDER reach than the parabola it looks like it is about.
#
# The obvious hazard is the sub-pixel fit: `impl::parabolicOffset` runs in
# `double` on both targets and the suite compares the resulting `disparity` and
# `nextPts` with memcmp rather than an epsilon, because a tolerance fitted to an
# observed difference cannot fail.
#
# The hazard that actually bites is one step earlier and is an INTEGER decision,
# not a last bit. Both window ops compute their anchor as
# `prevPts[p].x * scale - halfWinX` and then `impl::floorToLL` of it. Left to
# nvcc's default (`-fmad=true`) that multiply-subtract fuses into one FFMA with a
# single rounding where a baseline x86-64 host emits an FMUL and an FSUB with
# two -- and one ulp on either side of an integer boundary moves `floorToLL` by
# a WHOLE PIXEL. A different anchor is a different clipped window, which is a
# different cost for every candidate in the search, which is a different
# disparity. It would not look like a rounding difference; it would look like a
# plausible answer.
#
# Same shape as src/descriptor.cu's flag, and the same limitation: this pins the
# DEVICE to the unfused evaluation. GCC's own default is `-ffp-contract=fast`,
# so an FMA-capable host already fuses the same expression while a baseline
# x86-64 one does not. The flag makes the device agree with the unfused host; it
# does not make two hosts agree with each other.
set_source_files_properties(src/sparseMatch.cu PROPERTIES COMPILE_OPTIONS "-fmad=false")
