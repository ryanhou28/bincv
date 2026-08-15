# binCV Task Backlog

Executable breakdown of [ROADMAP.md](ROADMAP.md). Each task is scoped to roughly
one working session and is specified tightly enough that **no architectural
decision should be needed to complete it**.

If a task seems to require an architectural decision, that is a bug in this
document — see [Stop and ask](#stop-and-ask).

---

## How to use this file

1. Pick the lowest-numbered task whose dependencies are all `DONE`.
2. Read its spec, and the ARCHITECTURE sections it links.
3. Implement. Write the tests named in **Done when**.
4. Run the **Verify** commands. All must pass.
5. Mark the task `DONE` in this file and commit that with the work.

**Status:** `TODO` · `IN PROGRESS` · `DONE` · `BLOCKED`

### Stop and ask

Stop and surface the question rather than deciding, if:

- A task's spec is ambiguous or contradicts [ARCHITECTURE.md](ARCHITECTURE.md)
- A task requires a decision not recorded in
  [ARCHITECTURE §8](ARCHITECTURE.md#8-design-decisions)
- Something in scope turns out to be impossible as specified
- A measurement contradicts a claim in the docs — **this is valuable, report it**
- The work would add an operation not in the MVP set
  ([ARCHITECTURE §7](ARCHITECTURE.md#7-the-mvp-operation-set))

### Standing rules

These apply to every task and are not repeated per-task:

- **Kernels take views, never owning containers.**
  ([D-5](ARCHITECTURE.md#d-5-views-are-core-not-an-add-on))
- **Never expose a per-word popcount.** Reductions are bulk only.
  ([D-6](ARCHITECTURE.md#d-6-bulk-only-reductions))
- **No heap allocation inside kernels.**
- **Tier 1 operations must be bit-exact against OpenCV**, proven by a test.
  ([§5.1](ARCHITECTURE.md#51-three-tiers))
- **State the API tier in each public function's docstring.**
- **When memory and speed conflict, memory wins** unless the task says otherwise.
- Existing code is a prototype; replace it where it conflicts.
  ([D-7](ARCHITECTURE.md#d-7-existing-code-is-not-a-constraint))

### Verify commands

Referenced below as **V-ALL**:

```bash
# desktop, with OpenCV
cmake -S bincv-cpp -B bincv-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build bincv-cpp/build -j$(nproc)
(cd bincv-cpp/build && ctest --output-on-failure)

# core-only, no OpenCV
cmake -S bincv-cpp -B bincv-cpp/build-core -DCMAKE_BUILD_TYPE=Release -DBINCV_USE_OPENCV=OFF
cmake --build bincv-cpp/build-core -j$(nproc)
(cd bincv-cpp/build-core && ctest --output-on-failure)

# no exceptions (Tier 2 correctness)
cmake -S bincv-cpp -B bincv-cpp/build-noexcept -DBINCV_USE_OPENCV=OFF \
      -DCMAKE_CXX_FLAGS="-fno-exceptions"
cmake --build bincv-cpp/build-noexcept -j$(nproc)
(cd bincv-cpp/build-noexcept && ctest --output-on-failure)
```

All three must build **warning-free** and pass.

---

# Phase 1 — Container Foundation

Nothing else starts until this phase is `DONE`. Kernels written against the wrong
container get rewritten.

---

### T1.1 · Storage model · `TODO`

**Depends:** —
**Files:** `include/bincv-cpp/core/storage.hpp` (new)

**Goal:** Replace the hard-coded `std::vector` with a storage type that supports
owning and caller-provided memory.

**Spec**

```cpp
template <typename WordType>
class Storage {
public:
    Storage();                                  // empty, no allocation
    explicit Storage(size_t words);             // owning, zero-initialized
    Storage(WordType* ptr, size_t words);       // non-owning, wraps caller memory

    Storage(const Storage&);                    // deep copy (D-8)
    Storage& operator=(const Storage&);
    Storage(Storage&&) noexcept;
    Storage& operator=(Storage&&) noexcept;
    ~Storage();

    WordType*       data()       { return ptr_; }
    const WordType* data() const { return ptr_; }
    size_t size() const  { return words_; }     // in words
    bool   empty() const { return words_ == 0; }
    bool   ownsMemory() const { return owns_; }
};
```

- Value semantics: copying an owning `Storage` deep-copies
  ([D-8](ARCHITECTURE.md#d-8-value-semantics-not-reference-counting)).
- Copying a **non-owning** `Storage` produces a non-owning `Storage` over the same
  memory. It does not promote to owning.
- Owning allocation must not use `std::vector` — it needs to work without
  exceptions. Use `new[]`/`delete[]` or an aligned allocation helper.
- No allocation at all when constructed non-owning: this is the Tier 2 path.

**Done when**
- All five special members behave correctly, tested including self-assignment
- A non-owning `Storage` over a stack array performs zero allocations
- Moved-from storage is empty and safe to destroy
- Compiles under `-fno-exceptions`

**Verify:** V-ALL

**Do not:** add reference counting; add alignment logic (that is T1.3).

---

### T1.2 · Views · `TODO`

**Depends:** T1.1
**Files:** `include/bincv-cpp/core/view.hpp` (new)

**Goal:** The type kernels bind to.

**Spec**

```cpp
template <typename WordType>
struct BinMatView {                 // mutable
    WordType* ptr;
    size_t width, height;           // pixels
    size_t stride;                  // words per row

    static constexpr size_t WordBits = sizeof(WordType) * 8;
    bool empty() const;
    WordType*       row(size_t y);
    const WordType* row(size_t y) const;
};

template <typename WordType>
struct BinMatConstView { /* same, const WordType* ptr */ };
```

- Two distinct types, not const-templated
  ([D-9](ARCHITECTURE.md#d-9-two-view-types-not-a-const-templated-one)).
- Implicit conversion `BinMatView` → `BinMatConstView`.
- Views are non-owning and trivially copyable. No lifetime management.
- `stride` is in **words**, and is runtime — kernels never assume alignment.

**Done when**
- Both view types exist with the conversion
- `sizeof(BinMatView<uint32_t>)` is pointer + 3 words, no padding surprises
- A view can be constructed over a raw stack buffer and read/written correctly

**Verify:** V-ALL

**Do not:** add operations to views beyond accessors — kernels are free functions.

---

### T1.3 · Rework `BinMat` onto storage + views · `TODO`

**Depends:** T1.1, T1.2
**Files:** `binMat.hpp`, `impl/binMat_impl.hpp`

**Goal:** Make the existing container sit on the new foundation, and fix the
alignment default.

**Spec**

- Replace the internal `std::vector<WordType> storage` with `Storage<WordType>`.
- **Row stride defaults to word granularity**: `ceil(width / WordBits)` words, no
  padding ([D-4](ARCHITECTURE.md#d-4-word-granularity-alignment-by-default)).
  Larger alignment stays available as an opt-in constructor argument.
- Add `view()` and `constView()` returning the T1.2 types.
- Add a non-owning constructor taking `WordType* data, size_t strideWords`.
- Keep `clearTrailingBits()` — padding bits must stay zero so word-wise
  reductions cannot over-count.
- Value semantics per [D-8](ARCHITECTURE.md#d-8-value-semantics-not-reference-counting).

**Done when**
- `BinMat` allocates through `Storage`
- Default construction of a 640×480 `BinMat<uint32_t>` uses exactly 38400 bytes
  (word granularity, zero padding)
- A `BinMat` can wrap a caller-provided buffer and does not free it
- Existing behavioural tests pass against the reworked container

**Verify:** V-ALL, plus:
```bash
# expect 38400, not 46080
```

**Do not:** optimize any operation in this task. Container only.

---

### T1.4 · Error policy · `TODO`

**Depends:** T1.3
**Files:** `include/bincv-cpp/core/error.hpp` (new), all headers

**Goal:** Make Tier 2 real ([§5.3](ARCHITECTURE.md#53-error-policy)).

**Spec**

```cpp
// BINCV_THROW(ExceptionType, "message")
//   default:                 throws
//   BINCV_NO_EXCEPTIONS:     writes to stderr and calls std::abort()
// BINCV_ASSERT(cond, "msg")
//   debug builds only; compiles away entirely in release
```

- All existing validation throws route through `BINCV_THROW`.
- `at()` and `set()` become **debug-checked, unchecked in release** — bounds
  checks go through `BINCV_ASSERT`. This changes current behavior; that is
  intended ([D-7](ARCHITECTURE.md#d-7-existing-code-is-not-a-constraint)).
- Tests asserting `at()` throws must move to a debug-only test or be removed.
- Kernels never throw.

**Done when**
- The library compiles clean with `-fno-exceptions`
- A release build contains no bounds-check branches in `at()` (verify by
  inspecting `-O2 -S` output for a small caller)
- The no-exceptions build passes its tests

**Verify:** V-ALL — the `build-noexcept` configuration is the point of this task.

---

### T1.5 · `QuantMat<N>` multi-plane container · `TODO`

**Depends:** T1.3, T1.4
**Files:** `include/bincv-cpp/quantMat.hpp` (new)

**Goal:** The N-bit container. Required by pyramid level 1
([§7.2](ARCHITECTURE.md#72-pyramid-downsample--box-22)).

**Spec**

```cpp
template <size_t N, typename WordType = uint32_t>
class QuantMat {
    static_assert(N >= 1 && N <= 8, "N outside supported range");
public:
    QuantMat();
    QuantMat(int width, int height);
    QuantMat(WordType* data, int width, int height, size_t strideWords);

    static constexpr size_t Planes = N;

    BinMatView<WordType>      plane(size_t i);        // i in [0, N)
    BinMatConstView<WordType> plane(size_t i) const;
    // plane(0) is the least significant bit
};
```

- **One contiguous allocation**; plane *i* begins at offset `i * planeWords`
  ([§4.3](ARCHITECTURE.md#43-storage-model-and-views)).
- The container is **uninterpreted** — it holds N planes and assigns them no
  meaning. Signed/ternary interpretation is T1.6.
- `BinMat<WordType>` becomes an alias for the N=1 specialization, and that
  specialization keeps its hand-written single-plane paths.

**Done when**
- `QuantMat<3, uint32_t>` on 640×480 allocates exactly 3 × 38400 bytes
- `plane(i)` views are independently readable and writable
- Round-trip: write a known N-bit value pattern per pixel, read it back
- N=1 goes through `BinMat`'s specialized paths, not a plane loop

**Verify:** V-ALL

**Do not:** implement arithmetic across planes — that is Phase 2.

---

### T1.6 · Signed and ternary interpretation · `TODO`

**Depends:** T1.5
**Files:** `include/bincv-cpp/quantMat.hpp`

**Goal:** Sign-magnitude accessors, thin over `QuantMat`
([§4.2](ARCHITECTURE.md#42-signed-values-sign-magnitude)).

**Spec**

```cpp
// Signed N-bit: N magnitude planes + 1 sign plane, so N+1 planes total.
template <size_t N, typename WordType = uint32_t>
class SignedQuantMat {
public:
    BinMatView<WordType> magnitude(size_t i);   // i in [0, N)
    BinMatView<WordType> sign();                // plane N
    // ... const overloads
};

// Ternary {-1, 0, +1} is the N=1 case.
template <typename WordType = uint32_t>
using TernaryMat = SignedQuantMat<1, WordType>;
```

- Convention: `sign` bit set means negative. `magnitude == 0` means the value is
  zero and the sign bit is ignored (`-0` is not a distinct value).
- Thin wrapper over `QuantMat<N+1>` — no separate storage path.

**Done when**
- `TernaryMat` round-trips all three values per pixel
- The canonical-zero rule is documented and tested
- No storage duplication versus `QuantMat<N+1>`

**Verify:** V-ALL

---

### T1.7 · Google Test migration · `TODO`

**Depends:** T1.6
**Files:** `tests/`, `CMakeLists.txt`

**Goal:** Replace the interim harness (`tests/test_util.hpp`).

**Spec**
- Vendor GoogleTest via `FetchContent`, guarded so the **core-only and
  no-exceptions builds still work** — GTest needs exceptions, so the Tier 2
  configuration keeps a minimal assertion path.
- Port existing suites; preserve the core / interop split.
- `ctest` remains the entry point.

**Done when**
- All three V-ALL configurations pass
- Test count is at least what it was before the migration
- A deliberately failing assertion still produces a non-zero exit code
  (regression-check the harness itself)

**Verify:** V-ALL

---

### T1.8 · Verification script · `TODO`

**Depends:** T1.7
**Files:** `scripts/verify.sh` (new)

**Goal:** One command that runs everything a session should check before committing.

**Spec**
- Build and test all three V-ALL configurations
- Fail on any warning
- Print a short summary table
- Non-zero exit if anything fails

**Done when:** `./scripts/verify.sh` is green on a clean tree and red if any
configuration breaks.

---

# Phase 2 — Bit-Parallel Primitives

---

### T2.1 · Equivalence harness · `TODO`

**Depends:** T1.8
**Files:** `tests/equivalence.hpp` (new)

**Goal:** Build this **before** the kernels it validates
([§10.2](ARCHITECTURE.md#102-equivalence-harness)).

**Spec**

```cpp
// Assert a binCV Tier 1 op is bit-exact against its OpenCV equivalent.
// Converts BinMat -> CV_8U {0,255}, runs both, compares every pixel.
void expectBitExact(const BinMatConstView<W>& actual, const cv::Mat& expected);

// Random binary matrices at a given fill ratio, seeded and reproducible.
BinMat<W> randomBinary(int w, int h, float fillRatio, uint64_t seed);
```

- Test over a size matrix that includes **non-word-multiple widths** (e.g. 1, 7,
  31, 33, 63, 65, 70, 640) — that is where packing bugs live.
- Fill ratios spanning sparse to dense: 0.0, 0.01, 0.5, 0.99, 1.0.
- OpenCV-only; guarded by `BINCV_WITH_OPENCV`.

**Done when:** the harness exists and is demonstrated on one already-implemented
operation (`countNonZero` against `cv::countNonZero`).

**Verify:** V-ALL

---

### T2.2 · Logic operations · `TODO`

**Depends:** T2.1
**Files:** `include/bincv-cpp/ops/logic.hpp` (new)

**Goal:** The first real test of the performance thesis. API tier 1.

**Spec**

```cpp
void bitwiseAnd(BinMatConstView<W> a, BinMatConstView<W> b, BinMatView<W> dst);
void bitwiseOr (BinMatConstView<W> a, BinMatConstView<W> b, BinMatView<W> dst);
void bitwiseXor(BinMatConstView<W> a, BinMatConstView<W> b, BinMatView<W> dst);
void bitwiseNot(BinMatConstView<W> src, BinMatView<W> dst);
```

- Word-wise over each row. Strides may differ between arguments — do not assume
  a single contiguous run unless strides match and rows are dense.
- `bitwiseNot` sets padding bits; clear them afterwards so reductions stay correct.
- Also provide `QuantMat` overloads applying the op per plane.

**Done when**
- Bit-exact against `cv::bitwise_*` across the full T2.1 size and fill matrix
- A benchmark exists comparing against OpenCV on `CV_8U` with the same content
  ([§10.3](ARCHITECTURE.md#103-benchmark-denominator))
- Benchmark results committed under `results/`

**Verify:** V-ALL

---

### T2.3 · Horizontal shift · `TODO`

**Depends:** T2.2
**Files:** `include/bincv-cpp/ops/shift.hpp` (new)

**Goal:** Cross-word bit shifting. Morphology and the derivative both need it, and
it is the easiest primitive to get subtly wrong.

**Spec**

Bit convention, matching the existing implementation: column `c` lives at bit
`c % WordBits` of word `c / WordBits`, LSB first.

`shiftLeft(src, dst, k)` means `dst[r][c] = src[r][c + k]`, zero-filled at the
right edge. With `wordShift = k / WordBits` and `bitShift = k % WordBits`:

```
if bitShift == 0:
    dst[i] = src[i + wordShift]
else:
    dst[i] = (src[i + wordShift]     >> bitShift)
           | (src[i + wordShift + 1] << (WordBits - bitShift))
```

Out-of-range source words read as zero.

**`bitShift == 0` must be a separate branch** — `x << WordBits` is undefined
behavior, and this is the single most likely bug in this task.

`shiftRight` is the mirror image.

**Done when**
- Both directions correct for `k` from 0 through `2 * WordBits + 1`
- Correct at non-word-multiple widths
- Verified against a naive per-pixel reference shift, not just by inspection
- Padding bits are zero after every shift

**Verify:** V-ALL

**Do not:** hand-vectorize yet. Correct scalar first; NEON is Phase 5.

---

### T2.4 · Vertical shift and borders · `TODO`

**Depends:** T2.3
**Files:** `include/bincv-cpp/ops/shift.hpp`

**Goal:** Row-offset shifts and the border policy shared by all neighbourhood ops.

**Spec**
- Vertical shift is a row-offset copy — no bit manipulation.
- Implement `BorderType` handling from `core/types.hpp`: `BORDER_CONSTANT`,
  `BORDER_REPLICATE`, `BORDER_REFLECT`, `BORDER_REFLECT_101`, `BORDER_WRAP`.
- Border semantics must match OpenCV exactly — this is what makes downstream
  Tier 1 operations bit-exact.

**Done when**
- Every border type matches `cv::copyMakeBorder` on equivalent content
- Vertical shifts correct for offsets exceeding the image height

**Verify:** V-ALL

---

### T2.5 · Bulk reductions · `TODO`

**Depends:** T2.4
**Files:** `include/bincv-cpp/ops/reduce.hpp` (new)

**Goal:** Population counts, bulk only
([D-6](ARCHITECTURE.md#d-6-bulk-only-reductions)).

**Spec**

```cpp
size_t countNonZero(BinMatConstView<W> src);
size_t countNonZero(BinMatConstView<W> src, Rect region);
```

- Whole-word accumulation. Correctness depends on padding bits being zero.
- **No per-word popcount is exposed publicly**; any internal helper stays in
  `impl::` and is documented as internal.
- Scalar `__builtin_popcountll` for now — the vectorized form is Phase 5, and
  this API shape is what allows that swap without touching callers.

**Done when**
- Matches `cv::countNonZero` across the T2.1 matrix
- Region variant correct for regions not aligned to word boundaries
- Benchmark committed; expected to be a large win over the per-pixel loop
- No public per-word popcount exists (grep the public headers)

**Verify:** V-ALL

---

### T2.6 · Masked and windowed reductions · `TODO`

**Depends:** T2.5
**Files:** `include/bincv-cpp/ops/reduce.hpp`

**Goal:** The reduction shape the LK covariance needs
([§7.5](ARCHITECTURE.md#75-lk-gradient-covariance)). Designed now so a second
interface is not needed later.

**Spec**

```cpp
// popcount(a & b) over a region
size_t countAnd(BinMatConstView<W> a, BinMatConstView<W> b, Rect region);

// popcount(a & b & ~c) and popcount(a & b & c) in one pass over the region
struct SplitCount { size_t whenClear, whenSet; };
SplitCount countAndSplit(BinMatConstView<W> a, BinMatConstView<W> b,
                         BinMatConstView<W> c, Rect region);
```

`countAndSplit` is exactly the covariance cross-term: with `a = mag_x`,
`b = mag_y`, `c = sign_x ^ sign_y`, `whenClear` counts agreeing signs and
`whenSet` counts opposing ones.

- MVP recomputes per window. Do **not** build incremental/sliding accumulation —
  [E-3](ARCHITECTURE.md#9-open-questions-and-planned-experiments) decides whether
  that is worth it, and premature incremental state would fix the API shape early.
- Windows are large (31×31) and overlap heavily; note that in the docstring so the
  E-3 experiment has context.

**Done when**
- Both functions correct against naive per-pixel references
- Correct for windows straddling word boundaries and clipped at image edges
- Single-pass: `countAndSplit` reads each word once

**Verify:** V-ALL

---

### T2.7 · Majority and thresholded counts · `TODO`

**Depends:** T2.6
**Files:** `include/bincv-cpp/ops/bitslice.hpp` (new)

**Goal:** Small-count arithmetic. Feeds denoise and pyramid downsample.

**Spec**

```cpp
// Bitwise majority of three: (a&b) | (b&c) | (a&c)
W maj3(W a, W b, W c);

// Bit-sliced sum of k single-bit inputs -> ceil(log2(k+1)) result planes.
// Needed for k = 4 (2x2 box) and k = 9 (3x3 median).
void bitSlicedSum(const W* inputs, size_t k, W* outPlanes);

// Compare a bit-sliced value against a constant threshold -> 1-bit mask
W thresholdGE(const W* planes, size_t nPlanes, unsigned threshold);
```

**Done when**
- `maj3` matches a per-pixel median-of-3 reference
- `bitSlicedSum` correct for k = 3, 4, 9 against a per-pixel sum
- `thresholdGE` correct across all threshold values for the tested widths

**Verify:** V-ALL

---

# Phase 3 — VIO Frontend Operations

---

### T3.1 · Denoise — median of 3 · `TODO`

**Depends:** T2.7
**Files:** `include/bincv-cpp/ops/denoise.hpp` (new)

**Spec:** Three-pixel median. For binary input median equals majority, so this is
`maj3` over the three-pixel neighbourhood. Reference semantics (neighbourhood
orientation, border behavior) come from
`~/seal/SEAL/SEAL_HybVIO/HybVIO/SEAL/src/temporal_processing/denoise.cpp` —
**read it and match it**; do not invent the neighbourhood.

**Done when:** matches the reference implementation pixel-for-pixel on binary
input; benchmark committed.

**Verify:** V-ALL

---

### T3.2 · Threshold / binarize · `TODO`

**Depends:** T3.1
**Files:** `include/bincv-cpp/ops/threshold.hpp` (new)

**Spec:** Produce a 1-bit frame from a higher-precision source (`CV_8U` via
interop, or `QuantMat<N>`). Tier 1 semantics against `cv::threshold` for the
binary output case.

**Done when:** bit-exact against `cv::threshold` with `THRESH_BINARY`.

**Verify:** V-ALL

---

### T3.3 · Morphology · `TODO`

**Depends:** T3.2
**Files:** `include/bincv-cpp/ops/morphology.hpp` (new)

**Spec**

```cpp
void erode (BinMatConstView<W> src, BinMatView<W> dst, const StructuringElement&);
void dilate(BinMatConstView<W> src, BinMatView<W> dst, const StructuringElement&);
void morphologyEx(BinMatConstView<W> src, BinMatView<W> dst, MorphOp op,
                  const StructuringElement&);
```

- Dilate = OR of shifted copies; erode = AND of shifted copies. Built on T2.3/T2.4.
- Support `MORPH_RECT`, `MORPH_CROSS`, `MORPH_ELLIPSE` and all `MorphOp` values
  from `core/types.hpp`.
- Special-case 3×3 — it is the common case in practice.
- **API tier 1**: must be bit-exact against `cv::erode` / `cv::dilate` /
  `cv::morphologyEx`, including border behavior.

**Done when:** bit-exact across the T2.1 matrix for every shape and op;
benchmarks committed.

**Verify:** V-ALL

---

### T3.4 · Pyramid downsample · `TODO`

**Depends:** T3.3
**Files:** `include/bincv-cpp/ops/pyramid.hpp` (new)

**Goal:** The operation where output precision exceeds input precision
([§7.2](ARCHITECTURE.md#72-pyramid-downsample--box-22)).

**Spec**

```cpp
// Box 2x2 sum then subsample. Output is N-bit, chosen by the caller.
template <size_t NOut, size_t NIn, typename W>
void pyrDown(const QuantMat<NIn, W>& src, QuantMat<NOut, W>& dst);

template <typename W>
class Pyramid { /* levels, each its own QuantMat; bit depth per level */ };
```

- 2×2 sum via `bitSlicedSum` (T2.7), then requantize to `NOut` bits.
- **Requantization is a documented choice, not a default.** The reference lets
  precision grow into `CV_8U` (1 → 3 → 4 → 5 bits measured). Whether a capped
  `NOut` preserves tracking accuracy is
  [E-7](ARCHITECTURE.md#9-open-questions-and-planned-experiments) — implement the
  cap as a parameter and record the deviation in the docstring.
- **API tier 2** — same role as `cv::pyrDown`, deliberately different numerics.

**Done when**
- Level-1 output matches the reference to within the documented quantization
- Bit growth measured and recorded for a 4-level pyramid at several `NOut`
- Peak footprint of the pyramid measured and committed (feeds E-7)

**Verify:** V-ALL

---

### T3.5 · Binarized spatial derivative · `TODO`

**Depends:** T3.4
**Files:** `include/bincv-cpp/ops/derivative.hpp` (new)

**Goal:** First operation producing a multi-plane signed output — the real test of
the container design.

**Spec**

Level 0, 1-bit input → ternary output, shifts and masks only:

```
pos = shiftLeft (src, 1) & ~shiftRight(src, 1)
neg = shiftRight(src, 1) & ~shiftLeft (src, 1)
mag = pos | neg;   sign = neg
```

N-bit input → signed (N+1)-bit output via bit-sliced subtraction of the shifted
planes ([§7.4](ARCHITECTURE.md#74-spatial-derivative--binarized--1-0-1)).

Reference semantics: `SEAL/src/keypoint_tracking/gradients.cpp`,
`calcBinarizedDeriv` — kernel is `[-1, 0, 1]` in both axes.

**Done when**
- Ternary path matches a per-pixel reference on 1-bit input
- N-bit path matches a per-pixel reference for N = 2, 3
- Output is a valid `TernaryMat` / `SignedQuantMat` with the canonical-zero rule
  respected
- Benchmark against `cv::filter2D` with the same kernel, committed

**Verify:** V-ALL

---

### T3.6 · LK gradient covariance · `TODO`

**Depends:** T3.5
**Files:** `include/bincv-cpp/ops/covariance.hpp` (new)

**Goal:** The load-bearing operation
([§7.5](ARCHITECTURE.md#75-lk-gradient-covariance)).

**Spec**

```cpp
struct GradientCovariance { int64_t sumXX, sumYY, sumXY; };

// Ternary (pyramid level 0)
GradientCovariance gradientCovariance(const TernaryMat<W>& dx,
                                      const TernaryMat<W>& dy,
                                      Rect window);
```

Implemented entirely with T2.5/T2.6 reductions:

```
sumXX = countNonZero(dx.magnitude(0), window)
sumYY = countNonZero(dy.magnitude(0), window)
// cross term: one countAndSplit pass over (mag_x, mag_y, sign_x ^ sign_y)
sumXY = split.whenClear - split.whenSet
```

The `sign_x ^ sign_y` term needs a scratch plane; take it as a caller-provided
buffer — **no allocation inside the kernel**.

**Done when**
- Matches a per-pixel float reference exactly (all values are integers, so exact
  agreement is required, not approximate)
- Correct for windows clipped at image edges
- Benchmarked across window sizes 7, 15, 31 — results feed
  [E-3](ARCHITECTURE.md#9-open-questions-and-planned-experiments)

**Verify:** V-ALL

---

### T3.7 · Corner response · `TODO`

**Depends:** T3.6
**Files:** `include/bincv-cpp/ops/corner.hpp` (new)

**Spec:** Minimum-eigenvalue response from the T3.6 covariance, plus non-maximum
suppression and the quality/min-distance selection that
`goodFeaturesToTrack` performs. Reference: `SEAL/src/keypoint_detection/gftt.cpp`
with `gftt_corner_derivative_type: BINARIZED`. **API tier 2.**

**Done when:** detected corners match the reference within a documented tolerance
on real sequences; benchmark committed.

**Verify:** V-ALL

---

### T3.8 · Hybrid LK tracking · `TODO`

**Depends:** T3.7
**Files:** `include/bincv-cpp/ops/opticalFlow.hpp` (new)

**Spec:** Route (b) from
[§7.9](ARCHITECTURE.md#79-the-known-hard-problem-subpixel-interpolation):
bit-parallel window extraction and covariance accumulation, floating-point solve
and subpixel interpolation. **API tier 2.**

Reference: `SEAL/src/keypoint_tracking/SparsePyrLKOpticalFlowSealImpl.cpp`.

**Done when**
- Tracks features across real image sequences
- Flow vectors agree with the reference within a documented tolerance
- Peak footprint of the full frontend measured — this is the number Phase 4 needs

**Verify:** V-ALL

---

# Phase 4 — Validation

Phase 4 tasks are experiments. Each produces a committed measurement and a written
conclusion, and each may invalidate a decision — **that is the point**, and a
contradicted claim should be reported, not worked around.

| Task | Experiment | Produces |
|---|---|---|
| T4.1 | [E-1](ARCHITECTURE.md#9-open-questions-and-planned-experiments) row alignment beyond word granularity | Whether D-4 stands; whether profiles get built |
| T4.2 | [E-2](ARCHITECTURE.md#9-open-questions-and-planned-experiments) default word width on aarch64 | `BinMat`'s default template argument |
| T4.3 | [E-3](ARCHITECTURE.md#9-open-questions-and-planned-experiments) incremental vs recomputed window reductions | Whether T2.6 grows incremental state |
| T4.4 | [E-4](ARCHITECTURE.md#9-open-questions-and-planned-experiments) generic-N cost versus specialized paths | Whether N stays arbitrary |
| T4.5 | [E-7](ARCHITECTURE.md#9-open-questions-and-planned-experiments) pyramid level bit depths | Level depths; large share of footprint |
| T4.6 | [E-5](ARCHITECTURE.md#9-open-questions-and-planned-experiments) end-to-end accuracy, footprint, speed | **The project's headline result** |

T4.6 is the milestone the whole plan serves: equivalent trajectory accuracy,
several-fold smaller peak footprint, faster on the bit-parallel operation set
([ROADMAP success criteria](ROADMAP.md#success-criteria)).

---

# Phase 5 — Platform Hardening

Detailed once Phase 4 produces numbers — measurements determine which kernels are
worth vectorizing. Scope is fixed
([ROADMAP Phase 5](ROADMAP.md#phase-5--platform-hardening)): NEON reference
kernels, aarch64 cross-compilation and on-hardware validation, x86 portability,
and Tier 2 correctness in CI.
