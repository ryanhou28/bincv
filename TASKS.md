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

Tasks marked **⚙️ needs Pi** require the reference measurement device
([setup](docs/MEASUREMENT_HARDWARE.md)). **Skip them and continue** — they are not
on the critical path for implementation work. Everything through T3.5 proceeds
without hardware; see
[What works without it](docs/MEASUREMENT_HARDWARE.md#what-works-without-it).

**Do not substitute a laptop measurement to unblock one.** Closing an experiment on
non-authoritative hardware is worse than leaving it open, because a recorded wrong
answer stops anyone from asking again. Running one early on x86 as a *signal* is
fine, provided the entry stays `PARTIAL`. E-1, E-2 and E-3 were all closed on the
reference device, and the x86 pre-runs disagreed with the device on the sign of
several rows — see [X-11](EXPERIMENTS.md)'s X-7 caveat for why that was expected.

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
- **Performance and footprint choices are settled by measurement, not argument.**
  If a task needs such a choice and no experiment has decided it, that is a
  stop-and-ask — see below.

### Experiment tasks

Tasks marked **· E-n ·** are experiments, not implementations. They follow the
protocol in
[ARCHITECTURE §9](ARCHITECTURE.md#how-performance-and-footprint-decisions-get-made):

1. **Write the decision rule into [EXPERIMENTS.md](EXPERIMENTS.md) before
   measuring anything.** Each such task states its rule; copy it into the log
   first. Deciding afterward invites fitting the conclusion to the numbers.
2. Measure **alternatives**, on representative workloads, reporting **memory and
   speed together**.
3. Log method, result, and conclusion. Commit the measurement code.
4. Promote the conclusion to a D-record in
   [ARCHITECTURE §8](ARCHITECTURE.md#8-design-decisions), or reopen the existing
   one.

A result that contradicts a documented claim is a **finding**. Report it; do not
adjust code or docs to make the contradiction disappear.

Experiments run **in the phase whose code they gate**, which is why T2.8–T2.10 sit
in Phase 2 rather than at the end.

### Verify commands

Referenced below as **V-ALL**. Since T1.8 this is one command:

```bash
./scripts/verify.sh            # all four configurations, warnings fatal
./scripts/verify_arm.sh        # aarch64 correctness under emulation (T1.9)
```

It builds and tests four configurations, and a warning in any of them fails the
run — `-DBINCV_WERROR=ON` plus an independent scan of the build log:

| | build type | OpenCV | exceptions | `BINCV_DEBUG_CHECKS` | test backend |
|---|---|---|---|---|---|
| `build` | Release | yes | yes | 0 | Google Test |
| `build-core` | Release | no | yes | 0 | Google Test |
| `build-noexcept` | Release | no | **no** | 0 | built-in harness |
| `build-debug` | **Debug** | no | yes | **1** | Google Test |

The fourth was added by T1.8. Before it, every configuration was Release, so
`BINCV_ASSERT` — every bounds check in `at()` and `set()`, and every kernel
precondition — was compiled out of everything that could fail.

The individual commands are still in
[GETTING_STARTED](GETTING_STARTED.md#build-configurations) for when one configuration needs to
be driven by hand.

---

# Phase 1 — Container Foundation

Nothing else starts until this phase is `DONE`. Kernels written against the wrong
container get rewritten.

---

### T1.1 · Storage model · `DONE`

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

### T1.2 · Views · `DONE`

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

### T1.3 · Rework `BinMat` onto storage + views · `DONE`

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

### T1.4 · Error policy · `DONE`

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

**How the two halves stay covered** *(the part that is easy to get wrong: both
halves of this policy are invisible to an ordinary in-process test)*

- Every `BINCV_THROW` site and every `BINCV_ASSERT` site is a **death test**: one
  ctest test per case, driven by `tests/expect_fatal.cmake`, which requires the
  child to terminate *abnormally* and to print its diagnostic. A failed check
  ends the process, so nothing inside that process can report on it.
  `BINCV_CHECK_THROWS` cannot stand in — without exceptions it cannot evaluate
  its expression at all, so it reports a SKIP (44 of them in `test_binMat`), and
  a suite of only those checks would pass in `build-noexcept` while verifying
  nothing.
- The **checked** configuration is compiled by `tests/test_error_checked.cpp` and
  `tests/test_assert_abort.cpp`, which `#undef NDEBUG` before including anything.
  All three V-ALL builds are Release, so without that the live-assert half of the
  policy would be dead source in every configuration anyone runs. This is why
  V-ALL does **not** need a fourth Debug configuration.
- Both are regression-proven, not assumed: deleting a `BINCV_THROW` check or
  either `at()`/`set()` bounds check makes all three configurations fail.

---

### T1.5 · `QuantMat<N>` multi-plane container · `DONE`

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

### T1.6 · Signed and ternary interpretation · `DONE`

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

### T1.7 · Google Test migration · `DONE` — as a hybrid, deliberately

**Depends:** T1.6
**Files:** `tests/`, `tests/CMakeLists.txt`, `CMakeLists.txt`

**Goal:** Replace the interim harness (`tests/test_util.hpp`).

**Spec**
- Vendor GoogleTest via `FetchContent`, guarded so the **core-only and
  no-exceptions builds still work**.
- Port existing suites; preserve the core / interop split.
- `ctest` remains the entry point.

**Done when**
- All V-ALL configurations pass
- Test count is at least what it was before the migration
- A deliberately failing assertion still produces a non-zero exit code
  (regression-check the harness itself)

**What was built.** Google Test is the backend in three of the four
configurations. The built-in harness stays the backend of the dependency-free
one (core-only, `-fno-exceptions`), and both run the *same* case bodies through
the *same* check macros — `tests/test_util.hpp` is now a two-backend shim, not a
second test framework. Suites are named cases (`BinMat.Contract_uint8_t` and so
on) in both, so a `--gtest_filter` written for one narrows the other.

**Correction to this spec.** "GTest needs exceptions, so the Tier 2
configuration keeps a minimal assertion path" is **not true**, and the
implementation does not rely on it. Measured: googletest v1.14.0 compiles from
source under `-fno-exceptions` — it detects the absent `__EXCEPTIONS` and sets
`GTEST_HAS_EXCEPTIONS` to 0 itself — links, runs, reports a deliberate failure,
exits non-zero, and `EXPECT_DEATH` works. `-DBINCV_USE_GTEST=ON` is therefore
supported in that configuration.

The default declines it for a different and better reason: that configuration's
whole claim is that binCV needs a C++17 compiler and nothing else. Putting the
one gate the embedded claim rests on behind a network fetch and a 30k-line
desktop framework is how a gate goes dark. Neither is the death-test machinery
ported to `EXPECT_DEATH`: it forks and re-runs the test binary, which is
materially more fragile under the emulation T1.9 runs in, and those 33 cases are
the only validation coverage the no-exceptions build has.

**Verify:** V-ALL

---

### T1.8 · Verification script · `DONE`

**Depends:** T1.7
**Files:** `scripts/verify.sh` (new), `cmake/BincvWarnings.cmake` (new)

**Goal:** One command that runs everything a session should check before committing.

**Spec**
- Build and test all V-ALL configurations
- Fail on any warning
- Print a short summary table
- Non-zero exit if anything fails

**Done when:** `./scripts/verify.sh` is green on a clean tree and red if any
configuration breaks.

**What was built.** Two gates that had been documented but never enforced now
have teeth:

- **Warnings.** `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion`
  are applied to every first-party target through the `bincv_warnings` interface
  target — not to `bincv_core`, so a consumer's warning policy stays theirs.
  OpenCV's headers are included as `SYSTEM`, without which `-Wconversion` buried
  the gate under ~107 unactionable warnings per interop translation unit.
  `-Werror` is **off by default and on in `verify.sh`**: a compiler upgrade
  should not break every working tree at once, but nothing may be committed
  until the gate is green. A target that never links `bincv_warnings` cannot
  slip through, because `bincv_assert_warning_policy()` fails the *configure*
  step when one exists.
- **`BINCV_DEBUG_CHECKS == 1`.** The Debug configuration compiles it. Mutating
  `BinMat::at`'s bounds check to reject the last legal column fails 5 tests
  there and 1 in Release core-only — the Debug build runs all ~1900 checks
  through live assertions, which is coverage no Release configuration can give.

Clean builds by default: an incremental build does not recompile an unchanged
translation unit, so it does not re-emit its warnings, and a log scan over one
would pass vacuously. `--incremental` is available and says so.

**Follow-up (review of T1.7–T1.9): the teeth were partly false, and are now
real.** Four reviewers found nine confirmed ways for a green run to mean nothing;
all are fixed and each has a check that fails without the fix.

- **The wiring claim above was wrong as originally written.** `verify.sh` scanned
  the build log and the doc said that covered a target missing `bincv_warnings`.
  It cannot: such a target compiles with *no warning flags*, so it emits nothing
  to scan. Measured — a test target with three deliberate warnings gave
  `WARN 0 … PASS … ALL CONFIGURATIONS GREEN`, exit 0. Replaced with a
  configure-time assertion over `BUILDSYSTEM_TARGETS`. The log scan stays for what
  it *can* catch (linker and CMake warnings) and no longer claims more.
- **Gate self-check.** Two throwaway builds that must fail — a wired target that
  warns, and an unwired target — run before the configurations. Before this,
  neither half of the warning policy had ever been observed rejecting anything.
- **`CHECKS` is compared against a committed floor**
  (`tests/expected-checks.txt`, per configuration, per suite). It used to be
  printed and compared against nothing: dropping `test_storage` from
  `BINCV_CORE_TESTS` took core from 1892 checks to 1348 and still printed
  `ALL CONFIGURATIONS GREEN`.
- **Each configuration must be itself.** `BINCV_DEBUG_CHECKS` and
  `BINCV_EXCEPTIONS_ENABLED` are read back out of the built `test_error`.
  `CXXFLAGS=-DNDEBUG ./scripts/verify.sh --only debug` used to pass with every
  `BINCV_ASSERT` compiled away — the exact vacuity T1.8 exists to remove.
- **A configuration that ran no tests is red.** `ctest` exits 0 on an empty test
  set; a cached `-DBINCV_BUILD_TESTS=OFF` produced `PASS` with
  `No tests were found!!!` in the log.
- **The suite list is derived, not hardcoded** — from `ctest --show-only=json-v1`
  in `verify.sh` and from `tests/CMakeLists.txt` in `verify_arm.sh`. A suite
  appended to `BINCV_CORE_TESTS` used to run and be counted by neither.
- **`build-logs/checks-*.txt` is stamped and published only on PASS.** It was
  truncated unconditionally, so an aborted or `--only` run left `verify_arm.sh`
  diffing against a partial or stale reference.
- **Robustness.** A `head -15` on a warning list gave `sort` a SIGPIPE, which
  under `set -o pipefail` killed the whole run with exit 141 — no table, no
  `VERIFICATION FAILED` — reproduced 3/3 on a build with ~300 diagnostics. A
  non-build-tree `build-core/` aborted the run instead of failing one
  configuration. `--only` with no value exited 1 with zero output.

---

### T1.9 · aarch64 correctness verification · `DONE`

**Depends:** T1.8
**Files:** `scripts/verify_arm.sh` (new)

**Goal:** Catch aarch64 correctness bugs continuously, without ARM hardware.

**Why now:** aarch64 is the primary target
([ARCHITECTURE §2](ARCHITECTURE.md#2-target-platforms)), and the bugs that only
appear there — type-width assumptions, alignment faults, NEON intrinsics that fail
to compile — are cheap to catch early and expensive to find late. This is already
verified to work on the current machine: the core suite passes 261/261 under
emulation.

**Spec**

```bash
docker run --rm --platform linux/arm64 -v "$PWD":/src -w /src \
    arm64v8/gcc:12 bash -c '<build and run core + no-exceptions suites>'
```

- Runs the **core-only** and **no-exceptions** configurations; the OpenCV interop
  suite is out of scope (avoids installing OpenCV in the container).
- Script should skip with a clear message, not fail, when Docker or the arm64
  platform is unavailable.

**This is a correctness gate only.** It must print a warning that timings from
this environment are meaningless, so no one is tempted to benchmark in it — see
[EXPERIMENTS.md § Measurement platforms](EXPERIMENTS.md#measurement-platforms).

**Done when**
- `./scripts/verify_arm.sh` builds and passes the core suite under emulated aarch64
- Results match x86 exactly (same check counts, same pass/fail)
- The script warns against benchmarking in it
- Skips gracefully without Docker

**Do not:** benchmark here; do not treat a pass as ARM performance validation.

**Status: `DONE`.** Both configurations pass under emulated aarch64, warning-free
under the full flag set, with check counts identical to x86_64 — 1892 (core) and
1815 + 60 skipped (no-exceptions). The comparison is mechanical: `verify.sh`
writes per-suite counts to `bincv-cpp/build-logs/checks-*.txt` and this script
diffs against them, so "matches x86 exactly" is checked rather than eyeballed.
All 33 death tests run here too, driven by a shell loop that enforces the same
two conditions as `tests/expect_fatal.cmake` — died abnormally, and named the
reason — with the expected diagnostics parsed out of `tests/CMakeLists.txt` so
the two cannot drift apart.

`arm64v8/gcc:12` has a compiler and `make` and no `cmake`, so the suites are
compiled directly rather than configured. Installing cmake would put an apt
fetch inside an emulated container on the critical path of a gate meant to be
run often; compiling directly keeps it hermetic, which is the right property for
the configuration whose claim is that binCV needs nothing but a compiler.

**It found something on its first run**, though not the kind of thing it was
built for: `Storage::operator=` tripped `-Wuse-after-free`, which **GCC 12
enables in `-Wall` and GCC 11 does not have at all**. Reproduced on x86_64
`gcc:12`, so it is a compiler-version finding rather than an architecture one —
and it means "builds warning-free" was true of the desktop compiler and false of
the next one. The code was correct (an alias guard the compiler cannot follow);
it now installs the new descriptor before freeing the old block, which removes
the question instead of answering it. See `core/storage.hpp::adoptThenFree`.

**Follow-up (same review).** The aarch64 gate had four ways to look green while
verifying less than it said:

- **The skip is exit 77, not exit 0.** `verify.sh --arm` treated any zero exit as
  a pass, so on a machine without Docker it printed `aarch64: OK` directly under
  this script's own `aarch64 correctness was NOT verified`.
- **The x86_64 comparison always reports its outcome**, and is stamped. It used
  to sit inside `if reference exists`, so on a fresh clone — where `build-logs/`
  is gitignored and absent — the run printed `PASS`/`PASS` having compared
  nothing. And a reference left over from an earlier tree was diffed as if it
  described the current one; both sides now carry a content hash of
  `include/` + `tests/`, and a mismatch is `NOT PERFORMED`, not a pass.
- **The summary can no longer contradict the exit code.** A failed comparison
  printed `core-only PASS` / `-fno-exceptions PASS` and exited 1. There is now a
  third row for the comparison and an explicit `AARCH64 VERIFICATION FAILED`.
- **The death-test count is checked.** The case list is read out of each binary's
  usage output; an enumeration that parsed to nothing printed
  `death tests: 0 passed` and PASSed. The expected count now comes from
  `tests/CMakeLists.txt` (excluding the `if(BINCV_OPENCV_FOUND)` cases, which
  this container does not compile) and a mismatch fails.

---

### T1.10 · Cortex-A measurement runner · `DONE` · verified against real hardware 2026-08-16

**Depends:** T1.9
**Files:** `scripts/run_on_pi.sh` (new)
**Setup:** [docs/MEASUREMENT_HARDWARE.md](docs/MEASUREMENT_HARDWARE.md)

**Goal:** Make the reference device usable from an ordinary session, with the
measurement hazards enforced mechanically rather than remembered.

**Why this is Phase 1:** every Phase 2 experiment (T2.8–T2.10) closes on this
device. Without a runner, each of them either blocks on manual work or silently
settles for non-authoritative numbers.

**Spec**

```bash
scripts/run_on_pi.sh <target> <command>     # target e.g. pi@raspberrypi.local
```

1. **Preflight — refuse rather than warn.** Abort with a clear message if:
   - `uname -m` is not `aarch64` (a 32-bit OS answers a different question — see
     [EXPERIMENTS.md § Measuring on the Pi 4](EXPERIMENTS.md#measuring-on-the-pi-4))
   - `vcgencmd get_throttled` is non-zero before the run
   - the host is unreachable
2. Sync the repo (`rsync`, excluding build trees).
3. Set the governor to `performance`; restore the previous setting on exit,
   including on failure.
4. Build Release, core-only configuration.
5. Run the command pinned to one core via `taskset`.
6. **Re-check `vcgencmd get_throttled` afterwards. If non-zero, mark the results
   INVALID** — throttled numbers must not be recorded.
7. Print an environment block for pasting into the log: architecture, kernel, CPU
   model, governor, throttle state before/after, core pinning, compiler version.
8. Copy results back to `results/`.

**Done when**
- `./scripts/run_on_pi.sh <target> ./tests/test_binMat` builds and runs remotely
  and reports 261/261
- A 32-bit target is refused with an explanatory message
- A throttled run is reported INVALID, not recorded
- The governor is restored even when the command fails
- Skips with a clear message when no target is configured, so
  `scripts/verify.sh` remains usable without the Pi

**Do not:** make the Pi a hard dependency of the normal build or test flow. It is
a measurement device, not part of the development loop.

---

# Phase 2 — Bit-Parallel Primitives

---

### T2.1 · Equivalence harness · `DONE`

**Depends:** T1.8
**Files:** `tests/equivalence.hpp` (new), `tests/test_equivalence.cpp` (new)

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

**What was built.** `tests/equivalence.hpp` plus `tests/test_equivalence.cpp`
(3392 checks, `opencv` configuration only). The spec's two entry points are
there as written, and three things it did not ask for are, because without them
the harness would license kernels rather than judge them:

- **`firstMismatch` / `Mismatch::describe()`.** "Not equal" over a 640×480 frame
  is unactionable. Failures read
  `uint8_t 7x2 fill=0.50 [fromCVMat] -- first mismatch at row 0, col 0: expected
  0, actual 255; 10 of 14 pixels differ`, and the total separates a one-bit bug
  from a whole-image one at a glance.

- **A second, independent content generator** (`randomCvMask`), which builds the
  same content directly as `CV_8U` without constructing a `BinMat` or calling the
  unpacking path. `Equivalence.PackingAnchor` pins the conversion against it
  across the whole size and fill matrix. Without it the harness is circular:
  comparing a `BinMat` against its own conversion passes even when the conversion
  reads the wrong column, and — measured, see below — a shared conversion fault
  **cancels exactly** through a pointwise operation, which is the shape every
  T2.2–T2.7 test has.

- **Injectable faults, as `WILL_FAIL` ctest cases.** `tests/CMakeLists.txt`
  rebuilds the suite once per deliberate conversion fault, so "the harness can
  fail" is three ctest results rather than a claim. Neutering the injection makes
  all three go red; that was checked.

**Reproducibility.** SplitMix64 (four `uint64_t` operations, no state array, no
implementation freedom) plus a hand-written bit-to-pixel mapping.
`std::uniform_int_distribution` is **not** portable — the standard fixes what a
distribution means, not how it consumes its engine — so a recorded value would
differ between libstdc++, libc++ and MSVC and the failure would look like a
packing bug. One draw per pixel in row-major order, which is also what makes the
content **word-type independent**: `BinMat<uint8_t>` and `BinMat<uint64_t>` from
the same seed hold the same picture, checked directly. Golden `countNonZero` and
FNV-1a digests are committed for four cases.

**Matrix.** Widths 1, 7, 31, 33, 63, 65, 70, 640 × heights 1, 2, 3, 17, 37, 480 ×
fills 0.0, 0.01, 0.5, 0.99, 1.0 × four word types. All widths but 640 are
non-multiples of at least one word width; 0.0 and 1.0 are exact, not approximate.

**The faults, and what caught them** (checks passed, of 3392):

| injected | result | what stayed green |
|---|---|---|
| column off-by-one in the conversion | 2444 | `countNonZero` — a cyclic rotation is count-preserving |
| dropped trailing partial word | 1621 | — |
| transposed row/column | 1512 | `countNonZero`, **and the transpose case, where the fault cancels** |
| `countNonZero` skips the last column (real library bug) | 2745 | packing anchor |
| `fromCVMat` packs column x into bit x+1 (real library bug) | 2984 | `countNonZero` |

The two "stayed green" columns are the finding: a test whose two sides share the
conversion cannot see a bug in it. Only the anchor sweep saw all five.

---

### T2.2 · Logic operations · `DONE`

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

**What was built.** `include/bincv-cpp/ops/logic.hpp` (the four view kernels plus
per-plane `QuantMat<N>` overloads) and `tests/test_logic.cpp`. The suite is a
**core** test: its Tier 1 half sits behind `BINCV_WITH_OPENCV`, so the kernels are
compiled and checked in all four configurations — including Debug, the only one
where their `BINCV_ASSERT` preconditions are live. 65768 checks under `opencv`,
53672 under the other three, plus four death tests and three `WILL_FAIL` builds.

**Preconditions, and what each is worth.** Mismatched dimensions, a stride shorter
than `ceil(width / WordBits)`, and an unsafe destination alias are `BINCV_ASSERT`
sites; each has a death test. Two are *not* checkable and are documented instead:
a destination that is a **sub-width window onto a wider image** has its next
1–`WordBits−1` pixels cleared by the trailing-word mask (measured: 52 of 52 live
pixels in columns 70–95 of a 640-wide parent), and every address written is inside
the parent, so nothing can diagnose it. `Logic.TrailingWord_*` pins that behaviour
at bit granularity so a future change to it is a test failure rather than a
surprise.

**Aliasing — decided, documented, tested.** `dst` may be *exactly* one of the
sources (same first word, same stride), or share **no word** with it. In-place is
safe because these operations are pointwise in the word index. A destination that
overlaps a source at a different offset is undefined and asserted against, because
it cannot be diagnosed any other way — every address involved is valid memory and
only the answer is wrong. Recorded as
[D-11](ARCHITECTURE.md#d-11-kernels-alias-exactly-or-not-at-all), since
it binds every future `ops/` kernel and not only this one.

The "no shared word" half is checked **per row**, not by bounding span. The first
version compared spans and therefore rejected two views over one buffer whose rows
interleave — alternate row bands, left/right column tiles — which are legal under
D-5, correct in release, and aborted every Debug build. `Logic.AliasAccepts_*`
covers all three shapes.

**Four things were measured rather than argued, each by breaking something on
purpose and watching which cases went red.** The first two are the kernel; the
last two are the *suite*, and they are the ones worth reading.

| injected fault | before review | after |
|---|---|---|
| trailing-word mask removed from `applyBinary` | **0 fail** — 56044/56044 green | 1860 of 65768 fail |
| kernel writes zeros into `dst`'s in-stride padding | **0 fail** | 120 fail |
| `BINCV_EQUIVALENCE_INJECT_FAULT=1` (column off by one) | **0 fail**, exit 0 | 3368 fail, exit 1 |
| `BINCV_EQUIVALENCE_INJECT_FAULT=3` (transposed conversion) | **0 fail**, exit 0 | 5904 fail, exit 1 |

Rows 1 and 3–4 were **blind spots, not passes**. Every source in every sweep was
built through `set()`, so all padding was already zero and `Op(0,0) == 0` left the
mask nothing to do — `Logic.DirtySources_*` now sweeps sources wrapped over
all-ones buffers, which is a documented construction and the only one where the
mask is load-bearing for AND/OR/XOR. And the Tier 1 half built *both* sides through
`toCvMask`/`unpackTo8U`, so a fault in the conversion cancelled exactly through a
pointwise operation — precisely what `equivalence.hpp` property 2 predicts in
writing. OpenCV's inputs now come from `randomCvMask()`, the harness's independent
generator, and `tests/CMakeLists.txt` builds `test_logic` under faults 1–3 as
`WILL_FAIL` targets so the property is a ctest result. The earlier claim that the
mask accounted for "5317 failing checks" was **wrong**: all 5317 came from
`bitwiseNot`, which fails with or without a dirty source.

**Benchmark** (`benchmark/logic_benchmark.cpp`, results committed at
`results/logic_benchmark.log`). Denominator per
[§10.3](ARCHITECTURE.md#103-benchmark-denominator): `cv::bitwise_*` on the same
content as `CV_8U`, verified to produce the same image before anything is timed —
and a disagreement now skips the size and exits non-zero instead of printing a
table under a warning. Median of three pinned runs, x86_64 — **indicative, not
authoritative** ([EXPERIMENTS.md](EXPERIMENTS.md#measurement-platforms)).

**The denominator has to be named, because it changes the answer.** Two OpenCV 4.x
installs on this machine, both Release, both with the same dispatch list, differ by
**3.3× on `cv::bitwise_not` alone** (105 GB/s in distro 4.5.4, 32 GB/s in a locally
built 4.8.0-dev) while agreeing within noise on the binary operations. The figures
below use **4.5.4** — the faster denominator, and the one a user gets by installing
rather than building OpenCV. The benchmark now prints its OpenCV baseline/dispatch
lines so any table can be attributed.

| | 640×480 | 1024×1024 | 8192×4096 |
|---|---|---|---|
| `bitwiseAnd` binCV `uint32` | 0.00297 ns/px | 0.00308 ns/px | 0.01096 ns/px |
| `bitwiseAnd` OpenCV `CV_8U` | 0.02798 ns/px | 0.02851 ns/px | 0.16450 ns/px |
| ratio, and/or/xor (all 6 binCV rows) | **7.6–10.2×**, median 8.0× | **7.6–9.8×**, median 8.1× | 14.8–21.2× |
| ratio, `bitwiseNot` | **5.4–5.5×** | **5.5–5.6×** | 18.4–19.2× |
| buffer | 38400 B vs 307200 B | 131072 B vs 1048576 B | 4.19 MB vs 33.55 MB |
| both sides cache-resident? | yes | yes | **no — OpenCV only goes to DRAM** |

**One ratio for the binary operations, with its spread — not three.** AND, OR and
XOR are one shared loop differing in a single instruction; the earlier per-op
figures (8.0× / 10.1× / 8.1×) were code-placement noise, and the fast/slow cluster
swaps between word types and between runs. Anything above 8.0× at a cache-resident
size has no mechanism behind it and is the top of the noise band.

**`bitwiseNot` is the weak result, and the weakness is binCV's.** 5.5× against a
denominator that dispatches properly — *below* the 8× traffic ratio. binCV's unary
kernel manages ~72 GB/s where `cv::bitwise_not` manages ~105, so the representation
wins 8× and the kernel gives a third back. It is the one row where binCV is slower
per byte than OpenCV. The previously published **18.0× / 19.8× is withdrawn**: it
was a property of the local 4.8.0-dev build, not of binCV. Nothing was tuned in
response — an accurate negative result is the finding.

**The mechanism holds where the claim is made.** At both cache-resident sizes the
two sides move memory at about the same rate (~100–108 GB/s OpenCV, ~100–134 GB/s
binCV), so both are bandwidth-bound and the speedup lands on the 8× compression
ratio. At 8192×4096 the ratio roughly doubles for a *different* reason — binCV's
12.6 MB working set fits the 32 MiB L3 and OpenCV's 100 MB does not — and that
number must not be averaged with the others. **Word width is still not settled**
(`uint64` leads on some rows, trails on others); that is
[E-2](ARCHITECTURE.md#9-open-questions-and-planned-experiments)/T2.9.

**Two measurement-validity corrections, both recorded as
[X-5](EXPERIMENTS.md#x-5--bandwidth-ceiling-probes-for-the-t22-logic-benchmark--done):**
the physical-bound probe was described as "a steady 48–68 GB/s" and actually
reports 74–143 GB/s with a 1.65× spread at one footprint, so the 4× threshold
built on it needed 400–560 GB/s to fire; it is now 1.5× against a probe that
prints its own worst batch. And the `std::memcpy` context number was explained by
non-temporal stores above ~128 KB, which the probe's own output contradicts — it
comes back *up* to 21–28 GB/s at 33 MB. The swing is real, the explanation was
not, and it has been withdrawn rather than repeated.

---

### T2.3 · Horizontal shift · `DONE`

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

**What was built.** `include/bincv-cpp/ops/shift.hpp` — `shiftLeft`, `shiftRight`,
`shiftUp`, `shiftDown` and the 2-D `shift` they are all one-line wrappers over —
plus `tests/test_shift.cpp`, a **core** test whose Tier 1 half sits behind
`BINCV_WITH_OPENCV` so the kernels are compiled and checked in all four
configurations, Debug included. 307133 checks under `opencv`, 99695 under the
other three, plus five death tests. Also
`include/bincv-cpp/impl/kernel_util.hpp`: the row-tail mask, the stride check and
the D-11 overlap predicates moved out of `ops/logic.hpp` unchanged, because the
second kernel header needed the same three and a copy would have been a second
place for the aliasing rule to drift.

**The spec's recurrence is incomplete, and the gap is where the bug lives.**
"Out-of-range source words read as zero" covers whole words past the row. It does
not cover the **padding bits of the trailing partial word**, which sit at column
indices past `width` — outside the image — and which a `shiftLeft` by *k* moves
into destination columns `width-k … width-1`, which are live pixels. Sources with
dirty padding are a supported construction (`BinMat`'s wrap constructor documents
that a caller's padding is the caller's, and `test_logic` already sweeps it), so
every source word is read through `impl::extendedRowWord`, which substitutes the
BORDER value for everything at or past `width`. Measured with that substitution
removed — width 5, `uint8_t`, all five pixels clear in a wrapped word of `0xE0`,
so every set bit is padding:

| | correct | substitution removed |
|---|---|---|
| `shiftLeft(1)` | `0x00` | `0x10` — pixel 4 set |
| `shiftLeft(2)` | `0x00` | `0x18` — pixels 3, 4 |
| `shiftLeft(3)` | `0x00` | `0x1C` — pixels 2, 3, 4 |

Both words the recurrence reads are in range, so nothing about the word
arithmetic is wrong and no bounds check has anything to say —
`Shift.DirtySource_*` is the only thing that sees it, and 1492 checks go red
across the suite when it is missing.

**`bitShift == 0`, and how "no UB" was proven rather than asserted.** It is a
separate branch, and the proof is not a reading of the code:

- `tests/test_shift.cpp` sweeps `k` from 0 through `2*WordBits+1` at all four word
  widths, which reaches `k ∈ {0, WordBits, 2*WordBits}` — every `bitShift == 0`
  case — by construction rather than by someone remembering to add them.
- The whole suite is additionally compiled and run under
  `-fsanitize=undefined -fno-sanitize-recover=all`, twice: `-O2 -DNDEBUG` (the
  shipping configuration, assertions compiled out) and `-O1` without `NDEBUG` and
  with ASan as well (`BINCV_ASSERT` live). Both are clean, 99695/99695, zero
  diagnostics.
- **And the sanitizer was watched failing.** With the `bitShift == 0` branch
  removed, UBSan reports
  `ops/shift.hpp:316: runtime error: shift exponent 32 is too large for 32-bit
  type 'unsigned int'` and 1008 checks go red. Without the sanitizer this is the
  bug that hides: on x86 the natural encoding masks the shift count, so
  `x << WordBits` quietly yields `x` where the algebra wants 0, and every test
  written at `k = 1` still passes.

Reproduce: `g++ -std=c++17 -O2 -DNDEBUG -fsanitize=undefined
-fno-sanitize-recover=all -Iinclude -Itests tests/test_shift.cpp`.

**Seven mutations, and what each turned red** (checks passed, of 99695 core):

| injected fault | passed | what went red |
|---|---|---|
| `bitShift == 0` branch removed | 98687 | 7 case families, **and a UBSan shift-exponent diagnostic** |
| source's trailing word read as-is | 98203 | Borders, DirtySource, GuardWords |
| destination tail mask dropped | 91112 | 6 families, padding verdicts throughout |
| `BORDER_REFLECT_101` collapsed onto `BORDER_REFLECT` | 97743 | BorderIndex_Reference, Borders, DirtySource (and, with OpenCV, BorderInterpolate_OpenCv and OpenCv) |
| `shiftRight`'s direction inverted | 93131 | 6 families |
| vertical offset negated | 94921 | Borders, GuardWords, Strides, Vertical |
| out-of-range word read as `row[0]` instead of the fill | 98141 | 6 families |

The fourth is the one worth reading: a one-column difference between the two
reflect flavours is invisible in the middle of a frame, and it is what a
downstream `cv::erode` comparison would eventually fail on for reasons that would
look nothing like a border bug.

**In place is NOT supported, and that is a narrowing of D-11** rather than a
contradiction of it — recorded there. `ops/logic.hpp` accepts `dst == src`
because its kernels are pointwise in the word index; a shift reads words
`i ± wordShift`, and no row order rescues the vertical case under a non-constant
border. `impl::kernel_util.hpp` carries the two predicates separately so a kernel
picks the one matching its access pattern.

**Not hand-vectorized, and one optimisation deliberately not taken:** the row loop
calls a bounds-checked word accessor rather than splitting into a check-free
interior plus two edges. That is Phase 5's business; the branch is perfectly
predictable and the split doubles the number of index expressions that can be off
by one.

---

### T2.4 · Vertical shift and borders · `DONE`

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

**What was built.** Same file and same suite as T2.3. A vertical shift is a
row-index remap and no bit manipulation at all, which is also what makes the 2-D
`shift(src, dst, dx, dy, ...)` a **single pass with no scratch buffer**: each
destination row resolves its source row through the border mapping and is then
written by the horizontal path. That matters for T3.3 — morphology shifts by a
structuring element offset, which is 2-D, and a caller composing two axis-aligned
shifts would need a temporary between them, which "no heap in kernels" would make
the caller's problem.

**The border mapping is the Tier 1 promise, so it is tested as a function and not
only through images.** `impl::borderIndex` is `cv::borderInterpolate`: same five
types, `-1` for `BORDER_CONSTANT`, `0` for both reflect flavours at `len == 1`. It
is checked three ways —

1. against `cv::borderInterpolate` directly, every coordinate in
   `[-3*len-3, 3*len+3]` for 15 lengths × 5 types (`Shift.BorderInterpolate_OpenCv`);
2. against an **independently written iterative reference** in the test file — the
   `do { } while (out of range)` shape OpenCV documents, where the library uses a
   closed-form modulo into the pattern's period. Two different algorithms for one
   function, so the core-only configurations keep the property even without OpenCV
   (`Shift.BorderIndex_Reference`);
3. through whole shifted images against `cv::copyMakeBorder` over the T2.1 size
   matrix — `equivalenceWidths()` × `equivalenceHeights()` × `equivalenceFillRatios()`
   — at all four word widths, at 16 offsets per size, the last six of them *past*
   one or both extents (`Shift.OpenCv_*`).

An image comparison alone cannot do this job: the border reaches only `|dx|`
columns and `|dy|` rows, so at 640×480 with `dx = 1` a wrong mapping is 480 of
307200 pixels and any sampling test misses it.

**Point 3 was a claim before it was true, and a review caught it.** As first
written it swept `equivalenceWidths()` × `{1, 3, 17}` at a single fill of 0.5, and
its offsets were a fixed table reaching `|dy| = 3` — so `dy > height` at a height
above 1 was never put to `cv::copyMakeBorder` at all, which is the exact regime
this task's second done-when clause names. (At height 1 the case is degenerate:
every non-constant border maps every row to row 0.) Measured, with a clamp
injected into the row loop that is wrong *only* past the height —
`dy > src.height ? src.height : dy` feeding `borderIndex` — the core reference half
went red at 112589 of 112733 and **all four `Shift.OpenCv_*` stayed green**. The
horizontal analogue (clamping `dx` to the width) did fail, because
`{wordBits + 1, -1}` already exceeds a width of 1; the hole was one axis wide,
which is the shape of gap that "it is symmetric" reasoning misses. The sweep now
runs the whole matrix and adds `{0, ±(h+2)}`, `{±(w+2), 0}` and `{±(w+2), ±(h+2)}`,
under which that clamp fails 2768 checks and takes every `Shift.OpenCv_*` with it.
The kernel was never wrong — this was a missing acceptance check, and the cost of
closing it is 112733 → 307133 checks and about six seconds in the one
configuration that has OpenCV.

**Why a closed form rather than OpenCV's loop.** T2.4 requires vertical offsets
*exceeding the image height* to be correct, and T3.3 will pass structuring-element
offsets through unchanged. OpenCV's do-while converges by about `2*len` per
iteration, so shifting a 7-row image by 500 costs ~35 iterations per row; one
signed modulo costs the same at any offset. The two agree because both reflect
patterns are periodic — period `2*len` for `BORDER_REFLECT`, `2*len-2` for
`BORDER_REFLECT_101` — and mutation 4 above is what says so when they stop.

**The `cv::copyMakeBorder` denominator had to be built, and it is exact.** OpenCV
has no shift, but `dst(y,x) = extended-src(y+dy, x+dx)` and `copyMakeBorder` is
the function that materialises "extended src": pad by `top = max(0,-dy)`,
`bottom = max(0,dy)` (and likewise horizontally), then crop at
`(max(dx,0), max(dy,0))`. Corners come out right for free, because
`copyMakeBorder` builds them by applying both axes' mappings — which is exactly
what this file's row-then-column structure does.

**The fill decision, which T2.4 required and which is now
[D-12](ARCHITECTURE.md#d-12-a-shift-carries-a-border-and-the-fill-is-the-callers).**
Erode (AND of shifts) needs **ones** outside the image and dilate (OR of shifts)
needs **zeros**; one fixed fill makes one of them wrong at every edge. So the fill
is a parameter — `(BorderType, bool)`, defaulting to `BORDER_CONSTANT` / `false`,
which is the zero fill T2.3 specifies for the bare three-argument call. OpenCV
encodes the same asymmetry through `morphologyDefaultBorderValue()`, and that is a
claim about OpenCV, so it is **measured** rather than cited:
`Shift.MorphologyFillPremise` runs the real `cv::erode` and `cv::dilate` and pins
64/64 pixels surviving erosion of an all-white frame under the default border
against 36/64 under an explicit zero border.

---

### T2.5 · Bulk reductions · `DONE`

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

**What was built.** `include/bincv-cpp/ops/reduce.hpp` (both `countNonZero`
overloads plus T2.6's two masked forms) and `tests/test_reduce.cpp`, a **core**
test whose Tier 1 half sits behind `BINCV_WITH_OPENCV` so the kernels are compiled
and checked in all four configurations — including Debug, the only one where their
`BINCV_ASSERT` preconditions are live. 711724 checks under `opencv`, 546484 under
the other three, plus three death tests. `core/types.hpp` gained `Rect`, which the
spec used and the project did not have; it is `cv::Rect`'s four fields in
`cv::Rect`'s order, signed, because a 31×31 window centred near an edge has a
negative origin.

**No per-word popcount is reachable from outside the file**, which is the whole
interface (D-6). `impl::popcountWord` is the only place `__builtin_popcountll`
appears, and it is deliberately **not** in `impl/kernel_util.hpp` — the header
every kernel already includes — because a per-word popcount one include away from
every kernel is one that will be called from one. `grep -rn popcount
include/bincv-cpp` finds it, its four call sites inside the same file, this file's
own documentation, and two pre-existing comments that mention the word.

The builtin-free SWAR form beside it (`impl::popcountWordPortable`, for a toolchain
with no `__builtin_popcountll` — MSVC, and the claim that binCV needs a C++17
compiler and nothing else) is compiled **unconditionally** rather than inside the
`#if`, because every configuration this project verifies is GCC or clang and a
guarded definition would be source that nothing ever compiles.
`Reduce.PortablePopcount_*` runs it against the builtin over 20480 values per word
width, including both endpoints by name.

**The region is clipped, not rejected**, and that is a calling convention rather
than leniency: an LK window (§7.5) is out of range for every keypoint within 15
pixels of an edge, so clipping once here replaces the same four `min`/`max`
expressions in every call site. Negative origins, extents past either edge, and
rectangles wholly outside all count what exists and nothing else. The arithmetic
runs in `long long` before anything becomes a `size_t`: `x + width` overflows
`int`, and a negative origin converted to `size_t` is not "clipped to zero", it is
2⁶⁴ − k.

**The padding decision is [D-13](ARCHITECTURE.md#d-13-a-reduction-counts-pixels-never-padding)
and it departs from this task's wording.** T2.5 says "correctness depends on
padding bits being zero"; the shipped kernels do not depend on it — a bit at or
past `width` is never counted, at a cost of one AND per row. The reasons and the
measurement are in the D-record. The short version: sources with dirty padding are
a documented, supported construction, `ops/shift.hpp` already decided the same way,
and with the whole-image count changed to trust the invariant instead, 545919 of
546468 core checks still pass and only `Reduce.DirtyPadding_*` goes red.

**Benchmark** (`benchmark/reduce_benchmark.cpp`, results committed at
`results/reduce_benchmark.log`). Denominator per
[§10.3](ARCHITECTURE.md#103-benchmark-denominator): `cv::countNonZero` on the same
content as `CV_8U`, with every implementation's count compared before anything is
timed. **The result is a finding, and it is unflattering** — recorded as
[X-7](EXPERIMENTS.md):

| 640×480, x86_64 | as shipped | with `-mpopcnt` |
|---|---|---|
| `countNonZero` binCV `uint64` | 0.04544 ns/px | 0.00639 ns/px |
| `cv::countNonZero` on `CV_8U` | 0.01098 ns/px | 0.01304 ns/px |
| ratio | **0.24× — binCV 4.2× slower** | **2.04× — binCV 2.0× faster** |
| versus `BinMat::countNonZero()` | 6.1× faster | 35.3× faster |

binCV applies no `-march` flags (the top-level `CMakeLists.txt` detects AVX2 and
AVX-512 and deliberately does not enable them until runtime dispatch lands), so on
the shipping x86 build `__builtin_popcountll` is **a call to libgcc's
`__popcountdi2`, once per word** — verified in `-O2 -S` output; clang inlines a
~15-instruction SWAR sequence instead, and neither is `popcntq`. X-3 recorded
`popcntq` for x86_64, which is true of the ISA and not of this build. So T2.5's
done-when clause about the per-pixel loop holds (6.1×) and the Tier 1 speed
comparison does not. **Nothing was changed in response** — enabling an ISA baseline
is a dispatch decision (ROADMAP 2.3) that no experiment has settled, and the
aarch64 half of the same question is E-2/T2.9's to close on the reference device.
The bulk API is what makes the eventual fix a change to one file.

---

### T2.6 · Masked and windowed reductions · `DONE`

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

- MVP recomputes per window. Incremental/sliding accumulation was deliberately not
  built here, because [E-3](ARCHITECTURE.md#9-open-questions-and-planned-experiments)
  had not run and premature incremental state would have fixed the API shape early.
  **E-3 has now run and chose incremental** ([X-11](EXPERIMENTS.md)); it lands in
  [T2.11](#t211--t26-api-extensions-mandated-by-e-3--done), not here.
- Windows are large (31×31) and overlap heavily; note that in the docstring so the
  E-3 experiment has context.

**Done when**
- Both functions correct against naive per-pixel references
- Correct for windows straddling word boundaries and clipped at image edges
- Single-pass: `countAndSplit` reads each word once

**Verify:** V-ALL

**What was built.** Same file and same suite as T2.5. `countAnd` and
`countAndSplit` take views, read only, and allocate nothing — the AND happens a
word at a time inside the reduction, which is the reason these exist rather than
being spelled `bitwiseAnd` into a scratch image followed by `countNonZero`: the
scratch would be a heap allocation in a kernel, or a caller-provided buffer whose
only purpose is to be counted and thrown away.

**Aliasing is unrestricted, and that is the one place D-11 does not reach.**
Nothing is written, so `a`, `b` and `c` may be the same view or overlap
arbitrarily; `countAndSplit(m, m, m, r)` is well defined and tested. An alias
predicate copied in from `ops/logic.hpp` would have rejected exactly the calls the
covariance makes — overlapping windows of one frame — so the death-test list has a
dimension case and a stride case and deliberately no alias case.

**Single pass, and two popcounts per word rather than three.**

```
both  = a[i] & b[i] & mask
total = popcount(both)
set   = popcount(both & c[i])
whenSet += set;  whenClear += total - set
```

`whenClear` is `total - set`, never `popcount(a & b & ~c)`. Besides being one
popcount cheaper on a target where the popcount is the expensive part (D-6), it
never forms `~c` — which sets every padding bit of a trailing word and would count
phantom pixels unless the mask were applied a second time.

The row skeleton (head word, unmasked interior, tail word) is
`impl::visitRowWords`, shared by all four entry points, so "each word index is
visited exactly once, in ascending order, under the right mask" is a property of
**one** function — and `Reduce.Geometry_*` drives that function directly with a
recording visitor over an exhaustive cross product of origins and extents,
reconstructing the selected column set and comparing it against an independently
written clipping reference. A value comparison cannot see a double visit whose
word happens to be empty; this does.

**The covariance identity is tested as the thing it is for.**
`Reduce.Covariance_*` builds a `TernaryMat` pair, computes `sumXX`/`sumYY`/`sumXY`
through these primitives exactly as T3.6 will, and compares against a per-pixel
**float** reference over the same window — with exact equality, since every term is
a small integer and an approximate comparison would accept the off-by-one this
operation can actually have. Windows are 7, 15 and 31 at centres including all four
corners and past both edges, at six image sizes down to 1×1, and the sweep runs
twice: once with canonical-zero signs, and once with the **sign planes deliberately
dirtied** where the magnitude is zero. That second variant is not decoration —
T1.6's canonical-zero rule permits it and T3.5's derivative will produce it
(`sign = neg` is a whole-plane assignment), and the identity survives only because
the `a & b` factor removes those bits.

**T2.6 shipped recompute-per-window and no incremental state, pending E-3.**
Windows are 31×31 in practice and overlap heavily — consecutive keypoints, and
consecutive iterations on one keypoint, re-read almost the same words — which is
exactly the regime where a sliding accumulator might win. **E-3 has since measured
it and it does**, by **7.3× on a search sweep and 20× on a dense scan** at 31×31
for the form being adopted ([X-11](EXPERIMENTS.md)), so the
accumulator is being added in [T2.11](#t211--t26-api-extensions-mandated-by-e-3--done).
The signature chosen here survives that: incremental state arrives as an additional
entry point rather than as a change to this one.

**Eight mutations of `ops/reduce.hpp`, and what each turned red** (core checks
passed, out of the 546468 the suite had when they were run — the
portable-popcount case added 16 afterwards):

| injected fault | passed | what went red |
|---|---|---|
| tail mask dropped | 496202 | 6 case families |
| head mask dropped | 491251 | 6 families |
| one-word region drops the head mask | 488559 | 6 families |
| tail mask off by one (half-open confusion) | 511817 | 6 families |
| interior loop revisits `lastWord` | 443742 | 7 families, `Reduce.Geometry_*` included |
| `whenClear = total` (the split lost) | 470980 | 5 families |
| region not clipped to the width | **segfault** | 19 failing regions, then the process died |
| whole-image count trusts the padding invariant | 545919 | **`Reduce.DirtyPadding_*` only** |

The last row is the one worth reading, and it is why D-13 has a test rather than a
paragraph.

**What a four-reviewer pass then changed** (code and documents, no interface
change; the two performance findings were measured on the reference device with
decision rules written first, and both were *recorded* rather than acted on
because acting on either would have chosen T2.6's shape ahead of T2.10):

| finding | outcome |
|---|---|
| `SplitCount`'s halves are both `size_t`, so §7.5's `whenClear - whenSet` wraps to ~1.8e19 for every negatively correlated window — clean under `-Werror` | **fixed:** `SplitCount::crossTerm()` returns the signed difference; `Reduce.Covariance_*` now takes `sumXY` through it and pins the sign on a deterministic anti-correlated case |
| `impl::regionFromExtent` documented but did not assert its non-empty precondition; `x1 == 0` underflows into a ~5.8e17-word row range with `isEmpty == false` | **fixed:** `BINCV_ASSERT`, plus the `reduce-empty-extent` death test — unreachable from the public API today, which is why it needed a test rather than a comment |
| `Rect::empty/==/!=` had zero call sites and zero coverage anywhere in the repo | **fixed:** `impl::clipRegion` now calls `Rect::empty()` instead of re-spelling it, and `Reduce.Degenerate_*` exercises all three; each member's body was mutated to confirm the checks bite |
| the region overload's Tier 1 line claimed unconditional equality with `cv::countNonZero(src(region))`, but `cv::Mat::operator()(Rect)` *throws* outside the image | **fixed:** the claim is now stated over the rectangles OpenCV can express, which is also what `testOpenCvEquivalence` actually tests |
| the header's aarch64 sequence matched none of its four kernels | **fixed:** the D-6 preamble quotes what each kernel emits, with per-entry-point crossing counts (1 / 2 / 4); `reduce_benchmark.cpp`'s banner likewise |
| ARCHITECTURE §6.2 claimed in the present tense that the implementation keeps data in vector registers "without crossing back" | **false, and now measured:** the shipped bulk count runs at 0.99× of the per-word loop D-6 forbids exposing, with 1.8× available from a vector accumulator ([X-7b](EXPERIMENTS.md)). §6.2 now separates the settled interface from the implementation status; **no kernel changed** — vectorization is Phase 5 |
| ARCHITECTURE §7.5 still demanded incremental accumulation "from the start" | **fixed:** §7.5 now says the MVP recomputes and defers incremental state to E-3, and carries the exact snippet T3.6 should copy |
| composing the 2×2 covariance makes three traversals of one window | **measured at 1.30×** ([X-8](EXPERIMENTS.md)) and registered as a second axis of T2.10 — past that task's own 15% line. No entry point added |
| `countAndSplit`'s selector `c` must be frame-sized, so T3.6 cannot pass a window-sized buffer | **documented** on `countAndSplit` with the plane the caller forms and what it costs; the zero-plane four-argument alternative is registered as T2.10's third axis, since it is a memory-versus-speed choice and no experiment has settled it |
| `countNonZero(dx.magnitude(0), w)` — the spelling §7.5 and T2.6 printed — does not compile from a non-const container (D-9: deduction ignores the conversion) | **fixed in the documents,** not by adding overloads: `constMagnitude`/`constSign`/`constView` is the house spelling (`ops/logic.hpp` and every test already use it), so `reduce.hpp` and §7.5 now print it, and `Reduce.Covariance_*` compiles it from a deliberately non-const container so it cannot silently stop being the right advice |

---

### T2.7 · Majority and thresholded counts · `DONE`
<!-- kernel tasks continue; experiment tasks T2.8-T2.10 follow -->


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

**What was built.** `include/bincv-cpp/ops/bitslice.hpp` — the three word-level
primitives above, plus `bitSlicedSumPlanes(k)` (constexpr, so the caller can size
its plane array without allocating) and one view-level kernel,
`majority3(a, b, c, dst)`. **API Tier 3 throughout**: bit-sliced arithmetic has no
OpenCV counterpart ([§5.1](ARCHITECTURE.md#51-three-tiers)), so nothing here
borrows an OpenCV name. Tests in `tests/test_bitslice.cpp`, registered as a core
suite so the primitives are checked in the `-fno-exceptions` and Debug builds too.

**The adder network is the ripple, deliberately.** `bitSlicedSum` accumulates one
input at a time into the output planes through a chain of half adders, cutting the
chain at the plane where the running total provably cannot carry further (before
input *i* the total is at most *i*). That is not the cheapest network — the
textbook k = 4 form is two half adders and a full adder, 9 operations against this
file's 16 — and the task said to prefer a correct reference over a minimal one.
What the ripple buys is one loop nest correct for every k with a one-line
invariant, where a compressor tree is a different shape per k and the shapes that
matter (4 and 9) are the ones every pyramid level and every denoised frame depend
on. Phase 5 can replace the body; the exhaustive tests below are what would prove
the replacement.

**A view-level `majority3` was added**, and the reasoning is T3.1: the reference
three-pixel median filter takes the pixel above, the pixel itself and the pixel to
its right, so denoise is two shifts and one pointwise majority. Without the kernel
T3.1 would open-code a word loop — with its own stride handling, its own trailing
word mask and its own aliasing contract — next to the identical loop in
`ops/logic.hpp`. It takes views (D-5), is pointwise in the word index so it takes
the same half of [D-11](ARCHITECTURE.md#d-11-kernels-alias-exactly-or-not-at-all)
that the logic kernels do (`dst` may be `a`, `b` or `c` exactly), and masks its
trailing word so the destination's padding stays zero even when the sources' does
not. There is deliberately **no `QuantMat` overload**: bit 3 of the median of
three N-bit images is not the majority of the three bit 3s, so a per-plane loop
would compile, run, and be wrong.

**Exhaustive rather than sampled, because here that is affordable.** Every
primitive is pointwise in the bit *lane*, so k inputs have exactly 2^k distinct
per-lane patterns and the whole input space is enumerated: all 8 patterns for
`maj3`, all patterns at k = 1, 2, 3, 4, 9 and all 65536 at k = 16 for
`bitSlicedSum`, and for `thresholdGE` every value against every threshold from 0
to one past the maximum at 0–5 planes — which is that function's entire input
space at every plane count any k here reaches. The patterns are packed into the
*lanes* of the words under test, so the sweep also proves the lanes stay
independent: a carry leaking from lane L into L+1 would pass a test that used lane
0 only. The references are independent of the implementations — the median is the
sorting network `max(min(a,b), min(max(a,b),c))`, not `(a&b)|(b&c)|(a&c)`.

**Twelve mutations of `ops/bitslice.hpp`, and what each turned red** — the table
lives in `tests/expected-checks.txt` next to the floor it justifies. Two rows are
worth repeating here:

- **The trailing-word mask is held by one case family.** Storing `majority3`'s
  last word unmasked leaves 349888 of 349981 checks green, and only
  `BitSlice.DirtySources_*` fails — the same shape as
  [D-13](ARCHITECTURE.md#d-13-a-reduction-counts-pixels-never-padding)'s finding
  for `ops/reduce.hpp`.
- **Two mutations survived, and both were acted on rather than logged.**
  Narrowing `equal` in `thresholdGE`'s comparator changed no result anywhere in
  the exhaustive `(value, threshold)` enumeration — any lane it dropped had been
  added to `greater` in the same step — so two operations per plane came out and
  the mask is now named `notLess`. And deleting `c`'s half of `majority3`'s
  dimension assert left every check green in all four configurations, because an
  assert has no in-process test; that became the death test
  `test_assert_abort.bitslice-dims`, with `bitslice-alias`,
  `bitslice-short-stride` and `bitslice-sum-overlap` alongside it. The last is the
  only precondition in the project stated over raw pointers rather than views.

**A review finding turned into cases.** The stride sweep started as six mixed
combinations, none of which had `a`, `b` and `dst` tight with only `c`
over-strided — so deleting `c.stride == words` from the dense-path condition left
all 348061 checks green, and the kernel walked `c` as one contiguous run at the
wrong stride whenever the other three happened to be dense. That is what a caller
gets by mixing an over-aligned frame (D-4 makes alignment per-object) with tightly
packed ones. The list now runs the four "exactly one argument non-tight" shapes at
both non-tight flavours, and each of the four deletions fails.

**A review pass corrected three claims in the header and added a case family.**
`thresholdGE` returns a **full word and has no notion of `width`** — every lane is
answered, and at `threshold == 0` every lane is answered *yes* whatever the planes
hold, which is exactly the value T3.2's requantization sweep reaches by
arithmetic. A caller storing that into a row's trailing word without
`impl::rowTailMask` leaves padding bits set past `width`, which is
[D-13](ARCHITECTURE.md#d-13-a-reduction-counts-pixels-never-padding)'s
over-counting failure with nothing to diagnose it; `majority3` masks internally
because it owns its destination, and this one cannot. That contract is now a
docstring `@note`, a masked file-header example, and the `BitSlice.ThresholdPadding_*`
family (559 checks) — which asserts *both* halves: that the raw word really does
carry bits past `width`, so a future "fix" that masks inside `thresholdGE` fails
here on purpose, and that the documented remedy leaves padding zero while costing
no live pixel its answer. Measured: rewriting the test's masked store as an
unmasked one turns 159 of its checks red. The other two corrections are the
pyramid claims now recorded in [ARCHITECTURE §6.1](ARCHITECTURE.md#61-bit-parallel-primitives)
and on T3.4 — the header said the 2×2 box *was* `bitSlicedSum` at k = 4 without
saying that holds for a 1-bit source only, and `@param k` said "the MVP uses 3, 4
and 9" when k = 3 has no caller (the three-pixel median is `maj3`).

**Sanitizers.** Both spellings from
[GETTING_STARTED](GETTING_STARTED.md#sanitizers-for-the-kernels-where-undefined-behaviour-is-the-likely-bug)
were run over `test_bitslice` and are clean: UBSan at `-O2 -DNDEBUG` (the shipping
configuration) and UBSan+ASan at `-O1 -g` (assertions live). Worth doing here
because the file is full of shift counts — `1 << lane` in the enumeration,
`threshold >> p` in the comparator — and a shift at or past the word width is
undefined in exactly the way the T2.3 notes describe.

**No benchmark.** T2.7's done-when clauses are correctness only, and the
denominator question is not obvious: OpenCV's nearest equivalent to the composed
operation is `cv::min`/`cv::max` over three `CV_8U` frames, which is what the
OpenCV half already runs as an oracle. Measuring it as a *speed* claim belongs
with T3.1 and T3.2, where the operation has a caller and a working set to report —
and on the reference device, which is offline (`scripts/run_on_pi.sh` returns 77,
which is not a pass).

---

## Phase 2 experiments

These run **now**, not in Phase 4, because they gate code already written or about
to be. Follow the experiment protocol in
[ARCHITECTURE §9](ARCHITECTURE.md#how-performance-and-footprint-decisions-get-made)
and log results in [EXPERIMENTS.md](EXPERIMENTS.md).

**Write the decision rule down before measuring.** If a result contradicts a
documented claim, report it — do not adjust the code to fit the doc.

---

### T2.8 · E-1 · Does row alignment earn its memory? · `DONE`

**Depends:** T2.5
**Gates:** [D-4](ARCHITECTURE.md#d-4-word-granularity-alignment-by-default) — was
**provisional**, the only such decision in the project; **confirmed by this task
and no longer provisional**
**Completes:** [X-1](EXPERIMENTS.md), which measured cost but not benefit

**Question:** Does row alignment beyond word granularity measurably speed up any
bulk kernel, enough to justify up to 172% memory overhead?

**Decision rule** — *write this into the log before running anything:*
- Speedup < 5% on all kernels → D-4 confirmed, close E-1, **do not build a
  profile system**
- 5–20% → D-4 stands as default; larger alignment stays opt-in and is documented
  as worth it for specific kernels
- \> 20% on a kernel the frontend calls per frame → **reopen D-4**, report before
  changing anything

**Variants:** `rowAlignment` ∈ {word granularity, 16, 32, 64} bytes.
**Workload:** `bitwiseAnd` (T2.2) and `countNonZero` (T2.5) at 640×480 and 94×60
— the two extremes from X-1. Enough iterations for stable timing.
**Metric:** ns/pixel **and** allocated bytes. Both, per the protocol.
**Platform:** **close this on the Pi 4** via `scripts/run_on_pi.sh` (T1.10) — it is
the reference device and this is a cache question, which is exactly what a laptop
hides. Running x86 first is fine as a cheap signal, but x86 and Apple Silicon are
both non-authoritative here and cannot close E-1. Observe the Pi 4 measurement
discipline — 64-bit OS, throttle checks, governor, pinning:
[EXPERIMENTS.md § Measuring on the Pi 4](EXPERIMENTS.md#measuring-on-the-pi-4).
**Never measure this under emulation.**

**Done when:** [EXPERIMENTS.md](EXPERIMENTS.md) X-1 is `DONE` with the benefit
side filled in, D-4 is confirmed or reopened, and the benchmark is committed.

**RESULT — first band, D-4 CONFIRMED.** Reference device, two sets of three runs
([X-9](EXPERIMENTS.md), `bincv-cpp/results/alignment_benchmark.log`,
`benchmark/alignment_benchmark.cpp`); the second set was taken after a review
found the benchmark's own physical-bound check was computing the cache tier from
the wrong footprint, and it reproduced the first. Best of four alignments × two
kernels × two sizes was **1.015×**, inside both its 8.6% batch spread and its 1.6%
run-to-run scatter — a null result, not a small win. `countNonZero`, which has no
fast path and so isolates alignment alone, was flat to within **0.5%** at 640×480
across all four alignments. Two alignments were much worse: over-aligning disables
`ops/logic.hpp`'s contiguous fast path, so `bitwiseAnd` at 640×480 ran **3.3×
slower at align 32 and 4.8× slower at align 64** for 20% and 60% more memory.
**No profile system is built.** D-4 loses its "provisional" tag and X-1 is now
`DONE`.

---

### T2.9 · E-2 · Default word width · `DONE`

**Depends:** T2.5
**Gates:** `BinMat`'s default template argument — affects every kernel

**Question:** Is `uint32_t` the right default, or does `uint64_t` win on bulk
throughput?

**Decision rule** — *before measuring:*
- `uint64_t` wins by > 10% on bulk kernels **and** does not increase footprint at
  representative widths → change the default
- Within 10%, or footprint increases at small pyramid levels → keep `uint32_t`
  (memory wins ties)

Note the interaction: wider words round row strides up more coarsely, so the
footprint effect is worst exactly at upper pyramid levels. **Measure footprint at
94×60, not only at 640×480**, or this experiment will reach the wrong conclusion.

**Variants:** `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`.
**Workload:** same kernels and sizes as T2.8.
**Metric:** ns/pixel and allocated bytes at both resolutions.
**Platform:** the footprint half is architecture-independent and closes anywhere;
**the speed half closes on the Pi 4** (T1.10). This experiment is the most
sensitive of the three to a 32-bit OS — on `armv7l` every `uint64_t` operation is
synthesised from 32-bit pairs, so the result would describe the compiler rather
than the hardware. Confirm `aarch64` before recording anything.

**Done when:** logged as X-10 (the older "X-4" here was stale — X-4 is taken by
the set-pixel discontinuity entry), default confirmed or changed, benchmark
committed.

**RESULT — `uint32_t` KEPT, and the rule had to do real work.** Reference device,
three runs ([X-10](EXPERIMENTS.md), `bincv-cpp/results/wordwidth_benchmark.log`,
`benchmark/wordwidth_benchmark.cpp`). `uint64_t` reduces **1.94× faster at 640×480
and 1.56× at 94×60** (reproducible to 0.2%), and is a null result on `bitwiseAnd`,
which is memory-bound. So the ">10% on bulk kernels" clause is satisfied outright
— but the rule is a **conjunction**, and footprint rises **+33.3% at 94×60 and
+20.0% at 160×120** while costing nothing at either full frame. The second clause
fires and `uint32_t` stays: **a measured 1.94× declined on footprint grounds**,
which is principle 2 working as designed rather than an oversight. The trap this
task names was real — measured only at 640×480 every footprint row reads 0.0% and
the decision inverts. Promoted to
[D-14](ARCHITECTURE.md#d-14-uint32_t-is-the-default-word-type); the per-level word
width this exposes is registered as **E-9**, not decided.

---

### T2.10 · E-3 · Incremental versus recomputed window reductions · `DONE`

**Depends:** T2.6
**Gates:** T2.6's interface and T3.6's implementation

**Question:** At what window size does incremental/sliding accumulation beat
recomputation for overlapping windows?

**Decision rule** — *before measuring:*
- Recompute within 15% of incremental at 31×31 → **keep the simpler recompute
  API**, close E-3, and record that incremental state was rejected on data
- Incremental wins by > 15% at 31×31 → extend T2.6 with incremental state
  *before* T3.6 is written against the simpler form

**Variants:** recompute-per-window versus a sliding accumulator.

**SECOND AXIS, added by the T2.5/T2.6 review — composed versus fused.** T2.6's
primitives are each single-pass, but the 2×2 covariance composed out of them makes
**three** traversals of one window (`countNonZero` ×2 + `countAndSplit`), issuing
the popcounts a fused traversal would issue once. Measured at **1.30×** on the
reference device, three runs, with the popcount count identical on both sides — so
the cost is redundant traversal, past this task's own 15% line
([X-8](EXPERIMENTS.md), `bincv-cpp/results/reduce_target_benchmark.log`,
`benchmark/reduce_target_benchmark.cpp`).

- **Decision rule, same threshold:** a covariance-shaped entry point (returning
  `xx`, `yy`, `whenClear`, `whenSet` from one `visitRowWords` pass) beats the
  composition by > 15% at 31×31 → add it to T2.6 *before* T3.6 is written; within
  15% → keep the composition and record that the fused form was rejected on data.
- This axis is here rather than acted on because E-3 is the experiment whose stated
  gate is *T2.6's interface*, and adding an entry point because one benchmark liked
  it is the same mistake as adding incremental state for the same reason.
- A **third** interface question belongs with it, and it is a memory-versus-speed
  one, so it needs the same treatment: `countAndSplit`'s selector `c` must be a
  frame-sized plane (`sign_x ^ sign_y`, one bit per pixel, 38400 B at 640×480,
  formed once per pyramid level). A four-argument form taking `c0` and `c1` and
  XOR-ing them in the word loop would need no plane at all. Report both memory and
  speed, per CLAUDE.md, since this is precisely a case where the two goals may
  disagree.
**Workload:** window sizes 7, 15, 31 at realistic keypoint densities (~200
keypoints, per the reference `gftt_max_corners`); include the heavy-overlap case,
since that is what favors incremental.
**Metric:** ns per window, plus any additional memory the accumulator needs.
**Platform:** close on the Pi 4 (T1.10). The tradeoff turns on whether the
accumulator stays resident in a 32 KiB L1D — a laptop with four times the L1 would
favour incremental more than the deployment target does.

**Done when:** logged as the next free X-number (X-5 and X-8 are taken; X-8 already
carries this task's composed-versus-fused axis as a *finding*, not as its
decision), all three axes above answered, T2.6's API confirmed or extended, and
the benchmark committed.

**RESULT — all three axes moved OFF the simpler shape.** Reference device, three
runs ([X-11](EXPERIMENTS.md), `bincv-cpp/results/window_benchmark.log`,
`benchmark/window_benchmark.cpp`), logged as **X-11**.

| Axis | At 31×31 | Branch selected |
|---|---|---|
| 1 · incremental vs recompute | **7.3×** search, **20×** dense, for INC-ROW — the form adopted (INC-COL hits 36× on dense and is rejected; the 1.32× "sparse" column is the accumulator-split finding, not an incremental win) | extend T2.6 with incremental state before T3.6 |
| 2 · fused vs composed covariance | **1.27×** (`uint32_t`), **1.29×** (`uint64_t`) | add a covariance entry point before T3.6 |
| 3 · plane vs four-argument | plane **16–18% faster** per frame, a fifth plane at every level (**+25%** of the derivative working set; 38400 B at 640×480); four-arg **0 B** | four-argument form — memory wins the tiebreak |

**The surprise, and it needs stating plainly:** the sparse column is *not* an
incremental win. At 200 isolated keypoints the sliding path never executes, so the
"incremental" variant issues **identical popcounts over identical words** as the
shipped `countNonZero` and is still 1.32× faster at W=31. The difference is that
`impl::countViewRegion` accumulates into **one** `size_t` across the whole region —
one dependency chain through the popcount latency — where the variant accumulates
per row. That is a codegen finding about the shipped reduction, free of any
interface change, and it must not be read as evidence for incremental state.

The interface was **not** changed in the same commit as the measurement — that
inversion is what this task exists to prevent. The four changes the data mandates
are scheduled as **[T2.11](#t211--t26-api-extensions-mandated-by-e-3--done)**, and
**T3.6 is now blocked on T2.11 rather than on T2.10.**

---

### T2.11 · T2.6 API extensions mandated by E-3 · `DONE`

**Depends:** T2.10 (`DONE` — this task exists only because of what it measured)
**Gates:** [T3.6](#t36--lk-gradient-covariance--todo), which must be written against the
extended interface, not the current one
**Files:** `include/bincv-cpp/ops/reduce.hpp`, `tests/test_reduce.cpp`

Every item below is a **decision already made on data**
([X-11](EXPERIMENTS.md), [D-15](ARCHITECTURE.md#d-15-window-reductions-get-incremental-state-and-a-fused-covariance)),
not an open question. Do not re-measure to re-open one; if a number here looks
wrong, re-run `benchmark/window_benchmark.cpp` on the reference device and report
the discrepancy.

1. **Incremental window state, INC-ROW form.** A vertically-sliding accumulator
   over one column of window positions: the sum gains the incoming row's windowed
   popcount and loses the outgoing row's. Measured **7.3×** on an 8×8 search sweep
   and **20×** on a dense scan at 31×31. On **isolated** keypoints it is ~1.0×
   (0.98× at W=7): the sliding path never executes there, so that column's 1.32×
   is item 4's effect and not this one's — X-11 says so explicitly, and quoting it
   here as well would count one measurement twice. Note also that the 7.3× and 20×
   are against the **pre-item-4** recompute baseline; land item 4 first and they
   fall to roughly 5.6× and 15×, still far past the 15% line that selected this
   branch. It keeps one scalar of state and needs **no caller scratch**, which is
   why it is the form to expose rather than the per-column accumulator — that one
   is faster on a dense sweep (**36×**) but 12× *slower* on isolated keypoints and
   needs a `sweepWidth + W − 1` counter array, so it is a second shape and a
   second decision.
2. **A covariance-shaped entry point** returning `xx`, `yy`, `whenClear`,
   `whenSet` from a single `visitRowWords` pass. Measured **1.27×** (`uint32_t`)
   and **1.29×** (`uint64_t`) against the three-call composition at 31×31,
   reproducing [X-8](EXPERIMENTS.md)'s 1.30× and extending it across window sizes.
   The popcount count is identical on both sides, so this is redundant traversal
   and nothing else.
3. **A four-argument `countAndSplit(a, b, c0, c1, region)`** that XORs the two
   selector planes in the word loop. This one *costs* speed — the precomputed
   plane is 16–18% faster per frame even after paying to form it — and buys a
   **fifth plane at every pyramid level** on top of the four the covariance
   already reads: +25% of the derivative working set, which is 38400 B at 640×480
   and scales down with the level (~51 kB over four levels). [CLAUDE.md](CLAUDE.md)'s tiebreak decides it: memory wins when
   the goals conflict. Keep the three-argument overload; a caller that has already
   formed the plane for other reasons should not be made to unform it.
4. **Split `impl::countViewRegion`'s accumulator per row.** Not an interface
   change at all — the region reduction accumulates into one `size_t` across every
   row and word, which is a single dependency chain through the popcount latency.
   Per-row partial sums measured **1.15–1.32×** at LK window sizes on *identical*
   popcounts; this is the finding item 1's isolated-keypoint column actually
   measured. Cheapest item on this list, it benefits every region reduction in the
   file, and it is the one to land **first** — items 1–3 are all measured against
   the un-split baseline, so doing this first means their re-measurement reports
   the gain the shipped code will really have.

**Done when**
- All four land, each with a test in `tests/test_reduce.cpp`; the incremental and
  fused paths are checked to agree with the recompute/composed ones **window for
  window**, including windows clipped at the frame edge
- `benchmark/window_benchmark.cpp` is updated to time the shipped entry points
  rather than its own measurement copies, and re-run on the reference device
- `ops/reduce.hpp`'s docstrings state which shape to reach for and why, citing
  X-11 — a reader choosing between two entry points needs the access-pattern
  argument, not just the signatures
- [ARCHITECTURE §7.5](ARCHITECTURE.md#75-lk-gradient-covariance)'s code snippet is
  updated to the shapes this task lands — the fused covariance entry point and the
  four-argument `countAndSplit` — and its "`signXor` is a **frame-sized** plane"
  note stops being load-bearing, since after item 3 no caller is obliged to form
  one. (The paragraph that called incremental accumulation "not part of the
  interface" was already rewritten when T2.11 was scheduled; nothing is left to
  drop there.)
- `benchmark/window_benchmark.cpp`'s axis-1 numbers are re-measured **after** item
  4 lands, and the entry records the post-split ratios next to the pre-split ones
  rather than replacing them silently

**Verify:** V-ALL

**RESULT — all four landed, item 4 first.** `include/bincv-cpp/ops/reduce.hpp`
now carries:

| Item | What shipped |
|---|---|
| 4 | `impl::countRowRegion` / `countAndRowRegion` / `splitRowRegion` / `covarianceRowRegion` — every row body returns its OWN partial sum, and all four reductions loop over them. No interface change. |
| 1 | `SlidingWindowCount<W>` — one scalar of state, no caller scratch, column band clipped once at construction, rows clipped per position |
| 2 | `countCovariance(a, b, c, region)` → `CovarianceCount{xx, yy, xy}` from one `visitRowWords` pass |
| 3 | `countAndSplit(a, b, c0, c1, region)` and `countCovariance(a, b, c0, c1, region)` — selector XOR-ed in the word loop, no plane at any level. Both three-argument overloads kept. |

The fifth signature — the *fused* covariance taking two sign planes — is neither
item 2 nor item 3 alone but the conjunction the two decisions force: T3.6's spec
requires one traversal AND no scratch, and only that overload is both.

**Correctness.** `tests/test_reduce.cpp` gains `Reduce.Sliding_*` and
`Reduce.Fused_*`: whole-frame sweeps comparing every window position against
recompute and against the composition, edge-clipped positions included, plus a
freshly-constructed accumulator at each position. 546572 → 594184 core checks.
Mutation-tested — a `slideDown()` that never removes row 0 (a drift visible only
after the window passes the top of the image) fails all four `Reduce.Sliding_*`.

**Re-measured on the reference device, three runs**
([X-11b](EXPERIMENTS.md), `bincv-cpp/results/window_benchmark_t211.log`). Post-split
beside pre-split, at 31×31, INC-ROW:

| Pattern | X-11 pre-split | X-11b post-split | X-11's prediction |
|---|---|---|---|
| SEARCH | 7.33× | **5.96×** | ~5.6× — met |
| DENSE | 20.25× | **15.92×** | ~15× — met |
| SPARSE | 1.32× | 1.10× | ~1.0× — **not met**; see X-11b's amendment |

Axis 2 at 31×31: 1.20× (`uint32_t`), 1.27× (`uint64_t`) — still past 15%. Axis 3:
the plane is 11–14% faster per frame (was 16–18%) for the same +25% of the
derivative working set, so the tiebreak lands where it landed.

**AND ONE MEASUREMENT THAT CONTRADICTS THIS TASK'S OWN ITEM 4.** Timed directly
and interleaved rather than inferred, the accumulator split is worth **1.03–1.09×
at the LK window sizes, not the 1.15–1.32× recorded in X-11 and D-15**, and it is
a 5–6% loss at W=7 on the overlapping patterns. X-11 read that figure off axis 1's
isolated-keypoint column, which differs from a per-window `countNonZero` in two
ways rather than one. Reported in [X-11b](EXPERIMENTS.md) and amended in
[D-15](ARCHITECTURE.md#d-15-window-reductions-get-incremental-state-and-a-fused-covariance);
no decision moves, since the split costs nothing and still wins at W=15 and 31.

---

# Phase 3 — VIO Frontend Operations

---

### T3.1 · Denoise — median of 3 · `DONE`

**Depends:** T2.7
**Files:** `include/bincv-cpp/ops/denoise.hpp` (new), `tests/test_denoise.cpp`,
`benchmark/denoise_benchmark.cpp`

**Spec:** Three-pixel median. For binary input median equals majority, so this is
`maj3` over the three-pixel neighbourhood. Reference semantics (neighbourhood
orientation, border behavior) come from
`~/seal/SEAL/SEAL_HybVIO/HybVIO/SEAL/src/temporal_processing/denoise.cpp` —
**read it and match it**; do not invent the neighbourhood.

**Done when:** matches the reference implementation pixel-for-pixel on binary
input; benchmark committed.

**Verify:** V-ALL

**What landed.** `denoiseMedian3(src, dst)` — **API tier 3** by name, because
`cv::medianBlur` is a square-window median with a replicate border and this is
neither. The neighbourhood read out of the reference is **above, self, right** —
an asymmetric three-pixel L with no left and no below neighbour — and the border
is **zero fill**, which is not stated in the reference's comment but falls out of
its two `cv::Mat::zeros` neighbour matrices being written only on
`colRange(0, cols - 1)` and `rowRange(1, rows)`.

The test does not re-derive that from the comment: `tests/test_denoise.cpp` ports
the reference's own `cv::min`/`cv::max` calls, `cv::Mat::zeros` constructions
included, so the border comes from the same construction on both sides. It also
runs a per-pixel neighbourhood reference in the three configurations without
OpenCV, an all-ones border family (where a replicate border would give a
different image), and a family requiring the fused kernel to agree pixel for pixel
with `shiftDown` + `shiftLeft` + `majority3`.

**One pass, no scratch.** The above-neighbour is a row index and the
right-neighbour is computed into a register, so the kernel needs no caller-provided
buffer at all. [X-12](EXPERIMENTS.md) measured that against the composed spelling
on the reference device: **3.1–3.5× faster and half the memory**, so nothing was
traded and no D-record was earned. Against the reference implementation on the
same binary content stored as `CV_8U`: **57× at 640×480 with 28× less memory** —
quotable only with the working-set figure beside it, since it is substantially a
cache-residency result (X-12, and [X-6](EXPERIMENTS.md) before it). **The
pyramid-level ratios are not quotable in its place**: X-12 measures the
denominator's fixed per-call cost at 4.1 µs, which is 20% of the 94×60 frame and
0.4% of the 640×480 one, so ratios at different sizes are not comparable.

**A dead operation was removed rather than logged.** The kernel originally masked
its destination's trailing word as well as its source's; re-adding that mask
changes no check in any configuration, because masking the source already zeroes
both `c` and its shift past `width`. What replaced it is the coupling written down
in `medianRow3`: here the padding invariant is carried by a mask that exists for a
**border** reason, so a change to the border is a change to D-13 compliance.

---

### T3.2 · Threshold / binarize · `DONE`

**Depends:** T3.1
**Files:** `include/bincv-cpp/ops/threshold.hpp` (new), `tests/test_threshold.cpp`

**Spec:** Produce a 1-bit frame from a higher-precision source (`CV_8U` via
interop, or `QuantMat<N>`). Tier 1 semantics against `cv::threshold` for the
binary output case.

**Done when:** bit-exact against `cv::threshold` with `THRESH_BINARY`.

**Verify:** V-ALL

**What landed.** Two entry points, split by tier rather than by convenience:

- `threshold(const cv::Mat&, dst, thresh)` — **API tier 1**, bit-exact against
  `cv::threshold(src, dst, thresh, 255, THRESH_BINARY)` for every `thresh` with
  `|thresh| < 2^31`, so it takes OpenCV's name. Guarded by `BINCV_WITH_OPENCV`;
  nothing the embedded claim rests on sees it.
- `binarize(planes, dst, thresh)` and its `QuantMat<N>` wrapper — **API tier 3**,
  since OpenCV has no N-bit image type. It must not borrow the Tier 1 name, and
  does not. The arithmetic is T2.7's `thresholdGE`, whose whole (value, threshold)
  input space is already enumerated by `tests/test_bitslice.cpp`.

**The comparison is strictly greater than**, on both, and that is the whole risk
in this task: `>=` instead of `>` moves exactly the pixels *equal* to the
threshold, which a mid-range sample can miss. So the boundary is enumerated rather
than sampled — a 256-pixel ramp holding every `uint8` value exactly once,
thresholded at every integer 0..255 plus eight fractional and out-of-range values,
against `cv::threshold` on the same `cv::Mat`; and every threshold from 0 to
`MaxValue + 1` on the N-bit side. `cv::threshold` **floors** its `double` for a
`CV_8U` source, which is why 127.5 and 127 must give the same image and why the
kernel reduces the parameter to one integer cutoff.

**No `maxval` and no `type` parameter.** In a one-bit destination the set value is
1 by construction, `THRESH_BINARY_INV` is `bitwiseNot` of this, and the four
truncating types cannot be expressed at all — an enum whose every other value
asserted would be a promise the file cannot keep.

**One guard needed a case written for it.** `binarize`'s `thresh >= MaxValue`
shortcut is not an optimisation: without it `thresh + 1` wraps to 0 at `UINT_MAX`
and `thresholdGE` answers every lane, returning an all-ones image for the
threshold that should select least of all. No ordinary sweep reaches a threshold
that can wrap, so bypassing the shortcut left all 26 320 checks green until
`Threshold.BinarizeSaturated_*` was given a `~0u` case.

**Four shapes review found untested, each now swept** (see
`tests/expected-checks.txt` for the mutation that each one kills):

- **An over-aligned destination on the Tier 1 path.** Every Tier 1 destination was
  a `BinMat` at `DefaultRowAlignment`, where `alignedWidth == minRowWords` and the
  row stride is invisible; deleting the stride from the kernel left all 27 040
  checks green. D-4 is provisional, so a padded destination is a supported shape.
- **A non-continuous `cv::Mat`.** Every source was freshly allocated, so `src.step`
  was never exercised — and a cropped frame (`cv::Mat` ROI) is how a VIO frontend
  hands in a region. `Threshold.Roi_*` is new.
- **Thresholds outside `int`'s range**, where `cv::threshold` is undefined and
  cannot be the reference. `Threshold.OutOfDomain_*` pins binCV's answer against
  the arithmetic at ±1e300, ±∞, ±2^31, ±2^32 and `NaN`. The kernel's own reduction
  was restructured so no non-finite `double` reaches an `int` conversion.
- **More than eight planes.** The plane-view entry point takes `N` from its
  argument and `QuantMat`'s `N <= 8` cap does not reach it. At 33 planes with a
  pixel holding 2^32, `thresh = UINT_MAX - 1` selected it and `thresh = UINT_MAX`
  did not — an answer not monotone in the threshold, because `maxValue` saturates.
  `N > 32` is now a compile error (the cutoff such a caller needs is 2^32 and an
  `unsigned` cannot hold it) and `Threshold.BinarizeWideCutoff_*` sweeps `N = 32`,
  the widest that remains expressible.

---

### T3.3 · Morphology · `DONE`

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

**What was built.** `include/bincv-cpp/ops/morphology.hpp` (`StructuringElement`,
`erode`, `dilate`, `morphologyEx`, `morphologyExNeedsScratch`),
`tests/test_morphology.cpp` — a **core** suite whose Tier 1 half sits behind
`BINCV_WITH_OPENCV` — eight `test_assert_abort` death cases, and two benchmarks:
`benchmark/morphology_benchmark.cpp` (against OpenCV) and
`benchmark/morphology_path_benchmark.cpp` (binCV against binCV, pricing the 3×3
special case). 299181 checks under OpenCV, 172632 in the three configurations
without it, and 172632/172632 on the reference device (`throttled=0x0` before and
after).

**The structuring element, and why it is a value rather than a container.**
`{shape, cols, rows, anchorX, anchorY, mask}` — 32 bytes, no allocation, no
capacity limit. The three parametric shapes are evaluated on demand by
`spanOfRow()` / `activeAt()`, transcribed branch for branch from
`cv::getStructuringElement`, so any odd size works and `Morphology.ElementMatchesOpenCv`
pins all of it cell by cell at 121 sizes × 3 shapes × 4 anchors. Three of
OpenCV's behaviours are surprising and are inherited deliberately: a 3×3
`MORPH_ELLIPSE` is a PLUS, a `MORPH_CROSS` is centred on the ANCHOR rather than on
the element, and a 1×1 element is filled whatever its shape says. `mask` is a
non-owning pointer to a caller's cell array — the escape hatch for an arbitrary
shape, and the thing that makes an asymmetric element expressible.

**The kernel is the FUSED form of the composition, and that choice was measured,
not argued** — [D-16](ARCHITECTURE.md#d-16-morphology-fuses-the-shift-and-the-fold-and-only-the-compound-ops-take-scratch)
/ [X-13](EXPERIMENTS.md#x-13--t33-morphology-against-cverode--cvdilate--done). The
composed spelling (a `shift` per element cell into a temporary, combined with
`ops/logic.hpp`) would need a caller-provided scratch frame on `erode` and
`dilate` themselves, because a kernel may not allocate one. Fusing the fold into
the destination row removes it entirely. On the reference device at 640×480 the
fused form is 2.24× (erode 3×3) and 3.12× (dilate 3×3) faster than the composed
one at two frames rather than three — **and 1.18× SLOWER on a 5×5 ellipse**, which
is the branch the decision rule was written for and is recorded as a known cost
accepted for footprint rather than reversed.

**Against OpenCV on the reference device at 640×480 under `BORDER_CONSTANT`, with
the working set beside every number, including where binCV loses:** erode 3×3 rect
**0.99× (`uint32_t`) / 1.89× (`uint64_t`) at 8× less memory** — a tie at the
default word width — dilate 3×3 **1.48× / 2.73×**, `morphologyEx` OPEN **1.14× /
2.13× at 5.33× less memory** (not 8×: `cv::morphologyEx(OPEN)` allocates no
temporary, measured, so OpenCV holds two frames there against binCV's three), and
erode 5×5 ellipse **0.52× / 0.73×** — slower, at the same 8× less memory, because
the general path is 17 scalar shifted folds per word against a vectorised OpenCV
kernel. **On the four NON-CONSTANT border types binCV is slower again — 0.76× at
`uint32_t`** — because the `2 × reach` edge columns of each row are recomputed per
pixel; every published ratio names its border type. **The ladder rises as the frame
shrinks (2.92× at 94×60), and that is NOT cache residency**: `cv::erode`'s fixed
per-call cost is 2.77 µs against binCV's 0.22 µs — 18% of an entire 94×60 frame —
and binCV's own ns/pixel is nearly flat. The 640×480 number is the quotable one.

**Two of those numbers replace ones this record previously carried, and the
reason is worth keeping.** The erode 3×3 `uint32_t` ratio was quoted as 1.11×;
four runs now give a tie, while the *same call* timed in the other benchmark's
translation unit reproduces 1.11× at a 0.1% spread. The 9% is code layout between
two object files, so the case is a tie *within the instrument's precision* and
1.11× was quoted to a precision it does not have. The OPEN footprint ratio was
8.00× on the assumption that `cv::morphologyEx` needs a temporary; probed with
`VmHWM` one op per process, only `MORPH_GRADIENT` allocates one. See
[X-13](EXPERIMENTS.md#x-13--t33-morphology-against-cverode--cvdilate--done).

**The 3×3 special case T3.3 asks for is priced, not assumed.**
`benchmark/morphology_path_benchmark.cpp` runs the same kernel with the special
case refused: the general path costs **2.1×–3.7×** across the whole ladder at both
word widths. Its docstring's original justification (fewer `extendedRowWord`
calls) was wrong — the general path's window branch hoists that call per word too
— and what it actually removes is the per-cell loop, the data-dependent shift
count and the per-row span queries.

**A performance defect the border axis found.** The non-constant border fixup
walked every column of every row and skipped the interior by test, paying `width`
iterations to rewrite `2 × reach` pixels. At 640×480 that cost 241–260 µs against
19.5 µs for the same call under `BORDER_CONSTANT` and made binCV **6–10× slower
than `cv::erode`** on four of the five border types, while the benchmark measured
only the fifth. It now indexes the two bands directly.

**The asymmetric elements are the suite's load-bearing idea, and it is a
measurement.** All three parametric shapes are point-symmetric about their centre,
so at the default anchor negating every offset changes nothing and a rect/cross
suite cannot see an inverted shift sign. Measured before this suite existed: a
5040-case sweep of the three shapes at centred anchors passed **5040/5040 with the
sign flipped**. Against the shipped suite the same mutation fails 53016 checks —
**zero of which name a centred parametric element** — decomposing exactly by the
check family that reported them: 31584 one-cell-equals-`shift()`, 8732 Tier 1
against OpenCV, 6404 compound-op definitions, 6168 asymmetric elements, 128 wide
elements. `tests/expected-checks.txt` carries the full mutation table, now with
each of the **four** offset sites mutated on its own as well as together.

**The fallback recurrence had to be earned twice, and the second time was a
reviewer's find.** `morphRowGeneric`'s fallback — the branch taken when an element
reaches a whole word sideways — was reached only by wide *centred* rects and by a
sparse mask whose cells sat symmetrically about its anchor. Both satisfy `E == -E`,
so flipping that one site's offset sign left the suite at **298541/298541** while
the same mutation at any of the other three sites went red. Anchoring the wide
rects at column 0 and at `size - 1` as well makes the offset set asymmetric, and
the mutation now fails 42 checks. A path reached is not a path discriminated.

**Every precondition now has a death test.** T3.3 originally registered none, and
neutering all of `ops/morphology.hpp`'s `BINCV_ASSERT`s at once left the suite at
172632/172632 with assertions live — a correctness suite never violates the
preconditions it obeys. Eight cases in `tests/test_assert_abort.cpp` cover them,
two of which have no analogue anywhere else in the project: the **D-16 scratch
contract** (`morphologyEx` is the only kernel taking a caller-provided
intermediate, and an undersized one overruns the caller's array rather than
answering wrongly) and **element validity** (`StructuringElement::valid()` has no
other caller, and an element with no set cell yields a uniform frame that reads as
a plausible image).

**Two defects this suite caught in its own author's work**, both worth recording
because each looked right:
- The asymmetric catalogue's original 5×5 mask was the main DIAGONAL, which is
  invariant under a 180° rotation and therefore symmetric — `Morphology.ElementStructure`
  rejected it, and it was replaced by a wedge.
- The wide-element case originally used solid rects, and a mutation that
  perturbed the fallback recurrence's offset by one still passed 298381/298381:
  sliding a *contiguous* offset range by one only changes which columns fall
  outside the image, and those read the border either way. A sparse wide mask sees
  it.

**A performance defect the benchmark caught, and the rule it produced.**
`activeAt()` was first called once per (word, cell) in the inner loop. For
`MORPH_ELLIPSE` that is a `sqrt`, and a 5×5 ellipse erosion of a 640×480 frame ran
at **4.23 ns/pixel — 17× slower than `cv::erode`**. The shape query is now
evaluated once per element row (or once per call for the 3×3 table), which is
correct only because each parametric shape's row is a solid run — a property
`Morphology.ElementStructure` now asserts in both directions rather than assumes.

**`BORDER_WRAP` has no Tier 1 denominator anywhere.** `cv::morphologyEx` refuses
it by assertion (`columnBorderType != BORDER_WRAP`), so binCV's wrapped morphology
is stood behind entirely by the core half — the per-pixel reference over an
independently written iterative border mapping, and agreement with
`ops/shift.hpp`. That is the second reason this suite is core rather than
OpenCV-only.

**What was deliberately not built:** `iterations` (n > 1 needs a second buffer to
ping-pong through, and would make the signature understate its own memory), and a
literal `borderValue` on `morphologyEx` (each step uses the morphological default
for its own operation, which is what `cv::morphologyEx` does with its default).

---

### T3.4 · Pyramid downsample · `DONE`

> **⚠️ BOTH BLOCKING GAPS ARE NOW CLOSED.** The first by
> [D-17](ARCHITECTURE.md#d-17-horizontal-decimation-is-word-local) ahead of the
> rest of the task, the second by this one
> ([D-18](ARCHITECTURE.md#d-18-the-n-bit-box-is-a-multi-bit-adder-and-the-requantization-is-a-documented-rescale)).
> Both notices are kept as written, because what the gaps looked like before they
> were closed is the useful part of the record.
>
> ~~**No primitive expresses horizontal decimation**~~ — **RESOLVED.**
> `ops/resample.hpp` provides `decimateColumnsBy2(src, dst)` and the free
> `rowsDecimatedBy2(src)` view, chosen by measurement on the reference device
> ([X-14](EXPERIMENTS.md), [D-17](ARCHITECTURE.md#d-17-horizontal-decimation-is-word-local)).
> E-8 is closed and its framing was wrong: the winning route is word-local, so it
> takes `(src, dst)` and **needs no scratch and no prepared plan** — `pyrDown`
> carries nothing through for the subsample half.
>
> ~~**STILL OPEN — this task owns it.**~~ **RESOLVED.** From T2.7: the 2×2 box
> **is** `bitSlicedSum` at k = 4 only for a **1-bit** source. For an N-bit level
> the replication route costs k = 4·(2^N − 1) — 124 inputs at N = 5 — so an N-bit
> pyramid level needed a different formulation, not a bigger k. It got one: a
> bit-sliced **multi-bit** add, a tree of three ripple-carry additions costing
> **3·N + 1 full-adder stages**, equal to the single-bit route at N = 1 and 40×
> cheaper at N = 8. The replication route stays in `impl::` under test and under
> measurement so the comparison is reproducible ([X-15](EXPERIMENTS.md)).

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

- 2×2 sum via `bitSlicedSum` (T2.7), then requantize to `NOut` bits — **but read
  `ops/bitslice.hpp`'s "what pyrDown still needs" section first.** T2.7's review
  found two prerequisites this task owns and neither exists yet:
  - **`bitSlicedSum` is single-bit and equal-weight**, so it is the 2×2 box for
    `NIn == 1` only. At `NIn > 1` the only composition available is replicating
    plane *p* of each pixel 2^p times, which is correct and exponential
    (*k* = 4·(2^NIn − 1) — 124 inputs at `NIn = 5`, i.e.
    [§7.2](ARCHITECTURE.md#72-pyramid-downsample--box-22)'s level 3). Add the
    bit-sliced multi-bit add here, where the caller fixes its shape.
  - ~~**Nothing decimates horizontally.**~~ **Done, ahead of the rest of T3.4.**
    Vertical subsampling is a stride-doubled view and costs nothing; horizontal is
    `decimateColumnsBy2()` in `ops/resample.hpp`. The rule was written and
    committed before the device ran, E-8 is resolved, and the answer was that
    E-8's speed-against-footprint framing did not survive contact with the
    measurement ([X-14](EXPERIMENTS.md),
    [D-17](ARCHITECTURE.md#d-17-horizontal-decimation-is-word-local)).
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

**RESULT — shipped as `ops/pyramid.hpp`, and the reference disagreed with the
docs in two places.**

The formulation is `boxSum4` (3·N + 1 full-adder stages, linear in N) then a
requantization that is a constant multiply, a constant add and a restoring
division by a constant — quadratic in `NOut`, linear in `NIn`, exponential in
neither. Odd extents **replicate** the edge pixel so the divisor stays 4 and the
destination stays ceil(w/2) × ceil(h/2). The kernel takes **no scratch parameter
and allocates nothing** — the four 2×2 phases are gathered in registers with
[D-17](ARCHITECTURE.md#d-17-horizontal-decimation-is-word-local)'s word-local
unshuffle — and its stack is bounded at compile time and independent of image
size: `impl::pyrDownAutomaticWords(NIn, NOut) = 8·NIn + 2·NOut + 6` words plus
2·NIn + NOut row pointers, measuring on the reference device (aarch64)
**272 B at NIn = 1 / NOut = 3 / `uint64_t` and 912 B at 8 / 8**. (This block first said NIn + NOut + 2 words, which is the
widest single intermediate rather than the total and understated the frame by
5×–10×; the measurement and the two restructurings that were tried and rejected
for costing *more* stack are in [X-15](EXPERIMENTS.md).)
[D-18](ARCHITECTURE.md#d-18-the-n-bit-box-is-a-multi-bit-adder-and-the-requantization-is-a-documented-rescale)
records all three choices and the three deviations from the reference.

**Two documented claims turned out to be wrong, and are corrected rather than
worked around** ([X-15](EXPERIMENTS.md)):

- `cv::blur` on `CV_8U` **rounds the mean up**, not to nearest — measured, its
  2×2 box is `ceil((a+b+c+d)/4)`. That is where §7.2's `192` comes from.
- §7.2's **1/3/4/5 bits is a frame statistic, not a requirement.** The reachable
  alphabet is 2/5/17/65 — 1/3/5/7 bits — and X-2 counted what a 256² frame
  happened to contain. X-2's conclusion is unchanged and strengthened.

Footprint, 640×480 × 4 levels: **84 240 B uncapped down to 51 120 B
re-binarized**, against 408 000 B for the `CV_8U` equivalent — 4.84× to 7.98×.
The cap's whole range is **1.65×**, which is smaller than it looked and is what
E-7 (T4.1) is really trading against accuracy.

---

### T3.5 · Binarized spatial derivative · `DONE`

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

**RESULT — shipped as `ops/derivative.hpp`, and the container did not fight.**

`derivativeX` / `derivativeY`, **API tier 3** with tier 1 border semantics, taking
plane views (D-5) with `QuantMat` / `SignedQuantMat` wrappers. One pass per axis,
**no scratch and no allocation**; padding bits cleared in every destination plane
*including the sign plane*.
[D-19](ARCHITECTURE.md#d-19-the-derivatives-border-is-reflect-101-and-its-sign-is-the-borrow)
records the three choices this task had to make.

**Two properties of `cv::filter2D` decided the semantics, and both were checked
against the real function rather than against its documentation.** It
**correlates** — `dst(x) = src(x+1) − src(x−1)`, so the `+1` tap is the RIGHT
neighbour — and its default border is **`BORDER_REFLECT_101`, not zero**, which is
`cv::BORDER_DEFAULT`. `ops/derivative.hpp` therefore defaults to reflect-101,
**deliberately breaking with [D-12](ARCHITECTURE.md#d-12-a-shift-carries-a-border-and-the-fill-is-the-callers)'s
`BORDER_CONSTANT` default**, and that is the better answer as well as the
compatible one: reflect-101 makes both taps read the same pixel on the outer
column and row, so the derivative is exactly 0 there, where a zero fill
manufactures an edge around the whole frame for T3.7 to detect as a ring of
corners. `Derivative.OpenCvFilter2D_Direction` and `..._BorderDefault` pin both
against `cv::filter2D` itself.

**The N-bit path is `2·N` adder-class stages** — one ripple-borrow subtraction,
then one conditional two's-complement negate — against `2·(2^N − 1)` single-bit
inputs for the replication route T3.4 rejected. Measured on the reference device
at 640×480, the cost tracks the stage count ([X-16](EXPERIMENTS.md)).

**THE CANONICAL-ZERO RULE HOLDS BY CONSTRUCTION.** The sign plane *is* the
subtraction's borrow-out, and a borrow means `a < b`, which forces a non-zero
magnitude — so no input can produce a set sign over a zero magnitude and there is
no canonicalization pass. `Derivative.Reference*_*` asserts it per pixel across
the whole sweep (6 border cases × 140 sizes × 2 axes × 3 values of N × 4 word
widths) and counts violations separately from value mismatches so one cannot hide
the other.

**Against `cv::filter2D` on the reference device** ([X-16](EXPERIMENTS.md),
`benchmark/derivative_benchmark.cpp`, `results/derivative_benchmark_pi4.log`):
both axes cost **0.201 ns/pixel at 640×480 in 192 000 B**, against **5.007
ns/pixel in 1 536 000 B** for the two `cv::filter2D` calls the reference runs —
**24.9× faster in 8.0× less memory**. Unlike X-6, X-12 and X-13, that is **not
mainly a cache-residency result**: with each side's measured per-call floor
subtracted the ratio is 24.8× at 640×480 and 20.0× at 160×120 where both working
sets fit, so ~20× is arithmetic and at most a quarter is residency. The fused
kernel is also **2.94× faster than the composed spelling** as well as 1.40×
smaller, so X-16's live branch — record a speed cost accepted for footprint — did
not fire.

**What the container cost, since T3.5 existed to ask.** `SignedQuantMat` fit: the
destination width is exactly right (a difference of two N-bit values needs N
magnitude planes and a sign, which is `SignedQuantMat<N>`), the canonical-zero
rule falls out of the arithmetic, and `TernaryMat` is the N = 1 instance with no
adapter. Three frictions, all small and none worth an interface change:
`magnitude(i)` and `sign()` hand out one plane at a time, so a kernel wanting the
array writes a loop (`pyrDown` does the same for `QuantMat`); those accessors are
checked in every build, so the *container* wrapper can throw where the view kernel
cannot — with a loop index bounded by N, it cannot fire; and nothing in the
container can express "these two planes belong to one image" to the aliasing
predicate, so `ops/derivative.hpp` checks destination planes against each other
by hand. A sign plane aliasing a magnitude plane would otherwise compile, run, and
quietly break the canonical-zero rule.

**REVIEW OUTCOME — the kernel was right; the things standing behind it were not.**
Four reviewers found no sign inversion on either axis, no canonical-zero
violation, no dirty sign-plane padding, no undocumented border deviation and no
exponential N-bit path. What they found was that several of T3.5's *guarantees*
were unbacked or misdescribed, and all of it is now fixed:

- **The entire precondition block was deletable with every configuration green.**
  Replacing all fourteen conditions in `impl::checkDerivativeArgs` with a literal
  `true` left `test_derivative` reporting a byte-identical `47593/47593` in the
  Debug core-only build and left `verify.sh` ALL GREEN. That included the two
  checks with **no analogue anywhere else in the project** — destination versus
  destination, which is the class a multi-plane output introduces and whose
  failure mode is exactly the canonical-zero violation this task exists to
  prevent, visible only in T3.6's cross term. Six death tests now cover them
  (`derivative-dims`, `-short-stride`, `-in-place`, `-sign-alias`, `-mag-alias`,
  `-border-type`), and each was watched **fail** against the neutered kernel
  before being kept.
- **A mutation number was quoted against a kernel that no longer exists.** The
  "right-border fixup removed" row read 31700, which was measured while the source
  trailing-word mask was still in; against the shipped, unmasked kernel it is
  31672. The 28-check gap is exactly `Derivative.DirtyPadding*` — and it is the
  proof that the mask is dead *because* the fixup is there, so the two numbers are
  now both recorded as the coupling itself.
- `derivativeReplicatedInputs(n)` was **undefined for n ≥ 64** (`size_t{1} << n`);
  it returned 0 at 64 and 137438953470 at 100 under `-fsanitize=undefined`, which
  `-Wconversion` cannot see and the gate runs no sanitizer to catch. It now
  saturates, and `Derivative.Stages` pins both sides of the domain.
- The composed-versus-fused footprint multiplier was **1.67× in one place and
  1.40× in three others** under the same phrase — per-axis against both-axes.
  Both are stated, with 1.40× named as the one a footprint claim should use.

**ONE ITEM IS OPEN AND T3.5 IS NOT FULLY CLOSED ON IT.** The N-bit ladder's
`cv::filter2D` comparison was called *the denominator* in three places and was
never timed — it ran once, outside every timed region, as a correctness oracle —
so the **N ≥ 2 path, which every pyramid level above 0 runs, has no measured
OpenCV ratio**. The benchmark now times it and prints working-set columns beside
it, but the device re-run **tripped the soft temperature limit** (`throttled after
0x80000` → `RESULTS INVALID`) and the flag is sticky until reboot, so no device
number was taken and none is recorded. See the
[X-16 amendment](EXPERIMENTS.md) for the exact command that closes it. The
headline result above is unaffected — it comes from the main size table, and rule
3's linear-in-N verdict reads binCV's own curve.

---

### T3.6 · LK gradient covariance · `TODO`

**Depends:** T3.5 and **T2.11** (`DONE`). E-3 is settled (T2.10, [X-11](EXPERIMENTS.md)) and it went the way this line was written to guard against: incremental state and a fused covariance entry point both won, so T2.6's original shape was **not** the one to build on. T2.11 landed the extended interface, so this task is now unblocked on that side and writes against `countCovariance(dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(), dy.constSign(), window)` — one traversal, no scratch.
**Files:** `include/bincv-cpp/ops/covariance.hpp` (new)

**Goal:** The load-bearing operation
([§7.5](ARCHITECTURE.md#75-lk-gradient-covariance)).

**Spec** — written against the **T2.11** interface. The earlier version of this
spec prescribed exactly the shape E-3 rejected (three composed calls plus a
caller-provided frame-sized selector plane); [D-15](ARCHITECTURE.md#d-15-window-reductions-get-incremental-state-and-a-fused-covariance)
supersedes it.

```cpp
struct GradientCovariance { int64_t sumXX, sumYY, sumXY; };

// Ternary (pyramid level 0)
GradientCovariance gradientCovariance(const TernaryMat<W>& dx,
                                      const TernaryMat<W>& dy,
                                      Rect window);
```

Built on the **fused covariance entry point** T2.11 adds — one `visitRowWords`
pass returning `xx`, `yy`, `whenClear`, `whenSet` — not on three separate T2.6
calls. Composing it out of `countNonZero` ×2 plus `countAndSplit` costs 1.27–1.29×
for redundant traversal (X-11 axis 2), which is why the entry point exists.

`sumXY` is `whenClear − whenSet`, signed, via `crossTerm()`.

The selector is **not** a caller-provided frame-sized plane. T2.11's
four-argument `countAndSplit(a, b, c0, c1, region)` XORs `dx.sign()` and
`dy.sign()` inside the word loop, so the covariance needs **no scratch at all**;
the plane form stays available for a caller that already has one, and is 16–18%
faster when it does, but it is not what this operation requires (X-11 axis 3,
memory wins).

Where a caller sweeps a column of window positions — the corner response of T3.7,
and any search sweep — reach for T2.11's **INC-ROW** incremental form rather than
calling this per position.

**Done when**
- Matches a per-pixel float reference exactly (all values are integers, so exact
  agreement is required, not approximate)
- Correct for windows clipped at image edges
- Benchmarked across window sizes 7, 15, 31, against both the fused and composed
  forms, so the T2.11 entry point's advantage is confirmed at this level too

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

### T3.9 · E-4 · Generic-N cost versus specialized paths · `TODO`

**Depends:** T3.5
**Gates:** whether N stays arbitrary or gets capped

**Question:** Does the bit-sliced generic-N implementation regress the specialized
N=1 and ternary paths?

**Decision rule** — *before measuring:*
- Specialized paths within 5% of a hand-written binary-only implementation →
  arbitrary N confirmed at no cost to the common cases
- Regression > 5% → report before acting; options are stronger specialization or
  capping N, and which is right depends on where the cost comes from

**Variants:** `QuantMat<1>` and `TernaryMat` through the generic path versus their
specializations, versus a hand-written binary-only reference.
**Workload:** T3.5's derivative and T2.5's reductions.
**Metric:** ns/pixel and code size (`size` on the built object).

**Done when:** logged as X-6, specialization strategy confirmed or revised.

---

# Phase 4 — Validation

Phase 4 holds only the experiments that genuinely cannot run earlier — those
needing the complete frontend. The decisions gating Phases 1–3 were settled in
T2.8–T2.10 and T3.9, where they belong.

Each task produces a committed measurement and a written conclusion, and each may
invalidate a decision — **that is the point.** A contradicted claim gets reported,
not worked around.

### T4.1 · E-7 · Pyramid level bit depths · `TODO`

**Depends:** T3.8
**Question:** How many bits does each pyramid level need to preserve tracking
accuracy? Measured natural growth is 1/3/4/5 bits
([X-2](EXPERIMENTS.md)), but the reference never chose that — it fell out of
using `CV_8U`.
**Decision rule** — *before measuring:* adopt the smallest per-level depth whose
tracking accuracy is within the Phase-4 tolerance of the full-precision pipeline.
Report the accuracy/footprint curve, not just the chosen point.
**Metric:** trajectory accuracy versus pyramid footprint, per configuration.
**Also:** re-run X-2 against the real reference pyramid path, closing that entry's
caveat.

### T4.2 · E-6 · Hybrid LK versus binary block matching · `TODO`

**Depends:** T4.1
**Question:** Does fully bit-parallel tracking (census/Hamming) match hybrid LK's
accuracy, and what does it cost?
**Decision rule** — *before measuring:* switch only if accuracy is within
tolerance **and** the footprint or speed win is material; otherwise hybrid stands
and route (a) is closed.

### T4.3 · E-5 · End-to-end validation · `TODO`

**Depends:** T4.2

**Split deliberately, because the two halves need different things and only one
of them is binCV's to claim.**

**T4.3a — binCV's own result. Needs OpenCV and image sequences; no VIO stack.**
- tier 2 kernels agreeing with the reference frontend **frame by frame**: feature
  positions, flow vectors, track lifetimes
- **peak footprint** over the whole frontend operation set, measured end to end
- **speed** against the byte-per-pixel denominator

That is three of the four [success criteria](ROADMAP.md#success-criteria) —
the fourth (tier 1 bit-exactness) is already enforced per operation.

**T4.3b — a sufficiency check, not a binCV claim. Needs a VIO stack.**
Swap the frontend into an existing VIO framework and measure trajectory error.
`~/seal/SEAL/SEAL_HybVIO` ships Docker repro scripts, so this is "run their
harness with our frontend", not "build a VIO system".

**Record 4.3b as evidence that the kernels are sufficient, attributed to the
integration.** Trajectory accuracy is not binCV's result — building the VIO
framework is a separate repository's job
([ARCHITECTURE §1](ARCHITECTURE.md#what-bincv-is-not)). 4.3a must not be gated
behind 4.3b.

---

# Phase 5 — Platform Hardening

Detailed once Phase 4 produces numbers — measurements determine which kernels are
worth vectorizing. Scope is fixed
([ROADMAP Phase 5](ROADMAP.md#phase-5--platform-hardening)): NEON reference
kernels, aarch64 cross-compilation and on-hardware validation, x86 portability,
and Tier 2 correctness in CI.
