# binCV — working notes

## What this project is

binCV processes **low-bit-width image frames** — binary, ternary, few-bit quantized — at
their true bit width (1 bit per pixel, not 8), keeping OpenCV's API shape. It targets
desktop, mobile and embedded CPUs.

**Performance and memory footprint are co-equal goals. When they conflict and no explicit
choice has been made, memory wins.**

## Where to look

| | |
|---|---|
| [GitHub Issues](https://github.com/ryanhou28/bincv/issues) | **Start here.** All open work, labeled |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the library is put together, and why |
| [docs/API.md](docs/API.md) | Generated API reference — regenerate with `scripts/gen_api_index.py` |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Build, use, conventions |

Maintainer-only working files — the measurement log, the reference-device scripts, the
one-off probes — live in `.local/` and are **not** part of the repository.

## How performance and footprint decisions get made

**Measure the alternatives, weigh the result, then decide — and record all three.**
Argument alone does not settle a performance question; neither does a benchmark without a
stated decision rule.

- **Write the decision rule before measuring.** What result favors which choice, written
  down first. Deciding afterwards invites fitting the conclusion to the numbers.
- **Compare alternatives**, not one option, on representative workloads.
- **Report memory and speed together** — they trade off, so one alone cannot be weighed
  against goals that conflict.
- **Commit the benchmark.** Every performance claim must be reproducible.
- **Pick the right baseline.** The bar for a new implementation is the best existing
  option, not the worst. Measuring against a fallback nobody would use makes anything
  look like a win.

**A new operation gets a benchmark arm when it is written, even with no caller.** A kernel
nobody calls makes no performance claim, so it ships correct, untimed and unoptimized, and
nothing notices until something calls it. That has happened here: two kernels written
bit-exact and benchmarked by nobody turned out to be **78% of the whole frontend** the day
they got a caller.

**A vector arm must be switchable off, and the benchmark must show it is on.** A
mis-attached `#define` once compiled a vector block out entirely, and three consecutive
"improvements" were measured against it. Two cheap things catch this: a runtime switch so
the benchmark can time both arms, and a case where the fast path's own gate excludes it —
if that case does not report ~1.00×, the fast path is not running where you think it is.

## Verify before committing

```bash
./scripts/verify.sh      # ~35 s, four configurations, warnings fatal
./scripts/verify_arm.sh  # aarch64 correctness under emulation; skips without Docker
python3 scripts/check_links.py
```

`verify.sh` builds and tests four configurations — Release+OpenCV, Release core-only,
`-fno-exceptions` core-only, and **Debug** core-only — with `-DBINCV_WERROR=ON`, and exits
non-zero if anything fails. It starts with a **gate self-check**: two throwaway builds
that are *supposed* to fail. A gate nobody has watched fail is not known to work.

Read the two numbers in its summary table:

- **CTEST** — cases run.
- **CHECKS** — assertions executed. A drop is a regression even when every case still
  passes, so per-suite floors live in `bincv-cpp/tests/expected-checks.txt` and a count
  below one of them fails the run. Raising a floor is a reviewed edit
  (`./scripts/verify.sh --update-checks-baseline`, then commit the diff).

Each configuration also has to *be* the configuration it claims to be: `verify.sh` reads
the build flags back out of a built binary and fails on a mismatch.

**A third of `ops/opticalFlow.hpp` is invisible to every x86 build.** The NEON region is
behind `#if BINCV_HAVE_NEON && __aarch64__`, so an edit there can be structurally broken
and still pass all four x86 configurations. `verify_arm.sh` covers it under emulation.

**Warnings are project policy, not the script's.** They live in
`bincv-cpp/cmake/BincvWarnings.cmake` and are on in every build:
`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion`. `-Werror` is off by
default so a mid-edit build still finishes; the gate turns it on. Warnings apply to
first-party targets only — never to `bincv_core`'s interface, because a consumer's warning
policy is theirs.

`bincv_assert_warning_policy()` runs at configure time and **fails the configuration** if
a first-party target does not link `bincv_warnings`. That is a structural check rather
than a log scan, because a target compiled with no warning flags emits nothing for a log
scan to find.

**`-Wconversion` is the load-bearing one:** the library is templated on the word type, so
every mask and shift is compiled at 8, 16, 32 and 64 bits, and an expression that is exact
at `uint64_t` can truncate at `uint8_t`. Deliberate narrowing needs a `static_cast`, which
is where a reader is told the truncation is intended.

## Hard rules

Settled decisions. If one seems wrong, say so rather than working around it.

- **Kernels take views, never owning containers.** A kernel compiles once per
  `(WordType, N)` and must not care about its arguments' alignment or ownership.
- **Never expose a per-word popcount.** Reductions are bulk only — region, masked, or
  windowed. On aarch64 a per-word popcount pays two register-domain crossings per 64
  pixels. Internal helpers stay internal.
- **No heap allocation inside kernels.** Scratch buffers are caller-provided.
- **Value semantics** — copy means deep copy. No reference counting. Sharing is a view.
- **Padding bits stay zero.** Any operation that writes whole words past `width` must
  clear them, or word-wise reductions over-count.
- **Tier 1 operations must be bit-exact against OpenCV**, proven by a test. State the API
  tier in every public docstring.
- **A feature gate comes from the compiler's own macros wherever the compiler can know
  it.** Build-system defines are for what it genuinely cannot, and those get reported by
  `simdStatus()` rather than assumed.

## Scope

**binCV provides memory- and performance-optimized versions of operations a vision
pipeline already runs.** It takes no position on which algorithm a caller should use —
that is their choice, and binCV's job is to make the one they chose cheaper.

**The operation set follows the use cases that turn up, not a fixed taxonomy.** binCV is
not trying to replace OpenCV. An operation is in scope when it is on a path a caller
needs *and* binCV can make it smaller or faster; it is out of scope when binCV would add
nothing but a second implementation to keep correct.

Today that means image processing, features and tracking, and the geometry the frontend
consumes — RANSAC-based estimation is in scope for that reason. IMU fusion and bundle
adjustment are not, because no use case has asked for them yet. That is a statement about
what has come up, not a boundary on principle.

**The input boundary is a rule, not a list:** binCV accepts a **single-channel,
integer-typed, strided pixel array** and turns it into an N-bit matrix. Getting to that
array is the caller's — decoding, demosaicing and color conversion each turn one wide
image into another and leave the caller exactly as far from bits as before. Everything
from such an array down to bits is binCV's, **including sources wider than 8 bits**,
because downconverting first destroys small gradients before the threshold can see them.

GPU backends are a **TODO**, not out of scope. A CUDA prototype lives in `bincv-cuda/`.

## Style

- OpenCV conventions: `camelCase` functions, `PascalCase` types, `UPPER_CASE` constants,
  lowercase namespaces, destination as out-parameter.
- Tier 3 operations (no OpenCV equivalent) must **not** borrow OpenCV names.
- Match the comment density and idiom of surrounding code.
- **Comments explain the code, not the project's history.** No task numbers, no experiment
  identifiers, no "scheduled/deferred" notes — a reader of a header has no way to resolve
  them and does not need to. If a measurement explains why the code is shaped this way,
  give the number and the reason, not a citation.
- Commit messages: `[area] Summary`, then what changed and why.

## Stop and ask

- A spec is ambiguous or contradicts the design notes.
- A decision is needed that no measurement settles.
- Something in scope turns out to be impossible as specified.
- **A measurement contradicts a documented claim** — this is valuable; report it rather
  than adjusting the code to fit the doc.
