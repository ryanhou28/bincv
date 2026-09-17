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
one-off probes — live in `.local/` and `experiments/`, and are **not** part of the
repository.

The tree is one axis: the library, the things that exercise it, the places it runs.

```
include/bincv/   the library — header-only, zero dependencies
src/             the handful of non-header sources
tests/ benchmark/ examples/   consumers of the library
targets/         bare-metal harnesses that RUN it on a device
backends/        alternative compute backends (cuda/)
cmake/ scripts/ docs/
```

**A backend shares the representation and forks the kernels.** The format, the views
and the invariants have one definition, because the copy that drifts is silently wrong
in a way that looks like a correct answer. Kernels are not shared: a device traversal
has nothing in common with a row loop, and pretending otherwise costs the performance
the backend exists for. A backend is **never a drop-in dispatch target** — device types
stay device-typed, so no call can hide where its memory lives.

`backends/cuda/` is that backend: `bincv::cuda::` device-typed views over the host's
byte layout, `uint32_t` device words (a CUDA core is a 32-bit machine, and
`__ballot_sync` packs the format's own word in one instruction), kernels forked and
proven bit-exact against the host by `scripts/verify_cuda.sh`. See
[docs/ARCHITECTURE.md §8.5](docs/ARCHITECTURE.md) and [docs/reports/cuda.md](docs/reports/cuda.md).

## How performance and footprint decisions get made

**Measure the alternatives, weigh the result, then decide — and record all three.**
Argument alone does not settle a performance question; neither does a benchmark without a
stated decision rule.

- **Write the decision rule before measuring.** What result favors which choice, written
  down first. Deciding afterwards invites fitting the conclusion to the numbers.
- **The rule names its metrics and the magnitude required on each, and it is written
  per case.** There is no project-wide "X% is worth it" threshold, and inventing one
  is not the same as having one: a bar that came from nowhere makes an arbitrary
  judgement look derived, and the write-it-first rule then launders it. Say which
  metrics decide this case — speed, peak memory, code size, portability, how much
  hand-written code has to stay bit-exact forever — and how much of each is needed.
  **If the threshold is a judgement nobody has made yet, that is a "stop and ask",
  not a number to fill in.**
- **State what the measurement covers.** A microbenchmark result is not an end-to-end
  result. A kernel that is 12% faster in a loop that does nothing else moves a
  pipeline by 12% times its share of that pipeline, and quoting the first number
  where the second decides is how a real gain gets adopted for nothing — or a real
  one dismissed.
- **Compare alternatives**, not one option, on representative workloads.
- **Report memory and speed together** — they trade off, so one alone cannot be weighed
  against goals that conflict.
- **Commit the benchmark.** Every performance claim must be reproducible.
- **Pick the right baseline.** The bar for a new implementation is the best existing
  option, not the worst. Measuring against a fallback nobody would use makes anything
  look like a win.

**An operation SHIPS only when it holds up on both axes (owner's rule,
2026-09-15).** Correct-but-far-behind is a stage, not a product: a kernel that
loses its role comparison badly against the best existing option does not merge
on the strength of a stated price -- it gets optimized first, or the owner
explicitly accepts the gap with the memory-side argument stated. The premise of
this library is fast AND lightweight out of the box; half of that is not a
smaller claim, it is a different product.

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
./scripts/verify.sh           # ~35 s, four configurations, warnings fatal
./scripts/verify_cross.sh     # the other architecture under emulation; skips without Docker
./scripts/verify_cortex_m.sh  # Cortex-M7 compile gate; skips without arm-none-eabi
python3 scripts/check_links.py
```

`verify.sh` builds and tests four configurations — Release+OpenCV, Release core-only,
`-fno-exceptions` core-only, and **Debug** core-only — with `-DBINCV_WERROR=ON`, and exits
non-zero if anything fails. It starts with a **gate self-check**: two throwaway builds
that are *supposed* to fail. A gate nobody has watched fail is not known to work.

Read the two numbers in its summary table:

- **CTEST** — cases run.
- **CHECKS** — assertions executed. A drop is a regression even when every case still
  passes, so per-suite floors live in `tests/expected-checks.txt` and a count
  below one of them fails the run. Raising a floor is a reviewed edit
  (`./scripts/verify.sh --update-checks-baseline`, then commit the diff).

Each configuration also has to *be* the configuration it claims to be: `verify.sh` reads
the build flags back out of a built binary and fails on a mismatch.

**`verify.sh` covers the architecture it runs on, and nothing else.** A third of
`ops/opticalFlow.hpp` is invisible to every x86 build — the NEON region is behind
`#if BINCV_HAVE_NEON && __aarch64__` — and the AVX2 paths in `ops/pack.hpp` are invisible
to every aarch64 build, so an edit in either region can be structurally broken and still
pass every native configuration on the other side. `verify_cross.sh` detects the host
with `uname -m` and emulates the architecture the native run cannot see: aarch64 from an
x86_64 host, x86_64 from an aarch64 one. (`verify_arm.sh` still works; it forwards
there.) It refuses to compare check counts against a reference from its own
architecture — that diff cannot fail, so it reports `NOT PERFORMED` and exits 77.

**`verify_cortex_m.sh` is the only place `size_t` is 32 bits.** Every index, stride and
`planeWords()` is a `size_t`, and the four-word-type sweep is otherwise compiled solely at
64-bit pointer width — the first run of that gate found a test asserting `size_t{1} << 63`.
It is **compile-only** (the host cannot execute an M-profile image) and, like
`verify_cross.sh`, exits **77** when it cannot run at all, which is not a pass.

**Warnings are project policy, not the script's.** They live in
`cmake/BincvWarnings.cmake` and are on in every build:
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

**The operation set follows what users need, not a fixed taxonomy.** binCV is not
trying to replace OpenCV. An operation is in scope when it sits on a path **users**
run *and* binCV can make it smaller or faster. A library's users include people
outside this repository, so "no in-repo caller yet" is not a veto — that reading was
corrected by the owner (2026-09-11); an in-repo caller is what *prices* an operation
honestly (the benchmark-at-birth rule below), not a gate on whether it may exist. An
operation is out of scope when binCV would add nothing but a second implementation to
keep correct.

Today that means image processing, features and tracking, stereo, and the geometry the
frontend consumes — RANSAC-based estimation is in scope for that reason. IMU fusion and
bundle adjustment are out on the second prong, not the first: they are float linear
algebra with no low-bit-width representation to exploit, so binCV would make them
neither smaller nor faster.

**The input boundary is a rule, not a list:** binCV accepts a **single-channel,
integer-typed, strided pixel array** and turns it into an N-bit matrix. Getting to that
array is the caller's — decoding, demosaicing and color conversion each turn one wide
image into another and leave the caller exactly as far from bits as before. Everything
from such an array down to bits is binCV's, **including sources wider than 8 bits**,
because downconverting first destroys small gradients before the threshold can see them.

The GPU backend lives in `backends/cuda/`: it shares the representation and forks the
kernels (the decision above), providing logic, reductions, the sensor stage, census and
dense disparity on the device, each bit-exact against the host. On the reference GPU the
binary dense path beats `cv::cuda::StereoBM` on both speed and device memory
([docs/reports/cuda.md](docs/reports/cuda.md)).

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
