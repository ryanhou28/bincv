# binCV documentation

Five places, and which one you want depends on what you are doing.

| | for | what it is |
|---|---|---|
| [../README.md](../README.md) | anyone | what binCV is, a handful of headline comparisons, how to build |
| [../GETTING_STARTED.md](../GETTING_STARTED.md) | a user | build it, use it, conventions |
| [API.md](API.md) | a user | **the API reference** — every public entry point, its brief and its tier |
| [ARCHITECTURE.md](ARCHITECTURE.md) | a contributor | how the library is put together, and why |
| [reports/](reports/README.md) | anyone weighing it up | **what it costs and what it saves** — measured against OpenCV on x86-64, aarch64 and a GPU, wins and losses in the same tables |

**Reading it in order.** If you are deciding whether to adopt binCV, start at
[reports/README.md](reports/README.md) — its *At a glance* gives both sides' measured
figures for speed and for memory, in separate tables, with the report each pair was cut
from. If you are about to write a call, [API.md](API.md) names the entry point and
[GETTING_STARTED.md](../GETTING_STARTED.md) shows the shape of a program that uses it. If
you are about to change the library, [ARCHITECTURE.md](ARCHITECTURE.md) is the one that
says why the thing you are about to change is the way it is.

**Three pages inside the reports are prerequisites rather than results**, and they are
worth knowing about before quoting anything.
[reports/methodology-memory.md](reports/methodology-memory.md) names the four different
quantities that get called "memory" and the four measurement errors this project published.
[reports/methodology-timing.md](reports/methodology-timing.md) is its twin on the other
axis: how a difference between two timings is judged real rather than assumed. And
[reports/README.md](reports/README.md)'s *How the numbers are taken* states the denominator
rule, the one-thread rule and the correctness check that runs before any timing.

**The GPU backend is documented in three places and they do not repeat each other.**
[ARCHITECTURE.md](ARCHITECTURE.md) §8.5 is the design — why the representation is shared
and the kernels are forked; [../backends/cuda/README.md](../backends/cuda/README.md) is how
to build and use it; [reports/cuda.md](reports/cuda.md) is what it measures, against
`cv::cuda`.

## The API reference

[API.md](API.md) is **generated from the headers**:

```bash
python3 scripts/gen_api_index.py     # writes docs/API.md
```

It is committed so a reader in a browser has one without running anything. Every line is
the `@brief` from the declaration itself, so it cannot drift from the code without the
code changing — regenerate it in the same commit as any signature change.

For full signatures, parameters and the rationale paragraphs — often the useful part —
read the header, or generate the HTML:

```bash
doxygen docs/Doxyfile     # writes docs/api/html; needs doxygen installed
```

## API tiers

Every public entry point states one:

- **Tier 1** — bit-exact with the OpenCV function it names, proven by a test.
- **Tier 2** — OpenCV's role and call shape, different numerics.
- **Tier 3** — no OpenCV equivalent. **These deliberately do not borrow OpenCV names**,
  because a familiar name on different semantics is worse than an unfamiliar one.
