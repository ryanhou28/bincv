# binCV documentation

| | for | what it is |
|---|---|---|
| [../README.md](../README.md) | anyone | what binCV is, the numbers, how to build |
| [../GETTING_STARTED.md](../GETTING_STARTED.md) | a user | build it, use it, conventions |
| [API.md](API.md) | a user | **the API reference** — every public entry point, its brief and its tier |
| [ARCHITECTURE.md](ARCHITECTURE.md) | a contributor | how the library is put together, and why |
| [reports/](reports/README.md) | anyone weighing it up | **what it costs and what it saves** — measured against OpenCV on x86-64 and aarch64, wins and losses both |

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
