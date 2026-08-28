# The binCV document set

Six documents, and they are for different people. If you are reading in order, this
is the order.

| | for | what it is |
|---|---|---|
| [../README.md](../README.md) | **anyone** | what binCV is, the numbers, how to build |
| [../GETTING_STARTED.md](../GETTING_STARTED.md) | **a user** | build, test, benchmark, and a tour of the operation set |
| [API.md](API.md) | **a user** | **the API reference** — every public entry point, its brief and its tier, generated from the headers |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | **a contributor** | the design, the input contract, and every design decision with its evidence |
| [../TASKS.md](../TASKS.md) | **a contributor** | the backlog, by phase |
| [../EXPERIMENTS.md](../EXPERIMENTS.md) | **a sceptic** | the measurement log |

## The three record types, and why they are worth the trouble

Performance work in this repository leaves three kinds of trace, cross-linked:

- **D-records** — *decisions*, in [ARCHITECTURE §8](../ARCHITECTURE.md#8-design-decisions).
  What was chosen, why, and what evidence settled it. These are architecture decision
  records in everything but filename.
- **E-records** — *open questions*, in
  [ARCHITECTURE §9](../ARCHITECTURE.md#9-open-questions-and-planned-experiments). Each
  is a proposal with a decision rule attached, not a wish.
- **X-records** — *measurements*, in [EXPERIMENTS.md](../EXPERIMENTS.md). Each names its
  platform, its workload, and **the decision rule that was fixed before the numbers
  were taken**.

**The rule that makes the rest trustworthy:** a performance claim needs a committed
benchmark, a pre-registered decision rule, and both speed and memory reported.

It is not ceremony. That loop has caught several ceilings that overstated, an
optimisation measuring 1.75× in the kernel and 3.3× *slower* on the real workload, a
vector kernel that a mis-attached `#define` had compiled out of three consecutive
"improvements", two kernels that were correct and never timed until something called
them and they turned out to be 78% of the frontend, and headline figures that were
measuring something other than what they claimed — one of them for an entire working
session, and one that stood in the README.

## The API reference

[API.md](API.md) is **generated from the headers** by

```bash
python3 scripts/gen_api_index.py     # writes docs/API.md
```

and is committed, so a reader in a browser has one without running anything. It is a
view rather than a document: every line is the `@brief` from the declaration itself, so
it cannot drift from the code without the code changing. Regenerate it in the same
commit as any signature change.

For full signatures, parameters and the rationale paragraphs — which are often the
useful part — read the header, or generate the HTML:

```bash
doxygen docs/Doxyfile     # writes docs/api/html; needs doxygen installed
```

Every public entry point states its **API tier**:

- **Tier 1** — bit-exact with the OpenCV function it names, proven by a test.
- **Tier 2** — OpenCV's role and call shape, different numerics.
- **Tier 3** — no OpenCV equivalent. **These deliberately do not borrow OpenCV names**,
  because a familiar name on different semantics is worse than an unfamiliar one.

## A note on EXPERIMENTS.md's size

It is long because it is append-only and nothing is deleted when it turns out to be
wrong — a withdrawn conclusion stays, struck through, next to what replaced it. That is
the point of it: the corrections are the most useful part, and a log that quietly
removed its mistakes would be worth much less than one that keeps them.
