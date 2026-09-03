# How memory is measured

Every memory figure in these reports names the quantity it measures and the instrument
that read it. This page says what those quantities are, which instruments can see them,
and which instruments were tried and rejected.

It exists because this project got the memory comparison wrong four times in a row, each
time in a way that produced a confident number. The errors are listed below with their
sizes, because they are the reason for the rules.

## Four different quantities, routinely called "memory"

A comparison is only meaningful when both sides report the same one.

| quantity | what it means | what it is for |
|---|---|---|
| **cumulative allocation** | every byte ever handed out during the call | allocator traffic; **never** a footprint |
| **peak live heap** | high-water mark of simultaneously-live heap bytes | what must exist on the heap at once |
| **stack reservation** | how far the stack pointer descends | how big a thread's stack must be |
| **peak RSS** | resident pages, process-wide | what the OS charges the process |

Cumulative allocation is the dangerous one. A loop that takes and returns a 1 KiB buffer
a thousand times has a cumulative figure of 1 MiB and a peak live figure of 1 KiB. OpenCV
allocates and releases inside its RANSAC loop, so for these calls the two differ by more
than an order of magnitude.

**Peak live heap and stack reservation add; the others do not.** A working-set comparison
is the sum of those two on each side, and both sides must be measured, not one.

## The hazard this library is shaped for

binCV's hard rule is that **no kernel allocates**: scratch is caller-provided and working
arrays live on the stack. That makes every heap-only instrument flatter binCV
automatically, by an amount that has nothing to do with how much memory it uses.

Two implementations of identical work, each using 256 KiB of scratch, differing *only* in
storage class, measured here:

| implementation | heap-only instrument | stack instrument |
|---|---|---|
| scratch from `malloc` | 262,144 B | 7,784 B |
| scratch on the stack | **4,096 B** | **262,456 B** |

A heap-only instrument reports the stack version as **64× leaner while it uses exactly the
same memory**. This is not a subtle bias; it is the entire result.

Every comparison in these reports therefore measures **both** heap and stack on **both**
sides. A binCV memory claim backed only by allocation counts is not a memory claim, and
this table is why. It is also the direct explanation of error 2 below: dividing binCV's
stack by OpenCV's heap is this artifact with the sign flipped.

Two things follow that are worth saying plainly rather than leaving implied. Moving
memory from the heap to the stack **is** a real benefit — no allocator call, no
fragmentation, a footprint knowable before the call — and binCV's design takes it
deliberately. But it is not a reduction in bytes, and it carries a cost the byte count
does not show: a failed heap allocation returns a null pointer a caller can check, while
an exhausted stack terminates the process. That trade is the reason the stack figures here
are published as a budget a caller is expected to size against, rather than as a number
that has simply gone away.

## The errors, and what each cost

Each of these produced a number that looked reasonable and was published or nearly
published.

**1. Cumulative allocation reported as footprint.** OpenCV's five-point call was reported
at 323,088 B against binCV's 15,080 B. The 323,088 was the sum of every allocation; the
peak live figure was a small fraction of it. Wrong in binCV's favour by more than 10×.

**2. binCV's stack compared against OpenCV's heap.** Having fixed (1), the comparison
divided binCV's stack frame by OpenCV's peak live heap and concluded binCV used 2.1×
*more* memory. Those are different quantities and the ratio meant nothing. Measuring
both sides' stack reversed the conclusion.

**3. A published stack budget 27% low.** `essentialSolverStackBytes()` returned 4,536 B
while the compiled frame was 6,240 B. A caller sizing a thread from it would have been
short.

**4. A replaced `operator new` cannot see OpenCV.** `cv::Mat` allocates through
`cv::fastMalloc`, which calls `malloc` directly, so a replaced `operator new` never
observes the matrix data — the largest blocks in the call. Measured both ways in one
process, `cv::findEssentialMat` on 1,000 correspondences reports **2,744 B through
`operator new` and 46,968 B through `malloc`**: an under-count of 17×.

The tell was visible in the published table and went unread: the `operator new` figure was
a flat 2,744 B at 200, 500 and 1,000 correspondences. A solver's working set does not stay
constant as its input quadruples. Measured at the allocator it grows 16,568 → 28,024 →
46,968 → 84,952 B across 200 → 2,000 points, about 38 B per correspondence.

Errors 1 and 2 ran in opposite directions; 4 ran against binCV. Being wrong in your own
favour is the one that gets published, but the discipline is the same either way.

## The instruments

### Heap — allocator interposition

[`benchmark/heap_probe.cpp`](../../bincv-cpp/benchmark/heap_probe.cpp) defines `malloc`,
`free`, `calloc`, `realloc`, `memalign`, `aligned_alloc` and `posix_memalign`. The
executable's definitions preempt libc's for every dynamically linked caller, OpenCV
included, which is what makes it see `fastMalloc`. Block sizes come from
`malloc_usable_size`, so the figure includes the rounding the allocator actually
committed — a caller pays that whether or not the program asked for it.

It reports **peak live** and **allocator traffic** as separate rows, never combined.

**It proves itself before it reports.** `heapprobe::selfCheck()` runs four checks with
known answers: that a `malloc` of a known size is seen and returned on free, that
`operator new` is seen, that ten sequential take-and-release cycles read as one block and
not ten, and that balanced work nets to zero. A benchmark whose self-check fails prints no
figures and exits non-zero. The same reasoning as `verify.sh`'s gate self-check: an
instrument that has never returned a known answer is not evidence.

Cross-checked against `valgrind --tool=dhat`, which instruments every allocation and
reported 42,344 B where the probe read 46,968 B for the same call — the difference being
`malloc_usable_size` rounding, which DHAT does not count.

### Stack — guard-page bisection

[`benchmark/essential_stack_benchmark.cpp`](../../bincv-cpp/benchmark/essential_stack_benchmark.cpp)
runs the call on a thread with a bounded stack inside a forked child and bisects for the
smallest stack it survives, at 16 B granularity.

**The quantity is reservation, not bytes written.** A frame that is allocated and only
partly written must still be reserved. Stack painting and watermarking — the usual
embedded techniques — scan for bytes that changed and so report the smaller quantity;
they would understate this one.

The probe recovers two known answers first, workloads consuming 4,096 B and 16,384 B. It
reads them as 4,112 B and 16,400 B — one bisection quantum high, which is the expected
bias. If it misses them the run fails and prints nothing.

`PTHREAD_STACK_MIN` is 16 KiB and binCV's whole call fits under it, so bisecting the stack
*size* cannot resolve binCV at all. The probe fixes a generous stack and bisects the
padding consumed before the call instead: the largest padding a workload tolerates is
headroom it did not need, and the difference between two workloads' tolerated padding is
the difference in what they used. Every figure is net of an empty-workload baseline.

### Rejected for stack: Valgrind Massif `--stacks=yes`

Tried, and it fails the calibration above. It reported binCV's whole call using **less**
stack than an empty workload — 8,216 B against 10,088 B — which is impossible, and it
still did so with snapshot counts between 766 and 980.

The mechanism is worth stating precisely, because it is not that Massif's tracking is
imprecise. Massif hooks Valgrind's `new_mem_stack` / `die_mem_stack`, which fire on every
stack-pointer change, so its *tracking* is exact. What is periodic is the **recording**:
stack size is written into the output only at snapshots, and Massif's peak-snapshot logic
is driven by heap size, not stack size. A stack peak that rises and falls inside one call
is tracked exactly and then never written down. That is why it works for a large,
long-lived frame — a known 1 MiB frame reads within ~300 B — and fails for a transient
few-kilobyte one, which is exactly our case.

Two further Massif caveats, both documented, that matter for anyone repeating this:

- **The peak snapshot is only ever taken after a deallocation.** A program that allocates
  and exits without freeing has *no* peak recorded, and `--peak-inaccuracy=0.0` does not
  change that. By default Massif also only records a new peak when it exceeds the previous
  one by 1%.
- **`--max-stackframe` defaults to 2,000,000 B, and a larger stack-pointer move is treated
  as a stack switch and dropped** — not merely handled differently. The matching release is
  dropped too, so the accounting stays permanently inflated for the rest of the run. Any
  measurement involving a caller-provided scratch buffer larger than 2 MB on the stack
  needs this raised.

Massif's *heap* figures are sound. Its stack figures are not usable at this granularity,
and the calibration is what caught it.

### Buffer arithmetic, for pipelines

The image-pipeline reports compute peak working set from buffer geometry — the live
buffers of one stage, counted in bytes — rather than sampling. That is exact and
reproducible for binCV, whose kernels allocate nothing, so every buffer appears in a
caller's signature. **Where the OpenCV side allocates internally, buffer arithmetic cannot
see it and the allocator probe is required.**

## Rules

- **Name the quantity.** "Memory" alone is not a claim.
- **Same instrument on both sides**, in the same process, around one call.
- **Peak live and cumulative traffic are two rows**, never one number.
- **Measure the stack whenever the operation keeps working arrays on it.** For dense
  solvers that is where the memory is, and a heap-only comparison misses it entirely.
- **Calibrate the instrument in the benchmark**, so a broken probe fails loudly rather
  than reporting a plausible number.
- **A figure that does not move with the input size is suspect.** That is what would have
  caught error 4 on the day it was introduced.

## What these figures are not

**A measured stack figure is a lower bound on worst-case stack, not the worst case.** It
is the deepest the stack went on the paths that ran. The probe runs each workload over six
different scenes and reports the deepest, which widens coverage without changing what is
being claimed. This is not a limitation of this particular probe; it is the standing
result for measurement-based stack analysis, stated in the safety-critical literature as
bluntly as it can be put — repeated measurement with varied inputs *cannot* guarantee the
maximum is ever observed. Callers budgeting a thread from these numbers should keep
margin.

**A guard page catches a stack pointer that walks into it, not one that steps over it.**
A large frame that is reserved and left mostly unwritten is the case where this technique
can under-read, so the benchmark measures it: a 65,536 B frame with only its shallow end
written reads as 65,536 B here, with `-fstack-clash-protection` off. The limitation did
not bite at these frame sizes, and the diagnostic row is there so a future change that
makes it bite is visible rather than silent.

**The comparison is symmetric, which is what the numbers rest on.** Both sides are
measured by the same instrument in the same process around one call, and both sides'
figures are of the same quantity. That matters more than any individual instrument's
precision: a shared limitation moves both columns together, while an asymmetric one — our
stack against their heap, or a compile-time bound for us against a runtime measurement for
them — produces a ratio that means nothing. binCV has published that mistake twice.

**Memory figures are deterministic; timings on the development host are not.** The heap
and stack figures above are byte-identical across repeated runs. The x86-64 host is a WSL2
VM whose timing spread on these calls reaches 130%, so its *ratios* are indicative only;
the reference device, pinned and with its governor fixed, holds 0.1–0.6% and is where
timing claims come from.

## If a worst-case stack bound is ever needed

These reports give measured figures, not bounds. Should a caller need a real bound for
binCV's own code, the route is `-fstack-usage` summed along the call graph, and it comes
with conditions that have to be checked rather than assumed:

- Only `static` and `dynamic bounded` frames are a reliable maximum. A bare `dynamic`
  entry's number is explicitly **not** an upper bound — it is only the bounded part.
- The sum is unsound with recursion, indirect calls, or any function with no `.su` data.
  GCC 11's `-fcallgraph-info=su` emits the call graph decorated with frame sizes and marks
  both recursion edges and indirect-call placeholders, so a tool can refuse to answer
  instead of silently under-reporting.
- Frame sizes are a property of generated code, not of source: inlining, `-O` level and
  `-fstack-protector` all move them substantially. The `.su` data must come from the build
  being shipped.
- `-Wstack-usage=N` is documented as conservative and counts `alloca` and VLAs;
  `-Wframe-larger-than=N` is documented as approximate, *not* conservative, and excludes
  them. Only the first is usable for a footprint claim.

None of this is available for a prebuilt `libopencv`, which has no `.su` data — which is
the reason both sides here are measured with a binary-level instrument instead.

## Sources

- GCC, [Developer Options](https://gcc.gnu.org/onlinedocs/gcc/Developer-Options.html) —
  `-fstack-usage`, the `static` / `dynamic` / `bounded` definitions, `-fcallgraph-info`.
- GCC, [Warning Options](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html) —
  `-Wstack-usage` ("conservative") against `-Wframe-larger-than` ("approximate and not
  conservative").
- Valgrind, [Massif manual](https://valgrind.org/docs/manual/ms-manual.html) — peak
  recorded only after a deallocation, `--peak-inaccuracy`, `--stacks`, `--heap-admin`,
  `--pages-as-heap`.
- Valgrind, [DHAT manual](https://valgrind.org/docs/manual/dh-manual.html) — `Total`
  against `At t-gmax` against `At t-end`, the canonical cumulative / peak-live / leaked
  distinction. DHAT's own example has cumulative at 6.2× peak live.
- Valgrind, [core manual](https://valgrind.org/docs/manual/manual-core.html) —
  `--max-stackframe`.
- AdaCore, [The GNATstack Tool](https://docs.adacore.com/live/wave/gnatstack/html/gnatstack_ug/The_GNATstack_Tool.html)
  — the canonical list of conditions a static stack bound needs: no indirect calls, no
  variable-sized locals, no recursion, and access to the complete environment.
- Kästner and Ferdinand, [Efficient Verification of Non-Functional Safety Properties](https://www.absint.com/Kaestner_ISSC2011.pdf)
  (ISSC 2011) — measuring maximum stack usage gives a result for one run with fixed input,
  and repeated measurement cannot guarantee the maximum is ever observed.
- Rapita Systems, [Function pointers and their impact on stack analysis](https://www.rapitasystems.com/blog/function-pointers-and-their-impact-stack-analysis)
  — the failure mode of an incomplete call graph is *optimistic*: an unresolved indirect
  call is assumed to use zero stack.
- Memfault Interrupt, [Measuring Stack Usage the Hard Way](https://interrupt.memfault.com/blog/measuring-stack-usage)
  — painting under-counts a frame that is allocated but only partly written.
- Hertz and Berger, [Quantifying the Performance of Garbage Collection vs. Explicit Memory Management](https://cse.buffalo.edu/~mhertz/gcmalloc-oopsla-2005.pdf)
  (OOPSLA 2005) — the precedent for refusing a reported-allocation metric and measuring
  the real footprint instead, and for reporting time and space jointly.
- Heiser, [Systems Benchmarking Crimes](https://gernot-heiser.org/benchmarking-crimes.html)
  — on benchmarking a competitor no less carefully than your own system.
- SIGPLAN, [Empirical Evaluation Checklist](https://www.sigplan.org/Resources/EmpiricalEvaluation/)
  — "indirect or inappropriate proxy metric": a proxy substitutes for a direct measure
  only with explicit justification. Heap bytes as a proxy for footprint is exactly that.
- SEI CERT, [MEM05-C](https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/recommendations/memory-management-mem/mem05-c)
  — an exhausted stack can terminate the program, where a failed heap allocation returns a
  value the caller can check.
- Qualys, [The Stack Clash](https://www.qualys.com/2017/06/19/stack-clash/stack-clash.txt)
  — a stack pointer that jumps over the guard page raises no fault, which is the bound on
  the guard-page technique used here.
- Linux kernel, [sysctl/kernel](https://www.kernel.org/doc/Documentation/admin-guide/sysctl/kernel.rst)
  — `randomize_va_space`. Stack addresses and alignment vary with ASLR and with the size
  of the environment; pin both when a stack measurement has to be compared across runs.

## Reproduce

```bash
cmake -S bincv-cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/benchmark/essential_benchmark        # heap, both sides, with its self-check
./build/benchmark/essential_stack_benchmark  # stack, with its calibration rows
```

Both refuse to print figures if their self-check fails.
