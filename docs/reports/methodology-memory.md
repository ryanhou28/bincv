# How memory is measured

Every memory figure in these reports names the quantity it measures and the instrument that
read it. This page says what those quantities are, which instruments can see them, and which
were tried and rejected. It exists because this project got the memory comparison wrong four
times in a row, each time in a way that produced a confident number; the errors are listed
below with their sizes, because they are the reason for the rules.

## Four different quantities, routinely called "memory"

A comparison is only meaningful when both sides report the same one.

| quantity | what it means | what it is for |
|---|---|---|
| **cumulative allocation** | every byte ever handed out during the call | allocator traffic; **never** a footprint |
| **peak live heap** | high-water mark of simultaneously-live heap bytes | what must exist on the heap at once |
| **stack reservation** | how far the stack pointer descends | how big a thread's stack must be |
| **peak RSS** | resident pages, process-wide | what the OS charges the process |

Cumulative allocation is the dangerous one. A loop that takes and returns a 1 KiB buffer a
thousand times has a cumulative figure of 1 MiB and a peak live figure of 1 KiB. OpenCV
allocates and releases inside its RANSAC loop, so for these calls the two differ by more than
an order of magnitude.

**Peak live heap and stack reservation add; the others do not.** A working-set comparison is
the sum of those two on each side, and both sides must be measured.

## The hazard this library is shaped for

binCV's hard rule is that **no kernel allocates**: scratch is caller-provided and working
arrays live on the stack. That makes every heap-only instrument flatter binCV automatically,
by an amount that has nothing to do with how much memory it uses.

Two implementations of identical work, each using 256 KiB of scratch, differing *only* in
storage class:

| instrument | scratch from `malloc`, bytes | scratch on the stack, bytes | `malloc` ÷ stack |
|---|---|---|---|
| heap-only | 262,144 | 4,096 | **64×** |
| stack | 7,784 | 262,456 | not published |

A heap-only instrument reports the stack version as **64× leaner while it uses exactly the
same memory**. This is not a subtle bias; it is the entire result, and it is the direct
explanation of error 2 below — dividing binCV's stack by OpenCV's heap is this artifact with
the sign flipped. Every comparison in these reports therefore measures **both** heap and stack
on **both** sides.

Moving memory from the heap to the stack **is** a real benefit — no allocator call, no
fragmentation, a footprint knowable before the call. But it is not a reduction in bytes, and a
failed heap allocation returns a null pointer a caller can check where an exhausted stack
terminates the process. That is why the stack figures here are published as a budget a caller
is expected to size against.

## The errors, and what each cost

**1. Cumulative allocation reported as footprint.** OpenCV's five-point call was reported at
323,088 B against binCV's 15,080 B. The 323,088 was the sum of every allocation; the peak live
figure was a small fraction of it. Wrong in binCV's favour by more than 10×.

**2. binCV's stack compared against OpenCV's heap.** Having fixed (1), the comparison divided
binCV's stack frame by OpenCV's peak live heap and concluded binCV used 2.1× *more* memory.
Those are different quantities and the ratio meant nothing. Measuring both sides' stack
reversed the conclusion.

**3. A published stack budget 27% low.** `essentialSolverStackBytes()` returned 4,536 B while
the compiled frame was 6,240 B. A caller sizing a thread from it would have been short.

**4. A replaced `operator new` cannot see OpenCV.** `cv::Mat` allocates through
`cv::fastMalloc`, which calls `malloc` directly, so a replaced `operator new` never observes
the matrix data — the largest blocks in the call. Measured both ways in one process on
`cv::findEssentialMat` at 1,000 correspondences:

| instrument | OpenCV peak live heap, bytes | interposed ÷ `operator new` |
|---|---|---|
| replaced `operator new` | 2,744 | — |
| interposed `malloc` | 46,968 | **17×** |

An under-count of 17×, entirely on OpenCV's side of the comparison. The tell was in the
published table and went unread: the `operator new` figure was a flat 2,744 B at 200, 500 and
1,000 correspondences, and a solver's working set does not stay constant as its input
quadruples. Measured at the allocator it moves with the input, at about 38 B per
correspondence:

| correspondences | OpenCV peak live heap, bytes |
|---|---|
| 200 | 16,568 |
| 500 | 28,024 |
| 1,000 | 46,968 |
| 2,000 | 84,952 |

Errors 1 and 2 ran in opposite directions; 4 ran against binCV. Being wrong in your own
favour is the one that gets published, but the discipline is the same either way.

## The instruments

### Heap — allocator interposition

[`benchmark/heap_probe.cpp`](../../benchmark/heap_probe.cpp) defines `malloc`, `free`,
`calloc`, `realloc`, `memalign`, `aligned_alloc` and `posix_memalign`. The executable's
definitions preempt libc's for every dynamically linked caller, OpenCV included, which is what
makes it see `fastMalloc`. Block sizes come from `malloc_usable_size`, so the figure includes
the rounding the allocator committed. It reports **peak live** and **allocator traffic** as
separate rows, never combined.

**It proves itself before it reports.** `heapprobe::selfCheck()` runs four checks with known
answers: a `malloc` of a known size seen and returned on free, `operator new` seen, ten
sequential take-and-release cycles reading as one block and not ten, and balanced work netting
to zero. A benchmark whose self-check fails prints no figures and exits non-zero.

Cross-checked against `valgrind --tool=dhat`, which reported 42,344 B where the probe read
46,968 B for the same call — the difference being `malloc_usable_size` rounding, which DHAT
does not count.

### Stack — guard-page bisection

[`benchmark/essential_stack_benchmark.cpp`](../../benchmark/essential_stack_benchmark.cpp)
runs the call on a thread with a bounded stack inside a forked child and bisects for the
smallest stack it survives, at 16 B granularity.

**The quantity is reservation, not bytes written.** Stack painting and watermarking — the
usual embedded techniques — scan for bytes that changed, so they report the smaller quantity.

The probe recovers two known answers first:

| workload | stack it actually uses, bytes | what the probe reads, bytes |
|---|---|---|
| known small frame | 4,096 | 4,112 |
| known large frame | 16,384 | 16,400 |

One bisection quantum high in both cases, which is the expected bias. If the probe misses
them the run fails and prints nothing.

`PTHREAD_STACK_MIN` is 16 KiB and binCV's whole call fits under it, so bisecting the stack
*size* cannot resolve binCV at all. The probe fixes a generous stack and bisects the padding
consumed before the call instead: the largest padding a workload tolerates is headroom it did
not need, and the difference between two workloads' tolerated padding is the difference in
what they used. Every figure is net of an empty-workload baseline.

### Rejected for stack: Valgrind Massif `--stacks=yes`

Tried, and it fails the calibration above. It reported binCV's whole call at **8,216 B**
against an empty workload's **10,088 B** — the call using less stack than doing nothing, which
is impossible, and it still did so with snapshot counts between 766 and 980.

The mechanism is not imprecise tracking. Massif hooks `new_mem_stack` / `die_mem_stack`,
which fire on every stack-pointer change, so its tracking is exact. What is periodic is the
**recording**: stack size reaches the output only at snapshots, and Massif's peak-snapshot
logic is driven by heap size. A stack peak that rises and falls inside one call is tracked
exactly and never written down — which is why it works for a large, long-lived frame (a known
1 MiB frame reads within ~300 B) and fails for a transient few-kilobyte one, our case.

Two further documented caveats for anyone repeating this:

- **The peak snapshot is only ever taken after a deallocation.** A program that allocates and
  exits without freeing has *no* peak recorded, and `--peak-inaccuracy=0.0` does not change
  that. By default Massif also only records a new peak when it exceeds the previous by 1%.
- **`--max-stackframe` defaults to 2,000,000 B, and a larger stack-pointer move is treated as
  a stack switch and dropped** — the matching release is dropped too, so the accounting stays
  permanently inflated for the rest of the run.

Massif's *heap* figures are sound. Its stack figures are not usable at this granularity.

### Buffer arithmetic, for pipelines

The image-pipeline reports compute peak working set from buffer geometry — the live buffers of
one stage, in bytes — rather than sampling. That is exact for binCV, whose kernels allocate
nothing, so every buffer appears in a caller's signature. **Where the OpenCV side allocates
internally, buffer arithmetic cannot see it and the allocator probe is required.**

## Rules

- **Name the quantity.** "Memory" alone is not a claim.
- **Same instrument on both sides**, in the same process, around one call.
- **Peak live and cumulative traffic are two rows**, never one number.
- **Measure the stack whenever the operation keeps working arrays on it.**
- **Calibrate the instrument in the benchmark**, so a broken probe fails loudly rather than
  reporting a plausible number.
- **A figure that does not move with the input size is suspect.** That is what would have
  caught error 4 on the day it was introduced.

## What these figures are not

**A measured stack figure is a lower bound on worst-case stack, not the worst case.** It is
the deepest the stack went on the paths that ran; the probe runs each workload over six scenes
and reports the deepest. Repeated measurement with varied inputs *cannot* guarantee the
maximum is ever observed, so callers budgeting a thread from these numbers should keep margin.

**A guard page catches a stack pointer that walks into it, not one that steps over it.** A
large frame that is reserved and left mostly unwritten is where this technique can under-read,
so the benchmark measures that case directly — and **the two architectures differ**. A
65,536 B frame with only its shallow end written reads back as **65,536 B on x86-64 and 16 B
on aarch64**, where the stack pointer clears the guard page in one step and nothing faults.
The diagnostic row prints `UNDER-READ` when that happens, which is how this was found.

It does not affect the figures here, and that is checked rather than assumed:

- The *dense* calibration passes exactly on aarch64 — 4,096 B and 16,384 B both read back to
  the byte — and 4–16 KiB is the range the measured frames fall in. The failure needs a frame
  that is both large and untouched, which a dense linear-algebra solver does not produce.
- Cross-checked against the compiler on the device. Every frame in the five-point call graph
  is `static`, so `-fstack-usage` gives a real bound: `fivePointEssential` is 5,120 B and the
  deepest call path sums to about 6,432 B. The probe reads 6,928 B against that bound —
  *above* it, which is the safe direction and rules out a skipped frame.

```bash
# on the device, from the repository root
g++ -O2 -std=c++17 -I include -fstack-usage -c su_probe.cpp -o su_probe.o
sort -t$'\t' -k2 -rn su_probe.su      # frames, largest first
grep -c dynamic su_probe.su           # 0 means every frame is a real bound
```

**The comparison is symmetric, which is what the numbers rest on.** Both sides are measured by
the same instrument in the same process around one call, on the same quantity. A shared
limitation moves both columns together; an asymmetric one — our stack against their heap —
produces a ratio that means nothing. binCV has published that mistake twice.

**Memory figures are deterministic; timings on the development host are not.** The heap and
stack figures are byte-identical across repeated runs. The x86-64 host is a WSL2 VM whose
timing spread on these calls reaches 130%, so its *ratios* are indicative only; the reference
device, pinned and with its governor fixed, holds 0.1–0.6%.

## If a worst-case stack bound is ever needed

These reports give measured figures, not bounds. A real bound for binCV's own code comes from
`-fstack-usage` summed along the call graph, under conditions that must be checked rather than
assumed: only `static` and `dynamic bounded` frames are a reliable maximum; the sum is unsound
with recursion, indirect calls or any function with no `.su` data (GCC 11's
`-fcallgraph-info=su` marks both, so a tool can refuse to answer instead of under-reporting);
and frame sizes move with inlining, `-O` level and `-fstack-protector`, so the `.su` data must
come from the build being shipped. Of the two warnings, only `-Wstack-usage=N` is documented
as conservative — `-Wframe-larger-than=N` is approximate and excludes `alloca` and VLAs.

None of this is available for a prebuilt `libopencv`, which has no `.su` data — the reason
both sides here are measured with a binary-level instrument instead.

## Sources

- GCC, [Developer Options](https://gcc.gnu.org/onlinedocs/gcc/Developer-Options.html) —
  `-fstack-usage`, the `static` / `dynamic` / `bounded` definitions, `-fcallgraph-info`.
- GCC, [Warning Options](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html) —
  `-Wstack-usage` ("conservative") against `-Wframe-larger-than` ("approximate and not
  conservative").
- Valgrind — [Massif](https://valgrind.org/docs/manual/ms-manual.html) (peak recorded only
  after a deallocation; `--peak-inaccuracy`, `--stacks`),
  [DHAT](https://valgrind.org/docs/manual/dh-manual.html) (the cumulative / peak-live / leaked
  distinction; its own example has cumulative at 6.2× peak live), and the
  [core manual](https://valgrind.org/docs/manual/manual-core.html) (`--max-stackframe`).
- AdaCore, [The GNATstack Tool](https://docs.adacore.com/live/wave/gnatstack/html/gnatstack_ug/The_GNATstack_Tool.html)
  — the conditions a static stack bound needs.
- Kästner and Ferdinand, [Efficient Verification of Non-Functional Safety Properties](https://www.absint.com/Kaestner_ISSC2011.pdf)
  (ISSC 2011) — repeated measurement cannot guarantee the maximum is ever observed.
- Rapita Systems, [Function pointers and their impact on stack analysis](https://www.rapitasystems.com/blog/function-pointers-and-their-impact-stack-analysis)
  — an incomplete call graph fails *optimistically*.
- Memfault Interrupt, [Measuring Stack Usage the Hard Way](https://interrupt.memfault.com/blog/measuring-stack-usage)
  — painting under-counts a frame allocated but only partly written.
- Hertz and Berger, [Quantifying the Performance of GC vs. Explicit Memory Management](https://cse.buffalo.edu/~mhertz/gcmalloc-oopsla-2005.pdf)
  (OOPSLA 2005) — refusing a reported-allocation metric; reporting time and space jointly.
- Heiser, [Systems Benchmarking Crimes](https://gernot-heiser.org/benchmarking-crimes.html) —
  benchmark a competitor no less carefully than your own system.
- SIGPLAN, [Empirical Evaluation Checklist](https://www.sigplan.org/Resources/EmpiricalEvaluation/)
  — a proxy metric needs explicit justification; heap bytes as a proxy for footprint is one.
- SEI CERT, [MEM05-C](https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/recommendations/memory-management-mem/mem05-c)
  — an exhausted stack can terminate the program; a failed allocation returns a checkable value.
- Qualys, [The Stack Clash](https://www.qualys.com/2017/06/19/stack-clash/stack-clash.txt) — a
  stack pointer that jumps over the guard page raises no fault.
- Linux kernel, [sysctl/kernel](https://www.kernel.org/doc/Documentation/admin-guide/sysctl/kernel.rst)
  — `randomize_va_space`; pin ASLR and environment size when comparing stack measurements.

## Reproduce

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/benchmark/essential_benchmark        # heap, both sides, with its self-check
./build/benchmark/essential_stack_benchmark  # stack, with its calibration rows
```

Both refuse to print figures if their self-check fails.
