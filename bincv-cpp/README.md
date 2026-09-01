# binCV C++

CPU implementation of binCV. This is the primary implementation — see the
[top-level README](../README.md) for what the project is and
[ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the design.

The core is **header-only with no dependencies**. OpenCV is optional and enables
interop plus the comparison benchmarks.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

### Without OpenCV (core-only / embedded)

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release -DBINCV_USE_OPENCV=OFF
cmake --build build-core -j$(nproc)
```

### Options

| Option | Default | Effect |
|---|---|---|
| `BINCV_USE_OPENCV` | ON | Interop, interop tests, benchmarks |
| `BINCV_BUILD_TESTS` | ON | Build and register tests with ctest |
| `BINCV_BUILD_BENCHMARKS` | ON | Comparison benchmarks (needs OpenCV) |

## Benchmark

```bash
./build/benchmark/fill_benchmark --width 640 --height 480 \
    --iterations 100 --dtype binary --sparsity 0.5
```

Or the full sweep:

```bash
cd scripts && ./run_all_benchmarks.sh
```

Always benchmark a Release build. See
[GETTING_STARTED.md](../GETTING_STARTED.md) for the benchmarking rules —
particularly the correct comparison denominator.
