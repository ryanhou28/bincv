#!/bin/bash

# Root of the repository (assuming script is in bincv-cpp/scripts)
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
BIN_DIR="$ROOT_DIR/build/benchmark"

# Fixed config
WIDTH=640
HEIGHT=480
DTYPE="uint8"
SPARSITY=1.0
ITERATIONS=100

echo "Running benchmarks from: $BIN_DIR"
echo "Saving logs to:         $ROOT_DIR/results"

# Create results directory
mkdir -p "$ROOT_DIR/results"

# Detect all executable files in the benchmark binary directory
BENCHMARKS=()
while IFS= read -r -d '' file; do
    BENCHMARKS+=("$file")
done < <(find "$BIN_DIR" -maxdepth 1 -type f -executable -print0)

# Run each detected benchmark
for bench_path in "${BENCHMARKS[@]}"; do
    bench_name=$(basename "$bench_path")

    echo -e "\n=== [$bench_name] ==="
    "$bench_path" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --dtype "$DTYPE" \
        --sparsity "$SPARSITY" \
        --iterations "$ITERATIONS" \
        2>&1 | tee -a "$ROOT_DIR/results/${bench_name}.log"

    echo ""
done

echo -e "\n All benchmarks complete. Logs saved to: $ROOT_DIR/results/"
