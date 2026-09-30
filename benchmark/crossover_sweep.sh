#!/usr/bin/env bash
# The bit-width crossover sweep. One PROCESS per arm -- see bitwidth_crossover.cpp for why the
# single-process version was wrong. Run from the build directory:
#   bash ../benchmark/crossover_sweep.sh
# For a pinned, multi-launch log on the reference device, also from the build directory:
#   ../scripts/run_launches.sh -n 10 -g ../benchmark/crossover_sweep.sh
set -u
for i in $(seq 0 15); do ./benchmark/bitwidth_crossover "$i"; done
