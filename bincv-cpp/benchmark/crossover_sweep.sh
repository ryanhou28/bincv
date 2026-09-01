#!/usr/bin/env bash
# X-46's sweep. One PROCESS per arm -- see bitwidth_crossover.cpp for why the
# single-process version was wrong. Run from the build directory:
#   ./scripts/run_on_pi.sh pi4 'bash ../benchmark/crossover_sweep.sh'
set -u
for i in $(seq 0 15); do ./benchmark/bitwidth_crossover "$i"; done
