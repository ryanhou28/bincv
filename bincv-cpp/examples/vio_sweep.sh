#!/usr/bin/env bash
# The same frontend under two detection policies. Run from the build directory.
set -u
D="${1:?usage: vio_sweep.sh <frame-dir>}"
for lw in 1.0 0.6; do
  echo "########## BINCV_VIO_LOW=$lw ##########"
  BINCV_VIO_LOW=$lw ./examples/vio_frontend "$D"
done
