#!/usr/bin/env bash
# X-47's sweep -- one PROCESS per arm; see interop_roundtrip.cpp.
set -u
for i in $(seq 0 6); do ./benchmark/interop_roundtrip "$i"; done
