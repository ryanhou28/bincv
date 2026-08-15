# Experiment probes

Measurement code backing entries in [EXPERIMENTS.md](../../EXPERIMENTS.md).
Every logged result must be reproducible from a committed probe.

| Probe | Entry | Build |
|---|---|---|
| `x1_alignment_footprint.cpp` | X-1 · row alignment memory cost | `g++ -std=c++17 -O2 -I ../include x1_alignment_footprint.cpp -o x1 && ./x1` |
| `x2_pyramid_bit_growth.cpp` | X-2 · pyramid bit growth | `g++ -std=c++17 -O2 $(pkg-config --cflags opencv4) x2_pyramid_bit_growth.cpp -o x2 $(pkg-config --libs opencv4) && ./x2` |
| `x3_popcount_codegen.c` | X-3 · popcount codegen | `clang -O2 --target=aarch64-linux-gnu -S -o - x3_popcount_codegen.c` |

Probes are throwaway measurement tools, not library code — they are not built by
CMake and are not held to library conventions.
