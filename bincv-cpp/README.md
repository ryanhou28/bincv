# BinCV C++


## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Benchmark

After building:

```bash
cd scripts
chmod +x run_all_benchmarks.sh
./run_all_benchmarks.sh
```