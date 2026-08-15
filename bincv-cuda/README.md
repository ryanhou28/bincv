# binCV-cuda

> **Status: deferred / exploratory.**
>
> GPU backends are [Phase 6](../ROADMAP.md#phase-6--deferred) — not currently
> scoped. binCV's product is the CPU path targeting embedded and mobile
> ([ARCHITECTURE §1](../ARCHITECTURE.md#1-scope)); Jetson-class devices run that
> path on their ARM CPU today.
>
> This directory holds early CUDA edge-filter experiments that predate the
> current architecture. It does not use the bit-plane data model and is not
> wired into the main build. Treat it as a reference sketch, not a component.

When GPU work resumes, the storage model
([ARCHITECTURE §4.3](../ARCHITECTURE.md#43-storage-model-and-views)) is designed
to make it possible without an API break: non-owning views over externally
allocated memory are the same mechanism used for DMA and unified memory.

## Requirements

- NVIDIA GPU with CUDA support, CUDA toolkit installed
- CMake
- OpenCV

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```
