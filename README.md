# MCU-bench

**Microcontroller BLAS-style benchmark suite — C/C++ and MicroPython**

**Repository:** `ntampouratzis/MCU-bench`

---

## Overview

This repository contains a small, focused benchmark suite for microcontrollers (MCUs) implementing BLAS-style kernels with the goal of measuring **compute throughput** (FLOPS / OPs) and **operation rates** for both floating-point (exploiting the most recent FPUs) and integer workloads. The suite implements three kernels:

- **AXPY** (y := a \* x + y)
- **Matrix multiplication** (dense GEMM-like kernel)
- **Sparse matrix-vector multiply (SPMV)**

Each kernel is available in two language tracks:

1. **C/C++** (native builds that exercise MCU toolchains, optimized compiler flags and — where present — the hardware FPU and DSP units)
2. **MicroPython** (ease of use, accessibility, and a view of interpreter-level performance)

The benchmark contains both **single-core** and **multi-core** variants (where the target architecture supports multiple cores, e.g., RP2350 and ESP32-S3) and features **efficient synchronisation through FIFO** for multicore implementations. It is **highly parameterizable**, allowing configuration of number of cores, CPU frequency, and algorithm choice to match different performance scenarios. This makes it easy to explore hardware limits, compare architectures, and fine-tune workloads for optimal results on modern multi-core MCUs.

---

## Key goals

- Provide small, representative BLAS-like kernels (AXPY, GEMM, SPMV) that exercise FPUs/DSPs on modern MCUs.
- Measure both **floating-point** and **integer** performance.
- Support both **native C/C++** and **MicroPython** implementations to compare interpreter vs native performance.
- Support **single-core** and **multicore** runs with efficient FIFO-based synchronisation.

## Supported/tested targets (as of initial testing)

- **Raspberry Pi Pico 2 (RP2350)** — tested
- **ESP32-S3** — tested

## Repository layout (recommended / detected structure)

```
/ (root)
├─ c_c++/                  # native builds: C/C++ implementations + CMake / Makefiles
│  ├─ mcu_blas_float/             
|  |  ├─ CMakeLists.txt/   # kernel implementations + single/multicore variants
│  |  ├─ multicore.c       # C benchamrk suite for axpy, matmul, spmv using floating point operations (exploiting the FPU and DSPs)
│  ├─ mcu_blas_int/             
|  |  ├─ CMakeLists.txt/   # kernel implementations + single/multicore variants
│  |  ├─ multicore.c       # Cbenchamrk suite for axpy, matmul, spmv using integer operations
├─ micropython/          
|  ├─ mcu_blas_float.py/      # MicroPython benchamrk suite for axpy, matmul, spmv using floating point operations (exploiting the FPU and DSPs)
|  ├─ mcu_blas_float.py/      # MicroPython benchamrk suite for axpy, matmul, spmv using integer operations

```

## How the benchmarks work (methodology)
1. **Warm-up**: each test performs a configurable warm-up run to ensure caches, PLLs and dynamic frequency scaling settle and to reduce the impact of one-off initialization costs.
2. **Timed trials**: perform `N` trials of the kernel and measure elapsed cycles/time for each trial.
3. **Aggregate & statistical reporting**: compute mean, median, standard deviation, and min/max for the measured times. Report **converted units** (seconds, FLOPS, OPs).

**Important measurement details**
- Use high-resolution timers where possible.
- For floating-point FLOPS, compute the *exact* number of FLOPs executed by the kernel and divide by measured time.
- For integer OPs, similarly count integer operations.
- In multicore tests, rely on **FIFO-based synchronisation** to coordinate work between cores with minimal overhead.

## Running (MicroPython)
Download Thonny environment and build or obtain a MicroPython firmware for the target device (in case of RP2350 RISC-V based you can download the latest RISC-V firmware https://micropython.org/download/RPI_PICO2/).


## Running (C/C++)

