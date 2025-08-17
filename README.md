# MCU-bench

**A multi-core ARM & RISC-V Microcontroller benchmark suite — C/C++ and MicroPython**

---

## Overview

This repository contains a **multi-core** benchmark suite for microcontrollers (MCUs) implementing both basic operations and BLAS-style kernels. The main goal is the accurate measurement (**compute throughput** (FLOPS / OPs) and **operation rates**) for both floating-point (exploiting the most recent FPUs) and integer workloads, which is compatible in a wide range of MCU architectures (**ARM**, **RISC-V** and **Espressif**).

The suite implements four kernels:

- **AXPY** (y := a \* x + y)
- **Matrix multiplication** (dense GEMM-like kernel)
- **Sparse matrix-vector multiply (SPMV)**
- **Basic operations** (add, sub, mult, div)

Each kernel is available in two language tracks:

1. **C/C++** (native builds that exercise MCU toolchains, optimized compiler flags and — where present — the hardware FPU and DSP units). Our benchmark automatically detects the platform environment.
2. **MicroPython** (ease of use, accessibility, and a view of interpreter-level performance)

The benchmark contains both **single-core** and **multi-core** variants (where the target architecture supports multiple cores, e.g., RP2350 and ESP32-S3) and features **efficient synchronisation through FIFO** and **FreeRTOS queues** for multicore implementations. It is **highly parameterizable**, allowing configuration of number of cores, CPU frequency, and algorithm choice to match different performance scenarios. This makes it easy to explore hardware limits, compare architectures, and fine-tune workloads for optimal results on modern multi-core MCUs.

---

## Key goals

- Provide small, representative BLAS-like kernels (AXPY, GEMM, SPMV) that exercise FPUs/DSPs on modern MCUs.
- Measure both **floating-point** and **integer** performance.
- Support both **native C/C++** and **MicroPython** implementations to compare interpreter vs native performance.
- Support **single-core** and **multicore** runs with efficient FIFO-based synchronisation.

<!-- ## Supported/tested targets (as of initial testing)

- **Raspberry Pi Pico 2 (RP2350)** — tested
- **ESP32-S3** — tested -->

## Repository layout

```
/ (root)
├─ basic_operations/        # Basic multi-core MCU based implementation (add, sub, mult, div)  
│  ├─ micropython/             
|  |  ├─ mcu_operations.py  # micropython implementation (tested through Thonny)
│  ├─ c_c++/             
|  |  ├─ CMakeLists.txt/    # CMake (automatically detect the SDK environment - tested with Pico SDK and ESP-EDF)
│  |  ├─ multicore.c        # C/C++ implementation for add, sub, mult, div using floating point operations
├─ blas/                    # BLAS multi-core MCU based implementation (AXPY, MATMUL, SPMV)
│  ├─ micropython/             
|  |  ├─ mcu_blas.py        # micropython implementation (tested with Thonny)
│  ├─ c_c++/             
|  |  ├─ CMakeLists.txt/    # CMake (automatically detect the SDK environment - tested with Pico SDK and ESP-EDF)
│  |  ├─ multicore.c        # C/C++ implementation for axpy, matmul, spmv using both float and integer operations
```

## How the benchmarks work (methodology)
1. **Warm-up**: each test performs a configurable warm-up run to ensure caches, PLLs and dynamic frequency scaling settle and to reduce the impact of one-off initialization costs.
2. **Timed trials**: perform `N` trials of the kernel and measure elapsed cycles/time for each trial.
3. **Aggregate & statistical reporting**: compute mean, median, standard deviation, and min/max for the measured times. Report **converted units** (seconds, FLOPS, OPs).

**Important measurement details**
- Use high-resolution timers where possible.
- In multicore tests, rely on **FIFO-based synchronisation** to coordinate work between cores with minimal overhead.
- We have measured the **FLOPS/Watt** using the YOJOCK USB C Digital Multimeter.

## Running (MicroPython)
Download Thonny environment and build or obtain a MicroPython firmware for the target device (in case of RP2350 RISC-V based you can download the latest RISC-V firmware https://micropython.org/download/RPI_PICO2/).


## Running (C/C++)

