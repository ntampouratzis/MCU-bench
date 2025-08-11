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

The project contains both **single-core** and **multi-core** variants (where the target architecture supports multiple cores, e.g., RP2350 and ESP32-S3) and features **efficient synchronisation through FIFO** for multicore implementations. It is intended to evaluate real-world compute throughput on small devices and to be extensible to new boards and kernels.

---

## Key goals

- Provide small, representative BLAS-like kernels (AXPY, GEMM, SPMV) that exercise FPUs/DSPs on modern MCUs.
- Measure both **floating-point** and **integer** performance.
- Support both **native C/C++** and **MicroPython** implementations to compare interpreter vs native performance.
- Support **single-core** and **multicore** runs with efficient FIFO-based synchronisation.
