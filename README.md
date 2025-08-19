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

1. **C/C++** (native builds that exercise MCU toolchains, optimized compiler flags and — where present — the hardware FPU and DSP units). Our benchmark **automatically detects the platform environment**.
2. **MicroPython** (ease of use, accessibility, and a view of interpreter-level performance)

The benchmark contains both **single-core** and **multi-core** variants (where the target architecture supports multiple cores, e.g., RP2350 and ESP32-S3) and features **efficient synchronisation through FIFO** and **FreeRTOS queues** for multicore implementations. It is **highly parameterizable**, allowing configuration of number of cores, CPU frequency, and algorithm choice to match different performance scenarios. This makes it easy to explore hardware limits, compare architectures, and fine-tune workloads for optimal results on modern multi-core MCUs.

---

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

## How the benchmarks work
1. **Warm-up**: each test performs a configurable warm-up run to ensure caches, PLLs and dynamic frequency scaling settle and to reduce the impact of one-off initialization costs.
2. **Timed trials**: perform `N` trials of the kernel and measure elapsed cycles/time for each trial.
3. **Aggregate & statistical reporting**: compute mean, median, standard deviation, and min/max for the measured times. Report **converted units** (seconds, FLOPS, OPs).

**Important measurement details**
- Use high-resolution timers where possible.
- In multicore tests, rely on **FIFO-based synchronisation** as well as **FreeRTOS queues** to coordinate work between cores with minimal overhead.
- We have measured the **FLOPS/Watt** using the YOJOCK USB C Digital Multimeter.

## Running (MicroPython)
Download Thonny environment (https://thonny.org/) and build or obtain a MicroPython firmware for the target device.
- In case of Pico (RP2040 and RP2350): 
  1. Press the BOOTSEL button and hold it while you connect it through USB to computer (This puts your Raspberry Pi Pico into USB mass storage device mode).
  2. You may download the latest ARM or RISC-V firmware for RP2350 from here: https://micropython.org/download/RPI_PICO2/.
- In case of ESP32-S3: 
  1. Press the BOOTSEL button and hold it while you connect it through USB to computer
  2. Select configure Interpreter (right-bottom)
  3. Select Install or Update Micropython (esptool)
  4. Select the following:
  <img width="640" height="480" alt="3" src="https://github.com/user-attachments/assets/16262da9-9e96-4631-8ed6-5a2f67505d93" />
  <img width="1915" height="336" alt="5" src="https://github.com/user-attachments/assets/b2d44d09-3cd1-4116-96dc-f1cec596b332" />

## Running (C/C++)
In case of Pico (RP2040 and RP2350):
- Install Required Packages
```
sudo apt update
sudo apt install -y cmake gcc-arm-none-eabi libnewlib-arm-none-eabi build-essential git

wget https://github.com/raspberrypi/pico-sdk-tools/releases/download/v2.1.1-3/riscv-toolchain-15-x86_64-lin.tar.gz
sudo mkdir -p /opt/riscv/riscv-toolchain-15
sudo chown $USER /opt/riscv/riscv-toolchain-15
tar xvf riscv-toolchain-15-x86_64-lin.tar.gz -C /opt/riscv/riscv-toolchain-15

export PICO_TOOLCHAIN_PATH=/opt/riscv/riscv-toolchain-15/
```

- Get the Pico SDK 2.2.0
```
mkdir -p ~/pico
cd ~/pico
git clone -b 2.2.0 https://github.com/raspberrypi/pico-sdk.git
cd pico-sdk
git submodule update --init
echo "export PICO_SDK_PATH=$HOME/pico/pico-sdk" >> ~/.bashrc
source ~/.bashrc
```


- Get Example Code & copy the files
```
cd ~/pico
git clone -b 2.2.0 https://github.com/raspberrypi/pico-examples.git
cp ~/MCU-bench/blas/c_c++/* ~/pico/pico-examples/multicore/hello_multicore
```

- Prepare the project for ARM architecture
```
cmake -DPICO_PLATFORM=rp2350 ..
```

- OR prepare the project for ROSC-V architecture
```
cmake -DPICO_PLATFORM=rp2350-riscv ..
```

- Build the project
```
make hello_multicore
```

- Press the BOOTSEL button and hold it while you connect it through USB to computer & Copy the .uf2
```
cp ~/pico/pico-examples/build/multicore/hello_multicore/hello_multicore.uf2 /media/$USER/RP2350/
```

In case of ESP32 (ESP32-S3):
- You can use the instructions which described in the official tutorial: https://www.waveshare.com/wiki/ESP32-S3-Pico
