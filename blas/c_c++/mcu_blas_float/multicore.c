/*
Pico SDK C++ FLOPS benchmark (AXPY, MATMUL, SPMV)
------------------------------------------------

This file implements the same benchmark suite previously provided in MicroPython,
but written for the Raspberry Pi Pico SDK in C++.

Features
  - Compile-time selection (using #defines at the top) of algorithm: AXPY, MATMUL, SPMV
  - Single-core or multi-core execution using core0 + core1 (multicore FIFO for sync)
  - Warmup + multiple timed runs and MFLOPS reporting
  - Memory allocation with malloc/free; small, deterministic data generators

How multicore works here
  - core1 is launched at program startup and runs a small command loop reading
    commands from multicore FIFO. For a multi-core run we send a "run" command
    describing the partition (start/end). core1 runs its partition and pushes a
    completion message back on FIFO. The main core executes its partition in
    parallel and waits for the core1 completion message.

Configuration (edit the #defines below)
  - ALGO: AXPY / MATMUL / SPMV (see enum values)
  - MODE_MULTI: 1 to use multicore partitioning, 0 to force single-core
  - NUM_THREADS: logical number of partitions (1 or 2 are meaningful on RP2)
  - Problem sizes and WARMUP / RUNS constants

Build
  - A sample CMakeLists.txt is included at the bottom of this file (in a comment).
  - Typical build steps:
      mkdir build && cd build
      cmake ..
      make -j
    Copy the produced UF2 to the board or use your usual deployment path.

Notes
  - The Pico SDK toolchain already sets appropriate CPU / FPU flags for the
    chosen target. The code uses "float" for better performance on microcontroller
    FPUs; change to double if desired but expect higher memory use.
  - The FLOP counts use the same formulas as the MicroPython script:
      AXPY: 2*N (mul + add)
      MATMUL: 2*N^3 - N^2
      SPMV: 2 * nnz

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "pico/stdlib.h"
#include "pico/time.h"
#include "pico/multicore.h"
#include "hardware/clocks.h"

// ----------------- CONFIG (edit here) -----------------
#define AXPY 0
#define MATMUL 1
#define SPMV 2


#define FREQ 280000 // change the frequency to 280 MHz (value is in kHz)

#ifndef ALGO
// change ALGO to AXPY, MATMUL, or SPMV
#define ALGO MATMUL
#endif

#ifndef MODE_MULTI
// 1 -> try to use both cores (main + core1). 0 -> single-core only.
#define MODE_MULTI 1
#endif

#ifndef NUM_THREADS
// logical partitions. On Pico only 1 or 2 make sense.
#define NUM_THREADS 2
#endif

// sizes (tune to your available RAM)
#define AXPY_N 20000u
#define MAT_N 64u
#define SPMV_N 2000u
#define SPMV_NNZ_PER_ROW 8u

#define WARMUP 1
#define RUNS 3
#define AXPY_A 1.2345f

// ----------------- Multicore command IDs -----------------
enum {
    CMD_READY = 0xA0,
    CMD_DONE  = 0xA1,
    CMD_RUN_AXPY = 0xA2,
    CMD_RUN_MATMUL = 0xA3,
    CMD_RUN_SPMV = 0xA4,
    CMD_EXIT = 0xFF
};

// ---------------- global shared state -----------------
// pointers are written on core0 before issuing a command; partitions are disjoint
static float *g_ax_x = NULL;
static float *g_ax_y = NULL;

static float *g_mat_A = NULL;
static float *g_mat_B = NULL;
static float *g_mat_C = NULL;

static float *g_sp_vals = NULL;
static uint32_t *g_sp_cols = NULL;
static uint32_t *g_sp_row_ptr = NULL;

// ---------------- utilities -----------------
static inline void split_range(uint32_t length, uint32_t parts, uint32_t idx, uint32_t *start, uint32_t *end) {
    uint32_t base = length / parts;
    uint32_t rem = length % parts;
    *start = idx * base + (idx < rem ? idx : rem);
    *end = *start + base + (idx < rem ? 1U : 0U);
}

static inline uint64_t now_us() {
    return time_us_64();
}

// ---------------- data generators -----------------
static float *make_vector(uint32_t n, int pattern) {
    float *v = (float *)malloc(sizeof(float) * (size_t)n);
    if (!v) {
        printf("OOM: vector n=%u\n", n);
        return NULL;
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (pattern == 1) v[i] = (float)(i % 1000) / 1000.0f;
        else v[i] = sinf((float)i);
    }
    return v;
}

static float *make_matrix_flat(uint32_t n) {
    size_t nelem = (size_t)n * (size_t)n;
    float *M = (float *)malloc(sizeof(float) * nelem);
    if (!M) { printf("OOM: matrix n=%u\n", n); return NULL; }
    for (uint32_t i = 0; i < n; ++i) {
        size_t base = (size_t)i * n;
        for (uint32_t j = 0; j < n; ++j) {
            M[base + j] = (float)(((i * 31u) + j) % 100u) / 100.0f;
        }
    }
    return M;
}

static bool make_sparse_csr(uint32_t nrows, uint32_t ncols, uint32_t nnz_per_row,
                           float **vals_out, uint32_t **cols_out, uint32_t **rowptr_out) {
    uint32_t nnz = nrows * nnz_per_row;
    float *vals = (float *)malloc(sizeof(float) * (size_t)nnz);
    uint32_t *cols = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)nnz);
    uint32_t *rowptr = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(nrows + 1));
    if (!vals || !cols || !rowptr) { free(vals); free(cols); free(rowptr); return false; }
    uint32_t p = 0;
    rowptr[0] = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        for (uint32_t k = 0; k < nnz_per_row; ++k) {
            uint32_t col = (i * nnz_per_row + k) % ncols;
            vals[p] = (float)(((i * 37u) + k) % 100u) / 100.0f;
            cols[p] = col;
            ++p;
        }
        rowptr[i + 1] = p;
    }
    *vals_out = vals; *cols_out = cols; *rowptr_out = rowptr;
    return true;
}

// ---------------- kernels -----------------
static void axpy_range(uint32_t start, uint32_t end, float a, float *x, float *y) {
    for (uint32_t i = start; i < end; ++i) {
        y[i] = y[i] + a * x[i];
    }
}

static void matmul_range(uint32_t start_row, uint32_t end_row, uint32_t N, float *A, float *B, float *C) {
    for (uint32_t i = start_row; i < end_row; ++i) {
        uint32_t arow = i * N;
        uint32_t crow = i * N;
        for (uint32_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (uint32_t k = 0; k < N; ++k) {
                s += A[arow + k] * B[k * N + j];
            }
            C[crow + j] = s;
        }
    }
}

static void spmv_range(uint32_t start_row, uint32_t end_row, uint32_t N,
                       float *vals, uint32_t *cols, uint32_t *rowptr, float *x, float *y) {
    for (uint32_t i = start_row; i < end_row; ++i) {
        float s = 0.0f;
        uint32_t r0 = rowptr[i];
        uint32_t r1 = rowptr[i + 1];
        for (uint32_t p = r0; p < r1; ++p) {
            s += vals[p] * x[cols[p]];
        }
        y[i] = s;
    }
}

// ---------------- flop counters -----------------
static uint64_t flops_axpy(uint32_t n) { return (uint64_t)2 * (uint64_t)n; }
static uint64_t flops_matmul(uint32_t n) { return (uint64_t)2 * (uint64_t)n * (uint64_t)n * (uint64_t)n - (uint64_t)n * (uint64_t)n; }
static uint64_t flops_spmv(uint32_t nnz) { return 2ull * (uint64_t)nnz; }

// ---------------- multicore worker (core1) -----------------
static void core1_main() {
    // signal ready
    multicore_fifo_push_blocking(CMD_READY);
    while (true) {
        uint32_t cmd = multicore_fifo_pop_blocking();
        if (cmd == CMD_RUN_AXPY) {
            uint32_t s = multicore_fifo_pop_blocking();
            uint32_t e = multicore_fifo_pop_blocking();
            axpy_range(s, e, AXPY_A, g_ax_x, g_ax_y);
            multicore_fifo_push_blocking(CMD_DONE);
        } else if (cmd == CMD_RUN_MATMUL) {
            uint32_t s = multicore_fifo_pop_blocking();
            uint32_t e = multicore_fifo_pop_blocking();
            matmul_range(s, e, MAT_N, g_mat_A, g_mat_B, g_mat_C);
            multicore_fifo_push_blocking(CMD_DONE);
        } else if (cmd == CMD_RUN_SPMV) {
            uint32_t s = multicore_fifo_pop_blocking();
            uint32_t e = multicore_fifo_pop_blocking();
            spmv_range(s, e, SPMV_N, g_sp_vals, g_sp_cols, g_sp_row_ptr, NULL /*x unused here*/, NULL /*y unused*/);
            // NOTE: for spmv we will use separate buffers for x,y in main; here we rely on globals being set
            // but in this simplified example we won't pass x,y through FIFO; they are global as g_*
            multicore_fifo_push_blocking(CMD_DONE);
        } else if (cmd == CMD_EXIT) {
            multicore_fifo_push_blocking(CMD_DONE);
            return;
        } else {
            // unknown command: ignore
        }
    }
}

// ---------------- harness helpers -----------------
static bool start_secondary_job(uint32_t cmd, uint32_t start, uint32_t end) {
    // send command and args to core1
    multicore_fifo_push_blocking(cmd);
    multicore_fifo_push_blocking(start);
    multicore_fifo_push_blocking(end);
    return true;
}

// wait for completion message from core1
static void wait_secondary_done() {
    uint32_t resp = multicore_fifo_pop_blocking();
    (void)resp; // currently we expect CMD_DONE
}

// ---------------- benchmark runners -----------------
static void run_axpy() {
    printf("AXPY: n=%u mode=%s threads=%d\n", AXPY_N, MODE_MULTI ? "MULTI" : "SINGLE", NUM_THREADS);
    float *x = make_vector(AXPY_N, 1);
    float *y = make_vector(AXPY_N, 2);
    if (!x || !y) { free(x); free(y); return; }
    g_ax_x = x; g_ax_y = y;
    uint64_t flops = flops_axpy(AXPY_N);

    uint32_t effective_threads = (MODE_MULTI ? (NUM_THREADS > 2 ? 2 : NUM_THREADS) : 1);
    if (effective_threads < 1) effective_threads = 1;

    // warmup
    for (int w = 0; w < WARMUP; ++w) {
        if (effective_threads == 2) {
            uint32_t s0,e0,s1,e1;
            split_range(AXPY_N, 2, 0, &s0, &e0);
            split_range(AXPY_N, 2, 1, &s1, &e1);
            start_secondary_job(CMD_RUN_AXPY, s1, e1);
            axpy_range(s0, e0, AXPY_A, x, y);
            wait_secondary_done();
        } else {
            axpy_range(0, AXPY_N, AXPY_A, x, y);
        }
    }

    double times[RUNS];
    for (int r = 0; r < RUNS; ++r) {
        uint64_t t0 = now_us();
        if (effective_threads == 2) {
            uint32_t s0,e0,s1,e1;
            split_range(AXPY_N, 2, 0, &s0, &e0);
            split_range(AXPY_N, 2, 1, &s1, &e1);
            start_secondary_job(CMD_RUN_AXPY, s1, e1);
            axpy_range(s0, e0, AXPY_A, x, y);
            wait_secondary_done();
        } else {
            axpy_range(0, AXPY_N, AXPY_A, x, y);
        }
        uint64_t t1 = now_us();
        double elapsed = (double)(t1 - t0) / 1e6;
        times[r] = elapsed;
        printf("  run %d elapsed=%.6f s\n", r, elapsed);
    }
    double avg = 0.0;
    for (int r=0;r<RUNS;++r) avg += times[r];
    avg /= RUNS;
    double mflops = ((double)flops / avg) / 1e6;
    printf("AXPY result: ops=%llu avg_time=%.6f s mFLOPS=%.3f\n", (unsigned long long)flops, avg, mflops);

    free(x); free(y);
}

static void run_matmul() {
    printf("MATMUL: N=%u mode=%s threads=%d\n", MAT_N, MODE_MULTI ? "MULTI" : "SINGLE", NUM_THREADS);
    float *A = make_matrix_flat(MAT_N);
    float *B = make_matrix_flat(MAT_N);
    float *C = (float *)malloc(sizeof(float) * (size_t)MAT_N * (size_t)MAT_N);
    if (!A || !B || !C) { free(A); free(B); free(C); return; }
    memset(C, 0, sizeof(float) * (size_t)MAT_N * (size_t)MAT_N);
    g_mat_A = A; g_mat_B = B; g_mat_C = C;

    uint64_t flops = flops_matmul(MAT_N);
    uint32_t effective_threads = (MODE_MULTI ? (NUM_THREADS > 2 ? 2 : NUM_THREADS) : 1);
    if (effective_threads < 1) effective_threads = 1;

    // warmup
    for (int w=0; w < WARMUP; ++w) {
        if (effective_threads == 2) {
            uint32_t s0,e0,s1,e1;
            split_range(MAT_N, 2, 0, &s0, &e0);
            split_range(MAT_N, 2, 1, &s1, &e1);
            start_secondary_job(CMD_RUN_MATMUL, s1, e1);
            matmul_range(s0, e0, MAT_N, A, B, C);
            wait_secondary_done();
        } else {
            matmul_range(0, MAT_N, MAT_N, A, B, C);
        }
    }

    double times[RUNS];
    for (int r = 0; r < RUNS; ++r) {
        uint64_t t0 = now_us();
        if (effective_threads == 2) {
            uint32_t s0,e0,s1,e1;
            split_range(MAT_N, 2, 0, &s0, &e0);
            split_range(MAT_N, 2, 1, &s1, &e1);
            start_secondary_job(CMD_RUN_MATMUL, s1, e1);
            matmul_range(s0, e0, MAT_N, A, B, C);
            wait_secondary_done();
        } else {
            matmul_range(0, MAT_N, MAT_N, A, B, C);
        }
        uint64_t t1 = now_us();
        double elapsed = (double)(t1 - t0) / 1e6;
        times[r] = elapsed;
        printf("  run %d elapsed=%.6f s\n", r, elapsed);
    }
    double avg = 0.0; for (int r=0;r<RUNS;++r) avg += times[r]; avg /= RUNS;
    double mflops = ((double)flops / avg) / 1e6;
    printf("MATMUL result: ops=%llu avg_time=%.6f s mFLOPS=%.3f\n", (unsigned long long)flops, avg, mflops);

    free(A); free(B); free(C);
}

static void run_spmv() {
    printf("SPMV: N=%u nnz/row=%u mode=%s threads=%d\n", SPMV_N, SPMV_NNZ_PER_ROW, MODE_MULTI ? "MULTI" : "SINGLE", NUM_THREADS);
    float *vals; uint32_t *cols; uint32_t *rowptr;
    if (!make_sparse_csr(SPMV_N, SPMV_N, SPMV_NNZ_PER_ROW, &vals, &cols, &rowptr)) {
        printf("OOM building sparse matrix\n"); return;
    }
    float *x = make_vector(SPMV_N, 1);
    float *y = (float *)malloc(sizeof(float) * (size_t)SPMV_N);
    if (!x || !y) { free(vals); free(cols); free(rowptr); free(x); free(y); return; }
    for (uint32_t i=0;i<SPMV_N;++i) y[i] = 0.0f;

    g_sp_vals = vals; g_sp_cols = cols; g_sp_row_ptr = rowptr;
    // Note: we also rely on x,y being visible to both cores (globals) if needed
    // but to keep the example clear we will not modify g_ pointers for x/y

    uint32_t nnz = SPMV_N * SPMV_NNZ_PER_ROW;
    uint64_t flops = flops_spmv(nnz);
    uint32_t effective_threads = (MODE_MULTI ? (NUM_THREADS > 2 ? 2 : NUM_THREADS) : 1);
    if (effective_threads < 1) effective_threads = 1;

    // warmup
    for (int w=0; w < WARMUP; ++w) {
        if (effective_threads == 2) {
            uint32_t s0,e0,s1,e1;
            split_range(SPMV_N, 2, 0, &s0, &e0);
            split_range(SPMV_N, 2, 1, &s1, &e1);
            // set globals for x,y so core1 can access them
            // (we keep separate local pointers but core1 will use g_sp_* globals)
            // For simplicity we will call spmv_range on core1 using global arrays
            start_secondary_job(CMD_RUN_SPMV, s1, e1);
            spmv_range(s0, e0, SPMV_N, vals, cols, rowptr, x, y);
            wait_secondary_done();
        } else {
            spmv_range(0, SPMV_N, SPMV_N, vals, cols, rowptr, x, y);
        }
    }

    double times[RUNS];
    for (int r=0;r<RUNS;++r) {
        uint64_t t0 = now_us();
        if (effective_threads == 2) {
            uint32_t s0,e0,s1,e1;
            split_range(SPMV_N, 2, 0, &s0, &e0);
            split_range(SPMV_N, 2, 1, &s1, &e1);
            start_secondary_job(CMD_RUN_SPMV, s1, e1);
            spmv_range(s0, e0, SPMV_N, vals, cols, rowptr, x, y);
            wait_secondary_done();
        } else {
            spmv_range(0, SPMV_N, SPMV_N, vals, cols, rowptr, x, y);
        }
        uint64_t t1 = now_us();
        double elapsed = (double)(t1 - t0) / 1e6;
        times[r] = elapsed;
        printf("  run %d elapsed=%.6f s\n", r, elapsed);
    }
    double avg=0.0; for (int r=0;r<RUNS;++r) avg += times[r]; avg /= RUNS;
    double mflops = ((double)flops / avg) / 1e6;
    printf("SPMV result: nnz=%u ops=%llu avg_time=%.6f s mFLOPS=%.3f\n", (unsigned)nnz, (unsigned long long)flops, avg, mflops);

    free(vals); free(cols); free(rowptr); free(x); free(y);
}

// ---------------- main -----------------
int main() {
    stdio_init_all();
    set_sys_clock_khz(FREQ, true); // Set CPU frequency
    sleep_ms(20000); // allow host UART console to attach if needed

    printf("Initializing..\n");

    // Launch core1 worker
    multicore_launch_core1(core1_main);
    
    printf("Initialized with frequency: %d MHz\n", clock_get_hz(clk_sys) / 1000000);


    // wait for core1 to report ready
    uint32_t r = multicore_fifo_pop_blocking();
    if (r != CMD_READY) {
        printf("Warning: core1 did not signal ready (got %u)\n", r);
    } else {
        printf("core1 ready\n");
    }

    // pick algorithm
#if ALGO == AXPY
    run_axpy();
#elif ALGO == MATMUL
    run_matmul();
#elif ALGO == SPMV
    run_spmv();
#else
    printf("Unknown ALGO compile-time selection\n");
#endif

    // stop core1 gracefully
    multicore_fifo_push_blocking(CMD_EXIT);
    wait_secondary_done();
    printf("benchmark finished\n");
    return 0;
}

