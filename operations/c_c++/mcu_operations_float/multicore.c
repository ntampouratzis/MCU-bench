/*
 * pico2_flops_benchmark_c.c
 *
 * C Pico SDK FLOPS benchmark for Raspberry Pi Pico 2 (RP2350)
 * - Measures single-precision FLOPS for add/sub/mul/div.
 * - Compile-time selection of operation using #defines (edit top of file).
 * - Can run single-core (core 0) or dual-core (core 0 + core 1) using the SDK multicore API.
 * - Mirrors MicroPython benchmark settings: UNROLL=8, CHUNK=128, default DURATION_S=2.0f
 *
 * Build: (from project root)
 *   mkdir build
 *   cd build
 *   export PICO_SDK_PATH=/path/to/pico-sdk   # if not already set
 *   cmake .. -DPICO_BOARD=pico2 -DCMAKE_BUILD_TYPE=Release
 *   make -j4
 *
 * Copy the generated .uf2 to your Pico 2 and open a serial terminal (115200).
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "pico/stdlib.h"
#include "pico/time.h"
#include "pico/multicore.h"
#include "hardware/clocks.h"

// ----------------------------- Configuration (edit as desired) -----------------------------

#define FREQ 280000 // change the frequency to 280 MHz (value is in kHz)

// Uncomment exactly ONE operation:
//#define OP_ADD
//#define OP_SUB
//#define OP_MUL
#define OP_DIV

// Use multicore (1) or single-core (0)?
#define USE_MULTICORE 1

// Benchmark parameters (match MicroPython defaults)
#ifndef DURATION_S
#define DURATION_S 2.0f    // seconds
#endif
#ifndef UNROLL
#define UNROLL 8
#endif
#ifndef CHUNK
#define CHUNK 128
#endif
// -----------------------------------------------------------------------------------------

#if !defined(OP_ADD) && !defined(OP_SUB) && !defined(OP_MUL) && !defined(OP_DIV)
#error "Please define one of OP_ADD / OP_SUB / OP_MUL / OP_DIV"
#endif

#if defined(OP_ADD)
#define OP_NAME "add"
#elif defined(OP_SUB)
#define OP_NAME "sub"
#elif defined(OP_MUL)
#define OP_NAME "mul"
#elif defined(OP_DIV)
#define OP_NAME "div"
#endif


typedef struct {
    uint64_t ops;
    uint64_t start_us;
    uint64_t end_us;
    float checksum;
} bench_result_t;

// Shared result for multicore mode. Mark volatile to prevent compiler reordering/elimination.
static volatile bench_result_t multicore_result;

// Unroll macro (tuned for UNROLL == 8)
#if UNROLL != 8
#warning "This file is tuned for UNROLL==8; change DO_UNROLL if you change UNROLL."
#endif

#if defined(OP_ADD)
#define DO_UNROLL(x,y) \
    x += y; x += y; x += y; x += y; x += y; x += y; x += y; x += y
#elif defined(OP_SUB)
#define DO_UNROLL(x,y) \
    x -= y; x -= y; x -= y; x -= y; x -= y; x -= y; x -= y; x -= y
#elif defined(OP_MUL)
#define DO_UNROLL(x,y) \
    x *= y; x *= y; x *= y; x *= y; x *= y; x *= y; x *= y; x *= y
#elif defined(OP_DIV)
#define DO_UNROLL(x,y) \
    x /= y; x /= y; x /= y; x /= y; x /= y; x /= y; x /= y; x /= y
#endif

static void bench_worker(bench_result_t *res, int core_id, float duration_s) {
    const uint64_t duration_us = (uint64_t)(duration_s * 1e6f);

    // Choose operands suitable for the operation (avoid divide-by-zero)
#if defined(OP_ADD)
    float x = 1.234567f;
    const float y = 2.345678f;
#elif defined(OP_SUB)
    float x = 12345.6789f;
    const float y = 234.56789f;
#elif defined(OP_MUL)
    float x = 1.00000123f;
    const float y = 1.00000234f;
#elif defined(OP_DIV)
    float x = 12345.6789f;
    const float y = 1.2345f; // non-zero
#endif

    uint64_t local_ops = 0;
    // small warm-up
    for (int i = 0; i < 16; ++i) {
        DO_UNROLL(x, y);
        local_ops += UNROLL;
    }

    uint64_t start = time_us_64();
    while (true) {
        for (int c = 0; c < CHUNK; ++c) {
            DO_UNROLL(x, y);
            local_ops += UNROLL;
        }
        uint64_t now = time_us_64();
        if ((now - start) >= duration_us) break;
    }
    uint64_t end = time_us_64();

    res->ops = local_ops;
    res->start_us = start;
    res->end_us = end;
#if defined(OP_ADD) || defined(OP_SUB)
    res->checksum = x + y;
#else
    res->checksum = x * y;
#endif
}

// core1 entry point
void core1_entry() {
    // Signal readiness to core0
    multicore_fifo_push_blocking(0xC0FEBABE);

    // Wait for START token from core0
    (void)multicore_fifo_pop_blocking();

    // Run benchmark and write into shared result
    bench_worker((bench_result_t *)&multicore_result, 1, DURATION_S);

    // Signal completion to core0
    multicore_fifo_push_blocking(0xBEEFDEAD);
}

int main(void) {
    stdio_init_all();
    set_sys_clock_khz(FREQ, true); // Set CPU frequency
    sleep_ms(20000); // allow host UART console to attach if needed

    printf("Initializing..\n");

    printf("Pico 2 FLOPS benchmark (C)\n");
    printf("Operation: %s\n", OP_NAME);
#if USE_MULTICORE
    printf("Mode: multi-core (2 cores)\n");
#else
    printf("Mode: single-core\n");
#endif
    printf("Duration: %.2f s, UNROLL=%d, CHUNK=%d\n", (double)DURATION_S, UNROLL, CHUNK);
    printf("System clock: %u Hz\n", clock_get_hz(clk_sys));

#if USE_MULTICORE
    // Launch core 1
    multicore_launch_core1(core1_entry);

    // Wait for core1 readiness
    uint32_t tag = multicore_fifo_pop_blocking();
    if (tag != 0xC0FEBABE) {
        printf("Warning: unexpected readiness tag: 0x%08x\n", tag);
    }

    // Send start token to core1
    multicore_fifo_push_blocking(0xF00D);

    // Run worker on core0
    bench_result_t core0_res = {0};
    bench_worker(&core0_res, 0, DURATION_S);
    
    printf("Initialized with frequency: %d MHz\n", clock_get_hz(clk_sys) / 1000000);

    // Wait for completion tag from core1
    uint32_t done = multicore_fifo_pop_blocking(); (void)done;

    // Copy multicore_result into local structure
    bench_result_t core1_res;
    core1_res.ops = multicore_result.ops;
    core1_res.start_us = multicore_result.start_us;
    core1_res.end_us = multicore_result.end_us;
    core1_res.checksum = multicore_result.checksum;

    double dur0 = (core0_res.end_us - core0_res.start_us) / 1e6;
    double dur1 = (core1_res.end_us - core1_res.start_us) / 1e6;
    double flops0 = (double)core0_res.ops / dur0;
    double flops1 = (double)core1_res.ops / dur1;

    printf("\n=== Per-core results ===\n");
    printf("core0: ops=%llu time=%.6fs flops=%.1f checksum=%g\n",
           (unsigned long long)core0_res.ops, dur0, flops0, core0_res.checksum);
    printf("core1: ops=%llu time=%.6fs flops=%.1f checksum=%g\n",
           (unsigned long long)core1_res.ops, dur1, flops1, core1_res.checksum);

    uint64_t earliest_start = core0_res.start_us < core1_res.start_us ? core0_res.start_us : core1_res.start_us;
    uint64_t latest_end = core0_res.end_us > core1_res.end_us ? core0_res.end_us : core1_res.end_us;
    double wall_time = (latest_end - earliest_start) / 1e6;
    uint64_t total_ops = core0_res.ops + core1_res.ops;
    double aggregated_flops = (double)total_ops / wall_time;

    printf("----------------------------\n");
    sleep_ms(1000);
    printf("Aggregated: total_ops=%llu wall_time=%.6fs aggregated_FLOPS=%.1f\n",
           (unsigned long long)total_ops, wall_time, aggregated_flops);
    sleep_ms(1000);
    

#else // single-core
    bench_result_t core0_res = {0};
    bench_worker(&core0_res, 0, DURATION_S);
    double dur0 = (core0_res.end_us - core0_res.start_us) / 1e6;
    double flops0 = (double)core0_res.ops / dur0;
    printf("\n=== Single-core result ===\n");
    printf("ops=%llu time=%.6fs flops=%.1f checksum=%g\n",
           (unsigned long long)core0_res.ops, dur0, flops0, core0_res.checksum);
#endif

    printf("\nDone.\n");
    return 0;
}

