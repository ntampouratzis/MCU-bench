/*
 *
 * Single-source FLOPS benchmark for Pico SDK (Pico / Pico 2) and ESP-IDF (ESP32).
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>


/* -------------------- Target selection & SDK headers -------------------- */
#define TARGET_IDF 1
//#define TARGET_PICO 1

#if defined(TARGET_PICO)

  #include "pico/stdlib.h"
  #include "pico/time.h"
  #include "pico/multicore.h"
  #include "hardware/clocks.h"

#elif defined(TARGET_IDF)

  #include "freertos/FreeRTOS.h"
  #include "freertos/task.h"
  #include "freertos/semphr.h"
  #include "esp_timer.h"
  #include "esp_clk_tree.h"
  #include "soc/clk_tree_defs.h"

#else
  #error "Define exactly one of: TARGET_PICO or TARGET_IDF"
#endif

/* ----------------------------- Configuration ---------------------------- */

#ifndef FREQ
#define FREQ 240000 /* kHz, used only on Pico (RP2040/RP2350) */
#endif

/* Choose exactly one operation (OP_ADD, OP_SUB, OP_MUL, OP_DIV) */
#define OP_ADD

#ifndef USE_MULTICORE
#define USE_MULTICORE 1
#endif

#ifndef DURATION_S
#define DURATION_S 2.0f
#endif
#ifndef UNROLL
#define UNROLL 8
#endif
#ifndef CHUNK
#define CHUNK 128
#endif

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

/* ------------------------- Platform abstraction ------------------------- */

static inline void platform_sleep_ms(uint32_t ms) {
#if defined(TARGET_PICO)
    sleep_ms(ms);
#elif defined(TARGET_IDF)
    vTaskDelay(pdMS_TO_TICKS(ms));
#endif
}

static inline uint64_t platform_time_us_64(void) {
#if defined(TARGET_PICO)
    return time_us_64();
#elif defined(TARGET_IDF)
    return (uint64_t)esp_timer_get_time(); /* microseconds since boot */
#endif
}

static inline void platform_init_io_and_clock(void) {
#if defined(TARGET_PICO)
    stdio_init_all();
    set_sys_clock_khz(FREQ, true); /* honor FREQ on Pico */
#elif defined(TARGET_IDF)
    /* Nothing mandatory; serial is ready. We simply report current CPU freq. */
#endif
}

static inline uint32_t platform_cpu_hz(void) {
#if defined(TARGET_PICO)
    return clock_get_hz(clk_sys);
#elif defined(TARGET_IDF)

	uint32_t hz = 0;
    // Public API: get CPU clock
    esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU,
                      ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED, &hz);

    return hz;
#endif
}

/* ------------------------------- Benchmark ------------------------------ */

typedef struct {
    uint64_t ops;
    uint64_t start_us;
    uint64_t end_us;
    float checksum;
} bench_result_t;

/* Shared result for the "other" core in multicore mode. */
static volatile bench_result_t multicore_result;

#if UNROLL != 8
#warning "This file is tuned for UNROLL==8; update DO_UNROLL if you change UNROLL."
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

    /* Operands chosen to be safe/good for each op (avoid div-by-zero). */
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
    const float y = 1.2345f;
#endif

    uint64_t local_ops = 0;

    /* Small warmup */
    for (int i = 0; i < 16; ++i) {
        DO_UNROLL(x, y);
        local_ops += UNROLL;
    }

    uint64_t start = platform_time_us_64();
    while (true) {
        for (int c = 0; c < CHUNK; ++c) {
            DO_UNROLL(x, y);
            local_ops += UNROLL;
        }
        uint64_t now = platform_time_us_64();
        if ((now - start) >= duration_us) break;
    }
    uint64_t end = platform_time_us_64();

    res->ops = local_ops;
    res->start_us = start;
    res->end_us = end;
#if defined(OP_ADD) || defined(OP_SUB)
    res->checksum = x + y;
#else
    res->checksum = x * y;
#endif
}

/* ----------------------- Multicore launch & sync ------------------------ */

#if defined(TARGET_PICO)

/* Pico: use SDK multicore FIFO the same way as the original. */
static void core1_entry_pico(void) {
    multicore_fifo_push_blocking(0xC0FEBABE);      /* ready */
    (void)multicore_fifo_pop_blocking();           /* wait START */
    bench_worker((bench_result_t *)&multicore_result, 1, DURATION_S);
    multicore_fifo_push_blocking(0xBEEFDEAD);      /* done */
}

#elif defined(TARGET_IDF)

static SemaphoreHandle_t start_sem; /* signals start to core1 task          */
static SemaphoreHandle_t done_sem;  /* signals completion from core1 to main */

static void core1_task_idf(void *arg) {
    /* Signal readiness by simply waiting on start semaphore. */
    xSemaphoreTake(start_sem, portMAX_DELAY);              /* wait START */
    bench_worker((bench_result_t *)&multicore_result, 1, DURATION_S);
    xSemaphoreGive(done_sem);                              /* done */
    vTaskDelete(NULL);
}
#endif

/* --------------------------------- Main --------------------------------- */

#if defined(TARGET_PICO)

int main(void) {
    platform_init_io_and_clock();
    platform_sleep_ms(20000); /* keep: time for UART console to attach (Pico) */

    printf("Initializing..\n");
    printf("Pico 2 FLOPS benchmark (C)\n");
    printf("Operation: %s\n", OP_NAME);
#if USE_MULTICORE
    printf("Mode: multi-core (2 cores)\n");
#else
    printf("Mode: single-core\n");
#endif
    printf("Duration: %.2f s, UNROLL=%d, CHUNK=%d\n", (double)DURATION_S, UNROLL, CHUNK);
    printf("System clock: %u Hz\n", platform_cpu_hz());

#if USE_MULTICORE
    multicore_launch_core1(core1_entry_pico);

    /* Wait for core1 readiness */
    uint32_t tag = multicore_fifo_pop_blocking();
    if (tag != 0xC0FEBABE) {
        printf("Warning: unexpected readiness tag: 0x%08x\n", tag);
    }

    /* Start core1 and run core0 worker */
    multicore_fifo_push_blocking(0xF00D);

    bench_result_t core0_res = {0};
    bench_worker(&core0_res, 0, DURATION_S);

    printf("Initialized with frequency: %u MHz\n", platform_cpu_hz() / 1000000);

    /* Wait for core1 to finish */
    (void)multicore_fifo_pop_blocking();

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
    platform_sleep_ms(1000);
    printf("Aggregated: total_ops=%llu wall_time=%.6fs aggregated_FLOPS=%.1f\n",
           (unsigned long long)total_ops, wall_time, aggregated_flops);
    platform_sleep_ms(1000);
#else
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

#elif defined(TARGET_IDF)

/* ESP-IDF entry point */
void app_main(void) {
    platform_init_io_and_clock();

    printf("Initializing..\n");
    printf("ESP32 FLOPS benchmark (C)\n");
    printf("Operation: %s\n", OP_NAME);
#if USE_MULTICORE
    printf("Mode: multi-core (2 cores)\n");
#else
    printf("Mode: single-core\n");
#endif
    printf("Duration: %.2f s, UNROLL=%d, CHUNK=%d\n", (double)DURATION_S, UNROLL, CHUNK);
    printf("System clock: %u Hz\n", (unsigned) platform_cpu_hz());

#if USE_MULTICORE
    /* Create sync primitives */
    start_sem = xSemaphoreCreateBinary();
    done_sem  = xSemaphoreCreateBinary();

    /* Launch core1 task pinned to core 1 */
    xTaskCreatePinnedToCore(core1_task_idf, "bench_core1", 4096, NULL, 5, NULL, 1);

    /* Start core1, then run core0 worker from this task (pinned by IDF to core 0) */
    xSemaphoreGive(start_sem);

    bench_result_t core0_res = {0};
    bench_worker(&core0_res, 0, DURATION_S);

    /* Wait for core1 completion */
    xSemaphoreTake(done_sem, portMAX_DELAY);

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
    platform_sleep_ms(1000);
    printf("Aggregated: total_ops=%llu wall_time=%.6fs aggregated_FLOPS=%.1f\n",
           (unsigned long long)total_ops, wall_time, aggregated_flops);
    platform_sleep_ms(1000);
#else
    bench_result_t core0_res = {0};
    bench_worker(&core0_res, 0, DURATION_S);
    double dur0 = (core0_res.end_us - core0_res.start_us) / 1e6;
    double flops0 = (double)core0_res.ops / dur0;
    printf("\n=== Single-core result ===\n");
    printf("ops=%llu time=%.6fs flops=%.1f checksum=%g\n",
           (unsigned long long)core0_res.ops, dur0, flops0, core0_res.checksum);
#endif

    printf("\nDone.\n");
    fflush(stdout);
}
#endif
