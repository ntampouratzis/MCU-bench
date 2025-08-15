/*
Unified Pico SDK + ESP-IDF benchmark (AXPY / MATMUL / SPMV)
------------------------------------------------------------
Single-file source that builds for **either**
  - Pico SDK, or
  - ESP-IDF

Common kernels and runners are defined **once**; only a thin platform layer
(timing, multicore, init, sleep, entry point) is switched via preprocessor.

Select target:
  - **ESP-IDF**: define PORT_ESP32S3 
  - **Pico SDK**: define PORT_PICO

Configuration switches (top of file):
  - ALGO: AXPY, MATMUL, or SPMV (default MATMUL)
  - MODE_MULTI: 1=use two cores, 0=single-core
  - Problem sizes: AXPY_N, MAT_N, SPMV_N, SPMV_NNZ_PER_ROW

Notes:
  - ESP32-S3 uses FreeRTOS task + queues for the second core; timing via esp_timer.
  - Pico uses multicore FIFO + time_us_64(); can optionally set system clock.
  - For fairness, both use microsecond timers and the same partitioning logic.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

// ---------- Target  selection ----------
#define PORT_ESP32S3 1
//#define PORT_PICO 1

// ---------- Data Type  selection ----------
typedef float input_type;
//typedef int input_type;

#if !defined(PORT_ESP32S3) && !defined(PORT_PICO)
  #error "Select target: build under ESP-IDF (ESP_PLATFORM) or compile with -DPORT_PICO for Pico SDK."
#endif

// ---------- Platform headers ----------
#ifdef PORT_PICO
  #include "pico/stdlib.h"
  #include "pico/multicore.h"
  #include "hardware/clocks.h"
  #include "hardware/sync.h"
#endif

#ifdef PORT_ESP32S3
  #include "freertos/FreeRTOS.h"
  #include "freertos/task.h"
  #include "freertos/queue.h"
  #include "esp_timer.h"
  #include "esp_clk_tree.h"
  #include "soc/clk_tree_defs.h"

  //#include "esp_clk.h"
  #include "esp_pm.h"
  #include "esp_system.h"
  #include "esp_log.h"
#endif

// ---------- User config ----------
#define AXPY   0
#define MATMUL 1
#define SPMV   2

#ifndef ALGO
  #define ALGO MATMUL
#endif

#ifndef MODE_MULTI
  #define MODE_MULTI 1  // 1=use both cores, 0=single-core
#endif

#ifndef NUM_THREADS
  #define NUM_THREADS 2 // Only 1 or 2 make sense on these MCUs
#endif

// problem sizes (adjust for RAM)
#define AXPY_N 20000u
#define MAT_N  64u
#define SPMV_N 2000u
#define SPMV_NNZ_PER_ROW 8u

#define WARMUP 1
#define RUNS   3
#define AXPY_A 1.2345f

#ifdef PORT_PICO
  #ifndef PICO_SYS_CLOCK_KHZ
    #define PICO_SYS_CLOCK_KHZ 240000  // change the frequency
  #endif
#endif

// ---------- Common state ----------
enum { CMD_READY=0xA0, CMD_DONE=0xA1, CMD_RUN_AXPY=0xA2, CMD_RUN_MATMUL=0xA3, CMD_RUN_SPMV=0xA4, CMD_EXIT=0xFF };

typedef struct { uint32_t cmd, s, e; } core_cmd_t;

static input_type *g_ax_x=NULL, *g_ax_y=NULL;
static input_type *g_mat_A=NULL, *g_mat_B=NULL, *g_mat_C=NULL;
static input_type *g_sp_vals=NULL; static uint32_t *g_sp_cols=NULL; static uint32_t *g_sp_row_ptr=NULL;
static input_type *g_sp_x=NULL, *g_sp_y=NULL; // vectors used by both cores for SPMV

// ---------- Small utilities ----------
static inline void split_range(uint32_t length, uint32_t parts, uint32_t idx, uint32_t *start, uint32_t *end) {
    uint32_t base = length / parts;
    uint32_t rem  = length % parts;
    *start = idx * base + (idx < rem ? idx : rem);
    *end   = *start + base + (idx < rem ? 1U : 0U);
}

static inline uint64_t now_us(void) {
#ifdef PORT_PICO
    return (uint64_t)time_us_64();
#else
    return (uint64_t)esp_timer_get_time();
#endif
}

// ---------- Data generators (shared) ----------
static input_type *make_vector(uint32_t n, int pattern) {
    input_type *v = (input_type*)malloc(sizeof(input_type)*(size_t)n);
    if (!v) { printf("OOM: vector n=%u\n", (unsigned int)n); return NULL; }
    for (uint32_t i=0;i<n;++i) v[i] = (pattern==1) ? (input_type)(i%1000)/1000.0f : sinf((input_type)i);
    return v;
}
static input_type *make_matrix_flat(uint32_t n) {
    size_t nelem = (size_t)n*(size_t)n;
    input_type *M = (input_type*)malloc(sizeof(input_type)*nelem);
    if (!M) { printf("OOM: matrix n=%u\n", (unsigned int) n); return NULL; }
    for (uint32_t i=0;i<n;++i) {
        size_t base = (size_t)i*n;
        for (uint32_t j=0;j<n;++j) M[base+j] = (input_type)(((i*31u)+j)%100u)/100.0f;
    }
    return M;
}
static bool make_sparse_csr(uint32_t nrows, uint32_t ncols, uint32_t nnz_per_row,
                            input_type **vals_out, uint32_t **cols_out, uint32_t **rowptr_out) {
    uint32_t nnz = nrows*nnz_per_row;
    input_type *vals = (input_type*)malloc(sizeof(input_type)*(size_t)nnz);
    uint32_t *cols = (uint32_t*)malloc(sizeof(uint32_t)*(size_t)nnz);
    uint32_t *rowptr = (uint32_t*)malloc(sizeof(uint32_t)*(size_t)(nrows+1));
    if (!vals||!cols||!rowptr) { free(vals); free(cols); free(rowptr); return false; }
    uint32_t p=0; rowptr[0]=0;
    for (uint32_t i=0;i<nrows;++i) {
        for (uint32_t k=0;k<nnz_per_row;++k) {
            uint32_t col=(i*nnz_per_row+k)%ncols;
            vals[p]=(input_type)(((i*37u)+k)%100u)/100.0f;
            cols[p]=col; ++p;
        }
        rowptr[i+1]=p;
    }
    *vals_out=vals; *cols_out=cols; *rowptr_out=rowptr; return true;
}

// ---------- Kernels (shared) ----------
static void axpy_range(uint32_t start, uint32_t end, input_type a, input_type *x, input_type *y) {
    for (uint32_t i=start;i<end;++i) y[i] = y[i] + a * x[i];
}
static void matmul_range(uint32_t start_row, uint32_t end_row, uint32_t N, input_type *A, input_type *B, input_type *C) {
    for (uint32_t i=start_row;i<end_row;++i) {
        uint32_t arow=i*N, crow=i*N;
        for (uint32_t j=0;j<N;++j) {
            input_type s=0.0f; for (uint32_t k=0;k<N;++k) s += A[arow+k]*B[k*N+j];
            C[crow+j]=s;
        }
    }
}
static void spmv_range(uint32_t start_row, uint32_t end_row, uint32_t N,
                       input_type *vals, uint32_t *cols, uint32_t *rowptr, input_type *x, input_type *y) {
    (void)N; // not needed for CSR iteration
    for (uint32_t i=start_row;i<end_row;++i) {
        input_type s=0.0f; uint32_t r0=rowptr[i], r1=rowptr[i+1];
        for (uint32_t p=r0;p<r1;++p) s += vals[p]*x[cols[p]];
        y[i]=s;
    }
}

// ---------- FLOP counters (shared) ----------
static uint64_t flops_axpy(uint32_t n){ return 2ull*n; }
static uint64_t flops_matmul(uint32_t n){ return 2ull*n*n*n - 1ull*n*n; }
static uint64_t flops_spmv(uint32_t nnz){ return 2ull*nnz; }

// =====================================================
// Platform layer (init/teardown, timing, multicore glue)
// =====================================================
#ifdef PORT_PICO

static inline void platform_sleep_ms(uint32_t ms){ sleep_ms(ms); }
static inline void platform_print_cpu_freq(void){ printf("CPU frequency: %u MHz\n", (unsigned)(clock_get_hz(clk_sys)/1000000u)); }

static void worker_core1(void){
    multicore_fifo_push_blocking(CMD_READY);
    for(;;){
        uint32_t cmd = multicore_fifo_pop_blocking();
        if (cmd==CMD_EXIT){ multicore_fifo_push_blocking(CMD_DONE); break; }
        uint32_t s = multicore_fifo_pop_blocking();
        uint32_t e = multicore_fifo_pop_blocking();
        switch(cmd){
            case CMD_RUN_AXPY:   axpy_range(s,e,AXPY_A,g_ax_x,g_ax_y); break;
            case CMD_RUN_MATMUL: matmul_range(s,e,MAT_N,g_mat_A,g_mat_B,g_mat_C); break;
            case CMD_RUN_SPMV:   spmv_range(s,e,SPMV_N,g_sp_vals,g_sp_cols,g_sp_row_ptr,g_sp_x,g_sp_y); break;
            default: break;
        }
        multicore_fifo_push_blocking(CMD_DONE);
    }
}

static void platform_init(void){
    stdio_init_all();
#ifdef PICO_SYS_CLOCK_KHZ
    set_sys_clock_khz(PICO_SYS_CLOCK_KHZ, true);
#endif
    platform_sleep_ms(20000);
    multicore_launch_core1(worker_core1);
    // wait ready
    (void)multicore_fifo_pop_blocking();
    printf("core1 ready\n");
}
static void platform_start_secondary(uint32_t cmd, uint32_t s, uint32_t e){
    multicore_fifo_push_blocking(cmd);
    multicore_fifo_push_blocking(s);
    multicore_fifo_push_blocking(e);
}
static void platform_wait_secondary_done(void){ (void)multicore_fifo_pop_blocking(); }
static void platform_stop_secondary(void){
    multicore_fifo_push_blocking(CMD_EXIT);
    (void)multicore_fifo_pop_blocking(); // done
}

#endif // PORT_PICO

#ifdef PORT_ESP32S3
static QueueHandle_t s_cmd_q=NULL; static QueueHandle_t s_evt_q=NULL; static TaskHandle_t s_worker=NULL;
static inline void platform_sleep_ms(uint32_t ms){ vTaskDelay(pdMS_TO_TICKS(ms)); }
static inline void platform_print_cpu_freq(void){
        uint32_t hz = 0;
        // Public API: get CPU clock
        esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU,
                                     ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED,
                                     &hz);
        printf("CPU frequency: %u MHz\n", (unsigned)(hz / 1000000));
    }

static void worker_task(void *arg){
    (void)arg; uint32_t ready=CMD_READY; xQueueSend(s_evt_q,&ready,portMAX_DELAY);
    core_cmd_t c;
    for(;;){
        if (xQueueReceive(s_cmd_q,&c,portMAX_DELAY)==pdTRUE){
            if (c.cmd==CMD_EXIT){ uint32_t done=CMD_DONE; xQueueSend(s_evt_q,&done,portMAX_DELAY); vTaskDelete(NULL);} // no return
            switch(c.cmd){
                case CMD_RUN_AXPY:   axpy_range(c.s,c.e,AXPY_A,g_ax_x,g_ax_y); break;
                case CMD_RUN_MATMUL: matmul_range(c.s,c.e,MAT_N,g_mat_A,g_mat_B,g_mat_C); break;
                case CMD_RUN_SPMV:   spmv_range(c.s,c.e,SPMV_N,g_sp_vals,g_sp_cols,g_sp_row_ptr,g_sp_x,g_sp_y); break;
                default: break;
            }
            uint32_t done=CMD_DONE; xQueueSend(s_evt_q,&done,portMAX_DELAY);
        }
    }
}

static void platform_init(void){
    platform_sleep_ms(1500);
    s_cmd_q = xQueueCreate(4,sizeof(core_cmd_t));
    s_evt_q = xQueueCreate(4,sizeof(uint32_t));
    xTaskCreatePinnedToCore(worker_task, "core1_worker", 4096, NULL, 5, &s_worker, 1);
    uint32_t evt; xQueueReceive(s_evt_q,&evt,portMAX_DELAY); (void)evt; printf("core1 ready\n");
}
static void platform_start_secondary(uint32_t cmd, uint32_t s, uint32_t e){ core_cmd_t c={cmd,s,e}; xQueueSend(s_cmd_q,&c,portMAX_DELAY); }
static void platform_wait_secondary_done(void){ uint32_t evt; xQueueReceive(s_evt_q,&evt,portMAX_DELAY); (void)evt; }
static void platform_stop_secondary(void){ core_cmd_t c={CMD_EXIT,0,0}; xQueueSend(s_cmd_q,&c,portMAX_DELAY); uint32_t evt; xQueueReceive(s_evt_q,&evt,portMAX_DELAY); (void)evt; }
#endif // PORT_ESP32S3

// =====================================================
// Benchmark runners (shared)
// =====================================================
static void run_axpy(void){
    printf("AXPY: n=%u mode=%s threads=%d\n", AXPY_N, MODE_MULTI?"MULTI":"SINGLE", NUM_THREADS);
    input_type *x=make_vector(AXPY_N,1), *y=make_vector(AXPY_N,2); if(!x||!y){ free(x); free(y); return; }
    g_ax_x=x; g_ax_y=y; uint64_t flops=flops_axpy(AXPY_N);

    uint32_t th = (MODE_MULTI? (NUM_THREADS>2?2:NUM_THREADS):1); if (th<1) th=1;

    for(int w=0; w<WARMUP; ++w){
        if (th==2){ uint32_t s0,e0,s1,e1; split_range(AXPY_N,2,0,&s0,&e0); split_range(AXPY_N,2,1,&s1,&e1);
            platform_start_secondary(CMD_RUN_AXPY,s1,e1); axpy_range(s0,e0,AXPY_A,x,y); platform_wait_secondary_done();
        } else { axpy_range(0,AXPY_N,AXPY_A,x,y); }
    }

    double times[RUNS];
    for(int r=0;r<RUNS;++r){
        uint64_t t0=now_us();
        if (th==2){ uint32_t s0,e0,s1,e1; split_range(AXPY_N,2,0,&s0,&e0); split_range(AXPY_N,2,1,&s1,&e1);
            platform_start_secondary(CMD_RUN_AXPY,s1,e1); axpy_range(s0,e0,AXPY_A,x,y); platform_wait_secondary_done();
        } else { axpy_range(0,AXPY_N,AXPY_A,x,y); }
        uint64_t t1=now_us(); double el=(double)(t1-t0)/1e6; times[r]=el; printf("  run %d elapsed=%.6f s\n", r, el);
    }
    double avg=0.0; for(int r=0;r<RUNS;++r) avg+=times[r]; avg/=RUNS; double mflops=((double)flops/avg)/1e6;
    printf("AXPY result: ops=%llu avg_time=%.6f s mFLOPS=%.3f\n", (unsigned long long)flops, avg, mflops);
    free(x); free(y);
}

static void run_matmul(void){
    printf("MATMUL: N=%u mode=%s threads=%d\n", MAT_N, MODE_MULTI?"MULTI":"SINGLE", NUM_THREADS);
    input_type *A=make_matrix_flat(MAT_N), *B=make_matrix_flat(MAT_N), *C=(input_type*)malloc(sizeof(input_type)*(size_t)MAT_N*(size_t)MAT_N);
    if(!A||!B||!C){ free(A); free(B); free(C); return; }
    memset(C,0,sizeof(input_type)*(size_t)MAT_N*(size_t)MAT_N); g_mat_A=A; g_mat_B=B; g_mat_C=C;
    uint64_t flops=flops_matmul(MAT_N); uint32_t th=(MODE_MULTI?(NUM_THREADS>2?2:NUM_THREADS):1); if(th<1) th=1;

    for(int w=0; w<WARMUP; ++w){
        if(th==2){ uint32_t s0,e0,s1,e1; split_range(MAT_N,2,0,&s0,&e0); split_range(MAT_N,2,1,&s1,&e1);
            platform_start_secondary(CMD_RUN_MATMUL,s1,e1); matmul_range(s0,e0,MAT_N,A,B,C); platform_wait_secondary_done();
        } else { matmul_range(0,MAT_N,MAT_N,A,B,C); }
    }

    double times[RUNS];
    for(int r=0;r<RUNS;++r){
        uint64_t t0=now_us();
        if(th==2){ uint32_t s0,e0,s1,e1; split_range(MAT_N,2,0,&s0,&e0); split_range(MAT_N,2,1,&s1,&e1);
            platform_start_secondary(CMD_RUN_MATMUL,s1,e1); matmul_range(s0,e0,MAT_N,A,B,C); platform_wait_secondary_done();
        } else { matmul_range(0,MAT_N,MAT_N,A,B,C); }
        uint64_t t1=now_us(); double el=(double)(t1-t0)/1e6; times[r]=el; printf("  run %d elapsed=%.6f s\n", r, el);
    }
    double avg=0.0; for(int r=0;r<RUNS;++r) avg+=times[r]; avg/=RUNS; double mflops=((double)flops/avg)/1e6;
    printf("MATMUL result: ops=%llu avg_time=%.6f s mFLOPS=%.3f\n", (unsigned long long)flops, avg, mflops);
    free(A); free(B); free(C);
}

static void run_spmv(void){
    printf("SPMV: N=%u nnz/row=%u mode=%s threads=%d\n", SPMV_N, SPMV_NNZ_PER_ROW, MODE_MULTI?"MULTI":"SINGLE", NUM_THREADS);
    input_type *vals; uint32_t *cols; uint32_t *rowptr; if(!make_sparse_csr(SPMV_N,SPMV_N,SPMV_NNZ_PER_ROW,&vals,&cols,&rowptr)){ printf("OOM building sparse matrix\n"); return; }
    input_type *x=make_vector(SPMV_N,1); input_type *y=(input_type*)malloc(sizeof(input_type)*(size_t)SPMV_N); if(!x||!y){ free(vals); free(cols); free(rowptr); free(x); free(y); return; }
    memset(y,0,sizeof(input_type)*(size_t)SPMV_N);
    g_sp_vals=vals; g_sp_cols=cols; g_sp_row_ptr=rowptr; g_sp_x=x; g_sp_y=y;

    uint32_t nnz=SPMV_N*SPMV_NNZ_PER_ROW; uint64_t flops=flops_spmv(nnz); uint32_t th=(MODE_MULTI?(NUM_THREADS>2?2:NUM_THREADS):1); if(th<1) th=1;

    for(int w=0; w<WARMUP; ++w){
        if(th==2){ uint32_t s0,e0,s1,e1; split_range(SPMV_N,2,0,&s0,&e0); split_range(SPMV_N,2,1,&s1,&e1);
            platform_start_secondary(CMD_RUN_SPMV,s1,e1); spmv_range(s0,e0,SPMV_N,vals,cols,rowptr,x,y); platform_wait_secondary_done();
        } else { spmv_range(0,SPMV_N,SPMV_N,vals,cols,rowptr,x,y); }
    }

    double times[RUNS];
    for(int r=0;r<RUNS;++r){
        uint64_t t0=now_us();
        if(th==2){ uint32_t s0,e0,s1,e1; split_range(SPMV_N,2,0,&s0,&e0); split_range(SPMV_N,2,1,&s1,&e1);
            platform_start_secondary(CMD_RUN_SPMV,s1,e1); spmv_range(s0,e0,SPMV_N,vals,cols,rowptr,x,y); platform_wait_secondary_done();
        } else { spmv_range(0,SPMV_N,SPMV_N,vals,cols,rowptr,x,y); }
        uint64_t t1=now_us(); double el=(double)(t1-t0)/1e6; times[r]=el; printf("  run %d elapsed=%.6f s\n", r, el);
    }
    double avg=0.0; for(int r=0;r<RUNS;++r) avg+=times[r]; avg/=RUNS; double mflops=((double)flops/avg)/1e6;
    printf("SPMV result: nnz=%u ops=%llu avg_time=%.6f s mFLOPS=%.3f\n", (unsigned)nnz, (unsigned long long)flops, avg, mflops);

    free(vals); free(cols); free(rowptr); free(x); free(y);
}

// =====================================================
// Common top-level entry
// =====================================================
static void run_benchmark(void){
    printf("Initializing..\n");
    platform_init();
    platform_print_cpu_freq();
#if ALGO == AXPY
    run_axpy();
#elif ALGO == MATMUL
    run_matmul();
#elif ALGO == SPMV
    run_spmv();
#else
    printf("Unknown ALGO selection\n");
#endif
    platform_stop_secondary();
    printf("benchmark finished\n");
}

// Platform entry points
#ifdef PORT_PICO
int main(void){ run_benchmark(); return 0; }
#endif
#ifdef PORT_ESP32S3
void app_main(void){ run_benchmark(); }
#endif
