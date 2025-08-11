"""
RP Pico 2 MicroPython FLOPS benchmark (REVISED)
---------------------------------------------
This updated script fixes a threading bug and adds robust fallback logic.

Implemented kernels:
 - AXPY  : y := a*x + y
 - MATMUL: C := A * B (naive triple loop)
 - SPMV  : y := A_sparse * x (CSR-like format)

Key changes in this revision
 - Correct multi-core use: **main thread (core 0) participates** and at most **one** new thread
   is started on core 1. This avoids attempting to start multiple threads on core1 which
   causes `OSError: core1 in use` on the rp2 MicroPython port.
 - Graceful fallback: if starting the secondary thread fails (core1 busy or OSError), the
   benchmark falls back to single-threaded execution and prints a warning.
 - Safer lock handling with try/except to avoid deadlocks/leaked locks.
 - NUM_THREADS is capped to 2 for the rp2 port (one worker on each core).

Usage
 - Edit the CONFIG block below (ALGO, MODE, NUM_THREADS, sizes), upload and run.
 - If your MicroPython build does not have `_thread`, set MODE = "SINGLE".

"""

import time
import math
import gc
from array import array
import machine

# Try to import threading support; if not available we'll fall back to single-core
try:
    import _thread
    _THREAD_AVAILABLE = True
except Exception:
    _THREAD_AVAILABLE = False

# -------------------- CONFIG --------------------
machine.freq(280_000_000) #Declare the frequency (in Hz)
ALGO = "MATMUL"   # one of: "AXPY", "MATMUL", "SPMV"
MODE = "MULTI"    # "SINGLE" or "MULTI"
NUM_THREADS = 2    # logically the number of worker *partitions* you want (1 or 2)

# Size parameters (tune to your available RAM)
AXPY_N = 20000          # vector length for AXPY
MAT_N  = 64             # matrix dimension N (matrix is N x N)
SPMV_N = 1000           # rows/cols for SPMV
SPMV_NNZ_PER_ROW = 8    # nonzeros per row for SPMV

WARMUP = 1
RUNS = 3
AXPY_A = 1.2345

# ---------------- utility functions ----------------

def split_range(length, parts, idx):
    base = length // parts
    rem = length % parts
    start = idx * base + min(idx, rem)
    end = start + base + (1 if idx < rem else 0)
    return start, end


def now_s():
    return time.ticks_diff(time.ticks_us(), 0) / 1_000_000.0


# ---------------- data generators ----------------

def make_vector(n, pattern=1):
    a = array('f', (0.0 for _ in range(n)))
    for i in range(n):
        if pattern == 1:
            a[i] = float(i % 1000) / 1000.0
        else:
            a[i] = math.sin(i)
    return a


def make_matrix_flat(n):
    M = array('f', (0.0 for _ in range(n * n)))
    for i in range(n):
        base = i * n
        for j in range(n):
            M[base + j] = float(((i * 31) + j) % 100) / 100.0
    return M


def make_sparse_csr(nrows, ncols, nnz_per_row):
    vals = array('f')
    # try to use a 32-bit unsigned index if supported; fall back to 'H' if not
    try:
        cols = array('I')
    except Exception:
        cols = array('H')
    row_ptr = [0]
    for i in range(nrows):
        for k in range(nnz_per_row):
            col = (i * nnz_per_row + k) % ncols
            vals.append(float(((i * 37) + k) % 100) / 100.0)
            cols.append(col)
        row_ptr.append(len(vals))
    return vals, cols, row_ptr


# ---------------- kernel implementations ----------------

def axpy_worker(start, end, a, x, y, done_lock=None):
    # y[i] += a * x[i]
    for i in range(start, end):
        y[i] = y[i] + (a * x[i])
    if done_lock:
        done_lock.release()


def matmul_worker(start_row, end_row, N, A, B, C, done_lock=None):
    # C[i,j] = sum_k A[i,k] * B[k,j]
    for i in range(start_row, end_row):
        arow = i * N
        crow = i * N
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += A[arow + k] * B[k * N + j]
            C[crow + j] = s
    if done_lock:
        done_lock.release()


def spmv_worker(start_row, end_row, N, vals, cols, row_ptr, x, y, done_lock=None):
    for i in range(start_row, end_row):
        s = 0.0
        start = row_ptr[i]
        stop = row_ptr[i + 1]
        for idx in range(start, stop):
            s += vals[idx] * x[cols[idx]]
        y[i] = s
    if done_lock:
        done_lock.release()


# ---------------- flop counters ----------------

def flops_axpy(n):
    return 2 * n


def flops_matmul(n):
    # 1 multiply + 1 add per inner k, minus the final add per row
    return 2 * (n ** 3) - (n ** 2)


def flops_spmv(nnz):
    return 2 * nnz


# ---------------- harness helpers ----------------

def _start_secondary_and_run(main_range, secondary_range, worker, worker_args):
    """Start a single secondary thread for `secondary_range` (on core1) and run
       `worker` on the main thread for `main_range`. This function assumes `secondary_range`
       is a tuple (s,e). Returns True if secondary thread was started, False if fallback used.
    """
    s_main, e_main = main_range
    s_sec, e_sec = secondary_range
    lock = None
    # allocate+lock the join-lock only when we will start the secondary thread
    lock = _thread.allocate_lock()
    lock.acquire()
    try:
        # start thread on core1 and pass the lock so secondary will release when done
        _thread.start_new_thread(worker, (s_sec, e_sec) + worker_args + (lock,))
    except OSError as exc:
        # common error: core1 in use
        print("Warning: failed to start secondary thread ({}). Falling back to single-core.".format(exc))
        # cleanup our lock (no secondary will release it)
        try:
            lock.release()
        except Exception:
            pass
        # run full work on main thread
        worker(s_main, e_main, *worker_args, None)
        return False
    else:
        # run main partition on core0
        worker(s_main, e_main, *worker_args, None)
        # wait for secondary to finish
        lock.acquire()
        return True


# ---------------- benchmark runners ----------------

def run_axpy(n=AXPY_N, mode=MODE, threads=NUM_THREADS, runs=RUNS, warmup=WARMUP):
    print("AXPY: n=", n, "mode=", mode, "threads=", threads)
    x = make_vector(n, pattern=1)
    y = make_vector(n, pattern=2)
    a = float(AXPY_A)
    flops_per_run = flops_axpy(n)

    effective_threads = 1
    if mode == "MULTI" and _THREAD_AVAILABLE:
        effective_threads = min(max(1, threads), 2)
    else:
        effective_threads = 1

    # warmup
    for _ in range(warmup):
        if effective_threads == 2:
            s0, e0 = split_range(n, 2, 0)
            s1, e1 = split_range(n, 2, 1)
            _start_secondary_and_run((s0, e0), (s1, e1), axpy_worker, (a, x, y))
        else:
            axpy_worker(0, n, a, x, y, None)

    times = []
    for r in range(runs):
        gc.collect()
        start = time.ticks_us()
        if effective_threads == 2:
            s0, e0 = split_range(n, 2, 0)
            s1, e1 = split_range(n, 2, 1)
            _start_secondary_and_run((s0, e0), (s1, e1), axpy_worker, (a, x, y))
        else:
            axpy_worker(0, n, a, x, y, None)
        end = time.ticks_us()
        elapsed = time.ticks_diff(end, start) / 1_000_000.0
        times.append(elapsed)
        print("  run", r, "elapsed=%.6f s" % elapsed)

    avg = sum(times) / len(times)
    mflops = (flops_per_run / avg) / 1e6
    print("AXPY result: ops=", flops_per_run, "avg_time=%.6f s mFLOPS=%.3f" % (avg, mflops))
    return avg, mflops


def run_matmul(N=MAT_N, mode=MODE, threads=NUM_THREADS, runs=RUNS, warmup=WARMUP):
    print("MATMUL: N=", N, "mode=", mode, "threads=", threads)
    A = make_matrix_flat(N)
    B = make_matrix_flat(N)
    C = array('f', (0.0 for _ in range(N * N)))
    flops_per_run = flops_matmul(N)

    effective_threads = 1
    if mode == "MULTI" and _THREAD_AVAILABLE:
        effective_threads = min(max(1, threads), 2)
    else:
        effective_threads = 1

    # warmup
    for _ in range(warmup):
        if effective_threads == 2:
            s0, e0 = split_range(N, 2, 0)
            s1, e1 = split_range(N, 2, 1)
            _start_secondary_and_run((s0, e0), (s1, e1), matmul_worker, (N, A, B, C))
        else:
            matmul_worker(0, N, N, A, B, C, None)

    times = []
    for r in range(runs):
        gc.collect()
        start = time.ticks_us()
        if effective_threads == 2:
            s0, e0 = split_range(N, 2, 0)
            s1, e1 = split_range(N, 2, 1)
            _start_secondary_and_run((s0, e0), (s1, e1), matmul_worker, (N, A, B, C))
        else:
            matmul_worker(0, N, N, A, B, C, None)
        end = time.ticks_us()
        elapsed = time.ticks_diff(end, start) / 1_000_000.0
        times.append(elapsed)
        print("  run", r, "elapsed=%.6f s" % elapsed)

    avg = sum(times) / len(times)
    mflops = (flops_per_run / avg) / 1e6
    print("MATMUL result: ops=", flops_per_run, "avg_time=%.6f s mFLOPS=%.3f" % (avg, mflops))
    return avg, mflops


def run_spmv(N=SPMV_N, nnz_per_row=SPMV_NNZ_PER_ROW, mode=MODE, threads=NUM_THREADS, runs=RUNS, warmup=WARMUP):
    print("SPMV: N=", N, "nnz/row=", nnz_per_row, "mode=", mode, "threads=", threads)
    vals, cols, row_ptr = make_sparse_csr(N, N, nnz_per_row)
    x = make_vector(N, pattern=1)
    y = array('f', (0.0 for _ in range(N)))
    nnz = len(vals)
    flops_per_run = flops_spmv(nnz)

    effective_threads = 1
    if mode == "MULTI" and _THREAD_AVAILABLE:
        effective_threads = min(max(1, threads), 2)
    else:
        effective_threads = 1

    # warmup
    for _ in range(warmup):
        if effective_threads == 2:
            s0, e0 = split_range(N, 2, 0)
            s1, e1 = split_range(N, 2, 1)
            _start_secondary_and_run((s0, e0), (s1, e1), spmv_worker, (N, vals, cols, row_ptr, x, y))
        else:
            spmv_worker(0, N, N, vals, cols, row_ptr, x, y, None)

    times = []
    for r in range(runs):
        gc.collect()
        start = time.ticks_us()
        if effective_threads == 2:
            s0, e0 = split_range(N, 2, 0)
            s1, e1 = split_range(N, 2, 1)
            _start_secondary_and_run((s0, e0), (s1, e1), spmv_worker, (N, vals, cols, row_ptr, x, y))
        else:
            spmv_worker(0, N, N, vals, cols, row_ptr, x, y, None)
        end = time.ticks_us()
        elapsed = time.ticks_diff(end, start) / 1_000_000.0
        times.append(elapsed)
        print("  run", r, "elapsed=%.6f s" % elapsed)

    avg = sum(times) / len(times)
    mflops = (flops_per_run / avg) / 1e6
    print("SPMV result: nnz=", nnz, "ops=", flops_per_run, "avg_time=%.6f s mFLOPS=%.3f" % (avg, mflops))
    return avg, mflops


# ---------------- main runner ----------------

def run_selected():
    print("RP Pico 2 MicroPython FLOPS benchmark (revised)
")
    print("_thread available:", _THREAD_AVAILABLE)
    if MODE == "MULTI" and not _THREAD_AVAILABLE:
        print("WARNING: MULTI mode requested but _thread is not available on this build. Falling back to SINGLE.")

    if ALGO == "AXPY":
        return run_axpy(AXPY_N, MODE if _THREAD_AVAILABLE else "SINGLE", NUM_THREADS)
    elif ALGO == "MATMUL":
        return run_matmul(MAT_N, MODE if _THREAD_AVAILABLE else "SINGLE", NUM_THREADS)
    elif ALGO == "SPMV":
        return run_spmv(SPMV_N, SPMV_NNZ_PER_ROW, MODE if _THREAD_AVAILABLE else "SINGLE", NUM_THREADS)
    else:
        print("Unknown ALGO. Choose one of: AXPY, MATMUL, SPMV")


if __name__ == '__main__':
    run_selected()
