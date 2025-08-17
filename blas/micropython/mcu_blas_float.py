"""
MCU MicroPython FLOPS benchmark (dtype-configurable)
----------------------------------------------------
Adds DTYPE selection with minimal code changes.

DTYPE: "float32" | "float64" | "int32"
 - float32 -> array('f')
 - float64 -> array('d') if supported, else warns + falls back to 'f'
 - int32   -> prefers 4-byte signed int typecode ('i' or 'l'), else warns + picks a 4-byte unsigned

Everything else (kernels, threading, sizes, FLOPS math) remains the same.
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
machine.freq(240_000_000) #Declare the frequency (in Hz)

# >>>>>>>>>>>>>>>>>>> NEW: dtype selector <<<<<<<<<<<<<<<<<<
DTYPE = "float32"   # one of: "float32", "float64", "int32"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

ALGO = "MATMUL"     # one of: "AXPY", "MATMUL", "SPMV"
MODE = "MULTI"      # "SINGLE" or "MULTI"
NUM_THREADS = 2     # logically the number of worker *partitions* you want (1 or 2)

# Size parameters (tune to your available RAM)
AXPY_N = 20000          # vector length for AXPY
MAT_N  = 64             # matrix dimension N (matrix is N x N)
SPMV_N = 1000           # rows/cols for SPMV
SPMV_NNZ_PER_ROW = 8    # nonzeros per row for SPMV

WARMUP = 1
RUNS = 3
AXPY_A = 1.2345         # will be cast to dtype (see _cast_scalar)

# ---------------- dtype helpers (NEW) ----------------

def _detect_int32_typecode():
    """Find a 4-byte integer array typecode, preferring signed."""
    candidates = ('i', 'l', 'I', 'L')
    for tc in candidates:
        try:
            a = array(tc, (0,))
            if a.itemsize == 4:
                # Prefer signed if available
                if tc in ('i', 'l'):
                    return tc, True
                return tc, False
        except Exception:
            pass
    # Fallback: try 2-byte signed 'h' with a warning (may overflow in larger sizes)
    print("Warning: no 4-byte int array type found; using 'h' (16-bit). Results may overflow.")
    return 'h', True

def _init_dtype(dt):
    """Return (VAL_TC, ZERO, ONE, OPS_LABEL, IS_INT) based on DTYPE."""
    dt = (dt or "").lower()
    if dt == "float64":
        try:
            array('d', (0.0,))
            return 'd', 0.0, 1.0, "FLOPS", False
        except Exception:
            print("Warning: float64 ('d') arrays not supported on this build; falling back to float32 ('f').")
            return 'f', 0.0, 1.0, "FLOPS", False
    elif dt == "int32":
        tc, _is_signed = _detect_int32_typecode()
        # For int ops we still report OPS, not FLOPS
        return tc, 0, 1, "OPS", True
    # default: float32
    return 'f', 0.0, 1.0, "FLOPS", False

# Resolve dtype once
_VAL_TC, _ZERO, _ONE, _OPS_NAME, _IS_INT = _init_dtype(DTYPE)

def _cast_scalar(x):
    """Cast Python scalar to the current dtype (just int or float)."""
    return int(x) if _IS_INT else float(x)

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
    """
    For floats:
      pattern 1: ramp [0..1)
      pattern 2: sin(i)
    For int32:
      small ints to avoid overflow in MATMUL/AXPY accumulations.
    """
    if _IS_INT:
        # Keep values small (0..127) to minimize overflow risk
        a = array(_VAL_TC, (0 for _ in range(n)))
        for i in range(n):
            if pattern == 1:
                a[i] = (i % 128)
            else:
                a[i] = ((i * 7) + 3) % 128
        return a
    else:
        a = array(_VAL_TC, (_ZERO for _ in range(n)))
        for i in range(n):
            if pattern == 1:
                a[i] = float(i % 1000) / 1000.0
            else:
                a[i] = math.sin(i)
        return a

def make_matrix_flat(n):
    """
    N x N flattened matrix.
    For int32 we use small ints in [0..99] to keep sums in 32-bit range.
    """
    M = array(_VAL_TC, (_ZERO for _ in range(n * n)))
    if _IS_INT:
        for i in range(n):
            base = i * n
            for j in range(n):
                M[base + j] = ((i * 31) + j) % 100
    else:
        for i in range(n):
            base = i * n
            for j in range(n):
                M[base + j] = float(((i * 31) + j) % 100) / 100.0
    return M

def make_sparse_csr(nrows, ncols, nnz_per_row):
    # value array follows selected dtype
    vals = array(_VAL_TC)
    # prefer 32-bit unsigned for column indices; fall back to 'H'
    try:
        cols = array('I')
    except Exception:
        cols = array('H')
    row_ptr = [0]
    for i in range(nrows):
        for k in range(nnz_per_row):
            col = (i * nnz_per_row + k) % ncols
            if _IS_INT:
                vals.append(((i * 37) + k) % 100)
            else:
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
            s = _ZERO
            for k in range(N):
                s += A[arow + k] * B[k * N + j]
            C[crow + j] = s
    if done_lock:
        done_lock.release()

def spmv_worker(start_row, end_row, N, vals, cols, row_ptr, x, y, done_lock=None):
    for i in range(start_row, end_row):
        s = _ZERO
        start = row_ptr[i]
        stop = row_ptr[i + 1]
        for idx in range(start, stop):
            s += vals[idx] * x[cols[idx]]
        y[i] = s
    if done_lock:
        done_lock.release()

# ---------------- flop counters (unchanged logic) ----------------

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
       `worker` on the main thread for `main_range`. Returns True if secondary thread
       was started, False if fallback used.
    """
    s_main, e_main = main_range
    s_sec, e_sec = secondary_range
    lock = _thread.allocate_lock()
    lock.acquire()
    try:
        _thread.start_new_thread(worker, (s_sec, e_sec) + worker_args + (lock,))
    except OSError as exc:
        print("Warning: failed to start secondary thread ({}). Falling back to single-core.".format(exc))
        try:
            lock.release()
        except Exception:
            pass
        worker(s_main, e_main, *worker_args, None)
        return False
    else:
        worker(s_main, e_main, *worker_args, None)
        lock.acquire()
        return True

# ---------------- benchmark runners ----------------

def run_axpy(n=AXPY_N, mode=MODE, threads=NUM_THREADS, runs=RUNS, warmup=WARMUP):
    print("AXPY: n=", n, "mode=", mode, "threads=", threads, "dtype=", DTYPE)
    x = make_vector(n, pattern=1)
    y = make_vector(n, pattern=2)
    a = _cast_scalar(AXPY_A if not _IS_INT else 3)  # use a small int for int32
    flops_per_run = flops_axpy(n)

    effective_threads = 1
    if mode == "MULTI" and _THREAD_AVAILABLE:
        effective_threads = min(max(1, threads), 2)

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
    mrate = (flops_per_run / avg) / 1e6
    print("AXPY result: ops=", flops_per_run, "avg_time=%.6f s m%s=%.3f" % (avg, _OPS_NAME, mrate))
    return avg, mrate

def run_matmul(N=MAT_N, mode=MODE, threads=NUM_THREADS, runs=RUNS, warmup=WARMUP):
    print("MATMUL: N=", N, "mode=", mode, "threads=", threads, "dtype=", DTYPE)
    A = make_matrix_flat(N)
    B = make_matrix_flat(N)
    C = array(_VAL_TC, (_ZERO for _ in range(N * N)))
    flops_per_run = flops_matmul(N)

    effective_threads = 1
    if mode == "MULTI" and _THREAD_AVAILABLE:
        effective_threads = min(max(1, threads), 2)

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
    mrate = (flops_per_run / avg) / 1e6
    print("MATMUL result: ops=", flops_per_run, "avg_time=%.6f s m%s=%.3f" % (avg, _OPS_NAME, mrate))
    return avg, mrate

def run_spmv(N=SPMV_N, nnz_per_row=SPMV_NNZ_PER_ROW, mode=MODE, threads=NUM_THREADS, runs=RUNS, warmup=WARMUP):
    print("SPMV: N=", N, "nnz/row=", nnz_per_row, "mode=", mode, "threads=", threads, "dtype=", DTYPE)
    vals, cols, row_ptr = make_sparse_csr(N, N, nnz_per_row)
    x = make_vector(N, pattern=1)
    y = array(_VAL_TC, (_ZERO for _ in range(N)))
    nnz = len(vals)
    flops_per_run = flops_spmv(nnz)

    effective_threads = 1
    if mode == "MULTI" and _THREAD_AVAILABLE:
        effective_threads = min(max(1, threads), 2)

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
    mrate = (flops_per_run / avg) / 1e6
    print("SPMV result: nnz=", nnz, "ops=", flops_per_run, "avg_time=%.6f s m%s=%.3f" % (avg, _OPS_NAME, mrate))
    return avg, mrate

# ---------------- main runner ----------------

def run_selected():
    print("RP Pico 2 MicroPython FLOPS benchmark (revised)")
    print("_thread available:", _THREAD_AVAILABLE)
    print("DTYPE:", DTYPE, "-> array typecode:", _VAL_TC)
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

