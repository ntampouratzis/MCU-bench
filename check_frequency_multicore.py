import time, math, gc
from array import array
import machine

# Try threading
try:
    import _thread
    _THREAD_AVAILABLE = True
except Exception:
    _THREAD_AVAILABLE = False

# CONFIG
ALGO = "SPMV"   # AXPY, MATMUL, SPMV
NUM_THREADS = 2
AXPY_N = 20000
MAT_N = 64
SPMV_N = 1000
SPMV_NNZ_PER_ROW = 8
AXPY_A = 1.2345
WARMUP = 1
RUNS = 1  # For correctness check, 1 run is enough

# ---------- Utility ----------
def split_range(length, parts, idx):
    base = length // parts
    rem = length % parts
    start = idx * base + min(idx, rem)
    end = start + base + (1 if idx < rem else 0)
    return start, end

# ---------- Data ----------
def make_vector(n, pattern=1):
    a = array('f', (0.0 for _ in range(n)))
    for i in range(n):
        a[i] = (float(i % 1000) / 1000.0) if pattern == 1 else math.sin(i)
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
    try: cols = array('I')
    except: cols = array('H')
    row_ptr = [0]
    for i in range(nrows):
        for k in range(nnz_per_row):
            col = (i * nnz_per_row + k) % ncols
            vals.append(float(((i * 37) + k) % 100) / 100.0)
            cols.append(col)
        row_ptr.append(len(vals))
    return vals, cols, row_ptr

# ---------- Workers ----------
def axpy_worker(start, end, a, x, y, done_lock=None):
    for i in range(start, end):
        y[i] = y[i] + (a * x[i])
    if done_lock: done_lock.release()

def matmul_worker(start_row, end_row, N, A, B, C, done_lock=None):
    for i in range(start_row, end_row):
        arow = i * N
        crow = i * N
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += A[arow + k] * B[k * N + j]
            C[crow + j] = s
    if done_lock: done_lock.release()

def spmv_worker(start_row, end_row, N, vals, cols, row_ptr, x, y, done_lock=None):
    for i in range(start_row, end_row):
        s = 0.0
        for idx in range(row_ptr[i], row_ptr[i+1]):
            s += vals[idx] * x[cols[idx]]
        y[i] = s
    if done_lock: done_lock.release()

# ---------- Thread helper ----------
def _start_secondary_and_run(main_range, secondary_range, worker, worker_args):
    s_main, e_main = main_range
    s_sec, e_sec = secondary_range
    lock = _thread.allocate_lock()
    lock.acquire()
    try:
        _thread.start_new_thread(worker, (s_sec, e_sec) + worker_args + (lock,))
    except OSError as exc:
        print("Thread fail:", exc, "→ fallback to single core")
        lock.release()
        worker(s_main, e_main, *worker_args, None)
        return False
    else:
        worker(s_main, e_main, *worker_args, None)
        lock.acquire()
        return True

# ---------- Run one mode ----------
def run_algo_single():
    if ALGO == "AXPY":
        x = make_vector(AXPY_N, 1)
        y = make_vector(AXPY_N, 2)
        axpy_worker(0, AXPY_N, AXPY_A, x, y)
        return y
    elif ALGO == "MATMUL":
        A = make_matrix_flat(MAT_N)
        B = make_matrix_flat(MAT_N)
        C = array('f', (0.0 for _ in range(MAT_N * MAT_N)))
        matmul_worker(0, MAT_N, MAT_N, A, B, C)
        return C
    elif ALGO == "SPMV":
        vals, cols, row_ptr = make_sparse_csr(SPMV_N, SPMV_N, SPMV_NNZ_PER_ROW)
        x = make_vector(SPMV_N, 1)
        y = array('f', (0.0 for _ in range(SPMV_N)))
        spmv_worker(0, SPMV_N, SPMV_N, vals, cols, row_ptr, x, y)
        return y

def run_algo_multi():
    if ALGO == "AXPY":
        x = make_vector(AXPY_N, 1)
        y = make_vector(AXPY_N, 2)
        s0,e0 = split_range(AXPY_N, 2, 0)
        s1,e1 = split_range(AXPY_N, 2, 1)
        _start_secondary_and_run((s0,e0),(s1,e1), axpy_worker, (AXPY_A, x, y))
        return y
    elif ALGO == "MATMUL":
        A = make_matrix_flat(MAT_N)
        B = make_matrix_flat(MAT_N)
        C = array('f', (0.0 for _ in range(MAT_N * MAT_N)))
        s0,e0 = split_range(MAT_N, 2, 0)
        s1,e1 = split_range(MAT_N, 2, 1)
        _start_secondary_and_run((s0,e0),(s1,e1), matmul_worker, (MAT_N, A, B, C))
        return C
    elif ALGO == "SPMV":
        vals, cols, row_ptr = make_sparse_csr(SPMV_N, SPMV_N, SPMV_NNZ_PER_ROW)
        x = make_vector(SPMV_N, 1)
        y = array('f', (0.0 for _ in range(SPMV_N)))
        s0,e0 = split_range(SPMV_N, 2, 0)
        s1,e1 = split_range(SPMV_N, 2, 1)
        _start_secondary_and_run((s0,e0),(s1,e1), spmv_worker, (SPMV_N, vals, cols, row_ptr, x, y))
        return y

# ---------- Compare ----------
def compare_results(res1, res2):
    max_err = 0.0
    for a,b in zip(res1, res2):
        err = abs(a - b)
        if err > max_err: max_err = err
    return max_err

# ---------- Main ----------
for freq in (150_000_000, 280_000_000):
    print("\nTesting at", freq/1e6, "MHz")
    machine.freq(freq)
    gc.collect()

    ref = run_algo_single()
    gc.collect()
    multi = run_algo_multi()

    err = compare_results(ref, multi)
    print("Max abs error between single & multi-core:", err)

