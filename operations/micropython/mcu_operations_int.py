# ops_benchmark.py
# MicroPython OPS benchmark for Raspberry Pi Pico 2 (RP2350)
# - Measures floating-point operations/sec (add, sub, mul, div)
# - Can run single-core (main thread) or dual-core (main + worker via _thread)
# - Caveats: this measures throughput inside the MicroPython interpreter.
#   For raw hardware OPS you need native/C code (see notes below).
#
# Usage: copy this file to the Pico 2 and run. Follow prompts on the REPL.
#
# Author: generated with ChatGPT (GPT-5 Thinking mini)

import sys
import time
import machine

# CONFIG --- change defaults here or use interactive prompts
machine.freq(280_000_000) #Declare the frequency (in Hz)
DEFAULT_DURATION_S = 2.0    # target wall-clock seconds per run
UNROLL = 8                  # number of operations unrolled per inner block
CHUNK = 128                 # number of unrolled blocks between time checks
# Smaller CHUNK -> more frequent time checks (more overhead).
# Larger CHUNK -> less accurate stop time but less overhead.

OPS = ("add", "sub", "mul", "div")

try:
    import _thread
    THREAD_AVAILABLE = True
except Exception:
    _thread = None
    THREAD_AVAILABLE = False

# Utility wrappers (MicroPython compatible)
ticks_us = time.ticks_us
ticks_diff = time.ticks_diff
sleep_ms = time.sleep_ms

# Worker factory: returns a worker(tid, duration_us, results, ready, go) function
def make_worker(op):
    unroll = UNROLL
    chunk = CHUNK

    if op == "add":
        def worker(tid, duration_us, results, ready, go):
            x = 1
            y = 2
            local_ops = 0
            ready[tid] = True
            # wait for the 'go' flag (cooperative, yields with sleep_ms(0))
            while not go[0]:
                sleep_ms(0)
            start = ticks_us()
            while True:
                for _ in range(chunk):
                    # unrolled adds: unroll times
                    x += y; x += y; x += y; x += y; x += y; x += y; x += y; x += y
                    local_ops += unroll
                if ticks_diff(ticks_us(), start) >= duration_us:
                    break
            end = ticks_us()
            # store a small checksum (x+y) to avoid accidental optimization/remove
            results[tid] = (local_ops, start, end, x + y)
        return worker

    if op == "sub":
        def worker(tid, duration_us, results, ready, go):
            x = 12345
            y = 234
            local_ops = 0
            ready[tid] = True
            while not go[0]:
                sleep_ms(0)
            start = ticks_us()
            while True:
                for _ in range(chunk):
                    x -= y; x -= y; x -= y; x -= y; x -= y; x -= y; x -= y; x -= y
                    local_ops += unroll
                if ticks_diff(ticks_us(), start) >= duration_us:
                    break
            end = ticks_us()
            results[tid] = (local_ops, start, end, x - y)
        return worker

    if op == "mul":
        def worker(tid, duration_us, results, ready, go):
            x = 2
            y = 4
            local_ops = 0
            ready[tid] = True
            while not go[0]:
                sleep_ms(0)
            start = ticks_us()
            while True:
                for _ in range(chunk):
                    x *= y; x *= y; x *= y; x *= y; x *= y; x *= y; x *= y; x *= y
                    local_ops += unroll
                if ticks_diff(ticks_us(), start) >= duration_us:
                    break
            end = ticks_us()
            results[tid] = (local_ops, start, end, x * y)
        return worker

    if op == "div":
        def worker(tid, duration_us, results, ready, go):
            x = 12345
            y = 2
            local_ops = 0
            ready[tid] = True
            while not go[0]:
                sleep_ms(0)
            start = ticks_us()
            while True:
                for _ in range(chunk):
                    x /= y; x /= y; x /= y; x /= y; x /= y; x /= y; x /= y; x /= y
                    local_ops += unroll
                if ticks_diff(ticks_us(), start) >= duration_us:
                    break
            end = ticks_us()
            results[tid] = (local_ops, start, end, x / y)
        return worker

    raise ValueError("Unknown op: %r" % (op,))


def run_benchmark(op, cores=1, duration_s=DEFAULT_DURATION_S):
    if op not in OPS:
        raise ValueError("op must be one of %r" % (OPS,))

    if cores not in (1, 2):
        raise ValueError("cores must be 1 or 2 on Pico 2")

    if cores == 2 and not THREAD_AVAILABLE:
        print("Warning: _thread not available in this MicroPython build; falling back to single-core")
        cores = 1

    duration_us = int(duration_s * 1_000_000)
    worker = make_worker(op)

    # Shared containers (simple lists are fine for MicroPython threading)
    results = [None] * cores   # each entry -> (ops, start_us, end_us, checksum)
    ready = [False] * cores    # ready flags set by each worker when it is primed
    go = [False]               # single-element list used as mutable 'event'

    # Optionally do a short warm-up (helps with caches/JIT if any)
    print("Warm-up (0.15 s) ...")
    warm_worker = make_worker(op)
    warm_results = [None]
    warm_ready = [False]
    warm_go = [False]
    # run warm-up in main thread (quick run)
    warm_go[0] = True
    warm_worker(0, int(0.15 * 1_000_000), warm_results, warm_ready, warm_go)

    if cores == 1:
        print("Running single-core benchmark: op=%s duration=%.3fs" % (op, duration_s))
        # main thread will perform the work
        ready[0] = True  # main thread considered ready
        go[0] = True
        worker(0, duration_us, results, ready, go)
    else:
        print("Running multi-core benchmark (2 cores): op=%s duration=%.3fs" % (op, duration_s))
        # start worker on other core (tid = 1)
        _thread.start_new_thread(worker, (1, duration_us, results, ready, go))
        # mark main thread ready and wait for the secondary to be ready
        ready[0] = True
        # Wait for worker thread to signal readiness
        while not ready[1]:
            sleep_ms(0)
        # Start the benchmark for both threads at (approximately) the same time
        go[0] = True
        # Run same workload in the main thread too (tid = 0)
        worker(0, duration_us, results, ready, go)

        # wait until secondary thread writes its results
        while results[1] is None:
            sleep_ms(0)

    # Compute aggregated metrics
    total_ops = 0
    starts = []
    ends = []
    for i, r in enumerate(results):
        if r is None:
            continue
        ops_i, s_i, e_i, ch = r
        total_ops += ops_i
        starts.append(s_i)
        ends.append(e_i)

    if not starts:
        print("No results recorded.")
        return None

    earliest_start = min(starts)
    latest_end = max(ends)
    elapsed_us = ticks_diff(latest_end, earliest_start)
    elapsed_s = elapsed_us / 1_000_000.0 if elapsed_us > 0 else 0.0
    ops = total_ops / elapsed_s if elapsed_s > 0 else 0.0

    # Per-thread metrics printout
    print("\n=== Results ===")
    for i, r in enumerate(results):
        if r is None:
            print("tid=%d: no result" % i)
            continue
        ops_i, s_i, e_i, ch = r
        dur_i = ticks_diff(e_i, s_i) / 1_000_000.0
        ops_i = ops_i / dur_i if dur_i > 0 else 0.0
        print("tid=%d: ops=%d  time=%.6fs  ops=%.1f checksum=%r" % (i, ops_i, dur_i, ops_i, ch))

    print("----------------------------")
    print("Aggregated: total_ops=%d  wall_time=%.6fs  aggregated_OPS=%.1f" % (total_ops, elapsed_s, ops))
    print("Note: aggregated_OPS uses earliest-start .. latest-end wall time to reflect parallel work.")

    return {
        "op": op,
        "cores": cores,
        "duration_s": elapsed_s,
        "total_ops": total_ops,
        "ops": ops,
        "results": results,
    }


def interactive():
    print("MicroPython OPS benchmark for Raspberry Pi Pico 2\n")
    print("Available operations: %s" % (", ".join(OPS)))
    op = input("Choose operation (add/sub/mul/div) [add]: ").strip() or "add"
    if op not in OPS:
        print("Unknown op, using 'add'")
        op = "add"

    mode = input("Mode: single or multi (use two cores)? [single]: ").strip().lower() or "single"
    cores = 2 if mode.startswith("m") else 1
    if cores == 2 and not THREAD_AVAILABLE:
        print("Warning: _thread module not available in this MicroPython build. Falling back to single core.")
        cores = 1

    try:
        duration_s = float(input("Target duration (seconds) [%.1f]: " % DEFAULT_DURATION_S).strip() or DEFAULT_DURATION_S)
    except Exception:
        duration_s = DEFAULT_DURATION_S

    print("\nConfiguration: op=%s  cores=%d  duration=%.3fs  UNROLL=%d  CHUNK=%d" % (op, cores, duration_s, UNROLL, CHUNK))
    run_benchmark(op, cores=cores, duration_s=duration_s)


if __name__ == "__main__":
    # If run as a script, present interactive prompt in REPL
    try:
        interactive()
    except KeyboardInterrupt:
        print("\nAborted by user")
    except Exception as e:
        print("Error:", e)

