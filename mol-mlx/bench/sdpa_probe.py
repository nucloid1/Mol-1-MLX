"""Does MLX's fused SDPA handle triangle attention's shape without materializing scores?

Triangle attention, trunk blocks 8-47:  B = N rows, H = 10 heads, L = N, head_dim = 32,
with a [1, H, N, N] additive pair bias shared across rows.

One FRESH PROCESS per (variant, N): in-process peak accounting is polluted by the
allocator reusing cached buffers, which made an earlier version of this probe report
nonsense (a naive path peaking below its own score tensor). Here the inputs are
built and evaluated first, the peak counter is reset, and only then is the op run.

    sdpa_probe.py case <variant> <N>     one case, prints a single result line
    sdpa_probe.py                        driver: loops cases in subprocesses

Capped at 6 GiB per process by the memory guard; N stays <= 288.
"""
import subprocess
import sys
import time

H, D = 10, 32
SCALE = D ** -0.5


def run_case(variant: str, N: int) -> None:
    import mlx.core as mx
    from mol_mlx.guard import memory_guard

    with memory_guard(max_gib=6.0):
        key = mx.random.key(0)
        q = mx.random.normal((N, H, N, D), key=key)
        k = mx.random.normal((N, H, N, D), key=key)
        v = mx.random.normal((N, H, N, D), key=key)
        bias = mx.random.normal((1, H, N, N), key=key) * 0.5
        mx.eval(q, k, v, bias)

        def fused(q, k, v, b):
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=b)

        def naive(q, k, v, b):
            s = (q * SCALE) @ k.transpose(0, 1, 3, 2) + b
            return mx.softmax(s, axis=-1) @ v

        fn = fused if variant == "fused" else naive
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        out = fn(q, k, v, bias)
        mx.eval(out)
        peak = (mx.get_peak_memory() - base) / 2**20

        # warm timing (kernels compiled), a few repeats
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            mx.eval(fn(q, k, v, bias))
            ts.append(time.perf_counter() - t)
        ref = naive(q, k, v, bias)
        err = float(mx.abs(out - ref).max())
        out_mib = N * H * N * D * 4 / 2**20
        sc_mib = N * H * N * N * 4 / 2**20
        print(f"{N:5d} {variant:6s} out {out_mib:6.0f}  scores {sc_mib:6.0f}  peak-above-inputs {peak:7.0f} MiB"
              f"  ({peak / sc_mib:4.2f}x scores)  warm {min(ts) * 1e3:7.1f} ms  maxerr-vs-naive {err:.1e}")


def driver() -> None:
    print(f"head_dim {D}, heads {H}; peak is above the q/k/v/bias inputs, one process per row\n")
    for N in (96, 192, 288):
        for variant in ("fused", "naive"):
            r = subprocess.run([sys.executable, __file__, "case", variant, str(N)],
                               capture_output=True, text=True, timeout=300)
            line = [l for l in r.stdout.splitlines() if l.strip()]
            print(line[-1] if line else f"{N} {variant}: FAILED rc={r.returncode} {r.stderr.strip().splitlines()[-1:]}")
            sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "case":
        run_case(sys.argv[2], int(sys.argv[3]))
    else:
        driver()
