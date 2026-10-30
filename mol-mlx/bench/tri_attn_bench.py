"""Flash Metal kernel vs MLX's dense and fused paths, at triangle-attention shapes.

One fresh process per (variant, N): peak accounting in a long-lived process is
polluted by the allocator reusing cached blocks.

    tri_attn_bench.py case <variant> <N>
    tri_attn_bench.py                     driver
"""
import subprocess
import sys
import time

H, D = 10, 32
SCALE = D ** -0.5
CAP_GIB = 12.0


def run_case(variant: str, N: int) -> None:
    import mlx.core as mx
    from mol_mlx.guard import memory_guard
    from mol_mlx.tri_attn_metal import flash_triangle_attention

    with memory_guard(max_gib=CAP_GIB):
        q = mx.random.normal((N, H, N, D), key=mx.random.key(0))
        k = mx.random.normal((N, H, N, D), key=mx.random.key(1))
        v = mx.random.normal((N, H, N, D), key=mx.random.key(2))
        tri = mx.random.normal((H, N, N), key=mx.random.key(3)) * 0.5
        msk = mx.zeros((N, N))
        mx.eval(q, k, v, tri, msk)

        def flash():
            return flash_triangle_attention(q, k, v, tri, msk, SCALE)

        def dense():
            s = (q * SCALE) @ mx.swapaxes(k, -1, -2) + tri[None] + msk[:, None, None, :]
            return mx.softmax(s, axis=-1, precise=True) @ v

        def fused():
            bias = tri[None] + msk[:, None, None, :]
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=bias)

        fn = {"flash": flash, "dense": dense, "fused": fused}[variant]
        mx.eval(fn())                       # compile / warm
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        mx.eval(fn())
        peak = (mx.get_peak_memory() - base) / 2**20

        ts = []
        for _ in range(5):
            t = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t)
        scores_mib = N * H * N * N * 4 / 2**20
        print(f"{N:5d} {variant:6s} peak {peak:9.1f} MiB  ({peak / scores_mib:5.2f}x scores)"
              f"  best {min(ts) * 1e3:8.1f} ms  scores-would-be {scores_mib:8.0f} MiB")


def driver() -> None:
    print(f"heads {H}, head_dim {D}; peak measured above the q/k/v/bias inputs\n")
    for N in (128, 256, 384, 512):
        for variant in ("flash", "dense", "fused"):
            r = subprocess.run([sys.executable, __file__, "case", variant, str(N)],
                               capture_output=True, text=True, timeout=600)
            lines = [l for l in r.stdout.splitlines() if l.strip()]
            if lines:
                print(lines[-1])
            else:
                tail = (r.stderr or "").strip().splitlines()[-1:] or ["no output"]
                print(f"{N:5d} {variant:6s} FAILED rc={r.returncode}: {tail[0][:90]}")
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "case":
        run_case(sys.argv[2], int(sys.argv[3]))
    else:
        driver()
