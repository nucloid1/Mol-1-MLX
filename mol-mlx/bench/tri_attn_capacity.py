"""How large an N does each triangle-attention path survive under a hard memory cap?

MLX's peak counters proved unreliable here (a known 64 MiB allocation reported as
0.0 MiB while 256 MiB reported correctly), so this measures something that cannot
be misread: run one variant at one N in a fresh process under a hard RSS cap, and
record whether it finished. The largest surviving N is the capacity.

Timing is reported too -- wall clock over repeats, which the peak counters do not affect.

    tri_attn_capacity.py case <variant> <N> <cap_gib>
    tri_attn_capacity.py [cap_gib]
"""
import subprocess
import sys
import time

H, D = 10, 32
SCALE = D ** -0.5


def run_case(variant: str, N: int, cap: float) -> None:
    import mlx.core as mx
    from mol_mlx.guard import memory_guard
    from mol_mlx.tri_attn_metal import flash_triangle_attention

    with memory_guard(max_gib=cap) as peak:
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
        mx.eval(fn())
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t)
        inputs_mib = (3 * N * H * N * D + H * N * N + N * N) * 4 / 2**20
        print(f"OK {variant} N={N} ms={min(ts) * 1e3:.1f} rss_peak_gib={peak['rss']:.2f} "
              f"inputs_mib={inputs_mib:.0f}")


def driver(cap: float) -> None:
    sizes = [128, 192, 256, 320, 384, 448, 512]
    print(f"hard cap {cap} GiB per process; heads {H}, head_dim {D}")
    print(f"{'variant':8s} {'largest N':>9s}  per-N results")
    for variant in ("flash", "dense", "fused"):
        largest, notes = None, []
        for N in sizes:
            r = subprocess.run([sys.executable, __file__, "case", variant, str(N), str(cap)],
                               capture_output=True, text=True, timeout=900)
            line = next((l for l in r.stdout.splitlines() if l.startswith("OK ")), None)
            if line:
                ms = float(line.split("ms=")[1].split()[0])
                rss = float(line.split("rss_peak_gib=")[1].split()[0])
                largest = N
                notes.append(f"{N}:{ms:.0f}ms/{rss:.1f}G")
            else:
                killed = "memory_guard" in (r.stderr or "")
                notes.append(f"{N}:{'OOM' if killed else 'ERR'}")
                break
        print(f"{variant:8s} {str(largest):>9s}  " + "  ".join(notes))
        sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "case":
        run_case(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
    else:
        driver(float(sys.argv[1]) if len(sys.argv) > 1 else 6.0)
