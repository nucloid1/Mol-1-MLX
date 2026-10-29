"""v2 (simdgroup matrix tiles) vs v1 (one query per thread) vs dense MLX, triangle-attention shapes.

Time is best-of-7 wall clock after a warm-up. TFLOP/s counts the attention math only
(4*B*H*N^2*D: QK and PV at 2 FLOP per MAC), so it is comparable across variants and
excludes the surrounding projections. N=250 is not a multiple of 8, to show what the
padding copy costs v2. Each case is also checked against dense (max relative error).

Capped at 4 GiB; the largest case (N=384) holds under 1 GB of tensors.
"""
import time

import mlx.core as mx

from mol_mlx.guard import memory_guard
from mol_mlx.tri_attn_metal import flash_triangle_attention as v1
from mol_mlx.tri_attn_mma import flash_triangle_attention_mma as v2

H, D = 10, 32
SCALE = D ** -0.5


def dense(q, k, v, tri, msk):
    s = (q * SCALE) @ mx.swapaxes(k, -1, -2) + tri[None] + msk[:, None, None, :]
    return mx.softmax(s, axis=-1, precise=True) @ v


def timeit(fn, reps=7):
    mx.eval(fn())
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        best = min(best, time.perf_counter() - t)
    return best * 1e3


with memory_guard(max_gib=4.0):
    print(f"heads {H}, head_dim {D}; B = N batch rows; fp32\n")
    print(f"{'N':>4s} {'variant':7s} {'ms':>8s} {'TFLOP/s':>8s} {'vs v1':>7s} {'vs dense':>9s} {'max rel err':>12s}")
    for N in (128, 250, 256, 384):
        q = mx.random.normal((N, H, N, D), key=mx.random.key(0))
        k = mx.random.normal((N, H, N, D), key=mx.random.key(1))
        v = mx.random.normal((N, H, N, D), key=mx.random.key(2))
        tri = mx.random.normal((H, N, N), key=mx.random.key(3)) * 0.5
        msk = mx.zeros((N, N))
        mx.eval(q, k, v, tri, msk)
        ref = dense(q, k, v, tri, msk)
        mx.eval(ref)
        flop = 4.0 * N * H * N * N * D
        t = {}
        for name, fn in (("dense", lambda: dense(q, k, v, tri, msk)),
                         ("v1", lambda: v1(q, k, v, tri, msk, SCALE)),
                         ("v2", lambda: v2(q, k, v, tri, msk, SCALE))):
            t[name] = timeit(fn)
        for name, fn in (("dense", None), ("v1", v1), ("v2", v2)):
            err = 0.0 if fn is None else float(mx.abs(fn(q, k, v, tri, msk, SCALE) - ref).max()) / float(mx.abs(ref).max())
            print(f"{N:4d} {name:7s} {t[name]:8.2f} {flop / (t[name] * 1e-3) / 1e12:8.2f} "
                  f"{t['v1'] / t[name]:6.2f}x {t['dense'] / t[name]:8.2f}x {err:12.2e}")
        print()
        q = k = v = tri = msk = ref = None   # free before the next N (lambdas close over these names)
