"""Is bf16 / fp16 actually faster than fp32 on this GPU, at the shapes the stack uses?

Isolated matmuls only (a few hundred MB at most), under a 4 GiB cap. Shapes are the real
ones at N=256 (M = N*N = 65,536 pair rows):

  proj 448->448      tri-mul / tri-att projections in the 8 PairMixer blocks (c_z=448)
  proj 320->320      the same in the 40 Pairformer blocks (c_z=320)
  trans 448->1792    pair transition up-projection, PairMixer blocks
  trans 320->1280    pair transition up-projection, Pairformer blocks
  tri-mul bmm        the triangle contraction: [c, N, N] @ [c, N, N] batched over channels

Reports TFLOP/s per dtype and the ratio to fp32. fp16 is included for completeness only:
the single track reaches ~3e5 in this checkpoint, past fp16's 65,504 maximum.
"""
import time

import mlx.core as mx

from mol_mlx.guard import memory_guard

N = 256
M = N * N
DTYPES = [("fp32", mx.float32), ("fp16", mx.float16), ("bf16", mx.bfloat16)]


def bench(fn, flops, reps=7):
    mx.eval(fn())                                    # compile / warm
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return flops / min(ts) / 1e12, min(ts) * 1e3


def linear_case(k, n_out):
    flops = 2 * M * k * n_out
    out = {}
    for name, dt in DTYPES:
        x = (mx.random.normal((M, k), key=mx.random.key(0)) * 0.1).astype(dt)
        w = (mx.random.normal((n_out, k), key=mx.random.key(1)) * 0.1).astype(dt)
        mx.eval(x, w)
        out[name] = bench(lambda x=x, w=w: x @ w.T, flops)
    return out


def bmm_case(c):
    flops = 2 * c * N * N * N
    out = {}
    for name, dt in DTYPES:
        a = (mx.random.normal((c, N, N), key=mx.random.key(2)) * 0.1).astype(dt)
        b = (mx.random.normal((c, N, N), key=mx.random.key(3)) * 0.1).astype(dt)
        mx.eval(a, b)
        out[name] = bench(lambda a=a, b=b: a @ b, flops)
    return out


with memory_guard(max_gib=4.0):
    cases = [
        ("proj 448->448 (M=65536)", linear_case(448, 448)),
        ("proj 320->320", linear_case(320, 320)),
        ("trans 448->1792", linear_case(448, 1792)),
        ("trans 320->1280", linear_case(320, 1280)),
        ("tri-mul bmm c=448", bmm_case(448)),
        ("tri-mul bmm c=320", bmm_case(320)),
    ]
    print(f"N={N}, M={M}.  TFLOP/s (best of 7), and ratio to fp32\n")
    print(f"{'op':26s} {'fp32':>12s} {'fp16':>12s} {'bf16':>12s}   fp16/fp32  bf16/fp32")
    for label, r in cases:
        f32, f16, b16 = r["fp32"][0], r["fp16"][0], r["bf16"][0]
        print(f"{label:26s} {f32:12.2f} {f16:12.2f} {b16:12.2f}   {f16 / f32:8.2f}x  {b16 / f32:8.2f}x")
