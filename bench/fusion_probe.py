"""What can fusing the elementwise passes buy in the transition and triangle multiplication?

The two ops are ~69% of a Pairformer block and run at roughly half to 60% of the matmul roofline,
so the remainder is memory-bound passes (layer norm, sigmoid gating, multiplies, layout copies).
This probes candidate rewrites on the real block-8 weights at N=256 (c_z=320), checking each
against the current implementation (max relative error) before reading its time.

  Transition   current | one concatenated a/b matmul + compiled silu*b | + row chunks sized to
               stay cache-resident (the chunk size is swept)
  Tri-mul      current | one concatenated 5-way projection + compiled gate-combine

Capped at 4 GiB (N=256 pair tensors are 84 MB each).
"""
import os
import time

import mlx.core as mx

from mol_mlx.guard import memory_guard
from mol_mlx.layers import Transition, TriangleMultiplicativeUpdate, Weights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 256


def timed(fn, reps=5):
    mx.eval(fn())
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def rel(a, b):
    return float(mx.abs(a - b).max()) / float(mx.abs(b).max())


@mx.compile
def silu_mul(a, b):
    return a * mx.sigmoid(a) * b


@mx.compile
def gate_combine(ag, ap, bg, bp):
    return mx.sigmoid(ag) * ap, mx.sigmoid(bg) * bp


with memory_guard(max_gib=4.0):
    w = Weights(mx.load(os.path.join(ROOT, "weights", "pairformer_stack.safetensors")))
    z = mx.random.normal((N, N, 320), key=mx.random.key(0)) * 0.5
    mx.eval(z)

    # ------------------------------------------------------------------ transition
    tr = Transition(w, "blocks.8.pair_transition")
    w_ab = mx.concatenate([tr.a.weight, tr.b.weight], axis=0)          # [2*4c, c]
    hidden = tr.a.weight.shape[0]

    def tr_fused(x):
        y = tr.ln(x)
        ab = y @ w_ab.T
        return tr.o(silu_mul(ab[..., :hidden], ab[..., hidden:]))

    def tr_fused_chunked(x, rows):
        parts = [tr_fused(x[i:i + rows]) for i in range(0, x.shape[0], rows)]
        return mx.concatenate(parts, axis=0)

    ref = tr(z)
    mx.eval(ref)
    print(f"Transition, N={N}, c=320 (hidden {hidden}); error vs current, then ms\n")
    print(f"  {'variant':44s} {'rel err':>9s} {'ms':>8s}")
    print(f"  {'current (separate a, b matmuls)':44s} {0.0:9.1e} {timed(lambda: tr(z)):8.2f}")
    print(f"  {'concatenated a/b matmul + compiled silu*b':44s} {rel(tr_fused(z), ref):9.1e} {timed(lambda: tr_fused(z)):8.2f}")
    for rows in (4, 8, 16, 32, 64):
        mb = rows * N * hidden * 2 * 4 / 2**20
        print(f"  {f'  ... in chunks of {rows:2d} rows ({mb:5.1f} MB hidden)':44s} "
              f"{rel(tr_fused_chunked(z, rows), ref):9.1e} {timed(lambda r=rows: tr_fused_chunked(z, r)):8.2f}")
    for rows in (8, 16, 32, 64):
        print(f"  {f'current transition, chunk={rows}':44s} {rel(tr(z, chunk=rows), ref):9.1e} {timed(lambda r=rows: tr(z, chunk=r)):8.2f}")

    # ------------------------------------------------------------------ triangle multiplication
    tm = TriangleMultiplicativeUpdate(w, "blocks.8.tri_mul_out", outgoing=True)
    c = tm.a_p.weight.shape[0]
    w_cat = mx.concatenate([tm.a_p.weight, tm.b_p.weight, tm.a_g.weight, tm.b_g.weight, tm.g.weight], axis=0)

    def tm_fused(x):
        zn = tm.ln_in(x)
        p = zn @ w_cat.T                                       # one read of zn, one [N,N,5c] write
        ap, bp, ag, bg, g = (p[..., i * c:(i + 1) * c] for i in range(5))
        a, b = gate_combine(ag, ap, bg, bp)
        at, bt = mx.transpose(a, (2, 0, 1)), mx.transpose(b, (2, 1, 0))
        xo = mx.transpose(at @ bt, (1, 2, 0))
        return tm.z_out(tm.ln_out(xo)) * mx.sigmoid(g)

    ref = tm(z)
    mx.eval(ref)
    print(f"\nTriangle multiplication (outgoing), N={N}, c=320; error vs current, then ms\n")
    print(f"  {'variant':44s} {'rel err':>9s} {'ms':>8s}")
    print(f"  {'current':44s} {0.0:9.1e} {timed(lambda: tm(z)):8.2f}")
    print(f"  {'concatenated 5-way projection + compiled gate':44s} {rel(tm_fused(z), ref):9.1e} {timed(lambda: tm_fused(z)):8.2f}")

    # what is the floor? the matmul work alone, no elementwise at all
    zn = tm.ln_in(z)
    mx.eval(zn)
    print(f"\n  reference points (N={N}, c={c}):")
    print(f"    5 projections as ONE matmul (no elementwise)  {timed(lambda: zn @ w_cat.T):8.2f} ms")
    ap = mx.random.normal((N, N, c), key=mx.random.key(1))
    at_, bt_ = mx.transpose(ap, (2, 0, 1)), mx.transpose(ap, (2, 1, 0))
    mx.eval(at_, bt_)
    print(f"    the triangle contraction alone, c batched     {timed(lambda: at_ @ bt_):8.2f} ms")
    print(f"    one layer norm over [N,N,c]                   {timed(lambda: tm.ln_in(z)):8.2f} ms")
