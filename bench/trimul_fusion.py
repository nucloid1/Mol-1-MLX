"""Triangle multiplication: the 5-way concatenated projection LOST (23.5 vs 15.3 ms, see fusion_probe),
so test the smaller change: keep the separate projections but fuse the gating passes with mx.compile.

  current          sigmoid(a_g)*a_p as two passes each for a and b, then sigmoid(g) and a multiply
  compiled gates   one fused pass for each a, b
  + final gate     also fuse sigmoid(g(zn)) * out
  4-way concat     a_p|b_p|a_g|b_g in one matmul (g stays separate) + compiled gate on the slices
Real block-8 weights, N=256, 4 GiB cap.
"""
import os
import time

import mlx.core as mx

from mol_mlx.guard import memory_guard
from mol_mlx.layers import TriangleMultiplicativeUpdate, Weights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 256


def timed(fn, reps=7):
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
def gate(g, p):
    return mx.sigmoid(g) * p


@mx.compile
def gate_out(x, g):
    return x * mx.sigmoid(g)


with memory_guard(max_gib=4.0):
    w = Weights(mx.load(os.path.join(ROOT, "weights", "pairformer_stack.safetensors")))
    z = mx.random.normal((N, N, 320), key=mx.random.key(0)) * 0.5
    tm = TriangleMultiplicativeUpdate(w, "blocks.8.tri_mul_out", outgoing=True)
    c = 320
    w4 = mx.concatenate([tm.a_p.weight, tm.b_p.weight, tm.a_g.weight, tm.b_g.weight], axis=0)
    mx.eval(z, w4)

    def contract(a, b):
        return mx.transpose(mx.transpose(a, (2, 0, 1)) @ mx.transpose(b, (2, 1, 0)), (1, 2, 0))

    def v_gates(x, final=False):
        zn = tm.ln_in(x)
        a = gate(tm.a_g(zn), tm.a_p(zn))
        b = gate(tm.b_g(zn), tm.b_p(zn))
        o = tm.z_out(tm.ln_out(contract(a, b)))
        return gate_out(o, tm.g(zn)) if final else o * mx.sigmoid(tm.g(zn))

    def v_concat4(x):
        zn = tm.ln_in(x)
        p = zn @ w4.T
        a = gate(p[..., 2 * c:3 * c], p[..., :c])
        b = gate(p[..., 3 * c:], p[..., c:2 * c])
        return gate_out(tm.z_out(tm.ln_out(contract(a, b))), tm.g(zn))

    ref = tm(z)
    mx.eval(ref)
    print(f"Triangle multiplication, N={N}, c={c}\n")
    print(f"  {'variant':34s} {'rel err':>9s} {'ms':>7s}")
    print(f"  {'current':34s} {0.0:9.1e} {timed(lambda: tm(z)):7.2f}")
    print(f"  {'compiled gates':34s} {rel(v_gates(z), ref):9.1e} {timed(lambda: v_gates(z)):7.2f}")
    print(f"  {'compiled gates + final gate':34s} {rel(v_gates(z, True), ref):9.1e} {timed(lambda: v_gates(z, True)):7.2f}")
    print(f"  {'4-way concat + compiled gates':34s} {rel(v_concat4(z), ref):9.1e} {timed(lambda: v_concat4(z)):7.2f}")
