"""Interleaved A/B of the fused transition + triangle-multiplication ops on the full 48-block stack.

Same process, same weights and inputs, ABABAB so thermal state and session noise hit both arms
equally. "unfused" monkeypatches the previous implementation back in (separate a/b matmuls,
uncompiled sigmoid/multiply passes). N=256, hard cap 8 GiB.
"""
import os
import statistics
import time

import mlx.core as mx

from mol_mlx import layers
from mol_mlx.blocks import PairformerStack
from mol_mlx.guard import memory_guard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 256
ROUNDS = 5


def unfused_transition(self, x, chunk=None):
    def proj(x):
        y = self.ln(x)
        a = y @ self.ab.weight[:self.hidden].T
        b = y @ self.ab.weight[self.hidden:].T
        return self.o(a * mx.sigmoid(a) * b)
    if chunk is None or chunk >= x.shape[-2]:
        return proj(x)
    return mx.concatenate([proj(x[..., i:i + chunk, :]) for i in range(0, x.shape[-2], chunk)], axis=-2)


def unfused_trimul(self, z, mask=None):
    zn = self.ln_in(z)
    a = mx.sigmoid(self.a_g(zn)) * self.a_p(zn)
    b = mx.sigmoid(self.b_g(zn)) * self.b_p(zn)
    if self.outgoing:
        at, bt = mx.transpose(a, (2, 0, 1)), mx.transpose(b, (2, 1, 0))
    else:
        at, bt = mx.transpose(a, (2, 1, 0)), mx.transpose(b, (2, 0, 1))
    x = mx.transpose(at @ bt, (1, 2, 0))
    return self.z_out(self.ln_out(x)) * mx.sigmoid(self.g(zn))


fused = (layers.Transition.__call__, layers.TriangleMultiplicativeUpdate.__call__)
unfused = (unfused_transition, unfused_trimul)


def use(pair):
    layers.Transition.__call__, layers.TriangleMultiplicativeUpdate.__call__ = pair


with memory_guard(max_gib=8.0):
    w = mx.load(os.path.join(ROOT, "weights", "pairformer_stack.safetensors"))
    stack = PairformerStack(w, use_metal=True)
    s = mx.random.normal((N, 384), key=mx.random.key(0))
    z = mx.random.normal((N, N, 448), key=mx.random.key(1)) * 0.1
    mx.eval(s, z)
    for pair in (fused, unfused):                      # warm both (kernel + compile caches)
        use(pair)
        mx.eval(*stack(s, z))
    times = {"fused": [], "unfused": []}
    for _ in range(ROUNDS):
        for name, pair in (("unfused", unfused), ("fused", fused)):
            use(pair)
            t = time.perf_counter()
            mx.eval(*stack(s, z))
            times[name].append(time.perf_counter() - t)
    use(fused)
    ref = stack(s, z)[1]
    use(unfused)
    alt = stack(s, z)[1]
    err = float(mx.abs(ref - alt).max()) / float(mx.abs(alt).max())
    print(f"N={N}, 48-block stack, {ROUNDS} interleaved rounds (s):")
    for name, ts in times.items():
        print(f"  {name:8s} min {min(ts):.2f}  median {statistics.median(ts):.2f}  all {[round(t, 2) for t in ts]}")
    print(f"  fused speedup: min {min(times['unfused']) / min(times['fused']):.3f}x   "
          f"median {statistics.median(times['unfused']) / statistics.median(times['fused']):.3f}x")
    print(f"  fused vs unfused output, rel max err: {err:.1e}")
