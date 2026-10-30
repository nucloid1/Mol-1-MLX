"""Does the 48-block stack slow down under sustained back-to-back load? (N=97, MLX only.)

trunk_profile timed isolated best-of-3 calls (697 ms); inside the real 4-cycle loop the same stack
took ~833 ms per call. If consecutive calls drift upward, the gap is sustained-clock/cache behaviour
rather than missing work in the loop.
"""
import os
import time

import mlx.core as mx

from mol_mlx.blocks import PairformerStack
from mol_mlx.guard import memory_guard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 97

with memory_guard(max_gib=6.0):
    stack = PairformerStack(mx.load(os.path.join(ROOT, "weights", "pairformer_stack.safetensors")), use_metal=True)
    s = mx.random.normal((N, 384), key=mx.random.key(0))
    z = mx.random.normal((N, N, 448), key=mx.random.key(1)) * 0.1
    mx.eval(s, z)
    mx.eval(*stack(s, z))
    ts = []
    for _ in range(12):
        t = time.perf_counter()
        mx.eval(*stack(s, z))
        ts.append(time.perf_counter() - t)
    print("consecutive calls, ms:", [round(1e3 * t) for t in ts])
    time.sleep(20)
    t = time.perf_counter()
    mx.eval(*stack(s, z))
    print(f"after a 20 s rest, ms: {1e3 * (time.perf_counter() - t):.0f}")
