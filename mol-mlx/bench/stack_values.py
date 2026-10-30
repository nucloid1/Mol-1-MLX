"""Is stack time value-dependent? Same shapes (N=97), random normals vs the real captured activations."""
import os
import time

import mlx.core as mx
import numpy as np

from mol_mlx.blocks import PairformerStack
from mol_mlx.guard import memory_guard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 97


def ms(fn, reps=5):
    mx.eval(*fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(*fn())
        ts.append(time.perf_counter() - t)
    return [round(1e3 * t) for t in ts]


with memory_guard(max_gib=6.0):
    stack = PairformerStack(mx.load(os.path.join(ROOT, "weights", "pairformer_stack.safetensors")), use_metal=True)
    r = np.load(os.path.join(ROOT, "refs", f"recycle_n{N}.npz"))
    cases = {
        "random normal (z*0.1)": (mx.random.normal((N, 384), key=mx.random.key(0)),
                                  mx.random.normal((N, N, 448), key=mx.random.key(1)) * 0.1),
        "real z_init, s_init": (mx.array(r["s_init"]), mx.array(r["z_init"])),
    }
    for k in r.files:           # a post-MSA z / the real stack input if the refs hold one
        if k.endswith("z_in") and k.startswith("stack"):
            cases[f"real {k}"] = (mx.array(r[k.replace("z_in", "s_in")]) if k.replace("z_in", "s_in") in r.files
                                  else cases["real z_init, s_init"][0], mx.array(r[k]))
    for name, (s, z) in cases.items():
        mx.eval(s, z)
        print(f"{name:28s} |z|max {float(mx.abs(z).max()):9.3g} |s|max {float(mx.abs(s).max()):9.3g}  ms: {ms(lambda: stack(s, z))}")
    print("keys with 'stack':", [k for k in r.files if "stack" in k][:8])
