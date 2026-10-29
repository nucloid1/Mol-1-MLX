"""Same stack, same fixed inputs, N=97 -- but inside the process that has the real torch model on MPS.

Standalone MLX process: ~565 ms. If this is ~825 ms (what the hybrid loop sees), the slowdown comes
from sharing a process/GPU context with torch, not from anything in the recycle loop.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import mlx.core as mx
import numpy as np
import torch

from mol_mlx.guard import memory_guard
from mol_mlx.msa import Trunk
from trunk_bench import build

N = 97


def ms(fn, reps=5):
    mx.eval(*fn())
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(*fn())
        out.append(round(1e3 * (time.perf_counter() - t)))
    return out


with memory_guard(max_gib=12.0):
    trunk = Trunk(os.path.join(ROOT, "weights"))
    r = np.load(os.path.join(ROOT, "refs", f"recycle_n{N}.npz"))
    s, z = mx.array(r["s_init"]), mx.array(r["z_init"])
    mx.eval(s, z)
    print("before torch model loaded:        ", ms(lambda: trunk.stack(s, z)))
    model, f, kw = build(N, 4)
    torch.mps.synchronize()
    print("torch model on MPS, GPU idle:     ", ms(lambda: trunk.stack(s, z)))
    for m in (model.pairformer_stack, model.msa_module, model.template_embedder):
        m.to("cpu")
    torch.mps.empty_cache()
    print("after moving torch trunk to CPU:  ", ms(lambda: trunk.stack(s, z)))
    print("torch MPS driver GiB:", round(torch.mps.driver_allocated_memory() / 2**30, 2),
          " MLX active GiB:", round(mx.get_active_memory() / 2**30, 2))
