"""Baseline vs after-torch.load stack time (N=97) under the current MLX_MAX_* env settings."""
import os
import time

import mlx.core as mx
import numpy as np
import torch

from mol_mlx.guard import memory_guard
from mol_mlx.msa import Trunk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = "/Users/ethanputnam/Mol-1/data_cache/checkpoint/mol-1-base.pt"
N = 97


def ms(fn, reps=4):
    mx.eval(*fn())
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(*fn())
        out.append(round(1e3 * (time.perf_counter() - t)))
    return min(out)


with memory_guard(max_gib=14.0):
    trunk = Trunk(os.path.join(ROOT, "weights"))
    r = np.load(os.path.join(ROOT, "refs", f"recycle_n{N}.npz"))
    s, z = mx.array(r["s_init"]), mx.array(r["z_init"])
    mx.eval(s, z)
    run = lambda: trunk.stack(s, z)
    a = ms(run)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    b = ms(run)
    knobs = {k: os.environ[k] for k in os.environ if k.startswith("MLX_")}
    print(f"{str(knobs or 'defaults'):55s} clean {a:4d} ms   after torch.load {b:4d} ms   (+{b - a})")
