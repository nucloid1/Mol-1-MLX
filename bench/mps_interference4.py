"""Is torch's CPU thread pool spinning and stealing cores from MLX's dispatch? (N=97)

After a torch CPU load, compare process CPU-seconds to wall-seconds during one stack call, and see
whether limiting torch's threads recovers the speed.
"""
import os
import resource
import sys
import time

import mlx.core as mx
import numpy as np
import torch

from mol_mlx.guard import memory_guard
from mol_mlx.msa import Trunk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = "/Users/ethanputnam/Mol-1/data_cache/checkpoint/mol-1-base.pt"
N = 97


def ms_cpu(fn, reps=4):
    mx.eval(*fn())
    out = []
    for _ in range(reps):
        c0 = resource.getrusage(resource.RUSAGE_SELF)
        t = time.perf_counter()
        mx.eval(*fn())
        wall = time.perf_counter() - t
        c1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime)
        out.append(f"{1e3 * wall:.0f}ms cpu/wall={cpu / wall:.2f}")
    return out


with memory_guard(max_gib=14.0):
    trunk = Trunk(os.path.join(ROOT, "weights"))
    r = np.load(os.path.join(ROOT, "refs", f"recycle_n{N}.npz"))
    s, z = mx.array(r["s_init"]), mx.array(r["z_init"])
    mx.eval(s, z)
    run = lambda: trunk.stack(s, z)
    print("0. baseline                       ", ms_cpu(run))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("1. after torch.load (cpu)         ", ms_cpu(run))
    torch.set_num_threads(1)
    print("2. torch.set_num_threads(1)       ", ms_cpu(run))
    del ck
    print("3. checkpoint freed, 1 thread     ", ms_cpu(run))
