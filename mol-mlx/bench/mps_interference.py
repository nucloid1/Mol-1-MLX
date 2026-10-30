"""What about a torch/MPS-initialized process slows MLX? Bisect, timing the same stack after each step (N=97)."""
import os
import time

import mlx.core as mx
import numpy as np
import torch

from mol_mlx.guard import memory_guard
from mol_mlx.msa import Trunk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    run = lambda: trunk.stack(s, z)
    print("0. baseline (torch imported only)       ", ms(run))
    x = torch.ones(4, device="mps")
    torch.mps.synchronize()
    print("1. torch MPS context initialised        ", ms(run))
    big = torch.empty(int(1.5 * 2**30) // 4, device="mps")
    big.fill_(1.0)
    torch.mps.synchronize()
    print("2. + 1.5 GiB MPS tensor resident        ", ms(run))
    big2 = torch.empty(int(2.5 * 2**30) // 4, device="mps")
    big2.fill_(1.0)
    torch.mps.synchronize()
    print("3. + 4 GiB total resident               ", ms(run))
    del big, big2
    torch.mps.empty_cache()
    print("4. after freeing it and empty_cache     ", ms(run))
