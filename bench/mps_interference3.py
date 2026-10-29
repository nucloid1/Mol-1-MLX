"""Does plain process memory growth slow the MLX stack? (N=97) -- no torch model, no featurizer."""
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
    return out


def rss_gib():
    return int(os.popen(f"ps -o rss= -p {os.getpid()}").read()) / 2**20


with memory_guard(max_gib=14.0):
    trunk = Trunk(os.path.join(ROOT, "weights"))
    r = np.load(os.path.join(ROOT, "refs", f"recycle_n{N}.npz"))
    s, z = mx.array(r["s_init"]), mx.array(r["z_init"])
    mx.eval(s, z)
    run = lambda: trunk.stack(s, z)
    print(f"0. baseline                          rss {rss_gib():5.1f} GiB  {ms(run)}")
    buf = np.ones(6 * 2**30 // 8)          # 6 GiB, touched
    print(f"1. + 6 GiB touched numpy buffer      rss {rss_gib():5.1f} GiB  {ms(run)}")
    del buf
    print(f"2. buffer freed                      rss {rss_gib():5.1f} GiB  {ms(run)}")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"3. + checkpoint torch.load on CPU    rss {rss_gib():5.1f} GiB  {ms(run)}")
    del ck
    print(f"4. checkpoint freed                  rss {rss_gib():5.1f} GiB  {ms(run)}")
