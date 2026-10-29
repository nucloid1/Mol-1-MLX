"""Cheap in-process remedies for the post-torch.load slowdown of the MLX stack (N=97)."""
import os
import threading
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


with memory_guard(max_gib=14.0):
    trunk = Trunk(os.path.join(ROOT, "weights"))
    r = np.load(os.path.join(ROOT, "refs", f"recycle_n{N}.npz"))
    s, z = mx.array(r["s_init"]), mx.array(r["z_init"])
    mx.eval(s, z)
    run = lambda: trunk.stack(s, z)
    print("0. clean                              ", ms(run))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("1. after torch.load                   ", ms(run))
    mx.clear_cache()
    print("2. + mx.clear_cache()                 ", ms(run))
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("t", ms(run)))
    th.start(); th.join()
    print("3. run on a fresh Python thread       ", box["t"])
    stream = mx.new_stream(mx.gpu)
    with mx.stream(stream):
        print("4. on a dedicated mx stream           ", ms(run))
    del ck
    import gc; gc.collect()
    print("5. checkpoint freed + gc              ", ms(run))
