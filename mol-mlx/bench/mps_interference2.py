"""Which step of building the real Mol-1 model/features slows the MLX stack? Probe after each step (N=97)."""
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import mlx.core as mx
import numpy as np
import torch

from mol_mlx.guard import memory_guard
from mol_mlx.msa import Trunk

N = 97


def ms(fn, reps=4):
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

    def probe(label):
        print(f"{label:44s} {ms(run)}   threads={threading.active_count()} torch_threads={torch.get_num_threads()}")

    probe("0. baseline")
    from mol.runner import inference as inf
    from mol.data.inference import infer_dataloader as dl
    import mol.utils.torch_utils as tu
    probe("1. mol imported")

    def after(mod, name, label):
        orig = getattr(mod, name)

        def wrapped(*a, **k):
            out = orig(*a, **k)
            probe(label)
            return out
        setattr(mod, name, wrapped)

    after(inf.InferenceRunner, "__init__", "2. InferenceRunner built (model on MPS)")
    after(dl.InferenceDataset, "__getitem__", "3. dataset item featurized")
    from trunk_bench import build
    model, f, kw = build(N, 4)
    probe("4. build() finished")

    import gc
    print(f"\ntracked objects in the process: {len(gc.get_objects()):,}   gc thresholds {gc.get_threshold()}")
    stats0 = [s["collections"] for s in gc.get_stats()]
    ms(run)
    stats1 = [s["collections"] for s in gc.get_stats()]
    print("gc collections during 5 stack calls (gen0, gen1, gen2):", [b - a for a, b in zip(stats0, stats1)])
    gc.freeze()
    probe("5. gc.freeze() (heap moved to permanent gen)")
    gc.unfreeze()
    gc.disable()
    probe("6. gc.disable()")
    gc.enable()
