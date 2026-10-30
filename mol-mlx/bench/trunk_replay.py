"""MLX trunk loop on its own (no torch process), replaying captured real inputs and MSA samples.

Isolates "the loop" from "torch is also resident on the GPU": compare with trunk_breakdown, where
the same 4 cycles cost ~4.0 s inside the hybrid. N=97; refs hold 3 recorded cycles.
"""
import os
import sys
import time

import mlx.core as mx
import numpy as np

from mol_mlx.guard import memory_guard
from mol_mlx.hybrid import auto_chunks
from mol_mlx.msa import Trunk
from mol_mlx.template import pair_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 97
chunked = "--auto-chunks" in sys.argv

with memory_guard(max_gib=6.0):
    r = np.load(os.path.join(ROOT, "refs", f"recycle_n{N}.npz"))
    trunk = Trunk(os.path.join(ROOT, "weights"))
    z_init, s_init, s_inputs = mx.array(r["z_init"]), mx.array(r["s_init"]), mx.array(r["s_inputs"])
    feat = {k[5:]: r[k] for k in r.files if k.startswith("feat.")}
    at = pair_features(feat)
    cycles = sorted({int(k.split(".")[1][5:]) for k in r.files if k.startswith("msa.cycle") and ".sample." in k})
    samples = [{k[len(f"msa.cycle{c}.sample."):]: r[k] for k in r.files if k.startswith(f"msa.cycle{c}.sample.")}
               for c in cycles]
    print("recorded cycles:", cycles, "rows:", [s["msa"].shape[0] for s in samples])
    kw = auto_chunks(N) if chunked else {}
    mx.eval(z_init, s_init, s_inputs, at)

    def run():
        out = trunk(s_inputs, s_init, z_init, len(samples), sample_fn=lambda c: samples[c], template_at=at, **kw)
        mx.eval(*out)

    run()
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        run()
        ts.append(time.perf_counter() - t)
    print(f"{len(samples)} cycles, chunks={'auto' if chunked else 'none'}: {[round(t, 2) for t in ts]} s "
          f"({1e3 * min(ts) / len(samples):.0f} ms/cycle)")
