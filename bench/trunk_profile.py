"""Where does one MLX recycle cycle spend its time? Component timings, N=97 real inputs.

Each component is synced and timed on its own, after a warm-up, best of 3. MLX only (no torch),
so it is cheap and does not compete with other GPU work. Sample sizes are the real recorded ones.
Run it while nothing else is using the GPU.
"""
import os
import time

import mlx.core as mx
import numpy as np

from mol_mlx.guard import memory_guard
from mol_mlx.msa import Trunk
from mol_mlx.template import pair_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 97


def timed(fn, reps=3):
    mx.eval(fn())
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        best = min(best, time.perf_counter() - t)
    return best * 1e3


with memory_guard(max_gib=6.0):
    r = np.load(os.path.join(ROOT, "refs", f"recycle_n{N}.npz"))
    trunk = Trunk(os.path.join(ROOT, "weights"))
    z_init, s_init, s_inputs = mx.array(r["z_init"]), mx.array(r["s_init"]), mx.array(r["s_inputs"])
    feat = {k[5:]: r[k] for k in r.files if k.startswith("feat.")}
    at = pair_features(feat)
    sample = {k[len("msa.cycle0.sample."):]: r[k] for k in r.files if k.startswith("msa.cycle0.sample.")}
    rows = sample["msa"].shape[0]
    z = z_init + trunk.lin_z(trunk.ln_z(z_init))
    s = s_init
    mx.eval(z, s, at)

    rows_out = [
        ("recycle linears (z)", lambda: z_init + trunk.lin_z(trunk.ln_z(z))),
        ("template embedder, 4 empty slots", lambda: trunk.template(z, at)),
        (f"MSA module ({rows} rows, 4 blocks)", lambda: trunk.msa(sample, z, s_inputs)),
        ("pairformer stack (48 blocks)", lambda: trunk.stack(s, z)[1]),
    ]
    # finer: the MSA module's parts, using block 0's real shapes
    blk = trunk.msa.blocks[0]
    m0 = trunk.msa.lin_m(trunk.msa.featurize(sample)) + trunk.msa.lin_s(s_inputs)
    mx.eval(m0)
    rows_out += [
        ("  MSA block 0: outer product mean", lambda: blk.opm(m0)),
        ("  MSA block 0: msa stack", lambda: blk.msa_stack(m0, z)),
        ("  MSA block 0: pair stack (448-wide)", lambda: blk.pair_stack(None, z)[1]),
    ]
    tb = trunk.template.blocks[0]
    v = mx.zeros((N, N, 64))
    rows_out += [("  template: one 64-wide block", lambda: tb(None, v)[1])]
    times = [(name, timed(fn)) for name, fn in rows_out]
    total = sum(t for name, t in times[:4])
    print(f"N={N}, MSA sample {rows} rows; ms per component (one cycle):\n")
    for name, t in times:
        share = f"{100 * t / total:5.1f}%" if not name.startswith("  ") else ""
        print(f"  {name:40s} {t:8.1f} ms  {share}")
    print(f"\n  one cycle total (sum of the four top-level parts): {total:.1f} ms")
