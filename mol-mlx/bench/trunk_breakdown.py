"""Where does the hybrid trunk's wall time go at small N? (N=97, 4 cycles, trunk only, 12 GiB cap.)

trunk_bench's MLX number was ~1 s above 4 x the profiled MLX cycle. Time each part with a hard sync:
  prelude        torch input embedder + s_init + z_init (identical in the torch and hybrid paths)
  MLX trunk      Trunk.__call__ alone, torch->MLX conversion of its inputs excluded
  conversions    numpy/MLX array hand-off in and out
  total hybrid   model.get_pairformer_output after install()
  torch trunk    the original torch loop (the 'torch' column of trunk_bench), prelude included

    PYTHONPATH=. python bench/trunk_breakdown.py [N]
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 97
CYCLES = 4

import mlx.core as mx
import numpy as np
import torch

from mol_mlx import msa as msa_mod
from mol_mlx.guard import memory_guard
from mol_mlx.hybrid import install
from trunk_bench import build

# NO_MPS_PROBE=1 drops the guard's torch.mps.driver_allocated_memory() poll (a Metal API call made
# from a background thread every 0.1 s), to test whether the watchdog itself slows the loop.
PROBE = [] if os.environ.get("NO_MPS_PROBE") else [lambda: torch.mps.driver_allocated_memory() / 2**30]
with memory_guard(max_gib=12.0, extra=PROBE):
    model, f, kw = build(N, CYCLES)

    def sync():
        torch.mps.synchronize()

    def prelude():
        s_inputs = model.input_embedder(f, inplace_safe=False, chunk_size=kw["chunk_size"])
        s_init = model.linear_no_bias_sinit(s_inputs)
        z_init = (model.linear_no_bias_zinit1(s_init)[..., None, :]
                  + model.linear_no_bias_zinit2(s_init)[..., None, :, :])
        z_init = z_init + model.relative_position_encoding(f["relp"])
        z_init = z_init + model.linear_no_bias_token_bond(f["token_bonds"].unsqueeze(dim=-1))
        return s_inputs, s_init, z_init

    def best(fn, reps=3):
        out = []
        for _ in range(reps):
            torch.manual_seed(0)
            sync()
            t = time.perf_counter()
            with torch.no_grad():
                fn()
            sync()
            out.append(time.perf_counter() - t)
        return min(out)

    original = model.get_pairformer_output
    best(lambda: original(input_feature_dict=f, **kw), reps=1)          # warm-up
    t_torch = best(lambda: original(input_feature_dict=f, **kw))
    t_prelude = best(prelude)

    import mol.model.utils as mutils
    parts = {"msa sampling (torch)": 0.0, "template embedder": 0.0, "MSA module": 0.0, "pairformer stack": 0.0}
    rows_seen = []
    real_sample = mutils.sample_msa_feature_dict_random_without_replacement

    def timed_sample(*a, **k):
        sync()
        t = time.perf_counter()
        out = real_sample(*a, **k)
        sync()
        parts["msa sampling (torch)"] += time.perf_counter() - t
        rows_seen.append(int(out["msa"].shape[-2]))
        return out

    mutils.sample_msa_feature_dict_random_without_replacement = timed_sample
    trunk = install(model, os.path.join(ROOT, "weights"))

    class Timed:
        def __init__(self, inner, name):
            self.inner, self.name = inner, name

        def __call__(self, *a, **k):
            t = time.perf_counter()
            out = self.inner(*a, **k)
            mx.eval(*(out if isinstance(out, tuple) else (out,)))
            parts[self.name] += time.perf_counter() - t
            return out

        def __getattr__(self, n):
            return getattr(self.inner, n)

    trunk.template = Timed(trunk.template, "template embedder")
    trunk.msa = Timed(trunk.msa, "MSA module")
    trunk.stack = Timed(trunk.stack, "pairformer stack")
    acc = {"trunk": 0.0}
    cls = type(trunk)
    orig_call = cls.__call__

    def timed_call(self, *a, **k):
        t = time.perf_counter()
        out = orig_call(self, *a, **k)
        mx.eval(*out)
        acc["trunk"] += time.perf_counter() - t
        return out

    cls.__call__ = timed_call
    best(lambda: model.get_pairformer_output(input_feature_dict=f, **kw), reps=1)   # warm-up
    acc["trunk"] = 0.0
    for k in parts:
        parts[k] = 0.0
    rows_seen.clear()
    reps = 3
    t_hybrid = best(lambda: model.get_pairformer_output(input_feature_dict=f, **kw), reps=reps)
    t_trunk = acc["trunk"] / (reps)        # accumulated over `reps` calls

    s_inputs, s_init, z_init = prelude()
    sync()
    t = time.perf_counter()
    a = mx.array(z_init.detach().cpu().numpy())
    mx.eval(a)
    t_in = time.perf_counter() - t
    t = time.perf_counter()
    back = torch.from_numpy(np.array(a)).to(z_init.device)
    sync()
    t_out = time.perf_counter() - t

    print(f"N={N}, {CYCLES} cycles, trunk only, all times in s\n")
    print(f"  torch trunk loop, incl. prelude        {t_torch:6.2f}")
    print(f"    of which prelude (embedder, z_init)  {t_prelude:6.2f}")
    print(f"  hybrid total                           {t_hybrid:6.2f}")
    print(f"    MLX trunk call (loop + MSA sampling) {t_trunk:6.2f}   ({1e3 * t_trunk / CYCLES:.0f} ms/cycle)")
    print(f"    prelude                              {t_prelude:6.2f}")
    print(f"    z_init -> MLX / z back -> torch      {t_in:6.3f} / {t_out:6.3f}  (one array each way)")
    print(f"    unattributed                         {t_hybrid - t_trunk - t_prelude:6.2f}")
    print(f"  torch trunk loop WITHOUT prelude       {t_torch - t_prelude:6.2f}")
    print(f"\n  inside the MLX loop, per call of the trunk (mean of {reps}), s:")
    for k, v in parts.items():
        print(f"    {k:24s} {v / reps:6.2f}")
    rest = t_trunk - sum(parts.values()) / reps
    print(f"    {'everything else':24s} {rest:6.2f}   (recycle linears, glue)")
    print(f"    MSA rows drawn per cycle (all reps): {rows_seen}")
    print(f"  => MLX loop vs torch loop              {(t_torch - t_prelude) / t_trunk:6.2f}x")
