"""Run ``mol1-mlx`` under the memory watchdog: ``guarded_pred.py <cap_gib> <worker_cap_gib> pred <mol pred flags...>``.

The watchdog kills THIS process (the torch side) when the torch MPS allocator, MLX or the resident set passes
``cap_gib``; the MLX worker has its own cap. For big inputs, where an uncapped run can take the machine with it.
Prints the sampled peaks at the end (sampled every 0.1 s: a proxy, not an exact peak).
"""
import os
import sys
import time

cap, worker_cap = float(sys.argv[1]), float(sys.argv[2])
os.environ["MOLMLX_WORKER_CAP_GIB"] = str(worker_cap)
sys.argv = ["mol1-mlx"] + sys.argv[3:]

import torch  # noqa: E402

from mol_mlx.cli import main  # noqa: E402
from mol_mlx.guard import memory_guard  # noqa: E402

t0 = time.perf_counter()
code = 0
with memory_guard(max_gib=cap, extra=[lambda: torch.mps.driver_allocated_memory() / 2**30]) as peak:
    try:
        main()
    except SystemExit as e:
        code = e.code or 0
    sys.stderr.write(f"\n[guarded_pred] exit {code} after {time.perf_counter() - t0:.1f} s; sampled peaks (GiB): "
                     f"torch MPS {peak['extra']:.1f}, MLX {peak['mlx']:.1f}, resident set {peak['rss']:.1f}; cap {cap}\n")
sys.exit(code)
