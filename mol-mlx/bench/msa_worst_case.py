"""Worker memory with a DEEP MSA: N tokens, M sampled rows in one cycle (default N=586, M=7000), via the real
RemoteTrunk and hybrid.auto_chunks. Mol-1 samples up to 16384 rows per cycle on real alignments; the other
benchmarks used a 64-row synthetic one, which never exercises this.

    PYTHONPATH=. python bench/msa_worst_case.py [N] [M] [worker_cap_gib]
"""
import sys
import time

import numpy as np

from mol_mlx.hybrid import auto_chunks
from mol_mlx.worker import RemoteTrunk

N = int(sys.argv[1]) if len(sys.argv) > 1 else 586
M = int(sys.argv[2]) if len(sys.argv) > 2 else 7000
cap = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

rng = np.random.default_rng(0)
sample = {"msa": rng.integers(0, 32, (M, N), dtype=np.uint8),
          "has_deletion": rng.integers(0, 2, (M, N), dtype=np.uint8),
          "deletion_value": rng.random((M, N), dtype=np.float32)}
args = (rng.random((N, 449), dtype=np.float32), rng.standard_normal((N, 384), dtype=np.float32),
        (0.1 * rng.standard_normal((N, N, 448))).astype(np.float32))
w = RemoteTrunk("weights", cap_gib=cap)
print(f"N={N}, M={M} rows, chunks={auto_chunks(N)}, worker cap {cap} GiB", flush=True)
t = time.perf_counter()
s, z = w(*args, 1, [sample], **auto_chunks(N))
print(f"1 cycle: {w.last_seconds:.1f} s in the worker ({time.perf_counter() - t:.1f} s with start-up); "
      f"worker MLX peak {w.last_mlx_peak_gib:.1f} GiB; finite={bool(np.isfinite(z).all())}")
w.close()
