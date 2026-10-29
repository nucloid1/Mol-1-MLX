"""Hold the full torch model + features resident (N=97) until told to stop; for the cross-process probe."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

from mol_mlx.guard import memory_guard
from trunk_bench import build

ready, stop = sys.argv[1], sys.argv[2]
with memory_guard(max_gib=12.0):
    model, f, kw = build(97, 4)
    open(ready, "w").write("ok")
    deadline = time.time() + 240
    while not os.path.exists(stop) and time.time() < deadline:
        time.sleep(0.5)
