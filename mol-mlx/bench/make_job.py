"""Write a two-chain job (target + 24-residue binder, synthetic 64-row target MSA) of exactly N tokens.

    PYTHONPATH=.:scripts python bench/make_job.py <N> <out_dir>
"""
import json
import os
import sys

from capture_recycle_refs import BINDER, TARGET_POOL, synthetic_a3m

n, out = int(sys.argv[1]), sys.argv[2]
os.makedirs(out, exist_ok=True)
target = (TARGET_POOL * (n // len(TARGET_POOL) + 1))[: n - len(BINDER)]
open(f"{out}/A.a3m", "w").write(synthetic_a3m(target, 64, seed=n))
open(f"{out}/B.a3m", "w").write(f">query\n{BINDER}\n")
json.dump([{"name": f"n{n}", "sequences": [
    {"proteinChain": {"id": ["A"], "sequence": target, "count": 1, "unpairedMsaPath": f"{out}/A.a3m"}},
    {"proteinChain": {"id": ["B"], "sequence": BINDER, "count": 1, "unpairedMsaPath": f"{out}/B.a3m"}}]}],
    open(f"{out}/job.json", "w"))
print(f"job written: {len(target) + len(BINDER)} tokens -> {out}/job.json")
