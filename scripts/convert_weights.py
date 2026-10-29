"""Convert the Mol-1 checkpoint to per-module MLX safetensors, with strict key accounting.

The torch checkpoint is read once, the `module.` DDP prefix is stripped, and tensors are
grouped by their top-level module (the first path component) into weights/<group>.safetensors,
all float32. A manifest records every group's key count, parameter count and bytes, so the
porting status of every weight is visible and a key can never silently go unported.

    python scripts/convert_weights.py [checkpoint.pt]
"""
import collections
import json
import os
import sys

import mlx.core as mx
import numpy as np
import torch

from mol_mlx.guard import memory_guard

CKPT = "/Users/ethanputnam/Mol-1/data_cache/checkpoint/mol-1-base.pt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")


def main(ckpt_path: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]

    groups: dict[str, dict[str, np.ndarray]] = collections.defaultdict(dict)
    dtypes = collections.Counter()
    for key, tensor in state.items():
        name = key[len("module."):] if key.startswith("module.") else key
        group, _, rest = name.partition(".")
        dtypes[str(tensor.dtype)] += 1
        groups[group][rest or group] = tensor.detach().float().cpu().numpy()
    del state

    manifest = {"source": ckpt_path, "dtypes_in_checkpoint": dict(dtypes), "groups": {}}
    total_keys = total_params = 0
    for group, tensors in sorted(groups.items()):
        path = os.path.join(OUT, f"{group}.safetensors")
        mx.save_safetensors(path, {k: mx.array(v) for k, v in tensors.items()})
        n_params = sum(v.size for v in tensors.values())
        manifest["groups"][group] = {
            "file": os.path.basename(path), "keys": len(tensors),
            "params": n_params, "MiB": round(n_params * 4 / 2**20, 1), "ported": False,
        }
        total_keys += len(tensors)
        total_params += n_params
        print(f"  {group:24s} {len(tensors):5d} keys  {n_params / 1e6:8.2f}M params", flush=True)
    manifest["total_keys"], manifest["total_params"] = total_keys, total_params
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n{total_keys} keys, {total_params / 1e6:.2f}M params (checkpoint dtypes: {dict(dtypes)})")


if __name__ == "__main__":
    with memory_guard(max_gib=12.0):
        main(sys.argv[1] if len(sys.argv) > 1 else CKPT)
