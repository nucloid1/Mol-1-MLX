"""Capture ground-truth activations from the real Mol-1 checkpoint, on torch CPU.

CPU on purpose: the GPU stays free for MLX, only one framework's tensors are ever
resident on it, and CPU fp32 is the cleanest reference. Small N on purpose too
(48 / 72 / 97 -- 97 is not a multiple of any tile size, so edge bugs show).

Runs the trunk's recycling loop for two cycles and records, per cycle:
  * pairformer_stack input (s, z) and output (s, z)
and, on the SECOND cycle only (where z is no longer near-zero), the input and output of
every sub-module of the blocks listed in SUB_BLOCKS, plus whole-block in/out for
ALL_BLOCKS_N sizes.

Writes refs/n{N}.npz. Run from anywhere:
    PYTHONPATH=/Users/ethanputnam/Mol-1-MLX python capture_refs.py 48 72 97
"""
import json
import os
import sys
import tempfile
from typing import Mapping

import numpy as np
import torch

from mol_mlx.guard import memory_guard

SUB_BLOCKS = (0, 1, 8, 9, 47)          # a PairMixer pair, a Pairformer pair, and the last block
SUB_OPS = ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end",
           "pair_transition", "attention_pair_bias", "single_transition")
ALL_BLOCKS_N = 48                      # whole-block in/out for every block, at this N only
CKPT = "/Users/ethanputnam/Mol-1/data_cache/checkpoint/mol-1-base.pt"
REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "refs")

TARGET_POOL = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
               "SLEEKKGNYVVTDHGSCVRGFGTSEMFKRLEELGFKVIRVWQNPEHAPSDKRLGSTAYL")
BINDER = "SLEEKKGNYVVTDHGSCVRGFGTS"   # 24 residues


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def to_np(x):
    return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else None


def main(sizes):
    from mol.config.base import configs as configs_base
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs
    from mol.config.model_type import model_configs
    from mol.config.parser import parse_configs
    from mol.data.inference.infer_dataloader import InferenceDataset
    from mol.model.mol import update_input_feature_dict
    from mol.runner.inference import InferenceRunner

    os.makedirs(REFS, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="refs_")

    for n_total in sizes:
        n_target = n_total - len(BINDER)
        target = (TARGET_POOL * 2)[:n_target]
        for ch, seq in (("A", target), ("B", BINDER)):
            open(f"{tmp}/{n_total}_{ch}.a3m", "w").write(f">query\n{seq}\n")
        job = [{"name": f"ref{n_total}", "sequences": [
            {"proteinChain": {"id": ["A"], "sequence": target, "count": 1,
                              "unpairedMsaPath": f"{tmp}/{n_total}_A.a3m"}},
            {"proteinChain": {"id": ["B"], "sequence": BINDER, "count": 1,
                              "unpairedMsaPath": f"{tmp}/{n_total}_B.a3m"}}]}]
        path = f"{tmp}/in_{n_total}.json"
        json.dump(job, open(path, "w"))
        arg_str = (f"--input_json_path {path} --dump_dir {tmp}/out --use_msa true --use_template false "
                   f"--device cpu --load_checkpoint_path {CKPT} "
                   f"--triangle_multiplicative auto --triangle_attention auto")
        base = {**configs_base, **{"data": data_configs}, **inference_configs}
        first = parse_configs(configs=dict(base), arg_str=arg_str, fill_required_with_null=True)
        deep_update(base, model_configs[first.model_name])
        configs = parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)

        runner = InferenceRunner(configs)
        model = runner.model.eval()
        data, _, err = InferenceDataset(configs)[0]
        assert not err, err
        feats = data["input_feature_dict"]
        assert feats["restype"].shape[0] == n_total, (feats["restype"].shape, n_total)

        store = {}
        state = {"call": 0}
        handles = []

        def record(prefix, args, kwargs, out):
            for i, a in enumerate(args):
                if (v := to_np(a)) is not None:
                    store[f"{prefix}.in{i}"] = v
            for k, a in kwargs.items():
                if (v := to_np(a)) is not None:
                    store[f"{prefix}.kw_{k}"] = v
            outs = out if isinstance(out, (tuple, list)) else (out,)
            for i, o in enumerate(outs):
                if (v := to_np(o)) is not None:
                    store[f"{prefix}.out{i}"] = v

        stack = model.pairformer_stack

        def stack_hook(mod, args, kwargs, out):
            record(f"cycle{state['call']}.stack", args, kwargs, out)
            state["call"] += 1

        handles.append(stack.register_forward_hook(stack_hook, with_kwargs=True))

        def sub_hook(name):
            def hook(mod, args, kwargs, out):
                if state["call"] == 1:   # the second cycle (the stack hook has not fired for it yet)
                    record(name, args, kwargs, out)
            return hook

        for bi in SUB_BLOCKS:
            block = stack.blocks[bi]
            for op in SUB_OPS:
                if hasattr(block, op):
                    handles.append(getattr(block, op).register_forward_hook(
                        sub_hook(f"b{bi}.{op}"), with_kwargs=True))
        for bi, block in enumerate(stack.blocks):
            if bi in SUB_BLOCKS or n_total == ALL_BLOCKS_N:
                handles.append(block.register_forward_hook(sub_hook(f"b{bi}.block"), with_kwargs=True))

        with torch.no_grad():
            f = model.relative_position_encoding.generate_relp(dict(feats))
            f = update_input_feature_dict(f)
            s_inputs = model.input_embedder(f, inplace_safe=False, chunk_size=None)
            store["s_inputs"] = to_np(s_inputs)
            # Let the model's own recycling loop run for two cycles; the hooks do the capturing.
            s_inputs2, s, z = model.get_pairformer_output(f, N_cycle=2, inplace_safe=False, chunk_size=None)
            store["final.s"], store["final.z"] = to_np(s), to_np(z)
            store["final.distogram"] = to_np(model.distogram_head(z))
        for h in handles:
            h.remove()

        out_path = os.path.join(REFS, f"n{n_total}.npz")
        np.savez(out_path, **store)
        mib = os.path.getsize(out_path) / 2**20
        print(f"N={n_total}: {len(store)} tensors, {mib:.0f} MiB -> {out_path}", flush=True)
        store.clear()      # free the arrays now; `del store` would unbind a name the hooks close over
        del runner, model, feats


if __name__ == "__main__":
    with memory_guard(max_gib=12.0):
        main([int(a) for a in sys.argv[1:]] or [48])
