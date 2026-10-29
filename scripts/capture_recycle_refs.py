"""Reference activations for the recycle loop and the MSA module, from the REAL model on torch CPU.

Runs `Mol.get_pairformer_output` itself (not a re-implementation) for three cycles on a two-chain
system whose target chain has a small synthetic MSA (24 rows: substitutions, gaps and lowercase
insertions, so has_deletion / deletion_value are non-trivial). Hooks record:

  * s_inputs, s_init, z_init                           (the prelude the hybrid keeps in torch)
  * per msa_module call: the SAMPLED msa features (the model draws a random subset every cycle
    with torch's RNG, so the exact sample is recorded and replayed into MLX), z in/out
  * N=48 only: every MSA block's (m, z) in/out
  * the stack's in/out per cycle, and the final s, z, distogram

Writes refs/recycle_n{N}.npz. Small N only (48, 97), 12 GiB cap.

    PYTHONPATH=/Users/ethanputnam/Mol-1-MLX python capture_recycle_refs.py 48 97
"""
import json
import os
import random
import sys
import tempfile
from typing import Mapping

import numpy as np
import torch

from mol_mlx.guard import memory_guard

CKPT = "/Users/ethanputnam/Mol-1/data_cache/checkpoint/mol-1-base.pt"
REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "refs")
N_CYCLE = 3
MSA_ROWS = 24
BLOCKS_DEBUG_N = 48

TARGET_POOL = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
               "SLEEKKGNYVVTDHGSCVRGFGTSEMFKRLEELGFKVIRVWQNPEHAPSDKRLGSTAYL")
BINDER = "SLEEKKGNYVVTDHGSCVRGFGTS"
AA = "ACDEFGHIKLMNPQRSTVWY"


def synthetic_a3m(query: str, rows: int, seed: int) -> str:
    """Query plus homolog-like rows: substitutions, gaps, and lowercase insertions (a3m)."""
    rng = random.Random(seed)
    out = [f">query\n{query}"]
    for r in range(rows - 1):
        sub_rate = rng.uniform(0.1, 0.4)
        chars = []
        for ch in query:
            u = rng.random()
            if u < 0.06:
                chars.append("-")
            elif u < 0.06 + sub_rate:
                chars.append(rng.choice(AA))
            else:
                chars.append(ch)
            if rng.random() < 0.03:
                chars.append(rng.choice(AA).lower())     # an insertion: counted in deletion_value
        out.append(f">hit{r}\n{''.join(chars)}")
    return "\n".join(out) + "\n"


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def to_np(x):
    """Always COPY. A CPU tensor and its numpy view share memory, and torch updates activations in
    place later (the next MSA block's `m`), which would silently rewrite a stored 'output'."""
    return x.detach().cpu().numpy().copy() if isinstance(x, torch.Tensor) else None


def main(sizes):
    import mol.model.modules.pairformer as pf
    from mol.config.base import configs as configs_base
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs
    from mol.config.model_type import model_configs
    from mol.config.parser import parse_configs
    from mol.data.inference.infer_dataloader import InferenceDataset
    from mol.runner.inference import InferenceRunner

    os.makedirs(REFS, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="recycle_refs_")
    for n_total in sizes:
        target = (TARGET_POOL * 2)[: n_total - len(BINDER)]
        open(f"{tmp}/{n_total}_A.a3m", "w").write(synthetic_a3m(target, MSA_ROWS, seed=n_total))
        open(f"{tmp}/{n_total}_B.a3m", "w").write(f">query\n{BINDER}\n")
        job = [{"name": f"rec{n_total}", "sequences": [
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
        assert feats["restype"].shape[0] == n_total
        n_msa = int(feats["msa"].shape[0])
        print(f"N={n_total}: msa rows in the features: {n_msa}", flush=True)

        store = {}
        msa_calls = {"n": 0}
        sample_log = []
        orig_sampler = pf.sample_msa_feature_dict_random_without_replacement

        def logging_sampler(**kw):
            out = orig_sampler(**kw)
            sample_log.append({k: to_np(v) for k, v in out.items()})
            return out

        pf.sample_msa_feature_dict_random_without_replacement = logging_sampler
        handles = []

        # INPUTS MUST BE SNAPSHOT BEFORE THE FORWARD RUNS. A forward hook fires afterwards, and torch's
        # MSAStack (and others) update tensors in place, so an input recorded in a forward hook can
        # already be the OUTPUT. (The first version of this script did exactly that and produced
        # references that no implementation, including torch itself, could reproduce.)
        pending = {}

        def pre_msa(mod, args, kwargs):
            c = msa_calls["n"]
            z_in = args[1] if len(args) > 1 else kwargs["z"]
            s_in = args[2] if len(args) > 2 else kwargs["s_inputs"]
            pending[("msa", c)] = (to_np(z_in.clone()), to_np(s_in.clone()))
        handles.append(model.msa_module.register_forward_pre_hook(pre_msa, with_kwargs=True))

        def on_msa(mod, args, kwargs, out):
            c = msa_calls["n"]
            msa_calls["n"] += 1
            z_in, s_in = pending.pop(("msa", c))
            store[f"msa.cycle{c}.z_in"], store[f"msa.cycle{c}.z_out"] = z_in, to_np(out)
            store[f"msa.cycle{c}.s_inputs"] = s_in
            for k, v in sample_log[-1].items():
                store[f"msa.cycle{c}.sample.{k}"] = v
        handles.append(model.msa_module.register_forward_hook(on_msa, with_kwargs=True))

        if n_total == BLOCKS_DEBUG_N:
            def block_pre(b):
                def hook(mod, args, kwargs):
                    c = msa_calls["n"]
                    m_in = args[0] if args else kwargs["m"]
                    z_in = args[1] if len(args) > 1 else kwargs["z"]
                    pending[(b, c)] = (to_np(m_in.clone()), to_np(z_in.clone()))
                return hook

            def block_hook(b):
                def hook(mod, args, kwargs, out):
                    c = msa_calls["n"]            # the msa_module hook has not fired for this cycle yet
                    p = f"msablock{b}.cycle{c}"
                    store[f"{p}.m_in"], store[f"{p}.z_in"] = pending.pop((b, c))
                    if out[0] is not None:
                        store[f"{p}.m_out"] = to_np(out[0])
                    store[f"{p}.z_out"] = to_np(out[1])
                return hook
            for b, blk in enumerate(model.msa_module.blocks):
                handles.append(blk.register_forward_pre_hook(block_pre(b), with_kwargs=True))
                handles.append(blk.register_forward_hook(block_hook(b), with_kwargs=True))

        # The template embedder runs EVERY cycle even with templates off: the featurizer emits four
        # all-gap template slots, and the embedder reads z through a layernorm and a linear.
        tmpl_calls = {"n": 0}

        def pre_tmpl(mod, args, kwargs):
            z_in = args[1] if len(args) > 1 else kwargs["z"]
            pending[("tmpl", tmpl_calls["n"])] = to_np(z_in.clone())
        handles.append(model.template_embedder.register_forward_pre_hook(pre_tmpl, with_kwargs=True))

        def on_tmpl(mod, args, kwargs, out):
            c = tmpl_calls["n"]
            tmpl_calls["n"] += 1
            store[f"tmpl.cycle{c}.z_in"] = pending.pop(("tmpl", c))
            store[f"tmpl.cycle{c}.out"] = to_np(out) if isinstance(out, torch.Tensor) else np.zeros(1, np.float32)
        handles.append(model.template_embedder.register_forward_hook(on_tmpl, with_kwargs=True))

        stack_calls = {"n": 0}

        def pre_stack(mod, args, kwargs):
            pending[("stack", stack_calls["n"])] = (to_np(args[0].clone()), to_np(args[1].clone()))
        handles.append(model.pairformer_stack.register_forward_pre_hook(pre_stack, with_kwargs=True))

        def on_stack(mod, args, kwargs, out):
            c = stack_calls["n"]
            stack_calls["n"] += 1
            store[f"cycle{c}.stack.in0"], store[f"cycle{c}.stack.in1"] = pending.pop(("stack", c))
            store[f"cycle{c}.stack.out0"], store[f"cycle{c}.stack.out1"] = to_np(out[0]), to_np(out[1])
        handles.append(model.pairformer_stack.register_forward_hook(on_stack, with_kwargs=True))

        from mol.model.mol import update_input_feature_dict
        torch.manual_seed(0)
        with torch.no_grad():
            f = model.relative_position_encoding.generate_relp(dict(feats))
            f = update_input_feature_dict(f)
            s_inputs = model.input_embedder(f, inplace_safe=False, chunk_size=None)
            s_init = model.linear_no_bias_sinit(s_inputs)
            z_init = (model.linear_no_bias_zinit1(s_init)[..., None, :]
                      + model.linear_no_bias_zinit2(s_init)[..., None, :, :])
            z_init = (z_init + model.relative_position_encoding(f["relp"])
                      + model.linear_no_bias_token_bond(f["token_bonds"].unsqueeze(-1)))
            store["s_inputs"], store["s_init"], store["z_init"] = to_np(s_inputs), to_np(s_init), to_np(z_init)
            for k, v in f.items():
                if k.startswith("template_") or k == "asym_id":
                    store[f"feat.{k}"] = to_np(v)
            torch.manual_seed(0)
            _, s, z = model.get_pairformer_output(f, N_cycle=N_CYCLE, inplace_safe=False, chunk_size=None)
            store["final.s"], store["final.z"] = to_np(s), to_np(z)
            store["final.distogram"] = to_np(model.distogram_head(z))
        pf.sample_msa_feature_dict_random_without_replacement = orig_sampler
        for h in handles:
            h.remove()

        sizes_per_cycle = [int(c["msa"].shape[0]) for c in sample_log]
        print(f"  sampled MSA rows per cycle: {sizes_per_cycle} (of {n_msa})", flush=True)
        store["n_cycle"] = np.array(N_CYCLE)
        out_path = os.path.join(REFS, f"recycle_n{n_total}.npz")
        np.savez(out_path, **store)
        print(f"  {len(store)} tensors, {os.path.getsize(out_path) / 2**20:.0f} MiB -> {out_path}", flush=True)
        store.clear()
        del runner, model, feats


if __name__ == "__main__":
    with memory_guard(max_gib=12.0):
        main([int(a) for a in sys.argv[1:]] or [48])
