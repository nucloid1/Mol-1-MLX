"""Localize an MSA-block mismatch: run the REAL torch sub-ops and my MLX ones on the same inputs.

Stages of MSABlock 0: OuterProductMean -> z += opm -> MSAStack(m, z) -> pair_stack(z).
Each MLX stage is fed the TORCH output of the previous stage, so one wrong stage cannot hide behind
or be blamed on another.  Torch CPU, 12 GiB cap.
"""
import os
from typing import Mapping

import mlx.core as mx
import numpy as np
import torch

from mol_mlx.guard import memory_guard
from mol_mlx.msa import MSAModule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = "/Users/ethanputnam/Mol-1/data_cache/checkpoint/mol-1-base.pt"


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def rel(got, want):
    g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
    return float(np.abs(g - w).max()) / (float(np.abs(w).max()) or 1.0), tuple(g.shape) == tuple(w.shape)


with memory_guard(max_gib=12.0):
    from mol.config.base import configs as configs_base
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs
    from mol.config.model_type import model_configs
    from mol.config.parser import parse_configs
    from mol.runner.inference import InferenceRunner

    arg_str = (f"--input_json_path /dev/null --dump_dir /tmp/dbg --device cpu --load_checkpoint_path {CKPT} "
               f"--triangle_multiplicative auto --triangle_attention auto")
    base = {**configs_base, **{"data": data_configs}, **inference_configs}
    first = parse_configs(configs=dict(base), arg_str=arg_str, fill_required_with_null=True)
    deep_update(base, model_configs[first.model_name])
    configs = parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)
    model = InferenceRunner(configs).model.eval()

    r = np.load(os.path.join(ROOT, "refs", "recycle_n48.npz"))
    b = 0
    p = f"msablock{b}.cycle0"
    m_in, z_in = torch.tensor(r[f"{p}.m_in"]), torch.tensor(r[f"{p}.z_in"])
    blk = model.msa_module.blocks[b]
    mlx_blk = MSAModule(mx.load(os.path.join(ROOT, "weights", "msa_module.safetensors"))).blocks[b]
    print(f"m_in {tuple(m_in.shape)}  z_in {tuple(z_in.shape)}")

    with torch.no_grad():
        opm_t = blk.outer_product_mean_msa(m_in.clone(), inplace_safe=False, chunk_size=None)
        e, ok = rel(mlx_blk.opm(mx.array(m_in.numpy())), opm_t.numpy())
        print(f"1. OuterProductMean            rel {e:.2e}  shapes_match={ok}")

        z1 = z_in + opm_t
        m1_t = blk.msa_stack(m_in.clone(), z1.clone(), msa_chunk_size=None)   # updates its input in place
        e, ok = rel(mlx_blk.msa_stack(mx.array(m_in.numpy()), mx.array(z1.numpy())), m1_t.numpy())
        print(f"2. MSAStack (whole)            rel {e:.2e}  shapes_match={ok}")

        pwa_t = blk.msa_stack.msa_pair_weighted_averaging(m_in.clone(), z1.clone())
        mm, zz = mx.array(m_in.numpy()), mx.array(z1.numpy())
        pwa_mx = mlx_blk.msa_stack.pwa(mm, mlx_blk.msa_stack.pwa.pair_weights(zz))
        e, ok = rel(pwa_mx, pwa_t.numpy())
        print(f"   2a. pair-weighted averaging rel {e:.2e}  shapes_match={ok}")
        m_mid_t = m_in + pwa_t
        tr_t = blk.msa_stack.transition_m(m_mid_t)
        e, ok = rel(mlx_blk.msa_stack.transition(mx.array(m_mid_t.numpy())), tr_t.numpy())
        print(f"   2b. MSA transition          rel {e:.2e}  shapes_match={ok}")

        _, z2_t = blk.pair_stack(s=None, z=z1.clone(), pair_mask=None, triangle_multiplicative="torch",
                                 triangle_attention="torch", inplace_safe=False, chunk_size=None,
                                 pair_transition_chunk_size=None)
        _, z2_mx = mlx_blk.pair_stack(None, mx.array(z1.numpy()))
        e, ok = rel(z2_mx, z2_t.numpy())
        print(f"3. pair_stack (PairformerBlock) rel {e:.2e}  shapes_match={ok}")

        m_ref = r[f"{p}.m_out"]
        e, _ = rel(m1_t.numpy(), m_ref)
        print(f"\ntorch recomputed m_out vs captured m_out: rel {e:.2e}   <- must be ~0 or the reference differs")
        e, _ = rel(z2_t.numpy(), r[f"{p}.z_out"])
        print(f"torch recomputed z_out vs captured z_out: rel {e:.2e}")
