"""Hybrid inference: MLX runs the recycling trunk, torch runs everything else.

``install(model, weights_dir)`` replaces ONE method on a loaded Mol-1 model instance,
``get_pairformer_output``, which is the per-prediction trunk: the recycling loop with the template
embedder, the MSA module and the 48-block pairformer stack. Everything before it (input embedder,
``s_init``, ``z_init``, relative position encoding) and everything after it (distogram head,
diffusion sampler, confidence head, CIF output) is Mol-1's own torch code, untouched.

What crosses the boundary, once per prediction (not per cycle):
    torch -> MLX   s_inputs [N,449], s_init [N,384], z_init [N,N,448], the template slot features
    MLX -> torch   s [N,384], z [N,N,448]
and, per cycle, the sampled MSA rows. The model draws a random subset size and a random subset
every recycle from torch's RNG; that sampling stays in torch (the model's own function, same
arguments), so for a given seed the RNG stream and the samples are identical to a torch-only run.

By default the MLX trunk runs in a separate worker process (``isolate=True``, see worker.py): MLX is
slower (1.05-1.47x, most at small N) once torch has done its CPU-side work in the same process, and a
process boundary is the only cure found. Arrays cross through shared memory, and the per-cycle MSA samples are drawn up front
by the parent (the torch RNG stream is unchanged: nothing else consumes it inside the loop).

Falls back to the original torch method (and says so) for anything not ported: batched features,
MC-dropout, or an unexpected template layout.

Memory: the torch copies of the trunk modules are no longer used, so they are moved to the CPU
(``offload=True``), freeing ~1.5 GB of device memory; they come back only if a fallback happens.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from .msa import Trunk
from .template import pair_features
from .worker import RemoteTrunk

logger = logging.getLogger(__name__)

TEMPLATE_KEYS = ("template_aatype", "template_distogram", "template_pseudo_beta_mask",
                 "template_unit_vector", "template_backbone_frame_mask",
                 "template_pair_geometry_mask", "asym_id")
_GIB = float(1 << 30)


def auto_chunks(n_token: int) -> dict:
    """Memory-bounding chunk sizes from the token count (they change peak memory, never results).

    Each bound keeps one chunk's working tensor near 128-256 MB, which is far below the
    unchunked size at the 400-600 token range this is tuned for.
    """
    opm = max(1, int(128 * 2**20 // (n_token * 1024 * 4)))          # [ci, N, 1024] floats
    trans = max(1, int(256 * 2**20 // (n_token * 1792 * 4)))        # [rows, N, 4*448] floats
    # MSA stack row chunk: its transition builds [rows, N, 2*4*192] floats (a and b, concatenated). A
    # real alignment can put thousands of rows in one cycle (Mol-1 samples up to 16384), and 2048 rows at
    # N=586 is a 7.4 GiB intermediate; hold it to ~512 MiB (never below 32 rows, never above 2048).
    msa = min(2048, max(32, int(512 * 2**20 // (n_token * 1536 * 4))))
    return {"opm_chunk": min(opm, n_token), "transition_chunk": min(trans, n_token), "msa_chunk": msa}


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy()


def install(model: Any, weights_dir: str, offload: bool = True, use_metal: bool = True,
            isolate: bool = True, worker_cap_gib: Optional[float] = None):
    """Patch ``model.get_pairformer_output`` to run the trunk in MLX. Returns the Trunk (in-process)
    or the RemoteTrunk (``isolate=True``, the default; ``worker_cap_gib`` bounds the worker)."""
    import torch
    from mol.model.utils import sample_msa_feature_dict_random_without_replacement

    trunk = (RemoteTrunk(weights_dir, use_metal=use_metal, cap_gib=worker_cap_gib) if isolate
             else Trunk(weights_dir, use_metal=use_metal))
    original = model.get_pairformer_output
    torch_trunk = (model.pairformer_stack, model.msa_module, model.template_embedder)
    state = {"offloaded": False}

    def device_of() -> torch.device:
        return model.linear_no_bias_sinit.weight.device

    def fall_back(reason: str, **kwargs):
        logger.warning("MLX trunk not used (%s); running the torch trunk for this prediction", reason)
        if state["offloaded"]:
            for m in torch_trunk:
                m.to(device_of())
            state["offloaded"] = False
        return original(**kwargs)

    def get_pairformer_output(input_feature_dict, N_cycle, inplace_safe=False, chunk_size=None,
                              mc_dropout=False, mc_dropout_rate=0.4):
        kw = dict(input_feature_dict=input_feature_dict, N_cycle=N_cycle, inplace_safe=inplace_safe,
                  chunk_size=chunk_size, mc_dropout=mc_dropout, mc_dropout_rate=mc_dropout_rate)
        f = input_feature_dict
        if mc_dropout:
            return fall_back("mc_dropout", **kw)
        if f["restype"].ndim != 2:
            return fall_back("batched features", **kw)
        if "constraint_feature" in f:
            return fall_back("constraint features", **kw)

        device = device_of()
        # ---- prelude, exactly as Mol.get_pairformer_output does it (torch, once) ----
        s_inputs = model.input_embedder(f, inplace_safe=False, chunk_size=chunk_size)
        s_init = model.linear_no_bias_sinit(s_inputs)
        z_init = (model.linear_no_bias_zinit1(s_init)[..., None, :]
                  + model.linear_no_bias_zinit2(s_init)[..., None, :, :])
        z_init = z_init + model.relative_position_encoding(f["relp"])
        z_init = z_init + model.linear_no_bias_token_bond(f["token_bonds"].unsqueeze(dim=-1))
        n_token = int(s_inputs.shape[0])

        # ---- MSA sampling stays in torch: same function, same args, same RNG stream ----
        msa_module = model.msa_module
        has_msa = msa_module.n_blocks >= 1 and "msa" in f and f["msa"].dim() >= 2

        def sample_fn(cycle: int) -> Optional[dict]:
            if not has_msa:
                return None
            d = sample_msa_feature_dict_random_without_replacement(
                feat_dict=f,
                dim_dict={name: -2 for name in msa_module.input_feature},
                cutoff=msa_module.msa_configs["test_cutoff"],
                lower_bound=msa_module.msa_configs["test_lowerb"],
                strategy=msa_module.msa_configs["strategy"],
            )
            out = {k: _np(d[k]) for k in ("msa", "has_deletion", "deletion_value")}
            # Pre-drawn for every cycle and sent to the worker, so keep deep alignments compact. Token ids are
            # < 32 and the deletion flag is 0/1, so uint8 is exact (checked); deletion_value stays float32.
            for k in ("msa", "has_deletion"):
                narrow = out[k].astype(np.uint8)
                if not np.array_equal(narrow, out[k]):
                    raise ValueError(f"MSA feature {k!r} is not exactly representable as uint8")
                out[k] = narrow
            return out

        has_template = model.template_embedder.n_blocks > 0 and "template_aatype" in f
        template_feat = {k: _np(f[k]) for k in TEMPLATE_KEYS if k in f} if has_template else None
        template_at = pair_features(template_feat) if (has_template and not isolate) else None

        if offload and not state["offloaded"]:
            for m in torch_trunk:
                m.to("cpu")
            state["offloaded"] = True
            if device.type == "mps":
                torch.mps.empty_cache()

        if isolate:
            # ---- the MLX trunk, in the worker process ----
            samples = [sample_fn(c) for c in range(int(N_cycle))]
            s_np, z_np = trunk(_np(s_inputs), _np(s_init), _np(z_init), int(N_cycle), samples,
                               template_feat, **auto_chunks(n_token))
            return s_inputs, torch.from_numpy(s_np).to(device), torch.from_numpy(z_np).to(device)

        # ---- the MLX trunk, in this process ----
        s, z = trunk(
            mx.array(_np(s_inputs)), mx.array(_np(s_init)), mx.array(_np(z_init)), int(N_cycle),
            sample_fn=sample_fn, template_at=template_at, **auto_chunks(n_token),
        )
        mx.eval(s, z)
        s_t = torch.from_numpy(np.array(s)).to(device)
        z_t = torch.from_numpy(np.array(z)).to(device)
        del s, z, template_at
        mx.clear_cache()
        return s_inputs, s_t, z_t

    model.get_pairformer_output = get_pairformer_output
    logger.info("MLX trunk installed (offload=%s, metal tri-attn=%s, isolated worker=%s)",
                offload, use_metal, isolate)
    return trunk
