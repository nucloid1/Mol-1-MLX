"""Template embedder (AF3 Algorithm 16), in MLX.

NOT optional, even with templates off: the featurizer always emits four template slots (all-gap
restype, every mask zero when no templates are supplied) and the real model runs this embedder
on them every recycle. Its output is large, about 29 in magnitude against |z_init| ~ 36, so
skipping it changes the answer by a lot. (Validated against activations captured from the real
model; scripts/capture_recycle_refs.py.)

Spec, read from mol/model/modules/pairformer.py (eval, pair_mask = None):

    zn   = LN_z(z)                                           # once per call, shared by all slots
    at_t = concat[ dgram*m, pseudo_beta_mask*m,             # 39 + 1
                   onehot(aatype)[a,b -> b], onehot(aatype)[a,b -> a],    # 32 + 32
                   unit_vector*m, backbone_frame_mask*m ]   # 3 + 1   = 108 channels
           where m is the chain-pair mask (asym_id equality) or, when the featurizer supplies it,
           the per-slot template_pair_geometry_mask (grouped multichain structural templates)
    v_t  = linear_z(zn) + linear_a(at_t)                     # [N, N, 64]
    v_t  = LN_v( PairformerStack(c_z=64, 2 blocks, c_s=0)(v_t) )
    u    = linear_u( relu( sum_t v_t / (1e-7 + T) ) )        # [N, N, c_z]

``at_t`` does not depend on z, so it is built once per prediction (``pair_features``) and only the
z-dependent part runs each cycle.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np

from .blocks import PairformerBlock, TRIANGLE_HEAD_DIM
from .layers import LayerNorm, Linear, Weights

N_RESTYPE = 32          # len(STD_RESIDUES_WITH_GAP)


def pair_features(feat: dict) -> Optional[mx.array]:
    """Build the per-slot 108-channel pair features [T, N, N, 108] from the featurizer's arrays.

    Returns None when the feature dict carries no template slots (the model then skips the embedder).
    Expected keys: template_aatype [T,N], template_distogram [T,N,N,39], template_pseudo_beta_mask
    [T,N,N], template_unit_vector [T,N,N,3], template_backbone_frame_mask [T,N,N], asym_id [N], and
    optionally template_pair_geometry_mask [T,N,N].
    """
    if "template_aatype" not in feat:
        return None
    aatype = np.asarray(feat["template_aatype"])
    T, N = aatype.shape
    asym = np.asarray(feat["asym_id"])
    multichain = (asym[:, None] == asym[None, :]).astype(np.float32)
    geometry = feat.get("template_pair_geometry_mask")
    slots = []
    for t in range(T):
        m = np.asarray(geometry[t], np.float32) if geometry is not None else multichain
        oh = (aatype[t][:, None] == np.arange(N_RESTYPE)[None, :]).astype(np.float32)      # [N, 32]
        slots.append(np.concatenate([
            np.asarray(feat["template_distogram"][t], np.float32) * m[..., None],
            (np.asarray(feat["template_pseudo_beta_mask"][t], np.float32) * m)[..., None],
            np.broadcast_to(oh[None, :, :], (N, N, N_RESTYPE)),     # expand_at_dim(-3): value indexed by column
            np.broadcast_to(oh[:, None, :], (N, N, N_RESTYPE)),     # expand_at_dim(-2): value indexed by row
            np.asarray(feat["template_unit_vector"][t], np.float32) * m[..., None],
            (np.asarray(feat["template_backbone_frame_mask"][t], np.float32) * m)[..., None],
        ], axis=-1))
    return mx.array(np.stack(slots, axis=0))


class TemplateEmbedder:
    def __init__(self, w: dict, use_metal: bool = True) -> None:
        w = w if isinstance(w, Weights) else Weights(w)
        self.lin_z = Linear.load(w, "linear_no_bias_z")
        self.ln_z = LayerNorm.load(w, "layernorm_z")
        self.lin_a = Linear.load(w, "linear_no_bias_a")
        c = self.lin_z.weight.shape[0]
        n_blocks = len({k.split(".")[2] for k in w if k.startswith("pairformer_stack.blocks.")})
        self.blocks = [PairformerBlock(w, f"pairformer_stack.blocks.{i}", n_heads=c // TRIANGLE_HEAD_DIM,
                                       use_metal=use_metal, single=False) for i in range(n_blocks)]
        self.ln_v = LayerNorm.load(w, "layernorm_v")
        self.lin_u = Linear.load(w, "linear_no_bias_u")
        w.assert_all_used()

    def __call__(self, z: mx.array, at: mx.array, transition_chunk: Optional[int] = None) -> mx.array:
        zl = self.lin_z(self.ln_z(z))                       # [N, N, 64], shared across slots
        mx.eval(zl)
        T = at.shape[0]
        u = None
        for t in range(T):
            v = zl + self.lin_a(at[t])
            for block in self.blocks:
                _, v = block(None, v, transition_chunk=transition_chunk)
            v = self.ln_v(v)
            u = v if u is None else u + v
            mx.eval(u)
        return self.lin_u(mx.maximum(u / (1e-7 + T), 0.0))
