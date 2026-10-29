"""Mol-1 trunk blocks, the pairformer stack and the distogram head, in MLX.

Composition was read from mol/model/modules/pairformer.py (eval mode, ``inplace_safe=False``,
``pair_mask=None``; dropout is the identity in eval):

PairMixerBlock  (trunk blocks 0-7, c_z=448 -- NO triangle attention):
    z += tri_mul_out(z);  z += tri_mul_in(z);  z += pair_transition(z)
    s += attention_pair_bias(s, z);  s += single_transition(s)

PairformerBlock (trunk blocks 8-47, c_z=320, 10 triangle heads of dim 32):
    z += tri_mul_out(z);  z += tri_mul_in(z);  z += tri_att_start(z)
    z  = swap(z); z += tri_att_end(z); z = swap(z)      # ending node = transpose around a starting node
    z += pair_transition(z)
    s += attention_pair_bias(s, z);  s += single_transition(s)

PairformerStack: 8 PairMixer blocks, a 448->320 pair adapter, 40 Pairformer blocks,
a 320->448 pair adapter. The adapters touch z only.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from . import precision
from .layers import (
    Attention, Weights, key, LayerNorm, Linear, Transition, TriangleAttention, TriangleMultiplicativeUpdate,
)

N_PAIRMIXER = 8
N_BLOCKS = 48
SINGLE_HEADS, SINGLE_HEAD_DIM = 16, 24      # attention_pair_bias: c_s 384 = 16 * 24
TRIANGLE_HEAD_DIM = 32


class AttentionPairBias:
    """Single-track attention with a pair bias (``has_s=False``, the trunk case).

    ``a`` -> LN_a -> MHA(16 heads x 24), bias = linear_nobias_z(LN_z(z)) permuted to [H, N, N].
    The scores are only [H, N, N] (16 * N^2 floats, 23 MB at N=600), so MLX's dense path is
    fine here and no custom kernel is warranted.
    """

    def __init__(self, w: dict, prefix: str) -> None:
        self.ln_a = LayerNorm.load(w, key(prefix, "layernorm_a"))
        self.ln_z = LayerNorm.load(w, key(prefix, "layernorm_z"))
        self.proj_z = Linear.load(w, key(prefix, "linear_nobias_z"))
        self.attn = Attention(w, key(prefix, "attention"), SINGLE_HEADS, SINGLE_HEAD_DIM)

    def __call__(self, a: mx.array, z: mx.array) -> mx.array:
        bias = mx.transpose(self.proj_z(precision.lo(self.ln_z(z))), (2, 0, 1))   # [H, N, N]
        return self.attn(precision.lo(self.ln_a(a)), [bias])


class PairMixerBlock:
    """``single=False`` is the ``c_s=0`` variant (no single track), used by the MSA module's pair stack."""

    def __init__(self, w: dict, prefix: str, single: bool = True) -> None:
        self.single = single
        self.tri_mul_out = TriangleMultiplicativeUpdate(w, key(prefix, "tri_mul_out"), outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(w, key(prefix, "tri_mul_in"), outgoing=False)
        self.pair_transition = Transition(w, key(prefix, "pair_transition"))
        if single:
            self.attention_pair_bias = AttentionPairBias(w, key(prefix, "attention_pair_bias"))
            self.single_transition = Transition(w, key(prefix, "single_transition"))

    def __call__(self, s: mx.array, z: mx.array, transition_chunk: Optional[int] = None):
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.pair_transition(z, chunk=transition_chunk)
        return (self._single_track(s, z) if self.single else s), z

    def _single_track(self, s: mx.array, z: mx.array) -> mx.array:
        """The single track always runs fp32. Inside the stack z never reads s (only s reads z),
        so z can run in half precision on its own; s is [N, 384] -- negligible cost in fp32 --
        and reaches ~3e5 in this checkpoint, which overflows fp16 outright."""
        with precision.compute_dtype(None):
            s = s + self.attention_pair_bias(s, z)
            return s + self.single_transition(s)


class PairformerBlock(PairMixerBlock):
    def __init__(self, w: dict, prefix: str, n_heads: int, use_metal: bool = True,
                 single: bool = True) -> None:
        super().__init__(w, prefix, single=single)
        self.tri_att_start = TriangleAttention(
            w, key(prefix, "tri_att_start"), n_heads, TRIANGLE_HEAD_DIM, starting=True, use_metal=use_metal)
        self.tri_att_end = TriangleAttention(
            w, key(prefix, "tri_att_end"), n_heads, TRIANGLE_HEAD_DIM, starting=True, use_metal=use_metal)

    def __call__(self, s: mx.array, z: mx.array, transition_chunk: Optional[int] = None):
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.tri_att_start(z)
        zt = mx.swapaxes(z, -2, -3)
        zt = zt + self.tri_att_end(zt)
        z = mx.swapaxes(zt, -2, -3)
        z = z + self.pair_transition(z, chunk=transition_chunk)
        return (self._single_track(s, z) if self.single else s), z


class PairformerStack:
    """The trunk's 48-block stack with its two width adapters."""

    def __init__(self, w: dict, use_metal: bool = True) -> None:
        w = w if isinstance(w, Weights) else Weights(w)
        self.blocks = []
        for i in range(N_BLOCKS):
            prefix = f"blocks.{i}"
            if i < N_PAIRMIXER:
                self.blocks.append(PairMixerBlock(w, prefix))
            else:
                c_z = w[key(prefix, "tri_mul_out.linear_g.weight")].shape[0]
                self.blocks.append(PairformerBlock(w, prefix, n_heads=c_z // TRIANGLE_HEAD_DIM,
                                                   use_metal=use_metal))
        self.adapter_before = Linear.load(w, "adapters.before_block_8.pair_adapter")
        self.adapter_after = Linear.load(w, "adapters.after_blocks.pair_adapter")
        w.assert_all_used()    # every one of the checkpoint's pairformer_stack keys must be consumed

    def __call__(self, s: mx.array, z: mx.array, eval_every: int = 1,
                 transition_chunk: Optional[int] = None, on_block=None):
        """Run all blocks. ``eval_every`` forces evaluation each N blocks so the lazy graph
        cannot accumulate (an unevaluated 48-block graph is what reproduces the memory wall)."""
        for i, block in enumerate(self.blocks):
            if i == N_PAIRMIXER:
                z = self.adapter_before(z, low=False)
            s, z = block(s, z, transition_chunk=transition_chunk)
            if eval_every and (i + 1) % eval_every == 0:
                mx.eval(s, z)
            if on_block is not None:
                on_block(i, s, z)
        z = self.adapter_after(z, low=False)
        mx.eval(s, z)
        return s, z


class DistogramHead:
    """``logits = linear(z); logits + logits.T`` -- symmetrized over the two token axes."""

    def __init__(self, w: dict, prefix: str = "") -> None:
        self.linear = Linear.load(w, key(prefix, "linear"))

    def __call__(self, z: mx.array) -> mx.array:
        logits = self.linear(z, low=False)
        return logits + mx.swapaxes(logits, -2, -3)
