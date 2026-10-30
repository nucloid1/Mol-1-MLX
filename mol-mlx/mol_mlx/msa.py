"""MSA module and the recycling loop, in MLX.

Specs were read from mol/model/modules/pairformer.py and mol/model/triangular/layers.py, then
validated against activations captured from the real model (scripts/capture_recycle_refs.py).

MSAModule (4 blocks; c_m = 192, c_z = 448):
    msa_sample = linear_m([one_hot(msa, 32), has_deletion, deletion_value]) + linear_s(s_inputs)
    per block:  z += OuterProductMean(m)
                m  = MSAStack(m, z)          # not in the last block
                z  = PairformerBlock(c_s = 0)(z)      # a FULL 448-wide pair block, triangle attention included

The MSA module's pair stack is a whole Pairformer block, so each recycle carries four more of them
on top of the trunk; this reuses the validated triangle ops.

The sampled MSA rows come from the caller. The model draws a uniformly random subset size and a random
subset every recycle with torch's RNG; the hybrid keeps that sampling in torch so the RNG stream is
identical, and replays the sample here.

Memory: ``m`` is [M, N, 192] floats. The row-wise ops (pair-weighted averaging, the MSA transition)
run in chunks of ``msa_chunk`` rows; the outer product mean runs in chunks of ``opm_chunk`` output
rows. ``m`` itself is not chunked (neither is it in torch), so a very large MSA is still large.
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx

from .blocks import PairformerBlock, PairformerStack, TRIANGLE_HEAD_DIM
from .layers import LayerNorm, Linear, Transition, Weights, key
from .template import TemplateEmbedder

N_BLOCKS = 4
OPM_EPS = 1e-3
PWA_HEADS = 8       # MSAPairWeightedAveraging: 8 heads x c=8 = 64
PWA_C = 8


class OuterProductMean:
    """``out[i,j] = (linear_out(vec(sum_s a[s,i,:] (x) b[s,j,:])) ) / (M + eps)``.

    The division is applied AFTER linear_out, whose bias is therefore divided too; that is what
    the torch code does, so it is reproduced exactly.
    """

    def __init__(self, w: dict, prefix: str) -> None:
        self.ln = LayerNorm.load(w, key(prefix, "layer_norm"))
        self.l1 = Linear.load(w, key(prefix, "linear_1"))
        self.l2 = Linear.load(w, key(prefix, "linear_2"))
        self.out = Linear.load(w, key(prefix, "linear_out"))

    def __call__(self, m: mx.array, chunk: Optional[int] = None) -> mx.array:
        M, N, _ = m.shape
        ln = self.ln(m)
        a = self.l1(ln)                       # [M, N, c]
        b = self.l2(ln)
        c = a.shape[-1]
        b2 = b.reshape(M, N * c)              # [M, N*c]
        norm = M + OPM_EPS
        rows = N if chunk is None else chunk
        parts = []
        for i in range(0, N, rows):
            a_c = a[:, i:i + rows, :]         # [M, ci, c]
            ci = a_c.shape[1]
            outer = a_c.reshape(M, ci * c).T @ b2                 # [ci*c, N*c]
            outer = outer.reshape(ci, c, N, c)
            outer = mx.transpose(outer, (0, 2, 1, 3)).reshape(ci, N, c * c)   # [ci, N, c*c]
            parts.append(self.out(outer) / norm)
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)


class MSAPairWeightedAveraging:
    def __init__(self, w: dict, prefix: str) -> None:
        self.ln_m = LayerNorm.load(w, key(prefix, "layernorm_m"))
        self.ln_z = LayerNorm.load(w, key(prefix, "layernorm_z"))
        self.mv = Linear.load(w, key(prefix, "linear_no_bias_mv"))
        self.mg = Linear.load(w, key(prefix, "linear_no_bias_mg"))
        self.zb = Linear.load(w, key(prefix, "linear_no_bias_z"))
        self.out = Linear.load(w, key(prefix, "linear_no_bias_out"))

    def pair_weights(self, z: mx.array) -> mx.array:
        """softmax over j of the pair bias: [H, N(i), N(j)]. Row-independent, so computed once."""
        b = self.zb(self.ln_z(z))                         # [N, N, H]
        return mx.transpose(mx.softmax(b, axis=-2), (2, 0, 1))

    def __call__(self, m: mx.array, w_ij: mx.array) -> mx.array:
        M, N, _ = m.shape
        H, c = PWA_HEADS, PWA_C
        mn = self.ln_m(m)
        v = self.mv(mn).reshape(M, N, H, c)
        g = mx.sigmoid(self.mg(mn)).reshape(M, N, H, c)
        # wv[m,i,h,c] = sum_j w[i,j,h] v[m,j,h,c]  as one batched matmul over heads
        v_r = mx.transpose(v, (2, 1, 0, 3)).reshape(H, N, M * c)     # [H, N(j), M*c]
        wv = (w_ij @ v_r).reshape(H, N, M, c)                         # [H, N(i), M, c]
        wv = mx.transpose(wv, (2, 1, 0, 3))                           # [M, N, H, c]
        return self.out((g * wv).reshape(M, N, H * c))


class MSAStack:
    def __init__(self, w: dict, prefix: str) -> None:
        self.pwa = MSAPairWeightedAveraging(w, key(prefix, "msa_pair_weighted_averaging"))
        self.transition = Transition(w, key(prefix, "transition_m"))

    def __call__(self, m: mx.array, z: mx.array, msa_chunk: Optional[int] = None) -> mx.array:
        w_ij = self.pwa.pair_weights(z)
        M = m.shape[0]
        rows = M if msa_chunk is None else msa_chunk
        parts = []
        for i in range(0, M, rows):
            mc = m[i:i + rows]
            mc = mc + self.pwa(mc, w_ij)
            mc = mc + self.transition(mc)
            parts.append(mc)
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)


class MSABlock:
    def __init__(self, w: dict, prefix: str, is_last: bool, use_metal: bool = True) -> None:
        self.is_last = is_last
        self.opm = OuterProductMean(w, key(prefix, "outer_product_mean_msa"))
        if not is_last:
            self.msa_stack = MSAStack(w, key(prefix, "msa_stack"))
        c_z = w[key(prefix, "pair_stack.tri_mul_out.linear_g.weight")].shape[0]
        self.pair_stack = PairformerBlock(w, key(prefix, "pair_stack"), n_heads=c_z // TRIANGLE_HEAD_DIM,
                                          use_metal=use_metal, single=False)

    def __call__(self, m: mx.array, z: mx.array, msa_chunk: Optional[int] = None,
                 opm_chunk: Optional[int] = None, transition_chunk: Optional[int] = None):
        z = z + self.opm(m, chunk=opm_chunk)
        if not self.is_last:
            m = self.msa_stack(m, z, msa_chunk=msa_chunk)
        _, z = self.pair_stack(None, z, transition_chunk=transition_chunk)
        return (None if self.is_last else m), z


class MSAModule:
    def __init__(self, w: dict, use_metal: bool = True) -> None:
        w = w if isinstance(w, Weights) else Weights(w)
        self.lin_m = Linear.load(w, "linear_no_bias_m")
        self.lin_s = Linear.load(w, "linear_no_bias_s")
        self.blocks = [MSABlock(w, f"blocks.{i}", is_last=(i == N_BLOCKS - 1), use_metal=use_metal)
                       for i in range(N_BLOCKS)]
        w.assert_all_used()

    @staticmethod
    def featurize(sample: dict) -> mx.array:
        """The sampled MSA features -> [M, N, 34] = one_hot(msa, 32) | has_deletion | deletion_value."""
        msa = mx.array(sample["msa"]).astype(mx.int32)
        one_hot = (msa[..., None] == mx.arange(32)).astype(mx.float32)
        hd = mx.array(sample["has_deletion"]).astype(mx.float32)[..., None]
        dv = mx.array(sample["deletion_value"]).astype(mx.float32)[..., None]
        return mx.concatenate([one_hot, hd, dv], axis=-1)

    def __call__(self, sample: dict, z: mx.array, s_inputs: mx.array, msa_chunk: Optional[int] = None,
                 opm_chunk: Optional[int] = None, transition_chunk: Optional[int] = None) -> mx.array:
        m = self.lin_m(self.featurize(sample)) + self.lin_s(s_inputs)
        for block in self.blocks:
            m, z = block(m, z, msa_chunk=msa_chunk, opm_chunk=opm_chunk, transition_chunk=transition_chunk)
            if m is None:
                mx.eval(z)
            else:
                mx.eval(m, z)
        return z


class Trunk:
    """The recycling loop (Mol.get_pairformer_output after the prelude).

    The prelude (input embedder, s_init, z_init) runs once per prediction and stays in torch; this
    takes its results. Order per cycle, as in the model: recycle linear, template embedder, MSA
    module, single update, pairformer stack. The template embedder runs even with templates off
    (the featurizer supplies four all-gap slots) and its output is large; see template.py.
    """

    def __init__(self, weights_dir: str, use_metal: bool = True) -> None:
        import os

        def load(name):
            return Weights(mx.load(os.path.join(weights_dir, f"{name}.safetensors")))

        self.stack = PairformerStack(load("pairformer_stack"), use_metal=use_metal)
        self.msa = MSAModule(load("msa_module"), use_metal=use_metal)
        self.template = TemplateEmbedder(load("template_embedder"), use_metal=use_metal)
        self.ln_z = LayerNorm.load(load("layernorm_z_cycle"), "")
        self.lin_z = Linear.load(load("linear_no_bias_z_cycle"), "")
        self.ln_s = LayerNorm.load(load("layernorm_s"), "")
        self.lin_s = Linear.load(load("linear_no_bias_s"), "")

    def __call__(self, s_inputs: mx.array, s_init: mx.array, z_init: mx.array, n_cycle: int,
                 sample_fn: Callable[[int], dict], template_at: Optional[mx.array] = None,
                 msa_chunk: Optional[int] = None, opm_chunk: Optional[int] = None,
                 transition_chunk: Optional[int] = None, on_cycle: Optional[Callable] = None):
        """``template_at``: [T, N, N, 108] from ``template.pair_features`` (None skips the embedder,
        which matches the model only when the feature dict has no template slots at all)."""
        s = mx.zeros_like(s_init)
        z = mx.zeros_like(z_init)
        for cycle in range(n_cycle):
            z = z_init + self.lin_z(self.ln_z(z))
            if template_at is not None:
                z = z + self.template(z, template_at, transition_chunk=transition_chunk)
            sample = sample_fn(cycle)
            if sample is not None:      # None: the feature dict has no MSA, and the model skips the module
                z = self.msa(sample, z, s_inputs, msa_chunk=msa_chunk, opm_chunk=opm_chunk,
                             transition_chunk=transition_chunk)
            s = s_init + self.lin_s(self.ln_s(s))
            s, z = self.stack(s, z, transition_chunk=transition_chunk)
            if on_cycle is not None:
                on_cycle(cycle, s, z)
        return s, z
