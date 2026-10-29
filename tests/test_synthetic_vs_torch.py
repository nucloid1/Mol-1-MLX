"""Validate code whose real-checkpoint weights are numerically dead.

In the shipped checkpoint every ``attention_pair_bias`` output projection has max|w| of
about 1e-32 and ``single_transition`` is live only in blocks 40 and 43, so comparing
against captured activations says nothing about those code paths (0 matches 0). Here the
torch module itself is built with RANDOM NON-ZERO weights, run on CPU, and the same weights
are loaded into the MLX version.

    PYTHONPATH=. pytest tests/test_synthetic_vs_torch.py -q
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

from mol_mlx.blocks import AttentionPairBias, SINGLE_HEADS
from mol_mlx.layers import Transition

torch_mod = pytest.importorskip("mol.model.modules.transformer")
torch_prim = pytest.importorskip("mol.model.modules.primitives")


def randomize(module: torch.nn.Module, seed: int, scale: float = 0.2) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)


def to_mlx(module: torch.nn.Module) -> dict:
    return {k: mx.array(v.detach().float().numpy()) for k, v in module.state_dict().items()}


def rel(got, want):
    g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert g.shape == w.shape, f"{g.shape} != {w.shape}"
    return float(np.abs(g - w).max()) / float(np.abs(w).max())


@pytest.mark.parametrize("n,c_z", [(37, 320), (64, 448), (97, 320)])
def test_attention_pair_bias(n, c_z):
    ref = torch_mod.AttentionPairBias(has_s=False, create_offset_ln_z=True,
                                      n_heads=SINGLE_HEADS, c_a=384, c_z=c_z).eval()
    randomize(ref, seed=n)
    g = torch.Generator().manual_seed(100 + n)
    s = torch.randn(n, 384, generator=g)
    z = torch.randn(n, n, c_z, generator=g)
    with torch.no_grad():
        want = ref(a=s, s=None, z=z).numpy()
    assert np.abs(want).max() > 1e-2, "reference output is numerically zero; test would be vacuous"

    got = AttentionPairBias(to_mlx(ref), prefix="")(mx.array(s.numpy()), mx.array(z.numpy()))
    mx.eval(got)
    e = rel(got, want)
    print(f"\nAttentionPairBias N={n} c_z={c_z}: |out|max {np.abs(want).max():.3f}  rel err {e:.2e}")
    assert e < 2e-5


@pytest.mark.parametrize("c_in,n_fac,rows", [(384, 4, 41), (320, 4, 33)])
def test_transition(c_in, n_fac, rows):
    ref = torch_prim.Transition(c_in=c_in, n=n_fac).eval()
    randomize(ref, seed=c_in)
    x = torch.randn(rows, c_in, generator=torch.Generator().manual_seed(c_in))
    with torch.no_grad():
        want = ref(x).numpy()
    assert np.abs(want).max() > 1e-2
    got = Transition(to_mlx(ref), "")(mx.array(x.numpy()))
    mx.eval(got)
    e = rel(got, want)
    print(f"\nTransition c_in={c_in}: rel err {e:.2e}")
    assert e < 2e-5
