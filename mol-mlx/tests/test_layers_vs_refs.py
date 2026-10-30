"""Validate each MLX primitive against torch references captured from the real checkpoint.

Thresholds are absolute and relative max-error on fp32. The torch reference ran on
CPU, MLX runs on the GPU, so exact equality is not expected; what matters is that
error stays at accumulation-noise level and does not grow with N.

    PYTHONPATH=. pytest tests/test_layers_vs_refs.py -q
"""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from mol_mlx.layers import TriangleAttention, TriangleMultiplicativeUpdate, Transition

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "refs")
WEIGHTS = os.path.join(ROOT, "weights")
SIZES = [48, 72, 97]
# Relative to the tensor's own magnitude: block 47's pair transition legitimately
# outputs values near 1700, where 7e-4 absolute is 4e-7 relative.
RTOL = 1e-5
BLOCK_W = {0: 448, 1: 448, 8: 320, 9: 320, 47: 320}   # c_z by block index
HEADS = {448: 14, 320: 10}                            # c_z // 32, from the real weights


def _weights():
    if not hasattr(_weights, "cache"):
        _weights.cache = mx.load(os.path.join(WEIGHTS, "pairformer_stack.safetensors"))
    return _weights.cache


def _refs(n):
    key = f"r{n}"
    if not hasattr(_refs, key):
        setattr(_refs, key, np.load(os.path.join(REFS, f"n{n}.npz")))
    return getattr(_refs, key)


def err(got: mx.array, want: np.ndarray):
    g = np.asarray(got, dtype=np.float32)
    w = np.asarray(want, dtype=np.float32)
    assert g.shape == w.shape, f"shape {g.shape} != reference {w.shape}"
    a = float(np.abs(g - w).max())
    denom = float(np.abs(w).max()) or 1.0
    return a, a / denom


pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(REFS, "n48.npz")), reason="references not captured"
)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("bi", [0, 8, 47])
@pytest.mark.parametrize("op,outgoing", [("tri_mul_out", True), ("tri_mul_in", False)])
def test_triangle_multiplication(n, bi, op, outgoing):
    r = _refs(n)
    key_in, key_out = f"b{bi}.{op}.in0", f"b{bi}.{op}.out0"
    if key_in not in r:
        pytest.skip(f"{key_in} not captured")
    mod = TriangleMultiplicativeUpdate(_weights(), f"blocks.{bi}.{op}", outgoing)
    mask = r.get(f"b{bi}.{op}.kw_mask")
    out = mod(mx.array(r[key_in]), None if mask is None else mx.array(mask))
    mx.eval(out)
    a, rel = err(out, r[key_out])
    assert rel < RTOL, f"N={n} block {bi} {op}: abs {a:.3e} rel {rel:.3e}"


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("bi", [8, 47])
@pytest.mark.parametrize("op", ["tri_att_start", "tri_att_end"])
@pytest.mark.parametrize("use_metal", [False, True], ids=["dense", "metal"])
def test_triangle_attention(n, bi, op, use_metal):
    r = _refs(n)
    key_in, key_out = f"b{bi}.{op}.in0", f"b{bi}.{op}.out0"
    if key_in not in r:
        pytest.skip(f"{key_in} not captured")
    c_z = BLOCK_W[bi]
    # Both are starting=True; the block, not the module, does the transpose for the
    # ending-node variant (mol/model/modules/pairformer.py:89-98).
    mod = TriangleAttention(_weights(), f"blocks.{bi}.{op}", HEADS[c_z], 32, starting=True,
                            use_metal=use_metal)
    mask = r.get(f"b{bi}.{op}.kw_mask")
    out = mod(mx.array(r[key_in]), None if mask is None else mx.array(mask))
    mx.eval(out)
    a, rel = err(out, r[key_out])
    assert rel < RTOL, f"N={n} block {bi} {op}: abs {a:.3e} rel {rel:.3e}"


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("bi", [0, 8, 47])
@pytest.mark.parametrize("op", ["pair_transition", "single_transition"])
def test_transition(n, bi, op):
    r = _refs(n)
    key_in, key_out = f"b{bi}.{op}.in0", f"b{bi}.{op}.out0"
    if key_in not in r:
        pytest.skip(f"{key_in} not captured")
    mod = Transition(_weights(), f"blocks.{bi}.{op}")
    out = mod(mx.array(r[key_in]))
    mx.eval(out)
    a, rel = err(out, r[key_out])
    assert rel < RTOL, f"N={n} block {bi} {op}: abs {a:.3e} rel {rel:.3e}"


@pytest.mark.parametrize("n", SIZES)
def test_transition_chunking_matches(n):
    """Chunking is a memory knob. Not bitwise: MLX selects different matmul kernels by
    shape, so accumulation order differs. It must stay at rounding level."""
    r = _refs(n)
    key = "b8.pair_transition.in0"
    if key not in r:
        pytest.skip("not captured")
    mod = Transition(_weights(), "blocks.8.pair_transition")
    x = mx.array(r[key])
    full, chunked = mod(x), mod(x, chunk=16)
    mx.eval(full, chunked)
    a, rel = err(chunked, np.asarray(full, dtype=np.float32))
    assert rel < RTOL, f"chunked vs full: abs {a:.3e} rel {rel:.3e}"
