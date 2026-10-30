"""Blocks, the full stack and the distogram head against the real checkpoint's activations.

Errors are measured relative to each tensor's own magnitude. Per-block errors at N=48 are
printed so growth through the 48-block stack is visible rather than hidden in one number.

    PYTHONPATH=. pytest tests/test_stack_vs_refs.py -q -s
"""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from mol_mlx.blocks import PairformerStack, DistogramHead

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [48, 72, 97]
BLOCK_RTOL = 2e-5       # one block, fp32
STACK_RTOL = 2e-4       # 48 chained blocks, two cycles of accumulated fp32 error

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "refs", "n48.npz")), reason="references not captured"
)


def rel(got, want):
    g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert g.shape == w.shape, f"{g.shape} != {w.shape}"
    return float(np.abs(g - w).max()) / (float(np.abs(w).max()) or 1.0)


@pytest.fixture(scope="module")
def weights():
    return mx.load(os.path.join(ROOT, "weights", "pairformer_stack.safetensors"))


@pytest.fixture(scope="module")
def stack(weights):
    return PairformerStack(weights)


def refs(n):
    return np.load(os.path.join(ROOT, "refs", f"n{n}.npz"))


def test_every_block_at_n48(stack):
    """Each block fed the reference's own input, so errors do not compound across blocks."""
    r = refs(48)
    worst = (0.0, -1)
    rows = []
    for i, block in enumerate(stack.blocks):
        s_in, z_in = mx.array(r[f"b{i}.block.in0"]), mx.array(r[f"b{i}.block.in1"])
        s_out, z_out = block(s_in, z_in)
        mx.eval(s_out, z_out)
        es, ez = rel(s_out, r[f"b{i}.block.out0"]), rel(z_out, r[f"b{i}.block.out1"])
        rows.append((i, es, ez))
        worst = max(worst, (max(es, ez), i))
    print("\nper-block relative error (s, z):")
    for i, es, ez in rows:
        kind = "PairMixer " if i < 8 else "Pairformer"
        print(f"  block {i:2d} {kind}  s {es:.2e}  z {ez:.2e}")
    assert worst[0] < BLOCK_RTOL, f"worst block {worst[1]}: {worst[0]:.3e}"


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("cycle", [0, 1])
def test_full_stack(stack, n, cycle):
    r = refs(n)
    s_in, z_in = mx.array(r[f"cycle{cycle}.stack.in0"]), mx.array(r[f"cycle{cycle}.stack.in1"])
    s_out, z_out = stack(s_in, z_in)
    es, ez = rel(s_out, r[f"cycle{cycle}.stack.out0"]), rel(z_out, r[f"cycle{cycle}.stack.out1"])
    print(f"\nN={n} cycle {cycle}: s rel {es:.2e}  z rel {ez:.2e}")
    assert es < STACK_RTOL and ez < STACK_RTOL, f"s {es:.3e} z {ez:.3e}"


@pytest.mark.parametrize("n", SIZES)
def test_distogram_head(n):
    r = refs(n)
    head = DistogramHead(mx.load(os.path.join(ROOT, "weights", "distogram_head.safetensors")))
    logits = head(mx.array(r["final.z"]))
    mx.eval(logits)
    e = rel(logits, r["final.distogram"])
    print(f"\nN={n} distogram rel {e:.2e}")
    assert e < BLOCK_RTOL
