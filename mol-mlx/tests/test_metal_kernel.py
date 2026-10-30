"""Both Metal triangle-attention kernels must match the dense reference.

v1 = one query per thread (tri_attn_metal); v2 = simdgroup 8x8 matrix tiles (tri_attn_mma).
Sizes deliberately include N that are NOT multiples of 8 (v2 pads and masks the padded keys),
N=1..7 (a single partial tile), and 14 heads (the confidence head's width).
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from mol_mlx.tri_attn_metal import flash_triangle_attention as v1
from mol_mlx.tri_attn_mma import flash_triangle_attention_mma as v2

IMPLS = [pytest.param(v1, id="v1"), pytest.param(v2, id="v2-mma")]
CASES = [(5, 2), (16, 2), (48, 10), (53, 10), (97, 10), (128, 14), (130, 10)]


def dense(q, k, v, tri, msk, scale):
    s = (q * scale) @ mx.swapaxes(k, -1, -2) + tri[None] + msk[:, None, None, :]
    return mx.softmax(s, axis=-1, precise=True) @ v


def make(N, H, seed=0):
    D = 32
    ks = [mx.random.key(seed + i) for i in range(5)]
    q = mx.random.normal((N, H, N, D), key=ks[0])
    k = mx.random.normal((N, H, N, D), key=ks[1])
    v = mx.random.normal((N, H, N, D), key=ks[2])
    tri = mx.random.normal((H, N, N), key=ks[3]) * 0.5
    msk = mx.random.normal((N, N), key=ks[4]) * 0.1     # non-trivial mask bias too
    return q, k, v, tri, msk, D ** -0.5


def rel(a, b):
    return float(mx.abs(a - b).max()) / float(mx.abs(b).max())


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("N,H", CASES)
def test_matches_dense(impl, N, H):
    q, k, v, tri, msk, scale = make(N, H)
    got, want = impl(q, k, v, tri, msk, scale), dense(q, k, v, tri, msk, scale)
    mx.eval(got, want)
    assert got.shape == want.shape
    assert rel(got, want) < 1e-5, f"N={N} H={H}: rel {rel(got, want):.3e}"


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("N", [48, 53])
def test_mask_bias_excludes_keys(impl, N):
    """A -1e9 mask entry must drive that key's weight to zero (also across the pad boundary)."""
    H = 10
    q, k, v, tri, _, scale = make(N, H, seed=7)
    half = N // 2
    msk = mx.concatenate([mx.zeros((N, half)), mx.full((N, N - half), -1e9)], axis=1)
    got = impl(q, k, v, tri, msk, scale)
    s = (q * scale) @ mx.swapaxes(k, -1, -2) + tri[None]
    want = mx.softmax(s[..., :half], axis=-1, precise=True) @ v[:, :, :half, :]
    mx.eval(got, want)
    assert rel(got, want) < 1e-5, f"masked N={N}: rel {rel(got, want):.3e}"


@pytest.mark.parametrize("impl", IMPLS)
def test_large_logits_stay_finite(impl):
    """Online softmax must not overflow when scores are large (the max-subtraction is the point)."""
    N, H = 40, 4
    q, k, v, tri, msk, scale = make(N, H, seed=3)
    q, k = q * 12.0, k * 12.0
    got, want = impl(q, k, v, tri, msk, scale), dense(q, k, v, tri, msk, scale)
    mx.eval(got, want)
    assert bool(mx.all(mx.isfinite(got)).item())
    assert rel(got, want) < 1e-4


@pytest.mark.parametrize("impl", IMPLS)
def test_rejects_wrong_head_dim(impl):
    q = mx.zeros((4, 2, 4, 16))
    with pytest.raises(ValueError, match="head_dim"):
        impl(q, q, q, mx.zeros((2, 4, 4)), mx.zeros((4, 4)), 0.25)
