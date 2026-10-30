"""Mixed-precision switch for the pair-path sub-ops.

Default is ``None`` (everything fp32, bit-for-bit the validated behaviour). Inside
``compute_dtype(mx.bfloat16)`` the Linear projections, gating and triangle contraction run
in that dtype, while the residual streams ``s`` and ``z`` stay fp32: every sub-op's update
is added into an fp32 stream, and the width adapters and distogram head are forced back to
fp32. Residuals stay fp32 because they carry large values here (|s| reaches ~3e5, which
would overflow fp16 outright and lose precision in bf16).
"""
from __future__ import annotations

import contextlib

import mlx.core as mx

_state = {"dtype": None}


def get():
    return _state["dtype"]


@contextlib.contextmanager
def compute_dtype(dtype):
    prev = _state["dtype"]
    _state["dtype"] = dtype
    try:
        yield
    finally:
        _state["dtype"] = prev


def lo(x: mx.array) -> mx.array:
    """Cast to the active compute dtype (a no-op when none is set or already that dtype)."""
    dt = _state["dtype"]
    return x if dt is None or x.dtype == dt else x.astype(dt)
