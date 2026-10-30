"""Core Mol-1 primitives in MLX.

Each class mirrors one torch module and loads that module's checkpoint weights
unchanged. Semantics were read from the Mol-1 source, not assumed:

- ``LayerNorm``      mol/model/triangular/layers.py OpenFoldLayerNorm, eps 1e-5,
                     scale and offset independently optional.
- ``Linear``         OpenfoldLinear / LinearNoBias: ``y = x @ W.T (+ b)``.
- ``Transition``     mol/model/modules/primitives.py: SwiGLU,
                     ``W_o(silu(W_a·LN(x)) * W_b·LN(x))``.
- ``TriangleMultiplicativeUpdate``  mol/model/triangular/triangular.py, the
                     ``torch`` branch of ``forward`` (the branch MPS takes).
- ``Attention``      mol/model/triangular/layers.py: scale on q, additive
                     biases, sigmoid gate on the query, output projection.
- ``TriangleAttention`` mol/model/triangular/triangular.py.

Shapes follow the torch module exactly so reference tensors can be compared
element-wise without reordering.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Sequence

import mlx.core as mx

import os

from . import precision


def _p(w) -> mx.array:
    return w if isinstance(w, mx.array) else mx.array(w)


# Which Metal kernel triangle attention uses: "mma" (simdgroup matrix tiles, v2) or "v1"
# (one query per thread). v2 is ~3x faster on the attention math; v1 is kept for comparison.
DEFAULT_TRI_IMPL = os.environ.get("MOLMLX_TRI_IMPL", "mma")


class Weights(dict):
    """A weight dict that records which keys were read.

    ``LayerNorm`` scale/offset are legitimately optional in Mol-1, so a missing LayerNorm
    key cannot be an error at lookup time -- but a typo'd prefix then looks identical to
    "no affine" and silently changes the math. Tracking reads closes that: after a module
    is built, :py:meth:`assert_all_used` fails if any weight under it was never consumed.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.used: set[str] = set()

    def __getitem__(self, k):
        self.used.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k in self:
            self.used.add(k)
        return super().get(k, default)

    def unused(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self if k.startswith(prefix) and k not in self.used)

    def assert_all_used(self, prefix: str = "", allow: tuple = ()) -> None:
        left = [k for k in self.unused(prefix) if not any(k.startswith(a) for a in allow)]
        if left:
            raise KeyError(
                f"{len(left)} weight(s) under {prefix!r} were never used -- a mistyped key or an "
                f"unported parameter: {left[:6]}{' ...' if len(left) > 6 else ''}")


def key(prefix: str, name: str) -> str:
    """Join a weight-path prefix and a name; an empty prefix adds no leading dot."""
    return f"{prefix}.{name}" if prefix else name


class LayerNorm:
    """OpenFoldLayerNorm: normalize the last axis, optional affine, eps 1e-5."""

    def __init__(self, weight: Optional[mx.array] = None, bias: Optional[mx.array] = None,
                 eps: float = 1e-5) -> None:
        self.weight = None if weight is None else _p(weight)
        self.bias = None if bias is None else _p(bias)
        self.eps = eps

    def _affine(self, dtype):
        if dtype == self.weight.dtype:
            return self.weight, self.bias
        cache = self.__dict__.setdefault("_lo", {})
        if dtype not in cache:
            cache[dtype] = (self.weight.astype(dtype), self.bias.astype(dtype))
        return cache[dtype]

    def __call__(self, x: mx.array) -> mx.array:
        # mx.fast.layer_norm requires both or neither; call it only when it applies.
        if self.weight is not None and self.bias is not None:
            w, b = self._affine(x.dtype)
            return mx.fast.layer_norm(x, w, b, self.eps)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        out = (x - mean) * mx.rsqrt(var + self.eps)
        if self.weight is not None:
            out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def load(cls, w: dict, prefix: str, eps: float = 1e-5) -> "LayerNorm":
        return cls(w.get(key(prefix, "weight")), w.get(key(prefix, "bias")), eps)


class Linear:
    """``y = x @ W.T + b``; torch stores W as [out, in]."""

    def __init__(self, weight: mx.array, bias: Optional[mx.array] = None) -> None:
        self.weight = _p(weight)
        self.bias = None if bias is None else _p(bias)

    def __call__(self, x: mx.array, low: bool = True) -> mx.array:
        dt = precision.get() if low else None
        if dt is None:
            y = x @ self.weight.T
            return y if self.bias is None else y + self.bias
        cache = self.__dict__.setdefault("_lo", {})
        if dt not in cache:
            cache[dt] = (self.weight.astype(dt), None if self.bias is None else self.bias.astype(dt))
        w, b = cache[dt]
        y = (x if x.dtype == dt else x.astype(dt)) @ w.T
        return y if b is None else y + b

    @classmethod
    def load(cls, w: dict, prefix: str) -> "Linear":
        return cls(w[key(prefix, "weight")], w.get(key(prefix, "bias")))


class Transition:
    """SwiGLU transition: ``W_o(silu(W_a·LN(x)) * W_b·LN(x))``."""

    def __init__(self, w: dict, prefix: str) -> None:
        self.ln = LayerNorm.load(w, key(prefix, "layernorm1"))
        a = Linear.load(w, key(prefix, "linear_no_bias_a"))
        b = Linear.load(w, key(prefix, "linear_no_bias_b"))
        # One matmul for both branches (reads the normalized input once); neither has a bias.
        self.hidden = a.weight.shape[0]
        self.ab = Linear(mx.concatenate([a.weight, b.weight], axis=0))
        self.o = Linear.load(w, key(prefix, "linear_no_bias"))

    def __call__(self, x: mx.array, chunk: Optional[int] = None) -> mx.array:
        """``chunk`` splits along the penultimate axis; the op is pointwise, so
        chunking changes only peak memory, never the result."""
        if chunk is None:
            return self._proj(x)
        rows = x.shape[-2]
        if chunk >= rows:
            return self._proj(x)
        parts = [self._proj(x[..., i:i + chunk, :]) for i in range(0, rows, chunk)]
        return mx.concatenate(parts, axis=-2)

    def _proj(self, x: mx.array) -> mx.array:
        ab = self.ab(precision.lo(self.ln(x)))
        return self.o(_silu_mul(ab[..., :self.hidden], ab[..., self.hidden:]))


# Fused elementwise passes (one kernel each instead of one per op); shapeless so a new N does
# not retrace. Measured at N=256: transition 22.0 -> 18.2 ms, triangle multiplication 15.2 -> 13.9 ms.
@partial(mx.compile, shapeless=True)
def _silu_mul(a: mx.array, b: mx.array) -> mx.array:
    return a * mx.sigmoid(a) * b


@partial(mx.compile, shapeless=True)
def _gate(g: mx.array, p: mx.array) -> mx.array:
    return mx.sigmoid(g) * p


class TriangleMultiplicativeUpdate:
    """Algorithms 11/12, the ``torch`` branch Mol-1 takes off CUDA.

        zn = LN_in(z)
        a  = mask * sigmoid(a_g(zn)) * a_p(zn)
        b  = mask * sigmoid(b_g(zn)) * b_p(zn)
        outgoing:  x[i,j,c] = sum_k a[i,k,c] * b[j,k,c]
        incoming:  x[i,j,c] = sum_k a[k,i,c] * b[k,j,c]
        out = linear_z(LN_out(x)) * sigmoid(g(zn))
    """

    def __init__(self, w: dict, prefix: str, outgoing: bool) -> None:
        self.outgoing = outgoing
        self.ln_in = LayerNorm.load(w, key(prefix, "layer_norm_in"))
        self.ln_out = LayerNorm.load(w, key(prefix, "layer_norm_out"))
        self.a_p = Linear.load(w, key(prefix, "linear_a_p"))
        self.a_g = Linear.load(w, key(prefix, "linear_a_g"))
        self.b_p = Linear.load(w, key(prefix, "linear_b_p"))
        self.b_g = Linear.load(w, key(prefix, "linear_b_g"))
        self.z_out = Linear.load(w, key(prefix, "linear_z"))
        self.g = Linear.load(w, key(prefix, "linear_g"))

    def __call__(self, z: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        zn = precision.lo(self.ln_in(z))
        a = _gate(self.a_g(zn), self.a_p(zn))
        b = _gate(self.b_g(zn), self.b_p(zn))
        if mask is not None:
            m = mx.expand_dims(mask, -1).astype(a.dtype)
            a, b = m * a, m * b
        # [N, N, C] -> [C, N, N], then one batched matmul over the channel axis.
        if self.outgoing:
            at = mx.transpose(a, (2, 0, 1))   # [c, i, k]
            bt = mx.transpose(b, (2, 1, 0))   # [c, k, j]  (b[j,k,c] -> bt[c,k,j])
        else:
            at = mx.transpose(a, (2, 1, 0))   # [c, i, k]  (a[k,i,c] -> at[c,i,k])
            bt = mx.transpose(b, (2, 0, 1))   # [c, k, j]
        x = at @ bt                            # [c, i, j]
        x = mx.transpose(x, (1, 2, 0))         # [i, j, c]
        return _gate(self.g(zn), self.z_out(self.ln_out(x)))


class Attention:
    """OpenFold-style MHA with additive biases and a sigmoid query gate."""

    def __init__(self, w: dict, prefix: str, n_heads: int, head_dim: int) -> None:
        self.h, self.d = n_heads, head_dim
        self.q = Linear.load(w, key(prefix, "linear_q"))
        self.k = Linear.load(w, key(prefix, "linear_k"))
        self.v = Linear.load(w, key(prefix, "linear_v"))
        self.o = Linear.load(w, key(prefix, "linear_o"))
        self.g = Linear.load(w, key(prefix, "linear_g")) if key(prefix, "linear_g.weight") in w else None

    def _heads(self, x: mx.array) -> mx.array:
        # [..., N, H*D] -> [..., H, N, D]
        return mx.swapaxes(x.reshape(*x.shape[:-1], self.h, self.d), -2, -3)

    def attend_triangle_flash(self, x: mx.array, tri_bias: mx.array, mask_bias: mx.array,
                              impl: str = "mma") -> mx.array:
        """Same math as ``__call__`` with triangle attention's two biases, via the Metal kernel.

        ``x`` is [B, N, C]; ``tri_bias`` is [H, N, N] (shared across B); ``mask_bias`` is
        [B, N]. q is passed UNSCALED -- the kernel applies ``head_dim ** -0.5`` itself.
        """
        if impl == "v1":
            from .tri_attn_metal import flash_triangle_attention as kernel
        else:
            from .tri_attn_mma import flash_triangle_attention_mma as kernel

        q, k, v = self._heads(self.q(x)), self._heads(self.k(x)), self._heads(self.v(x))
        o = kernel(q, k, v, tri_bias, mask_bias, self.d ** -0.5)
        o = mx.swapaxes(o, -2, -3)
        if self.g is not None:
            o = o * mx.sigmoid(self.g(x)).reshape(*o.shape)
        return self.o(o.reshape(*o.shape[:-2], self.h * self.d))

    def __call__(self, x: mx.array, biases: Sequence[mx.array], sdpa=None) -> mx.array:
        q = self._heads(self.q(x)) * (self.d ** -0.5)
        k = self._heads(self.k(x))
        v = self._heads(self.v(x))
        bias = biases[0]
        for extra in biases[1:]:
            bias = bias + extra
        o = (sdpa or _sdpa_dense)(q, k, v, bias)   # [..., H, N, D]
        o = mx.swapaxes(o, -2, -3)                 # [..., N, H, D]
        if self.g is not None:
            gate = mx.sigmoid(self.g(x)).reshape(*o.shape)
            o = o * gate
        return self.o(o.reshape(*o.shape[:-2], self.h * self.d))


def _sdpa_dense(q: mx.array, k: mx.array, v: mx.array, bias: mx.array) -> mx.array:
    """Reference attention. Materializes [..., H, N, N]; correctness baseline."""
    scores = q @ mx.swapaxes(k, -1, -2) + bias
    return mx.softmax(scores, axis=-1, precise=True) @ v


class TriangleAttention:
    """Algorithms 13/14. ``starting=False`` transposes the first two axes around the call."""

    INF = 1e9

    def __init__(self, w: dict, prefix: str, n_heads: int, head_dim: int, starting: bool,
                 use_metal: bool = True, impl: Optional[str] = None) -> None:
        self.starting = starting
        self.impl = impl or DEFAULT_TRI_IMPL
        self.use_metal = use_metal and head_dim == 32
        self.ln = LayerNorm.load(w, key(prefix, "layer_norm"))
        self.lin = Linear.load(w, key(prefix, "linear"))
        self.mha = Attention(w, key(prefix, "mha"), n_heads, head_dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, sdpa=None) -> mx.array:
        if not self.starting:
            x = mx.swapaxes(x, -2, -3)
            mask = None if mask is None else mx.swapaxes(mask, -1, -2)
        if mask is None:
            mask = mx.ones(x.shape[:-1], dtype=x.dtype)
        xn = precision.lo(self.ln(x))
        if self.use_metal and sdpa is None and x.ndim == 3:
            # [N, N] row mask -> additive [B, N]; triangle bias stays [H, N, N], un-broadcast.
            mask_bias = self.INF * (mask - 1)
            tri_bias = mx.transpose(self.lin(xn), (2, 0, 1))
            out = self.mha.attend_triangle_flash(xn, tri_bias, mask_bias, impl=self.impl)
        else:
            # [I, 1, 1, J]
            mask_bias = (self.INF * (mask - 1))[..., :, None, None, :]
            # [H, I, J] -> [1, H, I, J]
            tri_bias = mx.expand_dims(mx.transpose(self.lin(xn), (2, 0, 1)), -4)
            out = self.mha(xn, [mask_bias, tri_bias], sdpa=sdpa)
        return mx.swapaxes(out, -2, -3) if not self.starting else out
