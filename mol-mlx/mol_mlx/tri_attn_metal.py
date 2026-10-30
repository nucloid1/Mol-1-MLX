"""Flash-style triangle attention as a hand-written Metal kernel.

Why this exists: MLX 0.32's ``mx.fast.scaled_dot_product_attention`` does not take
the fused path at triangle-attention shapes. Measured (bench/sdpa_probe.py, one
process per case, peak above inputs):

    N=192  scores 270 MiB -> peak 360 MiB   (1.33x the score tensor)
    N=288  scores 911 MiB -> peak 1114 MiB  (1.22x)

i.e. it materializes [B, H, N, N] exactly as the naive path does, and is no faster.
At N=600 that single tensor is 600*10*600*600*4 B = 8.0 GiB, per call, 80 calls
per cycle. That term is the reason Mol-1 peaks at 40 GB at 600 tokens.

This kernel is the standard online-softmax (FlashAttention) loop: each threadgroup
owns one (batch row, head, query tile), streams K/V tiles through threadgroup
memory, and keeps only the running max ``m``, running sum ``l`` and accumulator
``acc`` in registers. Nothing of size N*N is ever written to device memory.

Triangle attention's bias is the part that differs from ordinary attention, and it
is why the stock kernel cannot be used: the additive bias is
``mask_bias[b, 1, 1, j] + tri_bias[1, h, i, j]`` -- shared across heads for the
mask part, and shared across the batch (row) axis for the triangle part. The
triangle term is read straight from its [H, N, N] tensor inside the loop, so it
is never broadcast into a [B, H, N, N] temporary either.
"""

from __future__ import annotations

import mlx.core as mx

# One query per thread, TILE keys per iteration. head_dim is 32 for every triangle
# attention in Mol-1 (c_hidden=32), so the accumulator fits comfortably in registers.
_SOURCE = r"""
    constexpr uint TILE = 32;
    constexpr uint HEAD_DIM = 32;

    uint3 tgid = threadgroup_position_in_grid;
    uint3 tid = thread_position_in_threadgroup;

    const uint b = tgid.z;            // batch row (the I axis of triangle attention)
    const uint h = tgid.y;            // head
    const uint q_base = tgid.x * TILE;
    const uint q = q_base + tid.x;    // this thread's query index

    const uint B = dims[0];
    const uint H = dims[1];
    const uint N = dims[2];

    if (b >= B || h >= H) return;

    // Strides (row-major, contiguous).
    const uint qkv_row = N * HEAD_DIM;
    const uint qkv_head = H * qkv_row;
    const size_t qkv_off = (size_t)b * qkv_head + (size_t)h * qkv_row;
    const size_t tri_off = (size_t)h * N * N;   // tri_bias is [H, N, N]
    const size_t msk_off = (size_t)b * N;       // mask_bias is [B, N]

    threadgroup float k_tile[TILE][HEAD_DIM];
    threadgroup float v_tile[TILE][HEAD_DIM];

    float acc[HEAD_DIM];
    float qreg[HEAD_DIM];
    for (uint d = 0; d < HEAD_DIM; ++d) { acc[d] = 0.0f; }

    const bool active = (q < N);
    if (active) {
        for (uint d = 0; d < HEAD_DIM; ++d) {
            qreg[d] = q_in[qkv_off + (size_t)q * HEAD_DIM + d];
        }
    } else {
        for (uint d = 0; d < HEAD_DIM; ++d) { qreg[d] = 0.0f; }
    }

    float m_run = -INFINITY;   // running max
    float l_run = 0.0f;        // running sum of exp

    for (uint kb = 0; kb < N; kb += TILE) {
        // Cooperative load of this K/V tile into threadgroup memory.
        for (uint idx = tid.x; idx < TILE * HEAD_DIM; idx += TILE) {
            uint r = idx / HEAD_DIM;
            uint c = idx % HEAD_DIM;
            uint kcol = kb + r;
            float kv_k = 0.0f, kv_v = 0.0f;
            if (kcol < N) {
                kv_k = k_in[qkv_off + (size_t)kcol * HEAD_DIM + c];
                kv_v = v_in[qkv_off + (size_t)kcol * HEAD_DIM + c];
            }
            k_tile[r][c] = kv_k;
            v_tile[r][c] = kv_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint t = 0; t < TILE; ++t) {
                uint kcol = kb + t;
                if (kcol >= N) break;

                float s = 0.0f;
                for (uint d = 0; d < HEAD_DIM; ++d) {
                    s += qreg[d] * k_tile[t][d];
                }
                s *= scale;
                // Triangle attention's two additive bias terms, read in place.
                s += tri_bias[tri_off + (size_t)q * N + kcol];
                s += mask_bias[msk_off + kcol];

                // Online softmax rescale.
                float m_new = fmax(m_run, s);
                float correction = (m_run == -INFINITY) ? 0.0f : fast::exp(m_run - m_new);
                float p = fast::exp(s - m_new);
                l_run = l_run * correction + p;
                for (uint d = 0; d < HEAD_DIM; ++d) {
                    acc[d] = acc[d] * correction + p * v_tile[t][d];
                }
                m_run = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
        for (uint d = 0; d < HEAD_DIM; ++d) {
            out[qkv_off + (size_t)q * HEAD_DIM + d] = acc[d] * inv;
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="tri_flash_attention",
    input_names=["q_in", "k_in", "v_in", "tri_bias", "mask_bias", "scale", "dims"],
    output_names=["out"],
    source=_SOURCE,
)

TILE = 32
HEAD_DIM = 32


def flash_triangle_attention(
    q: mx.array, k: mx.array, v: mx.array, tri_bias: mx.array, mask_bias: mx.array, scale: float
) -> mx.array:
    """Attention over [B, H, N, D] with triangle attention's two broadcast biases.

    Args:
        q, k, v: ``[B, H, N, 32]``, contiguous float32. ``q`` must NOT be pre-scaled.
        tri_bias: ``[H, N, N]`` -- shared across the batch axis.
        mask_bias: ``[B, N]`` -- shared across heads and queries.
        scale: multiplied into the logits (``1/sqrt(head_dim)``).

    Returns:
        ``[B, H, N, 32]``.
    """
    B, H, N, D = q.shape
    if D != HEAD_DIM:
        raise ValueError(f"kernel is specialized for head_dim {HEAD_DIM}, got {D}")
    for name, arr, shape in (("k", k, q.shape), ("v", v, q.shape)):
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} != q shape {shape}")
    if tri_bias.shape != (H, N, N):
        raise ValueError(f"tri_bias shape {tri_bias.shape} != {(H, N, N)}")
    if mask_bias.shape != (B, N):
        raise ValueError(f"mask_bias shape {mask_bias.shape} != {(B, N)}")

    q, k, v = (mx.contiguous(x.astype(mx.float32)) for x in (q, k, v))
    tri_bias = mx.contiguous(tri_bias.astype(mx.float32))
    mask_bias = mx.contiguous(mask_bias.astype(mx.float32))

    n_qtiles = (N + TILE - 1) // TILE
    (out,) = _kernel(
        inputs=[q, k, v, tri_bias, mask_bias, mx.array(scale, dtype=mx.float32),
                mx.array([B, H, N], dtype=mx.uint32)],
        output_shapes=[(B, H, N, D)],
        output_dtypes=[mx.float32],
        grid=(n_qtiles * TILE, H, B),
        threadgroup=(TILE, 1, 1),
    )
    return out
