"""Triangle attention v2: flash attention on simdgroup 8x8 matrix tiles, fp32.

v1 (tri_attn_metal.py) puts one query on one thread, so each thread holds a 32-float query
and a 32-float accumulator and does a scalar 32-wide dot product per key. An ablation
(bench/kernel_ablation.py) showed the cost is in those two inner loops, not in the exp, the
accumulator rescale or the bias loads (removing those changed nothing), and the best reading
is register pressure and no operand reuse.

Here a simdgroup owns 8 queries and the work runs on hardware 8x8x8 multiply-accumulates:

    S   = sum over 4 head-dim chunks of  Q_c (8x8) @ K_c^T (8x8)        # 8 queries x 8 keys
    S   = S*scale + tri_bias + mask_bias ; padded keys masked
    online softmax per query ROW           # a row lives on 4 lanes {l, l^1, l^8, l^9}
    O_c = O_c*corr + P (8x8) @ V_c (8x8)                                # 4 accumulators

Layout, measured by bench/simdgroup_probe.py (not assumed): each lane holds two ADJACENT
elements of ONE row, row = ((lane>>1)&3) | (((lane>>4)&1)<<2),
first column = ((lane&1)<<1) | (((lane>>3)&1)<<2). So a row max or sum is two XOR shuffles
(xor 1, xor 8). Per lane that is ~16 floats of live state instead of ~64.

Masking uses a large finite negative, never -INFINITY: Metal compiles with fast-math, which
is allowed to assume no infinities and can break comparisons against them.

Padding: the kernel needs N to be a multiple of 8, so q/k/v/biases are zero-padded up to the
next multiple (a copy, only when needed) and the padded KEYS are masked inside the kernel.
Padded query rows compute finite garbage and are sliced off. Head dim is fixed at 32.
"""

from __future__ import annotations

import mlx.core as mx

SGS = 4            # simdgroups per threadgroup: 4 * 8 = 32 queries per threadgroup
HEAD_DIM = 32
NEG = -1.0e30

_SOURCE = r"""
    constexpr uint SGS_ = 4;
    constexpr float NEG_ = -1.0e30f;

    uint3 tgid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    const uint B = dims[0];
    const uint H = dims[1];
    const uint Np = dims[2];     // padded to a multiple of 8
    const uint Nr = dims[3];     // real length: keys >= Nr are masked
    const uint b = tgid.z;
    const uint h = tgid.y;
    const uint q0 = (tgid.x * SGS_ + sg) * 8;
    if (b >= B || h >= H || q0 >= Np) return;

    const size_t slab = ((size_t)b * H + h) * Np * 32;
    const device float* qp = q_in + slab;
    const device float* kp = k_in + slab;
    const device float* vp = v_in + slab;

    simdgroup_float8x8 Qm[4];
    simdgroup_float8x8 Om[4];
    simdgroup_float8x8 S;
    simdgroup_float8x8 Km;
    simdgroup_float8x8 Vm;
    for (uint c = 0; c < 4; ++c) {
        simdgroup_load(Qm[c], qp + (size_t)q0 * 32 + c * 8, 32);
        Om[c] = simdgroup_float8x8(0);
    }

    // The (row, first column) this lane owns within every 8x8 tile.
    const uint r  = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
    const uint c0 = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2);
    const uint qa = q0 + r;
    const size_t tri_row = (size_t)h * Np * Np + (size_t)qa * Np;
    const size_t msk_row = (size_t)b * Np;

    float m_run = NEG_;
    float l_run = 0.0f;

    for (uint kb = 0; kb < Np; kb += 8) {
        S = simdgroup_float8x8(0);
        for (uint c = 0; c < 4; ++c) {
            // K tile is [keys x dims]; load transposed so it multiplies as K^T.
            simdgroup_load(Km, kp + (size_t)kb * 32 + c * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(S, Qm[c], Km, S);
        }

        const uint k0 = kb + c0;
        float s0 = S.thread_elements()[0] * scale + tri_bias[tri_row + k0] + mask_bias[msk_row + k0];
        float s1 = S.thread_elements()[1] * scale + tri_bias[tri_row + k0 + 1] + mask_bias[msk_row + k0 + 1];
        if (k0 >= Nr) s0 = NEG_;
        if (k0 + 1 >= Nr) s1 = NEG_;

        // Row max over the 4 lanes that share this row.
        float mb = fmax(s0, s1);
        mb = fmax(mb, simd_shuffle_xor(mb, 1));
        mb = fmax(mb, simd_shuffle_xor(mb, 8));

        const float m_new = fmax(m_run, mb);
        const float corr = fast::exp(m_run - m_new);
        const float p0 = fast::exp(s0 - m_new);
        const float p1 = fast::exp(s1 - m_new);
        l_run = l_run * corr + p0 + p1;     // per-lane partial; reduced across the row once, at the end
        m_run = m_new;

        S.thread_elements()[0] = p0;        // S now holds P
        S.thread_elements()[1] = p1;
        for (uint c = 0; c < 4; ++c) {
            Om[c].thread_elements()[0] *= corr;
            Om[c].thread_elements()[1] *= corr;
            simdgroup_load(Vm, vp + (size_t)kb * 32 + c * 8, 32);
            simdgroup_multiply_accumulate(Om[c], S, Vm, Om[c]);
        }
    }

    float l_tot = l_run + simd_shuffle_xor(l_run, 1);
    l_tot += simd_shuffle_xor(l_tot, 8);
    const float inv = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    for (uint c = 0; c < 4; ++c) {
        Om[c].thread_elements()[0] *= inv;
        Om[c].thread_elements()[1] *= inv;
        simdgroup_store(Om[c], out + slab + (size_t)q0 * 32 + c * 8, 32);
    }
"""

_kernel = mx.fast.metal_kernel(
    name="tri_flash_mma",
    input_names=["q_in", "k_in", "v_in", "tri_bias", "mask_bias", "scale", "dims"],
    output_names=["out"],
    source=_SOURCE,
    header="#include <metal_simdgroup_matrix>\n",
)


def flash_triangle_attention_mma(
    q: mx.array, k: mx.array, v: mx.array, tri_bias: mx.array, mask_bias: mx.array, scale: float
) -> mx.array:
    """Same contract as ``tri_attn_metal.flash_triangle_attention`` (fp32, head_dim 32).

    ``q, k, v``: [B, H, N, 32], q NOT pre-scaled. ``tri_bias``: [H, N, N] (shared over B).
    ``mask_bias``: [B, N]. Returns [B, H, N, 32].
    """
    B, H, N, D = q.shape
    if D != HEAD_DIM:
        raise ValueError(f"kernel is specialized for head_dim {HEAD_DIM}, got {D}")
    for name, arr in (("k", k), ("v", v)):
        if arr.shape != q.shape:
            raise ValueError(f"{name} shape {arr.shape} != q shape {q.shape}")
    if tri_bias.shape != (H, N, N):
        raise ValueError(f"tri_bias shape {tri_bias.shape} != {(H, N, N)}")
    if mask_bias.shape != (B, N):
        raise ValueError(f"mask_bias shape {mask_bias.shape} != {(B, N)}")

    q, k, v = (x.astype(mx.float32) for x in (q, k, v))
    tri_bias, mask_bias = tri_bias.astype(mx.float32), mask_bias.astype(mx.float32)

    Np = (N + 7) // 8 * 8
    if Np != N:
        pad = Np - N
        q, k, v = (mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)]) for x in (q, k, v))
        tri_bias = mx.pad(tri_bias, [(0, 0), (0, pad), (0, pad)])
        mask_bias = mx.pad(mask_bias, [(0, 0), (0, pad)])
    q, k, v = (mx.contiguous(x) for x in (q, k, v))
    tri_bias, mask_bias = mx.contiguous(tri_bias), mx.contiguous(mask_bias)

    n_tg = (Np + 8 * SGS - 1) // (8 * SGS)
    (out,) = _kernel(
        inputs=[q, k, v, tri_bias, mask_bias, mx.array(scale, dtype=mx.float32),
                mx.array([B, H, Np, N], dtype=mx.uint32)],
        output_shapes=[(B, H, Np, D)],
        output_dtypes=[mx.float32],
        grid=(n_tg * 32 * SGS, H, B),
        threadgroup=(32 * SGS, 1, 1),
    )
    return out[:, :, :N, :] if Np != N else out
