"""Where does the CURRENT triangle-attention kernel spend its time? One-component ablations.

Each variant removes one piece of work from the v1 kernel (results are numerically wrong on
purpose; only the TIME is read). If removing a component barely changes the time it is not
the bottleneck; a large drop points at it. Costs are not additive (components overlap), so
this ranks suspects rather than partitioning the total.

  bias      skip the per-(q,k) tri_bias / mask_bias global loads
  exp       skip both exps per key (p = s scaled, correction = 1)
  rescale   skip the per-key 32-wide accumulator rescale (correction = 1, no exp for it)
  qk        reduce the 32-wide QK dot product to a single FMA
  pv        skip the PV accumulate entirely
  both      skip qk AND pv (leaves only softmax bookkeeping, bias loads, tile loads, barriers)

Capped at 4 GiB; N=256 is ~0.4 GB of tensors.
"""
import time

import mlx.core as mx

from mol_mlx.guard import memory_guard

N, H, D = 256, 10, 32
SCALE = D ** -0.5

SRC = r"""
    constexpr uint TILE = 32;
    constexpr uint HEAD_DIM = 32;
    uint3 tgid = threadgroup_position_in_grid;
    uint3 tid = thread_position_in_threadgroup;
    const uint b = tgid.z;
    const uint h = tgid.y;
    const uint q = tgid.x * TILE + tid.x;
    const uint B = dims[0];
    const uint HH = dims[1];
    const uint NN = dims[2];
    if (b >= B || h >= HH) return;

    const uint qkv_row = NN * HEAD_DIM;
    const size_t qkv_off = (size_t)b * HH * qkv_row + (size_t)h * qkv_row;
    const size_t tri_off = (size_t)h * NN * NN;
    const size_t msk_off = (size_t)b * NN;

    threadgroup float k_tile[TILE][HEAD_DIM];
    threadgroup float v_tile[TILE][HEAD_DIM];
    float acc[HEAD_DIM];
    float qreg[HEAD_DIM];
    for (uint d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;
    const bool active = (q < NN);
    for (uint d = 0; d < HEAD_DIM; ++d) qreg[d] = active ? q_in[qkv_off + (size_t)q * HEAD_DIM + d] : 0.0f;

    float m_run = -INFINITY;
    float l_run = 0.0f;
    for (uint kb = 0; kb < NN; kb += TILE) {
        for (uint idx = tid.x; idx < TILE * HEAD_DIM; idx += TILE) {
            uint r = idx / HEAD_DIM, c = idx % HEAD_DIM, kcol = kb + r;
            float kk = 0.0f, vv = 0.0f;
            if (kcol < NN) { kk = k_in[qkv_off + (size_t)kcol * HEAD_DIM + c]; vv = v_in[qkv_off + (size_t)kcol * HEAD_DIM + c]; }
            k_tile[r][c] = kk; v_tile[r][c] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint t = 0; t < TILE; ++t) {
                uint kcol = kb + t;
                if (kcol >= NN) break;
                float s = 0.0f;
                if (ABL_QK) { s = qreg[0] * k_tile[t][0]; }
                else { for (uint d = 0; d < HEAD_DIM; ++d) s += qreg[d] * k_tile[t][d]; }
                s *= scale;
                if (!ABL_BIAS) { s += tri_bias[tri_off + (size_t)q * NN + kcol]; s += mask_bias[msk_off + kcol]; }
                float m_new = fmax(m_run, s);
                float correction = 1.0f;
                float p;
                if (ABL_EXP) { p = s * 0.001f; }
                else {
                    if (!ABL_RESCALE) correction = (m_run == -INFINITY) ? 0.0f : fast::exp(m_run - m_new);
                    p = fast::exp(s - m_new);
                }
                l_run = l_run * correction + p;
                if (!ABL_PV) {
                    if (ABL_RESCALE) { for (uint d = 0; d < HEAD_DIM; ++d) acc[d] += p * v_tile[t][d]; }
                    else { for (uint d = 0; d < HEAD_DIM; ++d) acc[d] = acc[d] * correction + p * v_tile[t][d]; }
                } else { acc[0] += p; }
                m_run = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
        for (uint d = 0; d < HEAD_DIM; ++d) out[qkv_off + (size_t)q * HEAD_DIM + d] = acc[d] * inv;
    }
"""

kernel = mx.fast.metal_kernel(
    name="tri_ablate",
    input_names=["q_in", "k_in", "v_in", "tri_bias", "mask_bias", "scale", "dims"],
    output_names=["out"], source=SRC)

VARIANTS = {
    "baseline (v1)": dict(ABL_BIAS=0, ABL_EXP=0, ABL_RESCALE=0, ABL_QK=0, ABL_PV=0),
    "no bias loads": dict(ABL_BIAS=1, ABL_EXP=0, ABL_RESCALE=0, ABL_QK=0, ABL_PV=0),
    "no exp":        dict(ABL_BIAS=0, ABL_EXP=1, ABL_RESCALE=0, ABL_QK=0, ABL_PV=0),
    "no rescale":    dict(ABL_BIAS=0, ABL_EXP=0, ABL_RESCALE=1, ABL_QK=0, ABL_PV=0),
    "1-FMA QK":      dict(ABL_BIAS=0, ABL_EXP=0, ABL_RESCALE=0, ABL_QK=1, ABL_PV=0),
    "no PV":         dict(ABL_BIAS=0, ABL_EXP=0, ABL_RESCALE=0, ABL_QK=0, ABL_PV=1),
    "no QK, no PV":  dict(ABL_BIAS=0, ABL_EXP=0, ABL_RESCALE=0, ABL_QK=1, ABL_PV=1),
    "no exp,rescale,bias": dict(ABL_BIAS=1, ABL_EXP=1, ABL_RESCALE=1, ABL_QK=0, ABL_PV=0),
}


def run(flags):
    q = mx.random.normal((N, H, N, D), key=mx.random.key(0))
    k = mx.random.normal((N, H, N, D), key=mx.random.key(1))
    v = mx.random.normal((N, H, N, D), key=mx.random.key(2))
    tri = mx.random.normal((H, N, N), key=mx.random.key(3)) * 0.5
    msk = mx.zeros((N, N))
    mx.eval(q, k, v, tri, msk)
    n_qt = (N + 31) // 32

    def go():
        return kernel(inputs=[q, k, v, tri, msk, mx.array(SCALE, dtype=mx.float32), mx.array([N, H, N], dtype=mx.uint32)],
                      template=[(n, int(f)) for n, f in flags.items()],
                      output_shapes=[(N, H, N, D)], output_dtypes=[mx.float32],
                      grid=(n_qt * 32, H, N), threadgroup=(32, 1, 1))[0]
    mx.eval(go())
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        mx.eval(go())
        ts.append(time.perf_counter() - t)
    return min(ts) * 1e3


with memory_guard(max_gib=4.0):
    flop = 4.0 * N * H * N * N * D          # QK and PV, 2 flop per MAC
    print(f"N={N} H={H} D={D}: {flop / 1e9:.1f} GFLOP of attention math per call\n")
    base = None
    print(f"{'variant':24s} {'ms':>8s} {'vs baseline':>12s} {'TFLOP/s*':>9s}")
    for name, flags in VARIANTS.items():
        ms = run(flags)
        base = base or ms
        print(f"{name:24s} {ms:8.1f} {ms / base:11.2f}x {flop / (ms * 1e-3) / 1e12:9.2f}")
    print("\n* counts the full attention FLOP regardless of what was ablated, to compare variants")
