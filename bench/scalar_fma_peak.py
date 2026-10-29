"""Raw scalar FMA throughput by dtype on this GPU, with no memory traffic to speak of.

Answers one question: does half precision make SCALAR multiply-adds faster? Each thread runs
8 independent chains of fused multiply-adds in registers (independent so instruction latency
is hidden) and writes one value out. 2 FLOP per FMA.

If fp16 scalar FMA is no faster than fp32, then rewriting the attention kernel in fp16 while
keeping it scalar cannot fix its throughput; the structure (operand reuse) would have to.
Tiny memory (a few MB); capped at 2 GiB.
"""
import time

import mlx.core as mx

from mol_mlx.guard import memory_guard

THREADS = 1 << 21
ITER = 512
CHAINS = 8

SOURCE = """
    uint gid = thread_position_in_grid.x;
    T x0 = x[gid];
    T m = T(1.0001);
    T b = T(0.0001);
    T a0 = x0, a1 = x0 + T(1), a2 = x0 + T(2), a3 = x0 + T(3);
    T a4 = x0 + T(4), a5 = x0 + T(5), a6 = x0 + T(6), a7 = x0 + T(7);
    for (int i = 0; i < ITER; ++i) {
        a0 = fma(a0, m, b); a1 = fma(a1, m, b); a2 = fma(a2, m, b); a3 = fma(a3, m, b);
        a4 = fma(a4, m, b); a5 = fma(a5, m, b); a6 = fma(a6, m, b); a7 = fma(a7, m, b);
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
"""


def run(dtype, label):
    kernel = mx.fast.metal_kernel(name="fma_peak", input_names=["x"], output_names=["out"], source=SOURCE)
    x = mx.zeros((THREADS,), dtype=dtype)
    mx.eval(x)

    def go():
        return kernel(inputs=[x], template=[("T", dtype), ("ITER", ITER)],
                      output_shapes=[(THREADS,)], output_dtypes=[dtype],
                      grid=(THREADS, 1, 1), threadgroup=(256, 1, 1))[0]

    mx.eval(go())
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        mx.eval(go())
        ts.append(time.perf_counter() - t)
    flops = THREADS * ITER * CHAINS * 2
    tflops = flops / min(ts) / 1e12
    finite = bool(mx.all(mx.isfinite(go())).item())
    print(f"{label:6s} {tflops:7.2f} TFLOP/s   ({min(ts) * 1e3:7.1f} ms)   finite={finite}")
    return tflops


with memory_guard(max_gib=2.0):
    print(f"{THREADS} threads x {ITER} iterations x {CHAINS} independent FMA chains\n")
    f32 = run(mx.float32, "fp32")
    f16 = run(mx.float16, "fp16")
    print(f"\nfp16 scalar FMA is {f16 / f32:.2f}x the fp32 rate")
