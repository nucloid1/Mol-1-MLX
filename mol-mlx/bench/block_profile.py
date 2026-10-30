"""Where does one block's time go? Per-op wall time, synced after each op, N=256.

One PairMixer block (c_z=448, block 0) and one Pairformer block (c_z=320, block 8), each
timed op by op in fp32 and in fp16. Syncing after every op serializes the pipeline, so
the absolute times overstate a fused run, but the SHARES show which op class to attack.
The stack is 8 PairMixer + 40 Pairformer blocks, so shares are also given for the whole.

Capped at 4 GiB; a single block at N=256 needs well under that.
"""
import os
import time

import mlx.core as mx

from mol_mlx import precision
from mol_mlx.blocks import PairformerStack
from mol_mlx.guard import memory_guard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 256


def timed(fn, reps=3):
    mx.eval(fn())
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def profile(block, c_z, kind, dtype):
    z = mx.random.normal((N, N, c_z), key=mx.random.key(1)) * 0.1
    s = mx.random.normal((N, 384), key=mx.random.key(0))
    mx.eval(z, s)
    rows = []
    with precision.compute_dtype(dtype):
        rows.append(("tri_mul_out", timed(lambda: block.tri_mul_out(z))))
        rows.append(("tri_mul_in", timed(lambda: block.tri_mul_in(z))))
        if kind == "pairformer":
            rows.append(("tri_att_start", timed(lambda: block.tri_att_start(z))))
            rows.append(("tri_att_end", timed(lambda: block.tri_att_end(mx.swapaxes(z, 0, 1)))))
        rows.append(("pair_transition", timed(lambda: block.pair_transition(z, chunk=64))))
        rows.append(("single track", timed(lambda: block._single_track(s, z))))
    return rows


with memory_guard(max_gib=4.0):
    w = mx.load(os.path.join(ROOT, "weights", "pairformer_stack.safetensors"))
    stack = PairformerStack(w)
    for name, dt in (("fp32", None), ("fp16", mx.float16)):
        print(f"\n=== {name}, N={N} ===")
        total_stack = 0.0
        table = {}
        for kind, idx, c_z, count in (("pairmixer", 0, 448, 8), ("pairformer", 8, 320, 40)):
            rows = profile(stack.blocks[idx], c_z, kind, dt)
            block_ms = sum(t for _, t in rows)
            total_stack += block_ms * count
            table[kind] = (rows, block_ms, count)
        for kind, (rows, block_ms, count) in table.items():
            print(f"\n{kind} block (x{count}): {block_ms:7.1f} ms per block, "
                  f"{100 * block_ms * count / total_stack:4.1f}% of the stack")
            for op, t in rows:
                print(f"    {op:16s} {t:8.1f} ms  {100 * t / block_ms:5.1f}% of block")
        print(f"\nsum of per-op times x block counts: {total_stack / 1e3:.2f} s "
              f"(synced; the fused stack run is faster)")
