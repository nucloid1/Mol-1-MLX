"""What does running the stack's sub-ops in bf16 cost in accuracy?

Inputs are the real captured recycle inputs (cycle 1, where z is no longer near-zero) at
N=48/72/97; the reference is torch CPU fp32. Reported per precision mode:

  * relative error of the stack output s and z against the torch reference;
  * the distogram built from that z, versus the reference distogram: relative logit error,
    the fraction of token pairs whose most-likely distance bin changes, and the largest
    change in the contact probability (P[d < 8 A], what binder-design losses consume).

This is ONE pass through the stack. A real prediction recycles 10-20 times, feeding each
cycle's z into the next, so single-pass error is a lower bound on what a full run sees.
Recycling could damp that noise or amplify it; that is untested here.

Capped at 6 GiB. No full-model run.
"""
import os

import mlx.core as mx
import numpy as np

from mol_mlx import precision
from mol_mlx.blocks import DistogramHead, PairformerStack
from mol_mlx.guard import memory_guard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = (48, 72, 97)
MIN_BIN, MAX_BIN, NO_BINS = 2.3125, 21.6875, 64


def rel(got, want):
    g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
    return float(np.abs(g - w).max()) / float(np.abs(w).max())


def bin_centers():
    width = (MAX_BIN - MIN_BIN) / NO_BINS
    return MIN_BIN + width * 0.5 + width * np.arange(NO_BINS)


def contact_prob(logits):
    l = np.asarray(logits, np.float64)
    p = np.exp(l - l.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return p[..., bin_centers() < 8.0].sum(-1)


with memory_guard(max_gib=6.0):
    w = mx.load(os.path.join(ROOT, "weights", "pairformer_stack.safetensors"))
    stack = PairformerStack(w)
    head = DistogramHead(mx.load(os.path.join(ROOT, "weights", "distogram_head.safetensors")))
    modes = [("fp32", None), ("bf16", mx.bfloat16), ("fp16", mx.float16)]

    print("single pass, cycle-1 inputs, vs torch CPU fp32 reference\n")
    print(f"{'N':>3s} {'mode':5s} {'s rel':>9s} {'z rel':>9s} {'dgram rel':>10s} "
          f"{'bin flips':>10s} {'max |dP(contact)|':>18s}  finite")
    for n in SIZES:
        r = np.load(os.path.join(ROOT, "refs", f"n{n}.npz"))
        s_in, z_in = mx.array(r["cycle1.stack.in0"]), mx.array(r["cycle1.stack.in1"])
        want_dgram = r["final.distogram"]
        for name, dt in modes:
            with precision.compute_dtype(dt):
                s_out, z_out = stack(s_in, z_in)
                logits = head(z_out)
                mx.eval(logits)
            finite = bool(mx.all(mx.isfinite(z_out)).item() and mx.all(mx.isfinite(logits)).item())
            es, ez = rel(s_out, r["cycle1.stack.out0"]), rel(z_out, r["cycle1.stack.out1"])
            if finite:
                ed = rel(logits, want_dgram)
                flips = float(np.mean(np.argmax(np.asarray(logits), -1) != np.argmax(want_dgram, -1)))
                dp = float(np.abs(contact_prob(logits) - contact_prob(want_dgram)).max())
                print(f"{n:3d} {name:5s} {es:9.2e} {ez:9.2e} {ed:10.2e} {flips * 100:9.2f}% {dp:18.4f}  yes")
            else:
                print(f"{n:3d} {name:5s} {'-':>9s} {'-':>9s} {'-':>10s} {'-':>10s} {'-':>18s}  NO (overflow/NaN)")
        print()
