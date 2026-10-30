"""Per-phase time of a REAL Mol-1 prediction, diffusion_cache_mode off vs full. Torch/MPS, nothing ported.

Mol-1's `_main_inference_loop` returns time_tracker (pairformer / diffusion / confidence). With the cache off
(set by Mol1Oracle's LARGE_INPUT_FLAGS above 400 tokens) pair conditioning and every block's pair bias are
recomputed on each denoising step; this measures what that costs per step. 1 cycle (trunk cost is irrelevant),
1 sample, N_step steps; same model, same features, same seed for both modes.

    PYTHONPATH=.:scripts python bench/diffusion_cache_probe.py <N> <n_steps> <cap_gib> [worker] [no-breakdown] [only=full|off]

"worker": run the trunk in the MLX worker (keeps the torch process at diffusion/confidence memory only).
The memory flags of Mol1Oracle's LARGE_INPUT_FLAGS are applied to the config either way; only
diffusion_cache_mode is varied. "no-breakdown" skips the synced per-component pass.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import torch  # noqa: E402

from mol_mlx.guard import memory_guard  # noqa: E402
from trunk_bench import build  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
CAP = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
WORKER = "worker" in sys.argv[4:]
BREAKDOWN = "no-breakdown" not in sys.argv[4:]
ONLY = next((a.split("=", 1)[1] for a in sys.argv[4:] if a.startswith("only=")), None)   # "full" | "off"

with memory_guard(max_gib=CAP, extra=[lambda: torch.mps.driver_allocated_memory() / 2**30]) as peak:
    model, f, kw = build(N, 1)
    cfg = model.configs
    print(f"N={N} tokens, N_atom={int(f['atom_to_token_idx'].shape[-1])}, steps={STEPS}, samples=1; "
          f"enable_efficient_fusion={model.enable_efficient_fusion}", flush=True)
    cfg.sample_diffusion["N_sample"] = 1
    cfg.sample_diffusion["N_step"] = STEPS
    # Mol1Oracle LARGE_INPUT_FLAGS, except diffusion_cache_mode (varied below)
    applied = {}
    for name, value in {"dynamic_chunk_size": False, "chunk_size": 256,
                        "msa_pair_transition_chunk_size": 256, "trunk_pairmixer_transition_chunk_size": 32,
                        "trunk_pairformer_transition_chunk_size": 32, "confidence_pair_transition_chunk_size": 32,
                        "diffusion_conditioning_row_chunk_size": 384}.items():
        if hasattr(cfg.infer_setting, name):
            setattr(cfg.infer_setting, name, value)
            applied[name] = value
    print("applied infer_setting flags:", applied, flush=True)
    if WORKER:
        from mol_mlx.hybrid import install
        remote = install(model, os.path.join(ROOT, "weights"), isolate=True, worker_cap_gib=min(CAP, 24.0))

    def run(mode):
        cfg.diffusion_cache_mode = mode
        torch.manual_seed(0)
        torch.mps.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            pred, _, tt = model._main_inference_loop(
                input_feature_dict=dict(f), label_dict={}, N_cycle=1, mode="inference", inplace_safe=True,
                chunk_size=256)
        torch.mps.synchronize()
        return time.perf_counter() - t, tt, pred

    # ---- component breakdown: wrap the pieces of one denoiser call with hard syncs (adds overhead; separate pass)
    dm = model.diffusion_module
    acc = {}
    BD = BREAKDOWN

    def timed(obj, name, label):
        fn = getattr(obj, name)

        def wrapper(*a, **k):
            torch.mps.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            torch.mps.synchronize()
            acc[label] = acc.get(label, 0.0) + time.perf_counter() - t
            return out
        setattr(obj, name, wrapper)
        return lambda: setattr(obj, name, fn)

    undo = [timed(dm.diffusion_conditioning, "forward", "conditioning (s every step; z too when cache off)"),
            timed(dm.diffusion_conditioning, "prepare_cache", "  of which pair conditioning (prepare_cache)"),
            timed(dm.atom_attention_encoder, "forward", "atom attention encoder"),
            timed(dm.diffusion_transformer, "forward", "token transformer (24 blocks)"),
            timed(dm.atom_attention_decoder, "forward", "atom attention decoder"),
            timed(dm, "forward", "denoiser call, total")]
    run("full")                                           # warm-up (kernel compilation), result discarded
    for mode in (("full", "off") if BD else ()):
        acc.clear()
        run(mode)
        print(f"\ncomponents, cache {mode}, per denoising step (ms, mean over {STEPS} steps):")
        for k, v in acc.items():
            print(f"  {k:55s} {1e3 * v / STEPS:8.1f}")
    for u in undo:
        u()
    rows = {}
    for mode in ((ONLY, ONLY) if ONLY else ("full", "off", "full", "off")):   # interleaved so drift hits both
        wall, tt, pred = run(mode)
        rows.setdefault(mode, []).append((wall, tt))
        torch.mps.empty_cache()
    print(f"\n{'mode':>5s} {'run':>4s} {'wall':>7s} {'pairformer':>11s} {'diffusion':>10s} {'ms/step':>8s} {'confidence':>11s}")
    for mode, runs in rows.items():
        for i, (wall, tt) in enumerate(runs):
            print(f"{mode:>5s} {i:>4d} {wall:7.1f} {tt['pairformer']:11.1f} {tt['diffusion']:10.1f} "
                  f"{1e3 * tt['diffusion'] / STEPS:8.0f} {tt['confidence']:11.1f}")
    best = {m: min(r[1]["diffusion"] for r in rs) for m, rs in rows.items()}
    if len(best) == 2:
        print(f"\ndiffusion, best of 2: cache full {best['full']:.1f} s, off {best['off']:.1f} s -> "
              f"off costs {best['off'] / best['full']:.2f}x; sampled torch MPS peak {peak['extra']:.1f} GiB")
    else:
        (m, v), = best.items()
        print(f"\nONLY {m}: diffusion best of 2 {v:.1f} s ({1e3 * v / STEPS:.0f} ms/step); "
              f"sampled torch MPS peak {peak['extra']:.1f} GiB over this whole process")
